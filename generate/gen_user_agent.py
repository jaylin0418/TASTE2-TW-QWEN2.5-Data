#!/usr/bin/env python3
"""
Type 1: User/Agent dialogue generation.
Generates scenario → system_prompt → dialogue for each topic.

Usage:
    python generate/gen_user_agent.py --config conf/user_agent.yaml \
        --topic 藝術 --num-scenarios 200 --output output/user_agent/
    # Or run all topics in parallel:
    python generate/gen_user_agent.py --config conf/user_agent.yaml \
        --all-topics --num-scenarios 200 --output output/user_agent/ --workers 32
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from generate.base_generator import VLLMClient, extract_json, parse_dialogue, contains_english

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SCENARIOS = 5  # 每次請 LLM 生成幾個 scenario


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Merge base.yaml if defaults declared
    base_path = Path(config_path).parent / "base.yaml"
    if base_path.exists():
        with open(base_path) as f:
            base = yaml.safe_load(f)
        # Child config overrides base
        merged = {**base, **cfg}
        for k in ("gen", "vllm", "dialogue", "tts", "output"):
            if k in base and k in cfg:
                merged[k] = {**base[k], **cfg[k]}
        return merged
    return cfg


def gen_scenarios(client: VLLMClient, cfg: dict, topic: str, n: int) -> list[dict]:
    """Generate n scenarios for a topic in batches of BATCH_SCENARIOS."""
    prompt_tmpl = cfg["prompts"]["scenario"]
    taiwan_hints = cfg.get("taiwan_hints", "")
    scenarios = []
    batch = BATCH_SCENARIOS
    while len(scenarios) < n:
        remaining = n - len(scenarios)
        this_batch = min(batch, remaining)
        prompt = prompt_tmpl.format(
            topic=topic,
            n=this_batch,
            taiwan_hints=taiwan_hints,
        )
        for attempt in range(cfg["vllm"]["max_retries"]):
            try:
                raw = client.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=cfg["gen"]["temperature"],
                    top_p=cfg["gen"]["top_p"],
                    max_tokens=1024,
                )
                data = extract_json(raw)
                batch_scenarios = data.get("scenarios", [])
                if not batch_scenarios:
                    raise ValueError("Empty scenarios")
                for s in batch_scenarios:
                    sid = s.get("id", f"s{len(scenarios)+1:03d}")
                    desc = s.get("description", "")
                    if contains_english(desc):
                        logger.warning(f"Scenario contains English, skipping: {desc[:60]}")
                        continue
                    scenarios.append({"id": sid, "description": desc})
                break
            except Exception as e:
                logger.warning(f"Scenario gen failed (attempt {attempt+1}): {e}")
                time.sleep(cfg["vllm"]["retry_delay"])
        else:
            logger.error(f"Failed to generate scenarios for topic={topic} after retries")
    return scenarios[:n]


def gen_system_prompt(client: VLLMClient, cfg: dict, scenario: str) -> str:
    prompt_tmpl = cfg["prompts"]["system_prompt"]
    prompt = prompt_tmpl.format(scenario=scenario)
    for attempt in range(cfg["vllm"]["max_retries"]):
        try:
            result = client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=0.9,
                max_tokens=256,
            )
            if not contains_english(result):
                return result.strip()
        except Exception as e:
            logger.warning(f"System prompt gen failed (attempt {attempt+1}): {e}")
            time.sleep(cfg["vllm"]["retry_delay"])
    return f"這是一段關於{cfg.get('type','對話')}的情境對話。"


def gen_dialogue(client: VLLMClient, cfg: dict, scenario: str,
                 system_prompt: str, num_turns: int) -> list[dict]:
    prompt_tmpl = cfg["prompts"]["dialogue"]
    taiwan_hints = cfg.get("taiwan_hints", "")
    prompt = prompt_tmpl.format(
        scenario=scenario,
        system_prompt=system_prompt,
        num_turns=num_turns,
        taiwan_hints=taiwan_hints,
    )
    for attempt in range(cfg["vllm"]["max_retries"]):
        try:
            raw = client.chat(
                [{"role": "user", "content": prompt}],
                temperature=cfg["gen"]["temperature"],
                top_p=cfg["gen"]["top_p"],
                max_tokens=cfg["gen"]["max_tokens"],
            )
            turns = parse_dialogue(raw)
            if len(turns) < 4:
                raise ValueError(f"Too few turns: {len(turns)}")
            # Filter out any turn containing English
            clean = []
            for t in turns:
                if contains_english(t["text"]):
                    # Replace with empty marker; downstream will skip
                    logger.warning(f"English in turn, replacing: {t['text'][:50]}")
                    continue
                clean.append(t)
            if len(clean) < 4:
                raise ValueError("Too few clean turns after filtering")
            return clean
        except Exception as e:
            logger.warning(f"Dialogue gen failed (attempt {attempt+1}): {e}")
            time.sleep(cfg["vllm"]["retry_delay"])
    return []


def process_topic(client: VLLMClient, cfg: dict, topic: str,
                  num_scenarios: int, out_dir: Path, tts_backend: str):
    topic_dir = out_dir / topic
    topic_dir.mkdir(parents=True, exist_ok=True)
    done_file = topic_dir / "done.flag"
    if done_file.exists():
        logger.info(f"[{topic}] Already done, skipping.")
        return

    out_jsonl = topic_dir / "dialogues.jsonl"
    # Resume: count already written
    existing_ids = set()
    if out_jsonl.exists():
        with open(out_jsonl) as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    logger.info(f"[{topic}] Resuming from {len(existing_ids)} existing dialogues")

    scenarios = gen_scenarios(client, cfg, topic, num_scenarios)
    cfg_type = cfg.get("type", "user_agent")

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for i, sc in enumerate(scenarios):
            sc_id = sc["id"]
            dialogue_id = f"{cfg_type}_{topic}_{sc_id}"
            if dialogue_id in existing_ids:
                continue

            # Determine TTS backend: first half IndexTTS-2, second half BreezyVoice
            backend = "indextts" if i < num_scenarios // 2 else "breezyvoice"

            num_turns = random.choice(range(8, 17, 2))  # 8,10,12,14,16
            sys_prompt = gen_system_prompt(client, cfg, sc["description"])
            turns = gen_dialogue(client, cfg, sc["description"], sys_prompt, num_turns)

            if not turns:
                logger.warning(f"[{topic}] Empty dialogue for {sc_id}, skipping")
                continue

            record = {
                "id": dialogue_id,
                "type": cfg_type,
                "topic": topic,
                "tts_backend": backend,
                "scenario": sc["description"],
                "system_prompt": sys_prompt,
                "turns": [{"role": t["role"], "text": t["text"], "speed": "normal"}
                          for t in turns],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    done_file.touch()
    logger.info(f"[{topic}] Done. Dialogues saved to {out_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Generate User/Agent dialogues")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--topic", help="Single topic to process")
    parser.add_argument("--all-topics", action="store_true")
    parser.add_argument("--num-scenarios", type=int, default=None,
                        help="Override scenarios_per_topic from config")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--vllm-url", default=None, help="Override vLLM base URL")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.vllm_url:
        cfg["vllm"]["base_url"] = args.vllm_url
    num_scenarios = args.num_scenarios or cfg.get("scenarios_per_topic", 200)
    out_dir = Path(args.output)

    client = VLLMClient(
        base_url=cfg["vllm"]["base_url"],
        model=cfg["vllm"]["model"],
        api_key=cfg["vllm"].get("api_key", "token"),
        timeout=cfg["vllm"].get("timeout", 120),
        max_retries=cfg["vllm"].get("max_retries", 5),
        retry_delay=cfg["vllm"].get("retry_delay", 2.0),
    )

    topics = [args.topic] if args.topic else (cfg["topics"] if args.all_topics else [])
    if not topics:
        parser.error("Provide --topic or --all-topics")

    for topic in topics:
        process_topic(client, cfg, topic, num_scenarios, out_dir, cfg.get("type", "user_agent"))


if __name__ == "__main__":
    main()
