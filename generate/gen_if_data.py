#!/usr/bin/env python3
"""
Type 3: Instruction Following data generation.
Generates multi-task IF dialogues (甲 gives 3-6 IF tasks, 乙 executes).

Usage:
    python generate/gen_if_data.py --config conf/if_data.yaml \
        --num-dialogues 1600 --output output/if_data/
"""
import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from generate.base_generator import VLLMClient, parse_dialogue, contains_english
from generate.gen_user_agent import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def gen_if_dialogue(client: VLLMClient, cfg: dict, task_list: list[str]) -> list[dict]:
    num_tasks = len(task_list)
    num_turns = num_tasks * 2
    task_str = "、".join(task_list)
    prompt_tmpl = cfg["prompts"]["dialogue"]
    taiwan_hints = cfg.get("taiwan_hints", "")
    prompt = prompt_tmpl.format(
        num_turns=num_turns,
        num_tasks=num_tasks,
        task_list=task_str,
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
            if len(turns) < num_turns - 2:
                raise ValueError(f"Too few turns: {len(turns)} (expected ~{num_turns})")
            clean = [t for t in turns if not contains_english(t["text"])]
            if len(clean) < 4:
                raise ValueError("Too few clean turns")
            return clean
        except Exception as e:
            logger.warning(f"IF dialogue gen failed (attempt {attempt+1}): {e}")
            time.sleep(cfg["vllm"]["retry_delay"])
    return []


def main():
    parser = argparse.ArgumentParser(description="Generate IF dialogues")
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-dialogues", type=int, default=1600)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vllm-url", default=None)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.vllm_url:
        cfg["vllm"]["base_url"] = args.vllm_url

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"dialogues_w{args.worker_id:02d}.jsonl"

    categories = [c["name"] for c in cfg["if_categories"]]
    min_tasks = cfg.get("min_tasks", 3)
    max_tasks = cfg.get("max_tasks", 6)

    client = VLLMClient(
        base_url=cfg["vllm"]["base_url"],
        model=cfg["vllm"]["model"],
        api_key=cfg["vllm"].get("api_key", "token"),
        timeout=cfg["vllm"].get("timeout", 120),
        max_retries=cfg["vllm"].get("max_retries", 5),
        retry_delay=cfg["vllm"].get("retry_delay", 2.0),
    )

    # Determine this worker's slice
    total = args.num_dialogues
    per_worker = total // args.num_workers
    start = args.worker_id * per_worker
    end = start + per_worker if args.worker_id < args.num_workers - 1 else total

    # Load existing IDs to resume
    existing_ids: set[str] = set()
    if out_jsonl.exists():
        with open(out_jsonl) as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    logger.info(f"[Worker {args.worker_id}] Range [{start}, {end}), existing={len(existing_ids)}")

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for idx in range(start, end):
            dialogue_id = f"if_data_{idx:06d}"
            if dialogue_id in existing_ids:
                continue

            num_tasks = random.randint(min_tasks, max_tasks)
            task_list = random.sample(categories, min(num_tasks, len(categories)))

            # TTS backend: first half indextts, second half breezyvoice
            backend = "indextts" if idx < total // 2 else "breezyvoice"

            turns = gen_if_dialogue(client, cfg, task_list)
            if not turns:
                logger.warning(f"Empty dialogue for {dialogue_id}, skipping")
                continue

            record = {
                "id": dialogue_id,
                "type": "if_data",
                "tts_backend": backend,
                "if_tasks": task_list,
                "turns": [{"role": t["role"], "text": t["text"], "speed": "normal"}
                          for t in turns],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if (idx - start) % 50 == 0:
                logger.info(f"[Worker {args.worker_id}] {idx - start}/{end - start} done")


if __name__ == "__main__":
    main()
