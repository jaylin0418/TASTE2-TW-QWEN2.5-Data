#!/usr/bin/env python3
"""
Type 4: IF Control dialogue generation.
甲 gives IF tasks WITH explicit speed instructions; 乙 executes.
Speed state is determined per-task and stored in turn metadata.

Usage:
    python generate/gen_if_control.py --config conf/if_control.yaml \
        --num-dialogues 1600 --output output/if_control/
"""
import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from generate.base_generator import VLLMClient, parse_dialogue, contains_english
from generate.gen_user_agent import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPEED_LABELS = ["fast", "slow", "normal"]
SPEED_WEIGHTS = [1, 1, 1]  # equal probability

# Chinese descriptions passed to the LLM prompt
SPEED_CN = {
    "fast":           "快速",
    "slow":           "慢速",
    "normal":         "正常（需在指令中說出速度要求，例如：照正常速度、普通速度）",
    "normal_silent":  "（不提速度，甲直接給任務指令，不說任何速度要求）",
}


def gen_if_control_dialogue(client: VLLMClient, cfg: dict,
                             task_list: list[str], speed_list: list[str]) -> list[dict]:
    num_tasks = len(task_list)
    num_turns = num_tasks * 2
    task_str = "、".join(task_list)
    speed_str = "、".join(SPEED_CN[s] for s in speed_list)
    prompt_tmpl = cfg["prompts"]["dialogue"]
    taiwan_hints = cfg.get("taiwan_hints", "")
    prompt = prompt_tmpl.format(
        num_turns=num_turns,
        num_tasks=num_tasks,
        task_list=task_str,
        speed_list=speed_str,
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
                raise ValueError(f"Too few turns: {len(turns)}")
            clean = [t for t in turns if not contains_english(t["text"])]
            if len(clean) < 4:
                raise ValueError("Too few clean turns")
            return clean
        except Exception as e:
            logger.warning(f"IF control dialogue gen failed (attempt {attempt+1}): {e}")
            time.sleep(cfg["vllm"]["retry_delay"])
    return []


def assign_speeds_to_turns(turns: list[dict], speed_list: list[str]) -> list[dict]:
    """
    Assign speed labels to turns.
    User turn (even index 0,2,4,...): speed is embedded in text, no metadata tag → speed=""
    Agent turn (odd index 1,3,5,...): executes at that speed → speed=fast/slow/""
    Also rename roles: 甲→User, 乙→Agent.
    """
    role_map = {"甲": "User", "乙": "Agent"}
    for i, turn in enumerate(turns):
        turn["role"] = role_map.get(turn["role"], turn["role"])
        task_idx = i // 2
        speed = speed_list[task_idx] if task_idx < len(speed_list) else "normal"
        if i % 2 == 0:
            turn["speed"] = ""  # User: speed embedded in text
        else:
            # normal and normal_silent both → no speed tag for Agent
            turn["speed"] = speed if speed in ("fast", "slow") else ""
    return turns


def main():
    parser = argparse.ArgumentParser(description="Generate IF Control dialogues")
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

    # if_categories can be a list (uniform) or dict {name: weight}
    raw_cats = cfg["if_categories"]
    if isinstance(raw_cats, dict):
        categories = list(raw_cats.keys())
        cat_weights = list(raw_cats.values())
    else:
        categories = list(raw_cats)
        cat_weights = None  # uniform

    min_tasks = cfg.get("min_tasks", 3)
    max_tasks = cfg.get("max_tasks", 6)
    total = args.num_dialogues
    per_worker = total // args.num_workers
    start = args.worker_id * per_worker
    end = start + per_worker if args.worker_id < args.num_workers - 1 else total

    existing_ids: set[str] = set()
    if out_jsonl.exists():
        with open(out_jsonl) as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    client = VLLMClient(
        base_url=cfg["vllm"]["base_url"],
        model=cfg["vllm"]["model"],
        api_key=cfg["vllm"].get("api_key", "token"),
        timeout=cfg["vllm"].get("timeout", 120),
        max_retries=cfg["vllm"].get("max_retries", 5),
        retry_delay=cfg["vllm"].get("retry_delay", 2.0),
    )

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for idx in range(start, end):
            dialogue_id = f"if_control_{idx:06d}"
            if dialogue_id in existing_ids:
                continue

            num_tasks = random.randint(min_tasks, max_tasks)
            task_list = random.choices(categories, weights=cat_weights, k=num_tasks)
            # Deduplicate consecutive same types
            task_list = [task_list[0]] + [t for i, t in enumerate(task_list[1:]) if t != task_list[i]]
            speed_list_raw = random.choices(SPEED_LABELS, weights=SPEED_WEIGHTS, k=len(task_list))
            speed_list = [
                ("normal_silent" if random.random() < 0.5 else "normal") if s == "normal" else s
                for s in speed_list_raw
            ]

            backend = "indextts" if idx < total // 2 else "breezyvoice"
            turns = gen_if_control_dialogue(client, cfg, task_list, speed_list)
            if not turns:
                logger.warning(f"Empty dialogue for {dialogue_id}, skipping")
                continue

            turns = assign_speeds_to_turns(turns, speed_list)

            record = {
                "id": dialogue_id,
                "type": "type4_if_control",
                "tts_backend": backend,
                "if_tasks": task_list,
                "speed_sequence": speed_list,
                "turns": turns,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if (idx - start) % 50 == 0:
                logger.info(f"[Worker {args.worker_id}] {idx - start}/{end - start} done")


if __name__ == "__main__":
    main()
