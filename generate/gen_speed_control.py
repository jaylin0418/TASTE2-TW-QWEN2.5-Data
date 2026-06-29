#!/usr/bin/env python3
"""
Type 4 & 5: Speed Control dialogue generation.
Generates dialogue (user_agent or daily_conv style), then applies Speed FSM
to each turn's metadata. The dialogue TEXT itself does NOT mention speed —
speed is only applied at TTS time.

FSM spec:
  - First time leaving Normal:  P_leave_first  (default 0.75)
  - Fast/Slow → Normal:         P_return       (default 0.40)
  - After returning to Normal:  P_leave_again  (default 0.40)
  - Fast vs Slow: 50/50 when leaving Normal

Usage:
    # User/Agent + speed
    python generate/gen_speed_control.py --config conf/speed_control.yaml \
        --type speed_ua --topic 藝術 --num-scenarios 40 --output output/speed_ua/

    # Daily Conv + speed
    python generate/gen_speed_control.py --config conf/speed_control.yaml \
        --type speed_daily --topic 美食 --num-scenarios 40 --output output/speed_daily/
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
from generate.gen_user_agent import load_config, gen_scenarios, gen_system_prompt, gen_dialogue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Speed FSM ─────────────────────────────────────────────────────────────────

class SpeedFSM:
    STATES = ("normal", "fast", "slow")

    def __init__(self, p_leave_first: float = 0.75,
                 p_leave_again: float = 0.40,
                 p_return: float = 0.40):
        self.p_leave_first = p_leave_first
        self.p_leave_again = p_leave_again
        self.p_return = p_return
        self.state = "normal"
        self.has_left_normal = False  # True after first transition away from Normal

    def next(self) -> str:
        """Advance FSM by one turn and return new state."""
        if self.state == "normal":
            p_leave = self.p_leave_first if not self.has_left_normal else self.p_leave_again
            if random.random() < p_leave:
                self.state = random.choice(["fast", "slow"])
                self.has_left_normal = True
        else:  # fast or slow
            if random.random() < self.p_return:
                self.state = "normal"
        return self.state

    def assign_turns(self, num_turns: int) -> list[str]:
        """Return a speed label for each turn."""
        return [self.next() for _ in range(num_turns)]


def apply_speed_fsm(turns: list[dict], fsm_cfg: dict) -> list[dict]:
    fsm = SpeedFSM(
        p_leave_first=fsm_cfg.get("p_leave_first", 0.75),
        p_leave_again=fsm_cfg.get("p_leave_again", 0.40),
        p_return=fsm_cfg.get("p_return", 0.40),
    )
    speeds = fsm.assign_turns(len(turns))
    for turn, speed in zip(turns, speeds):
        turn["speed"] = speed
    return turns


# ── Main ──────────────────────────────────────────────────────────────────────

def process_topic(client: VLLMClient, cfg: dict, base_cfg: dict,
                  topic: str, num_scenarios: int, data_type: str,
                  out_dir: Path):
    topic_dir = out_dir / topic
    topic_dir.mkdir(parents=True, exist_ok=True)
    done_file = topic_dir / "done.flag"
    if done_file.exists():
        logger.info(f"[{topic}] Already done, skipping.")
        return

    out_jsonl = topic_dir / "dialogues.jsonl"
    existing_ids: set[str] = set()
    if out_jsonl.exists():
        with open(out_jsonl) as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    scenarios = gen_scenarios(client, base_cfg, topic, num_scenarios)
    fsm_cfg = cfg.get("fsm", {})

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for i, sc in enumerate(scenarios):
            sc_id = sc["id"]
            dialogue_id = f"{data_type}_{topic}_{sc_id}"
            if dialogue_id in existing_ids:
                continue

            backend = "indextts" if i < num_scenarios // 2 else "breezyvoice"
            num_turns = random.choice(range(8, 17, 2))
            sys_prompt = gen_system_prompt(client, base_cfg, sc["description"])
            turns = gen_dialogue(client, base_cfg, sc["description"], sys_prompt, num_turns)

            if not turns:
                logger.warning(f"[{topic}] Empty dialogue for {sc_id}, skipping")
                continue

            # Apply speed FSM
            for t in turns:
                t["speed"] = "normal"
            turns = apply_speed_fsm(turns, fsm_cfg)

            record = {
                "id": dialogue_id,
                "type": data_type,
                "topic": topic,
                "tts_backend": backend,
                "scenario": sc["description"],
                "system_prompt": sys_prompt,
                "turns": turns,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    done_file.touch()
    logger.info(f"[{topic}] Done.")


def main():
    parser = argparse.ArgumentParser(description="Generate Speed Control dialogues")
    parser.add_argument("--config", required=True)
    parser.add_argument("--type", required=True, choices=["speed_ua", "speed_daily"])
    parser.add_argument("--topic", help="Single topic")
    parser.add_argument("--all-topics", action="store_true")
    parser.add_argument("--num-scenarios", type=int, default=40)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vllm-url", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.vllm_url:
        cfg["vllm"]["base_url"] = args.vllm_url

    # Load the base dialogue config (ua or daily)
    base_config_key = "base_config_ua" if args.type == "speed_ua" else "base_config_daily"
    base_config_path = cfg.get(base_config_key)
    if not base_config_path:
        raise ValueError(f"Missing {base_config_key} in config")
    base_cfg = load_config(base_config_path)
    # Override vllm settings from speed config
    base_cfg["vllm"] = cfg["vllm"]

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

    out_dir = Path(args.output)
    for topic in topics:
        process_topic(client, cfg, base_cfg, topic, args.num_scenarios,
                      args.type, out_dir)


if __name__ == "__main__":
    main()
