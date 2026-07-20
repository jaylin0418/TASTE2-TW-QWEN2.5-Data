#!/usr/bin/env python3
"""Quick quality test for speed_ua generation."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate.gen_speed_ua import gen_speed_ua_dialogue, build_speed_plan
from generate.gen_user_agent import load_config, gen_scenarios, gen_system_prompt
from generate.base_generator import VLLMClient

REPO = Path(__file__).parent
CONFIG_PATH = REPO / "conf/speed_ua.yaml"
PREVIEW_PATH = REPO / "preview_output/type5_speed_ua.txt"
N_SAMPLES = 5
TOPIC = "健康"

def main():
    cfg = load_config(str(CONFIG_PATH))
    cfg["vllm"]["base_url"] = "http://localhost:8000/v1"
    client = VLLMClient(
        base_url=cfg["vllm"]["base_url"],
        model=cfg["vllm"]["model"],
        api_key=cfg["vllm"].get("api_key", "token"),
        timeout=cfg["vllm"].get("timeout", 120),
        max_retries=cfg["vllm"].get("max_retries", 5),
        retry_delay=cfg["vllm"].get("retry_delay", 2.0),
    )

    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    fout = open(PREVIEW_PATH, "w", encoding="utf-8")

    def out(s=""):
        print(s)
        fout.write(s + "\n")

    scenarios = gen_scenarios(client, cfg, TOPIC, N_SAMPLES)
    import random
    min_t = cfg["dialogue"]["min_turns"]
    max_t = cfg["dialogue"]["max_turns"]

    for i, sc in enumerate(scenarios[:N_SAMPLES]):
        sys_prompt = gen_system_prompt(client, cfg, sc["description"])
        num_turns = random.choice(range(min_t, max_t + 1, 2))
        turns = gen_speed_ua_dialogue(client, cfg, sc["description"], sys_prompt, num_turns)

        out(f"{'='*60}")
        out(f"Sample {i+1}/{N_SAMPLES}  Topic: {TOPIC}  Turns: {num_turns}")
        out(f"Scenario: {sc['description'][:60]}...")
        out(f"{'='*60}")
        if not turns:
            out("[FAILED]")
        else:
            for t in turns:
                role = t["role"]
                # User has no speed field; Agent always has speed field
                speed_tag = f" [{t['speed']}]" if "speed" in t else ""
                out(f"{role}：{t['text']}{speed_tag}")
        out()

    fout.close()
    print(f"\nSaved to {PREVIEW_PATH}")

if __name__ == "__main__":
    main()
