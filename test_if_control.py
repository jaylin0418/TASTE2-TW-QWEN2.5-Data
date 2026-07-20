#!/usr/bin/env python3
"""
Quick quality test for if_control generation.
Run from login node after vLLM server is up.

Usage:
    python3 test_if_control.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate.gen_if_control import gen_if_control_dialogue, SPEED_LABELS, SPEED_WEIGHTS
from generate.gen_user_agent import load_config
from generate.base_generator import VLLMClient

REPO = Path(__file__).parent
CONFIG_PATH = REPO / "conf/if_control.yaml"
PREVIEW_PATH = REPO / "preview_output/type4_if_control.txt"
N_SAMPLES = 10

def wait_for_server(url: str, timeout: int = 120):
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}/models", timeout=3)
            return True
        except Exception:
            time.sleep(5)
    return False

def get_vllm_url():
    node_files = sorted(REPO.glob("logs/vllm_node_*.json"))
    if not node_files:
        return "http://localhost:8000/v1"
    data = json.loads(node_files[-1].read_text())
    return f"http://{data['host']}:{data['ports'][0]}/v1"

def main():
    url = get_vllm_url()
    print(f"vLLM URL: {url}")
    print("Waiting for server...")
    if not wait_for_server(url):
        print("ERROR: server not ready")
        sys.exit(1)
    print("Server ready!\n")

    cfg = load_config(str(CONFIG_PATH))
    cfg["vllm"]["base_url"] = url
    client = VLLMClient(
        base_url=cfg["vllm"]["base_url"],
        model=cfg["vllm"]["model"],
        api_key=cfg["vllm"].get("api_key", "token"),
        timeout=cfg["vllm"].get("timeout", 120),
        max_retries=cfg["vllm"].get("max_retries", 5),
        retry_delay=cfg["vllm"].get("retry_delay", 2.0),
    )

    raw_cats = cfg["if_categories"]
    categories = list(raw_cats.keys()) if isinstance(raw_cats, dict) else list(raw_cats)
    cat_weights = list(raw_cats.values()) if isinstance(raw_cats, dict) else None
    min_tasks = cfg.get("min_tasks", 3)
    max_tasks = cfg.get("max_tasks", 6)

    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    fout = open(PREVIEW_PATH, "w", encoding="utf-8")

    def out(s=""):
        print(s)
        fout.write(s + "\n")

    for i in range(N_SAMPLES):
        import random
        num_tasks = random.randint(min_tasks, max_tasks)
        tasks = random.choices(categories, weights=cat_weights, k=num_tasks)
        speeds_raw = random.choices(SPEED_LABELS, weights=SPEED_WEIGHTS, k=len(tasks))
        speeds = [
            ("normal_silent" if random.random() < 0.5 else "normal") if s == "normal" else s
            for s in speeds_raw
        ]
        out(f"{'='*60}")
        out(f"Sample {i+1}/{N_SAMPLES}")
        out(f"Tasks : {tasks}")
        out(f"Speeds: {speeds}")
        out(f"{'='*60}")
        turns = gen_if_control_dialogue(client, cfg, tasks, speeds)
        if not turns:
            out("[FAILED to generate]")
        else:
            for t in turns:
                role = t["role"]
                speed = f" [{t.get('speed','')}]" if t.get("speed") else ""
                out(f"{role}：{t['text']}{speed}")
        out()

    fout.close()
    print(f"\nSaved to {PREVIEW_PATH}")

if __name__ == "__main__":
    main()
