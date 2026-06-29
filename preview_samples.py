"""
Generate 5 sample dialogues for each of the 6 data types using local Qwen model.
Uses transformers (breezyvoice_py310 env) so vLLM is not needed.

Usage:
    /home/jaylin0418/miniconda3/envs/breezyvoice_py310/bin/python3 preview_samples.py
"""
import sys, json, random, textwrap
from pathlib import Path

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "generate"))

MODEL_PATH = "/work/jaylin0418/model_merging/models/Qwen2.5-7B-Instruct"
OUT_DIR = REPO / "preview_output"
N_SAMPLES = 5
DIVIDER = "=" * 70

OUT_DIR.mkdir(exist_ok=True)


# ── Local Qwen client that mimics VLLMClient ──────────────────────────────────

class LocalQwenClient:
    def __init__(self, model_path: str):
        print(f"Loading model from {model_path} ...", flush=True)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("Model loaded.", flush=True)

    def chat(self, messages: list[dict], temperature: float = 0.85,
             top_p: float = 0.9, max_tokens: int = 2048) -> str:
        import torch
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tok.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(generated, skip_special_tokens=True).strip()

    def chat_system(self, system: str, user: str, **kw) -> str:
        return self.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            **kw,
        )

    def batch_chat(self, prompts, concurrency=1, **kw):
        return [self.chat(p, **kw) for p in prompts]


# ── Config loader ─────────────────────────────────────────────────────────────

import yaml

def load_cfg(name: str) -> dict:
    base = yaml.safe_load((REPO / "conf/base.yaml").read_text())
    extra = yaml.safe_load((REPO / f"conf/{name}.yaml").read_text())
    # Merge: extra overrides base at top level
    base.update(extra)
    return base


# ── Pretty printer ────────────────────────────────────────────────────────────

_current_file: list = []   # holds open file handle per type run


def print_sample(idx: int, type_name: str, system_prompt: str,
                 turns: list[dict], extra: str = ""):
    lines = []
    lines.append(f"\n{DIVIDER}")
    lines.append(f"[{type_name}] Sample #{idx+1}" + (f"  ({extra})" if extra else ""))
    lines.append(DIVIDER)
    if system_prompt:
        lines.append("System Prompt:")
        for line in textwrap.wrap(system_prompt, 66):
            lines.append(f"  {line}")
        lines.append("")
    for t in turns:
        role = t.get("role", "?")
        text = t.get("text", "")
        speed = t.get("speed", "normal")
        speed_tag = f"[{speed}] " if speed and speed != "normal" else ""
        lines.append(f"  {role}：{speed_tag}{text}")
    output = "\n".join(lines)
    print(output)
    if _current_file:
        _current_file[0].write(output + "\n")
        _current_file[0].flush()


def run_preview(type_name: str, fn, *args, **kwargs):
    """Run a preview function, tee-ing output to preview_output/<type_name>.txt"""
    out_path = OUT_DIR / f"{type_name}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        _current_file.clear()
        _current_file.append(f)
        fn(*args, **kwargs)
        _current_file.clear()
    print(f"\n→ 已存至 {out_path}")


# ── Type 1: user_agent ────────────────────────────────────────────────────────

def preview_user_agent(client, n=N_SAMPLES):
    from generate.gen_user_agent import gen_scenarios, gen_system_prompt, gen_dialogue
    cfg = load_cfg("user_agent")
    topics = cfg["topics"]
    print(f"\n{'#'*70}\nTYPE 1: user_agent\n{'#'*70}")
    for i in range(n):
        topic = random.choice(topics)
        scenarios = gen_scenarios(client, cfg, topic, 1)
        if not scenarios:
            print(f"  [Sample {i+1}] gen_scenarios failed, skip"); continue
        sc = scenarios[0]
        sys_prompt = gen_system_prompt(client, cfg, sc["description"])
        num_turns = random.choice(range(8, 17, 2))
        turns = gen_dialogue(client, cfg, sc["description"], sys_prompt, num_turns)
        print_sample(i, "user_agent", sys_prompt, turns, extra=f"topic={topic}")


# ── Type 2: daily_conv ────────────────────────────────────────────────────────

def preview_daily_conv(client, n=N_SAMPLES):
    # gen_daily_conv re-exports from gen_user_agent; import directly
    from generate.gen_user_agent import gen_scenarios, gen_system_prompt, gen_dialogue
    cfg = load_cfg("daily_conv")
    topics = cfg["topics"]
    print(f"\n{'#'*70}\nTYPE 2: daily_conv\n{'#'*70}")
    for i in range(n):
        topic = random.choice(topics)
        scenarios = gen_scenarios(client, cfg, topic, 1)
        if not scenarios:
            print(f"  [Sample {i+1}] failed, skip"); continue
        sc = scenarios[0]
        sys_prompt = gen_system_prompt(client, cfg, sc["description"])
        num_turns = random.choice(range(8, 17, 2))
        turns = gen_dialogue(client, cfg, sc["description"], sys_prompt, num_turns)
        print_sample(i, "daily_conv", sys_prompt, turns, extra=f"topic={topic}")


# ── Type 3: if_data ───────────────────────────────────────────────────────────

def preview_if_data(client, n=N_SAMPLES):
    from generate.gen_if_data import gen_if_dialogue
    cfg = load_cfg("if_data")
    categories = [c["name"] for c in cfg["if_categories"]]
    min_tasks = cfg.get("min_tasks", 3)
    max_tasks = cfg.get("max_tasks", 6)
    print(f"\n{'#'*70}\nTYPE 3: if_data (Instruction Following)\n{'#'*70}")
    for i in range(n):
        num_tasks = random.randint(min_tasks, max_tasks)
        task_list = random.sample(categories, min(num_tasks, len(categories)))
        turns = gen_if_dialogue(client, cfg, task_list)
        if not turns:
            print(f"  [Sample {i+1}] failed, skip"); continue
        # IF data has no separate system_prompt (the prompt is embedded)
        turns_with_speed = [{"role": t["role"], "text": t["text"], "speed": "normal"}
                            for t in turns]
        print_sample(i, "if_data", "", turns_with_speed,
                     extra=f"tasks={task_list}")


# ── Type 4: speed_ua ──────────────────────────────────────────────────────────

def preview_speed_ua(client, n=N_SAMPLES):
    from generate.gen_user_agent import gen_scenarios, gen_system_prompt, gen_dialogue
    from generate.gen_speed_control import apply_speed_fsm
    cfg_speed = load_cfg("speed_control")
    cfg_ua = load_cfg("user_agent")
    # speed_control adds fsm section; UA provides the prompt templates
    cfg = {**cfg_ua, **cfg_speed}
    topics = cfg["topics"]
    fsm_cfg = cfg.get("fsm", {})
    print(f"\n{'#'*70}\nTYPE 4: speed_ua\n{'#'*70}")
    for i in range(n):
        topic = random.choice(topics)
        scenarios = gen_scenarios(client, cfg_ua, topic, 1)
        if not scenarios:
            print(f"  [Sample {i+1}] failed, skip"); continue
        sc = scenarios[0]
        sys_prompt = gen_system_prompt(client, cfg_ua, sc["description"])
        num_turns = random.choice(range(8, 17, 2))
        turns = gen_dialogue(client, cfg_ua, sc["description"], sys_prompt, num_turns)
        turns = apply_speed_fsm(turns, fsm_cfg)
        print_sample(i, "speed_ua", sys_prompt, turns, extra=f"topic={topic}")


# ── Type 5: speed_daily ───────────────────────────────────────────────────────

def preview_speed_daily(client, n=N_SAMPLES):
    from generate.gen_user_agent import gen_scenarios, gen_system_prompt, gen_dialogue
    from generate.gen_speed_control import apply_speed_fsm
    cfg_speed = load_cfg("speed_control")
    cfg_daily = load_cfg("daily_conv")
    cfg = {**cfg_daily, **cfg_speed}
    topics = cfg["topics"]
    fsm_cfg = cfg.get("fsm", {})
    print(f"\n{'#'*70}\nTYPE 5: speed_daily\n{'#'*70}")
    for i in range(n):
        topic = random.choice(topics)
        scenarios = gen_scenarios(client, cfg_daily, topic, 1)
        if not scenarios:
            print(f"  [Sample {i+1}] failed, skip"); continue
        sc = scenarios[0]
        sys_prompt = gen_system_prompt(client, cfg_daily, sc["description"])
        num_turns = random.choice(range(8, 17, 2))
        turns = gen_dialogue(client, cfg_daily, sc["description"], sys_prompt, num_turns)
        turns = apply_speed_fsm(turns, fsm_cfg)
        print_sample(i, "speed_daily", sys_prompt, turns, extra=f"topic={topic}")


# ── Type 6: if_control ────────────────────────────────────────────────────────

def preview_if_control(client, n=N_SAMPLES):
    from generate.gen_if_control import (
        gen_if_control_dialogue, assign_speeds_to_turns,
        SPEED_LABELS, SPEED_WEIGHTS,
    )
    cfg = load_cfg("if_control")
    raw = cfg["if_categories"]
    # if_control.yaml has plain strings; if_data.yaml has dicts with "name"
    categories = [c if isinstance(c, str) else c["name"] for c in raw]
    min_tasks = cfg.get("min_tasks", 3)
    max_tasks = cfg.get("max_tasks", 6)
    print(f"\n{'#'*70}\nTYPE 6: if_control (IF + Speed Control)\n{'#'*70}")
    for i in range(n):
        num_tasks = random.randint(min_tasks, max_tasks)
        task_list = random.sample(categories, min(num_tasks, len(categories)))
        speed_list = random.choices(SPEED_LABELS, weights=SPEED_WEIGHTS, k=num_tasks)
        turns = gen_if_control_dialogue(client, cfg, task_list, speed_list)
        if not turns:
            print(f"  [Sample {i+1}] failed, skip"); continue
        turns = assign_speeds_to_turns(turns, speed_list)
        print_sample(i, "if_control", "", turns,
                     extra=f"tasks={list(zip(task_list, speed_list))}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    client = LocalQwenClient(MODEL_PATH)

    run_preview("type1_user_agent",   preview_user_agent,   client)
    run_preview("type2_daily_conv",   preview_daily_conv,   client)
    run_preview("type3_if_data",      preview_if_data,      client)
    run_preview("type4_speed_ua",     preview_speed_ua,     client)
    run_preview("type5_speed_daily",  preview_speed_daily,  client)
    run_preview("type6_if_control",   preview_if_control,   client)

    print(f"\n{DIVIDER}\nAll previews done.")
    print(f"輸出目錄：{OUT_DIR}")
