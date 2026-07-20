#!/usr/bin/env python3
"""
Type 5: Speed-UA — User/Agent dialogue with embedded speed instructions.

Speed is assigned per-exchange via FSM and baked into generation:
  甲 naturally embeds speed instructions in her turns when the FSM says to.
  乙 executes at the current (sticky) speed until 甲 explicitly changes it.

FSM transitions (per 甲 turn):
  normal → stay 60% | fast 20% | slow 20%
  fast   → stay 50% | normal 45% | slow 5%
  slow   → stay 50% | normal 45% | fast 5%
  (tuned for ~2 transitions per 5-6 turn dialogue)

Explicit rules:
  - Start normal:           never inject speed hint (optional by design)
  - Start fast/slow:        always inject
  - Transition to fast/slow: always inject
  - Transition to normal:   always inject ("回到正常速度")
  - Same state (sticky):    never inject

Usage:
    python generate/gen_speed_ua.py --config conf/speed_ua.yaml \
        --topic 健康 --num-scenarios 200 --output output/speed_ua/
    python generate/gen_speed_ua.py --config conf/speed_ua.yaml \
        --all-topics --num-scenarios 200 --output output/speed_ua/ --workers 8
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from generate.base_generator import VLLMClient, contains_english, contains_simplified
from generate.gen_user_agent import (
    load_config, gen_scenarios, gen_system_prompt,
    _clean_raw_turn, _is_hard_reject, _ngram_too_similar,
    _FAREWELL, _META,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── FSM ───────────────────────────────────────────────────────────────────────

_TRANS = {
    "normal": [("normal", 0.60), ("fast", 0.20), ("slow", 0.20)],
    "fast":   [("fast",   0.62), ("normal", 0.33), ("slow", 0.05)],
    "slow":   [("slow",   0.62), ("normal", 0.33), ("fast", 0.05)],
}

def _next_state(current: str) -> str:
    states, weights = zip(*_TRANS[current])
    return random.choices(states, weights=weights, k=1)[0]

def build_speed_plan(num_user_turns: int) -> list[dict]:
    """
    Returns a list of {state, explicit} for each 甲 turn.
    explicit=True means 甲 must embed a speed instruction this turn.
    """
    # Starting state
    r = random.random()
    if r < 0.50:
        state = "normal"
    elif r < 0.75:
        state = "fast"
    else:
        state = "slow"

    plan = []
    prev_state = None

    for i in range(num_user_turns):
        if i == 0:
            # First turn: explicit only if non-normal
            explicit = (state != "normal")
            plan.append({"state": state, "explicit": explicit})
            prev_state = state
        else:
            new_state = _next_state(state)
            if new_state != state:
                explicit = True   # any transition → always say it
            else:
                explicit = False  # sticky → silent
            state = new_state
            plan.append({"state": state, "explicit": explicit})
            prev_state = state

    return plan


# ── Speed hint injection ───────────────────────────────────────────────────────

_HINT_FAST = (
    "【速度提示（必須執行）：這輪在話裡自然說出要對方「說快一點」，"
    "口語融入句子（例：欸你說快一點、請說快一點、可以快點說嗎），不可省略速度詞。】"
)
_HINT_SLOW = (
    "【速度提示（必須執行）：這輪在話裡自然說出要對方「說慢一點」，"
    "口語融入句子（例：你說慢一點、可以慢慢說嗎、請慢點說），不可省略速度詞。】"
)
_HINT_NORMAL = (
    "【速度提示（必須執行）：前面有要求過說快或說慢，這輪要明確說回到正常速度，"
    "自然融入句子（例：好你用正常速度說就好、不用那麼快普通說），不可省略速度詞。】"
)

_HINT_FAST_RETRY   = "【重試強制：上輪輸出缺少速度詞。這輪話裡必須出現「快」相關語速請求（說快一點、快點說、請說快），否則不合格。】"
_HINT_SLOW_RETRY   = "【重試強制：上輪輸出缺少速度詞。這輪話裡必須出現「慢」相關語速請求（說慢一點、慢慢說、請說慢），否則不合格。】"
_HINT_NORMAL_RETRY = "【重試強制：上輪輸出缺少速度詞。這輪話裡必須出現「正常」或「普通」速度請求（用正常速度、普通說就好），否則不合格。】"

_RETRY_HINT = {"fast": _HINT_FAST_RETRY, "slow": _HINT_SLOW_RETRY, "normal": _HINT_NORMAL_RETRY}

# Keywords for validating that explicit turns contain speed instructions
_SPEED_KEYWORDS = {
    "fast":   ["快"],
    "slow":   ["慢"],
    "normal": ["正常", "普通", "不用那麼快", "不用那麼慢"],
}

def _has_speed_instruction(text: str, state: str) -> bool:
    return any(kw in text for kw in _SPEED_KEYWORDS.get(state, []))

def speed_hint(plan_entry: dict, prev_state: str | None) -> str:
    if not plan_entry["explicit"]:
        return ""
    state = plan_entry["state"]
    if state == "fast":
        return _HINT_FAST
    if state == "slow":
        return _HINT_SLOW
    if state == "normal":
        return _HINT_NORMAL
    return ""


# ── Dialogue generation ───────────────────────────────────────────────────────

def gen_speed_ua_dialogue(
    client: VLLMClient,
    cfg: dict,
    scenario: str,
    system_prompt_text: str,
    num_turns: int,
) -> list[dict]:
    """
    Turn-by-turn generation with speed FSM baked into 甲's system prompt.
    Returns list of {role, text, speed, explicit} dicts.
    """
    num_user_turns = (num_turns + 1) // 2
    speed_plan = build_speed_plan(num_user_turns)

    a_sys_template = cfg["prompts"]["a_turn"]
    b_sys_template = cfg["prompts"]["b_turn"]

    roles = cfg.get("roles", {})
    role_a = roles.get("a", "User")
    role_b = roles.get("b", "Agent")
    backchannel_rate = cfg.get("backchannel_rate", 0.10)  # lower for speed-ua

    rep_penalty = cfg.get("gen", {}).get("repetition_penalty", 1.0)

    turns: list[dict] = []
    user_turn_idx = 0
    current_speed = "normal"  # tracks the last User-requested speed for Agent to inherit

    for i in range(num_turns):
        is_a = (i % 2 == 0)
        role = role_a if is_a else role_b

        if is_a:
            plan_entry = speed_plan[user_turn_idx]
            hint = speed_hint(plan_entry, None)
            base_sys = a_sys_template.format(
                scenario=scenario, system_prompt=system_prompt_text)
        else:
            plan_entry = None
            hint = ""
            base_sys = b_sys_template.format(
                scenario=scenario, system_prompt=system_prompt_text)

        # Turn position hints
        if i == 0:
            base_sys += "\n\n【注意：這是對話的第一句。請主動開口說出你的問題或話題，不要用「嗯」「好」「讓我想想」等回應性詞語開頭。】"
        elif i == num_turns - 1:
            if is_a:
                base_sys += "\n\n【注意：這是最後一輪。說你最後的反應、追問或感受，不要道別、不要道謝、不要說再見。】"
            else:
                base_sys += "\n\n【注意：這是最後一輪。給一句補充提示或具體建議直接結束，絕對不能說「再見」「祝你」「有需要再來」之類。】"
        elif is_a:
            base_sys += "\n\n【你是使用者：說你的問題、反應或感受。禁止說「你可以…」「建議你…」等助理語氣。】"

        # Backchannel (lower rate to avoid clashing with speed hints)
        if is_a and i > 0 and not hint and random.random() < backchannel_rate:
            base_sys += "\n\n【注意：這次請輸出十個字以內的超短回應，例如：嗯、哦好、我查一下、讓我想想。】"

        # Speed hint injection (User only)
        if is_a and hint:
            base_sys += f"\n\n{hint}"

        # Mid-conversation progress hint
        if i > 0 and i % 3 == 0 and i < num_turns - 1:
            same_role_turns = [t["text"] for j, t in enumerate(turns) if (j % 2 == 0) == is_a]
            if same_role_turns:
                recent = "、".join(f'「{t[:15]}…」' for t in same_role_turns[-2:])
                base_sys += f"\n\n【推進提示：你已說過 {recent}。這一輪必須補充新資訊或推進話題，不可再重複。】"

        need_speed_check = is_a and plan_entry is not None and plan_entry["explicit"]

        # Generate with retry; add stronger hint on speed-validation failure
        text = None
        for attempt in range(5):
            sys_content = base_sys
            if attempt > 0 and need_speed_check:
                sys_content += f"\n\n{_RETRY_HINT[plan_entry['state']]}"

            messages = [{"role": "system", "content": sys_content}]
            for j, t in enumerate(turns):
                chat_role = "assistant" if ((j % 2 == 0) == is_a) else "user"
                messages.append({"role": chat_role, "content": t["text"]})

            try:
                raw = client.chat(
                    messages,
                    temperature=cfg["gen"]["temperature"],
                    top_p=cfg["gen"]["top_p"],
                    max_tokens=256,
                    repetition_penalty=rep_penalty,
                )
                text = _clean_raw_turn(raw, role_a, role_b)
                if not text or _is_hard_reject(text):
                    text = None
                    continue
                if contains_simplified(text) or contains_english(text):
                    text = None
                    continue
                if need_speed_check and not _has_speed_instruction(text, plan_entry["state"]):
                    text = None
                    continue
                break
            except Exception as e:
                logger.warning(f"Turn {i} attempt {attempt+1} failed: {e}")
                time.sleep(1)

        if text is None:
            logger.warning(f"Turn {i} failed after 5 attempts, skipping dialogue")
            return []

        # User turns: no speed field — User controls Agent speed, User's own voice is always normal
        # Agent turns: speed field set to current sticky speed
        if is_a:
            current_speed = plan_entry["state"]
            user_turn_idx += 1
            entry = {"role": role, "text": text}
        else:
            entry = {"role": role, "text": text, "speed": current_speed}

        turns.append(entry)

    return turns


# ── Main ─────────────────────────────────────────────────────────────────────

def process_topic(client: VLLMClient, cfg: dict, topic: str,
                  num_scenarios: int, out_dir: Path) -> int:
    out_file = out_dir / f"dialogues_{topic}.jsonl"
    existing: set[str] = set()
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                try:
                    existing.add(json.loads(line)["id"])
                except Exception:
                    pass

    scenarios = gen_scenarios(client, cfg, topic, num_scenarios)
    if not scenarios:
        logger.warning(f"No scenarios for topic: {topic}")
        return 0

    min_turns = cfg["dialogue"]["min_turns"]
    max_turns = cfg["dialogue"]["max_turns"]

    written = 0
    with open(out_file, "a", encoding="utf-8") as fout:
        for sc in scenarios:
            sc_id = sc["id"]
            dialogue_id = f"speed_ua_{topic}_{sc_id}"
            if dialogue_id in existing:
                continue

            sys_prompt = gen_system_prompt(client, cfg, sc["description"])
            if not sys_prompt:
                continue

            num_turns = random.choice(range(min_turns, max_turns + 1, 2))
            turns = gen_speed_ua_dialogue(client, cfg, sc["description"], sys_prompt, num_turns)
            if not turns:
                continue

            record = {
                "id": dialogue_id,
                "type": "type5_speed_ua",
                "tts_backend": "indextts" if written % 2 == 0 else "breezyvoice",
                "topic": topic,
                "scenario": sc["description"],
                "system_prompt": sys_prompt,
                "turns": turns,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            logger.info(f"[{topic}] {dialogue_id} written ({len(turns)} turns)")

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--all-topics", action="store_true")
    parser.add_argument("--num-scenarios", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vllm-url", default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.vllm_url:
        cfg["vllm"]["base_url"] = args.vllm_url

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = VLLMClient(
        base_url=cfg["vllm"]["base_url"],
        model=cfg["vllm"]["model"],
        api_key=cfg["vllm"].get("api_key", "token"),
        timeout=cfg["vllm"].get("timeout", 120),
        max_retries=cfg["vllm"].get("max_retries", 5),
        retry_delay=cfg["vllm"].get("retry_delay", 2.0),
    )

    topics = cfg.get("topics", [])
    if args.all_topics:
        target_topics = topics
    elif args.topic:
        target_topics = [args.topic]
    else:
        parser.error("Specify --topic or --all-topics")

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        def _run(topic):
            return process_topic(client, cfg, topic, args.num_scenarios, out_dir)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_run, target_topics))
        logger.info(f"Total written: {sum(results)}")
    else:
        total = 0
        for topic in target_topics:
            total += process_topic(client, cfg, topic, args.num_scenarios, out_dir)
        logger.info(f"Total written: {total}")


if __name__ == "__main__":
    main()
