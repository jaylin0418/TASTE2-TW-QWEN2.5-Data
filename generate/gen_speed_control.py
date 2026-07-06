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
from generate.base_generator import VLLMClient
from generate.gen_user_agent import load_config, gen_scenarios, gen_system_prompt, gen_dialogue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Speed phrase pools ────────────────────────────────────────────────────────

_SPEED_PHRASES = {
    "fast": [
        # 直接請求
        "請說快一點。", "說快點好嗎？", "能加快速度嗎？", "說快些。",
        "麻煩說快一點。", "講快點吧。", "請你說快一點。", "說快一點就好。",
        "能不能說快一點？", "說快點謝謝。", "可以說快點嗎？", "快一點說。",
        "說得快一些好嗎？", "講快一點謝謝。", "請說得快一點。",
        "加快一點說好嗎？", "能快點嗎？", "請加快速度說。",
        "說話速度快一點可以嗎？", "麻煩加快一點。",
        # 口語變化
        "快一點講可以嗎？", "拜託說快一點啦。", "速度加快一點吧。",
        "可不可以說快點？", "說話快一點好嗎？", "講話能不能快一點？",
        "麻煩你說快點。", "快點說吧。", "說話速度能快一點嗎？",
        "能不能加快一點？", "語速快一點好嗎？", "說快一點可以嗎？",
        "你可以說快一點嗎？", "速度稍微快一點好嗎？", "用快一點的速度說吧。",
        "說話稍微快點好嗎？", "快點講嘛。", "加速一點說可以嗎？",
        "說話能快一點嗎？", "快一點說好嗎？",
    ],
    "slow": [
        # 直接請求
        "請說慢一點。", "說慢點好嗎？", "能放慢速度嗎？", "說慢些。",
        "麻煩說慢一點。", "講慢點吧。", "請你說慢一點。", "說慢一點就好。",
        "能不能說慢一點？", "說慢點謝謝。", "可以說慢點嗎？", "慢一點說。",
        "說得慢一些好嗎？", "講慢一點謝謝。", "請說得慢一點。",
        "放慢一點說好嗎？", "能慢點嗎？", "請放慢速度說。",
        "說話速度慢一點可以嗎？", "麻煩放慢一點。",
        # 口語變化
        "慢一點講可以嗎？", "拜託說慢一點啦。", "速度放慢一點吧。",
        "可不可以說慢點？", "說話慢一點好嗎？", "講話能不能慢一點？",
        "麻煩你說慢點。", "慢點說吧。", "說話速度能慢一點嗎？",
        "能不能放慢一點？", "語速慢一點好嗎？", "說慢一點可以嗎？",
        "你可以說慢一點嗎？", "速度稍微慢一點好嗎？", "用慢一點的速度說吧。",
        "說話稍微慢點好嗎？", "慢點講嘛。", "放慢速度說可以嗎？",
        "說話能慢一點嗎？", "慢一點說好嗎？",
    ],
    "normal": [
        "一般速度說就好。", "說正常速度就好。", "普通速度就行了。",
        "不用那麼快，說正常就好。", "不用那麼慢了。", "說正常點就行了。",
        "速度正常就好。", "可以恢復正常速度了。",
        "不用特別快或慢，正常說就好。", "恢復一般速度就行了。",
        "說話速度正常就可以了。", "就照平常速度說吧。",
        "不用那麼趕，正常說就好。", "速度回到正常就行了。",
        "照一般速度說就好了。", "說話正常速度就可以了。",
        "恢復正常速度吧。", "正常說就好不用特別調整。",
    ],
}

# 50% 回到 normal 時不說任何速度措辭（另 50% 明確說出）
_P_SILENT_NORMAL = 0.50

_SPEED_KEYWORDS = {
    "slow": ["慢一點", "慢點", "放慢", "說慢", "慢慢", "慢速", "緩慢", "慢些"],
    "fast": ["快一點", "快點", "加快", "說快", "快速", "快些", "加速"],
    "normal": ["正常速度", "普通速度", "一般速度", "速度正常", "回到正常",
               "不用特別快", "不用那麼快", "不用那麼慢", "不用那麼趕",
               "照正常", "恢復正常", "正常說", "正常點", "照平常"],
}


def _detect_speed(text: str):
    """Return 'fast'/'slow'/'normal'/None based on speed keywords in User text."""
    for speed, kws in _SPEED_KEYWORDS.items():
        if any(kw in text for kw in kws):
            return speed
    return None


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
        self.has_left_normal = False

    def next_exchange(self) -> str:
        """Advance FSM by one exchange (User+Agent pair) and return new state."""
        if self.state == "normal":
            p_leave = self.p_leave_first if not self.has_left_normal else self.p_leave_again
            if random.random() < p_leave:
                self.state = random.choice(["fast", "slow"])
                self.has_left_normal = True
        else:
            if random.random() < self.p_return:
                self.state = "normal"
        return self.state


_SPEED_DIRECTION_ZH = {
    "fast": "加快",
    "slow": "放慢",
    "normal": "恢復正常",
}

_SPEED_LLM_SYSTEM = (
    "你是台灣口語對話改寫助理。\n"
    "任務：在使用者這句話裡，自然地加入一個要求對方「{direction}」語速的請求。\n"
    "要求：\n"
    "1. 語速請求要融入句子，不要生硬地硬接在最後\n"
    "2. 措辭每次都要不一樣，越口語越自然越好\n"
    "3. 只輸出改寫後的完整句子，不加任何說明\n"
    "4. 繁體中文台灣腔，不可出現英文或簡體字"
)


def _llm_inject_speed(client, text: str, speed: str) -> str:
    """Use LLM to naturally embed a speed request into the user's text."""
    direction = _SPEED_DIRECTION_ZH.get(speed, "加快")
    try:
        result = client.chat_system(
            system=_SPEED_LLM_SYSTEM.format(direction=direction),
            user=text,
            temperature=0.9,
            max_tokens=120,
        )
        result = result.strip().splitlines()[0].strip()
        if result:
            return result
    except Exception as e:
        logger.warning(f"LLM speed inject failed: {e}")
    # Fallback to static phrase
    phrase = random.choice(_SPEED_PHRASES[speed])
    return text.rstrip("。") + "，" + phrase


def apply_speed_fsm(turns: list[dict], fsm_cfg: dict,
                    client=None) -> list[dict]:
    """
    Apply speed FSM per exchange (User turn + Agent turn).

    Sticky rule: Agent speed only changes when User EXPLICITLY mentions speed.
    Silent turns (no phrase injected) keep the previous active speed.

    Injection rules:
      - First exchange in normal: 50% silent (no phrase), 50% inject normal phrase
      - Transitioning fast/slow → normal: always inject normal phrase (never silent)
      - Transitioning normal → fast/slow: always inject phrase
    """
    fsm = SpeedFSM(
        p_leave_first=fsm_cfg.get("p_leave_first", 0.75),
        p_leave_again=fsm_cfg.get("p_leave_again", 0.40),
        p_return=fsm_cfg.get("p_return", 0.40),
    )
    prev_speed = "normal"
    active_speed = "normal"  # what Agent actually uses (only changes on explicit instruction)
    is_first = True
    i = 0
    while i < len(turns):
        speed = fsm.next_exchange()

        # Determine whether User will explicitly mention speed this exchange
        explicit = False
        if is_first:
            explicit = (random.random() >= _P_SILENT_NORMAL)  # 50% chance at start
        elif speed != prev_speed:
            if speed == "normal":
                explicit = True  # fast/slow → normal: always explicit
            else:
                explicit = True  # normal → fast/slow: always explicit

        # User turn: inject phrase if explicit, no speed tag on User
        if i < len(turns):
            t = turns[i]
            if explicit:
                if client is not None:
                    candidate = _llm_inject_speed(client, t["text"], speed)
                    # Only accept LLM result if it contains a detectable keyword
                    if _detect_speed(candidate) == speed:
                        t["text"] = candidate
                    else:
                        phrase = random.choice(_SPEED_PHRASES[speed])
                        t["text"] = t["text"].rstrip("。") + "，" + phrase
                else:
                    phrase = random.choice(_SPEED_PHRASES[speed])
                    t["text"] = t["text"].rstrip("。") + "，" + phrase
                active_speed = speed  # FSM-injected instruction

            # Also catch naturally-occurring speed requests from LLM
            detected = _detect_speed(t["text"])
            if detected is not None:
                active_speed = detected

            t["speed"] = ""
            i += 1
            is_first = False

        # Agent turn: uses active_speed (sticky); normal → no tag
        if i < len(turns):
            turns[i]["speed"] = active_speed if active_speed != "normal" else ""
            i += 1

        prev_speed = speed
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
            turns = gen_dialogue(client, base_cfg, sc["description"], sys_prompt, num_turns)  # type: ignore[arg-type]

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
