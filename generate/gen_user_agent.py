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
import re
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from generate.base_generator import VLLMClient, extract_json, contains_english, contains_simplified, contains_korean, contains_emoji, contains_tailo, contains_foreign_script, fix_simplified

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SCENARIOS = 5  # 每次請 LLM 生成幾個 scenario


_FAREWELL = ("明天見", "下次見", "再見", "晚安", "掰掰", "後會有期",
             "有空再來", "有空再聊", "有需要再來", "期待下次", "保重",
             "一路平安", "路上平安", "路上小心", "祝你順利", "祝你成功",
             "教學順利", "工作順利", "祝順利", "注意安全")

_META = ("自動生成", "如需繼續", "請告訴我", "我是AI", "我是人工智能",
         "作為AI", "作為一個AI", "語言模型",
         "台式說法", "大陸腔", "怎麼說比較台", "台灣說法是",
         "台式一點應該", "應該說怎麼樣")

_HOKKIEN = ("昨暝", "今暝", "明暝", "呷飯", "呷水", "嘸", "哩", "汝",
            "伊", "阮", "怹", "佇", "欲去", "袂", "毋", "攏", "遐",
            "遮", "佮", "啥料", "啥米", "足讚", "真讚", "嘿呦")


def _ngram_too_similar(new_text: str, history: list[str], n: int = 4, threshold: float = 0.65) -> bool:
    """Return True if new_text shares >65% 4-gram overlap with a recent same-role turn."""
    if not history or len(new_text) < n:
        return False
    new_ngrams = set(new_text[i:i+n] for i in range(len(new_text) - n + 1))
    if not new_ngrams:
        return False
    for old in history[-3:]:
        if len(old) < n:
            continue
        old_ngrams = set(old[i:i+n] for i in range(len(old) - n + 1))
        if len(new_ngrams & old_ngrams) / len(new_ngrams) >= threshold:
            return True
    return False


def _is_hard_reject(text: str) -> bool:
    """Return True for text that can't be fixed by LLM (use last_text instead)."""
    if not text:
        return True
    if re.search(r'[一-鿿] [一-鿿] [一-鿿]', text):  # garbled spaced characters
        return True
    if re.search(r"[{}\[\]]|['\";]{2,}|->|=>|\)\s*[->]", text):  # code artifacts
        return True
    if re.search(r'[:;]-?[)(DP]|[)(]-?[:;]', text):  # ASCII emoticons
        return True
    if contains_korean(text):
        return True
    if contains_foreign_script(text):
        return True
    # Repetitive garbled output (model error artifacts)
    if re.search(r'(.{3,})\1{2,}', text):
        return True
    # Model meta-output / error messages
    if re.search(r'重新生成|重試輸出|輸入認知例外|請重新輸入', text):
        return True
    return False


def _clean_raw_turn(raw: str, role_a: str, role_b: str) -> str:
    """Strip role prefix, trim, cap length, add sentence-final punctuation."""
    text = raw.strip()
    text = text.splitlines()[0].strip() if text else ""
    for prefix in (f"{role_a}：", f"{role_a}:", f"{role_b}：", f"{role_b}:",
                   "甲：", "甲:", "乙：", "乙:", "User：", "User:",
                   "Agent：", "Agent:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    text = re.sub(r"[\s'\";\)\(\{\}\[\]>!?.]+$", "", text).strip()
    MAX_CHARS = 75
    if len(text) > MAX_CHARS:
        # Prefer cutting at a sentence-final boundary near the cap
        cut = MAX_CHARS
        for j in range(MAX_CHARS - 1, max(MAX_CHARS - 20, 0), -1):
            if text[j] in "。！？…」）》":
                cut = j + 1
                break
        text = text[:cut].rstrip("，、")
    if text and text[-1] not in "。！？…」）》":
        text += "。"
    return text


def _llm_fix_once(client, text: str, problem_desc: str,
                  fix_client=None) -> str | None:
    """One-shot LLM rewrite for a detected problem. Returns fixed text or None.
    Uses fix_client if provided (e.g. GPT-4o), otherwise falls back to client."""
    fc = fix_client if fix_client is not None else client
    try:
        fixed = fc.chat_system(
            system=(
                "你是繁體中文台灣腔改寫助理。\n"
                f"問題：這句話{problem_desc}\n"
                "要求：嚴格修正此問題，輸出改寫後的繁體中文台灣腔一行，不加任何說明。\n"
                "修正後不可出現英文字母、注音符號、emoji或簡體字。"
            ),
            user=text,
            temperature=0.3,
            max_tokens=150,
        )
        fixed = fixed.strip().splitlines()[0]
        return fixed or None
    except Exception as e:
        logger.warning(f"LLM fix exception: {e}")
        return None


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


_TW_TOPICS = {"台灣文化", "台灣旅遊", "台灣社會", "台灣民俗信仰"}

_TW_HINTS_BY_TOPIC = {
    "台灣旅遊": (
        "台灣旅遊在地特色提示（情境必須出現其中具體的地點或景點）：\n"
        "台北市：台北101、象山步道、信義商圈、西門町、師大夜市、公館、"
        "大安森林公園、中正紀念堂、國立故宮博物院、士林官邸、內湖科學園區、"
        "貓空纜車、關渡宮、淡水紅毛城、漁人碼頭、北投溫泉、陽明山竹子湖\n"
        "新北市：九份老街、金瓜石、平溪天燈、菁桐、十分瀑布、烏來老街溫泉、"
        "深坑臭豆腐街、三峽老街、鶯歌陶瓷、三芝、野柳地質公園、基隆廟口夜市\n"
        "桃園：大溪老街、拉拉山、石門水庫、小烏來瀑布\n"
        "新竹：內灣老街、北埔冷泉、玻璃工藝博物館、竹北\n"
        "苗栗：三義木雕、南庄老街、西湖渡假村、飛牛牧場\n"
        "台中：逢甲夜市、宮原眼科、彩虹眷村、大坑步道、東勢林場、"
        "清境農場、合歡山、武陵農場、梨山\n"
        "彰化：鹿港老街、八卦山大佛、溪洲公園、田尾公路花園\n"
        "南投：日月潭、集集綠色隧道、溪頭森林、惠蓀林場、奧萬大賞楓\n"
        "雲林：劍湖山、斗六太平老街、古坑咖啡、草嶺風景區\n"
        "嘉義：阿里山森林鐵路、奮起湖、瑞里竹林、布袋蚵田\n"
        "台南：赤崁樓、安平古堡、安平老街、孔廟、神農街、十鼓文化村、"
        "奇美博物館、七股鹽山、關子嶺溫泉、虎頭埤\n"
        "高雄：愛河、六合夜市、旗津海岸、駁二藝術特區、美濃客家文化、"
        "佛陀紀念館、茂林紫蝶幽谷、寶來溫泉\n"
        "屏東：墾丁大街、南灣、鵝鑾鼻、佳樂水、小琉球潮間帶、霧台原鄉\n"
        "花蓮：太魯閣峽谷、七星潭、鯉魚潭、花蓮港、石藝大街、瑞穗溫泉、"
        "光復糖廠、玉里溫泉、六十石山金針花海、富里鄉\n"
        "台東：池上稻田伯朗大道、台東熱氣球、三仙台、成功漁港、知本溫泉、"
        "鹿野高台、東河包子、綠島朝日溫泉、蘭嶼達悟族\n"
        "澎湖：天堂路、奎壁山摩西分海、菊島跨海大橋、漁翁島燈塔\n"
        "金門：金城老街、莒光樓、古寧頭、馬山觀測站、高粱酒廠\n"
        "馬祖：藍眼淚、芹壁聚落、北竿壁山、大漢據點\n"
        "交通：高鐵、台鐵、捷運、租機車、公路客運、台灣好行、觀光巴士\n"
        "住宿：民宿、背包客棧、溫泉旅館、露營區、特色城堡民宿"
    ),
    "台灣文化": (
        "台灣文化在地特色提示：\n"
        "飲食文化：滷肉飯、蚵仔煎、珍珠奶茶、臭豆腐、芒果冰、鹽酥雞、刈包、"
        "紅豆湯圓、米苔目、肉圓、虱目魚粥、大腸麵線\n"
        "夜市：士林、寧夏、逢甲、六合、花園夜市、瑞豐夜市\n"
        "傳統店舖：雜貨店、傳統市場、布行、中藥行、金紙店\n"
        "文化場所：廟宇、誠品書店、傳統戲院、南管北管表演場\n"
        "表演：歌仔戲、布袋戲、八家將、電子花車\n"
        "節慶：元宵花燈、清明掃墓、端午賽龍舟、中元普渡、中秋烤肉、尾牙\n"
        "語言特色：台語、客語、閩南語諺語"
    ),
    "台灣社會": (
        "台灣社會在地特色提示：\n"
        "生活場景：便利商店（7-11、全家）、夜市、傳統菜市場、捷運站、高鐵站\n"
        "社會議題：少子化、長照、房價、外送員、打工族、勞工假期、健保\n"
        "族群：本省、外省、原住民、新住民、東南亞移工\n"
        "世代話題：青年創業、返鄉、遠距工作、社群媒體使用\n"
        "教育：升學壓力、補習班、大學入學考試、技職教育\n"
        "媒體：LINE群組、PTT、Dcard、台灣新聞台"
    ),
    "台灣民俗信仰": (
        "台灣民俗信仰在地特色提示：\n"
        "主要信仰：媽祖、城隍爺、土地公、關聖帝君、王爺、玉皇大帝\n"
        "廟宇活動：進香、遶境、擲筊、問卜、收驚、安太歲\n"
        "知名廟宇：大甲鎮瀾宮、北港朝天宮、台南天后宮、行天宮、龍山寺\n"
        "民俗節慶：元宵燈節、清明、端午、中元普渡、中秋、冬至、尾牙\n"
        "特殊活動：蜂炮（鹽水）、炸寒單（台東）、平溪天燈、水燈排（頭城搶孤）\n"
        "祭祀習慣：初一十五拜拜、祖先牌位、金紙、鞭炮、牲禮"
    ),
}

_TW_HINTS = (
    "台灣在地特色提示：\n"
    "- 地點：台北、高雄、台中、台南、花蓮、台東\n"
    "- 場所：夜市、廟宇、捷運站、便利商店、傳統市場\n"
    "- 交通：捷運、高鐵、台鐵、機車、公車"
)


def gen_scenarios(client: VLLMClient, cfg: dict, topic: str, n: int,
                  id_offset: int = 0) -> list[dict]:
    """Generate n scenarios; id_offset shifts IDs so parallel shards don't collide."""
    prompt_tmpl = cfg["prompts"]["scenario"]
    taiwan_hints = _TW_HINTS_BY_TOPIC.get(topic, _TW_HINTS if topic in _TW_TOPICS else cfg.get("taiwan_hints", ""))
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
                    sid = f"s{id_offset + len(scenarios) + 1:04d}"
                    desc = fix_simplified(s.get("description", ""))
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
    for attempt in range(max(cfg["vllm"]["max_retries"], 8)):
        try:
            result = client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
                repetition_penalty=1.0,  # no penalty for single-shot system prompt
            )
            # Take up to 2 non-empty lines (1-2 sentence system prompt).
            # Stop before any line that looks like a dialogue turn.
            lines = []
            for ln in result.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                if any(ln.startswith(p) for p in ("甲：", "乙：", "User：", "Agent：", "-", "•")):
                    break
                lines.append(ln)
                if len(lines) == 2:
                    break
            text = "".join(lines)
            text = fix_simplified(text)
            if text and not contains_english(text):
                return text
        except Exception as e:
            logger.warning(f"System prompt gen failed (attempt {attempt+1}): {e}")
            time.sleep(cfg["vllm"]["retry_delay"])
    return f"這是一段關於{cfg.get('type','對話')}的情境對話。"


def gen_dialogue(client: VLLMClient, cfg: dict, scenario: str,
                 system_prompt: str, num_turns: int,
                 fix_client=None, b_client=None) -> list[dict]:
    """Generate dialogue turn by turn — one LLM call per turn, strict alternation.
    client is used for role-A (User) turns; b_client for role-B (Agent) turns.
    If b_client is None, client is used for both."""
    a_sys = cfg["prompts"]["a_turn"].format(
        scenario=scenario, system_prompt=system_prompt)
    b_sys = cfg["prompts"]["b_turn"].format(
        scenario=scenario, system_prompt=system_prompt)

    rep_penalty = cfg.get("gen", {}).get("repetition_penalty", 1.0)
    turns: list[dict] = []

    roles = cfg.get("roles", {})
    role_a = roles.get("a", "甲")
    role_b = roles.get("b", "乙")
    backchannel_rate = cfg.get("backchannel_rate", 0.15)

    for i in range(num_turns):
        is_a = (i % 2 == 0)
        role = role_a if is_a else role_b
        sys_content = a_sys if is_a else b_sys
        active_client = client if is_a else (b_client or client)

        # First turn: tell the model to open the conversation
        if i == 0:
            sys_content += "\n\n【注意：這是對話的第一句。請主動開口說出你的問題或話題，不要用「嗯」「好」「讓我想想」等回應性詞語開頭。】"
        # Last turn: role-specific, no farewell
        elif i == num_turns - 1:
            if is_a:
                sys_content += "\n\n【注意：這是最後一輪。說你最後的反應、追問或感受，不要道別、不要道謝、不要說再見。】"
            else:
                sys_content += "\n\n【注意：這是最後一輪。給一句補充提示或具體建議直接結束，絕對不能說「再見」「祝你」「有需要再來」「隨時來問」之類。】"
        # User role anchor: remind model of role + forbidden Agent-like patterns
        elif is_a:
            sys_content += "\n\n【你是使用者：說你的問題、反應或感受。禁止說「你可以…」「建議你…」「推薦你…」「你去…」等助理語氣。】"
        # Backchannel: non-first A turns have 15% chance of forced short response
        if is_a and i > 0 and random.random() < backchannel_rate:
            sys_content += "\n\n【注意：這次請輸出十個字以內的超短回應。例如：表示聽到了（嗯、哦好、哦這樣）、需要時間（等我一下、我看看、讓我想想）、接受說法（好好、知道了）。】"
        # Mid-conversation progress hint: inject every 3 turns (turns 3, 6, 9...)
        if i > 0 and i % 3 == 0 and i < num_turns - 1:
            same_role_turns = [t["text"] for j, t in enumerate(turns) if (j % 2 == 0) == is_a]
            if same_role_turns:
                recent = "、".join(f'「{t[:15]}…」' for t in same_role_turns[-2:])
                sys_content += f"\n\n【推進提示：你已說過 {recent}。這一輪必須補充新資訊或推進話題，不可再重複這些意思。】"

        # Build chat messages: own previous turns = "assistant", other's = "user"
        messages = [{"role": "system", "content": sys_content}]
        for j, t in enumerate(turns):
            chat_role = "assistant" if ((j % 2 == 0) == is_a) else "user"
            messages.append({"role": chat_role, "content": t["text"]})

        prev_text = turns[-2]["text"] if len(turns) >= 2 else None  # same-role prev turn
        prev_other_text = turns[-1]["text"] if len(turns) >= 1 else None  # other-role prev turn

        last_text = None

        # ── STEP 1: Generate once ────────────────────────────────────────────
        try:
            raw = active_client.chat(
                messages,
                temperature=cfg["gen"]["temperature"],
                top_p=cfg["gen"]["top_p"],
                max_tokens=256,
                repetition_penalty=rep_penalty,
            )
            text = _clean_raw_turn(raw, role_a, role_b)
        except Exception as e:
            logger.warning(f"Turn {i} generation failed: {e}")
            if last_text is not None:
                turns.append({"role": role, "text": last_text})
            else:
                break
            continue

        # ── STEP 2: Deterministic fixes ──────────────────────────────────────
        text = fix_simplified(text)

        # ── STEP 3: Hard rejects → downgrade to GPT-4o with full context ────
        if _is_hard_reject(text):
            logger.warning(f"Turn {i} hard reject: {text[:50]!r}")
            if fix_client is not None:
                try:
                    raw_regen = fix_client.chat(
                        messages,
                        temperature=0.85,
                        top_p=0.9,
                        max_tokens=256,
                    )
                    regen = _clean_raw_turn(raw_regen, role_a, role_b)
                    regen = fix_simplified(regen)
                    if not _is_hard_reject(regen):
                        logger.info(f"Turn {i} hard reject recovered via GPT-4o: {regen[:50]!r}")
                        text = regen
                    else:
                        logger.warning(f"Turn {i} GPT-4o regen still hard reject, using last_text")
                        text = last_text
                        if text is None:
                            break
                except Exception as e:
                    logger.warning(f"Turn {i} GPT-4o regen failed: {e}")
                    text = last_text
                    if text is None:
                        break
            else:
                text = last_text
                if text is None:
                    break

        # ── STEP 4: Collect same-role history for duplicate check ────────────
        same_role_history = [t["text"] for j, t in enumerate(turns) if (j % 2 == 0) == is_a]

        # ── STEP 5: Detect fixable problem → one LLM fix call ────────────────
        problem = None
        # User role violation: User sounding like Agent (highest priority)
        if is_a and re.search(
            r'你可以(試|去|用|考慮|選擇|找|看)'   # telling Agent what to do
            r'|^建議|^推薦你|你需要'              # advice-giving openers
            r'|^當然可以'                        # Agent-style affirmation opener
            r'|隨時歡迎(提問|詢問|來問)'          # Agent-style welcoming
            r'|幫你(搜|找|查|整理|準備|提供|蒐集)', # User offering to help Agent
            text
        ):
            problem = ("這是 User 的台詞，但聽起來像 AI 助理在說話。"
                       "請改寫成普通人跟 AI 說話的口氣：問問題、說自己的困擾/反應/感受/執行結果。"
                       "不要以「你可以」「建議你」「當然可以」「隨時歡迎提問」等 AI 語氣開頭。")
        elif contains_emoji(text):
            problem = "含有emoji表情符號，請移除emoji並保留文字意思"
        elif re.search(r'[ㄅ-ㄩˊˇˋ˙]', text):
            problem = "含有注音符號（如ㄟ/ㄛ等），請改用漢字表達語氣（ㄟ→欸、ㄛ→喔、ㄌㄏㄚ→啦）"
        elif contains_english(text):
            problem = ("含有英文字母，所有英文一律改成中文，包括歌手名、樂團名、頻道名、品牌名、軟體名。"
                       "改法：用中文功能或類別描述替代，例如：一位英國電音歌手、一個美國搖滾樂團、"
                       "一個教唱歌技巧的頻道、圖片收藏平台、影音平台、應用程式。"
                       "不可保留任何英文字母。")
        elif contains_simplified(text):
            problem = "含有簡體字，請改成繁體中文台灣腔"
        elif contains_tailo(text):
            problem = "含有台羅拼音，請改成繁體中文漢字"
        elif any(w in text for w in _HOKKIEN):
            problem = "含有閩南語詞彙，請改成繁體中文台灣腔說法"
        elif i < num_turns - 1 and any(f in text for f in _FAREWELL):
            problem = "非最後一輪，請去掉道別語句，改成繼續對話話題"
        elif re.search(r'[《》]', text):
            problem = "含有書名號《》，歌名、書名、專輯名直接用文字說明，不加任何符號包裹"
        elif re.search(r'(別忘了|記得要|注意要)[。！？]$', text):
            problem = "句子不完整，「別忘了」「記得要」後面缺少受詞，請補完整說出具體內容"
        elif any(m in text for m in _META):
            problem = "不可描述自己是AI或語言模型，請直接用台灣腔說出回應內容"

        if problem:
            fixed = _llm_fix_once(client, text, problem, fix_client=fix_client)
            if fixed and not _is_hard_reject(fixed):
                # Re-verify the fix actually resolved the original problem
                _user_role_pat = (r'你可以(試|去|用|考慮|選擇|找|看)|^建議|^推薦你|你需要'
                                  r'|^當然可以|隨時歡迎(提問|詢問|來問)'
                                  r'|幫你(搜|找|查|整理|準備|提供|蒐集)')
                still_bad = (
                    (is_a and bool(re.search(_user_role_pat, fixed)) if is_a and re.search(_user_role_pat, text) else False)
                    or (contains_emoji(fixed) if contains_emoji(text) else False)
                    or (re.search(r'[ㄅ-ㄩˊˇˋ˙]', fixed) if re.search(r'[ㄅ-ㄩˊˇˋ˙]', text) else False)
                    or (contains_english(fixed) if contains_english(text) else False)
                    or (contains_simplified(fixed) if contains_simplified(text) else False)
                    or (bool(re.search(r'[《》]', fixed)) if re.search(r'[《》]', text) else False)
                )
                if still_bad:
                    logger.warning(f"LLM fix didn't resolve problem, keeping original: {text[:50]!r}")
                else:
                    logger.info(f"LLM fix: {text[:30]!r} → {fixed[:30]!r}")
                    text = fixed
            else:
                logger.warning(f"LLM fix failed, keeping original: {text[:50]!r}")

        # ── STEP 4.5: Update last_text only after text is fully cleaned ───────
        if not (text == prev_text or text == prev_other_text or
                _ngram_too_similar(text, same_role_history)):
            last_text = text

        # ── STEP 6: Final similarity fallback ────────────────────────────────
        if (text == prev_text or text == prev_other_text or
                _ngram_too_similar(text, same_role_history)):
            logger.warning(f"Turn {i} too similar, using last_text")
            text = last_text or text

        turns.append({"role": role, "text": text})

    if len(turns) < 4:
        logger.warning(f"Only {len(turns)} turns generated")
        return []
    return turns


def process_topic(client: VLLMClient, cfg: dict, topic: str,
                  num_scenarios: int, out_dir: Path, b_client=None, fix_client=None,
                  scenario_offset: int = 0):
    topic_dir = out_dir / topic
    topic_dir.mkdir(parents=True, exist_ok=True)

    # With sharding (offset > 0) each worker writes its own partial file.
    # The job script merges and writes done.flag after all shards finish.
    # With no sharding (offset == 0), use done.flag as before.
    if scenario_offset == 0:
        done_file = topic_dir / "done.flag"
        if done_file.exists():
            logger.info(f"[{topic}] Already done, skipping.")
            return
        out_jsonl = topic_dir / "dialogues.jsonl"
    else:
        out_jsonl = topic_dir / f"dialogues_off{scenario_offset:05d}.jsonl"

    existing_ids = set()
    if out_jsonl.exists():
        with open(out_jsonl) as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    logger.info(f"[{topic}] offset={scenario_offset} resuming from {len(existing_ids)} existing")

    scenarios = gen_scenarios(client, cfg, topic, num_scenarios, id_offset=scenario_offset)
    cfg_type = cfg.get("type", "user_agent")

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for i, sc in enumerate(scenarios):
            sc_id = sc["id"]
            dialogue_id = f"{cfg_type}_{topic}_{sc_id}"
            if dialogue_id in existing_ids:
                continue

            backend = "breezyvoice" if i < int(num_scenarios * 0.75) else "indextts"

            num_turns = random.choice(range(8, 17, 2))  # 8,10,12,14,16
            sys_prompt = gen_system_prompt(client, cfg, sc["description"])
            turns = gen_dialogue(client, cfg, sc["description"], sys_prompt, num_turns,
                                 fix_client=fix_client, b_client=b_client)

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

    if scenario_offset == 0:
        done_file.touch()
    logger.info(f"[{topic}] offset={scenario_offset} done. Saved to {out_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Generate User/Agent dialogues")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--topic", help="Single topic to process")
    parser.add_argument("--all-topics", action="store_true")
    parser.add_argument("--num-scenarios", type=int, default=None,
                        help="Override scenarios_per_topic from config")
    parser.add_argument("--scenario-offset", type=int, default=0,
                        help="Starting scenario index for this shard (0-based)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--vllm-url", default=None, help="Override vLLM base URL for User (role-A)")
    parser.add_argument("--b-vllm-url", default=None, help="Override vLLM base URL for Agent (role-B); defaults to same as --vllm-url")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.vllm_url:
        cfg["vllm"]["base_url"] = args.vllm_url
    num_scenarios = args.num_scenarios or cfg.get("scenarios_per_topic", 200)
    out_dir = Path(args.output)

    def _make_vllm(url=None):
        return VLLMClient(
            base_url=url or cfg["vllm"]["base_url"],
            model=cfg["vllm"]["model"],
            api_key=cfg["vllm"].get("api_key", "token"),
            timeout=cfg["vllm"].get("timeout", 120),
            max_retries=cfg["vllm"].get("max_retries", 5),
            retry_delay=cfg["vllm"].get("retry_delay", 2.0),
        )

    a_client = _make_vllm()
    b_client = _make_vllm(args.b_vllm_url)  # same URL if not overridden

    # GPT-4o fix client: auto-init if OPENAI_API_KEY is set; falls back to Qwen if not
    fix_client = None
    try:
        from generate.base_generator import OpenAIClient
        fix_client = OpenAIClient(model="gpt-4o")
        logger.info("GPT-4o fix client ready.")
    except Exception as e:
        logger.info(f"No GPT-4o fix client (will use Qwen for fixes): {e}")

    topics = [args.topic] if args.topic else (cfg["topics"] if args.all_topics else [])
    if not topics:
        parser.error("Provide --topic or --all-topics")

    for topic in topics:
        process_topic(a_client, cfg, topic, num_scenarios, out_dir,
                      b_client=b_client, fix_client=fix_client,
                      scenario_offset=args.scenario_offset)


if __name__ == "__main__":
    main()
