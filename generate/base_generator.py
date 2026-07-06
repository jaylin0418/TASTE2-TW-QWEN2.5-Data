"""
Base LLM client for vLLM (OpenAI-compatible API).
Provides sync + async batch generation with retry and JSON parsing.
"""
import json
import os
import re
import time
import logging
import asyncio
from pathlib import Path
from typing import Any
import opencc as _opencc_lib
from openai import OpenAI, AsyncOpenAI, APIError, APITimeoutError

logger = logging.getLogger(__name__)


class VLLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "token",
                 timeout: int = 120, max_retries: int = 5, retry_delay: float = 2.0):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._async_client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    # ── Sync ────────────────────────────────────────────────────────────────

    def chat(self, messages: list[dict], temperature: float = 0.85,
             top_p: float = 0.9, max_tokens: int = 2048,
             repetition_penalty: float = 1.0) -> str:
        extra = {}
        if repetition_penalty != 1.0:
            extra["repetition_penalty"] = repetition_penalty
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body=extra or None,
                )
                return resp.choices[0].message.content.strip()
            except (APIError, APITimeoutError) as e:
                if attempt == self.max_retries - 1:
                    raise
                wait = self.retry_delay * (2 ** attempt)
                logger.warning(f"API error (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s")
                time.sleep(wait)

    def chat_system(self, system: str, user: str, **gen_kwargs) -> str:
        return self.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            **gen_kwargs,
        )

    # ── Async batch ─────────────────────────────────────────────────────────

    async def _async_chat(self, messages: list[dict], temperature: float,
                          top_p: float, max_tokens: int,
                          repetition_penalty: float = 1.0) -> str:
        extra = {}
        if repetition_penalty != 1.0:
            extra["repetition_penalty"] = repetition_penalty
        for attempt in range(self.max_retries):
            try:
                resp = await self._async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body=extra or None,
                )
                return resp.choices[0].message.content.strip()
            except (APIError, APITimeoutError) as e:
                if attempt == self.max_retries - 1:
                    raise
                wait = self.retry_delay * (2 ** attempt)
                logger.warning(f"Async API error (attempt {attempt+1}): {e}. Retry in {wait:.1f}s")
                await asyncio.sleep(wait)

    def batch_chat(self, prompts: list[dict], concurrency: int = 32,
                   temperature: float = 0.85, top_p: float = 0.9,
                   max_tokens: int = 2048, repetition_penalty: float = 1.0) -> list[str]:
        """Run multiple chat calls concurrently."""
        async def _run():
            sem = asyncio.Semaphore(concurrency)
            async def _one(msgs):
                async with sem:
                    return await self._async_chat(msgs, temperature, top_p, max_tokens, repetition_penalty)
            return await asyncio.gather(*[_one(p) for p in prompts])
        return asyncio.run(_run())


class OpenAIClient:
    """Direct OpenAI API client (gpt-4o). API key from OPENAI_API_KEY env var or .env file."""

    def __init__(self, model: str = "gpt-4o", timeout: int = 120,
                 max_retries: int = 5, retry_delay: float = 2.0):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        api_key = os.environ.get("OPENAI_API_KEY") or self._load_env_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=api_key, timeout=timeout)

    @staticmethod
    def _load_env_key() -> str:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip()
        return ""

    def chat(self, messages: list[dict], temperature: float = 0.85,
             top_p: float = 0.9, max_tokens: int = 2048,
             repetition_penalty: float = 1.0) -> str:
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except (APIError, APITimeoutError) as e:
                if attempt == self.max_retries - 1:
                    raise
                wait = self.retry_delay * (2 ** attempt)
                logger.warning(f"OpenAI API error (attempt {attempt+1}): {e}. Retry in {wait:.1f}s")
                time.sleep(wait)

    def chat_system(self, system: str, user: str, **kw) -> str:
        return self.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            **kw,
        )

    def batch_chat(self, prompts: list[dict], concurrency: int = 8,
                   temperature: float = 0.85, top_p: float = 0.9,
                   max_tokens: int = 2048, repetition_penalty: float = 1.0) -> list[str]:
        async def _run():
            sem = asyncio.Semaphore(concurrency)
            async_client = AsyncOpenAI(api_key=self._client.api_key, timeout=120)

            async def _one(msgs):
                async with sem:
                    for attempt in range(self.max_retries):
                        try:
                            resp = await async_client.chat.completions.create(
                                model=self.model,
                                messages=msgs,
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=max_tokens,
                            )
                            return resp.choices[0].message.content.strip()
                        except (APIError, APITimeoutError) as e:
                            if attempt == self.max_retries - 1:
                                raise
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))

            return await asyncio.gather(*[_one(p) for p in prompts])
        return asyncio.run(_run())


# ── JSON parsing helpers ─────────────────────────────────────────────────────

def extract_json(text: str) -> Any:
    """Extract the first JSON object or array from raw LLM output."""
    # Strip markdown code blocks
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find first { ... } or [ ... ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth, end = 0, -1
        for i, c in enumerate(text[start:], start):
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
    raise ValueError(f"No valid JSON found in: {text[:200]}")


def parse_dialogue(text: str) -> list[dict]:
    """
    Parse dialogue text into list of turns.
    Expected format per line: 「甲：...」 or 「乙：...」
    Returns: [{"role": "甲"/"乙", "text": "..."}]
    Enforces strict alternation: drops consecutive turns from the same role.
    """
    turns = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = None
        for role in ("甲", "乙"):
            for sep in ("：", ":"):
                prefix = f"{role}{sep}"
                if line.startswith(prefix):
                    parsed = {"role": role, "text": line[len(prefix):].strip()}
                    break
            if parsed:
                break
        if parsed:
            # Enforce strict alternation: skip if same role as previous turn
            if turns and turns[-1]["role"] == parsed["role"]:
                continue
            turns.append(parsed)
    return turns


# OpenCC converter: simplified → Taiwan Traditional Chinese (with Taiwan-specific phrases)
_converter = _opencc_lib.OpenCC('s2twp')


# Phrases that are mainland/Cantonese style → replace with Taiwanese equivalents
# English → Chinese replacements (run before contains_english check)
_EN2ZH = [
    # Tech / internet
    ('ChatGPT', '聊天機器人'),
    ('GPT', '語言模型'),
    ('AI', '人工智慧'),
    ('App', '應用程式'), ('APP', '應用程式'), ('app', '應用程式'),
    ('WiFi', '無線網路'), ('WIFI', '無線網路'), ('Wifi', '無線網路'),
    ('GPS', '衛星導航'),
    ('ATM', '自動提款機'),
    ('PC', '電腦'),
    ('TV', '電視'),
    ('USB', '隨身碟'),
    ('QR', '條碼'),
    ('OK', '好的'), ('ok', '好的'),
    # Social / platforms (replace before simpler substrings)
    ('YouTube', '影音平台'), ('Youtube', '影音平台'),
    ('Facebook', '臉書'), ('FB', '臉書'),
    ('Instagram', '社群平台'),
    ('LINE', '通訊軟體'), ('Line', '通訊軟體'),
    ('Google', '網路搜尋'),
    ('Dcard', '論壇'),
    ('PTT', '論壇'),
    ('TikTok', '短影音平台'),
    ('Netflix', '影音串流'),
    # Medical / other acronyms
    ('COVID', '新冠肺炎'), ('Covid', '新冠肺炎'),
    ('PCR', '核酸檢測'),
    ('MRT', '捷運'),
    ('HR', '人資'),
    # Breathing / exercise terms
    ('Inhalation', '吸氣'), ('inhalation', '吸氣'),
    ('Exhalation', '呼氣'), ('exhalation', '呼氣'),
    ('Inhale', '吸氣'), ('inhale', '吸氣'),
    ('Exhale', '呼氣'), ('exhale', '呼氣'),
    ('breath', '呼吸'), ('Breath', '呼吸'), ('BREATH', '呼吸'),
    ('HOLD', '憋住'), ('Hold', '憋住'), ('hold', '維持'),
    # Finance terms
    ('interest', '利息'), ('Interest', '利息'), ('INTEREST', '利息'),
    # Sales / business
    ('Sales', '業務'), ('sales', '業務'), ('SALES', '業務'),
    ('Sale', '特賣'), ('sale', '特賣'), ('SALE', '特賣'),
    ('Salesman', '業務員'), ('salesman', '業務員'), ('SALESMAN', '業務員'),
    # Casual English
    ('maybe', '也許'), ('Maybe', '也許'),
    ('OK', '好'), ('ok', '好'), ('okay', '好'),
    ('scene', '場景'), ('Scene', '場景'), ('scenes', '場景'),
    ('pattern', '規律'), ('Pattern', '規律'),
    ('READY', '準備好'), ('ready', '準備好'), ('Ready', '準備好'),
    ('stress', '壓力'), ('Stress', '壓力'), ('STRESS', '壓力'),
    ('Pressure', '壓力'), ('pressure', '壓力'), ('PRESSURE', '壓力'),
    # Farewell
    ('byebye', '掰掰'), ('Byebye', '掰掰'), ('BYEBYE', '掰掰'),
    ('bye', '掰掰'), ('Bye', '掰掰'), ('BYE', '掰掰'),
    # Camera / photography terms
    ('ISO', '感光度'), ('iso', '感光度'),
    ('Canon', '佳能相機'), ('Nikon', '尼康相機'), ('Sony', '索尼相機'),
    # Education platforms
    ('Coursera', '線上學習平臺'), ('coursera', '線上學習平臺'),
    ('Udemy', '線上課程平臺'), ('udemy', '線上課程平臺'),
    ('MOOCs', '開放式線上課程'), ('MOOC', '線上課程'), ('mooc', '線上課程'),
    ('COURSE', '課程'), ('Course', '課程'), ('course', '課程'),
    # Commerce / events
    ('discount', '折扣'), ('Discount', '折扣'), ('DISCOUNT', '折扣'),
    ('Discounts', '折扣優惠'), ('discounts', '折扣優惠'),
    ('TICKET', '票券'), ('Ticket', '票券'), ('ticket', '票券'),
    ('PayPal', '線上支付'), ('Paypal', '線上支付'),
    # Misc action words that slip through
    ('checking', '確認'), ('Checking', '確認'),
    ('confirmation', '確認'), ('Confirmation', '確認'),
    ('CONTACT', '聯絡'), ('Contact', '聯絡'), ('contact', '聯絡'),
    ('consult', '諮詢'), ('Consult', '諮詢'),
    ('Feedback', '回饋'), ('feedback', '回饋'), ('FEEDBACK', '回饋'),
    ('Satisfaction', '滿意度'), ('satisfaction', '滿意度'),
    ('Gifts', '禮物'), ('gifts', '禮物'), ('GIFTS', '禮物'),
    ('Platforms', '平台'), ('platforms', '平台'), ('Platform', '平台'),
    ('THESE', '這些'), ('These', '這些'),
    ('Stocks', '股票'), ('stocks', '股票'), ('STOCKS', '股票'),
    ('Hearing', '回音'), ('hearing', '聆聽'),
    # Performance / events
    ('SOLD OUT', '售完'), ('sold out', '售完'), ('Sold Out', '售完'),
    ('PERFORMANCE', '演出'), ('Performance', '演出'), ('performance', '演出'),
    ('SHOW', '演出'), ('Show', '演出'), ('show', '表演'),
    ('flash', '閃光燈'), ('Flash', '閃光燈'), ('FLASH', '閃光燈'),
    ('meetup', '聚會'), ('Meetup', '聚會'), ('meet up', '聚會'),
    ('buy', '買'), ('Buy', '買'), ('BUY', '買'),
    # City / scene / Taiwan places
    ('NIGHT CITY', '夜之城'), ('Night City', '夜之城'),
    ('NIGHT MARKET', '夜市'), ('Night Market', '夜市'), ('night market', '夜市'),
    ('NIGHT CLUB', '夜店'), ('Night Club', '夜店'), ('nightclub', '夜店'),
    ('skyline', '城市天際線'), ('Skyline', '城市天際線'),
    ('scape', '景觀'),
    ('Taipei', '臺北'), ('TAIPEI', '臺北'),
    ('Shilin', '士林'), ('SHILIN', '士林'),
    ('Jiufen', '九份'), ('Tainan', '臺南'), ('Taichung', '臺中'),
    ('Kaohsiung', '高雄'), ('Keelung', '基隆'),
    # Common casual English
    ('Relax', '放鬆'), ('relax', '放鬆'), ('RELAX', '放鬆'),
    ('Experience', '體驗'), ('experience', '體驗'), ('EXPERIENCE', '體驗'),
    ('WHICH', '哪個'), ('which', '哪個'),
    ('INCLUDED', '包含'), ('included', '包含'),
    ('WHETHER', '是否'), ('whether', '是否'),
    ('comfy', '舒適'), ('Comfy', '舒適'), ('COMFY', '舒適'),
    ('Smooth', '順利'), ('smooth', '順利'), ('SMOOTH', '順暢'),
    # Health / outdoor
    ('SPF', '防曬係數'),
    ('UV', '紫外線'), ('uv', '紫外線'),
    # Research / business
    ('Forrester', '市場研究機構'), ('Gartner', '市場研究機構'),
    ('IDC', '市場研究機構'),
    ('Crunchbase', '新創資料庫'),
    ('BUSINESS WEAR', '商務服裝'), ('business wear', '商務服裝'),
    # Coding / education platforms
    ('Khan Academy', '可汗學院'),
    ('Codecademy', '程式學習平台'), ('Codewars', '程式練習平台'),
    ('Codingame', '程式練習平台'),
    ('Tinkercad', '三維建模工具'),
    ('Scratch', '積木程式工具'),
    ('Coding', '程式設計'), ('coding', '程式設計'), ('CODING', '程式設計'),
    ('Showcase', '成果展示'), ('showcase', '成果展示'),
    ('Maker', '創客'), ('maker', '創客'),
]

_PHRASE_FIXES = [
    ('好嘞', '好啦'),
    ('對頭', '對啊'),
    ('出租車', '計程車'),
    ('塑料袋', '塑膠袋'),
    ('加油囉', '繼續加油'),
    ('哦。', '喔。'),
    ('哦，', '喔，'),
    ('哦！', '喔！'),
    ('哦？', '喔？'),
    ('嘅', '的'),
    # NOTE: 係 not replaced globally — would corrupt 關係/聯係 etc.
    ('喺', '在'),
    # Cantonese particles / words
    ('冇', '沒'),
    ('嚟', '來'),
    ('啲', '些'),
    ('唔係', '不是'),
    ('唔好', '不要'),
    ('咁', '這樣'),
    ('㗎', '啦'),
    # Simplified chars that OpenCC occasionally misses
    ('体', '體'),
    # Taiwan vs. mainland usage differences (not caught by openCC s2twp)
    ('復習', '複習'),
    ('復查', '複查'),
    ('電郵', '電子郵件'),
    ('打的', '搭計程車'),
    ('的士', '計程車'),
    ('巴士', '公車'),
    ('地鐵', '捷運'),
]


_NUM_ZH = {
    '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
    '10': '十', '11': '十一', '12': '十二', '13': '十三', '14': '十四',
    '15': '十五', '16': '十六', '17': '十七', '18': '十八', '19': '十九',
    '20': '二十', '30': '三十', '40': '四十', '50': '五十',
    '60': '六十', '70': '七十', '80': '八十', '90': '九十', '100': '一百',
}


def _num_to_zh(n: int) -> str:
    """Convert integer 0–100 to Chinese."""
    if str(n) in _NUM_ZH:
        return _NUM_ZH[str(n)]
    if n < 20:
        return '十' + _NUM_ZH[str(n % 10)]
    tens, ones = n // 10, n % 10
    return _NUM_ZH[str(tens * 10)] + (_NUM_ZH[str(ones)] if ones else '')


def _percent_to_tw(m: re.Match) -> str:
    """Replace '80%' with '八十趴', '25%' with '二十五趴', etc."""
    n = int(m.group(1))
    return _num_to_zh(n) + '趴'


def fix_simplified(text: str) -> str:
    """Auto-repair simplified chars, English terms, mainland/Cantonese phrases, strip emoji."""
    text = _converter.convert(text)
    # Convert percentages: "80%" → "八十趴"
    text = re.sub(r'(\d+)%', _percent_to_tw, text)
    for bad, good in _EN2ZH:
        text = re.sub(r'(?<![A-Za-z])' + re.escape(bad) + r'(?![A-Za-z])', good, text)
    for bad, good in _PHRASE_FIXES:
        text = text.replace(bad, good)
    # Strip emoji and misc symbol characters
    # Keep: ASCII/Latin (< 0x2600), CJK and related (0x2E80–0x9FFF),
    #        Fullwidth forms 0xFF00–0xFFEF (，！？「」 etc.)
    # Strip: Misc Symbols 0x2600–0x2DFF (☕★✓ etc.) and all 0x1F000+ emoji
    text = ''.join(c for c in text if (
        ord(c) < 0x2600
        or 0x2E80 <= ord(c) <= 0x9FFF
        or 0xFF00 <= ord(c) <= 0xFFEF
    ))
    # Remove spaces between/around CJK characters; collapse remaining spaces
    text = re.sub(r'(?<=[一-鿿＀-￯])\s+(?=[一-鿿＀-￯])', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Strip disallowed punctuation; keep ，。？！、： (and halfwidth , . ? ! :)
    text = re.sub(r'[；⋯…—–「」『』【】〔〕《》〈〉（）()｀～·•※◎□△▲★☆●○◇◆]', '', text)
    # Collapse repeated allowed punctuation
    text = re.sub(r'[，,]{2,}', '，', text)
    text = re.sub(r'[。.]{2,}', '。', text)
    return text


def contains_simplified(text: str) -> bool:
    """Return True if text still contains simplified chars (OpenCC can convert more)."""
    return _converter.convert(text) != text


def contains_english(text: str) -> bool:
    """Return True if text contains English words.
    Single ASCII letters glued to CJK (e.g. Q彈, X光, V領) are Taiwan idioms — allowed."""
    cleaned = re.sub(r'(?<=[^\x00-\x7F])[A-Za-z](?=[^\x00-\x7F])', '', text)  # CJK-L-CJK
    cleaned = re.sub(r'^[A-Za-z](?=[^\x00-\x7F])', '', cleaned)               # L-CJK at start
    cleaned = re.sub(r'(?<=[^\x00-\x7F])[A-Za-z]$', '', cleaned)              # CJK-L at end
    return bool(re.search(r'[A-Za-z]', cleaned))


def contains_tailo(text: str) -> bool:
    """Return True if text contains Tâi-lô romanized Hokkien (Latin with diacritics)."""
    return bool(re.search(r'[āáàâēéèêīíìîōóòôūúùûńm̄]', text, re.IGNORECASE))


def contains_korean(text: str) -> bool:
    """Return True if text contains any Korean (Hangul) character."""
    return bool(re.search(r'[가-힣ᄀ-ᇿ㄰-㆏]', text))


def contains_foreign_script(text: str) -> bool:
    """Return True if text contains Cyrillic, Thai, Arabic, Hebrew, or other non-CJK foreign scripts."""
    return bool(re.search(
        r'[Ѐ-ӿ'   # Cyrillic
        r'฀-๿'   # Thai
        r'؀-ۿ'   # Arabic
        r'֐-׿'   # Hebrew
        r'ऀ-ॿ'   # Devanagari
        r'぀-ヿ]', # Japanese kana (hiragana/katakana)
        text
    ))


def contains_emoji(text: str) -> bool:
    """Return True if text contains emoji or misc symbol characters."""
    return any(
        (0x2600 <= ord(c) <= 0x2DFF) or ord(c) >= 0x1F000
        for c in text
    )


def clean_turn(text: str) -> str:
    """Remove leading role prefix if accidentally included."""
    return re.sub(r'^[甲乙][：:]\s*', '', text).strip()
