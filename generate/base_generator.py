"""
Base LLM client for vLLM (OpenAI-compatible API).
Provides sync + async batch generation with retry and JSON parsing.
"""
import json
import re
import time
import logging
import asyncio
from typing import Any
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
             top_p: float = 0.9, max_tokens: int = 2048) -> str:
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
                          top_p: float, max_tokens: int) -> str:
        for attempt in range(self.max_retries):
            try:
                resp = await self._async_client.chat.completions.create(
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
                logger.warning(f"Async API error (attempt {attempt+1}): {e}. Retry in {wait:.1f}s")
                await asyncio.sleep(wait)

    def batch_chat(self, prompts: list[dict], concurrency: int = 32,
                   temperature: float = 0.85, top_p: float = 0.9,
                   max_tokens: int = 2048) -> list[str]:
        """Run multiple chat calls concurrently."""
        async def _run():
            sem = asyncio.Semaphore(concurrency)
            async def _one(msgs):
                async with sem:
                    return await self._async_chat(msgs, temperature, top_p, max_tokens)
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
    """
    turns = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        for role in ("甲", "乙"):
            prefix = f"{role}："
            if line.startswith(prefix):
                turns.append({"role": role, "text": line[len(prefix):].strip()})
                break
        else:
            # Try with ASCII colon
            for role in ("甲", "乙"):
                prefix = f"{role}:"
                if line.startswith(prefix):
                    turns.append({"role": role, "text": line[len(prefix):].strip()})
                    break
    return turns


def contains_english(text: str) -> bool:
    """Return True if text contains any ASCII letter."""
    return bool(re.search(r'[A-Za-z]', text))


def clean_turn(text: str) -> str:
    """Remove leading role prefix if accidentally included."""
    return re.sub(r'^[甲乙][：:]\s*', '', text).strip()
