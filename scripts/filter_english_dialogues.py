#!/usr/bin/env python3
"""
Delete dialogue directories whose turns contain foreign-script text.

Rules (mirrors tts_runner.py _has_foreign_script):
  UA / DC (strict=False):
    - Non-ASCII non-CJK alphabetic chars (Vietnamese, Thai, Arabic, kana…):
      2+ consecutive → reject.
    - ASCII Latin (English): token-level, 2+ consecutive English-dominant tokens → reject.
      Single abbreviations like GDP/App/AI are allowed.
  IF data (strict=True):
    - Any 2+ consecutive non-CJK alphabetic characters → reject.
      Abbreviations like GDP/App also cause rejection.

Usage:
    python3 filter_english_dialogues.py [--dry-run] [--type user_agent|daily_conv|if_data|all]
"""
import argparse
import json
import re
import shutil
import unicodedata
from pathlib import Path

TTS_ROOT = Path("/work/jaylin0418/TASTE2-TW-QWEN2.5-Data/tts_output")
TYPES = ["user_agent", "daily_conv", "if_data"]

_TOK_SPLIT = re.compile(r'[\s,;.!?\"\'.。，；！？、：]+')


def _is_cjk(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF or
        0x3400 <= cp <= 0x4DBF or
        0x20000 <= cp <= 0x2A6DF or
        0xF900 <= cp <= 0xFAFF or
        0x2E80 <= cp <= 0x2EFF or
        0x2F00 <= cp <= 0x2FDF or
        0x3000 <= cp <= 0x303F or
        0x3100 <= cp <= 0x312F or
        0x31A0 <= cp <= 0x31BF
    )


def has_foreign_script(text: str, strict: bool = False) -> bool:
    if strict:
        consec = 0
        for ch in text:
            cp = ord(ch)
            if unicodedata.category(ch).startswith("L") and not _is_cjk(cp):
                consec += 1
                if consec >= 2:
                    return True
            else:
                consec = 0
        return False

    # Rule 1: non-ASCII non-CJK (Vietnamese, Thai, etc.)
    consec = 0
    for ch in text:
        cp = ord(ch)
        if (unicodedata.category(ch).startswith("L")
                and not _is_cjk(cp) and not ch.isascii()):
            consec += 1
            if consec >= 2:
                return True
        else:
            consec = 0

    # Rule 2: ASCII Latin — word-token level
    consec = 0
    for tok in _TOK_SPLIT.split(text):
        if not tok:
            continue
        ascii_alpha = sum(1 for c in tok if c.isascii() and c.isalpha())
        if ascii_alpha >= 2 and ascii_alpha / max(len(tok), 1) > 0.6:
            consec += 1
            if consec >= 2:
                return True
        else:
            consec = 0

    return False


def dialogue_has_foreign(d: dict, strict: bool) -> tuple[bool, str]:
    """Return (should_delete, example_bad_text)."""
    for t in d.get("turns", []):
        text = t.get("text", "")
        if has_foreign_script(text, strict=strict):
            return True, text[:80]
    return False, ""


def process_type(data_type: str, dry_run: bool):
    base = TTS_ROOT / data_type
    if not base.exists():
        print(f"[SKIP] {base} not found")
        return

    strict = (data_type == "if_data")
    deleted_dlg = 0
    deleted_sec = 0.0
    kept_dlg = 0
    kept_sec = 0.0

    for meta_path in sorted(base.rglob("meta.json")):
        dlg_dir = meta_path.parent
        try:
            d = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Cannot read {meta_path}: {e}")
            continue

        dur = d.get("total_duration_sec", 0.0)
        bad, example = dialogue_has_foreign(d, strict=strict)

        if bad:
            deleted_dlg += 1
            deleted_sec += dur
            if dry_run:
                print(f"[DRY-RUN] would delete {dlg_dir.name}  ({dur:.1f}s)  eg: {example}")
            else:
                shutil.rmtree(dlg_dir)
                print(f"[DELETE] {dlg_dir.name}  ({dur:.1f}s)  eg: {example}")
        else:
            kept_dlg += 1
            kept_sec += dur

    tag = "[DRY-RUN]" if dry_run else "[DONE]"
    print(f"\n{tag} {data_type} (strict={strict}):")
    print(f"  deleted : {deleted_dlg} dlg = {deleted_sec/3600:.1f}h")
    print(f"  kept    : {kept_dlg} dlg = {kept_sec/3600:.1f}h")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--type", default="all",
                        choices=["user_agent", "daily_conv", "if_data", "all"])
    args = parser.parse_args()

    types = TYPES if args.type == "all" else [args.type]
    for t in types:
        process_type(t, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
