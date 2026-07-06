#!/usr/bin/env python3
"""
Fix tts_backend tags in output/if_data/dialogues_w*.jsonl.
Records with idx in [old_split, new_split) were tagged "indextts" but
should be "breezyvoice" given the extended total.

Usage:
    python3 scripts/retag_if_backends.py --old-total 5000 --new-total 9000
"""
import argparse
import json
from pathlib import Path

IF_OUTPUT = Path("/work/jaylin0418/TASTE2-TW-QWEN2.5-Data/output/if_data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-total", type=int, default=5000)
    parser.add_argument("--new-total", type=int, default=9000)
    args = parser.parse_args()

    old_split = int(args.old_total * 0.75)  # 3750
    new_split = int(args.new_total * 0.75)  # 6750

    print(f"Retagging idx [{old_split}, {new_split}): indextts → breezyvoice")

    total_retagged = 0
    for jsonl in sorted(IF_OUTPUT.glob("dialogues_w*.jsonl")):
        lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
        new_lines = []
        retagged = 0
        for line in lines:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Extract numeric index from ID like "if_data_003750"
            try:
                idx = int(rec["id"].split("_")[-1])
            except (KeyError, ValueError):
                new_lines.append(line)
                continue
            if old_split <= idx < new_split and rec.get("tts_backend") == "indextts":
                rec["tts_backend"] = "breezyvoice"
                retagged += 1
            new_lines.append(json.dumps(rec, ensure_ascii=False))
        jsonl.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        if retagged:
            print(f"  {jsonl.name}: retagged {retagged} records")
        total_retagged += retagged

    print(f"\nDone. Total retagged: {total_retagged}")


if __name__ == "__main__":
    main()
