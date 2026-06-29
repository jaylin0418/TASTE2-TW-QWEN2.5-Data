#!/usr/bin/env python3
"""
Convert TTS output JSONL → Parquet for TASTE2 SFT training.

Schema per row:
  idx            str       unique dialogue id
  type           str       data type (user_agent / daily_conv / if_data / speed_ua / speed_daily / if_control)
  topic          str       topic (empty for IF types)
  system_prompt  str       system prompt (empty for IF types)
  meta           struct    {data_type, tts_backend, if_tasks, speed_sequence}
  message        list      [{role, text, audio (bytes), timestamp_range [start_ms, end_ms], speed}]

Usage:
    python to_parquet/to_parquet_v5.py \
        --input  tts_output/user_agent/ \
        --output parquet/user_agent/ \
        --prefix ua \
        --chunk-gb 1.0
"""
import argparse
import glob
import json
import logging
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Arrow schema ──────────────────────────────────────────────────────────────

MESSAGE_STRUCT = pa.list_(pa.struct({
    "role":            pa.string(),
    "text":            pa.string(),
    "audio":           pa.large_binary(),
    "timestamp_range": pa.list_(pa.int64()),   # [start_ms, end_ms]
    "speed":           pa.string(),
}))

META_STRUCT = pa.struct({
    "data_type":       pa.string(),
    "tts_backend":     pa.string(),
    "if_tasks":        pa.list_(pa.string()),
    "speed_sequence":  pa.list_(pa.string()),
})

SCHEMA = pa.schema([
    ("idx",           pa.string()),
    ("type",          pa.string()),
    ("topic",         pa.string()),
    ("system_prompt", pa.string()),
    ("meta",          META_STRUCT),
    ("message",       MESSAGE_STRUCT),
])


def read_wav_bytes(wav_path: str) -> bytes | None:
    """Read a WAV file and return raw bytes (as stored by soundfile)."""
    if not os.path.exists(wav_path):
        return None
    with open(wav_path, "rb") as f:
        return f.read()


def record_to_row(record: dict, tts_base: Path) -> dict | None:
    turns = record.get("turns", [])
    if not turns:
        return None

    messages = []
    for turn in turns:
        wav_rel = turn.get("wav", "")
        wav_path = tts_base / wav_rel if wav_rel else None
        audio_bytes = read_wav_bytes(str(wav_path)) if wav_path else None
        if audio_bytes is None:
            logger.warning(f"Missing WAV: {wav_path}")
            continue

        start_ms = int(turn.get("timestamp_start", 0) * 1000)
        end_ms = int(turn.get("timestamp_end", 0) * 1000)
        messages.append({
            "role":            turn.get("role", ""),
            "text":            turn.get("text", ""),
            "audio":           audio_bytes,
            "timestamp_range": [start_ms, end_ms],
            "speed":           turn.get("speed", "normal"),
        })

    if not messages:
        return None

    return {
        "idx":           record.get("id", ""),
        "type":          record.get("type", ""),
        "topic":         record.get("topic", ""),
        "system_prompt": record.get("system_prompt", ""),
        "meta": {
            "data_type":      record.get("type", ""),
            "tts_backend":    record.get("tts_backend", ""),
            "if_tasks":       record.get("if_tasks", []),
            "speed_sequence": record.get("speed_sequence", []),
        },
        "message": messages,
    }


def rows_to_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=SCHEMA)


def estimate_size_gb(rows: list[dict]) -> float:
    total = sum(
        sum(len(m["audio"]) for m in r["message"]) for r in rows
    )
    return total / 1e9


def convert(input_dir: str, output_dir: str, prefix: str, chunk_gb: float):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all tts_output JSONL files
    jsonl_files = sorted(input_path.rglob("tts_output_w*.jsonl"))
    if not jsonl_files:
        # Also accept meta.json per-dialogue
        jsonl_files = sorted(input_path.rglob("meta.json"))

    logger.info(f"Found {len(jsonl_files)} JSONL files")

    rows_buffer: list[dict] = []
    part_idx = 0
    tts_base = input_path.parent

    def flush(rows: list[dict], idx: int):
        if not rows:
            return idx
        out_path = output_path / f"{prefix}_part_{idx:04d}.parquet"
        table = rows_to_table(rows)
        pq.write_table(table, str(out_path), row_group_size=200,
                       compression="snappy")
        logger.info(f"Written part {idx:04d}: {len(rows)} rows → {out_path}")
        return idx + 1

    for jf in jsonl_files:
        if jf.name == "meta.json":
            with open(jf) as f:
                record = json.load(f)
            row = record_to_row(record, tts_base)
            if row:
                rows_buffer.append(row)
        else:
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    row = record_to_row(record, tts_base)
                    if row:
                        rows_buffer.append(row)

        # Flush when buffer exceeds chunk_gb
        if estimate_size_gb(rows_buffer) >= chunk_gb:
            part_idx = flush(rows_buffer, part_idx)
            rows_buffer = []

    part_idx = flush(rows_buffer, part_idx)
    logger.info(f"Done. Total parts: {part_idx}")


def main():
    parser = argparse.ArgumentParser(description="Convert TTS output to Parquet")
    parser.add_argument("--input", required=True, help="TTS output directory")
    parser.add_argument("--output", required=True, help="Parquet output directory")
    parser.add_argument("--prefix", default="v5", help="Output filename prefix")
    parser.add_argument("--chunk-gb", type=float, default=1.0,
                        help="Max GB per Parquet file")
    args = parser.parse_args()
    convert(args.input, args.output, args.prefix, args.chunk_gb)


if __name__ == "__main__":
    main()
