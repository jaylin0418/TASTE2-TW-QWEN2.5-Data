#!/usr/bin/env python3
"""
Extract one WAV per speaker from Common Voice zh-TW parquet files.

Writes to:
  {out_dir}/wavs/{idx:04d}.wav          16kHz mono WAV
  {out_dir}/cv_pool.json                [{wav: rel_path, sentence: text}, ...]

Usage:
    python tts/prepare_cv_pool.py \
        --cv-dir /work/jaylin0418/common_voice_zh_TW \
        --out-dir /work/jaylin0418/common_voice_zh_TW/pool \
        --min-sec 1.5 --max-sec 10.0
"""
import argparse
import glob
import io
import json
import logging
import random
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import numpy as np
import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_SR = 16000


def bytes_to_wav_tensor(audio_bytes: bytes) -> tuple[torch.Tensor, int]:
    """Decode audio bytes to (1, T) tensor."""
    buf = io.BytesIO(audio_bytes)
    wav, sr = torchaudio.load(buf)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def resample_tensor(wav: torch.Tensor, src_sr: int, tgt_sr: int) -> torch.Tensor:
    if src_sr == tgt_sr:
        return wav
    return torchaudio.functional.resample(wav, src_sr, tgt_sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-dir", default="/work/jaylin0418/common_voice_zh_TW")
    parser.add_argument("--out-dir", default="/work/jaylin0418/common_voice_zh_TW/pool")
    parser.add_argument("--min-sec", type=float, default=1.5,
                        help="Minimum audio duration to accept")
    parser.add_argument("--max-sec", type=float, default=10.0,
                        help="Maximum audio duration to accept")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(
        glob.glob(str(Path(args.cv_dir) / "data" / "validated_without_test-*.parquet"))
    )
    if not parquet_files:
        parquet_files = sorted(
            glob.glob(str(Path(args.cv_dir) / "data" / "*.parquet"))
        )
    logger.info(f"Reading {len(parquet_files)} parquet files")

    # Collect rows per speaker
    speaker_rows: dict[str, list[dict]] = {}
    for pf in parquet_files:
        table = pq.read_table(pf, columns=["client_id", "audio", "sentence",
                                            "up_votes", "down_votes"])
        for row in table.to_pylist():
            cid = row["client_id"]
            # Skip low-quality rows
            if row["up_votes"] < row["down_votes"]:
                continue
            if not row["sentence"] or not row["audio"]["bytes"]:
                continue
            speaker_rows.setdefault(cid, []).append(row)
    logger.info(f"Speakers with valid rows: {len(speaker_rows)}")

    pool = []
    for idx, (cid, rows) in enumerate(speaker_rows.items()):
        random.shuffle(rows)
        saved = False
        for row in rows:
            try:
                wav, sr = bytes_to_wav_tensor(row["audio"]["bytes"])
                dur = wav.shape[1] / sr
                if dur < args.min_sec or dur > args.max_sec:
                    continue
                wav_16k = resample_tensor(wav, sr, TARGET_SR)
                wav_path = wav_dir / f"{idx:04d}.wav"
                torchaudio.save(str(wav_path), wav_16k, TARGET_SR)
                pool.append({
                    "wav": str(wav_path.relative_to(out_dir)),
                    "sentence": row["sentence"],
                    "speaker_id": cid[:16],
                    "duration_sec": round(dur, 2),
                })
                saved = True
                break
            except Exception as e:
                logger.warning(f"Speaker {cid[:8]} row failed: {e}")
                continue
        if not saved:
            logger.warning(f"No valid audio for speaker {cid[:8]}")
        if (idx + 1) % 50 == 0:
            logger.info(f"  Processed {idx+1}/{len(speaker_rows)} speakers")

    pool_json = out_dir / "cv_pool.json"
    with open(pool_json, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(pool)} speakers → {pool_json}")


if __name__ == "__main__":
    main()
