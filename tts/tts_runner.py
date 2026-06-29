#!/usr/bin/env python3
"""
Unified TTS runner for TASTE2-TW-QWEN2.5-Data pipeline.

Reads JSONL dialogue files, synthesizes each turn, applies speed stretching,
and writes per-dialogue WAV files + updated JSONL with audio metadata.

Backend selection:
  - "indextts":   IndexTTS-2, one fixed speaker per role per dialogue
  - "breezyvoice": BreezyVoice, random Common Voice zh-TW speaker per turn

Usage:
    python tts/tts_runner.py \
        --input  output/user_agent/藝術/dialogues.jsonl \
        --output tts_output/user_agent/藝術/ \
        --config conf/base.yaml \
        --indextts-dir /work/jaylin0418/cog-IndexTTS-2 \
        --common-voice-dir /work/jaylin0418/common_voice_zh_TW \
        --worker-id 0 --num-workers 1
"""
import argparse
import json
import logging
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from tts.speed_stretch import stretch_wav

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SILENCE_SEC = 0.25  # gap between turns


# ── Common Voice speaker pool ─────────────────────────────────────────────────

def load_cv_speakers(cv_dir: str) -> list[str]:
    """Return list of .wav paths from Common Voice clips/."""
    clips_dir = Path(cv_dir) / "clips"
    wavs = sorted(clips_dir.glob("*.wav"))
    if not wavs:
        # Also accept mp3; convert on the fly later
        mp3s = sorted(clips_dir.glob("*.mp3"))
        return [str(p) for p in mp3s]
    return [str(p) for p in wavs]


# ── IndexTTS-2 backend ────────────────────────────────────────────────────────

_indextts_model = None

def load_indextts(model_dir: str, repo_dir: str):
    global _indextts_model
    if _indextts_model is not None:
        return _indextts_model
    sys.path.insert(0, repo_dir)
    from indextts.infer_v2 import IndexTTS2
    _indextts_model = IndexTTS2(model_dir=model_dir)
    return _indextts_model


def synth_indextts(model, text: str, ref_wav: str, out_path: str, sr: int = 24000):
    model.infer(audio_prompt=ref_wav, text=text, output_path=out_path)


# ── BreezyVoice backend ───────────────────────────────────────────────────────

_breezy_model = None

def load_breezyvoice(nano4_dir: str):
    global _breezy_model
    if _breezy_model is not None:
        return _breezy_model
    sys.path.insert(0, nano4_dir)
    from cosyvoice.cli.cosyvoice import CosyVoice2
    _breezy_model = CosyVoice2("pretrained_models/CosyVoice2-0.5B",
                                load_jit=False, load_trt=False)
    return _breezy_model


def synth_breezyvoice(model, text: str, ref_wav: str, out_path: str, sr: int = 24000):
    import torchaudio
    import torch
    prompt_speech, prompt_sr = torchaudio.load(ref_wav)
    if prompt_sr != 16000:
        prompt_speech = torchaudio.functional.resample(prompt_speech, prompt_sr, 16000)
    for chunk in model.inference_zero_shot(
        text, "", prompt_speech_16k=prompt_speech[0], stream=False
    ):
        audio = chunk["tts_speech"].squeeze().numpy()
        sf.write(out_path, audio, sr)
        break


# ── Mix turns into single dialogue WAV ───────────────────────────────────────

def concat_wavs(wav_paths: list[str], silence_sec: float, out_path: str, sr: int = 24000):
    silence = np.zeros(int(silence_sec * sr), dtype=np.float32)
    segments = []
    for p in wav_paths:
        audio, audio_sr = sf.read(p, dtype="float32")
        if audio_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=audio_sr, target_sr=sr)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        segments.append(audio)
        segments.append(silence)
    full = np.concatenate(segments[:-1]) if segments else np.array([], dtype=np.float32)
    sf.write(out_path, full, sr)
    return len(full) / sr  # total duration in seconds


# ── Main synthesis loop ───────────────────────────────────────────────────────

def process_dialogue(record: dict, out_dir: Path, cfg: dict,
                     cv_speakers: list[str], indextts_model, breezy_model,
                     ref_audio_dir: Path, sr: int = 24000) -> dict | None:
    dlg_id = record["id"]
    backend = record.get("tts_backend", "indextts")
    turns = record.get("turns", [])
    if not turns:
        return None

    dlg_out_dir = out_dir / dlg_id
    dlg_out_dir.mkdir(parents=True, exist_ok=True)
    done_flag = dlg_out_dir / "done.flag"
    if done_flag.exists():
        # Load existing metadata
        meta_path = dlg_out_dir / "meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return None

    # Assign fixed speakers per role for IndexTTS-2
    roles = list({t["role"] for t in turns})
    ref_wavs = {}
    if backend == "indextts":
        ref_dir = ref_audio_dir / "indextts"
        ref_dir.mkdir(parents=True, exist_ok=True)
        for role in roles:
            role_refs = sorted(ref_dir.glob(f"{role}_*.wav"))
            if not role_refs:
                logger.warning(f"No ref audio for role={role} in {ref_dir}")
                return None
            ref_wavs[role] = str(random.choice(role_refs))

    turn_wavs = []
    turn_meta = []
    timestamp = 0.0

    for i, turn in enumerate(turns):
        role = turn["role"]
        text = turn["text"]
        speed = turn.get("speed", "normal")
        turn_wav = dlg_out_dir / f"turn_{i:03d}_{role}.wav"
        stretched_wav = dlg_out_dir / f"turn_{i:03d}_{role}_stretched.wav"

        # Synthesize
        if not turn_wav.exists():
            try:
                if backend == "indextts":
                    synth_indextts(indextts_model, text, ref_wavs[role], str(turn_wav), sr)
                else:
                    ref = random.choice(cv_speakers)
                    synth_breezyvoice(breezy_model, text, ref, str(turn_wav), sr)
            except Exception as e:
                logger.error(f"TTS failed for {dlg_id} turn {i}: {e}")
                return None

        # Apply speed stretch
        if not stretched_wav.exists():
            stretch_wav(str(turn_wav), str(stretched_wav), speed,
                        fast_factor=cfg.get("speed", {}).get("fast_factor", 0.77),
                        slow_factor=cfg.get("speed", {}).get("slow_factor", 1.33))

        audio, audio_sr = sf.read(str(stretched_wav), dtype="float32")
        duration = len(audio) / audio_sr
        turn_meta.append({
            **turn,
            "wav": str(stretched_wav.relative_to(out_dir.parent)),
            "timestamp_start": round(timestamp, 3),
            "timestamp_end": round(timestamp + duration, 3),
        })
        timestamp += duration + SILENCE_SEC
        turn_wavs.append(str(stretched_wav))

    # Concat full dialogue WAV
    full_wav = dlg_out_dir / "full.wav"
    total_dur = concat_wavs(turn_wavs, SILENCE_SEC, str(full_wav), sr)

    result = {**record, "turns": turn_meta,
              "full_wav": str(full_wav.relative_to(out_dir.parent)),
              "total_duration_sec": round(total_dur, 2)}
    (dlg_out_dir / "meta.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2))
    done_flag.touch()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", default="conf/base.yaml")
    parser.add_argument("--indextts-dir", default=None)
    parser.add_argument("--nano4-dir", default=None, help="BreezyVoice Nano4 dir")
    parser.add_argument("--common-voice-dir", default=None)
    parser.add_argument("--ref-audio-dir", default=None)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tts_cfg = cfg.get("tts", {})
    indextts_dir = args.indextts_dir or tts_cfg.get("indextts_repo_dir")
    indextts_model_dir = tts_cfg.get("indextts_model_dir")
    nano4_dir = args.nano4_dir
    cv_dir = args.common_voice_dir or tts_cfg.get("breezyvoice_common_voice_dir")
    ref_audio_dir = Path(args.ref_audio_dir or tts_cfg.get("ref_audio_dir", "ref_audio"))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = tts_cfg.get("sample_rate", 24000)

    # Load speaker pool
    cv_speakers = load_cv_speakers(cv_dir) if cv_dir else []

    # Read all records from JSONL
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Worker slice
    per_worker = len(records) // args.num_workers
    start = args.worker_id * per_worker
    end = start + per_worker if args.worker_id < args.num_workers - 1 else len(records)
    my_records = records[start:end]
    logger.info(f"Worker {args.worker_id}: processing {len(my_records)} dialogues")

    # Determine which backends are needed
    needs_indextts = any(r.get("tts_backend") == "indextts" for r in my_records)
    needs_breezy = any(r.get("tts_backend") == "breezyvoice" for r in my_records)

    indextts_model = None
    if needs_indextts and indextts_dir and indextts_model_dir:
        logger.info("Loading IndexTTS-2...")
        indextts_model = load_indextts(indextts_model_dir, indextts_dir)

    breezy_model = None
    if needs_breezy and nano4_dir:
        logger.info("Loading BreezyVoice...")
        breezy_model = load_breezyvoice(nano4_dir)

    out_jsonl = out_dir / f"tts_output_w{args.worker_id:02d}.jsonl"
    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for i, record in enumerate(my_records):
            result = process_dialogue(record, out_dir, cfg, cv_speakers,
                                      indextts_model, breezy_model, ref_audio_dir, sr)
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
            if i % 20 == 0:
                logger.info(f"Worker {args.worker_id}: {i}/{len(my_records)} done")

    logger.info(f"Worker {args.worker_id}: finished.")


if __name__ == "__main__":
    main()
