#!/usr/bin/env python3
"""
Extract a zip of ref-audio wavs, transcribe with Whisper, build ref_pool.json.

Usage:
    python tts/prepare_ref_pool.py \
        --zip  ref_audio/filtered_voice-20260704T153956Z-3-001.zip \
        --out  ref_audio/pool
"""
import argparse
import json
import zipfile
from pathlib import Path

import soundfile as sf

MIN_DURATION = 2.0
MAX_DURATION = 30.0


def transcribe_wavs(wav_paths: list[Path], model_name: str) -> dict[str, str]:
    import whisper
    print(f"Loading Whisper ({model_name})...")
    model = whisper.load_model(model_name)
    results = {}
    for i, p in enumerate(wav_paths):
        print(f"  [{i+1}/{len(wav_paths)}] {p.name}", flush=True)
        try:
            result = model.transcribe(str(p), language="zh", fp16=False)
            results[p.name] = result["text"].strip()
        except Exception as e:
            print(f"    WARNING: {p.name}: {e}")
            results[p.name] = ""
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--whisper-model", default="medium")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    out_dir = Path(args.out)
    wavs_dir = out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    # Extract — flatten any subdirectory
    print(f"Extracting {zip_path.name} → {wavs_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.lower().endswith(".wav"):
                target = wavs_dir / Path(member).name
                if not target.exists():
                    target.write_bytes(zf.read(member))
    wav_files = sorted(wavs_dir.glob("*.wav"))
    print(f"Extracted {len(wav_files)} wav files")

    # Filter by duration
    valid: list[tuple[Path, float]] = []
    for p in wav_files:
        try:
            info = sf.info(str(p))
            dur = info.frames / info.samplerate
            if MIN_DURATION <= dur <= MAX_DURATION:
                valid.append((p, dur))
            else:
                print(f"  SKIP duration={dur:.1f}s: {p.name}")
        except Exception as e:
            print(f"  SKIP read error: {p.name}: {e}")
    print(f"{len(valid)}/{len(wav_files)} files pass duration filter")

    # Transcribe
    transcriptions = transcribe_wavs([p for p, _ in valid], args.whisper_model)

    # Build pool JSON (same format as cv_pool.json; wav relative to out_dir)
    pool = [
        {
            "wav": f"wavs/{p.name}",
            "sentence": transcriptions.get(p.name, ""),
            "speaker_id": p.stem,
            "duration_sec": round(dur, 2),
        }
        for p, dur in valid
    ]

    pool_json = out_dir / "ref_pool.json"
    pool_json.write_text(json.dumps(pool, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(pool)} entries → {pool_json}")


if __name__ == "__main__":
    main()
