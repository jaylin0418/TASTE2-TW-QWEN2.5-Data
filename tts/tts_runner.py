#!/usr/bin/env python3
"""
TTS runner for TASTE2-TW-QWEN2.5-Data pipeline.

Two backends (same as open_source/dialogue_v1/syn_ver2_breezy.py):
  - indextts:   IndexTTS-2 loaded in-process
                Speaker ref = Common Voice zh-TW (fixed per role per dialogue, seeded by ID)
  - breezyvoice: BreezyVoice called via subprocess (batch_inference.py CSV pattern)
                Speaker ref = Common Voice zh-TW (random per turn for maximum diversity)

Both backends use cv_pool.json built by prepare_cv_pool.py.
cv_pool.json format: [{wav: relative_path, sentence: text, ...}, ...]

Usage:
    python tts/tts_runner.py \
        --input  output/user_agent/藝術/dialogues.jsonl \
        --output tts_output/user_agent/藝術/ \
        --config conf/base.yaml \
        --indextts-dir /work/jaylin0418/cog-IndexTTS-2 \
        --cv-pool      /work/jaylin0418/common_voice_zh_TW/pool/cv_pool.json \
        --breezy-repo  /home/jaylin0418/SpeechLab/tts_model/BreezyVoice \
        --breezy-python /home/jaylin0418/miniconda3/envs/breezyvoice_py310/bin/python \
        --worker-id 0 --num-workers 1
"""
import argparse
import csv
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from tts.speed_stretch import stretch_wav

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SILENCE_SEC = 0.25
SAMPLE_RATE = 24000

# ── Common Voice pool ─────────────────────────────────────────────────────────

def load_cv_pool(cv_pool_json: str) -> list[dict]:
    """
    Load cv_pool.json built by prepare_cv_pool.py.
    Returns list of {wav: rel_path, sentence: text, wav_abs: abs_path, ...}
    """
    with open(cv_pool_json, encoding="utf-8") as f:
        pool = json.load(f)
    cv_dir = Path(cv_pool_json).parent
    for entry in pool:
        entry["wav_abs"] = str(cv_dir / entry["wav"])
    return pool


def load_eleven_lab_neutral(ref_audio_dir: Path) -> list[dict]:
    """
    Load neutral (no-emotion-prefix) files from eleven_lab_emotion.
    Returns list of {wav_abs, sentence}
    """
    _EMOTION_PREFIXES = {
        "afraid","amusement","angry","anxiety","calm","compassion","contentment",
        "cry","disappointment","disgusted","envy","excitement","frustration",
        "gratitude","grief","guilt","happy","hope","hysteria","melancholic",
        "pitch","pride","relief","sad","sarcastic","shame","surprised",
        "volume","whisper",
    }
    trans_path = ref_audio_dir / "transcriptions.json"
    if not trans_path.exists():
        return []
    with open(trans_path, encoding="utf-8") as f:
        transcriptions = json.load(f)
    pool = []
    for rel_path, text in transcriptions.items():
        prefix = Path(rel_path).stem.split("_")[0].lower()
        if prefix not in _EMOTION_PREFIXES:
            pool.append({
                "wav_abs": str(ref_audio_dir / rel_path),
                "sentence": text,
            })
    return pool


# ── IndexTTS-2 ────────────────────────────────────────────────────────────────

_indextts_model = None

def load_indextts(indextts_dir: str) -> object:
    global _indextts_model
    if _indextts_model is not None:
        return _indextts_model
    sys.path.insert(0, indextts_dir)
    from indextts import infer_v2

    # Suppress noisy QwenEmotion errors (we don't use emotion)
    original_qwen = getattr(infer_v2, "QwenEmotion", None)
    if original_qwen and not getattr(original_qwen, "_patched", False):
        class _SafeQwenEmotion:
            _patched = True
            def __init__(self, model_dir):
                try:
                    self._inner = original_qwen(model_dir)
                except Exception:
                    self._inner = None
            def inference(self, text):
                if self._inner is None:
                    return {"calm": 1.0}
                try:
                    return self._inner.inference(text)
                except Exception:
                    return {"calm": 1.0}
        infer_v2.QwenEmotion = _SafeQwenEmotion

    from indextts.infer_v2 import IndexTTS2
    ckpt_dir = str(Path(indextts_dir) / "checkpoints")
    _indextts_model = IndexTTS2(
        cfg_path=str(Path(ckpt_dir) / "config.yaml"),
        model_dir=ckpt_dir,
        use_fp16=torch.cuda.is_available(),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        use_cuda_kernel=torch.cuda.is_available(),
    )
    return _indextts_model


def synth_indextts(tts, text: str, spk_ref: str, out_path: str) -> None:
    tts.infer(
        spk_audio_prompt=spk_ref,
        text=text,
        output_path=out_path,
        emo_audio_prompt=None,
        use_random=False,
        verbose=False,
    )


# ── BreezyVoice (subprocess) ──────────────────────────────────────────────────

BREEZY_MODEL_PATH = str(
    Path("/home/jaylin0418/SpeechLab/tts_model/BreezyVoice/checkpoints/hf_cache/"
         "models--MediaTek-Research--BreezyVoice-300M/snapshots/"
         "e33b502e0ac21c16b0ee0d00df66ac3fa737393d")
)


def _ensure_wav_mono_16k(src: str, dst: str) -> None:
    wav, sr = torchaudio.load(src)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    torchaudio.save(dst, wav, 16000)


def synth_breezyvoice_batch(
    turns: list[dict],
    turn_refs: list[dict],             # per-turn: {wav_abs, sentence}
    out_dir: Path,
    breezy_repo: str,
    breezy_python: str,
    breezy_model: str = BREEZY_MODEL_PATH,
) -> list[Path | None]:
    """
    Synthesize all turns in one BreezyVoice batch_inference.py subprocess call.
    Each turn gets its own speaker (turn_refs[i]).
    Returns list of output wav paths (None if a turn failed).
    """
    import shutil
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        prompt_dir = tmp / "prompts"
        raw_out_dir = tmp / "raw_out"
        prompt_dir.mkdir()
        raw_out_dir.mkdir()

        csv_path = tmp / "batch.csv"
        row_stems: list[str] = []
        csv_rows = []
        for i, (turn, ref) in enumerate(zip(turns, turn_refs)):
            out_stem = f"turn_{i:03d}"
            spk_stem = f"spk_{i:03d}"
            # Convert ref wav to 16kHz mono in prompt_dir
            _ensure_wav_mono_16k(ref["wav_abs"], str(prompt_dir / f"{spk_stem}.wav"))
            row_stems.append(out_stem)
            csv_rows.append({
                "speaker_prompt_audio_filename": spk_stem,
                "speaker_prompt_text_transcription": ref.get("sentence", ""),
                "content_to_synthesize": turn["text"],
                "output_audio_filename": out_stem,
            })

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "speaker_prompt_audio_filename",
                "speaker_prompt_text_transcription",
                "content_to_synthesize",
                "output_audio_filename",
            ])
            writer.writeheader()
            writer.writerows(csv_rows)

        cmd = [
            breezy_python,
            "batch_inference.py",
            "--csv_file", str(csv_path),
            "--speaker_prompt_audio_folder", str(prompt_dir),
            "--output_audio_folder", str(raw_out_dir),
            "--model_path", breezy_model,
        ]
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        try:
            subprocess.run(cmd, cwd=breezy_repo, env=env, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logger.error(f"BreezyVoice subprocess failed: {e.stderr.decode()[-500:]}")
            return [None] * len(turns)

        results = []
        for stem in row_stems:
            src = raw_out_dir / f"{stem}.wav"
            if src.exists():
                dst = out_dir / f"{stem}.wav"
                shutil.copy2(str(src), str(dst))
                results.append(dst)
            else:
                results.append(None)
        return results


# ── Audio concat ──────────────────────────────────────────────────────────────

def concat_wavs(wav_paths: list[str], silence_sec: float, out_path: str,
                sr: int = SAMPLE_RATE) -> float:
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
    if not segments:
        return 0.0
    full = np.concatenate(segments[:-1])
    sf.write(out_path, full, sr)
    return len(full) / sr


# ── Language quality gate ─────────────────────────────────────────────────────

def _is_cjk(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF or   # CJK Unified Ideographs
        0x3400 <= cp <= 0x4DBF or   # CJK Extension A
        0x20000 <= cp <= 0x2A6DF or # CJK Extension B
        0xF900 <= cp <= 0xFAFF or   # CJK Compatibility
        0x2E80 <= cp <= 0x2EFF or   # CJK Radicals Supplement
        0x2F00 <= cp <= 0x2FDF or   # Kangxi Radicals
        0x3000 <= cp <= 0x303F or   # CJK Symbols and Punctuation
        0x3100 <= cp <= 0x312F or   # Bopomofo
        0x31A0 <= cp <= 0x31BF      # Bopomofo Extended
    )


def _has_foreign_script(turns: list[dict], strict: bool = False) -> bool:
    """Return True if any turn contains foreign-script text.

    strict=False (UA/DC):
      - Non-ASCII non-CJK alphabetic chars (Vietnamese, Thai, etc.): 2+ consecutive → reject.
      - ASCII Latin (English): token-level, 2+ consecutive English-dominant tokens → reject.
        Single abbreviations like GDP/App/AI are allowed.

    strict=True (IF data):
      - Any 2+ consecutive non-CJK alphabetic characters → reject.
        No abbreviation allowance; even GDP/App cause rejection.
    """
    import re as _re
    _tok_split = _re.compile(r'[\s,;.!?\"\'.。，；！？、：]+')

    for turn in turns:
        text = turn.get("text", "")

        if strict:
            # Single rule: any 2+ consecutive non-CJK alphabetic chars
            consec = 0
            for ch in text:
                cp = ord(ch)
                if unicodedata.category(ch).startswith("L") and not _is_cjk(cp):
                    consec += 1
                    if consec >= 2:
                        return True
                else:
                    consec = 0
        else:
            # Rule 1: non-ASCII non-CJK chars (Vietnamese, Thai, Arabic, kana, etc.)
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

            # Rule 2: ASCII Latin — word-token level (allows single abbreviations)
            consec = 0
            for tok in _tok_split.split(text):
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


# ── Per-dialogue processing ───────────────────────────────────────────────────

INDEXTTS_CV_RATIO = 0.30   # 30% Common Voice, 70% eleven_lab for IndexTTS-2


def process_dialogue(
    record: dict,
    out_dir: Path,
    cfg: dict,
    cv_pool: list[dict],              # Common Voice pool [{wav_abs, sentence, ...}]
    eleven_lab_pool: list[dict],      # eleven_lab neutral pool [{wav_abs, sentence}]
    tts_model,                         # IndexTTS2 object or None
    breezy_repo: str,
    breezy_python: str,
    breezy_model: str,
    fast_factor: float = 0.77,
    slow_factor: float = 1.33,
) -> dict | None:
    dlg_id = record["id"]
    backend = record.get("tts_backend", "indextts")
    turns = record.get("turns", [])
    if not turns:
        return None

    dtype = record.get("type", "")
    if _has_foreign_script(turns, strict=(dtype == "if_data")):
        logger.debug(f"[{dlg_id}] skipped: foreign script detected")
        return None

    dlg_out_dir = out_dir / dlg_id
    dlg_out_dir.mkdir(parents=True, exist_ok=True)
    done_flag = dlg_out_dir / "done.flag"
    if done_flag.exists():
        meta_path = dlg_out_dir / "meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return None

    rng = random.Random(dlg_id)
    roles = sorted({t["role"] for t in turns})

    # Synthesize
    turn_wav_paths: list[Path | None] = []
    turn_ref_list: list[dict] = []  # per-turn ref audio info

    if backend == "indextts":
        if tts_model is None:
            logger.error("IndexTTS-2 model not loaded")
            return None
        # IndexTTS-2: fixed speaker per role, 70% eleven_lab / 30% Common Voice
        role_refs: dict[str, dict] = {}
        for role in roles:
            if rng.random() < INDEXTTS_CV_RATIO and cv_pool:
                role_refs[role] = rng.choice(cv_pool)
            else:
                role_refs[role] = rng.choice(eleven_lab_pool) if eleven_lab_pool else rng.choice(cv_pool)

        for i, turn in enumerate(turns):
            role = turn["role"]
            turn_wav = dlg_out_dir / f"turn_{i:03d}.wav"
            if not turn_wav.exists():
                try:
                    synth_indextts(tts_model, turn["text"],
                                   role_refs[role]["wav_abs"], str(turn_wav))
                except Exception as e:
                    logger.error(f"{dlg_id} turn {i}: {e}")
                    return None
            turn_wav_paths.append(turn_wav)
            turn_ref_list.append(role_refs[role])

    else:  # breezyvoice — fixed speaker per role for the whole dialogue
        breezy_role_refs: dict[str, dict] = {role: rng.choice(cv_pool) for role in roles}
        turn_refs = [breezy_role_refs[turn["role"]] for turn in turns]
        raw_paths = synth_breezyvoice_batch(
            turns, turn_refs,
            dlg_out_dir, breezy_repo, breezy_python, breezy_model,
        )
        if any(p is None for p in raw_paths):
            logger.warning(f"{dlg_id}: some BreezyVoice turns failed")
        turn_wav_paths = raw_paths
        turn_ref_list = turn_refs

    # Apply speed stretching per turn, then compute timestamps
    turn_meta = []
    turn_wavs = []
    timestamp = 0.0

    for i, (turn, turn_wav) in enumerate(zip(turns, turn_wav_paths)):
        if turn_wav is None or not turn_wav.exists():
            logger.warning(f"{dlg_id} turn {i}: missing wav, skipping")
            continue

        # Speed stretch for fast/slow turns (type4 if_control / type5 speed_ua Agent turns)
        spd = (turn.get("speed") or "normal").strip()
        if spd in ("fast", "slow"):
            stretched_path = dlg_out_dir / f"turn_{i:03d}_stretched.wav"
            if not stretched_path.exists():
                stretch_wav(str(turn_wav), str(stretched_path), spd,
                            fast_factor, slow_factor)
            effective_wav = stretched_path
        else:
            effective_wav = turn_wav

        ref = turn_ref_list[i] if i < len(turn_ref_list) else {}
        audio, audio_sr = sf.read(str(effective_wav), dtype="float32")
        duration = len(audio) / audio_sr
        turn_meta.append({
            **turn,
            "ref_wav": ref.get("wav", ref.get("wav_abs", "")),
            "ref_speaker_id": ref.get("speaker_id", ""),
            "wav": str(effective_wav.relative_to(out_dir.parent)),
            "timestamp_start": round(timestamp, 3),
            "timestamp_end": round(timestamp + duration, 3),
        })
        turn_wavs.append(str(effective_wav))
        timestamp += duration + SILENCE_SEC

    if not turn_wavs:
        return None

    full_wav = dlg_out_dir / "full.wav"
    total_dur = concat_wavs(turn_wavs, SILENCE_SEC, str(full_wav), SAMPLE_RATE)

    result = {
        **record,
        "turns": turn_meta,
        "full_wav": str(full_wav.relative_to(out_dir.parent)),
        "total_duration_sec": round(total_dur, 2),
    }
    (dlg_out_dir / "meta.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    done_flag.touch()
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def _slice_records(records: list, worker_id: int, num_workers: int) -> list:
    # Interleaved assignment so remainder is spread across all workers evenly
    return records[worker_id::num_workers]


def _wait_for_flag(flag_path: Path, max_wait: int, label: str) -> bool:
    import time
    waited = 0
    while not flag_path.exists():
        if waited >= max_wait:
            logger.warning(f"Timeout waiting for {label} after {max_wait}s")
            return False
        logger.info(f"[worker] waiting for {label} ({waited}s)")
        time.sleep(30)
        waited += 30
    return True


def _process_topic(topic_input: Path, topic_output: Path, cfg: dict,
                   cv_pool, eleven_lab_pool, tts_model_ref: list,
                   breezy_repo, breezy_python, breezy_model,
                   fast_factor, slow_factor,
                   worker_id: int, num_workers: int,
                   indextts_dir: str,
                   extra_inputs: list = None) -> None:
    # Collect records from topic_input and any extra shard files
    all_files = [topic_input] + (extra_inputs or [])
    records = []
    for f in all_files:
        for l in Path(f).read_text(encoding="utf-8").splitlines():
            if l.strip():
                records.append(json.loads(l))
    my_records = _slice_records(records, worker_id, num_workers)
    if not my_records:
        return

    # Lazy-load IndexTTS on first encounter
    if tts_model_ref[0] is None:
        if any(r.get("tts_backend", "indextts") == "indextts" for r in my_records):
            logger.info("Loading IndexTTS-2 (first indextts dialogue)...")
            tts_model_ref[0] = load_indextts(indextts_dir)
            logger.info("IndexTTS-2 ready.")

    topic_output.mkdir(parents=True, exist_ok=True)
    out_jsonl = topic_output / f"tts_output_w{worker_id:04d}.jsonl"
    done_ids = set()
    if out_jsonl.exists():
        for l in out_jsonl.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try:
                    done_ids.add(json.loads(l)["id"])
                except Exception:
                    pass

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for i, record in enumerate(my_records):
            if record.get("id") in done_ids:
                continue
            result = process_dialogue(
                record, topic_output, cfg,
                cv_pool, eleven_lab_pool, tts_model_ref[0],
                breezy_repo, breezy_python, breezy_model,
                fast_factor, slow_factor,
            )
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
            if i % 20 == 0:
                logger.info(f"Worker {worker_id} [{topic_output.name}]: {i}/{len(my_records)} done")


def main():
    parser = argparse.ArgumentParser()
    # ── Mode A: single topic (original, used by test_tts.job) ──────────────
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    # ── Mode B: persistent worker — all topics, model loaded once ──────────
    parser.add_argument("--input-base", default=None,
                        help="Base dir containing per-topic subdirs with dialogues.jsonl")
    parser.add_argument("--output-base", default=None,
                        help="Base dir for per-topic TTS output")
    parser.add_argument("--topics", nargs="+", default=None,
                        help="Ordered list of topics this worker should process")
    parser.add_argument("--max-wait-secs", type=int, default=86400)
    # ── Common ──────────────────────────────────────────────────────────────
    parser.add_argument("--config", default="conf/base.yaml")
    parser.add_argument("--indextts-dir", default="/work/jaylin0418/cog-IndexTTS-2")
    parser.add_argument("--cv-pool",
                        default="/work/jaylin0418/common_voice_zh_TW/pool/cv_pool.json")
    parser.add_argument("--ref-audio-dir",
                        default="/home/jaylin0418/SpeechLab/ref_audio/eleven_lab_emotion")
    parser.add_argument("--ref-pool", default=None)
    parser.add_argument("--breezy-repo",
                        default="/home/jaylin0418/SpeechLab/tts_model/BreezyVoice")
    parser.add_argument("--breezy-python",
                        default="/home/jaylin0418/miniconda3/envs/breezyvoice_py310/bin/python")
    parser.add_argument("--breezy-model", default=BREEZY_MODEL_PATH)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    speed_cfg = cfg.get("speed", {})
    fast_factor = speed_cfg.get("fast_factor", 0.77)
    slow_factor = speed_cfg.get("slow_factor", 1.33)

    if args.ref_pool:
        cv_pool = load_cv_pool(args.ref_pool)
        eleven_lab_pool = cv_pool
        logger.info(f"Custom ref pool: {len(cv_pool)} files")
    else:
        cv_pool = load_cv_pool(args.cv_pool)
        eleven_lab_pool = load_eleven_lab_neutral(Path(args.ref_audio_dir))

    tts_model_ref = [None]  # mutable ref for lazy load

    if args.input_base and args.topics:
        # ── Mode B: persistent worker ──────────────────────────────────────
        logger.info(f"Worker {args.worker_id}/{args.num_workers}: persistent mode, {len(args.topics)} topics")
        input_base = Path(args.input_base)
        output_base = Path(args.output_base)
        for topic in args.topics:
            done_flag = input_base / topic / "done.flag"
            if not _wait_for_flag(done_flag, args.max_wait_secs, f"{topic}/done.flag"):
                continue
            # Support sharded gen: read all dialogues*.jsonl files
            import glob as _glob
            dialogue_files = sorted(_glob.glob(str(input_base / topic / "dialogues*.jsonl")))
            if not dialogue_files:
                logger.warning(f"No dialogues*.jsonl in {input_base / topic}, skipping")
                continue
            input_file = Path(dialogue_files[0])
            extra_inputs = [Path(f) for f in dialogue_files[1:]]
            logger.info(f"Worker {args.worker_id}: START {topic}")
            _process_topic(
                input_file, output_base / topic, cfg,
                cv_pool, eleven_lab_pool, tts_model_ref,
                args.breezy_repo, args.breezy_python, args.breezy_model,
                fast_factor, slow_factor,
                args.worker_id, args.num_workers,
                args.indextts_dir,
                extra_inputs=extra_inputs,
            )
            logger.info(f"Worker {args.worker_id}: DONE  {topic}")
        logger.info(f"Worker {args.worker_id}: all topics finished.")

    else:
        # ── Mode A: single topic ───────────────────────────────────────────
        if not args.input or not args.output:
            raise ValueError("Either --input/--output (mode A) or --input-base/--output-base/--topics (mode B) required")

        records = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]
        my_records = _slice_records(records, args.worker_id, args.num_workers)
        logger.info(f"Worker {args.worker_id}: {len(my_records)} dialogues (mode A)")

        if any(r.get("tts_backend") == "indextts" for r in my_records):
            logger.info("Loading IndexTTS-2...")
            tts_model_ref[0] = load_indextts(args.indextts_dir)

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = out_dir / f"tts_output_w{args.worker_id:02d}.jsonl"
        with open(out_jsonl, "a", encoding="utf-8") as fout:
            for i, record in enumerate(my_records):
                result = process_dialogue(
                    record, out_dir, cfg,
                    cv_pool, eleven_lab_pool, tts_model_ref[0],
                    args.breezy_repo, args.breezy_python, args.breezy_model,
                    fast_factor, slow_factor,
                )
                if result:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                if i % 20 == 0:
                    logger.info(f"Worker {args.worker_id}: {i}/{len(my_records)} done")
        logger.info(f"Worker {args.worker_id}: finished.")


if __name__ == "__main__":
    main()
