"""
Time-stretching for speed control via audiostretchy (file-based API).
  speed: fast   → ratio < 1 (shorten = speed up)
  speed: slow   → ratio > 1 (lengthen = slow down)
  speed: normal → copy as-is
"""
import shutil

FAST_FACTOR = 0.77
SLOW_FACTOR = 1.33


def stretch_wav(in_path: str, out_path: str, speed: str,
                fast_factor: float = FAST_FACTOR,
                slow_factor: float = SLOW_FACTOR) -> None:
    if speed == "normal":
        shutil.copy2(in_path, out_path)
        return

    ratio = fast_factor if speed == "fast" else slow_factor
    try:
        from audiostretchy.stretch import stretch_audio
        stretch_audio(in_path, out_path, ratio=ratio)
    except (ImportError, Exception):
        # Fallback: librosa time-stretch
        import librosa
        import soundfile as sf
        audio, sr = sf.read(in_path, dtype="float32")
        stretched = librosa.effects.time_stretch(audio, rate=1.0 / ratio)
        sf.write(out_path, stretched, sr)
