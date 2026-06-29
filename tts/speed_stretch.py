"""
Time-stretching for speed control using audiostretchy.
  speed: fast  → factor < 1 (shorten = speed up)
  speed: slow  → factor > 1 (lengthen = slow down)
  speed: normal → no stretching
"""
import numpy as np
import soundfile as sf


FAST_FACTOR = 0.77
SLOW_FACTOR = 1.33


def stretch_audio(audio: np.ndarray, sr: int, speed: str) -> np.ndarray:
    """Apply time-stretching based on speed label. Returns stretched audio."""
    if speed == "normal":
        return audio
    factor = FAST_FACTOR if speed == "fast" else SLOW_FACTOR
    try:
        from audiostretchy.stretch import stretch_array
        stretched = stretch_array(audio, sr, ratio=factor)
        return stretched
    except ImportError:
        # Fallback: naive resampling (low quality, only for dev)
        import librosa
        target_len = int(len(audio) * factor)
        stretched = librosa.effects.time_stretch(audio.astype(np.float32),
                                                  rate=1.0 / factor)
        return stretched


def stretch_wav(in_path: str, out_path: str, speed: str,
                fast_factor: float = FAST_FACTOR,
                slow_factor: float = SLOW_FACTOR) -> None:
    """Read a WAV, apply speed stretch, write to out_path."""
    if speed == "normal":
        import shutil
        shutil.copy2(in_path, out_path)
        return

    audio, sr = sf.read(in_path, dtype="float32")
    factor = fast_factor if speed == "fast" else slow_factor
    try:
        from audiostretchy.stretch import stretch_array
        stretched = stretch_array(audio, sr, ratio=factor)
    except ImportError:
        import librosa
        stretched = librosa.effects.time_stretch(audio, rate=1.0 / factor)

    sf.write(out_path, stretched, sr)
