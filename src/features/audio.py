"""Load audio as mono float32 at the target sample rate."""

import librosa
import numpy as np
import soundfile as sf


def load_mono(path, target_sr):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # all our sources are mono already; this is a safe no-op
    if sr != target_sr:
        # soxr_hq applies an anti-aliasing low-pass at target_sr / 2 before decimating.
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    return audio.astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
