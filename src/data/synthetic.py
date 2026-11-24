"""Synthetic 'vessel' recordings for end-to-end smoke tests (never reported as results).

Each vessel gets a shaft-rate-like fundamental with harmonics (tonals) plus
broadband noise. Classes differ in their fundamental range, so the task is
learnable; each vessel has its own fundamental, so a model that memorises
vessels instead of classes is penalised by a grouped split, just like real data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

CLASS_F0_RANGES_HZ = {"Cargo": (40, 55), "Passengership": (60, 75), "Tanker": (25, 35), "Tug": (80, 100)}


def vessel_signal(f0, duration_s, sample_rate, rng):
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    signal = np.zeros_like(t)
    for harmonic in range(1, 30):
        amplitude = rng.uniform(0.2, 1.0) / harmonic ** 0.5
        signal += amplitude * np.sin(2 * np.pi * f0 * harmonic * t + rng.uniform(0, 2 * np.pi))
    noise = rng.normal(0, 1.0, size=t.shape)
    return signal / np.abs(signal).max() + 0.5 * noise


def ambient_signal(duration_s, sample_rate, rng):
    """Brown-ish noise: integrated white noise, high-passed by removing the running mean."""
    white = rng.normal(0, 1.0, size=int(duration_s * sample_rate))
    brown = np.cumsum(white)
    brown -= np.convolve(brown, np.ones(1000) / 1000, mode="same")
    return brown / np.abs(brown).max()


def write_dataset(data_root, dataset, sample_rate, vessels_per_class, recordings_per_vessel,
                  duration_s, n_ambient, gain, seed):
    """Write wavs under raw/<dataset>/ and return a manifest DataFrame."""
    rng = np.random.default_rng(seed)
    out_dir = Path(data_root) / "raw" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    def save(name, audio, label, group_id):
        path = out_dir / f"{name}.wav"
        sf.write(path, (gain * 0.1 * audio).astype(np.float32), sample_rate, subtype="FLOAT")
        records.append({
            "recording_id": f"{dataset}/{name}", "dataset": dataset,
            "path": str(path.relative_to(data_root)), "label": label,
            "group_id": group_id, "group_note": "synthetic", "duration_s": duration_s,
        })

    for label, (low, high) in CLASS_F0_RANGES_HZ.items():
        for v in range(vessels_per_class):
            f0 = rng.uniform(low, high)
            for r in range(recordings_per_vessel):
                audio = vessel_signal(f0 * rng.uniform(0.97, 1.03), duration_s, sample_rate, rng)
                save(f"{label}_v{v}_r{r}", audio, label, f"{dataset}:{label}_v{v}")
    for a in range(n_ambient):
        save(f"ambient_{a}", ambient_signal(duration_s, sample_rate, rng),
             "Natural ambient noise", f"{dataset}:ambient_{a}")
    return pd.DataFrame(records)
