"""Compute features once per recording and cache them as .npy files.

Layout under <data_root>/cache/<dataset>/:
  logmel/<safe_id>.npy   (n_mels, n_frames) float16, whole recording
  audio16k/<safe_id>.npy float16 waveform divided by its peak (for noise-mixing tests)
  audio16k_peak.csv      the peak values, to undo that scaling
  segments.csv           one row per segment: recording_id, segment_index
  mfcc_stats.npy         (n_segments, 4 * n_mfcc) float32, rows aligned with segments.csv
Training and evaluation only read these files.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.audio import load_mono
from src.features.logmel import log_mel, mfcc_summary, n_segments, segment_slice


def safe_id(recording_id):
    return recording_id.replace("/", "__")


def cache_dir(data_root, dataset):
    return Path(data_root) / "cache" / dataset


def build_cache(manifest, data_root, params):
    """Build the cache for one dataset's manifest. Recordings already cached are skipped."""
    datasets = manifest["dataset"].unique()
    if len(datasets) != 1:
        raise ValueError("build_cache expects a single-dataset manifest")
    root = cache_dir(data_root, datasets[0])
    (root / "logmel").mkdir(parents=True, exist_ok=True)
    (root / "audio16k").mkdir(parents=True, exist_ok=True)
    peak_path = root / "audio16k_peak.csv"
    peaks = pd.read_csv(peak_path).set_index("recording_id")["peak"].to_dict() if peak_path.exists() else {}

    for i, row in enumerate(manifest.itertuples()):
        logmel_path = root / "logmel" / f"{safe_id(row.recording_id)}.npy"
        audio_path = root / "audio16k" / f"{safe_id(row.recording_id)}.npy"
        if logmel_path.exists() and audio_path.exists() and row.recording_id in peaks:
            continue
        audio = load_mono(Path(data_root) / row.path, params.sample_rate)
        peak = float(np.abs(audio).max()) or 1.0
        np.save(audio_path, (audio / peak).astype(np.float16))
        np.save(logmel_path, log_mel(audio, params).astype(np.float16))
        peaks[row.recording_id] = peak
        print(f"[cache] {i + 1}/{len(manifest)} {row.recording_id}", flush=True)

    pd.DataFrame(sorted(peaks.items()), columns=["recording_id", "peak"]).to_csv(peak_path, index=False)
    _write_segment_tables(manifest, root, params)


def _write_segment_tables(manifest, root, params):
    segments, stats = [], []
    for row in manifest.itertuples():
        full = np.load(root / "logmel" / f"{safe_id(row.recording_id)}.npy").astype(np.float32)
        n_samples = (full.shape[1] - 1) * params.hop_length + params.n_fft
        for s in range(n_segments(n_samples, params)):
            segments.append({"recording_id": row.recording_id, "segment_index": s})
            stats.append(mfcc_summary(segment_slice(full, s, params), params.n_mfcc))
    pd.DataFrame(segments).to_csv(root / "segments.csv", index=False)
    np.save(root / "mfcc_stats.npy", np.stack(stats))


def load_segments(data_root, dataset):
    """Return (segments DataFrame, MFCC-stats array) with aligned rows."""
    root = cache_dir(data_root, dataset)
    return pd.read_csv(root / "segments.csv"), np.load(root / "mfcc_stats.npy")


def load_logmels(data_root, dataset, recording_ids):
    root = cache_dir(data_root, dataset)
    return {r: np.load(root / "logmel" / f"{safe_id(r)}.npy") for r in recording_ids}


def load_audio16k(data_root, dataset, recording_id):
    root = cache_dir(data_root, dataset)
    peaks = pd.read_csv(root / "audio16k_peak.csv").set_index("recording_id")["peak"]
    audio = np.load(root / "audio16k" / f"{safe_id(recording_id)}.npy").astype(np.float32)
    return audio * float(peaks[recording_id])
