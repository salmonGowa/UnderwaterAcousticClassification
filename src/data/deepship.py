"""Build the DeepShip manifest from the raw folder layout used by the official repo.

Expected layout (as in https://github.com/irfankamboh/DeepShip):
  raw/deepship/<Class>/<id>.wav
  raw/deepship/<Class>/<class>-metafile     (CSV, no header)

The metafile columns are not documented. What we infer, and why:
  col 0  file id                (matches the wav file name; numbering restarts in some files)
  col 2  ship name              (repeats across rows)
  col 5  duration in seconds    (sums to 47.07 h over all classes; README says 47 h 04 min)
Because ids restart, one wav can match several rows. We keep rows whose duration is
at least the wav duration (the GitHub copies are truncated at ~200 s), and if that
leaves several ships, we merge them into one group (see grouping.py).
"""

import csv
from pathlib import Path

import pandas as pd
import soundfile as sf

from src.data.grouping import groups_from_candidates

CLASSES = ["Cargo", "Passengership", "Tanker", "Tug"]
DURATION_TOLERANCE_S = 2.0


def read_metafile(path):
    rows = []
    with open(path, errors="replace") as f:
        for fields in csv.reader(f):
            if len(fields) < 6 or not fields[0].strip().isdigit():
                continue
            rows.append({
                "file_id": fields[0].strip(),
                "ship_name": " ".join(fields[2].upper().split()),
                "meta_duration_s": float(fields[5]),
            })
    return rows


def candidate_ships(file_id, wav_duration_s, meta_rows):
    """Return (candidate ship names, note) for one wav file."""
    same_id = [r for r in meta_rows if r["file_id"] == file_id]
    if not same_id:
        return [], "no metafile row"
    consistent = [r for r in same_id if r["meta_duration_s"] >= wav_duration_s - DURATION_TOLERANCE_S]
    chosen = consistent if consistent else same_id
    names = sorted({r["ship_name"] for r in chosen})
    if len(names) == 1:
        note = "unique match" if len(same_id) == 1 else "resolved by duration"
    else:
        note = f"ambiguous: merged {len(names)} ships"
    if not consistent:
        note += "; no duration-consistent row"
    return names, note


def build_manifest(data_root):
    raw = Path(data_root) / "raw" / "deepship"
    records, candidates = [], {}
    for class_name in CLASSES:
        class_dir = raw / class_name
        if not class_dir.exists():
            continue
        meta_rows = read_metafile(class_dir / f"{class_name.lower()}-metafile")
        for wav in sorted(class_dir.glob("*.wav"), key=lambda p: int(p.stem)):
            duration = sf.info(str(wav)).duration
            recording_id = f"deepship/{class_name}/{wav.stem}"
            names, note = candidate_ships(wav.stem, duration, meta_rows)
            if not names:
                names = [recording_id]  # fall back to one group per recording
            candidates[recording_id] = [f"deepship:{n}" for n in names]
            records.append({
                "recording_id": recording_id,
                "dataset": "deepship",
                "path": str(wav.relative_to(data_root)),
                "label": class_name,
                "group_note": note,
                "duration_s": round(duration, 3),
            })
    if not records:
        raise FileNotFoundError(f"no DeepShip wav files under {raw}")
    groups = groups_from_candidates(candidates)
    manifest = pd.DataFrame(records)
    manifest["group_id"] = manifest["recording_id"].map(groups)
    return manifest
