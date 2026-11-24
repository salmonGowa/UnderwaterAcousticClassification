"""A manifest is one CSV row per recording. Every later step reads only this table.

Columns:
  recording_id  unique across datasets, e.g. "deepship/Tanker/12"
  dataset       "deepship", "shipsear", ...
  path          audio path relative to data_root
  label         the dataset's own class name (task label maps are applied later)
  group_id      vessel group used for splitting (see grouping.py)
  group_note    how the group was decided, for auditing
  duration_s    duration of the audio file in seconds
"""

import pandas as pd

COLUMNS = ["recording_id", "dataset", "path", "label", "group_id", "group_note", "duration_s"]


def validate(manifest):
    missing = [c for c in COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"manifest is missing columns {missing}")
    if manifest["recording_id"].duplicated().any():
        raise ValueError("duplicate recording_id in manifest")
    if manifest[["label", "group_id"]].isna().any().any():
        raise ValueError("manifest has empty label or group_id")
    # A group must not mix classes, otherwise a grouped split cannot be stratified honestly.
    labels_per_group = manifest.groupby("group_id")["label"].nunique()
    mixed = labels_per_group[labels_per_group > 1]
    if len(mixed):
        raise ValueError(f"groups with more than one label: {list(mixed.index)[:5]}")
    return manifest


def save(manifest, path):
    validate(manifest)
    manifest[COLUMNS].sort_values("recording_id").to_csv(path, index=False)


def load(path):
    return validate(pd.read_csv(path, dtype={"group_id": str, "recording_id": str}))


def apply_label_map(manifest, label_map):
    """Keep recordings whose native label is in label_map and add a 'target' column."""
    kept = manifest[manifest["label"].isin(label_map)].copy()
    kept["target"] = kept["label"].map(label_map)
    return kept.reset_index(drop=True)
