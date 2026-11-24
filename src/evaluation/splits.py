"""Grouped splits: every recording of a vessel group lands on the same side.

All functions work on a recording-level table with columns recording_id,
group_id and target. Segments inherit their recording's side, so clip-level
leakage is impossible by construction.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def grouped_folds(recordings, n_folds, seed):
    """List of (train_recording_ids, test_recording_ids), stratified by class, grouped by vessel."""
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    ids = recordings["recording_id"].to_numpy()
    folds = []
    for train_idx, test_idx in splitter.split(ids, recordings["target"], groups=recordings["group_id"]):
        folds.append((set(ids[train_idx]), set(ids[test_idx])))
    return folds


def holdout_split(recordings, test_fraction, seed):
    """One grouped split: (dev_recording_ids, test_recording_ids). Test is touched once."""
    n_folds = int(round(1 / test_fraction))
    return grouped_folds(recordings, n_folds, seed)[0]


def assert_no_group_overlap(recordings, ids_a, ids_b):
    """Raise if any vessel group or recording appears on both sides."""
    if ids_a & ids_b:
        raise AssertionError(f"recordings on both sides: {sorted(ids_a & ids_b)[:5]}")
    group_of = recordings.set_index("recording_id")["group_id"]
    groups_a = set(group_of[list(ids_a)])
    groups_b = set(group_of[list(ids_b)])
    if groups_a & groups_b:
        raise AssertionError(f"groups on both sides: {sorted(groups_a & groups_b)[:5]}")


def class_counts(recordings, ids):
    subset = recordings[recordings["recording_id"].isin(ids)]
    return {
        "recordings": subset["target"].value_counts().sort_index().to_dict(),
        "groups": subset.groupby("target")["group_id"].nunique().sort_index().to_dict(),
    }


def segment_mask(segments, recording_ids):
    return np.asarray(segments["recording_id"].isin(recording_ids))
