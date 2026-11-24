"""The core leakage guarantees: no vessel group or recording on both sides of any split."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import splits


def make_recordings(n_groups_per_class=8, recordings_per_group=(1, 5), classes=("a", "b", "c"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in classes:
        for g in range(n_groups_per_class):
            for r in range(rng.integers(*recordings_per_group)):
                rows.append({"recording_id": f"{c}/{g}/{r}", "group_id": f"{c}-{g}", "target": c})
    return pd.DataFrame(rows)


def groups_of(recordings, ids):
    return set(recordings.set_index("recording_id").loc[list(ids), "group_id"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_folds_never_share_a_group(seed):
    recs = make_recordings(seed=seed)
    for train, test in splits.grouped_folds(recs, n_folds=4, seed=seed):
        assert not train & test
        assert not groups_of(recs, train) & groups_of(recs, test)
        splits.assert_no_group_overlap(recs, train, test)


def test_every_recording_is_tested_exactly_once():
    recs = make_recordings()
    folds = splits.grouped_folds(recs, n_folds=4, seed=0)
    tested = [r for _, test in folds for r in test]
    assert sorted(tested) == sorted(recs["recording_id"])


def test_holdout_and_inner_folds_are_disjoint():
    recs = make_recordings(n_groups_per_class=10)
    dev, test = splits.holdout_split(recs, test_fraction=0.2, seed=0)
    splits.assert_no_group_overlap(recs, dev, test)
    dev_recs = recs[recs["recording_id"].isin(dev)]
    for train, val in splits.grouped_folds(dev_recs, n_folds=5, seed=0):
        assert not (train | val) & test
        splits.assert_no_group_overlap(recs, train, val)


def test_split_is_deterministic():
    recs = make_recordings()
    assert splits.grouped_folds(recs, 4, seed=3) == splits.grouped_folds(recs, 4, seed=3)


def test_overlap_check_catches_a_leak():
    recs = make_recordings()
    a_group = recs.groupby("group_id")["recording_id"].apply(list)
    leaky_group = next(ids for ids in a_group if len(ids) >= 2)
    train = {leaky_group[0]}
    test = {leaky_group[1]}
    with pytest.raises(AssertionError, match="groups on both sides"):
        splits.assert_no_group_overlap(recs, train, test)


def test_segments_follow_their_recording():
    recs = make_recordings()
    segments = pd.DataFrame({"recording_id": np.repeat(recs["recording_id"].to_numpy(), 3)})
    train, test = splits.grouped_folds(recs, 4, seed=0)[0]
    train_mask = splits.segment_mask(segments, train)
    test_mask = splits.segment_mask(segments, test)
    assert not (train_mask & test_mask).any()
    assert (train_mask | test_mask).all()
