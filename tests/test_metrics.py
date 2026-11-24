import numpy as np
import pytest

from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.metrics import aggregate_to_recordings, chance_levels, headline


def test_chance_levels_balanced():
    y = np.repeat([0, 1, 2, 3], 25)
    chance = chance_levels(y, 4)
    assert chance["balanced_accuracy"] == 0.25
    assert chance["macro_f1"] == pytest.approx(0.25)


def test_chance_matches_simulated_random_guessing():
    rng = np.random.default_rng(0)
    y = rng.choice(3, size=200_000, p=[0.7, 0.2, 0.1])
    guess = rng.integers(0, 3, size=y.size)
    assert headline(y, guess, 3)["macro_f1"] == pytest.approx(chance_levels(y, 3)["macro_f1"], abs=0.005)


def test_aggregate_averages_probabilities_per_recording():
    ids, probs = aggregate_to_recordings(["b", "a", "b"], np.array([[0.2, 0.8], [0.9, 0.1], [0.6, 0.4]]))
    assert list(ids) == ["a", "b"]
    np.testing.assert_allclose(probs, [[0.9, 0.1], [0.4, 0.6]])


def test_bootstrap_interval_brackets_point_estimate():
    rng = np.random.default_rng(0)
    rec_ids = np.array([f"r{i}" for i in range(40)])
    rec_true = np.repeat([0, 1], 20)
    rec_pred = np.where(rng.random(40) < 0.8, rec_true, 1 - rec_true)
    seg_ids = np.repeat(rec_ids, 5)
    seg_true = np.repeat(rec_true, 5)
    seg_pred = np.repeat(rec_pred, 5)
    ci = bootstrap_ci(seg_ids, seg_true, seg_pred, rec_true, rec_pred, rec_ids, 2, n_resamples=300)
    point = headline(rec_true, rec_pred, 2)["macro_f1"]
    low, high = ci["recording"]["macro_f1"]
    assert low <= point <= high and high - low > 0
    # Segments copy their recording's prediction, so both levels must give the same interval.
    assert ci["segment"]["macro_f1"] == pytest.approx(ci["recording"]["macro_f1"], abs=1e-9)
