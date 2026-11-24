import warnings

import numpy as np
import pytest

from src.features.logmel import FeatureParams, log_mel, mfcc_summary, n_segments, segment_slice


@pytest.fixture(scope="module")
def params():
    return FeatureParams()


def test_slicing_the_full_spectrogram_equals_computing_each_segment(params):
    rng = np.random.default_rng(0)
    audio = rng.normal(size=params.sample_rate * 11).astype(np.float32)
    full = log_mel(audio, params)
    for s in range(n_segments(len(audio), params)):
        start = s * params.segment_hop_samples
        direct = log_mel(audio[start:start + params.segment_samples], params)
        np.testing.assert_allclose(segment_slice(full, s, params), direct, atol=1e-3)


def test_segment_count_and_shape(params):
    assert n_segments(params.sample_rate * 3, params) == 0           # shorter than 4 s
    assert n_segments(params.sample_rate * 4, params) == 1
    assert n_segments(params.sample_rate * 10, params) == 4          # starts at 0, 2, 4, 6 s
    full = log_mel(np.zeros(params.sample_rate * 10, dtype=np.float32) + 1e-3, params)
    assert segment_slice(full, 3, params).shape == (64, params.frames_per_segment)


def test_no_empty_mel_filters(params):
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # librosa warns on empty filters
        log_mel(np.random.default_rng(0).normal(size=params.sample_rate).astype(np.float32), params)


def test_mfcc_summary_ignores_overall_gain(params):
    audio = np.random.default_rng(1).normal(size=params.segment_samples).astype(np.float32)
    a = mfcc_summary(log_mel(audio, params), params.n_mfcc)
    b = mfcc_summary(log_mel(10 * audio, params), params.n_mfcc)  # +20 dB
    assert a.shape == (4 * params.n_mfcc,)
    np.testing.assert_allclose(a, b, atol=1e-3)
