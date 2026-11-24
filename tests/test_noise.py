import numpy as np
import pytest

from src.evaluation.noise import mix_at_snr, random_noise_chunk
from src.features.audio import rms


@pytest.mark.parametrize("snr_db", [20, 10, 5, 0, -5])
def test_added_noise_has_the_requested_snr(snr_db):
    rng = np.random.default_rng(0)
    clip = rng.normal(0, 0.3, 64000)
    noise = rng.normal(0, 2.0, 64000)
    added = mix_at_snr(clip, noise, snr_db) - clip
    assert 20 * np.log10(rms(clip) / rms(added)) == pytest.approx(snr_db, abs=1e-9)


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        mix_at_snr(np.ones(10), np.ones(9), 0)


def test_noise_chunk_is_a_window_of_a_pool_recording():
    pool = [np.arange(100.0), np.arange(1000.0, 1050.0)]
    chunk = random_noise_chunk(pool, 60, np.random.default_rng(0))   # only the first is long enough
    assert len(chunk) == 60 and chunk[0] < 100 and np.all(np.diff(chunk) == 1)
