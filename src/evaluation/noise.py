"""Mix real ambient recordings into test clips at a chosen RMS-based SNR.

SNR here is 20 * log10(rms(clip) / rms(added_noise)). The clip itself already
contains the ambient noise present when it was recorded, so this is the ratio of
the original clip to the *added* noise, not a true vessel-to-noise ratio.
"""

import numpy as np

from src.features.audio import rms


def mix_at_snr(clip, noise, snr_db):
    if len(noise) != len(clip):
        raise ValueError("clip and noise must have the same length")
    noise_rms = rms(noise)
    if noise_rms == 0:
        raise ValueError("noise is silent")
    scale = rms(clip) / (noise_rms * 10 ** (snr_db / 20))
    return clip + scale * noise


def random_noise_chunk(noise_pool, n_samples, rng):
    """Pick a random recording from the pool and a random window of n_samples from it."""
    long_enough = [n for n in noise_pool if len(n) >= n_samples]
    if not long_enough:
        raise ValueError("no ambient recording is long enough")
    source = long_enough[rng.integers(len(long_enough))]
    start = rng.integers(0, len(source) - n_samples + 1)
    return source[start:start + n_samples]
