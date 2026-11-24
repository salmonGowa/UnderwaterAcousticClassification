"""Log-mel spectrograms, segmenting, and MFCC summary statistics.

We compute one log-mel for the whole recording with center=False. Frame i then
covers samples [i * hop, i * hop + n_fft), so a segment that starts at a multiple
of the hop is exactly a slice of the full spectrogram. This lets us cache one
array per recording and cut 4 s segments out of it without recomputing.
tests/test_features.py checks this equality.
"""

import librosa
import numpy as np
import scipy.fft


class FeatureParams:
    def __init__(self, sample_rate=16000, segment_s=4.0, segment_hop_s=2.0, n_mels=64,
                 win_ms=25.0, hop_ms=10.0, n_fft=512, fmin=0.0, n_mfcc=20):
        self.sample_rate = sample_rate
        self.win_length = int(round(sample_rate * win_ms / 1000))
        self.hop_length = int(round(sample_rate * hop_ms / 1000))
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.n_mfcc = n_mfcc
        self.segment_samples = int(round(sample_rate * segment_s))
        self.segment_hop_samples = int(round(sample_rate * segment_hop_s))
        if self.segment_hop_samples % self.hop_length:
            raise ValueError("segment hop must be a whole number of frame hops")
        self.frames_per_segment = 1 + (self.segment_samples - self.n_fft) // self.hop_length
        self.segment_hop_frames = self.segment_hop_samples // self.hop_length

    @classmethod
    def from_config(cls, features_config):
        return cls(**features_config)


def log_mel(audio, params):
    """Return a (n_mels, n_frames) float32 array in dB (power, ref = 1.0)."""
    mel_power = librosa.feature.melspectrogram(
        y=audio, sr=params.sample_rate, n_fft=params.n_fft, win_length=params.win_length,
        hop_length=params.hop_length, n_mels=params.n_mels, fmin=params.fmin,
        fmax=params.sample_rate / 2, center=False, power=2.0)
    return librosa.power_to_db(mel_power, ref=1.0, amin=1e-12, top_db=None).astype(np.float32)


def n_segments(n_samples, params):
    if n_samples < params.segment_samples:
        return 0
    return 1 + (n_samples - params.segment_samples) // params.segment_hop_samples


def segment_slice(full_log_mel, segment_index, params):
    start = segment_index * params.segment_hop_frames
    return full_log_mel[:, start:start + params.frames_per_segment]


def normalise_segment(segment):
    """Subtract the segment's mean level (dB).

    Recordings come from different, uncalibrated hydrophones and gains. Removing the
    overall level keeps the spectral shape and stops a model from using loudness,
    which mostly reflects distance and equipment rather than vessel type.
    """
    return segment - segment.mean()


def mfcc_summary(segment_log_mel, n_mfcc):
    """MFCC mean and std over time, plus mean and std of their deltas: 4 * n_mfcc values."""
    mfcc = scipy.fft.dct(normalise_segment(segment_log_mel), type=2, axis=0, norm="ortho")[:n_mfcc]
    delta = np.diff(mfcc, axis=1)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1),
                           delta.mean(axis=1), delta.std(axis=1)]).astype(np.float32)
