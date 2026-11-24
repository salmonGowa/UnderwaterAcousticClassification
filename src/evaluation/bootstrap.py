"""Bootstrap confidence intervals that resample whole recordings, not segments.

Segments from one recording are strongly correlated, so resampling segments would
give intervals that are far too narrow. We resample recordings with replacement
within each class (stratified), so every replicate keeps the same class mix and
every class stays present.
"""

import numpy as np

from src.evaluation.metrics import headline


def bootstrap_ci(segment_recording_ids, segment_true, segment_pred,
                 recording_true, recording_pred, recording_ids,
                 n_classes, n_resamples=1000, seed=0, level=0.95):
    """Return {"segment": {...}, "recording": {...}}, each metric -> [low, high]."""
    rng = np.random.default_rng(seed)
    recording_ids = np.asarray(recording_ids)
    recording_true = np.asarray(recording_true)
    recording_pred = np.asarray(recording_pred)
    segment_rows = {r: np.flatnonzero(np.asarray(segment_recording_ids) == r) for r in recording_ids}
    by_class = [np.flatnonzero(recording_true == c) for c in range(n_classes)]
    by_class = [idx for idx in by_class if len(idx)]

    samples = {"segment": [], "recording": []}
    for _ in range(n_resamples):
        chosen = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in by_class])
        rows = np.concatenate([segment_rows[recording_ids[i]] for i in chosen])
        samples["segment"].append(headline(segment_true[rows], segment_pred[rows], n_classes))
        samples["recording"].append(headline(recording_true[chosen], recording_pred[chosen], n_classes))

    alpha = (1 - level) / 2
    result = {}
    for level_name, replicates in samples.items():
        result[level_name] = {
            metric: [float(np.quantile([r[metric] for r in replicates], alpha)),
                     float(np.quantile([r[metric] for r in replicates], 1 - alpha))]
            for metric in replicates[0]
        }
    return result
