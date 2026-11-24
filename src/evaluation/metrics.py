"""Metrics at segment level and recording level, plus reference levels."""

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)


def headline(y_true, y_pred, n_classes):
    labels = list(range(n_classes))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def full_report(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    report = headline(y_true, y_pred, len(class_names))
    report["per_class"] = {
        name: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for name, p, r, f, s in zip(class_names, precision, recall, f1, support)
    }
    report["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return report


def aggregate_to_recordings(recording_ids, probabilities):
    """Average segment probabilities per recording. Returns (unique ids, mean probabilities)."""
    recording_ids = np.asarray(recording_ids)
    unique_ids = np.unique(recording_ids)
    mean_probs = np.stack([probabilities[recording_ids == r].mean(axis=0) for r in unique_ids])
    return unique_ids, mean_probs


def chance_levels(y_true, n_classes):
    """Expected scores of a uniform random guesser (large-sample approximation).

    Balanced accuracy is exactly 1/K. For class c with prevalence p_c, precision
    is p_c and recall is 1/K, so F1_c = 2 * p_c * (1/K) / (p_c + 1/K).
    """
    prevalence = np.bincount(y_true, minlength=n_classes) / len(y_true)
    k = 1.0 / n_classes
    f1_per_class = 2 * prevalence * k / (prevalence + k)
    return {"macro_f1": float(f1_per_class.mean()), "balanced_accuracy": k}
