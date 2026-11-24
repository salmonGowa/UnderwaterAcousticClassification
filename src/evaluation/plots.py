"""Figures: confusion matrices, macro-F1 vs SNR, and example spectrograms for failure analysis."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.features.logmel import normalise_segment, segment_slice  # noqa: E402

# Fixed model -> colour/marker assignment (validated categorical slots 1-4), never cycled.
MODEL_STYLE = {
    "majority": {"color": "#2a78d6", "marker": "s", "label": "Majority class"},
    "logreg": {"color": "#eb6834", "marker": "o", "label": "Logistic regression (MFCC)"},
    "random_forest": {"color": "#1baf7a", "marker": "^", "label": "Random forest (MFCC)"},
    "cnn": {"color": "#eda100", "marker": "D", "label": "CNN (log-mel)"},
}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def confusion_figure(report, class_names, title, path):
    """Row-normalised confusion matrix (recall per true class), with raw counts in each cell."""
    counts = np.array(report["confusion_matrix"])
    rates = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    size = 1.1 * len(class_names) + 2
    fig, ax = plt.subplots(figsize=(size, size - 0.5), facecolor=SURFACE)
    ax.imshow(rates, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ink = SURFACE if rates[i, j] > 0.6 else INK
            ax.text(j, i, f"{rates[i, j]:.2f}\n({counts[i, j]})", ha="center", va="center", color=ink, fontsize=9)
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted", color=INK)
    ax.set_ylabel("True", color=INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def noise_figure(noise_result, level, title, path):
    """Macro-F1 against added-noise SNR, one line per model, labelled at the line end."""
    snrs = noise_result["snr_db"]
    x = np.arange(len(snrs))
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    for name, points in noise_result["models"].items():
        style = MODEL_STYLE[name]
        mean = np.array([p[level]["macro_f1_mean"] for p in points])
        std = np.array([p[level]["macro_f1_std"] for p in points])
        ax.plot(x, mean, color=style["color"], marker=style["marker"], markersize=8, linewidth=2,
                label=style["label"], markeredgecolor=SURFACE, markeredgewidth=2)
        if std.any():
            ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.15, linewidth=0)
        ax.annotate(style["label"].split(" (")[0], (x[-1], mean[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=8)
    ax.set_xticks(x, [str(s) if s == "clean" else f"{s} dB" for s in snrs])
    ax.set_xlabel("Added ambient noise: SNR (clip RMS / added-noise RMS)", color=INK)
    ax.set_ylabel(f"Macro-F1 ({level} level)", color=INK)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.3, len(snrs) - 0.3 + 1.2)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def most_confused_pair(report, class_names):
    counts = np.array(report["confusion_matrix"], dtype=float)
    np.fill_diagonal(counts, -1)
    i, j = np.unravel_index(counts.argmax(), counts.shape)
    return (class_names[i], class_names[j], int(counts[i, j])) if counts[i, j] > 0 else None


def failure_examples_figure(predictions, logmels, params, class_names, pair, path, n_examples=3, seed=0):
    """Top row: segments of true class A predicted as B. Bottom row: correctly classified A and B."""
    true_name, pred_name, _ = pair
    a, b = class_names.index(true_name), class_names.index(pred_name)
    t = np.array(predictions["segment_true"])
    p = np.array(predictions["segment_pred"])
    rng = np.random.default_rng(seed)
    groups = {
        f"true {true_name}, predicted {pred_name}": np.flatnonzero((t == a) & (p == b)),
        f"true {true_name}, correct": np.flatnonzero((t == a) & (p == a)),
        f"true {pred_name}, correct": np.flatnonzero((t == b) & (p == b)),
    }
    fig, axes = plt.subplots(len(groups), n_examples, figsize=(3.2 * n_examples, 2.4 * len(groups)),
                             facecolor=SURFACE, squeeze=False)
    seconds = params.frames_per_segment * params.hop_length / params.sample_rate
    for row, (caption, idx) in enumerate(groups.items()):
        chosen = rng.choice(idx, size=min(n_examples, len(idx)), replace=False) if len(idx) else []
        for col in range(n_examples):
            ax = axes[row, col]
            if col >= len(chosen):
                ax.axis("off")
                continue
            k = chosen[col]
            rec = predictions["segment_recording_ids"][k]
            seg = segment_slice(logmels[rec].astype(np.float32), predictions["segment_index"][k], params)
            ax.imshow(normalise_segment(seg), origin="lower", aspect="auto", cmap="magma",
                      extent=[0, seconds, 0, params.n_mels])
            ax.set_title(f"{rec} #{predictions['segment_index'][k]}", fontsize=7, color=INK)
            ax.set_xlabel("s", fontsize=7)
            ax.set_ylabel("mel bin", fontsize=7)
            ax.tick_params(labelsize=6)
        axes[row, 0].annotate(caption, (0, 1.25), xycoords="axes fraction", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
