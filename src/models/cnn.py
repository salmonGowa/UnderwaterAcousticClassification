"""A small CNN on log-mel segments, with early stopping on a grouped validation split."""

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.features.logmel import normalise_segment, segment_slice


class SegmentDataset(Dataset):
    """Cuts segments out of cached whole-recording log-mels on the fly (slicing, no recomputing)."""

    def __init__(self, logmels, segments, targets, params):
        self.logmels = logmels                       # recording_id -> (n_mels, n_frames)
        self.recording_ids = segments["recording_id"].to_numpy()
        self.segment_indices = segments["segment_index"].to_numpy()
        self.targets = np.asarray(targets)
        self.params = params

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        full = self.logmels[self.recording_ids[i]]
        x = segment_slice(full, self.segment_indices[i], self.params).astype(np.float32)
        x = normalise_segment(x) / 20.0              # roughly unit scale: 20 dB -> 1.0
        return torch.from_numpy(x[None]), int(self.targets[i])


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """Four conv blocks, then average over time only.

    Frequency position carries meaning in a spectrogram (a 50 Hz tonal is not a
    500 Hz tonal), so we keep the frequency axis and let the linear layer see it.
    """

    def __init__(self, n_classes, n_mels=64, channels=(16, 32, 64, 128), dropout=0.3):
        super().__init__()
        blocks, in_channels = [], 1
        for out_channels in channels:
            blocks.append(conv_block(in_channels, out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        freq_bins_left = n_mels // 2 ** len(channels)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                        nn.Linear(in_channels * freq_bins_left, n_classes))

    def forward(self, x):                       # x: (batch, 1, n_mels, frames)
        h = self.features(x).mean(dim=3)        # (batch, channels, freq_bins_left)
        return self.classifier(h)


def predict_proba(model, dataset, batch_size, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs = []
    with torch.no_grad():
        for x, _ in loader:
            outputs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
    return np.concatenate(outputs)


def recalibrate_batchnorm(model, dataset, batch_size, device, max_batches=50, seed=0):
    """Recompute BatchNorm running statistics as a plain average over training batches.

    During training they are an exponential moving average that starts at mean 0,
    var 1. With few batches it has not converged, so eval-mode outputs are wrong
    (on the synthetic smoke data: 100% train-mode accuracy, 25% eval-mode).
    """
    batchnorms = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for bn in batchnorms:
        bn.reset_running_stats()
        bn.momentum = None                     # None = cumulative average in PyTorch
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model.train()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i == max_batches:
                break
            model(x.to(device))
    for bn in batchnorms:
        bn.momentum = 0.1
    model.eval()


def train_cnn(train_set, val_set, n_classes, settings, seed, device):
    """Train with class-weighted cross-entropy; keep the epoch with the best validation macro-F1."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SmallCNN(n_classes).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    counts = np.bincount(train_set.targets, minlength=n_classes).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_set, batch_size=settings["batch_size"], shuffle=True, generator=generator)

    best_f1, best_state, best_epoch, epochs_without_gain = -1.0, None, 0, 0
    for epoch in range(settings["max_epochs"]):
        model.train()
        for x, y in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            optimiser.step()
        recalibrate_batchnorm(model, train_set, settings["batch_size"], device, seed=seed)
        val_pred = predict_proba(model, val_set, settings["batch_size"], device).argmax(axis=1)
        val_f1 = f1_score(val_set.targets, val_pred, average="macro", labels=list(range(n_classes)), zero_division=0)
        if val_f1 > best_f1:
            best_f1, best_epoch, epochs_without_gain = val_f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= settings["patience"]:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_macro_f1": float(best_f1)}
