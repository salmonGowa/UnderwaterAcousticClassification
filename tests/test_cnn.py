"""Regression test: the CNN must work in eval mode after training on only a few batches."""

import numpy as np
import torch

from src.models.cnn import predict_proba, train_cnn


class ToyDataset(torch.utils.data.Dataset):
    """Class k has extra energy in mel band 16k..16k+8: a frequency-position task."""

    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self.targets = rng.integers(0, 4, size=n)
        self.x = rng.normal(0, 0.1, size=(n, 1, 64, 64)).astype(np.float32)
        for i, k in enumerate(self.targets):
            self.x[i, 0, 16 * k:16 * k + 8] += 0.5

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), int(self.targets[i])


def test_cnn_learns_and_eval_mode_is_consistent():
    train, val = ToyDataset(192, seed=0), ToyDataset(64, seed=1)   # 3 batches per epoch
    settings = {"lr": 1e-3, "weight_decay": 0.0, "batch_size": 64, "max_epochs": 6, "patience": 6}
    model, info = train_cnn(train, val, 4, settings, seed=0, device="cpu")
    accuracy = (predict_proba(model, val, 64, "cpu").argmax(axis=1) == val.targets).mean()
    assert accuracy > 0.9, info
