"""Deep-learning detectors (PyTorch, CPU is enough): dense autoencoder and LSTM autoencoder.

Both are trained on NORMAL data only and flag what they cannot reconstruct well:
score = mean squared reconstruction error.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import BaseDetector, subsample

log = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
except ImportError:  # the rest of the project works without PyTorch
    torch, nn = None, None


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is not installed: pip install torch (or disable the deep models in config.yaml)")


def _train_loop(model, X_tr, X_va, epochs, batch_size, lr, patience, seed, label):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_tr_t, X_va_t = torch.from_numpy(X_tr), torch.from_numpy(X_va)
    best, best_state, wait, history = np.inf, None, 0, []
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), generator=gen)
        total = 0.0
        for i in range(0, len(perm), batch_size):
            xb = X_tr_t[perm[i:i + batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(X_va_t), X_va_t).item()
        history.append((total / len(X_tr_t), val))
        if val < best - 1e-5:
            best, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    log.info("%s: %d epochs, best validation reconstruction loss %.4f", label, len(history), best)
    return history


class AutoencoderDetector(BaseDetector):
    name, kind = "autoencoder", "unsupervised"

    def _build(self, d):
        h = self.params.get("hidden", [32, 8])
        layers, prev = [], d
        for size in h:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        for size in list(reversed(h[:-1])) + [d]:
            layers += [nn.Linear(prev, size)] + ([nn.ReLU()] if size != d else [])
            prev = size
        return nn.Sequential(*layers)

    def fit(self, X, y=None, groups=None):
        _require_torch()
        Xn = self._check_columns(X)
        Xs, _ = subsample(Xn, None, self.params.get("max_train_samples", 300000), self.seed)
        self.scaler = StandardScaler().fit(Xs)
        Z = np.clip(self.scaler.transform(Xs), -10, 10).astype(np.float32)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(Z))
        n_va = max(1, len(Z) // 10)
        self.net = self._build(Z.shape[1])
        self.history = _train_loop(self.net, Z[idx[n_va:]], Z[idx[:n_va]], self.params.get("epochs", 30),
                                   self.params.get("batch_size", 512), self.params.get("lr", 1e-3),
                                   self.params.get("patience", 4), self.seed, self.name)
        return self

    def score(self, X, groups=None):
        Z = np.clip(self.scaler.transform(self._check_columns(X)), -10, 10).astype(np.float32)
        self.net.eval()
        out = np.empty(len(Z))
        with torch.no_grad():
            for i in range(0, len(Z), 65536):
                zb = torch.from_numpy(Z[i:i + 65536])
                out[i:i + 65536] = ((self.net(zb) - zb) ** 2).mean(dim=1).numpy()
        return out


class _LSTMAE(nn.Module if nn else object):
    def __init__(self, d, hidden):
        super().__init__()
        self.enc = nn.LSTM(d, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, d)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        y, _ = self.dec(z)
        return self.out(y)


class LSTMAutoencoderDetector(BaseDetector):
    """Sequence autoencoder on the last `window` steps of each sensor (optional, slower)."""
    name, kind = "lstm_autoencoder", "unsupervised"

    def _windows(self, Z, groups, rows):
        """Window ending at each requested row, padded with the sensor's first row at the start."""
        W = self.params.get("window", 24)
        starts = self._group_start[rows]
        offs = np.arange(-W + 1, 1)
        idx = rows[:, None] + offs[None, :]
        idx = np.maximum(idx, starts[:, None])
        return Z[idx]

    def _index_groups(self, groups):
        groups = np.asarray(groups)
        change = np.r_[True, groups[1:] != groups[:-1]]
        first = np.flatnonzero(change)
        self._group_start = first[np.cumsum(change) - 1]

    def fit(self, X, y=None, groups=None):
        _require_torch()
        if groups is None:
            raise ValueError("LSTM autoencoder needs groups (sensor id per row, rows sorted by sensor and time)")
        Xn = self._check_columns(X)
        self.scaler = StandardScaler().fit(Xn)
        Z = np.clip(self.scaler.transform(Xn), -10, 10).astype(np.float32)
        self._index_groups(groups)
        rng = np.random.default_rng(self.seed)
        rows = rng.choice(len(Z), min(len(Z), self.params.get("max_train_samples", 40000)), replace=False)
        Wn = self._windows(Z, groups, rows).astype(np.float32)
        n_va = max(1, len(Wn) // 10)
        self.net = _LSTMAE(Z.shape[1], self.params.get("hidden", 32))
        self.history = _train_loop(self.net, Wn[n_va:], Wn[:n_va], self.params.get("epochs", 10),
                                   self.params.get("batch_size", 256), self.params.get("lr", 1e-3),
                                   self.params.get("patience", 3), self.seed, self.name)
        return self

    def score(self, X, groups=None):
        if groups is None:
            raise ValueError("LSTM autoencoder needs groups")
        Z = np.clip(self.scaler.transform(self._check_columns(X)), -10, 10).astype(np.float32)
        self._index_groups(groups)
        out = np.empty(len(Z))
        self.net.eval()
        with torch.no_grad():
            for i in range(0, len(Z), 8192):
                rows = np.arange(i, min(i + 8192, len(Z)))
                wb = torch.from_numpy(self._windows(Z, groups, rows).astype(np.float32))
                out[rows] = ((self.net(wb) - wb) ** 2).mean(dim=(1, 2)).numpy()
        return out
