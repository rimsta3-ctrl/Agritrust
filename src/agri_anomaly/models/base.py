"""Common interface shared by every detector, so the evaluation treats them identically.

    fit(X, y=None, groups=None)   train (y is ignored by unsupervised models)
    score(X, groups=None)         anomaly score, higher = more suspicious
    predict(X, groups=None)       0/1 using self.threshold (chosen on the validation split)

groups = sensor id of each row (only the sequence model needs it, to build time windows).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class BaseDetector:
    name: str = "base"
    kind: str = "unsupervised"        # rule | unsupervised | supervised
    trained_on: str = "normal data"

    def __init__(self, seed: int = 0, **params):
        self.seed = seed
        self.params = params
        self.threshold: float | None = None
        self.val_f1: float | None = None
        self.feature_names: list[str] | None = None

    # -- to implement
    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None, groups: np.ndarray | None = None):
        raise NotImplementedError

    def score(self, X: pd.DataFrame, groups: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    # -- shared
    def predict(self, X: pd.DataFrame, groups: np.ndarray | None = None, threshold: float | None = None) -> np.ndarray:
        thr = self.threshold if threshold is None else threshold
        if thr is None:
            raise RuntimeError(f"{self.name}: threshold not set (calibrate on validation first)")
        return (self.score(X, groups) >= thr).astype(np.int8)

    def _check_columns(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_names is None:
            self.feature_names = list(X.columns)
        return X[self.feature_names].to_numpy(dtype=np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "BaseDetector":
        return joblib.load(path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(threshold={self.threshold})"


def subsample(X: np.ndarray, y: np.ndarray | None, max_n: int | None, seed: int):
    """Random subsample; with labels, keeps every positive row and fills with negatives."""
    n = len(X)
    if not max_n or n <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    if y is None:
        idx = rng.choice(n, max_n, replace=False)
        return X[idx], None
    pos = np.flatnonzero(y != 0)
    neg = np.flatnonzero(y == 0)
    n_neg = max(0, max_n - len(pos))
    idx = np.concatenate([pos, rng.choice(neg, min(n_neg, len(neg)), replace=False)])
    rng.shuffle(idx)
    return X[idx], y[idx]
