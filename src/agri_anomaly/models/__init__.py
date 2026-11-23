"""MODEL LAYER: stage 1 (detection) and stage 2 (qualification) models, built from config.yaml."""
from __future__ import annotations

from .base import BaseDetector
from .classical import (IsolationForestDetector, OneClassSVMDetector, RandomForestDetector, TypeClassifier,
                        XGBoostDetector, ZScoreBaseline)
from .deep import AutoencoderDetector, LSTMAutoencoderDetector

STAGE1 = {
    "zscore_baseline": ZScoreBaseline,
    "isolation_forest": IsolationForestDetector,
    "one_class_svm": OneClassSVMDetector,
    "autoencoder": AutoencoderDetector,
    "lstm_autoencoder": LSTMAutoencoderDetector,
    "random_forest": RandomForestDetector,
    "xgboost": XGBoostDetector,
}


def build_stage1(cfg: dict) -> dict[str, BaseDetector]:
    seed = cfg["project"]["seed"]
    out = {}
    for name, params in cfg["models"]["stage1"].items():
        params = dict(params or {})
        if not params.pop("enabled", True):
            continue
        out[name] = STAGE1[name](seed=seed, **params)
    return out


def build_stage2(cfg: dict) -> dict[str, TypeClassifier]:
    seed = cfg["project"]["seed"]
    out = {}
    for name, params in cfg["models"]["stage2"].items():
        params = dict(params or {})
        if not params.pop("enabled", True):
            continue
        out[name] = TypeClassifier(algo="rf" if name.startswith("rf") else "xgb", seed=seed, **params)
    return out


__all__ = ["BaseDetector", "TypeClassifier", "STAGE1", "build_stage1", "build_stage2"]
