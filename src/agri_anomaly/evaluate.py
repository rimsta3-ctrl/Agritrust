"""EVALUATION LAYER: thresholds, metrics, per-scenario analysis, inference time."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)

from . import CATEGORIES


def choose_threshold(scores: np.ndarray, y: np.ndarray, strategy: str = "f1", target_fpr: float = 0.01) -> float:
    """Pick the decision threshold on the VALIDATION split (never on test)."""
    if strategy == "fpr" or y.sum() == 0:
        normal = scores[y == 0]
        return float(np.quantile(normal, 1 - target_fpr))
    prec, rec, thr = precision_recall_curve(y, scores)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    best = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[best]) if len(thr) else float(np.median(scores))


def binary_metrics(y: np.ndarray, scores: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) > 1
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, scores) if both else np.nan,
        "pr_auc": average_precision_score(y, scores) if both else np.nan,
        "fpr": fp / max(1, fp + tn),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def timed_scores(model, X, groups=None) -> tuple[np.ndarray, float]:
    """Scores and inference time in microseconds per sample."""
    t0 = time.perf_counter()
    s = model.score(X, groups)
    dt = time.perf_counter() - t0
    return s, 1e6 * dt / max(1, len(X))


def per_scenario(meta: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Point recall, event recall (>= 1 flagged point) and detection delay (steps) per scenario."""
    df = meta[["scenario", "event_id"]].copy()
    df["pred"] = pred
    df["pos"] = df.groupby("event_id").cumcount()
    rows = []
    for scen, g in df[df["event_id"] >= 0].groupby("scenario"):
        ev = g.groupby("event_id")
        detected = ev["pred"].max()
        first = g[g["pred"] == 1].groupby("event_id")["pos"].min()
        rows.append({"scenario": scen, "events": int(len(detected)),
                     "point_recall": g["pred"].mean(),
                     "event_recall": detected.mean(),
                     "mean_delay_steps": first.mean() if len(first) else np.nan})
    return pd.DataFrame(rows)


def stage2_report(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = [c for c in CATEGORIES if c != "normal"]
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    table = pd.DataFrame(rep).T
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    return table, cm


def end_to_end(y_true_cat: np.ndarray, detected: np.ndarray, type_pred: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """4-class evaluation of the full pipeline: normal / outlier / fault / falsification."""
    final = np.where(detected == 1, type_pred, "normal")
    cm = pd.DataFrame(confusion_matrix(y_true_cat, final, labels=CATEGORIES), index=CATEGORIES, columns=CATEGORIES)
    scores = {
        "accuracy": accuracy_score(y_true_cat, final),
        "macro_f1": f1_score(y_true_cat, final, labels=CATEGORIES, average="macro", zero_division=0),
        **{f"f1_{c}": f1_score(y_true_cat, final, labels=[c], average="macro", zero_division=0) for c in CATEGORIES},
    }
    return scores, cm
