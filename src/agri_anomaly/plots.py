"""Figures for the report (matplotlib, saved as PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

CAT_COLORS = {"outlier": "#d1495b", "fault": "#edae49", "falsification": "#00798c"}


def roc_pr_curves(y: np.ndarray, scores: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        p, r, _ = precision_recall_curve(y, s)
        axes[0].plot(fpr, tpr, label=name, lw=1.5)
        axes[1].plot(r, p, label=name, lw=1.5)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="ROC curve (test)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].axhline(y.mean(), color="k", ls="--", lw=0.8, label="random")
    axes[1].set(title="Precision-recall curve (test)", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def heatmap(table: pd.DataFrame, title: str, out: Path, fmt: str = ".2f", cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(1.2 * table.shape[1] + 3, 0.5 * table.shape[0] + 2))
    im = ax.imshow(table.to_numpy(dtype=float), cmap=cmap, aspect="auto", vmin=0,
                   vmax=None if fmt == "d" else 1)
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    vmax = np.nanmax(table.to_numpy(dtype=float)) or 1
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, format(int(v) if fmt == "d" else v, fmt), ha="center", va="center",
                        color="white" if (v / vmax if fmt == "d" else v) < 0.6 else "black", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def timeline(frame: pd.DataFrame, sensor: str, variable: str, score: np.ndarray, threshold: float,
             model_name: str, out: Path) -> None:
    """Measured vs original value, attacks shaded by category, score with threshold."""
    m = (frame["sensor_id"] == sensor).to_numpy()
    f = frame[m]
    s = score[m]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    if f"{variable}_clean" in f:
        a1.plot(f["timestamp"], f[f"{variable}_clean"], color="0.7", lw=1, label="original")
    a1.plot(f["timestamp"], f[variable], color="0.1", lw=0.8, label="received")
    flagged = s >= threshold
    a1.scatter(f["timestamp"][flagged], f[variable][flagged], s=6, color="red", zorder=3, label="flagged")
    for _, ev in f[f["event_id"] >= 0].groupby("event_id"):
        for ax in (a1, a2):
            ax.axvspan(ev["timestamp"].iloc[0], ev["timestamp"].iloc[-1], alpha=0.25,
                       color=CAT_COLORS.get(ev["category"].iloc[0], "grey"))
        a1.text(ev["timestamp"].iloc[0], a1.get_ylim()[1], ev["scenario"].iloc[0], fontsize=7, va="top")
    a2.plot(f["timestamp"], s, lw=0.8)
    a2.axhline(threshold, color="red", ls="--", lw=0.8)
    a1.set(title=f"{sensor} - {variable} - detector: {model_name}", ylabel=variable)
    a2.set(ylabel="anomaly score")
    a1.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
