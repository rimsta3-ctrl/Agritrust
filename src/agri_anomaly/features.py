"""FEATURE LAYER: turn each (sensor, time step) row into a feature vector.

Design rules
  * Causal: rolling windows only look at the current and past steps of the same sensor.
    Cross-sensor features use the other sensors at the SAME time step, never the future.
  * Relative: no absolute level (e.g. raw temperature) is given to the models, because the test
    period (summer) is warmer than the training period (spring). Features describe deviations
    and dynamics, which stay comparable across seasons.
  * Fitted on training data only: the per-sensor "usual position relative to the others"
    (one value per sensor and hour of day) is learned on the clean training split.

Features, for each measured variable v (temperature, humidity):
    v_diff        first difference                                  -> spikes, random, jumps
    v_std_s       rolling std, short window                         -> random (high), stuck (zero)
    v_std_l       rolling std, long window
    v_dev_med     value - rolling median (long)                     -> spikes, offsets onset
    v_z           (value - rolling mean) / rolling std (long)
    v_xdev        value - median of all sensors - usual offset      -> offset, drift, injection, replay
    v_xdev_mean   rolling mean of v_xdev (long)                     -> sustained deviations (drift)
    v_xdiff       own change - median change of the other sensors   -> moves "alone"
    v_xcorr       rolling corr. of own changes with the others'     -> replay / injection decorrelate
    v_flat        log(1 + number of identical consecutive values)   -> stuck
Network context (if present): rssi_dev, snr_dev (deviation from rolling median)
Time context: hour_sin, hour_cos, gap (log of steps since the last received message)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .attacks import LABEL_COLUMNS
from .utils import to_steps

EPS = 1e-6


def _groll(s: pd.Series, groups: pd.Series, window: int, how: str, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or max(2, window // 3)
    return s.groupby(groups, sort=False).transform(lambda x: getattr(x.rolling(window, min_periods=mp), how)())


class FeatureBuilder:
    def __init__(self, variables: list[str], network_variables: list[str], freq: str,
                 short_window: str = "30min", long_window: str = "3h", corr_window: str = "2h"):
        self.variables = list(variables)
        self.network_variables = list(network_variables)
        self.freq = freq
        self.ws = to_steps(short_window, freq)
        self.wl = to_steps(long_window, freq)
        self.wc = to_steps(corr_window, freq)
        self.offsets_: dict = {}
        self.global_offset_: dict = {}
        self.net_used_: list[str] = []
        self.feature_names_: list[str] = []

    @classmethod
    def from_config(cls, cfg: dict, freq: str) -> "FeatureBuilder":
        f = cfg["features"]
        return cls(cfg["data"]["variables"], cfg["data"].get("network_variables", []), freq,
                   f["short_window"], f["long_window"], f["corr_window"])

    # ------------------------------------------------------------------ fit
    def fit(self, df_train_clean: pd.DataFrame) -> "FeatureBuilder":
        df = df_train_clean.sort_values(["sensor_id", "timestamp"])
        hour = df["timestamp"].dt.hour
        for v in self.variables:
            xdev = df[v] - df.groupby("timestamp")[v].transform("median")
            self.offsets_[v] = xdev.groupby([df["sensor_id"], hour]).median()
            self.global_offset_[v] = float(xdev.median())
        self.net_used_ = [c for c in self.network_variables if c in df.columns and df[c].notna().mean() > 0.5]
        return self

    # ------------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame, keep_missing: bool = False) -> pd.DataFrame:
        if not self.offsets_:
            raise RuntimeError("call fit() on the clean training split first")
        df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
        g = df["sensor_id"]
        hour = df["timestamp"].dt.hour
        feats = {}

        for v in self.variables:
            x = df[v]
            diff = x.groupby(g, sort=False).diff()
            mean_l = _groll(x, g, self.wl, "mean")
            std_l = _groll(x, g, self.wl, "std")
            med_l = _groll(x, g, self.wl, "median")
            feats[f"{v}_diff"] = diff
            feats[f"{v}_std_s"] = _groll(x, g, self.ws, "std")
            feats[f"{v}_std_l"] = std_l
            feats[f"{v}_dev_med"] = x - med_l
            feats[f"{v}_z"] = (x - mean_l) / (std_l + 0.05)

            cross_med = df.groupby("timestamp")[v].transform("median")
            idx = pd.MultiIndex.from_arrays([g, hour])
            usual = self.offsets_[v].reindex(idx).to_numpy()
            usual = np.where(np.isnan(usual), self.global_offset_[v], usual)
            xdev = x - cross_med - usual
            feats[f"{v}_xdev"] = xdev
            feats[f"{v}_xdev_mean"] = _groll(xdev, g, self.wl, "mean")

            cross_diff = diff.groupby(df["timestamp"]).transform("median")
            feats[f"{v}_xdiff"] = diff - cross_diff
            feats[f"{v}_xcorr"] = self._rolling_corr(diff, cross_diff, g, self.wc)

            same = x.eq(x.groupby(g, sort=False).shift()) & x.notna()
            block = (~same).groupby(g, sort=False).cumsum()
            run = same.astype(int).groupby([g, block], sort=False).cumsum()
            feats[f"{v}_flat"] = np.log1p(run)

        for c in self.net_used_:
            feats[f"{c}_dev"] = df[c] - _groll(df[c], g, self.wl, "median")

        hfloat = hour + df["timestamp"].dt.minute / 60
        feats["hour_sin"] = np.sin(2 * np.pi * hfloat / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * hfloat / 24)
        if "received" in df.columns:
            rec = df["received"].astype(bool)
            block = rec.groupby(g, sort=False).cumsum()
            feats["gap"] = np.log1p(rec.groupby([g, block], sort=False).cumcount())
        else:
            feats["gap"] = 0.0

        F = pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
        self.feature_names_ = list(F.columns)
        meta_cols = ["timestamp", "sensor_id", *self.variables,
                     *[c for c in df.columns if c.endswith("_clean")],
                     *[c for c in LABEL_COLUMNS if c in df.columns]]
        out = pd.concat([df[meta_cols], F], axis=1)
        if not keep_missing:
            out = out[df[self.variables].notna().all(axis=1).to_numpy()].reset_index(drop=True)
        return out

    @staticmethod
    def _rolling_corr(a: pd.Series, b: pd.Series, g: pd.Series, w: int) -> pd.Series:
        a = a.fillna(0.0)
        b = b.fillna(0.0)
        ma, mb = _groll(a, g, w, "mean"), _groll(b, g, w, "mean")
        mab, maa, mbb = _groll(a * b, g, w, "mean"), _groll(a * a, g, w, "mean"), _groll(b * b, g, w, "mean")
        cov = mab - ma * mb
        va, vb = (maa - ma ** 2).clip(lower=0), (mbb - mb ** 2).clip(lower=0)
        denom = np.sqrt(va * vb)
        corr = np.where(denom > EPS, cov / np.maximum(denom, EPS), 0.0)
        return pd.Series(np.clip(corr, -1, 1), index=a.index)
