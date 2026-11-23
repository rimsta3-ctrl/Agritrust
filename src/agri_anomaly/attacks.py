"""SIMULATION LAYER: inject controlled faults and attacks, and keep the ground truth.

Each injected event modifies a window of one sensor and is written to an event log.
Every row of the returned frame gets four label columns:
    label      0 = normal, 1 = anomalous
    category   normal | outlier | fault | falsification     (the 4 cases of the project sheet)
    scenario   normal | spike | random | stuck | drift | offset | injection | replay
    event_id   -1 for normal rows, otherwise the id in the event log

Scenario -> category (a modelling choice, justify it in the report):
    outlier        spike      : 1-2 isolated points far from the normal level
    fault          random     : erratic values (failing sensor)
                   stuck      : sensor frozen on its last value
                   drift      : slow, growing bias (ageing / calibration loss)
    falsification  offset     : real values shifted by a constant (artificial modification)
                   injection  : fabricated but plausible values (typical daily profile of the sensor)
                   replay     : old real measurements re-sent (same time of day, d days earlier)
Limitation: the same shape can come from a fault or an attack (e.g. a frozen value). The
category is defined by how the data was produced, and context is what separates them.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .utils import to_steps

log = logging.getLogger(__name__)

LABEL_COLUMNS = ["label", "category", "scenario", "event_id"]


@dataclass
class AttackEvent:
    event_id: int
    split: str
    sensor_id: str
    variables: str
    scenario: str
    category: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_points: int
    intensity: float | None
    params: str


class AttackSimulator:
    def __init__(self, cfg_attacks: dict, variables: list[str], freq: str, seed: int = 0):
        self.cfg = cfg_attacks
        self.scenarios = cfg_attacks["scenarios"]
        self.variables = list(variables)
        self.freq = freq
        self.steps_per_day = to_steps("1D", freq)
        self.gap = to_steps(cfg_attacks.get("min_gap", "3h"), freq)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ setup
    def _prepare(self, df: pd.DataFrame) -> None:
        df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
        self.df = df
        # "clean" = original measurements (kept when the frame was already attacked)
        self.clean = {v: (df[f"{v}_clean"] if f"{v}_clean" in df.columns else df[v]).to_numpy(dtype=float).copy()
                      for v in self.variables}
        self.vals = {v: df[v].to_numpy(dtype=float).copy() for v in self.variables}
        n = len(df)
        if "label" in df.columns:   # already attacked frame: keep its labels
            self.label = df["label"].to_numpy(dtype=np.int8).copy()
            self.category = df["category"].to_numpy(dtype=object).copy()
            self.scenario = df["scenario"].to_numpy(dtype=object).copy()
            self.event_id = df["event_id"].to_numpy(dtype=np.int32).copy()
        else:
            self.label = np.zeros(n, dtype=np.int8)
            self.category = np.full(n, "normal", dtype=object)
            self.scenario = np.full(n, "normal", dtype=object)
            self.event_id = np.full(n, -1, dtype=np.int32)

        self.slices, self.stats = {}, {}
        hours = (df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60).to_numpy()
        self.hours = hours
        for sid, idx in df.groupby("sensor_id", sort=False).indices.items():
            a, b = int(idx.min()), int(idx.max()) + 1
            self.slices[sid] = (a, b)
            st = {}
            for v in self.variables:
                x = self.clean[v][a:b]
                std = np.nanstd(x)
                step = np.nanstd(np.diff(x))
                bins = np.floor(hours[a:b] * 2) / 2   # 30-min profile
                prof = pd.Series(x).groupby(bins).mean()
                st[v] = {"mean": np.nanmean(x), "std": std if std > 0 else 1.0,
                         "step": step if np.isfinite(step) and step > 0 else 0.1,
                         "profile_h": prof.index.to_numpy(), "profile_v": prof.to_numpy()}
            self.stats[sid] = st
        self.occupied = {sid: self.label[a:b].astype(bool).copy() for sid, (a, b) in self.slices.items()}
        self.events: list[AttackEvent] = []
        self._next_id = int(self.event_id.max()) + 1 if (self.event_id >= 0).any() else 0

    def _frame(self) -> pd.DataFrame:
        out = self.df.copy()
        for v in self.variables:
            out[f"{v}_clean"] = self.clean[v]
            out[v] = self.vals[v]
        out["label"] = self.label
        out["category"] = self.category
        out["scenario"] = self.scenario
        out["event_id"] = self.event_id
        return out

    # ------------------------------------------------------------------ sampling
    def _sample_intensity(self, scenario: str) -> float | None:
        rng_ = self.scenarios[scenario].get("intensity")
        if rng_ is None:
            return None
        if scenario == "replay":
            return float(self.rng.integers(int(rng_[0]), int(rng_[1]) + 1))
        return float(self.rng.uniform(*rng_))

    def _window_ok(self, sid: str, r: int, n: int, vars_: list[str], back: int = 0) -> bool:
        a, b = self.slices[sid]
        length = b - a
        if r < max(1, back) or r + n > length:
            return False
        lo, hi = max(0, r - self.gap), min(length, r + n + self.gap)
        if self.occupied[sid][lo:hi].any():
            return False
        for v in vars_:
            if np.isnan(self.vals[v][a + r - 1: a + r + n]).any():
                return False
            if back and np.isnan(self.clean[v][a + r - back: a + r - back + n]).any():
                return False
        return True

    # ------------------------------------------------------------------ public API
    def inject(self, df: pd.DataFrame, split: str, n_per_scenario: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Inject n_per_scenario events of every scenario at random places."""
        self._prepare(df)
        sensors = list(self.slices)
        for scenario, spec in self.scenarios.items():
            dmin, dmax = (to_steps(d, self.freq) for d in spec["duration"])
            done = 0
            for _ in range(n_per_scenario * 200):
                if done >= n_per_scenario:
                    break
                sid = sensors[self.rng.integers(len(sensors))]
                n = int(self.rng.integers(dmin, dmax + 1))
                vars_ = self.variables if spec.get("variables", "one") == "all" else \
                    [self.variables[self.rng.integers(len(self.variables))]]
                intensity = self._sample_intensity(scenario)
                back = int(intensity * self.steps_per_day) if scenario == "replay" else 0
                a, b = self.slices[sid]
                if b - a < n + back + 2:
                    continue
                r = int(self.rng.integers(max(1, back), b - a - n + 1))
                if not self._window_ok(sid, r, n, vars_, back):
                    continue
                self._apply(split, sid, scenario, r, n, vars_, intensity)
                done += 1
            if done < n_per_scenario:
                log.warning("[%s] only %d/%d '%s' events could be placed", split, done, n_per_scenario, scenario)
        return self._frame(), self.events_frame()

    def inject_one(self, df: pd.DataFrame, sensor_id: str, scenario: str, start: pd.Timestamp,
                   n_steps: int, intensity: float | None = None, variable: str | None = None,
                   split: str = "custom") -> tuple[pd.DataFrame, pd.DataFrame]:
        """Inject a single, fully specified event (used by the dashboard)."""
        self._prepare(df)
        spec = self.scenarios[scenario]
        if variable is None:
            vars_ = self.variables if spec.get("variables", "one") == "all" else [self.variables[0]]
        else:
            vars_ = self.variables if variable == "all" else [variable]
        if intensity is None and spec.get("intensity") is not None:
            intensity = float(np.mean(spec["intensity"]))
        a, b = self.slices[sensor_id]
        ts = self.df["timestamp"].to_numpy()[a:b]
        r = int(np.searchsorted(ts, np.datetime64(pd.Timestamp(start))))
        r = max(r, 1)
        back = int(intensity * self.steps_per_day) if scenario == "replay" else 0
        if scenario == "replay" and r < back:
            raise ValueError("not enough history before this start for a replay; pick a later start "
                             "or fewer days back")
        n_steps = min(n_steps, b - a - r)
        self.occupied[sensor_id][:] = False   # allow overlap with existing events in the dashboard
        self._apply(split, sensor_id, scenario, r, n_steps, vars_, intensity)
        return self._frame(), self.events_frame()

    def events_frame(self) -> pd.DataFrame:
        cols = list(AttackEvent.__dataclass_fields__)
        return pd.DataFrame([asdict(e) for e in self.events], columns=cols)

    # ------------------------------------------------------------------ scenarios
    def _apply(self, split, sid, scenario, r, n, vars_, intensity):
        a, _ = self.slices[sid]
        i0, i1 = a + r, a + r + n
        sign = float(self.rng.choice([-1.0, 1.0]))
        params = {"sign": sign}
        for v in vars_:
            st = self.stats[sid][v]
            x = self.vals[v]
            if scenario == "spike":
                x[i0:i1] = x[i0:i1] + sign * intensity * st["std"]
            elif scenario == "random":
                x[i0:i1] = self.rng.uniform(st["mean"] - intensity * st["std"],
                                            st["mean"] + intensity * st["std"], n)
            elif scenario == "stuck":
                x[i0:i1] = x[i0 - 1]
            elif scenario == "drift":
                x[i0:i1] = x[i0:i1] + sign * intensity * st["std"] * np.linspace(1 / n, 1.0, n)
            elif scenario == "offset":
                x[i0:i1] = x[i0:i1] + sign * intensity * st["std"]
            elif scenario == "injection":
                h = self.hours[i0:i1]
                fab = np.interp(h, st["profile_h"], st["profile_v"], period=24)
                fab = fab + self.rng.normal(0, 0.5 * st["step"], n)
                fab = fab + (x[i0 - 1] - fab[0]) * np.linspace(1.0, 0.0, n)   # stealthy start
                x[i0:i1] = fab
            elif scenario == "replay":
                back = int(intensity * self.steps_per_day)
                x[i0:i1] = self.clean[v][i0 - back:i1 - back]
                params["days_back"] = int(intensity)
            else:
                raise ValueError(f"unknown scenario {scenario}")
            if v == "humidity":
                x[i0:i1] = np.clip(x[i0:i1], 0, 100)

        eid = self._next_id
        self._next_id += 1
        cat = self.scenarios[scenario]["category"]
        self.label[i0:i1] = 1
        self.category[i0:i1] = cat
        self.scenario[i0:i1] = scenario
        self.event_id[i0:i1] = eid
        self.occupied[sid][r:r + n] = True
        ts = self.df["timestamp"]
        self.events.append(AttackEvent(eid, split, sid, "+".join(vars_), scenario, cat,
                                       ts.iloc[i0], ts.iloc[i1 - 1], n, intensity, json.dumps(params)))
