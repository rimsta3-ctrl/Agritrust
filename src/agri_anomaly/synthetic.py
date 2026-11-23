"""Synthetic greenhouse data with the same shape as the real dataset.

Used to run and test the whole pipeline before the real file is available. It mimics:
  * a shared daily temperature cycle with day-to-day weather changes (sunny / cloudy days),
  * sensor-specific offsets (position in the greenhouse), quantised readings (0.1 degC, 0.5 %RH),
  * humidity anti-correlated with temperature,
  * LoRaWAN metadata: RSSI slowly decreasing as plants grow, SNR,
  * irregular timing, ~0.5 % lost messages, a few battery outages, duplicates and glitches.
It is NOT a substitute for the real data in the report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate(n_sensors: int = 12, days: int = 40, freq: str = "5min",
             start: str = "2020-04-17 06:40:00", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grid = pd.date_range(start, periods=int(pd.Timedelta(days=days) / pd.Timedelta(freq)), freq=freq)
    n = len(grid)
    hours = grid.hour.to_numpy() + grid.minute.to_numpy() / 60
    day_idx = (grid.normalize() - grid[0].normalize()).days.to_numpy()

    # shared weather: daily mean follows a slow random walk, daily amplitude = sunny vs cloudy
    daily_mean = 21 + np.cumsum(rng.normal(0, 0.6, days + 1)).clip(-4, 4) + np.linspace(0, 3, days + 1)
    daily_amp = rng.uniform(2.5, 7.0, days + 1)
    shape = np.sin((hours - 8) / 24 * 2 * np.pi)  # peak around 14:00
    shared_t = daily_mean[day_idx] + daily_amp[day_idx] * shape
    shared_t = pd.Series(shared_t).rolling(6, min_periods=1, center=True).mean().to_numpy()

    frames = []
    for k in range(n_sensors):
        sid = f"AF{16 + k}"
        offset = rng.normal(0, 0.7)
        gain = rng.normal(1.0, 0.08)
        ar = np.zeros(n)
        eps = rng.normal(0, 0.06, n)
        for i in range(1, n):
            ar[i] = 0.95 * ar[i - 1] + eps[i]
        temp = shared_t[0] + gain * (shared_t - shared_t[0]) + offset + ar + rng.normal(0, 0.05, n)
        hum = 78 - 2.0 * (temp - daily_mean[day_idx]) + rng.normal(0, 0.8, n) + rng.normal(0, 2)
        hum = hum.clip(35, 99)
        rssi = rng.normal(-95, 6) - np.linspace(0, 6, n) + rng.normal(0, 2.0, n)
        snr = (rng.normal(6, 1.5) + rng.normal(0, 1.5, n)).clip(-15, 12)

        temp = np.round(temp, 1)
        hum = np.round(hum * 2) / 2
        ts = grid + pd.to_timedelta(rng.integers(-20, 21, n), unit="s")
        df = pd.DataFrame({"timestamp": ts, "sensor_id": sid, "temperature": temp, "humidity": hum,
                           "rssi": np.round(rssi), "snr": np.round(snr, 1)})

        keep = rng.random(n) > 0.005                       # lost messages
        if k in (1, 4):                                    # battery outage of a few hours
            s = rng.integers(n // 4, 3 * n // 4)
            keep[s:s + rng.integers(24, 72)] = False
        if k == n_sensors - 1:                             # sensor removed before the end
            keep[int(0.85 * n):] = False
        df = df[keep]
        glitch = rng.random(len(df)) < 0.0003              # hardware glitches (impossible values)
        df.loc[glitch, "temperature"] = rng.choice([-40.0, 85.0], glitch.sum())
        dup = df.sample(frac=0.001, random_state=int(rng.integers(1e9)))
        frames.append(pd.concat([df, dup]))

    out = pd.concat(frames, ignore_index=True).sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    return out
