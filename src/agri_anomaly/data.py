"""DATA LAYER: load the raw file, audit it, clean it, put it on a regular time grid, split it.

Output format used everywhere else ("long" format, one row per sensor and time step):
    timestamp | sensor_id | temperature | humidity | rssi | snr | received | interpolated
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

STANDARD_KEYS = ["timestamp", "sensor_id", "temperature", "humidity", "rssi", "snr"]

# regex patterns tried in order (first match wins) on lower-cased column names
_PATTERNS = {
    "temperature": [r"^temp(erature)?\b", r"temp"],
    "humidity": [r"^hum(idity)?\b", r"humid", r"^rh\b", r"\bhum"],
    "rssi": [r"rssi"],
    "snr": [r"snr"],
    "timestamp": [r"^timestamp$", r"^date ?time$", r"^time$", r"timestamp", r"time", r"date", r"received"],
    "sensor_id": [r"^sensor(_| )?id$", r"^dev(ice)?(_| )?(id|eui)$", r"^node(_| )?id$", r"sensor", r"device",
                  r"node", r"^id$", r"^name$"],
}
_DETECTION_ORDER = ["temperature", "humidity", "rssi", "snr", "timestamp", "sensor_id"]


class SchemaError(ValueError):
    pass


# ----------------------------------------------------------------------------- loading
def _detect_column(columns, key: str, used: set) -> str | None:
    candidates = {c: str(c).strip().lower() for c in columns if c not in used}
    for pattern in _PATTERNS[key]:
        for col, low in candidates.items():
            if re.search(pattern, low):
                return col
    return None


def _parse_time(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s
    elif pd.api.types.is_numeric_dtype(s):
        med = s.dropna().median()
        if med > 1e11:      # epoch milliseconds
            out = pd.to_datetime(s, unit="ms", errors="coerce")
        elif med > 1e8:     # epoch seconds
            out = pd.to_datetime(s, unit="s", errors="coerce")
        else:               # Excel serial days
            out = pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
    else:
        out = pd.to_datetime(s.astype(str), errors="coerce", format="mixed")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert(None)
    return out


def _to_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")


def standardize_columns(raw: pd.DataFrame, mapping: dict, sheet_name: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Rename the raw columns to the standard names. Returns (frame, mapping actually used)."""
    used: set = set()
    found: dict = {}
    for key in _DETECTION_ORDER:
        spec = mapping.get(key, "auto")
        if spec in (None, "none", "null"):
            continue
        if spec == "auto":
            col = _detect_column(raw.columns, key, used)
        elif isinstance(spec, list):
            missing = [c for c in spec if c not in raw.columns]
            col = None if missing else spec
        else:
            col = spec if spec in raw.columns else None
            if col is None:
                raise SchemaError(f"column '{spec}' (for {key}) not found. Available: {list(raw.columns)}")
        if col is not None:
            found[key] = col
            used.update(col if isinstance(col, list) else [col])

    if "timestamp" not in found:
        raise SchemaError(f"no timestamp column detected in {list(raw.columns)}")
    if not any(k in found for k in ("temperature", "humidity")):
        raise SchemaError(f"no temperature/humidity column detected in {list(raw.columns)}")

    out = pd.DataFrame(index=raw.index)
    ts_col = found["timestamp"]
    if isinstance(ts_col, list):
        out["timestamp"] = _parse_time(raw[ts_col].astype(str).agg(" ".join, axis=1))
    else:
        out["timestamp"] = _parse_time(raw[ts_col])

    if "sensor_id" in found:
        out["sensor_id"] = raw[found["sensor_id"]].astype(str)
    elif sheet_name is not None:
        out["sensor_id"] = str(sheet_name)
        found["sensor_id"] = f"<sheet name: {sheet_name}>"
    else:
        raise SchemaError("no sensor column detected and the file has a single sheet")
    out["sensor_id"] = out["sensor_id"].str.replace(r"\s+", "", regex=True)

    for key in ("temperature", "humidity", "rssi", "snr"):
        out[key] = _to_numeric(raw[found[key]]) if key in found else np.nan
    return out, found


def load_raw(path: str | Path, mapping: dict, cache: str | Path | None = None) -> pd.DataFrame:
    """Read .xlsx (all sheets), .csv or .parquet and return the standard long format."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download Greenhouse-1.xlsx from https://zenodo.org/records/5793685 "
            f"into data/raw/, or run with --synthetic."
        )
    if cache is not None:
        cache = Path(cache)
        if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
            log.info("Reading cached copy %s", cache)
            return pd.read_parquet(cache)

    log.info("Reading %s (an .xlsx of this size can take a minute)...", path.name)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
    elif suffix == ".csv":
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=";")
        sheets = {path.stem: df}
    elif suffix == ".parquet":
        sheets = {path.stem: pd.read_parquet(path)}
    else:
        raise ValueError(f"unsupported file type {suffix}")

    frames, mappings = [], {}
    for name, raw in sheets.items():
        if raw.empty:
            continue
        try:
            frame, used = standardize_columns(raw, mapping, sheet_name=name if len(sheets) > 1 else None)
            frames.append(frame)
            mappings[name] = used
        except SchemaError as e:
            log.warning("Skipping sheet '%s': %s", name, e)
    if not frames:
        raise SchemaError("no usable sheet. Set the column names in config.yaml > data > columns.")

    first = next(iter(mappings.values()))
    log.info("Column mapping detected (first sheet): %s", first)
    df = pd.concat(frames, ignore_index=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


# ----------------------------------------------------------------------------- audit
def audit(df: pd.DataFrame, cfg_data: dict) -> tuple[dict, pd.DataFrame]:
    """Data quality report: volume, coverage, intervals, missing values, duplicates, impossible values."""
    per_sensor = []
    for sid, g in df.groupby("sensor_id"):
        g = g.sort_values("timestamp")
        dt = g["timestamp"].diff().dt.total_seconds() / 60
        row = {
            "sensor_id": sid,
            "messages": len(g),
            "first": g["timestamp"].min(),
            "last": g["timestamp"].max(),
            "median_interval_min": round(float(dt.median()), 2) if len(g) > 1 else np.nan,
            "p95_interval_min": round(float(dt.quantile(0.95)), 2) if len(g) > 1 else np.nan,
            "gaps_over_1h": int((dt > 60).sum()),
            "duplicates": int(g.duplicated(["timestamp"]).sum()),
        }
        for v in ("temperature", "humidity", "rssi", "snr"):
            row[f"{v}_missing_%"] = round(100 * g[v].isna().mean(), 2)
        for v, (lo, hi) in cfg_data["physical_range"].items():
            row[f"{v}_out_of_range"] = int(((g[v] < lo) | (g[v] > hi)).sum())
        per_sensor.append(row)
    per_sensor = pd.DataFrame(per_sensor)

    summary = {
        "rows": int(len(df)),
        "sensors": int(df["sensor_id"].nunique()),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "unparsable_timestamps": int(df["timestamp"].isna().sum()),
        "median_interval_min": float(per_sensor["median_interval_min"].median()),
        "describe": df[["temperature", "humidity", "rssi", "snr"]].describe().round(2),
    }
    return summary, per_sensor


def audit_markdown(summary: dict, per_sensor: pd.DataFrame, cleaning: dict | None = None) -> str:
    lines = ["# Data audit", ""]
    for k in ("rows", "sensors", "start", "end", "unparsable_timestamps", "median_interval_min"):
        lines.append(f"- **{k}**: {summary[k]}")
    lines += ["", "## Variables", "", summary["describe"].to_markdown(), ""]
    if cleaning:
        lines += ["## Cleaning", ""] + [f"- **{k}**: {v}" for k, v in cleaning.items()] + [""]
    lines += ["## Per sensor", "", per_sensor.to_markdown(index=False), ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- cleaning
def hampel_mask(x: pd.Series, window: int, n_sigmas: float, min_abs: float) -> pd.Series:
    """True where x is an isolated spike (offline cleaning, centred window)."""
    med = x.rolling(window, center=True, min_periods=1).median()
    dev = (x - med).abs()
    mad = dev.rolling(window, center=True, min_periods=1).median()
    thr = np.maximum(n_sigmas * 1.4826 * mad, min_abs)
    return dev > thr


def clean(df: pd.DataFrame, cfg_data: dict) -> tuple[pd.DataFrame, dict]:
    stats = {"rows_in": int(len(df))}
    df = df.dropna(subset=["timestamp", "sensor_id"]).copy()
    stats["dropped_no_time_or_sensor"] = stats["rows_in"] - len(df)

    if cfg_data.get("start_after"):
        n = len(df)
        df = df[df["timestamp"] >= pd.Timestamp(cfg_data["start_after"])]
        stats["dropped_before_start"] = n - len(df)
    if cfg_data.get("end_before"):
        n = len(df)
        df = df[df["timestamp"] < pd.Timestamp(cfg_data["end_before"])]
        stats["dropped_after_end"] = n - len(df)

    n = len(df)
    df = df.sort_values(["sensor_id", "timestamp"]).drop_duplicates(["sensor_id", "timestamp"], keep="first")
    stats["dropped_duplicates"] = n - len(df)

    for v, (lo, hi) in cfg_data["physical_range"].items():
        bad = (df[v] < lo) | (df[v] > hi)
        df.loc[bad, v] = np.nan
        stats[f"{v}_out_of_range_removed"] = int(bad.sum())

    h = cfg_data["hampel"]
    for v in cfg_data["variables"]:
        mask = df.groupby("sensor_id", group_keys=False)[v].apply(
            lambda s: hampel_mask(s, h["window"], h["n_sigmas"], h["min_abs"].get(v, 0.0))
        )
        df.loc[mask.reindex(df.index, fill_value=False), v] = np.nan
        stats[f"{v}_spikes_removed"] = int(mask.sum())

    df = df.reset_index(drop=True)
    stats["rows_out"] = int(len(df))
    return df, stats


# ----------------------------------------------------------------------------- regular grid
_NICE_MINUTES = [1, 2, 5, 10, 15, 30, 60]


def infer_frequency(df: pd.DataFrame) -> str:
    """Median interval between messages, snapped to a 'nice' frequency."""
    dt = df.sort_values(["sensor_id", "timestamp"]).groupby("sensor_id")["timestamp"].diff()
    med = dt.dropna().median().total_seconds() / 60
    nice = min(_NICE_MINUTES, key=lambda m: abs(m - med))
    log.info("Median message interval = %.2f min -> resampling frequency %d min", med, nice)
    return f"{nice}min"


def resample_regular(df: pd.DataFrame, freq: str, value_cols: list[str], max_interp_steps: int,
                     min_coverage: float = 0.0) -> pd.DataFrame:
    """One row per sensor and time step. Short gaps are interpolated, long gaps stay NaN."""
    out = []
    t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
    global_steps = (t_max - t_min) / pd.Timedelta(freq) + 1
    primary = value_cols[0]
    for sid, g in df.groupby("sensor_id", sort=True):
        g = g.set_index("timestamp").sort_index()
        grid = g[value_cols].resample(freq).mean()
        grid["received"] = g[primary].resample(freq).size().reindex(grid.index, fill_value=0) > 0
        before = grid[primary].isna()
        grid[value_cols] = grid[value_cols].interpolate(method="time", limit=max_interp_steps, limit_area="inside")
        grid["interpolated"] = before & grid[primary].notna()
        coverage = grid[primary].notna().sum() / global_steps
        if coverage < min_coverage:
            log.warning("Dropping sensor %s: coverage %.0f %% < %.0f %%", sid, 100 * coverage, 100 * min_coverage)
            continue
        grid["sensor_id"] = sid
        out.append(grid.reset_index())
    res = pd.concat(out, ignore_index=True)
    return res[["timestamp", "sensor_id", *value_cols, "received", "interpolated"]]


# ----------------------------------------------------------------------------- split
def split_by_time(df: pd.DataFrame, ratios: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    """Chronological split on the global timeline (same cut dates for every sensor)."""
    ts = np.sort(df["timestamp"].unique())
    n = len(ts)
    t1 = ts[int(n * ratios["train"])]
    t2 = ts[int(n * (ratios["train"] + ratios["val"]))]
    splits = {
        "train": df[df["timestamp"] < t1],
        "val": df[(df["timestamp"] >= t1) & (df["timestamp"] < t2)],
        "test": df[df["timestamp"] >= t2],
    }
    splits = {k: v.reset_index(drop=True) for k, v in splits.items()}
    cuts = {"train_end": str(pd.Timestamp(t1)), "val_end": str(pd.Timestamp(t2))}
    return splits, cuts
