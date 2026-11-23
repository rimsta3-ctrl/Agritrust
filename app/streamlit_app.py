"""PROTOTYPE: dashboard to watch the sensors and the alerts raised by the detectors.

Run (after the pipeline has been run once):
    streamlit run app/streamlit_app.py

It loads the saved feature builder and models; nothing is retrained here.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from agri_anomaly.attacks import AttackSimulator  # noqa: E402
from agri_anomaly.config import load_config, resolve_path  # noqa: E402
from agri_anomaly.pipeline import _feature_names, _meta, load_models, pick_stage1  # noqa: E402

CAT_COLORS = {"outlier": "rgba(209,73,91,0.25)", "fault": "rgba(237,174,73,0.30)",
              "falsification": "rgba(0,121,140,0.25)"}
TYPE_COLORS = {"outlier": "#d1495b", "fault": "#e09f1f", "falsification": "#00798c"}

st.set_page_config(page_title="Greenhouse IoT - anomaly & falsification detector", layout="wide")


# ----------------------------------------------------------------------------- loading
@st.cache_resource
def load_everything():
    cfg = load_config(ROOT / "config.yaml")
    try:
        meta = _meta(cfg)
        fb = joblib.load(resolve_path(cfg, "models") / "feature_builder.joblib")
        s1, s2 = load_models(cfg)
        names = _feature_names(cfg)
    except FileNotFoundError:
        return None
    if not s1:
        return None
    return cfg, meta, fb, s1, s2, names


@st.cache_data
def load_split(name: str) -> pd.DataFrame:
    cfg = load_config(ROOT / "config.yaml")
    return pd.read_parquet(resolve_path(cfg, "processed") / f"{name}.parquet")


@st.cache_data(show_spinner="Computing features...")
def features_for(mode: str, sensor: str, scenario: str, start: str, steps: int, intensity: float,
                 variable: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg, meta, fb, *_ = load_everything()
    if mode == "simulated":
        frame = load_split("test_attacked")
        events = pd.read_csv(resolve_path(cfg, "reports") / "attack_log.csv", parse_dates=["start", "end"])
        events = events[events["split"] == "test"]
    else:
        sim = AttackSimulator(cfg["attacks"], cfg["data"]["variables"], meta["freq"], seed=cfg["project"]["seed"])
        frame, events = sim.inject_one(load_split("test"), sensor, scenario, pd.Timestamp(start), steps,
                                       intensity, variable)
    return fb.transform(frame), events


def score_sensor(F: pd.DataFrame, sensor: str, model_name: str):
    _, _, _, s1, s2, names = load_everything()
    sub = F[F["sensor_id"] == sensor].reset_index(drop=True)
    model = s1[model_name]
    scores = model.score(sub[names], sub["sensor_id"].to_numpy())
    best2 = max(s2, key=lambda k: getattr(s2[k], "val_macro_f1", 0)) if s2 else None
    types = s2[best2].predict(sub[names]) if best2 else np.full(len(sub), "anomaly", dtype=object)
    return sub, scores, types


# ----------------------------------------------------------------------------- figure
def make_figure(sub, variable, scores, thr, types, upto=None):
    if upto is not None:
        keep = np.arange(len(sub)) <= upto
        sub, scores, types = sub[keep], scores[keep], types[keep]
    flagged = scores >= thr
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.05)
    if f"{variable}_clean" in sub:
        fig.add_trace(go.Scattergl(x=sub["timestamp"], y=sub[f"{variable}_clean"], name="original value",
                                   line=dict(color="#bbbbbb", width=1)), 1, 1)
    fig.add_trace(go.Scattergl(x=sub["timestamp"], y=sub[variable], name="received value",
                               line=dict(color="#222222", width=1)), 1, 1)
    for t in ("outlier", "fault", "falsification"):
        m = flagged & (types == t)
        if m.any():
            fig.add_trace(go.Scattergl(x=sub["timestamp"][m], y=sub[variable][m], mode="markers",
                                       name=f"alert: {t}", marker=dict(color=TYPE_COLORS[t], size=6)), 1, 1)
    fig.add_trace(go.Scattergl(x=sub["timestamp"], y=scores, name="anomaly score",
                               line=dict(color="#3b6ea5", width=1)), 2, 1)
    fig.add_hline(y=thr, line=dict(color="red", dash="dash", width=1), row=2, col=1)
    for _, ev in sub[sub["event_id"] >= 0].groupby("event_id"):
        fig.add_vrect(x0=ev["timestamp"].iloc[0], x1=ev["timestamp"].iloc[-1], line_width=0,
                      fillcolor=CAT_COLORS.get(ev["category"].iloc[0], "rgba(128,128,128,0.2)"),
                      annotation_text=ev["scenario"].iloc[0], annotation_position="top left",
                      annotation_font_size=10)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08))
    fig.update_yaxes(title_text=variable, row=1, col=1)
    fig.update_yaxes(title_text="score", row=2, col=1)
    return fig


# ----------------------------------------------------------------------------- page
loaded = load_everything()
st.title("Greenhouse IoT - anomaly & falsification detector")
if loaded is None:
    st.error("No trained models found. Run the pipeline first:  `python scripts/run_all.py --synthetic`  "
             "(or without --synthetic once Greenhouse-1.xlsx is in data/raw/).")
    st.stop()
cfg, meta, fb, s1, s2, names = loaded
st.caption(f"Data source: **{meta['source']}** - time step {meta['freq']} - test period from "
           f"{meta['cuts']['val_end']}. Shaded areas = ground truth (simulated); coloured dots = alerts.")

with st.sidebar:
    st.header("Data")
    mode_label = st.radio("Attacks shown", ["Simulated test attacks", "Inject my own attack"])
    mode = "simulated" if mode_label.startswith("Simulated") else "custom"
    sensor = st.selectbox("Sensor", meta["sensors"])
    variable = st.selectbox("Variable", cfg["data"]["variables"])

    scenario, start, steps, intensity, attack_var = "", "", 0, 0.0, ""
    if mode == "custom":
        st.header("Attack to inject")
        scenario = st.selectbox("Scenario", list(cfg["attacks"]["scenarios"]))
        spec = cfg["attacks"]["scenarios"][scenario]
        st.caption(f"category: **{spec['category']}**")
        test_ts = load_split("test")
        ts = test_ts.loc[test_ts["sensor_id"] == sensor, "timestamp"]
        days = pd.date_range(ts.min().ceil("D"), ts.max().floor("D"), freq="6h")
        start = str(st.select_slider("Start", options=list(days), value=days[len(days) // 2],
                                     format_func=lambda d: d.strftime("%d %b %H:%M")))
        hours = st.slider("Duration (hours)", 0.25, 24.0, 4.0, 0.25)
        steps = max(1, int(pd.Timedelta(hours=hours) / pd.Timedelta(meta["freq"])))
        if spec.get("intensity"):
            lo, hi = spec["intensity"]
            label = "Days back" if scenario == "replay" else "Intensity (x sensor std)"
            intensity = float(st.slider(label, float(lo), float(hi), float(np.mean([lo, hi])),
                                        1.0 if scenario == "replay" else 0.1))
        attack_var = "all" if spec.get("variables") == "all" else variable

    st.header("Detector")
    default = pick_stage1(s1, "best_unsupervised")
    model_name = st.selectbox("Stage-1 model", list(s1), index=list(s1).index(default),
                              format_func=lambda k: f"{k} ({s1[k].kind})")

try:
    F, events = features_for(mode, sensor, scenario, start, steps, intensity, attack_var)
except ValueError as e:
    st.warning(str(e))
    st.stop()
sub, scores, types = score_sensor(F, sensor, model_name)

with st.sidebar:
    calib = float(s1[model_name].threshold)
    lo, hi = float(np.quantile(scores, 0.01)), float(np.quantile(scores, 0.999))
    lo, hi = min(lo, calib), max(hi, calib * 1.5 if calib > 0 else hi)
    thr = st.slider("Alert threshold", lo, hi, calib, (hi - lo) / 200 or 0.01,
                    help="Default = threshold calibrated on the validation split")
    st.header("Live replay")
    speed = st.slider("Steps per frame", 6, 288, 48)
    play = st.button("Play the test period as a live stream")

flagged = scores >= thr
truth = sub["label"].to_numpy() == 1
tp = int((flagged & truth).sum())
c = st.columns(5)
c[0].metric("Time steps", f"{len(sub):,}")
c[1].metric("Alerts", f"{int(flagged.sum()):,}")
c[2].metric("True anomalous steps", f"{int(truth.sum()):,}")
c[3].metric("Precision", f"{tp / max(1, flagged.sum()):.0%}")
c[4].metric("Recall", f"{tp / max(1, truth.sum()):.0%}")

chart = st.empty()
if play:
    for upto in range(0, len(sub), speed):
        chart.plotly_chart(make_figure(sub, variable, scores, thr, types, upto), width="stretch",
                           key=f"live_{upto}")
        time.sleep(0.05)
else:
    chart.plotly_chart(make_figure(sub, variable, scores, thr, types), width="stretch")

st.subheader("Alerts")
alerts = sub.loc[flagged, ["timestamp", variable]].copy()
alerts["score"] = scores[flagged].round(3)
alerts["predicted type"] = types[flagged]
alerts["true category"] = sub.loc[flagged, "category"].to_numpy()
alerts["true scenario"] = sub.loc[flagged, "scenario"].to_numpy()
st.dataframe(alerts, width="stretch", height=260, hide_index=True)

with st.expander("Ground-truth event log for this sensor"):
    st.dataframe(events[events["sensor_id"] == sensor], width="stretch", hide_index=True)
