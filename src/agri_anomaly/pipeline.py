"""The five pipeline stages. Scripts, notebooks and the dashboard all call these functions."""
from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
import pandas as pd

from . import data as D
from . import evaluate as E
from . import plots as P
from .attacks import AttackSimulator
from .config import ensure_dirs, resolve_path
from .features import FeatureBuilder
from .models import BaseDetector, build_stage1, build_stage2
from .synthetic import generate
from .utils import set_seed, timer

log = logging.getLogger("agri_anomaly.pipeline")
SPLITS = ["train", "val", "test"]


def _meta(cfg) -> dict:
    with open(resolve_path(cfg, "processed") / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def _feature_names(cfg) -> list[str]:
    with open(resolve_path(cfg, "features") / "feature_names.json", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================ 1. data
def prepare_data(cfg: dict, synthetic: bool = False) -> dict:
    ensure_dirs(cfg)
    set_seed(cfg["project"]["seed"])
    dc = cfg["data"]
    proc, rep = resolve_path(cfg, "processed"), resolve_path(cfg, "reports")

    if synthetic:
        s = cfg["synthetic"]
        raw = generate(s["n_sensors"], s["days"], s["freq"], s["start"], cfg["project"]["seed"])
        source = "synthetic"
        log.info("Generated synthetic data: %d rows, %d sensors", len(raw), raw["sensor_id"].nunique())
    else:
        raw = D.load_raw(resolve_path(cfg, "raw_file"), dc["columns"], cache=proc / "raw_cache.parquet")
        source = str(resolve_path(cfg, "raw_file").name)

    summary_full, _ = D.audit(raw, dc)
    win = raw.dropna(subset=["timestamp"])
    if dc.get("start_after"):
        win = win[win["timestamp"] >= pd.Timestamp(dc["start_after"])]
    if dc.get("end_before"):
        win = win[win["timestamp"] < pd.Timestamp(dc["end_before"])]
    summary_win, per_sensor = D.audit(win, dc)
    outages = D.network_outages(win)
    cleaned, stats = D.clean(raw, dc)
    freq = D.infer_frequency(cleaned) if dc["resample_freq"] == "auto" else dc["resample_freq"]
    net = [c for c in dc.get("network_variables", []) if raw[c].notna().mean() > 0.5]
    value_cols = list(dc["variables"]) + net
    grid = D.resample_regular(cleaned, freq, value_cols, dc["max_interp_steps"], dc["min_sensor_coverage"])
    stats["grid_rows"] = int(len(grid))
    stats["grid_missing_%"] = round(100 * grid[dc["variables"][0]].isna().mean(), 2)
    stats["interpolated_%"] = round(100 * grid["interpolated"].mean(), 2)
    splits, cuts = D.split_by_time(grid, dc["split"])

    for name, df in splits.items():
        df.to_parquet(proc / f"{name}.parquet", index=False)
    meta = {"source": source, "freq": freq, "value_cols": value_cols, "network_variables": net,
            "sensors": sorted(grid["sensor_id"].unique().tolist()), "cuts": cuts,
            "rows": {k: int(len(v)) for k, v in splits.items()}, "cleaning": stats}
    with open(proc / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    per_sensor.to_csv(rep / "data_audit_sensors.csv", index=False)
    label = f"({dc.get('start_after') or 'start'} -> {dc.get('end_before') or 'end'})"
    (rep / "data_audit.md").write_text(
        D.audit_markdown(summary_full, summary_win, per_sensor, stats, outages, label), encoding="utf-8")
    outages.to_csv(rep / "network_outages.csv", index=False)
    log.info("Data ready: freq=%s, sensors=%d, rows=%s, cuts=%s", freq, len(meta["sensors"]), meta["rows"], cuts)
    return meta


# ============================================================================ 2. simulation
def simulate_attacks(cfg: dict) -> pd.DataFrame:
    meta = _meta(cfg)
    proc, rep = resolve_path(cfg, "processed"), resolve_path(cfg, "reports")
    logs = []
    for k, split in enumerate(SPLITS):
        df = pd.read_parquet(proc / f"{split}.parquet")
        sim = AttackSimulator(cfg["attacks"], cfg["data"]["variables"], meta["freq"],
                              seed=cfg["project"]["seed"] + 100 * (k + 1))
        attacked, events = sim.inject(df, split, cfg["attacks"]["events_per_scenario"][split])
        attacked.to_parquet(proc / f"{split}_attacked.parquet", index=False)
        logs.append(events)
        log.info("[%s] %d events, %.1f %% of rows anomalous", split, len(events), 100 * attacked["label"].mean())
    log_df = pd.concat(logs, ignore_index=True)
    log_df.to_csv(rep / "attack_log.csv", index=False)
    return log_df


# ============================================================================ 3. features
def build_features(cfg: dict) -> list[str]:
    meta = _meta(cfg)
    proc, feat, mdir = resolve_path(cfg, "processed"), resolve_path(cfg, "features"), resolve_path(cfg, "models")
    train_clean = pd.read_parquet(proc / "train.parquet")
    fb = FeatureBuilder.from_config(cfg, meta["freq"]).fit(train_clean)

    tc = train_clean.assign(label=0, category="normal", scenario="normal", event_id=-1)
    frames = {"train_clean": tc}
    for split in SPLITS:
        frames[f"{split}_attacked"] = pd.read_parquet(proc / f"{split}_attacked.parquet")
    for name, df in frames.items():
        with timer(f"features {name}", log):
            F = fb.transform(df)
        F.to_parquet(feat / f"{name}.parquet", index=False)
    joblib.dump(fb, mdir / "feature_builder.joblib")
    with open(feat / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(fb.feature_names_, f, indent=2)
    log.info("%d features: %s", len(fb.feature_names_), fb.feature_names_)
    return fb.feature_names_


# ============================================================================ 4. training
def train_models(cfg: dict) -> pd.DataFrame:
    set_seed(cfg["project"]["seed"])
    feat, mdir, rep = resolve_path(cfg, "features"), resolve_path(cfg, "models"), resolve_path(cfg, "reports")
    names = _feature_names(cfg)
    tc = pd.read_parquet(feat / "train_clean.parquet")
    ta = pd.read_parquet(feat / "train_attacked.parquet")
    va = pd.read_parquet(feat / "val_attacked.parquet")
    thr_cfg = cfg["models"]["threshold"]
    rows = []

    for name, model in build_stage1(cfg).items():
        src = ta if model.kind == "supervised" else tc
        with timer(f"train {name}", log):
            t0 = time.perf_counter()
            model.fit(src[names], src["label"].to_numpy(), src["sensor_id"].to_numpy())
            train_s = time.perf_counter() - t0
        s_val = model.score(va[names], va["sensor_id"].to_numpy())
        model.threshold = E.choose_threshold(s_val, va["label"].to_numpy(), thr_cfg["strategy"], thr_cfg["target_fpr"])
        m = E.binary_metrics(va["label"].to_numpy(), s_val, (s_val >= model.threshold).astype(int))
        model.val_f1 = m["f1"]
        model.save(mdir / "stage1" / f"{name}.joblib")
        rows.append({"stage": 1, "model": name, "kind": model.kind, "trained_on": model.trained_on,
                     "train_rows": len(src), "train_time_s": round(train_s, 2), "threshold": model.threshold,
                     "val_f1": round(m["f1"], 4), "val_pr_auc": round(m["pr_auc"], 4)})

    anom_tr, anom_va = ta[ta["label"] == 1], va[va["label"] == 1]
    for name, clf in build_stage2(cfg).items():
        with timer(f"train {name}", log):
            clf.fit(anom_tr[names], anom_tr["category"].to_numpy())
        pred = clf.predict(anom_va[names])
        table, _ = E.stage2_report(anom_va["category"].to_numpy(), pred)
        clf.val_macro_f1 = float(table.loc["macro avg", "f1-score"])
        (mdir / "stage2").mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, mdir / "stage2" / f"{name}.joblib")
        rows.append({"stage": 2, "model": name, "kind": "supervised", "trained_on": "labelled anomalies",
                     "train_rows": len(anom_tr), "val_f1": round(clf.val_macro_f1, 4)})

    summary = pd.DataFrame(rows)
    summary.to_csv(rep / "training_summary.csv", index=False)
    log.info("\n%s", summary.to_string(index=False))
    return summary


# ============================================================================ 5. evaluation
def load_models(cfg: dict) -> tuple[dict, dict]:
    mdir = resolve_path(cfg, "models")
    s1 = {p.stem: BaseDetector.load(p) for p in sorted((mdir / "stage1").glob("*.joblib"))}
    s2 = {p.stem: joblib.load(p) for p in sorted((mdir / "stage2").glob("*.joblib"))}
    return s1, s2


def pick_stage1(s1: dict, which: str) -> str:
    if which in s1:
        return which
    pool = {k: m for k, m in s1.items() if which != "best_unsupervised" or m.kind == "unsupervised"}
    pool = pool or s1
    return max(pool, key=lambda k: pool[k].val_f1 or 0)


def evaluate_models(cfg: dict) -> dict:
    feat, rep = resolve_path(cfg, "features"), resolve_path(cfg, "reports")
    fig = rep / "figures"
    names = _feature_names(cfg)
    te = pd.read_parquet(feat / "test_attacked.parquet")
    X, y, grp = te[names], te["label"].to_numpy(), te["sensor_id"].to_numpy()
    s1, s2 = load_models(cfg)

    # ---- stage 1: detection
    rows, scen_rows, scores, preds = [], [], {}, {}
    for name, model in s1.items():
        s, us = E.timed_scores(model, X, grp)
        p = (s >= model.threshold).astype(int)
        scores[name], preds[name] = s, p
        rows.append({"model": name, "kind": model.kind, **E.binary_metrics(y, s, p),
                     "inference_us_per_sample": us, "val_f1": model.val_f1})
        sc = E.per_scenario(te, p)
        sc.insert(0, "model", name)
        scen_rows.append(sc)
    res1 = pd.DataFrame(rows).sort_values("f1", ascending=False)
    scen = pd.concat(scen_rows, ignore_index=True)
    res1.to_csv(rep / "results_stage1.csv", index=False)
    scen.to_csv(rep / "results_per_scenario.csv", index=False)
    P.roc_pr_curves(y, scores, fig / "roc_pr_curves.png")
    order = list(cfg["attacks"]["scenarios"])
    piv = scen.pivot(index="model", columns="scenario", values="event_recall").reindex(columns=order)
    P.heatmap(piv.loc[res1["model"]], "Event recall per scenario (test)", fig / "event_recall_per_scenario.png")

    # ---- stage 2: qualification on true anomalies
    anom = te[te["label"] == 1]
    res2_rows = []
    for name, clf in s2.items():
        pred = clf.predict(anom[names])
        table, cm = E.stage2_report(anom["category"].to_numpy(), pred)
        table.insert(0, "model", name)
        res2_rows.append(table.reset_index(names="class"))
        P.heatmap(cm, f"Stage 2 confusion matrix - {name} (test anomalies)", fig / f"stage2_cm_{name}.png",
                  fmt="d", cmap="Blues")
    res2 = pd.concat(res2_rows, ignore_index=True) if res2_rows else pd.DataFrame()
    res2.to_csv(rep / "results_stage2.csv", index=False)

    # ---- end-to-end: stage 1 + stage 2 -> normal / outlier / fault / falsification
    e2e_rows = []
    best2 = max(s2, key=lambda k: getattr(s2[k], "val_macro_f1", 0)) if s2 else None
    for which in cfg["evaluation"]["end_to_end_stage1"]:
        if best2 is None:
            break
        n1 = pick_stage1(s1, which)
        type_pred = s2[best2].predict(X)
        sc, cm = E.end_to_end(te["category"].to_numpy(), preds[n1], type_pred)
        e2e_rows.append({"selector": which, "stage1": n1, "stage2": best2, **sc})
        P.heatmap(cm, f"End-to-end confusion matrix: {n1} + {best2}", fig / f"end_to_end_cm_{which}.png",
                  fmt="d", cmap="Blues")
    e2e = pd.DataFrame(e2e_rows)
    e2e.to_csv(rep / "results_end_to_end.csv", index=False)

    # ---- example timelines with the best unsupervised detector (old ones removed first)
    for old in fig.glob("timeline_*.png"):
        old.unlink()
    n1 = pick_stage1(s1, "best_unsupervised")
    ev = te[te["event_id"] >= 0].drop_duplicates("event_id")
    top = ev["sensor_id"].value_counts().index[: cfg["evaluation"]["timeline_examples"]]
    for sid in top:
        rows_s = te["sensor_id"] == sid
        variable = max(cfg["data"]["variables"],
                       key=lambda v: (te.loc[rows_s, v] - te.loc[rows_s, f"{v}_clean"]).abs().sum())
        P.timeline(te, sid, variable, scores[n1], s1[n1].threshold, n1, fig / f"timeline_{sid}_{variable}.png")

    _write_summary(rep, res1, scen, res2, e2e, order)
    log.info("\nSTAGE 1 (test)\n%s", res1[["model", "kind", "precision", "recall", "f1", "roc_auc", "pr_auc",
                                            "fpr", "inference_us_per_sample"]].round(3).to_string(index=False))
    if len(e2e):
        log.info("\nEND TO END (test)\n%s", e2e.round(3).to_string(index=False))
    return {"stage1": res1, "per_scenario": scen, "stage2": res2, "end_to_end": e2e}


def _write_summary(rep, res1, scen, res2, e2e, order) -> None:
    cols = ["model", "kind", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr",
            "inference_us_per_sample"]
    piv = scen.pivot(index="model", columns="scenario", values="event_recall").reindex(columns=order)
    lines = ["# Results (test split)", "",
             "## Stage 1 - detection (normal vs anomalous)", "", res1[cols].round(3).to_markdown(index=False), "",
             "## Event recall per scenario (share of events with at least one flagged point)", "",
             piv.round(2).to_markdown(), ""]
    if len(res2):
        lines += ["## Stage 2 - qualification on true anomalies", "",
                  res2.round(3).to_markdown(index=False), ""]
    if len(e2e):
        lines += ["## End to end - normal / outlier / fault / falsification", "",
                  e2e.round(3).to_markdown(index=False), ""]
    (rep / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_all(cfg: dict, synthetic: bool = False) -> dict:
    with timer("1/5 prepare data", log):
        prepare_data(cfg, synthetic)
    with timer("2/5 simulate attacks", log):
        simulate_attacks(cfg)
    with timer("3/5 build features", log):
        build_features(cfg)
    with timer("4/5 train models", log):
        train_models(cfg)
    with timer("5/5 evaluate", log):
        return evaluate_models(cfg)
