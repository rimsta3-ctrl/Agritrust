import numpy as np
import pandas as pd
import pytest

from agri_anomaly.attacks import AttackSimulator


@pytest.fixture(scope="module")
def attacked(cfg, grid):
    sim = AttackSimulator(cfg["attacks"], cfg["data"]["variables"], "5min", seed=3)
    return sim.inject(grid, "test", n_per_scenario=3)


def test_every_scenario_is_injected_with_its_category(cfg, attacked):
    frame, events = attacked
    assert set(events["scenario"]) == set(cfg["attacks"]["scenarios"])
    for _, ev in events.iterrows():
        assert ev["category"] == cfg["attacks"]["scenarios"][ev["scenario"]]["category"]


def test_labels_match_the_event_log(attacked):
    frame, events = attacked
    assert frame["label"].sum() == events["n_points"].sum()
    assert (frame.loc[frame["label"] == 0, "category"] == "normal").all()
    # events do not overlap
    assert frame.loc[frame["event_id"] >= 0].groupby("event_id").size().tolist() == events["n_points"].tolist()


def test_normal_rows_are_untouched(cfg, attacked):
    frame, _ = attacked
    normal = frame["label"] == 0
    for v in cfg["data"]["variables"]:
        a, b = frame.loc[normal, v], frame.loc[normal, f"{v}_clean"]
        assert np.allclose(a, b, equal_nan=True)


def test_stuck_is_constant_and_replay_copies_the_past(cfg, attacked):
    frame, events = attacked
    for _, ev in events[events["scenario"] == "stuck"].iterrows():
        rows = frame[frame["event_id"] == ev["event_id"]]
        assert rows[ev["variables"]].nunique() == 1
    for _, ev in events[events["scenario"] == "replay"].iterrows():
        days = int(ev["intensity"])
        rows = frame[frame["event_id"] == ev["event_id"]]
        past = frame[(frame["sensor_id"] == ev["sensor_id"])].set_index("timestamp")
        src = past.loc[rows["timestamp"] - pd.Timedelta(days=days), "temperature_clean"].to_numpy()
        assert np.allclose(rows["temperature"].to_numpy(), src)


def test_single_injection_for_dashboard(cfg, grid):
    sim = AttackSimulator(cfg["attacks"], cfg["data"]["variables"], "5min", seed=0)
    start = grid["timestamp"].iloc[len(grid) // 12]
    frame, events = sim.inject_one(grid, "AF16", "offset", start, 12, intensity=1.0, variable="temperature")
    assert len(events) == 1 and frame["label"].sum() == 12
