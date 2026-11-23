import numpy as np

from agri_anomaly.features import FeatureBuilder


def _builder(cfg, grid):
    return FeatureBuilder.from_config(cfg, "5min").fit(grid)


def test_no_future_leakage(cfg, grid):
    """Changing the future must not change any feature of the past."""
    fb = _builder(cfg, grid)
    base = fb.transform(grid, keep_missing=True)
    cut = grid["timestamp"].quantile(0.7)
    modified = grid.copy()
    future = modified["timestamp"] >= cut
    modified.loc[future, "temperature"] += 10.0
    modified.loc[future & (modified["sensor_id"] == "AF17"), "humidity"] = 99.0
    after = fb.transform(modified, keep_missing=True)
    past = (base["timestamp"] < cut).to_numpy()
    cols = fb.feature_names_
    assert np.allclose(base.loc[past, cols].to_numpy(), after.loc[past, cols].to_numpy(), equal_nan=True)


def test_no_absolute_level_in_features(cfg, grid):
    fb = _builder(cfg, grid)
    fb.transform(grid)
    assert "temperature" not in fb.feature_names_ and "humidity" not in fb.feature_names_


def test_stuck_sensor_raises_flat_feature(cfg, grid):
    fb = _builder(cfg, grid)
    g = grid.copy()
    rows = g.index[(g["sensor_id"] == "AF18")][500:560]
    g.loc[rows, "temperature"] = g.loc[rows[0], "temperature"]
    F = fb.transform(g, keep_missing=True)
    assert F.loc[rows[-1], "temperature_flat"] > np.log1p(30)
    assert F.loc[rows[-1], "temperature_std_s"] < 1e-4
