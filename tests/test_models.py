import numpy as np
import pytest

from agri_anomaly.attacks import AttackSimulator
from agri_anomaly.evaluate import choose_threshold
from agri_anomaly.features import FeatureBuilder
from agri_anomaly.models import STAGE1, BaseDetector, TypeClassifier

FAST = {
    "isolation_forest": {"n_estimators": 50},
    "one_class_svm": {"max_train_samples": 2000},
    "autoencoder": {"epochs": 3},
    "lstm_autoencoder": {"epochs": 1, "max_train_samples": 2000, "window": 12},
    "random_forest": {"n_estimators": 30},
    "xgboost": {"n_estimators": 50},
}


@pytest.fixture(scope="module")
def tables(cfg, grid):
    fb = FeatureBuilder.from_config(cfg, "5min").fit(grid)
    clean = fb.transform(grid.assign(label=0, category="normal", scenario="normal", event_id=-1))
    sim = AttackSimulator(cfg["attacks"], cfg["data"]["variables"], "5min", seed=5)
    att, _ = sim.inject(grid, "x", n_per_scenario=4)
    return fb.feature_names_, clean, fb.transform(att)


@pytest.mark.parametrize("name", list(STAGE1))
def test_detector_interface(name, tables, tmp_path):
    if name in ("autoencoder", "lstm_autoencoder"):
        pytest.importorskip("torch")
    names, clean, att = tables
    model = STAGE1[name](seed=0, **FAST.get(name, {}))
    src = att if model.kind == "supervised" else clean
    model.fit(src[names], src["label"].to_numpy(), src["sensor_id"].to_numpy())
    s = model.score(att[names], att["sensor_id"].to_numpy())
    assert s.shape == (len(att),) and np.isfinite(s).all()
    y = att["label"].to_numpy()
    assert s[y == 1].mean() > s[y == 0].mean()      # anomalies look more suspicious on average
    model.threshold = choose_threshold(s, y)
    model.save(tmp_path / f"{name}.joblib")
    again = BaseDetector.load(tmp_path / f"{name}.joblib")
    assert np.array_equal(again.predict(att[names], att["sensor_id"].to_numpy()),
                          model.predict(att[names], att["sensor_id"].to_numpy()))


def test_threshold_strategies():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(50)].astype(int)
    s = np.r_[rng.normal(0, 1, 1000), rng.normal(3, 1, 50)]
    thr = choose_threshold(s, y, "fpr", target_fpr=0.05)
    assert abs((s[y == 0] >= thr).mean() - 0.05) < 0.01
    assert s.min() <= choose_threshold(s, y, "f1") <= s.max()


def test_type_classifier(tables):
    names, _, att = tables
    anom = att[att["label"] == 1]
    clf = TypeClassifier("rf", n_estimators=50).fit(anom[names], anom["category"].to_numpy())
    pred = clf.predict(anom[names])
    assert set(pred) <= {"outlier", "fault", "falsification"}
    assert (pred == anom["category"].to_numpy()).mean() > 0.8   # on its own training data
