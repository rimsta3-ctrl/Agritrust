import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agri_anomaly.config import load_config  # noqa: E402
from agri_anomaly import data as D  # noqa: E402
from agri_anomaly.synthetic import generate  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "config.yaml")


@pytest.fixture(scope="session")
def grid(cfg):
    """Small clean regular-grid dataset (6 sensors x 12 days)."""
    raw = generate(n_sensors=6, days=12, freq="5min", seed=1)
    cleaned, _ = D.clean(raw, cfg["data"])
    return D.resample_regular(cleaned, "5min", ["temperature", "humidity", "rssi", "snr"], 2)
