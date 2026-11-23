"""Small helpers: seeding, logging, timing, duration conversion."""
from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def get_logger(name: str = "agri_anomaly") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


@contextmanager
def timer(label: str, logger: logging.Logger | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    (logger or get_logger()).info("%s done in %.1f s", label, dt)


def to_steps(duration: str | int, freq: str | pd.Timedelta) -> int:
    """Convert a duration ('3h', '30min' or an int number of steps) to a number of grid steps (>= 1)."""
    if isinstance(duration, (int, np.integer)):
        return max(1, int(duration))
    steps = pd.Timedelta(duration) / pd.Timedelta(freq)
    return max(1, int(round(steps)))
