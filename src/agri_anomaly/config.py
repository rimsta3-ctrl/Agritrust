"""Load config.yaml and resolve paths relative to the project root."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(path.resolve().parent)
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Absolute path for cfg['paths'][key] (relative paths are relative to config.yaml)."""
    p = Path(cfg["paths"][key])
    p = p if p.is_absolute() else Path(cfg["_root"]) / p
    return p


def ensure_dirs(cfg: dict) -> None:
    for key in ("processed", "features", "models", "reports"):
        resolve_path(cfg, key).mkdir(parents=True, exist_ok=True)
    (resolve_path(cfg, "reports") / "figures").mkdir(parents=True, exist_ok=True)
