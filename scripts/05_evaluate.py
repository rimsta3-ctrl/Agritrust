"""Stage 5: evaluate every model on the test split and write tables and figures to reports/."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without pip install

from agri_anomaly.config import load_config  # noqa: E402
from agri_anomaly.pipeline import evaluate_models  # noqa: E402
from agri_anomaly.utils import get_logger  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    args = ap.parse_args()
    get_logger()
    cfg = load_config(args.config)
    evaluate_models(cfg)
