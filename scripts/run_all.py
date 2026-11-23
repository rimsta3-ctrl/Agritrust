"""Run the whole pipeline: data -> attacks -> features -> training -> evaluation."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without pip install

from agri_anomaly.config import load_config  # noqa: E402
from agri_anomaly.pipeline import run_all  # noqa: E402
from agri_anomaly.utils import get_logger  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    ap.add_argument("--synthetic", action="store_true", help="use generated data instead of the real file")
    args = ap.parse_args()
    get_logger()
    cfg = load_config(args.config)
    run_all(cfg, synthetic=args.synthetic)
