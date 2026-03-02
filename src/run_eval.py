"""
Quick-run wrapper for the CSEA evaluation pipeline.

Delegates to evaluate_rag.py which handles both dataset and live modes.

Usage:
  python src/run_eval.py                          # evaluate CSV dataset
  python src/run_eval.py --mode live --limit 15   # evaluate live Supabase logs
  python src/run_eval.py --mode both              # run both modes
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.evaluate_rag import evaluate_from_dataset, evaluate_from_logs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CSEA RAG Evaluation Runner")
    parser.add_argument(
        "--mode", choices=["dataset", "live", "both"], default="dataset",
        help="dataset = CSV with ground truth, live = Supabase logs, both = run both",
    )
    parser.add_argument("--csv", default="csea_evaluation_dataset.csv", help="CSV dataset path")
    parser.add_argument("--limit", type=int, default=10, help="Number of logs for live mode")
    args = parser.parse_args()

    if args.mode in ("dataset", "both"):
        evaluate_from_dataset(args.csv)
    if args.mode in ("live", "both"):
        evaluate_from_logs(args.limit)
