from __future__ import annotations

import argparse
import json
from pathlib import Path

from .comparison import run_comparison_suite


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired pilot ablation and same-model correctness checks."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "preexperiment" / "pilot_suite.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "preexperiment_outputs" / "pilot_v1_comparison_60s",
    )
    parser.add_argument(
        "--full-results-root",
        type=Path,
        default=ROOT / "preexperiment_outputs" / "pilot_v1_full_60s",
    )
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument(
        "--correctness-cases",
        nargs="+",
        default=["pilot_s_101", "pilot_s_102"],
    )
    parser.add_argument("--method-time-limit", type=float, default=60.0)
    parser.add_argument("--correctness-time-limit", type=float, default=120.0)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_comparison_suite(
        args.suite,
        args.output_root,
        args.full_results_root,
        case_ids=args.cases,
        correctness_case_ids=args.correctness_cases,
        method_time_limit=args.method_time_limit,
        correctness_time_limit=args.correctness_time_limit,
        threads=args.threads,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    if summary["passed_count"] != summary["case_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
