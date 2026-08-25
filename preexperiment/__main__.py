from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_suite


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and validate deterministic preliminary experiment cases.")
    parser.add_argument("--suite", type=Path, default=ROOT / "preexperiment" / "pilot_suite.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "preexperiment_outputs" / "pilot_v1")
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument(
        "--paper-time-limit",
        type=float,
        default=0.0,
        help="Positive values also run the full paper algorithm for every selected case.",
    )
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_suite(
        args.suite,
        args.output_root,
        case_ids=args.cases,
        paper_time_limit=args.paper_time_limit,
        threads=args.threads,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["passed_count"] != summary["case_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
