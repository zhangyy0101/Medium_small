from __future__ import annotations

import argparse
import json
from pathlib import Path

from .complete_mip_baseline import run_complete_mip_baselines


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the paper algorithm with the complete-zone MIP."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "preexperiment" / "pilot_suite.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "preexperiment_outputs" / "pilot_v1_complete_mip_60s",
    )
    parser.add_argument(
        "--full-results-root",
        type=Path,
        default=ROOT / "preexperiment_outputs" / "pilot_v1_full_60s",
    )
    parser.add_argument(
        "--paired-inputs-root",
        type=Path,
        default=ROOT / "preexperiment_outputs" / "pilot_v1_comparison_60s",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["pilot_m_201", "pilot_m_202", "pilot_l_301", "pilot_l_302"],
    )
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_complete_mip_baselines(
        args.suite,
        args.output_root,
        args.full_results_root,
        args.paired_inputs_root,
        case_ids=args.cases,
        time_limit=args.time_limit,
        threads=args.threads,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    if summary["passed_count"] != summary["case_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
