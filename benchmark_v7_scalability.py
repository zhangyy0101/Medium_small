"""Command-line entry point for V7 24/48/96 scale experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preexperiment.v7_scalability import run_v7_suite


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V7 two-stage exact CG and the same-model Complete MIP."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "preexperiment" / "scale_suite.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "preexperiment_outputs" / "v7_scale_24_48_96",
    )
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--peak-feasibility-time-limit", type=float, default=10.0)
    parser.add_argument("--stage1-time-limit", type=float, default=10.0)
    parser.add_argument("--stage1-pool-solutions", type=int, default=8)
    parser.add_argument("--stage1-pool-gap", type=float, default=0.10)
    parser.add_argument("--initial-candidate-area-cap", type=int, default=4)
    parser.add_argument("--root-time-limit", type=float, default=60.0)
    parser.add_argument("--root-maximum-iterations", type=int, default=100)
    parser.add_argument("--columns-per-bay-per-round", type=int, default=3)
    parser.add_argument("--integer-time-limit", type=float, default=40.0)
    parser.add_argument("--integer-mip-gap", type=float, default=0.0)
    parser.add_argument("--complete-mip-time-limit", type=float, default=120.0)
    parser.add_argument("--complete-mip-gap", type=float, default=0.0)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument(
        "--peak-utilization-headroom-fraction", type=float, default=0.50
    )
    parser.add_argument("--verbose-solver", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_v7_suite(
        args.suite,
        args.output_root,
        case_ids=args.cases,
        peak_feasibility_time_limit=args.peak_feasibility_time_limit,
        stage1_time_limit=args.stage1_time_limit,
        stage1_pool_solutions=args.stage1_pool_solutions,
        stage1_pool_gap=args.stage1_pool_gap,
        initial_candidate_area_cap=args.initial_candidate_area_cap,
        root_time_limit=args.root_time_limit,
        root_maximum_iterations=args.root_maximum_iterations,
        columns_per_bay_per_round=args.columns_per_bay_per_round,
        integer_time_limit=args.integer_time_limit,
        integer_mip_gap=args.integer_mip_gap,
        complete_mip_time_limit=args.complete_mip_time_limit,
        complete_mip_gap=args.complete_mip_gap,
        solver_threads=args.solver_threads,
        solver_seed=args.solver_seed,
        peak_utilization_headroom_fraction=(
            args.peak_utilization_headroom_fraction
        ),
        verbose_solver=args.verbose_solver,
        progress=lambda message: print(message, flush=True),
    )
    print(
        json.dumps(
            {
                "model_schema_version": summary["model_schema_version"],
                "case_count": summary["case_count"],
                "passed_count": summary["passed_count"],
                "partial_count": summary["partial_count"],
                "output_root": str(args.output_root.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
