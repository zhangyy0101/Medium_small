"""Command-line entry point for phase-separated V6 scalability runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preexperiment.v6_scalability import run_v6_suite


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run analytic peak-cap preparation, exact root CG, compact primal "
            "coverage, and the final V6 restricted integer master."
        )
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "preexperiment" / "pilot_suite.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "preexperiment_outputs" / "v6_scalability_smoke",
    )
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument(
        "--peak-feasibility-time-limit",
        "--peak-time-limit",
        dest="peak_feasibility_time_limit",
        type=float,
        default=10.0,
        help=(
            "time allowed to find a feasibility witness under the analytic "
            "epsilon cap; --peak-time-limit is a deprecated alias"
        ),
    )
    parser.add_argument("--pricing-time-limit", type=float, default=30.0)
    parser.add_argument("--root-total-time-limit", type=float, default=60.0)
    parser.add_argument("--maximum-phase-one-iterations", type=int, default=200)
    parser.add_argument("--maximum-business-iterations", type=int, default=500)
    parser.add_argument(
        "--phase-one-columns-per-group", type=int, default=512
    )
    parser.add_argument(
        "--business-columns-per-group", type=int, default=512
    )
    parser.add_argument(
        "--maximum-patterns-per-interval", type=int, default=8
    )
    parser.add_argument("--coverage-time-limit", type=float, default=60.0)
    parser.add_argument("--coverage-mip-gap", type=float, default=0.0)
    parser.add_argument("--coverage-pool-solutions", type=int, default=4)
    parser.add_argument("--integer-time-limit", type=float, default=60.0)
    parser.add_argument("--integer-mip-gap", type=float, default=0.0)
    parser.add_argument(
        "--root-only",
        action="store_true",
        help="stop after the exact root certificate and mark the case complete",
    )
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument(
        "--peak-utilization-headroom-fraction", type=float, default=0.50
    )
    parser.add_argument("--verbose-solver", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_v6_suite(
        args.suite,
        args.output_root,
        case_ids=args.cases,
        peak_feasibility_time_limit=args.peak_feasibility_time_limit,
        pricing_time_limit=args.pricing_time_limit,
        root_total_time_limit=args.root_total_time_limit,
        maximum_phase_one_iterations=args.maximum_phase_one_iterations,
        maximum_business_iterations=args.maximum_business_iterations,
        phase_one_columns_per_group=args.phase_one_columns_per_group,
        business_columns_per_group=args.business_columns_per_group,
        maximum_patterns_per_interval=(
            args.maximum_patterns_per_interval
        ),
        coverage_time_limit=args.coverage_time_limit,
        coverage_mip_gap=args.coverage_mip_gap,
        coverage_pool_solutions=args.coverage_pool_solutions,
        integer_time_limit=args.integer_time_limit,
        integer_mip_gap=args.integer_mip_gap,
        root_only=args.root_only,
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
                "failed_count": summary["failed_count"],
                "output_root": str(args.output_root.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if int(summary["passed_count"]) != int(summary["case_count"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
