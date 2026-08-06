from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd
from adapters.planning_input import load_planning_inputs
from run_yard_plan import DEFAULT_INPUT, DEFAULT_LARGE_PLAN, resolve_voyages
from yard_planning.planner import ColumnGenerationConfig, write_json
from yard_planning.voyage_plan_column_generation import (
    VoyagePlanColumnGenerationPlanner,
    VoyagePlanPricingConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark complete-voyage plan column generation with "
            "restricted row-level integer recovery."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--large-plan", type=Path, default=DEFAULT_LARGE_PLAN)
    parser.add_argument("--voyages", nargs="+", default=None)
    parser.add_argument("--planning-time", default=None)
    parser.add_argument("--total-time-limit", type=float, default=120.0)
    parser.add_argument("--max-pricing-iterations", type=int, default=60)
    parser.add_argument("--mip-time-limit", type=float, default=15.0)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument(
        "--root-only",
        action="store_true",
        help="Run only root LP column generation without integer recovery.",
    )
    parser.add_argument("--plans-per-pricing", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = InputAdapterGd.load_from_json(str(args.input.resolve()))
    large_plan = pd.read_csv(args.large_plan.resolve())
    voyages = resolve_voyages(large_plan, args.voyages)
    planning_time = pd.Timestamp(
        args.planning_time if args.planning_time else adapter.planning_time
    )
    if pd.isna(planning_time):
        raise SystemExit("planning_time is missing or invalid")
    inputs = load_planning_inputs(
        adapter,
        planning_time=planning_time.to_pydatetime(),
        voyages=voyages,
        big_plan=large_plan,
    )
    planner = VoyagePlanColumnGenerationPlanner(
        inputs.problem,
        ColumnGenerationConfig(
            max_iterations=args.max_pricing_iterations,
            total_time_limit=args.total_time_limit,
            mip_time_limit=args.mip_time_limit,
            mip_gap=0.0,
            solver_threads=args.solver_threads,
            verbose=False,
        ),
        VoyagePlanPricingConfig(
            plans_per_pricing=args.plans_per_pricing,
        ),
    )
    if not args.root_only:
        result = planner.solve_column_generation()
        summary_keys = (
            "algorithm",
            "status",
            "objective",
            "valid_lower_bound",
            "relative_gap",
            "root_exact",
            "root_status",
            "incumbent_source",
            "plan_count",
            "total_seconds",
        )
    else:
        result = planner.solve_root()
        summary_keys = (
            "algorithm",
            "status",
            "root_exact",
            "root_objective",
            "valid_lower_bound",
            "voyage_count",
            "group_count",
            "plan_count",
            "phase_iterations",
            "business_iterations",
            "pricing_seconds",
            "total_seconds",
        )
    summary = {key: result[key] for key in summary_keys}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output is not None:
        write_json(args.output.resolve(), result)


if __name__ == "__main__":
    main()
