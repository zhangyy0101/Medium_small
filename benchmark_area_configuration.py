from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd
from adapters.planning_input import load_planning_inputs
from run_yard_plan import DEFAULT_INPUT, DEFAULT_LARGE_PLAN, resolve_voyages
from yard_planning.area_branch_price import (
    AreaConfigurationBranchPricePlanner,
)
from yard_planning.area_configuration import (
    AdaptiveAreaPricingConfig,
)
from yard_planning.planner import ColumnGenerationConfig, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark adaptive area-configuration column generation at "
            "the root or with Branch-and-Price."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--large-plan", type=Path, default=DEFAULT_LARGE_PLAN)
    parser.add_argument("--voyages", nargs="+", default=None)
    parser.add_argument("--planning-time", default=None)
    parser.add_argument("--total-time-limit", type=float, default=120.0)
    parser.add_argument("--max-pricing-iterations", type=int, default=60)
    parser.add_argument("--full-pricing-frequency", type=int, default=2)
    parser.add_argument("--max-branch-nodes", type=int, default=200)
    parser.add_argument("--mip-time-limit", type=float, default=15.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument(
        "--algorithm",
        choices=("root", "branch-price"),
        default="root",
    )
    parser.add_argument("--direct-candidate-limit", type=int, default=2_000)
    parser.add_argument("--complex-area-pool-size", type=int, default=6)
    parser.add_argument("--simple-area-pool-size", type=int, default=1)
    parser.add_argument("--complex-time-weight", type=float, default=1.5)
    parser.add_argument("--certificate-time-fraction", type=float, default=0.15)
    parser.add_argument("--nested-max-iterations", type=int, default=24)
    parser.add_argument("--nested-time-fraction", type=float, default=0.70)
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
    planner = AreaConfigurationBranchPricePlanner(
        inputs.problem,
        ColumnGenerationConfig(
            max_iterations=args.max_pricing_iterations,
            max_branch_nodes=args.max_branch_nodes,
            total_time_limit=args.total_time_limit,
            mip_time_limit=args.mip_time_limit,
            mip_gap=args.mip_gap,
            solver_threads=args.solver_threads,
            verbose=False,
        ),
        AdaptiveAreaPricingConfig(
            direct_candidate_limit=args.direct_candidate_limit,
            complex_area_pool_size=args.complex_area_pool_size,
            simple_area_pool_size=args.simple_area_pool_size,
            complex_time_weight=args.complex_time_weight,
            certificate_time_fraction=args.certificate_time_fraction,
            full_sweep_frequency=args.full_pricing_frequency,
            nested_max_iterations=args.nested_max_iterations,
            nested_time_fraction=args.nested_time_fraction,
        ),
    )
    if args.algorithm == "branch-price":
        result = planner.solve_branch_and_price()
        summary_keys = (
            "algorithm",
            "status",
            "objective",
            "global_lower_bound",
            "relative_gap",
            "root_lower_bound",
            "processed_nodes",
            "open_nodes",
            "branch_counts",
            "configuration_count",
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
            "area_count",
            "group_count",
            "configuration_count",
            "phase_iterations",
            "business_iterations",
            "pricing_seconds",
            "total_seconds",
        )
    summary = {key: result[key] for key in summary_keys}
    if "adaptive_pricing" in result:
        summary["complex_areas"] = [
            profile
            for profile in result["adaptive_pricing"]["profiles"]
            if profile["strategy"] != "direct_exact_mip"
        ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output is not None:
        write_json(args.output.resolve(), result)


if __name__ == "__main__":
    main()
