from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd
from adapters.planning_input import load_planning_inputs
from benchmark_profile_benders import _summary
from run_yard_plan import DEFAULT_INPUT, DEFAULT_LARGE_PLAN, resolve_voyages
from yard_planning.contiguous_zone_generation import (
    ContiguousZoneConfig,
    ContiguousZoneGenerationPlanner,
)
from yard_planning.direct_milp import DirectMilpPlanner
from yard_planning.planner import ColumnGenerationConfig, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the isolated contiguous storage-zone stage gate."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--large-plan", type=Path, default=DEFAULT_LARGE_PLAN)
    parser.add_argument("--voyages", nargs="+", default=None)
    parser.add_argument("--planning-time", default=None)
    parser.add_argument("--total-time-limit", type=float, default=60.0)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--max-root-iterations", type=int, default=60)
    parser.add_argument("--columns-per-group", type=int, default=3)
    parser.add_argument("--integer-pool-per-group", type=int, default=100)
    parser.add_argument("--zone-mip-time-fraction", type=float, default=0.75)
    parser.add_argument("--fix-optimize-local-fraction", type=float, default=0.85)
    parser.add_argument("--fix-optimize-groups", type=int, default=18)
    parser.add_argument(
        "--fix-optimize-policy",
        choices=("disabled", "objective"),
        default="objective",
    )
    parser.add_argument("--fill-time-fraction", type=float, default=0.05)
    parser.add_argument("--root-only", action="store_true")
    parser.add_argument("--complete-zone-mip-only", action="store_true")
    parser.add_argument("--compare-complete-zone-lp", action="store_true")
    parser.add_argument("--compare-complete-zone-mip", action="store_true")
    parser.add_argument(
        "--compare-row-m0",
        action="store_true",
        help=(
            "Run the legacy row-level M0 only as a different-model "
            "structural reference; objectives are not subtracted."
        ),
    )
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
    common = ColumnGenerationConfig(
        total_time_limit=args.total_time_limit,
        mip_gap=0.0,
        solver_threads=args.solver_threads,
        verbose=False,
    )
    zone_config = ContiguousZoneConfig(
        max_root_iterations=args.max_root_iterations,
        columns_per_group_per_round=args.columns_per_group,
        integer_pool_columns_per_group=args.integer_pool_per_group,
        zone_mip_time_fraction=args.zone_mip_time_fraction,
        fix_optimize_local_fraction=args.fix_optimize_local_fraction,
        fix_optimize_group_count=args.fix_optimize_groups,
        fix_optimize_policy=args.fix_optimize_policy,
        fill_time_fraction=args.fill_time_fraction,
    )
    planner = ContiguousZoneGenerationPlanner(
        inputs.problem,
        common,
        zone_config,
    )
    if args.root_only and args.complete_zone_mip_only:
        raise SystemExit(
            "--root-only and --complete-zone-mip-only are mutually exclusive"
        )
    if args.root_only:
        output = {
            "contiguous_zone_root": planner.analyze_root(
                compare_complete_lp=args.compare_complete_zone_lp,
                compare_complete_mip=args.compare_complete_zone_mip,
            )
        }
    elif args.complete_zone_mip_only:
        output = {"complete_zone_mip": planner.analyze_complete_zone_mip()}
    else:
        result = planner.solve()
        output = {"contiguous_zone": _summary(result.diagnostics)}
        output["contiguous_zone_diagnostics"] = {
            key: value
            for key, value in result.diagnostics.items()
            if key.startswith("zone_")
            or key in {"independent_solution_validation", "total_seconds"}
        }
        if args.compare_row_m0:
            direct = DirectMilpPlanner(inputs.problem, common).solve()
            output["row_m0_different_model_reference"] = _summary(
                direct.diagnostics
            )
            output["cross_model_objective_difference_reported"] = False
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.output is not None:
        write_json(args.output.resolve(), output)


if __name__ == "__main__":
    main()
