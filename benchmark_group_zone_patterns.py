from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd
from adapters.planning_input import load_planning_inputs
from benchmark_profile_benders import _summary
from run_yard_plan import DEFAULT_INPUT, DEFAULT_LARGE_PLAN, resolve_voyages
from yard_planning.group_zone_pattern_generation import (
    GroupZonePatternConfig,
    GroupZonePatternGenerationPlanner,
)
from yard_planning.planner import ColumnGenerationConfig, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the isolated complete export-group contiguous-zone "
            "pattern generation stage gate."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--large-plan", type=Path, default=DEFAULT_LARGE_PLAN)
    parser.add_argument("--voyages", nargs="+", default=None)
    parser.add_argument("--planning-time", default=None)
    parser.add_argument("--total-time-limit", type=float, default=60.0)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--max-root-iterations", type=int, default=80)
    parser.add_argument("--patterns-per-group", type=int, default=3)
    parser.add_argument("--integer-patterns-per-group", type=int, default=40)
    parser.add_argument("--root-time-fraction", type=float, default=0.60)
    parser.add_argument("--fill-time-fraction", type=float, default=0.05)
    parser.add_argument("--fix-optimize-time-fraction", type=float, default=0.15)
    parser.add_argument("--fix-optimize-groups", type=int, default=12)
    parser.add_argument("--root-only", action="store_true")
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
    pattern_config = GroupZonePatternConfig(
        max_root_iterations=args.max_root_iterations,
        patterns_per_group_per_round=args.patterns_per_group,
        root_time_fraction=args.root_time_fraction,
        fill_time_fraction=args.fill_time_fraction,
        max_integer_patterns_per_group=args.integer_patterns_per_group,
        fix_optimize_time_fraction=args.fix_optimize_time_fraction,
        fix_optimize_group_count=args.fix_optimize_groups,
    )
    planner = GroupZonePatternGenerationPlanner(
        inputs.problem,
        common,
        pattern_config,
    )
    if args.root_only:
        output = {"group_zone_pattern_root": planner.analyze_root()}
    else:
        result = planner.solve()
        output = {
            "group_zone_patterns": _summary(result.diagnostics),
            "group_zone_pattern_diagnostics": {
                key: value
                for key, value in result.diagnostics.items()
                if key.startswith("group_pattern_")
                or key in {
                    "algorithm",
                    "independent_solution_validation",
                    "row_recourse_business_objective",
                    "total_seconds",
                    "zone_fill",
                }
            },
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.output is not None:
        write_json(args.output.resolve(), output)


if __name__ == "__main__":
    main()
