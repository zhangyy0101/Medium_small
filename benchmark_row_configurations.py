from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd
from adapters.planning_input import load_planning_inputs
from benchmark_profile_benders import _summary
from run_yard_plan import DEFAULT_INPUT, DEFAULT_LARGE_PLAN, resolve_voyages
from yard_planning.direct_milp import DirectMilpPlanner
from yard_planning.planner import ColumnGenerationConfig, write_json
from yard_planning.row_configuration_generation import (
    RowConfigurationGenerationPlanner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark physical-row configuration generation."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--large-plan", type=Path, default=DEFAULT_LARGE_PLAN)
    parser.add_argument("--voyages", nargs="+", default=None)
    parser.add_argument("--planning-time", default=None)
    parser.add_argument("--total-time-limit", type=float, default=120.0)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=60)
    parser.add_argument("--root-only", action="store_true")
    parser.add_argument("--compare-m0-lp", action="store_true")
    parser.add_argument("--compare-direct", action="store_true")
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
    config = ColumnGenerationConfig(
        total_time_limit=args.total_time_limit,
        max_iterations=args.max_iterations,
        mip_gap=0.0,
        solver_threads=args.solver_threads,
        verbose=False,
    )
    results = {}
    if args.root_only:
        results["row_configuration_root"] = (
            RowConfigurationGenerationPlanner(
                inputs.problem, config
            ).solve_root_relaxation(args.total_time_limit)
        )
    else:
        configuration = RowConfigurationGenerationPlanner(
            inputs.problem, config
        ).solve()
        results["row_configuration"] = _summary(
            configuration.diagnostics
        )
        results["row_configuration"]["root"] = (
            configuration.diagnostics["row_configuration_root"]
        )
        results["row_configuration"]["restricted_master"] = (
            configuration.diagnostics[
                "row_configuration_restricted_master"
            ]
        )
    if args.compare_m0_lp:
        results["m0_lp"] = RowConfigurationGenerationPlanner(
            inputs.problem, config
        ).solve_m0_lp_relaxation(args.total_time_limit)
    if args.compare_direct:
        direct = DirectMilpPlanner(inputs.problem, config).solve()
        results["direct"] = _summary(direct.diagnostics)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.output is not None:
        write_json(args.output.resolve(), results)


if __name__ == "__main__":
    main()
