from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd
from adapters.planning_input import load_planning_inputs
from run_yard_plan import DEFAULT_INPUT, DEFAULT_LARGE_PLAN, resolve_voyages
from yard_planning.direct_milp import DirectMilpPlanner
from yard_planning.logic_benders import (
    LogicBendersConfig,
    LogicBendersPlanner,
)
from yard_planning.planner import ColumnGenerationConfig, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark strengthened logic-based Benders and optionally M0 "
            "on the same row-allocation model."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--large-plan", type=Path, default=DEFAULT_LARGE_PLAN)
    parser.add_argument("--voyages", nargs="+", default=None)
    parser.add_argument("--planning-time", default=None)
    parser.add_argument("--total-time-limit", type=float, default=120.0)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--master-time-limit", type=float, default=10.0)
    parser.add_argument("--area-time-limit", type=float, default=5.0)
    parser.add_argument("--primal-seed-time-limit", type=float, default=10.0)
    parser.add_argument("--support-repair-iterations", type=int, default=5)
    parser.add_argument("--support-repair-fraction", type=float, default=0.02)
    parser.add_argument("--compare-direct", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _summary(diagnostics: dict) -> dict:
    summary = {
        key: diagnostics.get(key)
        for key in (
            "algorithm",
            "master_status",
            "final_business_objective",
            "complete_model_lower_bound",
            "complete_model_relative_gap",
            "lbbd_converged",
            "lbbd_termination_reason",
            "lbbd_iteration_count",
            "lbbd_cut_counts",
            "lbbd_area_subproblem_solve_count",
            "lbbd_area_subproblem_cache_hits",
            "lbbd_best_incumbent_source",
            "lbbd_support_repair",
            "lbbd_total_solve_seconds",
            "direct_total_solve_seconds",
        )
        if diagnostics.get(key) is not None
    }
    seed = diagnostics.get("lbbd_compact_primal_seed", {})
    summary["lbbd_compact_primal_seed"] = {
        key: seed.get(key)
        for key in (
            "attempted",
            "feasible",
            "status",
            "seconds",
            "objective",
            "bound_used",
        )
        if seed.get(key) is not None
    }
    return summary


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
    common_config = ColumnGenerationConfig(
        total_time_limit=args.total_time_limit,
        mip_gap=0.0,
        solver_threads=args.solver_threads,
        verbose=False,
    )
    lbbd = LogicBendersPlanner(
        inputs.problem,
        common_config,
        LogicBendersConfig(
            max_iterations=args.max_iterations,
            master_time_limit=args.master_time_limit,
            area_time_limit=args.area_time_limit,
            primal_seed_time_limit=args.primal_seed_time_limit,
            support_repair_iterations=args.support_repair_iterations,
            support_repair_fraction=args.support_repair_fraction,
        ),
    ).solve()
    results = {"lbbd": _summary(lbbd.diagnostics)}
    if args.compare_direct:
        direct = DirectMilpPlanner(inputs.problem, common_config).solve()
        results["direct"] = _summary(direct.diagnostics)
        results["objective_difference"] = round(
            float(lbbd.diagnostics["final_business_objective"])
            - float(direct.diagnostics["final_business_objective"]),
            10,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.output is not None:
        write_json(args.output.resolve(), results)


if __name__ == "__main__":
    main()
