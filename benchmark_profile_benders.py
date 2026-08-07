from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd
from adapters.planning_input import load_planning_inputs
from run_yard_plan import DEFAULT_INPUT, DEFAULT_LARGE_PLAN, resolve_voyages
from yard_planning.direct_milp import DirectMilpPlanner
from yard_planning.logic_benders import LogicBendersConfig
from yard_planning.planner import ColumnGenerationConfig, write_json
from yard_planning.profile_resource_benders import (
    ProfileResourceBendersPlanner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark row-profile LBBD and optionally M0 on the same model."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--large-plan", type=Path, default=DEFAULT_LARGE_PLAN)
    parser.add_argument("--voyages", nargs="+", default=None)
    parser.add_argument("--planning-time", default=None)
    parser.add_argument("--total-time-limit", type=float, default=120.0)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument(
        "--max-footprint-cut-rounds", type=int, default=40
    )
    parser.add_argument(
        "--master-feasibility-time-limit", type=float, default=30.0
    )
    parser.add_argument("--master-time-limit", type=float, default=20.0)
    parser.add_argument("--voyage-time-limit", type=float, default=8.0)
    parser.add_argument("--compare-direct", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _summary(diagnostics: dict) -> dict:
    keys = (
        "algorithm",
        "master_status",
        "planned_group_count",
        "planned_box_count",
        "candidate_row_location_count",
        "final_business_objective",
        "complete_model_lower_bound",
        "complete_model_absolute_gap",
        "complete_model_relative_gap",
        "profile_lbbd_converged",
        "profile_lbbd_termination_reason",
        "profile_lbbd_master_round_count",
        "profile_lbbd_matching_cut_count",
        "profile_lbbd_matching_solve_count",
        "profile_lbbd_matching_seconds",
        "profile_lbbd_matching_variable_count",
        "profile_lbbd_aggregate_feasibility_cut_count",
        "profile_lbbd_aggregate_optimality_cut_count",
        "profile_lbbd_aggregate_cut_binary_count",
        "profile_lbbd_global_subproblem_solve_count",
        "profile_lbbd_global_subproblem_seconds",
        "profile_lbbd_global_subproblem_build_seconds",
        "profile_lbbd_global_subproblem_variable_count",
        "profile_lbbd_global_subproblem_records",
        "profile_lbbd_initial_disaggregation",
        "profile_lbbd_voyage_solve_count",
        "profile_lbbd_voyage_build_seconds",
        "profile_lbbd_preparation_seconds",
        "profile_lbbd_master_build_seconds",
        "profile_lbbd_master_feasibility",
        "profile_lbbd_total_solve_seconds",
        "direct_model_build_seconds",
        "direct_total_solve_seconds",
        "direct_bound",
        "direct_mip_gap",
    )
    summary = {
        key: diagnostics.get(key)
        for key in keys
        if diagnostics.get(key) is not None
    }
    summary["profile_lbbd_master_rounds"] = [
        {
            key: row.get(key)
            for key in (
                "master_round",
                "master_status",
                "master_allowance",
                "master_seconds",
                "master_objective",
                "master_bound",
                "active_profile_state_count",
                "active_profile_unit_count",
                "matching_status",
                "matching_feasible",
                "matching_seconds",
                "cuts_added",
                "cut_kind",
                "candidate_objective",
                "global_disaggregation",
            )
        }
        for row in diagnostics.get("profile_lbbd_master_rounds", [])
    ]
    summary["profile_lbbd_master"] = diagnostics.get(
        "profile_lbbd_master", {}
    )
    summary["direct_model"] = diagnostics.get("direct_model", {})
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
    profile = ProfileResourceBendersPlanner(
        inputs.problem,
        common_config,
        LogicBendersConfig(
            max_iterations=args.max_footprint_cut_rounds,
            master_feasibility_time_limit=(
                args.master_feasibility_time_limit
            ),
            master_time_limit=args.master_time_limit,
            voyage_time_limit=args.voyage_time_limit,
        ),
    ).solve()
    results = {"profile_lbbd": _summary(profile.diagnostics)}
    if args.compare_direct:
        direct = DirectMilpPlanner(inputs.problem, common_config).solve()
        results["direct"] = _summary(direct.diagnostics)
        results["objective_difference"] = round(
            float(profile.diagnostics["final_business_objective"])
            - float(direct.diagnostics["final_business_objective"]),
            10,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.output is not None:
        write_json(args.output.resolve(), results)


if __name__ == "__main__":
    main()
