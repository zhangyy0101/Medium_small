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
    parser.add_argument(
        "--master-feasibility-time-limit", type=float, default=30.0
    )
    parser.add_argument("--master-time-limit", type=float, default=20.0)
    parser.add_argument("--voyage-time-limit", type=float, default=8.0)
    parser.add_argument("--compare-direct", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _summary(diagnostics: dict) -> dict:
    summary = {
        key: diagnostics.get(key)
        for key in (
            "algorithm",
            "master_status",
            "planned_group_count",
            "planned_box_count",
            "candidate_row_location_count",
            "final_business_objective",
            "complete_model_lower_bound",
            "complete_model_absolute_gap",
            "complete_model_relative_gap",
            "lbbd_converged",
            "lbbd_termination_reason",
            "lbbd_iteration_count",
            "lbbd_cut_counts",
            "lbbd_voyage_subproblem_solve_count",
            "lbbd_voyage_subproblem_cache_hits",
            "lbbd_voyage_subproblem_count",
            "lbbd_voyage_subproblem_build_seconds",
            "lbbd_voyage_subproblem_variable_count",
            "lbbd_best_incumbent_source",
            "lbbd_preparation_seconds",
            "lbbd_master_build_seconds",
            "lbbd_master_feasibility",
            "lbbd_total_solve_seconds",
            "direct_model_build_seconds",
            "direct_total_solve_seconds",
            "direct_bound",
            "direct_mip_gap",
            "direct_solver_incumbent_objective",
            "direct_objective_auxiliary_slack",
        )
        if diagnostics.get(key) is not None
    }
    summary["lbbd_iterations"] = [
        {
            key: row.get(key)
            for key in (
                "iteration",
                "master_status",
                "master_allowance",
                "master_seconds",
                "master_objective",
                "master_bound",
                "active_row_class_count",
                "cuts_added",
                "all_voyage_subproblems_feasible",
                "all_voyage_subproblems_optimal",
                "candidate_objective",
            )
        }
        for row in diagnostics.get("lbbd_iterations", [])
    ]
    for section, keys in (
        (
            "lbbd_master",
            (
                "master_variable_count",
                "group_bay_quantity_count",
                "row_footprint_template_count",
                "voyage_row_class_binary_count",
                "continuous_routing_flow_count",
                "operational_bay_row_count_count",
                "physical_row_owner_constraint_count",
                "owner_positive_flow_constraint_count",
                "row_count_box_upper_constraint_count",
                "row_count_owner_upper_constraint_count",
                "row_class_owner_cover_constraint_count",
                "row_capacity_envelope_count",
                "row_conflict_clique_count",
                "row_conflict_hall_capacity_count",
            ),
        ),
        (
            "direct_model",
            (
                "row_location_variable_count",
                "import_reservation_variable_count",
                "model_variable_count",
            ),
        ),
    ):
        values = diagnostics.get(section, {})
        summary[section] = {
            key: values.get(key) for key in keys if values.get(key) is not None
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
            master_feasibility_time_limit=(
                args.master_feasibility_time_limit
            ),
            master_time_limit=args.master_time_limit,
            voyage_time_limit=args.voyage_time_limit,
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
