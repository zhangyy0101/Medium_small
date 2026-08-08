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
from yard_planning.logic_benders import LogicBendersConfig
from yard_planning.planner import ColumnGenerationConfig, write_json
from yard_planning.selective_resource_benders import (
    SelectiveResourceBendersPlanner,
)


def _selective_summary(diagnostics: dict) -> dict:
    summary = {
        key: diagnostics[key]
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
        )
        if diagnostics.get(key) is not None
    }
    for key in (
        "selective_profile_state_count",
        "selective_initial_profile_state_count",
        "selective_promoted_profile_state_count",
        "selective_promoted_row_count",
        "selective_conflict_hyperedge_count",
        "selective_initial_conflict_hyperedge_coverage",
        "selective_initial_conflict_seed_count",
        "selective_profile_state_score_min",
        "selective_profile_state_score_max",
        "selective_oracle_solve_count",
        "selective_oracle_cache_hits",
        "selective_oracle_seconds",
        "conflict_repair_solve_count",
        "conflict_repair_success_count",
        "conflict_repair_seconds",
        "restricted_primal_solve_count",
        "restricted_primal_success_count",
        "restricted_primal_seconds",
        "selective_lbbd_converged",
        "selective_lbbd_termination_reason",
        "selective_lbbd_master_round_count",
        "selective_lbbd_best_master_round",
        "selective_lbbd_feasibility_cut_count",
        "selective_lbbd_optimality_cut_count",
        "selective_lbbd_local_optimality_cut_count",
        "selective_lbbd_global_optimality_cut_count",
        "selective_lbbd_cut_binary_count",
        "selective_lbbd_state_promotion_round_count",
        "selective_lbbd_bound_tightening_cut_count",
        "selective_lbbd_oracle_build_seconds",
        "selective_lbbd_oracle_variable_count",
        "resource_capacity_certificate_solve_count",
        "resource_capacity_certificate_seconds",
        "selective_lbbd_repair_solve_count",
        "selective_lbbd_repair_success_count",
        "selective_lbbd_repair_seconds",
        "selective_lbbd_restricted_primal_solve_count",
        "selective_lbbd_restricted_primal_success_count",
        "selective_lbbd_restricted_primal_seconds",
        "selective_lbbd_restricted_primal_solve_limit",
        "selective_lbbd_voyage_bound_solve_count",
        "selective_lbbd_voyage_bound_cache_hits",
        "selective_lbbd_voyage_bound_seconds",
        "selective_lbbd_preparation_seconds",
        "selective_lbbd_master_build_seconds",
        "selective_lbbd_master_start",
        "selective_lbbd_primal_start",
        "selective_lbbd_total_solve_seconds",
        "selective_lbbd_time_budget_policy",
    ):
        if diagnostics.get(key) is not None:
            summary[key] = diagnostics[key]
    summary["selective_lbbd_master_rounds"] = diagnostics.get(
        "selective_lbbd_master_rounds", []
    )
    summary["selective_lbbd_cut_records"] = diagnostics.get(
        "selective_lbbd_cut_records", []
    )
    summary["selective_lbbd_state_promotion_records"] = diagnostics.get(
        "selective_lbbd_state_promotion_records", []
    )
    summary["selective_lbbd_oracle_records"] = diagnostics.get(
        "selective_lbbd_oracle_records", []
    )
    summary["selective_lbbd_voyage_bound_records"] = diagnostics.get(
        "selective_lbbd_voyage_bound_records", []
    )
    summary["selective_lbbd_repair_records"] = diagnostics.get(
        "selective_lbbd_repair_records", []
    )
    summary["selective_lbbd_restricted_primal_records"] = diagnostics.get(
        "selective_lbbd_restricted_primal_records", []
    )
    summary["selective_lbbd_master"] = diagnostics.get(
        "selective_lbbd_master", {}
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark selective resource-state LBBD on the common model."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--large-plan", type=Path, default=DEFAULT_LARGE_PLAN)
    parser.add_argument("--voyages", nargs="+", default=None)
    parser.add_argument("--planning-time", default=None)
    parser.add_argument("--total-time-limit", type=float, default=120.0)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--max-cut-rounds", type=int, default=40)
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
        mip_gap=0.0,
        solver_threads=args.solver_threads,
        verbose=False,
    )
    selective = SelectiveResourceBendersPlanner(
        inputs.problem,
        config,
        LogicBendersConfig(
            max_iterations=args.max_cut_rounds,
        ),
    ).solve()
    results = {
        "selective_lbbd": _selective_summary(selective.diagnostics)
    }
    if args.compare_direct:
        direct = DirectMilpPlanner(inputs.problem, config).solve()
        results["direct"] = _summary(direct.diagnostics)
        results["objective_difference"] = round(
            float(selective.diagnostics["final_business_objective"])
            - float(direct.diagnostics["final_business_objective"]),
            10,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.output is not None:
        write_json(args.output.resolve(), results)


if __name__ == "__main__":
    main()
