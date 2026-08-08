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
from yard_planning.template_flow_benders import (
    TemplateFlowBendersConfig,
    TemplateFlowBendersPlanner,
    TemplateFlowDirectPlanner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the independent row-template classical Benders model."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--large-plan", type=Path, default=DEFAULT_LARGE_PLAN)
    parser.add_argument("--voyages", nargs="+", default=None)
    parser.add_argument("--planning-time", default=None)
    parser.add_argument("--total-time-limit", type=float, default=120.0)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=60)
    parser.add_argument("--compare-direct", action="store_true")
    parser.add_argument("--compare-template-direct", action="store_true")
    parser.add_argument(
        "--template-direct-only",
        action="store_true",
        help="solve only the compact direct reference for the redefined model",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _template_summary(diagnostics: dict) -> dict:
    keys = (
        "algorithm",
        "master_status",
        "planned_group_count",
        "planned_box_count",
        "candidate_row_location_count",
        "final_business_objective",
        "template_flow_realized_allocation_objective",
        "complete_model_lower_bound",
        "complete_model_absolute_gap",
        "complete_model_relative_gap",
        "template_flow_handling_class_count",
        "template_flow_template_count",
        "template_flow_arc_count",
        "template_flow_converged",
        "template_flow_termination_reason",
        "template_flow_iteration_count",
        "template_flow_root_closed",
        "template_flow_root_round_count",
        "template_flow_integer_round_count",
        "template_flow_feasibility_cut_count",
        "template_flow_optimality_cut_count",
        "template_flow_best_max_fractionality",
        "template_flow_preparation_seconds",
        "template_flow_master_build_seconds",
        "template_flow_recourse_build_seconds",
        "template_flow_seed_status",
        "template_flow_seed_allowance",
        "template_flow_seed_objective",
        "template_flow_seed_recourse_status",
        "template_flow_seed_recourse_objective",
        "template_flow_total_seconds",
        "template_flow_time_policy",
    )
    result = {
        key: diagnostics[key]
        for key in keys
        if diagnostics.get(key) is not None
    }
    result["template_flow_rounds"] = diagnostics.get(
        "template_flow_rounds", []
    )
    return result


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
    if args.template_direct_only:
        template_direct = TemplateFlowDirectPlanner(
            inputs.problem,
            common,
        ).solve()
        output = {"template_direct": _summary(template_direct.diagnostics)}
        print(json.dumps(output, ensure_ascii=False, indent=2))
        if args.output is not None:
            write_json(args.output.resolve(), output)
        return

    result = TemplateFlowBendersPlanner(
        inputs.problem,
        common,
        TemplateFlowBendersConfig(max_iterations=args.max_iterations),
    ).solve()
    output = {"template_flow_benders": _template_summary(result.diagnostics)}
    if args.compare_direct:
        direct = DirectMilpPlanner(inputs.problem, common).solve()
        output["direct"] = _summary(direct.diagnostics)
        output["objective_difference"] = round(
            float(result.diagnostics["final_business_objective"])
            - float(direct.diagnostics["final_business_objective"]),
            10,
        )
    if args.compare_template_direct:
        template_direct = TemplateFlowDirectPlanner(
            inputs.problem,
            common,
        ).solve()
        output["template_direct"] = _summary(template_direct.diagnostics)
        output["template_model_objective_difference"] = round(
            float(result.diagnostics["final_business_objective"])
            - float(template_direct.diagnostics["final_business_objective"]),
            10,
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.output is not None:
        write_json(args.output.resolve(), output)


if __name__ == "__main__":
    main()
