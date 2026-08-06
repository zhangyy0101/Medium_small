from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd, normalize_voyage_id
from adapters.planning_input import load_planning_inputs
from yard_planning.direct_milp import DirectMilpPlanner
from yard_planning.logic_benders import (
    LogicBendersConfig,
    LogicBendersPlanner,
)
from yard_planning.planner import (
    ColumnGenerationConfig,
    write_selected_locations,
    write_json,
    write_rows,
)
from yard_planning.output_validator import validate_output_files
from yard_planning.voyage_plan_column_generation import (
    VoyagePlanColumnGenerationPlanner,
    VoyagePlanPricingConfig,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "example" / "input_data.json"
DEFAULT_LARGE_PLAN = ROOT / "example" / "large_plan.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the export container row-allocation model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Full InputAdapterGd JSON input.")
    parser.add_argument("--large-plan", type=Path, default=DEFAULT_LARGE_PLAN, help="Large-plan allocation CSV.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--solver",
        choices=("cg", "direct", "lbbd"),
        default="cg",
        help=(
            "cg runs column generation; direct runs M0; lbbd runs the "
            "capacity- and conflict-strengthened logic-based Benders solver."
        ),
    )
    parser.add_argument("--voyages", nargs="+", default=None, help="Optional voyage subset; default is every voyage in the large plan.")
    parser.add_argument("--planning-time", default=None, help="Optional override; default is the JSON planning_time.")
    parser.add_argument("--total-time-limit", type=float, default=60.0)
    parser.add_argument("--mip-time-limit", type=float, default=15.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--max-pricing-iterations", type=int, default=60)
    parser.add_argument(
        "--plans-per-pricing",
        type=int,
        default=3,
        help="Maximum complete voyage plans returned by one pricing MIP.",
    )
    parser.add_argument(
        "--solver-threads",
        type=int,
        default=0,
        help="0 lets Gurobi choose; use a fixed positive value for experiments.",
    )
    parser.add_argument("--lbbd-max-iterations", type=int, default=40)
    parser.add_argument("--lbbd-master-time-limit", type=float, default=10.0)
    parser.add_argument("--lbbd-area-time-limit", type=float, default=5.0)
    parser.add_argument("--lbbd-primal-seed-time-limit", type=float, default=10.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    runtime_start = perf_counter()
    args = parse_args()
    input_path = args.input.resolve()
    large_plan_path = args.large_plan.resolve()
    stage_start = perf_counter()
    adapter = InputAdapterGd.load_from_json(str(input_path))
    input_json_seconds = perf_counter() - stage_start
    stage_start = perf_counter()
    large_plan = pd.read_csv(large_plan_path)
    large_plan_csv_seconds = perf_counter() - stage_start
    voyages = resolve_voyages(large_plan, args.voyages)
    planning_time = pd.Timestamp(args.planning_time if args.planning_time else adapter.planning_time)
    if pd.isna(planning_time):
        raise SystemExit("planning_time is missing or invalid")

    output_dir = create_output_dir(args.output_root.resolve(), args.run_name)
    print(f"input: {input_path}")
    print(f"large plan: {large_plan_path}")
    print(f"voyages ({len(voyages)}): {voyages}")
    print(f"output: {output_dir}")

    stage_start = perf_counter()
    inputs = load_planning_inputs(
        adapter,
        planning_time=planning_time.to_pydatetime(),
        voyages=voyages,
        big_plan=large_plan,
    )
    planning_input_seconds = perf_counter() - stage_start
    config = ColumnGenerationConfig(
        max_iterations=args.max_pricing_iterations,
        total_time_limit=args.total_time_limit,
        mip_time_limit=args.mip_time_limit,
        mip_gap=args.mip_gap,
        solver_threads=args.solver_threads,
        verbose=not args.quiet,
    )
    stage_start = perf_counter()
    if args.solver == "direct":
        planner = DirectMilpPlanner(inputs.problem, config)
    elif args.solver == "lbbd":
        planner = LogicBendersPlanner(
            inputs.problem,
            config,
            LogicBendersConfig(
                max_iterations=args.lbbd_max_iterations,
                master_time_limit=args.lbbd_master_time_limit,
                area_time_limit=args.lbbd_area_time_limit,
                primal_seed_time_limit=args.lbbd_primal_seed_time_limit,
            ),
        )
    else:
        planner = VoyagePlanColumnGenerationPlanner(
            inputs.problem,
            config,
            VoyagePlanPricingConfig(
                plans_per_pricing=args.plans_per_pricing,
            ),
        )
    planner_initialization_seconds = perf_counter() - stage_start
    stage_start = perf_counter()
    result = planner.solve()
    optimization_seconds = perf_counter() - stage_start
    runtime_breakdown = {
        "input_json": round(input_json_seconds, 3),
        "large_plan_csv": round(large_plan_csv_seconds, 3),
        "planning_input_preparation": round(planning_input_seconds, 3),
        "planner_initialization": round(planner_initialization_seconds, 3),
        "optimization": round(optimization_seconds, 3),
        "through_optimization_total": round(perf_counter() - runtime_start, 3),
    }
    result.diagnostics["runtime_breakdown_seconds"] = runtime_breakdown

    write_rows(output_dir / "bay_summary.csv", result.bay_summary_rows)
    write_rows(output_dir / "export_row_plan.csv", result.export_rows)
    write_rows(
        output_dir / "import_capacity_reservation.csv",
        result.import_reservation_rows,
    )
    write_selected_locations(
        output_dir / "selected_row_locations.csv",
        result.columns,
    )
    write_rows(output_dir / "declared_export_demand.csv", [asdict(row) for row in inputs.demand_rows])
    large_plan.to_csv(output_dir / "large_plan_used.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "diagnostics.json", result.diagnostics)
    output_validation = validate_output_files(
        inputs.problem,
        output_dir / "export_row_plan.csv",
        output_dir / "import_capacity_reservation.csv",
    )
    write_json(output_dir / "output_validation.json", output_validation)
    write_json(
        output_dir / "run_summary.json",
        {
            "input": str(input_path),
            "large_plan": str(large_plan_path),
            "planning_time": planning_time.isoformat(),
            "voyages": voyages,
            "bay_summary_row_count": len(result.bay_summary_rows),
            "export_row_plan_row_count": len(result.export_rows),
            "import_capacity_reservation_row_count": len(result.import_reservation_rows),
            "algorithm": result.diagnostics.get("algorithm"),
            "master_status": result.diagnostics.get("master_status"),
            "master_bound_scope": result.diagnostics.get("master_bound_scope"),
            "root_lp_lower_bound": result.diagnostics.get(
                "root_lp_lower_bound"
            ),
            "root_lp_exact": result.diagnostics.get(
                "root_lp_exact"
            ),
            "complete_model_lower_bound": result.diagnostics.get(
                "complete_model_lower_bound"
            ),
            "reported_mip_gap": result.diagnostics.get("master_mip_gap"),
            "complete_model_absolute_gap": result.diagnostics.get(
                "complete_model_absolute_gap"
            ),
            "complete_model_relative_gap": result.diagnostics.get(
                "complete_model_relative_gap"
            ),
            "complete_model_gap_source": result.diagnostics.get(
                "complete_model_gap_source"
            ),
            "direct_solve_seconds": result.diagnostics.get(
                "direct_total_solve_seconds"
            ),
            "direct_bound": result.diagnostics.get("direct_bound"),
            "runtime_breakdown_seconds": runtime_breakdown,
        },
    )
    print(f"runtime_seconds: {runtime_breakdown}")
    print(f"bay_summary: {output_dir / 'bay_summary.csv'}")
    print(f"export_row_plan: {output_dir / 'export_row_plan.csv'}")
    print(f"import_capacity_reservation: {output_dir / 'import_capacity_reservation.csv'}")
    print(f"diagnostics: {output_dir / 'diagnostics.json'}")


def resolve_voyages(large_plan: pd.DataFrame, requested: list[str] | None) -> list[str]:
    if requested:
        values = requested
    else:
        voyage_column = next((name for name in ("voy_id", "voyage_id", "VOY_ID") if name in large_plan.columns), None)
        if voyage_column is None:
            raise SystemExit("large plan must contain voy_id or voyage_id")
        values = large_plan[voyage_column].dropna().tolist()
    voyages = sorted({normalize_voyage_id(value) for value in values if normalize_voyage_id(value)})
    if not voyages:
        raise SystemExit("no voyages found in the large plan")
    return voyages


def create_output_dir(output_root: Path, run_name: str | None) -> Path:
    name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


if __name__ == "__main__":
    main()
