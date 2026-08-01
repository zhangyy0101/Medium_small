from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd, normalize_voyage_id
from adapters.input_adapter_standard import DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO, load_medium_small_inputs
from medium_small.column_generation_planner import (
    ColumnGenerationConfig,
    ColumnGenerationPlanner,
    write_columns,
    write_json,
    write_rows,
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
    parser.add_argument("--voyages", nargs="+", default=None, help="Optional voyage subset; default is every voyage in the large plan.")
    parser.add_argument("--planning-time", default=None, help="Optional override; default is the JSON planning_time.")
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument("--total-time-limit", type=float, default=240.0)
    parser.add_argument("--mip-time-limit", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--misplaced-bay-exclusion-ratio", type=float, default=DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-scip", action="store_true", help="Use the built-in fallback instead of SCIP.")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    args = parse_args()
    input_path = args.input.resolve()
    large_plan_path = args.large_plan.resolve()
    adapter = InputAdapterGd.load_from_json(str(input_path))
    large_plan = pd.read_csv(large_plan_path)
    voyages = resolve_voyages(large_plan, args.voyages)
    planning_time = pd.Timestamp(args.planning_time if args.planning_time else adapter.planning_time)
    if pd.isna(planning_time):
        raise SystemExit("planning_time is missing or invalid")

    output_dir = create_output_dir(args.output_root.resolve(), args.run_name)
    print(f"input: {input_path}")
    print(f"large plan: {large_plan_path}")
    print(f"voyages ({len(voyages)}): {voyages}")
    print(f"output: {output_dir}")

    inputs = load_medium_small_inputs(
        adapter,
        planning_time=planning_time.to_pydatetime(),
        voyages=voyages,
        horizon_hours=args.horizon_hours,
        misplaced_bay_exclusion_ratio=args.misplaced_bay_exclusion_ratio,
        big_plan=large_plan,
    )
    config = ColumnGenerationConfig(
        total_time_limit=args.total_time_limit,
        mip_time_limit=args.mip_time_limit,
        mip_gap=args.mip_gap,
        verbose=not args.quiet,
        use_scip=not args.no_scip,
    )
    result = ColumnGenerationPlanner(inputs.problem, config).solve()

    write_rows(output_dir / "area_bay_summary.csv", result.medium_rows)
    write_rows(output_dir / "export_row_plan.csv", result.small_rows)
    write_rows(output_dir / "unplaced_boxes.csv", result.unplaced_rows)
    write_columns(output_dir / "generated_columns.csv", result.columns)
    write_rows(output_dir / "declared_export_demand.csv", [asdict(row) for row in inputs.demand_rows])
    large_plan.to_csv(output_dir / "large_plan_used.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "diagnostics.json", result.diagnostics)
    write_json(
        output_dir / "run_summary.json",
        {
            "input": str(input_path),
            "large_plan": str(large_plan_path),
            "planning_time": planning_time.isoformat(),
            "voyages": voyages,
            "area_bay_summary_row_count": len(result.medium_rows),
            "export_row_plan_row_count": len(result.small_rows),
            "unplaced_row_count": len(result.unplaced_rows),
            "unplaced_boxes": result.diagnostics.get("unplaced_boxes"),
            "algorithm": result.diagnostics.get("algorithm"),
            "master_status": result.diagnostics.get("master_status"),
        },
    )
    print(f"area_bay_summary: {output_dir / 'area_bay_summary.csv'}")
    print(f"export_row_plan: {output_dir / 'export_row_plan.csv'}")
    print(f"unplaced_boxes: {output_dir / 'unplaced_boxes.csv'}")
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
