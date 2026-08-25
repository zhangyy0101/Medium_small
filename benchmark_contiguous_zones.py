from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd, normalize_voyage_id
from adapters.planning_input import classified_export_voyages, load_planning_inputs
from yard_planning.contiguous_zone_generation import (
    ContiguousZoneConfig,
    ContiguousZoneGenerationPlanner,
)
from yard_planning.direct_milp import DirectMilpPlanner
from yard_planning.planner import ColumnGenerationConfig, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "example" / "input_data.json"
def resolve_voyages(
    adapter: InputAdapterGd,
    requested: list[str] | None,
) -> list[str]:
    if requested:
        values = requested
    else:
        values = sorted(classified_export_voyages(adapter))
    voyages = sorted(
        {
            normalized
            for value in values
            if (normalized := normalize_voyage_id(value))
        }
    )
    if not voyages:
        raise SystemExit("no declared export voyages found in the shared input")
    return voyages


def summarize(diagnostics: dict) -> dict:
    keys = (
        "algorithm",
        "model_scope",
        "planned_group_count",
        "planned_box_count",
        "candidate_row_location_count",
        "zone_model_upper_bound",
        "zone_model_global_lower_bound",
        "zone_model_absolute_gap",
        "zone_model_relative_gap",
        "zone_model_lower_bound_source",
        "zone_selected_candidate_count",
        "zone_candidate_reduction",
        "total_seconds",
    )
    return {
        key: diagnostics[key]
        for key in keys
        if diagnostics.get(key) is not None
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve and benchmark the contiguous storage-zone model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--voyages", nargs="+", default=None)
    parser.add_argument("--planning-time", default=None)
    parser.add_argument("--total-time-limit", type=float, default=60.0)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--max-root-iterations", type=int, default=60)
    parser.add_argument("--columns-per-group", type=int, default=3)
    parser.add_argument("--integer-pool-per-group", type=int, default=100)
    parser.add_argument("--zone-mip-time-fraction", type=float, default=0.75)
    parser.add_argument("--fix-optimize-local-fraction", type=float, default=0.85)
    parser.add_argument("--fix-optimize-objective-mass", type=float, default=0.60)
    parser.add_argument("--fix-optimize-zone-fraction", type=float, default=0.35)
    parser.add_argument(
        "--fix-optimize-policy",
        choices=("disabled", "objective"),
        default="objective",
    )
    parser.add_argument("--fill-time-fraction", type=float, default=0.05)
    parser.add_argument(
        "--peak-utilization-headroom-fraction",
        type=float,
        default=0.50,
        help=(
            "Fraction of the distance from the instance utilization lower "
            "bound to 100%% retained as epsilon headroom."
        ),
    )
    parser.add_argument("--root-only", action="store_true")
    parser.add_argument("--complete-zone-mip-only", action="store_true")
    parser.add_argument("--compare-complete-zone-lp", action="store_true")
    parser.add_argument("--compare-complete-zone-mip", action="store_true")
    parser.add_argument(
        "--compare-row-m0",
        action="store_true",
        help=(
            "Run row-level M0 only as a different-model "
            "structural reference; objectives are not subtracted."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = InputAdapterGd.load_from_json(str(args.input.resolve()))
    voyages = resolve_voyages(adapter, args.voyages)
    planning_time = pd.Timestamp(
        args.planning_time if args.planning_time else adapter.planning_time
    )
    if pd.isna(planning_time):
        raise SystemExit("planning_time is missing or invalid")
    inputs = load_planning_inputs(
        adapter,
        planning_time=planning_time.to_pydatetime(),
        voyages=voyages,
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
        fix_optimize_objective_mass=args.fix_optimize_objective_mass,
        fix_optimize_zone_fraction=args.fix_optimize_zone_fraction,
        fix_optimize_policy=args.fix_optimize_policy,
        fill_time_fraction=args.fill_time_fraction,
        peak_utilization_headroom_fraction=(
            args.peak_utilization_headroom_fraction
        ),
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
        output = {"contiguous_zone": summarize(result.diagnostics)}
        output["contiguous_zone_diagnostics"] = {
            key: value
            for key, value in result.diagnostics.items()
            if key.startswith("zone_")
            or key
            in {
                "business_objective",
                "peak_utilization_policy",
                "independent_solution_validation",
                "total_seconds",
            }
        }
        if args.compare_row_m0:
            direct = DirectMilpPlanner(inputs.problem, common).solve()
            output["row_m0_different_model_reference"] = summarize(
                direct.diagnostics
            )
            output["cross_model_objective_difference_reported"] = False
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.output is not None:
        write_json(args.output.resolve(), output)


if __name__ == "__main__":
    main()
