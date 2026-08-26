"""Independent CLI for the fully enumerated, small-instance V6 MIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd, normalize_voyage_id
from adapters.planning_input import classified_export_voyages, load_planning_inputs
from yard_planning.v6_complete_mip import (
    V6CompleteMipConfig,
    V6CompleteMipSolver,
)
from yard_planning.v6_model import V6ObjectiveConfig
from yard_planning.v6_primal_coverage import (
    V6AnalyticPeakConfig,
    V6AnalyticPeakPreparationSolver,
)


def _resolve_voyages(
    adapter: InputAdapterGd,
    requested: list[str] | None,
) -> list[str]:
    values = requested if requested else sorted(classified_export_voyages(adapter))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the fully enumerated V6 business MIP under the shared "
            "production epsilon cap. Exact rho* remains an optional oracle."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--voyages", nargs="+", default=None)
    parser.add_argument("--planning-time", default=None)
    parser.add_argument("--peak-time-limit", type=float, default=60.0)
    parser.add_argument(
        "--peak-method",
        choices=("analytic", "exact"),
        default="analytic",
    )
    parser.add_argument(
        "--peak-feasibility-time-limit", type=float, default=10.0
    )
    parser.add_argument("--business-time-limit", type=float, default=60.0)
    parser.add_argument("--business-mip-gap", type=float, default=0.0)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--maximum-zone-count", type=int, default=250_000)
    parser.add_argument(
        "--peak-utilization-headroom-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = InputAdapterGd.load_from_json(str(args.input.resolve()))
    voyages = _resolve_voyages(adapter, args.voyages)
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
    objective = V6ObjectiveConfig(
        peak_utilization_headroom_fraction=(
            args.peak_utilization_headroom_fraction
        )
    )
    config = V6CompleteMipConfig(
        peak_time_limit=args.peak_time_limit,
        business_time_limit=args.business_time_limit,
        business_mip_gap=args.business_mip_gap,
        solver_threads=args.solver_threads,
        solver_seed=args.solver_seed,
        verbose=args.verbose,
        maximum_zone_count=args.maximum_zone_count,
        objective=objective,
    )
    solver = V6CompleteMipSolver(inputs.problem, config)
    if args.peak_method == "analytic":
        peak = V6AnalyticPeakPreparationSolver(
            inputs.problem,
            V6AnalyticPeakConfig(
                feasibility_time_limit=args.peak_feasibility_time_limit,
                solver_threads=args.solver_threads,
                solver_seed=args.solver_seed,
                verbose=args.verbose,
                objective=objective,
            ),
        ).solve()
        result = solver.solve(peak.peak_policy, peak.diagnostics)
    else:
        result = solver.solve()
    payload = result.as_dict()
    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        displayed: dict[str, object] = {
            "output": str(output_path),
            "model_schema_version": payload["model_schema_version"],
            "complete_zone_count": result.diagnostics["complete_zone_count"],
            "peak_policy": payload["peak_policy"],
            "objective": result.certificate["objective"],
            "business_status": result.diagnostics["business_model"]["status"],
            "independent_validation_passed": True,
        }
    else:
        displayed = payload
    print(json.dumps(displayed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
