"""Small-instance correctness CLI for exact V6 root column generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd, normalize_voyage_id
from adapters.planning_input import classified_export_voyages, load_planning_inputs
from yard_planning.v6_column_generation import (
    V6RootCgConfig,
    V6RootColumnGeneration,
)
from yard_planning.v6_complete_mip import V6CompleteMipConfig, V6CompleteMipSolver
from yard_planning.v6_model import V6ObjectiveConfig
from yard_planning.v6_primal_coverage import (
    V6AnalyticPeakConfig,
    V6AnalyticPeakPreparationSolver,
    V6CompactPeakConfig,
    V6CompactPeakUtilizationSolver,
    V6PrimalCoverageConfig,
    V6PrimalCoveredPipeline,
)
from yard_planning.v6_restricted_integer import (
    V6RestrictedIntegerConfig,
    V6RootCgIntegerPipeline,
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
            "Prepare the V6 epsilon cap, close the exact business root LP, and "
            "optionally construct and validate an integer incumbent."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--voyages", nargs="+", default=None)
    parser.add_argument("--planning-time", default=None)
    parser.add_argument("--peak-time-limit", type=float, default=60.0)
    parser.add_argument(
        "--peak-feasibility-time-limit", type=float, default=10.0
    )
    parser.add_argument(
        "--peak-method",
        choices=("analytic", "compact", "complete"),
        default="analytic",
        help=(
            "analytic is the production policy with a feasibility witness; "
            "compact and complete prove exact rho* only as diagnostics/oracles"
        ),
    )
    parser.add_argument("--maximum-zone-count", type=int, default=250_000)
    parser.add_argument("--pricing-time-limit", type=float, default=60.0)
    parser.add_argument("--root-total-time-limit", type=float, default=None)
    parser.add_argument("--maximum-phase-one-iterations", type=int, default=200)
    parser.add_argument("--maximum-business-iterations", type=int, default=500)
    parser.add_argument(
        "--phase-one-columns-per-group", type=int, default=512
    )
    parser.add_argument(
        "--business-columns-per-group", type=int, default=512
    )
    parser.add_argument(
        "--maximum-patterns-per-interval", type=int, default=8
    )
    parser.add_argument(
        "--restricted-integer-time-limit",
        type=float,
        default=None,
        help=(
            "when provided, solve the integer master over exactly the root-CG "
            "column pool; no full-zone or heuristic fallback is used"
        ),
    )
    parser.add_argument("--restricted-integer-mip-gap", type=float, default=0.0)
    parser.add_argument(
        "--primal-coverage-time-limit",
        type=float,
        default=None,
        help=(
            "when provided, generate V6-native compact row-atom primal "
            "columns before solving the restricted integer master"
        ),
    )
    parser.add_argument("--primal-coverage-mip-gap", type=float, default=0.0)
    parser.add_argument(
        "--primal-coverage-pool-solutions", type=int, default=4
    )
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument(
        "--peak-utilization-headroom-fraction", type=float, default=0.50
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
    peak_witness = None
    if args.peak_method == "analytic":
        analytic_peak = V6AnalyticPeakPreparationSolver(
            inputs.problem,
            V6AnalyticPeakConfig(
                feasibility_time_limit=args.peak_feasibility_time_limit,
                solver_threads=args.solver_threads,
                solver_seed=args.solver_seed,
                verbose=args.verbose,
                objective=objective,
            ),
        ).solve()
        peak_policy = analytic_peak.peak_policy
        peak_diagnostics = analytic_peak.diagnostics
        peak_witness = analytic_peak.witness
    elif args.peak_method == "compact":
        compact_peak = V6CompactPeakUtilizationSolver(
            inputs.problem,
            V6CompactPeakConfig(
                time_limit=args.peak_time_limit,
                solver_threads=args.solver_threads,
                solver_seed=args.solver_seed,
                verbose=args.verbose,
                objective=objective,
            ),
        ).solve()
        peak_policy = compact_peak.peak_policy
        peak_diagnostics = compact_peak.diagnostics
    else:
        peak_solver = V6CompleteMipSolver(
            inputs.problem,
            V6CompleteMipConfig(
                peak_time_limit=args.peak_time_limit,
                business_time_limit=1.0,
                solver_threads=args.solver_threads,
                solver_seed=args.solver_seed,
                verbose=args.verbose,
                maximum_zone_count=args.maximum_zone_count,
                objective=objective,
            ),
        )
        peak_policy, peak_diagnostics = peak_solver.solve_peak_reference()
    root_config = V6RootCgConfig(
        maximum_phase_one_iterations=args.maximum_phase_one_iterations,
        maximum_business_iterations=args.maximum_business_iterations,
        phase_one_columns_per_group=args.phase_one_columns_per_group,
        business_columns_per_group=args.business_columns_per_group,
        maximum_patterns_per_interval=(
            args.maximum_patterns_per_interval
        ),
        pricing_time_limit=args.pricing_time_limit,
        total_time_limit=args.root_total_time_limit,
        solver_threads=args.solver_threads,
        solver_seed=args.solver_seed,
        verbose=args.verbose,
        objective=objective,
    )
    pipeline_result = None
    covered_pipeline_result = None
    if args.primal_coverage_time_limit is not None:
        final_integer_time = (
            args.restricted_integer_time_limit
            if args.restricted_integer_time_limit is not None
            else args.primal_coverage_time_limit
        )
        covered_pipeline_result = V6PrimalCoveredPipeline(
            inputs.problem,
            peak_policy,
            root_config,
            V6PrimalCoverageConfig(
                time_limit=args.primal_coverage_time_limit,
                mip_gap=args.primal_coverage_mip_gap,
                solver_threads=args.solver_threads,
                solver_seed=args.solver_seed,
                verbose=args.verbose,
                maximum_pool_solutions=args.primal_coverage_pool_solutions,
                objective=objective,
            ),
            V6RestrictedIntegerConfig(
                time_limit=final_integer_time,
                mip_gap=args.restricted_integer_mip_gap,
                solver_threads=args.solver_threads,
                solver_seed=args.solver_seed,
                verbose=args.verbose,
                objective=objective,
            ),
            coverage_warm_start=peak_witness,
        ).solve()
        result = covered_pipeline_result.root
    elif args.restricted_integer_time_limit is None:
        result = V6RootColumnGeneration(
            inputs.problem,
            peak_policy,
            root_config,
        ).solve(
            initial_zones=(peak_witness.zones if peak_witness is not None else ())
        )
    else:
        pipeline_result = V6RootCgIntegerPipeline(
            inputs.problem,
            peak_policy,
            root_config,
            V6RestrictedIntegerConfig(
                time_limit=args.restricted_integer_time_limit,
                mip_gap=args.restricted_integer_mip_gap,
                solver_threads=args.solver_threads,
                solver_seed=args.solver_seed,
                verbose=args.verbose,
                objective=objective,
            ),
        ).solve()
        result = pipeline_result.root
    payload = {
        "model_schema_version": result.diagnostics["model_schema_version"],
        "result_role": result.diagnostics["result_role"],
        "root_lp_objective": result.objective,
        "peak_reference": peak_diagnostics,
        "peak_method": args.peak_method,
        "peak_policy": peak_policy.as_dict(),
        "generated_zones": [
            {
                "zone_id": zone.zone_id,
                "group_id": zone.group_id,
                "area_no": zone.area_no,
                "anchor_bay_keys": list(zone.anchor_bay_keys),
                "rows_by_anchor_bay": [
                    {"bay_key": bay_key, "row_nos": list(rows)}
                    for bay_key, rows in zone.rows_by_anchor_bay
                ],
                "capacity": zone.capacity,
                "root_value": result.zone_values.get(zone.zone_id, 0.0),
            }
            for zone in result.zones
        ],
        "group_bay_flow": [
            {
                "group_id": group_id,
                "bay_key": bay_key,
                "boxes": value,
            }
            for (group_id, bay_key), value in sorted(
                result.group_bay_flow.items()
            )
        ],
        "anonymous_import_reservation": [
            {
                "flow": flow,
                "size": size,
                "bay_key": bay_key,
                "boxes": value,
            }
            for (flow, size, bay_key), value in sorted(
                result.import_reservation.items()
            )
        ],
        "diagnostics": result.diagnostics,
    }
    if pipeline_result is not None:
        payload["restricted_integer"] = pipeline_result.integer.as_dict()
        payload["root_integer_diagnostics"] = pipeline_result.diagnostics
    if covered_pipeline_result is not None:
        payload["primal_coverage"] = {
            "selected_atom_indices": list(
                covered_pipeline_result.coverage.selected_atom_indices
            ),
            "objective": covered_pipeline_result.coverage.objective,
            "certificate": dict(
                covered_pipeline_result.coverage.certificate
            ),
            "diagnostics": dict(
                covered_pipeline_result.coverage.diagnostics
            ),
        }
        payload["restricted_integer"] = (
            covered_pipeline_result.integer.as_dict()
        )
        payload["root_integer_diagnostics"] = (
            covered_pipeline_result.diagnostics
        )
    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        displayed = {
            "output": str(output_path),
            "result_role": payload["result_role"],
            "root_lp_objective": payload["root_lp_objective"],
            "generated_zone_count": len(result.zones),
            "phase_one_iterations": result.diagnostics["phase_one"]["iterations"],
            "business_iterations": result.diagnostics["business"]["iterations"],
        }
        if pipeline_result is not None:
            displayed.update(
                {
                    "restricted_integer_objective": (
                        pipeline_result.integer.objective
                    ),
                    "restricted_integer_status": (
                        pipeline_result.integer.diagnostics["status"]
                    ),
                    "relative_root_gap": pipeline_result.diagnostics[
                        "relative_root_gap"
                    ],
                }
            )
        if covered_pipeline_result is not None:
            displayed.update(
                {
                    "primal_coverage_objective": (
                        covered_pipeline_result.coverage.objective
                    ),
                    "covered_integer_objective": (
                        covered_pipeline_result.integer.objective
                    ),
                    "coverage_added_zone_count": (
                        covered_pipeline_result.diagnostics[
                            "coverage_added_zone_count"
                        ]
                    ),
                    "relative_root_gap": covered_pipeline_result.diagnostics[
                        "relative_root_gap"
                    ],
                }
            )
    else:
        displayed = payload
    print(json.dumps(displayed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
