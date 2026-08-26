"""Reproducible, phase-separated scalability measurements for V6.

This module reuses only the deterministic input recipes from ``preexperiment``.
It never calls the frozen V5 planner.  Each V6 phase is run and recorded
separately so a failed feasibility certificate or root proof cannot be
mistaken for a completed production result.
"""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd
from adapters.planning_input import load_planning_inputs
from yard_planning.row_aware_zones import (
    RowAwareZone,
    build_v6_row_aware_bay_atoms,
)
from yard_planning.v6_column_generation import (
    V6RootCgConfig,
    V6RootColumnGeneration,
)
from yard_planning.v6_model import (
    V6_MODEL_SCHEMA_VERSION,
    V6ObjectiveConfig,
)
from yard_planning.v6_primal_coverage import (
    V6AnalyticPeakConfig,
    V6AnalyticPeakPreparationSolver,
    V6CompactPrimalCoverageSolver,
    V6PrimalCoverageConfig,
)
from yard_planning.v6_restricted_integer import (
    V6RestrictedIntegerConfig,
    V6RestrictedIntegerSolver,
    V6RestrictedIntegerWarmStart,
)

from .scenario_generator import ScenarioSpec, load_suite, materialize_scenario


Progress = Callable[[str], None]
Checkpoint = Callable[[Mapping[str, Any]], None]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _stage_error(error: Exception, started: float) -> dict[str, Any]:
    payload = {
        "state": "failed",
        "wall_seconds": perf_counter() - started,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    diagnostics = getattr(error, "diagnostics", None)
    if diagnostics is not None:
        payload["diagnostics"] = dict(diagnostics)
    return payload


def merge_v6_zone_pools(
    root_zones: Sequence[RowAwareZone],
    coverage_zones: Sequence[RowAwareZone],
) -> tuple[RowAwareZone, ...]:
    """Merge proof and primal columns without relying on transient zone IDs."""

    merged: list[RowAwareZone] = []
    signatures: set[tuple[int, ...]] = set()
    for zone in (*root_zones, *coverage_zones):
        signature = tuple(int(value) for value in zone.candidate_indices)
        if signature in signatures:
            continue
        signatures.add(signature)
        merged.append(zone)
    return tuple(merged)


def _problem_summary(problem, atoms, manifest: Mapping[str, Any]) -> dict[str, Any]:
    import_demand = [
        {"flow": str(flow), "size": str(size), "boxes": int(quantity)}
        for (flow, size), quantity in sorted(
            problem.import_demand_by_flow_size.items()
        )
    ]
    return {
        "export_group_count": len(problem.export_groups),
        "export_demand_boxes": sum(
            int(group.demand) for group in problem.export_groups
        ),
        "export_voyage_count": len(problem.target_voyages),
        "anonymous_import_boxes": sum(
            int(quantity)
            for quantity in problem.import_demand_by_flow_size.values()
        ),
        "anonymous_import_demand": import_demand,
        "yard_bay_count": len(problem.bays),
        "yard_area_count": len({bay.area_no for bay in problem.bays.values()}),
        "row_atom_count": len(atoms),
        "candidate_anchor_bay_count": len(
            {atom.anchor_bay_key for atom in atoms}
        ),
        "candidate_physical_row_count": len(
            {resource for atom in atoms for resource in atom.resources}
        ),
        "declared_export_rows": int(manifest["declared_export_rows"]),
        "declared_import_rows": int(manifest["declared_import_rows"]),
        "upstream_area_guidance_present": bool(problem.area_guidance_target),
        "upstream_import_area_reference_present": bool(
            problem.import_area_size_reference
        ),
    }


def run_v6_case(
    base: InputAdapterGd,
    spec: ScenarioSpec,
    *,
    objective: V6ObjectiveConfig,
    peak_config: V6AnalyticPeakConfig,
    root_config: V6RootCgConfig,
    coverage_config: V6PrimalCoverageConfig,
    integer_config: V6RestrictedIntegerConfig,
    stop_after_root: bool = False,
    progress: Progress = lambda _message: None,
    checkpoint: Checkpoint = lambda _payload: None,
) -> dict[str, Any]:
    """Run one case, stopping immediately when a required phase fails."""

    case_started = perf_counter()
    result: dict[str, Any] = {
        "case_id": spec.case_id,
        "seed": int(spec.seed),
        "generation_spec": asdict(spec),
        "model_schema_version": V6_MODEL_SCHEMA_VERSION,
        "algorithm": "v6_analytic_peak_exact_root_compact_primal_rim",
        "status": "running",
        "passed": False,
        "stages": {},
    }
    if not (
        peak_config.objective
        == root_config.objective
        == coverage_config.objective
        == integer_config.objective
        == objective
    ):
        raise ValueError("all V6 scalability phases must share one objective")

    def fail(stage: str, error: Exception, started: float) -> dict[str, Any]:
        result["stages"][stage] = _stage_error(error, started)
        result["status"] = "failed"
        result["failure_stage"] = stage
        result["total_wall_seconds"] = perf_counter() - case_started
        checkpoint(result)
        progress(f"[{spec.case_id}] failed at {stage}: {error}")
        return result

    stage_started = perf_counter()
    progress(f"[{spec.case_id}] materializing V6 input")
    try:
        adapter, manifest = materialize_scenario(base, spec)
        planning_time = pd.Timestamp(adapter.planning_time)
        if pd.isna(planning_time):
            raise ValueError("planning_time is missing or invalid")
        inputs = load_planning_inputs(
            adapter,
            planning_time=planning_time.to_pydatetime(),
            voyages=manifest["selected_export_voyages"],
        )
        problem = inputs.problem
        atoms, _anchor_limits = build_v6_row_aware_bay_atoms(problem)
        if problem.area_guidance_target or problem.import_area_size_reference:
            raise ValueError("materialized V6 input contains upstream area targets")
        result["input_recipe_model_schema_version"] = manifest[
            "model_schema_version"
        ]
        result["problem"] = _problem_summary(problem, atoms, manifest)
        result["stages"]["input"] = {
            "state": "completed",
            "wall_seconds": perf_counter() - stage_started,
        }
        checkpoint(result)
    except Exception as error:
        return fail("input", error, stage_started)

    stage_started = perf_counter()
    progress(f"[{spec.case_id}] preparing and certifying analytic peak cap")
    try:
        peak = V6AnalyticPeakPreparationSolver(problem, peak_config).solve()
        peak_diag = dict(peak.diagnostics)
        result["stages"]["peak_preparation"] = {
            "state": "completed",
            "wall_seconds": perf_counter() - stage_started,
            "analytic_load_lower_bound": (
                peak.peak_policy.minimum_feasible_utilization
            ),
            "epsilon_cap": peak.peak_policy.epsilon_cap,
            "witness_peak_utilization": peak_diag[
                "witness_peak_utilization"
            ],
            "diagnostics": peak_diag,
        }
        checkpoint(result)
    except Exception as error:
        return fail("peak_preparation", error, stage_started)

    stage_started = perf_counter()
    progress(f"[{spec.case_id}] closing exact root LP")
    try:
        root = V6RootColumnGeneration(
            problem,
            peak.peak_policy,
            root_config,
        ).solve(initial_zones=peak.witness.zones)
        result["stages"]["root_cg"] = {
            "state": "completed",
            "wall_seconds": perf_counter() - stage_started,
            "lower_bound": root.objective,
            "generated_zone_count": len(root.zones),
            "diagnostics": dict(root.diagnostics),
        }
        checkpoint(result)
    except Exception as error:
        return fail("root_cg", error, stage_started)

    if stop_after_root:
        result["metrics"] = {
            "analytic_load_lower_bound": (
                peak.peak_policy.minimum_feasible_utilization
            ),
            "epsilon_cap": peak.peak_policy.epsilon_cap,
            "peak_feasibility_witness_utilization": peak.diagnostics[
                "witness_peak_utilization"
            ],
            "root_lp_lower_bound": root.objective,
            "root_zone_count": len(root.zones),
        }
        result["status"] = "completed"
        result["passed"] = True
        result["run_mode"] = "root_only"
        result["total_wall_seconds"] = perf_counter() - case_started
        checkpoint(result)
        progress(
            f"[{spec.case_id}] completed root-only: "
            f"LB={root.objective:.6g}"
        )
        return result

    stage_started = perf_counter()
    progress(f"[{spec.case_id}] constructing compact primal coverage")
    try:
        coverage = V6CompactPrimalCoverageSolver(
            problem,
            peak.peak_policy,
            coverage_config,
            warm_start=peak.witness,
        ).solve()
        result["stages"]["compact_primal"] = {
            "state": "completed",
            "wall_seconds": perf_counter() - stage_started,
            "incumbent_objective": coverage.objective,
            "incumbent_zone_count": len(coverage.zones),
            "candidate_zone_count": len(coverage.candidate_zones),
            "diagnostics": dict(coverage.diagnostics),
        }
        checkpoint(result)
    except Exception as error:
        return fail("compact_primal", error, stage_started)

    stage_started = perf_counter()
    progress(f"[{spec.case_id}] solving merged restricted integer master")
    try:
        merged = merge_v6_zone_pools(root.zones, coverage.candidate_zones)
        integer = V6RestrictedIntegerSolver(
            problem,
            merged,
            peak.peak_policy,
            integer_config,
            warm_start=V6RestrictedIntegerWarmStart.from_zone_solution(
                coverage.zones,
                coverage.selected_zone_ids,
                coverage.zone_bay_flow,
                coverage.import_reservation,
                coverage.objective,
            ),
        ).solve()
        if integer.objective > coverage.objective + 1e-8:
            raise RuntimeError(
                "merged V6 RIM lost the compact primal incumbent: "
                f"rim={integer.objective}, compact={coverage.objective}"
            )
        if integer.objective + 1e-8 < root.objective:
            raise RuntimeError(
                "validated V6 incumbent is below the exact root bound: "
                f"ub={integer.objective}, lb={root.objective}"
            )
        absolute_gap = max(0.0, integer.objective - root.objective)
        scale_normalized_gap = absolute_gap / max(
            1.0, abs(integer.objective)
        )
        incumbent_relative_gap = absolute_gap / max(
            1e-12, abs(integer.objective)
        )
        result["stages"]["final_rim"] = {
            "state": "completed",
            "wall_seconds": perf_counter() - stage_started,
            "upper_bound": integer.objective,
            "merged_zone_count": len(merged),
            "selected_zone_count": len(integer.selected_zone_ids),
            "diagnostics": dict(integer.diagnostics),
        }
        result["metrics"] = {
            "analytic_load_lower_bound": (
                peak.peak_policy.minimum_feasible_utilization
            ),
            "epsilon_cap": peak.peak_policy.epsilon_cap,
            "peak_feasibility_witness_utilization": peak.diagnostics[
                "witness_peak_utilization"
            ],
            "root_lp_lower_bound": root.objective,
            "compact_primal_upper_bound": coverage.objective,
            "final_integer_upper_bound": integer.objective,
            "final_absolute_root_gap": absolute_gap,
            "final_scale_normalized_root_gap": scale_normalized_gap,
            "final_incumbent_relative_root_gap": incumbent_relative_gap,
            # Backward-compatible alias.  This is not the conventional MIP
            # relative gap when the normalized objective lies below one.
            "final_relative_root_gap": scale_normalized_gap,
            "root_zone_count": len(root.zones),
            "compact_primal_candidate_zone_count": len(
                coverage.candidate_zones
            ),
            "merged_zone_count": len(merged),
            "weighted_objective_categories": dict(
                integer.certificate["weighted_objective_categories"]
            ),
            "raw_objective_components": dict(integer.certificate["raw"]),
            "independent_validation_passed": bool(
                integer.certificate["validation"]["passed"]
            ),
        }
    except Exception as error:
        return fail("final_rim", error, stage_started)

    result["status"] = "completed"
    result["passed"] = True
    result["total_wall_seconds"] = perf_counter() - case_started
    checkpoint(result)
    progress(
        f"[{spec.case_id}] completed: LB={root.objective:.6g}, "
        f"UB={integer.objective:.6g}, absolute gap={absolute_gap:.6g}"
    )
    return result


def _flat_case(result: Mapping[str, Any]) -> dict[str, Any]:
    problem = result.get("problem", {})
    metrics = result.get("metrics", {})
    stages = result.get("stages", {})
    row: dict[str, Any] = {
        "case_id": result.get("case_id"),
        "seed": result.get("seed"),
        "status": result.get("status"),
        "passed": result.get("passed"),
        "failure_stage": result.get("failure_stage"),
        "export_group_count": problem.get("export_group_count"),
        "export_demand_boxes": problem.get("export_demand_boxes"),
        "anonymous_import_boxes": problem.get("anonymous_import_boxes"),
        "row_atom_count": problem.get("row_atom_count"),
        "analytic_load_lower_bound": metrics.get(
            "analytic_load_lower_bound"
        ),
        "root_lp_lower_bound": metrics.get("root_lp_lower_bound"),
        "final_integer_upper_bound": metrics.get("final_integer_upper_bound"),
        "final_relative_root_gap": metrics.get("final_relative_root_gap"),
        "final_incumbent_relative_root_gap": metrics.get(
            "final_incumbent_relative_root_gap"
        ),
        "root_zone_count": metrics.get("root_zone_count"),
        "merged_zone_count": metrics.get("merged_zone_count"),
        "total_wall_seconds": result.get("total_wall_seconds"),
    }
    for stage_name in (
        "input",
        "peak_preparation",
        "root_cg",
        "compact_primal",
        "final_rim",
    ):
        stage = stages.get(stage_name, {})
        row[f"{stage_name}_state"] = stage.get("state")
        row[f"{stage_name}_seconds"] = stage.get("wall_seconds")
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_v6_suite(
    suite_path: str | Path,
    output_root: str | Path,
    *,
    case_ids: Sequence[str] | None = None,
    peak_feasibility_time_limit: float = 10.0,
    pricing_time_limit: float = 30.0,
    root_total_time_limit: float = 60.0,
    maximum_phase_one_iterations: int = 200,
    maximum_business_iterations: int = 500,
    phase_one_columns_per_group: int = 512,
    business_columns_per_group: int = 512,
    maximum_patterns_per_interval: int = 8,
    coverage_time_limit: float = 60.0,
    coverage_mip_gap: float = 0.0,
    coverage_pool_solutions: int = 4,
    integer_time_limit: float = 60.0,
    integer_mip_gap: float = 0.0,
    root_only: bool = False,
    solver_threads: int = 1,
    solver_seed: int = 0,
    peak_utilization_headroom_fraction: float = 0.50,
    verbose_solver: bool = False,
    progress: Progress = lambda _message: None,
) -> dict[str, Any]:
    """Run selected deterministic cases and persist a checkpoint per phase."""

    suite_path = Path(suite_path).resolve()
    output_root = Path(output_root).resolve()
    base_path, all_specs = load_suite(suite_path)
    selected_ids = set(case_ids or ())
    specs = [
        spec
        for spec in all_specs
        if not selected_ids or spec.case_id in selected_ids
    ]
    missing = selected_ids - {spec.case_id for spec in specs}
    if missing:
        raise ValueError(f"unknown V6 scalability case IDs: {sorted(missing)}")
    if not specs:
        raise ValueError("no V6 scalability cases selected")

    objective = V6ObjectiveConfig(
        peak_utilization_headroom_fraction=(
            peak_utilization_headroom_fraction
        )
    )
    peak_config = V6AnalyticPeakConfig(
        feasibility_time_limit=peak_feasibility_time_limit,
        solver_threads=solver_threads,
        solver_seed=solver_seed,
        verbose=verbose_solver,
        objective=objective,
    )
    root_config = V6RootCgConfig(
        maximum_phase_one_iterations=maximum_phase_one_iterations,
        maximum_business_iterations=maximum_business_iterations,
        phase_one_columns_per_group=phase_one_columns_per_group,
        business_columns_per_group=business_columns_per_group,
        maximum_patterns_per_interval=maximum_patterns_per_interval,
        pricing_time_limit=pricing_time_limit,
        total_time_limit=root_total_time_limit,
        solver_threads=solver_threads,
        solver_seed=solver_seed,
        verbose=verbose_solver,
        objective=objective,
    )
    coverage_config = V6PrimalCoverageConfig(
        time_limit=coverage_time_limit,
        mip_gap=coverage_mip_gap,
        solver_threads=solver_threads,
        solver_seed=solver_seed,
        verbose=verbose_solver,
        maximum_pool_solutions=coverage_pool_solutions,
        objective=objective,
    )
    integer_config = V6RestrictedIntegerConfig(
        time_limit=integer_time_limit,
        mip_gap=integer_mip_gap,
        solver_threads=solver_threads,
        solver_seed=solver_seed,
        verbose=verbose_solver,
        objective=objective,
    )
    for config in (
        peak_config,
        root_config,
        coverage_config,
        integer_config,
    ):
        config.validate()

    base = InputAdapterGd.load_from_json(str(base_path))
    cases: list[dict[str, Any]] = []
    suite: dict[str, Any] = {
        "report_schema_version": 1,
        "model_schema_version": V6_MODEL_SCHEMA_VERSION,
        "algorithm": "v6_analytic_peak_exact_root_compact_primal_rim",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_path": str(suite_path),
        "base_input": str(base_path),
        "python_version": platform.python_version(),
        "configuration": {
            "run_mode": "root_only" if root_only else "full_pipeline",
            "peak": asdict(peak_config),
            "root": asdict(root_config),
            "compact_primal": asdict(coverage_config),
            "final_rim": asdict(integer_config),
        },
        "cases": cases,
    }

    def persist_suite() -> None:
        suite["case_count"] = len(cases)
        suite["passed_count"] = sum(bool(case.get("passed")) for case in cases)
        suite["failed_count"] = sum(
            case.get("status") == "failed" for case in cases
        )
        _write_json(output_root / "suite_summary.json", suite)
        _write_csv(
            output_root / "suite_summary.csv",
            [_flat_case(case) for case in cases],
        )

    for index, spec in enumerate(specs, start=1):
        progress(f"[{index}/{len(specs)}] starting {spec.case_id}")
        case_path = output_root / spec.case_id / "result.json"
        case_holder: dict[str, Any] = {}

        def checkpoint(payload: Mapping[str, Any]) -> None:
            case_holder.clear()
            case_holder.update(payload)
            _write_json(case_path, case_holder)

        result = run_v6_case(
            base,
            spec,
            objective=objective,
            peak_config=peak_config,
            root_config=root_config,
            coverage_config=coverage_config,
            integer_config=integer_config,
            stop_after_root=root_only,
            progress=progress,
            checkpoint=checkpoint,
        )
        cases.append(result)
        persist_suite()

    suite["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    persist_suite()
    return suite


__all__ = [
    "merge_v6_zone_pools",
    "run_v6_case",
    "run_v6_suite",
]
