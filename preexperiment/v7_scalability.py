"""Reproducible V7 scale runs with phase checkpoints and same-model MIP."""

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
from yard_planning.v7_atoms import build_v7_row_atoms
from yard_planning.v7_column_generation import (
    V7RootCgConfig,
    V7RootColumnGeneration,
    normalize_v7_pattern_ids,
    patterns_from_atom_solution,
)
from yard_planning.v7_complete_mip import (
    V7CompleteMipConfig,
    V7CompleteMipSolver,
)
from yard_planning.v7_integer import V7IntegerConfig, V7RestrictedIntegerSolver
from yard_planning.v7_model import (
    V7_MODEL_SCHEMA_VERSION,
    V7ObjectiveConfig,
    derive_v7_analytic_peak_policy,
)
from yard_planning.v7_pipeline import build_v7_stage1_guided_patterns
from yard_planning.v7_stage1_area import V7Stage1AreaSolver, V7Stage1Config

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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stage_error(error: Exception, started: float) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state": "failed",
        "wall_seconds": perf_counter() - started,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    diagnostics = getattr(error, "diagnostics", None)
    if diagnostics is not None:
        payload["diagnostics"] = dict(diagnostics)
    return payload


def _problem_summary(problem, atoms, manifest: Mapping[str, Any]) -> dict[str, Any]:
    groups_by_bay: dict[str, set[str]] = {}
    bays_by_group: dict[str, set[str]] = {}
    for atom in atoms:
        groups_by_bay.setdefault(atom.anchor_bay_key, set()).add(atom.group_id)
        bays_by_group.setdefault(atom.group_id, set()).add(atom.anchor_bay_key)
    group_bay_edges = sum(len(values) for values in bays_by_group.values())
    return {
        "export_group_count": len(problem.export_groups),
        "export_demand_boxes": sum(int(group.demand) for group in problem.export_groups),
        "anonymous_import_boxes": sum(
            int(quantity) for quantity in problem.import_demand_by_flow_size.values()
        ),
        "export_voyage_count": len(problem.target_voyages),
        "yard_bay_count": len(problem.bays),
        "yard_area_count": len({bay.area_no for bay in problem.bays.values()}),
        "row_atom_count": len(atoms),
        "candidate_anchor_bay_count": len(groups_by_bay),
        "group_bay_edge_count": group_bay_edges,
        "average_candidate_bays_per_group": (
            group_bay_edges / len(bays_by_group) if bays_by_group else 0.0
        ),
        "average_candidate_groups_per_bay": (
            sum(len(values) for values in groups_by_bay.values()) / len(groups_by_bay)
            if groups_by_bay
            else 0.0
        ),
        "maximum_candidate_groups_per_bay": max(
            (len(values) for values in groups_by_bay.values()), default=0
        ),
        "declared_export_rows": int(manifest["declared_export_rows"]),
        "declared_import_rows": int(manifest["declared_import_rows"]),
    }


def run_v7_case(
    base: InputAdapterGd,
    spec: ScenarioSpec,
    *,
    objective: V7ObjectiveConfig,
    peak_feasibility_time_limit: float,
    stage1_time_limit: float,
    stage1_pool_solutions: int,
    stage1_pool_gap: float,
    initial_candidate_area_cap: int,
    root_time_limit: float,
    root_maximum_iterations: int,
    columns_per_bay_per_round: int,
    integer_time_limit: float,
    integer_mip_gap: float,
    complete_mip_time_limit: float,
    complete_mip_gap: float,
    solver_threads: int,
    solver_seed: int,
    verbose_solver: bool,
    progress: Progress = lambda _message: None,
    checkpoint: Checkpoint = lambda _payload: None,
) -> dict[str, Any]:
    case_started = perf_counter()
    result: dict[str, Any] = {
        "case_id": spec.case_id,
        "seed": int(spec.seed),
        "generation_spec": asdict(spec),
        "model_schema_version": V7_MODEL_SCHEMA_VERSION,
        "algorithm": "v7_stage1_exact_bay_pricing_rim",
        "status": "running",
        "passed": False,
        "stages": {},
    }

    stage_started = perf_counter()
    progress(f"[{spec.case_id}] materializing V7 input and legal atoms")
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
        atoms, _anchor_limits = build_v7_row_atoms(problem)
        result["problem"] = _problem_summary(problem, atoms, manifest)
        result["stages"]["input"] = {
            "state": "completed",
            "wall_seconds": perf_counter() - stage_started,
        }
        checkpoint(result)
    except Exception as error:
        result["stages"]["input"] = _stage_error(error, stage_started)
        result["status"] = "failed"
        result["failure_stage"] = "input"
        result["total_wall_seconds"] = perf_counter() - case_started
        checkpoint(result)
        return result

    stage_started = perf_counter()
    progress(f"[{spec.case_id}] deriving and certifying the peak cap")
    try:
        peak_policy, peak_diagnostics = derive_v7_analytic_peak_policy(
            problem, objective, atoms=atoms
        )
        witness_solver = V7CompleteMipSolver(
            problem,
            V7CompleteMipConfig(
                time_limit=peak_feasibility_time_limit,
                mip_gap=1.0,
                solver_threads=solver_threads,
                solver_seed=solver_seed,
                verbose=verbose_solver,
                require_business_optimality=False,
                objective=objective,
            ),
        )
        witness = witness_solver.solve(peak_policy, feasibility_only=True)
        witness_patterns = patterns_from_atom_solution(
            problem, atoms, witness.selected_atom_indices
        )
        result["stages"]["peak_witness"] = {
            "state": "completed",
            "wall_seconds": perf_counter() - stage_started,
            "analytic_load_lower_bound": peak_policy.reference_utilization,
            "epsilon_cap": peak_policy.epsilon_cap,
            "witness_peak_utilization": witness.certificate["peak_utilization"]["maximum"],
            "witness_pattern_count": len(witness_patterns),
            "diagnostics": {
                "analytic": peak_diagnostics,
                "witness": dict(witness.diagnostics),
            },
        }
        checkpoint(result)
    except Exception as error:
        result["stages"]["peak_witness"] = _stage_error(error, stage_started)
        result["status"] = "failed"
        result["failure_stage"] = "peak_witness"
        result["total_wall_seconds"] = perf_counter() - case_started
        checkpoint(result)
        return result

    algorithm_completed = False
    stage1 = None
    root = None
    integer = None

    stage_started = perf_counter()
    progress(f"[{spec.case_id}] solving Stage 1 area model")
    try:
        stage1 = V7Stage1AreaSolver(
            problem,
            peak_policy,
            atoms,
            V7Stage1Config(
                time_limit=stage1_time_limit,
                maximum_pool_solutions=stage1_pool_solutions,
                pool_gap=stage1_pool_gap,
                initial_candidate_area_cap=initial_candidate_area_cap,
                solver_threads=solver_threads,
                solver_seed=solver_seed,
                verbose=verbose_solver,
                objective=objective,
            ),
        ).solve()
        guided_patterns = build_v7_stage1_guided_patterns(problem, atoms, stage1)
        initial_patterns = normalize_v7_pattern_ids(
            [*witness_patterns, *guided_patterns]
        )
        result["stages"]["stage1"] = {
            "state": "completed",
            "wall_seconds": perf_counter() - stage_started,
            "guided_pattern_count": len(guided_patterns),
            "initial_pattern_count": len(initial_patterns),
            "diagnostics": dict(stage1.diagnostics),
        }
        checkpoint(result)
    except Exception as error:
        result["stages"]["stage1"] = _stage_error(error, stage_started)
        result["algorithm_failure_stage"] = "stage1"
        checkpoint(result)

    if stage1 is not None:
        stage_started = perf_counter()
        progress(f"[{spec.case_id}] closing the V7 exact root")
        try:
            root = V7RootColumnGeneration(
                problem,
                peak_policy,
                stage1.active_areas_by_group,
                initial_patterns,
                V7RootCgConfig(
                    root_time_limit=root_time_limit,
                    maximum_iterations=root_maximum_iterations,
                    columns_per_bay_per_round=columns_per_bay_per_round,
                    solver_threads=solver_threads,
                    solver_seed=solver_seed,
                    verbose=verbose_solver,
                    objective=objective,
                ),
                atoms=atoms,
            ).solve()
            result["stages"]["root_cg"] = {
                "state": "completed",
                "wall_seconds": perf_counter() - stage_started,
                "lower_bound": root.objective,
                "pattern_count": len(root.patterns),
                "diagnostics": dict(root.diagnostics),
            }
            checkpoint(result)
        except Exception as error:
            result["stages"]["root_cg"] = _stage_error(error, stage_started)
            result["algorithm_failure_stage"] = "root_cg"
            checkpoint(result)

    if root is not None:
        stage_started = perf_counter()
        progress(f"[{spec.case_id}] solving the restricted integer master")
        try:
            integer_pool = normalize_v7_pattern_ids(
                [*root.patterns, *witness_patterns, *guided_patterns]
            )
            integer = V7RestrictedIntegerSolver(
                problem,
                peak_policy,
                integer_pool,
                V7IntegerConfig(
                    time_limit=integer_time_limit,
                    mip_gap=integer_mip_gap,
                    solver_threads=solver_threads,
                    solver_seed=solver_seed,
                    verbose=verbose_solver,
                    objective=objective,
                ),
                atoms=atoms,
            ).solve()
            result["stages"]["integer"] = {
                "state": "completed",
                "wall_seconds": perf_counter() - stage_started,
                "upper_bound": integer.objective,
                "diagnostics": dict(integer.diagnostics),
            }
            algorithm_completed = True
            checkpoint(result)
        except Exception as error:
            result["stages"]["integer"] = _stage_error(error, stage_started)
            result["algorithm_failure_stage"] = "integer"
            checkpoint(result)

    stage_started = perf_counter()
    progress(f"[{spec.case_id}] solving the same-model Complete Compact MIP")
    complete = None
    try:
        complete = V7CompleteMipSolver(
            problem,
            V7CompleteMipConfig(
                time_limit=complete_mip_time_limit,
                mip_gap=complete_mip_gap,
                solver_threads=solver_threads,
                solver_seed=solver_seed,
                verbose=verbose_solver,
                require_business_optimality=False,
                objective=objective,
            ),
        ).solve(peak_policy)
        result["stages"]["complete_mip"] = {
            "state": "completed",
            "wall_seconds": perf_counter() - stage_started,
            "upper_bound": complete.objective,
            "lower_bound": complete.diagnostics.get("solver_bound"),
            "gap": complete.diagnostics.get("certified_gap"),
            "diagnostics": dict(complete.diagnostics),
        }
    except Exception as error:
        result["stages"]["complete_mip"] = _stage_error(error, stage_started)
        result["baseline_failure_stage"] = "complete_mip"
        checkpoint(result)

    metrics: dict[str, Any] = {
        "analytic_load_lower_bound": peak_policy.reference_utilization,
        "epsilon_cap": peak_policy.epsilon_cap,
        "witness_peak_utilization": witness.certificate["peak_utilization"]["maximum"],
    }
    if algorithm_completed and root is not None and integer is not None:
        absolute_gap = max(0.0, integer.objective - root.objective)
        metrics.update(
            {
                "root_lp_lower_bound": root.objective,
                "v7_integer_upper_bound": integer.objective,
                "v7_absolute_root_gap": absolute_gap,
                "v7_relative_root_gap": absolute_gap
                / max(abs(integer.objective), 1e-12),
                "v7_root_pattern_count": len(root.patterns),
                "v7_independent_validation_passed": bool(
                    integer.certificate["validation"]["passed"]
                ),
                "v7_weighted_objective_categories": dict(
                    integer.certificate["weighted_objective_categories"]
                ),
                "v7_raw_objective_components": dict(integer.certificate["raw"]),
            }
        )
    if complete is not None:
        metrics.update(
            {
                "complete_mip_upper_bound": complete.objective,
                "complete_mip_lower_bound": complete.diagnostics.get("solver_bound"),
                "complete_mip_gap": complete.diagnostics.get("certified_gap"),
                "complete_mip_status": complete.diagnostics.get("status"),
                "complete_mip_independent_validation_passed": bool(
                    complete.certificate["validation"]["passed"]
                ),
            }
        )
    if algorithm_completed and complete is not None and integer is not None:
        metrics["v7_ub_improvement_vs_complete_mip_percent"] = 100.0 * (
            complete.objective - integer.objective
        ) / max(abs(complete.objective), 1e-12)

    result["metrics"] = metrics
    result["algorithm_completed"] = algorithm_completed
    result["complete_mip_completed"] = complete is not None
    result["passed"] = algorithm_completed and complete is not None
    result["status"] = "completed" if result["passed"] else "partial"
    result["total_wall_seconds"] = perf_counter() - case_started
    checkpoint(result)
    if result["passed"]:
        progress(
            f"[{spec.case_id}] completed: root={root.objective:.6g}, "
            f"V7 UB={integer.objective:.6g}, MIP UB={complete.objective:.6g}"
        )
    else:
        progress(
            f"[{spec.case_id}] partial: algorithm={algorithm_completed}, "
            f"complete_mip={complete is not None}"
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
        "algorithm_failure_stage": result.get("algorithm_failure_stage"),
        "baseline_failure_stage": result.get("baseline_failure_stage"),
        "export_group_count": problem.get("export_group_count"),
        "export_demand_boxes": problem.get("export_demand_boxes"),
        "anonymous_import_boxes": problem.get("anonymous_import_boxes"),
        "row_atom_count": problem.get("row_atom_count"),
        "group_bay_edge_count": problem.get("group_bay_edge_count"),
        "root_lp_lower_bound": metrics.get("root_lp_lower_bound"),
        "v7_integer_upper_bound": metrics.get("v7_integer_upper_bound"),
        "v7_relative_root_gap": metrics.get("v7_relative_root_gap"),
        "complete_mip_upper_bound": metrics.get("complete_mip_upper_bound"),
        "complete_mip_lower_bound": metrics.get("complete_mip_lower_bound"),
        "complete_mip_gap": metrics.get("complete_mip_gap"),
        "v7_ub_improvement_vs_complete_mip_percent": metrics.get(
            "v7_ub_improvement_vs_complete_mip_percent"
        ),
        "root_pattern_count": metrics.get("v7_root_pattern_count"),
        "total_wall_seconds": result.get("total_wall_seconds"),
    }
    for stage_name in (
        "input",
        "peak_witness",
        "stage1",
        "root_cg",
        "integer",
        "complete_mip",
    ):
        stage = stages.get(stage_name, {})
        row[f"{stage_name}_state"] = stage.get("state")
        row[f"{stage_name}_seconds"] = stage.get("wall_seconds")
    return row


def run_v7_suite(
    suite_path: str | Path,
    output_root: str | Path,
    *,
    case_ids: Sequence[str] | None = None,
    peak_feasibility_time_limit: float = 10.0,
    stage1_time_limit: float = 10.0,
    stage1_pool_solutions: int = 8,
    stage1_pool_gap: float = 0.10,
    initial_candidate_area_cap: int = 4,
    root_time_limit: float = 60.0,
    root_maximum_iterations: int = 100,
    columns_per_bay_per_round: int = 3,
    integer_time_limit: float = 40.0,
    integer_mip_gap: float = 0.0,
    complete_mip_time_limit: float = 120.0,
    complete_mip_gap: float = 0.0,
    solver_threads: int = 1,
    solver_seed: int = 0,
    peak_utilization_headroom_fraction: float = 0.50,
    verbose_solver: bool = False,
    progress: Progress = lambda _message: None,
) -> dict[str, Any]:
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
        raise ValueError(f"unknown V7 scalability case IDs: {sorted(missing)}")
    if not specs:
        raise ValueError("no V7 scalability cases selected")

    objective = V7ObjectiveConfig(
        peak_utilization_headroom_fraction=peak_utilization_headroom_fraction
    )
    objective.validate()
    base = InputAdapterGd.load_from_json(str(base_path))
    cases: list[dict[str, Any]] = []
    suite: dict[str, Any] = {
        "report_schema_version": 1,
        "model_schema_version": V7_MODEL_SCHEMA_VERSION,
        "algorithm": "v7_stage1_exact_bay_pricing_rim",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_path": str(suite_path),
        "base_input": str(base_path),
        "python_version": platform.python_version(),
        "configuration": {
            "objective": objective.as_dict(),
            "peak_feasibility_time_limit": peak_feasibility_time_limit,
            "stage1_time_limit": stage1_time_limit,
            "stage1_pool_solutions": stage1_pool_solutions,
            "stage1_pool_gap": stage1_pool_gap,
            "initial_candidate_area_cap": initial_candidate_area_cap,
            "root_time_limit": root_time_limit,
            "root_maximum_iterations": root_maximum_iterations,
            "columns_per_bay_per_round": columns_per_bay_per_round,
            "integer_time_limit": integer_time_limit,
            "integer_mip_gap": integer_mip_gap,
            "complete_mip_time_limit": complete_mip_time_limit,
            "complete_mip_gap": complete_mip_gap,
            "solver_threads": solver_threads,
            "solver_seed": solver_seed,
            "verbose_solver": verbose_solver,
            "pricing_method": "exact_support_cardinality_row_assignment_mip",
        },
        "cases": cases,
    }

    def persist_suite() -> None:
        suite["case_count"] = len(cases)
        suite["passed_count"] = sum(bool(case.get("passed")) for case in cases)
        suite["partial_count"] = sum(case.get("status") == "partial" for case in cases)
        _write_json(output_root / "suite_summary.json", suite)
        _write_csv(output_root / "suite_summary.csv", [_flat_case(case) for case in cases])

    for index, spec in enumerate(specs, start=1):
        progress(f"[{index}/{len(specs)}] starting {spec.case_id}")
        case_path = output_root / spec.case_id / "result.json"

        def checkpoint(payload: Mapping[str, Any]) -> None:
            _write_json(case_path, payload)

        case = run_v7_case(
            base,
            spec,
            objective=objective,
            peak_feasibility_time_limit=peak_feasibility_time_limit,
            stage1_time_limit=stage1_time_limit,
            stage1_pool_solutions=stage1_pool_solutions,
            stage1_pool_gap=stage1_pool_gap,
            initial_candidate_area_cap=initial_candidate_area_cap,
            root_time_limit=root_time_limit,
            root_maximum_iterations=root_maximum_iterations,
            columns_per_bay_per_round=columns_per_bay_per_round,
            integer_time_limit=integer_time_limit,
            integer_mip_gap=integer_mip_gap,
            complete_mip_time_limit=complete_mip_time_limit,
            complete_mip_gap=complete_mip_gap,
            solver_threads=solver_threads,
            solver_seed=solver_seed,
            verbose_solver=verbose_solver,
            progress=progress,
            checkpoint=checkpoint,
        )
        cases.append(case)
        persist_suite()

    suite["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    persist_suite()
    return suite


__all__ = ["run_v7_case", "run_v7_suite"]
