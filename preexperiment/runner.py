from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd
from adapters.planning_input import load_planning_inputs
from yard_planning.contiguous_zone_generation import ContiguousZoneConfig, ContiguousZoneGenerationPlanner
from yard_planning.output_validator import validate_output_files
from yard_planning.planner import ColumnGenerationConfig

from .scenario_generator import (
    MODEL_SCHEMA_VERSION,
    ScenarioSpec,
    load_suite,
    materialize_scenario,
)


def run_suite(
    suite_path: str | Path,
    output_root: str | Path,
    *,
    case_ids: Sequence[str] | None = None,
    paper_time_limit: float = 0.0,
    threads: int = 1,
    disable_unused_capacity_objective: bool = False,
    fix_optimize_policy: str = "conflict_multi_round",
    fix_optimize_max_rounds: int = 3,
    fix_optimize_round_zone_fractions: Sequence[float] = (
        0.12,
        0.22,
        0.35,
    ),
    initial_mip_dynamic_stopping_enabled: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    base_path, all_specs = load_suite(suite_path)
    requested = set(case_ids or [])
    specs = [spec for spec in all_specs if not requested or spec.case_id in requested]
    unknown = requested - {spec.case_id for spec in all_specs}
    if unknown:
        raise ValueError(f"Unknown pilot case IDs: {sorted(unknown)}")
    if not specs:
        raise ValueError("No pilot cases were selected.")

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base = InputAdapterGd.load_from_json(str(base_path))
    case_summaries: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        if progress is not None:
            progress(f"[{index}/{len(specs)}] starting {spec.case_id}")
        case_dir = output_root / spec.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            summary = validate_case(
                base,
                spec,
                case_dir,
                paper_time_limit=paper_time_limit,
                threads=threads,
                disable_unused_capacity_objective=(
                    disable_unused_capacity_objective
                ),
                fix_optimize_policy=fix_optimize_policy,
                fix_optimize_max_rounds=fix_optimize_max_rounds,
                fix_optimize_round_zone_fractions=(
                    fix_optimize_round_zone_fractions
                ),
                initial_mip_dynamic_stopping_enabled=(
                    initial_mip_dynamic_stopping_enabled
                ),
            )
        except Exception as exc:
            summary = {
                "case_id": spec.case_id,
                "seed": spec.seed,
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_json(case_dir / "validation.json", summary)
        case_summaries.append(summary)
        if progress is not None:
            outcome = "passed" if summary.get("passed") else "failed"
            progress(f"[{index}/{len(specs)}] {outcome} {spec.case_id}")

    suite_summary = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "suite_path": str(Path(suite_path).resolve()),
        "base_input": str(base_path),
        "case_count": len(case_summaries),
        "passed_count": sum(bool(item.get("passed")) for item in case_summaries),
        "paper_algorithm_executed": paper_time_limit > 0,
        "paper_time_limit": paper_time_limit,
        "threads": max(1, int(threads)),
        "objective_design": {
            "group_area_dispersion": "diagnostic_only",
            "unused_capacity_objective_enabled": bool(
                not disable_unused_capacity_objective
            ),
            "retained_weights_renormalized": bool(
                disable_unused_capacity_objective
            ),
        },
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "fix_optimize_experiment_config": {
            "policy": fix_optimize_policy,
            "max_rounds": int(fix_optimize_max_rounds),
            "round_zone_fractions": [
                float(value)
                for value in fix_optimize_round_zone_fractions
            ],
        },
        "initial_mip_experiment_config": {
            "dynamic_stopping_enabled": bool(
                initial_mip_dynamic_stopping_enabled
            )
        },
        "cases": case_summaries,
    }
    _write_json(output_root / "suite_summary.json", suite_summary)
    pd.DataFrame([_flat_case_summary(item) for item in case_summaries]).to_csv(
        output_root / "suite_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_difficulty_report(output_root / "difficulty_report.md", case_summaries)
    return suite_summary


def validate_case(
    base: InputAdapterGd,
    spec: ScenarioSpec,
    case_dir: str | Path,
    *,
    paper_time_limit: float,
    threads: int,
    disable_unused_capacity_objective: bool,
    fix_optimize_policy: str,
    fix_optimize_max_rounds: int,
    fix_optimize_round_zone_fractions: Sequence[float],
    initial_mip_dynamic_stopping_enabled: bool,
) -> dict[str, Any]:
    case_dir = Path(case_dir)
    adapter, generation_manifest = materialize_scenario(base, spec)
    _write_json(case_dir / "generation_manifest.json", generation_manifest)

    target_voyages = sorted(
        str(value)
        for value in generation_manifest["selected_export_voyages"]
    )
    if not target_voyages:
        raise ValueError("Generated scenario contains no classified export voyage.")
    planning_time = pd.Timestamp(adapter.planning_time)
    planning_inputs = load_planning_inputs(
        adapter,
        planning_time=planning_time.to_pydatetime(),
        voyages=target_voyages,
    )
    group_count = len(planning_inputs.problem.export_groups)
    export_boxes = sum(group.demand for group in planning_inputs.problem.export_groups)
    if group_count != generation_manifest["export_group_count"]:
        raise AssertionError(
            f"Paper group count mismatch: input={group_count}, manifest={generation_manifest['export_group_count']}"
        )
    if export_boxes != generation_manifest["declared_export_rows"]:
        raise AssertionError(
            f"Paper export demand mismatch: input={export_boxes}, manifest={generation_manifest['declared_export_rows']}"
        )

    summary: dict[str, Any] = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "case_id": spec.case_id,
        "seed": spec.seed,
        "passed": True,
        "shared_input_object": True,
        "generation": {
            "export_group_count": generation_manifest["export_group_count"],
            "declared_export_rows": generation_manifest["declared_export_rows"],
            "declared_import_rows": generation_manifest["declared_import_rows"],
            "demand_policy": generation_manifest["demand_policy"],
        },
        "integrated_input": {
            "large_plan_required": False,
            "target_voyages": target_voyages,
            "demand_policy": "declared_known_export_and_import_documents",
            "prediction_input_ignored": True,
        },
        "paper_input": {
            "prepared": True,
            "group_count": group_count,
            "export_boxes": export_boxes,
            "bay_count": len(planning_inputs.problem.bays),
            "anonymous_import_boxes": sum(
                planning_inputs.problem.import_demand_by_flow_size.values()
            ),
        },
        "paper_algorithm": {"executed": False},
    }

    if paper_time_limit > 0:
        common = ColumnGenerationConfig(
            total_time_limit=paper_time_limit,
            mip_gap=0.0,
            solver_seed=spec.seed,
            solver_threads=max(1, int(threads)),
            verbose=False,
        )
        result = ContiguousZoneGenerationPlanner(
            planning_inputs.problem,
            common,
            ContiguousZoneConfig(
                unused_capacity_objective_enabled=(
                    not disable_unused_capacity_objective
                ),
                fix_optimize_policy=fix_optimize_policy,
                fix_optimize_max_rounds=int(fix_optimize_max_rounds),
                fix_optimize_round_zone_fractions=tuple(
                    float(value)
                    for value in fix_optimize_round_zone_fractions
                ),
                initial_mip_dynamic_stopping_enabled=bool(
                    initial_mip_dynamic_stopping_enabled
                ),
            ),
        ).solve()
        export_path = case_dir / "paper_export_rows.csv"
        import_path = case_dir / "paper_import_reservations.csv"
        pd.DataFrame(result.export_rows).to_csv(export_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(result.import_reservation_rows).to_csv(import_path, index=False, encoding="utf-8-sig")
        external_validation = validate_output_files(
            planning_inputs.problem,
            export_path,
            import_path,
        )
        internal_validation = result.diagnostics.get("independent_solution_validation", {})
        if not internal_validation.get("passed") or not external_validation.get("passed"):
            raise AssertionError("Paper output did not pass both independent validators.")
        preparation = result.diagnostics.get("zone_preparation", {})
        root = result.diagnostics.get("zone_root", {})
        zone_mip = result.diagnostics.get("zone_mip", {})
        mip_start = result.diagnostics.get("mip_start_diagnostics", {})
        mip_progress = zone_mip.get("mip_progress", {})
        anytime = result.diagnostics.get("mip_anytime", {})
        fix_optimize = result.diagnostics.get("zone_fix_optimize", {})
        primal_pool = result.diagnostics.get("primal_pool_diagnostics", {})
        certificate = result.diagnostics.get("zone_objective_certificate", {})
        summary["paper_algorithm"] = {
            "executed": True,
            "algorithm": result.diagnostics.get("algorithm"),
            "algorithm_version": result.diagnostics.get("algorithm_version"),
            "total_seconds": result.diagnostics.get("total_seconds"),
            "upper_bound": result.diagnostics.get("zone_model_upper_bound"),
            "lower_bound": result.diagnostics.get("zone_model_global_lower_bound"),
            "relative_gap": result.diagnostics.get("zone_model_relative_gap"),
            "time_limit": paper_time_limit,
            "candidate_row_location_count": result.diagnostics.get("candidate_row_location_count"),
            "atomic_candidate_count": preparation.get("atomic_candidate_count"),
            "implicit_zone_count": preparation.get("zone_count"),
            "root_closed": root.get("closed"),
            "root_round_count": len(root.get("rounds", [])),
            "root_seconds": root.get("seconds"),
            "root_active_zone_count": root.get("active_zone_count"),
            "proof_primal_pool_separated": result.diagnostics.get(
                "proof_primal_pool_separated"
            ),
            "proof_pool_zone_count": result.diagnostics.get(
                "proof_pool_zone_count"
            ),
            "primal_pool_zone_count": result.diagnostics.get(
                "primal_pool_zone_count"
            ),
            "final_primal_pool_zone_count": result.diagnostics.get(
                "final_primal_pool_zone_count"
            ),
            "proof_to_primal_reduction_fraction": result.diagnostics.get(
                "proof_to_primal_reduction_fraction"
            ),
            "primal_pool_policy": primal_pool.get("policy"),
            "primal_pool_build_seconds": primal_pool.get("build_seconds"),
            "primal_pool_columns_by_origin": primal_pool.get(
                "primal_pool_columns_by_origin", {}
            ),
            "final_primal_master_columns_by_origin": primal_pool.get(
                "final_primal_master_columns_by_origin", {}
            ),
            "primal_pool_mandatory_start_zone_count": primal_pool.get(
                "mandatory_start_zone_count"
            ),
            "primal_pool_mandatory_overflow_count": primal_pool.get(
                "mandatory_overflow_count"
            ),
            "primal_pool_budget_overflow_group_count": primal_pool.get(
                "budget_overflow_group_count"
            ),
            "primal_pool_group_diagnostics": primal_pool.get(
                "per_group", {}
            ),
            "primal_pool_lp_warm_start": primal_pool.get(
                "lp_warm_start", {}
            ),
            "zone_mip_status": zone_mip.get("status"),
            "time_to_first_solution": anytime.get("time_to_first_solution"),
            "time_to_best_solution": anytime.get("time_to_best_solution"),
            "first_incumbent": anytime.get("first_incumbent"),
            "best_incumbent": anytime.get("best_incumbent"),
            "incumbent_trajectory": anytime.get("incumbent_trajectory", []),
            "initial_mip_seconds": zone_mip.get("seconds"),
            "initial_mip_nodes": mip_progress.get("node_count"),
            "initial_mip_solution_count": mip_progress.get("solution_count"),
            "initial_mip_termination_reason": zone_mip.get(
                "termination_reason"
            ),
            "initial_mip_stopping": zone_mip.get(
                "initial_mip_stopping", {}
            ),
            "generated_candidate_count": mip_start.get(
                "generated_candidate_count"
            ),
            "integer_search_policy": zone_mip.get(
                "integer_search_policy"
            ),
            "v5_multi_start_enabled": zone_mip.get(
                "v5_multi_start_enabled"
            ),
            "candidate_generation_executed": zone_mip.get(
                "candidate_generation_executed"
            ),
            "deduplicated_candidate_count": mip_start.get(
                "deduplicated_candidate_count"
            ),
            "candidate_generation_completed_count": mip_start.get(
                "candidate_generation_completed_count"
            ),
            "candidate_generation_interrupted_count": mip_start.get(
                "candidate_generation_interrupted_count"
            ),
            "candidate_duplicate_count": mip_start.get(
                "candidate_duplicate_count"
            ),
            "infeasible_candidate_count": mip_start.get(
                "infeasible_candidate_count"
            ),
            "budget_interrupted_candidate_count": mip_start.get(
                "budget_interrupted_candidate_count"
            ),
            "repaired_candidate_count": mip_start.get(
                "repaired_candidate_count"
            ),
            "feasible_repaired_count": mip_start.get(
                "feasible_repaired_count"
            ),
            "provided_mip_start_count": mip_start.get(
                "provided_mip_start_count"
            ),
            "candidate_generation_seconds": mip_start.get(
                "candidate_generation_seconds"
            ),
            "repair_time_limit_seconds": mip_start.get(
                "repair_time_limit_seconds"
            ),
            "repair_seconds": mip_start.get("repair_seconds"),
            "repair_attempt_count": mip_start.get("repair_attempt_count"),
            "repair_feasible_count": mip_start.get("repair_feasible_count"),
            "total_mip_start_seconds": mip_start.get(
                "total_mip_start_seconds"
            ),
            "start_preparation_budget_seconds": mip_start.get(
                "start_preparation_budget_seconds"
            ),
            "start_preparation_actual_seconds": mip_start.get(
                "start_preparation_actual_seconds"
            ),
            "start_preparation_budget_utilization": mip_start.get(
                "start_preparation_budget_utilization"
            ),
            "submitted_start_count": mip_start.get(
                "submitted_start_count"
            ),
            "start_budget_exhausted": mip_start.get(
                "start_budget_exhausted"
            ),
            "start_termination_reason": mip_start.get(
                "start_termination_reason"
            ),
            "best_repaired_start_objective": mip_start.get(
                "best_repaired_start_objective"
            ),
            "greedy_candidate_summaries": mip_start.get(
                "generated_candidate_summaries", []
            ),
            "deduplicated_candidate_summaries": mip_start.get(
                "candidate_summaries", []
            ),
            "repair_summaries": mip_start.get("repair_summaries", []),
            "mip_start_solver_submission": mip_start.get("solver_submission"),
            "fix_optimize_round_count": result.diagnostics.get(
                "fix_optimize_round_count"
            ),
            "fix_optimize_success_count": result.diagnostics.get(
                "fix_optimize_success_count"
            ),
            "fix_optimize_total_improvement": result.diagnostics.get(
                "fix_optimize_total_improvement"
            ),
            "fix_optimize_diagnostics": fix_optimize,
            "fix_optimize_policy": fix_optimize.get("policy"),
            "fix_optimize_stop_reason": fix_optimize.get("stop_reason"),
            "fix_optimize_attempted_seed_groups": fix_optimize.get(
                "attempted_seed_groups", []
            ),
            "fix_optimize_rounds": fix_optimize.get("rounds", []),
            "selected_zone_origin_counts": result.diagnostics.get(
                "zone_selected_origin_counts", {}
            ),
            "selected_candidate_count": result.diagnostics.get("zone_selected_candidate_count"),
            "candidate_reduction": result.diagnostics.get("zone_candidate_reduction"),
            "objective_components_raw": certificate.get("raw", {}),
            "objective_components_weighted": certificate.get(
                "weighted", {}
            ),
            "objective_weights": certificate.get("weights", {}),
            "objective_design": certificate.get(
                "objective_design",
                result.diagnostics.get("business_objective", {}).get(
                    "objective_design",
                    {},
                ),
            ),
            "peak_utilization_policy": _compact_peak_policy(
                result.diagnostics.get("peak_utilization_policy", {})
            ),
            "peak_utilization_result": certificate.get(
                "peak_utilization", {}
            ),
            "internal_validation": internal_validation,
            "external_validation": external_validation,
        }
        _enrich_paper_difficulty(summary["paper_algorithm"], paper_time_limit)

    _write_json(case_dir / "validation.json", summary)
    return summary


def _flat_case_summary(summary: dict[str, Any]) -> dict[str, Any]:
    generation = summary.get("generation", {})
    paper_input = summary.get("paper_input", {})
    paper = summary.get("paper_algorithm", {})
    return {
        "case_id": summary.get("case_id"),
        "seed": summary.get("seed"),
        "passed": summary.get("passed"),
        "error": summary.get("error", ""),
        "export_group_count": generation.get("export_group_count"),
        "declared_export_rows": generation.get("declared_export_rows"),
        "declared_import_rows": generation.get("declared_import_rows"),
        "paper_input_group_count": paper_input.get("group_count"),
        "paper_input_import_boxes": paper_input.get("anonymous_import_boxes"),
        "paper_executed": paper.get("executed"),
        "paper_seconds": paper.get("total_seconds"),
        "paper_gap": paper.get("relative_gap"),
        "time_pressure": paper.get("time_pressure"),
        "gap_band": paper.get("gap_band"),
        "candidate_rows": paper.get("candidate_row_location_count"),
        "implicit_zones": paper.get("implicit_zone_count"),
        "root_rounds": paper.get("root_round_count"),
        "root_seconds": paper.get("root_seconds"),
        "proof_pool_zones": paper.get("proof_pool_zone_count"),
        "primal_pool_zones": paper.get("primal_pool_zone_count"),
        "proof_to_primal_reduction": paper.get(
            "proof_to_primal_reduction_fraction"
        ),
        "zone_mip_status": paper.get("zone_mip_status"),
        "algorithm_version": paper.get("algorithm_version"),
        "time_to_first_solution": paper.get("time_to_first_solution"),
        "time_to_best_solution": paper.get("time_to_best_solution"),
        "first_incumbent": paper.get("first_incumbent"),
        "best_incumbent": paper.get("best_incumbent"),
        "provided_mip_start_count": paper.get("provided_mip_start_count"),
        "start_preparation_seconds": paper.get(
            "start_preparation_actual_seconds"
        ),
        "start_termination_reason": paper.get("start_termination_reason"),
        "feasible_repaired_count": paper.get("feasible_repaired_count"),
        "selected_candidates": paper.get("selected_candidate_count"),
        "candidate_reduction": paper.get("candidate_reduction"),
    }


def _compact_peak_policy(policy: dict[str, Any]) -> dict[str, Any]:
    components = policy.get("lower_bound_components", {})
    binding_scope = (
        max(components, key=lambda key: float(components[key]))
        if components
        else None
    )
    keys = (
        "type",
        "measure",
        "load_lower_bound",
        "headroom_fraction",
        "epsilon_cap",
        "export_slot_units",
        "anonymous_import_slot_units",
        "reachable_area_capacity",
        "integer_area_capacity_breakpoints_included",
        "terminal_approved_threshold_used",
    )
    compact = {key: policy.get(key) for key in keys if key in policy}
    compact["binding_lower_bound_scope"] = binding_scope
    return compact


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _gap_band(relative_gap: Any) -> str:
    if relative_gap is None:
        return "unknown"
    gap = float(relative_gap)
    if gap <= 0.03:
        return "low"
    if gap <= 0.10:
        return "moderate"
    return "high"


def _time_pressure(seconds: Any, time_limit: float) -> str:
    if seconds is None or time_limit <= 0:
        return "unknown"
    ratio = float(seconds) / float(time_limit)
    if ratio < 0.25:
        return "quick"
    if ratio < 0.90:
        return "intermediate"
    return "budget_limited"


def _enrich_paper_difficulty(paper: dict[str, Any], time_limit: float) -> None:
    paper["time_limit"] = float(time_limit)
    paper["time_utilization"] = (
        None
        if paper.get("total_seconds") is None or time_limit <= 0
        else float(paper["total_seconds"]) / float(time_limit)
    )
    paper["time_pressure"] = _time_pressure(paper.get("total_seconds"), time_limit)
    paper["gap_band"] = _gap_band(paper.get("relative_gap"))


def _write_difficulty_report(path: Path, cases: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Pilot difficulty report",
        "",
        "统一设置下的经验难度诊断。时间压力按用时占预算比例划分：quick < 25%、intermediate < 90%、budget_limited ≥ 90%；gap band 的 low/moderate/high 阈值为 gap ≤ 3%、3% < gap ≤ 10%、gap > 10%。",
        "",
        "| Case | Groups | Export boxes | Import boxes | Time (s) | LB | UB | Gap | Root rounds | Candidates | Time pressure | Gap band | Valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for case in cases:
        generation = case.get("generation", {})
        paper_input = case.get("paper_input", {})
        paper = case.get("paper_algorithm", {})
        gap = paper.get("relative_gap")
        lines.append(
            "| {case} | {groups} | {export} | {imports} | {seconds} | {lb} | {ub} | {gap} | {rounds} | {candidates} | {time_pressure} | {gap_band} | {valid} |".format(
                case=case.get("case_id", ""),
                groups=generation.get("export_group_count", ""),
                export=generation.get("declared_export_rows", ""),
                imports=paper_input.get("anonymous_import_boxes", ""),
                seconds=_format_number(paper.get("total_seconds"), 2),
                lb=_format_number(paper.get("lower_bound"), 6),
                ub=_format_number(paper.get("upper_bound"), 6),
                gap=_format_percent(gap),
                rounds=paper.get("root_round_count", ""),
                candidates=paper.get("candidate_row_location_count", ""),
                time_pressure=paper.get("time_pressure", "not_run"),
                gap_band=paper.get("gap_band", "not_run"),
                valid="yes" if case.get("passed") else "no",
            )
        )
    lines.extend(
        [
            "",
            "时间压力和 gap 区间仅用于 pilot 参数校准，不是论文中的正式分类或结论。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def refresh_suite_reports(output_root: str | Path, paper_time_limit: float) -> dict[str, Any]:
    """Rebuild diagnostic labels and tables from completed solver outputs."""

    output_root = Path(output_root).resolve()
    summary_path = output_root / "suite_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("model_schema_version", "")) != MODEL_SCHEMA_VERSION:
        raise ValueError(
            "Cannot refresh reports from an incompatible model schema."
        )
    summary["paper_time_limit"] = float(paper_time_limit)
    for case in summary.get("cases", []):
        paper = case.get("paper_algorithm", {})
        if paper.get("executed"):
            _enrich_paper_difficulty(paper, paper_time_limit)
        validation_path = output_root / str(case.get("case_id")) / "validation.json"
        if validation_path.exists():
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            validation_paper = validation.get("paper_algorithm", {})
            if validation_paper.get("executed"):
                _enrich_paper_difficulty(validation_paper, paper_time_limit)
            _write_json(validation_path, validation)
    _write_json(summary_path, summary)
    cases = summary.get("cases", [])
    pd.DataFrame([_flat_case_summary(item) for item in cases]).to_csv(
        output_root / "suite_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_difficulty_report(output_root / "difficulty_report.md", cases)
    return summary


def _format_number(value: Any, digits: int) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def _format_percent(value: Any) -> str:
    return "" if value is None else f"{100.0 * float(value):.2f}%"
