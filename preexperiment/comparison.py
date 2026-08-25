from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd
from adapters.planning_input import load_planning_inputs
from yard_planning.contiguous_zone_generation import (
    ContiguousZoneConfig,
    ContiguousZoneGenerationPlanner,
)
from yard_planning.output_validator import validate_output_files
from yard_planning.planner import ColumnGenerationConfig

from .scenario_generator import (
    MODEL_SCHEMA_VERSION,
    ScenarioSpec,
    load_suite,
    materialize_scenario,
)


ProgressCallback = Callable[[str], None]


def run_comparison_suite(
    suite_path: str | Path,
    output_root: str | Path,
    full_results_root: str | Path,
    *,
    case_ids: Sequence[str] | None = None,
    correctness_case_ids: Sequence[str] = ("pilot_s_101", "pilot_s_102"),
    method_time_limit: float = 60.0,
    correctness_time_limit: float = 120.0,
    threads: int = 1,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Compare the full method with disabled F&O, then check small exact models."""

    if method_time_limit <= 0 or correctness_time_limit <= 0:
        raise ValueError("Comparison time limits must be positive.")
    base_path, all_specs = load_suite(suite_path)
    specs = _select_specs(all_specs, case_ids)
    correctness_ids = set(correctness_case_ids)
    unknown_correctness = correctness_ids - {spec.case_id for spec in specs}
    if unknown_correctness:
        raise ValueError(
            "Correctness cases must be selected comparison cases: "
            f"{sorted(unknown_correctness)}"
        )

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    full_results_root = Path(full_results_root).resolve()
    full_by_id, full_metadata = _load_full_results(
        full_results_root,
        specs,
        method_time_limit,
    )
    base = InputAdapterGd.load_from_json(str(base_path))

    cases: list[dict[str, Any]] = []
    prepared: dict[str, tuple[Any, dict[str, Any]]] = {}
    for index, spec in enumerate(specs, start=1):
        _notify(
            progress,
            f"[{index}/{len(specs)}] {spec.case_id}: 开始关闭 F&O 的 "
            f"{method_time_limit:g} 秒对照",
        )
        case_dir = output_root / spec.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            problem, preparation = _prepare_problem(
                base,
                spec,
                case_dir,
                expected_manifest_path=(
                    full_results_root
                    / spec.case_id
                    / "generation_manifest.json"
                ),
            )
            prepared[spec.case_id] = (problem, preparation)
            no_fix = _run_no_fix_optimize(
                problem,
                spec,
                case_dir,
                time_limit=method_time_limit,
                threads=threads,
            )
            full = _copy_full_method_summary(full_by_id[spec.case_id])
            paired = compare_method_summaries(full, no_fix)
            case_summary = {
                "case_id": spec.case_id,
                "seed": spec.seed,
                "passed": True,
                "preparation": preparation,
                "full_method": full,
                "no_fix_optimize": no_fix,
                "paired_comparison": paired,
                "correctness": {"executed": False},
            }
            _write_json(case_dir / "comparison.json", case_summary)
            _notify(
                progress,
                f"[{index}/{len(specs)}] {spec.case_id}: 完成，"
                f"winner={paired['winner']}，F&O 相对改善={_format_percent(paired['relative_improvement_from_fix_optimize'])}",
            )
        except Exception as exc:
            case_summary = {
                "case_id": spec.case_id,
                "seed": spec.seed,
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "correctness": {"executed": False},
            }
            _write_json(case_dir / "comparison.json", case_summary)
            _notify(progress, f"[{index}/{len(specs)}] {spec.case_id}: 失败：{exc}")
        cases.append(case_summary)

    for spec in specs:
        if spec.case_id not in correctness_ids:
            continue
        case = next(item for item in cases if item["case_id"] == spec.case_id)
        if not case.get("passed"):
            continue
        _notify(
            progress,
            f"{spec.case_id}: 开始完整 LP/MIP 正确性对照（预算 {correctness_time_limit:g} 秒）",
        )
        try:
            problem, _preparation = prepared[spec.case_id]
            correctness = _run_correctness_comparison(
                problem,
                spec,
                time_limit=correctness_time_limit,
                threads=threads,
                full=case["full_method"],
                no_fix=case["no_fix_optimize"],
            )
            case["correctness"] = correctness
            if not correctness["passed"]:
                case["passed"] = False
            _write_json(output_root / spec.case_id / "correctness.json", correctness)
            _write_json(output_root / spec.case_id / "comparison.json", case)
            _notify(
                progress,
                f"{spec.case_id}: 正确性对照完成，passed={correctness['passed']}，"
                f"complete_mip={correctness['complete_mip_status']}",
            )
        except Exception as exc:
            case["passed"] = False
            case["correctness"] = {
                "executed": True,
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_json(output_root / spec.case_id / "comparison.json", case)
            _notify(progress, f"{spec.case_id}: 正确性对照失败：{exc}")

    aggregate = aggregate_comparisons(cases)
    summary = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "suite_path": str(Path(suite_path).resolve()),
        "base_input": str(base_path),
        "full_results_root": str(full_results_root),
        "full_results_metadata": full_metadata,
        "comparison_design": {
            "objective_sense": "minimize",
            "paired_input": True,
            "paired_generation_manifest_verified": True,
            "large_plan_required": False,
            "paired_seed": True,
            "paired_threads": int(threads),
            "paired_total_time_limit": float(method_time_limit),
            "full_policy": "objective_fix_optimize",
            "ablation_policy": "fix_optimize_disabled",
            "full_results_reused": True,
            "correctness_time_limit": float(correctness_time_limit),
            "correctness_cases": sorted(correctness_ids),
            "complete_lp_mip_model": "same_contiguous_zone_model",
        },
        "case_count": len(cases),
        "passed_count": sum(bool(case.get("passed")) for case in cases),
        "aggregate": aggregate,
        "cases": cases,
    }
    _write_json(output_root / "comparison_summary.json", summary)
    pd.DataFrame([_flat_case(case) for case in cases]).to_csv(
        output_root / "comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_report(output_root / "comparison_report.md", summary)
    return summary


def compare_method_summaries(
    full: dict[str, Any],
    no_fix: dict[str, Any],
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Return paired minimization metrics; positive improvement favors F&O."""

    full_ub = float(full["upper_bound"])
    no_fix_ub = float(no_fix["upper_bound"])
    improvement = no_fix_ub - full_ub
    if improvement > tolerance:
        winner = "full_fix_optimize"
    elif improvement < -tolerance:
        winner = "no_fix_optimize"
    else:
        winner = "tie"
    return {
        "winner": winner,
        "absolute_improvement_from_fix_optimize": improvement,
        "relative_improvement_from_fix_optimize": improvement
        / max(abs(no_fix_ub), 1e-12),
        "full_minus_no_fix_gap": float(full["relative_gap"])
        - float(no_fix["relative_gap"]),
        "full_minus_no_fix_seconds": float(full["total_seconds"])
        - float(no_fix["total_seconds"]),
        "root_lower_bound_difference": float(full["lower_bound"])
        - float(no_fix["lower_bound"]),
    }


def aggregate_comparisons(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paired = [
        case["paired_comparison"]
        for case in cases
        if case.get("paired_comparison") is not None
    ]
    improvements = [
        float(item["relative_improvement_from_fix_optimize"])
        for item in paired
    ]
    return {
        "comparable_case_count": len(paired),
        "full_fix_optimize_win_count": sum(
            item["winner"] == "full_fix_optimize" for item in paired
        ),
        "no_fix_optimize_win_count": sum(
            item["winner"] == "no_fix_optimize" for item in paired
        ),
        "tie_count": sum(item["winner"] == "tie" for item in paired),
        "mean_relative_improvement_from_fix_optimize": (
            mean(improvements) if improvements else None
        ),
        "all_method_outputs_valid": all(
            case.get("full_method", {}).get("valid")
            and case.get("no_fix_optimize", {}).get("valid")
            for case in cases
            if case.get("paired_comparison") is not None
        ),
        "correctness_case_count": sum(
            bool(case.get("correctness", {}).get("executed")) for case in cases
        ),
        "correctness_passed_count": sum(
            bool(case.get("correctness", {}).get("passed")) for case in cases
        ),
    }


def _select_specs(
    all_specs: Sequence[ScenarioSpec],
    case_ids: Sequence[str] | None,
) -> list[ScenarioSpec]:
    requested = set(case_ids or [])
    specs = [spec for spec in all_specs if not requested or spec.case_id in requested]
    unknown = requested - {spec.case_id for spec in all_specs}
    if unknown:
        raise ValueError(f"Unknown pilot case IDs: {sorted(unknown)}")
    if not specs:
        raise ValueError("No pilot cases were selected.")
    return specs


def _load_full_results(
    root: Path,
    specs: Sequence[ScenarioSpec],
    method_time_limit: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = root / "suite_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Full-method summary does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded_schema = str(payload.get("model_schema_version", ""))
    if recorded_schema != MODEL_SCHEMA_VERSION:
        raise ValueError(
            "Full-method results use an incompatible model schema: "
            f"recorded={recorded_schema!r}, required={MODEL_SCHEMA_VERSION!r}. "
            "Rerun the integrated paper model first."
        )
    recorded_limit = float(payload.get("paper_time_limit", 0.0))
    if not math.isclose(recorded_limit, method_time_limit, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Full-method time limit is {recorded_limit}, requested {method_time_limit}."
        )
    by_id = {str(case["case_id"]): case for case in payload.get("cases", [])}
    for spec in specs:
        case = by_id.get(spec.case_id)
        if case is None:
            raise ValueError(f"Missing full-method result for {spec.case_id}.")
        paper = case.get("paper_algorithm", {})
        if int(case.get("seed", -1)) != spec.seed:
            raise ValueError(f"Full-method seed mismatch for {spec.case_id}.")
        if not case.get("passed") or not paper.get("executed"):
            raise ValueError(f"Full-method result is not a passed run: {spec.case_id}.")
        if not paper.get("internal_validation", {}).get("passed"):
            raise ValueError(f"Full-method internal validation failed: {spec.case_id}.")
        if not paper.get("external_validation", {}).get("passed"):
            raise ValueError(f"Full-method external validation failed: {spec.case_id}.")
    metadata = {
        "summary_path": str(path),
        "recorded_time_limit": recorded_limit,
        "model_schema_version": recorded_schema,
        "case_count": int(payload.get("case_count", 0)),
        "passed_count": int(payload.get("passed_count", 0)),
    }
    return by_id, metadata


def _prepare_problem(
    base: InputAdapterGd,
    spec: ScenarioSpec,
    case_dir: Path,
    *,
    expected_manifest_path: Path,
) -> tuple[Any, dict[str, Any]]:
    adapter, manifest = materialize_scenario(base, spec)
    _write_json(case_dir / "generation_manifest.json", manifest)
    if not expected_manifest_path.exists():
        raise FileNotFoundError(
            "Full-method generation manifest does not exist: "
            f"{expected_manifest_path}"
        )
    expected_manifest = json.loads(
        expected_manifest_path.read_text(encoding="utf-8")
    )
    if manifest != expected_manifest:
        raise AssertionError(
            "Materialized scenario differs from the full-method run: "
            f"{spec.case_id}"
        )
    target_voyages = sorted(
        str(value) for value in manifest["selected_export_voyages"]
    )
    inputs = load_planning_inputs(
        adapter,
        planning_time=pd.Timestamp(adapter.planning_time).to_pydatetime(),
        voyages=target_voyages,
    )
    preparation = {
        "shared_materialized_input": True,
        "matches_full_method_generation_manifest": True,
        "large_plan_required": False,
        "group_count": len(inputs.problem.export_groups),
        "export_boxes": sum(group.demand for group in inputs.problem.export_groups),
        "import_boxes": sum(inputs.problem.import_demand_by_flow_size.values()),
    }
    return inputs.problem, preparation


def _run_no_fix_optimize(
    problem: Any,
    spec: ScenarioSpec,
    case_dir: Path,
    *,
    time_limit: float,
    threads: int,
) -> dict[str, Any]:
    common = ColumnGenerationConfig(
        total_time_limit=time_limit,
        mip_gap=0.0,
        solver_seed=spec.seed,
        solver_threads=max(1, int(threads)),
        verbose=False,
    )
    result = ContiguousZoneGenerationPlanner(
        problem,
        common,
        ContiguousZoneConfig(fix_optimize_policy="disabled"),
    ).solve()
    export_path = case_dir / "no_fix_export_rows.csv"
    import_path = case_dir / "no_fix_import_reservations.csv"
    pd.DataFrame(result.export_rows).to_csv(export_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(result.import_reservation_rows).to_csv(
        import_path,
        index=False,
        encoding="utf-8-sig",
    )
    external = validate_output_files(problem, export_path, import_path)
    diagnostics = result.diagnostics
    internal = diagnostics.get("independent_solution_validation", {})
    if not internal.get("passed") or not external.get("passed"):
        raise AssertionError("No-F&O output did not pass both independent validators.")
    root = diagnostics.get("zone_root", {})
    initial = diagnostics.get("zone_mip_initial", {})
    fix = diagnostics.get("zone_fix_optimize", {})
    return {
        "policy": "fix_optimize_disabled",
        "total_seconds": diagnostics.get("total_seconds"),
        "upper_bound": diagnostics.get("zone_model_upper_bound"),
        "lower_bound": diagnostics.get("zone_model_global_lower_bound"),
        "relative_gap": diagnostics.get("zone_model_relative_gap"),
        "root_closed": root.get("closed"),
        "root_round_count": len(root.get("rounds", [])),
        "root_seconds": root.get("seconds"),
        "initial_mip_status": initial.get("status"),
        "initial_mip_seconds": initial.get("seconds"),
        "initial_mip_objective": initial.get("objective"),
        "initial_mip_bound": initial.get("bound"),
        "fix_optimize_status": fix.get("status"),
        "selected_candidate_count": diagnostics.get("zone_selected_candidate_count"),
        "valid": True,
        "internal_validation": internal,
        "external_validation": external,
    }


def _copy_full_method_summary(case: dict[str, Any]) -> dict[str, Any]:
    paper = case["paper_algorithm"]
    return {
        "policy": "objective_fix_optimize",
        "total_seconds": paper["total_seconds"],
        "upper_bound": paper["upper_bound"],
        "lower_bound": paper["lower_bound"],
        "relative_gap": paper["relative_gap"],
        "root_closed": paper.get("root_closed"),
        "root_round_count": paper.get("root_round_count"),
        "root_seconds": paper.get("root_seconds"),
        "zone_mip_status": paper.get("zone_mip_status"),
        "selected_candidate_count": paper.get("selected_candidate_count"),
        "valid": bool(
            paper.get("internal_validation", {}).get("passed")
            and paper.get("external_validation", {}).get("passed")
        ),
        "source": "reused_passed_pilot_run",
    }


def _run_correctness_comparison(
    problem: Any,
    spec: ScenarioSpec,
    *,
    time_limit: float,
    threads: int,
    full: dict[str, Any],
    no_fix: dict[str, Any],
) -> dict[str, Any]:
    common = ColumnGenerationConfig(
        total_time_limit=time_limit,
        mip_gap=0.0,
        solver_seed=spec.seed,
        solver_threads=max(1, int(threads)),
        verbose=False,
    )
    result = ContiguousZoneGenerationPlanner(
        problem,
        common,
        ContiguousZoneConfig(),
    ).analyze_root(compare_complete_lp=True, compare_complete_mip=True)
    complete_lp = result.get("complete_zone_lp") or {}
    complete_mip = result.get("complete_zone_mip") or {}
    lp_difference = result.get("root_complete_lp_difference")
    lp_passed = (
        bool(result.get("closed"))
        and complete_lp.get("status") == "optimal"
        and lp_difference is not None
        and abs(float(lp_difference)) <= 1e-6
    )
    mip_status = complete_mip.get("status")
    mip_objective = complete_mip.get("objective")
    mip_optimal = mip_status == "optimal" and mip_objective is not None
    upper_bounds_dominate_optimum = (
        mip_optimal
        and float(full["upper_bound"]) + 1e-7 >= float(mip_objective)
        and float(no_fix["upper_bound"]) + 1e-7 >= float(mip_objective)
    )
    return {
        "executed": True,
        "passed": bool(lp_passed and mip_optimal and upper_bounds_dominate_optimum),
        "root_closed": result.get("closed"),
        "generated_root_objective": result.get("root_objective"),
        "complete_lp_status": complete_lp.get("status"),
        "complete_lp_objective": complete_lp.get("objective"),
        "root_complete_lp_difference": lp_difference,
        "root_matches_complete_lp": lp_passed,
        "complete_mip_status": mip_status,
        "complete_mip_objective": mip_objective,
        "complete_mip_bound": complete_mip.get("bound"),
        "complete_mip_relative_gap": complete_mip.get("relative_gap"),
        "complete_zone_count": complete_mip.get("zone_count"),
        "full_method_optimality_gap": (
            None
            if not mip_optimal
            else (float(full["upper_bound"]) - float(mip_objective))
            / max(abs(float(full["upper_bound"])), 1e-12)
        ),
        "no_fix_optimality_gap": (
            None
            if not mip_optimal
            else (float(no_fix["upper_bound"]) - float(mip_objective))
            / max(abs(float(no_fix["upper_bound"])), 1e-12)
        ),
        "heuristic_upper_bounds_are_feasible_against_optimum": (
            upper_bounds_dominate_optimum
        ),
        "total_seconds": result.get("total_seconds"),
    }


def _flat_case(case: dict[str, Any]) -> dict[str, Any]:
    prep = case.get("preparation", {})
    full = case.get("full_method", {})
    no_fix = case.get("no_fix_optimize", {})
    paired = case.get("paired_comparison", {})
    correctness = case.get("correctness", {})
    return {
        "case_id": case.get("case_id"),
        "seed": case.get("seed"),
        "passed": case.get("passed"),
        "error": case.get("error", ""),
        "groups": prep.get("group_count"),
        "export_boxes": prep.get("export_boxes"),
        "import_boxes": prep.get("import_boxes"),
        "full_seconds": full.get("total_seconds"),
        "full_ub": full.get("upper_bound"),
        "full_lb": full.get("lower_bound"),
        "full_gap": full.get("relative_gap"),
        "no_fix_seconds": no_fix.get("total_seconds"),
        "no_fix_ub": no_fix.get("upper_bound"),
        "no_fix_lb": no_fix.get("lower_bound"),
        "no_fix_gap": no_fix.get("relative_gap"),
        "winner": paired.get("winner"),
        "fix_optimize_relative_improvement": paired.get(
            "relative_improvement_from_fix_optimize"
        ),
        "root_lb_difference": paired.get("root_lower_bound_difference"),
        "correctness_passed": correctness.get("passed"),
        "complete_lp_status": correctness.get("complete_lp_status"),
        "root_complete_lp_difference": correctness.get(
            "root_complete_lp_difference"
        ),
        "complete_mip_status": correctness.get("complete_mip_status"),
        "complete_mip_objective": correctness.get("complete_mip_objective"),
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    lines = [
        "# Pilot algorithm comparison",
        "",
        "同一物化输入、同一 Gurobi seed、单线程和相同总时间预算下，对比完整算法与关闭 Fix-and-Optimize（F&O）的消融算法。集成模型不再读取大计划。目标为最小化，因此 UB 越小越好。",
        "",
        "| Case | Groups | Full UB | No-F&O UB | F&O improvement | Winner | Full gap | No-F&O gap | Valid |",
        "|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for case in summary["cases"]:
        prep = case.get("preparation", {})
        full = case.get("full_method", {})
        no_fix = case.get("no_fix_optimize", {})
        paired = case.get("paired_comparison", {})
        lines.append(
            "| {case_id} | {groups} | {full_ub} | {no_fix_ub} | {improvement} | {winner} | {full_gap} | {no_fix_gap} | {valid} |".format(
                case_id=case.get("case_id", ""),
                groups=prep.get("group_count", ""),
                full_ub=_format_number(full.get("upper_bound"), 6),
                no_fix_ub=_format_number(no_fix.get("upper_bound"), 6),
                improvement=_format_percent(
                    paired.get("relative_improvement_from_fix_optimize")
                ),
                winner=paired.get("winner", ""),
                full_gap=_format_percent(full.get("relative_gap")),
                no_fix_gap=_format_percent(no_fix.get("relative_gap")),
                valid="yes" if case.get("passed") else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Same-model correctness checks",
            "",
            "小规模算例将精确定价得到的根节点 LP 与全枚举 LP 对比，并用全枚举 MIP 给出同模型整数基准。",
            "",
            "| Case | Root LP | Complete LP | Difference | Complete MIP | MIP objective | Passed |",
            "|---|---:|---:|---:|---|---:|---|",
        ]
    )
    for case in summary["cases"]:
        correctness = case.get("correctness", {})
        if not correctness.get("executed"):
            continue
        lines.append(
            "| {case_id} | {root} | {complete_lp} | {difference} | {mip_status} | {mip_objective} | {passed} |".format(
                case_id=case.get("case_id", ""),
                root=_format_number(correctness.get("generated_root_objective"), 9),
                complete_lp=_format_number(correctness.get("complete_lp_objective"), 9),
                difference=_format_number(
                    correctness.get("root_complete_lp_difference"), 12
                ),
                mip_status=correctness.get("complete_mip_status", ""),
                mip_objective=_format_number(
                    correctness.get("complete_mip_objective"), 9
                ),
                passed="yes" if correctness.get("passed") else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- F&O wins: {aggregate['full_fix_optimize_win_count']}",
            f"- No-F&O wins: {aggregate['no_fix_optimize_win_count']}",
            f"- Ties: {aggregate['tie_count']}",
            "- Mean relative F&O improvement: "
            f"{_format_percent(aggregate['mean_relative_improvement_from_fix_optimize'])}",
            f"- Correctness checks passed: {aggregate['correctness_passed_count']}/{aggregate['correctness_case_count']}",
            "",
            "## Pilot interpretation",
            "",
            _pilot_interpretation(aggregate),
            "",
            "这些结果属于 pilot 证据，只用于判断算法是否值得进入正式实验；不能替代扩展数据集、外部基线和统计检验。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _notify(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _format_number(value: Any, digits: int) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def _format_percent(value: Any) -> str:
    return "" if value is None else f"{100.0 * float(value):.2f}%"


def _pilot_interpretation(aggregate: dict[str, Any]) -> str:
    wins = int(aggregate["full_fix_optimize_win_count"])
    losses = int(aggregate["no_fix_optimize_win_count"])
    checks = int(aggregate["correctness_case_count"])
    checks_passed = int(aggregate["correctness_passed_count"])
    if wins > 0 and losses == 0 and checks > 0 and checks == checks_passed:
        return (
            "当前结果支持 F&O 是有效的算法组成部分，并支持继续开展正式实验；"
            "它尚不能证明完整算法优于外部文献算法，因为这里的对照是内部消融。"
        )
    return (
        "当前内部消融或正确性证据尚不充分，应先检查失败算例并扩充 pilot，"
        "再决定是否进入正式实验。"
    )
