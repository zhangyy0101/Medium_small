from __future__ import annotations

import json
import os
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


def run_complete_mip_baselines(
    suite_path: str | Path,
    output_root: str | Path,
    full_results_root: str | Path,
    paired_inputs_root: str | Path,
    *,
    case_ids: Sequence[str] = (
        "pilot_m_201",
        "pilot_m_202",
        "pilot_l_301",
        "pilot_l_302",
    ),
    time_limit: float = 60.0,
    threads: int = 1,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the fully enumerated same-zone MIP under a common budget."""

    if time_limit <= 0:
        raise ValueError("The complete-MIP time limit must be positive.")
    base_path, all_specs = load_suite(suite_path)
    specs = _select_specs(all_specs, case_ids)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    full_results_root = Path(full_results_root).resolve()
    paired_inputs_root = Path(paired_inputs_root).resolve()
    full_summary = json.loads(
        (full_results_root / "suite_summary.json").read_text(encoding="utf-8")
    )
    recorded_schema = str(full_summary.get("model_schema_version", ""))
    if recorded_schema != MODEL_SCHEMA_VERSION:
        raise ValueError(
            "Full-method results use an incompatible model schema: "
            f"recorded={recorded_schema!r}, required={MODEL_SCHEMA_VERSION!r}."
        )
    recorded_limit = float(full_summary.get("paper_time_limit", 0.0))
    if abs(recorded_limit - float(time_limit)) > 1e-9:
        raise ValueError(
            f"Full method used {recorded_limit:g} seconds, not {time_limit:g}."
        )
    full_by_id = {
        str(case["case_id"]): case for case in full_summary.get("cases", [])
    }
    base = InputAdapterGd.load_from_json(str(base_path))

    cases: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        _notify(
            progress,
            f"[{index}/{len(specs)}] {spec.case_id}: 开始全枚举完整排区 MIP",
        )
        case_dir = output_root / spec.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            full_case = full_by_id.get(spec.case_id)
            if full_case is None:
                raise ValueError(f"Missing full-method result: {spec.case_id}")
            paper = full_case.get("paper_algorithm", {})
            if not full_case.get("passed") or not paper.get("executed"):
                raise ValueError(f"Full-method run is not valid: {spec.case_id}")
            problem, input_check = _load_paired_problem(
                base,
                spec,
                paired_inputs_root,
                full_results_root,
            )
            complete = _run_complete_mip(
                problem,
                spec,
                case_dir,
                time_limit=time_limit,
                threads=threads,
            )
            comparison = compare_complete_mip_result(paper, complete)
            case = {
                "case_id": spec.case_id,
                "seed": spec.seed,
                "passed": True,
                "input_check": input_check,
                "current_algorithm": _current_summary(paper),
                "complete_zone_mip": complete,
                "comparison": comparison,
            }
            _write_json(case_dir / "complete_mip_baseline.json", case)
            _notify(
                progress,
                f"[{index}/{len(specs)}] {spec.case_id}: 完成，"
                f"status={complete['status']}，winner={comparison['winner']}，"
                f"当前算法相对 UB 改善={_format_percent(comparison['current_relative_incumbent_improvement'])}",
            )
        except Exception as exc:
            case = {
                "case_id": spec.case_id,
                "seed": spec.seed,
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_json(case_dir / "complete_mip_baseline.json", case)
            _notify(progress, f"[{index}/{len(specs)}] {spec.case_id}: 失败：{exc}")
        cases.append(case)

    aggregate = aggregate_complete_mip_comparisons(cases)
    summary = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "suite_path": str(Path(suite_path).resolve()),
        "base_input": str(base_path),
        "full_results_root": str(full_results_root),
        "paired_inputs_root": str(paired_inputs_root),
        "design": {
            "objective_sense": "minimize",
            "same_contiguous_zone_model": True,
            "baseline": "fully_enumerated_complete_zone_mip",
            "current_algorithm": "exact_root_generation_restricted_mip_fix_optimize",
            "time_limit": float(time_limit),
            "threads": max(1, int(threads)),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "paired_seed": True,
            "paired_generation_manifests_verified": True,
            "large_plan_required": False,
            "final_fill_reserve_fraction": float(
                ContiguousZoneConfig().fill_time_fraction
            ),
            "exact_row_recourse_executed": True,
            "internal_and_external_output_validation": True,
        },
        "case_count": len(cases),
        "passed_count": sum(bool(case.get("passed")) for case in cases),
        "aggregate": aggregate,
        "cases": cases,
    }
    _write_json(output_root / "complete_mip_summary.json", summary)
    pd.DataFrame([_flat_case(case) for case in cases]).to_csv(
        output_root / "complete_mip_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_report(output_root / "complete_mip_report.md", summary)
    return summary


def compare_complete_mip_result(
    current: dict[str, Any],
    complete: dict[str, Any],
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    if not complete.get("has_solution") or complete.get("objective") is None:
        raise ValueError("The complete zone MIP did not produce an incumbent.")
    if complete.get("bound") is None:
        raise ValueError("The complete zone MIP did not produce a global bound.")
    current_ub = float(current["upper_bound"])
    complete_ub = float(complete["objective"])
    complete_bound = float(complete["bound"])
    if complete_bound > current_ub + tolerance:
        raise AssertionError(
            "Complete-MIP global bound exceeds the validated current incumbent: "
            f"bound={complete_bound}, current_ub={current_ub}"
        )
    improvement = complete_ub - current_ub
    if improvement > tolerance:
        winner = "current_algorithm"
    elif improvement < -tolerance:
        winner = "complete_zone_mip"
    else:
        winner = "tie"
    return {
        "winner": winner,
        "current_absolute_incumbent_improvement": improvement,
        "current_relative_incumbent_improvement": improvement
        / max(abs(complete_ub), 1e-12),
        "current_gap_using_complete_mip_bound": max(
            0.0,
            current_ub - complete_bound,
        )
        / max(abs(current_ub), 1e-12),
        "current_reported_root_lp_gap": float(current["relative_gap"]),
        "complete_mip_gap": float(complete["relative_gap"]),
        "current_minus_complete_seconds": float(current["total_seconds"])
        - float(complete["total_seconds"]),
    }


def aggregate_complete_mip_comparisons(
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    comparisons = [case["comparison"] for case in cases if case.get("passed")]
    improvements = [
        float(item["current_relative_incumbent_improvement"])
        for item in comparisons
    ]
    return {
        "comparable_case_count": len(comparisons),
        "current_algorithm_win_count": sum(
            item["winner"] == "current_algorithm" for item in comparisons
        ),
        "complete_zone_mip_win_count": sum(
            item["winner"] == "complete_zone_mip" for item in comparisons
        ),
        "tie_count": sum(item["winner"] == "tie" for item in comparisons),
        "mean_current_relative_incumbent_improvement": (
            mean(improvements) if improvements else None
        ),
        "all_complete_mip_outputs_valid": all(
            case.get("complete_zone_mip", {})
            .get("internal_validation", {})
            .get("passed")
            and case.get("complete_zone_mip", {})
            .get("external_validation", {})
            .get("passed")
            for case in cases
            if case.get("passed")
        ),
    }


def refresh_complete_mip_reports(output_root: str | Path) -> dict[str, Any]:
    """Rebuild aggregate tables from already completed baseline runs."""

    output_root = Path(output_root).resolve()
    summary_path = output_root / "complete_mip_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("model_schema_version", "")) != MODEL_SCHEMA_VERSION:
        raise ValueError(
            "Cannot refresh complete-MIP reports from an incompatible model schema."
        )
    cases = summary.get("cases", [])
    summary["aggregate"] = aggregate_complete_mip_comparisons(cases)
    summary.setdefault("design", {}).update(
        {
            "final_fill_reserve_fraction": float(
                ContiguousZoneConfig().fill_time_fraction
            ),
            "exact_row_recourse_executed": True,
            "internal_and_external_output_validation": True,
        }
    )
    _write_json(summary_path, summary)
    pd.DataFrame([_flat_case(case) for case in cases]).to_csv(
        output_root / "complete_mip_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_report(output_root / "complete_mip_report.md", summary)
    return summary


def _select_specs(
    all_specs: Sequence[ScenarioSpec],
    case_ids: Sequence[str],
) -> list[ScenarioSpec]:
    requested = set(case_ids)
    specs = [spec for spec in all_specs if spec.case_id in requested]
    unknown = requested - {spec.case_id for spec in all_specs}
    if unknown:
        raise ValueError(f"Unknown pilot case IDs: {sorted(unknown)}")
    if not specs:
        raise ValueError("No complete-MIP cases were selected.")
    return specs


def _load_paired_problem(
    base: InputAdapterGd,
    spec: ScenarioSpec,
    paired_inputs_root: Path,
    full_results_root: Path,
) -> tuple[Any, dict[str, Any]]:
    adapter, manifest = materialize_scenario(base, spec)
    paired_case_dir = paired_inputs_root / spec.case_id
    recorded_manifest = json.loads(
        (paired_case_dir / "generation_manifest.json").read_text(encoding="utf-8")
    )
    if manifest != recorded_manifest:
        raise AssertionError(f"Scenario manifest mismatch: {spec.case_id}")
    full_manifest = json.loads(
        (
            full_results_root
            / spec.case_id
            / "generation_manifest.json"
        ).read_text(encoding="utf-8")
    )
    if manifest != full_manifest:
        raise AssertionError(
            f"Full-method scenario manifest mismatch: {spec.case_id}"
        )
    target_voyages = sorted(
        str(value) for value in manifest["selected_export_voyages"]
    )
    inputs = load_planning_inputs(
        adapter,
        planning_time=pd.Timestamp(adapter.planning_time).to_pydatetime(),
        voyages=target_voyages,
    )
    return inputs.problem, {
        "manifest_matches": True,
        "full_method_manifest_matches": True,
        "large_plan_required": False,
        "group_count": len(inputs.problem.export_groups),
        "export_boxes": sum(group.demand for group in inputs.problem.export_groups),
        "import_boxes": sum(inputs.problem.import_demand_by_flow_size.values()),
    }


def _run_complete_mip(
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
    solution = ContiguousZoneGenerationPlanner(
        problem,
        common,
        ContiguousZoneConfig(),
    ).solve_complete_zone_mip()
    export_path = case_dir / "complete_mip_export_rows.csv"
    import_path = case_dir / "complete_mip_import_reservations.csv"
    pd.DataFrame(solution.export_rows).to_csv(
        export_path,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(solution.import_reservation_rows).to_csv(
        import_path,
        index=False,
        encoding="utf-8-sig",
    )
    external = validate_output_files(problem, export_path, import_path)
    diagnostics = solution.diagnostics
    internal = diagnostics.get("independent_solution_validation", {})
    if not internal.get("passed") or not external.get("passed"):
        raise AssertionError(
            "Complete-MIP output did not pass both independent validators."
        )
    zone_mip = diagnostics.get("zone_mip", {})
    certificate = diagnostics.get("zone_objective_certificate", {})
    objective = diagnostics.get("zone_model_upper_bound")
    if certificate.get("objective") is None or objective is None:
        raise AssertionError("Complete-MIP incumbent has no objective certificate.")
    if abs(float(certificate["objective"]) - float(objective)) > 1e-8:
        raise AssertionError("Complete-MIP objective certificate mismatch.")
    return {
        "algorithm": diagnostics.get("algorithm"),
        "status": zone_mip.get("status"),
        "has_solution": bool(zone_mip.get("has_solution")),
        "objective": objective,
        "bound": diagnostics.get("zone_model_global_lower_bound"),
        "absolute_gap": diagnostics.get("zone_model_absolute_gap"),
        "relative_gap": diagnostics.get("zone_model_relative_gap"),
        "zone_count": diagnostics.get("zone_preparation", {}).get("zone_count"),
        "selected_zone_count": zone_mip.get("selected_zone_count"),
        "selected_candidate_count": diagnostics.get(
            "zone_selected_candidate_count"
        ),
        "total_seconds": diagnostics.get("total_seconds"),
        "objective_certificate": certificate,
        "row_recourse": diagnostics.get("zone_fill"),
        "internal_validation": internal,
        "external_validation": external,
    }


def _current_summary(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": paper.get("algorithm"),
        "total_seconds": paper.get("total_seconds"),
        "upper_bound": paper.get("upper_bound"),
        "root_lp_lower_bound": paper.get("lower_bound"),
        "relative_gap": paper.get("relative_gap"),
        "valid": bool(
            paper.get("internal_validation", {}).get("passed")
            and paper.get("external_validation", {}).get("passed")
        ),
        "source": "reused_passed_paired_run",
    }


def _flat_case(case: dict[str, Any]) -> dict[str, Any]:
    inputs = case.get("input_check", {})
    current = case.get("current_algorithm", {})
    complete = case.get("complete_zone_mip", {})
    comparison = case.get("comparison", {})
    return {
        "case_id": case.get("case_id"),
        "seed": case.get("seed"),
        "passed": case.get("passed"),
        "error": case.get("error", ""),
        "groups": inputs.get("group_count"),
        "export_boxes": inputs.get("export_boxes"),
        "import_boxes": inputs.get("import_boxes"),
        "zone_count": complete.get("zone_count"),
        "current_seconds": current.get("total_seconds"),
        "current_ub": current.get("upper_bound"),
        "current_root_lp_lb": current.get("root_lp_lower_bound"),
        "current_reported_gap": current.get("relative_gap"),
        "complete_mip_seconds": complete.get("total_seconds"),
        "complete_mip_status": complete.get("status"),
        "complete_mip_ub": complete.get("objective"),
        "complete_mip_lb": complete.get("bound"),
        "complete_mip_gap": complete.get("relative_gap"),
        "winner": comparison.get("winner"),
        "current_relative_incumbent_improvement": comparison.get(
            "current_relative_incumbent_improvement"
        ),
        "current_gap_using_complete_mip_bound": comparison.get(
            "current_gap_using_complete_mip_bound"
        ),
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    time_limit = float(summary.get("design", {}).get("time_limit", 0.0))
    lines = [
        "# Complete-zone MIP baseline",
        "",
        "当前算法与全枚举完整排区 MIP 使用相同连续排区模型、物化输入、seed、单线程和 "
        f"{time_limit:g} 秒总预算。目标为最小化；current improvement 为正表示当前算法得到更好的整数解。",
        "",
        "| Case | Groups | Zones | Current UB | Complete MIP UB | Current improvement | Complete MIP LB | Current gap using MIP LB | MIP gap | Winner |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in summary["cases"]:
        inputs = case.get("input_check", {})
        current = case.get("current_algorithm", {})
        complete = case.get("complete_zone_mip", {})
        comparison = case.get("comparison", {})
        lines.append(
            "| {case_id} | {groups} | {zones} | {current_ub} | {mip_ub} | {improvement} | {mip_lb} | {current_gap} | {mip_gap} | {winner} |".format(
                case_id=case.get("case_id", ""),
                groups=inputs.get("group_count", ""),
                zones=complete.get("zone_count", ""),
                current_ub=_format_number(current.get("upper_bound"), 6),
                mip_ub=_format_number(complete.get("objective"), 6),
                improvement=_format_percent(
                    comparison.get("current_relative_incumbent_improvement")
                ),
                mip_lb=_format_number(complete.get("bound"), 6),
                current_gap=_format_percent(
                    comparison.get("current_gap_using_complete_mip_bound")
                ),
                mip_gap=_format_percent(complete.get("relative_gap")),
                winner=comparison.get("winner", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Current algorithm wins: {aggregate['current_algorithm_win_count']}",
            f"- Complete-zone MIP wins: {aggregate['complete_zone_mip_win_count']}",
            f"- Ties: {aggregate['tie_count']}",
            "- Mean current incumbent improvement: "
            f"{_format_percent(aggregate['mean_current_relative_incumbent_improvement'])}",
            "",
            "## Pilot interpretation",
            "",
            _pilot_interpretation(summary),
            "",
            "完整 MIP 的 Gurobi bound 是同模型全局下界，因此可用于重新评估当前算法 incumbent 的认证 gap。该比较衡量求解策略，不涉及不同模型之间的目标值混用。",
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


def _pilot_interpretation(summary: dict[str, Any]) -> str:
    aggregate = summary["aggregate"]
    current_wins = int(aggregate["current_algorithm_win_count"])
    complete_wins = int(aggregate["complete_zone_mip_win_count"])
    passed = [case for case in summary.get("cases", []) if case.get("passed")]
    if passed:
        largest = max(
            passed,
            key=lambda case: int(
                case.get("input_check", {}).get("group_count", 0)
            ),
        )
        smaller = [case for case in passed if case is not largest]
        if (
            largest.get("comparison", {}).get("winner")
            == "current_algorithm"
            and smaller
            and all(
                case.get("comparison", {}).get("winner")
                == "complete_zone_mip"
                for case in smaller
            )
        ):
            groups = largest.get("input_check", {}).get("group_count")
            improvement = largest.get("comparison", {}).get(
                "current_relative_incumbent_improvement"
            )
            return (
                f"首种子筛查呈现规模转折：完整排区 MIP 在较小四档获胜，"
                f"当前算法在最大 {groups} 箱组算例获胜 "
                f"{_format_percent(improvement)}。每档目前只有一个种子，"
                "该转折必须用第二种子确认后才能形成稳健结论。"
            )
    if complete_wins > current_wins:
        return (
            "当前算法在这些中大型 pilot 上尚未体现出相对完整排区 MIP 的"
            "求解优势。应先改进整数列池与邻域搜索，或验证更大规模下的"
            "全枚举瓶颈，再提出算法优越性结论。"
        )
    return (
        "当前 pilot 支持继续检验算法的规模优势，但仍需更多种子和正式"
        "统计实验。"
    )
