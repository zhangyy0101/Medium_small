"""Read-only Complete-MIP incumbent versus V7.1 Stage-1 coverage audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

from yard_planning.models import ProblemData
from yard_planning.v7_atoms import V7RowAtom
from yard_planning.v7_complete_mip import V7CompleteMipResult
from yard_planning.v7_model import V7ModelEvaluator, V7ObjectiveConfig
from yard_planning.v7_stage1_area import V7Stage1Result


def group_area_flow(
    problem: ProblemData,
    group_bay_flow: Mapping[tuple[str, str], int],
) -> dict[tuple[str, str], int]:
    """Aggregate positive group-bay flow into exact group-area support."""

    output: Counter[tuple[str, str]] = Counter()
    for (group_id, bay_key), quantity in group_bay_flow.items():
        if int(quantity) <= 0:
            continue
        output[(str(group_id), str(problem.bays[str(bay_key)].area_no))] += int(
            quantity
        )
    return dict(sorted(output.items()))


def select_stage1_domain_for_cap(
    group_diagnostics: Mapping[str, object],
    *,
    additional_candidate_area_cap: int,
    maximum_pool_candidate_areas: int,
) -> dict[str, object]:
    """Replay the production Stage-1 cap policy from its complete rankings."""

    cap = int(additional_candidate_area_cap)
    pool_maximum = int(maximum_pool_candidate_areas)
    if cap <= 0:
        raise ValueError("additional candidate-area cap must be positive")
    if not 0 <= pool_maximum <= cap:
        raise ValueError("maximum pool-candidate areas must lie within the cap")
    mandatory = {str(area) for area in group_diagnostics["mandatory_best_areas"]}
    ranked_pool = [str(area) for area in group_diagnostics["pool_candidate_areas"]]
    ranked_bay_local = [
        str(area) for area in group_diagnostics["bay_local_candidate_areas"]
    ]
    pool_slots = min(pool_maximum, len(ranked_pool), cap)
    chosen_pool = ranked_pool[:pool_slots]
    chosen_bay_local = [
        area for area in ranked_bay_local if area not in chosen_pool
    ][: cap - len(chosen_pool)]
    if len(chosen_pool) + len(chosen_bay_local) < cap:
        chosen_pool.extend(
            area
            for area in ranked_pool[len(chosen_pool) :]
            if area not in chosen_bay_local
        )
        chosen_pool = chosen_pool[: cap - len(chosen_bay_local)]
    final = mandatory | set(chosen_pool) | set(chosen_bay_local)
    return {
        "additional_candidate_area_cap": cap,
        "maximum_pool_candidate_areas": pool_maximum,
        "pool_slots": pool_slots,
        "mandatory_best_areas": sorted(mandatory),
        "selected_pool_alternatives": chosen_pool,
        "selected_bay_local_areas": chosen_bay_local,
        "final_restricted_areas": sorted(final),
    }


def audit_stage1_incumbent_coverage(
    problem: ProblemData,
    atoms: Sequence[V7RowAtom],
    stage1: V7Stage1Result,
    complete_mip: V7CompleteMipResult,
    *,
    witness: V7CompleteMipResult | None = None,
    caps: Iterable[int] = (2, 3),
    maximum_pool_candidate_areas: int = 1,
    objective: V7ObjectiveConfig | None = None,
) -> dict[str, object]:
    """Compare one Complete-MIP incumbent support with Stage-1 rankings."""

    normalized_caps = tuple(sorted({int(cap) for cap in caps}))
    if not normalized_caps or normalized_caps[0] <= 0:
        raise ValueError("coverage audit requires positive candidate-area caps")
    evaluator = V7ModelEvaluator(problem, atoms, objective or V7ObjectiveConfig())
    oracle_flow = group_area_flow(problem, complete_mip.group_bay_flow)
    witness_flow = (
        {} if witness is None else group_area_flow(problem, witness.group_bay_flow)
    )
    best_flow = {
        (str(group_id), str(area)): int(quantity)
        for (group_id, area), quantity in stage1.best_group_area_quantity.items()
        if int(quantity) > 0
    }
    group_diagnostics = {
        str(row["group_id"]): row
        for row in stage1.diagnostics["candidate_area_diagnostics"]["groups"]
    }
    selections: dict[int, dict[str, dict[str, object]]] = {
        cap: {
            group_id: select_stage1_domain_for_cap(
                row,
                additional_candidate_area_cap=cap,
                maximum_pool_candidate_areas=maximum_pool_candidate_areas,
            )
            for group_id, row in group_diagnostics.items()
        }
        for cap in normalized_caps
    }
    actual_cap = int(stage1.diagnostics["additional_candidate_area_cap"])
    if actual_cap in selections:
        for group_id, row in group_diagnostics.items():
            simulated = selections[actual_cap][group_id]["final_restricted_areas"]
            if list(simulated) != sorted(str(area) for area in row["final_restricted_areas"]):
                raise RuntimeError(
                    f"Stage-1 cap replay differs from production for {group_id}"
                )

    pool_frequency: Counter[tuple[str, str]] = Counter()
    pool_quantity_mass: Counter[tuple[str, str]] = Counter()
    pool_first_solution_rank: dict[tuple[str, str], int] = {}
    for rank, solution in enumerate(stage1.pool_solutions, start=1):
        for pair, quantity in solution.group_area_quantity.items():
            normalized = (str(pair[0]), str(pair[1]))
            if int(quantity) <= 0:
                continue
            pool_frequency[normalized] += 1
            pool_quantity_mass[normalized] += int(quantity)
            pool_first_solution_rank.setdefault(normalized, rank)

    anchors_by_pair: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for atom in atoms:
        anchors_by_pair[(atom.group_id, atom.area_no)].add(atom.anchor_bay_key)
    groups_by_id = {str(group.group_id): group for group in problem.export_groups}
    rows: list[dict[str, object]] = []
    for (group_id, area), quantity in sorted(
        oracle_flow.items(), key=lambda item: (item[0][0], -item[1], item[0][1])
    ):
        diagnostics = group_diagnostics[group_id]
        pool_ranking = [str(value) for value in diagnostics["pool_candidate_areas"]]
        bay_local_ranking = [
            str(value) for value in diagnostics["bay_local_candidate_areas"]
        ]
        anchors = sorted(anchors_by_pair[(group_id, area)])
        row: dict[str, object] = {
            "group_id": group_id,
            "area": area,
            "oracle_quantity": int(quantity),
            "group_demand": int(groups_by_id[group_id].demand),
            "oracle_group_quantity_fraction": int(quantity)
            / max(1, int(groups_by_id[group_id].demand)),
            "oracle_group_area_count": sum(
                1 for pair in oracle_flow if pair[0] == group_id
            ),
            "in_feasibility_witness_support": (group_id, area) in witness_flow,
            "feasibility_witness_quantity": int(
                witness_flow.get((group_id, area), 0)
            ),
            "in_stage1_best_support": (group_id, area) in best_flow,
            "stage1_best_quantity": int(best_flow.get((group_id, area), 0)),
            "stage1_pool_frequency": int(pool_frequency[(group_id, area)]),
            "stage1_pool_quantity_mass": int(
                pool_quantity_mass[(group_id, area)]
            ),
            "stage1_pool_first_solution_rank": pool_first_solution_rank.get(
                (group_id, area)
            ),
            "pool_candidate_rank": (
                pool_ranking.index(area) + 1 if area in pool_ranking else None
            ),
            "bay_local_candidate_rank": (
                bay_local_ranking.index(area) + 1
                if area in bay_local_ranking
                else None
            ),
            "reachable_capacity": int(
                diagnostics["reachable_capacity_by_area"][area]
            ),
            "capacity_slack_vs_group_demand": int(
                diagnostics["reachable_capacity_by_area"][area]
            )
            - int(groups_by_id[group_id].demand),
            "compatible_bay_count": int(
                diagnostics["compatible_bay_count_by_area"][area]
            ),
            "max_single_anchor_capacity": int(
                diagnostics["max_single_anchor_capacity_by_area"][area]
            ),
            "minimum_existing_proximity": min(
                (
                    evaluator.normalized_existing_proximity(group_id, anchor)
                    for anchor in anchors
                ),
                default=1.0,
            ),
            "minimum_berth_distance": min(
                (
                    evaluator.normalized_berth_distance(group_id, anchor)
                    for anchor in anchors
                ),
                default=1.0,
            ),
            "stage1_coarse_flow_cost": evaluator.area_coarse_cost(group_id, area),
        }
        for cap in normalized_caps:
            selection = selections[cap][group_id]
            selected_pool = set(selection["selected_pool_alternatives"])
            selected_bay_local = set(selection["selected_bay_local_areas"])
            mandatory = set(selection["mandatory_best_areas"])
            selected = area in set(selection["final_restricted_areas"])
            if area in mandatory:
                channel = "stage1_best"
                reason = "included_as_stage1_best"
            elif area in selected_pool:
                channel = "pool"
                reason = "included_by_pool_rank"
            elif area in selected_bay_local:
                channel = "bay_local"
                reason = "included_by_bay_local_rank"
            elif area in pool_ranking:
                channel = None
                reason = "pool_candidate_not_selected_under_cap"
            else:
                channel = None
                reason = "absent_from_stage1_pool_and_bay_local_not_selected"
            row[f"cap_{cap}_selected"] = selected
            row[f"cap_{cap}_selection_channel"] = channel
            row[f"cap_{cap}_coverage_reason"] = reason
            row[f"cap_{cap}_pool_slots"] = int(selection["pool_slots"])
        rows.append(row)

    total_quantity = sum(oracle_flow.values())
    group_ids = sorted({pair[0] for pair in oracle_flow})
    cap_summaries: dict[str, object] = {}
    for cap in normalized_caps:
        selected_pairs = {
            (row["group_id"], row["area"])
            for row in rows
            if bool(row[f"cap_{cap}_selected"])
        }
        covered_quantity = sum(oracle_flow[pair] for pair in selected_pairs)
        fully_covered_groups = sum(
            all(
                pair in selected_pairs
                for pair in oracle_flow
                if pair[0] == group_id
            )
            for group_id in group_ids
        )
        missing_rows = [row for row in rows if not row[f"cap_{cap}_selected"]]
        cap_summaries[str(cap)] = {
            "covered_oracle_pair_count": len(selected_pairs),
            "oracle_pair_count": len(oracle_flow),
            "pair_coverage_fraction": len(selected_pairs) / max(1, len(oracle_flow)),
            "covered_oracle_quantity": covered_quantity,
            "oracle_quantity": total_quantity,
            "quantity_coverage_fraction": covered_quantity / max(1, total_quantity),
            "fully_covered_group_count": fully_covered_groups,
            "group_count": len(group_ids),
            "missing_pair_count": len(missing_rows),
            "missing_quantity": sum(int(row["oracle_quantity"]) for row in missing_rows),
        }

    best_pairs = set(best_flow) & set(oracle_flow)
    witness_pairs = set(witness_flow) & set(oracle_flow)
    return {
        "audit": "v7_complete_mip_incumbent_vs_stage1_group_area_coverage",
        "model_schema_version": complete_mip.diagnostics["model_schema_version"],
        "complete_mip": {
            "objective": complete_mip.objective,
            "status": complete_mip.diagnostics["status"],
            "solver_bound": complete_mip.diagnostics["solver_bound"],
            "certified_gap": complete_mip.diagnostics["certified_gap"],
            "runtime_seconds": complete_mip.diagnostics["runtime_seconds"],
            "group_area_pair_count": len(oracle_flow),
            "export_quantity": total_quantity,
        },
        "stage1": {
            "objective": stage1.diagnostics["solver_objective"],
            "bound": stage1.diagnostics["solver_bound"],
            "gap": stage1.diagnostics["solver_gap"],
            "pool_solution_count": len(stage1.pool_solutions),
            "best_support_pair_coverage": len(best_pairs) / max(1, len(oracle_flow)),
            "witness_support_pair_coverage": len(witness_pairs)
            / max(1, len(oracle_flow)),
        },
        "caps": cap_summaries,
        "rows": rows,
    }


__all__ = [
    "audit_stage1_incumbent_coverage",
    "group_area_flow",
    "select_stage1_domain_for_cap",
]
