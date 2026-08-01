from __future__ import annotations

import csv
import heapq
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Iterable

from block_bay_planning.models import EXPORT_VOYAGE_ROW_NO_MIX_ATTR, Bay, ProblemData, SmallBoxGroup


SIZE_ORDER = {"45": 0, "20": 1, "40": 2}
EXPORT_FLOWS = frozenset({"OF"})
MANDATORY_BAY_NO_MIX_ATTRS = ("IYC_CSZ_CSIZECD",)
SIZE_NO_MIX_ATTRS = frozenset({"IYC_CSZ_CSIZECD", "SIZE", "SIZE_MODE"})


class _ReverseSortKey:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __lt__(self, other: "_ReverseSortKey") -> bool:
        return other.value < self.value


def _area_flow(flow: object) -> str:
    text = "" if flow is None else str(flow).strip().upper()
    if text == "OF":
        return "OF"
    if text in {"IF", "IZ", "T"}:
        return text
    return "OZ"


@dataclass(frozen=True)
class PlacementColumn:
    column_id: str
    group_id: str
    demand_source: str
    voyage_id: str
    flow: str
    port: str
    size: str
    big_plan_size: str
    height: str
    weight_class: str
    special_stow_code: str
    attributes: dict[str, str]
    area_no: str
    bay_key: str
    bay_no: str
    block_id: str
    block_bays: tuple[str, ...]
    quantity: int
    stack_units: int
    row_allocation: tuple[tuple[str, str, int], ...]
    quota_key: tuple[str, str, str, str]
    coarse_key: tuple[str, ...]
    fine_key: tuple[str, ...]
    coarse_cluster_key: tuple[str, ...]
    fine_cluster_key: tuple[str, ...]
    intrinsic_cost: float


@dataclass
class ColumnGenerationConfig:
    max_iterations: int = 30
    columns_per_iteration: int = 2500
    stalled_pricing_columns: int = 500
    min_pricing_iterations: int = 3
    pricing_early_stop_new_columns: int = 0
    pricing_max_no_improve_iterations: int = 3
    pricing_min_lp_improvement: float = 10_000.0
    pricing_min_lp_improvement_relative: float = 2e-5
    pricing_min_lp_improvement_per_group: float = 5.0
    pricing_min_lp_improvement_per_1000_columns: float = 100.0
    feasibility_early_stop_enabled: bool = False
    feasibility_early_stop_min_iteration: int = 1
    primal_expansion_columns: int = 800
    max_primal_expansion_rounds: int = 3
    primal_expansion_reduced_cost_limit: float = 1_000_000.0
    stage0_closure_enabled: bool = True
    stage0_closure_max_extra_columns: int = 1500
    initial_columns_per_group: int = 8
    max_candidate_bays_per_group: int = 500
    adaptive_pricing_enabled: bool = True
    pricing_candidate_bays_initial: int = 160
    pricing_candidate_bays_growth_per_iteration: int = 40
    pricing_candidate_bays_high_dual: int = 260
    pricing_candidate_bays_unplaced: int = 500
    total_time_limit: float = 240.0
    pricing_min_lp_time_limit: float = 12.0
    pricing_iteration_setup_reserve: float = 8.0
    staged_repair_reserve_fraction: float = 0.33
    staged_repair_reserve_min: float = 60.0
    staged_repair_reserve_max: float = 90.0
    staged_repair_secondary_order_min_remaining: float = 20.0
    staged_repair_rebuild_first_unplaced_threshold: int = 500
    mip_time_limit: float = 120.0
    mip_gap: float = 0.01
    post_repair_area_relayout_enabled: bool = True
    post_repair_area_relayout_max_patterns: int = 1
    verbose: bool = True
    use_scip: bool = True
    scip_disable_symmetry: bool = True
    full_column_pool: bool = False
    demand_mode: str = "original"
    medium_plan_quota: dict[tuple[str, str, str, str, str], int] | None = None
    medium_plan_bay_quota: dict[tuple[str, str, str, str, str, str], int] | None = None
    repair_can_exceed_medium_plan_quota: bool = False
    unplaced_penalty: float = 100_000.0
    document_unplaced_penalty_multiplier: float = 10.0
    forecast_unplaced_penalty_multiplier: float = 1.0
    required_area_reward: float = 1_000.0
    existing_coarse_bay_reward: float = 48.0
    existing_coarse_neighbor_bay_reward: float = 24.0
    existing_other_coarse_bay_penalty: float = 48.0
    existing_other_coarse_neighbor_bay_penalty: float = 12.0
    existing_coarse_neighbor_max_bay_distance: int = 12
    twenty_isolated_bay_reward: float = 80.0
    twenty_large_segment_loss_penalty: float = 220.0
    twenty_large_segment_fresh_loss_penalty: float = 240.0
    twenty_large_segment_used_zero_loss_reward: float = 160.0
    twenty_max_consecutive_bays: int = 2
    twenty_consecutive_violation_penalty: float = 50_000.0
    group_area_balance_penalty: float = 36.0
    medium_concentrated_group_threshold: int = 26
    medium_small_group_area_split_penalty: float = 2400.0
    medium_small_group_fragment_penalty: float = 90.0
    medium_large_group_min_area_boxes: int = 10
    medium_large_group_small_area_penalty: float = 900.0
    medium_large_group_area_open_penalty: float = 0.0
    medium_large_group_target_area_boxes: int = 60
    medium_large_group_area_excess_penalty: float = 4.0
    big_plan_area_deviation_penalty: float = 3.0
    big_plan_fallback_tier_penalty: float = 20.0
    export_e_area_max_bays_per_voyage_area: int = 2
    export_e_area_non_40_penalty: float = 300.0
    small_plan_group_area_split_penalty: float = 80.0
    small_plan_group_block_split_penalty: float = 35.0
    small_plan_group_bay_split_penalty: float = 8.0
    small_plan_coarse_area_block_split_penalty: float = 24.0
    small_plan_coarse_area_bay_split_penalty: float = 2.5
    berth_distance_penalty: float = 0.02
    active_loading_area_penalty: float = 16.0
    post_window_loading_area_reward: float = 5.0
    fallback_bay_penalty: float = 4.0
    non_preferred_block_penalty: float = 6.0


@dataclass
class ColumnGenerationResult:
    medium_rows: list[dict]
    small_rows: list[dict]
    diagnostics: dict
    unplaced_rows: list[dict] = field(default_factory=list)
    columns: list[PlacementColumn] = field(default_factory=list)


class ColumnGenerationPlanner:
    """Small-plan-first column generation.

    A column is a feasible placement pattern for one document-container fine
    group in one bay with a fixed quantity. The restricted master covers all
    fine groups and respects bay exclusivity/capacity. Big-plan area/size
    quantities are soft targets, and fallback areas remain available with a
    tiered inheritance penalty.
    """

    def __init__(self, problem: ProblemData, config: ColumnGenerationConfig | None = None) -> None:
        self.problem = problem
        self.config = config or ColumnGenerationConfig()
        self._medium_plan_quota: Counter[tuple[str, ...]] = Counter(self.config.medium_plan_quota or {})
        self._medium_plan_bay_quota: Counter[tuple[str, ...]] = Counter(
            self.config.medium_plan_bay_quota or {}
        )
        self.group_source: dict[str, str] = {}
        self.demand_stats: dict[str, int | str] = {}
        self.export_voyages = self._infer_export_voyages(problem)
        self.import_voyages = self._infer_import_voyages(problem)
        self.groups = sorted(self._build_planning_groups(), key=self._group_sort_key)
        self.groups_by_id = {group.group_id: group for group in self.groups}
        self._expand_user_bay_adjust_rules()
        self.bays = problem.bays
        self.attribute_rules = getattr(problem, "attribute_rules", None)
        self.bays_by_area: dict[str, list[str]] = defaultdict(list)
        self.area_edge_bays: dict[str, set[str]] = defaultdict(set)
        self.block_members_by_area: dict[str, dict[str, tuple[str, ...]]] = {}
        self.block_by_bay: dict[tuple[str, str], str] = {}
        self.block_bay_nos: dict[str, tuple[str, ...]] = {}
        self.area_size_height_cap: Counter[tuple[str, str, str]] = Counter()
        self.area_group_cap: Counter[tuple[str, str]] = Counter()
        self._area_group_cap_computed: set[tuple[str, str]] = set()
        self.quota_by_key: Counter[tuple[str, str, str, str]] = Counter()
        self.existing_coarse_area_load: Counter[tuple[str, ...]] = Counter(
            {
                tuple(key): int(value)
                for key, value in getattr(problem, "existing_coarse_area_load", {}).items()
                if int(value) > 0
            }
        )
        self.existing_coarse_bay_load: Counter[tuple[str, ...]] = Counter(
            {
                tuple(key): int(value)
                for key, value in getattr(problem, "existing_coarse_bay_load", {}).items()
                if int(value) > 0
            }
        )
        self.existing_coarse_bays: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
        self.existing_coarse_area_bays: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
        self.existing_area_coarse_bays: defaultdict[tuple[str, tuple[str, ...]], set[str]] = defaultdict(set)
        self.existing_bay_coarse_load: defaultdict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
        self.large_segment_by_bay: dict[str, tuple[str, ...]] = {}
        self.large_segment_base_pairs: dict[tuple[str, ...], int] = {}
        self.large_segment_static_loss_by_bay: dict[str, int] = {}
        self.twenty_run_segments: list[tuple[str, ...]] = []
        self.twenty_run_windows: list[tuple[str, ...]] = []
        self.twenty_run_windows_by_bay: defaultdict[str, list[tuple[str, ...]]] = defaultdict(list)
        self.existing_twenty_bays: set[str] = set()
        for (*coarse_key, area_no, bay_key), value in self.existing_coarse_bay_load.items():
            if value > 0:
                coarse_tuple = tuple(coarse_key)
                self.existing_coarse_bays[coarse_tuple].add(str(bay_key))
                self.existing_coarse_area_bays[coarse_tuple + (str(area_no),)].add(str(bay_key))
                self.existing_area_coarse_bays[(str(area_no), coarse_tuple)].add(str(bay_key))
                self.existing_bay_coarse_load[str(bay_key)][coarse_tuple] += int(value)
        self.group_demand = {group.group_id: int(group.demand) for group in self.groups}
        self.coarse_demand: Counter[tuple[str, ...]] = Counter()
        self.coarse_groups: defaultdict[tuple[str, ...], list[SmallBoxGroup]] = defaultdict(list)
        self.coarse_cluster_demand: Counter[tuple[str, ...]] = Counter()
        self.coarse_cluster_groups: defaultdict[tuple[str, ...], list[SmallBoxGroup]] = defaultdict(list)
        self.voyage_flow_size_demand: Counter[tuple[str, str, str]] = Counter()
        self._columns: list[PlacementColumn] = []
        self._column_keys: set[tuple[str, str, int, tuple[tuple[str, str, int], ...]]] = set()
        self._default_column_triplets: set[tuple[str, str, int]] = set()
        self._column_indices_by_triplet: defaultdict[tuple[str, str, int], list[int]] = defaultdict(list)
        self._candidate_cache: dict[tuple[str, str], list[tuple[str, int, float]]] = {}
        self._candidate_scope = "stage0"
        self._master_seed_selected: Counter[int] = Counter()
        self._master_seed_unplaced: Counter[str] = Counter()
        self._master_start_selected: Counter[int] = Counter()
        self._master_start_unplaced: Counter[str] = Counter()
        self._primal_expansion_rounds = 0
        self._prepare_yard_indexes()
        self._prepare_quota()
        for group in self.groups:
            coarse_key = self._coarse_key(group)
            coarse_cluster_key = self._coarse_cluster_key(group)
            self.coarse_demand[coarse_key] += group.demand
            self.coarse_groups[coarse_key].append(group)
            self.coarse_cluster_demand[coarse_cluster_key] += group.demand
            self.coarse_cluster_groups[coarse_cluster_key].append(group)
            self.voyage_flow_size_demand[(group.voyage_id, group.status, self._big_plan_size(group.size))] += group.demand

    @property
    def columns(self) -> list[PlacementColumn]:
        return self._columns

    def _expand_user_bay_adjust_rules(self) -> None:
        rules = getattr(self.problem, "user_bay_adjust_rules", []) or []
        if not rules:
            return
        requirements = {
            str(group_id): set(values)
            for group_id, values in getattr(self.problem, "user_group_bay_requirements", {}).items()
        }
        blocklist = {
            str(group_id): set(values)
            for group_id, values in getattr(self.problem, "user_group_bay_blocklist", {}).items()
        }
        matched = 0
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            voyage_id = str(rule.get("voyage_id", ""))
            attributes = rule.get("attributes", {}) if isinstance(rule.get("attributes", {}), dict) else {}
            required_bays = set(rule.get("required_bays", set()) or set())
            blocked_bays = set(rule.get("blocked_bays", set()) or set())
            for group in self.groups:
                if voyage_id and str(group.voyage_id) != voyage_id:
                    continue
                if not self._group_matches_bay_adjust_attributes(group, attributes):
                    continue
                matched += 1
                if required_bays:
                    requirements.setdefault(group.group_id, set()).update(required_bays)
                if blocked_bays:
                    blocklist.setdefault(group.group_id, set()).update(blocked_bays)
        for group_id, blocked in blocklist.items():
            if blocked and group_id in requirements:
                requirements[group_id].difference_update(blocked)
        self.problem.user_group_bay_requirements = {group_id: values for group_id, values in requirements.items() if values}
        self.problem.user_group_bay_blocklist = {group_id: values for group_id, values in blocklist.items() if values}
        summary = dict(getattr(self.problem, "user_bay_constraint_summary", {}) or {})
        summary["expanded_matched_planning_groups"] = matched
        summary["expanded_required_group_count"] = len(self.problem.user_group_bay_requirements)
        summary["expanded_blocked_group_count"] = len(self.problem.user_group_bay_blocklist)
        self.problem.user_bay_constraint_summary = summary

    def _group_matches_bay_adjust_attributes(self, group: SmallBoxGroup, attributes: dict) -> bool:
        for attr, expected in attributes.items():
            if self._group_attr_value(group, str(attr)) != str(expected):
                return False
        return True

    def _build_planning_groups(self) -> list[SmallBoxGroup]:
        mode = (self.config.demand_mode or "original").strip().lower().replace("_", "-")
        if mode not in {"original", "medium", "medium-with-doc-floor", "doc-only"}:
            raise ValueError(
                "demand_mode must be one of: original, medium, medium-with-doc-floor, doc-only"
            )

        source_doc_groups = list(self.problem.small_groups)
        source_doc_boxes = sum(group.demand for group in source_doc_groups)
        if mode == "doc-only":
            self.group_source = {group.group_id: "document" for group in source_doc_groups}
            self.demand_stats = {
                "demand_mode": mode,
                "source_doc_group_count": len(source_doc_groups),
                "source_doc_box_count": source_doc_boxes,
                "medium_target_group_count": len(self.problem.groups),
                "medium_target_box_count": sum(group.demand for group in self.problem.groups),
                "original_small_output_box_count": source_doc_boxes,
                "original_medium_output_box_count": 0,
                "forecast_fallback_group_count": 0,
                "forecast_fallback_box_count": 0,
                "dropped_doc_box_count": 0,
                "doc_boxes_outside_medium_target": 0,
                "planned_box_count": source_doc_boxes,
            }
            return source_doc_groups

        remaining: Counter[tuple[str, ...]] = Counter()
        representative_by_coarse: dict[tuple[str, ...], SmallBoxGroup] = {}
        for group in self.problem.groups:
            small_group = self._medium_group_as_small_group(group, group.demand)
            coarse_key = self._coarse_key(small_group)
            remaining[coarse_key] += group.demand
            representative_by_coarse.setdefault(coarse_key, small_group)

        if mode == "original":
            return self._build_original_planning_groups(source_doc_groups, source_doc_boxes, remaining, representative_by_coarse)

        planning_groups: list[SmallBoxGroup] = []
        dropped_doc_boxes = 0
        height_weights = self._forecast_height_weights(source_doc_groups)
        for group in sorted(source_doc_groups, key=lambda g: (g.voyage_id, g.status, g.port, SIZE_ORDER.get(g.size, 3), g.group_id)):
            key = self._coarse_key(group)
            if mode == "medium":
                take = min(group.demand, remaining.get(key, 0))
                if take <= 0:
                    dropped_doc_boxes += group.demand
                    continue
                if take < group.demand:
                    dropped_doc_boxes += group.demand - take
                planning_group = self._copy_group_with_demand(group, take)
                planning_groups.append(planning_group)
                self.group_source[planning_group.group_id] = "document"
                remaining[key] -= take
            else:
                planning_groups.append(group)
                self.group_source[group.group_id] = "document"
                remaining[key] -= group.demand

        forecast_group_count = 0
        forecast_box_count = 0
        non_export_fallback_suppressed_boxes = 0
        for coarse_key, qty in sorted(remaining.items()):
            if qty <= 0:
                continue
            representative = representative_by_coarse.get(coarse_key)
            if representative is None:
                continue
            if not self._is_export_voyage(representative.voyage_id):
                non_export_fallback_suppressed_boxes += int(qty)
                continue
            for height, height_qty in self._split_forecast_by_height(
                representative.voyage_id,
                representative.status,
                representative.port,
                representative.size,
                int(qty),
                height_weights,
            ):
                forecast_group_count += 1
                attributes = self._fallback_group_attributes(
                    representative.voyage_id,
                    representative.status,
                    representative.port,
                    representative.size,
                    height,
                    representative.weight_class,
                )
                attributes.update(getattr(representative, "attributes", {}) or {})
                group = SmallBoxGroup(
                    group_id=f"{representative.voyage_id}_F{forecast_group_count:03d}",
                    voyage_id=representative.voyage_id,
                    status=representative.status,
                    port=representative.port,
                    size=representative.size,
                    height=height,
                    weight_class=representative.weight_class,
                    demand=int(height_qty),
                    pre_stow=False,
                    special_stow=False,
                    special_stow_code="",
                    attributes=attributes,
                    area_allowlist=self._copy_area_allowlist(representative),
                )
                planning_groups.append(group)
                self.group_source[group.group_id] = "forecast_fallback"
                forecast_box_count += int(height_qty)

        self.demand_stats = {
            "demand_mode": mode,
            "source_doc_group_count": len(source_doc_groups),
            "source_doc_box_count": source_doc_boxes,
            "medium_target_group_count": len(self.problem.groups),
            "medium_target_box_count": sum(group.demand for group in self.problem.groups),
            "original_small_output_box_count": (
                source_doc_boxes if mode == "medium-with-doc-floor" else source_doc_boxes - dropped_doc_boxes
            ),
            "original_medium_output_box_count": sum(group.demand for group in self.problem.groups),
            "forecast_fallback_group_count": forecast_group_count,
            "forecast_fallback_box_count": forecast_box_count,
            "non_export_fallback_suppressed_box_count": non_export_fallback_suppressed_boxes,
            "dropped_doc_box_count": dropped_doc_boxes,
            "doc_boxes_outside_medium_target": 0,
            "planned_box_count": sum(group.demand for group in planning_groups),
        }
        return planning_groups

    def _build_original_planning_groups(
        self,
        source_doc_groups: list[SmallBoxGroup],
        source_doc_boxes: int,
        remaining_medium: Counter[tuple[str, ...]],
        representative_by_coarse: dict[tuple[str, ...], SmallBoxGroup],
    ) -> list[SmallBoxGroup]:
        """Build the same demand scopes as the SA + heuristic pipeline.

        The original pipeline writes the medium plan from ``problem.groups`` and
        writes the small plan from all current document groups.  Document boxes
        consume only the exact medium coarse target with the same voyage, flow,
        port, and true size; the data loader lifts medium demand beforehand when
        the document floor is higher than the forecast medium target.
        """
        planning_groups = list(source_doc_groups)
        self.group_source = {group.group_id: "document" for group in source_doc_groups}
        doc_boxes_outside_medium_target = 0

        for group in sorted(source_doc_groups, key=lambda g: (g.voyage_id, g.status, g.port, SIZE_ORDER.get(g.size, 3), g.group_id)):
            doc_boxes_outside_medium_target += self._consume_medium_target_for_document_group(
                remaining_medium,
                group,
            )

        height_weights = self._forecast_height_weights(source_doc_groups)
        forecast_group_count = 0
        forecast_box_count = 0
        non_export_fallback_suppressed_boxes = 0
        for coarse_key, qty in sorted(remaining_medium.items()):
            if qty <= 0:
                continue
            representative = representative_by_coarse.get(coarse_key)
            if representative is None:
                continue
            if not self._is_export_voyage(representative.voyage_id):
                non_export_fallback_suppressed_boxes += int(qty)
                continue
            for height, height_qty in self._split_forecast_by_height(
                representative.voyage_id,
                representative.status,
                representative.port,
                representative.size,
                int(qty),
                height_weights,
            ):
                forecast_group_count += 1
                attributes = self._fallback_group_attributes(
                    representative.voyage_id,
                    representative.status,
                    representative.port,
                    representative.size,
                    height,
                    representative.weight_class,
                )
                attributes.update(getattr(representative, "attributes", {}) or {})
                group = SmallBoxGroup(
                    group_id=f"{representative.voyage_id}_F{forecast_group_count:03d}",
                    voyage_id=representative.voyage_id,
                    status=representative.status,
                    port=representative.port,
                    size=representative.size,
                    height=height,
                    weight_class=representative.weight_class,
                    demand=int(height_qty),
                    pre_stow=False,
                    special_stow=False,
                    special_stow_code="",
                    attributes=attributes,
                    area_allowlist=self._copy_area_allowlist(representative),
                )
                planning_groups.append(group)
                self.group_source[group.group_id] = "forecast_fallback"
                forecast_box_count += int(height_qty)

        self.demand_stats = {
            "demand_mode": "original",
            "source_doc_group_count": len(source_doc_groups),
            "source_doc_box_count": source_doc_boxes,
            "medium_target_group_count": len(self.problem.groups),
            "medium_target_box_count": sum(group.demand for group in self.problem.groups),
            "original_small_output_box_count": source_doc_boxes,
            "original_medium_output_box_count": sum(group.demand for group in self.problem.groups),
            "forecast_fallback_group_count": forecast_group_count,
            "forecast_fallback_box_count": forecast_box_count,
            "non_export_fallback_suppressed_box_count": non_export_fallback_suppressed_boxes,
            "dropped_doc_box_count": 0,
            "doc_boxes_outside_medium_target": doc_boxes_outside_medium_target,
            "planned_box_count": sum(group.demand for group in planning_groups),
        }
        return planning_groups

    def _consume_medium_target_for_document_group(
        self,
        remaining: Counter[tuple[str, ...]],
        group: SmallBoxGroup,
    ) -> int:
        need = int(group.demand)
        exact_key = self._coarse_key(group)
        take = min(need, max(0, remaining.get(exact_key, 0)))
        if take > 0:
            remaining[exact_key] -= take
            need -= take
        if need <= 0:
            return 0
        return need

    @staticmethod
    def _copy_area_allowlist(group) -> set[str] | None:
        allowlist = getattr(group, "area_allowlist", None)
        if allowlist is None:
            return None
        return set(allowlist)

    @staticmethod
    def _copy_group_with_demand(group: SmallBoxGroup, demand: int) -> SmallBoxGroup:
        return SmallBoxGroup(
            group_id=group.group_id,
            voyage_id=group.voyage_id,
            status=group.status,
            port=group.port,
            size=group.size,
            height=group.height,
            weight_class=group.weight_class,
            demand=int(demand),
            pre_stow=group.pre_stow,
            special_stow=group.special_stow,
            special_stow_code=group.special_stow_code,
            attributes=dict(getattr(group, "attributes", {}) or {}),
            area_allowlist=ColumnGenerationPlanner._copy_area_allowlist(group),
        )

    @staticmethod
    def _medium_group_as_small_group(group, demand: int) -> SmallBoxGroup:
        return SmallBoxGroup(
            group_id=group.group_id,
            voyage_id=group.voyage_id,
            status=group.status,
            port=group.port,
            size=group.size,
            height=getattr(group, "height", "UNK") or "UNK",
            weight_class=getattr(group, "weight_class", "UNK") or "UNK",
            demand=int(demand),
            pre_stow=False,
            special_stow=False,
            special_stow_code="",
            attributes=dict(getattr(group, "attributes", {}) or {}),
            area_allowlist=ColumnGenerationPlanner._copy_area_allowlist(group),
        )

    def _fallback_group_attributes(
        self,
        voyage_id: str,
        flow: str,
        port: str,
        size: str,
        height: str,
        weight_class: str = "UNK",
    ) -> dict[str, str]:
        attrs: list[str] = []
        rules = getattr(self.problem, "attribute_rules", None)
        if rules is not None:
            rule_sets = [rules.coarse_for(voyage_id), rules.fine_for(voyage_id)]
            rule_sets.extend([rules.bay_no_mix_for(voyage_id), rules.row_no_mix_for(voyage_id)])
            for values in rule_sets:
                for attr in values:
                    text = str(attr)
                    if text and text not in attrs:
                        attrs.append(text)
        fallback_values = {
            "status": flow,
            "flow": flow,
            "IYC_STS_CSTATUSCD": flow,
            "size": size,
            "size_mode": size,
            "IYC_CSZ_CSIZECD": size,
            "port": port,
            "IYC_POT_UNLDPORT": port,
            "height": height,
            "IYC_CHEIGHTCD": height,
        }
        out: dict[str, str] = {}
        for attr in attrs:
            out[attr] = fallback_values.get(attr, "MIXED")
        return out

    @staticmethod
    def _forecast_height_weights(
        groups: list[SmallBoxGroup],
    ) -> dict[tuple, Counter[str]]:
        weights: dict[tuple, Counter[str]] = defaultdict(Counter)
        for group in groups:
            height = group.height or "HQ"
            keys = [
                (group.voyage_id, group.status, group.port, group.size),
                (group.voyage_id, group.status, group.size),
                (group.status, group.size),
                (group.size,),
                ("*",),
            ]
            for key in keys:
                weights[key][height] += group.demand
        return weights

    @classmethod
    def _split_forecast_by_height(
        cls,
        voyage_id: str,
        flow: str,
        port: str,
        size: str,
        qty: int,
        height_weights: dict[tuple, Counter[str]],
    ) -> list[tuple[str, int]]:
        if qty <= 0:
            return []
        for key in (
            (voyage_id, flow, port, size),
            (voyage_id, flow, size),
            (flow, size),
            (size,),
            ("*",),
        ):
            weights = height_weights.get(key)
            if weights:
                return cls._allocate_integer_by_weights(weights, qty)
        return [("HQ", qty)]

    @staticmethod
    def _allocate_integer_by_weights(weights: Counter[str], total: int) -> list[tuple[str, int]]:
        items = [(key, value) for key, value in sorted(weights.items()) if value > 0]
        if not items or total <= 0:
            return []
        source_total = sum(value for _key, value in items)
        raw = [value * total / source_total for _key, value in items]
        base = [int(value) for value in raw]
        remain = total - sum(base)
        order = sorted(range(len(raw)), key=lambda idx: raw[idx] - base[idx], reverse=True)
        for idx in order[:remain]:
            base[idx] += 1
        return [(key, qty) for (key, _value), qty in zip(items, base) if qty > 0]

    def solve(self) -> ColumnGenerationResult:
        self._build_initial_columns()
        base_initial_column_count = len(self._columns)
        seed_stats = self._seed_restricted_master_columns()
        if self.config.full_column_pool:
            before_full_pool = len(self._columns)
            self._expand_all_candidate_columns()
            full_pool_added_columns = len(self._columns) - before_full_pool
        else:
            full_pool_added_columns = 0
        diagnostics: dict = {
            "algorithm": "small_plan_first_column_generation",
            "target_voyages": self.problem.target_voyages,
            "user_area_constraints": getattr(self.problem, "user_area_constraint_summary", {}),
            "attribute_rules": self.attribute_rules.as_dict() if hasattr(self.attribute_rules, "as_dict") else {},
            "small_doc_group_count": int(self.demand_stats.get("source_doc_group_count", 0)),
            "small_doc_box_count": int(self.demand_stats.get("source_doc_box_count", 0)),
            "planned_group_count": len(self.groups),
            "planned_box_count": sum(group.demand for group in self.groups),
            "berth_distance_count": len(self.problem.berth_distances),
            "berth_by_voyage": self.problem.berth_by_voyage,
            "demand_mode": self.config.demand_mode,
            "demand_alignment": self.demand_stats,
            "medium_doc_floor_added_boxes": getattr(self.problem, "medium_doc_floor_added_boxes", 0),
            "medium_doc_floor_added_groups": getattr(self.problem, "medium_doc_floor_added_groups", 0),
            "medium_doc_floor_shifted_boxes": getattr(self.problem, "medium_doc_floor_shifted_boxes", 0),
            "medium_doc_floor_shifted_groups": getattr(self.problem, "medium_doc_floor_shifted_groups", 0),
            "medium_doc_floor_by_coarse_group": getattr(self.problem, "medium_doc_floor_by_coarse_group", {}),
            "medium_doc_floor_added_by_coarse_group": getattr(
                self.problem,
                "medium_doc_floor_added_by_coarse_group",
                {},
            ),
            "medium_doc_floor_shifted_by_coarse_group": getattr(
                self.problem,
                "medium_doc_floor_shifted_by_coarse_group",
                {},
            ),
            "initial_column_count": len(self._columns),
            "base_initial_column_count": base_initial_column_count,
            "full_column_pool": bool(self.config.full_column_pool),
            "full_pool_added_columns": full_pool_added_columns,
            **seed_stats,
            "pricing_iterations": [],
            "scip_available": False,
            "used_greedy_fallback": False,
            "concentration_penalties": {
                "fine_group_area": self.config.small_plan_group_area_split_penalty,
                "fine_group_block": self.config.small_plan_group_block_split_penalty,
                "fine_group_bay": self.config.small_plan_group_bay_split_penalty,
                "coarse_area_block": self.config.small_plan_coarse_area_block_split_penalty,
                "coarse_area_bay": self.config.small_plan_coarse_area_bay_split_penalty,
                "medium_concentrated_group_threshold": self.config.medium_concentrated_group_threshold,
                "medium_small_group_area_split": self.config.medium_small_group_area_split_penalty,
                "medium_small_group_fragment": self.config.medium_small_group_fragment_penalty,
                "medium_large_group_min_area_boxes": self.config.medium_large_group_min_area_boxes,
                "medium_large_group_small_area": self.config.medium_large_group_small_area_penalty,
                "medium_large_group_area_open": self.config.medium_large_group_area_open_penalty,
                "medium_large_group_target_area_boxes": self.config.medium_large_group_target_area_boxes,
                "medium_large_group_area_excess": self.config.medium_large_group_area_excess_penalty,
                "existing_same_coarse_bay_reward": self.config.existing_coarse_bay_reward,
                "existing_same_coarse_neighbor_bay_reward": self.config.existing_coarse_neighbor_bay_reward,
                "existing_other_coarse_bay_penalty": self.config.existing_other_coarse_bay_penalty,
                "existing_other_coarse_neighbor_bay_penalty": self.config.existing_other_coarse_neighbor_bay_penalty,
                "existing_coarse_neighbor_max_bay_distance": self.config.existing_coarse_neighbor_max_bay_distance,
                "twenty_max_consecutive_bays": self.config.twenty_max_consecutive_bays,
                "twenty_consecutive_violation_penalty": self.config.twenty_consecutive_violation_penalty,
            },
            "existing_coarse_anchors": {
                "mode": "stage_scope_bay_proximity",
                "area_key_count": len(self.existing_coarse_area_load),
                "bay_key_count": len(self.existing_coarse_bay_load),
                "box_count": int(sum(self.existing_coarse_bay_load.values())),
            },
            "inheritance_penalties": {
                "unplaced": self.config.unplaced_penalty,
                "required_area_reward": self.config.required_area_reward,
                "big_plan_area_deviation": self.config.big_plan_area_deviation_penalty,
                "big_plan_fallback_tier": self.config.big_plan_fallback_tier_penalty,
            },
            "export_e_area_controls": {
                "max_bays_per_voyage_area": self.config.export_e_area_max_bays_per_voyage_area,
                "non_40_penalty": self.config.export_e_area_non_40_penalty,
            },
        }

        selected: Counter[int]
        unplaced: Counter[str]
        try:
            if not self.config.use_scip:
                raise RuntimeError("SCIP disabled by config")
            selected, unplaced, master_stats = self._solve_by_column_generation()
            diagnostics.update(master_stats)
        except Exception as exc:
            diagnostics["used_greedy_fallback"] = True
            diagnostics["scip_failure"] = f"{type(exc).__name__}: {exc}"
            selected, unplaced = self._greedy_fallback()

        diagnostics.update(
            {
                "final_repair_enabled": False,
                "pre_repair_unplaced_boxes": sum(unplaced.values()),
                "post_repair_unplaced_boxes": sum(unplaced.values()),
                "used_unplaced_repair": False,
                "unplaced_repair_method": "not_run_after_column_generation",
            }
        )
        relayout_start = perf_counter()
        selected, relayout_stats = self._post_repair_area_relayout(selected, unplaced)
        relayout_stats["post_repair_area_relayout_elapsed_seconds"] = round(perf_counter() - relayout_start, 3)
        diagnostics.update(relayout_stats)

        if self._uses_original_output_scope():
            small_rows = self._make_small_rows(selected, allowed_sources={"document"})
        else:
            small_rows = self._make_small_rows(selected)
        medium_rows = self._make_medium_rows_from_selected_columns(selected, plan_level="medium")
        unplaced_rows = self._unplaced_group_details(unplaced)
        consistency_stats = self._small_medium_consistency_stats(small_rows, medium_rows)
        bay_consistency_stats = self._small_medium_bay_consistency_stats(small_rows, medium_rows)
        medium_fragmentation = self._medium_fragmentation_stats(medium_rows)
        diagnostics.update(
            {
                "final_column_count": len(self._columns),
                "selected_column_count": sum(1 for qty in selected.values() if qty > 0),
                "medium_plan_granularity": "bay",
                "medium_plan_source": "selected_columns",
                "small_row_count": len(small_rows),
                "medium_row_count": len(medium_rows),
                "planned_small_boxes": sum(int(row["planned_boxes"]) for row in small_rows),
                "planned_medium_boxes": sum(int(row["planned_boxes"]) for row in medium_rows),
                "planned_medium_by_source": self._planned_medium_by_source(medium_rows),
                "medium_area_rows_below_min_boxes": self._count_medium_area_rows_below_min(medium_rows),
                "medium_fragmentation": medium_fragmentation,
                "medium_big_plan_inheritance": self._medium_big_plan_inheritance_stats(medium_rows),
                "final_medium_inheritance_energy_components": self._medium_inheritance_energy_components(medium_rows),
                "export_e_area_usage": self._export_e_area_usage(selected),
                "export_voyage_row_no_mix": self._export_voyage_row_no_mix_stats(selected),
                "twenty_consecutive_bay_rule": self._twenty_run_diagnostics(selected),
                "user_required_area_usage": self._user_required_area_usage(medium_rows, small_rows),
                "user_area_constraint_violations": self._user_area_constraint_violations(medium_rows, small_rows),
                "unplaced_boxes": sum(unplaced.values()),
                "unplaced_by_group": {key: qty for key, qty in sorted(unplaced.items()) if qty > 0},
                "unplaced_group_details": unplaced_rows,
                **consistency_stats,
                **bay_consistency_stats,
            }
        )
        return ColumnGenerationResult(
            medium_rows=medium_rows,
            small_rows=small_rows,
            diagnostics=diagnostics,
            unplaced_rows=unplaced_rows,
            columns=self._columns,
        )

    def _repair_or_replace_unplaced_solution(
        self,
        selected: Counter[int],
        unplaced: Counter[str],
    ) -> tuple[Counter[int], Counter[str], dict]:
        initial_unplaced = sum(unplaced.values())
        stats: dict = {
            "pre_repair_unplaced_boxes": initial_unplaced,
            "post_repair_unplaced_boxes": initial_unplaced,
            "used_unplaced_repair": False,
            "unplaced_repair_method": "",
        }
        if initial_unplaced <= 0:
            return selected, unplaced, stats

        staged_selected, staged_unplaced, stage_stats = self._staged_repair_selected_solution(selected)
        candidates = [
            ("master_incumbent", selected, unplaced),
            ("staged_fallback_repair", staged_selected, staged_unplaced),
        ]
        method, best_selected, best_unplaced = min(
            candidates,
            key=lambda item: self._solution_rank(item[1], item[2]),
        )
        stats.update(
            {
                "used_unplaced_repair": method != "master_incumbent",
                "unplaced_repair_method": method,
                "staged_repair_unplaced_boxes": sum(staged_unplaced.values()),
                "staged_repair_iterations": stage_stats.get("iterations", []),
                "staged_repair_candidates": stage_stats.get("candidates", []),
                "staged_repair_selected_candidate": stage_stats.get("selected_candidate", ""),
                "post_repair_unplaced_boxes": sum(best_unplaced.values()),
            }
        )
        return best_selected, best_unplaced, stats

    def _post_repair_area_relayout(
        self,
        selected: Counter[int],
        unplaced: Counter[str],
    ) -> tuple[Counter[int], dict]:
        stats: dict = {
            "post_repair_area_relayout_enabled": bool(getattr(self.config, "post_repair_area_relayout_enabled", True)),
            "post_repair_area_relayout_used": False,
            "post_repair_area_relayout_accepted": False,
            "post_repair_area_relayout_skip_reason": "",
            "post_repair_area_relayout_unplaced_boxes": sum(unplaced.values()),
            "post_repair_area_relayout_partial_solution": sum(unplaced.values()) > 0,
        }
        if not stats["post_repair_area_relayout_enabled"]:
            stats["post_repair_area_relayout_skip_reason"] = "disabled"
            return selected, stats
        if self.config.medium_plan_bay_quota is not None:
            stats["post_repair_area_relayout_skip_reason"] = "fixed_medium_bay_quota"
            return selected, stats

        target_group_area = self._selected_group_area_quantities(selected)
        if not target_group_area:
            stats["post_repair_area_relayout_skip_reason"] = "empty_solution"
            return selected, stats

        before_metrics = self._area_relayout_concentration_metrics(selected)
        candidate, candidate_stats = self._build_area_relayout_solution(target_group_area, selected)
        stats.update(candidate_stats)
        stats["post_repair_area_relayout_before"] = before_metrics
        if candidate is None:
            stats["post_repair_area_relayout_skip_reason"] = candidate_stats.get("post_repair_area_relayout_failure_reason", "failed")
            return selected, stats
        if self._selected_group_area_quantities(candidate) != target_group_area:
            stats["post_repair_area_relayout_skip_reason"] = "area_quantities_changed"
            return selected, stats

        after_metrics = self._area_relayout_concentration_metrics(candidate)
        stats["post_repair_area_relayout_after"] = after_metrics
        stats["post_repair_area_relayout_delta_score"] = round(before_metrics["score"] - after_metrics["score"], 6)
        stats["post_repair_area_relayout_used"] = True
        before_rank = (before_metrics["twenty_consecutive_violation_windows"], before_metrics["score"])
        after_rank = (after_metrics["twenty_consecutive_violation_windows"], after_metrics["score"])
        if after_rank < before_rank:
            stats["post_repair_area_relayout_accepted"] = True
            return candidate, stats
        stats["post_repair_area_relayout_skip_reason"] = "no_concentration_improvement"
        return selected, stats

    def _selected_group_area_quantities(self, selected: Counter[int]) -> Counter[tuple[str, str]]:
        quantities: Counter[tuple[str, str]] = Counter()
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            quantities[(col.group_id, col.area_no)] += int(chosen) * int(col.quantity)
        return quantities

    def _area_relayout_concentration_metrics(self, selected: Counter[int]) -> dict[str, float | int]:
        fine_area_bays: set[tuple[tuple[str, ...], str, str]] = set()
        coarse_area_bays: Counter[tuple[str, ...]] = Counter()
        existing_anchor_score = 0.0
        existing_same_coarse_bay_boxes = 0
        existing_same_coarse_neighbor_boxes = 0
        existing_other_coarse_bay_boxes = 0
        existing_other_coarse_neighbor_boxes = 0
        twenty_segment_used_bays: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
        twenty_zero_loss_boxes = 0
        twenty_isolated_empty_boxes = 0
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            qty = int(chosen) * int(col.quantity)
            fine_area_bays.add((col.fine_cluster_key, col.area_no, col.bay_key))
            coarse_area_bays[col.coarse_cluster_key + (col.area_no, col.bay_key)] += qty
            anchor_score, anchor_category = self._existing_anchor_relayout_component(col, int(chosen))
            existing_anchor_score += anchor_score
            if anchor_category == "same_bay":
                existing_same_coarse_bay_boxes += qty
            elif anchor_category == "same_neighbor":
                existing_same_coarse_neighbor_boxes += qty
            elif anchor_category == "other_bay":
                existing_other_coarse_bay_boxes += qty
            elif anchor_category == "other_neighbor":
                existing_other_coarse_neighbor_boxes += qty
            if col.size == "20":
                segment = self._large_segment_key_for_20_bay(col.bay_key)
                if segment is None:
                    if not self._bay_existing_size_modes(col.bay_key):
                        twenty_isolated_empty_boxes += qty
                else:
                    twenty_segment_used_bays[segment].add(col.bay_key)
                    if self._twenty_segment_static_loss_for_bay(col.bay_key) <= 0:
                        twenty_zero_loss_boxes += qty

        fine_area_pairs = {(fine_key, area_no) for fine_key, area_no, _bay_key in fine_area_bays}
        coarse_area_pairs = {key[:-1] for key in coarse_area_bays}
        twenty_segment_loss = self._twenty_segment_total_loss(twenty_segment_used_bays)
        twenty_run_violations = self._twenty_run_violation_count(selected)
        baseline_twenty_run_violations = self._twenty_run_violation_count_for_bays(set(self.existing_twenty_bays))
        fine_excess_bays = max(0, len(fine_area_bays) - len(fine_area_pairs))
        coarse_excess_bays = max(0, len(coarse_area_bays) - len(coarse_area_pairs))
        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        coarse_tail_boxes = 0
        if min_boxes > 0:
            for qty in coarse_area_bays.values():
                if 0 < qty < min_boxes:
                    coarse_tail_boxes += min_boxes - int(qty)
        score = (
            1000.0 * fine_excess_bays
            + 360.0 * coarse_excess_bays
            + 24.0 * coarse_tail_boxes
            + 10.0 * len(fine_area_bays)
            + 4.0 * len(coarse_area_bays)
            + existing_anchor_score
            + self._twenty_segment_loss_penalty() * twenty_segment_loss
            + self._twenty_consecutive_violation_penalty()
            * max(0, twenty_run_violations - baseline_twenty_run_violations)
            - float(getattr(self.config, "twenty_isolated_bay_reward", 0.0) or 0.0)
            * min(50, twenty_isolated_empty_boxes + twenty_zero_loss_boxes)
        )
        return {
            "score": round(score, 6),
            "fine_area_bays": len(fine_area_bays),
            "fine_area_pairs": len(fine_area_pairs),
            "fine_excess_bays": fine_excess_bays,
            "coarse_area_bays": len(coarse_area_bays),
            "coarse_area_pairs": len(coarse_area_pairs),
            "coarse_excess_bays": coarse_excess_bays,
            "coarse_tail_boxes": coarse_tail_boxes,
            "existing_anchor_score": round(existing_anchor_score, 6),
            "existing_same_coarse_bay_boxes": existing_same_coarse_bay_boxes,
            "existing_same_coarse_neighbor_boxes": existing_same_coarse_neighbor_boxes,
            "existing_other_coarse_bay_boxes": existing_other_coarse_bay_boxes,
            "existing_other_coarse_neighbor_boxes": existing_other_coarse_neighbor_boxes,
            "twenty_large_segment_loss": twenty_segment_loss,
            "twenty_large_segment_used_bay_count": sum(len(bays) for bays in twenty_segment_used_bays.values()),
            "twenty_consecutive_violation_windows": twenty_run_violations,
            "twenty_consecutive_new_violation_windows": max(
                0,
                twenty_run_violations - baseline_twenty_run_violations,
            ),
            "twenty_zero_loss_boxes": twenty_zero_loss_boxes,
            "twenty_isolated_empty_boxes": twenty_isolated_empty_boxes,
        }

    def _build_area_relayout_solution(
        self,
        target_group_area: Counter[tuple[str, str]],
        original_selected: Counter[int],
    ) -> tuple[Counter[int] | None, dict]:
        selected: Counter[int] = Counter()
        state = self._empty_selection_state()
        relayout_state = {
            "coarse_area_bay_load": Counter(),
            "group_area_bay_load": Counter(),
        }
        stats: dict = {
            "post_repair_area_relayout_group_area_pairs": len(target_group_area),
            "post_repair_area_relayout_selected_columns": 0,
            "post_repair_area_relayout_failure_reason": "",
            "post_repair_area_relayout_areas_attempted": 0,
            "post_repair_area_relayout_areas_relaid": 0,
            "post_repair_area_relayout_areas_kept_original": 0,
        }

        coarse_area_total: Counter[tuple[str, ...]] = Counter()
        for (group_id, area_no), qty in target_group_area.items():
            group = self.groups_by_id.get(group_id)
            if group is None:
                stats["post_repair_area_relayout_failure_reason"] = "missing_group"
                stats["post_repair_area_relayout_failed_group"] = group_id
                return None, stats
            coarse_area_total[self._coarse_cluster_key(group) + (area_no,)] += int(qty)

        by_area_coarse: defaultdict[tuple[str, tuple[str, ...]], list[tuple[SmallBoxGroup, str, int]]] = defaultdict(list)
        target_areas: set[str] = set()
        for (group_id, area_no), qty in target_group_area.items():
            qty = int(qty)
            if qty <= 0:
                continue
            target_areas.add(area_no)
            group = self.groups_by_id[group_id]
            by_area_coarse[(area_no, self._coarse_cluster_key(group))].append((group, area_no, qty))

        original_by_area: defaultdict[str, list[tuple[int, PlacementColumn, int]]] = defaultdict(list)
        for idx, chosen in original_selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            original_by_area[col.area_no].append((idx, col, int(chosen)))

        for area_no in sorted(target_areas):
            stats["post_repair_area_relayout_areas_attempted"] += 1
            trial_selected = Counter(selected)
            trial_state = self._copy_selection_state_for_relayout(state)
            trial_relayout_state = self._copy_area_relayout_state(relayout_state)
            area_success = True
            area_coarse_keys = sorted(
                [key for key in by_area_coarse if key[0] == area_no],
                key=lambda key: (
                    0 if coarse_area_total[key[1] + (key[0],)] <= self.config.medium_concentrated_group_threshold else 1,
                    coarse_area_total[key[1] + (key[0],)] if coarse_area_total[key[1] + (key[0],)] <= self.config.medium_concentrated_group_threshold else -coarse_area_total[key[1] + (key[0],)],
                    key[1],
                ),
            )
            for area_coarse_key in area_coarse_keys:
                entries = sorted(
                    by_area_coarse[area_coarse_key],
                    key=lambda item: (-int(item[2]), self._group_sort_key(item[0])),
                )
                for group, entry_area_no, target_qty in entries:
                    remaining = int(target_qty)
                    while remaining > 0:
                        choice = self._best_area_relayout_column(group, entry_area_no, remaining, trial_state, trial_relayout_state)
                        if choice is None:
                            area_success = False
                            stats["post_repair_area_relayout_failure_reason"] = "partial_area_fallback"
                            stats["post_repair_area_relayout_last_failed_group"] = group.group_id
                            stats["post_repair_area_relayout_last_failed_area"] = entry_area_no
                            stats["post_repair_area_relayout_last_failed_remaining"] = remaining
                            break
                        idx, col = choice
                        self._apply_column_to_state(col, trial_state)
                        self._apply_column_to_area_relayout_state(col, trial_relayout_state)
                        trial_selected[idx] += 1
                        remaining -= int(col.quantity)
                    if not area_success:
                        break
                if not area_success:
                    break
            if area_success:
                selected = trial_selected
                state = trial_state
                relayout_state = trial_relayout_state
                stats["post_repair_area_relayout_areas_relaid"] += 1
            else:
                stats["post_repair_area_relayout_areas_kept_original"] += 1
                for idx, col, chosen in original_by_area.get(area_no, []):
                    for _ in range(chosen):
                        self._apply_column_to_state(col, state)
                        self._apply_column_to_area_relayout_state(col, relayout_state)
                        selected[idx] += 1

        stats["post_repair_area_relayout_selected_columns"] = sum(1 for qty in selected.values() if qty > 0)
        return selected, stats

    @staticmethod
    def _copy_selection_state_for_relayout(state: dict) -> dict:
        copied = {}
        for key, value in state.items():
            if isinstance(value, Counter):
                copied[key] = Counter(value)
            elif isinstance(value, set):
                copied[key] = set(value)
            elif isinstance(value, dict):
                copied[key] = dict(value)
            else:
                copied[key] = value
        return copied

    @staticmethod
    def _copy_area_relayout_state(state: dict) -> dict:
        return {key: Counter(value) for key, value in state.items()}

    def _best_area_relayout_column(
        self,
        group: SmallBoxGroup,
        area_no: str,
        remaining: int,
        state: dict,
        relayout_state: dict,
    ) -> tuple[int, PlacementColumn] | None:
        best: tuple[tuple[int, float, int, int, str], int, PlacementColumn] | None = None
        for bay_key in self.bays_by_area.get(area_no, []):
            if self._max_quantity_in_bay(group, bay_key) <= 0:
                continue
            base_cost = self._column_base_cost(group, bay_key)
            capacity = self._remaining_capacity_for_group_bay(group, bay_key, state, remaining, enforce_quota=False)
            if capacity <= 0:
                continue
            qty = min(int(remaining), int(capacity))
            patterns = self._row_allocation_patterns_for_column(
                group,
                bay_key,
                qty,
                state=state,
                max_patterns=max(1, int(getattr(self.config, "post_repair_area_relayout_max_patterns", 8) or 8)),
            )
            for pattern in patterns:
                idx = self._ensure_area_relayout_column(group, bay_key, qty, base_cost, pattern)
                if idx is None:
                    continue
                col = self._columns[idx]
                if not self._column_fits_state(col, state, remaining, enforce_quota=False):
                    continue
                score = (
                    self._twenty_run_incremental_violation(group, bay_key, state),
                    self._area_relayout_column_score(col, relayout_state, remaining)
                    + self._twenty_bay_state_cost(group, bay_key, state),
                    0 if int(col.quantity) >= int(remaining) else 1,
                    -int(col.quantity),
                    col.bay_key,
                )
                candidate = (score, idx, col)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            return None
        return best[1], best[2]

    def _ensure_area_relayout_column(
        self,
        group: SmallBoxGroup,
        bay_key: str,
        quantity: int,
        base_cost: float,
        row_allocation: tuple[tuple[str, str, int], ...],
    ) -> int | None:
        signature = self._row_allocation_signature(row_allocation)
        triplet = (group.group_id, bay_key, int(quantity))
        for idx in self._column_indices_by_triplet.get(triplet, ()):
            col = self._columns[idx]
            if col.row_allocation == signature:
                return idx
        before = len(self._columns)
        self._add_column(group, bay_key, int(quantity), base_cost, row_allocation=signature)
        for idx in range(before, len(self._columns)):
            col = self._columns[idx]
            if (
                col.group_id == group.group_id
                and col.bay_key == bay_key
                and int(col.quantity) == int(quantity)
                and col.row_allocation == signature
            ):
                return idx
        return None

    def _area_relayout_column_score(self, col: PlacementColumn, relayout_state: dict, remaining: int) -> float:
        group_bay_key = (col.fine_cluster_key, col.area_no, col.bay_key)
        coarse_bay_key = col.coarse_cluster_key + (col.area_no, col.bay_key)
        existing_group_bay = int(relayout_state["group_area_bay_load"][group_bay_key])
        existing_coarse_bay = int(relayout_state["coarse_area_bay_load"][coarse_bay_key])
        score = 0.0
        if existing_group_bay <= 0:
            score += 1000.0
        else:
            score -= min(300.0, float(existing_group_bay))
        if existing_coarse_bay <= 0:
            score += 360.0
        else:
            score -= min(180.0, float(existing_coarse_bay))
        score -= 12.0 * int(col.quantity)
        if int(remaining) - int(col.quantity) > 0:
            score += 80.0
        score += self._existing_anchor_relayout_component(col)[0]
        score += 0.01 * self.bays[col.bay_key].bay_order
        return score

    def _existing_anchor_relayout_component(
        self,
        col: PlacementColumn,
        quantity_multiplier: int = 1,
    ) -> tuple[float, str | None]:
        group = self.groups_by_id.get(col.group_id)
        if group is None:
            return 0.0, None
        box_multiplier = max(1, int(quantity_multiplier)) * max(1, int(col.quantity))
        existing_bay_load = self._existing_coarse_bay_load_for_group(group, col.bay_key)
        if existing_bay_load > 0:
            score = -float(self.config.existing_coarse_bay_reward) * min(5, existing_bay_load) * box_multiplier
            return score, "same_bay"

        same_distance = self._existing_same_coarse_bay_distance(group, col.bay_key)
        same_reward = self._existing_neighbor_reward(same_distance) if same_distance is not None else 0.0
        if same_reward > 0:
            return -same_reward * box_multiplier, "same_neighbor"

        existing_other_load = self._existing_other_coarse_bay_load_for_group(group, col.bay_key)
        if existing_other_load > 0:
            score = float(self.config.existing_other_coarse_bay_penalty) * min(5, existing_other_load) * box_multiplier
            return score, "other_bay"

        other_distance = self._existing_other_coarse_bay_distance(group, col.bay_key)
        other_penalty = self._existing_other_neighbor_penalty(other_distance) if other_distance is not None else 0.0
        if other_penalty > 0:
            return other_penalty * box_multiplier, "other_neighbor"
        return 0.0, None

    def _apply_column_to_area_relayout_state(self, col: PlacementColumn, relayout_state: dict) -> None:
        group_bay_key = (col.fine_cluster_key, col.area_no, col.bay_key)
        coarse_bay_key = col.coarse_cluster_key + (col.area_no, col.bay_key)
        relayout_state["group_area_bay_load"][group_bay_key] += int(col.quantity)
        relayout_state["coarse_area_bay_load"][coarse_bay_key] += int(col.quantity)

    def _solution_rank(self, selected: Counter[int], unplaced: Counter[str]) -> tuple[int, int, int, float, int]:
        return (
            self._document_unplaced_boxes(unplaced),
            sum(unplaced.values()),
            self._twenty_run_violation_count(selected),
            self._selected_solution_energy(selected, unplaced),
            sum(1 for qty in selected.values() if qty > 0),
        )

    def _selected_solution_energy(self, selected: Counter[int], unplaced: Counter[str]) -> float:
        energy = sum(
            self._unplaced_penalty_for_group_id(group_id) * max(0, int(qty))
            for group_id, qty in unplaced.items()
        )
        actual_quota: Counter[tuple[str, str, str, str]] = Counter()
        actual_coarse_area: Counter[tuple[str, ...]] = Counter()
        used_group_area: set[tuple[tuple[str, ...], str]] = set()
        used_group_block: set[tuple[tuple[str, ...], str]] = set()
        used_coarse_area_block: set[tuple[str, ...]] = set()
        used_coarse_area_bay: set[tuple[str, ...]] = set()
        used_voyage_area: set[tuple[str, str]] = set()
        used_twenty_segment_bays: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            multiplier = int(chosen)
            energy += (col.intrinsic_cost + self.config.small_plan_group_bay_split_penalty) * multiplier
            qty = col.quantity * multiplier
            actual_quota[col.quota_key] += qty
            actual_coarse_area[col.coarse_cluster_key + (col.area_no,)] += qty
            energy += self._area_fallback_tier_penalty_for_column(col) * qty
            used_group_area.add((col.fine_cluster_key, col.area_no))
            if col.block_id:
                used_group_block.add((col.fine_cluster_key, col.block_id))
                used_coarse_area_block.add(col.coarse_cluster_key + (col.area_no, col.block_id))
            used_coarse_area_bay.add(col.coarse_cluster_key + (col.area_no, col.bay_key))
            used_voyage_area.add((col.voyage_id, col.area_no))
            if col.size == "20":
                segment = self._large_segment_key_for_20_bay(col.bay_key)
                if segment is not None:
                    used_twenty_segment_bays[segment].add(col.bay_key)

        energy += self.config.small_plan_group_area_split_penalty * len(used_group_area)
        energy += self.config.small_plan_group_block_split_penalty * len(used_group_block)
        energy += self.config.small_plan_coarse_area_block_split_penalty * len(used_coarse_area_block)
        energy += self.config.small_plan_coarse_area_bay_split_penalty * len(used_coarse_area_bay)
        energy += sum(self._voyage_area_cost(voyage_id, area_no) for voyage_id, area_no in used_voyage_area)
        energy += self._twenty_segment_loss_penalty() * self._twenty_segment_total_loss(used_twenty_segment_bays)
        energy += self._twenty_consecutive_violation_penalty() * max(
            0,
            self._twenty_run_violation_count(selected)
            - self._twenty_run_violation_count_for_bays(set(self.existing_twenty_bays)),
        )

        target_keys = set(actual_quota)
        for key, qty in self.quota_by_key.items():
            voyage_id, flow, _area_no, big_size = key
            if qty > 0 and self.voyage_flow_size_demand[(voyage_id, flow, big_size)] > 0:
                target_keys.add(key)
        for voyage_id, flow, area_no, big_size in target_keys:
            target = self._area_size_target(voyage_id, flow, area_no, big_size)
            energy += self.config.big_plan_area_deviation_penalty * abs(actual_quota.get((voyage_id, flow, area_no, big_size), 0) - target)

        by_coarse: defaultdict[tuple[str, ...], list[float]] = defaultdict(list)
        for key, qty in actual_coarse_area.items():
            by_coarse[tuple(key[:-1])].append(float(qty))
        for coarse_key, quantities in by_coarse.items():
            demand = max(1, int(self._coarse_metric_demand(coarse_key, sum(quantities))))
            if self._prefers_concentrated_coarse_key(coarse_key):
                if quantities:
                    energy += self.config.medium_small_group_area_split_penalty * max(0, len(quantities) - 1)
                    energy -= self.config.medium_small_group_fragment_penalty * max(quantities)
            else:
                energy += self.config.medium_large_group_area_open_penalty * max(
                    0,
                    len(quantities) - self._target_large_group_area_count(coarse_key, demand),
                )
                target_boxes = max(1, int(self.config.medium_large_group_target_area_boxes or 1))
                energy += self.config.medium_large_group_area_excess_penalty * sum(
                    max(0.0, qty - target_boxes) for qty in quantities
                )
                min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
                if min_boxes > 0:
                    small_area_penalty = self.config.medium_large_group_small_area_penalty / max(1.0, min_boxes)
                    energy += sum(small_area_penalty * max(0.0, min_boxes - qty) for qty in quantities if qty > 0)
                if len(quantities) > 1:
                    pair_penalty = self.config.group_area_balance_penalty / max(1.0, demand) / max(1, len(quantities) - 1)
                    for left_index, left in enumerate(quantities):
                        for right in quantities[left_index + 1 :]:
                            energy += pair_penalty * abs(left - right)
        return float(energy)

    def _document_unplaced_boxes(self, unplaced: Counter[str]) -> int:
        return sum(
            max(0, int(qty))
            for group_id, qty in unplaced.items()
            if self.group_source.get(group_id, "document") == "document"
        )

    def _source_rank_for_group_id(self, group_id: str) -> int:
        return 0 if self.group_source.get(group_id, "document") == "document" else 1

    def _source_rank_for_group(self, group: SmallBoxGroup) -> int:
        return self._source_rank_for_group_id(group.group_id)

    def _unplaced_penalty_for_group_id(self, group_id: str) -> float:
        source = self.group_source.get(group_id, "document")
        multiplier = (
            self.config.document_unplaced_penalty_multiplier
            if source == "document"
            else self.config.forecast_unplaced_penalty_multiplier
        )
        return float(self.config.unplaced_penalty) * max(0.0, float(multiplier))

    def _unplaced_objective_for_group(self, group: SmallBoxGroup, objective_mode: str) -> float:
        if objective_mode == "min_unplaced":
            source = self.group_source.get(group.group_id, "document")
            multiplier = (
                self.config.document_unplaced_penalty_multiplier
                if source == "document"
                else self.config.forecast_unplaced_penalty_multiplier
            )
            return max(1.0, float(multiplier))
        return self._unplaced_penalty_for_group_id(group.group_id)

    def _area_fallback_tier_penalty_for_column(self, col: PlacementColumn) -> float:
        tier = self._area_fallback_tier_for_attrs(
            col.voyage_id,
            col.flow,
            col.area_no,
            col.big_plan_size,
        )
        return float(self.config.big_plan_fallback_tier_penalty) * tier

    def _repair_from_column_priority(
        self,
        column_values: dict[int, float],
        allow_new_columns: bool = True,
    ) -> tuple[Counter[int], Counter[str]]:
        selected: Counter[int] = Counter()
        state = self._empty_selection_state()
        placed: Counter[str] = Counter()
        priority = sorted(
            ((float(value), idx) for idx, value in column_values.items() if value > 1e-6),
            reverse=True,
        )
        for _value, idx in priority:
            if idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            remaining = int(self.group_demand.get(col.group_id, 0)) - int(placed.get(col.group_id, 0))
            if remaining <= 0 or col.quantity > remaining:
                continue
            if not self._column_fits_state(col, state, remaining):
                continue
            self._apply_column_to_state(col, state)
            selected[idx] = 1
            placed[col.group_id] += col.quantity
        return self._repair_selected_solution(selected, allow_new_columns=allow_new_columns)

    def _seed_restricted_master_columns(self) -> dict[str, int]:
        before = len(self._columns)
        selected, unplaced = self._repair_selected_solution(Counter())
        self._master_seed_selected = selected
        self._master_seed_unplaced = unplaced
        self._master_start_selected = selected
        self._master_start_unplaced = unplaced
        return {
            "master_seed_added_columns": len(self._columns) - before,
            "master_seed_selected_columns": sum(1 for qty in selected.values() if qty > 0),
            "master_seed_unplaced_boxes": sum(unplaced.values()),
        }

    def _uses_original_output_scope(self) -> bool:
        return (self.config.demand_mode or "original").strip().lower().replace("_", "-") == "original"

    def _count_medium_area_rows_below_min(self, medium_rows: list[dict]) -> int:
        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        if min_boxes <= 1:
            return 0
        area_counter: Counter[tuple[str, str, str, str, str]] = Counter()
        for row in medium_rows:
            qty = int(row.get("planned_boxes", 0) or 0)
            if qty <= 0:
                continue
            key = (
                str(row.get("voyage_id", "")),
                str(row.get("flow", "")),
                str(row.get("port", "")),
                str(row.get("size", "")),
                str(row.get("area_no", "")),
            )
            area_counter[key] += qty
        return sum(
            1
            for qty in area_counter.values()
            if 0 < qty < min_boxes
        )

    def _medium_fragmentation_stats(self, medium_rows: list[dict]) -> dict[str, int | float]:
        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        concentration_threshold = max(0, int(self.config.medium_concentrated_group_threshold or 0))
        by_coarse_area: Counter[tuple[str, str, str, str, str]] = Counter()
        for row in medium_rows:
            qty = int(row.get("planned_boxes", 0) or 0)
            if qty <= 0:
                continue
            key = (
                str(row.get("voyage_id", "")),
                str(row.get("flow", "")),
                str(row.get("port", "")),
                str(row.get("size", "")),
                str(row.get("area_no", "")),
            )
            by_coarse_area[key] += qty
        by_coarse: defaultdict[tuple[str, str, str, str], list[int]] = defaultdict(list)
        for (*coarse_key, _area_no), qty in by_coarse_area.items():
            by_coarse[tuple(coarse_key)].append(qty)
        tiny_rows = 0
        small_groups = 0
        small_multi_area_groups = 0
        large_groups = 0
        large_tiny_rows = 0
        max_area_count = 0
        for quantities in by_coarse.values():
            total = sum(quantities)
            area_count = len(quantities)
            max_area_count = max(max_area_count, area_count)
            tiny_count = sum(1 for qty in quantities if min_boxes > 1 and 0 < qty < min_boxes)
            tiny_rows += tiny_count
            if concentration_threshold > 0 and total <= concentration_threshold:
                small_groups += 1
                if area_count > 1:
                    small_multi_area_groups += 1
            else:
                large_groups += 1
                large_tiny_rows += tiny_count
        return {
            "coarse_group_count": len(by_coarse),
            "tiny_area_rows": tiny_rows,
            "small_coarse_group_count": small_groups,
            "small_coarse_multi_area_groups": small_multi_area_groups,
            "large_coarse_group_count": large_groups,
            "large_coarse_tiny_area_rows": large_tiny_rows,
            "max_area_count_per_coarse_group": max_area_count,
            "average_area_count_per_coarse_group": round(
                sum(len(quantities) for quantities in by_coarse.values()) / max(1, len(by_coarse)),
                4,
            ),
        }

    @staticmethod
    def _small_medium_consistency_stats(small_rows: list[dict], medium_rows: list[dict]) -> dict[str, int]:
        small_counter: Counter[tuple[str, str, str, str, str]] = Counter()
        medium_counter: Counter[tuple[str, str, str, str, str]] = Counter()
        for row in small_rows:
            key = (
                str(row.get("voyage_id", "")),
                str(row.get("flow", "")),
                str(row.get("port", "")),
                str(row.get("size", "")),
                str(row.get("area_no", "")),
            )
            small_counter[key] += int(row.get("planned_boxes", 0) or 0)
        for row in medium_rows:
            key = (
                str(row.get("voyage_id", "")),
                str(row.get("flow", "")),
                str(row.get("port", "")),
                str(row.get("size", "")),
                str(row.get("area_no", "")),
            )
            medium_counter[key] += int(row.get("planned_boxes", 0) or 0)
        violations = 0
        shortage = 0
        for key, qty in small_counter.items():
            excess = qty - medium_counter.get(key, 0)
            if excess > 0:
                violations += 1
                shortage += excess
        return {
            "small_medium_consistency_violations": violations,
            "small_medium_consistency_shortage_boxes": shortage,
        }

    @staticmethod
    def _small_medium_bay_consistency_stats(small_rows: list[dict], medium_rows: list[dict]) -> dict[str, int]:
        small_counter: Counter[tuple[str, str, str, str, str, str]] = Counter()
        medium_counter: Counter[tuple[str, str, str, str, str, str]] = Counter()
        for row in small_rows:
            area_no = str(row.get("area_no", ""))
            bay_key = str(row.get("bay_key") or f"{area_no}-{row.get('bay_no', '')}" if row.get("bay_no") else "")
            key = (
                str(row.get("voyage_id", "")),
                str(row.get("flow", "")),
                str(row.get("port", "")),
                str(row.get("size", "")),
                area_no,
                bay_key,
            )
            small_counter[key] += int(row.get("planned_boxes", 0) or 0)
        for row in medium_rows:
            area_no = str(row.get("area_no", ""))
            bay_key = str(row.get("bay_key") or f"{area_no}-{row.get('bay_no', '')}" if row.get("bay_no") else "")
            key = (
                str(row.get("voyage_id", "")),
                str(row.get("flow", "")),
                str(row.get("port", "")),
                str(row.get("size", "")),
                area_no,
                bay_key,
            )
            medium_counter[key] += int(row.get("planned_boxes", 0) or 0)
        violations = 0
        shortage = 0
        for key, qty in small_counter.items():
            excess = qty - medium_counter.get(key, 0)
            if excess > 0:
                violations += 1
                shortage += excess
        return {
            "small_medium_bay_consistency_violations": violations,
            "small_medium_bay_consistency_shortage_boxes": shortage,
        }

    def _solve_by_column_generation(self) -> tuple[Counter[int], Counter[str], dict]:
        from pyscipopt import Model, quicksum

        solve_start = perf_counter()
        total_time_limit = max(0.0, float(getattr(self.config, "total_time_limit", 0.0) or 0.0))
        reserve_fraction = max(
            0.0,
            float(getattr(self.config, "staged_repair_reserve_fraction", 0.33) or 0.0),
        )
        reserve_min = max(
            0.0,
            float(getattr(self.config, "staged_repair_reserve_min", 60.0) or 0.0),
        )
        reserve_max = max(
            reserve_min,
            float(getattr(self.config, "staged_repair_reserve_max", 90.0) or reserve_min),
        )
        staged_repair_reserve = 0.0
        if total_time_limit > 0:
            staged_repair_reserve = min(
                total_time_limit * 0.8,
                min(reserve_max, max(reserve_min, total_time_limit * reserve_fraction)),
            )
        pricing_min_lp_time_limit = max(
            1.0,
            float(getattr(self.config, "pricing_min_lp_time_limit", 12.0) or 1.0),
        )
        pricing_iteration_setup_reserve = max(
            0.0,
            float(getattr(self.config, "pricing_iteration_setup_reserve", 8.0) or 0.0),
        )
        pricing_start_min_remaining = (
            staged_repair_reserve + pricing_min_lp_time_limit + pricing_iteration_setup_reserve
            if total_time_limit > 0
            else 0.0
        )

        def remaining_total_time() -> float | None:
            if total_time_limit <= 0:
                return None
            return total_time_limit - (perf_counter() - solve_start)

        def total_time_low(min_remaining_seconds: float = 0.0) -> bool:
            remaining = remaining_total_time()
            return remaining is not None and remaining <= min_remaining_seconds

        final_lp_bound = None
        best_start_source = "greedy_seed"
        best_start_selected = Counter(self._master_seed_selected)
        best_start_unplaced = Counter(self._master_seed_unplaced)
        last_lp_unplaced: Counter[str] = Counter(self._master_seed_unplaced)
        best_lp_objective: float | None = None
        no_improve_iterations = 0

        stats = {
            "scip_available": True,
            "pricing_iterations": [],
            "pricing_stop_reason": "",
            "pricing_iterations_run": 0,
            "total_time_limit": total_time_limit,
            "total_time_limit_scope": "column_generation_solve_excludes_planner_init_output",
            "staged_repair_reserved_seconds": staged_repair_reserve,
            "staged_repair_reserve_fraction": reserve_fraction,
            "staged_repair_reserve_min_seconds": reserve_min,
            "staged_repair_reserve_max_seconds": reserve_max,
            "pricing_start_min_remaining_seconds": pricing_start_min_remaining,
            "pricing_min_lp_time_limit": pricing_min_lp_time_limit,
            "pricing_iteration_setup_reserve_seconds": pricing_iteration_setup_reserve,
            "pricing_min_iterations": max(0, int(getattr(self.config, "min_pricing_iterations", 0) or 0)),
            "pricing_early_stop_new_columns": max(
                0,
                int(getattr(self.config, "pricing_early_stop_new_columns", 0) or 0),
            ),
            "pricing_max_no_improve_iterations": max(
                0,
                int(getattr(self.config, "pricing_max_no_improve_iterations", 0) or 0),
            ),
            "pricing_min_lp_improvement": max(
                0.0,
                float(getattr(self.config, "pricing_min_lp_improvement", 0.0) or 0.0),
            ),
            "pricing_min_lp_improvement_mode": "adaptive",
            "pricing_min_lp_improvement_relative": max(
                0.0,
                float(getattr(self.config, "pricing_min_lp_improvement_relative", 0.0) or 0.0),
            ),
            "pricing_min_lp_improvement_per_group": max(
                0.0,
                float(getattr(self.config, "pricing_min_lp_improvement_per_group", 0.0) or 0.0),
            ),
            "pricing_min_lp_improvement_per_1000_columns": max(
                0.0,
                float(getattr(self.config, "pricing_min_lp_improvement_per_1000_columns", 0.0) or 0.0),
            ),
            "pricing_no_improve_iterations": 0,
            "feasibility_early_stop_enabled": bool(
                getattr(self.config, "feasibility_early_stop_enabled", False)
            ),
            "feasibility_early_stop_min_iteration": max(
                0,
                int(getattr(self.config, "feasibility_early_stop_min_iteration", 0) or 0),
            ),
            "feasibility_early_stop_checks": [],
            "pricing_light_repair_checks": [],
            "pricing_light_repair_best_source": best_start_source,
            "pricing_light_repair_best_unplaced_boxes": sum(best_start_unplaced.values()),
            "pricing_light_repair_best_objective": round(
                self._selected_solution_energy(best_start_selected, best_start_unplaced),
                6,
            ),
        }
        pricing_iterations = 0 if self.config.full_column_pool else self.config.max_iterations
        for iteration in range(pricing_iterations):
            if total_time_low(pricing_start_min_remaining):
                stats["pricing_stop_reason"] = "total_time_limit"
                break
            iteration_start = perf_counter()
            if self.config.verbose:
                print(
                    f"[column-generation-scip] building LP iter={iteration} columns={len(self._columns)}",
                    flush=True,
                )
            lp_model, lp_vars, lp_constraints = self._build_restricted_master(Model, quicksum, relax=True)
            lp_time_limit = float(self.config.mip_time_limit)
            remaining = remaining_total_time()
            if remaining is not None:
                available_lp_time = remaining - staged_repair_reserve
                if available_lp_time < pricing_min_lp_time_limit:
                    stats["pricing_stop_reason"] = "total_time_limit"
                    stats["pricing_skipped_lp_iteration"] = iteration
                    stats["pricing_skipped_lp_available_seconds"] = round(max(0.0, available_lp_time), 3)
                    self._free_scip_model(lp_model)
                    break
                lp_time_limit = min(lp_time_limit, available_lp_time)
            self._set_scip_param(lp_model, "limits/time", lp_time_limit)
            if self.config.verbose:
                print(f"[column-generation-scip] solving LP iter={iteration} time_limit={lp_time_limit:.1f}s", flush=True)
            lp_model.optimize()
            lp_status = self._scip_status_name(lp_model)
            if lp_status not in {"optimal"}:
                stats["pricing_stop_reason"] = "lp_" + lp_status.replace(" ", "_")
                stats["pricing_interrupted_iteration"] = iteration
                stats["pricing_interrupted_lp_status"] = lp_status
                self._free_scip_model(lp_model)
                break
            lp_objective = self._scip_objective_value(lp_model)
            final_lp_bound = lp_objective
            lp_unplaced = self._scip_unplaced_values(lp_model, lp_vars)
            last_lp_unplaced = Counter(lp_unplaced)
            lp_column_values = self._scip_column_values(lp_model, lp_vars)
            lp_repair_selected, lp_repair_unplaced = self._repair_from_column_priority(
                lp_column_values
            )
            lp_repair_objective = self._selected_solution_energy(lp_repair_selected, lp_repair_unplaced)
            incumbent_rank_before_light_repair = self._solution_rank(best_start_selected, best_start_unplaced)
            light_repair_rank = self._solution_rank(lp_repair_selected, lp_repair_unplaced)
            light_repair_accepted = light_repair_rank < incumbent_rank_before_light_repair
            if self._solution_rank(lp_repair_selected, lp_repair_unplaced) < self._solution_rank(
                best_start_selected,
                best_start_unplaced,
            ):
                best_start_source = f"lp_guided_iter_{iteration}"
                best_start_selected = lp_repair_selected
                best_start_unplaced = lp_repair_unplaced
            effective_min_lp_improvement = self._effective_pricing_min_lp_improvement(
                lp_objective if best_lp_objective is None else best_lp_objective,
                len(self._columns),
            )
            lp_objective_improvement = (
                None
                if best_lp_objective is None
                else best_lp_objective - lp_objective
            )
            lp_improved = best_lp_objective is None or (
                lp_objective_improvement is not None
                and lp_objective_improvement > effective_min_lp_improvement
            )
            if lp_improved:
                best_lp_objective = lp_objective
            if lp_improved or light_repair_accepted:
                no_improve_iterations = 0
            else:
                no_improve_iterations += 1
            stats["pricing_no_improve_iterations"] = no_improve_iterations
            light_repair_record = {
                "iteration": iteration,
                "accepted": light_repair_accepted,
                "unplaced_boxes": sum(lp_repair_unplaced.values()),
                "objective": round(lp_repair_objective, 6),
                "selected_columns": sum(1 for qty in lp_repair_selected.values() if qty > 0),
                "best_source_after_check": best_start_source,
                "best_unplaced_boxes_after_check": sum(best_start_unplaced.values()),
                "best_objective_after_check": round(
                    self._selected_solution_energy(best_start_selected, best_start_unplaced),
                    6,
                ),
                "lp_objective_improvement": (
                    round(lp_objective_improvement, 6)
                    if lp_objective_improvement is not None
                    else None
                ),
                "effective_min_lp_improvement": round(effective_min_lp_improvement, 6),
                "lp_objective_improved": lp_improved,
                "pricing_no_improve_iterations": no_improve_iterations,
            }
            stats["pricing_light_repair_checks"].append(light_repair_record)
            stats["pricing_light_repair_best_source"] = best_start_source
            stats["pricing_light_repair_best_unplaced_boxes"] = sum(best_start_unplaced.values())
            stats["pricing_light_repair_best_objective"] = light_repair_record["best_objective_after_check"]
            pricing_stats = self._price_columns(lp_model, lp_constraints, iteration, lp_unplaced, lp_column_values)
            new_count = int(pricing_stats.get("new_columns", 0) or 0)
            stats["pricing_iterations"].append(
                {
                    "iteration": iteration,
                    "columns": len(self._columns),
                    "lp_objective": lp_objective,
                    "lp_unplaced_boxes": sum(lp_unplaced.values()),
                    "lp_guided_repair_unplaced_boxes": sum(lp_repair_unplaced.values()),
                    "lp_guided_repair_objective": round(lp_repair_objective, 6),
                    "lp_guided_repair_accepted": light_repair_accepted,
                    "lp_guided_repair_columns": sum(1 for qty in lp_repair_selected.values() if qty > 0),
                    "lp_objective_improvement": (
                        round(lp_objective_improvement, 6)
                        if lp_objective_improvement is not None
                        else None
                    ),
                    "effective_min_lp_improvement": round(effective_min_lp_improvement, 6),
                    "lp_objective_improved": lp_improved,
                    "pricing_no_improve_iterations": no_improve_iterations,
                    **pricing_stats,
                }
            )
            if self.config.verbose:
                print(
                    f"[column-generation-scip] iter={iteration} lp={lp_objective:.3f} "
                    f"columns={len(self._columns)} new={new_count} elapsed={perf_counter() - iteration_start:.1f}s",
                    flush=True,
                )
            self._free_scip_model(lp_model)
            stats["pricing_iterations_run"] = iteration + 1
            if total_time_low(max(5.0, staged_repair_reserve)):
                stats["pricing_stop_reason"] = "total_time_limit"
                break
            if (
                stats["feasibility_early_stop_enabled"]
                and iteration + 1 >= int(stats["feasibility_early_stop_min_iteration"])
            ):
                if total_time_low(max(10.0, staged_repair_reserve)):
                    stats["pricing_stop_reason"] = "total_time_limit"
                    stats["feasibility_early_stop_checks"].append(
                        {
                            "iteration": iteration,
                            "skipped_reason": "total_time_limit",
                            "remaining_seconds": round(max(0.0, remaining_total_time() or 0.0), 3),
                        }
                    )
                    stats["pricing_iterations"][-1].update(
                        {
                            "feasibility_early_stop_skipped_reason": "total_time_limit",
                        }
                    )
                    break
                check_start = perf_counter()
                early_source_unplaced = Counter(best_start_unplaced)
                for group_id, qty in lp_unplaced.items():
                    early_source_unplaced[group_id] = max(
                        int(early_source_unplaced.get(group_id, 0)),
                        int(qty),
                    )
                if self.config.verbose:
                    print(
                        "[column-generation-scip] feasibility check "
                        f"iter={iteration} source_unplaced={sum(qty for qty in early_source_unplaced.values() if qty > 0)} "
                        f"columns={len(self._columns)}",
                        flush=True,
                    )
                early_closure_stats = self._stage0_unplaced_column_closure(early_source_unplaced)
                early_repaired_selected, _early_repaired_unplaced = self._repair_selected_solution(best_start_selected)
                early_selected, early_unplaced, early_stage_stats = self._staged_repair_selected_solution(
                    early_repaired_selected,
                    allow_new_columns=True,
                )
                early_unplaced_boxes = sum(early_unplaced.values())
                early_record = {
                    "iteration": iteration,
                    "source_unplaced_boxes": sum(qty for qty in early_source_unplaced.values() if qty > 0),
                    "closure_added_columns": early_closure_stats.get("stage0_closure_added_columns", 0),
                    "closure_hit_limit": early_closure_stats.get("stage0_closure_hit_limit", False),
                    "staged_repair_unplaced_boxes": early_unplaced_boxes,
                    "staged_repair_selected_candidate": early_stage_stats.get("selected_candidate", ""),
                    "staged_repair_seconds": round(perf_counter() - check_start, 3),
                }
                stats["feasibility_early_stop_checks"].append(early_record)
                stats["pricing_iterations"][-1].update(
                    {
                        "feasibility_early_stop_unplaced_boxes": early_unplaced_boxes,
                        "feasibility_early_stop_seconds": early_record["staged_repair_seconds"],
                        "feasibility_early_stop_added_columns": early_record["closure_added_columns"],
                    }
                )
                if self.config.verbose:
                    print(
                        "[column-generation-scip] feasibility check "
                        f"iter={iteration} unplaced={early_unplaced_boxes} "
                        f"added={early_record['closure_added_columns']} "
                        f"elapsed={early_record['staged_repair_seconds']:.1f}s",
                        flush=True,
                    )
                if self._solution_rank(early_selected, early_unplaced) < self._solution_rank(
                    best_start_selected,
                    best_start_unplaced,
                ):
                    best_start_source = f"pricing_iter_{iteration}_feasibility_repair"
                    best_start_selected = early_selected
                    best_start_unplaced = early_unplaced
                if early_unplaced_boxes <= 0 and self._twenty_run_new_violation_count(early_selected) <= 0:
                    stats["pricing_stop_reason"] = "feasibility_repair_zero_unplaced"
                    break
            if new_count == 0:
                stats["pricing_stop_reason"] = "no_new_columns"
                break
            min_iterations = int(stats["pricing_min_iterations"])
            early_stop_new_columns = int(stats["pricing_early_stop_new_columns"])
            if (
                early_stop_new_columns > 0
                and iteration + 1 >= min_iterations
                and new_count < early_stop_new_columns
            ):
                stats["pricing_stop_reason"] = "few_new_columns"
                break
            max_no_improve = int(stats["pricing_max_no_improve_iterations"])
            if (
                max_no_improve > 0
                and iteration + 1 >= min_iterations
                and no_improve_iterations >= max_no_improve
            ):
                stats["pricing_stop_reason"] = "no_lp_or_integer_improvement"
                break
        if not stats["pricing_stop_reason"]:
            stats["pricing_stop_reason"] = "max_iterations"
        stats["column_generation_pricing_elapsed_seconds"] = round(perf_counter() - solve_start, 3)

        post_pricing_repair_start = perf_counter()
        repair_start_source = best_start_source
        repair_start_selected = Counter(best_start_selected)
        repair_start_unplaced = Counter(best_start_unplaced)
        closure_source_unplaced = Counter(repair_start_unplaced)
        for group_id, qty in last_lp_unplaced.items():
            closure_source_unplaced[group_id] = max(
                int(closure_source_unplaced.get(group_id, 0)),
                int(qty),
            )
        closure_stats = self._stage0_unplaced_column_closure(closure_source_unplaced)
        stats.update(closure_stats)

        if self.config.verbose:
            print(
                "[column-generation-scip] final staged repair "
                f"source={repair_start_source} "
                f"input_unplaced={sum(repair_start_unplaced.values())} "
                f"columns={len(self._columns)}",
                flush=True,
            )
        repair_columns_before = len(self._columns)
        greedy_selected, greedy_unplaced = self._repair_selected_solution(
            repair_start_selected,
            allow_new_columns=True,
        )
        repair_deadline = solve_start + total_time_limit if total_time_limit > 0 else None
        secondary_order_min_remaining = max(
            0.0,
            float(getattr(self.config, "staged_repair_secondary_order_min_remaining", 20.0) or 0.0),
        )
        rebuild_first_threshold = max(
            0,
            int(getattr(self.config, "staged_repair_rebuild_first_unplaced_threshold", 500) or 0),
        )
        repair_start_unplaced_boxes = sum(repair_start_unplaced.values())
        staged_stats = {"iterations": [], "candidates": [], "skipped_candidates": [], "stop_reason": "not_run"}
        rebuild_stats = {"iterations": [], "candidates": [], "skipped_candidates": [], "stop_reason": "not_run"}
        staged_selected: Counter[int] | None = None
        staged_unplaced: Counter[str] | None = None
        rebuild_selected: Counter[int] | None = None
        rebuild_unplaced: Counter[str] | None = None
        repair_candidates = [
            ("pricing_incumbent", repair_start_selected, repair_start_unplaced),
            ("greedy_completion", greedy_selected, greedy_unplaced),
        ]
        repair_candidate_order = (
            ["staged_rebuild", "staged_insert"]
            if repair_start_unplaced_boxes >= rebuild_first_threshold
            else ["staged_insert", "staged_rebuild"]
        )
        skipped_repair_candidates: list[dict] = []

        def zero_unplaced_candidate_exists() -> bool:
            return any(
                sum(candidate_unplaced.values()) <= 0
                and self._twenty_run_new_violation_count(candidate_selected) <= 0
                for _label, candidate_selected, candidate_unplaced in repair_candidates
            )

        def run_staged_candidate(label: str) -> None:
            nonlocal staged_selected, staged_unplaced, staged_stats, rebuild_selected, rebuild_unplaced, rebuild_stats
            candidate_start = perf_counter()
            if label == "staged_rebuild":
                selected, unplaced, candidate_stats = self._staged_repair_selected_solution(
                    Counter(),
                    allow_new_columns=True,
                    stop_after_zero=True,
                    deadline=repair_deadline,
                    min_remaining_for_secondary=secondary_order_min_remaining,
                )
                candidate_stats["elapsed_seconds"] = round(perf_counter() - candidate_start, 3)
                rebuild_selected = selected
                rebuild_unplaced = unplaced
                rebuild_stats = candidate_stats
            else:
                selected, unplaced, candidate_stats = self._staged_repair_selected_solution(
                    greedy_selected,
                    allow_new_columns=True,
                    stop_after_zero=True,
                    deadline=repair_deadline,
                    min_remaining_for_secondary=secondary_order_min_remaining,
                )
                candidate_stats["elapsed_seconds"] = round(perf_counter() - candidate_start, 3)
                staged_selected = selected
                staged_unplaced = unplaced
                staged_stats = candidate_stats
            repair_candidates.append((label, selected, unplaced))

        for candidate_label in repair_candidate_order:
            if zero_unplaced_candidate_exists():
                skipped_repair_candidates.append(
                    {
                        "candidate": candidate_label,
                        "skipped_reason": "zero_unplaced_candidate_already_found",
                    }
                )
                continue
            run_staged_candidate(candidate_label)

        selected_method, best_start_selected, best_start_unplaced = min(
            repair_candidates,
            key=lambda item: self._solution_rank(item[1], item[2]),
        )
        if selected_method != "pricing_incumbent":
            best_start_source = f"{repair_start_source}_{selected_method}"

        final_unplaced_boxes = sum(best_start_unplaced.values())
        objective = self._selected_solution_energy(best_start_selected, best_start_unplaced)
        final_status = (
            "staged_repair_zero_unplaced"
            if final_unplaced_boxes <= 0
            else "staged_repair_unproven_unplaced"
        )
        self._master_start_selected = best_start_selected
        self._master_start_unplaced = best_start_unplaced
        stats.update(
            {
                "final_staged_repair_enabled": True,
                "final_staged_repair_input_source": repair_start_source,
                "final_staged_repair_input_unplaced_boxes": repair_start_unplaced_boxes,
                "final_staged_repair_added_columns": len(self._columns) - repair_columns_before,
                "final_staged_repair_selected_candidate": selected_method,
                "final_staged_repair_unplaced_boxes": final_unplaced_boxes,
                "final_staged_repair_status": final_status,
                "final_staged_repair_candidate_order": repair_candidate_order,
                "final_staged_repair_skipped_candidates": skipped_repair_candidates,
                "final_staged_repair_secondary_order_min_remaining_seconds": secondary_order_min_remaining,
                "final_staged_repair_rebuild_first_unplaced_threshold": rebuild_first_threshold,
                "final_staged_repair_iterations": (
                    rebuild_stats.get("iterations", [])
                    if selected_method == "staged_rebuild"
                    else staged_stats.get("iterations", [])
                ),
                "final_staged_repair_order_candidates": (
                    rebuild_stats.get("candidates", [])
                    if selected_method == "staged_rebuild"
                    else staged_stats.get("candidates", [])
                ),
                "final_staged_insert_iterations": staged_stats.get("iterations", []),
                "final_staged_insert_order_candidates": staged_stats.get("candidates", []),
                "final_staged_insert_skipped_order_candidates": staged_stats.get("skipped_candidates", []),
                "final_staged_insert_stop_reason": staged_stats.get("stop_reason", ""),
                "final_staged_insert_elapsed_seconds": staged_stats.get("elapsed_seconds", 0.0),
                "final_staged_rebuild_iterations": rebuild_stats.get("iterations", []),
                "final_staged_rebuild_order_candidates": rebuild_stats.get("candidates", []),
                "final_staged_rebuild_skipped_order_candidates": rebuild_stats.get("skipped_candidates", []),
                "final_staged_rebuild_stop_reason": rebuild_stats.get("stop_reason", ""),
                "final_staged_rebuild_elapsed_seconds": rebuild_stats.get("elapsed_seconds", 0.0),
                "final_staged_repair_candidates": [
                    {
                        "candidate": label,
                        "document_unplaced_boxes": self._document_unplaced_boxes(candidate_unplaced),
                        "unplaced_boxes": sum(candidate_unplaced.values()),
                        "twenty_consecutive_violation_windows": self._twenty_run_violation_count(candidate_selected),
                        "objective": round(self._selected_solution_energy(candidate_selected, candidate_unplaced), 6),
                        "selected_columns": sum(1 for qty in candidate_selected.values() if qty > 0),
                    }
                    for label, candidate_selected, candidate_unplaced in repair_candidates
                ],
                "feasibility_repair_source": best_start_source,
                "feasibility_repair_unplaced_boxes": final_unplaced_boxes,
                "feasibility_repair_columns": sum(1 for qty in best_start_selected.values() if qty > 0),
                "master_algorithm": "column_generation_staged_repair",
                "master_status": final_status,
                "master_objective": objective,
                "master_primal_bound": objective,
                "master_dual_bound": final_lp_bound,
                "master_mip_gap": self._relative_gap(objective, final_lp_bound),
                "master_mip_gap_is_reliable": False,
                "restricted_master_lp_bound": final_lp_bound,
                "restricted_master_lp_unplaced_boxes": sum(last_lp_unplaced.values()),
                "restricted_master_lp_fractional_column_count": None,
                "restricted_master_lp_solve_seconds": 0.0,
                "master_mip_start_source": best_start_source,
                "master_mip_start_repaired_columns": sum(1 for qty in best_start_selected.values() if qty > 0),
                "master_mip_start_repaired_unplaced_boxes": final_unplaced_boxes,
                "master_solve_seconds": round(perf_counter() - solve_start, 3),
                "post_pricing_repair_elapsed_seconds": round(perf_counter() - post_pricing_repair_start, 3),
            }
        )
        if self.config.verbose:
            print(
                "[column-generation-scip] final staged repair "
                f"status={final_status} unplaced={final_unplaced_boxes} "
                f"elapsed={stats['post_pricing_repair_elapsed_seconds']:.1f}s",
                flush=True,
            )
        return best_start_selected, best_start_unplaced, stats

    def _stage0_unplaced_column_closure(self, unplaced: Counter[str]) -> dict:
        stats = {
            "stage0_closure_enabled": bool(getattr(self.config, "stage0_closure_enabled", False)),
            "stage0_closure_source_unplaced_boxes": sum(qty for qty in unplaced.values() if qty > 0),
            "stage0_closure_group_count": sum(1 for qty in unplaced.values() if qty > 0),
            "stage0_closure_added_columns": 0,
            "stage0_closure_hit_limit": False,
        }
        if not stats["stage0_closure_enabled"] or not unplaced:
            return stats
        limit = max(0, int(getattr(self.config, "stage0_closure_max_extra_columns", 0) or 0))
        added = 0
        groups = [
            self.groups_by_id[group_id]
            for group_id, qty in sorted(unplaced.items(), key=lambda item: (-int(item[1]), item[0]))
            if qty > 0 and group_id in self.groups_by_id
        ]
        for group in groups:
            unplaced_qty = max(1, int(unplaced.get(group.group_id, 0)))
            for bay_key, max_qty, base_cost in self._candidate_bays_for_group(group, scope="stage0"):
                for qty in self._stage0_closure_quantity_options(group, max_qty, unplaced_qty):
                    if self._add_column(group, bay_key, qty, base_cost):
                        added += 1
                        if limit > 0 and added >= limit:
                            stats["stage0_closure_added_columns"] = added
                            stats["stage0_closure_hit_limit"] = True
                            return stats
        stats["stage0_closure_added_columns"] = added
        return stats

    def _stage0_closure_quantity_options(self, group: SmallBoxGroup, max_qty: int, unplaced_qty: int) -> list[int]:
        max_qty = max(0, min(int(max_qty), int(group.demand)))
        if max_qty <= 0:
            return []
        values = set(self._quantity_options(group, max_qty))
        values.update(
            qty
            for qty in (
                1,
                2,
                3,
                4,
                5,
                max_qty // 3,
                max_qty // 2,
                (2 * max_qty) // 3,
                max_qty,
                min(unplaced_qty, max_qty),
                unplaced_qty % max_qty if max_qty > 0 else 0,
            )
            if qty > 0
        )
        if max_qty <= 16:
            values.update(range(1, max_qty + 1))
        else:
            values.update(range(1, min(10, max_qty) + 1))
            values.update(qty for qty in range(15, max_qty + 1, 5))
        return sorted(qty for qty in values if 0 < qty <= max_qty)

    def _scip_unplaced_values(self, model, lp_vars) -> Counter[str]:
        return Counter(
            {
                group_id: int(round(self._scip_value(model, var)))
                for group_id, var in lp_vars["unplaced"].items()
                if self._scip_value(model, var) > 1e-6
            }
        )

    def _scip_column_values(self, model, lp_vars) -> dict[int, float]:
        return {
            idx: self._scip_value(model, var)
            for idx, var in lp_vars["column"].items()
            if self._scip_value(model, var) > 1e-6
        }

    def _configure_scip_output(self, model) -> None:
        self._configure_scip_stability(model)
        if not self.config.verbose:
            try:
                model.hideOutput()
                return
            except Exception:
                pass
            self._try_set_scip_param(model, "display/verblevel", 0)

    def _configure_scip_stability(self, model) -> None:
        if not getattr(self.config, "scip_disable_symmetry", True):
            return
        for name, value in (
            ("misc/usesymmetry", 0),
            ("propagating/symmetry/maxgenerators", 0),
            ("propagating/symmetry/symtiming", 0),
            ("propagating/symmetry/ofsymcomptiming", 0),
        ):
            self._try_set_scip_param(model, name, value)

    @staticmethod
    def _try_set_scip_param(model, name: str, value: object) -> bool:
        try:
            ColumnGenerationPlanner._set_scip_param(model, name, value)
            return True
        except Exception:
            return False

    @staticmethod
    def _set_scip_param(model, name: str, value: object) -> None:
        setters = (getattr(model, "setParam", None), getattr(model, "setRealParam", None), getattr(model, "setIntParam", None))
        last_error: Exception | None = None
        for setter in setters:
            if setter is None:
                continue
            try:
                setter(name, value)
                return
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error

    @staticmethod
    def _scip_status_name(model) -> str:
        return str(model.getStatus()).lower()

    @staticmethod
    def _scip_solution_count(model) -> int:
        try:
            return int(model.getNSols())
        except Exception:
            try:
                return 1 if model.getBestSol() is not None else 0
            except Exception:
                return 0

    @staticmethod
    def _scip_objective_value(model) -> float:
        try:
            return float(model.getObjVal())
        except Exception:
            return float("nan")

    @staticmethod
    def _scip_gap(model) -> float:
        try:
            return float(model.getGap())
        except Exception:
            return 0.0

    @staticmethod
    def _scip_primal_bound(model) -> float:
        for method_name in ("getPrimalbound", "getPrimalBound"):
            method = getattr(model, method_name, None)
            if method is None:
                continue
            try:
                return float(method())
            except Exception:
                continue
        return float("nan")

    @staticmethod
    def _scip_dual_bound(model) -> float:
        for method_name in ("getDualbound", "getDualBound"):
            method = getattr(model, method_name, None)
            if method is None:
                continue
            try:
                return float(method())
            except Exception:
                continue
        return float("nan")

    @staticmethod
    def _relative_gap(primal: float | None, bound: float | None) -> float | None:
        if primal is None or bound is None:
            return None
        try:
            primal_value = float(primal)
            bound_value = float(bound)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(primal_value) or not math.isfinite(bound_value):
            return None
        return max(0.0, (primal_value - bound_value) / max(1.0, abs(primal_value)))

    @staticmethod
    def _scip_value(model, var) -> float:
        return float(model.getVal(var))

    @staticmethod
    def _scip_dual(model, constr) -> float:
        for method_name in ("getDualsolLinear", "getDualsol"):
            method = getattr(model, method_name, None)
            if method is None:
                continue
            try:
                return float(method(constr))
            except Exception:
                continue
        return 0.0

    @staticmethod
    def _free_scip_model(model) -> None:
        for method_name in ("freeTransform", "freeProb"):
            method = getattr(model, method_name, None)
            if method is None:
                continue
            try:
                method()
                return
            except Exception:
                continue

    def _effective_pricing_min_lp_improvement(
        self,
        lp_reference: float | None,
        column_count: int,
    ) -> float:
        values = [
            max(0.0, float(getattr(self.config, "pricing_min_lp_improvement", 0.0) or 0.0)),
        ]
        relative = max(
            0.0,
            float(getattr(self.config, "pricing_min_lp_improvement_relative", 0.0) or 0.0),
        )
        if lp_reference is not None and math.isfinite(float(lp_reference)):
            values.append(abs(float(lp_reference)) * relative)
        per_group = max(
            0.0,
            float(getattr(self.config, "pricing_min_lp_improvement_per_group", 0.0) or 0.0),
        )
        if per_group > 0:
            values.append(len(self.groups) * per_group)
        per_1000_columns = max(
            0.0,
            float(getattr(self.config, "pricing_min_lp_improvement_per_1000_columns", 0.0) or 0.0),
        )
        if per_1000_columns > 0:
            values.append(max(0, int(column_count)) / 1000.0 * per_1000_columns)
        return max(values)

    def _build_restricted_master(
        self,
        Model,
        quicksum,
        relax: bool,
        objective_mode: str = "full",
    ):
        model = Model("yard_small_plan_column_generation_scip")
        self._configure_scip_output(model)
        try:
            model.setMinimize()
        except Exception:
            pass
        column_vtype = "C" if relax else "B"
        columns = {
            idx: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=column_vtype,
                obj=0.0 if objective_mode == "min_unplaced" else col.intrinsic_cost + self.config.small_plan_group_bay_split_penalty,
                name=f"col_{idx}",
            )
            for idx, col in enumerate(self._columns)
        }
        unplaced = {
            group.group_id: model.addVar(
                lb=0.0,
                ub=group.demand,
                vtype="C" if relax else "I",
                obj=self._unplaced_objective_for_group(group, objective_mode),
                name=f"unplaced_{group.group_id}",
            )
            for group in self.groups
        }

        group_cols: defaultdict[str, list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_capacity_cols: defaultdict[str, list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_size_capacity_cols: defaultdict[tuple[str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_port_size_cols: defaultdict[tuple[str, str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        row_capacity_cols: defaultdict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        row_size_capacity_cols: defaultdict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
        row_attr_choice_cols: defaultdict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
        group_bay_cols: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        bay_attr_choice_cols: defaultdict[tuple[str, str, str, str], list[int]] = defaultdict(list)
        coarse_area_cols: defaultdict[tuple[str, ...], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        medium_coarse_area_cols: defaultdict[tuple[str, ...], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        medium_coarse_area_bay_cols: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
        area_size_cols: defaultdict[tuple[str, str, str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        group_area_cols: defaultdict[tuple[tuple[str, ...], str], list[int]] = defaultdict(list)
        group_block_cols: defaultdict[tuple[tuple[str, ...], str], list[int]] = defaultdict(list)
        coarse_area_block_cols: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
        coarse_area_bay_cols: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
        voyage_area_cols: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        export_e_area_bay_cols: defaultdict[tuple[str, str, str], list[int]] = defaultdict(list)
        twenty_cols_by_bay: defaultdict[str, list[int]] = defaultdict(list)
        edge_45_cols: defaultdict[str, list[int]] = defaultdict(list)
        edge_non45_cols: defaultdict[str, list[int]] = defaultdict(list)
        for idx, col in enumerate(self._columns):
            group_cols[col.group_id].append((idx, col))
            for footprint_key in self._placement_footprint_keys(col.bay_key, col.size):
                bay_capacity_cols[footprint_key].append((idx, col))
                bay_port_size_cols[(footprint_key, self._row_mix_key_for_column(col), col.size)].append((idx, col))
                for attr in self._bay_no_mix_attrs_for_column(col):
                    scope = self._attr_voyage_scope(attr, col.voyage_id)
                    bay_attr_choice_cols[(footprint_key, attr, scope, self._column_attr_value(col, attr))].append(idx)
            for footprint_key, row_no, qty in col.row_allocation:
                row_capacity_cols[(footprint_key, row_no)].append((idx, int(qty)))
                row_size_capacity_cols[(footprint_key, row_no, col.size)].append((idx, int(qty)))
                for attr in self._row_no_mix_attrs_for_column(col):
                    scope = self._attr_voyage_scope(attr, col.voyage_id)
                    row_attr_choice_cols[(footprint_key, row_no, attr, scope, self._column_attr_value(col, attr))].append(idx)
            bay_size_capacity_cols[(col.bay_key, col.size)].append((idx, col))
            group_bay_cols[(col.group_id, col.bay_key)].append(idx)
            coarse_area_cols[col.coarse_cluster_key + (col.area_no,)].append((idx, col))
            medium_coarse_area_cols[col.coarse_key + (col.area_no,)].append((idx, col))
            medium_coarse_area_bay_cols[col.coarse_key + (col.area_no, col.bay_key)].append(idx)
            area_size_cols[col.quota_key].append((idx, col))
            group_area_cols[(col.fine_cluster_key, col.area_no)].append(idx)
            if col.block_id:
                group_block_cols[(col.fine_cluster_key, col.block_id)].append(idx)
                coarse_area_block_cols[col.coarse_cluster_key + (col.area_no, col.block_id)].append(idx)
            coarse_area_bay_cols[col.coarse_cluster_key + (col.area_no, col.bay_key)].append(idx)
            voyage_area_cols[(col.voyage_id, col.area_no)].append(idx)
            if self._is_export_e_column(col):
                export_e_area_bay_cols[(col.voyage_id, col.area_no, col.bay_key)].append(idx)
            if col.size == "20":
                twenty_cols_by_bay[col.bay_key].append(idx)
            if col.bay_key in self.area_edge_bays.get(col.area_no, set()):
                if col.size == "45":
                    edge_45_cols[col.area_no].append(idx)
                else:
                    edge_non45_cols[col.area_no].append(idx)

        group_cover = {}
        for group in self.groups:
            expr = quicksum(col.quantity * columns[idx] for idx, col in group_cols.get(group.group_id, []))
            group_cover[group.group_id] = model.addCons(expr + unplaced[group.group_id] == group.demand, name=f"cover_{group.group_id}")

        required_area_limit = {}
        for voyage_id, areas in sorted(getattr(self.problem, "user_voyage_area_requirements", {}).items()):
            for area_no in sorted(areas):
                indices = voyage_area_cols.get((voyage_id, area_no), [])
                if not indices:
                    continue
                required_area_limit[(voyage_id, area_no)] = model.addCons(
                    quicksum(self._columns[idx].quantity * columns[idx] for idx in indices) >= 1.0,
                    name=f"user_required_area_{len(required_area_limit)}",
                )

        required_group_bay_limit = {}
        for group_id, bay_keys in sorted(getattr(self.problem, "user_group_bay_requirements", {}).items()):
            for bay_key in sorted(bay_keys):
                indices = group_bay_cols.get((group_id, bay_key), [])
                required_group_bay_limit[(group_id, bay_key)] = model.addCons(
                    quicksum(self._columns[idx].quantity * columns[idx] for idx in indices) >= 1.0,
                    name=f"user_required_bay_{len(required_group_bay_limit)}",
                )

        bay_capacity_limit = {}
        for bay_key, items in bay_capacity_cols.items():
            bay_capacity_limit[bay_key] = model.addCons(
                quicksum(col.quantity * columns[idx] for idx, col in items) <= self.bays[bay_key].physical_capacity,
                name=f"bay_cap_{bay_key}",
            )
        bay_size_limit = {}
        for key, items in bay_size_capacity_cols.items():
            bay_key, size = key
            bay_size_limit[key] = model.addCons(
                quicksum(col.quantity * columns[idx] for idx, col in items) <= self.bays[bay_key].cap_by_size.get(size, 0),
                name=f"bay_size_{bay_key}_{size}",
            )
        row_capacity_limit = {}
        for key, items in row_capacity_cols.items():
            bay_key, row_no = key
            cap = int(self.bays[bay_key].row_physical_capacity.get(row_no, self.bays[bay_key].physical_capacity))
            row_capacity_limit[key] = model.addCons(
                quicksum(qty * columns[idx] for idx, qty in items) <= cap,
                name=f"row_cap_{bay_key}_{row_no}",
            )
        row_size_limit = {}
        for key, items in row_size_capacity_cols.items():
            bay_key, row_no, size = key
            cap = int(self.bays[bay_key].row_cap_by_size.get(size, {}).get(row_no, self.bays[bay_key].cap_by_size.get(size, 0)))
            row_size_limit[key] = model.addCons(
                quicksum(qty * columns[idx] for idx, qty in items) <= cap,
                name=f"row_size_{bay_key}_{row_no}_{size}",
            )
        bay_port_stack_link = {}
        bay_port_stack_limit = {}
        bay_stack_total_limit = {}
        bay_stack_vars = {}
        stack_vtype = "C" if relax else "I"
        for key, items in sorted(bay_port_size_cols.items()):
            bay_key, port, size = key
            sample_group = self.groups_by_id.get(items[0][1].group_id) if items else None
            if sample_group is None:
                continue
            stack_count = self._stack_count_for_group(bay_key, size, sample_group)
            unit_capacity = self._stack_unit_capacity_for_group(bay_key, size, sample_group)
            if stack_count <= 0 or unit_capacity <= 0:
                continue
            stack_var = model.addVar(lb=0.0, ub=stack_count, vtype=stack_vtype, name=f"stack_{bay_key}_{port}_{size}")
            bay_stack_vars[key] = stack_var
            load = quicksum(col.quantity * columns[idx] for idx, col in items)
            bay_port_stack_link[key] = model.addCons(load <= unit_capacity * stack_var, name=f"stack_load_{bay_key}_{port}_{size}")
            bay_port_stack_limit[key] = model.addCons(stack_var <= stack_count, name=f"stack_port_cap_{bay_key}_{port}_{size}")
        stack_vars_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        for (bay_key, _port, size), stack_var in bay_stack_vars.items():
            stack_vars_by_bay_size[(bay_key, size)].append(stack_var)
        for (bay_key, size), stack_vars in stack_vars_by_bay_size.items():
            stack_count = self._stack_count_for_bay_size(bay_key, size)
            if stack_count > 0:
                bay_stack_total_limit[(bay_key, size)] = model.addCons(
                    quicksum(stack_vars) <= stack_count,
                    name=f"stack_total_{bay_key}_{size}",
                )
        group_bay_limit = {
            key: model.addCons(quicksum(columns[idx] for idx in indices) <= 1.0, name=f"group_bay_{key[0]}_{key[1]}")
            for key, indices in group_bay_cols.items()
        }
        export_e_area_bay_limit = self._add_export_e_area_bay_limits(
            quicksum,
            model,
            columns,
            export_e_area_bay_cols,
            relax=relax,
        )
        twenty_run_limits = self._add_twenty_run_limits(
            quicksum,
            model,
            columns,
            twenty_cols_by_bay,
            relax=relax,
            objective_mode=objective_mode,
        )
        quota_limit = {}
        medium_plan_quota_limit = {}
        for key, items in area_size_cols.items():
            cap = int(self.quota_by_key.get(key, 0))
            if cap <= 0:
                continue
            quota_limit[key] = model.addCons(
                quicksum(col.quantity * columns[idx] for idx, col in items) <= cap,
                name=f"big_quota_{len(quota_limit)}",
            )
        if self.config.medium_plan_quota is not None:
            for key, items in medium_coarse_area_cols.items():
                cap = int(self._medium_plan_quota.get(key, 0))
                medium_plan_quota_limit[key] = model.addCons(
                    quicksum(col.quantity * columns[idx] for idx, col in items) <= cap,
                    name=f"medium_quota_{len(medium_plan_quota_limit)}",
                )
        if self.config.medium_plan_bay_quota is not None:
            for key, indices in medium_coarse_area_bay_cols.items():
                cap = int(self._medium_plan_bay_quota.get(key, 0))
                medium_plan_quota_limit[key] = model.addCons(
                    quicksum(self._columns[idx].quantity * columns[idx] for idx in indices) <= cap,
                    name=f"medium_bay_quota_{len(medium_plan_quota_limit)}",
                )

        seed_unplaced_limit = None
        if not relax and (self._master_seed_selected or self._master_seed_unplaced):
            seed_unplaced_limit = model.addCons(
                quicksum(unplaced.values()) <= sum(self._master_seed_unplaced.values()),
                name="seed_unplaced_cap",
            )

        relaxed_objective_constraints = {}
        if objective_mode == "min_unplaced":
            pass
        elif relax:
            relaxed_objective_constraints = self._add_relaxed_master_objectives(
                quicksum,
                model,
                columns,
                area_size_cols,
                group_area_cols,
                group_block_cols,
                coarse_area_block_cols,
                coarse_area_bay_cols,
                voyage_area_cols,
            )
        else:
            self._add_integer_master_objectives(
                quicksum,
                model,
                columns,
                coarse_area_cols,
                area_size_cols,
                group_area_cols,
                group_block_cols,
                coarse_area_block_cols,
                coarse_area_bay_cols,
                voyage_area_cols,
                edge_45_cols,
                edge_non45_cols,
                bay_attr_choice_cols,
                row_attr_choice_cols,
            )
        return model, {"column": columns, "unplaced": unplaced}, {
            "group_cover": group_cover,
            "bay_capacity_limit": bay_capacity_limit,
            "bay_size_limit": bay_size_limit,
            "row_capacity_limit": row_capacity_limit,
            "row_size_limit": row_size_limit,
            "twenty_run_bay_use_link": twenty_run_limits.get("bay_use_link", {}),
            "bay_port_stack_link": bay_port_stack_link,
            "bay_port_stack_limit": bay_port_stack_limit,
            "bay_stack_total_limit": bay_stack_total_limit,
            "group_bay_limit": group_bay_limit,
            "export_e_area_bay_limit": export_e_area_bay_limit,
            "quota_limit": quota_limit,
            "medium_plan_quota_limit": medium_plan_quota_limit,
            "required_area_limit": required_area_limit,
            "required_group_bay_limit": required_group_bay_limit,
            "seed_unplaced_limit": seed_unplaced_limit,
            **relaxed_objective_constraints,
        }

    def _add_export_e_area_bay_limits(
        self,
        quicksum,
        model,
        columns,
        export_e_area_bay_cols: dict[tuple[str, str, str], list[int]],
        relax: bool,
    ) -> dict[tuple, object]:
        limit = self._export_e_area_max_bays()
        if limit is None or not export_e_area_bay_cols:
            return {}
        constraints: dict[tuple, object] = {}
        use_vtype = "C" if relax else "B"
        use_by_voyage_area: defaultdict[tuple[str, str], list] = defaultdict(list)
        for (voyage_id, area_no, bay_key), indices in sorted(export_e_area_bay_cols.items()):
            if not indices:
                continue
            use = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=use_vtype,
                name=f"use_export_e_{voyage_id}_{area_no}_{bay_key}",
            )
            constraints[("bay_use", voyage_id, area_no, bay_key)] = model.addCons(
                quicksum(columns[idx] for idx in indices) <= len(indices) * use,
                name=f"export_e_bay_use_{voyage_id}_{area_no}_{bay_key}",
            )
            use_by_voyage_area[(voyage_id, area_no)].append(use)
        for (voyage_id, area_no), use_vars in sorted(use_by_voyage_area.items()):
            constraints[("area_limit", voyage_id, area_no)] = model.addCons(
                quicksum(use_vars) <= limit,
                name=f"export_e_area_bay_limit_{voyage_id}_{area_no}",
            )
        return constraints

    def _add_twenty_run_limits(
        self,
        quicksum,
        model,
        columns,
        twenty_cols_by_bay: dict[str, list[int]],
        relax: bool,
        objective_mode: str,
    ) -> dict[str, dict]:
        fixed_window_count = sum(
            1
            for window in self.twenty_run_windows
            if all(bay_key in self.existing_twenty_bays for bay_key in window)
        )
        active_windows = [
            window
            for window in self.twenty_run_windows
            if not all(bay_key in self.existing_twenty_bays for bay_key in window)
        ]
        penalty = 0.0 if objective_mode == "min_unplaced" else self._twenty_consecutive_violation_penalty()
        if fixed_window_count > 0 and penalty > 0:
            model.addVar(
                lb=float(fixed_window_count),
                ub=float(fixed_window_count),
                obj=penalty,
                name="use20_existing_violation_constant",
            )
        if not active_windows:
            return {"bay_use_link": {}, "window_limit": {}}
        use_vtype = "C" if relax else "B"
        violation_vtype = "C" if relax else "B"
        relevant_bays = sorted({bay_key for window in active_windows for bay_key in window})
        use_by_bay = {}
        bay_use_link = {}
        for bay_key in relevant_bays:
            existing = bay_key in self.existing_twenty_bays
            indices = twenty_cols_by_bay.get(bay_key, [])
            use = model.addVar(
                lb=1.0 if existing else 0.0,
                ub=1.0 if existing or indices else 0.0,
                vtype=use_vtype,
                name=f"use20_{bay_key}",
            )
            use_by_bay[bay_key] = use
            if indices:
                bay_use_link[bay_key] = model.addCons(
                    quicksum(columns[idx] for idx in indices) <= len(indices) * use,
                    name=f"use20_link_{bay_key}",
                )

        window_limit = {}
        for window_index, window in enumerate(active_windows):
            violation = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=violation_vtype,
                obj=penalty,
                name=f"use20_violation_{window_index}",
            )
            window_limit[window] = model.addCons(
                quicksum(use_by_bay[bay_key] for bay_key in window)
                <= len(window) - 1 + violation,
                name=f"use20_window_{window_index}",
            )
        return {"bay_use_link": bay_use_link, "window_limit": window_limit}

    def _add_relaxed_master_objectives(
        self,
        quicksum,
        model,
        columns,
        area_size_cols,
        group_area_cols,
        group_block_cols,
        coarse_area_block_cols,
        coarse_area_bay_cols,
        voyage_area_cols,
    ) -> dict[str, dict]:
        area_size_keys = set(area_size_cols)
        for key, qty in self.quota_by_key.items():
            voyage_id, flow, _area_no, big_size = key
            if qty > 0 and self.voyage_flow_size_demand[(voyage_id, flow, big_size)] > 0:
                area_size_keys.add(key)

        big_plan_deviation_balance = {}
        for key in sorted(area_size_keys):
            items = area_size_cols.get(key, [])
            voyage_id, flow, area_no, big_size = key
            target = self._area_size_target(voyage_id, flow, area_no, big_size)
            pos = model.addVar(lb=0.0, obj=self.config.big_plan_area_deviation_penalty, name=f"lp_big_pos_{len(model.getVars())}")
            neg = model.addVar(lb=0.0, obj=self.config.big_plan_area_deviation_penalty, name=f"lp_big_neg_{len(model.getVars())}")
            actual = quicksum(col.quantity * columns[idx] for idx, col in items)
            big_plan_deviation_balance[key] = model.addCons(actual - target == pos - neg)

        fixed_use_constraints = {}
        for key, indices in group_area_cols.items():
            use = model.addVar(lb=0.0, ub=1.0, obj=self.config.small_plan_group_area_split_penalty)
            fixed_use_constraints[("group_area",) + key] = model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for key, indices in group_block_cols.items():
            use = model.addVar(lb=0.0, ub=1.0, obj=self.config.small_plan_group_block_split_penalty)
            fixed_use_constraints[("group_block",) + key] = model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for key, indices in coarse_area_block_cols.items():
            use = model.addVar(lb=0.0, ub=1.0, obj=self.config.small_plan_coarse_area_block_split_penalty)
            fixed_use_constraints[("coarse_area_block",) + key] = model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for key, indices in coarse_area_bay_cols.items():
            use = model.addVar(lb=0.0, ub=1.0, obj=self.config.small_plan_coarse_area_bay_split_penalty)
            fixed_use_constraints[("coarse_area_bay",) + key] = model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for (voyage_id, area_no), indices in voyage_area_cols.items():
            cost = self._voyage_area_cost(voyage_id, area_no)
            if abs(cost) <= 1e-9:
                continue
            use = model.addVar(lb=0.0, ub=1.0, obj=cost)
            total = quicksum(columns[idx] for idx in indices)
            fixed_use_constraints[("voyage_area", voyage_id, area_no)] = model.addCons(total <= len(indices) * use)
            if cost < 0:
                fixed_use_constraints[("voyage_area_reward", voyage_id, area_no)] = model.addCons(use <= total)
        return {
            "big_plan_deviation_balance": big_plan_deviation_balance,
            "fixed_use_objective_limit": fixed_use_constraints,
        }

    def _add_integer_master_objectives(
        self,
        quicksum,
        model,
        columns,
        coarse_area_cols,
        area_size_cols,
        group_area_cols,
        group_block_cols,
        coarse_area_block_cols,
        coarse_area_bay_cols,
        voyage_area_cols,
        edge_45_cols,
        edge_non45_cols,
        bay_attr_choice_cols,
        row_attr_choice_cols,
    ) -> None:
        coarse_area_keys = set(coarse_area_cols)
        self._add_coarse_group_area_objectives(quicksum, model, columns, coarse_area_keys, coarse_area_cols)

        area_size_keys = set(area_size_cols)
        for key, qty in self.quota_by_key.items():
            voyage_id, flow, _area_no, big_size = key
            if qty > 0 and self.voyage_flow_size_demand[(voyage_id, flow, big_size)] > 0:
                area_size_keys.add(key)
        for key in sorted(area_size_keys):
            items = area_size_cols.get(key, [])
            voyage_id, flow, area_no, big_size = key
            target = self._area_size_target(voyage_id, flow, area_no, big_size)
            pos = model.addVar(lb=0.0, obj=self.config.big_plan_area_deviation_penalty, name=f"big_pos_{len(model.getVars())}")
            neg = model.addVar(lb=0.0, obj=self.config.big_plan_area_deviation_penalty, name=f"big_neg_{len(model.getVars())}")
            actual = quicksum(col.quantity * columns[idx] for idx, col in items)
            model.addCons(actual - target == pos - neg)

        for (fine_key, area_no), indices in group_area_cols.items():
            use = model.addVar(vtype="B", obj=self.config.small_plan_group_area_split_penalty, name=f"use_ga_{self._key_name(fine_key)}_{area_no}")
            model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for (fine_key, block_id), indices in group_block_cols.items():
            use = model.addVar(vtype="B", obj=self.config.small_plan_group_block_split_penalty, name=f"use_gb_{self._key_name(fine_key)}_{block_id}")
            model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for key, indices in coarse_area_block_cols.items():
            coarse_key = tuple(key[:-2])
            area_no, block_id = key[-2], key[-1]
            name_key = self._key_name(coarse_key)
            use = model.addVar(
                vtype="B",
                obj=self.config.small_plan_coarse_area_block_split_penalty,
                name=f"use_cab_{name_key}_{area_no}_{block_id}",
            )
            model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for key, indices in coarse_area_bay_cols.items():
            coarse_key = tuple(key[:-2])
            area_no, bay_key = key[-2], key[-1]
            name_key = self._key_name(coarse_key)
            use = model.addVar(
                vtype="B",
                obj=self.config.small_plan_coarse_area_bay_split_penalty,
                name=f"use_cay_{name_key}_{area_no}_{bay_key}",
            )
            model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)

        for (voyage_id, area_no), indices in voyage_area_cols.items():
            cost = self._voyage_area_cost(voyage_id, area_no)
            use = model.addVar(vtype="B", obj=cost, name=f"use_va_{voyage_id}_{area_no}")
            total = quicksum(columns[idx] for idx in indices)
            model.addCons(total <= len(indices) * use)
            if cost < 0:
                model.addCons(use <= total)

        for area_no in set(edge_45_cols) | set(edge_non45_cols):
            has45 = model.addVar(vtype="B", name=f"area_has45_{area_no}")
            if edge_45_cols.get(area_no):
                model.addCons(quicksum(columns[idx] for idx in edge_45_cols[area_no]) <= len(edge_45_cols[area_no]) * has45)
            if edge_non45_cols.get(area_no):
                model.addCons(quicksum(columns[idx] for idx in edge_non45_cols[area_no]) <= len(edge_non45_cols[area_no]) * (1 - has45))

        self._add_bay_compatibility_constraints(
            quicksum,
            model,
            columns,
            bay_attr_choice_cols,
        )
        self._add_row_compatibility_constraints(
            quicksum,
            model,
            columns,
            row_attr_choice_cols,
        )

    def _add_coarse_group_area_objectives(
        self,
        quicksum,
        model,
        columns,
        coarse_area_keys: set[tuple[str, ...]],
        coarse_area_cols: dict[tuple[str, ...], list[tuple[int, PlacementColumn]]],
    ) -> None:
        area_keys_by_coarse: defaultdict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
        for key in coarse_area_keys:
            coarse_key = tuple(key[:-1])
            area_keys_by_coarse[coarse_key].append(key)

        for coarse_key, area_keys in sorted(area_keys_by_coarse.items()):
            demand = self._coarse_metric_demand(coarse_key)
            if demand <= 0:
                continue
            actual_by_area = self._add_coarse_area_actual_variables(
                quicksum,
                model,
                columns,
                coarse_key,
                area_keys,
                coarse_area_cols,
                demand,
            )
            if self._prefers_concentrated_coarse_key(coarse_key):
                self._add_concentrated_coarse_group_objective(
                    model,
                    coarse_key,
                    area_keys,
                    actual_by_area,
                    demand,
                )
            else:
                self._add_large_coarse_group_balance_objective(
                    model,
                    coarse_key,
                    area_keys,
                    actual_by_area,
                    demand,
                )

    def _add_coarse_area_actual_variables(
        self,
        quicksum,
        model,
        columns,
        coarse_key: tuple[str, ...],
        area_keys: list[tuple[str, ...]],
        coarse_area_cols: dict[tuple[str, ...], list[tuple[int, PlacementColumn]]],
        demand: int,
    ) -> dict[tuple[str, ...], object]:
        actual_by_area = {}
        name_key = self._key_name(coarse_key)
        for key in sorted(area_keys):
            *_, area_no = key
            items = coarse_area_cols.get(key, [])
            actual = model.addVar(
                lb=0.0,
                ub=float(demand),
                name=f"coarse_actual_{name_key}_{area_no}",
            )
            model.addCons(actual == quicksum(col.quantity * columns[idx] for idx, col in items))
            actual_by_area[key] = actual
        return actual_by_area

    def _add_concentrated_coarse_group_objective(
        self,
        model,
        coarse_key: tuple[str, ...],
        area_keys: list[tuple[str, ...]],
        actual_by_area: dict[tuple[str, ...], object],
        demand: int,
    ) -> None:
        name_key = self._key_name(coarse_key)
        largest = model.addVar(
            lb=0.0,
            ub=float(demand),
            obj=-self.config.medium_small_group_fragment_penalty,
            name=f"coarse_largest_{name_key}",
        )
        primary_vars = []
        for key in sorted(area_keys):
            *_, area_no = key
            actual = actual_by_area[key]
            use = model.addVar(
                vtype="B",
                obj=self.config.medium_small_group_area_split_penalty,
                name=f"use_conc_area_{name_key}_{area_no}",
            )
            primary = model.addVar(
                vtype="B",
                obj=-self.config.medium_small_group_area_split_penalty,
                name=f"primary_conc_area_{name_key}_{area_no}",
            )
            model.addCons(actual <= demand * use)
            model.addCons(primary <= use)
            model.addCons(largest <= actual + demand * (1 - primary))
            primary_vars.append(primary)
        if primary_vars:
            primary_sum = sum(primary_vars)
            model.addCons(primary_sum <= 1)
            model.addCons(largest <= demand * primary_sum)

    def _add_large_coarse_group_balance_objective(
        self,
        model,
        coarse_key: tuple[str, ...],
        area_keys: list[tuple[str, ...]],
        actual_by_area: dict[tuple[str, ...], object],
        demand: int,
    ) -> None:
        name_key = self._key_name(coarse_key)
        area_terms = []
        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        small_area_penalty = self.config.medium_large_group_small_area_penalty / max(1.0, min_boxes)
        use_vars = []
        for key in sorted(area_keys):
            *_, area_no = key
            actual = actual_by_area[key]
            use = model.addVar(vtype="B", name=f"use_bal_area_{name_key}_{area_no}")
            model.addCons(actual <= demand * use)
            model.addCons(actual >= use)
            use_vars.append(use)
            if min_boxes > 0:
                shortage = model.addVar(
                    lb=0.0,
                    obj=small_area_penalty,
                    name=f"small_bal_area_{name_key}_{area_no}",
                )
                model.addCons(shortage >= min_boxes * use - actual)
            area_terms.append((area_no, actual, use))

        target_area_count = self._target_large_group_area_count(coarse_key, demand)
        if use_vars and self.config.medium_large_group_area_open_penalty > 0:
            extra_areas = model.addVar(
                lb=0.0,
                obj=self.config.medium_large_group_area_open_penalty,
                name=f"extra_bal_area_{name_key}",
            )
            model.addCons(extra_areas >= sum(use_vars) - target_area_count)

        target_boxes = max(1, int(self.config.medium_large_group_target_area_boxes or 1))
        excess_penalty = max(0.0, float(self.config.medium_large_group_area_excess_penalty or 0.0))
        if excess_penalty > 0:
            for area_no, actual, _use in area_terms:
                excess = model.addVar(
                    lb=0.0,
                    obj=excess_penalty,
                    name=f"excess_bal_area_{name_key}_{area_no}",
                )
                model.addCons(excess >= actual - target_boxes)

        if len(area_terms) <= 1:
            return
        pair_penalty = self.config.group_area_balance_penalty / max(1.0, demand) / max(1, len(area_terms) - 1)
        for left_index, (_left_area, left_actual, left_use) in enumerate(area_terms):
            for _right_area, right_actual, right_use in area_terms[left_index + 1 :]:
                diff = model.addVar(lb=0.0, obj=pair_penalty, name=f"bal_diff_{len(model.getVars())}")
                inactive = demand * (2 - left_use - right_use)
                model.addCons(diff >= left_actual - right_actual - inactive)
                model.addCons(diff >= right_actual - left_actual - inactive)

    def _add_bay_compatibility_constraints(
        self,
        quicksum,
        model,
        columns,
        bay_attr_choice_cols: dict[tuple[str, str, str, str], list[int]],
    ) -> None:
        use_by_bay_attr: defaultdict[tuple[str, str, str], list] = defaultdict(list)
        for (bay_key, attr, scope, value), indices in sorted(bay_attr_choice_cols.items()):
            scope_name = scope or "ALL"
            use = model.addVar(vtype="B", name=f"bay_use_{attr}_{scope_name}_{bay_key}_{value}")
            model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
            use_by_bay_attr[(bay_key, attr, scope)].append(use)
        for (bay_key, attr, scope), uses in use_by_bay_attr.items():
            scope_name = scope or "ALL"
            model.addCons(quicksum(uses) <= 1, name=f"bay_one_{attr}_{scope_name}_{bay_key}")

    def _add_row_compatibility_constraints(
        self,
        quicksum,
        model,
        columns,
        row_attr_choice_cols: dict[tuple[str, str, str, str, str], list[int]],
    ) -> None:
        use_by_row_attr: defaultdict[tuple[str, str, str, str], list] = defaultdict(list)
        for (bay_key, row_no, attr, scope, value), indices in sorted(row_attr_choice_cols.items()):
            scope_name = scope or "ALL"
            use = model.addVar(vtype="B", name=f"row_use_{attr}_{scope_name}_{bay_key}_{row_no}_{value}")
            model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
            use_by_row_attr[(bay_key, row_no, attr, scope)].append(use)
        for (bay_key, row_no, attr, scope), uses in use_by_row_attr.items():
            scope_name = scope or "ALL"
            model.addCons(quicksum(uses) <= 1, name=f"row_one_{attr}_{scope_name}_{bay_key}_{row_no}")

    def _price_columns(
        self,
        lp_model,
        lp_constraints: dict,
        iteration: int,
        lp_unplaced: Counter[str] | None = None,
        lp_column_values: dict[int, float] | None = None,
    ) -> dict:
        group_dual = {group_id: self._scip_dual(lp_model, constr) for group_id, constr in lp_constraints["group_cover"].items()}
        group_dual = self._effective_group_duals(group_dual, lp_unplaced or Counter())
        bay_capacity_dual = {bay_key: self._scip_dual(lp_model, constr) for bay_key, constr in lp_constraints["bay_capacity_limit"].items()}
        bay_size_dual = {key: self._scip_dual(lp_model, constr) for key, constr in lp_constraints["bay_size_limit"].items()}
        twenty_run_bay_use_dual = {
            bay_key: self._scip_dual(lp_model, constr)
            for bay_key, constr in lp_constraints.get("twenty_run_bay_use_link", {}).items()
        }
        bay_port_stack_dual = {
            key: self._scip_dual(lp_model, constr)
            for key, constr in lp_constraints.get("bay_port_stack_link", {}).items()
        }
        group_bay_dual = {key: self._scip_dual(lp_model, constr) for key, constr in lp_constraints["group_bay_limit"].items()}
        quota_dual = {
            key: self._scip_dual(lp_model, constr)
            for key, constr in lp_constraints.get("quota_limit", {}).items()
        }
        medium_plan_quota_dual = {
            key: self._scip_dual(lp_model, constr)
            for key, constr in lp_constraints.get("medium_plan_quota_limit", {}).items()
        }
        big_plan_deviation_dual = {
            key: self._scip_dual(lp_model, constr)
            for key, constr in lp_constraints.get("big_plan_deviation_balance", {}).items()
        }
        fixed_use_dual = {
            key: self._scip_dual(lp_model, constr)
            for key, constr in lp_constraints.get("fixed_use_objective_limit", {}).items()
        }
        lp_column_values = lp_column_values or {}
        lp_quota_actual = self._column_values_quota_actual(lp_column_values)
        lp_coarse_area_actual = self._column_values_coarse_area_actual(lp_column_values)
        lp_unplaced = lp_unplaced or Counter()
        negative_heap: list[tuple[_ReverseSortKey, int, float, tuple[float, SmallBoxGroup, str, int, float]]] = []
        stalled_heap: list[tuple[_ReverseSortKey, int, float, tuple[float, SmallBoxGroup, str, int, float]]] = []
        primal_heap: list[tuple[_ReverseSortKey, int, tuple, tuple[tuple, float, SmallBoxGroup, str, int, float]]] = []
        sequence = 0
        scanned = 0
        skipped_existing = 0
        negative_count = 0
        best_reduced: float | None = None
        candidate_bays_available = 0
        candidate_bays_considered = 0
        candidate_bay_limits: list[int] = []
        candidate_bay_limited_groups = 0
        candidate_bay_unplaced_groups = 0
        candidate_bay_high_dual_groups = 0
        adaptive_enabled = bool(getattr(self.config, "adaptive_pricing_enabled", True))
        negative_limit = max(0, int(getattr(self.config, "columns_per_iteration", 0) or 0))
        stalled_limit = max(0, int(getattr(self.config, "stalled_pricing_columns", 0) or 0)) if iteration == 0 else 0
        primal_limit = max(0, int(getattr(self.config, "primal_expansion_columns", 0) or 0))
        primal_rounds_allowed = self._primal_expansion_rounds < max(
            0,
            int(self.config.max_primal_expansion_rounds or 0),
        )
        reduced_limit = float(getattr(self.config, "primal_expansion_reduced_cost_limit", 0.0) or 0.0)
        max_group_dual = max((abs(float(value)) for value in group_dual.values()), default=0.0)

        def keep_best(heap: list, limit: int, key, payload) -> None:
            nonlocal sequence
            if limit <= 0:
                return
            item = (_ReverseSortKey(key), sequence, key, payload)
            sequence += 1
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif key < heap[0][2]:
                heapq.heapreplace(heap, item)

        def heap_payloads(heap: list) -> list:
            return [payload for _reverse_key, _seq, _key, payload in sorted(heap, key=lambda item: item[2])]

        for group in self.groups:
            group_candidates = self._candidate_bays_for_group(group)
            candidate_bays_available += len(group_candidates)
            bay_limit, limit_reason = self._pricing_candidate_bay_limit(
                group,
                len(group_candidates),
                iteration,
                group_dual.get(group.group_id, 0.0),
                max_group_dual,
                lp_unplaced,
            )
            if not adaptive_enabled:
                bay_limit = len(group_candidates)
                limit_reason = "full"
            bay_limit = min(len(group_candidates), max(0, int(bay_limit)))
            if bay_limit < len(group_candidates):
                candidate_bay_limited_groups += 1
            if limit_reason == "lp_unplaced":
                candidate_bay_unplaced_groups += 1
            elif limit_reason == "high_dual":
                candidate_bay_high_dual_groups += 1
            candidate_bay_limits.append(bay_limit)
            candidate_bays_considered += bay_limit
            for bay_key, max_qty, base_cost in group_candidates[:bay_limit]:
                for qty in self._quantity_options(group, max_qty):
                    triplet = (group.group_id, bay_key, qty)
                    if triplet in self._default_column_triplets:
                        skipped_existing += 1
                        continue
                    scanned += 1
                    area_no = self.bays[bay_key].area_no
                    coarse_area_key = self._coarse_key(group) + (area_no,)
                    block_id = self.block_by_bay.get((area_no, bay_key), "")
                    coarse_cluster_key = self._coarse_cluster_key(group)
                    fine_cluster_key = self._fine_cluster_key(group)
                    group_area_key = ("group_area", fine_cluster_key, area_no)
                    group_block_key = ("group_block", fine_cluster_key, block_id)
                    coarse_area_bay_key = ("coarse_area_bay",) + coarse_cluster_key + (area_no, bay_key)
                    coarse_area_block_key = ("coarse_area_block",) + coarse_cluster_key + (area_no, block_id)
                    voyage_area_key = ("voyage_area", group.voyage_id, area_no)
                    reduced = (
                        base_cost
                        + self.config.small_plan_group_bay_split_penalty
                        - group_dual.get(group.group_id, 0.0) * qty
                        - bay_capacity_dual.get(bay_key, 0.0) * qty
                        - bay_size_dual.get((bay_key, group.size), 0.0) * qty
                        - (twenty_run_bay_use_dual.get(bay_key, 0.0) if group.size == "20" else 0.0)
                        - sum(
                            bay_port_stack_dual.get((footprint_key, self._row_mix_key_for_group(group), group.size), 0.0) * qty
                            for footprint_key in self._placement_footprint_keys(bay_key, group.size)
                        )
                        - group_bay_dual.get((group.group_id, bay_key), 0.0)
                        - quota_dual.get(self._quota_key(group, area_no), 0.0) * qty
                        - medium_plan_quota_dual.get(coarse_area_key, 0.0) * qty
                        - medium_plan_quota_dual.get(coarse_area_key + (bay_key,), 0.0) * qty
                        - big_plan_deviation_dual.get(self._quota_key(group, area_no), 0.0) * qty
                        + self._fixed_use_reduced_adjustment(
                            fixed_use_dual,
                            group_area_key,
                            self.config.small_plan_group_area_split_penalty,
                        )
                        + self._fixed_use_reduced_adjustment(
                            fixed_use_dual,
                            coarse_area_bay_key,
                            self.config.small_plan_coarse_area_bay_split_penalty,
                        )
                        + self._fixed_use_reduced_adjustment(
                            fixed_use_dual,
                            voyage_area_key,
                            self._voyage_area_cost(group.voyage_id, area_no),
                        )
                    )
                    if block_id:
                        reduced += self._fixed_use_reduced_adjustment(
                            fixed_use_dual,
                            group_block_key,
                            self.config.small_plan_group_block_split_penalty,
                        )
                        reduced += self._fixed_use_reduced_adjustment(
                            fixed_use_dual,
                            coarse_area_block_key,
                            self.config.small_plan_coarse_area_block_split_penalty,
                        )
                    candidate = (reduced, group, bay_key, qty, base_cost)
                    if best_reduced is None or reduced < best_reduced:
                        best_reduced = reduced
                    if stalled_limit > 0:
                        keep_best(stalled_heap, stalled_limit, reduced, candidate)
                    if reduced < -1e-6:
                        negative_count += 1
                        keep_best(negative_heap, negative_limit, reduced, candidate)
                    elif negative_count == 0 and primal_rounds_allowed and primal_limit > 0:
                        if reduced_limit <= 0 or reduced <= reduced_limit:
                            primal_score = self._primal_expansion_score(
                                group,
                                bay_key,
                                qty,
                                base_cost,
                                lp_quota_actual,
                                lp_coarse_area_actual,
                            )
                            keep_best(
                                primal_heap,
                                primal_limit,
                                primal_score,
                                (primal_score, reduced, group, bay_key, qty, base_cost),
                            )
        if negative_count:
            selected_candidates = heap_payloads(negative_heap)
            mode = "negative_reduced_cost"
        elif iteration == 0:
            selected_candidates = heap_payloads(stalled_heap)
            mode = "stalled_best_reduced_cost"
        elif primal_rounds_allowed:
            selected_candidates = [
                (reduced, group, bay_key, qty, base_cost)
                for _score, reduced, group, bay_key, qty, base_cost in heap_payloads(primal_heap)
            ]
            if selected_candidates:
                self._primal_expansion_rounds += 1
                mode = "integer_primal_expansion"
            else:
                mode = "no_primal_candidate_under_limit"
        else:
            selected_candidates = []
            mode = "no_improving_column"
        added = 0
        for _reduced, group, bay_key, qty, base_cost in selected_candidates:
            if self._add_column(group, bay_key, qty, base_cost):
                added += 1
        return {
            "new_columns": added,
            "pricing_mode": mode,
            "pricing_topk_heap": True,
            "adaptive_pricing_enabled": adaptive_enabled,
            "scanned_candidates": scanned,
            "skipped_existing_column_triplets": skipped_existing,
            "candidate_bays_available": candidate_bays_available,
            "candidate_bays_considered": candidate_bays_considered,
            "candidate_bay_limited_groups": candidate_bay_limited_groups,
            "candidate_bay_unplaced_groups": candidate_bay_unplaced_groups,
            "candidate_bay_high_dual_groups": candidate_bay_high_dual_groups,
            "candidate_bay_limit_min": min(candidate_bay_limits) if candidate_bay_limits else 0,
            "candidate_bay_limit_max": max(candidate_bay_limits) if candidate_bay_limits else 0,
            "candidate_bay_limit_avg": round(
                sum(candidate_bay_limits) / max(1, len(candidate_bay_limits)),
                3,
            ),
            "negative_reduced_candidates": negative_count,
            "primal_expansion_rounds_used": self._primal_expansion_rounds,
            "primal_expansion_reduced_cost_limit": self.config.primal_expansion_reduced_cost_limit,
            "best_reduced_cost": round(best_reduced, 6) if best_reduced is not None else None,
            "worst_added_reduced_cost": round(selected_candidates[-1][0], 6) if selected_candidates else None,
        }

    @staticmethod
    def _fixed_use_reduced_adjustment(fixed_use_dual: dict, key: tuple, fixed_cost: float) -> float:
        if key in fixed_use_dual:
            return -float(fixed_use_dual.get(key, 0.0))
        return float(fixed_cost)

    def _column_values_quota_actual(self, column_values: dict[int, float]) -> Counter[tuple[str, str, str, str]]:
        actual: Counter[tuple[str, str, str, str]] = Counter()
        for idx, value in column_values.items():
            if value <= 1e-9 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            actual[col.quota_key] += col.quantity * float(value)
        return actual

    def _column_values_coarse_area_actual(self, column_values: dict[int, float]) -> Counter[tuple[str, ...]]:
        actual: Counter[tuple[str, ...]] = Counter()
        for idx, value in column_values.items():
            if value <= 1e-9 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            actual[col.coarse_cluster_key + (col.area_no,)] += col.quantity * float(value)
        return actual

    def _primal_expansion_score(
        self,
        group: SmallBoxGroup,
        bay_key: str,
        qty: int,
        base_cost: float,
        lp_quota_actual: Counter[tuple[str, str, str, str]],
        lp_coarse_area_actual: Counter[tuple[str, ...]],
    ) -> tuple:
        bay = self.bays[bay_key]
        area_no = bay.area_no
        quota_key = self._quota_key(group, area_no)
        target = self._area_size_target(group.voyage_id, group.status, area_no, self._big_plan_size(group.size))
        quota_gap = target - lp_quota_actual.get(quota_key, 0.0)
        coarse_area_key = self._coarse_cluster_key(group) + (area_no,)
        existing_area_qty = lp_coarse_area_actual.get(coarse_area_key, 0.0)
        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        if self._prefers_concentrated_coarse_key(self._coarse_cluster_key(group)):
            shape_score = (0 if existing_area_qty > 1e-6 else 1, -existing_area_qty)
        else:
            after_qty = existing_area_qty + qty
            shortage_after = max(0.0, float(min_boxes) - after_qty) if after_qty > 1e-6 else 0.0
            shape_score = (0 if existing_area_qty > 1e-6 else 1, shortage_after)
        return (
            0 if quota_gap > 1e-6 else 1,
            -min(float(qty), max(0.0, quota_gap)),
            shape_score,
            self._voyage_area_cost(group.voyage_id, area_no),
            base_cost,
            -qty,
            bay.bay_order,
        )

    def _effective_group_duals(self, group_dual: dict[str, float], lp_unplaced: Counter[str]) -> dict[str, float]:
        out = dict(group_dual)
        for group_id, qty in lp_unplaced.items():
            penalty = self._unplaced_penalty_for_group_id(group_id)
            threshold = penalty * 0.1
            if qty > 0 and out.get(group_id, 0.0) < threshold:
                out[group_id] = penalty
        return out

    def _build_initial_columns(self) -> None:
        for group in self.groups:
            candidates = self._candidate_bays_for_group(group)
            for bay_key, max_qty, base_cost in candidates[: self.config.initial_columns_per_group]:
                for qty in self._quantity_options(group, max_qty):
                    self._add_column(group, bay_key, qty, base_cost)

    def _expand_all_candidate_columns(self) -> None:
        for group in self.groups:
            for bay_key, max_qty, base_cost in self._candidate_bays_for_group(group):
                for qty in self._quantity_options(group, max_qty):
                    self._add_column(group, bay_key, qty, base_cost)

    def _bay_no_mix_attrs(self, voyage_id: object = None) -> tuple[str, ...]:
        if self.attribute_rules is not None and voyage_id is not None and hasattr(self.attribute_rules, "bay_no_mix_for"):
            attrs = self.attribute_rules.bay_no_mix_for(voyage_id)
        else:
            attrs = getattr(self.attribute_rules, "bay_no_mix_attributes", ())
        ordered: list[str] = list(MANDATORY_BAY_NO_MIX_ATTRS)
        seen = set(ordered)
        for attr in attrs:
            name = str(attr).strip()
            if not name:
                continue
            if self._is_size_no_mix_attr(name):
                name = MANDATORY_BAY_NO_MIX_ATTRS[0]
            if name not in seen:
                ordered.append(name)
                seen.add(name)
        return tuple(ordered)

    def _row_no_mix_attrs(self, voyage_id: object = None) -> tuple[str, ...]:
        if self.attribute_rules is not None and voyage_id is not None and hasattr(self.attribute_rules, "row_no_mix_for"):
            attrs = self.attribute_rules.row_no_mix_for(voyage_id)
        else:
            attrs = getattr(self.attribute_rules, "row_no_mix_attributes", ())
        return tuple(str(attr) for attr in attrs if str(attr))

    @staticmethod
    def _infer_import_voyages(problem: ProblemData) -> set[str]:
        declared_exports = getattr(problem, "export_voyages", None)
        if declared_exports is not None:
            all_voyages = {str(voyage_id) for voyage_id in getattr(problem, "target_voyages", []) if str(voyage_id)}
            all_voyages.update(
                str(group.voyage_id)
                for group in list(getattr(problem, "groups", []) or [])
                + list(getattr(problem, "small_groups", []) or [])
                if str(group.voyage_id)
            )
            return all_voyages - {str(voyage_id) for voyage_id in declared_exports if str(voyage_id)}
        flows_by_voyage: defaultdict[str, set[str]] = defaultdict(set)
        for row in getattr(problem, "big_plan", []) or []:
            flows_by_voyage[str(row.voyage_id)].add(str(row.flow))
        for group in list(getattr(problem, "groups", []) or []) + list(getattr(problem, "small_groups", []) or []):
            flows_by_voyage[str(group.voyage_id)].add(str(group.status))
        return {
            voyage_id
            for voyage_id, flows in flows_by_voyage.items()
            if flows and not any(flow in EXPORT_FLOWS for flow in flows)
        }

    @staticmethod
    def _infer_export_voyages(problem: ProblemData) -> set[str]:
        declared = getattr(problem, "export_voyages", None)
        if declared is not None:
            return {str(voyage_id) for voyage_id in declared if str(voyage_id)}
        return {
            str(group.voyage_id)
            for group in list(getattr(problem, "groups", []) or [])
            + list(getattr(problem, "small_groups", []) or [])
            if str(group.status) in EXPORT_FLOWS
        }

    def _is_import_voyage(self, voyage_id: object) -> bool:
        return str(voyage_id) in self.import_voyages

    def _is_export_voyage(self, voyage_id: object) -> bool:
        return str(voyage_id) in self.export_voyages

    def _bay_no_mix_attrs_for_group(self, group: SmallBoxGroup) -> tuple[str, ...]:
        return self._bay_no_mix_attrs(group.voyage_id)

    def _row_no_mix_attrs_for_group(self, group: SmallBoxGroup) -> tuple[str, ...]:
        attrs = list(self._row_no_mix_attrs(group.voyage_id))
        if self._is_export_voyage(group.voyage_id) and EXPORT_VOYAGE_ROW_NO_MIX_ATTR not in attrs:
            attrs.append(EXPORT_VOYAGE_ROW_NO_MIX_ATTR)
        return tuple(attrs)

    def _bay_no_mix_attrs_for_column(self, col: PlacementColumn) -> tuple[str, ...]:
        return self._bay_no_mix_attrs(col.voyage_id)

    def _row_no_mix_attrs_for_column(self, col: PlacementColumn) -> tuple[str, ...]:
        attrs = list(self._row_no_mix_attrs(col.voyage_id))
        if self._is_export_voyage(col.voyage_id) and EXPORT_VOYAGE_ROW_NO_MIX_ATTR not in attrs:
            attrs.append(EXPORT_VOYAGE_ROW_NO_MIX_ATTR)
        return tuple(attrs)

    @staticmethod
    def _group_attr_value(group: SmallBoxGroup, attr: str) -> str:
        attrs = getattr(group, "attributes", {}) or {}
        value = attrs.get(attr, "")
        if isinstance(value, bool):
            return "1" if value else "0"
        if value not in (None, ""):
            return str(value)
        text = str(attr).strip()
        upper = text.upper()
        fallback = {
            "IYC_STS_CSTATUSCD": group.status,
            "STATUS": group.status,
            "FLOW": group.status,
            "IYC_CSZ_CSIZECD": group.size,
            "SIZE": group.size,
            "SIZE_MODE": group.size,
            "IYC_POT_UNLDPORT": group.port,
            "PORT": group.port,
            "IYC_CHEIGHTCD": group.height,
            "HEIGHT": group.height,
            "IYC_CWEIGHT": group.weight_class,
            "WEIGHT": group.weight_class,
            "WEIGHT_CLASS": group.weight_class,
            "IYC_EVOY_ID": group.voyage_id,
            "IYC_IVOY_ID": group.voyage_id,
            "VOYAGE_ID": group.voyage_id,
            EXPORT_VOYAGE_ROW_NO_MIX_ATTR.upper(): group.voyage_id,
        }.get(upper, "")
        return str(fallback)

    @staticmethod
    def _column_attr_value(col: PlacementColumn, attr: str) -> str:
        attrs = getattr(col, "attributes", {}) or {}
        value = attrs.get(attr, "")
        if isinstance(value, bool):
            return "1" if value else "0"
        if value not in (None, ""):
            return str(value)
        text = str(attr).strip()
        upper = text.upper()
        fallback = {
            "IYC_STS_CSTATUSCD": col.flow,
            "STATUS": col.flow,
            "FLOW": col.flow,
            "IYC_CSZ_CSIZECD": col.size,
            "SIZE": col.size,
            "SIZE_MODE": col.size,
            "IYC_POT_UNLDPORT": col.port,
            "PORT": col.port,
            "IYC_CHEIGHTCD": col.height,
            "HEIGHT": col.height,
            "IYC_CWEIGHT": col.weight_class,
            "WEIGHT": col.weight_class,
            "WEIGHT_CLASS": col.weight_class,
            "IYC_EVOY_ID": col.voyage_id,
            "IYC_IVOY_ID": col.voyage_id,
            "VOYAGE_ID": col.voyage_id,
            EXPORT_VOYAGE_ROW_NO_MIX_ATTR.upper(): col.voyage_id,
        }.get(upper, "")
        return str(fallback)

    def _row_mix_key_for_group(self, group: SmallBoxGroup) -> str:
        return "|".join(f"{attr}={self._group_attr_value(group, attr)}" for attr in self._row_no_mix_attrs_for_group(group)) or "__all__"

    def _row_mix_key_for_column(self, col: PlacementColumn) -> str:
        return "|".join(f"{attr}={self._column_attr_value(col, attr)}" for attr in self._row_no_mix_attrs_for_column(col)) or "__all__"

    @staticmethod
    def _is_size_no_mix_attr(attr: str) -> bool:
        return str(attr).strip().upper() in SIZE_NO_MIX_ATTRS

    @staticmethod
    def _is_global_row_no_mix_attr(attr: str) -> bool:
        return (
            str(attr).strip().upper() == EXPORT_VOYAGE_ROW_NO_MIX_ATTR.upper()
            or ColumnGenerationPlanner._is_size_no_mix_attr(attr)
        )

    def _attr_voyage_scope(self, attr: str, voyage_id: object) -> str:
        return "" if self._is_global_row_no_mix_attr(attr) else str(voyage_id)

    def _bay_state_attr_key(self, bay_key: str, attr: str, voyage_id: object) -> tuple[str, str, str]:
        return (bay_key, attr, self._attr_voyage_scope(attr, voyage_id))

    def _row_state_attr_key(self, bay_key: str, row_no: str, attr: str, voyage_id: object) -> tuple[str, str, str, str]:
        return (bay_key, str(row_no), attr, self._attr_voyage_scope(attr, voyage_id))

    def _existing_bay_attr_values(self, bay: Bay, attr: str, voyage_id: object) -> set[str]:
        if self._is_size_no_mix_attr(attr):
            values = set(getattr(bay, "existing_attrs", {}).get(attr, set()))
            return values or set(getattr(bay, "existing_size_modes", set()))
        by_voyage = getattr(bay, "existing_attrs_by_voyage", {}) or {}
        return set(by_voyage.get(str(voyage_id), {}).get(attr, set()))

    def _existing_row_attr_values(self, bay: Bay, row_no: str, attr: str, voyage_id: object) -> set[str]:
        if self._is_global_row_no_mix_attr(attr):
            row_attrs = getattr(bay, "existing_attrs_by_row", {}).get(str(row_no), {})
            return set(row_attrs.get(attr, set()))
        by_row_voyage = getattr(bay, "existing_attrs_by_row_by_voyage", {}) or {}
        return set(by_row_voyage.get(str(row_no), {}).get(str(voyage_id), {}).get(attr, set()))

    def _row_existing_attrs_allow_group(self, bay: Bay, row_no: str, group: SmallBoxGroup) -> bool:
        for attr in self._row_no_mix_attrs_for_group(group):
            values = self._existing_row_attr_values(bay, str(row_no), attr, group.voyage_id)
            expected = self._group_attr_value(group, attr)
            if attr == EXPORT_VOYAGE_ROW_NO_MIX_ATTR:
                if values and values != {expected}:
                    return False
            elif values and expected not in values:
                return False
        return True

    def _bay_existing_attrs_allow_group(self, group: SmallBoxGroup, footprint: tuple[str, ...]) -> bool:
        for key in footprint:
            bay = self.bays[key]
            for attr in self._bay_no_mix_attrs_for_group(group):
                values = self._existing_bay_attr_values(bay, attr, group.voyage_id)
                if values and values != {self._group_attr_value(group, attr)}:
                    return False
        return True

    def _bay_state_attrs_allow_group(self, group: SmallBoxGroup, footprint: tuple[str, ...], state: dict) -> bool:
        used_attrs = state.setdefault("bay_used_attrs", {})
        for key in footprint:
            for attr in self._bay_no_mix_attrs_for_group(group):
                state_key = self._bay_state_attr_key(key, attr, group.voyage_id)
                value = self._group_attr_value(group, attr)
                if used_attrs.get(state_key, value) != value:
                    return False
        return True

    def _row_stack_capacities_for_group(self, bay_key: str, size: str, group: SmallBoxGroup) -> list[int]:
        return [cap for _row_no, cap in self._row_stack_capacity_items_for_group(bay_key, size, group)]

    def _row_stack_capacity_items_for_group(self, bay_key: str, size: str, group: SmallBoxGroup) -> list[tuple[str, int]]:
        bay = self.bays[bay_key]
        row_caps = bay.row_cap_by_size.get(size, {}) or {}
        if not row_caps and bay.row_physical_capacity:
            row_caps = bay.row_physical_capacity
        has_row_caps = bool(row_caps)
        capacities: list[tuple[str, int]] = []
        for row_no, cap in row_caps.items():
            if not self._row_existing_attrs_allow_group(bay, str(row_no), group):
                continue
            cap_int = int(cap)
            if cap_int > 0:
                capacities.append((str(row_no), cap_int))
        if has_row_caps:
            return capacities
        fallback = int(bay.cap_by_size.get(size, 0) or bay.physical_capacity)
        return [("__bay__", fallback)] if fallback > 0 else []

    def _row_capacity_items_for_group(
        self,
        footprint_key: str,
        size: str,
        group: SmallBoxGroup,
        state: dict | None = None,
    ) -> list[tuple[str, int]]:
        bay = self.bays[footprint_key]
        row_caps = bay.row_cap_by_size.get(size, {}) or {}
        if not row_caps and bay.row_physical_capacity:
            row_caps = bay.row_physical_capacity
        if row_caps:
            out: list[tuple[str, int]] = []
            for row_no, raw_cap in row_caps.items():
                row_no = str(row_no)
                if not self._row_existing_attrs_allow_group(bay, row_no, group):
                    continue
                cap = int(raw_cap)
                if state is not None:
                    row_key = (footprint_key, row_no)
                    row_size_key = (footprint_key, row_no, size)
                    cap = min(
                        cap - int(state["row_load"][row_key]),
                        int(bay.row_physical_capacity.get(row_no, raw_cap)) - int(state["row_load"][row_key]),
                        int(raw_cap) - int(state["row_size_load"][row_size_key]),
                    )
                    for attr in self._row_no_mix_attrs_for_group(group):
                        value = self._group_attr_value(group, attr)
                        state_key = self._row_state_attr_key(footprint_key, row_no, attr, group.voyage_id)
                        used = state["row_used_attrs"].get(state_key, value)
                        if used != value:
                            cap = 0
                            break
                if cap > 0:
                    out.append((row_no, int(cap)))
            return sorted(out, key=lambda item: self._row_sort_key(item[0]))
        cap = int(bay.cap_by_size.get(size, 0) or bay.physical_capacity)
        if state is not None:
            cap = min(
                cap - int(state["bay_load"][footprint_key]),
                int(bay.physical_capacity) - int(state["bay_load"][footprint_key]),
                int(bay.cap_by_size.get(size, cap)) - int(state["bay_size_load"][(footprint_key, size)]),
            )
        return [("__bay__", cap)] if cap > 0 else []

    @staticmethod
    def _row_sort_key(row_no: str) -> tuple[int, str]:
        try:
            return int(row_no), row_no
        except ValueError:
            return 10**9, row_no

    @staticmethod
    def _row_allocation_signature(row_allocation: tuple[tuple[str, str, int], ...]) -> tuple[tuple[str, str, int], ...]:
        return tuple(sorted((str(bay_key), str(row_no), int(qty)) for bay_key, row_no, qty in row_allocation if int(qty) > 0))

    def _row_allocation_patterns_for_column(
        self,
        group: SmallBoxGroup,
        bay_key: str,
        quantity: int,
        state: dict | None = None,
        max_patterns: int = 6,
    ) -> list[tuple[tuple[str, str, int], ...]]:
        if quantity <= 0:
            return []
        footprint = self._placement_footprint_keys(bay_key, group.size)
        if not footprint:
            return []
        per_bay = {
            key: dict(self._row_capacity_items_for_group(key, group.size, group, state=state))
            for key in footprint
        }
        if any(not caps for caps in per_bay.values()):
            return []
        common_rows = set.intersection(*(set(caps) for caps in per_bay.values()))
        if not common_rows:
            return []
        row_caps = {
            row_no: min(per_bay[key][row_no] for key in footprint)
            for row_no in common_rows
        }
        row_caps = {row_no: cap for row_no, cap in row_caps.items() if cap > 0}
        if sum(row_caps.values()) < quantity:
            return []

        ordered_rows = sorted(row_caps, key=self._row_sort_key)
        row_orders: list[list[str]] = [ordered_rows, list(reversed(ordered_rows))]
        row_orders.extend([[row_no] + [other for other in ordered_rows if other != row_no] for row_no in ordered_rows if row_caps[row_no] >= quantity])

        patterns: list[tuple[tuple[str, str, int], ...]] = []
        seen: set[tuple[tuple[str, str, int], ...]] = set()
        for order in row_orders:
            remaining = int(quantity)
            by_row: list[tuple[str, int]] = []
            for row_no in order:
                if remaining <= 0:
                    break
                take = min(remaining, row_caps[row_no])
                if take > 0:
                    by_row.append((row_no, take))
                    remaining -= take
            if remaining > 0:
                continue
            allocation = self._row_allocation_signature(
                tuple((footprint_key, row_no, take) for row_no, take in by_row for footprint_key in footprint)
            )
            if allocation in seen:
                continue
            seen.add(allocation)
            patterns.append(allocation)
            if len(patterns) >= max_patterns:
                break
        return patterns

    def _row_capacity_for_column(
        self,
        group: SmallBoxGroup,
        bay_key: str,
        state: dict | None = None,
    ) -> int:
        """Return row-feasible capacity without constructing allocations."""
        footprint = self._placement_footprint_keys(bay_key, group.size)
        if not footprint:
            return 0
        per_bay = {
            key: dict(self._row_capacity_items_for_group(key, group.size, group, state=state))
            for key in footprint
        }
        if any(not caps for caps in per_bay.values()):
            return 0
        common_rows = set.intersection(*(set(caps) for caps in per_bay.values()))
        return sum(
            min(per_bay[key][row_no] for key in footprint)
            for row_no in common_rows
        )

    def _stack_count_for_group(self, bay_key: str, size: str, group: SmallBoxGroup) -> int:
        return len(self._row_stack_capacities_for_group(bay_key, size, group))

    def _stack_count_for_bay_size(self, bay_key: str, size: str) -> int:
        bay = self.bays[bay_key]
        row_caps = bay.row_cap_by_size.get(size, {}) or bay.row_physical_capacity
        if row_caps:
            return sum(1 for cap in row_caps.values() if int(cap) > 0)
        return 1 if int(bay.cap_by_size.get(size, 0) or bay.physical_capacity) > 0 else 0

    def _stack_unit_capacity_for_group(self, bay_key: str, size: str, group: SmallBoxGroup) -> int:
        capacities = self._row_stack_capacities_for_group(bay_key, size, group)
        return max(capacities) if capacities else 0

    def _stack_units_for_quantity(self, bay_key: str, size: str, group: SmallBoxGroup, quantity: int) -> int:
        if quantity <= 0:
            return 0
        unit_capacity = self._stack_unit_capacity_for_group(bay_key, size, group)
        if unit_capacity <= 0:
            return 10**9
        return int(math.ceil(quantity / unit_capacity))

    def _column_stack_units(self, group: SmallBoxGroup, bay_key: str, quantity: int) -> int:
        units = [
            self._stack_units_for_quantity(key, group.size, group, quantity)
            for key in self._placement_footprint_keys(bay_key, group.size)
        ]
        return max(units) if units else 10**9

    def _remaining_stack_capacity_for_group_bay(self, group: SmallBoxGroup, bay_key: str, state: dict) -> int:
        capacity = 10**9
        row_mix_key = self._row_mix_key_for_group(group)
        for footprint_key in self._placement_footprint_keys(bay_key, group.size):
            unit_capacity = self._stack_unit_capacity_for_group(footprint_key, group.size, group)
            port_stack_count = self._stack_count_for_group(footprint_key, group.size, group)
            total_stack_count = self._stack_count_for_bay_size(footprint_key, group.size)
            if unit_capacity <= 0 or port_stack_count <= 0 or total_stack_count <= 0:
                return 0
            port_key = (footprint_key, row_mix_key, group.size)
            total_key = (footprint_key, group.size)
            current_load = state["bay_port_size_load"][port_key]
            current_units = self._stack_units_for_quantity(footprint_key, group.size, group, current_load)
            other_units = state["bay_stack_used"][total_key] - current_units
            capacity = min(capacity, port_stack_count * unit_capacity - current_load)
            capacity = min(capacity, max(0, total_stack_count - other_units) * unit_capacity - current_load)
        return max(0, int(capacity))

    def _apply_stack_usage_to_state(self, group: SmallBoxGroup, bay_key: str, quantity: int, state: dict) -> None:
        row_mix_key = self._row_mix_key_for_group(group)
        for footprint_key in self._placement_footprint_keys(bay_key, group.size):
            port_key = (footprint_key, row_mix_key, group.size)
            total_key = (footprint_key, group.size)
            before = state["bay_port_size_load"][port_key]
            before_units = self._stack_units_for_quantity(footprint_key, group.size, group, before)
            after = before + quantity
            after_units = self._stack_units_for_quantity(footprint_key, group.size, group, after)
            state["bay_port_size_load"][port_key] = after
            state["bay_stack_used"][total_key] += after_units - before_units

    def _add_column(
        self,
        group: SmallBoxGroup,
        bay_key: str,
        quantity: int,
        base_cost: float,
        row_allocation: tuple[tuple[str, str, int], ...] | None = None,
        state: dict | None = None,
    ) -> bool:
        if quantity <= 0:
            return False
        if not self._user_bay_policy_allows(group, bay_key):
            return False
        stack_units = self._column_stack_units(group, bay_key, quantity)
        if stack_units >= 10**9:
            return False
        allocations = [self._row_allocation_signature(row_allocation)] if row_allocation else self._row_allocation_patterns_for_column(group, bay_key, quantity, state=state)
        allocations = [allocation for allocation in allocations if allocation]
        if not allocations:
            return False
        bay = self.bays[bay_key]
        block_id = self.block_by_bay.get((bay.area_no, bay_key), "")
        added = False
        triplet = (group.group_id, bay_key, quantity)
        for allocation in allocations:
            key = triplet + (allocation,)
            if key in self._column_keys:
                continue
            column_id = f"C{len(self._columns) + 1:07d}"
            col = PlacementColumn(
                column_id=column_id,
                group_id=group.group_id,
                demand_source=self.group_source.get(group.group_id, "document"),
                voyage_id=group.voyage_id,
                flow=group.status,
                port=group.port,
                size=group.size,
                big_plan_size=self._big_plan_size(group.size),
                height=group.height,
                weight_class=group.weight_class,
                special_stow_code=group.special_stow_code,
                attributes=dict(getattr(group, "attributes", {}) or {}),
                area_no=bay.area_no,
                bay_key=bay_key,
                bay_no=bay.bay_no,
                block_id=block_id,
                block_bays=self.block_bay_nos.get(block_id, ()),
                quantity=quantity,
                stack_units=stack_units,
                row_allocation=allocation,
                quota_key=self._quota_key(group, bay.area_no),
                coarse_key=self._coarse_key(group),
                fine_key=self._fine_key(group),
                coarse_cluster_key=self._coarse_cluster_key(group),
                fine_cluster_key=self._fine_cluster_key(group),
                intrinsic_cost=base_cost,
            )
            idx = len(self._columns)
            self._columns.append(col)
            self._column_keys.add(key)
            self._column_indices_by_triplet[triplet].append(idx)
            added = True
        if row_allocation is None and state is None:
            self._default_column_triplets.add(triplet)
        return added

    def _greedy_fallback(self) -> tuple[Counter[int], Counter[str]]:
        selected, unplaced, _stats = self._staged_repair_selected_solution(Counter())
        return selected, unplaced

    def _repair_selected_solution(
        self,
        selected: Counter[int],
        allow_new_columns: bool = True,
    ) -> tuple[Counter[int], Counter[str]]:
        repaired, state, placed = self._selection_state(selected)
        unplaced: Counter[str] = Counter()
        enforce_medium_plan_quota = self._repair_enforces_medium_plan_quota()
        for group in self.groups:
            remaining = int(group.demand) - int(placed.get(group.group_id, 0))
            while remaining > 0:
                choice = self._best_repair_column(
                    group,
                    state,
                    remaining,
                    repaired,
                    allow_new_columns=allow_new_columns,
                    enforce_medium_plan_quota=enforce_medium_plan_quota,
                )
                if choice is None:
                    break
                idx, col = choice
                self._apply_column_to_state(col, state)
                repaired[idx] = 1
                placed[group.group_id] += col.quantity
                remaining -= col.quantity
            if remaining > 0:
                unplaced[group.group_id] = remaining
        return repaired, unplaced

    def _staged_repair_selected_solution(
        self,
        selected: Counter[int],
        allow_new_columns: bool = True,
        stop_after_zero: bool = False,
        deadline: float | None = None,
        min_remaining_for_secondary: float = 0.0,
    ) -> tuple[Counter[int], Counter[str], dict]:
        candidates: list[tuple[str, Counter[int], Counter[str], list[dict]]] = []
        skipped_candidates: list[dict] = []
        stop_reason = "all_candidates_evaluated"
        repair_variants: list[tuple[str, str, list[SmallBoxGroup], bool]] = []
        for label, group_order in self._staged_repair_group_orders():
            repair_variants.append((label, f"{label}_twenty_preferred", group_order, True))
            if self.twenty_run_windows:
                repair_variants.append((label, f"{label}_twenty_relaxed", group_order, False))
        preferred_rule_affected: dict[str, bool] = {}
        for order_index, (order_label, label, group_order, prefer_twenty_run_rule) in enumerate(repair_variants):
            if not prefer_twenty_run_rule and not preferred_rule_affected.get(order_label, True):
                skipped_candidates.append(
                    {
                        "candidate": label,
                        "skipped_reason": "twenty_rule_did_not_change_preferred_choices",
                    }
                )
                continue
            remaining_seconds = None if deadline is None else deadline - perf_counter()
            if (
                order_index > 0
                and remaining_seconds is not None
                and remaining_seconds <= float(min_remaining_for_secondary)
            ):
                skipped_candidates.append(
                    {
                        "candidate": label,
                        "skipped_reason": "time_budget",
                        "remaining_seconds": round(max(0.0, remaining_seconds), 3),
                    }
                )
                stop_reason = "time_budget"
                continue
            repaired, unplaced, stats = self._staged_repair_selected_solution_for_order(
                selected,
                group_order,
                allow_new_columns=allow_new_columns,
                prefer_twenty_run_rule=prefer_twenty_run_rule,
            )
            if prefer_twenty_run_rule:
                preferred_rule_affected[order_label] = any(
                    bool(stage_stats.get("twenty_run_rule_affected_choice"))
                    for stage_stats in stats
                )
            candidates.append((label, repaired, unplaced, stats))
            if (
                stop_after_zero
                and sum(unplaced.values()) <= 0
                and self._twenty_run_new_violation_count(repaired) <= 0
            ):
                stop_reason = "zero_unplaced"
                break
        label, best_selected, best_unplaced, best_stats = min(
            candidates,
            key=lambda item: self._solution_rank(item[1], item[2]),
        )
        return best_selected, best_unplaced, {
            "selected_candidate": label,
            "iterations": best_stats,
            "candidates": [
                {
                    "candidate": candidate_label,
                    "document_unplaced_boxes": self._document_unplaced_boxes(candidate_unplaced),
                    "unplaced_boxes": sum(candidate_unplaced.values()),
                    "twenty_consecutive_violation_windows": self._twenty_run_violation_count(candidate_selected),
                    "objective": round(self._selected_solution_energy(candidate_selected, candidate_unplaced), 6),
                    "selected_columns": sum(1 for qty in candidate_selected.values() if qty > 0),
                }
                for candidate_label, candidate_selected, candidate_unplaced, _candidate_stats in candidates
            ],
            "skipped_candidates": skipped_candidates,
            "stop_reason": stop_reason,
        }

    def _staged_repair_group_orders(self) -> list[tuple[str, list[SmallBoxGroup]]]:
        groups = list(self.groups)
        orders: list[tuple[str, list[SmallBoxGroup]]] = [
            ("coarse_concentration_first", self._coarse_concentration_repair_order(groups)),
            ("default", groups),
        ]
        unique: list[tuple[str, list[SmallBoxGroup]]] = []
        seen: set[tuple[str, ...]] = set()
        for label, ordered_groups in orders:
            signature = tuple(group.group_id for group in ordered_groups)
            if signature in seen:
                continue
            seen.add(signature)
            unique.append((label, ordered_groups))
        return unique

    def _coarse_concentration_repair_order(self, groups: list[SmallBoxGroup]) -> list[SmallBoxGroup]:
        by_coarse: defaultdict[tuple[str, ...], list[SmallBoxGroup]] = defaultdict(list)
        for group in groups:
            by_coarse[self._coarse_cluster_key(group)].append(group)

        threshold = max(0, int(self.config.medium_concentrated_group_threshold or 0))

        def coarse_rank(coarse_key: tuple[str, ...]) -> tuple:
            demand = int(self._coarse_metric_demand(coarse_key))
            small_group = threshold > 0 and demand <= threshold
            groups_for_key = by_coarse.get(coarse_key, [])
            representative = groups_for_key[0] if groups_for_key else None
            size_rank = min((SIZE_ORDER.get(group.size, 3) for group in groups_for_key), default=3)
            quota_bucket_count = 0
            if representative is not None:
                buckets = self._coarse_quota_buckets(coarse_key)
                quota_bucket_count = sum(
                    1
                    for (quota_voyage, quota_flow, _area_no, quota_size), quota in self.quota_by_key.items()
                    if (quota_voyage, quota_flow, quota_size) in buckets and quota > 0
                )
            return (
                0 if small_group else 1,
                demand if small_group else -demand,
                size_rank,
                quota_bucket_count,
                coarse_key,
            )

        ordered: list[SmallBoxGroup] = []
        for coarse_key in sorted(by_coarse, key=coarse_rank):
            ordered.extend(
                sorted(
                    by_coarse[coarse_key],
                    key=lambda group: (
                        self._source_rank_for_group(group),
                        -int(group.demand),
                        group.height,
                        group.weight_class,
                        group.group_id,
                    ),
                )
            )
        return ordered

    def _staged_repair_selected_solution_for_order(
        self,
        selected: Counter[int],
        group_order: list[SmallBoxGroup],
        allow_new_columns: bool = True,
        prefer_twenty_run_rule: bool = True,
    ) -> tuple[Counter[int], Counter[str], list[dict]]:
        repaired, state, placed = self._selection_state(selected)
        stages = [
            ("stage1a", True, True),
            ("stage1b", False, True),
            ("stage2", False, True),
            ("stage3", False, False),
        ]
        base_enforce_medium_plan_quota = self._repair_enforces_medium_plan_quota()
        stats: list[dict] = []
        for stage, enforce_quota, stage_enforce_medium_plan_quota in stages:
            enforce_medium_plan_quota = base_enforce_medium_plan_quota and stage_enforce_medium_plan_quota
            before_unplaced = sum(max(0, int(group.demand) - int(placed.get(group.group_id, 0))) for group in self.groups)
            if before_unplaced <= 0:
                break
            before_columns = len(self._columns)
            before_selected = sum(1 for qty in repaired.values() if qty > 0)
            for group in group_order:
                remaining = int(group.demand) - int(placed.get(group.group_id, 0))
                while remaining > 0:
                    choice = self._best_repair_column(
                        group,
                        state,
                        remaining,
                        repaired,
                        allow_new_columns=allow_new_columns,
                        stage=stage,
                        enforce_quota=enforce_quota,
                        enforce_medium_plan_quota=enforce_medium_plan_quota,
                        prefer_twenty_run_rule=prefer_twenty_run_rule,
                    )
                    if choice is None:
                        break
                    idx, col = choice
                    self._apply_column_to_state(col, state)
                    repaired[idx] = 1
                    placed[group.group_id] += col.quantity
                    remaining -= col.quantity
            after_unplaced = sum(max(0, int(group.demand) - int(placed.get(group.group_id, 0))) for group in self.groups)
            stats.append(
                {
                    "stage": stage,
                    "enforce_big_plan_quota": enforce_quota,
                    "enforce_medium_plan_quota": enforce_medium_plan_quota,
                    "scope": self._staged_repair_stage_scope_label(stage),
                    "before_unplaced_boxes": before_unplaced,
                    "after_unplaced_boxes": after_unplaced,
                    "placed_boxes": before_unplaced - after_unplaced,
                    "added_columns": len(self._columns) - before_columns,
                    "added_selected_columns": sum(1 for qty in repaired.values() if qty > 0) - before_selected,
                    "twenty_run_rule_affected_choice": bool(
                        state.get("twenty_run_rule_affected_choice", False)
                    ),
                }
            )
        unplaced = Counter(
            {
                group.group_id: int(group.demand) - int(placed.get(group.group_id, 0))
                for group in self.groups
                if int(group.demand) - int(placed.get(group.group_id, 0)) > 0
            }
        )
        return repaired, unplaced, stats

    @staticmethod
    def _staged_repair_stage_scope_label(stage: str) -> str:
        return {
            "stage1a": "strict_big_plan_quota",
            "stage1b": "same_group_big_plan_area",
            "stage2": "any_big_plan_area",
            "stage3": "compatible_yard_area",
        }.get(stage, stage)

    def _selection_state(self, selected: Counter[int]) -> tuple[Counter[int], dict, Counter[str]]:
        repaired: Counter[int] = Counter()
        placed: Counter[str] = Counter()
        state = self._empty_selection_state()
        for idx, chosen in sorted(selected.items()):
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            group = self.groups_by_id.get(col.group_id)
            if group is None:
                continue
            remaining = int(group.demand) - int(placed.get(group.group_id, 0))
            if remaining <= 0 or col.quantity > remaining:
                continue
            if not self._column_fits_state(col, state, remaining):
                continue
            self._apply_column_to_state(col, state)
            repaired[idx] = 1
            placed[group.group_id] += col.quantity
        return repaired, state, placed

    def _empty_selection_state(self) -> dict:
        return {
            "bay_load": Counter(),
            "bay_size_load": Counter(),
            "bay_port_size_load": Counter(),
            "bay_stack_used": Counter(),
            "row_load": Counter(),
            "row_size_load": Counter(),
            "row_used_attrs": {},
            "bay_used_size": {},
            "bay_used_attrs": {},
            "twenty_segment_used_bays": set(),
            "twenty_run_used_bays": set(self.existing_twenty_bays),
            "twenty_run_rule_affected_choice": False,
            "group_bay_used": set(),
            "used_group_area": set(),
            "used_group_block": set(),
            "used_coarse_area_block": set(),
            "used_coarse_area_bay": set(),
            "used_voyage_area": set(),
            "coarse_area_used": Counter(),
            "export_e_area_bay_count": Counter(),
            "export_e_area_bays_used": set(),
            "area_edge_has45": set(),
            "area_edge_has_non45": set(),
            "big_plan_quota_used": Counter(),
            "medium_plan_quota_used": Counter(),
            "medium_plan_bay_quota_used": Counter(),
        }

    def _repair_enforces_medium_plan_quota(self) -> bool:
        return not bool(getattr(self.config, "repair_can_exceed_medium_plan_quota", False))

    def _ensure_repair_columns(self, group: SmallBoxGroup, state: dict, remaining: int) -> None:
        enforce_medium_plan_quota = self._repair_enforces_medium_plan_quota()
        for bay_key, _max_qty, base_cost in self._candidate_bays_for_group(group):
            capacity = self._remaining_capacity_for_group_bay(
                group,
                bay_key,
                state,
                remaining,
                enforce_medium_plan_quota=enforce_medium_plan_quota,
            )
            if capacity <= 0:
                continue
            self._add_column(group, bay_key, min(remaining, capacity), base_cost, state=state)

    def _best_repair_column(
        self,
        group: SmallBoxGroup,
        state: dict,
        remaining: int,
        selected: Counter[int],
        allow_new_columns: bool = True,
        stage: str | None = None,
        enforce_quota: bool = True,
        enforce_medium_plan_quota: bool = True,
        prefer_twenty_run_rule: bool = True,
    ) -> tuple[int, PlacementColumn] | None:
        best: tuple[tuple[int, int, float, int, int, str], str, int, float] | None = None
        relaxed_best: tuple[tuple[int, float, int, int, str], str, int, float] | None = None
        priority_rank_by_area: dict[str, int] = {}
        coarse_area_cost_by_area_qty: dict[tuple[str, int], float] = {}
        coarse_key = self._coarse_cluster_key(group)
        fine_key = self._fine_cluster_key(group)
        for bay_key, _max_qty, base_cost in self._candidate_bays_for_group(group, scope=stage):
            area_no = self.bays[bay_key].area_no
            capacity = self._remaining_capacity_for_group_bay(
                group,
                bay_key,
                state,
                remaining,
                enforce_quota=enforce_quota,
                enforce_medium_plan_quota=enforce_medium_plan_quota,
            )
            if capacity <= 0:
                continue
            qty = min(remaining, capacity)
            area_qty_key = (area_no, qty)
            if area_qty_key not in coarse_area_cost_by_area_qty:
                coarse_area_cost_by_area_qty[area_qty_key] = self._coarse_area_incremental_repair_cost(
                    coarse_key + (area_no,), qty, state["coarse_area_used"]
                )
            if area_no not in priority_rank_by_area:
                priority_rank_by_area[area_no] = self._priority_area_rank(group, area_no)
            common_score = (
                priority_rank_by_area[area_no],
                self._repair_column_score(
                    group, bay_key, qty, base_cost, state,
                    coarse_key=coarse_key,
                    fine_key=fine_key,
                    coarse_area_incremental_cost=coarse_area_cost_by_area_qty[area_qty_key],
                ),
                0 if qty >= remaining else 1,
                -qty,
                bay_key,
            )
            incremental_violation = (
                self._twenty_run_incremental_violation(group, bay_key, state)
                if prefer_twenty_run_rule
                else 0
            )
            score = (incremental_violation, *common_score)
            candidate = (score, bay_key, qty, base_cost)
            if best is None or candidate < best:
                best = candidate
            if prefer_twenty_run_rule:
                relaxed_candidate = (common_score, bay_key, qty, base_cost)
                if relaxed_best is None or relaxed_candidate < relaxed_best:
                    relaxed_best = relaxed_candidate
        if best is None:
            return None
        if (
            prefer_twenty_run_rule
            and relaxed_best is not None
            and (best[1], best[2]) != (relaxed_best[1], relaxed_best[2])
        ):
            state["twenty_run_rule_affected_choice"] = True
        _score, bay_key, qty, base_cost = best
        idx = self._column_index_for(group.group_id, bay_key, qty, state=state)
        if idx is None:
            if not allow_new_columns:
                return None
            self._add_column(group, bay_key, qty, base_cost, state=state)
            idx = self._column_index_for(group.group_id, bay_key, qty, state=state)
            if idx is None:
                return None
        if selected.get(idx, 0) > 0:
            return None
        col = self._columns[idx]
        # Capacity and policy were checked above against the unchanged state;
        # _column_index_for(..., state=state) checked this row allocation.
        return idx, col

    def _repair_column_score(
        self,
        group: SmallBoxGroup,
        bay_key: str,
        qty: int,
        base_cost: float,
        state: dict,
        coarse_key: tuple[str, ...] | None = None,
        fine_key: tuple[str, ...] | None = None,
        coarse_area_incremental_cost: float | None = None,
    ) -> float:
        bay = self.bays[bay_key]
        area_no = bay.area_no
        block_id = self.block_by_bay.get((area_no, bay_key), "")
        quota_key = self._quota_key(group, area_no)
        coarse_key = coarse_key if coarse_key is not None else self._coarse_cluster_key(group)
        fine_key = fine_key if fine_key is not None else self._fine_cluster_key(group)
        coarse_area_key = coarse_key + (area_no,)

        score = (
            base_cost
            + self.config.small_plan_group_bay_split_penalty
        )

        if (fine_key, area_no) not in state["used_group_area"]:
            score += self.config.small_plan_group_area_split_penalty
        if block_id and (fine_key, block_id) not in state["used_group_block"]:
            score += self.config.small_plan_group_block_split_penalty
        if block_id and coarse_key + (area_no, block_id) not in state["used_coarse_area_block"]:
            score += self.config.small_plan_coarse_area_block_split_penalty
        if coarse_key + (area_no, bay_key) not in state["used_coarse_area_bay"]:
            score += self.config.small_plan_coarse_area_bay_split_penalty
        if self._user_bay_policy_requires(group, bay_key):
            score -= float(self.config.required_area_reward)
        if (group.voyage_id, area_no) not in state["used_voyage_area"]:
            score += self._voyage_area_cost(group.voyage_id, area_no)

        before_quota = state["big_plan_quota_used"][quota_key]
        target = self._area_size_target(group.voyage_id, group.status, area_no, self._big_plan_size(group.size))
        score += self.config.big_plan_area_deviation_penalty * (
            abs(before_quota + qty - target) - abs(before_quota - target)
        )
        score += (
            coarse_area_incremental_cost
            if coarse_area_incremental_cost is not None
            else self._coarse_area_incremental_repair_cost(coarse_area_key, qty, state["coarse_area_used"])
        )
        score += self._twenty_bay_state_cost(group, bay_key, state)
        return float(score)

    def _coarse_area_incremental_repair_cost(
        self,
        coarse_area_key: tuple[str, ...],
        qty: int,
        coarse_area_used: Counter[tuple[str, ...]],
    ) -> float:
        coarse_key = tuple(coarse_area_key[:-1])
        area_no = coarse_area_key[-1]
        quantities = {
            key[-1]: float(value)
            for key, value in coarse_area_used.items()
            if tuple(key[:-1]) == coarse_key and value > 0
        }
        before = self._coarse_area_distribution_energy(coarse_key, quantities)
        quantities[area_no] = quantities.get(area_no, 0.0) + float(qty)
        after = self._coarse_area_distribution_energy(coarse_key, quantities)
        return after - before

    def _coarse_area_distribution_energy(
        self,
        coarse_key: tuple[str, ...],
        quantities_by_area: dict[str, float],
    ) -> float:
        quantities = [qty for qty in quantities_by_area.values() if qty > 0]
        if not quantities:
            return 0.0
        if self._prefers_concentrated_coarse_key(coarse_key):
            return (
                self.config.medium_small_group_area_split_penalty * max(0, len(quantities) - 1)
                - self.config.medium_small_group_fragment_penalty * max(quantities)
            )

        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        energy = 0.0
        demand = max(1, int(self._coarse_metric_demand(coarse_key, sum(quantities))))
        energy += self.config.medium_large_group_area_open_penalty * max(
            0,
            len(quantities) - self._target_large_group_area_count(coarse_key, demand),
        )
        target_boxes = max(1, int(self.config.medium_large_group_target_area_boxes or 1))
        energy += self.config.medium_large_group_area_excess_penalty * sum(
            max(0.0, qty - target_boxes) for qty in quantities
        )
        if min_boxes > 0:
            small_area_penalty = self.config.medium_large_group_small_area_penalty / max(1.0, min_boxes)
            energy += sum(small_area_penalty * max(0.0, min_boxes - qty) for qty in quantities if qty > 0)

        if len(quantities) > 1:
            pair_penalty = self.config.group_area_balance_penalty / max(1.0, demand) / max(1, len(quantities) - 1)
            for left_index, left in enumerate(quantities):
                for right in quantities[left_index + 1 :]:
                    energy += pair_penalty * abs(left - right)
        return float(energy)

    def _column_index_for(self, group_id: str, bay_key: str, quantity: int, state: dict | None = None) -> int | None:
        for idx in self._column_indices_by_triplet.get((group_id, bay_key, quantity), ()):
            col = self._columns[idx]
            if state is not None and not self._column_row_allocation_fits_state(col, state):
                continue
            return idx
        return None

    def _column_fits_state(
        self,
        col: PlacementColumn,
        state: dict,
        remaining: int,
        enforce_quota: bool = True,
        enforce_medium_plan_quota: bool = True,
    ) -> bool:
        group = self.groups_by_id.get(col.group_id)
        if group is None or col.quantity <= 0 or col.quantity > remaining:
            return False
        if not self._column_row_allocation_fits_state(col, state):
            return False
        return col.quantity <= self._remaining_capacity_for_group_bay(
            group,
            col.bay_key,
            state,
            remaining,
            enforce_quota=enforce_quota,
            enforce_medium_plan_quota=enforce_medium_plan_quota,
        )

    def _column_row_allocation_fits_state(self, col: PlacementColumn, state: dict) -> bool:
        group = self.groups_by_id.get(col.group_id)
        if group is None:
            return False
        by_footprint: Counter[str] = Counter()
        for footprint_key, row_no, qty in col.row_allocation:
            qty = int(qty)
            if qty <= 0:
                continue
            bay = self.bays[footprint_key]
            row_key = (footprint_key, row_no)
            row_size_key = (footprint_key, row_no, col.size)
            physical_cap = int(bay.row_physical_capacity.get(row_no, bay.physical_capacity))
            size_cap = int(bay.row_cap_by_size.get(col.size, {}).get(row_no, bay.cap_by_size.get(col.size, 0)))
            if state["row_load"][row_key] + qty > physical_cap:
                return False
            if state["row_size_load"][row_size_key] + qty > size_cap:
                return False
            if not self._row_existing_attrs_allow_group(bay, row_no, group):
                return False
            for attr in self._row_no_mix_attrs_for_group(group):
                value = self._column_attr_value(col, attr)
                state_key = self._row_state_attr_key(footprint_key, row_no, attr, col.voyage_id)
                used = state["row_used_attrs"].get(state_key, value)
                if used != value:
                    return False
            by_footprint[footprint_key] += qty
        return bool(by_footprint) and all(qty == col.quantity for qty in by_footprint.values())

    def _remaining_capacity_for_group_bay(
        self,
        group: SmallBoxGroup,
        bay_key: str,
        state: dict,
        remaining: int,
        enforce_quota: bool = True,
        enforce_medium_plan_quota: bool = True,
    ) -> int:
        if (group.group_id, bay_key) in state["group_bay_used"]:
            return 0
        if not self._user_bay_policy_allows(group, bay_key):
            return 0
        bay = self.bays[bay_key]
        footprint = self._placement_footprint_keys(bay_key, group.size)
        if not footprint:
            return 0
        if not self._bay_state_attrs_allow_group(group, footprint, state):
            return 0
        e_area_limit = self._export_e_area_max_bays()
        if e_area_limit is not None and self._is_export_e_group_area(group, bay.area_no):
            e_area_key = (group.voyage_id, bay.area_no)
            e_bay_key = e_area_key + (bay_key,)
            if (
                e_bay_key not in state["export_e_area_bays_used"]
                and state["export_e_area_bay_count"][e_area_key] >= e_area_limit
            ):
                return 0
        is_edge = bay_key in self.area_edge_bays.get(bay.area_no, set())
        if is_edge and group.size == "45" and bay.area_no in state["area_edge_has_non45"]:
            return 0
        if is_edge and group.size != "45" and bay.area_no in state["area_edge_has45"]:
            return 0
        capacity = int(remaining)
        for key in footprint:
            capacity = min(capacity, self.bays[key].physical_capacity - state["bay_load"][key])
        capacity = min(capacity, bay.cap_by_size.get(group.size, 0) - state["bay_size_load"][(bay_key, group.size)])
        capacity = min(capacity, self._row_capacity_for_column(group, bay_key, state=state))
        quota_key = self._quota_key(group, bay.area_no)
        quota = self.quota_by_key.get(quota_key, 0)
        if enforce_quota and quota > 0:
            capacity = min(capacity, quota - state["big_plan_quota_used"][quota_key])
        if enforce_medium_plan_quota and self.config.medium_plan_quota is not None:
            coarse_area_key = self._coarse_key(group) + (bay.area_no,)
            capacity = min(
                capacity,
                self._medium_plan_quota[coarse_area_key] - state["medium_plan_quota_used"][coarse_area_key],
            )
        if enforce_medium_plan_quota and self.config.medium_plan_bay_quota is not None:
            coarse_bay_key = self._coarse_key(group) + (bay.area_no, bay_key)
            capacity = min(
                capacity,
                self._medium_plan_bay_quota[coarse_bay_key]
                - state["medium_plan_bay_quota_used"][coarse_bay_key],
            )
        return max(0, int(capacity))

    def _apply_column_to_state(self, col: PlacementColumn, state: dict) -> None:
        footprint = self._placement_footprint_keys(col.bay_key, col.size)
        for key in footprint:
            state["bay_load"][key] += col.quantity
            state["bay_used_size"][key] = col.size
            for attr in self._bay_no_mix_attrs_for_column(col):
                state_key = self._bay_state_attr_key(key, attr, col.voyage_id)
                state.setdefault("bay_used_attrs", {})[state_key] = self._column_attr_value(col, attr)
        state["bay_size_load"][(col.bay_key, col.size)] += col.quantity
        group = self.groups_by_id.get(col.group_id)
        if group is not None:
            self._apply_stack_usage_to_state(group, col.bay_key, col.quantity, state)
        for footprint_key, row_no, qty in col.row_allocation:
            qty = int(qty)
            if qty <= 0:
                continue
            state["row_load"][(footprint_key, row_no)] += qty
            state["row_size_load"][(footprint_key, row_no, col.size)] += qty
            for attr in self._row_no_mix_attrs_for_column(col):
                state_key = self._row_state_attr_key(footprint_key, row_no, attr, col.voyage_id)
                state["row_used_attrs"][state_key] = self._column_attr_value(col, attr)
        state["group_bay_used"].add((col.group_id, col.bay_key))
        state["used_group_area"].add((col.fine_cluster_key, col.area_no))
        if col.block_id:
            state["used_group_block"].add((col.fine_cluster_key, col.block_id))
            state["used_coarse_area_block"].add(col.coarse_cluster_key + (col.area_no, col.block_id))
        state["used_coarse_area_bay"].add(col.coarse_cluster_key + (col.area_no, col.bay_key))
        state["used_voyage_area"].add((col.voyage_id, col.area_no))
        state["coarse_area_used"][col.coarse_cluster_key + (col.area_no,)] += col.quantity
        if self._is_export_e_column(col):
            e_area_key = (col.voyage_id, col.area_no)
            e_bay_key = e_area_key + (col.bay_key,)
            if e_bay_key not in state["export_e_area_bays_used"]:
                state["export_e_area_bays_used"].add(e_bay_key)
                state["export_e_area_bay_count"][e_area_key] += 1
        if col.bay_key in self.area_edge_bays.get(col.area_no, set()):
            if col.size == "45":
                state["area_edge_has45"].add(col.area_no)
            else:
                state["area_edge_has_non45"].add(col.area_no)
        if col.size == "20":
            state.setdefault("twenty_run_used_bays", set(self.existing_twenty_bays)).add(col.bay_key)
            segment = self._large_segment_key_for_20_bay(col.bay_key)
            if segment is not None:
                state.setdefault("twenty_segment_used_bays", set()).add(col.bay_key)
        state["big_plan_quota_used"][col.quota_key] += col.quantity
        state["medium_plan_quota_used"][col.coarse_key + (col.area_no,)] += col.quantity
        state["medium_plan_bay_quota_used"][col.coarse_key + (col.area_no, col.bay_key)] += col.quantity

    def _legacy_greedy_fallback(self) -> tuple[Counter[int], Counter[str]]:
        selected: Counter[int] = Counter()
        unplaced: Counter[str] = Counter()
        bay_load: Counter[str] = Counter()
        bay_size_load: Counter[tuple[str, str]] = Counter()
        bay_used_size: dict[str, str] = {}
        bay_used_attrs: dict[tuple[str, str, str], str] = {}
        group_bay_used: set[tuple[str, str]] = set()
        area_edge_has45: set[str] = set()
        area_edge_has_non45: set[str] = set()
        export_e_area_bay_count: Counter[tuple[str, str]] = Counter()
        export_e_area_bays_used: set[tuple[str, str, str]] = set()
        medium_plan_quota = self._medium_plan_quota
        medium_plan_bay_quota = self._medium_plan_bay_quota
        big_plan_quota_used: Counter[tuple[str, str, str, str]] = Counter()
        medium_plan_quota_used: Counter[tuple[str, ...]] = Counter()
        medium_plan_bay_quota_used: Counter[tuple[str, ...]] = Counter()
        columns_by_group: defaultdict[str, list[tuple[int, PlacementColumn]]] = defaultdict(list)
        for idx, col in enumerate(self._columns):
            columns_by_group[col.group_id].append((idx, col))
        for group in self.groups:
            remaining = group.demand
            for idx, col in sorted(columns_by_group[group.group_id], key=lambda item: (item[1].intrinsic_cost, -item[1].quantity)):
                if remaining <= 0:
                    break
                bay = self.bays[col.bay_key]
                if (col.group_id, col.bay_key) in group_bay_used:
                    continue
                footprint = self._placement_footprint_keys(col.bay_key, col.size)
                if not footprint:
                    continue
                bay_attr_conflict = False
                for key in footprint:
                    for attr in self._bay_no_mix_attrs_for_column(col):
                        state_key = self._bay_state_attr_key(key, attr, col.voyage_id)
                        value = self._column_attr_value(col, attr)
                        if bay_used_attrs.get(state_key, value) != value:
                            bay_attr_conflict = True
                            break
                    if bay_attr_conflict:
                        break
                if bay_attr_conflict:
                    continue
                if any(bay_load[key] + col.quantity > self.bays[key].physical_capacity for key in footprint):
                    continue
                if bay_size_load[(col.bay_key, col.size)] + col.quantity > bay.cap_by_size.get(col.size, 0):
                    continue
                quota = self.quota_by_key.get(col.quota_key, 0)
                if quota > 0 and big_plan_quota_used[col.quota_key] + col.quantity > quota:
                    continue
                coarse_area_key = col.coarse_key + (col.area_no,)
                if self.config.medium_plan_quota is not None and (
                    medium_plan_quota_used[coarse_area_key] + col.quantity > medium_plan_quota[coarse_area_key]
                ):
                    continue
                coarse_bay_key = col.coarse_key + (col.area_no, col.bay_key)
                if self.config.medium_plan_bay_quota is not None and (
                    medium_plan_bay_quota_used[coarse_bay_key] + col.quantity > medium_plan_bay_quota[coarse_bay_key]
                ):
                    continue
                e_area_limit = self._export_e_area_max_bays()
                if e_area_limit is not None and self._is_export_e_column(col):
                    e_area_key = (col.voyage_id, col.area_no)
                    e_bay_key = e_area_key + (col.bay_key,)
                    if e_bay_key not in export_e_area_bays_used and export_e_area_bay_count[e_area_key] >= e_area_limit:
                        continue
                if col.quantity > remaining:
                    continue
                is_edge = col.bay_key in self.area_edge_bays.get(col.area_no, set())
                if is_edge and col.size == "45" and col.area_no in area_edge_has_non45:
                    continue
                if is_edge and col.size != "45" and col.area_no in area_edge_has45:
                    continue
                selected[idx] = 1
                for key in footprint:
                    bay_load[key] += col.quantity
                    bay_used_size[key] = col.size
                    for attr in self._bay_no_mix_attrs_for_column(col):
                        state_key = self._bay_state_attr_key(key, attr, col.voyage_id)
                        bay_used_attrs[state_key] = self._column_attr_value(col, attr)
                bay_size_load[(col.bay_key, col.size)] += col.quantity
                group_bay_used.add((col.group_id, col.bay_key))
                if is_edge and col.size == "45":
                    area_edge_has45.add(col.area_no)
                elif is_edge:
                    area_edge_has_non45.add(col.area_no)
                big_plan_quota_used[col.quota_key] += col.quantity
                medium_plan_quota_used[coarse_area_key] += col.quantity
                medium_plan_bay_quota_used[coarse_bay_key] += col.quantity
                if self._is_export_e_column(col):
                    e_area_key = (col.voyage_id, col.area_no)
                    e_bay_key = e_area_key + (col.bay_key,)
                    if e_bay_key not in export_e_area_bays_used:
                        export_e_area_bays_used.add(e_bay_key)
                        export_e_area_bay_count[e_area_key] += 1
                remaining -= col.quantity
            if remaining > 0:
                unplaced[group.group_id] = remaining
        return selected, unplaced

    def _candidate_bays_for_group(self, group: SmallBoxGroup, scope: str | None = None) -> list[tuple[str, int, float]]:
        scope = scope or self._candidate_scope
        cache_key = (scope, group.group_id)
        cached = self._candidate_cache.get(cache_key)
        if cached is not None:
            return cached
        out: list[tuple[str, int, float]] = []
        for area_no in self._candidate_areas_for_group(group, scope=scope):
            for bay_key in self.bays_by_area.get(area_no, []):
                if not self._user_bay_policy_allows(group, bay_key):
                    continue
                max_qty = self._max_quantity_in_bay(group, bay_key)
                if max_qty <= 0:
                    continue
                cost = self._column_base_cost(group, bay_key)
                out.append((bay_key, min(max_qty, group.demand), cost))
        if self._prefers_concentrated_coarse_key(self._coarse_cluster_key(group)):
            out.sort(
                key=lambda item: (
                    self._priority_area_rank(group, self.bays[item[0]].area_no),
                    0 if self._user_bay_policy_requires(group, item[0]) else 1,
                    self._existing_coarse_bay_rank(group, item[0]),
                    self._export_e_area_size_rank(group, self.bays[item[0]].area_no),
                    self._area_fallback_tier_for_group(group, self.bays[item[0]].area_no),
                    self._concentrated_area_sort_key(group, self.bays[item[0]].area_no),
                    item[2],
                    -item[1],
                    self.bays[item[0]].bay_order,
                )
            )
        else:
            out.sort(
                key=lambda item: (
                    self._priority_area_rank(group, self.bays[item[0]].area_no),
                    0 if self._user_bay_policy_requires(group, item[0]) else 1,
                    self._existing_coarse_bay_rank(group, item[0]),
                    self._export_e_area_size_rank(group, self.bays[item[0]].area_no),
                    self._area_fallback_tier_for_group(group, self.bays[item[0]].area_no),
                    item[2],
                    -item[1],
                    self.bays[item[0]].area_no,
                    self.bays[item[0]].bay_order,
                )
            )
        out = self._limit_candidate_bays(group, out, scope=scope)
        self._candidate_cache[cache_key] = out
        return out

    def _pricing_candidate_bay_limit(
        self,
        group: SmallBoxGroup,
        available_count: int,
        iteration: int,
        group_dual_value: float,
        max_group_dual: float,
        lp_unplaced: Counter[str],
    ) -> tuple[int, str]:
        if available_count <= 0:
            return 0, "empty"
        if not bool(getattr(self.config, "adaptive_pricing_enabled", True)):
            return available_count, "full"

        base = max(1, int(getattr(self.config, "pricing_candidate_bays_initial", 0) or 0))
        growth = max(0, int(getattr(self.config, "pricing_candidate_bays_growth_per_iteration", 0) or 0))
        limit = base + growth * max(0, int(iteration))
        reason = "adaptive_base"

        if lp_unplaced.get(group.group_id, 0) > 1e-6:
            limit = max(
                limit,
                max(1, int(getattr(self.config, "pricing_candidate_bays_unplaced", 0) or 0)),
            )
            reason = "lp_unplaced"
        elif max_group_dual > 1e-9 and abs(float(group_dual_value)) >= 0.75 * max_group_dual:
            limit = max(
                limit,
                max(1, int(getattr(self.config, "pricing_candidate_bays_high_dual", 0) or 0)),
            )
            reason = "high_dual"

        return min(available_count, max(1, int(limit))), reason

    def _candidate_areas_for_group(self, group: SmallBoxGroup, scope: str | None = None) -> list[str]:
        scope = scope or self._candidate_scope

        return sorted(
            [
                area_no
                for area_no in self.bays_by_area
                if self._candidate_area_base_scope(group, area_no, scope)
                and self._user_area_policy_allows(group.voyage_id, area_no)
                and self._area_supports_group_flow(group, area_no)
                and self._port_sail_area_policy_allows(group, area_no)
            ],
            key=lambda area_no: (
                self._priority_area_rank(group, area_no),
                self._export_e_area_size_rank(group, area_no),
                self._area_fallback_tier_for_group(group, area_no),
                area_no,
            ),
        )

    def _candidate_area_base_scope(self, group: SmallBoxGroup, area_no: str, scope: str) -> bool:
        big_size = self._big_plan_size(group.size)
        if scope in {"stage0", "stage1a"}:
            return self.quota_by_key.get((group.voyage_id, group.status, area_no, big_size), 0) > 0
        if scope == "stage1b":
            return self._is_big_plan_area_for_group(group, area_no)
        if scope == "stage2":
            return self._is_any_big_plan_area(area_no)
        return True

    def _priority_area_rank(self, group: SmallBoxGroup, area_no: str) -> int:
        priority = getattr(self.problem, "user_voyage_area_priority", {}).get(group.voyage_id, set())
        if not priority:
            return 0
        return 0 if area_no in priority else 1

    @staticmethod
    def _is_e_area(area_no: object) -> bool:
        return str(area_no or "").strip().upper().startswith("E")

    def _is_export_e_group_area(self, group: SmallBoxGroup, area_no: str) -> bool:
        return self._is_export_voyage(group.voyage_id) and self._is_e_area(area_no)

    def _is_export_e_column(self, col: PlacementColumn) -> bool:
        return self._is_export_voyage(col.voyage_id) and self._is_e_area(col.area_no)

    def _export_e_area_max_bays(self) -> int | None:
        limit = int(getattr(self.config, "export_e_area_max_bays_per_voyage_area", 2) or 0)
        if limit < 0:
            return None
        return limit

    def _export_e_area_size_rank(self, group: SmallBoxGroup, area_no: str) -> int:
        if not self._is_export_e_group_area(group, area_no):
            return 0
        return 0 if group.size in {"40", "45"} else 1

    def _export_e_area_non_40_penalty(self, group: SmallBoxGroup, area_no: str) -> float:
        if self._export_e_area_size_rank(group, area_no) <= 0:
            return 0.0
        return max(0.0, float(getattr(self.config, "export_e_area_non_40_penalty", 0.0) or 0.0))

    def _user_area_policy_allows(self, voyage_id: str, area_no: str) -> bool:
        allow = getattr(self.problem, "user_voyage_area_allowlist", {}).get(voyage_id, set())
        block = getattr(self.problem, "user_voyage_area_blocklist", {}).get(voyage_id, set())
        if area_no in block:
            return False
        if allow and area_no not in allow:
            return False
        return True

    def _port_sail_area_policy_allows(self, group: SmallBoxGroup, area_no: str) -> bool:
        if self._is_export_voyage(group.voyage_id):
            return True
        allowlist = getattr(group, "area_allowlist", None)
        if allowlist is None:
            return True
        area_code = str(area_no or "").strip().upper()
        return area_code in {str(area or "").strip().upper() for area in allowlist}

    def _user_bay_policy_allows(self, group: SmallBoxGroup, bay_key: str) -> bool:
        voyage_allowlists = getattr(self.problem, "user_voyage_bay_allowlist", {})
        if group.voyage_id in voyage_allowlists and bay_key not in voyage_allowlists[group.voyage_id]:
            return False
        blocked = getattr(self.problem, "user_group_bay_blocklist", {}).get(group.group_id, set())
        return bay_key not in blocked

    def _user_bay_policy_requires(self, group: SmallBoxGroup, bay_key: str) -> bool:
        required = getattr(self.problem, "user_group_bay_requirements", {}).get(group.group_id, set())
        return bay_key in required

    def _area_supports_group_flow(self, group: SmallBoxGroup, area_no: str) -> bool:
        if self._is_big_plan_area_for_group(group, area_no):
            return True
        functions = self.problem.area_functions.get(area_no, set())
        return _area_flow(group.status) in functions

    def _limit_candidate_bays(
        self,
        group: SmallBoxGroup,
        candidates: list[tuple[str, int, float]],
        scope: str | None = None,
    ) -> list[tuple[str, int, float]]:
        limit = max(0, int(self.config.max_candidate_bays_per_group or 0))
        if limit <= 0 or len(candidates) <= limit:
            return candidates
        required_bays = getattr(self.problem, "user_group_bay_requirements", {}).get(group.group_id, set())
        required = [item for item in candidates if item[0] in required_bays]
        remaining_limit = max(0, limit - len(required))
        priority_areas = getattr(self.problem, "user_voyage_area_priority", {}).get(group.voyage_id, set())
        if priority_areas:
            priority = [
                item
                for item in candidates
                if item[0] not in required_bays and self.bays[item[0]].area_no in priority_areas
            ]
            non_priority = [
                item
                for item in candidates
                if item[0] not in required_bays and self.bays[item[0]].area_no not in priority_areas
            ]
            return required + priority[:remaining_limit] + non_priority[: max(0, remaining_limit - len(priority))]
        preferred_areas = set(self._area_weights(group))
        if not preferred_areas:
            return required + [item for item in candidates if item[0] not in required_bays][:remaining_limit]
        preferred = [item for item in candidates if item[0] not in required_bays and self.bays[item[0]].area_no in preferred_areas]
        fallback = [item for item in candidates if item[0] not in required_bays and self.bays[item[0]].area_no not in preferred_areas]
        if not fallback:
            return required + preferred[:remaining_limit]
        fallback_limit = min(len(fallback), max(1, remaining_limit // 4)) if remaining_limit > 0 else 0
        preferred_limit = max(0, remaining_limit - fallback_limit)
        return required + preferred[:preferred_limit] + fallback[:fallback_limit]

    def _max_quantity_in_bay(self, group: SmallBoxGroup, bay_key: str) -> int:
        bay = self.bays[bay_key]
        if bay.cap_by_size.get(group.size, 0) <= 0:
            return 0
        footprint = self._placement_footprint_keys(bay_key, group.size)
        if not footprint:
            return 0
        if not self._bay_existing_attrs_allow_group(group, footprint):
            return 0
        is_edge = bay_key in self.area_edge_bays.get(bay.area_no, set())
        if group.size == "20" and is_edge:
            return 0
        if group.size == "45" and not is_edge:
            return 0
        footprint_capacity = min(self.bays[key].physical_capacity for key in footprint)
        stack_capacity = min(
            self._stack_count_for_group(key, group.size, group) * self._stack_unit_capacity_for_group(key, group.size, group)
            for key in footprint
        )
        return max(0, min(group.demand, footprint_capacity, bay.cap_by_size.get(group.size, 0), stack_capacity))

    def _placement_footprint_keys(self, bay_key: str, size: str) -> tuple[str, ...]:
        bay = self.bays[bay_key]
        if size in {"40", "45"}:
            if not bay.large_bay_partner_key:
                return ()
            return (bay_key, bay.large_bay_partner_key)
        return (bay_key,)

    def _column_base_cost(self, group: SmallBoxGroup, bay_key: str) -> float:
        bay = self.bays[bay_key]
        cost = 0.0
        if not self.block_by_bay.get((bay.area_no, bay_key)):
            cost += self.config.non_preferred_block_penalty
        cost += self._export_e_area_non_40_penalty(group, bay.area_no)
        if bay.is_fallback_bay:
            cost += self.config.fallback_bay_penalty
        if self._user_bay_policy_requires(group, bay_key):
            cost -= float(self.config.required_area_reward)
        existing_bay_load = self._existing_coarse_bay_load_for_group(group, bay_key)
        if existing_bay_load > 0:
            cost -= float(self.config.existing_coarse_bay_reward) * min(5, existing_bay_load)
        else:
            same_distance = self._existing_same_coarse_bay_distance(group, bay_key)
            if same_distance is not None:
                cost -= self._existing_neighbor_reward(same_distance)
            else:
                existing_other_load = self._existing_other_coarse_bay_load_for_group(group, bay_key)
                if existing_other_load > 0:
                    cost += float(self.config.existing_other_coarse_bay_penalty) * min(5, existing_other_load)
                other_distance = self._existing_other_coarse_bay_distance(group, bay_key)
                if other_distance is not None:
                    cost += self._existing_other_neighbor_penalty(other_distance)
        if group.special_stow or group.pre_stow:
            cost -= 1.0
        cost += self._twenty_bay_static_cost(group, bay_key)
        return cost

    def _area_fallback_tier_penalty(self, group: SmallBoxGroup, area_no: str) -> float:
        tier = self._area_fallback_tier_for_group(group, area_no)
        if tier == 0:
            return 0.0
        return float(self.config.big_plan_fallback_tier_penalty) * tier

    def _is_big_plan_area_for_group(self, group: SmallBoxGroup, area_no: str) -> bool:
        if area_no in self.problem.assigned_areas.get((group.voyage_id, group.status), set()):
            return True
        if self._is_export_voyage(group.voyage_id):
            return any(area_no in self.problem.assigned_areas.get((group.voyage_id, flow), set()) for flow in EXPORT_FLOWS)
        return False

    def _is_any_big_plan_area(self, area_no: str) -> bool:
        return any(row.area_no == area_no and row.planned_boxes > 0 for row in self.problem.big_plan)

    def _area_fallback_tier_for_group(self, group: SmallBoxGroup, area_no: str) -> int:
        return self._area_fallback_tier_for_attrs(
            group.voyage_id,
            group.status,
            area_no,
            self._big_plan_size(group.size),
        )

    def _area_fallback_tier_for_attrs(self, voyage_id: str, flow: str, area_no: str, big_size: str) -> int:
        if self.quota_by_key.get((voyage_id, flow, area_no, big_size), 0) > 0:
            return 0
        if area_no in self.problem.assigned_areas.get((voyage_id, flow), set()):
            return 1
        if self._is_export_voyage(voyage_id) and any(
            area_no in self.problem.assigned_areas.get((voyage_id, compatible_flow), set())
            for compatible_flow in EXPORT_FLOWS
        ):
            return 1
        if self._is_any_big_plan_area(area_no):
            return 2
        return 3

    def _quantity_options(self, group: SmallBoxGroup, max_qty: int) -> list[int]:
        values = {1, max_qty, min(group.demand, max_qty)}
        if group.demand % max_qty:
            values.add(group.demand % max_qty)
        values.add(max(1, max_qty // 2))
        return sorted(qty for qty in values if 0 < qty <= max_qty)

    def _prepare_yard_indexes(self) -> None:
        for key, bay in self.bays.items():
            self.bays_by_area[bay.area_no].append(key)
        for keys in self.bays_by_area.values():
            keys.sort(key=lambda bay_key: self.bays[bay_key].bay_order)
        for area_no, keys in self.bays_by_area.items():
            if keys:
                edge_keys = {keys[0], keys[-1]}
                last_key = keys[-1]
                for bay_key in keys:
                    if self.bays[bay_key].large_bay_partner_key == last_key:
                        edge_keys.add(bay_key)
                self.area_edge_bays[area_no] = edge_keys
            blocks = self._six_bay_blocks_for_area(area_no, keys)
            self.block_members_by_area[area_no] = blocks
            for block_id, members in blocks.items():
                self.block_bay_nos[block_id] = tuple(self.bays[bay_key].bay_no for bay_key in members)
                for bay_key in members:
                    self.block_by_bay[(area_no, bay_key)] = block_id
        self._prepare_twenty_run_indexes()
        self._prepare_large_segment_preservation_indexes()
        heights_by_size: defaultdict[str, set[str]] = defaultdict(set)
        for group in self.groups:
            heights_by_size[group.size].add(group.height)
        for bay_key, bay in self.bays.items():
            is_edge = bay_key in self.area_edge_bays.get(bay.area_no, set())
            for size, heights in heights_by_size.items():
                if size == "20" and is_edge:
                    continue
                if size == "45" and not is_edge:
                    continue
                cap = bay.cap_by_size.get(size, 0)
                if cap <= 0:
                    continue
                footprint = self._placement_footprint_keys(bay_key, size)
                if not footprint:
                    continue
                for height in heights:
                    self.area_size_height_cap[(bay.area_no, size, height)] += cap

    def _prepare_twenty_run_indexes(self) -> None:
        self.twenty_run_segments.clear()
        self.twenty_run_windows.clear()
        self.twenty_run_windows_by_bay.clear()
        self.existing_twenty_bays = {
            bay_key
            for bay_key in self.bays
            if "20" in self._bay_existing_size_modes(bay_key)
        }
        max_consecutive = max(0, int(getattr(self.config, "twenty_max_consecutive_bays", 2) or 0))
        if max_consecutive <= 0:
            return

        def flush(segment: list[str]) -> None:
            if not segment:
                return
            segment_key = tuple(segment)
            self.twenty_run_segments.append(segment_key)
            window_size = max_consecutive + 1
            for start in range(0, len(segment_key) - window_size + 1):
                window = segment_key[start : start + window_size]
                self.twenty_run_windows.append(window)
                for bay_key in window:
                    self.twenty_run_windows_by_bay[bay_key].append(window)

        for _area_no, keys in self.bays_by_area.items():
            segment: list[str] = []
            for bay_key in keys:
                bay = self.bays[bay_key]
                can_hold_twenty = (
                    bay_key in self.existing_twenty_bays
                    or int(bay.cap_by_size.get("20", 0) or 0) > 0
                    or any(int(value or 0) > 0 for value in (bay.row_cap_by_size.get("20", {}) or {}).values())
                )
                if not can_hold_twenty:
                    flush(segment)
                    segment = []
                    continue
                if segment and not self._segment_bays_are_consecutive(segment[-1], bay_key):
                    flush(segment)
                    segment = []
                segment.append(bay_key)
            flush(segment)

    def _prepare_large_segment_preservation_indexes(self) -> None:
        self.large_segment_by_bay.clear()
        self.large_segment_base_pairs.clear()
        self.large_segment_static_loss_by_bay.clear()

        def flush(segment: list[str]) -> None:
            if len(segment) < 2:
                return
            segment_key = tuple(segment)
            base_pairs = self._segment_pair_capacity(segment_key)
            self.large_segment_base_pairs[segment_key] = base_pairs
            for key in segment_key:
                self.large_segment_by_bay[key] = segment_key
                self.large_segment_static_loss_by_bay[key] = max(
                    0,
                    base_pairs - self._segment_pair_capacity_after_removed(segment_key, {key}),
                )

        for area_no, keys in self.bays_by_area.items():
            segment: list[str] = []
            for bay_key in keys:
                if not self._bay_can_participate_in_large_segment(bay_key):
                    flush(segment)
                    segment = []
                    continue
                if segment and not self._segment_bays_are_consecutive(segment[-1], bay_key):
                    flush(segment)
                    segment = []
                segment.append(bay_key)
            flush(segment)

    def _bay_can_participate_in_large_segment(self, bay_key: str) -> bool:
        bay = self.bays.get(bay_key)
        if bay is None or int(getattr(bay, "physical_capacity", 0) or 0) <= 0:
            return False
        existing_sizes = self._bay_existing_size_modes(bay_key)
        if "20" in existing_sizes:
            return False
        if "40" in existing_sizes and "45" in existing_sizes:
            return False
        if existing_sizes & {"40", "45"}:
            return True
        if int(bay.cap_by_size.get("40", 0) or 0) > 0 or int(bay.cap_by_size.get("45", 0) or 0) > 0:
            return True
        for size in ("40", "45"):
            if any(int(value or 0) > 0 for value in (bay.row_cap_by_size.get(size, {}) or {}).values()):
                return True
        return False

    def _segment_bays_are_consecutive(self, left_key: str, right_key: str) -> bool:
        left = self.bays.get(left_key)
        right = self.bays.get(right_key)
        if left is None or right is None or left.area_no != right.area_no:
            return False
        try:
            return int(left.bay_no) + 2 == int(right.bay_no)
        except (TypeError, ValueError):
            return int(right.bay_order) - int(left.bay_order) == 1

    def _segment_pair_capacity(self, bay_keys: Iterable[str]) -> int:
        ordered = sorted(
            (key for key in bay_keys if key in self.bays),
            key=lambda key: (self.bays[key].area_no, self.bays[key].bay_order, key),
        )
        total = 0
        run_len = 0
        prev_key: str | None = None
        for key in ordered:
            if prev_key is not None and self._segment_bays_are_consecutive(prev_key, key):
                run_len += 1
            else:
                total += run_len // 2
                run_len = 1
            prev_key = key
        total += run_len // 2
        return total

    def _segment_pair_capacity_after_removed(self, segment: tuple[str, ...], removed_bays: set[str]) -> int:
        if not removed_bays:
            return self.large_segment_base_pairs.get(segment, self._segment_pair_capacity(segment))
        return self._segment_pair_capacity(key for key in segment if key not in removed_bays)

    def _large_segment_key_for_20_bay(self, bay_key: str) -> tuple[str, ...] | None:
        return self.large_segment_by_bay.get(bay_key)

    def _twenty_run_used_bays(self, selected: Counter[int] | None = None) -> set[str]:
        used = set(self.existing_twenty_bays)
        if selected is None:
            return used
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            if col.size == "20":
                used.add(col.bay_key)
        return used

    def _twenty_run_violation_count_for_bays(self, used_bays: set[str]) -> int:
        return sum(1 for window in self.twenty_run_windows if all(key in used_bays for key in window))

    def _twenty_run_violation_count(self, selected: Counter[int]) -> int:
        return self._twenty_run_violation_count_for_bays(self._twenty_run_used_bays(selected))

    def _twenty_run_new_violation_count(self, selected: Counter[int]) -> int:
        baseline = self._twenty_run_violation_count_for_bays(set(self.existing_twenty_bays))
        return max(0, self._twenty_run_violation_count(selected) - baseline)

    def _twenty_run_incremental_violation(
        self,
        group: SmallBoxGroup,
        bay_key: str,
        state: dict | None,
    ) -> int:
        if group.size != "20" or state is None:
            return 0
        used = state.get("twenty_run_used_bays", self.existing_twenty_bays)
        if bay_key in used:
            return 0
        return sum(
            1
            for window in self.twenty_run_windows_by_bay.get(bay_key, ())
            if all(key == bay_key or key in used for key in window)
        )

    def _twenty_run_diagnostics(self, selected: Counter[int]) -> dict:
        planned_twenty_bays = {
            self._columns[idx].bay_key
            for idx, chosen in selected.items()
            if chosen > 0 and 0 <= idx < len(self._columns) and self._columns[idx].size == "20"
        }
        final_used = self.existing_twenty_bays | planned_twenty_bays
        baseline_violations = self._twenty_run_violation_count_for_bays(set(self.existing_twenty_bays))
        final_violations = self._twenty_run_violation_count_for_bays(final_used)
        runs: list[dict] = []
        max_run_length = 0
        max_consecutive = max(0, int(getattr(self.config, "twenty_max_consecutive_bays", 2) or 0))
        for segment in self.twenty_run_segments:
            run: list[str] = []

            def flush_run() -> None:
                nonlocal max_run_length
                if not run:
                    return
                max_run_length = max(max_run_length, len(run))
                if len(run) > max_consecutive:
                    first = self.bays[run[0]]
                    runs.append(
                        {
                            "area_no": first.area_no,
                            "bay_nos": [self.bays[key].bay_no for key in run],
                            "bay_keys": list(run),
                            "length": len(run),
                            "excess_bays": len(run) - max_consecutive,
                            "existing_bay_nos": [
                                self.bays[key].bay_no for key in run if key in self.existing_twenty_bays
                            ],
                            "planned_bay_nos": [
                                self.bays[key].bay_no for key in run if key in planned_twenty_bays
                            ],
                        }
                    )

            for bay_key in segment:
                if bay_key in final_used:
                    run.append(bay_key)
                else:
                    flush_run()
                    run = []
            flush_run()
        return {
            "max_allowed_consecutive_bays": max_consecutive,
            "classification_scope": "global_bay_size_mode_existing_and_new",
            "existing_twenty_bay_count": len(self.existing_twenty_bays),
            "planned_twenty_bay_count": len(planned_twenty_bays),
            "baseline_violation_windows": baseline_violations,
            "final_violation_windows": final_violations,
            "new_violation_windows": max(0, final_violations - baseline_violations),
            "overlength_run_count": len(runs),
            "max_consecutive_twenty_bays": max_run_length,
            "overlength_runs": runs,
        }

    def _bay_existing_size_modes(self, bay_key: str) -> set[str]:
        bay = self.bays.get(bay_key)
        if bay is None:
            return set()
        return {str(size) for size in getattr(bay, "existing_size_modes", set()) if str(size)}

    def _twenty_segment_static_loss_for_bay(self, bay_key: str) -> int:
        return max(0, int(self.large_segment_static_loss_by_bay.get(bay_key, 0)))

    def _twenty_segment_incremental_loss(self, bay_key: str, state: dict | None) -> int:
        segment = self._large_segment_key_for_20_bay(bay_key)
        if segment is None:
            return 0
        used = set(state.get("twenty_segment_used_bays", set())) if state is not None else set()
        used_in_segment = {key for key in used if self.large_segment_by_bay.get(key) == segment}
        if bay_key in used_in_segment:
            return 0
        before = self._segment_pair_capacity_after_removed(segment, used_in_segment)
        after = self._segment_pair_capacity_after_removed(segment, used_in_segment | {bay_key})
        return max(0, before - after)

    def _twenty_segment_total_loss(self, used_bays_by_segment: dict[tuple[str, ...], set[str]]) -> int:
        loss = 0
        for segment, used_bays in used_bays_by_segment.items():
            base_pairs = self.large_segment_base_pairs.get(segment, self._segment_pair_capacity(segment))
            remaining_pairs = self._segment_pair_capacity_after_removed(segment, set(used_bays))
            loss += max(0, base_pairs - remaining_pairs)
        return loss

    def _twenty_bay_static_cost(self, group: SmallBoxGroup, bay_key: str) -> float:
        if group.size != "20":
            return 0.0
        reward = float(getattr(self.config, "twenty_isolated_bay_reward", 0.0) or 0.0)
        segment = self._large_segment_key_for_20_bay(bay_key)
        if segment is None:
            return 0.0 if self._bay_existing_size_modes(bay_key) else -reward
        loss = self._twenty_segment_static_loss_for_bay(bay_key)
        if loss <= 0:
            return -reward
        return self._twenty_segment_loss_penalty() * loss

    def _twenty_bay_state_cost(self, group: SmallBoxGroup, bay_key: str, state: dict | None) -> float:
        if group.size != "20":
            return 0.0
        reward = float(getattr(self.config, "twenty_isolated_bay_reward", 0.0) or 0.0)
        segment = self._large_segment_key_for_20_bay(bay_key)
        if segment is None:
            if self._bay_existing_size_modes(bay_key):
                return 0.0
            if state is not None and state.get("bay_load", Counter()).get(bay_key, 0) > 0:
                return 0.0
            return -reward
        loss = self._twenty_segment_incremental_loss(bay_key, state)
        if loss <= 0:
            used = set(state.get("twenty_segment_used_bays", set())) if state is not None else set()
            if any(self.large_segment_by_bay.get(key) == segment for key in used):
                return -float(getattr(self.config, "twenty_large_segment_used_zero_loss_reward", 0.0) or 0.0)
            return -reward
        return float(getattr(self.config, "twenty_large_segment_fresh_loss_penalty", 0.0) or 0.0) * loss

    def _twenty_segment_loss_penalty(self) -> float:
        return max(0.0, float(getattr(self.config, "twenty_large_segment_loss_penalty", 0.0) or 0.0))

    def _twenty_consecutive_violation_penalty(self) -> float:
        return max(0.0, float(getattr(self.config, "twenty_consecutive_violation_penalty", 0.0) or 0.0))

    def _prepare_quota(self) -> None:
        for (voyage_id, flow, area_no, big_size), qty in getattr(self.problem, "area_size_quota", {}).items():
            if qty > 0:
                self.quota_by_key[(voyage_id, flow, area_no, big_size)] += int(qty)
        if self.quota_by_key:
            return
        requested_keys = {
            (group.voyage_id, group.status, self._big_plan_size(group.size))
            for group in self.groups
        }
        for voyage_id, flow, big_size in requested_keys:
            compatible = ({flow} | set(EXPORT_FLOWS)) if self._is_export_voyage(voyage_id) else {flow}
            for row in self.problem.big_plan:
                row_size = self._big_plan_size(row.size_mode)
                if (
                    row.voyage_id == voyage_id
                    and row.flow in compatible
                    and (row_size == big_size or row.size_mode == "ALL")
                ):
                    self.quota_by_key[(voyage_id, flow, row.area_no, big_size)] += row.planned_boxes

    def _area_weights(self, group: SmallBoxGroup) -> Counter[str]:
        weights: Counter[str] = Counter()
        big_size = self._big_plan_size(group.size)
        for (voyage_id, flow, area_no, size), qty in self.quota_by_key.items():
            if voyage_id == group.voyage_id and flow == group.status and size == big_size and qty > 0:
                weights[area_no] += qty
        return weights

    def _quota_key(self, group: SmallBoxGroup, area_no: str) -> tuple[str, str, str, str]:
        return group.voyage_id, group.status, area_no, self._big_plan_size(group.size)

    def _configured_coarse_attrs_for_group(self, group: SmallBoxGroup) -> tuple[str, ...]:
        if not self._is_export_voyage(group.voyage_id):
            attrs = getattr(group, "attributes", {}) or {}
            if str(attrs.get("IYC_EVOY_ID", "") or "").strip():
                return ("IYC_CSZ_CSIZECD", "IYC_EVOY_ID")
            return ("IYC_CSZ_CSIZECD", "IYC_POT_UNLDPORT")
        rules = getattr(self.problem, "attribute_rules", None)
        if rules is None or not hasattr(rules, "coarse_for"):
            return ("IYC_CSZ_CSIZECD", "IYC_POT_UNLDPORT")
        return tuple(str(attr) for attr in rules.coarse_for(group.voyage_id) if str(attr))

    def _configured_fine_attrs_for_group(self, group: SmallBoxGroup) -> tuple[str, ...]:
        attrs: list[str] = list(self._configured_coarse_attrs_for_group(group))
        rules = getattr(self.problem, "attribute_rules", None)
        fine_attrs = (
            rules.fine_for(group.voyage_id)
            if rules is not None and hasattr(rules, "fine_for")
            else ()
        )
        for attr in fine_attrs:
            text = str(attr)
            if text and text not in attrs:
                attrs.append(text)
        return tuple(attrs)

    def _configured_fine_cluster_attrs_for_group(self, group: SmallBoxGroup) -> tuple[str, ...]:
        attrs: list[str] = list(self._configured_coarse_attrs_for_group(group))
        rules = getattr(self.problem, "attribute_rules", None)
        if not self._is_export_voyage(group.voyage_id):
            fine_attrs = (
                getattr(rules, "import_shared_fine_group_attributes", ())
                if rules is not None
                else ()
            )
        else:
            fine_attrs = (
                rules.fine_for(group.voyage_id)
                if rules is not None and hasattr(rules, "fine_for")
                else ()
            )
        for attr in fine_attrs:
            text = str(attr)
            if text and text not in attrs:
                attrs.append(text)
        return tuple(attrs)

    def _attribute_group_key(self, group: SmallBoxGroup, attrs: tuple[str, ...]) -> tuple[str, ...]:
        return (str(group.voyage_id), *(f"{attr}={self._group_attr_value(group, attr)}" for attr in attrs))

    def _attribute_cluster_key(self, group: SmallBoxGroup, attrs: tuple[str, ...]) -> tuple[str, ...]:
        scope = str(group.voyage_id) if self._is_export_voyage(group.voyage_id) else "IMPORT"
        return (scope, *(f"{attr}={self._group_attr_value(group, attr)}" for attr in attrs))

    def _coarse_key(self, group: SmallBoxGroup) -> tuple[str, ...]:
        return self._attribute_group_key(group, self._configured_coarse_attrs_for_group(group))

    def _fine_key(self, group: SmallBoxGroup) -> tuple[str, ...]:
        return self._attribute_group_key(group, self._configured_fine_attrs_for_group(group))

    def _coarse_cluster_key(self, group: SmallBoxGroup) -> tuple[str, ...]:
        return self._attribute_cluster_key(group, self._configured_coarse_attrs_for_group(group))

    def _fine_cluster_key(self, group: SmallBoxGroup) -> tuple[str, ...]:
        return self._attribute_cluster_key(group, self._configured_fine_cluster_attrs_for_group(group))

    def _existing_anchor_key(self, group: SmallBoxGroup) -> tuple[str, ...]:
        return self._coarse_cluster_key(group)

    def _coarse_output_value(self, coarse_key: tuple[str, ...], field: str, default: str = "MIXED") -> str:
        groups = self.coarse_groups.get(coarse_key, [])
        if not groups:
            return default
        if field == "flow":
            values = {str(group.status) for group in groups if str(group.status)}
        elif field == "port":
            values = {str(group.port) for group in groups if str(group.port)}
        elif field == "size":
            values = {str(group.size) for group in groups if str(group.size)}
        else:
            attr_values = {
                str((getattr(group, "attributes", {}) or {}).get(field, ""))
                for group in groups
                if str((getattr(group, "attributes", {}) or {}).get(field, ""))
            }
            values = attr_values
        if len(values) == 1:
            return next(iter(values))
        return default

    def _existing_coarse_bay_load_for_group(self, group: SmallBoxGroup, bay_key: str) -> int:
        bay = self.bays.get(bay_key)
        if bay is None:
            return 0
        return int(self.existing_coarse_bay_load.get(self._existing_anchor_key(group) + (bay.area_no, bay_key), 0))

    def _existing_same_coarse_bay_distance(self, group: SmallBoxGroup, bay_key: str) -> int | None:
        bay = self.bays.get(bay_key)
        if bay is None:
            return None
        anchor_bays = self.existing_coarse_area_bays.get(self._existing_anchor_key(group) + (bay.area_no,), set())
        if not anchor_bays:
            return None
        distances = [abs(self.bays[key].bay_order - bay.bay_order) for key in anchor_bays if key in self.bays]
        return min(distances) if distances else None

    def _existing_other_coarse_bay_distance(self, group: SmallBoxGroup, bay_key: str) -> int | None:
        bay = self.bays.get(bay_key)
        if bay is None:
            return None
        coarse_key = self._existing_anchor_key(group)
        distances: list[int] = []
        for (area_no, existing_coarse_key), anchor_bays in self.existing_area_coarse_bays.items():
            if area_no != bay.area_no or existing_coarse_key == coarse_key:
                continue
            distances.extend(abs(self.bays[key].bay_order - bay.bay_order) for key in anchor_bays if key in self.bays)
        return min(distances) if distances else None

    def _existing_neighbor_reward(self, distance: int) -> float:
        max_distance = max(0, int(getattr(self.config, "existing_coarse_neighbor_max_bay_distance", 0) or 0))
        if max_distance <= 0 or distance > max_distance:
            return 0.0
        scale = (max_distance - int(distance) + 1) / (max_distance + 1)
        return float(self.config.existing_coarse_neighbor_bay_reward) * scale

    def _existing_other_neighbor_penalty(self, distance: int) -> float:
        max_distance = max(0, int(getattr(self.config, "existing_coarse_neighbor_max_bay_distance", 0) or 0))
        if max_distance <= 0 or distance > max_distance:
            return 0.0
        scale = (max_distance - int(distance) + 1) / (max_distance + 1)
        return float(self.config.existing_other_coarse_neighbor_bay_penalty) * scale

    def _existing_other_coarse_bay_load_for_group(self, group: SmallBoxGroup, bay_key: str) -> int:
        coarse_key = self._existing_anchor_key(group)
        return sum(
            int(value)
            for existing_coarse_key, value in self.existing_bay_coarse_load.get(bay_key, Counter()).items()
            if existing_coarse_key != coarse_key
        )

    def _existing_coarse_bay_rank(self, group: SmallBoxGroup, bay_key: str) -> tuple[int, int, int]:
        bay_load = self._existing_coarse_bay_load_for_group(group, bay_key)
        if bay_load > 0:
            return (0, 0, -bay_load)
        distance = self._existing_same_coarse_bay_distance(group, bay_key)
        if distance is not None:
            return (1, int(distance), 0)
        return (2, 0, 0)

    def _big_plan_size(self, size: str) -> str:
        return "40" if size == "45" else size if size in {"20", "40"} else "40"

    @staticmethod
    def _key_name(key: tuple[str, ...]) -> str:
        out = []
        for part in key:
            text = str(part)
            for old, new in (("|", "_"), ("=", "_"), (" ", "_"), (":", "_"), ("/", "_"), ("\\", "_")):
                text = text.replace(old, new)
            out.append(text)
        return "_".join(out) or "key"

    def _prefers_concentrated_coarse_key(self, coarse_key: tuple[str, ...]) -> bool:
        threshold = int(self.config.medium_concentrated_group_threshold or 0)
        return threshold > 0 and self._coarse_metric_demand(coarse_key) <= threshold

    def _coarse_metric_demand(self, coarse_key: tuple[str, ...], fallback: int | None = None) -> int:
        if coarse_key in self.coarse_cluster_demand:
            return int(self.coarse_cluster_demand[coarse_key])
        if coarse_key in self.coarse_demand:
            return int(self.coarse_demand[coarse_key])
        return int(fallback or 0)

    def _target_large_group_area_count(self, coarse_key: tuple[str, ...], demand: int | None = None) -> int:
        if self._prefers_concentrated_coarse_key(coarse_key):
            return 1
        target_boxes = max(1, int(self.config.medium_large_group_target_area_boxes or 1))
        total = max(1, int(self._coarse_metric_demand(coarse_key, demand or 1) if demand is None else demand))
        return max(1, math.ceil(total / target_boxes))

    def _concentrated_area_sort_key(self, group: SmallBoxGroup, area_no: str) -> tuple[int, int, int, int, str]:
        coarse_key = self._coarse_cluster_key(group)
        demand = max(int(self._coarse_metric_demand(coarse_key, group.demand)), int(group.demand))
        quota = self.quota_by_key.get(self._quota_key(group, area_no), 0)
        group_cap = self._area_group_capacity(group, area_no)
        height_cap = self.area_size_height_cap.get((area_no, group.size, group.height), 0)
        candidate_cap = group_cap if group_cap > 0 else height_cap
        in_big_plan = self._is_big_plan_area_for_group(group, area_no)
        useful_cap = min(quota, candidate_cap) if in_big_plan and quota > 0 else candidate_cap
        return (0 if in_big_plan else 1, 0 if useful_cap >= demand else 1, -useful_cap, -quota, area_no)

    def _area_group_capacity(self, group: SmallBoxGroup, area_no: str) -> int:
        key = (group.group_id, area_no)
        if key not in self._area_group_cap_computed:
            self.area_group_cap[key] = sum(
                self._max_quantity_in_bay(group, bay_key)
                for bay_key in self.bays_by_area.get(area_no, [])
            )
            self._area_group_cap_computed.add(key)
        return int(self.area_group_cap.get(key, 0))

    def _coarse_quota_buckets(self, coarse_key: tuple[str, ...]) -> set[tuple[str, str, str]]:
        groups = self.coarse_groups.get(coarse_key)
        if not groups:
            groups = self.coarse_cluster_groups.get(coarse_key, [])
        return {
            (group.voyage_id, group.status, self._big_plan_size(group.size))
            for group in groups
        }

    def _coarse_area_target(self, coarse_key: tuple[str, ...], area_no: str) -> float:
        buckets = self._coarse_quota_buckets(coarse_key)
        total = sum(
            qty
            for (v, f, _a, s), qty in self.quota_by_key.items()
            if (v, f, s) in buckets
        )
        if total <= 0:
            return 0.0
        quota = sum(
            qty
            for (v, f, a, s), qty in self.quota_by_key.items()
            if a == area_no and (v, f, s) in buckets
        )
        return self._coarse_metric_demand(coarse_key) * quota / total

    def _area_size_target(self, voyage_id: str, flow: str, area_no: str, big_size: str) -> float:
        total = sum(qty for (v, f, _a, s), qty in self.quota_by_key.items() if v == voyage_id and f == flow and s == big_size)
        demand = self.voyage_flow_size_demand[(voyage_id, flow, big_size)]
        if total <= 0:
            return 0.0
        return demand * self.quota_by_key.get((voyage_id, flow, area_no, big_size), 0) / total

    def _voyage_area_cost(self, voyage_id: str, area_no: str) -> float:
        cost = 0.0
        if area_no in getattr(self.problem, "user_voyage_area_requirements", {}).get(voyage_id, set()):
            cost -= float(self.config.required_area_reward)
        berth = self.problem.berth_by_voyage.get(voyage_id, "")
        if berth:
            distance = self.problem.berth_distances.get((area_no, berth))
            if distance is not None:
                cost += self.config.berth_distance_penalty * distance / 100.0
        if self._area_has_loading_during_window(voyage_id, area_no):
            cost += self.config.active_loading_area_penalty
        if self._area_has_loading_after_window(voyage_id, area_no):
            cost -= self.config.post_window_loading_area_reward
        return cost

    def _user_required_area_usage(self, medium_rows: list[dict], small_rows: list[dict]) -> dict:
        requirements = getattr(self.problem, "user_voyage_area_requirements", {})
        if not requirements:
            return {}
        medium_used: Counter[tuple[str, str]] = Counter()
        small_used: Counter[tuple[str, str]] = Counter()
        for row in medium_rows:
            key = (str(row.get("voyage_id", "")), str(row.get("area_no", "")))
            medium_used[key] += int(row.get("planned_boxes", 0) or 0)
        for row in small_rows:
            key = (str(row.get("voyage_id", "")), str(row.get("area_no", "")))
            small_used[key] += int(row.get("planned_boxes", 0) or 0)
        out = {}
        for voyage_id, areas in sorted(requirements.items()):
            voyage_usage = {}
            for area_no in sorted(areas):
                medium_boxes = int(medium_used[(voyage_id, area_no)])
                small_boxes = int(small_used[(voyage_id, area_no)])
                voyage_usage[area_no] = {
                    "medium_boxes": medium_boxes,
                    "small_boxes": small_boxes,
                    "satisfied": medium_boxes + small_boxes > 0,
                }
            out[voyage_id] = voyage_usage
        return out

    def _user_area_constraint_violations(self, medium_rows: list[dict], small_rows: list[dict]) -> dict:
        required_usage = self._user_required_area_usage(medium_rows, small_rows)
        unmet_required = []
        for voyage_id, area_usage in required_usage.items():
            for area_no, usage in area_usage.items():
                if not usage.get("satisfied", False):
                    unmet_required.append(
                        {
                            "voyage_id": voyage_id,
                            "area_no": area_no,
                            "reason": "required_area_not_used",
                        }
                    )

        allowlist = getattr(self.problem, "user_voyage_area_allowlist", {})
        blocklist = getattr(self.problem, "user_voyage_area_blocklist", {})
        bay_requirements = getattr(self.problem, "user_group_bay_requirements", {})
        bay_blocklist = getattr(self.problem, "user_group_bay_blocklist", {})
        forbidden_usage: Counter[tuple[str, str, str]] = Counter()
        outside_only_usage: Counter[tuple[str, str, str]] = Counter()
        required_bay_usage: Counter[tuple[str, str]] = Counter()
        forbidden_bay_usage: Counter[tuple[str, str, str]] = Counter()
        for plan_level, rows in (("medium", medium_rows), ("small", small_rows)):
            for row in rows:
                qty = int(row.get("planned_boxes", 0) or 0)
                if qty <= 0:
                    continue
                voyage_id = str(row.get("voyage_id", ""))
                area_no = str(row.get("area_no", ""))
                group_id = str(row.get("group_id", ""))
                bay_key = str(row.get("bay_key", ""))
                if area_no in blocklist.get(voyage_id, set()):
                    forbidden_usage[(voyage_id, area_no, plan_level)] += qty
                allowed = allowlist.get(voyage_id, set())
                if allowed and area_no not in allowed:
                    outside_only_usage[(voyage_id, area_no, plan_level)] += qty
                if group_id and bay_key in bay_requirements.get(group_id, set()):
                    required_bay_usage[(group_id, bay_key)] += qty
                if group_id and bay_key in bay_blocklist.get(group_id, set()):
                    forbidden_bay_usage[(group_id, bay_key, plan_level)] += qty

        forbidden = [
            {"voyage_id": voyage_id, "area_no": area_no, "plan_level": level, "boxes": qty}
            for (voyage_id, area_no, level), qty in sorted(forbidden_usage.items())
        ]
        outside_only = [
            {"voyage_id": voyage_id, "area_no": area_no, "plan_level": level, "boxes": qty}
            for (voyage_id, area_no, level), qty in sorted(outside_only_usage.items())
        ]
        unmet_required_bays = [
            {"group_id": group_id, "bay_key": bay_key, "reason": "required_bay_not_used"}
            for group_id, bay_keys in sorted(bay_requirements.items())
            for bay_key in sorted(bay_keys)
            if required_bay_usage[(group_id, bay_key)] <= 0
        ]
        forbidden_bays = [
            {"group_id": group_id, "bay_key": bay_key, "plan_level": level, "boxes": qty}
            for (group_id, bay_key, level), qty in sorted(forbidden_bay_usage.items())
        ]
        return {
            "has_violations": bool(unmet_required or forbidden or outside_only or unmet_required_bays or forbidden_bays),
            "unmet_required_areas": unmet_required,
            "forbidden_area_usage": forbidden,
            "outside_only_area_usage": outside_only,
            "unmet_required_bays": unmet_required_bays,
            "forbidden_bay_usage": forbidden_bays,
        }

    def _export_e_area_usage(self, selected: Counter[int]) -> list[dict]:
        bay_usage: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        box_usage: Counter[tuple[str, str]] = Counter()
        size_usage: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            if not self._is_export_e_column(col):
                continue
            key = (col.voyage_id, col.area_no)
            bay_usage[key].add(col.bay_key)
            boxes = int(col.quantity) * int(chosen)
            box_usage[key] += boxes
            size_usage[key][col.size] += boxes
        limit = self._export_e_area_max_bays()
        return [
            {
                "voyage_id": voyage_id,
                "area_no": area_no,
                "used_bays": len(bays),
                "max_bays": "" if limit is None else int(limit),
                "planned_boxes": int(box_usage[(voyage_id, area_no)]),
                "planned_20": int(size_usage[(voyage_id, area_no)].get("20", 0)),
                "planned_40": int(size_usage[(voyage_id, area_no)].get("40", 0)),
                "planned_45": int(size_usage[(voyage_id, area_no)].get("45", 0)),
                "bay_keys": sorted(bays),
            }
            for (voyage_id, area_no), bays in sorted(bay_usage.items())
        ]

    def _export_voyage_row_no_mix_stats(self, selected: Counter[int]) -> dict:
        """Audit the export-voyage row isolation hard constraint in the final solution."""
        new_voyages_by_row: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        new_boxes_by_row_voyage: Counter[tuple[str, str, str]] = Counter()
        planned_export_boxes = 0
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            if not self._is_export_voyage(col.voyage_id):
                continue
            planned_export_boxes += int(col.quantity) * int(chosen)
            for footprint_key, row_no, qty in col.row_allocation:
                row_qty = int(qty) * int(chosen)
                if row_qty <= 0:
                    continue
                row_key = (str(footprint_key), str(row_no))
                voyage_id = str(col.voyage_id)
                new_voyages_by_row[row_key].add(voyage_id)
                new_boxes_by_row_voyage[row_key + (voyage_id,)] += row_qty

        new_new_conflicts = []
        existing_new_conflicts = []
        for (bay_key, row_no), new_voyages in sorted(new_voyages_by_row.items()):
            if len(new_voyages) > 1:
                new_new_conflicts.append(
                    {
                        "bay_key": bay_key,
                        "row_no": row_no,
                        "new_export_voyages": sorted(new_voyages),
                    }
                )
            bay = self.bays.get(bay_key)
            existing_voyages = (
                self._existing_row_attr_values(
                    bay,
                    row_no,
                    EXPORT_VOYAGE_ROW_NO_MIX_ATTR,
                    "",
                )
                if bay is not None
                else set()
            )
            for voyage_id in sorted(new_voyages):
                if existing_voyages and existing_voyages != {voyage_id}:
                    existing_new_conflicts.append(
                        {
                            "bay_key": bay_key,
                            "row_no": row_no,
                            "existing_export_voyages": sorted(existing_voyages),
                            "new_export_voyage": voyage_id,
                            "new_boxes": int(new_boxes_by_row_voyage[(bay_key, row_no, voyage_id)]),
                        }
                    )

        return {
            "rule": "different export voyages cannot share one yard row",
            "scope": "area_bay_row",
            "classification": "voyage_direction",
            "applies_to_voyages": sorted(self.export_voyages),
            "planned_export_boxes": int(planned_export_boxes),
            "used_export_rows": len(new_voyages_by_row),
            "new_new_conflict_count": len(new_new_conflicts),
            "existing_new_conflict_count": len(existing_new_conflicts),
            "has_violations": bool(new_new_conflicts or existing_new_conflicts),
            "new_new_conflicts": new_new_conflicts,
            "existing_new_conflicts": existing_new_conflicts,
        }

    def _unplaced_group_details(self, unplaced: Counter[str]) -> list[dict]:
        details = []
        for group_id, qty in sorted(unplaced.items()):
            if qty <= 0:
                continue
            group = self.groups_by_id.get(group_id)
            if group is None:
                details.append({"group_id": group_id, "unplaced_boxes": int(qty)})
                continue
            details.append(
                {
                    "group_id": group_id,
                    "demand_source": self.group_source.get(group_id, "document"),
                    "voyage_id": group.voyage_id,
                    "flow": group.status,
                    "port": group.port,
                    "size": group.size,
                    "height": group.height,
                    "weight_class": group.weight_class,
                    "special_stow_code": group.special_stow_code,
                    "demand": int(group.demand),
                    "unplaced_boxes": int(qty),
                }
            )
        return details

    def _area_has_loading_during_window(self, voyage_id: str, area_no: str) -> bool:
        window_start, window_end = self.problem.voyage_windows[voyage_id]
        return any(
            op.voyage_id != voyage_id and _time_windows_overlap(op.start_time, op.end_time, window_start, window_end)
            for op in self.problem.area_operations.get(area_no, [])
        )

    def _area_has_loading_after_window(self, voyage_id: str, area_no: str) -> bool:
        _window_start, window_end = self.problem.voyage_windows[voyage_id]
        prefer_end = window_end + timedelta(hours=24)
        return any(
            op.voyage_id != voyage_id and _time_windows_overlap(op.start_time, op.end_time, window_end, prefer_end)
            for op in self.problem.area_operations.get(area_no, [])
        )

    def _medium_big_plan_inheritance_stats(self, medium_rows: list[dict]) -> dict[str, float | int]:
        actual = self._medium_area_size_counter(medium_rows)
        total = sum(actual.values())
        inherited = sum(min(qty, int(self.quota_by_key.get(key, 0) or 0)) for key, qty in actual.items())
        deviated = max(0, total - inherited)
        return {
            "total_boxes": total,
            "inherited_boxes": inherited,
            "deviated_boxes": deviated,
            "inheritance_ratio": round(inherited / total, 6) if total else 1.0,
            "deviation_ratio": round(deviated / total, 6) if total else 0.0,
        }

    def _medium_inheritance_energy_components(self, medium_rows: list[dict]) -> dict[str, float]:
        actual = self._medium_area_size_counter(medium_rows)
        components = {
            "big_plan_fallback_tier": 0.0,
            "big_plan_deviation": 0.0,
        }
        for (voyage_id, flow, area_no, big_size), qty in actual.items():
            tier = self._area_fallback_tier_for_attrs(voyage_id, flow, area_no, big_size)
            if tier > 0:
                components["big_plan_fallback_tier"] += self.config.big_plan_fallback_tier_penalty * tier * qty

        targets = self._effective_big_plan_area_size_targets()
        for key in set(actual) | set(targets):
            components["big_plan_deviation"] += self.config.big_plan_area_deviation_penalty * abs(
                actual.get(key, 0) - targets.get(key, 0.0)
            )
        components["total"] = sum(components.values())
        return {key: round(value, 4) for key, value in components.items()}

    def _medium_area_size_counter(self, medium_rows: list[dict]) -> Counter[tuple[str, str, str, str]]:
        actual: Counter[tuple[str, str, str, str]] = Counter()
        for row in medium_rows:
            qty = int(row.get("planned_boxes", 0) or 0)
            if qty <= 0:
                continue
            voyage_id = str(row.get("voyage_id", ""))
            flow = str(row.get("flow", "OF") or "OF")
            area_no = str(row.get("area_no", ""))
            big_size = self._big_plan_size(str(row.get("size", "")))
            actual[(voyage_id, flow, area_no, big_size)] += qty
        return actual

    @staticmethod
    def _planned_medium_by_source(medium_rows: list[dict]) -> dict[str, int]:
        out: Counter[str] = Counter()
        for row in medium_rows:
            document_boxes = int(row.get("document_boxes", 0) or 0)
            forecast_boxes = int(row.get("forecast_fallback_boxes", 0) or 0)
            known = document_boxes + forecast_boxes
            if document_boxes > 0:
                out["document"] += document_boxes
            if forecast_boxes > 0:
                out["forecast_fallback"] += forecast_boxes
            remainder = int(row.get("planned_boxes", 0) or 0) - known
            if remainder > 0:
                out["unknown"] += remainder
        return {key: int(value) for key, value in sorted(out.items())}

    def _effective_big_plan_area_size_targets(self) -> dict[tuple[str, str, str, str], float]:
        targets: dict[tuple[str, str, str, str], float] = {}
        keys = {
            (voyage_id, flow, big_size)
            for voyage_id, flow, _area_no, big_size in self.quota_by_key
            if self.voyage_flow_size_demand[(voyage_id, flow, big_size)] > 0
        }
        for voyage_id, flow, big_size in keys:
            total = sum(
                qty
                for (v, f, _area_no, s), qty in self.quota_by_key.items()
                if v == voyage_id and f == flow and s == big_size
            )
            demand = self.voyage_flow_size_demand[(voyage_id, flow, big_size)]
            if total <= 0 or demand <= 0:
                continue
            target_total = min(demand, total)
            for (v, f, area_no, s), quota in self.quota_by_key.items():
                if v == voyage_id and f == flow and s == big_size and quota > 0:
                    targets[(v, f, area_no, s)] = target_total * quota / total
        return targets

    def _make_small_rows(self, selected: Counter[int], allowed_sources: set[str] | None = None) -> list[dict]:
        counter: Counter[tuple] = Counter()
        for idx, chosen in selected.items():
            if chosen <= 0:
                continue
            col = self._columns[idx]
            source = self.group_source.get(col.group_id, "document")
            if allowed_sources is not None and source not in allowed_sources:
                continue
            qty_by_row: Counter[str] = Counter()
            for _footprint_key, row_no, qty in col.row_allocation:
                qty_by_row[str(row_no)] = max(qty_by_row[str(row_no)], int(qty))
            for row_no, row_qty in qty_by_row.items():
                row_specific_allocation = self._format_row_allocation(
                    tuple(item for item in col.row_allocation if str(item[1]) == row_no)
                )
                dynamic_attrs = tuple(sorted((str(k), str(v)) for k, v in (col.attributes or {}).items()))
                counter[
                    (
                        col.voyage_id,
                        col.group_id,
                        col.flow,
                        col.port,
                        col.size,
                        col.height,
                        col.weight_class,
                        col.special_stow_code,
                        col.area_no,
                        col.bay_no,
                        row_no,
                        col.block_id,
                        col.block_bays,
                        row_specific_allocation,
                        dynamic_attrs,
                    )
                ] += row_qty * chosen
        block_total: Counter[str] = Counter()
        for key, qty in counter.items():
            block_id = key[11]
            if block_id:
                block_total[block_id] += qty
        rows: list[dict] = []
        for key, qty in sorted(counter.items()):
            voyage_id, group_id, flow, port, size, height, weight_class, special_code, area_no, bay_no, row_no, block_id, block_bays, row_allocation, dynamic_attrs = key
            row = {
                "plan_level": "small",
                "voyage_id": voyage_id,
                "group_id": group_id,
                "demand_source": self.group_source.get(group_id, "document"),
                "flow": flow,
                "port": port,
                "size": size,
                "height": height,
                "weight_class": weight_class,
                "special_stow": bool(special_code),
                "special_stow_code": special_code or "NORMAL",
                "area_no": area_no,
                "bay_no": bay_no,
                "row_no": row_no,
                "row_allocation": row_allocation,
                "six_bay_block_id": block_id,
                "six_bay_block_bays": "|".join(block_bays) if block_id else "",
                "six_bay_block_total_boxes": block_total.get(block_id, 0) if block_id else 0,
                "planned_boxes": qty,
            }
            for attr, value in dynamic_attrs:
                if attr and attr not in row:
                    row[attr] = value
            rows.append(row)
        return rows

    @staticmethod
    def _format_row_allocation(row_allocation: tuple[tuple[str, str, int], ...]) -> str:
        return "|".join(f"{bay_key}:{row_no}:{int(qty)}" for bay_key, row_no, qty in row_allocation if int(qty) > 0)

    def _medium_output_row(
        self,
        plan_level: str,
        voyage_id: str,
        flow: str,
        port: str,
        size: str,
        area_no: str,
        bay_key: str,
        bay_no: str,
        block_id: str,
        block_bays: tuple[str, ...],
        qty: int,
        source_counts: Counter[str] | None = None,
        attributes: dict[str, str] | None = None,
    ) -> dict:
        source_counts = Counter(source_counts or {})
        row = {
            "plan_level": plan_level,
            "voyage_id": voyage_id,
            "flow": flow,
            "port": port,
            "size": size,
            "area_no": area_no,
            "bay_no": bay_no,
            "six_bay_block_id": block_id,
            "six_bay_block_bays": "|".join(block_bays) if block_id else "",
            "planned_boxes": qty,
            "document_boxes": int(source_counts.get("document", 0)),
            "forecast_fallback_boxes": int(source_counts.get("forecast_fallback", 0)),
        }
        for attr, value in sorted((attributes or {}).items()):
            if attr and attr not in row:
                row[attr] = value
        return row

    def _make_medium_rows(self, small_rows: list[dict]) -> list[dict]:
        counter: Counter[tuple] = Counter()
        for row in small_rows:
            area_no = str(row["area_no"])
            bay_no = str(row.get("bay_no", ""))
            bay_key = str(row.get("bay_key") or f"{area_no}-{bay_no}" if bay_no else "")
            block_id = str(row.get("six_bay_block_id", ""))
            block_bays = tuple(str(row.get("six_bay_block_bays", "")).split("|")) if row.get("six_bay_block_bays") else ()
            dynamic_attrs = tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in row.items()
                    if key
                    not in {
                        "plan_level",
                        "voyage_id",
                        "group_id",
                        "demand_source",
                        "flow",
                        "port",
                        "size",
                        "height",
                        "weight_class",
                        "special_stow",
                        "special_stow_code",
                        "area_no",
                        "bay_no",
                        "row_no",
                        "row_allocation",
                        "six_bay_block_id",
                        "six_bay_block_bays",
                        "six_bay_block_total_boxes",
                        "planned_boxes",
                    }
                )
            )
            key = (
                str(row["voyage_id"]),
                str(row["flow"]),
                str(row["port"]),
                str(row["size"]),
                area_no,
                bay_key,
                bay_no,
                block_id,
                block_bays,
                dynamic_attrs,
            )
            counter[key] += int(row["planned_boxes"])
        rows: list[dict] = []
        for (voyage_id, flow, port, size, area_no, bay_key, bay_no, block_id, block_bays, dynamic_attrs), qty in sorted(counter.items()):
            rows.append(
                self._medium_output_row(
                    "medium_from_small",
                    voyage_id,
                    flow,
                    port,
                    size,
                    area_no,
                    bay_key,
                    bay_no,
                    block_id,
                    block_bays,
                    qty,
                    attributes=dict(dynamic_attrs),
                )
            )
        return rows

    def _make_medium_rows_from_selected_columns(self, selected: Counter[int], plan_level: str = "medium") -> list[dict]:
        counter: Counter[tuple] = Counter()
        source_counter: defaultdict[tuple, Counter[str]] = defaultdict(Counter)
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            qty = col.quantity * int(chosen)
            key = (
                col.voyage_id,
                col.flow,
                col.port,
                col.size,
                col.area_no,
                col.bay_key,
                col.bay_no,
                col.block_id,
                col.block_bays,
                tuple(sorted((str(k), str(v)) for k, v in (col.attributes or {}).items())),
            )
            counter[key] += qty
            source_counter[key][self.group_source.get(col.group_id, "document")] += qty
        rows: list[dict] = []
        for (voyage_id, flow, port, size, area_no, bay_key, bay_no, block_id, block_bays, dynamic_attrs), qty in sorted(counter.items()):
            if qty > 0:
                key = (voyage_id, flow, port, size, area_no, bay_key, bay_no, block_id, block_bays, dynamic_attrs)
                rows.append(
                    self._medium_output_row(
                        plan_level,
                        voyage_id,
                        flow,
                        port,
                        size,
                        area_no,
                        bay_key,
                        bay_no,
                        block_id,
                        block_bays,
                        qty,
                        source_counter[key],
                        dict(dynamic_attrs),
                    )
                )
        return rows

    def _make_original_medium_rows(self, selected: Counter[int]) -> list[dict]:
        return self._make_medium_rows_from_selected_columns(selected, plan_level="medium")

    def _make_original_medium_area_rows(self, selected: Counter[int]) -> list[dict]:
        selected_coarse_area_weights: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        selected_area_weights: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
        small_lower_by_coarse_area: Counter[tuple[str, ...]] = Counter()
        for idx, chosen in selected.items():
            if chosen <= 0:
                continue
            col = self._columns[idx]
            selected_coarse_area_weights[col.coarse_key][col.area_no] += col.quantity * chosen
            selected_area_weights[(col.voyage_id, col.flow, col.big_plan_size)][col.area_no] += col.quantity * chosen
            if self.group_source.get(col.group_id, "document") == "document":
                lower_key = col.coarse_key + (col.area_no,)
                lower_qty = col.quantity * chosen
                small_lower_by_coarse_area[lower_key] += lower_qty

        remaining_quota: Counter[tuple[str, str, str, str]] = Counter(self.quota_by_key)
        counter: Counter[tuple[str, ...]] = Counter()
        medium_remaining_by_coarse: Counter[tuple[str, ...]] = Counter()
        representative_by_coarse: dict[tuple[str, ...], object] = {}
        for group in self.problem.groups:
            coarse_key = self._coarse_key(self._medium_group_as_small_group(group, group.demand))
            medium_remaining_by_coarse[coarse_key] += int(group.demand)
            representative_by_coarse.setdefault(coarse_key, group)

        for key, qty in small_lower_by_coarse_area.items():
            if qty <= 0:
                continue
            coarse_key = tuple(key[:-1])
            area_no = key[-1]
            counter[coarse_key + (area_no,)] += qty
            representative = representative_by_coarse.get(coarse_key)
            if representative is not None:
                remaining_quota[(representative.voyage_id, representative.status, area_no, self._big_plan_size(representative.size))] -= qty
            self._consume_medium_remaining_for_small_lower(
                medium_remaining_by_coarse,
                coarse_key,
                int(qty),
            )

        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        concentration_threshold = max(0, int(self.config.medium_concentrated_group_threshold or 0))
        sorted_groups = sorted(
            representative_by_coarse.values(),
            key=lambda g: (
                g.voyage_id,
                g.status,
                SIZE_ORDER.get(g.size, 3),
                0 if concentration_threshold > 0 and int(medium_remaining_by_coarse[self._coarse_key(self._medium_group_as_small_group(g, g.demand))]) <= concentration_threshold else 1,
                (
                    int(medium_remaining_by_coarse[self._coarse_key(self._medium_group_as_small_group(g, g.demand))])
                    if concentration_threshold > 0 and int(medium_remaining_by_coarse[self._coarse_key(self._medium_group_as_small_group(g, g.demand))]) <= concentration_threshold
                    else -int(medium_remaining_by_coarse[self._coarse_key(self._medium_group_as_small_group(g, g.demand))])
                ),
                g.port,
                g.group_id,
            ),
        )
        for group in sorted_groups:
            big_size = self._big_plan_size(group.size)
            coarse_key = self._coarse_key(self._medium_group_as_small_group(group, group.demand))
            remaining_target = int(medium_remaining_by_coarse.get(coarse_key, 0))
            if remaining_target <= 0:
                continue
            quota_key_prefix = (group.voyage_id, group.status)
            caps = Counter(
                {
                    area_no: qty
                    for (voyage_id, flow, area_no, size), qty in remaining_quota.items()
                    if voyage_id == quota_key_prefix[0] and flow == quota_key_prefix[1] and size == big_size and qty > 0
                }
            )
            weights = Counter(
                {
                    area_no: qty
                    for area_no, qty in selected_coarse_area_weights.get(coarse_key, Counter()).items()
                    if caps.get(area_no, 0) > 0
                }
            )
            if not weights:
                weights = Counter(
                    {
                        area_no: qty
                        for area_no, qty in selected_area_weights.get((group.voyage_id, group.status, big_size), Counter()).items()
                        if caps.get(area_no, 0) > 0
                    }
                )
            if not weights:
                weights = Counter(caps)
            if not weights:
                weights = Counter(selected_coarse_area_weights.get(coarse_key, Counter()))
            if not weights:
                weights = Counter(selected_area_weights.get((group.voyage_id, group.status, big_size), Counter()))
            allocation = self._allocate_area_quantity(
                remaining_target,
                weights,
                self._caps_with_overflow(caps, weights, remaining_target),
                min_boxes=min_boxes,
                concentrate=(concentration_threshold > 0 and remaining_target <= concentration_threshold),
            )
            for area_no, qty in allocation.items():
                if qty <= 0:
                    continue
                counter[coarse_key + (area_no,)] += qty
                remaining_quota[(group.voyage_id, group.status, area_no, big_size)] -= qty

        counter = self._compact_medium_counter(
            counter,
            small_lower_by_coarse_area,
            representative_by_coarse,
            selected_coarse_area_weights,
            selected_area_weights,
        )

        rows: list[dict] = []
        for key, qty in sorted(counter.items()):
            coarse_key = tuple(key[:-1])
            area_no = key[-1]
            representative = representative_by_coarse.get(coarse_key)
            if representative is None:
                continue
            voyage_id = representative.voyage_id
            flow = representative.status
            port = self._coarse_output_value(coarse_key, "port", "MIXED")
            size = self._coarse_output_value(coarse_key, "size", "MIXED")
            window_start, window_end = self.problem.voyage_windows[voyage_id]
            row = {
                    "plan_level": "medium",
                    "voyage_id": voyage_id,
                    "flow": flow,
                    "port": port,
                    "size": size,
                    "window_start": window_start.isoformat(sep=" "),
                    "window_end": window_end.isoformat(sep=" "),
                    "area_loading_during_window": self._area_has_loading_during_window(voyage_id, area_no),
                    "area_loading_after_window_24h": self._area_has_loading_after_window(voyage_id, area_no),
                    "area_no": area_no,
                    "planned_boxes": qty,
            }
            for attr, value in sorted((getattr(representative, "attributes", {}) or {}).items()):
                if attr and attr not in row:
                    row[attr] = value
            rows.append(row)
        return rows

    def _compact_medium_counter(
        self,
        counter: Counter[tuple[str, ...]],
        small_lower_by_coarse_area: Counter[tuple[str, ...]],
        representative_by_coarse: dict[tuple[str, ...], object],
        selected_coarse_area_weights: dict[tuple[str, ...], Counter[str]],
        selected_area_weights: dict[tuple[str, str, str], Counter[str]],
    ) -> Counter[tuple[str, ...]]:
        compacted: Counter[tuple[str, ...]] = Counter()
        by_coarse: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        for key, qty in counter.items():
            if qty > 0:
                by_coarse[tuple(key[:-1])][key[-1]] += int(qty)

        for coarse_key, quantities in by_coarse.items():
            lower = Counter(
                {
                    area_no: int(small_lower_by_coarse_area.get(coarse_key + (area_no,), 0))
                    for area_no in quantities
                    if small_lower_by_coarse_area.get(coarse_key + (area_no,), 0) > 0
                }
            )
            group = representative_by_coarse.get(coarse_key)
            weights = Counter(selected_coarse_area_weights.get(coarse_key, Counter()))
            quota_caps: Counter[str] = Counter()
            if group is not None:
                big_size = self._big_plan_size(group.size)
                weights.update(selected_area_weights.get((group.voyage_id, group.status, big_size), Counter()))
                preferred_areas = set()
                for (voyage_id, flow, area_no, size), quota in self.quota_by_key.items():
                    if voyage_id == group.voyage_id and flow == group.status and size == big_size and quota > 0:
                        weights[area_no] += int(quota)
                        quota_caps[area_no] += int(quota)
                        preferred_areas.add(area_no)
            else:
                preferred_areas = set()
            if not weights:
                weights = Counter(quantities)

            new_quantities = self._compact_coarse_medium_distribution(
                coarse_key,
                quantities,
                lower,
                weights,
                preferred_areas,
                quota_caps,
            )
            for area_no, qty in new_quantities.items():
                if qty > 0:
                    compacted[coarse_key + (area_no,)] += int(qty)
        return compacted

    def _compact_coarse_medium_distribution(
        self,
        coarse_key: tuple[str, ...],
        quantities: Counter[str],
        lower: Counter[str],
        weights: Counter[str],
        preferred_areas: set[str],
        quota_caps: Counter[str],
    ) -> Counter[str]:
        total = int(sum(quantities.values()))
        if total <= 0:
            return Counter()
        lower = Counter({area: min(int(qty), int(quantities.get(area, qty))) for area, qty in lower.items() if qty > 0})
        lower_total = int(sum(lower.values()))
        movable = max(0, total - lower_total)
        if movable <= 0:
            return Counter({area: qty for area, qty in lower.items() if qty > 0})

        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        concentration_threshold = max(0, int(self.config.medium_concentrated_group_threshold or 0))
        candidates = set(quantities) | set(lower) | set(quota_caps) | {area for area, qty in weights.items() if qty > 0}
        if not candidates:
            return Counter({area: qty for area, qty in quantities.items() if qty > 0})

        def area_sort_key(area_no: str) -> tuple[int, int, int, int, str]:
            return (
                0 if area_no in preferred_areas else 1,
                0 if lower.get(area_no, 0) > 0 else 1,
                -int(weights.get(area_no, 0)),
                -int(quantities.get(area_no, 0)),
                area_no,
            )

        if concentration_threshold > 0 and total <= concentration_threshold:
            primary = sorted(candidates, key=area_sort_key)[0]
            out = Counter(lower)
            out[primary] += movable
            return Counter({area: qty for area, qty in out.items() if qty > 0})

        target_count = max(
            1,
            len([area for area, qty in lower.items() if qty > 0]),
            self._target_large_group_area_count(coarse_key, total),
        )
        target_areas = {area for area, qty in lower.items() if qty > 0}
        target_areas.update(
            area
            for area, qty in quantities.items()
            if qty > 0 and area in preferred_areas
        )
        for area_no in sorted(candidates, key=area_sort_key):
            selected_cap = sum(max(int(lower.get(area, 0)), int(quota_caps.get(area, 0))) for area in target_areas)
            if len(target_areas) >= target_count and selected_cap >= total:
                break
            target_areas.add(area_no)
        if not target_areas:
            target_areas.add(sorted(candidates, key=area_sort_key)[0])

        out = Counter(lower)
        remaining = movable
        if min_boxes > 1:
            for area_no in sorted(
                target_areas,
                key=lambda area: (out.get(area, 0) <= 0, out.get(area, 0), -weights.get(area, 0), area),
            ):
                if remaining <= 0:
                    break
                current = int(out.get(area_no, 0))
                if current <= 0 or current >= min_boxes:
                    continue
                add = min(remaining, min_boxes - current)
                out[area_no] += add
                remaining -= add

        if remaining > 0:
            target_weights = Counter(
                {
                    area: max(1, int(weights.get(area, 0) or quantities.get(area, 0) or 1))
                    for area in target_areas
                }
            )
            movable_caps = Counter(
                {
                    area: max(0, max(int(quota_caps.get(area, 0)), int(lower.get(area, 0))) - int(out.get(area, 0)))
                    for area in target_areas
                }
            )
            if sum(movable_caps.values()) < remaining:
                movable_caps = self._caps_with_overflow(movable_caps, target_weights, remaining)
            for area_no, qty in self._allocate_area_quantity(remaining, target_weights, movable_caps).items():
                out[area_no] += qty

        return self._merge_movable_medium_fragments(out, lower, weights, min_boxes)

    @staticmethod
    def _merge_movable_medium_fragments(
        quantities: Counter[str],
        lower: Counter[str],
        weights: Counter[str],
        min_boxes: int,
    ) -> Counter[str]:
        if min_boxes <= 1:
            return Counter({area: qty for area, qty in quantities.items() if qty > 0})
        out = Counter({area: int(qty) for area, qty in quantities.items() if qty > 0})
        for area_no in sorted(list(out), key=lambda area: (out[area], int(weights.get(area, 0)), area)):
            qty = out.get(area_no, 0)
            if qty <= 0 or qty >= min_boxes:
                continue
            movable = qty - int(lower.get(area_no, 0))
            if movable <= 0:
                continue
            recipients = [area for area, target_qty in out.items() if area != area_no and target_qty > 0]
            if not recipients:
                continue
            target = sorted(
                recipients,
                key=lambda area: (
                    0 if out[area] >= min_boxes else 1,
                    -int(weights.get(area, 0)),
                    -out[area],
                    area,
                ),
            )[0]
            out[area_no] -= movable
            out[target] += movable
            if out[area_no] <= 0:
                del out[area_no]
        return Counter({area: qty for area, qty in out.items() if qty > 0})

    def _consume_medium_remaining_for_small_lower(
        self,
        medium_remaining_by_coarse: Counter[tuple[str, ...]],
        coarse_key: tuple[str, ...],
        qty: int,
    ) -> int:
        need = max(0, int(qty))
        if need <= 0:
            return 0
        take = min(need, max(0, medium_remaining_by_coarse.get(coarse_key, 0)))
        if take > 0:
            medium_remaining_by_coarse[coarse_key] -= take
            need -= take
        return need

    @staticmethod
    def _caps_with_overflow(caps: Counter[str], weights: Counter[str], total: int) -> Counter[str]:
        adjusted = Counter({area: int(qty) for area, qty in caps.items() if qty > 0})
        missing = max(0, int(total) - sum(adjusted.values()))
        if missing <= 0:
            return adjusted
        overflow_areas = [area for area, qty in sorted(weights.items()) if qty > 0]
        if not overflow_areas:
            overflow_areas = sorted(adjusted)
        if not overflow_areas:
            return adjusted
        overflow_weights = Counter({area: max(1, int(weights.get(area, 0))) for area in overflow_areas})
        for area, qty in ColumnGenerationPlanner._allocate_integer_by_weights(overflow_weights, missing):
            adjusted[area] += qty
        return adjusted

    @classmethod
    def _allocate_area_quantity(
        cls,
        total: int,
        weights: Counter[str],
        caps: Counter[str],
        min_boxes: int = 0,
        concentrate: bool = False,
    ) -> Counter[str]:
        allocation: Counter[str] = Counter()
        remaining = max(0, int(total))
        available = Counter({area: int(cap) for area, cap in caps.items() if cap > 0})
        if remaining <= 0 or not available:
            return allocation
        if concentrate or (min_boxes > 1 and remaining < min_boxes):
            for area in sorted(
                available,
                key=lambda item: (
                    0 if available[item] >= remaining else 1,
                    -int(weights.get(item, 0)),
                    -available[item],
                    item,
                ),
            ):
                if remaining <= 0:
                    break
                take = min(remaining, available[area])
                if take <= 0:
                    continue
                allocation[area] += take
                remaining -= take
            return allocation

        while remaining > 0 and available:
            candidate_weights = Counter({area: max(0, int(weights.get(area, 0))) for area in available})
            if sum(candidate_weights.values()) <= 0:
                candidate_weights = Counter(available)
            if min_boxes > 1 and total >= min_boxes:
                filtered = Counter(
                    {
                        area: weight
                        for area, weight in candidate_weights.items()
                        if available[area] >= min_boxes
                    }
                )
                if filtered:
                    candidate_weights = filtered
                candidate_weights = cls._remove_tiny_weighted_areas(candidate_weights, remaining, min_boxes)
            raw_allocation = Counter(dict(cls._allocate_integer_by_weights(candidate_weights, remaining)))
            progress = 0
            overflow = 0
            for area in list(available):
                take = min(available[area], raw_allocation.get(area, 0))
                if take > 0:
                    allocation[area] += take
                    available[area] -= take
                    progress += take
                    if available[area] <= 0:
                        del available[area]
                overflow += max(0, raw_allocation.get(area, 0) - take)
            remaining -= progress
            if remaining <= 0:
                break
            if progress == 0:
                area = max(available, key=lambda key: (available[key], weights.get(key, 0), key))
                take = min(remaining, available[area])
                allocation[area] += take
                available[area] -= take
                remaining -= take
                if available[area] <= 0:
                    del available[area]
            elif overflow <= 0 and progress < remaining:
                weights = Counter(available)
        return cls._merge_small_fragments(allocation, caps, weights, min_boxes)

    @classmethod
    def _remove_tiny_weighted_areas(
        cls,
        weights: Counter[str],
        total: int,
        min_boxes: int,
    ) -> Counter[str]:
        if min_boxes <= 1 or total < min_boxes or len(weights) <= 1:
            return weights
        active = Counter({area: weight for area, weight in weights.items() if weight > 0})
        if len(active) <= 1:
            return weights
        while len(active) > 1:
            trial = Counter(dict(cls._allocate_integer_by_weights(active, total)))
            tiny_areas = [
                area
                for area, qty in trial.items()
                if 0 < qty < min_boxes
            ]
            if not tiny_areas or len(tiny_areas) >= len(active):
                break
            for area in tiny_areas:
                del active[area]
        return active or weights

    @staticmethod
    def _merge_small_fragments(
        allocation: Counter[str],
        caps: Counter[str],
        weights: Counter[str],
        min_boxes: int,
    ) -> Counter[str]:
        if min_boxes <= 1 or sum(allocation.values()) < min_boxes:
            return allocation
        merged = Counter({area: qty for area, qty in allocation.items() if qty > 0})
        for area in sorted(list(merged), key=lambda key: (merged[key], int(weights.get(key, 0)), key)):
            qty = merged.get(area, 0)
            if qty <= 0 or qty >= min_boxes:
                continue
            recipients = [
                target
                for target, target_qty in merged.items()
                if target != area and target_qty > 0 and int(caps.get(target, 0)) - target_qty >= qty
            ]
            if not recipients:
                continue
            target = sorted(
                recipients,
                key=lambda key: (
                    0 if merged[key] >= min_boxes else 1,
                    -int(weights.get(key, 0)),
                    -merged[key],
                    -(int(caps.get(key, 0)) - merged[key]),
                    key,
                ),
            )[0]
            merged[target] += qty
            del merged[area]
        while True:
            small_areas = [
                area
                for area, qty in merged.items()
                if 0 < qty < min_boxes
            ]
            total_small = sum(merged[area] for area in small_areas)
            if len(small_areas) <= 1 or total_small < min_boxes:
                break
            candidates: list[str] = []
            for area, cap in caps.items():
                current = merged.get(area, 0)
                if area in small_areas:
                    room = int(cap) - current
                    if current + room >= total_small:
                        candidates.append(area)
                elif current > 0:
                    if int(cap) - current >= total_small:
                        candidates.append(area)
                elif int(cap) >= total_small:
                    candidates.append(area)
            if not candidates:
                break
            target = sorted(
                candidates,
                key=lambda key: (
                    0 if merged.get(key, 0) >= min_boxes else 1,
                    0 if key in small_areas else 1,
                    -int(weights.get(key, 0)),
                    -int(caps.get(key, 0)),
                    key,
                ),
            )[0]
            for area in list(small_areas):
                if area == target:
                    continue
                merged[target] += merged[area]
                del merged[area]
            if merged[target] > int(caps.get(target, 0)):
                break
        return merged

    def _six_bay_blocks_for_area(self, area_no: str, bay_keys: list[str]) -> dict[str, tuple[str, ...]]:
        blocks: dict[str, tuple[str, ...]] = {}
        start = 0
        block_index = 1
        while start <= len(bay_keys) - 6:
            members = tuple(bay_keys[start : start + 6])
            if self._is_preferred_six_bay_block(members):
                blocks[f"{area_no}-SB{block_index:02d}"] = members
                block_index += 1
                start += 6
            else:
                start += 1
        return blocks

    def _is_preferred_six_bay_block(self, bay_keys: tuple[str, ...]) -> bool:
        if len(bay_keys) != 6:
            return False
        member_set = set(bay_keys)
        large_starts = [
            key for key in bay_keys
            if (
                (self.bays[key].cap_by_size.get("40", 0) > 0 or self.bays[key].cap_by_size.get("45", 0) > 0)
                and self.bays[key].large_bay_partner_key in member_set
            )
        ]
        for left_index, left in enumerate(large_starts):
            left_pair = {left, self.bays[left].large_bay_partner_key}
            for right in large_starts[left_index + 1:]:
                right_pair = {right, self.bays[right].large_bay_partner_key}
                if left_pair & right_pair:
                    continue
                remaining = [key for key in bay_keys if key not in left_pair and key not in right_pair]
                if sum(1 for key in remaining if self.bays[key].cap_by_size.get("20", 0) > 0) >= 2:
                    return True
        return False

    def _group_sort_key(self, group: SmallBoxGroup) -> tuple[int, int, int, int, str, str, str]:
        return (
            self._source_rank_for_group(group),
            SIZE_ORDER.get(group.size, 3),
            len(self._area_weights(group)) if hasattr(self, "quota_by_key") else 0,
            -group.demand,
            group.voyage_id,
            group.group_id,
            group.port,
        )


def write_rows(path: str | Path, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with out.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_columns(path: str | Path, columns: Iterable[PlacementColumn]) -> None:
    rows = []
    for col in columns:
        row = {
            "column_id": col.column_id,
            "group_id": col.group_id,
            "demand_source": col.demand_source,
            "voyage_id": col.voyage_id,
            "flow": col.flow,
            "port": col.port,
            "size": col.size,
            "height": col.height,
            "area_no": col.area_no,
            "bay_no": col.bay_no,
            "row_allocation": ColumnGenerationPlanner._format_row_allocation(col.row_allocation),
            "six_bay_block_id": col.block_id,
            "quantity": col.quantity,
            "stack_units": col.stack_units,
            "intrinsic_cost": round(col.intrinsic_cost, 6),
        }
        for attr, value in sorted((col.attributes or {}).items()):
            if attr and attr not in row:
                row[attr] = value
        rows.append(row)
    write_rows(path, rows)


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _time_windows_overlap(start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime) -> bool:
    return start_a < end_b and start_b < end_a

