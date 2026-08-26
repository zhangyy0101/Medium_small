"""Shared yard-planning data, business rules, and output infrastructure."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Iterable

from .models import (
    EXPORT_GROUP_IDENTITY_ATTRIBUTES,
    EXPORT_VOYAGE_ROW_NO_MIX_ATTR,
    Bay,
    ExportGroup,
    ProblemData,
)

SIZE_ORDER = {"45": 0, "20": 1, "40": 2}
EXPORT_FLOWS = frozenset({"OF"})
MANDATORY_BAY_NO_MIX_ATTRS = ("IYC_CSZ_CSIZECD", "IYC_CHEIGHTCD")
SIZE_NO_MIX_ATTRS = frozenset({"IYC_CSZ_CSIZECD", "SIZE", "SIZE_MODE"})
HEIGHT_NO_MIX_ATTRS = frozenset({"IYC_CHEIGHTCD", "HEIGHT"})


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
    voyage_id: str
    flow: str
    port: str
    size: str
    big_plan_size: str
    height: str
    attributes: dict[str, str]
    area_no: str
    bay_key: str
    bay_no: str
    quantity: int
    stack_units: int
    row_allocation: tuple[tuple[str, str, int], ...]
    quota_key: tuple[str, str, str, str]
    group_key: tuple[str, ...]
    intrinsic_cost: float


@dataclass
class ColumnGenerationConfig:
    max_iterations: int = 60
    reduced_cost_tolerance: float = 1e-7
    total_time_limit: float = 60.0
    mip_time_limit: float = 15.0
    mip_gap: float = 0.01
    lp_method: int = 1
    solver_seed: int = 0
    solver_threads: int = 0
    verbose: bool = True
    # Legacy V5 row-location objective weights.  These are intentionally not
    # the V6 three-category objective; new V6 solvers consume V6ObjectiveConfig.
    area_dispersion_weight: float = 0.290
    row_dispersion_weight: float = 0.240
    existing_group_proximity_weight: float = 0.070
    area_guidance_weight: float = 0.270
    berth_distance_weight: float = 0.130


@dataclass
class ColumnGenerationResult:
    bay_summary_rows: list[dict]
    export_rows: list[dict]
    diagnostics: dict
    import_reservation_rows: list[dict] = field(default_factory=list)
    columns: list[PlacementColumn] = field(default_factory=list)


class YardPlanningBase:
    """Shared legacy V5 row-location layer and V6 input primitives.

    This class is not the V6 zone formulation.  New V6 solvers must use the
    contract and evaluator in :mod:`yard_planning.v6_model`.
    """

    def __init__(self, problem: ProblemData, config: ColumnGenerationConfig | None = None) -> None:
        self.problem = problem
        self.config = config or ColumnGenerationConfig()
        self.demand_stats: dict[str, int | str] = {}
        self.export_voyages = self._infer_export_voyages(problem)
        self.attribute_rules = problem.attribute_rules
        self.groups = sorted(self._build_planning_groups(), key=self._group_sort_key)
        self._validate_unique_operational_groups()
        self.groups_by_id = {group.group_id: group for group in self.groups}
        self.bays = problem.bays
        self.bays_by_area: dict[str, list[str]] = defaultdict(list)
        self.area_edge_bays: dict[str, set[str]] = defaultdict(set)
        self.quota_by_key: Counter[tuple[str, str, str, str]] = Counter()
        self.import_area_size_reference: Counter[tuple[str, str, str]] = Counter(
            {
                (str(flow), str(area_no), str(size)): int(qty)
                for (flow, area_no, size), qty in problem.import_area_size_reference.items()
                if int(qty) > 0
            }
        )
        self.import_total_by_flow_size: Counter[tuple[str, str]] = Counter(
            {
                (str(flow), str(size)): int(qty)
                for (flow, size), qty in problem.import_demand_by_flow_size.items()
                if int(qty) > 0
            }
        )
        if not self.import_total_by_flow_size:
            for (flow, _area_no, size), qty in self.import_area_size_reference.items():
                self.import_total_by_flow_size[(flow, size)] += int(qty)
        self.existing_group_area_load: Counter[tuple[str, ...]] = Counter(
            {
                tuple(key): int(value)
                for key, value in problem.existing_group_area_load.items()
                if int(value) > 0
            }
        )
        self.existing_group_bay_load: Counter[tuple[str, ...]] = Counter(
            {
                tuple(key): int(value)
                for key, value in problem.existing_group_bay_load.items()
                if int(value) > 0
            }
        )
        self.existing_group_bays: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
        self.existing_group_area_bays: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
        self.reachable_anchor_group_ids: set[str] = set()
        self.import_reservation_candidates: dict[
            tuple[str, str], list[tuple[str, int]]
        ] = {}
        self._final_import_reservation: Counter[tuple[str, str, str]] = Counter()
        for (*group_key, area_no, bay_key), value in self.existing_group_bay_load.items():
            if value > 0:
                group_tuple = tuple(group_key)
                self.existing_group_bays[group_tuple].add(str(bay_key))
                self.existing_group_area_bays[group_tuple + (str(area_no),)].add(str(bay_key))
        self.group_demand = {group.group_id: int(group.demand) for group in self.groups}
        self.voyage_flow_size_demand: Counter[tuple[str, str, str]] = Counter()
        self._base_placements_by_group: dict[
            str, tuple[PlacementColumn, ...]
        ] = {}
        self._base_location_capacity_cache: dict[tuple, int] = {}
        self._columns: list[PlacementColumn] = []
        self._column_keys: set[tuple[str, str, int, tuple[tuple[str, str, int], ...]]] = set()
        self._candidate_cache: dict[str, list[tuple[str, int, float]]] = {}
        self._objective_scales: dict[str, float] = {}
        self._berth_distance_bounds: dict[str, tuple[float, float]] = {}
        self._base_feasible_placement_count = 0
        self._master_bay_capacity_keys: set[str] = set()
        self._master_bay_size_keys: set[tuple[str, str]] = set()
        self._master_row_capacity_keys: set[tuple[str, str]] = set()
        self._master_row_size_keys: set[tuple[str, str, str]] = set()
        self._master_stack_keys: set[tuple[str, str, str]] = set()
        self._master_stack_sample_group: dict[tuple[str, str, str], str] = {}
        self._master_area_guidance_keys: set[tuple[str, str, str, str]] = set()
        self._master_bay_attr_choice_keys: set[tuple[str, str, str, str]] = set()
        self._master_row_attr_choice_keys: set[tuple[str, str, str, str, str]] = set()
        self._master_bay_attr_big_m: dict[tuple[str, str, str, str], int] = {}
        self._master_row_attr_big_m: dict[tuple[str, str, str, str, str], int] = {}
        self._validate_column_generation_config()
        self._prepare_yard_indexes()
        self._prepare_import_reservation_candidates()
        self._prepare_quota()
        self._prepare_reachable_anchor_groups()
        self._validate_objective_weights()
        self._prepare_berth_distance_bounds()
        for group in self.groups:
            self.voyage_flow_size_demand[(group.voyage_id, group.status, self._big_plan_size(group.size))] += group.demand

    @property
    def columns(self) -> list[PlacementColumn]:
        return self._columns

    def _validate_column_generation_config(self) -> None:
        if int(self.config.max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")

    def _build_planning_groups(self) -> list[ExportGroup]:
        """Return declared, not-yet-arrived export groups only."""
        groups = [
            group
            for group in self.problem.export_groups
            if str(group.status) in EXPORT_FLOWS and int(group.demand) > 0
        ]
        boxes = sum(int(group.demand) for group in groups)
        self.demand_stats = {
            "declared_export_group_count": len(groups),
            "declared_export_box_count": boxes,
            "planning_group_count": len(groups),
            "planning_box_count": boxes,
            "demand_policy": "declared_export_only",
        }
        return groups

    def _validate_unique_operational_groups(self) -> None:
        """Enforce the fixed V6 export-group identity contract."""
        group_ids: set[str] = set()
        operational_keys: dict[tuple[str, ...], str] = {}
        for group in self.groups:
            if group.group_id in group_ids:
                raise ValueError(f"duplicate export group_id: {group.group_id}")
            group_ids.add(group.group_id)
            key = self._operational_group_key(group)
            previous = operational_keys.get(key)
            if previous is not None:
                raise ValueError(
                    "declared export demand must be aggregated to the single "
                    "operational group definition before optimization: "
                    f"groups={previous},{group.group_id}, key={key}"
                )
            operational_keys[key] = group.group_id

        # Size and height are bay-level no-mix attributes; voyage and discharge
        # port are row-level no-mix attributes.  Therefore every pair of
        # distinct fixed export groups must be incompatible on one physical
        # row.  Fail at the input boundary if a future configuration weakens
        # that invariant silently.
        for first_index, first in enumerate(self.groups):
            for second in self.groups[first_index + 1 :]:
                if not self._groups_are_incompatible_on_one_row(first, second):
                    raise ValueError(
                        "V6 requires different export groups to be mutually "
                        "exclusive on every physical row: "
                        f"groups={first.group_id},{second.group_id}"
                    )

    def _assemble_result(
        self,
        selected: Counter[int],
        diagnostics: dict,
    ) -> ColumnGenerationResult:
        """Build and independently validate common solver output artifacts."""

        import_reservation_rows = self._make_import_reservation_rows()
        diagnostics["import_capacity_reservation"].update(
            self._import_reservation_diagnostics()

        )
        objective_components = self._selected_objective_components(selected)
        diagnostics["final_business_objective"] = objective_components["weighted_total"]
        diagnostics["final_business_objective_components"] = objective_components
        diagnostics["independent_solution_validation"] = self._validate_final_solution(
            selected
        )

        export_rows = self._make_export_rows(selected)

        bay_summary_rows = self._make_bay_summary_rows(selected)
        consistency_stats = self._row_area_summary_consistency_stats(export_rows, bay_summary_rows)
        bay_consistency_stats = self._row_bay_summary_consistency_stats(export_rows, bay_summary_rows)
        operational_group_dispersion = self._operational_group_dispersion_stats(export_rows)
        legacy_guidance_diagnostics = {}
        if self.quota_by_key:
            legacy_guidance_diagnostics = {
                "area_summary_big_plan_inheritance": (
                    self._area_summary_big_plan_inheritance_stats(
                        bay_summary_rows
                    )
                ),
                "final_area_summary_inheritance_energy_components": (
                    self._area_summary_inheritance_energy_components(
                        bay_summary_rows
                    )
                ),
            }
        diagnostics.update(
            {
                "expanded_integer_row_location_count": len(self._columns),
                "selected_location_count": sum(1 for qty in selected.values() if qty > 0),
                "summary_granularity": "bay",
                "export_row_count": len(export_rows),
                "bay_summary_row_count": len(bay_summary_rows),
                "planned_export_boxes": sum(int(row["planned_boxes"]) for row in export_rows),
                "operational_group_dispersion": operational_group_dispersion,
                "capacity_reservation_margins": self._capacity_reservation_margins(selected),
                "export_voyage_row_no_mix": self._export_voyage_row_no_mix_stats(selected),
                **legacy_guidance_diagnostics,
                **consistency_stats,
                **bay_consistency_stats,
            }
        )
        return ColumnGenerationResult(
            bay_summary_rows=bay_summary_rows,
            export_rows=export_rows,
            diagnostics=diagnostics,
            import_reservation_rows=import_reservation_rows,
            columns=self._columns,
        )

    def _validate_final_solution(
        self,
        selected: Counter[int],
    ) -> dict[str, int | bool]:
        """Recheck the incumbent independently of the solver model."""
        normalized = Counter(
            {idx: int(round(value)) for idx, value in selected.items() if int(round(value)) > 0}
        )
        errors = self._import_reservation_validation_errors()
        try:
            repaired, _state, placed = self._selection_state(normalized)
        except RuntimeError as exc:
            repaired, placed = Counter(), Counter()
            errors.append(str(exc))
        if repaired != normalized:
            errors.append("selected row locations violate a physical or no-mix rule")
        for group in self.groups:
            assigned = int(placed.get(group.group_id, 0))
            if assigned != int(group.demand):
                errors.append(
                    f"demand balance failed for {group.group_id}: "
                    f"assigned={assigned}, demand={group.demand}"
                )
        for idx in normalized:
            col = self._columns[idx]
            if col.size == "45" and col.bay_key not in self.area_edge_bays.get(col.area_no, set()):
                errors.append(
                    f"45-ft row location {col.column_id} is not on an edge large bay"
                )
        if errors:
            raise RuntimeError("Independent solution validation failed: " + "; ".join(errors[:10]))
        return {
            "passed": True,
            "selected_locations_checked": len(normalized),
            "groups_checked": len(self.groups),
            "assigned_boxes_checked": int(sum(placed.values())),
            "import_reserved_boxes_checked": int(sum(self._final_import_reservation.values())),
        }

    def _import_reservation_validation_errors(self) -> list[str]:
        errors: list[str] = []
        actual: Counter[tuple[str, str]] = Counter()
        for (flow, size, bay_key), qty in self._final_import_reservation.items():
            if qty <= 0:
                continue
            actual[(flow, size)] += int(qty)
            bay = self.bays.get(bay_key)
            if bay is None:
                errors.append(f"unknown import reserve bay: {bay_key}")
                continue
            if flow not in self.problem.area_functions.get(bay.area_no, set()):
                errors.append(
                    f"import reserve violates area function: flow={flow}, area={bay.area_no}"
                )
            if self._import_reservation_capacity(bay_key, size) <= 0:
                errors.append(
                    f"import reserve violates bay size: size={size}, bay={bay_key}"
                )
        for key, required in self.import_total_by_flow_size.items():
            if int(actual[key]) != int(required):
                errors.append(
                    f"import reserve total mismatch: flow={key[0]}, size={key[1]}, "
                    f"reserved={actual[key]}, required={required}"
                )
        for key, qty in actual.items():
            if key not in self.import_total_by_flow_size and qty > 0:
                errors.append(
                    f"unexpected import reserve total: flow={key[0]}, size={key[1]}, reserved={qty}"
                )
        return errors

    def _capacity_reservation_margins(self, selected: Counter[int]) -> dict[str, dict[str, int]]:
        """Report joint export/import capacity use after the final plan."""
        export_bay_load: Counter[str] = Counter()
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            qty = int(col.quantity) * int(chosen)
            for bay_key in self._placement_footprint_keys(col.bay_key, col.size):
                export_bay_load[bay_key] += qty

        import_bay_load: Counter[str] = Counter()
        import_area_boxes: Counter[str] = Counter()
        import_area_slot_units: Counter[str] = Counter()
        for (_flow, size, bay_key), qty in self._final_import_reservation.items():
            if qty <= 0:
                continue
            area_no = self.bays[bay_key].area_no
            import_area_boxes[area_no] += int(qty)
            footprint = self._placement_footprint_keys(bay_key, size)
            import_area_slot_units[area_no] += int(qty) * len(footprint)
            for footprint_key in footprint:
                import_bay_load[footprint_key] += int(qty)

        areas = sorted(
            set(self.bays_by_area)
            & (
                {self.bays[key].area_no for key in export_bay_load}
                | set(import_area_boxes)
                | {area for _flow, area, _size in self.import_area_size_reference}
            )
        )
        result: dict[str, dict[str, int]] = {}
        for area_no in areas:
            keys = self.bays_by_area.get(area_no, [])
            physical = sum(int(self.bays[key].physical_capacity) for key in keys)
            export_slots = sum(export_bay_load[key] for key in keys)
            import_slots = sum(import_bay_load[key] for key in keys)
            reference_boxes = sum(
                int(qty)
                for (flow, reference_area, size), qty in self.import_area_size_reference.items()
                if reference_area == area_no
            )
            values = {
                "physical_slot_capacity": int(physical),
                "export_slot_use": int(export_slots),
                "import_reserved_slot_use": int(import_slots),
                "residual_slot_units": int(physical - export_slots - import_slots),
                "import_reserved_boxes": int(import_area_boxes[area_no]),
            }
            if self.import_area_size_reference:
                values["import_reference_boxes"] = int(reference_boxes)
            result[area_no] = values
        return result

    def _make_import_reservation_rows(self) -> list[dict]:
        rows: list[dict] = []
        for (flow, size, bay_key), qty in sorted(self._final_import_reservation.items()):
            if qty <= 0 or bay_key not in self.bays:
                continue
            bay = self.bays[bay_key]
            rows.append(
                {
                    "flow": flow,
                    "size": size,
                    "area_no": bay.area_no,
                    "bay_key": bay_key,
                    "bay_no": bay.bay_no,
                    "reserved_boxes": int(qty),
                    "footprint_slot_units": int(qty)
                    * len(self._placement_footprint_keys(bay_key, size)),
                    "reservation_scope": "anonymous_capacity",
                }
            )
        return rows

    def _import_reservation_diagnostics(self) -> dict[str, object]:
        actual: Counter[tuple[str, str, str]] = Counter()
        for (flow, size, bay_key), qty in self._final_import_reservation.items():
            if qty > 0 and bay_key in self.bays:
                actual[(flow, self.bays[bay_key].area_no, size)] += int(qty)
        keys = (
            set(actual) | set(self.import_area_size_reference)
            if self.import_area_size_reference
            else set()
        )
        l1 = sum(
            abs(int(actual.get(key, 0)) - int(self.import_area_size_reference.get(key, 0)))
            for key in keys
        )
        diagnostics = {
            "reserved_boxes": int(sum(actual.values())),
            "reserved_by_flow_area_size": {
                f"{flow}|{area}|{size}": int(qty)
                for (flow, area, size), qty in sorted(actual.items())
                if qty > 0
            },
            "area_reference_used": bool(self.import_area_size_reference),
        }
        if self.import_area_size_reference:
            diagnostics.update(
                {
                    "area_l1_deviation": int(l1),
                    "boxes_shifted_between_areas": int(l1 // 2),
                }
            )
        return diagnostics

    @staticmethod
    def _operational_group_dispersion_stats(rows: list[dict]) -> dict[str, int | float | str]:
        areas_by_group: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
        rows_by_group: defaultdict[tuple[str, ...], set[tuple[str, str, str]]] = defaultdict(set)
        for row in rows:
            group_key = (
                str(row.get("voyage_id", "")),
                str(row.get("flow", "")),
                str(row.get("port", "")),
                str(row.get("size", "")),
                str(row.get("height", "")),
            )
            area_no = str(row.get("area_no", ""))
            bay_no = str(row.get("bay_no", ""))
            row_no = str(row.get("row_no", ""))
            if area_no:
                areas_by_group[group_key].add(area_no)
            if area_no and bay_no and row_no:
                rows_by_group[group_key].add((area_no, bay_no, row_no))
        group_count = len(set(areas_by_group) | set(rows_by_group))
        area_counts = [len(areas_by_group[key]) for key in set(areas_by_group) | set(rows_by_group)]
        row_counts = [len(rows_by_group[key]) for key in set(areas_by_group) | set(rows_by_group)]
        return {
            "group_definition": "voyage|flow|destination_port|size|height",
            "group_count": group_count,
            "total_used_areas": sum(area_counts),
            "total_used_rows": sum(row_counts),
            "max_areas_per_group": max(area_counts, default=0),
            "max_rows_per_group": max(row_counts, default=0),
            "average_areas_per_group": round(sum(area_counts) / group_count, 3) if group_count else 0.0,
            "average_rows_per_group": round(sum(row_counts) / group_count, 3) if group_count else 0.0,
        }

    def _selected_objective_components(
        self,
        selected: Counter[int],
    ) -> dict[str, float | dict[str, float]]:
        """Evaluate raw, normalized and weighted business criteria."""
        actual_quota: Counter[tuple[str, str, str, str]] = Counter()
        used_group_area: set[tuple[tuple[str, ...], str]] = set()
        used_group_row: set[tuple[tuple[str, ...], str, str]] = set()
        used_groups: set[tuple[str, ...]] = set()
        proximity_sum = 0.0
        berth_distance_sum = 0.0
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            multiplier = int(chosen)
            qty = col.quantity * multiplier
            actual_quota[col.quota_key] += qty
            used_groups.add(col.group_key)
            used_group_area.add((col.group_key, col.area_no))
            for bay_key, row_no, row_qty in col.row_allocation:
                # A 40/45-ft unit occupies a paired physical footprint but is
                # one operational row assignment, identified by its primary
                # large-bay key.
                if row_qty > 0 and bay_key == col.bay_key:
                    used_group_row.add((col.group_key, bay_key, row_no))
            group = self.groups_by_id[col.group_id]
            proximity_sum += self._normalized_existing_proximity(group, col.bay_key) * qty
            berth_distance_sum += self._normalized_berth_distance(col.voyage_id, col.area_no) * qty

        target_keys = {
            key
            for key in actual_quota
            if self._has_area_guidance(key[0], key[1], key[3])
        }
        for key, qty in self.quota_by_key.items():
            voyage_id, flow, _area_no, big_size = key
            if qty > 0 and self.voyage_flow_size_demand[(voyage_id, flow, big_size)] > 0:
                target_keys.add(key)
        export_guidance_l1 = 0.0
        for voyage_id, flow, area_no, big_size in target_keys:
            target = self._area_size_target(voyage_id, flow, area_no, big_size)
            export_guidance_l1 += abs(
                actual_quota.get((voyage_id, flow, area_no, big_size), 0) - target
            )
        import_actual_by_area: Counter[tuple[str, str, str]] = Counter()
        for (flow, size, bay_key), qty in self._final_import_reservation.items():
            if qty > 0 and bay_key in self.bays:
                import_actual_by_area[(flow, self.bays[bay_key].area_no, size)] += int(qty)
        import_keys = (
            set(self.import_area_size_reference) | set(import_actual_by_area)
            if self.import_area_size_reference
            else set()
        )
        import_guidance_l1 = sum(
            abs(
                int(import_actual_by_area.get(key, 0))
                - int(self.import_area_size_reference.get(key, 0))
            )
            for key in import_keys
        )
        guidance_l1 = float(export_guidance_l1 + import_guidance_l1)

        raw = {
            "extra_operational_group_areas": float(max(0, len(used_group_area) - len(used_groups))),
            "extra_operational_group_rows": float(max(0, len(used_group_row) - len(used_groups))),
            "existing_group_normalized_distance_sum": float(proximity_sum),
            "area_guidance_l1_deviation": float(guidance_l1),
            "export_area_guidance_l1_deviation": float(export_guidance_l1),
            "import_reservation_area_l1_deviation": float(import_guidance_l1),
            "berth_normalized_distance_sum": float(berth_distance_sum),
        }
        normalized = {
            "area_dispersion": raw["extra_operational_group_areas"] / self._objective_scale("area_dispersion"),
            "row_dispersion": raw["extra_operational_group_rows"] / self._objective_scale("row_dispersion"),
            "existing_group_proximity": raw["existing_group_normalized_distance_sum"] / self._objective_scale("existing_group_proximity"),
            "area_guidance": raw["area_guidance_l1_deviation"] / self._objective_scale("area_guidance_l1"),
            "berth_distance": raw["berth_normalized_distance_sum"] / self._objective_scale("berth_distance"),
        }
        weights = self._objective_weights()
        weighted = {key: normalized[key] * weights[key] for key in normalized}
        return {
            "raw": {key: round(value, 8) for key, value in raw.items()},
            "normalized": {key: round(value, 8) for key, value in normalized.items()},
            "weighted": {key: round(value, 8) for key, value in weighted.items()},
            "weighted_total": round(sum(weighted.values()), 8),
        }

    def _selected_solution_energy(self, selected: Counter[int]) -> float:
        return float(self._selected_objective_components(selected)["weighted_total"])

    @staticmethod
    def _absolute_deviation_auxiliary_slack(
        solver_objective: float,
        reconstructed_objective: float,
        *,
        context: str,
        tolerance: float = 1e-6,
    ) -> float:
        """Validate and measure removable slack in L1 auxiliary variables.

        A time-limited MIP incumbent can retain equal positive and negative
        deviation values although reducing both preserves feasibility and
        improves its objective.  The row plan reconstruction evaluates the
        canonical minimum deviation for the same integer assignment.  It may
        therefore be lower than Gurobi's stored incumbent, but never higher.
        """
        slack = float(solver_objective) - float(reconstructed_objective)
        if slack < -abs(float(tolerance)):
            raise RuntimeError(
                f"{context} objective is below the reconstructed row plan: "
                f"model={solver_objective}, "
                f"reconstructed={reconstructed_objective}"
            )
        return max(0.0, slack)

    @staticmethod
    def _row_area_summary_consistency_stats(export_rows: list[dict], bay_summary_rows: list[dict]) -> dict[str, int]:
        row_counter: Counter[tuple[str, str, str, str, str]] = Counter()
        summary_counter: Counter[tuple[str, str, str, str, str]] = Counter()
        for row in export_rows:
            key = (
                str(row.get("voyage_id", "")),
                str(row.get("flow", "")),
                str(row.get("port", "")),
                str(row.get("size", "")),
                str(row.get("area_no", "")),
            )
            row_counter[key] += int(row.get("planned_boxes", 0) or 0)
        for row in bay_summary_rows:
            key = (
                str(row.get("voyage_id", "")),
                str(row.get("flow", "")),
                str(row.get("port", "")),
                str(row.get("size", "")),
                str(row.get("area_no", "")),
            )
            summary_counter[key] += int(row.get("planned_boxes", 0) or 0)
        violations = 0
        shortage = 0
        for key, qty in row_counter.items():
            excess = qty - summary_counter.get(key, 0)
            if excess > 0:
                violations += 1
                shortage += excess
        return {
            "row_area_summary_consistency_violations": violations,
            "row_area_summary_consistency_shortage_boxes": shortage,
        }

    @staticmethod
    def _row_bay_summary_consistency_stats(export_rows: list[dict], bay_summary_rows: list[dict]) -> dict[str, int]:
        row_counter: Counter[tuple[str, str, str, str, str, str]] = Counter()
        summary_counter: Counter[tuple[str, str, str, str, str, str]] = Counter()

        for row in export_rows:
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
            row_counter[key] += int(row.get("planned_boxes", 0) or 0)
        for row in bay_summary_rows:
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

            summary_counter[key] += int(row.get("planned_boxes", 0) or 0)
        violations = 0
        shortage = 0
        for key, qty in row_counter.items():
            excess = qty - summary_counter.get(key, 0)
            if excess > 0:
                violations += 1
                shortage += excess
        return {
            "row_bay_summary_consistency_violations": violations,
            "row_bay_summary_consistency_shortage_boxes": shortage,
        }

    def _gurobi_import_reservation_values(
        self,
        model,
        variables: dict,
    ) -> Counter[tuple[str, str, str]]:
        return Counter(
            {
                key: int(round(self._gurobi_value(model, var)))
                for key, var in variables.get("import_reserve", {}).items()
                if self._gurobi_value(model, var) > 1e-6
            }
        )

    def _configure_gurobi_output(self, model) -> None:
        if not self.config.verbose:
            try:
                model.hideOutput()
                return
            except Exception:
                pass
            self._try_set_gurobi_param(model, "OutputFlag", 0)

    @staticmethod
    def _try_set_gurobi_param(model, name: str, value: object) -> bool:
        try:
            YardPlanningBase._set_gurobi_param(model, name, value)
            return True
        except Exception:
            return False

    @staticmethod
    def _set_gurobi_param(model, name: str, value: object) -> None:
        model.setParam(name, value)

    @staticmethod
    def _gurobi_status_name(model) -> str:
        return str(model.getStatusName()).lower()

    @staticmethod
    def _gurobi_solution_count(model) -> int:
        try:
            return int(model.getSolutionCount())
        except Exception:
            return 0

    @staticmethod
    def _gurobi_objective_value(model) -> float:
        try:
            return float(model.getObjectiveValue())
        except Exception:
            return float("nan")

    @staticmethod
    def _gurobi_gap(model) -> float:
        try:
            return float(model.getMipGap())
        except Exception:
            return 0.0

    @staticmethod
    def _gurobi_dual_bound(model) -> float:
        try:
            return float(model.getBestBound())
        except Exception:
            return float("nan")

    @staticmethod
    def _gurobi_value(model, var) -> float:
        return float(model.getValue(var))

    @staticmethod
    def _free_gurobi_model(model) -> None:
        try:
            model.dispose()
        except Exception:
            pass

    def _initialize_phase_one_objective(
        self,
        model,
        variables: dict,
    ) -> None:
        """Capture the business objective and activate feasibility Phase I."""
        model.update()
        variables["_phase_one_objective_variables"] = tuple(
            variables["phase_one_artificial"].values()
        )
        variables["_business_objective_terms"] = [
            (variable, model.getVarObjective(variable))
            for variable in model.getVars()
            if abs(model.getVarObjective(variable)) > 0.0
        ]
        self._activate_master_objective(
            model,
            variables,
            objective="phase_one",
        )

    @staticmethod
    def _activate_master_objective(
        model,
        variables: dict,
        objective: str,
    ) -> None:
        """Switch the persistent LP from feasibility to business cost."""
        if objective not in {"phase_one", "business"}:
            raise ValueError(f"unknown master objective: {objective}")
        for variable in variables.get("_phase_one_objective_variables", ()):
            model.setVarObjective(
                variable, 1.0 if objective == "phase_one" else 0.0
            )
        for variable, coefficient in variables.get(
            "_business_objective_terms", ()
        ):
            model.setVarObjective(
                variable,
                coefficient if objective == "business" else 0.0,
            )
        model.update()

    def _business_objective_specification(self) -> dict:
        return {
            "type": "normalized_weighted_business_objective",
            "hard_demand_balance": True,
            "weights": self._objective_weights(),
        }

    def _add_import_reference_deviation(
        self,
        quicksum,
        model,
        import_by_flow_area_size: dict[tuple[str, str, str], list],
        objective_mode: str,
    ) -> dict[tuple[str, str, str], object]:
        """Penalize the minimum area adjustment of anonymous import reserve."""
        if not self.import_area_size_reference:
            return {}
        keys = set(self.import_area_size_reference) | set(import_by_flow_area_size)
        balance = {}
        penalty = 0.0 if objective_mode == "phase_one" else self._area_guidance_penalty()
        for flow, area_no, size in sorted(keys):
            target = int(self.import_area_size_reference.get((flow, area_no, size), 0))
            pos = model.addVar(
                lb=0.0,
                obj=penalty,
                name=f"import_guide_pos_{flow}_{area_no}_{size}",
            )
            neg = model.addVar(
                lb=0.0,
                obj=penalty,
                name=f"import_guide_neg_{flow}_{area_no}_{size}",
            )
            actual = quicksum(import_by_flow_area_size.get((flow, area_no, size), []))
            balance[(flow, area_no, size)] = model.addConstr(
                actual - target == pos - neg,
                name=f"import_guide_balance_{flow}_{area_no}_{size}",
            )
        return balance

    def _initialize_location_pool(self) -> None:
        """Initialize the row-location pool used for a final solution."""
        self._columns.clear()
        self._column_keys.clear()

    def _placement_master_coefficients(
        self,
        col: PlacementColumn,
    ) -> dict[str, Counter[object]]:
        out: dict[str, Counter[object]] = defaultdict(Counter)
        quantity = int(col.quantity)
        footprint = self._placement_footprint_keys(col.bay_key, col.size)
        stack_value = self._row_mix_key_for_column(col)
        for footprint_key in footprint:
            out["bay_capacity_limit"][footprint_key] += quantity
            out["bay_port_stack_link"][(footprint_key, stack_value, col.size)] += quantity
            for attr in self._bay_no_mix_attrs_for_column(col):
                scope = self._attr_voyage_scope(attr, col.voyage_id)
                out["bay_attr_link"][(
                    footprint_key,
                    attr,
                    scope,
                    self._column_attr_value(col, attr),
                )] += quantity
        out["bay_size_limit"][(col.bay_key, col.size)] += quantity
        for footprint_key, row_no, qty in col.row_allocation:
            out["row_capacity_limit"][(footprint_key, row_no)] += int(qty)
            out["row_size_limit"][(footprint_key, row_no, col.size)] += int(qty)
            for attr in self._row_no_mix_attrs_for_column(col):
                scope = self._attr_voyage_scope(attr, col.voyage_id)
                out["row_attr_link"][(
                    footprint_key,
                    row_no,
                    attr,
                    scope,
                    self._column_attr_value(col, attr),
                )] += int(qty)
        out["area_guidance_balance"][col.quota_key] += quantity
        return out

    def _master_dual_snapshot(
        self,
        model,
        constraints: dict,
    ) -> dict[tuple[str, object], float]:
        duals: dict[tuple[str, object], float] = {}
        for section, rows in constraints.items():
            if not isinstance(rows, dict):
                continue
            for key, row in rows.items():
                if row is not None:
                    duals[(section, key)] = float(model.getLinearDual(row))
        return duals

    def _add_bay_compatibility_constraints(
        self,
        quicksum,
        model,
        location_variables,
        items_by_key,
        relax: bool,
        auxiliary_integer: dict[tuple[str, ...], object] | None = None,
    ) -> dict[str, dict]:
        uses_by_scope: defaultdict[tuple[str, str, str], list] = defaultdict(list)
        links = {}
        choices = {}
        for key in sorted(self._master_bay_attr_choice_keys):
            bay_key, attr, scope, value = key
            use = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="C" if relax else "B",
                name=f"bay_use_{self._key_name(key)}",
            )
            if auxiliary_integer is not None:
                auxiliary_integer[("bay_attr", *key)] = use
            items = items_by_key.get(key, [])
            links[key] = model.addConstr(
                quicksum(
                    coefficient * location_variables[idx]
                    for idx, coefficient in items
                )
                <= self._master_bay_attr_big_m[key] * use,
                name=f"bay_attr_link_{self._key_name(key)}",
            )
            uses_by_scope[(bay_key, attr, scope)].append(use)
        for key, uses in uses_by_scope.items():
            choices[key] = model.addConstr(
                quicksum(uses) <= 1,
                name=f"bay_attr_one_{self._key_name(key)}",
            )
        return {"bay_attr_link": links, "bay_attr_one": choices}

    def _add_row_compatibility_constraints(
        self,
        quicksum,
        model,
        location_variables,
        items_by_key,
        relax: bool,
        auxiliary_integer: dict[tuple[str, ...], object] | None = None,
    ) -> dict[str, dict]:
        uses_by_scope: defaultdict[tuple[str, str, str, str], list] = (
            defaultdict(list)
        )
        links = {}
        choices = {}
        for key in sorted(self._master_row_attr_choice_keys):
            bay_key, row_no, attr, scope, value = key
            use = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="C" if relax else "B",
                name=f"row_use_{self._key_name(key)}",
            )
            if auxiliary_integer is not None:
                auxiliary_integer[("row_attr", *key)] = use
            items = items_by_key.get(key, [])
            links[key] = model.addConstr(
                quicksum(
                    coefficient * location_variables[idx]
                    for idx, coefficient in items
                )
                <= self._master_row_attr_big_m[key] * use,
                name=f"row_attr_link_{self._key_name(key)}",
            )
            uses_by_scope[(bay_key, row_no, attr, scope)].append(use)
        for key, uses in uses_by_scope.items():
            choices[key] = model.addConstr(
                quicksum(uses) <= 1,
                name=f"row_attr_one_{self._key_name(key)}",
            )
        return {"row_attr_link": links, "row_attr_one": choices}

    @staticmethod
    def _seconds_until(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, float(deadline) - perf_counter())

    def _set_remaining_time_limit(
        self, model, deadline: float | None
    ) -> bool:
        remaining = self._seconds_until(deadline)
        if remaining is None:
            return True
        if remaining <= 1e-6:
            return False
        self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
        return True

    def _iter_feasible_base_placements(
        self,
        group: ExportGroup,
    ) -> Iterable[PlacementColumn]:
        """Enumerate feasible one-box row locations used by both models."""
        for bay_key, _max_qty, _unused_cost in self._candidate_bays_for_group(group):
            bay = self.bays[bay_key]
            footprint = self._placement_footprint_keys(bay_key, group.size)
            if not footprint:
                continue
            per_bay = {
                key: dict(self._row_capacity_items_for_group(key, group.size, group))
                for key in footprint
            }
            if any(not values for values in per_bay.values()):
                continue
            common_rows = sorted(
                set.intersection(*(set(values) for values in per_bay.values())),
                key=self._row_sort_key,
            )
            for row_no in common_rows:
                if min(per_bay[key][row_no] for key in footprint) <= 0:
                    continue
                allocation = self._row_allocation_signature(
                    tuple((key, row_no, 1) for key in footprint)
                )
                yield PlacementColumn(
                    column_id="",
                    group_id=group.group_id,
                    voyage_id=group.voyage_id,
                    flow=group.status,
                    port=group.port,
                    size=group.size,
                    big_plan_size=self._big_plan_size(group.size),
                    height=group.height,
                    attributes=dict(group.attributes),
                    area_no=bay.area_no,
                    bay_key=bay_key,
                    bay_no=bay.bay_no,
                    quantity=1,
                    stack_units=1,
                    row_allocation=allocation,
                    quota_key=self._quota_key(group, bay.area_no),
                    group_key=self._operational_group_key(group),
                    intrinsic_cost=(
                        self._column_base_cost(group, bay_key)
                        + self._berth_distance_cost(group.voyage_id, bay.area_no, 1)
                    ),
                )

    def _base_placements_for_group(
        self,
        group: ExportGroup,
    ) -> tuple[PlacementColumn, ...]:
        cached = self._base_placements_by_group.get(group.group_id)
        if cached is None:
            cached = tuple(self._iter_feasible_base_placements(group))
            self._base_placements_by_group[group.group_id] = cached
        return cached

    @staticmethod
    def _base_placement_static_key(column: PlacementColumn) -> tuple:
        """Identify a row location independently of its assigned quantity."""
        return (
            column.group_id,
            column.bay_key,
            tuple(
                (str(bay_key), str(row_no))
                for bay_key, row_no, _quantity in column.row_allocation
            ),
        )

    def _feasible_bay_capacity_without_demand(self, group: ExportGroup, bay_key: str) -> int:
        footprint = self._placement_footprint_keys(bay_key, group.size)
        if not footprint:
            return 0
        bay = self.bays[bay_key]
        return max(
            0,
            min(
                int(bay.cap_by_size.get(group.size, 0)),
                *(int(self.bays[key].physical_capacity) for key in footprint),
                *(
                    self._stack_count_for_group(key, group.size, group)
                    * self._stack_unit_capacity_for_group(key, group.size, group)
                    for key in footprint
                ),
            ),
        )

    def _base_location_capacity(
        self,
        group: ExportGroup,
        column: PlacementColumn,
    ) -> int:
        cache_key = self._base_placement_static_key(column)
        cached = self._base_location_capacity_cache.get(cache_key)
        if cached is not None:
            return int(cached)
        row_capacities: list[int] = []
        for footprint_key, row_no, _quantity in column.row_allocation:
            row_cap = dict(
                self._row_capacity_items_for_group(
                    footprint_key,
                    group.size,
                    group,
                )
            ).get(str(row_no), 0)
            row_capacities.append(int(row_cap))
        if not row_capacities:
            return 0
        capacity = max(
            0,
            min(
                self._feasible_bay_capacity_without_demand(group, column.bay_key),
                *row_capacities,
            ),
        )
        self._base_location_capacity_cache[cache_key] = int(capacity)
        return int(capacity)

    @staticmethod
    def _column_identity(
        column: PlacementColumn,
    ) -> tuple[str, str, int, tuple[tuple[str, str, int], ...]]:
        return (
            column.group_id,
            column.bay_key,
            int(column.quantity),
            column.row_allocation,
        )

    def _append_generated_column(self, candidate: PlacementColumn) -> int:
        key = self._column_identity(candidate)
        if key in self._column_keys:
            raise ValueError(f"duplicate generated column: {key}")
        column = replace(candidate, column_id=f"C{len(self._columns) + 1:07d}")
        index = len(self._columns)
        self._columns.append(column)
        self._column_keys.add(key)
        return index

    def _prepare_master_index_sets(self) -> None:
        """Create fixed resource rows and the feasible placement pool."""
        self._master_bay_capacity_keys.clear()
        self._master_bay_size_keys.clear()
        self._master_row_capacity_keys.clear()
        self._master_row_size_keys.clear()
        self._master_stack_keys.clear()
        self._master_stack_sample_group.clear()
        self._master_area_guidance_keys.clear()
        self._master_bay_attr_choice_keys.clear()
        self._master_row_attr_choice_keys.clear()
        self._master_bay_attr_big_m.clear()
        self._master_row_attr_big_m.clear()
        self._base_placements_by_group.clear()
        self._base_location_capacity_cache.clear()
        self._base_feasible_placement_count = 0
        bay_attr_groups: defaultdict[tuple[str, str, str, str], set[str]] = defaultdict(set)
        row_attr_groups: defaultdict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
        for group in self.groups:
            placements = tuple(self._iter_feasible_base_placements(group))
            self._base_placements_by_group[group.group_id] = placements
            for column in placements:
                self._base_feasible_placement_count += 1
                footprint = self._placement_footprint_keys(column.bay_key, column.size)
                self._master_bay_capacity_keys.update(footprint)
                self._master_bay_size_keys.add((column.bay_key, column.size))
                stack_value = self._row_mix_key_for_column(column)
                for footprint_key in footprint:
                    stack_key = (footprint_key, stack_value, column.size)
                    self._master_stack_keys.add(stack_key)
                    self._master_stack_sample_group.setdefault(stack_key, group.group_id)
                    for attr in self._bay_no_mix_attrs_for_column(column):
                        scope = self._attr_voyage_scope(attr, column.voyage_id)
                        attr_key = (
                            footprint_key,
                            attr,
                            scope,
                            self._column_attr_value(column, attr),
                        )
                        self._master_bay_attr_choice_keys.add(attr_key)
                        bay_attr_groups[attr_key].add(group.group_id)
                for footprint_key, row_no, _qty in column.row_allocation:
                    self._master_row_capacity_keys.add((footprint_key, row_no))
                    self._master_row_size_keys.add((footprint_key, row_no, column.size))
                    for attr in self._row_no_mix_attrs_for_column(column):
                        scope = self._attr_voyage_scope(attr, column.voyage_id)
                        attr_key = (
                            footprint_key,
                            row_no,
                            attr,
                            scope,
                            self._column_attr_value(column, attr),
                        )
                        self._master_row_attr_choice_keys.add(attr_key)
                        row_attr_groups[attr_key].add(group.group_id)
                if self._has_area_guidance(
                    column.voyage_id,
                    column.flow,
                    column.big_plan_size,
                ):
                    self._master_area_guidance_keys.add(column.quota_key)
        for key, group_ids in bay_attr_groups.items():
            bay_key, _attr, _scope, _value = key
            relevant_demand = sum(self.group_demand[group_id] for group_id in group_ids)
            self._master_bay_attr_big_m[key] = self._tight_link_bound(
                relevant_demand,
                self.bays[bay_key].physical_capacity,
            )
        for key, group_ids in row_attr_groups.items():
            bay_key, row_no, _attr, _scope, _value = key
            relevant_demand = sum(self.group_demand[group_id] for group_id in group_ids)
            row_capacity = int(
                self.bays[bay_key].row_physical_capacity.get(
                    row_no,
                    self.bays[bay_key].physical_capacity,
                )
            )
            self._master_row_attr_big_m[key] = self._tight_link_bound(
                relevant_demand,
                row_capacity,
            )
        for key, qty in self.quota_by_key.items():
            voyage_id, flow, _area_no, big_size = key
            if qty > 0 and self.voyage_flow_size_demand[(voyage_id, flow, big_size)] > 0:
                self._master_area_guidance_keys.add(key)
        for (_flow, size), candidates in self.import_reservation_candidates.items():
            for bay_key, _capacity in candidates:
                self._master_bay_size_keys.add((bay_key, size))
                self._master_bay_capacity_keys.update(
                    self._placement_footprint_keys(bay_key, size)
                )


    @staticmethod
    def _tight_link_bound(relevant_demand: int, physical_capacity: int) -> int:
        """Return the smallest positive safe link bound from known limits."""
        demand = max(0, int(relevant_demand))
        capacity = max(0, int(physical_capacity))
        bound = min(demand, capacity)
        if bound <= 0:
            raise ValueError(
                "a master activation key has no positive demand-capacity bound: "
                f"demand={demand}, capacity={capacity}"
            )
        return bound

    @staticmethod
    def _bound_summary(values: Iterable[int]) -> dict[str, int | float]:
        bounds = [int(value) for value in values]
        if not bounds:
            return {"count": 0, "minimum": 0, "maximum": 0, "average": 0.0}
        return {
            "count": len(bounds),
            "minimum": min(bounds),
            "maximum": max(bounds),
            "average": round(sum(bounds) / len(bounds), 3),
        }

    def _big_m_diagnostics(self) -> dict[str, object]:
        return {
            "previous_global_bound": int(sum(group.demand for group in self.groups)),
            "method": (
                "min(relevant_group_demand, feasible physical capacity)"
            ),
            "bay_attribute_links": self._bound_summary(self._master_bay_attr_big_m.values()),
            "row_attribute_links": self._bound_summary(self._master_row_attr_big_m.values()),
        }

    def _bay_no_mix_attrs(self, voyage_id: object = None) -> tuple[str, ...]:
        attrs = (
            self.attribute_rules.bay_no_mix_for(voyage_id)
            if voyage_id is not None
            else self.attribute_rules.bay_no_mix_attributes
        )
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
        attrs = (
            self.attribute_rules.row_no_mix_for(voyage_id)
            if voyage_id is not None
            else self.attribute_rules.row_no_mix_attributes
        )
        return tuple(str(attr) for attr in attrs if str(attr))

    @staticmethod
    def _infer_export_voyages(problem: ProblemData) -> set[str]:
        declared = problem.export_voyages
        if declared is not None:
            return {str(voyage_id) for voyage_id in declared if str(voyage_id)}
        return {
            str(group.voyage_id)
            for group in problem.export_groups
            if str(group.status) in EXPORT_FLOWS
        }

    def _is_export_voyage(self, voyage_id: object) -> bool:
        return str(voyage_id) in self.export_voyages

    def _bay_no_mix_attrs_for_group(self, group: ExportGroup) -> tuple[str, ...]:
        return self._bay_no_mix_attrs(group.voyage_id)

    def _row_no_mix_attrs_for_group(self, group: ExportGroup) -> tuple[str, ...]:
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

    def _groups_are_incompatible_on_one_row(
        self, first: ExportGroup, second: ExportGroup
    ) -> bool:
        """Whether declared no-mix rules forbid the pair on one physical row."""
        for attr in set(self._bay_no_mix_attrs_for_group(first)) & set(
            self._bay_no_mix_attrs_for_group(second)
        ):
            if self._attr_voyage_scope(
                attr, first.voyage_id
            ) == self._attr_voyage_scope(attr, second.voyage_id) and (
                self._group_attr_value(first, attr)
                != self._group_attr_value(second, attr)
            ):
                return True
        for attr in set(self._row_no_mix_attrs_for_group(first)) & set(
            self._row_no_mix_attrs_for_group(second)
        ):
            if self._attr_voyage_scope(
                attr, first.voyage_id
            ) == self._attr_voyage_scope(attr, second.voyage_id) and (
                self._group_attr_value(first, attr)
                != self._group_attr_value(second, attr)
            ):
                return True
        return False

    @staticmethod
    def _group_attr_value(group: ExportGroup, attr: str) -> str:
        attrs = group.attributes
        value = attrs.get(attr, "")
        if isinstance(value, bool):
            return "1" if value else "0"
        if value not in (None, ""):
            return str(value)
        text = str(attr).strip()
        upper = text.upper()
        default_value = {
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
            "IYC_EVOY_ID": group.voyage_id,
            "IYC_IVOY_ID": group.voyage_id,
            "VOYAGE_ID": group.voyage_id,
            EXPORT_VOYAGE_ROW_NO_MIX_ATTR.upper(): group.voyage_id,
        }.get(upper, "")
        return str(default_value)

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
        default_value = {
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
            "IYC_EVOY_ID": col.voyage_id,
            "IYC_IVOY_ID": col.voyage_id,
            "VOYAGE_ID": col.voyage_id,
            EXPORT_VOYAGE_ROW_NO_MIX_ATTR.upper(): col.voyage_id,
        }.get(upper, "")
        return str(default_value)

    def _row_mix_key_for_group(self, group: ExportGroup) -> str:
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
            or YardPlanningBase._is_size_no_mix_attr(attr)
        )

    def _attr_voyage_scope(self, attr: str, voyage_id: object) -> str:
        is_height = str(attr).strip().upper() in HEIGHT_NO_MIX_ATTRS
        return "" if self._is_global_row_no_mix_attr(attr) or is_height else str(voyage_id)

    def _bay_state_attr_key(self, bay_key: str, attr: str, voyage_id: object) -> tuple[str, str, str]:
        return (bay_key, attr, self._attr_voyage_scope(attr, voyage_id))

    def _row_state_attr_key(self, bay_key: str, row_no: str, attr: str, voyage_id: object) -> tuple[str, str, str, str]:
        return (bay_key, str(row_no), attr, self._attr_voyage_scope(attr, voyage_id))

    def _existing_bay_attr_values(self, bay: Bay, attr: str, voyage_id: object) -> set[str]:
        if self._is_size_no_mix_attr(attr):
            values = set(bay.existing_attrs.get(attr, set()))
            return values or set(bay.existing_size_modes)
        if str(attr).strip().upper() in HEIGHT_NO_MIX_ATTRS:
            return set(bay.existing_heights)
        by_voyage = bay.existing_attrs_by_voyage
        return set(by_voyage.get(str(voyage_id), {}).get(attr, set()))

    def _existing_row_attr_values(self, bay: Bay, row_no: str, attr: str, voyage_id: object) -> set[str]:
        if self._is_global_row_no_mix_attr(attr):
            row_attrs = bay.existing_attrs_by_row.get(str(row_no), {})
            return set(row_attrs.get(attr, set()))
        by_row_voyage = bay.existing_attrs_by_row_by_voyage
        return set(by_row_voyage.get(str(row_no), {}).get(str(voyage_id), {}).get(attr, set()))

    def _row_existing_attrs_allow_group(self, bay: Bay, row_no: str, group: ExportGroup) -> bool:
        for attr in self._row_no_mix_attrs_for_group(group):
            values = self._existing_row_attr_values(bay, str(row_no), attr, group.voyage_id)
            expected = self._group_attr_value(group, attr)
            if attr == EXPORT_VOYAGE_ROW_NO_MIX_ATTR:
                if values and values != {expected}:
                    return False
            elif values and expected not in values:
                return False
        return True

    def _bay_existing_attrs_allow_group(self, group: ExportGroup, footprint: tuple[str, ...]) -> bool:
        for key in footprint:
            bay = self.bays[key]
            for attr in self._bay_no_mix_attrs_for_group(group):
                values = self._existing_bay_attr_values(bay, attr, group.voyage_id)
                if values and values != {self._group_attr_value(group, attr)}:
                    return False
        return True

    def _bay_state_attrs_allow_group(self, group: ExportGroup, footprint: tuple[str, ...], state: dict) -> bool:
        used_attrs = state.setdefault("bay_used_attrs", {})
        for key in footprint:
            for attr in self._bay_no_mix_attrs_for_group(group):
                state_key = self._bay_state_attr_key(key, attr, group.voyage_id)
                value = self._group_attr_value(group, attr)
                if used_attrs.get(state_key, value) != value:
                    return False
        return True

    def _row_stack_capacities_for_group(self, bay_key: str, size: str, group: ExportGroup) -> list[int]:
        return [cap for _row_no, cap in self._row_stack_capacity_items_for_group(bay_key, size, group)]

    def _row_stack_capacity_items_for_group(self, bay_key: str, size: str, group: ExportGroup) -> list[tuple[str, int]]:
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
        default_capacity = int(bay.cap_by_size.get(size, 0) or bay.physical_capacity)
        return [("__bay__", default_capacity)] if default_capacity > 0 else []

    def _row_capacity_items_for_group(
        self,
        footprint_key: str,
        size: str,
        group: ExportGroup,
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

    def _row_capacity_for_column(
        self,
        group: ExportGroup,
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

    def _stack_count_for_group(self, bay_key: str, size: str, group: ExportGroup) -> int:
        return len(self._row_stack_capacities_for_group(bay_key, size, group))

    def _stack_count_for_bay_size(self, bay_key: str, size: str) -> int:
        bay = self.bays[bay_key]
        row_caps = bay.row_cap_by_size.get(size, {}) or bay.row_physical_capacity
        if row_caps:
            return sum(1 for cap in row_caps.values() if int(cap) > 0)
        return 1 if int(bay.cap_by_size.get(size, 0) or bay.physical_capacity) > 0 else 0

    def _stack_unit_capacity_for_group(self, bay_key: str, size: str, group: ExportGroup) -> int:
        capacities = self._row_stack_capacities_for_group(bay_key, size, group)
        return max(capacities) if capacities else 0

    def _stack_units_for_quantity(self, bay_key: str, size: str, group: ExportGroup, quantity: int) -> int:
        if quantity <= 0:
            return 0
        unit_capacity = self._stack_unit_capacity_for_group(bay_key, size, group)
        if unit_capacity <= 0:
            return 10**9
        return int(math.ceil(quantity / unit_capacity))

    def _remaining_import_reservation_capacity(self, bay_key: str, size: str, state: dict) -> int:
        capacity = self._import_reservation_capacity(bay_key, size)
        footprint = self._placement_footprint_keys(bay_key, size)
        if any(key in state["export_used_bays"] for key in footprint):
            return 0
        if any(
            state["import_used_size"].get(key, size) != size
            for key in footprint
        ):
            return 0
        for key in footprint:
            capacity = min(
                capacity,
                int(self.bays[key].physical_capacity) - int(state["bay_load"][key]),
            )
        capacity = min(
            capacity,
            int(self.bays[bay_key].cap_by_size.get(size, 0))
            - int(state["bay_size_load"][(bay_key, size)]),
        )
        return max(0, int(capacity))

    def _apply_import_reservation_quantity(
        self,
        bay_key: str,
        size: str,
        quantity: int,
        state: dict,
    ) -> None:
        if quantity <= 0:
            return
        footprint = self._placement_footprint_keys(bay_key, size)
        state["area_slot_load"][self.bays[bay_key].area_no] += int(quantity) * len(footprint)
        for key in footprint:
            state["bay_load"][key] += int(quantity)
            state["import_used_bays"].add(key)
            state["import_used_size"][key] = size
        state["bay_size_load"][(bay_key, size)] += int(quantity)

    def _apply_import_reservation_to_state(
        self,
        reservation: Counter[tuple[str, str, str]],
        state: dict,
    ) -> None:
        candidate_keys = {
            (flow, size, bay_key)
            for (flow, size), candidates in self.import_reservation_candidates.items()
            for bay_key, _capacity in candidates
        }
        for (flow, size, bay_key), qty in sorted(reservation.items()):
            if qty <= 0:
                continue
            if (flow, size, bay_key) not in candidate_keys:
                raise RuntimeError(
                    f"invalid import reserve option: flow={flow}, size={size}, bay={bay_key}"
                )
            if int(qty) > self._remaining_import_reservation_capacity(bay_key, size, state):
                raise RuntimeError(
                    f"import reserve exceeds bay capacity: flow={flow}, size={size}, "
                    f"bay={bay_key}, quantity={qty}"
                )
            self._apply_import_reservation_quantity(bay_key, size, int(qty), state)

    def _apply_stack_usage_to_state(self, group: ExportGroup, bay_key: str, quantity: int, state: dict) -> None:
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

    def _selection_state(self, selected: Counter[int]) -> tuple[Counter[int], dict, Counter[str]]:
        repaired: Counter[int] = Counter()
        placed: Counter[str] = Counter()
        state = self._empty_selection_state()
        self._apply_import_reservation_to_state(self._final_import_reservation, state)
        for idx, chosen in sorted(selected.items()):
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            group = self.groups_by_id.get(col.group_id)
            if group is None:
                continue
            for _copy in range(int(round(chosen))):
                remaining = int(group.demand) - int(placed.get(group.group_id, 0))
                if remaining <= 0 or col.quantity > remaining:
                    break
                if not self._column_fits_state(col, state, remaining):
                    break
                self._apply_column_to_state(col, state)
                repaired[idx] += 1
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
            "area_slot_load": Counter(),
            "row_used_attrs": {},
            "row_used_group": {},
            "bay_used_size": {},
            "bay_used_attrs": {},
            "export_used_bays": set(),
            "import_used_bays": set(),
            "import_used_size": {},
            "used_group_area": set(),
            "used_voyage_area": set(),
            "big_plan_quota_used": Counter(),
        }

    def _column_fits_state(
        self,
        col: PlacementColumn,
        state: dict,
        remaining: int,
        enforce_quota: bool = True,
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
            row_owner = state["row_used_group"].get(row_key, col.group_id)
            if row_owner != col.group_id:
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
        group: ExportGroup,
        bay_key: str,
        state: dict,
        remaining: int,
        enforce_quota: bool = True,
    ) -> int:
        bay = self.bays[bay_key]
        footprint = self._placement_footprint_keys(bay_key, group.size)
        if not footprint:
            return 0
        if any(key in state["import_used_bays"] for key in footprint):
            return 0
        if not self._bay_state_attrs_allow_group(group, footprint, state):
            return 0
        capacity = int(remaining)
        for key in footprint:
            capacity = min(capacity, self.bays[key].physical_capacity - state["bay_load"][key])
        capacity = min(capacity, bay.cap_by_size.get(group.size, 0) - state["bay_size_load"][(bay_key, group.size)])
        capacity = min(capacity, self._row_capacity_for_column(group, bay_key, state=state))

        return max(0, int(capacity))

    def _apply_column_to_state(self, col: PlacementColumn, state: dict) -> None:
        footprint = self._placement_footprint_keys(col.bay_key, col.size)
        state["area_slot_load"][col.area_no] += col.quantity * len(footprint)
        for key in footprint:
            state["bay_load"][key] += col.quantity
            state["export_used_bays"].add(key)
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
            state["row_used_group"][(footprint_key, row_no)] = col.group_id
            for attr in self._row_no_mix_attrs_for_column(col):
                state_key = self._row_state_attr_key(footprint_key, row_no, attr, col.voyage_id)
                state["row_used_attrs"][state_key] = self._column_attr_value(col, attr)
        state["used_group_area"].add((col.group_key, col.area_no))
        state["used_voyage_area"].add((col.voyage_id, col.area_no))
        state["big_plan_quota_used"][col.quota_key] += col.quantity

    def _candidate_bays_for_group(self, group: ExportGroup) -> list[tuple[str, int, float]]:
        cached = self._candidate_cache.get(group.group_id)
        if cached is not None:
            return cached
        out: list[tuple[str, int, float]] = []
        for area_no in self._candidate_areas_for_group(group):
            for bay_key in self.bays_by_area.get(area_no, []):
                max_qty = self._max_quantity_in_bay(group, bay_key)
                if max_qty <= 0:
                    continue
                cost = self._column_base_cost(group, bay_key)
                out.append((bay_key, min(max_qty, group.demand), cost))
        out.sort(
            key=lambda item: (
                0 if self._is_big_plan_area_for_group(group, self.bays[item[0]].area_no) else 1,
                self._existing_group_bay_rank(group, item[0]),
                item[2],
                -item[1],
                self.bays[item[0]].area_no,
                self.bays[item[0]].bay_order,
            )
        )
        self._candidate_cache[group.group_id] = out
        return out

    def _candidate_areas_for_group(self, group: ExportGroup) -> list[str]:
        return sorted(
            [
                area_no
                for area_no in self.bays_by_area
                if self._area_supports_group_flow(group, area_no)
            ],
            key=lambda area_no: (
                0 if self._is_big_plan_area_for_group(group, area_no) else 1,
                area_no,
            ),
        )

    def _area_supports_group_flow(self, group: ExportGroup, area_no: str) -> bool:
        functions = self.problem.area_functions.get(area_no, set())
        return _area_flow(group.status) in functions

    def _max_quantity_in_bay(self, group: ExportGroup, bay_key: str) -> int:
        bay = self.bays[bay_key]
        if bay.cap_by_size.get(group.size, 0) <= 0:
            return 0
        footprint = self._placement_footprint_keys(bay_key, group.size)
        if not footprint:
            return 0
        if not self._bay_existing_attrs_allow_group(group, footprint):
            return 0
        is_edge = bay_key in self.area_edge_bays.get(bay.area_no, set())
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

    def _column_base_cost(self, group: ExportGroup, bay_key: str) -> float:
        cost = (
            self.config.existing_group_proximity_weight
            * self._normalized_existing_proximity(group, bay_key)
            / self._objective_scale("existing_group_proximity")
        )
        # Import capacity is represented by joint master constraints and has
        # no pair-loss or other per-column objective cost.
        return cost

    def _is_big_plan_area_for_group(self, group: ExportGroup, area_no: str) -> bool:
        big_size = self._big_plan_size(group.size)
        return any(
            voyage_id == group.voyage_id
            and flow == group.status
            and candidate_area == area_no
            and size == big_size
            and qty > 0
            for (voyage_id, flow, candidate_area, size), qty in self.quota_by_key.items()
        )

    def _prepare_yard_indexes(self) -> None:
        for key, bay in self.bays.items():
            self.bays_by_area[bay.area_no].append(key)
        for keys in self.bays_by_area.values():
            keys.sort(key=lambda bay_key: self.bays[bay_key].bay_order)
        for area_no, keys in self.bays_by_area.items():
            if keys:
                boundary_keys = {keys[0], keys[-1]}
                self.area_edge_bays[area_no] = {
                    bay_key
                    for bay_key in keys
                    if self.bays[bay_key].large_bay_partner_key
                    and (
                        bay_key in boundary_keys
                        or self.bays[bay_key].large_bay_partner_key in boundary_keys
                    )
                }

    def _prepare_import_reservation_candidates(self) -> None:
        """Build anonymous import reserve options using only flow and size."""
        self.import_reservation_candidates.clear()
        for flow, size in sorted(self.import_total_by_flow_size):
            candidates: list[tuple[str, int]] = []
            for area_no in sorted(self.bays_by_area):
                if flow not in self.problem.area_functions.get(area_no, set()):
                    continue
                for bay_key in self.bays_by_area[area_no]:
                    capacity = self._import_reservation_capacity(bay_key, size)
                    if capacity > 0:
                        candidates.append((bay_key, capacity))
            self.import_reservation_candidates[(flow, size)] = candidates

    def _import_reservation_capacity(self, bay_key: str, size: str) -> int:
        """Return anonymous import capacity under the V6 bay-size contract."""
        bay = self.bays.get(bay_key)
        if bay is None or size not in {"20", "40"}:
            return 0
        footprint = self._placement_footprint_keys(bay_key, size)
        if not footprint:
            return 0
        # Anonymous imports still activate a physical size state on every bay
        # in their footprint.  They may only enter a residual bay whose
        # existing size state is empty or already matches that size.
        for key in footprint:
            existing_modes = {
                str(mode) for mode in self.bays[key].existing_size_modes if str(mode)
            }
            if existing_modes and existing_modes != {str(size)}:
                return 0
        return max(
            0,
            min(
                int(bay.cap_by_size.get(size, 0) or 0),
                *(int(self.bays[key].physical_capacity) for key in footprint),
            ),
        )

    def _prepare_quota(self) -> None:
        for (voyage_id, flow, area_no, big_size), qty in self.problem.area_guidance_target.items():
            if qty > 0:
                self.quota_by_key[(voyage_id, flow, area_no, big_size)] += int(qty)

    def _prepare_reachable_anchor_groups(self) -> None:
        """Keep proximity anchors only when their area remains feasible.

        An incumbent anchor in a closed, function-incompatible, or completely
        full area cannot guide the current decision. Charging every candidate
        the same maximum distance would add a constant to the objective without
        changing the allocation.
        """
        self.reachable_anchor_group_ids.clear()
        for group in self.groups:
            anchor_key = self._existing_anchor_key(group)
            anchor_areas = {
                str(key[-2])
                for key, quantity in self.existing_group_bay_load.items()
                if int(quantity) > 0 and tuple(key[:-2]) == anchor_key
            }
            if not anchor_areas:
                continue
            for area_no in anchor_areas:
                if not self._area_supports_group_flow(group, area_no):
                    continue
                if any(
                    self._max_quantity_in_bay(group, bay_key) > 0
                    for bay_key in self.bays_by_area.get(area_no, ())
                ):
                    self.reachable_anchor_group_ids.add(group.group_id)
                    break

    def _quota_key(self, group: ExportGroup, area_no: str) -> tuple[str, str, str, str]:
        return group.voyage_id, group.status, area_no, self._big_plan_size(group.size)

    def _operational_group_key(self, group: ExportGroup) -> tuple[str, ...]:
        return self._attribute_cluster_key(
            group,
            EXPORT_GROUP_IDENTITY_ATTRIBUTES,
        )

    def _attribute_cluster_key(self, group: ExportGroup, attrs: tuple[str, ...]) -> tuple[str, ...]:
        scope = str(group.voyage_id) if self._is_export_voyage(group.voyage_id) else "IMPORT"
        return (
            scope,
            f"flow={group.status}",
            *(f"{attr}={self._group_attr_value(group, attr)}" for attr in attrs),
        )

    def _existing_anchor_key(self, group: ExportGroup) -> tuple[str, ...]:
        return self._operational_group_key(group)

    def _existing_group_bay_load_for_group(self, group: ExportGroup, bay_key: str) -> int:
        bay = self.bays.get(bay_key)
        if bay is None:
            return 0
        return int(self.existing_group_bay_load.get(self._existing_anchor_key(group) + (bay.area_no, bay_key), 0))

    def _existing_same_group_bay_distance(self, group: ExportGroup, bay_key: str) -> int | None:
        bay = self.bays.get(bay_key)
        if bay is None:
            return None
        anchor_bays = self.existing_group_area_bays.get(self._existing_anchor_key(group) + (bay.area_no,), set())
        if not anchor_bays:
            return None
        distances = [abs(self.bays[key].bay_order - bay.bay_order) for key in anchor_bays if key in self.bays]
        return min(distances) if distances else None

    def _normalized_existing_proximity(self, group: ExportGroup, bay_key: str) -> float:
        """Return a [0, 1] cost relative to incumbent exact-group anchors.

        Reusing an incumbent bay costs zero. Within an anchored area, bay
        distance is divided by that area's full bay-order span. Selecting an
        area without an anchor costs one. Groups without any incumbent anchor
        are neutral and therefore contribute zero.
        """
        if group.group_id not in self.reachable_anchor_group_ids:
            return 0.0
        bay = self.bays.get(bay_key)
        if bay is None:
            return 0.0
        group_key = self._existing_anchor_key(group)
        if not self.existing_group_bays.get(group_key):
            return 0.0
        distance = self._existing_same_group_bay_distance(group, bay_key)
        if distance is None:
            return 1.0
        orders = [self.bays[key].bay_order for key in self.bays_by_area.get(bay.area_no, ())]
        span = max(orders, default=0) - min(orders, default=0)
        if span <= 0:
            return 0.0
        return min(1.0, max(0.0, float(distance) / float(span)))

    def _existing_group_bay_rank(self, group: ExportGroup, bay_key: str) -> tuple[int, int, int]:
        bay_load = self._existing_group_bay_load_for_group(group, bay_key)
        if bay_load > 0:
            return (0, 0, -bay_load)
        distance = self._existing_same_group_bay_distance(group, bay_key)
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

    def _area_size_target(self, voyage_id: str, flow: str, area_no: str, big_size: str) -> float:
        total = sum(qty for (v, f, _a, s), qty in self.quota_by_key.items() if v == voyage_id and f == flow and s == big_size)
        demand = self.voyage_flow_size_demand[(voyage_id, flow, big_size)]
        if total <= 0:
            return 0.0
        return demand * self.quota_by_key.get((voyage_id, flow, area_no, big_size), 0) / total

    def _has_area_guidance(self, voyage_id: str, flow: str, big_size: str) -> bool:
        return any(
            v == voyage_id and f == flow and size == big_size and qty > 0
            for (v, f, _area, size), qty in self.quota_by_key.items()
        )

    def _objective_weights(self) -> dict[str, float]:
        return {
            "area_dispersion": float(self.config.area_dispersion_weight),
            "row_dispersion": float(self.config.row_dispersion_weight),
            "existing_group_proximity": float(self.config.existing_group_proximity_weight),
            "area_guidance": float(self.config.area_guidance_weight),
            "berth_distance": float(self.config.berth_distance_weight),
        }

    def _validate_objective_weights(self) -> None:
        weights = self._objective_weights()
        if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError(f"business objective weights must be finite and nonnegative: {weights}")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"business objective weights must sum to 1, got {total}: {weights}")

    def _anchored_group_demand(self) -> int:
        return sum(
            int(group.demand)
            for group in self.groups
            if group.group_id in self.reachable_anchor_group_ids
        )

    def _guided_demand(self) -> int:
        return sum(
            int(qty)
            for (voyage_id, flow, big_size), qty in self.voyage_flow_size_demand.items()
            if self._has_area_guidance(voyage_id, flow, big_size)
        )

    def _objective_scale(self, key: str) -> float:
        if key in self._objective_scales:
            return max(1.0, float(self._objective_scales[key]))
        default_scale = {
            "existing_group_proximity": self._anchored_group_demand(),
            "area_guidance_l1": 2
            * (self._guided_demand() + sum(self.import_total_by_flow_size.values())),
            "berth_distance": sum(group.demand for group in self.groups),
        }.get(key, 1.0)

        return max(1.0, float(default_scale))

    def _prepare_objective_normalization(self) -> None:
        demand_by_group: Counter[tuple[str, ...]] = Counter()
        areas_by_group: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
        rows_by_group: defaultdict[tuple[str, ...], set[tuple[str, str]]] = defaultdict(set)
        for group in self.groups:
            group_key = self._operational_group_key(group)
            demand_by_group[group_key] += int(group.demand)
            for column in self._base_placements_for_group(group):
                areas_by_group[group_key].add(column.area_no)
                rows_by_group[group_key].update(
                    (bay_key, row_no)
                    for bay_key, row_no, _quantity in column.row_allocation
                )
        area_scale = sum(
            max(0, min(int(demand), len(areas_by_group[key])) - 1)
            for key, demand in demand_by_group.items()
        )
        row_scale = sum(
            max(0, min(int(demand), len(rows_by_group[key])) - 1)
            for key, demand in demand_by_group.items()
        )
        self._objective_scales = {
            "area_dispersion": float(max(1, area_scale)),
            "row_dispersion": float(max(1, row_scale)),
            "existing_group_proximity": float(max(1, self._anchored_group_demand())),
            "area_guidance_l1": float(
                max(
                    1,
                    2
                    * (
                        self._guided_demand()
                        + sum(self.import_total_by_flow_size.values())
                    ),
                )
            ),
            "berth_distance": float(max(1, sum(group.demand for group in self.groups))),

        }

    def _prepare_berth_distance_bounds(self) -> None:
        areas_by_voyage: defaultdict[str, set[str]] = defaultdict(set)
        for group in self.groups:
            for bay_key, _capacity, _cost in self._candidate_bays_for_group(group):
                areas_by_voyage[group.voyage_id].add(self.bays[bay_key].area_no)
        for voyage_id, areas in areas_by_voyage.items():
            berth = self.problem.berth_by_voyage.get(voyage_id, "")
            if not berth:
                raise ValueError(f"missing berth mapping for detailed export voyage {voyage_id}")
            distances: list[float] = []
            for area_no in sorted(areas):
                distance = self.problem.berth_distances.get((area_no, berth))
                if distance is None:
                    raise ValueError(
                        "missing berth-area distance for detailed export allocation: "
                        f"voyage={voyage_id}, berth={berth}, area={area_no}"
                    )
                if not math.isfinite(float(distance)) or float(distance) <= 0:
                    raise ValueError(
                        "berth-area distance must be a positive finite value: "
                        f"voyage={voyage_id}, berth={berth}, area={area_no}, distance={distance!r}"
                    )
                distances.append(float(distance))
            if distances:
                self._berth_distance_bounds[voyage_id] = (min(distances), max(distances))

    def _normalized_berth_distance(self, voyage_id: str, area_no: str) -> float:
        berth = self.problem.berth_by_voyage.get(voyage_id, "")
        distance = self.problem.berth_distances.get((area_no, berth))
        if distance is None or voyage_id not in self._berth_distance_bounds:
            raise ValueError(
                "missing normalized berth-area distance data: "
                f"voyage={voyage_id}, berth={berth}, area={area_no}"
            )
        lower, upper = self._berth_distance_bounds[voyage_id]
        if upper <= lower:
            return 0.0
        return min(1.0, max(0.0, (float(distance) - lower) / (upper - lower)))

    def _area_guidance_penalty(self) -> float:
        return float(self.config.area_guidance_weight) / self._objective_scale("area_guidance_l1")

    def _area_activation_penalty(self) -> float:
        return float(self.config.area_dispersion_weight) / self._objective_scale("area_dispersion")

    def _row_activation_penalty(self) -> float:
        return float(self.config.row_dispersion_weight) / self._objective_scale("row_dispersion")

    def _berth_distance_cost(self, voyage_id: str, area_no: str, quantity: int) -> float:
        """Quantity-weighted berth-to-yard travel cost."""
        berth = self.problem.berth_by_voyage.get(voyage_id, "")
        if not berth:
            raise ValueError(
                f"missing berth mapping for detailed export voyage {voyage_id}"
            )
        distance = self.problem.berth_distances.get((area_no, berth))
        if distance is None:
            raise ValueError(
                "missing berth-area distance for detailed export allocation: "
                f"voyage={voyage_id}, berth={berth}, area={area_no}"
            )
        if not math.isfinite(float(distance)) or float(distance) <= 0:
            raise ValueError(
                "berth-area distance must be a positive finite value: "
                f"voyage={voyage_id}, berth={berth}, area={area_no}, distance={distance!r}"
            )
        return (
            float(self.config.berth_distance_weight)
            * self._normalized_berth_distance(voyage_id, area_no)
            * max(0, int(quantity))
            / self._objective_scale("berth_distance")
        )

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

    def _area_summary_big_plan_inheritance_stats(self, bay_summary_rows: list[dict]) -> dict[str, float | int]:
        actual = self._area_summary_size_counter(bay_summary_rows)
        total = sum(actual.values())
        inherited = sum(min(qty, int(self.quota_by_key.get(key, 0) or 0)) for key, qty in actual.items())
        transferred = max(0, total - inherited)
        return {
            "total_boxes": total,
            "inherited_boxes": inherited,
            "transferred_boxes": transferred,
            "inheritance_ratio": round(inherited / total, 6) if total else 1.0,
            "transfer_ratio": round(transferred / total, 6) if total else 0.0,
        }

    def _area_summary_inheritance_energy_components(self, bay_summary_rows: list[dict]) -> dict[str, float]:
        actual = self._area_summary_size_counter(bay_summary_rows)
        components = {
            "area_guidance_transfer": 0.0,
        }
        targets = self._effective_big_plan_area_size_targets()
        evaluated_keys = set(targets) | {
            key
            for key in actual
            if self._has_area_guidance(key[0], key[1], key[3])
        }
        for key in evaluated_keys:
            components["area_guidance_transfer"] += self._area_guidance_penalty() * abs(
                actual.get(key, 0) - targets.get(key, 0.0)
            )
        components["total"] = sum(components.values())
        return {key: round(value, 4) for key, value in components.items()}

    def _area_summary_size_counter(self, bay_summary_rows: list[dict]) -> Counter[tuple[str, str, str, str]]:
        actual: Counter[tuple[str, str, str, str]] = Counter()
        for row in bay_summary_rows:
            qty = int(row.get("planned_boxes", 0) or 0)
            if qty <= 0:
                continue
            voyage_id = str(row.get("voyage_id", ""))
            flow = str(row.get("flow", "OF") or "OF")
            area_no = str(row.get("area_no", ""))
            big_size = self._big_plan_size(str(row.get("size", "")))
            actual[(voyage_id, flow, area_no, big_size)] += qty
        return actual

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

    def _make_export_rows(self, selected: Counter[int]) -> list[dict]:
        counter: Counter[tuple] = Counter()
        for idx, chosen in selected.items():
            if chosen <= 0:
                continue
            col = self._columns[idx]
            qty_by_row: Counter[str] = Counter()
            for _footprint_key, row_no, qty in col.row_allocation:
                qty_by_row[str(row_no)] = max(qty_by_row[str(row_no)], int(qty))
            for row_no, row_qty in qty_by_row.items():
                row_allocation = self._format_row_allocation(
                    tuple(
                        (bay_key, candidate_row, 1)
                        for bay_key, candidate_row, qty in col.row_allocation
                        if str(candidate_row) == row_no and int(qty) > 0
                    )
                )
                dynamic_attrs = tuple(sorted((str(k), str(v)) for k, v in (col.attributes or {}).items()))
                key = (
                    col.voyage_id, col.group_id, col.flow, col.port, col.size, col.height,
                    col.area_no, col.bay_key, col.bay_no, row_no, row_allocation, dynamic_attrs,
                )
                counter[key] += row_qty * chosen

        rows: list[dict] = []
        for key, qty in sorted(counter.items()):
            (
                voyage_id, group_id, flow, port, size, height, area_no,
                bay_key, bay_no, row_no, row_allocation, dynamic_attrs,
            ) = key
            row = {
                "voyage_id": voyage_id,
                "group_id": group_id,
                "flow": flow,
                "port": port,
                "size": size,
                "height": height,
                "area_no": area_no,
                "bay_key": bay_key,
                "bay_no": bay_no,
                "row_no": row_no,
                "row_allocation": row_allocation,
                "planned_boxes": qty,
            }
            for attr, value in dynamic_attrs:
                if attr and attr not in row:
                    row[attr] = value
            rows.append(row)
        return rows

    @staticmethod
    def _format_row_allocation(row_allocation: tuple[tuple[str, str, int], ...]) -> str:
        return "|".join(
            f"{bay_key}:{row_no}:{int(qty)}"
            for bay_key, row_no, qty in row_allocation
            if int(qty) > 0
        )

    def _make_bay_summary_rows(self, selected: Counter[int]) -> list[dict]:
        counter: Counter[tuple] = Counter()
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            dynamic_attrs = tuple(sorted((str(k), str(v)) for k, v in (col.attributes or {}).items()))
            key = (
                col.voyage_id, col.flow, col.port, col.size,
                col.area_no, col.bay_key, col.bay_no, dynamic_attrs,
            )
            counter[key] += col.quantity * int(chosen)

        rows: list[dict] = []
        for key, qty in sorted(counter.items()):
            voyage_id, flow, port, size, area_no, bay_key, bay_no, dynamic_attrs = key
            row = {
                "voyage_id": voyage_id,
                "flow": flow,
                "port": port,
                "size": size,
                "area_no": area_no,
                "bay_key": bay_key,
                "bay_no": bay_no,
                "planned_boxes": qty,
            }
            for attr, value in dynamic_attrs:
                if attr and attr not in row:
                    row[attr] = value
            rows.append(row)
        return rows

    def _group_sort_key(
        self, group: ExportGroup
    ) -> tuple[int, int, str, str, str]:
        return (
            SIZE_ORDER.get(group.size, 3),
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


def write_selected_locations(
    path: str | Path,
    locations: Iterable[PlacementColumn],
) -> None:
    rows = []
    for col in locations:
        row = {
            "location_id": col.column_id,
            "group_id": col.group_id,
            "voyage_id": col.voyage_id,
            "flow": col.flow,
            "port": col.port,
            "size": col.size,
            "height": col.height,
            "area_no": col.area_no,
            "bay_no": col.bay_no,
            "row_allocation": YardPlanningBase._format_row_allocation(
                col.row_allocation
            ),
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
