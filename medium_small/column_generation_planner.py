from __future__ import annotations

import csv
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
MANDATORY_BAY_NO_MIX_ATTRS = ("IYC_CSZ_CSIZECD", "IYC_CHEIGHTCD")
SIZE_NO_MIX_ATTRS = frozenset({"IYC_CSZ_CSIZECD", "SIZE", "SIZE_MODE"})
HEIGHT_NO_MIX_ATTRS = frozenset({"IYC_CHEIGHTCD", "HEIGHT"})


class _GurobiModelAdapter:
    """Small compatibility layer around :class:`gurobipy.Model`.

    Keeping model construction behind this adapter lets the mathematical
    constraints remain unchanged while the optimization backend is Gurobi.
    Variables, expressions, constraints, LP duals, and bounds are all native
    Gurobi objects.
    """

    def __init__(self, name: str) -> None:
        import gurobipy as gp

        self._gp = gp
        self._model = gp.Model(name)

    def addVar(self, **kwargs):
        return self._model.addVar(**kwargs)

    def addCons(self, expression, name: str | None = None):
        return self._model.addConstr(expression, name=name or "")

    def getVars(self):
        return self._model.getVars()

    def setMinimize(self) -> None:
        self._model.ModelSense = self._gp.GRB.MINIMIZE

    def setParam(self, name: str, value: object) -> None:
        self._model.setParam(name, value)

    def hideOutput(self) -> None:
        self._model.Params.OutputFlag = 0

    def optimize(self) -> None:
        self._model.optimize()

    def getStatus(self) -> str:
        status_names = {
            self._gp.GRB.LOADED: "loaded",
            self._gp.GRB.OPTIMAL: "optimal",
            self._gp.GRB.INFEASIBLE: "infeasible",
            self._gp.GRB.INF_OR_UNBD: "inforunbd",
            self._gp.GRB.UNBOUNDED: "unbounded",
            self._gp.GRB.CUTOFF: "cutoff",
            self._gp.GRB.ITERATION_LIMIT: "iterationlimit",
            self._gp.GRB.NODE_LIMIT: "nodelimit",
            self._gp.GRB.TIME_LIMIT: "timelimit",
            self._gp.GRB.SOLUTION_LIMIT: "solutionlimit",
            self._gp.GRB.INTERRUPTED: "interrupted",
            self._gp.GRB.NUMERIC: "numeric",
            self._gp.GRB.SUBOPTIMAL: "suboptimal",
        }
        return status_names.get(self._model.Status, str(self._model.Status))

    def getNSols(self) -> int:
        return int(self._model.SolCount)

    def getBestSol(self):
        return object() if self._model.SolCount > 0 else None

    def getObjVal(self) -> float:
        return float(self._model.ObjVal)

    def getGap(self) -> float:
        return float(self._model.MIPGap) if self._model.IsMIP else 0.0

    def getPrimalbound(self) -> float:
        return float(self._model.ObjVal)

    def getDualbound(self) -> float:
        return float(self._model.ObjBound)

    @staticmethod
    def getVal(var) -> float:
        return float(var.X)


    @staticmethod
    def getDualsolLinear(constr) -> float:
        return float(constr.Pi)

    def freeProb(self) -> None:
        self._model.dispose()


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
    group_key: tuple[str, ...]
    intrinsic_cost: float


@dataclass
class ColumnGenerationConfig:
    max_iterations: int = 30
    initial_columns_per_group: int = 8
    total_time_limit: float = 240.0
    mip_time_limit: float = 120.0
    mip_gap: float = 0.01
    verbose: bool = True
    use_gurobi: bool = True
    allow_greedy_fallback: bool = False
    pricing_unplaced_penalty: float = 1_000_000.0
    # Stage-2 policy weights. Every component is first mapped to a natural
    # dimensionless scale, so these values express policy preference only.
    area_dispersion_weight: float = 0.20
    row_dispersion_weight: float = 0.17
    existing_group_proximity_weight: float = 0.13
    area_guidance_weight: float = 0.22
    large_pair_capacity_weight: float = 0.17
    berth_distance_weight: float = 0.11


@dataclass
class ColumnGenerationResult:
    area_bay_rows: list[dict]
    small_rows: list[dict]
    diagnostics: dict
    unplaced_rows: list[dict] = field(default_factory=list)
    columns: list[PlacementColumn] = field(default_factory=list)


class ColumnGenerationPlanner:
    """Declared-export row assignment with exact unit-flow pricing.

    A column is one feasible group/bay/row unit flow.  Its integer master
    variable gives the number of identical containers assigned there.  The master uses
    only declared, not-yet-gated-in export containers; import containers enter
    as existing occupancy and big-plan area reservations, never as assignment
    demand.  Big-plan quantities guide area allocation but are not a second
    executable planning layer.
    """

    def __init__(self, problem: ProblemData, config: ColumnGenerationConfig | None = None) -> None:
        self.problem = problem
        self.config = config or ColumnGenerationConfig()
        self.group_source: dict[str, str] = {}
        self.demand_stats: dict[str, int | str] = {}
        self.export_voyages = self._infer_export_voyages(problem)
        self.import_voyages = self._infer_import_voyages(problem)
        self.groups = sorted(self._build_planning_groups(), key=self._group_sort_key)
        self.groups_by_id = {group.group_id: group for group in self.groups}
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
        self.import_area_size_reservation: Counter[tuple[str, str]] = Counter(
            getattr(problem, "import_area_size_reservation", {}) or {}
        )
        self.existing_group_area_load: Counter[tuple[str, ...]] = Counter(
            {
                tuple(key): int(value)
                for key, value in getattr(problem, "existing_group_area_load", {}).items()
                if int(value) > 0
            }
        )
        self.existing_group_bay_load: Counter[tuple[str, ...]] = Counter(
            {
                tuple(key): int(value)
                for key, value in getattr(problem, "existing_group_bay_load", {}).items()
                if int(value) > 0
            }
        )
        self.existing_group_bays: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
        self.existing_group_area_bays: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
        self.large_segment_by_bay: dict[str, tuple[str, ...]] = {}
        self.large_segment_base_pairs: dict[tuple[str, ...], int] = {}
        self.large_segment_static_loss_by_bay: dict[str, int] = {}
        self.large_pair_capacity: dict[tuple[str, str], int] = {}
        self.large_pair_capacity_45: dict[tuple[str, str], int] = {}
        self.large_pair_by_member: dict[str, tuple[str, str]] = {}
        self.existing_twenty_bays: set[str] = set()
        for (*group_key, area_no, bay_key), value in self.existing_group_bay_load.items():
            if value > 0:
                group_tuple = tuple(group_key)
                self.existing_group_bays[group_tuple].add(str(bay_key))
                self.existing_group_area_bays[group_tuple + (str(area_no),)].add(str(bay_key))
        self.group_demand = {group.group_id: int(group.demand) for group in self.groups}
        self.voyage_flow_size_demand: Counter[tuple[str, str, str]] = Counter()
        self._columns: list[PlacementColumn] = []
        self._active_column_indices: set[int] = set()
        self._column_keys: set[tuple[str, str, int, tuple[tuple[str, str, int], ...]]] = set()
        self._candidate_cache: dict[tuple[str, str], list[tuple[str, int, float]]] = {}
        self._candidate_scope = "all"
        self._objective_scales: dict[str, float] = {}
        self._berth_distance_bounds: dict[str, tuple[float, float]] = {}
        self._master_seed_selected: Counter[int] = Counter()
        self._master_seed_unplaced: Counter[str] = Counter()
        self._master_start_selected: Counter[int] = Counter()
        self._master_start_unplaced: Counter[str] = Counter()
        self._prepare_yard_indexes()
        self._prepare_quota()
        self._validate_objective_weights()
        self._prepare_berth_distance_bounds()
        for group in self.groups:
            self.voyage_flow_size_demand[(group.voyage_id, group.status, self._big_plan_size(group.size))] += group.demand

    @property
    def columns(self) -> list[PlacementColumn]:
        return self._columns



    def _build_planning_groups(self) -> list[SmallBoxGroup]:
        """Return declared, not-yet-arrived export groups only."""
        groups = [
            group
            for group in (getattr(self.problem, "small_groups", []) or getattr(self.problem, "groups", []) or [])
            if str(group.status) in EXPORT_FLOWS and int(group.demand) > 0
        ]
        for group in groups:
            self.group_source[group.group_id] = "document"
        boxes = sum(int(group.demand) for group in groups)
        self.demand_stats = {
            "declared_export_group_count": len(groups),
            "declared_export_box_count": boxes,
            "planning_group_count": len(groups),
            "planning_box_count": boxes,
            "demand_policy": "declared_export_only",
        }
        return groups
    def solve(self) -> ColumnGenerationResult:
        self._build_unit_flow_column_universe()
        self._prepare_objective_normalization()
        base_initial_column_count = len(self._active_column_indices)
        self._master_seed_selected = Counter()
        self._master_seed_unplaced = Counter(
            {group.group_id: int(group.demand) for group in self.groups}
        )
        seed_stats = {
            "seed_method": "all_unplaced_feasible_start",
            "seed_selected_columns": 0,
            "seed_unplaced_boxes": int(sum(self._master_seed_unplaced.values())),
        }
        initially_inactive_unit_flows = len(self._columns) - base_initial_column_count
        diagnostics: dict = {
            "algorithm": "export_row_column_generation",
            "model_scope": "export_declared_containers_row_allocation",
            "detailed_allocation_direction": "export_only",
            "target_voyages": self.problem.target_voyages,
            "attribute_rules": self.attribute_rules.as_dict() if hasattr(self.attribute_rules, "as_dict") else {},
            "declared_export_group_count": int(self.demand_stats.get("declared_export_group_count", 0)),
            "declared_export_box_count": int(self.demand_stats.get("declared_export_box_count", 0)),
            "planned_group_count": len(self.groups),
            "planned_box_count": sum(group.demand for group in self.groups),
            "berth_distance_count": len(self.problem.berth_distances),
            "berth_by_voyage": self.problem.berth_by_voyage,
            "demand_alignment": self.demand_stats,
            "area_guidance": {
                "source": "normalized_export_big_plan_new_qty",
                "role": "soft_spatial_reference_only",
                "target_boxes": int(sum(self.quota_by_key.values())),
                "unguided_boxes": int(
                    sum(
                        group.demand
                        for group in self.groups
                        if not any(
                            key[0] == group.voyage_id
                            and key[1] == group.status
                            and key[3] == self._big_plan_size(group.size)
                            and qty > 0
                            for key, qty in self.quota_by_key.items()
                        )
                    )
                ),
            },
            "aggregate_capacity_reservations": {
                "source_quantity_field": "new_qty",
                "import_boxes": int(sum(self.import_area_size_reservation.values())),
                "export_boxes": 0,
                "export_policy": "no_forecast_reservation",
                "accepted_big_plan_sizes": ["20", "40"],
                "import_by_area_size": {
                    f"{area}|{size}": int(qty)
                    for (area, size), qty in sorted(self.import_area_size_reservation.items())
                },
            },
            "import_large_pair_reservation": {
                "required_by_area": {
                    area_no: int(
                        sum(
                            qty
                            for (reserved_area, size), qty in self.import_area_size_reservation.items()
                            if reserved_area == area_no and size in {"40", "45"}
                        )
                    )
                    for area_no in sorted({area for area, _size in self.import_area_size_reservation})
                },
                "base_pair_capacity_by_area": {
                    area_no: int(
                        sum(
                            capacity
                            for pair, capacity in self.large_pair_capacity.items()
                            if self.bays[pair[0]].area_no == area_no
                        )
                    )
                    for area_no in sorted({self.bays[pair[0]].area_no for pair in self.large_pair_capacity})
                },
            },
            "initial_column_count": len(self._active_column_indices),
            "base_initial_column_count": base_initial_column_count,
            "exact_unit_flow_pricing": True,
            "restricted_master_is_active_only": True,
            "initially_inactive_unit_flow_count": initially_inactive_unit_flows,
            **seed_stats,
            "pricing_iterations": [],
            "gurobi_available": False,
            "used_greedy_fallback": False,
            "secondary_objective_normalization": {
                "weights": self._objective_weights(),
                "scales": dict(self._objective_scales),
                "weight_sum": round(sum(self._objective_weights().values()), 10),
                "method": "natural_instance_scale",
            },
            "existing_operational_group_anchors": {
                "mode": "exact_group_bay_proximity",
                "area_key_count": len(self.existing_group_area_load),
                "bay_key_count": len(self.existing_group_bay_load),
                "box_count": int(sum(self.existing_group_bay_load.values())),
            },
            "objective_coefficients": {
                "pricing_unplaced_penalty": self.config.pricing_unplaced_penalty,
                "area_guidance_l1_unit": self._area_guidance_penalty(),
                "area_activation_unit": self._area_activation_penalty(),
                "row_activation_unit": self._row_activation_penalty(),
                "large_pair_capacity_unit": self._twenty_segment_loss_penalty(),
            },
        }

        selected: Counter[int]
        unplaced: Counter[str]
        if not self.config.use_gurobi:
            diagnostics["used_greedy_fallback"] = True
            diagnostics["fallback_reason"] = "explicitly_disabled_gurobi"
            selected, unplaced = self._greedy_fallback()
        else:
            try:
                selected, unplaced, master_stats = self._solve_by_column_generation()
                diagnostics.update(master_stats)
            except Exception as exc:
                if not self.config.allow_greedy_fallback:
                    raise
                diagnostics["used_greedy_fallback"] = True
                diagnostics["gurobi_failure"] = f"{type(exc).__name__}: {exc}"
                selected, unplaced = self._greedy_fallback()

        objective_components = self._selected_objective_components(selected, unplaced)
        diagnostics["final_secondary_objective"] = objective_components["weighted_total"]
        diagnostics["final_secondary_objective_components"] = objective_components
        diagnostics["independent_solution_validation"] = self._validate_final_solution(
            selected, unplaced
        )

        if self._uses_original_output_scope():
            small_rows = self._make_small_rows(selected, allowed_sources={"document"})
        else:
            small_rows = self._make_small_rows(selected)
        area_bay_rows = self._make_area_bay_rows_from_selected_columns(selected, plan_level="medium")
        unplaced_rows = self._unplaced_group_details(unplaced)
        consistency_stats = self._row_area_summary_consistency_stats(small_rows, area_bay_rows)
        bay_consistency_stats = self._row_bay_summary_consistency_stats(small_rows, area_bay_rows)
        operational_group_dispersion = self._operational_group_dispersion_stats(small_rows)
        diagnostics.update(
            {
                "final_column_count": len(self._columns),
                "selected_column_count": sum(1 for qty in selected.values() if qty > 0),
                "summary_granularity": "bay",
                "export_row_count": len(small_rows),
                "area_bay_summary_row_count": len(area_bay_rows),
                "planned_export_boxes": sum(int(row["planned_boxes"]) for row in small_rows),
                "planned_area_summary_by_source": self._planned_area_summary_by_source(area_bay_rows),
                "operational_group_dispersion": operational_group_dispersion,
                "area_summary_big_plan_inheritance": self._area_summary_big_plan_inheritance_stats(area_bay_rows),
                "final_area_summary_inheritance_energy_components": self._area_summary_inheritance_energy_components(area_bay_rows),
                "capacity_reservation_margins": self._capacity_reservation_margins(selected),
                "export_voyage_row_no_mix": self._export_voyage_row_no_mix_stats(selected),
                "unplaced_boxes": sum(unplaced.values()),
                "unplaced_by_group": {key: qty for key, qty in sorted(unplaced.items()) if qty > 0},
                "unplaced_group_details": unplaced_rows,
                **consistency_stats,
                **bay_consistency_stats,
            }
        )
        return ColumnGenerationResult(
            area_bay_rows=area_bay_rows,
            small_rows=small_rows,
            diagnostics=diagnostics,
            unplaced_rows=unplaced_rows,
            columns=self._columns,
        )

    def _validate_final_solution(
        self,
        selected: Counter[int],
        unplaced: Counter[str],
    ) -> dict[str, int | bool]:
        """Recheck the incumbent independently of the solver model."""
        normalized = Counter(
            {idx: int(round(value)) for idx, value in selected.items() if int(round(value)) > 0}
        )
        repaired, _state, placed = self._selection_state(normalized)
        errors: list[str] = []
        if repaired != normalized:
            errors.append("selected columns violate a physical or no-mix rule")
        for group in self.groups:
            assigned = int(placed.get(group.group_id, 0))
            missing = int(unplaced.get(group.group_id, 0))
            if assigned + missing != int(group.demand):
                errors.append(
                    f"demand balance failed for {group.group_id}: "
                    f"assigned={assigned}, unplaced={missing}, demand={group.demand}"
                )
        for idx in normalized:
            col = self._columns[idx]
            if col.size == "45" and col.bay_key not in self.area_edge_bays.get(col.area_no, set()):
                errors.append(f"45-ft column {col.column_id} is not on an edge large bay")
        if errors:
            raise RuntimeError("Independent solution validation failed: " + "; ".join(errors[:10]))
        return {
            "passed": True,
            "selected_columns_checked": len(normalized),
            "groups_checked": len(self.groups),
            "assigned_boxes_checked": int(sum(placed.values())),
            "unplaced_boxes_checked": int(sum(unplaced.values())),
        }

    def _capacity_reservation_margins(self, selected: Counter[int]) -> dict[str, dict[str, int]]:
        """Report residual slot and large-pair capacity after the final plan."""
        selected_slot_units: Counter[str] = Counter()
        selected_twenty_bays: set[str] = set()
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            selected_slot_units[col.area_no] += (
                int(col.quantity)
                * int(chosen)
                * len(self._placement_footprint_keys(col.bay_key, col.size))
            )
            if col.size == "20":
                selected_twenty_bays.add(col.bay_key)

        areas = sorted(set(selected_slot_units) | {area for area, _size in self.import_area_size_reservation})
        result: dict[str, dict[str, int]] = {}
        for area_no in areas:
            physical = sum(int(bay.physical_capacity) for bay in self.bays.values() if bay.area_no == area_no)
            reserved_slots = sum(
                int(qty) * (2 if size in {"40", "45"} else 1)
                for (reserved_area, size), qty in self.import_area_size_reservation.items()
                if reserved_area == area_no
            )
            pair_base = sum(
                int(capacity)
                for pair, capacity in self.large_pair_capacity.items()
                if self.bays[pair[0]].area_no == area_no
            )
            pair_loss = self._large_pair_capacity_loss(selected_twenty_bays, area_no=area_no)
            pair_required = sum(
                int(qty)
                for (reserved_area, size), qty in self.import_area_size_reservation.items()
                if reserved_area == area_no and size in {"40", "45"}
            )
            result[area_no] = {
                "residual_slot_units": int(physical - reserved_slots - selected_slot_units[area_no]),
                "large_pair_base_capacity": int(pair_base),
                "large_pair_capacity_loss": int(pair_loss),
                "import_large_pair_required": int(pair_required),
                "residual_large_pair_margin": int(pair_base - pair_loss - pair_required),
            }
        return result

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
        unplaced: Counter[str],
    ) -> dict[str, float | dict[str, float]]:
        """Evaluate raw, normalized and weighted stage-2 criteria."""
        actual_quota: Counter[tuple[str, str, str, str]] = Counter()
        used_group_area: set[tuple[tuple[str, ...], str]] = set()
        used_group_row: set[tuple[tuple[str, ...], str, str]] = set()
        used_groups: set[tuple[str, ...]] = set()
        used_twenty_bays: set[str] = set()
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
            if col.size == "20":
                used_twenty_bays.add(col.bay_key)
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
        guidance_l1 = 0.0
        for voyage_id, flow, area_no, big_size in target_keys:
            target = self._area_size_target(voyage_id, flow, area_no, big_size)
            guidance_l1 += abs(
                actual_quota.get((voyage_id, flow, area_no, big_size), 0) - target
            )

        raw = {
            "extra_operational_group_areas": float(max(0, len(used_group_area) - len(used_groups))),
            "extra_operational_group_rows": float(max(0, len(used_group_row) - len(used_groups))),
            "existing_group_normalized_distance_sum": float(proximity_sum),
            "area_guidance_l1_deviation": float(guidance_l1),
            "large_pair_capacity_loss": float(self._large_pair_capacity_loss(used_twenty_bays)),
            "berth_normalized_distance_sum": float(berth_distance_sum),
        }
        normalized = {
            "area_dispersion": raw["extra_operational_group_areas"] / self._objective_scale("area_dispersion"),
            "row_dispersion": raw["extra_operational_group_rows"] / self._objective_scale("row_dispersion"),
            "existing_group_proximity": raw["existing_group_normalized_distance_sum"] / self._objective_scale("existing_group_proximity"),
            "area_guidance": raw["area_guidance_l1_deviation"] / self._objective_scale("area_guidance_l1"),
            "large_pair_capacity": raw["large_pair_capacity_loss"] / self._objective_scale("large_pair_capacity"),
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

    def _selected_solution_energy(self, selected: Counter[int], unplaced: Counter[str]) -> float:
        # Unplaced quantity is handled lexicographically and is not blended
        # into the normalized secondary objective.
        return float(self._selected_objective_components(selected, unplaced)["weighted_total"])


    def _source_rank_for_group_id(self, group_id: str) -> int:
        return 0 if self.group_source.get(group_id, "document") == "document" else 1

    def _source_rank_for_group(self, group: SmallBoxGroup) -> int:
        return self._source_rank_for_group_id(group.group_id)

    def _unplaced_penalty_for_group_id(self, group_id: str) -> float:
        return max(1.0, float(self.config.pricing_unplaced_penalty))

    def _unplaced_objective_for_group(self, group: SmallBoxGroup, objective_mode: str) -> float:
        if objective_mode == "min_unplaced":
            return 1.0
        return self._unplaced_penalty_for_group_id(group.group_id)




    def _uses_original_output_scope(self) -> bool:
        return False



    @staticmethod
    def _row_area_summary_consistency_stats(small_rows: list[dict], area_bay_rows: list[dict]) -> dict[str, int]:
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
        for row in area_bay_rows:
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
            "row_area_summary_consistency_violations": violations,
            "row_area_summary_consistency_shortage_boxes": shortage,
        }

    @staticmethod
    def _row_bay_summary_consistency_stats(small_rows: list[dict], area_bay_rows: list[dict]) -> dict[str, int]:
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
        for row in area_bay_rows:
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
            "row_bay_summary_consistency_violations": violations,
            "row_bay_summary_consistency_shortage_boxes": shortage,
        }

    def _solve_by_column_generation(self) -> tuple[Counter[int], Counter[str], dict]:
        """Run exact two-phase LP pricing and solve the complete unit-flow MIP."""
        from gurobipy import quicksum

        Model = _GurobiModelAdapter
        pricing_start = perf_counter()
        stats: dict = {
            "gurobi_available": True,
            "pricing_method": "exact_auxiliary_lp_over_complete_unit_flow_universe",
            "pricing_iterations": [],
            "pricing_stop_reason": "",
            "full_unit_flow_count": len(self._columns),
        }

        def run_phase(
            phase_name: str,
            objective_mode: str,
            fixed_unplaced_total: float | None = None,
        ) -> tuple[float, Counter[str]]:
            last_objective = 0.0
            last_unplaced: Counter[str] = Counter()
            for iteration in range(max(1, int(self.config.max_iterations))):
                model, variables, _constraints = self._build_restricted_master(
                    Model,
                    quicksum,
                    relax=True,
                    objective_mode=objective_mode,
                    fixed_unplaced_total=fixed_unplaced_total,
                )
                self._set_gurobi_param(model, "TimeLimit", float(self.config.mip_time_limit))
                model.optimize()
                status = self._gurobi_status_name(model)
                if status != "optimal":
                    self._free_gurobi_model(model)
                    raise RuntimeError(
                        f"Strict {phase_name} pricing LP is not optimal: {status}"
                    )
                last_objective = self._gurobi_objective_value(model)
                last_unplaced = Counter(
                    {
                        group_id: self._gurobi_value(model, var)
                        for group_id, var in variables["unplaced"].items()
                        if self._gurobi_value(model, var) > 1e-8
                    }
                )
                pricing = self._solve_exact_pricing_oracle(
                    Model,
                    quicksum,
                    phase_name,
                    objective_mode,
                    fixed_unplaced_total,
                    last_objective,
                )
                stats["pricing_iterations"].append(
                    {
                        "phase": phase_name,
                        "iteration": iteration,
                        "lp_objective": last_objective,
                        "lp_unplaced_boxes": float(
                            sum(self._gurobi_value(model, var) for var in variables["unplaced"].values())
                        ),
                        **pricing,
                    }
                )
                self._free_gurobi_model(model)
                if pricing["new_columns"] == 0:
                    return last_objective, last_unplaced
            raise RuntimeError(
                f"Strict {phase_name} pricing exceeded max_iterations="
                f"{self.config.max_iterations}"
            )

        _phase1_objective, phase1_unplaced = run_phase(
            "minimum_unplaced",
            "min_unplaced",
        )
        phase1_unplaced_total = float(sum(phase1_unplaced.values()))
        phase2_objective, phase2_unplaced = run_phase(
            "secondary_objective",
            "full",
            fixed_unplaced_total=phase1_unplaced_total,
        )
        stats.update(
            {
                "pricing_stop_reason": "two_phase_reduced_cost_convergence",
                "pricing_iterations_run": len(stats["pricing_iterations"]),
                "pricing_phase1_lp_unplaced_boxes": phase1_unplaced_total,
                "pricing_phase2_lp_objective": phase2_objective,
                "column_generation_pricing_elapsed_seconds": round(
                    perf_counter() - pricing_start, 3
                ),
            }
        )

        self._active_column_indices = set(range(len(self._columns)))
        integer_start = perf_counter()
        selected, unplaced, integer_stats = self._solve_lexicographic_integer_master(
            Counter(self._master_seed_selected),
            Counter(self._master_seed_unplaced),
            float(self.config.total_time_limit) if self.config.total_time_limit > 0 else None,
        )
        stats.update(integer_stats)
        stats.update(
            {
                "master_algorithm": "strict_two_phase_column_generation_full_unit_flow_mip",
                "master_bound_scope": "complete_unit_flow_universe",
                "master_status": (
                    "lexicographic_integer_master"
                    if integer_stats.get("lexicographic_integer_master_used")
                    else "integer_master_incomplete"
                ),
                "master_solve_seconds": round(perf_counter() - integer_start, 3),
                "master_objective": self._selected_solution_energy(selected, unplaced),
                "master_mip_gap": integer_stats.get("lexicographic_stage2_gap"),
                "restricted_master_lp_unplaced_boxes": float(sum(phase2_unplaced.values())),
            }
        )
        if not integer_stats.get("lexicographic_integer_master_used"):
            raise RuntimeError(
                "The final two-stage integer master did not produce a stage-2 solution"
            )
        return selected, unplaced, stats

    def _solve_exact_pricing_oracle(
        self,
        Model,
        quicksum,
        phase_name: str,
        objective_mode: str,
        fixed_unplaced_total: float | None,
        restricted_objective: float,
    ) -> dict:
        """Solve the exact auxiliary LP over all not-yet-admitted unit flows.

        The restricted master itself contains active variables only. The
        auxiliary pricing LP shares its constraints and objective but admits
        the complete unit-flow universe. Any inactive flow used by its optimum
        is a certificate that the restricted master is incomplete; admitting
        those flows reproduces the oracle optimum in the next iteration.
        """
        saved_active = set(self._active_column_indices)
        self._active_column_indices = set(range(len(self._columns)))
        oracle, oracle_vars, _constraints = self._build_restricted_master(
            Model,
            quicksum,
            relax=True,
            objective_mode=objective_mode,
            fixed_unplaced_total=fixed_unplaced_total,
        )
        self._set_gurobi_param(oracle, "TimeLimit", float(self.config.mip_time_limit))
        oracle.optimize()
        status = self._gurobi_status_name(oracle)
        if status != "optimal":
            self._free_gurobi_model(oracle)
            self._active_column_indices = saved_active
            raise RuntimeError(f"Exact {phase_name} pricing oracle is not optimal: {status}")
        oracle_objective = self._gurobi_objective_value(oracle)
        improving = {
            idx
            for idx, var in oracle_vars["column"].items()
            if idx not in saved_active and self._gurobi_value(oracle, var) > 1e-8
        }
        self._free_gurobi_model(oracle)
        tolerance = 1e-7 * max(1.0, abs(float(restricted_objective)))
        improvement = float(restricted_objective) - float(oracle_objective)
        if improvement > tolerance and not improving:
            self._active_column_indices = saved_active
            raise RuntimeError(
                f"Exact {phase_name} pricing found an objective improvement "
                "but no entering unit-flow column"
            )
        if improvement <= tolerance:
            improving.clear()
        self._active_column_indices = saved_active | improving
        return {
            "new_columns": len(improving),
            "pricing_mode": "exact_auxiliary_unit_flow_lp",
            "exact_pricing": True,
            "inactive_columns_scanned": len(self._columns) - len(saved_active),
            "active_column_count": len(self._active_column_indices),
            "pricing_oracle_objective": oracle_objective,
            "pricing_objective_improvement": max(0.0, improvement),
        }

    def _solve_lexicographic_integer_master(
        self,
        start_selected: Counter[int],
        start_unplaced: Counter[str],
        remaining_seconds: float | None,
    ) -> tuple[Counter[int], Counter[str], dict]:
        """Solve the restricted integer master in two exact objective stages."""
        from gurobipy import quicksum

        Model = _GurobiModelAdapter
        stats = {
            "lexicographic_integer_master_used": False,
            "lexicographic_stage1_status": "not_run",
            "lexicographic_stage2_status": "not_run",
            "lexicographic_stage1_optimal": False,
            "lexicographic_stage2_optimal": False,
        }
        available = float(self.config.mip_time_limit) * 2.0
        if remaining_seconds is not None:
            available = max(0.0, min(available, float(remaining_seconds)))
        if available < 2.0:
            stats["lexicographic_skip_reason"] = "insufficient_time"
            return Counter(start_selected), Counter(start_unplaced), stats

        def apply_start(vars_by_kind: dict, selected: Counter[int], unplaced: Counter[str]) -> None:
            for idx, var in vars_by_kind["column"].items():
                var.Start = float(max(0, int(selected.get(idx, 0))))
            for group_id, var in vars_by_kind["unplaced"].items():
                var.Start = float(max(0, int(unplaced.get(group_id, 0))))

        stage1, stage1_vars, _ = self._build_restricted_master(
            Model,
            quicksum,
            relax=False,
            objective_mode="min_unplaced",
        )
        apply_start(stage1_vars, start_selected, start_unplaced)
        stage1_limit = max(1.0, available / 2.0)
        self._set_gurobi_param(stage1, "TimeLimit", stage1_limit)
        self._set_gurobi_param(stage1, "MIPGap", 0.0)
        stage1.optimize()
        stage1_status = self._gurobi_status_name(stage1)
        stats["lexicographic_stage1_status"] = stage1_status
        stats["lexicographic_stage1_optimal"] = stage1_status == "optimal"
        if stage1_status != "optimal":
            self._free_gurobi_model(stage1)
            raise RuntimeError(
                "Lexicographic stage 1 must be proven optimal before stage 2; "
                f"status={stage1_status}"
            )
        stage1_selected = Counter(
            {
                idx: int(round(self._gurobi_value(stage1, var)))
                for idx, var in stage1_vars["column"].items()
                if self._gurobi_value(stage1, var) > 0.5
            }
        )
        stage1_unplaced = self._gurobi_unplaced_values(stage1, stage1_vars)
        optimum_unplaced = int(sum(stage1_unplaced.values()))
        stats["lexicographic_stage1_unplaced_boxes"] = optimum_unplaced
        stats["lexicographic_stage1_bound"] = self._gurobi_dual_bound(stage1)
        self._free_gurobi_model(stage1)

        stage2, stage2_vars, _ = self._build_restricted_master(
            Model,
            quicksum,
            relax=False,
            objective_mode="full",
            fixed_unplaced_total=optimum_unplaced,
        )
        preferred_selected = start_selected if sum(start_unplaced.values()) == optimum_unplaced else stage1_selected
        preferred_unplaced = start_unplaced if sum(start_unplaced.values()) == optimum_unplaced else stage1_unplaced
        apply_start(stage2_vars, preferred_selected, preferred_unplaced)
        self._set_gurobi_param(stage2, "TimeLimit", max(1.0, available - stage1_limit))
        self._set_gurobi_param(stage2, "MIPGap", max(0.0, float(self.config.mip_gap)))
        stage2.optimize()
        stage2_status = self._gurobi_status_name(stage2)
        stats["lexicographic_stage2_status"] = stage2_status
        stats["lexicographic_stage2_optimal"] = stage2_status == "optimal"
        if self._gurobi_solution_count(stage2) <= 0:
            self._free_gurobi_model(stage2)
            stats["lexicographic_skip_reason"] = "stage2_no_solution"
            return stage1_selected, stage1_unplaced, stats
        selected = Counter(
            {
                idx: int(round(self._gurobi_value(stage2, var)))
                for idx, var in stage2_vars["column"].items()
                if self._gurobi_value(stage2, var) > 0.5
            }
        )
        unplaced = self._gurobi_unplaced_values(stage2, stage2_vars)
        stats.update(
            {
                "lexicographic_integer_master_used": True,
                "lexicographic_stage2_objective": self._gurobi_objective_value(stage2),
                "lexicographic_stage2_bound": self._gurobi_dual_bound(stage2),
                "lexicographic_stage2_gap": self._gurobi_gap(stage2),
            }
        )
        self._free_gurobi_model(stage2)
        return selected, unplaced, stats



    def _gurobi_unplaced_values(self, model, lp_vars) -> Counter[str]:
        return Counter(
            {
                group_id: int(round(self._gurobi_value(model, var)))
                for group_id, var in lp_vars["unplaced"].items()
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
            ColumnGenerationPlanner._set_gurobi_param(model, name, value)
            return True
        except Exception:
            return False

    @staticmethod
    def _set_gurobi_param(model, name: str, value: object) -> None:
        model.setParam(name, value)

    @staticmethod
    def _gurobi_status_name(model) -> str:
        return str(model.getStatus()).lower()

    @staticmethod
    def _gurobi_solution_count(model) -> int:
        try:
            return int(model.getNSols())
        except Exception:
            try:
                return 1 if model.getBestSol() is not None else 0
            except Exception:
                return 0

    @staticmethod
    def _gurobi_objective_value(model) -> float:
        try:
            return float(model.getObjVal())
        except Exception:
            return float("nan")

    @staticmethod
    def _gurobi_gap(model) -> float:
        try:
            return float(model.getGap())
        except Exception:
            return 0.0


    @staticmethod
    def _gurobi_dual_bound(model) -> float:
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
    def _gurobi_value(model, var) -> float:
        return float(model.getVal(var))


    @staticmethod
    def _free_gurobi_model(model) -> None:
        for method_name in ("freeTransform", "freeProb"):
            method = getattr(model, method_name, None)
            if method is None:
                continue
            try:
                method()
                return
            except Exception:
                continue


    def _build_restricted_master(
        self,
        Model,
        quicksum,
        relax: bool,
        objective_mode: str = "full",
        fixed_unplaced_total: float | None = None,
    ):
        model = Model("yard_export_row_column_generation_gurobi")
        self._configure_gurobi_output(model)
        try:
            model.setMinimize()
        except Exception:
            pass
        column_vtype = "C" if relax else "I"
        columns = {
            idx: model.addVar(
                lb=0.0,
                ub=float(self.group_demand[col.group_id]),
                vtype=column_vtype,
                obj=0.0 if objective_mode == "min_unplaced" else col.intrinsic_cost,
                name=f"col_{idx}",
            )
            for idx, col in enumerate(self._columns)
            if idx in self._active_column_indices
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
        area_physical_cols: defaultdict[str, list[tuple[int, PlacementColumn, int]]] = defaultdict(list)
        bay_size_capacity_cols: defaultdict[tuple[str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_port_size_cols: defaultdict[tuple[str, str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        row_capacity_cols: defaultdict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        row_size_capacity_cols: defaultdict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
        row_attr_choice_cols: defaultdict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
        bay_attr_choice_cols: defaultdict[tuple[str, str, str, str], list[int]] = defaultdict(list)
        area_size_cols: defaultdict[tuple[str, str, str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        group_area_cols: defaultdict[tuple[tuple[str, ...], str], list[int]] = defaultdict(list)
        group_row_cols: defaultdict[tuple[tuple[str, ...], str, str], list[int]] = defaultdict(list)
        twenty_cols_by_large_pair: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, col in enumerate(self._columns):
            if idx not in columns:
                continue
            group_cols[col.group_id].append((idx, col))
            for footprint_key in self._placement_footprint_keys(col.bay_key, col.size):
                bay_capacity_cols[footprint_key].append((idx, col))
                bay_port_size_cols[(footprint_key, self._row_mix_key_for_column(col), col.size)].append((idx, col))
                for attr in self._bay_no_mix_attrs_for_column(col):
                    scope = self._attr_voyage_scope(attr, col.voyage_id)
                    bay_attr_choice_cols[(footprint_key, attr, scope, self._column_attr_value(col, attr))].append(idx)
            footprint_units = len(self._placement_footprint_keys(col.bay_key, col.size))
            area_physical_cols[col.area_no].append((idx, col, footprint_units))
            for footprint_key, row_no, qty in col.row_allocation:
                row_capacity_cols[(footprint_key, row_no)].append((idx, int(qty)))
                row_size_capacity_cols[(footprint_key, row_no, col.size)].append((idx, int(qty)))
                for attr in self._row_no_mix_attrs_for_column(col):
                    scope = self._attr_voyage_scope(attr, col.voyage_id)
                    row_attr_choice_cols[(footprint_key, row_no, attr, scope, self._column_attr_value(col, attr))].append(idx)
            bay_size_capacity_cols[(col.bay_key, col.size)].append((idx, col))
            area_size_cols[col.quota_key].append((idx, col))
            group_area_cols[(col.group_key, col.area_no)].append(idx)
            for footprint_key, row_no, _qty in col.row_allocation:
                if footprint_key == col.bay_key:
                    group_row_cols[(col.group_key, footprint_key, row_no)].append(idx)
            if col.size == "20":
                pair = self.large_pair_by_member.get(col.bay_key)
                if pair is not None:
                    twenty_cols_by_large_pair[pair].append(idx)

        group_cover = {}
        for group in self.groups:
            expr = quicksum(col.quantity * columns[idx] for idx, col in group_cols.get(group.group_id, []))
            group_cover[group.group_id] = model.addCons(expr + unplaced[group.group_id] == group.demand, name=f"cover_{group.group_id}")

        bay_capacity_limit = {}
        for bay_key, items in bay_capacity_cols.items():
            bay_capacity_limit[bay_key] = model.addCons(
                quicksum(col.quantity * columns[idx] for idx, col in items) <= self.bays[bay_key].physical_capacity,
                name=f"bay_cap_{bay_key}",
            )
        area_reservation_limit = {}
        reservation_keys = set(self.import_area_size_reservation)
        for area_no in sorted({area for area, _size in reservation_keys}):
            physical_capacity = sum(
                int(bay.physical_capacity)
                for bay in self.bays.values()
                if bay.area_no == area_no
            )
            reserved_slot_units = 0
            for (reserved_area, size), qty in self.import_area_size_reservation.items():
                if reserved_area == area_no:
                    reserved_slot_units += int(qty) * (2 if size in {"40", "45"} else 1)
            available_for_detail = max(0, physical_capacity - reserved_slot_units)
            area_reservation_limit[area_no] = model.addCons(
                quicksum(
                    col.quantity * footprint_units * columns[idx]
                    for idx, col, footprint_units in area_physical_cols.get(area_no, [])
                ) <= available_for_detail,
                name=f"area_after_aggregate_reservation_{area_no}",
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
        large_pair_limits = self._add_large_pair_preservation_constraints(
            quicksum,
            model,
            columns,
            twenty_cols_by_large_pair,
            relax=relax,
            objective_mode=objective_mode,
        )
        # Big-plan allocations are soft inheritance targets. Hard upper bounds
        # are intentionally omitted so detailed declared boxes can recover from
        # stale or physically incompatible upstream allocations.
        quota_limit = {}
        seed_unplaced_limit = None
        if not relax and (self._master_seed_selected or self._master_seed_unplaced):
            seed_unplaced_limit = model.addCons(
                quicksum(unplaced.values()) <= sum(self._master_seed_unplaced.values()),
                name="seed_unplaced_cap",
            )
        lexicographic_unplaced_limit = None
        if fixed_unplaced_total is not None:
            lexicographic_unplaced_limit = model.addCons(
                quicksum(unplaced.values()) == float(fixed_unplaced_total),
                name="lexicographic_unplaced_total",
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
                group_row_cols,
            )
        else:
            self._add_integer_master_objectives(
                quicksum,
                model,
                columns,
                area_size_cols,
                group_area_cols,
                group_row_cols,
            )
        self._add_bay_compatibility_constraints(
            quicksum, model, columns, bay_attr_choice_cols, relax=relax
        )
        self._add_row_compatibility_constraints(
            quicksum, model, columns, row_attr_choice_cols, relax=relax
        )
        return model, {"column": columns, "unplaced": unplaced}, {
            "group_cover": group_cover,
            "bay_capacity_limit": bay_capacity_limit,
            "area_reservation_limit": area_reservation_limit,
            "bay_size_limit": bay_size_limit,
            "row_capacity_limit": row_capacity_limit,
            "row_size_limit": row_size_limit,
            "large_pair_loss_link": large_pair_limits.get("loss_link", {}),
            "import_large_pair_reservation": large_pair_limits.get("reservation", {}),
            "bay_port_stack_link": bay_port_stack_link,
            "bay_port_stack_limit": bay_port_stack_limit,
            "bay_stack_total_limit": bay_stack_total_limit,
            "quota_limit": quota_limit,
            "seed_unplaced_limit": seed_unplaced_limit,
            "lexicographic_unplaced_limit": lexicographic_unplaced_limit,
            **relaxed_objective_constraints,
        }

    def _add_large_pair_preservation_constraints(
        self,
        quicksum,
        model,
        columns,
        twenty_cols_by_large_pair: dict[tuple[str, str], list[int]],
        relax: bool,
        objective_mode: str,
    ) -> dict[str, dict]:
        """Preserve usable large-container pairs and price their exact loss.

        A pair is lost once any selected 20-ft placement occupies either
        member. Known 40/45-ft import reservations must fit in the remaining
        pair capacity.
        """
        unavailable_by_pair = {}
        loss_link = {}
        vtype = "C" if relax else "B"
        loss_penalty = 0.0 if objective_mode == "min_unplaced" else self._twenty_segment_loss_penalty()
        for pair, capacity in sorted(self.large_pair_capacity.items()):
            indices = sorted(set(twenty_cols_by_large_pair.get(pair, [])))
            unavailable = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=vtype,
                obj=loss_penalty * int(capacity),
                name=f"large_pair_unavailable_{self._key_name(pair)}",
            )
            unavailable_by_pair[pair] = unavailable
            for idx in indices:
                loss_link[(pair, idx)] = model.addCons(
                    columns[idx]
                    <= max(1, self.group_demand[self._columns[idx].group_id]) * unavailable,
                    name=f"large_pair_loss_link_{len(loss_link)}",
                )
            if indices:
                model.addCons(
                    unavailable <= quicksum(columns[idx] for idx in indices),
                    name=f"large_pair_loss_exact_{self._key_name(pair)}",
                )
            else:
                model.addCons(unavailable == 0.0, name=f"large_pair_loss_zero_{self._key_name(pair)}")

        reservation = {}
        import_large_by_area: Counter[str] = Counter()
        import_45_by_area: Counter[str] = Counter()
        for (area_no, size), qty in self.import_area_size_reservation.items():
            if size in {"40", "45"} and int(qty) > 0:
                import_large_by_area[str(area_no)] += int(qty)
            if size == "45" and int(qty) > 0:
                import_45_by_area[str(area_no)] += int(qty)
        for area_no, required in sorted(import_large_by_area.items()):
            pairs = [
                (pair, capacity)
                for pair, capacity in self.large_pair_capacity.items()
                if self.bays[pair[0]].area_no == area_no
            ]
            reservation[area_no] = model.addCons(
                quicksum(int(capacity) * (1.0 - unavailable_by_pair[pair]) for pair, capacity in pairs)
                >= int(required),
                name=f"import_large_pair_reservation_{area_no}",
            )
            required_45 = int(import_45_by_area.get(area_no, 0))
            if required_45 > 0:
                reservation[(area_no, "45")] = model.addCons(
                    quicksum(
                        int(self.large_pair_capacity_45.get(pair, 0)) * (1.0 - unavailable_by_pair[pair])
                        for pair, _capacity in pairs
                    )
                    >= required_45,
                    name=f"import_45_pair_reservation_{area_no}",
                )
        return {"loss_link": loss_link, "reservation": reservation}



    def _add_relaxed_master_objectives(
        self,
        quicksum,
        model,
        columns,
        area_size_cols,
        group_area_cols,
        group_row_cols,
    ) -> dict[str, dict]:
        area_size_keys = {
            key
            for key in area_size_cols
            if self._has_area_guidance(key[0], key[1], key[3])
        }
        for key, qty in self.quota_by_key.items():
            voyage_id, flow, _area_no, big_size = key
            if qty > 0 and self.voyage_flow_size_demand[(voyage_id, flow, big_size)] > 0:
                area_size_keys.add(key)

        area_guidance_balance = {}
        for key in sorted(area_size_keys):
            items = area_size_cols.get(key, [])
            voyage_id, flow, area_no, big_size = key
            target = self._area_size_target(voyage_id, flow, area_no, big_size)
            pos = model.addVar(lb=0.0, obj=self._area_guidance_penalty(), name=f"lp_guide_pos_{len(model.getVars())}")
            neg = model.addVar(lb=0.0, obj=self._area_guidance_penalty(), name=f"lp_guide_neg_{len(model.getVars())}")
            actual = quicksum(col.quantity * columns[idx] for idx, col in items)
            area_guidance_balance[key] = model.addCons(actual - target == pos - neg)

        fixed_use_constraints = {}
        area_use_by_group: defaultdict[tuple[str, ...], list] = defaultdict(list)
        row_use_by_group: defaultdict[tuple[str, ...], list] = defaultdict(list)
        column_indices_by_group: defaultdict[tuple[str, ...], set[int]] = defaultdict(set)
        for key, indices in group_area_cols.items():
            use = model.addVar(lb=0.0, ub=1.0, obj=self._area_activation_penalty())
            group_key, _area_no = key
            demand = sum(
                group.demand
                for group in self.groups
                if self._operational_group_key(group) == group_key
            )
            fixed_use_constraints[("group_area",) + key] = model.addCons(
                quicksum(self._columns[idx].quantity * columns[idx] for idx in indices)
                <= max(1, demand) * use
            )
            area_use_by_group[group_key].append(use)
            column_indices_by_group[group_key].update(indices)
        for key, indices in group_row_cols.items():
            use = model.addVar(lb=0.0, ub=1.0, obj=self._row_activation_penalty())
            group_key, _bay_key, _row_no = key
            demand = sum(
                group.demand
                for group in self.groups
                if self._operational_group_key(group) == group_key
            )
            fixed_use_constraints[("group_row",) + key] = model.addCons(
                quicksum(columns[idx] for idx in indices) <= max(1, demand) * use
            )
            row_use_by_group[group_key].append(use)
            column_indices_by_group[group_key].update(indices)
        for group_key, indices in column_indices_by_group.items():
            demand = sum(
                group.demand
                for group in self.groups
                if self._operational_group_key(group) == group_key
            )
            baseline = model.addVar(
                lb=0.0,
                ub=1.0,
                obj=-(self._area_activation_penalty() + self._row_activation_penalty()),
                name=f"lp_group_used_{self._key_name(group_key)}",
            )
            assigned = quicksum(columns[idx] for idx in sorted(indices))
            fixed_use_constraints[("group_used_upper",) + group_key] = model.addCons(
                assigned <= max(1, demand) * baseline
            )
            fixed_use_constraints[("group_used_lower",) + group_key] = model.addCons(
                baseline <= assigned
            )
            fixed_use_constraints[("group_used_area",) + group_key] = model.addCons(
                baseline <= quicksum(area_use_by_group[group_key])
            )
            fixed_use_constraints[("group_used_row",) + group_key] = model.addCons(
                baseline <= quicksum(row_use_by_group[group_key])
            )
        return {
            "area_guidance_balance": area_guidance_balance,
            "fixed_use_objective_limit": fixed_use_constraints,
        }

    def _add_integer_master_objectives(
        self,
        quicksum,
        model,
        columns,
        area_size_cols,
        group_area_cols,
        group_row_cols,
    ) -> None:
        area_size_keys = {
            key
            for key in area_size_cols
            if self._has_area_guidance(key[0], key[1], key[3])
        }
        for key, qty in self.quota_by_key.items():
            voyage_id, flow, _area_no, big_size = key
            if qty > 0 and self.voyage_flow_size_demand[(voyage_id, flow, big_size)] > 0:
                area_size_keys.add(key)
        for key in sorted(area_size_keys):
            items = area_size_cols.get(key, [])
            voyage_id, flow, area_no, big_size = key
            target = self._area_size_target(voyage_id, flow, area_no, big_size)
            pos = model.addVar(lb=0.0, obj=self._area_guidance_penalty(), name=f"guide_pos_{len(model.getVars())}")
            neg = model.addVar(lb=0.0, obj=self._area_guidance_penalty(), name=f"guide_neg_{len(model.getVars())}")
            actual = quicksum(col.quantity * columns[idx] for idx, col in items)
            model.addCons(actual - target == pos - neg)

        area_use_by_group: defaultdict[tuple[str, ...], list] = defaultdict(list)
        row_use_by_group: defaultdict[tuple[str, ...], list] = defaultdict(list)
        column_indices_by_group: defaultdict[tuple[str, ...], set[int]] = defaultdict(set)
        for (group_key, area_no), indices in group_area_cols.items():
            use = model.addVar(vtype="B", obj=self._area_activation_penalty(), name=f"use_ga_{self._key_name(group_key)}_{area_no}")
            demand = sum(
                group.demand
                for group in self.groups
                if self._operational_group_key(group) == group_key
            )
            model.addCons(
                quicksum(self._columns[idx].quantity * columns[idx] for idx in indices)
                <= max(1, demand) * use
            )
            area_use_by_group[group_key].append(use)
            column_indices_by_group[group_key].update(indices)
        for (group_key, bay_key, row_no), indices in group_row_cols.items():
            use = model.addVar(
                vtype="B",
                obj=self._row_activation_penalty(),
                name=f"use_gr_{self._key_name(group_key)}_{bay_key}_{row_no}",
            )
            demand = sum(
                group.demand
                for group in self.groups
                if self._operational_group_key(group) == group_key
            )
            model.addCons(quicksum(columns[idx] for idx in indices) <= max(1, demand) * use)
            row_use_by_group[group_key].append(use)
            column_indices_by_group[group_key].update(indices)
        for group_key, indices in column_indices_by_group.items():
            demand = sum(
                group.demand
                for group in self.groups
                if self._operational_group_key(group) == group_key
            )
            baseline = model.addVar(
                vtype="B",
                obj=-(self._area_activation_penalty() + self._row_activation_penalty()),
                name=f"group_used_{self._key_name(group_key)}",
            )
            assigned = quicksum(columns[idx] for idx in sorted(indices))
            model.addCons(assigned <= max(1, demand) * baseline)
            model.addCons(baseline <= assigned)
            model.addCons(baseline <= quicksum(area_use_by_group[group_key]))
            model.addCons(baseline <= quicksum(row_use_by_group[group_key]))
    def _add_bay_compatibility_constraints(
        self,
        quicksum,
        model,
        columns,
        bay_attr_choice_cols: dict[tuple[str, str, str, str], list[int]],
        relax: bool,
    ) -> None:
        big_m = max(1, sum(group.demand for group in self.groups))
        vtype = "C" if relax else "B"
        use_by_bay_attr: defaultdict[tuple[str, str, str], list] = defaultdict(list)
        for (bay_key, attr, scope, value), indices in sorted(bay_attr_choice_cols.items()):
            scope_name = scope or "GLOBAL"
            use = model.addVar(lb=0.0, ub=1.0, vtype=vtype, name=f"bay_use_{attr}_{scope_name}_{bay_key}_{value}")
            model.addCons(quicksum(columns[idx] for idx in indices) <= big_m * use)
            use_by_bay_attr[(bay_key, attr, scope)].append(use)
        for (bay_key, attr, scope), uses in use_by_bay_attr.items():
            scope_name = scope or "GLOBAL"
            model.addCons(quicksum(uses) <= 1, name=f"bay_one_{attr}_{scope_name}_{bay_key}")

    def _add_row_compatibility_constraints(
        self,
        quicksum,
        model,
        columns,
        row_attr_choice_cols: dict[tuple[str, str, str, str, str], list[int]],
        relax: bool,
    ) -> None:
        big_m = max(1, sum(group.demand for group in self.groups))
        vtype = "C" if relax else "B"
        use_by_row_attr: defaultdict[tuple[str, str, str, str], list] = defaultdict(list)
        for (bay_key, row_no, attr, scope, value), indices in sorted(row_attr_choice_cols.items()):
            scope_name = scope or "GLOBAL"
            use = model.addVar(lb=0.0, ub=1.0, vtype=vtype, name=f"row_use_{attr}_{scope_name}_{bay_key}_{row_no}_{value}")
            model.addCons(quicksum(columns[idx] for idx in indices) <= big_m * use)
            use_by_row_attr[(bay_key, row_no, attr, scope)].append(use)
        for (bay_key, row_no, attr, scope), uses in use_by_row_attr.items():
            scope_name = scope or "GLOBAL"
            model.addCons(quicksum(uses) <= 1, name=f"row_one_{attr}_{scope_name}_{bay_key}_{row_no}")








    def _build_unit_flow_column_universe(self) -> None:
        """Create every feasible group/bay/row placement option as metadata.

        A column carries an integer number of identical containers on one row.
        Combining these unit-flow options represents every integer row
        allocation, so no capped list of multi-row templates is required.
        Only a small seed subset is admitted to the initial master.
        """
        self._columns.clear()
        self._column_keys.clear()
        self._active_column_indices.clear()
        for group in self.groups:
            seed_bays = {
                bay_key
                for bay_key, _capacity, _cost in self._candidate_bays_for_group(group)[
                    : max(1, int(self.config.initial_columns_per_group))
                ]
            }
            for bay_key, _max_qty, base_cost in self._candidate_bays_for_group(group):
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
                    row_capacity = min(per_bay[key][row_no] for key in footprint)
                    if row_capacity <= 0:
                        continue
                    allocation = self._row_allocation_signature(
                        tuple((key, row_no, 1) for key in footprint)
                    )
                    column_id = f"C{len(self._columns) + 1:07d}"
                    column = PlacementColumn(
                        column_id=column_id,
                        group_id=group.group_id,
                        demand_source="document",
                        voyage_id=group.voyage_id,
                        flow=group.status,
                        port=group.port,
                        size=group.size,
                        big_plan_size=self._big_plan_size(group.size),
                        height=group.height,
                        weight_class="",
                        special_stow_code="",
                        attributes=dict(group.attributes or {}),
                        area_no=bay.area_no,
                        bay_key=bay_key,
                        bay_no=bay.bay_no,
                        block_id="",
                        block_bays=(),
                        quantity=1,
                        stack_units=1,
                        row_allocation=allocation,
                        quota_key=self._quota_key(group, bay.area_no),
                        group_key=self._operational_group_key(group),
                        intrinsic_cost=(
                            base_cost
                            + self._berth_distance_cost(group.voyage_id, bay.area_no, 1)
                        ),
                    )
                    idx = len(self._columns)
                    self._columns.append(column)
                    self._column_keys.add((group.group_id, bay_key, 1, allocation))
                    if bay_key in seed_bays:
                        self._active_column_indices.add(idx)

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
        is_height = str(attr).strip().upper() in HEIGHT_NO_MIX_ATTRS
        return "" if self._is_global_row_no_mix_attr(attr) or is_height else str(voyage_id)

    def _bay_state_attr_key(self, bay_key: str, attr: str, voyage_id: object) -> tuple[str, str, str]:
        return (bay_key, attr, self._attr_voyage_scope(attr, voyage_id))

    def _row_state_attr_key(self, bay_key: str, row_no: str, attr: str, voyage_id: object) -> tuple[str, str, str, str]:
        return (bay_key, str(row_no), attr, self._attr_voyage_scope(attr, voyage_id))

    def _existing_bay_attr_values(self, bay: Bay, attr: str, voyage_id: object) -> set[str]:
        if self._is_size_no_mix_attr(attr):
            values = set(getattr(bay, "existing_attrs", {}).get(attr, set()))
            return values or set(getattr(bay, "existing_size_modes", set()))
        if str(attr).strip().upper() in HEIGHT_NO_MIX_ATTRS:
            return set(getattr(bay, "existing_heights", set()))
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


    def _greedy_fallback(self) -> tuple[Counter[int], Counter[str]]:
        selected: Counter[int] = Counter()
        placed: Counter[str] = Counter()
        state = self._empty_selection_state()
        indices_by_group: defaultdict[str, list[int]] = defaultdict(list)
        for idx, column in enumerate(self._columns):
            indices_by_group[column.group_id].append(idx)
        for group in self.groups:
            ordered = sorted(
                indices_by_group[group.group_id],
                key=lambda idx: (
                    self._columns[idx].intrinsic_cost,
                    self.bays[self._columns[idx].bay_key].bay_order,
                    self._columns[idx].row_allocation,
                ),
            )
            for idx in ordered:
                column = self._columns[idx]
                while placed[group.group_id] < group.demand:
                    remaining = group.demand - placed[group.group_id]
                    if not self._column_fits_state(column, state, remaining):
                        break
                    self._apply_column_to_state(column, state)
                    selected[idx] += 1
                    placed[group.group_id] += 1
        unplaced = Counter(
            {
                group.group_id: group.demand - placed[group.group_id]
                for group in self.groups
                if placed[group.group_id] < group.demand
            }
        )
        return selected, unplaced







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
            "bay_used_size": {},
            "bay_used_attrs": {},
            "twenty_segment_used_bays": set(),
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
    ) -> int:
        bay = self.bays[bay_key]
        footprint = self._placement_footprint_keys(bay_key, group.size)
        if not footprint:
            return 0
        if not self._bay_state_attrs_allow_group(group, footprint, state):
            return 0
        if group.size == "20":
            pair = self.large_pair_by_member.get(bay_key)
            used_twenty = set(state.get("twenty_segment_used_bays", set()))
            if pair is not None and pair[0] not in used_twenty and pair[1] not in used_twenty:
                base_capacity = sum(
                    capacity
                    for candidate, capacity in self.large_pair_capacity.items()
                    if self.bays[candidate[0]].area_no == bay.area_no
                )
                current_loss = self._large_pair_capacity_loss(used_twenty, area_no=bay.area_no)
                required = sum(
                    int(qty)
                    for (area_no, size), qty in self.import_area_size_reservation.items()
                    if area_no == bay.area_no and size in {"40", "45"}
                )
                if base_capacity - current_loss - self.large_pair_capacity[pair] < required:
                    return 0
                required_45 = sum(
                    int(qty)
                    for (area_no, size), qty in self.import_area_size_reservation.items()
                    if area_no == bay.area_no and size == "45"
                )
                if required_45 > 0:
                    base_45 = sum(
                        capacity
                        for candidate, capacity in self.large_pair_capacity_45.items()
                        if self.bays[candidate[0]].area_no == bay.area_no
                    )
                    current_45_loss = sum(
                        capacity
                        for candidate, capacity in self.large_pair_capacity_45.items()
                        if self.bays[candidate[0]].area_no == bay.area_no
                        and (candidate[0] in used_twenty or candidate[1] in used_twenty)
                    )
                    if base_45 - current_45_loss - self.large_pair_capacity_45.get(pair, 0) < required_45:
                        return 0
        capacity = int(remaining)
        for key in footprint:
            capacity = min(capacity, self.bays[key].physical_capacity - state["bay_load"][key])
        capacity = min(capacity, bay.cap_by_size.get(group.size, 0) - state["bay_size_load"][(bay_key, group.size)])
        capacity = min(capacity, self._row_capacity_for_column(group, bay_key, state=state))
        area_physical_capacity = sum(
            int(item.physical_capacity) for item in self.bays.values() if item.area_no == bay.area_no
        )
        reserved_units = sum(
            int(qty) * (2 if size in {"40", "45"} else 1)
            for (area_no, size), qty in (
                self.import_area_size_reservation
            ).items()
            if area_no == bay.area_no
        )
        area_remaining_units = max(0, area_physical_capacity - reserved_units - state["area_slot_load"][bay.area_no])
        capacity = min(capacity, area_remaining_units // max(1, len(footprint)))
        return max(0, int(capacity))

    def _apply_column_to_state(self, col: PlacementColumn, state: dict) -> None:
        footprint = self._placement_footprint_keys(col.bay_key, col.size)
        state["area_slot_load"][col.area_no] += col.quantity * len(footprint)
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
        state["used_group_area"].add((col.group_key, col.area_no))
        state["used_voyage_area"].add((col.voyage_id, col.area_no))
        if col.size == "20":
            state.setdefault("twenty_segment_used_bays", set()).add(col.bay_key)
        state["big_plan_quota_used"][col.quota_key] += col.quantity


    def _candidate_bays_for_group(self, group: SmallBoxGroup, scope: str | None = None) -> list[tuple[str, int, float]]:
        scope = scope or self._candidate_scope
        cache_key = (scope, group.group_id)
        cached = self._candidate_cache.get(cache_key)
        if cached is not None:
            return cached
        out: list[tuple[str, int, float]] = []
        for area_no in self._candidate_areas_for_group(group, scope=scope):
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
        self._candidate_cache[cache_key] = out
        return out


    def _candidate_areas_for_group(self, group: SmallBoxGroup, scope: str | None = None) -> list[str]:
        scope = scope or self._candidate_scope

        return sorted(
            [
                area_no
                for area_no in self.bays_by_area
                if self._candidate_area_base_scope(group, area_no, scope)
                and self._area_supports_group_flow(group, area_no)
            ],
            key=lambda area_no: (
                0 if self._is_big_plan_area_for_group(group, area_no) else 1,
                area_no,
            ),
        )

    def _candidate_area_base_scope(self, group: SmallBoxGroup, area_no: str, scope: str) -> bool:
        big_size = self._big_plan_size(group.size)
        return True












    def _area_supports_group_flow(self, group: SmallBoxGroup, area_no: str) -> bool:
        if self._is_big_plan_area_for_group(group, area_no):
            return True
        functions = self.problem.area_functions.get(area_no, set())
        return _area_flow(group.status) in functions


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
        cost = (
            self.config.existing_group_proximity_weight
            * self._normalized_existing_proximity(group, bay_key)
            / self._objective_scale("existing_group_proximity")
        )
        # The exact 20-ft opportunity loss is modeled jointly through
        # large-pair loss variables in the master problem. Do not add a second
        # per-column proxy cost here.
        return cost


    def _is_big_plan_area_for_group(self, group: SmallBoxGroup, area_no: str) -> bool:
        if area_no in self.problem.assigned_areas.get((group.voyage_id, group.status), set()):
            return True
        if self._is_export_voyage(group.voyage_id):
            return any(area_no in self.problem.assigned_areas.get((group.voyage_id, flow), set()) for flow in EXPORT_FLOWS)
        return False





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
            blocks = self._six_bay_blocks_for_area(area_no, keys)
            self.block_members_by_area[area_no] = blocks
            for block_id, members in blocks.items():
                self.block_bay_nos[block_id] = tuple(self.bays[bay_key].bay_no for bay_key in members)
                for bay_key in members:
                    self.block_by_bay[(area_no, bay_key)] = block_id
        self._prepare_large_segment_preservation_indexes()
        self._prepare_large_pair_capacity_indexes()
        heights_by_size: defaultdict[str, set[str]] = defaultdict(set)
        for group in self.groups:
            heights_by_size[group.size].add(group.height)
        for bay_key, bay in self.bays.items():
            is_edge = bay_key in self.area_edge_bays.get(bay.area_no, set())
            for size, heights in heights_by_size.items():
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

    def _prepare_large_pair_capacity_indexes(self) -> None:
        """Index physical 40/45-ft pairs for reservation and loss accounting."""
        self.large_pair_capacity.clear()
        self.large_pair_capacity_45.clear()
        self.large_pair_by_member.clear()
        for bay_key, bay in self.bays.items():
            partner_key = str(getattr(bay, "large_bay_partner_key", "") or "")
            if not partner_key or partner_key not in self.bays:
                continue
            pair = (bay_key, partner_key)
            capacity = max(
                int(bay.cap_by_size.get("40", 0) or 0),
                int(bay.cap_by_size.get("45", 0) or 0),
            )
            if capacity <= 0:
                continue
            self.large_pair_capacity[pair] = capacity
            self.large_pair_capacity_45[pair] = int(bay.cap_by_size.get("45", 0) or 0)
            self.large_pair_by_member[bay_key] = pair
            self.large_pair_by_member[partner_key] = pair

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








    def _bay_existing_size_modes(self, bay_key: str) -> set[str]:
        bay = self.bays.get(bay_key)
        if bay is None:
            return set()
        return {str(size) for size in getattr(bay, "existing_size_modes", set()) if str(size)}




    def _large_pair_capacity_loss(self, used_twenty_bays: set[str], area_no: str | None = None) -> int:
        return sum(
            int(capacity)
            for pair, capacity in self.large_pair_capacity.items()
            if (area_no is None or self.bays[pair[0]].area_no == area_no)
            if pair[0] in used_twenty_bays or pair[1] in used_twenty_bays
        )


    def _twenty_segment_loss_penalty(self) -> float:
        return (
            float(self.config.large_pair_capacity_weight)
            / self._objective_scale("large_pair_capacity")
        )


    def _prepare_quota(self) -> None:
        for (voyage_id, flow, area_no, big_size), qty in getattr(self.problem, "area_guidance_target", {}).items():
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
                    and row_size == big_size
                ):
                    self.quota_by_key[(voyage_id, flow, row.area_no, big_size)] += row.new_boxes

    def _area_weights(self, group: SmallBoxGroup) -> Counter[str]:
        weights: Counter[str] = Counter()
        big_size = self._big_plan_size(group.size)
        for (voyage_id, flow, area_no, size), qty in self.quota_by_key.items():
            if voyage_id == group.voyage_id and flow == group.status and size == big_size and qty > 0:
                weights[area_no] += qty
        return weights

    def _quota_key(self, group: SmallBoxGroup, area_no: str) -> tuple[str, str, str, str]:
        return group.voyage_id, group.status, area_no, self._big_plan_size(group.size)

    def _operational_group_key(self, group: SmallBoxGroup) -> tuple[str, ...]:
        rules = getattr(self.problem, "attribute_rules", None)
        attrs = tuple(getattr(rules, "group_attributes", ()) or MANDATORY_BAY_NO_MIX_ATTRS)
        return self._attribute_cluster_key(group, attrs)





    def _attribute_cluster_key(self, group: SmallBoxGroup, attrs: tuple[str, ...]) -> tuple[str, ...]:
        scope = str(group.voyage_id) if self._is_export_voyage(group.voyage_id) else "IMPORT"
        return (
            scope,
            f"flow={group.status}",
            *(f"{attr}={self._group_attr_value(group, attr)}" for attr in attrs),
        )





    def _existing_anchor_key(self, group: SmallBoxGroup) -> tuple[str, ...]:
        return self._operational_group_key(group)


    def _existing_group_bay_load_for_group(self, group: SmallBoxGroup, bay_key: str) -> int:
        bay = self.bays.get(bay_key)
        if bay is None:
            return 0
        return int(self.existing_group_bay_load.get(self._existing_anchor_key(group) + (bay.area_no, bay_key), 0))

    def _existing_same_group_bay_distance(self, group: SmallBoxGroup, bay_key: str) -> int | None:
        bay = self.bays.get(bay_key)
        if bay is None:
            return None
        anchor_bays = self.existing_group_area_bays.get(self._existing_anchor_key(group) + (bay.area_no,), set())
        if not anchor_bays:
            return None
        distances = [abs(self.bays[key].bay_order - bay.bay_order) for key in anchor_bays if key in self.bays]
        return min(distances) if distances else None

    def _normalized_existing_proximity(self, group: SmallBoxGroup, bay_key: str) -> float:
        """Return a [0, 1] cost relative to incumbent exact-group anchors.

        Reusing an incumbent bay costs zero. Within an anchored area, bay
        distance is divided by that area's full bay-order span. Selecting an
        area without an anchor costs one. Groups without any incumbent anchor
        are neutral and therefore contribute zero.
        """
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

    def _existing_group_bay_rank(self, group: SmallBoxGroup, bay_key: str) -> tuple[int, int, int]:
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
            "large_pair_capacity": float(self.config.large_pair_capacity_weight),
            "berth_distance": float(self.config.berth_distance_weight),
        }

    def _validate_objective_weights(self) -> None:
        weights = self._objective_weights()
        if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError(f"secondary objective weights must be finite and nonnegative: {weights}")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"secondary objective weights must sum to 1, got {total}: {weights}")

    def _anchored_group_demand(self) -> int:
        return sum(
            int(group.demand)
            for group in self.groups
            if self.existing_group_bays.get(self._existing_anchor_key(group))
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
        fallback = {
            "existing_group_proximity": self._anchored_group_demand(),
            "area_guidance_l1": 2 * self._guided_demand(),
            "large_pair_capacity": sum(self.large_pair_capacity.values()),
            "berth_distance": sum(group.demand for group in self.groups),
        }.get(key, 1.0)
        return max(1.0, float(fallback))

    def _prepare_objective_normalization(self) -> None:
        demand_by_group: Counter[tuple[str, ...]] = Counter()
        areas_by_group: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
        rows_by_group: defaultdict[tuple[str, ...], set[tuple[str, str]]] = defaultdict(set)
        for group in self.groups:
            demand_by_group[self._operational_group_key(group)] += int(group.demand)
        for col in self._columns:
            areas_by_group[col.group_key].add(col.area_no)
            for bay_key, row_no, qty in col.row_allocation:
                if qty > 0 and bay_key == col.bay_key:
                    rows_by_group[col.group_key].add((bay_key, row_no))
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
            "area_guidance_l1": float(max(1, 2 * self._guided_demand())),
            "large_pair_capacity": float(max(1, sum(self.large_pair_capacity.values()))),
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

    def _area_summary_big_plan_inheritance_stats(self, area_bay_rows: list[dict]) -> dict[str, float | int]:
        actual = self._area_summary_size_counter(area_bay_rows)
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

    def _area_summary_inheritance_energy_components(self, area_bay_rows: list[dict]) -> dict[str, float]:
        actual = self._area_summary_size_counter(area_bay_rows)
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

    def _area_summary_size_counter(self, area_bay_rows: list[dict]) -> Counter[tuple[str, str, str, str]]:
        actual: Counter[tuple[str, str, str, str]] = Counter()
        for row in area_bay_rows:
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
    def _planned_area_summary_by_source(area_bay_rows: list[dict]) -> dict[str, int]:
        out: Counter[str] = Counter()
        for row in area_bay_rows:
            document_boxes = int(row.get("document_boxes", 0) or 0)
            if document_boxes > 0:
                out["document"] += document_boxes
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

    def _area_bay_output_row(
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
        }
        for attr, value in sorted((attributes or {}).items()):
            if attr and attr not in row:
                row[attr] = value
        return row


    def _make_area_bay_rows_from_selected_columns(self, selected: Counter[int], plan_level: str = "medium") -> list[dict]:
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
                    self._area_bay_output_row(
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

