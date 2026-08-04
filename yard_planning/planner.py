from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Iterable

from .models import EXPORT_VOYAGE_ROW_NO_MIX_ATTR, Bay, ExportGroup, ProblemData

SIZE_ORDER = {"45": 0, "20": 1, "40": 2}
EXPORT_FLOWS = frozenset({"OF"})
MANDATORY_BAY_NO_MIX_ATTRS = ("IYC_CSZ_CSIZECD", "IYC_CHEIGHTCD")
SIZE_NO_MIX_ATTRS = frozenset({"IYC_CSZ_CSIZECD", "SIZE", "SIZE_MODE"})
HEIGHT_NO_MIX_ATTRS = frozenset({"IYC_CHEIGHTCD", "HEIGHT"})


class _GurobiModelAdapter:
    """API adapter around :class:`gurobipy.Model`.

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

    def addPricedVar(self, terms: Iterable[tuple[float, object]], **kwargs):
        """Add one variable with coefficients in existing master rows."""
        column = self._gp.Column()
        for coefficient, constraint in terms:
            if constraint is not None and abs(float(coefficient)) > 0.0:
                column.addTerms(float(coefficient), constraint)
        return self._model.addVar(column=column, **kwargs)

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


@dataclass(frozen=True)
class PackingPattern:
    """One complete yard-allocation decision for one export group."""

    pattern_id: str
    group_id: str
    placements: tuple[PlacementColumn, ...]
    unplaced: int
    phase2_cost: float


@dataclass
class ColumnGenerationConfig:
    max_iterations: int = 60
    min_columns_per_group_per_iteration: int = 1
    max_columns_per_group_per_iteration: int = 12
    adaptive_pricing_fraction: float = 0.75
    heuristic_pricing_variants: int = 12
    dual_stabilization_alpha: float = 0.65
    exact_pricing_time_limit: float = 60.0
    pattern_lp_gap_tolerance: float = 0.005
    raw_dual_check_interval: int = 5
    reduced_cost_tolerance: float = 1e-7
    total_time_limit: float = 60.0
    mip_time_limit: float = 30.0
    mip_gap: float = 0.01
    lp_method: int = 1
    solver_seed: int = 0
    solver_threads: int = 0
    verbose: bool = True
    # Stage-2 policy weights. Every component is first mapped to a natural
    # dimensionless scale. The baseline is calibrated against realized
    # component contributions and reachable-anchor coverage, while retaining
    # the declared policy hierarchy.
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
    unplaced_rows: list[dict] = field(default_factory=list)
    import_reservation_rows: list[dict] = field(default_factory=list)
    columns: list[PlacementColumn] = field(default_factory=list)


class ColumnGenerationPlanner:
    """Declared-export row assignment with stabilized packing-pattern pricing.

    A column is a complete feasible row-level allocation pattern for one
    export group, possibly retaining some of that group's demand as unplaced.
    The master uses
    only declared, not-yet-gated-in export containers. Incoming imports are
    represented by anonymous size-compatible bay-capacity reservations: their
    flow/size totals are fixed, while their area distribution may deviate from
    the upstream big-plan reference. They never become detailed row demand.
    """

    def __init__(self, problem: ProblemData, config: ColumnGenerationConfig | None = None) -> None:
        self.problem = problem
        self.config = config or ColumnGenerationConfig()
        self.demand_stats: dict[str, int | str] = {}
        self.export_voyages = self._infer_export_voyages(problem)
        self.groups = sorted(self._build_planning_groups(), key=self._group_sort_key)
        self.groups_by_id = {group.group_id: group for group in self.groups}
        self.bays = problem.bays
        self.attribute_rules = problem.attribute_rules
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
        self.import_total_by_flow_size: Counter[tuple[str, str]] = Counter()
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
        self._patterns: list[PackingPattern] = []
        self._pattern_keys: set[tuple] = set()
        self._artificial_pattern_by_group: dict[str, int] = {}
        self._pricing_calls_by_phase: Counter[str] = Counter()
        self._columns: list[PlacementColumn] = []
        self._column_keys: set[tuple[str, str, int, tuple[tuple[str, str, int], ...]]] = set()
        self._candidate_cache: dict[str, list[tuple[str, int, float]]] = {}
        self._objective_scales: dict[str, float] = {}
        self._berth_distance_bounds: dict[str, tuple[float, float]] = {}
        self._initial_unplaced_start: Counter[str] = Counter()
        self._base_feasible_placement_count = 0
        self._master_bay_capacity_keys: set[str] = set()
        self._master_bay_size_keys: set[tuple[str, str]] = set()
        self._master_row_capacity_keys: set[tuple[str, str]] = set()
        self._master_row_size_keys: set[tuple[str, str, str]] = set()
        self._master_stack_keys: set[tuple[str, str, str]] = set()
        self._master_stack_sample_group: dict[tuple[str, str, str], str] = {}
        self._master_area_guidance_keys: set[tuple[str, str, str, str]] = set()
        self._master_group_area_keys: set[tuple[tuple[str, ...], str]] = set()
        self._master_group_row_keys: set[tuple[tuple[str, ...], str, str]] = set()
        self._master_operational_group_keys: set[tuple[str, ...]] = set()
        self._master_bay_attr_choice_keys: set[tuple[str, str, str, str]] = set()
        self._master_row_attr_choice_keys: set[tuple[str, str, str, str, str]] = set()
        self._master_bay_attr_big_m: dict[tuple[str, str, str, str], int] = {}
        self._master_row_attr_big_m: dict[tuple[str, str, str, str, str], int] = {}
        self._master_group_area_big_m: dict[tuple[tuple[str, ...], str], int] = {}
        self._master_group_row_big_m: dict[tuple[tuple[str, ...], str, str], int] = {}
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
        minimum = int(self.config.min_columns_per_group_per_iteration)
        maximum = int(self.config.max_columns_per_group_per_iteration)
        fraction = float(self.config.adaptive_pricing_fraction)
        if minimum <= 0 or maximum < minimum:
            raise ValueError(
                "adaptive pricing batch bounds must satisfy "
                f"0 < minimum <= maximum, got minimum={minimum}, maximum={maximum}"
            )
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError(
                "adaptive_pricing_fraction must be in (0, 1], "
                f"got {fraction}"
            )
        if int(self.config.max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        variants = int(self.config.heuristic_pricing_variants)
        if variants <= 0:
            raise ValueError("heuristic_pricing_variants must be positive")
        alpha = float(self.config.dual_stabilization_alpha)
        if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError(
                "dual_stabilization_alpha must be in (0, 1], "
                f"got {alpha}"
            )
        if float(self.config.exact_pricing_time_limit) <= 0.0:
            raise ValueError("exact_pricing_time_limit must be positive")
        lp_gap = float(self.config.pattern_lp_gap_tolerance)
        if not math.isfinite(lp_gap) or lp_gap < 0.0:
            raise ValueError("pattern_lp_gap_tolerance must be finite and nonnegative")
        if int(self.config.raw_dual_check_interval) <= 0:
            raise ValueError("raw_dual_check_interval must be positive")

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
    def solve(self) -> ColumnGenerationResult:
        self._prepare_master_index_sets()
        self._prepare_objective_normalization()
        self._initialize_pattern_generation()
        self._initial_unplaced_start = Counter(
            {group.group_id: int(group.demand) for group in self.groups}
        )
        seed_stats = {
            "initialization": "one_all-unplaced_artificial_pattern_per_group",
            "initial_generated_patterns": len(self._patterns),
            "initial_unplaced_boxes": int(sum(self._initial_unplaced_start.values())),
        }
        diagnostics: dict = {
            "algorithm": "stabilized_export_group_packing_pattern_column_generation",
            "model_scope": "export_declared_containers_row_allocation",
            "detailed_allocation_direction": "export_only",
            "target_voyages": self.problem.target_voyages,
            "attribute_rules": self.attribute_rules.as_dict(),
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
            "import_capacity_reservation": {
                "source_quantity_field": "new_qty",
                "role": "anonymous_size_compatible_capacity_only",
                "area_policy": "weighted_l1_deviation_from_big_plan_reference",
                "constraint_scope": ["area_function", "bay_size", "physical_capacity"],
                "excluded_constraints": ["bay_no_mix", "row_no_mix", "container_group_attributes"],
                "import_boxes": int(sum(self.import_area_size_reference.values())),
                "accepted_big_plan_sizes": ["20", "40"],
                "reference_by_flow_area_size": {
                    f"{flow}|{area}|{size}": int(qty)
                    for (flow, area, size), qty in sorted(self.import_area_size_reference.items())
                },
                "required_by_flow_size": {
                    f"{flow}|{size}": int(qty)
                    for (flow, size), qty in sorted(self.import_total_by_flow_size.items())
                },
                "candidate_bay_capacity_upper_bound_sum_by_flow_size": {
                    f"{flow}|{size}": int(sum(capacity for _bay_key, capacity in candidates))
                    for (flow, size), candidates in sorted(self.import_reservation_candidates.items())
                },
            },
            "initial_pattern_count": len(self._patterns),
            "base_feasible_placement_count": self._base_feasible_placement_count,
            "pattern_universe": "implicit_exponential",
            "exact_pattern_pricing": True,
            "pricing_configuration": {
                "persistent_restricted_master": True,
                "lp_method": int(self.config.lp_method),
                "solver_seed": int(self.config.solver_seed),
                "solver_threads": int(self.config.solver_threads),
                "decomposition": "one_complete_row_layout_pattern_per_export_group",
                "heuristic_pricing_variants": int(
                    self.config.heuristic_pricing_variants
                ),
                "dual_stabilization_alpha": float(
                    self.config.dual_stabilization_alpha
                ),
                "exact_pricing_time_limit": float(
                    self.config.exact_pricing_time_limit
                ),
                "pattern_lp_gap_tolerance": float(
                    self.config.pattern_lp_gap_tolerance
                ),
                "raw_dual_check_interval": int(
                    self.config.raw_dual_check_interval
                ),
                "adaptive_batch": {
                    "minimum_per_group": int(
                        self.config.min_columns_per_group_per_iteration
                    ),
                    "maximum_per_group": int(
                        self.config.max_columns_per_group_per_iteration
                    ),
                    "negative_candidate_fraction": float(
                        self.config.adaptive_pricing_fraction
                    ),
                },
            },
            "restricted_master_contains_generated_patterns_only": True,
            "branch_and_price": False,
            **seed_stats,
            "pricing_iterations": [],
            "gurobi_available": True,
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
                "reachable_group_count": len(self.reachable_anchor_group_ids),
                "reachable_group_demand": self._anchored_group_demand(),
                "unreachable_anchor_groups": sorted(
                    group.group_id
                    for group in self.groups
                    if self.existing_group_bays.get(self._existing_anchor_key(group))
                    and group.group_id not in self.reachable_anchor_group_ids
                ),
                "unreachable_anchor_policy": "neutral_excluded_from_objective_scale",
            },
            "objective_coefficients": {
                "phase2_unplaced_coefficient": 0.0,
                "area_guidance_l1_unit": self._area_guidance_penalty(),
                "area_activation_unit": self._area_activation_penalty(),
                "row_activation_unit": self._row_activation_penalty(),
            },
            "big_m_tightening": self._big_m_diagnostics(),
        }

        selected, unplaced, master_stats = self._solve_by_column_generation()
        diagnostics.update(master_stats)

        import_reservation_rows = self._make_import_reservation_rows()
        diagnostics["import_capacity_reservation"].update(
            self._import_reservation_diagnostics()

        )
        objective_components = self._selected_objective_components(selected, unplaced)
        diagnostics["final_secondary_objective"] = objective_components["weighted_total"]
        diagnostics["final_secondary_objective_components"] = objective_components
        diagnostics["independent_solution_validation"] = self._validate_final_solution(
            selected, unplaced
        )

        export_rows = self._make_export_rows(selected)

        bay_summary_rows = self._make_bay_summary_rows(selected)
        unplaced_rows = self._unplaced_group_details(unplaced)
        consistency_stats = self._row_area_summary_consistency_stats(export_rows, bay_summary_rows)
        bay_consistency_stats = self._row_bay_summary_consistency_stats(export_rows, bay_summary_rows)
        operational_group_dispersion = self._operational_group_dispersion_stats(export_rows)
        diagnostics.update(
            {
                "integer_location_pool_size": len(self._columns),
                "selected_location_count": sum(1 for qty in selected.values() if qty > 0),
                "summary_granularity": "bay",
                "export_row_count": len(export_rows),
                "bay_summary_row_count": len(bay_summary_rows),
                "planned_export_boxes": sum(int(row["planned_boxes"]) for row in export_rows),
                "operational_group_dispersion": operational_group_dispersion,
                "area_summary_big_plan_inheritance": self._area_summary_big_plan_inheritance_stats(bay_summary_rows),
                "final_area_summary_inheritance_energy_components": self._area_summary_inheritance_energy_components(bay_summary_rows),
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
            bay_summary_rows=bay_summary_rows,
            export_rows=export_rows,
            diagnostics=diagnostics,
            unplaced_rows=unplaced_rows,
            import_reservation_rows=import_reservation_rows,
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
            missing = int(unplaced.get(group.group_id, 0))
            if assigned + missing != int(group.demand):
                errors.append(
                    f"demand balance failed for {group.group_id}: "
                    f"assigned={assigned}, unplaced={missing}, demand={group.demand}"
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
            "unplaced_boxes_checked": int(sum(unplaced.values())),
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
            result[area_no] = {
                "physical_slot_capacity": int(physical),
                "export_slot_use": int(export_slots),
                "import_reserved_slot_use": int(import_slots),
                "residual_slot_units": int(physical - export_slots - import_slots),
                "import_reference_boxes": int(reference_boxes),
                "import_reserved_boxes": int(import_area_boxes[area_no]),
            }
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
        keys = set(actual) | set(self.import_area_size_reference)
        l1 = sum(
            abs(int(actual.get(key, 0)) - int(self.import_area_size_reference.get(key, 0)))
            for key in keys
        )
        return {
            "reserved_boxes": int(sum(actual.values())),
            "reserved_by_flow_area_size": {
                f"{flow}|{area}|{size}": int(qty)
                for (flow, area, size), qty in sorted(actual.items())
                if qty > 0
            },
            "area_l1_deviation": int(l1),
            "boxes_shifted_between_areas": int(l1 // 2),
        }

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
        import_keys = set(self.import_area_size_reference) | set(import_actual_by_area)
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

    def _selected_solution_energy(self, selected: Counter[int], unplaced: Counter[str]) -> float:
        # Unplaced quantity is handled lexicographically and is not blended
        # into the normalized secondary objective.
        return float(self._selected_objective_components(selected, unplaced)["weighted_total"])

    def _unplaced_objective_for_group(self, group: ExportGroup, objective_mode: str) -> float:
        # Stage 2 fixes the stage-1 optimum with an equality, so an additional
        # penalty is redundant and would only worsen objective scaling.
        return 1.0 if objective_mode == "min_unplaced" else 0.0

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

    def _solve_location_pool_integer_master(
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

        stage1, stage1_vars, _ = self._build_location_pool_integer_master(
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
        stage1_import_reservation = self._gurobi_import_reservation_values(stage1, stage1_vars)
        self._final_import_reservation = stage1_import_reservation
        optimum_unplaced = int(sum(stage1_unplaced.values()))
        stats["lexicographic_stage1_unplaced_boxes"] = optimum_unplaced
        stats["lexicographic_stage1_bound"] = self._gurobi_dual_bound(stage1)
        self._free_gurobi_model(stage1)

        stage2, stage2_vars, _ = self._build_location_pool_integer_master(
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
        self._final_import_reservation = self._gurobi_import_reservation_values(
            stage2, stage2_vars
        )
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

    def _build_location_pool_integer_master(
        self,
        Model,
        quicksum,
        relax: bool,
        objective_mode: str = "full",
        fixed_unplaced_total: float | None = None,
    ):
        model = Model("yard_export_row_column_generation_gurobi")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(model, "Threads", int(self.config.solver_threads))
        if relax:
            self._set_gurobi_param(model, "Method", int(self.config.lp_method))
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
        import_reserve = {
            (flow, size, bay_key): model.addVar(
                lb=0.0,
                ub=float(capacity),
                vtype="C" if relax else "I",
                obj=0.0,
                name=f"import_reserve_{flow}_{size}_{self._key_name((bay_key,))}",
            )
            for (flow, size), candidates in sorted(self.import_reservation_candidates.items())
            for bay_key, capacity in candidates
        }

        group_cols: defaultdict[str, list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_capacity_cols: defaultdict[str, list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_size_capacity_cols: defaultdict[tuple[str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_port_size_cols: defaultdict[tuple[str, str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        row_capacity_cols: defaultdict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        row_size_capacity_cols: defaultdict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
        row_attr_choice_cols: defaultdict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
        bay_attr_choice_cols: defaultdict[tuple[str, str, str, str], list[int]] = defaultdict(list)
        area_size_cols: defaultdict[tuple[str, str, str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        group_area_cols: defaultdict[tuple[tuple[str, ...], str], list[int]] = defaultdict(list)
        group_row_cols: defaultdict[tuple[tuple[str, ...], str, str], list[int]] = defaultdict(list)
        import_by_bay: defaultdict[str, list] = defaultdict(list)
        import_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        import_by_flow_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        import_by_flow_area_size: defaultdict[tuple[str, str, str], list] = defaultdict(list)
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

        for key, var in import_reserve.items():
            flow, size, bay_key = key
            area_no = self.bays[bay_key].area_no
            for footprint_key in self._placement_footprint_keys(bay_key, size):
                import_by_bay[footprint_key].append(var)
            import_by_bay_size[(bay_key, size)].append(var)
            import_by_flow_size[(flow, size)].append(var)
            import_by_flow_area_size[(flow, area_no, size)].append(var)

        group_cover = {}
        for group in self.groups:
            expr = quicksum(col.quantity * columns[idx] for idx, col in group_cols.get(group.group_id, []))
            group_cover[group.group_id] = model.addCons(expr + unplaced[group.group_id] == group.demand, name=f"cover_{group.group_id}")

        bay_capacity_limit = {}
        for bay_key in sorted(self._master_bay_capacity_keys):
            items = bay_capacity_cols.get(bay_key, [])
            bay_capacity_limit[bay_key] = model.addCons(
                quicksum(col.quantity * columns[idx] for idx, col in items)
                + quicksum(import_by_bay.get(bay_key, []))
                <= self.bays[bay_key].physical_capacity,
                name=f"bay_cap_{bay_key}",
            )
        bay_size_limit = {}
        for key in sorted(self._master_bay_size_keys):
            items = bay_size_capacity_cols.get(key, [])
            bay_key, size = key
            bay_size_limit[key] = model.addCons(
                quicksum(col.quantity * columns[idx] for idx, col in items)
                + quicksum(import_by_bay_size.get(key, []))
                <= self.bays[bay_key].cap_by_size.get(size, 0),
                name=f"bay_size_{bay_key}_{size}",
            )
        row_capacity_limit = {}
        for key in sorted(self._master_row_capacity_keys):
            items = row_capacity_cols.get(key, [])
            bay_key, row_no = key
            cap = int(self.bays[bay_key].row_physical_capacity.get(row_no, self.bays[bay_key].physical_capacity))
            row_capacity_limit[key] = model.addCons(
                quicksum(qty * columns[idx] for idx, qty in items) <= cap,
                name=f"row_cap_{bay_key}_{row_no}",
            )
        row_size_limit = {}
        for key in sorted(self._master_row_size_keys):
            items = row_size_capacity_cols.get(key, [])
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
        for key in sorted(self._master_stack_keys):
            items = bay_port_size_cols.get(key, [])
            bay_key, port, size = key
            sample_group = self.groups_by_id.get(self._master_stack_sample_group.get(key, ""))
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
        import_total_balance = {}
        for key, required in sorted(self.import_total_by_flow_size.items()):
            candidates = import_by_flow_size.get(key, [])
            if not candidates:
                raise ValueError(
                    "import capacity reservation has no function- and size-compatible bay: "
                    f"flow={key[0]}, size={key[1]}, required={required}"
                )
            import_total_balance[key] = model.addCons(
                quicksum(candidates) == int(required),
                name=f"import_total_{key[0]}_{key[1]}",
            )
        import_reference_balance = self._add_import_reference_deviation(
            quicksum,
            model,
            import_by_flow_area_size,
            objective_mode=objective_mode,
        )

        # Big-plan allocations are soft inheritance targets. Hard upper bounds
        # are intentionally omitted so detailed declared boxes can recover from
        # stale or physically incompatible upstream allocations.
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
        bay_compatibility_constraints = self._add_location_pool_bay_compatibility_constraints(
            quicksum, model, columns, bay_attr_choice_cols, relax=relax
        )
        row_compatibility_constraints = self._add_location_pool_row_compatibility_constraints(
            quicksum, model, columns, row_attr_choice_cols, relax=relax
        )
        return model, {
            "column": columns,
            "unplaced": unplaced,
            "import_reserve": import_reserve,
        }, {
            "group_cover": group_cover,
            "bay_capacity_limit": bay_capacity_limit,
            "bay_size_limit": bay_size_limit,
            "row_capacity_limit": row_capacity_limit,
            "row_size_limit": row_size_limit,
            "import_total_balance": import_total_balance,
            "import_reference_balance": import_reference_balance,
            "bay_port_stack_link": bay_port_stack_link,
            "bay_port_stack_limit": bay_port_stack_limit,
            "bay_stack_total_limit": bay_stack_total_limit,
            "lexicographic_unplaced_limit": lexicographic_unplaced_limit,
            **bay_compatibility_constraints,
            **row_compatibility_constraints,
            **relaxed_objective_constraints,
        }

    def _add_import_reference_deviation(
        self,
        quicksum,
        model,
        import_by_flow_area_size: dict[tuple[str, str, str], list],
        objective_mode: str,
    ) -> dict[tuple[str, str, str], object]:
        """Penalize the minimum area adjustment of anonymous import reserve."""
        keys = set(self.import_area_size_reference) | set(import_by_flow_area_size)
        balance = {}
        penalty = 0.0 if objective_mode == "min_unplaced" else self._area_guidance_penalty()
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
            balance[(flow, area_no, size)] = model.addCons(
                actual - target == pos - neg,
                name=f"import_guide_balance_{flow}_{area_no}_{size}",
            )
        return balance

    def _add_relaxed_master_objectives(
        self,
        quicksum,
        model,
        columns,
        area_size_cols,
        group_area_cols,
        group_row_cols,
    ) -> dict[str, dict]:
        area_guidance_balance = {}
        for key in sorted(self._master_area_guidance_keys):
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
        for key in sorted(self._master_group_area_keys):
            indices = group_area_cols.get(key, [])
            use = model.addVar(lb=0.0, ub=1.0, obj=self._area_activation_penalty())
            group_key, _area_no = key
            fixed_use_constraints[("group_area",) + key] = model.addCons(
                quicksum(self._columns[idx].quantity * columns[idx] for idx in indices)
                <= self._master_group_area_big_m[key] * use
            )
            area_use_by_group[group_key].append(use)
            column_indices_by_group[group_key].update(indices)
        for key in sorted(self._master_group_row_keys):
            indices = group_row_cols.get(key, [])
            use = model.addVar(lb=0.0, ub=1.0, obj=self._row_activation_penalty())
            group_key, _bay_key, _row_no = key
            fixed_use_constraints[("group_row",) + key] = model.addCons(
                quicksum(columns[idx] for idx in indices)
                <= self._master_group_row_big_m[key] * use
            )
            row_use_by_group[group_key].append(use)
            column_indices_by_group[group_key].update(indices)
        for group_key in sorted(self._master_operational_group_keys):
            indices = column_indices_by_group.get(group_key, set())
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
        for key in sorted(self._master_area_guidance_keys):
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
        for group_key, area_no in sorted(self._master_group_area_keys):
            indices = group_area_cols.get((group_key, area_no), [])
            use = model.addVar(vtype="B", obj=self._area_activation_penalty(), name=f"use_ga_{self._key_name(group_key)}_{area_no}")
            model.addCons(
                quicksum(self._columns[idx].quantity * columns[idx] for idx in indices)
                <= self._master_group_area_big_m[(group_key, area_no)] * use
            )
            area_use_by_group[group_key].append(use)
            column_indices_by_group[group_key].update(indices)
        for group_key, bay_key, row_no in sorted(self._master_group_row_keys):
            indices = group_row_cols.get((group_key, bay_key, row_no), [])
            use = model.addVar(
                vtype="B",
                obj=self._row_activation_penalty(),
                name=f"use_gr_{self._key_name(group_key)}_{bay_key}_{row_no}",
            )
            model.addCons(
                quicksum(columns[idx] for idx in indices)
                <= self._master_group_row_big_m[(group_key, bay_key, row_no)] * use
            )
            row_use_by_group[group_key].append(use)
            column_indices_by_group[group_key].update(indices)
        for group_key in sorted(self._master_operational_group_keys):
            indices = column_indices_by_group.get(group_key, set())
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
    def _add_location_pool_bay_compatibility_constraints(
        self,
        quicksum,
        model,
        columns,
        bay_attr_choice_cols: dict[tuple[str, str, str, str], list[int]],
        relax: bool,
    ) -> dict[str, dict]:
        vtype = "C" if relax else "B"
        use_by_bay_attr: defaultdict[tuple[str, str, str], list] = defaultdict(list)
        link_constraints = {}
        one_constraints = {}
        for bay_key, attr, scope, value in sorted(self._master_bay_attr_choice_keys):
            key = (bay_key, attr, scope, value)
            indices = bay_attr_choice_cols.get(key, [])
            scope_name = scope or "GLOBAL"
            use = model.addVar(lb=0.0, ub=1.0, vtype=vtype, name=f"bay_use_{attr}_{scope_name}_{bay_key}_{value}")
            link_constraints[key] = model.addCons(
                quicksum(columns[idx] for idx in indices)
                <= self._master_bay_attr_big_m[key] * use
            )
            use_by_bay_attr[(bay_key, attr, scope)].append(use)
        for (bay_key, attr, scope), uses in use_by_bay_attr.items():
            scope_name = scope or "GLOBAL"
            one_constraints[(bay_key, attr, scope)] = model.addCons(
                quicksum(uses) <= 1,
                name=f"bay_one_{attr}_{scope_name}_{bay_key}",
            )
        return {
            "bay_attr_link": link_constraints,
            "bay_attr_one": one_constraints,
        }

    def _add_location_pool_row_compatibility_constraints(
        self,
        quicksum,
        model,
        columns,
        row_attr_choice_cols: dict[tuple[str, str, str, str, str], list[int]],
        relax: bool,
    ) -> dict[str, dict]:
        vtype = "C" if relax else "B"
        use_by_row_attr: defaultdict[tuple[str, str, str, str], list] = defaultdict(list)
        link_constraints = {}
        one_constraints = {}
        for bay_key, row_no, attr, scope, value in sorted(self._master_row_attr_choice_keys):
            key = (bay_key, row_no, attr, scope, value)
            indices = row_attr_choice_cols.get(key, [])
            scope_name = scope or "GLOBAL"
            use = model.addVar(lb=0.0, ub=1.0, vtype=vtype, name=f"row_use_{attr}_{scope_name}_{bay_key}_{row_no}_{value}")
            link_constraints[key] = model.addCons(
                quicksum(columns[idx] for idx in indices)
                <= self._master_row_attr_big_m[key] * use
            )
            use_by_row_attr[(bay_key, row_no, attr, scope)].append(use)
        for (bay_key, row_no, attr, scope), uses in use_by_row_attr.items():
            scope_name = scope or "GLOBAL"
            one_constraints[(bay_key, row_no, attr, scope)] = model.addCons(
                quicksum(uses) <= 1,
                name=f"row_one_{attr}_{scope_name}_{bay_key}_{row_no}",
            )
        return {
            "row_attr_link": link_constraints,
            "row_attr_one": one_constraints,
        }

    def _initialize_pattern_generation(self) -> None:
        """Seed the master with one all-unplaced pattern for every group."""
        self._patterns.clear()
        self._pattern_keys.clear()
        self._artificial_pattern_by_group.clear()
        self._pricing_calls_by_phase.clear()
        self._columns.clear()
        self._column_keys.clear()
        for group in self.groups:
            idx = self._append_pattern(
                self._make_pattern(group, (), int(group.demand))
            )
            self._artificial_pattern_by_group[group.group_id] = idx

    def _make_pattern(
        self,
        group: ExportGroup,
        placements: Iterable[PlacementColumn],
        unplaced: int,
    ) -> PackingPattern:
        ordered = tuple(
            sorted(
                (col for col in placements if int(col.quantity) > 0),
                key=lambda col: (
                    col.area_no,
                    self.bays[col.bay_key].bay_order,
                    col.row_allocation,
                ),
            )
        )
        placed = sum(int(col.quantity) for col in ordered)
        if placed + int(unplaced) != int(group.demand):
            raise ValueError(
                "packing pattern must partition one group demand exactly: "
                f"group={group.group_id}, placed={placed}, unplaced={unplaced}, "
                f"demand={group.demand}"
            )
        used_areas = {col.area_no for col in ordered}
        used_rows = {
            (col.bay_key, row_no)
            for col in ordered
            for bay_key, row_no, qty in col.row_allocation
            if int(qty) > 0 and bay_key == col.bay_key
        }
        phase2_cost = sum(
            float(col.intrinsic_cost) * int(col.quantity) for col in ordered
        )
        if ordered:
            phase2_cost += self._area_activation_penalty() * max(
                0, len(used_areas) - 1
            )
            phase2_cost += self._row_activation_penalty() * max(
                0, len(used_rows) - 1
            )
        return PackingPattern(
            pattern_id="",
            group_id=group.group_id,
            placements=ordered,
            unplaced=int(unplaced),
            phase2_cost=float(phase2_cost),
        )

    @staticmethod
    def _pattern_identity(pattern: PackingPattern) -> tuple:
        return (
            pattern.group_id,
            int(pattern.unplaced),
            tuple(
                (col.bay_key, int(col.quantity), col.row_allocation)
                for col in pattern.placements
            ),
        )

    def _append_pattern(self, candidate: PackingPattern) -> int:
        key = self._pattern_identity(candidate)
        if key in self._pattern_keys:
            raise ValueError(f"duplicate packing pattern: {key}")
        pattern = replace(
            candidate,
            pattern_id=f"P{len(self._patterns) + 1:07d}",
        )
        idx = len(self._patterns)
        self._patterns.append(pattern)
        self._pattern_keys.add(key)
        return idx

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

    def _pattern_master_coefficients(
        self,
        pattern: PackingPattern,
    ) -> dict[str, Counter[object]]:
        out: dict[str, Counter[object]] = defaultdict(Counter)
        out["group_convexity"][pattern.group_id] = 1
        out["fixed_unplaced_total"]["total"] = int(pattern.unplaced)
        for col in pattern.placements:
            for section, values in self._placement_master_coefficients(col).items():
                out[section].update(values)
        return out

    def _adaptive_pricing_batch_size(self, negative_count: int) -> int:
        count = max(0, int(negative_count))
        if count == 0:
            return 0
        proportional = int(
            math.ceil(count * float(self.config.adaptive_pricing_fraction))
        )
        target = max(
            int(self.config.min_columns_per_group_per_iteration), proportional
        )
        return min(
            count,
            int(self.config.max_columns_per_group_per_iteration),
            target,
        )

    @staticmethod
    def _dual_value(
        duals: dict[tuple[str, object], float],
        section: str,
        key: object,
    ) -> float:
        return float(duals.get((section, key), 0.0))

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
                    duals[(section, key)] = float(model.getDualsolLinear(row))
        return duals

    def _stabilized_duals(
        self,
        raw: dict[tuple[str, object], float],
        center: dict[tuple[str, object], float] | None,
    ) -> dict[tuple[str, object], float]:
        if center is None:
            return dict(raw)
        alpha = float(self.config.dual_stabilization_alpha)
        keys = set(raw) | set(center)
        return {
            key: alpha * float(raw.get(key, 0.0))
            + (1.0 - alpha) * float(center.get(key, 0.0))
            for key in keys
        }

    def _pattern_reduced_cost(
        self,
        pattern: PackingPattern,
        duals: dict[tuple[str, object], float],
        objective_mode: str,
    ) -> float:
        reduced = (
            float(pattern.unplaced)
            if objective_mode == "min_unplaced"
            else float(pattern.phase2_cost)
        )
        for section, values in self._pattern_master_coefficients(pattern).items():
            for key, coefficient in values.items():
                reduced -= float(coefficient) * self._dual_value(
                    duals, section, key
                )
        return float(reduced)

    def _placement_reduced_unit_cost(
        self,
        col: PlacementColumn,
        duals: dict[tuple[str, object], float],
        objective_mode: str,
    ) -> float:
        reduced = (
            0.0 if objective_mode == "min_unplaced" else float(col.intrinsic_cost)
        )
        unit = replace(
            col,
            quantity=1,
            row_allocation=tuple(
                (bay_key, row_no, 1)
                for bay_key, row_no, _qty in col.row_allocation
            ),
        )
        for section, values in self._placement_master_coefficients(unit).items():
            for key, coefficient in values.items():
                reduced -= float(coefficient) * self._dual_value(
                    duals, section, key
                )
        return float(reduced)

    def _price_packing_patterns(
        self,
        raw_duals: dict[tuple[str, object], float],
        stabilized_duals: dict[tuple[str, object], float],
        phase_name: str,
        objective_mode: str,
    ) -> tuple[dict, list[int]]:
        """Use cheap stabilized pricing first and exact MIP pricing for proof."""
        tolerance = max(1e-12, float(self.config.reduced_cost_tolerance))
        call_number = int(self._pricing_calls_by_phase[phase_name])
        self._pricing_calls_by_phase[phase_name] += 1
        use_heuristic = call_number < 3
        heuristic_by_group: dict[str, list[tuple[float, PackingPattern]]] = {
            group.group_id: [] for group in self.groups
        }
        for group in self.groups if use_heuristic else ():
            candidates: list[tuple[float, PackingPattern]] = []
            for pattern in self._heuristic_group_patterns(
                group, stabilized_duals, objective_mode
            ):
                if self._pattern_identity(pattern) in self._pattern_keys:
                    continue
                reduced = self._pattern_reduced_cost(
                    pattern, raw_duals, objective_mode
                )
                if reduced < -tolerance:
                    candidates.append((reduced, pattern))
            candidates.sort(
                key=lambda item: (item[0], self._pattern_identity(item[1]))
            )
            heuristic_by_group[group.group_id] = candidates

        heuristic_negative_total = sum(
            len(values) for values in heuristic_by_group.values()
        )
        # Heuristic pricing is used only to populate the first RMPs cheaply.
        # Later iterations solve the exact pricing MIPs with stabilized duals;
        # raw duals are used once more whenever no entering pattern is found,
        # which is the only test accepted as the LP optimality certificate.
        run_exact_search = (
            not use_heuristic
            or heuristic_negative_total <= 2 * len(self.groups)
        )
        force_raw_check = (
            not use_heuristic
            and (call_number - 3) % int(self.config.raw_dual_check_interval)
            == int(self.config.raw_dual_check_interval) - 1
        )
        candidate_by_group = heuristic_by_group
        minimum_by_group: dict[str, float] = {}
        exact_seconds = 0.0
        exact_performed = False

        def merge_exact_pool(
            pricing_duals: dict[tuple[str, object], float],
        ) -> dict[str, float]:
            nonlocal exact_seconds
            minima: dict[str, float] = {}
            for group in self.groups:
                started = perf_counter()
                patterns, minimum = self._solve_exact_group_pricing(
                    group, pricing_duals, objective_mode
                )
                exact_seconds += perf_counter() - started
                minima[group.group_id] = minimum
                candidates = list(candidate_by_group.get(group.group_id, []))
                seen_identities = {
                    self._pattern_identity(pattern)
                    for _reduced, pattern in candidates
                }
                for pattern in patterns:
                    identity = self._pattern_identity(pattern)
                    if identity in self._pattern_keys or identity in seen_identities:
                        continue
                    reduced = self._pattern_reduced_cost(
                        pattern, raw_duals, objective_mode
                    )
                    if reduced < -tolerance:
                        candidates.append((reduced, pattern))
                        seen_identities.add(identity)
                candidate_by_group[group.group_id] = sorted(
                    candidates,
                    key=lambda item: (item[0], self._pattern_identity(item[1])),
                )
            return minima

        if run_exact_search:
            candidate_by_group = {
                group_id: list(values)
                for group_id, values in heuristic_by_group.items()
            }
            stabilized_minima = merge_exact_pool(stabilized_duals)
            if force_raw_check or not any(candidate_by_group.values()):
                minimum_by_group = merge_exact_pool(raw_duals)
                exact_performed = True
            elif stabilized_duals == raw_duals:
                minimum_by_group = stabilized_minima

        entering: list[tuple[float, PackingPattern]] = []
        negative_counts: dict[str, int] = {}
        selected_batch: dict[str, int] = {}
        for group in self.groups:
            candidates = candidate_by_group.get(group.group_id, [])
            batch = self._adaptive_pricing_batch_size(len(candidates))
            negative_counts[group.group_id] = len(candidates)
            selected_batch[group.group_id] = batch
            entering.extend(candidates[:batch])
        entering.sort(
            key=lambda item: (item[0], self._pattern_identity(item[1]))
        )
        new_indices = [
            self._append_pattern(pattern) for _reduced, pattern in entering
        ]
        return {
            "new_patterns": len(new_indices),
            "pricing_mode": (
                "raw_dual_exact_group_pattern_mip"
                if exact_performed
                else "stabilized_exact_group_pattern_mip"
                if run_exact_search
                else "stabilized_constructive_pattern_pool"
            ),
            "exact_pricing_performed": exact_performed,
            "exact_pricing": exact_performed,
            "dual_stabilization": True,
            "adaptive_batch": True,
            "negative_candidates_by_group": negative_counts,
            "selected_batch_by_group": selected_batch,
            "generated_pattern_count": len(self._patterns),
            "minimum_reduced_cost": min(
                minimum_by_group.values(), default=None
            ),
            "minimum_reduced_cost_by_group": {
                key: value for key, value in sorted(minimum_by_group.items())
            },
            "valid_lower_bound_correction": (
                sum(min(0.0, value) for value in minimum_by_group.values())
                if exact_performed
                else None
            ),
            "exact_pricing_elapsed_seconds": round(exact_seconds, 3),
            "phase": phase_name,
        }, new_indices

    def _heuristic_group_patterns(
        self,
        group: ExportGroup,
        duals: dict[tuple[str, object], float],
        objective_mode: str,
    ) -> list[PackingPattern]:
        candidates = list(self._iter_feasible_base_placements(group))
        if not candidates:
            return []
        score_by_identity = {
            id(col): self._placement_reduced_unit_cost(
                col, duals, objective_mode
            )
            for col in candidates
        }
        scored = sorted(
            candidates,
            key=lambda col: (
                score_by_identity[id(col)],
                col.area_no,
                self.bays[col.bay_key].bay_order,
                col.row_allocation,
            ),
        )
        seeds: list[PlacementColumn] = []
        seen_areas: set[str] = set()
        for col in scored:
            if not seeds or col.area_no not in seen_areas:
                seeds.append(col)
                seen_areas.add(col.area_no)
            if len(seeds) >= int(self.config.heuristic_pricing_variants):
                break
        for col in scored:
            if len(seeds) >= int(self.config.heuristic_pricing_variants):
                break
            if col not in seeds:
                seeds.append(col)

        patterns: list[PackingPattern] = []
        seen: set[tuple] = set()
        for seed in seeds:
            order = sorted(
                scored,
                key=lambda col: (
                    score_by_identity[id(col)]
                    + (
                        0.0
                        if col.area_no == seed.area_no
                        else self._area_activation_penalty()
                    )
                    + (
                        0.0
                        if col.bay_key == seed.bay_key
                        else self._row_activation_penalty()
                    ),
                    0 if col is seed else 1,
                    col.area_no,
                    self.bays[col.bay_key].bay_order,
                    col.row_allocation,
                ),
            )
            order = [seed] + [col for col in order if col is not seed]
            state = self._empty_selection_state()
            placements: list[PlacementColumn] = []
            remaining = int(group.demand)
            for unit in order:
                if remaining <= 0:
                    break
                capacity = self._specific_row_remaining_capacity(
                    group, unit, state, remaining
                )
                if capacity <= 0:
                    continue
                placement = replace(
                    unit,
                    quantity=capacity,
                    stack_units=self._stack_units_for_quantity(
                        unit.bay_key, unit.size, group, capacity
                    ),
                    row_allocation=tuple(
                        (bay_key, row_no, capacity)
                        for bay_key, row_no, _qty in unit.row_allocation
                    ),
                )
                if not self._column_fits_state(placement, state, remaining):
                    continue
                self._apply_column_to_state(placement, state)
                placements.append(placement)
                remaining -= capacity
            pattern = self._make_pattern(group, placements, remaining)
            identity = self._pattern_identity(pattern)
            if identity not in seen:
                seen.add(identity)
                patterns.append(pattern)
        return patterns

    def _specific_row_remaining_capacity(
        self,
        group: ExportGroup,
        unit: PlacementColumn,
        state: dict,
        remaining: int,
    ) -> int:
        capacity = self._remaining_capacity_for_group_bay(
            group,
            unit.bay_key,
            state,
            remaining,
            enforce_quota=False,
        )
        for footprint_key, row_no, _qty in unit.row_allocation:
            bay = self.bays[footprint_key]
            capacity = min(
                capacity,
                int(
                    bay.row_physical_capacity.get(
                        row_no, bay.physical_capacity
                    )
                )
                - int(state["row_load"][(footprint_key, row_no)]),
                int(
                    bay.row_cap_by_size.get(unit.size, {}).get(
                        row_no, bay.cap_by_size.get(unit.size, 0)
                    )
                )
                - int(
                    state["row_size_load"][(footprint_key, row_no, unit.size)]
                ),
            )
        return max(0, int(capacity))

    def _solve_exact_group_pricing(
        self,
        group: ExportGroup,
        duals: dict[tuple[str, object], float],
        objective_mode: str,
    ) -> tuple[list[PackingPattern], float]:
        import gurobipy as gp

        candidates = list(self._iter_feasible_base_placements(group))
        model = gp.Model(f"price_{self._key_name((group.group_id,))}")
        model.Params.OutputFlag = 1 if self.config.verbose else 0
        model.Params.Seed = int(self.config.solver_seed)
        if int(self.config.solver_threads) > 0:
            model.Params.Threads = int(self.config.solver_threads)
        model.Params.TimeLimit = float(self.config.exact_pricing_time_limit)
        model.Params.MIPGap = 0.0
        model.Params.PoolSearchMode = 1
        model.Params.PoolSolutions = int(
            self.config.max_columns_per_group_per_iteration
        )

        x = {
            idx: model.addVar(
                lb=0.0,
                ub=float(
                    min(
                        int(group.demand),
                        self._base_location_capacity(group, candidate),
                    )
                ),
                vtype=gp.GRB.INTEGER,
                name=f"x_{idx}",
            )
            for idx, candidate in enumerate(candidates)
        }
        unplaced = model.addVar(
            lb=0.0,
            ub=float(group.demand),
            vtype=gp.GRB.INTEGER,
            name="unplaced",
        )
        model.addConstr(
            gp.quicksum(x.values()) + unplaced == int(group.demand)
        )

        by_bay: defaultdict[str, list] = defaultdict(list)
        by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        by_row: defaultdict[tuple[str, str], list] = defaultdict(list)
        by_row_size: defaultdict[tuple[str, str, str], list] = defaultdict(list)
        by_stack: defaultdict[tuple[str, str], list] = defaultdict(list)
        by_area: defaultdict[str, list] = defaultdict(list)
        for idx, candidate in enumerate(candidates):
            by_area[candidate.area_no].append(x[idx])
            for footprint_key in self._placement_footprint_keys(
                candidate.bay_key, candidate.size
            ):
                by_bay[footprint_key].append(x[idx])
                by_stack[(footprint_key, candidate.size)].append(x[idx])
            by_bay_size[(candidate.bay_key, candidate.size)].append(x[idx])
            for footprint_key, row_no, _qty in candidate.row_allocation:
                by_row[(footprint_key, row_no)].append(x[idx])
                by_row_size[(footprint_key, row_no, candidate.size)].append(
                    x[idx]
                )
        for bay_key, variables in by_bay.items():
            model.addConstr(
                gp.quicksum(variables)
                <= int(self.bays[bay_key].physical_capacity)
            )
        for (bay_key, size), variables in by_bay_size.items():
            model.addConstr(
                gp.quicksum(variables)
                <= int(self.bays[bay_key].cap_by_size.get(size, 0))
            )
        for (bay_key, row_no), variables in by_row.items():
            bay = self.bays[bay_key]
            model.addConstr(
                gp.quicksum(variables)
                <= int(
                    bay.row_physical_capacity.get(row_no, bay.physical_capacity)
                )
            )
        for (bay_key, row_no, size), variables in by_row_size.items():
            bay = self.bays[bay_key]
            model.addConstr(
                gp.quicksum(variables)
                <= int(
                    bay.row_cap_by_size.get(size, {}).get(
                        row_no, bay.cap_by_size.get(size, 0)
                    )
                )
            )
        for (bay_key, size), variables in by_stack.items():
            stack_count = self._stack_count_for_group(bay_key, size, group)
            unit_capacity = self._stack_unit_capacity_for_group(
                bay_key, size, group
            )
            model.addConstr(
                gp.quicksum(variables)
                <= int(stack_count * unit_capacity)
            )

        placed = model.addVar(vtype=gp.GRB.BINARY, name="placed")
        total_placed = gp.quicksum(x.values())
        model.addConstr(total_placed <= int(group.demand) * placed)
        model.addConstr(placed <= total_placed)
        area_use = {}
        for area_no, variables in by_area.items():
            use = model.addVar(
                vtype=gp.GRB.BINARY, name=f"use_area_{area_no}"
            )
            area_use[area_no] = use
            load = gp.quicksum(variables)
            model.addConstr(load <= int(group.demand) * use)
            model.addConstr(use <= load)
        row_use = {}
        model.update()
        for idx in x:
            use = model.addVar(vtype=gp.GRB.BINARY, name=f"use_row_{idx}")
            row_use[idx] = use
            ub = max(1, int(round(x[idx].UB)))
            model.addConstr(x[idx] <= ub * use)
            model.addConstr(use <= x[idx])

        objective = gp.LinExpr()
        unplaced_cost = 1.0 if objective_mode == "min_unplaced" else 0.0
        unplaced_cost -= self._dual_value(
            duals, "fixed_unplaced_total", "total"
        )
        objective += unplaced_cost * unplaced
        for idx, candidate in enumerate(candidates):
            objective += self._placement_reduced_unit_cost(
                candidate, duals, objective_mode
            ) * x[idx]
        if objective_mode != "min_unplaced":
            objective += self._area_activation_penalty() * gp.quicksum(
                area_use.values()
            )
            objective += self._row_activation_penalty() * gp.quicksum(
                row_use.values()
            )
            objective -= (
                self._area_activation_penalty()
                + self._row_activation_penalty()
            ) * placed
        model.setObjective(objective, gp.GRB.MINIMIZE)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            status = int(model.Status)
            model.dispose()
            raise RuntimeError(
                "exact packing-pattern pricing must be proven optimal: "
                f"group={group.group_id}, status={status}"
            )
        minimum_reduced_cost = float(model.ObjVal) - self._dual_value(
            duals, "group_convexity", group.group_id
        )
        patterns: list[PackingPattern] = []
        seen: set[tuple] = set()
        for solution_number in range(int(model.SolCount)):
            model.Params.SolutionNumber = solution_number
            placements: list[PlacementColumn] = []
            for idx, candidate in enumerate(candidates):
                quantity = int(round(x[idx].Xn))
                if quantity <= 0:
                    continue
                placements.append(
                    replace(
                        candidate,
                        quantity=quantity,
                        stack_units=self._stack_units_for_quantity(
                            candidate.bay_key,
                            candidate.size,
                            group,
                            quantity,
                        ),
                        row_allocation=tuple(
                            (bay_key, row_no, quantity)
                            for bay_key, row_no, _qty in candidate.row_allocation
                        ),
                    )
                )
            pattern = self._make_pattern(
                group, placements, int(round(unplaced.Xn))
            )
            identity = self._pattern_identity(pattern)
            if identity not in seen:
                seen.add(identity)
                patterns.append(pattern)
        model.dispose()
        return patterns, minimum_reduced_cost

    def _solve_by_column_generation(self) -> tuple[Counter[int], Counter[str], dict]:
        """Solve the two lexicographic pattern masters to a certified LP gap."""
        from gurobipy import quicksum

        pricing_start = perf_counter()
        stats: dict = {
            "gurobi_available": True,
            "pricing_method": "stabilized_heuristic_then_exact_group_pattern_pricing",
            "pricing_iterations": [],
            "pricing_stop_reason": "",
            "base_feasible_placement_count": self._base_feasible_placement_count,
        }

        def run_phase(
            phase_name: str,
            objective_mode: str,
            fixed_unplaced_total: float | None = None,
        ) -> tuple[float, float, float, float, str]:
            model, variables, constraints = self._build_restricted_master(
                _GurobiModelAdapter,
                quicksum,
                relax=True,
                objective_mode=objective_mode,
                fixed_unplaced_total=fixed_unplaced_total,
            )
            dual_center: dict[tuple[str, object], float] | None = None
            last_objective = 0.0
            last_unplaced = 0.0
            try:
                for iteration in range(int(self.config.max_iterations)):
                    model.optimize()
                    status = self._gurobi_status_name(model)
                    if status != "optimal":
                        raise RuntimeError(
                            f"Strict {phase_name} pattern-master LP is not optimal: "
                            f"{status}"
                        )
                    last_objective = self._gurobi_objective_value(model)
                    last_unplaced = sum(
                        self._patterns[idx].unplaced
                        * self._gurobi_value(model, var)
                        for idx, var in variables["pattern"].items()
                    )
                    if (
                        objective_mode == "min_unplaced"
                        and last_objective <= 1e-9
                    ):
                        stats["pricing_iterations"].append(
                            {
                                "phase": phase_name,
                                "iteration": iteration,
                                "lp_objective": last_objective,
                                "lp_unplaced_boxes": float(last_unplaced),
                                "new_patterns": 0,
                                "pricing_mode": "nonnegative_stage1_objective_bound",
                                "exact_pricing_performed": False,
                                "valid_lp_lower_bound": 0.0,
                                "certified_lp_gap": 0.0,
                            }
                        )
                        return (
                            last_objective,
                            last_unplaced,
                            0.0,
                            0.0,
                            "nonnegative_stage1_objective_bound",
                        )
                    raw_duals = self._master_dual_snapshot(model, constraints)
                    stabilized = self._stabilized_duals(
                        raw_duals, dual_center
                    )
                    pricing, new_indices = self._price_packing_patterns(
                        raw_duals,
                        stabilized,
                        phase_name,
                        objective_mode,
                    )
                    correction = pricing.get("valid_lower_bound_correction")
                    valid_lower_bound = None
                    certified_lp_gap = None
                    if correction is not None:
                        valid_lower_bound = float(last_objective) + float(
                            correction
                        )
                        if objective_mode == "min_unplaced":
                            valid_lower_bound = max(0.0, valid_lower_bound)
                        certified_lp_gap = max(
                            0.0,
                            (float(last_objective) - valid_lower_bound)
                            / max(abs(float(last_objective)), 1.0),
                        )
                    stats["pricing_iterations"].append(
                        {
                            "phase": phase_name,
                            "iteration": iteration,
                            "lp_objective": last_objective,
                            "lp_unplaced_boxes": float(last_unplaced),
                            "valid_lp_lower_bound": valid_lower_bound,
                            "certified_lp_gap": certified_lp_gap,
                            **pricing,
                        }
                    )
                    if (
                        certified_lp_gap is not None
                        and certified_lp_gap
                        <= float(self.config.pattern_lp_gap_tolerance)
                    ):
                        return (
                            last_objective,
                            last_unplaced,
                            float(valid_lower_bound),
                            float(certified_lp_gap),
                            "certified_pattern_lp_gap",
                        )
                    if not new_indices:
                        if not pricing["exact_pricing_performed"]:
                            raise RuntimeError(
                                "pattern pricing stopped without an exact certificate"
                            )
                        return (
                            last_objective,
                            last_unplaced,
                            float(last_objective),
                            0.0,
                            "no_negative_reduced_cost_pattern",
                        )
                    self._add_patterns_to_restricted_master(
                        model,
                        variables,
                        constraints,
                        new_indices,
                        objective_mode,
                    )
                    dual_center = stabilized
            finally:
                self._free_gurobi_model(model)
            raise RuntimeError(
                f"Strict {phase_name} pattern pricing exceeded max_iterations="
                f"{self.config.max_iterations}"
            )

        (
            _phase1_objective,
            phase1_unplaced,
            _phase1_lower_bound,
            _phase1_gap,
            phase1_stop_reason,
        ) = run_phase(
            "minimum_unplaced", "min_unplaced"
        )
        if abs(phase1_unplaced - round(phase1_unplaced)) > 1e-7:
            raise RuntimeError(
                "stage-1 pattern LP has a fractional unplaced optimum; "
                "cannot define the integer lexicographic boundary"
            )
        phase1_unplaced = float(round(phase1_unplaced))
        (
            phase2_rmp_objective,
            phase2_unplaced,
            phase2_lower_bound,
            phase2_lp_gap,
            phase2_stop_reason,
        ) = run_phase(
            "secondary_objective",
            "full",
            fixed_unplaced_total=phase1_unplaced,
        )
        stats.update(
            {
                "pricing_stop_reason": "two_phase_certified_pattern_pricing",
                "pricing_phase_stop_reasons": {
                    "minimum_unplaced": phase1_stop_reason,
                    "secondary_objective": phase2_stop_reason,
                },
                "pricing_iterations_run": len(stats["pricing_iterations"]),
                "pricing_phase1_lp_unplaced_boxes": phase1_unplaced,
                "pricing_phase2_rmp_objective": phase2_rmp_objective,
                "pricing_phase2_lp_lower_bound": phase2_lower_bound,
                "pricing_phase2_certified_gap": phase2_lp_gap,
                "pattern_generation_pricing_elapsed_seconds": round(
                    perf_counter() - pricing_start, 3
                ),
            }
        )

        integer_start = perf_counter()
        artificial_start = Counter(
            {idx: 1 for idx in self._artificial_pattern_by_group.values()}
        )
        selected_patterns, unplaced, integer_stats = (
            self._solve_lexicographic_integer_master(
                artificial_start,
                float(self.config.total_time_limit)
                if self.config.total_time_limit > 0
                else None,
            )
        )
        if not integer_stats.get("lexicographic_integer_master_used"):
            raise RuntimeError(
                "The final two-stage integer pattern master produced no stage-2 solution"
            )
        pattern_count = len(self._patterns)
        selected, expanded_unplaced = self._expand_pattern_selection(
            selected_patterns
        )
        if expanded_unplaced != unplaced:
            raise RuntimeError(
                "packing-pattern expansion changed unplaced quantities"
            )
        final_objective = self._selected_solution_energy(selected, unplaced)
        model_objective = float(
            integer_stats.get("lexicographic_stage2_objective", final_objective)
        )
        if abs(model_objective - final_objective) > 1e-6:
            raise RuntimeError(
                "integer pattern objective differs from reconstructed row plan: "
                f"master={model_objective}, reconstructed={final_objective}"
            )
        pattern_integer_objective = final_objective
        selected, unplaced, polishing_stats = (
            self._polish_with_pattern_location_pool(selected, unplaced)
        )
        if not polishing_stats.get("lexicographic_integer_master_used"):
            raise RuntimeError("pattern-location polishing produced no stage-2 solution")
        final_objective = self._selected_solution_energy(selected, unplaced)
        if final_objective > pattern_integer_objective + 1e-7:
            raise RuntimeError(
                "pattern-location polishing worsened the incumbent objective: "
                f"before={pattern_integer_objective}, after={final_objective}"
            )
        final_unplaced = int(sum(unplaced.values()))
        lp_gap: float | None = None
        gap_reason = "available"
        if abs(phase1_unplaced - final_unplaced) <= 1e-7:
            denominator = max(abs(final_objective), 1e-12)
            lp_gap = max(
                0.0,
                (final_objective - phase2_lower_bound) / denominator,
            )
        else:
            gap_reason = (
                "phase2 LP and integer pattern master fix different unplaced totals"
            )
        stats.update(polishing_stats)
        stats.update(
            {
                "master_algorithm": (
                    "stabilized_pattern_column_generation_then_"
                    "pattern_pool_fix_and_optimize"
                ),
                "master_bound_scope": "complete_group_pattern_lp_relaxation",
                "master_status": "lexicographic_pattern_pool_integer_polishing",
                "master_solve_seconds": round(
                    perf_counter() - integer_start, 3
                ),
                "master_objective": final_objective,
                "master_mip_gap": polishing_stats.get(
                    "lexicographic_stage2_gap"
                ),
                "restricted_integer_master": {
                    "pattern_count": pattern_count,
                    "selected_pattern_count": int(sum(selected_patterns.values())),
                    "objective": integer_stats.get(
                        "lexicographic_stage2_objective"
                    ),
                    "bound": integer_stats.get("lexicographic_stage2_bound"),
                    "mip_gap": integer_stats.get("lexicographic_stage2_gap"),
                    "stage1_unplaced_boxes": integer_stats.get(
                        "lexicographic_stage1_unplaced_boxes"
                    ),
                },
                "pattern_pool_integer_polishing": {
                    **polishing_stats,
                    "objective_before_polishing": pattern_integer_objective,
                    "objective_after_polishing": final_objective,
                    "improvement": max(
                        0.0, pattern_integer_objective - final_objective
                    ),
                },
                "complete_pattern_lp_lower_bound": phase2_lower_bound,
                "complete_model_lp_bound_gap": lp_gap,
                "complete_model_lp_bound_gap_reason": gap_reason,
                "complete_model_certified_gap": lp_gap,
                "complete_model_certified_gap_source": (
                    "exact_group_pattern_pricing_lp_lower_bound"
                ),
                "restricted_master_lp_unplaced_boxes": float(
                    phase2_unplaced
                ),
                "generated_pattern_count": pattern_count,
                "selected_pattern_count": int(sum(selected_patterns.values())),
            }
        )
        return selected, unplaced, stats

    def _add_patterns_to_restricted_master(
        self,
        model,
        variables: dict,
        constraints: dict,
        pattern_indices: Iterable[int],
        objective_mode: str,
    ) -> None:
        """Insert new pattern variables while preserving the RMP basis."""
        for idx in pattern_indices:
            pattern = self._patterns[int(idx)]
            terms: list[tuple[float, object]] = []
            for section, values in self._pattern_master_coefficients(
                pattern
            ).items():
                rows = constraints.get(section, {})
                for key, coefficient in values.items():
                    row = rows.get(key)
                    if row is not None and coefficient:
                        terms.append((float(coefficient), row))
            variables["pattern"][idx] = model.addPricedVar(
                terms,
                lb=0.0,
                ub=1.0,
                vtype="C",
                obj=(
                    float(pattern.unplaced)
                    if objective_mode == "min_unplaced"
                    else float(pattern.phase2_cost)
                ),
                name=f"pattern_{idx}",
            )

    def _expand_pattern_selection(
        self,
        selected_patterns: Counter[int],
    ) -> tuple[Counter[int], Counter[str]]:
        """Convert selected patterns into the existing row-output structure."""
        self._columns.clear()
        self._column_keys.clear()
        selected: Counter[int] = Counter()
        unplaced: Counter[str] = Counter()
        for pattern_idx, chosen in sorted(selected_patterns.items()):
            if chosen <= 0:
                continue
            pattern = self._patterns[pattern_idx]
            if int(chosen) != 1:
                raise RuntimeError(
                    "group convexity permits exactly one selected pattern per group"
                )
            if pattern.unplaced > 0:
                unplaced[pattern.group_id] += int(pattern.unplaced)
            for placement in pattern.placements:
                idx = self._append_generated_column(placement)
                selected[idx] = 1
        return selected, unplaced

    def _polish_with_pattern_location_pool(
        self,
        incumbent_selected: Counter[int],
        incumbent_unplaced: Counter[str],
    ) -> tuple[Counter[int], Counter[str], dict]:
        """Recombine all row locations discovered inside generated patterns.

        This is a compact fix-and-optimize neighborhood, not branch-and-price:
        pattern pricing remains finished, while the integer polishing model may
        exchange individual row allocations among the generated group patterns.
        """
        incumbent_columns = list(self._columns)
        pool_candidates: dict[
            tuple[str, str, int, tuple[tuple[str, str, int], ...]],
            PlacementColumn,
        ] = {}
        for pattern in self._patterns:
            for placement in pattern.placements:
                unit = replace(
                    placement,
                    column_id="",
                    quantity=1,
                    stack_units=1,
                    row_allocation=tuple(
                        (bay_key, row_no, 1)
                        for bay_key, row_no, qty in placement.row_allocation
                        if int(qty) > 0
                    ),
                )
                pool_candidates.setdefault(self._column_identity(unit), unit)

        self._columns.clear()
        self._column_keys.clear()
        pool_index: dict[
            tuple[str, str, int, tuple[tuple[str, str, int], ...]], int
        ] = {}
        for key, unit in sorted(
            pool_candidates.items(),
            key=lambda item: (
                item[1].group_id,
                item[1].area_no,
                self.bays[item[1].bay_key].bay_order,
                item[1].row_allocation,
            ),
        ):
            pool_index[key] = self._append_generated_column(unit)

        start_selected: Counter[int] = Counter()
        for idx, chosen in incumbent_selected.items():
            if chosen <= 0:
                continue
            placement = incumbent_columns[idx]
            unit = replace(
                placement,
                column_id="",
                quantity=1,
                stack_units=1,
                row_allocation=tuple(
                    (bay_key, row_no, 1)
                    for bay_key, row_no, qty in placement.row_allocation
                    if int(qty) > 0
                ),
            )
            pool_idx = pool_index.get(self._column_identity(unit))
            if pool_idx is None:
                raise RuntimeError("incumbent row location is absent from pattern pool")
            start_selected[pool_idx] += int(placement.quantity) * int(chosen)

        selected, unplaced, solve_stats = (
            self._solve_location_pool_integer_master(
                start_selected,
                Counter(incumbent_unplaced),
                float(self.config.total_time_limit)
                if self.config.total_time_limit > 0
                else None,
            )
        )
        solve_stats["location_pool_size"] = len(self._columns)
        solve_stats["source_pattern_count"] = len(self._patterns)
        solve_stats["method"] = "pattern_location_pool_fix_and_optimize"
        return selected, unplaced, solve_stats

    def _build_restricted_master(
        self,
        Model,
        quicksum,
        relax: bool,
        objective_mode: str = "full",
        fixed_unplaced_total: float | None = None,
    ):
        model = Model("yard_export_group_pattern_master_gurobi")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        if relax:
            self._set_gurobi_param(model, "Method", int(self.config.lp_method))
        model.setMinimize()
        pattern_vtype = "C" if relax else "B"
        patterns = {
            idx: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=pattern_vtype,
                obj=(
                    float(pattern.unplaced)
                    if objective_mode == "min_unplaced"
                    else float(pattern.phase2_cost)
                ),
                name=f"pattern_{idx}",
            )
            for idx, pattern in enumerate(self._patterns)
        }
        import_reserve = {
            (flow, size, bay_key): model.addVar(
                lb=0.0,
                ub=float(capacity),
                vtype="C" if relax else "I",
                obj=0.0,
                name=(
                    f"import_reserve_{flow}_{size}_"
                    f"{self._key_name((bay_key,))}"
                ),
            )
            for (flow, size), candidates in sorted(
                self.import_reservation_candidates.items()
            )
            for bay_key, capacity in candidates
        }

        coefficient_rows: defaultdict[
            str, defaultdict[object, list[tuple[int, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        for idx, pattern in enumerate(self._patterns):
            for section, values in self._pattern_master_coefficients(
                pattern
            ).items():
                for key, coefficient in values.items():
                    if coefficient:
                        coefficient_rows[section][key].append(
                            (idx, float(coefficient))
                        )

        import_by_bay: defaultdict[str, list] = defaultdict(list)
        import_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        import_by_flow_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        import_by_flow_area_size: defaultdict[tuple[str, str, str], list] = (
            defaultdict(list)
        )
        for (flow, size, bay_key), var in import_reserve.items():
            area_no = self.bays[bay_key].area_no
            for footprint_key in self._placement_footprint_keys(bay_key, size):
                import_by_bay[footprint_key].append(var)
            import_by_bay_size[(bay_key, size)].append(var)
            import_by_flow_size[(flow, size)].append(var)
            import_by_flow_area_size[(flow, area_no, size)].append(var)

        group_convexity = {}
        for group in self.groups:
            items = coefficient_rows["group_convexity"].get(
                group.group_id, []
            )
            group_convexity[group.group_id] = model.addCons(
                quicksum(coefficient * patterns[idx] for idx, coefficient in items)
                == 1.0,
                name=f"choose_pattern_{group.group_id}",
            )

        bay_capacity_limit = {}
        for bay_key in sorted(self._master_bay_capacity_keys):
            items = coefficient_rows["bay_capacity_limit"].get(bay_key, [])
            bay_capacity_limit[bay_key] = model.addCons(
                quicksum(coefficient * patterns[idx] for idx, coefficient in items)
                + quicksum(import_by_bay.get(bay_key, []))
                <= int(self.bays[bay_key].physical_capacity),
                name=f"bay_cap_{self._key_name((bay_key,))}",
            )
        bay_size_limit = {}
        for key in sorted(self._master_bay_size_keys):
            bay_key, size = key
            items = coefficient_rows["bay_size_limit"].get(key, [])
            bay_size_limit[key] = model.addCons(
                quicksum(coefficient * patterns[idx] for idx, coefficient in items)
                + quicksum(import_by_bay_size.get(key, []))
                <= int(self.bays[bay_key].cap_by_size.get(size, 0)),
                name=f"bay_size_{self._key_name((bay_key, size))}",
            )
        row_capacity_limit = {}
        for key in sorted(self._master_row_capacity_keys):
            bay_key, row_no = key
            bay = self.bays[bay_key]
            items = coefficient_rows["row_capacity_limit"].get(key, [])
            row_capacity_limit[key] = model.addCons(
                quicksum(coefficient * patterns[idx] for idx, coefficient in items)
                <= int(
                    bay.row_physical_capacity.get(row_no, bay.physical_capacity)
                ),
                name=f"row_cap_{self._key_name((bay_key, row_no))}",
            )
        row_size_limit = {}
        for key in sorted(self._master_row_size_keys):
            bay_key, row_no, size = key
            bay = self.bays[bay_key]
            items = coefficient_rows["row_size_limit"].get(key, [])
            row_size_limit[key] = model.addCons(
                quicksum(coefficient * patterns[idx] for idx, coefficient in items)
                <= int(
                    bay.row_cap_by_size.get(size, {}).get(
                        row_no, bay.cap_by_size.get(size, 0)
                    )
                ),
                name=f"row_size_{self._key_name((bay_key, row_no, size))}",
            )

        bay_port_stack_link = {}
        bay_stack_total_limit = {}
        stack_vars_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        for key in sorted(self._master_stack_keys):
            bay_key, port, size = key
            sample_group = self.groups_by_id.get(
                self._master_stack_sample_group.get(key, "")
            )
            if sample_group is None:
                continue
            stack_count = self._stack_count_for_group(
                bay_key, size, sample_group
            )
            unit_capacity = self._stack_unit_capacity_for_group(
                bay_key, size, sample_group
            )
            if stack_count <= 0 or unit_capacity <= 0:
                continue
            stack_var = model.addVar(
                lb=0.0,
                ub=stack_count,
                vtype="C" if relax else "I",
                name=f"stack_{self._key_name(key)}",
            )
            items = coefficient_rows["bay_port_stack_link"].get(key, [])
            bay_port_stack_link[key] = model.addCons(
                quicksum(coefficient * patterns[idx] for idx, coefficient in items)
                <= unit_capacity * stack_var,
                name=f"stack_load_{self._key_name(key)}",
            )
            stack_vars_by_bay_size[(bay_key, size)].append(stack_var)
        for key, stack_vars in stack_vars_by_bay_size.items():
            stack_count = self._stack_count_for_bay_size(*key)
            bay_stack_total_limit[key] = model.addCons(
                quicksum(stack_vars) <= stack_count,
                name=f"stack_total_{self._key_name(key)}",
            )

        import_total_balance = {}
        for key, required in sorted(self.import_total_by_flow_size.items()):
            candidates = import_by_flow_size.get(key, [])
            if not candidates:
                raise ValueError(
                    "import capacity reservation has no compatible bay: "
                    f"flow={key[0]}, size={key[1]}, required={required}"
                )
            import_total_balance[key] = model.addCons(
                quicksum(candidates) == int(required),
                name=f"import_total_{self._key_name(key)}",
            )
        import_reference_balance = self._add_import_reference_deviation(
            quicksum,
            model,
            import_by_flow_area_size,
            objective_mode=objective_mode,
        )

        fixed_unplaced = {}
        if fixed_unplaced_total is not None:
            items = coefficient_rows["fixed_unplaced_total"].get("total", [])
            fixed_unplaced["total"] = model.addCons(
                quicksum(coefficient * patterns[idx] for idx, coefficient in items)
                == float(fixed_unplaced_total),
                name="fixed_total_unplaced",
            )
        area_guidance_balance = self._add_pattern_area_guidance_objective(
            quicksum,
            model,
            patterns,
            coefficient_rows["area_guidance_balance"],
            objective_mode,
        )
        bay_compatibility = self._add_pattern_bay_compatibility_constraints(
            quicksum,
            model,
            patterns,
            coefficient_rows["bay_attr_link"],
            relax,
        )
        row_compatibility = self._add_pattern_row_compatibility_constraints(
            quicksum,
            model,
            patterns,
            coefficient_rows["row_attr_link"],
            relax,
        )
        return model, {
            "pattern": patterns,
            "import_reserve": import_reserve,
        }, {
            "group_convexity": group_convexity,
            "bay_capacity_limit": bay_capacity_limit,
            "bay_size_limit": bay_size_limit,
            "row_capacity_limit": row_capacity_limit,
            "row_size_limit": row_size_limit,
            "bay_port_stack_link": bay_port_stack_link,
            "bay_stack_total_limit": bay_stack_total_limit,
            "import_total_balance": import_total_balance,
            "import_reference_balance": import_reference_balance,
            "fixed_unplaced_total": fixed_unplaced,
            "area_guidance_balance": area_guidance_balance,
            **bay_compatibility,
            **row_compatibility,
        }

    def _add_pattern_area_guidance_objective(
        self,
        quicksum,
        model,
        patterns,
        items_by_key,
        objective_mode: str,
    ) -> dict[tuple[str, str, str, str], object]:
        if objective_mode == "min_unplaced":
            return {}
        balance = {}
        for key in sorted(self._master_area_guidance_keys):
            voyage_id, flow, area_no, big_size = key
            target = self._area_size_target(
                voyage_id, flow, area_no, big_size
            )
            pos = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"guide_pos_{self._key_name(key)}",
            )
            neg = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"guide_neg_{self._key_name(key)}",
            )
            items = items_by_key.get(key, [])
            actual = quicksum(
                coefficient * patterns[idx] for idx, coefficient in items
            )
            balance[key] = model.addCons(
                actual - target == pos - neg,
                name=f"guide_balance_{self._key_name(key)}",
            )
        return balance

    def _add_pattern_bay_compatibility_constraints(
        self,
        quicksum,
        model,
        patterns,
        items_by_key,
        relax: bool,
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
            items = items_by_key.get(key, [])
            links[key] = model.addCons(
                quicksum(
                    coefficient * patterns[idx] for idx, coefficient in items
                )
                <= self._master_bay_attr_big_m[key] * use,
                name=f"bay_attr_link_{self._key_name(key)}",
            )
            uses_by_scope[(bay_key, attr, scope)].append(use)
        for key, uses in uses_by_scope.items():
            choices[key] = model.addCons(
                quicksum(uses) <= 1,
                name=f"bay_attr_one_{self._key_name(key)}",
            )
        return {"bay_attr_link": links, "bay_attr_one": choices}

    def _add_pattern_row_compatibility_constraints(
        self,
        quicksum,
        model,
        patterns,
        items_by_key,
        relax: bool,
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
            items = items_by_key.get(key, [])
            links[key] = model.addCons(
                quicksum(
                    coefficient * patterns[idx] for idx, coefficient in items
                )
                <= self._master_row_attr_big_m[key] * use,
                name=f"row_attr_link_{self._key_name(key)}",
            )
            uses_by_scope[(bay_key, row_no, attr, scope)].append(use)
        for key, uses in uses_by_scope.items():
            choices[key] = model.addCons(
                quicksum(uses) <= 1,
                name=f"row_attr_one_{self._key_name(key)}",
            )
        return {"row_attr_link": links, "row_attr_one": choices}

    def _solve_lexicographic_integer_master(
        self,
        start_selected: Counter[int],
        remaining_seconds: float | None,
    ) -> tuple[Counter[int], Counter[str], dict]:
        """Choose one generated packing pattern per group in two stages."""
        from gurobipy import quicksum

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
            return Counter(start_selected), Counter(self._initial_unplaced_start), stats

        def apply_start(variables: dict, selected: Counter[int]) -> None:
            for idx, var in variables["pattern"].items():
                var.Start = float(1 if selected.get(idx, 0) > 0 else 0)

        stage1, vars1, _constraints1 = self._build_restricted_master(
            _GurobiModelAdapter,
            quicksum,
            relax=False,
            objective_mode="min_unplaced",
        )
        apply_start(vars1, start_selected)
        stage1_limit = max(1.0, available / 2.0)
        self._set_gurobi_param(stage1, "TimeLimit", stage1_limit)
        self._set_gurobi_param(stage1, "MIPGap", 0.0)
        stage1.optimize()
        status1 = self._gurobi_status_name(stage1)
        stats["lexicographic_stage1_status"] = status1
        stats["lexicographic_stage1_optimal"] = status1 == "optimal"
        if status1 != "optimal":
            self._free_gurobi_model(stage1)
            raise RuntimeError(
                "Integer pattern-master stage 1 must be optimal before stage 2: "
                f"status={status1}"
            )
        selected1 = self._gurobi_selected_patterns(stage1, vars1)
        unplaced1 = self._unplaced_from_pattern_selection(selected1)
        self._final_import_reservation = self._gurobi_import_reservation_values(
            stage1, vars1
        )
        optimum_unplaced = int(sum(unplaced1.values()))
        stats["lexicographic_stage1_unplaced_boxes"] = optimum_unplaced
        stats["lexicographic_stage1_bound"] = self._gurobi_dual_bound(stage1)
        self._free_gurobi_model(stage1)

        stage2, vars2, _constraints2 = self._build_restricted_master(
            _GurobiModelAdapter,
            quicksum,
            relax=False,
            objective_mode="full",
            fixed_unplaced_total=optimum_unplaced,
        )
        apply_start(vars2, selected1)
        self._set_gurobi_param(
            stage2, "TimeLimit", max(1.0, available - stage1_limit)
        )
        self._set_gurobi_param(
            stage2, "MIPGap", max(0.0, float(self.config.mip_gap))
        )
        stage2.optimize()
        status2 = self._gurobi_status_name(stage2)
        stats["lexicographic_stage2_status"] = status2
        stats["lexicographic_stage2_optimal"] = status2 == "optimal"
        if self._gurobi_solution_count(stage2) <= 0:
            self._free_gurobi_model(stage2)
            stats["lexicographic_skip_reason"] = "stage2_no_solution"
            return selected1, unplaced1, stats
        selected2 = self._gurobi_selected_patterns(stage2, vars2)
        unplaced2 = self._unplaced_from_pattern_selection(selected2)
        self._final_import_reservation = self._gurobi_import_reservation_values(
            stage2, vars2
        )
        stats.update(
            {
                "lexicographic_integer_master_used": True,
                "lexicographic_stage2_objective": self._gurobi_objective_value(
                    stage2
                ),
                "lexicographic_stage2_bound": self._gurobi_dual_bound(stage2),
                "lexicographic_stage2_gap": self._gurobi_gap(stage2),
            }
        )
        self._free_gurobi_model(stage2)
        return selected2, unplaced2, stats

    def _gurobi_selected_patterns(self, model, variables: dict) -> Counter[int]:
        return Counter(
            {
                idx: int(round(self._gurobi_value(model, var)))
                for idx, var in variables["pattern"].items()
                if self._gurobi_value(model, var) > 0.5
            }
        )

    def _unplaced_from_pattern_selection(
        self,
        selected: Counter[int],
    ) -> Counter[str]:
        unplaced: Counter[str] = Counter()
        for idx, chosen in selected.items():
            if chosen > 0:
                pattern = self._patterns[idx]
                unplaced[pattern.group_id] += int(pattern.unplaced) * int(chosen)
        return Counter({key: value for key, value in unplaced.items() if value > 0})

    def _iter_feasible_base_placements(
        self,
        group: ExportGroup,
    ) -> Iterable[PlacementColumn]:
        """Enumerate feasible one-box row locations used by pattern pricing."""
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
        return max(
            0,
            min(
                self._feasible_bay_capacity_without_demand(group, column.bay_key),
                *row_capacities,
            ),
        )

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
        """Create the fixed row index set shared by every restricted master."""
        self._master_bay_capacity_keys.clear()
        self._master_bay_size_keys.clear()
        self._master_row_capacity_keys.clear()
        self._master_row_size_keys.clear()
        self._master_stack_keys.clear()
        self._master_stack_sample_group.clear()
        self._master_area_guidance_keys.clear()
        self._master_group_area_keys.clear()
        self._master_group_row_keys.clear()
        self._master_operational_group_keys.clear()
        self._master_bay_attr_choice_keys.clear()
        self._master_row_attr_choice_keys.clear()
        self._master_bay_attr_big_m.clear()
        self._master_row_attr_big_m.clear()
        self._master_group_area_big_m.clear()
        self._master_group_row_big_m.clear()
        self._base_feasible_placement_count = 0
        bay_attr_groups: defaultdict[tuple[str, str, str, str], set[str]] = defaultdict(set)
        row_attr_groups: defaultdict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
        group_area_groups: defaultdict[tuple[tuple[str, ...], str], set[str]] = defaultdict(set)
        group_row_groups: defaultdict[tuple[tuple[str, ...], str, str], set[str]] = defaultdict(set)
        group_area_location_caps: defaultdict[
            tuple[tuple[str, ...], str], dict[tuple[str, str], int]
        ] = defaultdict(dict)
        group_area_bay_caps: defaultdict[
            tuple[tuple[str, ...], str], dict[str, int]
        ] = defaultdict(dict)
        group_row_location_caps: dict[tuple[tuple[str, ...], str, str], int] = {}

        for group in self.groups:
            group_key = self._operational_group_key(group)
            self._master_operational_group_keys.add(group_key)
            for column in self._iter_feasible_base_placements(group):
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
                group_area_key = (group_key, column.area_no)
                self._master_group_area_keys.add(group_area_key)
                group_area_groups[group_area_key].add(group.group_id)
                anchor_row = next(
                    row_no
                    for bay_key, row_no, _qty in column.row_allocation
                    if bay_key == column.bay_key
                )
                group_row_key = (group_key, column.bay_key, anchor_row)
                self._master_group_row_keys.add(group_row_key)
                group_row_groups[group_row_key].add(group.group_id)
                location_capacity = self._base_location_capacity(group, column)
                if location_capacity <= 0:
                    raise ValueError(
                        "feasible base placement has no positive location capacity: "
                        f"group={group.group_id}, bay={column.bay_key}, row={anchor_row}"
                    )
                location_key = (column.bay_key, anchor_row)
                group_area_location_caps[group_area_key][location_key] = max(
                    group_area_location_caps[group_area_key].get(location_key, 0),
                    location_capacity,
                )
                group_area_bay_caps[group_area_key][column.bay_key] = max(
                    group_area_bay_caps[group_area_key].get(column.bay_key, 0),
                    self._feasible_bay_capacity_without_demand(group, column.bay_key),
                )
                group_row_location_caps[group_row_key] = max(
                    group_row_location_caps.get(group_row_key, 0),
                    location_capacity,
                )

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
        for key, group_ids in group_area_groups.items():
            relevant_demand = sum(self.group_demand[group_id] for group_id in group_ids)
            row_caps_by_bay: defaultdict[str, int] = defaultdict(int)
            for (bay_key, _row_no), capacity in group_area_location_caps[key].items():
                row_caps_by_bay[bay_key] += int(capacity)
            feasible_area_capacity = sum(
                min(int(bay_capacity), int(row_caps_by_bay.get(bay_key, 0)))
                for bay_key, bay_capacity in group_area_bay_caps[key].items()
            )
            self._master_group_area_big_m[key] = self._tight_link_bound(
                relevant_demand,
                feasible_area_capacity,
            )
        for key, group_ids in group_row_groups.items():
            relevant_demand = sum(self.group_demand[group_id] for group_id in group_ids)
            self._master_group_row_big_m[key] = self._tight_link_bound(
                relevant_demand,
                group_row_location_caps[key],
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
                "min(relevant_group_demand, feasible_group_specific_capacity); "
                "group-area bounds aggregate feasible bay-row capacities"
            ),
            "bay_attribute_links": self._bound_summary(self._master_bay_attr_big_m.values()),
            "row_attribute_links": self._bound_summary(self._master_row_attr_big_m.values()),
            "group_area_links": self._bound_summary(self._master_group_area_big_m.values()),
            "group_row_links": self._bound_summary(self._master_group_row_big_m.values()),
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
            "bay_used_size": {},
            "bay_used_attrs": {},
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
        """Capacity using only physical footprint and enabled bay size."""
        bay = self.bays.get(bay_key)
        if bay is None or size not in {"20", "40"}:
            return 0
        footprint = self._placement_footprint_keys(bay_key, size)
        if not footprint:
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

    def _area_weights(self, group: ExportGroup) -> Counter[str]:
        weights: Counter[str] = Counter()
        big_size = self._big_plan_size(group.size)
        for (voyage_id, flow, area_no, size), qty in self.quota_by_key.items():
            if voyage_id == group.voyage_id and flow == group.status and size == big_size and qty > 0:
                weights[area_no] += qty
        return weights

    def _quota_key(self, group: ExportGroup, area_no: str) -> tuple[str, str, str, str]:
        return group.voyage_id, group.status, area_no, self._big_plan_size(group.size)

    def _operational_group_key(self, group: ExportGroup) -> tuple[str, ...]:
        attrs = tuple(self.problem.attribute_rules.group_attributes or MANDATORY_BAY_NO_MIX_ATTRS)
        return self._attribute_cluster_key(group, attrs)

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
            raise ValueError(f"secondary objective weights must be finite and nonnegative: {weights}")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"secondary objective weights must sum to 1, got {total}: {weights}")

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
            demand_by_group[self._operational_group_key(group)] += int(group.demand)
        for group_key, area_no in self._master_group_area_keys:
            areas_by_group[group_key].add(area_no)
        for group_key, bay_key, row_no in self._master_group_row_keys:
            rows_by_group[group_key].add((bay_key, row_no))
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
                    "voyage_id": group.voyage_id,
                    "flow": group.status,
                    "port": group.port,
                    "size": group.size,
                    "height": group.height,
                    "demand": int(group.demand),
                    "unplaced_boxes": int(qty),
                }
            )
        return details

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

    def _group_sort_key(self, group: ExportGroup) -> tuple[int, int, int, str, str, str]:
        return (
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
            "row_allocation": ColumnGenerationPlanner._format_row_allocation(col.row_allocation),
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
