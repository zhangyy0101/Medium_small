"""Contiguous-zone generation and objective-guided Fix-and-Optimize.

The model represents concentration through dedicated contiguous zones. A hard
export group reserves one or more contiguous runs of compatible physical
row footprints and sends its actual declared quantity through those runs.
Selected footprints reserve their complete compatible capacity, which makes
physical conflicts additive and gives a genuine column-generation structure.
The final compact row MILP realizes the certified group-bay flows and the same
anonymous import reservation without changing the primary zone decisions.

The solver combines an exactly priced root, a restricted integer master,
objective-guided primal improvement, and certified row-level recourse.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from time import perf_counter

from .direct_milp import DirectMilpPlanner
from .gurobi_backend import (
    GurobiModel,
    MipProgressRecorder,
    StagnationStoppingMipProgressRecorder,
)
from .planner import ColumnGenerationConfig, ColumnGenerationResult

Resource = tuple[str, str]
StripKey = tuple[str, str, str]
ALGORITHM_VERSION = "integrated_zone_v5_dynamic_initial_stopping_phase4"
V5_MULTI_START_POLICY = "v5_multi_start"
COMPLETE_MIP_BASELINE_POLICY = "complete_mip_baseline"


class _RangeMinimumTree:
    """Static range-minimum oracle with deterministic leftmost tie-breaking."""

    def __init__(self, values: list[float]) -> None:
        size = 1
        while size < len(values):
            size *= 2
        self._size = size
        self._tree = [(math.inf, -1)] * (2 * size)
        for index, value in enumerate(values):
            self._tree[size + index] = (float(value), index)
        for index in range(size - 1, 0, -1):
            self._tree[index] = min(
                self._tree[2 * index],
                self._tree[2 * index + 1],
            )

    def query(self, lower: int, upper: int) -> tuple[float, int]:
        """Return the minimum on the nonempty half-open interval [lower, upper)."""

        if lower >= upper:
            raise ValueError("range-minimum query must be nonempty")
        lower += self._size
        upper += self._size
        result = (math.inf, -1)
        while lower < upper:
            if lower & 1:
                result = min(result, self._tree[lower])
                lower += 1
            if upper & 1:
                upper -= 1
                result = min(result, self._tree[upper])
            lower //= 2
            upper //= 2
        return result


@dataclass(frozen=True)
class ContiguousZoneConfig:
    """Dimensionless controls for the contiguous-zone algorithm."""

    max_root_iterations: int = 60
    reduced_cost_tolerance: float = 1e-8
    columns_per_group_per_round: int = 3
    integer_pool_columns_per_group: int = 100
    root_time_fraction: float = 0.50
    zone_mip_time_fraction: float = 0.75
    initial_mip_dynamic_stopping_enabled: bool = False
    initial_mip_max_remaining_fraction: float = 0.50
    initial_mip_min_total_fraction: float = 0.05
    initial_mip_stagnation_total_fraction: float = 0.10
    initial_mip_min_relative_improvement: float = 1e-4
    fix_optimize_local_fraction: float = 0.85
    fix_optimize_objective_mass: float = 0.60
    fix_optimize_zone_fraction: float = 0.35
    fix_optimize_policy: str = "conflict_multi_round"
    fix_optimize_max_rounds: int = 3
    fix_optimize_round_zone_fractions: tuple[float, ...] = (
        0.12,
        0.22,
        0.35,
    )
    fill_time_fraction: float = 0.05
    max_repaired_start_candidates: int = 8
    mip_start_repair_total_fraction: float = 0.08
    mip_start_total_time_fraction: float = 0.10
    max_mip_starts: int = 6
    primal_pool_apply_lp_warm_start: bool = True
    shortage_penalty: float = 1_000.0
    # Integrated paper-model weights after removing the redundant group-area
    # objective.  Its former mass is redistributed proportionally so the
    # remaining pairwise preferences do not change.
    voyage_area_dispersion_weight: float = 0.1500
    zone_dispersion_weight: float = 0.3500
    existing_group_proximity_weight: float = 0.1250
    unused_capacity_weight: float = 0.2125
    berth_distance_weight: float = 0.1625
    # Reproducible objective ablation.  When disabled, the unused-capacity
    # weight is set to zero and all other weights are renormalized
    # proportionally at model construction time.
    unused_capacity_objective_enabled: bool = True
    # No terminal-approved utilization threshold is available.  The epsilon
    # cap is therefore placed halfway between the instance load lower bound and
    # full use of residual capacity by default.  This is a declared experiment
    # parameter and must be included in sensitivity analysis.
    peak_utilization_headroom_fraction: float = 0.50

    def validate(self) -> None:
        if int(self.max_root_iterations) <= 0:
            raise ValueError("max_root_iterations must be positive")
        if float(self.reduced_cost_tolerance) <= 0.0:
            raise ValueError("reduced_cost_tolerance must be positive")
        if int(self.columns_per_group_per_round) <= 0:
            raise ValueError("columns_per_group_per_round must be positive")
        if int(self.integer_pool_columns_per_group) < 0:
            raise ValueError("integer_pool_columns_per_group cannot be negative")
        if not 0.0 < float(self.root_time_fraction) < 1.0:
            raise ValueError("root_time_fraction must lie strictly between 0 and 1")
        if not 0.0 < float(self.zone_mip_time_fraction) < 1.0:
            raise ValueError(
                "zone_mip_time_fraction must lie strictly between 0 and 1"
            )
        if not isinstance(self.initial_mip_dynamic_stopping_enabled, bool):
            raise ValueError(
                "initial_mip_dynamic_stopping_enabled must be boolean"
            )
        if not 0.0 < float(self.initial_mip_max_remaining_fraction) <= 1.0:
            raise ValueError(
                "initial_mip_max_remaining_fraction must lie in (0, 1]"
            )
        if not 0.0 <= float(self.initial_mip_min_total_fraction) < 1.0:
            raise ValueError(
                "initial_mip_min_total_fraction must lie in [0, 1)"
            )
        if not 0.0 < float(self.initial_mip_stagnation_total_fraction) < 1.0:
            raise ValueError(
                "initial_mip_stagnation_total_fraction must lie in (0, 1)"
            )
        if float(self.initial_mip_min_relative_improvement) <= 0.0:
            raise ValueError(
                "initial_mip_min_relative_improvement must be positive"
            )
        if not 0.0 < float(self.fix_optimize_local_fraction) < 1.0:
            raise ValueError(
                "fix_optimize_local_fraction must lie strictly between 0 and 1"
            )
        if not 0.0 < float(self.fix_optimize_objective_mass) <= 1.0:
            raise ValueError(
                "fix_optimize_objective_mass must lie in (0, 1]"
            )
        if not 0.0 < float(self.fix_optimize_zone_fraction) <= 1.0:
            raise ValueError(
                "fix_optimize_zone_fraction must lie in (0, 1]"
            )
        if self.fix_optimize_policy not in {
            "disabled",
            "objective",
            "conflict_multi_round",
            "hybrid_multi_round",
        }:
            raise ValueError(
                "fix_optimize_policy must be disabled, objective, "
                "conflict_multi_round, or hybrid_multi_round: "
                f"{self.fix_optimize_policy!r}"
            )
        if int(self.fix_optimize_max_rounds) <= 0:
            raise ValueError("fix_optimize_max_rounds must be positive")
        round_fractions = tuple(
            float(value) for value in self.fix_optimize_round_zone_fractions
        )
        if len(round_fractions) < int(self.fix_optimize_max_rounds):
            raise ValueError(
                "fix_optimize_round_zone_fractions must provide one value "
                "per configured round"
            )
        if any(value <= 0.0 or value > 1.0 for value in round_fractions):
            raise ValueError(
                "fix_optimize_round_zone_fractions must lie in (0, 1]"
            )
        if any(
            later + 1e-12 < earlier
            for earlier, later in zip(round_fractions, round_fractions[1:])
        ):
            raise ValueError(
                "fix_optimize_round_zone_fractions must be nondecreasing"
            )
        if not 0.0 < float(self.fill_time_fraction) < 1.0:
            raise ValueError("fill_time_fraction must lie strictly between 0 and 1")
        if int(self.max_repaired_start_candidates) <= 0:
            raise ValueError("max_repaired_start_candidates must be positive")
        if not 0.0 < float(self.mip_start_repair_total_fraction) < 1.0:
            raise ValueError(
                "mip_start_repair_total_fraction must lie strictly between 0 and 1"
            )
        if not 0.0 < float(self.mip_start_total_time_fraction) < 1.0:
            raise ValueError(
                "mip_start_total_time_fraction must lie strictly between 0 and 1"
            )
        if (
            float(self.mip_start_repair_total_fraction)
            > float(self.mip_start_total_time_fraction)
        ):
            raise ValueError(
                "mip_start_repair_total_fraction cannot exceed the total "
                "mip-start preparation fraction"
            )
        if int(self.max_mip_starts) <= 0:
            raise ValueError("max_mip_starts must be positive")
        if not isinstance(self.primal_pool_apply_lp_warm_start, bool):
            raise ValueError("primal_pool_apply_lp_warm_start must be boolean")
        if float(self.shortage_penalty) <= 0.0:
            raise ValueError("shortage_penalty must be positive")
        if not 0.0 <= float(self.peak_utilization_headroom_fraction) <= 1.0:
            raise ValueError(
                "peak_utilization_headroom_fraction must lie in [0, 1]"
            )
        objective_weights = {
            "voyage_area_dispersion": self.voyage_area_dispersion_weight,
            "zone_dispersion": self.zone_dispersion_weight,
            "existing_group_proximity": self.existing_group_proximity_weight,
            "unused_capacity": self.unused_capacity_weight,
            "berth_distance": self.berth_distance_weight,
        }
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in objective_weights.values()
        ):
            raise ValueError(
                "zone business objective weights must be finite and "
                f"nonnegative: {objective_weights}"
            )
        if abs(sum(float(value) for value in objective_weights.values()) - 1.0) > 1e-9:
            raise ValueError(
                "zone business objective weights must sum to 1: "
                f"{objective_weights}"
            )


@dataclass(frozen=True)
class ContiguousZone:
    """One single-group contiguous run on an ordered physical row strip."""

    zone_id: int
    group_id: str
    area_no: str
    row_no: str
    candidate_indices: tuple[int, ...]
    capacity: int
    resources: tuple[Resource, ...]
    anchor_bay_loads: tuple[tuple[str, int], ...]
    bay_loads: tuple[tuple[str, int], ...]
    bay_size_loads: tuple[tuple[tuple[str, str], int], ...]
    stack_uses: tuple[tuple[tuple[str, str], int], ...]
    bay_attr_uses: tuple[tuple[tuple[str, str, str, str], int], ...]
    objective_cost: float


@dataclass(frozen=True)
class RootSnapshot:
    """Exact-root state retained after the proof master is destroyed."""

    objective: float
    duals: dict[tuple[str, object], float]
    zone_values: dict[int, float]
    export_flow_values: dict[tuple[str, str], float]
    import_values: dict[tuple[str, str, str], float]
    area_values: dict[tuple[str, str], float]
    voyage_area_values: dict[tuple[str, str], float]
    attr_values: dict[tuple[str, str, str, str], float]
    lp_warm_start: dict[str, dict[str, float]] | None
    proof_zone_indices: frozenset[int]


class ContiguousZoneGenerationPlanner(DirectMilpPlanner):
    """Generate dedicated contiguous zones, then perform exact row filling."""

    def __init__(
        self,
        problem,
        config: ColumnGenerationConfig | None = None,
        zone_config: ContiguousZoneConfig | None = None,
    ) -> None:
        # Make the integrated contract explicit even when an old caller passes
        # historical large-plan reference fields on ProblemData.
        if (
            not problem.import_demand_by_flow_size
            and any(
                int(quantity) > 0
                for quantity in problem.import_area_size_reference.values()
            )
        ):
            raise ValueError(
                "integrated paper model requires direct anonymous import "
                "demand_by_flow_size; legacy import area references are not input"
            )
        integrated_problem = replace(
            problem,
            area_guidance_target={},
            import_area_size_reference={},
        )
        super().__init__(integrated_problem, config)
        self.zone_config = zone_config or ContiguousZoneConfig()
        self.zone_config.validate()
        self._zones: list[ContiguousZone] = []
        self._zone_indices_by_group: defaultdict[str, list[int]] = defaultdict(list)
        self._zone_indices_by_strip: defaultdict[StripKey, list[int]] = defaultdict(list)
        self._zone_id_by_signature: dict[tuple[int, ...], int] = {}
        self._strip_runs: dict[StripKey, tuple[tuple[int, ...], ...]] = {}
        self._zone_capacity_limit_by_strip: dict[StripKey, int] = {}
        self._possible_zone_count = 0
        self._possible_zone_count_by_group: Counter[str] = Counter()
        self._atomic_capacity: list[int] = []
        self._atomic_attr_keys: list[tuple[tuple[str, str, str, str], ...]] = []
        self._atomic_resources: list[tuple[Resource, ...]] = []
        self._candidate_areas_by_group: dict[str, frozenset[str]] = {}
        self._candidate_resources_by_group: dict[
            str, frozenset[Resource]
        ] = {}
        self._zone_objective_scales: dict[str, float] = {}
        self._peak_utilization_policy: dict[str, object] = {}

    def _zone_objective_weights(self) -> dict[str, float]:
        weights = {
            "voyage_area_dispersion": float(
                self.zone_config.voyage_area_dispersion_weight
            ),
            "zone_dispersion": float(
                self.zone_config.zone_dispersion_weight
            ),
            "existing_group_proximity": float(
                self.zone_config.existing_group_proximity_weight
            ),
            "unused_capacity": float(
                self.zone_config.unused_capacity_weight
            ),
            "berth_distance": float(
                self.zone_config.berth_distance_weight
            ),
        }
        if not self.zone_config.unused_capacity_objective_enabled:
            weights["unused_capacity"] = 0.0
            retained_total = sum(weights.values())
            if retained_total <= 0.0:
                raise ValueError(
                    "unused-capacity ablation leaves no positive objective weight"
                )
            weights = {
                key: value / retained_total
                for key, value in weights.items()
            }
        return weights

    def _zone_objective_scale(self, key: str) -> float:
        value = self._zone_objective_scales.get(key)
        if value is None:
            raise RuntimeError(
                f"zone objective normalization is not prepared: {key}"
            )
        return max(1.0, float(value))

    def _voyage_area_activation_penalty(self) -> float:
        return (
            self._zone_objective_weights()["voyage_area_dispersion"]
            / self._zone_objective_scale("voyage_area_dispersion")
        )

    def _zone_activation_penalty(self) -> float:
        return (
            self._zone_objective_weights()["zone_dispersion"]
            / self._zone_objective_scale("zone_dispersion")
        )

    def _zone_existing_proximity_unit_cost(
        self, group_id: str, bay_key: str
    ) -> float:
        group = self.groups_by_id[group_id]
        return (
            self._zone_objective_weights()["existing_group_proximity"]
            * self._normalized_existing_proximity(group, bay_key)
            / self._zone_objective_scale("existing_group_proximity")
        )

    def _zone_berth_distance_unit_cost(
        self, group_id: str, area_no: str
    ) -> float:
        group = self.groups_by_id[group_id]
        return (
            self._zone_objective_weights()["berth_distance"]
            * self._normalized_berth_distance(group.voyage_id, area_no)
            / self._zone_objective_scale("berth_distance")
        )

    def _zone_flow_unit_cost(self, group_id: str, bay_key: str) -> float:
        return self._zone_existing_proximity_unit_cost(
            group_id, bay_key
        ) + self._zone_berth_distance_unit_cost(
            group_id, self.bays[bay_key].area_no
        )

    def _zone_business_objective_specification(self) -> dict[str, object]:
        return {
            "type": "normalized_weighted_contiguous_zone_objective",
            "decision_level": (
                "voyage_area_group_area_contiguous_zone_and_export_flow"
            ),
            "hard_demand_balance": True,
            "upstream_large_plan_required": False,
            "weights": self._zone_objective_weights(),
            "scales": dict(self._zone_objective_scales),
            "objective_design": {
                "group_area_dispersion": "diagnostic_only",
                "unused_capacity_objective_enabled": bool(
                    self.zone_config.unused_capacity_objective_enabled
                ),
            },
            "peak_utilization_epsilon_constraint": dict(
                self._peak_utilization_policy
            ),
            "row_recourse_role": (
                "exact_feasibility_and_secondary_quality_only"
            ),
        }

    @staticmethod
    def _integrated_row_quality_components(
        components: dict[str, object],
    ) -> dict[str, object]:
        """Remove unused legacy-guidance labels from recourse diagnostics."""

        cleaned = dict(components)
        raw = dict(cleaned.get("raw", {}))
        for key in (
            "area_guidance_l1_deviation",
            "export_area_guidance_l1_deviation",
            "import_reservation_area_l1_deviation",
        ):
            raw.pop(key, None)
        cleaned["raw"] = raw
        for section in ("normalized", "weighted"):
            values = dict(cleaned.get(section, {}))
            values.pop("area_guidance", None)
            cleaned[section] = values
        cleaned["large_plan_guidance_used"] = False
        return cleaned

    @staticmethod
    def _anchor_row(column) -> str:
        return str(
            next(
                row_no
                for bay_key, row_no, _quantity in column.row_allocation
                if bay_key == column.bay_key
            )
        )

    def _candidate_footprint_orders(self, index: int) -> tuple[int, ...]:
        column = self._columns[index]
        return tuple(
            sorted(
                int(self.bays[bay_key].bay_order)
                for bay_key, _row_no, _quantity in column.row_allocation
            )
        )

    def _split_contiguous_runs(
        self, candidate_indices: list[int]
    ) -> tuple[tuple[int, ...], ...]:
        ordered = sorted(
            candidate_indices,
            key=lambda index: self._candidate_footprint_orders(index),
        )
        runs: list[list[int]] = []
        for index in ordered:
            if not runs:
                runs.append([index])
                continue
            previous_orders = self._candidate_footprint_orders(runs[-1][-1])
            current_orders = self._candidate_footprint_orders(index)
            if current_orders[0] - previous_orders[-1] == 2:
                runs[-1].append(index)
            else:
                runs.append([index])
        return tuple(tuple(run) for run in runs)

    def _make_zone(
        self,
        group_id: str,
        area_no: str,
        row_no: str,
        candidate_indices: tuple[int, ...],
    ) -> ContiguousZone:
        group = self.groups_by_id[group_id]
        resources: set[Resource] = set()
        anchor_bay_loads: Counter[str] = Counter()
        bay_loads: Counter[str] = Counter()
        bay_size_loads: Counter[tuple[str, str]] = Counter()
        stack_uses: Counter[tuple[str, str]] = Counter()
        bay_attr_uses: Counter[tuple[str, str, str, str]] = Counter()
        capacity = 0
        for index in candidate_indices:
            column = self._columns[index]
            row_capacity = int(self._base_location_capacity(group, column))
            if row_capacity <= 0:
                raise RuntimeError(f"zone contains a zero-capacity candidate: {index}")
            capacity += row_capacity
            anchor_bay_loads[str(column.bay_key)] += row_capacity
            for bay_key, candidate_row, _quantity in column.row_allocation:
                resource = (str(bay_key), str(candidate_row))
                if resource in resources:
                    raise RuntimeError(
                        "a contiguous zone contains overlapping row footprints: "
                        f"group={group_id}, resource={resource}"
                    )
                resources.add(resource)
                bay_loads[str(bay_key)] += row_capacity
                stack_uses[(str(bay_key), str(column.size))] += 1
                for attr in self._bay_no_mix_attrs_for_column(column):
                    scope = self._attr_voyage_scope(attr, column.voyage_id)
                    value = self._column_attr_value(column, attr)
                    bay_attr_uses[(str(bay_key), attr, scope, value)] += 1
            bay_size_loads[(str(column.bay_key), str(column.size))] += row_capacity
        zone_penalty = self._zone_activation_penalty()
        return ContiguousZone(
            zone_id=-1,
            group_id=group_id,
            area_no=area_no,
            row_no=row_no,
            candidate_indices=candidate_indices,
            capacity=int(capacity),
            resources=tuple(sorted(resources)),
            anchor_bay_loads=tuple(sorted(anchor_bay_loads.items())),
            bay_loads=tuple(sorted(bay_loads.items())),
            bay_size_loads=tuple(sorted(bay_size_loads.items())),
            stack_uses=tuple(sorted(stack_uses.items())),
            bay_attr_uses=tuple(sorted(bay_attr_uses.items())),
            objective_cost=float(
                zone_penalty + self._unused_capacity_unit_cost() * capacity
            ),
        )

    def _unused_capacity_unit_cost(self) -> float:
        return (
            self._zone_objective_weights()["unused_capacity"]
            / self._zone_objective_scale("unused_capacity")
        )

    def _register_zone(
        self,
        strip_key: StripKey,
        candidate_indices: tuple[int, ...],
    ) -> int:
        existing = self._zone_id_by_signature.get(candidate_indices)
        if existing is not None:
            return existing
        group_id, area_no, row_no = strip_key
        zone_id = len(self._zones)
        zone = replace(
            self._make_zone(group_id, area_no, row_no, candidate_indices),
            zone_id=zone_id,
        )
        self._zones.append(zone)
        self._zone_id_by_signature[candidate_indices] = zone_id
        self._zone_indices_by_group[group_id].append(zone_id)
        self._zone_indices_by_strip[strip_key].append(zone_id)
        return zone_id

    def _iter_zone_signatures(
        self,
        group_id: str | None = None,
    ):
        for strip_key, runs in self._strip_runs.items():
            if group_id is not None and strip_key[0] != group_id:
                continue
            capacity_limit = int(self._zone_capacity_limit_by_strip[strip_key])
            for run in runs:
                for start in range(len(run)):
                    running_capacity = 0
                    for end in range(start, len(run)):
                        running_capacity += int(self._atomic_capacity[run[end]])
                        if running_capacity > capacity_limit:
                            break
                        yield strip_key, tuple(run[start : end + 1])

    def _materialize_all_zones(self) -> None:
        for strip_key, signature in self._iter_zone_signatures():
            self._register_zone(strip_key, signature)

    def _prepare_zones(self) -> dict[str, float | int]:
        started = perf_counter()
        self._prepare_master_index_sets()
        self._prepare_objective_normalization()
        self._initialize_location_pool()
        strip_candidates: defaultdict[StripKey, list[int]] = defaultdict(list)
        for group in self.groups:
            for candidate in self._base_placements_for_group(group):
                index = self._append_generated_column(candidate)
                strip_candidates[
                    (group.group_id, candidate.area_no, self._anchor_row(candidate))
                ].append(index)

        self._atomic_capacity = []
        self._atomic_attr_keys = []
        self._atomic_resources = []
        for column in self._columns:
            group = self.groups_by_id[column.group_id]
            self._atomic_capacity.append(
                int(self._base_location_capacity(group, column))
            )
            attr_keys = []
            resources = []
            for bay_key, row_no, _quantity in column.row_allocation:
                resources.append((str(bay_key), str(row_no)))
                for attr in self._bay_no_mix_attrs_for_column(column):
                    attr_keys.append(
                        (
                            str(bay_key),
                            attr,
                            self._attr_voyage_scope(attr, column.voyage_id),
                            self._column_attr_value(column, attr),
                        )
                    )
            self._atomic_attr_keys.append(tuple(attr_keys))
            self._atomic_resources.append(tuple(resources))

        candidate_areas_by_group: defaultdict[str, set[str]] = defaultdict(set)
        candidate_resources_by_group: defaultdict[
            str, set[Resource]
        ] = defaultdict(set)
        for index, column in enumerate(self._columns):
            candidate_areas_by_group[column.group_id].add(str(column.area_no))
            candidate_resources_by_group[column.group_id].update(
                self._atomic_resources[index]
            )
        self._candidate_areas_by_group = {
            group.group_id: frozenset(
                candidate_areas_by_group[group.group_id]
            )
            for group in self.groups
        }
        self._candidate_resources_by_group = {
            group.group_id: frozenset(
                candidate_resources_by_group[group.group_id]
            )
            for group in self.groups
        }

        self._zones = []
        self._zone_id_by_signature.clear()
        self._zone_indices_by_group.clear()
        self._zone_indices_by_strip.clear()
        self._strip_runs.clear()
        self._zone_capacity_limit_by_strip.clear()
        excluded_by_capacity_rule_count = 0
        possible_zone_count = 0
        possible_zone_count_by_group: Counter[str] = Counter()
        for strip_key, indices in sorted(strip_candidates.items()):
            group_id, _area_no, _row_no = strip_key
            runs = self._split_contiguous_runs(indices)
            self._strip_runs[strip_key] = runs
            maximum_row_capacity = max(
                (self._atomic_capacity[index] for index in indices),
                default=0,
            )
            self._zone_capacity_limit_by_strip[strip_key] = (
                int(self.groups_by_id[group_id].demand)
                + int(maximum_row_capacity)
            )
            for run in runs:
                for start in range(len(run)):
                    running_capacity = 0
                    for end in range(start, len(run)):
                        running_capacity += int(self._atomic_capacity[run[end]])
                        if running_capacity > int(
                            self._zone_capacity_limit_by_strip[strip_key]
                        ):
                            excluded_by_capacity_rule_count += len(run) - end
                            break
                        possible_zone_count += 1
                        possible_zone_count_by_group[group_id] += 1
        self._possible_zone_count = possible_zone_count
        self._possible_zone_count_by_group = possible_zone_count_by_group
        if self._possible_zone_count <= 0 and self.groups:
            raise RuntimeError("contiguous-zone model has no candidate zones")
        atomic_count_by_group: Counter[str] = Counter(
            column.group_id for column in self._columns
        )
        natural_zone_expansion = sum(
            max(
                0,
                min(
                    int(group.demand),
                    int(atomic_count_by_group[group.group_id]),
                )
                - 1,
            )
            for group in self.groups
        )
        areas_by_voyage: defaultdict[str, set[str]] = defaultdict(set)
        demand_by_voyage: Counter[str] = Counter()
        for group in self.groups:
            demand_by_voyage[group.voyage_id] += int(group.demand)
        for column in self._columns:
            areas_by_voyage[column.voyage_id].add(column.area_no)
        natural_voyage_area_expansion = sum(
            max(
                0,
                min(
                    int(demand),
                    len(areas_by_voyage[voyage_id]),
                )
                - 1,
            )
            for voyage_id, demand in demand_by_voyage.items()
        )
        total_export_demand = sum(int(group.demand) for group in self.groups)
        self._zone_objective_scales = {
            "voyage_area_dispersion": float(
                max(1, natural_voyage_area_expansion)
            ),
            "zone_dispersion": float(max(1, natural_zone_expansion)),
            "existing_group_proximity": self._objective_scale(
                "existing_group_proximity"
            ),
            "unused_capacity": float(max(1, total_export_demand)),
            "berth_distance": self._objective_scale("berth_distance"),
        }
        self._prepare_peak_utilization_policy()
        return {
            "preparation_seconds": perf_counter() - started,
            "atomic_candidate_count": len(self._columns),
            "strip_count": len(strip_candidates),
            "zone_count": self._possible_zone_count,
            "materialized_zone_count": len(self._zones),
            "zone_candidate_policy": (
                "contiguous_intervals_with_one_atomic_row_capacity_slack"
            ),
            "excluded_by_zone_capacity_rule_count": (
                excluded_by_capacity_rule_count
            ),
            "objective_scales": dict(self._zone_objective_scales),
            "peak_utilization_policy": dict(self._peak_utilization_policy),
        }

    @staticmethod
    def _container_slot_units(size: str) -> int:
        """Return the physical 20-foot-bay footprints used by one box."""

        return 2 if str(size) in {"40", "45"} else 1

    def _prepare_peak_utilization_policy(self) -> None:
        """Build a reproducible epsilon cap from the instance workload.

        The lower bound is the strongest of several aggregate load/capacity
        ratios (whole instance, export/import separately, voyage, group and
        import flow-size).  The cap adds a declared fraction of the remaining
        distance to one.  It is a residual-capacity consumption proxy, not a
        claim about an unpublished terminal safety threshold.
        """

        area_capacity = {
            area_no: int(
                sum(
                    int(self.bays[bay_key].physical_capacity)
                    for bay_key in bay_keys
                )
            )
            for area_no, bay_keys in self.bays_by_area.items()
        }
        export_areas_by_group: defaultdict[str, set[str]] = defaultdict(set)
        export_areas_by_voyage: defaultdict[str, set[str]] = defaultdict(set)
        export_areas: set[str] = set()
        for column in self._columns:
            export_areas_by_group[column.group_id].add(column.area_no)
            export_areas_by_voyage[column.voyage_id].add(column.area_no)
            export_areas.add(column.area_no)
        import_areas_by_flow_size: defaultdict[tuple[str, str], set[str]] = (
            defaultdict(set)
        )
        import_areas: set[str] = set()
        for key, candidates in self.import_reservation_candidates.items():
            for bay_key, _capacity in candidates:
                area_no = self.bays[bay_key].area_no
                import_areas_by_flow_size[key].add(area_no)
                import_areas.add(area_no)

        def capacity_of(areas: set[str]) -> int:
            return int(sum(area_capacity.get(area, 0) for area in areas))

        export_slot_units = int(
            sum(
                int(group.demand) * self._container_slot_units(group.size)
                for group in self.groups
            )
        )
        import_slot_units = int(
            sum(
                int(quantity) * self._container_slot_units(size)
                for (_flow, size), quantity in self.import_total_by_flow_size.items()
            )
        )
        candidate_floors: dict[str, float] = {}

        def discrete_load_floor(load: int, areas: set[str]) -> float:
            capacities = [
                int(area_capacity.get(area, 0))
                for area in areas
                if int(area_capacity.get(area, 0)) > 0
            ]
            if not capacities or sum(capacities) < int(load):
                return math.inf
            lower = 0.0
            upper = 1.0
            for _iteration in range(60):
                midpoint = (lower + upper) / 2.0
                usable = sum(
                    int(math.floor(midpoint * capacity + 1e-12))
                    for capacity in capacities
                )
                if usable >= int(load):
                    upper = midpoint
                else:
                    lower = midpoint
            return min(1.0, upper + 1e-9)

        def add_floor(name: str, load: int, areas: set[str]) -> None:
            if load <= 0:
                return
            capacity = capacity_of(areas)
            if capacity <= 0:
                raise ValueError(
                    "positive planning load has no residual physical capacity: "
                    f"scope={name}, load={load}"
                )
            candidate_floors[name] = max(
                float(load) / float(capacity),
                discrete_load_floor(load, areas),
            )

        add_floor(
            "all_workload",
            export_slot_units + import_slot_units,
            export_areas | import_areas,
        )
        add_floor("all_exports", export_slot_units, export_areas)
        add_floor("all_imports", import_slot_units, import_areas)
        for voyage_id, areas in sorted(export_areas_by_voyage.items()):
            load = sum(
                int(group.demand) * self._container_slot_units(group.size)
                for group in self.groups
                if group.voyage_id == voyage_id
            )
            add_floor(f"export_voyage:{voyage_id}", load, areas)
        for group in self.groups:
            add_floor(
                f"export_group:{group.group_id}",
                int(group.demand) * self._container_slot_units(group.size),
                export_areas_by_group[group.group_id],
            )
        for (flow, size), quantity in sorted(
            self.import_total_by_flow_size.items()
        ):
            add_floor(
                f"import_flow_size:{flow}|{size}",
                int(quantity) * self._container_slot_units(size),
                import_areas_by_flow_size[(flow, size)],
            )

        lower_bound = max(candidate_floors.values(), default=0.0)
        if lower_bound > 1.0 + 1e-9:
            raise ValueError(
                "planned export/import workload exceeds reachable residual "
                f"capacity: utilization_lower_bound={lower_bound:.6f}"
            )
        lower_bound = min(1.0, max(0.0, lower_bound))
        headroom = float(
            self.zone_config.peak_utilization_headroom_fraction
        )
        cap = lower_bound + headroom * (1.0 - lower_bound)
        self._peak_utilization_policy = {
            "type": "data_derived_epsilon_constraint",
            "measure": "planned_slot_units_over_residual_physical_capacity",
            "load_lower_bound": float(lower_bound),
            "headroom_fraction": headroom,
            "epsilon_cap": float(min(1.0, cap)),
            "export_slot_units": export_slot_units,
            "anonymous_import_slot_units": import_slot_units,
            "reachable_area_capacity": capacity_of(
                export_areas | import_areas
            ),
            "area_capacity": dict(sorted(area_capacity.items())),
            "lower_bound_components": dict(sorted(candidate_floors.items())),
            "integer_area_capacity_breakpoints_included": True,
            "terminal_approved_threshold_used": False,
        }

    def _master_index_sets(self) -> dict[str, object]:
        physical_resources: set[Resource] = set()
        stack_keys: set[tuple[str, str]] = set()
        attr_keys: set[tuple[str, str, str, str]] = set()
        area_pairs: set[tuple[str, str]] = set()
        voyage_area_pairs: set[tuple[str, str]] = set()
        groups_by_voyage_area: defaultdict[tuple[str, str], set[str]] = (
            defaultdict(set)
        )
        atomic_count_by_area_pair: Counter[tuple[str, str]] = Counter()
        flow_columns: dict[tuple[str, str], object] = {}
        for column in self._columns:
            area_key = (str(column.group_id), str(column.area_no))
            area_pairs.add(area_key)
            voyage_area_key = (
                str(column.voyage_id),
                str(column.area_no),
            )
            voyage_area_pairs.add(voyage_area_key)
            groups_by_voyage_area[voyage_area_key].add(
                str(column.group_id)
            )
            atomic_count_by_area_pair[area_key] += 1
            flow_columns.setdefault(
                (str(column.group_id), str(column.bay_key)),
                column,
            )
            for bay_key, row_no, _quantity in column.row_allocation:
                physical_resources.add((str(bay_key), str(row_no)))
                stack_keys.add((str(bay_key), str(column.size)))
                for attr in self._bay_no_mix_attrs_for_column(column):
                    scope = self._attr_voyage_scope(attr, column.voyage_id)
                    value = self._column_attr_value(column, attr)
                    attr_keys.add((str(bay_key), attr, scope, value))
        physical_resources = sorted(physical_resources)
        stack_keys = sorted(stack_keys)
        attr_keys = sorted(attr_keys)
        attr_scopes: defaultdict[tuple[str, str, str], list] = defaultdict(list)
        for key in attr_keys:
            attr_scopes[key[:3]].append(key)
        return {
            "physical_resources": physical_resources,
            "stack_keys": stack_keys,
            "attr_keys": attr_keys,
            "attr_scopes": attr_scopes,
            "area_pairs": sorted(area_pairs),
            "voyage_area_pairs": sorted(voyage_area_pairs),
            "groups_by_voyage_area": groups_by_voyage_area,
            "atomic_count_by_area_pair": atomic_count_by_area_pair,
            "flow_columns": flow_columns,
        }

    def _build_zone_master(self):
        from gurobipy import quicksum

        model = GurobiModel("yard_contiguous_zone_rmp")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Method", int(self.config.lp_method))
        self._set_gurobi_param(model, "OptimalityTol", 1e-9)
        self._set_gurobi_param(model, "FeasibilityTol", 1e-9)
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(model, "Threads", int(self.config.solver_threads))
        model.setMinimize()
        sets = self._master_index_sets()
        zero = model.addVar(lb=0.0, ub=0.0, name="zone_zero")
        # Every positive-demand voyage uses at least one area and every group
        # uses at least one zone.  Remove those unavoidable activations so the
        # objective measures only extra dispersion.
        voyage_count = len({group.voyage_id for group in self.groups})
        dispersion_baseline = (
            voyage_count * self._voyage_area_activation_penalty()
            + len(self.groups) * self._zone_activation_penalty()
        )
        baseline = model.addVar(
            lb=1.0,
            ub=1.0,
            obj=-float(dispersion_baseline),
            name="zone_dispersion_baseline",
        )
        unused_unit_cost = self._unused_capacity_unit_cost()
        shortage = {
            group.group_id: model.addVar(
                lb=0.0,
                obj=float(self.zone_config.shortage_penalty),
                name=f"zone_short_{self._key_name((group.group_id,))}",
            )
            for group in self.groups
        }
        export_flow = {
            key: model.addVar(
                lb=0.0,
                ub=float(self.groups_by_id[key[0]].demand),
                vtype="C",
                obj=self._zone_flow_unit_cost(*key) - unused_unit_cost,
                name=f"zone_flow_{self._key_name(key)}",
            )
            for key, column in sorted(sets["flow_columns"].items())
        }
        area_use = {
            key: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="C",
                name=f"zone_group_area_{self._key_name(key)}",
            )
            for key in sets["area_pairs"]
        }
        voyage_area_use = {
            key: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="C",
                obj=self._voyage_area_activation_penalty(),
                name=f"zone_voyage_area_{self._key_name(key)}",
            )
            for key in sets["voyage_area_pairs"]
        }
        attr_state = {
            key: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="C",
                name=f"zone_attr_{self._key_name(key)}",
            )
            for key in sets["attr_keys"]
        }
        import_reserve = {
            (flow, size, bay_key): model.addVar(
                lb=0.0,
                ub=float(capacity),
                vtype="C",
                name=f"zone_import_{flow}_{size}_{self._key_name((bay_key,))}",
            )
            for (flow, size), candidates in sorted(
                self.import_reservation_candidates.items()
            )
            for bay_key, capacity in candidates
        }

        import_by_bay: defaultdict[str, list] = defaultdict(list)
        import_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        import_by_flow_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        import_slot_load_by_area: defaultdict[str, list[tuple[int, object]]] = (
            defaultdict(list)
        )
        for (flow, size, bay_key), variable in import_reserve.items():
            area_no = self.bays[bay_key].area_no
            footprint = self._placement_footprint_keys(bay_key, size)
            for footprint_key in footprint:
                import_by_bay[footprint_key].append(variable)
            import_by_bay_size[(bay_key, size)].append(variable)
            import_by_flow_size[(flow, size)].append(variable)
            import_slot_load_by_area[area_no].append(
                (len(footprint), variable)
            )

        flow_by_group: defaultdict[str, list] = defaultdict(list)
        flow_by_area: defaultdict[tuple[str, str], list] = defaultdict(list)
        export_slot_load_by_area: defaultdict[
            str, list[tuple[int, object]]
        ] = (
            defaultdict(list)
        )
        for key, variable in export_flow.items():
            group_id, bay_key = key
            column = sets["flow_columns"][key]
            flow_by_group[group_id].append(variable)
            flow_by_area[(group_id, column.area_no)].append(variable)
            export_slot_load_by_area[column.area_no].append(
                (
                    len(
                        self._placement_footprint_keys(
                            column.bay_key, column.size
                        )
                    ),
                    variable,
                )
            )

        constraints: dict[str, dict] = defaultdict(dict)
        for group in self.groups:
            constraints["group_balance"][group.group_id] = model.addConstr(
                quicksum(flow_by_group[group.group_id])
                + shortage[group.group_id]
                == int(group.demand),
                name=f"zone_group_{self._key_name((group.group_id,))}",
            )
        for key, variable in export_flow.items():
            constraints["flow_capacity"][key] = model.addConstr(
                variable <= zero,
                name=f"zone_flow_capacity_{self._key_name(key)}",
            )
        for resource in sets["physical_resources"]:
            constraints["physical_resource"][resource] = model.addConstr(
                zero <= 1.0,
                name=f"zone_resource_{self._key_name(resource)}",
            )
        for bay_key in sorted(self._master_bay_capacity_keys):
            constraints["bay_capacity"][bay_key] = model.addConstr(
                quicksum(import_by_bay.get(bay_key, []))
                <= int(self.bays[bay_key].physical_capacity),
                name=f"zone_bay_{self._key_name((bay_key,))}",
            )
        for key in sorted(self._master_bay_size_keys):
            bay_key, size = key
            constraints["bay_size"][key] = model.addConstr(
                quicksum(import_by_bay_size.get(key, []))
                <= int(self.bays[bay_key].cap_by_size.get(size, 0)),
                name=f"zone_size_{self._key_name(key)}",
            )
        for key in sets["stack_keys"]:
            constraints["stack_count"][key] = model.addConstr(
                zero <= int(self._stack_count_for_bay_size(*key)),
                name=f"zone_stack_{self._key_name(key)}",
            )
        for key in sets["attr_keys"]:
            bay_key = key[0]
            big_m = max(1, len(self.bays[bay_key].row_physical_capacity))
            constraints["attr_link"][key] = model.addConstr(
                zero <= big_m * attr_state[key],
                name=f"zone_attr_link_{self._key_name(key)}",
            )
            constraints["attr_presence"][key] = model.addConstr(
                attr_state[key] <= zero,
                name=f"zone_attr_presence_{self._key_name(key)}",
            )
        for scope, keys in sorted(sets["attr_scopes"].items()):
            constraints["attr_choice"][scope] = model.addConstr(
                quicksum(attr_state[key] for key in keys) <= 1.0,
                name=f"zone_attr_choice_{self._key_name(scope)}",
            )
        for key in sets["area_pairs"]:
            demand = int(self.groups_by_id[key[0]].demand)
            assigned = quicksum(flow_by_area[key])
            constraints["area_link"][key] = model.addConstr(
                assigned <= max(1, demand) * area_use[key],
                name=f"zone_area_link_{self._key_name(key)}",
            )
            constraints["area_presence"][key] = model.addConstr(
                area_use[key] <= assigned,
                name=f"zone_area_presence_{self._key_name(key)}",
            )
            constraints["area_zone_support"][key] = model.addConstr(
                area_use[key] <= zero,
                name=f"zone_area_support_{self._key_name(key)}",
            )
            constraints["area_zone_upper"][key] = model.addConstr(
                zero
                <= max(1, int(sets["atomic_count_by_area_pair"][key]))
                * area_use[key],
                name=f"zone_area_zone_upper_{self._key_name(key)}",
            )
            constraints["area_flow_cover"][key] = model.addConstr(
                assigned <= zero,
                name=f"zone_area_cover_{self._key_name(key)}",
            )
        for voyage_area_key in sets["voyage_area_pairs"]:
            voyage_id, area_no = voyage_area_key
            group_area_variables = [
                area_use[(group_id, area_no)]
                for group_id in sorted(
                    sets["groups_by_voyage_area"][voyage_area_key]
                )
            ]
            for group_id in sorted(
                sets["groups_by_voyage_area"][voyage_area_key]
            ):
                constraints["voyage_area_link"][(
                    group_id,
                    voyage_id,
                    area_no,
                )] = model.addConstr(
                    area_use[(group_id, area_no)]
                    <= voyage_area_use[voyage_area_key],
                    name=(
                        "zone_voyage_area_link_"
                        f"{self._key_name((group_id, voyage_id, area_no))}"
                    ),
                )
            constraints["voyage_area_presence"][voyage_area_key] = (
                model.addConstr(
                    voyage_area_use[voyage_area_key]
                    <= quicksum(group_area_variables),
                    name=(
                        "zone_voyage_area_presence_"
                        f"{self._key_name(voyage_area_key)}"
                    ),
                )
            )
        for key, required in sorted(self.import_total_by_flow_size.items()):
            constraints["import_total"][key] = model.addConstr(
                quicksum(import_by_flow_size.get(key, [])) == int(required),
                name=f"zone_import_total_{self._key_name(key)}",
            )
        peak_cap = float(self._peak_utilization_policy["epsilon_cap"])
        area_capacity = self._peak_utilization_policy["area_capacity"]
        peak_areas = sorted(
            set(export_slot_load_by_area) | set(import_slot_load_by_area)
        )
        for area_no in peak_areas:
            planned_slot_load = quicksum(
                coefficient * variable
                for coefficient, variable in (
                    export_slot_load_by_area.get(area_no, [])
                    + import_slot_load_by_area.get(area_no, [])
                )
            )
            constraints["peak_utilization"][area_no] = model.addConstr(
                planned_slot_load
                <= peak_cap * int(area_capacity.get(area_no, 0)),
                name=(
                    "zone_peak_utilization_"
                    f"{self._key_name((area_no,))}"
                ),
            )
        model.update()
        return model, {
            "baseline": baseline,
            "shortage": shortage,
            "export_flow": export_flow,
            "area_use": area_use,
            "voyage_area_use": voyage_area_use,
            "attr_state": attr_state,
            "import_reserve": import_reserve,
            "constraints": constraints,
            "zone": {},
            "active_zone_indices": set(),
        }

    def _zone_coefficients(
        self, zone: ContiguousZone
    ) -> tuple[tuple[str, object, float], ...]:
        coefficients: list[tuple[str, object, float]] = []
        area_key = (zone.group_id, zone.area_no)
        coefficients.append(("area_zone_support", area_key, -1.0))
        coefficients.append(("area_zone_upper", area_key, 1.0))
        coefficients.append(
            (
                "area_flow_cover",
                area_key,
                -float(
                    min(
                        int(zone.capacity),
                        int(self.groups_by_id[zone.group_id].demand),
                    )
                ),
            )
        )
        coefficients.extend(
            (
                "flow_capacity",
                (zone.group_id, bay_key),
                -float(value),
            )
            for bay_key, value in zone.anchor_bay_loads
        )
        coefficients.extend(
            ("physical_resource", resource, 1.0)
            for resource in zone.resources
        )
        coefficients.extend(
            ("bay_capacity", key, float(value))
            for key, value in zone.bay_loads
        )
        coefficients.extend(
            ("bay_size", key, float(value))
            for key, value in zone.bay_size_loads
        )
        coefficients.extend(
            ("stack_count", key, float(value))
            for key, value in zone.stack_uses
        )
        coefficients.extend(
            ("attr_link", key, float(value))
            for key, value in zone.bay_attr_uses
        )
        coefficients.extend(
            ("attr_presence", key, -float(value))
            for key, value in zone.bay_attr_uses
        )
        return tuple(coefficients)

    def _add_zone_variable(self, model, variables: dict, zone_index: int):
        if zone_index in variables["active_zone_indices"]:
            return variables["zone"][zone_index]
        zone = self._zones[zone_index]
        terms = []
        for section, key, coefficient in self._zone_coefficients(zone):
            row = variables["constraints"].get(section, {}).get(key)
            if row is None and section == "area_zone_upper":
                # This proof-only strengthening is removed before primal
                # integer search; newly injected primal columns then omit it.
                continue
            if row is None:
                raise RuntimeError(
                    f"zone coefficient has no master row: {section}, {key}"
                )
            terms.append((coefficient, row))
        variable = model.addPricedVar(
            terms,
            lb=0.0,
            ub=1.0,
            vtype="C",
            obj=float(zone.objective_cost),
            name=f"zone_{zone_index}",
        )
        variables["zone"][zone_index] = variable
        variables["active_zone_indices"].add(zone_index)
        return variable

    @staticmethod
    def _remove_proof_only_area_rows(model, variables: dict) -> int:
        rows = list(
            variables["constraints"].get("area_zone_upper", {}).values()
        )
        if rows:
            model.removeConstraints(rows)
            variables["constraints"].pop("area_zone_upper", None)
            model.update()
        return len(rows)

    def _dual_snapshot(self, model, constraints: dict) -> dict[tuple[str, object], float]:
        duals = {}
        for section in (
            "flow_capacity",
            "physical_resource",
            "bay_capacity",
            "bay_size",
            "stack_count",
            "attr_link",
            "attr_presence",
            "area_zone_support",
            "area_zone_upper",
            "area_flow_cover",
        ):
            for key, row in constraints.get(section, {}).items():
                duals[(section, key)] = float(model.getLinearDual(row))
        return duals

    def _atomic_zone_reduced_cost(
        self,
        candidate_index: int,
        duals: dict[tuple[str, object], float],
    ) -> float:
        column = self._columns[candidate_index]
        capacity = self._atomic_capacity[candidate_index]
        value = capacity * self._unused_capacity_unit_cost()
        value += capacity * float(
            duals.get(
                ("flow_capacity", (column.group_id, column.bay_key)),
                0.0,
            )
        )
        value -= capacity * float(
            duals.get(("bay_size", (column.bay_key, column.size)), 0.0)
        )
        for resource in self._atomic_resources[candidate_index]:
            bay_key, _row_no = resource
            value -= float(duals.get(("physical_resource", resource), 0.0))
            value -= capacity * float(
                duals.get(("bay_capacity", str(bay_key)), 0.0)
            )
            value -= float(
                duals.get(("stack_count", (str(bay_key), column.size)), 0.0)
            )
            for key in self._atomic_attr_keys[candidate_index]:
                if key[0] != bay_key:
                    continue
                value -= float(duals.get(("attr_link", key), 0.0))
                value += float(duals.get(("attr_presence", key), 0.0))
        return value

    def _price_zone_signatures(
        self,
        duals: dict[tuple[str, object], float],
        *,
        per_group_limit: int,
        active_zone_indices: set[int],
        improving_only: bool,
        forbidden_signatures: set[tuple[int, ...]] | None = None,
    ) -> dict[str, object]:
        """Return the exact top-K interval columns without enumerating all intervals.

        For a fixed start row, the reduced cost is a prefix-sum expression on
        either side of the demand breakpoint.  A range-minimum tree therefore
        identifies the best end row in each regime.  Splitting that range after
        each heap extraction enumerates interval columns in exact reduced-cost
        order and stops as soon as K columns (or the first nonnegative column)
        has been certified.
        """

        tolerance = float(self.zone_config.reduced_cost_tolerance)
        active_signatures = {
            self._zones[index].candidate_indices for index in active_zone_indices
        }
        excluded_signatures = active_signatures.union(forbidden_signatures or set())
        atomic_cost = {
            index: self._atomic_zone_reduced_cost(index, duals)
            for index in range(len(self._columns))
        }
        atomic_capacity = self._atomic_capacity
        group_heaps: defaultdict[str, list[tuple]] = defaultdict(list)
        minimum_reduced_cost = math.inf
        discovered_improving_zone_count = 0
        extracted_interval_count = 0
        initialized_interval_state_count = 0
        serial = 0

        def push_range(
            heap: list[tuple],
            *,
            strip_key: StripKey,
            run: tuple[int, ...],
            start: int,
            lower: int,
            upper: int,
            offset: float,
            tree: _RangeMinimumTree,
        ) -> None:
            nonlocal serial
            if lower > upper:
                return
            prefix_value, endpoint = tree.query(lower, upper + 1)
            heappush(
                heap,
                (
                    float(offset + prefix_value),
                    serial,
                    strip_key,
                    run,
                    start,
                    lower,
                    upper,
                    endpoint,
                    offset,
                    tree,
                ),
            )
            serial += 1

        for strip_key, runs in sorted(self._strip_runs.items()):
            group_id, area_no, _row_no = strip_key
            demand = int(self.groups_by_id[group_id].demand)
            capacity_limit = int(self._zone_capacity_limit_by_strip[strip_key])
            fixed_cost = self._zone_activation_penalty()
            fixed_cost += float(
                duals.get(
                    ("area_zone_support", (group_id, area_no)),
                    0.0,
                )
            )
            fixed_cost -= float(
                duals.get(
                    ("area_zone_upper", (group_id, area_no)),
                    0.0,
                )
            )
            area_cover_dual = float(
                duals.get(("area_flow_cover", (group_id, area_no)), 0.0)
            )
            heap = group_heaps[group_id]
            for run in runs:
                prefix_capacity = [0]
                prefix_cost = [0.0]
                prefix_cost_with_cover = [0.0]
                for index in run:
                    prefix_capacity.append(
                        prefix_capacity[-1] + int(atomic_capacity[index])
                    )
                    prefix_cost.append(
                        prefix_cost[-1] + float(atomic_cost[index])
                    )
                    prefix_cost_with_cover.append(
                        prefix_cost_with_cover[-1]
                        + float(atomic_cost[index])
                        + area_cover_dual * int(atomic_capacity[index])
                    )
                raw_tree = _RangeMinimumTree(prefix_cost)
                covered_tree = _RangeMinimumTree(prefix_cost_with_cover)
                for start in range(len(run)):
                    maximum_endpoint = (
                        bisect_right(
                            prefix_capacity,
                            prefix_capacity[start] + capacity_limit,
                            lo=start + 1,
                        )
                        - 1
                    )
                    demand_endpoint = (
                        bisect_right(
                            prefix_capacity,
                            prefix_capacity[start] + demand,
                            lo=start + 1,
                        )
                        - 1
                    )
                    lower_endpoint = start + 1
                    covered_upper = min(maximum_endpoint, demand_endpoint)
                    if lower_endpoint <= covered_upper:
                        push_range(
                            heap,
                            strip_key=strip_key,
                            run=run,
                            start=start,
                            lower=lower_endpoint,
                            upper=covered_upper,
                            offset=(
                                fixed_cost - prefix_cost_with_cover[start]
                            ),
                            tree=covered_tree,
                        )
                        initialized_interval_state_count += 1
                    excess_lower = max(lower_endpoint, demand_endpoint + 1)
                    if excess_lower <= maximum_endpoint:
                        push_range(
                            heap,
                            strip_key=strip_key,
                            run=run,
                            start=start,
                            lower=excess_lower,
                            upper=maximum_endpoint,
                            offset=(
                                fixed_cost
                                - prefix_cost[start]
                                + area_cover_dual * demand
                            ),
                            tree=raw_tree,
                        )
                        initialized_interval_state_count += 1

        selected: list[tuple[float, StripKey, tuple[int, ...]]] = []
        for group_id in sorted(group_heaps):
            heap = group_heaps[group_id]
            group_selected = 0
            while heap and group_selected < int(per_group_limit):
                (
                    reduced_cost,
                    _serial,
                    strip_key,
                    run,
                    start,
                    lower,
                    upper,
                    endpoint,
                    offset,
                    tree,
                ) = heappop(heap)
                push_range(
                    heap,
                    strip_key=strip_key,
                    run=run,
                    start=start,
                    lower=lower,
                    upper=endpoint - 1,
                    offset=offset,
                    tree=tree,
                )
                push_range(
                    heap,
                    strip_key=strip_key,
                    run=run,
                    start=start,
                    lower=endpoint + 1,
                    upper=upper,
                    offset=offset,
                    tree=tree,
                )
                signature = tuple(run[start:endpoint])
                if signature in excluded_signatures:
                    continue
                extracted_interval_count += 1
                minimum_reduced_cost = min(minimum_reduced_cost, reduced_cost)
                if reduced_cost < -tolerance:
                    discovered_improving_zone_count += 1
                if improving_only and reduced_cost >= -tolerance:
                    break
                selected.append((reduced_cost, strip_key, signature))
                group_selected += 1
        selected.sort(key=lambda item: (item[0], item[1], item[2]))
        return {
            "selected": selected,
            "minimum_reduced_cost": minimum_reduced_cost,
            # Pricing stops once the exact top-K set has been certified, so
            # this is intentionally a discovered count rather than an
            # expensive count over every improving interval.
            "improving_zone_count": discovered_improving_zone_count,
            "improving_zone_count_is_lower_bound": True,
            "evaluated_zone_count": extracted_interval_count,
            "initialized_interval_state_count": initialized_interval_state_count,
            "pricing_method": "exact_prefix_rmq_top_k",
        }

    def _add_priced_signatures(
        self,
        model,
        variables: dict,
        selected: list[tuple[float, StripKey, tuple[int, ...]]],
    ) -> list[tuple[float, int]]:
        added: list[tuple[float, int]] = []
        for reduced_cost, strip_key, signature in selected:
            zone_index = self._register_zone(strip_key, signature)
            if zone_index in variables["active_zone_indices"]:
                continue
            self._add_zone_variable(model, variables, zone_index)
            added.append((reduced_cost, zone_index))
        if added:
            model.update()
        return added

    def _run_root_generation(
        self,
        model,
        variables: dict,
        deadline: float,
    ) -> dict[str, object]:
        started = perf_counter()
        rounds = []
        closed = False
        minimum_reduced_cost = -math.inf
        last_root_objective = None
        last_root_shortage = None
        previous_pricing_objective = None
        stagnant_round_count = 0
        for iteration in range(1, int(self.zone_config.max_root_iterations) + 1):
            remaining = deadline - perf_counter()
            if remaining <= 1e-6:
                break
            self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
            model.optimize()
            status = self._gurobi_status_name(model)
            if status != "optimal":
                rounds.append(
                    {
                        "iteration": iteration,
                        "status": status,
                        "active_zone_count": len(variables["active_zone_indices"]),
                    }
                )
                break
            duals = self._dual_snapshot(model, variables["constraints"])
            shortage_value = sum(
                self._gurobi_value(model, variable)
                for variable in variables["shortage"].values()
            )
            last_root_objective = self._gurobi_objective_value(model)
            last_root_shortage = shortage_value
            variables["last_root_duals"] = duals
            variables["last_root_zone_values"] = {
                zone_index: self._gurobi_value(model, variable)
                for zone_index, variable in variables["zone"].items()
            }
            variables["last_root_import_values"] = {
                key: self._gurobi_value(model, variable)
                for key, variable in variables["import_reserve"].items()
            }
            variables["last_root_export_flow_values"] = {
                key: self._gurobi_value(model, variable)
                for key, variable in variables["export_flow"].items()
            }
            variables["last_root_area_values"] = {
                key: self._gurobi_value(model, variable)
                for key, variable in variables["area_use"].items()
            }
            variables["last_root_voyage_area_values"] = {
                key: self._gurobi_value(model, variable)
                for key, variable in variables["voyage_area_use"].items()
            }
            variables["last_root_attr_values"] = {
                key: self._gurobi_value(model, variable)
                for key, variable in variables["attr_state"].items()
            }
            average_zone_count = self._possible_zone_count / max(1, len(self.groups))
            phase_one_batch = max(
                int(self.zone_config.columns_per_group_per_round),
                min(24, int(math.ceil(math.sqrt(average_zone_count)))),
            )
            if shortage_value > 1e-6:
                batch = phase_one_batch
                stagnant_round_count = 0
            else:
                if (
                    previous_pricing_objective is not None
                    and abs(last_root_objective - previous_pricing_objective)
                    <= 1e-8 * max(1.0, abs(last_root_objective))
                ):
                    stagnant_round_count += 1
                else:
                    stagnant_round_count = 0
                batch = min(
                    phase_one_batch,
                    int(self.zone_config.columns_per_group_per_round)
                    * (2 ** min(stagnant_round_count, 3)),
                )
            previous_pricing_objective = last_root_objective
            pricing = self._price_zone_signatures(
                duals,
                per_group_limit=batch,
                active_zone_indices=variables["active_zone_indices"],
                improving_only=True,
            )
            minimum_reduced_cost = float(pricing["minimum_reduced_cost"])
            selected = pricing["selected"]
            rounds.append(
                {
                    "iteration": iteration,
                    "status": status,
                    "objective": self._gurobi_objective_value(model),
                    "shortage": shortage_value,
                    "minimum_reduced_cost": minimum_reduced_cost,
                    "improving_zone_count": pricing["improving_zone_count"],
                    "priced_interval_count": pricing["evaluated_zone_count"],
                    "pricing_interval_state_count": pricing[
                        "initialized_interval_state_count"
                    ],
                    "pricing_method": pricing["pricing_method"],
                    "added_zone_count": len(selected),
                    "columns_per_group_batch": batch,
                    "stagnant_round_count": stagnant_round_count,
                    "active_zone_count": len(variables["active_zone_indices"]),
                }
            )
            if not selected:
                closed = True
                break
            self._add_priced_signatures(model, variables, selected)
        root_objective = last_root_objective
        root_shortage = last_root_shortage
        if self._gurobi_solution_count(model) > 0:
            root_objective = self._gurobi_objective_value(model)
            root_shortage = sum(
                self._gurobi_value(model, variable)
                for variable in variables["shortage"].values()
            )
        if closed and self._gurobi_solution_count(model) > 0:
            variables["last_root_lp_warm_start"] = model.captureLpWarmStart()
        return {
            "closed": closed,
            "rounds": rounds,
            "root_objective": root_objective,
            "root_shortage": root_shortage,
            "minimum_reduced_cost": minimum_reduced_cost,
            "active_zone_count": len(variables["active_zone_indices"]),
            "positive_zone_count": sum(
                value > 1e-8
                for value in variables.get("last_root_zone_values", {}).values()
            ),
            "fractional_zone_count": sum(
                1e-8 < value < 1.0 - 1e-8
                for value in variables.get("last_root_zone_values", {}).values()
            ),
            "seconds": perf_counter() - started,
        }

    def _capture_root_snapshot(
        self,
        variables: dict,
        root: dict[str, object],
    ) -> RootSnapshot:
        """Detach the certified root state from the proof-model lifecycle."""

        if not root.get("closed") or root.get("root_objective") is None:
            raise RuntimeError("cannot snapshot an unclosed exact root")
        required = (
            "last_root_duals",
            "last_root_zone_values",
            "last_root_export_flow_values",
            "last_root_import_values",
            "last_root_area_values",
            "last_root_voyage_area_values",
            "last_root_attr_values",
        )
        missing = [key for key in required if key not in variables]
        if missing:
            raise RuntimeError(
                "closed root is missing snapshot state: " + ", ".join(missing)
            )
        return RootSnapshot(
            objective=float(root["root_objective"]),
            duals=dict(variables["last_root_duals"]),
            zone_values=dict(variables["last_root_zone_values"]),
            export_flow_values=dict(
                variables["last_root_export_flow_values"]
            ),
            import_values=dict(variables["last_root_import_values"]),
            area_values=dict(variables["last_root_area_values"]),
            voyage_area_values=dict(
                variables["last_root_voyage_area_values"]
            ),
            attr_values=dict(variables["last_root_attr_values"]),
            lp_warm_start=variables.get("last_root_lp_warm_start"),
            proof_zone_indices=frozenset(variables["active_zone_indices"]),
        )

    @staticmethod
    def _install_root_snapshot(
        variables: dict,
        snapshot: RootSnapshot,
    ) -> None:
        """Expose snapshot values through the existing start-helper contract."""

        variables["last_root_duals"] = dict(snapshot.duals)
        variables["last_root_zone_values"] = dict(snapshot.zone_values)
        variables["last_root_export_flow_values"] = dict(
            snapshot.export_flow_values
        )
        variables["last_root_import_values"] = dict(snapshot.import_values)
        variables["last_root_area_values"] = dict(snapshot.area_values)
        variables["last_root_voyage_area_values"] = dict(
            snapshot.voyage_area_values
        )
        variables["last_root_attr_values"] = dict(snapshot.attr_values)
        variables["last_root_lp_warm_start"] = snapshot.lp_warm_start

    @staticmethod
    def _spatial_diversity_ranking(
        records: list[
            tuple[StripKey, tuple[int, ...], int, float]
        ],
    ) -> list[tuple[int, ...]]:
        """Interleave strong candidates across areas and physical strips."""

        by_strip: defaultdict[
            StripKey,
            list[tuple[StripKey, tuple[int, ...], int, float]],
        ] = defaultdict(list)
        for record in records:
            by_strip[record[0]].append(record)
        for strip_records in by_strip.values():
            strip_records.sort(
                key=lambda item: (
                    item[3] / max(1, item[2]),
                    -item[2],
                    len(item[1]),
                    item[1],
                )
            )
        strips_by_area: defaultdict[str, list[StripKey]] = defaultdict(list)
        for strip_key in sorted(by_strip):
            strips_by_area[strip_key[1]].append(strip_key)
        strip_order: list[StripKey] = []
        maximum_strip_count = max(
            (len(strips) for strips in strips_by_area.values()),
            default=0,
        )
        for position in range(maximum_strip_count):
            for area_no in sorted(strips_by_area):
                strips = strips_by_area[area_no]
                if position < len(strips):
                    strip_order.append(strips[position])
        ranking: list[tuple[int, ...]] = []
        maximum_depth = max(
            (len(by_strip[strip_key]) for strip_key in strip_order),
            default=0,
        )
        for depth in range(maximum_depth):
            for strip_key in strip_order:
                candidates = by_strip[strip_key]
                if depth < len(candidates):
                    ranking.append(candidates[depth][1])
        return ranking

    def _build_integrality_aware_primal_pool(
        self,
        snapshot: RootSnapshot,
    ) -> tuple[
        set[int],
        defaultdict[int, set[str]],
        dict[str, object],
    ]:
        """Compress proof columns into a diversified integer-search pool."""

        started = perf_counter()
        requested_limit = int(self.zone_config.integer_pool_columns_per_group)
        records_by_group: defaultdict[
            str,
            list[tuple[StripKey, tuple[int, ...], int, float]],
        ] = defaultdict(list)
        record_by_signature: dict[
            tuple[int, ...],
            tuple[StripKey, tuple[int, ...], int, float],
        ] = {}
        for strip_key, signature in self._iter_zone_signatures():
            capacity = sum(
                int(self._atomic_capacity[index]) for index in signature
            )
            record = (
                strip_key,
                signature,
                int(capacity),
                float(
                    self._zone_activation_penalty()
                    + self._unused_capacity_unit_cost() * capacity
                ),
            )
            records_by_group[strip_key[0]].append(record)
            record_by_signature[signature] = record

        budgets = {
            group.group_id: min(
                requested_limit,
                max(
                    1,
                    int(
                        math.ceil(
                            math.sqrt(
                                self._possible_zone_count_by_group[
                                    group.group_id
                                ]
                            )
                        )
                    ),
                ),
            )
            for group in self.groups
        }
        maximum_budget = max(budgets.values(), default=0)
        reduced_cost_by_group: defaultdict[str, list[tuple[int, ...]]] = (
            defaultdict(list)
        )
        pricing: dict[str, object] = {
            "selected": [],
            "evaluated_zone_count": 0,
            "pricing_method": "not_executed",
        }
        if maximum_budget > 0:
            pricing = self._price_zone_signatures(
                snapshot.duals,
                per_group_limit=maximum_budget,
                active_zone_indices=set(),
                improving_only=False,
            )
            for _reduced_cost, strip_key, signature in pricing["selected"]:
                reduced_cost_by_group[strip_key[0]].append(signature)

        proof_indices_by_group: defaultdict[str, list[int]] = defaultdict(list)
        for zone_index in snapshot.proof_zone_indices:
            proof_indices_by_group[self._zones[zone_index].group_id].append(
                zone_index
            )

        selected_indices: set[int] = set()
        provenance: defaultdict[int, set[str]] = defaultdict(set)
        per_group_diagnostics: dict[str, object] = {}
        channel_names = (
            "root_support",
            "reduced_cost",
            "capacity_fit",
            "business_efficiency",
            "spatial_diversity",
        )
        for group in self.groups:
            group_id = group.group_id
            records = records_by_group[group_id]
            demand = int(group.demand)
            root_ranking = [
                self._zones[index].candidate_indices
                for index in sorted(
                    proof_indices_by_group[group_id],
                    key=lambda index: (
                        -float(snapshot.zone_values.get(index, 0.0)),
                        self._zones[index].objective_cost
                        / max(1, self._zones[index].capacity),
                        self._zones[index].candidate_indices,
                    ),
                )
                if float(snapshot.zone_values.get(index, 0.0)) > 1e-8
            ]
            capacity_ranking = [
                item[1]
                for item in sorted(
                    records,
                    key=lambda item: (
                        0 if item[2] >= demand else 1,
                        abs(item[2] - demand),
                        item[3] / max(1, item[2]),
                        len(item[1]),
                        item[0],
                        item[1],
                    ),
                )
            ]
            business_ranking = [
                item[1]
                for item in sorted(
                    records,
                    key=lambda item: (
                        item[3] / max(1, item[2]),
                        max(0, item[2] - demand),
                        abs(item[2] - demand),
                        item[0],
                        item[1],
                    ),
                )
            ]
            channels = {
                "root_support": root_ranking,
                "reduced_cost": list(reduced_cost_by_group[group_id]),
                "capacity_fit": capacity_ranking,
                "business_efficiency": business_ranking,
                "spatial_diversity": self._spatial_diversity_ranking(records),
            }
            positions = {name: 0 for name in channel_names}
            selected_signatures: set[tuple[int, ...]] = set()
            signature_origins: defaultdict[tuple[int, ...], set[str]] = (
                defaultdict(set)
            )
            budget = int(budgets[group_id])
            while len(selected_signatures) < budget:
                progressed = False
                for channel_name in channel_names:
                    if len(selected_signatures) >= budget:
                        break
                    ranking = channels[channel_name]
                    position = positions[channel_name]
                    while position < len(ranking):
                        signature = ranking[position]
                        position += 1
                        if signature in selected_signatures:
                            signature_origins[signature].add(channel_name)
                            continue
                        selected_signatures.add(signature)
                        signature_origins[signature].add(channel_name)
                        progressed = True
                        break
                    positions[channel_name] = position
                if not progressed:
                    break

            for signature in sorted(selected_signatures):
                strip_key = record_by_signature[signature][0]
                zone_index = self._register_zone(strip_key, signature)
                selected_indices.add(zone_index)
                provenance[zone_index].update(signature_origins[signature])
            per_group_diagnostics[group_id] = {
                "possible_zone_count": int(
                    self._possible_zone_count_by_group[group_id]
                ),
                "nominal_budget": budget,
                "selected_zone_count": len(selected_signatures),
                "channel_candidate_counts": {
                    name: len(channels[name]) for name in channel_names
                },
                "selected_by_origin": {
                    name: sum(
                        name in origins
                        for origins in signature_origins.values()
                    )
                    for name in channel_names
                },
            }

        base_origin_counts = {
            origin: sum(
                origin in provenance[zone_index]
                for zone_index in selected_indices
            )
            for origin in channel_names
        }
        return selected_indices, provenance, {
            "policy": "group_specific_round_robin_diversified_columns",
            "proof_primal_pool_separated": True,
            "proof_pool_zone_count": len(snapshot.proof_zone_indices),
            "implicit_zone_count": int(self._possible_zone_count),
            "requested_columns_per_group_limit": requested_limit,
            "group_budget_policy": (
                "min(integer_pool_columns_per_group,"
                "ceil(sqrt(possible_zone_count_by_group)))"
            ),
            "channel_order": list(channel_names),
            "base_primal_pool_zone_count": len(selected_indices),
            "base_columns_by_origin": base_origin_counts,
            "per_group": per_group_diagnostics,
            "reduced_cost_pricing_method": pricing["pricing_method"],
            "reduced_cost_priced_interval_count": pricing[
                "evaluated_zone_count"
            ],
            "mandatory_start_policy": "retain_all_submitted_repaired_start_zones",
            "deterministic": True,
            "build_seconds": perf_counter() - started,
        }

    def _finalize_primal_pool_diagnostics(
        self,
        diagnostics: dict[str, object],
        provenance: defaultdict[int, set[str]],
        base_pool_indices: set[int],
        primal_pool_indices: set[int],
        mandatory_start_indices: set[int],
    ) -> None:
        """Add mandatory starts and summarize the actual integer pool."""

        for zone_index in mandatory_start_indices:
            provenance[zone_index].add("greedy_start")
        budgets = {
            group_id: int(values["nominal_budget"])
            for group_id, values in diagnostics["per_group"].items()
        }
        actual_by_group: Counter[str] = Counter(
            self._zones[index].group_id for index in primal_pool_indices
        )
        mandatory_by_group: Counter[str] = Counter(
            self._zones[index].group_id for index in mandatory_start_indices
        )
        overflow_by_group = {
            group_id: max(0, int(actual_by_group[group_id]) - budget)
            for group_id, budget in budgets.items()
        }
        for group_id, values in diagnostics["per_group"].items():
            values["actual_zone_count_after_mandatory"] = int(
                actual_by_group[group_id]
            )
            values["mandatory_start_zone_count"] = int(
                mandatory_by_group[group_id]
            )
            values["budget_overflow"] = overflow_by_group[group_id] > 0
            values["mandatory_overflow_count"] = int(
                overflow_by_group[group_id]
            )
        origins = sorted(
            {
                origin
                for index in primal_pool_indices
                for origin in provenance[index]
            }
        )
        proof_count = int(diagnostics["proof_pool_zone_count"])
        primal_count = len(primal_pool_indices)
        diagnostics.update(
            {
                "primal_pool_zone_count": primal_count,
                "mandatory_start_zone_count": len(mandatory_start_indices),
                "mandatory_added_zone_count": len(
                    mandatory_start_indices - base_pool_indices
                ),
                "budget_overflow": any(overflow_by_group.values()),
                "budget_overflow_group_count": sum(
                    value > 0 for value in overflow_by_group.values()
                ),
                "mandatory_overflow_count": sum(overflow_by_group.values()),
                "primal_pool_columns_by_origin": {
                    origin: sum(
                        origin in provenance[index]
                        for index in primal_pool_indices
                    )
                    for origin in origins
                },
                "primal_pool_zone_origins": {
                    str(index): sorted(provenance[index])
                    for index in sorted(primal_pool_indices)
                },
                "proof_to_primal_reduction_fraction": (
                    1.0 - primal_count / proof_count
                    if proof_count > 0
                    else None
                ),
                "primal_to_proof_ratio": (
                    primal_count / proof_count if proof_count > 0 else None
                ),
                "primal_pool_less_than_proof_pool": (
                    primal_count < proof_count
                ),
            }
        )

    def _enrich_integer_pool(self, model, variables: dict) -> dict[str, object]:
        requested_limit = int(self.zone_config.integer_pool_columns_per_group)
        average_zone_count = self._possible_zone_count / max(1, len(self.groups))
        limit = min(
            requested_limit,
            max(1, int(math.ceil(math.sqrt(average_zone_count)))),
        )
        if limit <= 0:
            return {
                "added_zone_count": 0,
                "requested_columns_per_group_limit": requested_limit,
                "effective_columns_per_group_limit": limit,
            }
        if self._gurobi_solution_count(model) > 0:
            duals = self._dual_snapshot(model, variables["constraints"])
        else:
            duals = variables.get("last_root_duals")
        if duals is None:
            return {
                "added_zone_count": 0,
                "requested_columns_per_group_limit": requested_limit,
                "effective_columns_per_group_limit": limit,
                "reason": "no_valid_root_duals",
            }
        pricing = self._price_zone_signatures(
            duals,
            per_group_limit=limit,
            active_zone_indices=variables["active_zone_indices"],
            improving_only=False,
        )
        selected = pricing["selected"]
        added = self._add_priced_signatures(model, variables, selected)
        return {
            "added_zone_count": len(added),
            "requested_columns_per_group_limit": requested_limit,
            "effective_columns_per_group_limit": limit,
            "limit_policy": "min_requested_and_ceil_sqrt_average_zone_count",
            "largest_added_reduced_cost": max(
                (value for value, _zone_index in added), default=None
            ),
            "priced_interval_count": pricing["evaluated_zone_count"],
            "active_zone_count_after_enrichment": len(
                variables["active_zone_indices"]
            ),
        }

    def _on_demand_greedy_support(
        self,
        variables: dict,
    ) -> tuple[list[tuple[StripKey, tuple[int, ...]]], dict[str, int], float | None]:
        """Return the best support only after every configured policy is tried."""

        candidates, diagnostics = self._generate_greedy_support_candidates(variables)
        self._last_greedy_candidate_diagnostics = diagnostics
        if not candidates:
            return [], {}, None
        best = candidates[0]
        return (
            list(best["support"]),
            dict(best["covered"]),
            float(best["import_protection"]),
        )

    def _generate_greedy_support_candidates(
        self,
        variables: dict,
        *,
        deadline: float | None = None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        """Enumerate Phase-1 strategies, with optional deadline interruption."""

        context = self._greedy_candidate_context(variables)
        generated: list[dict[str, object]] = []
        generated_summaries: list[dict[str, object]] = []
        interrupted_count = 0
        for policy, ordering, protection in self._greedy_candidate_strategies(
            context["group_ids"]
        ):
            candidate, summary = self._generate_greedy_support_candidate(
                context,
                ordering_policy=policy,
                group_order=ordering,
                import_protection=protection,
                deadline=deadline,
            )
            generated_summaries.append(summary)
            if candidate is None:
                interrupted_count += 1
                break
            generated.append(candidate)

        deduplicated: dict[tuple, dict[str, object]] = {}
        duplicate_count = 0
        for candidate in generated:
            key = candidate["support_key"]
            prior = deduplicated.get(key)
            if prior is None:
                deduplicated[key] = candidate
            else:
                duplicate_count += 1
                if candidate["cheap_score"] < prior["cheap_score"]:
                    deduplicated[key] = candidate
        candidates = sorted(
            deduplicated.values(),
            key=lambda candidate: candidate["cheap_score"],
        )
        diagnostics = {
            "generated_candidate_count": len(generated),
            "deduplicated_candidate_count": len(candidates),
            "complete_candidate_count": sum(
                bool(candidate["complete_export_cover"])
                for candidate in candidates
            ),
            "infeasible_candidate_count": sum(
                candidate["generation_status"] == "infeasible"
                for candidate in generated
            ),
            "budget_interrupted_candidate_count": interrupted_count,
            "candidate_generation_completed_count": len(generated),
            "candidate_generation_interrupted_count": interrupted_count,
            "candidate_duplicate_count": duplicate_count,
            "candidate_summaries": [
                self._greedy_candidate_summary(candidate)
                for candidate in candidates
            ],
            "generated_candidate_summaries": generated_summaries,
        }
        return candidates, diagnostics

    def _greedy_candidate_context(self, variables: dict) -> dict[str, object]:
        """Prepare immutable data shared by deterministic greedy strategies."""

        import_values = variables.get("last_root_import_values", {})
        import_by_bay: Counter[str] = Counter()
        import_by_bay_size: Counter[tuple[str, str]] = Counter()
        import_slot_load_by_area: Counter[str] = Counter()
        for (_flow, size, bay_key), value in import_values.items():
            if value <= 1e-8:
                continue
            for footprint_key in self._placement_footprint_keys(bay_key, size):
                import_by_bay[str(footprint_key)] += float(value)
            import_by_bay_size[(str(bay_key), str(size))] += float(value)
            import_slot_load_by_area[self.bays[bay_key].area_no] += (
                float(value)
                * len(self._placement_footprint_keys(bay_key, size))
            )

        group_ids = [group.group_id for group in self.groups]
        total_demand = sum(int(self.groups_by_id[key].demand) for key in group_ids)
        return {
            "variables": variables,
            "root_zone_values": variables.get("last_root_zone_values", {}),
            "import_by_bay": import_by_bay,
            "import_by_bay_size": import_by_bay_size,
            "import_slot_load_by_area": import_slot_load_by_area,
            "group_ids": group_ids,
            "total_demand": total_demand,
            "unregistered_zone_cache": {},
            "zone_slot_load_cache": {},
        }

    def _greedy_candidate_orderings(
        self,
        group_ids: list[str],
        *,
        include_voyage_clustered: bool = True,
    ) -> list[tuple[str, list[str]]]:
        orderings: list[tuple[str, list[str]]] = [
            (
                "candidate_scarcity",
                sorted(
                    group_ids,
                    key=lambda key: (
                        self._possible_zone_count_by_group[key],
                        key,
                    ),
                ),
            ),
            (
                "demand_descending",
                sorted(
                    group_ids,
                    key=lambda key: (-int(self.groups_by_id[key].demand), key),
                ),
            ),
            ("group_id", sorted(group_ids)),
        ]
        if include_voyage_clustered:
            orderings.append((
                "voyage_clustered",
                sorted(
                    group_ids,
                    key=lambda key: (
                        self.groups_by_id[key].voyage_id,
                        self._possible_zone_count_by_group[key],
                        -int(self.groups_by_id[key].demand),
                        key,
                    ),
                ),
            ))
        return orderings

    def _greedy_candidate_strategies(
        self,
        group_ids: list[str],
    ) -> list[tuple[str, list[str], float]]:
        """Round-robin the unchanged Phase-1 strategies across orderings."""

        return [
            (policy, ordering, float(protection))
            for protection in (1.0, 0.75, 0.50, 0.25, 0.0)
            for policy, ordering in self._greedy_candidate_orderings(group_ids)
        ]

    def _greedy_zone_for(
        self,
        context: dict[str, object],
        strip_key: StripKey,
        signature: tuple[int, ...],
    ) -> ContiguousZone:
        zone_index = self._zone_id_by_signature.get(signature)
        if zone_index is not None:
            return self._zones[zone_index]
        cache = context["unregistered_zone_cache"]
        zone = cache.get(signature)
        if zone is None:
            zone = self._make_zone(*strip_key, signature)
            cache[signature] = zone
        return zone

    def _greedy_zone_slot_load(
        self,
        context: dict[str, object],
        zone: ContiguousZone,
    ) -> Counter[str]:
        cache = context["zone_slot_load_cache"]
        cached = cache.get(zone.candidate_indices)
        if cached is not None:
            return cached
        load: Counter[str] = Counter()
        group = self.groups_by_id[zone.group_id]
        for bay_key, value in zone.anchor_bay_loads:
            load[self.bays[bay_key].area_no] += int(value) * len(
                self._placement_footprint_keys(bay_key, group.size)
            )
        cache[zone.candidate_indices] = load
        return load

    def _attempt_greedy_support(
        self,
        context: dict[str, object],
        group_order: list[str],
        import_protection: float,
        *,
        deadline: float | None,
        seed_root_zones: bool = False,
        enforce_peak_cap: bool = True,
    ) -> tuple[
        list[tuple[StripKey, tuple[int, ...]]],
        dict[str, int],
        str,
    ]:
        """Build one support and distinguish infeasibility from interruption."""

        def budget_exhausted() -> bool:
            return deadline is not None and perf_counter() >= deadline

        chosen: list[tuple[StripKey, tuple[int, ...]]] = []
        chosen_signatures: set[tuple[int, ...]] = set()
        covered: Counter[str] = Counter()
        resources: set[Resource] = set()
        bay_load: Counter[str] = Counter(
            {
                key: import_protection * value
                for key, value in context["import_by_bay"].items()
            }
        )
        bay_size_load: Counter[tuple[str, str]] = Counter(
            {
                key: import_protection * value
                for key, value in context["import_by_bay_size"].items()
            }
        )
        stack_load: Counter[tuple[str, str]] = Counter()
        attr_value: dict[tuple[str, str, str], str] = {}
        area_slot_load: Counter[str] = Counter(
            {
                key: import_protection * value
                for key, value in context["import_slot_load_by_area"].items()
            }
        )

        def feasible(zone: ContiguousZone) -> bool:
            if any(resource in resources for resource in zone.resources):
                return False
            if any(
                bay_load[key] + int(value)
                > int(self.bays[key].physical_capacity) + 1e-7
                for key, value in zone.bay_loads
            ):
                return False
            if any(
                bay_size_load[key] + int(value)
                > int(self.bays[key[0]].cap_by_size.get(key[1], 0)) + 1e-7
                for key, value in zone.bay_size_loads
            ):
                return False
            if any(
                stack_load[key] + int(value)
                > int(self._stack_count_for_bay_size(*key))
                for key, value in zone.stack_uses
            ):
                return False
            if enforce_peak_cap:
                peak_cap = float(self._peak_utilization_policy["epsilon_cap"])
                area_capacity = self._peak_utilization_policy["area_capacity"]
                if any(
                    area_slot_load[area_no] + int(value)
                    > peak_cap * int(area_capacity.get(area_no, 0)) + 1e-7
                    for area_no, value in self._greedy_zone_slot_load(
                        context, zone
                    ).items()
                ):
                    return False
            return all(
                attr_value.get(key[:3], key[3]) == key[3]
                for key, _value in zone.bay_attr_uses
            )

        def add(strip_key: StripKey, zone: ContiguousZone) -> None:
            signature = zone.candidate_indices
            chosen.append((strip_key, signature))
            chosen_signatures.add(signature)
            covered[zone.group_id] += int(zone.capacity)
            resources.update(zone.resources)
            for key, value in zone.bay_loads:
                bay_load[key] += int(value)
            for key, value in zone.bay_size_loads:
                bay_size_load[key] += int(value)
            for key, value in zone.stack_uses:
                stack_load[key] += int(value)
            for key, _value in zone.bay_attr_uses:
                attr_value[key[:3]] = key[3]
            if enforce_peak_cap:
                area_slot_load.update(
                    self._greedy_zone_slot_load(context, zone)
                )

        if seed_root_zones:
            root_values = context["root_zone_values"]
            for zone_index in sorted(
                root_values,
                key=lambda index: -float(root_values.get(index, 0.0)),
            ):
                if budget_exhausted():
                    return chosen, dict(covered), "budget_exhausted"
                if root_values.get(zone_index, 0.0) <= 1e-8:
                    break
                zone = self._zones[zone_index]
                demand = int(self.groups_by_id[zone.group_id].demand)
                if covered[zone.group_id] < demand and feasible(zone):
                    add((zone.group_id, zone.area_no, zone.row_no), zone)

        for group_id in group_order:
            if budget_exhausted():
                return chosen, dict(covered), "budget_exhausted"
            demand = int(self.groups_by_id[group_id].demand)
            while covered[group_id] < demand:
                if budget_exhausted():
                    return chosen, dict(covered), "budget_exhausted"
                remaining = demand - covered[group_id]
                best = None
                best_key = None
                for strip_key, signature in self._iter_zone_signatures(group_id):
                    if budget_exhausted():
                        return chosen, dict(covered), "budget_exhausted"
                    if signature in chosen_signatures:
                        continue
                    zone = self._greedy_zone_for(context, strip_key, signature)
                    if not feasible(zone):
                        continue
                    key = (
                        0 if zone.capacity >= remaining else 1,
                        max(0, zone.capacity - remaining),
                        zone.objective_cost / max(1, zone.capacity),
                        -zone.capacity,
                        strip_key,
                        signature,
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best = (strip_key, zone)
                if best is None:
                    return chosen, dict(covered), "infeasible"
                add(*best)
        return chosen, dict(covered), "complete"

    def _generate_greedy_support_candidate(
        self,
        context: dict[str, object],
        *,
        ordering_policy: str,
        group_order: list[str],
        import_protection: float,
        deadline: float | None,
        seed_root_zones: bool = False,
        enforce_peak_cap: bool = True,
    ) -> tuple[dict[str, object] | None, dict[str, object]]:
        chosen, covered, status = self._attempt_greedy_support(
            context,
            group_order,
            import_protection,
            deadline=deadline,
            seed_root_zones=seed_root_zones,
            enforce_peak_cap=enforce_peak_cap,
        )
        group_ids = context["group_ids"]
        covered_boxes = sum(
            min(int(self.groups_by_id[key].demand), covered.get(key, 0))
            for key in group_ids
        )
        summary = {
            "ordering_policy": ordering_policy,
            "import_protection": float(import_protection),
            "covered_boxes": int(covered_boxes),
            "complete_export_cover": status == "complete",
            "selected_zone_count": len(chosen),
            "generation_status": status,
        }
        if status == "budget_exhausted":
            return None, summary

        support_cost = 0.0
        reserved_capacity = 0
        for strip_key, signature in chosen:
            if deadline is not None and perf_counter() >= deadline:
                summary["generation_status"] = "budget_exhausted"
                summary["complete_export_cover"] = False
                return None, summary
            zone = self._greedy_zone_for(context, strip_key, signature)
            support_cost += float(zone.objective_cost)
            reserved_capacity += int(zone.capacity)
        support_key = tuple(sorted(chosen))
        complete = status == "complete"
        candidate = {
            "support": chosen,
            "covered": covered,
            "support_key": support_key,
            "ordering_policy": ordering_policy,
            "import_protection": float(import_protection),
            "covered_boxes": int(covered_boxes),
            "complete_export_cover": complete,
            "generation_status": status,
            "selected_zone_count": len(chosen),
            "reserved_capacity": reserved_capacity,
            "cheap_support_cost": support_cost,
            "cheap_score": (
                0 if complete else 1,
                -covered_boxes,
                len(chosen),
                max(0, reserved_capacity - int(context["total_demand"])),
                support_cost,
                support_key,
            ),
        }
        return candidate, summary

    @staticmethod
    def _greedy_candidate_summary(candidate: dict[str, object]) -> dict[str, object]:
        return {
            "ordering_policy": candidate["ordering_policy"],
            "import_protection": candidate["import_protection"],
            "covered_boxes": candidate["covered_boxes"],
            "complete_export_cover": candidate["complete_export_cover"],
            "selected_zone_count": candidate["selected_zone_count"],
            "generation_status": candidate["generation_status"],
        }

    def _baseline_v4_on_demand_greedy_support(
        self,
        variables: dict,
    ) -> tuple[list[tuple[StripKey, tuple[int, ...]]], dict[str, int], float | None]:
        """Reproduce the pre-Phase-1 single-start strategy for the baseline."""

        context = self._greedy_candidate_context(variables)
        best_chosen: list[tuple[StripKey, tuple[int, ...]]] = []
        best_covered: dict[str, int] = {}
        best_protection: float | None = None
        orderings = self._greedy_candidate_orderings(
            context["group_ids"],
            include_voyage_clustered=False,
        )
        for protection in (1.0, 0.0):
            for _policy, ordering in orderings:
                chosen, covered, status = self._attempt_greedy_support(
                    context,
                    ordering,
                    protection,
                    deadline=None,
                    seed_root_zones=True,
                    enforce_peak_cap=False,
                )
                covered_boxes = sum(
                    min(
                        int(self.groups_by_id[key].demand),
                        covered.get(key, 0),
                    )
                    for key in context["group_ids"]
                )
                best_boxes = sum(
                    min(
                        int(self.groups_by_id[key].demand),
                        best_covered.get(key, 0),
                    )
                    for key in context["group_ids"]
                )
                if covered_boxes > best_boxes:
                    best_chosen = chosen
                    best_covered = covered
                    best_protection = protection
                if status == "complete":
                    return chosen, covered, protection
        return best_chosen, best_covered, best_protection

    def _export_flow_start_for_zones(
        self,
        selected_zone_indices: set[int],
        variables: dict,
    ) -> dict[tuple[str, str], int]:
        capacity: Counter[tuple[str, str]] = Counter()
        for zone_index in selected_zone_indices:
            zone = self._zones[zone_index]
            for bay_key, value in zone.anchor_bay_loads:
                capacity[(zone.group_id, bay_key)] += int(value)
        root_values = variables.get("last_root_export_flow_values", {})
        column_by_key = {
            (column.group_id, column.bay_key): column for column in self._columns
        }
        assigned: dict[tuple[str, str], int] = {}
        for group in self.groups:
            keys = [key for key in capacity if key[0] == group.group_id]
            for key in keys:
                assigned[key] = min(
                    int(capacity[key]),
                    max(0, int(math.floor(float(root_values.get(key, 0.0))))),
                )
            remaining = int(group.demand) - sum(assigned[key] for key in keys)
            while remaining > 0:
                available = [
                    key for key in keys if assigned[key] < int(capacity[key])
                ]
                if not available:
                    return {}
                key = min(
                    available,
                    key=lambda candidate: (
                        -max(
                            0.0,
                            float(root_values.get(candidate, 0.0))
                            - assigned[candidate],
                        ),
                        self._zone_flow_unit_cost(*candidate),
                        candidate,
                    ),
                )
                root_deficit = max(
                    0.0,
                    float(root_values.get(key, 0.0)) - assigned[key],
                )
                quantity = min(
                    remaining,
                    int(capacity[key]) - assigned[key],
                    max(1, int(math.ceil(root_deficit - 1e-9))),
                )
                assigned[key] += quantity
                remaining -= quantity
            if remaining > 0:
                return {}
        return {key: value for key, value in assigned.items() if value > 0}

    def _import_reservation_mip_start(self, variables: dict) -> dict:
        """Round the root import flow while preserving every size total."""

        root_values = variables.get("last_root_import_values", {})
        if not root_values:
            return {}
        result = {}
        keys_by_flow_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        for key in variables["import_reserve"]:
            keys_by_flow_size[key[:2]].append(key)
        for flow_size, keys in sorted(keys_by_flow_size.items()):
            required = int(self.import_total_by_flow_size[flow_size])
            for key in keys:
                variable = variables["import_reserve"][key]
                result[key] = min(
                    int(round(float(variable.UB))),
                    max(0, int(math.floor(float(root_values.get(key, 0.0))))),
                )
            remaining = required - sum(result[key] for key in keys)
            while remaining > 0:
                available = [
                    key
                    for key in keys
                    if result[key]
                    < int(round(float(variables["import_reserve"][key].UB)))
                ]
                if not available:
                    return {}
                key = min(
                    available,
                    key=lambda candidate: (
                        -(
                            float(root_values.get(candidate, 0.0))
                            - result[candidate]
                        ),
                        candidate,
                    ),
                )
                capacity = int(
                    round(float(variables["import_reserve"][key].UB))
                )
                quantity = min(remaining, capacity - result[key])
                result[key] += quantity
                remaining -= quantity
        return result

    def _apply_flow_state_mip_start(
        self,
        selected_zone_indices: set[int],
        export_flow_start: dict[tuple[str, str], int],
        variables: dict,
    ) -> dict[str, object]:
        for key, variable in variables["export_flow"].items():
            variable.Start = float(export_flow_start.get(key, 0))
        used_areas = {
            (group_id, self.bays[bay_key].area_no)
            for (group_id, bay_key), quantity in export_flow_start.items()
            if quantity > 0
        }
        for key, variable in variables["area_use"].items():
            variable.Start = 1.0 if key in used_areas else 0.0
        used_voyage_areas = {
            (
                self.groups_by_id[group_id].voyage_id,
                area_no,
            )
            for group_id, area_no in used_areas
        }
        for key, variable in variables["voyage_area_use"].items():
            variable.Start = 1.0 if key in used_voyage_areas else 0.0
        used_attr_states = {
            key
            for zone_index in selected_zone_indices
            for key, value in self._zones[zone_index].bay_attr_uses
            if value > 0
        }
        for key, variable in variables["attr_state"].items():
            variable.Start = 1.0 if key in used_attr_states else 0.0
        import_start = self._import_reservation_mip_start(variables)
        if import_start:
            for key, variable in variables["import_reserve"].items():
                variable.Start = float(import_start.get(key, 0))
        for variable in variables["shortage"].values():
            variable.Start = 0.0
        return {
            "positive_export_flow_start_count": sum(
                value > 0 for value in export_flow_start.values()
            ),
            "positive_import_flow_start_count": sum(
                value > 0 for value in import_start.values()
            ),
            "used_area_start_count": len(used_areas),
            "used_voyage_area_start_count": len(used_voyage_areas),
            "used_attribute_state_start_count": len(used_attr_states),
        }

    def _legacy_gurobi_repair_start(self, model, variables: dict) -> dict[str, object]:
        """Build a deterministic, LP-guided integer support for Gurobi repair.

        The start protects the root LP's anonymous import allocation while it
        packs whole export zones.  Gurobi is allowed to repair the deliberately
        partial start (absolute-deviation helper variables are left for
        Gurobi to complete).
        """

        chosen, covered, protection = (
            self._baseline_v4_on_demand_greedy_support(variables)
        )
        complete = all(
            covered.get(group.group_id, 0) >= int(group.demand)
            for group in self.groups
        )
        if complete:
            selected_indices = {
                self._register_zone(strip_key, signature)
                for strip_key, signature in chosen
            }
            added_seed_zone_count = 0
            for zone_index in sorted(selected_indices):
                if zone_index not in variables["active_zone_indices"]:
                    self._add_zone_variable(model, variables, zone_index)
                    added_seed_zone_count += 1
            if added_seed_zone_count:
                model.update()
            for zone_index, variable in variables["zone"].items():
                variable.Start = 1.0 if zone_index in selected_indices else 0.0
            export_flow_start = self._export_flow_start_for_zones(
                selected_indices,
                variables,
            )
            flow_start = self._apply_flow_state_mip_start(
                selected_indices,
                export_flow_start,
                variables,
            )
            return {
                "selected_zone_count": len(selected_indices),
                "covered_group_count": len(self.groups),
                "group_count": len(self.groups),
                "complete_export_cover": True,
                "import_protection_fraction": protection,
                "added_seed_zone_count": added_seed_zone_count,
                "on_demand_support": True,
                **flow_start,
            }

        all_zone_indices = list(range(len(self._zones)))
        root_values = variables.get("last_root_zone_values", {})
        import_values = variables.get("last_root_import_values", {})
        import_by_bay: Counter[str] = Counter()
        import_by_bay_size: Counter[tuple[str, str]] = Counter()
        for (_flow, size, bay_key), value in import_values.items():
            if value <= 1e-8:
                continue
            for footprint_key in self._placement_footprint_keys(bay_key, size):
                import_by_bay[str(footprint_key)] += float(value)
            import_by_bay_size[(str(bay_key), str(size))] += float(value)

        candidates_by_group: defaultdict[str, list[int]] = defaultdict(list)
        for zone_index in all_zone_indices:
            candidates_by_group[self._zones[zone_index].group_id].append(zone_index)

        def attempt(
            group_order: list[str], import_protection: float
        ) -> tuple[set[int], dict[str, int]]:
            selected: set[int] = set()
            covered: Counter[str] = Counter()
            resources: set[Resource] = set()
            bay_load: Counter[str] = Counter(
                {
                    key: import_protection * value
                    for key, value in import_by_bay.items()
                }
            )
            bay_size_load: Counter[tuple[str, str]] = Counter(
                {
                    key: import_protection * value
                    for key, value in import_by_bay_size.items()
                }
            )
            stack_load: Counter[tuple[str, str]] = Counter()
            attr_value: dict[tuple[str, str, str], str] = {}

            def feasible(zone: ContiguousZone) -> bool:
                if any(resource in resources for resource in zone.resources):
                    return False
                for bay_key, value in zone.bay_loads:
                    if (
                        bay_load[bay_key] + int(value)
                        > int(self.bays[bay_key].physical_capacity) + 1e-7
                    ):
                        return False
                for key, value in zone.bay_size_loads:
                    if (
                        bay_size_load[key] + int(value)
                        > int(self.bays[key[0]].cap_by_size.get(key[1], 0)) + 1e-7
                    ):
                        return False
                for key, value in zone.stack_uses:
                    if (
                        stack_load[key] + int(value)
                        > int(self._stack_count_for_bay_size(*key))
                    ):
                        return False
                for key, _value in zone.bay_attr_uses:
                    scope = key[:3]
                    prior = attr_value.get(scope)
                    if prior is not None and prior != key[3]:
                        return False
                return True

            def add(zone_index: int) -> None:
                zone = self._zones[zone_index]
                selected.add(zone_index)
                covered[zone.group_id] += int(zone.capacity)
                resources.update(zone.resources)
                for key, value in zone.bay_loads:
                    bay_load[key] += int(value)
                for key, value in zone.bay_size_loads:
                    bay_size_load[key] += int(value)
                for key, value in zone.stack_uses:
                    stack_load[key] += int(value)
                for key, _value in zone.bay_attr_uses:
                    attr_value[key[:3]] = key[3]

            # Preserve the strongest compatible pieces of the fractional root.
            for zone_index in sorted(
                all_zone_indices,
                key=lambda index: (
                    -float(root_values.get(index, 0.0)),
                    self._zones[index].objective_cost
                    / max(1, self._zones[index].capacity),
                ),
            ):
                zone = self._zones[zone_index]
                if root_values.get(zone_index, 0.0) <= 1e-8:
                    break
                demand = int(self.groups_by_id[zone.group_id].demand)
                if covered[zone.group_id] < demand and feasible(zone):
                    add(zone_index)

            for group_id in group_order:
                demand = int(self.groups_by_id[group_id].demand)
                while covered[group_id] < demand:
                    remaining = demand - covered[group_id]
                    feasible_indices = [
                        index
                        for index in candidates_by_group[group_id]
                        if index not in selected and feasible(self._zones[index])
                    ]
                    if not feasible_indices:
                        return selected, dict(covered)
                    zone_index = min(
                        feasible_indices,
                        key=lambda index: (
                            0 if self._zones[index].capacity >= remaining else 1,
                            max(0, self._zones[index].capacity - remaining),
                            -float(root_values.get(index, 0.0)),
                            self._zones[index].objective_cost
                            / max(1, self._zones[index].capacity),
                            -self._zones[index].capacity,
                        ),
                    )
                    add(zone_index)
            return selected, dict(covered)

        group_ids = [group.group_id for group in self.groups]
        orderings = [
            sorted(group_ids, key=lambda key: len(candidates_by_group[key])),
            sorted(group_ids, key=lambda key: -int(self.groups_by_id[key].demand)),
            sorted(group_ids),
        ]
        best_selected: set[int] = set()
        best_covered: dict[str, int] = {}
        used_import_protection = None
        for import_protection in (1.0, 0.0):
            for ordering in orderings:
                selected, covered = attempt(ordering, import_protection)
                if sum(
                    min(int(self.groups_by_id[key].demand), covered.get(key, 0))
                    for key in group_ids
                ) > sum(
                    min(
                        int(self.groups_by_id[key].demand),
                        best_covered.get(key, 0),
                    )
                    for key in group_ids
                ):
                    best_selected, best_covered = selected, covered
                    used_import_protection = import_protection
                if all(
                    covered.get(key, 0) >= int(self.groups_by_id[key].demand)
                    for key in group_ids
                ):
                    best_selected, best_covered = selected, covered
                    used_import_protection = import_protection
                    break
            if len(best_covered) == len(group_ids) and all(
                best_covered.get(key, 0) >= int(self.groups_by_id[key].demand)
                for key in group_ids
            ):
                break

        added_seed_zone_count = 0
        for zone_index in sorted(best_selected):
            if zone_index not in variables["active_zone_indices"]:
                self._add_zone_variable(model, variables, zone_index)
                added_seed_zone_count += 1
        if added_seed_zone_count:
            model.update()

        for zone_index, variable in variables["zone"].items():
            variable.Start = 1.0 if zone_index in best_selected else 0.0
        export_flow_start = self._export_flow_start_for_zones(
            best_selected,
            variables,
        )
        flow_start = self._apply_flow_state_mip_start(
            best_selected,
            export_flow_start,
            variables,
        )
        return {
            "selected_zone_count": len(best_selected),
            "covered_group_count": sum(
                best_covered.get(key, 0) >= int(self.groups_by_id[key].demand)
                for key in group_ids
            ),
            "group_count": len(group_ids),
            "complete_export_cover": all(
                best_covered.get(key, 0) >= int(self.groups_by_id[key].demand)
                for key in group_ids
            ),
            "import_protection_fraction": used_import_protection,
            "added_seed_zone_count": added_seed_zone_count,
            **flow_start,
        }

    def _repair_zone_support(
        self,
        selected_zone_indices: set[int],
        deadline: float,
        start_source_variables: dict | None = None,
    ) -> dict[str, object]:
        """Jointly repair export flow, anonymous import, activations, and peak load."""

        started = perf_counter()
        if not selected_zone_indices:
            return {
                "status": "empty_support",
                "feasible": False,
                "seconds": 0.0,
            }
        model, variables = self._build_zone_master()
        try:
            self._remove_proof_only_area_rows(model, variables)
            for zone_index in sorted(selected_zone_indices):
                self._add_zone_variable(model, variables, zone_index)
            for variable in variables["zone"].values():
                variable.LB = 1.0
                variable.UB = 1.0
                variable.VType = "B"
            for variable in variables["shortage"].values():
                variable.UB = 0.0
            for key in ("area_use", "voyage_area_use", "attr_state"):
                for variable in variables[key].values():
                    variable.VType = "B"
            for key in ("export_flow", "import_reserve"):
                for variable in variables[key].values():
                    variable.VType = "I"
            warm_start = {"provided": False}
            if start_source_variables is not None:
                export_start = self._export_flow_start_for_zones(
                    selected_zone_indices,
                    start_source_variables,
                )
                warm_start = self._apply_flow_state_mip_start(
                    selected_zone_indices,
                    export_start,
                    variables,
                )
                import_start = self._import_reservation_mip_start(
                    start_source_variables
                )
                for key, variable in variables["import_reserve"].items():
                    variable.Start = float(import_start.get(key, 0))
                warm_start["positive_import_flow_start_count"] = sum(
                    int(value) > 0 for value in import_start.values()
                )
                warm_start["provided"] = bool(
                    export_start
                    and (
                        import_start
                        or not self.import_total_by_flow_size
                    )
                )
            model.update()
            remaining = deadline - perf_counter()
            if remaining <= 1e-6:
                return {
                    "status": "time_limit_before_repair",
                    "feasible": False,
                    "warm_start": warm_start,
                    "seconds": perf_counter() - started,
                }
            self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
            self._set_gurobi_param(model, "MIPGap", 0.0)
            self._set_gurobi_param(model, "MIPFocus", 1)
            self._set_gurobi_param(model, "Heuristics", 0.25)
            model.optimize()
            status = self._gurobi_status_name(model)
            if self._gurobi_solution_count(model) <= 0:
                return {
                    "status": status,
                    "feasible": False,
                    "warm_start": warm_start,
                    "seconds": perf_counter() - started,
                }
            export_flow = {
                key: int(round(self._gurobi_value(model, variable)))
                for key, variable in variables["export_flow"].items()
                if self._gurobi_value(model, variable) > 1e-7
            }
            import_reserve = {
                key: int(round(self._gurobi_value(model, variable)))
                for key, variable in variables["import_reserve"].items()
                if self._gurobi_value(model, variable) > 1e-7
            }
            reconstruction = self._reconstruct_zone_objective(
                selected_zone_indices,
                export_flow,
                import_reserve,
            )
            return {
                "status": status,
                "feasible": True,
                "selected_zone_indices": set(selected_zone_indices),
                "export_flow": export_flow,
                "import_reserve": import_reserve,
                "objective": float(reconstruction["objective"]),
                "objective_reconstruction": reconstruction,
                "warm_start": warm_start,
                "seconds": perf_counter() - started,
            }
        finally:
            self._free_gurobi_model(model)

    def _repaired_start_values(
        self,
        repaired: dict[str, object],
        variables: dict,
    ) -> dict[object, float]:
        """Map one certified repaired solution onto the restricted master."""

        selected = set(repaired["selected_zone_indices"])
        export_flow = dict(repaired["export_flow"])
        import_reserve = dict(repaired["import_reserve"])
        used_areas = {
            (group_id, self.bays[bay_key].area_no)
            for (group_id, bay_key), quantity in export_flow.items()
            if int(quantity) > 0
        }
        used_voyage_areas = {
            (self.groups_by_id[group_id].voyage_id, area_no)
            for group_id, area_no in used_areas
        }
        used_attributes = {
            key
            for zone_index in selected
            for key, value in self._zones[zone_index].bay_attr_uses
            if value > 0
        }
        start: dict[object, float] = {variables["baseline"]: 1.0}
        start.update(
            {
                variable: 1.0 if zone_index in selected else 0.0
                for zone_index, variable in variables["zone"].items()
            }
        )
        start.update(
            {
                variable: float(export_flow.get(key, 0))
                for key, variable in variables["export_flow"].items()
            }
        )
        start.update(
            {
                variable: 1.0 if key in used_areas else 0.0
                for key, variable in variables["area_use"].items()
            }
        )
        start.update(
            {
                variable: 1.0 if key in used_voyage_areas else 0.0
                for key, variable in variables["voyage_area_use"].items()
            }
        )
        start.update(
            {
                variable: 1.0 if key in used_attributes else 0.0
                for key, variable in variables["attr_state"].items()
            }
        )
        start.update(
            {
                variable: float(import_reserve.get(key, 0))
                for key, variable in variables["import_reserve"].items()
            }
        )
        start.update({variable: 0.0 for variable in variables["shortage"].values()})
        return start

    def _greedy_zone_mip_start(
        self,
        model,
        variables: dict,
        *,
        start_deadline: float,
        start_budget_seconds: float,
        repair_time_limit: float,
    ) -> dict[str, object]:
        """Lazily generate and repair starts under one shared hard deadline."""

        started = perf_counter()
        context = self._greedy_candidate_context(variables)
        generated_summaries: list[dict[str, object]] = []
        candidate_summaries: list[dict[str, object]] = []
        seen_supports: set[tuple] = set()
        repaired: list[dict[str, object]] = []
        pending_start_pairs: list[tuple[dict[str, object], dict[object, float]]] = []
        repair_summaries: list[dict[str, object]] = []
        added_seed_zone_count = 0
        solver_submission: dict[str, object] | None = None
        generation_seconds = 0.0
        repair_seconds = 0.0
        generation_completed_count = 0
        generation_interrupted_count = 0
        duplicate_count = 0
        infeasible_count = 0
        termination_reason = "all_strategies_exhausted"

        strategies = self._greedy_candidate_strategies(context["group_ids"])
        for policy, ordering, protection in strategies:
            if perf_counter() >= start_deadline:
                termination_reason = "budget_exhausted"
                break
            generated_started = perf_counter()
            candidate, generated_summary = self._generate_greedy_support_candidate(
                context,
                ordering_policy=policy,
                group_order=ordering,
                import_protection=protection,
                deadline=start_deadline,
            )
            generation_seconds += perf_counter() - generated_started
            generated_summaries.append(generated_summary)
            if candidate is None:
                generation_interrupted_count += 1
                termination_reason = "budget_exhausted"
                break
            generation_completed_count += 1
            if candidate["generation_status"] == "infeasible":
                infeasible_count += 1
            support_key = candidate["support_key"]
            if support_key in seen_supports:
                duplicate_count += 1
                continue
            seen_supports.add(support_key)
            candidate_summaries.append(
                self._greedy_candidate_summary(candidate)
            )
            if not bool(candidate["complete_export_cover"]):
                continue
            if (
                len(repair_summaries)
                >= int(self.zone_config.max_repaired_start_candidates)
            ):
                termination_reason = "max_repair_candidates_reached"
                break

            remaining_repair_budget = max(
                0.0,
                float(repair_time_limit) - repair_seconds,
            )
            remaining_start_budget = max(0.0, start_deadline - perf_counter())
            if remaining_repair_budget <= 1e-6:
                termination_reason = "repair_budget_exhausted"
                break
            if remaining_start_budget <= 1e-6:
                termination_reason = "budget_exhausted"
                break
            remaining_repair_slots = max(
                1,
                int(self.zone_config.max_repaired_start_candidates)
                - len(repair_summaries),
            )
            candidate_deadline = min(
                start_deadline,
                perf_counter()
                + remaining_repair_budget / remaining_repair_slots,
            )
            selected = {
                self._register_zone(strip_key, signature)
                for strip_key, signature in candidate["support"]
            }
            repair_started = perf_counter()
            result = self._repair_zone_support(
                selected,
                candidate_deadline,
                start_source_variables=variables,
            )
            repair_seconds += perf_counter() - repair_started
            summary = {
                "ordering_policy": candidate["ordering_policy"],
                "import_protection": candidate["import_protection"],
                "selected_zone_count": len(selected),
                "status": result["status"],
                "feasible": bool(result["feasible"]),
                "objective": result.get("objective"),
                "seconds": result["seconds"],
                "warm_start": result.get("warm_start", {}),
            }
            repair_summaries.append(summary)
            if result["feasible"]:
                result["candidate"] = summary
                repaired.append(result)
                preparation_interrupted = False
                new_zone_variables = []
                for zone_index in sorted(result["selected_zone_indices"]):
                    if perf_counter() >= start_deadline:
                        preparation_interrupted = True
                        break
                    if zone_index not in variables["active_zone_indices"]:
                        zone_variable = self._add_zone_variable(
                            model,
                            variables,
                            zone_index,
                        )
                        new_zone_variables.append(zone_variable)
                        added_seed_zone_count += 1
                if preparation_interrupted:
                    termination_reason = "budget_exhausted"
                    break
                if new_zone_variables:
                    model.update()
                    for zone_variable in new_zone_variables:
                        for _prior_result, prior_start in pending_start_pairs:
                            prior_start[zone_variable] = 0.0
                start_values = self._repaired_start_values(result, variables)
                pending_start_pairs.append((result, start_values))
                pending_start_pairs.sort(
                    key=lambda pair: (
                        float(pair[0]["objective"]),
                        tuple(sorted(pair[0]["selected_zone_indices"])),
                    )
                )
                solver_submission = model.apply_mip_starts(
                    [pair[1] for pair in pending_start_pairs],
                    deadline=start_deadline,
                )
                submitted_count = int(
                    solver_submission["provided_mip_start_count"]
                )
                pending_start_pairs = pending_start_pairs[:submitted_count]
                if solver_submission.get("deadline_exhausted"):
                    termination_reason = "budget_exhausted"
                    break
                if (
                    len(pending_start_pairs)
                    >= int(self.zone_config.max_mip_starts)
                ):
                    termination_reason = "max_starts_reached"
                    break

        pending_start_pairs.sort(
            key=lambda pair: (
                float(pair[0]["objective"]),
                tuple(sorted(pair[0]["selected_zone_indices"])),
            )
        )
        pending_results = [pair[0] for pair in pending_start_pairs]
        pending_starts = [pair[1] for pair in pending_start_pairs]
        variables["pending_repaired_mip_starts"] = []
        variables["certified_repaired_start_zone_indices"] = {
            zone_index
            for result in pending_results
            for zone_index in result["selected_zone_indices"]
        }

        actual_seconds = perf_counter() - started
        budget_seconds = max(0.0, float(start_budget_seconds))
        budget_exhausted = bool(
            termination_reason == "budget_exhausted"
            or perf_counter() >= start_deadline
        )
        if not pending_results and not repaired and termination_reason == "all_strategies_exhausted":
            termination_reason = "no_feasible_start"
        return {
            "generated_candidate_count": generation_completed_count,
            "deduplicated_candidate_count": len(seen_supports),
            "complete_candidate_count": sum(
                bool(summary["complete_export_cover"])
                for summary in candidate_summaries
            ),
            "infeasible_candidate_count": infeasible_count,
            "budget_interrupted_candidate_count": generation_interrupted_count,
            "candidate_generation_completed_count": generation_completed_count,
            "candidate_generation_interrupted_count": generation_interrupted_count,
            "candidate_duplicate_count": duplicate_count,
            "candidate_summaries": candidate_summaries,
            "generated_candidate_summaries": generated_summaries,
            "repaired_candidate_count": len(repair_summaries),
            "repair_attempt_count": len(repair_summaries),
            "feasible_repaired_count": len(repaired),
            "repair_feasible_count": len(repaired),
            "provided_mip_start_count": len(pending_starts),
            "submitted_start_count": 0,
            "mandatory_start_zone_count": len(
                variables["certified_repaired_start_zone_indices"]
            ),
            "best_repaired_start_objective": (
                float(pending_results[0]["objective"])
                if pending_results
                else None
            ),
            "added_seed_zone_count": added_seed_zone_count,
            "candidate_generation_seconds": generation_seconds,
            "repair_time_limit_seconds": float(repair_time_limit),
            "repair_seconds": repair_seconds,
            "start_preparation_budget_seconds": budget_seconds,
            "start_preparation_actual_seconds": actual_seconds,
            "start_preparation_budget_utilization": (
                actual_seconds / budget_seconds if budget_seconds > 0.0 else None
            ),
            "total_mip_start_seconds": actual_seconds,
            "start_budget_exhausted": budget_exhausted,
            "start_termination_reason": termination_reason,
            "ordering_families_with_feasible_start": sorted(
                {
                    str(result["candidate"]["ordering_policy"])
                    for result in repaired
                }
            ),
            "repair_summaries": repair_summaries,
            "solver_submission": solver_submission,
        }

    def _solve_complete_zone_lp(self, deadline: float) -> dict[str, object]:
        self._materialize_all_zones()
        model, variables = self._build_zone_master()
        try:
            for zone in self._zones:
                self._add_zone_variable(model, variables, zone.zone_id)
            model.update()
            remaining = deadline - perf_counter()
            if remaining <= 1e-6:
                return {"status": "time_limit_before_solve"}
            self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
            model.optimize()
            status = self._gurobi_status_name(model)
            return {
                "status": status,
                "objective": (
                    self._gurobi_objective_value(model)
                    if self._gurobi_solution_count(model) > 0
                    else None
                ),
                "shortage": (
                    sum(
                        self._gurobi_value(model, variable)
                        for variable in variables["shortage"].values()
                    )
                    if self._gurobi_solution_count(model) > 0
                    else None
                ),
                "zone_count": self._possible_zone_count,
            }
        finally:
            self._free_gurobi_model(model)

    def _solve_complete_zone_mip(self, deadline: float) -> dict[str, object]:
        self._materialize_all_zones()
        model, variables = self._build_zone_master()
        try:
            for zone in self._zones:
                self._add_zone_variable(model, variables, zone.zone_id)
            model.update()
            selected, _export_flow, _import_reserve, stats = self._integerize_zone_master(
                model,
                variables,
                deadline,
                start_policy=COMPLETE_MIP_BASELINE_POLICY,
                progress_phase="complete_zone_mip",
            )
            stats["zone_count"] = self._possible_zone_count
            stats["selected_candidate_count"] = len(
                {
                    candidate_index
                    for zone_index in selected
                    for candidate_index in self._zones[
                        zone_index
                    ].candidate_indices
                }
            )
            return stats
        finally:
            self._free_gurobi_model(model)

    def analyze_root(
        self,
        *,
        compare_complete_lp: bool = False,
        compare_complete_mip: bool = False,
    ) -> dict[str, object]:
        started = perf_counter()
        total_limit = max(0.01, float(self.config.total_time_limit))
        deadline = started + total_limit
        preparation = self._prepare_zones()
        model, variables = self._build_zone_master()
        try:
            root_deadline = perf_counter() + max(
                0.0, deadline - perf_counter()
            ) * float(self.zone_config.root_time_fraction)
            root = self._run_root_generation(model, variables, root_deadline)
        finally:
            self._free_gurobi_model(model)
        complete = None
        if compare_complete_lp and perf_counter() < deadline:
            complete = self._solve_complete_zone_lp(deadline)
        complete_mip = None
        if compare_complete_mip and perf_counter() < deadline:
            complete_mip = self._solve_complete_zone_mip(deadline)
        result = {
            "algorithm": "contiguous_zone_root_generation",
            "model_scope": "dedicated_contiguous_row_zone_support",
            **preparation,
            **root,
            "complete_zone_lp": complete,
            "complete_zone_mip": complete_mip,
            "total_seconds": perf_counter() - started,
        }
        if (
            complete
            and complete.get("status") == "optimal"
            and root.get("closed")
            and root.get("root_objective") is not None
        ):
            result["root_complete_lp_difference"] = float(
                root["root_objective"]
            ) - float(complete["objective"])
        return result

    def analyze_complete_zone_mip(self) -> dict[str, object]:
        """Solve the fully enumerated zone model as the same-model baseline."""

        started = perf_counter()
        deadline = started + max(0.01, float(self.config.total_time_limit))
        preparation = self._prepare_zones()
        self._materialize_all_zones()
        model, variables = self._build_zone_master()
        try:
            for zone in self._zones:
                self._add_zone_variable(model, variables, zone.zone_id)
            model.update()
            selected, export_flow, import_reserve, stats = self._integerize_zone_master(
                model,
                variables,
                deadline,
                start_policy=COMPLETE_MIP_BASELINE_POLICY,
                progress_phase="complete_zone_mip",
            )
        finally:
            self._free_gurobi_model(model)
        certificate = (
            self._zone_objective_certificate(
                selected,
                export_flow,
                import_reserve,
                float(stats["objective"]),
            )
            if selected and stats.get("has_solution")
            else None
        )
        return {
            "algorithm": "complete_contiguous_zone_mip",
            "model_scope": "dedicated_contiguous_row_zone_support",
            "baseline_role": "same_model_fully_enumerated_direct_mip",
            **preparation,
            **stats,
            "objective_certificate": certificate,
            "selected_candidate_count": len(
                {
                    candidate_index
                    for zone_index in selected
                    for candidate_index in self._zones[
                        zone_index
                    ].candidate_indices
                }
            ),
            "total_seconds": perf_counter() - started,
        }

    def solve_complete_zone_mip(self) -> ColumnGenerationResult:
        """Solve and independently validate the end-to-end complete-zone baseline."""

        started = perf_counter()
        self._solve_started_at = started
        total_limit = max(0.01, float(self.config.total_time_limit))
        deadline = started + total_limit
        fill_reserve = total_limit * float(self.zone_config.fill_time_fraction)
        preparation = self._prepare_zones()
        self._materialize_all_zones()
        model, variables = self._build_zone_master()
        try:
            proof_only_area_cut_count = self._remove_proof_only_area_rows(
                model,
                variables,
            )
            for zone in self._zones:
                self._add_zone_variable(model, variables, zone.zone_id)
            model.update()
            (
                selected_zones,
                export_flow,
                import_reserve,
                zone_mip,
            ) = self._integerize_zone_master(
                model,
                variables,
                deadline - fill_reserve,
                start_policy=COMPLETE_MIP_BASELINE_POLICY,
                progress_phase="complete_zone_mip",
            )
            if not selected_zones:
                raise RuntimeError(
                    "complete zone MIP did not obtain an integer support: "
                    f"{zone_mip}"
                )
        finally:
            self._free_gurobi_model(model)

        objective_certificate = self._zone_objective_certificate(
            selected_zones,
            export_flow,
            import_reserve,
            float(zone_mip["objective"]),
        )
        zone_upper_bound = float(objective_certificate["objective"])
        zone_global_lower_bound = float(zone_mip["bound"])
        zone_absolute_gap = max(
            0.0,
            zone_upper_bound - zone_global_lower_bound,
        )
        zone_relative_gap = zone_absolute_gap / max(
            abs(zone_upper_bound),
            1e-12,
        )
        selected, fill = self._solve_restricted_fill(
            selected_zones,
            export_flow,
            import_reserve,
            deadline,
        )
        candidate_indices = {
            index
            for zone_index in selected_zones
            for index in self._zones[zone_index].candidate_indices
        }
        diagnostics = {
            "algorithm": "complete_contiguous_zone_mip_with_exact_recourse",
            "algorithm_version": "complete_zone_mip_v4_model_phase1_1_isolated",
            "model_scope": "actual_quantity_flow_on_dedicated_contiguous_row_zones",
            "baseline_role": "same_model_fully_enumerated_end_to_end_mip",
            "formulation": "fully_enumerated_zone_flow_master_plus_exact_row_recourse",
            "decomposition": "none_before_flow_fixed_exact_row_recourse",
            "planned_group_count": len(self.groups),
            "planned_box_count": sum(group.demand for group in self.groups),
            "candidate_row_location_count": len(self._columns),
            "zone_preparation": preparation,
            "zone_root_proof_only_area_activation_cut_count": (
                proof_only_area_cut_count
            ),
            "zone_mip": zone_mip,
            "mip_anytime": self._aggregate_mip_progress(
                [zone_mip.get("mip_progress")]
            ),
            "zone_model_upper_bound": zone_upper_bound,
            "zone_model_global_lower_bound": zone_global_lower_bound,
            "zone_model_absolute_gap": zone_absolute_gap,
            "zone_model_relative_gap": zone_relative_gap,
            "zone_model_lower_bound_source": "complete_zone_mip_gurobi_bound",
            "zone_selected_candidate_count": len(candidate_indices),
            "zone_objective_certificate": objective_certificate,
            "zone_fill": fill,
            "zone_time_policy": {
                "complete_mip_total_fraction": (
                    1.0 - float(self.zone_config.fill_time_fraction)
                ),
                "final_fill_total_fraction": float(
                    self.zone_config.fill_time_fraction
                ),
                "integer_search_policy": COMPLETE_MIP_BASELINE_POLICY,
                "v5_multi_start_enabled": False,
            },
            "master_status": zone_mip["status"],
            "master_objective": zone_upper_bound,
            "master_bound_scope": "complete_redefined_zone_model",
            "row_recourse_status": fill["status"],
            "row_recourse_secondary_quality_objective": fill["objective"],
            "complete_model_lower_bound": zone_global_lower_bound,
            "complete_model_absolute_gap": zone_absolute_gap,
            "complete_model_relative_gap": zone_relative_gap,
            "complete_model_gap_source": "complete_zone_mip_gurobi_bound",
            "restricted_fill_lower_bound": fill["bound"],
            "hard_demand_balance": True,
            "business_objective_normalization": {
                "weights": self._zone_objective_weights(),
                "scales": dict(self._zone_objective_scales),
                "method": "natural_instance_scale",
            },
            "business_objective": self._zone_business_objective_specification(),
            "peak_utilization_policy": dict(
                self._peak_utilization_policy
            ),
            "import_capacity_reservation": {
                "source": "declared_import_documents_excluding_in_yard_boxes",
                "role": "anonymous_size_compatible_capacity_only",
                "area_policy": "endogenous_feasible_reservation_without_area_objective",
                "constraint_scope": [
                    "area_function",
                    "bay_size",
                    "physical_capacity",
                    "peak_utilization_epsilon_cap",
                ],
                "excluded_constraints": [
                    "bay_no_mix",
                    "row_no_mix",
                    "container_group_attributes",
                    "voyage_area_dispersion_objective",
                    "contiguous_zone_objective",
                    "existing_group_proximity_objective",
                    "unused_export_zone_capacity_objective",
                    "berth_distance_objective",
                ],
                "import_boxes": int(
                    sum(self.import_total_by_flow_size.values())
                ),
            },
            "total_seconds": perf_counter() - started,
            "research_stage_gate": True,
            "production_solver_registered": False,
        }
        result = self._assemble_result(selected, diagnostics)
        row_quality_objective = result.diagnostics["final_business_objective"]
        row_quality_components = result.diagnostics[
            "final_business_objective_components"
        ]
        row_quality_components = self._integrated_row_quality_components(
            row_quality_components
        )
        result.diagnostics["row_recourse_secondary_quality_objective"] = (
            row_quality_objective
        )
        result.diagnostics["row_recourse_secondary_quality_components"] = (
            row_quality_components
        )
        result.diagnostics["final_business_objective"] = zone_upper_bound
        result.diagnostics["final_business_objective_components"] = {
            "raw": objective_certificate["raw"],
            "normalized": objective_certificate["normalized"],
            "weighted": objective_certificate["weighted"],
            "weighted_total": zone_upper_bound,
        }
        result.columns = self._selected_direct_columns(selected)
        return result

    def _initialize_complete_mip_start_from_lp(
        self,
        model,
        variables: dict,
        deadline: float,
    ) -> dict[str, object]:
        """Provide the full-MIP greedy start with a feasible LP allocation.

        Restricted masters already arrive here with root-generation values.
        The direct complete-zone baseline does not.  Without this initialization
        its nominal import-protection step sees an empty allocation and can build
        an export support that leaves no room for the required imports.
        """

        if variables.get("last_root_zone_values"):
            return {
                "executed": False,
                "source": "existing_root_generation_values",
            }
        started = perf_counter()
        remaining = deadline - started
        if remaining <= 1e-6:
            return {
                "executed": False,
                "source": "complete_lp",
                "status": "time_limit_before_lp_start",
                "seconds": 0.0,
            }
        lp_limit = min(
            remaining,
            max(0.01, 0.10 * float(self.config.total_time_limit)),
        )
        self._set_gurobi_param(model, "TimeLimit", lp_limit)
        model.optimize()
        status = self._gurobi_status_name(model)
        has_solution = self._gurobi_solution_count(model) > 0
        if has_solution:
            variables["last_root_zone_values"] = {
                key: self._gurobi_value(model, variable)
                for key, variable in variables["zone"].items()
            }
            variables["last_root_import_values"] = {
                key: self._gurobi_value(model, variable)
                for key, variable in variables["import_reserve"].items()
            }
            variables["last_root_export_flow_values"] = {
                key: self._gurobi_value(model, variable)
                for key, variable in variables["export_flow"].items()
            }
            variables["last_root_area_values"] = {
                key: self._gurobi_value(model, variable)
                for key, variable in variables["area_use"].items()
            }
            variables["last_root_voyage_area_values"] = {
                key: self._gurobi_value(model, variable)
                for key, variable in variables["voyage_area_use"].items()
            }
            variables["last_root_attr_values"] = {
                key: self._gurobi_value(model, variable)
                for key, variable in variables["attr_state"].items()
            }
        return {
            "executed": True,
            "source": "complete_lp",
            "status": status,
            "has_solution": has_solution,
            "objective": (
                self._gurobi_objective_value(model) if has_solution else None
            ),
            "fractional_import_variable_count": (
                sum(
                    1e-8 < float(value) < 1.0 - 1e-8
                    for value in variables.get(
                        "last_root_import_values", {}
                    ).values()
                )
                if has_solution
                else None
            ),
            "time_limit": float(lp_limit),
            "seconds": perf_counter() - started,
        }

    def _finalize_mip_progress(
        self,
        recorder: MipProgressRecorder,
        model,
    ) -> dict[str, object]:
        progress = recorder.finalize(model)
        solve_started = getattr(self, "_solve_started_at", recorder.started_at)
        progress["global_start_offset_seconds"] = max(
            0.0,
            recorder.started_at - solve_started,
        )
        return progress

    @staticmethod
    def _aggregate_mip_progress(
        progresses: list[dict[str, object] | None],
    ) -> dict[str, object]:
        """Merge same-objective MIP phases onto the solve's wall-clock axis."""

        events = []
        for progress in progresses:
            if not progress:
                continue
            offset = float(progress.get("global_start_offset_seconds", 0.0))
            for event in progress.get("incumbent_trajectory", []):
                shifted = dict(event)
                shifted["elapsed_seconds"] = offset + float(
                    event["elapsed_seconds"]
                )
                shifted["phase"] = progress.get("phase")
                events.append(shifted)
        events.sort(key=lambda event: (float(event["elapsed_seconds"]), event["event_type"]))
        incumbent_events = [
            event for event in events if event.get("incumbent") is not None
        ]
        first = incumbent_events[0] if incumbent_events else None
        best = None
        best_value = math.inf
        for event in incumbent_events:
            value = float(event["incumbent"])
            if value < best_value - 1e-9:
                best_value = value
                best = event
        return {
            "time_to_first_solution": (
                float(first["elapsed_seconds"]) if first else None
            ),
            "time_to_best_solution": (
                float(best["elapsed_seconds"]) if best else None
            ),
            "first_incumbent": (
                float(first["incumbent"]) if first else None
            ),
            "best_incumbent": (
                float(best["incumbent"]) if best else None
            ),
            "node_count": sum(
                float(progress.get("node_count", 0.0))
                for progress in progresses
                if progress
            ),
            "solution_count": sum(
                int(progress.get("solution_count", 0))
                for progress in progresses
                if progress
            ),
            "incumbent_trajectory": events,
        }

    def _integerize_zone_master(
        self,
        model,
        variables: dict,
        deadline: float,
        *,
        start_policy: str,
        progress_phase: str = "initial_zone_mip",
    ) -> tuple[
        set[int],
        dict[tuple[str, str], int],
        dict[tuple[str, str, str], int],
        dict[str, object],
    ]:
        started = perf_counter()
        if start_policy not in {
            V5_MULTI_START_POLICY,
            COMPLETE_MIP_BASELINE_POLICY,
        }:
            raise ValueError(f"unknown integer-search start policy: {start_policy!r}")
        lp_start = self._initialize_complete_mip_start_from_lp(
            model,
            variables,
            deadline,
        )
        if start_policy == V5_MULTI_START_POLICY:
            start_phase_begin = perf_counter()
            start_budget_seconds = (
                max(0.0, float(self.config.total_time_limit))
                * float(self.zone_config.mip_start_total_time_fraction)
            )
            start_deadline = min(
                deadline,
                start_phase_begin + start_budget_seconds,
            )
            repair_time_limit = min(
                max(0.0, start_deadline - perf_counter()),
                max(0.0, float(self.config.total_time_limit))
                * float(self.zone_config.mip_start_repair_total_fraction),
            )
            mip_start = self._greedy_zone_mip_start(
                model,
                variables,
                start_deadline=start_deadline,
                start_budget_seconds=start_budget_seconds,
                repair_time_limit=repair_time_limit,
            )
            mip_start.update(
                {
                    "integer_search_policy": start_policy,
                    "v5_multi_start_enabled": True,
                    "baseline_start_type": None,
                }
            )
        else:
            start_phase_begin = perf_counter()
            start_deadline = None
            baseline_start = self._legacy_gurobi_repair_start(model, variables)
            baseline_start_count = int(
                int(baseline_start.get("selected_zone_count", 0)) > 0
            )
            baseline_seconds = perf_counter() - start_phase_begin
            mip_start = {
                "integer_search_policy": start_policy,
                "v5_multi_start_enabled": False,
                "baseline_start_type": "v4_single_partial_gurobi_repair_start",
                "baseline_start": baseline_start,
                "candidate_generation_executed": False,
                "candidate_generation_seconds": 0.0,
                "candidate_generation_completed_count": 0,
                "candidate_generation_interrupted_count": 0,
                "generated_candidate_count": 0,
                "deduplicated_candidate_count": 0,
                "complete_candidate_count": 0,
                "candidate_duplicate_count": 0,
                "infeasible_candidate_count": 0,
                "budget_interrupted_candidate_count": 0,
                "repair_executed": False,
                "repair_seconds": 0.0,
                "repair_attempt_count": 0,
                "repair_feasible_count": 0,
                "repaired_candidate_count": 0,
                "feasible_repaired_count": 0,
                "provided_mip_start_count": baseline_start_count,
                "submitted_start_count": baseline_start_count,
                "submitted_mip_start_count": baseline_start_count,
                "start_preparation_budget_seconds": None,
                "start_preparation_actual_seconds": baseline_seconds,
                "start_preparation_budget_utilization": None,
                "total_mip_start_seconds": baseline_seconds,
                "start_budget_exhausted": False,
                "start_termination_reason": (
                    "baseline_native_start_prepared"
                    if baseline_start_count
                    else "no_feasible_start"
                ),
                "solver_submission": {
                    "mode": "native_variable_start",
                    "provided_mip_start_count": baseline_start_count,
                    "solver_acceptance_observed": False,
                },
            }
        for variable in variables["zone"].values():
            variable.VType = "B"
        for variable in variables["area_use"].values():
            variable.VType = "B"
        for variable in variables["voyage_area_use"].values():
            variable.VType = "B"
        for variable in variables["attr_state"].values():
            variable.VType = "B"
        for variable in variables["export_flow"].values():
            variable.VType = "I"
        for variable in variables["import_reserve"].values():
            variable.VType = "I"
        for variable in variables["shortage"].values():
            variable.UB = 0.0
        model.update()
        if start_policy == V5_MULTI_START_POLICY:
            pending_starts = variables.pop("pending_repaired_mip_starts", [])
            submission = mip_start.get("solver_submission")
            if submission is None:
                submission = model.apply_mip_starts(
                    pending_starts,
                    deadline=start_deadline,
                )
            mip_start["solver_submission"] = submission
            submitted_count = int(submission["provided_mip_start_count"])
            mip_start["provided_mip_start_count"] = submitted_count
            mip_start["submitted_start_count"] = submitted_count
            mip_start["submitted_mip_start_count"] = submitted_count
            if submission.get("deadline_exhausted"):
                mip_start["start_budget_exhausted"] = True
                mip_start["start_termination_reason"] = "budget_exhausted"
            actual_start_seconds = perf_counter() - start_phase_begin
            mip_start["start_preparation_actual_seconds"] = actual_start_seconds
            mip_start["total_mip_start_seconds"] = actual_start_seconds
            budget_seconds = float(
                mip_start["start_preparation_budget_seconds"]
            )
            mip_start["start_preparation_budget_utilization"] = (
                actual_start_seconds / budget_seconds
                if budget_seconds > 0.0
                else None
            )
            mip_start["candidate_generation_executed"] = bool(
                int(mip_start["candidate_generation_completed_count"])
                + int(mip_start["candidate_generation_interrupted_count"])
            )
            mip_start["repair_executed"] = bool(
                int(mip_start["repair_attempt_count"])
            )
        policy_diagnostics = {
            "integer_search_policy": start_policy,
            "v5_multi_start_enabled": bool(
                mip_start["v5_multi_start_enabled"]
            ),
            "candidate_generation_executed": bool(
                mip_start.get("candidate_generation_executed", False)
            ),
            "candidate_generation_seconds": float(
                mip_start.get("candidate_generation_seconds", 0.0)
            ),
            "repair_executed": bool(mip_start.get("repair_executed", False)),
            "repair_seconds": float(mip_start.get("repair_seconds", 0.0)),
            "submitted_mip_start_count": int(
                mip_start.get("submitted_mip_start_count", 0)
            ),
            "baseline_start_type": mip_start.get("baseline_start_type"),
        }
        remaining = deadline - perf_counter()
        if remaining <= 1e-6:
            return set(), {}, {}, {
                "status": "time_limit_before_zone_mip",
                "lp_start": lp_start,
                "mip_start": mip_start,
                **policy_diagnostics,
                "seconds": perf_counter() - started,
            }
        self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
        self._set_gurobi_param(model, "MIPGap", 0.0)
        self._set_gurobi_param(model, "MIPFocus", 1)
        self._set_gurobi_param(model, "Heuristics", 0.20)
        dynamic_stopping = bool(
            start_policy == V5_MULTI_START_POLICY
            and self.zone_config.initial_mip_dynamic_stopping_enabled
        )
        if dynamic_stopping:
            total_time_limit = max(
                0.0,
                float(self.config.total_time_limit),
            )
            progress_recorder = StagnationStoppingMipProgressRecorder(
                phase=progress_phase,
                minimum_run_seconds=(
                    total_time_limit
                    * float(self.zone_config.initial_mip_min_total_fraction)
                ),
                stagnation_seconds=(
                    total_time_limit
                    * float(
                        self.zone_config.initial_mip_stagnation_total_fraction
                    )
                ),
                minimum_relative_improvement=float(
                    self.zone_config.initial_mip_min_relative_improvement
                ),
            )
        else:
            progress_recorder = MipProgressRecorder(phase=progress_phase)
        model.optimize(progress_recorder)
        progress = self._finalize_mip_progress(progress_recorder, model)
        status = self._gurobi_status_name(model)
        stopping_diagnostics = (
            progress_recorder.stopping_diagnostics()
            if isinstance(
                progress_recorder,
                StagnationStoppingMipProgressRecorder,
            )
            else {
                "enabled": False,
                "callback_termination_reason": None,
            }
        )
        has_solution = self._gurobi_solution_count(model) > 0
        if stopping_diagnostics.get("callback_termination_reason"):
            termination_reason = str(
                stopping_diagnostics["callback_termination_reason"]
            )
        elif status == "optimal":
            termination_reason = "optimal"
        elif not has_solution:
            termination_reason = "no_solution"
        else:
            termination_reason = "time_limit"
        stopping_diagnostics["termination_reason"] = termination_reason
        stopping_diagnostics["hard_deadline_seconds"] = max(
            0.0,
            float(remaining),
        )
        if not has_solution:
            return set(), {}, {}, {
                "status": status,
                "has_solution": False,
                "lp_start": lp_start,
                "mip_start": mip_start,
                "mip_progress": progress,
                "termination_reason": termination_reason,
                "initial_mip_stopping": stopping_diagnostics,
                **policy_diagnostics,
                "seconds": perf_counter() - started,
            }
        selected = {
            zone_index
            for zone_index, variable in variables["zone"].items()
            if self._gurobi_value(model, variable) > 0.5
        }
        export_flow = {
            key: int(round(self._gurobi_value(model, variable)))
            for key, variable in variables["export_flow"].items()
            if self._gurobi_value(model, variable) > 1e-7
        }
        import_reserve = {
            key: int(round(self._gurobi_value(model, variable)))
            for key, variable in variables["import_reserve"].items()
            if self._gurobi_value(model, variable) > 1e-7
        }
        objective = self._gurobi_objective_value(model)
        best_start_objective = mip_start.get("best_repaired_start_objective")
        if (
            best_start_objective is not None
            and objective > float(best_start_objective) + 1e-6
        ):
            raise RuntimeError(
                "restricted MIP incumbent is worse than a submitted certified "
                f"start: incumbent={objective}, start={best_start_objective}"
            )
        return selected, export_flow, import_reserve, {
            "status": status,
            "has_solution": True,
            "objective": self._gurobi_objective_value(model),
            "bound": self._gurobi_dual_bound(model),
            "absolute_gap": max(
                0.0,
                self._gurobi_objective_value(model)
                - self._gurobi_dual_bound(model),
            ),
            "relative_gap": max(
                0.0,
                self._gurobi_objective_value(model)
                - self._gurobi_dual_bound(model),
            )
            / max(abs(self._gurobi_objective_value(model)), 1e-12),
            "selected_zone_count": len(selected),
            "lp_start": lp_start,
            "mip_start": mip_start,
            "mip_progress": progress,
            "termination_reason": termination_reason,
            "initial_mip_stopping": stopping_diagnostics,
            **policy_diagnostics,
            "seconds": perf_counter() - started,
        }

    def _objective_contribution_by_group(
        self,
        selected_zone_indices: set[int],
        export_flow: dict[tuple[str, str], int],
    ) -> dict[str, float]:
        """Attribute the incumbent objective to groups for seed selection."""

        zones_by_group: defaultdict[str, list[int]] = defaultdict(list)
        for zone_index in selected_zone_indices:
            zones_by_group[self._zones[zone_index].group_id].append(zone_index)
        used_areas_by_voyage: defaultdict[str, set[str]] = defaultdict(set)
        flow_cost: Counter[str] = Counter()
        for (group_id, bay_key), quantity in export_flow.items():
            if int(quantity) <= 0:
                continue
            area_no = self.bays[bay_key].area_no
            used_areas_by_voyage[
                self.groups_by_id[group_id].voyage_id
            ].add(area_no)
            flow_cost[group_id] += (
                self._zone_flow_unit_cost(group_id, bay_key)
                - self._unused_capacity_unit_cost()
            ) * int(quantity)
        voyage_demand: Counter[str] = Counter()
        for group in self.groups:
            voyage_demand[group.voyage_id] += int(group.demand)
        contribution_by_group = {}
        for group in self.groups:
            group_id = group.group_id
            zone_cost = sum(
                float(self._zones[index].objective_cost)
                for index in zones_by_group[group_id]
            ) - self._zone_activation_penalty()
            voyage_area_cost = (
                self._voyage_area_activation_penalty()
                * max(
                    0,
                    len(used_areas_by_voyage[group.voyage_id]) - 1,
                )
                * int(group.demand)
                / max(1, int(voyage_demand[group.voyage_id]))
            )
            contribution = max(
                0.0,
                zone_cost
                + flow_cost[group_id]
                + voyage_area_cost,
            )
            contribution_by_group[group_id] = float(contribution)
        return contribution_by_group

    @staticmethod
    def _set_jaccard(left: frozenset | set, right: frozenset | set) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    def _incumbent_area_state(
        self,
        export_flow: dict[tuple[str, str], int],
        import_reserve: dict[tuple[str, str, str], int],
    ) -> tuple[dict[str, frozenset[str]], dict[str, float], dict[str, float]]:
        """Return used areas, utilization, and cap-relative pressure."""

        used_areas_by_group: defaultdict[str, set[str]] = defaultdict(set)
        planned_slot_load_by_area: Counter[str] = Counter()
        for (group_id, bay_key), quantity in export_flow.items():
            if int(quantity) <= 0:
                continue
            area_no = str(self.bays[bay_key].area_no)
            used_areas_by_group[group_id].add(area_no)
            planned_slot_load_by_area[area_no] += (
                self._container_slot_units(
                    self.groups_by_id[group_id].size
                )
                * int(quantity)
            )
        for (area_no, _flow, size), quantity in import_reserve.items():
            if int(quantity) <= 0:
                continue
            planned_slot_load_by_area[str(area_no)] += (
                self._container_slot_units(size) * int(quantity)
            )
        area_capacity = self._peak_utilization_policy["area_capacity"]
        utilization = {
            str(area_no): (
                float(planned_slot_load_by_area.get(str(area_no), 0))
                / max(1.0, float(capacity))
            )
            for area_no, capacity in area_capacity.items()
        }
        peak_cap = max(
            1e-12,
            float(self._peak_utilization_policy["epsilon_cap"]),
        )
        pressure = {
            area_no: min(1.0, max(0.0, value / peak_cap))
            for area_no, value in utilization.items()
        }
        return (
            {
                group.group_id: frozenset(
                    used_areas_by_group[group.group_id]
                )
                for group in self.groups
            },
            dict(sorted(utilization.items())),
            dict(sorted(pressure.items())),
        )

    def _build_group_conflict_scores(
        self,
        selected_zone_indices: set[int],
        export_flow: dict[tuple[str, str], int],
        import_reserve: dict[tuple[str, str, str], int],
    ) -> tuple[
        dict[tuple[str, str], dict[str, float]],
        dict[str, object],
    ]:
        """Build equal-weight incumbent-aware group coupling scores.

        The four components are normalized independently to [0, 1].  No zone
        pairs are materialized: candidate reachability comes from atomic row
        placements prepared before root generation.
        """

        incumbent_resources: defaultdict[str, set[Resource]] = defaultdict(set)
        for zone_index in selected_zone_indices:
            zone = self._zones[zone_index]
            incumbent_resources[zone.group_id].update(zone.resources)
        used_areas, utilization, pressure = self._incumbent_area_state(
            export_flow,
            import_reserve,
        )
        group_ids = sorted(group.group_id for group in self.groups)
        scores: dict[tuple[str, str], dict[str, float]] = {}
        component_totals: Counter[str] = Counter()
        nonzero_pair_count = 0
        maximum_score = 0.0
        for left_position, left_id in enumerate(group_ids):
            left_group = self.groups_by_id[left_id]
            left_areas = self._candidate_areas_by_group[left_id]
            left_candidates = self._candidate_resources_by_group[left_id]
            left_incumbent = frozenset(incumbent_resources[left_id])
            for right_id in group_ids[left_position + 1 :]:
                right_group = self.groups_by_id[right_id]
                right_areas = self._candidate_areas_by_group[right_id]
                right_candidates = self._candidate_resources_by_group[right_id]
                right_incumbent = frozenset(incumbent_resources[right_id])
                same_voyage = float(
                    left_group.voyage_id == right_group.voyage_id
                )
                shared_area = self._set_jaccard(left_areas, right_areas)
                candidate_resource_overlap = self._set_jaccard(
                    left_candidates,
                    right_candidates,
                )
                left_release = (
                    len(left_incumbent & right_candidates)
                    / len(left_incumbent)
                    if left_incumbent
                    else 0.0
                )
                right_release = (
                    len(right_incumbent & left_candidates)
                    / len(right_incumbent)
                    if right_incumbent
                    else 0.0
                )
                resource_conflict = max(
                    candidate_resource_overlap,
                    left_release,
                    right_release,
                )
                exchange_areas = (
                    (used_areas[left_id] & right_areas)
                    | (used_areas[right_id] & left_areas)
                    | (
                        left_areas
                        & right_areas
                        & (used_areas[left_id] | used_areas[right_id])
                    )
                )
                peak_exchange = max(
                    (pressure.get(area_no, 0.0) for area_no in exchange_areas),
                    default=0.0,
                )
                components = {
                    "same_voyage": same_voyage,
                    "shared_area": float(shared_area),
                    "resource_conflict": float(resource_conflict),
                    "peak_exchange": float(peak_exchange),
                }
                score = sum(components.values()) / len(components)
                scores[(left_id, right_id)] = {
                    **components,
                    "score": float(score),
                }
                for name, value in components.items():
                    component_totals[name] += float(value)
                if score > 1e-12:
                    nonzero_pair_count += 1
                maximum_score = max(maximum_score, score)
        pair_count = len(scores)
        return scores, {
            "policy": "equal_weight_four_component_group_conflict_graph",
            "component_names": [
                "same_voyage",
                "shared_area",
                "resource_conflict",
                "peak_exchange",
            ],
            "component_weights": {
                "same_voyage": 0.25,
                "shared_area": 0.25,
                "resource_conflict": 0.25,
                "peak_exchange": 0.25,
            },
            "pair_count": pair_count,
            "nonzero_pair_count": nonzero_pair_count,
            "maximum_score": float(maximum_score),
            "mean_score": (
                sum(values["score"] for values in scores.values())
                / pair_count
                if pair_count
                else 0.0
            ),
            "mean_components": {
                name: float(component_totals[name]) / pair_count
                if pair_count
                else 0.0
                for name in (
                    "same_voyage",
                    "shared_area",
                    "resource_conflict",
                    "peak_exchange",
                )
            },
            "incumbent_utilization_by_area": utilization,
            "incumbent_pressure_by_area": pressure,
            "full_zone_pair_materialization_used": False,
        }

    @staticmethod
    def _conflict_pair_key(left_id: str, right_id: str) -> tuple[str, str]:
        return tuple(sorted((left_id, right_id)))

    def _select_conflict_fix_optimize_neighborhood(
        self,
        selected_zone_indices: set[int],
        export_flow: dict[tuple[str, str], int],
        import_reserve: dict[tuple[str, str, str], int],
        *,
        round_id: int,
        candidate_zone_fraction: float,
        excluded_seed_groups: set[str] | None = None,
    ) -> tuple[set[str], dict[str, object]]:
        """Select one seed and its strongest incumbent-aware conflicts."""

        contribution_by_group = self._objective_contribution_by_group(
            selected_zone_indices,
            export_flow,
        )
        conflict_scores, graph = self._build_group_conflict_scores(
            selected_zone_indices,
            export_flow,
            import_reserve,
        )
        candidate_count_by_group = {
            group.group_id: int(
                self._possible_zone_count_by_group[group.group_id]
            )
            for group in self.groups
        }
        objective_ranked = sorted(
            contribution_by_group,
            key=lambda group_id: (
                -contribution_by_group[group_id],
                candidate_count_by_group[group_id],
                group_id,
            ),
        )
        excluded = set(excluded_seed_groups or set())
        available_seeds = [
            group_id for group_id in objective_ranked if group_id not in excluded
        ]
        seed_cycle_reset = not available_seeds and bool(objective_ranked)
        if seed_cycle_reset:
            available_seeds = objective_ranked
        seed_group = available_seeds[0] if available_seeds else None
        candidate_budget = max(
            1,
            int(
                math.ceil(
                    float(candidate_zone_fraction)
                    * max(1, int(self._possible_zone_count))
                )
            ),
        )
        if seed_group is None:
            return set(), {
                "policy": "conflict_aware_objective_seed_under_zone_budget",
                "round_id": int(round_id),
                "seed_group": None,
                "selected_groups": [],
                "selected_group_count": 0,
                "candidate_zone_fraction": float(candidate_zone_fraction),
                "candidate_zone_budget": candidate_budget,
                "selected_candidate_zone_count": 0,
                "conflict_graph": graph,
            }
        selected = [seed_group]
        selected_set = {seed_group}
        selected_candidate_count = candidate_count_by_group[seed_group]
        selected_links = []
        while len(selected_set) < len(objective_ranked):
            ranked_candidates = []
            for group_id in objective_ranked:
                if group_id in selected_set:
                    continue
                if (
                    selected_candidate_count
                    + candidate_count_by_group[group_id]
                    > candidate_budget
                ):
                    continue
                best_anchor = None
                best_score = -1.0
                best_components = None
                for anchor in selected:
                    values = conflict_scores[
                        self._conflict_pair_key(group_id, anchor)
                    ]
                    if (
                        values["score"] > best_score + 1e-12
                        or (
                            abs(values["score"] - best_score) <= 1e-12
                            and (best_anchor is None or anchor < best_anchor)
                        )
                    ):
                        best_anchor = anchor
                        best_score = float(values["score"])
                        best_components = values
                ranked_candidates.append(
                    (
                        -best_score,
                        -contribution_by_group[group_id],
                        candidate_count_by_group[group_id],
                        group_id,
                        best_anchor,
                        best_components,
                    )
                )
            if not ranked_candidates:
                break
            (
                _negative_score,
                _negative_contribution,
                candidate_count,
                group_id,
                anchor,
                components,
            ) = min(ranked_candidates)
            selected.append(group_id)
            selected_set.add(group_id)
            selected_candidate_count += candidate_count
            selected_links.append(
                {
                    "group_id": group_id,
                    "anchor_group": anchor,
                    **dict(components or {}),
                }
            )
        return selected_set, {
            "policy": "conflict_aware_objective_seed_under_zone_budget",
            "round_id": int(round_id),
            "seed_group": seed_group,
            "seed_contribution": float(contribution_by_group[seed_group]),
            "seed_cycle_reset": seed_cycle_reset,
            "excluded_seed_groups": sorted(excluded),
            "selected_groups": list(selected),
            "selected_group_count": len(selected),
            "candidate_zone_fraction": float(candidate_zone_fraction),
            "candidate_zone_budget": candidate_budget,
            "selected_candidate_zone_count": selected_candidate_count,
            "candidate_budget_exceeded_by_seed": (
                candidate_count_by_group[seed_group] > candidate_budget
            ),
            "selected_conflict_links": selected_links,
            "objective_contribution_by_group": dict(
                sorted(contribution_by_group.items())
            ),
            "conflict_graph": graph,
        }

    def _objective_fix_optimize_neighborhood(
        self,
        selected_zone_indices: set[int],
        export_flow: dict[tuple[str, str], int],
        *,
        candidate_zone_fraction: float | None = None,
    ) -> tuple[set[str], dict[str, object]]:
        """Legacy objective-only neighborhood retained only for ablation."""

        contribution_by_group = self._objective_contribution_by_group(
            selected_zone_indices,
            export_flow,
        )

        candidate_count_by_group = {
            group.group_id: int(
                self._possible_zone_count_by_group[group.group_id]
            )
            for group in self.groups
        }
        total_contribution = float(sum(contribution_by_group.values()))
        target_contribution = (
            float(self.zone_config.fix_optimize_objective_mass)
            * total_contribution
        )
        zone_fraction = float(
            self.zone_config.fix_optimize_zone_fraction
            if candidate_zone_fraction is None
            else candidate_zone_fraction
        )
        candidate_budget = max(
            1,
            int(
                math.ceil(
                    zone_fraction
                    * max(1, int(self._possible_zone_count))
                )
            ),
        )
        objective_ranked = sorted(
            contribution_by_group,
            key=lambda group_id: (
                -contribution_by_group[group_id],
                candidate_count_by_group[group_id],
                group_id,
            ),
        )
        selected: list[str] = []
        skipped_by_budget: list[str] = []
        selected_candidate_count = 0
        selected_contribution = 0.0
        for group_id in objective_ranked:
            candidate_count = candidate_count_by_group[group_id]
            if (
                selected
                and selected_candidate_count + candidate_count
                > candidate_budget
            ):
                skipped_by_budget.append(group_id)
                continue
            selected.append(group_id)
            selected_candidate_count += candidate_count
            selected_contribution += contribution_by_group[group_id]
            if (
                total_contribution <= 1e-12
                or selected_contribution + 1e-12 >= target_contribution
            ):
                break
        if not selected and objective_ranked:
            group_id = objective_ranked[0]
            selected.append(group_id)
            selected_candidate_count = candidate_count_by_group[group_id]
            selected_contribution = contribution_by_group[group_id]
        achieved_mass = (
            selected_contribution / total_contribution
            if total_contribution > 1e-12
            else 1.0
        )
        target_met = (
            total_contribution <= 1e-12
            or selected_contribution + 1e-12 >= target_contribution
        )
        return set(selected), {
            "policy": "objective_mass_under_candidate_zone_fraction",
            "objective_mass_target": float(
                self.zone_config.fix_optimize_objective_mass
            ),
            "objective_mass_achieved": float(achieved_mass),
            "objective_mass_target_met": target_met,
            "binding_condition": (
                "objective_mass_target"
                if target_met
                else "candidate_zone_budget"
            ),
            "attributable_objective_total": total_contribution,
            "selected_attributable_objective": selected_contribution,
            "candidate_zone_fraction_limit": zone_fraction,
            "candidate_zone_budget": candidate_budget,
            "selected_candidate_zone_count": selected_candidate_count,
            "selected_candidate_zone_fraction": (
                selected_candidate_count / max(1, self._possible_zone_count)
            ),
            "candidate_budget_exceeded_by_first_group": (
                len(selected) == 1
                and selected_candidate_count > candidate_budget
            ),
            "selected_group_count": len(selected),
            "selected_groups": list(selected),
            "skipped_by_candidate_budget_count": len(skipped_by_budget),
        }

    def _solve_objective_fix_optimize_subproblem(
        self,
        selected_zone_indices: set[int],
        export_flow: dict[tuple[str, str], int],
        import_reserve: dict[tuple[str, str, str], int],
        deadline: float,
        *,
        neighborhood: set[str] | None = None,
        neighborhood_selection: dict[str, object] | None = None,
        progress_phase: str = "fix_optimize_local_mip",
    ) -> tuple[
        set[int],
        dict[tuple[str, str], int],
        dict[tuple[str, str, str], int],
        dict[str, object],
    ]:
        """Jointly reopen complete zones for the objective neighborhood."""

        started = perf_counter()
        remaining = deadline - started
        if remaining <= 1e-6:
            return set(), {}, {}, {
                "status": "time_limit_before_fix_optimize",
                "build_seconds": 0.0,
                "local_mip_seconds": 0.0,
                "seconds": 0.0,
            }
        if neighborhood is None or neighborhood_selection is None:
            neighborhood, neighborhood_selection = (
                self._objective_fix_optimize_neighborhood(
                    selected_zone_indices, export_flow
                )
            )
        materialized_before = len(self._zones)
        for group_id in sorted(neighborhood):
            for strip_key, signature in self._iter_zone_signatures(group_id):
                self._register_zone(strip_key, signature)
        neighborhood_zone_indices = {
            index
            for group_id in neighborhood
            for index in self._zone_indices_by_group[group_id]
        }
        fixed_zone_indices = {
            index
            for index in selected_zone_indices
            if self._zones[index].group_id not in neighborhood
        }
        active_zone_indices = neighborhood_zone_indices | fixed_zone_indices
        model, variables = self._build_zone_master()
        try:
            self._remove_proof_only_area_rows(model, variables)
            for zone_index in sorted(active_zone_indices):
                self._add_zone_variable(model, variables, zone_index)
            for zone_index, variable in variables["zone"].items():
                if self._zones[zone_index].group_id not in neighborhood:
                    variable.LB = 1.0
                    variable.UB = 1.0
                variable.Start = (
                    1.0 if zone_index in selected_zone_indices else 0.0
                )
                variable.VType = "B"
            for variable in variables["shortage"].values():
                variable.UB = 0.0
                variable.Start = 0.0
            for key, variable in variables["export_flow"].items():
                if key[0] not in neighborhood:
                    fixed = int(export_flow.get(key, 0))
                    variable.LB = float(fixed)
                    variable.UB = float(fixed)
                variable.Start = float(export_flow.get(key, 0))
                variable.VType = "I"
            used_areas = {
                (group_id, self.bays[bay_key].area_no)
                for (group_id, bay_key), quantity in export_flow.items()
                if int(quantity) > 0
            }
            for key, variable in variables["area_use"].items():
                variable.Start = 1.0 if key in used_areas else 0.0
                variable.VType = "B"
            used_voyage_areas = {
                (
                    self.groups_by_id[group_id].voyage_id,
                    area_no,
                )
                for group_id, area_no in used_areas
            }
            for key, variable in variables["voyage_area_use"].items():
                variable.Start = 1.0 if key in used_voyage_areas else 0.0
                variable.VType = "B"
            used_attributes = {
                key
                for zone_index in selected_zone_indices
                for key, value in self._zones[zone_index].bay_attr_uses
                if value > 0
            }
            for key, variable in variables["attr_state"].items():
                variable.Start = 1.0 if key in used_attributes else 0.0
                variable.VType = "B"
            for key, variable in variables["import_reserve"].items():
                variable.Start = float(import_reserve.get(key, 0))
                variable.VType = "I"
            model.update()
            build_seconds = perf_counter() - started
            remaining = deadline - perf_counter()
            if remaining <= 1e-6:
                return set(), {}, {}, {
                    "status": "time_limit_after_fix_optimize_build",
                    "has_solution": False,
                    "neighborhood": sorted(neighborhood),
                    "neighborhood_group_count": len(neighborhood),
                    "neighborhood_selection": neighborhood_selection,
                    "active_zone_count": len(active_zone_indices),
                    "materialized_zone_count": len(self._zones)
                    - materialized_before,
                    "build_seconds": build_seconds,
                    "local_mip_seconds": 0.0,
                    "seconds": perf_counter() - started,
                }
            self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
            self._set_gurobi_param(model, "MIPGap", 0.0)
            self._set_gurobi_param(model, "MIPFocus", 1)
            self._set_gurobi_param(model, "Heuristics", 0.30)
            progress_recorder = MipProgressRecorder(
                phase=progress_phase
            )
            model.optimize(progress_recorder)
            progress = self._finalize_mip_progress(progress_recorder, model)
            status = self._gurobi_status_name(model)
            if self._gurobi_solution_count(model) <= 0:
                return set(), {}, {}, {
                    "status": status,
                    "has_solution": False,
                    "neighborhood": sorted(neighborhood),
                    "neighborhood_group_count": len(neighborhood),
                    "neighborhood_selection": neighborhood_selection,
                    "active_zone_count": len(active_zone_indices),
                    "mip_progress": progress,
                    "build_seconds": build_seconds,
                    "local_mip_seconds": float(
                        progress.get("wall_seconds", 0.0)
                    ),
                    "seconds": perf_counter() - started,
                }
            selected = {
                index
                for index, variable in variables["zone"].items()
                if self._gurobi_value(model, variable) > 0.5
            }
            flow = {
                key: int(round(self._gurobi_value(model, variable)))
                for key, variable in variables["export_flow"].items()
                if self._gurobi_value(model, variable) > 1e-7
            }
            imports = {
                key: int(round(self._gurobi_value(model, variable)))
                for key, variable in variables["import_reserve"].items()
                if self._gurobi_value(model, variable) > 1e-7
            }
            objective = self._gurobi_objective_value(model)
            bound = self._gurobi_dual_bound(model)
            certificate = self._zone_objective_certificate(
                selected, flow, imports, objective
            )
            return selected, flow, imports, {
                "status": status,
                "has_solution": True,
                "objective": float(certificate["objective"]),
                "solver_objective": objective,
                "bound": bound,
                "relative_gap": max(0.0, objective - bound)
                / max(abs(objective), 1e-12),
                "neighborhood": sorted(neighborhood),
                "neighborhood_group_count": len(neighborhood),
                "neighborhood_selection": neighborhood_selection,
                "active_zone_count": len(active_zone_indices),
                "materialized_zone_count": len(self._zones) - materialized_before,
                "selected_zone_count": len(selected),
                "positive_export_flow_count": len(flow),
                "positive_import_reservation_count": len(imports),
                "mip_progress": progress,
                "build_seconds": build_seconds,
                "local_mip_seconds": float(
                    progress.get("wall_seconds", 0.0)
                ),
                "seconds": perf_counter() - started,
            }
        finally:
            self._free_gurobi_model(model)

    def _run_objective_fix_optimize(
        self,
        model,
        variables: dict,
        incumbent_zones: set[int],
        incumbent_export_flow: dict[tuple[str, str], int],
        incumbent_import_reserve: dict[tuple[str, str, str], int],
        deadline: float,
        zone_provenance: defaultdict[int, set[str]] | None = None,
    ) -> tuple[
        set[int],
        dict[tuple[str, str], int],
        dict[tuple[str, str, str], int],
        dict[str, object],
    ]:
        """Run conflict-aware adaptive F&O without changing the master model."""

        started = perf_counter()
        incumbent_certificate = self._zone_objective_certificate(
            incumbent_zones,
            incumbent_export_flow,
            incumbent_import_reserve,
            self._gurobi_objective_value(model),
        )
        initial_objective = float(incumbent_certificate["objective"])
        current_zones = set(incumbent_zones)
        current_flow = dict(incumbent_export_flow)
        current_import = dict(incumbent_import_reserve)
        current_objective = initial_objective
        unsuccessful_seeds: set[str] = set()
        attempted_seeds: list[str] = []
        rounds: list[dict[str, object]] = []
        policy = self.zone_config.fix_optimize_policy
        if policy == "objective":
            max_rounds = 1
            round_fractions = (float(self.zone_config.fix_optimize_zone_fraction),)
        else:
            max_rounds = int(self.zone_config.fix_optimize_max_rounds)
            round_fractions = tuple(
                float(value)
                for value in self.zone_config.fix_optimize_round_zone_fractions[
                    :max_rounds
                ]
            )
        stop_reason = "max_rounds"
        for round_offset in range(max_rounds):
            round_id = round_offset + 1
            remaining = deadline - perf_counter()
            if remaining <= 1e-6:
                stop_reason = "time_limit_before_round"
                break
            remaining_rounds = max_rounds - round_offset
            outer_deadline = min(
                deadline,
                perf_counter() + remaining / remaining_rounds,
            )
            selection_started = perf_counter()
            objective_round = policy == "objective" or (
                policy == "hybrid_multi_round" and round_offset == 0
            )
            if objective_round:
                neighborhood, selection = (
                    self._objective_fix_optimize_neighborhood(
                        current_zones,
                        current_flow,
                        candidate_zone_fraction=(
                            round_fractions[round_offset]
                            if policy == "hybrid_multi_round"
                            else None
                        ),
                    )
                )
                selection = {
                    **selection,
                    "round_id": round_id,
                    "seed_group": (
                        selection.get("selected_groups", [None])[0]
                        if selection.get("selected_groups")
                        else None
                    ),
                    "selection_family": "objective",
                }
            else:
                neighborhood, selection = (
                    self._select_conflict_fix_optimize_neighborhood(
                        current_zones,
                        current_flow,
                        current_import,
                        round_id=round_id,
                        candidate_zone_fraction=round_fractions[round_offset],
                        excluded_seed_groups=unsuccessful_seeds,
                    )
                )
                selection = {
                    **selection,
                    "selection_family": "conflict",
                }
            selection_seconds = perf_counter() - selection_started
            seed_group = selection.get("seed_group")
            if seed_group is not None:
                attempted_seeds.append(str(seed_group))
            if not neighborhood:
                stop_reason = "no_neighborhood"
                break
            (
                next_zones,
                next_flow,
                next_import,
                round_diagnostics,
            ) = self._run_fix_optimize_round(
                model,
                variables,
                current_zones,
                current_flow,
                current_import,
                current_objective,
                round_id=round_id,
                neighborhood=neighborhood,
                neighborhood_selection=selection,
                selection_seconds=selection_seconds,
                deadline=outer_deadline,
                zone_provenance=zone_provenance,
            )
            rounds.append(round_diagnostics)
            current_zones = next_zones
            current_flow = next_flow
            current_import = next_import
            current_objective = float(
                round_diagnostics["objective_after"]
            )
            if round_diagnostics["improved"]:
                unsuccessful_seeds.clear()
            elif seed_group is not None:
                unsuccessful_seeds.add(str(seed_group))
        if len(rounds) < max_rounds and stop_reason == "max_rounds":
            stop_reason = "time_limit"
        absolute_improvement = max(0.0, initial_objective - current_objective)
        return current_zones, current_flow, current_import, {
            "status": (
                "completed" if stop_reason == "max_rounds" else stop_reason
            ),
            "policy": policy,
            "initial_objective": initial_objective,
            "final_objective": current_objective,
            "rounds": rounds,
            "round_count": len(rounds),
            "successful_round_count": sum(
                bool(values.get("improved")) for values in rounds
            ),
            "attempted_seed_groups": attempted_seeds,
            "unsuccessful_seed_groups_at_end": sorted(unsuccessful_seeds),
            "round_zone_fractions": list(round_fractions),
            "stop_reason": stop_reason,
            "added_zone_count": sum(
                int(values.get("added_zone_count", 0)) for values in rounds
            ),
            "improved": absolute_improvement > 1e-9,
            "absolute_improvement": absolute_improvement,
            "relative_improvement": absolute_improvement
            / max(abs(initial_objective), 1e-12),
            "seconds": perf_counter() - started,
        }

    def _run_fix_optimize_round(
        self,
        model,
        variables: dict,
        incumbent_zones: set[int],
        incumbent_export_flow: dict[tuple[str, str], int],
        incumbent_import_reserve: dict[tuple[str, str, str], int],
        incumbent_objective: float,
        *,
        round_id: int,
        neighborhood: set[str],
        neighborhood_selection: dict[str, object],
        selection_seconds: float,
        deadline: float,
        zone_provenance: defaultdict[int, set[str]] | None,
    ) -> tuple[
        set[int],
        dict[tuple[str, str], int],
        dict[tuple[str, str, str], int],
        dict[str, object],
    ]:
        """Solve one local neighborhood and feed its used columns to master."""

        started = perf_counter()
        local_deadline = perf_counter() + max(
            0.0,
            deadline - perf_counter(),
        ) * float(self.zone_config.fix_optimize_local_fraction)
        local_zones, local_flow, local_import, local = (
            self._solve_objective_fix_optimize_subproblem(
                incumbent_zones,
                incumbent_export_flow,
                incumbent_import_reserve,
                local_deadline,
                neighborhood=set(neighborhood),
                neighborhood_selection=neighborhood_selection,
                progress_phase=f"fix_optimize_round_{round_id}_local_mip",
            )
        )
        local_objective = (
            float(local["objective"]) if local_zones else None
        )
        added_indices: list[int] = []
        if local_zones:
            for zone_index in sorted(local_zones):
                if zone_provenance is not None:
                    zone_provenance[zone_index].add(
                        f"fix_opt_round_{round_id}"
                    )
                if zone_index not in variables["active_zone_indices"]:
                    variable = self._add_zone_variable(
                        model,
                        variables,
                        zone_index,
                    )
                    variable.VType = "B"
                    added_indices.append(zone_index)
        use_local_start = (
            local_objective is not None
            and local_objective <= incumbent_objective + 1e-9
        )
        start_zones = local_zones if use_local_start else incumbent_zones
        start_flow = local_flow if use_local_start else incumbent_export_flow
        start_import = (
            local_import if use_local_start else incumbent_import_reserve
        )
        if local_zones:
            self._apply_zone_master_start(
                variables,
                start_zones,
                start_flow,
                start_import,
            )
            model.update()
        best_zones = set(incumbent_zones)
        best_flow = dict(incumbent_export_flow)
        best_import = dict(incumbent_import_reserve)
        best_objective = float(incumbent_objective)
        if local_objective is not None and local_objective < best_objective - 1e-9:
            best_zones = set(local_zones)
            best_flow = dict(local_flow)
            best_import = dict(local_import)
            best_objective = local_objective
        master_diagnostics: dict[str, object] = {
            "status": "not_run",
            "mip_progress": None,
            "seconds": 0.0,
        }
        remaining = deadline - perf_counter()
        if local_zones and remaining > 1e-6:
            self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
            recorder = MipProgressRecorder(
                phase=f"fix_optimize_round_{round_id}_master_reoptimization"
            )
            master_started = perf_counter()
            model.optimize(recorder)
            progress = self._finalize_mip_progress(recorder, model)
            master_seconds = perf_counter() - master_started
            master_status = self._gurobi_status_name(model)
            master_diagnostics = {
                "status": master_status,
                "mip_progress": progress,
                "seconds": master_seconds,
            }
            if self._gurobi_solution_count(model) > 0:
                candidate_zones = {
                    index
                    for index, variable in variables["zone"].items()
                    if self._gurobi_value(model, variable) > 0.5
                }
                candidate_flow = {
                    key: int(round(self._gurobi_value(model, variable)))
                    for key, variable in variables["export_flow"].items()
                    if self._gurobi_value(model, variable) > 1e-7
                }
                candidate_import = {
                    key: int(round(self._gurobi_value(model, variable)))
                    for key, variable in variables["import_reserve"].items()
                    if self._gurobi_value(model, variable) > 1e-7
                }
                certificate = self._zone_objective_certificate(
                    candidate_zones,
                    candidate_flow,
                    candidate_import,
                    self._gurobi_objective_value(model),
                )
                candidate_objective = float(certificate["objective"])
                master_diagnostics["objective"] = candidate_objective
                if candidate_objective < best_objective - 1e-9:
                    best_zones = candidate_zones
                    best_flow = candidate_flow
                    best_import = candidate_import
                    best_objective = candidate_objective
        elif local_zones:
            master_diagnostics["status"] = (
                "time_limit_before_master_reoptimization"
            )
        absolute_improvement = max(0.0, incumbent_objective - best_objective)
        local_progress = local.get("mip_progress", {})
        round_diagnostics = {
            "round_id": int(round_id),
            "seed_group": neighborhood_selection.get("seed_group"),
            "selected_groups": list(
                neighborhood_selection.get(
                    "selected_groups", sorted(neighborhood)
                )
            ),
            "neighborhood_group_count": len(neighborhood),
            "candidate_zone_budget": neighborhood_selection.get(
                "candidate_zone_budget"
            ),
            "candidate_zone_count": neighborhood_selection.get(
                "selected_candidate_zone_count"
            ),
            "objective_before": float(incumbent_objective),
            "local_objective": local_objective,
            "master_reoptimized_objective": master_diagnostics.get(
                "objective"
            ),
            "objective_after": float(best_objective),
            "absolute_improvement": absolute_improvement,
            "relative_improvement": absolute_improvement
            / max(abs(incumbent_objective), 1e-12),
            "selection_seconds": float(selection_seconds),
            "build_seconds": float(local.get("build_seconds", 0.0)),
            "local_mip_seconds": float(
                local.get("local_mip_seconds", 0.0)
            ),
            "master_reoptimization_seconds": float(
                master_diagnostics.get("seconds", 0.0)
            ),
            "local_nodes": float(local_progress.get("node_count", 0.0)),
            "local_time_to_first": local_progress.get(
                "time_to_first_solution"
            ),
            "local_time_to_best": local_progress.get(
                "time_to_best_solution"
            ),
            "added_zone_count": len(added_indices),
            "added_zone_indices": added_indices,
            "improved": absolute_improvement > 1e-9,
            "neighborhood_selection": neighborhood_selection,
            "local": local,
            "master_reoptimization": master_diagnostics,
            "seconds": perf_counter() - started,
        }
        return best_zones, best_flow, best_import, round_diagnostics

    def _apply_zone_master_start(
        self,
        variables: dict,
        selected_zones: set[int],
        export_flow: dict[tuple[str, str], int],
        import_reserve: dict[tuple[str, str, str], int],
    ) -> None:
        """Apply one certified incumbent as a native master MIP start."""

        for zone_index, variable in variables["zone"].items():
            variable.Start = 1.0 if zone_index in selected_zones else 0.0
        for key, variable in variables["export_flow"].items():
            variable.Start = float(export_flow.get(key, 0))
        for key, variable in variables["import_reserve"].items():
            variable.Start = float(import_reserve.get(key, 0))
        used_areas = {
            (group_id, self.bays[bay_key].area_no)
            for (group_id, bay_key), quantity in export_flow.items()
            if int(quantity) > 0
        }
        for key, variable in variables["area_use"].items():
            variable.Start = 1.0 if key in used_areas else 0.0
        used_voyage_areas = {
            (self.groups_by_id[group_id].voyage_id, area_no)
            for group_id, area_no in used_areas
        }
        for key, variable in variables["voyage_area_use"].items():
            variable.Start = 1.0 if key in used_voyage_areas else 0.0
        used_attributes = {
            key
            for zone_index in selected_zones
            for key, value in self._zones[zone_index].bay_attr_uses
            if value > 0
        }
        for key, variable in variables["attr_state"].items():
            variable.Start = 1.0 if key in used_attributes else 0.0

    def _construct_zone_row_realization(
        self,
        selected_zone_indices: set[int],
        candidate_indices: set[int],
        export_flow: dict[tuple[str, str], int],
    ) -> tuple[Counter[int], dict[str, object]]:
        resource_owner: dict[Resource, int] = {}
        overlaps = []
        for zone_index in sorted(selected_zone_indices):
            for resource in self._zones[zone_index].resources:
                prior = resource_owner.setdefault(resource, zone_index)
                if prior != zone_index:
                    overlaps.append((resource, prior, zone_index))
        if overlaps:
            raise RuntimeError(
                "selected zones overlap dedicated physical rows: "
                f"{overlaps[:5]}"
            )
        candidates_by_flow: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        for index in sorted(candidate_indices):
            column = self._columns[index]
            candidates_by_flow[(column.group_id, column.bay_key)].append(index)
        selected: Counter[int] = Counter()
        shortfalls = {}
        for key in sorted(set(candidates_by_flow) | set(export_flow)):
            remaining = int(export_flow.get(key, 0))
            for index in sorted(
                candidates_by_flow.get(key, []),
                key=lambda candidate_index: (
                    float(self._columns[candidate_index].intrinsic_cost),
                    candidate_index,
                ),
            ):
                quantity = min(remaining, int(self._atomic_capacity[index]))
                if quantity > 0:
                    selected[index] = quantity
                    remaining -= quantity
                if remaining <= 0:
                    break
            if remaining > 0:
                shortfalls[key] = remaining
        if shortfalls:
            raise RuntimeError(
                "selected zone capacities cannot realize master export flows: "
                f"{shortfalls}"
            )
        assigned_by_group: Counter[str] = Counter()
        for (group_id, _bay_key), quantity in export_flow.items():
            assigned_by_group[group_id] += int(quantity)
        demand_mismatches = {
            group.group_id: (
                int(assigned_by_group[group.group_id]),
                int(group.demand),
            )
            for group in self.groups
            if int(assigned_by_group[group.group_id]) != int(group.demand)
        }
        if demand_mismatches:
            raise RuntimeError(
                "zone master export flows do not match group demand: "
                f"{demand_mismatches}"
            )
        return selected, {
            "certified": True,
            "proof": (
                "constructive_assignment_within_disjoint_dedicated_"
                "row_capacities"
            ),
            "positive_flow_count": sum(value > 0 for value in export_flow.values()),
            "constructed_row_count": len(selected),
            "constructed_box_count": sum(selected.values()),
            "dedicated_resource_count": len(resource_owner),
            "overlapping_resource_count": 0,
            "shortfall_count": 0,
            "demand_mismatch_count": 0,
        }

    def _reconstruct_zone_objective(
        self,
        selected_zone_indices: set[int],
        export_flow: dict[tuple[str, str], int],
        import_reserve: dict[tuple[str, str, str], int],
    ) -> dict[str, object]:
        """Reconstruct the primary objective without relying on solver state."""

        reserved_capacity = sum(
            int(self._zones[index].capacity)
            for index in selected_zone_indices
        )
        assigned_boxes = sum(int(value) for value in export_flow.values())
        if assigned_boxes > reserved_capacity:
            raise RuntimeError(
                "zone solution assigns more export flow than reserved capacity: "
                f"assigned={assigned_boxes}, reserved={reserved_capacity}"
            )
        used_group_areas = {
            (group_id, self.bays[bay_key].area_no)
            for (group_id, bay_key), quantity in export_flow.items()
            if int(quantity) > 0
        }
        used_voyage_areas = {
            (self.groups_by_id[group_id].voyage_id, area_no)
            for group_id, area_no in used_group_areas
        }
        voyage_count = len({group.voyage_id for group in self.groups})
        if len(selected_zone_indices) < len(self.groups):
            raise RuntimeError(
                "a positive-demand group has no selected contiguous zone"
            )
        if len(used_group_areas) < len(self.groups):
            raise RuntimeError(
                "a positive-demand group has no positive area flow"
            )

        proximity_sum = 0.0
        berth_distance_sum = 0.0
        planned_slot_load_by_area: Counter[str] = Counter()
        for (group_id, bay_key), quantity in export_flow.items():
            if int(quantity) <= 0:
                continue
            group = self.groups_by_id[group_id]
            proximity_sum += (
                self._normalized_existing_proximity(group, bay_key)
                * int(quantity)
            )
            berth_distance_sum += (
                self._normalized_berth_distance(
                    group.voyage_id, self.bays[bay_key].area_no
                )
                * int(quantity)
            )
            planned_slot_load_by_area[self.bays[bay_key].area_no] += (
                int(quantity)
                * len(
                    self._placement_footprint_keys(
                        bay_key, group.size
                    )
                )
            )
        for (_flow, size, bay_key), quantity in import_reserve.items():
            if int(quantity) <= 0:
                continue
            planned_slot_load_by_area[self.bays[bay_key].area_no] += (
                int(quantity)
                * len(self._placement_footprint_keys(bay_key, size))
            )
        area_capacity = self._peak_utilization_policy["area_capacity"]
        utilization_by_area = {
            area_no: (
                float(load) / float(area_capacity[area_no])
                if int(area_capacity.get(area_no, 0)) > 0
                else math.inf
            )
            for area_no, load in sorted(planned_slot_load_by_area.items())
        }
        maximum_utilization = max(utilization_by_area.values(), default=0.0)
        peak_cap = float(self._peak_utilization_policy["epsilon_cap"])
        if maximum_utilization > peak_cap + 1e-7:
            raise RuntimeError(
                "zone solution violates the peak-utilization epsilon cap: "
                f"actual={maximum_utilization:.9f}, cap={peak_cap:.9f}"
            )
        raw = {
            "extra_voyage_areas": float(
                len(used_voyage_areas) - voyage_count
            ),
            "extra_group_areas": float(
                len(used_group_areas) - len(self.groups)
            ),
            "extra_contiguous_zones": float(
                len(selected_zone_indices) - len(self.groups)
            ),
            "existing_group_normalized_distance_sum": float(proximity_sum),
            "unused_reserved_capacity_boxes": float(
                reserved_capacity - assigned_boxes
            ),
            "berth_normalized_distance_sum": float(berth_distance_sum),
        }
        normalized = {
            "voyage_area_dispersion": raw["extra_voyage_areas"]
            / self._zone_objective_scale("voyage_area_dispersion"),
            "zone_dispersion": raw["extra_contiguous_zones"]
            / self._zone_objective_scale("zone_dispersion"),
            "existing_group_proximity": raw[
                "existing_group_normalized_distance_sum"
            ]
            / self._zone_objective_scale("existing_group_proximity"),
            "unused_capacity": raw["unused_reserved_capacity_boxes"]
            / self._zone_objective_scale("unused_capacity"),
            "berth_distance": raw["berth_normalized_distance_sum"]
            / self._zone_objective_scale("berth_distance"),
        }
        weights = self._zone_objective_weights()
        weighted = {
            key: float(normalized[key]) * float(weights[key])
            for key in weights
        }
        reconstructed = float(sum(weighted.values()))
        return {
            "objective": reconstructed,
            "raw": raw,
            "normalized": normalized,
            "weighted": weighted,
            "components": weighted,
            "weights": weights,
            "objective_design": {
                "group_area_dispersion": "diagnostic_only",
                "unused_capacity_objective_enabled": bool(
                    self.zone_config.unused_capacity_objective_enabled
                ),
            },
            "scales": dict(self._zone_objective_scales),
            "selected_zone_count": len(selected_zone_indices),
            "reserved_export_capacity": reserved_capacity,
            "assigned_export_boxes": assigned_boxes,
            "unused_reserved_capacity_boxes": reserved_capacity - assigned_boxes,
            "positive_export_flow_count": sum(
                int(value) > 0 for value in export_flow.values()
            ),
            "positive_import_reservation_count": sum(
                int(value) > 0 for value in import_reserve.values()
            ),
            "peak_utilization": {
                "maximum": float(maximum_utilization),
                "epsilon_cap": peak_cap,
                "slack": float(peak_cap - maximum_utilization),
                "planned_slot_load_by_area": dict(
                    sorted(planned_slot_load_by_area.items())
                ),
                "utilization_by_area": utilization_by_area,
                "anonymous_import_included": True,
            },
        }

    def _zone_objective_certificate(
        self,
        selected_zone_indices: set[int],
        export_flow: dict[tuple[str, str], int],
        import_reserve: dict[tuple[str, str, str], int],
        solver_objective: float,
    ) -> dict[str, object]:
        """Certify solver consistency separately from objective reconstruction."""

        reconstruction = self._reconstruct_zone_objective(
            selected_zone_indices,
            export_flow,
            import_reserve,
        )
        difference = float(solver_objective) - float(reconstruction["objective"])
        if abs(difference) > 1e-6:
            raise RuntimeError(
                "contiguous-zone solver objective does not match its physical "
                "decision certificate: "
                f"model={solver_objective}, "
                f"reconstructed={reconstruction['objective']}"
            )
        return {
            **reconstruction,
            "certified": True,
            "solver_incumbent_objective": float(solver_objective),
            "absolute_reconstruction_difference": abs(difference),
        }

    def _solve_restricted_fill(
        self,
        selected_zone_indices: set[int],
        export_flow: dict[tuple[str, str], int],
        import_reserve: dict[tuple[str, str, str], int],
        deadline: float,
    ) -> tuple[Counter[int], dict[str, object]]:
        from gurobipy import quicksum

        candidate_indices = {
            index
            for zone_index in selected_zone_indices
            for index in self._zones[zone_index].candidate_indices
        }
        constructive_start, recourse_certificate = (
            self._construct_zone_row_realization(
                selected_zone_indices,
                candidate_indices,
                export_flow,
            )
        )
        ordered_indices = sorted(candidate_indices)
        row_locations = [self._columns[index] for index in ordered_indices]
        started = perf_counter()
        model, variables, model_stats = self.build_compact_row_milp(
            row_locations,
            GurobiModel,
            quicksum,
        )
        local_by_flow: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        for local_index, global_index in enumerate(ordered_indices):
            column = self._columns[global_index]
            local_by_flow[(column.group_id, column.bay_key)].append(local_index)
        missing_positive = sorted(
            key
            for key, quantity in export_flow.items()
            if quantity > 0 and not local_by_flow.get(key)
        )
        if missing_positive:
            self._free_gurobi_model(model)
            raise RuntimeError(
                "zone flow has no detailed row candidate on selected support: "
                f"{missing_positive[:5]}"
            )
        for key, local_indices in sorted(local_by_flow.items()):
            model.addConstr(
                quicksum(variables["column"][index] for index in local_indices)
                == int(export_flow.get(key, 0)),
                name=f"zone_flow_fix_{self._key_name(key)}",
            )
        for key, variable in sorted(variables["import_reserve"].items()):
            fixed_quantity = float(import_reserve.get(key, 0))
            variable.LB = fixed_quantity
            variable.UB = fixed_quantity
        local_index_by_global = {
            global_index: local_index
            for local_index, global_index in enumerate(ordered_indices)
        }
        for global_index, quantity in constructive_start.items():
            variables["column"][local_index_by_global[global_index]].Start = float(
                quantity
            )
        model.update()
        model_stats["zone_flow_fix_count"] = len(local_by_flow)
        model_stats["zone_import_fix_count"] = len(variables["import_reserve"])
        try:
            remaining = deadline - perf_counter()
            if remaining <= 1e-6:
                raise RuntimeError("no time remains for exact zone filling")
            self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
            self._set_gurobi_param(model, "MIPGap", 0.0)
            model.optimize()
            status = self._gurobi_status_name(model)
            if self._gurobi_solution_count(model) <= 0:
                raise RuntimeError(
                    "selected contiguous zones have no exact row filling: "
                    f"status={status}, candidate_count={len(row_locations)}"
                )
            local_selected = self.selected_compact_row_values(model, variables)
            selected = Counter(
                {
                    ordered_indices[local_index]: quantity
                    for local_index, quantity in local_selected.items()
                }
            )
            self._final_import_reservation = self._gurobi_import_reservation_values(
                model, variables
            )
            bound = self._gurobi_dual_bound(model)
        finally:
            self._free_gurobi_model(model)
        realized_flow: Counter[tuple[str, str]] = Counter()
        for index, quantity in selected.items():
            column = self._columns[index]
            realized_flow[(column.group_id, column.bay_key)] += int(quantity)
        flow_mismatches = {
            key: (int(realized_flow.get(key, 0)), int(export_flow.get(key, 0)))
            for key in set(realized_flow) | set(export_flow)
            if int(realized_flow.get(key, 0)) != int(export_flow.get(key, 0))
        }
        if flow_mismatches:
            raise RuntimeError(
                "exact row fill changed the certified zone-master flow: "
                f"{flow_mismatches}"
            )
        objective = self._selected_solution_energy(selected)
        return selected, {
            "status": status,
            "objective": objective,
            "bound": bound,
            "absolute_gap": max(0.0, objective - bound),
            "relative_gap": max(0.0, objective - bound)
            / max(abs(objective), 1e-12),
            "candidate_count": len(row_locations),
            "model": model_stats,
            "recourse_certificate": recourse_certificate,
            "flow_realization_mismatch_count": 0,
            "seconds": perf_counter() - started,
        }

    def solve(self) -> ColumnGenerationResult:
        started = perf_counter()
        self._solve_started_at = started
        total_limit = max(0.01, float(self.config.total_time_limit))
        deadline = started + total_limit
        preparation = self._prepare_zones()
        proof_model, proof_variables = self._build_zone_master()
        try:
            root_deadline = perf_counter() + max(
                0.0, deadline - perf_counter()
            ) * float(self.zone_config.root_time_fraction)
            root = self._run_root_generation(
                proof_model,
                proof_variables,
                root_deadline,
            )
            if not root.get("closed"):
                raise RuntimeError(
                    "zone root pricing did not close within its adaptive time "
                    "envelope; no global zone-model bound is available"
                )
            if root.get("root_shortage") is None or float(
                root["root_shortage"]
            ) > 1e-6:
                raise RuntimeError(
                    "zone root did not cover all export demand before integerization: "
                    f"shortage={root.get('root_shortage')}"
                )
            root_snapshot = self._capture_root_snapshot(
                proof_variables,
                root,
            )
            proof_pool_indices = set(root_snapshot.proof_zone_indices)
        finally:
            self._free_gurobi_model(proof_model)

        (
            base_primal_pool_indices,
            zone_provenance,
            primal_pool_diagnostics,
        ) = self._build_integrality_aware_primal_pool(root_snapshot)
        model, variables = self._build_zone_master()
        try:
            proof_only_area_cut_count = self._remove_proof_only_area_rows(
                model, variables
            )
            for zone_index in sorted(base_primal_pool_indices):
                self._add_zone_variable(model, variables, zone_index)
            model.update()
            self._install_root_snapshot(variables, root_snapshot)
            if self.zone_config.primal_pool_apply_lp_warm_start:
                warm_start_diagnostics = {
                    "enabled": True,
                    "provided": root_snapshot.lp_warm_start is not None,
                    **model.applyLpWarmStart(root_snapshot.lp_warm_start),
                }
            else:
                warm_start_diagnostics = {
                    "enabled": False,
                    "provided": root_snapshot.lp_warm_start is not None,
                    "matched_primal": 0,
                    "matched_dual": 0,
                }
            primal_pool_diagnostics["lp_warm_start"] = warm_start_diagnostics
            fill_reserve = total_limit * float(
                self.zone_config.fill_time_fraction
            )
            if self.zone_config.fix_optimize_policy == "disabled":
                zone_mip_deadline = deadline - fill_reserve
            elif self.zone_config.initial_mip_dynamic_stopping_enabled:
                zone_mip_deadline = perf_counter() + max(
                    0.0,
                    deadline - fill_reserve - perf_counter(),
                ) * float(
                    self.zone_config.initial_mip_max_remaining_fraction
                )
            else:
                zone_mip_deadline = perf_counter() + max(
                    0.0, deadline - perf_counter()
                ) * float(self.zone_config.zone_mip_time_fraction)
            (
                initial_zones,
                initial_export_flow,
                initial_import_reserve,
                zone_mip_initial,
            ) = self._integerize_zone_master(
                model,
                variables,
                zone_mip_deadline,
                start_policy=V5_MULTI_START_POLICY,
            )
            if not initial_zones:
                raise RuntimeError(
                    "restricted zone master did not obtain an integer support: "
                    f"{zone_mip_initial}"
                )
            mandatory_start_indices = set(
                variables.get(
                    "certified_repaired_start_zone_indices",
                    set(),
                )
            )
            primal_pool_indices = set(variables["active_zone_indices"])
            self._finalize_primal_pool_diagnostics(
                primal_pool_diagnostics,
                zone_provenance,
                set(base_primal_pool_indices),
                primal_pool_indices,
                mandatory_start_indices,
            )
            if self.zone_config.fix_optimize_policy == "disabled":
                selected_zones = set(initial_zones)
                selected_export_flow = initial_export_flow
                selected_import_reserve = initial_import_reserve
                fix_optimize = {
                    "status": "disabled",
                    "policy": "disabled",
                    "initial_objective": float(
                        zone_mip_initial["objective"]
                    ),
                    "final_objective": float(
                        zone_mip_initial["objective"]
                    ),
                    "added_zone_count": 0,
                    "improved": False,
                    "absolute_improvement": 0.0,
                    "relative_improvement": 0.0,
                    "rounds": [],
                    "round_count": 0,
                    "successful_round_count": 0,
                    "seconds": 0.0,
                }
            else:
                (
                    selected_zones,
                    selected_export_flow,
                    selected_import_reserve,
                    fix_optimize,
                ) = self._run_objective_fix_optimize(
                    model,
                    variables,
                    set(initial_zones),
                    initial_export_flow,
                    initial_import_reserve,
                    deadline - fill_reserve,
                    zone_provenance,
                )
            final_primal_pool_indices = set(variables["active_zone_indices"])
            primal_pool_diagnostics["final_primal_master_zone_count"] = len(
                final_primal_pool_indices
            )
            final_pool_origins = sorted(
                {
                    origin
                    for zone_index in final_primal_pool_indices
                    for origin in zone_provenance[zone_index]
                }
            )
            primal_pool_diagnostics[
                "final_primal_master_columns_by_origin"
            ] = {
                origin: sum(
                    origin in zone_provenance[zone_index]
                    for zone_index in final_primal_pool_indices
                )
                for origin in final_pool_origins
            }
            zone_mip = zone_mip_initial
        finally:
            self._free_gurobi_model(model)
        selected_support_solver_objective = float(
            fix_optimize.get(
                "final_objective", zone_mip_initial["objective"]
            )
        )
        objective_certificate = self._zone_objective_certificate(
            set(selected_zones),
            selected_export_flow,
            selected_import_reserve,
            selected_support_solver_objective,
        )
        zone_upper_bound = float(objective_certificate["objective"])
        selected_support_source = (
            "initial_restricted_mip"
            if set(selected_zones) == set(initial_zones)
            and selected_export_flow == initial_export_flow
            and selected_import_reserve == initial_import_reserve
            else "post_search_best"
        )
        candidate_indices = {
            index
            for zone_index in selected_zones
            for index in self._zones[zone_index].candidate_indices
        }
        selected, fill = self._solve_restricted_fill(
            set(selected_zones),
            selected_export_flow,
            selected_import_reserve,
            deadline,
        )
        selected_support_zone_objective = zone_upper_bound
        zone_global_lower_bound = float(root_snapshot.objective)
        zone_absolute_gap = max(0.0, zone_upper_bound - zone_global_lower_bound)
        zone_relative_gap = zone_absolute_gap / max(abs(zone_upper_bound), 1e-12)
        selected_zone_origin_counts: Counter[str] = Counter()
        for zone_index in selected_zones:
            origins = zone_provenance.get(zone_index) or {"unclassified"}
            selected_zone_origin_counts.update(origins)
        primal_pool_diagnostics["final_selected_zones_by_origin"] = dict(
            sorted(selected_zone_origin_counts.items())
        )
        fix_optimize_progresses = []
        for round_diagnostics in fix_optimize.get("rounds", []):
            fix_optimize_progresses.extend(
                [
                    round_diagnostics.get("local", {}).get("mip_progress"),
                    round_diagnostics.get("master_reoptimization", {}).get(
                        "mip_progress"
                    ),
                ]
            )
        mip_anytime = self._aggregate_mip_progress(
            [
                zone_mip_initial.get("mip_progress"),
                *fix_optimize_progresses,
            ]
        )
        fix_improvement = float(fix_optimize.get("absolute_improvement", 0.0))
        diagnostics = {
            "algorithm": (
                "exact_proof_cg_diversified_primal_pool_"
                f"{self.zone_config.fix_optimize_policy}_"
                "fix_optimize_with_exact_recourse"
            ),
            "algorithm_version": ALGORITHM_VERSION,
            "model_scope": "actual_quantity_flow_on_dedicated_contiguous_row_zones",
            "formulation": "zone_flow_master_plus_flow_fixed_exact_row_recourse",
            "decomposition": (
                "exact_rmq_proof_master_snapshot_diversified_primal_"
                f"master_{self.zone_config.fix_optimize_policy}_"
                "fix_optimize_then_certified_row_realization"
            ),
            "planned_group_count": len(self.groups),
            "planned_box_count": sum(group.demand for group in self.groups),
            "candidate_row_location_count": len(self._columns),
            "zone_preparation": preparation,
            "zone_root": root,
            "zone_root_proof_only_area_activation_cut_count": (
                proof_only_area_cut_count
            ),
            "zone_root_proof_cuts_retained_in_primal_search": False,
            "root_snapshot": {
                "objective": root_snapshot.objective,
                "proof_zone_count": len(root_snapshot.proof_zone_indices),
                "positive_zone_count": sum(
                    value > 1e-8
                    for value in root_snapshot.zone_values.values()
                ),
                "lp_warm_start_captured": (
                    root_snapshot.lp_warm_start is not None
                ),
                "proof_model_destroyed_before_primal_master": True,
            },
            "zone_pool_enrichment": primal_pool_diagnostics,
            "primal_pool_diagnostics": primal_pool_diagnostics,
            "proof_primal_pool_separated": True,
            "proof_pool_zone_count": len(proof_pool_indices),
            "primal_pool_zone_count": len(primal_pool_indices),
            "final_primal_pool_zone_count": len(final_primal_pool_indices),
            "proof_to_primal_reduction_fraction": (
                primal_pool_diagnostics[
                    "proof_to_primal_reduction_fraction"
                ]
            ),
            "primal_pool_columns_by_origin": primal_pool_diagnostics[
                "primal_pool_columns_by_origin"
            ],
            "zone_mip": zone_mip,
            "zone_mip_initial": zone_mip_initial,
            "zone_fix_optimize": fix_optimize,
            "mip_anytime": mip_anytime,
            "mip_start_diagnostics": zone_mip_initial.get("mip_start", {}),
            "fix_optimize_round_count": int(
                fix_optimize.get("round_count", 0)
            ),
            "fix_optimize_success_count": int(
                fix_optimize.get("successful_round_count", 0)
            ),
            "fix_optimize_total_improvement": fix_improvement,
            "zone_model_upper_bound": zone_upper_bound,
            "zone_model_global_lower_bound": zone_global_lower_bound,
            "zone_model_absolute_gap": zone_absolute_gap,
            "zone_model_relative_gap": zone_relative_gap,
            "zone_model_lower_bound_source": "closed_exact_zone_pricing_root",
            "zone_restricted_pool_bound": zone_mip.get("bound"),
            "zone_restricted_pool_bound_is_global": False,
            "zone_selected_candidate_count": len(candidate_indices),
            "zone_selected_support_source": selected_support_source,
            "zone_selected_origin_counts": dict(
                sorted(selected_zone_origin_counts.items())
            ),
            "zone_selected_support_objective": selected_support_zone_objective,
            "zone_objective_certificate": objective_certificate,
            "zone_candidate_reduction": 1.0
            - len(candidate_indices) / max(1, len(self._columns)),
            "zone_fill": fill,
            "zone_time_policy": {
                "root_remaining_fraction": float(
                    self.zone_config.root_time_fraction
                ),
                "initial_mip_remaining_fraction": float(
                    self.zone_config.zone_mip_time_fraction
                ),
                "initial_mip_dynamic_stopping_enabled": bool(
                    self.zone_config.initial_mip_dynamic_stopping_enabled
                ),
                "initial_mip_max_remaining_fraction": float(
                    self.zone_config.initial_mip_max_remaining_fraction
                ),
                "initial_mip_min_total_fraction": float(
                    self.zone_config.initial_mip_min_total_fraction
                ),
                "initial_mip_stagnation_total_fraction": float(
                    self.zone_config.initial_mip_stagnation_total_fraction
                ),
                "initial_mip_min_relative_improvement": float(
                    self.zone_config.initial_mip_min_relative_improvement
                ),
                "fix_optimize_local_remaining_fraction": float(
                    self.zone_config.fix_optimize_local_fraction
                ),
                "fix_optimize_objective_mass": float(
                    self.zone_config.fix_optimize_objective_mass
                ),
                "fix_optimize_candidate_zone_fraction": float(
                    self.zone_config.fix_optimize_zone_fraction
                ),
                "fix_optimize_group_policy": (
                    self.zone_config.fix_optimize_policy
                ),
                "fix_optimize_max_rounds": int(
                    self.zone_config.fix_optimize_max_rounds
                ),
                "fix_optimize_round_zone_fractions": list(
                    self.zone_config.fix_optimize_round_zone_fractions
                ),
                "final_fill_total_fraction": float(
                    self.zone_config.fill_time_fraction
                ),
                "mip_start_repair_total_fraction": float(
                    self.zone_config.mip_start_repair_total_fraction
                ),
                "mip_start_total_time_fraction": float(
                    self.zone_config.mip_start_total_time_fraction
                ),
                "integer_search_policy": V5_MULTI_START_POLICY,
                "v5_multi_start_enabled": True,
                "proof_primal_pool_policy": (
                    "group_specific_round_robin_diversified_columns"
                ),
                "primal_pool_lp_warm_start_enabled": bool(
                    self.zone_config.primal_pool_apply_lp_warm_start
                ),
            },
            "master_status": fix_optimize["status"],
            "master_objective": zone_upper_bound,
            "master_bound_scope": "complete_redefined_zone_model",
            "row_recourse_status": fill["status"],
            "row_recourse_secondary_quality_objective": fill["objective"],
            "complete_model_lower_bound": zone_global_lower_bound,
            "complete_model_absolute_gap": zone_absolute_gap,
            "complete_model_relative_gap": zone_relative_gap,
            "complete_model_gap_source": (
                "closed_exact_zone_pricing_root; not_an_M0_bound"
            ),
            "restricted_fill_lower_bound": fill["bound"],
            "restricted_fill_absolute_gap": fill["absolute_gap"],
            "restricted_fill_relative_gap": fill["relative_gap"],
            "hard_demand_balance": True,
            "business_objective_normalization": {
                "weights": self._zone_objective_weights(),
                "scales": dict(self._zone_objective_scales),
                "method": "natural_instance_scale",
            },
            "business_objective": (
                self._zone_business_objective_specification()
            ),
            "peak_utilization_policy": dict(
                self._peak_utilization_policy
            ),
            "import_capacity_reservation": {
                "source": "declared_import_documents_excluding_in_yard_boxes",
                "role": "anonymous_size_compatible_capacity_only",
                "area_policy": "endogenous_feasible_reservation_without_area_objective",
                "constraint_scope": [
                    "area_function",
                    "bay_size",
                    "physical_capacity",
                    "peak_utilization_epsilon_cap",
                ],
                "excluded_constraints": [
                    "bay_no_mix",
                    "row_no_mix",
                    "container_group_attributes",
                    "voyage_area_dispersion_objective",
                    "contiguous_zone_objective",
                    "existing_group_proximity_objective",
                    "unused_export_zone_capacity_objective",
                    "berth_distance_objective",
                ],
                "import_boxes": int(
                    sum(self.import_total_by_flow_size.values())
                ),
            },
            "total_seconds": perf_counter() - started,
            "research_stage_gate": True,
            "production_solver_registered": False,
        }
        result = self._assemble_result(selected, diagnostics)
        row_quality_objective = result.diagnostics[
            "final_business_objective"
        ]
        row_quality_components = result.diagnostics[
            "final_business_objective_components"
        ]
        row_quality_components = self._integrated_row_quality_components(
            row_quality_components
        )
        result.diagnostics["row_recourse_secondary_quality_objective"] = (
            row_quality_objective
        )
        result.diagnostics["row_recourse_secondary_quality_components"] = (
            row_quality_components
        )
        result.diagnostics["final_business_objective"] = zone_upper_bound
        result.diagnostics["final_business_objective_components"] = {
            "raw": objective_certificate["raw"],
            "normalized": objective_certificate["normalized"],
            "weighted": objective_certificate["weighted"],
            "weighted_total": zone_upper_bound,
        }
        result.columns = self._selected_direct_columns(selected)
        return result


__all__ = [
    "ContiguousZone",
    "ContiguousZoneConfig",
    "ContiguousZoneGenerationPlanner",
    "RootSnapshot",
]
