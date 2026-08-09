"""Independent contiguous-zone generation and Fix-and-Optimize experiment.

The experiment deliberately changes the concentration representation.  A
hard export group reserves one or more contiguous runs of compatible physical
row footprints and sends its actual declared quantity through those runs.
Selected footprints reserve their complete compatible capacity, which makes
physical conflicts additive and gives a genuine column-generation structure.
The final compact row MILP realizes the certified group-bay flows and the same
anonymous import reservation without changing the primary zone decisions.

Nothing in this module is registered as a production solver.  It is an
isolated stage gate for the redefined model boundary, an exact priced root,
and conflict-guided primal improvement.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from time import perf_counter

from .direct_milp import DirectMilpPlanner
from .gurobi_backend import GurobiModel
from .planner import ColumnGenerationConfig, ColumnGenerationResult

Resource = tuple[str, str]
StripKey = tuple[str, str, str]


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
    """Dimensionless controls for the isolated root-stage experiment."""

    max_root_iterations: int = 60
    reduced_cost_tolerance: float = 1e-8
    columns_per_group_per_round: int = 3
    integer_pool_columns_per_group: int = 100
    root_time_fraction: float = 0.40
    zone_mip_time_fraction: float = 0.75
    branch_price_time_fraction: float = 0.70
    branch_probe_time_fraction: float = 0.25
    branch_min_gap_closure: float = 0.01
    fix_optimize_local_fraction: float = 0.85
    fix_optimize_group_count: int = 12
    fill_time_fraction: float = 0.05
    max_branch_nodes: int = 200
    shortage_penalty: float = 1_000.0
    # Unified zone-model business weights.  They are deliberately independent
    # of the legacy compact-row objective: the zone model is the optimization
    # model, while row filling is an exact feasibility recourse.
    area_dispersion_weight: float = 0.25
    zone_dispersion_weight: float = 0.22
    existing_group_proximity_weight: float = 0.08
    area_guidance_weight: float = 0.22
    unused_capacity_weight: float = 0.13
    berth_distance_weight: float = 0.10

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
        if not 0.0 < float(self.branch_price_time_fraction) < 1.0:
            raise ValueError(
                "branch_price_time_fraction must lie strictly between 0 and 1"
            )
        if not 0.0 < float(self.branch_probe_time_fraction) < 1.0:
            raise ValueError(
                "branch_probe_time_fraction must lie strictly between 0 and 1"
            )
        if not 0.0 <= float(self.branch_min_gap_closure) < 1.0:
            raise ValueError("branch_min_gap_closure must lie in [0, 1)")
        if not 0.0 < float(self.fix_optimize_local_fraction) < 1.0:
            raise ValueError(
                "fix_optimize_local_fraction must lie strictly between 0 and 1"
            )
        if int(self.fix_optimize_group_count) <= 0:
            raise ValueError("fix_optimize_group_count must be positive")
        if not 0.0 < float(self.fill_time_fraction) < 1.0:
            raise ValueError("fill_time_fraction must lie strictly between 0 and 1")
        if int(self.max_branch_nodes) <= 0:
            raise ValueError("max_branch_nodes must be positive")
        if float(self.shortage_penalty) <= 0.0:
            raise ValueError("shortage_penalty must be positive")
        objective_weights = {
            "area_dispersion": self.area_dispersion_weight,
            "zone_dispersion": self.zone_dispersion_weight,
            "existing_group_proximity": self.existing_group_proximity_weight,
            "area_guidance": self.area_guidance_weight,
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
    quota_key: tuple[str, str, str, str]
    objective_cost: float


@dataclass(frozen=True)
class ZoneBranchDecision:
    category: str
    key: object
    sense: str
    bound: float


@dataclass(frozen=True)
class ZoneBranchNode:
    node_id: int
    depth: int
    lower_bound_estimate: float
    decisions: tuple[ZoneBranchDecision, ...]
    lp_warm_start: dict[str, dict[str, float]] | None = None


class ContiguousZoneGenerationPlanner(DirectMilpPlanner):
    """Generate dedicated contiguous zones, then perform exact row filling."""

    def __init__(
        self,
        problem,
        config: ColumnGenerationConfig | None = None,
        zone_config: ContiguousZoneConfig | None = None,
    ) -> None:
        super().__init__(problem, config)
        self.zone_config = zone_config or ContiguousZoneConfig()
        self.zone_config.validate()
        self._zones: list[ContiguousZone] = []
        self._zone_indices_by_group: defaultdict[str, list[int]] = defaultdict(list)
        self._zone_indices_by_strip: defaultdict[StripKey, list[int]] = defaultdict(list)
        self._zone_id_by_signature: dict[tuple[int, ...], int] = {}
        self._strip_runs: dict[StripKey, tuple[tuple[int, ...], ...]] = {}
        self._possible_zone_count = 0
        self._possible_zone_count_by_group: Counter[str] = Counter()
        self._atomic_capacity: list[int] = []
        self._atomic_attr_keys: list[tuple[tuple[str, str, str, str], ...]] = []
        self._atomic_resources: list[tuple[Resource, ...]] = []
        self._zone_objective_scales: dict[str, float] = {}

    def _zone_objective_weights(self) -> dict[str, float]:
        return {
            "area_dispersion": float(
                self.zone_config.area_dispersion_weight
            ),
            "zone_dispersion": float(
                self.zone_config.zone_dispersion_weight
            ),
            "existing_group_proximity": float(
                self.zone_config.existing_group_proximity_weight
            ),
            "area_guidance": float(
                self.zone_config.area_guidance_weight
            ),
            "unused_capacity": float(
                self.zone_config.unused_capacity_weight
            ),
            "berth_distance": float(
                self.zone_config.berth_distance_weight
            ),
        }

    def _zone_objective_scale(self, key: str) -> float:
        value = self._zone_objective_scales.get(key)
        if value is None:
            raise RuntimeError(
                f"zone objective normalization is not prepared: {key}"
            )
        return max(1.0, float(value))

    def _zone_area_activation_penalty(self) -> float:
        return (
            float(self.zone_config.area_dispersion_weight)
            / self._zone_objective_scale("area_dispersion")
        )

    def _zone_activation_penalty(self) -> float:
        return (
            float(self.zone_config.zone_dispersion_weight)
            / self._zone_objective_scale("zone_dispersion")
        )

    def _zone_guidance_penalty(self) -> float:
        return (
            float(self.zone_config.area_guidance_weight)
            / self._zone_objective_scale("area_guidance")
        )

    def _zone_existing_proximity_unit_cost(
        self, group_id: str, bay_key: str
    ) -> float:
        group = self.groups_by_id[group_id]
        return (
            float(self.zone_config.existing_group_proximity_weight)
            * self._normalized_existing_proximity(group, bay_key)
            / self._zone_objective_scale("existing_group_proximity")
        )

    def _zone_berth_distance_unit_cost(
        self, group_id: str, area_no: str
    ) -> float:
        group = self.groups_by_id[group_id]
        return (
            float(self.zone_config.berth_distance_weight)
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
            "decision_level": "group_contiguous_zone_and_area_flow",
            "hard_demand_balance": True,
            "weights": self._zone_objective_weights(),
            "scales": dict(self._zone_objective_scales),
            "row_recourse_role": (
                "exact_feasibility_and_secondary_quality_only"
            ),
        }

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
        quota_keys = set()
        for index in candidate_indices:
            column = self._columns[index]
            row_capacity = int(self._base_location_capacity(group, column))
            if row_capacity <= 0:
                raise RuntimeError(f"zone contains a zero-capacity candidate: {index}")
            capacity += row_capacity
            anchor_bay_loads[str(column.bay_key)] += row_capacity
            quota_keys.add(column.quota_key)
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
        if len(quota_keys) != 1:
            raise RuntimeError(
                f"one zone spans multiple large-plan keys: {sorted(quota_keys)}"
            )
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
            quota_key=next(iter(quota_keys)),
            objective_cost=float(
                zone_penalty + self._unused_capacity_unit_cost() * capacity
            ),
        )

    def _unused_capacity_unit_cost(self) -> float:
        return (
            float(self.zone_config.unused_capacity_weight)
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
            demand = int(self.groups_by_id[strip_key[0]].demand)
            maximum_row_capacity = max(
                (
                    int(
                        self._base_location_capacity(
                            self.groups_by_id[strip_key[0]], self._columns[index]
                        )
                    )
                    for run in runs
                    for index in run
                ),
                default=0,
            )
            capacity_limit = demand + maximum_row_capacity
            for run in runs:
                for start in range(len(run)):
                    running_capacity = 0
                    for end in range(start, len(run)):
                        index = run[end]
                        running_capacity += int(
                            self._base_location_capacity(
                                self.groups_by_id[strip_key[0]],
                                self._columns[index],
                            )
                        )
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

        self._zones = []
        self._zone_id_by_signature.clear()
        self._zone_indices_by_group.clear()
        self._zone_indices_by_strip.clear()
        self._strip_runs.clear()
        dominated_long_zone_count = 0
        possible_zone_count = 0
        possible_zone_count_by_group: Counter[str] = Counter()
        for strip_key, indices in sorted(strip_candidates.items()):
            group_id, area_no, row_no = strip_key
            demand = int(self.groups_by_id[group_id].demand)
            capacities = [
                int(
                    self._base_location_capacity(
                        self.groups_by_id[group_id], self._columns[index]
                    )
                )
                for index in indices
            ]
            maximum_row_capacity = max(capacities, default=0)
            runs = self._split_contiguous_runs(indices)
            self._strip_runs[strip_key] = runs
            for run in runs:
                for start in range(len(run)):
                    running_capacity = 0
                    for end in range(start, len(run)):
                        index = run[end]
                        running_capacity += int(
                            self._base_location_capacity(
                                self.groups_by_id[group_id], self._columns[index]
                            )
                        )
                        if running_capacity > demand + maximum_row_capacity:
                            dominated_long_zone_count += len(run) - end
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
        total_export_demand = sum(int(group.demand) for group in self.groups)
        self._zone_objective_scales = {
            "area_dispersion": self._objective_scale("area_dispersion"),
            "zone_dispersion": float(max(1, natural_zone_expansion)),
            "existing_group_proximity": self._objective_scale(
                "existing_group_proximity"
            ),
            "area_guidance": self._objective_scale("area_guidance_l1"),
            "unused_capacity": float(max(1, total_export_demand)),
            "berth_distance": self._objective_scale("berth_distance"),
        }
        return {
            "preparation_seconds": perf_counter() - started,
            "atomic_candidate_count": len(self._columns),
            "strip_count": len(strip_candidates),
            "zone_count": self._possible_zone_count,
            "materialized_zone_count": len(self._zones),
            "dominated_long_zone_count": dominated_long_zone_count,
            "objective_scales": dict(self._zone_objective_scales),
        }

    def _master_index_sets(self) -> dict[str, object]:
        physical_resources: set[Resource] = set()
        stack_keys: set[tuple[str, str]] = set()
        attr_keys: set[tuple[str, str, str, str]] = set()
        area_pairs: set[tuple[str, str]] = set()
        flow_columns: dict[tuple[str, str], object] = {}
        for column in self._columns:
            area_key = (str(column.group_id), str(column.area_no))
            area_pairs.add(area_key)
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
        # Every positive-demand group necessarily uses at least one area and
        # one zone.  Subtract those unavoidable activations so the reported
        # objective measures *extra* dispersion instead of a constant base.
        dispersion_baseline = len(self.groups) * (
            self._zone_area_activation_penalty()
            + self._zone_activation_penalty()
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
                obj=self._zone_area_activation_penalty(),
                name=f"zone_area_{self._key_name(key)}",
            )
            for key in sets["area_pairs"]
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
        import_by_flow_area_size: defaultdict[tuple[str, str, str], list] = defaultdict(list)
        for (flow, size, bay_key), variable in import_reserve.items():
            area_no = self.bays[bay_key].area_no
            for footprint_key in self._placement_footprint_keys(bay_key, size):
                import_by_bay[footprint_key].append(variable)
            import_by_bay_size[(bay_key, size)].append(variable)
            import_by_flow_size[(flow, size)].append(variable)
            import_by_flow_area_size[(flow, area_no, size)].append(variable)

        flow_by_group: defaultdict[str, list] = defaultdict(list)
        flow_by_area: defaultdict[tuple[str, str], list] = defaultdict(list)
        flow_by_guidance: defaultdict[tuple[str, str, str, str], list] = (
            defaultdict(list)
        )
        for key, variable in export_flow.items():
            group_id, bay_key = key
            column = sets["flow_columns"][key]
            flow_by_group[group_id].append(variable)
            flow_by_area[(group_id, column.area_no)].append(variable)
            flow_by_guidance[column.quota_key].append(variable)

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
            constraints["area_flow_cover"][key] = model.addConstr(
                assigned <= zero,
                name=f"zone_area_cover_{self._key_name(key)}",
            )
        for key in sorted(self._master_area_guidance_keys):
            target = self._area_size_target(*key)
            positive = model.addVar(
                lb=0.0,
                obj=self._zone_guidance_penalty(),
                name=f"zone_guide_pos_{self._key_name(key)}",
            )
            negative = model.addVar(
                lb=0.0,
                obj=self._zone_guidance_penalty(),
                name=f"zone_guide_neg_{self._key_name(key)}",
            )
            constraints["export_guidance"][key] = model.addConstr(
                quicksum(flow_by_guidance.get(key, [])) - float(target)
                == positive - negative,
                name=f"zone_guide_{self._key_name(key)}",
            )
        for key, required in sorted(self.import_total_by_flow_size.items()):
            constraints["import_total"][key] = model.addConstr(
                quicksum(import_by_flow_size.get(key, [])) == int(required),
                name=f"zone_import_total_{self._key_name(key)}",
            )
        constraints["import_guidance"] = (
            self._add_zone_import_reference_deviation(
                quicksum,
                model,
                import_by_flow_area_size,
            )
        )
        model.update()
        return model, {
            "baseline": baseline,
            "shortage": shortage,
            "export_flow": export_flow,
            "area_use": area_use,
            "attr_state": attr_state,
            "import_reserve": import_reserve,
            "constraints": constraints,
            "zone": {},
            "active_zone_indices": set(),
        }

    def _add_zone_import_reference_deviation(
        self,
        quicksum,
        model,
        import_by_flow_area_size: dict[tuple[str, str, str], list],
    ) -> dict[tuple[str, str, str], object]:
        """Use the unified zone-model guidance coefficient for imports."""

        keys = set(self.import_area_size_reference) | set(
            import_by_flow_area_size
        )
        balances = {}
        for flow, area_no, size in sorted(keys):
            target = int(
                self.import_area_size_reference.get(
                    (flow, area_no, size), 0
                )
            )
            positive = model.addVar(
                lb=0.0,
                obj=self._zone_guidance_penalty(),
                name=f"zone_import_guide_pos_{flow}_{area_no}_{size}",
            )
            negative = model.addVar(
                lb=0.0,
                obj=self._zone_guidance_penalty(),
                name=f"zone_import_guide_neg_{flow}_{area_no}_{size}",
            )
            actual = quicksum(
                import_by_flow_area_size.get((flow, area_no, size), [])
            )
            balances[(flow, area_no, size)] = model.addConstr(
                actual - target == positive - negative,
                name=f"zone_import_guide_{flow}_{area_no}_{size}",
            )
        return balances

    def _zone_coefficients(
        self, zone: ContiguousZone
    ) -> tuple[tuple[str, object, float], ...]:
        coefficients: list[tuple[str, object, float]] = []
        area_key = (zone.group_id, zone.area_no)
        coefficients.append(("area_zone_support", area_key, -1.0))
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
            maximum_row_capacity = max(
                (atomic_capacity[index] for run in runs for index in run),
                default=0,
            )
            capacity_limit = demand + maximum_row_capacity
            fixed_cost = self._zone_activation_penalty()
            fixed_cost += float(
                duals.get(
                    ("area_zone_support", (group_id, area_no)),
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
        }

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
        """Construct a whole-zone support while materializing only chosen runs."""

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

        group_ids = [group.group_id for group in self.groups]
        orderings = [
            sorted(
                group_ids,
                key=lambda key: self._possible_zone_count_by_group[key],
            ),
            sorted(group_ids, key=lambda key: -int(self.groups_by_id[key].demand)),
            sorted(group_ids),
        ]

        def attempt(
            group_order: list[str],
            import_protection: float,
        ) -> tuple[list[tuple[StripKey, tuple[int, ...]]], dict[str, int]]:
            chosen: list[tuple[StripKey, tuple[int, ...]]] = []
            chosen_signatures: set[tuple[int, ...]] = set()
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

            for zone_index in sorted(
                root_values,
                key=lambda index: -float(root_values.get(index, 0.0)),
            ):
                if root_values.get(zone_index, 0.0) <= 1e-8:
                    break
                zone = self._zones[zone_index]
                demand = int(self.groups_by_id[zone.group_id].demand)
                if covered[zone.group_id] < demand and feasible(zone):
                    strip_key = (
                        zone.group_id,
                        zone.area_no,
                        zone.row_no,
                    )
                    add(strip_key, zone)

            for group_id in group_order:
                demand = int(self.groups_by_id[group_id].demand)
                while covered[group_id] < demand:
                    remaining = demand - covered[group_id]
                    best = None
                    best_key = None
                    for strip_key, signature in self._iter_zone_signatures(group_id):
                        if signature in chosen_signatures:
                            continue
                        zone_index = self._zone_id_by_signature.get(signature)
                        zone = (
                            self._zones[zone_index]
                            if zone_index is not None
                            else self._make_zone(
                                strip_key[0],
                                strip_key[1],
                                strip_key[2],
                                signature,
                            )
                        )
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
                        return chosen, dict(covered)
                    add(*best)
            return chosen, dict(covered)

        best_chosen: list[tuple[StripKey, tuple[int, ...]]] = []
        best_covered: dict[str, int] = {}
        best_protection = None
        for protection in (1.0, 0.0):
            for ordering in orderings:
                chosen, covered = attempt(ordering, protection)
                covered_boxes = sum(
                    min(int(self.groups_by_id[key].demand), covered.get(key, 0))
                    for key in group_ids
                )
                best_boxes = sum(
                    min(
                        int(self.groups_by_id[key].demand),
                        best_covered.get(key, 0),
                    )
                    for key in group_ids
                )
                if covered_boxes > best_boxes:
                    best_chosen = chosen
                    best_covered = covered
                    best_protection = protection
                if covered_boxes == sum(
                    int(self.groups_by_id[key].demand) for key in group_ids
                ):
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
        used_attr_states = {
            key
            for zone_index in selected_zone_indices
            for key, value in self._zones[zone_index].bay_attr_uses
            if value > 0
        }
        for key, variable in variables["attr_state"].items():
            variable.Start = 1.0 if key in used_attr_states else 0.0
        import_start = self._import_reservation_mip_start(variables)
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
            "used_attribute_state_start_count": len(used_attr_states),
        }

    def _greedy_zone_mip_start(self, model, variables: dict) -> dict[str, object]:
        """Build a deterministic, LP-guided integer support for Gurobi repair.

        The start protects the root LP's anonymous import allocation while it
        packs whole export zones.  Gurobi is allowed to repair the deliberately
        partial start (absolute-deviation helper variables are left for
        Gurobi to complete).
        """

        chosen, covered, protection = self._on_demand_greedy_support(variables)
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
                model, variables, deadline
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
            selected, _export_flow, _import_reserve, stats = self._integerize_zone_master(
                model, variables, deadline
            )
        finally:
            self._free_gurobi_model(model)
        return {
            "algorithm": "complete_contiguous_zone_mip",
            "model_scope": "dedicated_contiguous_row_zone_support",
            **preparation,
            **stats,
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

    def _integerize_zone_master(
        self,
        model,
        variables: dict,
        deadline: float,
    ) -> tuple[
        set[int],
        dict[tuple[str, str], int],
        dict[tuple[str, str, str], int],
        dict[str, object],
    ]:
        mip_start = self._greedy_zone_mip_start(model, variables)
        for variable in variables["zone"].values():
            variable.VType = "B"
        for variable in variables["area_use"].values():
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
        remaining = deadline - perf_counter()
        if remaining <= 1e-6:
            return set(), {}, {}, {"status": "time_limit_before_zone_mip"}
        self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
        self._set_gurobi_param(model, "MIPGap", 0.0)
        self._set_gurobi_param(model, "MIPFocus", 1)
        self._set_gurobi_param(model, "Heuristics", 0.20)
        model.optimize()
        status = self._gurobi_status_name(model)
        if self._gurobi_solution_count(model) <= 0:
            return set(), {}, {}, {
                "status": status,
                "has_solution": False,
                "mip_start": mip_start,
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
            "mip_start": mip_start,
        }

    def _resume_integerized_zone_master(
        self,
        model,
        variables: dict,
        deadline: float,
    ) -> tuple[
        set[int],
        dict[tuple[str, str], int],
        dict[tuple[str, str, str], int],
        dict[str, object],
    ]:
        """Continue the existing restricted-MIP tree without modifying it."""

        remaining = deadline - perf_counter()
        if remaining <= 1e-6:
            return set(), {}, {}, {
                "status": "time_limit_before_zone_mip_resume"
            }
        self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
        model.optimize()
        status = self._gurobi_status_name(model)
        if self._gurobi_solution_count(model) <= 0:
            return set(), {}, {}, {
                "status": status,
                "has_solution": False,
                "persistent_search_continued": True,
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
            "persistent_search_continued": True,
        }

    def _conflict_fix_optimize_neighborhood(
        self,
        selected_zone_indices: set[int],
        export_flow: dict[tuple[str, str], int],
    ) -> set[str]:
        """Choose high-contribution groups and their physical competitors."""

        zones_by_group: defaultdict[str, list[int]] = defaultdict(list)
        for zone_index in selected_zone_indices:
            zones_by_group[self._zones[zone_index].group_id].append(zone_index)
        used_areas: defaultdict[str, set[str]] = defaultdict(set)
        flow_cost: Counter[str] = Counter()
        export_by_quota: Counter[tuple[str, str, str, str]] = Counter()
        flow_by_group_quota: Counter[
            tuple[str, tuple[str, str, str, str]]
        ] = Counter()
        for (group_id, bay_key), quantity in export_flow.items():
            if int(quantity) <= 0:
                continue
            used_areas[group_id].add(self.bays[bay_key].area_no)
            flow_cost[group_id] += (
                self._zone_flow_unit_cost(group_id, bay_key)
                - self._unused_capacity_unit_cost()
            ) * int(quantity)
            quota_key = self._quota_key(
                self.groups_by_id[group_id], self.bays[bay_key].area_no
            )
            export_by_quota[quota_key] += int(quantity)
            flow_by_group_quota[(group_id, quota_key)] += int(quantity)
        guidance_cost: Counter[str] = Counter()
        for quota_key in self._master_area_guidance_keys:
            actual = float(export_by_quota.get(quota_key, 0))
            deviation = abs(actual - float(self._area_size_target(*quota_key)))
            if deviation <= 1e-12:
                continue
            positive_groups = [
                group.group_id
                for group in self.groups
                if flow_by_group_quota.get((group.group_id, quota_key), 0) > 0
            ]
            if positive_groups:
                denominator = max(1.0, actual)
                for group_id in positive_groups:
                    guidance_cost[group_id] += (
                        self._zone_guidance_penalty()
                        * deviation
                        * flow_by_group_quota[(group_id, quota_key)]
                        / denominator
                    )
        score = {}
        for group in self.groups:
            group_id = group.group_id
            zone_cost = sum(
                float(self._zones[index].objective_cost)
                for index in zones_by_group[group_id]
            ) - self._zone_activation_penalty()
            area_cost = self._zone_area_activation_penalty() * max(
                0, len(used_areas[group_id]) - 1
            )
            contribution = max(
                0.0,
                zone_cost
                + flow_cost[group_id]
                + area_cost
                + guidance_cost[group_id],
            )
            density = contribution / max(1, int(group.demand))
            score[group_id] = contribution + density
        count = min(
            len(score),
            int(self.zone_config.fix_optimize_group_count),
        )
        selected_resources: defaultdict[str, set[Resource]] = defaultdict(set)
        for zone_index in selected_zone_indices:
            zone = self._zones[zone_index]
            selected_resources[zone.group_id].update(zone.resources)
        candidate_resources: defaultdict[str, set[Resource]] = defaultdict(set)
        candidate_areas: defaultdict[str, set[str]] = defaultdict(set)
        for index, column in enumerate(self._columns):
            candidate_resources[column.group_id].update(
                self._atomic_resources[index]
            )
            candidate_areas[column.group_id].add(column.area_no)

        high_contribution_count = min(count, max(1, count // 2))
        neighborhood = sorted(
            score,
            key=lambda group_id: (-score[group_id], group_id),
        )[:high_contribution_count]
        remaining_groups = set(score) - set(neighborhood)
        score_scale = max(max(score.values(), default=0.0), 1e-12)
        while remaining_groups and len(neighborhood) < count:
            best_group = max(
                remaining_groups,
                key=lambda group_id: (
                    sum(
                        len(
                            selected_resources[chosen]
                            & candidate_resources[group_id]
                        )
                        + len(
                            selected_resources[group_id]
                            & candidate_resources[chosen]
                        )
                        for chosen in neighborhood
                    ),
                    sum(
                        int(
                            bool(
                                candidate_areas[group_id]
                                & candidate_areas[chosen]
                            )
                        )
                        for chosen in neighborhood
                    ),
                    score[group_id] / score_scale,
                    group_id,
                ),
            )
            neighborhood.append(best_group)
            remaining_groups.remove(best_group)
        return set(neighborhood)

    def _solve_conflict_fix_optimize_subproblem(
        self,
        selected_zone_indices: set[int],
        export_flow: dict[tuple[str, str], int],
        import_reserve: dict[tuple[str, str, str], int],
        deadline: float,
    ) -> tuple[
        set[int],
        dict[tuple[str, str], int],
        dict[tuple[str, str, str], int],
        dict[str, object],
    ]:
        """Jointly reopen complete zones for one conflict neighborhood."""

        remaining = deadline - perf_counter()
        if remaining <= 1e-6:
            return set(), {}, {}, {"status": "time_limit_before_fix_optimize"}
        neighborhood = self._conflict_fix_optimize_neighborhood(
            selected_zone_indices, export_flow
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
            self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
            self._set_gurobi_param(model, "MIPGap", 0.0)
            self._set_gurobi_param(model, "MIPFocus", 1)
            self._set_gurobi_param(model, "Heuristics", 0.30)
            model.optimize()
            status = self._gurobi_status_name(model)
            if self._gurobi_solution_count(model) <= 0:
                return set(), {}, {}, {
                    "status": status,
                    "has_solution": False,
                    "neighborhood": sorted(neighborhood),
                    "neighborhood_group_count": len(neighborhood),
                    "active_zone_count": len(active_zone_indices),
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
                "active_zone_count": len(active_zone_indices),
                "materialized_zone_count": len(self._zones) - materialized_before,
                "selected_zone_count": len(selected),
                "positive_export_flow_count": len(flow),
                "positive_import_reservation_count": len(imports),
            }
        finally:
            self._free_gurobi_model(model)

    def _run_conflict_fix_optimize(
        self,
        model,
        variables: dict,
        incumbent_zones: set[int],
        incumbent_export_flow: dict[tuple[str, str], int],
        incumbent_import_reserve: dict[tuple[str, str, str], int],
        deadline: float,
    ) -> tuple[
        set[int],
        dict[tuple[str, str], int],
        dict[tuple[str, str, str], int],
        dict[str, object],
    ]:
        """Feed an exact conflict-neighborhood solution back into the zone RMP."""

        started = perf_counter()
        incumbent_certificate = self._zone_objective_certificate(
            incumbent_zones,
            incumbent_export_flow,
            incumbent_import_reserve,
            self._gurobi_objective_value(model),
        )
        incumbent_objective = float(incumbent_certificate["objective"])
        local_deadline = perf_counter() + max(
            0.0, deadline - perf_counter()
        ) * float(self.zone_config.fix_optimize_local_fraction)
        local_zones, local_flow, local_import, local = (
            self._solve_conflict_fix_optimize_subproblem(
                incumbent_zones,
                incumbent_export_flow,
                incumbent_import_reserve,
                local_deadline,
            )
        )
        if not local_zones:
            return (
                incumbent_zones,
                incumbent_export_flow,
                incumbent_import_reserve,
                {
                    "status": "local_subproblem_has_no_solution",
                    "initial_objective": incumbent_objective,
                    "local": local,
                    "added_zone_count": 0,
                    "improved": False,
                    "seconds": perf_counter() - started,
                },
            )
        added = 0
        for zone_index in sorted(local_zones):
            if zone_index not in variables["active_zone_indices"]:
                variable = self._add_zone_variable(model, variables, zone_index)
                variable.VType = "B"
                added += 1
        use_local_start = float(local["objective"]) <= incumbent_objective + 1e-9
        start_zones = local_zones if use_local_start else incumbent_zones
        start_flow = local_flow if use_local_start else incumbent_export_flow
        start_import = (
            local_import if use_local_start else incumbent_import_reserve
        )
        for zone_index, variable in variables["zone"].items():
            variable.Start = 1.0 if zone_index in start_zones else 0.0
        for key, variable in variables["export_flow"].items():
            variable.Start = float(start_flow.get(key, 0))
        for key, variable in variables["import_reserve"].items():
            variable.Start = float(start_import.get(key, 0))
        used_areas = {
            (group_id, self.bays[bay_key].area_no)
            for (group_id, bay_key), quantity in start_flow.items()
            if int(quantity) > 0
        }
        for key, variable in variables["area_use"].items():
            variable.Start = 1.0 if key in used_areas else 0.0
        used_attributes = {
            key
            for zone_index in start_zones
            for key, value in self._zones[zone_index].bay_attr_uses
            if value > 0
        }
        for key, variable in variables["attr_state"].items():
            variable.Start = 1.0 if key in used_attributes else 0.0
        model.update()
        remaining = deadline - perf_counter()
        if remaining <= 1e-6:
            local_objective = float(local["objective"])
            candidate_zones = local_zones if use_local_start else incumbent_zones
            candidate_flow = local_flow if use_local_start else incumbent_export_flow
            candidate_import = (
                local_import if use_local_start else incumbent_import_reserve
            )
            return candidate_zones, candidate_flow, candidate_import, {
                "status": "time_limit_before_master_reoptimization",
                "initial_objective": incumbent_objective,
                "local": local,
                "added_zone_count": added,
                "improved": use_local_start
                and local_objective < incumbent_objective - 1e-9,
                "absolute_improvement": max(
                    0.0, incumbent_objective - local_objective
                )
                if use_local_start
                else 0.0,
                "relative_improvement": max(
                    0.0, incumbent_objective - local_objective
                )
                / max(abs(incumbent_objective), 1e-12)
                if use_local_start
                else 0.0,
                "final_objective": (
                    local_objective
                    if use_local_start
                    else incumbent_objective
                ),
                "seconds": perf_counter() - started,
            }
        self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
        model.optimize()
        status = self._gurobi_status_name(model)
        if self._gurobi_solution_count(model) <= 0:
            local_improves = (
                float(local["objective"]) < incumbent_objective - 1e-9
            )
            return (
                local_zones if local_improves else incumbent_zones,
                local_flow if local_improves else incumbent_export_flow,
                local_import if local_improves else incumbent_import_reserve,
                {
                    "status": status,
                    "initial_objective": incumbent_objective,
                    "local": local,
                    "added_zone_count": added,
                    "improved": local_improves,
                    "final_objective": (
                        float(local["objective"])
                        if local_improves
                        else incumbent_objective
                    ),
                    "seconds": perf_counter() - started,
                },
            )
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
        candidate_certificate = self._zone_objective_certificate(
            candidate_zones,
            candidate_flow,
            candidate_import,
            self._gurobi_objective_value(model),
        )
        candidate_objective = float(candidate_certificate["objective"])
        best_zones = incumbent_zones
        best_flow = incumbent_export_flow
        best_import = incumbent_import_reserve
        best_objective = incumbent_objective
        if float(local["objective"]) < best_objective - 1e-9:
            best_zones = local_zones
            best_flow = local_flow
            best_import = local_import
            best_objective = float(local["objective"])
        if candidate_objective < best_objective - 1e-9:
            best_zones = candidate_zones
            best_flow = candidate_flow
            best_import = candidate_import
            best_objective = candidate_objective
        return best_zones, best_flow, best_import, {
            "status": status,
            "initial_objective": incumbent_objective,
            "local": local,
            "added_zone_count": added,
            "master_reoptimization": {
                "status": status,
                "objective": self._gurobi_objective_value(model),
            },
            "improved": best_objective < incumbent_objective - 1e-9,
            "absolute_improvement": max(
                0.0, incumbent_objective - best_objective
            ),
            "relative_improvement": max(
                0.0, incumbent_objective - best_objective
            )
            / max(abs(incumbent_objective), 1e-12),
            "final_objective": best_objective,
            "seconds": perf_counter() - started,
        }

    def _root_branch_seed(self, variables: dict) -> dict[str, object] | None:
        """Reuse the certified root solution instead of resolving it in B&P."""

        values_by_category = {
            "area_use": variables.get("last_root_area_values", {}),
            "attr_state": variables.get("last_root_attr_values", {}),
            "zone": variables.get("last_root_zone_values", {}),
            "export_flow": variables.get("last_root_export_flow_values", {}),
            "import_reserve": variables.get("last_root_import_values", {}),
        }
        for category in (
            "area_use",
            "attr_state",
            "zone",
            "export_flow",
            "import_reserve",
        ):
            fractional = []
            for raw_key, raw_value in values_by_category[category].items():
                value = float(raw_value)
                distance = abs(value - round(value))
                if distance <= 1e-7:
                    continue
                key = (
                    self._zones[int(raw_key)].candidate_indices
                    if category == "zone"
                    else raw_key
                )
                fractional.append((distance, key, value))
            if fractional:
                _distance, key, value = max(
                    fractional,
                    key=lambda item: (item[0], str(item[1])),
                )
                return {
                    "category": category,
                    "key": key,
                    "value": value,
                    "fractional_count": len(fractional),
                }
        return None

    @staticmethod
    def _branch_variable(variables: dict, decision: ZoneBranchDecision):
        if decision.category == "zone":
            return variables["zone"][decision.key]
        return variables[decision.category][decision.key]

    def _build_branch_price_node(self, node: ZoneBranchNode):
        model, variables = self._build_zone_master()
        forbidden_signatures = {
            tuple(decision.key)
            for decision in node.decisions
            if decision.category == "zone"
            and decision.sense == "le"
            and decision.bound < 0.5
        }
        for zone in self._zones:
            if zone.candidate_indices in forbidden_signatures:
                continue
            self._add_zone_variable(model, variables, zone.zone_id)
        for variable in variables["shortage"].values():
            variable.UB = 0.0
        artificial = []
        penalty = float(self.zone_config.shortage_penalty)
        for position, decision in enumerate(node.decisions):
            if decision.category == "zone":
                signature = tuple(decision.key)
                if decision.sense == "le" and decision.bound < 0.5:
                    # The pricing oracle and the node RMP both omit this exact
                    # zone signature, so no explicit row is required.
                    continue
                zone_index = self._zone_id_by_signature[signature]
                decision = ZoneBranchDecision(
                    category="zone",
                    key=zone_index,
                    sense=decision.sense,
                    bound=decision.bound,
                )
            variable = self._branch_variable(variables, decision)
            slack = model.addVar(
                lb=0.0,
                obj=penalty,
                name=f"bp_artificial_{node.node_id}_{position}",
            )
            artificial.append(slack)
            if decision.sense == "le":
                model.addConstr(
                    variable <= float(decision.bound) + slack,
                    name=f"bp_upper_{node.node_id}_{position}",
                )
            else:
                model.addConstr(
                    variable + slack >= float(decision.bound),
                    name=f"bp_lower_{node.node_id}_{position}",
                )
        model.update()
        warm_start_stats = model.applyLpWarmStart(node.lp_warm_start)
        return (
            model,
            variables,
            artificial,
            forbidden_signatures,
            warm_start_stats,
        )

    def _solve_branch_price_node(
        self,
        node: ZoneBranchNode,
        deadline: float,
    ) -> dict[str, object]:
        (
            model,
            variables,
            artificial,
            forbidden_signatures,
            warm_start_stats,
        ) = self._build_branch_price_node(node)
        rounds = 0
        minimum_reduced_cost = -math.inf
        try:
            for rounds in range(1, int(self.zone_config.max_root_iterations) + 1):
                remaining = deadline - perf_counter()
                if remaining <= 1e-6:
                    return {
                        "closed": False,
                        "status": "time_limit_before_node_lp",
                        "rounds": rounds - 1,
                    }
                self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
                model.optimize()
                status = self._gurobi_status_name(model)
                if status != "optimal":
                    return {
                        "closed": False,
                        "status": status,
                        "rounds": rounds,
                    }
                duals = self._dual_snapshot(model, variables["constraints"])
                pricing = self._price_zone_signatures(
                    duals,
                    per_group_limit=int(
                        self.zone_config.columns_per_group_per_round
                    ),
                    active_zone_indices=variables["active_zone_indices"],
                    improving_only=True,
                    forbidden_signatures=forbidden_signatures,
                )
                minimum_reduced_cost = float(pricing["minimum_reduced_cost"])
                if not pricing["selected"]:
                    break
                self._add_priced_signatures(
                    model,
                    variables,
                    pricing["selected"],
                )
            else:
                return {
                    "closed": False,
                    "status": "iteration_limit",
                    "rounds": rounds,
                }

            artificial_value = sum(
                self._gurobi_value(model, variable) for variable in artificial
            )
            if artificial_value > 1e-7:
                return {
                    "closed": True,
                    "status": "infeasible",
                    "artificial_value": artificial_value,
                    "rounds": rounds,
                    "minimum_reduced_cost": minimum_reduced_cost,
                    "lp_warm_start": warm_start_stats,
                }
            objective = self._gurobi_objective_value(model)
            fractional_by_kind: defaultdict[str, list[tuple]] = defaultdict(list)
            for key, variable in variables["area_use"].items():
                value = self._gurobi_value(model, variable)
                distance = abs(value - round(value))
                if distance > 1e-7:
                    fractional_by_kind["area_use"].append(
                        (distance, "area_use", key, value)
                    )
            for key, variable in variables["attr_state"].items():
                value = self._gurobi_value(model, variable)
                distance = abs(value - round(value))
                if distance > 1e-7:
                    fractional_by_kind["attr_state"].append(
                        (distance, "attr_state", key, value)
                    )
            for zone_index, variable in variables["zone"].items():
                value = self._gurobi_value(model, variable)
                distance = abs(value - round(value))
                if distance > 1e-7:
                    fractional_by_kind["zone"].append(
                        (
                            distance,
                            "zone",
                            self._zones[zone_index].candidate_indices,
                            value,
                        )
                    )
            category = next(
                (
                    candidate
                    for candidate in ("area_use", "attr_state", "zone")
                    if fractional_by_kind[candidate]
                ),
                None,
            )
            if category is None:
                for key, variable in variables["export_flow"].items():
                    value = self._gurobi_value(model, variable)
                    distance = abs(value - round(value))
                    if distance > 1e-7:
                        fractional_by_kind["export_flow"].append(
                            (distance, "export_flow", key, value)
                        )
                if fractional_by_kind["export_flow"]:
                    category = "export_flow"
            if category is None:
                for key, variable in variables["import_reserve"].items():
                    value = self._gurobi_value(model, variable)
                    distance = abs(value - round(value))
                    if distance > 1e-7:
                        fractional_by_kind["import_reserve"].append(
                            (distance, "import_reserve", key, value)
                        )
                if fractional_by_kind["import_reserve"]:
                    category = "import_reserve"
            if category is not None:
                fractional = [
                    item
                    for values in fractional_by_kind.values()
                    for item in values
                ]
                _distance, _category, key, value = max(
                    fractional_by_kind[category],
                    key=lambda item: (item[0], str(item[2])),
                )
                return {
                    "closed": True,
                    "status": "fractional",
                    "objective": objective,
                    "branch_category": category,
                    "branch_key": key,
                    "branch_value": value,
                    "fractional_variable_count": len(fractional),
                    "fractional_by_category": {
                        key: len(values)
                        for key, values in fractional_by_kind.items()
                        if values
                    },
                    "rounds": rounds,
                    "minimum_reduced_cost": minimum_reduced_cost,
                    "lp_warm_start": warm_start_stats,
                    "child_lp_warm_start": model.captureLpWarmStart(),
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
            return {
                "closed": True,
                "status": "integer",
                "objective": objective,
                "selected_zones": selected,
                "export_flow": export_flow,
                "import_reserve": import_reserve,
                "rounds": rounds,
                "minimum_reduced_cost": minimum_reduced_cost,
            }
        finally:
            self._free_gurobi_model(model)

    def _run_branch_and_price(
        self,
        *,
        root_lower_bound: float,
        root_branch_seed: dict[str, object] | None,
        incumbent_objective: float,
        incumbent_zones: set[int],
        incumbent_export_flow: dict[tuple[str, str], int],
        incumbent_import_reserve: dict[tuple[str, str, str], int],
        root_lp_warm_start: dict[str, dict[str, float]] | None,
        deadline: float,
    ) -> tuple[
        set[int],
        dict[tuple[str, str], int],
        dict[tuple[str, str, str], int],
        dict[str, object],
    ]:
        started = perf_counter()
        next_node_id = 0
        queue: list[tuple[float, int, ZoneBranchNode]] = []
        root_reused = root_branch_seed is not None
        if root_branch_seed is None:
            root = ZoneBranchNode(
                node_id=next_node_id,
                depth=0,
                lower_bound_estimate=float(root_lower_bound),
                decisions=(),
                lp_warm_start=root_lp_warm_start,
            )
            heappush(queue, (root.lower_bound_estimate, root.node_id, root))
            next_node_id += 1
        else:
            value = float(root_branch_seed["value"])
            category = str(root_branch_seed["category"])
            key = root_branch_seed["key"]
            for sense, bound in (
                ("le", math.floor(value)),
                ("ge", math.ceil(value)),
            ):
                decision = ZoneBranchDecision(category, key, sense, float(bound))
                child = ZoneBranchNode(
                    node_id=next_node_id,
                    depth=1,
                    lower_bound_estimate=float(root_lower_bound),
                    decisions=(decision,),
                    lp_warm_start=root_lp_warm_start,
                )
                heappush(queue, (child.lower_bound_estimate, child.node_id, child))
                next_node_id += 1
        best_objective = float(incumbent_objective)
        best_zones = set(incumbent_zones)
        best_export_flow = dict(incumbent_export_flow)
        best_import_reserve = dict(incumbent_import_reserve)
        processed = 0
        pruned_by_bound = 0
        infeasible_nodes = 0
        integer_nodes = 0
        generated_at_start = len(self._zones)
        node_log = []
        interrupted_node_bound = None
        stopped_for_low_progress = False
        warm_started_node_count = 0
        initial_gap = max(0.0, best_objective - float(root_lower_bound))
        allocated_branch_time = max(1e-9, deadline - started)
        probe_deadline = started + allocated_branch_time * float(
            self.zone_config.branch_probe_time_fraction
        )
        while (
            queue
            and processed < int(self.zone_config.max_branch_nodes)
            and perf_counter() < deadline
        ):
            estimate, _queue_id, node = heappop(queue)
            if estimate >= best_objective - 1e-9:
                pruned_by_bound += 1
                continue
            node_deadline = (
                min(deadline, probe_deadline)
                if processed == 0
                else deadline
            )
            result = self._solve_branch_price_node(node, node_deadline)
            processed += 1
            if node.lp_warm_start:
                warm_started_node_count += 1
            if len(node_log) < 50:
                node_log.append(
                    {
                        "node_id": node.node_id,
                        "depth": node.depth,
                        "estimate": estimate,
                        **{
                            key: result.get(key)
                            for key in (
                                "status",
                                "objective",
                                "rounds",
                                "fractional_variable_count",
                                "branch_category",
                                "branch_value",
                                "fractional_by_category",
                            )
                            if key in result
                        },
                    }
                )
            if not result.get("closed"):
                interrupted_node_bound = estimate
                heappush(queue, (estimate, node.node_id, node))
                if node_deadline < deadline - 1e-9:
                    stopped_for_low_progress = True
                break
            if result["status"] == "infeasible":
                infeasible_nodes += 1
                continue
            node_bound = float(result["objective"])
            if node_bound >= best_objective - 1e-9:
                pruned_by_bound += 1
                continue
            if result["status"] == "integer":
                integer_nodes += 1
                best_objective = node_bound
                best_zones = set(result["selected_zones"])
                best_export_flow = dict(result["export_flow"])
                best_import_reserve = dict(result["import_reserve"])
                continue
            value = float(result["branch_value"])
            category = str(result["branch_category"])
            key = result["branch_key"]
            lower = math.floor(value)
            upper = math.ceil(value)
            child_lp_warm_start = result.get("child_lp_warm_start")
            for sense, bound in (("le", lower), ("ge", upper)):
                decision = ZoneBranchDecision(category, key, sense, float(bound))
                child = ZoneBranchNode(
                    node_id=next_node_id,
                    depth=node.depth + 1,
                    lower_bound_estimate=node_bound,
                    decisions=node.decisions + (decision,),
                    lp_warm_start=child_lp_warm_start,
                )
                heappush(
                    queue,
                    (child.lower_bound_estimate, child.node_id, child),
                )
                next_node_id += 1

            if processed >= (2 if root_reused else 3) and queue and initial_gap > 1e-12:
                live_lower_bound = min(entry[0] for entry in queue)
                gap_closure = max(
                    0.0,
                    (live_lower_bound - float(root_lower_bound)) / initial_gap,
                )
                allocated = max(1e-9, deadline - started)
                elapsed_fraction = (perf_counter() - started) / allocated
                if (
                    elapsed_fraction
                    >= float(self.zone_config.branch_probe_time_fraction)
                    and gap_closure
                    < float(self.zone_config.branch_min_gap_closure)
                ):
                    stopped_for_low_progress = True
                    break

        global_lower_bound = min(
            (entry[0] for entry in queue),
            default=best_objective,
        )
        absolute_gap = max(0.0, best_objective - global_lower_bound)
        return best_zones, best_export_flow, best_import_reserve, {
            "status": (
                "optimal"
                if not queue and interrupted_node_bound is None
                else "low_progress"
                if stopped_for_low_progress
                else "time_limit"
                if perf_counter() >= deadline or interrupted_node_bound is not None
                else "node_limit"
            ),
            "incumbent_objective": best_objective,
            "global_lower_bound": global_lower_bound,
            "absolute_gap": absolute_gap,
            "relative_gap": absolute_gap / max(abs(best_objective), 1e-12),
            "processed_node_count": processed,
            "open_node_count": len(queue),
            "pruned_by_bound_count": pruned_by_bound,
            "infeasible_node_count": infeasible_nodes,
            "integer_node_count": integer_nodes,
            "generated_zone_count": len(self._zones) - generated_at_start,
            "root_solution_reused": root_reused,
            "warm_started_node_count": warm_started_node_count,
            "root_branch_category": (
                root_branch_seed.get("category") if root_branch_seed else None
            ),
            "stopped_for_low_progress": stopped_for_low_progress,
            "gap_closure_fraction": max(
                0.0,
                (global_lower_bound - float(root_lower_bound))
                / max(initial_gap, 1e-12),
            ),
            "seconds": perf_counter() - started,
            "node_log": node_log,
        }

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

    def _zone_objective_certificate(
        self,
        selected_zone_indices: set[int],
        export_flow: dict[tuple[str, str], int],
        import_reserve: dict[tuple[str, str, str], int],
        solver_objective: float,
    ) -> dict[str, object]:
        """Reconstruct the primary zone objective from physical decisions."""

        flow_columns = self._master_index_sets()["flow_columns"]
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
        if len(selected_zone_indices) < len(self.groups):
            raise RuntimeError(
                "a positive-demand group has no selected contiguous zone"
            )
        if len(used_group_areas) < len(self.groups):
            raise RuntimeError(
                "a positive-demand group has no positive area flow"
            )

        export_by_quota: Counter[tuple[str, str, str, str]] = Counter()
        for key, quantity in export_flow.items():
            export_by_quota[flow_columns[key].quota_key] += int(quantity)
        export_guidance_deviation = sum(
            abs(
                float(export_by_quota.get(key, 0))
                - float(self._area_size_target(*key))
            )
            for key in self._master_area_guidance_keys
        )
        import_by_area: Counter[tuple[str, str, str]] = Counter()
        for (flow, size, bay_key), quantity in import_reserve.items():
            if int(quantity) > 0:
                import_by_area[
                    (flow, self.bays[bay_key].area_no, size)
                ] += int(quantity)
        import_guidance_deviation = sum(
            abs(
                int(import_by_area.get(key, 0))
                - int(self.import_area_size_reference.get(key, 0))
            )
            for key in set(import_by_area) | set(self.import_area_size_reference)
        )
        proximity_sum = 0.0
        berth_distance_sum = 0.0
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
        raw = {
            "extra_group_areas": float(
                len(used_group_areas) - len(self.groups)
            ),
            "extra_contiguous_zones": float(
                len(selected_zone_indices) - len(self.groups)
            ),
            "existing_group_normalized_distance_sum": float(proximity_sum),
            "large_plan_l1_deviation": float(
                export_guidance_deviation + import_guidance_deviation
            ),
            "unused_reserved_capacity_boxes": float(
                reserved_capacity - assigned_boxes
            ),
            "berth_normalized_distance_sum": float(berth_distance_sum),
        }
        normalized = {
            "area_dispersion": raw["extra_group_areas"]
            / self._zone_objective_scale("area_dispersion"),
            "zone_dispersion": raw["extra_contiguous_zones"]
            / self._zone_objective_scale("zone_dispersion"),
            "existing_group_proximity": raw[
                "existing_group_normalized_distance_sum"
            ]
            / self._zone_objective_scale("existing_group_proximity"),
            "area_guidance": raw["large_plan_l1_deviation"]
            / self._zone_objective_scale("area_guidance"),
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
        auxiliary_slack = self._absolute_deviation_auxiliary_slack(
            solver_objective,
            reconstructed,
            context="contiguous-zone master",
        )
        return {
            "certified": True,
            "objective": reconstructed,
            "solver_incumbent_objective": float(solver_objective),
            "solver_auxiliary_slack": auxiliary_slack,
            "absolute_reconstruction_difference": abs(
                float(solver_objective) - reconstructed
            ),
            "raw": raw,
            "normalized": normalized,
            "weighted": weighted,
            "components": weighted,
            "weights": weights,
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
            "export_guidance_l1_deviation": export_guidance_deviation,
            "import_guidance_l1_deviation": import_guidance_deviation,
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
        total_limit = max(0.01, float(self.config.total_time_limit))
        deadline = started + total_limit
        preparation = self._prepare_zones()
        model, variables = self._build_zone_master()
        try:
            root_deadline = perf_counter() + max(
                0.0, deadline - perf_counter()
            ) * float(self.zone_config.root_time_fraction)
            root = self._run_root_generation(model, variables, root_deadline)
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
            pool_enrichment = self._enrich_integer_pool(model, variables)
            zone_mip_deadline = perf_counter() + max(
                0.0, deadline - perf_counter()
            ) * float(self.zone_config.zone_mip_time_fraction)
            (
                initial_zones,
                initial_export_flow,
                initial_import_reserve,
                zone_mip_initial,
            ) = self._integerize_zone_master(
                model, variables, zone_mip_deadline
            )
            if not initial_zones:
                raise RuntimeError(
                    "restricted zone master did not obtain an integer support: "
                    f"{zone_mip_initial}"
                )
            fill_reserve = total_limit * float(
                self.zone_config.fill_time_fraction
            )
            (
                selected_zones,
                selected_export_flow,
                selected_import_reserve,
                fix_optimize,
            ) = self._run_conflict_fix_optimize(
                model,
                variables,
                set(initial_zones),
                initial_export_flow,
                initial_import_reserve,
                deadline - fill_reserve,
            )
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
        zone_global_lower_bound = float(root["root_objective"])
        zone_absolute_gap = max(0.0, zone_upper_bound - zone_global_lower_bound)
        zone_relative_gap = zone_absolute_gap / max(abs(zone_upper_bound), 1e-12)
        diagnostics = {
            "algorithm": (
                "contiguous_zone_generation_conflict_fix_optimize_"
                "with_exact_recourse"
            ),
            "model_scope": "actual_quantity_flow_on_dedicated_contiguous_row_zones",
            "formulation": "zone_flow_master_plus_flow_fixed_exact_row_recourse",
            "decomposition": (
                "exact_rmq_interval_pricing_restricted_integer_master_"
                "conflict_fix_optimize_then_certified_row_realization"
            ),
            "planned_group_count": len(self.groups),
            "planned_box_count": sum(group.demand for group in self.groups),
            "candidate_row_location_count": len(self._columns),
            "zone_preparation": preparation,
            "zone_root": root,
            "zone_pool_enrichment": pool_enrichment,
            "zone_mip": zone_mip,
            "zone_mip_initial": zone_mip_initial,
            "zone_fix_optimize": fix_optimize,
            "zone_model_upper_bound": zone_upper_bound,
            "zone_model_global_lower_bound": zone_global_lower_bound,
            "zone_model_absolute_gap": zone_absolute_gap,
            "zone_model_relative_gap": zone_relative_gap,
            "zone_model_lower_bound_source": "closed_exact_zone_pricing_root",
            "zone_restricted_pool_bound": zone_mip.get("bound"),
            "zone_restricted_pool_bound_is_global": False,
            "zone_selected_candidate_count": len(candidate_indices),
            "zone_selected_support_source": selected_support_source,
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
                "fix_optimize_local_remaining_fraction": float(
                    self.zone_config.fix_optimize_local_fraction
                ),
                "fix_optimize_group_cap": int(
                    self.zone_config.fix_optimize_group_count
                ),
                "fix_optimize_group_policy": (
                    "objective_conflict_ranking_up_to_group_cap"
                ),
                "final_fill_total_fraction": float(
                    self.zone_config.fill_time_fraction
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
            "import_capacity_reservation": {
                "source_quantity_field": "new_qty",
                "role": "anonymous_size_compatible_capacity_only",
                "area_policy": "weighted_l1_deviation_from_big_plan_reference",
                "constraint_scope": [
                    "area_function",
                    "bay_size",
                    "physical_capacity",
                ],
                "excluded_constraints": [
                    "bay_no_mix",
                    "row_no_mix",
                    "container_group_attributes",
                ],
                "import_boxes": int(sum(self.import_area_size_reference.values())),
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
]
