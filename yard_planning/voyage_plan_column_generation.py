"""Complete-voyage plan column generation with exact row-level pricing."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from time import perf_counter

from .direct_milp import DirectMilpPlanner
from .gurobi_backend import GurobiModel
from .planner import ColumnGenerationResult, PlacementColumn


VoyageKey = str


@dataclass(frozen=True)
class VoyagePlanPricingConfig:
    """Instance-independent controls for complete-voyage pricing."""

    plans_per_pricing: int = 3


@dataclass(frozen=True)
class VoyagePlan:
    """One complete integer allocation of every group in one voyage."""

    plan_id: str
    voyage_id: str
    placements: tuple[PlacementColumn, ...]
    business_cost: float
    master_coefficients: tuple[tuple[str, object, float], ...]


@dataclass(frozen=True)
class VoyageAreaLocalPattern:
    """One partial integer plan for one voyage in one yard area."""

    voyage_id: str
    area_no: str
    placements: tuple[PlacementColumn, ...]
    business_cost: float
    group_quantities: tuple[tuple[str, int], ...]
    master_coefficients: tuple[tuple[str, object, float], ...]


@dataclass
class _VoyagePricingModel:
    voyage_id: str
    model: GurobiModel
    candidates: tuple[PlacementColumn, ...]
    placement_variables: dict[int, object]
    area_use_variables: dict[tuple[tuple[str, ...], str], object]
    row_use_variables: dict[tuple[tuple[str, ...], str, str], object]
    offset_variable: object
    build_seconds: float


@dataclass
class _LocalPricingModel:
    pair: tuple[str, str]
    model: GurobiModel
    candidates: tuple[PlacementColumn, ...]
    placement_variables: dict[int, object]
    area_use_variables: dict[tuple[str, ...], object]
    row_use_variables: dict[tuple[tuple[str, ...], str, str], object]
    build_seconds: float


class VoyagePlanColumnGenerationPlanner(DirectMilpPlanner):
    """Dantzig--Wolfe decomposition with one complete-plan block per voyage.

    A pricing MIP assigns all groups of one voyage across all feasible yard
    areas and rows.  The outer master coordinates only global shared yard
    resources, attribute states, export guidance, and anonymous import
    reservation.  This keeps the number of convexity blocks proportional to
    the number of voyages rather than voyage-area pairs.
    """

    def __init__(
        self,
        problem,
        config=None,
        pricing_config: VoyagePlanPricingConfig | None = None,
    ) -> None:
        super().__init__(problem, config)
        self.pricing_config = pricing_config or VoyagePlanPricingConfig()
        if int(self.pricing_config.plans_per_pricing) <= 0:
            raise ValueError("plans_per_pricing must be positive")
        self._voyage_candidates: dict[VoyageKey, tuple[PlacementColumn, ...]] = {}
        self._base_candidate_by_static_key: dict[tuple, PlacementColumn] = {}
        self._plans: list[VoyagePlan] = []
        self._plan_index_by_identity: dict[tuple, int] = {}
        self._pricing_models: dict[VoyageKey, _VoyagePricingModel] = {}
        self._local_candidates: dict[
            tuple[str, str], tuple[PlacementColumn, ...]
        ] = {}
        self._local_patterns: dict[
            tuple[str, str], list[VoyageAreaLocalPattern]
        ] = defaultdict(list)
        self._local_pattern_identities: dict[tuple[str, str], set[tuple]] = (
            defaultdict(set)
        )
        self._local_pricing_models: dict[
            tuple[str, str], _LocalPricingModel
        ] = {}
        self._seed_model_build_seconds = 0.0
        self._initialization_records: list[dict] = []

    @staticmethod
    def _dual(
        duals: dict[tuple[str, object], float],
        section: str,
        key: object,
    ) -> float:
        return float(duals.get((section, key), 0.0))

    @staticmethod
    def _plan_identity(plan: VoyagePlan) -> tuple:
        return (
            plan.voyage_id,
            tuple(
                (
                    placement.group_id,
                    placement.bay_key,
                    int(placement.quantity),
                    placement.row_allocation,
                )
                for placement in plan.placements
            ),
        )

    def _append_plan(self, plan: VoyagePlan) -> int:
        identity = self._plan_identity(plan)
        if identity in self._plan_index_by_identity:
            raise ValueError(f"duplicate complete-voyage plan: {identity}")
        index = len(self._plans)
        stored = replace(plan, plan_id=f"VP{index + 1:07d}")
        self._plans.append(stored)
        self._plan_index_by_identity[identity] = index
        return index

    def _aggregate_master_coefficients(
        self, placements: tuple[PlacementColumn, ...]
    ) -> tuple[tuple[str, object, float], ...]:
        coefficients: defaultdict[str, Counter[object]] = defaultdict(Counter)
        for placement in placements:
            for section, values in self._placement_master_coefficients(
                placement
            ).items():
                coefficients[section].update(values)
        return tuple(
            (section, key, float(value))
            for section, values in sorted(coefficients.items())
            for key, value in sorted(values.items(), key=lambda item: repr(item[0]))
            if abs(float(value)) > 0.0
        )

    def _initialize_candidate_blocks(self) -> tuple[VoyageKey, ...]:
        self._prepare_master_index_sets()
        self._prepare_objective_normalization()
        self._plans.clear()
        self._plan_index_by_identity.clear()
        self._voyage_candidates.clear()
        self._local_candidates.clear()
        self._local_patterns.clear()
        self._local_pattern_identities.clear()
        self._base_candidate_by_static_key.clear()
        self._initialization_records.clear()
        self._seed_model_build_seconds = 0.0
        by_voyage: defaultdict[str, list[PlacementColumn]] = defaultdict(list)
        for group in self.groups:
            placements = self._base_placements_for_group(group)
            if not placements:
                raise ValueError(
                    "declared export group has no feasible row location: "
                    f"group={group.group_id}"
                )
            for candidate in placements:
                by_voyage[group.voyage_id].append(candidate)
                self._base_candidate_by_static_key[
                    self._base_placement_static_key(candidate)
                ] = candidate
        self._voyage_candidates = {
            voyage_id: tuple(
                sorted(
                    candidates,
                    key=lambda candidate: (
                        candidate.group_id,
                        candidate.area_no,
                        self.bays[candidate.bay_key].bay_order,
                        candidate.row_allocation,
                    ),
                )
            )
            for voyage_id, candidates in sorted(by_voyage.items())
        }
        if not self._voyage_candidates:
            raise ValueError("complete-voyage decomposition has no pricing block")
        local_candidates: defaultdict[
            tuple[str, str], list[PlacementColumn]
        ] = defaultdict(list)
        for voyage_id, candidates in self._voyage_candidates.items():
            for candidate in candidates:
                local_candidates[(voyage_id, candidate.area_no)].append(candidate)
        self._local_candidates = {
            pair: tuple(candidates)
            for pair, candidates in sorted(local_candidates.items())
        }
        for voyage_id, area_no in self._local_candidates:
            self._append_local_pattern(
                VoyageAreaLocalPattern(
                    voyage_id=voyage_id,
                    area_no=area_no,
                    placements=(),
                    business_cost=0.0,
                    group_quantities=(),
                    master_coefficients=(),
                )
            )
        return tuple(self._voyage_candidates)

    @staticmethod
    def _local_pattern_identity(pattern: VoyageAreaLocalPattern) -> tuple:
        return tuple(
            (
                placement.group_id,
                placement.bay_key,
                int(placement.quantity),
                placement.row_allocation,
            )
            for placement in pattern.placements
        )

    def _append_local_pattern(
        self, pattern: VoyageAreaLocalPattern
    ) -> bool:
        pair = (pattern.voyage_id, pattern.area_no)
        identity = self._local_pattern_identity(pattern)
        if identity in self._local_pattern_identities[pair]:
            return False
        self._local_pattern_identities[pair].add(identity)
        self._local_patterns[pair].append(pattern)
        return True

    def _split_plan_into_local_patterns(self, plan: VoyagePlan) -> None:
        by_area: defaultdict[str, list[PlacementColumn]] = defaultdict(list)
        for placement in plan.placements:
            by_area[placement.area_no].append(placement)
        for area_no, placements in by_area.items():
            placement_tuple = tuple(
                sorted(
                    placements,
                    key=lambda placement: (
                        placement.group_id,
                        self.bays[placement.bay_key].bay_order,
                        placement.row_allocation,
                    ),
                )
            )
            used_groups = {placement.group_key for placement in placement_tuple}
            used_rows = {
                (
                    placement.group_key,
                    placement.bay_key,
                    next(
                        row_no
                        for bay_key, row_no, _qty in placement.row_allocation
                        if bay_key == placement.bay_key
                    ),
                )
                for placement in placement_tuple
            }
            cost = sum(
                float(placement.intrinsic_cost) * int(placement.quantity)
                for placement in placement_tuple
            )
            cost += self._area_activation_penalty() * len(used_groups)
            cost += self._row_activation_penalty() * len(used_rows)
            quantities: Counter[str] = Counter()
            for placement in placement_tuple:
                quantities[placement.group_id] += int(placement.quantity)
            self._append_local_pattern(
                VoyageAreaLocalPattern(
                    voyage_id=plan.voyage_id,
                    area_no=area_no,
                    placements=placement_tuple,
                    business_cost=float(cost),
                    group_quantities=tuple(sorted(quantities.items())),
                    master_coefficients=self._aggregate_master_coefficients(
                        placement_tuple
                    ),
                )
            )

    def _plan_from_solution(
        self,
        pricing: _VoyagePricingModel,
        solution_number: int | None = None,
    ) -> VoyagePlan:
        def value(variable) -> float:
            if solution_number is None:
                return self._gurobi_value(pricing.model, variable)
            return pricing.model.getPoolValue(variable, solution_number)

        placements: list[PlacementColumn] = []
        assigned_by_group: Counter[str] = Counter()
        business_cost = 0.0
        for index, variable in pricing.placement_variables.items():
            quantity = int(round(value(variable)))
            if quantity <= 0:
                continue
            candidate = pricing.candidates[index]
            group = self.groups_by_id[candidate.group_id]
            placement = replace(
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
                    for bay_key, row_no, _old in candidate.row_allocation
                ),
            )
            placements.append(placement)
            assigned_by_group[candidate.group_id] += quantity
            business_cost += float(candidate.intrinsic_cost) * quantity

        voyage_groups = [
            group for group in self.groups if group.voyage_id == pricing.voyage_id
        ]
        for group in voyage_groups:
            if assigned_by_group[group.group_id] != int(group.demand):
                raise RuntimeError(
                    "complete-voyage pricing returned an incomplete plan: "
                    f"voyage={pricing.voyage_id}, group={group.group_id}, "
                    f"assigned={assigned_by_group[group.group_id]}, "
                    f"required={group.demand}"
                )
        used_areas = {
            (placement.group_key, placement.area_no)
            for placement in placements
        }
        used_rows = {
            (
                placement.group_key,
                placement.bay_key,
                next(
                    row_no
                    for bay_key, row_no, _qty in placement.row_allocation
                    if bay_key == placement.bay_key
                ),
            )
            for placement in placements
        }
        operational_groups = {
            self._operational_group_key(group) for group in voyage_groups
        }
        business_cost += self._area_activation_penalty() * (
            len(used_areas) - len(operational_groups)
        )
        business_cost += self._row_activation_penalty() * (
            len(used_rows) - len(operational_groups)
        )
        placements.sort(
            key=lambda placement: (
                placement.group_id,
                placement.area_no,
                self.bays[placement.bay_key].bay_order,
                placement.row_allocation,
            )
        )
        placement_tuple = tuple(placements)
        return VoyagePlan(
            plan_id="",
            voyage_id=pricing.voyage_id,
            placements=placement_tuple,
            business_cost=float(business_cost),
            master_coefficients=self._aggregate_master_coefficients(
                placement_tuple
            ),
        )

    def _build_voyage_pricing_model(
        self, voyage_id: VoyageKey
    ) -> _VoyagePricingModel:
        from gurobipy import quicksum

        started = perf_counter()
        candidates = self._voyage_candidates[voyage_id]
        model = GurobiModel(
            f"complete_voyage_pricing_{self._key_name((voyage_id,))}"
        )
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        self._set_gurobi_param(model, "MIPGap", 0.0)
        model.setMinimize()

        capacities = {
            index: self._base_location_capacity(
                self.groups_by_id[candidate.group_id], candidate
            )
            for index, candidate in enumerate(candidates)
        }
        placement_variables = {
            index: model.addVar(
                lb=0.0,
                ub=float(capacities[index]),
                vtype="I",
                name=f"x_{index}",
            )
            for index in range(len(candidates))
        }
        coefficient_rows: defaultdict[
            str, defaultdict[object, list[tuple[int, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        group_indices: defaultdict[str, list[int]] = defaultdict(list)
        group_area_indices: defaultdict[
            tuple[tuple[str, ...], str], list[int]
        ] = defaultdict(list)
        group_row_indices: defaultdict[
            tuple[tuple[str, ...], str, str], list[int]
        ] = defaultdict(list)
        operational_group_demand: Counter[tuple[str, ...]] = Counter()
        for group in self.groups:
            if group.voyage_id == voyage_id:
                operational_group_demand[
                    self._operational_group_key(group)
                ] += int(group.demand)
        for index, candidate in enumerate(candidates):
            group_indices[candidate.group_id].append(index)
            group_area_indices[(candidate.group_key, candidate.area_no)].append(
                index
            )
            anchor_row = next(
                row_no
                for bay_key, row_no, _qty in candidate.row_allocation
                if bay_key == candidate.bay_key
            )
            group_row_indices[
                (candidate.group_key, candidate.bay_key, anchor_row)
            ].append(index)
            for section, values in self._placement_master_coefficients(
                candidate
            ).items():
                for key, coefficient in values.items():
                    if coefficient:
                        coefficient_rows[section][key].append(
                            (index, float(coefficient))
                        )

        for bay_key, items in sorted(
            coefficient_rows["bay_capacity_limit"].items()
        ):
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                <= int(self.bays[bay_key].physical_capacity),
                name=f"voyage_bay_cap_{self._key_name((bay_key,))}",
            )
        for key, items in sorted(
            coefficient_rows["bay_size_limit"].items()
        ):
            bay_key, size = key
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                <= int(self.bays[bay_key].cap_by_size.get(size, 0)),
                name=f"voyage_bay_size_{self._key_name(key)}",
            )
        for key, items in sorted(
            coefficient_rows["row_capacity_limit"].items()
        ):
            bay_key, row_no = key
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                <= int(
                    self.bays[bay_key].row_physical_capacity.get(
                        row_no, self.bays[bay_key].physical_capacity
                    )
                ),
                name=f"voyage_row_cap_{self._key_name(key)}",
            )
        for key, items in sorted(
            coefficient_rows["row_size_limit"].items()
        ):
            bay_key, row_no, size = key
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                <= int(
                    self.bays[bay_key].row_cap_by_size.get(size, {}).get(
                        row_no,
                        self.bays[bay_key].cap_by_size.get(size, 0),
                    )
                ),
                name=f"voyage_row_size_{self._key_name(key)}",
            )

        stacks_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        for key, items in sorted(
            coefficient_rows["bay_port_stack_link"].items()
        ):
            bay_key, _mix_key, size = key
            group = self.groups_by_id.get(
                self._master_stack_sample_group.get(key, "")
            )
            if group is None:
                continue
            stack_count = self._stack_count_for_group(bay_key, size, group)
            unit_capacity = self._stack_unit_capacity_for_group(
                bay_key, size, group
            )
            if stack_count <= 0 or unit_capacity <= 0:
                continue
            stack = model.addVar(
                lb=0.0,
                ub=float(stack_count),
                vtype="I",
                name=f"voyage_stack_{self._key_name(key)}",
            )
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                <= unit_capacity * stack,
                name=f"voyage_stack_link_{self._key_name(key)}",
            )
            stacks_by_bay_size[(bay_key, size)].append(stack)
        for key, stacks in sorted(stacks_by_bay_size.items()):
            model.addConstr(
                quicksum(stacks) <= self._stack_count_for_bay_size(*key),
                name=f"voyage_stack_total_{self._key_name(key)}",
            )

        voyage_groups = [
            group for group in self.groups if group.voyage_id == voyage_id
        ]
        for group in voyage_groups:
            indices = group_indices.get(group.group_id, [])
            if not indices:
                raise ValueError(
                    "complete-voyage pricing has no candidate for group: "
                    f"voyage={voyage_id}, group={group.group_id}"
                )
            model.addConstr(
                quicksum(placement_variables[index] for index in indices)
                == int(group.demand),
                name=f"voyage_group_demand_{group.group_id}",
            )

        area_use_variables: dict[
            tuple[tuple[str, ...], str], object
        ] = {}
        for key, indices in sorted(group_area_indices.items()):
            group_key, area_no = key
            upper = min(
                int(operational_group_demand[group_key]),
                sum(capacities[index] for index in indices),
            )
            use = model.addVar(
                vtype="B",
                name=(
                    f"voyage_group_area_{self._key_name(group_key)}_"
                    f"{area_no}"
                ),
            )
            assigned = quicksum(
                placement_variables[index] for index in indices
            )
            model.addConstr(
                assigned <= max(1, upper) * use,
                name=(
                    f"voyage_group_area_upper_"
                    f"{self._key_name((*group_key, area_no))}"
                ),
            )
            model.addConstr(
                use <= assigned,
                name=(
                    f"voyage_group_area_lower_"
                    f"{self._key_name((*group_key, area_no))}"
                ),
            )
            area_use_variables[key] = use

        row_use_variables: dict[
            tuple[tuple[str, ...], str, str], object
        ] = {}
        for key, indices in sorted(group_row_indices.items()):
            group_key, bay_key, row_no = key
            upper = min(
                int(operational_group_demand[group_key]),
                sum(capacities[index] for index in indices),
            )
            use = model.addVar(
                vtype="B",
                name=(
                    f"voyage_group_row_{self._key_name(group_key)}_"
                    f"{self._key_name((bay_key, row_no))}"
                ),
            )
            assigned = quicksum(
                placement_variables[index] for index in indices
            )
            model.addConstr(
                assigned <= max(1, upper) * use,
                name=(
                    f"voyage_group_row_upper_"
                    f"{self._key_name((*group_key, bay_key, row_no))}"
                ),
            )
            model.addConstr(
                use <= assigned,
                name=(
                    f"voyage_group_row_lower_"
                    f"{self._key_name((*group_key, bay_key, row_no))}"
                ),
            )
            row_use_variables[key] = use

        self._add_local_compatibility_constraints(
            model,
            quicksum,
            placement_variables,
            coefficient_rows,
        )
        operational_group_count = len(operational_group_demand)
        offset_variable = model.addVar(
            lb=1.0,
            ub=1.0,
            name=f"voyage_cost_offset_{self._key_name((voyage_id,))}",
        )
        model.update()
        return _VoyagePricingModel(
            voyage_id=voyage_id,
            model=model,
            candidates=candidates,
            placement_variables=placement_variables,
            area_use_variables=area_use_variables,
            row_use_variables=row_use_variables,
            offset_variable=offset_variable,
            build_seconds=perf_counter() - started,
        )

    def _add_local_compatibility_constraints(
        self,
        model,
        quicksum,
        placement_variables: dict[int, object],
        coefficient_rows,
    ) -> None:
        bay_uses: defaultdict[tuple[str, str, str], list] = defaultdict(list)
        for key, items in sorted(
            coefficient_rows["bay_attr_link"].items()
        ):
            use = model.addVar(
                vtype="B", name=f"local_bay_use_{self._key_name(key)}"
            )
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                <= self._master_bay_attr_big_m[key] * use,
                name=f"local_bay_attr_link_{self._key_name(key)}",
            )
            bay_key, attr, scope, _value = key
            bay_uses[(bay_key, attr, scope)].append(use)
        for key, uses in sorted(bay_uses.items()):
            model.addConstr(
                quicksum(uses) <= 1,
                name=f"local_bay_attr_one_{self._key_name(key)}",
            )

        row_uses: defaultdict[tuple[str, str, str, str], list] = (
            defaultdict(list)
        )
        for key, items in sorted(
            coefficient_rows["row_attr_link"].items()
        ):
            use = model.addVar(
                vtype="B", name=f"local_row_use_{self._key_name(key)}"
            )
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                <= self._master_row_attr_big_m[key] * use,
                name=f"local_row_attr_link_{self._key_name(key)}",
            )
            bay_key, row_no, attr, scope, _value = key
            row_uses[(bay_key, row_no, attr, scope)].append(use)
        for key, uses in sorted(row_uses.items()):
            model.addConstr(
                quicksum(uses) <= 1,
                name=f"local_row_attr_one_{self._key_name(key)}",
            )

    def _build_local_pricing_model(
        self, pair: tuple[str, str]
    ) -> _LocalPricingModel:
        from gurobipy import quicksum

        started = perf_counter()
        voyage_id, area_no = pair
        candidates = self._local_candidates[pair]
        model = GurobiModel(
            f"local_area_pricing_{self._key_name((voyage_id, area_no))}"
        )
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        self._set_gurobi_param(model, "MIPGap", 0.0)
        model.setMinimize()
        capacities = {
            index: self._base_location_capacity(
                self.groups_by_id[candidate.group_id], candidate
            )
            for index, candidate in enumerate(candidates)
        }
        placement_variables = {
            index: model.addVar(
                lb=0.0,
                ub=float(capacities[index]),
                vtype="I",
                name=f"x_{index}",
            )
            for index in range(len(candidates))
        }
        coefficient_rows: defaultdict[
            str, defaultdict[object, list[tuple[int, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        group_indices: defaultdict[str, list[int]] = defaultdict(list)
        group_area_indices: defaultdict[tuple[str, ...], list[int]] = (
            defaultdict(list)
        )
        group_row_indices: defaultdict[
            tuple[tuple[str, ...], str, str], list[int]
        ] = defaultdict(list)
        operational_group_demand: Counter[tuple[str, ...]] = Counter()
        for group in self.groups:
            if group.voyage_id == voyage_id:
                operational_group_demand[
                    self._operational_group_key(group)
                ] += int(group.demand)
        for index, candidate in enumerate(candidates):
            group_indices[candidate.group_id].append(index)
            group_area_indices[candidate.group_key].append(index)
            anchor_row = next(
                row_no
                for bay_key, row_no, _qty in candidate.row_allocation
                if bay_key == candidate.bay_key
            )
            group_row_indices[
                (candidate.group_key, candidate.bay_key, anchor_row)
            ].append(index)
            for section, values in self._placement_master_coefficients(
                candidate
            ).items():
                for key, coefficient in values.items():
                    if coefficient:
                        coefficient_rows[section][key].append(
                            (index, float(coefficient))
                        )

        for bay_key, items in sorted(
            coefficient_rows["bay_capacity_limit"].items()
        ):
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                <= int(self.bays[bay_key].physical_capacity),
                name=f"local_bay_cap_{self._key_name((bay_key,))}",
            )
        for key, items in sorted(
            coefficient_rows["bay_size_limit"].items()
        ):
            bay_key, size = key
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                <= int(self.bays[bay_key].cap_by_size.get(size, 0)),
                name=f"local_bay_size_{self._key_name(key)}",
            )
        for key, items in sorted(
            coefficient_rows["row_capacity_limit"].items()
        ):
            bay_key, row_no = key
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                <= int(
                    self.bays[bay_key].row_physical_capacity.get(
                        row_no, self.bays[bay_key].physical_capacity
                    )
                ),
                name=f"local_row_cap_{self._key_name(key)}",
            )
        for key, items in sorted(
            coefficient_rows["row_size_limit"].items()
        ):
            bay_key, row_no, size = key
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                <= int(
                    self.bays[bay_key].row_cap_by_size.get(size, {}).get(
                        row_no,
                        self.bays[bay_key].cap_by_size.get(size, 0),
                    )
                ),
                name=f"local_row_size_{self._key_name(key)}",
            )

        stacks_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        for key, items in sorted(
            coefficient_rows["bay_port_stack_link"].items()
        ):
            bay_key, _mix_key, size = key
            group = self.groups_by_id.get(
                self._master_stack_sample_group.get(key, "")
            )
            if group is None:
                continue
            stack_count = self._stack_count_for_group(bay_key, size, group)
            unit_capacity = self._stack_unit_capacity_for_group(
                bay_key, size, group
            )
            if stack_count <= 0 or unit_capacity <= 0:
                continue
            stack = model.addVar(
                lb=0.0,
                ub=float(stack_count),
                vtype="I",
                name=f"local_stack_{self._key_name(key)}",
            )
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                <= unit_capacity * stack,
                name=f"local_stack_link_{self._key_name(key)}",
            )
            stacks_by_bay_size[(bay_key, size)].append(stack)
        for key, stacks in sorted(stacks_by_bay_size.items()):
            model.addConstr(
                quicksum(stacks) <= self._stack_count_for_bay_size(*key),
                name=f"local_stack_total_{self._key_name(key)}",
            )

        for group_id, indices in sorted(group_indices.items()):
            model.addConstr(
                quicksum(placement_variables[index] for index in indices)
                <= int(self.group_demand[group_id]),
                name=f"local_group_bound_{group_id}",
            )

        area_use_variables: dict[tuple[str, ...], object] = {}
        for group_key, indices in sorted(group_area_indices.items()):
            upper = min(
                int(operational_group_demand[group_key]),
                sum(capacities[index] for index in indices),
            )
            use = model.addVar(
                vtype="B",
                name=f"local_group_area_{self._key_name(group_key)}",
            )
            assigned = quicksum(
                placement_variables[index] for index in indices
            )
            model.addConstr(
                assigned <= max(1, upper) * use,
                name=f"local_group_area_upper_{self._key_name(group_key)}",
            )
            model.addConstr(
                use <= assigned,
                name=f"local_group_area_lower_{self._key_name(group_key)}",
            )
            area_use_variables[group_key] = use

        row_use_variables: dict[
            tuple[tuple[str, ...], str, str], object
        ] = {}
        for key, indices in sorted(group_row_indices.items()):
            group_key, bay_key, row_no = key
            upper = min(
                int(operational_group_demand[group_key]),
                sum(capacities[index] for index in indices),
            )
            use = model.addVar(
                vtype="B",
                name=(
                    f"local_group_row_{self._key_name(group_key)}_"
                    f"{self._key_name((bay_key, row_no))}"
                ),
            )
            assigned = quicksum(
                placement_variables[index] for index in indices
            )
            model.addConstr(
                assigned <= max(1, upper) * use,
                name=(
                    f"local_group_row_upper_"
                    f"{self._key_name((*group_key, bay_key, row_no))}"
                ),
            )
            model.addConstr(
                use <= assigned,
                name=(
                    f"local_group_row_lower_"
                    f"{self._key_name((*group_key, bay_key, row_no))}"
                ),
            )
            row_use_variables[key] = use

        self._add_local_compatibility_constraints(
            model,
            quicksum,
            placement_variables,
            coefficient_rows,
        )
        model.update()
        return _LocalPricingModel(
            pair=pair,
            model=model,
            candidates=candidates,
            placement_variables=placement_variables,
            area_use_variables=area_use_variables,
            row_use_variables=row_use_variables,
            build_seconds=perf_counter() - started,
        )

    def _local_pattern_from_solution(
        self,
        pricing: _LocalPricingModel,
        solution_number: int | None = None,
    ) -> VoyageAreaLocalPattern:
        def value(variable) -> float:
            if solution_number is None:
                return self._gurobi_value(pricing.model, variable)
            return pricing.model.getPoolValue(variable, solution_number)

        placements: list[PlacementColumn] = []
        quantities: Counter[str] = Counter()
        business_cost = 0.0
        for index, variable in pricing.placement_variables.items():
            quantity = int(round(value(variable)))
            if quantity <= 0:
                continue
            candidate = pricing.candidates[index]
            group = self.groups_by_id[candidate.group_id]
            placement = replace(
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
                    for bay_key, row_no, _old in candidate.row_allocation
                ),
            )
            placements.append(placement)
            quantities[candidate.group_id] += quantity
            business_cost += float(candidate.intrinsic_cost) * quantity
        used_groups = {placement.group_key for placement in placements}
        used_rows = {
            (
                placement.group_key,
                placement.bay_key,
                next(
                    row_no
                    for bay_key, row_no, _qty in placement.row_allocation
                    if bay_key == placement.bay_key
                ),
            )
            for placement in placements
        }
        business_cost += self._area_activation_penalty() * len(used_groups)
        business_cost += self._row_activation_penalty() * len(used_rows)
        placements.sort(
            key=lambda placement: (
                placement.group_id,
                self.bays[placement.bay_key].bay_order,
                placement.row_allocation,
            )
        )
        placement_tuple = tuple(placements)
        return VoyageAreaLocalPattern(
            voyage_id=pricing.pair[0],
            area_no=pricing.pair[1],
            placements=placement_tuple,
            business_cost=float(business_cost),
            group_quantities=tuple(sorted(quantities.items())),
            master_coefficients=self._aggregate_master_coefficients(
                placement_tuple
            ),
        )

    def _set_pricing_objective(
        self,
        pricing: _VoyagePricingModel,
        duals: dict[tuple[str, object], float],
        objective_mode: str,
    ) -> None:
        for index, variable in pricing.placement_variables.items():
            candidate = pricing.candidates[index]
            coefficient = (
                0.0
                if objective_mode == "phase_one"
                else float(candidate.intrinsic_cost)
            )
            for section, values in self._placement_master_coefficients(
                candidate
            ).items():
                for key, value in values.items():
                    coefficient -= float(value) * self._dual(
                        duals, section, key
                    )
            pricing.model.setVarObjective(variable, coefficient)
        for variable in pricing.area_use_variables.values():
            pricing.model.setVarObjective(
                variable,
                0.0
                if objective_mode == "phase_one"
                else self._area_activation_penalty(),
            )
        for variable in pricing.row_use_variables.values():
            pricing.model.setVarObjective(
                variable,
                0.0
                if objective_mode == "phase_one"
                else self._row_activation_penalty(),
            )
        voyage_group_count = len(
            {
                self._operational_group_key(group)
                for group in self.groups
                if group.voyage_id == pricing.voyage_id
            }
        )
        offset = -voyage_group_count * (
            self._area_activation_penalty()
            + self._row_activation_penalty()
        )
        pricing.model.setVarObjective(
            pricing.offset_variable,
            0.0 if objective_mode == "phase_one" else offset,
        )
        pricing.model.update()

    def _plan_reduced_cost(
        self,
        plan: VoyagePlan,
        duals: dict[tuple[str, object], float],
        objective_mode: str,
    ) -> float:
        reduced = (
            0.0 if objective_mode == "phase_one" else float(plan.business_cost)
        )
        reduced -= self._dual(
            duals, "voyage_convexity", plan.voyage_id
        )
        for section, key, coefficient in plan.master_coefficients:
            reduced -= coefficient * self._dual(duals, section, key)
        return float(reduced)

    def _price_voyage_direct(
        self,
        voyage_id: VoyageKey,
        duals: dict[tuple[str, object], float],
        objective_mode: str,
        time_limit: float,
    ) -> dict:
        pricing = self._pricing_models.get(voyage_id)
        if pricing is None:
            pricing = self._build_voyage_pricing_model(voyage_id)
            self._pricing_models[voyage_id] = pricing
        self._set_pricing_objective(pricing, duals, objective_mode)
        pool_size = (
            1
            if objective_mode == "phase_one"
            else int(self.pricing_config.plans_per_pricing)
        )
        self._set_gurobi_param(pricing.model, "SolutionLimit", 2_000_000_000)
        self._set_gurobi_param(pricing.model, "PoolSolutions", pool_size)
        self._set_gurobi_param(
            pricing.model, "PoolSearchMode", 1 if pool_size > 1 else 0
        )
        self._set_gurobi_param(
            pricing.model, "TimeLimit", max(0.01, time_limit)
        )
        started = perf_counter()
        pricing.model.optimize()
        elapsed = perf_counter() - started
        status = self._gurobi_status_name(pricing.model)
        solution_count = self._gurobi_solution_count(pricing.model)
        plans: list[VoyagePlan] = []
        reduced_costs: list[float] = []
        seen: set[tuple] = set()
        for solution_number in range(min(solution_count, pool_size)):
            plan = self._plan_from_solution(pricing, solution_number)
            identity = self._plan_identity(plan)
            if identity in seen:
                continue
            seen.add(identity)
            reduced_cost = self._plan_reduced_cost(
                plan, duals, objective_mode
            )
            expected = pricing.model.getPoolObjective(
                solution_number
            ) - self._dual(duals, "voyage_convexity", voyage_id)
            if abs(expected - reduced_cost) > 1e-6 * (
                1.0 + abs(reduced_cost)
            ):
                raise RuntimeError(
                    "complete-voyage pricing reduced-cost mismatch: "
                    f"voyage={voyage_id}, model={expected}, plan={reduced_cost}"
                )
            plans.append(plan)
            reduced_costs.append(reduced_cost)
        convexity_dual = self._dual(
            duals, "voyage_convexity", voyage_id
        )
        raw_bound = self._gurobi_dual_bound(pricing.model)
        reduced_cost_lower_bound = (
            raw_bound - convexity_dual
            if math.isfinite(raw_bound)
            else -math.inf
        )
        return {
            "voyage_id": voyage_id,
            "status": status,
            "optimal": status == "optimal",
            "plans": plans,
            "reduced_costs": reduced_costs,
            "solution_reduced_cost": (
                min(reduced_costs) if reduced_costs else math.inf
            ),
            "reduced_cost_lower_bound": reduced_cost_lower_bound,
            "seconds": elapsed,
            "candidate_count": len(pricing.candidates),
            "returned_plan_count": len(plans),
        }

    def _local_adjusted_cost(
        self,
        pattern: VoyageAreaLocalPattern,
        outer_duals: dict[tuple[str, object], float],
        objective_mode: str,
    ) -> float:
        cost = (
            0.0
            if objective_mode == "phase_one"
            else float(pattern.business_cost)
        )
        for section, key, coefficient in pattern.master_coefficients:
            cost -= coefficient * self._dual(outer_duals, section, key)
        return float(cost)

    def _set_local_pricing_objective(
        self,
        pricing: _LocalPricingModel,
        outer_duals: dict[tuple[str, object], float],
        inner_duals: dict[tuple[str, object], float],
        objective_mode: str,
    ) -> None:
        for index, variable in pricing.placement_variables.items():
            candidate = pricing.candidates[index]
            coefficient = (
                0.0
                if objective_mode == "phase_one"
                else float(candidate.intrinsic_cost)
            )
            coefficient -= self._dual(
                inner_duals, "group_demand_balance", candidate.group_id
            )
            for section, values in self._placement_master_coefficients(
                candidate
            ).items():
                for key, value in values.items():
                    coefficient -= float(value) * self._dual(
                        outer_duals, section, key
                    )
            pricing.model.setVarObjective(variable, coefficient)
        for variable in pricing.area_use_variables.values():
            pricing.model.setVarObjective(
                variable,
                0.0
                if objective_mode == "phase_one"
                else self._area_activation_penalty(),
            )
        for variable in pricing.row_use_variables.values():
            pricing.model.setVarObjective(
                variable,
                0.0
                if objective_mode == "phase_one"
                else self._row_activation_penalty(),
            )
        pricing.model.update()

    def _price_local_pair(
        self,
        pair: tuple[str, str],
        outer_duals: dict[tuple[str, object], float],
        inner_duals: dict[tuple[str, object], float],
        objective_mode: str,
        time_limit: float,
    ) -> dict:
        pricing = self._local_pricing_models.get(pair)
        if pricing is None:
            pricing = self._build_local_pricing_model(pair)
            self._local_pricing_models[pair] = pricing
        self._set_local_pricing_objective(
            pricing, outer_duals, inner_duals, objective_mode
        )
        pool_size = int(self.pricing_config.plans_per_pricing)
        self._set_gurobi_param(pricing.model, "SolutionLimit", 2_000_000_000)
        self._set_gurobi_param(pricing.model, "PoolSolutions", pool_size)
        self._set_gurobi_param(
            pricing.model, "PoolSearchMode", 1 if pool_size > 1 else 0
        )
        self._set_gurobi_param(
            pricing.model, "TimeLimit", max(0.01, time_limit)
        )
        started = perf_counter()
        pricing.model.optimize()
        elapsed = perf_counter() - started
        status = self._gurobi_status_name(pricing.model)
        solution_count = self._gurobi_solution_count(pricing.model)
        patterns: list[VoyageAreaLocalPattern] = []
        reduced_costs: list[float] = []
        seen: set[tuple] = set()
        convexity_dual = self._dual(
            inner_duals, "area_convexity", pair[1]
        )
        for solution_number in range(min(solution_count, pool_size)):
            pattern = self._local_pattern_from_solution(
                pricing, solution_number
            )
            identity = self._local_pattern_identity(pattern)
            if identity in seen:
                continue
            seen.add(identity)
            reduced = self._local_adjusted_cost(
                pattern, outer_duals, objective_mode
            )
            reduced -= convexity_dual
            for group_id, quantity in pattern.group_quantities:
                reduced -= quantity * self._dual(
                    inner_duals, "group_demand_balance", group_id
                )
            expected = (
                pricing.model.getPoolObjective(solution_number)
                - convexity_dual
            )
            if abs(expected - reduced) > 1e-6 * (1.0 + abs(reduced)):
                raise RuntimeError(
                    "local-pattern reduced-cost mismatch: "
                    f"pair={pair}, model={expected}, pattern={reduced}"
                )
            patterns.append(pattern)
            reduced_costs.append(float(reduced))
        raw_bound = self._gurobi_dual_bound(pricing.model)
        lower_bound = (
            raw_bound - convexity_dual
            if math.isfinite(raw_bound)
            else -math.inf
        )
        return {
            "pair": pair,
            "status": status,
            "optimal": status == "optimal",
            "patterns": patterns,
            "reduced_costs": reduced_costs,
            "solution_reduced_cost": (
                min(reduced_costs) if reduced_costs else math.inf
            ),
            "reduced_cost_lower_bound": float(lower_bound),
            "seconds": elapsed,
            "candidate_count": len(pricing.candidates),
        }

    def _build_inner_master(
        self,
        voyage_id: str,
        outer_duals: dict[tuple[str, object], float],
        objective_mode: str,
        integer: bool,
    ):
        from gurobipy import quicksum

        model = GurobiModel(
            f"inner_voyage_{'mip' if integer else 'lp'}_"
            f"{self._key_name((voyage_id,))}"
        )
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        if not integer:
            self._set_gurobi_param(model, "Method", int(self.config.lp_method))
        model.setMinimize()
        pairs = tuple(
            pair for pair in self._local_candidates if pair[0] == voyage_id
        )
        variables: dict[tuple[tuple[str, str], int], object] = {}
        for pair in pairs:
            for index, pattern in enumerate(self._local_patterns[pair]):
                variables[(pair, index)] = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    vtype="B" if integer else "C",
                    obj=self._local_adjusted_cost(
                        pattern, outer_duals, objective_mode
                    ),
                    name=(
                        f"local_pattern_{self._key_name(pair)}_{index}"
                    ),
                )
        constraints: dict[str, dict] = defaultdict(dict)
        for pair in pairs:
            constraints["area_convexity"][pair[1]] = model.addConstr(
                quicksum(
                    variables[(pair, index)]
                    for index in range(len(self._local_patterns[pair]))
                )
                == 1.0,
                name=f"inner_area_{self._key_name(pair)}",
            )
        for group in self.groups:
            if group.voyage_id != voyage_id:
                continue
            terms = []
            for pair in pairs:
                for index, pattern in enumerate(self._local_patterns[pair]):
                    quantity = dict(pattern.group_quantities).get(
                        group.group_id, 0
                    )
                    if quantity:
                        terms.append(quantity * variables[(pair, index)])
            constraints["group_demand_balance"][group.group_id] = (
                model.addConstr(
                    quicksum(terms) == int(group.demand),
                    name=f"inner_group_{group.group_id}",
                )
            )
        group_count = len(
            {
                self._operational_group_key(group)
                for group in self.groups
                if group.voyage_id == voyage_id
            }
        )
        offset = -group_count * (
            self._area_activation_penalty()
            + self._row_activation_penalty()
        )
        model.addVar(
            lb=1.0,
            ub=1.0,
            obj=0.0 if objective_mode == "phase_one" else offset,
            name=f"inner_offset_{self._key_name((voyage_id,))}",
        )
        model.update()
        return model, variables, constraints, pairs

    def _add_local_patterns_to_inner_master(
        self,
        model,
        variables: dict,
        constraints: dict,
        pair: tuple[str, str],
        indices: list[int],
        outer_duals: dict[tuple[str, object], float],
        objective_mode: str,
    ) -> None:
        for index in indices:
            pattern = self._local_patterns[pair][index]
            terms = [(1.0, constraints["area_convexity"][pair[1]])]
            terms.extend(
                (
                    float(quantity),
                    constraints["group_demand_balance"][group_id],
                )
                for group_id, quantity in pattern.group_quantities
            )
            variables[(pair, index)] = model.addPricedVar(
                terms,
                lb=0.0,
                ub=1.0,
                vtype="C",
                obj=self._local_adjusted_cost(
                    pattern, outer_duals, objective_mode
                ),
                name=f"local_pattern_{self._key_name(pair)}_{index}",
            )
        model.update()

    def _solve_inner_relaxation(
        self,
        voyage_id: str,
        outer_duals: dict[tuple[str, object], float],
        objective_mode: str,
        deadline: float,
    ) -> dict:
        model, variables, constraints, pairs = self._build_inner_master(
            voyage_id, outer_duals, objective_mode, integer=False
        )
        records: list[dict] = []
        best_valid_lower_bound = -math.inf
        last_objective = math.inf
        exact = False
        try:
            for iteration in range(1, int(self.config.max_iterations) + 1):
                if not self._set_remaining_time_limit(model, deadline):
                    break
                model.optimize()
                if self._gurobi_status_name(model) != "optimal":
                    break
                last_objective = self._gurobi_objective_value(model)
                inner_duals = self._master_dual_snapshot(model, constraints)
                pair_order = tuple(
                    sorted(
                        pairs,
                        key=lambda pair: -len(self._local_candidates[pair]),
                    )
                )
                remaining_weight = sum(
                    math.sqrt(max(1, len(self._local_candidates[pair])))
                    for pair in pair_order
                )
                local_results: list[dict] = []
                added = 0
                for pair in pair_order:
                    remaining = self._seconds_until(deadline)
                    if remaining is not None and remaining <= 1e-6:
                        break
                    weight = math.sqrt(
                        max(1, len(self._local_candidates[pair]))
                    )
                    time_limit = max(
                        0.01, float(remaining) * weight / remaining_weight
                    )
                    result = self._price_local_pair(
                        pair,
                        outer_duals,
                        inner_duals,
                        objective_mode,
                        time_limit,
                    )
                    local_results.append(result)
                    remaining_weight -= weight
                    new_pair_indices: list[int] = []
                    for pattern, reduced in zip(
                        result["patterns"],
                        result["reduced_costs"],
                        strict=True,
                    ):
                        if float(reduced) >= -max(
                            1e-9, float(self.config.reduced_cost_tolerance)
                        ):
                            continue
                        if not self._append_local_pattern(pattern):
                            continue
                        new_pair_indices.append(
                            len(self._local_patterns[pair]) - 1
                        )
                    if new_pair_indices:
                        self._add_local_patterns_to_inner_master(
                            model,
                            variables,
                            constraints,
                            pair,
                            new_pair_indices,
                            outer_duals,
                            objective_mode,
                        )
                        added += len(new_pair_indices)
                complete = len(local_results) == len(pairs)
                finite = complete and all(
                    math.isfinite(
                        float(result["reduced_cost_lower_bound"])
                    )
                    for result in local_results
                )
                correction = (
                    sum(
                        min(
                            0.0,
                            float(result["reduced_cost_lower_bound"]),
                        )
                        for result in local_results
                    )
                    if finite
                    else None
                )
                if correction is not None:
                    best_valid_lower_bound = max(
                        best_valid_lower_bound,
                        last_objective + correction,
                    )
                exact_sweep = complete and all(
                    bool(result["optimal"]) for result in local_results
                )
                records.append(
                    {
                        "iteration": iteration,
                        "restricted_objective": last_objective,
                        "priced_area_count": len(local_results),
                        "area_count": len(pairs),
                        "new_local_patterns": added,
                        "exact_sweep": exact_sweep,
                        "valid_lower_bound_correction": correction,
                        "pricing_seconds": sum(
                            float(result["seconds"])
                            for result in local_results
                        ),
                    }
                )
                if added:
                    continue
                if exact_sweep:
                    exact = True
                    best_valid_lower_bound = last_objective
                break
            return {
                "exact": exact,
                "objective": last_objective,
                "valid_lower_bound": best_valid_lower_bound,
                "records": records,
                "area_count": len(pairs),
            }
        finally:
            self._free_gurobi_model(model)

    def _plan_from_local_selection(
        self,
        voyage_id: str,
        patterns: list[VoyageAreaLocalPattern],
    ) -> VoyagePlan:
        placements = tuple(
            sorted(
                (
                    placement
                    for pattern in patterns
                    for placement in pattern.placements
                ),
                key=lambda placement: (
                    placement.group_id,
                    placement.area_no,
                    self.bays[placement.bay_key].bay_order,
                    placement.row_allocation,
                ),
            )
        )
        quantities: Counter[str] = Counter()
        for placement in placements:
            quantities[placement.group_id] += int(placement.quantity)
        for group in self.groups:
            if group.voyage_id != voyage_id:
                continue
            if quantities[group.group_id] != int(group.demand):
                raise RuntimeError(
                    "inner local-pattern master returned incomplete voyage: "
                    f"voyage={voyage_id}, group={group.group_id}"
                )
        group_count = len(
            {
                self._operational_group_key(group)
                for group in self.groups
                if group.voyage_id == voyage_id
            }
        )
        business_cost = sum(pattern.business_cost for pattern in patterns)
        business_cost -= group_count * (
            self._area_activation_penalty()
            + self._row_activation_penalty()
        )
        return VoyagePlan(
            plan_id="",
            voyage_id=voyage_id,
            placements=placements,
            business_cost=float(business_cost),
            master_coefficients=self._aggregate_master_coefficients(placements),
        )

    def _solve_inner_integer_master(
        self,
        voyage_id: str,
        outer_duals: dict[tuple[str, object], float],
        objective_mode: str,
        time_limit: float,
    ) -> dict:
        model, variables, _constraints, pairs = self._build_inner_master(
            voyage_id, outer_duals, objective_mode, integer=True
        )
        try:
            self._set_gurobi_param(model, "TimeLimit", max(0.01, time_limit))
            self._set_gurobi_param(model, "MIPGap", 0.0)
            pool_size = int(self.pricing_config.plans_per_pricing)
            self._set_gurobi_param(model, "PoolSolutions", pool_size)
            self._set_gurobi_param(
                model, "PoolSearchMode", 1 if pool_size > 1 else 0
            )
            model.optimize()
            status = self._gurobi_status_name(model)
            if self._gurobi_solution_count(model) <= 0:
                return {"status": status, "plans": []}
            plans: list[VoyagePlan] = []
            identities: set[tuple] = set()
            solution_count = min(
                self._gurobi_solution_count(model), pool_size
            )
            for solution_number in range(solution_count):
                selected_patterns = [
                    self._local_patterns[pair][index]
                    for (pair, index), variable in variables.items()
                    if model.getPoolValue(variable, solution_number) > 0.5
                ]
                if len(selected_patterns) != len(pairs):
                    continue
                plan = self._plan_from_local_selection(
                    voyage_id, selected_patterns
                )
                identity = self._plan_identity(plan)
                if identity in identities:
                    continue
                identities.add(identity)
                plans.append(plan)
            return {
                "status": status,
                "plans": plans,
                "objective": self._gurobi_objective_value(model),
                "bound": self._gurobi_dual_bound(model),
                "gap": self._gurobi_gap(model),
            }
        finally:
            self._free_gurobi_model(model)

    def _price_voyage(
        self,
        voyage_id: VoyageKey,
        duals: dict[tuple[str, object], float],
        objective_mode: str,
        time_limit: float,
    ) -> dict:
        started = perf_counter()
        total_limit = max(0.02, float(time_limit))
        nested_limit = max(0.02, 0.60 * total_limit)
        deadline = started + nested_limit
        integer_reserve = max(0.05, min(1.0, 0.20 * nested_limit))
        relaxation_deadline = max(started + 0.01, deadline - integer_reserve)
        relaxation = self._solve_inner_relaxation(
            voyage_id,
            duals,
            objective_mode,
            relaxation_deadline,
        )
        remaining = max(0.01, deadline - perf_counter())
        integer = self._solve_inner_integer_master(
            voyage_id, duals, objective_mode, remaining
        )
        plans = list(integer.get("plans", []))
        convexity_dual = self._dual(
            duals, "voyage_convexity", voyage_id
        )
        reduced_costs = [
            self._plan_reduced_cost(candidate, duals, objective_mode)
            for candidate in plans
        ]
        inner_lower = float(relaxation.get("valid_lower_bound", -math.inf))
        reduced_lower = (
            inner_lower - convexity_dual
            if math.isfinite(inner_lower)
            else -math.inf
        )
        tolerance = max(1e-9, float(self.config.reduced_cost_tolerance))
        restricted_objective = float(integer.get("objective", math.inf))
        inner_exact_integer = (
            bool(relaxation.get("exact"))
            and math.isfinite(restricted_objective)
            and abs(restricted_objective - float(relaxation["objective"]))
            <= tolerance * (1.0 + abs(restricted_objective))
        )
        certified = reduced_lower >= -tolerance or inner_exact_integer
        result = {
            "voyage_id": voyage_id,
            "status": "optimal" if certified else "bounded",
            "optimal": certified,
            "plans": plans,
            "reduced_costs": reduced_costs,
            "solution_reduced_cost": (
                min(reduced_costs) if reduced_costs else math.inf
            ),
            "reduced_cost_lower_bound": reduced_lower,
            "seconds": perf_counter() - started,
            "candidate_count": len(self._voyage_candidates[voyage_id]),
            "returned_plan_count": len(plans),
            "inner_relaxation_exact": bool(relaxation.get("exact")),
            "inner_integer_status": integer.get("status"),
            "inner_integer_gap": integer.get("gap"),
            "inner_area_count": int(relaxation.get("area_count", 0)),
            "inner_iteration_count": len(relaxation.get("records", [])),
            "inner_records": relaxation.get("records", []),
        }
        has_novel_negative_plan = any(
            float(value) < -tolerance
            and self._plan_identity(candidate)
            not in self._plan_index_by_identity
            for candidate, value in zip(plans, reduced_costs, strict=True)
        )
        best_reduced = min(reduced_costs, default=math.inf)
        weak_nested_incumbent = (
            has_novel_negative_plan
            and best_reduced < -tolerance
            and reduced_lower < -tolerance
            and abs(best_reduced) <= 0.5 * abs(reduced_lower)
        )
        if certified or (
            has_novel_negative_plan and not weak_nested_incumbent
        ):
            return result

        direct_remaining = max(
            0.01, started + total_limit - perf_counter()
        )
        direct = self._price_voyage_direct(
            voyage_id,
            duals,
            objective_mode,
            direct_remaining,
        )
        combined: dict[tuple, tuple[VoyagePlan, float]] = {}
        for candidate, reduced in zip(plans, reduced_costs, strict=True):
            combined[self._plan_identity(candidate)] = (
                candidate,
                float(reduced),
            )
        for candidate, reduced in zip(
            direct["plans"], direct["reduced_costs"], strict=True
        ):
            identity = self._plan_identity(candidate)
            previous = combined.get(identity)
            if previous is None or float(reduced) < float(previous[1]):
                combined[identity] = (candidate, float(reduced))
        ordered = sorted(combined.values(), key=lambda item: float(item[1]))
        result["plans"] = [item[0] for item in ordered]
        result["reduced_costs"] = [float(item[1]) for item in ordered]
        result["solution_reduced_cost"] = (
            float(ordered[0][1]) if ordered else math.inf
        )
        direct_lower = float(direct["reduced_cost_lower_bound"])
        if math.isfinite(direct_lower):
            result["reduced_cost_lower_bound"] = max(
                float(result["reduced_cost_lower_bound"]), direct_lower
            )
        result["optimal"] = bool(direct["optimal"])
        result["status"] = str(direct["status"])
        result["seconds"] = perf_counter() - started
        result["exact_certification_used"] = True
        result["exact_certification_seconds"] = float(direct["seconds"])
        return result

    def _generate_initial_plans(
        self,
        voyages: tuple[VoyageKey, ...],
        deadline: float | None,
    ) -> None:
        remaining_count = len(voyages)
        for voyage_id in sorted(
            voyages,
            key=lambda value: -len(self._voyage_candidates[value]),
        ):
            remaining = self._seconds_until(deadline)
            if remaining is not None and remaining <= 1e-6:
                raise RuntimeError(
                    "time limit expired while constructing initial voyage plans"
                )
            pricing = self._pricing_models.get(voyage_id)
            if pricing is None:
                pricing = self._build_voyage_pricing_model(voyage_id)
                self._pricing_models[voyage_id] = pricing
            for variable in pricing.model.getVars():
                pricing.model.setVarObjective(variable, 0.0)
            self._set_gurobi_param(pricing.model, "SolutionLimit", 1)
            self._set_gurobi_param(pricing.model, "PoolSearchMode", 0)
            time_limit = (
                max(0.05, min(5.0, remaining / remaining_count))
                if remaining is not None
                else max(1.0, float(self.config.mip_time_limit))
            )
            self._set_gurobi_param(pricing.model, "TimeLimit", time_limit)
            pricing.model.update()
            started = perf_counter()
            pricing.model.optimize()
            elapsed = perf_counter() - started
            status = self._gurobi_status_name(pricing.model)
            if self._gurobi_solution_count(pricing.model) <= 0:
                raise RuntimeError(
                    "cannot construct a complete feasible plan for voyage: "
                    f"voyage={voyage_id}, status={status}"
                )
            plan = self._plan_from_solution(pricing)
            self._append_plan(plan)
            self._split_plan_into_local_patterns(plan)
            self._initialization_records.append(
                {
                    "voyage_id": voyage_id,
                    "status": status,
                    "candidate_count": len(pricing.candidates),
                    "seconds": elapsed,
                }
            )
            remaining_count -= 1
        self._seed_model_build_seconds = sum(
            pricing.build_seconds for pricing in self._pricing_models.values()
        )

    def _plan_terms(
        self, plan: VoyagePlan, constraints: dict
    ) -> list[tuple[float, object]]:
        terms: list[tuple[float, object]] = [
            (1.0, constraints["voyage_convexity"][plan.voyage_id])
        ]
        for section, key, coefficient in plan.master_coefficients:
            row = constraints.get(section, {}).get(key)
            if row is not None:
                terms.append((float(coefficient), row))
        return terms

    def _plan_coefficient_rows(self):
        rows: defaultdict[
            str, defaultdict[object, list[tuple[int, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        for index, plan in enumerate(self._plans):
            for section, key, coefficient in plan.master_coefficients:
                rows[section][key].append((index, float(coefficient)))
        return rows

    def _phase_artificial(
        self,
        model,
        variables: dict,
        family: str,
        key: object,
        upper: float | None = None,
    ):
        kwargs = {
            "lb": 0.0,
            "name": f"phase_{family}_{self._key_name(key if isinstance(key, tuple) else (key,))}",
        }
        if upper is not None:
            kwargs["ub"] = max(0.0, float(upper))
        variable = model.addVar(**kwargs)
        variables["phase_one_artificial"][(family, key)] = variable
        return variable

    def _add_master_compatibility_with_phase_slacks(
        self,
        quicksum,
        model,
        plan_variables: dict[int, object],
        coefficient_rows,
        variables: dict,
        constraints: dict,
    ) -> None:
        bay_uses_by_scope: defaultdict[tuple[str, str, str], list] = (
            defaultdict(list)
        )
        for key in sorted(self._master_bay_attr_choice_keys):
            bay_key, attr, scope, _value = key
            use = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="C",
                name=f"bay_use_{self._key_name(key)}",
            )
            items = coefficient_rows["bay_attr_link"].get(key, [])
            artificial = self._phase_artificial(
                model,
                variables,
                "bay_attr_link",
                key,
            )
            constraints["bay_attr_link"][key] = model.addConstr(
                quicksum(
                    coefficient * plan_variables[index]
                    for index, coefficient in items
                )
                <= self._master_bay_attr_big_m[key] * use + artificial,
                name=f"bay_attr_link_{self._key_name(key)}",
            )
            bay_uses_by_scope[(bay_key, attr, scope)].append(use)
        for key, uses in sorted(bay_uses_by_scope.items()):
            artificial = self._phase_artificial(
                model, variables, "bay_attr_one", key, len(uses)
            )
            constraints["bay_attr_one"][key] = model.addConstr(
                quicksum(uses) <= 1.0 + artificial,
                name=f"bay_attr_one_{self._key_name(key)}",
            )

        row_uses_by_scope: defaultdict[tuple[str, str, str, str], list] = (
            defaultdict(list)
        )
        for key in sorted(self._master_row_attr_choice_keys):
            bay_key, row_no, attr, scope, _value = key
            use = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="C",
                name=f"row_use_{self._key_name(key)}",
            )
            items = coefficient_rows["row_attr_link"].get(key, [])
            artificial = self._phase_artificial(
                model,
                variables,
                "row_attr_link",
                key,
            )
            constraints["row_attr_link"][key] = model.addConstr(
                quicksum(
                    coefficient * plan_variables[index]
                    for index, coefficient in items
                )
                <= self._master_row_attr_big_m[key] * use + artificial,
                name=f"row_attr_link_{self._key_name(key)}",
            )
            row_uses_by_scope[(bay_key, row_no, attr, scope)].append(use)
        for key, uses in sorted(row_uses_by_scope.items()):
            artificial = self._phase_artificial(
                model, variables, "row_attr_one", key, len(uses)
            )
            constraints["row_attr_one"][key] = model.addConstr(
                quicksum(uses) <= 1.0 + artificial,
                name=f"row_attr_one_{self._key_name(key)}",
            )

    def _build_plan_master(self, voyages: tuple[VoyageKey, ...]):
        from gurobipy import quicksum

        model = GurobiModel("complete_voyage_plan_root_master")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        self._set_gurobi_param(model, "Method", int(self.config.lp_method))
        model.setMinimize()
        variables: dict[str, dict] = {
            "plan": {},
            "phase_one_artificial": {},
        }
        for index, plan in enumerate(self._plans):
            variables["plan"][index] = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="C",
                obj=float(plan.business_cost),
                name=f"voyage_plan_{index}",
            )
        import_reserve = {
            (flow, size, bay_key): model.addVar(
                lb=0.0,
                ub=float(capacity),
                vtype="C",
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
        variables["import_reserve"] = import_reserve
        coefficient_rows = self._plan_coefficient_rows()
        constraints: dict[str, dict] = defaultdict(dict)

        for voyage_id in voyages:
            indices = [
                index
                for index, plan in enumerate(self._plans)
                if plan.voyage_id == voyage_id
            ]
            if not indices:
                raise RuntimeError(
                    f"voyage has no initial complete plan: {voyage_id}"
                )
            constraints["voyage_convexity"][voyage_id] = model.addConstr(
                quicksum(variables["plan"][index] for index in indices) == 1.0,
                name=f"voyage_convexity_{self._key_name((voyage_id,))}",
            )

        import_by_bay: defaultdict[str, list] = defaultdict(list)
        import_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        import_by_flow_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        import_by_flow_area_size: defaultdict[
            tuple[str, str, str], list
        ] = defaultdict(list)
        for (flow, size, bay_key), variable in import_reserve.items():
            area_no = self.bays[bay_key].area_no
            for footprint_key in self._placement_footprint_keys(bay_key, size):
                import_by_bay[footprint_key].append(variable)
            import_by_bay_size[(bay_key, size)].append(variable)
            import_by_flow_size[(flow, size)].append(variable)
            import_by_flow_area_size[(flow, area_no, size)].append(variable)

        for bay_key in sorted(self._master_bay_capacity_keys):
            items = coefficient_rows["bay_capacity_limit"].get(bay_key, [])
            artificial = self._phase_artificial(
                model,
                variables,
                "bay_capacity_limit",
                bay_key,
            )
            constraints["bay_capacity_limit"][bay_key] = model.addConstr(
                quicksum(
                    coefficient * variables["plan"][index]
                    for index, coefficient in items
                )
                + quicksum(import_by_bay.get(bay_key, []))
                <= int(self.bays[bay_key].physical_capacity) + artificial,
                name=f"bay_cap_{self._key_name((bay_key,))}",
            )
        for key in sorted(self._master_bay_size_keys):
            bay_key, size = key
            items = coefficient_rows["bay_size_limit"].get(key, [])
            capacity = int(self.bays[bay_key].cap_by_size.get(size, 0))
            artificial = self._phase_artificial(
                model, variables, "bay_size_limit", key
            )
            constraints["bay_size_limit"][key] = model.addConstr(
                quicksum(
                    coefficient * variables["plan"][index]
                    for index, coefficient in items
                )
                + quicksum(import_by_bay_size.get(key, []))
                <= capacity + artificial,
                name=f"bay_size_{self._key_name(key)}",
            )
        for key in sorted(self._master_row_capacity_keys):
            bay_key, row_no = key
            items = coefficient_rows["row_capacity_limit"].get(key, [])
            capacity = int(
                self.bays[bay_key].row_physical_capacity.get(
                    row_no, self.bays[bay_key].physical_capacity
                )
            )
            artificial = self._phase_artificial(
                model, variables, "row_capacity_limit", key
            )
            constraints["row_capacity_limit"][key] = model.addConstr(
                quicksum(
                    coefficient * variables["plan"][index]
                    for index, coefficient in items
                )
                <= capacity + artificial,
                name=f"row_cap_{self._key_name(key)}",
            )
        for key in sorted(self._master_row_size_keys):
            bay_key, row_no, size = key
            items = coefficient_rows["row_size_limit"].get(key, [])
            capacity = int(
                self.bays[bay_key].row_cap_by_size.get(size, {}).get(
                    row_no,
                    self.bays[bay_key].cap_by_size.get(size, 0),
                )
            )
            artificial = self._phase_artificial(
                model, variables, "row_size_limit", key
            )
            constraints["row_size_limit"][key] = model.addConstr(
                quicksum(
                    coefficient * variables["plan"][index]
                    for index, coefficient in items
                )
                <= capacity + artificial,
                name=f"row_size_{self._key_name(key)}",
            )

        stacks_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        for key in sorted(self._master_stack_keys):
            bay_key, _mix_key, size = key
            group = self.groups_by_id.get(
                self._master_stack_sample_group.get(key, "")
            )
            if group is None:
                continue
            stack_count = self._stack_count_for_group(bay_key, size, group)
            unit_capacity = self._stack_unit_capacity_for_group(
                bay_key, size, group
            )
            if stack_count <= 0 or unit_capacity <= 0:
                continue
            stack = model.addVar(
                lb=0.0,
                ub=float(stack_count),
                vtype="C",
                name=f"stack_{self._key_name(key)}",
            )
            items = coefficient_rows["bay_port_stack_link"].get(key, [])
            artificial = self._phase_artificial(
                model,
                variables,
                "bay_port_stack_link",
                key,
            )
            constraints["bay_port_stack_link"][key] = model.addConstr(
                quicksum(
                    coefficient * variables["plan"][index]
                    for index, coefficient in items
                )
                <= unit_capacity * stack + artificial,
                name=f"stack_load_{self._key_name(key)}",
            )
            stacks_by_bay_size[(bay_key, size)].append(stack)
        for key, stacks in sorted(stacks_by_bay_size.items()):
            capacity = self._stack_count_for_bay_size(*key)
            artificial = self._phase_artificial(
                model, variables, "bay_stack_total_limit", key
            )
            constraints["bay_stack_total_limit"][key] = model.addConstr(
                quicksum(stacks) <= capacity + artificial,
                name=f"stack_total_{self._key_name(key)}",
            )

        for key, required in sorted(self.import_total_by_flow_size.items()):
            artificial = self._phase_artificial(
                model, variables, "import_total_balance", key, required
            )
            constraints["import_total_balance"][key] = model.addConstr(
                quicksum(import_by_flow_size.get(key, [])) + artificial
                == int(required),
                name=f"import_total_{self._key_name(key)}",
            )

        for key in sorted(self._master_area_guidance_keys):
            target = self._area_size_target(*key)
            positive = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"guide_pos_{self._key_name(key)}",
            )
            negative = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"guide_neg_{self._key_name(key)}",
            )
            items = coefficient_rows["area_guidance_balance"].get(key, [])
            constraints["area_guidance_balance"][key] = model.addConstr(
                quicksum(
                    coefficient * variables["plan"][index]
                    for index, coefficient in items
                )
                - target
                == positive - negative,
                name=f"guide_balance_{self._key_name(key)}",
            )

        constraints["import_reference_balance"] = (
            self._add_import_reference_deviation(
                quicksum,
                model,
                import_by_flow_area_size,
                objective_mode="full",
            )
        )
        self._add_master_compatibility_with_phase_slacks(
            quicksum,
            model,
            variables["plan"],
            coefficient_rows,
            variables,
            constraints,
        )
        model.update()
        self._initialize_phase_one_objective(model, variables)
        return model, variables, constraints

    def _add_plans_to_master(
        self,
        model,
        variables: dict,
        constraints: dict,
        indices: list[int],
        objective_mode: str,
    ) -> None:
        for index in indices:
            plan = self._plans[index]
            variable = model.addPricedVar(
                self._plan_terms(plan, constraints),
                lb=0.0,
                ub=1.0,
                vtype="C",
                obj=(
                    0.0
                    if objective_mode == "phase_one"
                    else float(plan.business_cost)
                ),
                name=f"voyage_plan_{index}",
            )
            variables["plan"][index] = variable
            if objective_mode == "phase_one":
                variables.setdefault("_business_objective_terms", []).append(
                    (variable, float(plan.business_cost))
                )
        model.update()

    def _price_all_voyages(
        self,
        voyages: tuple[VoyageKey, ...],
        duals: dict[tuple[str, object], float],
        objective_mode: str,
        deadline: float | None,
    ) -> tuple[dict, list[int]]:
        tolerance = max(1e-9, float(self.config.reduced_cost_tolerance))
        results: list[dict] = []
        new_indices: list[int] = []
        pricing_order = tuple(
            sorted(
                voyages,
                key=lambda voyage_id: (
                    -len(self._voyage_candidates[voyage_id]),
                    voyage_id,
                ),
            )
        )
        remaining_weight = sum(
            math.sqrt(max(1, len(self._voyage_candidates[voyage_id])))
            for voyage_id in pricing_order
        )
        for voyage_id in pricing_order:
            remaining = self._seconds_until(deadline)
            if remaining is not None and remaining <= 1e-6:
                break
            voyage_weight = math.sqrt(
                max(1, len(self._voyage_candidates[voyage_id]))
            )
            time_limit = (
                max(0.01, remaining * voyage_weight / remaining_weight)
                if remaining is not None
                else max(1.0, float(self.config.mip_time_limit))
            )
            result = self._price_voyage(
                voyage_id, duals, objective_mode, time_limit
            )
            results.append(result)
            remaining_weight -= voyage_weight
            for plan, reduced_cost in zip(
                result["plans"], result["reduced_costs"], strict=True
            ):
                if float(reduced_cost) >= -tolerance:
                    continue
                identity = self._plan_identity(plan)
                if identity in self._plan_index_by_identity:
                    continue
                new_indices.append(self._append_plan(plan))
                self._split_plan_into_local_patterns(plan)
        complete = len(results) == len(voyages)
        finite_bounds = complete and all(
            math.isfinite(float(result["reduced_cost_lower_bound"]))
            for result in results
        )
        exact = complete and all(bool(result["optimal"]) for result in results)
        correction = (
            sum(
                min(0.0, float(result["reduced_cost_lower_bound"]))
                for result in results
            )
            if finite_bounds
            else None
        )
        return {
            "voyage_count": len(voyages),
            "priced_voyage_count": len(results),
            "exact": exact,
            "valid_lower_bound_correction": correction,
            "new_plans": len(new_indices),
            "minimum_solution_reduced_cost": min(
                (
                    float(result["solution_reduced_cost"])
                    for result in results
                ),
                default=math.inf,
            ),
            "pricing_seconds": sum(
                float(result["seconds"]) for result in results
            ),
            "pricing_results": [
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"plans", "reduced_costs"}
                }
                for result in results
            ],
        }, new_indices

    def _solve_root_lp(
        self,
        voyages: tuple[VoyageKey, ...],
        deadline: float | None,
    ) -> dict:
        from gurobipy import quicksum

        started = perf_counter()
        model, variables, constraints = self._build_plan_master(voyages)
        records: list[dict] = []
        phase_iterations = 0
        business_iterations = 0
        best_valid_lower_bound = 0.0
        last_plan_values: dict[int, float] = {}
        try:
            phase_complete = False
            for iteration in range(1, int(self.config.max_iterations) + 1):
                if not self._set_remaining_time_limit(model, deadline):
                    break
                model.optimize()
                if self._gurobi_status_name(model) != "optimal":
                    break
                phase_iterations = iteration
                artificial = sum(
                    self._gurobi_value(model, variable)
                    for variable in variables["phase_one_artificial"].values()
                )
                if artificial <= 1e-7:
                    phase_complete = True
                    break
                duals = self._master_dual_snapshot(model, constraints)
                pricing, new_indices = self._price_all_voyages(
                    voyages, duals, "phase_one", deadline
                )
                pricing.update(
                    {
                        "stage": "phase_one",
                        "iteration": iteration,
                        "restricted_master_objective": artificial,
                    }
                )
                records.append(pricing)
                if new_indices:
                    self._add_plans_to_master(
                        model,
                        variables,
                        constraints,
                        new_indices,
                        "phase_one",
                    )
                    continue
                if pricing["exact"]:
                    return {
                        "status": "infeasible",
                        "valid_lower_bound": math.inf,
                        "phase_iterations": phase_iterations,
                        "business_iterations": 0,
                        "records": records,
                        "plan_values": {},
                        "seconds": perf_counter() - started,
                    }
                break
            if not phase_complete:
                return {
                    "status": "time_limit",
                    "valid_lower_bound": best_valid_lower_bound,
                    "phase_iterations": phase_iterations,
                    "business_iterations": 0,
                    "records": records,
                    "plan_values": last_plan_values,
                    "seconds": perf_counter() - started,
                }

            constraints["phase_one_zero"][0] = model.addConstr(
                quicksum(variables["phase_one_artificial"].values()) == 0.0,
                name="complete_voyage_phase_one_zero",
            )
            self._activate_master_objective(
                model, variables, objective="business"
            )
            for iteration in range(1, int(self.config.max_iterations) + 1):
                if not self._set_remaining_time_limit(model, deadline):
                    break
                model.optimize()
                if self._gurobi_status_name(model) != "optimal":
                    break
                business_iterations = iteration
                restricted_objective = self._gurobi_objective_value(model)
                last_plan_values = {
                    int(index): self._gurobi_value(model, variable)
                    for index, variable in variables["plan"].items()
                    if self._gurobi_value(model, variable) > 1e-9
                }
                duals = self._master_dual_snapshot(model, constraints)
                pricing, new_indices = self._price_all_voyages(
                    voyages, duals, "business", deadline
                )
                pricing["dual_source"] = "raw"
                pricing.update(
                    {
                        "stage": "business",
                        "iteration": iteration,
                        "restricted_master_objective": restricted_objective,
                    }
                )
                records.append(pricing)
                correction = pricing.get("valid_lower_bound_correction")
                if correction is not None:
                    best_valid_lower_bound = max(
                        best_valid_lower_bound,
                        restricted_objective + float(correction),
                    )
                if new_indices:
                    self._add_plans_to_master(
                        model,
                        variables,
                        constraints,
                        new_indices,
                        "business",
                    )
                    continue
                if pricing["exact"]:
                    return {
                        "status": "optimal",
                        "bound": restricted_objective,
                        "valid_lower_bound": restricted_objective,
                        "phase_iterations": phase_iterations,
                        "business_iterations": business_iterations,
                        "records": records,
                        "plan_values": last_plan_values,
                        "seconds": perf_counter() - started,
                    }
                break
            return {
                "status": "time_limit",
                "valid_lower_bound": best_valid_lower_bound,
                "phase_iterations": phase_iterations,
                "business_iterations": business_iterations,
                "records": records,
                "plan_values": last_plan_values,
                "seconds": perf_counter() - started,
            }
        finally:
            self._free_gurobi_model(model)

    def _recovery_pool(
        self, active_plan_indices: set[int] | None = None
    ) -> list[PlacementColumn]:
        selected: dict[tuple, PlacementColumn] = {}
        plan_indices = set(active_plan_indices or ())
        recent_by_voyage: Counter[str] = Counter()
        for index in range(len(self._plans) - 1, -1, -1):
            plan = self._plans[index]
            if recent_by_voyage[plan.voyage_id] >= 8:
                continue
            plan_indices.add(index)
            recent_by_voyage[plan.voyage_id] += 1
        if not plan_indices:
            plan_indices.update(range(len(self._plans)))
        for index in sorted(plan_indices):
            for placement in self._plans[index].placements:
                key = self._base_placement_static_key(placement)
                base = self._base_candidate_by_static_key.get(key)
                if base is not None:
                    selected[key] = base
        for group in self.groups:
            candidates = sorted(
                self._base_placements_for_group(group),
                key=lambda candidate: (
                    float(candidate.intrinsic_cost),
                    candidate.area_no,
                    self.bays[candidate.bay_key].bay_order,
                    candidate.row_allocation,
                ),
            )
            covered = sum(
                self._base_location_capacity(group, candidate)
                for candidate in candidates
                if self._base_placement_static_key(candidate) in selected
            )
            for candidate in candidates:
                if covered >= int(group.demand):
                    break
                key = self._base_placement_static_key(candidate)
                if key in selected:
                    continue
                selected[key] = candidate
                covered += self._base_location_capacity(group, candidate)
        return sorted(
            selected.values(),
            key=lambda candidate: (
                candidate.group_id,
                candidate.area_no,
                self.bays[candidate.bay_key].bay_order,
                candidate.row_allocation,
            ),
        )

    def _solve_restricted_row_milp(
        self,
        time_limit: float,
        active_plan_indices: set[int] | None = None,
    ) -> dict:
        from gurobipy import quicksum

        if time_limit <= 1e-6:
            return {"status": "not_run", "selected": None, "seconds": 0.0}
        started = perf_counter()
        recovery_pool = self._recovery_pool(active_plan_indices)
        self._initialize_location_pool()
        for candidate in recovery_pool:
            self._append_generated_column(candidate)
        model, variables, model_stats = self.build_compact_row_milp(
            self._columns,
            GurobiModel,
            quicksum,
        )
        try:
            build_seconds = perf_counter() - started
            remaining = max(0.01, time_limit - build_seconds)
            self._set_gurobi_param(model, "TimeLimit", remaining)
            self._set_gurobi_param(model, "MIPGap", 0.0)
            model.optimize()
            status = self._gurobi_status_name(model)
            if self._gurobi_solution_count(model) <= 0:
                return {
                    "status": status,
                    "selected": None,
                    "seconds": perf_counter() - started,
                    "candidate_count": len(self._columns),
                    "model": model_stats,
                }
            selected = self.selected_compact_row_values(model, variables)
            self._final_import_reservation = (
                self._gurobi_import_reservation_values(model, variables)
            )
            return {
                "status": status,
                "selected": selected,
                "objective": self._gurobi_objective_value(model),
                "restricted_bound": self._gurobi_dual_bound(model),
                "restricted_gap": self._gurobi_gap(model),
                "seconds": perf_counter() - started,
                "candidate_count": len(self._columns),
                "model_build_seconds": build_seconds,
                "model": model_stats,
            }
        finally:
            self._free_gurobi_model(model)

    def _dispose_pricing_models(self) -> None:
        for pricing in self._pricing_models.values():
            self._free_gurobi_model(pricing.model)
        self._pricing_models.clear()
        for pricing in self._local_pricing_models.values():
            self._free_gurobi_model(pricing.model)
        self._local_pricing_models.clear()

    def solve_root(self) -> dict:
        started = perf_counter()
        total_limit = float(self.config.total_time_limit)
        deadline = (
            None if total_limit <= 0.0 else started + max(0.01, total_limit)
        )
        voyages = self._initialize_candidate_blocks()
        try:
            self._generate_initial_plans(voyages, deadline)
            root = self._solve_root_lp(voyages, deadline)
            records = list(root.get("records", []))
            root_exact = root["status"] == "optimal"
            return {
                "algorithm": "complete_voyage_plan_root_column_generation",
                "status": root["status"],
                "root_exact": root_exact,
                "root_objective": (
                    float(root["bound"]) if root_exact else math.nan
                ),
                "valid_lower_bound": float(
                    root.get("valid_lower_bound", 0.0)
                ),
                "voyage_count": len(voyages),
                "group_count": len(self.groups),
                "plan_count": len(self._plans),
                "phase_iterations": int(root.get("phase_iterations", 0)),
                "business_iterations": int(
                    root.get("business_iterations", 0)
                ),
                "initialization_records": list(self._initialization_records),
                "pricing_model_build_seconds": sum(
                    pricing.build_seconds
                    for pricing in self._local_pricing_models.values()
                ) + self._seed_model_build_seconds,
                "pricing_seconds": sum(
                    float(record.get("pricing_seconds", 0.0))
                    for record in records
                ),
                "records": records,
                "total_seconds": perf_counter() - started,
            }
        finally:
            self._dispose_pricing_models()

    def solve_column_generation(self) -> dict:
        started = perf_counter()
        total_limit = float(self.config.total_time_limit)
        deadline = (
            None if total_limit <= 0.0 else started + max(0.01, total_limit)
        )
        voyages = self._initialize_candidate_blocks()
        recovery_reserve = min(
            max(2.0, float(self.config.mip_time_limit)),
            max(2.0, 0.20 * total_limit),
            max(0.5, 0.5 * total_limit),
        )
        root_deadline = (
            None
            if deadline is None
            else max(started + 0.01, deadline - recovery_reserve)
        )
        try:
            self._generate_initial_plans(voyages, root_deadline)
            root = self._solve_root_lp(voyages, root_deadline)
            remaining = self._seconds_until(deadline)
            recovery_limit = float(self.config.mip_time_limit)
            if remaining is not None:
                recovery_limit = min(recovery_limit, max(0.0, remaining))
            active_plan_indices = {
                int(index)
                for index, value in root.get("plan_values", {}).items()
                if float(value) > 1e-7
            }
            recovery = self._solve_restricted_row_milp(
                recovery_limit,
                active_plan_indices,
            )
            selected = recovery.get("selected")
            if selected is None:
                raise RuntimeError(
                    "complete-voyage column generation did not produce a "
                    "feasible restricted row incumbent"
                )
            objective = float(recovery["objective"])
            valid_lower_bound = float(root.get("valid_lower_bound", 0.0))
            if valid_lower_bound > objective + 1e-7:
                raise RuntimeError(
                    "complete-voyage root lower bound exceeds incumbent: "
                    f"lower={valid_lower_bound}, upper={objective}"
                )
            absolute_gap = max(0.0, objective - valid_lower_bound)
            relative_gap = absolute_gap / max(abs(objective), 1e-12)
            root_exact = root["status"] == "optimal"
            status = (
                "optimal"
                if relative_gap <= 1e-9
                else "feasible_with_root_lp_bound"
                if root_exact
                else str(root["status"])
            )
            records = list(root.get("records", []))
            return {
                "algorithm": "complete_voyage_plan_column_generation",
                "status": status,
                "objective": objective,
                "valid_lower_bound": valid_lower_bound,
                "absolute_gap": absolute_gap,
                "relative_gap": relative_gap,
                "root_exact": root_exact,
                "root_status": root["status"],
                "root_objective": (
                    float(root["bound"]) if root_exact else math.nan
                ),
                "incumbent_source": "restricted_row_milp",
                "selected_locations": dict(selected),
                "restricted_row_recovery": {
                    key: value
                    for key, value in recovery.items()
                    if key != "selected"
                },
                "recovery_reserve_seconds": recovery_reserve,
                "voyage_count": len(voyages),
                "group_count": len(self.groups),
                "plan_count": len(self._plans),
                "phase_iterations": int(root.get("phase_iterations", 0)),
                "business_iterations": int(
                    root.get("business_iterations", 0)
                ),
                "initialization_records": list(self._initialization_records),
                "pricing_model_build_seconds": sum(
                    pricing.build_seconds
                    for pricing in self._local_pricing_models.values()
                ) + self._seed_model_build_seconds,
                "pricing_seconds": sum(
                    float(record.get("pricing_seconds", 0.0))
                    for record in records
                ),
                "pricing_records": records,
                "total_seconds": perf_counter() - started,
            }
        finally:
            self._dispose_pricing_models()

    def _algorithm_diagnostics(self, solve_result: dict) -> dict:
        return {
            "algorithm": "complete_voyage_plan_column_generation_gurobi",
            "model_scope": "export_declared_containers_row_allocation",
            "detailed_allocation_direction": "export_only",
            "target_voyages": self.problem.target_voyages,
            "attribute_rules": self.attribute_rules.as_dict(),
            "declared_export_group_count": int(
                self.demand_stats.get("declared_export_group_count", 0)
            ),
            "declared_export_box_count": int(
                self.demand_stats.get("declared_export_box_count", 0)
            ),
            "planned_group_count": len(self.groups),
            "planned_box_count": sum(group.demand for group in self.groups),
            "berth_distance_count": len(self.problem.berth_distances),
            "berth_by_voyage": self.problem.berth_by_voyage,
            "demand_alignment": self.demand_stats,
            "base_feasible_placement_count": self._base_feasible_placement_count,
            "formulation": "complete_voyage_plan_master",
            "decomposition": "one_exact_integer_pricing_block_per_voyage",
            "inner_pricing_structure": (
                "area_partitioned_row_locations_coordinated_within_voyage"
            ),
            "gurobi_available": True,
            "master_status": solve_result["status"],
            "master_bound_scope": (
                "complete_voyage_dantzig_wolfe_root_relaxation"
            ),
            "complete_model_lower_bound": solve_result["valid_lower_bound"],
            "complete_model_absolute_gap": solve_result["absolute_gap"],
            "complete_model_relative_gap": solve_result["relative_gap"],
            "complete_model_gap_source": (
                "exact_complete_voyage_plan_lp"
                if solve_result["root_exact"]
                else "valid_complete_voyage_pricing_bound"
            ),
            "master_mip_gap": solve_result["relative_gap"],
            "root_lp_lower_bound": solve_result["valid_lower_bound"],
            "root_lp_exact": solve_result["root_exact"],
            "business_objective_normalization": {
                "weights": self._objective_weights(),
                "scales": dict(self._objective_scales),
                "weight_sum": round(
                    sum(self._objective_weights().values()), 10
                ),
                "method": "natural_instance_scale",
            },
            "business_objective": self._business_objective_specification(),
            "objective_coefficients": {
                "area_guidance_l1_unit": self._area_guidance_penalty(),
                "area_activation_unit": self._area_activation_penalty(),
                "row_activation_unit": self._row_activation_penalty(),
            },
            "big_m_tightening": self._big_m_diagnostics(),
            "import_capacity_reservation": {
                "source_quantity_field": "new_qty",
                "role": "anonymous_size_compatible_capacity_only",
                "area_policy": (
                    "weighted_l1_deviation_from_big_plan_reference"
                ),
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
                "import_boxes": int(
                    sum(self.import_area_size_reference.values())
                ),
            },
            "voyage_plan_column_generation_statistics": {
                key: value
                for key, value in solve_result.items()
                if key != "selected_locations"
            },
        }

    def solve(self) -> ColumnGenerationResult:
        solve_result = self.solve_column_generation()
        selected = Counter(
            {
                int(index): int(value)
                for index, value in solve_result["selected_locations"].items()
            }
        )
        result = self._assemble_result(
            selected, self._algorithm_diagnostics(solve_result)
        )
        reconstructed = float(result.diagnostics["final_business_objective"])
        reported = float(solve_result["objective"])
        if abs(reconstructed - reported) > 1e-6 * max(1.0, abs(reported)):
            raise RuntimeError(
                "restricted row incumbent objective mismatch: "
                f"reconstructed={reconstructed}, reported={reported}"
            )
        result.columns = self._selected_direct_columns(selected)
        return result


__all__ = [
    "VoyagePlan",
    "VoyagePlanColumnGenerationPlanner",
    "VoyagePlanPricingConfig",
]
