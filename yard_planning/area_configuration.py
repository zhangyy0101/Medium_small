"""Reusable area-configuration master and exact pricing infrastructure."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from time import perf_counter

from .gurobi_backend import GurobiModel
from .planner import (
    BranchDecision,
    ColumnGenerationConfig,
    PlacementColumn,
    YardPlanningBase,
)


@dataclass(frozen=True)
class AdaptiveAreaPricingConfig:
    """Instance-independent controls for structural area pricing.

    A pricing problem is block guided only when its candidate count exceeds
    ``direct_candidate_limit`` and its bay-footprint graph has more than one
    component.  Area identifiers never enter the decision.
    """

    direct_candidate_limit: int = 2_000
    complex_area_pool_size: int = 6
    simple_area_pool_size: int = 1
    complex_time_weight: float = 1.5
    certificate_time_fraction: float = 0.15
    full_sweep_frequency: int = 2
    nested_max_iterations: int = 24
    nested_time_fraction: float = 0.70


@dataclass(frozen=True)
class AreaPricingProfile:
    area_no: str
    candidate_count: int
    footprint_blocks: tuple[tuple[str, ...], ...]
    largest_block_candidate_count: int
    strategy: str


@dataclass(frozen=True)
class AreaConfiguration:
    """One complete integer allocation for all relevant demand in one area."""

    configuration_id: str
    area_no: str
    placements: tuple[PlacementColumn, ...]
    import_reservations: tuple[tuple[str, str, str, int], ...]
    business_cost: float
    group_quantities: tuple[tuple[str, int], ...]
    export_guidance_quantities: tuple[
        tuple[tuple[str, str, str, str], int], ...
    ]
    import_total_quantities: tuple[tuple[tuple[str, str], int], ...]
    import_reference_quantities: tuple[
        tuple[tuple[str, str, str], int], ...
    ]


@dataclass
class _AreaPricingModel:
    area_no: str
    model: GurobiModel
    candidates: tuple[PlacementColumn, ...]
    placement_variables: dict[int, object]
    import_variables: dict[tuple[str, str, str], object]
    area_use_variables: dict[str, object]
    row_use_variables: dict[tuple[str, str, str], object]
    charged_row_use_keys: frozenset[tuple[str, str, str]]
    strategy: str
    build_seconds: float


@dataclass
class _NestedAreaPricingState:
    area_no: str
    blocks: tuple[tuple[str, ...], ...]
    block_models: dict[int, _AreaPricingModel]
    block_configurations: dict[int, list[AreaConfiguration]]
    block_configuration_keys: dict[int, set[tuple]]
    equivalent_block_classes: tuple[tuple[int, ...], ...]
    build_seconds: float = 0.0
    coordination_lp_model: GurobiModel | None = None
    coordination_lp_variables: dict[tuple[int, int], object] = field(
        default_factory=dict
    )
    coordination_lp_area_use: dict[str, object] = field(
        default_factory=dict
    )
    coordination_lp_constraints: dict[str, dict] = field(
        default_factory=dict
    )
    coordination_lp_registered_counts: dict[int, int] = field(
        default_factory=dict
    )


class AreaConfigurationPlanner(YardPlanningBase):
    """Reusable Dantzig--Wolfe area-pricing infrastructure.

    The Branch-and-Price solver uses the same master coefficients and pricing
    models at the root and every child node, so local packing conflicts remain
    convexified consistently.
    """

    def __init__(
        self,
        problem,
        config: ColumnGenerationConfig | None = None,
        area_pricing_config: AdaptiveAreaPricingConfig | None = None,
    ) -> None:
        super().__init__(problem, config)
        self.area_pricing_config = (
            area_pricing_config or AdaptiveAreaPricingConfig()
        )
        if self.area_pricing_config.direct_candidate_limit <= 0:
            raise ValueError("direct_candidate_limit must be positive")
        if self.area_pricing_config.complex_area_pool_size <= 0:
            raise ValueError("complex_area_pool_size must be positive")
        if self.area_pricing_config.simple_area_pool_size <= 0:
            raise ValueError("simple_area_pool_size must be positive")
        if self.area_pricing_config.complex_time_weight < 1.0:
            raise ValueError("complex_time_weight must be at least one")
        if self.area_pricing_config.full_sweep_frequency <= 0:
            raise ValueError("full_sweep_frequency must be positive")
        if self.area_pricing_config.nested_max_iterations <= 0:
            raise ValueError("nested_max_iterations must be positive")
        nested_time_fraction = float(
            self.area_pricing_config.nested_time_fraction
        )
        if not 0.0 < nested_time_fraction < 1.0:
            raise ValueError("nested_time_fraction must be in (0, 1)")
        certificate_fraction = float(
            self.area_pricing_config.certificate_time_fraction
        )
        if not 0.0 < certificate_fraction < 1.0:
            raise ValueError("certificate_time_fraction must be in (0, 1)")
        self._area_configurations: list[AreaConfiguration] = []
        self._area_configuration_index_by_identity: dict[tuple, int] = {}
        self._area_pricing_models: dict[str, _AreaPricingModel] = {}
        self._area_pricing_profiles: dict[str, AreaPricingProfile] = {}
        self._nested_area_pricing_states: dict[
            str, _NestedAreaPricingState
        ] = {}

    def _initialize_area_configuration_pool(self) -> tuple[str, ...]:
        self._prepare_master_index_sets()
        self._prepare_objective_normalization()
        areas = self._configuration_areas()
        if not areas:
            raise ValueError("area-configuration model has no usable area")
        if not self._area_configurations:
            for area_no in areas:
                self._append_area_configuration(
                    self._zero_area_configuration(area_no)
                )
        return areas

    @staticmethod
    def _configuration_identity(configuration: AreaConfiguration) -> tuple:
        return (
            configuration.area_no,
            tuple(
                (
                    placement.group_id,
                    placement.bay_key,
                    int(placement.quantity),
                    placement.row_allocation,
                )
                for placement in configuration.placements
            ),
            configuration.import_reservations,
        )

    def _append_area_configuration(
        self, configuration: AreaConfiguration
    ) -> int:
        identity = self._configuration_identity(configuration)
        if identity in self._area_configuration_index_by_identity:
            raise ValueError(f"duplicate area configuration: {identity}")
        index = len(self._area_configurations)
        stored = replace(
            configuration,
            configuration_id=f"AC{index + 1:07d}",
        )
        self._area_configurations.append(stored)
        self._area_configuration_index_by_identity[identity] = index
        return index

    def _ensure_area_configuration(
        self, configuration: AreaConfiguration
    ) -> int:
        identity = self._configuration_identity(configuration)
        existing = self._area_configuration_index_by_identity.get(identity)
        if existing is not None:
            return int(existing)
        return self._append_area_configuration(configuration)

    def _configuration_areas(self) -> tuple[str, ...]:
        export_areas = {
            candidate.area_no
            for group in self.groups
            for candidate in self._base_placements_for_group(group)
        }
        import_areas = {
            self.bays[bay_key].area_no
            for candidates in self.import_reservation_candidates.values()
            for bay_key, _capacity in candidates
        }
        return tuple(sorted(export_areas | import_areas))

    def _area_candidates(self, area_no: str) -> tuple[PlacementColumn, ...]:
        return tuple(
            candidate
            for group in self.groups
            for candidate in self._base_placements_for_group(group)
            if candidate.area_no == area_no
        )

    def _build_area_pricing_profile(self, area_no: str) -> AreaPricingProfile:
        candidates = self._area_candidates(area_no)
        graph: defaultdict[str, set[str]] = defaultdict(set)

        def connect(footprint: tuple[str, ...]) -> None:
            keys = tuple(
                sorted(
                    key
                    for key in set(footprint)
                    if self.bays[key].area_no == area_no
                )
            )
            for key in keys:
                graph[key]
                graph[key].update(other for other in keys if other != key)

        for candidate in candidates:
            connect(tuple(key for key, _row, _qty in candidate.row_allocation))
        for (_flow, size), available in sorted(
            self.import_reservation_candidates.items()
        ):
            for bay_key, _capacity in available:
                if self.bays[bay_key].area_no == area_no:
                    connect(self._placement_footprint_keys(bay_key, size))

        unvisited = set(graph)
        blocks: list[tuple[str, ...]] = []
        while unvisited:
            seed = min(unvisited)
            unvisited.remove(seed)
            component = {seed}
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                for neighbour in sorted(graph[current]):
                    if neighbour in unvisited:
                        unvisited.remove(neighbour)
                        component.add(neighbour)
                        frontier.append(neighbour)
            blocks.append(tuple(sorted(component)))
        blocks.sort()
        block_by_bay = {
            bay_key: block_index
            for block_index, block in enumerate(blocks)
            for bay_key in block
        }
        block_candidate_counts: Counter[int] = Counter()
        for candidate in candidates:
            footprint_blocks = {
                block_by_bay[key]
                for key, _row, _qty in candidate.row_allocation
            }
            if len(footprint_blocks) != 1:
                raise ValueError(
                    "area footprint decomposition split one placement: "
                    f"area={area_no}, candidate={candidate.column_id}"
                )
            block_candidate_counts[next(iter(footprint_blocks))] += 1
        complex_area = (
            len(candidates)
            > int(self.area_pricing_config.direct_candidate_limit)
            and len(blocks) > 1
        )
        return AreaPricingProfile(
            area_no=area_no,
            candidate_count=len(candidates),
            footprint_blocks=tuple(blocks),
            largest_block_candidate_count=max(
                block_candidate_counts.values(), default=0
            ),
            strategy=(
                "adaptive_block_guided_multicolumn"
                if complex_area
                else "direct_exact_mip"
            ),
        )

    def _area_pricing_profile(self, area_no: str) -> AreaPricingProfile:
        profile = self._area_pricing_profiles.get(area_no)
        if profile is None:
            profile = self._build_area_pricing_profile(area_no)
            self._area_pricing_profiles[area_no] = profile
        return profile

    def _zero_area_configuration(self, area_no: str) -> AreaConfiguration:
        return AreaConfiguration(
            configuration_id="",
            area_no=area_no,
            placements=(),
            import_reservations=(),
            business_cost=0.0,
            group_quantities=(),
            export_guidance_quantities=(),
            import_total_quantities=(),
            import_reference_quantities=(),
        )

    def _area_configuration_from_allocations(
        self,
        area_no: str,
        placements: tuple[PlacementColumn, ...],
        import_reservations: tuple[tuple[str, str, str, int], ...],
    ) -> AreaConfiguration:
        group_quantities: Counter[str] = Counter()
        export_guidance: Counter[tuple[str, str, str, str]] = Counter()
        import_totals: Counter[tuple[str, str]] = Counter()
        import_reference: Counter[tuple[str, str, str]] = Counter()
        used_groups: set[str] = set()
        used_rows: set[tuple[str, str, str]] = set()
        business_cost = 0.0
        for placement in placements:
            if placement.area_no != area_no:
                raise ValueError(
                    "area configuration contains a foreign placement: "
                    f"configuration_area={area_no}, "
                    f"placement_area={placement.area_no}"
                )
            group_quantities[placement.group_id] += int(placement.quantity)
            export_guidance[placement.quota_key] += int(placement.quantity)
            used_groups.add(placement.group_id)
            anchor_row = next(
                row_no
                for bay_key, row_no, _quantity in placement.row_allocation
                if bay_key == placement.bay_key
            )
            used_rows.add((placement.group_id, placement.bay_key, anchor_row))
            business_cost += (
                float(placement.intrinsic_cost) * int(placement.quantity)
            )
        for flow, size, bay_key, quantity in import_reservations:
            if self.bays[bay_key].area_no != area_no:
                raise ValueError(
                    "area configuration contains a foreign import reservation: "
                    f"configuration_area={area_no}, bay={bay_key}"
                )
            import_totals[(flow, size)] += int(quantity)
            import_reference[(flow, area_no, size)] += int(quantity)
        business_cost += self._area_activation_penalty() * len(used_groups)
        business_cost += self._row_activation_penalty() * len(used_rows)
        return AreaConfiguration(
            configuration_id="",
            area_no=area_no,
            placements=tuple(
                sorted(
                    placements,
                    key=lambda placement: (
                        placement.group_id,
                        self.bays[placement.bay_key].bay_order,
                        placement.row_allocation,
                    ),
                )
            ),
            import_reservations=tuple(sorted(import_reservations)),
            business_cost=float(business_cost),
            group_quantities=tuple(sorted(group_quantities.items())),
            export_guidance_quantities=tuple(sorted(export_guidance.items())),
            import_total_quantities=tuple(sorted(import_totals.items())),
            import_reference_quantities=tuple(sorted(import_reference.items())),
        )

    def _local_compatibility_constraints(
        self,
        model,
        quicksum,
        variables: dict[int, object],
        coefficient_rows: dict[str, dict[object, list[tuple[int, float]]]],
        area_no: str,
    ) -> dict[str, dict]:
        constraints: dict[str, dict] = defaultdict(dict)
        bay_choice_variables: defaultdict[tuple[str, str, str], list] = (
            defaultdict(list)
        )
        for key, items in sorted(
            coefficient_rows.get("bay_attr_link", {}).items()
        ):
            bay_key, attr, scope, value = key
            if self.bays[bay_key].area_no != area_no or not items:
                continue
            use = model.addVar(
                vtype="B",
                name=f"local_bay_attr_{self._key_name(key)}",
            )
            constraints["bay_attr_link"][key] = model.addConstr(
                quicksum(
                    coefficient * variables[index]
                    for index, coefficient in items
                )
                <= self._master_bay_attr_big_m[key] * use,
                name=f"local_bay_attr_link_{self._key_name(key)}",
            )
            bay_choice_variables[(bay_key, attr, scope)].append(use)
        for key, choices in sorted(bay_choice_variables.items()):
            constraints["bay_attr_one"][key] = model.addConstr(
                quicksum(choices) <= 1,
                name=f"local_bay_attr_one_{self._key_name(key)}",
            )

        return constraints

    def _build_area_pricing_model(
        self,
        area_no: str,
        *,
        footprint_block: tuple[str, ...] | None = None,
    ) -> _AreaPricingModel:
        from gurobipy import quicksum

        started = perf_counter()
        profile = self._area_pricing_profile(area_no)
        block_suffix = (
            ""
            if footprint_block is None
            else "_block_" + self._key_name(footprint_block)
        )
        model = GurobiModel(f"area_pricing_{area_no}{block_suffix}")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        self._set_gurobi_param(model, "MIPGap", 0.0)
        model.setMinimize()

        allowed_bays = (
            None if footprint_block is None else frozenset(footprint_block)
        )
        candidates = tuple(
            candidate
            for candidate in self._area_candidates(area_no)
            if allowed_bays is None
            or all(
                bay_key in allowed_bays
                for bay_key, _row_no, _quantity in (
                    candidate.row_allocation
                )
            )
        )
        block_by_bay = {
            bay_key: block_index
            for block_index, block in enumerate(profile.footprint_blocks)
            for bay_key in block
        }
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
                name=f"x_{area_no}_{index}",
            )
            for index in range(len(candidates))
        }
        import_variables = {
            (flow, size, bay_key): model.addVar(
                lb=0.0,
                ub=float(min(int(capacity), int(required))),
                vtype="I",
                name=(
                    f"r_{flow}_{size}_{self._key_name((bay_key,))}"
                ),
            )
            for (flow, size), available in sorted(
                self.import_reservation_candidates.items()
            )
            for bay_key, capacity in available
            if self.bays[bay_key].area_no == area_no
            and (allowed_bays is None or bay_key in allowed_bays)
            for required in (self.import_total_by_flow_size[(flow, size)],)
        }

        coefficient_rows: defaultdict[
            str, defaultdict[object, list[tuple[int, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        group_indices: defaultdict[str, list[int]] = defaultdict(list)
        physical_group_row_indices: defaultdict[
            tuple[str, str, str], list[int]
        ] = defaultdict(list)
        charged_row_use_keys: set[tuple[str, str, str]] = set()
        for index, candidate in enumerate(candidates):
            group_indices[candidate.group_id].append(index)
            anchor_row = next(
                row_no
                for bay_key, row_no, _quantity in candidate.row_allocation
                if bay_key == candidate.bay_key
            )
            charged_row_use_keys.add(
                (candidate.group_id, candidate.bay_key, anchor_row)
            )
            for footprint_key, row_no, _quantity in candidate.row_allocation:
                physical_group_row_indices[
                    (candidate.group_id, footprint_key, row_no)
                ].append(index)
            for section, values in self._placement_master_coefficients(
                candidate
            ).items():
                for key, coefficient in values.items():
                    if coefficient:
                        coefficient_rows[section][key].append(
                            (index, float(coefficient))
                        )

        import_by_bay: defaultdict[str, list] = defaultdict(list)
        import_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        for (flow, size, bay_key), variable in import_variables.items():
            for footprint_key in self._placement_footprint_keys(bay_key, size):
                import_by_bay[footprint_key].append(variable)
            import_by_bay_size[(bay_key, size)].append(variable)

        local_bays = {
            bay_key
            for bay_key in self._master_bay_capacity_keys
            if self.bays[bay_key].area_no == area_no
            and (allowed_bays is None or bay_key in allowed_bays)
        }
        for bay_key in sorted(local_bays):
            items = coefficient_rows["bay_capacity_limit"].get(bay_key, [])
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                + quicksum(import_by_bay.get(bay_key, []))
                <= int(self.bays[bay_key].physical_capacity),
                name=f"local_bay_cap_{self._key_name((bay_key,))}",
            )
        for key in sorted(self._master_bay_size_keys):
            bay_key, size = key
            if self.bays[bay_key].area_no != area_no or (
                allowed_bays is not None and bay_key not in allowed_bays
            ):
                continue
            items = coefficient_rows["bay_size_limit"].get(key, [])
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                + quicksum(import_by_bay_size.get(key, []))
                <= int(self.bays[bay_key].cap_by_size.get(size, 0)),
                name=f"local_bay_size_{self._key_name(key)}",
            )
        for key in sorted(self._master_row_capacity_keys):
            bay_key, row_no = key
            if self.bays[bay_key].area_no != area_no or (
                allowed_bays is not None and bay_key not in allowed_bays
            ):
                continue
            items = coefficient_rows["row_capacity_limit"].get(key, [])
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
        for key in sorted(self._master_row_size_keys):
            bay_key, row_no, size = key
            if self.bays[bay_key].area_no != area_no or (
                allowed_bays is not None and bay_key not in allowed_bays
            ):
                continue
            items = coefficient_rows["row_size_limit"].get(key, [])
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

        stack_variables_by_bay_size: defaultdict[tuple[str, str], list] = (
            defaultdict(list)
        )
        for key in sorted(self._master_stack_keys):
            bay_key, _port, size = key
            if self.bays[bay_key].area_no != area_no or (
                allowed_bays is not None and bay_key not in allowed_bays
            ):
                continue
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
            items = coefficient_rows["bay_port_stack_link"].get(key, [])
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                <= unit_capacity * stack,
                name=f"local_stack_load_{self._key_name(key)}",
            )
            stack_variables_by_bay_size[(bay_key, size)].append(stack)
        for key, stack_variables in stack_variables_by_bay_size.items():
            model.addConstr(
                quicksum(stack_variables)
                <= self._stack_count_for_bay_size(*key),
                name=f"local_stack_total_{self._key_name(key)}",
            )

        for group_id, indices in sorted(group_indices.items()):
            model.addConstr(
                quicksum(placement_variables[index] for index in indices)
                <= int(self.group_demand[group_id]),
                name=f"local_group_bound_{group_id}",
            )

        area_use_variables: dict[str, object] = {}
        if footprint_block is None:
            for group_id, indices in sorted(group_indices.items()):
                upper = min(
                    int(self.group_demand[group_id]),
                    sum(capacities[index] for index in indices),
                )
                use = model.addVar(
                    vtype="B",
                    name=f"local_group_area_use_{group_id}_{area_no}",
                )
                model.addConstr(
                    quicksum(
                        placement_variables[index] for index in indices
                    )
                    <= max(1, upper) * use,
                    name=f"local_group_area_link_{group_id}_{area_no}",
                )
                model.addConstr(
                    use
                    <= quicksum(
                        placement_variables[index] for index in indices
                    ),
                    name=f"local_group_area_use_lower_{group_id}_{area_no}",
                )
                area_use_variables[group_id] = use

        row_use_variables: dict[tuple[str, str, str], object] = {}
        row_choices: defaultdict[tuple[str, str], list] = defaultdict(list)
        for key, indices in sorted(physical_group_row_indices.items()):
            group_id, bay_key, row_no = key
            upper = min(
                int(self.group_demand[group_id]),
                sum(capacities[index] for index in indices),
            )
            use = model.addVar(
                vtype="B",
                name=(
                    f"local_group_row_use_{group_id}_"
                    f"{self._key_name((bay_key, row_no))}"
                ),
            )
            model.addConstr(
                quicksum(placement_variables[index] for index in indices)
                <= max(1, upper) * use,
                name=(
                    f"local_group_row_link_{group_id}_"
                    f"{self._key_name((bay_key, row_no))}"
                ),
            )
            model.addConstr(
                use
                <= quicksum(
                    placement_variables[index] for index in indices
                ),
                name=(
                    f"local_group_row_use_lower_{group_id}_"
                    f"{self._key_name((bay_key, row_no))}"
                ),
            )
            row_use_variables[key] = use
            row_choices[(bay_key, row_no)].append(use)
        for key, choices in sorted(row_choices.items()):
            group_ids = sorted(
                group_id
                for group_id, bay_key, row_no in row_use_variables
                if (bay_key, row_no) == key
            )
            for position, first_id in enumerate(group_ids):
                for second_id in group_ids[position + 1 :]:
                    if not self._groups_are_incompatible_on_one_row(
                        self.groups_by_id[first_id],
                        self.groups_by_id[second_id],
                    ):
                        raise ValueError(
                            "compact area pricing can use one group selector "
                            "per physical row only when every candidate group "
                            "pair is incompatible under the declared bay/row "
                            "no-mix rules: "
                            f"area={area_no}, row={key}, "
                            f"groups={first_id},{second_id}"
                        )
            model.addConstr(
                quicksum(choices) <= 1,
                name=f"local_physical_row_one_group_{self._key_name(key)}",
            )

        self._local_compatibility_constraints(
            model,
            quicksum,
            placement_variables,
            coefficient_rows,
            area_no,
        )
        model.update()
        if (
            footprint_block is None
            and profile.strategy == "adaptive_block_guided_multicolumn"
        ):
            # Partition 0 contains linking variables shared by all structural
            # blocks.  Positive partitions follow connected components of the
            # bay-footprint graph and activate Gurobi's partition heuristic.
            # The full MIP is still solved and supplies every exact bound.
            for variable in model.getVars():
                variable.Partition = 0
            for index, variable in placement_variables.items():
                footprint_blocks = {
                    block_by_bay[key]
                    for key, _row, _qty in candidates[index].row_allocation
                }
                variable.Partition = next(iter(footprint_blocks)) + 1
            for (_flow, _size, bay_key), variable in import_variables.items():
                variable.Partition = block_by_bay[bay_key] + 1
            for (_group_id, bay_key, _row_no), variable in (
                row_use_variables.items()
            ):
                variable.Partition = block_by_bay[bay_key] + 1
            model.update()
        return _AreaPricingModel(
            area_no=area_no,
            model=model,
            candidates=candidates,
            placement_variables=placement_variables,
            import_variables=import_variables,
            area_use_variables=area_use_variables,
            row_use_variables=row_use_variables,
            charged_row_use_keys=frozenset(charged_row_use_keys),
            strategy=(
                profile.strategy
                if footprint_block is None
                else "nested_block_exact_mip"
            ),
            build_seconds=perf_counter() - started,
        )

    def _configuration_from_pricing_solution(
        self,
        pricing: _AreaPricingModel,
        solution_number: int | None = None,
    ) -> AreaConfiguration:
        def value(variable) -> float:
            if solution_number is None:
                return self._gurobi_value(pricing.model, variable)
            return pricing.model.getPoolValue(variable, solution_number)

        placements: list[PlacementColumn] = []
        group_quantities: Counter[str] = Counter()
        export_guidance: Counter[tuple[str, str, str, str]] = Counter()
        used_groups: set[str] = set()
        used_rows: set[tuple[str, str, str]] = set()
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
                    for bay_key, row_no, _old_quantity in candidate.row_allocation
                ),
            )
            placements.append(placement)
            group_quantities[candidate.group_id] += quantity
            export_guidance[candidate.quota_key] += quantity
            used_groups.add(candidate.group_id)
            anchor_row = next(
                row_no
                for bay_key, row_no, _qty in candidate.row_allocation
                if bay_key == candidate.bay_key
            )
            used_rows.add((candidate.group_id, candidate.bay_key, anchor_row))
            business_cost += float(candidate.intrinsic_cost) * quantity

        import_reservations: list[tuple[str, str, str, int]] = []
        import_totals: Counter[tuple[str, str]] = Counter()
        import_reference: Counter[tuple[str, str, str]] = Counter()
        for key, variable in pricing.import_variables.items():
            quantity = int(round(value(variable)))
            if quantity <= 0:
                continue
            flow, size, bay_key = key
            import_reservations.append((flow, size, bay_key, quantity))
            import_totals[(flow, size)] += quantity
            import_reference[(flow, pricing.area_no, size)] += quantity

        business_cost += self._area_activation_penalty() * len(used_groups)
        business_cost += self._row_activation_penalty() * len(used_rows)
        placements.sort(
            key=lambda placement: (
                placement.group_id,
                self.bays[placement.bay_key].bay_order,
                placement.row_allocation,
            )
        )
        import_reservations.sort()
        return AreaConfiguration(
            configuration_id="",
            area_no=pricing.area_no,
            placements=tuple(placements),
            import_reservations=tuple(import_reservations),
            business_cost=float(business_cost),
            group_quantities=tuple(sorted(group_quantities.items())),
            export_guidance_quantities=tuple(sorted(export_guidance.items())),
            import_total_quantities=tuple(sorted(import_totals.items())),
            import_reference_quantities=tuple(sorted(import_reference.items())),
        )

    @staticmethod
    def _dual(
        duals: dict[tuple[str, object], float],
        section: str,
        key: object,
    ) -> float:
        return float(duals.get((section, key), 0.0))

    def _configuration_reduced_cost(
        self,
        configuration: AreaConfiguration,
        duals: dict[tuple[str, object], float],
        objective_mode: str,
        decisions: tuple[BranchDecision, ...] = (),
    ) -> float:
        reduced = (
            0.0
            if objective_mode == "phase_one"
            else float(configuration.business_cost)
        )
        reduced -= self._dual(
            duals, "area_convexity", configuration.area_no
        )
        for group_id, quantity in configuration.group_quantities:
            reduced -= quantity * self._dual(
                duals, "group_demand", group_id
            )
        for key, quantity in configuration.export_guidance_quantities:
            reduced -= quantity * self._dual(
                duals, "export_guidance", key
            )
        for key, quantity in configuration.import_total_quantities:
            reduced -= quantity * self._dual(
                duals, "import_total", key
            )
        for key, quantity in configuration.import_reference_quantities:
            reduced -= quantity * self._dual(
                duals, "import_reference", key
            )
        for decision_index, decision in enumerate(decisions):
            reduced -= self._configuration_branch_coefficient(
                configuration, decision
            ) * self._dual(duals, "branch_decision", decision_index)
        return float(reduced)

    @staticmethod
    def _configuration_branch_coefficient(
        configuration: AreaConfiguration,
        decision: BranchDecision,
    ) -> int:
        if decision.section == "branch_group_area_quantity":
            group_id, target_area = decision.key
            if configuration.area_no != target_area:
                return 0
            return int(dict(configuration.group_quantities).get(group_id, 0))
        if decision.section == "branch_row_use":
            group_id, target_bay, target_row = decision.key
            return int(
                any(
                    placement.group_id == group_id
                    and placement.bay_key == target_bay
                    and any(
                        bay_key == target_bay
                        and row_no == target_row
                        and int(quantity) > 0
                        for bay_key, row_no, quantity in (
                            placement.row_allocation
                        )
                    )
                    for placement in configuration.placements
                )
            )
        if decision.section == "branch_row_quantity":
            group_id, target_bay, target_row = decision.key
            return sum(
                int(quantity)
                for placement in configuration.placements
                if placement.group_id == group_id
                and placement.bay_key == target_bay
                for bay_key, row_no, quantity in placement.row_allocation
                if bay_key == target_bay and row_no == target_row
            )
        if decision.section == "branch_import_quantity":
            flow, size, target_bay = decision.key
            return sum(
                int(quantity)
                for config_flow, config_size, bay_key, quantity in (
                    configuration.import_reservations
                )
                if (config_flow, config_size, bay_key)
                == (flow, size, target_bay)
            )
        raise ValueError(f"unknown area branch section: {decision.section}")

    @staticmethod
    def _pricing_branch_solution_coefficient(
        pricing: _AreaPricingModel,
        decision: BranchDecision,
        solution_number: int,
    ) -> float:
        value = lambda variable: pricing.model.getPoolValue(
            variable, solution_number
        )
        if decision.section == "branch_group_area_quantity":
            group_id, target_area = decision.key
            if pricing.area_no != target_area:
                return 0.0
            return sum(
                value(variable)
                for index, variable in pricing.placement_variables.items()
                if pricing.candidates[index].group_id == group_id
            )
        if decision.section == "branch_row_use":
            variable = pricing.row_use_variables.get(tuple(decision.key))
            return 0.0 if variable is None else value(variable)
        if decision.section == "branch_row_quantity":
            group_id, target_bay, target_row = decision.key
            return sum(
                value(variable)
                for index, variable in pricing.placement_variables.items()
                for candidate in (pricing.candidates[index],)
                if candidate.group_id == group_id
                and candidate.bay_key == target_bay
                and any(
                    bay_key == target_bay and row_no == target_row
                    for bay_key, row_no, _quantity in candidate.row_allocation
                )
            )
        if decision.section == "branch_import_quantity":
            variable = pricing.import_variables.get(tuple(decision.key))
            return 0.0 if variable is None else value(variable)
        raise ValueError(f"unknown area branch section: {decision.section}")

    def _price_area_mip(
        self,
        area_no: str,
        duals: dict[tuple[str, object], float],
        objective_mode: str,
        time_limit: float,
        decisions: tuple[BranchDecision, ...] = (),
        *,
        stop_after_negative: bool = False,
    ) -> dict:
        pricing = self._area_pricing_models.get(area_no)
        if pricing is None:
            pricing = self._build_area_pricing_model(area_no)
            self._area_pricing_models[area_no] = pricing
        model = pricing.model
        for index, variable in pricing.placement_variables.items():
            candidate = pricing.candidates[index]
            coefficient = (
                0.0
                if objective_mode == "phase_one"
                else float(candidate.intrinsic_cost)
            )
            coefficient -= self._dual(
                duals, "group_demand", candidate.group_id
            )
            coefficient -= self._dual(
                duals, "export_guidance", candidate.quota_key
            )
            model.setVarObjective(variable, coefficient)
        for variable in pricing.area_use_variables.values():
            model.setVarObjective(
                variable,
                0.0
                if objective_mode == "phase_one"
                else self._area_activation_penalty(),
            )
        for key, variable in pricing.row_use_variables.items():
            model.setVarObjective(
                variable,
                0.0
                if objective_mode == "phase_one"
                else (
                    self._row_activation_penalty()
                    if key in pricing.charged_row_use_keys
                    else 0.0
                ),
            )
        for (flow, size, _bay_key), variable in pricing.import_variables.items():
            coefficient = -self._dual(
                duals, "import_total", (flow, size)
            )
            coefficient -= self._dual(
                duals,
                "import_reference",
                (flow, area_no, size),
            )
            model.setVarObjective(variable, coefficient)
        # Gurobi applies objective-attribute writes lazily.  Flush the base
        # objective before branch dual adjustments read those coefficients.
        model.update()
        for decision_index, decision in enumerate(decisions):
            branch_dual = self._dual(
                duals, "branch_decision", decision_index
            )
            if abs(branch_dual) <= 1e-14:
                continue
            if decision.section == "branch_group_area_quantity":
                group_id, target_area = decision.key
                if pricing.area_no != target_area:
                    continue
                for index, variable in pricing.placement_variables.items():
                    if pricing.candidates[index].group_id == group_id:
                        model.setVarObjective(
                            variable,
                            model.getVarObjective(variable) - branch_dual,
                        )
            elif decision.section == "branch_row_quantity":
                group_id, target_bay, target_row = decision.key
                for index, variable in pricing.placement_variables.items():
                    candidate = pricing.candidates[index]
                    if (
                        candidate.group_id == group_id
                        and candidate.bay_key == target_bay
                        and any(
                            bay_key == target_bay and row_no == target_row
                            for bay_key, row_no, _quantity in (
                                candidate.row_allocation
                            )
                        )
                    ):
                        model.setVarObjective(
                            variable,
                            model.getVarObjective(variable) - branch_dual,
                        )
            elif decision.section == "branch_row_use":
                key = tuple(decision.key)
                variable = pricing.row_use_variables.get(key)
                if variable is not None and key in pricing.charged_row_use_keys:
                    model.setVarObjective(
                        variable,
                        model.getVarObjective(variable) - branch_dual,
                    )
            elif decision.section == "branch_import_quantity":
                variable = pricing.import_variables.get(tuple(decision.key))
                if variable is not None:
                    model.setVarObjective(
                        variable,
                        model.getVarObjective(variable) - branch_dual,
                    )
            else:
                raise ValueError(
                    f"unknown area branch section: {decision.section}"
                )
        model.update()
        pool_size = 1
        if objective_mode == "business":
            pool_size = (
                int(self.area_pricing_config.complex_area_pool_size)
                if pricing.strategy == "adaptive_block_guided_multicolumn"
                else int(self.area_pricing_config.simple_area_pool_size)
            )
        self._set_gurobi_param(
            model,
            "PartitionPlace",
            15
            if (
                objective_mode == "business"
                and pricing.strategy
                == "adaptive_block_guided_multicolumn"
            )
            else 0,
        )
        self._set_gurobi_param(model, "PoolSolutions", pool_size)
        self._set_gurobi_param(
            model,
            "PoolSearchMode",
            1 if pool_size > 1 else 0,
        )
        convexity_dual = self._dual(duals, "area_convexity", area_no)
        tolerance = max(
            1e-9, float(self.config.reduced_cost_tolerance)
        )
        self._set_gurobi_param(
            model,
            "BestObjStop",
            convexity_dual - tolerance
            if stop_after_negative
            else -1e100,
        )
        self._set_gurobi_param(model, "TimeLimit", max(0.01, time_limit))
        started = perf_counter()
        model.optimize()
        elapsed = perf_counter() - started
        status = self._gurobi_status_name(model)
        solution_count = self._gurobi_solution_count(model)
        configurations: list[AreaConfiguration] = []
        configuration_reduced_costs: list[float] = []
        seen_identities: set[tuple] = set()
        for solution_number in range(min(solution_count, pool_size)):
            candidate_configuration = self._configuration_from_pricing_solution(
                pricing,
                solution_number=solution_number,
            )
            identity = self._configuration_identity(candidate_configuration)
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            candidate_reduced_cost = self._configuration_reduced_cost(
                candidate_configuration,
                duals,
                objective_mode,
                decisions,
            )
            for decision in decisions:
                reconstructed_coefficient = (
                    self._configuration_branch_coefficient(
                        candidate_configuration, decision
                    )
                )
                pricing_coefficient = (
                    self._pricing_branch_solution_coefficient(
                        pricing, decision, solution_number
                    )
                )
                if abs(pricing_coefficient - reconstructed_coefficient) > 1e-6:
                    raise RuntimeError(
                        "area branch coefficient reconstruction mismatch: "
                        f"area={area_no}, decision={decision}, "
                        f"pricing={pricing_coefficient}, "
                        f"configuration={reconstructed_coefficient}"
                    )
            pool_objective = pricing.model.getPoolObjective(solution_number)
            expected = pool_objective - self._dual(
                duals, "area_convexity", area_no
            )
            if abs(expected - candidate_reduced_cost) > 1e-6 * (
                1.0 + abs(candidate_reduced_cost)
            ):
                branch_duals = tuple(
                    self._dual(duals, "branch_decision", decision_index)
                    for decision_index in range(len(decisions))
                )
                raise RuntimeError(
                    "area pricing pool reduced-cost mismatch: "
                    f"area={area_no}, solution={solution_number}, "
                    f"model={expected}, configuration={candidate_reduced_cost}, "
                    f"decisions={decisions}, branch_duals={branch_duals}"
                )
            configurations.append(candidate_configuration)
            configuration_reduced_costs.append(candidate_reduced_cost)
        configuration = configurations[0] if configurations else None
        solution_reduced_cost = (
            math.inf
            if configuration is None
            else self._configuration_reduced_cost(
                configuration, duals, objective_mode, decisions
            )
        )
        if configuration is not None:
            raw_objective = self._gurobi_objective_value(model)
            expected = raw_objective - convexity_dual
            if abs(expected - solution_reduced_cost) > 1e-6 * (
                1.0 + abs(solution_reduced_cost)
            ):
                raise RuntimeError(
                    "area pricing reduced-cost mismatch: "
                    f"area={area_no}, model={expected}, "
                    f"configuration={solution_reduced_cost}"
                )
        raw_bound = self._gurobi_dual_bound(model)
        reduced_cost_lower_bound = (
            raw_bound - convexity_dual
            if math.isfinite(raw_bound)
            else -math.inf
        )
        return {
            "area_no": area_no,
            "status": status,
            "optimal": status == "optimal",
            "configuration": configuration,
            "configurations": configurations,
            "configuration_reduced_costs": configuration_reduced_costs,
            "solution_reduced_cost": solution_reduced_cost,
            "reduced_cost_lower_bound": reduced_cost_lower_bound,
            "seconds": elapsed,
            "build_seconds": pricing.build_seconds,
            "candidate_count": len(pricing.candidates),
            "model_variable_count": len(model.getVars()),
            "pricing_strategy": pricing.strategy,
            "pool_solution_count": solution_count,
            "returned_configuration_count": len(configurations),
        }

    def _nested_area_pricing_state(
        self, area_no: str
    ) -> _NestedAreaPricingState:
        state = self._nested_area_pricing_states.get(area_no)
        if state is not None:
            return state
        started = perf_counter()
        blocks = self._area_pricing_profile(area_no).footprint_blocks
        block_models = {
            block_index: self._build_area_pricing_model(
                area_no, footprint_block=block
            )
            for block_index, block in enumerate(blocks)
        }
        block_configurations = {
            block_index: [self._zero_area_configuration(area_no)]
            for block_index in range(len(blocks))
        }
        block_configuration_keys = {
            block_index: {
                self._configuration_identity(configurations[0])
            }
            for block_index, configurations in (
                block_configurations.items()
            )
        }
        equivalent_blocks: defaultdict[tuple, list[int]] = defaultdict(list)
        for block_index, block in enumerate(blocks):
            equivalent_blocks[
                self._nested_block_equivalence_signature(
                    block,
                    block_models[block_index],
                )
            ].append(block_index)
        state = _NestedAreaPricingState(
            area_no=area_no,
            blocks=blocks,
            block_models=block_models,
            block_configurations=block_configurations,
            block_configuration_keys=block_configuration_keys,
            equivalent_block_classes=tuple(
                tuple(indices) for indices in equivalent_blocks.values()
            ),
            build_seconds=perf_counter() - started,
        )
        self._nested_area_pricing_states[area_no] = state
        return state

    @staticmethod
    def _normalized_nested_value(value):
        if isinstance(value, dict):
            return tuple(
                (key, AreaConfigurationPlanner._normalized_nested_value(item))
                for key, item in sorted(value.items())
            )
        if isinstance(value, (set, frozenset)):
            return tuple(sorted(value))
        if isinstance(value, (list, tuple)):
            return tuple(
                AreaConfigurationPlanner._normalized_nested_value(item)
                for item in value
            )
        return value

    def _nested_block_equivalence_signature(
        self,
        block: tuple[str, ...],
        pricing: _AreaPricingModel,
    ) -> tuple:
        """Canonicalize a physical block independently of bay identifiers."""
        block_role = {bay_key: index for index, bay_key in enumerate(block)}
        bay_signature = []
        for bay_key in block:
            bay = self.bays[bay_key]
            partner_role = block_role.get(bay.large_bay_partner_key, -1)
            bay_signature.append(
                (
                    int(bay.physical_capacity),
                    self._normalized_nested_value(bay.cap_by_size),
                    self._normalized_nested_value(bay.row_cap_by_size),
                    self._normalized_nested_value(bay.row_physical_capacity),
                    int(partner_role),
                    self._normalized_nested_value(bay.existing_size_modes),
                    self._normalized_nested_value(bay.existing_heights),
                    self._normalized_nested_value(bay.existing_ports_by_row),
                    self._normalized_nested_value(bay.existing_attrs),
                    self._normalized_nested_value(bay.existing_attrs_by_row),
                    self._normalized_nested_value(
                        bay.existing_attrs_by_voyage
                    ),
                    self._normalized_nested_value(
                        bay.existing_attrs_by_row_by_voyage
                    ),
                )
            )
        candidate_signature = tuple(
            (
                candidate.group_id,
                candidate.voyage_id,
                candidate.flow,
                candidate.port,
                candidate.size,
                candidate.big_plan_size,
                candidate.height,
                self._normalized_nested_value(candidate.attributes),
                int(block_role[candidate.bay_key]),
                tuple(
                    (int(block_role[bay_key]), str(row_no), int(quantity))
                    for bay_key, row_no, quantity in (
                        candidate.row_allocation
                    )
                ),
                candidate.quota_key,
                candidate.group_key,
                float(candidate.intrinsic_cost),
                int(
                    self._base_location_capacity(
                        self.groups_by_id[candidate.group_id], candidate
                    )
                ),
            )
            for candidate in pricing.candidates
        )
        import_signature = tuple(
            sorted(
                (
                    flow,
                    size,
                    int(block_role[bay_key]),
                )
                for flow, size, bay_key in pricing.import_variables
            )
        )
        row_use_signature = tuple(
            sorted(
                (
                    group_id,
                    int(block_role[bay_key]),
                    row_no,
                    (group_id, bay_key, row_no)
                    in pricing.charged_row_use_keys,
                )
                for group_id, bay_key, row_no in pricing.row_use_variables
            )
        )
        return (
            pricing.model.getFingerprint(),
            tuple(bay_signature),
            candidate_signature,
            import_signature,
            row_use_signature,
        )

    def _block_configuration_from_solution(
        self,
        pricing: _AreaPricingModel,
        solution_number: int | None = None,
    ) -> AreaConfiguration:
        configuration = self._configuration_from_pricing_solution(
            pricing,
            solution_number=solution_number,
        )
        local_business_cost = float(configuration.business_cost) - (
            self._area_activation_penalty()
            * len(configuration.group_quantities)
        )
        return replace(
            configuration,
            business_cost=float(local_business_cost),
        )

    def _block_configuration_base_objective(
        self,
        configuration: AreaConfiguration,
        duals: dict[tuple[str, object], float],
        objective_mode: str,
        decisions: tuple[BranchDecision, ...],
    ) -> float:
        return self._configuration_reduced_cost(
            configuration,
            duals,
            objective_mode,
            decisions,
        ) + self._dual(
            duals, "area_convexity", configuration.area_no
        )

    def _nested_group_area_upper(
        self, state: _NestedAreaPricingState
    ) -> dict[str, int]:
        return {
            group.group_id: min(
                int(self.group_demand[group.group_id]),
                sum(
                    self._base_location_capacity(group, candidate)
                    for candidate in self._area_candidates(state.area_no)
                    if candidate.group_id == group.group_id
                ),
            )
            for group in self.groups
        }

    def _solve_nested_coordination_lp(
        self,
        state: _NestedAreaPricingState,
        duals: dict[tuple[str, object], float],
        objective_mode: str,
        decisions: tuple[BranchDecision, ...],
        time_limit: float,
    ) -> dict:
        from gurobipy import quicksum

        model = state.coordination_lp_model
        if model is None:
            model = GurobiModel(f"nested_area_coord_{state.area_no}_lp")
            self._configure_gurobi_output(model)
            self._set_gurobi_param(
                model, "Seed", int(self.config.solver_seed)
            )
            if int(self.config.solver_threads) > 0:
                self._set_gurobi_param(
                    model, "Threads", int(self.config.solver_threads)
                )
            model.setMinimize()
            configuration_variables: dict[tuple[int, int], object] = {}
            for block_index, configurations in sorted(
                state.block_configurations.items()
            ):
                for configuration_index in range(len(configurations)):
                    configuration_variables[
                        (block_index, configuration_index)
                    ] = model.addVar(
                        lb=0.0,
                        ub=1.0,
                        vtype="C",
                        name=(
                            f"nested_lambda_{block_index}_"
                            f"{configuration_index}"
                        ),
                    )
            area_use = {
                group.group_id: model.addVar(
                    lb=0.0,
                    ub=1.0,
                    vtype="C",
                    name=f"nested_area_use_{group.group_id}",
                )
                for group in self.groups
            }
            convexity = {
                block_index: model.addConstr(
                    quicksum(
                        configuration_variables[
                            (block_index, configuration_index)
                        ]
                        for configuration_index in range(
                            len(configurations)
                        )
                    )
                    == 1.0,
                    name=f"nested_block_convexity_{block_index}",
                )
                for block_index, configurations in sorted(
                    state.block_configurations.items()
                )
            }
            group_area_upper = self._nested_group_area_upper(state)
            group_upper = {}
            group_lower = {}
            for group in self.groups:
                group_id = group.group_id
                quantity_terms = []
                for block_index, configurations in sorted(
                    state.block_configurations.items()
                ):
                    for configuration_index, configuration in enumerate(
                        configurations
                    ):
                        quantity = int(
                            dict(configuration.group_quantities).get(
                                group_id, 0
                            )
                        )
                        if quantity:
                            quantity_terms.append(
                                quantity
                                * configuration_variables[
                                    (block_index, configuration_index)
                                ]
                            )
                quantity_expression = quicksum(quantity_terms)
                group_upper[group_id] = model.addConstr(
                    quantity_expression
                    <= max(1, group_area_upper[group_id])
                    * area_use[group_id],
                    name=f"nested_group_upper_{group_id}",
                )
                group_lower[group_id] = model.addConstr(
                    area_use[group_id] <= quantity_expression,
                    name=f"nested_group_lower_{group_id}",
                )
            model.update()
            state.coordination_lp_model = model
            state.coordination_lp_variables = configuration_variables
            state.coordination_lp_area_use = area_use
            state.coordination_lp_constraints = {
                "convexity": convexity,
                "group_upper": group_upper,
                "group_lower": group_lower,
            }
            state.coordination_lp_registered_counts = {
                block_index: len(configurations)
                for block_index, configurations in (
                    state.block_configurations.items()
                )
            }

        constraints = state.coordination_lp_constraints
        for block_index, configurations in sorted(
            state.block_configurations.items()
        ):
            registered = int(
                state.coordination_lp_registered_counts.get(block_index, 0)
            )
            for configuration_index in range(
                registered, len(configurations)
            ):
                configuration = configurations[configuration_index]
                terms: list[tuple[float, object]] = [
                    (1.0, constraints["convexity"][block_index])
                ]
                for group_id, quantity in configuration.group_quantities:
                    terms.append(
                        (
                            float(quantity),
                            constraints["group_upper"][group_id],
                        )
                    )
                    terms.append(
                        (
                            -float(quantity),
                            constraints["group_lower"][group_id],
                        )
                    )
                state.coordination_lp_variables[
                    (block_index, configuration_index)
                ] = model.addPricedVar(
                    terms,
                    lb=0.0,
                    ub=1.0,
                    name=(
                        f"nested_lambda_{block_index}_"
                        f"{configuration_index}"
                    ),
                )
            state.coordination_lp_registered_counts[block_index] = len(
                configurations
            )

        for (block_index, configuration_index), variable in (
            state.coordination_lp_variables.items()
        ):
            configuration = state.block_configurations[block_index][
                configuration_index
            ]
            model.setVarObjective(
                variable,
                self._block_configuration_base_objective(
                    configuration,
                    duals,
                    objective_mode,
                    decisions,
                ),
            )
        area_activation_cost = (
            0.0
            if objective_mode == "phase_one"
            else self._area_activation_penalty()
        )
        for variable in state.coordination_lp_area_use.values():
            model.setVarObjective(variable, area_activation_cost)
        model.update()
        self._set_gurobi_param(
            model, "TimeLimit", max(0.01, float(time_limit))
        )
        model.optimize()
        status = self._gurobi_status_name(model)
        if status != "optimal":
            return {"status": status}
        return {
            "status": status,
            "objective": self._gurobi_objective_value(model),
            "duals": {
                section: {
                    key: float(model.getLinearDual(row))
                    for key, row in rows.items()
                }
                for section, rows in constraints.items()
            },
        }

    def _build_nested_coordination_mip(
        self,
        state: _NestedAreaPricingState,
        duals: dict[tuple[str, object], float],
        objective_mode: str,
        decisions: tuple[BranchDecision, ...],
        time_limit: float,
    ) -> dict:
        from gurobipy import quicksum

        model = GurobiModel(f"nested_area_coord_{state.area_no}_mip")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        model.setMinimize()
        configuration_variables: dict[tuple[int, int], object] = {}
        for block_index, configurations in sorted(
            state.block_configurations.items()
        ):
            for configuration_index, configuration in enumerate(
                configurations
            ):
                configuration_variables[
                    (block_index, configuration_index)
                ] = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    vtype="B",
                    obj=self._block_configuration_base_objective(
                        configuration,
                        duals,
                        objective_mode,
                        decisions,
                    ),
                    name=(
                        f"nested_lambda_{block_index}_"
                        f"{configuration_index}"
                    ),
                )
        area_activation_cost = (
            0.0
            if objective_mode == "phase_one"
            else self._area_activation_penalty()
        )
        area_use = {
            group.group_id: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="B",
                obj=area_activation_cost,
                name=f"nested_area_use_{group.group_id}",
            )
            for group in self.groups
        }
        for block_index, configurations in sorted(
            state.block_configurations.items()
        ):
            model.addConstr(
                quicksum(
                    configuration_variables[
                        (block_index, configuration_index)
                    ]
                    for configuration_index in range(len(configurations))
                )
                == 1.0,
                name=f"nested_block_convexity_{block_index}",
            )
        group_area_upper = self._nested_group_area_upper(state)
        for group in self.groups:
            group_id = group.group_id
            quantity_terms = []
            for block_index, configurations in sorted(
                state.block_configurations.items()
            ):
                for configuration_index, configuration in enumerate(
                    configurations
                ):
                    quantity = int(
                        dict(configuration.group_quantities).get(
                            group_id, 0
                        )
                    )
                    if quantity:
                        quantity_terms.append(
                            quantity
                            * configuration_variables[
                                (block_index, configuration_index)
                            ]
                        )
            quantity_expression = quicksum(quantity_terms)
            model.addConstr(
                quantity_expression
                <= max(1, group_area_upper[group_id]) * area_use[group_id],
                name=f"nested_group_upper_{group_id}",
            )
            model.addConstr(
                area_use[group_id] <= quantity_expression,
                name=f"nested_group_lower_{group_id}",
            )
        self._set_gurobi_param(
            model, "TimeLimit", max(0.01, float(time_limit))
        )
        self._set_gurobi_param(model, "MIPGap", 0.0)
        model.optimize()
        status = self._gurobi_status_name(model)
        solution_count = self._gurobi_solution_count(model)
        selected: dict[int, int] = {}
        if solution_count > 0:
            for block_index, configurations in sorted(
                state.block_configurations.items()
            ):
                best_index = max(
                    range(len(configurations)),
                    key=lambda configuration_index: self._gurobi_value(
                        model,
                        configuration_variables[
                            (block_index, configuration_index)
                        ],
                    ),
                )
                selected[block_index] = int(best_index)
        result = {
            "model": model,
            "status": status,
            "solution_count": solution_count,
            "objective": (
                self._gurobi_objective_value(model)
                if solution_count > 0
                else math.inf
            ),
            "bound": (
                self._gurobi_dual_bound(model)
                if solution_count > 0
                else -math.inf
            ),
            "selected": selected,
        }
        return result

    def _price_nested_block(
        self,
        pricing: _AreaPricingModel,
        block_index: int,
        duals: dict[tuple[str, object], float],
        coordination_duals: dict,
        objective_mode: str,
        decisions: tuple[BranchDecision, ...],
        time_limit: float,
        pool_size: int = 1,
    ) -> dict:
        model = pricing.model
        group_adjustment = {
            group.group_id: (
                -float(
                    coordination_duals["group_upper"].get(
                        group.group_id, 0.0
                    )
                )
                + float(
                    coordination_duals["group_lower"].get(
                        group.group_id, 0.0
                    )
                )
            )
            for group in self.groups
        }
        for index, variable in pricing.placement_variables.items():
            candidate = pricing.candidates[index]
            coefficient = (
                0.0
                if objective_mode == "phase_one"
                else float(candidate.intrinsic_cost)
            )
            coefficient -= self._dual(
                duals, "group_demand", candidate.group_id
            )
            coefficient -= self._dual(
                duals, "export_guidance", candidate.quota_key
            )
            coefficient += group_adjustment[candidate.group_id]
            model.setVarObjective(variable, coefficient)
        for key, variable in pricing.row_use_variables.items():
            model.setVarObjective(
                variable,
                0.0
                if objective_mode == "phase_one"
                else (
                    self._row_activation_penalty()
                    if key in pricing.charged_row_use_keys
                    else 0.0
                ),
            )
        for (flow, size, _bay_key), variable in (
            pricing.import_variables.items()
        ):
            coefficient = -self._dual(
                duals, "import_total", (flow, size)
            )
            coefficient -= self._dual(
                duals,
                "import_reference",
                (flow, pricing.area_no, size),
            )
            model.setVarObjective(variable, coefficient)
        model.update()
        for decision_index, decision in enumerate(decisions):
            branch_dual = self._dual(
                duals, "branch_decision", decision_index
            )
            if abs(branch_dual) <= 1e-14:
                continue
            if decision.section == "branch_group_area_quantity":
                group_id, target_area = decision.key
                if pricing.area_no != target_area:
                    continue
                for index, variable in pricing.placement_variables.items():
                    if pricing.candidates[index].group_id == group_id:
                        model.setVarObjective(
                            variable,
                            model.getVarObjective(variable) - branch_dual,
                        )
            elif decision.section == "branch_row_quantity":
                group_id, target_bay, target_row = decision.key
                for index, variable in pricing.placement_variables.items():
                    candidate = pricing.candidates[index]
                    if (
                        candidate.group_id == group_id
                        and candidate.bay_key == target_bay
                        and any(
                            bay_key == target_bay and row_no == target_row
                            for bay_key, row_no, _quantity in (
                                candidate.row_allocation
                            )
                        )
                    ):
                        model.setVarObjective(
                            variable,
                            model.getVarObjective(variable) - branch_dual,
                        )
            elif decision.section == "branch_row_use":
                key = tuple(decision.key)
                variable = pricing.row_use_variables.get(key)
                if variable is not None and key in pricing.charged_row_use_keys:
                    model.setVarObjective(
                        variable,
                        model.getVarObjective(variable) - branch_dual,
                    )
            elif decision.section == "branch_import_quantity":
                variable = pricing.import_variables.get(tuple(decision.key))
                if variable is not None:
                    model.setVarObjective(
                        variable,
                        model.getVarObjective(variable) - branch_dual,
                    )
            else:
                raise ValueError(
                    f"unknown area branch section: {decision.section}"
                )
        model.update()
        pool_size = max(1, int(pool_size))
        self._set_gurobi_param(model, "PoolSolutions", pool_size)
        self._set_gurobi_param(
            model,
            "PoolSearchMode",
            1 if pool_size > 1 else 0,
        )
        self._set_gurobi_param(model, "PartitionPlace", 0)
        self._set_gurobi_param(model, "BestObjStop", -1e100)
        self._set_gurobi_param(
            model, "TimeLimit", max(0.01, float(time_limit))
        )
        started = perf_counter()
        model.optimize()
        elapsed = perf_counter() - started
        status = self._gurobi_status_name(model)
        solution_count = self._gurobi_solution_count(model)
        convexity_dual = float(
            coordination_duals["convexity"].get(block_index, 0.0)
        )
        configurations: list[AreaConfiguration] = []
        configuration_reduced_costs: list[float] = []
        seen_identities: set[tuple] = set()
        for solution_number in range(min(solution_count, pool_size)):
            candidate_configuration = self._block_configuration_from_solution(
                pricing,
                solution_number=solution_number,
            )
            identity = self._configuration_identity(candidate_configuration)
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            candidate_reduced_cost = self._block_configuration_base_objective(
                candidate_configuration,
                duals,
                objective_mode,
                decisions,
            )
            candidate_reduced_cost += sum(
                int(quantity) * group_adjustment[group_id]
                for group_id, quantity in (
                    candidate_configuration.group_quantities
                )
            )
            candidate_reduced_cost -= convexity_dual
            expected = (
                model.getPoolObjective(solution_number) - convexity_dual
            )
            if abs(expected - candidate_reduced_cost) > 1e-6 * (
                1.0 + abs(candidate_reduced_cost)
            ):
                raise RuntimeError(
                    "nested block reduced-cost mismatch: "
                    f"area={pricing.area_no}, block={block_index}, "
                    f"solution={solution_number}, model={expected}, "
                    f"configuration={candidate_reduced_cost}"
                )
            configurations.append(candidate_configuration)
            configuration_reduced_costs.append(candidate_reduced_cost)
        configuration = configurations[0] if configurations else None
        reduced_cost = (
            configuration_reduced_costs[0]
            if configuration_reduced_costs
            else math.inf
        )
        raw_bound = self._gurobi_dual_bound(model)
        return {
            "block_index": block_index,
            "status": status,
            "optimal": status == "optimal",
            "configuration": configuration,
            "configurations": configurations,
            "configuration_reduced_costs": configuration_reduced_costs,
            "solution_reduced_cost": float(reduced_cost),
            "reduced_cost_lower_bound": (
                float(raw_bound - convexity_dual)
                if math.isfinite(raw_bound)
                else -math.inf
            ),
            "seconds": elapsed,
            "candidate_count": len(pricing.candidates),
            "pool_solution_count": solution_count,
            "returned_configuration_count": len(configurations),
        }

    @staticmethod
    def _nested_block_classes_for_decisions(
        state: _NestedAreaPricingState,
        decisions: tuple[BranchDecision, ...],
    ) -> tuple[tuple[int, ...], ...]:
        location_specific_sections = {
            "branch_row_quantity",
            "branch_row_use",
            "branch_import_quantity",
        }
        if any(
            decision.section in location_specific_sections
            for decision in decisions
        ):
            return tuple((index,) for index in range(len(state.blocks)))
        return state.equivalent_block_classes

    def _translate_nested_block_configuration(
        self,
        state: _NestedAreaPricingState,
        configuration: AreaConfiguration,
        source_index: int,
        target_index: int,
    ) -> AreaConfiguration:
        bay_mapping = dict(
            zip(
                state.blocks[source_index],
                state.blocks[target_index],
                strict=True,
            )
        )
        placements = tuple(
            replace(
                placement,
                bay_key=bay_mapping[placement.bay_key],
                bay_no=self.bays[bay_mapping[placement.bay_key]].bay_no,
                row_allocation=tuple(
                    (bay_mapping[bay_key], row_no, int(quantity))
                    for bay_key, row_no, quantity in (
                        placement.row_allocation
                    )
                ),
            )
            for placement in configuration.placements
        )
        imports = tuple(
            (
                flow,
                size,
                bay_mapping[bay_key],
                int(quantity),
            )
            for flow, size, bay_key, quantity in (
                configuration.import_reservations
            )
        )
        return replace(
            configuration,
            configuration_id="",
            placements=placements,
            import_reservations=imports,
        )

    def _translate_nested_block_result(
        self,
        state: _NestedAreaPricingState,
        result: dict,
        source_index: int,
        target_index: int,
        duals: dict[tuple[str, object], float],
        coordination_duals: dict,
        objective_mode: str,
        decisions: tuple[BranchDecision, ...],
    ) -> dict:
        translated = dict(result)
        translated_configurations: list[AreaConfiguration] = []
        for configuration in result["configurations"]:
            translated_configuration = (
                self._translate_nested_block_configuration(
                    state,
                    configuration,
                    source_index,
                    target_index,
                )
            )
            source_base = self._block_configuration_base_objective(
                configuration,
                duals,
                objective_mode,
                decisions,
            )
            target_base = self._block_configuration_base_objective(
                translated_configuration,
                duals,
                objective_mode,
                decisions,
            )
            if abs(source_base - target_base) > 1e-9 * (
                1.0 + abs(source_base)
            ):
                raise RuntimeError(
                    "equivalent nested blocks have different master costs: "
                    f"area={state.area_no}, source={source_index}, "
                    f"target={target_index}, source_cost={source_base}, "
                    f"target_cost={target_base}"
                )
            translated_configurations.append(translated_configuration)
        source_convexity = float(
            coordination_duals["convexity"].get(source_index, 0.0)
        )
        target_convexity = float(
            coordination_duals["convexity"].get(target_index, 0.0)
        )
        convexity_shift = source_convexity - target_convexity
        translated_reduced_costs = [
            float(reduced_cost) + convexity_shift
            for reduced_cost in result["configuration_reduced_costs"]
        ]
        translated["configurations"] = translated_configurations
        translated["configuration_reduced_costs"] = (
            translated_reduced_costs
        )
        translated["configuration"] = (
            translated_configurations[0]
            if translated_configurations
            else None
        )
        if math.isfinite(float(result["solution_reduced_cost"])):
            translated["solution_reduced_cost"] = (
                translated_reduced_costs[0]
            )
        if math.isfinite(float(result["reduced_cost_lower_bound"])):
            translated["reduced_cost_lower_bound"] = float(
                result["reduced_cost_lower_bound"]
            ) + convexity_shift
        translated.update(
            {
                "block_index": target_index,
                "seconds": 0.0,
                "candidate_count": len(
                    state.block_models[target_index].candidates
                ),
                "shared_pricing_representative": source_index,
            }
        )
        return translated

    def _combine_nested_block_configurations(
        self,
        state: _NestedAreaPricingState,
        selected: dict[int, int],
    ) -> AreaConfiguration:
        placements: list[PlacementColumn] = []
        import_reservations: list[tuple[str, str, str, int]] = []
        for block_index in range(len(state.blocks)):
            configuration_index = int(selected[block_index])
            configuration = state.block_configurations[block_index][
                configuration_index
            ]
            placements.extend(configuration.placements)
            import_reservations.extend(configuration.import_reservations)
        return self._area_configuration_from_allocations(
            state.area_no,
            tuple(placements),
            tuple(import_reservations),
        )

    def _nested_area_result(
        self,
        state: _NestedAreaPricingState,
        configuration: AreaConfiguration,
        candidate_reduced_cost: float,
        reduced_cost_lower_bound: float,
        nested_records: list[dict],
        closed: bool,
        started: float,
        *,
        certified_nonnegative: bool,
    ) -> dict:
        return {
            "area_no": state.area_no,
            "status": (
                "nested_lp_certificate"
                if certified_nonnegative
                else "nested_negative_configuration"
            ),
            "optimal": certified_nonnegative,
            "configuration": configuration,
            "configurations": [configuration],
            "configuration_reduced_costs": [candidate_reduced_cost],
            "solution_reduced_cost": candidate_reduced_cost,
            "reduced_cost_lower_bound": reduced_cost_lower_bound,
            "seconds": perf_counter() - started,
            "build_seconds": state.build_seconds,
            "candidate_count": sum(
                len(model.candidates)
                for model in state.block_models.values()
            ),
            "model_variable_count": sum(
                len(model.model.getVars())
                for model in state.block_models.values()
            ),
            "pricing_strategy": "nested_exact_block_column_generation",
            "pool_solution_count": 1,
            "returned_configuration_count": 1,
            "nested_pricing": {
                "block_count": len(state.blocks),
                "block_configuration_count": sum(
                    len(configurations)
                    for configurations in (
                        state.block_configurations.values()
                    )
                ),
                "iterations": len(nested_records),
                "lp_closed": closed,
                "records": nested_records,
            },
        }

    def _price_area_nested(
        self,
        area_no: str,
        duals: dict[tuple[str, object], float],
        objective_mode: str,
        time_limit: float,
        decisions: tuple[BranchDecision, ...],
    ) -> dict | None:
        started = perf_counter()
        state = self._nested_area_pricing_state(area_no)
        total_budget = max(0.02, float(time_limit))
        nested_budget = max(
            0.02,
            total_budget
            * float(self.area_pricing_config.nested_time_fraction),
        )
        nested_deadline = started + nested_budget
        generation_deadline = started + 0.75 * nested_budget
        tolerance = max(1e-9, float(self.config.reduced_cost_tolerance))
        best_area_lower_bound = -math.inf
        nested_records: list[dict] = []
        closed = False
        block_classes = self._nested_block_classes_for_decisions(
            state, decisions
        )
        largest_shared_pool = max(
            (
                min(
                    len(block_class),
                    int(self.area_pricing_config.complex_area_pool_size),
                )
                for block_class in block_classes
            ),
            default=1,
        )
        nested_iteration_limit = max(
            2,
            math.ceil(
                int(self.area_pricing_config.nested_max_iterations)
                / max(1, largest_shared_pool)
            ),
        )

        for iteration in range(
            1, nested_iteration_limit + 1
        ):
            remaining = generation_deadline - perf_counter()
            if remaining <= 1e-6:
                break
            coordination = self._solve_nested_coordination_lp(
                state,
                duals,
                objective_mode,
                decisions,
                time_limit=min(remaining, max(0.05, 0.15 * remaining)),
            )
            if coordination["status"] != "optimal":
                break
            coordination_objective = float(coordination["objective"])
            coordination_duals = dict(coordination["duals"])

            block_results: list[dict] = []
            new_block_configurations = 0
            solved_block_models = 0
            for position, block_class in enumerate(block_classes):
                remaining = generation_deadline - perf_counter()
                if remaining <= 1e-6:
                    break
                remaining_classes = len(block_classes) - position
                representative = block_class[0]
                representative_result = self._price_nested_block(
                    state.block_models[representative],
                    representative,
                    duals,
                    coordination_duals,
                    objective_mode,
                    decisions,
                    max(0.01, remaining / max(1, remaining_classes)),
                    pool_size=min(
                        len(block_class),
                        int(
                            self.area_pricing_config.complex_area_pool_size
                        ),
                    ),
                )
                solved_block_models += 1
                for block_index in block_class:
                    result = (
                        representative_result
                        if block_index == representative
                        else self._translate_nested_block_result(
                            state,
                            representative_result,
                            representative,
                            block_index,
                            duals,
                            coordination_duals,
                            objective_mode,
                            decisions,
                        )
                    )
                    block_results.append(result)
                    for configuration, reduced_cost in zip(
                        result["configurations"],
                        result["configuration_reduced_costs"],
                        strict=True,
                    ):
                        if float(reduced_cost) >= -tolerance:
                            continue
                        identity = self._configuration_identity(
                            configuration
                        )
                        if (
                            identity
                            in state.block_configuration_keys[block_index]
                        ):
                            continue
                        state.block_configuration_keys[block_index].add(
                            identity
                        )
                        state.block_configurations[block_index].append(
                            configuration
                        )
                        new_block_configurations += 1
            bounds_available = (
                len(block_results) == len(state.blocks)
                and all(
                    math.isfinite(
                        float(result["reduced_cost_lower_bound"])
                    )
                    for result in block_results
                )
            )
            area_lower_bound = (
                coordination_objective
                + sum(
                    min(
                        0.0,
                        float(result["reduced_cost_lower_bound"]),
                    )
                    for result in block_results
                )
                if bounds_available
                else -math.inf
            )
            best_area_lower_bound = max(
                best_area_lower_bound, area_lower_bound
            )
            nested_records.append(
                {
                    "iteration": iteration,
                    "coordination_objective": coordination_objective,
                    "area_lower_bound": area_lower_bound,
                    "new_block_configurations": new_block_configurations,
                    "priced_block_count": len(block_results),
                    "solved_block_model_count": solved_block_models,
                    "equivalent_block_class_count": len(block_classes),
                    "all_block_pricing_optimal": (
                        len(block_results) == len(state.blocks)
                        and all(
                            bool(result["optimal"])
                            for result in block_results
                        )
                    ),
                    "block_pricing_seconds": sum(
                        float(result["seconds"])
                        for result in block_results
                    ),
                }
            )
            if new_block_configurations:
                continue
            if (
                len(block_results) == len(state.blocks)
                and all(bool(result["optimal"]) for result in block_results)
            ):
                best_area_lower_bound = max(
                    best_area_lower_bound, coordination_objective
                )
                closed = True
            break

        convexity_dual = self._dual(duals, "area_convexity", area_no)
        reduced_cost_lower_bound = (
            best_area_lower_bound - convexity_dual
            if math.isfinite(best_area_lower_bound)
            else -math.inf
        )
        if reduced_cost_lower_bound >= -tolerance:
            configuration = self._zero_area_configuration(area_no)
            candidate_reduced_cost = self._configuration_reduced_cost(
                configuration,
                duals,
                objective_mode,
                decisions,
            )
            return self._nested_area_result(
                state,
                configuration,
                candidate_reduced_cost,
                reduced_cost_lower_bound,
                nested_records,
                closed,
                started,
                certified_nonnegative=True,
            )

        remaining = nested_deadline - perf_counter()
        if remaining <= 1e-6:
            return None
        coordination_mip = self._build_nested_coordination_mip(
            state,
            duals,
            objective_mode,
            decisions,
            time_limit=remaining,
        )
        try:
            if coordination_mip["solution_count"] <= 0:
                return None
            configuration = self._combine_nested_block_configurations(
                state, coordination_mip["selected"]
            )
            candidate_reduced_cost = self._configuration_reduced_cost(
                configuration,
                duals,
                objective_mode,
                decisions,
            )
            expected = (
                float(coordination_mip["objective"])
                - convexity_dual
            )
            if abs(expected - candidate_reduced_cost) > 1e-6 * (
                1.0 + abs(candidate_reduced_cost)
            ):
                raise RuntimeError(
                    "nested area coordination objective mismatch: "
                    f"area={area_no}, model={expected}, "
                    f"configuration={candidate_reduced_cost}"
                )
        finally:
            self._free_gurobi_model(coordination_mip["model"])

        certified_nonnegative = (
            reduced_cost_lower_bound >= -tolerance
        )
        negative_candidate = candidate_reduced_cost < -tolerance
        duplicate_candidate = (
            self._configuration_identity(configuration)
            in self._area_configuration_index_by_identity
        )
        if not certified_nonnegative and (
            not negative_candidate or duplicate_candidate
        ):
            return None
        return self._nested_area_result(
            state,
            configuration,
            candidate_reduced_cost,
            reduced_cost_lower_bound,
            nested_records,
            closed,
            started,
            certified_nonnegative=certified_nonnegative,
        )

    def _price_area(
        self,
        area_no: str,
        duals: dict[tuple[str, object], float],
        objective_mode: str,
        time_limit: float,
        decisions: tuple[BranchDecision, ...] = (),
        *,
        stop_after_negative: bool = False,
    ) -> dict:
        profile = self._area_pricing_profile(area_no)
        started = perf_counter()
        if profile.strategy == "adaptive_block_guided_multicolumn":
            nested = self._price_area_nested(
                area_no,
                duals,
                objective_mode,
                time_limit,
                decisions,
            )
            if nested is not None:
                return nested
        remaining = max(0.01, float(time_limit) - (perf_counter() - started))
        result = self._price_area_mip(
            area_no,
            duals,
            objective_mode,
            remaining,
            decisions,
            stop_after_negative=stop_after_negative,
        )
        if profile.strategy == "adaptive_block_guided_multicolumn":
            result["pricing_strategy"] = (
                "nested_pricing_with_complete_area_mip_fallback"
            )
            result["nested_fallback_seconds"] = perf_counter() - started - float(
                result["seconds"]
            )
        return result

    def _master_coefficient_terms(
        self,
        configuration: AreaConfiguration,
        constraints: dict,
        decisions: tuple[BranchDecision, ...] = (),
    ) -> list[tuple[float, object]]:
        terms: list[tuple[float, object]] = [
            (1.0, constraints["area_convexity"][configuration.area_no])
        ]
        terms.extend(
            (float(quantity), constraints["group_demand"][group_id])
            for group_id, quantity in configuration.group_quantities
        )
        terms.extend(
            (float(quantity), constraints["export_guidance"][key])
            for key, quantity in configuration.export_guidance_quantities
        )
        terms.extend(
            (float(quantity), constraints["import_total"][key])
            for key, quantity in configuration.import_total_quantities
        )
        terms.extend(
            (float(quantity), constraints["import_reference"][key])
            for key, quantity in configuration.import_reference_quantities
        )
        terms.extend(
            (
                float(
                    self._configuration_branch_coefficient(
                        configuration, decision
                    )
                ),
                constraints["branch_decision"][decision_index],
            )
            for decision_index, decision in enumerate(decisions)
            if self._configuration_branch_coefficient(
                configuration, decision
            )
        )
        return terms


    def _build_area_master(
        self,
        areas: tuple[str, ...],
        decisions: tuple[BranchDecision, ...] = (),
    ):
        from gurobipy import quicksum

        model = GurobiModel("area_configuration_root_master")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        self._set_gurobi_param(model, "Method", int(self.config.lp_method))
        model.setMinimize()
        variables: dict[str, dict] = {
            "configuration": {},
            "phase_one_artificial": {},
        }
        for index, configuration in enumerate(self._area_configurations):
            variables["configuration"][index] = model.addVar(
                lb=0.0,
                ub=float("inf"),
                vtype="C",
                obj=float(configuration.business_cost),
                name=f"area_configuration_{index}",
            )

        constraints: dict[str, dict] = defaultdict(dict)
        for area_no in areas:
            indices = [
                index
                for index, configuration in enumerate(
                    self._area_configurations
                )
                if configuration.area_no == area_no
            ]
            constraints["area_convexity"][area_no] = model.addConstr(
                quicksum(
                    variables["configuration"][index] for index in indices
                )
                == 1.0,
                name=f"area_convexity_{area_no}",
            )

        for group in self.groups:
            artificial = model.addVar(
                lb=0.0,
                ub=float(group.demand),
                name=f"phase_group_{group.group_id}",
            )
            variables["phase_one_artificial"][
                ("group", group.group_id)
            ] = artificial
            terms = [
                quantity * variables["configuration"][index]
                for index, configuration in enumerate(
                    self._area_configurations
                )
                for config_group_id, quantity in configuration.group_quantities
                if config_group_id == group.group_id
            ]
            constraints["group_demand"][group.group_id] = model.addConstr(
                quicksum(terms) + artificial == int(group.demand),
                name=f"area_master_group_{group.group_id}",
            )
        for key, required in sorted(self.import_total_by_flow_size.items()):
            artificial = model.addVar(
                lb=0.0,
                ub=float(required),
                name=f"phase_import_{self._key_name(key)}",
            )
            variables["phase_one_artificial"][("import", *key)] = artificial
            terms = [
                quantity * variables["configuration"][index]
                for index, configuration in enumerate(
                    self._area_configurations
                )
                for config_key, quantity in configuration.import_total_quantities
                if config_key == key
            ]
            constraints["import_total"][key] = model.addConstr(
                quicksum(terms) + artificial == int(required),
                name=f"area_master_import_{self._key_name(key)}",
            )

        for key in sorted(self._master_area_guidance_keys):
            positive = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"area_guide_pos_{self._key_name(key)}",
            )
            negative = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"area_guide_neg_{self._key_name(key)}",
            )
            target = self._area_size_target(*key)
            terms = [
                quantity * variables["configuration"][index]
                for index, configuration in enumerate(
                    self._area_configurations
                )
                for config_key, quantity in configuration.export_guidance_quantities
                if config_key == key
            ]
            constraints["export_guidance"][key] = model.addConstr(
                quicksum(terms) - target == positive - negative,
                name=f"area_master_guide_{self._key_name(key)}",
            )

        import_reference_keys = set(self.import_area_size_reference)
        import_reference_keys.update(
            (flow, area_no, size)
            for area_no in areas
            for flow, size in self.import_total_by_flow_size
            if any(
                self.bays[bay_key].area_no == area_no
                for bay_key, _capacity in self.import_reservation_candidates.get(
                    (flow, size), []
                )
            )
        )
        for key in sorted(import_reference_keys):
            positive = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"import_ref_pos_{self._key_name(key)}",
            )
            negative = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"import_ref_neg_{self._key_name(key)}",
            )
            target = int(self.import_area_size_reference.get(key, 0))
            terms = [
                quantity * variables["configuration"][index]
                for index, configuration in enumerate(
                    self._area_configurations
                )
                for config_key, quantity in configuration.import_reference_quantities
                if config_key == key
            ]
            constraints["import_reference"][key] = model.addConstr(
                quicksum(terms) - target == positive - negative,
                name=f"area_master_import_ref_{self._key_name(key)}",
            )

        objective_offset = -len(self.groups) * (
            self._area_activation_penalty()
            + self._row_activation_penalty()
        )
        variables["objective_offset"] = {
            "constant": model.addVar(
                lb=1.0,
                ub=1.0,
                obj=objective_offset,
                name="area_configuration_objective_offset",
            )
        }
        for decision_index, decision in enumerate(decisions):
            expression = quicksum(
                self._configuration_branch_coefficient(
                    configuration, decision
                )
                * variables["configuration"][index]
                for index, configuration in enumerate(
                    self._area_configurations
                )
                if self._configuration_branch_coefficient(
                    configuration, decision
                )
            )
            artificial = model.addVar(
                lb=0.0,
                name=f"phase_branch_{decision_index}",
            )
            variables["phase_one_artificial"][
                ("branch", decision_index)
            ] = artificial
            if decision.sense == "L":
                constraints["branch_decision"][decision_index] = (
                    model.addConstr(
                        expression
                        <= int(decision.rhs) + artificial,
                        name=f"area_branch_{decision_index}_le",
                    )
                )
            elif decision.sense == "G":
                constraints["branch_decision"][decision_index] = (
                    model.addConstr(
                        expression + artificial >= int(decision.rhs),
                        name=f"area_branch_{decision_index}_ge",
                    )
                )
            else:
                raise ValueError(f"unknown branch sense: {decision.sense}")
        model.update()
        self._initialize_phase_one_objective(model, variables)
        return model, variables, constraints

    def _add_configurations_to_master(
        self,
        model,
        variables: dict,
        constraints: dict,
        indices: list[int],
        objective_mode: str,
        decisions: tuple[BranchDecision, ...] = (),
    ) -> None:
        for index in indices:
            configuration = self._area_configurations[index]
            variable = model.addPricedVar(
                self._master_coefficient_terms(
                    configuration, constraints, decisions
                ),
                lb=0.0,
                ub=float("inf"),
                obj=(
                    0.0
                    if objective_mode == "phase_one"
                    else float(configuration.business_cost)
                ),
                name=f"area_configuration_{index}",
            )
            variables["configuration"][index] = variable
            if objective_mode == "phase_one":
                variables.setdefault("_business_objective_terms", []).append(
                    (variable, float(configuration.business_cost))
                )
        model.update()

    def _append_negative_result_configurations(
        self,
        result: dict,
        tolerance: float,
    ) -> list[int]:
        """Store every new negative-cost member returned by one pricing MIP."""
        indices: list[int] = []
        configurations = result.get("configurations", [])
        reduced_costs = result.get("configuration_reduced_costs", [])
        for configuration, reduced_cost in zip(
            configurations, reduced_costs, strict=True
        ):
            if float(reduced_cost) >= -tolerance:
                continue
            if (
                self._configuration_identity(configuration)
                in self._area_configuration_index_by_identity
            ):
                continue
            indices.append(self._append_area_configuration(configuration))
        return indices

    @staticmethod
    def _pricing_result_summary(result: dict) -> dict:
        return {
            key: value
            for key, value in result.items()
            if key not in {"configuration", "configurations"}
        }

    def _price_all_areas(
        self,
        areas: tuple[str, ...],
        duals: dict[tuple[str, object], float],
        objective_mode: str,
        deadline: float | None,
        decisions: tuple[BranchDecision, ...] = (),
        pricing_areas: tuple[str, ...] | None = None,
    ) -> tuple[dict, list[int]]:
        started = perf_counter()
        tolerance = max(1e-9, float(self.config.reduced_cost_tolerance))
        selected_areas = areas if pricing_areas is None else pricing_areas
        if len(set(selected_areas)) != len(selected_areas) or not set(
            selected_areas
        ).issubset(areas):
            raise ValueError("pricing_areas must be unique members of areas")
        results: list[dict] = []
        new_indices: list[int] = []
        for position, area_no in enumerate(selected_areas):
            remaining = self._seconds_until(deadline)
            if remaining is not None and remaining <= 1e-6:
                break
            remaining_areas = selected_areas[position:]
            weights = [
                (
                    float(self.area_pricing_config.complex_time_weight)
                    if self._area_pricing_profile(key).strategy
                    == "adaptive_block_guided_multicolumn"
                    else 1.0
                )
                for key in remaining_areas
            ]
            area_limit = (
                60.0
                if remaining is None
                else max(0.02, remaining * weights[0] / sum(weights))
            )
            result = self._price_area(
                area_no,
                duals,
                objective_mode,
                area_limit,
                decisions,
                stop_after_negative=selected_areas != areas,
            )
            results.append(result)
            new_indices.extend(
                self._append_negative_result_configurations(result, tolerance)
            )
        all_areas_priced = {
            str(result["area_no"]) for result in results
        } == set(areas)
        exact = all_areas_priced and all(
            bool(result["optimal"]) for result in results
        )
        bounds_available = all_areas_priced and all(
            math.isfinite(float(result["reduced_cost_lower_bound"]))
            for result in results
        )
        correction = (
            sum(
                min(0.0, float(result["reduced_cost_lower_bound"]))
                for result in results
            )
            if bounds_available
            else None
        )
        return {
            "pricing_mode": (
                "full_area_discovery_sweep"
                if selected_areas == areas
                else "active_area_discovery_sweep"
            ),
            "objective_mode": objective_mode,
            "all_areas_priced": all_areas_priced,
            "selected_area_count": len(selected_areas),
            "total_area_count": len(areas),
            "exact": exact,
            "bounds_available": bounds_available,
            "valid_lower_bound_correction": correction,
            "new_configurations": len(new_indices),
            "minimum_solution_reduced_cost": min(
                (
                    float(result["solution_reduced_cost"])
                    for result in results
                ),
                default=0.0,
            ),
            "minimum_reduced_cost_lower_bound": min(
                (
                    float(result["reduced_cost_lower_bound"])
                    for result in results
                ),
                default=-math.inf,
            ),
            "pricing_seconds": perf_counter() - started,
            "area_results": [
                self._pricing_result_summary(result)
                for result in results
            ],
        }, new_indices

    def _complete_targeted_area_certificate(
        self,
        pricing: dict,
        duals: dict[tuple[str, object], float],
        objective_mode: str,
        deadline: float | None,
        decisions: tuple[BranchDecision, ...] = (),
    ) -> tuple[dict, list[int]]:
        """Spend the remaining round on only the uncertified area models."""
        started = perf_counter()
        summaries = {
            str(result["area_no"]): dict(result)
            for result in pricing.get("area_results", [])
        }
        all_areas = self._configuration_areas()
        hard_areas = [
            area_no
            for area_no in all_areas
            if area_no not in summaries
            or not bool(summaries[area_no].get("optimal", False))
        ]
        hard_areas.sort(
            key=lambda area_no: (
                -int(
                    summaries.get(area_no, {}).get(
                        "candidate_count",
                        self._area_pricing_profile(area_no).candidate_count,
                    )
                ),
                area_no,
            )
        )
        new_indices: list[int] = []
        tolerance = max(1e-9, float(self.config.reduced_cost_tolerance))
        for position, area_no in enumerate(hard_areas):
            remaining = self._seconds_until(deadline)
            if remaining is not None and remaining <= 1e-6:
                break
            remaining_hard = hard_areas[position:]
            weights = [
                max(
                    1.0,
                    math.sqrt(
                        float(
                            summaries.get(key, {}).get(
                                "candidate_count",
                                self._area_pricing_profile(
                                    key
                                ).candidate_count,
                            )
                        )
                    ),
                )
                for key in remaining_hard
            ]
            area_limit = (
                60.0
                if remaining is None
                else max(0.02, remaining * weights[0] / sum(weights))
            )
            result = self._price_area(
                area_no,
                duals,
                objective_mode,
                area_limit,
                decisions,
            )
            new_indices.extend(
                self._append_negative_result_configurations(result, tolerance)
            )
            summaries[area_no] = self._pricing_result_summary(result)

        ordered_results = [
            summaries[key] for key in sorted(summaries)
        ]
        all_areas_priced = {
            str(result["area_no"]) for result in ordered_results
        } == set(all_areas)
        exact = all_areas_priced and all(
            bool(result.get("optimal", False))
            for result in ordered_results
        )
        bounds_available = all_areas_priced and all(
            math.isfinite(
                float(result.get("reduced_cost_lower_bound", -math.inf))
            )
            for result in ordered_results
        )
        correction = (
            sum(
                min(
                    0.0,
                    float(result["reduced_cost_lower_bound"]),
                )
                for result in ordered_results
            )
            if bounds_available
            else None
        )
        pricing.update(
            {
                "pricing_mode": "discovery_plus_targeted_exact_certificate",
                "all_areas_priced": all_areas_priced,
                "exact": exact,
                "bounds_available": bounds_available,
                "valid_lower_bound_correction": correction,
                "new_configurations": int(
                    pricing.get("new_configurations", 0)
                )
                + len(new_indices),
                "minimum_solution_reduced_cost": min(
                    (
                        float(result["solution_reduced_cost"])
                        for result in ordered_results
                    ),
                    default=0.0,
                ),
                "minimum_reduced_cost_lower_bound": min(
                    (
                        float(result["reduced_cost_lower_bound"])
                        for result in ordered_results
                    ),
                    default=-math.inf,
                ),
                "pricing_seconds": float(
                    pricing.get("pricing_seconds", 0.0)
                )
                + perf_counter()
                - started,
                "targeted_certificate_seconds": perf_counter() - started,
                "targeted_area_count": len(hard_areas),
                "selected_area_count": len(ordered_results),
                "total_area_count": len(all_areas),
                "area_results": ordered_results,
            }
        )
        return pricing, new_indices

    def _adaptive_pricing_diagnostics(
        self, areas: tuple[str, ...]
    ) -> dict:
        return {
            "direct_candidate_limit": int(
                self.area_pricing_config.direct_candidate_limit
            ),
            "complex_area_pool_size": int(
                self.area_pricing_config.complex_area_pool_size
            ),
            "simple_area_pool_size": int(
                self.area_pricing_config.simple_area_pool_size
            ),
            "complex_time_weight": float(
                self.area_pricing_config.complex_time_weight
            ),
            "certificate_time_fraction": float(
                self.area_pricing_config.certificate_time_fraction
            ),
            "full_sweep_frequency": int(
                self.area_pricing_config.full_sweep_frequency
            ),
            "nested_max_iterations": int(
                self.area_pricing_config.nested_max_iterations
            ),
            "nested_time_fraction": float(
                self.area_pricing_config.nested_time_fraction
            ),
            "selection_rule": (
                "candidate_count_above_limit_and_multiple_footprint_blocks"
            ),
            "equivalent_block_sharing": {
                "enabled": True,
                "detection": (
                    "canonical_physical_data_candidate_semantics_and_"
                    "gurobi_structure_fingerprint"
                ),
                "representative_solution_pool": (
                    "min_equivalent_class_size_and_complex_area_pool_size"
                ),
                "location_specific_branching": "sharing_disabled",
                "bound_role": "exact_equivalent_subproblem_reuse",
            },
            "profiles": [
                {
                    "area_no": profile.area_no,
                    "candidate_count": profile.candidate_count,
                    "footprint_block_count": len(profile.footprint_blocks),
                    "largest_block_candidate_count": (
                        profile.largest_block_candidate_count
                    ),
                    "strategy": profile.strategy,
                }
                for profile in (
                    self._area_pricing_profile(area_no)
                    for area_no in areas
                )
            ],
            "exactness": (
                "complex_area_bounds_come_from_exact_block_pricing_"
                "corrections;_a_nonnegative_nested_lp_bound_certifies_"
                "outer_pricing_closure;_ambiguous_cases_fall_back_to_"
                "the_complete_area_mip"
            ),
            "nested_states_built": len(self._nested_area_pricing_states),
            "nested_equivalent_block_classes": {
                state.area_no: {
                    "physical_block_count": len(state.blocks),
                    "equivalent_class_count": len(
                        state.equivalent_block_classes
                    ),
                    "class_sizes": [
                        len(block_class)
                        for block_class in state.equivalent_block_classes
                    ],
                }
                for state in self._nested_area_pricing_states.values()
            },
            "nested_block_configuration_count": sum(
                len(configurations)
                for state in self._nested_area_pricing_states.values()
                for configurations in state.block_configurations.values()
            ),
        }

    @staticmethod
    def _area_formulation_diagnostics() -> dict:
        return {
            "column": "complete_integer_area_configuration",
            "master_rows": [
                "area_convexity",
                "export_group_demand",
                "import_flow_size_total",
                "export_area_guidance_l1",
                "import_area_reference_l1",
            ],
            "local_constraints": [
                "bay_and_row_capacity",
                "size_and_paired_footprint",
                "stack_capacity",
                "bay_no_mix",
                "row_no_mix",
                "area_and_row_activation",
                "anonymous_import_capacity",
            ],
        }

    def _dispose_area_pricing_models(self) -> None:
        for pricing in self._area_pricing_models.values():
            self._free_gurobi_model(pricing.model)
        for state in self._nested_area_pricing_states.values():
            for pricing in state.block_models.values():
                self._free_gurobi_model(pricing.model)
            if state.coordination_lp_model is not None:
                self._free_gurobi_model(state.coordination_lp_model)


__all__ = [
    "AdaptiveAreaPricingConfig",
    "AreaConfiguration",
    "AreaConfigurationPlanner",
    "AreaPricingProfile",
]
