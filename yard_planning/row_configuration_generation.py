"""Physical-row configuration column generation and restricted master MIP.

The Dantzig--Wolfe blocks are physical row tracks ``(area, row_no)``.  A
configuration is a complete integer allocation on one track and therefore
enforces row capacity, size compatibility, footprint overlap, and row/bay
no-mix locally.  Demand, bay-wide resources, import reservation, and the
normalized business objective remain linking rows in the master.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from time import perf_counter

from .direct_milp import DirectMilpPlanner
from .gurobi_backend import GurobiModel
from .planner import (
    ColumnGenerationConfig,
    ColumnGenerationResult,
    PlacementColumn,
)

RowBlock = tuple[str, str]


@dataclass(frozen=True)
class RowConfiguration:
    """One integer-feasible allocation for a physical row track."""

    configuration_id: str
    block: RowBlock
    allocations: tuple[tuple[int, int], ...]
    business_cost: float
    coefficients: tuple[tuple[str, object, float], ...]


@dataclass
class _PricingModel:
    block: RowBlock
    model: GurobiModel
    candidate_indices: tuple[int, ...]
    variables: dict[int, object]
    coefficient_maps: dict[int, dict[tuple[str, object], float]]
    build_seconds: float


class RowConfigurationGenerationPlanner(DirectMilpPlanner):
    """Generate complete physical-row configurations, then solve their MIP."""

    def __init__(self, problem, config: ColumnGenerationConfig | None = None):
        super().__init__(problem, config)
        self._row_blocks: tuple[RowBlock, ...] = ()
        self._candidate_indices_by_block: dict[RowBlock, tuple[int, ...]] = {}
        self._configurations: list[RowConfiguration] = []
        self._configuration_index: dict[tuple, int] = {}
        self._configuration_indices_by_block: defaultdict[
            RowBlock, list[int]
        ] = defaultdict(list)
        self._pricing_models: dict[RowBlock, _PricingModel] = {}

    @staticmethod
    def _anchor_row(column: PlacementColumn) -> str:
        return str(
            next(
                row_no
                for bay_key, row_no, _quantity in column.row_allocation
                if bay_key == column.bay_key
            )
        )

    def _prepare_row_configuration_data(self) -> None:
        self._prepare_master_index_sets()
        self._prepare_objective_normalization()
        self._initialize_location_pool()
        block_indices: defaultdict[RowBlock, list[int]] = defaultdict(list)
        for group in self.groups:
            for candidate in self._base_placements_for_group(group):
                index = self._append_generated_column(candidate)
                block_indices[
                    (candidate.area_no, self._anchor_row(candidate))
                ].append(index)
        self._row_blocks = tuple(sorted(block_indices))
        self._candidate_indices_by_block = {
            block: tuple(indices)
            for block, indices in sorted(block_indices.items())
        }
        if not self._row_blocks:
            raise ValueError("row-configuration model has no physical blocks")

    @staticmethod
    def _configuration_identity(
        block: RowBlock, allocations: tuple[tuple[int, int], ...]
    ) -> tuple:
        return block, allocations

    def _pattern_coefficients(
        self, allocations: tuple[tuple[int, int], ...]
    ) -> tuple[tuple[str, object, float], ...]:
        values: defaultdict[str, Counter[object]] = defaultdict(Counter)
        for index, quantity in allocations:
            column = self._columns[index]
            amount = int(quantity)
            values["group_demand_balance"][column.group_id] += amount
            values["group_area_activation"][(
                column.group_key,
                column.area_no,
            )] += amount
            values["group_row_activation"][(
                column.group_key,
                column.bay_key,
                self._anchor_row(column),
            )] += amount
            values["group_used_upper"][column.group_key] += amount
            values["group_used_lower"][column.group_key] -= amount
            for section, coefficients in (
                self._placement_master_coefficients(column).items()
            ):
                if section in {
                    "row_capacity_limit",
                    "row_size_limit",
                    "row_attr_link",
                }:
                    continue
                for key, coefficient in coefficients.items():
                    values[section][key] += float(coefficient) * amount
        return tuple(
            (section, key, float(value))
            for section in sorted(values)
            for key, value in sorted(
                values[section].items(), key=lambda item: repr(item[0])
            )
            if abs(float(value)) > 0.0
        )

    def _configuration_from_allocations(
        self,
        block: RowBlock,
        allocations: dict[int, int] | Counter[int],
    ) -> RowConfiguration:
        normalized = tuple(
            sorted(
                (int(index), int(quantity))
                for index, quantity in allocations.items()
                if int(quantity) > 0
            )
        )
        return RowConfiguration(
            configuration_id="",
            block=block,
            allocations=normalized,
            business_cost=float(
                sum(
                    int(quantity) * float(self._columns[index].intrinsic_cost)
                    for index, quantity in normalized
                )
            ),
            coefficients=self._pattern_coefficients(normalized),
        )

    def _ensure_configuration(
        self, configuration: RowConfiguration
    ) -> tuple[int, bool]:
        identity = self._configuration_identity(
            configuration.block, configuration.allocations
        )
        existing = self._configuration_index.get(identity)
        if existing is not None:
            return int(existing), False
        index = len(self._configurations)
        stored = replace(
            configuration,
            configuration_id=f"RC{index + 1:07d}",
        )
        self._configurations.append(stored)
        self._configuration_index[identity] = index
        self._configuration_indices_by_block[stored.block].append(index)
        return index, True

    def _single_location_seeds(self) -> None:
        """Seed each group with a few quantity-complete row alternatives."""
        candidates_by_group: defaultdict[str, list[int]] = defaultdict(list)
        for index, column in enumerate(self._columns):
            candidates_by_group[column.group_id].append(index)
        for group in self.groups:
            ranked = sorted(
                candidates_by_group[group.group_id],
                key=lambda index: (
                    float(self._columns[index].intrinsic_cost),
                    self._columns[index].area_no,
                    self._columns[index].bay_key,
                    self._anchor_row(self._columns[index]),
                ),
            )
            distinct_blocks: set[RowBlock] = set()
            for index in ranked:
                column = self._columns[index]
                block = (column.area_no, self._anchor_row(column))
                if block in distinct_blocks:
                    continue
                capacity = min(
                    int(group.demand),
                    int(self._base_location_capacity(group, column)),
                )
                for quantity in sorted({1, max(1, capacity)}):
                    self._ensure_configuration(
                        self._configuration_from_allocations(
                            block, {index: quantity}
                        )
                    )
                distinct_blocks.add(block)
                if len(distinct_blocks) >= 3:
                    break

    def _add_local_attribute_compatibility(
        self,
        model,
        quicksum,
        variables: dict[int, object],
        items_by_key: dict[object, list[tuple[int, float]]],
        *,
        row_level: bool,
    ) -> None:
        uses_by_scope: defaultdict[tuple, list] = defaultdict(list)
        for position, (key, items) in enumerate(
            sorted(items_by_key.items(), key=lambda item: repr(item[0]))
        ):
            if not items:
                continue
            use = model.addVar(
                vtype="B",
                name=("local_row_attr_" if row_level else "local_bay_attr_")
                + str(position),
            )
            local_upper = sum(
                float(coefficient)
                * float(
                    self._base_location_capacity(
                        self.groups_by_id[self._columns[index].group_id],
                        self._columns[index],
                    )
                )
                for index, coefficient in items
            )
            model.addConstr(
                quicksum(
                    coefficient * variables[index]
                    for index, coefficient in items
                )
                <= max(1.0, local_upper) * use,
                name=("local_row_attr_link_" if row_level else "local_bay_attr_link_")
                + str(position),
            )
            if row_level:
                bay_key, row_no, attr, scope, _value = key
                uses_by_scope[(bay_key, row_no, attr, scope)].append(use)
            else:
                bay_key, attr, scope, _value = key
                uses_by_scope[(bay_key, attr, scope)].append(use)
        for position, uses in enumerate(uses_by_scope.values()):
            model.addConstr(
                quicksum(uses) <= 1,
                name=("local_row_attr_one_" if row_level else "local_bay_attr_one_")
                + str(position),
            )

    def _build_pricing_model(self, block: RowBlock) -> _PricingModel:
        from gurobipy import quicksum

        started = perf_counter()
        area_no, row_no = block
        candidate_indices = self._candidate_indices_by_block[block]
        model = GurobiModel(
            f"row_configuration_price_{area_no}_{self._key_name((row_no,))}"
        )
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        self._set_gurobi_param(model, "MIPGap", 0.0)
        model.setMinimize()
        variables = {
            index: model.addVar(
                lb=0.0,
                ub=float(
                    min(
                        int(self.group_demand[self._columns[index].group_id]),
                        int(
                            self._base_location_capacity(
                                self.groups_by_id[
                                    self._columns[index].group_id
                                ],
                                self._columns[index],
                            )
                        ),
                    )
                ),
                vtype="I",
                name=f"price_x_{index}",
            )
            for index in candidate_indices
        }
        local_rows: defaultdict[
            str, defaultdict[object, list[tuple[int, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        coefficient_maps: dict[
            int, dict[tuple[str, object], float]
        ] = {}
        group_indices: defaultdict[str, list[int]] = defaultdict(list)
        for index in candidate_indices:
            column = self._columns[index]
            group_indices[column.group_id].append(index)
            unit_coefficients = self._pattern_coefficients(((index, 1),))
            coefficient_maps[index] = {
                (section, key): float(value)
                for section, key, value in unit_coefficients
            }
            for section, values in (
                self._placement_master_coefficients(column).items()
            ):
                if section not in {
                    "row_capacity_limit",
                    "row_size_limit",
                    "row_attr_link",
                    "bay_attr_link",
                }:
                    continue
                for key, coefficient in values.items():
                    if coefficient:
                        local_rows[section][key].append(
                            (index, float(coefficient))
                        )

        for key, items in sorted(local_rows["row_capacity_limit"].items()):
            bay_key, physical_row = key
            model.addConstr(
                quicksum(
                    coefficient * variables[index]
                    for index, coefficient in items
                )
                <= int(
                    self.bays[bay_key].row_physical_capacity.get(
                        physical_row,
                        self.bays[bay_key].physical_capacity,
                    )
                ),
                name=f"local_row_cap_{self._key_name(key)}",
            )
        for key, items in sorted(local_rows["row_size_limit"].items()):
            bay_key, physical_row, size = key
            model.addConstr(
                quicksum(
                    coefficient * variables[index]
                    for index, coefficient in items
                )
                <= int(
                    self.bays[bay_key].row_cap_by_size.get(size, {}).get(
                        physical_row,
                        self.bays[bay_key].cap_by_size.get(size, 0),
                    )
                ),
                name=f"local_row_size_{self._key_name(key)}",
            )
        for group_id, indices in sorted(group_indices.items()):
            model.addConstr(
                quicksum(variables[index] for index in indices)
                <= int(self.group_demand[group_id]),
                name=f"local_group_{group_id}",
            )
        self._add_local_attribute_compatibility(
            model,
            quicksum,
            variables,
            local_rows["bay_attr_link"],
            row_level=False,
        )
        self._add_local_attribute_compatibility(
            model,
            quicksum,
            variables,
            local_rows["row_attr_link"],
            row_level=True,
        )
        model.update()
        return _PricingModel(
            block=block,
            model=model,
            candidate_indices=candidate_indices,
            variables=variables,
            coefficient_maps=coefficient_maps,
            build_seconds=perf_counter() - started,
        )

    def _ensure_pricing_models(self) -> None:
        for block in self._row_blocks:
            if block not in self._pricing_models:
                self._pricing_models[block] = self._build_pricing_model(block)

    def _price_block(
        self,
        pricing: _PricingModel,
        duals: dict[tuple[str, object], float],
        phase: str,
        allowance: float,
        batch_size: int,
    ) -> dict:
        model = pricing.model
        for index, variable in pricing.variables.items():
            coefficient = (
                float(self._columns[index].intrinsic_cost)
                if phase == "business"
                else 0.0
            )
            coefficient -= sum(
                float(value) * float(duals.get(key, 0.0))
                for key, value in pricing.coefficient_maps[index].items()
            )
            model.setVarObjective(variable, coefficient)
        convexity_dual = float(
            duals.get(("block_convexity", pricing.block), 0.0)
        )
        self._set_gurobi_param(model, "TimeLimit", max(0.01, allowance))
        self._set_gurobi_param(model, "PoolSearchMode", 2)
        self._set_gurobi_param(model, "PoolSolutions", max(1, batch_size))
        model.update()
        started = perf_counter()
        model.optimize()
        seconds = perf_counter() - started
        status = self._gurobi_status_name(model)
        solution_count = self._gurobi_solution_count(model)
        best_bound = self._gurobi_dual_bound(model) - convexity_dual
        configurations: list[tuple[RowConfiguration, float]] = []
        for solution_number in range(min(solution_count, batch_size)):
            allocations = {
                index: int(round(model.getPoolValue(variable, solution_number)))
                for index, variable in pricing.variables.items()
                if model.getPoolValue(variable, solution_number) > 0.5
            }
            if not allocations:
                continue
            reduced_cost = (
                float(model.getPoolObjective(solution_number))
                - convexity_dual
            )
            configurations.append(
                (
                    self._configuration_from_allocations(
                        pricing.block, allocations
                    ),
                    reduced_cost,
                )
            )
        return {
            "block": pricing.block,
            "status": status,
            "optimal": status == "optimal",
            "seconds": seconds,
            "lower_bound_reduced_cost": best_bound,
            "configurations": configurations,
        }

    def _enrich_group_consolidation(self, deadline: float) -> dict:
        """Add a small, scale-free set of integer-friendly pure-group rows."""
        blocks_by_group: defaultdict[str, dict[RowBlock, float]] = defaultdict(dict)
        for block, indices in self._candidate_indices_by_block.items():
            for index in indices:
                column = self._columns[index]
                current = blocks_by_group[column.group_id].get(block, math.inf)
                blocks_by_group[column.group_id][block] = min(
                    current, float(column.intrinsic_cost)
                )
        attempted = 0
        added_count = 0
        seconds = 0.0
        block_limit = max(
            2,
            min(
                6,
                int(math.ceil(math.log2(max(2, len(self._row_blocks))))),
            ),
        )
        for group in self.groups:
            ranked_blocks = sorted(
                blocks_by_group[group.group_id],
                key=lambda block: (
                    blocks_by_group[group.group_id][block],
                    repr(block),
                ),
            )[:block_limit]
            for block in ranked_blocks:
                remaining = deadline - perf_counter()
                if remaining <= 0.02:
                    return {
                        "attempted": attempted,
                        "added": added_count,
                        "seconds": seconds,
                        "complete": False,
                    }
                pricing = self._pricing_models[block]
                original_bounds = {
                    index: float(variable.UB)
                    for index, variable in pricing.variables.items()
                }
                for index, variable in pricing.variables.items():
                    if self._columns[index].group_id == group.group_id:
                        pricing.model.setVarObjective(
                            variable,
                            -1.0
                            + 1e-6
                            * float(self._columns[index].intrinsic_cost),
                        )
                    else:
                        variable.UB = 0.0
                        pricing.model.setVarObjective(variable, 0.0)
                self._set_gurobi_param(
                    pricing.model,
                    "TimeLimit",
                    min(0.25, max(0.02, remaining)),
                )
                self._set_gurobi_param(pricing.model, "PoolSearchMode", 0)
                started = perf_counter()
                pricing.model.update()
                pricing.model.optimize()
                elapsed = perf_counter() - started
                seconds += elapsed
                attempted += 1
                if self._gurobi_solution_count(pricing.model) > 0:
                    allocations = {
                        index: int(
                            round(self._gurobi_value(pricing.model, variable))
                        )
                        for index, variable in pricing.variables.items()
                        if self._gurobi_value(pricing.model, variable) > 0.5
                    }
                    if allocations:
                        _index, added = self._ensure_configuration(
                            self._configuration_from_allocations(
                                block, allocations
                            )
                        )
                        added_count += int(added)
                for index, variable in pricing.variables.items():
                    variable.UB = original_bounds[index]

        return {
            "attempted": attempted,
            "added": added_count,
            "seconds": seconds,
            "complete": True,
        }

    def _master_linking_keys(self) -> dict[str, dict]:
        operational_demand: Counter[tuple[str, ...]] = Counter()
        area_capacity: Counter[tuple[tuple[str, ...], str]] = Counter()
        row_capacity: Counter[tuple[tuple[str, ...], str, str]] = Counter()
        for group in self.groups:
            operational_demand[self._operational_group_key(group)] += int(
                group.demand
            )
        for index, column in enumerate(self._columns):
            capacity = int(
                self._base_location_capacity(
                    self.groups_by_id[column.group_id], column
                )
            )
            area_capacity[(column.group_key, column.area_no)] += capacity
            row_capacity[(
                column.group_key,
                column.bay_key,
                self._anchor_row(column),
            )] += capacity
        return {
            "operational_demand": dict(operational_demand),
            "area_upper": {
                key: min(int(operational_demand[key[0]]), int(value))
                for key, value in area_capacity.items()
            },
            "row_upper": {
                key: min(int(operational_demand[key[0]]), int(value))
                for key, value in row_capacity.items()
            },
        }

    def _add_configuration_variable(
        self,
        model,
        variables: dict,
        constraints: dict[str, dict],
        configuration_index: int,
        *,
        relax: bool,
        phase: str,
    ) -> object:
        configuration = self._configurations[configuration_index]
        terms = [
            (1.0, constraints["block_convexity"][configuration.block])
        ]
        for section, key, coefficient in configuration.coefficients:
            row = constraints.get(section, {}).get(key)
            if row is not None:
                terms.append((float(coefficient), row))
        variable = model.addPricedVar(
            terms,
            lb=0.0,
            ub=1.0,
            vtype="C" if relax else "B",
            obj=(
                float(configuration.business_cost)
                if phase == "business"
                else 0.0
            ),
            name=f"lambda_{configuration.configuration_id}",
        )
        variables["configuration"][configuration_index] = variable
        if phase == "phase_one":
            variables["_business_objective_terms"].append(
                (variable, float(configuration.business_cost))
            )
        return variable

    def _build_configuration_master(
        self,
        *,
        relax: bool,
        phase_one: bool,
    ):
        from gurobipy import quicksum

        model = GurobiModel(
            "row_configuration_lp" if relax else "row_configuration_mip"
        )
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        model.setMinimize()
        integer_type = "C" if relax else "I"
        binary_type = "C" if relax else "B"
        constraints: dict[str, dict] = defaultdict(dict)
        variables: dict = {
            "configuration": {},
            "phase_one_artificial": {},
            "_business_objective_terms": [],
        }
        import_reserve = {
            (flow, size, bay_key): model.addVar(
                lb=0.0,
                ub=float(capacity),
                vtype=integer_type,
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

        for group in self.groups:
            if phase_one:
                shortage = model.addVar(
                    lb=0.0, obj=1.0, name=f"shortage_{group.group_id}"
                )
                excess = model.addVar(
                    lb=0.0, obj=1.0, name=f"excess_{group.group_id}"
                )
                variables["phase_one_artificial"][(
                    group.group_id,
                    "shortage",
                )] = shortage
                variables["phase_one_artificial"][(
                    group.group_id,
                    "excess",
                )] = excess
                expression = shortage - excess
            else:
                expression = quicksum([])
            constraints["group_demand_balance"][group.group_id] = (
                model.addConstr(
                    expression == int(group.demand),
                    name=f"group_demand_{group.group_id}",
                )
            )

        import_by_bay: defaultdict[str, list] = defaultdict(list)
        import_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        import_by_flow_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        import_by_flow_area_size: defaultdict[tuple[str, str, str], list] = (
            defaultdict(list)
        )
        for (flow, size, bay_key), variable in import_reserve.items():
            area_no = self.bays[bay_key].area_no
            for footprint_key in self._placement_footprint_keys(bay_key, size):
                import_by_bay[footprint_key].append(variable)
            import_by_bay_size[(bay_key, size)].append(variable)
            import_by_flow_size[(flow, size)].append(variable)
            import_by_flow_area_size[(flow, area_no, size)].append(variable)

        for bay_key in sorted(self._master_bay_capacity_keys):
            constraints["bay_capacity_limit"][bay_key] = model.addConstr(
                quicksum(import_by_bay.get(bay_key, []))
                <= int(self.bays[bay_key].physical_capacity),
                name=f"bay_cap_{self._key_name((bay_key,))}",
            )
        for key in sorted(self._master_bay_size_keys):
            bay_key, size = key
            constraints["bay_size_limit"][key] = model.addConstr(
                quicksum(import_by_bay_size.get(key, []))
                <= int(self.bays[bay_key].cap_by_size.get(size, 0)),
                name=f"bay_size_{self._key_name(key)}",
            )

        stack_variables_by_bay_size: defaultdict[tuple[str, str], list] = (
            defaultdict(list)
        )
        for key in sorted(self._master_stack_keys):
            bay_key, _port, size = key
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
                vtype=integer_type,
                name=f"stack_{self._key_name(key)}",
            )
            constraints["bay_port_stack_link"][key] = model.addConstr(
                -unit_capacity * stack <= 0.0,
                name=f"stack_load_{self._key_name(key)}",
            )
            stack_variables_by_bay_size[(bay_key, size)].append(stack)
        for key, stacks in stack_variables_by_bay_size.items():
            constraints["bay_stack_total_limit"][key] = model.addConstr(
                quicksum(stacks) <= self._stack_count_for_bay_size(*key),
                name=f"stack_total_{self._key_name(key)}",
            )

        for key, required in sorted(self.import_total_by_flow_size.items()):
            candidates = import_by_flow_size.get(key, [])
            if not candidates:
                raise ValueError(
                    "import capacity reservation has no compatible row-"
                    f"configuration bay: key={key}, required={required}"
                )
            constraints["import_total_balance"][key] = model.addConstr(
                quicksum(candidates) == int(required),
                name=f"import_total_{self._key_name(key)}",
            )

        for key in sorted(self._master_area_guidance_keys):
            voyage_id, flow, area_no, big_size = key
            target = self._area_size_target(
                voyage_id, flow, area_no, big_size
            )
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
            constraints["area_guidance_balance"][key] = model.addConstr(
                -positive + negative == target,
                name=f"guide_balance_{self._key_name(key)}",
            )

        linking = self._master_linking_keys()
        area_uses_by_group: defaultdict[tuple[str, ...], list] = defaultdict(list)
        row_uses_by_group: defaultdict[tuple[str, ...], list] = defaultdict(list)
        for key, upper in sorted(
            linking["area_upper"].items(), key=lambda item: repr(item[0])
        ):
            group_key, area_no = key
            use = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=binary_type,
                obj=self._area_activation_penalty(),
                name=(
                    f"use_group_area_{self._key_name(group_key)}_{area_no}"
                ),
            )
            constraints["group_area_activation"][key] = model.addConstr(
                -max(1, int(upper)) * use <= 0.0,
                name=f"group_area_link_{self._key_name((*group_key, area_no))}",
            )
            area_uses_by_group[group_key].append(use)
        for key, upper in sorted(
            linking["row_upper"].items(), key=lambda item: repr(item[0])
        ):
            group_key, bay_key, row_no = key
            use = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=binary_type,
                obj=self._row_activation_penalty(),
                name=(
                    f"use_group_row_{self._key_name(group_key)}_"
                    f"{self._key_name((bay_key, row_no))}"
                ),
            )
            constraints["group_row_activation"][key] = model.addConstr(
                -max(1, int(upper)) * use <= 0.0,
                name=(
                    "group_row_link_"
                    f"{self._key_name((*group_key, bay_key, row_no))}"
                ),
            )
            row_uses_by_group[group_key].append(use)
        for group_key, demand in sorted(
            linking["operational_demand"].items(),
            key=lambda item: repr(item[0]),
        ):
            used = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=binary_type,
                obj=-(
                    self._area_activation_penalty()
                    + self._row_activation_penalty()
                ),
                name=f"group_used_{self._key_name(group_key)}",
            )
            constraints["group_used_upper"][group_key] = model.addConstr(
                -max(1, int(demand)) * used <= 0.0,
                name=f"group_used_upper_{self._key_name(group_key)}",
            )
            constraints["group_used_lower"][group_key] = model.addConstr(
                used <= 0.0,
                name=f"group_used_lower_{self._key_name(group_key)}",
            )

        constraints["import_reference_balance"] = (
            self._add_import_reference_deviation(
                quicksum,
                model,
                import_by_flow_area_size,
                objective_mode="full",
            )
        )

        bay_uses_by_scope: defaultdict[tuple[str, str, str], list] = (
            defaultdict(list)
        )
        for key in sorted(self._master_bay_attr_choice_keys):
            bay_key, attr, scope, _value = key
            use = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=binary_type,
                name=f"bay_use_{self._key_name(key)}",
            )
            constraints["bay_attr_link"][key] = model.addConstr(
                -self._master_bay_attr_big_m[key] * use <= 0.0,
                name=f"bay_attr_link_{self._key_name(key)}",
            )
            bay_uses_by_scope[(bay_key, attr, scope)].append(use)
        for key, uses in bay_uses_by_scope.items():
            constraints["bay_attr_one"][key] = model.addConstr(
                quicksum(uses) <= 1,
                name=f"bay_attr_one_{self._key_name(key)}",
            )

        for block in self._row_blocks:
            constraints["block_convexity"][block] = model.addConstr(
                quicksum([]) <= 1.0,
                name=f"block_{self._key_name(block)}",
            )
        model.update()
        phase = "phase_one" if phase_one else "business"
        if phase_one:
            artificial_ids = {
                id(variable)
                for variable in variables["phase_one_artificial"].values()
            }
            variables["_business_objective_terms"] = [
                (variable, model.getVarObjective(variable))
                for variable in model.getVars()
                if abs(model.getVarObjective(variable)) > 0.0
                and id(variable) not in artificial_ids
            ]
            for variable, _coefficient in variables[
                "_business_objective_terms"
            ]:
                model.setVarObjective(variable, 0.0)
        for configuration_index in range(len(self._configurations)):
            self._add_configuration_variable(
                model,
                variables,
                constraints,
                configuration_index,
                relax=relax,
                phase=phase,
            )
        model.update()
        return model, variables, constraints

    @staticmethod
    def _phase_one_value(model, variables: dict) -> float:
        return float(
            sum(
                model.getValue(variable)
                for variable in variables["phase_one_artificial"].values()
            )
        )

    @staticmethod
    def _activate_business_objective(model, variables: dict) -> None:
        for variable in variables["phase_one_artificial"].values():
            model.setVarObjective(variable, 0.0)
            variable.UB = 0.0
        for variable, coefficient in variables[
            "_business_objective_terms"
        ]:
            model.setVarObjective(variable, float(coefficient))
        model.update()

    def _configuration_raw_reduced_cost(
        self,
        configuration: RowConfiguration,
        duals: dict[tuple[str, object], float],
        phase: str,
    ) -> float:
        value = (
            float(configuration.business_cost)
            if phase == "business"
            else 0.0
        )
        value -= float(
            duals.get(("block_convexity", configuration.block), 0.0)
        )
        value -= sum(
            float(coefficient)
            * float(duals.get((section, key), 0.0))
            for section, key, coefficient in configuration.coefficients
        )
        return value

    def _solve_root_generation(self, deadline: float) -> dict:
        preparation_started = perf_counter()
        self._prepare_row_configuration_data()
        self._single_location_seeds()
        self._ensure_pricing_models()
        integer_enrichment = self._enrich_group_consolidation(deadline)
        preparation_seconds = perf_counter() - preparation_started
        if perf_counter() >= deadline:
            return {
                "status": "time_limit_during_preparation",
                "closed": False,
                "valid_lower_bound": 0.0,
                "objective": None,
                "iterations": [],
                "preparation_seconds": preparation_seconds,
                "master_build_seconds": 0.0,
            }
        master_started = perf_counter()
        model, variables, constraints = self._build_configuration_master(
            relax=True,
            phase_one=True,
        )
        master_build_seconds = perf_counter() - master_started
        registered_configuration_count = len(self._configurations)
        phase = "phase_one"
        iterations: list[dict] = []
        valid_lower_bound = 0.0
        closed = False
        status = "iteration_limit"
        last_business_objective = None
        tolerance = max(1e-9, float(self.config.reduced_cost_tolerance))
        try:
            for iteration in range(1, int(self.config.max_iterations) + 1):
                remaining = deadline - perf_counter()
                if remaining <= 1e-6:
                    status = "time_limit"
                    break
                self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
                self._set_gurobi_param(model, "Method", int(self.config.lp_method))
                model.optimize()
                master_status = self._gurobi_status_name(model)
                if master_status != "optimal":
                    status = f"master_{master_status}"
                    break
                master_objective = self._gurobi_objective_value(model)
                phase_one_value = self._phase_one_value(model, variables)
                if phase == "business":
                    last_business_objective = master_objective
                if phase == "phase_one" and phase_one_value <= tolerance:
                    self._activate_business_objective(model, variables)
                    phase = "business"
                    iterations.append(
                        {
                            "iteration": iteration,
                            "phase": "phase_one",
                            "master_objective": master_objective,
                            "phase_one_value": phase_one_value,
                            "columns_added": 0,
                            "transition": "business",
                        }
                    )
                    continue

                duals = self._master_dual_snapshot(model, constraints)
                block_results = []
                new_indices: list[int] = []
                negative_lower_bounds = 0.0
                all_optimal = True
                batch_size = min(
                    8,
                    max(2, int(math.ceil(math.sqrt(len(self.groups)) / 2))),
                )
                for position, block in enumerate(self._row_blocks):
                    remaining = deadline - perf_counter()
                    blocks_left = len(self._row_blocks) - position
                    if remaining <= 1e-6:
                        all_optimal = False
                        break
                    allowance = min(
                        0.75,
                        max(0.02, 0.80 * remaining / max(1, blocks_left)),
                    )
                    result = self._price_block(
                        self._pricing_models[block],
                        duals,
                        phase,
                        allowance,
                        batch_size,
                    )
                    block_results.append(result)
                    all_optimal = all_optimal and bool(result["optimal"])
                    if result["optimal"]:
                        negative_lower_bounds += min(
                            0.0,
                            float(result["lower_bound_reduced_cost"]),
                        )
                    for configuration, _reported_reduced_cost in result[
                        "configurations"
                    ]:
                        raw_reduced_cost = self._configuration_raw_reduced_cost(
                            configuration, duals, phase
                        )
                        if raw_reduced_cost >= -tolerance:
                            continue
                        configuration_index, added = self._ensure_configuration(
                            configuration
                        )
                        if added:
                            new_indices.append(configuration_index)

                for configuration_index in new_indices:
                    self._add_configuration_variable(
                        model,
                        variables,
                        constraints,
                        configuration_index,
                        relax=True,
                        phase=phase,
                    )
                if new_indices:
                    registered_configuration_count += len(new_indices)
                    model.update()
                lagrangian_bound = None
                if phase == "business" and all_optimal:
                    lagrangian_bound = float(
                        master_objective + negative_lower_bounds
                    )
                    valid_lower_bound = max(
                        valid_lower_bound, lagrangian_bound
                    )
                iterations.append(
                    {
                        "iteration": iteration,
                        "phase": phase,
                        "master_objective": master_objective,
                        "phase_one_value": phase_one_value,
                        "columns_added": len(new_indices),
                        "priced_block_count": len(block_results),
                        "all_pricing_optimal": all_optimal,
                        "minimum_reduced_cost": min(
                            (
                                float(result["lower_bound_reduced_cost"])
                                for result in block_results
                            ),
                            default=None,
                        ),
                        "lagrangian_lower_bound": lagrangian_bound,
                        "pricing_seconds": round(
                            sum(
                                float(result["seconds"])
                                for result in block_results
                            ),
                            4,
                        ),
                    }
                )
                if not new_indices:
                    if phase == "phase_one":
                        if all_optimal:
                            status = "phase_one_infeasible"
                            break
                        status = "pricing_time_limit"
                        break
                    if all_optimal:
                        closed = True
                        valid_lower_bound = master_objective
                        status = "optimal"
                    else:
                        status = "pricing_time_limit"
                    break
            objective = last_business_objective
            if self._gurobi_solution_count(model) > 0 and phase == "business":
                objective = self._gurobi_objective_value(model)
            return {
                "status": status,
                "closed": closed,
                "phase": phase,
                "valid_lower_bound": valid_lower_bound,
                "objective": objective,
                "iterations": iterations,
                "preparation_seconds": preparation_seconds,
                "master_build_seconds": master_build_seconds,
                "configuration_count": len(self._configurations),
                "registered_configuration_count": registered_configuration_count,
                "row_block_count": len(self._row_blocks),
                "pricing_model_build_seconds": sum(
                    pricing.build_seconds
                    for pricing in self._pricing_models.values()
                ),
                "pricing_candidate_count": sum(
                    len(pricing.candidate_indices)
                    for pricing in self._pricing_models.values()
                ),
                "integer_enrichment": integer_enrichment,
            }
        finally:
            self._free_gurobi_model(model)

    def solve_root_relaxation(self, time_limit: float | None = None) -> dict:
        started = perf_counter()
        limit = (
            float(time_limit)
            if time_limit is not None
            else float(self.config.total_time_limit)
        )
        if limit <= 0.0:
            limit = max(2.0, 2.0 * float(self.config.mip_time_limit))
        try:
            result = self._solve_root_generation(started + limit)
            result["total_seconds"] = perf_counter() - started
            return result
        finally:
            self._dispose_pricing_models()

    def _dispose_pricing_models(self) -> None:
        for pricing in self._pricing_models.values():
            self._free_gurobi_model(pricing.model)
        self._pricing_models.clear()

    def _solve_restricted_master(
        self, deadline: float
    ) -> tuple[Counter[int] | None, dict]:
        started = perf_counter()
        model, variables, _constraints = self._build_configuration_master(
            relax=False,
            phase_one=False,
        )
        try:
            build_seconds = perf_counter() - started
            remaining = deadline - perf_counter()
            if remaining <= 1e-6:
                return None, {
                    "status": "time_limit_during_build",
                    "build_seconds": build_seconds,
                    "seconds": perf_counter() - started,
                }
            self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
            self._set_gurobi_param(
                model, "MIPGap", max(0.0, float(self.config.mip_gap))
            )
            self._set_gurobi_param(model, "MIPFocus", 1)
            model.optimize()
            status = self._gurobi_status_name(model)
            if self._gurobi_solution_count(model) <= 0:
                return None, {
                    "status": status,
                    "build_seconds": build_seconds,
                    "seconds": perf_counter() - started,
                    "configuration_count": len(self._configurations),
                }
            selected_configurations = {
                index: int(round(self._gurobi_value(model, variable)))
                for index, variable in variables["configuration"].items()
                if self._gurobi_value(model, variable) > 0.5
            }
            selected: Counter[int] = Counter()
            for configuration_index, chosen in selected_configurations.items():
                if chosen != 1:
                    raise RuntimeError(
                        "row configuration multiplier must be binary"
                    )
                for index, quantity in self._configurations[
                    configuration_index
                ].allocations:
                    selected[index] += int(quantity)
            self._final_import_reservation = (
                self._gurobi_import_reservation_values(model, variables)
            )
            solver_objective = self._gurobi_objective_value(model)
            solver_bound = self._gurobi_dual_bound(model)
            return selected, {
                "status": status,
                "objective": solver_objective,
                "bound": solver_bound,
                "mip_gap": self._gurobi_gap(model),
                "build_seconds": build_seconds,
                "seconds": perf_counter() - started,
                "selected_configuration_count": len(
                    selected_configurations
                ),
                "configuration_count": len(self._configurations),
            }
        finally:
            self._free_gurobi_model(model)

    def solve(self) -> ColumnGenerationResult:
        started = perf_counter()
        total_limit = float(self.config.total_time_limit)
        if total_limit <= 0.0:
            total_limit = max(2.0, 2.0 * float(self.config.mip_time_limit))
        deadline = started + total_limit
        root_deadline = started + 0.70 * total_limit
        try:
            root = self._solve_root_generation(root_deadline)
        finally:
            self._dispose_pricing_models()
        selected, restricted = self._solve_restricted_master(deadline)
        if selected is None:
            raise RuntimeError(
                "row-configuration restricted master did not find a complete "
                f"allocation; root={root}, restricted={restricted}"
            )
        validation = self._validate_final_solution(selected)
        reconstructed = self._selected_solution_energy(selected)
        auxiliary_slack = self._absolute_deviation_auxiliary_slack(
            float(restricted["objective"]),
            reconstructed,
            context="row configuration restricted master",
        )
        lower_bound = min(
            reconstructed, float(root.get("valid_lower_bound", 0.0))
        )
        absolute_gap = max(0.0, reconstructed - lower_bound)
        relative_gap = absolute_gap / max(abs(reconstructed), 1e-12)
        diagnostics = {
            "algorithm": "physical_row_configuration_generation_gurobi",
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
            "formulation": "physical_row_track_configuration_master",
            "decomposition": "dantzig_wolfe_by_area_physical_row_track",
            "master_algorithm": "row_configuration_generation_and_rmp_mip",
            "master_status": restricted["status"],
            "master_bound_scope": (
                "closed_row_configuration_lp"
                if root.get("closed")
                else "lagrangian_row_configuration_bound"
            ),
            "master_objective": reconstructed,
            "master_mip_gap": relative_gap,
            "complete_model_lower_bound": lower_bound,
            "complete_model_absolute_gap": absolute_gap,
            "complete_model_relative_gap": relative_gap,
            "complete_model_gap_source": "row_configuration_pricing",
            "hard_demand_balance": True,
            "candidate_row_location_count": len(self._columns),
            "selected_location_count": len(selected),
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
            "area_guidance": {
                "source": "normalized_export_big_plan_new_qty",
                "role": "soft_spatial_reference_only",
                "target_boxes": int(sum(self.quota_by_key.values())),
            },
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
                "import_boxes": int(
                    sum(self.import_area_size_reference.values())
                ),
            },
            "row_configuration_root": root,
            "row_configuration_restricted_master": restricted,
            "row_configuration_objective_auxiliary_slack": auxiliary_slack,
            "row_configuration_validation": validation,
            "row_configuration_total_seconds": perf_counter() - started,
        }
        result = self._assemble_result(selected, diagnostics)
        result.columns = self._selected_direct_columns(selected)
        return result

    def solve_m0_lp_relaxation(
        self, time_limit: float | None = None
    ) -> dict:
        """Solve the complete compact M0 continuous relaxation for comparison."""
        from gurobipy import quicksum

        started = perf_counter()
        self._prepare_master_index_sets()
        self._prepare_objective_normalization()
        self._initialize_location_pool()
        for group in self.groups:
            for candidate in self._base_placements_for_group(group):
                self._append_generated_column(candidate)
        model, _variables, stats = self.build_compact_row_milp(
            self._columns, GurobiModel, quicksum
        )
        try:
            for variable in model.getVars():
                variable.VType = "C"
            model.update()
            build_seconds = perf_counter() - started
            limit = (
                float(time_limit)
                if time_limit is not None
                else float(self.config.total_time_limit)
            )
            remaining = limit - build_seconds
            if remaining <= 1e-6:
                return {
                    "status": "time_limit_during_build",
                    "objective": None,
                    "build_seconds": build_seconds,
                    "seconds": perf_counter() - started,
                    "model": stats,
                }
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
                "build_seconds": build_seconds,
                "seconds": perf_counter() - started,
                "model": stats,
            }
        finally:
            self._free_gurobi_model(model)


__all__ = ["RowConfiguration", "RowConfigurationGenerationPlanner"]
