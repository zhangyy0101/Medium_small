"""Exact node tree over adaptive area-configuration columns."""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import replace
from time import perf_counter

from .area_configuration import AreaConfigurationPlanner
from .direct_milp import DirectMilpPlanner
from .gurobi_backend import GurobiModel
from .planner import (
    BranchDecision,
    BranchPriceNode,
    ColumnGenerationResult,
    PlacementColumn,
)


class AreaConfigurationBranchPricePlanner(AreaConfigurationPlanner):
    """Branch-and-Price using original row and import allocation decisions."""

    def solve(self) -> ColumnGenerationResult:
        """Solve the paper algorithm and expand its incumbent to row output."""
        tree_result = self.solve_branch_and_price()
        selected_configurations = tree_result.get(
            "selected_configurations"
        )
        if selected_configurations is None:
            raise RuntimeError(
                "area-configuration Branch-and-Price ended without a "
                "feasible incumbent"
            )
        selected_locations = self._expand_area_configuration_selection(
            selected_configurations
        )
        diagnostics = self._paper_algorithm_diagnostics(tree_result)
        result = self._assemble_result(selected_locations, diagnostics)
        reconstructed = float(
            result.diagnostics["final_business_objective"]
        )
        reported = float(tree_result["objective"])
        if abs(reconstructed - reported) > 1e-7 * max(
            1.0, abs(reported)
        ):
            raise RuntimeError(
                "expanded row solution objective differs from the "
                "area-configuration incumbent: "
                f"expanded={reconstructed}, reported={reported}"
            )
        return result

    def _expand_area_configuration_selection(
        self,
        selected_configurations: dict[int, int],
    ) -> Counter[int]:
        """Convert one selected configuration per area to row locations."""
        self._columns.clear()
        self._column_keys.clear()
        self._final_import_reservation.clear()
        selected_locations: Counter[int] = Counter()
        for configuration_index, chosen in sorted(
            selected_configurations.items()
        ):
            if int(chosen) != 1:
                raise RuntimeError(
                    "area configuration selection must be binary"
                )
            configuration = self._area_configurations[
                int(configuration_index)
            ]
            for placement in configuration.placements:
                location_index = self._append_generated_column(placement)
                selected_locations[location_index] = 1
            for flow, size, bay_key, quantity in (
                configuration.import_reservations
            ):
                self._final_import_reservation[
                    (flow, size, bay_key)
                ] += int(quantity)
        return selected_locations

    def _paper_algorithm_diagnostics(self, tree_result: dict) -> dict:
        """Build the common output diagnostics for the paper algorithm."""
        areas = self._configuration_areas()
        return {
            "algorithm": (
                "nested_area_configuration_branch_and_price_gurobi"
            ),
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
            "planned_box_count": sum(
                group.demand for group in self.groups
            ),
            "berth_distance_count": len(self.problem.berth_distances),
            "berth_by_voyage": self.problem.berth_by_voyage,
            "demand_alignment": self.demand_stats,
            "base_feasible_placement_count": (
                self._base_feasible_placement_count
            ),
            "formulation": "integer_area_configuration_master",
            "decomposition": "nested_physical_block_area_pricing",
            "branch_and_price": True,
            "gurobi_available": True,
            "master_status": tree_result["status"],
            "master_bound_scope": "complete_branch_and_price_tree",
            "complete_model_lower_bound": tree_result[
                "global_lower_bound"
            ],
            "complete_model_absolute_gap": tree_result["absolute_gap"],
            "complete_model_relative_gap": tree_result["relative_gap"],
            "complete_model_gap_source": (
                "exact_area_pricing_and_branch_tree"
            ),
            "master_mip_gap": tree_result["relative_gap"],
            "root_lp_lower_bound": tree_result[
                "root_lower_bound"
            ],
            "root_lp_exact": bool(
                tree_result["node_records"]
                and tree_result["node_records"][0]["status"] == "optimal"
            ),
            "area_guidance": {
                "source": "normalized_export_big_plan_new_qty",
                "role": "soft_spatial_reference_only",
                "target_boxes": int(sum(self.quota_by_key.values())),
            },
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
                "accepted_big_plan_sizes": ["20", "40"],
            },
            "business_objective_normalization": {
                "weights": self._objective_weights(),
                "scales": dict(self._objective_scales),
                "weight_sum": round(
                    sum(self._objective_weights().values()), 10
                ),
                "method": "natural_instance_scale",
            },
            "business_objective": (
                self._business_objective_specification()
            ),
            "objective_coefficients": {
                "area_guidance_l1_unit": (
                    self._area_guidance_penalty()
                ),
                "area_activation_unit": (
                    self._area_activation_penalty()
                ),
                "row_activation_unit": self._row_activation_penalty(),
            },
            "big_m_tightening": self._big_m_diagnostics(),
            "adaptive_area_pricing": (
                self._adaptive_pricing_diagnostics(areas)
            ),
            "area_formulation": self._area_formulation_diagnostics(),
            "branch_and_price_statistics": {
                key: value
                for key, value in tree_result.items()
                if key != "selected_configurations"
            },
        }

    def _scheduled_pricing_areas(
        self,
        areas: tuple[str, ...],
        productive_areas: set[str],
        iteration: int,
    ) -> tuple[str, ...]:
        frequency = int(self.area_pricing_config.full_sweep_frequency)
        full_sweep = (
            iteration == 1
            or iteration % frequency == 0
            or not productive_areas
        )
        if full_sweep:
            return areas
        selected = tuple(
            area_no for area_no in areas if area_no in productive_areas
        )
        return selected or areas

    def _productive_pricing_areas(self, pricing: dict) -> set[str]:
        tolerance = max(
            1e-9, float(self.config.reduced_cost_tolerance)
        )
        return {
            str(result["area_no"])
            for result in pricing.get("area_results", [])
            if float(result.get("solution_reduced_cost", math.inf))
            < -tolerance
        }



    def _solve_area_node_lp(
        self,
        node: BranchPriceNode,
        areas: tuple[str, ...],
        deadline: float | None,
        *,
        allow_incumbent_heuristic: bool = True,
    ) -> dict:
        """Solve one node LP to certified area-pricing closure."""
        from gurobipy import quicksum

        started = perf_counter()
        model, variables, constraints = self._build_area_master(
            areas,
            node.decisions,
        )
        records: list[dict] = []
        phase_iterations = 0
        business_iterations = 0
        best_valid_lower_bound = max(0.0, float(node.inherited_bound))
        try:
            phase_complete = False
            productive_areas = set(areas)
            for iteration in range(1, int(self.config.max_iterations) + 1):
                if not self._set_node_time_limit(model, deadline):
                    return self._unfinished_node_result(
                        node,
                        "time_limit",
                        best_valid_lower_bound,
                        phase_iterations,
                        business_iterations,
                        records,
                        started,
                    )
                model.optimize()
                status = self._gurobi_status_name(model)
                if status != "optimal":
                    return self._unfinished_node_result(
                        node,
                        "time_limit" if status == "timelimit" else status,
                        best_valid_lower_bound,
                        phase_iterations,
                        business_iterations,
                        records,
                        started,
                    )
                phase_iterations = iteration
                artificial = sum(
                    self._gurobi_value(model, variable)
                    for variable in variables[
                        "phase_one_artificial"
                    ].values()
                )
                if artificial <= 1e-7:
                    phase_complete = True
                    break
                duals = self._master_dual_snapshot(model, constraints)
                pricing_areas = self._scheduled_pricing_areas(
                    areas, productive_areas, iteration
                )
                pricing, new_indices = self._price_all_areas(
                    areas,
                    duals,
                    "phase_one",
                    deadline,
                    node.decisions,
                    pricing_areas,
                )
                if not new_indices and not bool(pricing["exact"]):
                    pricing, certificate_indices = (
                        self._complete_targeted_area_certificate(
                            pricing,
                            duals,
                            "phase_one",
                            deadline,
                            node.decisions,
                        )
                    )
                    new_indices.extend(certificate_indices)
                productive_areas = self._productive_pricing_areas(pricing)
                pricing.update(
                    {
                        "stage": "phase_one",
                        "node_id": node.node_id,
                        "node_depth": node.depth,
                        "iteration": iteration,
                        "restricted_master_objective": artificial,
                    }
                )
                records.append(pricing)
                if new_indices:
                    self._add_configurations_to_master(
                        model,
                        variables,
                        constraints,
                        new_indices,
                        "phase_one",
                        node.decisions,
                    )
                    continue
                if pricing["exact"]:
                    return {
                        "status": "infeasible",
                        "node_id": node.node_id,
                        "valid_lower_bound": math.inf,
                        "phase_iterations": phase_iterations,
                        "business_iterations": 0,
                        "records": records,
                        "seconds": perf_counter() - started,
                    }
                return self._unfinished_node_result(
                    node,
                    "time_limit",
                    best_valid_lower_bound,
                    phase_iterations,
                    business_iterations,
                    records,
                    started,
                )
            if not phase_complete:
                return self._unfinished_node_result(
                    node,
                    "column_iteration_limit",
                    best_valid_lower_bound,
                    phase_iterations,
                    business_iterations,
                    records,
                    started,
                )

            constraints["phase_one_zero"][0] = model.addConstr(
                quicksum(variables["phase_one_artificial"].values()) == 0.0,
                name=f"area_phase_one_zero_node_{node.node_id}",
            )
            self._activate_master_objective(
                model, variables, objective="business"
            )
            model.update()
            incumbent_candidate: dict | None = None
            row_recombination_attempted = False
            productive_areas = set(areas)

            for iteration in range(1, int(self.config.max_iterations) + 1):
                if not self._set_node_time_limit(model, deadline):
                    return self._unfinished_node_result(
                        node,
                        "time_limit",
                        best_valid_lower_bound,
                        phase_iterations,
                        business_iterations,
                        records,
                        started,
                        incumbent_candidate,
                    )
                model.optimize()
                status = self._gurobi_status_name(model)
                if status != "optimal":
                    return self._unfinished_node_result(
                        node,
                        "time_limit" if status == "timelimit" else status,
                        best_valid_lower_bound,
                        phase_iterations,
                        business_iterations,
                        records,
                        started,
                        incumbent_candidate,
                    )
                business_iterations = iteration
                restricted_objective = self._gurobi_objective_value(model)
                duals = self._master_dual_snapshot(model, constraints)
                lp_configuration_values = {
                    int(index): self._gurobi_value(model, variable)
                    for index, variable in variables[
                        "configuration"
                    ].items()
                    if self._gurobi_value(model, variable) > 1e-7
                }
                pricing_areas = self._scheduled_pricing_areas(
                    areas, productive_areas, iteration
                )
                pricing, new_indices = self._price_all_areas(
                    areas,
                    duals,
                    "business",
                    deadline,
                    node.decisions,
                    pricing_areas,
                )
                remaining = self._seconds_until(deadline)
                proactive_certificate = (
                    remaining is not None
                    and remaining
                    <= max(
                        10.0,
                        float(
                            self.area_pricing_config.certificate_time_fraction
                        )
                        * float(self.config.total_time_limit),
                    )
                )
                if (
                    not bool(pricing["exact"])
                    and (not new_indices or proactive_certificate)
                ):
                    pricing, certificate_indices = (
                        self._complete_targeted_area_certificate(
                            pricing,
                            duals,
                            "business",
                            deadline,
                            node.decisions,
                        )
                    )
                    new_indices.extend(certificate_indices)
                productive_areas = self._productive_pricing_areas(pricing)
                pricing.update(
                    {
                        "stage": "business",
                        "node_id": node.node_id,
                        "node_depth": node.depth,
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
                    self._add_configurations_to_master(
                        model,
                        variables,
                        constraints,
                        new_indices,
                        "business",
                        node.decisions,
                    )
                remaining = self._seconds_until(deadline)
                pricing_closed = bool(pricing["exact"]) and not new_indices
                recombination_time_trigger = (
                    remaining is not None
                    and remaining
                    <= max(
                        2.0 * float(self.config.mip_time_limit),
                        0.65 * float(self.config.total_time_limit),
                    )
                )
                if (
                    allow_incumbent_heuristic
                    and node.node_id == 0
                    and not row_recombination_attempted
                    and (pricing_closed or recombination_time_trigger)
                ):
                    row_recombination_attempted = True
                    heuristic_limit = min(
                        float(self.config.mip_time_limit),
                        max(
                            2.0,
                            0.10 * float(self.config.total_time_limit),
                        ),
                    )
                    if remaining is not None:
                        heuristic_limit = min(
                            heuristic_limit, max(0.0, 0.15 * remaining)
                        )
                    previous_count = len(self._area_configurations)
                    incumbent_candidate = self._solve_area_row_recombination(
                        areas,
                        heuristic_limit,
                        lp_configuration_values,
                    )
                    heuristic_indices = list(
                        range(previous_count, len(self._area_configurations))
                    )
                    if heuristic_indices:
                        self._add_configurations_to_master(
                            model,
                            variables,
                            constraints,
                            heuristic_indices,
                            "business",
                            node.decisions,
                        )
                        new_indices.extend(heuristic_indices)
                if new_indices:
                    continue
                if pricing["exact"]:
                    configuration_values = {
                        int(index): self._gurobi_value(model, variable)
                        for index, variable in variables[
                            "configuration"
                        ].items()
                        if self._gurobi_value(model, variable) > 1e-9
                    }
                    return {
                        "status": "optimal",
                        "node_id": node.node_id,
                        "bound": restricted_objective,
                        "valid_lower_bound": restricted_objective,
                        "configuration_values": configuration_values,
                        "incumbent_candidate": incumbent_candidate,
                        "phase_iterations": phase_iterations,
                        "business_iterations": business_iterations,
                        "records": records,
                        "seconds": perf_counter() - started,
                    }
                return self._unfinished_node_result(
                    node,
                    "time_limit",
                    best_valid_lower_bound,
                    phase_iterations,
                    business_iterations,
                    records,
                    started,
                    incumbent_candidate,
                )
            return self._unfinished_node_result(
                node,
                "column_iteration_limit",
                best_valid_lower_bound,
                phase_iterations,
                business_iterations,
                records,
                started,
                incumbent_candidate,
            )
        finally:
            self._free_gurobi_model(model)

    @staticmethod
    def _unfinished_node_result(
        node: BranchPriceNode,
        status: str,
        valid_lower_bound: float,
        phase_iterations: int,
        business_iterations: int,
        records: list[dict],
        started: float,
        incumbent_candidate: dict | None = None,
    ) -> dict:
        return {
            "status": status,
            "node_id": node.node_id,
            "valid_lower_bound": valid_lower_bound,
            "phase_iterations": phase_iterations,
            "business_iterations": business_iterations,
            "records": records,
            "incumbent_candidate": incumbent_candidate,
            "seconds": perf_counter() - started,
        }

    def _aggregate_area_original_values(
        self,
        configuration_values: dict[int, float],
    ) -> tuple[
        dict[tuple[str, str, str], float],
        dict[tuple[str, str, str], float],
        dict[tuple[str, str, str], float],
    ]:
        row_use: defaultdict[tuple[str, str, str], float] = defaultdict(float)
        row_quantity: defaultdict[tuple[str, str, str], float] = defaultdict(
            float
        )
        import_quantity: defaultdict[
            tuple[str, str, str], float
        ] = defaultdict(float)
        for index, value in configuration_values.items():
            if value <= 1e-9:
                continue
            configuration = self._area_configurations[index]
            used_rows: set[tuple[str, str, str]] = set()
            for placement in configuration.placements:
                for bay_key, row_no, quantity in placement.row_allocation:
                    if bay_key != placement.bay_key:
                        continue
                    key = (placement.group_id, placement.bay_key, row_no)
                    row_quantity[key] += float(value) * int(quantity)
                    if int(quantity) > 0:
                        used_rows.add(key)
            for key in used_rows:
                row_use[key] += float(value)
            for flow, size, bay_key, quantity in (
                configuration.import_reservations
            ):
                import_quantity[(flow, size, bay_key)] += (
                    float(value) * int(quantity)
                )
        return dict(row_use), dict(row_quantity), dict(import_quantity)

    def _select_area_branch_pair(
        self,
        configuration_values: dict[int, float],
    ) -> tuple[BranchDecision, BranchDecision] | None:
        group_area_quantity: defaultdict[tuple[str, str], float] = (
            defaultdict(float)
        )
        for index, value in configuration_values.items():
            if value <= 1e-9:
                continue
            configuration = self._area_configurations[index]
            for group_id, quantity in configuration.group_quantities:
                group_area_quantity[(group_id, configuration.area_no)] += (
                    float(value) * int(quantity)
                )
        row_use, row_quantity, import_quantity = (
            self._aggregate_area_original_values(configuration_values)
        )
        for section, values in (
            ("branch_group_area_quantity", group_area_quantity),
            ("branch_row_quantity", row_quantity),
            ("branch_row_use", row_use),
            ("branch_import_quantity", import_quantity),
        ):
            selected = self._most_fractional_value(values)
            if selected is None:
                continue
            key, value = selected
            return (
                BranchDecision(section, key, "L", math.floor(value)),
                BranchDecision(section, key, "G", math.ceil(value)),
            )
        return None

    def _solve_area_row_recombination(
        self,
        areas: tuple[str, ...],
        time_limit: float,
        lp_configuration_values: dict[int, float] | None = None,
    ) -> dict | None:
        """Build an incumbent from LP-active and recent generated rows."""
        from gurobipy import quicksum

        if time_limit <= 1e-6:
            return None
        support: dict[tuple, PlacementColumn] = {}

        def add_unit_support(placement: PlacementColumn) -> None:
            key = self._base_placement_static_key(placement)
            support.setdefault(
                key,
                replace(
                    placement,
                    quantity=1,
                    stack_units=1,
                    row_allocation=tuple(
                        (bay_key, row_no, 1)
                        for bay_key, row_no, _quantity in (
                            placement.row_allocation
                        )
                    ),
                ),
            )

        active_configuration_indices = (
            set(range(len(self._area_configurations)))
            if lp_configuration_values is None
            else {
                int(index)
                for index, value in lp_configuration_values.items()
                if float(value) > 1e-7
            }
        )
        outer_history = 2
        configurations_by_area: defaultdict[str, list[int]] = defaultdict(list)
        for index, configuration in enumerate(self._area_configurations):
            configurations_by_area[configuration.area_no].append(index)
        recent_configuration_indices = {
            index
            for area_no in areas
            for index in configurations_by_area.get(area_no, [])[
                -outer_history:
            ]
        }
        screened_configuration_indices = (
            active_configuration_indices | recent_configuration_indices
        )
        for index in sorted(screened_configuration_indices):
            for placement in self._area_configurations[index].placements:
                add_unit_support(placement)
        outer_configuration_support_count = len(support)
        nested_block_history = 2
        for state in self._nested_area_pricing_states.values():
            for configurations in state.block_configurations.values():
                for configuration in configurations[-nested_block_history:]:
                    for placement in configuration.placements:
                        add_unit_support(placement)
        supported_groups = {
            placement.group_id for placement in support.values()
        }
        if any(
            group.group_id not in supported_groups for group in self.groups
        ):
            return None

        screened_locations = sorted(
            support.values(), key=self._base_placement_static_key
        )
        started = perf_counter()
        model = None
        try:
            model, variables, model_stats = (
                DirectMilpPlanner.build_compact_row_milp(
                    self,
                    screened_locations,
                    GurobiModel,
                    quicksum,
                )
            )
            remaining = float(time_limit) - (perf_counter() - started)
            if remaining <= 1e-6:
                return None
            self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
            self._set_gurobi_param(
                model, "MIPGap", max(0.0, float(self.config.mip_gap))
            )
            self._set_gurobi_param(model, "MIPFocus", 1)
            model.optimize()
            if self._gurobi_solution_count(model) <= 0:
                return None
            selected_locations = DirectMilpPlanner.selected_compact_row_values(
                self, model, variables
            )
            import_values = self._gurobi_import_reservation_values(
                model, variables
            )
            placements_by_area: defaultdict[
                str, list[PlacementColumn]
            ] = defaultdict(list)
            for index, quantity in selected_locations.items():
                if int(quantity) <= 0:
                    continue
                unit = screened_locations[int(index)]
                group = self.groups_by_id[unit.group_id]
                placements_by_area[unit.area_no].append(
                    replace(
                        unit,
                        quantity=int(quantity),
                        stack_units=self._stack_units_for_quantity(
                            unit.bay_key,
                            unit.size,
                            group,
                            int(quantity),
                        ),
                        row_allocation=tuple(
                            (bay_key, row_no, int(quantity))
                            for bay_key, row_no, _old_quantity in (
                                unit.row_allocation
                            )
                        ),
                    )
                )
            imports_by_area: defaultdict[
                str, list[tuple[str, str, str, int]]
            ] = defaultdict(list)
            for (flow, size, bay_key), quantity in import_values.items():
                if int(quantity) > 0:
                    imports_by_area[self.bays[bay_key].area_no].append(
                        (flow, size, bay_key, int(quantity))
                    )
            selected_configurations: dict[int, int] = {}
            for area_no in areas:
                configuration = self._area_configuration_from_allocations(
                    area_no,
                    tuple(placements_by_area.get(area_no, [])),
                    tuple(imports_by_area.get(area_no, [])),
                )
                selected_configurations[
                    self._ensure_area_configuration(configuration)
                ] = 1
            return {
                "status": self._gurobi_status_name(model),
                "objective": self._gurobi_objective_value(model),
                "bound": self._gurobi_dual_bound(model),
                "gap": self._gurobi_gap(model),
                "selected_configurations": selected_configurations,
                "screened_row_location_count": len(screened_locations),
                "outer_configuration_row_location_count": (
                    outer_configuration_support_count
                ),
                "active_outer_configuration_count": len(
                    active_configuration_indices
                ),
                "recent_outer_configuration_count": len(
                    recent_configuration_indices
                ),
                "outer_configuration_history": outer_history,
                "nested_block_row_location_count": (
                    len(screened_locations)
                    - outer_configuration_support_count
                ),
                "nested_block_history": nested_block_history,
                "model_variable_count": model_stats["model_variable_count"],
                "seconds": perf_counter() - started,
            }
        finally:
            if model is not None:
                self._free_gurobi_model(model)

    @staticmethod
    def _integral_configuration_selection(
        values: dict[int, float], tolerance: float = 1e-6
    ) -> dict[int, int] | None:
        if any(abs(value - round(value)) > tolerance for value in values.values()):
            return None
        return {
            int(index): int(round(value))
            for index, value in values.items()
            if int(round(value)) > 0
        }

    def _validate_selected_area_configurations(
        self,
        areas: tuple[str, ...],
        selected: dict[int, int],
    ) -> None:
        selected_by_area: Counter[str] = Counter()
        group_quantities: Counter[str] = Counter()
        import_quantities: Counter[tuple[str, str]] = Counter()
        for index, chosen in selected.items():
            if int(chosen) != 1:
                raise RuntimeError(
                    "area configuration selection must be binary"
                )
            configuration = self._area_configurations[int(index)]
            selected_by_area[configuration.area_no] += 1
            group_quantities.update(dict(configuration.group_quantities))
            import_quantities.update(
                dict(configuration.import_total_quantities)
            )
        for area_no in areas:
            if selected_by_area[area_no] != 1:
                raise RuntimeError(
                    "exactly one configuration must be selected per area: "
                    f"area={area_no}, selected={selected_by_area[area_no]}"
                )
        for group_id, required in self.group_demand.items():
            if group_quantities[group_id] != int(required):
                raise RuntimeError(
                    "selected area configurations violate export demand: "
                    f"group={group_id}, selected={group_quantities[group_id]}, "
                    f"required={required}"
                )
        for key, required in self.import_total_by_flow_size.items():
            if import_quantities[key] != int(required):
                raise RuntimeError(
                    "selected area configurations violate import reservation: "
                    f"flow_size={key}, selected={import_quantities[key]}, "
                    f"required={required}"
                )

    def solve_root(self) -> dict:
        """Solve the root relaxation with the same node engine as the tree."""
        started = perf_counter()
        total_limit = float(self.config.total_time_limit)
        deadline = (
            None if total_limit <= 0.0 else started + max(0.01, total_limit)
        )
        areas = self._initialize_area_configuration_pool()
        root = BranchPriceNode(node_id=0, depth=0)
        try:
            result = self._solve_area_node_lp(
                root,
                areas,
                deadline,
                allow_incumbent_heuristic=False,
            )
        finally:
            self._dispose_area_pricing_models()

        records = list(result.get("records", []))
        root_exact = result["status"] == "optimal"
        return {
            "algorithm": "adaptive_area_configuration_root_column_generation",
            "status": result["status"],
            "root_exact": root_exact,
            "root_objective": (
                float(result["bound"]) if root_exact else math.nan
            ),
            "valid_lower_bound": float(
                result.get("valid_lower_bound", 0.0)
            ),
            "area_count": len(areas),
            "group_count": len(self.groups),
            "configuration_count": len(self._area_configurations),
            "phase_iterations": int(result.get("phase_iterations", 0)),
            "business_iterations": int(
                result.get("business_iterations", 0)
            ),
            "pricing_model_build_seconds": sum(
                pricing.build_seconds
                for pricing in self._area_pricing_models.values()
            ),
            "pricing_seconds": sum(
                float(record.get("pricing_seconds", 0.0))
                for record in records
            ),
            "records": records,
            "total_seconds": perf_counter() - started,
            "adaptive_pricing": self._adaptive_pricing_diagnostics(areas),
            "formulation": self._area_formulation_diagnostics(),
        }

    def solve_branch_and_price(self) -> dict:
        """Solve the area-configuration formulation by exact node pricing."""
        started = perf_counter()
        total_limit = float(self.config.total_time_limit)
        deadline = (
            None if total_limit <= 0.0 else started + max(0.01, total_limit)
        )
        areas = self._initialize_area_configuration_pool()
        root = BranchPriceNode(node_id=0, depth=0)
        queue: list[tuple[float, int, BranchPriceNode]] = [
            (root.inherited_bound, root.node_id, root)
        ]
        next_node_id = 1
        incumbent_objective = math.inf
        incumbent_selection: dict[int, int] | None = None
        root_bound = math.nan
        processed_nodes = 0
        infeasible_nodes = 0
        pruned_nodes = 0
        branch_counts: Counter[str] = Counter()
        node_records: list[dict] = []
        pricing_records: list[dict] = []
        termination = "optimal"
        try:
            while queue:
                if processed_nodes >= int(self.config.max_branch_nodes):
                    termination = "node_limit"
                    break
                if math.isfinite(incumbent_objective):
                    current_bound = min(
                        float(queue[0][0]), incumbent_objective
                    )
                    current_gap = max(
                        0.0, incumbent_objective - current_bound
                    ) / max(abs(incumbent_objective), 1e-12)
                    if current_gap <= max(0.0, float(self.config.mip_gap)):
                        termination = "gap_limit"
                        break
                remaining = self._seconds_until(deadline)
                if remaining is not None and remaining <= 1e-6:
                    termination = "time_limit"
                    break
                _priority, _queued_id, node = heapq.heappop(queue)
                if node.inherited_bound >= incumbent_objective - 1e-9:
                    pruned_nodes += 1
                    continue
                node_deadline = deadline
                allocated_node_seconds = remaining
                time_sliced_node = False
                if node.depth > 0 and remaining is not None:
                    active_node_count = len(queue) + 1
                    node_slice = max(
                        2.0,
                        min(
                            10.0,
                            remaining / max(1, active_node_count),
                        ),
                    )
                    node_deadline = min(
                        deadline,
                        perf_counter() + node_slice,
                    )
                    allocated_node_seconds = node_slice
                    time_sliced_node = node_deadline < deadline - 1e-9
                node_result = self._solve_area_node_lp(
                    node, areas, node_deadline
                )
                processed_nodes += 1
                pricing_records.extend(node_result.get("records", []))
                status = str(node_result["status"])
                valid_bound = float(
                    node_result.get(
                        "valid_lower_bound", node.inherited_bound
                    )
                )
                if node.node_id == 0 and math.isfinite(valid_bound):
                    root_bound = valid_bound
                node_incumbent = node_result.get("incumbent_candidate")
                if (
                    node_incumbent is not None
                    and float(node_incumbent["objective"])
                    < incumbent_objective - 1e-9
                ):
                    incumbent_objective = float(
                        node_incumbent["objective"]
                    )
                    incumbent_selection = dict(
                        node_incumbent["selected_configurations"]
                    )
                node_records.append(
                    {
                        "node_id": node.node_id,
                        "depth": node.depth,
                        "status": status,
                        "bound": node_result.get("bound"),
                        "valid_lower_bound": valid_bound,
                        "phase_iterations": node_result.get(
                            "phase_iterations", 0
                        ),
                        "business_iterations": node_result.get(
                            "business_iterations", 0
                        ),
                        "seconds": node_result.get("seconds", 0.0),
                        "time_slice_seconds": (
                            allocated_node_seconds
                        ),
                        "incumbent_candidate": (
                            None
                            if node_incumbent is None
                            else {
                                key: node_incumbent[key]
                                for key in (
                                    "status",
                                    "objective",
                                    "bound",
                                    "gap",
                                    "screened_row_location_count",
                                    "outer_configuration_row_location_count",
                                    "nested_block_row_location_count",
                                    "nested_block_history",
                                    "active_outer_configuration_count",
                                    "recent_outer_configuration_count",
                                    "outer_configuration_history",
                                    "model_variable_count",
                                    "seconds",
                                )
                            }
                        ),
                    }
                )
                if status == "infeasible":
                    infeasible_nodes += 1
                    continue
                if status != "optimal":
                    heapq.heappush(
                        queue,
                        (
                            valid_bound,
                            node.node_id,
                            replace(node, inherited_bound=valid_bound),
                        ),
                    )
                    if status == "time_limit" and time_sliced_node:
                        continue
                    termination = status
                    break
                node_bound = float(node_result["bound"])
                if node.node_id == 0:
                    root_bound = node_bound
                if node_bound >= incumbent_objective - 1e-9:
                    pruned_nodes += 1
                    continue

                configuration_values = node_result["configuration_values"]
                integral = self._integral_configuration_selection(
                    configuration_values
                )
                if integral is not None:
                    incumbent_objective = node_bound
                    incumbent_selection = integral
                    continue

                branch_pair = self._select_area_branch_pair(
                    configuration_values
                )
                if branch_pair is None:
                    if incumbent_objective <= node_bound + 1e-7:
                        pruned_nodes += 1
                        continue
                    termination = "branching_stalled"
                    heapq.heappush(
                        queue,
                        (
                            node_bound,
                            node.node_id,
                            replace(node, inherited_bound=node_bound),
                        ),
                    )
                    break
                for decision in branch_pair:
                    child = BranchPriceNode(
                        node_id=next_node_id,
                        depth=node.depth + 1,
                        decisions=(*node.decisions, decision),
                        inherited_bound=node_bound,
                    )
                    next_node_id += 1
                    heapq.heappush(
                        queue,
                        (child.inherited_bound, child.node_id, child),
                    )
                branch_counts[branch_pair[0].section] += 1
        finally:
            self._dispose_area_pricing_models()

        open_bound = min(
            (item[0] for item in queue),
            default=(
                incumbent_objective
                if math.isfinite(incumbent_objective)
                else root_bound
            ),
        )
        global_bound = min(open_bound, incumbent_objective)
        absolute_gap = (
            math.inf
            if not math.isfinite(incumbent_objective)
            or not math.isfinite(global_bound)
            else max(0.0, incumbent_objective - global_bound)
        )
        relative_gap = (
            math.inf
            if not math.isfinite(absolute_gap)
            else absolute_gap / max(abs(incumbent_objective), 1e-12)
        )
        if not queue and incumbent_selection is not None:
            termination = "optimal"
            global_bound = incumbent_objective
            absolute_gap = 0.0
            relative_gap = 0.0
        if incumbent_selection is not None:
            self._validate_selected_area_configurations(
                areas, incumbent_selection
            )
        return {
            "algorithm": "adaptive_area_configuration_branch_and_price",
            "node_lp_engine": (
                "shared_root_and_child_exact_area_column_generation"
            ),
            "incumbent_heuristic": {
                "method": "screened_row_recombination",
                "scope": "root_once",
                "bound_role": "primal_upper_bound_only",
                "trigger": "root_closure_or_35_percent_elapsed",
            },
            "status": termination,
            "objective": incumbent_objective,
            "global_lower_bound": global_bound,
            "absolute_gap": absolute_gap,
            "relative_gap": relative_gap,
            "root_lower_bound": root_bound,
            "selected_configurations": incumbent_selection,
            "processed_nodes": processed_nodes,
            "open_nodes": len(queue),
            "infeasible_nodes": infeasible_nodes,
            "pruned_nodes": pruned_nodes,
            "branch_counts": dict(branch_counts),
            "configuration_count": len(self._area_configurations),
            "node_records": node_records,
            "pricing_records": pricing_records,
            "total_seconds": perf_counter() - started,
        }


__all__ = ["AreaConfigurationBranchPricePlanner"]
