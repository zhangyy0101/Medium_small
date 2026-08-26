"""Complete compact M0 MILP used as an exact algorithmic baseline."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from time import perf_counter

from .gurobi_backend import GurobiModel
from .planner import (
    ColumnGenerationResult,
    PlacementColumn,
    YardPlanningBase,
)


class DirectMilpPlanner(YardPlanningBase):
    """M0 baseline: solve the complete row-location formulation directly."""

    def solve(self) -> ColumnGenerationResult:
        self._prepare_master_index_sets()
        self._prepare_objective_normalization()
        self._initialize_location_pool()
        for group in self.groups:
            for candidate in self._base_placements_for_group(group):
                self._append_generated_column(candidate)

        selected, solve_stats = self._solve_direct_milp()
        diagnostics = {
            "algorithm": "direct_compact_row_location_milp_gurobi",
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
            "formulation": "complete_integer_row_location_milp",
            "decomposition": "none",
            "gurobi_available": True,
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
                "role": "anonymous_bay_capacity_reservation",
                "area_policy": (
                    "weighted_l1_deviation_from_big_plan_reference"
                ),
                "constraint_scope": [
                    "area_function",
                    "bay_size_state",
                    "physical_capacity",
                    "export_import_bay_exclusivity",
                ],
                "excluded_constraints": [
                    "export_height_state",
                    "export_row_group_assignment",
                    "export_group_objectives",
                ],
                "import_boxes": int(
                    sum(self.import_area_size_reference.values())
                ),
            },
            **solve_stats,
        }
        result = self._assemble_result(selected, diagnostics)
        result.columns = self._selected_direct_columns(selected)
        return result

    def _selected_direct_columns(
        self, selected: Counter[int]
    ) -> list[PlacementColumn]:
        output: list[PlacementColumn] = []
        for index, multiplier in sorted(selected.items()):
            if int(multiplier) <= 0:
                continue
            candidate = self._columns[index]
            group = self.groups_by_id[candidate.group_id]
            quantity = int(candidate.quantity) * int(multiplier)
            output.append(
                replace(
                    candidate,
                    column_id=f"M0_{len(output) + 1:07d}",
                    quantity=quantity,
                    stack_units=self._stack_units_for_quantity(
                        candidate.bay_key,
                        candidate.size,
                        group,
                        quantity,
                    ),
                    row_allocation=tuple(
                        (bay_key, row_no, int(row_quantity) * int(multiplier))
                        for bay_key, row_no, row_quantity in
                        candidate.row_allocation
                    ),
                )
            )
        return output

    def _solve_direct_milp(
        self,
    ) -> tuple[Counter[int], dict]:
        from gurobipy import quicksum

        started = perf_counter()
        total_limit = float(self.config.total_time_limit)
        if total_limit <= 0.0:
            total_limit = max(2.0, 2.0 * float(self.config.mip_time_limit))
        model, variables, model_stats = self.build_compact_row_milp(
            self._columns,
            GurobiModel,
            quicksum,
        )
        try:
            model_build_seconds = perf_counter() - started
            remaining_limit = total_limit - model_build_seconds
            if remaining_limit <= 1e-6:
                raise RuntimeError(
                    "M0 model construction consumed the complete time limit"
                )
            self._set_gurobi_param(
                model, "TimeLimit", max(0.01, remaining_limit)
            )
            self._set_gurobi_param(
                model, "MIPGap", max(0.0, float(self.config.mip_gap))
            )
            model.optimize()
            status = self._gurobi_status_name(model)
            if self._gurobi_solution_count(model) <= 0:
                raise RuntimeError(
                    "M0 cannot assign all declared export containers; "
                    f"status={status}"
                )
            selected = self.selected_compact_row_values(model, variables)
            self._final_import_reservation = (
                self._gurobi_import_reservation_values(model, variables)
            )
            solver_objective = self._gurobi_objective_value(model)
            solver_bound = self._gurobi_dual_bound(model)
            solver_gap = self._gurobi_gap(model)
        finally:
            self._free_gurobi_model(model)

        reconstructed = self._selected_solution_energy(selected)
        objective_auxiliary_slack = (
            self._absolute_deviation_auxiliary_slack(
                solver_objective,
                reconstructed,
                context="M0",
            )
        )
        absolute_gap = max(0.0, reconstructed - solver_bound)
        relative_gap = absolute_gap / max(abs(reconstructed), 1e-12)
        return selected, {
            "master_algorithm": "direct_compact_row_location_milp",
            "master_status": status,
            "master_bound_scope": "complete_direct_milp",
            "master_objective": reconstructed,
            "master_mip_gap": solver_gap,
            "complete_model_lower_bound": solver_bound,
            "complete_model_absolute_gap": absolute_gap,
            "complete_model_relative_gap": relative_gap,
            "complete_model_gap_source": "gurobi_direct_milp_bound",
            "direct_status": status,
            "direct_objective": reconstructed,
            "direct_solver_incumbent_objective": solver_objective,
            "direct_objective_auxiliary_slack": (
                objective_auxiliary_slack
            ),
            "direct_bound": solver_bound,
            "direct_mip_gap": solver_gap,
            "direct_total_solve_seconds": round(
                perf_counter() - started, 3
            ),
            "direct_model_build_seconds": round(model_build_seconds, 3),
            "direct_model": model_stats,
            "persistent_direct_model": True,
            "hard_demand_balance": True,
            "candidate_row_location_count": len(self._columns),
            "selected_location_count": len(selected),
        }

    def selected_compact_row_values(
        self, model, variables: dict
    ) -> Counter[int]:
        return Counter(
            {
                index: int(round(self._gurobi_value(model, variable)))
                for index, variable in variables["column"].items()
                if self._gurobi_value(model, variable) > 0.5
            }
        )

    def build_compact_row_milp(
        self,
        row_locations: list[PlacementColumn] | tuple[PlacementColumn, ...],
        Model,
        quicksum,
    ):
        """Build the common compact row model on explicit candidate rows."""
        model = Model("yard_compact_row_milp")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        model.setMinimize()
        capacities = {
            index: self._base_location_capacity(
                self.groups_by_id[column.group_id], column
            )
            for index, column in enumerate(row_locations)
        }
        columns = {
            index: model.addVar(
                lb=0.0,
                ub=float(capacities[index]),
                vtype="I",
                obj=float(column.intrinsic_cost),
                name=f"x_{index}",
            )
            for index, column in enumerate(row_locations)
        }
        import_reserve = {
            (flow, size, bay_key): model.addVar(
                lb=0.0,
                ub=float(capacity),
                vtype="I",
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
        group_resource_columns: defaultdict[
            tuple[str, str, str], list[tuple[int, float]]
        ] = defaultdict(list)
        group_columns: defaultdict[str, list[int]] = defaultdict(list)
        group_area_columns: defaultdict[tuple[tuple[str, ...], str], list[int]] = (
            defaultdict(list)
        )
        group_row_columns: defaultdict[
            tuple[tuple[str, ...], str, str], list[int]
        ] = defaultdict(list)
        operational_group_columns: defaultdict[tuple[str, ...], list[int]] = (
            defaultdict(list)
        )
        operational_group_demand: Counter[tuple[str, ...]] = Counter()
        for group in self.groups:
            operational_group_demand[
                self._operational_group_key(group)
            ] += int(group.demand)
        for index, column in enumerate(row_locations):
            group_columns[column.group_id].append(index)
            operational_group_columns[column.group_key].append(index)
            group_area_columns[(column.group_key, column.area_no)].append(index)
            anchor_row = next(
                row_no
                for bay_key, row_no, _quantity in column.row_allocation
                if bay_key == column.bay_key
            )
            group_row_columns[
                (column.group_key, column.bay_key, anchor_row)
            ].append(index)
            for footprint_key, row_no, row_quantity in column.row_allocation:
                if int(row_quantity) > 0:
                    group_resource_columns[
                        (column.group_id, str(footprint_key), str(row_no))
                    ].append((index, float(row_quantity)))
            for section, values in self._placement_master_coefficients(
                column
            ).items():
                for key, coefficient in values.items():
                    if coefficient:
                        coefficient_rows[section][key].append(
                            (index, float(coefficient))
                        )

        import_by_bay: defaultdict[str, list] = defaultdict(list)
        import_by_anchor_bay_size: defaultdict[tuple[str, str], list] = (
            defaultdict(list)
        )
        import_by_physical_bay_size: defaultdict[tuple[str, str], list] = (
            defaultdict(list)
        )
        import_by_flow_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        import_by_flow_area_size: defaultdict[tuple[str, str, str], list] = (
            defaultdict(list)
        )
        for (flow, size, bay_key), variable in import_reserve.items():
            area_no = self.bays[bay_key].area_no
            for footprint_key in self._placement_footprint_keys(bay_key, size):
                import_by_bay[footprint_key].append(variable)
                import_by_physical_bay_size[(footprint_key, size)].append(
                    variable
                )
            import_by_anchor_bay_size[(bay_key, size)].append(variable)
            import_by_flow_size[(flow, size)].append(variable)
            import_by_flow_area_size[(flow, area_no, size)].append(variable)

        export_by_bay = coefficient_rows["bay_capacity_limit"]
        allocation_bay_keys = sorted(set(export_by_bay) | set(import_by_bay))
        export_bay_use = {
            bay_key: model.addVar(
                vtype="B",
                name=f"export_bay_use_{self._key_name((bay_key,))}",
            )
            for bay_key in allocation_bay_keys
        }
        import_bay_use = {
            bay_key: model.addVar(
                vtype="B",
                name=f"import_bay_use_{self._key_name((bay_key,))}",
            )
            for bay_key in allocation_bay_keys
        }
        import_size_state = {
            key: model.addVar(
                vtype="B",
                name=f"import_size_state_{self._key_name(key)}",
            )
            for key in sorted(import_by_physical_bay_size)
        }
        group_row_owner = {
            key: model.addVar(
                vtype="B",
                name=f"group_row_owner_{self._key_name(key)}",
            )
            for key in sorted(group_resource_columns)
        }

        constraints: dict[str, dict] = defaultdict(dict)
        for group in self.groups:
            assigned = quicksum(
                columns[index]
                for index in group_columns.get(group.group_id, [])
            )
            constraints["group_demand_balance"][group.group_id] = model.addConstr(
                assigned == int(group.demand),
                name=f"group_demand_{group.group_id}",
            )

        for bay_key in sorted(self._master_bay_capacity_keys):
            items = coefficient_rows["bay_capacity_limit"].get(bay_key, [])
            constraints["bay_capacity_limit"][bay_key] = model.addConstr(
                quicksum(
                    coefficient * columns[index]
                    for index, coefficient in items
                )
                + quicksum(import_by_bay.get(bay_key, []))
                <= int(self.bays[bay_key].physical_capacity),
                name=f"bay_cap_{self._key_name((bay_key,))}",
            )
        for key in sorted(self._master_bay_size_keys):
            bay_key, size = key
            items = coefficient_rows["bay_size_limit"].get(key, [])
            constraints["bay_size_limit"][key] = model.addConstr(
                quicksum(
                    coefficient * columns[index]
                    for index, coefficient in items
                )
                + quicksum(import_by_anchor_bay_size.get(key, []))
                <= int(self.bays[bay_key].cap_by_size.get(size, 0)),
                name=f"bay_size_{self._key_name(key)}",
            )
        for key in sorted(self._master_row_capacity_keys):
            bay_key, row_no = key
            items = coefficient_rows["row_capacity_limit"].get(key, [])
            constraints["row_capacity_limit"][key] = model.addConstr(
                quicksum(
                    coefficient * columns[index]
                    for index, coefficient in items
                )
                <= int(
                    self.bays[bay_key].row_physical_capacity.get(
                        row_no, self.bays[bay_key].physical_capacity
                    )
                ),
                name=f"row_cap_{self._key_name(key)}",
            )
        for key in sorted(self._master_row_size_keys):
            bay_key, row_no, size = key
            items = coefficient_rows["row_size_limit"].get(key, [])
            constraints["row_size_limit"][key] = model.addConstr(
                quicksum(
                    coefficient * columns[index]
                    for index, coefficient in items
                )
                <= int(
                    self.bays[bay_key].row_cap_by_size.get(size, {}).get(
                        row_no,
                        self.bays[bay_key].cap_by_size.get(size, 0),
                    )
                ),
                name=f"row_size_{self._key_name(key)}",
            )

        owners_by_resource: defaultdict[tuple[str, str], list] = defaultdict(list)
        for key, owner in group_row_owner.items():
            _group_id, bay_key, row_no = key
            resource = (bay_key, row_no)
            owners_by_resource[resource].append(owner)
            items = group_resource_columns[key]
            load = quicksum(
                coefficient * columns[index]
                for index, coefficient in items
            )
            capacity = int(
                self.bays[bay_key].row_physical_capacity.get(
                    row_no,
                    self.bays[bay_key].physical_capacity,
                )
            )
            constraints["group_row_owner_link"][key] = model.addConstr(
                load <= max(1, capacity) * owner,
                name=f"group_row_owner_link_{self._key_name(key)}",
            )
            constraints["group_row_owner_presence"][key] = model.addConstr(
                owner <= load,
                name=f"group_row_owner_presence_{self._key_name(key)}",
            )
        for resource, owners in sorted(owners_by_resource.items()):
            constraints["physical_row_single_group"][resource] = model.addConstr(
                quicksum(owners) <= 1,
                name=f"physical_row_single_group_{self._key_name(resource)}",
            )

        import_sizes_by_bay: defaultdict[str, list] = defaultdict(list)
        for (bay_key, size), state_variable in import_size_state.items():
            import_sizes_by_bay[bay_key].append(state_variable)
            size_load = quicksum(
                import_by_physical_bay_size[(bay_key, size)]
            )
            capacity = max(1, int(self.bays[bay_key].physical_capacity))
            constraints["import_size_state_link"][(bay_key, size)] = (
                model.addConstr(
                    size_load <= capacity * state_variable,
                    name=(
                        "import_size_state_link_"
                        f"{self._key_name((bay_key, size))}"
                    ),
                )
            )
            constraints["import_size_state_presence"][(bay_key, size)] = (
                model.addConstr(
                    state_variable <= size_load,
                    name=(
                        "import_size_state_presence_"
                        f"{self._key_name((bay_key, size))}"
                    ),
                )
            )
        for bay_key in allocation_bay_keys:
            export_load = quicksum(
                coefficient * columns[index]
                for index, coefficient in export_by_bay.get(bay_key, [])
            )
            import_load = quicksum(import_by_bay.get(bay_key, []))
            capacity = max(1, int(self.bays[bay_key].physical_capacity))
            constraints["export_bay_use_link"][bay_key] = model.addConstr(
                export_load <= capacity * export_bay_use[bay_key],
                name=f"export_bay_use_link_{self._key_name((bay_key,))}",
            )
            constraints["export_bay_use_presence"][bay_key] = model.addConstr(
                export_bay_use[bay_key] <= export_load,
                name=f"export_bay_use_presence_{self._key_name((bay_key,))}",
            )
            constraints["import_bay_use_link"][bay_key] = model.addConstr(
                import_load <= capacity * import_bay_use[bay_key],
                name=f"import_bay_use_link_{self._key_name((bay_key,))}",
            )
            constraints["import_bay_use_presence"][bay_key] = model.addConstr(
                import_bay_use[bay_key] <= import_load,
                name=f"import_bay_use_presence_{self._key_name((bay_key,))}",
            )
            constraints["export_import_bay_exclusive"][bay_key] = model.addConstr(
                export_bay_use[bay_key] + import_bay_use[bay_key] <= 1,
                name=(
                    "export_import_bay_exclusive_"
                    f"{self._key_name((bay_key,))}"
                ),
            )
            constraints["import_bay_single_size"][bay_key] = model.addConstr(
                quicksum(import_sizes_by_bay.get(bay_key, []))
                <= import_bay_use[bay_key],
                name=f"import_bay_single_size_{self._key_name((bay_key,))}",
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
            stack_variable = model.addVar(
                lb=0.0,
                ub=float(stack_count),
                vtype="I",
                name=f"stack_{self._key_name(key)}",
            )
            items = coefficient_rows["bay_port_stack_link"].get(key, [])
            constraints["bay_port_stack_link"][key] = model.addConstr(
                quicksum(
                    coefficient * columns[index]
                    for index, coefficient in items
                )
                <= unit_capacity * stack_variable,
                name=f"stack_load_{self._key_name(key)}",
            )
            stack_variables_by_bay_size[(bay_key, size)].append(stack_variable)
        for key, stack_variables in stack_variables_by_bay_size.items():
            constraints["bay_stack_total_limit"][key] = model.addConstr(
                quicksum(stack_variables)
                <= self._stack_count_for_bay_size(*key),
                name=f"stack_total_{self._key_name(key)}",
            )

        for key, required in sorted(self.import_total_by_flow_size.items()):
            candidates = import_by_flow_size.get(key, [])
            if not candidates:
                raise ValueError(
                    "import capacity reservation has no compatible bay: "
                    f"flow={key[0]}, size={key[1]}, required={required}"
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
            items = coefficient_rows["area_guidance_balance"].get(key, [])
            constraints["area_guidance_balance"][key] = model.addConstr(
                quicksum(
                    coefficient * columns[index]
                    for index, coefficient in items
                )
                - target
                == positive - negative,
                name=f"guide_balance_{self._key_name(key)}",
            )

        area_uses_by_group: defaultdict[tuple[str, ...], list] = (
            defaultdict(list)
        )
        row_uses_by_group: defaultdict[tuple[str, ...], list] = (
            defaultdict(list)
        )
        for key, indices in sorted(group_area_columns.items()):
            group_key, area_no = key
            upper = min(
                int(operational_group_demand[group_key]),
                sum(capacities[index] for index in indices),
            )
            use = model.addVar(
                vtype="B",
                obj=self._area_activation_penalty(),
                name=(
                    f"use_group_area_{self._key_name(group_key)}_"
                    f"{area_no}"
                ),
            )
            constraints["group_area_activation"][key] = model.addConstr(
                quicksum(columns[index] for index in indices)
                <= max(1, upper) * use,
                name=f"group_area_link_{self._key_name((*group_key, area_no))}",
            )
            area_uses_by_group[group_key].append(use)
        for key, indices in sorted(group_row_columns.items()):
            group_key, bay_key, row_no = key
            upper = min(
                int(operational_group_demand[group_key]),
                sum(capacities[index] for index in indices),
            )
            use = model.addVar(
                vtype="B",
                obj=self._row_activation_penalty(),
                name=(
                    f"use_group_row_{self._key_name(group_key)}_"
                    f"{self._key_name((bay_key, row_no))}"
                ),
            )
            constraints["group_row_activation"][key] = model.addConstr(
                quicksum(columns[index] for index in indices)
                <= max(1, upper) * use,
                name=(
                    f"group_row_link_"
                    f"{self._key_name((*group_key, bay_key, row_no))}"
                ),
            )
            row_uses_by_group[group_key].append(use)
        for group_key, indices in sorted(
            operational_group_columns.items()
        ):
            used = model.addVar(
                vtype="B",
                obj=-(
                    self._area_activation_penalty()
                    + self._row_activation_penalty()
                ),
                name=f"group_used_{self._key_name(group_key)}",
            )
            assigned = quicksum(columns[index] for index in indices)
            demand = max(1, int(operational_group_demand[group_key]))
            constraints["group_used_upper"][group_key] = model.addConstr(
                assigned <= demand * used,
                name=f"group_used_upper_{self._key_name(group_key)}",
            )
            constraints["group_used_lower"][group_key] = model.addConstr(
                used <= assigned,
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
        bay_compatibility = self._add_bay_compatibility_constraints(
            quicksum,
            model,
            columns,
            coefficient_rows["bay_attr_link"],
            relax=False,
        )
        constraints.update(bay_compatibility)

        model.update()
        model_stats = {
            "row_location_variable_count": len(columns),
            "import_reservation_variable_count": len(import_reserve),
            "group_row_owner_variable_count": len(group_row_owner),
            "export_bay_use_variable_count": len(export_bay_use),
            "import_bay_use_variable_count": len(import_bay_use),
            "import_size_state_variable_count": len(import_size_state),
            "model_variable_count": len(model.getVars()),
            "constraint_count_by_family": {
                key: len(values)
                for key, values in sorted(constraints.items())
                if isinstance(values, dict)
            },
        }
        return model, {
            "column": columns,
            "import_reserve": import_reserve,
            "group_row_owner": group_row_owner,
            "export_bay_use": export_bay_use,
            "import_bay_use": import_bay_use,
            "import_size_state": import_size_state,
        }, model_stats


__all__ = ["DirectMilpPlanner"]
