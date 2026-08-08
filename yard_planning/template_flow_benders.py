"""Classical Benders decomposition on row templates and continuous box flow.

This is an isolated research formulation.  The master chooses a hard
no-mix state, export capacity, stack use, and anonymous import reserve for
complete physical row footprints.  Given those capacities, the recourse LP
sends declared export demand to compatible templates.  Concentration is
therefore measured for operational handling classes induced by the hard
no-mix attributes, rather than for raw input group identifiers.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from time import perf_counter

from .direct_milp import DirectMilpPlanner
from .gurobi_backend import GurobiModel
from .planner import ColumnGenerationConfig, ColumnGenerationResult


@dataclass(frozen=True)
class TemplateFlowBendersConfig:
    """Numerical controls for the independent decomposition experiment."""

    max_iterations: int = 60
    cut_tolerance: float = 1e-7
    flow_integrality_tolerance: float = 1e-5
    incumbent_seed_time_fraction: float = 0.35
    root_time_fraction: float = 0.20
    integer_master_time_fraction: float = 0.85

    def validate(self) -> None:
        if int(self.max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        if float(self.cut_tolerance) <= 0.0:
            raise ValueError("cut_tolerance must be positive")
        if float(self.flow_integrality_tolerance) <= 0.0:
            raise ValueError("flow_integrality_tolerance must be positive")
        if not 0.0 < float(self.incumbent_seed_time_fraction) < 1.0:
            raise ValueError(
                "incumbent_seed_time_fraction must lie strictly between 0 and 1"
            )
        if not 0.0 < float(self.root_time_fraction) < 1.0:
            raise ValueError("root_time_fraction must lie strictly between 0 and 1")
        if not 0.0 < float(self.integer_master_time_fraction) < 1.0:
            raise ValueError(
                "integer_master_time_fraction must lie strictly between 0 and 1"
            )


@dataclass(frozen=True)
class RowTemplate:
    """One hard-state choice on a complete mixed-size row footprint."""

    template_id: int
    handling_class: tuple[str, ...]
    area_no: str
    footprint: tuple[tuple[str, str], ...]
    capacity: int
    arc_indices: tuple[int, ...]


class TemplateFlowBendersPlanner(DirectMilpPlanner):
    """Row-template master plus a persistent continuous-flow recourse LP."""

    def __init__(
        self,
        problem,
        config: ColumnGenerationConfig | None = None,
        benders_config: TemplateFlowBendersConfig | None = None,
    ) -> None:
        super().__init__(problem, config)
        self.benders_config = benders_config or TemplateFlowBendersConfig()
        self.benders_config.validate()
        self._templates: tuple[RowTemplate, ...] = ()
        self._class_groups: defaultdict[tuple[str, ...], list[str]] = defaultdict(list)
        self._template_area_targets: dict[
            tuple[str, str, str, str], int
        ] = {}

    def _handling_class_key(self, group) -> tuple[str, ...]:
        """Return exactly the attributes that create hard storage states."""
        attrs = []
        for attr in (
            *self._bay_no_mix_attrs_for_group(group),
            *self._row_no_mix_attrs_for_group(group),
        ):
            if attr not in attrs:
                attrs.append(attr)
        return (
            str(group.voyage_id),
            f"flow={group.status}",
            *(f"{attr}={self._group_attr_value(group, attr)}" for attr in attrs),
        )

    def _operational_group_key(self, group) -> tuple[str, ...]:
        """Use the redefined hard handling class for concentration costs."""
        if not hasattr(self, "attribute_rules"):
            # The base constructor validates the untouched input groups before
            # it has finished installing the normalized attribute rules.
            return super()._operational_group_key(group)
        return self._handling_class_key(group)

    def _prepare_template_objective_normalization(self) -> None:
        demand_by_class: Counter[tuple[str, ...]] = Counter()
        areas_by_class: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
        rows_by_class: defaultdict[
            tuple[str, ...], set[tuple[tuple[str, str], ...]]
        ] = defaultdict(set)
        for group in self.groups:
            key = self._handling_class_key(group)
            demand_by_class[key] += int(group.demand)
            for candidate in self._base_placements_for_group(group):
                areas_by_class[key].add(candidate.area_no)
                rows_by_class[key].add(
                    tuple(
                        sorted(
                            (str(bay_key), str(row_no))
                            for bay_key, row_no, _quantity in candidate.row_allocation
                        )
                    )
                )
        area_scale = sum(
            max(0, min(int(demand), len(areas_by_class[key])) - 1)
            for key, demand in demand_by_class.items()
        )
        row_scale = sum(
            max(0, min(int(demand), len(rows_by_class[key])) - 1)
            for key, demand in demand_by_class.items()
        )
        self._objective_scales = {
            "area_dispersion": float(max(1, area_scale)),
            "row_dispersion": float(max(1, row_scale)),
            "existing_group_proximity": float(
                max(1, self._anchored_group_demand())
            ),
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
            "berth_distance": float(
                max(1, sum(group.demand for group in self.groups))
            ),
        }

    def _prepare_integer_area_targets(self) -> None:
        """Apportion large-plan shares to integral declared-box targets."""
        self._template_area_targets.clear()
        for (voyage_id, status, big_size), demand in sorted(
            self.voyage_flow_size_demand.items()
        ):
            weighted = [
                (area_no, int(quantity))
                for (candidate_voyage, candidate_status, area_no, candidate_size), quantity
                in self.quota_by_key.items()
                if candidate_voyage == voyage_id
                and candidate_status == status
                and candidate_size == big_size
                and int(quantity) > 0
            ]
            total_weight = sum(quantity for _area_no, quantity in weighted)
            if int(demand) <= 0 or total_weight <= 0:
                continue
            floors = {}
            remainders = []
            for area_no, quantity in weighted:
                numerator = int(demand) * quantity
                floors[area_no] = numerator // total_weight
                remainders.append(
                    (-(numerator % total_weight), str(area_no))
                )
            remaining = int(demand) - sum(floors.values())
            for _negative_remainder, area_no in sorted(remainders)[:remaining]:
                floors[area_no] += 1
            for area_no, quantity in floors.items():
                self._template_area_targets[
                    (voyage_id, status, area_no, big_size)
                ] = int(quantity)

    def _area_size_target(
        self, voyage_id: str, flow: str, area_no: str, big_size: str
    ) -> float:
        key = (voyage_id, flow, area_no, big_size)
        if self._template_area_targets and key in self._template_area_targets:
            return float(self._template_area_targets[key])
        return super()._area_size_target(voyage_id, flow, area_no, big_size)

    def _prepare_templates(self) -> dict[str, float | int]:
        started = perf_counter()
        self._prepare_master_index_sets()
        self._prepare_template_objective_normalization()
        self._prepare_integer_area_targets()
        self._initialize_location_pool()
        self._class_groups.clear()
        for group in self.groups:
            class_key = self._handling_class_key(group)
            self._class_groups[class_key].append(group.group_id)
            for candidate in self._base_placements_for_group(group):
                self._append_generated_column(
                    replace(candidate, group_key=class_key)
                )

        arcs_by_key: defaultdict[tuple, list[int]] = defaultdict(list)
        capacity_by_arc: dict[int, int] = {}
        for index, column in enumerate(self._columns):
            group = self.groups_by_id[column.group_id]
            footprint = tuple(
                sorted(
                    (str(bay_key), str(row_no))
                    for bay_key, row_no, _quantity in column.row_allocation
                )
            )
            key = (
                self._handling_class_key(
                    self.groups_by_id[column.group_id]
                ),
                footprint,
            )
            arcs_by_key[key].append(index)
            capacity_by_arc[index] = int(
                self._base_location_capacity(group, column)
            )

        templates: list[RowTemplate] = []
        for (class_key, footprint), indices in sorted(
            arcs_by_key.items(), key=lambda item: str(item[0])
        ):
            positive = [capacity_by_arc[index] for index in indices if capacity_by_arc[index] > 0]
            if not positive:
                continue
            areas = {self._columns[index].area_no for index in indices}
            if len(areas) != 1:
                raise RuntimeError(
                    "a physical row template spans multiple yard areas: "
                    f"class={class_key}, footprint={footprint}, areas={areas}"
                )
            capacity = min(positive)
            template_id = len(templates)
            template = RowTemplate(
                template_id=template_id,
                handling_class=tuple(class_key),
                area_no=next(iter(areas)),
                footprint=tuple(footprint),
                capacity=int(capacity),
                arc_indices=tuple(sorted(indices)),
            )
            templates.append(template)
        self._templates = tuple(templates)
        if not self._templates and self.groups:
            raise RuntimeError("template-flow formulation has no feasible row templates")
        return {
            "preparation_seconds": perf_counter() - started,
            "template_count": len(self._templates),
            "flow_arc_count": len(self._columns),
            "handling_class_count": len(self._class_groups),
        }

    def _build_master(self):
        from gurobipy import quicksum

        model = GurobiModel("yard_template_flow_benders_master")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(model, "Threads", int(self.config.solver_threads))
        model.setMinimize()

        template_use = {
            template.template_id: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="C",
                obj=self._row_activation_penalty(),
                name=f"template_{template.template_id}",
            )
            for template in self._templates
        }
        template_capacity = {
            template.template_id: model.addVar(
                lb=0.0,
                ub=float(template.capacity),
                vtype="C",
                name=f"template_capacity_{template.template_id}",
            )
            for template in self._templates
        }
        area_pairs = sorted(
            {
                (template.handling_class, template.area_no)
                for template in self._templates
            },
            key=str,
        )
        area_use = {
            key: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="C",
                obj=self._area_activation_penalty(),
                name=(
                    f"class_area_{self._key_name(key[0])}_"
                    f"{self._key_name((key[1],))}"
                ),
            )
            for key in area_pairs
        }
        theta = model.addVar(lb=0.0, obj=1.0, name="recourse_theta")
        class_count = len(self._class_groups)
        objective_offset = -class_count * (
            self._area_activation_penalty() + self._row_activation_penalty()
        )
        fixed_one = model.addVar(
            lb=1.0,
            ub=1.0,
            obj=objective_offset,
            name="concentration_baseline",
        )
        import_reserve = {
            (status, size, bay_key): model.addVar(
                lb=0.0,
                ub=float(capacity),
                vtype="C",
                name=(
                    f"template_import_{status}_{size}_"
                    f"{self._key_name((bay_key,))}"
                ),
            )
            for (status, size), candidates in sorted(
                self.import_reservation_candidates.items()
            )
            for bay_key, capacity in candidates
        }

        templates_by_resource: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        templates_by_class: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
        templates_by_class_area: defaultdict[
            tuple[tuple[str, ...], str], list[int]
        ] = defaultdict(list)
        templates_by_stack_key: defaultdict[
            tuple[str, str, str], list[int]
        ] = defaultdict(list)
        stack_sample_group: dict[tuple[str, str, str], str] = {}
        for template in self._templates:
            templates_by_class[template.handling_class].append(template.template_id)
            templates_by_class_area[
                (template.handling_class, template.area_no)
            ].append(template.template_id)
            for resource in template.footprint:
                templates_by_resource[resource].append(template.template_id)
            representative = self._columns[template.arc_indices[0]]
            stack_value = self._row_mix_key_for_column(representative)
            for bay_key, _row_no in template.footprint:
                stack_key = (bay_key, stack_value, representative.size)
                templates_by_stack_key[stack_key].append(template.template_id)
                stack_sample_group.setdefault(
                    stack_key, representative.group_id
                )

        constraints: dict[str, dict] = defaultdict(dict)
        for template in self._templates:
            constraints["template_capacity_link"][template.template_id] = model.addConstr(
                template_capacity[template.template_id]
                <= template.capacity * template_use[template.template_id],
                name=f"template_capacity_link_{template.template_id}",
            )
        for resource, template_ids in sorted(templates_by_resource.items()):
            constraints["physical_row_template_choice"][resource] = model.addConstr(
                quicksum(template_use[index] for index in template_ids) <= 1.0,
                name=f"template_row_{self._key_name(resource)}",
            )

        templates_by_bay: defaultdict[str, list[int]] = defaultdict(list)
        templates_by_bay_size: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        for template in self._templates:
            representative = self._columns[template.arc_indices[0]]
            for bay_key, _row_no in template.footprint:
                templates_by_bay[bay_key].append(template.template_id)
            templates_by_bay_size[
                (representative.bay_key, representative.size)
            ].append(template.template_id)
        import_by_bay: defaultdict[str, list] = defaultdict(list)
        import_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        import_by_flow_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        import_by_flow_area_size: defaultdict[tuple[str, str, str], list] = defaultdict(list)
        for (status, size, bay_key), variable in import_reserve.items():
            area_no = self.bays[bay_key].area_no
            for footprint_key in self._placement_footprint_keys(bay_key, size):
                import_by_bay[footprint_key].append(variable)
            import_by_bay_size[(bay_key, size)].append(variable)
            import_by_flow_size[(status, size)].append(variable)
            import_by_flow_area_size[(status, area_no, size)].append(variable)
        for bay_key in sorted(self._master_bay_capacity_keys):
            constraints["reserved_bay_capacity"][bay_key] = model.addConstr(
                quicksum(
                    template_capacity[index]
                    for index in templates_by_bay.get(bay_key, [])
                )
                + quicksum(import_by_bay.get(bay_key, []))
                <= int(self.bays[bay_key].physical_capacity),
                name=f"template_reserved_bay_{self._key_name((bay_key,))}",
            )
        for key in sorted(self._master_bay_size_keys):
            bay_key, size = key
            constraints["reserved_bay_size"][key] = model.addConstr(
                quicksum(
                    template_capacity[index]
                    for index in templates_by_bay_size.get(key, [])
                )
                + quicksum(import_by_bay_size.get(key, []))
                <= int(self.bays[bay_key].cap_by_size.get(size, 0)),
                name=f"template_reserved_size_{self._key_name(key)}",
            )

        stack_variables = {}
        stacks_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        for key, template_ids in sorted(templates_by_stack_key.items()):
            bay_key, _stack_value, size = key
            group = self.groups_by_id[stack_sample_group[key]]
            stack_count = self._stack_count_for_group(bay_key, size, group)
            unit_capacity = self._stack_unit_capacity_for_group(
                bay_key, size, group
            )
            if stack_count <= 0 or unit_capacity <= 0:
                continue
            stack_variable = model.addVar(
                lb=0.0,
                ub=float(stack_count),
                vtype="C",
                name=f"template_stack_{self._key_name(key)}",
            )
            stack_variables[key] = stack_variable
            stacks_by_bay_size[(bay_key, size)].append(stack_variable)
            constraints["stack_load_link"][key] = model.addConstr(
                quicksum(
                    template_capacity[index]
                    for index in template_ids
                )
                <= unit_capacity * stack_variable,
                name=f"template_stack_load_{self._key_name(key)}",
            )
        for key, variables in sorted(stacks_by_bay_size.items()):
            constraints["stack_total_limit"][key] = model.addConstr(
                quicksum(variables) <= self._stack_count_for_bay_size(*key),
                name=f"template_stack_total_{self._key_name(key)}",
            )
        for key, required in sorted(self.import_total_by_flow_size.items()):
            constraints["import_total"][key] = model.addConstr(
                quicksum(import_by_flow_size.get(key, [])) == int(required),
                name=f"template_import_total_{self._key_name(key)}",
            )
        constraints["import_guidance"] = self._add_import_reference_deviation(
            quicksum,
            model,
            import_by_flow_area_size,
            objective_mode="full",
        )
        for key, template_ids in sorted(templates_by_class_area.items(), key=str):
            for template_id in template_ids:
                constraints["class_area_link"][(key, template_id)] = model.addConstr(
                    template_use[template_id] <= area_use[key],
                    name=f"template_area_link_{template_id}",
                )
            constraints["class_area_reverse"][key] = model.addConstr(
                area_use[key]
                <= quicksum(template_use[index] for index in template_ids),
                name=(
                    f"template_area_reverse_{self._key_name(key[0])}_"
                    f"{self._key_name((key[1],))}"
                ),
            )
        for class_key, template_ids in sorted(templates_by_class.items(), key=str):
            required = sum(
                int(self.groups_by_id[group_id].demand)
                for group_id in self._class_groups[class_key]
            )
            constraints["class_capacity_balance"][class_key] = model.addConstr(
                quicksum(template_capacity[index] for index in template_ids)
                == required,
                name=f"template_class_balance_{self._key_name(class_key)}",
            )
            class_area_variables = [
                area_use[key]
                for key in templates_by_class_area
                if key[0] == class_key
            ]
            constraints["class_minimum_area_count"][class_key] = model.addConstr(
                quicksum(class_area_variables) >= 1.0,
                name=f"template_class_area_count_{self._key_name(class_key)}",
            )

        attr_uses: defaultdict[tuple[str, str, str], list] = defaultdict(list)
        attr_state: dict[tuple[str, str, str, str], object] = {}
        attr_templates: defaultdict[tuple[str, str, str, str], set[int]] = defaultdict(set)
        for template in self._templates:
            representative = self._columns[template.arc_indices[0]]
            for bay_key, _row_no in template.footprint:
                for attr in self._bay_no_mix_attrs_for_column(representative):
                    scope = self._attr_voyage_scope(attr, representative.voyage_id)
                    value = self._column_attr_value(representative, attr)
                    attr_templates[(bay_key, attr, scope, value)].add(
                        template.template_id
                    )
        for key in sorted(attr_templates):
            state = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="C",
                name=f"template_bay_state_{self._key_name(key)}",
            )
            attr_state[key] = state
            attr_uses[key[:3]].append(state)
            for template_id in sorted(attr_templates[key]):
                constraints["bay_state_link"][(key, template_id)] = model.addConstr(
                    template_use[template_id] <= state,
                    name=f"template_bay_state_link_{template_id}_{self._key_name(key)}",
                )
        for key, states in sorted(attr_uses.items()):
            constraints["bay_state_choice"][key] = model.addConstr(
                quicksum(states) <= 1.0,
                name=f"template_bay_state_choice_{self._key_name(key)}",
            )
        model.update()
        return model, {
            "template_use": template_use,
            "template_capacity": template_capacity,
            "area_use": area_use,
            "theta": theta,
            "fixed_one": fixed_one,
            "attr_state": attr_state,
            "stack_variables": stack_variables,
            "import_reserve": import_reserve,
            "constraints": constraints,
        }

    def _build_recourse(self):
        from gurobipy import quicksum

        model = GurobiModel("yard_template_flow_recourse")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Method", int(self.config.lp_method))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(model, "Threads", int(self.config.solver_threads))
        model.setMinimize()

        flow = {
            index: model.addVar(
                lb=0.0,
                obj=float(column.intrinsic_cost),
                name=f"flow_{index}",
            )
            for index, column in enumerate(self._columns)
        }
        group_shortage = {
            group.group_id: model.addVar(
                lb=0.0,
                obj=1.0,
                name=f"short_group_{self._key_name((group.group_id,))}",
            )
            for group in self.groups
        }

        arcs_by_group: defaultdict[str, list[int]] = defaultdict(list)
        coefficient_rows: defaultdict[
            str, defaultdict[object, list[tuple[int, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        for index, column in enumerate(self._columns):
            arcs_by_group[column.group_id].append(index)
            for section, values in self._placement_master_coefficients(column).items():
                for key, coefficient in values.items():
                    if coefficient:
                        coefficient_rows[section][key].append(
                            (index, float(coefficient))
                        )

        constraints: dict[str, dict] = defaultdict(dict)
        for group in self.groups:
            constraints["group_demand"][group.group_id] = model.addConstr(
                quicksum(flow[index] for index in arcs_by_group[group.group_id])
                + group_shortage[group.group_id]
                == int(group.demand),
                name=f"flow_group_{self._key_name((group.group_id,))}",
            )
        for template in self._templates:
            constraints["template_link"][template.template_id] = model.addConstr(
                quicksum(flow[index] for index in template.arc_indices) <= 0.0,
                name=f"flow_template_{template.template_id}",
            )
        for key in sorted(self._master_area_guidance_keys):
            voyage_id, status, area_no, big_size = key
            target = int(self._template_area_targets.get(key, 0))
            positive = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"flow_guide_pos_{self._key_name(key)}",
            )
            negative = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"flow_guide_neg_{self._key_name(key)}",
            )
            constraints["export_guidance"][key] = model.addConstr(
                quicksum(
                    coefficient * flow[index]
                    for index, coefficient in coefficient_rows[
                        "area_guidance_balance"
                    ].get(key, [])
                )
                - target
                == positive - negative,
                name=f"flow_guide_{self._key_name(key)}",
            )
        model.update()
        shortage_ids = {
            id(variable)
            for variable in group_shortage.values()
        }
        business_terms = [
            (variable, model.getVarObjective(variable))
            for variable in model.getVars()
            if id(variable) not in shortage_ids
            and abs(model.getVarObjective(variable)) > 0.0
        ]
        for variable, _coefficient in business_terms:
            model.setVarObjective(variable, 0.0)
        model.update()
        return model, {
            "flow": flow,
            "group_shortage": group_shortage,
            "business_terms": tuple(business_terms),
            "constraints": constraints,
        }

    @staticmethod
    def _set_variable_upper_bound(variable, value: float) -> None:
        variable.UB = float(value)

    def _activate_recourse_phase(self, model, variables: dict, phase: str) -> None:
        if phase not in {"feasibility", "business"}:
            raise ValueError(f"unknown recourse phase: {phase}")
        shortage = tuple(variables["group_shortage"].values())
        for variable in shortage:
            self._set_variable_upper_bound(
                variable, math.inf if phase == "feasibility" else 0.0
            )
            model.setVarObjective(variable, 1.0 if phase == "feasibility" else 0.0)
        for variable, coefficient in variables["business_terms"]:
            model.setVarObjective(
                variable, coefficient if phase == "business" else 0.0
            )
        model.update()

    def _update_template_links(
        self, recourse_variables: dict, capacity_values: dict[int, float]
    ) -> None:
        for template in self._templates:
            constraint = recourse_variables["constraints"]["template_link"][
                template.template_id
            ]
            constraint.RHS = float(
                max(0.0, capacity_values[template.template_id])
            )

    def _recourse_cut_coefficients(
        self,
        recourse_model,
        recourse_variables: dict,
        capacity_values: dict[int, float],
        objective: float,
    ) -> tuple[float, dict[int, float]]:
        coefficients = {}
        selected_value = 0.0
        for template in self._templates:
            row = recourse_variables["constraints"]["template_link"][
                template.template_id
            ]
            coefficient = float(recourse_model.getLinearDual(row))
            if abs(coefficient) > 1e-12:
                coefficients[template.template_id] = coefficient
            selected_value += coefficient * float(
                capacity_values[template.template_id]
            )
        return float(objective) - selected_value, coefficients

    def _flow_fractionality(self, model, variables: dict) -> float:
        values = [
            self._gurobi_value(model, variable)
            for variable in variables["flow"].values()
        ]
        return max(
            (abs(value - round(value)) for value in values), default=0.0
        )

    def _integer_flow_selection(self, model, variables: dict):
        selected = Counter(
            {
                index: int(round(self._gurobi_value(model, variable)))
                for index, variable in variables["flow"].items()
                if self._gurobi_value(model, variable) > 0.5
            }
        )
        return selected

    def solve(self) -> ColumnGenerationResult:
        from gurobipy import quicksum

        started = perf_counter()
        total_limit = max(0.01, float(self.config.total_time_limit))
        deadline = started + total_limit
        preparation = self._prepare_templates()
        master_build_started = perf_counter()
        master, master_variables = self._build_master()
        master_build_seconds = perf_counter() - master_build_started
        recourse_build_started = perf_counter()
        recourse, recourse_variables = self._build_recourse()
        recourse_build_seconds = perf_counter() - recourse_build_started

        binary_master_variables = (
            *master_variables["template_use"].values(),
            *master_variables["area_use"].values(),
            *master_variables["attr_state"].values(),
        )
        integer_master_variables = (
            *master_variables["template_capacity"].values(),
            *master_variables["stack_variables"].values(),
            *master_variables["import_reserve"].values(),
        )
        for variable in binary_master_variables:
            variable.VType = "B"
        for variable in integer_master_variables:
            variable.VType = "I"
        master.update()
        self._set_gurobi_param(master, "MIPFocus", 1)
        self._set_gurobi_param(master, "Heuristics", 0.5)
        self._set_gurobi_param(master, "SolutionLimit", 1)
        seed_allowance = max(
            0.01,
            (deadline - perf_counter())
            * float(self.benders_config.incumbent_seed_time_fraction),
        )
        self._set_gurobi_param(master, "TimeLimit", seed_allowance)
        master.optimize()
        seed_status = self._gurobi_status_name(master)
        seed_values: dict[int, float] = {}
        seed_objective = None
        seed_bound = None
        seed_selected: Counter[int] | None = None
        seed_imports: Counter[tuple[str, str, str]] | None = None
        seed_recourse_objective = None
        seed_flow_fractionality = math.inf
        seed_recourse_status = "not_solved"
        if self._gurobi_solution_count(master) > 0:
            seed_values = {
                id(variable): self._gurobi_value(master, variable)
                for variable in (*binary_master_variables, *integer_master_variables)
            }
            seed_objective = self._gurobi_objective_value(master)
            seed_bound = self._gurobi_dual_bound(master)
            seed_capacities = {
                template_id: self._gurobi_value(master, variable)
                for template_id, variable in master_variables[
                    "template_capacity"
                ].items()
            }
            seed_imports = Counter(
                {
                    key: int(round(self._gurobi_value(master, variable)))
                    for key, variable in master_variables[
                        "import_reserve"
                    ].items()
                    if self._gurobi_value(master, variable) > 0.5
                }
            )
            self._update_template_links(recourse_variables, seed_capacities)
            self._activate_recourse_phase(
                recourse, recourse_variables, "feasibility"
            )
            self._set_gurobi_param(
                recourse,
                "TimeLimit",
                max(0.01, deadline - perf_counter()),
            )
            recourse.optimize()
            seed_recourse_status = self._gurobi_status_name(recourse)
            if (
                seed_recourse_status == "optimal"
                and self._gurobi_objective_value(recourse)
                <= float(self.benders_config.cut_tolerance)
            ):
                self._activate_recourse_phase(
                    recourse, recourse_variables, "business"
                )
                recourse.optimize()
                seed_recourse_status = self._gurobi_status_name(recourse)
                if seed_recourse_status == "optimal":
                    seed_recourse_objective = self._gurobi_objective_value(
                        recourse
                    )
                    seed_flow_fractionality = self._flow_fractionality(
                        recourse, recourse_variables
                    )
                    if seed_flow_fractionality <= float(
                        self.benders_config.flow_integrality_tolerance
                    ):
                        seed_selected = self._integer_flow_selection(
                            recourse, recourse_variables
                        )
        for variable in (*binary_master_variables, *integer_master_variables):
            variable.VType = "C"
        master.update()
        root_started = perf_counter()
        root_deadline = root_started + max(
            0.0, deadline - root_started
        ) * float(self.benders_config.root_time_fraction)

        best_selected = seed_selected
        best_imports = seed_imports if seed_selected is not None else None
        best_recourse = (
            float(seed_recourse_objective)
            if seed_recourse_objective is not None
            else math.inf
        )
        best_fixed = (
            float(seed_objective)
            if seed_selected is not None and seed_objective is not None
            else math.inf
        )
        best_upper = best_fixed + best_recourse
        best_flow_fractionality = (
            seed_flow_fractionality if seed_selected is not None else math.inf
        )
        valid_lower_bound = (
            float(seed_bound)
            if seed_bound is not None and math.isfinite(float(seed_bound))
            else -math.inf
        )
        feasibility_cut_count = 0
        optimality_cut_count = 0
        rounds = []
        converged = False
        termination = "iteration_limit"
        master_status = "not_solved"
        master_phase = "root_lp"
        root_round_count = 0
        integer_round_count = 0
        root_closed = False
        root_round_limit = min(
            20,
            max(1, int(self.benders_config.max_iterations) // 2),
        )

        def integerize_master() -> None:
            nonlocal master_phase
            for variable in binary_master_variables:
                try:
                    start = seed_values.get(
                        id(variable), self._gurobi_value(master, variable)
                    )
                    variable.Start = 1.0 if start > 0.5 else 0.0
                except Exception:
                    pass
                variable.VType = "B"
            for variable in integer_master_variables:
                try:
                    start = seed_values.get(
                        id(variable), self._gurobi_value(master, variable)
                    )
                    variable.Start = round(start)
                except Exception:
                    pass
                variable.VType = "I"
            master_variables["theta"].Start = 1.0
            master.update()
            self._set_gurobi_param(master, "MIPFocus", 1)
            self._set_gurobi_param(master, "Heuristics", 0.5)
            master_phase = "integer"

        try:
            for iteration in range(1, int(self.benders_config.max_iterations) + 1):
                remaining = deadline - perf_counter()
                if remaining <= 1e-6:
                    termination = "time_limit"
                    break
                if master_phase == "root_lp" and (
                    perf_counter() >= root_deadline
                    or root_deadline - perf_counter() <= 0.05 * total_limit
                ):
                    integerize_master()
                    termination = "root_time_share"
                if master_phase == "root_lp":
                    master_allowance = min(
                        remaining,
                        max(0.01, root_deadline - perf_counter()),
                    )
                else:
                    master_allowance = max(
                        0.01,
                        remaining
                        * float(
                            self.benders_config.integer_master_time_fraction
                        ),
                    )
                    self._set_gurobi_param(
                        master,
                        "SolutionLimit",
                        1 if best_selected is None else 2_000_000_000,
                    )
                self._set_gurobi_param(master, "TimeLimit", master_allowance)
                self._set_gurobi_param(master, "MIPGap", 0.0)
                master.optimize()
                master_status = self._gurobi_status_name(master)
                if self._gurobi_solution_count(master) <= 0:
                    if master_phase == "root_lp" and perf_counter() < deadline:
                        integerize_master()
                        termination = f"root_master_{master_status}"
                        continue
                    termination = f"master_{master_status}"
                    break
                master_bound = self._gurobi_dual_bound(master)
                master_objective = self._gurobi_objective_value(master)
                if math.isfinite(master_bound):
                    valid_lower_bound = max(valid_lower_bound, master_bound)
                template_values = {
                    template_id: self._gurobi_value(master, variable)
                    for template_id, variable in master_variables[
                        "template_use"
                    ].items()
                }
                capacity_values = {
                    template_id: self._gurobi_value(master, variable)
                    for template_id, variable in master_variables[
                        "template_capacity"
                    ].items()
                }
                current_imports = Counter(
                    {
                        key: int(round(self._gurobi_value(master, variable)))
                        for key, variable in master_variables[
                            "import_reserve"
                        ].items()
                        if self._gurobi_value(master, variable) > 0.5
                    }
                )
                selected_templates = {
                    template_id
                    for template_id, value in template_values.items()
                    if value > 0.5
                }
                theta_value = self._gurobi_value(master, master_variables["theta"])
                fixed_cost = master_objective - theta_value
                self._update_template_links(recourse_variables, capacity_values)

                remaining = deadline - perf_counter()
                if remaining <= 1e-6:
                    termination = "time_limit"
                    break
                self._activate_recourse_phase(
                    recourse, recourse_variables, "feasibility"
                )
                self._set_gurobi_param(recourse, "TimeLimit", max(0.01, remaining))
                recourse.optimize()
                recourse_status = self._gurobi_status_name(recourse)
                if recourse_status != "optimal":
                    termination = f"feasibility_recourse_{recourse_status}"
                    break
                phase_one = self._gurobi_objective_value(recourse)
                cut_kind = "none"
                cut_violation = 0.0
                flow_fractionality = None
                recourse_objective = None
                if phase_one > float(self.benders_config.cut_tolerance):
                    intercept, coefficients = self._recourse_cut_coefficients(
                        recourse,
                        recourse_variables,
                        capacity_values,
                        phase_one,
                    )
                    expression = intercept + quicksum(
                        coefficient
                        * master_variables["template_capacity"][template_id]
                        for template_id, coefficient in coefficients.items()
                    )
                    master.addConstr(
                        expression <= 0.0,
                        name=f"template_flow_feasibility_cut_{iteration}",
                    )
                    master.update()
                    feasibility_cut_count += 1
                    cut_kind = "feasibility"
                    cut_violation = phase_one
                else:
                    self._activate_recourse_phase(
                        recourse, recourse_variables, "business"
                    )
                    remaining = deadline - perf_counter()
                    if remaining <= 1e-6:
                        termination = "time_limit"
                        break
                    self._set_gurobi_param(
                        recourse, "TimeLimit", max(0.01, remaining)
                    )
                    recourse.optimize()
                    recourse_status = self._gurobi_status_name(recourse)
                    if recourse_status != "optimal":
                        termination = f"business_recourse_{recourse_status}"
                        break
                    recourse_objective = self._gurobi_objective_value(recourse)
                    flow_fractionality = self._flow_fractionality(
                        recourse, recourse_variables
                    )
                    intercept, coefficients = self._recourse_cut_coefficients(
                        recourse,
                        recourse_variables,
                        capacity_values,
                        recourse_objective,
                    )
                    cut_violation = max(0.0, recourse_objective - theta_value)
                    if cut_violation > float(self.benders_config.cut_tolerance):
                        master.addConstr(
                            master_variables["theta"]
                            >= intercept
                            + quicksum(
                                coefficient
                                * master_variables["template_capacity"][template_id]
                                for template_id, coefficient in coefficients.items()
                            ),
                            name=f"template_flow_optimality_cut_{iteration}",
                        )
                        master.update()
                        optimality_cut_count += 1
                        cut_kind = "optimality"

                    if master_phase == "integer" and flow_fractionality <= float(
                        self.benders_config.flow_integrality_tolerance
                    ):
                        selected = self._integer_flow_selection(
                            recourse, recourse_variables
                        )
                        imports = current_imports
                        self._final_import_reservation = imports
                        model_upper = fixed_cost + recourse_objective
                        if model_upper + 1e-10 < best_upper:
                            best_upper = model_upper
                            best_recourse = recourse_objective
                            best_fixed = fixed_cost
                            best_selected = selected
                            best_imports = imports
                            best_flow_fractionality = flow_fractionality

                    if (
                        cut_violation
                        <= float(self.benders_config.cut_tolerance)
                        and master_status == "optimal"
                    ):
                        if master_phase == "root_lp":
                            root_closed = True
                        else:
                            converged = True
                            termination = "optimal"

                if master_phase == "root_lp":
                    root_round_count += 1
                else:
                    integer_round_count += 1
                rounds.append(
                    {
                        "iteration": iteration,
                        "master_phase": master_phase,
                        "master_status": master_status,
                        "master_allowance": master_allowance,
                        "master_objective": master_objective,
                        "master_bound": master_bound,
                        "positive_template_count": sum(
                            1 for value in template_values.values() if value > 1e-8
                        ),
                        "selected_template_count": len(selected_templates),
                        "phase_one_shortage": phase_one,
                        "recourse_status": recourse_status,
                        "recourse_objective": recourse_objective,
                        "flow_max_fractionality": flow_fractionality,
                        "cut_kind": cut_kind,
                        "cut_violation": cut_violation,
                        "elapsed_seconds": perf_counter() - started,
                    }
                )
                if master_phase == "root_lp" and (
                    root_closed
                    or root_round_count >= root_round_limit
                    or perf_counter() >= root_deadline
                ):
                    integerize_master()
                    termination = (
                        "root_closed" if root_closed else "root_round_limit"
                    )
                    continue
                if converged:
                    break
        finally:
            self._free_gurobi_model(master)
            self._free_gurobi_model(recourse)

        if best_selected is None or best_imports is None:
            raise RuntimeError(
                "template-flow Benders did not obtain an integral feasible "
                f"flow before termination: reason={termination}, "
                f"best_fractionality={best_flow_fractionality}, "
                f"rounds={rounds[-3:]}"
            )
        self._final_import_reservation = best_imports
        lower = valid_lower_bound if math.isfinite(valid_lower_bound) else 0.0
        lower = min(lower, best_upper)
        absolute_gap = max(0.0, best_upper - lower)
        diagnostics = {
            "algorithm": "row_template_classical_benders_gurobi",
            "model_scope": "hard_state_row_templates_with_continuous_box_flow",
            "formulation": "row_template_master_and_continuous_flow_recourse",
            "decomposition": "classical_benders",
            "master_status": "optimal" if converged else master_status,
            "planned_group_count": len(self.groups),
            "planned_box_count": sum(group.demand for group in self.groups),
            "candidate_row_location_count": len(self._columns),
            "template_flow_handling_class_definition": (
                "export_voyage|flow|hard_bay_and_row_no_mix_attributes"
            ),
            "template_flow_template_count": len(self._templates),
            "template_flow_arc_count": len(self._columns),
            "template_flow_handling_class_count": len(self._class_groups),
            "template_flow_converged": converged,
            "template_flow_termination_reason": termination,
            "template_flow_iteration_count": len(rounds),
            "template_flow_root_closed": root_closed,
            "template_flow_root_round_count": root_round_count,
            "template_flow_integer_round_count": integer_round_count,
            "template_flow_feasibility_cut_count": feasibility_cut_count,
            "template_flow_optimality_cut_count": optimality_cut_count,
            "template_flow_rounds": rounds,
            "template_flow_best_fixed_template_cost": best_fixed,
            "template_flow_best_recourse_cost": best_recourse,
            "template_flow_best_max_fractionality": best_flow_fractionality,
            "template_flow_model_objective": best_upper,
            "complete_model_lower_bound": lower,
            "complete_model_absolute_gap": absolute_gap,
            "complete_model_relative_gap": absolute_gap
            / max(abs(best_upper), 1e-12),
            "complete_model_gap_source": "benders_master_bound",
            "template_flow_preparation_seconds": preparation[
                "preparation_seconds"
            ],
            "template_flow_master_build_seconds": master_build_seconds,
            "template_flow_recourse_build_seconds": recourse_build_seconds,
            "template_flow_seed_status": seed_status,
            "template_flow_seed_allowance": seed_allowance,
            "template_flow_seed_objective": seed_objective,
            "template_flow_seed_recourse_status": seed_recourse_status,
            "template_flow_seed_recourse_objective": seed_recourse_objective,
            "template_flow_total_seconds": perf_counter() - started,
            "template_flow_time_policy": {
                "type": "dimensionless_shared_deadline",
                "incumbent_seed_time_fraction_of_post_build_remaining": float(
                    self.benders_config.incumbent_seed_time_fraction
                ),
                "root_budget_basis": "post_build_remaining_time",
                "root_time_fraction": float(
                    self.benders_config.root_time_fraction
                ),
                "integer_master_time_fraction_of_remaining": float(
                    self.benders_config.integer_master_time_fraction
                ),
            },
            "hard_demand_balance": True,
            "gurobi_available": True,
            "business_objective_normalization": {
                "weights": self._objective_weights(),
                "scales": dict(self._objective_scales),
                "method": "hard_handling_class_natural_instance_scale",
            },
            "business_objective": {
                "type": "template_concentration_plus_continuous_flow_cost",
                "hard_demand_balance": True,
                "concentration_scope": "hard_handling_class",
                "weights": self._objective_weights(),
            },
            "import_capacity_reservation": {
                "source_quantity_field": "new_qty",
                "role": "anonymous_size_compatible_capacity_only",
                "area_policy": "weighted_l1_deviation_from_big_plan_reference",
                "constraint_scope": ["area_function", "bay_size", "physical_capacity"],
                "excluded_constraints": [
                    "bay_no_mix",
                    "row_no_mix",
                    "container_group_attributes",
                ],
                "import_boxes": int(sum(self.import_area_size_reference.values())),
            },
        }
        result = self._assemble_result(best_selected, diagnostics)
        realized = float(result.diagnostics["final_business_objective"])
        result.diagnostics["template_flow_realized_allocation_objective"] = realized
        result.diagnostics["final_business_objective"] = round(best_upper, 8)
        result.diagnostics["master_objective"] = round(best_upper, 8)
        result.columns = self._selected_direct_columns(best_selected)
        return result


class TemplateFlowDirectPlanner(TemplateFlowBendersPlanner):
    """Compact direct reference for exactly the redefined template model."""

    def solve(self) -> ColumnGenerationResult:
        started = perf_counter()
        preparation = self._prepare_templates()
        configured_limit = float(self.config.total_time_limit)
        self.config.total_time_limit = max(
            0.01, configured_limit - (perf_counter() - started)
        )
        try:
            selected, solve_stats = self._solve_direct_milp()
        finally:
            self.config.total_time_limit = configured_limit
        diagnostics = {
            "algorithm": "row_template_compact_direct_gurobi",
            "model_scope": "hard_handling_class_integer_area_target",
            "formulation": "complete_integer_row_location_milp",
            "decomposition": "none",
            "master_status": solve_stats.get("master_status", "unknown"),
            "planned_group_count": len(self.groups),
            "planned_box_count": sum(group.demand for group in self.groups),
            "candidate_row_location_count": len(self._columns),
            "template_flow_handling_class_count": len(self._class_groups),
            "template_flow_template_count": len(self._templates),
            "template_flow_preparation_seconds": preparation[
                "preparation_seconds"
            ],
            "template_flow_total_seconds": perf_counter() - started,
            "hard_demand_balance": True,
            "gurobi_available": True,
            "business_objective_normalization": {
                "weights": self._objective_weights(),
                "scales": dict(self._objective_scales),
                "method": "hard_handling_class_natural_instance_scale",
            },
            "business_objective": {
                "type": "integer_target_hard_class_concentration",
                "hard_demand_balance": True,
                "concentration_scope": "hard_handling_class",
                "weights": self._objective_weights(),
            },
            "import_capacity_reservation": {
                "source_quantity_field": "new_qty",
                "role": "anonymous_size_compatible_capacity_only",
                "area_policy": "integer_target_weighted_l1_deviation",
                "constraint_scope": ["area_function", "bay_size", "physical_capacity"],
                "excluded_constraints": [
                    "bay_no_mix",
                    "row_no_mix",
                    "container_group_attributes",
                ],
                "import_boxes": int(sum(self.import_area_size_reference.values())),
            },
            **solve_stats,
        }
        result = self._assemble_result(selected, diagnostics)
        result.columns = self._selected_direct_columns(selected)
        return result


__all__ = [
    "RowTemplate",
    "TemplateFlowBendersConfig",
    "TemplateFlowBendersPlanner",
    "TemplateFlowDirectPlanner",
]
