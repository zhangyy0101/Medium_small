"""Independent contiguous-zone generation experiment.

The experiment deliberately changes the concentration representation.  A
hard export group first reserves one or more contiguous runs of compatible
physical row footprints.  The restricted original row MILP then fills only
the selected runs.  The zone master is a dedicated-capacity support model:
selected row footprints reserve their complete compatible capacity, which
makes all physical conflicts additive and gives a genuine column-generation
structure.

Nothing in this module is registered as a production solver.  It is a stage
gate for deciding whether the new model boundary deserves a Branch-and-Price
implementation.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from time import perf_counter

from .direct_milp import DirectMilpPlanner
from .gurobi_backend import GurobiModel
from .planner import ColumnGenerationConfig, ColumnGenerationResult

Resource = tuple[str, str]
StripKey = tuple[str, str, str]


@dataclass(frozen=True)
class ContiguousZoneConfig:
    """Dimensionless controls for the isolated root-stage experiment."""

    max_root_iterations: int = 60
    reduced_cost_tolerance: float = 1e-8
    columns_per_group_per_round: int = 3
    integer_pool_columns_per_group: int = 100
    root_time_fraction: float = 0.40
    zone_mip_time_fraction: float = 0.65
    shortage_penalty: float = 1_000.0
    unused_capacity_weight: float = 0.02

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
        if float(self.shortage_penalty) <= 0.0:
            raise ValueError("shortage_penalty must be positive")
        if float(self.unused_capacity_weight) < 0.0:
            raise ValueError("unused_capacity_weight cannot be negative")


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
    bay_loads: tuple[tuple[str, int], ...]
    bay_size_loads: tuple[tuple[tuple[str, str], int], ...]
    stack_uses: tuple[tuple[tuple[str, str], int], ...]
    bay_attr_uses: tuple[tuple[tuple[str, str, str, str], int], ...]
    quota_key: tuple[str, str, str, str]
    objective_cost: float


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
        self._zones: tuple[ContiguousZone, ...] = ()
        self._zone_indices_by_group: defaultdict[str, list[int]] = defaultdict(list)
        self._zone_indices_by_strip: defaultdict[StripKey, list[int]] = defaultdict(list)

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
        bay_loads: Counter[str] = Counter()
        bay_size_loads: Counter[tuple[str, str]] = Counter()
        stack_uses: Counter[tuple[str, str]] = Counter()
        bay_attr_uses: Counter[tuple[str, str, str, str]] = Counter()
        capacity = 0
        variable_cost = 0.0
        quota_keys = set()
        for index in candidate_indices:
            column = self._columns[index]
            row_capacity = int(self._base_location_capacity(group, column))
            if row_capacity <= 0:
                raise RuntimeError(f"zone contains a zero-capacity candidate: {index}")
            capacity += row_capacity
            variable_cost += row_capacity * float(column.intrinsic_cost)
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
        zone_penalty = float(self.config.row_dispersion_weight) / max(
            1, len(self.groups)
        )
        return ContiguousZone(
            zone_id=-1,
            group_id=group_id,
            area_no=area_no,
            row_no=row_no,
            candidate_indices=candidate_indices,
            capacity=int(capacity),
            resources=tuple(sorted(resources)),
            bay_loads=tuple(sorted(bay_loads.items())),
            bay_size_loads=tuple(sorted(bay_size_loads.items())),
            stack_uses=tuple(sorted(stack_uses.items())),
            bay_attr_uses=tuple(sorted(bay_attr_uses.items())),
            quota_key=next(iter(quota_keys)),
            objective_cost=float(zone_penalty + variable_cost),
        )

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

        zones: list[ContiguousZone] = []
        self._zone_indices_by_group.clear()
        self._zone_indices_by_strip.clear()
        dominated_long_zone_count = 0
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
            for run in self._split_contiguous_runs(indices):
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
                        zone = self._make_zone(
                            group_id,
                            area_no,
                            row_no,
                            tuple(run[start : end + 1]),
                        )
                        zone_id = len(zones)
                        zone = replace(zone, zone_id=zone_id)
                        zones.append(zone)
                        self._zone_indices_by_group[group_id].append(zone_id)
                        self._zone_indices_by_strip[strip_key].append(zone_id)
        self._zones = tuple(zones)
        if not self._zones and self.groups:
            raise RuntimeError("contiguous-zone model has no candidate zones")
        return {
            "preparation_seconds": perf_counter() - started,
            "atomic_candidate_count": len(self._columns),
            "strip_count": len(strip_candidates),
            "zone_count": len(self._zones),
            "dominated_long_zone_count": dominated_long_zone_count,
        }

    def _master_index_sets(self) -> dict[str, object]:
        physical_resources = sorted(
            {resource for zone in self._zones for resource in zone.resources}
        )
        stack_keys = sorted(
            {key for zone in self._zones for key, _value in zone.stack_uses}
        )
        attr_keys = sorted(
            {key for zone in self._zones for key, _value in zone.bay_attr_uses}
        )
        attr_scopes: defaultdict[tuple[str, str, str], list] = defaultdict(list)
        for key in attr_keys:
            attr_scopes[key[:3]].append(key)
        area_pairs = sorted(
            {(zone.group_id, zone.area_no) for zone in self._zones}
        )
        zones_by_area: Counter[tuple[str, str]] = Counter(
            (zone.group_id, zone.area_no) for zone in self._zones
        )
        return {
            "physical_resources": physical_resources,
            "stack_keys": stack_keys,
            "attr_keys": attr_keys,
            "attr_scopes": attr_scopes,
            "area_pairs": area_pairs,
            "zones_by_area": zones_by_area,
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
        unused_unit_cost = float(self.zone_config.unused_capacity_weight) / max(
            1, sum(group.demand for group in self.groups)
        )
        shortage = {
            group.group_id: model.addVar(
                lb=0.0,
                obj=float(self.zone_config.shortage_penalty),
                name=f"zone_short_{self._key_name((group.group_id,))}",
            )
            for group in self.groups
        }
        excess = {
            group.group_id: model.addVar(
                lb=0.0,
                obj=unused_unit_cost,
                name=f"zone_excess_{self._key_name((group.group_id,))}",
            )
            for group in self.groups
        }
        area_use = {
            key: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="C",
                obj=self._area_activation_penalty(),
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

        constraints: dict[str, dict] = defaultdict(dict)
        for group in self.groups:
            constraints["group_balance"][group.group_id] = model.addConstr(
                shortage[group.group_id] - excess[group.group_id]
                == int(group.demand),
                name=f"zone_group_{self._key_name((group.group_id,))}",
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
        for scope, keys in sorted(sets["attr_scopes"].items()):
            constraints["attr_choice"][scope] = model.addConstr(
                quicksum(attr_state[key] for key in keys) <= 1.0,
                name=f"zone_attr_choice_{self._key_name(scope)}",
            )
        for key in sets["area_pairs"]:
            big_m = max(1, int(sets["zones_by_area"][key]))
            constraints["area_link"][key] = model.addConstr(
                zero <= big_m * area_use[key],
                name=f"zone_area_link_{self._key_name(key)}",
            )
        for key in sorted(self._master_area_guidance_keys):
            target = self._area_size_target(*key)
            positive = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"zone_guide_pos_{self._key_name(key)}",
            )
            negative = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"zone_guide_neg_{self._key_name(key)}",
            )
            constraints["export_guidance"][key] = model.addConstr(
                zero - float(target) == positive - negative,
                name=f"zone_guide_{self._key_name(key)}",
            )
        for key, required in sorted(self.import_total_by_flow_size.items()):
            constraints["import_total"][key] = model.addConstr(
                quicksum(import_by_flow_size.get(key, [])) == int(required),
                name=f"zone_import_total_{self._key_name(key)}",
            )
        constraints["import_guidance"] = self._add_import_reference_deviation(
            quicksum,
            model,
            import_by_flow_area_size,
            objective_mode="full",
        )
        model.update()
        return model, {
            "shortage": shortage,
            "excess": excess,
            "area_use": area_use,
            "attr_state": attr_state,
            "import_reserve": import_reserve,
            "constraints": constraints,
            "zone": {},
            "active_zone_indices": set(),
        }

    def _zone_coefficients(
        self, zone: ContiguousZone
    ) -> tuple[tuple[str, object, float], ...]:
        coefficients: list[tuple[str, object, float]] = [
            ("group_balance", zone.group_id, float(zone.capacity)),
            ("area_link", (zone.group_id, zone.area_no), 1.0),
            ("export_guidance", zone.quota_key, float(zone.capacity)),
        ]
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
            "group_balance",
            "physical_resource",
            "bay_capacity",
            "bay_size",
            "stack_count",
            "attr_link",
            "area_link",
            "export_guidance",
        ):
            for key, row in constraints.get(section, {}).items():
                duals[(section, key)] = float(model.getLinearDual(row))
        return duals

    def _zone_reduced_cost(
        self,
        zone: ContiguousZone,
        duals: dict[tuple[str, object], float],
    ) -> float:
        value = float(zone.objective_cost)
        for section, key, coefficient in self._zone_coefficients(zone):
            value -= float(coefficient) * float(duals.get((section, key), 0.0))
        return value

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
            improving_by_group: defaultdict[str, list[tuple[float, int]]] = defaultdict(list)
            minimum_reduced_cost = math.inf
            for zone in self._zones:
                if zone.zone_id in variables["active_zone_indices"]:
                    continue
                reduced_cost = self._zone_reduced_cost(zone, duals)
                minimum_reduced_cost = min(minimum_reduced_cost, reduced_cost)
                if reduced_cost < -float(self.zone_config.reduced_cost_tolerance):
                    improving_by_group[zone.group_id].append(
                        (reduced_cost, zone.zone_id)
                    )
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
            average_zone_count = len(self._zones) / max(1, len(self.groups))
            phase_one_batch = max(
                int(self.zone_config.columns_per_group_per_round),
                min(12, int(math.ceil(math.sqrt(average_zone_count)))),
            )
            batch = (
                phase_one_batch
                if shortage_value > 1e-6
                else int(self.zone_config.columns_per_group_per_round)
            )
            selected = []
            for group_id in sorted(improving_by_group):
                selected.extend(sorted(improving_by_group[group_id])[:batch])
            rounds.append(
                {
                    "iteration": iteration,
                    "status": status,
                    "objective": self._gurobi_objective_value(model),
                    "shortage": shortage_value,
                    "minimum_reduced_cost": minimum_reduced_cost,
                    "improving_zone_count": sum(
                        len(values) for values in improving_by_group.values()
                    ),
                    "added_zone_count": len(selected),
                    "columns_per_group_batch": batch,
                    "active_zone_count": len(variables["active_zone_indices"]),
                }
            )
            if not selected:
                closed = True
                break
            for _reduced_cost, zone_index in selected:
                self._add_zone_variable(model, variables, zone_index)
            model.update()
        root_objective = last_root_objective
        root_shortage = last_root_shortage
        if self._gurobi_solution_count(model) > 0:
            root_objective = self._gurobi_objective_value(model)
            root_shortage = sum(
                self._gurobi_value(model, variable)
                for variable in variables["shortage"].values()
            )
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
        limit = int(self.zone_config.integer_pool_columns_per_group)
        if limit <= 0:
            return {"added_zone_count": 0, "columns_per_group_limit": limit}
        if self._gurobi_solution_count(model) > 0:
            duals = self._dual_snapshot(model, variables["constraints"])
        else:
            duals = variables.get("last_root_duals")
        if duals is None:
            return {
                "added_zone_count": 0,
                "columns_per_group_limit": limit,
                "reason": "no_valid_root_duals",
            }
        ranked: defaultdict[str, list[tuple[float, int]]] = defaultdict(list)
        for zone in self._zones:
            if zone.zone_id in variables["active_zone_indices"]:
                continue
            ranked[zone.group_id].append(
                (self._zone_reduced_cost(zone, duals), zone.zone_id)
            )
        selected = []
        for group_id in sorted(ranked):
            selected.extend(sorted(ranked[group_id])[:limit])
        for _reduced_cost, zone_index in selected:
            self._add_zone_variable(model, variables, zone_index)
        model.update()
        return {
            "added_zone_count": len(selected),
            "columns_per_group_limit": limit,
            "largest_added_reduced_cost": max(
                (value for value, _zone_index in selected), default=None
            ),
            "active_zone_count_after_enrichment": len(
                variables["active_zone_indices"]
            ),
        }

    def _greedy_zone_mip_start(self, model, variables: dict) -> dict[str, object]:
        """Build a deterministic, LP-guided integer support for Gurobi repair.

        The start protects the root LP's anonymous import allocation while it
        packs whole export zones.  Gurobi is allowed to repair the deliberately
        partial start (helper and import variables are not fixed).
        """

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
        for group_id, variable in variables["shortage"].items():
            variable.Start = 0.0
            excess = max(
                0,
                int(best_covered.get(group_id, 0))
                - int(self.groups_by_id[group_id].demand),
            )
            variables["excess"][group_id].Start = float(excess)
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
        }

    def _solve_complete_zone_lp(self, deadline: float) -> dict[str, object]:
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
                "zone_count": len(self._zones),
            }
        finally:
            self._free_gurobi_model(model)

    def _solve_complete_zone_mip(self, deadline: float) -> dict[str, object]:
        model, variables = self._build_zone_master()
        try:
            for zone in self._zones:
                self._add_zone_variable(model, variables, zone.zone_id)
            model.update()
            selected, stats = self._integerize_zone_master(
                model, variables, deadline
            )
            stats["zone_count"] = len(self._zones)
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
        model, variables = self._build_zone_master()
        try:
            for zone in self._zones:
                self._add_zone_variable(model, variables, zone.zone_id)
            model.update()
            selected, stats = self._integerize_zone_master(
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
    ) -> tuple[set[int], dict[str, object]]:
        mip_start = self._greedy_zone_mip_start(model, variables)
        for variable in variables["zone"].values():
            variable.VType = "B"
        for variable in variables["area_use"].values():
            variable.VType = "B"
        for variable in variables["attr_state"].values():
            variable.VType = "B"
        for variable in variables["import_reserve"].values():
            variable.VType = "I"
        for variable in variables["shortage"].values():
            variable.UB = 0.0
        model.update()
        remaining = deadline - perf_counter()
        if remaining <= 1e-6:
            return set(), {"status": "time_limit_before_zone_mip"}
        self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
        self._set_gurobi_param(model, "MIPGap", 0.0)
        self._set_gurobi_param(model, "MIPFocus", 1)
        self._set_gurobi_param(model, "Heuristics", 0.20)
        model.optimize()
        status = self._gurobi_status_name(model)
        if self._gurobi_solution_count(model) <= 0:
            return set(), {
                "status": status,
                "has_solution": False,
                "mip_start": mip_start,
            }
        selected = {
            zone_index
            for zone_index, variable in variables["zone"].items()
            if self._gurobi_value(model, variable) > 0.5
        }
        return selected, {
            "status": status,
            "has_solution": True,
            "objective": self._gurobi_objective_value(model),
            "bound": self._gurobi_dual_bound(model),
            "selected_zone_count": len(selected),
            "mip_start": mip_start,
        }

    def _solve_restricted_fill(
        self,
        candidate_indices: set[int],
        deadline: float,
    ) -> tuple[Counter[int], dict[str, object]]:
        from gurobipy import quicksum

        ordered_indices = sorted(candidate_indices)
        row_locations = [self._columns[index] for index in ordered_indices]
        started = perf_counter()
        model, variables, model_stats = self.build_compact_row_milp(
            row_locations,
            GurobiModel,
            quicksum,
        )
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
            selected_zones, zone_mip = self._integerize_zone_master(
                model, variables, zone_mip_deadline
            )
        finally:
            self._free_gurobi_model(model)
        if not selected_zones:
            raise RuntimeError(
                "restricted zone master did not obtain an integer support: "
                f"{zone_mip}"
            )
        candidate_indices = {
            index
            for zone_index in selected_zones
            for index in self._zones[zone_index].candidate_indices
        }
        selected, fill = self._solve_restricted_fill(candidate_indices, deadline)
        zone_upper_bound = float(zone_mip["objective"])
        zone_global_lower_bound = float(root["root_objective"])
        zone_absolute_gap = max(0.0, zone_upper_bound - zone_global_lower_bound)
        zone_relative_gap = zone_absolute_gap / max(abs(zone_upper_bound), 1e-12)
        diagnostics = {
            "algorithm": "contiguous_zone_generation_and_exact_fill",
            "model_scope": "dedicated_contiguous_row_zone_support_then_original_fill",
            "formulation": "zone_support_master_plus_restricted_exact_row_milp",
            "decomposition": "column_generation_then_exact_fill",
            "planned_group_count": len(self.groups),
            "planned_box_count": sum(group.demand for group in self.groups),
            "candidate_row_location_count": len(self._columns),
            "zone_preparation": preparation,
            "zone_root": root,
            "zone_pool_enrichment": pool_enrichment,
            "zone_mip": zone_mip,
            "zone_model_upper_bound": zone_upper_bound,
            "zone_model_global_lower_bound": zone_global_lower_bound,
            "zone_model_absolute_gap": zone_absolute_gap,
            "zone_model_relative_gap": zone_relative_gap,
            "zone_model_lower_bound_source": "exact_column_generation_root_lp",
            "zone_restricted_pool_bound": zone_mip.get("bound"),
            "zone_restricted_pool_bound_is_global": False,
            "zone_selected_candidate_count": len(candidate_indices),
            "zone_candidate_reduction": 1.0
            - len(candidate_indices) / max(1, len(self._columns)),
            "zone_fill": fill,
            "master_status": fill["status"],
            "master_objective": fill["objective"],
            "complete_model_lower_bound": None,
            "complete_model_absolute_gap": None,
            "complete_model_relative_gap": None,
            "complete_model_gap_source": (
                "not_certified_for_original_M0; selected-zone fill bound is local"
            ),
            "restricted_fill_lower_bound": fill["bound"],
            "restricted_fill_absolute_gap": fill["absolute_gap"],
            "restricted_fill_relative_gap": fill["relative_gap"],
            "hard_demand_balance": True,
            "business_objective_normalization": {
                "weights": self._objective_weights(),
                "scales": dict(self._objective_scales),
                "method": "natural_instance_scale",
            },
            "business_objective": self._business_objective_specification(),
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
        result.columns = self._selected_direct_columns(selected)
        return result


__all__ = [
    "ContiguousZone",
    "ContiguousZoneConfig",
    "ContiguousZoneGenerationPlanner",
]
