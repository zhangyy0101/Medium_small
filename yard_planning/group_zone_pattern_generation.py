"""Independent complete-group contiguous-zone pattern generation experiment.

One Dantzig--Wolfe column is a complete integer plan for one export group: it
selects all of that group's dedicated contiguous zones and routes every
declared box to anchor bays.  The master chooses one pattern per group and
coordinates physical rows, bay/size/stack resources, hard bay states, large-
plan guidance, and anonymous import reservation.

The module deliberately does not register a production solver and never calls
M0 or the single-zone algorithm as a repair/fallback chain.  It subclasses the
single-zone experiment only to reuse its common data preparation, exact row
recourse, and independently validated output assembly.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from time import perf_counter

from .contiguous_zone_generation import ContiguousZoneGenerationPlanner
from .gurobi_backend import GurobiModel
from .planner import ColumnGenerationConfig, ColumnGenerationResult


@dataclass(frozen=True)
class GroupZonePatternConfig:
    """Dimensionless controls for the isolated group-pattern stage gate."""

    max_root_iterations: int = 80
    patterns_per_group_per_round: int = 3
    reduced_cost_tolerance: float = 1e-6
    root_time_fraction: float = 0.60
    fill_time_fraction: float = 0.05
    max_integer_patterns_per_group: int = 40
    fix_optimize_time_fraction: float = 0.15
    fix_optimize_group_count: int = 12

    def validate(self) -> None:
        if int(self.max_root_iterations) <= 0:
            raise ValueError("max_root_iterations must be positive")
        if int(self.patterns_per_group_per_round) <= 0:
            raise ValueError("patterns_per_group_per_round must be positive")
        if float(self.reduced_cost_tolerance) <= 0.0:
            raise ValueError("reduced_cost_tolerance must be positive")
        if not 0.0 < float(self.root_time_fraction) < 1.0:
            raise ValueError("root_time_fraction must lie strictly between 0 and 1")
        if not 0.0 < float(self.fill_time_fraction) < 1.0:
            raise ValueError("fill_time_fraction must lie strictly between 0 and 1")
        if int(self.max_integer_patterns_per_group) < 0:
            raise ValueError("max_integer_patterns_per_group cannot be negative")
        if not 0.0 <= float(self.fix_optimize_time_fraction) < 1.0:
            raise ValueError("fix_optimize_time_fraction must lie in [0, 1)")
        if int(self.fix_optimize_group_count) <= 0:
            raise ValueError("fix_optimize_group_count must be positive")


@dataclass(frozen=True)
class GroupZonePattern:
    pattern_id: int
    group_id: str
    zone_indices: tuple[int, ...]
    export_flow: tuple[tuple[str, int], ...]
    resources: tuple[tuple[str, str], ...]
    bay_loads: tuple[tuple[str, int], ...]
    bay_size_loads: tuple[tuple[tuple[str, str], int], ...]
    stack_uses: tuple[tuple[tuple[str, str], int], ...]
    bay_attr_uses: tuple[
        tuple[tuple[str, str, str, str], int], ...
    ]
    guidance_loads: tuple[tuple[tuple[str, str, str, str], int], ...]
    used_areas: tuple[str, ...]
    candidate_indices: tuple[int, ...]
    objective_cost: float


@dataclass
class _GroupPricingModel:
    group_id: str
    model: GurobiModel
    zone: dict[int, object]
    flow: dict[str, object]
    area_use: dict[str, object]
    dispersion_baseline: object


class GroupZonePatternGenerationPlanner(ContiguousZoneGenerationPlanner):
    """Generate complete group-zone plans and coordinate them in one master."""

    def __init__(
        self,
        problem,
        config: ColumnGenerationConfig | None = None,
        pattern_config: GroupZonePatternConfig | None = None,
    ) -> None:
        super().__init__(problem, config)
        self.pattern_config = pattern_config or GroupZonePatternConfig()
        self.pattern_config.validate()
        self._patterns: list[GroupZonePattern] = []
        self._pattern_ids_by_group: defaultdict[str, list[int]] = defaultdict(list)
        self._pattern_id_by_signature: dict[tuple, int] = {}
        self._pricing_models: dict[str, _GroupPricingModel] = {}
        self._flow_column_by_key: dict[tuple[str, str], object] = {}

    def _prepare_group_patterns(self) -> dict[str, object]:
        started = perf_counter()
        zone_preparation = self._prepare_zones()
        materialization_started = perf_counter()
        self._materialize_all_zones()
        sets = self._master_index_sets()
        self._flow_column_by_key = dict(sets["flow_columns"])
        self._patterns.clear()
        self._pattern_ids_by_group.clear()
        self._pattern_id_by_signature.clear()
        self._pricing_models.clear()
        return {
            **zone_preparation,
            "materialized_zone_count": len(self._zones),
            "zone_materialization_seconds": (
                perf_counter() - materialization_started
            ),
            "total_preparation_seconds": perf_counter() - started,
        }

    def _make_group_pattern(
        self,
        group_id: str,
        zone_indices: set[int] | tuple[int, ...],
        export_flow: dict[str, int],
    ) -> GroupZonePattern:
        group = self.groups_by_id[group_id]
        ordered_zones = tuple(sorted(int(index) for index in zone_indices))
        positive_flow = {
            str(bay_key): int(quantity)
            for bay_key, quantity in export_flow.items()
            if int(quantity) > 0
        }
        if sum(positive_flow.values()) != int(group.demand):
            raise RuntimeError(
                "group pattern does not assign complete demand: "
                f"group={group_id}, assigned={sum(positive_flow.values())}, "
                f"demand={group.demand}"
            )
        resources: set[tuple[str, str]] = set()
        bay_loads: Counter[str] = Counter()
        bay_size_loads: Counter[tuple[str, str]] = Counter()
        stack_uses: Counter[tuple[str, str]] = Counter()
        bay_attr_uses: Counter[tuple[str, str, str, str]] = Counter()
        anchor_capacity: Counter[str] = Counter()
        candidate_indices: set[int] = set()
        reserved_capacity = 0
        for zone_index in ordered_zones:
            zone = self._zones[zone_index]
            if zone.group_id != group_id:
                raise RuntimeError("group pattern contains a foreign zone")
            if any(resource in resources for resource in zone.resources):
                raise RuntimeError("group pattern contains overlapping zones")
            resources.update(zone.resources)
            reserved_capacity += int(zone.capacity)
            candidate_indices.update(zone.candidate_indices)
            anchor_capacity.update(dict(zone.anchor_bay_loads))
            bay_loads.update(dict(zone.bay_loads))
            bay_size_loads.update(dict(zone.bay_size_loads))
            stack_uses.update(dict(zone.stack_uses))
            bay_attr_uses.update(dict(zone.bay_attr_uses))
        unsupported = {
            bay_key: quantity - int(anchor_capacity.get(bay_key, 0))
            for bay_key, quantity in positive_flow.items()
            if quantity > int(anchor_capacity.get(bay_key, 0))
        }
        if unsupported:
            raise RuntimeError(
                f"group pattern flow exceeds selected zone capacity: {unsupported}"
            )
        guidance_loads: Counter[tuple[str, str, str, str]] = Counter()
        used_areas = set()
        location_cost = 0.0
        for bay_key, quantity in positive_flow.items():
            column = self._flow_column_by_key[(group_id, bay_key)]
            guidance_loads[column.quota_key] += int(quantity)
            used_areas.add(str(column.area_no))
            location_cost += self._zone_flow_unit_cost(
                group_id, bay_key
            ) * int(quantity)
        unused_unit = self._unused_capacity_unit_cost()
        zone_cost = sum(
            float(self._zones[index].objective_cost)
            for index in ordered_zones
        )
        objective = (
            zone_cost
            + location_cost
            - unused_unit * int(group.demand)
            + self._zone_area_activation_penalty() * len(used_areas)
            - self._zone_activation_penalty()
            - self._zone_area_activation_penalty()
        )
        return GroupZonePattern(
            pattern_id=-1,
            group_id=group_id,
            zone_indices=ordered_zones,
            export_flow=tuple(sorted(positive_flow.items())),
            resources=tuple(sorted(resources)),
            bay_loads=tuple(sorted(bay_loads.items())),
            bay_size_loads=tuple(sorted(bay_size_loads.items())),
            stack_uses=tuple(sorted(stack_uses.items())),
            bay_attr_uses=tuple(sorted(bay_attr_uses.items())),
            guidance_loads=tuple(sorted(guidance_loads.items())),
            used_areas=tuple(sorted(used_areas)),
            candidate_indices=tuple(sorted(candidate_indices)),
            objective_cost=float(objective),
        )

    @staticmethod
    def _pattern_signature(pattern: GroupZonePattern) -> tuple:
        return (
            pattern.group_id,
            pattern.zone_indices,
            pattern.export_flow,
        )

    def _register_pattern(self, pattern: GroupZonePattern) -> int:
        signature = self._pattern_signature(pattern)
        existing = self._pattern_id_by_signature.get(signature)
        if existing is not None:
            return existing
        pattern_id = len(self._patterns)
        registered = replace(pattern, pattern_id=pattern_id)
        self._patterns.append(registered)
        self._pattern_id_by_signature[signature] = pattern_id
        self._pattern_ids_by_group[registered.group_id].append(pattern_id)
        return pattern_id

    def _build_group_pricing_model(self, group_id: str) -> _GroupPricingModel:
        from gurobipy import quicksum

        group = self.groups_by_id[group_id]
        zone_indices = tuple(self._zone_indices_by_group[group_id])
        model = GurobiModel(f"group_zone_price_{self._key_name((group_id,))}")
        self._configure_gurobi_output(model)
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(model, "Threads", int(self.config.solver_threads))
        self._set_gurobi_param(model, "MIPGap", 0.0)
        self._set_gurobi_param(model, "MIPFocus", 2)
        model.setMinimize()
        dispersion_baseline = model.addVar(
            lb=1.0,
            ub=1.0,
            name="group_pattern_dispersion_baseline",
        )
        zone = {
            index: model.addVar(vtype="B", name=f"z_{index}")
            for index in zone_indices
        }
        bays = sorted(
            {
                bay_key
                for index in zone_indices
                for bay_key, _capacity in self._zones[index].anchor_bay_loads
            }
        )
        flow = {
            bay_key: model.addVar(
                lb=0.0,
                ub=float(group.demand),
                # With binary zone/area decisions fixed, the bounded
                # single-commodity demand simplex has integral extreme
                # points.  Keeping flow continuous removes thousands of
                # unnecessary general integers without changing a pattern.
                vtype="C",
                name=f"f_{self._key_name((bay_key,))}",
            )
            for bay_key in bays
        }
        areas = sorted({self.bays[bay_key].area_no for bay_key in bays})
        area_use = {
            area_no: model.addVar(
                vtype="B", name=f"a_{self._key_name((area_no,))}"
            )
            for area_no in areas
        }
        model.addConstr(
            quicksum(flow.values()) == int(group.demand),
            name="complete_group_demand",
        )
        zones_by_anchor_bay: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        zones_by_resource: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        zones_by_bay: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        zones_by_bay_size: defaultdict[
            tuple[str, str], list[tuple[int, int]]
        ] = defaultdict(list)
        zones_by_stack: defaultdict[
            tuple[str, str], list[tuple[int, int]]
        ] = defaultdict(list)
        for index in zone_indices:
            candidate = self._zones[index]
            for key, value in candidate.anchor_bay_loads:
                zones_by_anchor_bay[key].append((index, int(value)))
            for key in candidate.resources:
                zones_by_resource[key].append(index)
            for key, value in candidate.bay_loads:
                zones_by_bay[key].append((index, int(value)))
            for key, value in candidate.bay_size_loads:
                zones_by_bay_size[key].append((index, int(value)))
            for key, value in candidate.stack_uses:
                zones_by_stack[key].append((index, int(value)))
        for bay_key in bays:
            model.addConstr(
                flow[bay_key]
                <= quicksum(
                    value * zone[index]
                    for index, value in zones_by_anchor_bay[bay_key]
                ),
                name=f"flow_cap_{self._key_name((bay_key,))}",
            )
        for resource, indices in sorted(zones_by_resource.items()):
            model.addConstr(
                quicksum(zone[index] for index in indices) <= 1.0,
                name=f"resource_{self._key_name(resource)}",
            )
        for bay_key, items in sorted(zones_by_bay.items()):
            model.addConstr(
                quicksum(value * zone[index] for index, value in items)
                <= int(self.bays[bay_key].physical_capacity),
                name=f"bay_{self._key_name((bay_key,))}",
            )
        for key, items in sorted(zones_by_bay_size.items()):
            model.addConstr(
                quicksum(value * zone[index] for index, value in items)
                <= int(self.bays[key[0]].cap_by_size.get(key[1], 0)),
                name=f"size_{self._key_name(key)}",
            )
        for key, items in sorted(zones_by_stack.items()):
            model.addConstr(
                quicksum(value * zone[index] for index, value in items)
                <= int(self._stack_count_for_bay_size(*key)),
                name=f"stack_{self._key_name(key)}",
            )
        # Every location in one operational group has the same hard attribute
        # values and candidate generation already filters conflicts with
        # existing stock.  Its local no-mix state is therefore an identity;
        # cross-group no-mix remains fully represented by master coefficients.
        flow_by_area: defaultdict[str, list] = defaultdict(list)
        for bay_key, variable in flow.items():
            flow_by_area[self.bays[bay_key].area_no].append(variable)
        for area_no, variable in area_use.items():
            assigned = quicksum(flow_by_area[area_no])
            model.addConstr(
                assigned <= int(group.demand) * variable,
                name=f"area_upper_{self._key_name((area_no,))}",
            )
            model.addConstr(
                variable <= assigned,
                name=f"area_lower_{self._key_name((area_no,))}",
            )
        model.update()
        pricing = _GroupPricingModel(
            group_id,
            model,
            zone,
            flow,
            area_use,
            dispersion_baseline,
        )
        self._pricing_models[group_id] = pricing
        return pricing

    def _pattern_coefficients(
        self, pattern: GroupZonePattern
    ) -> tuple[tuple[str, object, float], ...]:
        values: list[tuple[str, object, float]] = [
            ("convexity", pattern.group_id, 1.0)
        ]
        values.extend(
            ("physical_resource", key, 1.0) for key in pattern.resources
        )
        values.extend(
            ("bay_capacity", key, float(value))
            for key, value in pattern.bay_loads
        )
        values.extend(
            ("bay_size", key, float(value))
            for key, value in pattern.bay_size_loads
        )
        values.extend(
            ("stack_count", key, float(value))
            for key, value in pattern.stack_uses
        )
        values.extend(
            ("attr_link", key, float(value))
            for key, value in pattern.bay_attr_uses
        )
        values.extend(
            ("attr_presence", key, -float(value))
            for key, value in pattern.bay_attr_uses
        )
        values.extend(
            # The empty guidance row is stored as ``pos - neg = -target``.
            # A generated flow therefore enters that canonical row with a
            # negative coefficient: pos - neg - flow = -target.
            ("export_guidance", key, -float(value))
            for key, value in pattern.guidance_loads
        )
        return tuple(values)

    def _build_pattern_master(self):
        from gurobipy import quicksum

        model = GurobiModel("complete_group_zone_pattern_master")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Method", int(self.config.lp_method))
        self._set_gurobi_param(model, "OptimalityTol", 1e-9)
        self._set_gurobi_param(model, "FeasibilityTol", 1e-9)
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(model, "Threads", int(self.config.solver_threads))
        model.setMinimize()
        sets = self._master_index_sets()
        zero = model.addVar(lb=0.0, ub=0.0, name="pattern_zero")
        artificial = {
            group.group_id: model.addVar(
                lb=0.0,
                ub=1.0,
                obj=1.0,
                name=f"pattern_art_{self._key_name((group.group_id,))}",
            )
            for group in self.groups
        }
        attr_state = {
            key: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="C",
                name=f"pattern_attr_{self._key_name(key)}",
            )
            for key in sets["attr_keys"]
        }
        import_reserve = {
            (flow, size, bay_key): model.addVar(
                lb=0.0,
                ub=float(capacity),
                vtype="C",
                name=f"pattern_import_{flow}_{size}_{self._key_name((bay_key,))}",
            )
            for (flow, size), candidates in sorted(
                self.import_reservation_candidates.items()
            )
            for bay_key, capacity in candidates
        }
        constraints: dict[str, dict] = defaultdict(dict)
        for group in self.groups:
            constraints["convexity"][group.group_id] = model.addConstr(
                artificial[group.group_id] == 1.0,
                name=f"pattern_convexity_{self._key_name((group.group_id,))}",
            )
        for resource in sets["physical_resources"]:
            constraints["physical_resource"][resource] = model.addConstr(
                zero <= 1.0,
                name=f"pattern_resource_{self._key_name(resource)}",
            )
        import_by_bay: defaultdict[str, list] = defaultdict(list)
        import_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        import_by_flow_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        import_by_area: defaultdict[tuple[str, str, str], list] = defaultdict(list)
        for (flow, size, bay_key), variable in import_reserve.items():
            for footprint_key in self._placement_footprint_keys(bay_key, size):
                import_by_bay[footprint_key].append(variable)
            import_by_bay_size[(bay_key, size)].append(variable)
            import_by_flow_size[(flow, size)].append(variable)
            import_by_area[(flow, self.bays[bay_key].area_no, size)].append(variable)
        for bay_key in sorted(self._master_bay_capacity_keys):
            constraints["bay_capacity"][bay_key] = model.addConstr(
                quicksum(import_by_bay.get(bay_key, []))
                <= int(self.bays[bay_key].physical_capacity),
                name=f"pattern_bay_{self._key_name((bay_key,))}",
            )
        for key in sorted(self._master_bay_size_keys):
            constraints["bay_size"][key] = model.addConstr(
                quicksum(import_by_bay_size.get(key, []))
                <= int(self.bays[key[0]].cap_by_size.get(key[1], 0)),
                name=f"pattern_size_{self._key_name(key)}",
            )
        for key in sets["stack_keys"]:
            constraints["stack_count"][key] = model.addConstr(
                zero <= int(self._stack_count_for_bay_size(*key)),
                name=f"pattern_stack_{self._key_name(key)}",
            )
        for key in sets["attr_keys"]:
            big_m = max(1, len(self.bays[key[0]].row_physical_capacity))
            constraints["attr_link"][key] = model.addConstr(
                zero <= big_m * attr_state[key],
                name=f"pattern_attr_link_{self._key_name(key)}",
            )
            constraints["attr_presence"][key] = model.addConstr(
                attr_state[key] <= zero,
                name=f"pattern_attr_presence_{self._key_name(key)}",
            )
        for scope, keys in sorted(sets["attr_scopes"].items()):
            constraints["attr_choice"][scope] = model.addConstr(
                quicksum(attr_state[key] for key in keys) <= 1.0,
                name=f"pattern_attr_choice_{self._key_name(scope)}",
            )
        business_terms: list[tuple[object, float]] = []
        for key in sorted(self._master_area_guidance_keys):
            target = float(self._area_size_target(*key))
            pos = model.addVar(lb=0.0, name=f"pattern_guide_pos_{self._key_name(key)}")
            neg = model.addVar(lb=0.0, name=f"pattern_guide_neg_{self._key_name(key)}")
            business_terms.extend(
                ((pos, self._zone_guidance_penalty()), (neg, self._zone_guidance_penalty()))
            )
            constraints["export_guidance"][key] = model.addConstr(
                -target == pos - neg,
                name=f"pattern_guide_{self._key_name(key)}",
            )
        for key, required in sorted(self.import_total_by_flow_size.items()):
            constraints["import_total"][key] = model.addConstr(
                quicksum(import_by_flow_size.get(key, [])) == int(required),
                name=f"pattern_import_total_{self._key_name(key)}",
            )
        for key in sorted(set(import_by_area) | set(self.import_area_size_reference)):
            target = int(self.import_area_size_reference.get(key, 0))
            pos = model.addVar(lb=0.0, name=f"pattern_import_pos_{self._key_name(key)}")
            neg = model.addVar(lb=0.0, name=f"pattern_import_neg_{self._key_name(key)}")
            business_terms.extend(
                ((pos, self._zone_guidance_penalty()), (neg, self._zone_guidance_penalty()))
            )
            constraints["import_guidance"][key] = model.addConstr(
                quicksum(import_by_area.get(key, [])) - float(target) == pos - neg,
                name=f"pattern_import_guide_{self._key_name(key)}",
            )
        model.update()
        return model, {
            "artificial": artificial,
            "attr_state": attr_state,
            "import_reserve": import_reserve,
            "constraints": constraints,
            "pattern": {},
            "active_pattern_ids": set(),
            "business_terms": business_terms,
            "business_active": False,
        }

    def _seed_complete_group_patterns(self, model, variables: dict) -> dict:
        """Add one globally compatible deterministic pattern per group.

        This is a primal initialization of the new formulation, not a call to
        another solver chain.  It uses the common whole-zone packing routine
        only to construct disjoint physical support, then converts that support
        and its exact integer flow into native complete-group columns.
        """

        chosen, covered, import_protection = self._on_demand_greedy_support(
            {
                "last_root_zone_values": {},
                "last_root_import_values": {},
            }
        )
        complete = all(
            int(covered.get(group.group_id, 0)) >= int(group.demand)
            for group in self.groups
        )
        if not complete:
            return {
                "complete": False,
                "covered_group_count": sum(
                    int(covered.get(group.group_id, 0)) >= int(group.demand)
                    for group in self.groups
                ),
                "group_count": len(self.groups),
                "added_pattern_count": 0,
                "import_protection_fraction": import_protection,
            }
        selected_zones = {
            self._register_zone(strip_key, signature)
            for strip_key, signature in chosen
        }
        export_flow = self._export_flow_start_for_zones(selected_zones, {})
        if not export_flow:
            return {
                "complete": False,
                "covered_group_count": len(self.groups),
                "group_count": len(self.groups),
                "added_pattern_count": 0,
                "reason": "integer_flow_construction_failed",
                "import_protection_fraction": import_protection,
            }
        zones_by_group: defaultdict[str, set[int]] = defaultdict(set)
        flow_by_group: defaultdict[str, dict[str, int]] = defaultdict(dict)
        for zone_index in selected_zones:
            zones_by_group[self._zones[zone_index].group_id].add(zone_index)
        for (group_id, bay_key), quantity in export_flow.items():
            flow_by_group[group_id][bay_key] = int(quantity)
        seed_pattern_ids = set()
        for group in self.groups:
            pattern_id = self._register_pattern(
                self._make_group_pattern(
                    group.group_id,
                    zones_by_group[group.group_id],
                    flow_by_group[group.group_id],
                )
            )
            self._add_pattern_variable(model, variables, pattern_id)
            seed_pattern_ids.add(pattern_id)
        model.update()
        for pattern_id, variable in variables["pattern"].items():
            variable.Start = 1.0 if pattern_id in seed_pattern_ids else 0.0
        variables["seed_pattern_ids"] = seed_pattern_ids
        return {
            "complete": True,
            "covered_group_count": len(self.groups),
            "group_count": len(self.groups),
            "selected_zone_count": len(selected_zones),
            "added_pattern_count": len(seed_pattern_ids),
            "positive_export_flow_count": len(export_flow),
            "import_protection_fraction": import_protection,
        }

    def _add_pattern_variable(self, model, variables: dict, pattern_id: int):
        if pattern_id in variables["active_pattern_ids"]:
            return variables["pattern"][pattern_id]
        pattern = self._patterns[pattern_id]
        terms = []
        for section, key, coefficient in self._pattern_coefficients(pattern):
            row = variables["constraints"].get(section, {}).get(key)
            if row is None:
                raise RuntimeError(
                    f"pattern coefficient has no master row: {section}, {key}"
                )
            terms.append((coefficient, row))
        variable = model.addPricedVar(
            terms,
            lb=0.0,
            ub=1.0,
            vtype="C",
            obj=(pattern.objective_cost if variables["business_active"] else 0.0),
            name=f"group_pattern_{pattern_id}",
        )
        variables["pattern"][pattern_id] = variable
        variables["active_pattern_ids"].add(pattern_id)
        return variable

    def _activate_business_objective(self, model, variables: dict) -> None:
        if variables["business_active"]:
            return
        for variable in variables["artificial"].values():
            model.setVarObjective(variable, 0.0)
            variable.UB = 0.0
        for pattern_id, variable in variables["pattern"].items():
            model.setVarObjective(variable, self._patterns[pattern_id].objective_cost)
        for variable, coefficient in variables["business_terms"]:
            model.setVarObjective(variable, coefficient)
        variables["business_active"] = True
        model.update()

    def _master_duals(self, model, variables: dict) -> dict[tuple[str, object], float]:
        duals = {}
        for section in (
            "convexity",
            "physical_resource",
            "bay_capacity",
            "bay_size",
            "stack_count",
            "attr_link",
            "attr_presence",
            "export_guidance",
        ):
            for key, row in variables["constraints"].get(section, {}).items():
                duals[(section, key)] = float(model.getLinearDual(row))
        return duals

    def _pattern_reduced_cost(
        self,
        pattern: GroupZonePattern,
        duals: dict[tuple[str, object], float],
        business_active: bool,
    ) -> float:
        value = float(pattern.objective_cost) if business_active else 0.0
        for section, key, coefficient in self._pattern_coefficients(pattern):
            value -= float(coefficient) * float(duals.get((section, key), 0.0))
        return value

    def _set_group_pricing_objective(
        self,
        pricing: _GroupPricingModel,
        duals: dict[tuple[str, object], float],
        business_active: bool,
    ) -> None:
        pricing.model.setVarObjective(
            pricing.dispersion_baseline,
            -(
                self._zone_activation_penalty()
                + self._zone_area_activation_penalty()
            )
            if business_active
            else 0.0,
        )
        for zone_index, variable in pricing.zone.items():
            zone = self._zones[zone_index]
            value = float(zone.objective_cost) if business_active else 0.0
            for resource in zone.resources:
                value -= float(duals.get(("physical_resource", resource), 0.0))
            for key, coefficient in zone.bay_loads:
                value -= coefficient * float(duals.get(("bay_capacity", key), 0.0))
            for key, coefficient in zone.bay_size_loads:
                value -= coefficient * float(duals.get(("bay_size", key), 0.0))
            for key, coefficient in zone.stack_uses:
                value -= coefficient * float(duals.get(("stack_count", key), 0.0))
            for key, coefficient in zone.bay_attr_uses:
                value -= coefficient * float(duals.get(("attr_link", key), 0.0))
                value += coefficient * float(duals.get(("attr_presence", key), 0.0))
            pricing.model.setVarObjective(variable, value)
        unused_unit = self._unused_capacity_unit_cost()
        for bay_key, variable in pricing.flow.items():
            column = self._flow_column_by_key[(pricing.group_id, bay_key)]
            value = (
                self._zone_flow_unit_cost(pricing.group_id, bay_key)
                - unused_unit
                if business_active
                else 0.0
            )
            value += float(duals.get(("export_guidance", column.quota_key), 0.0))
            pricing.model.setVarObjective(variable, value)
        for variable in pricing.area_use.values():
            pricing.model.setVarObjective(
                variable,
                self._zone_area_activation_penalty()
                if business_active
                else 0.0,
            )
        pricing.model.update()

    def _price_group(
        self,
        group_id: str,
        duals: dict[tuple[str, object], float],
        business_active: bool,
        deadline: float,
        *,
        batch_size: int | None = None,
        improving_only: bool = True,
        threshold_only: bool = False,
    ) -> dict[str, object]:
        pricing = self._pricing_models.get(group_id)
        if pricing is None:
            pricing = self._build_group_pricing_model(group_id)
        self._set_group_pricing_objective(pricing, duals, business_active)
        remaining = deadline - perf_counter()
        if remaining <= 1e-6:
            return {"status": "time_limit_before_pricing", "patterns": []}
        batch = max(
            1,
            int(
                self.pattern_config.patterns_per_group_per_round
                if batch_size is None
                else batch_size
            ),
        )
        self._set_gurobi_param(pricing.model, "TimeLimit", max(0.01, remaining))
        convexity_dual = float(duals.get(("convexity", group_id), 0.0))
        tolerance = float(self.pattern_config.reduced_cost_tolerance)
        if improving_only and threshold_only:
            # Root pricing is a threshold decision: either exhibit one
            # negative-reduced-cost pattern or prove that none exists.  These
            # symmetric stops avoid solving a nearly degenerate pricing MIP
            # farther than the master actually needs.
            pricing_target = convexity_dual - tolerance
            self._set_gurobi_param(pricing.model, "BestObjStop", pricing_target)
            self._set_gurobi_param(pricing.model, "BestBdStop", pricing_target)
            self._set_gurobi_param(pricing.model, "MIPFocus", 3)
            batch = 1
        else:
            self._set_gurobi_param(pricing.model, "BestObjStop", -math.inf)
            self._set_gurobi_param(pricing.model, "BestBdStop", math.inf)
            self._set_gurobi_param(pricing.model, "MIPFocus", 2)
        self._set_gurobi_param(pricing.model, "PoolSolutions", max(1, batch))
        self._set_gurobi_param(
            pricing.model,
            "PoolSearchMode",
            (2 if not improving_only else 1) if batch > 1 else 0,
        )
        pricing.model.optimize()
        status = self._gurobi_status_name(pricing.model)
        pricing_bound = self._gurobi_dual_bound(pricing.model) - convexity_dual
        active_signatures = {
            self._pattern_signature(self._patterns[index])
            for index in self._pattern_ids_by_group[group_id]
        }
        patterns = []
        minimum_reduced_cost = math.inf
        available_solutions = self._gurobi_solution_count(pricing.model)
        solution_count = min(batch, available_solutions)
        for solution_number in range(solution_count):
            zones = {
                index
                for index, variable in pricing.zone.items()
                if pricing.model.getPoolValue(variable, solution_number) > 0.5
            }
            anchor_capacity: Counter[str] = Counter()
            for index in zones:
                anchor_capacity.update(dict(self._zones[index].anchor_bay_loads))
            raw_flow = {
                bay_key: max(
                    0.0,
                    min(
                        float(anchor_capacity.get(bay_key, 0)),
                        float(pricing.model.getPoolValue(variable, solution_number)),
                    ),
                )
                for bay_key, variable in pricing.flow.items()
            }
            flow = {
                bay_key: min(int(anchor_capacity[bay_key]), int(math.floor(value + 1e-8)))
                for bay_key, value in raw_flow.items()
                if value > 1e-8
            }
            remaining_flow = int(self.groups_by_id[group_id].demand) - sum(
                flow.values()
            )
            for bay_key in sorted(
                raw_flow,
                key=lambda key: (
                    -(raw_flow[key] - math.floor(raw_flow[key] + 1e-8)),
                    float(
                        self._zone_flow_unit_cost(group_id, key)
                    ),
                    key,
                ),
            ):
                if remaining_flow <= 0:
                    break
                available = int(anchor_capacity[bay_key]) - int(flow.get(bay_key, 0))
                if available <= 0:
                    continue
                quantity = min(remaining_flow, available)
                flow[bay_key] = int(flow.get(bay_key, 0)) + quantity
                remaining_flow -= quantity
            flow = {key: value for key, value in flow.items() if value > 0}
            pattern = self._make_group_pattern(group_id, zones, flow)
            reduced_cost = self._pattern_reduced_cost(
                pattern, duals, business_active
            )
            minimum_reduced_cost = min(minimum_reduced_cost, reduced_cost)
            accepted_cost = reduced_cost < -tolerance or not improving_only
            if accepted_cost and self._pattern_signature(pattern) not in active_signatures:
                patterns.append((reduced_cost, pattern))
                active_signatures.add(self._pattern_signature(pattern))
        certified = status == "optimal" or pricing_bound >= -tolerance
        unresolved = not certified and not patterns
        return {
            "status": status,
            "patterns": patterns,
            "minimum_reduced_cost": minimum_reduced_cost,
            "pricing_objective": (
                self._gurobi_objective_value(pricing.model) - convexity_dual
                if available_solutions > 0
                else None
            ),
            "pricing_lower_bound": pricing_bound,
            "certified": certified,
            "unresolved": unresolved,
        }

    def _run_pattern_root(self, model, variables: dict, deadline: float) -> dict:
        rounds = []
        closed = False
        phase = "phase_one"
        certified_lower_bound = None
        best_global_lower_bound = -math.inf
        last_rmp_objective = None
        for iteration in range(1, int(self.pattern_config.max_root_iterations) + 1):
            remaining = deadline - perf_counter()
            if remaining <= 1e-6:
                break
            self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
            model.optimize()
            status = self._gurobi_status_name(model)
            if status != "optimal":
                rounds.append({"iteration": iteration, "status": status, "phase": phase})
                break
            rmp_objective = self._gurobi_objective_value(model)
            last_rmp_objective = rmp_objective
            artificial_value = sum(
                self._gurobi_value(model, variable)
                for variable in variables["artificial"].values()
            )
            if phase == "phase_one" and artificial_value <= 1e-7:
                self._activate_business_objective(model, variables)
                phase = "business"
                continue
            duals = self._master_duals(model, variables)
            added = []
            minimum_rc = math.inf
            interrupted_group = None
            partial_pricing_groups = []
            unresolved_pricing_groups = []
            pricing_lower_bounds = {}
            group_ids = [group.group_id for group in self.groups]
            for position, group_id in enumerate(group_ids):
                group_remaining = deadline - perf_counter()
                if group_remaining <= 1e-6:
                    interrupted_group = group_id
                    break
                allowance = group_remaining / max(1, len(group_ids) - position)
                result = self._price_group(
                    group_id,
                    duals,
                    variables["business_active"],
                    perf_counter() + allowance,
                    threshold_only=(phase == "business" and iteration >= 6),
                )
                if "minimum_reduced_cost" not in result:
                    interrupted_group = group_id
                    break
                if result.get("unresolved", False):
                    unresolved_pricing_groups.append(group_id)
                if not result.get("certified", False):
                    partial_pricing_groups.append(group_id)
                minimum_rc = min(
                    minimum_rc, float(result["minimum_reduced_cost"])
                )
                pricing_lower_bounds[group_id] = float(
                    result["pricing_lower_bound"]
                )
                for reduced_cost, pattern in result["patterns"]:
                    pattern_id = self._register_pattern(pattern)
                    if pattern_id in variables["active_pattern_ids"]:
                        continue
                    self._add_pattern_variable(model, variables, pattern_id)
                    added.append((reduced_cost, pattern_id))
            if added:
                model.update()
            rounds.append(
                {
                    "iteration": iteration,
                    "status": status,
                    "phase": phase,
                    "objective": rmp_objective,
                    "artificial_value": artificial_value,
                    "minimum_reduced_cost": minimum_rc,
                    "added_pattern_count": len(added),
                    "active_pattern_count": len(variables["active_pattern_ids"]),
                    "interrupted_group": interrupted_group,
                    "partial_pricing_groups": partial_pricing_groups,
                    "unresolved_pricing_groups": unresolved_pricing_groups,
                    "pricing_lower_bound_correction": sum(
                        min(0.0, value)
                        for value in pricing_lower_bounds.values()
                    ),
                }
            )
            if interrupted_group is not None:
                break
            if len(pricing_lower_bounds) == len(group_ids):
                round_global_lower_bound = rmp_objective + sum(
                    min(0.0, value)
                    for value in pricing_lower_bounds.values()
                )
                best_global_lower_bound = max(
                    best_global_lower_bound, round_global_lower_bound
                )
                rounds[-1]["certified_global_lower_bound"] = (
                    round_global_lower_bound
                )
            if not added:
                if phase == "phase_one" and artificial_value > 1e-7:
                    raise RuntimeError(
                        "complete-group pattern Phase I closed with artificial use: "
                        f"{artificial_value}"
                    )
                closed = phase == "business" and not unresolved_pricing_groups
                if closed:
                    certified_lower_bound = best_global_lower_bound
                    break
                # The persistent pricing MIPs retain their search state.  A
                # further pass can improve their valid lower bounds even when
                # the restricted master itself has not changed.
                continue
        objective = last_rmp_objective
        return {
            "closed": closed,
            "phase": phase,
            "objective": objective,
            "certified_lower_bound": certified_lower_bound,
            "global_lower_bound": (
                best_global_lower_bound
                if math.isfinite(best_global_lower_bound)
                else None
            ),
            "rounds": rounds,
            "active_pattern_count": len(variables["active_pattern_ids"]),
            "pattern_count_by_group": {
                group.group_id: sum(
                    pattern_id in variables["active_pattern_ids"]
                    for pattern_id in self._pattern_ids_by_group[group.group_id]
                )
                for group in self.groups
            },
        }

    def _enrich_pattern_pool(self, model, variables: dict, deadline: float) -> dict:
        if not variables["business_active"] or self._gurobi_solution_count(model) <= 0:
            return {"added_pattern_count": 0, "reason": "no_business_root"}
        duals = self._master_duals(model, variables)
        added = 0
        per_group_cap = int(self.pattern_config.max_integer_patterns_per_group)
        group_ids = [group.group_id for group in self.groups]
        for position, group_id in enumerate(group_ids):
            active_for_group = sum(
                pattern_id in variables["active_pattern_ids"]
                for pattern_id in self._pattern_ids_by_group[group_id]
            )
            if active_for_group >= per_group_cap:
                continue
            remaining = deadline - perf_counter()
            if remaining <= 1e-6:
                break
            allowance = remaining / max(1, len(group_ids) - position)
            result = self._price_group(
                group_id,
                duals,
                True,
                perf_counter() + allowance,
                batch_size=per_group_cap - active_for_group,
                improving_only=False,
            )
            if result.get("unresolved", False):
                break
            for _reduced_cost, pattern in result["patterns"]:
                pattern_id = self._register_pattern(pattern)
                if pattern_id not in variables["active_pattern_ids"]:
                    self._add_pattern_variable(model, variables, pattern_id)
                    added += 1
        if added:
            model.update()
        return {
            "added_pattern_count": added,
            "active_pattern_count": len(variables["active_pattern_ids"]),
            "per_group_cap": per_group_cap,
            "pricing_mode": "exact_pool_best_distinct_patterns",
        }

    def _solve_integer_pattern_master(
        self, model, variables: dict, deadline: float
    ) -> tuple[set[int], dict, dict]:
        for variable in variables["pattern"].values():
            variable.VType = "B"
        for variable in variables["attr_state"].values():
            variable.VType = "B"
        for variable in variables["import_reserve"].values():
            variable.VType = "I"
        model.update()
        remaining = deadline - perf_counter()
        if remaining <= 1e-6:
            return set(), {}, {"status": "time_limit_before_integer_master"}
        self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
        self._set_gurobi_param(model, "MIPGap", 0.0)
        self._set_gurobi_param(model, "MIPFocus", 1)
        self._set_gurobi_param(model, "Heuristics", 0.25)
        model.optimize()
        status = self._gurobi_status_name(model)
        if self._gurobi_solution_count(model) <= 0:
            return set(), {}, {"status": status, "has_solution": False}
        selected = {
            pattern_id
            for pattern_id, variable in variables["pattern"].items()
            if self._gurobi_value(model, variable) > 0.5
        }
        imports = {
            key: int(round(self._gurobi_value(model, variable)))
            for key, variable in variables["import_reserve"].items()
            if self._gurobi_value(model, variable) > 1e-7
        }
        objective = self._gurobi_objective_value(model)
        bound = self._gurobi_dual_bound(model)
        return selected, imports, {
            "status": status,
            "has_solution": True,
            "objective": objective,
            "bound": bound,
            "absolute_gap": max(0.0, objective - bound),
            "relative_gap": max(0.0, objective - bound)
            / max(abs(objective), 1e-12),
            "selected_pattern_count": len(selected),
        }

    def _selected_pattern_decisions(
        self, selected_pattern_ids: set[int]
    ) -> tuple[set[int], dict[tuple[str, str], int]]:
        zones = set()
        flow = {}
        selected_by_group = Counter(
            self._patterns[index].group_id for index in selected_pattern_ids
        )
        mismatches = {
            group.group_id: selected_by_group[group.group_id]
            for group in self.groups
            if selected_by_group[group.group_id] != 1
        }
        if mismatches:
            raise RuntimeError(
                f"integer pattern master did not choose one pattern per group: {mismatches}"
            )
        for pattern_id in selected_pattern_ids:
            pattern = self._patterns[pattern_id]
            zones.update(pattern.zone_indices)
            for bay_key, quantity in pattern.export_flow:
                flow[(pattern.group_id, bay_key)] = int(quantity)
        return zones, flow

    def _pattern_objective_certificate(
        self,
        selected_pattern_ids: set[int],
        import_reserve: dict[tuple[str, str, str], int],
        solver_objective: float,
    ) -> dict[str, object]:
        """Reconstruct the pattern-master incumbent without solver auxiliaries."""

        pattern_cost = sum(
            float(self._patterns[index].objective_cost)
            for index in selected_pattern_ids
        )
        export_by_quota: Counter[tuple[str, str, str, str]] = Counter()
        for index in selected_pattern_ids:
            export_by_quota.update(dict(self._patterns[index].guidance_loads))
        export_deviation = sum(
            abs(
                float(export_by_quota.get(key, 0))
                - float(self._area_size_target(*key))
            )
            for key in self._master_area_guidance_keys
        )
        import_by_area: Counter[tuple[str, str, str]] = Counter()
        for (flow, size, bay_key), quantity in import_reserve.items():
            if int(quantity) > 0:
                import_by_area[(flow, self.bays[bay_key].area_no, size)] += int(
                    quantity
                )
        import_deviation = sum(
            abs(
                int(import_by_area.get(key, 0))
                - int(self.import_area_size_reference.get(key, 0))
            )
            for key in set(import_by_area) | set(self.import_area_size_reference)
        )
        guidance_cost = self._zone_guidance_penalty() * (
            export_deviation + import_deviation
        )
        reconstructed = float(pattern_cost + guidance_cost)
        auxiliary_slack = self._absolute_deviation_auxiliary_slack(
            solver_objective,
            reconstructed,
            context="complete-group pattern master",
        )
        return {
            "certified": True,
            "objective": reconstructed,
            "solver_incumbent_objective": float(solver_objective),
            "solver_auxiliary_slack": float(auxiliary_slack),
            "absolute_reconstruction_difference": abs(
                reconstructed - float(solver_objective)
            ),
            "components": {
                "selected_pattern_cost": float(pattern_cost),
                "large_plan_guidance": float(guidance_cost),
            },
            "selected_pattern_count": len(selected_pattern_ids),
            "export_guidance_l1_deviation": int(export_deviation),
            "import_guidance_l1_deviation": int(import_deviation),
        }

    def _fix_optimize_neighborhood(
        self,
        selected_pattern_ids: set[int],
        current_import: dict[tuple[str, str, str], int],
        deadline: float,
    ) -> dict[str, object]:
        """Reoptimize the most expensive interacting groups on complete zones."""

        remaining = deadline - perf_counter()
        if remaining <= 1e-6 or not selected_pattern_ids:
            return {"status": "skipped_no_time", "new_pattern_ids": set()}
        selected_by_group = {
            self._patterns[index].group_id: self._patterns[index]
            for index in selected_pattern_ids
        }
        area_sets = {
            group.group_id: set(self._candidate_areas_for_group(group))
            for group in self.groups
        }
        base_score = {
            group_id: float(pattern.objective_cost)
            / max(1, int(self.groups_by_id[group_id].demand))
            for group_id, pattern in selected_by_group.items()
        }
        seed_group = max(base_score, key=lambda key: (base_score[key], key))
        seed = self.groups_by_id[seed_group]
        ranked = sorted(
            selected_by_group,
            key=lambda group_id: (
                -int(
                    bool(area_sets[group_id] & area_sets[seed_group])
                    and self.groups_by_id[group_id].size == seed.size
                ),
                -int(self.groups_by_id[group_id].voyage_id == seed.voyage_id),
                -base_score[group_id],
                group_id,
            ),
        )
        count = min(
            len(ranked),
            int(self.pattern_config.fix_optimize_group_count),
            max(3, int(math.ceil(math.sqrt(len(ranked))))),
        )
        neighborhood = set(ranked[:count])
        current_zones, current_flow = self._selected_pattern_decisions(
            selected_pattern_ids
        )
        model, variables = self._build_zone_master()
        try:
            active_zone_indices = set()
            for group in self.groups:
                if group.group_id in neighborhood:
                    active_zone_indices.update(
                        self._zone_indices_by_group[group.group_id]
                    )
                else:
                    active_zone_indices.update(
                        selected_by_group[group.group_id].zone_indices
                    )
            for zone_index in sorted(active_zone_indices):
                self._add_zone_variable(model, variables, zone_index)
            for zone_index, variable in variables["zone"].items():
                group_id = self._zones[zone_index].group_id
                if group_id not in neighborhood:
                    variable.LB = 1.0
                    variable.UB = 1.0
                variable.Start = 1.0 if zone_index in current_zones else 0.0
                variable.VType = "B"
            for group_id, variable in variables["shortage"].items():
                variable.UB = 0.0
                variable.Start = 0.0
            for key, variable in variables["export_flow"].items():
                if key[0] not in neighborhood:
                    fixed = int(current_flow.get(key, 0))
                    variable.LB = float(fixed)
                    variable.UB = float(fixed)
                variable.Start = float(current_flow.get(key, 0))
                variable.VType = "I"
            used_areas = {
                (group_id, self.bays[bay_key].area_no)
                for (group_id, bay_key), quantity in current_flow.items()
                if int(quantity) > 0
            }
            for key, variable in variables["area_use"].items():
                variable.Start = 1.0 if key in used_areas else 0.0
                variable.VType = "B"
            used_attr = {
                key
                for zone_index in current_zones
                for key, value in self._zones[zone_index].bay_attr_uses
                if value > 0
            }
            for key, variable in variables["attr_state"].items():
                variable.Start = 1.0 if key in used_attr else 0.0
                variable.VType = "B"
            for key, variable in variables["import_reserve"].items():
                variable.Start = float(current_import.get(key, 0))
                variable.VType = "I"
            model.update()
            self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
            self._set_gurobi_param(model, "MIPGap", 0.0)
            self._set_gurobi_param(model, "MIPFocus", 1)
            self._set_gurobi_param(model, "Heuristics", 0.30)
            model.optimize()
            status = self._gurobi_status_name(model)
            if self._gurobi_solution_count(model) <= 0:
                return {
                    "status": status,
                    "has_solution": False,
                    "neighborhood": sorted(neighborhood),
                    "new_pattern_ids": set(),
                }
            selected_zones = {
                index
                for index, variable in variables["zone"].items()
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
            zones_by_group: defaultdict[str, set[int]] = defaultdict(set)
            flow_by_group: defaultdict[str, dict[str, int]] = defaultdict(dict)
            for index in selected_zones:
                zones_by_group[self._zones[index].group_id].add(index)
            for (group_id, bay_key), quantity in export_flow.items():
                flow_by_group[group_id][bay_key] = int(quantity)
            new_pattern_ids = set()
            for group_id in neighborhood:
                new_pattern_ids.add(
                    self._register_pattern(
                        self._make_group_pattern(
                            group_id,
                            zones_by_group[group_id],
                            flow_by_group[group_id],
                        )
                    )
                )
            objective = self._gurobi_objective_value(model)
            bound = self._gurobi_dual_bound(model)
            return {
                "status": status,
                "has_solution": True,
                "objective": objective,
                "bound": bound,
                "relative_gap": max(0.0, objective - bound)
                / max(abs(objective), 1e-12),
                "neighborhood": sorted(neighborhood),
                "neighborhood_group_count": len(neighborhood),
                "active_zone_count": len(active_zone_indices),
                "new_pattern_ids": new_pattern_ids,
                "import_reserve": import_reserve,
            }
        finally:
            self._free_gurobi_model(model)

    def _apply_fix_optimize_patterns(
        self,
        model,
        variables: dict,
        selected_pattern_ids: set[int],
        current_import: dict[tuple[str, str, str], int],
        deadline: float,
    ) -> tuple[set[int], dict[tuple[str, str, str], int], dict[str, object]]:
        """Generate one exact local solution and reopen the integer RMP."""

        local_deadline = perf_counter() + max(0.0, deadline - perf_counter()) * 0.70
        local = self._fix_optimize_neighborhood(
            selected_pattern_ids, current_import, local_deadline
        )
        new_pattern_ids = set(local.pop("new_pattern_ids", set()))
        local_import = local.pop("import_reserve", {})
        if local_import:
            local["positive_import_reservation_count"] = len(local_import)
            local["reserved_import_box_count"] = int(sum(local_import.values()))
        if not new_pattern_ids:
            return selected_pattern_ids, current_import, {
                "local": local,
                "added_pattern_count": 0,
                "master_reoptimization": {"status": "not_run"},
            }
        added = 0
        for pattern_id in sorted(new_pattern_ids):
            if pattern_id not in variables["active_pattern_ids"]:
                variable = self._add_pattern_variable(model, variables, pattern_id)
                variable.VType = "B"
                added += 1
        for pattern_id, variable in variables["pattern"].items():
            variable.Start = 1.0 if pattern_id in selected_pattern_ids else 0.0
        for key, variable in variables["import_reserve"].items():
            variable.Start = float(current_import.get(key, 0))
        model.update()
        remaining = deadline - perf_counter()
        if remaining <= 1e-6:
            return selected_pattern_ids, current_import, {
                "local": local,
                "added_pattern_count": added,
                "master_reoptimization": {"status": "time_limit_before_reopt"},
            }
        self._set_gurobi_param(model, "TimeLimit", max(0.01, remaining))
        model.optimize()
        status = self._gurobi_status_name(model)
        if self._gurobi_solution_count(model) <= 0:
            return selected_pattern_ids, current_import, {
                "local": local,
                "added_pattern_count": added,
                "master_reoptimization": {"status": status, "has_solution": False},
            }
        selected = {
            pattern_id
            for pattern_id, variable in variables["pattern"].items()
            if self._gurobi_value(model, variable) > 0.5
        }
        imports = {
            key: int(round(self._gurobi_value(model, variable)))
            for key, variable in variables["import_reserve"].items()
            if self._gurobi_value(model, variable) > 1e-7
        }
        objective = self._gurobi_objective_value(model)
        return selected, imports, {
            "local": local,
            "added_pattern_count": added,
            "master_reoptimization": {
                "status": status,
                "has_solution": True,
                "objective": objective,
                "bound": self._gurobi_dual_bound(model),
                "selected_pattern_count": len(selected),
            },
        }

    def analyze_root(self) -> dict[str, object]:
        """Run only exact root generation and retain incomplete diagnostics."""

        started = perf_counter()
        deadline = started + max(0.01, float(self.config.total_time_limit))
        preparation = self._prepare_group_patterns()
        model, variables = self._build_pattern_master()
        try:
            seed = self._seed_complete_group_patterns(model, variables)
            root = self._run_pattern_root(model, variables, deadline)
        finally:
            self._free_gurobi_model(model)
            for pricing in self._pricing_models.values():
                self._free_gurobi_model(pricing.model)
            self._pricing_models.clear()
        return {
            "algorithm": "complete_group_zone_pattern_exact_root",
            "preparation": preparation,
            "seed": seed,
            "root": root,
            "zone_count": len(self._zones),
            "pattern_count": len(self._patterns),
            "total_seconds": perf_counter() - started,
        }

    def solve(self) -> ColumnGenerationResult:
        started = perf_counter()
        deadline = started + max(0.01, float(self.config.total_time_limit))
        preparation = self._prepare_group_patterns()
        model, variables = self._build_pattern_master()
        try:
            seed = self._seed_complete_group_patterns(model, variables)
            root_deadline = perf_counter() + max(0.0, deadline - perf_counter()) * float(
                self.pattern_config.root_time_fraction
            )
            root = self._run_pattern_root(model, variables, root_deadline)
            if root["phase"] != "business" or root["global_lower_bound"] is None:
                raise RuntimeError(
                    "complete-group pattern root did not obtain a certified "
                    "pricing-corrected lower bound within its time envelope"
                )
            root_lower_bound = float(root["global_lower_bound"])
            enrichment_deadline = perf_counter() + max(
                0.0, deadline - perf_counter()
            ) * 0.10
            enrichment = self._enrich_pattern_pool(
                model, variables, enrichment_deadline
            )
            fill_reserve = float(self.config.total_time_limit) * float(
                self.pattern_config.fill_time_fraction
            )
            fix_optimize_reserve = float(self.config.total_time_limit) * float(
                self.pattern_config.fix_optimize_time_fraction
            )
            selected_patterns, import_reserve, integer_master = (
                self._solve_integer_pattern_master(
                    model,
                    variables,
                    deadline - fill_reserve - fix_optimize_reserve,
                )
            )
            if not selected_patterns:
                raise RuntimeError(
                    "integer complete-group pattern master has no feasible solution: "
                    f"{integer_master}"
                )
            integer_master_initial = dict(integer_master)
            selected_patterns, import_reserve, fix_optimize = (
                self._apply_fix_optimize_patterns(
                    model,
                    variables,
                    selected_patterns,
                    import_reserve,
                    deadline - fill_reserve,
                )
            )
            reoptimized = fix_optimize.get("master_reoptimization", {})
            if reoptimized.get("has_solution", False):
                objective = float(reoptimized["objective"])
                bound = float(reoptimized["bound"])
                integer_master = {
                    "status": reoptimized["status"],
                    "has_solution": True,
                    "objective": objective,
                    "bound": bound,
                    "absolute_gap": max(0.0, objective - bound),
                    "relative_gap": max(0.0, objective - bound)
                    / max(abs(objective), 1e-12),
                    "selected_pattern_count": len(selected_patterns),
                }
        finally:
            self._free_gurobi_model(model)
            for pricing in self._pricing_models.values():
                self._free_gurobi_model(pricing.model)
            self._pricing_models.clear()
        selected_zones, export_flow = self._selected_pattern_decisions(
            selected_patterns
        )
        objective_certificate = self._pattern_objective_certificate(
            selected_patterns,
            import_reserve,
            float(integer_master["objective"]),
        )
        zone_objective_certificate = self._zone_objective_certificate(
            selected_zones,
            export_flow,
            import_reserve,
            float(objective_certificate["objective"]),
        )
        selected, fill = self._solve_restricted_fill(
            selected_zones,
            export_flow,
            import_reserve,
            deadline,
        )
        upper_bound = float(integer_master["objective"])
        absolute_gap = max(0.0, upper_bound - root_lower_bound)
        diagnostics = {
            "algorithm": "complete_group_zone_pattern_generation_and_exact_recourse",
            "model_scope": "complete_group_patterns_on_dedicated_contiguous_zones",
            "formulation": "dantzig_wolfe_group_pattern_master",
            "decomposition": "exact_integer_group_pricing_then_restricted_integer_master",
            "planned_group_count": len(self.groups),
            "planned_box_count": sum(group.demand for group in self.groups),
            "candidate_row_location_count": len(self._columns),
            "group_pattern_preparation": preparation,
            "group_pattern_seed": seed,
            "group_pattern_root": root,
            "group_pattern_enrichment": enrichment,
            "group_pattern_integer_master": integer_master,
            "group_pattern_integer_master_initial": integer_master_initial,
            "group_pattern_fix_optimize": fix_optimize,
            "group_pattern_objective_certificate": objective_certificate,
            "zone_objective_certificate": zone_objective_certificate,
            "group_pattern_count": len(self._patterns),
            "selected_group_pattern_count": len(selected_patterns),
            "selected_zone_count": len(selected_zones),
            "selected_candidate_count": len(
                {
                    candidate
                    for zone_index in selected_zones
                    for candidate in self._zones[zone_index].candidate_indices
                }
            ),
            "group_pattern_model_upper_bound": upper_bound,
            "group_pattern_model_global_lower_bound": root_lower_bound,
            "group_pattern_model_absolute_gap": absolute_gap,
            "group_pattern_model_relative_gap": absolute_gap
            / max(abs(upper_bound), 1e-12),
            "group_pattern_model_lower_bound_source": (
                "closed_exact_group_pricing_root"
                if root["closed"]
                else "pricing_mip_bound_corrected_restricted_master_dual"
            ),
            "group_pattern_integer_pool_bound": integer_master.get("bound"),
            "group_pattern_integer_pool_bound_is_global": False,
            "zone_fill": fill,
            "master_status": integer_master["status"],
            "master_objective": upper_bound,
            "master_bound_scope": "complete_group_pattern_model",
            "row_recourse_status": fill["status"],
            "row_recourse_secondary_quality_objective": fill["objective"],
            "complete_model_lower_bound": root_lower_bound,
            "complete_model_absolute_gap": absolute_gap,
            "complete_model_relative_gap": absolute_gap
            / max(abs(upper_bound), 1e-12),
            "complete_model_gap_source": (
                "closed_exact_group_pattern_root"
                if root["closed"]
                else "certified_incomplete_group_pricing_bounds"
            ),
            "restricted_fill_lower_bound": fill["bound"],
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
                "constraint_scope": ["area_function", "bay_size", "physical_capacity"],
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
        result.diagnostics["row_recourse_secondary_quality_objective"] = (
            result.diagnostics["final_business_objective"]
        )
        result.diagnostics["row_recourse_secondary_quality_components"] = (
            result.diagnostics["final_business_objective_components"]
        )
        result.diagnostics["final_business_objective"] = upper_bound
        result.diagnostics["final_business_objective_components"] = {
            "raw": zone_objective_certificate["raw"],
            "normalized": zone_objective_certificate["normalized"],
            "weighted": zone_objective_certificate["weighted"],
            "weighted_total": upper_bound,
        }
        result.columns = self._selected_direct_columns(selected)
        return result


__all__ = [
    "GroupZonePatternConfig",
    "GroupZonePattern",
    "GroupZonePatternGenerationPlanner",
]
