"""Logic-based Benders decomposition with capacity and conflict cuts."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from time import perf_counter

from .direct_milp import DirectMilpPlanner
from .gurobi_backend import GurobiModel
from .planner import ColumnGenerationResult, PlacementColumn


OperationalKey = tuple[str, ...]


@dataclass(frozen=True)
class LogicBendersConfig:
    """Small, instance-independent controls for the LBBD loop."""

    max_iterations: int = 40
    master_time_limit: float = 10.0
    area_time_limit: float = 5.0
    primal_seed_time_limit: float = 10.0
    support_repair_iterations: int = 5
    support_repair_fraction: float = 0.02
    max_cliques_per_area: int = 100


@dataclass
class _AreaSubproblem:
    area_no: str
    model: GurobiModel
    placement_variables: dict[int, object]
    import_variables: dict[tuple[str, str, str], object]
    group_balance: dict[str, object]
    import_balance: dict[tuple[str, str], object]
    build_seconds: float


@dataclass
class _SupportRepairNeighborhood:
    limit: object
    variables: tuple[object, ...]
    constraints: tuple[object, ...]
    target_radius: int


class LogicBendersPlanner(DirectMilpPlanner):
    """Exact LBBD over area quantities and independent row-packing models."""

    def __init__(
        self,
        problem,
        config=None,
        benders_config: LogicBendersConfig | None = None,
    ) -> None:
        super().__init__(problem, config)
        self.benders_config = benders_config or LogicBendersConfig()
        if int(self.benders_config.max_iterations) <= 0:
            raise ValueError("LBBD max_iterations must be positive")
        if float(self.benders_config.master_time_limit) <= 0.0:
            raise ValueError("LBBD master_time_limit must be positive")
        if float(self.benders_config.area_time_limit) <= 0.0:
            raise ValueError("LBBD area_time_limit must be positive")
        if float(self.benders_config.primal_seed_time_limit) < 0.0:
            raise ValueError("LBBD primal_seed_time_limit cannot be negative")
        if int(self.benders_config.support_repair_iterations) < 0:
            raise ValueError("LBBD support_repair_iterations cannot be negative")
        if not 0.0 <= float(
            self.benders_config.support_repair_fraction
        ) <= 1.0:
            raise ValueError(
                "LBBD support_repair_fraction must be between zero and one"
            )
        if int(self.benders_config.max_cliques_per_area) <= 0:
            raise ValueError("LBBD max_cliques_per_area must be positive")

        self._areas: tuple[str, ...] = ()
        self._candidate_indices_by_group_area: dict[
            tuple[str, str], tuple[int, ...]
        ] = {}
        self._candidate_indices_by_area: dict[str, tuple[int, ...]] = {}
        self._operational_groups: dict[OperationalKey, tuple[str, ...]] = {}
        self._representative_group: dict[OperationalKey, object] = {}
        self._operational_area_groups: dict[
            tuple[OperationalKey, str], tuple[str, ...]
        ] = {}
        self._row_upper: dict[tuple[OperationalKey, str], int] = {}
        self._row_capacity_profile: dict[
            tuple[OperationalKey, str], tuple[int, ...]
        ] = {}
        self._proximity_capacity_profile: dict[
            tuple[str, str], tuple[tuple[float, int], ...]
        ] = {}
        self._slot_count: dict[tuple[OperationalKey, str], int] = {}
        self._footprint_factor: dict[OperationalKey, int] = {}
        self._physical_slots: dict[
            tuple[OperationalKey, str], frozenset[tuple[str, str]]
        ] = {}
        self._conflict_cliques: dict[
            str, tuple[tuple[OperationalKey, ...], ...]
        ] = {}
        self._import_area_capacity: dict[tuple[str, str, str], int] = {}
        self._area_subproblems: dict[str, _AreaSubproblem] = {}
        self._primal_seed_selected: Counter[int] = Counter()
        self._primal_seed_import: Counter[tuple[str, str, str]] = Counter()
        self._primal_seed_group_area: Counter[tuple[str, str]] = Counter()
        self._primal_seed_import_area: Counter[
            tuple[str, str, str]
        ] = Counter()

    @staticmethod
    def _footprint_units(size: str) -> int:
        return 2 if str(size) in {"40", "45"} else 1

    def _prepare_lbbd(self) -> None:
        self._prepare_master_index_sets()
        self._prepare_objective_normalization()
        self._initialize_location_pool()
        candidate_indices_by_group_area: defaultdict[
            tuple[str, str], list[int]
        ] = defaultdict(list)
        candidate_indices_by_area: defaultdict[str, list[int]] = defaultdict(list)
        for group in self.groups:
            for candidate in self._base_placements_for_group(group):
                index = self._append_generated_column(candidate)
                candidate_indices_by_group_area[
                    (group.group_id, candidate.area_no)
                ].append(index)
                candidate_indices_by_area[candidate.area_no].append(index)
        self._candidate_indices_by_group_area = {
            key: tuple(indices)
            for key, indices in sorted(candidate_indices_by_group_area.items())
        }
        self._candidate_indices_by_area = {
            area_no: tuple(indices)
            for area_no, indices in sorted(candidate_indices_by_area.items())
        }

        operational_groups: defaultdict[OperationalKey, list[str]] = defaultdict(list)
        representatives = {}
        for group in self.groups:
            key = self._operational_group_key(group)
            operational_groups[key].append(group.group_id)
            representatives.setdefault(key, group)
        self._operational_groups = {
            key: tuple(sorted(group_ids))
            for key, group_ids in sorted(operational_groups.items())
        }
        self._representative_group = representatives

        operational_area_groups: defaultdict[
            tuple[OperationalKey, str], set[str]
        ] = defaultdict(set)
        physical_slots: defaultdict[
            tuple[OperationalKey, str], set[tuple[str, str]]
        ] = defaultdict(set)
        anchor_slots: defaultdict[
            tuple[OperationalKey, str], set[tuple[str, str]]
        ] = defaultdict(set)
        row_upper: defaultdict[tuple[OperationalKey, str], int] = defaultdict(int)
        anchor_capacity: defaultdict[
            tuple[OperationalKey, str], dict[tuple[str, str], int]
        ] = defaultdict(dict)
        proximity_capacity: defaultdict[
            tuple[str, str], Counter[float]
        ] = defaultdict(Counter)
        for (group_id, area_no), indices in self._candidate_indices_by_group_area.items():
            group = self.groups_by_id[group_id]
            operational_key = self._operational_group_key(group)
            operational_area_groups[(operational_key, area_no)].add(group_id)
            for index in indices:
                candidate = self._columns[index]
                capacity = self._base_location_capacity(group, candidate)
                row_upper[(operational_key, area_no)] = max(
                    row_upper[(operational_key, area_no)],
                    capacity,
                )
                anchor_row = next(
                    row_no
                    for bay_key, row_no, _quantity in candidate.row_allocation
                    if bay_key == candidate.bay_key
                )
                anchor_slots[(operational_key, area_no)].add(
                    (candidate.bay_key, anchor_row)
                )
                slot = (candidate.bay_key, anchor_row)
                anchor_capacity[(operational_key, area_no)][slot] = max(
                    capacity,
                    anchor_capacity[(operational_key, area_no)].get(slot, 0),
                )
                proximity_capacity[(group_id, area_no)][
                    round(
                        self._column_base_cost(group, candidate.bay_key),
                        15,
                    )
                ] += capacity
                physical_slots[(operational_key, area_no)].update(
                    (bay_key, row_no)
                    for bay_key, row_no, _quantity in candidate.row_allocation
                )
        self._operational_area_groups = {
            key: tuple(sorted(group_ids))
            for key, group_ids in sorted(operational_area_groups.items())
        }
        self._row_upper = dict(row_upper)
        self._row_capacity_profile = {
            key: tuple(sorted(values.values(), reverse=True))
            for key, values in sorted(anchor_capacity.items())
        }
        self._proximity_capacity_profile = {
            key: tuple(sorted((cost, int(capacity)) for cost, capacity in values.items()))
            for key, values in sorted(proximity_capacity.items())
        }
        self._slot_count = {
            key: len(values) for key, values in anchor_slots.items()
        }
        self._physical_slots = {
            key: frozenset(values) for key, values in physical_slots.items()
        }
        self._footprint_factor = {
            key: self._footprint_units(representatives[key].size)
            for key in self._operational_groups
        }

        import_area_capacity: Counter[tuple[str, str, str]] = Counter()
        for (flow, size), candidates in self.import_reservation_candidates.items():
            for bay_key, capacity in candidates:
                import_area_capacity[
                    (flow, size, self.bays[bay_key].area_no)
                ] += int(capacity)
        self._import_area_capacity = {
            key: min(
                int(value),
                int(self.import_total_by_flow_size[(key[0], key[1])]),
            )
            for key, value in import_area_capacity.items()
            if int(value) > 0
        }
        areas = set(self._candidate_indices_by_area)
        areas.update(area_no for _flow, _size, area_no in self._import_area_capacity)
        self._areas = tuple(sorted(areas))
        self._conflict_cliques = {
            area_no: self._area_conflict_cliques(area_no)
            for area_no in self._areas
        }

    def _area_conflict_cliques(
        self, area_no: str
    ) -> tuple[tuple[OperationalKey, ...], ...]:
        nodes = sorted(
            key
            for key, candidate_area in self._operational_area_groups
            if candidate_area == area_no
        )
        neighbors: dict[OperationalKey, set[OperationalKey]] = {
            node: set() for node in nodes
        }
        for position, first in enumerate(nodes):
            first_group = self._representative_group[first]
            first_slots = self._physical_slots[(first, area_no)]
            for second in nodes[position + 1 :]:
                if not first_slots.intersection(
                    self._physical_slots[(second, area_no)]
                ):
                    continue
                if not self._groups_are_incompatible_on_one_row(
                    first_group, self._representative_group[second]
                ):
                    continue
                neighbors[first].add(second)
                neighbors[second].add(first)

        maximal: list[tuple[OperationalKey, ...]] = []

        def expand(
            clique: set[OperationalKey],
            candidates: set[OperationalKey],
            excluded: set[OperationalKey],
        ) -> None:
            if not candidates and not excluded:
                if len(clique) >= 2:
                    maximal.append(tuple(sorted(clique)))
                return
            pivot_pool = candidates | excluded
            pivot = max(
                pivot_pool,
                key=lambda node: len(candidates & neighbors[node]),
                default=None,
            )
            extension = candidates - (neighbors[pivot] if pivot is not None else set())
            for node in list(sorted(extension)):
                expand(
                    clique | {node},
                    candidates & neighbors[node],
                    excluded & neighbors[node],
                )
                candidates.remove(node)
                excluded.add(node)

        expand(set(), set(nodes), set())
        ranked = sorted(
            set(maximal),
            key=lambda clique: (-len(clique), repr(clique)),
        )
        return tuple(ranked[: int(self.benders_config.max_cliques_per_area)])

    def _area_physical_capacity(self, area_no: str) -> int:
        return sum(
            int(self.bays[bay_key].physical_capacity)
            for bay_key in self.bays_by_area.get(area_no, ())
        )

    def _area_size_capacity(self, area_no: str, size: str) -> int:
        return sum(
            int(self.bays[bay_key].cap_by_size.get(size, 0))
            for bay_key in self.bays_by_area.get(area_no, ())
        )

    def _area_edge_capacity(self, area_no: str) -> int:
        return sum(
            int(self.bays[bay_key].cap_by_size.get("45", 0))
            for bay_key in self.area_edge_bays.get(area_no, ())
        )

    def _area_slot_capacity(
        self, area_no: str, slots: set[tuple[str, str]]
    ) -> int:
        return sum(
            int(
                self.bays[bay_key].row_physical_capacity.get(
                    row_no, self.bays[bay_key].physical_capacity
                )
            )
            for bay_key, row_no in slots
            if self.bays[bay_key].area_no == area_no
        )

    def _build_master(self):
        from gurobipy import quicksum

        model = GurobiModel("yard_logic_benders_master")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        self._set_gurobi_param(model, "MIPGap", 0.0)
        self._set_gurobi_param(model, "DualReductions", 0)
        model.setMinimize()

        quantity = {
            (group.group_id, area_no): model.addVar(
                lb=0.0,
                ub=float(group.demand),
                vtype="I",
                obj=self._berth_distance_cost(
                    group.voyage_id, area_no, 1
                ),
                name=(
                    f"q_{group.group_id}_{self._key_name((area_no,))}"
                ),
            )
            for group in self.groups
            for area_no in self._areas
            if (group.group_id, area_no)
            in self._candidate_indices_by_group_area
        }
        area_use = {
            (operational_key, area_no): model.addVar(
                vtype="B",
                obj=self._area_activation_penalty(),
                name=(
                    f"use_area_{self._key_name(operational_key)}_"
                    f"{self._key_name((area_no,))}"
                ),
            )
            for operational_key, area_no in self._operational_area_groups
        }
        row_count = {
            (operational_key, area_no): model.addVar(
                lb=0.0,
                ub=float(self._slot_count[(operational_key, area_no)]),
                vtype="I",
                name=(
                    f"rows_{self._key_name(operational_key)}_"
                    f"{self._key_name((area_no,))}"
                ),
            )
            for operational_key, area_no in self._operational_area_groups
        }
        import_quantity = {
            key: model.addVar(
                lb=0.0,
                ub=float(capacity),
                vtype="I",
                name=f"import_area_{self._key_name(key)}",
            )
            for key, capacity in sorted(self._import_area_capacity.items())
        }
        theta = {
            area_no: model.addVar(
                lb=0.0,
                obj=1.0,
                name=f"theta_{self._key_name((area_no,))}",
            )
            for area_no in self._areas
        }
        proximity_lower_bound = {
            key: model.addVar(
                lb=0.0,
                name=(
                    f"proximity_lb_{self._key_name((key[0], key[1]))}"
                ),
            )
            for key in sorted(quantity)
        }
        constraints: dict[str, dict] = defaultdict(dict)

        for group in self.groups:
            terms = [
                variable
                for (group_id, _area_no), variable in quantity.items()
                if group_id == group.group_id
            ]
            if not terms:
                raise ValueError(
                    "LBBD group has no feasible area: "
                    f"group={group.group_id}"
                )
            constraints["group_demand"][group.group_id] = model.addConstr(
                quicksum(terms) == int(group.demand),
                name=f"demand_{group.group_id}",
            )

        for key, group_ids in sorted(self._operational_area_groups.items()):
            operational_key, area_no = key
            terms = [
                quantity[(group_id, area_no)]
                for group_id in group_ids
                if (group_id, area_no) in quantity
            ]
            demand = sum(self.group_demand[group_id] for group_id in group_ids)
            assigned = quicksum(terms)
            constraints["area_use_upper"][key] = model.addConstr(
                assigned <= int(demand) * area_use[key],
                name=(
                    f"area_use_upper_{self._key_name(operational_key)}_"
                    f"{self._key_name((area_no,))}"
                ),
            )
            constraints["area_use_lower"][key] = model.addConstr(
                area_use[key] <= assigned,
                name=(
                    f"area_use_lower_{self._key_name(operational_key)}_"
                    f"{self._key_name((area_no,))}"
                ),
            )
            capacities = self._row_capacity_profile[key]
            prefix_capacity = 0
            previous_slope = None
            for breakpoint, slope in enumerate(capacities):
                if previous_slope is None or slope != previous_slope:
                    constraints["row_cover"][(
                        operational_key,
                        area_no,
                        breakpoint,
                    )] = model.addConstr(
                        assigned
                        <= prefix_capacity
                        + int(slope) * (row_count[key] - breakpoint),
                        name=(
                            f"row_cover_{self._key_name(operational_key)}_"
                            f"{self._key_name((area_no, str(breakpoint)))}"
                        ),
                    )
                prefix_capacity += int(slope)
                previous_slope = int(slope)
            constraints["row_use_lower"][key] = model.addConstr(
                area_use[key] <= row_count[key],
                name=(
                    f"row_use_lower_{self._key_name(operational_key)}_"
                    f"{self._key_name((area_no,))}"
                ),
            )
            constraints["row_use_upper"][key] = model.addConstr(
                row_count[key]
                <= self._slot_count[key] * area_use[key],
                name=(
                    f"row_use_upper_{self._key_name(operational_key)}_"
                    f"{self._key_name((area_no,))}"
                ),
            )

        for key, required in sorted(self.import_total_by_flow_size.items()):
            terms = [
                variable
                for (flow, size, _area_no), variable
                in import_quantity.items()
                if (flow, size) == key
            ]
            if not terms:
                raise ValueError(
                    "LBBD import demand has no feasible area: "
                    f"flow={key[0]}, size={key[1]}"
                )
            constraints["import_total"][key] = model.addConstr(
                quicksum(terms) == int(required),
                name=f"import_total_{self._key_name(key)}",
            )

        for area_no in self._areas:
            export_physical = [
                self._footprint_units(self.groups_by_id[group_id].size)
                * variable
                for (group_id, candidate_area), variable in quantity.items()
                if candidate_area == area_no
            ]
            import_physical = [
                self._footprint_units(size) * variable
                for (flow, size, candidate_area), variable
                in import_quantity.items()
                if candidate_area == area_no
            ]
            constraints["area_physical_capacity"][area_no] = model.addConstr(
                quicksum(export_physical) + quicksum(import_physical)
                <= self._area_physical_capacity(area_no),
                name=f"area_physical_{self._key_name((area_no,))}",
            )
            for size in ("20", "40", "45"):
                export_size = [
                    variable
                    for (group_id, candidate_area), variable in quantity.items()
                    if candidate_area == area_no
                    and self.groups_by_id[group_id].size == size
                ]
                import_size = [
                    variable
                    for (_flow, import_size_value, candidate_area), variable
                    in import_quantity.items()
                    if candidate_area == area_no
                    and import_size_value == size
                ]
                if not export_size and not import_size:
                    continue
                constraints["area_size_capacity"][(area_no, size)] = (
                    model.addConstr(
                        quicksum(export_size) + quicksum(import_size)
                        <= self._area_size_capacity(area_no, size),
                        name=(
                            f"area_size_{self._key_name((area_no, size))}"
                        ),
                    )
                )
            forty_five = [
                variable
                for (group_id, candidate_area), variable in quantity.items()
                if candidate_area == area_no
                and self.groups_by_id[group_id].size == "45"
            ]
            if forty_five:
                constraints["edge_45_capacity"][area_no] = model.addConstr(
                    quicksum(forty_five)
                    <= self._area_edge_capacity(area_no),
                    name=f"edge_45_{self._key_name((area_no,))}",
                )

        conflict_cut_count = 0
        hall_cut_count = 0
        for area_no, cliques in self._conflict_cliques.items():
            for clique_index, clique in enumerate(cliques):
                slots: set[tuple[str, str]] = set()
                row_terms = []
                quantity_terms = []
                for operational_key in clique:
                    key = (operational_key, area_no)
                    slots.update(self._physical_slots[key])
                    row_terms.append(
                        self._footprint_factor[operational_key]
                        * row_count[key]
                    )
                    for group_id in self._operational_area_groups[key]:
                        variable = quantity.get((group_id, area_no))
                        if variable is not None:
                            quantity_terms.append(
                                self._footprint_factor[operational_key]
                                * variable
                            )
                if row_terms:
                    constraints["conflict_clique"][(area_no, clique_index)] = (
                        model.addConstr(
                            quicksum(row_terms) <= len(slots),
                            name=(
                                f"conflict_clique_"
                                f"{self._key_name((area_no, str(clique_index)))}"
                            ),
                        )
                    )
                    conflict_cut_count += 1
                if quantity_terms and slots:
                    capacity = self._area_slot_capacity(area_no, slots)
                    constraints["conflict_hall_capacity"][(
                        area_no,
                        clique_index,
                    )] = model.addConstr(
                        quicksum(quantity_terms) <= capacity,
                        name=(
                            f"conflict_hall_"
                            f"{self._key_name((area_no, str(clique_index)))}"
                        ),
                    )
                    hall_cut_count += 1

        export_by_guidance: defaultdict[tuple[str, str, str, str], list] = (
            defaultdict(list)
        )
        for (group_id, area_no), variable in quantity.items():
            group = self.groups_by_id[group_id]
            export_by_guidance[
                self._quota_key(group, area_no)
            ].append(variable)
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
            constraints["export_guidance"][key] = model.addConstr(
                quicksum(export_by_guidance.get(key, [])) - target
                == positive - negative,
                name=f"guide_{self._key_name(key)}",
            )

        import_reference_keys = set(self.import_area_size_reference) | {
            (flow, area_no, size)
            for flow, size, area_no in import_quantity
        }
        for flow, area_no, size in sorted(import_reference_keys):
            target = int(
                self.import_area_size_reference.get(
                    (flow, area_no, size), 0
                )
            )
            positive = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"import_guide_pos_{flow}_{area_no}_{size}",
            )
            negative = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"import_guide_neg_{flow}_{area_no}_{size}",
            )
            actual = import_quantity.get((flow, size, area_no))
            constraints["import_guidance"][(flow, area_no, size)] = (
                model.addConstr(
                    (actual if actual is not None else 0.0) - target
                    == positive - negative,
                    name=f"import_guide_{flow}_{area_no}_{size}",
                )
            )

        for area_no in self._areas:
            lower_bound_terms = []
            for (group_id, candidate_area), variable in quantity.items():
                if candidate_area != area_no:
                    continue
                key = (group_id, area_no)
                proximity = proximity_lower_bound[key]
                prefix_quantity = 0
                prefix_cost = 0.0
                for segment_index, (unit_cost, capacity) in enumerate(
                    self._proximity_capacity_profile[key]
                ):
                    constraints["proximity_envelope"][(
                        group_id,
                        area_no,
                        segment_index,
                    )] = model.addConstr(
                        proximity
                        >= prefix_cost
                        + float(unit_cost) * (variable - prefix_quantity),
                        name=(
                            f"proximity_envelope_"
                            f"{self._key_name((group_id, area_no, str(segment_index)))}"
                        ),
                    )
                    prefix_quantity += int(capacity)
                    prefix_cost += float(unit_cost) * int(capacity)
                lower_bound_terms.append(proximity)
            lower_bound_terms.extend(
                self._row_activation_penalty() * variable
                for (operational_key, candidate_area), variable
                in row_count.items()
                if candidate_area == area_no
            )
            constraints["theta_analytic"][area_no] = model.addConstr(
                theta[area_no] >= quicksum(lower_bound_terms),
                name=f"theta_analytic_{self._key_name((area_no,))}",
            )

        offset = -len(self._operational_groups) * (
            self._area_activation_penalty()
            + self._row_activation_penalty()
        )
        model.addVar(
            lb=1.0,
            ub=1.0,
            obj=offset,
            name="lbbd_objective_offset",
        )
        model.update()
        return model, {
            "quantity": quantity,
            "area_use": area_use,
            "row_count": row_count,
            "import_quantity": import_quantity,
            "theta": theta,
            "proximity_lower_bound": proximity_lower_bound,
        }, {
            "conflict_clique_cut_count": conflict_cut_count,
            "conflict_hall_capacity_cut_count": hall_cut_count,
            "master_variable_count": len(model.getVars()),
            "row_capacity_envelope_cut_count": len(
                constraints["row_cover"]
            ),
            "proximity_envelope_cut_count": len(
                constraints["proximity_envelope"]
            ),
        }

    def _add_area_compatibility(
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
                vtype="B", name=f"area_bay_use_{self._key_name(key)}"
            )
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                <= self._master_bay_attr_big_m[key] * use,
                name=f"area_bay_attr_link_{self._key_name(key)}",
            )
            bay_key, attr, scope, _value = key
            bay_uses[(bay_key, attr, scope)].append(use)
        for key, uses in sorted(bay_uses.items()):
            model.addConstr(
                quicksum(uses) <= 1,
                name=f"area_bay_attr_one_{self._key_name(key)}",
            )

        row_uses: defaultdict[tuple[str, str, str, str], list] = (
            defaultdict(list)
        )
        for key, items in sorted(
            coefficient_rows["row_attr_link"].items()
        ):
            use = model.addVar(
                vtype="B", name=f"area_row_use_{self._key_name(key)}"
            )
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                <= self._master_row_attr_big_m[key] * use,
                name=f"area_row_attr_link_{self._key_name(key)}",
            )
            bay_key, row_no, attr, scope, _value = key
            row_uses[(bay_key, row_no, attr, scope)].append(use)
        for key, uses in sorted(row_uses.items()):
            model.addConstr(
                quicksum(uses) <= 1,
                name=f"area_row_attr_one_{self._key_name(key)}",
            )

    def _build_area_subproblem(self, area_no: str) -> _AreaSubproblem:
        from gurobipy import quicksum

        started = perf_counter()
        model = GurobiModel(
            f"yard_area_subproblem_{self._key_name((area_no,))}"
        )
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        self._set_gurobi_param(model, "MIPGap", 0.0)
        self._set_gurobi_param(model, "DualReductions", 0)
        model.setMinimize()
        candidate_indices = self._candidate_indices_by_area.get(area_no, ())
        capacities = {
            index: self._base_location_capacity(
                self.groups_by_id[self._columns[index].group_id],
                self._columns[index],
            )
            for index in candidate_indices
        }
        placement_variables = {
            index: model.addVar(
                lb=0.0,
                ub=float(capacities[index]),
                vtype="I",
                obj=self._column_base_cost(
                    self.groups_by_id[self._columns[index].group_id],
                    self._columns[index].bay_key,
                ),
                name=f"x_{index}",
            )
            for index in candidate_indices
        }
        import_variables = {
            (flow, size, bay_key): model.addVar(
                lb=0.0,
                ub=float(capacity),
                vtype="I",
                name=(
                    f"import_{flow}_{size}_{self._key_name((bay_key,))}"
                ),
            )
            for (flow, size), candidates in sorted(
                self.import_reservation_candidates.items()
            )
            for bay_key, capacity in candidates
            if self.bays[bay_key].area_no == area_no
        }

        coefficient_rows: defaultdict[
            str, defaultdict[object, list[tuple[int, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        group_indices: defaultdict[str, list[int]] = defaultdict(list)
        group_row_indices: defaultdict[
            tuple[OperationalKey, str, str], list[int]
        ] = defaultdict(list)
        operational_demand: Counter[OperationalKey] = Counter()
        for group in self.groups:
            operational_demand[self._operational_group_key(group)] += int(
                group.demand
            )
        for index in candidate_indices:
            candidate = self._columns[index]
            group_indices[candidate.group_id].append(index)
            anchor_row = next(
                row_no
                for bay_key, row_no, _quantity in candidate.row_allocation
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

        import_by_bay: defaultdict[str, list] = defaultdict(list)
        import_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        import_by_flow_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        for (flow, size, bay_key), variable in import_variables.items():
            for footprint_key in self._placement_footprint_keys(bay_key, size):
                import_by_bay[footprint_key].append(variable)
            import_by_bay_size[(bay_key, size)].append(variable)
            import_by_flow_size[(flow, size)].append(variable)

        group_balance = {}
        for group_id, indices in sorted(group_indices.items()):
            group_balance[group_id] = model.addConstr(
                quicksum(placement_variables[index] for index in indices)
                == 0,
                name=f"area_group_{group_id}",
            )

        area_bay_keys = set(coefficient_rows["bay_capacity_limit"])
        area_bay_keys.update(import_by_bay)
        for bay_key in sorted(area_bay_keys):
            items = coefficient_rows["bay_capacity_limit"].get(bay_key, [])
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                + quicksum(import_by_bay.get(bay_key, []))
                <= int(self.bays[bay_key].physical_capacity),
                name=f"area_bay_cap_{self._key_name((bay_key,))}",
            )
        area_bay_size_keys = set(coefficient_rows["bay_size_limit"])
        area_bay_size_keys.update(import_by_bay_size)
        for key in sorted(area_bay_size_keys):
            bay_key, size = key
            items = coefficient_rows["bay_size_limit"].get(key, [])
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                + quicksum(import_by_bay_size.get(key, []))
                <= int(self.bays[bay_key].cap_by_size.get(size, 0)),
                name=f"area_bay_size_{self._key_name(key)}",
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
                name=f"area_row_cap_{self._key_name(key)}",
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
                name=f"area_row_size_{self._key_name(key)}",
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
                name=f"area_stack_{self._key_name(key)}",
            )
            model.addConstr(
                quicksum(
                    coefficient * placement_variables[index]
                    for index, coefficient in items
                )
                <= unit_capacity * stack,
                name=f"area_stack_link_{self._key_name(key)}",
            )
            stacks_by_bay_size[(bay_key, size)].append(stack)
        for key, stacks in sorted(stacks_by_bay_size.items()):
            model.addConstr(
                quicksum(stacks) <= self._stack_count_for_bay_size(*key),
                name=f"area_stack_total_{self._key_name(key)}",
            )

        for key, indices in sorted(group_row_indices.items()):
            operational_key, bay_key, row_no = key
            upper = min(
                int(operational_demand[operational_key]),
                sum(capacities[index] for index in indices),
            )
            use = model.addVar(
                vtype="B",
                obj=self._row_activation_penalty(),
                name=(
                    f"area_group_row_{self._key_name(operational_key)}_"
                    f"{self._key_name((bay_key, row_no))}"
                ),
            )
            assigned = quicksum(
                placement_variables[index] for index in indices
            )
            model.addConstr(
                assigned <= max(1, upper) * use,
                name=(
                    f"area_group_row_upper_"
                    f"{self._key_name((*operational_key, bay_key, row_no))}"
                ),
            )
            model.addConstr(
                use <= assigned,
                name=(
                    f"area_group_row_lower_"
                    f"{self._key_name((*operational_key, bay_key, row_no))}"
                ),
            )

        import_balance = {}
        for key, variables in sorted(import_by_flow_size.items()):
            import_balance[key] = model.addConstr(
                quicksum(variables) == 0,
                name=f"area_import_{self._key_name(key)}",
            )
        self._add_area_compatibility(
            model,
            quicksum,
            placement_variables,
            coefficient_rows,
        )
        model.update()
        return _AreaSubproblem(
            area_no=area_no,
            model=model,
            placement_variables=placement_variables,
            import_variables=import_variables,
            group_balance=group_balance,
            import_balance=import_balance,
            build_seconds=perf_counter() - started,
        )

    @staticmethod
    def _set_constraint_rhs(constraint, value: float) -> None:
        constraint.RHS = float(value)

    def _solve_area_subproblem(
        self,
        area_no: str,
        group_quantities: dict[str, int],
        import_quantities: dict[tuple[str, str], int],
        time_limit: float,
    ) -> dict:
        subproblem = self._area_subproblems.get(area_no)
        if subproblem is None:
            subproblem = self._build_area_subproblem(area_no)
            self._area_subproblems[area_no] = subproblem
        for group_id, constraint in subproblem.group_balance.items():
            self._set_constraint_rhs(
                constraint, int(group_quantities.get(group_id, 0))
            )
        for key, constraint in subproblem.import_balance.items():
            self._set_constraint_rhs(
                constraint, int(import_quantities.get(key, 0))
            )
        seed_groups = {
            group_id: int(quantity)
            for (group_id, candidate_area), quantity
            in self._primal_seed_group_area.items()
            if candidate_area == area_no and int(quantity) > 0
        }
        seed_imports = {
            (flow, size): int(quantity)
            for (flow, size, candidate_area), quantity
            in self._primal_seed_import_area.items()
            if candidate_area == area_no and int(quantity) > 0
        }
        if group_quantities == seed_groups and import_quantities == seed_imports:
            for index, variable in subproblem.placement_variables.items():
                variable.Start = float(
                    self._primal_seed_selected.get(index, 0)
                )
            for key, variable in subproblem.import_variables.items():
                variable.Start = float(self._primal_seed_import.get(key, 0))
        subproblem.model.update()
        self._set_gurobi_param(
            subproblem.model, "TimeLimit", max(0.01, time_limit)
        )
        started = perf_counter()
        subproblem.model.optimize()
        elapsed = perf_counter() - started
        status = self._gurobi_status_name(subproblem.model)
        solution_count = self._gurobi_solution_count(subproblem.model)
        result = {
            "area_no": area_no,
            "status": status,
            "optimal": status == "optimal",
            "feasible": solution_count > 0,
            "seconds": elapsed,
            "selected": None,
            "import_reservation": None,
            "objective": math.inf,
            "bound": -math.inf,
            "conflict_group_ids": (),
            "conflict_import_keys": (),
        }
        if solution_count <= 0:
            if status == "infeasible":
                try:
                    subproblem.model._model.computeIIS()
                    result["conflict_group_ids"] = tuple(
                        sorted(
                            group_id
                            for group_id, constraint
                            in subproblem.group_balance.items()
                            if group_quantities.get(group_id, 0) > 0
                            and bool(constraint.IISConstr)
                        )
                    )
                    result["conflict_import_keys"] = tuple(
                        sorted(
                            key
                            for key, constraint
                            in subproblem.import_balance.items()
                            if import_quantities.get(key, 0) > 0
                            and bool(constraint.IISConstr)
                        )
                    )
                except Exception:
                    pass
            return result
        selected = Counter(
            {
                index: int(
                    round(
                        self._gurobi_value(subproblem.model, variable)
                    )
                )
                for index, variable in subproblem.placement_variables.items()
                if self._gurobi_value(subproblem.model, variable) > 0.5
            }
        )
        import_reservation = Counter(
            {
                key: int(
                    round(
                        self._gurobi_value(subproblem.model, variable)
                    )
                )
                for key, variable in subproblem.import_variables.items()
                if self._gurobi_value(subproblem.model, variable) > 0.5
            }
        )
        result.update(
            {
                "selected": selected,
                "import_reservation": import_reservation,
                "objective": self._gurobi_objective_value(
                    subproblem.model
                ),
                "bound": self._gurobi_dual_bound(subproblem.model),
                "gap": self._gurobi_gap(subproblem.model),
            }
        )
        return result

    def _master_assignment(
        self, model, variables: dict
    ) -> tuple[dict, dict, dict]:
        quantities = {
            key: int(round(self._gurobi_value(model, variable)))
            for key, variable in variables["quantity"].items()
            if self._gurobi_value(model, variable) > 0.5
        }
        imports = {
            key: int(round(self._gurobi_value(model, variable)))
            for key, variable in variables["import_quantity"].items()
            if self._gurobi_value(model, variable) > 0.5
        }
        theta = {
            area_no: float(self._gurobi_value(model, variable))
            for area_no, variable in variables["theta"].items()
        }
        return quantities, imports, theta

    def _area_master_components(
        self,
        area_no: str,
        variables: dict,
        quantities: dict[tuple[str, str], int],
        imports: dict[tuple[str, str, str], int],
    ) -> list[tuple[object, int, int, str]]:
        components: list[tuple[object, int, int, str]] = []
        for key, value in sorted(quantities.items()):
            group_id, candidate_area = key
            if candidate_area != area_no or int(value) <= 0:
                continue
            components.append(
                (
                    variables["quantity"][key],
                    int(value),
                    int(self.group_demand[group_id]),
                    f"q_{group_id}",
                )
            )
        for key, value in sorted(imports.items()):
            flow, size, candidate_area = key
            if candidate_area != area_no or int(value) <= 0:
                continue
            components.append(
                (
                    variables["import_quantity"][key],
                    int(value),
                    int(self._import_area_capacity[key]),
                    f"r_{flow}_{size}",
                )
            )
        return components

    def _add_dominance_cut(
        self,
        model,
        variables: dict,
        area_no: str,
        quantities: dict[tuple[str, str], int],
        imports: dict[tuple[str, str, str], int],
        cut_index: int,
        kind: str,
        lower_bound: float = 0.0,
    ) -> int:
        from gurobipy import quicksum

        components = self._area_master_components(
            area_no, variables, quantities, imports
        )
        if not components:
            raise RuntimeError(
                f"cannot add {kind} cut for an empty area {area_no}"
            )
        decreases = []
        for component_index, (variable, incumbent, upper, label) in enumerate(
            components
        ):
            decrease = model.addVar(
                vtype="B",
                name=(
                    f"{kind}_decrease_{cut_index}_{component_index}_"
                    f"{self._key_name((area_no, label))}"
                ),
            )
            model.addConstr(
                variable >= incumbent - upper * decrease,
                name=(
                    f"{kind}_decrease_lb_{cut_index}_{component_index}_"
                    f"{self._key_name((area_no,))}"
                ),
            )
            model.addConstr(
                variable
                <= incumbent - 1 + upper * (1 - decrease),
                name=(
                    f"{kind}_decrease_ub_{cut_index}_{component_index}_"
                    f"{self._key_name((area_no,))}"
                ),
            )
            decreases.append(decrease)
        if kind == "feasibility":
            model.addConstr(
                quicksum(decreases) >= 1,
                name=(
                    f"logic_feasibility_{cut_index}_"
                    f"{self._key_name((area_no,))}"
                ),
            )
        elif kind == "optimality":
            value = max(0.0, float(lower_bound))
            model.addConstr(
                variables["theta"][area_no]
                >= value * (1 - quicksum(decreases)),
                name=(
                    f"logic_optimality_{cut_index}_"
                    f"{self._key_name((area_no,))}"
                ),
            )
        else:
            raise ValueError(f"unknown LBBD cut kind: {kind}")
        model.update()
        return len(decreases)

    def _add_violated_active_hall_cut(
        self,
        model,
        variables: dict,
        area_no: str,
        quantities: dict[tuple[str, str], int],
        signatures: set[tuple],
        cut_index: int,
    ) -> bool:
        from gurobipy import quicksum

        active_keys = sorted(
            {
                self._operational_group_key(self.groups_by_id[group_id])
                for (group_id, candidate_area), value in quantities.items()
                if candidate_area == area_no and int(value) > 0
            }
        )
        if len(active_keys) < 2:
            return False
        signature = (area_no, tuple(active_keys))
        if signature in signatures:
            return False
        slots: set[tuple[str, str]] = set()
        terms = []
        incumbent_load = 0
        for operational_key in active_keys:
            key = (operational_key, area_no)
            slots.update(self._physical_slots.get(key, ()))
            factor = self._footprint_factor[operational_key]
            for group_id in self._operational_area_groups.get(key, ()):
                variable = variables["quantity"].get((group_id, area_no))
                if variable is None:
                    continue
                terms.append(factor * variable)
                incumbent_load += factor * int(
                    quantities.get((group_id, area_no), 0)
                )
        capacity = self._area_slot_capacity(area_no, slots)
        if not terms or incumbent_load <= capacity:
            return False
        model.addConstr(
            quicksum(terms) <= capacity,
            name=(
                f"dynamic_conflict_hall_{cut_index}_"
                f"{self._key_name((area_no,))}"
            ),
        )
        signatures.add(signature)
        model.update()
        return True

    def _add_conflict_core_capacity_cut(
        self,
        master,
        variables: dict,
        area_no: str,
        group_ids: set[str],
        import_keys: set[tuple[str, str]],
        quantities: dict[tuple[str, str], int],
        imports: dict[tuple[str, str, str], int],
        signatures: set[tuple],
        cut_index: int,
        time_limit: float,
    ) -> bool:
        """Compute the exact aggregate capacity of an IIS demand core."""
        from gurobipy import quicksum

        if not group_ids and not import_keys:
            return False
        signature = (
            area_no,
            tuple(sorted(group_ids)),
            tuple(sorted(import_keys)),
        )
        if signature in signatures:
            return False
        signatures.add(signature)
        subproblem = self._area_subproblems[area_no]
        all_variables = tuple(subproblem.model.getVars())
        original_objectives = tuple(
            subproblem.model.getVarObjective(variable)
            for variable in all_variables
        )
        original_group_rows = {
            key: (constraint.Sense, float(constraint.RHS))
            for key, constraint in subproblem.group_balance.items()
        }
        original_import_rows = {
            key: (constraint.Sense, float(constraint.RHS))
            for key, constraint in subproblem.import_balance.items()
        }
        try:
            for variable in all_variables:
                subproblem.model.setVarObjective(variable, 0.0)
            for index, variable in subproblem.placement_variables.items():
                if self._columns[index].group_id in group_ids:
                    subproblem.model.setVarObjective(variable, -1.0)
            for (flow, size, _bay_key), variable in (
                subproblem.import_variables.items()
            ):
                if (flow, size) in import_keys:
                    subproblem.model.setVarObjective(variable, -1.0)
            for group_id, constraint in subproblem.group_balance.items():
                if group_id in group_ids:
                    constraint.Sense = "<"
                    constraint.RHS = float(self.group_demand[group_id])
                else:
                    constraint.Sense = "="
                    constraint.RHS = 0.0
            for key, constraint in subproblem.import_balance.items():
                if key in import_keys:
                    constraint.Sense = "<"
                    constraint.RHS = float(
                        self.import_total_by_flow_size[key]
                    )
                else:
                    constraint.Sense = "="
                    constraint.RHS = 0.0
            subproblem.model.update()
            self._set_gurobi_param(
                subproblem.model,
                "TimeLimit",
                max(0.01, float(time_limit)),
            )
            subproblem.model.optimize()
            if self._gurobi_status_name(subproblem.model) != "optimal":
                return False
            capacity = int(
                round(-self._gurobi_objective_value(subproblem.model))
            )
            incumbent_load = sum(
                int(quantities.get((group_id, area_no), 0))
                for group_id in group_ids
            ) + sum(
                int(imports.get((flow, size, area_no), 0))
                for flow, size in import_keys
            )
            if incumbent_load <= capacity:
                return False
            terms = [
                variables["quantity"][(group_id, area_no)]
                for group_id in sorted(group_ids)
                if (group_id, area_no) in variables["quantity"]
            ]
            terms.extend(
                variables["import_quantity"][(flow, size, area_no)]
                for flow, size in sorted(import_keys)
                if (flow, size, area_no)
                in variables["import_quantity"]
            )
            master.addConstr(
                quicksum(terms) <= capacity,
                name=(
                    f"conflict_core_capacity_{cut_index}_"
                    f"{self._key_name((area_no,))}"
                ),
            )
            master.update()
            return True
        finally:
            for variable, objective in zip(
                all_variables, original_objectives
            ):
                subproblem.model.setVarObjective(variable, objective)
            for key, (sense, rhs) in original_group_rows.items():
                constraint = subproblem.group_balance[key]
                constraint.Sense = sense
                constraint.RHS = rhs
            for key, (sense, rhs) in original_import_rows.items():
                constraint = subproblem.import_balance[key]
                constraint.Sense = sense
                constraint.RHS = rhs
            subproblem.model.update()

    @staticmethod
    def _area_assignment_key(
        area_no: str,
        group_quantities: dict[str, int],
        import_quantities: dict[tuple[str, str], int],
    ) -> tuple:
        return (
            area_no,
            tuple(sorted(group_quantities.items())),
            tuple(sorted(import_quantities.items())),
        )

    def _selected_lbbd_columns(
        self, selected: Counter[int]
    ) -> list[PlacementColumn]:
        return [
            replace(column, column_id=f"LBBD_{index:07d}")
            for index, column in enumerate(
                self._selected_direct_columns(selected), start=1
            )
        ]

    def _base_diagnostics(self) -> dict:
        return {
            "algorithm": "logic_based_benders_capacity_conflict_gurobi",
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
            "formulation": "area_quantity_master_with_exact_row_subproblems",
            "decomposition": "logic_based_benders_by_yard_area",
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
        }

    def _solve_compact_primal_seed(self, deadline: float) -> dict:
        """Find only a primal seed; never use this model's bound."""
        from gurobipy import quicksum

        started = perf_counter()
        result = {
            "attempted": True,
            "feasible": False,
            "status": "not_solved",
            "seconds": 0.0,
            "build_seconds": 0.0,
            "objective": None,
            "bound_used": False,
            "selected": None,
            "import_reservation": None,
        }
        model, variables, model_stats = self.build_compact_row_milp(
            self._columns,
            GurobiModel,
            quicksum,
        )
        try:
            result["build_seconds"] = round(perf_counter() - started, 3)
            result["model"] = model_stats
            remaining = deadline - perf_counter()
            allowance = min(
                float(self.benders_config.primal_seed_time_limit),
                remaining,
            )
            if allowance <= 1e-6:
                result["status"] = "time_limit_before_solve"
                return result
            self._set_gurobi_param(model, "TimeLimit", max(0.01, allowance))
            self._set_gurobi_param(model, "MIPGap", 0.0)
            model.optimize()
            result["status"] = self._gurobi_status_name(model)
            if self._gurobi_solution_count(model) <= 0:
                return result
            selected = self.selected_compact_row_values(model, variables)
            import_reservation = self._gurobi_import_reservation_values(
                model, variables
            )
            self._final_import_reservation = import_reservation
            reconstructed = self._selected_solution_energy(selected)
            solver_objective = self._gurobi_objective_value(model)
            if abs(reconstructed - solver_objective) > 1e-6:
                raise RuntimeError(
                    "LBBD primal-seed objective differs from reconstructed "
                    f"plan: model={solver_objective}, "
                    f"reconstructed={reconstructed}"
                )
            result.update(
                {
                    "feasible": True,
                    "objective": reconstructed,
                    "selected": selected,
                    "import_reservation": import_reservation,
                }
            )
            return result
        finally:
            result["seconds"] = round(perf_counter() - started, 3)
            self._free_gurobi_model(model)

    def _apply_master_start(
        self,
        variables: dict,
        selected: Counter[int],
        import_reservation: Counter[tuple[str, str, str]],
    ) -> tuple[
        Counter[tuple[str, str]],
        Counter[tuple[str, str, str]],
    ]:
        export_by_area: Counter[tuple[str, str]] = Counter()
        rows_by_area: defaultdict[
            tuple[OperationalKey, str], set[tuple[str, str]]
        ] = defaultdict(set)
        for index, multiplier in selected.items():
            if int(multiplier) <= 0:
                continue
            column = self._columns[index]
            quantity = int(column.quantity) * int(multiplier)
            export_by_area[(column.group_id, column.area_no)] += quantity
            anchor_row = next(
                row_no
                for bay_key, row_no, _quantity in column.row_allocation
                if bay_key == column.bay_key
            )
            rows_by_area[(column.group_key, column.area_no)].add(
                (column.bay_key, anchor_row)
            )
        import_by_area: Counter[tuple[str, str, str]] = Counter()
        for (flow, size, bay_key), quantity in import_reservation.items():
            import_by_area[(flow, size, self.bays[bay_key].area_no)] += int(
                quantity
            )
        for key, variable in variables["quantity"].items():
            variable.Start = float(export_by_area.get(key, 0))
        for key, variable in variables["import_quantity"].items():
            variable.Start = float(import_by_area.get(key, 0))
        for key, variable in variables["area_use"].items():
            variable.Start = 1.0 if export_by_area.get(
                (self._operational_groups[key[0]][0], key[1]), 0
            ) > 0 else 0.0
        for key, variable in variables["row_count"].items():
            variable.Start = float(len(rows_by_area.get(key, ())))
        self._primal_seed_selected = Counter(selected)
        self._primal_seed_import = Counter(import_reservation)
        self._primal_seed_group_area = Counter(export_by_area)
        self._primal_seed_import_area = Counter(import_by_area)
        return export_by_area, import_by_area

    def _add_support_repair_neighborhood(
        self,
        master,
        variables: dict,
        export_by_area: Counter[tuple[str, str]],
        import_by_area: Counter[tuple[str, str, str]],
    ) -> _SupportRepairNeighborhood | None:
        """Create an exact L1 neighborhood, initially fixed at the seed."""
        from gurobipy import quicksum

        if int(self.benders_config.support_repair_iterations) <= 0:
            return None
        seeded_variables = [
            (variable, int(export_by_area.get(key, 0)), f"export_{index}")
            for index, (key, variable) in enumerate(
                sorted(variables["quantity"].items())
            )
        ]
        seeded_variables.extend(
            (
                variable,
                int(import_by_area.get(key, 0)),
                f"import_{index}",
            )
            for index, (key, variable) in enumerate(
                sorted(variables["import_quantity"].items())
            )
        )
        if not seeded_variables:
            return None
        total_quantity = sum(self.group_demand.values()) + sum(
            self.import_total_by_flow_size.values()
        )
        target_radius = max(
            0,
            int(
                math.ceil(
                    float(self.benders_config.support_repair_fraction)
                    * total_quantity
                )
            ),
        )
        deviations = []
        links = []
        for variable, seed_value, label in seeded_variables:
            deviation = master.addVar(
                lb=0.0,
                name=f"support_repair_deviation_{label}",
            )
            links.append(
                master.addConstr(
                    deviation >= variable - seed_value,
                    name=f"support_repair_positive_{label}",
                )
            )
            links.append(
                master.addConstr(
                    deviation >= seed_value - variable,
                    name=f"support_repair_negative_{label}",
                )
            )
            deviations.append(deviation)
        limit = master.addConstr(
            quicksum(deviations) <= 0,
            name="primal_support_repair_neighborhood",
        )
        master.update()
        return _SupportRepairNeighborhood(
            limit=limit,
            variables=tuple(deviations),
            constraints=tuple(links),
            target_radius=target_radius,
        )

    @staticmethod
    def _remove_support_repair_neighborhood(
        master,
        neighborhood: _SupportRepairNeighborhood | None,
    ) -> None:
        if neighborhood is None:
            return
        master._model.remove(
            [neighborhood.limit, *neighborhood.constraints]
        )
        master._model.remove(list(neighborhood.variables))
        master.update()

    def solve(self) -> ColumnGenerationResult:
        """Run the capacity-strengthened conflict-graph LBBD algorithm."""
        started = perf_counter()
        total_limit = float(self.config.total_time_limit)
        if total_limit <= 0.0:
            total_limit = max(2.0, 2.0 * float(self.config.mip_time_limit))
        deadline = started + total_limit
        self._prepare_lbbd()
        preparation_seconds = perf_counter() - started
        if perf_counter() >= deadline:
            raise RuntimeError(
                "LBBD preprocessing consumed the complete time limit"
            )

        master, variables, master_stats = self._build_master()
        master_build_seconds = perf_counter() - started - preparation_seconds
        iteration_rows: list[dict] = []
        cache: dict[tuple, dict] = {}
        hall_signatures: set[tuple] = set()
        core_capacity_signatures: set[tuple] = set()
        best_selected: Counter[int] | None = None
        best_import: Counter[tuple[str, str, str]] | None = None
        best_objective = math.inf
        best_iteration = 0
        valid_lower_bound = -math.inf
        master_status = "not_solved"
        logic_feasibility_cuts = 0
        logic_optimality_cuts = 0
        dynamic_hall_cuts = 0
        conflict_core_capacity_cuts = 0
        dominance_binary_count = 0
        area_solve_count = 0
        area_cache_hits = 0
        converged = False
        termination_reason = "iteration_limit"
        primal_seed = {"attempted": False, "bound_used": False}
        best_source = "none"
        support_repair_neighborhood = None
        support_repair_radius = 0
        support_repair_variable_count = 0
        support_repair_iterations = 0
        support_repair_iteration_limit = 0
        support_repair_improvements = 0
        support_repair_skipped_optimality_cuts = 0

        try:
            for iteration in range(1, int(self.benders_config.max_iterations) + 1):
                iteration_in_support_repair = (
                    support_repair_neighborhood is not None
                )
                remaining = deadline - perf_counter()
                if remaining <= 1e-6:
                    termination_reason = "time_limit"
                    break
                master_time_slice = float(
                    self.benders_config.master_time_limit
                )
                if iteration_in_support_repair:
                    master_time_slice = min(
                        master_time_slice,
                        float(self.benders_config.area_time_limit),
                    )
                self._set_gurobi_param(
                    master,
                    "TimeLimit",
                    max(0.01, min(master_time_slice, remaining)),
                )
                master_started = perf_counter()
                master.optimize()
                master_seconds = perf_counter() - master_started
                master_status = self._gurobi_status_name(master)
                if self._gurobi_solution_count(master) <= 0:
                    if master_status == "infeasible":
                        termination_reason = "master_infeasible"
                    else:
                        termination_reason = "master_without_incumbent"
                    break
                master_bound = self._gurobi_dual_bound(master)
                if (
                    not iteration_in_support_repair
                    and math.isfinite(master_bound)
                ):
                    valid_lower_bound = max(valid_lower_bound, master_bound)
                master_objective = self._gurobi_objective_value(master)
                quantities, imports, theta = self._master_assignment(
                    master, variables
                )
                active_areas = sorted(
                    {
                        area_no
                        for (_group_id, area_no), value in quantities.items()
                        if value > 0
                    }
                    | {
                        area_no
                        for (_flow, _size, area_no), value in imports.items()
                        if value > 0
                    }
                )

                selected = Counter()
                import_reservation: Counter[tuple[str, str, str]] = Counter()
                area_objective = 0.0
                area_records: list[dict] = []
                all_feasible = True
                all_optimal = True
                incomplete = False
                cuts_this_iteration = 0

                for area_position, area_no in enumerate(active_areas):
                    remaining = deadline - perf_counter()
                    if remaining <= 1e-6:
                        incomplete = True
                        all_feasible = False
                        all_optimal = False
                        termination_reason = "time_limit"
                        break
                    group_quantities = {
                        group_id: value
                        for (group_id, candidate_area), value
                        in quantities.items()
                        if candidate_area == area_no and value > 0
                    }
                    import_quantities = {
                        (flow, size): value
                        for (flow, size, candidate_area), value
                        in imports.items()
                        if candidate_area == area_no and value > 0
                    }
                    assignment_key = self._area_assignment_key(
                        area_no, group_quantities, import_quantities
                    )
                    area_result = cache.get(assignment_key)
                    cached = area_result is not None
                    if cached:
                        area_cache_hits += 1
                    else:
                        unsolved_areas = len(active_areas) - area_position
                        fair_share = remaining / max(1, unsolved_areas)
                        area_result = self._solve_area_subproblem(
                            area_no,
                            group_quantities,
                            import_quantities,
                            min(
                                float(self.benders_config.area_time_limit),
                                fair_share,
                            ),
                        )
                        area_solve_count += 1
                        if area_result["optimal"] or (
                            area_result["status"] == "infeasible"
                        ):
                            cache[assignment_key] = area_result

                    area_records.append(
                        {
                            "area_no": area_no,
                            "status": area_result["status"],
                            "cached": cached,
                            "seconds": round(float(area_result["seconds"]), 4),
                            "objective": (
                                None
                                if not math.isfinite(area_result["objective"])
                                else float(area_result["objective"])
                            ),
                            "bound": (
                                None
                                if not math.isfinite(area_result["bound"])
                                else float(area_result["bound"])
                            ),
                            "conflict_core_size": len(
                                area_result.get("conflict_group_ids", ())
                            )
                            + len(
                                area_result.get("conflict_import_keys", ())
                            ),
                        }
                    )
                    if area_result["status"] == "infeasible":
                        all_feasible = False
                        all_optimal = False
                        conflict_group_ids = set(
                            area_result.get("conflict_group_ids", ())
                        )
                        conflict_import_keys = set(
                            area_result.get("conflict_import_keys", ())
                        )
                        conflict_core_available = bool(
                            conflict_group_ids or conflict_import_keys
                        )
                        core_quantities = {
                            key: value
                            for key, value in quantities.items()
                            if key[1] == area_no
                            and (
                                not conflict_core_available
                                or key[0] in conflict_group_ids
                            )
                        }
                        core_imports = {
                            key: value
                            for key, value in imports.items()
                            if key[2] == area_no
                            and (
                                not conflict_core_available
                                or (key[0], key[1])
                                in conflict_import_keys
                            )
                        }
                        if not core_quantities and not core_imports:
                            core_quantities = {
                                key: value
                                for key, value in quantities.items()
                                if key[1] == area_no
                            }
                            core_imports = {
                                key: value
                                for key, value in imports.items()
                                if key[2] == area_no
                            }
                        if self._add_violated_active_hall_cut(
                            master,
                            variables,
                            area_no,
                            core_quantities,
                            hall_signatures,
                            dynamic_hall_cuts + 1,
                        ):
                            dynamic_hall_cuts += 1
                            cuts_this_iteration += 1
                        capacity_cut_time = min(
                            2.0, max(0.0, deadline - perf_counter())
                        )
                        if capacity_cut_time > 1e-6 and (
                            self._add_conflict_core_capacity_cut(
                                master,
                                variables,
                                area_no,
                                conflict_group_ids,
                                conflict_import_keys,
                                quantities,
                                imports,
                                core_capacity_signatures,
                                conflict_core_capacity_cuts + 1,
                                capacity_cut_time,
                            )
                        ):
                            conflict_core_capacity_cuts += 1
                            cuts_this_iteration += 1
                        dominance_binary_count += self._add_dominance_cut(
                            master,
                            variables,
                            area_no,
                            core_quantities,
                            core_imports,
                            logic_feasibility_cuts + 1,
                            "feasibility",
                        )
                        logic_feasibility_cuts += 1
                        cuts_this_iteration += 1
                        continue
                    if not area_result["feasible"]:
                        all_feasible = False
                        all_optimal = False
                        incomplete = True
                        termination_reason = "area_subproblem_without_incumbent"
                        continue

                    selected.update(area_result["selected"])
                    import_reservation.update(
                        area_result["import_reservation"]
                    )
                    area_objective += float(area_result["objective"])
                    if not area_result["optimal"]:
                        all_optimal = False
                    cut_value = (
                        float(area_result["objective"])
                        if area_result["optimal"]
                        else float(area_result["bound"])
                    )
                    if (
                        math.isfinite(cut_value)
                        and cut_value > float(theta.get(area_no, 0.0)) + 1e-7
                    ):
                        if iteration_in_support_repair:
                            support_repair_skipped_optimality_cuts += 1
                        else:
                            dominance_binary_count += self._add_dominance_cut(
                                master,
                                variables,
                                area_no,
                                quantities,
                                imports,
                                logic_optimality_cuts + 1,
                                "optimality",
                                cut_value,
                            )
                            logic_optimality_cuts += 1
                            cuts_this_iteration += 1

                candidate_objective = None
                decomposition_objective = None
                if all_feasible and len(area_records) == len(active_areas):
                    self._final_import_reservation = import_reservation
                    candidate_objective = self._selected_solution_energy(selected)
                    decomposition_objective = (
                        master_objective
                        - sum(float(value) for value in theta.values())
                        + area_objective
                    )
                    if abs(candidate_objective - decomposition_objective) > 1e-6:
                        raise RuntimeError(
                            "LBBD objective decomposition differs from the "
                            "reconstructed row plan: "
                            f"decomposed={decomposition_objective}, "
                            f"reconstructed={candidate_objective}"
                        )
                    if candidate_objective + 1e-9 < best_objective:
                        if iteration_in_support_repair:
                            support_repair_improvements += 1
                        best_objective = candidate_objective
                        best_selected = Counter(selected)
                        best_import = Counter(import_reservation)
                        best_iteration = iteration
                        best_source = "exact_area_subproblems"

                iteration_rows.append(
                    {
                        "iteration": iteration,
                        "master_status": master_status,
                        "master_seconds": round(master_seconds, 4),
                        "master_objective": master_objective,
                        "master_bound": master_bound,
                        "master_bound_is_global": (
                            not iteration_in_support_repair
                        ),
                        "support_repair_iteration": (
                            iteration_in_support_repair
                        ),
                        "active_area_count": len(active_areas),
                        "cuts_added": cuts_this_iteration,
                        "all_area_subproblems_feasible": all_feasible,
                        "all_area_subproblems_optimal": all_optimal,
                        "candidate_objective": candidate_objective,
                        "area_subproblems": area_records,
                    }
                )
                if (
                    best_selected is None
                    and not primal_seed["attempted"]
                    and float(self.benders_config.primal_seed_time_limit) > 0.0
                    and deadline - perf_counter() > 1e-6
                ):
                    primal_seed = self._solve_compact_primal_seed(deadline)
                    if primal_seed["feasible"]:
                        seeded_selected = Counter(primal_seed["selected"])
                        seeded_import = Counter(
                            primal_seed["import_reservation"]
                        )
                        seeded_objective = float(primal_seed["objective"])
                        best_selected = seeded_selected
                        best_import = seeded_import
                        best_objective = seeded_objective
                        best_iteration = iteration
                        best_source = "compact_primal_seed"
                        seed_export, seed_import_by_area = (
                            self._apply_master_start(
                                variables,
                                seeded_selected,
                                seeded_import,
                            )
                        )
                        support_repair_neighborhood = (
                            self._add_support_repair_neighborhood(
                                master,
                                variables,
                                seed_export,
                                seed_import_by_area,
                            )
                        )
                        if support_repair_neighborhood is not None:
                            support_repair_radius = (
                                support_repair_neighborhood.target_radius
                            )
                            support_repair_iteration_limit = (
                                1
                                if support_repair_radius <= 0
                                else int(
                                    self.benders_config.support_repair_iterations
                                )
                            )
                            support_repair_variable_count = len(
                                support_repair_neighborhood.variables
                            )
                        master.update()
                if iteration_in_support_repair:
                    support_repair_iterations += 1
                    if support_repair_neighborhood is not None:
                        support_repair_neighborhood.limit.RHS = float(
                            support_repair_neighborhood.target_radius
                            * support_repair_iterations
                        )
                        master.update()
                    if (
                        support_repair_iterations
                        >= support_repair_iteration_limit
                    ):
                        self._remove_support_repair_neighborhood(
                            master, support_repair_neighborhood
                        )
                        support_repair_neighborhood = None
                if incomplete and cuts_this_iteration == 0:
                    if iteration_in_support_repair:
                        self._remove_support_repair_neighborhood(
                            master, support_repair_neighborhood
                        )
                        support_repair_neighborhood = None
                        termination_reason = "support_repair_stalled"
                        continue
                    break
                if incomplete:
                    termination_reason = "continuing_after_feasibility_cuts"
                    continue
                if (
                    all_feasible
                    and all_optimal
                    and cuts_this_iteration == 0
                    and master_status == "optimal"
                    and not iteration_in_support_repair
                ):
                    converged = True
                    termination_reason = "optimality_proven"
                    break
                if cuts_this_iteration == 0 and master_status != "optimal":
                    termination_reason = "master_time_slice"
                    continue
            else:
                termination_reason = "iteration_limit"

            if best_selected is None or best_import is None:
                raise RuntimeError(
                    "LBBD did not find a feasible complete row allocation; "
                    f"termination={termination_reason}, "
                    f"master_status={master_status}"
                )
            self._final_import_reservation = best_import
            if converged:
                valid_lower_bound = best_objective
            if not math.isfinite(valid_lower_bound):
                valid_lower_bound = 0.0
            valid_lower_bound = min(valid_lower_bound, best_objective)
            absolute_gap = max(0.0, best_objective - valid_lower_bound)
            relative_gap = absolute_gap / max(abs(best_objective), 1e-12)
            diagnostics = {
                **self._base_diagnostics(),
                "master_algorithm": "logic_based_benders_decomposition",
                "master_status": "optimal" if converged else master_status,
                "master_bound_scope": "valid_strengthened_lbbd_master",
                "master_objective": best_objective,
                "master_mip_gap": relative_gap,
                "complete_model_lower_bound": valid_lower_bound,
                "complete_model_absolute_gap": absolute_gap,
                "complete_model_relative_gap": relative_gap,
                "complete_model_gap_source": "lbbd_master_bound",
                "hard_demand_balance": True,
                "candidate_row_location_count": len(self._columns),
                "selected_location_count": len(best_selected),
                "lbbd_converged": converged,
                "lbbd_termination_reason": termination_reason,
                "lbbd_iteration_count": len(iteration_rows),
                "lbbd_best_iteration": best_iteration,
                "lbbd_best_incumbent_source": best_source,
                "lbbd_iterations": iteration_rows,
                "lbbd_cut_counts": {
                    "initial_conflict_clique": master_stats[
                        "conflict_clique_cut_count"
                    ],
                    "initial_conflict_hall_capacity": master_stats[
                        "conflict_hall_capacity_cut_count"
                    ],
                    "dynamic_conflict_hall_capacity": dynamic_hall_cuts,
                    "dynamic_conflict_core_capacity": (
                        conflict_core_capacity_cuts
                    ),
                    "logic_feasibility": logic_feasibility_cuts,
                    "logic_optimality": logic_optimality_cuts,
                    "dominance_binary_variables": dominance_binary_count,
                },
                "lbbd_area_subproblem_solve_count": area_solve_count,
                "lbbd_area_subproblem_cache_hits": area_cache_hits,
                "lbbd_area_subproblem_count": len(self._area_subproblems),
                "lbbd_preparation_seconds": round(preparation_seconds, 3),
                "lbbd_master_build_seconds": round(master_build_seconds, 3),
                "lbbd_total_solve_seconds": round(
                    perf_counter() - started, 3
                ),
                "lbbd_master": master_stats,
                "lbbd_compact_primal_seed": {
                    key: value
                    for key, value in primal_seed.items()
                    if key not in {
                        "selected",
                        "import_reservation",
                    }
                },
                "lbbd_support_repair": {
                    "enabled": int(
                        self.benders_config.support_repair_iterations
                    ) > 0,
                    "iterations": support_repair_iterations,
                    "iteration_limit": support_repair_iteration_limit,
                    "radius_step_boxes": support_repair_radius,
                    "maximum_planned_radius_boxes": (
                        support_repair_radius
                        * max(
                            0,
                            support_repair_iteration_limit - 1,
                        )
                    ),
                    "last_used_radius_boxes": (
                        support_repair_radius
                        * max(0, support_repair_iterations - 1)
                    ),
                    "quantity_deviation_variable_count": (
                        support_repair_variable_count
                    ),
                    "incumbent_improvements": support_repair_improvements,
                    "skipped_conditional_optimality_cuts": (
                        support_repair_skipped_optimality_cuts
                    ),
                    "bound_used": False,
                },
                "persistent_area_subproblems": True,
                "compact_primal_seed_bound_used": False,
                "column_generation_called": False,
            }
            result = self._assemble_result(best_selected, diagnostics)
            result.columns = self._selected_lbbd_columns(best_selected)
            return result
        finally:
            if support_repair_neighborhood is not None:
                try:
                    self._remove_support_repair_neighborhood(
                        master, support_repair_neighborhood
                    )
                except Exception:
                    pass
            self._free_gurobi_model(master)
            for subproblem in self._area_subproblems.values():
                self._free_gurobi_model(subproblem.model)


__all__ = ["LogicBendersConfig", "LogicBendersPlanner"]
