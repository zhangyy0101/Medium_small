"""Voyage-row-resource logic-based Benders decomposition."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from time import perf_counter

from .direct_milp import DirectMilpPlanner
from .gurobi_backend import GurobiModel
from .planner import ColumnGenerationResult, PlacementColumn


OperationalKey = tuple[str, ...]
PhysicalSlot = tuple[str, str]
QuantityKey = tuple[str, str]
OwnerKey = tuple[int, str]
TemplateSignature = tuple[
    str, str, str, tuple[PhysicalSlot, ...]
]


@dataclass(frozen=True)
class LogicBendersConfig:
    """Instance-independent controls for the voyage-resource LBBD loop."""

    max_iterations: int = 40
    master_feasibility_time_limit: float = 30.0
    master_time_limit: float = 20.0
    voyage_time_limit: float = 8.0
    max_cliques_per_bay: int = 100


@dataclass(frozen=True)
class _RowTemplate:
    template_id: int
    voyage_id: str
    bay_key: str
    row_no: str
    area_no: str
    slots: tuple[PhysicalSlot, ...]
    candidate_indices: tuple[int, ...]


@dataclass
class _VoyageSubproblem:
    voyage_id: str
    model: GurobiModel
    placement_variables: dict[int, object]
    quantity_balance: dict[QuantityKey, object]
    owner_limits: dict[int, object]
    build_seconds: float
    variable_count: int


class VoyageResourceBendersPlanner(DirectMilpPlanner):
    """LBBD with voyage-owned row footprints and exact voyage subproblems."""

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
        if float(self.benders_config.master_feasibility_time_limit) < 0.0:
            raise ValueError(
                "LBBD master_feasibility_time_limit cannot be negative"
            )
        if float(self.benders_config.voyage_time_limit) <= 0.0:
            raise ValueError("LBBD voyage_time_limit must be positive")
        if int(self.benders_config.max_cliques_per_bay) <= 0:
            raise ValueError("LBBD max_cliques_per_bay must be positive")

        self._voyages: tuple[str, ...] = ()
        self._group_ids_by_voyage: dict[str, tuple[str, ...]] = {}
        self._candidate_capacity: dict[int, int] = {}
        self._candidate_indices_by_group_bay: dict[
            QuantityKey, tuple[int, ...]
        ] = {}
        self._quantity_upper: dict[QuantityKey, int] = {}
        self._quantity_representative: dict[QuantityKey, int] = {}
        self._template_by_candidate: dict[int, int] = {}
        self._owner_key_by_candidate: dict[int, OwnerKey] = {}
        self._owner_keys_by_template: dict[
            int, tuple[OwnerKey, ...]
        ] = {}
        self._templates: dict[int, _RowTemplate] = {}
        self._template_ids_by_voyage: dict[str, tuple[int, ...]] = {}
        self._template_ids_by_voyage_bay: dict[
            tuple[str, str], tuple[int, ...]
        ] = {}
        self._template_ids_by_slot: dict[
            PhysicalSlot, tuple[int, ...]
        ] = {}
        self._operational_groups: dict[OperationalKey, tuple[str, ...]] = {}
        self._representative_group: dict[OperationalKey, object] = {}
        self._operational_bay_groups: dict[
            tuple[OperationalKey, str], tuple[str, ...]
        ] = {}
        self._row_capacity_profile: dict[
            tuple[OperationalKey, str], tuple[int, ...]
        ] = {}
        self._row_cliques_by_voyage_bay: dict[
            tuple[str, str], tuple[tuple[OperationalKey, ...], ...]
        ] = {}
        self._voyage_subproblems: dict[str, _VoyageSubproblem] = {}

    @staticmethod
    def _anchor_row(column: PlacementColumn) -> str:
        return next(
            str(row_no)
            for bay_key, row_no, quantity in column.row_allocation
            if bay_key == column.bay_key and int(quantity) > 0
        )

    @staticmethod
    def _template_signature(column: PlacementColumn) -> TemplateSignature:
        slots = tuple(
            sorted(
                (str(bay_key), str(row_no))
                for bay_key, row_no, quantity in column.row_allocation
                if int(quantity) > 0
            )
        )
        return (
            str(column.voyage_id),
            str(column.bay_key),
            VoyageResourceBendersPlanner._anchor_row(column),
            slots,
        )

    def _prepare_decomposition(self) -> None:
        self._prepare_master_index_sets()
        self._prepare_objective_normalization()
        self._initialize_location_pool()

        by_group_bay: defaultdict[QuantityKey, list[int]] = defaultdict(list)
        by_voyage: defaultdict[str, list[str]] = defaultdict(list)
        for group in self.groups:
            by_voyage[group.voyage_id].append(group.group_id)
            for candidate in self._base_placements_for_group(group):
                index = self._append_generated_column(candidate)
                by_group_bay[(group.group_id, candidate.bay_key)].append(index)
        self._voyages = tuple(sorted(by_voyage))
        self._group_ids_by_voyage = {
            voyage_id: tuple(sorted(group_ids))
            for voyage_id, group_ids in sorted(by_voyage.items())
        }
        self._candidate_indices_by_group_bay = {
            key: tuple(indices)
            for key, indices in sorted(by_group_bay.items())
        }
        self._candidate_capacity = {
            index: int(
                self._base_location_capacity(
                    self.groups_by_id[self._columns[index].group_id],
                    self._columns[index],
                )
            )
            for index in range(len(self._columns))
        }
        self._quantity_representative = {
            key: indices[0]
            for key, indices in self._candidate_indices_by_group_bay.items()
        }
        self._quantity_upper = {
            key: min(
                int(self.group_demand[key[0]]),
                sum(self._candidate_capacity[index] for index in indices),
            )
            for key, indices in self._candidate_indices_by_group_bay.items()
        }

        candidates_by_signature: defaultdict[
            TemplateSignature, list[int]
        ] = defaultdict(list)
        for index, column in enumerate(self._columns):
            candidates_by_signature[self._template_signature(column)].append(
                index
            )
        templates: dict[int, _RowTemplate] = {}
        template_by_candidate = {}
        by_template_voyage: defaultdict[str, list[int]] = defaultdict(list)
        by_template_voyage_bay: defaultdict[
            tuple[str, str], list[int]
        ] = defaultdict(list)
        by_slot: defaultdict[PhysicalSlot, list[int]] = defaultdict(list)
        for template_id, signature in enumerate(
            sorted(candidates_by_signature), start=1
        ):
            voyage_id, bay_key, row_no, slots = signature
            indices = tuple(sorted(candidates_by_signature[signature]))
            template = _RowTemplate(
                template_id=template_id,
                voyage_id=voyage_id,
                bay_key=bay_key,
                row_no=row_no,
                area_no=self.bays[bay_key].area_no,
                slots=slots,
                candidate_indices=indices,
            )
            templates[template_id] = template
            by_template_voyage[voyage_id].append(template_id)
            by_template_voyage_bay[(voyage_id, bay_key)].append(template_id)
            for slot in slots:
                by_slot[slot].append(template_id)
            for index in indices:
                template_by_candidate[index] = template_id
        self._templates = templates
        self._template_by_candidate = template_by_candidate
        owner_key_by_candidate = {
            index: (
                template_by_candidate[index],
                self._row_mix_key_for_column(self._columns[index]),
            )
            for index in range(len(self._columns))
        }
        owner_keys_by_template: defaultdict[int, set[OwnerKey]] = (
            defaultdict(set)
        )
        for owner_key in owner_key_by_candidate.values():
            owner_keys_by_template[owner_key[0]].add(owner_key)
        self._owner_key_by_candidate = owner_key_by_candidate
        self._owner_keys_by_template = {
            template_id: tuple(sorted(owner_keys))
            for template_id, owner_keys in sorted(
                owner_keys_by_template.items()
            )
        }
        self._template_ids_by_voyage = {
            key: tuple(values)
            for key, values in sorted(by_template_voyage.items())
        }
        self._template_ids_by_voyage_bay = {
            key: tuple(values)
            for key, values in sorted(by_template_voyage_bay.items())
        }
        self._template_ids_by_slot = {
            key: tuple(values) for key, values in sorted(by_slot.items())
        }

        operational_groups: defaultdict[
            OperationalKey, list[str]
        ] = defaultdict(list)
        representatives = {}
        operational_bay_groups: defaultdict[
            tuple[OperationalKey, str], set[str]
        ] = defaultdict(set)
        profile_by_operational_bay: defaultdict[
            tuple[OperationalKey, str], dict[int, int]
        ] = defaultdict(dict)
        for group in self.groups:
            operational_key = self._operational_group_key(group)
            operational_groups[operational_key].append(group.group_id)
            representatives.setdefault(operational_key, group)
        for quantity_key, indices in self._candidate_indices_by_group_bay.items():
            group_id, bay_key = quantity_key
            operational_key = self._operational_group_key(
                self.groups_by_id[group_id]
            )
            operational_bay_groups[(operational_key, bay_key)].add(group_id)
            for index in indices:
                template_id = self._template_by_candidate[index]
                profile = profile_by_operational_bay[
                    (operational_key, bay_key)
                ]
                profile[template_id] = max(
                    profile.get(template_id, 0),
                    self._candidate_capacity[index],
                )
        self._operational_groups = {
            key: tuple(sorted(group_ids))
            for key, group_ids in sorted(operational_groups.items())
        }
        self._representative_group = representatives
        self._operational_bay_groups = {
            key: tuple(sorted(group_ids))
            for key, group_ids in sorted(operational_bay_groups.items())
        }
        self._row_capacity_profile = {
            key: tuple(sorted(profile.values(), reverse=True))
            for key, profile in sorted(profile_by_operational_bay.items())
        }
        self._row_cliques_by_voyage_bay = {
            key: self._maximal_row_cliques(*key)
            for key in self._template_ids_by_voyage_bay
        }

    def _maximal_row_cliques(
        self, voyage_id: str, bay_key: str
    ) -> tuple[tuple[OperationalKey, ...], ...]:
        nodes = sorted(
            operational_key
            for operational_key, candidate_bay in self._operational_bay_groups
            if candidate_bay == bay_key
            and self._representative_group[operational_key].voyage_id
            == voyage_id
        )
        neighbors = {node: set() for node in nodes}
        for position, first in enumerate(nodes):
            for second in nodes[position + 1 :]:
                if self._groups_are_incompatible_on_one_row(
                    self._representative_group[first],
                    self._representative_group[second],
                ):
                    neighbors[first].add(second)
                    neighbors[second].add(first)
        maximal: list[tuple[OperationalKey, ...]] = []

        def expand(clique, candidates, excluded) -> None:
            if not candidates and not excluded:
                if clique:
                    maximal.append(tuple(sorted(clique)))
                return
            pool = candidates | excluded
            pivot = max(
                pool,
                key=lambda node: len(candidates & neighbors[node]),
                default=None,
            )
            extension = candidates - (
                neighbors[pivot] if pivot is not None else set()
            )
            for node in sorted(extension):
                expand(
                    clique | {node},
                    candidates & neighbors[node],
                    excluded & neighbors[node],
                )
                candidates.remove(node)
                excluded.add(node)

        expand(set(), set(nodes), set())
        ranked = sorted(
            set(maximal), key=lambda clique: (-len(clique), repr(clique))
        )
        return tuple(
            ranked[: int(self.benders_config.max_cliques_per_bay)]
        )

    @staticmethod
    def _set_constraint_rhs(constraint, value: float) -> None:
        constraint.RHS = float(value)

    def _build_master(self):
        from gurobipy import quicksum

        model = GurobiModel("voyage_row_resource_lbbd_master")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        self._set_gurobi_param(model, "MIPGap", 0.0)
        self._set_gurobi_param(model, "DualReductions", 0)
        self._set_gurobi_param(model, "MIPFocus", 1)
        self._set_gurobi_param(model, "Heuristics", 0.5)
        model.setMinimize()

        quantity = {
            key: model.addVar(
                lb=0.0,
                ub=float(self._quantity_upper[key]),
                vtype="I",
                obj=float(
                    self._columns[self._quantity_representative[key]].intrinsic_cost
                ),
                name=(
                    f"q_bay_{key[0]}_{self._key_name((key[1],))}"
                ),
            )
            for key in sorted(self._candidate_indices_by_group_bay)
        }
        owner = {
            owner_key: model.addVar(
                vtype="B",
                name=f"owner_state_{owner_key[0]}_{position}",
            )
            for template_id, owner_keys in sorted(
                self._owner_keys_by_template.items()
            )
            for position, owner_key in enumerate(owner_keys, start=1)
        }
        routing_flow = {
            index: model.addVar(
                lb=0.0,
                ub=float(self._candidate_capacity[index]),
                name=f"routing_flow_{index}",
            )
            for index in range(len(self._columns))
        }
        row_count = {
            key: model.addVar(
                lb=0.0,
                ub=float(len(self._row_capacity_profile[key])),
                vtype="I",
                name=(
                    f"rows_{self._key_name(key[0])}_"
                    f"{self._key_name((key[1],))}"
                ),
            )
            for key in sorted(self._operational_bay_groups)
        }
        operational_area_keys = sorted(
            {
                (operational_key, self.bays[bay_key].area_no)
                for operational_key, bay_key in self._operational_bay_groups
            }
        )
        area_use = {
            key: model.addVar(
                vtype="B",
                obj=self._area_activation_penalty(),
                name=(
                    f"use_area_{self._key_name(key[0])}_"
                    f"{self._key_name((key[1],))}"
                ),
            )
            for key in operational_area_keys
        }
        theta = {
            voyage_id: model.addVar(
                lb=0.0,
                obj=1.0,
                name=f"theta_{self._key_name((voyage_id,))}",
            )
            for voyage_id in self._voyages
        }
        import_reserve = {
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
        }
        constraints: dict[str, dict] = defaultdict(dict)

        for group in self.groups:
            keys = [
                key for key in quantity if key[0] == group.group_id
            ]
            if not keys:
                raise ValueError(
                    "LBBD group has no feasible bay: "
                    f"group={group.group_id}"
                )
            constraints["group_demand"][group.group_id] = model.addConstr(
                quicksum(quantity[key] for key in keys)
                == int(group.demand),
                name=f"demand_{group.group_id}",
            )

        for key, variable in sorted(quantity.items()):
            indices = self._candidate_indices_by_group_bay[key]
            constraints["routing_flow_balance"][key] = model.addConstr(
                quicksum(routing_flow[index] for index in indices)
                == variable,
                name=(
                    f"routing_balance_{key[0]}_"
                    f"{self._key_name((key[1],))}"
                ),
            )
            for index in indices:
                owner_key = self._owner_key_by_candidate[index]
                constraints["routing_owner_link"][index] = model.addConstr(
                    routing_flow[index]
                    <= self._candidate_capacity[index]
                    * owner[owner_key],
                    name=f"routing_owner_{index}",
                )

        for template_id, template in sorted(self._templates.items()):
            capacity = max(
                self._candidate_capacity[index]
                for index in template.candidate_indices
            )
            constraints["routing_template_capacity"][template_id] = (
                model.addConstr(
                    quicksum(
                        routing_flow[index]
                        for index in template.candidate_indices
                    )
                    <= capacity
                    * quicksum(
                        owner[owner_key]
                        for owner_key in self._owner_keys_by_template[
                            template_id
                        ]
                    ),
                    name=f"routing_template_capacity_{template_id}",
                )
            )
            constraints["owner_state_exclusive"][template_id] = (
                model.addConstr(
                    quicksum(
                        owner[owner_key]
                        for owner_key in self._owner_keys_by_template[
                            template_id
                        ]
                    )
                    <= 1.0,
                    name=f"owner_state_exclusive_{template_id}",
                )
            )
            for owner_key in self._owner_keys_by_template[template_id]:
                owner_indices = [
                    index
                    for index in template.candidate_indices
                    if self._owner_key_by_candidate[index] == owner_key
                ]
                constraints["owner_positive_flow"][owner_key] = (
                    model.addConstr(
                        owner[owner_key]
                        <= quicksum(
                            routing_flow[index]
                            for index in owner_indices
                        ),
                        name=(
                            f"owner_positive_flow_{owner_key[0]}_"
                            f"{self._key_name((owner_key[1],))}"
                        ),
                    )
                )

        for voyage_bay, template_ids in sorted(
            self._template_ids_by_voyage_bay.items()
        ):
            voyage_id, bay_key = voyage_bay
            operational_keys = [
                operational_key
                for operational_key, candidate_bay
                in self._operational_bay_groups
                if candidate_bay == bay_key
                and self._representative_group[
                    operational_key
                ].voyage_id == voyage_id
            ]
            constraints["owner_count_upper"][voyage_bay] = model.addConstr(
                quicksum(
                    owner[owner_key]
                    for template_id in template_ids
                    for owner_key in self._owner_keys_by_template[
                        template_id
                    ]
                )
                <= quicksum(
                    row_count[(operational_key, bay_key)]
                    for operational_key in operational_keys
                ),
                name=f"owner_count_upper_{self._key_name(voyage_bay)}",
            )

        for slot, template_ids in sorted(self._template_ids_by_slot.items()):
            constraints["physical_row_owner"][slot] = model.addConstr(
                quicksum(
                    owner[owner_key]
                    for template_id in template_ids
                    for owner_key in self._owner_keys_by_template[
                        template_id
                    ]
                )
                <= 1.0,
                name=f"physical_owner_{self._key_name(slot)}",
            )

        for key, group_ids in sorted(self._operational_bay_groups.items()):
            operational_key, bay_key = key
            assigned = quicksum(
                quantity[(group_id, bay_key)]
                for group_id in group_ids
                if (group_id, bay_key) in quantity
            )
            compatible_owners = sorted(
                {
                    self._owner_key_by_candidate[index]
                    for group_id in group_ids
                    for index in self._candidate_indices_by_group_bay.get(
                        (group_id, bay_key), ()
                    )
                }
            )
            constraints["row_count_box_upper"][key] = model.addConstr(
                row_count[key] <= assigned,
                name=(
                    f"row_count_box_upper_"
                    f"{self._key_name((*operational_key, bay_key))}"
                ),
            )
            constraints["row_count_owner_upper"][key] = model.addConstr(
                row_count[key]
                <= quicksum(
                    owner[owner_key] for owner_key in compatible_owners
                ),
                name=(
                    f"row_count_owner_upper_"
                    f"{self._key_name((*operational_key, bay_key))}"
                ),
            )
            prefix_capacity = 0
            previous_slope = None
            for breakpoint, slope in enumerate(
                self._row_capacity_profile[key]
            ):
                if previous_slope is None or slope != previous_slope:
                    cover_key = (operational_key, bay_key, breakpoint)
                    constraints["row_cover"][cover_key] = model.addConstr(
                        assigned
                        <= prefix_capacity
                        + int(slope) * (row_count[key] - breakpoint),
                        name=(
                            f"row_cover_{self._key_name(operational_key)}_"
                            f"{self._key_name((bay_key, str(breakpoint)))}"
                        ),
                    )
                prefix_capacity += int(slope)
                previous_slope = int(slope)

        owner_keys_by_class: defaultdict[
            tuple[str, str, str], list[OwnerKey]
        ] = defaultdict(list)
        for owner_key in owner:
            template = self._templates[owner_key[0]]
            owner_keys_by_class[
                (template.voyage_id, template.bay_key, owner_key[1])
            ].append(owner_key)
        class_owner_cover_count = 0
        for class_key, owner_keys in sorted(owner_keys_by_class.items()):
            voyage_id, bay_key, row_class = class_key
            compatible_operational_keys = [
                operational_key
                for operational_key, candidate_bay
                in self._operational_bay_groups
                if candidate_bay == bay_key
                and self._representative_group[
                    operational_key
                ].voyage_id == voyage_id
                and self._row_mix_key_for_group(
                    self._representative_group[operational_key]
                )
                == row_class
            ]
            constraints["row_class_owner_cover"][class_key] = (
                model.addConstr(
                    quicksum(owner[owner_key] for owner_key in owner_keys)
                    <= quicksum(
                        row_count[(operational_key, bay_key)]
                        for operational_key in compatible_operational_keys
                    ),
                    name=(
                        "row_class_owner_cover_"
                        f"{self._key_name(class_key)}"
                    ),
                )
            )
            class_owner_cover_count += 1

        clique_count = 0
        clique_capacity_count = 0
        for voyage_bay, cliques in sorted(
            self._row_cliques_by_voyage_bay.items()
        ):
            voyage_id, bay_key = voyage_bay
            for clique_index, clique in enumerate(cliques):
                owner_keys = sorted(
                    {
                        self._owner_key_by_candidate[index]
                        for operational_key in clique
                        for group_id in self._operational_bay_groups.get(
                            (operational_key, bay_key), ()
                        )
                        for index in self._candidate_indices_by_group_bay.get(
                            (group_id, bay_key), ()
                        )
                    }
                )
                if not owner_keys:
                    continue
                cut_key = (voyage_id, bay_key, clique_index)
                constraints["row_conflict_clique"][cut_key] = model.addConstr(
                    quicksum(
                        row_count[(operational_key, bay_key)]
                        for operational_key in clique
                    )
                    <= quicksum(
                        owner[owner_key] for owner_key in owner_keys
                    ),
                    name=(
                        f"row_clique_{self._key_name((voyage_id, bay_key, str(clique_index)))}"
                    ),
                )
                clique_count += 1
                clique_group_ids = {
                    group_id
                    for operational_key in clique
                    for group_id in self._operational_bay_groups.get(
                        (operational_key, bay_key), ()
                    )
                }
                clique_quantity_keys = [
                    (group_id, bay_key)
                    for group_id in sorted(clique_group_ids)
                    if (group_id, bay_key) in quantity
                ]
                capacity_by_owner = {
                    owner_key: max(
                        (
                            self._candidate_capacity[index]
                            for quantity_key in clique_quantity_keys
                            for index in self._candidate_indices_by_group_bay[
                                quantity_key
                            ]
                            if self._owner_key_by_candidate[index]
                            == owner_key
                        ),
                        default=0,
                    )
                    for owner_key in owner_keys
                }
                constraints["row_conflict_hall_capacity"][cut_key] = (
                    model.addConstr(
                        quicksum(
                            quantity[quantity_key]
                            for quantity_key in clique_quantity_keys
                        )
                        <= quicksum(
                            capacity * owner[owner_key]
                            for owner_key, capacity
                            in capacity_by_owner.items()
                        ),
                        name=(
                            f"row_clique_capacity_"
                            f"{self._key_name((voyage_id, bay_key, str(clique_index)))}"
                        ),
                    )
                )
                clique_capacity_count += 1

        q_coefficients: defaultdict[
            str, defaultdict[object, list[tuple[QuantityKey, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        for key, index in self._quantity_representative.items():
            for section, values in self._placement_master_coefficients(
                self._columns[index]
            ).items():
                for coefficient_key, coefficient in values.items():
                    if coefficient:
                        q_coefficients[section][coefficient_key].append(
                            (key, float(coefficient))
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
            items = q_coefficients["bay_capacity_limit"].get(bay_key, [])
            constraints["bay_capacity"][bay_key] = model.addConstr(
                quicksum(
                    coefficient * quantity[key]
                    for key, coefficient in items
                )
                + quicksum(import_by_bay.get(bay_key, []))
                <= int(self.bays[bay_key].physical_capacity),
                name=f"bay_capacity_{self._key_name((bay_key,))}",
            )
        for bay_size in sorted(self._master_bay_size_keys):
            bay_key, size = bay_size
            items = q_coefficients["bay_size_limit"].get(bay_size, [])
            constraints["bay_size_capacity"][bay_size] = model.addConstr(
                quicksum(
                    coefficient * quantity[key]
                    for key, coefficient in items
                )
                + quicksum(import_by_bay_size.get(bay_size, []))
                <= int(self.bays[bay_key].cap_by_size.get(size, 0)),
                name=f"bay_size_{self._key_name(bay_size)}",
            )

        stack_variables_by_bay_size: defaultdict[tuple[str, str], list] = (
            defaultdict(list)
        )
        for stack_key in sorted(self._master_stack_keys):
            bay_key, _mix_key, size = stack_key
            group = self.groups_by_id.get(
                self._master_stack_sample_group.get(stack_key, "")
            )
            if group is None:
                continue
            stack_count = self._stack_count_for_group(
                bay_key, size, group
            )
            unit_capacity = self._stack_unit_capacity_for_group(
                bay_key, size, group
            )
            if stack_count <= 0 or unit_capacity <= 0:
                continue
            stack = model.addVar(
                lb=0.0,
                ub=float(stack_count),
                vtype="I",
                name=f"stack_{self._key_name(stack_key)}",
            )
            items = q_coefficients["bay_port_stack_link"].get(
                stack_key, []
            )
            constraints["stack_load"][stack_key] = model.addConstr(
                quicksum(
                    coefficient * quantity[key]
                    for key, coefficient in items
                )
                <= unit_capacity * stack,
                name=f"stack_load_{self._key_name(stack_key)}",
            )
            stack_variables_by_bay_size[(bay_key, size)].append(stack)
        for bay_size, stacks in stack_variables_by_bay_size.items():
            constraints["stack_total"][bay_size] = model.addConstr(
                quicksum(stacks)
                <= self._stack_count_for_bay_size(*bay_size),
                name=f"stack_total_{self._key_name(bay_size)}",
            )

        for key, required in sorted(self.import_total_by_flow_size.items()):
            candidates = import_by_flow_size.get(key, [])
            if not candidates:
                raise ValueError(
                    "LBBD import demand has no compatible bay: "
                    f"flow={key[0]}, size={key[1]}"
                )
            constraints["import_total"][key] = model.addConstr(
                quicksum(candidates) == int(required),
                name=f"import_total_{self._key_name(key)}",
            )

        for guidance_key in sorted(self._master_area_guidance_keys):
            target = self._area_size_target(*guidance_key)
            positive = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"guide_pos_{self._key_name(guidance_key)}",
            )
            negative = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"guide_neg_{self._key_name(guidance_key)}",
            )
            items = q_coefficients["area_guidance_balance"].get(
                guidance_key, []
            )
            constraints["export_guidance"][guidance_key] = model.addConstr(
                quicksum(
                    coefficient * quantity[key]
                    for key, coefficient in items
                )
                - target
                == positive - negative,
                name=f"guidance_{self._key_name(guidance_key)}",
            )

        for key, use in area_use.items():
            operational_key, area_no = key
            q_keys = [
                (group_id, bay_key)
                for group_id in self._operational_groups[operational_key]
                for bay_key in self.bays_by_area.get(area_no, ())
                if (group_id, bay_key) in quantity
            ]
            assigned = quicksum(quantity[q_key] for q_key in q_keys)
            demand = sum(
                self.group_demand[group_id]
                for group_id in self._operational_groups[operational_key]
            )
            constraints["area_use_upper"][key] = model.addConstr(
                assigned <= int(demand) * use,
                name=(
                    f"area_use_upper_{self._key_name((*operational_key, area_no))}"
                ),
            )
            constraints["area_use_lower"][key] = model.addConstr(
                use <= assigned,
                name=(
                    f"area_use_lower_{self._key_name((*operational_key, area_no))}"
                ),
            )

        for voyage_id in self._voyages:
            constraints["theta_analytic"][voyage_id] = model.addConstr(
                theta[voyage_id]
                >= self._row_activation_penalty()
                * quicksum(
                    variable
                    for (operational_key, _bay_key), variable
                    in row_count.items()
                    if self._representative_group[
                        operational_key
                    ].voyage_id == voyage_id
                ),
                name=f"theta_analytic_{self._key_name((voyage_id,))}",
            )

        constraints["import_reference"] = self._add_import_reference_deviation(
            quicksum,
            model,
            import_by_flow_area_size,
            objective_mode="full",
        )
        bay_compatibility = self._add_bay_compatibility_constraints(
            quicksum,
            model,
            quantity,
            q_coefficients["bay_attr_link"],
            relax=False,
        )
        constraints.update(bay_compatibility)

        offset = -len(self._operational_groups) * (
            self._area_activation_penalty()
            + self._row_activation_penalty()
        )
        model.addVar(
            lb=1.0,
            ub=1.0,
            obj=offset,
            name="voyage_lbbd_objective_offset",
        )
        model.update()
        return model, {
            "quantity": quantity,
            "owner": owner,
            "routing_flow": routing_flow,
            "row_count": row_count,
            "area_use": area_use,
            "theta": theta,
            "import_reserve": import_reserve,
        }, {
            "master_variable_count": len(model.getVars()),
            "group_bay_quantity_count": len(quantity),
            "row_footprint_template_count": len(self._templates),
            "voyage_row_class_binary_count": len(owner),
            "continuous_routing_flow_count": len(routing_flow),
            "operational_bay_row_count_count": len(row_count),
            "physical_row_owner_constraint_count": len(
                constraints["physical_row_owner"]
            ),
            "owner_positive_flow_constraint_count": len(
                constraints["owner_positive_flow"]
            ),
            "row_count_box_upper_constraint_count": len(
                constraints["row_count_box_upper"]
            ),
            "row_count_owner_upper_constraint_count": len(
                constraints["row_count_owner_upper"]
            ),
            "row_class_owner_cover_constraint_count": (
                class_owner_cover_count
            ),
            "row_capacity_envelope_count": len(constraints["row_cover"]),
            "row_conflict_clique_count": clique_count,
            "row_conflict_hall_capacity_count": clique_capacity_count,
        }

    def _initialize_master_incumbent(
        self, model, deadline: float
    ) -> dict:
        """Build a feasible skeleton, then polish it in the true objective."""
        started = perf_counter()
        variables = tuple(model.getVars())
        objectives = tuple(
            model.getVarObjective(variable) for variable in variables
        )
        remaining = deadline - perf_counter()
        allowance = min(
            float(self.benders_config.master_feasibility_time_limit),
            remaining,
        )
        result = {
            "attempted": allowance > 1e-6,
            "feasible": False,
            "status": "not_solved",
            "seconds": 0.0,
            "feasibility_seconds": 0.0,
            "polish_seconds": 0.0,
        }
        objectives_restored = False
        try:
            if allowance <= 1e-6:
                result["status"] = "time_limit_before_solve"
                return result
            for variable in variables:
                model.setVarObjective(variable, 0.0)
            model.update()
            feasibility_allowance = min(
                10.0, max(5.0, 0.4 * allowance), allowance
            )
            self._set_gurobi_param(
                model, "TimeLimit", max(0.01, feasibility_allowance)
            )
            self._set_gurobi_param(model, "SolutionLimit", 1)
            phase_started = perf_counter()
            model.optimize()
            result["feasibility_seconds"] = round(
                perf_counter() - phase_started, 3
            )
            result["status"] = self._gurobi_status_name(model)
            result["feasible"] = self._gurobi_solution_count(model) > 0
            if result["feasible"]:
                for variable in variables:
                    variable.Start = self._gurobi_value(model, variable)
            for variable, objective in zip(variables, objectives):
                model.setVarObjective(variable, objective)
            model.update()
            objectives_restored = True
            polish_allowance = min(
                max(0.0, allowance - (perf_counter() - started)),
                max(0.0, deadline - perf_counter()),
            )
            if result["feasible"] and polish_allowance > 1e-6:
                self._set_gurobi_param(
                    model, "TimeLimit", max(0.01, polish_allowance)
                )
                self._set_gurobi_param(model, "SolutionLimit", 5)
                phase_started = perf_counter()
                model.optimize()
                result["polish_seconds"] = round(
                    perf_counter() - phase_started, 3
                )
                result["status"] = self._gurobi_status_name(model)
                result["feasible"] = (
                    self._gurobi_solution_count(model) > 0
                )
                if result["feasible"]:
                    for variable in variables:
                        variable.Start = self._gurobi_value(
                            model, variable
                        )
            result["solution_count"] = self._gurobi_solution_count(model)
            result["objective"] = (
                self._gurobi_objective_value(model)
                if result["feasible"]
                else None
            )
            return result
        finally:
            if not objectives_restored:
                for variable, objective in zip(variables, objectives):
                    model.setVarObjective(variable, objective)
                model.update()
            self._try_set_gurobi_param(model, "SolutionLimit", 2_000_000_000)
            result["seconds"] = round(perf_counter() - started, 3)

    def _voyage_validation_reserve(self, remaining: float) -> float:
        """Reserve enough wall time to build and solve every voyage MIP."""
        estimate = (
            2.0
            + 0.00012 * len(self._columns)
            + 0.75 * len(self._voyages)
        )
        return min(
            max(0.0, float(remaining) - 0.01),
            min(20.0, max(5.0, estimate)),
        )

    def _master_search_allowance(
        self, iteration: int, remaining: float
    ) -> float:
        """Use one long initial tree; only post-cut solves use time slices."""
        reserve = self._voyage_validation_reserve(remaining)
        usable = max(0.01, float(remaining) - reserve)
        if int(iteration) == 1:
            return usable
        return min(float(self.benders_config.master_time_limit), usable)

    def _build_voyage_subproblem(
        self, voyage_id: str
    ) -> _VoyageSubproblem:
        from gurobipy import quicksum

        started = perf_counter()
        model = GurobiModel(
            f"voyage_row_subproblem_{self._key_name((voyage_id,))}"
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

        indices = tuple(
            index
            for index, column in enumerate(self._columns)
            if column.voyage_id == voyage_id
        )
        placement = {
            index: model.addVar(
                lb=0.0,
                ub=float(self._candidate_capacity[index]),
                vtype="I",
                name=f"x_{index}",
            )
            for index in indices
        }
        coefficient_rows: defaultdict[
            str, defaultdict[object, list[tuple[int, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        group_bay_indices: defaultdict[QuantityKey, list[int]] = (
            defaultdict(list)
        )
        group_row_indices: defaultdict[
            tuple[OperationalKey, str, str], list[int]
        ] = defaultdict(list)
        for index in indices:
            column = self._columns[index]
            group_bay_indices[(column.group_id, column.bay_key)].append(index)
            group_row_indices[
                (column.group_key, column.bay_key, self._anchor_row(column))
            ].append(index)
            for section, values in self._placement_master_coefficients(
                column
            ).items():
                for key, coefficient in values.items():
                    if coefficient:
                        coefficient_rows[section][key].append(
                            (index, float(coefficient))
                        )

        quantity_balance = {}
        for key in sorted(group_bay_indices):
            quantity_balance[key] = model.addConstr(
                quicksum(
                    placement[index] for index in group_bay_indices[key]
                )
                == 0.0,
                name=(
                    f"quantity_{key[0]}_{self._key_name((key[1],))}"
                ),
            )

        owner_limits = {}
        for index, variable in placement.items():
            owner_limits[index] = model.addConstr(
                variable <= 0.0,
                name=f"owner_limit_{index}",
            )

        for row_key, items in sorted(
            coefficient_rows["row_capacity_limit"].items()
        ):
            bay_key, row_no = row_key
            model.addConstr(
                quicksum(
                    coefficient * placement[index]
                    for index, coefficient in items
                )
                <= int(
                    self.bays[bay_key].row_physical_capacity.get(
                        row_no, self.bays[bay_key].physical_capacity
                    )
                ),
                name=f"row_capacity_{self._key_name(row_key)}",
            )
        for row_size, items in sorted(
            coefficient_rows["row_size_limit"].items()
        ):
            bay_key, row_no, size = row_size
            model.addConstr(
                quicksum(
                    coefficient * placement[index]
                    for index, coefficient in items
                )
                <= int(
                    self.bays[bay_key].row_cap_by_size.get(size, {}).get(
                        row_no,
                        self.bays[bay_key].cap_by_size.get(size, 0),
                    )
                ),
                name=f"row_size_{self._key_name(row_size)}",
            )

        uses_by_scope: defaultdict[
            tuple[str, str, str, str], list
        ] = defaultdict(list)
        for attr_key, items in sorted(
            coefficient_rows["row_attr_link"].items()
        ):
            bay_key, row_no, attr, scope, _value = attr_key
            use = model.addVar(
                vtype="B", name=f"row_attr_{self._key_name(attr_key)}"
            )
            model.addConstr(
                quicksum(
                    coefficient * placement[index]
                    for index, coefficient in items
                )
                <= self._master_row_attr_big_m[attr_key] * use,
                name=f"row_attr_link_{self._key_name(attr_key)}",
            )
            uses_by_scope[(bay_key, row_no, attr, scope)].append(use)
        for scope_key, uses in sorted(uses_by_scope.items()):
            model.addConstr(
                quicksum(uses) <= 1.0,
                name=f"row_attr_one_{self._key_name(scope_key)}",
            )

        for row_key, row_indices in sorted(group_row_indices.items()):
            operational_key, bay_key, row_no = row_key
            group_ids = self._operational_groups[operational_key]
            relevant_demand = sum(
                self.group_demand[group_id] for group_id in group_ids
            )
            upper = min(
                int(relevant_demand),
                sum(self._candidate_capacity[index] for index in row_indices),
            )
            use = model.addVar(
                vtype="B",
                obj=self._row_activation_penalty(),
                name=(
                    f"use_row_{self._key_name(operational_key)}_"
                    f"{self._key_name((bay_key, row_no))}"
                ),
            )
            assigned = quicksum(
                placement[index] for index in row_indices
            )
            model.addConstr(
                assigned <= max(1, upper) * use,
                name=(
                    f"use_row_upper_{self._key_name((*operational_key, bay_key, row_no))}"
                ),
            )
            model.addConstr(
                use <= assigned,
                name=(
                    f"use_row_lower_{self._key_name((*operational_key, bay_key, row_no))}"
                ),
            )

        model.update()
        return _VoyageSubproblem(
            voyage_id=voyage_id,
            model=model,
            placement_variables=placement,
            quantity_balance=quantity_balance,
            owner_limits=owner_limits,
            build_seconds=perf_counter() - started,
            variable_count=len(model.getVars()),
        )

    def _solve_voyage_subproblem(
        self,
        voyage_id: str,
        quantities: dict[QuantityKey, int],
        active_owners: set[OwnerKey],
        routing_hint: dict[int, float],
        time_limit: float,
    ) -> dict:
        subproblem = self._voyage_subproblems.get(voyage_id)
        if subproblem is None:
            subproblem = self._build_voyage_subproblem(voyage_id)
            self._voyage_subproblems[voyage_id] = subproblem
        for key, constraint in subproblem.quantity_balance.items():
            self._set_constraint_rhs(
                constraint, int(quantities.get(key, 0))
            )
        for index, constraint in subproblem.owner_limits.items():
            self._set_constraint_rhs(
                constraint,
                self._candidate_capacity[index]
                if self._owner_key_by_candidate[index] in active_owners
                else 0,
            )
            subproblem.placement_variables[index].Start = float(
                routing_hint.get(index, 0.0)
            )
        subproblem.model.update()
        self._set_gurobi_param(
            subproblem.model, "TimeLimit", max(0.01, float(time_limit))
        )
        started = perf_counter()
        subproblem.model.optimize()
        elapsed = perf_counter() - started
        status = self._gurobi_status_name(subproblem.model)
        result = {
            "voyage_id": voyage_id,
            "status": status,
            "optimal": status == "optimal",
            "feasible": self._gurobi_solution_count(subproblem.model) > 0,
            "seconds": elapsed,
            "selected": None,
            "objective": math.inf,
            "bound": -math.inf,
            "conflict_quantity_keys": (),
        }
        if not result["feasible"]:
            if status == "infeasible":
                try:
                    subproblem.model._model.computeIIS()
                    result["conflict_quantity_keys"] = tuple(
                        sorted(
                            key
                            for key, constraint
                            in subproblem.quantity_balance.items()
                            if int(quantities.get(key, 0)) > 0
                            and bool(constraint.IISConstr)
                        )
                    )
                except Exception:
                    pass
            return result
        result["selected"] = Counter(
            {
                index: int(
                    round(self._gurobi_value(subproblem.model, variable))
                )
                for index, variable in subproblem.placement_variables.items()
                if self._gurobi_value(subproblem.model, variable) > 0.5
            }
        )
        result["objective"] = self._gurobi_objective_value(
            subproblem.model
        )
        result["bound"] = self._gurobi_dual_bound(subproblem.model)
        return result

    def _master_assignment(self, model, variables: dict) -> tuple:
        quantities = {
            key: int(round(self._gurobi_value(model, variable)))
            for key, variable in variables["quantity"].items()
            if self._gurobi_value(model, variable) > 0.5
        }
        active_owners = {
            owner_key
            for owner_key, variable in variables["owner"].items()
            if self._gurobi_value(model, variable) > 0.5
        }
        imports = Counter(
            {
                key: int(round(self._gurobi_value(model, variable)))
                for key, variable in variables["import_reserve"].items()
                if self._gurobi_value(model, variable) > 0.5
            }
        )
        theta = {
            voyage_id: float(self._gurobi_value(model, variable))
            for voyage_id, variable in variables["theta"].items()
        }
        routing_hint = {
            index: float(self._gurobi_value(model, variable))
            for index, variable in variables["routing_flow"].items()
            if self._gurobi_value(model, variable) > 1e-8
        }
        return quantities, active_owners, imports, theta, routing_hint

    @staticmethod
    def _voyage_assignment_key(
        voyage_id: str,
        quantities: dict[QuantityKey, int],
        active_owners: set[OwnerKey],
    ) -> tuple:
        return (
            voyage_id,
            tuple(sorted(quantities.items())),
            tuple(sorted(active_owners)),
        )

    def _add_hall_capacity_cut(
        self,
        master,
        variables: dict,
        voyage_id: str,
        core_keys: tuple[QuantityKey, ...],
        quantities: dict[QuantityKey, int],
        active_owners: set[OwnerKey],
        signatures: set[tuple],
        cut_index: int,
    ) -> bool:
        from gurobipy import quicksum

        keys = tuple(
            sorted(
                key
                for key in core_keys
                if int(quantities.get(key, 0)) > 0
            )
        )
        if not keys:
            return False
        capacity_by_owner: dict[OwnerKey, int] = {}
        for key in keys:
            for index in self._candidate_indices_by_group_bay.get(key, ()):
                owner_key = self._owner_key_by_candidate[index]
                capacity_by_owner[owner_key] = max(
                    capacity_by_owner.get(owner_key, 0),
                    self._candidate_capacity[index],
                )
        signature = (voyage_id, keys)
        if signature in signatures or not capacity_by_owner:
            return False
        incumbent_load = sum(int(quantities[key]) for key in keys)
        incumbent_capacity = sum(
            capacity
            for owner_key, capacity in capacity_by_owner.items()
            if owner_key in active_owners
        )
        if incumbent_load <= incumbent_capacity:
            return False
        master.addConstr(
            quicksum(variables["quantity"][key] for key in keys)
            <= quicksum(
                capacity * variables["owner"][owner_key]
                for owner_key, capacity in capacity_by_owner.items()
            ),
            name=(
                f"voyage_hall_{cut_index}_"
                f"{self._key_name((voyage_id,))}"
            ),
        )
        signatures.add(signature)
        master.update()
        return True

    def _add_conditional_cut(
        self,
        master,
        variables: dict,
        voyage_id: str,
        quantities: dict[QuantityKey, int],
        active_owners: set[OwnerKey],
        cut_index: int,
        kind: str,
        lower_bound: float = 0.0,
        core_keys: tuple[QuantityKey, ...] = (),
    ) -> int:
        from gurobipy import quicksum

        relevant_keys = tuple(
            sorted(
                key
                for key, value in quantities.items()
                if int(value) > 0
                and self.groups_by_id[key[0]].voyage_id == voyage_id
                and (not core_keys or key in core_keys)
            )
        )
        if not relevant_keys:
            relevant_keys = tuple(
                sorted(
                    key
                    for key, value in quantities.items()
                    if int(value) > 0
                    and self.groups_by_id[key[0]].voyage_id == voyage_id
                )
            )
        decreases = []
        for position, key in enumerate(relevant_keys):
            incumbent = int(quantities[key])
            upper = int(self._quantity_upper[key])
            decrease = master.addVar(
                vtype="B",
                name=(
                    f"{kind}_decrease_{cut_index}_{position}_"
                    f"{self._key_name((voyage_id, key[0], key[1]))}"
                ),
            )
            master.addConstr(
                variables["quantity"][key]
                >= incumbent - upper * decrease,
                name=(
                    f"{kind}_decrease_lb_{cut_index}_{position}_"
                    f"{self._key_name((voyage_id,))}"
                ),
            )
            master.addConstr(
                variables["quantity"][key]
                <= incumbent - 1 + upper * (1 - decrease),
                name=(
                    f"{kind}_decrease_ub_{cut_index}_{position}_"
                    f"{self._key_name((voyage_id,))}"
                ),
            )
            decreases.append(decrease)

        relevant_bays = {key[1] for key in relevant_keys}
        relevant_owners = {
            owner_key
            for bay_key in relevant_bays
            for template_id in self._template_ids_by_voyage_bay.get(
                (voyage_id, bay_key), ()
            )
            for owner_key in self._owner_keys_by_template[template_id]
        }
        new_owner_terms = [
            variables["owner"][owner_key]
            for owner_key in sorted(relevant_owners - active_owners)
        ]
        change = quicksum(decreases) + quicksum(new_owner_terms)
        if kind == "feasibility":
            master.addConstr(
                change >= 1.0,
                name=(
                    f"voyage_logic_feasibility_{cut_index}_"
                    f"{self._key_name((voyage_id,))}"
                ),
            )
        elif kind == "optimality":
            value = max(0.0, float(lower_bound))
            master.addConstr(
                variables["theta"][voyage_id] >= value * (1 - change),
                name=(
                    f"voyage_logic_optimality_{cut_index}_"
                    f"{self._key_name((voyage_id,))}"
                ),
            )
        else:
            raise ValueError(f"unknown LBBD cut kind: {kind}")
        master.update()
        return len(decreases)

    def _apply_master_start(
        self,
        variables: dict,
        selected: Counter[int],
        imports: Counter[tuple[str, str, str]],
    ) -> None:
        q_values: Counter[QuantityKey] = Counter()
        owner_values: set[OwnerKey] = set()
        row_values: defaultdict[
            tuple[OperationalKey, str], set[str]
        ] = defaultdict(set)
        area_values: set[tuple[OperationalKey, str]] = set()
        for index, multiplier in selected.items():
            if int(multiplier) <= 0:
                continue
            column = self._columns[index]
            q_values[(column.group_id, column.bay_key)] += int(multiplier)
            owner_values.add(self._owner_key_by_candidate[index])
            row_values[(column.group_key, column.bay_key)].add(
                self._anchor_row(column)
            )
            area_values.add((column.group_key, column.area_no))
        for key, variable in variables["quantity"].items():
            variable.Start = float(q_values.get(key, 0))
        for owner_key, variable in variables["owner"].items():
            variable.Start = 1.0 if owner_key in owner_values else 0.0
        for index, variable in variables["routing_flow"].items():
            variable.Start = float(selected.get(index, 0))
        for key, variable in variables["row_count"].items():
            variable.Start = float(len(row_values.get(key, ())))
        for key, variable in variables["area_use"].items():
            variable.Start = 1.0 if key in area_values else 0.0
        for key, variable in variables["import_reserve"].items():
            variable.Start = float(imports.get(key, 0))

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
            "algorithm": "voyage_row_resource_logic_benders_gurobi",
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
            "formulation": (
                "group_bay_quantity_and_voyage_row_class_resource_master"
            ),
            "decomposition": "exact_row_allocation_by_export_voyage",
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

    def solve(self) -> ColumnGenerationResult:
        """Run voyage-row-resource LBBD without a global repair model."""
        started = perf_counter()
        total_limit = float(self.config.total_time_limit)
        if total_limit <= 0.0:
            total_limit = max(2.0, 2.0 * float(self.config.mip_time_limit))
        deadline = started + total_limit
        self._prepare_decomposition()
        preparation_seconds = perf_counter() - started
        if perf_counter() >= deadline:
            raise RuntimeError(
                "LBBD preprocessing consumed the complete time limit"
            )
        master, variables, master_stats = self._build_master()
        master_build_seconds = (
            perf_counter() - started - preparation_seconds
        )
        master_feasibility = self._initialize_master_incumbent(
            master, deadline
        )

        cache: dict[tuple, dict] = {}
        hall_signatures: set[tuple] = set()
        iteration_rows: list[dict] = []
        best_selected: Counter[int] | None = None
        best_import: Counter[tuple[str, str, str]] | None = None
        best_objective = math.inf
        best_iteration = 0
        valid_lower_bound = -math.inf
        master_status = "not_solved"
        feasibility_cuts = 0
        hall_capacity_cuts = 0
        optimality_cuts = 0
        conditional_binaries = 0
        subproblem_solves = 0
        subproblem_cache_hits = 0
        master_recovery_count = 0
        master_recovery_seconds = 0.0
        converged = False
        termination_reason = "iteration_limit"

        try:
            for iteration in range(
                1, int(self.benders_config.max_iterations) + 1
            ):
                remaining = deadline - perf_counter()
                if remaining <= 1e-6:
                    termination_reason = "time_limit"
                    break
                master_allowance = self._master_search_allowance(
                    iteration, remaining
                )
                self._set_gurobi_param(
                    master,
                    "TimeLimit",
                    max(0.01, min(master_allowance, remaining)),
                )
                master_started = perf_counter()
                master.optimize()
                master_seconds = perf_counter() - master_started
                master_status = self._gurobi_status_name(master)
                if self._gurobi_solution_count(master) <= 0:
                    recovery = self._initialize_master_incumbent(
                        master, deadline
                    )
                    master_recovery_count += 1
                    master_recovery_seconds += float(recovery["seconds"])
                    remaining = deadline - perf_counter()
                    if recovery["feasible"] and remaining > 1e-6:
                        self._set_gurobi_param(
                            master,
                            "TimeLimit",
                            max(0.01, min(2.0, remaining)),
                        )
                        self._set_gurobi_param(master, "SolutionLimit", 1)
                        recovery_started = perf_counter()
                        master.optimize()
                        master_seconds += (
                            perf_counter() - recovery_started
                        )
                        self._try_set_gurobi_param(
                            master, "SolutionLimit", 2_000_000_000
                        )
                        master_status = self._gurobi_status_name(master)
                    if self._gurobi_solution_count(master) <= 0:
                        termination_reason = (
                            "master_infeasible"
                            if master_status == "infeasible"
                            else "master_without_incumbent"
                        )
                        break
                master_bound = self._gurobi_dual_bound(master)
                if math.isfinite(master_bound):
                    valid_lower_bound = max(
                        valid_lower_bound, master_bound
                    )
                master_objective = self._gurobi_objective_value(master)
                (
                    quantities,
                    active_owners,
                    imports,
                    theta,
                    routing_hint,
                ) = (
                    self._master_assignment(master, variables)
                )

                selected: Counter[int] = Counter()
                subproblem_objective = 0.0
                voyage_records = []
                all_feasible = True
                all_optimal = True
                incomplete = False
                cuts_this_iteration = 0
                for voyage_position, voyage_id in enumerate(self._voyages):
                    remaining = deadline - perf_counter()
                    if remaining <= 1e-6:
                        all_feasible = False
                        all_optimal = False
                        incomplete = True
                        termination_reason = "time_limit"
                        break
                    voyage_quantities = {
                        key: value
                        for key, value in quantities.items()
                        if self.groups_by_id[key[0]].voyage_id == voyage_id
                    }
                    voyage_owners = {
                        owner_key
                        for owner_key in active_owners
                        if self._templates[owner_key[0]].voyage_id
                        == voyage_id
                    }
                    voyage_routing_hint = {
                        index: value
                        for index, value in routing_hint.items()
                        if self._columns[index].voyage_id == voyage_id
                    }
                    assignment_key = self._voyage_assignment_key(
                        voyage_id,
                        voyage_quantities,
                        voyage_owners,
                    )
                    voyage_result = cache.get(assignment_key)
                    cached = voyage_result is not None
                    if cached:
                        subproblem_cache_hits += 1
                    else:
                        unsolved = len(self._voyages) - voyage_position
                        fair_share = remaining / max(1, unsolved)
                        voyage_result = self._solve_voyage_subproblem(
                            voyage_id,
                            voyage_quantities,
                            voyage_owners,
                            voyage_routing_hint,
                            min(
                                float(
                                    self.benders_config.voyage_time_limit
                                ),
                                fair_share,
                            ),
                        )
                        subproblem_solves += 1
                        if voyage_result["optimal"] or (
                            voyage_result["status"] == "infeasible"
                        ):
                            cache[assignment_key] = voyage_result

                    voyage_records.append(
                        {
                            "voyage_id": voyage_id,
                            "status": voyage_result["status"],
                            "cached": cached,
                            "seconds": round(
                                float(voyage_result["seconds"]), 4
                            ),
                            "objective": (
                                None
                                if not math.isfinite(
                                    voyage_result["objective"]
                                )
                                else float(voyage_result["objective"])
                            ),
                            "bound": (
                                None
                                if not math.isfinite(voyage_result["bound"])
                                else float(voyage_result["bound"])
                            ),
                            "conflict_quantity_count": len(
                                voyage_result.get(
                                    "conflict_quantity_keys", ()
                                )
                            ),
                        }
                    )
                    if voyage_result["status"] == "infeasible":
                        all_feasible = False
                        all_optimal = False
                        core_keys = tuple(
                            voyage_result.get(
                                "conflict_quantity_keys", ()
                            )
                        ) or tuple(sorted(voyage_quantities))
                        if self._add_hall_capacity_cut(
                            master,
                            variables,
                            voyage_id,
                            core_keys,
                            voyage_quantities,
                            voyage_owners,
                            hall_signatures,
                            hall_capacity_cuts + 1,
                        ):
                            hall_capacity_cuts += 1
                        else:
                            conditional_binaries += (
                                self._add_conditional_cut(
                                    master,
                                    variables,
                                    voyage_id,
                                    voyage_quantities,
                                    voyage_owners,
                                    feasibility_cuts + 1,
                                    "feasibility",
                                    core_keys=core_keys,
                                )
                            )
                            feasibility_cuts += 1
                        cuts_this_iteration += 1
                        continue
                    if not voyage_result["feasible"]:
                        all_feasible = False
                        all_optimal = False
                        incomplete = True
                        termination_reason = (
                            "voyage_subproblem_without_incumbent"
                        )
                        continue

                    selected.update(voyage_result["selected"])
                    subproblem_objective += float(
                        voyage_result["objective"]
                    )
                    if not voyage_result["optimal"]:
                        all_optimal = False
                    cut_value = (
                        float(voyage_result["objective"])
                        if voyage_result["optimal"]
                        else float(voyage_result["bound"])
                    )
                    if (
                        math.isfinite(cut_value)
                        and cut_value
                        > float(theta.get(voyage_id, 0.0)) + 1e-7
                    ):
                        conditional_binaries += self._add_conditional_cut(
                            master,
                            variables,
                            voyage_id,
                            voyage_quantities,
                            voyage_owners,
                            optimality_cuts + 1,
                            "optimality",
                            lower_bound=cut_value,
                        )
                        optimality_cuts += 1
                        cuts_this_iteration += 1

                candidate_objective = None
                decomposition_objective = None
                decomposition_auxiliary_slack = None
                if all_feasible and len(voyage_records) == len(self._voyages):
                    self._final_import_reservation = Counter(imports)
                    candidate_objective = self._selected_solution_energy(
                        selected
                    )
                    decomposition_objective = (
                        master_objective
                        - sum(theta.values())
                        + subproblem_objective
                    )
                    decomposition_auxiliary_slack = (
                        self._absolute_deviation_auxiliary_slack(
                            decomposition_objective,
                            candidate_objective,
                            context="voyage-resource LBBD decomposition",
                        )
                    )
                    if candidate_objective + 1e-9 < best_objective:
                        best_selected = Counter(selected)
                        best_import = Counter(imports)
                        best_objective = candidate_objective
                        best_iteration = iteration
                        self._apply_master_start(
                            variables, best_selected, best_import
                        )
                        master.update()

                iteration_rows.append(
                    {
                        "iteration": iteration,
                        "master_status": master_status,
                        "master_allowance": round(master_allowance, 4),
                        "master_seconds": round(master_seconds, 4),
                        "master_objective": master_objective,
                        "master_bound": master_bound,
                        "master_bound_is_global": True,
                        "active_row_class_count": len(active_owners),
                        "cuts_added": cuts_this_iteration,
                        "all_voyage_subproblems_feasible": all_feasible,
                        "all_voyage_subproblems_optimal": all_optimal,
                        "candidate_objective": candidate_objective,
                        "decomposition_objective": decomposition_objective,
                        "decomposition_auxiliary_slack": (
                            decomposition_auxiliary_slack
                        ),
                        "voyage_subproblems": voyage_records,
                    }
                )
                if incomplete and cuts_this_iteration == 0:
                    break
                if incomplete:
                    continue
                if (
                    all_feasible
                    and all_optimal
                    and cuts_this_iteration == 0
                    and master_status == "optimal"
                ):
                    converged = True
                    termination_reason = "optimality_proven"
                    break
                if cuts_this_iteration == 0 and master_status != "optimal":
                    termination_reason = (
                        "continuous_master_limit_with_verified_incumbent"
                    )
                    break
            else:
                termination_reason = "iteration_limit"

            if best_selected is None or best_import is None:
                last_with_voyages = next(
                    (
                        row
                        for row in reversed(iteration_rows)
                        if row.get("voyage_subproblems")
                    ),
                    {},
                )
                last_voyage_statuses = (
                    [
                        (
                            row["voyage_id"],
                            row["status"],
                        )
                        for row in last_with_voyages.get(
                            "voyage_subproblems", []
                        )
                    ]
                    if last_with_voyages
                    else []
                )
                raise RuntimeError(
                    "voyage-resource LBBD did not find a complete row "
                    "allocation; "
                    f"termination={termination_reason}, "
                    f"master_status={master_status}, "
                    f"iterations={len(iteration_rows)}, "
                    f"subproblem_solves={subproblem_solves}, "
                    f"master_feasibility={master_feasibility}, "
                    f"hall_cuts={hall_capacity_cuts}, "
                    f"feasibility_cuts={feasibility_cuts}, "
                    f"last_voyage_statuses={last_voyage_statuses}"
                )
            self._final_import_reservation = best_import
            if converged:
                valid_lower_bound = best_objective
            if not math.isfinite(valid_lower_bound):
                valid_lower_bound = 0.0
            valid_lower_bound = min(valid_lower_bound, best_objective)
            absolute_gap = max(0.0, best_objective - valid_lower_bound)
            relative_gap = absolute_gap / max(abs(best_objective), 1e-12)
            subproblem_build_seconds = sum(
                item.build_seconds
                for item in self._voyage_subproblems.values()
            )
            diagnostics = {
                **self._base_diagnostics(),
                "master_algorithm": "voyage_row_resource_lbbd",
                "master_status": "optimal" if converged else master_status,
                "master_bound_scope": "valid_voyage_resource_master",
                "master_objective": best_objective,
                "master_mip_gap": relative_gap,
                "complete_model_lower_bound": valid_lower_bound,
                "complete_model_absolute_gap": absolute_gap,
                "complete_model_relative_gap": relative_gap,
                "complete_model_gap_source": "voyage_resource_master_bound",
                "hard_demand_balance": True,
                "candidate_row_location_count": len(self._columns),
                "selected_location_count": len(best_selected),
                "lbbd_converged": converged,
                "lbbd_termination_reason": termination_reason,
                "lbbd_iteration_count": len(iteration_rows),
                "lbbd_best_iteration": best_iteration,
                "lbbd_best_incumbent_source": "exact_voyage_subproblems",
                "lbbd_iterations": iteration_rows,
                "lbbd_cut_counts": {
                    "initial_row_conflict_clique": master_stats[
                        "row_conflict_clique_count"
                    ],
                    "dynamic_hall_capacity": hall_capacity_cuts,
                    "logic_feasibility": feasibility_cuts,
                    "logic_optimality": optimality_cuts,
                    "conditional_binary_variables": conditional_binaries,
                },
                "lbbd_voyage_subproblem_solve_count": subproblem_solves,
                "lbbd_voyage_subproblem_cache_hits": subproblem_cache_hits,
                "lbbd_voyage_subproblem_count": len(self._voyages),
                "lbbd_voyage_subproblem_build_seconds": round(
                    subproblem_build_seconds, 3
                ),
                "lbbd_voyage_subproblem_variable_count": sum(
                    item.variable_count
                    for item in self._voyage_subproblems.values()
                ),
                "lbbd_preparation_seconds": round(
                    preparation_seconds, 3
                ),
                "lbbd_master_build_seconds": round(
                    master_build_seconds, 3
                ),
                "lbbd_master_feasibility": master_feasibility,
                "lbbd_master_recovery_count": master_recovery_count,
                "lbbd_master_recovery_seconds": round(
                    master_recovery_seconds, 3
                ),
                "lbbd_total_solve_seconds": round(
                    perf_counter() - started, 3
                ),
                "lbbd_master": master_stats,
            }
            result = self._assemble_result(best_selected, diagnostics)
            result.columns = self._selected_lbbd_columns(best_selected)
            return result
        finally:
            self._free_gurobi_model(master)
            for subproblem in self._voyage_subproblems.values():
                self._free_gurobi_model(subproblem.model)


LogicBendersPlanner = VoyageResourceBendersPlanner


__all__ = [
    "LogicBendersConfig",
    "LogicBendersPlanner",
    "VoyageResourceBendersPlanner",
]
