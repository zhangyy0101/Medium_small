"""Row-profile aggregation with exact physical-footprint disaggregation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from time import perf_counter

from .gurobi_backend import GurobiModel
from .planner import ColumnGenerationResult
from .voyage_resource_benders import (
    LogicBendersConfig,
    OperationalKey,
    OwnerKey,
    PhysicalSlot,
    QuantityKey,
    VoyageResourceBendersPlanner,
)


ProfileState = tuple[int, str]
ProfileRoute = tuple[QuantityKey, int]


@dataclass(frozen=True)
class _ResourceProfile:
    profile_id: int
    voyage_id: str
    bay_key: str
    slot_bays: tuple[str, ...]
    template_ids: tuple[int, ...]
    multiplicity: int


@dataclass
class _GlobalProfileSubproblem:
    """Exact row disaggregation for one fixed aggregate master solution."""

    model: GurobiModel
    placement_variables: dict[int, object]
    owner_variables: dict[OwnerKey, object]
    quantity_balance: dict[QuantityKey, object]
    profile_balance: dict[ProfileState, object]
    build_seconds: float
    variable_count: int


class ProfileResourceBendersPlanner(VoyageResourceBendersPlanner):
    """LBBD over exchangeable row profiles plus exact footprint matching."""

    def __init__(self, problem, config=None, benders_config=None) -> None:
        super().__init__(problem, config, benders_config)
        self._profiles: dict[int, _ResourceProfile] = {}
        self._profile_by_template: dict[int, int] = {}
        self._profile_states: tuple[ProfileState, ...] = ()
        self._profile_ids_by_voyage_bay: dict[
            tuple[str, str], tuple[int, ...]
        ] = {}
        self._profile_ids_by_slot_bay: dict[str, tuple[int, ...]] = {}
        self._profile_ids_by_physical_pool: dict[
            tuple[tuple[PhysicalSlot, ...], ...], tuple[int, ...]
        ] = {}
        self._overlap_pool_bounds: tuple[
            tuple[tuple[int, ...], int], ...
        ] = ()
        self._profile_route_capacity: dict[ProfileRoute, int] = {}
        self._profile_routes_by_quantity: dict[
            QuantityKey, tuple[ProfileRoute, ...]
        ] = {}
        self._profile_routes_by_profile: dict[
            int, tuple[ProfileRoute, ...]
        ] = {}
        self._candidate_by_quantity_template: dict[
            tuple[QuantityKey, int], int
        ] = {}
        self._global_profile_subproblem: _GlobalProfileSubproblem | None = (
            None
        )

    @staticmethod
    def _interval_packing_bound(
        footprints: set[tuple[PhysicalSlot, ...]],
    ) -> int:
        """Return a safe row-footprint packing bound.

        Real yard footprints are intervals along one physical row.  For that
        case the earliest-finish interval schedule is exact.  The fallback
        uses the number of physical slots, which is always a valid upper
        bound and keeps unusual input layouts safe.
        """
        if not footprints:
            return 0
        slots = tuple(sorted({slot for item in footprints for slot in item}))
        position = {slot: index for index, slot in enumerate(slots)}
        intervals = []
        for footprint in footprints:
            indices = sorted(position[slot] for slot in footprint)
            if indices != list(range(indices[0], indices[-1] + 1)):
                return len(slots)
            intervals.append((indices[0], indices[-1]))
        selected = 0
        last_end = -1
        for start, end in sorted(set(intervals), key=lambda item: item[1]):
            if start > last_end:
                selected += 1
                last_end = end
        return selected

    def _prepare_overlap_pool_bounds(self) -> None:
        """Build Hall-style bounds for partially overlapping profile pools."""
        option_slots = {
            profile_id: frozenset(
                self._templates[template_id].slots
                for template_id in profile.template_ids
            )
            for profile_id, profile in self._profiles.items()
        }
        profiles_by_slot: defaultdict[PhysicalSlot, set[int]] = (
            defaultdict(set)
        )
        slots_by_profile = {}
        for profile_id, footprints in option_slots.items():
            slots = {slot for footprint in footprints for slot in footprint}
            slots_by_profile[profile_id] = slots
            for slot in slots:
                profiles_by_slot[slot].add(profile_id)

        unseen = set(option_slots)
        components = []
        while unseen:
            root = unseen.pop()
            component = {root}
            frontier = [root]
            while frontier:
                profile_id = frontier.pop()
                neighbours = {
                    neighbour
                    for slot in slots_by_profile[profile_id]
                    for neighbour in profiles_by_slot[slot]
                }
                neighbours &= unseen
                unseen -= neighbours
                component.update(neighbours)
                frontier.extend(neighbours)
            components.append(tuple(sorted(component)))

        components = sorted(components)
        exact_pool_bounds = {
            tuple(sorted(profile_ids)): len(pool_signature)
            for pool_signature, profile_ids
            in self._profile_ids_by_physical_pool.items()
            if len(profile_ids) > 1
        }
        overlap_bounds = []
        for profile_ids in components:
            if len(profile_ids) <= 1:
                continue
            footprints = {
                footprint
                for profile_id in profile_ids
                for footprint in option_slots[profile_id]
            }
            upper = self._interval_packing_bound(footprints)
            trivial = sum(
                self._profiles[profile_id].multiplicity
                for profile_id in profile_ids
            )
            if upper >= trivial:
                continue
            exact_upper = exact_pool_bounds.get(profile_ids)
            if exact_upper is not None and upper >= exact_upper:
                continue
            overlap_bounds.append((profile_ids, int(upper)))
        self._overlap_pool_bounds = tuple(overlap_bounds)

    def _profile_signature(self, template_id: int) -> tuple:
        template = self._templates[template_id]
        slot_profiles = []
        for bay_key, row_no in template.slots:
            bay = self.bays[bay_key]
            slot_profiles.append(
                (
                    bay_key,
                    int(
                        bay.row_physical_capacity.get(
                            row_no, bay.physical_capacity
                        )
                    ),
                    tuple(
                        sorted(
                            (
                                str(size),
                                int(
                                    values.get(
                                        row_no,
                                        bay.cap_by_size.get(size, 0),
                                    )
                                ),
                            )
                            for size, values in bay.row_cap_by_size.items()
                        )
                    ),
                )
            )
        candidate_profile = []
        for index in template.candidate_indices:
            column = self._columns[index]
            group = self.groups_by_id[column.group_id]
            candidate_profile.append(
                (
                    group.group_id,
                    group.size,
                    group.height,
                    self._row_mix_key_for_group(group),
                    int(self._candidate_capacity[index]),
                )
            )
        return (
            template.voyage_id,
            template.bay_key,
            tuple(slot_profiles),
            tuple(sorted(candidate_profile)),
        )

    def _prepare_profiles(self) -> None:
        by_signature: defaultdict[tuple, list[int]] = defaultdict(list)
        for template_id in self._templates:
            by_signature[self._profile_signature(template_id)].append(
                template_id
            )
        profiles = {}
        profile_by_template = {}
        by_voyage_bay: defaultdict[tuple[str, str], list[int]] = (
            defaultdict(list)
        )
        by_slot_bay: defaultdict[str, list[int]] = defaultdict(list)
        for profile_id, signature in enumerate(
            sorted(by_signature), start=1
        ):
            voyage_id, bay_key, slot_profiles, _candidate_profile = (
                signature
            )
            template_ids = tuple(sorted(by_signature[signature]))
            profile = _ResourceProfile(
                profile_id=profile_id,
                voyage_id=voyage_id,
                bay_key=bay_key,
                slot_bays=tuple(item[0] for item in slot_profiles),
                template_ids=template_ids,
                multiplicity=len(template_ids),
            )
            profiles[profile_id] = profile
            by_voyage_bay[(voyage_id, bay_key)].append(profile_id)
            for slot_bay in set(profile.slot_bays):
                by_slot_bay[slot_bay].append(profile_id)
            for template_id in template_ids:
                profile_by_template[template_id] = profile_id
        self._profiles = profiles
        self._profile_by_template = profile_by_template
        self._profile_ids_by_voyage_bay = {
            key: tuple(values)
            for key, values in sorted(by_voyage_bay.items())
        }
        self._profile_ids_by_slot_bay = {
            key: tuple(values)
            for key, values in sorted(by_slot_bay.items())
        }
        by_physical_pool: defaultdict[
            tuple[tuple[PhysicalSlot, ...], ...], list[int]
        ] = defaultdict(list)
        for profile_id, profile in profiles.items():
            pool_signature = tuple(
                sorted(
                    self._templates[template_id].slots
                    for template_id in profile.template_ids
                )
            )
            by_physical_pool[pool_signature].append(profile_id)
        self._profile_ids_by_physical_pool = {
            signature: tuple(sorted(profile_ids))
            for signature, profile_ids in sorted(
                by_physical_pool.items(), key=lambda item: repr(item[0])
            )
        }
        self._prepare_overlap_pool_bounds()

        profile_states = {
            (
                profile_by_template[owner_key[0]],
                owner_key[1],
            )
            for owner_key in self._owner_key_by_candidate.values()
        }
        self._profile_states = tuple(sorted(profile_states))

        route_capacity: dict[ProfileRoute, int] = {}
        by_quantity: defaultdict[QuantityKey, set[ProfileRoute]] = (
            defaultdict(set)
        )
        by_profile: defaultdict[int, set[ProfileRoute]] = defaultdict(set)
        candidate_by_quantity_template = {}
        for quantity_key, indices in (
            self._candidate_indices_by_group_bay.items()
        ):
            for index in indices:
                template_id = self._template_by_candidate[index]
                profile_id = profile_by_template[template_id]
                route = (quantity_key, profile_id)
                route_capacity[route] = max(
                    route_capacity.get(route, 0),
                    int(self._candidate_capacity[index]),
                )
                by_quantity[quantity_key].add(route)
                by_profile[profile_id].add(route)
                candidate_by_quantity_template[
                    (quantity_key, template_id)
                ] = index
        self._profile_route_capacity = route_capacity
        self._profile_routes_by_quantity = {
            key: tuple(sorted(routes))
            for key, routes in sorted(by_quantity.items())
        }
        self._profile_routes_by_profile = {
            profile_id: tuple(sorted(routes))
            for profile_id, routes in sorted(by_profile.items())
        }
        self._candidate_by_quantity_template = (
            candidate_by_quantity_template
        )

    def _build_profile_master(self):
        from gurobipy import quicksum

        model = GurobiModel("profile_resource_lbbd_master")
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
                    self._columns[
                        self._quantity_representative[key]
                    ].intrinsic_cost
                ),
                name=f"q_bay_{key[0]}_{self._key_name((key[1],))}",
            )
            for key in sorted(self._candidate_indices_by_group_bay)
        }
        profile_use = {
            state: model.addVar(
                lb=0.0,
                ub=float(self._profiles[state[0]].multiplicity),
                vtype="I",
                name=(
                    f"profile_use_{state[0]}_"
                    f"{self._key_name((state[1],))}"
                ),
            )
            for state in self._profile_states
        }
        routing_flow = {
            route: model.addVar(
                lb=0.0,
                ub=float(
                    self._profile_route_capacity[route]
                    * self._profiles[route[1]].multiplicity
                ),
                name=(
                    f"profile_flow_{route[0][0]}_"
                    f"{self._key_name((route[0][1], str(route[1])))}"
                ),
            )
            for route in sorted(self._profile_route_capacity)
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
            keys = [key for key in quantity if key[0] == group.group_id]
            if not keys:
                raise ValueError(
                    "profile LBBD group has no feasible bay: "
                    f"group={group.group_id}"
                )
            constraints["group_demand"][group.group_id] = model.addConstr(
                quicksum(quantity[key] for key in keys)
                == int(group.demand),
                name=f"demand_{group.group_id}",
            )

        for quantity_key, variable in sorted(quantity.items()):
            routes = self._profile_routes_by_quantity[quantity_key]
            constraints["routing_balance"][quantity_key] = model.addConstr(
                quicksum(routing_flow[route] for route in routes)
                == variable,
                name=(
                    f"profile_balance_{quantity_key[0]}_"
                    f"{self._key_name((quantity_key[1],))}"
                ),
            )
            row_class = self._row_mix_key_for_group(
                self.groups_by_id[quantity_key[0]]
            )
            for route in routes:
                state = (route[1], row_class)
                constraints["routing_owner_link"][route] = model.addConstr(
                    routing_flow[route]
                    <= self._profile_route_capacity[route]
                    * profile_use[state],
                    name=(
                        f"profile_owner_link_{quantity_key[0]}_"
                        f"{self._key_name((quantity_key[1], str(route[1])))}"
                    ),
                )

        states_by_profile: defaultdict[int, list[ProfileState]] = (
            defaultdict(list)
        )
        for state in profile_use:
            states_by_profile[state[0]].append(state)
        for profile_id, profile in sorted(self._profiles.items()):
            states = states_by_profile[profile_id]
            constraints["profile_multiplicity"][profile_id] = (
                model.addConstr(
                    quicksum(profile_use[state] for state in states)
                    <= int(profile.multiplicity),
                    name=f"profile_multiplicity_{profile_id}",
                )
            )
            for state in states:
                state_routes = [
                    route
                    for route in self._profile_routes_by_profile[profile_id]
                    if self._row_mix_key_for_group(
                        self.groups_by_id[route[0][0]]
                    )
                    == state[1]
                ]
                state_capacity = max(
                    self._profile_route_capacity[route]
                    for route in state_routes
                )
                constraints["profile_state_capacity"][state] = (
                    model.addConstr(
                        quicksum(
                            routing_flow[route] for route in state_routes
                        )
                        <= state_capacity * profile_use[state],
                        name=(
                            f"profile_state_capacity_{profile_id}_"
                            f"{self._key_name((state[1],))}"
                        ),
                    )
                )
                constraints["profile_positive_flow"][state] = (
                    model.addConstr(
                        profile_use[state]
                        <= quicksum(
                            routing_flow[route] for route in state_routes
                        ),
                        name=(
                            f"profile_positive_flow_{profile_id}_"
                            f"{self._key_name((state[1],))}"
                        ),
                    )
                )

        shared_pool_count = 0
        for pool_index, (pool_signature, profile_ids) in enumerate(
            self._profile_ids_by_physical_pool.items(), start=1
        ):
            states = [
                state
                for profile_id in profile_ids
                for state in states_by_profile[profile_id]
            ]
            if len(profile_ids) <= 1:
                continue
            constraints["shared_physical_pool"][pool_index] = (
                model.addConstr(
                    quicksum(profile_use[state] for state in states)
                    <= len(pool_signature),
                    name=f"shared_physical_pool_{pool_index}",
                )
            )
            shared_pool_count += 1

        overlap_pool_count = 0
        for pool_index, (profile_ids, upper) in enumerate(
            self._overlap_pool_bounds, start=1
        ):
            constraints["overlap_physical_pool"][pool_index] = (
                model.addConstr(
                    quicksum(
                        profile_use[state]
                        for profile_id in profile_ids
                        for state in states_by_profile[profile_id]
                    )
                    <= int(upper),
                    name=f"overlap_physical_pool_{pool_index}",
                )
            )
            overlap_pool_count += 1

        physical_slot_count = Counter(
            slot[0] for slot in self._template_ids_by_slot
        )
        for slot_bay, profile_ids in sorted(
            self._profile_ids_by_slot_bay.items()
        ):
            constraints["aggregate_physical_rows"][slot_bay] = (
                model.addConstr(
                    quicksum(
                        profile.slot_bays.count(slot_bay)
                        * profile_use[state]
                        for profile_id in profile_ids
                        for profile in (self._profiles[profile_id],)
                        for state in states_by_profile[profile_id]
                    )
                    <= int(physical_slot_count[slot_bay]),
                    name=(
                        f"aggregate_physical_rows_"
                        f"{self._key_name((slot_bay,))}"
                    ),
                )
            )

        for voyage_bay, profile_ids in sorted(
            self._profile_ids_by_voyage_bay.items()
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
                    profile_use[state]
                    for profile_id in profile_ids
                    for state in states_by_profile[profile_id]
                )
                <= quicksum(
                    row_count[(operational_key, bay_key)]
                    for operational_key in operational_keys
                ),
                name=f"profile_owner_count_{self._key_name(voyage_bay)}",
            )

        for key, group_ids in sorted(self._operational_bay_groups.items()):
            operational_key, bay_key = key
            assigned = quicksum(
                quantity[(group_id, bay_key)]
                for group_id in group_ids
                if (group_id, bay_key) in quantity
            )
            compatible_states = sorted(
                {
                    (
                        self._profile_by_template[
                            self._template_by_candidate[index]
                        ],
                        self._owner_key_by_candidate[index][1],
                    )
                    for group_id in group_ids
                    for index in self._candidate_indices_by_group_bay.get(
                        (group_id, bay_key), ()
                    )
                }
            )
            constraints["row_count_box_upper"][key] = model.addConstr(
                row_count[key] <= assigned,
                name=(
                    f"profile_row_box_{self._key_name((*operational_key, bay_key))}"
                ),
            )
            constraints["row_count_owner_upper"][key] = model.addConstr(
                row_count[key]
                <= quicksum(
                    profile_use[state] for state in compatible_states
                ),
                name=(
                    f"profile_row_owner_{self._key_name((*operational_key, bay_key))}"
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
                            f"profile_row_cover_{self._key_name(operational_key)}_"
                            f"{self._key_name((bay_key, str(breakpoint)))}"
                        ),
                    )
                prefix_capacity += int(slope)
                previous_slope = int(slope)

        states_by_class: defaultdict[
            tuple[str, str, str], list[ProfileState]
        ] = defaultdict(list)
        for state in profile_use:
            profile = self._profiles[state[0]]
            states_by_class[
                (profile.voyage_id, profile.bay_key, state[1])
            ].append(state)
        for class_key, states in sorted(states_by_class.items()):
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
                    quicksum(profile_use[state] for state in states)
                    <= quicksum(
                        row_count[(operational_key, bay_key)]
                        for operational_key in compatible_operational_keys
                    ),
                    name=(
                        f"profile_class_cover_{self._key_name(class_key)}"
                    ),
                )
            )

        clique_count = 0
        clique_capacity_count = 0
        for voyage_bay, cliques in sorted(
            self._row_cliques_by_voyage_bay.items()
        ):
            voyage_id, bay_key = voyage_bay
            for clique_index, clique in enumerate(cliques):
                states = sorted(
                    {
                        (
                            self._profile_by_template[
                                self._template_by_candidate[index]
                            ],
                            self._owner_key_by_candidate[index][1],
                        )
                        for operational_key in clique
                        for group_id in self._operational_bay_groups.get(
                            (operational_key, bay_key), ()
                        )
                        for index in self._candidate_indices_by_group_bay.get(
                            (group_id, bay_key), ()
                        )
                    }
                )
                if not states:
                    continue
                cut_key = (voyage_id, bay_key, clique_index)
                constraints["row_conflict_clique"][cut_key] = (
                    model.addConstr(
                        quicksum(
                            row_count[(operational_key, bay_key)]
                            for operational_key in clique
                        )
                        <= quicksum(
                            profile_use[state] for state in states
                        ),
                        name=(
                            f"profile_clique_"
                            f"{self._key_name((voyage_id, bay_key, str(clique_index)))}"
                        ),
                    )
                )
                clique_count += 1
                clique_group_ids = {
                    group_id
                    for operational_key in clique
                    for group_id in self._operational_bay_groups.get(
                        (operational_key, bay_key), ()
                    )
                }
                quantity_keys = [
                    (group_id, bay_key)
                    for group_id in sorted(clique_group_ids)
                    if (group_id, bay_key) in quantity
                ]
                capacity_by_state = {}
                for state in states:
                    capacity_by_state[state] = max(
                        (
                            self._profile_route_capacity[
                                (quantity_key, state[0])
                            ]
                            for quantity_key in quantity_keys
                            if (quantity_key, state[0])
                            in self._profile_route_capacity
                            and self._row_mix_key_for_group(
                                self.groups_by_id[quantity_key[0]]
                            )
                            == state[1]
                        ),
                        default=0,
                    )
                constraints["row_conflict_hall_capacity"][cut_key] = (
                    model.addConstr(
                        quicksum(
                            quantity[quantity_key]
                            for quantity_key in quantity_keys
                        )
                        <= quicksum(
                            capacity * profile_use[state]
                            for state, capacity
                            in capacity_by_state.items()
                        ),
                        name=(
                            f"profile_clique_capacity_"
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
        import_by_bay_size: defaultdict[tuple[str, str], list] = (
            defaultdict(list)
        )
        import_by_flow_size: defaultdict[tuple[str, str], list] = (
            defaultdict(list)
        )
        import_by_flow_area_size: defaultdict[
            tuple[str, str, str], list
        ] = defaultdict(list)
        for (flow, size, bay_key), variable in import_reserve.items():
            area_no = self.bays[bay_key].area_no
            for footprint_key in self._placement_footprint_keys(
                bay_key, size
            ):
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
                name=f"profile_bay_capacity_{self._key_name((bay_key,))}",
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
                name=f"profile_bay_size_{self._key_name(bay_size)}",
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
                name=f"profile_stack_{self._key_name(stack_key)}",
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
                name=f"profile_stack_load_{self._key_name(stack_key)}",
            )
            stack_variables_by_bay_size[(bay_key, size)].append(stack)
        for bay_size, stacks in stack_variables_by_bay_size.items():
            constraints["stack_total"][bay_size] = model.addConstr(
                quicksum(stacks)
                <= self._stack_count_for_bay_size(*bay_size),
                name=f"profile_stack_total_{self._key_name(bay_size)}",
            )

        for key, required in sorted(self.import_total_by_flow_size.items()):
            candidates = import_by_flow_size.get(key, [])
            if not candidates:
                raise ValueError(
                    "profile LBBD import demand has no compatible bay: "
                    f"flow={key[0]}, size={key[1]}"
                )
            constraints["import_total"][key] = model.addConstr(
                quicksum(candidates) == int(required),
                name=f"profile_import_total_{self._key_name(key)}",
            )

        for guidance_key in sorted(self._master_area_guidance_keys):
            target = self._area_size_target(*guidance_key)
            positive = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"profile_guide_pos_{self._key_name(guidance_key)}",
            )
            negative = model.addVar(
                lb=0.0,
                obj=self._area_guidance_penalty(),
                name=f"profile_guide_neg_{self._key_name(guidance_key)}",
            )
            items = q_coefficients["area_guidance_balance"].get(
                guidance_key, []
            )
            constraints["export_guidance"][guidance_key] = (
                model.addConstr(
                    quicksum(
                        coefficient * quantity[key]
                        for key, coefficient in items
                    )
                    - target
                    == positive - negative,
                    name=(
                        f"profile_guidance_{self._key_name(guidance_key)}"
                    ),
                )
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
                    f"profile_area_upper_"
                    f"{self._key_name((*operational_key, area_no))}"
                ),
            )
            constraints["area_use_lower"][key] = model.addConstr(
                use <= assigned,
                name=(
                    f"profile_area_lower_"
                    f"{self._key_name((*operational_key, area_no))}"
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
                name=f"profile_theta_{self._key_name((voyage_id,))}",
            )

        constraints["import_reference"] = self._add_import_reference_deviation(
            quicksum,
            model,
            import_by_flow_area_size,
            objective_mode="full",
        )
        constraints.update(
            self._add_bay_compatibility_constraints(
                quicksum,
                model,
                quantity,
                q_coefficients["bay_attr_link"],
                relax=False,
            )
        )

        offset = -len(self._operational_groups) * (
            self._area_activation_penalty()
            + self._row_activation_penalty()
        )
        model.addVar(
            lb=1.0,
            ub=1.0,
            obj=offset,
            name="profile_lbbd_objective_offset",
        )
        model.update()
        variables = {
            "quantity": quantity,
            "profile_use": profile_use,
            "routing_flow": routing_flow,
            "row_count": row_count,
            "area_use": area_use,
            "theta": theta,
            "import_reserve": import_reserve,
            "all_master_variables": tuple(model.getVars()),
        }
        stats = {
            "master_variable_count": len(model.getVars()),
            "group_bay_quantity_count": len(quantity),
            "concrete_row_template_count": len(self._templates),
            "row_resource_profile_count": len(self._profiles),
            "profile_owner_integer_count": len(profile_use),
            "profile_routing_flow_count": len(routing_flow),
            "operational_bay_row_count_count": len(row_count),
            "aggregate_physical_row_constraint_count": len(
                constraints["aggregate_physical_rows"]
            ),
            "shared_physical_pool_constraint_count": shared_pool_count,
            "overlap_physical_pool_constraint_count": overlap_pool_count,
            "profile_state_capacity_constraint_count": len(
                constraints["profile_state_capacity"]
            ),
            "profile_positive_flow_constraint_count": len(
                constraints["profile_positive_flow"]
            ),
            "row_capacity_envelope_count": len(constraints["row_cover"]),
            "row_conflict_clique_count": clique_count,
            "row_conflict_hall_capacity_count": clique_capacity_count,
        }
        return model, variables, stats

    def _profile_assignment_from_values(
        self, variables: dict, value_of
    ) -> tuple:
        def positive_values(items, tolerance: float) -> dict:
            values = {}
            for key, variable in items:
                value = float(value_of(variable))
                if value > tolerance:
                    values[key] = value
            return values

        quantities = {
            key: int(round(value))
            for key, value in positive_values(
                variables["quantity"].items(), 0.5
            ).items()
        }
        profile_counts = {
            state: int(round(value))
            for state, value in positive_values(
                variables["profile_use"].items(), 0.5
            ).items()
        }
        routing = positive_values(
            variables["routing_flow"].items(), 1e-8
        )
        imports = Counter(
            {
                key: int(round(value))
                for key, value in positive_values(
                    variables["import_reserve"].items(), 0.5
                ).items()
            }
        )
        theta = {
            voyage_id: float(value_of(variable))
            for voyage_id, variable in variables["theta"].items()
        }
        return quantities, profile_counts, routing, imports, theta

    def _profile_master_assignment(self, model, variables: dict) -> tuple:
        return self._profile_assignment_from_values(
            variables,
            lambda variable: self._gurobi_value(model, variable),
        )

    @staticmethod
    def _start_value(variable) -> float:
        value = float(variable.Start)
        return value if math.isfinite(value) and abs(value) < 1e50 else 0.0

    def _profile_master_start_assignment(self, variables: dict) -> tuple:
        """Read the saved zero-objective skeleton after Gurobi resets X."""
        return self._profile_assignment_from_values(
            variables, self._start_value
        )

    def _owner_options_for_state(
        self, state: ProfileState
    ) -> tuple[OwnerKey, ...]:
        profile_id, row_class = state
        return tuple(
            sorted(
                (template_id, row_class)
                for template_id in self._profiles[
                    profile_id
                ].template_ids
                if (template_id, row_class)
                in self._owner_keys_by_template[template_id]
            )
        )

    def _solve_footprint_matching(
        self,
        profile_counts: dict[ProfileState, int],
        time_limit: float,
    ) -> dict:
        from gurobipy import quicksum

        started = perf_counter()
        model = GurobiModel("profile_footprint_disaggregation")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        self._set_gurobi_param(model, "MIPGap", 0.0)
        self._set_gurobi_param(model, "DualReductions", 0)
        self._set_gurobi_param(
            model, "TimeLimit", max(0.01, float(time_limit))
        )
        model.setMinimize()
        owner = {
            owner_key: model.addVar(
                vtype="B",
                name=(
                    f"match_{owner_key[0]}_"
                    f"{self._key_name((owner_key[1],))}"
                ),
            )
            for state, count in sorted(profile_counts.items())
            if int(count) > 0
            for owner_key in self._owner_options_for_state(state)
        }
        count_constraints = {}
        for state, count in sorted(profile_counts.items()):
            if int(count) <= 0:
                continue
            options = self._owner_options_for_state(state)
            count_constraints[state] = model.addConstr(
                quicksum(owner[owner_key] for owner_key in options)
                == int(count),
                name=(
                    f"match_count_{state[0]}_"
                    f"{self._key_name((state[1],))}"
                ),
            )
        by_template: defaultdict[int, list] = defaultdict(list)
        by_slot: defaultdict[PhysicalSlot, list] = defaultdict(list)
        for owner_key, variable in owner.items():
            template = self._templates[owner_key[0]]
            by_template[owner_key[0]].append(variable)
            for slot in template.slots:
                by_slot[slot].append(variable)
        for template_id, values in sorted(by_template.items()):
            model.addConstr(
                quicksum(values) <= 1.0,
                name=f"match_template_{template_id}",
            )
        for slot, values in sorted(by_slot.items()):
            model.addConstr(
                quicksum(values) <= 1.0,
                name=f"match_slot_{self._key_name(slot)}",
            )
        model.update()
        model.optimize()
        status = self._gurobi_status_name(model)
        feasible = self._gurobi_solution_count(model) > 0
        active_owners = {
            owner_key
            for owner_key, variable in owner.items()
            if feasible and self._gurobi_value(model, variable) > 0.5
        }
        core_states: tuple[ProfileState, ...] = ()
        if not feasible and status == "infeasible":
            try:
                model._model.computeIIS()
                core_states = tuple(
                    sorted(
                        state
                        for state, constraint
                        in count_constraints.items()
                        if bool(constraint.IISConstr)
                    )
                )
            except Exception:
                core_states = ()
        result = {
            "status": status,
            "feasible": feasible,
            "optimal": status == "optimal",
            "active_owners": active_owners,
            "core_states": core_states,
            "seconds": perf_counter() - started,
            "variable_count": len(owner),
        }
        self._free_gurobi_model(model)
        return result

    def _profile_packing_upper_bound(
        self,
        states: tuple[ProfileState, ...],
        time_limit: float,
    ) -> tuple[int | None, dict]:
        from gurobipy import quicksum

        started = perf_counter()
        model = GurobiModel("profile_conflict_core_packing")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        self._set_gurobi_param(model, "MIPGap", 0.0)
        self._set_gurobi_param(
            model, "TimeLimit", max(0.01, float(time_limit))
        )
        model.setMinimize()
        owner = {
            owner_key: model.addVar(
                vtype="B", obj=-1.0, name=f"pack_{owner_key[0]}"
            )
            for state in states
            for owner_key in self._owner_options_for_state(state)
        }
        by_template: defaultdict[int, list] = defaultdict(list)
        by_slot: defaultdict[PhysicalSlot, list] = defaultdict(list)
        for owner_key, variable in owner.items():
            template = self._templates[owner_key[0]]
            by_template[owner_key[0]].append(variable)
            for slot in template.slots:
                by_slot[slot].append(variable)
        for values in by_template.values():
            model.addConstr(quicksum(values) <= 1.0)
        for values in by_slot.values():
            model.addConstr(quicksum(values) <= 1.0)
        model.update()
        model.optimize()
        bound = self._gurobi_dual_bound(model)
        upper = (
            int(math.floor(-bound + 1e-7))
            if math.isfinite(bound)
            else None
        )
        record = {
            "status": self._gurobi_status_name(model),
            "seconds": perf_counter() - started,
            "variable_count": len(owner),
            "upper_bound": upper,
        }
        self._free_gurobi_model(model)
        return upper, record

    def _add_profile_packing_cut(
        self,
        master,
        variables: dict,
        states: tuple[ProfileState, ...],
        profile_counts: dict[ProfileState, int],
        signatures: set[tuple[ProfileState, ...]],
        cut_index: int,
        time_limit: float,
    ) -> tuple[bool, dict]:
        from gurobipy import quicksum

        states = tuple(sorted(states or tuple(profile_counts)))
        if not states or states in signatures:
            return False, {}
        upper, record = self._profile_packing_upper_bound(
            states, time_limit
        )
        incumbent = sum(int(profile_counts.get(state, 0)) for state in states)
        if upper is None or incumbent <= int(upper):
            return False, record
        master.addConstr(
            quicksum(
                variables["profile_use"][state] for state in states
            )
            <= int(upper),
            name=f"profile_packing_cut_{cut_index}",
        )
        master.update()
        signatures.add(states)
        record["incumbent_count"] = incumbent
        return True, record

    def _concrete_routing_hint(
        self,
        routing: dict[ProfileRoute, float],
        active_owners: set[OwnerKey],
    ) -> dict[int, float]:
        hint: Counter[int] = Counter()
        remaining_by_template = {
            owner_key[0]: max(
                self._candidate_capacity[index]
                for index in self._templates[
                    owner_key[0]
                ].candidate_indices
            )
            for owner_key in active_owners
        }
        for route, value in sorted(routing.items()):
            quantity_key, profile_id = route
            row_class = self._row_mix_key_for_group(
                self.groups_by_id[quantity_key[0]]
            )
            remaining = float(value)
            owners = sorted(
                owner_key
                for owner_key in active_owners
                if self._profile_by_template[owner_key[0]] == profile_id
                and owner_key[1] == row_class
            )
            for owner_key in owners:
                if remaining <= 1e-8:
                    break
                template_id = owner_key[0]
                index = self._candidate_by_quantity_template.get(
                    (quantity_key, template_id)
                )
                if index is None:
                    continue
                amount = min(
                    remaining,
                    float(self._candidate_capacity[index]),
                    float(remaining_by_template[template_id]),
                )
                if amount <= 1e-8:
                    continue
                hint[index] += amount
                remaining_by_template[template_id] -= amount
                remaining -= amount
        return dict(hint)

    def _apply_profile_master_start(
        self,
        variables: dict,
        selected: Counter[int],
        imports: Counter[tuple[str, str, str]],
    ) -> None:
        for variable in variables.get("all_master_variables", ()):
            variable.Start = 1e101
        q_values: Counter[QuantityKey] = Counter()
        route_values: Counter[ProfileRoute] = Counter()
        used_templates_by_state: defaultdict[ProfileState, set[int]] = (
            defaultdict(set)
        )
        row_values: defaultdict[
            tuple[OperationalKey, str], set[str]
        ] = defaultdict(set)
        area_values: set[tuple[OperationalKey, str]] = set()
        for index, multiplier in selected.items():
            if int(multiplier) <= 0:
                continue
            column = self._columns[index]
            quantity_key = (column.group_id, column.bay_key)
            profile_id = self._profile_by_template[
                self._template_by_candidate[index]
            ]
            state = (profile_id, self._owner_key_by_candidate[index][1])
            q_values[quantity_key] += int(multiplier)
            route_values[(quantity_key, profile_id)] += int(multiplier)
            used_templates_by_state[state].add(
                self._template_by_candidate[index]
            )
            row_values[(column.group_key, column.bay_key)].add(
                self._anchor_row(column)
            )
            area_values.add((column.group_key, column.area_no))
        for key, variable in variables["quantity"].items():
            variable.Start = float(q_values.get(key, 0))
        for state, variable in variables["profile_use"].items():
            variable.Start = float(
                len(used_templates_by_state.get(state, ()))
            )
        for route, variable in variables["routing_flow"].items():
            variable.Start = float(route_values.get(route, 0))
        for key, variable in variables["row_count"].items():
            variable.Start = float(len(row_values.get(key, ())))
        for key, variable in variables["area_use"].items():
            variable.Start = 1.0 if key in area_values else 0.0
        rows_by_voyage: Counter[str] = Counter()
        for (operational_key, _bay_key), rows in row_values.items():
            voyage_id = self._representative_group[
                operational_key
            ].voyage_id
            rows_by_voyage[voyage_id] += len(rows)
        for voyage_id, variable in variables["theta"].items():
            variable.Start = (
                self._row_activation_penalty()
                * rows_by_voyage.get(voyage_id, 0)
            )
        for key, variable in variables["import_reserve"].items():
            variable.Start = float(imports.get(key, 0))

    def _verify_profile_assignment(
        self,
        quantities: dict[QuantityKey, int],
        profile_counts: dict[ProfileState, int],
        routing: dict[ProfileRoute, float],
        imports: Counter[tuple[str, str, str]],
        deadline: float,
        cache: dict[tuple, dict],
    ) -> dict:
        """Match profiles and solve every exact voyage row subproblem."""
        remaining = deadline - perf_counter()
        matching = self._solve_footprint_matching(
            profile_counts, min(5.0, max(0.01, remaining))
        )
        result = {
            "verified": False,
            "matching": matching,
            "selected": Counter(),
            "imports": Counter(imports),
            "objective": None,
            "voyage_records": [],
            "voyage_solve_count": 0,
            "voyage_cache_hits": 0,
            "subproblem_objective": 0.0,
            "all_optimal": False,
        }
        if not matching["feasible"]:
            return result

        active_owners = set(matching["active_owners"])
        concrete_hint = self._concrete_routing_hint(
            routing, active_owners
        )
        selected: Counter[int] = Counter()
        all_optimal = True
        for voyage_position, voyage_id in enumerate(self._voyages):
            remaining = deadline - perf_counter()
            if remaining <= 1e-6:
                result["voyage_records"].append(
                    {
                        "voyage_id": voyage_id,
                        "status": "time_limit_before_solve",
                        "cached": False,
                        "seconds": 0.0,
                    }
                )
                return result
            voyage_quantities = {
                key: value
                for key, value in quantities.items()
                if self.groups_by_id[key[0]].voyage_id == voyage_id
            }
            voyage_owners = {
                owner_key
                for owner_key in active_owners
                if self._templates[owner_key[0]].voyage_id == voyage_id
            }
            voyage_hint = {
                index: value
                for index, value in concrete_hint.items()
                if self._columns[index].voyage_id == voyage_id
            }
            assignment_key = self._voyage_assignment_key(
                voyage_id, voyage_quantities, voyage_owners
            )
            voyage_result = cache.get(assignment_key)
            cached = voyage_result is not None
            if cached:
                result["voyage_cache_hits"] += 1
            else:
                fair_share = remaining / max(
                    1, len(self._voyages) - voyage_position
                )
                voyage_result = self._solve_voyage_subproblem(
                    voyage_id,
                    voyage_quantities,
                    voyage_owners,
                    voyage_hint,
                    min(
                        float(self.benders_config.voyage_time_limit),
                        fair_share,
                    ),
                )
                result["voyage_solve_count"] += 1
                if voyage_result["optimal"]:
                    cache[assignment_key] = voyage_result
            result["voyage_records"].append(
                {
                    "voyage_id": voyage_id,
                    "status": voyage_result["status"],
                    "cached": cached,
                    "seconds": round(
                        float(voyage_result["seconds"]), 4
                    ),
                }
            )
            if not voyage_result["feasible"]:
                return result
            selected.update(voyage_result["selected"])
            result["subproblem_objective"] += float(
                voyage_result["objective"]
            )
            all_optimal = all_optimal and bool(voyage_result["optimal"])

        self._final_import_reservation = Counter(imports)
        result["selected"] = selected
        result["objective"] = self._selected_solution_energy(selected)
        result["all_optimal"] = all_optimal
        result["verified"] = True
        return result

    def _build_global_profile_subproblem(
        self,
    ) -> _GlobalProfileSubproblem:
        """Build the exact joint row model used to certify aggregate cuts.

        The fast verifier fixes one arbitrary footprint matching and solves
        the voyages separately.  Failure of that particular matching is not
        a proof that the aggregate master point is infeasible.  This model
        chooses every concrete footprint and every row placement jointly,
        while fixing only the aggregate quantities and profile-state counts.
        It is therefore a valid feasibility and recourse oracle.
        """
        from gurobipy import quicksum

        started = perf_counter()
        model = GurobiModel("profile_exact_global_disaggregation")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        self._set_gurobi_param(model, "MIPGap", 0.0)
        self._set_gurobi_param(model, "DualReductions", 0)
        model.setMinimize()

        placement = {
            index: model.addVar(
                lb=0.0,
                ub=float(self._candidate_capacity[index]),
                vtype="I",
                name=f"global_x_{index}",
            )
            for index in range(len(self._columns))
        }
        owner_keys = tuple(sorted(set(self._owner_key_by_candidate.values())))
        owner = {
            owner_key: model.addVar(
                vtype="B",
                name=(
                    f"global_owner_{owner_key[0]}_"
                    f"{self._key_name((owner_key[1],))}"
                ),
            )
            for owner_key in owner_keys
        }

        coefficient_rows: defaultdict[
            str, defaultdict[object, list[tuple[int, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        indices_by_owner: defaultdict[OwnerKey, list[int]] = defaultdict(list)
        group_row_indices: defaultdict[
            tuple[OperationalKey, str, str], list[int]
        ] = defaultdict(list)
        for index, column in enumerate(self._columns):
            indices_by_owner[self._owner_key_by_candidate[index]].append(index)
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
        for key, indices in sorted(
            self._candidate_indices_by_group_bay.items()
        ):
            quantity_balance[key] = model.addConstr(
                quicksum(placement[index] for index in indices) == 0.0,
                name=(
                    f"global_quantity_{key[0]}_"
                    f"{self._key_name((key[1],))}"
                ),
            )

        profile_balance = {}
        for state in self._profile_states:
            options = self._owner_options_for_state(state)
            profile_balance[state] = model.addConstr(
                quicksum(owner[owner_key] for owner_key in options) == 0.0,
                name=(
                    f"global_profile_{state[0]}_"
                    f"{self._key_name((state[1],))}"
                ),
            )

        for owner_key, indices in sorted(indices_by_owner.items()):
            owner_variable = owner[owner_key]
            for index in indices:
                model.addConstr(
                    placement[index]
                    <= self._candidate_capacity[index] * owner_variable,
                    name=f"global_owner_link_{index}",
                )
            model.addConstr(
                owner_variable
                <= quicksum(placement[index] for index in indices),
                name=(
                    f"global_owner_positive_{owner_key[0]}_"
                    f"{self._key_name((owner_key[1],))}"
                ),
            )

        for template_id, owner_options in sorted(
            self._owner_keys_by_template.items()
        ):
            model.addConstr(
                quicksum(owner[key] for key in owner_options) <= 1.0,
                name=f"global_template_{template_id}",
            )
        for slot, template_ids in sorted(self._template_ids_by_slot.items()):
            model.addConstr(
                quicksum(
                    owner[owner_key]
                    for template_id in template_ids
                    for owner_key in self._owner_keys_by_template[
                        template_id
                    ]
                )
                <= 1.0,
                name=f"global_slot_{self._key_name(slot)}",
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
                name=f"global_row_capacity_{self._key_name(row_key)}",
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
                name=f"global_row_size_{self._key_name(row_size)}",
            )

        uses_by_scope: defaultdict[
            tuple[str, str, str, str], list
        ] = defaultdict(list)
        for attr_key, items in sorted(
            coefficient_rows["row_attr_link"].items()
        ):
            bay_key, row_no, attr, scope, _value = attr_key
            use = model.addVar(
                vtype="B",
                name=f"global_row_attr_{self._key_name(attr_key)}",
            )
            model.addConstr(
                quicksum(
                    coefficient * placement[index]
                    for index, coefficient in items
                )
                <= self._master_row_attr_big_m[attr_key] * use,
                name=f"global_row_attr_link_{self._key_name(attr_key)}",
            )
            uses_by_scope[(bay_key, row_no, attr, scope)].append(use)
        for scope_key, uses in sorted(uses_by_scope.items()):
            model.addConstr(
                quicksum(uses) <= 1.0,
                name=f"global_row_attr_one_{self._key_name(scope_key)}",
            )

        for row_key, indices in sorted(group_row_indices.items()):
            operational_key, bay_key, row_no = row_key
            group_ids = self._operational_groups[operational_key]
            upper = min(
                sum(self.group_demand[group_id] for group_id in group_ids),
                sum(self._candidate_capacity[index] for index in indices),
            )
            use = model.addVar(
                vtype="B",
                obj=self._row_activation_penalty(),
                name=(
                    f"global_use_row_{self._key_name(operational_key)}_"
                    f"{self._key_name((bay_key, row_no))}"
                ),
            )
            assigned = quicksum(placement[index] for index in indices)
            model.addConstr(
                assigned <= max(1, int(upper)) * use,
                name=(
                    f"global_use_row_upper_"
                    f"{self._key_name((*operational_key, bay_key, row_no))}"
                ),
            )
            model.addConstr(
                use <= assigned,
                name=(
                    f"global_use_row_lower_"
                    f"{self._key_name((*operational_key, bay_key, row_no))}"
                ),
            )

        model.update()
        return _GlobalProfileSubproblem(
            model=model,
            placement_variables=placement,
            owner_variables=owner,
            quantity_balance=quantity_balance,
            profile_balance=profile_balance,
            build_seconds=perf_counter() - started,
            variable_count=len(model.getVars()),
        )

    def _solve_global_profile_subproblem(
        self,
        quantities: dict[QuantityKey, int],
        profile_counts: dict[ProfileState, int],
        time_limit: float,
        selected_hint: Counter[int] | None = None,
    ) -> dict:
        """Certify a fixed aggregate point without fixing a footprint match."""
        call_started = perf_counter()
        if self._global_profile_subproblem is None:
            self._global_profile_subproblem = (
                self._build_global_profile_subproblem()
            )
        subproblem = self._global_profile_subproblem
        for key, constraint in subproblem.quantity_balance.items():
            self._set_constraint_rhs(constraint, quantities.get(key, 0))
        for state, constraint in subproblem.profile_balance.items():
            self._set_constraint_rhs(
                constraint, profile_counts.get(state, 0)
            )
        hint = selected_hint or Counter()
        active_hint = {
            self._owner_key_by_candidate[index]
            for index, value in hint.items()
            if int(value) > 0
        }
        for index, variable in subproblem.placement_variables.items():
            variable.Start = float(hint.get(index, 0))
        for owner_key, variable in subproblem.owner_variables.items():
            variable.Start = 1.0 if owner_key in active_hint else 0.0
        subproblem.model.update()
        solve_allowance = max(
            0.01,
            float(time_limit) - (perf_counter() - call_started),
        )
        self._set_gurobi_param(
            subproblem.model, "TimeLimit", solve_allowance
        )
        started = perf_counter()
        subproblem.model.optimize()
        elapsed = perf_counter() - started
        status = self._gurobi_status_name(subproblem.model)
        feasible = self._gurobi_solution_count(subproblem.model) > 0
        result = {
            "status": status,
            "feasible": feasible,
            "optimal": status == "optimal",
            "seconds": elapsed,
            "selected": Counter(),
            "objective": math.inf,
            "bound": self._gurobi_dual_bound(subproblem.model),
            "conflict_quantity_keys": (),
            "conflict_profile_states": (),
            "variable_count": subproblem.variable_count,
        }
        if feasible:
            result["selected"] = Counter(
                {
                    index: int(
                        round(
                            self._gurobi_value(subproblem.model, variable)
                        )
                    )
                    for index, variable
                    in subproblem.placement_variables.items()
                    if self._gurobi_value(subproblem.model, variable) > 0.5
                }
            )
            result["objective"] = self._gurobi_objective_value(
                subproblem.model
            )
            return result
        if status == "infeasible":
            try:
                subproblem.model._model.computeIIS()
                result["conflict_quantity_keys"] = tuple(
                    sorted(
                        key
                        for key, constraint
                        in subproblem.quantity_balance.items()
                        if bool(constraint.IISConstr)
                    )
                )
                result["conflict_profile_states"] = tuple(
                    sorted(
                        state
                        for state, constraint
                        in subproblem.profile_balance.items()
                        if bool(constraint.IISConstr)
                    )
                )
            except Exception:
                pass
        return result

    def _add_aggregate_logic_cut(
        self,
        master,
        variables: dict,
        quantities: dict[QuantityKey, int],
        profile_counts: dict[ProfileState, int],
        cut_index: int,
        kind: str,
        signatures: set[tuple],
        lower_bound: float = 0.0,
        core_quantity_keys: tuple[QuantityKey, ...] = (),
        core_profile_states: tuple[ProfileState, ...] = (),
    ) -> tuple[bool, int]:
        """Add an exact conditional cut in aggregate integer space."""
        from gurobipy import quicksum

        if kind not in {"feasibility", "optimality"}:
            raise ValueError(f"unknown aggregate logic cut kind: {kind}")
        if kind == "feasibility" and core_quantity_keys:
            quantity_keys = tuple(sorted(set(core_quantity_keys)))
            exact_quantity_directions = True
        else:
            # Fixing every positive allocation from the incumbent fixes the
            # complete group-to-bay vector: each group demand is an equality,
            # so activating a zero component necessarily decreases a positive
            # one.  This avoids thousands of zero-valued terms in an
            # optimality no-good cut.
            quantity_keys = tuple(
                sorted(
                    key
                    for key, value in quantities.items()
                    if int(value) > 0
                )
            )
            exact_quantity_directions = False
        states = tuple(
            sorted(
                core_profile_states
                if core_profile_states
                else self._profile_states
            )
        )
        signature = (
            kind,
            tuple(
                (key, int(quantities.get(key, 0)))
                for key in quantity_keys
            ),
            tuple(
                (state, int(profile_counts.get(state, 0)))
                for state in states
            ),
            round(float(lower_bound), 10) if kind == "optimality" else None,
        )
        if signature in signatures:
            return False, 0

        change_terms = []
        binary_count = 0
        for position, key in enumerate(quantity_keys):
            incumbent = int(quantities.get(key, 0))
            upper = int(self._quantity_upper[key])
            if incumbent <= 0:
                change_terms.append(variables["quantity"][key])
                continue
            decrease = master.addVar(
                vtype="B",
                name=f"aggregate_{kind}_{cut_index}_qdown_{position}",
            )
            master.addConstr(
                variables["quantity"][key]
                >= incumbent - upper * decrease,
                name=f"aggregate_{kind}_{cut_index}_qdown_lb_{position}",
            )
            master.addConstr(
                variables["quantity"][key]
                <= incumbent - 1 + upper * (1 - decrease),
                name=f"aggregate_{kind}_{cut_index}_qdown_ub_{position}",
            )
            change_terms.append(decrease)
            binary_count += 1
            if exact_quantity_directions and incumbent < upper:
                increase = master.addVar(
                    vtype="B",
                    name=(
                        f"aggregate_{kind}_{cut_index}_qup_{position}"
                    ),
                )
                master.addConstr(
                    variables["quantity"][key]
                    >= (incumbent + 1) * increase,
                    name=(
                        f"aggregate_{kind}_{cut_index}_qup_lb_{position}"
                    ),
                )
                master.addConstr(
                    variables["quantity"][key]
                    <= incumbent + (upper - incumbent) * increase,
                    name=(
                        f"aggregate_{kind}_{cut_index}_qup_ub_{position}"
                    ),
                )
                master.addConstr(
                    decrease + increase <= 1.0,
                    name=(
                        f"aggregate_{kind}_{cut_index}_q_direction_"
                        f"{position}"
                    ),
                )
                change_terms.append(increase)
                binary_count += 1

        for position, state in enumerate(states):
            incumbent = int(profile_counts.get(state, 0))
            upper = int(self._profiles[state[0]].multiplicity)
            variable = variables["profile_use"][state]
            if incumbent == 0:
                change_terms.append(variable)
                continue
            down = master.addVar(
                vtype="B",
                name=f"aggregate_{kind}_{cut_index}_pdown_{position}",
            )
            master.addConstr(
                variable >= incumbent * (1 - down),
                name=f"aggregate_{kind}_{cut_index}_pdown_lb_{position}",
            )
            master.addConstr(
                variable
                <= incumbent - 1 + (upper - incumbent + 1) * (1 - down),
                name=f"aggregate_{kind}_{cut_index}_pdown_ub_{position}",
            )
            change_terms.append(down)
            binary_count += 1
            if incumbent < upper:
                up = master.addVar(
                    vtype="B",
                    name=f"aggregate_{kind}_{cut_index}_pup_{position}",
                )
                master.addConstr(
                    variable >= (incumbent + 1) * up,
                    name=f"aggregate_{kind}_{cut_index}_pup_lb_{position}",
                )
                master.addConstr(
                    variable <= incumbent + (upper - incumbent) * up,
                    name=f"aggregate_{kind}_{cut_index}_pup_ub_{position}",
                )
                master.addConstr(
                    down + up <= 1.0,
                    name=f"aggregate_{kind}_{cut_index}_p_direction_{position}",
                )
                change_terms.append(up)
                binary_count += 1

        change = quicksum(change_terms)
        if kind == "feasibility":
            master.addConstr(
                change >= 1.0,
                name=f"aggregate_logic_feasibility_{cut_index}",
            )
        else:
            value = max(0.0, float(lower_bound))
            master.addConstr(
                quicksum(variables["theta"].values())
                >= value * (1 - change),
                name=f"aggregate_logic_optimality_{cut_index}",
            )
        master.update()
        signatures.add(signature)
        return True, binary_count

    def _voyage_validation_reserve(self, remaining: float) -> float:
        estimate = (
            2.0
            + 0.00008 * len(self._columns)
            + 0.50 * len(self._voyages)
        )
        return min(
            max(0.0, float(remaining) - 0.01),
            min(12.0, max(6.0, estimate)),
        )

    def _master_search_allowance(
        self, master_round: int, remaining: float
    ) -> float:
        reserve = self._voyage_validation_reserve(remaining)
        if int(master_round) == 1:
            return max(0.01, float(remaining) - reserve)
        return max(
            0.01,
            min(
                float(self.benders_config.master_time_limit),
                0.60 * float(remaining),
            ),
        )

    def _initialize_master_incumbent(self, model, deadline: float) -> dict:
        """Find one skeleton; exact disaggregation supplies the polished start."""
        started = perf_counter()
        variables = tuple(model.getVars())
        objectives = tuple(
            model.getVarObjective(variable) for variable in variables
        )
        allowance = min(
            float(self.benders_config.master_feasibility_time_limit),
            max(0.0, deadline - perf_counter()),
        )
        result = {
            "attempted": allowance > 1e-6,
            "feasible": False,
            "status": "not_solved",
            "seconds": 0.0,
            "feasibility_seconds": 0.0,
            "strategy": "feasibility_then_exact_disaggregation",
        }
        objectives_restored = False
        try:
            if allowance <= 1e-6:
                result["status"] = "time_limit_before_solve"
                return result
            for variable in variables:
                model.setVarObjective(variable, 0.0)
            model.update()
            feasibility_allowance = min(15.0, 0.55 * allowance, allowance)
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
            result["solution_count"] = self._gurobi_solution_count(model)
            if result["feasible"]:
                for variable in variables:
                    variable.Start = self._gurobi_value(model, variable)
            for variable, objective in zip(variables, objectives):
                model.setVarObjective(variable, objective)
            model.update()
            objectives_restored = True
            result["objective"] = None
            return result
        finally:
            if not objectives_restored:
                for variable, objective in zip(variables, objectives):
                    model.setVarObjective(variable, objective)
                model.update()
            self._try_set_gurobi_param(
                model, "SolutionLimit", 2_000_000_000
            )
            result["seconds"] = round(perf_counter() - started, 3)

    def _base_profile_diagnostics(self) -> dict:
        diagnostics = super()._base_diagnostics()
        diagnostics.update(
            {
                "algorithm": "row_profile_resource_logic_benders_gurobi",
                "formulation": (
                    "group_bay_quantity_and_aggregated_row_profile_master"
                ),
                "decomposition": (
                    "fast_matching_and_voyage_checks_with_exact_global_"
                    "disaggregation_oracle"
                ),
                "benders_cut_validity": (
                    "aggregate_IIS_logic_feasibility_and_conditional_"
                    "recourse_optimality"
                ),
            }
        )
        return diagnostics

    def solve(self) -> ColumnGenerationResult:
        """Solve the profile master and exactly disaggregate every incumbent."""
        started = perf_counter()
        total_limit = float(self.config.total_time_limit)
        if total_limit <= 0.0:
            total_limit = max(2.0, 2.0 * float(self.config.mip_time_limit))
        deadline = started + total_limit
        self._prepare_decomposition()
        self._prepare_profiles()
        preparation_seconds = perf_counter() - started
        if perf_counter() >= deadline:
            raise RuntimeError(
                "profile LBBD preprocessing consumed the complete time limit"
            )
        master, variables, master_stats = self._build_profile_master()
        master_build_seconds = (
            perf_counter() - started - preparation_seconds
        )
        master_feasibility = self._initialize_master_incumbent(
            master, deadline
        )

        best_selected: Counter[int] | None = None
        best_import: Counter[tuple[str, str, str]] | None = None
        best_objective = math.inf
        best_master_round = 0
        valid_lower_bound = -math.inf
        master_status = "not_solved"
        master_round_rows: list[dict] = []
        matching_cut_signatures: set[tuple[ProfileState, ...]] = set()
        matching_cuts = 0
        matching_solves = 0
        matching_seconds = 0.0
        matching_variables = 0
        packing_records = []
        aggregate_cut_signatures: set[tuple] = set()
        aggregate_feasibility_cuts = 0
        aggregate_optimality_cuts = 0
        aggregate_cut_binaries = 0
        global_subproblem_solves = 0
        global_subproblem_seconds = 0.0
        global_subproblem_records = []
        initial_disaggregation = []
        voyage_solves = 0
        voyage_cache_hits = 0
        cache: dict[tuple, dict] = {}
        termination_reason = "iteration_limit"
        converged = False

        try:
            for warmup_iteration in range(1, 6):
                has_live_solution = self._gurobi_solution_count(master) > 0
                has_saved_skeleton = bool(
                    warmup_iteration == 1
                    and master_feasibility.get("feasible", False)
                )
                if not has_live_solution and not has_saved_skeleton:
                    break
                if has_live_solution:
                    warm_assignment = self._profile_master_assignment(
                        master, variables
                    )
                else:
                    warm_assignment = (
                        self._profile_master_start_assignment(variables)
                    )
                (
                    warm_quantities,
                    warm_counts,
                    warm_routing,
                    warm_imports,
                    _warm_theta,
                ) = warm_assignment
                remaining = deadline - perf_counter()
                if remaining <= 1e-6:
                    break
                warm_validation = self._verify_profile_assignment(
                    warm_quantities,
                    warm_counts,
                    warm_routing,
                    warm_imports,
                    deadline,
                    cache,
                )
                warm_match = warm_validation["matching"]
                matching_solves += 1
                matching_seconds += float(warm_match["seconds"])
                matching_variables = max(
                    matching_variables,
                    int(warm_match["variable_count"]),
                )
                warm_record = {
                    "disaggregation_round": warmup_iteration,
                    "matching_status": warm_match["status"],
                    "matching_feasible": warm_match["feasible"],
                    "matching_seconds": round(
                        float(warm_match["seconds"]), 4
                    ),
                    "cut_added": False,
                    "master_reoptimization_seconds": 0.0,
                    "exact_start_verified": bool(
                        warm_validation["verified"]
                    ),
                    "candidate_objective": warm_validation["objective"],
                    "voyage_subproblems": warm_validation[
                        "voyage_records"
                    ],
                }
                voyage_solves += int(
                    warm_validation["voyage_solve_count"]
                )
                voyage_cache_hits += int(
                    warm_validation["voyage_cache_hits"]
                )
                if warm_match["feasible"]:
                    if warm_validation["verified"]:
                        best_selected = Counter(
                            warm_validation["selected"]
                        )
                        best_import = Counter(warm_validation["imports"])
                        best_objective = float(
                            warm_validation["objective"]
                        )
                        self._apply_profile_master_start(
                            variables, best_selected, best_import
                        )
                        master.update()
                        master_feasibility[
                            "exact_start_verified"
                        ] = True
                        master_feasibility[
                            "exact_start_objective"
                        ] = best_objective
                    else:
                        master_feasibility[
                            "exact_start_verified"
                        ] = False
                    initial_disaggregation.append(warm_record)
                    break
                remaining = deadline - perf_counter()
                added, packing_record = self._add_profile_packing_cut(
                    master,
                    variables,
                    tuple(warm_match.get("core_states", ())),
                    warm_counts,
                    matching_cut_signatures,
                    matching_cuts + 1,
                    min(5.0, max(0.01, remaining)),
                )
                packing_added = added
                if packing_record:
                    packing_records.append(packing_record)
                if packing_added:
                    warm_record["cut_kind"] = "profile_packing"
                if (
                    not added
                    and warm_match["status"] == "infeasible"
                ):
                    added, binary_count = self._add_aggregate_logic_cut(
                        master,
                        variables,
                        warm_quantities,
                        warm_counts,
                        aggregate_feasibility_cuts + 1,
                        "feasibility",
                        aggregate_cut_signatures,
                        core_profile_states=tuple(
                            warm_match.get("core_states", ())
                        ),
                    )
                    if added:
                        aggregate_feasibility_cuts += 1
                        aggregate_cut_binaries += binary_count
                        warm_record["cut_kind"] = (
                            "conditional_matching_feasibility"
                        )
                warm_record["cut_added"] = added
                if not added:
                    initial_disaggregation.append(warm_record)
                    break
                if packing_added:
                    matching_cuts += 1
                remaining = deadline - perf_counter()
                if remaining <= 1e-6:
                    initial_disaggregation.append(warm_record)
                    break
                self._set_gurobi_param(
                    master,
                    "TimeLimit",
                    max(0.01, min(10.0, 0.35 * remaining)),
                )
                self._set_gurobi_param(master, "SolutionLimit", 1)
                reopt_started = perf_counter()
                master.optimize()
                warm_record["master_reoptimization_seconds"] = round(
                    perf_counter() - reopt_started, 4
                )
                self._try_set_gurobi_param(
                    master, "SolutionLimit", 2_000_000_000
                )
                initial_disaggregation.append(warm_record)

            for master_round in range(
                1, int(self.benders_config.max_iterations) + 1
            ):
                remaining = deadline - perf_counter()
                if remaining <= 1e-6:
                    termination_reason = "time_limit"
                    break
                master_allowance = self._master_search_allowance(
                    master_round, remaining
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
                    termination_reason = (
                        "profile_master_infeasible"
                        if master_status == "infeasible"
                        else "profile_master_without_incumbent"
                    )
                    break
                master_bound = self._gurobi_dual_bound(master)
                if math.isfinite(master_bound):
                    valid_lower_bound = max(valid_lower_bound, master_bound)
                master_objective = self._gurobi_objective_value(master)
                (
                    quantities,
                    profile_counts,
                    routing,
                    imports,
                    theta,
                ) = self._profile_master_assignment(master, variables)

                verification = self._verify_profile_assignment(
                    quantities,
                    profile_counts,
                    routing,
                    imports,
                    deadline,
                    cache,
                )
                matching_result = verification["matching"]
                matching_solves += 1
                matching_seconds += float(matching_result["seconds"])
                matching_variables = max(
                    matching_variables,
                    int(matching_result["variable_count"]),
                )
                voyage_solves += int(verification["voyage_solve_count"])
                voyage_cache_hits += int(
                    verification["voyage_cache_hits"]
                )
                if not matching_result["feasible"]:
                    remaining = deadline - perf_counter()
                    added, packing_record = self._add_profile_packing_cut(
                        master,
                        variables,
                        tuple(matching_result.get("core_states", ())),
                        profile_counts,
                        matching_cut_signatures,
                        matching_cuts + 1,
                        min(5.0, max(0.01, remaining)),
                    )
                    packing_added = added
                    if packing_record:
                        packing_records.append(packing_record)
                    cut_kind = None
                    if (
                        not added
                        and matching_result["status"] == "infeasible"
                    ):
                        added, binary_count = (
                            self._add_aggregate_logic_cut(
                                master,
                                variables,
                                quantities,
                                profile_counts,
                                aggregate_feasibility_cuts + 1,
                                "feasibility",
                                aggregate_cut_signatures,
                                core_profile_states=tuple(
                                    matching_result.get(
                                        "core_states", ()
                                    )
                                ),
                            )
                        )
                        if added:
                            aggregate_feasibility_cuts += 1
                            aggregate_cut_binaries += binary_count
                            cut_kind = (
                                "conditional_matching_feasibility"
                            )
                    master_round_rows.append(
                        {
                            "master_round": master_round,
                            "master_status": master_status,
                            "master_allowance": round(
                                master_allowance, 4
                            ),
                            "master_seconds": round(master_seconds, 4),
                            "master_objective": master_objective,
                            "master_bound": master_bound,
                            "active_profile_state_count": len(
                                profile_counts
                            ),
                            "active_profile_unit_count": sum(
                                profile_counts.values()
                            ),
                            "matching_status": matching_result["status"],
                            "matching_feasible": False,
                            "cuts_added": int(added),
                            "cut_kind": cut_kind or (
                                "profile_packing"
                                if packing_added
                                else None
                            ),
                            "candidate_objective": None,
                        }
                    )
                    if not added:
                        termination_reason = (
                            "footprint_matching_without_valid_cut"
                        )
                        break
                    if packing_added:
                        matching_cuts += 1
                    continue
                all_feasible = bool(verification["verified"])
                all_optimal = bool(verification["all_optimal"])
                voyage_records = verification["voyage_records"]
                candidate_objective = verification["objective"]
                selected = Counter(verification["selected"])
                row_recourse = float(verification["subproblem_objective"])
                theta_total = sum(theta.values())
                cuts_added = 0
                cut_kind = None
                global_result = None
                if (
                    all_feasible
                    and row_recourse <= theta_total + 1e-7
                ):
                    all_optimal = True

                # A feasible fast disaggregation whose cost meets theta is
                # already an exact certificate: theta is a valid row-cost
                # lower bound and the concrete placement attains it.  Only a
                # failed matching/row check or a positive recourse gap needs
                # the larger joint oracle.
                needs_global_oracle = (
                    not all_feasible
                    or row_recourse > theta_total + 1e-7
                )
                if needs_global_oracle:
                    remaining = deadline - perf_counter()
                    if remaining > 1e-6:
                        oracle_allowance = min(
                            remaining,
                            max(
                                2.0,
                                min(
                                    2.0
                                    * float(
                                        self.benders_config.voyage_time_limit
                                    ),
                                    0.60 * remaining,
                                ),
                            ),
                        )
                        global_result = (
                            self._solve_global_profile_subproblem(
                                quantities,
                                profile_counts,
                                oracle_allowance,
                                selected if all_feasible else None,
                            )
                        )
                        global_subproblem_solves += 1
                        global_subproblem_seconds += float(
                            global_result["seconds"]
                        )
                        global_subproblem_records.append(
                            {
                                "master_round": master_round,
                                "status": global_result["status"],
                                "feasible": global_result["feasible"],
                                "optimal": global_result["optimal"],
                                "seconds": round(
                                    float(global_result["seconds"]), 4
                                ),
                                "objective": (
                                    global_result["objective"]
                                    if global_result["feasible"]
                                    else None
                                ),
                                "bound": global_result["bound"],
                                "conflict_quantity_count": len(
                                    global_result[
                                        "conflict_quantity_keys"
                                    ]
                                ),
                                "conflict_profile_state_count": len(
                                    global_result[
                                        "conflict_profile_states"
                                    ]
                                ),
                            }
                        )

                if global_result is not None:
                    all_feasible = bool(global_result["feasible"])
                    all_optimal = bool(global_result["optimal"])
                    if all_feasible:
                        selected = Counter(global_result["selected"])
                        row_recourse = float(global_result["objective"])
                        candidate_objective = (
                            self._selected_solution_energy(selected)
                        )
                        oracle_bound = float(global_result["bound"])
                        # Equality with the valid master lower estimate also
                        # proves recourse optimality after a time-limited MIP.
                        if row_recourse <= theta_total + 1e-7:
                            all_optimal = True
                        elif (
                            math.isfinite(oracle_bound)
                            and oracle_bound > theta_total + 1e-7
                        ):
                            added, binary_count = (
                                self._add_aggregate_logic_cut(
                                    master,
                                    variables,
                                    quantities,
                                    profile_counts,
                                    aggregate_optimality_cuts + 1,
                                    "optimality",
                                    aggregate_cut_signatures,
                                    lower_bound=oracle_bound,
                                )
                            )
                            if added:
                                aggregate_optimality_cuts += 1
                                aggregate_cut_binaries += binary_count
                                cuts_added += 1
                                cut_kind = "aggregate_optimality"
                    elif global_result["status"] == "infeasible":
                        added, binary_count = self._add_aggregate_logic_cut(
                            master,
                            variables,
                            quantities,
                            profile_counts,
                            aggregate_feasibility_cuts + 1,
                            "feasibility",
                            aggregate_cut_signatures,
                            core_quantity_keys=tuple(
                                global_result["conflict_quantity_keys"]
                            ),
                            core_profile_states=tuple(
                                global_result["conflict_profile_states"]
                            ),
                        )
                        if added:
                            aggregate_feasibility_cuts += 1
                            aggregate_cut_binaries += binary_count
                            cuts_added += 1
                            cut_kind = "aggregate_feasibility"

                if all_feasible:
                    decomposition_objective = (
                        master_objective - theta_total + row_recourse
                    )
                    self._absolute_deviation_auxiliary_slack(
                        decomposition_objective,
                        candidate_objective,
                        context="profile-resource LBBD decomposition",
                    )
                    if candidate_objective + 1e-9 < best_objective:
                        best_selected = Counter(selected)
                        best_import = Counter(imports)
                        best_objective = candidate_objective
                        best_master_round = master_round
                        self._apply_profile_master_start(
                            variables, best_selected, best_import
                        )
                        master.update()
                elif global_result is None and voyage_records:
                    last_status = voyage_records[-1]["status"]
                    termination_reason = (
                        "time_limit"
                        if last_status == "time_limit_before_solve"
                        else "global_disaggregation_not_attempted"
                    )
                elif global_result is not None and not cuts_added:
                    termination_reason = (
                        "global_disaggregation_without_certificate"
                    )

                master_round_rows.append(
                    {
                        "master_round": master_round,
                        "master_status": master_status,
                        "master_allowance": round(master_allowance, 4),
                        "master_seconds": round(master_seconds, 4),
                        "master_objective": master_objective,
                        "master_bound": master_bound,
                        "active_profile_state_count": len(profile_counts),
                        "active_profile_unit_count": sum(
                            profile_counts.values()
                        ),
                        "matching_status": matching_result["status"],
                        "matching_feasible": True,
                        "matching_seconds": round(
                            float(matching_result["seconds"]), 4
                        ),
                        "cuts_added": cuts_added,
                        "cut_kind": cut_kind,
                        "all_voyage_subproblems_feasible": all_feasible,
                        "all_voyage_subproblems_optimal": all_optimal,
                        "candidate_objective": candidate_objective,
                        "voyage_subproblems": voyage_records,
                        "global_disaggregation": (
                            global_subproblem_records[-1]
                            if global_result is not None
                            else None
                        ),
                    }
                )
                if cuts_added:
                    termination_reason = "aggregate_cut_added"
                    continue
                if not all_feasible:
                    break
                if master_status == "optimal" and all_optimal:
                    converged = True
                    termination_reason = "optimality_proven"
                    break
                termination_reason = (
                    "continuous_profile_master_limit_with_verified_incumbent"
                )
                break
            else:
                termination_reason = "iteration_limit"

            if best_selected is None or best_import is None:
                raise RuntimeError(
                    "profile-resource LBBD did not find a complete row "
                    "allocation; "
                    f"termination={termination_reason}, "
                    f"master_status={master_status}, "
                    f"master_rounds={len(master_round_rows)}, "
                    f"matching_cuts={matching_cuts}, "
                    f"voyage_solves={voyage_solves}, "
                    f"master_feasibility={master_feasibility}, "
                    f"initial_disaggregation={initial_disaggregation}, "
                    f"master_round_rows={master_round_rows}, "
                    f"packing_records={packing_records}"
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
                **self._base_profile_diagnostics(),
                "master_algorithm": "row_profile_resource_lbbd",
                "master_status": "optimal" if converged else master_status,
                "master_bound_scope": "valid_profile_resource_relaxation",
                "master_objective": best_objective,
                "master_mip_gap": relative_gap,
                "complete_model_lower_bound": valid_lower_bound,
                "complete_model_absolute_gap": absolute_gap,
                "complete_model_relative_gap": relative_gap,
                "complete_model_gap_source": "profile_resource_master_bound",
                "hard_demand_balance": True,
                "candidate_row_location_count": len(self._columns),
                "selected_location_count": len(best_selected),
                "profile_lbbd_converged": converged,
                "profile_lbbd_termination_reason": termination_reason,
                "profile_lbbd_master_round_count": len(
                    master_round_rows
                ),
                "profile_lbbd_best_master_round": best_master_round,
                "profile_lbbd_master_rounds": master_round_rows,
                "profile_lbbd_matching_cut_count": matching_cuts,
                "profile_lbbd_matching_solve_count": matching_solves,
                "profile_lbbd_matching_seconds": round(
                    matching_seconds, 3
                ),
                "profile_lbbd_matching_variable_count": matching_variables,
                "profile_lbbd_packing_records": packing_records,
                "profile_lbbd_aggregate_feasibility_cut_count": (
                    aggregate_feasibility_cuts
                ),
                "profile_lbbd_aggregate_optimality_cut_count": (
                    aggregate_optimality_cuts
                ),
                "profile_lbbd_aggregate_cut_binary_count": (
                    aggregate_cut_binaries
                ),
                "profile_lbbd_global_subproblem_solve_count": (
                    global_subproblem_solves
                ),
                "profile_lbbd_global_subproblem_seconds": round(
                    global_subproblem_seconds, 3
                ),
                "profile_lbbd_global_subproblem_build_seconds": round(
                    self._global_profile_subproblem.build_seconds, 3
                ) if self._global_profile_subproblem is not None else 0.0,
                "profile_lbbd_global_subproblem_variable_count": (
                    self._global_profile_subproblem.variable_count
                    if self._global_profile_subproblem is not None
                    else 0
                ),
                "profile_lbbd_global_subproblem_records": (
                    global_subproblem_records
                ),
                "profile_lbbd_initial_disaggregation": (
                    initial_disaggregation
                ),
                "profile_lbbd_voyage_solve_count": voyage_solves,
                "profile_lbbd_voyage_cache_hits": voyage_cache_hits,
                "profile_lbbd_voyage_build_seconds": round(
                    subproblem_build_seconds, 3
                ),
                "profile_lbbd_preparation_seconds": round(
                    preparation_seconds, 3
                ),
                "profile_lbbd_master_build_seconds": round(
                    master_build_seconds, 3
                ),
                "profile_lbbd_master_feasibility": master_feasibility,
                "profile_lbbd_total_solve_seconds": round(
                    perf_counter() - started, 3
                ),
                "profile_lbbd_master": master_stats,
            }
            result = self._assemble_result(best_selected, diagnostics)
            result.columns = self._selected_lbbd_columns(best_selected)
            return result
        finally:
            self._free_gurobi_model(master)
            for subproblem in self._voyage_subproblems.values():
                self._free_gurobi_model(subproblem.model)
            if self._global_profile_subproblem is not None:
                self._free_gurobi_model(
                    self._global_profile_subproblem.model
                )


__all__ = ["ProfileResourceBendersPlanner"]
