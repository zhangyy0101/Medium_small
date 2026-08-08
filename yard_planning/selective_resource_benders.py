"""Selective resource-state LBBD with one exact row-recourse oracle."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations
from time import perf_counter

from .profile_resource_benders import (
    ProfileState,
    ProfileResourceBendersPlanner,
)
from .gurobi_backend import GurobiModel
from .voyage_resource_benders import QuantityKey


class SelectiveResourceBendersPlanner(ProfileResourceBendersPlanner):
    """Solve a selective master and certify every point by exact recourse.

    Profile states are strengthening variables, not first-stage decisions.
    Only group-bay quantities are fixed in the exact row model.  This makes
    feasibility and optimality cuts conditional on the actual planning
    decision and keeps the master-oracle loop independent of the strict
    profile solver's matching and per-voyage verification chain.
    """

    # Fractions of the single wall-clock limit.  These are scale-free phase
    # envelopes, not reservations: unused time immediately returns to the
    # common pool and every solve is also capped by the time still available.
    _TIME_SHARES = {
        "initial_feasibility": 0.22,
        "formal_master_reserve": 0.35,
        "oracle": 0.06,
        "initial_repair": 0.13,
        "incumbent_repair": 0.08,
        "initial_master": 0.18,
        "later_master": 0.20,
        "local_bounds": 0.04,
        "primal_polish": 0.08,
        "capacity_separation": 0.015,
        "safety_margin": 0.01,
    }

    def __init__(self, problem, config=None, benders_config=None) -> None:
        super().__init__(problem, config, benders_config)
        self._selected_profile_states: tuple[ProfileState, ...] = ()
        self._selection_scores: dict[ProfileState, float] = {}
        self._selective_oracle_cache: dict[tuple, dict] = {}
        self._selective_oracle_solve_count = 0
        self._selective_oracle_cache_hits = 0
        self._selective_oracle_seconds = 0.0
        self._selective_oracle_build_seconds = 0.0
        self._selective_oracle_variable_count = 0
        self._conflict_repair_solve_count = 0
        self._conflict_repair_success_count = 0
        self._conflict_repair_seconds = 0.0
        self._restricted_primal_solve_count = 0
        self._restricted_primal_success_count = 0
        self._restricted_primal_seconds = 0.0
        self._neighbourhood_group_visits: Counter[str] = Counter()
        self._selective_master_variables: dict | None = None
        self._resource_capacity_cache: dict[tuple[QuantityKey, ...], dict] = {}
        self._resource_capacity_solve_count = 0
        self._resource_capacity_seconds = 0.0
        self._voyage_bound_cache: dict[tuple, dict] = {}
        self._voyage_bound_solve_count = 0
        self._voyage_bound_cache_hits = 0
        self._voyage_bound_seconds = 0.0

    def _total_time_limit(self) -> float:
        total = float(self.config.total_time_limit)
        if total <= 0.0:
            total = max(2.0, 2.0 * float(self.config.mip_time_limit))
        return total

    def _time_share(self, phase: str) -> float:
        return self._TIME_SHARES[phase] * self._total_time_limit()

    def _solve_threshold(self) -> float:
        """Small scale-aware guard against launching unusable solver calls."""
        return max(0.05, 0.002 * self._total_time_limit())

    def _prepare_profiles(self) -> None:
        super()._prepare_profiles()
        self._select_resource_states()

    def _select_resource_states(self) -> None:
        """Select a sublinear, voyage-balanced set of scarce states."""
        overlap_count: Counter[int] = Counter()
        for profile_ids, _upper in self._overlap_pool_bounds:
            overlap_count.update(profile_ids)
        shared_pool_count: Counter[int] = Counter()
        for profile_ids in self._profile_ids_by_physical_pool.values():
            if len(profile_ids) > 1:
                shared_pool_count.update(profile_ids)

        bay_rank_by_group = {}
        for group in self.groups:
            candidates = self._candidate_bays_for_group(group)
            bay_rank_by_group[group.group_id] = {
                bay_key: 1.0 - rank / max(1, len(candidates) - 1)
                for rank, (bay_key, _capacity, _cost) in enumerate(
                    candidates
                )
            }

        scores = {}
        states_by_voyage: defaultdict[str, list[ProfileState]] = defaultdict(
            list
        )
        for state in self._profile_states:
            profile_id, row_class = state
            profile = self._profiles[profile_id]
            routes = [
                route
                for route in self._profile_routes_by_profile[profile_id]
                if self._row_mix_key_for_group(
                    self.groups_by_id[route[0][0]]
                )
                == row_class
            ]
            group_ids = {route[0][0] for route in routes}
            compatible_demand = sum(
                int(self.group_demand[group_id]) for group_id in group_ids
            )
            unit_capacity = max(
                (self._profile_route_capacity[route] for route in routes),
                default=1,
            )
            pressure = compatible_demand / max(
                1, int(unit_capacity) * int(profile.multiplicity)
            )
            footprint_length = len(
                self._templates[profile.template_ids[0]].slots
            )
            preferred_share = sum(
                self._is_big_plan_area_for_group(
                    self.groups_by_id[route[0][0]],
                    self.bays[profile.bay_key].area_no,
                )
                for route in routes
            ) / max(1, len(routes))
            candidate_rank = sum(
                bay_rank_by_group[route[0][0]].get(
                    profile.bay_key, 0.0
                )
                for route in routes
            ) / max(1, len(routes))
            scores[state] = float(
                3.0 * min(3.0, pressure)
                + 2.0 * max(0, footprint_length - 1)
                + 1.5 * overlap_count[profile_id]
                + 0.75 * shared_pool_count[profile_id]
                + 1.0 / max(1, profile.multiplicity)
                + 4.0 * preferred_share
                + 2.0 * candidate_rank
            )
            states_by_voyage[profile.voyage_id].append(state)

        state_count = len(self._profile_states)
        budget = min(
            state_count,
            max(64, int(math.ceil(state_count**0.75))),
        )
        voyage_quota = max(1, budget // max(1, len(states_by_voyage)))
        ranking_key = lambda state: (-scores[state], repr(state))
        selected: set[ProfileState] = set()
        for states in states_by_voyage.values():
            selected.update(
                sorted(states, key=ranking_key)[:voyage_quota]
            )
        if len(selected) < budget:
            for state in sorted(self._profile_states, key=ranking_key):
                selected.add(state)
                if len(selected) >= budget:
                    break
        elif len(selected) > budget:
            selected = set(sorted(selected, key=ranking_key)[:budget])
        self._selection_scores = scores
        self._selected_profile_states = tuple(sorted(selected))

    def _build_profile_master(self):
        model, variables, stats = super()._build_profile_master()
        selected = set(self._selected_profile_states)
        for state, variable in variables["profile_use"].items():
            if state not in selected:
                variable.VType = "C"
        row_keys = tuple(sorted(variables["row_count"]))
        row_budget = min(
            len(row_keys),
            max(32, int(math.ceil(len(row_keys) ** 0.65))),
        )
        row_scores = {}
        row_keys_by_voyage: defaultdict[str, list[tuple]] = defaultdict(list)
        for key in row_keys:
            operational_key, bay_key = key
            group_ids = self._operational_bay_groups.get(key, ())
            compatible_states = {
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
            relaxed_share = (
                sum(state not in selected for state in compatible_states)
                / max(1, len(compatible_states))
            )
            demand = sum(
                int(self.group_demand[group_id])
                for group_id in self._operational_groups[operational_key]
            )
            first_row_capacity = max(
                self._row_capacity_profile.get(key, (1,)),
                default=1,
            )
            row_scores[key] = (
                2.0 * relaxed_share
                + min(3.0, demand / max(1, first_row_capacity))
            )
            voyage_id = self._representative_group[
                operational_key
            ].voyage_id
            row_keys_by_voyage[voyage_id].append(key)
        row_ranking = lambda key: (-row_scores[key], repr(key))
        row_quota = max(
            1, row_budget // max(1, len(row_keys_by_voyage))
        )
        relaxed_row_keys = set()
        for keys in row_keys_by_voyage.values():
            relaxed_row_keys.update(
                sorted(keys, key=row_ranking)[:row_quota]
            )
        if len(relaxed_row_keys) < row_budget:
            for key in sorted(row_keys, key=row_ranking):
                relaxed_row_keys.add(key)
                if len(relaxed_row_keys) >= row_budget:
                    break
        elif len(relaxed_row_keys) > row_budget:
            relaxed_row_keys = set(
                sorted(relaxed_row_keys, key=row_ranking)[:row_budget]
            )
        for key in relaxed_row_keys:
            variables["row_count"][key].VType = "C"
        model.update()
        stats.update(
            {
                "selected_profile_state_integer_count": len(selected),
                "relaxed_profile_state_count": (
                    len(variables["profile_use"]) - len(selected)
                ),
                "selected_profile_state_fraction": round(
                    len(selected)
                    / max(1, len(variables["profile_use"])),
                    8,
                ),
                "row_count_integrality": (
                    "selective_continuous_relaxation_exactly_checked_by_"
                    "row_oracle"
                ),
                "integer_row_count_count": (
                    len(row_keys) - len(relaxed_row_keys)
                ),
                "relaxed_row_count_count": len(relaxed_row_keys),
                "relaxed_row_count_fraction": round(
                    len(relaxed_row_keys) / max(1, len(row_keys)), 8
                ),
                "row_relaxation_rule": (
                    "voyage_balanced_profile_relaxation_and_row_pressure"
                ),
                "selection_rule": (
                    "voyage_balanced_scarcity_and_business_relevance_"
                    "score_with_sublinear_budget"
                ),
            }
        )
        self._selective_master_variables = variables
        return model, variables, stats

    def _master_feasibility_slice(self, allowance: float) -> float:
        """Scale the feasibility cap with the common wall-clock budget.

        ``SolutionLimit=1`` still stops immediately after a skeleton is
        found; the proportional cap only gives harder instances additional
        search without identifying an instance class in the algorithm.
        """
        return min(
            float(allowance),
            self._time_share("initial_feasibility"),
        )

    def _initialize_master_incumbent(self, model, deadline: float) -> dict:
        """Probe a coarse row relaxation, then restore the formal master.

        This phase supplies only a quantity point for the exact oracle and the
        conflict-repair neighbourhood.  The formal master keeps its selected
        row-count integrality, and no profile state is promoted from the
        resulting primal solution.
        """
        if self._selective_master_variables is None:
            raise RuntimeError("selective master variables are unavailable")
        row_variables = tuple(
            self._selective_master_variables["row_count"].values()
        )
        original_types = tuple(variable.VType for variable in row_variables)
        for variable in row_variables:
            variable.VType = "C"
        self._try_set_gurobi_param(model, "Heuristics", 1.0)
        self._try_set_gurobi_param(model, "PumpPasses", 20)
        model.update()
        try:
            result = super()._initialize_master_incumbent(model, deadline)
        finally:
            self._try_set_gurobi_param(model, "Heuristics", 0.5)
            self._try_set_gurobi_param(model, "PumpPasses", -1)
            for variable, variable_type in zip(
                row_variables, original_types
            ):
                variable.VType = variable_type
            model.update()
        result["probe_row_count_integrality"] = "continuous"
        result["formal_row_count_integrality_restored"] = True
        return result

    def _global_fixed_profile_states(self) -> tuple[ProfileState, ...]:
        return ()

    def _logic_cut_profile_states(self) -> tuple[ProfileState, ...]:
        return ()

    @staticmethod
    def _selective_assignment_key(
        quantities: dict[QuantityKey, int],
    ) -> tuple:
        return tuple(sorted(quantities.items()))

    def _solve_exact_recourse(
        self,
        quantities: dict[QuantityKey, int],
        time_limit: float,
        selected_hint: Counter[int] | None = None,
    ) -> dict:
        """Solve the exact joint row oracle only on the positive support.

        Every omitted group-bay key is fixed to zero by the queried master
        point, so removing its row-placement variables is an exact model
        reduction.  Rebuilding this sparse model is substantially more
        stable than constructing one all-candidate oracle before any solve.
        """
        from gurobipy import quicksum

        key = self._selective_assignment_key(quantities)
        cached = self._selective_oracle_cache.get(key)
        if cached is not None:
            self._selective_oracle_cache_hits += 1
            result = dict(cached)
            result["seconds"] = 0.0
            result["cached"] = True
            return result
        started = perf_counter()
        positive_keys = tuple(
            sorted(
                quantity_key
                for quantity_key, value in quantities.items()
                if int(value) > 0
            )
        )
        global_indices = tuple(
            sorted(
                {
                    index
                    for quantity_key in positive_keys
                    for index in self._candidate_indices_by_group_bay.get(
                        quantity_key, ()
                    )
                }
            )
        )
        result = {
            "status": "not_solved",
            "feasible": False,
            "optimal": False,
            "seconds": 0.0,
            "selected": Counter(),
            "objective": math.inf,
            "bound": -math.inf,
            "conflict_quantity_keys": (),
            "conflict_profile_states": (),
            "variable_count": 0,
            "support_quantity_count": len(positive_keys),
            "support_row_location_count": len(global_indices),
            "cached": False,
        }
        if not global_indices:
            result["status"] = "empty_positive_support"
            result["total_seconds"] = perf_counter() - started
            return result

        row_locations = [self._columns[index] for index in global_indices]
        model, variables, model_stats = self.build_compact_row_milp(
            row_locations, GurobiModel, quicksum
        )
        self._selective_oracle_variable_count = max(
            self._selective_oracle_variable_count,
            int(model_stats["model_variable_count"]),
        )
        result["variable_count"] = int(
            model_stats["model_variable_count"]
        )
        result["model"] = model_stats
        try:
            local_by_quantity: defaultdict[QuantityKey, list[int]] = (
                defaultdict(list)
            )
            for local_index, global_index in enumerate(global_indices):
                column = self._columns[global_index]
                local_by_quantity[
                    (column.group_id, column.bay_key)
                ].append(local_index)
            quantity_balance = {
                quantity_key: model.addConstr(
                    quicksum(
                        variables["column"][local_index]
                        for local_index
                        in local_by_quantity[quantity_key]
                    )
                    == int(quantities[quantity_key]),
                    name=(
                        f"sparse_oracle_quantity_{quantity_key[0]}_"
                        f"{self._key_name((quantity_key[1],))}"
                    ),
                )
                for quantity_key in positive_keys
            }
            model.update()
            row_penalty = self._row_activation_penalty()
            for variable in model.getVars():
                model.setVarObjective(
                    variable,
                    row_penalty
                    if variable.VarName.startswith("use_group_row_")
                    else 0.0,
                )
            hint = selected_hint or Counter()
            for local_index, global_index in enumerate(global_indices):
                variables["column"][local_index].Start = float(
                    hint.get(global_index, 0)
                )
            model.update()
            build_seconds = perf_counter() - started
            self._selective_oracle_build_seconds += build_seconds
            result["build_seconds"] = round(build_seconds, 4)
            self._set_gurobi_param(model, "MIPGap", 0.0)
            self._set_gurobi_param(model, "DualReductions", 0)
            solve_allowance = float(time_limit) - (
                perf_counter() - started
            )
            if solve_allowance <= self._solve_threshold():
                result["status"] = "time_limit_during_build"
                return result
            self._set_gurobi_param(model, "TimeLimit", solve_allowance)
            solve_started = perf_counter()
            model.optimize()
            result["seconds"] = perf_counter() - solve_started
            result["status"] = self._gurobi_status_name(model)
            result["feasible"] = self._gurobi_solution_count(model) > 0
            result["optimal"] = result["status"] == "optimal"
            result["bound"] = self._gurobi_dual_bound(model)
            if result["feasible"]:
                local_selected = self.selected_compact_row_values(
                    model, variables
                )
                result["selected"] = Counter(
                    {
                        global_indices[local_index]: int(value)
                        for local_index, value in local_selected.items()
                        if int(value) > 0
                    }
                )
                result["objective"] = self._gurobi_objective_value(model)
                return result
            if result["status"] == "infeasible":
                iis_allowance = float(time_limit) - (
                    perf_counter() - started
                )
                if iis_allowance > self._solve_threshold():
                    try:
                        self._set_gurobi_param(
                            model, "TimeLimit", iis_allowance
                        )
                        model._model.computeIIS()
                        result["conflict_quantity_keys"] = tuple(
                            sorted(
                                quantity_key
                                for quantity_key, constraint
                                in quantity_balance.items()
                                if bool(constraint.IISConstr)
                            )
                        )
                    except Exception:
                        pass
            return result
        finally:
            result["total_seconds"] = perf_counter() - started
            self._selective_oracle_solve_count += 1
            self._selective_oracle_seconds += float(result["seconds"])
            if result["optimal"] or result["status"] == "infeasible":
                self._selective_oracle_cache[key] = dict(result)
            self._free_gurobi_model(model)

    def _validate_exact_quantities(
        self,
        quantities: dict[QuantityKey, int],
        selected: Counter[int],
    ) -> None:
        actual: Counter[QuantityKey] = Counter()
        for index, value in selected.items():
            if int(value) > 0:
                column = self._columns[index]
                actual[(column.group_id, column.bay_key)] += int(value)
        expected = Counter(
            {
                key: int(value)
                for key, value in quantities.items()
                if int(value) > 0
            }
        )
        if actual != expected:
            raise RuntimeError(
                "selective recourse changed fixed group-bay quantities: "
                f"expected={expected}, actual={actual}"
            )

    def _add_monotone_iis_feasibility_cut(
        self,
        master,
        variables: dict,
        quantities: dict[QuantityKey, int],
        conflict_keys: tuple[QuantityKey, ...],
        cut_index: int,
        signatures: set[tuple],
    ) -> tuple[bool, int]:
        """Exclude the upward-closed packing conflict certified by an IIS."""
        from gurobipy import quicksum

        core = tuple(
            sorted(
                key
                for key in set(conflict_keys)
                if int(quantities.get(key, 0)) > 0
            )
        )
        signature = (
            "monotone_iis",
            tuple((key, int(quantities[key])) for key in core),
        )
        if not core or signature in signatures:
            return False, 0
        decreases = []
        for position, key in enumerate(core):
            incumbent = int(quantities[key])
            upper = int(self._quantity_upper[key])
            decrease = master.addVar(
                vtype="B",
                name=f"selective_iis_down_{cut_index}_{position}",
            )
            master.addConstr(
                variables["quantity"][key]
                >= incumbent - upper * decrease,
                name=f"selective_iis_down_lb_{cut_index}_{position}",
            )
            master.addConstr(
                variables["quantity"][key]
                <= incumbent - 1 + upper * (1 - decrease),
                name=f"selective_iis_down_ub_{cut_index}_{position}",
            )
            decreases.append(decrease)
        master.addConstr(
            quicksum(decreases) >= 1.0,
            name=f"selective_monotone_iis_{cut_index}",
        )
        master.update()
        signatures.add(signature)
        return True, len(decreases)

    def _monotone_iis_support_limit(self) -> int:
        """Scale the fallback logic-cut support with the master dimension.

        A fixed 64-key threshold discarded useful medium conflict sets in
        larger instances.  The square-root rule admits those sets without
        allowing a single IIS to add an unbounded number of binaries.
        """
        quantity_state_count = max(
            1,
            len(getattr(self, "_quantity_upper", {})),
        )
        return min(
            512,
            max(64, int(math.ceil(3.0 * math.sqrt(quantity_state_count)))),
        )

    def _certify_core_resource_capacity(
        self,
        conflict_keys: tuple[QuantityKey, ...],
        time_limit: float,
    ) -> dict:
        """Upper-bound a typed IIS core by exact footprint-flow packing.

        The auxiliary MIP routes each quantity key only to compatible row
        owners, caps every key by its valid master upper bound, and enforces
        physical-footprint exclusivity.  Its maximization bound is therefore
        a valid upper bound on the total master quantity carried by the core.
        """
        from gurobipy import quicksum

        core = tuple(sorted(set(conflict_keys)))
        cached = self._resource_capacity_cache.get(core)
        if cached is not None:
            return {**cached, "cached": True, "seconds": 0.0}
        started = perf_counter()
        if len(core) > 64:
            return {
                "status": "skipped_large_IIS_core",
                "certified": False,
                "capacity_upper_bound": None,
                "owner_count": None,
                "seconds": round(perf_counter() - started, 4),
                "cached": False,
            }
        owner_capacity = {}
        route_capacity = {}
        for key in core:
            for index in self._candidate_indices_by_group_bay.get(key, ()):
                owner_key = self._owner_key_by_candidate[index]
                owner_capacity[owner_key] = max(
                    owner_capacity.get(owner_key, 0),
                    int(self._candidate_capacity[index]),
                )
                route = (key, owner_key)
                route_capacity[route] = max(
                    route_capacity.get(route, 0),
                    int(self._candidate_capacity[index]),
                )
        result = {
            "status": "empty",
            "certified": False,
            "capacity_upper_bound": None,
            "owner_count": len(owner_capacity),
            "typed_route_count": len(route_capacity),
            "seconds": 0.0,
            "cached": False,
        }
        if not owner_capacity or not route_capacity:
            return result
        model = GurobiModel("selective_IIS_resource_capacity")
        self._configure_gurobi_output(model)
        self._set_gurobi_param(model, "Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            self._set_gurobi_param(
                model, "Threads", int(self.config.solver_threads)
            )
        model.setMaximize()
        owner = {
            owner_key: model.addVar(
                vtype="B",
                name=(
                    f"core_owner_{owner_key[0]}_"
                    f"{self._key_name((owner_key[1],))}"
                ),
            )
            for owner_key, capacity in sorted(owner_capacity.items())
        }
        flow = {
            route: model.addVar(
                lb=0.0,
                ub=float(
                    min(
                        int(capacity),
                        int(self._quantity_upper[route[0]]),
                    )
                ),
                vtype="I",
                obj=1.0,
                name=(
                    f"core_flow_{route[0][0]}_"
                    f"{self._key_name((route[0][1], str(route[1][0]), route[1][1]))}"
                ),
            )
            for route, capacity in sorted(route_capacity.items())
        }
        routes_by_owner: defaultdict[object, list] = defaultdict(list)
        routes_by_quantity: defaultdict[QuantityKey, list] = defaultdict(list)
        for route, variable in flow.items():
            routes_by_owner[route[1]].append(variable)
            routes_by_quantity[route[0]].append(variable)
        for owner_key, variable in owner.items():
            model.addConstr(
                quicksum(routes_by_owner[owner_key])
                <= int(owner_capacity[owner_key]) * variable,
                name=(
                    f"core_owner_capacity_{owner_key[0]}_"
                    f"{self._key_name((owner_key[1],))}"
                ),
            )
        for key, routes in sorted(routes_by_quantity.items()):
            model.addConstr(
                quicksum(routes) <= int(self._quantity_upper[key]),
                name=(
                    f"core_quantity_upper_{key[0]}_"
                    f"{self._key_name((key[1],))}"
                ),
            )
        owners_by_slot: defaultdict[object, list] = defaultdict(list)
        for owner_key, variable in owner.items():
            for slot in self._templates[owner_key[0]].slots:
                owners_by_slot[slot].append(variable)
        for slot, slot_owners in sorted(owners_by_slot.items()):
            model.addConstr(
                quicksum(slot_owners) <= 1.0,
                name=f"core_slot_{self._key_name(slot)}",
            )
        model.update()
        try:
            self._set_gurobi_param(
                model, "TimeLimit", max(0.01, float(time_limit))
            )
            model.optimize()
            result["status"] = self._gurobi_status_name(model)
            upper_bound = self._gurobi_dual_bound(model)
            if math.isfinite(upper_bound):
                result["certified"] = True
                result["capacity_upper_bound"] = int(
                    math.ceil(float(upper_bound) - 1e-8)
                )
        finally:
            elapsed = perf_counter() - started
            result["seconds"] = round(elapsed, 4)
            self._resource_capacity_solve_count += 1
            self._resource_capacity_seconds += elapsed
            self._free_gurobi_model(model)
        if result["certified"]:
            self._resource_capacity_cache[core] = dict(result)
        return result

    def _conflict_capacity_subsets(
        self,
        core: tuple[QuantityKey, ...],
    ) -> tuple[tuple[QuantityKey, ...], ...]:
        """Return a small deterministic Hall-separation family for an IIS."""
        keys = tuple(sorted(set(core)))
        subsets = {keys} if len(keys) <= 64 else set()
        if 2 <= len(keys) <= 5:
            for size in range(2, len(keys)):
                subsets.update(combinations(keys, size))

        structured: defaultdict[tuple, set[QuantityKey]] = defaultdict(set)
        for key in keys:
            group = self.groups_by_id[key[0]]
            row_class = self._row_mix_key_for_group(group)
            labels = (
                ("bay", key[1]),
                ("bay_class", key[1], row_class),
                ("voyage", group.voyage_id),
                ("row_class", row_class),
                ("voyage_class", group.voyage_id, row_class),
                ("size_class", group.size, row_class),
            )
            for label in labels:
                structured[label].add(key)
        subsets.update(
            tuple(sorted(values))
            for values in structured.values()
            if 2 <= len(values) <= 64
        )
        return tuple(
            sorted(
                subsets,
                key=lambda subset: (-len(subset), repr(subset)),
            )
        )

    def _add_certified_feasibility_cut(
        self,
        master,
        variables: dict,
        quantities: dict[QuantityKey, int],
        conflict_keys: tuple[QuantityKey, ...],
        cut_index: int,
        signatures: set[tuple],
    ) -> dict:
        """Prefer a lifted physical-capacity cut, then use the IIS cut."""
        from gurobipy import quicksum

        core = tuple(
            sorted(
                key
                for key in set(conflict_keys)
                if int(quantities.get(key, 0)) > 0
            )
        )
        started = perf_counter()
        candidate_cores = self._conflict_capacity_subsets(core)
        separation_budget = self._time_share("capacity_separation")
        candidate_budget = separation_budget / max(
            1,
            min(3, len(candidate_cores)),
        )
        capacity = {
            "status": "not_attempted",
            "certified": False,
            "capacity_upper_bound": None,
        }
        cut_core = core
        best_violation = 0
        for candidate_core in candidate_cores:
            remaining = separation_budget - (perf_counter() - started)
            if remaining <= self._solve_threshold():
                break
            candidate = self._certify_core_resource_capacity(
                candidate_core,
                min(candidate_budget, remaining),
            )
            candidate_upper = candidate.get("capacity_upper_bound")
            if candidate_upper is None:
                continue
            candidate_total = sum(
                int(quantities[key]) for key in candidate_core
            )
            violation = candidate_total - int(candidate_upper)
            if violation > best_violation:
                best_violation = violation
                capacity = candidate
                cut_core = candidate_core
            elif candidate_core == core and not capacity.get("certified"):
                capacity = candidate
        upper = capacity.get("capacity_upper_bound")
        incumbent_total = sum(int(quantities[key]) for key in cut_core)
        signature = ("typed_resource_capacity", cut_core, upper)
        if best_violation > 0 and signature not in signatures:
            master.addConstr(
                quicksum(variables["quantity"][key] for key in cut_core)
                <= int(upper),
                name=f"selective_resource_capacity_{cut_index}",
            )
            master.update()
            signatures.add(signature)
            return {
                "added": True,
                "binary_count": 0,
                "kind": "IIS_physical_resource_capacity",
                "capacity_upper_bound": int(upper),
                "incumbent_core_quantity": incumbent_total,
                "capacity_core": cut_core,
                "capacity_certificate": capacity,
            }
        monotone_support_limit = self._monotone_iis_support_limit()
        if len(core) > monotone_support_limit:
            return {
                "added": False,
                "binary_count": 0,
                "kind": "large_IIS_without_compact_certificate",
                "monotone_support_limit": monotone_support_limit,
                "capacity_upper_bound": upper,
                "incumbent_core_quantity": sum(
                    int(quantities[key]) for key in core
                ),
                "capacity_core": core,
                "capacity_certificate": capacity,
            }
        added, binary_count = self._add_monotone_iis_feasibility_cut(
            master,
            variables,
            quantities,
            core,
            cut_index,
            signatures,
        )
        return {
            "added": added,
            "binary_count": binary_count,
            "kind": "monotone_IIS_capacity_feasibility",
            "monotone_support_limit": monotone_support_limit,
            "capacity_upper_bound": upper,
            "incumbent_core_quantity": sum(
                int(quantities[key]) for key in core
            ),
            "capacity_core": core,
            "capacity_certificate": capacity,
        }

    def _selected_row_cost_by_voyage(
        self,
        selected: Counter[int],
    ) -> dict[str, float]:
        used_rows: defaultdict[str, set[tuple]] = defaultdict(set)
        for index, value in selected.items():
            if int(value) <= 0:
                continue
            column = self._columns[index]
            used_rows[column.voyage_id].add(
                (
                    column.group_key,
                    column.bay_key,
                    self._anchor_row(column),
                )
            )
        penalty = self._row_activation_penalty()
        return {
            voyage_id: penalty * len(rows)
            for voyage_id, rows in used_rows.items()
        }

    def _solve_voyage_recourse_bound(
        self,
        voyage_id: str,
        quantities: dict[QuantityKey, int],
        time_limit: float,
        selected_hint: Counter[int] | None = None,
    ) -> dict:
        voyage_quantities = {
            key: int(value)
            for key, value in quantities.items()
            if self.groups_by_id[key[0]].voyage_id == voyage_id
        }
        cache_key = (voyage_id, tuple(sorted(voyage_quantities.items())))
        cached = self._voyage_bound_cache.get(cache_key)
        if cached is not None:
            self._voyage_bound_cache_hits += 1
            return {**cached, "cached": True, "seconds": 0.0}
        active_owners = {
            owner_key
            for owner_key in self._owner_key_by_candidate.values()
            if self._templates[owner_key[0]].voyage_id == voyage_id
        }
        hint = selected_hint or Counter()
        routing_hint = {
            index: float(value)
            for index, value in hint.items()
            if int(value) > 0
            and self._columns[index].voyage_id == voyage_id
        }
        result = self._solve_voyage_subproblem(
            voyage_id,
            voyage_quantities,
            active_owners,
            routing_hint,
            time_limit,
        )
        self._voyage_bound_solve_count += 1
        self._voyage_bound_seconds += float(result["seconds"])
        result["cached"] = False
        if result["optimal"] or result["status"] == "infeasible":
            self._voyage_bound_cache[cache_key] = dict(result)
        return result

    def _add_voyage_optimality_cut(
        self,
        master,
        variables: dict,
        voyage_id: str,
        quantities: dict[QuantityKey, int],
        lower_bound: float,
        cut_index: int,
        signatures: set[tuple],
    ) -> tuple[bool, int, int]:
        """Condition a valid standalone-voyage bound on its quantity vector."""
        from gurobipy import quicksum

        keys = tuple(
            sorted(
                key
                for key, value in quantities.items()
                if int(value) > 0
                and self.groups_by_id[key[0]].voyage_id == voyage_id
            )
        )
        signature = (
            "voyage_optimality",
            voyage_id,
            tuple((key, int(quantities[key])) for key in keys),
            round(float(lower_bound), 10),
        )
        if not keys or signature in signatures:
            return False, 0, len(keys)
        changes = []
        for position, key in enumerate(keys):
            incumbent = int(quantities[key])
            upper = int(self._quantity_upper[key])
            decrease = master.addVar(
                vtype="B",
                name=(
                    f"voyage_opt_{cut_index}_{voyage_id}_down_{position}"
                ),
            )
            master.addConstr(
                variables["quantity"][key]
                >= incumbent - upper * decrease,
                name=(
                    f"voyage_opt_{cut_index}_{voyage_id}_lb_{position}"
                ),
            )
            master.addConstr(
                variables["quantity"][key]
                <= incumbent - 1 + upper * (1 - decrease),
                name=(
                    f"voyage_opt_{cut_index}_{voyage_id}_ub_{position}"
                ),
            )
            changes.append(decrease)
        master.addConstr(
            variables["theta"][voyage_id]
            >= max(0.0, float(lower_bound))
            * (1 - quicksum(changes)),
            name=f"voyage_optimality_{cut_index}_{voyage_id}",
        )
        master.update()
        signatures.add(signature)
        return True, len(changes), len(keys)

    def _conflict_repair_candidate_indices(
        self,
        quantities: dict[QuantityKey, int],
        conflict_keys: tuple[QuantityKey, ...],
        selected_hint: Counter[int] | None = None,
    ) -> tuple[tuple[int, ...], dict]:
        """Build a small exact-row neighbourhood around an infeasible point."""
        bays_by_group: defaultdict[str, set[str]] = defaultdict(set)
        for (group_id, bay_key), quantity in quantities.items():
            if int(quantity) > 0:
                bays_by_group[group_id].add(bay_key)
        if selected_hint:
            for index, value in selected_hint.items():
                if int(value) > 0:
                    column = self._columns[index]
                    bays_by_group[column.group_id].add(column.bay_key)

        iis_conflict_groups = {
            group_id
            for group_id, bay_key in conflict_keys
            if (
                group_id in self.groups_by_id
                and int(quantities.get((group_id, bay_key), 0)) > 0
            )
        }
        conflict_group_limit = 16
        conflict_groups = set(
            sorted(
                iis_conflict_groups,
                key=lambda group_id: (
                    -sum(
                        int(value)
                        for (key_group_id, _bay_key), value
                        in quantities.items()
                        if key_group_id == group_id
                    ),
                    group_id,
                ),
            )[:conflict_group_limit]
        )
        added_alternative_bays = 0
        for group in self.groups:
            chosen = bays_by_group[group.group_id]
            candidates = self._candidate_bays_for_group(group)
            if group.group_id in conflict_groups:
                per_area: Counter[str] = Counter()
                alternative_limit = 12
                per_area_limit = 3
            else:
                per_area = Counter()
                alternative_limit = 4
                per_area_limit = 2
            added = 0
            for bay_key, _capacity, _cost in candidates:
                if bay_key in chosen:
                    continue
                area_no = self.bays[bay_key].area_no
                if per_area[area_no] >= per_area_limit:
                    continue
                chosen.add(bay_key)
                per_area[area_no] += 1
                added += 1
                added_alternative_bays += 1
                if added >= alternative_limit:
                    break

        indices = {
            index
            for group_id, bay_keys in bays_by_group.items()
            for bay_key in bay_keys
            for index in self._candidate_indices_by_group_bay.get(
                (group_id, bay_key), ()
            )
        }
        return tuple(sorted(indices)), {
            "conflict_group_count": len(conflict_groups),
            "iis_conflict_group_count": len(iis_conflict_groups),
            "conflict_group_limit": conflict_group_limit,
            "repair_group_bay_count": sum(
                len(bay_keys) for bay_keys in bays_by_group.values()
            ),
            "added_alternative_bay_count": added_alternative_bays,
        }

    def _solve_conflict_directed_repair(
        self,
        quantities: dict[QuantityKey, int],
        conflict_keys: tuple[QuantityKey, ...],
        time_limit: float,
        selected_hint: Counter[int] | None = None,
        import_hint: Counter[tuple[str, str, str]] | None = None,
        secure_first_feasible: bool = False,
    ) -> dict:
        """Repair an infeasible master point in a restricted exact-row MIP.

        The original normalized business objective is primary.  L1 change in
        group-bay quantities only breaks ties, because preserving a poor
        feasibility skeleton is less important than obtaining a strong upper
        bound.  The solution is only an incumbent and a MIP start: it never
        changes a master variable type or the validity of the master bound.
        """
        from gurobipy import quicksum

        started = perf_counter()
        global_indices, neighbourhood = (
            self._conflict_repair_candidate_indices(
                quantities, conflict_keys, selected_hint
            )
        )
        result = {
            "status": "not_solved",
            "feasible": False,
            "seconds": 0.0,
            "selected": Counter(),
            "imports": Counter(),
            "objective": None,
            "quantity_l1_change": None,
            "relocated_boxes": None,
            "candidate_row_location_count": len(global_indices),
            **neighbourhood,
        }
        self._conflict_repair_solve_count += 1
        if not global_indices:
            result["status"] = "empty_neighbourhood"
            return result

        row_locations = [self._columns[index] for index in global_indices]
        model, variables, model_stats = self.build_compact_row_milp(
            row_locations,
            GurobiModel,
            quicksum,
        )
        result["model"] = model_stats
        try:
            local_by_group_bay: defaultdict[QuantityKey, list[int]] = (
                defaultdict(list)
            )
            for local_index, global_index in enumerate(global_indices):
                column = self._columns[global_index]
                local_by_group_bay[
                    (column.group_id, column.bay_key)
                ].append(local_index)

            deviation_terms = []
            target_keys = set(local_by_group_bay) | {
                key for key, value in quantities.items() if int(value) > 0
            }
            for position, key in enumerate(sorted(target_keys)):
                positive = model.addVar(
                    lb=0.0,
                    name=f"repair_dev_pos_{position}",
                )
                negative = model.addVar(
                    lb=0.0,
                    name=f"repair_dev_neg_{position}",
                )
                actual = quicksum(
                    variables["column"][index]
                    for index in local_by_group_bay.get(key, ())
                )
                model.addConstr(
                    actual - int(quantities.get(key, 0))
                    == positive - negative,
                    name=f"repair_dev_balance_{position}",
                )
                deviation_terms.extend((positive, negative))

            model.update()
            original_objective = model._model.getObjective()
            model._model.setObjectiveN(
                original_objective,
                0,
                priority=2,
                weight=1.0,
                abstol=0.0,
                reltol=0.0,
                name="business_objective",
            )
            model._model.setObjectiveN(
                quicksum(deviation_terms),
                1,
                priority=1,
                weight=1.0,
                abstol=0.0,
                reltol=0.0,
                name="minimum_group_bay_change_tiebreak",
            )
            for local_index, global_index in enumerate(global_indices):
                variables["column"][local_index].Start = float(
                    (selected_hint or {}).get(global_index, 0)
                )
            for key, variable in variables["import_reserve"].items():
                variable.Start = float((import_hint or {}).get(key, 0))

            build_seconds = perf_counter() - started
            solve_allowance = float(time_limit) - build_seconds
            result["build_seconds"] = round(build_seconds, 4)
            if solve_allowance <= 1e-6:
                result["status"] = "time_limit_during_build"
                return result
            self._set_gurobi_param(
                model, "TimeLimit", max(0.01, solve_allowance)
            )
            self._set_gurobi_param(model, "MIPFocus", 1)
            self._set_gurobi_param(model, "Heuristics", 0.75)
            self._try_set_gurobi_param(model, "PumpPasses", 10)
            self._try_set_gurobi_param(model, "RINS", 10)
            self._set_gurobi_param(model, "MIPGap", 0.0)
            if secure_first_feasible:
                self._set_gurobi_param(model, "SolutionLimit", 1)
            solve_started = perf_counter()
            model.optimize()
            feasibility_seconds = perf_counter() - solve_started
            improvement_seconds = 0.0
            if (
                secure_first_feasible
                and self._gurobi_solution_count(model) > 0
            ):
                remaining_solve_time = (
                    float(time_limit) - (perf_counter() - started)
                )
                if remaining_solve_time > self._solve_threshold():
                    self._set_gurobi_param(
                        model, "SolutionLimit", 2_000_000_000
                    )
                    self._set_gurobi_param(
                        model, "TimeLimit", remaining_solve_time
                    )
                    improvement_started = perf_counter()
                    model.optimize()
                    improvement_seconds = (
                        perf_counter() - improvement_started
                    )
            result["status"] = self._gurobi_status_name(model)
            if self._gurobi_solution_count(model) <= 0:
                return result

            local_selected = self.selected_compact_row_values(
                model, variables
            )
            selected = Counter(
                {
                    global_indices[local_index]: int(value)
                    for local_index, value in local_selected.items()
                    if int(value) > 0
                }
            )
            imports = self._gurobi_import_reservation_values(
                model, variables
            )
            actual_quantities: Counter[QuantityKey] = Counter()
            for index, value in selected.items():
                column = self._columns[index]
                actual_quantities[
                    (column.group_id, column.bay_key)
                ] += int(value) * int(column.quantity)
            l1_change = sum(
                abs(
                    int(actual_quantities.get(key, 0))
                    - int(quantities.get(key, 0))
                )
                for key in set(actual_quantities) | set(quantities)
            )
            previous_imports = self._final_import_reservation
            self._final_import_reservation = Counter(imports)
            try:
                validation = self._validate_final_solution(selected)
                objective = self._selected_solution_energy(selected)
            finally:
                self._final_import_reservation = previous_imports
            result.update(
                {
                    "feasible": True,
                    "selected": selected,
                    "imports": imports,
                    "objective": objective,
                    "quantity_l1_change": int(l1_change),
                    "relocated_boxes": int(l1_change // 2),
                    "validation": validation,
                    "secure_first_feasible": bool(
                        secure_first_feasible
                    ),
                    "feasibility_search_seconds": round(
                        feasibility_seconds, 4
                    ),
                    "same_model_improvement_seconds": round(
                        improvement_seconds, 4
                    ),
                }
            )
            self._conflict_repair_success_count += 1
            return result
        finally:
            elapsed = perf_counter() - started
            result["seconds"] = round(elapsed, 4)
            self._conflict_repair_seconds += elapsed
            self._free_gurobi_model(model)

    def _select_repair_groups(
        self,
        selected: Counter[int],
        conflict_keys: tuple[QuantityKey, ...],
    ) -> tuple[str, ...]:
        """Choose a voyage-balanced batch with the largest business loss."""
        areas: defaultdict[str, set[str]] = defaultdict(set)
        rows: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
        nonpreferred: Counter[str] = Counter()
        intrinsic: Counter[str] = Counter()
        for index, value in selected.items():
            if int(value) <= 0:
                continue
            column = self._columns[index]
            group = self.groups_by_id[column.group_id]
            amount = int(value) * int(column.quantity)
            areas[column.group_id].add(column.area_no)
            rows[column.group_id].add(
                (column.bay_key, self._anchor_row(column))
            )
            if not self._is_big_plan_area_for_group(
                group, column.area_no
            ):
                nonpreferred[column.group_id] += amount
            intrinsic[column.group_id] += amount * float(
                column.intrinsic_cost
            )
        conflict_groups = {key[0] for key in conflict_keys}
        scores = {
            group.group_id: (
                20.0 * (group.group_id in conflict_groups)
                + 5.0
                * nonpreferred[group.group_id]
                / max(1, int(group.demand))
                + 2.0 * max(0, len(areas[group.group_id]) - 1)
                + max(0, len(rows[group.group_id]) - 1)
                + intrinsic[group.group_id]
                / max(1, int(group.demand))
            )
            for group in self.groups
        }
        batch_size = min(
            max(1, len(self.groups) - 1),
            12,
            max(6, int(math.ceil(len(self.groups) / 6))),
        )
        groups_by_voyage: defaultdict[str, list[str]] = defaultdict(list)
        for group in self.groups:
            groups_by_voyage[group.voyage_id].append(group.group_id)
        ranking = lambda group_id: (
            self._neighbourhood_group_visits[group_id],
            -scores[group_id],
            group_id,
        )
        quota = max(1, batch_size // max(1, len(groups_by_voyage)))
        chosen = set(conflict_groups)
        for group_ids in groups_by_voyage.values():
            chosen.update(sorted(group_ids, key=ranking)[:quota])
        for group_id in sorted(scores, key=ranking):
            if len(chosen) >= batch_size:
                break
            chosen.add(group_id)
        if len(chosen) > batch_size:
            mandatory = sorted(conflict_groups, key=ranking)[:batch_size]
            other = sorted(set(chosen) - set(mandatory), key=ranking)
            chosen = set(mandatory + other[: batch_size - len(mandatory)])
        return tuple(sorted(chosen))

    def _solve_exact_restricted_primal(
        self,
        selected_hint: Counter[int],
        import_hint: Counter[tuple[str, str, str]],
        conflict_keys: tuple[QuantityKey, ...],
        time_limit: float,
    ) -> dict:
        """Reoptimize one group batch on all rows and fix the remainder."""
        from gurobipy import quicksum

        started = perf_counter()
        free_groups = set(
            self._select_repair_groups(selected_hint, conflict_keys)
        )
        self._neighbourhood_group_visits.update(free_groups)
        global_indices = {
            index
            for (group_id, _bay_key), indices
            in self._candidate_indices_by_group_bay.items()
            if group_id in free_groups
            for index in indices
        }
        global_indices.update(
            index
            for index, value in selected_hint.items()
            if int(value) > 0
        )
        ordered_indices = tuple(sorted(global_indices))
        result = {
            "status": "not_solved",
            "feasible": False,
            "seconds": 0.0,
            "objective": None,
            "selected": Counter(),
            "imports": Counter(),
            "optimized_group_count": len(free_groups),
            "optimized_groups": tuple(sorted(free_groups)),
            "conflict_group_count": len({key[0] for key in conflict_keys}),
            "candidate_row_location_count": len(ordered_indices),
        }
        row_locations = [self._columns[index] for index in ordered_indices]
        model, variables, model_stats = self.build_compact_row_milp(
            row_locations, GurobiModel, quicksum
        )
        result["model"] = model_stats
        try:
            for local_index, global_index in enumerate(ordered_indices):
                variable = variables["column"][local_index]
                start_value = int(selected_hint.get(global_index, 0))
                variable.Start = float(start_value)
                if self._columns[global_index].group_id not in free_groups:
                    variable.LB = float(start_value)
                    variable.UB = float(start_value)
            for key, variable in variables["import_reserve"].items():
                variable.Start = float(import_hint.get(key, 0))
            model.update()
            build_seconds = perf_counter() - started
            result["build_seconds"] = round(build_seconds, 4)
            solve_allowance = float(time_limit) - build_seconds
            if solve_allowance <= 1e-6:
                result["status"] = "time_limit_during_build"
                return result
            self._set_gurobi_param(
                model, "TimeLimit", max(0.01, solve_allowance)
            )
            self._set_gurobi_param(model, "MIPFocus", 1)
            self._set_gurobi_param(model, "Heuristics", 0.75)
            self._try_set_gurobi_param(model, "PumpPasses", 10)
            self._try_set_gurobi_param(model, "RINS", 10)
            model.optimize()
            result["status"] = self._gurobi_status_name(model)
            if self._gurobi_solution_count(model) <= 0:
                return result
            local_selected = self.selected_compact_row_values(
                model, variables
            )
            selected = Counter(
                {
                    ordered_indices[local_index]: int(value)
                    for local_index, value in local_selected.items()
                    if int(value) > 0
                }
            )
            imports = self._gurobi_import_reservation_values(
                model, variables
            )
            previous_imports = self._final_import_reservation
            self._final_import_reservation = Counter(imports)
            try:
                validation = self._validate_final_solution(selected)
                objective = self._selected_solution_energy(selected)
            finally:
                self._final_import_reservation = previous_imports
            result.update(
                {
                    "feasible": True,
                    "selected": selected,
                    "imports": imports,
                    "objective": objective,
                    "validation": validation,
                }
            )
            return result
        finally:
            result["seconds"] = round(perf_counter() - started, 4)
            self._free_gurobi_model(model)

    def _master_round_allowance(
        self,
        master_round: int,
        remaining: float,
        after_feasibility_cut: bool = False,
    ) -> float:
        """Use bounded master slices so cuts, oracle, and repair can alternate."""
        future_work = max(
            self._time_share("primal_polish"),
            2.0 * self._time_share("oracle"),
        )
        reserve = min(0.35 * remaining, future_work)
        if int(master_round) == 1:
            # The first incumbent is a cut-discovery point.  A shorter slice
            # leaves enough time for at least one cut/reoptimize cycle and
            # independent upper-bound improvement under the shared deadline.
            fraction = 0.45
        elif after_feasibility_cut:
            fraction = 0.65
        else:
            fraction = 0.55
        phase = (
            "initial_master"
            if int(master_round) == 1
            else "later_master"
        )
        return max(
            0.01,
            min(
                self._time_share(phase),
                fraction * remaining,
                max(0.01, remaining - reserve),
            ),
        )

    def _oracle_allowance(self, remaining: float) -> float:
        """Allocate oracle time from scale and live remaining time.

        The allocation is proportional to the common experiment budget and
        cannot consume more than half of what remains.  It therefore scales
        without a case-specific per-oracle number of seconds.
        """
        return max(
            0.01,
            min(
                remaining,
                0.50 * remaining,
                self._time_share("oracle"),
            ),
        )

    def _base_profile_diagnostics(self) -> dict:
        diagnostics = super()._base_profile_diagnostics()
        selected_scores = [
            self._selection_scores[state]
            for state in self._selected_profile_states
        ]
        diagnostics.update(
            {
                "algorithm": "selective_resource_state_lbbd_gurobi",
                "formulation": (
                    "integer_group_bay_master_with_selective_profile_and_"
                    "row_count_integrality"
                ),
                "decomposition": (
                    "group_bay_master_joint_exact_row_recourse_conflict_"
                    "repair_and_row_neighbourhood"
                ),
                "benders_cut_validity": (
                    "certified_physical_capacity_or_monotone_IIS_"
                    "feasibility_and_quantity_conditional_optimality"
                ),
                "selective_profile_state_count": len(
                    self._selected_profile_states
                ),
                "selective_profile_state_score_min": (
                    round(min(selected_scores), 6)
                    if selected_scores
                    else None
                ),
                "selective_profile_state_score_max": (
                    round(max(selected_scores), 6)
                    if selected_scores
                    else None
                ),
                "selective_oracle_solve_count": (
                    self._selective_oracle_solve_count
                ),
                "selective_oracle_cache_hits": (
                    self._selective_oracle_cache_hits
                ),
                "selective_oracle_seconds": round(
                    self._selective_oracle_seconds, 3
                ),
                "conflict_repair_solve_count": (
                    self._conflict_repair_solve_count
                ),
                "conflict_repair_success_count": (
                    self._conflict_repair_success_count
                ),
                "conflict_repair_seconds": round(
                    self._conflict_repair_seconds, 3
                ),
                "restricted_primal_solve_count": (
                    self._restricted_primal_solve_count
                ),
                "restricted_primal_success_count": (
                    self._restricted_primal_success_count
                ),
                "restricted_primal_seconds": round(
                    self._restricted_primal_seconds, 3
                ),
                "resource_capacity_certificate_solve_count": (
                    self._resource_capacity_solve_count
                ),
                "resource_capacity_certificate_seconds": round(
                    self._resource_capacity_seconds, 3
                ),
            }
        )
        return diagnostics

    def solve(self):
        """Run one independent master-oracle-cut loop."""
        started = perf_counter()
        total_limit = self._total_time_limit()
        deadline = started + total_limit
        formal_master_reserve = self._time_share(
            "formal_master_reserve"
        )

        self._prepare_decomposition()
        self._prepare_profiles()
        preparation_seconds = perf_counter() - started
        if perf_counter() >= deadline:
            raise RuntimeError(
                "selective LBBD preprocessing consumed the complete time limit"
            )
        master, variables, master_stats = self._build_profile_master()
        master_build_seconds = (
            perf_counter() - started - preparation_seconds
        )
        master_start = self._initialize_master_incumbent(master, deadline)
        if perf_counter() >= deadline:
            self._free_gurobi_model(master)
            raise RuntimeError(
                "selective LBBD model construction consumed the complete "
                "time limit"
            )
        master_start["strategy"] = (
            "relaxed_row_skeleton_then_exact_oracle_and_conflict_repair"
        )

        best_selected: Counter[int] | None = None
        best_import: Counter[tuple[str, str, str]] | None = None
        best_objective = math.inf
        best_master_round = 0
        master_status = "not_solved"
        termination_reason = "iteration_limit"
        converged = False
        cut_signatures: set[tuple] = set()
        cut_records: list[dict] = []
        pending_cut_records: list[int] = []
        feasibility_cut_count = 0
        optimality_cut_count = 0
        cut_binary_count = 0
        master_rounds: list[dict] = []
        oracle_records: list[dict] = []
        repair_records: list[dict] = []
        restricted_primal_records: list[dict] = []
        voyage_bound_records: list[dict] = []
        local_optimality_cut_count = 0
        global_optimality_cut_count = 0
        restricted_primal_limit = min(
            3, max(1, int(math.ceil(len(self.groups) / 30)))
        )
        valid_lower_bound = -math.inf
        primal_start_record = {
            "attempted": False,
            "verified": False,
            "status": "not_attempted",
            "seconds": 0.0,
            "objective": None,
        }
        feasibility_recovery_pending = False

        def accept_candidate(
            selected: Counter[int],
            imports: Counter[tuple[str, str, str]],
            objective: float,
            round_index: int,
        ) -> bool:
            nonlocal best_selected, best_import, best_objective
            nonlocal best_master_round
            if float(objective) + 1e-9 >= best_objective:
                return False
            best_selected = Counter(selected)
            best_import = Counter(imports)
            best_objective = float(objective)
            best_master_round = int(round_index)
            self._final_import_reservation = Counter(imports)
            return True

        def record_oracle(round_index: int, result: dict) -> None:
            oracle_records.append(
                {
                    "master_round": round_index,
                    "status": result["status"],
                    "feasible": bool(result["feasible"]),
                    "optimal": bool(result["optimal"]),
                    "cached": bool(result.get("cached", False)),
                    "seconds": round(float(result["seconds"]), 4),
                    "total_seconds": round(
                        float(result.get("total_seconds", result["seconds"])),
                        4,
                    ),
                    "objective": (
                        float(result["objective"])
                        if result["feasible"]
                        else None
                    ),
                    "bound": (
                        float(result["bound"])
                        if math.isfinite(float(result["bound"]))
                        else None
                    ),
                    "conflict_quantity_count": len(
                        result["conflict_quantity_keys"]
                    ),
                }
            )

        def run_repair(
            round_index: int,
            source: str,
            quantities: dict[QuantityKey, int],
            conflict_keys: tuple[QuantityKey, ...],
            import_hint: Counter[tuple[str, str, str]],
        ) -> dict | None:
            remaining = deadline - perf_counter()
            repair_deadline = deadline
            if round_index == 0 and best_selected is not None:
                repair_deadline = min(
                    deadline, deadline - formal_master_reserve
                )
            available = repair_deadline - perf_counter()
            threshold = self._solve_threshold()
            if (
                remaining <= threshold
                or available <= threshold
                or len(repair_records) >= 4
            ):
                return None
            observed_repair_build = max(
                (
                    float(record.get("build_seconds", 0.0))
                    for record in repair_records
                ),
                default=0.0,
            )
            allowance = (
                min(self._time_share("initial_repair"), available)
                if best_selected is None
                else min(
                    self._time_share("incumbent_repair"),
                    available,
                    0.50 * available,
                )
            )
            if allowance <= observed_repair_build + threshold:
                return None
            phase_deadline = perf_counter() + allowance
            secure_first_feasible = best_selected is None
            solve_window = max(
                4.0 * threshold,
                0.015 * total_limit,
            )
            repair_fraction = (
                0.45 if secure_first_feasible else 0.55
            )
            repair_allowance = min(
                allowance,
                max(
                    repair_fraction * allowance,
                    observed_repair_build + solve_window,
                ),
            )
            repair = self._solve_conflict_directed_repair(
                quantities,
                conflict_keys,
                repair_allowance,
                best_selected,
                best_import or import_hint,
                secure_first_feasible=secure_first_feasible,
            )
            record = {
                key: value
                for key, value in repair.items()
                if key not in {"selected", "imports"}
            }
            record.update(
                {
                    "master_round": round_index,
                    "source": source,
                    "allowance": round(allowance, 4),
                    "feasibility_repair_allowance": round(
                        repair_allowance, 4
                    ),
                    "observed_repair_build_seconds": round(
                        observed_repair_build, 4
                    ),
                    "target_repair_solve_window": round(
                        solve_window, 4
                    ),
                    "improved_incumbent": False,
                }
            )
            if repair["feasible"]:
                polish_allowance = phase_deadline - perf_counter()
                observed_polish_build = max(
                    [
                        float(item.get("build_seconds", 0.0))
                        for item in restricted_primal_records
                    ]
                    + [
                        float(
                            item.get(
                                "restricted_primal_improvement", {}
                            ).get("build_seconds", 0.0)
                        )
                        for item in repair_records
                    ],
                    default=0.0,
                )
                record["remaining_polish_allowance"] = round(
                    max(0.0, polish_allowance), 4
                )
                record["observed_polish_build_seconds"] = round(
                    observed_polish_build, 4
                )
                if polish_allowance > observed_polish_build + threshold:
                    self._restricted_primal_solve_count += 1
                    polish_started = perf_counter()
                    polish = self._solve_exact_restricted_primal(
                        Counter(repair["selected"]),
                        Counter(repair["imports"]),
                        conflict_keys,
                        polish_allowance,
                    )
                    polish_seconds = perf_counter() - polish_started
                    self._restricted_primal_seconds += polish_seconds
                    polish_record = {
                        key: value
                        for key, value in polish.items()
                        if key not in {"selected", "imports"}
                    }
                    record["restricted_primal_improvement"] = polish_record
                    if polish["feasible"]:
                        self._restricted_primal_success_count += 1
                        if (
                            float(polish["objective"])
                            + 1e-9
                            < float(repair["objective"])
                        ):
                            repair = polish
                            record["restricted_primal_selected"] = True
                improved = accept_candidate(
                    Counter(repair["selected"]),
                    Counter(repair["imports"]),
                    float(repair["objective"]),
                    round_index,
                )
                record["improved_incumbent"] = improved
                if best_selected is not None and best_import is not None:
                    self._apply_profile_master_start(
                        variables, best_selected, best_import
                    )
                    master.update()
            repair_records.append(record)
            return repair

        def run_upper_bound_neighbourhood(
            round_index: int,
            source: str,
            conflict_keys: tuple[QuantityKey, ...],
            allowance: float,
            allow_terminal_extra: bool = False,
        ) -> dict | None:
            observed_build_time = max(
                (
                    float(record.get("build_seconds", 0.0))
                    for record in restricted_primal_records
                ),
                default=0.0,
            )
            if (
                allowance <= self._solve_threshold()
                or allowance
                <= observed_build_time + self._solve_threshold()
                or best_selected is None
                or best_import is None
                or (
                    not allow_terminal_extra
                    and self._restricted_primal_solve_count
                    >= restricted_primal_limit
                )
            ):
                return None
            self._restricted_primal_solve_count += 1
            polish_started = perf_counter()
            polish = self._solve_exact_restricted_primal(
                best_selected,
                best_import,
                conflict_keys,
                allowance,
            )
            self._restricted_primal_seconds += (
                perf_counter() - polish_started
            )
            record = {
                key: value
                for key, value in polish.items()
                if key not in {"selected", "imports"}
            }
            record.update(
                {
                    "master_round": round_index,
                    "source": source,
                    "allowance": round(allowance, 4),
                    "improved_incumbent": False,
                }
            )
            if polish["feasible"]:
                self._restricted_primal_success_count += 1
                record["improved_incumbent"] = accept_candidate(
                    Counter(polish["selected"]),
                    Counter(polish["imports"]),
                    float(polish["objective"]),
                    round_index,
                )
            restricted_primal_records.append(record)
            return record

        try:
            if master_start.get("feasible", False):
                quantities, _counts, _routing, imports, _theta = (
                    self._profile_master_start_assignment(variables)
                )
                remaining = deadline - perf_counter()
                if remaining > 1e-6:
                    primal_start_record["attempted"] = True
                    warm_recourse = self._solve_exact_recourse(
                        quantities,
                        self._oracle_allowance(remaining),
                    )
                    record_oracle(0, warm_recourse)
                    primal_start_record.update(
                        {
                            "status": warm_recourse["status"],
                            "seconds": round(
                                float(warm_recourse["seconds"]), 4
                            ),
                            "feasible": bool(warm_recourse["feasible"]),
                            "optimal": bool(warm_recourse["optimal"]),
                        }
                    )
                    if warm_recourse["feasible"]:
                        warm_selected = Counter(
                            warm_recourse["selected"]
                        )
                        self._validate_exact_quantities(
                            quantities, warm_selected
                        )
                        self._final_import_reservation = Counter(imports)
                        warm_objective = self._selected_solution_energy(
                            warm_selected
                        )
                        accept_candidate(
                            warm_selected,
                            Counter(imports),
                            warm_objective,
                            0,
                        )
                        primal_start_record["verified"] = True
                        primal_start_record["objective"] = warm_objective
                        self._apply_profile_master_start(
                            variables, best_selected, best_import
                        )
                        master.update()
                    elif warm_recourse["status"] == "infeasible":
                        conflict_keys = tuple(
                            warm_recourse["conflict_quantity_keys"]
                        )
                        cut_info = self._add_certified_feasibility_cut(
                            master,
                            variables,
                            quantities,
                            conflict_keys,
                            feasibility_cut_count + 1,
                            cut_signatures,
                        )
                        if cut_info["added"]:
                            binary_count = int(
                                cut_info["binary_count"]
                            )
                            feasibility_cut_count += 1
                            cut_binary_count += binary_count
                            cut_records.append(
                                {
                                    "cut_index": len(cut_records) + 1,
                                    "master_round": 0,
                                    "kind": cut_info["kind"],
                                    "binary_count": binary_count,
                                    "theta_before": 0.0,
                                    "recourse_lower_bound": None,
                                    "lower_bound_before": None,
                                    "capacity_upper_bound": cut_info[
                                        "capacity_upper_bound"
                                    ],
                                    "capacity_core": cut_info[
                                        "capacity_core"
                                    ],
                                    "incumbent_core_quantity": cut_info[
                                        "incumbent_core_quantity"
                                    ],
                                    "capacity_certificate": cut_info[
                                        "capacity_certificate"
                                    ],
                                    "monotone_support_limit": cut_info.get(
                                        "monotone_support_limit"
                                    ),
                                }
                            )
                            pending_cut_records.append(
                                len(cut_records) - 1
                            )
                            feasibility_recovery_pending = True
                        repair = run_repair(
                            0,
                            "initial_infeasible_skeleton",
                            quantities,
                            conflict_keys,
                            Counter(imports),
                        )
                        if repair and repair["feasible"]:
                            primal_start_record["verified"] = True
                            primal_start_record["repaired"] = True
                            primal_start_record["objective"] = repair[
                                "objective"
                            ]
                    else:
                        # A timed oracle supplies no infeasibility proof, so
                        # it must not generate a cut.  It can still seed the
                        # independent exact repair with the positive support;
                        # only a fully validated repair is accepted as an
                        # incumbent and MIP Start.
                        support_keys = tuple(
                            sorted(
                                key
                                for key, value in quantities.items()
                                if int(value) > 0
                            )
                        )
                        repair = run_repair(
                            0,
                            "initial_uncertified_skeleton",
                            quantities,
                            support_keys,
                            Counter(imports),
                        )
                        if repair and repair["feasible"]:
                            primal_start_record["verified"] = True
                            primal_start_record["repaired"] = True
                            primal_start_record["objective"] = repair[
                                "objective"
                            ]

            for master_round in range(
                1, int(self.benders_config.max_iterations) + 1
            ):
                remaining = deadline - perf_counter()
                if remaining <= 1e-6:
                    termination_reason = "time_limit"
                    break
                master_allowance = self._master_round_allowance(
                    master_round,
                    remaining,
                    after_feasibility_cut=(
                        feasibility_recovery_pending
                    ),
                )
                if master_allowance <= 0.05:
                    termination_reason = "time_limit"
                    break
                self._set_gurobi_param(
                    master, "TimeLimit", master_allowance
                )
                master_started = perf_counter()
                master.optimize()
                feasibility_recovery_pending = False
                master_seconds = perf_counter() - master_started
                master_status = self._gurobi_status_name(master)
                if self._gurobi_solution_count(master) <= 0:
                    master_bound = self._gurobi_dual_bound(master)
                    if math.isfinite(master_bound):
                        valid_lower_bound = max(
                            valid_lower_bound, master_bound
                        )
                    for record_index in pending_cut_records:
                        record = cut_records[record_index]
                        record["lower_bound_after_next_master"] = (
                            valid_lower_bound
                        )
                        before = record.get("lower_bound_before")
                        record["certified_lower_bound_lift"] = (
                            max(0.0, valid_lower_bound - float(before))
                            if before is not None
                            else None
                        )
                    pending_cut_records.clear()
                    master_rounds.append(
                        {
                            "master_round": master_round,
                            "master_status": master_status,
                            "master_allowance": round(
                                master_allowance, 4
                            ),
                            "master_seconds": round(master_seconds, 4),
                            "master_objective": None,
                            "master_bound": master_bound,
                            "cuts_added": 0,
                            "cut_kind": None,
                            "candidate_objective": None,
                        }
                    )
                    if (
                        master_status == "timelimit"
                        and deadline - perf_counter() > 1e-6
                    ):
                        termination_reason = "master_search_continues"
                        continue
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
                for record_index in pending_cut_records:
                    record = cut_records[record_index]
                    record["lower_bound_after_next_master"] = (
                        valid_lower_bound
                    )
                    before = record.get("lower_bound_before")
                    record["certified_lower_bound_lift"] = (
                        max(0.0, valid_lower_bound - float(before))
                        if before is not None
                        else None
                    )
                pending_cut_records.clear()

                master_objective = self._gurobi_objective_value(master)
                quantities, _counts, _routing, imports, theta = (
                    self._profile_master_assignment(master, variables)
                )
                theta_total = float(sum(theta.values()))
                self._final_import_reservation = Counter(imports)
                remaining = deadline - perf_counter()
                if remaining <= 1e-6:
                    termination_reason = "time_limit_before_recourse"
                    break
                recourse = self._solve_exact_recourse(
                    quantities,
                    self._oracle_allowance(remaining),
                    best_selected,
                )
                record_oracle(master_round, recourse)

                selected = Counter()
                candidate_objective = None
                recourse_objective = None
                cuts_added = 0
                cut_kinds: list[str] = []
                if recourse["feasible"]:
                    selected = Counter(recourse["selected"])
                    self._validate_exact_quantities(quantities, selected)
                    recourse_objective = float(recourse["objective"])
                    candidate_objective = self._selected_solution_energy(
                        selected
                    )
                    decomposition_objective = (
                        master_objective
                        - theta_total
                        + recourse_objective
                    )
                    self._absolute_deviation_auxiliary_slack(
                        decomposition_objective,
                        candidate_objective,
                        context="selective LBBD decomposition",
                    )
                    accept_candidate(
                        selected,
                        Counter(imports),
                        candidate_objective,
                        master_round,
                    )

                recourse_bound = float(recourse["bound"])
                new_cut_details = []
                local_cut_added = False
                if recourse["feasible"]:
                    row_cost_by_voyage = (
                        self._selected_row_cost_by_voyage(selected)
                    )
                    voyage_candidates = sorted(
                        (
                            (
                                float(row_cost_by_voyage.get(voyage_id, 0.0))
                                - float(theta.get(voyage_id, 0.0)),
                                voyage_id,
                            )
                            for voyage_id in self._voyages
                            if float(row_cost_by_voyage.get(voyage_id, 0.0))
                            > float(theta.get(voyage_id, 0.0)) + 1e-7
                        ),
                        key=lambda item: (-item[0], item[1]),
                    )[:3]
                    local_budget = min(
                        self._time_share("local_bounds"),
                        max(
                            0.0,
                            deadline
                            - perf_counter()
                            - self._time_share("safety_margin"),
                        ),
                    )
                    for position, (estimated_gap, voyage_id) in enumerate(
                        voyage_candidates
                    ):
                        remaining_candidates = (
                            len(voyage_candidates) - position
                        )
                        allowance = local_budget / max(
                            1, remaining_candidates
                        )
                        if allowance <= self._solve_threshold():
                            break
                        bound_result = self._solve_voyage_recourse_bound(
                            voyage_id,
                            quantities,
                            allowance,
                            selected,
                        )
                        local_budget = max(
                            0.0,
                            local_budget - float(bound_result["seconds"]),
                        )
                        voyage_bound = float(bound_result["bound"])
                        bound_record = {
                            "master_round": master_round,
                            "voyage_id": voyage_id,
                            "status": bound_result["status"],
                            "optimal": bool(bound_result["optimal"]),
                            "feasible": bool(bound_result["feasible"]),
                            "cached": bool(bound_result.get("cached", False)),
                            "seconds": round(
                                float(bound_result["seconds"]), 4
                            ),
                            "estimated_component_gap": estimated_gap,
                            "theta_before": float(theta[voyage_id]),
                            "lower_bound": (
                                voyage_bound
                                if math.isfinite(voyage_bound)
                                else None
                            ),
                            "cut_added": False,
                        }
                        if (
                            math.isfinite(voyage_bound)
                            and voyage_bound
                            > float(theta[voyage_id]) + 1e-7
                        ):
                            added, binary_count, support_count = (
                                self._add_voyage_optimality_cut(
                                    master,
                                    variables,
                                    voyage_id,
                                    quantities,
                                    voyage_bound,
                                    optimality_cut_count + 1,
                                    cut_signatures,
                                )
                            )
                            bound_record["support_quantity_count"] = (
                                support_count
                            )
                            if added:
                                optimality_cut_count += 1
                                local_optimality_cut_count += 1
                                local_cut_added = True
                                bound_record["cut_added"] = True
                                new_cut_details.append(
                                    {
                                        "kind": (
                                            "voyage_quantity_recourse_"
                                            "optimality"
                                        ),
                                        "binary_count": binary_count,
                                        "support_quantity_count": (
                                            support_count
                                        ),
                                        "theta_before": float(
                                            theta[voyage_id]
                                        ),
                                        "recourse_lower_bound": (
                                            voyage_bound
                                        ),
                                        "component_id": voyage_id,
                                    }
                                )
                        voyage_bound_records.append(bound_record)
                if recourse["status"] == "infeasible":
                    cut_info = self._add_certified_feasibility_cut(
                        master,
                        variables,
                        quantities,
                        tuple(recourse["conflict_quantity_keys"]),
                        feasibility_cut_count + 1,
                        cut_signatures,
                    )
                    if cut_info["added"]:
                        feasibility_cut_count += 1
                        feasibility_recovery_pending = True
                        new_cut_details.append(
                            {
                                "kind": cut_info["kind"],
                                "binary_count": int(
                                    cut_info["binary_count"]
                                ),
                                "theta_before": 0.0,
                                "recourse_lower_bound": None,
                                "component_id": recourse.get(
                                    "conflict_component_id"
                                ),
                                "capacity_upper_bound": cut_info[
                                    "capacity_upper_bound"
                                ],
                                "capacity_core": cut_info[
                                    "capacity_core"
                                ],
                                "incumbent_core_quantity": cut_info[
                                    "incumbent_core_quantity"
                                ],
                                "capacity_certificate": cut_info[
                                    "capacity_certificate"
                                ],
                                "monotone_support_limit": cut_info.get(
                                    "monotone_support_limit"
                                ),
                            }
                        )
                elif (
                    recourse["feasible"]
                    and not local_cut_added
                    and math.isfinite(recourse_bound)
                    and recourse_bound > theta_total + 1e-7
                ):
                    added, binary_count = self._add_aggregate_logic_cut(
                        master,
                        variables,
                        quantities,
                        {},
                        optimality_cut_count + 1,
                        "optimality",
                        cut_signatures,
                        lower_bound=recourse_bound,
                    )
                    if added:
                        optimality_cut_count += 1
                        global_optimality_cut_count += 1
                        new_cut_details.append(
                            {
                                "kind": "quantity_recourse_optimality",
                                "binary_count": binary_count,
                                "theta_before": theta_total,
                                "recourse_lower_bound": recourse_bound,
                                "component_id": None,
                            }
                        )

                for detail in new_cut_details:
                    cuts_added += 1
                    cut_kinds.append(detail["kind"])
                    cut_binary_count += int(detail["binary_count"])
                    cut_records.append(
                        {
                            "cut_index": len(cut_records) + 1,
                            "master_round": master_round,
                            **detail,
                            "lower_bound_before": (
                                valid_lower_bound
                                if math.isfinite(valid_lower_bound)
                                else None
                            ),
                        }
                    )
                    pending_cut_records.append(len(cut_records) - 1)

                if (
                    recourse["feasible"]
                    and not any(
                        detail["kind"]
                        in {
                            "IIS_physical_resource_capacity",
                            "monotone_IIS_capacity_feasibility",
                        }
                        for detail in new_cut_details
                    )
                    and self._restricted_primal_solve_count
                    < restricted_primal_limit
                ):
                    remaining_for_polish = deadline - perf_counter()
                    run_upper_bound_neighbourhood(
                        master_round,
                        "post_oracle_upper_bound_polish",
                        (),
                        min(
                            self._time_share("primal_polish"),
                            max(
                                0.0,
                                remaining_for_polish
                                - self._time_share("safety_margin"),
                            ),
                        ),
                    )
                repair = None
                if not recourse["feasible"]:
                    repair_keys = tuple(
                        recourse["conflict_quantity_keys"]
                    ) or tuple(
                        sorted(
                            key
                            for key, value in quantities.items()
                            if int(value) > 0
                        )
                    )
                    repair = run_repair(
                        master_round,
                        (
                            "oracle_IIS_conflict"
                            if recourse["status"] == "infeasible"
                            else "oracle_uncertified_point"
                        ),
                        quantities,
                        repair_keys,
                        Counter(imports),
                    )
                    if repair and repair["feasible"]:
                        candidate_objective = float(repair["objective"])
                cut_kind = "+".join(cut_kinds) if cut_kinds else None
                if best_selected is not None and best_import is not None:
                    self._apply_profile_master_start(
                        variables, best_selected, best_import
                    )
                    master.update()

                master_rounds.append(
                    {
                        "master_round": master_round,
                        "master_status": master_status,
                        "master_allowance": round(master_allowance, 4),
                        "master_seconds": round(master_seconds, 4),
                        "master_objective": master_objective,
                        "master_bound": master_bound,
                        "theta_total": theta_total,
                        "recourse_status": recourse["status"],
                        "recourse_feasible": bool(recourse["feasible"]),
                        "recourse_optimal": bool(recourse["optimal"]),
                        "recourse_objective": recourse_objective,
                        "recourse_bound": recourse_bound,
                        "cuts_added": cuts_added,
                        "cut_kind": cut_kind,
                        "candidate_objective": candidate_objective,
                        "repair_status": (
                            repair["status"] if repair else None
                        ),
                    }
                )

                if cuts_added:
                    termination_reason = "master_strengthened"
                    continue
                if not recourse["feasible"]:
                    termination_reason = "recourse_without_certificate"
                    break
                if (
                    master_status == "optimal"
                    and recourse["optimal"]
                    and recourse_objective is not None
                    and recourse_objective <= theta_total + 1e-7
                ):
                    converged = True
                    termination_reason = "optimality_proven"
                    break
                termination_reason = "time_limited_master_or_recourse"
                break
            else:
                termination_reason = "iteration_limit"

            terminal_remaining = deadline - perf_counter()
            if (
                not converged
                and best_selected is not None
                and best_import is not None
                and terminal_remaining > self._solve_threshold()
            ):
                run_upper_bound_neighbourhood(
                    max(1, len(master_rounds)),
                    "terminal_remaining_time_polish",
                    (),
                    max(
                        0.0,
                        terminal_remaining
                        - self._time_share("safety_margin"),
                    ),
                    allow_terminal_extra=True,
                )

            if best_selected is None or best_import is None:
                raise RuntimeError(
                    "selective LBBD did not find a complete row allocation; "
                    f"termination={termination_reason}, "
                    f"master_status={master_status}, "
                    f"master_start={master_start}, "
                    f"rounds={master_rounds}, oracle={oracle_records}"
                )
            self._final_import_reservation = best_import
            if converged:
                valid_lower_bound = best_objective
            if not math.isfinite(valid_lower_bound):
                valid_lower_bound = 0.0
            valid_lower_bound = min(valid_lower_bound, best_objective)
            absolute_gap = max(0.0, best_objective - valid_lower_bound)
            relative_gap = absolute_gap / max(abs(best_objective), 1e-12)
            tightened_cut_count = sum(
                record.get("certified_lower_bound_lift") is not None
                and float(record["certified_lower_bound_lift"]) > 1e-9
                for record in cut_records
            )
            diagnostics = {
                **self._base_profile_diagnostics(),
                "master_algorithm": "selective_resource_state_lbbd",
                "master_status": "optimal" if converged else master_status,
                "master_bound_scope": "valid_selective_master_relaxation",
                "master_objective": best_objective,
                "master_mip_gap": relative_gap,
                "complete_model_lower_bound": valid_lower_bound,
                "complete_model_absolute_gap": absolute_gap,
                "complete_model_relative_gap": relative_gap,
                "complete_model_gap_source": "selective_master_bound",
                "hard_demand_balance": True,
                "candidate_row_location_count": len(self._columns),
                "selected_location_count": len(best_selected),
                "selective_lbbd_converged": converged,
                "selective_lbbd_termination_reason": termination_reason,
                "selective_lbbd_master_round_count": len(master_rounds),
                "selective_lbbd_best_master_round": best_master_round,
                "selective_lbbd_master_rounds": master_rounds,
                "selective_lbbd_feasibility_cut_count": (
                    feasibility_cut_count
                ),
                "selective_lbbd_optimality_cut_count": (
                    optimality_cut_count
                ),
                "selective_lbbd_local_optimality_cut_count": (
                    local_optimality_cut_count
                ),
                "selective_lbbd_global_optimality_cut_count": (
                    global_optimality_cut_count
                ),
                "selective_lbbd_cut_binary_count": cut_binary_count,
                "selective_lbbd_cut_records": cut_records,
                "selective_lbbd_bound_tightening_cut_count": (
                    tightened_cut_count
                ),
                "selective_lbbd_oracle_records": oracle_records,
                "selective_lbbd_voyage_bound_records": (
                    voyage_bound_records
                ),
                "selective_lbbd_voyage_bound_solve_count": (
                    self._voyage_bound_solve_count
                ),
                "selective_lbbd_voyage_bound_cache_hits": (
                    self._voyage_bound_cache_hits
                ),
                "selective_lbbd_voyage_bound_seconds": round(
                    self._voyage_bound_seconds, 3
                ),
                "selective_lbbd_repair_records": repair_records,
                "selective_lbbd_restricted_primal_records": (
                    restricted_primal_records
                ),
                "selective_lbbd_repair_solve_count": (
                    self._conflict_repair_solve_count
                ),
                "selective_lbbd_repair_success_count": (
                    self._conflict_repair_success_count
                ),
                "selective_lbbd_repair_seconds": round(
                    self._conflict_repair_seconds, 3
                ),
                "selective_lbbd_restricted_primal_solve_count": (
                    self._restricted_primal_solve_count
                ),
                "selective_lbbd_restricted_primal_success_count": (
                    self._restricted_primal_success_count
                ),
                "selective_lbbd_restricted_primal_seconds": round(
                    self._restricted_primal_seconds, 3
                ),
                "selective_lbbd_restricted_primal_solve_limit": (
                    restricted_primal_limit
                ),
                "selective_lbbd_oracle_build_seconds": round(
                    self._selective_oracle_build_seconds, 3
                ),
                "selective_lbbd_oracle_variable_count": (
                    self._selective_oracle_variable_count
                ),
                "selective_lbbd_preparation_seconds": round(
                    preparation_seconds, 3
                ),
                "selective_lbbd_master_build_seconds": round(
                    master_build_seconds, 3
                ),
                "selective_lbbd_master_start": master_start,
                "selective_lbbd_primal_start": primal_start_record,
                "selective_lbbd_total_solve_seconds": round(
                    perf_counter() - started, 3
                ),
                "selective_lbbd_time_budget_policy": {
                    "kind": "shared_deadline_proportional_feedback",
                    "total_time_limit": total_limit,
                    "phase_share_upper_bounds": dict(self._TIME_SHARES),
                    "unused_time_reallocated": True,
                    "terminal_unused_time_goes_to_primal_polish": True,
                    "sparse_oracle_build_included_in_allowance": True,
                    "repair_budget_uses_observed_build_time": True,
                    "first_master_is_cut_discovery_slice": True,
                    "adaptive_monotone_iis_support_limit": (
                        self._monotone_iis_support_limit()
                    ),
                    "uncertified_timeout_generates_cut": False,
                },
                "selective_lbbd_master": master_stats,
            }
            result = self._assemble_result(best_selected, diagnostics)
            result.columns = self._selected_lbbd_columns(best_selected)
            return result
        finally:
            self._free_gurobi_model(master)
            for subproblem in self._voyage_subproblems.values():
                self._free_gurobi_model(subproblem.model)


__all__ = ["SelectiveResourceBendersPlanner"]
