"""V7.1 Stage 1 coarse group-to-area MIP and restricted-domain construction."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median
from time import perf_counter
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .gurobi_backend import GurobiModel
from .models import ProblemData
from .v7_atoms import V7RowAtom, build_v7_row_atoms
from .v7_model import (
    V7_MODEL_SCHEMA_VERSION,
    V7ModelEvaluator,
    V7ObjectiveConfig,
    V7PeakUtilizationPolicy,
)


class V7Stage1IncompleteError(RuntimeError):
    def __init__(self, message: str, diagnostics: Mapping[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class V7Stage1Config:
    time_limit: float = 10.0
    maximum_pool_solutions: int = 8
    pool_gap: float = 0.10
    additional_candidate_area_cap: int = 5
    maximum_pool_candidate_areas: int = 1
    solver_threads: int = 1
    solver_seed: int = 0
    verbose: bool = False
    objective: V7ObjectiveConfig = field(default_factory=V7ObjectiveConfig)

    def validate(self) -> None:
        if not math.isfinite(float(self.time_limit)) or float(self.time_limit) <= 0:
            raise ValueError("V7 Stage-1 time limit must be positive")
        if int(self.maximum_pool_solutions) <= 0:
            raise ValueError("V7 Stage-1 solution-pool size must be positive")
        if not math.isfinite(float(self.pool_gap)) or float(self.pool_gap) < 0:
            raise ValueError("V7 Stage-1 pool gap must be nonnegative")
        if int(self.additional_candidate_area_cap) <= 0:
            raise ValueError("V7 Stage-1 additional area cap must be positive")
        if not 0 <= int(self.maximum_pool_candidate_areas) <= int(
            self.additional_candidate_area_cap
        ):
            raise ValueError(
                "maximum pool-candidate areas must lie within the additional cap"
            )
        if int(self.solver_threads) < 0:
            raise ValueError("V7 solver_threads must be nonnegative")
        self.objective.validate()


@dataclass(frozen=True)
class V7Stage1PoolSolution:
    solution_number: int
    objective: float
    group_area_quantity: Mapping[tuple[str, str], int]
    group_area_bay_incidence: Mapping[tuple[str, str], int]
    used_areas_by_group: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class V7Stage1Result:
    best_group_area_quantity: Mapping[tuple[str, str], int]
    restricted_areas_by_group: Mapping[str, frozenset[str]]
    pool_solutions: tuple[V7Stage1PoolSolution, ...]
    diagnostics: Mapping[str, object]

    @property
    def quota_fixed_in_stage2(self) -> bool:
        return False


@dataclass(frozen=True)
class V7Stage1PackingMetadata:
    reachable_capacity: Mapping[tuple[str, str], int]
    compatible_anchor_bay_count: Mapping[tuple[str, str], int]
    max_single_anchor_capacity: Mapping[tuple[str, str], int]
    usable_physical_bay_count: Mapping[str, int]
    footprint_width_by_group: Mapping[str, int]


@dataclass(frozen=True)
class V7Stage1BayLocalAreaCandidate:
    group_id: str
    area: str
    shortage_boxes: int
    supplied_boxes: int
    local_objective: float
    used_anchor_bays: tuple[str, ...]
    used_anchor_bay_count: int
    normalized_span: float
    flow_cost: float
    extra_bay_cost: float
    span_cost: float


def freeze_restricted_areas(
    areas_by_group: Mapping[str, Iterable[str]],
) -> Mapping[str, frozenset[str]]:
    """Return an immutable normalized candidate-area mapping."""

    return MappingProxyType(
        {
            str(group_id): frozenset(str(area) for area in areas)
            for group_id, areas in sorted(areas_by_group.items())
        }
    )


def build_v7_stage1_packing_metadata(
    problem: ProblemData,
    atoms: Sequence[V7RowAtom],
) -> V7Stage1PackingMetadata:
    """Build the simple area packing metadata used by the V7.1 coarse MIP."""

    groups_by_id = {str(group.group_id): group for group in problem.export_groups}
    capacity_by_anchor: Counter[tuple[str, str, str]] = Counter()
    physical_by_anchor: dict[tuple[str, str, str], tuple[str, ...]] = {}
    physical_bays_by_area: defaultdict[str, set[str]] = defaultdict(set)
    for atom in atoms:
        key = (atom.group_id, atom.area_no, atom.anchor_bay_key)
        capacity_by_anchor[key] += int(atom.capacity)
        physical_by_anchor[key] = atom.physical_bays
        physical_bays_by_area[atom.area_no].update(atom.physical_bays)

    anchors_by_pair: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for (group_id, area, anchor), capacity in capacity_by_anchor.items():
        group = groups_by_id[group_id]
        capacity = min(
            int(capacity),
            int(problem.bays[anchor].cap_by_size.get(str(group.size), 0)),
            *(
                int(problem.bays[physical].physical_capacity)
                for physical in physical_by_anchor[(group_id, area, anchor)]
            ),
        )
        if capacity > 0:
            anchors_by_pair[(group_id, area)].append(int(capacity))

    reachable: dict[tuple[str, str], int] = {}
    compatible_count: dict[tuple[str, str], int] = {}
    maximum: dict[tuple[str, str], int] = {}
    for pair, capacities in sorted(anchors_by_pair.items()):
        reachable[pair] = sum(capacities)
        compatible_count[pair] = len(capacities)
        maximum[pair] = max(capacities)

    return V7Stage1PackingMetadata(
        reachable_capacity=MappingProxyType(reachable),
        compatible_anchor_bay_count=MappingProxyType(compatible_count),
        max_single_anchor_capacity=MappingProxyType(maximum),
        usable_physical_bay_count=MappingProxyType(
            {
                area: len(bays)
                for area, bays in sorted(physical_bays_by_area.items())
            }
        ),
        footprint_width_by_group=MappingProxyType(
            {
                group_id: 2 if str(group.size) in {"40", "45"} else 1
                for group_id, group in sorted(groups_by_id.items())
            }
        ),
    )


def build_v7_bay_local_area_candidates(
    problem: ProblemData,
    atoms: Sequence[V7RowAtom],
    evaluator: V7ModelEvaluator,
) -> tuple[
    Mapping[str, tuple[str, ...]],
    Mapping[tuple[str, str], V7Stage1BayLocalAreaCandidate],
]:
    """Rank areas by a deterministic single-group bay-level placement proxy.

    The proxy opens actual compatible anchor bays and charges the same flow,
    extra-bay, and within-area span coefficients as the V7 objective.  It is a
    candidate-ranking oracle only: it never fixes Stage-2 quantities or claims
    joint feasibility across groups.
    """

    groups_by_id = {str(group.group_id): group for group in problem.export_groups}
    capacity_by_anchor: Counter[tuple[str, str, str]] = Counter()
    physical_by_anchor: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for atom in atoms:
        key = (atom.group_id, atom.area_no, atom.anchor_bay_key)
        capacity_by_anchor[key] += int(atom.capacity)
        physical_by_anchor[key] = atom.physical_bays
    anchor_capacity: dict[tuple[str, str, str], int] = {}
    for key, capacity in capacity_by_anchor.items():
        group_id, _area, anchor = key
        group = groups_by_id[group_id]
        feasible = min(
            int(capacity),
            int(problem.bays[anchor].cap_by_size.get(str(group.size), 0)),
            *(
                int(problem.bays[physical].physical_capacity)
                for physical in physical_by_anchor[key]
            ),
        )
        if feasible > 0:
            anchor_capacity[key] = feasible

    by_pair: defaultdict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for (group_id, area, anchor), capacity in sorted(anchor_capacity.items()):
        by_pair[(group_id, area)].append((anchor, int(capacity)))
    bay_use_cost = evaluator.group_bay_use_objective_coefficient()
    span_weight = evaluator.group_area_span_objective_coefficient()
    candidates: dict[tuple[str, str], V7Stage1BayLocalAreaCandidate] = {}
    for (group_id, area), anchors in sorted(by_pair.items()):
        demand = int(groups_by_id[group_id].demand)

        def flow_coefficient(item: tuple[str, int]) -> float:
            return evaluator.group_bay_flow_objective_coefficient(
                group_id, item[0]
            )

        orderings = (
            sorted(
                anchors,
                key=lambda item: (
                    flow_coefficient(item),
                    -int(item[1]),
                    int(problem.bays[item[0]].bay_order),
                    item[0],
                ),
            ),
            sorted(
                anchors,
                key=lambda item: (
                    flow_coefficient(item) + bay_use_cost / max(1, int(item[1])),
                    -int(item[1]),
                    int(problem.bays[item[0]].bay_order),
                    item[0],
                ),
            ),
            sorted(
                anchors,
                key=lambda item: (
                    int(problem.bays[item[0]].bay_order),
                    flow_coefficient(item),
                    -int(item[1]),
                    item[0],
                ),
            ),
        )
        plans: list[V7Stage1BayLocalAreaCandidate] = []
        for ordering in orderings:
            remaining = demand
            used: list[tuple[str, int]] = []
            flow_cost = 0.0
            for anchor, capacity in ordering:
                if remaining <= 0:
                    break
                quantity = min(remaining, int(capacity))
                if quantity <= 0:
                    continue
                used.append((anchor, quantity))
                flow_cost += quantity * flow_coefficient((anchor, capacity))
                remaining -= quantity
            normalized_orders = [
                evaluator.normalized_bay_order(group_id, anchor)
                for anchor, _quantity in used
            ]
            span = (
                max(normalized_orders) - min(normalized_orders)
                if normalized_orders
                else 0.0
            )
            extra_bay_cost = max(0, len(used) - 1) * bay_use_cost
            span_cost = span * span_weight
            plans.append(
                V7Stage1BayLocalAreaCandidate(
                    group_id=group_id,
                    area=area,
                    shortage_boxes=max(0, remaining),
                    supplied_boxes=demand - max(0, remaining),
                    local_objective=flow_cost + extra_bay_cost + span_cost,
                    used_anchor_bays=tuple(anchor for anchor, _quantity in used),
                    used_anchor_bay_count=len(used),
                    normalized_span=span,
                    flow_cost=flow_cost,
                    extra_bay_cost=extra_bay_cost,
                    span_cost=span_cost,
                )
            )
        candidates[(group_id, area)] = min(
            plans,
            key=lambda plan: (
                int(plan.shortage_boxes),
                float(plan.local_objective),
                int(plan.used_anchor_bay_count),
                plan.used_anchor_bays,
            ),
        )

    rankings: dict[str, tuple[str, ...]] = {}
    for group_id in sorted(groups_by_id):
        group_candidates = [
            candidate
            for (candidate_group, _area), candidate in candidates.items()
            if candidate_group == group_id
        ]
        rankings[group_id] = tuple(
            candidate.area
            for candidate in sorted(
                group_candidates,
                key=lambda candidate: (
                    int(candidate.shortage_boxes),
                    float(candidate.local_objective),
                    int(candidate.used_anchor_bay_count),
                    candidate.area,
                ),
            )
        )
    return MappingProxyType(rankings), MappingProxyType(candidates)


def build_v7_restricted_area_domain(
    group_ids: Iterable[str],
    legal_areas_by_group: Mapping[str, Iterable[str]],
    pool: Sequence[V7Stage1PoolSolution],
    *,
    additional_candidate_area_cap: int,
    maximum_pool_candidate_areas: int = 1,
    bay_local_areas_by_group: Mapping[str, Iterable[str]] | None = None,
    ranking_tail_by_pair: Mapping[tuple[str, str], tuple[object, ...]] | None = None,
) -> tuple[Mapping[str, frozenset[str]], Mapping[str, object]]:
    """Combine mandatory Stage-1 best support and capped alternatives."""

    if not pool:
        raise ValueError("V7.1 restricted-area construction requires a Stage-1 pool")
    cap = int(additional_candidate_area_cap)
    if cap <= 0:
        raise ValueError("additional_candidate_area_cap must be positive")
    pool_maximum = int(maximum_pool_candidate_areas)
    if not 0 <= pool_maximum <= cap:
        raise ValueError("maximum_pool_candidate_areas must lie within the cap")
    legal = {
        str(group_id): {str(area) for area in areas}
        for group_id, areas in legal_areas_by_group.items()
    }
    bay_local = {
        str(group_id): [str(area) for area in areas]
        for group_id, areas in (bay_local_areas_by_group or {}).items()
    }
    best = pool[0]
    frequency: Counter[tuple[str, str]] = Counter()
    mass: Counter[tuple[str, str]] = Counter()
    first_rank: dict[tuple[str, str], int] = {}
    pool_union: defaultdict[str, set[str]] = defaultdict(set)
    for rank, solution in enumerate(pool):
        for pair, quantity in solution.group_area_quantity.items():
            pair = (str(pair[0]), str(pair[1]))
            if int(quantity) <= 0:
                continue
            frequency[pair] += 1
            mass[pair] += int(quantity)
            first_rank.setdefault(pair, rank)
            pool_union[pair[0]].add(pair[1])

    tails = ranking_tail_by_pair or {}
    selected: dict[str, set[str]] = {}
    group_diagnostics: list[dict[str, object]] = []
    for raw_group_id in sorted(str(value) for value in group_ids):
        group_id = str(raw_group_id)
        legal_areas = legal.get(group_id, set())
        best_areas = set(best.used_areas_by_group.get(group_id, ()))
        illegal_mandatory = best_areas - legal_areas
        if illegal_mandatory:
            raise ValueError(
                f"V7.1 mandatory support is outside the legal atom domain for "
                f"{group_id}: {sorted(illegal_mandatory)}"
            )
        mandatory = best_areas
        pool_optional = (pool_union.get(group_id, set()) - mandatory) & legal_areas
        ranked_pool = sorted(
            pool_optional,
            key=lambda area: (
                -frequency[(group_id, area)],
                -mass[(group_id, area)],
                first_rank.get((group_id, area), len(pool)),
                *tails.get((group_id, area), ()),
                area,
            ),
        )
        ranked_bay_local = [
            area
            for area in bay_local.get(group_id, ())
            if area in legal_areas and area not in mandatory
        ]
        ranked_bay_local.extend(
            area
            for area in sorted(
                legal_areas - mandatory - set(ranked_bay_local),
                key=lambda area: (
                    *tails.get((group_id, area), ()),
                    area,
                ),
            )
        )
        pool_slots = min(pool_maximum, len(ranked_pool), cap)
        chosen_pool = ranked_pool[:pool_slots]
        chosen_bay_local = [
            area for area in ranked_bay_local if area not in chosen_pool
        ][: cap - len(chosen_pool)]
        if len(chosen_pool) + len(chosen_bay_local) < cap:
            chosen_pool.extend(
                area
                for area in ranked_pool[len(chosen_pool) :]
                if area not in chosen_bay_local
            )
            chosen_pool = chosen_pool[: cap - len(chosen_bay_local)]
        chosen_optional = [*chosen_pool, *chosen_bay_local]
        final = set(mandatory) | set(chosen_optional)
        if not final:
            raise RuntimeError(
                f"V7.1 Stage 1 produced no restricted area for {group_id}"
            )
        selected[group_id] = final
        group_diagnostics.append(
            {
                "group_id": group_id,
                "mandatory_best_areas": sorted(best_areas),
                "pool_candidate_areas": ranked_pool,
                "bay_local_candidate_areas": ranked_bay_local,
                "selected_pool_alternatives": chosen_pool,
                "selected_bay_local_areas": chosen_bay_local,
                "selected_optional_areas": chosen_optional,
                "final_restricted_areas": sorted(final),
                "additional_candidate_area_cap": cap,
                "maximum_pool_candidate_areas": pool_maximum,
                "mandatory_support_size": len(mandatory),
            }
        )
    return freeze_restricted_areas(selected), {
        "additional_candidate_area_cap": cap,
        "maximum_pool_candidate_areas": pool_maximum,
        "groups": group_diagnostics,
    }


def stage1_graph_diagnostics(
    atoms: Sequence[V7RowAtom],
    restricted_areas_by_group: Mapping[
        str, Sequence[str] | set[str] | frozenset[str]
    ],
) -> dict[str, object]:
    legal_areas: defaultdict[str, set[str]] = defaultdict(set)
    legal_bays: defaultdict[str, set[str]] = defaultdict(set)
    groups_by_bay: defaultdict[str, set[str]] = defaultdict(set)
    for atom in atoms:
        legal_areas[atom.group_id].add(atom.area_no)
        legal_bays[atom.group_id].add(atom.anchor_bay_key)
        groups_by_bay[atom.anchor_bay_key].add(atom.group_id)
    active = {
        str(group_id): {str(area) for area in areas}
        for group_id, areas in restricted_areas_by_group.items()
    }
    active_bays: defaultdict[str, set[str]] = defaultdict(set)
    active_groups_by_bay: defaultdict[str, set[str]] = defaultdict(set)
    for atom in atoms:
        if atom.area_no in active.get(atom.group_id, set()):
            active_bays[atom.group_id].add(atom.anchor_bay_key)
            active_groups_by_bay[atom.anchor_bay_key].add(atom.group_id)
    full_area_edges = sum(len(values) for values in legal_areas.values())
    active_area_edges = sum(len(active.get(group_id, set())) for group_id in legal_areas)
    full_bay_edges = sum(len(values) for values in legal_bays.values())
    active_bay_edges = sum(len(values) for values in active_bays.values())
    active_counts = [len(active.get(group_id, set())) for group_id in sorted(legal_areas)]
    return {
        "full_legal_domain": {
            "group_area_edge_count": full_area_edges,
            "group_bay_edge_count": full_bay_edges,
            "average_candidate_bays_per_group": (
                full_bay_edges / len(legal_bays) if legal_bays else 0.0
            ),
            "average_candidate_groups_per_bay": (
                sum(len(values) for values in groups_by_bay.values()) / len(groups_by_bay)
                if groups_by_bay
                else 0.0
            ),
        },
        "stage1_restricted_domain": {
            "group_area_edge_count": active_area_edges,
            "group_bay_edge_count": active_bay_edges,
            "average_candidate_bays_per_group": (
                active_bay_edges / len(legal_bays) if legal_bays else 0.0
            ),
            "average_candidate_groups_per_bay": (
                sum(len(values) for values in active_groups_by_bay.values())
                / len(active_groups_by_bay)
                if active_groups_by_bay
                else 0.0
            ),
        },
        "active_edge_reduction_ratio": (
            1.0 - active_area_edges / full_area_edges if full_area_edges else 0.0
        ),
        "group_bay_edge_reduction_ratio": (
            1.0 - active_bay_edges / full_bay_edges if full_bay_edges else 0.0
        ),
        "average_legal_areas_per_group": (
            full_area_edges / len(legal_areas) if legal_areas else 0.0
        ),
        "average_active_areas_per_group": (
            sum(active_counts) / len(active_counts) if active_counts else 0.0
        ),
        "median_active_areas": float(median(active_counts)) if active_counts else 0.0,
        "fraction_active_le_1": (
            sum(value <= 1 for value in active_counts) / len(active_counts)
            if active_counts
            else 0.0
        ),
        "fraction_active_le_2": (
            sum(value <= 2 for value in active_counts) / len(active_counts)
            if active_counts
            else 0.0
        ),
        "fraction_active_le_3": (
            sum(value <= 3 for value in active_counts) / len(active_counts)
            if active_counts
            else 0.0
        ),
        "fraction_active_le_4": (
            sum(value <= 4 for value in active_counts) / len(active_counts)
            if active_counts
            else 0.0
        ),
        "max_active_areas": max(active_counts, default=0),
    }


class V7Stage1AreaSolver:
    """Choose a frozen candidate-area domain without imposing Stage-2 quotas."""

    def __init__(
        self,
        problem: ProblemData,
        peak_policy: V7PeakUtilizationPolicy,
        atoms: Sequence[V7RowAtom] | None = None,
        config: V7Stage1Config | None = None,
    ) -> None:
        self.problem = problem
        self.peak_policy = peak_policy
        self.config = config or V7Stage1Config()
        self.config.validate()
        if atoms is None:
            atoms, _limits = build_v7_row_atoms(problem)
        self.atoms = tuple(atoms)
        self.evaluator = V7ModelEvaluator(
            problem,
            self.atoms,
            self.config.objective,
        )
        self.groups = self.evaluator.groups
        self.groups_by_id = self.evaluator.groups_by_id
        self.packing = build_v7_stage1_packing_metadata(problem, self.atoms)
        self.reachable_capacity = dict(self.packing.reachable_capacity)
        (
            self.bay_local_areas_by_group,
            self.bay_local_candidate_by_pair,
        ) = build_v7_bay_local_area_candidates(
            problem,
            self.atoms,
            self.evaluator,
        )

    def _configure(self, model: GurobiModel) -> None:
        if not self.config.verbose:
            model.hideOutput()
        model.setMinimize()
        model.setParam("TimeLimit", float(self.config.time_limit))
        model.setParam("Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            model.setParam("Threads", int(self.config.solver_threads))
        model.setParam("PoolSearchMode", 2)
        model.setParam("PoolSolutions", int(self.config.maximum_pool_solutions))
        model.setParam("PoolGap", 1e100)
        # The normalized Stage-1 optimum may be exactly zero, in which case a
        # purely relative PoolGap admits no non-identical support.  Use the same
        # normalized tolerance as an absolute pool envelope as well.
        model.setParam("PoolGapAbs", float(self.config.pool_gap))
        model.setParam("MIPGap", 0.0)

    def solve(self) -> V7Stage1Result:
        started = perf_counter()
        model = GurobiModel("v7_stage1_group_area")
        self._configure(model)
        gp = model._gp
        quicksum = gp.quicksum
        constraints: defaultdict[str, list[object]] = defaultdict(list)
        pairs = sorted(self.reachable_capacity)
        area_use = {
            pair: model.addVar(
                vtype="B",
                obj=self.evaluator.group_area_use_objective_coefficient(),
                name=f"Y_group_area_{index}",
            )
            for index, pair in enumerate(pairs)
        }
        area_quantity = {
            pair: model.addVar(
                lb=0.0,
                ub=float(self.reachable_capacity[pair]),
                vtype="I",
                obj=self.evaluator.area_coarse_cost(*pair),
                name=f"Q_group_area_{index}",
            )
            for index, pair in enumerate(pairs)
        }
        bay_incidence = {
            pair: model.addVar(
                lb=0.0,
                ub=float(self.packing.compatible_anchor_bay_count[pair]),
                vtype="I",
                name=f"N_group_area_{index}",
            )
            for index, pair in enumerate(pairs)
        }
        # Gurobi's solution pool must distinguish coarse area supports, not
        # alternative Q/N values for the same Y support.
        for variable in (*area_quantity.values(), *bay_incidence.values()):
            variable.PoolIgnore = 1
        for pair in pairs:
            constraints["quantity_use_upper"].append(
                model.addConstr(
                    area_quantity[pair]
                    <= int(self.reachable_capacity[pair]) * area_use[pair]
                )
            )
            constraints["quantity_use_lower"].append(
                model.addConstr(area_quantity[pair] >= area_use[pair])
            )
            constraints["minimum_compatible_bay_incidence"].append(
                model.addConstr(
                    area_quantity[pair]
                    <= int(self.packing.max_single_anchor_capacity[pair])
                    * bay_incidence[pair]
                )
            )
            constraints["bay_incidence_area_link"].append(
                model.addConstr(
                    bay_incidence[pair]
                    <= int(self.packing.compatible_anchor_bay_count[pair])
                    * area_use[pair]
                )
            )
        for group in self.groups:
            constraints["exact_group_demand"].append(
                model.addConstr(
                    quicksum(
                        variable
                        for (group_id, _area), variable in area_quantity.items()
                        if group_id == group.group_id
                    )
                    == int(group.demand)
                )
            )
        area_capacity: Counter[str] = Counter()
        for bay in self.problem.bays.values():
            area_capacity[str(bay.area_no)] += int(bay.physical_capacity)
        for area in sorted({pair[1] for pair in pairs}):
            terms = [
                (2 if self.groups_by_id[group_id].size in {"40", "45"} else 1, variable)
                for (group_id, area_no), variable in area_quantity.items()
                if area_no == area
            ]
            constraints["aggregate_area_peak_capacity"].append(
                model.addConstr(
                    quicksum(coefficient * variable for coefficient, variable in terms)
                    <= float(self.peak_policy.epsilon_cap) * int(area_capacity[area])
                )
            )
            constraints["aggregate_three_groups_per_bay_proxy"].append(
                model.addConstr(
                    quicksum(
                        int(self.packing.footprint_width_by_group[group_id])
                        * variable
                        for (group_id, area_no), variable in bay_incidence.items()
                        if area_no == area
                    )
                    <= 3
                    * int(self.packing.usable_physical_bay_count.get(area, 0))
                )
            )
        constant = model.addVar(
            lb=1.0,
            ub=1.0,
            obj=-len(self.groups)
            * self.evaluator.group_area_use_objective_coefficient(),
            name="stage1_area_count_constant",
        )
        constant.PoolIgnore = 1
        model.update()
        try:
            model.optimize()
            status = model.getStatusName()
            base_diagnostics = {
                "algorithm": "v7_stage1_group_area_coarse_mip",
                "model_schema_version": V7_MODEL_SCHEMA_VERSION,
                "status": status,
                "solution_count": model.getSolutionCount(),
                "runtime_seconds": model.getRuntime(),
                "solver_objective": (
                    model.getObjectiveValue() if model.getSolutionCount() else None
                ),
                "solver_bound": model.getBestBound(),
                "solver_gap": model.getMipGap() if model.getSolutionCount() else None,
                "reachable_group_area_edge_count": len(pairs),
                "constraint_count_by_family": {
                    key: len(value) for key, value in sorted(constraints.items())
                },
                "stage2_quota_fixed": False,
                "candidate_cap_is_business_constraint": False,
            }
            if model.getSolutionCount() <= 0:
                raise V7Stage1IncompleteError(
                    f"V7 Stage 1 found no feasible solution: status={status}",
                    base_diagnostics,
                )
            raw_solution_count = min(
                int(model.getSolutionCount()), int(self.config.maximum_pool_solutions)
            )
            pool_by_support: dict[
                tuple[tuple[str, str], ...], V7Stage1PoolSolution
            ] = {}
            for solution_number in range(raw_solution_count):
                quantity = {
                    pair: int(round(model.getPoolValue(variable, solution_number)))
                    for pair, variable in area_quantity.items()
                    if model.getPoolValue(variable, solution_number) > 0.5
                }
                incidence = {
                    pair: int(round(model.getPoolValue(variable, solution_number)))
                    for pair, variable in bay_incidence.items()
                    if model.getPoolValue(variable, solution_number) > 0.5
                }
                used: defaultdict[str, list[str]] = defaultdict(list)
                for (group_id, area), value in quantity.items():
                    if value > 0:
                        used[group_id].append(area)
                solution = V7Stage1PoolSolution(
                    solution_number=solution_number,
                    objective=float(model.getPoolObjective(solution_number)),
                    group_area_quantity=quantity,
                    group_area_bay_incidence=incidence,
                    used_areas_by_group={
                        group.group_id: tuple(sorted(used[group.group_id]))
                        for group in self.groups
                    },
                )
                support = tuple(sorted(quantity))
                previous = pool_by_support.get(support)
                if previous is None or solution.objective < previous.objective:
                    pool_by_support[support] = solution
            pool = sorted(
                pool_by_support.values(),
                key=lambda solution: (solution.objective, solution.solution_number),
            )
            restricted, candidate_diagnostics = self._build_restricted_domain(pool)
            graph = stage1_graph_diagnostics(self.atoms, restricted)
            diagnostics = {
                **base_diagnostics,
                "pool_solution_count": len(pool),
                "raw_pool_solution_count": raw_solution_count,
                "unique_y_support_count": len(pool),
                "duplicate_y_support_count": raw_solution_count - len(pool),
                "pool_distinguished_by_y_support": True,
                "additional_candidate_area_cap": int(
                    self.config.additional_candidate_area_cap
                ),
                "maximum_pool_candidate_areas": int(
                    self.config.maximum_pool_candidate_areas
                ),
                "candidate_area_policy": (
                    "mandatory_stage1_best_plus_capped_pool_and_bay_local"
                ),
                "restricted_domain": graph,
                "candidate_area_diagnostics": candidate_diagnostics,
                "packing_metadata": self._packing_diagnostics(pool[0]),
                "total_seconds": perf_counter() - started,
            }
            return V7Stage1Result(
                best_group_area_quantity=dict(pool[0].group_area_quantity),
                restricted_areas_by_group=restricted,
                pool_solutions=tuple(pool),
                diagnostics=diagnostics,
            )
        finally:
            model.dispose()

    def _build_restricted_domain(
        self,
        pool: Sequence[V7Stage1PoolSolution],
    ) -> tuple[Mapping[str, frozenset[str]], Mapping[str, object]]:
        legal: defaultdict[str, set[str]] = defaultdict(set)
        for atom in self.atoms:
            legal[atom.group_id].add(atom.area_no)
        ranking_tail = {
            pair: (
                int(candidate.shortage_boxes),
                float(candidate.local_objective),
                int(candidate.used_anchor_bay_count),
            )
            for pair, candidate in self.bay_local_candidate_by_pair.items()
        }
        restricted, diagnostics = build_v7_restricted_area_domain(
            (group.group_id for group in self.groups),
            legal,
            pool,
            additional_candidate_area_cap=(
                self.config.additional_candidate_area_cap
            ),
            maximum_pool_candidate_areas=(
                self.config.maximum_pool_candidate_areas
            ),
            bay_local_areas_by_group=self.bay_local_areas_by_group,
            ranking_tail_by_pair=ranking_tail,
        )
        enriched_groups: list[dict[str, object]] = []
        for raw in diagnostics["groups"]:
            row = dict(raw)
            group_id = str(row["group_id"])
            group_areas = sorted(legal[group_id])
            row.update(
                {
                    "reachable_capacity_by_area": {
                        area: int(
                            self.packing.reachable_capacity[(group_id, area)]
                        )
                        for area in group_areas
                    },
                    "compatible_bay_count_by_area": {
                        area: int(
                            self.packing.compatible_anchor_bay_count[
                                (group_id, area)
                            ]
                        )
                        for area in group_areas
                    },
                    "max_single_anchor_capacity_by_area": {
                        area: int(
                            self.packing.max_single_anchor_capacity[
                                (group_id, area)
                            ]
                        )
                        for area in group_areas
                    },
                    "bay_local_candidate_by_area": {
                        area: {
                            "rank": list(
                                self.bay_local_areas_by_group[group_id]
                            ).index(area)
                            + 1,
                            "shortage_boxes": int(
                                self.bay_local_candidate_by_pair[
                                    (group_id, area)
                                ].shortage_boxes
                            ),
                            "supplied_boxes": int(
                                self.bay_local_candidate_by_pair[
                                    (group_id, area)
                                ].supplied_boxes
                            ),
                            "local_objective": float(
                                self.bay_local_candidate_by_pair[
                                    (group_id, area)
                                ].local_objective
                            ),
                            "used_anchor_bays": list(
                                self.bay_local_candidate_by_pair[
                                    (group_id, area)
                                ].used_anchor_bays
                            ),
                            "used_anchor_bay_count": int(
                                self.bay_local_candidate_by_pair[
                                    (group_id, area)
                                ].used_anchor_bay_count
                            ),
                            "normalized_span": float(
                                self.bay_local_candidate_by_pair[
                                    (group_id, area)
                                ].normalized_span
                            ),
                        }
                        for area in group_areas
                    },
                }
            )
            enriched_groups.append(row)
        return restricted, {
            **dict(diagnostics),
            "groups": enriched_groups,
        }

    def _packing_diagnostics(
        self,
        best: V7Stage1PoolSolution,
    ) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        for group in self.groups:
            for area in sorted(
                area
                for group_id, area in self.reachable_capacity
                if group_id == group.group_id
            ):
                pair = (group.group_id, area)
                quantity = int(best.group_area_quantity.get(pair, 0))
                maximum = int(self.packing.max_single_anchor_capacity[pair])
                output.append(
                    {
                        "group_id": group.group_id,
                        "area": area,
                        "reachable_capacity": int(
                            self.packing.reachable_capacity[pair]
                        ),
                        "compatible_bay_count": int(
                            self.packing.compatible_anchor_bay_count[pair]
                        ),
                        "max_single_anchor_capacity": maximum,
                        "minimum_bay_incidence": (
                            int(math.ceil(quantity / maximum)) if quantity else 0
                        ),
                    }
                )
        return output


__all__ = [
    "V7Stage1AreaSolver",
    "V7Stage1BayLocalAreaCandidate",
    "V7Stage1Config",
    "V7Stage1IncompleteError",
    "V7Stage1PackingMetadata",
    "V7Stage1PoolSolution",
    "V7Stage1Result",
    "build_v7_bay_local_area_candidates",
    "build_v7_restricted_area_domain",
    "build_v7_stage1_packing_metadata",
    "freeze_restricted_areas",
    "stage1_graph_diagnostics",
]
