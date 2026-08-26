"""V7 Stage 1 coarse group-to-area MIP and active-set construction."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median
from time import perf_counter
from typing import Mapping, Sequence

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
    initial_candidate_area_cap: int = 4
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
        if int(self.initial_candidate_area_cap) <= 0:
            raise ValueError("V7 Stage-1 active-area cap must be positive")
        if int(self.solver_threads) < 0:
            raise ValueError("V7 solver_threads must be nonnegative")
        self.objective.validate()


@dataclass(frozen=True)
class V7Stage1PoolSolution:
    solution_number: int
    objective: float
    group_area_quantity: Mapping[tuple[str, str], int]
    used_areas_by_group: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class V7Stage1Result:
    best_group_area_quantity: Mapping[tuple[str, str], int]
    active_areas_by_group: Mapping[str, frozenset[str]]
    pool_solutions: tuple[V7Stage1PoolSolution, ...]
    diagnostics: Mapping[str, object]

    @property
    def quota_fixed_in_stage2(self) -> bool:
        return False


def stage1_graph_diagnostics(
    atoms: Sequence[V7RowAtom],
    active_areas_by_group: Mapping[str, Sequence[str] | set[str] | frozenset[str]],
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
        for group_id, areas in active_areas_by_group.items()
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
        "stage1_active_domain": {
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
    """Generate a sparse area active set without imposing Stage-2 quotas."""

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
        self.reachable_capacity = self._reachable_capacity()

    def _reachable_capacity(self) -> dict[tuple[str, str], int]:
        capacity: Counter[tuple[str, str]] = Counter()
        for atom in self.atoms:
            capacity[(atom.group_id, atom.area_no)] += int(atom.capacity)
        return {
            key: min(int(self.groups_by_id[key[0]].demand), int(value))
            for key, value in capacity.items()
            if int(value) > 0
        }

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
        model.setParam("PoolGap", float(self.config.pool_gap))
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
        model.addVar(
            lb=1.0,
            ub=1.0,
            obj=-len(self.groups)
            * self.evaluator.group_area_use_objective_coefficient(),
            name="stage1_area_count_constant",
        )
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
            solution_count = min(
                int(model.getSolutionCount()), int(self.config.maximum_pool_solutions)
            )
            pool: list[V7Stage1PoolSolution] = []
            for solution_number in range(solution_count):
                quantity = {
                    pair: int(round(model.getPoolValue(variable, solution_number)))
                    for pair, variable in area_quantity.items()
                    if model.getPoolValue(variable, solution_number) > 0.5
                }
                used: defaultdict[str, list[str]] = defaultdict(list)
                for (group_id, area), value in quantity.items():
                    if value > 0:
                        used[group_id].append(area)
                pool.append(
                    V7Stage1PoolSolution(
                        solution_number=solution_number,
                        objective=float(model.getPoolObjective(solution_number)),
                        group_area_quantity=quantity,
                        used_areas_by_group={
                            group.group_id: tuple(sorted(used[group.group_id]))
                            for group in self.groups
                        },
                    )
                )
            active = self._build_active_set(pool)
            graph = stage1_graph_diagnostics(self.atoms, active)
            diagnostics = {
                **base_diagnostics,
                "pool_solution_count": len(pool),
                "initial_candidate_area_cap": int(
                    self.config.initial_candidate_area_cap
                ),
                "active_set": graph,
                "total_seconds": perf_counter() - started,
            }
            return V7Stage1Result(
                best_group_area_quantity=dict(pool[0].group_area_quantity),
                active_areas_by_group={
                    group_id: frozenset(areas)
                    for group_id, areas in active.items()
                },
                pool_solutions=tuple(pool),
                diagnostics=diagnostics,
            )
        finally:
            model.dispose()

    def _build_active_set(
        self,
        pool: Sequence[V7Stage1PoolSolution],
    ) -> dict[str, set[str]]:
        best = pool[0]
        frequency: Counter[tuple[str, str]] = Counter()
        mass: Counter[tuple[str, str]] = Counter()
        union: defaultdict[str, set[str]] = defaultdict(set)
        for solution in pool:
            for pair, quantity in solution.group_area_quantity.items():
                if int(quantity) <= 0:
                    continue
                frequency[pair] += 1
                mass[pair] += int(quantity)
                union[pair[0]].add(pair[1])
        output: dict[str, set[str]] = {}
        cap = int(self.config.initial_candidate_area_cap)
        for group in self.groups:
            group_id = group.group_id
            best_areas = set(best.used_areas_by_group.get(group_id, ()))
            candidates = union.get(group_id, set())
            target = max(cap, len(best_areas))
            ranked = sorted(
                candidates,
                key=lambda area: (
                    0 if area in best_areas else 1,
                    -frequency[(group_id, area)],
                    -mass[(group_id, area)],
                    self.evaluator.area_coarse_cost(group_id, area),
                    area,
                ),
            )
            chosen = set(ranked[:target]) | best_areas
            if not chosen:
                raise RuntimeError(f"V7 Stage 1 produced no active area for {group_id}")
            output[group_id] = chosen
        return output


__all__ = [
    "V7Stage1AreaSolver",
    "V7Stage1Config",
    "V7Stage1IncompleteError",
    "V7Stage1PoolSolution",
    "V7Stage1Result",
    "stage1_graph_diagnostics",
]
