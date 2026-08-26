"""End-to-end V7.1 hierarchical restricted-CG production interface."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Mapping

from .models import ProblemData
from .v7_atoms import V7RowAtom, build_v7_row_atoms
from .v7_bay_patterns import (
    V7BayPattern,
    build_v7_pattern_from_atom_indices,
)
from .v7_column_generation import (
    V7_1_ALGORITHM_VERSION,
    V7GlobalPricingAudit,
    V7RootCgConfig,
    V7RootCgResult,
    V7RootColumnGeneration,
    normalize_v7_pattern_ids,
    not_executed_global_pricing_audit,
    patterns_from_atom_solution,
)
from .v7_complete_mip import (
    V7CompleteMipConfig,
    V7CompleteMipResult,
    V7CompleteMipSolver,
)
from .v7_integer import V7IntegerConfig, V7IntegerResult, V7RestrictedIntegerSolver
from .v7_model import (
    V7_MODEL_SCHEMA_VERSION,
    V7ObjectiveConfig,
    V7PeakUtilizationPolicy,
    derive_v7_analytic_peak_policy,
)
from .v7_stage1_area import V7Stage1AreaSolver, V7Stage1Config, V7Stage1Result


_MINIMUM_PHASE_LAUNCH_SECONDS = 0.1


class V7AlgorithmBudgetExhaustedError(RuntimeError):
    """Raised when the shared production budget cannot launch a required stage."""


@dataclass(frozen=True)
class V7AlgorithmTimeBudget:
    """Allocate one shared wall-clock budget across the production stages."""

    total_time_limit: float
    minimum_integer_time: float = 5.0

    @property
    def minimum_phase_launch_seconds(self) -> float:
        return _MINIMUM_PHASE_LAUNCH_SECONDS

    def validate(self) -> None:
        if (
            not math.isfinite(float(self.total_time_limit))
            or float(self.total_time_limit) <= 0
        ):
            raise ValueError("V7 total algorithm time limit must be positive")
        if (
            not math.isfinite(float(self.minimum_integer_time))
            or float(self.minimum_integer_time) <= 0
        ):
            raise ValueError("V7 minimum integer time must be positive")
        minimum_required = float(self.minimum_integer_time) + 2.0 * (
            _MINIMUM_PHASE_LAUNCH_SECONDS
        )
        if float(self.total_time_limit) <= minimum_required:
            raise ValueError(
                "V7 total algorithm time limit cannot reserve Stage 1, root, "
                "and integer launch time"
            )

    def remaining(self, elapsed_seconds: float) -> float:
        return max(0.0, float(self.total_time_limit) - float(elapsed_seconds))

    def soft_stage_limit(
        self,
        stage: str,
        *,
        elapsed_seconds: float,
        soft_limit: float,
        future_reserve: float,
    ) -> float:
        available = self.remaining(elapsed_seconds) - float(future_reserve)
        allocated = min(float(soft_limit), available)
        if allocated < _MINIMUM_PHASE_LAUNCH_SECONDS:
            raise V7AlgorithmBudgetExhaustedError(
                f"V7 shared time budget cannot launch {stage}: "
                f"remaining={self.remaining(elapsed_seconds):.6f}, "
                f"future_reserve={float(future_reserve):.6f}"
            )
        return allocated

    def integer_stage_limit(
        self,
        *,
        elapsed_seconds: float,
        optional_ceiling: float | None = None,
    ) -> float:
        allocated = self.remaining(elapsed_seconds)
        if optional_ceiling is not None:
            allocated = min(allocated, float(optional_ceiling))
        if allocated < _MINIMUM_PHASE_LAUNCH_SECONDS:
            raise V7AlgorithmBudgetExhaustedError(
                "V7 shared time budget was exhausted before the integer master"
            )
        return allocated


@dataclass(frozen=True)
class V7Config:
    objective: V7ObjectiveConfig = field(default_factory=V7ObjectiveConfig)
    peak_headroom_fraction: float = 0.50
    max_new_groups_per_physical_bay: int = 3

    algorithm_time_limit: float = 120.0
    minimum_integer_time: float = 5.0
    peak_feasibility_time_limit: float = 10.0
    stage1_time_limit: float = 10.0
    stage1_pool_solutions: int = 8
    stage1_pool_gap: float = 0.10
    additional_candidate_area_cap: int = 5
    maximum_pool_candidate_areas: int = 1

    root_time_limit: float = 30.0
    root_maximum_iterations: int = 100
    reduced_cost_tolerance: float = 1e-8
    columns_per_bay_per_round: int = 3

    integer_time_limit: float | None = None
    integer_mip_gap: float = 0.0
    solver_threads: int = 1
    solver_seed: int = 0
    verbose: bool = False

    def validate(self) -> None:
        self.objective.validate()
        if not math.isclose(
            float(self.peak_headroom_fraction),
            float(self.objective.peak_utilization_headroom_fraction),
            abs_tol=1e-12,
        ):
            raise ValueError("V7 config peak headroom and objective config disagree")
        if int(self.max_new_groups_per_physical_bay) != 3:
            raise ValueError("V7 baseline fixes max new groups per physical bay to 3")
        if int(self.additional_candidate_area_cap) <= 0:
            raise ValueError("V7 additional candidate area cap must be positive")
        if not 0 <= int(self.maximum_pool_candidate_areas) <= int(
            self.additional_candidate_area_cap
        ):
            raise ValueError("V7 maximum pool-candidate area count is invalid")
        for name, value in (
            ("algorithm_time_limit", self.algorithm_time_limit),
            ("minimum_integer_time", self.minimum_integer_time),
            ("peak_feasibility_time_limit", self.peak_feasibility_time_limit),
            ("stage1_time_limit", self.stage1_time_limit),
            ("root_time_limit", self.root_time_limit),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"V7 {name} must be positive")
        if self.integer_time_limit is not None and (
            not math.isfinite(float(self.integer_time_limit))
            or float(self.integer_time_limit) <= 0
        ):
            raise ValueError("V7 optional integer time ceiling must be positive")
        V7AlgorithmTimeBudget(
            total_time_limit=self.algorithm_time_limit,
            minimum_integer_time=self.minimum_integer_time,
        ).validate()


@dataclass(frozen=True)
class V7ProductionResult:
    peak_policy: V7PeakUtilizationPolicy
    witness: V7CompleteMipResult
    stage1: V7Stage1Result
    root: V7RootCgResult
    integer: V7IntegerResult
    global_pricing_audit: V7GlobalPricingAudit
    restricted_lp_bound: float
    restricted_integer_ub: float
    restricted_mip_gap: float | None
    global_lower_bound: float | None
    global_gap: float | None
    diagnostics: Mapping[str, object]

    @property
    def objective(self) -> float:
        return self.integer.objective


def build_v7_stage1_guided_patterns(
    problem: ProblemData,
    atoms: tuple[V7RowAtom, ...],
    stage1: V7Stage1Result,
) -> tuple[V7BayPattern, ...]:
    """Prioritize one-group seeds for the best Stage-1 Q[g,a] support."""

    atoms_by_index = {atom.candidate_index: atom for atom in atoms}
    chosen: list[V7BayPattern] = []
    for (group_id, area), quantity in sorted(
        stage1.best_group_area_quantity.items(),
        key=lambda item: (-int(item[1]), item[0]),
    ):
        if area not in stage1.restricted_areas_by_group[group_id]:
            raise RuntimeError("Stage-1 best support escaped the restricted domain")
        best_by_anchor: dict[str, V7RowAtom] = {}
        for atom in atoms:
            if atom.group_id != group_id or atom.area_no != area:
                continue
            current = best_by_anchor.get(atom.anchor_bay_key)
            if current is None or (
                -int(atom.capacity), atom.row_no, atom.candidate_index
            ) < (-int(current.capacity), current.row_no, current.candidate_index):
                best_by_anchor[atom.anchor_bay_key] = atom
        supplied = 0
        for atom in sorted(
            best_by_anchor.values(),
            key=lambda value: (
                -int(value.capacity),
                int(problem.bays[value.anchor_bay_key].bay_order),
                value.row_no,
            ),
        ):
            chosen.append(
                build_v7_pattern_from_atom_indices(
                    problem,
                    atoms_by_index,
                    [atom.candidate_index],
                    pattern_id=len(chosen),
                )
            )
            supplied += int(atom.capacity)
            if supplied >= int(quantity):
                break
    return normalize_v7_pattern_ids(chosen)


def build_v7_deterministic_one_group_patterns(
    problem: ProblemData,
    atoms: tuple[V7RowAtom, ...],
    restricted_areas_by_group: Mapping[str, frozenset[str]],
) -> tuple[V7BayPattern, ...]:
    """Add one deterministic legal one-group seed per restricted group-area."""

    atoms_by_index = {atom.candidate_index: atom for atom in atoms}
    candidates: defaultdict[tuple[str, str], list[V7RowAtom]] = defaultdict(list)
    for atom in atoms:
        if atom.area_no in restricted_areas_by_group.get(atom.group_id, frozenset()):
            candidates[(atom.group_id, atom.area_no)].append(atom)
    patterns: list[V7BayPattern] = []
    for pair in sorted(candidates):
        atom = min(
            candidates[pair],
            key=lambda value: (
                -int(value.capacity),
                int(problem.bays[value.anchor_bay_key].bay_order),
                value.row_no,
                value.candidate_index,
            ),
        )
        patterns.append(
            build_v7_pattern_from_atom_indices(
                problem,
                atoms_by_index,
                [atom.candidate_index],
                pattern_id=len(patterns),
            )
        )
    return normalize_v7_pattern_ids(patterns)


def solve_v7(
    problem: ProblemData,
    config: V7Config | None = None,
) -> V7ProductionResult:
    """Solve the V7 two-stage pipeline without hidden fallback enumeration."""

    cfg = config or V7Config()
    cfg.validate()
    started = perf_counter()
    atoms, _anchor_limits = build_v7_row_atoms(problem)
    budget = V7AlgorithmTimeBudget(
        total_time_limit=cfg.algorithm_time_limit,
        minimum_integer_time=cfg.minimum_integer_time,
    )
    budget.validate()
    budget_started = perf_counter()
    peak_policy, peak_diagnostics = derive_v7_analytic_peak_policy(
        problem,
        cfg.objective,
        atoms=atoms,
    )
    allocated_time_limits: dict[str, float] = {}
    allocated_time_limits["peak_witness"] = budget.soft_stage_limit(
        "peak witness",
        elapsed_seconds=perf_counter() - budget_started,
        soft_limit=cfg.peak_feasibility_time_limit,
        future_reserve=(
            cfg.minimum_integer_time
            + 2.0 * budget.minimum_phase_launch_seconds
        ),
    )
    complete_solver = V7CompleteMipSolver(
        problem,
        V7CompleteMipConfig(
            time_limit=allocated_time_limits["peak_witness"],
            mip_gap=0.0,
            solver_threads=cfg.solver_threads,
            solver_seed=cfg.solver_seed,
            verbose=cfg.verbose,
            require_business_optimality=False,
            objective=cfg.objective,
        ),
    )
    witness = complete_solver.solve(peak_policy, feasibility_only=True)
    witness_patterns = patterns_from_atom_solution(
        problem, atoms, witness.selected_atom_indices
    )

    allocated_time_limits["stage1"] = budget.soft_stage_limit(
        "Stage 1",
        elapsed_seconds=perf_counter() - budget_started,
        soft_limit=cfg.stage1_time_limit,
        future_reserve=(
            cfg.minimum_integer_time + budget.minimum_phase_launch_seconds
        ),
    )
    stage1 = V7Stage1AreaSolver(
        problem,
        peak_policy,
        atoms,
        V7Stage1Config(
            time_limit=allocated_time_limits["stage1"],
            maximum_pool_solutions=cfg.stage1_pool_solutions,
            pool_gap=cfg.stage1_pool_gap,
            additional_candidate_area_cap=cfg.additional_candidate_area_cap,
            maximum_pool_candidate_areas=(
                cfg.maximum_pool_candidate_areas
            ),
            solver_threads=cfg.solver_threads,
            solver_seed=cfg.solver_seed,
            verbose=cfg.verbose,
            objective=cfg.objective,
        ),
    ).solve()
    guided_patterns = build_v7_stage1_guided_patterns(problem, atoms, stage1)
    deterministic_patterns = build_v7_deterministic_one_group_patterns(
        problem, atoms, stage1.restricted_areas_by_group
    )
    initial_search_patterns = normalize_v7_pattern_ids(
        [*guided_patterns, *deterministic_patterns]
    )

    allocated_time_limits["root_cg"] = budget.soft_stage_limit(
        "restricted root CG",
        elapsed_seconds=perf_counter() - budget_started,
        soft_limit=cfg.root_time_limit,
        future_reserve=cfg.minimum_integer_time,
    )
    root = V7RootColumnGeneration(
        problem,
        peak_policy,
        stage1.restricted_areas_by_group,
        initial_search_patterns,
        V7RootCgConfig(
            root_time_limit=allocated_time_limits["root_cg"],
            maximum_iterations=cfg.root_maximum_iterations,
            reduced_cost_tolerance=cfg.reduced_cost_tolerance,
            columns_per_bay_per_round=cfg.columns_per_bay_per_round,
            solver_threads=cfg.solver_threads,
            solver_seed=cfg.solver_seed,
            verbose=cfg.verbose,
            objective=cfg.objective,
        ),
        atoms=atoms,
        feasibility_proof_patterns=witness_patterns,
    ).solve()
    integer_pool = normalize_v7_pattern_ids(
        [
            *root.patterns,
            *witness_patterns,
            *guided_patterns,
            *deterministic_patterns,
        ]
    )
    allocated_time_limits["integer"] = budget.integer_stage_limit(
        elapsed_seconds=perf_counter() - budget_started,
        optional_ceiling=cfg.integer_time_limit,
    )
    integer = V7RestrictedIntegerSolver(
        problem,
        peak_policy,
        integer_pool,
        V7IntegerConfig(
            time_limit=allocated_time_limits["integer"],
            wall_clock_deadline=(budget_started + cfg.algorithm_time_limit),
            mip_gap=cfg.integer_mip_gap,
            solver_threads=cfg.solver_threads,
            solver_seed=cfg.solver_seed,
            verbose=cfg.verbose,
            objective=cfg.objective,
        ),
        atoms=atoms,
        restricted_areas_by_group=stage1.restricted_areas_by_group,
        feasibility_proof_patterns=witness_patterns,
    ).solve()
    global_audit = not_executed_global_pricing_audit()
    restricted_mip_gap = integer.diagnostics.get("restricted_mip_gap")
    return V7ProductionResult(
        peak_policy=peak_policy,
        witness=witness,
        stage1=stage1,
        root=root,
        integer=integer,
        global_pricing_audit=global_audit,
        restricted_lp_bound=float(root.restricted_lp_bound),
        restricted_integer_ub=float(integer.objective),
        restricted_mip_gap=(
            None if restricted_mip_gap is None else float(restricted_mip_gap)
        ),
        global_lower_bound=None,
        global_gap=None,
        diagnostics={
            "algorithm": V7_1_ALGORITHM_VERSION,
            "model_schema_version": V7_MODEL_SCHEMA_VERSION,
            "legal_atom_count": len(atoms),
            "peak_preparation": peak_diagnostics,
            "witness_pattern_count": len(witness_patterns),
            "stage1_guided_pattern_count": len(guided_patterns),
            "deterministic_one_group_pattern_count": len(
                deterministic_patterns
            ),
            "root_pattern_count": len(root.patterns),
            "integer_pattern_pool_count": len(integer_pool),
            "restricted_area_policy": (
                "stage1_best_plus_capped_pool_and_bay_local"
            ),
            "feasibility_proof_columns_expand_pricing_domain": False,
            "restricted_areas_by_group": {
                group_id: sorted(areas)
                for group_id, areas in stage1.restricted_areas_by_group.items()
            },
            "stage1_quota_fixed": False,
            "restricted_root_closed": bool(
                root.diagnostics["restricted_root_closed"]
            ),
            "restricted_lp_bound": float(root.restricted_lp_bound),
            "integer_feasible": True,
            "restricted_integer_ub": float(integer.objective),
            "restricted_mip_gap": restricted_mip_gap,
            "global_pricing_audit": {
                "executed": global_audit.executed,
                "minimum_reduced_cost": global_audit.minimum_reduced_cost,
                "globally_root_certified": (
                    global_audit.globally_root_certified
                ),
            },
            "global_lower_bound": None,
            "global_gap": None,
            "production_global_expansion_enabled": False,
            "complete_pattern_enumeration_used": False,
            "performance_experiments_run": False,
            "algorithm_time_budget": {
                "policy": "shared_total_with_soft_phase_caps",
                "total_time_limit": float(cfg.algorithm_time_limit),
                "minimum_integer_time": float(cfg.minimum_integer_time),
                "integer_optional_ceiling": (
                    None
                    if cfg.integer_time_limit is None
                    else float(cfg.integer_time_limit)
                ),
                "allocated_time_limits": dict(allocated_time_limits),
                "production_wall_seconds": perf_counter() - budget_started,
                "remaining_seconds": budget.remaining(
                    perf_counter() - budget_started
                ),
                "global_pricing_audit_included": False,
            },
            "total_seconds": perf_counter() - started,
        },
    )


__all__ = [
    "V7AlgorithmBudgetExhaustedError",
    "V7AlgorithmTimeBudget",
    "V7Config",
    "V7ProductionResult",
    "build_v7_deterministic_one_group_patterns",
    "build_v7_stage1_guided_patterns",
    "solve_v7",
]
