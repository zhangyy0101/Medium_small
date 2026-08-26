"""End-to-end V7 two-stage production interface."""

from __future__ import annotations

import math
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
    V7RootCgConfig,
    V7RootCgResult,
    V7RootColumnGeneration,
    normalize_v7_pattern_ids,
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


@dataclass(frozen=True)
class V7Config:
    objective: V7ObjectiveConfig = field(default_factory=V7ObjectiveConfig)
    peak_headroom_fraction: float = 0.50
    max_new_groups_per_physical_bay: int = 3

    peak_feasibility_time_limit: float = 10.0
    stage1_time_limit: float = 10.0
    stage1_pool_solutions: int = 8
    stage1_pool_gap: float = 0.10
    initial_candidate_area_cap: int = 4

    root_time_limit: float = 30.0
    root_maximum_iterations: int = 100
    reduced_cost_tolerance: float = 1e-8
    columns_per_bay_per_round: int = 3

    integer_time_limit: float = 30.0
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
        for name, value in (
            ("peak_feasibility_time_limit", self.peak_feasibility_time_limit),
            ("stage1_time_limit", self.stage1_time_limit),
            ("root_time_limit", self.root_time_limit),
            ("integer_time_limit", self.integer_time_limit),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"V7 {name} must be positive")


@dataclass(frozen=True)
class V7ProductionResult:
    peak_policy: V7PeakUtilizationPolicy
    witness: V7CompleteMipResult
    stage1: V7Stage1Result
    root: V7RootCgResult
    integer: V7IntegerResult
    diagnostics: Mapping[str, object]

    @property
    def objective(self) -> float:
        return self.integer.objective


def build_v7_stage1_guided_patterns(
    problem: ProblemData,
    atoms: tuple[V7RowAtom, ...],
    stage1: V7Stage1Result,
) -> tuple[V7BayPattern, ...]:
    """Create a small legal one-group seed set without enumerating all patterns."""

    atoms_by_index = {atom.candidate_index: atom for atom in atoms}
    chosen: list[V7BayPattern] = []
    for group_id, areas in sorted(stage1.active_areas_by_group.items()):
        for area in sorted(areas):
            candidates = [
                atom
                for atom in atoms
                if atom.group_id == group_id and atom.area_no == area
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda atom: (
                    -int(atom.capacity),
                    int(problem.bays[atom.anchor_bay_key].bay_order),
                    atom.row_no,
                )
            )
            atom = candidates[0]
            # One atom is always a legal one-group Bay Pattern.  This is a
            # seed only; it does not impose the Stage-1 Q[g,a] quantity.
            chosen.append(
                build_v7_pattern_from_atom_indices(
                    problem,
                    atoms_by_index,
                    [atom.candidate_index],
                    pattern_id=len(chosen),
                )
            )
    return normalize_v7_pattern_ids(chosen)


def solve_v7(
    problem: ProblemData,
    config: V7Config | None = None,
) -> V7ProductionResult:
    """Solve the V7 two-stage pipeline without hidden fallback enumeration."""

    cfg = config or V7Config()
    cfg.validate()
    started = perf_counter()
    atoms, _anchor_limits = build_v7_row_atoms(problem)
    peak_policy, peak_diagnostics = derive_v7_analytic_peak_policy(
        problem,
        cfg.objective,
        atoms=atoms,
    )
    complete_solver = V7CompleteMipSolver(
        problem,
        V7CompleteMipConfig(
            time_limit=cfg.peak_feasibility_time_limit,
            mip_gap=1.0,
            solver_threads=cfg.solver_threads,
            solver_seed=cfg.solver_seed,
            verbose=cfg.verbose,
            require_business_optimality=False,
            objective=cfg.objective,
        ),
    )
    witness = complete_solver.solve(peak_policy, feasibility_only=True)

    stage1 = V7Stage1AreaSolver(
        problem,
        peak_policy,
        atoms,
        V7Stage1Config(
            time_limit=cfg.stage1_time_limit,
            maximum_pool_solutions=cfg.stage1_pool_solutions,
            pool_gap=cfg.stage1_pool_gap,
            initial_candidate_area_cap=cfg.initial_candidate_area_cap,
            solver_threads=cfg.solver_threads,
            solver_seed=cfg.solver_seed,
            verbose=cfg.verbose,
            objective=cfg.objective,
        ),
    ).solve()
    witness_patterns = patterns_from_atom_solution(
        problem, atoms, witness.selected_atom_indices
    )
    guided_patterns = build_v7_stage1_guided_patterns(problem, atoms, stage1)
    initial_patterns = normalize_v7_pattern_ids(
        [*witness_patterns, *guided_patterns]
    )

    root = V7RootColumnGeneration(
        problem,
        peak_policy,
        stage1.active_areas_by_group,
        initial_patterns,
        V7RootCgConfig(
            root_time_limit=cfg.root_time_limit,
            maximum_iterations=cfg.root_maximum_iterations,
            reduced_cost_tolerance=cfg.reduced_cost_tolerance,
            columns_per_bay_per_round=cfg.columns_per_bay_per_round,
            solver_threads=cfg.solver_threads,
            solver_seed=cfg.solver_seed,
            verbose=cfg.verbose,
            objective=cfg.objective,
        ),
        atoms=atoms,
    ).solve()
    integer_pool = normalize_v7_pattern_ids(
        [*root.patterns, *witness_patterns, *guided_patterns]
    )
    integer = V7RestrictedIntegerSolver(
        problem,
        peak_policy,
        integer_pool,
        V7IntegerConfig(
            time_limit=cfg.integer_time_limit,
            mip_gap=cfg.integer_mip_gap,
            solver_threads=cfg.solver_threads,
            solver_seed=cfg.solver_seed,
            verbose=cfg.verbose,
            objective=cfg.objective,
        ),
        atoms=atoms,
    ).solve()
    return V7ProductionResult(
        peak_policy=peak_policy,
        witness=witness,
        stage1=stage1,
        root=root,
        integer=integer,
        diagnostics={
            "algorithm": "v7_two_stage_area_then_global_bay_pattern_cg",
            "model_schema_version": V7_MODEL_SCHEMA_VERSION,
            "legal_atom_count": len(atoms),
            "peak_preparation": peak_diagnostics,
            "witness_pattern_count": len(witness_patterns),
            "stage1_guided_pattern_count": len(guided_patterns),
            "root_pattern_count": len(root.patterns),
            "integer_pattern_pool_count": len(integer_pool),
            "stage1_quota_fixed": False,
            "full_domain_certification_passed": bool(
                root.diagnostics["root_closed"]
            ),
            "complete_pattern_enumeration_used": False,
            "performance_experiments_run": False,
            "total_seconds": perf_counter() - started,
        },
    )


__all__ = [
    "V7Config",
    "V7ProductionResult",
    "build_v7_stage1_guided_patterns",
    "solve_v7",
]
