"""Restricted integer master for the V7 Bay Pattern formulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter
from typing import Mapping, Sequence

from .models import ProblemData
from .v7_atoms import V7RowAtom, build_v7_row_atoms
from .v7_bay_patterns import V7BayPattern
from .v7_column_generation import (
    V7GlobalPatternMaster,
    V7MasterSolution,
    normalize_v7_pattern_ids,
)
from .v7_model import (
    V7_MODEL_SCHEMA_VERSION,
    V7ModelEvaluator,
    V7ObjectiveConfig,
    V7PeakUtilizationPolicy,
)


@dataclass(frozen=True)
class V7IntegerConfig:
    time_limit: float = 30.0
    mip_gap: float = 0.0
    solver_threads: int = 1
    solver_seed: int = 0
    verbose: bool = False
    objective: V7ObjectiveConfig = field(default_factory=V7ObjectiveConfig)

    def validate(self) -> None:
        if not math.isfinite(float(self.time_limit)) or float(self.time_limit) <= 0:
            raise ValueError("V7 integer time limit must be positive")
        if not math.isfinite(float(self.mip_gap)) or not 0 <= float(self.mip_gap) <= 1:
            raise ValueError("V7 integer MIP gap must lie in [0, 1]")
        if int(self.solver_threads) < 0:
            raise ValueError("V7 solver_threads must be nonnegative")
        self.objective.validate()


@dataclass(frozen=True)
class V7IntegerResult:
    selected_pattern_ids: tuple[int, ...]
    selected_atom_indices: tuple[int, ...]
    group_bay_flow: Mapping[tuple[str, str], int]
    import_reservation: Mapping[tuple[str, str, str], int]
    patterns: tuple[V7BayPattern, ...]
    peak_policy: V7PeakUtilizationPolicy
    certificate: Mapping[str, object]
    diagnostics: Mapping[str, object]

    @property
    def objective(self) -> float:
        return float(self.certificate["objective"])


class V7RestrictedIntegerSolver:
    def __init__(
        self,
        problem: ProblemData,
        peak_policy: V7PeakUtilizationPolicy,
        patterns: Sequence[V7BayPattern],
        config: V7IntegerConfig | None = None,
        *,
        atoms: Sequence[V7RowAtom] | None = None,
    ) -> None:
        self.problem = problem
        self.peak_policy = peak_policy
        self.config = config or V7IntegerConfig()
        self.config.validate()
        if atoms is None:
            atoms, _limits = build_v7_row_atoms(problem)
        self.atoms = tuple(atoms)
        self.patterns = normalize_v7_pattern_ids(patterns)
        if not self.patterns:
            raise ValueError("V7 integer master requires an explicit pattern pool")
        self.evaluator = V7ModelEvaluator(
            problem, self.atoms, self.config.objective
        )

    def solve(self) -> V7IntegerResult:
        started = perf_counter()
        master = V7GlobalPatternMaster(
            self.problem,
            self.atoms,
            self.patterns,
            self.peak_policy,
            self.config.objective,
            integral=True,
            time_limit=self.config.time_limit,
            mip_gap=self.config.mip_gap,
            solver_threads=self.config.solver_threads,
            solver_seed=self.config.solver_seed,
            verbose=self.config.verbose,
        )
        try:
            solution: V7MasterSolution = master.solve()
            selected_pattern_ids = tuple(
                sorted(
                    pattern_id
                    for pattern_id, value in solution.pattern_values.items()
                    if value > 0.5
                )
            )
            selected_atom_indices = master.selected_atom_indices(solution)
            flow = {
                key: int(round(value))
                for key, value in solution.group_bay_flow.items()
                if value > 0.5
            }
            imports = {
                key: int(round(value))
                for key, value in solution.import_reservation.items()
                if value > 0.5
            }
            certificate = self.evaluator.evaluate(
                selected_atom_indices,
                flow,
                imports,
                self.peak_policy,
            )
            certified_objective = float(certificate["objective"])
            objective_difference = solution.objective - certified_objective
            if certified_objective > solution.objective + 1e-8:
                raise RuntimeError(
                    "V7 evaluator objective exceeds the integer incumbent: "
                    f"solver={solution.objective}, evaluator={certified_objective}"
                )
            progress = solution.diagnostics.get("progress") or {}
            best_bound = solution.diagnostics.get("solver_bound")
            diagnostics = {
                "algorithm": "v7_restricted_global_bay_pattern_integer_master",
                "model_schema_version": V7_MODEL_SCHEMA_VERSION,
                "status": solution.diagnostics["status"],
                "time_to_first_incumbent": progress.get("time_to_first_solution"),
                "first_ub": progress.get("first_incumbent"),
                "best_ub": certified_objective,
                "best_bound": best_bound,
                "gap": (
                    max(0.0, certified_objective - float(best_bound))
                    / max(abs(certified_objective), 1e-12)
                    if best_bound is not None
                    else None
                ),
                "patterns_used": len(selected_pattern_ids),
                "pattern_pool_size": len(self.patterns),
                "groups_per_physical_bay_distribution": dict(
                    sorted(certificate["groups_per_physical_bay"].items())
                ),
                "areas_per_group": certificate["areas_per_group"],
                "bays_per_group": certificate["bays_per_group"],
                "span_per_group_area": certificate["span_by_group_area"],
                "solver_evaluator_objective_difference": objective_difference,
                "incumbent_auxiliary_objective_repaired": bool(
                    objective_difference > 1e-8
                ),
                "independent_validation_passed": True,
                "runtime_seconds": solution.diagnostics["runtime_seconds"],
                "total_seconds": perf_counter() - started,
            }
            return V7IntegerResult(
                selected_pattern_ids=selected_pattern_ids,
                selected_atom_indices=selected_atom_indices,
                group_bay_flow=flow,
                import_reservation=imports,
                patterns=master.patterns,
                peak_policy=self.peak_policy,
                certificate=certificate,
                diagnostics=diagnostics,
            )
        finally:
            master.dispose()


__all__ = [
    "V7IntegerConfig",
    "V7IntegerResult",
    "V7RestrictedIntegerSolver",
]
