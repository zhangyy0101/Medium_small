"""Shared native-Gurobi model façade for every yard-planning solver."""

from __future__ import annotations

import math
from collections.abc import Iterable
from time import perf_counter


class MipProgressRecorder:
    """Collect a compact, solver-independent anytime trace from Gurobi."""

    def __init__(
        self,
        *,
        phase: str,
        meaningful_bound_relative_change: float = 1e-4,
    ) -> None:
        self.phase = str(phase)
        self.meaningful_bound_relative_change = float(
            meaningful_bound_relative_change
        )
        self.started_at = perf_counter()
        self.events: list[dict[str, object]] = []
        self._first_incumbent: float | None = None
        self._best_incumbent: float | None = None
        self._time_to_first_solution: float | None = None
        self._time_to_best_solution: float | None = None
        self._last_bound: float | None = None
        self._last_node_count = 0.0
        self._last_solution_count = 0

    @staticmethod
    def _finite(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and abs(number) < 1e99 else None

    def _append(
        self,
        *,
        elapsed_seconds: float,
        incumbent: float | None,
        best_bound: float | None,
        node_count: float,
        solution_count: int,
        event_type: str,
    ) -> None:
        self._last_node_count = max(self._last_node_count, float(node_count))
        self._last_solution_count = max(self._last_solution_count, int(solution_count))
        self.events.append(
            {
                "elapsed_seconds": max(0.0, float(elapsed_seconds)),
                "incumbent": incumbent,
                "best_bound": best_bound,
                "node_count": float(node_count),
                "solution_count": int(solution_count),
                "event_type": str(event_type),
            }
        )

    def _record_incumbent(
        self,
        *,
        elapsed_seconds: float,
        incumbent: float | None,
        best_bound: float | None,
        node_count: float,
        solution_count: int,
    ) -> None:
        if incumbent is None:
            return
        if self._first_incumbent is None:
            self._first_incumbent = incumbent
            self._best_incumbent = incumbent
            self._time_to_first_solution = elapsed_seconds
            self._time_to_best_solution = elapsed_seconds
            event_type = "first_incumbent"
        elif self._best_incumbent is None or incumbent < self._best_incumbent - 1e-9:
            self._best_incumbent = incumbent
            self._time_to_best_solution = elapsed_seconds
            event_type = "new_incumbent"
        else:
            return
        self._append(
            elapsed_seconds=elapsed_seconds,
            incumbent=incumbent,
            best_bound=best_bound,
            node_count=node_count,
            solution_count=solution_count,
            event_type=event_type,
        )

    def _record_bound(
        self,
        *,
        elapsed_seconds: float,
        incumbent: float | None,
        best_bound: float | None,
        node_count: float,
        solution_count: int,
    ) -> None:
        if best_bound is None:
            return
        threshold = max(
            1e-8,
            self.meaningful_bound_relative_change
            * max(1.0, abs(self._last_bound or 0.0)),
        )
        if self._last_bound is not None and abs(best_bound - self._last_bound) < threshold:
            return
        self._last_bound = best_bound
        self._append(
            elapsed_seconds=elapsed_seconds,
            incumbent=incumbent,
            best_bound=best_bound,
            node_count=node_count,
            solution_count=solution_count,
            event_type="bound_change",
        )

    def __call__(self, model, where: int) -> None:
        """Native Gurobi callback entry point."""

        gp = model._gp if hasattr(model, "_gp") else None
        if gp is None:
            import gurobipy as gp

        callback = gp.GRB.Callback
        try:
            if where == callback.MIPSOL:
                elapsed = float(model.cbGet(callback.RUNTIME))
                incumbent = self._finite(model.cbGet(callback.MIPSOL_OBJ))
                bound = self._finite(model.cbGet(callback.MIPSOL_OBJBND))
                nodes = float(model.cbGet(callback.MIPSOL_NODCNT))
                self._record_incumbent(
                    elapsed_seconds=elapsed,
                    incumbent=incumbent,
                    best_bound=bound,
                    node_count=nodes,
                    solution_count=self._last_solution_count + 1,
                )
            elif where == callback.MIP:
                elapsed = float(model.cbGet(callback.RUNTIME))
                incumbent = self._finite(model.cbGet(callback.MIP_OBJBST))
                bound = self._finite(model.cbGet(callback.MIP_OBJBND))
                nodes = float(model.cbGet(callback.MIP_NODCNT))
                solutions = int(model.cbGet(callback.MIP_SOLCNT))
                self._record_incumbent(
                    elapsed_seconds=elapsed,
                    incumbent=incumbent,
                    best_bound=bound,
                    node_count=nodes,
                    solution_count=solutions,
                )
                self._record_bound(
                    elapsed_seconds=elapsed,
                    incumbent=incumbent,
                    best_bound=bound,
                    node_count=nodes,
                    solution_count=solutions,
                )
        except Exception:
            # Progress tracing is diagnostic only and must never interrupt a solve.
            return

    def finalize(self, model: "GurobiModel") -> dict[str, object]:
        """Append the terminal solver state and return a JSON-safe summary."""

        elapsed = model.getRuntime()
        solver_solution_count = model.getSolutionCount()
        solution_count = max(self._last_solution_count, solver_solution_count)
        incumbent = (
            self._finite(model.getObjectiveValue())
            if solver_solution_count > 0
            else None
        )
        bound = self._finite(model.getBestBound())
        self._record_incumbent(
            elapsed_seconds=elapsed,
            incumbent=incumbent,
            best_bound=bound,
            node_count=model.getNodeCount(),
            solution_count=solution_count,
        )
        self._append(
            elapsed_seconds=elapsed,
            incumbent=incumbent,
            best_bound=bound,
            node_count=model.getNodeCount(),
            solution_count=solution_count,
            event_type="final",
        )
        return {
            "phase": self.phase,
            "time_to_first_solution": self._time_to_first_solution,
            "time_to_best_solution": self._time_to_best_solution,
            "first_incumbent": self._first_incumbent,
            "best_incumbent": self._best_incumbent,
            "node_count": float(model.getNodeCount()),
            "solution_count": int(solution_count),
            "incumbent_trajectory": list(self.events),
            "wall_seconds": max(0.0, perf_counter() - self.started_at),
        }


class GurobiModel:
    """Small, typed façade over the native Gurobi model used by all solvers."""

    def __init__(self, name: str) -> None:
        import gurobipy as gp

        self._gp = gp
        self._model = gp.Model(name)

    def addVar(self, **kwargs):
        return self._model.addVar(**kwargs)

    def addPricedVar(self, terms: Iterable[tuple[float, object]], **kwargs):
        """Add one variable with coefficients in existing master rows."""
        column = self._gp.Column()
        for coefficient, constraint in terms:
            if constraint is not None and abs(float(coefficient)) > 0.0:
                column.addTerms(float(coefficient), constraint)
        return self._model.addVar(column=column, **kwargs)

    def addConstr(self, expression, name: str | None = None):
        return self._model.addConstr(expression, name=name or "")

    def getVars(self):
        return self._model.getVars()

    def captureLpWarmStart(self) -> dict[str, dict[str, float]]:
        """Capture a name-addressed primal/dual LP start for a rebuilt model."""

        self._model.update()
        return {
            "primal": {
                variable.VarName: float(variable.X)
                for variable in self._model.getVars()
            },
            "dual": {
                constraint.ConstrName: float(constraint.Pi)
                for constraint in self._model.getConstrs()
            },
        }

    def applyLpWarmStart(
        self,
        warm_start: dict[str, dict[str, float]] | None,
    ) -> dict[str, int]:
        """Apply matching parent LP values and initialize newly added rows."""

        if not warm_start:
            return {"matched_primal": 0, "matched_dual": 0}
        self._model.update()
        primal = warm_start.get("primal", {})
        dual = warm_start.get("dual", {})
        matched_primal = 0
        matched_dual = 0
        for variable in self._model.getVars():
            value = primal.get(variable.VarName)
            if value is None:
                value = max(0.0, float(variable.LB))
            else:
                matched_primal += 1
                value = min(float(variable.UB), max(float(variable.LB), value))
            variable.PStart = float(value)
        for constraint in self._model.getConstrs():
            value = dual.get(constraint.ConstrName)
            if value is None:
                value = 0.0
            else:
                matched_dual += 1
            constraint.DStart = float(value)
        self._model.Params.LPWarmStart = 2
        return {
            "matched_primal": matched_primal,
            "matched_dual": matched_dual,
        }

    def getFingerprint(self) -> int:
        """Return Gurobi's structural model fingerprint."""
        self._model.update()
        return int(self._model.Fingerprint)

    def getPoolValue(self, variable, solution_number: int) -> float:
        self._model.Params.SolutionNumber = int(solution_number)
        return float(variable.Xn)

    def getPoolObjective(self, solution_number: int) -> float:
        self._model.Params.SolutionNumber = int(solution_number)
        return float(self._model.PoolObjVal)

    def update(self) -> None:
        self._model.update()

    def removeConstraints(self, constraints: Iterable[object]) -> None:
        self._model.remove(list(constraints))

    @staticmethod
    def getVarObjective(variable) -> float:
        return float(variable.Obj)

    @staticmethod
    def setVarObjective(variable, coefficient: float) -> None:
        variable.Obj = float(coefficient)

    def setMinimize(self) -> None:
        self._model.ModelSense = self._gp.GRB.MINIMIZE

    def setMaximize(self) -> None:
        self._model.ModelSense = self._gp.GRB.MAXIMIZE

    def setParam(self, name: str, value: object) -> None:
        self._model.setParam(name, value)

    def hideOutput(self) -> None:
        self._model.Params.OutputFlag = 0

    def optimize(self, callback=None) -> None:
        if callback is None:
            self._model.optimize()
        else:
            self._model.optimize(callback)

    def terminate(self) -> None:
        self._model.terminate()

    def apply_mip_starts(
        self,
        starts: Iterable[dict[object, float]],
        *,
        deadline: float | None = None,
    ) -> dict[str, object]:
        """Submit starts, stopping cleanly if their shared deadline expires."""

        materialized = list(starts)
        self._model.update()
        self._model.NumStart = len(materialized)
        assigned_value_counts = []
        variables = self._model.getVars()
        deadline_exhausted = False
        for start_number, start in enumerate(materialized):
            if deadline is not None and perf_counter() >= deadline:
                deadline_exhausted = True
                break
            self._model.Params.StartNumber = int(start_number)
            interrupted = False
            for index, variable in enumerate(variables):
                if (
                    deadline is not None
                    and index % 256 == 0
                    and perf_counter() >= deadline
                ):
                    interrupted = True
                    deadline_exhausted = True
                    break
                variable.Start = self._gp.GRB.UNDEFINED
            if interrupted:
                break
            assigned = 0
            for index, (variable, value) in enumerate(start.items()):
                if (
                    deadline is not None
                    and index % 256 == 0
                    and perf_counter() >= deadline
                ):
                    interrupted = True
                    deadline_exhausted = True
                    break
                variable.Start = float(value)
                assigned += 1
            if interrupted:
                break
            assigned_value_counts.append(assigned)
        self._model.NumStart = len(assigned_value_counts)
        if assigned_value_counts:
            self._model.Params.StartNumber = 0
        return {
            "requested_mip_start_count": len(materialized),
            "provided_mip_start_count": len(assigned_value_counts),
            "assigned_value_counts": assigned_value_counts,
            "deadline_exhausted": deadline_exhausted,
            "solver_acceptance_observed": False,
        }

    def applyMipStarts(
        self,
        starts: Iterable[dict[object, float]],
        *,
        deadline: float | None = None,
    ) -> dict[str, object]:
        """Backward-compatible alias for the snake-case façade method."""

        return self.apply_mip_starts(starts, deadline=deadline)

    def getStatusName(self) -> str:
        status_names = {
            self._gp.GRB.LOADED: "loaded",
            self._gp.GRB.OPTIMAL: "optimal",
            self._gp.GRB.INFEASIBLE: "infeasible",
            self._gp.GRB.INF_OR_UNBD: "inforunbd",
            self._gp.GRB.UNBOUNDED: "unbounded",
            self._gp.GRB.CUTOFF: "cutoff",
            self._gp.GRB.ITERATION_LIMIT: "iterationlimit",
            self._gp.GRB.NODE_LIMIT: "nodelimit",
            self._gp.GRB.TIME_LIMIT: "timelimit",
            self._gp.GRB.SOLUTION_LIMIT: "solutionlimit",
            self._gp.GRB.INTERRUPTED: "interrupted",
            self._gp.GRB.NUMERIC: "numeric",
            self._gp.GRB.SUBOPTIMAL: "suboptimal",
        }
        user_objective_limit = getattr(
            self._gp.GRB, "USER_OBJ_LIMIT", None
        )
        if user_objective_limit is not None:
            status_names[user_objective_limit] = "userobjlimit"
        return status_names.get(self._model.Status, str(self._model.Status))

    def getSolutionCount(self) -> int:
        return int(self._model.SolCount)

    def getObjectiveValue(self) -> float:
        return float(self._model.ObjVal)

    def getMipGap(self) -> float:
        return float(self._model.MIPGap) if self._model.IsMIP else 0.0

    def getBestBound(self) -> float:
        return float(self._model.ObjBound)

    def getNodeCount(self) -> float:
        return float(self._model.NodeCount)

    def getRuntime(self) -> float:
        return float(self._model.Runtime)

    @staticmethod
    def getValue(variable) -> float:
        return float(variable.X)

    @staticmethod
    def getLinearDual(constraint) -> float:
        return float(constraint.Pi)

    def dispose(self) -> None:
        self._model.dispose()


__all__ = ["GurobiModel", "MipProgressRecorder"]
