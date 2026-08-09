"""Shared native-Gurobi model façade for every yard-planning solver."""

from __future__ import annotations

from collections.abc import Iterable


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

    def optimize(self) -> None:
        self._model.optimize()

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

    @staticmethod
    def getValue(variable) -> float:
        return float(variable.X)

    @staticmethod
    def getLinearDual(constraint) -> float:
        return float(constraint.Pi)

    def dispose(self) -> None:
        self._model.dispose()


__all__ = ["GurobiModel"]
