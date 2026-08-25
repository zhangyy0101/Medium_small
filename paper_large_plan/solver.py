from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from .models import LargePlanData, LargePlanSolution, VF, VFA


STATUS_NAMES = {
    GRB.LOADED: "LOADED",
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.CUTOFF: "CUTOFF",
    GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
    GRB.NODE_LIMIT: "NODE_LIMIT",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    GRB.INTERRUPTED: "INTERRUPTED",
    GRB.NUMERIC: "NUMERIC",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
}


def _optional_float_attr(model: gp.Model, name: str) -> float | None:
    try:
        return float(model.getAttr(name))
    except (AttributeError, gp.GurobiError):
        return None


def solve_large_plan(
    data: LargePlanData,
    *,
    time_limit: float = 120.0,
    mip_gap: float = 0.001,
    threads: int = 1,
    seed: int = 0,
    verbose: bool = True,
    keep_model: bool = False,
    gurobi_params: Mapping[str, Any] | None = None,
) -> LargePlanSolution:
    """Solve four strict objective levels with Gurobi multi-objective MIP."""

    model = gp.Model("paper_large_plan")
    model.Params.OutputFlag = int(verbose)
    model.Params.TimeLimit = max(0.0, float(time_limit))
    model.Params.MIPGap = max(0.0, float(mip_gap))
    model.Params.Threads = max(1, int(threads))
    model.Params.Seed = int(seed)
    for name, value in (gurobi_params or {}).items():
        model.setParam(str(name), value)

    demand20 = {(v, f): data.new_demand("20", v, f) for v in data.voyages for f in data.flows}
    demand40 = {(v, f): data.new_demand("40", v, f) for v in data.voyages for f in data.flows}
    x20: dict[VFA, gp.Var] = {}
    x40: dict[VFA, gp.Var] = {}
    for voyage in data.voyages:
        for flow in data.flows:
            for area in data.areas:
                if flow not in data.area_functions.get(area, frozenset()):
                    continue
                if demand20[voyage, flow] > 0 and data.capacity20_direct[area] > 0 and data.capacity20_equiv[area] > 0:
                    upper = min(demand20[voyage, flow], data.capacity20_direct[area], data.capacity20_equiv[area])
                    x20[voyage, flow, area] = model.addVar(
                        vtype=GRB.INTEGER,
                        lb=0,
                        ub=upper,
                        name=f"new20[{voyage},{flow},{area}]",
                    )
                if demand40[voyage, flow] > 0 and data.capacity40[area] > 0 and data.capacity20_equiv[area] >= 2:
                    upper = min(demand40[voyage, flow], data.capacity40[area], data.capacity20_equiv[area] // 2)
                    x40[voyage, flow, area] = model.addVar(
                        vtype=GRB.INTEGER,
                        lb=0,
                        ub=upper,
                        name=f"new40[{voyage},{flow},{area}]",
                    )

    shortage20: dict[VF, gp.Var] = {}
    shortage40: dict[VF, gp.Var] = {}
    for key, quantity in demand20.items():
        if quantity > 0:
            shortage20[key] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=quantity, name=f"short20[{key[0]},{key[1]}]")
    for key, quantity in demand40.items():
        if quantity > 0:
            shortage40[key] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=quantity, name=f"short40[{key[0]},{key[1]}]")

    for (voyage, flow), shortage in shortage20.items():
        model.addConstr(
            gp.quicksum(var for (v, f, _area), var in x20.items() if v == voyage and f == flow) + shortage
            == demand20[voyage, flow],
            name=f"demand20[{voyage},{flow}]",
        )
    for (voyage, flow), shortage in shortage40.items():
        model.addConstr(
            gp.quicksum(var for (v, f, _area), var in x40.items() if v == voyage and f == flow) + shortage
            == demand40[voyage, flow],
            name=f"demand40[{voyage},{flow}]",
        )

    by_area20: defaultdict[str, list[gp.Var]] = defaultdict(list)
    by_area40: defaultdict[str, list[gp.Var]] = defaultdict(list)
    by_voyage_area: defaultdict[tuple[str, str], list[gp.Var]] = defaultdict(list)
    for (voyage, _flow, area), variable in x20.items():
        by_area20[area].append(variable)
        by_voyage_area[voyage, area].append(variable)
    for (voyage, _flow, area), variable in x40.items():
        by_area40[area].append(variable)
        by_voyage_area[voyage, area].append(variable)
    for area in data.areas:
        load20 = gp.quicksum(by_area20[area])
        load40 = gp.quicksum(by_area40[area])
        model.addConstr(load20 <= data.capacity20_direct[area], name=f"capacity20_direct[{area}]")
        model.addConstr(load40 <= data.capacity40[area], name=f"capacity40[{area}]")
        model.addConstr(load20 + 2 * load40 <= data.capacity20_equiv[area], name=f"capacity_shared[{area}]")

    snapshot_by_voyage_area = {
        (voyage, area): sum(
            source.get((voyage, flow, area), 0)
            for source in (data.snapshot20, data.snapshot40)
            for flow in data.flows
        )
        for voyage in data.voyages
        for area in data.areas
    }
    used: dict[tuple[str, str], gp.Var] = {}
    for voyage in data.voyages:
        voyage_new_total = sum(demand20[voyage, flow] + demand40[voyage, flow] for flow in data.flows)
        for area in data.areas:
            variables = by_voyage_area[voyage, area]
            snapshot_used = int(snapshot_by_voyage_area[voyage, area] > 0)
            if not variables and not snapshot_used:
                continue
            used[voyage, area] = model.addVar(vtype=GRB.BINARY, name=f"used[{voyage},{area}]")
            total = gp.quicksum(variables)
            if snapshot_used:
                model.addConstr(used[voyage, area] == 1, name=f"snapshot_used[{voyage},{area}]")
            else:
                model.addConstr(total <= max(1, voyage_new_total) * used[voyage, area], name=f"used_upper[{voyage},{area}]")
                model.addConstr(used[voyage, area] <= total, name=f"used_lower[{voyage},{area}]")

    peak_utilization = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="peak_utilization")
    for area in data.areas:
        capacity = data.capacity20_equiv[area]
        if capacity > 0:
            model.addConstr(
                gp.quicksum(by_area20[area]) + 2 * gp.quicksum(by_area40[area])
                <= capacity * peak_utilization,
                name=f"peak[{area}]",
            )

    shortage_expr = gp.quicksum(shortage20.values()) + gp.quicksum(shortage40.values())
    area_count_expr = gp.quicksum(used.values())
    max_distance = max(data.distance.values(), default=1.0) or 1.0
    distance_expr = gp.quicksum(
        (data.distance[voyage, area] / max_distance) * variable
        for source in (x20, x40)
        for (voyage, _flow, area), variable in source.items()
    )
    model.ModelSense = GRB.MINIMIZE
    model.setObjectiveN(shortage_expr, index=0, priority=4, weight=1.0, abstol=0.0, reltol=0.0, name="shortage")
    model.setObjectiveN(area_count_expr, index=1, priority=3, weight=1.0, abstol=0.0, reltol=0.0, name="area_dispersion")
    model.setObjectiveN(peak_utilization, index=2, priority=2, weight=1.0, abstol=1e-6, reltol=0.0, name="peak_utilization")
    model.setObjectiveN(distance_expr, index=3, priority=1, weight=1.0, abstol=1e-6, reltol=0.0, name="berth_distance")
    model.optimize()

    status = int(model.Status)
    has_solution = model.SolCount > 0
    status_name = STATUS_NAMES.get(status, f"STATUS_{status}")
    if not has_solution:
        return LargePlanSolution(
            status=status,
            status_name=status_name,
            has_solution=False,
            runtime=float(model.Runtime),
            mip_gap=None,
            objective_value=None,
            objective_components={},
            new20={},
            new40={},
            shortage20={},
            shortage40={},
            model=model if keep_model else None,
        )

    new20_values = {key: int(round(variable.X)) for key, variable in x20.items() if variable.X > 0.5}
    new40_values = {key: int(round(variable.X)) for key, variable in x40.items() if variable.X > 0.5}
    short20_values = {key: int(round(variable.X)) for key, variable in shortage20.items() if variable.X > 0.5}
    short40_values = {key: int(round(variable.X)) for key, variable in shortage40.items() if variable.X > 0.5}
    components = {
        "shortage": float(sum(short20_values.values()) + sum(short40_values.values())),
        "voyage_area_count": float(sum(round(variable.X) for variable in used.values())),
        "peak_new_slot_utilization": float(peak_utilization.X),
        "normalized_berth_distance": float(
            sum(
                data.distance[voyage, area] / max_distance * quantity
                for source in (new20_values, new40_values)
                for (voyage, _flow, area), quantity in source.items()
            )
        ),
    }
    return LargePlanSolution(
        status=status,
        status_name=status_name,
        has_solution=True,
        runtime=float(model.Runtime),
        mip_gap=_optional_float_attr(model, "MIPGap") if model.IsMIP else 0.0,
        objective_value=float(model.ObjVal),
        objective_components=components,
        new20=new20_values,
        new40=new40_values,
        shortage20=short20_values,
        shortage40=short40_values,
        model=model if keep_model else None,
    )
