"""V6 projected restricted master and exact row-aware zone pricing.

This module implements only the root-LP correctness layer.  It deliberately
does not import or reuse the V5 static pool, F&O, adaptive expansion, local
branching, stabilization, cuts, or branch-and-price code.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Iterable, Mapping, Sequence

from .gurobi_backend import GurobiModel, MipProgressRecorder
from .models import ExportGroup, ProblemData
from .row_aware_zones import (
    RowAwareBayAtom,
    RowAwareZone,
    build_v6_row_aware_bay_atoms,
    make_row_aware_zone,
    v6_footprint,
)
from .v6_model import (
    V6_MODEL_SCHEMA_VERSION,
    V6ModelEvaluator,
    V6ObjectiveConfig,
    V6PeakUtilizationPolicy,
)


RowKey = object
DualSnapshot = Mapping[str, Mapping[RowKey, float]]


class V6RootCgIncompleteError(RuntimeError):
    """Exact root CG stopped before its pricing certificate was complete."""

    def __init__(self, message: str, diagnostics: Mapping[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class V6RootCgConfig:
    """Deterministic execution controls for V6 root column generation."""

    maximum_phase_one_iterations: int = 200
    maximum_business_iterations: int = 500
    reduced_cost_tolerance: float = 1e-8
    feasibility_tolerance: float = 1e-8
    pricing_time_limit: float = 60.0
    phase_one_columns_per_group: int = 512
    business_columns_per_group: int = 512
    maximum_patterns_per_interval: int = 8
    total_time_limit: float | None = None
    solver_threads: int = 1
    solver_seed: int = 0
    verbose: bool = False
    objective: V6ObjectiveConfig = field(default_factory=V6ObjectiveConfig)

    def validate(self) -> None:
        if int(self.maximum_phase_one_iterations) <= 0:
            raise ValueError("V6 Phase-I iteration limit must be positive")
        if int(self.maximum_business_iterations) <= 0:
            raise ValueError("V6 business iteration limit must be positive")
        if not math.isfinite(float(self.reduced_cost_tolerance)) or float(
            self.reduced_cost_tolerance
        ) <= 0.0:
            raise ValueError("V6 reduced-cost tolerance must be positive")
        if not math.isfinite(float(self.feasibility_tolerance)) or float(
            self.feasibility_tolerance
        ) <= 0.0:
            raise ValueError("V6 feasibility tolerance must be positive")
        if not math.isfinite(float(self.pricing_time_limit)) or float(
            self.pricing_time_limit
        ) <= 0.0:
            raise ValueError("V6 pricing time limit must be positive")
        if int(self.phase_one_columns_per_group) <= 0:
            raise ValueError("V6 Phase-I pricing batch must be positive")
        if int(self.business_columns_per_group) <= 0:
            raise ValueError("V6 business pricing batch must be positive")
        if int(self.maximum_patterns_per_interval) <= 0:
            raise ValueError("V6 interval-pattern depth must be positive")
        if self.total_time_limit is not None and (
            not math.isfinite(float(self.total_time_limit))
            or float(self.total_time_limit) <= 0.0
        ):
            raise ValueError("V6 root-CG total time limit must be positive")
        if int(self.solver_threads) < 0:
            raise ValueError("V6 solver_threads must be nonnegative")
        self.objective.validate()


@dataclass(frozen=True)
class V6MasterSolution:
    phase: str
    objective: float
    total_artificial_deficit: float
    zone_values: Mapping[int, float]
    group_bay_flow: Mapping[tuple[str, str], float]
    import_reservation: Mapping[tuple[str, str, str], float]
    duals: Mapping[str, Mapping[RowKey, float]]
    runtime_seconds: float


@dataclass(frozen=True)
class V6IntegerMasterSolution:
    phase: str
    status: str
    objective: float
    best_bound: float
    mip_gap: float
    proven_optimal: bool
    selected_zone_ids: tuple[int, ...]
    group_bay_flow: Mapping[tuple[str, str], int]
    import_reservation: Mapping[tuple[str, str, str], int]
    runtime_seconds: float
    progress: Mapping[str, object]


@dataclass(frozen=True)
class V6PricingResult:
    group_id: str
    area_no: str
    reduced_cost: float
    zone: RowAwareZone
    solver_objective: float
    runtime_seconds: float


@dataclass(frozen=True)
class V6RootCgResult:
    objective: float
    zones: tuple[RowAwareZone, ...]
    zone_values: Mapping[int, float]
    group_bay_flow: Mapping[tuple[str, str], float]
    import_reservation: Mapping[tuple[str, str, str], float]
    diagnostics: Mapping[str, object]


def _candidate_bays_by_group(
    atoms: Sequence[RowAwareBayAtom],
) -> dict[str, set[str]]:
    output: defaultdict[str, set[str]] = defaultdict(set)
    for atom in atoms:
        output[atom.group_id].add(atom.anchor_bay_key)
    return dict(output)


def _zone_direct_objective(
    evaluator: V6ModelEvaluator,
    zone: RowAwareZone,
    phase: str,
) -> float:
    if phase == "phase_one":
        return 0.0
    if phase != "business":
        raise ValueError(f"unknown V6 restricted-master phase: {phase}")
    return (
        evaluator.zone_fixed_activation_objective_coefficient()
        + evaluator.unused_capacity_unit_objective_coefficient()
        * int(zone.capacity)
    )


def _zone_column_terms(
    problem: ProblemData,
    groups_by_id: Mapping[str, ExportGroup],
    zone: RowAwareZone,
) -> tuple[tuple[str, RowKey, float], ...]:
    """Return every restricted-master coefficient of one zone column."""

    group = groups_by_id[zone.group_id]
    terms: list[tuple[str, RowKey, float]] = []
    capacity_by_bay = dict(zone.anchor_bay_capacities)
    for bay_key in zone.anchor_bay_keys:
        capacity = float(capacity_by_bay[bay_key])
        pair = (zone.group_id, bay_key)
        terms.extend(
            (
                ("group_bay_flow_lower", pair, 1.0),
                ("group_bay_flow_upper", pair, -capacity),
                ("same_group_bay_nonoverlap", pair, 1.0),
                (
                    "reserved_anchor_size_capacity",
                    (bay_key, str(group.size)),
                    capacity,
                ),
            )
        )
        for physical_bay_key in v6_footprint(problem, bay_key, group.size):
            terms.append(
                (
                    "reserved_physical_capacity",
                    physical_bay_key,
                    capacity,
                )
            )
    for resource in zone.resources:
        terms.append(("physical_row_exclusive", resource, 1.0))
    aggregated: Counter[tuple[str, RowKey]] = Counter()
    for family, key, coefficient in terms:
        aggregated[(family, key)] += float(coefficient)
    return tuple(
        (family, key, coefficient)
        for (family, key), coefficient in sorted(
            aggregated.items(), key=lambda item: (item[0][0], repr(item[0][1]))
        )
        if abs(coefficient) > 0.0
    )


def _reduced_cost(
    problem: ProblemData,
    groups_by_id: Mapping[str, ExportGroup],
    evaluator: V6ModelEvaluator,
    zone: RowAwareZone,
    duals: DualSnapshot,
    phase: str,
) -> float:
    value = _zone_direct_objective(evaluator, zone, phase)
    for family, key, coefficient in _zone_column_terms(
        problem, groups_by_id, zone
    ):
        value -= coefficient * float(duals.get(family, {}).get(key, 0.0))
    return float(value)


class V6ProjectedRestrictedMaster:
    """Column-friendly LP with the same V6 integer feasible decisions.

    Zone-specific flows are projected onto global ``q[group, bay]`` variables:
    ``sum(x_z) <= q_gb <= sum(C_zb x_z)``.  The same-group/bay nonoverlap row
    makes this an exact integer projection while retaining fixed master rows.
    """

    def __init__(
        self,
        problem: ProblemData,
        atoms: Sequence[RowAwareBayAtom],
        anchor_capacity_limits: Mapping[tuple[str, str], int],
        evaluator: V6ModelEvaluator,
        peak_policy: V6PeakUtilizationPolicy,
        *,
        phase: str,
        initial_zones: Sequence[RowAwareZone] = (),
        solver_threads: int = 1,
        solver_seed: int = 0,
        verbose: bool = False,
        integral: bool = False,
        time_limit: float | None = None,
        mip_gap: float = 0.0,
    ) -> None:
        if phase not in {"phase_one", "business"}:
            raise ValueError(f"unknown V6 restricted-master phase: {phase}")
        if integral and phase != "business":
            raise ValueError("V6 integer master is defined only for business phase")
        if time_limit is not None and (
            not math.isfinite(float(time_limit)) or float(time_limit) <= 0.0
        ):
            raise ValueError("V6 integer-master time limit must be positive")
        if not math.isfinite(float(mip_gap)) or not 0.0 <= float(mip_gap) <= 1.0:
            raise ValueError("V6 integer-master MIP gap must lie in [0, 1]")
        if abs(
            float(peak_policy.headroom_fraction)
            - float(evaluator.config.peak_utilization_headroom_fraction)
        ) > 1e-12:
            raise ValueError("V6 peak policy and objective config disagree")
        self.problem = problem
        self.atoms = tuple(atoms)
        self.anchor_capacity_limits = dict(anchor_capacity_limits)
        self.evaluator = evaluator
        self.peak_policy = peak_policy
        self.phase = phase
        self.integral = bool(integral)
        self.groups = evaluator.groups
        self.groups_by_id = evaluator.groups_by_id
        model_kind = "integer" if self.integral else "lp"
        self.model = GurobiModel(f"v6_projected_{model_kind}_{phase}")
        if not verbose:
            self.model.hideOutput()
        self.model.setMinimize()
        self.model.setParam("Seed", int(solver_seed))
        if int(solver_threads) > 0:
            self.model.setParam("Threads", int(solver_threads))
        self.model.setParam("FeasibilityTol", 1e-9)
        if self.integral:
            self.model.setParam("MIPGap", float(mip_gap))
            self.model.setParam("IntFeasTol", 1e-9)
            if time_limit is not None:
                self.model.setParam("TimeLimit", float(time_limit))
        else:
            self.model.setParam("OptimalityTol", 1e-9)
            self.model.setParam("Method", 1)
        self.rows: defaultdict[str, dict[RowKey, object]] = defaultdict(dict)
        self.zone_variables: dict[int, object] = {}
        self.zones_by_id: dict[int, RowAwareZone] = {}
        self.signatures: set[tuple[int, ...]] = set()
        self.group_bay_flow: dict[tuple[str, str], object] = {}
        self.artificial_deficit: dict[str, object] = {}
        self.import_reservation: dict[tuple[str, str, str], object] = {}
        self.export_size_state: dict[tuple[str, str], object] = {}
        self.export_height_state: dict[tuple[str, str], object] = {}
        self.export_use: dict[str, object] = {}
        self.import_use: dict[str, object] = {}
        self.import_size_state: dict[tuple[str, str], object] = {}
        self.voyage_area_use: dict[tuple[str, str], object] = {}
        self._build_fixed_master()
        for zone in initial_zones:
            self.add_zone(zone)
        self.model.update()

    def _add_row(
        self,
        family: str,
        key: RowKey,
        expression,
        *,
        sense: str,
        rhs: float,
    ) -> object:
        if key in self.rows[family]:
            raise ValueError(f"duplicate V6 master row: {family}, {key}")
        if sense == "<=":
            relation = expression <= float(rhs)
        elif sense == ">=":
            relation = expression >= float(rhs)
        elif sense == "==":
            relation = expression == float(rhs)
        else:
            raise ValueError(f"unknown row sense: {sense}")
        row = self.model.addConstr(
            relation,
            name=f"{family}_{len(self.rows[family])}",
        )
        self.rows[family][key] = row
        return row

    def _build_fixed_master(self) -> None:
        gp = self.model._gp
        quicksum = gp.quicksum
        atoms_by_pair: defaultdict[
            tuple[str, str], list[RowAwareBayAtom]
        ] = defaultdict(list)
        for atom in self.atoms:
            atoms_by_pair[(atom.group_id, atom.anchor_bay_key)].append(atom)
        missing = [
            group.group_id
            for group in self.groups
            if not any(key[0] == group.group_id for key in atoms_by_pair)
        ]
        if missing:
            raise ValueError(f"V6 groups have no pricing atoms: {missing}")

        for pair in sorted(atoms_by_pair):
            group = self.groups_by_id[pair[0]]
            self.group_bay_flow[pair] = self.model.addVar(
                lb=0.0,
                ub=float(group.demand),
                vtype="I" if self.integral else "C",
                obj=(
                    self.evaluator.group_bay_flow_objective_coefficient(*pair)
                    if self.phase == "business"
                    else 0.0
                ),
                name=f"q_group_bay_{len(self.group_bay_flow)}",
            )
        for group in self.groups:
            if self.phase == "phase_one":
                self.artificial_deficit[group.group_id] = self.model.addVar(
                    lb=0.0,
                    ub=float(group.demand),
                    obj=1.0,
                    vtype="C",
                    name=f"phase_one_deficit_{len(self.artificial_deficit)}",
                )
            expression = quicksum(
                variable
                for (group_id, _bay_key), variable in self.group_bay_flow.items()
                if group_id == group.group_id
            )
            if self.phase == "phase_one":
                expression += self.artificial_deficit[group.group_id]
            self._add_row(
                "export_group_demand",
                group.group_id,
                expression,
                sense="==",
                rhs=float(group.demand),
            )

        for pair in sorted(atoms_by_pair):
            q = self.group_bay_flow[pair]
            self._add_row(
                "group_bay_flow_lower", pair, -q, sense="<=", rhs=0.0
            )
            self._add_row(
                "group_bay_flow_upper", pair, q, sense="<=", rhs=0.0
            )
            self._add_row(
                "same_group_bay_nonoverlap",
                pair,
                gp.LinExpr(),
                sense="<=",
                rhs=1.0,
            )

        resources = sorted(
            {resource for atom in self.atoms for resource in atom.resources}
        )
        for resource in resources:
            self._add_row(
                "physical_row_exclusive",
                resource,
                gp.LinExpr(),
                sense="<=",
                rhs=1.0,
            )
        physical_bays = sorted({bay_key for bay_key, _row_no in resources})
        for bay_key in physical_bays:
            problem_capacity = int(
                self.problem.bays[bay_key].physical_capacity
            )
            self._add_row(
                "reserved_physical_capacity",
                bay_key,
                gp.LinExpr(),
                sense="<=",
                rhs=float(problem_capacity),
            )
            if int(problem_capacity) <= 0:
                raise ValueError(f"V6 physical bay has no capacity: {bay_key}")
        anchor_size_capacity: dict[tuple[str, str], int] = {}
        for (group_id, bay_key), _atom_limit in sorted(
            self.anchor_capacity_limits.items()
        ):
            group = self.groups_by_id[group_id]
            key = (bay_key, str(group.size))
            capacity = int(
                self.problem.bays[bay_key].cap_by_size.get(group.size, 0)
            )
            anchor_size_capacity[key] = capacity
        for key, capacity in sorted(anchor_size_capacity.items()):
            self._add_row(
                "reserved_anchor_size_capacity",
                key,
                gp.LinExpr(),
                sense="<=",
                rhs=float(capacity),
            )

        export_q_by_physical: defaultdict[str, list[object]] = defaultdict(list)
        export_q_by_physical_size: defaultdict[
            tuple[str, str], list[object]
        ] = defaultdict(list)
        export_q_by_physical_height: defaultdict[
            tuple[str, str], list[object]
        ] = defaultdict(list)
        export_q_by_voyage_area: defaultdict[
            tuple[str, str], list[object]
        ] = defaultdict(list)
        planned_load_by_area: defaultdict[
            str, list[tuple[int, object]]
        ] = defaultdict(list)
        for (group_id, bay_key), variable in self.group_bay_flow.items():
            group = self.groups_by_id[group_id]
            footprint = v6_footprint(self.problem, bay_key, group.size)
            for physical_bay_key in footprint:
                export_q_by_physical[physical_bay_key].append(variable)
                export_q_by_physical_size[
                    (physical_bay_key, str(group.size))
                ].append(variable)
                export_q_by_physical_height[
                    (physical_bay_key, str(group.height))
                ].append(variable)
            area_no = str(self.problem.bays[bay_key].area_no)
            export_q_by_voyage_area[(group.voyage_id, area_no)].append(variable)
            planned_load_by_area[area_no].append((len(footprint), variable))

        total_export_demand = max(
            1, sum(int(group.demand) for group in self.groups)
        )
        size_states = {
            key: self.model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="B" if self.integral else "C",
                name=f"export_size_{index}",
            )
            for index, key in enumerate(sorted(export_q_by_physical_size))
        }
        height_states = {
            key: self.model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="B" if self.integral else "C",
                name=f"export_height_{index}",
            )
            for index, key in enumerate(sorted(export_q_by_physical_height))
        }
        self.export_size_state = size_states
        self.export_height_state = height_states
        for key, variables in sorted(export_q_by_physical_size.items()):
            self._add_row(
                "export_size_state_link",
                key,
                quicksum(variables) - total_export_demand * size_states[key],
                sense="<=",
                rhs=0.0,
            )
        for bay_key in sorted({key[0] for key in size_states}):
            self._add_row(
                "export_size_state_choice",
                bay_key,
                quicksum(
                    variable
                    for (physical, _size), variable in size_states.items()
                    if physical == bay_key
                ),
                sense="<=",
                rhs=1.0,
            )
        for key, variables in sorted(export_q_by_physical_height.items()):
            self._add_row(
                "export_height_state_link",
                key,
                quicksum(variables) - total_export_demand * height_states[key],
                sense="<=",
                rhs=0.0,
            )
        for bay_key in sorted({key[0] for key in height_states}):
            self._add_row(
                "export_height_state_choice",
                bay_key,
                quicksum(
                    variable
                    for (physical, _height), variable in height_states.items()
                    if physical == bay_key
                ),
                sense="<=",
                rhs=1.0,
            )

        import_rows = [
            (flow, size, bay_key, capacity)
            for (flow, size), candidates in sorted(
                self.evaluator.import_candidates.items()
            )
            for bay_key, capacity in candidates
        ]
        for flow, size, bay_key, capacity in import_rows:
            self.import_reservation[(flow, size, bay_key)] = self.model.addVar(
                lb=0.0,
                ub=float(capacity),
                vtype="I" if self.integral else "C",
                name=f"p_import_{len(self.import_reservation)}",
            )
        import_by_flow_size: defaultdict[
            tuple[str, str], list[object]
        ] = defaultdict(list)
        import_by_anchor_size: defaultdict[
            tuple[str, str], list[object]
        ] = defaultdict(list)
        import_by_physical: defaultdict[str, list[object]] = defaultdict(list)
        import_by_physical_size: defaultdict[
            tuple[str, str], list[object]
        ] = defaultdict(list)
        for (flow, size, bay_key), variable in self.import_reservation.items():
            import_by_flow_size[(flow, size)].append(variable)
            import_by_anchor_size[(bay_key, size)].append(variable)
            footprint = v6_footprint(self.problem, bay_key, size)
            for physical_bay_key in footprint:
                import_by_physical[physical_bay_key].append(variable)
                import_by_physical_size[(physical_bay_key, size)].append(variable)
            planned_load_by_area[str(self.problem.bays[bay_key].area_no)].append(
                (len(footprint), variable)
            )
        for raw_key, demand in sorted(
            self.problem.import_demand_by_flow_size.items()
        ):
            key = tuple(map(str, raw_key))
            self._add_row(
                "import_demand",
                key,
                quicksum(import_by_flow_size.get(key, [])),
                sense="==",
                rhs=float(demand),
            )
        for key, variables in sorted(import_by_anchor_size.items()):
            self._add_row(
                "import_anchor_size_capacity",
                key,
                quicksum(variables),
                sense="<=",
                rhs=float(self.problem.bays[key[0]].cap_by_size.get(key[1], 0)),
            )

        allocation_bays = sorted(set(export_q_by_physical) | set(import_by_physical))
        export_use = {
            bay_key: self.model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="B" if self.integral else "C",
                name=f"export_use_{index}",
            )
            for index, bay_key in enumerate(allocation_bays)
        }
        import_use = {
            bay_key: self.model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="B" if self.integral else "C",
                name=f"import_use_{index}",
            )
            for index, bay_key in enumerate(allocation_bays)
        }
        import_size_states = {
            key: self.model.addVar(
                lb=0.0,
                ub=1.0,
                vtype="B" if self.integral else "C",
                name=f"import_size_{index}",
            )
            for index, key in enumerate(sorted(import_by_physical_size))
        }
        self.export_use = export_use
        self.import_use = import_use
        self.import_size_state = import_size_states
        for bay_key in allocation_bays:
            export_load = quicksum(export_q_by_physical.get(bay_key, []))
            import_load = quicksum(import_by_physical.get(bay_key, []))
            if export_q_by_physical.get(bay_key):
                self._add_row(
                    "export_bay_use_link",
                    bay_key,
                    export_load - total_export_demand * export_use[bay_key],
                    sense="<=",
                    rhs=0.0,
                )
                self._add_row(
                    "export_bay_use_presence",
                    bay_key,
                    export_use[bay_key] - export_load,
                    sense="<=",
                    rhs=0.0,
                )
            else:
                self._add_row(
                    "export_bay_use_zero",
                    bay_key,
                    export_use[bay_key],
                    sense="==",
                    rhs=0.0,
                )
            if import_by_physical.get(bay_key):
                capacity = max(1, int(self.problem.bays[bay_key].physical_capacity))
                self._add_row(
                    "import_physical_capacity",
                    bay_key,
                    import_load - capacity * import_use[bay_key],
                    sense="<=",
                    rhs=0.0,
                )
                self._add_row(
                    "import_bay_use_presence",
                    bay_key,
                    import_use[bay_key] - import_load,
                    sense="<=",
                    rhs=0.0,
                )
                for key in sorted(
                    key for key in import_size_states if key[0] == bay_key
                ):
                    self._add_row(
                        "import_size_state_link",
                        key,
                        quicksum(import_by_physical_size[key])
                        - capacity * import_size_states[key],
                        sense="<=",
                        rhs=0.0,
                    )
                    self._add_row(
                        "import_size_state_presence",
                        key,
                        import_size_states[key]
                        - quicksum(import_by_physical_size[key]),
                        sense="<=",
                        rhs=0.0,
                    )
                self._add_row(
                    "import_size_state_choice",
                    bay_key,
                    quicksum(
                        variable
                        for (physical, _size), variable in import_size_states.items()
                        if physical == bay_key
                    )
                    - import_use[bay_key],
                    sense="<=",
                    rhs=0.0,
                )
            else:
                self._add_row(
                    "import_bay_use_zero",
                    bay_key,
                    import_use[bay_key],
                    sense="==",
                    rhs=0.0,
                )
            self._add_row(
                "export_import_bay_exclusive",
                bay_key,
                export_use[bay_key] + import_use[bay_key],
                sense="<=",
                rhs=1.0,
            )

        if self.phase == "business":
            demand_by_voyage: Counter[str] = Counter()
            for group in self.groups:
                demand_by_voyage[group.voyage_id] += int(group.demand)
            for index, (key, variables) in enumerate(
                sorted(export_q_by_voyage_area.items())
            ):
                use = self.model.addVar(
                    lb=0.0,
                    ub=1.0,
                    vtype="B" if self.integral else "C",
                    obj=self.evaluator.voyage_area_objective_coefficient(),
                    name=f"voyage_area_use_{index}",
                )
                self.voyage_area_use[key] = use
                assigned = quicksum(variables)
                self._add_row(
                    "voyage_area_link",
                    key,
                    assigned - int(demand_by_voyage[key[0]]) * use,
                    sense="<=",
                    rhs=0.0,
                )
                self._add_row(
                    "voyage_area_presence",
                    key,
                    use - assigned,
                    sense="<=",
                    rhs=0.0,
                )
            self.model.addVar(
                lb=1.0,
                ub=1.0,
                obj=self.evaluator.objective_constant(),
                name="business_objective_constant",
            )

        area_capacity: Counter[str] = Counter()
        for bay in self.problem.bays.values():
            area_capacity[str(bay.area_no)] += int(bay.physical_capacity)
        for area_no, terms in sorted(planned_load_by_area.items()):
            capacity = int(area_capacity[area_no])
            if capacity <= 0:
                raise ValueError(f"V6 used area has no capacity: {area_no}")
            self._add_row(
                "peak_utilization",
                area_no,
                quicksum(coefficient * variable for coefficient, variable in terms),
                sense="<=",
                rhs=capacity * float(self.peak_policy.epsilon_cap),
            )

        self.model.update()

    def apply_integer_warm_start(
        self,
        *,
        selected_zone_signatures: Iterable[tuple[int, ...]],
        group_bay_flow: Mapping[tuple[str, str], int],
        import_reservation: Mapping[tuple[str, str, str], int],
    ) -> dict[str, object]:
        """Install a complete V6 incumbent start on an integer master."""

        if not self.integral or self.phase != "business":
            raise ValueError(
                "V6 integer warm starts require an integral business master"
            )
        signature_to_zone_id = {
            tuple(zone.candidate_indices): zone_id
            for zone_id, zone in self.zones_by_id.items()
        }
        requested = {
            tuple(int(value) for value in signature)
            for signature in selected_zone_signatures
        }
        missing = requested - set(signature_to_zone_id)
        if missing:
            raise ValueError(
                "V6 integer warm start references zones absent from the "
                f"restricted pool: count={len(missing)}"
            )
        selected_ids = {
            signature_to_zone_id[signature] for signature in requested
        }
        warm_flow = {
            tuple(map(str, key)): int(quantity)
            for key, quantity in group_bay_flow.items()
            if int(quantity) > 0
        }
        warm_imports = {
            tuple(map(str, key)): int(quantity)
            for key, quantity in import_reservation.items()
            if int(quantity) > 0
        }
        unknown_flow = set(warm_flow) - set(self.group_bay_flow)
        unknown_imports = set(warm_imports) - set(self.import_reservation)
        if unknown_flow or unknown_imports:
            raise ValueError(
                "V6 integer warm start references unknown flow variables: "
                f"export={len(unknown_flow)}, import={len(unknown_imports)}"
            )

        for zone_id, variable in self.zone_variables.items():
            variable.Start = 1.0 if zone_id in selected_ids else 0.0
        for key, variable in self.group_bay_flow.items():
            variable.Start = float(warm_flow.get(tuple(map(str, key)), 0))
        for key, variable in self.import_reservation.items():
            variable.Start = float(warm_imports.get(tuple(map(str, key)), 0))

        export_physical: set[str] = set()
        export_sizes: set[tuple[str, str]] = set()
        export_heights: set[tuple[str, str]] = set()
        used_voyage_areas: set[tuple[str, str]] = set()
        for (group_id, bay_key), quantity in warm_flow.items():
            if quantity <= 0:
                continue
            group = self.groups_by_id[group_id]
            for physical_bay_key in v6_footprint(
                self.problem, bay_key, group.size
            ):
                export_physical.add(physical_bay_key)
                export_sizes.add((physical_bay_key, str(group.size)))
                export_heights.add((physical_bay_key, str(group.height)))
            used_voyage_areas.add(
                (
                    str(group.voyage_id),
                    str(self.problem.bays[bay_key].area_no),
                )
            )

        import_physical: set[str] = set()
        import_sizes: set[tuple[str, str]] = set()
        for (_flow, size, bay_key), quantity in warm_imports.items():
            if quantity <= 0:
                continue
            for physical_bay_key in v6_footprint(
                self.problem, bay_key, size
            ):
                import_physical.add(physical_bay_key)
                import_sizes.add((physical_bay_key, size))

        for key, variable in self.export_size_state.items():
            variable.Start = 1.0 if key in export_sizes else 0.0
        for key, variable in self.export_height_state.items():
            variable.Start = 1.0 if key in export_heights else 0.0
        for bay_key, variable in self.export_use.items():
            variable.Start = 1.0 if bay_key in export_physical else 0.0
        for bay_key, variable in self.import_use.items():
            variable.Start = 1.0 if bay_key in import_physical else 0.0
        for key, variable in self.import_size_state.items():
            variable.Start = 1.0 if key in import_sizes else 0.0
        for key, variable in self.voyage_area_use.items():
            variable.Start = 1.0 if key in used_voyage_areas else 0.0
        self.model.update()
        return {
            "provided": True,
            "applied": True,
            "selected_zone_count": len(selected_ids),
            "positive_group_bay_flow_count": len(warm_flow),
            "positive_import_reservation_count": len(warm_imports),
            "auxiliary_states_completed": True,
        }

    def add_zone(self, raw_zone: RowAwareZone) -> RowAwareZone:
        signature = tuple(raw_zone.candidate_indices)
        if signature in self.signatures:
            raise ValueError(f"duplicate V6 master zone signature: {signature}")
        if raw_zone.group_id not in self.groups_by_id:
            raise ValueError(f"V6 master zone has unknown group: {raw_zone.group_id}")
        zone_id = len(self.zones_by_id)
        zone = replace(raw_zone, zone_id=zone_id)
        column_terms: list[tuple[float, object]] = []
        for family, key, coefficient in _zone_column_terms(
            self.problem, self.groups_by_id, zone
        ):
            row = self.rows.get(family, {}).get(key)
            if row is None:
                raise ValueError(
                    f"V6 priced zone references absent master row: {family}, {key}"
                )
            column_terms.append((coefficient, row))
        variable = self.model.addPricedVar(
            column_terms,
            lb=0.0,
            # Every nonempty zone appears in at least one
            # same_group_bay_nonoverlap row whose RHS is one, so this upper
            # bound is redundant.  Omitting it in the LP prevents an active
            # column at its explicit upper bound from retaining a negative
            # reduced cost; exact pricing can then consider the full universe
            # without an ever-growing family of no-good rows.
            ub=(1.0 if self.integral else self.model._gp.GRB.INFINITY),
            vtype="B" if self.integral else "C",
            obj=_zone_direct_objective(self.evaluator, zone, self.phase),
            name=f"x_zone_{zone_id}",
        )
        self.zones_by_id[zone_id] = zone
        self.zone_variables[zone_id] = variable
        self.signatures.add(signature)
        self.model.update()
        return zone

    def solve(self) -> V6MasterSolution:
        if self.integral:
            raise ValueError("use solve_integer() for a V6 integer master")
        started = perf_counter()
        self.model.optimize()
        status = self.model.getStatusName()
        if status != "optimal":
            raise RuntimeError(f"V6 restricted master is not optimal: {status}")
        duals = {
            family: {
                key: self.model.getLinearDual(row)
                for key, row in family_rows.items()
            }
            for family, family_rows in self.rows.items()
        }
        return V6MasterSolution(
            phase=self.phase,
            objective=self.model.getObjectiveValue(),
            total_artificial_deficit=sum(
                self.model.getValue(variable)
                for variable in self.artificial_deficit.values()
            ),
            zone_values={
                zone_id: self.model.getValue(variable)
                for zone_id, variable in self.zone_variables.items()
                if self.model.getValue(variable) > 1e-10
            },
            group_bay_flow={
                key: self.model.getValue(variable)
                for key, variable in self.group_bay_flow.items()
                if self.model.getValue(variable) > 1e-10
            },
            import_reservation={
                key: self.model.getValue(variable)
                for key, variable in self.import_reservation.items()
                if self.model.getValue(variable) > 1e-10
            },
            duals=duals,
            runtime_seconds=perf_counter() - started,
        )

    def solve_integer(self) -> V6IntegerMasterSolution:
        if not self.integral:
            raise ValueError("solve_integer() requires an integral V6 master")
        started = perf_counter()
        recorder = MipProgressRecorder(phase="v6_restricted_integer_master")
        self.model.optimize(callback=recorder)
        progress = recorder.finalize(self.model)
        status = self.model.getStatusName()
        if self.model.getSolutionCount() <= 0:
            raise RuntimeError(
                "V6 restricted integer master has no feasible incumbent: "
                f"status={status}, column_count={len(self.zones_by_id)}"
            )
        selected = tuple(
            sorted(
                zone_id
                for zone_id, variable in self.zone_variables.items()
                if self.model.getValue(variable) > 0.5
            )
        )
        return V6IntegerMasterSolution(
            phase=self.phase,
            status=status,
            objective=self.model.getObjectiveValue(),
            best_bound=self.model.getBestBound(),
            mip_gap=self.model.getMipGap(),
            proven_optimal=status == "optimal",
            selected_zone_ids=selected,
            group_bay_flow={
                key: int(round(self.model.getValue(variable)))
                for key, variable in self.group_bay_flow.items()
                if self.model.getValue(variable) > 0.5
            },
            import_reservation={
                key: int(round(self.model.getValue(variable)))
                for key, variable in self.import_reservation.items()
                if self.model.getValue(variable) > 0.5
            },
            runtime_seconds=perf_counter() - started,
            progress=progress,
        )

    def reduced_cost(self, zone: RowAwareZone, duals: DualSnapshot) -> float:
        return _reduced_cost(
            self.problem,
            self.groups_by_id,
            self.evaluator,
            zone,
            duals,
            self.phase,
        )

    def dispose(self) -> None:
        self.model.dispose()


class V6ExactZonePricing:
    """Exact k-best dynamic-programming pricing for V6 row-aware zones."""

    def __init__(
        self,
        problem: ProblemData,
        atoms: Sequence[RowAwareBayAtom],
        anchor_capacity_limits: Mapping[tuple[str, str], int],
        evaluator: V6ModelEvaluator,
        config: V6RootCgConfig,
    ) -> None:
        self.problem = problem
        self.atoms = tuple(atoms)
        self.atoms_by_index = {atom.candidate_index: atom for atom in atoms}
        self.anchor_capacity_limits = dict(anchor_capacity_limits)
        self.evaluator = evaluator
        self.config = config
        self.groups_by_id = evaluator.groups_by_id
        self.runs_by_group = self._build_runs()
        self.atom_indices_by_group: dict[str, frozenset[int]] = {
            group_id: frozenset(
                atom.candidate_index
                for run in runs
                for _bay_key, bay_atoms in run
                for atom in bay_atoms
            )
            for group_id, runs in self.runs_by_group.items()
        }

    def _build_runs(
        self,
    ) -> dict[
        str,
        tuple[tuple[tuple[str, tuple[RowAwareBayAtom, ...]], ...], ...],
    ]:
        by_group_area_bay: defaultdict[
            tuple[str, str], defaultdict[str, list[RowAwareBayAtom]]
        ] = defaultdict(lambda: defaultdict(list))
        for atom in self.atoms:
            by_group_area_bay[(atom.group_id, atom.area_no)][
                atom.anchor_bay_key
            ].append(atom)
        output: defaultdict[
            str,
            list[tuple[tuple[str, tuple[RowAwareBayAtom, ...]], ...]],
        ] = defaultdict(list)
        for (group_id, _area_no), atoms_by_bay in sorted(
            by_group_area_bay.items()
        ):
            ordered = sorted(
                atoms_by_bay,
                key=lambda bay_key: (
                    atoms_by_bay[bay_key][0].footprint_orders,
                    bay_key,
                ),
            )
            runs: list[list[str]] = []
            for bay_key in ordered:
                if not runs:
                    runs.append([bay_key])
                    continue
                previous = atoms_by_bay[runs[-1][-1]][0].footprint_orders
                current = atoms_by_bay[bay_key][0].footprint_orders
                if current[0] - previous[-1] == 2:
                    runs[-1].append(bay_key)
                else:
                    runs.append([bay_key])
            for run in runs:
                output[group_id].append(
                    tuple(
                        (
                            bay_key,
                            tuple(
                                sorted(
                                    atoms_by_bay[bay_key],
                                    key=lambda candidate: (
                                        candidate.candidate_index
                                    ),
                                )
                            ),
                        )
                        for bay_key in run
                    )
                )
        return {key: tuple(value) for key, value in output.items()}

    @staticmethod
    def _dual(duals: DualSnapshot, family: str, key: RowKey) -> float:
        return float(duals.get(family, {}).get(key, 0.0))

    def _atom_objective(
        self,
        atom: RowAwareBayAtom,
        duals: DualSnapshot,
        phase: str,
    ) -> float:
        group = self.groups_by_id[atom.group_id]
        pair = (atom.group_id, atom.anchor_bay_key)
        direct = (
            self.evaluator.unused_capacity_unit_objective_coefficient()
            * atom.capacity
            if phase == "business"
            else 0.0
        )
        value = direct
        value += atom.capacity * self._dual(
            duals, "group_bay_flow_upper", pair
        )
        value -= atom.capacity * self._dual(
            duals,
            "reserved_anchor_size_capacity",
            (atom.anchor_bay_key, str(group.size)),
        )
        for physical_bay_key in v6_footprint(
            self.problem, atom.anchor_bay_key, group.size
        ):
            value -= atom.capacity * self._dual(
                duals, "reserved_physical_capacity", physical_bay_key
            )
        for resource in atom.resources:
            value -= self._dual(duals, "physical_row_exclusive", resource)
        return float(value)

    def _bay_objective(
        self,
        group_id: str,
        bay_key: str,
        duals: DualSnapshot,
    ) -> float:
        pair = (group_id, bay_key)
        return -(
            self._dual(duals, "group_bay_flow_lower", pair)
            + self._dual(duals, "same_group_bay_nonoverlap", pair)
        )

    @staticmethod
    def _retain_k_labels(
        labels: Iterable[tuple[float, tuple[int, ...]]],
        limit: int,
    ) -> list[tuple[float, tuple[int, ...]]]:
        """Return deterministic exact k-best distinct cost/signature labels."""

        best_by_signature: dict[tuple[int, ...], float] = {}
        for cost, signature in labels:
            previous = best_by_signature.get(signature)
            if previous is None or float(cost) < previous:
                best_by_signature[signature] = float(cost)
        return sorted(
            (
                (cost, signature)
                for signature, cost in best_by_signature.items()
            ),
            key=lambda item: (item[0], item[1]),
        )[: max(1, int(limit))]

    def _k_best_bay_patterns(
        self,
        group_id: str,
        bay_key: str,
        atoms: Sequence[RowAwareBayAtom],
        duals: DualSnapshot,
        phase: str,
        limit: int,
    ) -> list[tuple[float, tuple[int, ...]]]:
        """Solve the exact k-best nonempty row-subset knapsack for one bay."""

        resources: list[tuple[str, str]] = [
            resource for atom in atoms for resource in atom.resources
        ]
        if len(resources) != len(set(resources)):
            raise RuntimeError(
                "V6 bay-pattern DP requires distinct row resources within "
                f"one anchor bay: group={group_id}, bay={bay_key}"
            )
        capacity_limit = int(
            self.anchor_capacity_limits[(str(group_id), str(bay_key))]
        )
        states: dict[int, list[tuple[float, tuple[int, ...]]]] = {
            0: [(0.0, ())]
        }
        for atom in sorted(atoms, key=lambda item: item.candidate_index):
            candidates: defaultdict[
                int, list[tuple[float, tuple[int, ...]]]
            ] = defaultdict(list)
            atom_cost = self._atom_objective(atom, duals, phase)
            for capacity, labels in states.items():
                candidates[capacity].extend(labels)
                next_capacity = capacity + int(atom.capacity)
                if next_capacity > capacity_limit:
                    continue
                candidates[next_capacity].extend(
                    (
                        cost + atom_cost,
                        signature + (int(atom.candidate_index),),
                    )
                    for cost, signature in labels
                )
            states = {
                capacity: self._retain_k_labels(labels, limit)
                for capacity, labels in candidates.items()
            }
        bay_cost = self._bay_objective(group_id, bay_key, duals)
        return self._retain_k_labels(
            (
                (cost + bay_cost, signature)
                for capacity, labels in states.items()
                if capacity > 0
                for cost, signature in labels
            ),
            limit,
        )

    def _price_group_by_k_best_dp(
        self,
        group_id: str,
        duals: DualSnapshot,
        *,
        phase: str,
        excluded_signatures: set[tuple[int, ...]],
        maximum_columns: int,
        patterns_per_interval: int,
        deadline: float | None,
    ) -> tuple[V6PricingResult, ...]:
        """Exact pricing with a diversified negative-column batch.

        The best pattern of every interval is exact, so an empty negative
        batch is an exact pricing certificate.  Additional k-best patterns
        accelerate degenerate tails but are not claimed to be a globally
        ordered top-k list.
        """

        started = perf_counter()
        group_indices = self.atom_indices_by_group.get(
            str(group_id), frozenset()
        )
        relevant_excluded = {
            signature
            for signature in excluded_signatures
            if signature and set(signature) <= group_indices
        }
        # At most len(excluded) better labels can be skipped before the first
        # unseen one.  Retaining k+excluded labels locally is therefore enough
        # for the exact global k-best unseen set.
        # With no explicit variable upper bound, an active interval-best
        # column has nonnegative reduced cost in the RMP.  Consequently one
        # exact best row pattern per interval is sufficient to detect every
        # negative column.  Extra local labels are needed only to step past
        # explicitly excluded signatures in standalone/oracle calls.
        local_search_limit = max(
            1,
            int(patterns_per_interval),
            1 + len(relevant_excluded),
        )
        fixed_cost = (
            self.evaluator.zone_fixed_activation_objective_coefficient()
            if phase == "business"
            else 0.0
        )
        group_candidates: list[tuple[float, tuple[int, ...]]] = []
        for run_bays in self.runs_by_group.get(str(group_id), ()):
            if deadline is not None and perf_counter() >= deadline:
                raise RuntimeError(
                    "V6 exact root CG reached its total time limit"
                )
            bay_keys = tuple(bay_key for bay_key, _atoms in run_bays)
            patterns_by_bay = {
                bay_key: self._k_best_bay_patterns(
                    group_id,
                    bay_key,
                    bay_atoms,
                    duals,
                    phase,
                    local_search_limit,
                )
                for bay_key, bay_atoms in run_bays
            }
            for start in range(len(bay_keys)):
                partial: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
                for end in range(start, len(bay_keys)):
                    combined = (
                        (
                            left_cost + right_cost,
                            tuple(sorted(left_signature + right_signature)),
                        )
                        for left_cost, left_signature in partial
                        for right_cost, right_signature in patterns_by_bay[
                            bay_keys[end]
                        ]
                    )
                    partial = self._retain_k_labels(
                        combined, local_search_limit
                    )
                    group_candidates.extend(
                        (cost + fixed_cost, signature)
                        for cost, signature in partial
                    )
        ranked = self._retain_k_labels(
            group_candidates,
            max(1, len(group_candidates)),
        )
        unseen_ranked = [
            (cost, signature)
            for cost, signature in ranked
            if signature not in relevant_excluded
        ]
        if not unseen_ranked:
            return ()
        tolerance = float(self.config.reduced_cost_tolerance)
        if unseen_ranked[0][0] >= -tolerance:
            selected_ranked = unseen_ranked[:1]
        else:
            negative = [
                item for item in unseen_ranked if item[0] < -tolerance
            ]
            selected_ranked: list[tuple[float, tuple[int, ...]]] = []
            used_intervals: set[tuple[str, tuple[str, ...]]] = set()
            for item in negative:
                signature = item[1]
                atoms = [self.atoms_by_index[index] for index in signature]
                interval_key = (
                    atoms[0].area_no,
                    tuple(
                        sorted(
                            {atom.anchor_bay_key for atom in atoms},
                            key=lambda bay_key: (
                                next(
                                    atom.footprint_orders
                                    for atom in atoms
                                    if atom.anchor_bay_key == bay_key
                                ),
                                bay_key,
                            ),
                        )
                    ),
                )
                if interval_key in used_intervals:
                    continue
                used_intervals.add(interval_key)
                selected_ranked.append(item)
                if len(selected_ranked) >= int(maximum_columns):
                    break
            if len(selected_ranked) < int(maximum_columns):
                selected_signatures = {item[1] for item in selected_ranked}
                selected_ranked.extend(
                    item
                    for item in negative
                    if item[1] not in selected_signatures
                )
                selected_ranked = selected_ranked[: int(maximum_columns)]
        results: list[V6PricingResult] = []
        for cost, signature in selected_ranked:
            zone = make_row_aware_zone(
                self.atoms_by_index[index] for index in signature
            )
            reduced_cost = _reduced_cost(
                self.problem,
                self.groups_by_id,
                self.evaluator,
                zone,
                duals,
                phase,
            )
            if not math.isclose(
                reduced_cost,
                cost,
                rel_tol=1e-8,
                abs_tol=1e-8,
            ):
                raise RuntimeError(
                    "V6 DP pricing objective differs from reconstructed "
                    f"reduced cost: dp={cost}, reconstructed={reduced_cost}"
                )
            results.append(
                V6PricingResult(
                    group_id=str(group_id),
                    area_no=zone.area_no,
                    reduced_cost=reduced_cost,
                    zone=zone,
                    solver_objective=float(cost),
                    runtime_seconds=0.0,
                )
            )
        if results:
            results[0] = replace(
                results[0],
                runtime_seconds=perf_counter() - started,
            )
        return tuple(results)

    def price_group(
        self,
        group_id: str,
        duals: DualSnapshot,
        *,
        phase: str,
        excluded_signatures: Iterable[tuple[int, ...]] = (),
        deadline: float | None = None,
    ) -> V6PricingResult | None:
        columns = self.price_group_columns(
            group_id,
            duals,
            phase=phase,
            excluded_signatures=excluded_signatures,
            maximum_columns=1,
            deadline=deadline,
        )
        return columns[0] if columns else None

    def price_group_columns(
        self,
        group_id: str,
        duals: DualSnapshot,
        *,
        phase: str,
        excluded_signatures: Iterable[tuple[int, ...]] = (),
        maximum_columns: int,
        patterns_per_interval: int = 1,
        deadline: float | None = None,
    ) -> tuple[V6PricingResult, ...]:
        """Return exact negative columns without constructing pricing MIPs."""

        group_id = str(group_id)
        if int(maximum_columns) <= 0:
            raise ValueError("V6 pricing maximum_columns must be positive")
        local_deadline = perf_counter() + float(self.config.pricing_time_limit)
        if deadline is not None:
            local_deadline = min(local_deadline, float(deadline))
            if local_deadline <= perf_counter():
                raise RuntimeError(
                    "V6 exact root CG reached its total time limit"
                )
        return self._price_group_by_k_best_dp(
            group_id,
            duals,
            phase=phase,
            excluded_signatures={
                tuple(int(value) for value in signature)
                for signature in excluded_signatures
            },
            maximum_columns=int(maximum_columns),
            patterns_per_interval=int(patterns_per_interval),
            deadline=local_deadline,
        )

    def dispose(self) -> None:
        """Pricing owns no external solver resources."""

    def exhaustive_price_group(
        self,
        group_id: str,
        zones: Sequence[RowAwareZone],
        duals: DualSnapshot,
        *,
        phase: str,
        excluded_signatures: Iterable[tuple[int, ...]] = (),
    ) -> V6PricingResult | None:
        """Small-case correctness oracle that explicitly scans every zone."""

        excluded = {tuple(signature) for signature in excluded_signatures}
        candidates = [
            zone
            for zone in zones
            if zone.group_id == str(group_id)
            and tuple(zone.candidate_indices) not in excluded
        ]
        if not candidates:
            return None
        started = perf_counter()
        zone = min(
            candidates,
            key=lambda item: (
                _reduced_cost(
                    self.problem,
                    self.groups_by_id,
                    self.evaluator,
                    item,
                    duals,
                    phase,
                ),
                item.area_no,
                item.candidate_indices,
            ),
        )
        reduced_cost = _reduced_cost(
            self.problem,
            self.groups_by_id,
            self.evaluator,
            zone,
            duals,
            phase,
        )
        return V6PricingResult(
            group_id=str(group_id),
            area_no=zone.area_no,
            reduced_cost=reduced_cost,
            zone=replace(zone, zone_id=-1),
            solver_objective=reduced_cost,
            runtime_seconds=perf_counter() - started,
        )


class V6RootColumnGeneration:
    """Close the V6 business root LP after a temporary feasibility phase."""

    def __init__(
        self,
        problem: ProblemData,
        peak_policy: V6PeakUtilizationPolicy,
        config: V6RootCgConfig | None = None,
    ) -> None:
        self.problem = problem
        self.peak_policy = peak_policy
        self.config = config or V6RootCgConfig()
        self.config.validate()
        self.atoms, self.anchor_capacity_limits = build_v6_row_aware_bay_atoms(
            problem
        )
        candidate_bays = _candidate_bays_by_group(self.atoms)
        self.evaluator = V6ModelEvaluator(
            problem,
            (),
            self.config.objective,
            candidate_bays_by_group=candidate_bays,
        )
        self.pricing = V6ExactZonePricing(
            problem,
            self.atoms,
            self.anchor_capacity_limits,
            self.evaluator,
            self.config,
        )

    def _run_phase(
        self,
        phase: str,
        initial_zones: Sequence[RowAwareZone],
        maximum_iterations: int,
        deadline: float | None,
    ) -> tuple[V6MasterSolution, tuple[RowAwareZone, ...], list[dict[str, object]]]:
        master = V6ProjectedRestrictedMaster(
            self.problem,
            self.atoms,
            self.anchor_capacity_limits,
            self.evaluator,
            self.peak_policy,
            phase=phase,
            initial_zones=initial_zones,
            solver_threads=self.config.solver_threads,
            solver_seed=self.config.solver_seed,
            verbose=self.config.verbose,
        )
        trace: list[dict[str, object]] = []
        previous_objective: float | None = None
        stagnant_rounds = 0
        try:
            try:
                for iteration in range(1, int(maximum_iterations) + 1):
                    if deadline is not None and perf_counter() >= deadline:
                        raise RuntimeError(
                            "V6 exact root CG reached its total time limit"
                        )
                    solution = master.solve()
                    priced: list[V6PricingResult] = []
                    pricing_wall_seconds = 0.0
                    batch_size = (
                        int(self.config.phase_one_columns_per_group)
                        if phase == "phase_one"
                        else int(self.config.business_columns_per_group)
                    )
                    if previous_objective is not None and math.isclose(
                        solution.objective,
                        previous_objective,
                        rel_tol=1e-10,
                        abs_tol=1e-10,
                    ):
                        stagnant_rounds += 1
                    else:
                        stagnant_rounds = 0
                    previous_objective = solution.objective
                    pattern_depth = min(
                        int(self.config.maximum_patterns_per_interval),
                        2 ** min(stagnant_rounds, 3),
                    )
                    for group in self.evaluator.groups:
                        pricing_started = perf_counter()
                        candidates = self.pricing.price_group_columns(
                            group.group_id,
                            solution.duals,
                            phase=phase,
                            excluded_signatures=(),
                            maximum_columns=batch_size,
                            patterns_per_interval=pattern_depth,
                            deadline=deadline,
                        )
                        pricing_wall_seconds += perf_counter() - pricing_started
                        priced.extend(candidates)
                    negative = [
                        candidate
                        for candidate in priced
                        if candidate.reduced_cost
                        < -float(self.config.reduced_cost_tolerance)
                        and tuple(candidate.zone.candidate_indices)
                        not in master.signatures
                    ]
                    trace.append(
                        {
                            "iteration": iteration,
                            "objective": solution.objective,
                            "artificial_deficit": solution.total_artificial_deficit,
                            "column_count": len(master.zones_by_id),
                            "minimum_reduced_cost": (
                                min(item.reduced_cost for item in priced)
                                if priced
                                else None
                            ),
                            "added_column_count": len(negative),
                            "columns_per_group_batch": batch_size,
                            "patterns_per_interval": pattern_depth,
                            "stagnant_round_count": stagnant_rounds,
                            "master_runtime_seconds": solution.runtime_seconds,
                            "pricing_runtime_seconds": pricing_wall_seconds,
                            "pricing_dp_runtime_seconds": sum(
                                item.runtime_seconds for item in priced
                            ),
                        }
                    )
                    if not negative:
                        if (
                            phase == "phase_one"
                            and solution.total_artificial_deficit
                            > float(self.config.feasibility_tolerance)
                        ):
                            raise RuntimeError(
                                "V6 Phase-I closed with positive artificial deficit; "
                                "the full projected model is infeasible under the peak cap"
                            )
                        return (
                            solution,
                            tuple(master.zones_by_id.values()),
                            trace,
                        )
                    for candidate in negative:
                        master.add_zone(candidate.zone)
                raise RuntimeError(
                    f"V6 {phase} column generation reached its iteration limit"
                )
            except RuntimeError as error:
                message = str(error)
                incomplete = (
                    "total time limit" in message
                    or "status=timelimit" in message
                    or "iteration limit" in message
                )
                if incomplete:
                    raise V6RootCgIncompleteError(
                        message,
                        {
                            "algorithm": "v6_exact_root_column_generation",
                            "model_schema_version": V6_MODEL_SCHEMA_VERSION,
                            "phase": phase,
                            "closed_by_exact_pricing": False,
                            "completed_iterations": len(trace),
                            "generated_zone_count": len(master.zones_by_id),
                            "trace": list(trace),
                        },
                    ) from error
                raise
        finally:
            master.dispose()

    def solve(
        self,
        initial_zones: Sequence[RowAwareZone] = (),
    ) -> V6RootCgResult:
        started = perf_counter()
        deadline = (
            started + float(self.config.total_time_limit)
            if self.config.total_time_limit is not None
            else None
        )
        seed_zones = tuple(initial_zones)
        try:
            if seed_zones:
                phase_one = None
                feasible_zones = seed_zones
                phase_one_trace: list[dict[str, object]] = []
            else:
                phase_one, feasible_zones, phase_one_trace = self._run_phase(
                    "phase_one",
                    (),
                    self.config.maximum_phase_one_iterations,
                    deadline,
                )
            business, final_zones, business_trace = self._run_phase(
                "business",
                feasible_zones,
                self.config.maximum_business_iterations,
                deadline,
            )
        finally:
            self.pricing.dispose()
        diagnostics = {
            "algorithm": "v6_exact_root_column_generation",
            "model_schema_version": V6_MODEL_SCHEMA_VERSION,
            "result_role": "root_lp_lower_bound_not_integer_incumbent",
            "root_rewrite": {
                "feasibility_start": (
                    "provided_feasible_seed"
                    if seed_zones
                    else "artificial_variable_phase_one"
                ),
                "pricing_method": (
                    "exact_k_best_bay_knapsack_and_contiguous_interval_dp"
                ),
                "pricing_mip_count": 0,
                "lp_zone_upper_bound": (
                    "implicit_via_same_group_bay_nonoverlap"
                ),
                "columns_per_group": {
                    "phase_one": int(
                        self.config.phase_one_columns_per_group
                    ),
                    "business": int(
                        self.config.business_columns_per_group
                    ),
                },
                "maximum_patterns_per_interval": int(
                    self.config.maximum_patterns_per_interval
                ),
            },
            "peak_policy": self.peak_policy.as_dict(),
            "atom_count": len(self.atoms),
            "phase_one": {
                "iterations": len(phase_one_trace),
                "skipped_by_feasible_seed": bool(seed_zones),
                "seed_zone_count": len(seed_zones),
                "final_objective": (
                    0.0 if phase_one is None else phase_one.objective
                ),
                "final_artificial_deficit": (
                    0.0
                    if phase_one is None
                    else phase_one.total_artificial_deficit
                ),
                "trace": phase_one_trace,
            },
            "business": {
                "iterations": len(business_trace),
                "final_objective": business.objective,
                "closed_by_exact_pricing": True,
                "master_runtime_seconds": sum(
                    float(item["master_runtime_seconds"])
                    for item in business_trace
                ),
                "pricing_runtime_seconds": sum(
                    float(item["pricing_runtime_seconds"])
                    for item in business_trace
                ),
                "trace": business_trace,
            },
            "generated_zone_count": len(final_zones),
            "total_seconds": perf_counter() - started,
        }
        return V6RootCgResult(
            objective=business.objective,
            zones=final_zones,
            zone_values=business.zone_values,
            group_bay_flow=business.group_bay_flow,
            import_reservation=business.import_reservation,
            diagnostics=diagnostics,
        )


__all__ = [
    "V6ExactZonePricing",
    "V6IntegerMasterSolution",
    "V6MasterSolution",
    "V6PricingResult",
    "V6ProjectedRestrictedMaster",
    "V6RootCgConfig",
    "V6RootCgIncompleteError",
    "V6RootCgResult",
    "V6RootColumnGeneration",
]
