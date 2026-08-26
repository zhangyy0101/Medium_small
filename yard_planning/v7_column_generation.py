"""Global V7 bay-pattern RMP and exact active-set column generation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Iterable, Mapping, Sequence

from .gurobi_backend import GurobiModel, MipProgressRecorder
from .models import ProblemData
from .row_aware_zones import v6_footprint
from .v7_atoms import V7RowAtom, atoms_by_group_bay, build_v7_row_atoms
from .v7_bay_patterns import (
    V7BayPattern,
    V7ExactBayPricing,
    build_v7_pattern_from_atom_indices,
    enumerate_v7_bay_patterns,
)
from .v7_model import (
    V7_MODEL_SCHEMA_VERSION,
    V7ModelEvaluator,
    V7ObjectiveConfig,
    V7PeakUtilizationPolicy,
)


class V7RootCgIncompleteError(RuntimeError):
    def __init__(self, message: str, diagnostics: Mapping[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class V7RootCgConfig:
    root_time_limit: float = 30.0
    maximum_iterations: int = 100
    reduced_cost_tolerance: float = 1e-8
    columns_per_bay_per_round: int = 3
    solver_threads: int = 1
    solver_seed: int = 0
    verbose: bool = False
    objective: V7ObjectiveConfig = field(default_factory=V7ObjectiveConfig)

    def validate(self) -> None:
        if not math.isfinite(float(self.root_time_limit)) or float(self.root_time_limit) <= 0:
            raise ValueError("V7 root time limit must be positive")
        if int(self.maximum_iterations) <= 0:
            raise ValueError("V7 root maximum iterations must be positive")
        if float(self.reduced_cost_tolerance) <= 0:
            raise ValueError("V7 reduced-cost tolerance must be positive")
        if int(self.columns_per_bay_per_round) <= 0:
            raise ValueError("V7 columns per bay must be positive")
        if int(self.solver_threads) < 0:
            raise ValueError("V7 solver_threads must be nonnegative")
        self.objective.validate()


@dataclass(frozen=True)
class V7MasterSolution:
    objective: float
    pattern_values: Mapping[int, float]
    group_bay_flow: Mapping[tuple[str, str], float]
    import_reservation: Mapping[tuple[str, str, str], float]
    duals: Mapping[str, Mapping[object, float]]
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class V7RootCgResult:
    objective: float
    patterns: tuple[V7BayPattern, ...]
    solution: V7MasterSolution
    active_areas_by_group: Mapping[str, frozenset[str]]
    diagnostics: Mapping[str, object]


def patterns_from_atom_solution(
    problem: ProblemData,
    atoms: Sequence[V7RowAtom],
    selected_atom_indices: Iterable[int],
    *,
    start_pattern_id: int = 0,
) -> tuple[V7BayPattern, ...]:
    atoms_by_index = {atom.candidate_index: atom for atom in atoms}
    by_anchor: defaultdict[str, list[int]] = defaultdict(list)
    for index in selected_atom_indices:
        atom = atoms_by_index[int(index)]
        by_anchor[atom.anchor_bay_key].append(atom.candidate_index)
    patterns = []
    for _anchor, indices in sorted(by_anchor.items()):
        patterns.append(
            build_v7_pattern_from_atom_indices(
                problem,
                atoms_by_index,
                indices,
                pattern_id=start_pattern_id + len(patterns),
            )
        )
    return tuple(patterns)


def normalize_v7_pattern_ids(
    patterns: Iterable[V7BayPattern],
) -> tuple[V7BayPattern, ...]:
    unique: dict[tuple[int, ...], V7BayPattern] = {}
    for pattern in patterns:
        unique.setdefault(pattern.signature, pattern)
    return tuple(
        replace(pattern, pattern_id=index)
        for index, pattern in enumerate(
            sorted(unique.values(), key=lambda item: (item.anchor_bay_key, item.signature))
        )
    )


class V7GlobalPatternMaster:
    """Build and solve one global LP/MIP over a supplied Bay Pattern pool."""

    def __init__(
        self,
        problem: ProblemData,
        atoms: Sequence[V7RowAtom],
        patterns: Sequence[V7BayPattern],
        peak_policy: V7PeakUtilizationPolicy,
        objective: V7ObjectiveConfig | None = None,
        *,
        integral: bool = False,
        time_limit: float = 30.0,
        mip_gap: float = 0.0,
        solver_threads: int = 1,
        solver_seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.problem = problem
        self.atoms = tuple(atoms)
        self.patterns = normalize_v7_pattern_ids(patterns)
        self.patterns_by_id = {pattern.pattern_id: pattern for pattern in self.patterns}
        self.peak_policy = peak_policy
        self.objective = objective or V7ObjectiveConfig()
        self.objective.validate()
        self.integral = bool(integral)
        self.time_limit = float(time_limit)
        self.mip_gap = float(mip_gap)
        self.solver_threads = int(solver_threads)
        self.solver_seed = int(solver_seed)
        self.verbose = bool(verbose)
        self.evaluator = V7ModelEvaluator(problem, self.atoms, self.objective)
        self.groups = self.evaluator.groups
        self.groups_by_id = self.evaluator.groups_by_id
        self.atoms_by_pair = atoms_by_group_bay(self.atoms)
        self._validate_patterns()
        self.model: GurobiModel | None = None
        self.pattern_variables: dict[int, object] = {}
        self.group_bay_flow_variables: dict[tuple[str, str], object] = {}
        self.import_variables: dict[tuple[str, str, str], object] = {}
        self.constraints: defaultdict[str, dict[object, object]] = defaultdict(dict)

    def _validate_patterns(self) -> None:
        atoms_by_index = {atom.candidate_index: atom for atom in self.atoms}
        for pattern in self.patterns:
            self._validate_pattern(pattern, atoms_by_index)

    def _validate_pattern(
        self,
        pattern: V7BayPattern,
        atoms_by_index: Mapping[int, V7RowAtom] | None = None,
    ) -> None:
        source = atoms_by_index or {
            atom.candidate_index: atom for atom in self.atoms
        }
        rebuilt = build_v7_pattern_from_atom_indices(
            self.problem,
            source,
            pattern.candidate_indices,
            pattern_id=pattern.pattern_id,
        )
        if replace(rebuilt, pattern_id=pattern.pattern_id) != replace(
            pattern, pattern_id=pattern.pattern_id
        ):
            raise ValueError(f"invalid V7 Bay Pattern: {pattern.pattern_id}")

    def _configure(self, model: GurobiModel) -> None:
        if not self.verbose:
            model.hideOutput()
        model.setMinimize()
        model.setParam("TimeLimit", self.time_limit)
        model.setParam("Seed", self.solver_seed)
        if self.solver_threads > 0:
            model.setParam("Threads", self.solver_threads)
        if self.integral:
            model.setParam("MIPGap", self.mip_gap)
        model.setParam("FeasibilityTol", 1e-9)

    def _patterns_for_pair(self, pair: tuple[str, str]) -> list[V7BayPattern]:
        return [
            pattern
            for pattern in self.patterns
            if pattern.anchor_bay_key == pair[1] and pair[0] in pattern.active_groups
        ]

    def _group_physical_keys(self) -> dict[tuple[str, str], set[str]]:
        anchors: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        for pair in self.atoms_by_pair:
            group = self.groups_by_id[pair[0]]
            for physical in v6_footprint(self.problem, pair[1], group.size):
                anchors[(pair[0], physical)].add(pair[1])
        return dict(anchors)

    def build(self) -> None:
        if self.model is not None:
            raise RuntimeError("V7 pattern master is already built")
        model = GurobiModel("v7_global_pattern_integer" if self.integral else "v7_global_pattern_lp")
        self._configure(model)
        self.model = model
        gp = model._gp
        quicksum = gp.quicksum
        binary = "B" if self.integral else "C"
        integer = "I" if self.integral else "C"

        self.pattern_variables = {
            pattern.pattern_id: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=binary,
                name=f"lambda_pattern_{pattern.pattern_id}",
            )
            for pattern in self.patterns
        }
        by_anchor: defaultdict[str, list[V7BayPattern]] = defaultdict(list)
        for pattern in self.patterns:
            by_anchor[pattern.anchor_bay_key].append(pattern)
        for anchor in sorted({atom.anchor_bay_key for atom in self.atoms}):
            self.constraints["bay_pattern_choice"][anchor] = model.addConstr(
                quicksum(
                    self.pattern_variables[pattern.pattern_id]
                    for pattern in by_anchor.get(anchor, [])
                )
                <= 1,
                name=f"bay_pattern_choice_{anchor}",
            )

        group_bay_use = {
            pair: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=binary,
                obj=self.evaluator.group_bay_use_objective_coefficient(),
                name=f"u_group_bay_{index}",
            )
            for index, pair in enumerate(sorted(self.atoms_by_pair))
        }
        self.group_bay_flow_variables = {
            pair: model.addVar(
                lb=0.0,
                ub=float(self.groups_by_id[pair[0]].demand),
                vtype=integer,
                obj=self.evaluator.group_bay_flow_objective_coefficient(*pair),
                name=f"q_group_bay_{index}",
            )
            for index, pair in enumerate(sorted(self.atoms_by_pair))
        }
        for pair in sorted(self.atoms_by_pair):
            pair_patterns = self._patterns_for_pair(pair)
            self.constraints["q_pattern_capacity"][pair] = model.addConstr(
                self.group_bay_flow_variables[pair]
                - quicksum(
                    pattern.capacity_for(pair[0])
                    * self.pattern_variables[pattern.pattern_id]
                    for pattern in pair_patterns
                )
                <= 0,
                name=f"q_pattern_capacity_{pair[0]}_{pair[1]}",
            )
            self.constraints["u_pattern_identity"][pair] = model.addConstr(
                group_bay_use[pair]
                - quicksum(
                    self.pattern_variables[pattern.pattern_id]
                    for pattern in pair_patterns
                )
                == 0,
                name=f"u_pattern_identity_{pair[0]}_{pair[1]}",
            )
            self.constraints["positive_group_bay_flow"][pair] = model.addConstr(
                self.group_bay_flow_variables[pair] >= group_bay_use[pair]
            )
            self.constraints["group_bay_flow_use_upper"][pair] = model.addConstr(
                self.group_bay_flow_variables[pair]
                <= int(self.groups_by_id[pair[0]].demand) * group_bay_use[pair]
            )
        for group in self.groups:
            self.constraints["exact_export_demand"][group.group_id] = model.addConstr(
                quicksum(
                    variable
                    for (group_id, _bay), variable in self.group_bay_flow_variables.items()
                    if group_id == group.group_id
                )
                == int(group.demand)
            )

        resources = sorted(
            {resource for atom in self.atoms for resource in atom.physical_resources}
        )
        for resource in resources:
            self.constraints["physical_row_exclusivity"][resource] = model.addConstr(
                quicksum(
                    self.pattern_variables[pattern.pattern_id]
                    for pattern in self.patterns
                    if resource in pattern.physical_resources
                )
                <= 1
            )
        physical_bays = sorted(
            {physical for atom in self.atoms for physical in atom.physical_bays}
        )
        for physical in physical_bays:
            self.constraints["reserved_physical_capacity"][physical] = model.addConstr(
                quicksum(
                    sum(dict(pattern.group_capacities).values())
                    * self.pattern_variables[pattern.pattern_id]
                    for pattern in self.patterns
                    if physical in pattern.physical_bays
                )
                <= int(self.problem.bays[physical].physical_capacity)
            )
        anchor_size_keys = sorted(
            {(atom.anchor_bay_key, atom.size) for atom in self.atoms}
        )
        for key in anchor_size_keys:
            anchor, size = key
            self.constraints["reserved_anchor_size_capacity"][key] = model.addConstr(
                quicksum(
                    sum(dict(pattern.group_capacities).values())
                    * self.pattern_variables[pattern.pattern_id]
                    for pattern in self.patterns
                    if pattern.anchor_bay_key == anchor and pattern.size_mode == size
                )
                <= int(self.problem.bays[anchor].cap_by_size.get(size, 0))
            )

        group_area_pairs = sorted(
            (group.group_id, area)
            for group in self.groups
            for area in self.evaluator.candidate_areas[group.group_id]
        )
        group_area_use = {
            key: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=binary,
                obj=self.evaluator.group_area_use_objective_coefficient(),
                name=f"y_group_area_{index}",
            )
            for index, key in enumerate(group_area_pairs)
        }
        span_lower = {
            key: model.addVar(lb=0.0, ub=1.0, name=f"span_lower_{index}")
            for index, key in enumerate(group_area_pairs)
        }
        span_upper = {
            key: model.addVar(lb=0.0, ub=1.0, name=f"span_upper_{index}")
            for index, key in enumerate(group_area_pairs)
        }
        span = {
            key: model.addVar(
                lb=0.0,
                ub=1.0,
                obj=self.evaluator.group_area_span_objective_coefficient(),
                name=f"span_{index}",
            )
            for index, key in enumerate(group_area_pairs)
        }
        for key in group_area_pairs:
            group_id, area = key
            pairs = [
                pair
                for pair in group_bay_use
                if pair[0] == group_id
                and str(self.problem.bays[pair[1]].area_no) == area
            ]
            for pair in pairs:
                use = group_bay_use[pair]
                order = self.evaluator.normalized_bay_order(*pair)
                self.constraints["group_bay_to_area"][(key, pair[1])] = model.addConstr(
                    use <= group_area_use[key]
                )
                self.constraints["span_upper_envelope"][(key, pair[1])] = model.addConstr(
                    span_upper[key] >= order * use
                )
                self.constraints["span_lower_envelope"][(key, pair[1])] = model.addConstr(
                    span_lower[key] <= order + 1 - use
                )
            self.constraints["group_area_presence"][key] = model.addConstr(
                group_area_use[key] <= quicksum(group_bay_use[pair] for pair in pairs)
            )
            self.constraints["span_lower_area_link"][key] = model.addConstr(
                span_lower[key] <= group_area_use[key]
            )
            self.constraints["span_upper_area_link"][key] = model.addConstr(
                span_upper[key] <= group_area_use[key]
            )
            self.constraints["span_definition"][key] = model.addConstr(
                span[key] >= span_upper[key] - span_lower[key]
            )

        group_physical_anchors = self._group_physical_keys()
        physical_group_use = {
            key: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=binary,
                name=f"u_physical_group_{index}",
            )
            for index, key in enumerate(sorted(group_physical_anchors))
        }
        for key, anchors in sorted(group_physical_anchors.items()):
            group_id, physical = key
            incident = [
                pattern
                for pattern in self.patterns
                if group_id in pattern.active_groups
                and physical in pattern.physical_bays
            ]
            bound = max(1, len(anchors))
            pattern_sum = quicksum(
                self.pattern_variables[pattern.pattern_id] for pattern in incident
            )
            self.constraints["physical_group_pattern_upper"][key] = model.addConstr(
                pattern_sum <= bound * physical_group_use[key]
            )
            self.constraints["physical_group_pattern_presence"][key] = model.addConstr(
                physical_group_use[key] <= pattern_sum
            )
        for physical in sorted({key[1] for key in physical_group_use}):
            self.constraints["max_three_groups_per_physical_bay"][physical] = model.addConstr(
                quicksum(
                    variable
                    for (group_id, bay_key), variable in physical_group_use.items()
                    if bay_key == physical
                )
                <= 3
            )

        size_states = {
            (physical, size): model.addVar(lb=0.0, ub=1.0, vtype=binary)
            for physical in sorted({key[1] for key in physical_group_use})
            for size in sorted(
                {
                    self.groups_by_id[group_id].size
                    for group_id, bay_key in physical_group_use
                    if bay_key == physical
                }
            )
        }
        height_states = {
            (physical, height): model.addVar(lb=0.0, ub=1.0, vtype=binary)
            for physical in sorted({key[1] for key in physical_group_use})
            for height in sorted(
                {
                    self.groups_by_id[group_id].height
                    for group_id, bay_key in physical_group_use
                    if bay_key == physical
                }
            )
        }
        for (group_id, physical), use in physical_group_use.items():
            group = self.groups_by_id[group_id]
            self.constraints["export_size_state_link"][(group_id, physical)] = model.addConstr(
                use <= size_states[(physical, group.size)]
            )
            self.constraints["export_height_state_link"][(group_id, physical)] = model.addConstr(
                use <= height_states[(physical, group.height)]
            )
        for physical in sorted({key[1] for key in physical_group_use}):
            self.constraints["export_size_state_choice"][physical] = model.addConstr(
                quicksum(
                    variable
                    for (bay_key, _size), variable in size_states.items()
                    if bay_key == physical
                )
                <= 1
            )
            self.constraints["export_height_state_choice"][physical] = model.addConstr(
                quicksum(
                    variable
                    for (bay_key, _height), variable in height_states.items()
                    if bay_key == physical
                )
                <= 1
            )

        import_rows = [
            (flow, size, bay_key, capacity)
            for (flow, size), candidates in sorted(self.evaluator.import_candidates.items())
            for bay_key, capacity in candidates
        ]
        self.import_variables = {
            (flow, size, bay_key): model.addVar(
                lb=0.0,
                ub=float(capacity),
                vtype=integer,
                name=f"p_import_{index}",
            )
            for index, (flow, size, bay_key, capacity) in enumerate(import_rows)
        }
        import_by_flow_size: defaultdict[tuple[str, str], list[object]] = defaultdict(list)
        import_by_anchor_size: defaultdict[tuple[str, str], list[object]] = defaultdict(list)
        import_by_physical: defaultdict[str, list[object]] = defaultdict(list)
        import_by_physical_size: defaultdict[tuple[str, str], list[object]] = defaultdict(list)
        for (flow, size, anchor), variable in self.import_variables.items():
            import_by_flow_size[(flow, size)].append(variable)
            import_by_anchor_size[(anchor, size)].append(variable)
            for physical in v6_footprint(self.problem, anchor, size):
                import_by_physical[physical].append(variable)
                import_by_physical_size[(physical, size)].append(variable)
        for key, demand in sorted(self.problem.import_demand_by_flow_size.items()):
            normalized = tuple(map(str, key))
            self.constraints["exact_import_demand"][normalized] = model.addConstr(
                quicksum(import_by_flow_size.get(normalized, [])) == int(demand)
            )
        for key, values in sorted(import_by_anchor_size.items()):
            self.constraints["import_anchor_size_capacity"][key] = model.addConstr(
                quicksum(values)
                <= int(self.problem.bays[key[0]].cap_by_size.get(key[1], 0))
            )

        allocation_physical = sorted(
            {key[1] for key in physical_group_use} | set(import_by_physical)
        )
        export_use = {
            physical: model.addVar(lb=0.0, ub=1.0, vtype=binary)
            for physical in allocation_physical
        }
        import_use = {
            physical: model.addVar(lb=0.0, ub=1.0, vtype=binary)
            for physical in allocation_physical
        }
        import_size_state = {
            key: model.addVar(lb=0.0, ub=1.0, vtype=binary)
            for key in sorted(import_by_physical_size)
        }
        for physical in allocation_physical:
            export_states = [
                variable
                for (group_id, bay_key), variable in physical_group_use.items()
                if bay_key == physical
            ]
            for index, state in enumerate(export_states):
                self.constraints["export_use_lower"][(physical, index)] = model.addConstr(
                    state <= export_use[physical]
                )
            if export_states:
                self.constraints["export_use_upper"][physical] = model.addConstr(
                    export_use[physical] <= quicksum(export_states)
                )
            else:
                self.constraints["export_use_zero"][physical] = model.addConstr(
                    export_use[physical] == 0
                )
            imports = import_by_physical.get(physical, [])
            if imports:
                capacity = max(1, int(self.problem.bays[physical].physical_capacity))
                self.constraints["import_physical_capacity"][physical] = model.addConstr(
                    quicksum(imports) <= capacity * import_use[physical]
                )
                self.constraints["import_use_presence"][physical] = model.addConstr(
                    import_use[physical] <= quicksum(imports)
                )
                for key, state in import_size_state.items():
                    if key[0] != physical:
                        continue
                    self.constraints["import_size_state_link"][key] = model.addConstr(
                        quicksum(import_by_physical_size[key]) <= capacity * state
                    )
                    self.constraints["import_size_state_presence"][key] = model.addConstr(
                        state <= quicksum(import_by_physical_size[key])
                    )
                self.constraints["import_size_state_choice"][physical] = model.addConstr(
                    quicksum(
                        state
                        for (bay_key, _size), state in import_size_state.items()
                        if bay_key == physical
                    )
                    <= import_use[physical]
                )
            else:
                self.constraints["import_use_zero"][physical] = model.addConstr(
                    import_use[physical] == 0
                )
            self.constraints["export_import_exclusivity"][physical] = model.addConstr(
                export_use[physical] + import_use[physical] <= 1
            )

        planned_load_by_area: defaultdict[str, list[tuple[int, object]]] = defaultdict(list)
        for (group_id, anchor), variable in self.group_bay_flow_variables.items():
            footprint = v6_footprint(self.problem, anchor, self.groups_by_id[group_id].size)
            planned_load_by_area[str(self.problem.bays[anchor].area_no)].append(
                (len(footprint), variable)
            )
        for (_flow, size, anchor), variable in self.import_variables.items():
            footprint = v6_footprint(self.problem, anchor, size)
            planned_load_by_area[str(self.problem.bays[anchor].area_no)].append(
                (len(footprint), variable)
            )
        area_capacity: Counter[str] = Counter()
        for bay in self.problem.bays.values():
            area_capacity[str(bay.area_no)] += int(bay.physical_capacity)
        for area, terms in sorted(planned_load_by_area.items()):
            self.constraints["peak_utilization_hard_cap"][area] = model.addConstr(
                quicksum(coefficient * variable for coefficient, variable in terms)
                <= float(self.peak_policy.epsilon_cap) * int(area_capacity[area])
            )
        model.addVar(
            lb=1.0,
            ub=1.0,
            obj=self.evaluator.objective_constant(),
            name="v7_objective_constant",
        )
        model.update()

    def solve(self) -> V7MasterSolution:
        if self.model is None:
            self.build()
        model = self.model
        assert model is not None
        started = perf_counter()
        recorder = MipProgressRecorder(phase="v7_restricted_integer_master") if self.integral else None
        model.optimize(callback=recorder) if recorder is not None else model.optimize()
        progress = recorder.finalize(model) if recorder is not None else None
        status = model.getStatusName()
        accepted = (
            {"optimal", "suboptimal", "timelimit", "solutionlimit"}
            if self.integral
            else {"optimal", "suboptimal"}
        )
        if model.getSolutionCount() <= 0 or status not in accepted:
            raise V7RootCgIncompleteError(
                f"V7 global pattern master is incomplete: status={status}",
                {
                    "status": status,
                    "solution_count": model.getSolutionCount(),
                    "pattern_count": len(self.patterns),
                    "integral": self.integral,
                },
            )
        duals: dict[str, dict[object, float]] = {}
        if not self.integral:
            duals = {
                family: {
                    key: model.getLinearDual(row) for key, row in rows.items()
                }
                for family, rows in self.constraints.items()
            }
        return V7MasterSolution(
            objective=model.getObjectiveValue(),
            pattern_values={
                pattern_id: model.getValue(variable)
                for pattern_id, variable in self.pattern_variables.items()
                if model.getValue(variable) > 1e-9
            },
            group_bay_flow={
                key: model.getValue(variable)
                for key, variable in self.group_bay_flow_variables.items()
                if model.getValue(variable) > 1e-9
            },
            import_reservation={
                key: model.getValue(variable)
                for key, variable in self.import_variables.items()
                if model.getValue(variable) > 1e-9
            },
            duals=duals,
            diagnostics={
                "status": status,
                "integral": self.integral,
                "runtime_seconds": model.getRuntime(),
                "wall_seconds": perf_counter() - started,
                "pattern_count": len(self.patterns),
                "constraint_count_by_family": {
                    family: len(rows) for family, rows in sorted(self.constraints.items())
                },
                "solver_bound": model.getBestBound(),
                "solver_gap": model.getMipGap() if self.integral else 0.0,
                "progress": progress,
            },
        )

    def add_patterns(
        self,
        patterns: Iterable[V7BayPattern],
    ) -> tuple[V7BayPattern, ...]:
        """Add new LP columns in place and preserve the current simplex basis."""

        if self.integral:
            raise RuntimeError("V7 integer master does not accept dynamic columns")
        if self.model is None:
            self.build()
        model = self.model
        assert model is not None
        existing = {pattern.signature for pattern in self.patterns}
        next_id = max(self.patterns_by_id, default=-1) + 1
        added: list[V7BayPattern] = []
        atoms_by_index = {atom.candidate_index: atom for atom in self.atoms}
        for raw in sorted(
            patterns, key=lambda item: (item.anchor_bay_key, item.signature)
        ):
            if raw.signature in existing:
                continue
            pattern = replace(raw, pattern_id=next_id)
            self._validate_pattern(pattern, atoms_by_index)
            total_capacity = sum(dict(pattern.group_capacities).values())
            terms: list[tuple[float, object]] = [
                (
                    1.0,
                    self.constraints["bay_pattern_choice"][
                        pattern.anchor_bay_key
                    ],
                ),
                (
                    float(total_capacity),
                    self.constraints["reserved_anchor_size_capacity"][
                        (pattern.anchor_bay_key, pattern.size_mode)
                    ],
                ),
            ]
            for group_id, capacity in pattern.group_capacities:
                pair = (group_id, pattern.anchor_bay_key)
                terms.extend(
                    [
                        (
                            -float(capacity),
                            self.constraints["q_pattern_capacity"][pair],
                        ),
                        (-1.0, self.constraints["u_pattern_identity"][pair]),
                    ]
                )
                for physical in pattern.physical_bays:
                    key = (group_id, physical)
                    terms.extend(
                        [
                            (
                                1.0,
                                self.constraints[
                                    "physical_group_pattern_upper"
                                ][key],
                            ),
                            (
                                -1.0,
                                self.constraints[
                                    "physical_group_pattern_presence"
                                ][key],
                            ),
                        ]
                    )
            terms.extend(
                (
                    1.0,
                    self.constraints["physical_row_exclusivity"][resource],
                )
                for resource in pattern.physical_resources
            )
            terms.extend(
                (
                    float(total_capacity),
                    self.constraints["reserved_physical_capacity"][physical],
                )
                for physical in pattern.physical_bays
            )
            variable = model.addPricedVar(
                terms,
                lb=0.0,
                ub=1.0,
                vtype="C",
                name=f"lambda_pattern_{pattern.pattern_id}",
            )
            self.pattern_variables[pattern.pattern_id] = variable
            self.patterns_by_id[pattern.pattern_id] = pattern
            added.append(pattern)
            existing.add(pattern.signature)
            next_id += 1
        if added:
            self.patterns = (*self.patterns, *added)
            model.update()
        return tuple(added)

    def pattern_reduced_cost(
        self,
        pattern: V7BayPattern,
        duals: Mapping[str, Mapping[object, float]],
    ) -> float:
        total_capacity = sum(dict(pattern.group_capacities).values())
        terms: list[tuple[str, object, float]] = [
            ("bay_pattern_choice", pattern.anchor_bay_key, 1.0),
            (
                "reserved_anchor_size_capacity",
                (pattern.anchor_bay_key, pattern.size_mode),
                float(total_capacity),
            ),
        ]
        for group_id, capacity in pattern.group_capacities:
            pair = (group_id, pattern.anchor_bay_key)
            terms.extend(
                [
                    ("q_pattern_capacity", pair, -float(capacity)),
                    ("u_pattern_identity", pair, -1.0),
                ]
            )
            for physical in pattern.physical_bays:
                key = (group_id, physical)
                terms.extend(
                    [
                        ("physical_group_pattern_upper", key, 1.0),
                        ("physical_group_pattern_presence", key, -1.0),
                    ]
                )
        for resource in pattern.physical_resources:
            terms.append(("physical_row_exclusivity", resource, 1.0))
        for physical in pattern.physical_bays:
            terms.append(
                ("reserved_physical_capacity", physical, float(total_capacity))
            )
        return -sum(
            float(duals.get(family, {}).get(key, 0.0)) * coefficient
            for family, key, coefficient in terms
        )

    def pattern_atom_reduced_cost(
        self,
        atom: V7RowAtom,
        duals: Mapping[str, Mapping[object, float]],
    ) -> float:
        """Return the additive reduced-cost contribution of one row atom."""

        pair = (atom.group_id, atom.anchor_bay_key)
        terms: list[tuple[str, object, float]] = [
            (
                "q_pattern_capacity",
                pair,
                -float(atom.capacity),
            ),
            (
                "reserved_anchor_size_capacity",
                (atom.anchor_bay_key, atom.size),
                float(atom.capacity),
            ),
        ]
        terms.extend(
            ("physical_row_exclusivity", resource, 1.0)
            for resource in atom.physical_resources
        )
        terms.extend(
            (
                "reserved_physical_capacity",
                physical,
                float(atom.capacity),
            )
            for physical in atom.physical_bays
        )
        return -sum(
            float(duals.get(family, {}).get(key, 0.0)) * coefficient
            for family, key, coefficient in terms
        )

    def pattern_support_reduced_cost(
        self,
        anchor_bay_key: str,
        size: str,
        height: str,
        support: tuple[str, ...],
        physical_bays: tuple[str, ...],
        duals: Mapping[str, Mapping[object, float]],
    ) -> float:
        """Return the fixed reduced-cost contribution of one group support."""

        del size, height  # State feasibility is enforced by legal atoms and pricing.
        terms: list[tuple[str, object, float]] = [
            ("bay_pattern_choice", str(anchor_bay_key), 1.0)
        ]
        for group_id in support:
            pair = (str(group_id), str(anchor_bay_key))
            terms.append(("u_pattern_identity", pair, -1.0))
            for physical in physical_bays:
                key = (str(group_id), str(physical))
                terms.extend(
                    [
                        ("physical_group_pattern_upper", key, 1.0),
                        ("physical_group_pattern_presence", key, -1.0),
                    ]
                )
        return -sum(
            float(duals.get(family, {}).get(key, 0.0)) * coefficient
            for family, key, coefficient in terms
        )

    def pattern_bay_reduced_cost(
        self,
        anchor_bay_key: str,
        duals: Mapping[str, Mapping[object, float]],
    ) -> float:
        return -float(
            duals.get("bay_pattern_choice", {}).get(str(anchor_bay_key), 0.0)
        )

    def pattern_group_support_reduced_cost(
        self,
        anchor_bay_key: str,
        size: str,
        height: str,
        group_id: str,
        physical_bays: tuple[str, ...],
        duals: Mapping[str, Mapping[object, float]],
    ) -> float:
        del size, height
        pair = (str(group_id), str(anchor_bay_key))
        terms: list[tuple[str, object, float]] = [
            ("u_pattern_identity", pair, -1.0)
        ]
        for physical in physical_bays:
            key = (str(group_id), str(physical))
            terms.extend(
                [
                    ("physical_group_pattern_upper", key, 1.0),
                    ("physical_group_pattern_presence", key, -1.0),
                ]
            )
        return -sum(
            float(duals.get(family, {}).get(key, 0.0)) * coefficient
            for family, key, coefficient in terms
        )

    def selected_atom_indices(self, solution: V7MasterSolution) -> tuple[int, ...]:
        indices = {
            index
            for pattern_id, value in solution.pattern_values.items()
            if value > 0.5
            for index in self.patterns_by_id[int(pattern_id)].candidate_indices
        }
        return tuple(sorted(indices))

    def dispose(self) -> None:
        if self.model is not None:
            self.model.dispose()
            self.model = None


def build_full_v7_pattern_universe(
    problem: ProblemData,
    atoms: Sequence[V7RowAtom],
    *,
    maximum_patterns: int = 100_000,
) -> tuple[V7BayPattern, ...]:
    patterns: list[V7BayPattern] = []
    for anchor in sorted({atom.anchor_bay_key for atom in atoms}):
        new = enumerate_v7_bay_patterns(
            problem,
            atoms,
            anchor,
            start_pattern_id=len(patterns),
            maximum_patterns=max(0, maximum_patterns - len(patterns)),
        )
        patterns.extend(new)
        if len(patterns) > int(maximum_patterns):
            raise RuntimeError("V7 full pattern micro oracle exceeded its safety limit")
    return normalize_v7_pattern_ids(patterns)


def solve_full_pattern_oracle(
    problem: ProblemData,
    peak_policy: V7PeakUtilizationPolicy,
    *,
    atoms: Sequence[V7RowAtom] | None = None,
    objective: V7ObjectiveConfig | None = None,
    integral: bool = False,
    time_limit: float = 10.0,
) -> tuple[V7MasterSolution, tuple[V7BayPattern, ...]]:
    if atoms is None:
        atoms, _limits = build_v7_row_atoms(problem)
    patterns = build_full_v7_pattern_universe(problem, atoms)
    master = V7GlobalPatternMaster(
        problem,
        atoms,
        patterns,
        peak_policy,
        objective,
        integral=integral,
        time_limit=time_limit,
    )
    try:
        return master.solve(), patterns
    finally:
        master.dispose()


class V7RootColumnGeneration:
    """Active-domain CG with mandatory exact full-domain certification."""

    def __init__(
        self,
        problem: ProblemData,
        peak_policy: V7PeakUtilizationPolicy,
        active_areas_by_group: Mapping[str, Iterable[str]],
        initial_patterns: Sequence[V7BayPattern],
        config: V7RootCgConfig | None = None,
        *,
        atoms: Sequence[V7RowAtom] | None = None,
    ) -> None:
        self.problem = problem
        self.peak_policy = peak_policy
        self.config = config or V7RootCgConfig()
        self.config.validate()
        if atoms is None:
            atoms, _limits = build_v7_row_atoms(problem)
        self.atoms = tuple(atoms)
        self.evaluator = V7ModelEvaluator(
            problem, self.atoms, self.config.objective
        )
        self.active = {
            str(group_id): {str(area) for area in areas}
            for group_id, areas in active_areas_by_group.items()
        }
        for group in self.evaluator.groups:
            if not self.active.get(group.group_id):
                raise ValueError(f"V7 root has no active area for {group.group_id}")
        self.patterns = list(normalize_v7_pattern_ids(initial_patterns))
        if not self.patterns:
            raise ValueError("V7 root requires explicit feasible initial patterns")
        self.pricing = V7ExactBayPricing(
            problem,
            self.atoms,
            columns_per_bay=self.config.columns_per_bay_per_round,
            reduced_cost_tolerance=self.config.reduced_cost_tolerance,
        )

    def _allowed_groups(self, anchor: str, full_domain: bool) -> set[str] | None:
        if full_domain:
            return None
        area = str(self.problem.bays[anchor].area_no)
        return {
            group.group_id
            for group in self.evaluator.groups
            if area in self.active[group.group_id]
        }

    def solve(self) -> V7RootCgResult:
        started = perf_counter()
        initial_count = len(self.patterns)
        iterations = 0
        expansions = 0
        activated_pairs: set[tuple[str, str]] = set()
        certification_count = 0
        certification_seconds = 0.0
        pricing_rows: list[dict[str, object]] = []
        iteration_rows: list[dict[str, object]] = []
        master_solve_seconds = 0.0
        active_pricing_seconds = 0.0
        final_minimum = math.inf
        solution: V7MasterSolution | None = None
        master = V7GlobalPatternMaster(
            self.problem,
            self.atoms,
            self.patterns,
            self.peak_policy,
            self.config.objective,
            integral=False,
            time_limit=self.config.root_time_limit,
            solver_threads=self.config.solver_threads,
            solver_seed=self.config.solver_seed,
            verbose=self.config.verbose,
        )

        def incomplete_diagnostics() -> dict[str, object]:
            return {
                "root_closed": False,
                "iterations": iterations,
                "pattern_count": len(master.patterns),
                "last_root_objective": (
                    float(solution.objective) if solution is not None else None
                ),
                "root_total_seconds": perf_counter() - started,
                "master_solve_seconds": master_solve_seconds,
                "active_pricing_seconds": active_pricing_seconds,
                "full_domain_certification_count": certification_count,
                "full_domain_certification_seconds": certification_seconds,
                "iteration_summaries": list(iteration_rows),
            }

        try:
            while iterations < int(self.config.maximum_iterations):
                if perf_counter() - started > float(self.config.root_time_limit):
                    raise V7RootCgIncompleteError(
                        "V7 root CG reached its declared time limit",
                        incomplete_diagnostics(),
                    )
                iterations += 1
                iteration_started = perf_counter()
                if master.model is not None:
                    master.model.setParam(
                        "TimeLimit",
                        max(
                            0.1,
                            self.config.root_time_limit
                            - (perf_counter() - started),
                        ),
                    )
                solution = master.solve()
                master_solve_seconds += float(
                    solution.diagnostics.get("wall_seconds", 0.0)
                )
                existing = {pattern.signature for pattern in master.patterns}
                new_patterns: list[V7BayPattern] = []
                active_minimum = math.inf
                active_started = perf_counter()
                for anchor in sorted({atom.anchor_bay_key for atom in self.atoms}):
                    result = self.pricing.price_bay_exact_mip(
                        anchor,
                        lambda atom, m=master, d=solution.duals: m.pattern_atom_reduced_cost(atom, d),
                        lambda bay, m=master, d=solution.duals: m.pattern_bay_reduced_cost(bay, d),
                        lambda bay, size, height, group, physical, m=master, d=solution.duals: m.pattern_group_support_reduced_cost(
                            bay, size, height, group, physical, d
                        ),
                        verify_reduced_cost=lambda pattern, m=master, d=solution.duals: m.pattern_reduced_cost(pattern, d),
                        allowed_groups=self._allowed_groups(anchor, False),
                        excluded_signatures=existing,
                        solver_threads=self.config.solver_threads,
                    )
                    if result.minimum_reduced_cost is not None:
                        active_minimum = min(active_minimum, result.minimum_reduced_cost)
                    new_patterns.extend(result.returned_patterns)
                    pricing_rows.append(
                        {"iteration": iterations, "domain": "active", **result.diagnostics}
                    )
                active_pricing_seconds += perf_counter() - active_started
                if new_patterns:
                    added = master.add_patterns(new_patterns)
                    self.patterns = list(master.patterns)
                    iteration_rows.append(
                        {
                            "iteration": iterations,
                            "domain": "active",
                            "objective": float(solution.objective),
                            "minimum_reduced_cost": (
                                None
                                if not math.isfinite(active_minimum)
                                else float(active_minimum)
                            ),
                            "columns_added": len(added),
                            "pattern_count": len(master.patterns),
                            "seconds": perf_counter() - iteration_started,
                        }
                    )
                    continue

                certification_count += 1
                certification_started = perf_counter()
                full_new: list[V7BayPattern] = []
                full_minimum = math.inf
                for anchor in sorted({atom.anchor_bay_key for atom in self.atoms}):
                    result = self.pricing.price_bay_exact_mip(
                        anchor,
                        lambda atom, m=master, d=solution.duals: m.pattern_atom_reduced_cost(atom, d),
                        lambda bay, m=master, d=solution.duals: m.pattern_bay_reduced_cost(bay, d),
                        lambda bay, size, height, group, physical, m=master, d=solution.duals: m.pattern_group_support_reduced_cost(
                            bay, size, height, group, physical, d
                        ),
                        verify_reduced_cost=lambda pattern, m=master, d=solution.duals: m.pattern_reduced_cost(pattern, d),
                        allowed_groups=None,
                        excluded_signatures=existing,
                        solver_threads=self.config.solver_threads,
                    )
                    if result.minimum_reduced_cost is not None:
                        full_minimum = min(full_minimum, result.minimum_reduced_cost)
                    full_new.extend(result.returned_patterns)
                    pricing_rows.append(
                        {"iteration": iterations, "domain": "full", **result.diagnostics}
                    )
                certification_seconds += perf_counter() - certification_started
                final_minimum = min(active_minimum, full_minimum)
                if full_new:
                    for pattern in full_new:
                        area = str(self.problem.bays[pattern.anchor_bay_key].area_no)
                        for group_id in pattern.active_groups:
                            pair = (group_id, area)
                            if area not in self.active[group_id]:
                                self.active[group_id].add(area)
                                activated_pairs.add(pair)
                                expansions += 1
                    added = master.add_patterns(full_new)
                    self.patterns = list(master.patterns)
                    iteration_rows.append(
                        {
                            "iteration": iterations,
                            "domain": "full",
                            "objective": float(solution.objective),
                            "minimum_reduced_cost": (
                                None
                                if not math.isfinite(full_minimum)
                                else float(full_minimum)
                            ),
                            "columns_added": len(added),
                            "pattern_count": len(master.patterns),
                            "seconds": perf_counter() - iteration_started,
                        }
                    )
                    continue
                iteration_rows.append(
                    {
                        "iteration": iterations,
                        "domain": "full_certified",
                        "objective": float(solution.objective),
                        "minimum_reduced_cost": (
                            0.0
                            if not math.isfinite(final_minimum)
                            else float(final_minimum)
                        ),
                        "columns_added": 0,
                        "pattern_count": len(master.patterns),
                        "seconds": perf_counter() - iteration_started,
                    }
                )
                root_closed = True
                break
            else:
                raise V7RootCgIncompleteError(
                    "V7 root CG reached its iteration limit",
                    incomplete_diagnostics(),
                )
            assert solution is not None
            self.patterns = list(master.patterns)
            diagnostics = {
                "algorithm": "v7_active_area_global_bay_pattern_cg",
                "model_schema_version": V7_MODEL_SCHEMA_VERSION,
                "root_objective": float(solution.objective),
                "root_iterations": iterations,
                "root_total_seconds": perf_counter() - started,
                "master_solve_seconds": master_solve_seconds,
                "active_pricing_seconds": active_pricing_seconds,
                "initial_pattern_count": initial_count,
                "final_pattern_count": len(self.patterns),
                "active_area_expansion_count": expansions,
                "group_area_pairs_activated_by_global_pricing": [
                    list(pair) for pair in sorted(activated_pairs)
                ],
                "full_domain_certification_count": certification_count,
                "full_domain_certification_seconds": certification_seconds,
                "final_minimum_reduced_cost": (
                    0.0 if not math.isfinite(final_minimum) else float(final_minimum)
                ),
                "root_closed": root_closed,
                "stage1_quota_fixed": False,
                "full_domain_exact_pricing_required": True,
                "incremental_master_column_addition": True,
                "iteration_summaries": iteration_rows,
                "pricing_iterations": pricing_rows,
            }
            return V7RootCgResult(
                objective=float(solution.objective),
                patterns=tuple(self.patterns),
                solution=solution,
                active_areas_by_group={
                    group_id: frozenset(areas) for group_id, areas in self.active.items()
                },
                diagnostics=diagnostics,
            )
        finally:
            master.dispose()


__all__ = [
    "V7GlobalPatternMaster",
    "V7MasterSolution",
    "V7RootCgConfig",
    "V7RootCgIncompleteError",
    "V7RootCgResult",
    "V7RootColumnGeneration",
    "build_full_v7_pattern_universe",
    "normalize_v7_pattern_ids",
    "patterns_from_atom_solution",
    "solve_full_pattern_oracle",
]
