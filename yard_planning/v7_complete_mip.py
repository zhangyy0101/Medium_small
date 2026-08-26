"""Direct compact implementation of the formal zone-free V7 model."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Mapping

from .gurobi_backend import GurobiModel, MipProgressRecorder
from .models import ProblemData
from .row_aware_zones import v6_footprint
from .v7_atoms import V7RowAtom, atoms_by_group_bay, build_v7_row_atoms
from .v7_model import (
    V7_MODEL_SCHEMA_VERSION,
    V7ModelEvaluator,
    V7ObjectiveConfig,
    V7PeakUtilizationPolicy,
    derive_v7_analytic_peak_policy,
)


class V7CompleteMipIncompleteError(RuntimeError):
    def __init__(self, message: str, diagnostics: Mapping[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class V7CompleteMipConfig:
    time_limit: float = 30.0
    mip_gap: float = 0.0
    solver_threads: int = 1
    solver_seed: int = 0
    verbose: bool = False
    require_business_optimality: bool = True
    max_new_groups_per_physical_bay: int = 3
    objective: V7ObjectiveConfig = field(default_factory=V7ObjectiveConfig)

    def validate(self) -> None:
        if not math.isfinite(float(self.time_limit)) or float(self.time_limit) <= 0:
            raise ValueError("V7 complete-MIP time limit must be positive")
        if not math.isfinite(float(self.mip_gap)) or not 0 <= float(self.mip_gap) <= 1:
            raise ValueError("V7 complete-MIP gap must lie in [0, 1]")
        if int(self.solver_threads) < 0:
            raise ValueError("V7 solver_threads must be nonnegative")
        if int(self.max_new_groups_per_physical_bay) != 3:
            raise ValueError("V7 baseline fixes max new groups per physical bay to 3")
        self.objective.validate()


@dataclass(frozen=True)
class V7CompleteMipResult:
    selected_atom_indices: tuple[int, ...]
    group_bay_flow: Mapping[tuple[str, str], int]
    import_reservation: Mapping[tuple[str, str, str], int]
    peak_policy: V7PeakUtilizationPolicy
    certificate: Mapping[str, object]
    diagnostics: Mapping[str, object]

    @property
    def objective(self) -> float:
        return float(self.certificate["objective"])


@dataclass
class _V7CompactArtifacts:
    model: GurobiModel
    atom_selected: dict[int, object]
    group_bay_flow: dict[tuple[str, str], object]
    import_reservation: dict[tuple[str, str, str], object]
    constraints: dict[str, list[object]]


class V7CompleteMipSolver:
    """Solve V7 directly over row atoms; no zone appears in this model."""

    def __init__(
        self,
        problem: ProblemData,
        config: V7CompleteMipConfig | None = None,
    ) -> None:
        self.problem = problem
        self.config = config or V7CompleteMipConfig()
        self.config.validate()
        self.atoms, self.anchor_capacity_limits = build_v7_row_atoms(problem)
        self.atoms_by_pair = atoms_by_group_bay(self.atoms)
        self.evaluator = V7ModelEvaluator(
            problem,
            self.atoms,
            self.config.objective,
            max_new_groups_per_physical_bay=(
                self.config.max_new_groups_per_physical_bay
            ),
        )
        self.groups = self.evaluator.groups
        self.groups_by_id = self.evaluator.groups_by_id

    def _configure(self, model: GurobiModel) -> None:
        if not self.config.verbose:
            model.hideOutput()
        model.setMinimize()
        model.setParam("TimeLimit", float(self.config.time_limit))
        model.setParam("MIPGap", float(self.config.mip_gap))
        model.setParam("Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            model.setParam("Threads", int(self.config.solver_threads))
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("IntFeasTol", 1e-9)

    def _build(
        self,
        peak_policy: V7PeakUtilizationPolicy,
        *,
        business_objective: bool,
        relax_integrality: bool = False,
        patterns: tuple[object, ...] = (),
    ) -> _V7CompactArtifacts:
        del patterns  # Reserved for a common oracle signature; compact uses atoms.
        model = GurobiModel(
            "v7_complete_business" if business_objective else "v7_peak_witness"
        )
        self._configure(model)
        gp = model._gp
        quicksum = gp.quicksum
        binary = "C" if relax_integrality else "B"
        integer = "C" if relax_integrality else "I"
        constraints: defaultdict[str, list[object]] = defaultdict(list)

        atom_selected = {
            atom.candidate_index: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=binary,
                name=f"z_atom_{atom.candidate_index}",
            )
            for atom in self.atoms
        }
        group_bay_use = {
            pair: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=binary,
                obj=(
                    self.evaluator.group_bay_use_objective_coefficient()
                    if business_objective
                    else 0.0
                ),
                name=f"u_group_bay_{index}",
            )
            for index, pair in enumerate(sorted(self.atoms_by_pair))
        }
        group_bay_flow = {
            pair: model.addVar(
                lb=0.0,
                ub=float(self.groups_by_id[pair[0]].demand),
                vtype=integer,
                obj=(
                    self.evaluator.group_bay_flow_objective_coefficient(*pair)
                    if business_objective
                    else 0.0
                ),
                name=f"q_group_bay_{index}",
            )
            for index, pair in enumerate(sorted(self.atoms_by_pair))
        }

        for pair, pair_atoms in sorted(self.atoms_by_pair.items()):
            selected_sum = quicksum(
                atom_selected[atom.candidate_index] for atom in pair_atoms
            )
            reserved = quicksum(
                atom.capacity * atom_selected[atom.candidate_index]
                for atom in pair_atoms
            )
            for atom in pair_atoms:
                constraints["atom_to_group_bay_use"].append(
                    model.addConstr(
                        atom_selected[atom.candidate_index] <= group_bay_use[pair]
                    )
                )
            constraints["group_bay_atom_presence"].append(
                model.addConstr(group_bay_use[pair] <= selected_sum)
            )
            constraints["positive_group_bay_flow"].append(
                model.addConstr(group_bay_flow[pair] >= group_bay_use[pair])
            )
            constraints["group_bay_flow_use_upper"].append(
                model.addConstr(
                    group_bay_flow[pair]
                    <= int(self.groups_by_id[pair[0]].demand) * group_bay_use[pair]
                )
            )
            constraints["group_bay_row_capacity"].append(
                model.addConstr(group_bay_flow[pair] <= reserved)
            )
            constraints["anchor_capacity_limit"].append(
                model.addConstr(
                    reserved <= int(self.anchor_capacity_limits[pair])
                )
            )
        for group in self.groups:
            constraints["exact_export_demand"].append(
                model.addConstr(
                    quicksum(
                        variable
                        for (group_id, _bay), variable in group_bay_flow.items()
                        if group_id == group.group_id
                    )
                    == int(group.demand)
                )
            )

        atoms_by_resource: defaultdict[tuple[str, str], list[V7RowAtom]] = defaultdict(list)
        reserved_by_physical: defaultdict[str, list[tuple[int, object]]] = defaultdict(list)
        reserved_by_anchor_size: defaultdict[
            tuple[str, str], list[tuple[int, object]]
        ] = defaultdict(list)
        for atom in self.atoms:
            variable = atom_selected[atom.candidate_index]
            for resource in atom.physical_resources:
                atoms_by_resource[resource].append(atom)
            reserved_by_anchor_size[(atom.anchor_bay_key, atom.size)].append(
                (atom.capacity, variable)
            )
            for physical in atom.physical_bays:
                reserved_by_physical[physical].append((atom.capacity, variable))
        for resource, values in sorted(atoms_by_resource.items()):
            constraints["physical_row_exclusivity"].append(
                model.addConstr(
                    quicksum(atom_selected[atom.candidate_index] for atom in values)
                    <= 1
                )
            )
        for physical, terms in sorted(reserved_by_physical.items()):
            constraints["reserved_physical_capacity"].append(
                model.addConstr(
                    quicksum(coefficient * variable for coefficient, variable in terms)
                    <= int(self.problem.bays[physical].physical_capacity)
                )
            )
        for (bay_key, size), terms in sorted(reserved_by_anchor_size.items()):
            constraints["reserved_anchor_size_capacity"].append(
                model.addConstr(
                    quicksum(coefficient * variable for coefficient, variable in terms)
                    <= int(self.problem.bays[bay_key].cap_by_size.get(size, 0))
                )
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
                obj=(
                    self.evaluator.group_area_use_objective_coefficient()
                    if business_objective
                    else 0.0
                ),
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
        group_area_span = {
            key: model.addVar(
                lb=0.0,
                ub=1.0,
                obj=(
                    self.evaluator.group_area_span_objective_coefficient()
                    if business_objective
                    else 0.0
                ),
                name=f"span_group_area_{index}",
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
                constraints["group_bay_to_area"].append(
                    model.addConstr(use <= group_area_use[key])
                )
                constraints["span_upper_envelope"].append(
                    model.addConstr(span_upper[key] >= order * use)
                )
                constraints["span_lower_envelope"].append(
                    model.addConstr(span_lower[key] <= order + 1 - use)
                )
            constraints["group_area_presence"].append(
                model.addConstr(group_area_use[key] <= quicksum(group_bay_use[p] for p in pairs))
            )
            constraints["span_lower_area_link"].append(
                model.addConstr(span_lower[key] <= group_area_use[key])
            )
            constraints["span_upper_area_link"].append(
                model.addConstr(span_upper[key] <= group_area_use[key])
            )
            constraints["span_definition"].append(
                model.addConstr(group_area_span[key] >= span_upper[key] - span_lower[key])
            )

        group_physical_to_uses: defaultdict[
            tuple[str, str], list[object]
        ] = defaultdict(list)
        for pair, variable in group_bay_use.items():
            group = self.groups_by_id[pair[0]]
            for physical in self._footprint(pair[1], group.size):
                group_physical_to_uses[(pair[0], physical)].append(variable)
        physical_group_use = {
            key: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=binary,
                name=f"u_physical_group_{index}",
            )
            for index, key in enumerate(sorted(group_physical_to_uses))
        }
        for key, uses in sorted(group_physical_to_uses.items()):
            for use in uses:
                constraints["physical_group_use_lower"].append(
                    model.addConstr(use <= physical_group_use[key])
                )
            constraints["physical_group_use_upper"].append(
                model.addConstr(physical_group_use[key] <= quicksum(uses))
            )
        physical_bays = sorted({physical for _group, physical in physical_group_use})
        for physical in physical_bays:
            constraints["max_three_groups_per_physical_bay"].append(
                model.addConstr(
                    quicksum(
                        variable
                        for (group_id, bay_key), variable in physical_group_use.items()
                        if bay_key == physical
                    )
                    <= 3
                )
            )

        size_states = {
            (physical, size): model.addVar(
                lb=0.0, ub=1.0, vtype=binary, name=f"export_size_{physical}_{size}"
            )
            for physical in physical_bays
            for size in sorted(
                {
                    self.groups_by_id[group_id].size
                    for group_id, bay_key in physical_group_use
                    if bay_key == physical
                }
            )
        }
        height_states = {
            (physical, height): model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=binary,
                name=f"export_height_{physical}_{height}",
            )
            for physical in physical_bays
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
            constraints["export_size_state_link"].append(
                model.addConstr(use <= size_states[(physical, group.size)])
            )
            constraints["export_height_state_link"].append(
                model.addConstr(use <= height_states[(physical, group.height)])
            )
        for physical in physical_bays:
            constraints["export_size_state_choice"].append(
                model.addConstr(
                    quicksum(
                        variable
                        for (bay_key, _state), variable in size_states.items()
                        if bay_key == physical
                    )
                    <= 1
                )
            )
            constraints["export_height_state_choice"].append(
                model.addConstr(
                    quicksum(
                        variable
                        for (bay_key, _state), variable in height_states.items()
                        if bay_key == physical
                    )
                    <= 1
                )
            )

        import_rows = [
            (flow, size, bay_key, capacity)
            for (flow, size), candidates in sorted(self.evaluator.import_candidates.items())
            for bay_key, capacity in candidates
        ]
        import_reservation = {
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
        for (flow, size, bay_key), variable in import_reservation.items():
            import_by_flow_size[(flow, size)].append(variable)
            import_by_anchor_size[(bay_key, size)].append(variable)
            for physical in self._footprint(bay_key, size):
                import_by_physical[physical].append(variable)
                import_by_physical_size[(physical, size)].append(variable)
        for key, demand in sorted(self.problem.import_demand_by_flow_size.items()):
            normalized = tuple(map(str, key))
            constraints["exact_import_demand"].append(
                model.addConstr(
                    quicksum(import_by_flow_size.get(normalized, [])) == int(demand)
                )
            )
        for (bay_key, size), variables in sorted(import_by_anchor_size.items()):
            constraints["import_anchor_size_capacity"].append(
                model.addConstr(
                    quicksum(variables)
                    <= int(self.problem.bays[bay_key].cap_by_size.get(size, 0))
                )
            )

        allocation_physical = sorted(set(physical_bays) | set(import_by_physical))
        export_use = {
            physical: model.addVar(lb=0.0, ub=1.0, vtype=binary, name=f"export_use_{i}")
            for i, physical in enumerate(allocation_physical)
        }
        import_use = {
            physical: model.addVar(lb=0.0, ub=1.0, vtype=binary, name=f"import_use_{i}")
            for i, physical in enumerate(allocation_physical)
        }
        import_size_state = {
            key: model.addVar(lb=0.0, ub=1.0, vtype=binary, name=f"import_size_{i}")
            for i, key in enumerate(sorted(import_by_physical_size))
        }
        for physical in allocation_physical:
            export_group_states = [
                variable
                for (group_id, bay_key), variable in physical_group_use.items()
                if bay_key == physical
            ]
            for state in export_group_states:
                constraints["export_use_lower"].append(
                    model.addConstr(state <= export_use[physical])
                )
            if export_group_states:
                constraints["export_use_upper"].append(
                    model.addConstr(export_use[physical] <= quicksum(export_group_states))
                )
            else:
                constraints["export_use_zero"].append(
                    model.addConstr(export_use[physical] == 0)
                )
            imports = import_by_physical.get(physical, [])
            if imports:
                capacity = max(1, int(self.problem.bays[physical].physical_capacity))
                constraints["import_physical_capacity"].append(
                    model.addConstr(quicksum(imports) <= capacity * import_use[physical])
                )
                constraints["import_use_presence"].append(
                    model.addConstr(import_use[physical] <= quicksum(imports))
                )
                for key, state in import_size_state.items():
                    if key[0] != physical:
                        continue
                    constraints["import_size_state_link"].append(
                        model.addConstr(
                            quicksum(import_by_physical_size[key]) <= capacity * state
                        )
                    )
                    constraints["import_size_state_presence"].append(
                        model.addConstr(state <= quicksum(import_by_physical_size[key]))
                    )
                constraints["import_size_state_choice"].append(
                    model.addConstr(
                        quicksum(
                            state
                            for (bay_key, _size), state in import_size_state.items()
                            if bay_key == physical
                        )
                        <= import_use[physical]
                    )
                )
            else:
                constraints["import_use_zero"].append(
                    model.addConstr(import_use[physical] == 0)
                )
            constraints["export_import_physical_exclusivity"].append(
                model.addConstr(export_use[physical] + import_use[physical] <= 1)
            )

        planned_load_by_area: defaultdict[str, list[tuple[int, object]]] = defaultdict(list)
        for (group_id, bay_key), variable in group_bay_flow.items():
            footprint = self._footprint(bay_key, self.groups_by_id[group_id].size)
            planned_load_by_area[str(self.problem.bays[bay_key].area_no)].append(
                (len(footprint), variable)
            )
        for (_flow, size, bay_key), variable in import_reservation.items():
            footprint = self._footprint(bay_key, size)
            planned_load_by_area[str(self.problem.bays[bay_key].area_no)].append(
                (len(footprint), variable)
            )
        area_capacity: Counter[str] = Counter()
        for bay in self.problem.bays.values():
            area_capacity[str(bay.area_no)] += int(bay.physical_capacity)
        for area, terms in sorted(planned_load_by_area.items()):
            constraints["peak_utilization_hard_cap"].append(
                model.addConstr(
                    quicksum(coefficient * variable for coefficient, variable in terms)
                    <= float(peak_policy.epsilon_cap) * int(area_capacity[area])
                )
            )
        if business_objective:
            model.addVar(
                lb=1.0,
                ub=1.0,
                obj=self.evaluator.objective_constant(),
                name="v7_objective_constant",
            )
        model.update()
        return _V7CompactArtifacts(
            model=model,
            atom_selected=atom_selected,
            group_bay_flow=group_bay_flow,
            import_reservation=import_reservation,
            constraints=dict(constraints),
        )

    def _footprint(self, bay_key: str, size: str) -> tuple[str, ...]:
        return tuple(v6_footprint(self.problem, str(bay_key), str(size)))

    def solve(
        self,
        peak_policy: V7PeakUtilizationPolicy | None = None,
        *,
        feasibility_only: bool = False,
    ) -> V7CompleteMipResult:
        started = perf_counter()
        if peak_policy is None:
            peak_policy, peak_diagnostics = derive_v7_analytic_peak_policy(
                self.problem,
                self.config.objective,
                atoms=self.atoms,
            )
        else:
            peak_diagnostics = {"provided": True, **peak_policy.as_dict()}
        artifacts = self._build(
            peak_policy,
            business_objective=not feasibility_only,
        )
        recorder = MipProgressRecorder(
            phase="v7_peak_feasibility" if feasibility_only else "v7_complete_business"
        )
        try:
            artifacts.model.optimize(callback=recorder)
            progress = recorder.finalize(artifacts.model)
            status = artifacts.model.getStatusName()
            diagnostics = {
                "algorithm": "v7_complete_compact_row_atom_mip",
                "model_schema_version": V7_MODEL_SCHEMA_VERSION,
                "objective_mode": "feasibility" if feasibility_only else "business",
                "status": status,
                "proven_optimal": status == "optimal",
                "solution_count": artifacts.model.getSolutionCount(),
                "solver_objective": (
                    artifacts.model.getObjectiveValue()
                    if artifacts.model.getSolutionCount() > 0
                    else None
                ),
                "solver_bound": artifacts.model.getBestBound(),
                "solver_gap": (
                    artifacts.model.getMipGap()
                    if artifacts.model.getSolutionCount() > 0
                    else None
                ),
                "runtime_seconds": artifacts.model.getRuntime(),
                "atom_count": len(self.atoms),
                "constraint_count_by_family": {
                    key: len(value) for key, value in sorted(artifacts.constraints.items())
                },
                "peak_preparation": peak_diagnostics,
                "progress": progress,
                "zone_variables_used": False,
                "area_balance_secondary_objective_used": False,
            }
            if artifacts.model.getSolutionCount() <= 0:
                raise V7CompleteMipIncompleteError(
                    f"V7 compact model found no incumbent: status={status}", diagnostics
                )
            if (
                not feasibility_only
                and self.config.require_business_optimality
                and status != "optimal"
            ):
                raise V7CompleteMipIncompleteError(
                    f"V7 compact business oracle is incomplete: status={status}",
                    diagnostics,
                )
            selected = tuple(
                sorted(
                    index
                    for index, variable in artifacts.atom_selected.items()
                    if artifacts.model.getValue(variable) > 0.5
                )
            )
            flow = {
                key: int(round(artifacts.model.getValue(variable)))
                for key, variable in artifacts.group_bay_flow.items()
                if artifacts.model.getValue(variable) > 0.5
            }
            imports = {
                key: int(round(artifacts.model.getValue(variable)))
                for key, variable in artifacts.import_reservation.items()
                if artifacts.model.getValue(variable) > 0.5
            }
            certificate = self.evaluator.evaluate(
                selected,
                flow,
                imports,
                peak_policy,
            )
            reconstructed = float(certificate["objective"])
            difference = None
            if not feasibility_only:
                solver_objective = artifacts.model.getObjectiveValue()
                difference = solver_objective - reconstructed
                if reconstructed > solver_objective + 1e-8:
                    raise RuntimeError(
                        "V7 evaluator objective exceeds the compact incumbent: "
                        f"solver={solver_objective}, evaluator={reconstructed}"
                    )
                if status == "optimal" and not math.isclose(
                    solver_objective,
                    reconstructed,
                    rel_tol=1e-8,
                    abs_tol=1e-8,
                ):
                    raise RuntimeError(
                        "Optimal V7 compact objective differs from evaluator: "
                        f"solver={solver_objective}, evaluator={reconstructed}"
                    )
            diagnostics.update(
                {
                    "selected_atom_count": len(selected),
                    "positive_group_bay_count": len(flow),
                    "positive_import_count": len(imports),
                    "solver_evaluator_objective_difference": difference,
                    "certified_upper_bound": reconstructed,
                    "certified_gap": (
                        max(0.0, reconstructed - artifacts.model.getBestBound())
                        / max(abs(reconstructed), 1e-12)
                        if not feasibility_only
                        else None
                    ),
                    "incumbent_auxiliary_objective_repaired": bool(
                        not feasibility_only
                        and difference is not None
                        and difference > 1e-8
                    ),
                    "independent_validation_passed": True,
                    "total_seconds": perf_counter() - started,
                }
            )
            return V7CompleteMipResult(
                selected_atom_indices=selected,
                group_bay_flow=flow,
                import_reservation=imports,
                peak_policy=peak_policy,
                certificate=certificate,
                diagnostics=diagnostics,
            )
        finally:
            artifacts.model.dispose()


__all__ = [
    "V7CompleteMipConfig",
    "V7CompleteMipIncompleteError",
    "V7CompleteMipResult",
    "V7CompleteMipSolver",
]
