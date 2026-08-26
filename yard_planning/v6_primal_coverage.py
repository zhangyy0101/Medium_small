"""V6-native primal column coverage without complete zone enumeration.

The compact MIP selects row atoms and used anchor bays directly.  Consecutive
used bays of one group/area are then merged into valid V6 zones and injected
into the restricted integer master.  Exact root pricing remains unchanged.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Mapping, Sequence

from .gurobi_backend import GurobiModel, MipProgressRecorder
from .models import ProblemData
from .row_aware_zones import (
    RowAwareBayAtom,
    RowAwareZone,
    build_v6_row_aware_bay_atoms,
    make_row_aware_zone,
    v6_footprint,
)
from .v6_column_generation import (
    V6RootCgConfig,
    V6RootCgResult,
    V6RootColumnGeneration,
)
from .v6_model import (
    V6_MODEL_SCHEMA_VERSION,
    V6ModelEvaluator,
    V6ObjectiveConfig,
    V6PeakUtilizationPolicy,
    derive_v6_analytic_peak_policy,
)
from .v6_restricted_integer import (
    V6RestrictedIntegerConfig,
    V6RestrictedIntegerResult,
    V6RestrictedIntegerSolver,
    V6RestrictedIntegerWarmStart,
)


class V6CompactSolveIncompleteError(RuntimeError):
    """A compact solve ended before the proof required by its role."""

    def __init__(self, message: str, diagnostics: Mapping[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class V6PrimalCoverageConfig:
    """Execution controls for the compact V6 primal constructor."""

    time_limit: float = 60.0
    mip_gap: float = 0.0
    solver_threads: int = 1
    solver_seed: int = 0
    verbose: bool = False
    maximum_pool_solutions: int = 4
    objective: V6ObjectiveConfig = field(default_factory=V6ObjectiveConfig)

    def validate(self) -> None:
        if not math.isfinite(float(self.time_limit)) or float(
            self.time_limit
        ) <= 0.0:
            raise ValueError("V6 primal-coverage time limit must be positive")
        if not math.isfinite(float(self.mip_gap)) or not 0.0 <= float(
            self.mip_gap
        ) <= 1.0:
            raise ValueError("V6 primal-coverage MIP gap must lie in [0, 1]")
        if int(self.solver_threads) < 0:
            raise ValueError("V6 solver_threads must be nonnegative")
        if int(self.maximum_pool_solutions) <= 0:
            raise ValueError("V6 primal coverage pool size must be positive")
        self.objective.validate()


@dataclass(frozen=True)
class V6CompactPeakConfig:
    """Execution controls for the non-enumerative V6 min-max reference."""

    time_limit: float = 300.0
    solver_threads: int = 1
    solver_seed: int = 0
    verbose: bool = False
    objective: V6ObjectiveConfig = field(default_factory=V6ObjectiveConfig)

    def validate(self) -> None:
        if not math.isfinite(float(self.time_limit)) or float(
            self.time_limit
        ) <= 0.0:
            raise ValueError("V6 compact peak time limit must be positive")
        if int(self.solver_threads) < 0:
            raise ValueError("V6 solver_threads must be nonnegative")
        self.objective.validate()


@dataclass(frozen=True)
class V6AnalyticPeakConfig:
    """Controls the short feasibility certificate for the analytic cap."""

    feasibility_time_limit: float = 10.0
    solver_threads: int = 1
    solver_seed: int = 0
    verbose: bool = False
    objective: V6ObjectiveConfig = field(default_factory=V6ObjectiveConfig)

    def validate(self) -> None:
        if not math.isfinite(float(self.feasibility_time_limit)) or float(
            self.feasibility_time_limit
        ) <= 0.0:
            raise ValueError("V6 peak-feasibility time limit must be positive")
        if int(self.solver_threads) < 0:
            raise ValueError("V6 solver_threads must be nonnegative")
        self.objective.validate()


@dataclass(frozen=True)
class V6PrimalCoverageResult:
    """One validated compact-MIP incumbent and its recovered V6 zones."""

    selected_atom_indices: tuple[int, ...]
    selected_zone_ids: tuple[int, ...]
    zones: tuple[RowAwareZone, ...]
    candidate_zones: tuple[RowAwareZone, ...]
    zone_bay_flow: Mapping[tuple[int, str], int]
    import_reservation: Mapping[tuple[str, str, str], int]
    peak_policy: V6PeakUtilizationPolicy
    certificate: Mapping[str, object]
    diagnostics: Mapping[str, object]

    @property
    def objective(self) -> float:
        return float(self.certificate["objective"])


@dataclass(frozen=True)
class V6PrimalCoveredPipelineResult:
    root: V6RootCgResult
    coverage: V6PrimalCoverageResult
    integer: V6RestrictedIntegerResult
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class V6CompactPeakResult:
    peak_policy: V6PeakUtilizationPolicy
    selected_atom_indices: tuple[int, ...]
    zones: tuple[RowAwareZone, ...]
    zone_bay_flow: Mapping[tuple[int, str], int]
    import_reservation: Mapping[tuple[str, str, str], int]
    certificate: Mapping[str, object]
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class V6AnalyticPeakResult:
    peak_policy: V6PeakUtilizationPolicy
    witness: V6PrimalCoverageResult
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class V6ProductionPipelineResult:
    peak: V6AnalyticPeakResult
    planning: V6PrimalCoveredPipelineResult
    diagnostics: Mapping[str, object]


def _candidate_bays_by_group(
    atoms: Sequence[RowAwareBayAtom],
) -> dict[str, set[str]]:
    output: defaultdict[str, set[str]] = defaultdict(set)
    for atom in atoms:
        output[atom.group_id].add(atom.anchor_bay_key)
    return dict(output)


def _ordered_runs(
    atoms: Sequence[RowAwareBayAtom],
) -> dict[str, tuple[tuple[str, ...], ...]]:
    by_group_area_bay: defaultdict[
        tuple[str, str], defaultdict[str, list[RowAwareBayAtom]]
    ] = defaultdict(lambda: defaultdict(list))
    for atom in atoms:
        by_group_area_bay[(atom.group_id, atom.area_no)][
            atom.anchor_bay_key
        ].append(atom)
    output: defaultdict[str, list[tuple[str, ...]]] = defaultdict(list)
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
        output[group_id].extend(tuple(run) for run in runs)
    return {group_id: tuple(runs) for group_id, runs in output.items()}


class V6CompactPrimalCoverageSolver:
    """Find a V6 incumbent by selecting row atoms instead of enumerating zones."""

    def __init__(
        self,
        problem: ProblemData,
        peak_policy: V6PeakUtilizationPolicy | None,
        config: V6PrimalCoverageConfig | None = None,
        *,
        _objective_mode: str = "business",
        warm_start: V6PrimalCoverageResult | None = None,
    ) -> None:
        if _objective_mode not in {
            "business",
            "peak_minmax",
            "peak_feasibility",
        }:
            raise ValueError(f"unknown V6 compact objective: {_objective_mode}")
        if _objective_mode in {"business", "peak_feasibility"} and peak_policy is None:
            raise ValueError("V6 compact fixed-cap MIP requires a peak policy")
        if _objective_mode == "peak_minmax" and peak_policy is not None:
            raise ValueError("V6 compact peak MIP derives rather than accepts rho*")
        if warm_start is not None and _objective_mode != "business":
            raise ValueError("V6 compact warm start is supported only in business mode")
        if warm_start is not None and not math.isclose(
            float(warm_start.peak_policy.epsilon_cap),
            float(peak_policy.epsilon_cap),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("V6 compact warm start uses a different peak cap")
        self.problem = problem
        self.peak_policy = peak_policy
        self.objective_mode = _objective_mode
        self.warm_start = warm_start
        self.config = config or V6PrimalCoverageConfig()
        self.config.validate()
        self.atoms, self.anchor_capacity_limits = build_v6_row_aware_bay_atoms(
            problem
        )
        self.atoms_by_index = {
            atom.candidate_index: atom for atom in self.atoms
        }
        self.candidate_bays = _candidate_bays_by_group(self.atoms)
        self.evaluator = V6ModelEvaluator(
            problem,
            (),
            self.config.objective,
            candidate_bays_by_group=self.candidate_bays,
        )
        self.groups = self.evaluator.groups
        self.groups_by_id = self.evaluator.groups_by_id
        self.runs_by_group = _ordered_runs(self.atoms)

    def _configure_model(self, model: GurobiModel) -> None:
        if not self.config.verbose:
            model.hideOutput()
        model.setMinimize()
        model.setParam("Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            model.setParam("Threads", int(self.config.solver_threads))
        model.setParam("TimeLimit", float(self.config.time_limit))
        model.setParam("MIPGap", float(self.config.mip_gap))
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("IntFeasTol", 1e-9)
        if int(self.config.maximum_pool_solutions) > 1:
            model.setParam("PoolSearchMode", 2)
            model.setParam(
                "PoolSolutions", int(self.config.maximum_pool_solutions)
            )

    def _recover_zones(
        self,
        selected_atom_indices: Sequence[int],
        group_bay_flow: Mapping[tuple[str, str], int],
    ) -> tuple[tuple[RowAwareZone, ...], dict[tuple[int, str], int]]:
        selected_by_group_bay: defaultdict[
            tuple[str, str], list[RowAwareBayAtom]
        ] = defaultdict(list)
        for index in selected_atom_indices:
            atom = self.atoms_by_index[int(index)]
            selected_by_group_bay[
                (atom.group_id, atom.anchor_bay_key)
            ].append(atom)

        zones: list[RowAwareZone] = []
        zone_flow: dict[tuple[int, str], int] = {}
        covered_pairs: set[tuple[str, str]] = set()
        for group in self.groups:
            selected_bays = {
                bay_key
                for (group_id, bay_key), bay_atoms in selected_by_group_bay.items()
                if group_id == group.group_id and bay_atoms
            }
            for full_run in self.runs_by_group.get(group.group_id, ()):
                current: list[str] = []

                def flush() -> None:
                    if not current:
                        return
                    chosen_atoms = [
                        atom
                        for bay_key in current
                        for atom in selected_by_group_bay[
                            (group.group_id, bay_key)
                        ]
                    ]
                    zone = make_row_aware_zone(
                        chosen_atoms,
                        zone_id=len(zones),
                    )
                    zones.append(zone)
                    for bay_key in current:
                        pair = (group.group_id, bay_key)
                        quantity = int(group_bay_flow.get(pair, 0))
                        if quantity <= 0:
                            raise RuntimeError(
                                "selected V6 primal atom bay has no positive flow: "
                                f"pair={pair}"
                            )
                        zone_flow[(zone.zone_id, bay_key)] = quantity
                        covered_pairs.add(pair)
                    current.clear()

                for bay_key in full_run:
                    if bay_key in selected_bays:
                        current.append(bay_key)
                    else:
                        flush()
                flush()

        positive_pairs = {
            key for key, quantity in group_bay_flow.items() if int(quantity) > 0
        }
        if positive_pairs != covered_pairs:
            raise RuntimeError(
                "V6 compact primal zone recovery did not cover all positive "
                f"group-bay flows: missing={sorted(positive_pairs-covered_pairs)}, "
                f"extra={sorted(covered_pairs-positive_pairs)}"
            )
        return tuple(zones), zone_flow

    def _apply_business_warm_start(
        self,
        *,
        atom_selected: Mapping[int, object],
        bay_used: Mapping[tuple[str, str], object],
        group_bay_flow: Mapping[tuple[str, str], object],
        run_start: Mapping[tuple[str, str], object],
        export_size_state: Mapping[tuple[str, str], object],
        export_height_state: Mapping[tuple[str, str], object],
        import_reservation: Mapping[tuple[str, str, str], object],
        export_use: Mapping[str, object],
        import_use: Mapping[str, object],
        import_size_state: Mapping[tuple[str, str], object],
        voyage_area_use: Mapping[tuple[str, str], object],
    ) -> dict[str, object]:
        witness = self.warm_start
        if witness is None:
            return {"provided": False, "applied": False}

        zones_by_id = {zone.zone_id: zone for zone in witness.zones}
        warm_flow: Counter[tuple[str, str]] = Counter()
        for (zone_id, bay_key), quantity in witness.zone_bay_flow.items():
            zone = zones_by_id.get(int(zone_id))
            if zone is None:
                raise ValueError(
                    f"V6 warm start references unknown zone: {zone_id}"
                )
            warm_flow[(zone.group_id, str(bay_key))] += int(quantity)
        unknown_atoms = set(witness.selected_atom_indices) - set(atom_selected)
        if unknown_atoms:
            raise ValueError(
                f"V6 warm start references unknown row atoms: {sorted(unknown_atoms)}"
            )

        selected_atoms = set(witness.selected_atom_indices)
        positive_pairs = {
            pair for pair, quantity in warm_flow.items() if int(quantity) > 0
        }
        for index, variable in atom_selected.items():
            variable.Start = 1.0 if index in selected_atoms else 0.0
        for pair, variable in bay_used.items():
            variable.Start = 1.0 if pair in positive_pairs else 0.0
        for pair, variable in group_bay_flow.items():
            variable.Start = float(warm_flow.get(pair, 0))

        for group in self.groups:
            for run in self.runs_by_group.get(group.group_id, ()):
                previous_used = False
                for bay_key in run:
                    pair = (group.group_id, bay_key)
                    used = pair in positive_pairs
                    if pair in run_start:
                        run_start[pair].Start = 1.0 if used and not previous_used else 0.0
                    previous_used = used

        export_physical: set[str] = set()
        export_sizes: set[tuple[str, str]] = set()
        export_heights: set[tuple[str, str]] = set()
        used_voyage_areas: set[tuple[str, str]] = set()
        for (group_id, bay_key), quantity in warm_flow.items():
            if int(quantity) <= 0:
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
        for key, variable in export_size_state.items():
            variable.Start = 1.0 if key in export_sizes else 0.0
        for key, variable in export_height_state.items():
            variable.Start = 1.0 if key in export_heights else 0.0
        for key, variable in voyage_area_use.items():
            variable.Start = 1.0 if key in used_voyage_areas else 0.0

        warm_imports = {
            tuple(map(str, key)): int(quantity)
            for key, quantity in witness.import_reservation.items()
            if int(quantity) > 0
        }
        import_physical: set[str] = set()
        import_sizes: set[tuple[str, str]] = set()
        for (_flow, size, bay_key), quantity in warm_imports.items():
            if int(quantity) <= 0:
                continue
            for physical_bay_key in v6_footprint(
                self.problem, bay_key, size
            ):
                import_physical.add(physical_bay_key)
                import_sizes.add((physical_bay_key, size))
        for key, variable in import_reservation.items():
            variable.Start = float(warm_imports.get(tuple(map(str, key)), 0))
        for bay_key, variable in export_use.items():
            variable.Start = 1.0 if bay_key in export_physical else 0.0
        for bay_key, variable in import_use.items():
            variable.Start = 1.0 if bay_key in import_physical else 0.0
        for key, variable in import_size_state.items():
            variable.Start = 1.0 if key in import_sizes else 0.0

        return {
            "provided": True,
            "applied": True,
            "selected_atom_count": len(selected_atoms),
            "positive_group_bay_flow_count": len(positive_pairs),
            "positive_import_reservation_count": len(warm_imports),
            "source": "v6_analytic_peak_feasibility_witness",
        }

    def solve(self) -> V6PrimalCoverageResult:
        total_started = perf_counter()
        business = self.objective_mode == "business"
        minmax = self.objective_mode == "peak_minmax"
        model = GurobiModel(f"v6_compact_{self.objective_mode}")
        self._configure_model(model)
        gp = model._gp
        quicksum = gp.quicksum
        constraints: defaultdict[str, list[object]] = defaultdict(list)

        atoms_by_pair: defaultdict[
            tuple[str, str], list[RowAwareBayAtom]
        ] = defaultdict(list)
        for atom in self.atoms:
            atoms_by_pair[(atom.group_id, atom.anchor_bay_key)].append(atom)
        missing_groups = [
            group.group_id
            for group in self.groups
            if not any(pair[0] == group.group_id for pair in atoms_by_pair)
        ]
        if missing_groups:
            raise ValueError(
                f"V6 groups have no compact primal atoms: {missing_groups}"
            )

        unused_unit = (
            self.evaluator.unused_capacity_unit_objective_coefficient()
            if business
            else 0.0
        )
        atom_selected = {
            atom.candidate_index: model.addVar(
                vtype="B",
                obj=unused_unit * int(atom.capacity),
                name=f"a_atom_{atom.candidate_index}",
            )
            for atom in self.atoms
        }
        bay_used = {
            pair: model.addVar(
                vtype="B",
                name=f"u_group_bay_{index}",
            )
            for index, pair in enumerate(sorted(atoms_by_pair))
        }
        group_bay_flow = {
            pair: model.addVar(
                lb=0.0,
                ub=float(self.groups_by_id[pair[0]].demand),
                vtype="I",
                obj=(
                    self.evaluator.group_bay_flow_objective_coefficient(*pair)
                    if business
                    else 0.0
                ),
                name=f"q_group_bay_{index}",
            )
            for index, pair in enumerate(sorted(atoms_by_pair))
        }
        run_start: dict[tuple[str, str], object] = {}
        if business:
            for group in self.groups:
                for run in self.runs_by_group.get(group.group_id, ()):
                    previous = None
                    for bay_key in run:
                        pair = (group.group_id, bay_key)
                        start = model.addVar(
                            vtype="B",
                            obj=self.evaluator.zone_fixed_activation_objective_coefficient(),
                            name=f"run_start_{len(run_start)}",
                        )
                        run_start[pair] = start
                        if previous is None:
                            constraints["run_start_first"].append(
                                model.addConstr(start == bay_used[pair])
                            )
                        else:
                            constraints["run_start_lower"].append(
                                model.addConstr(
                                    start >= bay_used[pair] - bay_used[previous]
                                )
                            )
                            constraints["run_start_current"].append(
                                model.addConstr(start <= bay_used[pair])
                            )
                            constraints["run_start_previous"].append(
                                model.addConstr(start <= 1 - bay_used[previous])
                            )
                        previous = pair

        for pair, pair_atoms in sorted(atoms_by_pair.items()):
            atom_sum = quicksum(
                atom_selected[atom.candidate_index] for atom in pair_atoms
            )
            reserved = quicksum(
                atom.capacity * atom_selected[atom.candidate_index]
                for atom in pair_atoms
            )
            for atom in pair_atoms:
                constraints["atom_bay_link"].append(
                    model.addConstr(
                        atom_selected[atom.candidate_index] <= bay_used[pair]
                    )
                )
            constraints["bay_atom_presence"].append(
                model.addConstr(bay_used[pair] <= atom_sum)
            )
            constraints["positive_bay_flow"].append(
                model.addConstr(group_bay_flow[pair] >= bay_used[pair])
            )
            constraints["bay_flow_capacity"].append(
                model.addConstr(group_bay_flow[pair] <= reserved)
            )
            constraints["anchor_capacity"].append(
                model.addConstr(
                    reserved <= int(self.anchor_capacity_limits[pair])
                )
            )
        for group in self.groups:
            constraints["export_group_demand"].append(
                model.addConstr(
                    quicksum(
                        variable
                        for (group_id, _bay_key), variable in group_bay_flow.items()
                        if group_id == group.group_id
                    )
                    == int(group.demand)
                )
            )

        atoms_by_resource: defaultdict[
            tuple[str, str], list[RowAwareBayAtom]
        ] = defaultdict(list)
        reserved_by_physical: defaultdict[
            str, list[tuple[int, object]]
        ] = defaultdict(list)
        reserved_by_anchor_size: defaultdict[
            tuple[str, str], list[tuple[int, object]]
        ] = defaultdict(list)
        for atom in self.atoms:
            variable = atom_selected[atom.candidate_index]
            for resource in atom.resources:
                atoms_by_resource[resource].append(atom)
            group = self.groups_by_id[atom.group_id]
            reserved_by_anchor_size[
                (atom.anchor_bay_key, str(group.size))
            ].append((atom.capacity, variable))
            for physical_bay_key in v6_footprint(
                self.problem, atom.anchor_bay_key, group.size
            ):
                reserved_by_physical[physical_bay_key].append(
                    (atom.capacity, variable)
                )
        for resource, resource_atoms in sorted(atoms_by_resource.items()):
            constraints["physical_row_exclusive"].append(
                model.addConstr(
                    quicksum(
                        atom_selected[atom.candidate_index]
                        for atom in resource_atoms
                    )
                    <= 1
                )
            )
        for bay_key, terms in sorted(reserved_by_physical.items()):
            constraints["reserved_physical_capacity"].append(
                model.addConstr(
                    quicksum(coefficient * variable for coefficient, variable in terms)
                    <= int(self.problem.bays[bay_key].physical_capacity)
                )
            )
        for (bay_key, size), terms in sorted(
            reserved_by_anchor_size.items()
        ):
            constraints["reserved_anchor_size_capacity"].append(
                model.addConstr(
                    quicksum(
                        coefficient * variable
                        for coefficient, variable in terms
                    )
                    <= int(self.problem.bays[bay_key].cap_by_size.get(size, 0))
                )
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
        for (group_id, bay_key), variable in group_bay_flow.items():
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
        export_size_state = {
            key: model.addVar(vtype="B", name=f"export_size_{index}")
            for index, key in enumerate(sorted(export_q_by_physical_size))
        }
        export_height_state = {
            key: model.addVar(vtype="B", name=f"export_height_{index}")
            for index, key in enumerate(sorted(export_q_by_physical_height))
        }
        for key, variables in sorted(export_q_by_physical_size.items()):
            constraints["export_size_state_link"].append(
                model.addConstr(
                    quicksum(variables)
                    <= total_export_demand * export_size_state[key]
                )
            )
        for bay_key in sorted({key[0] for key in export_size_state}):
            constraints["export_size_state_choice"].append(
                model.addConstr(
                    quicksum(
                        variable
                        for (physical, _size), variable in export_size_state.items()
                        if physical == bay_key
                    )
                    <= 1
                )
            )
        for key, variables in sorted(export_q_by_physical_height.items()):
            constraints["export_height_state_link"].append(
                model.addConstr(
                    quicksum(variables)
                    <= total_export_demand * export_height_state[key]
                )
            )
        for bay_key in sorted({key[0] for key in export_height_state}):
            constraints["export_height_state_choice"].append(
                model.addConstr(
                    quicksum(
                        variable
                        for (physical, _height), variable in export_height_state.items()
                        if physical == bay_key
                    )
                    <= 1
                )
            )

        import_rows = [
            (flow, size, bay_key, capacity)
            for (flow, size), candidates in sorted(
                self.evaluator.import_candidates.items()
            )
            for bay_key, capacity in candidates
        ]
        import_reservation = {
            (flow, size, bay_key): model.addVar(
                lb=0.0,
                ub=float(capacity),
                vtype="I",
                name=f"p_import_{index}",
            )
            for index, (flow, size, bay_key, capacity) in enumerate(import_rows)
        }
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
        for (flow, size, bay_key), variable in import_reservation.items():
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
            constraints["import_demand"].append(
                model.addConstr(
                    quicksum(import_by_flow_size.get(key, [])) == int(demand)
                )
            )
        for key, variables in sorted(import_by_anchor_size.items()):
            constraints["import_anchor_size_capacity"].append(
                model.addConstr(
                    quicksum(variables)
                    <= int(self.problem.bays[key[0]].cap_by_size.get(key[1], 0))
                )
            )

        allocation_bays = sorted(set(export_q_by_physical) | set(import_by_physical))
        export_use = {
            bay_key: model.addVar(vtype="B", name=f"export_use_{index}")
            for index, bay_key in enumerate(allocation_bays)
        }
        import_use = {
            bay_key: model.addVar(vtype="B", name=f"import_use_{index}")
            for index, bay_key in enumerate(allocation_bays)
        }
        import_size_state = {
            key: model.addVar(vtype="B", name=f"import_size_{index}")
            for index, key in enumerate(sorted(import_by_physical_size))
        }
        for bay_key in allocation_bays:
            export_load = quicksum(export_q_by_physical.get(bay_key, []))
            import_load = quicksum(import_by_physical.get(bay_key, []))
            if export_q_by_physical.get(bay_key):
                constraints["export_bay_use_link"].append(
                    model.addConstr(
                        export_load <= total_export_demand * export_use[bay_key]
                    )
                )
                constraints["export_bay_use_presence"].append(
                    model.addConstr(export_use[bay_key] <= export_load)
                )
            else:
                constraints["export_bay_use_zero"].append(
                    model.addConstr(export_use[bay_key] == 0)
                )
            if import_by_physical.get(bay_key):
                capacity = max(1, int(self.problem.bays[bay_key].physical_capacity))
                constraints["import_physical_capacity"].append(
                    model.addConstr(import_load <= capacity * import_use[bay_key])
                )
                constraints["import_bay_use_presence"].append(
                    model.addConstr(import_use[bay_key] <= import_load)
                )
                for key in sorted(
                    key for key in import_size_state if key[0] == bay_key
                ):
                    constraints["import_size_state_link"].append(
                        model.addConstr(
                            quicksum(import_by_physical_size[key])
                            <= capacity * import_size_state[key]
                        )
                    )
                    constraints["import_size_state_presence"].append(
                        model.addConstr(
                            import_size_state[key]
                            <= quicksum(import_by_physical_size[key])
                        )
                    )
                constraints["import_size_state_choice"].append(
                    model.addConstr(
                        quicksum(
                            variable
                            for (physical, _size), variable in import_size_state.items()
                            if physical == bay_key
                        )
                        <= import_use[bay_key]
                    )
                )
            else:
                constraints["import_bay_use_zero"].append(
                    model.addConstr(import_use[bay_key] == 0)
                )
            constraints["export_import_bay_exclusive"].append(
                model.addConstr(export_use[bay_key] + import_use[bay_key] <= 1)
            )

        voyage_area_use: dict[tuple[str, str], object] = {}
        if business:
            demand_by_voyage: Counter[str] = Counter()
            for group in self.groups:
                demand_by_voyage[group.voyage_id] += int(group.demand)
            voyage_area_use = {
                key: model.addVar(
                    vtype="B",
                    obj=self.evaluator.voyage_area_objective_coefficient(),
                    name=f"voyage_area_{index}",
                )
                for index, key in enumerate(sorted(export_q_by_voyage_area))
            }
            for key, variables in sorted(export_q_by_voyage_area.items()):
                assigned = quicksum(variables)
                constraints["voyage_area_link"].append(
                    model.addConstr(
                        assigned
                        <= int(demand_by_voyage[key[0]]) * voyage_area_use[key]
                    )
                )
                constraints["voyage_area_presence"].append(
                    model.addConstr(voyage_area_use[key] <= assigned)
                )
            model.addVar(
                lb=1.0,
                ub=1.0,
                obj=self.evaluator.objective_constant(),
                name="business_objective_constant",
            )

        peak_utilization = (
            model.addVar(
                lb=0.0,
                ub=1.0,
                obj=1.0,
                name="rho_peak",
            )
            if minmax
            else None
        )
        area_capacity: Counter[str] = Counter()
        for bay in self.problem.bays.values():
            area_capacity[str(bay.area_no)] += int(bay.physical_capacity)
        for area_no, terms in sorted(planned_load_by_area.items()):
            capacity = int(area_capacity[area_no])
            if capacity <= 0:
                raise ValueError(f"V6 used area has no capacity: {area_no}")
            constraints["peak_utilization"].append(
                model.addConstr(
                    quicksum(coefficient * variable for coefficient, variable in terms)
                    <= capacity
                    * (
                        peak_utilization
                        if minmax
                        else float(self.peak_policy.epsilon_cap)
                    )
                )
            )

        model.update()
        warm_start_diagnostics = self._apply_business_warm_start(
            atom_selected=atom_selected,
            bay_used=bay_used,
            group_bay_flow=group_bay_flow,
            run_start=run_start,
            export_size_state=export_size_state,
            export_height_state=export_height_state,
            import_reservation=import_reservation,
            export_use=export_use,
            import_use=import_use,
            import_size_state=import_size_state,
            voyage_area_use=voyage_area_use,
        )
        fingerprint = model.getFingerprint()
        recorder = MipProgressRecorder(
            phase=f"v6_compact_{self.objective_mode}"
        )
        try:
            model.optimize(callback=recorder)
            progress = recorder.finalize(model)
            status = model.getStatusName()
            incomplete_diagnostics = {
                "algorithm": "v6_compact_row_atom_primal_coverage",
                "objective_mode": self.objective_mode,
                "model_schema_version": V6_MODEL_SCHEMA_VERSION,
                "complete_zone_enumeration_used": False,
                "status": status,
                "proven_optimal": status == "optimal",
                "solution_count": model.getSolutionCount(),
                "solver_objective": (
                    model.getObjectiveValue()
                    if model.getSolutionCount() > 0
                    else None
                ),
                "solver_bound": model.getBestBound(),
                "solver_gap": (
                    model.getMipGap() if model.getSolutionCount() > 0 else None
                ),
                "runtime_seconds": model.getRuntime(),
                "atom_count": len(self.atoms),
                "fingerprint": fingerprint,
                "constraint_count_by_family": {
                    family: len(rows)
                    for family, rows in sorted(constraints.items())
                },
                "progress": progress,
                "total_seconds": perf_counter() - total_started,
            }
            if model.getSolutionCount() <= 0:
                raise V6CompactSolveIncompleteError(
                    "V6 compact primal coverage found no feasible incumbent: "
                    f"status={status}",
                    incomplete_diagnostics,
                )
            if minmax and status != "optimal":
                raise V6CompactSolveIncompleteError(
                    "V6 compact rho* must be proven optimal before the "
                    f"business pipeline can run: status={status}",
                    incomplete_diagnostics,
                )
            pool_solution_count = min(
                model.getSolutionCount(),
                int(self.config.maximum_pool_solutions),
            )
            candidate_zones: list[RowAwareZone] = []
            candidate_signatures: set[tuple[int, ...]] = set()
            pool_rows: list[dict[str, object]] = []
            best_payload = None
            for solution_number in range(pool_solution_count):
                selected_indices = tuple(
                    sorted(
                        index
                        for index, variable in atom_selected.items()
                        if model.getPoolValue(variable, solution_number) > 0.5
                    )
                )
                solution_flow = {
                    key: int(round(model.getPoolValue(variable, solution_number)))
                    for key, variable in group_bay_flow.items()
                    if model.getPoolValue(variable, solution_number) > 0.5
                }
                solution_imports = {
                    key: int(round(model.getPoolValue(variable, solution_number)))
                    for key, variable in import_reservation.items()
                    if model.getPoolValue(variable, solution_number) > 0.5
                }
                solution_zones, solution_zone_flow = self._recover_zones(
                    selected_indices,
                    solution_flow,
                )
                added = 0
                for zone in solution_zones:
                    signature = tuple(zone.candidate_indices)
                    if signature in candidate_signatures:
                        continue
                    candidate_signatures.add(signature)
                    candidate_zones.append(zone)
                    added += 1
                pool_rows.append(
                    {
                        "solution_number": solution_number,
                        "objective": model.getPoolObjective(solution_number),
                        "selected_atom_count": len(selected_indices),
                        "recovered_zone_count": len(solution_zones),
                        "new_candidate_zone_count": added,
                    }
                )
                if solution_number == 0:
                    best_payload = (
                        selected_indices,
                        solution_imports,
                        solution_zones,
                        solution_zone_flow,
                    )
            if best_payload is None:
                raise RuntimeError("V6 primal coverage solution pool is empty")
            (
                selected_atom_indices,
                raw_imports,
                zones,
                zone_bay_flow,
            ) = best_payload
            selected_zone_ids = tuple(zone.zone_id for zone in zones)
            certificate_evaluator = V6ModelEvaluator(
                self.problem,
                zones,
                self.config.objective,
                candidate_bays_by_group=self.candidate_bays,
            )
            if not minmax:
                effective_peak_policy = self.peak_policy
            else:
                raw_minimum = model.getPoolValue(peak_utilization, 0)
                if raw_minimum < -1e-8 or raw_minimum > 1.0 + 1e-8:
                    raise RuntimeError(
                        "V6 compact rho* lies outside [0, 1]: "
                        f"{raw_minimum}"
                    )
                effective_peak_policy = V6PeakUtilizationPolicy(
                    minimum_feasible_utilization=min(
                        1.0, max(0.0, float(raw_minimum))
                    ),
                    headroom_fraction=float(
                        self.config.objective.peak_utilization_headroom_fraction
                    ),
                    minimum_source="compact_full_v6_row_atom_minmax_mip",
                )
            certificate = certificate_evaluator.evaluate(
                selected_zone_ids,
                zone_bay_flow,
                raw_imports,
                effective_peak_policy,
            )
            solver_objective = model.getObjectiveValue()
            reconstructed = float(certificate["objective"])
            if business:
                if not math.isclose(
                    solver_objective,
                    reconstructed,
                    rel_tol=1e-8,
                    abs_tol=1e-8,
                ):
                    raise RuntimeError(
                        "V6 compact primal objective differs from independent "
                        f"reconstruction: solver={solver_objective}, "
                        f"evaluator={reconstructed}"
                    )
            elif minmax:
                actual_peak = float(certificate["peak_utilization"]["maximum"])
                if not math.isclose(
                    solver_objective,
                    actual_peak,
                    rel_tol=1e-8,
                    abs_tol=1e-8,
                ):
                    raise RuntimeError(
                        "V6 compact rho* differs from independently reconstructed "
                        f"peak: solver={solver_objective}, actual={actual_peak}"
                    )
            diagnostics = {
                "algorithm": "v6_compact_row_atom_primal_coverage",
                "objective_mode": self.objective_mode,
                "model_schema_version": V6_MODEL_SCHEMA_VERSION,
                "complete_zone_enumeration_used": False,
                "status": status,
                "proven_optimal": status == "optimal",
                "solver_objective": solver_objective,
                "solver_bound": model.getBestBound(),
                "solver_gap": model.getMipGap(),
                "runtime_seconds": model.getRuntime(),
                "atom_count": len(self.atoms),
                "selected_atom_count": len(selected_atom_indices),
                "recovered_zone_count": len(zones),
                "pool_solution_count": pool_solution_count,
                "coverage_candidate_zone_count": len(candidate_zones),
                "pool_solutions": pool_rows,
                "fingerprint": fingerprint,
                "constraint_count_by_family": {
                    family: len(rows)
                    for family, rows in sorted(constraints.items())
                },
                "solver_evaluator_objective_difference": (
                    solver_objective - reconstructed
                    if business
                    else (
                        solver_objective
                        - float(certificate["peak_utilization"]["maximum"])
                        if minmax
                        else None
                    )
                ),
                "independent_validation_passed": bool(
                    certificate["validation"]["passed"]
                ),
                "progress": progress,
                "warm_start": warm_start_diagnostics,
                "total_seconds": perf_counter() - total_started,
            }
            return V6PrimalCoverageResult(
                selected_atom_indices=selected_atom_indices,
                selected_zone_ids=selected_zone_ids,
                zones=zones,
                candidate_zones=tuple(candidate_zones),
                zone_bay_flow=zone_bay_flow,
                import_reservation=raw_imports,
                peak_policy=effective_peak_policy,
                certificate=certificate,
                diagnostics=diagnostics,
            )
        finally:
            model.dispose()


class V6CompactPeakUtilizationSolver:
    """Prove V6 rho* on the compact row-atom formulation."""

    def __init__(
        self,
        problem: ProblemData,
        config: V6CompactPeakConfig | None = None,
    ) -> None:
        self.problem = problem
        self.config = config or V6CompactPeakConfig()
        self.config.validate()

    def solve(self) -> V6CompactPeakResult:
        internal = V6CompactPrimalCoverageSolver(
            self.problem,
            None,
            V6PrimalCoverageConfig(
                time_limit=self.config.time_limit,
                mip_gap=0.0,
                solver_threads=self.config.solver_threads,
                solver_seed=self.config.solver_seed,
                verbose=self.config.verbose,
                maximum_pool_solutions=1,
                objective=self.config.objective,
            ),
            _objective_mode="peak_minmax",
        ).solve()
        diagnostics = {
            **dict(internal.diagnostics),
            "algorithm": "v6_compact_row_atom_peak_minmax",
            "minimum_feasible_utilization": float(
                internal.peak_policy.minimum_feasible_utilization
            ),
            "epsilon_cap": float(internal.peak_policy.epsilon_cap),
            "minimum_source": "compact_full_v6_row_atom_minmax_mip",
            "proven_optimal": True,
        }
        return V6CompactPeakResult(
            peak_policy=internal.peak_policy,
            selected_atom_indices=internal.selected_atom_indices,
            zones=internal.zones,
            zone_bay_flow=internal.zone_bay_flow,
            import_reservation=internal.import_reservation,
            certificate=internal.certificate,
            diagnostics=diagnostics,
        )


class V6AnalyticPeakPreparationSolver:
    """Build a V6-native analytic cap and certify one feasible witness."""

    def __init__(
        self,
        problem: ProblemData,
        config: V6AnalyticPeakConfig | None = None,
    ) -> None:
        self.problem = problem
        self.config = config or V6AnalyticPeakConfig()
        self.config.validate()

    def solve(self) -> V6AnalyticPeakResult:
        started = perf_counter()
        atoms, _anchor_limits = build_v6_row_aware_bay_atoms(self.problem)
        policy, analytic = derive_v6_analytic_peak_policy(
            self.problem,
            self.config.objective,
            atoms=atoms,
        )
        witness = V6CompactPrimalCoverageSolver(
            self.problem,
            policy,
            V6PrimalCoverageConfig(
                time_limit=self.config.feasibility_time_limit,
                mip_gap=0.0,
                solver_threads=self.config.solver_threads,
                solver_seed=self.config.solver_seed,
                verbose=self.config.verbose,
                maximum_pool_solutions=1,
                objective=self.config.objective,
            ),
            _objective_mode="peak_feasibility",
        ).solve()
        actual_peak = float(
            witness.certificate["peak_utilization"]["maximum"]
        )
        if actual_peak > policy.epsilon_cap + 1e-9:
            raise RuntimeError(
                "V6 analytic peak feasibility witness exceeds its cap: "
                f"actual={actual_peak}, cap={policy.epsilon_cap}"
            )
        diagnostics = {
            "algorithm": "v6_analytic_peak_cap_with_feasibility_certificate",
            "model_schema_version": V6_MODEL_SCHEMA_VERSION,
            "minmax_mip_solved": False,
            "rho_star_claimed": False,
            "analytic_policy": analytic,
            "peak_policy": policy.as_dict(),
            "feasibility_certificate": dict(witness.diagnostics),
            "witness_peak_utilization": actual_peak,
            "witness_slack": float(policy.epsilon_cap - actual_peak),
            "feasibility_certified": True,
            "total_seconds": perf_counter() - started,
        }
        return V6AnalyticPeakResult(
            peak_policy=policy,
            witness=witness,
            diagnostics=diagnostics,
        )


class V6PrimalCoveredPipeline:
    """Combine exact root proof columns with compact-primal coverage columns."""

    def __init__(
        self,
        problem: ProblemData,
        peak_policy: V6PeakUtilizationPolicy,
        root_config: V6RootCgConfig | None = None,
        coverage_config: V6PrimalCoverageConfig | None = None,
        integer_config: V6RestrictedIntegerConfig | None = None,
        coverage_warm_start: V6PrimalCoverageResult | None = None,
    ) -> None:
        self.problem = problem
        self.peak_policy = peak_policy
        self.root_config = root_config or V6RootCgConfig()
        self.coverage_config = coverage_config or V6PrimalCoverageConfig(
            objective=self.root_config.objective
        )
        self.integer_config = integer_config or V6RestrictedIntegerConfig(
            objective=self.root_config.objective
        )
        self.coverage_warm_start = coverage_warm_start
        self.root_config.validate()
        self.coverage_config.validate()
        self.integer_config.validate()
        if not (
            self.root_config.objective
            == self.coverage_config.objective
            == self.integer_config.objective
        ):
            raise ValueError(
                "V6 root, primal coverage, and integer master must share one "
                "objective configuration"
            )

    def solve(self) -> V6PrimalCoveredPipelineResult:
        started = perf_counter()
        root = V6RootColumnGeneration(
            self.problem,
            self.peak_policy,
            self.root_config,
        ).solve(
            initial_zones=(
                self.coverage_warm_start.zones
                if self.coverage_warm_start is not None
                else ()
            )
        )
        coverage = V6CompactPrimalCoverageSolver(
            self.problem,
            self.peak_policy,
            self.coverage_config,
            warm_start=self.coverage_warm_start,
        ).solve()
        merged: list[RowAwareZone] = []
        signatures: set[tuple[int, ...]] = set()
        for zone in (*root.zones, *coverage.candidate_zones):
            signature = tuple(zone.candidate_indices)
            if signature in signatures:
                continue
            signatures.add(signature)
            merged.append(zone)
        integer = V6RestrictedIntegerSolver(
            self.problem,
            merged,
            self.peak_policy,
            self.integer_config,
            warm_start=V6RestrictedIntegerWarmStart.from_zone_solution(
                coverage.zones,
                coverage.selected_zone_ids,
                coverage.zone_bay_flow,
                coverage.import_reservation,
                coverage.objective,
            ),
        ).solve()
        if integer.objective > coverage.objective + 1e-8:
            raise RuntimeError(
                "V6 merged integer pool lost its compact primal incumbent: "
                f"merged={integer.objective}, coverage={coverage.objective}"
            )
        if integer.objective + 1e-8 < root.objective:
            raise RuntimeError(
                "V6 covered integer incumbent is below the exact root bound: "
                f"incumbent={integer.objective}, root={root.objective}"
            )
        absolute_gap = max(0.0, integer.objective - root.objective)
        diagnostics = {
            "algorithm": "v6_exact_root_plus_compact_primal_coverage",
            "model_schema_version": V6_MODEL_SCHEMA_VERSION,
            "complete_zone_enumeration_used": False,
            "root_lp_lower_bound": root.objective,
            "compact_primal_objective": coverage.objective,
            "covered_integer_upper_bound": integer.objective,
            "absolute_root_gap": absolute_gap,
            "relative_root_gap": absolute_gap
            / max(1.0, abs(integer.objective)),
            "root_zone_count": len(root.zones),
            "coverage_incumbent_zone_count": len(coverage.zones),
            "coverage_candidate_zone_count": len(coverage.candidate_zones),
            "coverage_added_zone_count": len(merged) - len(root.zones),
            "merged_zone_count": len(merged),
            "root_bound_preserved": True,
            "integer_validation_passed": True,
            "total_seconds": perf_counter() - started,
        }
        return V6PrimalCoveredPipelineResult(
            root=root,
            coverage=coverage,
            integer=integer,
            diagnostics=diagnostics,
        )


class V6ProductionPipeline:
    """Run analytic peak preparation, exact root CG, primal coverage, and RIM."""

    def __init__(
        self,
        problem: ProblemData,
        peak_config: V6AnalyticPeakConfig | None = None,
        root_config: V6RootCgConfig | None = None,
        coverage_config: V6PrimalCoverageConfig | None = None,
        integer_config: V6RestrictedIntegerConfig | None = None,
    ) -> None:
        self.problem = problem
        self.peak_config = peak_config or V6AnalyticPeakConfig()
        self.root_config = root_config or V6RootCgConfig(
            objective=self.peak_config.objective
        )
        self.coverage_config = coverage_config or V6PrimalCoverageConfig(
            objective=self.peak_config.objective
        )
        self.integer_config = integer_config or V6RestrictedIntegerConfig(
            objective=self.peak_config.objective
        )
        self.peak_config.validate()
        self.root_config.validate()
        self.coverage_config.validate()
        self.integer_config.validate()
        if not (
            self.peak_config.objective
            == self.root_config.objective
            == self.coverage_config.objective
            == self.integer_config.objective
        ):
            raise ValueError(
                "all V6 production-pipeline phases must share one objective "
                "and peak-headroom configuration"
            )

    def solve(self) -> V6ProductionPipelineResult:
        started = perf_counter()
        peak = V6AnalyticPeakPreparationSolver(
            self.problem,
            self.peak_config,
        ).solve()
        planning = V6PrimalCoveredPipeline(
            self.problem,
            peak.peak_policy,
            self.root_config,
            self.coverage_config,
            self.integer_config,
            coverage_warm_start=peak.witness,
        ).solve()
        diagnostics = {
            "algorithm": "v6_analytic_peak_exact_root_compact_primal_rim",
            "model_schema_version": V6_MODEL_SCHEMA_VERSION,
            "complete_zone_enumeration_used": False,
            "analytic_load_lower_bound": (
                peak.peak_policy.minimum_feasible_utilization
            ),
            "epsilon_cap": peak.peak_policy.epsilon_cap,
            "root_lp_lower_bound": planning.root.objective,
            "integer_upper_bound": planning.integer.objective,
            "relative_root_gap": planning.diagnostics["relative_root_gap"],
            "peak_preparation_seconds": peak.diagnostics["total_seconds"],
            "rho_star_proof_required": False,
            "planning_seconds": planning.diagnostics["total_seconds"],
            "total_seconds": perf_counter() - started,
        }
        return V6ProductionPipelineResult(
            peak=peak,
            planning=planning,
            diagnostics=diagnostics,
        )


__all__ = [
    "V6AnalyticPeakConfig",
    "V6AnalyticPeakPreparationSolver",
    "V6AnalyticPeakResult",
    "V6CompactSolveIncompleteError",
    "V6CompactPeakConfig",
    "V6CompactPeakResult",
    "V6CompactPeakUtilizationSolver",
    "V6CompactPrimalCoverageSolver",
    "V6PrimalCoverageConfig",
    "V6PrimalCoverageResult",
    "V6PrimalCoveredPipeline",
    "V6PrimalCoveredPipelineResult",
    "V6ProductionPipeline",
    "V6ProductionPipelineResult",
]
