"""V7 bay-capacity patterns and exact support-enumeration pricing."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import combinations, product
from time import perf_counter
from typing import Callable, Iterable, Mapping, Sequence

from .gurobi_backend import GurobiModel
from .models import ProblemData
from .v7_atoms import V7RowAtom


class V7PatternEnumerationIncompleteError(RuntimeError):
    """An explicit safety limit stopped an otherwise exact enumeration."""


class V7PricingIncompleteError(RuntimeError):
    """An exact pricing subproblem did not finish with a proof."""


@dataclass(frozen=True)
class V7BayPattern:
    pattern_id: int
    anchor_bay_key: str
    group_capacities: tuple[tuple[str, int], ...]
    group_rows: tuple[tuple[str, tuple[str, ...]], ...]
    physical_resources: tuple[tuple[str, str], ...]
    physical_bays: tuple[str, ...]
    size_mode: str
    height_mode: str
    active_groups: tuple[str, ...]
    candidate_indices: tuple[int, ...]

    @property
    def signature(self) -> tuple[int, ...]:
        return tuple(self.candidate_indices)

    @property
    def master_column_signature(self) -> tuple[object, ...]:
        """Return the coefficient-equivalence class used by the master.

        ``group_rows`` and atom indices are recovery details.  Two patterns
        with the same values below produce identical coefficients in every
        master row and can therefore share one deterministic representative.
        """

        return (
            self.anchor_bay_key,
            self.size_mode,
            self.height_mode,
            self.group_capacities,
            self.physical_resources,
            self.physical_bays,
        )

    def capacity_for(self, group_id: str) -> int:
        return int(dict(self.group_capacities).get(str(group_id), 0))


@dataclass(frozen=True)
class V7PricingResult:
    anchor_bay_key: str
    minimum_reduced_cost: float | None
    minimum_pattern: V7BayPattern | None
    returned_patterns: tuple[V7BayPattern, ...]
    diagnostics: Mapping[str, object]


@dataclass
class _V7PersistentPricingState:
    """One structurally fixed pricing MIP reused across CG iterations."""

    model: GurobiModel
    atom_selected: Mapping[int, object]
    group_used: Mapping[str, object]
    bay_constant: object
    variable_count: int
    constraint_count: int


def _pattern_from_atoms(
    problem: ProblemData,
    atoms: Sequence[V7RowAtom],
    *,
    pattern_id: int,
    max_new_groups_per_physical_bay: int = 3,
) -> V7BayPattern | None:
    selected = tuple(sorted(atoms, key=lambda atom: atom.candidate_index))
    if not selected:
        return None
    anchors = {atom.anchor_bay_key for atom in selected}
    sizes = {atom.size for atom in selected}
    heights = {atom.height for atom in selected}
    groups = {atom.group_id for atom in selected}
    if len(anchors) != 1 or len(sizes) != 1 or len(heights) != 1:
        return None
    if len(groups) > int(max_new_groups_per_physical_bay):
        return None
    resources = [resource for atom in selected for resource in atom.physical_resources]
    if len(resources) != len(set(resources)):
        return None
    physical_bays = tuple(sorted({bay_key for bay_key, _row in resources}))
    capacity_by_physical: Counter[str] = Counter()
    capacity_by_group: Counter[str] = Counter()
    rows_by_group: defaultdict[str, set[str]] = defaultdict(set)
    anchor_size_capacity = 0
    for atom in selected:
        capacity_by_group[atom.group_id] += int(atom.capacity)
        rows_by_group[atom.group_id].add(atom.row_no)
        anchor_size_capacity += int(atom.capacity)
        for physical in atom.physical_bays:
            capacity_by_physical[physical] += int(atom.capacity)
    if any(
        capacity > int(problem.bays[physical].physical_capacity)
        for physical, capacity in capacity_by_physical.items()
    ):
        return None
    anchor = next(iter(anchors))
    size = next(iter(sizes))
    if anchor_size_capacity > int(problem.bays[anchor].cap_by_size.get(size, 0)):
        return None
    if any(len(groups) > max_new_groups_per_physical_bay for _physical in physical_bays):
        return None
    return V7BayPattern(
        pattern_id=int(pattern_id),
        anchor_bay_key=str(anchor),
        group_capacities=tuple(sorted(capacity_by_group.items())),
        group_rows=tuple(
            (group_id, tuple(sorted(rows)))
            for group_id, rows in sorted(rows_by_group.items())
        ),
        physical_resources=tuple(sorted(resources)),
        physical_bays=physical_bays,
        size_mode=str(size),
        height_mode=str(next(iter(heights))),
        active_groups=tuple(sorted(groups)),
        candidate_indices=tuple(atom.candidate_index for atom in selected),
    )


def build_v7_pattern_from_atom_indices(
    problem: ProblemData,
    atoms_by_index: Mapping[int, V7RowAtom],
    candidate_indices: Iterable[int],
    *,
    pattern_id: int,
) -> V7BayPattern:
    selected = [atoms_by_index[int(index)] for index in candidate_indices]
    pattern = _pattern_from_atoms(problem, selected, pattern_id=pattern_id)
    if pattern is None:
        raise ValueError("selected V7 row atoms do not form one legal bay pattern")
    return pattern


def enumerate_v7_bay_patterns(
    problem: ProblemData,
    atoms: Sequence[V7RowAtom],
    anchor_bay_key: str,
    *,
    allowed_groups: Iterable[str] | None = None,
    start_pattern_id: int = 0,
    maximum_patterns: int | None = None,
) -> tuple[V7BayPattern, ...]:
    """Exact 1/2/3-support row-partition enumeration for one anchor bay."""

    allowed = None if allowed_groups is None else {str(value) for value in allowed_groups}
    local = tuple(
        atom
        for atom in atoms
        if atom.anchor_bay_key == str(anchor_bay_key)
        and (allowed is None or atom.group_id in allowed)
    )
    by_state: defaultdict[tuple[str, str], list[V7RowAtom]] = defaultdict(list)
    for atom in local:
        by_state[(atom.size, atom.height)].append(atom)
    signatures: set[tuple[int, ...]] = set()
    patterns: list[V7BayPattern] = []
    support_evaluated = 0
    for _state, state_atoms in sorted(by_state.items()):
        groups = sorted({atom.group_id for atom in state_atoms})
        atoms_by_group_row = {
            (atom.group_id, atom.row_no): atom for atom in state_atoms
        }
        rows = sorted({atom.row_no for atom in state_atoms})
        for support_size in range(1, min(3, len(groups)) + 1):
            for support in combinations(groups, support_size):
                support_evaluated += 1
                choices_by_row = []
                for row_no in rows:
                    choices = [None]
                    choices.extend(
                        atoms_by_group_row[(group_id, row_no)]
                        for group_id in support
                        if (group_id, row_no) in atoms_by_group_row
                    )
                    choices_by_row.append(tuple(choices))
                for assignment in product(*choices_by_row):
                    selected = tuple(atom for atom in assignment if atom is not None)
                    if {atom.group_id for atom in selected} != set(support):
                        continue
                    signature = tuple(sorted(atom.candidate_index for atom in selected))
                    if signature in signatures:
                        continue
                    pattern = _pattern_from_atoms(
                        problem,
                        selected,
                        pattern_id=start_pattern_id + len(patterns),
                    )
                    if pattern is None:
                        continue
                    signatures.add(signature)
                    patterns.append(pattern)
                    if maximum_patterns is not None and len(patterns) > int(maximum_patterns):
                        raise V7PatternEnumerationIncompleteError(
                            "V7 exact pattern enumeration exceeded its explicit "
                            f"limit ({maximum_patterns}); no truncated pool was returned"
                        )
    return tuple(patterns)


def exhaustive_v7_bay_patterns(
    problem: ProblemData,
    atoms: Sequence[V7RowAtom],
    anchor_bay_key: str,
    *,
    allowed_groups: Iterable[str] | None = None,
    start_pattern_id: int = 0,
    maximum_patterns: int = 100_000,
) -> tuple[V7BayPattern, ...]:
    """Independent brute-force subset oracle for tiny correctness instances."""

    allowed = None if allowed_groups is None else {str(value) for value in allowed_groups}
    local = tuple(
        atom
        for atom in atoms
        if atom.anchor_bay_key == str(anchor_bay_key)
        and (allowed is None or atom.group_id in allowed)
    )
    patterns: list[V7BayPattern] = []
    for count in range(1, len(local) + 1):
        for subset in combinations(local, count):
            pattern = _pattern_from_atoms(
                problem,
                subset,
                pattern_id=start_pattern_id + len(patterns),
            )
            if pattern is None:
                continue
            patterns.append(pattern)
            if len(patterns) > int(maximum_patterns):
                raise V7PatternEnumerationIncompleteError(
                    "V7 exhaustive micro oracle exceeded its safety limit"
                )
    unique: dict[tuple[int, ...], V7BayPattern] = {}
    for pattern in patterns:
        unique.setdefault(pattern.signature, pattern)
    return tuple(
        replace(pattern, pattern_id=start_pattern_id + index)
        for index, pattern in enumerate(
            sorted(unique.values(), key=lambda value: value.signature)
        )
    )


class V7ExactBayPricing:
    """Exact support-size 1/2/3 pricing with multi-column output."""

    def __init__(
        self,
        problem: ProblemData,
        atoms: Sequence[V7RowAtom],
        *,
        columns_per_bay: int = 3,
        reduced_cost_tolerance: float = 1e-8,
    ) -> None:
        if int(columns_per_bay) <= 0:
            raise ValueError("V7 pricing columns_per_bay must be positive")
        self.problem = problem
        self.atoms = tuple(atoms)
        self.columns_per_bay = int(columns_per_bay)
        self.tolerance = float(reduced_cost_tolerance)
        atoms_by_anchor: defaultdict[str, list[V7RowAtom]] = defaultdict(list)
        for atom in self.atoms:
            atoms_by_anchor[str(atom.anchor_bay_key)].append(atom)
        self.atoms_by_anchor = {
            anchor: tuple(values)
            for anchor, values in sorted(atoms_by_anchor.items())
        }
        self.atoms_by_index = {
            int(atom.candidate_index): atom for atom in self.atoms
        }
        self._persistent_mip_states: dict[
            tuple[object, ...], _V7PersistentPricingState
        ] = {}

    def dispose(self) -> None:
        """Release persistent native pricing models."""

        for state in self._persistent_mip_states.values():
            state.model.dispose()
        self._persistent_mip_states.clear()

    def price_bay(
        self,
        anchor_bay_key: str,
        reduced_cost: Callable[[V7BayPattern], float],
        *,
        allowed_groups: Iterable[str] | None = None,
        excluded_signatures: Iterable[tuple[int, ...]] = (),
        exhaustive: bool = False,
    ) -> V7PricingResult:
        started = perf_counter()
        excluded = {tuple(value) for value in excluded_signatures}
        allowed = (
            None
            if allowed_groups is None
            else {str(value) for value in allowed_groups}
        )
        anchor_atoms = self.atoms_by_anchor.get(str(anchor_bay_key), ())
        local_groups = {
            atom.group_id
            for atom in anchor_atoms
            if allowed is None or atom.group_id in allowed
        }
        generator = exhaustive_v7_bay_patterns if exhaustive else enumerate_v7_bay_patterns
        patterns = generator(
            self.problem,
            anchor_atoms,
            str(anchor_bay_key),
            allowed_groups=allowed,
        )
        priced = [
            (float(reduced_cost(pattern)), pattern)
            for pattern in patterns
            if pattern.signature not in excluded
        ]
        priced.sort(key=lambda item: (item[0], item[1].signature))
        negative = [
            pattern
            for value, pattern in priced
            if value < -self.tolerance
        ][: self.columns_per_bay]
        minimum_value = priced[0][0] if priced else None
        minimum_pattern = priced[0][1] if priced else None
        support_counts = Counter(len(pattern.active_groups) for pattern in patterns)
        return V7PricingResult(
            anchor_bay_key=str(anchor_bay_key),
            minimum_reduced_cost=minimum_value,
            minimum_pattern=minimum_pattern,
            returned_patterns=tuple(negative),
            diagnostics={
                "candidate_group_count": len(local_groups),
                "support_count_total": sum(support_counts.values()),
                "support_count_evaluated": sum(support_counts.values()),
                "support_count_pruned": 0,
                "pattern_count": len(patterns),
                "pricing_seconds": perf_counter() - started,
                "min_reduced_cost": minimum_value,
                "returned_negative_columns": len(negative),
                "exact": True,
                "exhaustive_micro_oracle": bool(exhaustive),
            },
        )

    def price_bay_additive_dp(
        self,
        anchor_bay_key: str,
        atom_reduced_cost: Callable[[V7RowAtom], float],
        support_reduced_cost: Callable[
            [str, str, str, tuple[str, ...], tuple[str, ...]], float
        ],
        *,
        verify_reduced_cost: Callable[[V7BayPattern], float] | None = None,
        allowed_groups: Iterable[str] | None = None,
        excluded_signatures: Iterable[tuple[int, ...]] = (),
    ) -> V7PricingResult:
        """Price one bay exactly using one row DP over supports of size <= 3.

        Every pattern reduced cost is a support constant plus independent
        row-atom terms.  The dynamic program builds all legal supports at the
        same time and keeps the best assignments by
        ``(actual-support, reserved-capacity)``.  Its retention count includes
        every already-generated local signature, so excluding RMP columns
        cannot hide the next top-K improving pattern.
        """

        started = perf_counter()
        anchor = str(anchor_bay_key)
        allowed = (
            None
            if allowed_groups is None
            else {str(value) for value in allowed_groups}
        )
        excluded = {tuple(value) for value in excluded_signatures}
        local = tuple(
            atom
            for atom in self.atoms_by_anchor.get(anchor, ())
            if allowed is None or atom.group_id in allowed
        )
        local_indices = {atom.candidate_index for atom in local}
        local_excluded_count = sum(
            bool(set(signature)) and set(signature) <= local_indices
            for signature in excluded
        )
        retention = self.columns_per_bay + local_excluded_count + 1
        atoms_by_index = {atom.candidate_index: atom for atom in local}
        by_state: defaultdict[tuple[str, str], list[V7RowAtom]] = defaultdict(list)
        for atom in local:
            by_state[(atom.size, atom.height)].append(atom)

        priced: list[tuple[float, V7BayPattern]] = []
        terminal: list[
            tuple[
                float,
                tuple[int, ...],
                str,
                str,
                tuple[str, ...],
                tuple[str, ...],
            ]
        ] = []
        support_sets: set[tuple[str, str, tuple[str, ...]]] = set()
        transition_count = 0
        pruned_assignment_count = 0
        for (size, height), state_atoms in sorted(by_state.items()):
            groups = sorted({atom.group_id for atom in state_atoms})
            rows = sorted({atom.row_no for atom in state_atoms})
            atom_by_group_row = {
                (atom.group_id, atom.row_no): atom for atom in state_atoms
            }
            physical_bays = tuple(sorted(state_atoms[0].physical_bays))
            if any(tuple(sorted(atom.physical_bays)) != physical_bays for atom in state_atoms):
                raise RuntimeError("V7 pricing state contains inconsistent footprints")
            capacity_limit = min(
                int(self.problem.bays[anchor].cap_by_size.get(size, 0)),
                *(int(self.problem.bays[physical].physical_capacity) for physical in physical_bays),
            )
            # (support, capacity) -> sorted (row-cost, signature) alternatives.
            states: dict[
                tuple[tuple[str, ...], int],
                list[tuple[float, tuple[int, ...]]],
            ] = {((), 0): [(0.0, ())]}
            for row_no in rows:
                row_atoms = [
                    atom_by_group_row[(group_id, row_no)]
                    for group_id in groups
                    if (group_id, row_no) in atom_by_group_row
                ]
                next_states: defaultdict[
                    tuple[tuple[str, ...], int],
                    list[tuple[float, tuple[int, ...]]],
                ] = defaultdict(list)
                for (support, capacity), alternatives in states.items():
                    if len(support) >= 3:
                        choices = [
                            atom for atom in row_atoms if atom.group_id in support
                        ]
                    else:
                        choices = row_atoms
                    for cost, signature in alternatives:
                        transition_count += 1
                        next_states[(support, capacity)].append((cost, signature))
                        for atom in choices:
                            transition_count += 1
                            new_capacity = capacity + int(atom.capacity)
                            if new_capacity > capacity_limit:
                                continue
                            new_support = (
                                support
                                if atom.group_id in support
                                else tuple(sorted((*support, atom.group_id)))
                            )
                            next_states[(new_support, new_capacity)].append(
                                (
                                    cost + float(atom_reduced_cost(atom)),
                                    (*signature, atom.candidate_index),
                                )
                            )
                states = {}
                for key, alternatives in next_states.items():
                    unique: dict[tuple[int, ...], float] = {}
                    for cost, signature in alternatives:
                        normalized = tuple(sorted(signature))
                        previous = unique.get(normalized)
                        if previous is None or cost < previous:
                            unique[normalized] = cost
                    ranked = sorted(
                        ((cost, signature) for signature, cost in unique.items()),
                        key=lambda item: (item[0], item[1]),
                    )
                    pruned_assignment_count += max(0, len(ranked) - retention)
                    states[key] = ranked[:retention]

            support_constants: dict[tuple[str, ...], float] = {}
            for (support, _capacity), alternatives in states.items():
                if not support:
                    continue
                support_sets.add((size, height, support))
                constant = support_constants.setdefault(
                    support,
                    float(
                        support_reduced_cost(
                            anchor,
                            size,
                            height,
                            support,
                            physical_bays,
                        )
                    ),
                )
                terminal.extend(
                    (
                        constant + cost,
                        signature,
                        size,
                        height,
                        support,
                        physical_bays,
                    )
                    for cost, signature in alternatives
                )

        terminal.sort(key=lambda item: (item[0], item[1]))
        for dp_value, signature, _size, _height, _support, _physical in terminal[:retention]:
            if signature in excluded:
                continue
            pattern = build_v7_pattern_from_atom_indices(
                self.problem,
                atoms_by_index,
                signature,
                pattern_id=0,
            )
            value = (
                float(verify_reduced_cost(pattern))
                if verify_reduced_cost is not None
                else float(dp_value)
            )
            if verify_reduced_cost is not None and abs(value - dp_value) > 1e-8:
                raise RuntimeError(
                    "V7 additive pricing decomposition disagrees with "
                    f"the master reduced cost: dp={dp_value}, master={value}"
                )
            priced.append((value, pattern))

        unique_priced: dict[tuple[int, ...], tuple[float, V7BayPattern]] = {}
        for value, pattern in priced:
            previous = unique_priced.get(pattern.signature)
            if previous is None or value < previous[0]:
                unique_priced[pattern.signature] = (value, pattern)
        ranked_patterns = sorted(
            unique_priced.values(), key=lambda item: (item[0], item[1].signature)
        )
        negative = tuple(
            pattern
            for value, pattern in ranked_patterns
            if value < -self.tolerance
        )[: self.columns_per_bay]
        minimum_value = ranked_patterns[0][0] if ranked_patterns else None
        minimum_pattern = ranked_patterns[0][1] if ranked_patterns else None
        return V7PricingResult(
            anchor_bay_key=anchor,
            minimum_reduced_cost=minimum_value,
            minimum_pattern=minimum_pattern,
            returned_patterns=negative,
            diagnostics={
                "candidate_group_count": len({atom.group_id for atom in local}),
                "support_count_total": len(support_sets),
                "support_count_evaluated": len(support_sets),
                "support_count_pruned": 0,
                "pricing_seconds": perf_counter() - started,
                "min_reduced_cost": minimum_value,
                "returned_negative_columns": len(negative),
                "exact": True,
                "method": "joint_support_row_partition_dp",
                "dp_transition_count": transition_count,
                "dp_pruned_assignment_count": pruned_assignment_count,
                "excluded_local_signature_count": local_excluded_count,
            },
        )

    def price_bay_exact_mip(
        self,
        anchor_bay_key: str,
        atom_reduced_cost: Callable[[V7RowAtom], float],
        bay_reduced_cost: Callable[[str], float],
        group_support_reduced_cost: Callable[
            [str, str, str, str, tuple[str, ...]], float
        ],
        *,
        verify_reduced_cost: Callable[[V7BayPattern], float] | None = None,
        allowed_groups: Iterable[str] | None = None,
        excluded_signatures: Iterable[tuple[int, ...]] = (),
        excluded_master_column_signatures: Iterable[
            tuple[object, ...]
        ] = (),
        solver_threads: int = 1,
        excluded_signatures_are_dual_feasible: bool = False,
    ) -> V7PricingResult:
        """Solve exact per-state 0-1 pricing MIPs and return global top-K.

        The formulation chooses at most three groups and assigns each physical
        row to at most one of them.  Gurobi's systematic solution-pool mode
        proves the requested best alternatives for every size/height state;
        every recovered column is then checked against the master reduced
        cost when ``verify_reduced_cost`` is supplied.

        In a column-generation loop, signatures already present in an optimal
        RMP are dual-feasible by construction.  When
        ``excluded_signatures_are_dual_feasible`` is true, those signatures
        are filtered after pricing instead of being converted into an
        ever-growing family of no-good rows.  A materially negative existing
        column raises an error rather than allowing a false root closure.
        """

        started = perf_counter()
        anchor = str(anchor_bay_key)
        allowed = (
            None
            if allowed_groups is None
            else {str(value) for value in allowed_groups}
        )
        excluded = {tuple(sorted(value)) for value in excluded_signatures}
        excluded_master_columns = set(excluded_master_column_signatures)
        local = tuple(
            atom
            for atom in self.atoms_by_anchor.get(anchor, ())
            if allowed is None or atom.group_id in allowed
        )
        by_state: defaultdict[tuple[str, str], list[V7RowAtom]] = defaultdict(list)
        for atom in local:
            by_state[(atom.size, atom.height)].append(atom)

        priced: list[tuple[float, V7BayPattern]] = []
        total_solver_seconds = 0.0
        total_nodes = 0.0
        total_variables = 0
        total_constraints = 0
        excluded_local_count = 0
        solved_states = 0
        screened_states = 0
        state_lower_bounds: list[float] = []
        persistent_state_build_count = 0
        persistent_state_reuse_count = 0
        bay_cost = float(bay_reduced_cost(anchor))
        for (size, height), state_atoms in sorted(by_state.items()):
            groups = sorted({atom.group_id for atom in state_atoms})
            if not groups:
                continue
            physical_bays = tuple(sorted(state_atoms[0].physical_bays))
            if any(tuple(sorted(atom.physical_bays)) != physical_bays for atom in state_atoms):
                raise RuntimeError("V7 pricing state contains inconsistent footprints")
            state_indices = {atom.candidate_index for atom in state_atoms}
            state_excluded = [
                signature
                for signature in excluded
                if signature and set(signature) <= state_indices
            ]
            excluded_local_count += len(state_excluded)

            # Exact safe screening.  This lower bound relaxes the links
            # between selected groups and rows, the positive-pattern
            # requirement, and the capacity limit.  It can therefore only be
            # more optimistic than the pricing MIP.  A nonnegative bound
            # proves that this size/height state cannot contain an improving
            # column and avoids constructing a Gurobi model altogether.
            atom_costs = {
                atom.candidate_index: float(atom_reduced_cost(atom))
                for atom in state_atoms
            }
            group_costs = {
                group_id: float(
                    group_support_reduced_cost(
                        anchor,
                        size,
                        height,
                        group_id,
                        physical_bays,
                    )
                )
                for group_id in groups
            }
            atoms_by_row: defaultdict[str, list[V7RowAtom]] = defaultdict(list)
            for atom in state_atoms:
                atoms_by_row[atom.row_no].append(atom)
            optimistic_group_cost = sum(
                value
                for value in sorted(group_costs.values())[:3]
                if value < 0.0
            )
            optimistic_row_cost = sum(
                min(
                    0.0,
                    min(atom_costs[atom.candidate_index] for atom in row_atoms),
                )
                for row_atoms in atoms_by_row.values()
            )
            state_lower_bound = (
                bay_cost + optimistic_group_cost + optimistic_row_cost
            )
            state_lower_bounds.append(float(state_lower_bound))
            if state_lower_bound >= -self.tolerance:
                screened_states += 1
                continue

            pool_solution_limit = self.columns_per_bay * (
                4 if excluded_signatures_are_dual_feasible else 1
            )
            cache_key = (
                anchor,
                size,
                height,
                tuple(sorted(state_indices)),
                max(1, int(solver_threads)),
                pool_solution_limit,
            )
            persistent_state = (
                self._persistent_mip_states.get(cache_key)
                if excluded_signatures_are_dual_feasible
                else None
            )
            ephemeral_model = not excluded_signatures_are_dual_feasible
            if persistent_state is None:
                model = GurobiModel(f"v7_price_{anchor}_{size}_{height}")
                model.hideOutput()
                model.setMinimize()
                model.setParam("Threads", max(1, int(solver_threads)))
                model.setParam("MIPGap", 0.0)
                model.setParam("PoolSearchMode", 2)
                model.setParam("PoolSolutions", pool_solution_limit)
                atom_selected = {
                    atom.candidate_index: model.addVar(
                        vtype="B",
                        obj=0.0,
                        name=f"z_{atom.candidate_index}",
                    )
                    for atom in state_atoms
                }
                group_used = {
                    group_id: model.addVar(
                        vtype="B",
                        obj=0.0,
                        name=f"u_{index}",
                    )
                    for index, group_id in enumerate(groups)
                }
                gp = model._gp
                quicksum = gp.quicksum
                by_row: defaultdict[str, list[object]] = defaultdict(list)
                by_group: defaultdict[str, list[object]] = defaultdict(list)
                for atom in state_atoms:
                    variable = atom_selected[atom.candidate_index]
                    by_row[atom.row_no].append(variable)
                    by_group[atom.group_id].append(variable)
                    model.addConstr(variable <= group_used[atom.group_id])
                for variables in by_row.values():
                    model.addConstr(quicksum(variables) <= 1)
                for group_id, variables in by_group.items():
                    model.addConstr(group_used[group_id] <= quicksum(variables))
                model.addConstr(quicksum(group_used.values()) >= 1)
                model.addConstr(quicksum(group_used.values()) <= 3)
                capacity_limit = min(
                    int(self.problem.bays[anchor].cap_by_size.get(size, 0)),
                    *(
                        int(self.problem.bays[physical].physical_capacity)
                        for physical in physical_bays
                    ),
                )
                model.addConstr(
                    quicksum(
                        atom.capacity * atom_selected[atom.candidate_index]
                        for atom in state_atoms
                    )
                    <= capacity_limit
                )
                if not excluded_signatures_are_dual_feasible:
                    for signature in state_excluded:
                        selected = set(signature)
                        model.addConstr(
                            quicksum(
                                atom_selected[index] for index in selected
                            )
                            - quicksum(
                                atom_selected[index]
                                for index in sorted(state_indices - selected)
                            )
                            <= len(selected) - 1
                        )
                bay_constant = model.addVar(
                    lb=1.0,
                    ub=1.0,
                    obj=0.0,
                    name="bay_constant",
                )
                model.update()
                constraint_count = (
                    len(state_atoms)
                    + len(by_row)
                    + len(by_group)
                    + (
                        0
                        if excluded_signatures_are_dual_feasible
                        else len(state_excluded)
                    )
                    + 3
                )
                persistent_state = _V7PersistentPricingState(
                    model=model,
                    atom_selected=atom_selected,
                    group_used=group_used,
                    bay_constant=bay_constant,
                    variable_count=len(model.getVars()),
                    constraint_count=constraint_count,
                )
                if excluded_signatures_are_dual_feasible:
                    self._persistent_mip_states[cache_key] = persistent_state
                    persistent_state_build_count += 1
            else:
                persistent_state_reuse_count += 1
            model = persistent_state.model
            atom_selected = persistent_state.atom_selected
            group_used = persistent_state.group_used
            for index, variable in atom_selected.items():
                model.setVarObjective(variable, atom_costs[index])
            for group_id, variable in group_used.items():
                model.setVarObjective(variable, group_costs[group_id])
            model.setVarObjective(persistent_state.bay_constant, bay_cost)
            model.update()
            total_variables += persistent_state.variable_count
            total_constraints += persistent_state.constraint_count
            try:
                model.optimize()
                status = model.getStatusName()
                if status == "infeasible":
                    # Every legal pattern in this state may already be present
                    # in the RMP and removed by exact no-good constraints.
                    total_solver_seconds += model.getRuntime()
                    total_nodes += model.getNodeCount()
                    solved_states += 1
                    continue
                if status != "optimal":
                    raise V7PricingIncompleteError(
                        "V7 exact pricing MIP did not prove its solution pool: "
                        f"anchor={anchor}, state={(size, height)}, status={status}"
                    )
                solved_states += 1
                total_solver_seconds += model.getRuntime()
                total_nodes += model.getNodeCount()
                solution_count = min(
                    model.getSolutionCount(), pool_solution_limit
                )
                for solution_number in range(solution_count):
                    signature = tuple(
                        sorted(
                            index
                            for index, variable in atom_selected.items()
                            if model.getPoolValue(variable, solution_number) > 0.5
                        )
                    )
                    if not signature:
                        continue
                    pattern = build_v7_pattern_from_atom_indices(
                        self.problem,
                        self.atoms_by_index,
                        signature,
                        pattern_id=0,
                    )
                    pool_value = float(model.getPoolObjective(solution_number))
                    value = (
                        float(verify_reduced_cost(pattern))
                        if verify_reduced_cost is not None
                        else pool_value
                    )
                    if verify_reduced_cost is not None and abs(value - pool_value) > 1e-8:
                        raise RuntimeError(
                            "V7 pricing MIP objective disagrees with the master "
                            f"reduced cost: mip={pool_value}, master={value}"
                        )
                    if signature in excluded:
                        if (
                            excluded_signatures_are_dual_feasible
                            and value < -self.tolerance
                        ):
                            raise V7PricingIncompleteError(
                                "An existing RMP column has materially negative "
                                "reduced cost under the returned master duals: "
                                f"anchor={anchor}, value={value}"
                            )
                        continue
                    if pattern.master_column_signature in excluded_master_columns:
                        if (
                            excluded_signatures_are_dual_feasible
                            and value < -self.tolerance
                        ):
                            raise V7PricingIncompleteError(
                                "A master-equivalent existing RMP column has "
                                "materially negative reduced cost under the "
                                f"returned master duals: anchor={anchor}, "
                                f"value={value}"
                            )
                        continue
                    priced.append((value, pattern))
            finally:
                if ephemeral_model:
                    model.dispose()

        unique_priced: dict[tuple[object, ...], tuple[float, V7BayPattern]] = {}
        for value, pattern in priced:
            key = (
                pattern.master_column_signature
                if excluded_signatures_are_dual_feasible
                else pattern.signature
            )
            previous = unique_priced.get(key)
            if previous is None or value < previous[0]:
                unique_priced[key] = (value, pattern)
        ranked = sorted(
            unique_priced.values(), key=lambda item: (item[0], item[1].signature)
        )
        negative = tuple(
            pattern for value, pattern in ranked if value < -self.tolerance
        )[: self.columns_per_bay]
        minimum_value = ranked[0][0] if ranked else None
        minimum_pattern = ranked[0][1] if ranked else None
        return V7PricingResult(
            anchor_bay_key=anchor,
            minimum_reduced_cost=minimum_value,
            minimum_pattern=minimum_pattern,
            returned_patterns=negative,
            diagnostics={
                "candidate_group_count": len({atom.group_id for atom in local}),
                "support_count_total": None,
                "support_count_evaluated": None,
                "support_count_pruned": None,
                "pricing_seconds": perf_counter() - started,
                "pricing_solver_seconds": total_solver_seconds,
                "pricing_node_count": total_nodes,
                "pricing_state_count": solved_states,
                "pricing_state_total": len(by_state),
                "pricing_state_screened_by_lower_bound": screened_states,
                "persistent_pricing_state_build_count": (
                    persistent_state_build_count
                ),
                "persistent_pricing_state_reuse_count": (
                    persistent_state_reuse_count
                ),
                "persistent_pricing_enabled": bool(
                    excluded_signatures_are_dual_feasible
                ),
                "minimum_state_relaxation_lower_bound": (
                    min(state_lower_bounds) if state_lower_bounds else None
                ),
                "pricing_variable_count": total_variables,
                "pricing_constraint_count": total_constraints,
                "min_reduced_cost": minimum_value,
                "returned_negative_columns": len(negative),
                "exact": True,
                "method": "exact_support_cardinality_row_assignment_mip",
                "excluded_local_signature_count": excluded_local_count,
                "excluded_master_column_signature_count": len(
                    excluded_master_columns
                ),
                "excluded_signature_mode": (
                    "dual_feasible_filter"
                    if excluded_signatures_are_dual_feasible
                    else "exact_no_good_constraints"
                ),
                "no_good_constraint_count": (
                    0
                    if excluded_signatures_are_dual_feasible
                    else excluded_local_count
                ),
                "solution_pool_oversampling_factor": (
                    4 if excluded_signatures_are_dual_feasible else 1
                ),
                "master_equivalent_pool_columns_removed": (
                    len(priced) - len(unique_priced)
                ),
                "anchor_atom_index_used": True,
            },
        )


__all__ = [
    "V7BayPattern",
    "V7ExactBayPricing",
    "V7PatternEnumerationIncompleteError",
    "V7PricingIncompleteError",
    "V7PricingResult",
    "build_v7_pattern_from_atom_indices",
    "enumerate_v7_bay_patterns",
    "exhaustive_v7_bay_patterns",
]
