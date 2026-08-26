"""V6 row-aware contiguous-bay zone primitives.

This module intentionally contains only the complete, small-instance zone
universe.  It is the correctness oracle for the later pricing implementation;
it is not intended to enumerate production-size instances.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Iterable, Mapping, Protocol, Sequence

from .models import (
    EXPORT_VOYAGE_ROW_NO_MIX_ATTR,
    ExportGroup,
    ProblemData,
    existing_export_group_key,
)


PhysicalRowResource = tuple[str, str]


class RowLocationLike(Protocol):
    """Minimal row-location interface accepted by the V6 zone oracle."""

    group_id: str
    area_no: str
    bay_key: str
    row_allocation: tuple[tuple[str, str, int], ...]


@dataclass(frozen=True)
class RowAwareBayAtom:
    """One group-specific, anchor-bay row option and its full footprint."""

    candidate_index: int
    group_id: str
    area_no: str
    anchor_bay_key: str
    anchor_row_no: str
    capacity: int
    resources: tuple[PhysicalRowResource, ...]
    footprint_orders: tuple[int, ...]


@dataclass(frozen=True)
class RowAwareZone:
    """A group-specific contiguous bay interval with selected row resources."""

    zone_id: int
    group_id: str
    area_no: str
    anchor_bay_keys: tuple[str, ...]
    anchor_bay_capacities: tuple[tuple[str, int], ...]
    candidate_indices: tuple[int, ...]
    rows_by_anchor_bay: tuple[tuple[str, tuple[str, ...]], ...]
    capacity: int
    resources: tuple[PhysicalRowResource, ...]
    physical_bay_keys: tuple[str, ...]


def _row_sort_key(row_no: str) -> tuple[int, str]:
    try:
        return int(row_no), row_no
    except ValueError:
        return 10**9, row_no


def v6_footprint(
    problem: ProblemData,
    bay_key: str,
    size: str,
) -> tuple[str, ...]:
    """Return the complete physical-bay footprint of one V6 anchor bay."""

    bay = problem.bays.get(str(bay_key))
    if bay is None:
        return ()
    if str(size) in {"40", "45"}:
        partner = str(bay.large_bay_partner_key or "")
        if not partner or partner not in problem.bays:
            return ()
        return (str(bay_key), partner)
    if str(size) == "20":
        return (str(bay_key),)
    return ()


def v6_edge_large_bays(problem: ProblemData) -> set[str]:
    """Return 40/45 anchor bays whose footprint touches an area boundary."""

    by_area: defaultdict[str, list[str]] = defaultdict(list)
    for bay_key, bay in problem.bays.items():
        by_area[str(bay.area_no)].append(str(bay_key))
    output: set[str] = set()
    for keys in by_area.values():
        keys.sort(key=lambda key: problem.bays[key].bay_order)
        if not keys:
            continue
        boundaries = {keys[0], keys[-1]}
        output.update(
            key
            for key in keys
            if problem.bays[key].large_bay_partner_key
            and (
                key in boundaries
                or problem.bays[key].large_bay_partner_key in boundaries
            )
        )
    return output


def _group_allows_row(
    problem: ProblemData,
    group: ExportGroup,
    footprint: tuple[str, ...],
    row_no: str,
) -> bool:
    for bay_key in footprint:
        bay = problem.bays[bay_key]
        sizes = {str(value) for value in bay.existing_size_modes if str(value)}
        heights = {str(value) for value in bay.existing_heights if str(value)}
        ports = {
            str(value)
            for value in bay.existing_ports_by_row.get(str(row_no), set())
            if str(value)
        }
        voyages = {
            str(value)
            for value in bay.existing_attrs_by_row.get(str(row_no), {}).get(
                EXPORT_VOYAGE_ROW_NO_MIX_ATTR,
                set(),
            )
            if str(value)
        }
        exact_groups = {
            tuple(str(part) for part in value)
            for value in bay.existing_group_keys_by_row.get(
                str(row_no), set()
            )
        }
        if sizes and sizes != {str(group.size)}:
            return False
        if heights and str(group.height) not in heights:
            return False
        if exact_groups:
            if existing_export_group_key(group) not in exact_groups:
                return False
        else:
            # Backward-compatible fallback for hand-built ProblemData that
            # predates exact historical group keys.  Real adapted inputs
            # always use the exact joint key above, avoiding voyage/port
            # cross-product matches.
            if ports and str(group.port) not in ports:
                return False
            if voyages and str(group.voyage_id) not in voyages:
                return False
    return True


def build_v6_row_aware_bay_atoms(
    problem: ProblemData,
) -> tuple[tuple[RowAwareBayAtom, ...], dict[tuple[str, str], int]]:
    """Build V6 row atoms directly from ``ProblemData`` without V5 planners."""

    atoms: list[RowAwareBayAtom] = []
    anchor_capacity_limits: dict[tuple[str, str], int] = {}
    edge_large_bays = v6_edge_large_bays(problem)
    groups = sorted(
        (
            group
            for group in problem.export_groups
            if str(group.status) == "OF" and int(group.demand) > 0
        ),
        key=lambda group: str(group.group_id),
    )
    bays = sorted(
        problem.bays.values(),
        key=lambda bay: (str(bay.area_no), int(bay.bay_order), str(bay.bay_key)),
    )
    for group in groups:
        for anchor in bays:
            if "OF" not in problem.area_functions.get(str(anchor.area_no), set()):
                continue
            footprint = v6_footprint(problem, anchor.bay_key, group.size)
            if not footprint:
                continue
            if any(
                str(problem.bays[key].area_no) != str(anchor.area_no)
                for key in footprint
            ):
                continue
            if group.size == "45" and anchor.bay_key not in edge_large_bays:
                continue
            anchor_limit = min(
                int(anchor.cap_by_size.get(group.size, 0)),
                *(int(problem.bays[key].physical_capacity) for key in footprint),
            )
            if anchor_limit <= 0:
                continue
            common_rows = set.intersection(
                *(
                    set(problem.bays[key].row_physical_capacity)
                    & set(problem.bays[key].row_cap_by_size.get(group.size, {}))
                    for key in footprint
                )
            )
            valid_atom_count = 0
            for row_no in sorted(common_rows, key=_row_sort_key):
                if not _group_allows_row(problem, group, footprint, row_no):
                    continue
                capacity = min(
                    anchor_limit,
                    *(
                        min(
                            int(problem.bays[key].row_physical_capacity[row_no]),
                            int(
                                problem.bays[key]
                                .row_cap_by_size[group.size][row_no]
                            ),
                        )
                        for key in footprint
                    ),
                )
                if capacity <= 0:
                    continue
                atoms.append(
                    RowAwareBayAtom(
                        candidate_index=len(atoms),
                        group_id=str(group.group_id),
                        area_no=str(anchor.area_no),
                        anchor_bay_key=str(anchor.bay_key),
                        anchor_row_no=str(row_no),
                        capacity=int(capacity),
                        resources=tuple(
                            (str(key), str(row_no)) for key in footprint
                        ),
                        footprint_orders=tuple(
                            int(problem.bays[key].bay_order) for key in footprint
                        ),
                    )
                )
                valid_atom_count += 1
            if valid_atom_count:
                anchor_capacity_limits[
                    (str(group.group_id), str(anchor.bay_key))
                ] = int(anchor_limit)
    return tuple(atoms), anchor_capacity_limits


def build_row_aware_bay_atoms(
    row_locations: Sequence[RowLocationLike],
    capacity_by_index: Mapping[int, int],
    bay_order_by_key: Mapping[str, int],
) -> tuple[RowAwareBayAtom, ...]:
    """Convert explicit row locations into immutable V6 bay-row atoms."""

    atoms: list[RowAwareBayAtom] = []
    seen: set[tuple[str, str, str, str]] = set()
    for index, column in enumerate(row_locations):
        capacity = int(capacity_by_index.get(index, 0))
        if capacity <= 0:
            continue
        resources = tuple(
            sorted(
                {
                    (str(bay_key), str(row_no))
                    for bay_key, row_no, quantity in column.row_allocation
                    if int(quantity) > 0
                }
            )
        )
        if not resources:
            continue
        anchor_rows = {
            str(row_no)
            for bay_key, row_no, quantity in column.row_allocation
            if str(bay_key) == str(column.bay_key) and int(quantity) > 0
        }
        if len(anchor_rows) != 1:
            raise ValueError(
                "a row-aware atom must use exactly one anchor-bay row: "
                f"candidate_index={index}, anchor_bay={column.bay_key}"
            )
        anchor_row_no = next(iter(anchor_rows))
        static_key = (
            str(column.group_id),
            str(column.area_no),
            str(column.bay_key),
            anchor_row_no,
        )
        if static_key in seen:
            raise ValueError(f"duplicate V6 bay-row atom: {static_key}")
        seen.add(static_key)
        try:
            footprint_orders = tuple(
                sorted({int(bay_order_by_key[bay_key]) for bay_key, _row in resources})
            )
        except KeyError as exc:
            raise ValueError(
                f"missing bay order for row-aware atom resource: {exc.args[0]}"
            ) from exc
        atoms.append(
            RowAwareBayAtom(
                candidate_index=index,
                group_id=str(column.group_id),
                area_no=str(column.area_no),
                anchor_bay_key=str(column.bay_key),
                anchor_row_no=anchor_row_no,
                capacity=capacity,
                resources=resources,
                footprint_orders=footprint_orders,
            )
        )
    return tuple(
        sorted(
            atoms,
            key=lambda atom: (
                atom.group_id,
                atom.area_no,
                atom.footprint_orders,
                atom.anchor_bay_key,
                _row_sort_key(atom.anchor_row_no),
            ),
        )
    )


def _nonempty_row_subsets(
    atoms: Sequence[RowAwareBayAtom],
    capacity_limit: int | None = None,
) -> tuple[tuple[RowAwareBayAtom, ...], ...]:
    options: list[tuple[RowAwareBayAtom, ...]] = []
    for count in range(1, len(atoms) + 1):
        for subset in combinations(atoms, count):
            if capacity_limit is not None and sum(
                atom.capacity for atom in subset
            ) > int(capacity_limit):
                continue
            resources = [resource for atom in subset for resource in atom.resources]
            if len(resources) != len(set(resources)):
                continue
            options.append(tuple(subset))
    return tuple(options)


def _contiguous_bay_runs(
    atoms_by_bay: Mapping[str, Sequence[RowAwareBayAtom]],
) -> tuple[tuple[str, ...], ...]:
    footprints: dict[str, tuple[int, ...]] = {}
    for bay_key, atoms in atoms_by_bay.items():
        values = {atom.footprint_orders for atom in atoms}
        if len(values) != 1:
            raise ValueError(
                "all row atoms of one anchor bay must share one footprint: "
                f"bay={bay_key}, footprints={sorted(values)}"
            )
        footprints[bay_key] = next(iter(values))
    ordered = sorted(
        atoms_by_bay,
        key=lambda bay_key: (footprints[bay_key], bay_key),
    )
    runs: list[list[str]] = []
    for bay_key in ordered:
        if not runs:
            runs.append([bay_key])
            continue
        previous = footprints[runs[-1][-1]]
        current = footprints[bay_key]
        if current[0] - previous[-1] == 2:
            runs[-1].append(bay_key)
        else:
            runs.append([bay_key])
    return tuple(tuple(run) for run in runs)


def make_row_aware_zone(
    selected_atoms: Iterable[RowAwareBayAtom],
    *,
    zone_id: int = -1,
) -> RowAwareZone:
    """Build one immutable zone from a valid nonempty row-atom selection.

    The caller is responsible for enforcing anchor capacity and interval
    contiguity.  This constructor performs the structural checks shared by the
    complete enumeration oracle and the exact-pricing implementation.
    """

    atoms = tuple(selected_atoms)
    if not atoms:
        raise ValueError("a V6 row-aware zone must contain at least one atom")
    group_ids = {atom.group_id for atom in atoms}
    area_nos = {atom.area_no for atom in atoms}
    if len(group_ids) != 1 or len(area_nos) != 1:
        raise ValueError("all V6 zone atoms must share one group and one area")
    candidate_indices = tuple(sorted(atom.candidate_index for atom in atoms))
    if len(candidate_indices) != len(set(candidate_indices)):
        raise ValueError("a V6 row-aware zone cannot repeat an atom")
    resources = tuple(resource for atom in atoms for resource in atom.resources)
    if len(resources) != len(set(resources)):
        raise ValueError("a V6 row-aware zone cannot repeat a physical row")

    by_bay: defaultdict[str, list[RowAwareBayAtom]] = defaultdict(list)
    for atom in atoms:
        by_bay[atom.anchor_bay_key].append(atom)
    footprints: dict[str, tuple[int, ...]] = {}
    for bay_key, bay_atoms in by_bay.items():
        values = {atom.footprint_orders for atom in bay_atoms}
        if len(values) != 1:
            raise ValueError(
                "all atoms at one V6 anchor bay must share a footprint"
            )
        footprints[bay_key] = next(iter(values))
    anchor_bay_keys = tuple(
        sorted(by_bay, key=lambda bay_key: (footprints[bay_key], bay_key))
    )
    if any(
        current[0] - previous[-1] != 2
        for previous, current in zip(
            (footprints[key] for key in anchor_bay_keys),
            (footprints[key] for key in anchor_bay_keys[1:]),
        )
    ):
        raise ValueError("V6 zone anchor bays must form one contiguous interval")

    rows_by_anchor_bay = tuple(
        (
            bay_key,
            tuple(
                sorted(
                    (atom.anchor_row_no for atom in by_bay[bay_key]),
                    key=_row_sort_key,
                )
            ),
        )
        for bay_key in anchor_bay_keys
    )
    anchor_bay_capacities = tuple(
        (bay_key, sum(atom.capacity for atom in by_bay[bay_key]))
        for bay_key in anchor_bay_keys
    )
    return RowAwareZone(
        zone_id=int(zone_id),
        group_id=next(iter(group_ids)),
        area_no=next(iter(area_nos)),
        anchor_bay_keys=anchor_bay_keys,
        anchor_bay_capacities=anchor_bay_capacities,
        candidate_indices=candidate_indices,
        rows_by_anchor_bay=rows_by_anchor_bay,
        capacity=sum(atom.capacity for atom in atoms),
        resources=tuple(sorted(set(resources))),
        physical_bay_keys=tuple(
            sorted({bay_key for bay_key, _row_no in resources})
        ),
    )


def enumerate_row_aware_zones(
    atoms: Iterable[RowAwareBayAtom],
    group_demand_by_id: Mapping[str, int],
    anchor_capacity_limits: Mapping[tuple[str, str], int] | None = None,
    maximum_zone_count: int | None = None,
) -> tuple[RowAwareZone, ...]:
    """Fully enumerate V6 zones for tests and small exact references.

    For every contiguous anchor-bay interval, at least one compatible row atom
    is chosen in each bay.  The chosen row numbers may differ between bays and
    several rows may be included in one bay.  A zone may not contain the same
    physical ``(bay, row)`` resource twice.
    """

    by_group_area: defaultdict[
        tuple[str, str], defaultdict[str, list[RowAwareBayAtom]]
    ] = defaultdict(lambda: defaultdict(list))
    for atom in atoms:
        by_group_area[(atom.group_id, atom.area_no)][
            atom.anchor_bay_key
        ].append(atom)

    zones: list[RowAwareZone] = []
    signatures: set[tuple[int, ...]] = set()
    anchor_capacity_limits = anchor_capacity_limits or {}
    if maximum_zone_count is not None and int(maximum_zone_count) <= 0:
        raise ValueError("maximum_zone_count must be positive when provided")
    for (group_id, area_no), atoms_by_bay in sorted(by_group_area.items()):
        demand = int(group_demand_by_id.get(group_id, 0))
        if demand <= 0:
            continue
        options_by_bay = {
            bay_key: _nonempty_row_subsets(
                sorted(
                    bay_atoms,
                    key=lambda atom: _row_sort_key(atom.anchor_row_no),
                ),
                anchor_capacity_limits.get((group_id, bay_key)),
            )
            for bay_key, bay_atoms in atoms_by_bay.items()
        }
        for run in _contiguous_bay_runs(atoms_by_bay):
            for start in range(len(run)):
                for end in range(start, len(run)):
                    interval = run[start : end + 1]

                    def extend(
                        position: int,
                        selected_atoms: tuple[RowAwareBayAtom, ...],
                        used_resources: frozenset[PhysicalRowResource],
                        used_capacity: int,
                    ) -> None:
                        if position == len(interval):
                            signature = tuple(
                                sorted(atom.candidate_index for atom in selected_atoms)
                            )
                            if signature in signatures:
                                return
                            signatures.add(signature)
                            if (
                                maximum_zone_count is not None
                                and len(zones) >= int(maximum_zone_count)
                            ):
                                raise RuntimeError(
                                    "complete V6 zone enumeration exceeded its "
                                    f"execution safety limit ({maximum_zone_count}); "
                                    "the legal universe was not truncated"
                                )
                            zone = make_row_aware_zone(selected_atoms)
                            if zone.anchor_bay_keys != tuple(interval):
                                raise RuntimeError(
                                    "V6 zone constructor changed the enumerated interval"
                                )
                            if zone.capacity != used_capacity:
                                raise RuntimeError(
                                    "V6 zone constructor changed enumerated capacity"
                                )
                            zones.append(zone)
                            return

                        bay_key = interval[position]
                        for option in options_by_bay[bay_key]:
                            option_capacity = sum(atom.capacity for atom in option)
                            next_capacity = used_capacity + option_capacity
                            option_resources = frozenset(
                                resource for atom in option for resource in atom.resources
                            )
                            if used_resources & option_resources:
                                continue
                            extend(
                                position + 1,
                                selected_atoms + option,
                                used_resources | option_resources,
                                next_capacity,
                            )

                    extend(0, (), frozenset(), 0)

    return tuple(
        replace(zone, zone_id=index)
        for index, zone in enumerate(zones)
    )


def build_complete_row_aware_zone_universe(
    row_locations: Sequence[RowLocationLike],
    capacity_by_index: Mapping[int, int],
    bay_order_by_key: Mapping[str, int],
    group_demand_by_id: Mapping[str, int],
) -> tuple[RowAwareZone, ...]:
    """Build the small-instance V6 zone universe in one call."""

    atoms = build_row_aware_bay_atoms(
        row_locations,
        capacity_by_index,
        bay_order_by_key,
    )
    return enumerate_row_aware_zones(atoms, group_demand_by_id)


def build_complete_v6_zone_universe(
    problem: ProblemData,
    maximum_zone_count: int | None = None,
) -> tuple[RowAwareZone, ...]:
    """Build the self-contained complete V6 universe for small instances."""

    atoms, anchor_capacity_limits = build_v6_row_aware_bay_atoms(problem)
    demand_by_group = {
        str(group.group_id): int(group.demand)
        for group in problem.export_groups
        if str(group.status) == "OF" and int(group.demand) > 0
    }
    return enumerate_row_aware_zones(
        atoms,
        demand_by_group,
        anchor_capacity_limits,
        maximum_zone_count,
    )


__all__ = [
    "PhysicalRowResource",
    "RowLocationLike",
    "RowAwareBayAtom",
    "RowAwareZone",
    "build_complete_row_aware_zone_universe",
    "build_complete_v6_zone_universe",
    "build_row_aware_bay_atoms",
    "build_v6_row_aware_bay_atoms",
    "enumerate_row_aware_zones",
    "make_row_aware_zone",
    "v6_edge_large_bays",
    "v6_footprint",
]
