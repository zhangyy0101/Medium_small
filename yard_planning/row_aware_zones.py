"""V6 row-aware contiguous-bay zone primitives.

This module intentionally contains only the complete, small-instance zone
universe.  It is the correctness oracle for the later pricing implementation;
it is not intended to enumerate production-size instances.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from .planner import PlacementColumn


PhysicalRowResource = tuple[str, str]


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
    candidate_indices: tuple[int, ...]
    rows_by_anchor_bay: tuple[tuple[str, tuple[str, ...]], ...]
    capacity: int
    resources: tuple[PhysicalRowResource, ...]


def _row_sort_key(row_no: str) -> tuple[int, str]:
    try:
        return int(row_no), row_no
    except ValueError:
        return 10**9, row_no


def build_row_aware_bay_atoms(
    row_locations: Sequence[PlacementColumn],
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
    capacity_limit: int,
) -> tuple[tuple[RowAwareBayAtom, ...], ...]:
    options: list[tuple[RowAwareBayAtom, ...]] = []
    for count in range(1, len(atoms) + 1):
        for subset in combinations(atoms, count):
            capacity = sum(atom.capacity for atom in subset)
            if capacity > capacity_limit:
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


def enumerate_row_aware_zones(
    atoms: Iterable[RowAwareBayAtom],
    group_demand_by_id: Mapping[str, int],
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
    for (group_id, area_no), atoms_by_bay in sorted(by_group_area.items()):
        demand = int(group_demand_by_id.get(group_id, 0))
        if demand <= 0:
            continue
        atomic_capacity = max(
            atom.capacity for bay_atoms in atoms_by_bay.values() for atom in bay_atoms
        )
        capacity_limit = demand + atomic_capacity
        options_by_bay = {
            bay_key: _nonempty_row_subsets(
                sorted(
                    bay_atoms,
                    key=lambda atom: _row_sort_key(atom.anchor_row_no),
                ),
                capacity_limit,
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
                            rows_by_bay = tuple(
                                (
                                    bay_key,
                                    tuple(
                                        sorted(
                                            (
                                                atom.anchor_row_no
                                                for atom in selected_atoms
                                                if atom.anchor_bay_key == bay_key
                                            ),
                                            key=_row_sort_key,
                                        )
                                    ),
                                )
                                for bay_key in interval
                            )
                            zones.append(
                                RowAwareZone(
                                    zone_id=-1,
                                    group_id=group_id,
                                    area_no=area_no,
                                    anchor_bay_keys=tuple(interval),
                                    candidate_indices=signature,
                                    rows_by_anchor_bay=rows_by_bay,
                                    capacity=used_capacity,
                                    resources=tuple(sorted(used_resources)),
                                )
                            )
                            return

                        bay_key = interval[position]
                        for option in options_by_bay[bay_key]:
                            option_capacity = sum(atom.capacity for atom in option)
                            next_capacity = used_capacity + option_capacity
                            if next_capacity > capacity_limit:
                                continue
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
    row_locations: Sequence[PlacementColumn],
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


__all__ = [
    "PhysicalRowResource",
    "RowAwareBayAtom",
    "RowAwareZone",
    "build_complete_row_aware_zone_universe",
    "build_row_aware_bay_atoms",
    "enumerate_row_aware_zones",
]
