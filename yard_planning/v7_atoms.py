"""Zone-free V7 row atoms.

The V7 formal model allocates export flow to anchor bays and selects physical
row resources directly.  This module intentionally reuses the frozen V6.1
compatibility helpers, because size/footprint/legacy-state semantics are
unchanged, while exposing a V7-specific immutable atom type.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .models import ProblemData
from .row_aware_zones import build_v6_row_aware_bay_atoms, v6_footprint


PhysicalRowResource = tuple[str, str]


@dataclass(frozen=True)
class V7RowAtom:
    """One legal group/anchor-bay row-capacity option and its footprint."""

    candidate_index: int
    group_id: str
    area_no: str
    anchor_bay_key: str
    row_no: str
    capacity: int
    physical_resources: tuple[PhysicalRowResource, ...]
    physical_bays: tuple[str, ...]
    size: str
    height: str

    @property
    def signature(self) -> tuple[str, str, str]:
        return (self.group_id, self.anchor_bay_key, self.row_no)


def build_v7_row_atoms(
    problem: ProblemData,
) -> tuple[tuple[V7RowAtom, ...], dict[tuple[str, str], int]]:
    """Build the full legal V7 atom domain without any Stage-1 filtering."""

    legacy_atoms, anchor_limits = build_v6_row_aware_bay_atoms(problem)
    groups = {str(group.group_id): group for group in problem.export_groups}
    atoms: list[V7RowAtom] = []
    for source in legacy_atoms:
        group = groups[str(source.group_id)]
        footprint = v6_footprint(
            problem,
            str(source.anchor_bay_key),
            str(group.size),
        )
        if not footprint:
            raise RuntimeError(
                "V7 source atom unexpectedly has no physical footprint: "
                f"{source.candidate_index}"
            )
        resources = tuple(
            sorted((str(bay_key), str(row_no)) for bay_key, row_no in source.resources)
        )
        expected = tuple(
            sorted((str(bay_key), str(source.anchor_row_no)) for bay_key in footprint)
        )
        if resources != expected:
            raise RuntimeError(
                "V7 atom footprint differs from its physical row resources: "
                f"source={source.candidate_index}"
            )
        atoms.append(
            V7RowAtom(
                candidate_index=len(atoms),
                group_id=str(source.group_id),
                area_no=str(source.area_no),
                anchor_bay_key=str(source.anchor_bay_key),
                row_no=str(source.anchor_row_no),
                capacity=int(source.capacity),
                physical_resources=resources,
                physical_bays=tuple(str(key) for key in footprint),
                size=str(group.size),
                height=str(group.height),
            )
        )
    return tuple(atoms), {
        (str(group_id), str(bay_key)): int(capacity)
        for (group_id, bay_key), capacity in anchor_limits.items()
    }


def atoms_by_group_bay(
    atoms: Iterable[V7RowAtom],
) -> dict[tuple[str, str], tuple[V7RowAtom, ...]]:
    grouped: defaultdict[tuple[str, str], list[V7RowAtom]] = defaultdict(list)
    for atom in atoms:
        grouped[(atom.group_id, atom.anchor_bay_key)].append(atom)
    return {
        key: tuple(sorted(values, key=lambda atom: (atom.row_no, atom.candidate_index)))
        for key, values in sorted(grouped.items())
    }


def candidate_bays_by_group(
    atoms: Iterable[V7RowAtom],
) -> dict[str, set[str]]:
    output: defaultdict[str, set[str]] = defaultdict(set)
    for atom in atoms:
        output[atom.group_id].add(atom.anchor_bay_key)
    return dict(output)


def candidate_areas_by_group(
    atoms: Iterable[V7RowAtom],
) -> dict[str, set[str]]:
    output: defaultdict[str, set[str]] = defaultdict(set)
    for atom in atoms:
        output[atom.group_id].add(atom.area_no)
    return dict(output)


def filter_atoms_by_active_areas(
    atoms: Sequence[V7RowAtom],
    active_areas_by_group: Mapping[str, Iterable[str]],
) -> tuple[V7RowAtom, ...]:
    active = {
        str(group_id): {str(area) for area in areas}
        for group_id, areas in active_areas_by_group.items()
    }
    return tuple(
        atom
        for atom in atoms
        if atom.area_no in active.get(atom.group_id, set())
    )


__all__ = [
    "PhysicalRowResource",
    "V7RowAtom",
    "atoms_by_group_bay",
    "build_v7_row_atoms",
    "candidate_areas_by_group",
    "candidate_bays_by_group",
    "filter_atoms_by_active_areas",
]
