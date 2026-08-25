from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


VF = tuple[str, str]
VFA = tuple[str, str, str]


@dataclass(frozen=True)
class LargePlanData:
    """Minimal macro-yard model data built from the shared paper input."""

    voyages: tuple[str, ...]
    flows: tuple[str, ...]
    areas: tuple[str, ...]
    direction_by_voyage: Mapping[str, str]
    demand20: Mapping[VF, int]
    demand40: Mapping[VF, int]
    snapshot20: Mapping[VFA, int]
    snapshot40: Mapping[VFA, int]
    capacity20_equiv: Mapping[str, int]
    capacity20_direct: Mapping[str, int]
    capacity40: Mapping[str, int]
    area_functions: Mapping[str, frozenset[str]]
    berth_by_voyage: Mapping[str, str]
    distance: Mapping[tuple[str, str], float]
    planning_time: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def snapshot_total(self, size: str, voyage: str, flow: str) -> int:
        source = self.snapshot20 if size == "20" else self.snapshot40
        return sum(source.get((voyage, flow, area), 0) for area in self.areas)

    def new_demand(self, size: str, voyage: str, flow: str) -> int:
        source = self.demand20 if size == "20" else self.demand40
        return max(0, int(source.get((voyage, flow), 0)) - self.snapshot_total(size, voyage, flow))


@dataclass
class LargePlanSolution:
    status: int
    status_name: str
    has_solution: bool
    runtime: float
    mip_gap: float | None
    objective_value: float | None
    objective_components: dict[str, float]
    new20: dict[VFA, int]
    new40: dict[VFA, int]
    shortage20: dict[VF, int]
    shortage40: dict[VF, int]
    model: Any | None = None

    @property
    def shortage_total(self) -> int:
        return sum(self.shortage20.values()) + sum(self.shortage40.values())
