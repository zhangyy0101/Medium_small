from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


DEFAULT_GROUP_ATTRIBUTES = ("IYC_CSZ_CSIZECD", "IYC_POT_UNLDPORT", "IYC_CHEIGHTCD")
DEFAULT_BAY_NO_MIX_ATTRIBUTES = ("IYC_CHEIGHTCD",)
DEFAULT_ROW_NO_MIX_ATTRIBUTES = ("IYC_POT_UNLDPORT",)
EXPORT_VOYAGE_ROW_NO_MIX_ATTR = "__EXPORT_VOYAGE_ID"


@dataclass(frozen=True)
class AttributeRules:
    group_attributes: tuple[str, ...] = DEFAULT_GROUP_ATTRIBUTES
    bay_no_mix_attributes: tuple[str, ...] = DEFAULT_BAY_NO_MIX_ATTRIBUTES
    row_no_mix_attributes: tuple[str, ...] = DEFAULT_ROW_NO_MIX_ATTRIBUTES

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "group_attributes": list(self.group_attributes),
            "bay_no_mix_attributes": list(self.bay_no_mix_attributes),
            "row_no_mix_attributes": list(self.row_no_mix_attributes),
        }

    def group_for(self, voyage_id: object) -> tuple[str, ...]:
        return self.group_attributes

    def bay_no_mix_for(self, voyage_id: object) -> tuple[str, ...]:
        return self.bay_no_mix_attributes

    def row_no_mix_for(self, voyage_id: object) -> tuple[str, ...]:
        return self.row_no_mix_attributes


@dataclass
class Bay:
    """Bay state and residual row capacity at the planning instant."""

    area_no: str
    bay_no: str
    bay_key: str
    bay_order: int
    cap_by_size: dict[str, int] = field(default_factory=dict)
    physical_capacity: int = 0
    row_cap_by_size: dict[str, dict[str, int]] = field(default_factory=dict)
    row_physical_capacity: dict[str, int] = field(default_factory=dict)
    large_bay_partner_no: str = ""
    large_bay_partner_key: str = ""
    existing_size_modes: set[str] = field(default_factory=set)
    existing_heights: set[str] = field(default_factory=set)
    existing_ports: set[str] = field(default_factory=set)
    existing_ports_by_row: dict[str, set[str]] = field(default_factory=dict)
    existing_attrs: dict[str, set[str]] = field(default_factory=dict)
    existing_attrs_by_row: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    existing_attrs_by_voyage: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    existing_attrs_by_row_by_voyage: dict[str, dict[str, dict[str, set[str]]]] = field(default_factory=dict)


@dataclass(frozen=True)
class BigPlanRow:
    """One upstream area-plan row; ``new_boxes`` is read from ``new_qty``."""

    voyage_id: str
    flow: str
    area_no: str
    new_boxes: int
    size_mode: str
    plan_date: str = ""


@dataclass(frozen=True)
class VoyageSchedule:
    voyage_id: str
    receive_start: datetime
    receive_end: datetime
    berth_no: str
    berth_time: datetime
    departure_time: datetime


@dataclass(frozen=True)
class ExportGroup:
    group_id: str
    voyage_id: str
    status: str
    port: str
    size: str
    height: str
    demand: int
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DeclaredExportDemand:
    voyage_id: str
    flow: str
    port: str
    size: str
    declared_boxes: int


@dataclass
class ProblemData:
    """Inputs to export row allocation and anonymous import reservation."""

    export_groups: list[ExportGroup]
    bays: dict[str, Bay]
    area_guidance_target: dict[tuple[str, str, str, str], int]
    area_functions: dict[str, set[str]]
    target_voyages: list[str]
    export_voyages: set[str] | None = None
    import_area_size_reference: dict[tuple[str, str, str], int] = field(default_factory=dict)
    existing_group_area_load: dict[tuple[str, ...], int] = field(default_factory=dict)
    existing_group_bay_load: dict[tuple[str, ...], int] = field(default_factory=dict)
    berth_distances: dict[tuple[str, str], float] = field(default_factory=dict)
    berth_by_voyage: dict[str, str] = field(default_factory=dict)
    attribute_rules: AttributeRules = field(default_factory=AttributeRules)


@dataclass(frozen=True)
class PlanningInputs:
    demand_rows: list[DeclaredExportDemand]
    problem: ProblemData
