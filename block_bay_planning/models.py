from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


DEFAULT_COARSE_GROUP_ATTRIBUTES = ("IYC_CSZ_CSIZECD", "IYC_POT_UNLDPORT")
DEFAULT_FINE_GROUP_ATTRIBUTES = ("IYC_CSZ_CSIZECD", "IYC_POT_UNLDPORT", "IYC_CHEIGHTCD")
DEFAULT_BAY_NO_MIX_ATTRIBUTES = ("IYC_CHEIGHTCD",)
DEFAULT_ROW_NO_MIX_ATTRIBUTES = ("IYC_POT_UNLDPORT",)
EXPORT_VOYAGE_ROW_NO_MIX_ATTR = "__EXPORT_VOYAGE_ID"
DEFAULT_WEIGHT_LEVELS = (0, 10, 15, 20, 25, 30)


@dataclass(frozen=True)
class AttributeRules:
    coarse_group_attributes: tuple[str, ...] = DEFAULT_COARSE_GROUP_ATTRIBUTES
    fine_group_attributes: tuple[str, ...] = DEFAULT_FINE_GROUP_ATTRIBUTES
    bay_no_mix_attributes: tuple[str, ...] = DEFAULT_BAY_NO_MIX_ATTRIBUTES
    row_no_mix_attributes: tuple[str, ...] = DEFAULT_ROW_NO_MIX_ATTRIBUTES
    weight_levels: tuple[int, ...] = DEFAULT_WEIGHT_LEVELS
    coarse_group_attributes_by_voyage: dict[str, tuple[str, ...]] = field(default_factory=dict)
    fine_group_attributes_by_voyage: dict[str, tuple[str, ...]] = field(default_factory=dict)
    bay_no_mix_attributes_by_voyage: dict[str, tuple[str, ...]] = field(default_factory=dict)
    row_no_mix_attributes_by_voyage: dict[str, tuple[str, ...]] = field(default_factory=dict)
    weight_levels_by_voyage: dict[str, tuple[int, ...]] = field(default_factory=dict)
    weight_group_voyages: frozenset[str] = frozenset()
    import_shared_fine_group_attributes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "coarse_group_attributes": list(self.coarse_group_attributes),
            "fine_group_attributes": list(self.fine_group_attributes),
            "bay_no_mix_attributes": list(self.bay_no_mix_attributes),
            "row_no_mix_attributes": list(self.row_no_mix_attributes),
            "weight_levels": list(self.weight_levels),
            "coarse_group_attributes_by_voyage": {
                voyage: list(values) for voyage, values in sorted(self.coarse_group_attributes_by_voyage.items())
            },
            "fine_group_attributes_by_voyage": {
                voyage: list(values) for voyage, values in sorted(self.fine_group_attributes_by_voyage.items())
            },
            "bay_no_mix_attributes_by_voyage": {
                voyage: list(values) for voyage, values in sorted(self.bay_no_mix_attributes_by_voyage.items())
            },
            "row_no_mix_attributes_by_voyage": {
                voyage: list(values) for voyage, values in sorted(self.row_no_mix_attributes_by_voyage.items())
            },
            "weight_levels_by_voyage": {
                voyage: list(values) for voyage, values in sorted(self.weight_levels_by_voyage.items())
            },
            "weight_group_voyages": sorted(self.weight_group_voyages),
            "import_shared_fine_group_attributes": list(self.import_shared_fine_group_attributes),
        }

    def coarse_for(self, voyage_id: object) -> tuple[str, ...]:
        return self.coarse_group_attributes_by_voyage.get(_voyage_key(voyage_id), self.coarse_group_attributes)

    def fine_for(self, voyage_id: object) -> tuple[str, ...]:
        return self.fine_group_attributes_by_voyage.get(_voyage_key(voyage_id), self.fine_group_attributes)

    def bay_no_mix_for(self, voyage_id: object) -> tuple[str, ...]:
        return self.bay_no_mix_attributes_by_voyage.get(_voyage_key(voyage_id), self.bay_no_mix_attributes)

    def row_no_mix_for(self, voyage_id: object) -> tuple[str, ...]:
        return self.row_no_mix_attributes_by_voyage.get(_voyage_key(voyage_id), self.row_no_mix_attributes)

    def weight_levels_for(self, voyage_id: object) -> tuple[int, ...]:
        return self.weight_levels_by_voyage.get(_voyage_key(voyage_id), self.weight_levels)

    def weight_group_enabled_for(self, voyage_id: object) -> bool:
        return _voyage_key(voyage_id) in self.weight_group_voyages


def _voyage_key(value: object) -> str:
    text = "" if value is None else str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


@dataclass(frozen=True)
class BoxGroup:
    """同一类出口箱需求。

    中计划按航次、卸港、尺寸聚合分配箱量；小计划再按资料箱细分属性组
    分配箱量到贝位。求解过程不再把需求展开成单箱决策变量。

    注意：属性组保留真实尺寸，例如真实 45ft 不会和真实 40ft 合成同一组。
    """

    group_id: str
    voyage_id: str
    size: str
    height: str
    status: str
    port: str
    operator: str
    ctype: str
    weight_class: str
    reefer: bool
    dangerous: bool
    over_limit: bool
    special_codes: tuple[str, ...]
    demand: int
    attributes: dict[str, str] = field(default_factory=dict)
    area_allowlist: set[str] | None = None

    @property
    def size_mode(self) -> str:
        """中计划和小计划使用的尺寸口径。

        当前实现区分 20ft、40ft 和 45ft；其它异常尺寸兜底为 40ft。
        """
        return self.size if self.size in {"20", "40", "45"} else "40"

    @property
    def big_plan_size_mode(self) -> str:
        """大计划尺寸口径。

        大计划输出只区分 20/40，45ft 按 40ft 箱区模式继承；小计划落贝
        仍使用 ``size_mode`` 保留 45ft 自身约束。
        """
        return "40" if self.size_mode == "45" else self.size_mode

    @property
    def special_signature(self) -> str:
        return self.business_special_signature

    @property
    def business_special_signature(self) -> str:
        return "+".join(self.special_codes) if self.special_codes else "NORMAL"

@dataclass
class Bay:
    """贝位状态和容量快照。

    `cap_by_size` 按 YBY_ENABLECSIZECD 保存 20/40/45 尺寸容量；同时
    `physical_capacity` 限制同一贝位的总占用不超量。
    """

    area_no: str
    bay_no: str
    bay_key: str
    block_id: str
    block_bays: tuple[str, ...]
    block_bay_count: int
    block_boundary_adjusted: bool
    bay_order: int
    cap_by_size: dict[str, int] = field(default_factory=dict)
    physical_capacity: int = 0
    row_cap_by_size: dict[str, dict[str, int]] = field(default_factory=dict)
    row_physical_capacity: dict[str, int] = field(default_factory=dict)
    large_bay_partner_no: str = ""
    large_bay_partner_key: str = ""
    existing_size_modes: set[str] = field(default_factory=set)
    existing_heights: set[str] = field(default_factory=set)
    existing_special_signatures: set[str] = field(default_factory=set)
    existing_ports: set[str] = field(default_factory=set)
    existing_ports_by_row: dict[str, set[str]] = field(default_factory=dict)
    existing_attrs: dict[str, set[str]] = field(default_factory=dict)
    existing_attrs_by_row: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    existing_attrs_by_voyage: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    existing_attrs_by_row_by_voyage: dict[str, dict[str, dict[str, set[str]]]] = field(default_factory=dict)
    fallback_reasons: set[str] = field(default_factory=set)
    locked: bool = False

    @property
    def is_fallback_bay(self) -> bool:
        return bool(self.fallback_reasons)

    def capacity_for(self, group: BoxGroup) -> int:
        return int(self.cap_by_size.get(group.size_mode, 0))


@dataclass(frozen=True)
class BigPlanRow:
    """大计划输出的一行。

    `size_mode` 使用大计划口径，只允许 `20`、`40` 或 `ALL`。其中 `40`
    同时包含真实 40ft 和真实 45ft。
    """

    voyage_id: str
    flow: str
    area_no: str
    planned_boxes: int
    size_mode: str = "ALL"
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
class AreaOperation:
    area_no: str
    voyage_id: str
    start_time: datetime
    end_time: datetime


@dataclass(frozen=True)
class SmallBoxGroup:
    group_id: str
    voyage_id: str
    status: str
    port: str
    size: str
    height: str
    weight_class: str
    demand: int
    pre_stow: bool = False
    special_stow: bool = False
    special_stow_code: str = ""
    attributes: dict[str, str] = field(default_factory=dict)
    area_allowlist: set[str] | None = None


@dataclass
class ProblemData:
    """中计划和小计划共用的求解输入。

    `area_quota` and `area_size_quota` keep the big-plan area pattern used as
    medium-plan soft targets. Hard feasibility is checked against yard capacity.
    """

    groups: list[BoxGroup]
    small_groups: list[SmallBoxGroup]
    bays: dict[str, Bay]
    big_plan: list[BigPlanRow]
    assigned_areas: dict[tuple[str, str], set[str]]
    area_quota: dict[tuple[str, str, str], int]
    area_size_quota: dict[tuple[str, str, str, str], int]
    area_functions: dict[str, set[str]]
    business_special_codes: set[str]
    planning_time: datetime
    horizon_hours: float
    voyage_windows: dict[str, tuple[datetime, datetime]]
    area_operations: dict[str, list[AreaOperation]]
    target_voyages: list[str]
    export_voyages: set[str] | None = None
    existing_coarse_area_load: dict[tuple[str, ...], int] = field(default_factory=dict)
    existing_coarse_bay_load: dict[tuple[str, ...], int] = field(default_factory=dict)
    berth_distances: dict[tuple[str, str], float] = field(default_factory=dict)
    berth_by_voyage: dict[str, str] = field(default_factory=dict)
    tops_reserved_slot_count: int = 0
    tops_closed_bay_count: int = 0
    misplaced_bay_exclusion_ratio: float = 0.0
    misplaced_excluded_bay_count: int = 0
    medium_doc_floor_added_boxes: int = 0
    medium_doc_floor_added_groups: int = 0
    medium_doc_floor_shifted_boxes: int = 0
    medium_doc_floor_shifted_groups: int = 0
    medium_doc_floor_by_coarse_group: dict[str, int] = field(default_factory=dict)
    medium_doc_floor_added_by_coarse_group: dict[str, int] = field(default_factory=dict)
    medium_doc_floor_shifted_by_coarse_group: dict[str, int] = field(default_factory=dict)
    user_voyage_area_allowlist: dict[str, set[str]] = field(default_factory=dict)
    user_voyage_area_blocklist: dict[str, set[str]] = field(default_factory=dict)
    user_voyage_area_requirements: dict[str, set[str]] = field(default_factory=dict)
    user_area_constraint_summary: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    user_voyage_bay_allowlist: dict[str, set[str]] = field(default_factory=dict)
    user_group_bay_requirements: dict[str, set[str]] = field(default_factory=dict)
    user_group_bay_blocklist: dict[str, set[str]] = field(default_factory=dict)
    user_bay_adjust_rules: list[dict[str, object]] = field(default_factory=list)
    user_bay_constraint_summary: dict[str, object] = field(default_factory=dict)
    attribute_rules: AttributeRules = field(default_factory=AttributeRules)


