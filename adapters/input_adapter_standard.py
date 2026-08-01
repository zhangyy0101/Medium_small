from __future__ import annotations

import csv
import json
import math
import re
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
for path in (MODULE_DIR, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from .input_adapter_gd import (
        DEFAULT_DETAIL_ATTR,
        DEFAULT_ROUGH_ATTR,
        DEFAULT_WEIGHT_LEVEL,
        InputAdapterGd,
    )
except ImportError:  # pragma: no cover - keeps compatibility with script-style imports.
    try:
        from input_adapter_gd import (
            DEFAULT_DETAIL_ATTR,
            DEFAULT_ROUGH_ATTR,
            DEFAULT_WEIGHT_LEVEL,
            InputAdapterGd,
        )
    except ImportError:  # pragma: no cover - keeps compatibility with the production package layout.
        from adapter.input.guandong.input_adapter_gd import (
            DEFAULT_DETAIL_ATTR,
            DEFAULT_ROUGH_ATTR,
            DEFAULT_WEIGHT_LEVEL,
            InputAdapterGd,
        )

from block_bay_planning.models import AttributeRules, EXPORT_VOYAGE_ROW_NO_MIX_ATTR



DEFAULT_PLANNING_TIME = "2026-05-19 09:30:00"
DEFAULT_EXPORT_VESSELS = None
DEFAULT_IMPORT_VESSELS = None
DEFAULT_TARGET_VOYAGES = ()
DEFAULT_TARGET_BIG_PLAN_FLOWS = frozenset({"OF", "IF", "IZ", "T", "OZ"})
KNOWN_EXPORT_SNAPSHOT_FLOWS = {"OF", "OZ", "T"}
UNKNOWN_EXPORT_SNAPSHOT_FLOW_FALLBACK = "OF"
DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO = 2.0 / 3.0
SIZE_MODES = ("20", "40", "45")
WEIGHT_ATTRIBUTE_NAMES = {"weight", "weight_class", "IYC_CWEIGHT"}


@dataclass(frozen=True)
class YardPlanningWeights:
    miss: float = 100.0
    operation: float = 50.0
    of_area: float = 40.0
    distance: float = 30.0
    share: float = 20.0
    berth_conflict: float = 25.0
    adjustment: float = 10.0
    balance: float = 5.0
    priority_area: float = 0.01
    import_cluster_area: float = 5.0


@dataclass(frozen=True)
class LargePlanningConfig:
    flow_aliases: Mapping[str, str] = field(
        default_factory=lambda: {"IE": "OZ", "RF": "OZ", "RE": "OZ"}
    )
    berth_conflict_threshold_hours: float = 2.0
    weights: YardPlanningWeights = field(default_factory=YardPlanningWeights)
    required_area_penalty: float = 10000.0
    allow_unmet_demand: bool = True
    strict_validation: bool = True

    def active_flow_aliases(self, disable_default_flow_aliases: bool = False) -> dict[str, str]:
        if disable_default_flow_aliases:
            return {}
        return dict(self.flow_aliases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_aliases": dict(self.flow_aliases),
            "berth_conflict_threshold_hours": self.berth_conflict_threshold_hours,
            "weights": asdict(self.weights),
            "required_area_penalty": self.required_area_penalty,
            "allow_unmet_demand": self.allow_unmet_demand,
            "strict_validation": self.strict_validation,
        }


@dataclass
class LargePlanningData:
    V: Sequence[str]
    F: Sequence[str]
    A: Sequence[str]
    D20: Mapping[tuple[str, str], float]
    D40: Mapping[tuple[str, str], float]
    C20: Mapping[str, float]
    C40: Mapping[str, float]
    H: Mapping[str, float]
    distance: Mapping[tuple[str, str], float]
    U: Mapping[tuple[str, str], int]
    E20: Mapping[tuple[str, str, str], int]
    E40: Mapping[tuple[str, str, str], int]
    C20Direct: Mapping[str, float] | None = None
    port_area_demand20: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    port_area_demand40: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    port_area_allowlist: Mapping[str, Sequence[str]] = field(default_factory=dict)
    import_cluster_demand20: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    import_cluster_demand40: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    import_cluster_snapshot20: Mapping[tuple[str, str, str, str], float] = field(default_factory=dict)
    import_cluster_snapshot40: Mapping[tuple[str, str, str, str], float] = field(default_factory=dict)
    S20: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    S40: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    L20: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    L40: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    Q20: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    Q40: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    R20S: Mapping[tuple[str, str], float] | None = None
    R40S: Mapping[tuple[str, str], float] | None = None
    TOPS20: Mapping[tuple[str, str], float] = field(default_factory=dict)
    TOPS40: Mapping[tuple[str, str], float] = field(default_factory=dict)
    Cbar20: Mapping[tuple[str, str], float] | None = None
    Cbar20Direct: Mapping[tuple[str, str], float] | None = None
    Cbar40: Mapping[tuple[str, str], float] | None = None
    P20: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    P40: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    O: Mapping[str, int] = field(default_factory=dict)
    M: Mapping[str, float] | None = None
    OFWorkLanes: Mapping[str, float] = field(default_factory=dict)
    berth_conflict_pairs: Sequence[tuple[str, str]] = field(default_factory=tuple)
    allowed_areas_by_vessel: Mapping[str, Sequence[str]] = field(default_factory=dict)
    required_areas_by_vessel: Mapping[str, Sequence[str]] = field(default_factory=dict)
    priority_areas_by_vessel: Mapping[str, Sequence[str]] = field(default_factory=dict)
    bay_limit_areas_by_vessel: Mapping[str, Sequence[str]] = field(default_factory=dict)
    bay_limit_C20: Mapping[tuple[str, str], float] = field(default_factory=dict)
    bay_limit_C20Direct: Mapping[tuple[str, str], float] = field(default_factory=dict)
    bay_limit_C40: Mapping[tuple[str, str], float] = field(default_factory=dict)
    weights: YardPlanningWeights = field(default_factory=YardPlanningWeights)
    required_area_penalty: float = 10000.0
    allow_unmet_demand: bool = True
    strict_validation: bool = True
    default_U: int = 1
    default_E20: int = 1
    default_E40: int = 1
    default_distance: float = 0.0
    default_H: float | None = None
    name: str = "planning_large"


@dataclass(frozen=True)
class PlanningInputArtifacts:
    data: LargePlanningData
    planning_time: pd.Timestamp
    export_vessels: list[str]
    import_vessels: list[str]
    area_functions: dict[str, set[str]]
    berth_by_vessel: dict[str, str]
    previous_plan_rows: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class DemandRow:
    voyage_id: str
    flow: str
    port: str
    size_mode: str
    predicted_boxes: int
    ratio_target_boxes: int
    doc_boxes: int
    yard_boxes: int
    planned_boxes: int
    planning_stage: str
    planning_ratio: float


@dataclass(frozen=True)
class BoxGroup:
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
        return self.size if self.size in {"20", "40", "45"} else "40"

    @property
    def big_plan_size_mode(self) -> str:
        return "40" if self.size_mode == "45" else self.size_mode

    @property
    def special_signature(self) -> str:
        return self.business_special_signature

    @property
    def business_special_signature(self) -> str:
        return "+".join(self.special_codes) if self.special_codes else "NORMAL"


@dataclass
class Bay:
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
    allowed_areas_by_voyage: dict[str, set[str]] = field(default_factory=dict)
    user_voyage_area_allowlist: dict[str, set[str]] = field(default_factory=dict)
    user_voyage_area_blocklist: dict[str, set[str]] = field(default_factory=dict)
    user_voyage_area_priority: dict[str, set[str]] = field(default_factory=dict)
    user_voyage_area_requirements: dict[str, set[str]] = field(default_factory=dict)
    user_voyage_bay_allowlist: dict[str, set[str]] = field(default_factory=dict)
    user_group_bay_requirements: dict[str, set[str]] = field(default_factory=dict)
    user_group_bay_blocklist: dict[str, set[str]] = field(default_factory=dict)
    user_bay_adjust_rules: list[dict[str, Any]] = field(default_factory=list)
    user_area_constraint_summary: dict[str, Any] = field(default_factory=dict)
    user_bay_constraint_summary: dict[str, Any] = field(default_factory=dict)
    tops_reserved_slot_count: int = 0
    tops_closed_bay_count: int = 0
    misplaced_bay_exclusion_ratio: float = 0.0
    misplaced_excluded_bay_count: int = 0
    attribute_rules: AttributeRules = field(default_factory=AttributeRules)


@dataclass(frozen=True)
class MediumSmallInputs:
    big_plan: list[BigPlanRow]
    demand_rows: list[DemandRow]
    problem: ProblemData


class RollingPlanningState:
    def __init__(self, plan_history: pd.DataFrame) -> None:
        # self.state_dir = state_dir
        self.history = plan_history

    def read_history(self) -> pd.DataFrame:
        if self.history is None:
            return pd.DataFrame(
                columns=[
                    "run_id",
                    "planning_time",
                    "voy_id",
                    "flow",
                    "area_no",
                    "size",
                    "planned_qty",
                    "status_name",
                    "objective_value",
                ]
            )
        if "planning_time" in self.history.columns:
            self.history["planning_time"] = pd.to_datetime(self.history["planning_time"], errors="coerce")
        return self.history

    def latest_previous_plan(self, planning_time: pd.Timestamp, vessels: Sequence[str]) -> pd.DataFrame:
        history = self.read_history()
        if history.empty:
            return history
        vessel_set = {normalize_code(v) for v in vessels if normalize_code(v)}
        history = history[history["voy_id"].map(normalize_code).isin(vessel_set)].copy()
        history = history[history["planning_time"] < planning_time].copy()
        if history.empty:
            return history
        latest_by_vessel = history.groupby("voy_id")["planning_time"].transform("max")
        return history[history["planning_time"] == latest_by_vessel].copy()

    def build_previous_plan_params(
        self,
        planning_time: pd.Timestamp,
        vessels: Sequence[str],
    ) -> tuple[dict[tuple[str, str, str], float], dict[tuple[str, str, str], float], dict[str, int], pd.DataFrame]:
        previous = self.latest_previous_plan(planning_time, vessels)
        p20: dict[tuple[str, str, str], float] = {}
        p40: dict[tuple[str, str, str], float] = {}
        old_flags = {v: 0 for v in vessels}
        if previous.empty:
            return p20, p40, old_flags, previous
        previous = previous.copy()
        previous["voy_id"] = previous["voy_id"].map(normalize_code)
        previous["flow"] = previous["flow"].map(normalize_code)
        previous["area_no"] = previous["area_no"].map(normalize_code)
        previous["size"] = previous["size"].map(normalize_size_large)
        previous["planned_qty"] = pd.to_numeric(previous["planned_qty"], errors="coerce").fillna(0.0)
        for vessel in previous["voy_id"].dropna().unique():
            old_flags[str(vessel)] = 1
        grouped = previous.groupby(["voy_id", "flow", "area_no", "size"], dropna=False)["planned_qty"].sum()
        for (vessel, flow, area, size), qty in grouped.items():
            if not vessel or not flow or not area:
                continue
            key = (str(vessel), str(flow), str(area))
            if size == "20":
                p20[key] = float(qty)
            elif size == "40":
                p40[key] = float(qty)
        return p20, p40, old_flags, previous

    def append_solution(self, planning_time: pd.Timestamp, solution: Any) -> pd.DataFrame:
        rows = []
        run_id = uuid.uuid4().hex[:12]
        for size, values in (("20", solution.x20), ("40", solution.x40)):
            for (vessel, flow, area), qty in values.items():
                if qty <= 0:
                    continue
                rows.append(
                    {
                        "run_id": run_id,
                        "planning_time": planning_time.isoformat(),
                        "voy_id": vessel,
                        "flow": flow,
                        "area_no": area,
                        "size": size,
                        "planned_qty": int(qty),
                        "status_name": solution.status_name,
                        "objective_value": solution.objective_value,
                    }
                )
        new_rows = pd.DataFrame(rows)
        if new_rows.empty:
            return new_rows
        # self.state_dir.mkdir(parents=True, exist_ok=True)
        old = self.read_history()
        if not old.empty:
            old["planning_time"] = pd.to_datetime(old["planning_time"], errors="coerce")
            replacing_vessels = set(new_rows["voy_id"].map(normalize_code))
            old = old[
                ~(
                    old["planning_time"].eq(planning_time)
                    & old["voy_id"].map(normalize_code).isin(replacing_vessels)
                )
            ].copy()
        combined = pd.concat([old, new_rows], ignore_index=True)
        # combined.to_csv(self.plan_history_path, index=False, encoding="utf-8-sig")
        return new_rows


def parse_datetime(value: object) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return pd.to_datetime(value, unit="D", origin="1899-12-30").to_pydatetime()
        except (ValueError, TypeError, OverflowError):
            return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        return parsed.to_pydatetime()
    return None


def parse_planning_time(value: str) -> pd.Timestamp:
    planning_time = pd.Timestamp(value)
    if pd.isna(planning_time):
        raise ValueError(f"Invalid planning time: {value}")
    return planning_time


def normalize_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return ""
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def normalize_text(value: Any, default: str = "") -> str:
    text = normalize_code(value)
    return text or default


def normalize_voyage(value: Any, fallback: str = "") -> str:
    text = normalize_text(value, fallback)
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def normalize_size_large(value: Any) -> str:
    code = normalize_code(value)
    if code.startswith("20"):
        return "20"
    if code.startswith(("40", "45")):
        return "40"
    return ""


def normalize_size_small(value: Any) -> str:
    code = normalize_code(value)
    return code if code in {"20", "40", "45"} else "40"


def normalize_big_plan_size(value: Any) -> str:
    code = normalize_code(value)
    if code == "ALL":
        return "ALL"
    return normalize_size_small(code) if code else "ALL"


def normalize_flow(value: Any, aliases: Mapping[str, str] | None = None, default: str = "") -> str:
    flow = normalize_code(value)
    if not flow:
        return default
    return (aliases or {}).get(flow, flow)


def normalize_medium_small_flow(value: Any, default: str = "OF") -> str:
    return normalize_flow(value, default=default)


def medium_small_area_flow(flow: Any) -> str:
    normalized = normalize_medium_small_flow(flow, default="OF")
    if normalized == "OF":
        return "OF"
    if normalized in {"IF", "IZ", "T"}:
        return normalized
    return "OZ"


def area_allows_flow(area: Any, flow: Any, area_functions: Mapping[str, set[str]]) -> bool:
    area_code = normalize_code(area)
    flow_code = normalize_code(flow)
    if not area_code or not flow_code:
        return False
    return flow_code in area_functions.get(area_code, set())


def normalize_export_snapshot_flow(value: Any, aliases: Mapping[str, str] | None = None) -> str:
    return "OF"


def has_import_voyage(value: Any) -> bool:
    return bool(normalize_voyage(value))


def planning_excluded_mask(rows: pd.DataFrame) -> pd.Series:
    if rows.empty or "planning_excluded" not in rows.columns:
        return pd.Series(False, index=rows.index)
    return rows["planning_excluded"].fillna(False).astype(bool)


def planning_included_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    return rows.loc[~planning_excluded_mask(rows)].copy()


def yard_transshipment_mask(rows: pd.DataFrame) -> pd.Series:
    if rows.empty or "IYC_EVOY_ID" not in rows.columns or "IYC_IVOY_ID" not in rows.columns:
        return pd.Series(False, index=rows.index)
    export_voyage = rows["IYC_EVOY_ID"].map(lambda value: bool(normalize_voyage(value)))
    import_voyage = rows["IYC_IVOY_ID"].map(lambda value: bool(normalize_voyage(value)))
    return export_voyage & import_voyage


def medium_small_yard_included_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    return rows.loc[~yard_transshipment_mask(rows)].copy()


def normalize_voyage_list(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        voyage = normalize_voyage(value)
        if not voyage or voyage in seen:
            continue
        seen.add(voyage)
        out.append(voyage)
    return out


def normalize_area_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    seen: set[str] = set()
    try:
        iterator = list(values)
    except TypeError:
        iterator = [values]
    for value in iterator:
        area = normalize_code(value)
        if not area or area in seen:
            continue
        seen.add(area)
        out.append(area)
    return out


def normalize_area(value: Any) -> str:
    return normalize_text(value, "")


def read_port_sail_area(input_guandong: InputAdapterGd) -> dict[str, set[str]]:
    raw = getattr(input_guandong, "port_sail_area", {}) or {}
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, set[str]] = {}
    for port, areas in raw.items():
        port_code = normalize_text(port)
        area_set = set(normalize_area_list(areas))
        if port_code and area_set:
            out[port_code] = area_set
    return out


def canonical_attribute_tuple(values: Any, default: Sequence[str]) -> tuple[str, ...]:
    if values is None:
        values = default
    if isinstance(values, str):
        values = [values]
    try:
        iterator = list(values)
    except TypeError:
        iterator = [values]
    out: list[str] = []
    seen: set[str] = set()
    for value in iterator:
        raw = "" if value is None else str(value).strip()
        if not raw:
            continue
        if raw not in seen:
            seen.add(raw)
            out.append(raw)
    return tuple(out) or tuple(default)


def attribute_output_name(attr: object) -> str:
    return "" if attr is None else str(attr).strip()


def is_size_no_mix_attribute(attr: object) -> bool:
    return attribute_output_name(attr).upper() in {"IYC_CSZ_CSIZECD", "SIZE", "SIZE_MODE"}


def is_weight_attribute(attr: object) -> bool:
    raw = attribute_output_name(attr)
    return raw in WEIGHT_ATTRIBUTE_NAMES or raw.lower() in {item.lower() for item in WEIGHT_ATTRIBUTE_NAMES}


def raw_attribute_text(value: Any, default: str = "MIXED") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def dynamic_attribute_value(
    row: Mapping[str, Any],
    attr: object,
    *,
    levels: Sequence[int] = DEFAULT_WEIGHT_LEVEL,
    default: str = "MIXED",
) -> str:
    raw = attribute_output_name(attr)
    if not raw:
        return default
    if is_weight_attribute(raw):
        return weight_class(row.get(raw), levels)
    return raw_attribute_text(row.get(raw), default)


def dynamic_attributes_from_row(
    row: Mapping[str, Any],
    attrs: Sequence[str],
    *,
    levels: Sequence[int] = DEFAULT_WEIGHT_LEVEL,
    default: str = "MIXED",
) -> dict[str, str]:
    out: dict[str, str] = {}
    for attr in attrs:
        name = attribute_output_name(attr)
        if name:
            out[name] = dynamic_attribute_value(row, name, levels=levels, default=default)
    return out


def canonical_weight_levels(values: Any, default: Sequence[int] = DEFAULT_WEIGHT_LEVEL) -> tuple[int, ...]:
    if values is None:
        values = default
    if isinstance(values, str):
        values = re.split(r"[,|;/\s]+", values.strip())
    try:
        iterator = list(values)
    except TypeError:
        iterator = [values]
    levels: list[int] = []
    for value in iterator:
        try:
            levels.append(int(round(float(value))))
        except (TypeError, ValueError):
            continue
    levels = sorted(set(levels))
    return tuple(levels) if levels else tuple(int(value) for value in default)


def has_rule_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(has_rule_content(item) for item in value.values())
    try:
        iterator = list(value)
    except TypeError:
        return True
    return any(has_rule_content(item) for item in iterator)


def voyage_rule_map(
    value: Any,
    voyages: Sequence[str],
    default: Sequence[Any],
    canonicalizer,
    *,
    fill_missing: bool = True,
) -> dict[str, tuple]:
    normalized_voyages = [normalize_voyage(voyage) for voyage in voyages if normalize_voyage(voyage)]
    if isinstance(value, Mapping):
        out = {
            normalize_voyage(voyage): canonicalizer(rules, default)
            for voyage, rules in value.items()
            if normalize_voyage(voyage) and (fill_missing or has_rule_content(rules))
        }
    elif value is None:
        out = {}
    else:
        out = {}
        if fill_missing or has_rule_content(value):
            shared = canonicalizer(value, default)
            out = {voyage: shared for voyage in normalized_voyages}
    if fill_missing:
        for voyage in normalized_voyages:
            out.setdefault(voyage, canonicalizer(None, default))
    return out


def read_attribute_rules(input_guandong: InputAdapterGd, voyages: Sequence[str]) -> AttributeRules:
    raw_weight_level = getattr(input_guandong, "weight_level", None)
    weight_levels_by_voyage = voyage_rule_map(
        raw_weight_level,
        voyages,
        DEFAULT_WEIGHT_LEVEL,
        canonical_weight_levels,
        fill_missing=False,
    )
    coarse_group_attributes_by_voyage = voyage_rule_map(
        getattr(input_guandong, "rough_attr", None),
        voyages,
        DEFAULT_ROUGH_ATTR,
        canonical_attribute_tuple,
    )
    fine_group_attributes_by_voyage = voyage_rule_map(
        getattr(input_guandong, "detail_attr", None),
        voyages,
        DEFAULT_DETAIL_ATTR,
        canonical_attribute_tuple,
    )
    export_voyages = classified_export_voyages(input_guandong)
    import_voyages = [
        normalize_voyage(voyage)
        for voyage in voyages
        if normalize_voyage(voyage) and normalize_voyage(voyage) not in export_voyages
    ]
    import_shared_fine_group_attributes = common_attribute_intersection(
        fine_group_attributes_by_voyage.get(voyage, canonical_attribute_tuple(DEFAULT_DETAIL_ATTR, AttributeRules().fine_group_attributes))
        for voyage in import_voyages
    )
    return AttributeRules(
        coarse_group_attributes=canonical_attribute_tuple(DEFAULT_ROUGH_ATTR, AttributeRules().coarse_group_attributes),
        fine_group_attributes=canonical_attribute_tuple(DEFAULT_DETAIL_ATTR, AttributeRules().fine_group_attributes),
        bay_no_mix_attributes=(),
        row_no_mix_attributes=(),
        weight_levels=canonical_weight_levels(DEFAULT_WEIGHT_LEVEL),
        coarse_group_attributes_by_voyage=coarse_group_attributes_by_voyage,
        fine_group_attributes_by_voyage=fine_group_attributes_by_voyage,
        bay_no_mix_attributes_by_voyage=voyage_rule_map(
            getattr(input_guandong, "bay_rules", None),
            voyages,
            (),
            canonical_attribute_tuple,
            fill_missing=False,
        ),
        row_no_mix_attributes_by_voyage=voyage_rule_map(
            getattr(input_guandong, "row_rules", None),
            voyages,
            (),
            canonical_attribute_tuple,
            fill_missing=False,
        ),
        weight_levels_by_voyage=weight_levels_by_voyage,
        weight_group_voyages=frozenset(weight_levels_by_voyage),
        import_shared_fine_group_attributes=import_shared_fine_group_attributes,
    )


def common_attribute_intersection(values: Iterable[Sequence[str]]) -> tuple[str, ...]:
    sequences = [tuple(str(attr) for attr in attrs if str(attr)) for attrs in values]
    if not sequences:
        return ()
    common = set(sequences[0])
    for attrs in sequences[1:]:
        common.intersection_update(attrs)
    return tuple(attr for attr in sequences[0] if attr in common)


def small_groupby_columns(attribute_rules: AttributeRules, voyage_id: str) -> tuple[str, ...]:
    attrs: list[str] = []
    attrs.extend(attribute_rules.coarse_for(voyage_id))
    attrs.extend(attribute_rules.fine_for(voyage_id))
    # Keep no-mix attributes on the planning groups so bay/row compatibility
    # can be evaluated even when they are not part of the user fine group.
    attrs.extend(attribute_rules.bay_no_mix_for(voyage_id))
    attrs.extend(attribute_rules.row_no_mix_for(voyage_id))
    if getattr(attribute_rules, "weight_group_enabled_for", lambda _voyage_id: False)(voyage_id):
        attrs.append("IYC_CWEIGHT")
    columns: list[str] = []
    for attr in attrs:
        column = attribute_output_name(attr)
        if column and column not in columns:
            columns.append(column)
    return tuple(columns)


def medium_groupby_attributes(attribute_rules: AttributeRules, voyage_id: str) -> tuple[str, ...]:
    attrs: list[str] = []
    attrs.extend(attribute_rules.coarse_for(voyage_id))
    out: list[str] = []
    for attr in attrs:
        name = attribute_output_name(attr)
        if name and name not in out:
            out.append(name)
    return tuple(out)


def unique_attribute_names(attrs: Sequence[object]) -> tuple[str, ...]:
    out: list[str] = []
    for attr in attrs:
        name = attribute_output_name(attr)
        if name and name not in out:
            out.append(name)
    return tuple(out)


def import_base_group_attributes(row: Mapping[str, Any], size_mode: str, port: str) -> tuple[tuple[str, ...], dict[str, str], str]:
    evoy = normalize_voyage(row.get("IYC_EVOY_ID"))
    if evoy:
        return (
            ("IYC_CSZ_CSIZECD", "IYC_EVOY_ID"),
            {"IYC_CSZ_CSIZECD": size_mode, "IYC_EVOY_ID": evoy},
            "MIXED",
        )
    return (
        ("IYC_CSZ_CSIZECD", "IYC_POT_UNLDPORT"),
        {"IYC_CSZ_CSIZECD": size_mode, "IYC_POT_UNLDPORT": port},
        port,
    )


def import_small_groupby_columns(attribute_rules: AttributeRules, voyage_id: str, row: Mapping[str, Any], size_mode: str, port: str) -> tuple[str, ...]:
    base_attrs, _base_values, _port_label = import_base_group_attributes(row, size_mode, port)
    attrs: list[str] = list(base_attrs)
    attrs.extend(getattr(attribute_rules, "import_shared_fine_group_attributes", ()))
    # These are not part of the user fine group; they are carried only so the
    # no-mix constraints have a concrete value to compare.
    attrs.extend(attribute_rules.bay_no_mix_for(voyage_id))
    attrs.extend(attribute_rules.row_no_mix_for(voyage_id))
    if getattr(attribute_rules, "weight_group_enabled_for", lambda _voyage_id: False)(voyage_id):
        attrs.append("IYC_CWEIGHT")
    return unique_attribute_names(attrs)


def normalized_doc_record(row: Mapping[str, Any], flow: str, size_mode: str, port: str) -> dict[str, Any]:
    record = dict(row)
    record["IYC_STS_CSTATUSCD"] = flow
    record["IYC_CSZ_CSIZECD"] = size_mode
    record["IYC_POT_UNLDPORT"] = port
    evoy = normalize_voyage(row.get("IYC_EVOY_ID"))
    if evoy:
        record["IYC_EVOY_ID"] = evoy
    return record


def large_plan_adjust_entry(adjust_plan_info: Mapping[str, Any] | None, vessel: str) -> Mapping[str, Any]:
    if not isinstance(adjust_plan_info, Mapping):
        return {}
    source = adjust_plan_info.get("adjust_plan_info", adjust_plan_info)
    if not isinstance(source, Mapping):
        return {}
    large_plan = source.get("large_plan", source)
    if not isinstance(large_plan, Mapping):
        return {}
    if "add" in large_plan or "remove" in large_plan:
        return large_plan
    return (
        large_plan.get(vessel)
        or large_plan.get(str(vessel))
        or large_plan.get(normalize_voyage(vessel))
        or {}
    )


def _voyage_control_value(raw: Any, vessel: str) -> tuple[Any, bool]:
    if raw is None:
        return None, False
    if isinstance(raw, Mapping):
        for key in (vessel, str(vessel), normalize_voyage(vessel)):
            if key in raw:
                return raw[key], True
        return None, False
    return raw, True


def _voyage_control_bool(raw: Any, vessel: str) -> bool:
    value, present = _voyage_control_value(raw, vessel)
    return bool(value) if present else False


def _voyage_control_area_list(raw: Any, vessel: str) -> tuple[list[str], bool]:
    value, present = _voyage_control_value(raw, vessel)
    return normalize_area_list(value), present


def build_large_area_controls(
    *,
    vessels: Sequence[str],
    areas: Sequence[str],
    user_design: bool | Mapping[str, Any],
    user_design_large_plan_area: Sequence[str] | Mapping[str, Any] | None,
    voyage_limit_areas: Sequence[str] | Mapping[str, Any] | None = None,
    voyage_priority_areas: Sequence[str] | Mapping[str, Any] | None = None,
    adjust_plan_info: Mapping[str, Any] | None,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], dict[str, Any]]:
    area_set = set(areas)
    allowed_by_vessel: dict[str, list[str]] = {}
    required_by_vessel: dict[str, list[str]] = {}
    priority_by_vessel: dict[str, list[str]] = {}
    per_vessel: dict[str, dict[str, Any]] = {}
    adjust_unknown_areas: dict[str, list[str]] = {}
    design_ignored: dict[str, list[str]] = {}
    limit_ignored: dict[str, list[str]] = {}
    priority_ignored: dict[str, list[str]] = {}
    design_by_vessel: dict[str, list[str]] = {}
    limit_by_vessel: dict[str, list[str]] = {}
    raw_priority_by_vessel: dict[str, list[str]] = {}
    active_design_vessels: list[str] = []
    active_limit_vessels: list[str] = []

    for vessel in vessels:
        design_requested = _voyage_control_bool(user_design, vessel)
        design_areas_raw, _design_present = _voyage_control_area_list(user_design_large_plan_area, vessel)
        design_areas = [area for area in design_areas_raw if area in area_set]
        active_user_design = bool(design_requested and design_areas)

        limit_areas_raw, limit_present = _voyage_control_area_list(voyage_limit_areas, vessel)
        limit_areas = [area for area in limit_areas_raw if area in area_set]
        active_limit = bool(limit_present and limit_areas and not active_user_design)

        priority_areas_raw, priority_present = _voyage_control_area_list(voyage_priority_areas, vessel)
        priority_areas_valid = [area for area in priority_areas_raw if area in area_set]

        entry = large_plan_adjust_entry(adjust_plan_info, vessel)
        add_raw = normalize_area_list(entry.get("add")) if isinstance(entry, Mapping) else []
        remove_raw = normalize_area_list(entry.get("remove")) if isinstance(entry, Mapping) else []
        add = [area for area in add_raw if area in area_set]
        remove = [area for area in remove_raw if area in area_set]

        unknown = sorted((set(add_raw) | set(remove_raw)) - area_set)
        if unknown:
            adjust_unknown_areas[vessel] = unknown
        if set(design_areas_raw) - area_set:
            design_ignored[vessel] = sorted(set(design_areas_raw) - area_set)
        if set(limit_areas_raw) - area_set:
            limit_ignored[vessel] = sorted(set(limit_areas_raw) - area_set)
        if set(priority_areas_raw) - area_set:
            priority_ignored[vessel] = sorted(set(priority_areas_raw) - area_set)

        if active_user_design:
            allowed = set(design_areas)
            active_design_vessels.append(vessel)
        elif active_limit:
            allowed = set(limit_areas)
            active_limit_vessels.append(vessel)
        else:
            allowed = set(areas)
        allowed.difference_update(remove)
        required = sorted(set(add) & allowed)
        priority_areas = sorted(set(priority_areas_valid) & allowed) if (priority_present and not active_user_design) else []

        allowed_by_vessel[vessel] = sorted(allowed)
        if required:
            required_by_vessel[vessel] = required
        if priority_areas:
            priority_by_vessel[vessel] = priority_areas
        if design_areas:
            design_by_vessel[vessel] = design_areas
        if active_limit:
            limit_by_vessel[vessel] = limit_areas
        if priority_areas_valid:
            raw_priority_by_vessel[vessel] = priority_areas_valid
        per_vessel[vessel] = {
            "allowed_count": len(allowed),
            "allowed_areas": sorted(allowed),
            "user_design_requested": design_requested,
            "user_design_active": active_user_design,
            "user_design_large_plan_area": design_areas,
            "limit_active": active_limit,
            "limit_areas": limit_areas if active_limit else [],
            "priority_areas": priority_areas,
            "required_areas": required,
            "removed_areas": sorted(set(remove)),
            "add_areas_ignored_by_allowed_scope": sorted(set(add) - allowed),
        }

    diagnostics = {
        "user_design_requested": bool(active_design_vessels) or any(
            _voyage_control_bool(user_design, vessel) for vessel in vessels
        ),
        "user_design_active": bool(active_design_vessels),
        "user_design_active_vessels": active_design_vessels,
        "user_design_large_plan_area": sorted(set().union(*(set(values) for values in design_by_vessel.values()))) if design_by_vessel else [],
        "user_design_large_plan_area_by_vessel": design_by_vessel,
        "user_design_large_plan_area_ignored": design_ignored,
        "voyage_limit_areas_active_vessels": active_limit_vessels,
        "voyage_limit_areas_by_vessel": limit_by_vessel,
        "voyage_limit_areas_ignored": limit_ignored,
        "voyage_priority_areas_by_vessel": priority_by_vessel,
        "voyage_priority_areas_requested_by_vessel": raw_priority_by_vessel,
        "voyage_priority_areas_ignored": priority_ignored,
        "adjust_plan_info_large_plan_present": bool(
            isinstance(adjust_plan_info, Mapping) and adjust_plan_info.get("large_plan", adjust_plan_info)
        ),
        "adjust_plan_unknown_areas": adjust_unknown_areas,
        "area_controls_by_vessel": per_vessel,
    }
    return allowed_by_vessel, required_by_vessel, priority_by_vessel, diagnostics


def build_medium_small_area_controls(
    *,
    vessels: Sequence[str],
    areas: Sequence[str],
    user_design_large_plan_area: Sequence[str] | Mapping[str, Any] | None,
    voyage_limit_areas: Sequence[str] | Mapping[str, Any] | None = None,
    voyage_priority_areas: Sequence[str] | Mapping[str, Any] | None = None,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, Any]]:
    """Build hard per-voyage area scopes for medium/small planning.

    Medium/small planning treats both user-design areas and voyage-limit areas
    as hard boundaries.  A more specific user-design area list wins over the
    broader voyage limit when both are supplied for the same voyage.
    """
    area_set = set(areas)
    default_allowed = set(area_set)
    allowed_by_voyage: dict[str, set[str]] = {}
    per_voyage: dict[str, dict[str, Any]] = {}
    design_active_vessels: list[str] = []
    limit_active_vessels: list[str] = []
    design_ignored: dict[str, list[str]] = {}
    limit_ignored: dict[str, list[str]] = {}
    priority_by_voyage: dict[str, set[str]] = {}
    priority_ignored: dict[str, list[str]] = {}
    priority_outside_allowed: dict[str, list[str]] = {}

    for raw_vessel in vessels:
        vessel = normalize_voyage(raw_vessel)
        design_raw, design_present = _voyage_control_area_list(user_design_large_plan_area, vessel)
        design_valid = [area for area in design_raw if area in area_set]
        design_active = bool(design_present and design_raw)

        limit_raw, limit_present = _voyage_control_area_list(voyage_limit_areas, vessel)
        limit_valid = [area for area in limit_raw if area in area_set]
        limit_active = bool(limit_present and limit_raw and not design_active)
        priority_raw, priority_present = _voyage_control_area_list(voyage_priority_areas, vessel)
        priority_valid = [area for area in priority_raw if area in area_set]

        if set(design_raw) - area_set:
            design_ignored[vessel] = sorted(set(design_raw) - area_set)
        if set(limit_raw) - area_set:
            limit_ignored[vessel] = sorted(set(limit_raw) - area_set)
        if set(priority_raw) - area_set:
            priority_ignored[vessel] = sorted(set(priority_raw) - area_set)

        if design_active:
            allowed = set(design_valid)
            source = "user_design_large_plan_area"
            design_active_vessels.append(vessel)
        elif limit_active:
            allowed = set(limit_valid)
            source = "voyage_limit_areas"
            limit_active_vessels.append(vessel)
        else:
            allowed = set(default_allowed)
            source = "default_all_function_areas"
        priority_allowed = set(priority_valid) & allowed if priority_present else set()
        if priority_allowed:
            priority_by_voyage[vessel] = priority_allowed
        if priority_present and set(priority_valid) - allowed:
            priority_outside_allowed[vessel] = sorted(set(priority_valid) - allowed)

        allowed_by_voyage[vessel] = allowed
        per_voyage[vessel] = {
            "source": source,
            "allowed_count": len(allowed),
            "allowed_areas": sorted(allowed),
            "priority_areas": sorted(priority_allowed),
            "priority_areas_requested": priority_raw,
            "user_design_large_plan_area_requested": design_raw,
            "user_design_large_plan_area_valid": design_valid,
            "voyage_limit_areas_requested": limit_raw,
            "voyage_limit_areas_valid": limit_valid,
            "strict_boundary": source != "default_all_function_areas",
        }

    diagnostics = {
        "strict_area_boundary_enabled": bool(design_active_vessels or limit_active_vessels),
        "user_design_large_plan_area_active_vessels": design_active_vessels,
        "voyage_limit_areas_active_vessels": limit_active_vessels,
        "user_design_large_plan_area_ignored": design_ignored,
        "voyage_limit_areas_ignored": limit_ignored,
        "voyage_priority_areas_by_voyage": {
            voyage_id: sorted(areas)
            for voyage_id, areas in sorted(priority_by_voyage.items())
        },
        "voyage_priority_areas_ignored": priority_ignored,
        "voyage_priority_areas_outside_allowed": priority_outside_allowed,
        "area_controls_by_voyage": per_voyage,
    }
    return allowed_by_voyage, priority_by_voyage, diagnostics


def build_medium_small_bay_controls(
    input_guandong: InputAdapterGd,
    groups: Sequence[BoxGroup],
    small_groups: Sequence[SmallBoxGroup],
    bays: Mapping[str, Bay],
) -> tuple[dict[str, set[str]], dict[str, set[str]], list[dict[str, Any]], dict[str, Any]]:
    adjust_plan_info = getattr(input_guandong, "adjust_plan_info", {})
    required: defaultdict[str, set[str]] = defaultdict(set)
    blocked: defaultdict[str, set[str]] = defaultdict(set)
    rule_records: list[dict[str, Any]] = []
    summary = {
        "matched_rules": 0,
        "matched_groups": 0,
        "unknown_bays": [],
        "ignored_rules": 0,
    }

    group_by_voyage: defaultdict[str, list[Any]] = defaultdict(list)
    for group in list(groups) + list(small_groups):
        group_by_voyage[normalize_voyage(getattr(group, "voyage_id", ""))].append(group)

    for plan_level in ("medium_plan", "small_plan"):
        for voyage_id, rules in _plan_adjust_rules(adjust_plan_info, plan_level).items():
            normalized_voyage = normalize_voyage(voyage_id)
            if isinstance(rules, Mapping):
                rules = [rules]
            if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
                summary["ignored_rules"] += 1
                continue
            for rule in rules:
                if not isinstance(rule, Mapping):
                    summary["ignored_rules"] += 1
                    continue
                attr_filter = _canonical_adjust_attributes(rule.get("attribute", {}))
                add_bays, add_unknown = _canonical_adjust_bays(rule.get("add"), bays)
                remove_bays, remove_unknown = _canonical_adjust_bays(rule.get("remove"), bays)
                rule_records.append(
                    {
                        "plan_level": plan_level,
                        "voyage_id": normalized_voyage,
                        "attributes": dict(attr_filter),
                        "required_bays": set(add_bays),
                        "blocked_bays": set(remove_bays),
                    }
                )
                matched_groups = [
                    group
                    for group in group_by_voyage.get(normalized_voyage, [])
                    if _group_matches_adjust_attributes(group, attr_filter)
                ]
                if not matched_groups:
                    continue
                summary["matched_rules"] += 1
                summary["matched_groups"] += len(matched_groups)
                for group in matched_groups:
                    required[group.group_id].update(add_bays)
                    blocked[group.group_id].update(remove_bays)
                for item in add_unknown + remove_unknown:
                    summary["unknown_bays"].append(
                        {"plan_level": plan_level, "voyage_id": normalized_voyage, "bay": item}
                    )

    cleaned_required = {group_id: bays - blocked.get(group_id, set()) for group_id, bays in required.items()}
    cleaned_required = {group_id: values for group_id, values in cleaned_required.items() if values}
    cleaned_blocked = {group_id: values for group_id, values in blocked.items() if values}
    summary["required_group_count"] = len(cleaned_required)
    summary["blocked_group_count"] = len(cleaned_blocked)
    summary["required_bay_count"] = sum(len(values) for values in cleaned_required.values())
    summary["blocked_bay_count"] = sum(len(values) for values in cleaned_blocked.values())
    return cleaned_required, cleaned_blocked, rule_records, summary


def build_user_design_area_bay_allowlist(
    input_guandong: InputAdapterGd,
    bays: Mapping[str, Bay],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    raw = getattr(input_guandong, "user_design_area_bay", {}) or {}
    allowlist: dict[str, set[str]] = {}
    unknown: list[dict[str, str]] = []
    if not isinstance(raw, Mapping):
        return {}, {"configured_voyages": [], "unknown_bays": [], "ignored": True}

    for voyage_id, area_bays in raw.items():
        voyage = normalize_voyage(voyage_id)
        if not voyage or not isinstance(area_bays, Mapping):
            continue
        allowed: set[str] = set()
        for area_no, bay_values in area_bays.items():
            area = normalize_area(area_no)
            values = bay_values if isinstance(bay_values, (list, tuple, set)) else [bay_values]
            for value in values:
                bay_no = normalize_bay(value)
                matches = {
                    key
                    for key, bay in bays.items()
                    if normalize_area(bay.area_no) == area and normalize_bay(bay.bay_no) == bay_no
                }
                if matches:
                    allowed.update(matches)
                else:
                    unknown.append({"voyage_id": voyage, "area_no": area, "bay_no": bay_no})
        allowlist[voyage] = allowed
    return allowlist, {
        "configured_voyages": sorted(allowlist),
        "allowed_bay_count": sum(len(values) for values in allowlist.values()),
        "unknown_bays": unknown,
        "ignored": False,
    }


def build_user_design_area_bay_large_limits(
    input_guandong: InputAdapterGd,
    available_slots: pd.DataFrame,
    vessels: Sequence[str],
    areas: Sequence[str],
) -> tuple[
    dict[str, list[str]],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[str, Any],
]:
    raw = getattr(input_guandong, "user_design_area_bay", {}) or {}
    if not isinstance(raw, Mapping):
        return {}, {}, {}, {}, {"configured_voyages": [], "ignored": True, "unknown_bays": []}

    vessel_set = {normalize_voyage(vessel) for vessel in vessels}
    area_set = {normalize_area(area) for area in areas}
    all_slot_pairs: set[tuple[str, str]] = set()
    slot_source = getattr(input_guandong, "bay_slots_detail", None)
    if isinstance(slot_source, pd.DataFrame) and not slot_source.empty:
        all_slot_pairs = set(
            zip(
                slot_source.get("YAA_AREANO", pd.Series(dtype=object)).map(normalize_area),
                slot_source.get("YBY_BAYNO", pd.Series(dtype=object)).map(normalize_bay),
            )
        )

    selected_bays: dict[tuple[str, str], set[str]] = {}
    areas_by_vessel: dict[str, set[str]] = {}
    unknown_bays: list[dict[str, str]] = []
    for voyage_id, area_bays in raw.items():
        voyage = normalize_voyage(voyage_id)
        if not voyage or voyage not in vessel_set or not isinstance(area_bays, Mapping):
            continue
        for area_no, bay_values in area_bays.items():
            area = normalize_area(area_no)
            values = bay_values if isinstance(bay_values, (list, tuple, set)) else [bay_values]
            if not area or area not in area_set:
                for value in values:
                    unknown_bays.append(
                        {"voyage_id": voyage, "area_no": area, "bay_no": normalize_bay(value)}
                    )
                continue
            areas_by_vessel.setdefault(voyage, set()).add(area)
            target_bays = selected_bays.setdefault((voyage, area), set())
            for value in values:
                bay = normalize_bay(value)
                if not bay:
                    continue
                target_bays.add(bay)
                if all_slot_pairs and (area, bay) not in all_slot_pairs:
                    unknown_bays.append({"voyage_id": voyage, "area_no": area, "bay_no": bay})

    cap20: dict[tuple[str, str], float] = {}
    cap20_direct: dict[tuple[str, str], float] = {}
    cap40: dict[tuple[str, str], float] = {}
    for (voyage, area), bay_set in selected_bays.items():
        sub = available_slots[
            available_slots["area_no"].map(normalize_area).eq(area)
            & available_slots["bay_no"].map(normalize_bay).isin(bay_set)
        ].copy()
        cap20[voyage, area] = float(sub["slot_uid"].nunique()) if "slot_uid" in sub else 0.0
        cap20_direct[voyage, area] = (
            float(sub.loc[sub["enable_20"], "slot_uid"].nunique())
            if "slot_uid" in sub and "enable_20" in sub
            else 0.0
        )
        cap40[voyage, area] = (
            float(sub.loc[sub["enable_40"], "slot_uid"].nunique())
            if "slot_uid" in sub and "enable_40" in sub
            else 0.0
        )

    area_lists = {voyage: sorted(values) for voyage, values in sorted(areas_by_vessel.items())}
    summary = {
        "configured_voyages": sorted(area_lists),
        "limited_area_by_vessel": area_lists,
        "selected_bay_count_by_vessel_area": {
            f"{voyage}|{area}": len(bays)
            for (voyage, area), bays in sorted(selected_bays.items())
        },
        "capacity20_by_vessel_area": {f"{v}|{a}": cap20[v, a] for v, a in sorted(cap20)},
        "capacity20_direct_by_vessel_area": {
            f"{v}|{a}": cap20_direct[v, a] for v, a in sorted(cap20_direct)
        },
        "capacity40_by_vessel_area": {f"{v}|{a}": cap40[v, a] for v, a in sorted(cap40)},
        "unknown_bays": unknown_bays,
        "ignored": False,
    }
    return area_lists, cap20, cap20_direct, cap40, summary


def _plan_adjust_rules(adjust_plan_info: Mapping[str, Any] | None, plan_level: str) -> Mapping[str, Any]:
    if not isinstance(adjust_plan_info, Mapping):
        return {}
    source = adjust_plan_info.get("adjust_plan_info", adjust_plan_info)
    if not isinstance(source, Mapping):
        return {}
    rules = source.get(plan_level, {})
    return rules if isinstance(rules, Mapping) else {}


def _canonical_adjust_attributes(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        attr = canonical_attribute_tuple([key], [str(key)])[0]
        out[attr] = _normalize_adjust_attr_value(attr, value)
    return out


def _group_matches_adjust_attributes(group: Any, attr_filter: Mapping[str, str]) -> bool:
    for attr, expected in attr_filter.items():
        attrs = getattr(group, "attributes", {}) or {}
        actual = _normalize_adjust_attr_value(attr, attrs.get(attr, ""))
        if actual != expected:
            return False
    return True


def _normalize_adjust_attr_value(attr: str, value: Any) -> str:
    if is_weight_attribute(attr):
        return weight_class(value)
    return raw_attribute_text(value, "")


def _canonical_adjust_bays(raw: Any, bays: Mapping[str, Bay]) -> tuple[set[str], list[str]]:
    out: set[str] = set()
    unknown: list[str] = []
    if raw is None:
        return out, unknown
    items: list[tuple[str, Any]] = []
    if isinstance(raw, Mapping):
        items = [(normalize_area(area), bay_values) for area, bay_values in raw.items()]
    else:
        items = [("", raw)]
    for area_no, bay_values in items:
        values = bay_values if isinstance(bay_values, Sequence) and not isinstance(bay_values, (str, bytes)) else [bay_values]
        for value in values:
            key = _canonical_adjust_bay_key(area_no, value, bays)
            if key:
                out.add(key)
            else:
                unknown.append(f"{area_no}|{value}" if area_no else str(value))
    return out, unknown


def _canonical_adjust_bay_key(area_no: str, bay_value: Any, bays: Mapping[str, Bay]) -> str:
    text = normalize_text(bay_value, "")
    if not text:
        return ""
    candidates = []
    if "|" in text:
        candidates.append(text)
    if "-" in text:
        left, right = text.split("-", 1)
        candidates.append(f"{normalize_area(left)}|{normalize_text(right, '')}")
    if area_no:
        candidates.append(f"{area_no}|{text}")
    candidates.append(text)
    for candidate in candidates:
        if candidate in bays:
            return candidate
    return ""


def _adapter_vessel_items(input_guandong: InputAdapterGd) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for raw_voyage, content in (input_guandong.vessel_containers or {}).items():
        voyage = normalize_voyage(raw_voyage)
        if not voyage or not isinstance(content, dict):
            continue
        items.append((voyage, content))
    return items


def _has_container_frame(content: Mapping[str, Any]) -> bool:
    frame = content.get("doc_cntrs")
    return isinstance(frame, pd.DataFrame) and not frame.empty


def _take_over_vessels(input_guandong: InputAdapterGd, direction: str) -> list[str]:
    take_over = getattr(input_guandong, "take_over_vessel", {}) or {}
    if not isinstance(take_over, Mapping):
        return []
    return normalize_voyage_list(take_over.get(direction, []))


def discover_export_vessels(input_guandong: InputAdapterGd) -> list[str]:
    take_over_exports = _take_over_vessels(input_guandong, "E")
    if take_over_exports:
        return [
            vessel
            for vessel in take_over_exports
            if isinstance(input_guandong.vessel_containers.get(vessel, {}), Mapping)
            and normalize_code(input_guandong.vessel_containers.get(vessel, {}).get("type")) != "I"
            and (
                _has_container_frame(input_guandong.vessel_containers.get(vessel, {}))
                or bool(
                    input_guandong.vessel_containers.get(vessel, {}).get("predict_cntrs")
                    or input_guandong.vessel_containers.get(vessel, {}).get("cntr_volume")
                )
            )
        ]
    vessels: set[str] = set()
    for voyage, content in _adapter_vessel_items(input_guandong):
        vessel_type = normalize_code(content.get("type"))
        has_prediction = bool(content.get("predict_cntrs") or content.get("cntr_volume"))
        if vessel_type != "I" and (_has_container_frame(content) or has_prediction):
            vessels.add(voyage)
    return sorted(vessels)


def discover_import_vessels(input_guandong: InputAdapterGd) -> list[str]:
    take_over_imports = _take_over_vessels(input_guandong, "I")
    if take_over_imports:
        return [
            vessel
            for vessel in take_over_imports
            if isinstance(input_guandong.vessel_containers.get(vessel, {}), Mapping)
            and normalize_code(input_guandong.vessel_containers.get(vessel, {}).get("type")) == "I"
            and _has_container_frame(input_guandong.vessel_containers.get(vessel, {}))
        ]
    vessels: set[str] = set()
    for voyage, content in _adapter_vessel_items(input_guandong):
        if normalize_code(content.get("type")) == "I" and _has_container_frame(content):
            vessels.add(voyage)
    return sorted(vessels)

def classified_export_voyages(input_guandong: InputAdapterGd) -> set[str]:
    """Return voyages explicitly classified as export, independent of box flow."""
    cache = adapter_runtime_cache(input_guandong)
    cached = cache.get("classified_export_voyages")
    if isinstance(cached, set):
        return set(cached)
    exports = set(_take_over_vessels(input_guandong, "E"))
    exports.update(
        voyage
        for voyage, content in _adapter_vessel_items(input_guandong)
        if normalize_code(content.get("type")) == "E"
    )
    vessel_info = read_vessel_info(input_guandong)
    exports.update(
        normalize_voyage(row.get("voy_id"))
        for row in vessel_info.to_dict("records")
        if row.get("ie_flag") == "E" and normalize_voyage(row.get("voy_id"))
    )
    cache["classified_export_voyages"] = set(exports)
    return exports


def normalize_bay(value: Any) -> str:
    code = normalize_code(value)
    if code.isdigit():
        return f"{int(code):02d}"
    return code


def normalize_row(value: Any) -> str:
    return normalize_bay(value)


def parse_enable_size_flags(value: Any) -> tuple[bool, bool]:
    if value is None or pd.isna(value):
        return True, True
    tokens = re.findall(r"\d+", str(value))
    if not tokens:
        return True, True
    sizes = {normalize_size_large(token) for token in tokens}
    return "20" in sizes, "40" in sizes


def enabled_size_modes(value: Any) -> set[str]:
    text = normalize_code(value)
    if not text:
        return set(SIZE_MODES)
    modes = set()
    for part in re.findall(r"\d+", text):
        if part in {"20", "40", "45"}:
            modes.add(part)
    return modes


def size_enabled_mask(values: pd.Series, size_mode: str) -> pd.Series:
    text = values.astype("string")
    stripped = text.str.strip()
    missing = values.isna() | stripped.str.lower().isin(["", "nan", "none", "<na>"]).fillna(False)
    enabled = stripped.str.contains(rf"(?<!\d){re.escape(size_mode)}(?!\d)", regex=True, na=False)
    return missing | enabled


def _sheet_has_area_and_berths(path: Path, sheet: str) -> bool:
    try:
        frame = pd.read_excel(path, sheet_name=sheet, nrows=2)
    except Exception:
        return False
    columns = {str(column) for column in frame.columns}
    return "area_no" in columns and any(re.fullmatch(r"B\d+", column) for column in columns)


def adapter_runtime_cache(input_guandong: InputAdapterGd) -> dict[str, Any]:
    cache = getattr(input_guandong, "_standard_runtime_cache", None)
    if isinstance(cache, dict):
        return cache
    cache = {}
    try:
        setattr(input_guandong, "_standard_runtime_cache", cache)
    except Exception:
        return {}
    return cache


def read_vessel_info(input_guandong: InputAdapterGd) -> pd.DataFrame:
    cache = adapter_runtime_cache(input_guandong)
    cached = cache.get("vessel_info")
    if isinstance(cached, pd.DataFrame):
        return cached.copy()
    frame = input_guandong.vessel_berth_info
    frame = frame.copy()
    frame["voy_id"] = frame["VOY_ID"].map(normalize_voyage)
    frame["ie_flag"] = frame["VOY_IEFG"].map(normalize_code)
    frame["voyage_direction"] = frame["ie_flag"]
    frame["berth_no"] = frame.get("VBT_BTH_ABTHNO", pd.Series(index=frame.index)).map(normalize_code)
    fallback_berth = frame.get("VBT_BTH_PBTHNO", pd.Series(index=frame.index)).map(normalize_code)
    frame["berth_no"] = frame["berth_no"].where(frame["berth_no"].ne(""), fallback_berth)
    frame["berth_key"] = frame["berth_no"].map(lambda value: f"B{value}" if value and not str(value).startswith("B") else value)
    for column in ["SCD_RCVSTDT", "SCD_RCVEDDT", "VBT_ABTHDT", "VBT_PBTHDT", "VBT_ADPTDT", "VBT_PDPTDT"]:
        if column in frame.columns:
            frame[column] = frame[column].map(parse_datetime)
    frame["planned_berth_time"] = pd.to_datetime(
        frame.get("VBT_PBTHDT", pd.Series(index=frame.index, dtype=object)),
        errors="coerce",
    )
    frame["planned_departure_time"] = pd.to_datetime(
        frame.get("VBT_PDPTDT", pd.Series(index=frame.index, dtype=object)),
        errors="coerce",
    )
    cache["vessel_info"] = frame
    return frame.copy()


def read_berths_for_vessels(vessel_info: pd.DataFrame, vessels: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for vessel in vessels:
        rows = vessel_info[vessel_info["voy_id"].eq(vessel)]
        if rows.empty:
            continue
        berth = normalize_code(rows.iloc[0].get("berth_key"))
        if berth:
            result[vessel] = berth
    return result


def read_vessel_schedules(input_guandong: InputAdapterGd) -> dict[str, VoyageSchedule]:
    frame = read_vessel_info(input_guandong)
    schedules: dict[str, VoyageSchedule] = {}
    for row in frame.to_dict("records"):
        if row.get("ie_flag") != "E":
            continue
        voyage_id = normalize_voyage(row.get("voy_id"))
        receive_start = row.get("SCD_RCVSTDT")
        receive_end = row.get("SCD_RCVEDDT")
        berth_time = row.get("VBT_ABTHDT") or row.get("VBT_PBTHDT")
        departure_time = row.get("VBT_ADPTDT") or row.get("VBT_PDPTDT")
        berth_no = normalize_code(row.get("berth_no"))
        if not (voyage_id and receive_start and receive_end and berth_time and departure_time):
            continue
        schedules[voyage_id] = VoyageSchedule(
            voyage_id=voyage_id,
            receive_start=receive_start,
            receive_end=receive_end,
            berth_no=berth_no,
            berth_time=berth_time,
            departure_time=departure_time,
        )
    return schedules


def read_target_vessel_schedules(
    input_guandong: InputAdapterGd,
    target_voyages: list[str],
    planning_time: datetime,
    horizon_hours: float,
) -> dict[str, VoyageSchedule]:
    schedules = read_vessel_schedules(input_guandong)
    frame = read_vessel_info(input_guandong)
    rows_by_voyage = {
        normalize_voyage(row.get("voy_id")): row
        for row in frame.to_dict("records")
        if row.get("ie_flag") == "E"
    }
    for voyage_id in target_voyages:
        if voyage_id in schedules:
            continue
        row = rows_by_voyage.get(voyage_id, {})
        berth_no = normalize_code(row.get("berth_no"))
        schedules[voyage_id] = VoyageSchedule(
            voyage_id=voyage_id,
            receive_start=planning_time,
            receive_end=planning_time + timedelta(hours=horizon_hours),
            berth_no=berth_no,
            berth_time=planning_time + timedelta(hours=horizon_hours),
            departure_time=planning_time + timedelta(hours=horizon_hours + 12),
        )
    return schedules


def read_area_functions_large(input_guandong: InputAdapterGd) -> tuple[list[str], dict[str, set[str]], dict[str, float]]:
    # frame = pd.read_excel(find_area_function_file(input_guandong))
    frame = input_guandong.area_function_info
    area_col = _first_existing(set(frame.columns), ["area_no", "AREA_NO", "YAA_AREANO"])
    type_col = _first_existing(set(frame.columns), ["cntr_type", "CNTR_TYPE", "function", "FUNCTION"])
    load_col = _first_existing(set(frame.columns), ["load_capacity", "H", "capacity", "作业能力", "浣滀笟鑳藉姏"])
    if not area_col or not type_col:
        raise KeyError("Area function workbook must contain area_no and cntr_type columns.")
    area_functions: dict[str, set[str]] = {}
    load_capacity: dict[str, float] = {}
    for row in frame.to_dict("records"):
        area = normalize_code(row.get(area_col))
        if not area:
            continue
        funcs = {normalize_code(part) for part in str(row.get(type_col, "")).split(",") if normalize_code(part)}
        area_functions[area] = funcs
        value = pd.to_numeric(row.get(load_col), errors="coerce") if load_col else pd.NA
        load_capacity[area] = float(value) if pd.notna(value) else 999999.0
    return sorted(area_functions), area_functions, load_capacity


def read_area_functions(input_guandong: InputAdapterGd) -> dict[str, set[str]]:
    _, area_functions, _ = read_area_functions_large(input_guandong)
    return area_functions


def read_distance_matrix(
    input_guandong: InputAdapterGd,
    areas: Sequence[str] | None = None,
    berth_by_vessel: Mapping[str, str] | None = None,
) -> dict[tuple[str, str], float]:
    frame = input_guandong.berth_area_dist_matrix
    area_filter = set(areas or [])
    berth_columns = [column for column in frame.columns if str(column).upper().startswith("B")]
    berth_keys = {normalize_code(column): column for column in berth_columns}
    if berth_by_vessel:
        matrix = frame.copy()
        matrix["area_no"] = matrix["area_no"].map(normalize_code)
        matrix = matrix.dropna(subset=["area_no"]).set_index("area_no")
        distances: dict[tuple[str, str], float] = {}
        for vessel, berth in berth_by_vessel.items():
            berth_key = normalize_code(berth)
            column = berth_keys.get(berth_key)
            if column is None:
                raise KeyError(f"Distance matrix does not contain berth column {berth_key}.")
            for area in areas or list(matrix.index):
                if area not in matrix.index:
                    raise KeyError(f"Distance matrix does not contain area {area}.")
                distances[(vessel, area)] = float(matrix.loc[area, column])
        return distances

    distances: dict[tuple[str, str], float] = {}
    for row in frame.to_dict("records"):
        area = normalize_code(row.get("area_no"))
        if not area or (area_filter and area not in area_filter):
            continue
        for berth in berth_columns:
            berth_key = normalize_code(berth)
            value = row.get(berth)
            if pd.notna(value):
                distances[(area, berth_key)] = float(value)
    return distances


def read_snapshot(input_guandong: InputAdapterGd) -> pd.DataFrame:
    return input_guandong.bay_slots_detail


def extract_current_snapshot_rows(
    snapshot: pd.DataFrame,
    export_vessels: Sequence[str],
    import_vessels: Sequence[str],
    flow_aliases: Mapping[str, str],
) -> pd.DataFrame:
    occupied = snapshot[snapshot["HAS_CONTAINER"].fillna(0).astype(int).eq(1)].copy()
    if occupied.empty:
        return occupied
    occupied["e_voy"] = occupied["IYC_EVOY_ID"].map(normalize_voyage)
    occupied["i_voy"] = occupied["IYC_IVOY_ID"].map(normalize_voyage)
    export_set = set(export_vessels)
    import_set = set(import_vessels)
    rows = occupied[occupied["e_voy"].isin(export_set) | occupied["i_voy"].isin(import_set)].copy()
    rows["voy_id"] = rows["e_voy"].where(rows["e_voy"].isin(export_set), rows["i_voy"])
    rows["direction"] = rows["e_voy"].isin(export_set).map(lambda is_export: "E" if is_export else "I")
    rows["area_no"] = rows["YAA_AREANO"].map(normalize_code)
    rows["bay_no"] = rows["YBY_BAYNO"].map(normalize_bay)
    rows["cntr_id"] = rows["IYC_CNTRID"].map(normalize_code)
    rows["size"] = rows["IYC_CSZ_CSIZECD"].map(normalize_size_large)
    export_mask = rows["direction"].eq("E")
    transshipment_mask = export_mask & rows["i_voy"].map(has_import_voyage)
    rows["planning_excluded"] = transshipment_mask
    rows["planning_exclusion_reason"] = ""
    rows.loc[transshipment_mask, "planning_exclusion_reason"] = "export_snapshot_has_import_voyage"
    rows.loc[export_mask, "flow"] = rows.loc[export_mask, "IYC_STS_CSTATUSCD"].map(
        lambda value: normalize_export_snapshot_flow(value, flow_aliases)
    )
    rows.loc[~export_mask, "flow"] = rows.loc[~export_mask, "IYC_STS_CSTATUSCD"].map(
        lambda value: medium_small_area_flow(normalize_flow(value, flow_aliases))
    )
    return rows


def build_bay_total_slot_counts(snapshot: pd.DataFrame, areas: Sequence[str]) -> dict[tuple[str, str], int]:
    work = snapshot[snapshot["YAA_AREANO"].map(normalize_code).isin(set(areas))].copy()
    work["area_no"] = work["YAA_AREANO"].map(normalize_code)
    work["bay_no"] = work["YBY_BAYNO"].map(normalize_bay)
    counts = work.groupby(["area_no", "bay_no"], sort=False).size()
    return {(str(area), str(bay)): int(count) for (area, bay), count in counts.items()}


def identify_bad_bays(
    current_snapshot: pd.DataFrame,
    area_functions: Mapping[str, set[str]],
    bay_total_slots: Mapping[tuple[str, str], int],
    ratio: float = 2.0 / 3.0,
) -> set[tuple[str, str]]:
    if current_snapshot.empty:
        return set()
    current_snapshot = planning_included_rows(current_snapshot)
    current_snapshot = current_snapshot[current_snapshot.get("direction", "").eq("E")].copy()
    if current_snapshot.empty:
        return set()
    current_snapshot = current_snapshot[
        current_snapshot["cntr_id"].notna()
        & current_snapshot["cntr_id"].astype(str).ne("")
        & current_snapshot["cntr_id"].astype(str).ne("-1")
    ].copy()
    current_snapshot = current_snapshot.drop_duplicates(["cntr_id", "area_no", "bay_no"])
    bad_counts: Counter[tuple[str, str]] = Counter()
    for row in current_snapshot.to_dict("records"):
        area = normalize_code(row.get("area_no"))
        bay = normalize_bay(row.get("bay_no"))
        flow = normalize_code(row.get("flow"))
        if area and bay and flow and not area_allows_flow(area, flow, area_functions):
            bad_counts[(area, bay)] += 1
    bad_bays: set[tuple[str, str]] = set()
    for key, count in bad_counts.items():
        total_slots = bay_total_slots.get(key, 0)
        if total_slots > 0 and count > total_slots * ratio:
            bad_bays.add(key)
    return bad_bays


def build_snapshot_count_params(
    current_snapshot: pd.DataFrame,
    area_functions: Mapping[str, set[str]],
    areas: set[str],
) -> tuple[
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], float],
]:
    l20: dict[tuple[str, str, str], float] = {}
    l40: dict[tuple[str, str, str], float] = {}
    q20: dict[tuple[str, str, str], float] = {}
    q40: dict[tuple[str, str, str], float] = {}
    if current_snapshot.empty:
        return l20, l40, q20, q40
    unique_containers = planning_included_rows(current_snapshot)
    unique_containers = unique_containers[unique_containers["cntr_id"].notna()].copy()
    unique_containers = unique_containers.sort_values(["cntr_id", "area_no", "bay_no"]).drop_duplicates(
        "cntr_id",
        keep="first",
    )
    unique_containers = unique_containers[unique_containers["cntr_id"].astype(str).ne("-1")].copy()
    grouped = unique_containers.groupby(["voy_id", "flow", "area_no", "size"], dropna=False).size()
    for (vessel, flow, area, size), qty in grouped.items():
        if not vessel or not flow or not area or area not in areas:
            continue
        target = (l20 if size == "20" else l40) if area_allows_flow(area, flow, area_functions) else (q20 if size == "20" else q40)
        target[(str(vessel), str(flow), str(area))] = target.get((str(vessel), str(flow), str(area)), 0.0) + float(qty)
    return l20, l40, q20, q40


def prepare_slot_frame(snapshot: pd.DataFrame, areas: Sequence[str], bad_bays: set[tuple[str, str]]) -> pd.DataFrame:
    work = snapshot.copy()
    work["area_no"] = work["YAA_AREANO"].map(normalize_code)
    work["bay_no"] = work["YBY_BAYNO"].map(normalize_bay)
    work = work[work["area_no"].isin(set(areas))].copy()
    if bad_bays:
        mask = [(area, bay) not in bad_bays for area, bay in zip(work["area_no"], work["bay_no"])]
        work = work.loc[mask].copy()
    work = work[work["HAS_CONTAINER"].fillna(0).astype(int).eq(0)].copy()
    parsed = work["YBY_ENABLECSIZECD"].map(parse_enable_size_flags)
    work["enable_20"] = parsed.map(lambda item: item[0])
    work["enable_40"] = parsed.map(lambda item: item[1])
    work["slot_uid"] = (
        work["area_no"].astype(str)
        + "|"
        + work["bay_no"].astype(str)
        + "|"
        + work["YST_ROWNO"].map(normalize_row).astype(str)
        + "|"
        + work["YST_TIERNO"].map(normalize_row).astype(str)
        + "|"
        + work["YST_SLOTNO"].map(normalize_row).astype(str)
    )
    return work


def count_slots_by_area(slots: pd.DataFrame, areas: Sequence[str]) -> dict[str, float]:
    counts = slots.groupby("area_no").size() if not slots.empty else pd.Series(dtype=int)
    return {area: float(counts.get(area, 0)) for area in areas}


def normalize_container_frame(df: pd.DataFrame, flow_aliases: Mapping[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["cntr_id"] = df["IYC_CNTRID"].map(normalize_code)
    df["e_voy"] = df["IYC_EVOY_ID"].map(normalize_voyage)
    df["i_voy"] = df["IYC_IVOY_ID"].map(normalize_voyage)
    df["size"] = df["IYC_CSZ_CSIZECD"].map(normalize_size_large)
    df["port"] = df.get("IYC_POT_UNLDPORT", pd.Series(index=df.index, dtype=object)).map(
        lambda value: normalize_text(value, "UNK")
    )
    df["raw_flow"] = df["IYC_STS_CSTATUSCD"].map(normalize_code)
    df["flow"] = df["raw_flow"].map(lambda value: medium_small_area_flow(normalize_flow(value, flow_aliases)))
    return df


def merge_snapshot_and_doc(doc: pd.DataFrame, snapshot_rows: pd.DataFrame) -> pd.DataFrame:
    doc_part = doc[["cntr_id", "flow", "size"]].copy()
    doc_part["source_rank"] = 1
    if snapshot_rows.empty:
        snap_part = doc_part.iloc[0:0].copy()
    else:
        snap_part = snapshot_rows[["cntr_id", "flow", "size"]].copy()
        snap_part["source_rank"] = 0
    merged = pd.concat([snap_part, doc_part], ignore_index=True)
    merged = merged[merged["cntr_id"].notna() & merged["flow"].notna() & merged["size"].notna()].copy()
    return merged.sort_values("source_rank").drop_duplicates("cntr_id", keep="first")


def unique_snapshot_rows(snapshot_rows: pd.DataFrame) -> pd.DataFrame:
    if snapshot_rows.empty:
        return snapshot_rows.copy()
    rows = snapshot_rows[snapshot_rows["cntr_id"].notna()].copy()
    rows = rows[rows["cntr_id"].astype(str).ne("-1")]
    return rows.sort_values(["cntr_id", "area_no", "bay_no"]).drop_duplicates("cntr_id", keep="first")


def valid_doc_rows(doc: pd.DataFrame) -> pd.DataFrame:
    if doc.empty:
        return doc.copy()
    rows = doc[doc["cntr_id"].notna() & doc["flow"].notna() & doc["size"].notna()].copy()
    return rows.sort_values("cntr_id").drop_duplicates("cntr_id", keep="first")


def add_grouped_demand(
    rows: pd.DataFrame,
    vessel: str,
    d20: dict[tuple[str, str], float],
    d40: dict[tuple[str, str], float],
) -> None:
    if rows.empty:
        return
    grouped = rows.groupby(["flow", "size"]).size()
    for (flow, size), qty in grouped.items():
        target = d20 if size == "20" else d40
        key = (vessel, str(flow))
        target[key] = target.get(key, 0.0) + float(qty)


def add_import_port_area_demand(
    rows: pd.DataFrame,
    vessel: str,
    port_sail_area: Mapping[str, set[str]],
    demand20: dict[tuple[str, str, str], float],
    demand40: dict[tuple[str, str, str], float],
) -> int:
    if rows.empty or not port_sail_area:
        return 0
    work = rows.copy()
    if "port" not in work.columns:
        work["port"] = work.get("IYC_POT_UNLDPORT", pd.Series(index=work.index, dtype=object)).map(
            lambda value: normalize_text(value, "UNK")
        )
    work = work[work["port"].isin(set(port_sail_area))].copy()
    if work.empty:
        return 0
    grouped = work.groupby(["flow", "size", "port"]).size()
    for (flow, size, port), qty in grouped.items():
        target = demand20 if size == "20" else demand40
        key = (vessel, str(flow), str(port))
        target[key] = target.get(key, 0.0) + float(qty)
    return int(grouped.sum())


def import_cluster_key(row: Mapping[str, Any]) -> str:
    second_voyage = normalize_voyage(row.get("e_voy") or row.get("IYC_EVOY_ID"))
    if second_voyage:
        return f"SECOND:{second_voyage}"
    port = normalize_text(row.get("port") or row.get("IYC_POT_UNLDPORT"), "UNK")
    return f"PORT:{port or 'UNK'}"


def add_import_cluster_demand(
    rows: pd.DataFrame,
    vessel: str,
    demand20: dict[tuple[str, str, str], float],
    demand40: dict[tuple[str, str, str], float],
    snapshot20: dict[tuple[str, str, str, str], float] | None = None,
    snapshot40: dict[tuple[str, str, str, str], float] | None = None,
) -> dict[str, int]:
    if rows.empty:
        return {"rows": 0, "second_voyage_rows": 0, "port_rows": 0, "groups": 0}
    cluster_rows: list[dict[str, Any]] = []
    for record in rows.to_dict("records"):
        flow = normalize_text(record.get("flow"))
        size = normalize_size_large(record.get("size"))
        if not flow or size not in {"20", "40"}:
            continue
        cluster = import_cluster_key(record)
        area = normalize_area(record.get("area_no"))
        cluster_rows.append({"flow": flow, "size": size, "cluster": cluster, "area": area})
    if not cluster_rows:
        return {"rows": 0, "second_voyage_rows": 0, "port_rows": 0, "groups": 0}
    work = pd.DataFrame(cluster_rows)
    grouped = work.groupby(["flow", "size", "cluster"]).size()
    for (flow, size, cluster), qty in grouped.items():
        target = demand20 if size == "20" else demand40
        key = (vessel, str(flow), str(cluster))
        target[key] = target.get(key, 0.0) + float(qty)
    if snapshot20 is not None and snapshot40 is not None:
        snapshot_work = work[work["area"].astype(str).ne("")].copy()
        grouped_snapshot = snapshot_work.groupby(["flow", "size", "cluster", "area"]).size()
        for (flow, size, cluster, area), qty in grouped_snapshot.items():
            target_snapshot = snapshot20 if size == "20" else snapshot40
            key = (vessel, str(flow), str(cluster), str(area))
            target_snapshot[key] = target_snapshot.get(key, 0.0) + float(qty)
    second_rows = int(work["cluster"].astype(str).str.startswith("SECOND:").sum())
    return {
        "rows": int(len(work)),
        "second_voyage_rows": second_rows,
        "port_rows": int(len(work) - second_rows),
        "groups": int(work["cluster"].nunique()),
    }


def read_prediction_counts(input_guandong: InputAdapterGd, vessel_id: str) -> tuple[float, float]:
    content = input_guandong.vessel_containers.get(vessel_id, {})
    predict_data = content.get("predict_cntrs") or content.get("cntr_volume") or {}
    total20 = 0.0
    total40 = 0.0
    for raw_size, payload in predict_data.items():
        size = normalize_size_large(raw_size)
        total = pd.to_numeric((payload or {}).get("total_volume"), errors="coerce")
        total = float(total) if pd.notna(total) else 0.0
        if size == "20":
            total20 += total
        elif size == "40":
            total40 += total
    return total20, total40


def read_prediction_work_lanes(path: Path) -> float:
    xls = pd.ExcelFile(path)
    sheet_name = next(
        (
            name
            for name in xls.sheet_names
            if "work" in str(name).lower() or "lane" in str(name).lower()
        ),
        xls.sheet_names[-1],
    )
    frame = pd.read_excel(path, sheet_name=sheet_name)
    candidates: list[float] = []

    def collect(value: Any) -> None:
        if value is None or pd.isna(value):
            return
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric) and float(numeric) > 0:
            candidates.append(float(numeric))

    for column in frame.columns:
        collect(column)
    for value in frame.to_numpy().ravel():
        collect(value)
    if not candidates:
        raise ValueError(f"No positive work-lane count in {path}")
    return candidates[0]


def build_demand_params(
    input_guandong: InputAdapterGd,
    export_vessels: Sequence[str],
    import_vessels: Sequence[str],
    current_snapshot: pd.DataFrame,
    flow_aliases: Mapping[str, str],
    covered_areas: Sequence[str] | None = None,
) -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str, str], float],
    dict[tuple[str, str, str, str], float],
    dict[str, dict[str, Any]],
]:
    d20: dict[tuple[str, str], float] = {}
    d40: dict[tuple[str, str], float] = {}
    port_demand20: dict[tuple[str, str, str], float] = {}
    port_demand40: dict[tuple[str, str, str], float] = {}
    import_cluster_demand20: dict[tuple[str, str, str], float] = {}
    import_cluster_demand40: dict[tuple[str, str, str], float] = {}
    import_cluster_snapshot20: dict[tuple[str, str, str, str], float] = {}
    import_cluster_snapshot40: dict[tuple[str, str, str, str], float] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    port_sail_area = read_port_sail_area(input_guandong)
    covered_area_set = {normalize_area(area) for area in covered_areas or [] if normalize_area(area)}
    for vessel in export_vessels:
        doc_container = input_guandong.vessel_containers.get(vessel, {}).get("doc_cntrs", None)
        if isinstance(doc_container, pd.DataFrame):
            doc = normalize_container_frame(doc_container, flow_aliases)
            doc = doc[doc["e_voy"].eq(vessel)].copy()
        else:
            doc = pd.DataFrame(columns=["cntr_id", "e_voy", "i_voy", "size", "raw_flow", "flow"])
        snap_all = current_snapshot[current_snapshot["voy_id"].eq(vessel)].copy()
        snap_excluded = snap_all.loc[planning_excluded_mask(snap_all)].copy()
        snap = planning_included_rows(snap_all)
        unique_snap = unique_snapshot_rows(snap)
        unique_snap_all = unique_snapshot_rows(snap_all)
        excluded_snapshot_ids = set(snap_excluded["cntr_id"].dropna().astype(str))
        doc["flow"] = "OF"
        doc_excluded = doc[doc["cntr_id"].astype(str).isin(excluded_snapshot_ids)].copy()
        doc = doc[~doc["cntr_id"].astype(str).isin(excluded_snapshot_ids)].copy()
        covered_snap = unique_snap[unique_snap["area_no"].isin(covered_area_set)].copy() if covered_area_set else unique_snap
        doc_unique = valid_doc_rows(doc)
        snapshot_ids = set(unique_snap_all["cntr_id"].astype(str))
        doc_new = doc_unique[~doc_unique["cntr_id"].astype(str).isin(snapshot_ids)].copy()
        merged = merge_snapshot_and_doc(doc, unique_snap)
        detail20 = float((merged["size"] == "20").sum())
        detail40 = float((merged["size"] == "40").sum())
        add_grouped_demand(covered_snap, vessel, d20, d40)
        add_grouped_demand(doc_new, vessel, d20, d40)
        pred20, pred40 = read_prediction_counts(input_guandong, vessel)
        extra20 = max(0.0, pred20 - detail20)
        extra40 = max(0.0, pred40 - detail40)
        if extra20 > 0:
            d20[(vessel, "OF")] = d20.get((vessel, "OF"), 0.0) + extra20
        if extra40 > 0:
            d40[(vessel, "OF")] = d40.get((vessel, "OF"), 0.0) + extra40
        diagnostics[vessel] = {
            "type": "export",
            "doc_rows": int(len(doc) + len(doc_excluded)),
            "doc_planning_rows": int(len(doc)),
            "doc_snapshot_transshipment_rows_excluded": int(len(doc_excluded)),
            "snapshot_rows": int(len(snap_all)),
            "snapshot_transshipment_rows_excluded": int(len(snap_excluded)),
            "snapshot_unique_rows": int(len(unique_snap)),
            "covered_snapshot_rows": int(len(covered_snap)),
            "doc_new_rows": int(len(doc_new)),
            "dedup_rows": int(len(merged)),
            "prediction20": float(pred20),
            "prediction40": float(pred40),
            "known_rows_for_prediction_offset": int(len(merged)),
            "extra_prediction20_to_OF": float(extra20),
            "extra_prediction40_to_OF": float(extra40),
        }
    for vessel in import_vessels:
        doc_container = input_guandong.vessel_containers.get(vessel, {}).get("doc_cntrs", None)
        if isinstance(doc_container, pd.DataFrame):
            doc = normalize_container_frame(doc_container, flow_aliases)
            doc = doc[doc["i_voy"].eq(vessel)].copy()
        else:
            doc = pd.DataFrame(columns=["cntr_id", "e_voy", "i_voy", "size", "raw_flow", "flow"])
        snap = planning_included_rows(current_snapshot[current_snapshot["voy_id"].eq(vessel)].copy())
        unique_snap = unique_snapshot_rows(snap)
        covered_snap = unique_snap[unique_snap["area_no"].isin(covered_area_set)].copy() if covered_area_set else unique_snap
        doc_unique = valid_doc_rows(doc)
        snapshot_ids = set(unique_snap["cntr_id"].astype(str))
        doc_new = doc_unique[~doc_unique["cntr_id"].astype(str).isin(snapshot_ids)].copy()
        merged = merge_snapshot_and_doc(doc, unique_snap)
        add_grouped_demand(covered_snap, vessel, d20, d40)
        add_grouped_demand(doc_new, vessel, d20, d40)
        snapshot_cluster_stats = add_import_cluster_demand(
            covered_snap,
            vessel,
            import_cluster_demand20,
            import_cluster_demand40,
            import_cluster_snapshot20,
            import_cluster_snapshot40,
        )
        doc_cluster_stats = add_import_cluster_demand(
            doc_new,
            vessel,
            import_cluster_demand20,
            import_cluster_demand40,
        )
        port_constrained_rows = add_import_port_area_demand(
            doc_new,
            vessel,
            port_sail_area,
            port_demand20,
            port_demand40,
        )
        diagnostics[vessel] = {
            "type": "import",
            "doc_rows": int(len(doc)),
            "snapshot_rows": int(len(snap)),
            "snapshot_unique_rows": int(len(unique_snap)),
            "covered_snapshot_rows": int(len(covered_snap)),
            "doc_new_rows": int(len(doc_new)),
            "dedup_rows": int(len(merged)),
            "port_area_constrained_doc_new_rows": port_constrained_rows,
            "import_cluster_snapshot_rows": snapshot_cluster_stats["rows"],
            "import_cluster_doc_new_rows": doc_cluster_stats["rows"],
            "import_cluster_second_voyage_rows": (
                snapshot_cluster_stats["second_voyage_rows"] + doc_cluster_stats["second_voyage_rows"]
            ),
            "import_cluster_port_rows": snapshot_cluster_stats["port_rows"] + doc_cluster_stats["port_rows"],
            "import_cluster_group_count": len(
                {
                    key[2]
                    for key in import_cluster_demand20
                    if key[0] == vessel
                }
                | {
                    key[2]
                    for key in import_cluster_demand40
                    if key[0] == vessel
                }
            ),
        }
    return (
        d20,
        d40,
        port_demand20,
        port_demand40,
        import_cluster_demand20,
        import_cluster_demand40,
        import_cluster_snapshot20,
        import_cluster_snapshot40,
        diagnostics,
    )


def parse_tops_time(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def parse_tops_area_bay(value: Any) -> tuple[str, str]:
    code = normalize_code(value).replace(".0", "")
    if not code:
        return "", ""
    if len(code) < 4:
        code = code.zfill(4)
    return code[:2], normalize_bay(code[-2:])


def bay_code_value(value: Any) -> int | None:
    code = normalize_code(value)
    if not code:
        return None
    total = 0
    for char in code:
        if "0" <= char <= "9":
            digit = ord(char) - ord("0")
        elif "A" <= char <= "Z":
            digit = ord(char) - ord("A") + 10
        else:
            return None
        total = total * 36 + digit
    return total


def slot_range_mask(values: pd.Series, start_value: Any, end_value: Any, value_parser: Any) -> pd.Series:
    start = normalize_code(start_value)
    end = normalize_code(end_value)
    if not start and not end:
        return pd.Series(True, index=values.index)
    if start and not end:
        return values.map(normalize_code).eq(start)
    if end and not start:
        return values.map(normalize_code).eq(end)
    start_key = value_parser(start)
    end_key = value_parser(end)
    if start_key is None or end_key is None:
        allowed = {value for value in [start, end] if value}
        return values.map(normalize_code).isin(allowed)
    lo = min(start_key, end_key)
    hi = max(start_key, end_key)
    parsed_values = values.map(value_parser)
    return parsed_values.map(lambda value: value is not None and lo <= value <= hi)


def slot_range_mask_preparsed(
    values: pd.Series,
    parsed_values: pd.Series,
    start_value: Any,
    end_value: Any,
    value_parser: Any,
) -> pd.Series:
    start = normalize_code(start_value)
    end = normalize_code(end_value)
    if not start and not end:
        return pd.Series(True, index=values.index)
    if start and not end:
        return values.map(normalize_code).eq(start)
    if end and not start:
        return values.map(normalize_code).eq(end)
    start_key = value_parser(start)
    end_key = value_parser(end)
    if start_key is None or end_key is None:
        allowed = {value for value in [start, end] if value}
        return values.map(normalize_code).isin(allowed)
    lo = min(start_key, end_key)
    hi = max(start_key, end_key)
    return parsed_values.notna() & parsed_values.ge(lo) & parsed_values.le(hi)


def bay_range_mask(values: pd.Series, start_bay: str, end_bay: str) -> pd.Series:
    return slot_range_mask(values, start_bay, end_bay, bay_code_value)


def row_range_mask(values: pd.Series, start_row: str, end_row: str) -> pd.Series:
    return slot_range_mask(values, start_row, end_row, bay_code_value)


def count_tops_blocked_slots(tops_rows: pd.DataFrame, slots: pd.DataFrame, vessel: str) -> dict[tuple[str, str], float]:
    blocked_by_area: dict[str, set[Any]] = {}
    if tops_rows.empty or slots.empty:
        return {}
    slots_by_area = {area: sub for area, sub in slots.groupby("area_no")}
    for _, tops in tops_rows.iterrows():
        start_area, start_bay = parse_tops_area_bay(tops.get("SPR_STBAY"))
        end_area, end_bay = parse_tops_area_bay(tops.get("SPR_EDBAY"))
        area = start_area or end_area
        if start_area and end_area and start_area != end_area:
            area = end_area
        if not area or area not in slots_by_area:
            continue
        sub = slots_by_area[area]
        matched = sub[bay_range_mask(sub["bay_no"], start_bay, end_bay)]
        if matched.empty:
            continue
        blocked_by_area.setdefault(area, set()).update(matched["slot_uid"].tolist())
    return {(vessel, area): float(len(uids)) for area, uids in blocked_by_area.items()}


def active_tops_rows(input_guandong: InputAdapterGd, planning_time: datetime) -> pd.DataFrame:
    cache = adapter_runtime_cache(input_guandong)
    tops = cache.get("tops_plan_normalized")
    if not isinstance(tops, pd.DataFrame):
        raw_tops = input_guandong.tops_plan
        if raw_tops is None or raw_tops.empty:
            tops = pd.DataFrame(columns=["condition_vessel", "start_time", "end_time"])
            cache["tops_plan_normalized"] = tops
            return tops.copy()
        tops = raw_tops.copy()
        tops["condition_vessel"] = tops["SPL_CONDITIONCODE"].map(normalize_voyage)
        tops["start_time"] = parse_tops_time(tops["SPL_STDATE"])
        tops["end_time"] = parse_tops_time(tops["SPL_EDDATE"])
        if "SPL_ISVALID" in tops.columns:
            tops = tops[tops["SPL_ISVALID"].astype(str).str.upper().eq("Y")].copy()
        if "SPR_ISVALID" in tops.columns:
            tops = tops[tops["SPR_ISVALID"].astype(str).str.upper().eq("Y")].copy()
        cache["tops_plan_normalized"] = tops
    if tops.empty:
        return tops.copy()
    return tops[(tops["start_time"] <= planning_time) & (planning_time <= tops["end_time"])].copy()


def compute_departure_operation_deductions(
    *,
    vessel_info: pd.DataFrame,
    snapshot: pd.DataFrame,
    planning_time: pd.Timestamp,
    areas: Sequence[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    planning_date = pd.Timestamp(planning_time).date()
    area_set = set(areas)
    info = vessel_info.copy()
    direction = info.get("voyage_direction", info.get("ie_flag", pd.Series(index=info.index, dtype=object))).map(normalize_code)
    info = info[
        direction.eq("E")
        & info["planned_berth_time"].notna()
        & info["planned_departure_time"].notna()
    ].copy()
    if info.empty:
        return {area: 0.0 for area in areas}, {"active_export_voyages": [], "counts_by_voyage_area": []}

    berth_dates = info["planned_berth_time"].dt.date
    departure_dates = info["planned_departure_time"].dt.date
    active = info[(berth_dates <= planning_date) & (planning_date <= departure_dates)].copy()
    active = active.drop_duplicates("voy_id")
    if active.empty:
        return {area: 0.0 for area in areas}, {"active_export_voyages": [], "counts_by_voyage_area": []}

    days_remaining = {
        row["voy_id"]: max(1, (row["planned_departure_time"].date() - planning_date).days + 1)
        for _, row in active.iterrows()
        if row.get("voy_id")
    }
    active_voyages = set(days_remaining)
    occupied = snapshot[snapshot["HAS_CONTAINER"].fillna(0).astype(int).eq(1)].copy()
    if not occupied.empty:
        occupied["_e_voy"] = occupied["IYC_EVOY_ID"].map(normalize_voyage)
        occupied["_area_no"] = occupied["YAA_AREANO"].map(normalize_code)
        occupied["_cntr_id"] = occupied["IYC_CNTRID"].map(normalize_code)
        occupied = occupied[
            occupied["_e_voy"].isin(active_voyages)
            & occupied["_area_no"].isin(area_set)
            & occupied["_cntr_id"].notna()
        ].copy()
        occupied = occupied[~occupied["_cntr_id"].isin({"", "-1"})].copy()

    deductions = {area: 0.0 for area in areas}
    count_rows: list[dict[str, Any]] = []
    if not occupied.empty:
        unique_containers = occupied.sort_values(["_cntr_id", "_area_no"]).drop_duplicates("_cntr_id", keep="first")
        grouped = unique_containers.groupby(["_e_voy", "_area_no"], dropna=False)["_cntr_id"].nunique()
        for (voyage, area), count in grouped.items():
            if not voyage or not area or area not in deductions:
                continue
            days = days_remaining.get(voyage, 1)
            deduction = float(math.ceil(float(count) / float(days))) if count else 0.0
            deductions[area] += deduction
            count_rows.append(
                {
                    "voy_id": voyage,
                    "area_no": area,
                    "yard_container_count": int(count),
                    "days_until_planned_departure_inclusive": int(days),
                    "daily_departure_operation_deduction": deduction,
                }
            )

    diagnostics = {
        "active_export_voyages": sorted(active_voyages),
        "counts_by_voyage_area": sorted(count_rows, key=lambda row: (row["voy_id"], row["area_no"])),
    }
    return deductions, diagnostics


def build_close_export_berth_pairs(
    *,
    vessel_info: pd.DataFrame,
    export_vessels: Sequence[str],
    threshold_hours: float,
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    export_set = set(export_vessels)
    info = vessel_info[vessel_info["voy_id"].isin(export_set)].drop_duplicates("voy_id").copy()
    berth_time_by_voyage = {
        row["voy_id"]: row["planned_berth_time"]
        for _, row in info.iterrows()
        if row.get("voy_id") and pd.notna(row.get("planned_berth_time"))
    }

    pairs: list[tuple[str, str]] = []
    details: list[dict[str, Any]] = []
    for left, right in combinations(export_vessels, 2):
        left_time = berth_time_by_voyage.get(left)
        right_time = berth_time_by_voyage.get(right)
        if left_time is None or right_time is None:
            continue
        if left_time.date() != right_time.date():
            continue
        delta_hours = abs((left_time - right_time).total_seconds()) / 3600.0
        if delta_hours <= threshold_hours:
            pairs.append((left, right))
            details.append(
                {
                    "left_voy_id": left,
                    "right_voy_id": right,
                    "left_planned_berth_time": left_time.isoformat(),
                    "right_planned_berth_time": right_time.isoformat(),
                    "delta_hours": float(delta_hours),
                }
            )
    return pairs, details


def compute_tops_capacity_deductions(
    input_guandong: InputAdapterGd,
    planning_time: datetime,
    vessels: Sequence[str],
    bay20_equiv: pd.DataFrame,
    bay20_direct: pd.DataFrame,
    bay40: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], dict[tuple[str, str], float], int]:
    active = active_tops_rows(input_guandong, planning_time)
    tops20: dict[tuple[str, str], float] = {}
    tops20_direct: dict[tuple[str, str], float] = {}
    tops40: dict[tuple[str, str], float] = {}
    for vessel in vessels:
        relevant = active[active["condition_vessel"] != vessel].copy()
        tops20.update(count_tops_blocked_slots(relevant, bay20_equiv, vessel))
        tops20_direct.update(count_tops_blocked_slots(relevant, bay20_direct, vessel))
        tops40.update(count_tops_blocked_slots(relevant, bay40, vessel))
    return tops20, tops20_direct, tops40, int(len(active))


def build_availability_flags(
    vessels: Sequence[str],
    flows: Sequence[str],
    areas: Sequence[str],
    area_functions: Mapping[str, set[str]],
    cbar20: Mapping[tuple[str, str], float],
    cbar20_direct: Mapping[tuple[str, str], float],
    cbar40: Mapping[tuple[str, str], float],
) -> tuple[dict[tuple[str, str, str], int], dict[tuple[str, str, str], int]]:
    e20: dict[tuple[str, str, str], int] = {}
    e40: dict[tuple[str, str, str], int] = {}
    for vessel in vessels:
        for flow in flows:
            for area in areas:
                func_ok = area_allows_flow(area, flow, area_functions)
                e20[(vessel, flow, area)] = int(
                    func_ok and cbar20_direct.get((vessel, area), 0.0) > 0 and cbar20.get((vessel, area), 0.0) > 0
                )
                e40[(vessel, flow, area)] = int(
                    func_ok and cbar40.get((vessel, area), 0.0) > 0 and cbar20.get((vessel, area), 0.0) >= 2
                )
    return e20, e40


def build_large_inputs(
    input_guandong: InputAdapterGd,
    planning_time: pd.Timestamp,
    export_vessels: Sequence[str] | None = DEFAULT_EXPORT_VESSELS,
    import_vessels: Sequence[str] | None = DEFAULT_IMPORT_VESSELS,
    disable_default_flow_aliases: bool = False,
    config: LargePlanningConfig | None = None,
) -> tuple[PlanningInputArtifacts, RollingPlanningState]:
    config = config or LargePlanningConfig()
    flow_aliases = config.active_flow_aliases(disable_default_flow_aliases)
    if export_vessels is None:
        export_vessels = discover_export_vessels(input_guandong)
    if import_vessels is None:
        import_vessels = discover_import_vessels(input_guandong)
    export_vessels = normalize_voyage_list(export_vessels)
    import_vessels = normalize_voyage_list(import_vessels)
    all_vessels = export_vessels + import_vessels
    state = RollingPlanningState(input_guandong.history_plan_info)

    vessel_info = read_vessel_info(input_guandong)
    areas, area_functions, load_capacity = read_area_functions_large(input_guandong)
    berth_by_vessel = read_berths_for_vessels(vessel_info, all_vessels)
    distance = read_distance_matrix(input_guandong, areas, berth_by_vessel)
    (
        allowed_areas_by_vessel,
        required_areas_by_vessel,
        priority_areas_by_vessel,
        area_control_diagnostics,
    ) = build_large_area_controls(
        vessels=all_vessels,
        areas=areas,
        user_design=getattr(input_guandong, "user_design", False),
        user_design_large_plan_area=getattr(input_guandong, "user_design_large_plan_area", []),
        voyage_limit_areas=getattr(input_guandong, "voyage_limit_areas", {}),
        voyage_priority_areas=getattr(input_guandong, "voyage_priority_areas", {}),
        adjust_plan_info=getattr(input_guandong, "adjust_plan_info", {}),
    )
    user_design_active = bool(area_control_diagnostics["user_design_active"])

    snapshot = read_snapshot(input_guandong)
    current_snapshot = extract_current_snapshot_rows(snapshot, export_vessels, import_vessels, flow_aliases)
    bay_total_slots = build_bay_total_slot_counts(snapshot, areas)
    bad_bays = identify_bad_bays(current_snapshot, area_functions, bay_total_slots)
    l20, l40, q20, q40 = build_snapshot_count_params(current_snapshot, area_functions, set(areas))
    if user_design_active:
        departure_deductions = {area: 0.0 for area in areas}
        departure_avoidance = {
            "active_export_voyages": [],
            "counts_by_voyage_area": [],
            "disabled_reason": "user_design_large_plan_area",
        }
        effective_load_capacity = dict(load_capacity)
        close_berth_pairs: list[tuple[str, str]] = []
        close_berth_diagnostics: list[dict[str, Any]] = []
    else:
        departure_deductions, departure_avoidance = compute_departure_operation_deductions(
            vessel_info=vessel_info,
            snapshot=snapshot,
            planning_time=planning_time,
            areas=areas,
        )
        effective_load_capacity = {
            area: max(0.0, float(load_capacity.get(area, 0.0)) - float(departure_deductions.get(area, 0.0)))
            for area in areas
        }
        close_berth_pairs, close_berth_diagnostics = build_close_export_berth_pairs(
            vessel_info=vessel_info,
            export_vessels=export_vessels,
            threshold_hours=config.berth_conflict_threshold_hours,
        )

    available_slots = prepare_slot_frame(snapshot, areas, bad_bays)
    bay20_equiv = available_slots
    bay20_direct = available_slots[available_slots["enable_20"]].copy()
    bay40 = available_slots[available_slots["enable_40"]].copy()
    c20 = count_slots_by_area(bay20_equiv, areas)
    c20_direct = count_slots_by_area(bay20_direct, areas)
    c40 = count_slots_by_area(bay40, areas)
    (
        bay_limit_areas_by_vessel,
        bay_limit_c20,
        bay_limit_c20_direct,
        bay_limit_c40,
        bay_limit_summary,
    ) = build_user_design_area_bay_large_limits(
        input_guandong,
        available_slots,
        all_vessels,
        areas,
    )

    tops20, tops20_direct, tops40, active_tops_count = compute_tops_capacity_deductions(
        input_guandong,
        planning_time.to_pydatetime(),
        all_vessels,
        bay20_equiv,
        bay20_direct,
        bay40,
    )
    cbar20 = {(v, a): max(0.0, c20.get(a, 0.0) - tops20.get((v, a), 0.0)) for v in all_vessels for a in areas}
    cbar20_direct = {
        (v, a): max(0.0, c20_direct.get(a, 0.0) - tops20_direct.get((v, a), 0.0))
        for v in all_vessels
        for a in areas
    }
    cbar40 = {(v, a): max(0.0, c40.get(a, 0.0) - tops40.get((v, a), 0.0)) for v in all_vessels for a in areas}

    (
        d20,
        d40,
        port_demand20,
        port_demand40,
        import_cluster_demand20,
        import_cluster_demand40,
        import_cluster_snapshot20,
        import_cluster_snapshot40,
        demand_diagnostics,
    ) = build_demand_params(
        input_guandong,
        export_vessels,
        import_vessels,
        current_snapshot,
        flow_aliases,
        covered_areas=areas,
    )
    of_work_lanes = {
        vessel: input_guandong.vessel_containers.get(vessel, {}).get("work_lanes", 0.0)
        for vessel in export_vessels
    }
    of_work_lanes.update({vessel: 0.0 for vessel in import_vessels})
    flows = sorted(
        {flow for funcs in area_functions.values() for flow in funcs}
        | {flow for _, flow in d20}
        | {flow for _, flow in d40}
        | {flow for _, flow, _ in l20}
        | {flow for _, flow, _ in l40}
        | {flow for _, flow, _ in q20}
        | {flow for _, flow, _ in q40}
    )
    u = {(a, f): int(area_allows_flow(a, f, area_functions)) for a in areas for f in flows}
    e20, e40 = build_availability_flags(all_vessels, flows, areas, area_functions, cbar20, cbar20_direct, cbar40)
    p20, p40, old_flags, previous_rows = state.build_previous_plan_params(planning_time, all_vessels)
    model_data = LargePlanningData(
        V=all_vessels,
        F=flows,
        A=areas,
        D20=d20,
        D40=d40,
        port_area_demand20=port_demand20,
        port_area_demand40=port_demand40,
        port_area_allowlist=read_port_sail_area(input_guandong),
        import_cluster_demand20=import_cluster_demand20,
        import_cluster_demand40=import_cluster_demand40,
        import_cluster_snapshot20=import_cluster_snapshot20,
        import_cluster_snapshot40=import_cluster_snapshot40,
        L20=l20,
        L40=l40,
        Q20=q20,
        Q40=q40,
        C20=c20,
        C20Direct=c20_direct,
        C40=c40,
        Cbar20=cbar20,
        Cbar20Direct=cbar20_direct,
        Cbar40=cbar40,
        H=effective_load_capacity,
        distance=distance,
        U=u,
        E20=e20,
        E40=e40,
        P20=p20,
        P40=p40,
        O=old_flags,
        OFWorkLanes=of_work_lanes,
        berth_conflict_pairs=close_berth_pairs,
        allowed_areas_by_vessel=allowed_areas_by_vessel,
        required_areas_by_vessel=required_areas_by_vessel,
        priority_areas_by_vessel=priority_areas_by_vessel,
        bay_limit_areas_by_vessel=bay_limit_areas_by_vessel,
        bay_limit_C20=bay_limit_c20,
        bay_limit_C20Direct=bay_limit_c20_direct,
        bay_limit_C40=bay_limit_c40,
        weights=config.weights,
        required_area_penalty=config.required_area_penalty,
        allow_unmet_demand=config.allow_unmet_demand,
        strict_validation=config.strict_validation,
    )
    diagnostics = {
        "data_dir": "",
        "area_count": len(areas),
        "flows": flows,
        "flow_aliases": flow_aliases,
        **area_control_diagnostics,
        "bad_bay_count": len(bad_bays),
        "bad_bay_sample": sorted(list(bad_bays))[:20],
        "current_snapshot_rows": int(len(current_snapshot)),
        "active_tops_rows": int(active_tops_count),
        "capacity20_total": float(sum(c20.values())),
        "capacity20_direct_total": float(sum(c20_direct.values())),
        "capacity40_total": float(sum(c40.values())),
        "load_capacity_original_total": float(sum(load_capacity.values())),
        "load_capacity_effective_total": float(sum(effective_load_capacity.values())),
        "departure_operation_deduction_total": float(sum(departure_deductions.values())),
        "departure_operation_deductions_by_area": departure_deductions,
        "departure_operation_avoidance": departure_avoidance,
        "large_planning_config": config.to_dict(),
        "close_berth_conflict_threshold_hours": config.berth_conflict_threshold_hours,
        "close_berth_conflict_pairs": [list(pair) for pair in close_berth_pairs],
        "close_berth_conflict_pair_details": close_berth_diagnostics,
        "old_vessels": sorted([v for v, flag in old_flags.items() if flag]),
        "of_work_lanes": of_work_lanes,
        "of_area_limits": {vessel: 2.0 * lanes for vessel, lanes in of_work_lanes.items()},
        "port_area_constraint_ports": sorted(read_port_sail_area(input_guandong)),
        "port_area_constrained_demand20_total": float(sum(port_demand20.values())),
        "port_area_constrained_demand40_total": float(sum(port_demand40.values())),
        "port_area_constrained_group_count": int(len(port_demand20) + len(port_demand40)),
        "import_cluster_demand20_total": float(sum(import_cluster_demand20.values())),
        "import_cluster_demand40_total": float(sum(import_cluster_demand40.values())),
        "import_cluster_group_count": int(
            len({key[2] for key in import_cluster_demand20} | {key[2] for key in import_cluster_demand40})
        ),
        "import_cluster_snapshot_group_area_count": int(
            len(import_cluster_snapshot20) + len(import_cluster_snapshot40)
        ),
        "user_design_area_bay_large_limits": bay_limit_summary,
        "demand": demand_diagnostics,
    }
    return (
        PlanningInputArtifacts(
            data=model_data,
            planning_time=planning_time,
            export_vessels=export_vessels,
            import_vessels=import_vessels,
            area_functions=area_functions,
            berth_by_vessel=berth_by_vessel,
            previous_plan_rows=previous_rows,
            diagnostics=diagnostics,
        ),
        state,
    )


def allocation_output_rows(
    solution: Any,
    data: LargePlanningData,
    include_zero: bool = False,
    planning_time: pd.Timestamp | datetime | str | None = None,
) -> list[dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if planning_time is not None:
        metadata["planning_time"] = pd.Timestamp(planning_time).isoformat()
    if getattr(solution, "status_name", None) is not None:
        metadata["status_name"] = solution.status_name
    if getattr(solution, "objective_value", None) is not None:
        metadata["objective_value"] = solution.objective_value
    rows: list[dict[str, Any]] = []
    for size, values in (("20", solution.x20), ("40", solution.x40)):
        for key, qty in values.items():
            snapshot_qty = snapshot_quantity_for_output(data, size, key)
            new_qty = max(0.0, float(qty) - snapshot_qty)
            if not include_zero and not qty and not snapshot_qty and not new_qty:
                continue
            vessel, flow, area = key
            rows.append(
                {
                    "voy_id": vessel,
                    "flow": flow,
                    "area_no": area,
                    "size": size,
                    "planned_qty": int(qty),
                    "snapshot_qty": float(snapshot_qty),
                    "new_qty": float(new_qty),
                    **metadata,
                }
            )
    return rows


def snapshot_quantity_for_output(data: LargePlanningData, size: str, key: tuple[str, str, str]) -> float:
    if size == "20":
        return float(data.S20.get(key, data.L20.get(key, 0.0) + data.Q20.get(key, 0.0)))
    return float(data.S40.get(key, data.L40.get(key, 0.0) + data.Q40.get(key, 0.0)))


def count_flow_function_mismatch_rows(allocation: pd.DataFrame, area_functions: Mapping[str, set[str]], qty_column: str) -> int:
    if allocation.empty or qty_column not in allocation.columns:
        return 0
    rows = allocation[pd.to_numeric(allocation[qty_column], errors="coerce").fillna(0.0) > 1e-6].copy()
    if rows.empty:
        return 0
    return int(rows.apply(lambda row: not area_allows_flow(row["area_no"], row["flow"], area_functions), axis=1).sum())


def write_large_outputs(output_dir: Path, artifacts: PlanningInputArtifacts, solution: Any, state_rows: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    allocation = pd.DataFrame(allocation_output_rows(solution, artifacts.data, planning_time=artifacts.planning_time))
    allocation.to_csv(output_dir / "allocation.csv", index=False, encoding="utf-8-sig")
    allocation_new = allocation[pd.to_numeric(allocation.get("new_qty", 0), errors="coerce").fillna(0.0) > 0].copy()
    allocation_new.to_csv(output_dir / "allocation_new.csv", index=False, encoding="utf-8-sig")
    state_rows.to_csv(output_dir / "state_rows_appended.csv", index=False, encoding="utf-8-sig")
    diagnostics = {
        **artifacts.diagnostics,
        "status": solution.status,
        "status_name": solution.status_name,
        "objective_value": solution.objective_value,
        "best_bound": solution.best_bound,
        "mip_gap": solution.mip_gap,
        "runtime": solution.runtime,
        "objective_components": solution.objective_components,
        "objective_components_normalized": getattr(solution, "objective_components_normalized", {}),
        "objective_layer_values": getattr(solution, "objective_layer_values", {}),
        "unmet20": {str(k): v for k, v in getattr(solution, "s20", {}).items()},
        "unmet40": {str(k): v for k, v in getattr(solution, "s40", {}).items()},
        "required_area_unmet": {
            str(k): v for k, v in getattr(solution, "required_area_unmet", {}).items()
        },
        "berth_conflict_shared": {
            str(k): v for k, v in getattr(solution, "berth_conflict_shared", {}).items()
        },
        "allocation_rows": int(len(allocation)),
        "allocation_new_rows": int(len(allocation_new)),
        "flow_function_mismatch_total": count_flow_function_mismatch_rows(allocation, artifacts.area_functions, "planned_qty"),
        "flow_function_mismatch_new": count_flow_function_mismatch_rows(allocation, artifacts.area_functions, "new_qty"),
    }
    write_json(output_dir / "diagnostics.json", diagnostics)


def read_closed_areas(input_guandong: InputAdapterGd) -> set[str]:
    return input_guandong.closed_area


def calculate_medium_demands(
    input_guandong: InputAdapterGd,
    voyage_ids: list[str] | tuple[str, ...] = DEFAULT_TARGET_VOYAGES,
    planning_time: datetime | None = None,
    big_plan_caps: dict[tuple[str, str, str], int] | None = None,
    horizon_hours: float = 24.0,
) -> list[DemandRow]:
    planning_time = planning_time or parse_datetime(DEFAULT_PLANNING_TIME) or datetime(2026, 5, 19, 9, 30)
    schedules = read_vessel_schedules(input_guandong)
    normalized_voyages = [normalize_voyage(v) for v in voyage_ids]
    export_voyages = classified_export_voyages(input_guandong)
    yard_by_voyage = read_yard_by_voyage_port_size(input_guandong, set(normalized_voyages), planning_time)
    rows: list[DemandRow] = []
    for voyage_id in normalized_voyages:
        receive_start = schedules.get(voyage_id).receive_start if voyage_id in schedules else planning_time
        stage, ratio = planning_stage(receive_start, planning_time, horizon_hours)
        docs = read_doc_by_port_size(input_guandong, voyage_id)
        yard_counts = yard_by_voyage.get(voyage_id, Counter())
        if voyage_id not in export_voyages:
            predicted = Counter()
            ratio_targets = Counter()
            planned_source = Counter(docs)
        else:
            predicted = read_predicted_by_port_size(input_guandong, voyage_id)
            ratio_targets = ratio_targets_by_port(predicted, ratio)
            planned_source = choose_planned_source(net_prediction_targets(ratio_targets, yard_counts), docs)
        for flow, size_mode, port in sorted(planned_source):
            yard_boxes = yard_counts.get((flow, size_mode, port), 0)
            planned = max(0, planned_source[(flow, size_mode, port)])
            if planned <= 0:
                continue
            rows.append(
                DemandRow(
                    voyage_id=voyage_id,
                    flow=flow,
                    port=port,
                    size_mode=size_mode,
                    predicted_boxes=predicted.get((flow, size_mode, port), 0),
                    ratio_target_boxes=ratio_targets.get((flow, size_mode, port), 0),
                    doc_boxes=docs.get((flow, size_mode, port), 0),
                    yard_boxes=yard_boxes,
                    planned_boxes=planned,
                    planning_stage=stage,
                    planning_ratio=ratio,
                )
            )
    if big_plan_caps:
        rows = cap_demand_rows_by_big_plan(rows, big_plan_caps)
    return rows


def net_prediction_targets(
    ratio_targets: Counter[tuple[str, str, str]],
    yard_counts: Counter[tuple[str, str, str]],
) -> Counter[tuple[str, str, str]]:
    return Counter(
        {
            key: max(0, int(qty) - int(yard_counts.get(key, 0)))
            for key, qty in ratio_targets.items()
            if max(0, int(qty) - int(yard_counts.get(key, 0))) > 0
        }
    )


def is_import_voyage(input_guandong: InputAdapterGd, voyage_id: object) -> bool:
    voyage = normalize_voyage(voyage_id)
    return bool(voyage) and voyage not in classified_export_voyages(input_guandong)


def read_yard_by_voyage_port_size(
    input_guandong: InputAdapterGd,
    voyage_ids: set[str],
    planning_time: datetime,
) -> dict[str, Counter[tuple[str, str, str]]]:
    frame = getattr(input_guandong, "bay_slots_detail", None)
    if not voyage_ids or not isinstance(frame, pd.DataFrame) or frame.empty or "HAS_CONTAINER" not in frame.columns:
        return {}
    occupied = frame.loc[frame["HAS_CONTAINER"].fillna(0).astype(int).eq(1)].copy()
    if occupied.empty:
        return {}
    if "IYC_INYTM" in occupied.columns:
        in_time = pd.to_datetime(occupied["IYC_INYTM"], errors="coerce")
        occupied = occupied.loc[in_time.isna() | (in_time <= pd.Timestamp(planning_time))]
    occupied = medium_small_yard_included_rows(occupied)
    if occupied.empty:
        return {}
    occupied["_container_key"] = [container_identity(row, index) for index, row in occupied.iterrows()]
    occupied = occupied.drop_duplicates("_container_key")
    occupied["_flow"] = occupied.get("IYC_STS_CSTATUSCD", pd.Series(index=occupied.index, dtype=object)).map(
        lambda value: normalize_medium_small_flow(value, default="OF")
    )
    occupied["_size"] = occupied.get("IYC_CSZ_CSIZECD", pd.Series(index=occupied.index, dtype=object)).map(
        normalize_size_small
    )
    occupied["_port"] = occupied.get("IYC_POT_UNLDPORT", pd.Series(index=occupied.index, dtype=object)).map(
        lambda value: normalize_text(value, "UNK")
    )
    out: dict[str, Counter[tuple[str, str, str]]] = defaultdict(Counter)

    voyage_columns = []
    if "IYC_EVOY_ID" in occupied.columns:
        voyage_columns.append("IYC_EVOY_ID")
    if "IYC_IVOY_ID" in occupied.columns:
        voyage_columns.append("IYC_IVOY_ID")
    seen_voyage_containers: set[tuple[str, str]] = set()
    for voyage_column in voyage_columns:
        work = occupied.copy()
        work["_voyage"] = work[voyage_column].map(normalize_voyage)
        work = work.loc[work["_voyage"].isin(voyage_ids)].copy()
        if work.empty:
            continue
        work = work.drop_duplicates(["_voyage", "_container_key"])
        keep_mask = []
        for voyage_id, container_key in zip(work["_voyage"], work["_container_key"]):
            key = (str(voyage_id), str(container_key))
            keep = key not in seen_voyage_containers
            keep_mask.append(keep)
            if keep:
                seen_voyage_containers.add(key)
        work = work.loc[keep_mask].copy()
        if work.empty:
            continue
        counts = work.groupby(["_voyage", "_flow", "_size", "_port"], sort=False).size()
        for (voyage_id, flow, size, port), count in counts.items():
            out[str(voyage_id)][(str(flow), str(size), str(port))] += int(count)
    return dict(out)


def existing_coarse_group_loads(
    input_guandong: InputAdapterGd,
    planning_time: datetime,
    target_voyages: set[str],
    valid_bay_keys: set[str],
    attribute_rules: AttributeRules | None = None,
) -> tuple[Counter[tuple[str, ...]], Counter[tuple[str, ...]]]:
    """Count current yard boxes for target voyages by the configured coarse key.

    Current yard boxes are not part of the new demand, but their location is
    useful as a soft anchor for placing the same configured coarse group nearby.
    """
    attribute_rules = attribute_rules or read_attribute_rules(input_guandong, sorted(target_voyages))
    frame = getattr(input_guandong, "bay_slots_detail", None)
    area_load: Counter[tuple[str, ...]] = Counter()
    bay_load: Counter[tuple[str, ...]] = Counter()
    if (
        not target_voyages
        or not valid_bay_keys
        or not isinstance(frame, pd.DataFrame)
        or frame.empty
        or "HAS_CONTAINER" not in frame.columns
    ):
        return area_load, bay_load

    occupied = frame.loc[frame["HAS_CONTAINER"].fillna(0).astype(int).eq(1)].copy()
    if occupied.empty:
        return area_load, bay_load
    if "IYC_INYTM" in occupied.columns:
        in_time = pd.to_datetime(occupied["IYC_INYTM"], errors="coerce")
        occupied = occupied.loc[in_time.isna() | (in_time <= pd.Timestamp(planning_time))]
    occupied = medium_small_yard_included_rows(occupied)
    if occupied.empty:
        return area_load, bay_load

    occupied["_area"] = occupied.get("YAA_AREANO", pd.Series(index=occupied.index, dtype=object)).map(normalize_code)
    occupied["_bay_no"] = occupied.get("YBY_BAYNO", pd.Series(index=occupied.index, dtype=object)).map(normalize_bay)
    occupied["_bay_key"] = occupied["_area"] + "|" + occupied["_bay_no"]
    occupied = occupied.loc[occupied["_bay_key"].isin(valid_bay_keys)].copy()
    if occupied.empty:
        return area_load, bay_load

    occupied["_container_key"] = [container_identity(row, index) for index, row in occupied.iterrows()]
    voyage_columns = []
    if "IYC_EVOY_ID" in occupied.columns:
        voyage_columns.append("IYC_EVOY_ID")
    if "IYC_IVOY_ID" in occupied.columns:
        voyage_columns.append("IYC_IVOY_ID")
    seen_voyage_containers: set[tuple[str, str]] = set()
    seen_anchor_containers: set[tuple[tuple[str, ...], str]] = set()
    export_voyages = classified_export_voyages(input_guandong)
    for voyage_column in voyage_columns:
        work = occupied.copy()
        work["_voyage"] = work[voyage_column].map(normalize_voyage)
        work = work.loc[work["_voyage"].isin(target_voyages)].copy()
        if work.empty:
            continue
        keep_mask = []
        for voyage_id, container_key in zip(work["_voyage"], work["_container_key"]):
            key = (str(voyage_id), str(container_key))
            keep = key not in seen_voyage_containers
            keep_mask.append(keep)
            if keep:
                seen_voyage_containers.add(key)
        work = work.loc[keep_mask].copy()
        if work.empty:
            continue
        for row in work.to_dict("records"):
            voyage_id = str(row.get("_voyage", ""))
            flow = normalize_medium_small_flow(row.get("IYC_STS_CSTATUSCD"), default="OF")
            size = normalize_size_small(row.get("IYC_CSZ_CSIZECD"))
            port = normalize_text(row.get("IYC_POT_UNLDPORT"), "UNK")
            record = normalized_doc_record(row, flow, size, port)
            record["IYC_EVOY_ID"] = normalize_voyage(row.get("IYC_EVOY_ID"))
            record["IYC_IVOY_ID"] = normalize_voyage(row.get("IYC_IVOY_ID"))
            coarse_key = configured_coarse_anchor_key(record, voyage_id, attribute_rules, export_voyages)
            container_key = str(row.get("_container_key", ""))
            anchor_container_key = (coarse_key, container_key)
            if container_key and anchor_container_key in seen_anchor_containers:
                continue
            if container_key:
                seen_anchor_containers.add(anchor_container_key)
            area_no = str(row.get("_area", ""))
            bay_key = str(row.get("_bay_key", ""))
            if not area_no or not bay_key:
                continue
            area_load[coarse_key + (area_no,)] += 1
            bay_load[coarse_key + (area_no, bay_key)] += 1
    return area_load, bay_load


def configured_coarse_anchor_key(
    row: Mapping[str, Any],
    voyage_id: str,
    attribute_rules: AttributeRules,
    export_voyages: set[str],
) -> tuple[str, ...]:
    flow = normalize_medium_small_flow(row.get("IYC_STS_CSTATUSCD"), default="OF")
    size = normalize_size_small(row.get("IYC_CSZ_CSIZECD"))
    port = normalize_text(row.get("IYC_POT_UNLDPORT"), "UNK")
    if voyage_id in export_voyages:
        attrs = medium_groupby_attributes(attribute_rules, voyage_id)
    else:
        attrs, _values, _port_label = import_base_group_attributes(row, size, port)
    values = dynamic_attributes_from_row(
        row,
        attrs,
        levels=attribute_rules.weight_levels_for(voyage_id),
    )
    scope = str(voyage_id) if voyage_id in export_voyages else "IMPORT"
    return (scope, *(f"{attr}={values.get(attr, 'MIXED')}" for attr in attrs))


def container_identity(row: pd.Series, index: object) -> str:
    number = normalize_code(row.get("IYC_CNTRNO"))
    if number:
        return f"NO:{number}"
    cntr_id = normalize_code(row.get("IYC_CNTRID"))
    if cntr_id and cntr_id not in {"-1", "0"}:
        return f"ID:{cntr_id}"
    return f"ROW:{index}"


def medium_demand_caps_from_big_plan(
    big_plan: list[BigPlanRow],
    target_voyages: list[str] | tuple[str, ...] | set[str],
    planning_time: datetime,
    target_flows: set[str] | frozenset[str] = DEFAULT_TARGET_BIG_PLAN_FLOWS,
) -> dict[tuple[str, str, str], int]:
    plan_date = planning_time.date().isoformat()
    voyage_set = {normalize_voyage(voyage_id) for voyage_id in target_voyages}
    normalized_flows = {medium_small_area_flow(flow) for flow in target_flows}
    size_pool: Counter[tuple[str, str, str]] = Counter()
    all_size_pool: Counter[tuple[str, str]] = Counter()
    for row in big_plan:
        if row.voyage_id not in voyage_set:
            continue
        row_flow = medium_small_area_flow(row.flow)
        if row_flow not in normalized_flows:
            continue
        if row.plan_date and row.plan_date != plan_date:
            continue
        if row.size_mode == "ALL":
            all_size_pool[(row.voyage_id, row_flow)] += row.planned_boxes
        else:
            size_pool[(row.voyage_id, row_flow, "40" if row.size_mode == "45" else row.size_mode)] += row.planned_boxes

    caps: dict[tuple[str, str, str], int] = {}
    for (voyage_id, flow, size_mode), qty in size_pool.items():
        if qty <= 0:
            continue
        caps[(voyage_id, flow, size_mode)] = qty
    for (voyage_id, flow), qty in all_size_pool.items():
        if qty <= 0 or any(v == voyage_id and f == flow for v, f, _size in size_pool):
            continue
        caps[(voyage_id, flow, "ALL")] = qty
    return caps


def cap_demand_rows_by_big_plan(
    rows: list[DemandRow],
    big_plan_caps: dict[tuple[str, str, str], int],
) -> list[DemandRow]:
    grouped: defaultdict[tuple[str, str, str], list[DemandRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.voyage_id, row.flow, "40" if row.size_mode == "45" else row.size_mode)].append(row)

    capped_rows: list[DemandRow] = []
    for key, items in grouped.items():
        total = sum(item.planned_boxes for item in items)
        cap_flow = medium_small_area_flow(key[1])
        cap = big_plan_caps.get((key[0], cap_flow, key[2]))
        if cap is None:
            cap = big_plan_caps.get((key[0], cap_flow, "ALL"))
        if cap is None:
            continue
        if total <= cap:
            capped_rows.extend(items)
            continue
        scaled = largest_remainder_scale([item.planned_boxes for item in items], total, cap)
        for item, planned_boxes in zip(items, scaled):
            if planned_boxes > 0:
                capped_rows.append(replace(item, planned_boxes=planned_boxes))
    return sorted(capped_rows, key=lambda row: (row.voyage_id, row.flow, row.size_mode, row.port))


def planning_stage(receive_start: datetime, planning_time: datetime, horizon_hours: float = 24.0) -> tuple[str, float]:
    horizon = timedelta(hours=horizon_hours if horizon_hours > 0 else 24.0)
    if planning_time < receive_start:
        if planning_time + horizon >= receive_start:
            return "before_open_within_24h", 0.70
        return "before_open_beyond_24h", 0.0
    elapsed = planning_time - receive_start
    if elapsed < horizon:
        return "open_first_24h", 0.70
    if elapsed < horizon * 2:
        return "open_second_24h", 0.90
    return "open_third_24h_or_later", 1.00


def read_predicted_by_port_size(input_guandong: InputAdapterGd, voyage_id: str) -> Counter[tuple[str, str, str]]:
    vessel_containers = input_guandong.vessel_containers.get(voyage_id, {}).get("predict_cntrs", {})

    counter: Counter[tuple[str, str, str]] = Counter()
    # for row in frame.to_dict("records"):
    for size, volume_info in vessel_containers.items():
        for port, volume in volume_info.get("detail_info", {}).items():
            size_mode = normalize_size_small(size)
            port = normalize_text(port, "UNK")
            flow = normalize_flow("OF", default="OF")
            counter[(flow, size_mode, port)] += int(round(float(volume or 0)))
    return counter


def ratio_targets_by_port(predicted: Counter[tuple[str, str, str]], ratio: float) -> Counter[tuple[str, str, str]]:
    targets: Counter[tuple[str, str, str]] = Counter()
    by_flow_size: defaultdict[tuple[str, str], list[tuple[tuple[str, str, str], int]]] = defaultdict(list)
    for key, count in predicted.items():
        by_flow_size[(key[0], key[1])].append((key, count))
    for items in by_flow_size.values():
        total = sum(count for _, count in items)
        target_total = int(round(total * ratio))
        scaled = largest_remainder_scale([count for _, count in items], total, target_total)
        for (key, _), qty in zip(items, scaled):
            if qty > 0:
                targets[key] += qty
    return targets


def read_doc_by_port_size(input_guandong: InputAdapterGd, voyage_id: str) -> Counter[tuple[str, str, str]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    frame = input_guandong.vessel_containers.get(voyage_id, {}).get("doc_cntrs", None)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return counter
    work = pd.DataFrame(
        {
            "flow": frame.get("IYC_STS_CSTATUSCD", pd.Series(index=frame.index, dtype=object)).map(
                lambda value: normalize_medium_small_flow(value, default="OF")
            ),
            "size": frame.get("IYC_CSZ_CSIZECD", pd.Series(index=frame.index, dtype=object)).map(normalize_size_small),
            "port": frame.get("IYC_POT_UNLDPORT", pd.Series(index=frame.index, dtype=object)).map(
                lambda value: normalize_text(value, "UNK")
            ),
        }
    )
    counts = work.groupby(["flow", "size", "port"], sort=False).size()
    counter.update({(str(flow), str(size), str(port)): int(count) for (flow, size, port), count in counts.items()})
    return counter


def choose_planned_source(
    ratio_targets: Counter[tuple[str, str, str]],
    docs: Counter[tuple[str, str, str]],
) -> Counter[tuple[str, str, str]]:
    planned: Counter[tuple[str, str, str]] = Counter()
    for key in sorted(set(ratio_targets) | set(docs)):
        qty = max(int(ratio_targets.get(key, 0)), int(docs.get(key, 0)))
        if qty > 0:
            planned[key] += qty
    return planned


def largest_remainder_scale(counts: list[int], source_total: int, target_total: int) -> list[int]:
    if target_total <= 0:
        return [0 for _ in counts]
    if source_total <= 0:
        out = [0 for _ in counts]
        if out:
            out[0] = target_total
        return out
    raw = [count * target_total / source_total for count in counts]
    base = [int(value) for value in raw]
    remain = target_total - sum(base)
    order = sorted(range(len(raw)), key=lambda idx: raw[idx] - base[idx], reverse=True)
    for idx in order[:remain]:
        base[idx] += 1
    return base


def write_demand_rows(path: Path, rows: list[DemandRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        fieldnames = [
            "voyage_id",
            "flow",
            "port",
            "size_mode",
            "predicted_boxes",
            "ratio_target_boxes",
            "doc_boxes",
            "yard_boxes",
            "planned_boxes",
            "planning_stage",
            "planning_ratio",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def read_big_plan(large_plan: pd.DataFrame) -> list[BigPlanRow]:
    counter: Counter[tuple[str, str, str, str, str]] = Counter()
    rows: list[BigPlanRow] = []
    reader = large_plan.to_dict(orient='records')
    fieldnames = set(large_plan.columns.tolist())

    if {"voyage_id", "area_no"}.issubset(fieldnames) and (
        {"qty_20", "qty_40"}.issubset(fieldnames)
        or {"planned_20", "planned_40"}.issubset(fieldnames)
        or {"20", "40"}.issubset(fieldnames)
    ):
        qty20_field = _first_existing(fieldnames, ["qty_20", "planned_20", "20", "c20", "C20"])
        qty40_field = _first_existing(fieldnames, ["qty_40", "planned_40", "40", "c40", "C40"])
        qty45_field = _first_existing(fieldnames, ["qty_45", "planned_45", "45", "c45", "C45"])
        date_field = _first_existing(fieldnames, ["plan_date", "date", "work_date", "planning_date", "day"])
        flow_field = _first_existing(fieldnames, ["flow", "cntr_type", "status"])
        for row in reader:
            flow = normalize_medium_small_flow(row.get(flow_field), default="OF") if flow_field else "OF"
            voyage_id = normalize_voyage(row.get("voyage_id"))
            area_no = normalize_code(row.get("area_no"))
            plan_date = date_key(normalize_text(row.get(date_field))) if date_field else ""
            for size_mode, field_name in (("20", qty20_field), ("40", qty40_field), ("45", qty45_field)):
                if not field_name:
                    continue
                boxes = int(round(float(row.get(field_name, 0) or 0)))
                if boxes > 0:
                    counter[(voyage_id, flow, area_no, size_mode, plan_date)] += boxes
    elif {"voy_id", "area_no"}.issubset(fieldnames) and (
        "new_qty" in fieldnames or "planned_qty" in fieldnames
    ):
        qty_field = "new_qty" if "new_qty" in fieldnames else "planned_qty"
        date_field = _first_existing(fieldnames, ["plan_date", "date", "work_date", "planning_date", "day"])
        flow_field = _first_existing(fieldnames, ["flow", "cntr_type", "status"])
        for row in reader:
            flow = normalize_medium_small_flow(row.get(flow_field), default="OF") if flow_field else "OF"
            boxes = int(round(float(row.get(qty_field, 0) or 0)))
            if boxes > 0:
                counter[
                    (
                        normalize_voyage(row["voy_id"]),
                        flow,
                        normalize_code(row["area_no"]),
                        normalize_big_plan_size(row.get("size")),
                        date_key(normalize_text(row.get(date_field))) if date_field else "",
                    )
                ] += boxes
    elif {"voyage_id", "area_no", "planned_boxes"}.issubset(fieldnames):
        size_field = "size_mode" if "size_mode" in fieldnames else "size"
        date_field = _first_existing(fieldnames, ["plan_date", "date", "work_date", "planning_date", "day"])
        flow_field = _first_existing(fieldnames, ["flow", "cntr_type", "status"])
        for row in reader:
            flow = normalize_medium_small_flow(row.get(flow_field), default="OF") if flow_field else "OF"
            boxes = int(round(float(row["planned_boxes"])))
            if boxes > 0:
                counter[
                    (
                        normalize_voyage(row["voyage_id"]),
                        flow,
                        normalize_code(row["area_no"]),
                        normalize_big_plan_size(row.get(size_field)),
                        date_key(normalize_text(row.get(date_field))) if date_field else "",
                    )
                ] += boxes
    else:
        raise ValueError(f"Unsupported big plan columns: {sorted(fieldnames)}")
    rows = [
        BigPlanRow(voyage_id, flow, area_no, boxes, size_mode, plan_date)
        for (voyage_id, flow, area_no, size_mode, plan_date), boxes in sorted(counter.items())
        if boxes > 0
    ]
    if not rows:
        raise ValueError("big plan file contains no positive planned boxes")
    return rows


def load_port_demand_groups(
    input_guandong: InputAdapterGd,
    voyage_ids: list[str],
    planning_time: datetime,
    attribute_rules: AttributeRules,
    big_plan_caps: dict[tuple[str, str, str], int] | None = None,
    horizon_hours: float = 24.0,
    demand_rows: list[DemandRow] | None = None,
) -> tuple[list[BoxGroup], list[DemandRow]]:
    if demand_rows is None:
        demand_rows = calculate_medium_demands(
            input_guandong,
            voyage_ids,
            planning_time,
            big_plan_caps=big_plan_caps,
            horizon_hours=horizon_hours,
        )
    else:
        demand_rows = list(demand_rows)
    groups: list[BoxGroup] = []
    group_index: defaultdict[str, int] = defaultdict(int)
    counters: defaultdict[str, Counter[tuple]] = defaultdict(Counter)
    port_sail_area = read_port_sail_area(input_guandong)
    doc_frames: dict[str, pd.DataFrame] = {}
    export_voyages = classified_export_voyages(input_guandong)
    for voyage_id in voyage_ids:
        frame = input_guandong.vessel_containers.get(voyage_id, {}).get("doc_cntrs", None)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        if not frame.empty:
            doc_frames[voyage_id] = frame

    import_counters: Counter[tuple] = Counter()
    for voyage_id, frame in doc_frames.items():
        export_voyage = voyage_id in export_voyages
        if export_voyage:
            continue
        for record in frame.to_dict("records"):
            flow = normalize_medium_small_flow(record.get("IYC_STS_CSTATUSCD"), default="OF")
            size = normalize_size_small(record.get("IYC_CSZ_CSIZECD"))
            port = normalize_text(record.get("IYC_POT_UNLDPORT"), "UNK")
            normalized_record = normalized_doc_record(record, flow, size, port)
            group_attrs, values, port_label = import_base_group_attributes(normalized_record, size, port)
            core = (flow, size, port_label)
            key = tuple(values.get(attr, "MIXED") for attr in group_attrs)
            constraint_port = port if port in port_sail_area else ""
            import_counters[(voyage_id, flow, "40" if size == "45" else size, core, group_attrs, key, constraint_port)] += 1

    import_by_cap_key: defaultdict[tuple[str, str, str], list[tuple[tuple, int]]] = defaultdict(list)
    for full_key, qty in import_counters.items():
        voyage_id, flow, big_size, _core, _group_attrs, _key, _constraint_port = full_key
        import_by_cap_key[(voyage_id, medium_small_area_flow(flow), big_size)].append((full_key, int(qty)))
    for cap_key, items in import_by_cap_key.items():
        total = sum(qty for _key, qty in items)
        cap = None
        if big_plan_caps:
            cap = big_plan_caps.get(cap_key)
            if cap is None:
                cap = big_plan_caps.get((cap_key[0], cap_key[1], "ALL"))
            if cap is None:
                continue
        target_total = min(total, int(cap)) if cap is not None else total
        if target_total <= 0:
            continue
        scaled = largest_remainder_scale([qty for _key, qty in items], total, target_total)
        for (full_key, _qty), planned_qty in zip(items, scaled):
            if planned_qty <= 0:
                continue
            voyage_id, _flow, _big_size, core, group_attrs, key, constraint_port = full_key
            counters[voyage_id][(core, group_attrs, key, constraint_port)] += int(planned_qty)

    for row in demand_rows:
        if row.voyage_id not in export_voyages:
            continue
        base_attrs = {
            "status": row.flow,
            "flow": row.flow,
            "IYC_STS_CSTATUSCD": row.flow,
            "size": row.size_mode,
            "size_mode": row.size_mode,
            "IYC_CSZ_CSIZECD": row.size_mode,
            "port": row.port,
            "IYC_POT_UNLDPORT": row.port,
            "height": "UNK",
            "IYC_CHEIGHTCD": "UNK",
            "weight_class": "UNK",
            "weight": "UNK",
            "special_stow_code": "",
            "pre_stow": False,
        }
        group_attrs = medium_groupby_attributes(attribute_rules, row.voyage_id)
        frame = doc_frames.get(row.voyage_id)
        matching = pd.DataFrame()
        if frame is not None and not frame.empty:
            status = frame.get("IYC_STS_CSTATUSCD", pd.Series(index=frame.index, dtype=object)).map(
                lambda value: normalize_medium_small_flow(value, default="OF")
            )
            size = frame.get("IYC_CSZ_CSIZECD", pd.Series(index=frame.index, dtype=object)).map(normalize_size_small)
            port = frame.get("IYC_POT_UNLDPORT", pd.Series(index=frame.index, dtype=object)).map(lambda value: normalize_text(value, "UNK"))
            matching = frame.loc[status.eq(row.flow) & size.eq(row.size_mode) & port.eq(row.port)].copy()
        core = (row.flow, row.size_mode, row.port)
        if not matching.empty:
            weights: Counter[tuple] = Counter()
            levels = attribute_rules.weight_levels_for(row.voyage_id)
            for record in matching.to_dict("records"):
                values = dynamic_attributes_from_row(record, group_attrs, levels=levels)
                weights[tuple(values.get(attr, "MIXED") for attr in group_attrs)] += 1
            items = sorted(weights.items())
            scaled = largest_remainder_scale([count for _, count in items], sum(weights.values()), int(row.planned_boxes))
            for (key, _count), qty in zip(items, scaled):
                if qty > 0:
                    counters[row.voyage_id][(core, group_attrs, key, "")] += int(qty)
        else:
            values = dynamic_attributes_from_row(base_attrs, group_attrs, levels=attribute_rules.weight_levels_for(row.voyage_id))
            key = tuple(values.get(attr, "MIXED") for attr in group_attrs)
            counters[row.voyage_id][(core, group_attrs, key, "")] += int(row.planned_boxes)
    for voyage_id in sorted(counters):
        for (core, group_attrs, key, constraint_port), demand in sorted(counters[voyage_id].items(), key=lambda item: (item[0][0], item[0][2])):
            core_status, core_size, core_port = core
            values = dict(zip(group_attrs, key))
            group_index[voyage_id] += 1
            status = str(core_status)
            size = str(core_size)
            port = str(core_port)
            groups.append(
                BoxGroup(
                    group_id=f"{voyage_id}_P{group_index[voyage_id]:03d}",
                    voyage_id=voyage_id,
                    size=size,
                    height=str(values.get("IYC_CHEIGHTCD", "UNK")),
                    status=status,
                    port=port,
                    operator="UNK",
                    ctype="UNK",
                    weight_class=str(values.get("IYC_CWEIGHT", "UNK")),
                    reefer=False,
                    dangerous=False,
                    over_limit=False,
                    special_codes=(),
                    demand=demand,
                    attributes={str(k): str(v) for k, v in values.items()},
                    area_allowlist=set(port_sail_area[constraint_port]) if constraint_port else None,
                )
            )
    return groups, demand_rows


def current_yard_container_keys(input_guandong: InputAdapterGd, planning_time: datetime) -> tuple[set[str], set[str]]:
    frame = getattr(input_guandong, "bay_slots_detail", None)
    if not isinstance(frame, pd.DataFrame) or frame.empty or "HAS_CONTAINER" not in frame.columns:
        return set(), set()
    occupied = frame.loc[frame["HAS_CONTAINER"].fillna(0).astype(int).eq(1)].copy()
    if occupied.empty:
        return set(), set()
    if "IYC_INYTM" in occupied.columns:
        in_time = pd.to_datetime(occupied["IYC_INYTM"], errors="coerce")
        occupied = occupied.loc[in_time.isna() | (in_time <= pd.Timestamp(planning_time))]
    occupied = medium_small_yard_included_rows(occupied)
    ids = set()
    numbers = set()
    if "IYC_CNTRID" in occupied.columns:
        ids = {
            key
            for key in (normalize_code(value) for value in occupied["IYC_CNTRID"])
            if key and key not in {"-1", "0"}
        }
    if "IYC_CNTRNO" in occupied.columns:
        numbers = {
            key
            for key in (normalize_code(value) for value in occupied["IYC_CNTRNO"])
            if key and key not in {"-1", "0"}
        }
    return ids, numbers


def load_small_doc_groups(
    input_guandong: InputAdapterGd,
    voyage_ids: list[str],
    attribute_rules: AttributeRules,
    planning_time: datetime | None = None,
    big_plan_caps: dict[tuple[str, str, str], int] | None = None,
) -> list[SmallBoxGroup]:
    planning_time = planning_time or parse_datetime(DEFAULT_PLANNING_TIME) or datetime(2026, 5, 19, 9, 30)
    groups: list[SmallBoxGroup] = []
    port_sail_area = read_port_sail_area(input_guandong)
    export_voyages = classified_export_voyages(input_guandong)
    for voyage_id in voyage_ids:
        frame = input_guandong.vessel_containers.get(voyage_id, {}).get("doc_cntrs", None)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        export_voyage = voyage_id in export_voyages
        levels = attribute_rules.weight_levels_for(voyage_id)
        group_by_weight = getattr(attribute_rules, "weight_group_enabled_for", lambda _voyage_id: False)(voyage_id)
        counter: Counter[tuple] = Counter()
        for row in frame.to_dict("records"):
            flow = normalize_medium_small_flow(row.get("IYC_STS_CSTATUSCD"), default="OF")
            size = normalize_size_small(row.get("IYC_CSZ_CSIZECD"))
            port = normalize_text(row.get("IYC_POT_UNLDPORT"), "UNK")
            normalized_record = normalized_doc_record(row, flow, size, port)
            if export_voyage:
                group_columns = small_groupby_columns(attribute_rules, voyage_id)
                port_label = port
            else:
                group_columns = import_small_groupby_columns(attribute_rules, voyage_id, normalized_record, size, port)
                _base_attrs, _base_values, port_label = import_base_group_attributes(normalized_record, size, port)
            row_weight_class = weight_class(row.get("IYC_CWEIGHT"), levels) if group_by_weight else "MIXED"
            core = (
                flow,
                size,
                port_label,
                normalize_text(row.get("IYC_CHEIGHTCD"), "UNK"),
                row_weight_class,
                explicit_special_stow_code(row),
                "0",
            )
            values = dynamic_attributes_from_row(normalized_record, group_columns, levels=levels)
            constraint_port = port if not export_voyage and port in port_sail_area else ""
            counter[(core, group_columns, tuple(values.get(column, "") for column in group_columns), constraint_port)] += 1
        planned_counter: Counter[tuple] = Counter()
        import_by_cap_key: defaultdict[tuple[str, str, str], list[tuple[tuple, int]]] = defaultdict(list)
        for key, qty in counter.items():
            core, _group_columns, _dynamic_key, _constraint_port = key
            flow, size, _port, _height, _weight, _special_code, _pre_stow_value = core
            if export_voyage:
                planned_counter[key] += int(qty)
                continue
            import_by_cap_key[(voyage_id, medium_small_area_flow(flow), "40" if size == "45" else size)].append((key, int(qty)))
        for cap_key, items in import_by_cap_key.items():
            total = sum(qty for _key, qty in items)
            cap = None
            if big_plan_caps:
                cap = big_plan_caps.get(cap_key)
                if cap is None:
                    cap = big_plan_caps.get((cap_key[0], cap_key[1], "ALL"))
                if cap is None:
                    continue
            target_total = min(total, int(cap)) if cap is not None else total
            if target_total <= 0:
                continue
            scaled = largest_remainder_scale([qty for _key, qty in items], total, target_total)
            for (key, _qty), planned_qty in zip(items, scaled):
                if planned_qty > 0:
                    planned_counter[key] += int(planned_qty)

        for index, (key, demand) in enumerate(sorted(planned_counter.items()), start=1):
            core, group_columns, dynamic_key, constraint_port = key
            status, size, port, height, weight, special_code, pre_stow_value = core
            values = dict(zip(group_columns, dynamic_key))
            groups.append(
                SmallBoxGroup(
                    group_id=f"{voyage_id}_S{index:03d}",
                    voyage_id=voyage_id,
                    status=str(status),
                    port=str(port),
                    size=str(size),
                    height=str(height),
                    weight_class=str(weight),
                    demand=demand,
                    pre_stow=str(pre_stow_value) == "1",
                    special_stow=bool(special_code),
                    special_stow_code=str(special_code),
                    attributes={str(k): str(v) for k, v in values.items()},
                    area_allowlist=set(port_sail_area[constraint_port]) if constraint_port else None,
                )
            )
    return groups


def build_bays(
    input_guandong: InputAdapterGd,
    allowed_areas: set[str],
    closed_areas: set[str],
    planning_time: datetime,
    vessel_schedules: dict[str, VoyageSchedule],
    target_voyages: set[str],
    area_functions: dict[str, set[str]],
    misplaced_bay_exclusion_ratio: float,
    attribute_rules: AttributeRules | None = None,
) -> tuple[dict[str, Bay], int, int, int]:
    frame = input_guandong.bay_slots_detail.copy()
    frame["YAA_AREANO"] = frame["YAA_AREANO"].map(normalize_code)
    frame["YBY_BAYNO"] = frame["YBY_BAYNO"].map(normalize_bay)
    frame["YST_ROWNO"] = frame["YST_ROWNO"].map(normalize_row)
    frame = frame[frame["YAA_AREANO"].isin(allowed_areas) & ~frame["YAA_AREANO"].isin(closed_areas)].copy()

    original_bay_capacity = total_slots_by_bay(frame)
    reserved_slots, closed_bays = tops_reserved_slots(input_guandong, frame, planning_time, target_voyages)
    frame = drop_reserved_slots(frame, reserved_slots)
    excluded_bays = misplaced_bays_to_exclude(
        frame,
        area_functions,
        vessel_schedules,
        planning_time,
        misplaced_bay_exclusion_ratio,
        original_bay_capacity,
        target_voyages,
    )
    frame = drop_bays(frame, excluded_bays)

    large_bay_partner_by_bay = large_bay_partner_lookup_by_bay(frame)
    existing_large_pairs_by_member = existing_large_pair_members_by_bay(
        frame,
        vessel_schedules,
        planning_time,
    )
    large_shadow_slots = active_large_container_shadow_slots(
        frame,
        vessel_schedules,
        planning_time,
        large_bay_partner_by_bay,
        existing_large_pairs_by_member,
    )
    available = drop_shadow_slots(
        available_or_released_slots(frame, vessel_schedules, planning_time),
        large_shadow_slots,
    )
    cap_by_size = capacity_by_bay_size(available)
    physical_cap = physical_capacity_by_bay(available)
    row_cap_by_size = capacity_by_bay_row_size(available)
    row_physical_cap = physical_capacity_by_bay_row(available)
    row_cap_by_size_by_bay: dict[str, defaultdict[tuple[str, str], dict[str, int]]] = {
        size: defaultdict(dict) for size in SIZE_MODES
    }
    for size in SIZE_MODES:
        for (area_no, bay_no, row_no), qty in row_cap_by_size[size].items():
            row_cap_by_size_by_bay[size][(area_no, bay_no)][row_no] = qty
    row_physical_cap_by_bay: defaultdict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for (area_no, bay_no, row_no), qty in row_physical_cap.items():
        row_physical_cap_by_bay[(area_no, bay_no)][row_no] = qty
    existing_attrs = existing_bay_attributes(
        frame,
        vessel_schedules,
        planning_time,
        attribute_rules,
        large_bay_partner_by_bay=large_bay_partner_by_bay,
        existing_large_pairs_by_member=existing_large_pairs_by_member,
    )
    bays: dict[str, Bay] = {}
    by_area: defaultdict[str, list[str]] = defaultdict(list)
    large_bay_partner: dict[tuple[str, str], str] = {}
    for area_no, bay_no in sorted(physical_cap, key=lambda item: (item[0], bay_sort_key(item[1]))):
        bay_key = f"{area_no}|{bay_no}"
        by_area[area_no].append(bay_key)
    block_lookup: dict[str, tuple[str, tuple[str, ...], bool]] = {}
    for area_no, bay_keys in by_area.items():
        bay_nos = [key.split("|", 1)[1] for key in bay_keys]
        large_bay_partner.update(
            apply_large_bay_pair_capacities(
                area_no,
                bay_nos,
                cap_by_size,
                physical_cap,
                existing_large_pairs_by_member,
            )
        )
        big_starts = {
            bay
            for bay in bay_nos
            if cap_by_size["40"].get((area_no, bay), 0) > 0 or cap_by_size["45"].get((area_no, bay), 0) > 0
        }
        for index, members, adjusted in make_yard_blocks(bay_nos, big_starts):
            block_id = f"{area_no}-B{index:02d}"
            member_keys = tuple(f"{area_no}|{bay_no}" for bay_no in members)
            for key in member_keys:
                block_lookup[key] = (block_id, member_keys, adjusted)
    for (area_no, bay_no), physical in physical_cap.items():
        bay_key = f"{area_no}|{bay_no}"
        block_id, block_members, adjusted = block_lookup.get(bay_key, ("", tuple(), False))
        attrs = existing_attrs.get((area_no, bay_no), {})
        bays[bay_key] = Bay(
            area_no=area_no,
            bay_no=bay_no,
            bay_key=bay_key,
            block_id=block_id,
            block_bays=block_members,
            block_bay_count=len(block_members),
            block_boundary_adjusted=adjusted,
            bay_order=bay_code_value(bay_no) or 0,
            cap_by_size={size: cap_by_size[size].get((area_no, bay_no), 0) for size in SIZE_MODES},
            physical_capacity=physical,
            row_cap_by_size={
                size: dict(row_cap_by_size_by_bay[size].get((area_no, bay_no), {}))
                for size in SIZE_MODES
            },
            row_physical_capacity=dict(row_physical_cap_by_bay.get((area_no, bay_no), {})),
            large_bay_partner_no=large_bay_partner.get((area_no, bay_no), ""),
            large_bay_partner_key=(
                f"{area_no}|{large_bay_partner[(area_no, bay_no)]}"
                if (area_no, bay_no) in large_bay_partner
                else ""
            ),
            existing_size_modes=set(attrs.get("sizes", set())),
            existing_heights=set(attrs.get("heights", set())),
            existing_special_signatures=set(attrs.get("specials", set())),
            existing_ports=set(attrs.get("ports", set())),
            existing_attrs={str(k): set(v) for k, v in attrs.get("attributes", {}).items()},
            existing_attrs_by_row={
                str(row_no): {str(k): set(v) for k, v in row_attrs.items()}
                for row_no, row_attrs in attrs.get("attributes_by_row", {}).items()
            },
            existing_attrs_by_voyage={
                str(voyage): {str(k): set(v) for k, v in voyage_attrs.items()}
                for voyage, voyage_attrs in attrs.get("attributes_by_voyage", {}).items()
            },
            existing_attrs_by_row_by_voyage={
                str(row_no): {
                    str(voyage): {str(k): set(v) for k, v in voyage_attrs.items()}
                    for voyage, voyage_attrs in row_voyage_attrs.items()
                }
                for row_no, row_voyage_attrs in attrs.get("attributes_by_row_by_voyage", {}).items()
            },
        )
    return bays, len(reserved_slots), len(closed_bays), len(excluded_bays)


def tops_reserved_slots(
    input_guandong: InputAdapterGd,
    frame: pd.DataFrame,
    planning_time: datetime,
    target_voyages: set[str],
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str]]]:
    active = active_tops_rows(input_guandong, planning_time)
    active = active[~active["condition_vessel"].isin({normalize_voyage(v) for v in target_voyages})].copy()
    reserved: set[tuple[str, str, str]] = set()
    closed_bays: set[tuple[str, str]] = set()
    if active.empty or frame.empty:
        return reserved, closed_bays
    empty = frame[frame["HAS_CONTAINER"].fillna(0).astype(int).eq(0)].copy()
    empty["_bay_code"] = empty["YBY_BAYNO"].map(bay_code_value)
    empty["_row_code"] = empty["YST_ROWNO"].map(bay_code_value)
    by_area = {area: sub for area, sub in empty.groupby("YAA_AREANO")}
    for _, tops in active.iterrows():
        start_area, start_bay = parse_tops_area_bay(tops.get("SPR_STBAY"))
        end_area, end_bay = parse_tops_area_bay(tops.get("SPR_EDBAY"))
        area = start_area or end_area
        if start_area and end_area and start_area != end_area:
            area = end_area
        if not area or area not in by_area:
            continue
        sub = by_area[area]
        matched = sub[
            slot_range_mask_preparsed(
                sub["YBY_BAYNO"],
                sub["_bay_code"],
                start_bay,
                end_bay,
                bay_code_value,
            )
        ].copy()
        if matched.empty:
            continue
        start_row = normalize_row(tops.get("SPR_STROW"))
        end_row = normalize_row(tops.get("SPR_EDROW"))
        if start_row or end_row:
            matched = matched[
                slot_range_mask_preparsed(
                    matched["YST_ROWNO"],
                    matched["_row_code"],
                    start_row,
                    end_row,
                    bay_code_value,
                )
            ]
        for row in matched.to_dict("records"):
            reserved.add((row["YAA_AREANO"], row["YBY_BAYNO"], row["YST_ROWNO"]))
            closed_bays.add((row["YAA_AREANO"], row["YBY_BAYNO"]))
    return reserved, closed_bays


def drop_reserved_slots(frame: pd.DataFrame, reserved_slots: set[tuple[str, str, str]]) -> pd.DataFrame:
    if not reserved_slots or frame.empty:
        return frame
    empty = frame["HAS_CONTAINER"].fillna(0).astype(int).eq(0)
    keys = list(zip(frame["YAA_AREANO"], frame["YBY_BAYNO"], frame["YST_ROWNO"]))
    mask = [not (is_empty and key in reserved_slots) for is_empty, key in zip(empty, keys)]
    return frame.loc[mask].copy()


def drop_bays(frame: pd.DataFrame, excluded_bays: set[tuple[str, str]]) -> pd.DataFrame:
    if not excluded_bays or frame.empty:
        return frame
    keys = list(zip(frame["YAA_AREANO"], frame["YBY_BAYNO"]))
    return frame.loc[[key not in excluded_bays for key in keys]].copy()


def active_occupied(frame: pd.DataFrame, vessel_schedules: dict[str, VoyageSchedule], planning_time: datetime) -> pd.DataFrame:
    """Current physical yard occupancy; vessel schedule arguments are kept for caller compatibility."""
    return frame[frame["HAS_CONTAINER"].fillna(0).astype(int).eq(1)].copy()


def available_or_released_slots(
    frame: pd.DataFrame,
    vessel_schedules: dict[str, VoyageSchedule],
    planning_time: datetime,
) -> pd.DataFrame:
    """Current physical empty slots only; planned departures do not release snapshot containers."""
    return frame[frame["HAS_CONTAINER"].fillna(0).astype(int).eq(0)].copy()


def slot_identity(row: Mapping[str, Any], area_no: str | None = None, bay_no: str | None = None) -> tuple[str, str, str, str, str]:
    return (
        normalize_code(area_no if area_no is not None else row.get("YAA_AREANO")),
        normalize_bay(bay_no if bay_no is not None else row.get("YBY_BAYNO")),
        normalize_row(row.get("YST_ROWNO")),
        normalize_row(row.get("YST_TIERNO")),
        normalize_row(row.get("YST_SLOTNO")),
    )


def slot_identities(frame: pd.DataFrame) -> list[tuple[str, str, str, str, str]]:
    if frame.empty:
        return []
    areas = frame.get("YAA_AREANO", pd.Series(index=frame.index, dtype=object)).map(normalize_code)
    bays = frame.get("YBY_BAYNO", pd.Series(index=frame.index, dtype=object)).map(normalize_bay)
    rows = frame.get("YST_ROWNO", pd.Series(index=frame.index, dtype=object)).map(normalize_row)
    tiers = frame.get("YST_TIERNO", pd.Series(index=frame.index, dtype=object)).map(normalize_row)
    slots = frame.get("YST_SLOTNO", pd.Series(index=frame.index, dtype=object)).map(normalize_row)
    return list(zip(areas, bays, rows, tiers, slots))


def large_bay_partner_lookup_by_bay(frame: pd.DataFrame) -> dict[tuple[str, str], str]:
    if frame.empty:
        return {}
    bay_numbers = (
        frame[["YAA_AREANO", "YBY_BAYNO"]]
        .drop_duplicates()
        .assign(
            YAA_AREANO=lambda data: data["YAA_AREANO"].map(normalize_code),
            YBY_BAYNO=lambda data: data["YBY_BAYNO"].map(normalize_bay),
        )
        .drop_duplicates()
    )
    out: dict[tuple[str, str], str] = {}
    for area_no, area_frame in bay_numbers.groupby("YAA_AREANO", sort=False):
        ordered = sorted((str(value) for value in area_frame["YBY_BAYNO"] if str(value)), key=bay_sort_key)
        bay_set = set(ordered)
        for bay_no in ordered:
            try:
                next_bay = str(int(bay_no) + 2).zfill(max(2, len(bay_no)))
                prev_bay = str(int(bay_no) - 2).zfill(max(2, len(bay_no)))
            except ValueError:
                continue
            if next_bay in bay_set and are_consecutive_small_bays(bay_no, next_bay):
                out[(str(area_no), bay_no)] = next_bay
            elif prev_bay in bay_set and are_consecutive_small_bays(prev_bay, bay_no):
                out[(str(area_no), bay_no)] = prev_bay
    return out


def infer_large_pair_for_slot(row: Mapping[str, Any], bay_set_by_area: Mapping[str, set[str]]) -> tuple[str, str, str] | None:
    area_no = normalize_code(row.get("YAA_AREANO"))
    bay_no = normalize_bay(row.get("YBY_BAYNO"))
    if not area_no or not bay_no:
        return None
    bay_set = bay_set_by_area.get(area_no, set())
    try:
        next_bay = str(int(bay_no) + 2).zfill(max(2, len(bay_no)))
        prev_bay = str(int(bay_no) - 2).zfill(max(2, len(bay_no)))
    except ValueError:
        return None
    _enable20, enable40 = parse_enable_size_flags(row.get("YBY_ENABLECSIZECD"))
    if enable40 and next_bay in bay_set and are_consecutive_small_bays(bay_no, next_bay):
        return area_no, bay_no, next_bay
    if prev_bay in bay_set and are_consecutive_small_bays(prev_bay, bay_no):
        return area_no, prev_bay, bay_no
    if next_bay in bay_set and are_consecutive_small_bays(bay_no, next_bay):
        return area_no, bay_no, next_bay
    return None


def consecutive_large_pairs_from_bays(area_no: str, bay_nos: set[str]) -> list[tuple[str, str, str]]:
    ordered = sorted((bay_no for bay_no in bay_nos if bay_no), key=bay_sort_key)
    pairs: list[tuple[str, str, str]] = []
    for left, right in zip(ordered, ordered[1:]):
        if are_consecutive_small_bays(left, right):
            pairs.append((area_no, left, right))
    return pairs


def existing_large_pair_members_by_bay(
    frame: pd.DataFrame,
    vessel_schedules: dict[str, VoyageSchedule],
    planning_time: datetime,
) -> dict[tuple[str, str], set[frozenset[str]]]:
    bay_set_by_area: defaultdict[str, set[str]] = defaultdict(set)
    if frame.empty:
        return {}
    for row in frame[["YAA_AREANO", "YBY_BAYNO"]].drop_duplicates().to_dict("records"):
        area_no = normalize_code(row.get("YAA_AREANO"))
        bay_no = normalize_bay(row.get("YBY_BAYNO"))
        if area_no and bay_no:
            bay_set_by_area[area_no].add(bay_no)
    occupied = active_occupied(frame, vessel_schedules, planning_time)
    out: defaultdict[tuple[str, str], set[frozenset[str]]] = defaultdict(set)
    if occupied.empty:
        return {}
    occupied = occupied.copy()
    occupied["_area_no"] = occupied.get("YAA_AREANO", pd.Series(index=occupied.index, dtype=object)).map(normalize_code)
    occupied["_bay_no"] = occupied.get("YBY_BAYNO", pd.Series(index=occupied.index, dtype=object)).map(normalize_bay)
    occupied["_size"] = occupied.get("IYC_CSZ_CSIZECD", pd.Series(index=occupied.index, dtype=object)).map(normalize_size_small)
    occupied["_cntr_id"] = occupied.get("IYC_CNTRID", pd.Series(index=occupied.index, dtype=object)).map(normalize_code)
    large_occupied = occupied[occupied["_size"].isin({"40", "45"})].copy()
    resolved_indices: set[int] = set()
    valid_container_rows = large_occupied[
        large_occupied["_cntr_id"].notna()
        & large_occupied["_cntr_id"].astype(str).ne("")
        & large_occupied["_cntr_id"].astype(str).ne("-1")
    ]
    for (_area_no, _cntr_id), group in valid_container_rows.groupby(["_area_no", "_cntr_id"], sort=False):
        area_no = normalize_code(_area_no)
        bay_nos = {normalize_bay(value) for value in group["_bay_no"] if normalize_bay(value)}
        actual_pairs = consecutive_large_pairs_from_bays(area_no, bay_nos)
        if not actual_pairs:
            continue
        resolved_indices.update(int(idx) for idx in group.index)
        for pair in actual_pairs:
            _area_no, left, right = pair
            pair_members = frozenset((left, right))
            out[(_area_no, left)].add(pair_members)
            out[(_area_no, right)].add(pair_members)
    unresolved_large = large_occupied.loc[[idx not in resolved_indices for idx in large_occupied.index]]
    for row in unresolved_large.to_dict("records"):
        size = normalize_size_small(row.get("IYC_CSZ_CSIZECD"))
        if size not in {"40", "45"}:
            continue
        pair = infer_large_pair_for_slot(row, bay_set_by_area)
        if pair is None:
            continue
        area_no, left, right = pair
        pair_members = frozenset((left, right))
        out[(area_no, left)].add(pair_members)
        out[(area_no, right)].add(pair_members)
    return dict(out)


def large_pair_conflicts_existing(
    area_no: str,
    left: str,
    right: str,
    existing_large_pairs_by_member: Mapping[tuple[str, str], set[frozenset[str]]] | None,
) -> bool:
    if not existing_large_pairs_by_member:
        return False
    candidate = frozenset((left, right))
    for bay_no in (left, right):
        for existing_pair in existing_large_pairs_by_member.get((area_no, bay_no), set()):
            if existing_pair != candidate:
                return True
    return False


def active_large_container_shadow_slots(
    frame: pd.DataFrame,
    vessel_schedules: dict[str, VoyageSchedule],
    planning_time: datetime,
    large_bay_partner_by_bay: Mapping[tuple[str, str], str],
    existing_large_pairs_by_member: Mapping[tuple[str, str], set[frozenset[str]]] | None = None,
) -> set[tuple[str, str, str, str, str]]:
    if not large_bay_partner_by_bay and not existing_large_pairs_by_member:
        return set()
    occupied = active_occupied(frame, vessel_schedules, planning_time)
    if occupied.empty:
        return set()
    existing_large_pairs_by_member = existing_large_pairs_by_member or {}
    out: set[tuple[str, str, str, str, str]] = set()
    for row in occupied.to_dict("records"):
        size = normalize_size_small(row.get("IYC_CSZ_CSIZECD"))
        if size not in {"40", "45"}:
            continue
        area_no = normalize_code(row.get("YAA_AREANO"))
        bay_no = normalize_bay(row.get("YBY_BAYNO"))
        pair_sets = existing_large_pairs_by_member.get((area_no, bay_no), set())
        if pair_sets:
            for pair in pair_sets:
                for partner_bay in pair:
                    if partner_bay != bay_no:
                        out.add(slot_identity(row, area_no=area_no, bay_no=partner_bay))
            continue
        partner_bay = large_bay_partner_by_bay.get((area_no, bay_no))
        if partner_bay:
            out.add(slot_identity(row, area_no=area_no, bay_no=partner_bay))
    return out


def drop_shadow_slots(
    frame: pd.DataFrame,
    shadow_slots: set[tuple[str, str, str, str, str]],
) -> pd.DataFrame:
    if frame.empty or not shadow_slots:
        return frame
    keep = [key not in shadow_slots for key in slot_identities(frame)]
    return frame.loc[keep].copy()


def capacity_by_bay_size(base: pd.DataFrame) -> dict[str, dict[tuple[str, str], int]]:
    out: dict[str, dict[tuple[str, str], int]] = {}
    for size_mode in SIZE_MODES:
        sub = base[size_enabled_mask(base["YBY_ENABLECSIZECD"], size_mode)].copy()
        counts = sub.groupby(["YAA_AREANO", "YBY_BAYNO"]).size() if not sub.empty else pd.Series(dtype=int)
        out[size_mode] = {(str(a), str(b)): int(v) for (a, b), v in counts.items()}
    return out


def physical_capacity_by_bay(base: pd.DataFrame) -> dict[tuple[str, str], int]:
    counts = base.groupby(["YAA_AREANO", "YBY_BAYNO"]).size() if not base.empty else pd.Series(dtype=int)
    return {(str(a), str(b)): int(v) for (a, b), v in counts.items()}


def apply_large_bay_pair_capacities(
    area_no: str,
    ordered_bays: list[str],
    cap_by_size: dict[str, dict[tuple[str, str], int]],
    physical_cap: dict[tuple[str, str], int],
    existing_large_pairs_by_member: Mapping[tuple[str, str], set[frozenset[str]]] | None = None,
) -> dict[tuple[str, str], str]:
    partner_by_start: dict[tuple[str, str], str] = {}
    original = {
        size_mode: {
            bay_no: int(cap_by_size[size_mode].get((area_no, bay_no), 0))
            for bay_no in ordered_bays
        }
        for size_mode in ("40", "45")
    }
    for size_mode in ("40", "45"):
        for bay_no in ordered_bays:
            cap_by_size[size_mode][(area_no, bay_no)] = 0
    idx = 0
    while idx < len(ordered_bays) - 1:
        left = ordered_bays[idx]
        right = ordered_bays[idx + 1]
        if not are_consecutive_small_bays(left, right):
            idx += 1
            continue
        if large_pair_conflicts_existing(area_no, left, right, existing_large_pairs_by_member):
            idx += 1
            continue
        left_key = (area_no, left)
        right_key = (area_no, right)
        pair_physical = min(int(physical_cap.get(left_key, 0)), int(physical_cap.get(right_key, 0)))
        if pair_physical <= 0:
            idx += 1
            continue
        has_large_capacity = False
        for size_mode in ("40", "45"):
            pair_cap = min(original[size_mode].get(left, 0), original[size_mode].get(right, 0), pair_physical)
            cap_by_size[size_mode][left_key] = pair_cap
            has_large_capacity = has_large_capacity or pair_cap > 0
        if has_large_capacity:
            partner_by_start[left_key] = right
        idx += 2
    return partner_by_start


def are_consecutive_small_bays(left: str, right: str) -> bool:
    try:
        return int(left) + 2 == int(right)
    except ValueError:
        return False


def capacity_by_bay_row_size(base: pd.DataFrame) -> dict[str, dict[tuple[str, str, str], int]]:
    out: dict[str, dict[tuple[str, str, str], int]] = {}
    for size_mode in SIZE_MODES:
        sub = base[size_enabled_mask(base["YBY_ENABLECSIZECD"], size_mode)].copy()
        counts = sub.groupby(["YAA_AREANO", "YBY_BAYNO", "YST_ROWNO"]).size() if not sub.empty else pd.Series(dtype=int)
        out[size_mode] = {(str(a), str(b), str(r)): int(v) for (a, b, r), v in counts.items()}
    return out


def physical_capacity_by_bay_row(base: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    counts = base.groupby(["YAA_AREANO", "YBY_BAYNO", "YST_ROWNO"]).size() if not base.empty else pd.Series(dtype=int)
    return {(str(a), str(b), str(r)): int(v) for (a, b, r), v in counts.items()}


def total_slots_by_bay(base: pd.DataFrame) -> Counter[tuple[str, str]]:
    slots: Counter[tuple[str, str]] = Counter()
    if base.empty:
        return slots
    counts = base.groupby(["YAA_AREANO", "YBY_BAYNO"], sort=False).size()
    slots.update({(str(a), str(b)): int(v) for (a, b), v in counts.items()})
    return slots


def misplaced_bays_to_exclude(
    base: pd.DataFrame,
    area_functions: dict[str, set[str]],
    vessel_schedules: dict[str, VoyageSchedule],
    planning_time: datetime,
    ratio: float,
    original_bay_capacity: Counter[tuple[str, str]],
    target_voyages: set[str],
) -> set[tuple[str, str]]:
    if ratio <= 0:
        return set()
    occupied = active_occupied(base, vessel_schedules, planning_time)
    if occupied.empty:
        return set()
    target_voyages = {normalize_voyage(v) for v in target_voyages}
    occupied = occupied[occupied["IYC_EVOY_ID"].map(normalize_voyage).isin(target_voyages)].copy()
    if occupied.empty:
        return set()
    occupied["_flow"] = occupied["IYC_STS_CSTATUSCD"].map(lambda value: normalize_flow(value, default="OF"))
    bad = occupied[
        [
            not area_allows_flow(area, flow, area_functions)
            for area, flow in zip(occupied["YAA_AREANO"], occupied["_flow"])
        ]
    ].copy()
    if bad.empty:
        return set()
    counts = bad.groupby(["YAA_AREANO", "YBY_BAYNO"]).size()
    return {
        (str(area), str(bay))
        for (area, bay), count in counts.items()
        if int(count) > original_bay_capacity.get((str(area), str(bay)), 0) * ratio
    }


def existing_bay_attributes(
    frame: pd.DataFrame,
    vessel_schedules: dict[str, VoyageSchedule],
    planning_time: datetime,
    attribute_rules: AttributeRules | None = None,
    large_bay_partner_by_bay: Mapping[tuple[str, str], str] | None = None,
    existing_large_pairs_by_member: Mapping[tuple[str, str], set[frozenset[str]]] | None = None,
) -> dict[tuple[str, str], dict[str, set[str]]]:
    occupied = active_occupied(frame, vessel_schedules, planning_time)
    out: dict[tuple[str, str], dict[str, set[str]]] = {}
    dynamic_attrs: list[str] = []
    if attribute_rules is not None:
        for attrs in (
            attribute_rules.bay_no_mix_attributes,
            attribute_rules.row_no_mix_attributes,
            *(attribute_rules.bay_no_mix_attributes_by_voyage.values()),
            *(attribute_rules.row_no_mix_attributes_by_voyage.values()),
        ):
            for attr in attrs:
                name = attribute_output_name(attr)
                if name and name not in dynamic_attrs:
                    dynamic_attrs.append(name)
    large_bay_partner_by_bay = large_bay_partner_by_bay or {}
    existing_large_pairs_by_member = existing_large_pairs_by_member or {}
    for row in occupied.to_dict("records"):
        area_no = normalize_code(row.get("YAA_AREANO"))
        bay_no = normalize_bay(row.get("YBY_BAYNO"))
        size = normalize_size_small(row.get("IYC_CSZ_CSIZECD"))
        row_voyages = {
            voyage
            for voyage in (
                normalize_voyage(row.get("IYC_EVOY_ID")),
                normalize_voyage(row.get("IYC_IVOY_ID")),
            )
            if voyage
        }
        target_keys = [(area_no, bay_no)]
        if size in {"40", "45"}:
            pair_sets = existing_large_pairs_by_member.get((area_no, bay_no), set())
            if pair_sets:
                for pair in pair_sets:
                    for partner_bay in pair:
                        if partner_bay != bay_no:
                            target_keys.append((area_no, partner_bay))
            else:
                partner_bay = large_bay_partner_by_bay.get((area_no, bay_no))
                if partner_bay:
                    target_keys.append((area_no, partner_bay))
        for key in dict.fromkeys(target_keys):
            attrs = out.setdefault(
                key,
                {
                    "sizes": set(),
                    "heights": set(),
                    "specials": set(),
                    "ports": set(),
                    "attributes": {},
                    "attributes_by_row": {},
                    "attributes_by_voyage": {},
                    "attributes_by_row_by_voyage": {},
                },
            )
            attrs["sizes"].add(size)
            attrs["heights"].add(normalize_text(row.get("IYC_CHEIGHTCD"), "UNK"))
            special = special_stow_code(row) or "NORMAL"
            attrs["specials"].add(special)
            port = normalize_text(row.get("IYC_POT_UNLDPORT"))
            if port:
                attrs["ports"].add(port)
            row_no = normalize_row(row.get("YST_ROWNO"))
            row_attrs = attrs["attributes_by_row"].setdefault(row_no, {}) if row_no else {}
            export_voyage = normalize_voyage(row.get("IYC_EVOY_ID"))
            if export_voyage and row_no:
                row_attrs.setdefault(EXPORT_VOYAGE_ROW_NO_MIX_ATTR, set()).add(export_voyage)
            for attr in dynamic_attrs:
                value = dynamic_attribute_value(row, attr)
                if value:
                    if is_size_no_mix_attribute(attr):
                        attrs["attributes"].setdefault(attr, set()).add(value)
                        if row_attrs is not None:
                            row_attrs.setdefault(attr, set()).add(value)
                    else:
                        for voyage in row_voyages:
                            attrs["attributes_by_voyage"].setdefault(voyage, {}).setdefault(attr, set()).add(value)
                            if row_no:
                                attrs["attributes_by_row_by_voyage"].setdefault(row_no, {}).setdefault(
                                    voyage, {}
                                ).setdefault(attr, set()).add(value)
    return out


def make_yard_blocks(ordered_bays: list[str], big_bay_starts: set[str], target_bay_count: int = 6) -> list[tuple[int, tuple[str, ...], bool]]:
    blocks: list[tuple[int, tuple[str, ...], bool]] = []
    start = 0
    block_index = 1
    while start < len(ordered_bays):
        if len(ordered_bays) - start <= target_bay_count:
            end = len(ordered_bays)
            adjusted = False
        else:
            target_end = start + target_bay_count
            end = nearest_safe_block_end(ordered_bays, big_bay_starts, start, target_end)
            adjusted = end != target_end
        blocks.append((block_index, tuple(ordered_bays[start:end]), adjusted))
        block_index += 1
        start = end
    return blocks


def nearest_safe_block_end(ordered_bays: list[str], big_bay_starts: set[str], start: int, target_end: int) -> int:
    min_end = start + 1
    max_end = len(ordered_bays)
    for offset in range(0, len(ordered_bays) + 1):
        probes = [target_end] if offset == 0 else [target_end + offset, target_end - offset]
        for end in probes:
            if not (min_end <= end <= max_end):
                continue
            if end == len(ordered_bays) or ordered_bays[end - 1] not in big_bay_starts:
                return end
    return len(ordered_bays)


def build_area_operations(input_guandong: InputAdapterGd, vessel_schedules: dict[str, VoyageSchedule]) -> dict[str, list[AreaOperation]]:
    operations: defaultdict[str, list[AreaOperation]] = defaultdict(list)
    tops = active_tops_rows(input_guandong, datetime.max.replace(year=2099))
    if tops.empty:
        return dict(operations)
    for row in tops.to_dict("records"):
        start_area, _ = parse_tops_area_bay(row.get("SPR_STBAY"))
        end_area, _ = parse_tops_area_bay(row.get("SPR_EDBAY"))
        area = start_area or end_area
        voyage = normalize_voyage(row.get("condition_vessel"))
        start_time = row.get("start_time")
        end_time = row.get("end_time")
        if area and voyage and start_time and end_time:
            operations[area].append(AreaOperation(area, voyage, start_time, end_time))
    return dict(operations)


def build_problem(
    input_guandong: InputAdapterGd,
    big_plan: list[BigPlanRow],
    planning_time: datetime,
    horizon_hours: float,
    target_voyages: list[str],
    misplaced_bay_exclusion_ratio: float,
    demand_rows: list[DemandRow] | None = None,
) -> ProblemData:
    closed = read_closed_areas(input_guandong)
    area_functions = read_area_functions(input_guandong)
    function_areas = set(area_functions)
    area_quota: dict[tuple[str, str, str], int] = {}
    area_size_quota: dict[tuple[str, str, str, str], int] = {}
    assigned_areas: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    cleaned_plan: list[BigPlanRow] = []
    target_voyages = [normalize_voyage(v) for v in target_voyages]
    attribute_rules = read_attribute_rules(input_guandong, target_voyages)
    (
        large_allowed_areas_by_voyage,
        _required_areas_by_voyage,
        _priority_areas_by_voyage,
        _area_control_diagnostics,
    ) = build_large_area_controls(
        vessels=target_voyages,
        areas=sorted(function_areas),
        user_design=getattr(input_guandong, "user_design", False),
        user_design_large_plan_area=getattr(input_guandong, "user_design_large_plan_area", []),
        voyage_limit_areas=getattr(input_guandong, "voyage_limit_areas", {}),
        voyage_priority_areas=getattr(input_guandong, "voyage_priority_areas", {}),
        adjust_plan_info=getattr(input_guandong, "adjust_plan_info", {}),
    )
    strict_allowed_areas_by_voyage, priority_areas_by_voyage, user_area_constraint_summary = build_medium_small_area_controls(
        vessels=target_voyages,
        areas=sorted(function_areas),
        user_design_large_plan_area=getattr(input_guandong, "user_design_large_plan_area", {}),
        voyage_limit_areas=getattr(input_guandong, "voyage_limit_areas", {}),
        voyage_priority_areas=getattr(input_guandong, "voyage_priority_areas", {}),
    )
    user_area_constraint_summary["large_area_controls"] = _area_control_diagnostics
    allowed_areas_by_voyage: dict[str, set[str]] = {}
    for voyage_id in target_voyages:
        strict_entry = user_area_constraint_summary["area_controls_by_voyage"].get(voyage_id, {})
        if strict_entry.get("strict_boundary"):
            allowed_areas_by_voyage[voyage_id] = set(strict_allowed_areas_by_voyage.get(voyage_id, set()))
        else:
            allowed_areas_by_voyage[voyage_id] = set(
                large_allowed_areas_by_voyage.get(voyage_id, sorted(function_areas))
            )
    vessel_schedules = read_target_vessel_schedules(input_guandong, target_voyages, planning_time, horizon_hours)
    plan_date = planning_time.date().isoformat()
    target_big_plan_flows = {medium_small_area_flow(flow) for flow in DEFAULT_TARGET_BIG_PLAN_FLOWS}
    input_plan = [
        row for row in big_plan if row.voyage_id in target_voyages and (not row.plan_date or row.plan_date == plan_date)
    ]
    allowed_areas = set().union(*(set(areas) for areas in allowed_areas_by_voyage.values())) if allowed_areas_by_voyage else set(function_areas)
    skipped_outside_user_scope: Counter[tuple[str, str]] = Counter()
    skipped_closed_area: Counter[tuple[str, str]] = Counter()
    skipped_flow_function: Counter[tuple[str, str]] = Counter()
    for row in input_plan:
        if row.area_no not in allowed_areas_by_voyage.get(row.voyage_id, set(function_areas)):
            skipped_outside_user_scope[(row.voyage_id, row.area_no)] += row.planned_boxes
            continue
        if row.area_no in closed:
            skipped_closed_area[(row.voyage_id, row.area_no)] += row.planned_boxes
            continue
        plan_flow = medium_small_area_flow(row.flow)
        if plan_flow not in target_big_plan_flows:
            continue
        if not area_allows_flow(row.area_no, plan_flow, area_functions):
            skipped_flow_function[(row.voyage_id, row.area_no)] += row.planned_boxes
            continue
        cleaned_plan.append(row)
        assigned_areas[(row.voyage_id, plan_flow)].add(row.area_no)
    # Medium/small demand uses actual demand; big-plan rows below remain area inheritance targets.
    groups, _demand_rows = load_port_demand_groups(
        input_guandong,
        target_voyages,
        planning_time,
        attribute_rules,
        big_plan_caps=None,
        horizon_hours=horizon_hours,
        demand_rows=demand_rows,
    )
    small_groups = load_small_doc_groups(
        input_guandong,
        target_voyages,
        attribute_rules,
        planning_time,
        big_plan_caps=None,
    )
    demand_by_voyage_size: Counter[tuple[str, str, str]] = Counter()
    for group in groups:
        demand_by_voyage_size[(group.voyage_id, group.status, group.big_plan_size_mode)] += group.demand
    raw_area_quota: Counter[tuple[str, str, str]] = Counter()
    raw_area_size_quota: Counter[tuple[str, str, str, str]] = Counter()
    raw_all_size_area_quota: Counter[tuple[str, str, str]] = Counter()
    missing_big_plan_area_pattern: Counter[tuple[str, str, str]] = Counter()
    for row in cleaned_plan:
        plan_flow = medium_small_area_flow(row.flow)
        raw_area_quota[(row.voyage_id, plan_flow, row.area_no)] += row.planned_boxes
        if row.size_mode == "ALL":
            raw_all_size_area_quota[(row.voyage_id, plan_flow, row.area_no)] += row.planned_boxes
        else:
            raw_area_size_quota[(row.voyage_id, plan_flow, row.area_no, row.size_mode)] += row.planned_boxes
    for voyage_id in target_voyages:
        flows = sorted({flow for (v, flow, _size), qty in demand_by_voyage_size.items() if v == voyage_id and qty > 0})
        for flow in flows:
            source_flow = medium_small_area_flow(flow)
            compatible_plan_flows = {source_flow}
            for size_mode in SIZE_MODES:
                target_qty = demand_by_voyage_size[(voyage_id, flow, size_mode)]
                if target_qty <= 0:
                    continue
                exact_upper = Counter(
                    {
                        area_no: qty
                        for (v, f, area_no, size), qty in raw_area_size_quota.items()
                        if v == voyage_id and f in compatible_plan_flows and size == size_mode and qty > 0
                    }
                )
                if exact_upper:
                    # if sum(exact_upper.values()) < target_qty:
                    #     raise ValueError(
                    #         f"medium demand exceeds big-plan strict upper bound for "
                    #         f"voyage={voyage_id}, flow={flow}, size={size_mode}: "
                    #         f"demand={target_qty}, big_plan_upper={sum(exact_upper.values())}"
                    #     )
                    for area_no, qty in exact_upper.items():
                        area_quota[(voyage_id, flow, area_no)] = raw_area_quota[(voyage_id, source_flow, area_no)]
                        area_size_quota[(voyage_id, flow, area_no, size_mode)] = qty
                        assigned_areas[(voyage_id, flow)].add(area_no)
                    continue
                all_size_weights = Counter(
                    {
                        area_no: qty
                        for (v, f, area_no), qty in raw_all_size_area_quota.items()
                        if v == voyage_id and f in compatible_plan_flows and qty > 0
                    }
                )
                if not all_size_weights:
                    missing_big_plan_area_pattern[(voyage_id, flow, size_mode)] += target_qty
                    continue
                big_plan_total = sum(all_size_weights.values())
                allocations = Counter(all_size_weights) if big_plan_total <= target_qty else allocate_by_weights(dict(all_size_weights), target_qty)
                for area_no, qty in allocations.items():
                    if qty <= 0:
                        continue
                    area_quota[(voyage_id, flow, area_no)] = raw_area_quota[(voyage_id, source_flow, area_no)]
                    area_size_quota[(voyage_id, flow, area_no, size_mode)] = qty
                    assigned_areas[(voyage_id, flow)].add(area_no)
    voyage_windows = {
        voyage_id: (
            vessel_schedules[voyage_id].receive_start if voyage_id in vessel_schedules else planning_time,
            (vessel_schedules[voyage_id].receive_start if voyage_id in vessel_schedules else planning_time)
            + timedelta(hours=horizon_hours),
        )
        for voyage_id in target_voyages
    }
    bays, reserved_count, closed_bay_count, misplaced_count = build_bays(
        input_guandong,
        allowed_areas,
        closed,
        planning_time,
        vessel_schedules,
        set(target_voyages),
        area_functions,
        misplaced_bay_exclusion_ratio,
        attribute_rules,
    )
    existing_coarse_area_load, existing_coarse_bay_load = existing_coarse_group_loads(
        input_guandong,
        planning_time,
        set(target_voyages),
        set(bays),
        attribute_rules,
    )
    bay_requirements, bay_blocklist, bay_adjust_rules, bay_constraint_summary = build_medium_small_bay_controls(
        input_guandong,
        groups,
        small_groups,
        bays,
    )
    voyage_bay_allowlist, voyage_bay_summary = build_user_design_area_bay_allowlist(input_guandong, bays)
    bay_constraint_summary["user_design_area_bay"] = voyage_bay_summary
    user_area_constraint_summary.update(
        {
            "allowed_areas_by_voyage": {
                voyage_id: sorted(areas)
                for voyage_id, areas in sorted(allowed_areas_by_voyage.items())
            },
            "effective_yard_areas": sorted(allowed_areas),
            "input_big_plan_row_count": len(input_plan),
            "accepted_big_plan_row_count": len(cleaned_plan),
            "skipped_big_plan_boxes_outside_user_scope": {
                f"{voyage_id}|{area_no}": int(qty)
                for (voyage_id, area_no), qty in sorted(skipped_outside_user_scope.items())
                if qty > 0
            },
            "skipped_big_plan_boxes_closed_area": {
                f"{voyage_id}|{area_no}": int(qty)
                for (voyage_id, area_no), qty in sorted(skipped_closed_area.items())
                if qty > 0
            },
            "skipped_big_plan_boxes_flow_function": {
                f"{voyage_id}|{area_no}": int(qty)
                for (voyage_id, area_no), qty in sorted(skipped_flow_function.items())
                if qty > 0
            },
            "missing_big_plan_area_pattern_boxes": {
                f"{voyage_id}|{flow}|{size_mode}": int(qty)
                for (voyage_id, flow, size_mode), qty in sorted(missing_big_plan_area_pattern.items())
                if qty > 0
            },
        }
    )
    area_operations = build_area_operations(input_guandong, vessel_schedules)
    berth_distances = read_distance_matrix(input_guandong)
    berth_by_voyage = {
        voyage_id: f"B{vessel_schedules[voyage_id].berth_no}"
        for voyage_id in target_voyages
        if voyage_id in vessel_schedules and vessel_schedules[voyage_id].berth_no
    }
    target_voyage_set = set(target_voyages)
    export_voyages = classified_export_voyages(input_guandong) & target_voyage_set
    return ProblemData(
        groups=groups,
        small_groups=small_groups,
        bays=bays,
        big_plan=cleaned_plan,
        assigned_areas=dict(assigned_areas),
        area_quota=area_quota,
        area_size_quota=area_size_quota,
        area_functions=area_functions,
        business_special_codes=collect_business_special_codes(groups),
        planning_time=planning_time,
        horizon_hours=horizon_hours,
        voyage_windows=voyage_windows,
        area_operations=area_operations,
        target_voyages=target_voyages,
        export_voyages=export_voyages,
        existing_coarse_area_load=dict(existing_coarse_area_load),
        existing_coarse_bay_load=dict(existing_coarse_bay_load),
        berth_distances=berth_distances,
        berth_by_voyage=berth_by_voyage,
        allowed_areas_by_voyage={
            voyage_id: set(allowed_areas_by_voyage.get(voyage_id, set(function_areas)))
            for voyage_id in target_voyages
        },
        user_voyage_area_allowlist={
            voyage_id: set(allowed_areas_by_voyage.get(voyage_id, set(function_areas)))
            for voyage_id in target_voyages
        },
        user_voyage_area_blocklist={
            voyage_id: set(function_areas) - set(allowed_areas_by_voyage.get(voyage_id, set(function_areas)))
            for voyage_id in target_voyages
        },
        user_voyage_area_priority={
            voyage_id: set(priority_areas_by_voyage.get(voyage_id, set()))
            for voyage_id in target_voyages
        },
        user_voyage_area_requirements={
            voyage_id: set(_required_areas_by_voyage.get(voyage_id, []))
            for voyage_id in target_voyages
        },
        user_voyage_bay_allowlist=voyage_bay_allowlist,
        user_group_bay_requirements=bay_requirements,
        user_group_bay_blocklist=bay_blocklist,
        user_bay_adjust_rules=bay_adjust_rules,
        user_area_constraint_summary=user_area_constraint_summary,
        user_bay_constraint_summary=bay_constraint_summary,
        tops_reserved_slot_count=reserved_count,
        tops_closed_bay_count=closed_bay_count,
        misplaced_bay_exclusion_ratio=misplaced_bay_exclusion_ratio,
        misplaced_excluded_bay_count=misplaced_count,
        attribute_rules=attribute_rules,
    )


def load_medium_small_inputs(
    input_guandong: InputAdapterGd,
    planning_time: datetime,
    voyages: Sequence[str],
    horizon_hours: float,
    misplaced_bay_exclusion_ratio: float,
    big_plan: pd.DataFrame | Sequence[BigPlanRow] | None = None,
) -> MediumSmallInputs:
    if big_plan is None:
        big_plan_rows = read_big_plan(input_guandong.large_plan)
    elif isinstance(big_plan, pd.DataFrame):
        big_plan_rows = read_big_plan(big_plan)
    else:
        big_plan_rows = list(big_plan)
    demand_rows = calculate_medium_demands(
        input_guandong,
        list(voyages),
        planning_time,
        big_plan_caps=None,
        horizon_hours=horizon_hours,
    )
    problem = build_problem(
        input_guandong,
        big_plan_rows,
        planning_time=planning_time,
        horizon_hours=horizon_hours,
        target_voyages=list(voyages),
        misplaced_bay_exclusion_ratio=misplaced_bay_exclusion_ratio,
        demand_rows=demand_rows,
    )
    return MediumSmallInputs(big_plan=big_plan_rows, demand_rows=demand_rows, problem=problem)


def allocate_by_weights(weights: dict[str, int], target_total: int) -> dict[str, int]:
    items = [(key, value) for key, value in sorted(weights.items()) if value > 0]
    if not items or target_total <= 0:
        return {}
    source_total = sum(value for _, value in items)
    raw = [value * target_total / source_total for _, value in items]
    base = [int(value) for value in raw]
    remain = target_total - sum(base)
    order = sorted(range(len(raw)), key=lambda idx: raw[idx] - base[idx], reverse=True)
    for idx in order[:remain]:
        base[idx] += 1
    return {key: qty for (key, _), qty in zip(items, base) if qty > 0}


def weight_class(value: object, levels: Sequence[int] = DEFAULT_WEIGHT_LEVEL) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "UNK"
    tons = float(numeric) / 1000.0
    ordered = sorted({int(level) for level in levels})
    if not ordered:
        ordered = list(DEFAULT_WEIGHT_LEVEL)
    for index, (lo, hi) in enumerate(zip(ordered, ordered[1:]), start=1):
        if lo <= tons < hi:
            return str(index)
    if tons >= ordered[-1]:
        return str(len(ordered))
    return "UNK"


def business_special_codes(row: Mapping[str, Any]) -> set[str]:
    codes = set()
    ctype = normalize_code(row.get("IYC_CTYPECD"))
    if ctype in {"RF", "DG", "OT", "TK"}:
        codes.add(ctype)
    if pd.notna(row.get("IYC_SETTMPT")):
        codes.add("REEFER")
    if pd.notna(row.get("IYC_DTP_DNGGCD")):
        codes.add("DANGER")
    if pd.notna(row.get("IYC_OVLMTCD")):
        codes.add("OVER")
    return codes


def special_stow_code(row: Mapping[str, Any]) -> str:
    return "+".join(sorted(business_special_codes(row)))


def explicit_special_stow_code(row: Mapping[str, Any]) -> str:
    return ""


def collect_business_special_codes(groups: list[BoxGroup]) -> set[str]:
    out: set[str] = set()
    for group in groups:
        out.update(group.special_codes)
    return out


def bay_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        nums = re.findall(r"\d+", value)
        return (int(nums[0]) if nums else 9999), value


def date_key(value: str) -> str:
    if not value:
        return ""
    parsed = parse_datetime(value)
    return parsed.date().isoformat() if parsed else value


def _first_existing(fieldnames: set[Any], candidates: list[str]) -> str | None:
    string_names = {str(field): field for field in fieldnames}
    for candidate in candidates:
        if candidate in string_names:
            return string_names[candidate]
    return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "DEFAULT_EXPORT_VESSELS",
    "DEFAULT_IMPORT_VESSELS",
    "DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO",
    "DEFAULT_PLANNING_TIME",
    "DEFAULT_TARGET_VOYAGES",
    "LargePlanningConfig",
    "LargePlanningData",
    "MediumSmallInputs",
    "PlanningInputArtifacts",
    "RollingPlanningState",
    "YardPlanningWeights",
    "allocation_output_rows",
    "build_large_inputs",
    "load_medium_small_inputs",
    "parse_datetime",
    "parse_planning_time",
    "write_demand_rows",
    "write_json",
    "write_large_outputs",
]
