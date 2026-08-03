from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Mapping, Sequence

import pandas as pd

from .input_adapter_gd import InputAdapterGd

from yard_planning.models import (
    AttributeRules, Bay, BigPlanRow, DeclaredExportDemand, EXPORT_VOYAGE_ROW_NO_MIX_ATTR,
    ExportGroup, PlanningInputs, ProblemData,
)


DEFAULT_TARGET_BIG_PLAN_FLOWS = frozenset({"OF", "IF", "IZ", "T", "OZ"})
SIZE_MODES = ("20", "40", "45")


def date_key(value: str) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


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


def normalize_container_size(value: Any) -> str:
    code = normalize_code(value)
    return code if code in {"20", "40", "45"} else "40"


def normalize_capacity_size(value: Any) -> str:
    code = normalize_code(value)
    if code.startswith("20"):
        return "20"
    if code.startswith(("40", "45")):
        return "40"
    return ""


def normalize_big_plan_size(value: Any) -> str:
    code = normalize_code(value)
    if code in {"20", "40"}:
        return code
    raise ValueError(
        f"big-plan size must be 20 or 40, got {value!r}; "
        "ALL, 45, blank, and unknown values are not accepted"
    )


def normalize_flow(value: Any, aliases: Mapping[str, str] | None = None, default: str = "") -> str:
    flow = normalize_code(value)
    if not flow:
        return default
    return (aliases or {}).get(flow, flow)


def normalize_planning_flow(value: Any, default: str = "OF") -> str:
    return normalize_flow(value, default=default)


def planning_area_flow(flow: Any) -> str:
    normalized = normalize_planning_flow(flow, default="OF")
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


def yard_transshipment_mask(rows: pd.DataFrame) -> pd.Series:
    if rows.empty or "IYC_EVOY_ID" not in rows.columns or "IYC_IVOY_ID" not in rows.columns:
        return pd.Series(False, index=rows.index)
    export_voyage = rows["IYC_EVOY_ID"].map(lambda value: bool(normalize_voyage(value)))
    import_voyage = rows["IYC_IVOY_ID"].map(lambda value: bool(normalize_voyage(value)))
    return export_voyage & import_voyage


def active_yard_rows(rows: pd.DataFrame) -> pd.DataFrame:
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


def attribute_output_name(attr: object) -> str:
    return "" if attr is None else str(attr).strip()


def is_size_no_mix_attribute(attr: object) -> bool:
    return attribute_output_name(attr).upper() in {"IYC_CSZ_CSIZECD", "SIZE", "SIZE_MODE"}


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
    default: str = "MIXED",
) -> str:
    raw = attribute_output_name(attr)
    if not raw:
        return default
    return raw_attribute_text(row.get(raw), default)


def dynamic_attributes_from_row(
    row: Mapping[str, Any],
    attrs: Sequence[str],
    *,
    default: str = "MIXED",
) -> dict[str, str]:
    out: dict[str, str] = {}
    for attr in attrs:
        name = attribute_output_name(attr)
        if name:
            out[name] = dynamic_attribute_value(row, name, default=default)
    return out


def read_attribute_rules(input_guandong: InputAdapterGd, voyages: Sequence[str]) -> AttributeRules:
    return AttributeRules(
        group_attributes=("IYC_CSZ_CSIZECD", "IYC_POT_UNLDPORT", "IYC_CHEIGHTCD"),
        bay_no_mix_attributes=("IYC_CHEIGHTCD",),
        row_no_mix_attributes=("IYC_POT_UNLDPORT",),
    )


def export_groupby_columns(attribute_rules: AttributeRules, voyage_id: str) -> tuple[str, ...]:
    attrs: list[str] = []
    attrs.extend(attribute_rules.group_for(voyage_id))

    # Keep no-mix attributes on planning groups so compatibility can be
    # evaluated even when an attribute is not part of the group identity.
    attrs.extend(attribute_rules.bay_no_mix_for(voyage_id))
    attrs.extend(attribute_rules.row_no_mix_for(voyage_id))
    columns: list[str] = []
    for attr in attrs:
        column = attribute_output_name(attr)
        if column and column not in columns:
            columns.append(column)
    return tuple(columns)


def operational_group_attributes(attribute_rules: AttributeRules, voyage_id: str) -> tuple[str, ...]:
    attrs: list[str] = []
    attrs.extend(attribute_rules.group_for(voyage_id))
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


def normalized_doc_record(row: Mapping[str, Any], flow: str, size_mode: str, port: str) -> dict[str, Any]:
    record = dict(row)
    record["IYC_STS_CSTATUSCD"] = flow
    record["IYC_CSZ_CSIZECD"] = size_mode
    record["IYC_POT_UNLDPORT"] = port
    evoy = normalize_voyage(row.get("IYC_EVOY_ID"))
    if evoy:
        record["IYC_EVOY_ID"] = evoy
    return record


def _adapter_vessel_items(input_guandong: InputAdapterGd) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for raw_voyage, content in (input_guandong.vessel_containers or {}).items():
        voyage = normalize_voyage(raw_voyage)
        if not voyage or not isinstance(content, dict):
            continue
        items.append((voyage, content))
    return items


def _take_over_vessels(input_guandong: InputAdapterGd, direction: str) -> list[str]:
    take_over = getattr(input_guandong, "take_over_vessel", {}) or {}
    if not isinstance(take_over, Mapping):
        return []
    return normalize_voyage_list(take_over.get(direction, []))


def classified_export_voyages(input_guandong: InputAdapterGd) -> set[str]:
    """Return voyages explicitly classified as export, independent of box flow."""
    cache = planning_runtime_cache(input_guandong)
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


def bay_sort_key(value: Any) -> tuple[int, str]:
    code = normalize_bay(value)
    numeric = bay_code_value(code)
    return (numeric if numeric is not None else 10**9, code)


def normalize_row(value: Any) -> str:
    return normalize_bay(value)


def parse_enable_size_flags(value: Any) -> tuple[bool, bool]:
    if value is None or pd.isna(value):
        return True, True
    tokens = re.findall(r"\d+", str(value))
    if not tokens:
        return True, True
    sizes = {normalize_capacity_size(token) for token in tokens}
    return "20" in sizes, "40" in sizes


def size_enabled_mask(values: pd.Series, size_mode: str) -> pd.Series:
    text = values.astype("string")
    stripped = text.str.strip()
    missing = values.isna() | stripped.str.lower().isin(["", "nan", "none", "<na>"]).fillna(False)
    enabled = stripped.str.contains(rf"(?<!\d){re.escape(size_mode)}(?!\d)", regex=True, na=False)
    return missing | enabled


def planning_runtime_cache(input_guandong: InputAdapterGd) -> dict[str, Any]:
    cache = getattr(input_guandong, "_planning_runtime_cache", None)
    if isinstance(cache, dict):
        return cache
    cache = {}
    try:
        setattr(input_guandong, "_planning_runtime_cache", cache)
    except Exception:
        return {}
    return cache


def read_vessel_info(input_guandong: InputAdapterGd) -> pd.DataFrame:
    cache = planning_runtime_cache(input_guandong)
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
    cache["vessel_info"] = frame
    return frame.copy()


def read_export_berths(input_guandong: InputAdapterGd, target_voyages: Sequence[str]) -> dict[str, str]:
    frame = read_vessel_info(input_guandong)
    target_set = {normalize_voyage(value) for value in target_voyages}
    berths: dict[str, str] = {}
    for row in frame.to_dict("records"):
        if row.get("ie_flag") != "E":
            continue
        voyage_id = normalize_voyage(row.get("voy_id"))
        berth_no = normalize_code(row.get("berth_no"))
        if voyage_id in target_set and berth_no:
            berths[voyage_id] = f"B{berth_no}" if not berth_no.startswith("B") else berth_no
    return berths


def read_area_functions(input_guandong: InputAdapterGd) -> dict[str, set[str]]:
    frame = input_guandong.area_function_info
    area_col = _first_existing(set(frame.columns), ["area_no", "AREA_NO", "YAA_AREANO"])
    type_col = _first_existing(set(frame.columns), ["cntr_type", "CNTR_TYPE", "function", "FUNCTION"])
    if not area_col or not type_col:
        raise KeyError("Area function input must contain area_no and cntr_type columns.")
    functions: dict[str, set[str]] = {}
    for row in frame.to_dict("records"):
        area = normalize_code(row.get(area_col))
        if area:
            functions[area] = {
                normalize_code(part)
                for part in str(row.get(type_col, "")).split(",")
                if normalize_code(part)
            }
    return functions


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


def active_tops_rows(input_guandong: InputAdapterGd, planning_time: datetime) -> pd.DataFrame:
    cache = planning_runtime_cache(input_guandong)
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


def read_closed_areas(input_guandong: InputAdapterGd) -> set[str]:
    return input_guandong.closed_area


def calculate_declared_export_demand(
    input_guandong: InputAdapterGd,
    voyage_ids: Sequence[str],
) -> list[DeclaredExportDemand]:
    """Summarize declared, not-yet-gated-in export demand only."""
    export_voyages = classified_export_voyages(input_guandong)
    rows: list[DeclaredExportDemand] = []
    for voyage_id in (normalize_voyage(value) for value in voyage_ids):
        if voyage_id not in export_voyages:
            continue
        for (flow, size, port), quantity in sorted(
            read_doc_by_port_size(input_guandong, voyage_id).items()
        ):
            rows.append(
                DeclaredExportDemand(
                    voyage_id=voyage_id,
                    flow=flow,
                    port=port,
                    size=size,
                    declared_boxes=int(quantity),
                )
            )
    return rows


def existing_operational_group_loads(
    input_guandong: InputAdapterGd,
    planning_time: datetime,
    target_voyages: set[str],
    valid_bay_keys: set[str],
    attribute_rules: AttributeRules | None = None,
) -> tuple[Counter[tuple[str, ...]], Counter[tuple[str, ...]]]:
    """Count current yard boxes by the single operational-group key.

    Current yard boxes are not part of the new demand, but their location is
    useful as a soft anchor for placing the same operational group nearby.
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
    occupied = active_yard_rows(occupied)
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
            flow = normalize_planning_flow(row.get("IYC_STS_CSTATUSCD"), default="OF")
            size = normalize_container_size(row.get("IYC_CSZ_CSIZECD"))
            port = normalize_text(row.get("IYC_POT_UNLDPORT"), "UNK")
            record = normalized_doc_record(row, flow, size, port)
            record["IYC_EVOY_ID"] = normalize_voyage(row.get("IYC_EVOY_ID"))
            record["IYC_IVOY_ID"] = normalize_voyage(row.get("IYC_IVOY_ID"))
            group_key = configured_operational_group_key(record, voyage_id, attribute_rules, export_voyages)
            container_key = str(row.get("_container_key", ""))
            anchor_container_key = (group_key, container_key)
            if container_key and anchor_container_key in seen_anchor_containers:
                continue
            if container_key:
                seen_anchor_containers.add(anchor_container_key)
            area_no = str(row.get("_area", ""))
            bay_key = str(row.get("_bay_key", ""))
            if not area_no or not bay_key:
                continue
            area_load[group_key + (area_no,)] += 1
            bay_load[group_key + (area_no, bay_key)] += 1
    return area_load, bay_load


def configured_operational_group_key(
    row: Mapping[str, Any],
    voyage_id: str,
    attribute_rules: AttributeRules,
    export_voyages: set[str],

) -> tuple[str, ...]:
    flow = normalize_planning_flow(row.get("IYC_STS_CSTATUSCD"), default="OF")
    size = normalize_container_size(row.get("IYC_CSZ_CSIZECD"))
    port = normalize_text(row.get("IYC_POT_UNLDPORT"), "UNK")
    if voyage_id in export_voyages:
        attrs = operational_group_attributes(attribute_rules, voyage_id)
    else:
        attrs, _values, _port_label = import_base_group_attributes(row, size, port)
    values = dynamic_attributes_from_row(row, attrs)
    scope = str(voyage_id) if voyage_id in export_voyages else "IMPORT"
    return (scope, f"flow={flow}", *(f"{attr}={values.get(attr, 'MIXED')}" for attr in attrs))


def container_identity(row: pd.Series, index: object) -> str:
    number = normalize_code(row.get("IYC_CNTRNO"))
    if number:
        return f"NO:{number}"
    cntr_id = normalize_code(row.get("IYC_CNTRID"))
    if cntr_id and cntr_id not in {"-1", "0"}:
        return f"ID:{cntr_id}"
    return f"ROW:{index}"


def read_doc_by_port_size(input_guandong: InputAdapterGd, voyage_id: str) -> Counter[tuple[str, str, str]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    frame = input_guandong.vessel_containers.get(voyage_id, {}).get("doc_cntrs", None)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return counter
    work = pd.DataFrame(
        {
            "flow": frame.get("IYC_STS_CSTATUSCD", pd.Series(index=frame.index, dtype=object)).map(
                lambda value: normalize_planning_flow(value, default="OF")
            ),
            "size": frame.get("IYC_CSZ_CSIZECD", pd.Series(index=frame.index, dtype=object)).map(normalize_container_size),
            "port": frame.get("IYC_POT_UNLDPORT", pd.Series(index=frame.index, dtype=object)).map(
                lambda value: normalize_text(value, "UNK")
            ),
        }
    )
    counts = work.groupby(["flow", "size", "port"], sort=False).size()
    counter.update({(str(flow), str(size), str(port)): int(count) for (flow, size, port), count in counts.items()})
    return counter


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
        date_field = _first_existing(fieldnames, ["plan_date", "date", "work_date", "planning_date", "day"])
        flow_field = _first_existing(fieldnames, ["flow", "cntr_type", "status"])
        for row in reader:
            flow = normalize_planning_flow(row.get(flow_field), default="OF") if flow_field else "OF"
            voyage_id = normalize_voyage(row.get("voyage_id"))
            area_no = normalize_code(row.get("area_no"))
            plan_date = date_key(normalize_text(row.get(date_field))) if date_field else ""
            for size_mode, field_name in (("20", qty20_field), ("40", qty40_field)):
                if not field_name:
                    continue
                boxes = int(round(float(row.get(field_name, 0) or 0)))
                if boxes > 0:
                    counter[(voyage_id, flow, area_no, size_mode, plan_date)] += boxes
    elif {"voy_id", "area_no", "new_qty"}.issubset(fieldnames):
        # new_qty is the allocation for containers that have not entered the
        # yard. planned_qty includes snapshot occupancy and must never be used
        # as downstream demand or reservation.
        qty_field = "new_qty"
        date_field = _first_existing(fieldnames, ["plan_date", "date", "work_date", "planning_date", "day"])
        flow_field = _first_existing(fieldnames, ["flow", "cntr_type", "status"])
        for row in reader:
            flow = normalize_planning_flow(row.get(flow_field), default="OF") if flow_field else "OF"
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
    elif {"voyage_id", "area_no", "new_boxes"}.issubset(fieldnames):
        size_field = "size_mode" if "size_mode" in fieldnames else "size"
        date_field = _first_existing(fieldnames, ["plan_date", "date", "work_date", "planning_date", "day"])
        flow_field = _first_existing(fieldnames, ["flow", "cntr_type", "status"])
        for row in reader:
            flow = normalize_planning_flow(row.get(flow_field), default="OF") if flow_field else "OF"
            boxes = int(round(float(row["new_boxes"])))
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


def load_export_groups(
    input_guandong: InputAdapterGd,
    voyage_ids: list[str],
    attribute_rules: AttributeRules,
) -> list[ExportGroup]:
    """Aggregate declared, not-yet-gated-in export containers."""
    groups: list[ExportGroup] = []
    export_voyages = classified_export_voyages(input_guandong)
    for voyage_id in voyage_ids:
        if voyage_id not in export_voyages:
            continue
        frame = input_guandong.vessel_containers.get(voyage_id, {}).get("doc_cntrs", None)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        group_columns = export_groupby_columns(attribute_rules, voyage_id)
        counter: Counter[tuple] = Counter()
        for row in frame.to_dict("records"):
            flow = normalize_planning_flow(row.get("IYC_STS_CSTATUSCD"), default="OF")
            size = normalize_container_size(row.get("IYC_CSZ_CSIZECD"))
            port = normalize_text(row.get("IYC_POT_UNLDPORT"), "UNK")
            height = normalize_text(row.get("IYC_CHEIGHTCD"), "UNK")
            record = normalized_doc_record(row, flow, size, port)
            values = dynamic_attributes_from_row(record, group_columns)
            key = (flow, size, port, height, tuple(values.get(column, "") for column in group_columns))
            counter[key] += 1
        for index, ((flow, size, port, height, dynamic_key), demand) in enumerate(sorted(counter.items()), start=1):
            groups.append(
                ExportGroup(
                    group_id=f"{voyage_id}_E{index:03d}", voyage_id=voyage_id,
                    status=str(flow), port=str(port), size=str(size), height=str(height),
                    demand=int(demand),
                    attributes={str(key): str(value) for key, value in zip(group_columns, dynamic_key)},
                )
            )
    return groups


def build_bays(
    input_guandong: InputAdapterGd,
    allowed_areas: set[str],
    closed_areas: set[str],
    planning_time: datetime,
    target_voyages: set[str],
    attribute_rules: AttributeRules | None = None,
) -> dict[str, Bay]:
    frame = input_guandong.bay_slots_detail.copy()
    frame["YAA_AREANO"] = frame["YAA_AREANO"].map(normalize_code)
    frame["YBY_BAYNO"] = frame["YBY_BAYNO"].map(normalize_bay)
    frame["YST_ROWNO"] = frame["YST_ROWNO"].map(normalize_row)
    frame = frame[frame["YAA_AREANO"].isin(allowed_areas) & ~frame["YAA_AREANO"].isin(closed_areas)].copy()

    reserved_slots = tops_reserved_slots(input_guandong, frame, planning_time, target_voyages)
    frame = drop_reserved_slots(frame, reserved_slots)

    large_bay_partner_by_bay = large_bay_partner_lookup_by_bay(frame)
    existing_large_pairs_by_member = existing_large_pair_members_by_bay(frame)
    large_shadow_slots = active_large_container_shadow_slots(
        frame,
        large_bay_partner_by_bay,
        existing_large_pairs_by_member,
    )
    available = drop_shadow_slots(available_empty_slots(frame), large_shadow_slots)
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
    for area_no, bay_keys in by_area.items():
        bay_nos = [key.split("|", 1)[1] for key in bay_keys]
        large_bay_partner.update(
            apply_large_bay_pair_capacities(
                area_no, bay_nos, cap_by_size, physical_cap, existing_large_pairs_by_member,
            )
        )
    for (area_no, bay_no), physical in physical_cap.items():
        bay_key = f"{area_no}|{bay_no}"
        attrs = existing_attrs.get((area_no, bay_no), {})
        bays[bay_key] = Bay(
            area_no=area_no,
            bay_no=bay_no,
            bay_key=bay_key,
            bay_order=bay_code_value(bay_no) or 0,
            cap_by_size={size: cap_by_size[size].get((area_no, bay_no), 0) for size in SIZE_MODES},
            physical_capacity=physical,
            row_cap_by_size={
                size: dict(row_cap_by_size_by_bay[size].get((area_no, bay_no), {}))
                for size in SIZE_MODES
            },
            row_physical_capacity=dict(row_physical_cap_by_bay.get((area_no, bay_no), {})),
            large_bay_partner_key=(
                f"{area_no}|{large_bay_partner[(area_no, bay_no)]}"
                if (area_no, bay_no) in large_bay_partner
                else ""
            ),
            existing_size_modes=set(attrs.get("sizes", set())),
            existing_heights=set(attrs.get("heights", set())),
            existing_ports_by_row={
                str(row_no): set(values)
                for row_no, values in attrs.get("ports_by_row", {}).items()
            },
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
    return bays


def tops_reserved_slots(
    input_guandong: InputAdapterGd,
    frame: pd.DataFrame,
    planning_time: datetime,
    target_voyages: set[str],
) -> set[tuple[str, str, str]]:
    active = active_tops_rows(input_guandong, planning_time)
    active = active[~active["condition_vessel"].isin({normalize_voyage(v) for v in target_voyages})].copy()
    reserved: set[tuple[str, str, str]] = set()
    if active.empty or frame.empty:
        return reserved
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
    return reserved


def drop_reserved_slots(frame: pd.DataFrame, reserved_slots: set[tuple[str, str, str]]) -> pd.DataFrame:
    if not reserved_slots or frame.empty:
        return frame
    empty = frame["HAS_CONTAINER"].fillna(0).astype(int).eq(0)
    keys = list(zip(frame["YAA_AREANO"], frame["YBY_BAYNO"], frame["YST_ROWNO"]))
    mask = [not (is_empty and key in reserved_slots) for is_empty, key in zip(empty, keys)]
    return frame.loc[mask].copy()


def active_occupied(frame: pd.DataFrame) -> pd.DataFrame:
    """Return containers physically occupying the yard snapshot."""
    return frame[frame["HAS_CONTAINER"].fillna(0).astype(int).eq(1)].copy()


def available_empty_slots(frame: pd.DataFrame) -> pd.DataFrame:
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
            if next_bay in bay_set and are_consecutive_bays(bay_no, next_bay):
                out[(str(area_no), bay_no)] = next_bay
            elif prev_bay in bay_set and are_consecutive_bays(prev_bay, bay_no):
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
    if enable40 and next_bay in bay_set and are_consecutive_bays(bay_no, next_bay):
        return area_no, bay_no, next_bay
    if prev_bay in bay_set and are_consecutive_bays(prev_bay, bay_no):
        return area_no, prev_bay, bay_no
    if next_bay in bay_set and are_consecutive_bays(bay_no, next_bay):
        return area_no, bay_no, next_bay
    return None


def consecutive_large_pairs_from_bays(area_no: str, bay_nos: set[str]) -> list[tuple[str, str, str]]:
    ordered = sorted((bay_no for bay_no in bay_nos if bay_no), key=bay_sort_key)
    pairs: list[tuple[str, str, str]] = []
    for left, right in zip(ordered, ordered[1:]):
        if are_consecutive_bays(left, right):
            pairs.append((area_no, left, right))
    return pairs


def existing_large_pair_members_by_bay(
    frame: pd.DataFrame,
) -> dict[tuple[str, str], set[frozenset[str]]]:
    bay_set_by_area: defaultdict[str, set[str]] = defaultdict(set)
    if frame.empty:
        return {}
    for row in frame[["YAA_AREANO", "YBY_BAYNO"]].drop_duplicates().to_dict("records"):
        area_no = normalize_code(row.get("YAA_AREANO"))
        bay_no = normalize_bay(row.get("YBY_BAYNO"))
        if area_no and bay_no:
            bay_set_by_area[area_no].add(bay_no)
    occupied = active_occupied(frame)
    out: defaultdict[tuple[str, str], set[frozenset[str]]] = defaultdict(set)
    if occupied.empty:
        return {}
    occupied = occupied.copy()
    occupied["_area_no"] = occupied.get("YAA_AREANO", pd.Series(index=occupied.index, dtype=object)).map(normalize_code)
    occupied["_bay_no"] = occupied.get("YBY_BAYNO", pd.Series(index=occupied.index, dtype=object)).map(normalize_bay)
    occupied["_size"] = occupied.get("IYC_CSZ_CSIZECD", pd.Series(index=occupied.index, dtype=object)).map(normalize_container_size)
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
        size = normalize_container_size(row.get("IYC_CSZ_CSIZECD"))
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
    large_bay_partner_by_bay: Mapping[tuple[str, str], str],
    existing_large_pairs_by_member: Mapping[tuple[str, str], set[frozenset[str]]] | None = None,
) -> set[tuple[str, str, str, str, str]]:
    if not large_bay_partner_by_bay and not existing_large_pairs_by_member:
        return set()
    occupied = active_occupied(frame)
    if occupied.empty:
        return set()
    existing_large_pairs_by_member = existing_large_pairs_by_member or {}
    out: set[tuple[str, str, str, str, str]] = set()
    for row in occupied.to_dict("records"):
        size = normalize_container_size(row.get("IYC_CSZ_CSIZECD"))
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
        if not are_consecutive_bays(left, right):
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


def are_consecutive_bays(left: str, right: str) -> bool:
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


def existing_bay_attributes(
    frame: pd.DataFrame,
    attribute_rules: AttributeRules | None = None,
    large_bay_partner_by_bay: Mapping[tuple[str, str], str] | None = None,
    existing_large_pairs_by_member: Mapping[tuple[str, str], set[frozenset[str]]] | None = None,
) -> dict[tuple[str, str], dict[str, set[str]]]:
    occupied = active_occupied(frame)
    out: dict[tuple[str, str], dict[str, set[str]]] = {}
    dynamic_attrs: list[str] = []
    if attribute_rules is not None:
        for attrs in (
            attribute_rules.bay_no_mix_attributes,
            attribute_rules.row_no_mix_attributes,
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
        size = normalize_container_size(row.get("IYC_CSZ_CSIZECD"))
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
                    "ports_by_row": {},
                    "attributes": {},
                    "attributes_by_row": {},
                    "attributes_by_voyage": {},
                    "attributes_by_row_by_voyage": {},
                },
            )
            attrs["sizes"].add(size)
            attrs["heights"].add(normalize_text(row.get("IYC_CHEIGHTCD"), "UNK"))
            port = normalize_text(row.get("IYC_POT_UNLDPORT"))
            row_no = normalize_row(row.get("YST_ROWNO"))
            if port and row_no:
                attrs["ports_by_row"].setdefault(row_no, set()).add(port)
            row_attrs = attrs["attributes_by_row"].setdefault(row_no, {}) if row_no else {}
            if row_no and row_voyages:
                row_attrs.setdefault(EXPORT_VOYAGE_ROW_NO_MIX_ATTR, set()).update(row_voyages)
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


def build_problem(
    input_guandong: InputAdapterGd,
    big_plan: list[BigPlanRow],
    planning_time: datetime,
    target_voyages: list[str],
) -> ProblemData:
    closed = read_closed_areas(input_guandong)
    area_functions = read_area_functions(input_guandong)
    function_areas = set(area_functions)

    area_guidance_target: dict[tuple[str, str, str, str], int] = {}
    cleaned_plan: list[BigPlanRow] = []
    target_voyages = [normalize_voyage(v) for v in target_voyages]
    # Fixed, publication-oriented grouping and compatibility rules.
    attribute_rules = AttributeRules(
        group_attributes=("IYC_CSZ_CSIZECD", "IYC_POT_UNLDPORT", "IYC_CHEIGHTCD"),
        bay_no_mix_attributes=("IYC_CHEIGHTCD",),
        row_no_mix_attributes=("IYC_POT_UNLDPORT",),
    )
    # Paper model: remove terminal-specific manual allow/block/required-area
    # controls. Feasibility is defined by yard functions and the upstream big
    # plan; operator overrides remain outside the mathematical model.
    allowed_areas_by_voyage = {voyage_id: set(function_areas) for voyage_id in target_voyages}
    berth_by_voyage = read_export_berths(input_guandong, target_voyages)
    plan_date = planning_time.date().isoformat()
    target_big_plan_flows = {planning_area_flow(flow) for flow in DEFAULT_TARGET_BIG_PLAN_FLOWS}
    target_voyage_set = set(target_voyages)
    all_export_voyages = classified_export_voyages(input_guandong)
    export_voyages = all_export_voyages & target_voyage_set
    # Detailed row allocation is limited to the selected export voyages, but
    # every import row in the same big-plan snapshot is an external capacity
    # commitment. Keeping these scopes separate prevents ``--voyages`` from
    # accidentally disabling import-capacity protection.
    input_plan = [
        row
        for row in big_plan
        if (not row.plan_date or row.plan_date == plan_date)
        and (row.voyage_id in export_voyages or row.voyage_id not in all_export_voyages)
    ]
    allowed_areas = set().union(*(set(areas) for areas in allowed_areas_by_voyage.values())) if allowed_areas_by_voyage else set(function_areas)
    skipped_closed_area: Counter[tuple[str, str]] = Counter()
    skipped_flow_function: Counter[tuple[str, str]] = Counter()
    for row in input_plan:
        plan_flow = planning_area_flow(row.flow)
        if plan_flow not in target_big_plan_flows:
            continue
        is_import = row.voyage_id not in all_export_voyages
        if not is_import:
            if row.area_no not in allowed_areas_by_voyage.get(row.voyage_id, set(function_areas)):
                continue
            if row.area_no in closed:
                skipped_closed_area[(row.voyage_id, row.area_no)] += row.new_boxes
                continue
            if not area_allows_flow(row.area_no, plan_flow, area_functions):
                skipped_flow_function[(row.voyage_id, row.area_no)] += row.new_boxes
                continue
        cleaned_plan.append(row)
    # Export groups contain declared, not-yet-gated-in containers only.
    export_groups = load_export_groups(input_guandong, target_voyages, attribute_rules)
    # Only declared export containers receive detailed row-level decisions.
    export_groups = [group for group in export_groups if group.voyage_id in export_voyages]

    demand_by_voyage_size: Counter[tuple[str, str, str]] = Counter()
    for group in export_groups:
        big_size = "40" if group.size == "45" else group.size
        demand_by_voyage_size[(group.voyage_id, group.status, big_size)] += group.demand
    upstream_area_size_weights: Counter[tuple[str, str, str, str]] = Counter()
    for row in cleaned_plan:
        if row.voyage_id not in export_voyages:
            continue
        plan_flow = planning_area_flow(row.flow)
        upstream_area_size_weights[(row.voyage_id, plan_flow, row.area_no, row.size_mode)] += row.new_boxes

    # Import new_qty supplies a reference distribution for anonymous capacity
    # reservation.  Its total is conserved by flow and size, while its area
    # distribution may move when the upstream area is not physically usable.
    # Export new_qty remains a soft area-distribution reference only.
    import_area_size_reference: Counter[tuple[str, str, str]] = Counter()
    for row in cleaned_plan:
        if row.voyage_id not in all_export_voyages:
            import_area_size_reference[
                (planning_area_flow(row.flow), row.area_no, row.size_mode)
            ] += row.new_boxes
    for voyage_id in target_voyages:
        flows = sorted({flow for (v, flow, _size), qty in demand_by_voyage_size.items() if v == voyage_id and qty > 0})
        for flow in flows:
            source_flow = planning_area_flow(flow)
            compatible_plan_flows = {source_flow}
            for size_mode in SIZE_MODES:
                target_qty = demand_by_voyage_size[(voyage_id, flow, size_mode)]
                if target_qty <= 0:
                    continue
                exact_upper = Counter(
                    {
                        area_no: qty
                        for (v, f, area_no, size), qty in upstream_area_size_weights.items()
                        if v == voyage_id and f in compatible_plan_flows and size == size_mode and qty > 0
                    }
                )
                if exact_upper:
                    # Normalize the upstream distribution to the declared
                    # export demand. No forecast-only quantity survives as a
                    # downstream target or capacity reservation.
                    allocations = allocate_by_weights(dict(exact_upper), target_qty)
                    for area_no, qty in allocations.items():
                        area_guidance_target[(voyage_id, flow, area_no, size_mode)] = qty
                    continue
    bays = build_bays(
        input_guandong,
        allowed_areas,
        closed,
        planning_time,
        set(target_voyages),
        attribute_rules,
    )
    existing_group_area_load, existing_group_bay_load = existing_operational_group_loads(
        input_guandong,
        planning_time,
        set(target_voyages),
        set(bays),
        attribute_rules,
    )
    berth_distances = read_distance_matrix(input_guandong)
    return ProblemData(
        export_groups=export_groups,
        bays=bays,
        area_guidance_target=area_guidance_target,
        area_functions=area_functions,
        target_voyages=target_voyages,
        export_voyages=export_voyages,
        import_area_size_reference=dict(import_area_size_reference),
        existing_group_area_load=dict(existing_group_area_load),
        existing_group_bay_load=dict(existing_group_bay_load),
        berth_distances=berth_distances,
        berth_by_voyage=berth_by_voyage,
        attribute_rules=attribute_rules,
    )


def load_planning_inputs(
    input_guandong: InputAdapterGd,
    planning_time: datetime,
    voyages: Sequence[str],
    big_plan: pd.DataFrame | Sequence[BigPlanRow] | None = None,
) -> PlanningInputs:
    if big_plan is None:
        big_plan_rows = read_big_plan(input_guandong.large_plan)
    elif isinstance(big_plan, pd.DataFrame):
        big_plan_rows = read_big_plan(big_plan)
    else:
        big_plan_rows = list(big_plan)
    demand_rows = calculate_declared_export_demand(input_guandong, voyages)
    problem = build_problem(
        input_guandong,
        big_plan_rows,
        planning_time=planning_time,
        target_voyages=list(voyages),
    )
    return PlanningInputs(demand_rows=demand_rows, problem=problem)


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


def _first_existing(columns: set[str], candidates: Sequence[str]) -> str | None:
    names = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in names:
            return names[candidate.lower()]
    return None


__all__ = [
    "PlanningInputs",
    "load_planning_inputs",
]
