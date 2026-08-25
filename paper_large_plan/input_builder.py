from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd

from .models import LargePlanData, VF, VFA


SNAPSHOT_COLUMNS = {
    "HAS_CONTAINER",
    "YAA_AREANO",
    "YBY_ENABLECSIZECD",
    "IYC_CNTRID",
    "IYC_EVOY_ID",
    "IYC_IVOY_ID",
    "IYC_CSZ_CSIZECD",
    "IYC_STS_CSTATUSCD",
}


def normalize_code(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def normalize_voyage(value: Any) -> str:
    text = normalize_code(value)
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def normalize_size(value: Any) -> str:
    code = normalize_code(value)
    if code.startswith("20"):
        return "20"
    if code.startswith(("40", "45")):
        return "40"
    return ""


def planning_area_flow(value: Any) -> str:
    flow = normalize_code(value) or "OF"
    if flow in {"OF", "IF", "IZ", "T"}:
        return flow
    return "OZ"


def parse_enable_size_flags(value: Any) -> tuple[bool, bool]:
    if value is None:
        return True, True
    try:
        if pd.isna(value):
            return True, True
    except (TypeError, ValueError):
        pass
    tokens = re.findall(r"\d+", str(value))
    if not tokens:
        return True, True
    modes = {normalize_size(token) for token in tokens}
    return "20" in modes, "40" in modes


def build_large_plan_data(input_data: InputAdapterGd) -> LargePlanData:
    """Build the deterministic known-box large plan without TOPS."""

    _validate_input_frames(input_data)
    content_by_voyage = {
        normalize_voyage(voyage): content
        for voyage, content in (input_data.vessel_containers or {}).items()
        if normalize_voyage(voyage) and isinstance(content, Mapping)
    }
    export_voyages = _discover_export_voyages(input_data, content_by_voyage)
    import_voyages = _discover_import_voyages(input_data, content_by_voyage)
    voyages = tuple(export_voyages + [v for v in import_voyages if v not in set(export_voyages)])
    if not voyages:
        raise ValueError("No export or import voyages with planning demand were found in the shared input.")

    direction_by_voyage = {
        **{voyage: "E" for voyage in export_voyages},
        **{voyage: "I" for voyage in import_voyages},
    }
    area_functions = _read_area_functions(input_data.area_function_info)
    closed_areas = {normalize_code(area) for area in (input_data.closed_area or set()) if normalize_code(area)}
    slot_areas = {
        area
        for area in input_data.bay_slots_detail["YAA_AREANO"].map(normalize_code).unique()
        if area
    }
    areas = tuple(sorted((set(area_functions) & slot_areas) - closed_areas))
    if not areas:
        raise ValueError("No usable yard area remains after applying area functions and closed_area.")

    snapshot_rows = _extract_snapshot_rows(
        input_data.bay_slots_detail,
        export_voyages,
        import_voyages,
    )
    snapshot20, snapshot40 = _snapshot_counts(snapshot_rows, set(areas))
    demand20, demand40, demand_diagnostics = _build_demands(
        content_by_voyage,
        export_voyages,
        import_voyages,
        snapshot_rows,
        set(areas),
    )
    capacity20_equiv, capacity20_direct, capacity40 = _build_empty_capacities(
        input_data.bay_slots_detail,
        set(areas),
    )
    berth_by_voyage = _read_berths(input_data.vessel_berth_info, voyages, direction_by_voyage)
    distance = _read_distances(
        input_data.berth_area_dist_matrix,
        voyages,
        areas,
        berth_by_voyage,
    )
    flows = tuple(
        sorted(
            {flow for _voyage, flow in demand20}
            | {flow for _voyage, flow in demand40}
            | {flow for _voyage, flow, _area in snapshot20}
            | {flow for _voyage, flow, _area in snapshot40}
        )
    )
    planning_time = pd.Timestamp(input_data.planning_time)
    if pd.isna(planning_time):
        raise ValueError("planning_time is missing or invalid in the shared input.")

    return LargePlanData(
        voyages=voyages,
        flows=flows,
        areas=areas,
        direction_by_voyage=direction_by_voyage,
        demand20=dict(demand20),
        demand40=dict(demand40),
        snapshot20=dict(snapshot20),
        snapshot40=dict(snapshot40),
        capacity20_equiv=capacity20_equiv,
        capacity20_direct=capacity20_direct,
        capacity40=capacity40,
        area_functions={area: frozenset(area_functions[area]) for area in areas},
        berth_by_voyage=berth_by_voyage,
        distance=distance,
        planning_time=planning_time.isoformat(),
        diagnostics={
            "export_voyages": export_voyages,
            "import_voyages": import_voyages,
            "closed_areas": sorted(closed_areas),
            "demand_policy": "yard_snapshot_plus_declared_documents_only",
            "prediction_input_ignored": True,
            "demand": demand_diagnostics,
        },
    )


def _validate_input_frames(input_data: InputAdapterGd) -> None:
    frames = {
        "bay_slots_detail": input_data.bay_slots_detail,
        "area_function_info": input_data.area_function_info,
        "vessel_berth_info": input_data.vessel_berth_info,
        "berth_area_dist_matrix": input_data.berth_area_dist_matrix,
    }
    for name, frame in frames.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{name} must be a non-empty DataFrame in the shared input.")
    missing = SNAPSHOT_COLUMNS - set(input_data.bay_slots_detail.columns)
    if missing:
        raise KeyError(f"bay_slots_detail is missing columns: {sorted(missing)}")


def _normalized_take_over(input_data: InputAdapterGd, direction: str) -> list[str]:
    raw = (input_data.take_over_vessel or {}).get(direction, [])
    out: list[str] = []
    for value in raw:
        voyage = normalize_voyage(value)
        if voyage and voyage not in out:
            out.append(voyage)
    return out


def _doc_frame(content: Mapping[str, Any]) -> pd.DataFrame | None:
    frame = content.get("doc_cntrs")
    return frame if isinstance(frame, pd.DataFrame) and not frame.empty else None


def _discover_export_voyages(
    input_data: InputAdapterGd,
    content_by_voyage: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    candidates = _normalized_take_over(input_data, "E") or sorted(content_by_voyage)
    return [
        voyage
        for voyage in candidates
        if voyage in content_by_voyage
        and normalize_code(content_by_voyage[voyage].get("type")) != "I"
        and _doc_frame(content_by_voyage[voyage]) is not None
    ]


def _discover_import_voyages(
    input_data: InputAdapterGd,
    content_by_voyage: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    candidates = _normalized_take_over(input_data, "I") or sorted(content_by_voyage)
    return [
        voyage
        for voyage in candidates
        if voyage in content_by_voyage
        and normalize_code(content_by_voyage[voyage].get("type")) == "I"
        and _doc_frame(content_by_voyage[voyage]) is not None
    ]


def _read_area_functions(frame: pd.DataFrame) -> dict[str, set[str]]:
    area_col = _first_column(frame, ("area_no", "AREA_NO", "YAA_AREANO"))
    function_col = _first_column(frame, ("cntr_type", "CNTR_TYPE", "function", "FUNCTION"))
    if area_col is None or function_col is None:
        raise KeyError("area_function_info must contain area_no and cntr_type columns.")
    functions: dict[str, set[str]] = {}
    for row in frame.to_dict("records"):
        area = normalize_code(row.get(area_col))
        if area:
            functions[area] = {
                normalize_code(part)
                for part in str(row.get(function_col, "")).split(",")
                if normalize_code(part)
            }
    return functions


def _extract_snapshot_rows(
    snapshot: pd.DataFrame,
    export_voyages: Sequence[str],
    import_voyages: Sequence[str],
) -> pd.DataFrame:
    occupied = snapshot[pd.to_numeric(snapshot["HAS_CONTAINER"], errors="coerce").fillna(0).eq(1)].copy()
    if occupied.empty:
        return pd.DataFrame(columns=["voyage", "direction", "flow", "size", "area", "cntr_id", "excluded"])
    export_set = set(export_voyages)
    import_set = set(import_voyages)
    occupied["e_voy"] = occupied["IYC_EVOY_ID"].map(normalize_voyage)
    occupied["i_voy"] = occupied["IYC_IVOY_ID"].map(normalize_voyage)
    occupied = occupied[occupied["e_voy"].isin(export_set) | occupied["i_voy"].isin(import_set)].copy()
    occupied["direction"] = occupied["e_voy"].isin(export_set).map(lambda value: "E" if value else "I")
    occupied["voyage"] = occupied["e_voy"].where(occupied["direction"].eq("E"), occupied["i_voy"])
    occupied["flow"] = occupied["IYC_STS_CSTATUSCD"].map(planning_area_flow)
    occupied.loc[occupied["direction"].eq("E"), "flow"] = "OF"
    occupied["size"] = occupied["IYC_CSZ_CSIZECD"].map(normalize_size)
    occupied["area"] = occupied["YAA_AREANO"].map(normalize_code)
    occupied["cntr_id"] = occupied["IYC_CNTRID"].map(normalize_code)
    occupied["excluded"] = occupied["direction"].eq("E") & occupied["i_voy"].ne("")
    rows = occupied[["voyage", "direction", "flow", "size", "area", "cntr_id", "excluded"]].copy()
    rows = rows[rows["cntr_id"].ne("") & rows["cntr_id"].ne("-1") & rows["size"].isin({"20", "40"})]
    return rows.sort_values(["cntr_id", "area"]).drop_duplicates("cntr_id", keep="first")


def _snapshot_counts(snapshot_rows: pd.DataFrame, areas: set[str]) -> tuple[Counter[VFA], Counter[VFA]]:
    count20: Counter[VFA] = Counter()
    count40: Counter[VFA] = Counter()
    included = snapshot_rows[~snapshot_rows["excluded"] & snapshot_rows["area"].isin(areas)]
    for (voyage, flow, area, size), quantity in included.groupby(["voyage", "flow", "area", "size"]).size().items():
        target = count20 if size == "20" else count40
        target[(str(voyage), str(flow), str(area))] += int(quantity)
    return count20, count40


def _normalized_doc(content: Mapping[str, Any], voyage: str, direction: str) -> pd.DataFrame:
    frame = _doc_frame(content)
    if frame is None:
        return pd.DataFrame(columns=["cntr_id", "flow", "size"])
    index = frame.index
    e_voy = frame.get("IYC_EVOY_ID", pd.Series("", index=index)).map(normalize_voyage)
    i_voy = frame.get("IYC_IVOY_ID", pd.Series("", index=index)).map(normalize_voyage)
    work = pd.DataFrame(
        {
            "cntr_id": frame.get("IYC_CNTRID", pd.Series("", index=index)).map(normalize_code),
            "flow": frame.get("IYC_STS_CSTATUSCD", pd.Series("OF", index=index)).map(planning_area_flow),
            "size": frame.get("IYC_CSZ_CSIZECD", pd.Series("", index=index)).map(normalize_size),
            "e_voy": e_voy,
            "i_voy": i_voy,
        },
        index=index,
    )
    voyage_column = "e_voy" if direction == "E" else "i_voy"
    work = work[work[voyage_column].eq(voyage)].copy()
    if direction == "E":
        work["flow"] = "OF"
    work = work[work["cntr_id"].ne("") & work["cntr_id"].ne("-1") & work["size"].isin({"20", "40"})]
    return work.sort_values("cntr_id").drop_duplicates("cntr_id", keep="first")


def _build_demands(
    content_by_voyage: Mapping[str, Mapping[str, Any]],
    export_voyages: Sequence[str],
    import_voyages: Sequence[str],
    snapshot_rows: pd.DataFrame,
    areas: set[str],
) -> tuple[Counter[VF], Counter[VF], dict[str, dict[str, Any]]]:
    demand20: Counter[VF] = Counter()
    demand40: Counter[VF] = Counter()
    diagnostics: dict[str, dict[str, Any]] = {}
    for voyage in [*export_voyages, *import_voyages]:
        direction = "E" if voyage in set(export_voyages) else "I"
        content = content_by_voyage[voyage]
        snap_all = snapshot_rows[snapshot_rows["voyage"].eq(voyage)].copy()
        snap = snap_all[~snap_all["excluded"]].copy()
        covered_snap = snap[snap["area"].isin(areas)].copy()
        doc = _normalized_doc(content, voyage, direction)
        excluded_ids = set(snap_all.loc[snap_all["excluded"], "cntr_id"])
        if excluded_ids:
            doc = doc[~doc["cntr_id"].isin(excluded_ids)].copy()
        snapshot_ids = set(snap_all["cntr_id"])
        doc_new = doc[~doc["cntr_id"].isin(snapshot_ids)].copy()
        for rows in (covered_snap, doc_new):
            for (flow, size), quantity in rows.groupby(["flow", "size"]).size().items():
                target = demand20 if size == "20" else demand40
                target[(voyage, str(flow))] += int(quantity)

        diagnostics[voyage] = {
            "direction": direction,
            "snapshot_rows": int(len(snap_all)),
            "snapshot_excluded_transshipment_rows": int(snap_all["excluded"].sum()),
            "doc_rows": int(len(doc)),
            "doc_new_rows": int(len(doc_new)),
        }
    return demand20, demand40, diagnostics


def _build_empty_capacities(
    snapshot: pd.DataFrame,
    areas: set[str],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    empty = snapshot[pd.to_numeric(snapshot["HAS_CONTAINER"], errors="coerce").fillna(0).eq(0)].copy()
    empty["area"] = empty["YAA_AREANO"].map(normalize_code)
    empty = empty[empty["area"].isin(areas)].copy()
    flags = empty["YBY_ENABLECSIZECD"].map(parse_enable_size_flags)
    empty["enable20"] = flags.map(lambda item: item[0])
    empty["enable40"] = flags.map(lambda item: item[1])
    all_counts = empty.groupby("area").size()
    counts20 = empty[empty["enable20"]].groupby("area").size()
    counts40 = empty[empty["enable40"]].groupby("area").size()
    return (
        {area: int(all_counts.get(area, 0)) for area in sorted(areas)},
        {area: int(counts20.get(area, 0)) for area in sorted(areas)},
        {area: int(counts40.get(area, 0)) for area in sorted(areas)},
    )


def _read_berths(
    frame: pd.DataFrame,
    voyages: Sequence[str],
    direction_by_voyage: Mapping[str, str],
) -> dict[str, str]:
    required = {"VOY_ID", "VOY_IEFG"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"vessel_berth_info is missing columns: {sorted(missing)}")
    work = frame.copy()
    work["voyage"] = work["VOY_ID"].map(normalize_voyage)
    work["direction"] = work["VOY_IEFG"].map(normalize_code)
    actual = work.get("VBT_BTH_ABTHNO", pd.Series("", index=work.index)).map(normalize_code)
    planned = work.get("VBT_BTH_PBTHNO", pd.Series("", index=work.index)).map(normalize_code)
    work["berth"] = actual.where(actual.ne(""), planned)
    result: dict[str, str] = {}
    for voyage in voyages:
        rows = work[work["voyage"].eq(voyage) & work["direction"].eq(direction_by_voyage[voyage])]
        if rows.empty or not rows.iloc[0]["berth"]:
            raise ValueError(f"No berth was found for voyage {voyage}.")
        berth = str(rows.iloc[0]["berth"])
        result[voyage] = berth if berth.startswith("B") else f"B{berth}"
    return result


def _read_distances(
    frame: pd.DataFrame,
    voyages: Sequence[str],
    areas: Sequence[str],
    berth_by_voyage: Mapping[str, str],
) -> dict[tuple[str, str], float]:
    area_col = _first_column(frame, ("area_no", "AREA_NO", "YAA_AREANO"))
    if area_col is None:
        raise KeyError("berth_area_dist_matrix must contain area_no.")
    work = frame.copy()
    work["_area"] = work[area_col].map(normalize_code)
    work = work.drop_duplicates("_area", keep="first").set_index("_area")
    columns = {normalize_code(column): column for column in frame.columns}
    distance: dict[tuple[str, str], float] = {}
    for voyage in voyages:
        berth = berth_by_voyage[voyage]
        column = columns.get(berth)
        if column is None:
            raise KeyError(f"berth_area_dist_matrix has no column for {berth} (voyage {voyage}).")
        for area in areas:
            if area not in work.index:
                raise KeyError(f"berth_area_dist_matrix has no row for area {area}.")
            value = pd.to_numeric(work.at[area, column], errors="coerce")
            if pd.isna(value):
                raise ValueError(f"Distance is missing for voyage {voyage}, berth {berth}, area {area}.")
            distance[(voyage, area)] = float(value)
    return distance


def _first_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    by_lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None
