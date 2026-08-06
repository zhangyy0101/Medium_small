from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from .models import (
    DEFAULT_BAY_NO_MIX_ATTRIBUTES,
    DEFAULT_ROW_NO_MIX_ATTRIBUTES,
    EXPORT_VOYAGE_ROW_NO_MIX_ATTR,
    ExportGroup,
    ProblemData,
)


SIZE_ATTRIBUTES = frozenset({"IYC_CSZ_CSIZECD", "SIZE", "SIZE_MODE"})
HEIGHT_ATTRIBUTES = frozenset({"IYC_CHEIGHTCD", "HEIGHT"})


def _group_attribute_value(group: ExportGroup, attribute: str) -> str:
    value = group.attributes.get(attribute, "")
    if isinstance(value, bool):
        return "1" if value else "0"
    if value not in (None, ""):
        return str(value)
    return str(
        {
            "IYC_STS_CSTATUSCD": group.status,
            "STATUS": group.status,
            "FLOW": group.status,
            "IYC_CSZ_CSIZECD": group.size,
            "SIZE": group.size,
            "SIZE_MODE": group.size,
            "IYC_POT_UNLDPORT": group.port,
            "PORT": group.port,
            "IYC_CHEIGHTCD": group.height,
            "HEIGHT": group.height,
            "IYC_EVOY_ID": group.voyage_id,
            "IYC_IVOY_ID": group.voyage_id,
            "VOYAGE_ID": group.voyage_id,
            EXPORT_VOYAGE_ROW_NO_MIX_ATTR.upper(): group.voyage_id,
        }.get(str(attribute).strip().upper(), "")
    )


def _attribute_scope(attribute: str, voyage_id: str) -> str:
    upper = str(attribute).strip().upper()
    return "" if upper in SIZE_ATTRIBUTES | HEIGHT_ATTRIBUTES | {EXPORT_VOYAGE_ROW_NO_MIX_ATTR.upper()} else voyage_id


def _bay_no_mix_attributes(problem: ProblemData, voyage_id: str) -> tuple[str, ...]:
    rules = getattr(problem, "attribute_rules", None)
    configured = (
        rules.bay_no_mix_for(voyage_id)
        if rules is not None
        else DEFAULT_BAY_NO_MIX_ATTRIBUTES
    )
    ordered = ["IYC_CSZ_CSIZECD", "IYC_CHEIGHTCD"]
    for attribute in configured:
        name = str(attribute).strip()
        if name.upper() in SIZE_ATTRIBUTES:
            name = "IYC_CSZ_CSIZECD"
        if name and name not in ordered:
            ordered.append(name)
    return tuple(ordered)


def _row_no_mix_attributes(problem: ProblemData, voyage_id: str) -> tuple[str, ...]:
    rules = getattr(problem, "attribute_rules", None)
    configured = (
        rules.row_no_mix_for(voyage_id)
        if rules is not None
        else DEFAULT_ROW_NO_MIX_ATTRIBUTES
    )
    ordered = [str(attribute) for attribute in configured if str(attribute)]
    export_voyages = getattr(problem, "export_voyages", None)
    is_export = (
        voyage_id in {str(value) for value in export_voyages}
        if export_voyages is not None
        else any(str(group.voyage_id) == voyage_id for group in problem.export_groups)
    )
    if is_export and EXPORT_VOYAGE_ROW_NO_MIX_ATTR not in ordered:
        ordered.append(EXPORT_VOYAGE_ROW_NO_MIX_ATTR)
    return tuple(ordered)


def _existing_bay_attribute_values(bay, attribute: str, voyage_id: str) -> set[str]:
    upper = str(attribute).strip().upper()
    if upper in SIZE_ATTRIBUTES:
        return set(bay.existing_attrs.get(attribute, set())) or set(bay.existing_size_modes)
    if upper in HEIGHT_ATTRIBUTES:
        return set(bay.existing_heights)
    return set(
        bay.existing_attrs_by_voyage.get(str(voyage_id), {}).get(attribute, set())
    )


def _existing_row_attribute_values(
    bay,
    row_no: str,
    attribute: str,
    voyage_id: str,
) -> set[str]:
    upper = str(attribute).strip().upper()
    if upper in SIZE_ATTRIBUTES | {EXPORT_VOYAGE_ROW_NO_MIX_ATTR.upper()}:
        return set(
            bay.existing_attrs_by_row.get(str(row_no), {}).get(attribute, set())
        )
    return set(
        bay.existing_attrs_by_row_by_voyage
        .get(str(row_no), {})
        .get(str(voyage_id), {})
        .get(attribute, set())
    )


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return []
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_integer(raw_value: object, label: str, errors: list[str]) -> int | None:
    try:
        numeric = float(raw_value or 0)
    except (TypeError, ValueError):
        errors.append(f"non-numeric integer field: {label}, value={raw_value}")
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        errors.append(f"non-integer field: {label}, value={raw_value}")
        return None
    return int(numeric)


def validate_output_files(
    problem: ProblemData,
    export_row_plan_path: str | Path,
    import_reservation_path: str | Path,
) -> dict[str, int | bool]:
    """Validate written CSVs using input data only, without planner state."""
    plan = _read_rows(export_row_plan_path)
    import_rows = _read_rows(import_reservation_path)
    errors: list[str] = []

    groups_by_id = {group.group_id: group for group in problem.export_groups}
    demand = {group_id: int(group.demand) for group_id, group in groups_by_id.items()}
    assigned: Counter[str] = Counter()
    bay_load: Counter[str] = Counter()
    bay_size_load: Counter[tuple[str, str]] = Counter()
    row_load: Counter[tuple[str, str]] = Counter()
    row_size_load: Counter[tuple[str, str, str]] = Counter()
    bay_sizes: defaultdict[str, set[str]] = defaultdict(set)
    bay_heights: defaultdict[str, set[str]] = defaultdict(set)
    row_voyages: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    row_ports: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    bay_attribute_values: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
    row_attribute_values: defaultdict[tuple[str, str, str, str], set[str]] = defaultdict(set)

    bays_by_area: defaultdict[str, list[str]] = defaultdict(list)
    for key, bay in problem.bays.items():
        bays_by_area[bay.area_no].append(key)
    edge_large_bays: set[str] = set()
    for keys in bays_by_area.values():
        keys.sort(key=lambda key: problem.bays[key].bay_order)
        if not keys:
            continue
        boundaries = {keys[0], keys[-1]}
        edge_large_bays.update(
            key
            for key in keys
            if problem.bays[key].large_bay_partner_key
            and (
                key in boundaries
                or problem.bays[key].large_bay_partner_key in boundaries
            )
        )

    for row in plan:
        group_id = str(row.get("group_id", ""))
        qty = _parse_integer(
            row.get("planned_boxes", 0),
            f"planned_boxes[group={group_id}]",
            errors,
        )
        if qty is None:
            continue
        area = str(row.get("area_no", ""))
        bay_no = str(row.get("bay_no", ""))
        bay_key = f"{area}|{bay_no}"
        row_no = str(row.get("row_no", ""))
        size = str(row.get("size", ""))
        height = str(row.get("height", ""))
        voyage = str(row.get("voyage_id", ""))
        port = str(row.get("port", ""))
        if group_id not in demand:
            errors.append(f"unknown group in output: {group_id}")
            continue
        group = groups_by_id[group_id]
        expected_identity = {
            "voyage_id": str(group.voyage_id),
            "flow": str(group.status),
            "port": str(group.port),
            "size": str(group.size),
            "height": str(group.height),
        }
        for field, expected in expected_identity.items():
            actual = str(row.get(field, ""))
            if actual != expected:
                errors.append(
                    f"group identity mismatch: group={group_id}, field={field}, "
                    f"output={actual}, input={expected}"
                )
        for attribute, expected in group.attributes.items():
            actual = str(row.get(str(attribute), ""))
            expected_text = "1" if expected is True else "0" if expected is False else str(expected)
            if actual != expected_text:
                errors.append(
                    f"group attribute mismatch: group={group_id}, attribute={attribute}, "
                    f"output={actual}, input={expected_text}"
                )
        if qty <= 0 or bay_key not in problem.bays:
            errors.append(f"invalid output row for group {group_id}: bay={bay_key}, qty={qty}")
            continue
        bay = problem.bays[bay_key]
        provided_bay_key = str(row.get("bay_key", ""))
        if provided_bay_key and provided_bay_key != bay_key:
            errors.append(
                f"bay identity mismatch: output={provided_bay_key}, area/bay={bay_key}"
            )
        if area != bay.area_no or bay_no != bay.bay_no:
            errors.append(
                f"bay coordinates mismatch: key={bay_key}, output={area}|{bay_no}, "
                f"input={bay.area_no}|{bay.bay_no}"
            )
        required_flow = "OF" if group.status == "OF" else str(group.status)
        if required_flow not in problem.area_functions.get(area, set()):
            errors.append(
                f"export area-function violation: group={group_id}, "
                f"flow={required_flow}, area={area}"
            )
        footprint = [bay_key]
        if size in {"40", "45"}:
            if (
                not bay.large_bay_partner_key
                or bay.large_bay_partner_key not in problem.bays
            ):
                errors.append(f"large container lacks paired bay: {group_id}, {bay_key}")
                continue
            footprint.append(bay.large_bay_partner_key)
        if size == "45" and bay_key not in edge_large_bays:
            errors.append(f"45-ft container is not on an edge large bay: {group_id}, {bay_key}")
        expected_row_allocation = "|".join(
            f"{key}:{row_no}:1" for key in sorted(footprint)
        )
        actual_row_allocation = str(row.get("row_allocation", ""))
        if actual_row_allocation != expected_row_allocation:
            errors.append(
                f"row footprint mismatch: group={group_id}, output={actual_row_allocation}, "
                f"expected={expected_row_allocation}"
            )
        assigned[group_id] += qty
        for key in footprint:
            footprint_bay = problem.bays[key]
            if footprint_bay.existing_size_modes and size not in footprint_bay.existing_size_modes:
                errors.append(
                    f"incumbent size conflict: {key}, existing={sorted(footprint_bay.existing_size_modes)}, new={size}"
                )
            if footprint_bay.existing_heights and height not in footprint_bay.existing_heights:
                errors.append(
                    f"incumbent height conflict: {key}, existing={sorted(footprint_bay.existing_heights)}, new={height}"
                )
            existing_ports = footprint_bay.existing_ports_by_row.get(row_no, set())
            if existing_ports and port not in existing_ports:
                errors.append(
                    f"incumbent row-port conflict: {key}, row={row_no}, "
                    f"existing={sorted(existing_ports)}, new={port}"
                )
            existing_voyages = (
                footprint_bay.existing_attrs_by_row.get(row_no, {}).get(
                    EXPORT_VOYAGE_ROW_NO_MIX_ATTR, set()
                )
            )
            if existing_voyages and existing_voyages != {voyage}:
                errors.append(
                    f"incumbent row-voyage conflict: {key}, row={row_no}, "
                    f"existing={sorted(existing_voyages)}, new={voyage}"
                )
            bay_load[key] += qty
            row_load[(key, row_no)] += qty
            row_size_load[(key, row_no, size)] += qty
            bay_sizes[key].add(size)
            bay_heights[key].add(height)
            row_voyages[(key, row_no)].add(voyage)
            row_ports[(key, row_no)].add(port)
            for attribute in _bay_no_mix_attributes(problem, group.voyage_id):
                scope = _attribute_scope(attribute, str(group.voyage_id))
                value = _group_attribute_value(group, attribute)
                bay_attribute_values[(key, attribute, scope)].add(value)
                existing_values = _existing_bay_attribute_values(
                    footprint_bay,
                    attribute,
                    str(group.voyage_id),
                )
                if existing_values and existing_values != {value}:
                    errors.append(
                        f"incumbent bay-attribute conflict: bay={key}, "
                        f"attribute={attribute}, existing={sorted(existing_values)}, new={value}"
                    )
            for attribute in _row_no_mix_attributes(problem, group.voyage_id):
                scope = _attribute_scope(attribute, str(group.voyage_id))
                value = _group_attribute_value(group, attribute)
                row_attribute_values[(key, row_no, attribute, scope)].add(value)
                existing_values = _existing_row_attribute_values(
                    footprint_bay,
                    row_no,
                    attribute,
                    str(group.voyage_id),
                )
                if attribute == EXPORT_VOYAGE_ROW_NO_MIX_ATTR:
                    compatible = not existing_values or existing_values == {value}
                else:
                    compatible = not existing_values or value in existing_values
                if not compatible:
                    errors.append(
                        f"incumbent row-attribute conflict: bay={key}, row={row_no}, "
                        f"attribute={attribute}, existing={sorted(existing_values)}, new={value}"
                    )
        bay_size_load[(bay_key, size)] += qty

    import_required: Counter[tuple[str, str]] = Counter()
    for (flow, _area, size), qty in problem.import_area_size_reference.items():
        import_required[(str(flow), str(size))] += int(qty)
    import_reserved: Counter[tuple[str, str]] = Counter()
    for row in import_rows:
        flow = str(row.get("flow", ""))
        size = str(row.get("size", ""))
        area = str(row.get("area_no", ""))
        bay_no = str(row.get("bay_no", ""))
        bay_key = str(row.get("bay_key", "")) or f"{area}|{bay_no}"
        qty = _parse_integer(
            row.get("reserved_boxes", 0),
            f"reserved_boxes[flow={flow},size={size},bay={bay_key}]",
            errors,
        )
        if qty is None:
            continue
        if qty <= 0 or bay_key not in problem.bays:
            errors.append(
                f"invalid import reservation: flow={flow}, size={size}, bay={bay_key}, qty={qty}"
            )
            continue
        bay = problem.bays[bay_key]
        if area != bay.area_no:
            errors.append(
                f"import reservation area mismatch: bay={bay_key}, output={area}, input={bay.area_no}"
            )
        if bay_no and bay_no != bay.bay_no:
            errors.append(
                f"import reservation bay-number mismatch: bay={bay_key}, "
                f"output={bay_no}, input={bay.bay_no}"
            )
        if flow not in problem.area_functions.get(bay.area_no, set()):
            errors.append(
                f"import area-function violation: flow={flow}, area={bay.area_no}, bay={bay_key}"
            )
        if size not in {"20", "40"} or int(bay.cap_by_size.get(size, 0)) <= 0:
            errors.append(f"import bay-size violation: size={size}, bay={bay_key}")
            continue
        footprint = [bay_key]
        if size == "40":
            partner = str(bay.large_bay_partner_key or "")
            if not partner or partner not in problem.bays:
                errors.append(f"import 40-ft reservation lacks paired bay: {bay_key}")
                continue
            footprint.append(partner)
        output_slot_units = str(row.get("footprint_slot_units", "")).strip()
        if output_slot_units:
            expected_slot_units = qty * len(footprint)
            parsed_slot_units = _parse_integer(
                output_slot_units,
                f"footprint_slot_units[bay={bay_key}]",
                errors,
            )
            if parsed_slot_units is not None and parsed_slot_units != expected_slot_units:
                errors.append(
                    f"import footprint units mismatch: bay={bay_key}, "
                    f"output={output_slot_units}, expected={expected_slot_units}"
                )
        reservation_scope = str(row.get("reservation_scope", "")).strip()
        if reservation_scope and reservation_scope != "anonymous_capacity":
            errors.append(
                f"invalid import reservation scope: bay={bay_key}, scope={reservation_scope}"
            )
        import_reserved[(flow, size)] += qty
        for key in footprint:
            bay_load[key] += qty
        bay_size_load[(bay_key, size)] += qty

    for group_id, qty in demand.items():
        if assigned[group_id] != qty:
            errors.append(
                f"demand balance: {group_id}, assigned={assigned[group_id]}, "
                f"demand={qty}"
            )
    for key in sorted(set(import_required) | set(import_reserved)):
        if int(import_reserved[key]) != int(import_required[key]):
            errors.append(
                f"import reservation total: flow={key[0]}, size={key[1]}, "
                f"reserved={import_reserved[key]}, required={import_required[key]}"
            )
    for key, load in bay_load.items():
        if load > problem.bays[key].physical_capacity:
            errors.append(f"bay capacity: {key}, load={load}")
    for (key, size), load in bay_size_load.items():
        if load > int(problem.bays[key].cap_by_size.get(size, 0)):
            errors.append(f"bay-size capacity: {key}, size={size}, load={load}")
    for (key, row_no), load in row_load.items():
        bay = problem.bays[key]
        default_cap = bay.physical_capacity if row_no == "__bay__" else 0
        cap = int(bay.row_physical_capacity.get(row_no, default_cap))
        if load > cap:
            errors.append(f"row capacity: {key}, row={row_no}, load={load}, cap={cap}")
    for (key, row_no, size), load in row_size_load.items():
        bay = problem.bays[key]
        default_cap = bay.cap_by_size.get(size, 0) if row_no == "__bay__" else 0
        cap = int(bay.row_cap_by_size.get(size, {}).get(row_no, default_cap))
        if load > cap:
            errors.append(
                f"row-size capacity: {key}, row={row_no}, size={size}, load={load}, cap={cap}"
            )
    for key, values in bay_sizes.items():
        if len(values) > 1:
            errors.append(f"bay size mixing: {key}, values={sorted(values)}")
    for key, values in bay_heights.items():
        if len(values) > 1:
            errors.append(f"bay height mixing: {key}, values={sorted(values)}")
    for key, values in row_voyages.items():
        if len(values) > 1:
            errors.append(f"row voyage mixing: {key}, values={sorted(values)}")
    for key, values in row_ports.items():
        if len(values) > 1:
            errors.append(f"row port mixing: {key}, values={sorted(values)}")
    for (bay_key, attribute, scope), values in bay_attribute_values.items():
        if len(values) > 1:
            errors.append(
                f"bay attribute mixing: bay={bay_key}, attribute={attribute}, "
                f"scope={scope or 'GLOBAL'}, values={sorted(values)}"
            )
    for (bay_key, row_no, attribute, scope), values in row_attribute_values.items():
        if len(values) > 1:
            errors.append(
                f"row attribute mixing: bay={bay_key}, row={row_no}, "
                f"attribute={attribute}, scope={scope or 'GLOBAL'}, values={sorted(values)}"
            )

    if errors:
        raise ValueError("Output validation failed: " + "; ".join(errors[:20]))
    return {
        "passed": True,
        "plan_rows_checked": len(plan),
        "groups_checked": len(demand),
        "assigned_boxes_checked": int(sum(assigned.values())),
        "import_reservation_rows_checked": len(import_rows),
        "import_reserved_boxes_checked": int(sum(import_reserved.values())),
        "bay_attribute_states_checked": len(bay_attribute_values),
        "row_attribute_states_checked": len(row_attribute_values),
        "row_footprints_checked": len(plan),
    }
