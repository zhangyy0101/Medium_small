from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from block_bay_planning.models import EXPORT_VOYAGE_ROW_NO_MIX_ATTR, ProblemData


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return []
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_output_files(
    problem: ProblemData,
    export_row_plan_path: str | Path,
    unplaced_path: str | Path,
) -> dict[str, int | bool]:
    """Validate written CSVs using input data only, without planner state."""
    plan = _read_rows(export_row_plan_path)
    unplaced_rows = _read_rows(unplaced_path)
    errors: list[str] = []

    demand = {group.group_id: int(group.demand) for group in problem.small_groups}
    assigned: Counter[str] = Counter()
    unplaced: Counter[str] = Counter()
    bay_load: Counter[str] = Counter()
    bay_size_load: Counter[tuple[str, str]] = Counter()
    row_load: Counter[tuple[str, str]] = Counter()
    row_size_load: Counter[tuple[str, str, str]] = Counter()
    area_slot_load: Counter[str] = Counter()
    bay_sizes: defaultdict[str, set[str]] = defaultdict(set)
    bay_heights: defaultdict[str, set[str]] = defaultdict(set)
    row_voyages: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    row_ports: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    used_twenty_bays: set[str] = set()

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
        qty = int(float(row.get("planned_boxes", 0) or 0))
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
        if qty <= 0 or bay_key not in problem.bays:
            errors.append(f"invalid output row for group {group_id}: bay={bay_key}, qty={qty}")
            continue
        bay = problem.bays[bay_key]
        footprint = [bay_key]
        if size in {"40", "45"}:
            if not bay.large_bay_partner_key:
                errors.append(f"large container lacks paired bay: {group_id}, {bay_key}")
                continue
            footprint.append(bay.large_bay_partner_key)
        if size == "45" and bay_key not in edge_large_bays:
            errors.append(f"45-ft container is not on an edge large bay: {group_id}, {bay_key}")
        assigned[group_id] += qty
        area_slot_load[area] += qty * len(footprint)
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
        bay_size_load[(bay_key, size)] += qty
        if size == "20":
            used_twenty_bays.add(bay_key)

    for row in unplaced_rows:
        group_id = str(row.get("group_id", ""))
        qty = int(float(row.get("unplaced_boxes", row.get("quantity", 0)) or 0))
        if group_id:
            unplaced[group_id] += qty

    for group_id, qty in demand.items():
        if assigned[group_id] + unplaced[group_id] != qty:
            errors.append(
                f"demand balance: {group_id}, assigned={assigned[group_id]}, "
                f"unplaced={unplaced[group_id]}, demand={qty}"
            )
    for key, load in bay_load.items():
        if load > problem.bays[key].physical_capacity:
            errors.append(f"bay capacity: {key}, load={load}")
    for (key, size), load in bay_size_load.items():
        if load > int(problem.bays[key].cap_by_size.get(size, 0)):
            errors.append(f"bay-size capacity: {key}, size={size}, load={load}")
    for (key, row_no), load in row_load.items():
        cap = int(problem.bays[key].row_physical_capacity.get(row_no, 0))
        if load > cap:
            errors.append(f"row capacity: {key}, row={row_no}, load={load}, cap={cap}")
    for (key, row_no, size), load in row_size_load.items():
        cap = int(problem.bays[key].row_cap_by_size.get(size, {}).get(row_no, 0))
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

    for area, keys in bays_by_area.items():
        physical = sum(problem.bays[key].physical_capacity for key in keys)
        reserved = sum(
            qty * (2 if size in {"40", "45"} else 1)
            for (reserved_area, size), qty in problem.import_area_size_reservation.items()
            if reserved_area == area
        )
        if area_slot_load[area] + reserved > physical:
            errors.append(
                f"area reservation: {area}, export={area_slot_load[area]}, "
                f"import={reserved}, capacity={physical}"
            )

    # Reconstruct the model's usable large-bay pairs from input geometry.  A
    # newly occupied 20-ft member makes the pair unavailable to an incoming
    # 40/45-ft container.
    large_pairs: dict[tuple[str, str], tuple[int, int]] = {}
    for key, bay in problem.bays.items():
        partner = str(bay.large_bay_partner_key or "")
        if not partner or partner not in problem.bays:
            continue
        capacity = max(int(bay.cap_by_size.get("40", 0)), int(bay.cap_by_size.get("45", 0)))
        if capacity > 0:
            large_pairs[(key, partner)] = (capacity, int(bay.cap_by_size.get("45", 0)))
    for area in bays_by_area:
        available_large = sum(
            capacity
            for pair, (capacity, _capacity_45) in large_pairs.items()
            if problem.bays[pair[0]].area_no == area
            and not (set(pair) & used_twenty_bays)
        )
        required_large = sum(
            int(qty)
            for (reserved_area, size), qty in problem.import_area_size_reservation.items()
            if reserved_area == area and size in {"40", "45"}
        )
        if available_large < required_large:
            errors.append(
                f"import large-pair reservation: {area}, available={available_large}, "
                f"required={required_large}"
            )
        available_45 = sum(
            capacity_45
            for pair, (_capacity, capacity_45) in large_pairs.items()
            if problem.bays[pair[0]].area_no == area
            and not (set(pair) & used_twenty_bays)
        )
        required_45 = int(problem.import_area_size_reservation.get((area, "45"), 0))
        if available_45 < required_45:
            errors.append(
                f"import 45-ft pair reservation: {area}, available={available_45}, "
                f"required={required_45}"
            )

    if errors:
        raise ValueError("Output validation failed: " + "; ".join(errors[:20]))
    return {
        "passed": True,
        "plan_rows_checked": len(plan),
        "groups_checked": len(demand),
        "assigned_boxes_checked": int(sum(assigned.values())),
        "unplaced_boxes_checked": int(sum(unplaced.values())),
    }
