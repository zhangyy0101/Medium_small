from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime
from pathlib import Path
import sys
from typing import Hashable, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.input_adapter_gd import (  # noqa: E402
    InputAdapterGd,
    SafeJSONEncoder,
    normalize_voyage_id,
)
from adapters.planning_input import normalize_container_size  # noqa: E402


DEFAULT_BASE_INPUT = ROOT / "example" / "input_data.json"
DEFAULT_BASE_PLAN = ROOT / "example" / "large_plan.csv"
DEFAULT_OUTPUT = ROOT / "example" / "diverse_voyages_3x"
SOURCE_EXPORT_VOYAGES = ("390121", "390131")
GROUP_FIELDS = (
    "IYC_CSZ_CSIZECD",
    "IYC_CHEIGHTCD",
    "IYC_POT_UNLDPORT",
)
VOLUME_SPREAD = 0.12


def portable_source_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic diversified multi-voyage case while "
            "retaining the original yard snapshot and import commitments."
        )
    )
    parser.add_argument("--base-input", type=Path, default=DEFAULT_BASE_INPUT)
    parser.add_argument("--base-large-plan", type=Path, default=DEFAULT_BASE_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--copies",
        type=int,
        default=3,
        help="Total scale, including each original detailed export voyage.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated input, plan, and manifest.",
    )
    return parser.parse_args()


def synthetic_voyage_id(source: str, copy_index: int) -> str:
    return str(int(source) + 100_000 * copy_index)


def _stable_text(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def largest_remainder_allocation(
    weights: Mapping[Hashable, float],
    target_total: int,
    *,
    minimum_each: int = 0,
) -> dict[Hashable, int]:
    """Allocate an integer total deterministically without losing mass."""
    keys = sorted(weights, key=lambda value: str(value))
    if target_total < minimum_each * len(keys):
        raise ValueError("target total is smaller than the required minima")
    if any(float(weights[key]) < 0 for key in keys):
        raise ValueError("allocation weights must be nonnegative")
    remaining = target_total - minimum_each * len(keys)
    weight_total = sum(float(weights[key]) for key in keys)
    if not keys or weight_total <= 0:
        if target_total:
            raise ValueError("positive target requires at least one positive weight")
        return {key: minimum_each for key in keys}

    quotas = {
        key: remaining * float(weights[key]) / weight_total
        for key in keys
    }
    allocation = {
        key: minimum_each + math.floor(quotas[key])
        for key in keys
    }
    residue = target_total - sum(allocation.values())
    priority = sorted(
        keys,
        key=lambda key: (-(quotas[key] - math.floor(quotas[key])), str(key)),
    )
    for key in priority[:residue]:
        allocation[key] += 1
    if sum(allocation.values()) != target_total:
        raise AssertionError("largest-remainder allocation did not conserve total")
    return allocation


def clone_volume_targets(source_count: int, copies: int) -> dict[int, int]:
    clone_count = copies - 1
    if clone_count <= 0:
        return {}
    if clone_count == 1:
        return {1: source_count}
    factors = {
        copy_index: 1.0
        + VOLUME_SPREAD
        * (2.0 * (copy_index - 1) / (clone_count - 1) - 1.0)
        for copy_index in range(1, copies)
    }
    return {
        int(key): value
        for key, value in largest_remainder_allocation(
            factors,
            source_count * clone_count,
        ).items()
    }


def centered_rank(value: str, observed: list[str]) -> float:
    if len(observed) <= 1:
        return 0.0
    return 2.0 * observed.index(value) / (len(observed) - 1) - 1.0


def group_profile_score(
    key: tuple[str, str, str],
    observed: tuple[list[str], list[str], list[str]],
) -> float:
    size, height, port = key
    sizes, heights, ports = observed
    return (
        0.14 * centered_rank(size, sizes)
        + 0.06 * centered_rank(height, heights)
        + 0.10 * centered_rank(port, ports)
    )


def clone_direction(copy_index: int, copies: int) -> float:
    clone_count = copies - 1
    if clone_count <= 1:
        return 1.0
    return 2.0 * (copy_index - 1) / (clone_count - 1) - 1.0


def unique_container_values(length: int, target_voyage: str, field: str) -> pd.Series:
    prefix = "ID" if field == "IYC_CNTRID" else "NO"
    return pd.Series(
        [
            f"S{target_voyage}_{prefix}_{position:06d}"
            for position in range(1, length + 1)
        ],
        dtype="string",
    )


def diversified_documents(
    source_documents: pd.DataFrame,
    target_voyage: str,
    target_count: int,
    copy_index: int,
    copies: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    missing = [field for field in GROUP_FIELDS if field not in source_documents]
    if missing:
        raise KeyError(f"source documents are missing grouping fields: {missing}")
    source = source_documents.copy(deep=True)
    source["_group_key"] = list(
        zip(
            *(
                source[field].map(_stable_text)
                for field in GROUP_FIELDS
            )
        )
    )
    keys = sorted(source["_group_key"].unique())
    observed = tuple(
        sorted(source[field].map(_stable_text).unique())
        for field in GROUP_FIELDS
    )
    base_counts = source["_group_key"].value_counts().to_dict()
    direction = clone_direction(copy_index, copies)
    weights = {
        key: int(base_counts[key])
        * (1.0 + direction * group_profile_score(key, observed))
        for key in keys
    }
    allocation = largest_remainder_allocation(
        weights,
        target_count,
        minimum_each=1,
    )

    pieces: list[pd.DataFrame] = []
    profile: list[dict[str, object]] = []
    for group_position, key in enumerate(keys):
        candidates = source.loc[
            source["_group_key"].map(lambda value: value == key)
        ].drop(
            columns="_group_key"
        )
        count = int(allocation[key])
        offset = (copy_index * 17 + group_position * 11) % len(candidates)
        positions = [(offset + position) % len(candidates) for position in range(count)]
        pieces.append(candidates.iloc[positions].copy(deep=True))
        profile.append(
            {
                "size": key[0],
                "height": key[1],
                "port": key[2],
                "source_qty": int(base_counts[key]),
                "synthetic_qty": count,
            }
        )

    documents = pd.concat(pieces, ignore_index=True)
    documents["IYC_EVOY_ID"] = int(target_voyage)
    if "IYC_EVESVOYAGE" in documents:
        documents["IYC_EVESVOYAGE"] = f"SYN_{target_voyage}"
    for field in ("IYC_CNTRID", "IYC_CNTRNO"):
        if field in documents:
            documents[field] = unique_container_values(
                len(documents), target_voyage, field
            )
    if len(documents) != target_count:
        raise AssertionError("synthetic document total was not conserved")
    return documents, profile


def complete_distance_berths(adapter: InputAdapterGd) -> list[str]:
    matrix = adapter.berth_area_dist_matrix
    berths = []
    for column in matrix.columns:
        berth = str(column).upper()
        if not berth.startswith("B"):
            continue
        values = pd.to_numeric(matrix[column], errors="coerce")
        if values.notna().all():
            berths.append(berth)
    if not berths:
        raise ValueError("distance matrix has no complete berth column")
    return sorted(berths, key=lambda value: (int(value[1:]), value))


def synthetic_berth(
    berths: list[str],
    source_position: int,
    copy_index: int,
) -> str:
    # For the base seven-berth data this yields B2, B4, B5, and B7.
    position = 1 + 3 * (copy_index - 1) + 2 * source_position
    return berths[position % len(berths)]


def _iso(value: pd.Timestamp) -> str:
    return value.to_pydatetime().isoformat()


def diversified_berth_rows(
    frame: pd.DataFrame,
    source_voyage: str,
    target_voyage: str,
    target_berth: str,
    planning_time: datetime,
    source_position: int,
    copy_index: int,
) -> tuple[pd.DataFrame, str]:
    normalized = frame["VOY_ID"].map(normalize_voyage_id)
    rows = frame.loc[normalized.eq(source_voyage)].copy(deep=True)
    if rows.empty:
        raise ValueError(f"source export voyage has no berth row: {source_voyage}")
    rows["VOY_ID"] = int(target_voyage)
    rows["VBT_BTH_PBTHNO"] = target_berth.removeprefix("B")
    if "VBT_BTH_ABTHNO" in rows:
        rows["VBT_BTH_ABTHNO"] = None

    arrival_days = 3 + 8 * (copy_index - 1) + 3 * source_position
    arrival = pd.Timestamp(planning_time).normalize() + pd.Timedelta(
        days=arrival_days, hours=8 + source_position * 2
    )
    source_arrival = pd.to_datetime(rows.iloc[0].get("VBT_PBTHDT"), errors="coerce")
    source_departure = pd.to_datetime(rows.iloc[0].get("VBT_PDPTDT"), errors="coerce")
    turnaround = source_departure - source_arrival
    if pd.isna(turnaround) or turnaround <= pd.Timedelta(0):
        turnaround = pd.Timedelta(days=5)
    rows["SCD_RCVSTDT"] = _iso(arrival - pd.Timedelta(days=10))
    rows["SCD_RCVEDDT"] = _iso(arrival - pd.Timedelta(days=1))
    rows["VBT_PBTHDT"] = _iso(arrival)
    rows["VBT_PDPTDT"] = _iso(arrival + turnaround)
    for field in ("VBT_ABTHDT", "VBT_ADPTDT"):
        if field in rows:
            rows[field] = None
    for field in ("VSL_CNNAME", "VSL_ENNAME"):
        if field in rows:
            rows[field] = rows[field].astype("string") + f" SYN {target_voyage}"
    return rows, _iso(arrival)


def diversified_plan_rows(
    source_plan: pd.DataFrame,
    voyage_column: str,
    target_voyage: str,
    documents: pd.DataFrame,
    copy_index: int,
    copies: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    if "new_qty" not in source_plan:
        raise KeyError("diversified generation requires the new_qty plan format")
    positive = source_plan.loc[
        pd.to_numeric(source_plan["new_qty"], errors="coerce").fillna(0).gt(0)
    ].copy(deep=True)
    direction = clone_direction(copy_index, copies)
    document_sizes = documents["IYC_CSZ_CSIZECD"].map(normalize_container_size)
    document_sizes = document_sizes.map(lambda size: "40" if size == "45" else size)
    size_demand = document_sizes.value_counts().to_dict()
    pieces: list[pd.DataFrame] = []
    distribution: dict[str, dict[str, int]] = {}

    for size, target_total in sorted(size_demand.items()):
        size_rows = positive.loc[
            positive["size"].map(lambda value: str(int(float(value)))).eq(size)
        ].copy(deep=True)
        if size_rows.empty:
            raise ValueError(
                f"source plan has no positive {size}-foot row for {target_voyage}"
            )
        area_weights = (
            size_rows.assign(
                _new_qty=pd.to_numeric(size_rows["new_qty"], errors="raise")
            )
            .groupby("area_no", sort=True)["_new_qty"]
            .sum()
            .to_dict()
        )
        areas = sorted(area_weights, key=str)
        diversified_weights = {
            area: float(area_weights[area])
            * (
                1.0
                + direction
                * 0.20
                * centered_rank(str(area), [str(value) for value in areas])
            )
            for area in areas
        }
        allocation = largest_remainder_allocation(
            diversified_weights,
            int(target_total),
        )
        distribution[size] = {
            str(area): int(qty) for area, qty in allocation.items() if qty > 0
        }
        for area, quantity in allocation.items():
            if quantity <= 0:
                continue
            row = size_rows.loc[size_rows["area_no"].eq(area)].iloc[[0]].copy()
            row[voyage_column] = int(target_voyage)
            row["snapshot_qty"] = 0
            row["new_qty"] = int(quantity)
            row["planned_qty"] = int(quantity)
            pieces.append(row)

    plan = pd.concat(pieces, ignore_index=True)
    if int(plan["new_qty"].sum()) != len(documents):
        raise AssertionError("synthetic plan and declared demand totals differ")
    return plan, distribution


def build_case(
    base_input: Path,
    base_large_plan: Path,
    copies: int,
) -> tuple[InputAdapterGd, pd.DataFrame, dict]:
    if copies < 2:
        raise ValueError("copies must be at least 2 for a larger-voyage case")
    adapter = InputAdapterGd.load_from_json(str(base_input))
    large_plan = pd.read_csv(base_large_plan)
    voyage_column = next(
        (
            name
            for name in ("voy_id", "voyage_id", "VOY_ID")
            if name in large_plan.columns
        ),
        None,
    )
    if voyage_column is None:
        raise ValueError("large plan has no voyage-id column")

    export_takeover = [
        normalize_voyage_id(value)
        for value in adapter.take_over_vessel.get("E", [])
    ]
    complete_berths = complete_distance_berths(adapter)
    cloned_plan_rows: list[pd.DataFrame] = []
    cloned_berth_rows: list[pd.DataFrame] = []
    mapping: dict[str, list[str]] = {
        source: [source] for source in SOURCE_EXPORT_VOYAGES
    }
    profiles: dict[str, dict[str, object]] = {}
    plan_voyages = large_plan[voyage_column].map(normalize_voyage_id)

    for source_position, source_voyage in enumerate(SOURCE_EXPORT_VOYAGES):
        source_content = adapter.vessel_containers.get(source_voyage)
        if not isinstance(source_content, dict):
            raise ValueError(f"source voyage is missing: {source_voyage}")
        source_documents = source_content.get("doc_cntrs")
        if not isinstance(source_documents, pd.DataFrame) or source_documents.empty:
            raise ValueError(f"source voyage has no documents: {source_voyage}")
        source_plan = large_plan.loc[
            plan_voyages.eq(source_voyage)
        ].copy(deep=True)
        if source_plan.empty:
            raise ValueError(
                f"source export voyage has no large-plan row: {source_voyage}"
            )
        target_counts = clone_volume_targets(len(source_documents), copies)

        for copy_index in range(1, copies):
            target_voyage = synthetic_voyage_id(source_voyage, copy_index)
            if target_voyage in adapter.vessel_containers:
                raise ValueError(f"synthetic voyage already exists: {target_voyage}")
            documents, group_profile = diversified_documents(
                source_documents,
                target_voyage,
                target_counts[copy_index],
                copy_index,
                copies,
            )
            cloned_content = copy.deepcopy(source_content)
            cloned_content["doc_cntrs"] = documents
            cloned_content["type"] = "E"
            adapter.vessel_containers[target_voyage] = cloned_content

            berth = synthetic_berth(
                complete_berths,
                source_position,
                copy_index,
            )
            berth_rows, planned_berth_time = diversified_berth_rows(
                adapter.vessel_berth_info,
                source_voyage,
                target_voyage,
                berth,
                adapter.planning_time,
                source_position,
                copy_index,
            )
            plan_rows, plan_distribution = diversified_plan_rows(
                source_plan,
                voyage_column,
                target_voyage,
                documents,
                copy_index,
                copies,
            )
            cloned_berth_rows.append(berth_rows)
            cloned_plan_rows.append(plan_rows)
            export_takeover.append(target_voyage)
            mapping[source_voyage].append(target_voyage)
            profiles[target_voyage] = {
                "source_voyage": source_voyage,
                "declared_rows": len(documents),
                "volume_ratio_to_source": round(
                    len(documents) / len(source_documents), 6
                ),
                "estimated_berth": berth,
                "planned_berth_time": planned_berth_time,
                "group_profile": group_profile,
                "large_plan_new_qty": plan_distribution,
            }

    adapter.take_over_vessel["E"] = list(dict.fromkeys(export_takeover))
    adapter.vessel_berth_info = pd.concat(
        [adapter.vessel_berth_info, *cloned_berth_rows],
        ignore_index=True,
    )
    expanded_plan = pd.concat(
        [large_plan, *cloned_plan_rows], ignore_index=True
    )

    declared_rows = sum(
        len(adapter.vessel_containers[voyage]["doc_cntrs"])
        for voyages in mapping.values()
        for voyage in voyages
    )
    source_rows = sum(
        len(adapter.vessel_containers[voyage]["doc_cntrs"])
        for voyage in SOURCE_EXPORT_VOYAGES
    )
    if declared_rows != copies * source_rows:
        raise AssertionError("case does not conserve the requested total scale")
    manifest = {
        "case": f"diverse_voyages_{copies}x",
        "base_input": portable_source_path(base_input),
        "base_large_plan": portable_source_path(base_large_plan),
        "generation_policy": "deterministic_empirical_profile_perturbation",
        "yard_snapshot_policy": "unchanged_from_base_case",
        "import_commitment_policy": "unchanged_from_base_case",
        "synthetic_snapshot_policy": "zero; planned_qty equals new_qty",
        "demand_policy": (
            "source-observed size-height-port combinations only; clone volumes "
            "and combination shares vary while the overall total is conserved"
        ),
        "berth_policy": (
            "distinct estimated berths selected only from complete B-columns "
            "in the supplied berth-area distance matrix"
        ),
        "source_export_voyages": list(SOURCE_EXPORT_VOYAGES),
        "export_voyage_mapping": mapping,
        "synthetic_voyage_profiles": profiles,
        "detailed_export_voyage_count": sum(
            len(values) for values in mapping.values()
        ),
        "source_declared_export_rows": source_rows,
        "declared_export_container_rows": declared_rows,
        "declared_scale": declared_rows / source_rows,
        "large_plan_row_count": int(len(expanded_plan)),
    }
    return adapter, expanded_plan, manifest


def write_case(
    adapter: InputAdapterGd,
    large_plan: pd.DataFrame,
    manifest: dict,
    output_dir: Path,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input_data.json"
    plan_path = output_dir / "large_plan.csv"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (input_path, plan_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "generated case already exists; pass --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )

    with input_path.open("w", encoding="utf-8") as handle:
        json.dump(
            adapter.to_dict(),
            handle,
            cls=SafeJSONEncoder,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    large_plan.to_csv(plan_path, index=False, encoding="utf-8-sig")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"input={input_path} ({input_path.stat().st_size} bytes)")
    print(f"large_plan={plan_path} ({plan_path.stat().st_size} bytes)")
    print(f"manifest={manifest_path}")


def main() -> None:
    args = parse_args()
    adapter, large_plan, manifest = build_case(
        args.base_input.resolve(),
        args.base_large_plan.resolve(),
        args.copies,
    )
    write_case(
        adapter,
        large_plan,
        manifest,
        args.output_dir.resolve(),
        args.overwrite,
    )


if __name__ == "__main__":
    main()
