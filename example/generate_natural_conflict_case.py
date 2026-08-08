from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.input_adapter_gd import InputAdapterGd, normalize_voyage_id  # noqa: E402
from adapters.planning_input import normalize_container_size  # noqa: E402
from example.generate_diverse_voyages_case import (  # noqa: E402
    DEFAULT_BASE_INPUT,
    DEFAULT_BASE_PLAN,
    GROUP_FIELDS,
    largest_remainder_allocation,
    unique_container_values,
    write_case,
)
from example.generate_many_groups_case import (  # noqa: E402
    build_case as build_many_groups_case,
)


DEFAULT_OUTPUT = ROOT / "example" / "natural_conflict_peak"
DEFAULT_VOYAGE_COPIES = 3
DEFAULT_VOLUME_SCALE = 1.4
DEFAULT_FORTY_FIVE_SHARE = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reproducible peak-demand conflict case on the unchanged "
            "real yard snapshot. Demand is diversified rather than replicated."
        )
    )
    parser.add_argument("--base-input", type=Path, default=DEFAULT_BASE_INPUT)
    parser.add_argument(
        "--base-large-plan", type=Path, default=DEFAULT_BASE_PLAN
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--voyage-copies",
        type=int,
        default=DEFAULT_VOYAGE_COPIES,
        help=(
            "Diversified scale of each of the two source voyage families; "
            "three creates six non-identical detailed voyages."
        ),
    )
    parser.add_argument(
        "--volume-scale", type=float, default=DEFAULT_VOLUME_SCALE
    )
    parser.add_argument(
        "--forty-five-share",
        type=float,
        default=DEFAULT_FORTY_FIVE_SHARE,
        help=(
            "Share of large-container demand converted from 40-ft high-cube "
            "groups to 45-ft high-cube groups."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated input, plan, and manifest.",
    )
    return parser.parse_args()


def _stable_group_key(row: pd.Series) -> tuple[str, str, str]:
    return (
        normalize_container_size(row["IYC_CSZ_CSIZECD"]),
        str(row["IYC_CHEIGHTCD"]),
        str(row["IYC_POT_UNLDPORT"]),
    )


def _voyage_target_totals(
    source_totals: dict[str, int],
    volume_scale: float,
) -> dict[str, int]:
    voyages = sorted(source_totals)
    center = (len(voyages) - 1) / 2.0
    weights = {
        voyage: int(source_totals[voyage])
        * (1.0 + 0.035 * (position - center))
        for position, voyage in enumerate(voyages)
    }
    return {
        str(voyage): int(quantity)
        for voyage, quantity in largest_remainder_allocation(
            weights,
            round(sum(source_totals.values()) * float(volume_scale)),
        ).items()
    }


def _group_target_counts(
    source_counts: Counter[tuple[str, str, str]],
    target_total: int,
    voyage_position: int,
) -> dict[tuple[str, str, str], int]:
    weights = {
        key: int(source_quantity)
        * (
            1.0
            + 0.025
            * (((voyage_position + 3) * (position + 5)) % 7 - 3)
        )
        for position, (key, source_quantity) in enumerate(
            sorted(source_counts.items())
        )
    }
    return {
        key: int(quantity)
        for key, quantity in largest_remainder_allocation(
            weights,
            int(target_total),
            minimum_each=1,
        ).items()
    }


def _forty_five_counts(
    group_targets: dict[tuple[str, str, str], int],
    share: float,
) -> dict[tuple[str, str, str], int]:
    eligible = {
        key: quantity
        for key, quantity in group_targets.items()
        if key[0] == "40" and key[1].upper() == "HQ" and quantity >= 2
    }
    large_total = sum(
        quantity
        for (size, _height, _port), quantity in group_targets.items()
        if size == "40"
    )
    target = min(
        round(large_total * float(share)),
        sum(quantity - 1 for quantity in eligible.values()),
    )
    if target <= 0 or not eligible:
        return {}
    minimum = 1 if target >= len(eligible) else 0
    allocation = largest_remainder_allocation(
        eligible,
        target,
        minimum_each=minimum,
    )
    return {
        key: min(int(quantity), int(eligible[key]) - 1)
        for key, quantity in allocation.items()
        if int(quantity) > 0
    }


def expand_peak_documents(
    documents: pd.DataFrame,
    voyage_id: str,
    target_total: int,
    voyage_position: int,
    forty_five_share: float,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    source = documents.copy(deep=True)
    source["_group_key"] = source.apply(_stable_group_key, axis=1)
    source_counts = Counter(source["_group_key"])
    targets = _group_target_counts(
        source_counts,
        target_total,
        voyage_position,
    )
    forty_five = _forty_five_counts(targets, forty_five_share)

    pieces: list[pd.DataFrame] = []
    profile: list[dict[str, object]] = []
    for group_position, key in enumerate(sorted(targets)):
        candidates = source.loc[
            source["_group_key"].map(lambda value: value == key)
        ].drop(
            columns="_group_key"
        )
        target_quantity = int(targets[key])
        offset = (23 * voyage_position + 13 * group_position) % len(candidates)
        positions = [
            (offset + position) % len(candidates)
            for position in range(target_quantity)
        ]
        expanded = candidates.iloc[positions].copy(deep=True)
        converted = int(forty_five.get(key, 0))
        if converted > 0:
            expanded.iloc[
                :converted,
                expanded.columns.get_loc("IYC_CSZ_CSIZECD"),
            ] = "45"
        pieces.append(expanded)
        profile.append(
            {
                "source_size": key[0],
                "height": key[1],
                "port": key[2],
                "source_quantity": int(source_counts[key]),
                "target_quantity": target_quantity,
                "converted_to_45": converted,
            }
        )

    out = pd.concat(pieces, ignore_index=True)
    out["IYC_EVOY_ID"] = int(voyage_id)
    for field in ("IYC_CNTRID", "IYC_CNTRNO"):
        if field in out:
            out[field] = unique_container_values(len(out), voyage_id, field)
    if len(out) != int(target_total):
        raise AssertionError("peak expansion did not conserve voyage demand")
    return out, profile


def _plan_size(value: object) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def rebuild_export_plan(
    large_plan: pd.DataFrame,
    adapter: InputAdapterGd,
    voyages: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, int]]]]:
    voyage_column = next(
        name
        for name in ("voy_id", "voyage_id", "VOY_ID")
        if name in large_plan.columns
    )
    normalized_voyages = large_plan[voyage_column].map(normalize_voyage_id)
    retained = large_plan.loc[~normalized_voyages.isin(voyages)].copy()
    pieces = [retained]
    distributions: dict[str, dict[str, dict[str, int]]] = {}

    for voyage_position, voyage_id in enumerate(voyages):
        source_rows = large_plan.loc[
            normalized_voyages.eq(voyage_id)
        ].copy(deep=True)
        # Retain upstream rows that describe only containers already in the
        # yard.  They do not become declared demand, but dropping them would
        # destroy the meaning of planned_qty in the generated input.
        static_rows = source_rows.loc[
            pd.to_numeric(source_rows["new_qty"], errors="coerce")
            .fillna(0)
            .le(0)
        ].copy(deep=True)
        if not static_rows.empty:
            pieces.append(static_rows)
        documents = adapter.vessel_containers[voyage_id]["doc_cntrs"]
        big_sizes = documents["IYC_CSZ_CSIZECD"].map(
            lambda value: "40"
            if normalize_container_size(value) == "45"
            else normalize_container_size(value)
        )
        size_demand = big_sizes.value_counts().to_dict()
        voyage_distribution: dict[str, dict[str, int]] = {}
        for size, demand in sorted(size_demand.items()):
            candidates = source_rows.loc[
                source_rows["size"].map(_plan_size).eq(size)
                & pd.to_numeric(
                    source_rows["new_qty"], errors="coerce"
                ).fillna(0).gt(0)
            ].copy(deep=True)
            if candidates.empty:
                raise ValueError(
                    f"voyage {voyage_id} has no positive {size}-ft plan row"
                )
            area_weights = (
                candidates.assign(
                    _weight=pd.to_numeric(
                        candidates["new_qty"], errors="raise"
                    )
                )
                .groupby("area_no", sort=True)["_weight"]
                .sum()
                .to_dict()
            )
            areas = sorted(area_weights, key=str)
            weights = {
                area: float(area_weights[area])
                * (
                    1.0
                    + 0.04
                    * (((voyage_position + 2) * (position + 1)) % 5 - 2)
                )
                for position, area in enumerate(areas)
            }
            allocation = largest_remainder_allocation(weights, int(demand))
            voyage_distribution[size] = {
                str(area): int(quantity)
                for area, quantity in allocation.items()
                if int(quantity) > 0
            }
            for area, quantity in allocation.items():
                if quantity <= 0:
                    continue
                row = candidates.loc[candidates["area_no"].eq(area)].iloc[
                    [0]
                ].copy(deep=True)
                raw_snapshot = pd.to_numeric(
                    row.iloc[0].get("snapshot_qty", 0),
                    errors="coerce",
                )
                snapshot = (
                    0 if pd.isna(raw_snapshot) else int(round(raw_snapshot))
                )
                row["new_qty"] = int(quantity)
                row["snapshot_qty"] = snapshot
                row["planned_qty"] = snapshot + int(quantity)
                pieces.append(row)
        distributions[voyage_id] = voyage_distribution

    rebuilt = pd.concat(pieces, ignore_index=True)
    return rebuilt, distributions


def build_case(
    base_input: Path,
    base_large_plan: Path,
    *,
    volume_scale: float = DEFAULT_VOLUME_SCALE,
    forty_five_share: float = DEFAULT_FORTY_FIVE_SHARE,
    voyage_copies: int = DEFAULT_VOYAGE_COPIES,
) -> tuple[InputAdapterGd, pd.DataFrame, dict]:
    if float(volume_scale) < 1.0:
        raise ValueError("volume_scale must be at least 1")
    if int(voyage_copies) < 2:
        raise ValueError("voyage_copies must be at least 2")
    if not 0.0 <= float(forty_five_share) < 0.25:
        raise ValueError("forty_five_share must be in [0, 0.25)")

    adapter, source_plan, source_manifest = build_many_groups_case(
        base_input,
        base_large_plan,
        copies=int(voyage_copies),
    )
    voyages = list(source_manifest["detailed_export_voyages"])
    source_totals = {
        voyage_id: len(adapter.vessel_containers[voyage_id]["doc_cntrs"])
        for voyage_id in voyages
    }
    voyage_targets = _voyage_target_totals(source_totals, volume_scale)
    profiles: dict[str, dict[str, object]] = {}
    for position, voyage_id in enumerate(voyages):
        expanded, profile = expand_peak_documents(
            adapter.vessel_containers[voyage_id]["doc_cntrs"],
            voyage_id,
            voyage_targets[voyage_id],
            position,
            forty_five_share,
        )
        adapter.vessel_containers[voyage_id]["doc_cntrs"] = expanded
        group_counts = Counter(
            zip(
                expanded["IYC_CSZ_CSIZECD"].map(normalize_container_size),
                expanded["IYC_CHEIGHTCD"].astype(str),
                expanded["IYC_POT_UNLDPORT"].astype(str),
            )
        )
        profiles[voyage_id] = {
            "source_boxes": source_totals[voyage_id],
            "target_boxes": len(expanded),
            "target_to_source_ratio": round(
                len(expanded) / source_totals[voyage_id], 6
            ),
            "group_count": len(group_counts),
            "size_totals": dict(
                sorted(
                    Counter(
                        expanded["IYC_CSZ_CSIZECD"].map(
                            normalize_container_size
                        )
                    ).items()
                )
            ),
            "group_profile": profile,
        }

    large_plan, plan_distribution = rebuild_export_plan(
        source_plan,
        adapter,
        voyages,
    )
    total_boxes = sum(voyage_targets.values())
    total_groups = sum(
        int(profile["group_count"]) for profile in profiles.values()
    )
    size_totals = Counter()
    for profile in profiles.values():
        size_totals.update(profile["size_totals"])
    export_slot_demand = sum(
        int(quantity) * (2 if size in {"40", "45"} else 1)
        for size, quantity in size_totals.items()
    )
    manifest = {
        "case": "natural_conflict_peak",
        "base_input": source_manifest["base_input"],
        "base_large_plan": source_manifest["base_large_plan"],
        "generation_policy": (
            "diversified peak declared demand on the unchanged real yard "
            "snapshot; no copied voyage profiles and no artificial closures"
        ),
        "yard_snapshot_policy": "unchanged_from_base_case",
        "import_commitment_policy": "unchanged_from_base_case",
        "conflict_mechanism": (
            "concurrent voyage/port row ownership, bay-level height separation, "
            "large-container paired footprints, and scarce edge footprints for "
            "a small 45-ft high-cube share"
        ),
        "volume_scale_requested": float(volume_scale),
        "diversified_voyage_copies": int(voyage_copies),
        "volume_scale_realized": round(
            total_boxes
            / source_manifest["declared_export_container_rows"],
            6,
        ),
        "forty_five_share_of_large_requested": float(
            forty_five_share
        ),
        "detailed_export_voyages": voyages,
        "detailed_export_voyage_count": len(voyages),
        "total_export_group_count": total_groups,
        "declared_export_container_rows": total_boxes,
        "export_size_totals": dict(sorted(size_totals.items())),
        "export_physical_slot_demand": int(export_slot_demand),
        "voyage_profiles": profiles,
        "large_plan_distribution": plan_distribution,
        "large_plan_row_count": int(len(large_plan)),
    }
    return adapter, large_plan, manifest


def main() -> None:
    args = parse_args()
    adapter, large_plan, manifest = build_case(
        args.base_input.resolve(),
        args.base_large_plan.resolve(),
        volume_scale=args.volume_scale,
        forty_five_share=args.forty_five_share,
        voyage_copies=args.voyage_copies,
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
