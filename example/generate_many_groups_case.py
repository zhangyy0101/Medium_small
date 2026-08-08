from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.input_adapter_gd import InputAdapterGd  # noqa: E402
from adapters.planning_input import normalize_container_size  # noqa: E402
from example.generate_diverse_voyages_case import (
    DEFAULT_BASE_INPUT,
    DEFAULT_BASE_PLAN,
    GROUP_FIELDS,
    build_case as build_diverse_case,
    largest_remainder_allocation,
    write_case,
)  # noqa: E402


DEFAULT_OUTPUT = ROOT / "example" / "many_groups_6v_12g"
PORTS = ("DEHAM", "JPHKT", "JPNGO", "NLRTM")
HEIGHTS_BY_SIZE = {
    "20": ("PQ",),
    "40": ("HQ", "PQ"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a six-voyage case with twelve realistic size-height-port "
            "groups per detailed export voyage."
        )
    )
    parser.add_argument("--base-input", type=Path, default=DEFAULT_BASE_INPUT)
    parser.add_argument(
        "--base-large-plan", type=Path, default=DEFAULT_BASE_PLAN
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--copies",
        type=int,
        default=3,
        help="Total diversified scale, with two source voyage families.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated input, plan, and manifest.",
    )
    return parser.parse_args()


def _group_weights(
    combinations: tuple[tuple[str, str], ...],
    voyage_position: int,
) -> dict[tuple[str, str], float]:
    """Deterministic, non-identical shares without changing voyage totals."""
    return {
        combination: 1.0
        + 0.06
        * (((voyage_position + 2) * (position + 3)) % 9 - 4)
        for position, combination in enumerate(combinations)
    }


def expand_voyage_groups(
    documents: pd.DataFrame,
    voyage_id: str,
    voyage_position: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Recombine observed attribute values into twelve positive groups."""
    missing = [field for field in GROUP_FIELDS if field not in documents]
    if missing:
        raise KeyError(f"documents are missing grouping fields: {missing}")

    source = documents.copy(deep=True)
    normalized_sizes = source["IYC_CSZ_CSIZECD"].map(
        normalize_container_size
    )
    unsupported = sorted(set(normalized_sizes) - set(HEIGHTS_BY_SIZE))
    if unsupported:
        raise ValueError(
            f"unsupported export sizes for many-group case: {unsupported}"
        )

    pieces: list[pd.DataFrame] = []
    profile: list[dict[str, object]] = []
    for size, heights in HEIGHTS_BY_SIZE.items():
        size_rows = source.loc[normalized_sizes.eq(size)].copy(deep=True)
        combinations = tuple(
            (height, port) for height in heights for port in PORTS
        )
        if len(size_rows) < len(combinations):
            raise ValueError(
                f"voyage {voyage_id} has too few {size}-ft boxes to create "
                f"{len(combinations)} positive groups"
            )
        allocation = largest_remainder_allocation(
            _group_weights(combinations, voyage_position),
            len(size_rows),
            minimum_each=1,
        )
        offset = (17 * voyage_position + (3 if size == "40" else 0)) % len(
            size_rows
        )
        ordered = pd.concat(
            [size_rows.iloc[offset:], size_rows.iloc[:offset]],
            ignore_index=True,
        )
        cursor = 0
        for height, port in combinations:
            quantity = int(allocation[(height, port)])
            group_rows = ordered.iloc[cursor : cursor + quantity].copy(
                deep=True
            )
            cursor += quantity
            group_rows["IYC_CHEIGHTCD"] = height
            group_rows["IYC_POT_UNLDPORT"] = port
            pieces.append(group_rows)
            profile.append(
                {
                    "size": size,
                    "height": height,
                    "port": port,
                    "quantity": quantity,
                }
            )
        if cursor != len(size_rows):
            raise AssertionError("group expansion did not conserve size demand")

    expanded = pd.concat(pieces, ignore_index=True)
    if len(expanded) != len(source):
        raise AssertionError("group expansion did not conserve voyage demand")
    group_count = expanded.groupby(list(GROUP_FIELDS), dropna=False).ngroups
    if group_count != 12:
        raise AssertionError(
            f"voyage {voyage_id} has {group_count} groups instead of 12"
        )
    return expanded, profile


def build_case(
    base_input: Path,
    base_large_plan: Path,
    copies: int = 3,
) -> tuple[InputAdapterGd, pd.DataFrame, dict]:
    adapter, large_plan, diverse_manifest = build_diverse_case(
        base_input,
        base_large_plan,
        copies=copies,
    )
    detailed_voyages = [
        voyage
        for source in diverse_manifest["source_export_voyages"]
        for voyage in diverse_manifest["export_voyage_mapping"][source]
    ]
    profiles: dict[str, dict[str, object]] = {}
    total_groups = 0
    for position, voyage_id in enumerate(detailed_voyages):
        content = adapter.vessel_containers[voyage_id]
        source_size_totals = Counter(
            content["doc_cntrs"]["IYC_CSZ_CSIZECD"].map(
                normalize_container_size
            )
        )
        expanded, profile = expand_voyage_groups(
            content["doc_cntrs"],
            voyage_id,
            position,
        )
        content["doc_cntrs"] = expanded
        counts = Counter(
            zip(
                expanded["IYC_CSZ_CSIZECD"].map(normalize_container_size),
                expanded["IYC_CHEIGHTCD"].astype(str),
                expanded["IYC_POT_UNLDPORT"].astype(str),
            )
        )
        total_groups += len(counts)
        profiles[voyage_id] = {
            "declared_rows": len(expanded),
            "group_count": len(counts),
            "source_size_totals": dict(sorted(source_size_totals.items())),
            "expanded_size_totals": dict(
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

    manifest = {
        "case": f"many_groups_{len(detailed_voyages)}v_12g",
        "base_input": diverse_manifest["base_input"],
        "base_large_plan": diverse_manifest["base_large_plan"],
        "generation_policy": (
            "six diversified voyages with deterministic recombination of "
            "globally observed size-height-port values"
        ),
        "yard_snapshot_policy": "unchanged_from_base_case",
        "import_commitment_policy": "unchanged_from_base_case",
        "large_plan_policy": (
            "unchanged from the diversified six-voyage case; voyage and "
            "size totals are conserved"
        ),
        "attribute_palette": {
            "ports": list(PORTS),
            "heights_by_size": {
                size: list(values)
                for size, values in HEIGHTS_BY_SIZE.items()
            },
        },
        "source_export_voyages": diverse_manifest[
            "source_export_voyages"
        ],
        "export_voyage_mapping": diverse_manifest[
            "export_voyage_mapping"
        ],
        "detailed_export_voyages": detailed_voyages,
        "detailed_export_voyage_count": len(detailed_voyages),
        "groups_per_voyage": 12,
        "total_export_group_count": total_groups,
        "declared_export_container_rows": sum(
            profile["declared_rows"] for profile in profiles.values()
        ),
        "voyage_profiles": profiles,
        "large_plan_row_count": int(len(large_plan)),
    }
    return adapter, large_plan, manifest


def main() -> None:
    args = parse_args()
    adapter, large_plan, manifest = build_case(
        args.base_input.resolve(),
        args.base_large_plan.resolve(),
        copies=args.copies,
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
