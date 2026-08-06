from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.input_adapter_gd import (
    InputAdapterGd,
    SafeJSONEncoder,
    normalize_voyage_id,
)


DEFAULT_BASE_INPUT = ROOT / "example" / "input_data.json"
DEFAULT_BASE_PLAN = ROOT / "example" / "large_plan.csv"
DEFAULT_OUTPUT = ROOT / "example" / "more_voyages_3x"
SOURCE_EXPORT_VOYAGES = ("390121", "390131")


def portable_source_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone the detailed export voyages while retaining the original "
            "yard snapshot and import-capacity commitments."
        )
    )
    parser.add_argument("--base-input", type=Path, default=DEFAULT_BASE_INPUT)
    parser.add_argument("--base-large-plan", type=Path, default=DEFAULT_BASE_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--copies",
        type=int,
        default=3,
        help="Total copies of each source export voyage, including the original.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated input and large-plan file.",
    )
    return parser.parse_args()


def synthetic_voyage_id(source: str, copy_index: int) -> str:
    return str(int(source) + 100_000 * copy_index)


def unique_container_values(
    values: pd.Series,
    target_voyage: str,
    field: str,
) -> pd.Series:
    prefix = "ID" if field == "IYC_CNTRID" else "NO"
    return pd.Series(
        [
            f"S{target_voyage}_{prefix}_{position:06d}"
            for position in range(1, len(values) + 1)
        ],
        index=values.index,
        dtype="string",
    )


def clone_export_content(
    content: dict,
    target_voyage: str,
) -> dict:
    cloned = copy.deepcopy(content)
    documents = content.get("doc_cntrs")
    if not isinstance(documents, pd.DataFrame) or documents.empty:
        raise ValueError("source export voyage has no declared-container rows")
    documents = documents.copy(deep=True)
    documents["IYC_EVOY_ID"] = int(target_voyage)
    if "IYC_EVESVOYAGE" in documents:
        documents["IYC_EVESVOYAGE"] = (
            documents["IYC_EVESVOYAGE"].astype("string")
            + f"_SYN_{target_voyage}"
        )
    for field in ("IYC_CNTRID", "IYC_CNTRNO"):
        if field in documents:
            documents[field] = unique_container_values(
                documents[field], target_voyage, field
            )
    cloned["doc_cntrs"] = documents
    cloned["type"] = "E"
    return cloned


def clone_berth_rows(
    frame: pd.DataFrame,
    source_voyage: str,
    target_voyage: str,
) -> pd.DataFrame:
    normalized = frame["VOY_ID"].map(normalize_voyage_id)
    rows = frame.loc[normalized.eq(source_voyage)].copy(deep=True)
    if rows.empty:
        raise ValueError(f"source export voyage has no berth row: {source_voyage}")
    rows["VOY_ID"] = int(target_voyage)
    for field in ("VSL_CNNAME", "VSL_ENNAME"):
        if field in rows:
            rows[field] = rows[field].astype("string") + f" SYN {target_voyage}"
    return rows


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
    cloned_plan_rows: list[pd.DataFrame] = []
    cloned_berth_rows: list[pd.DataFrame] = []
    mapping: dict[str, list[str]] = {
        source: [source] for source in SOURCE_EXPORT_VOYAGES
    }
    plan_voyages = large_plan[voyage_column].map(normalize_voyage_id)

    for copy_index in range(1, copies):
        for source_voyage in SOURCE_EXPORT_VOYAGES:
            target_voyage = synthetic_voyage_id(source_voyage, copy_index)
            if target_voyage in adapter.vessel_containers:
                raise ValueError(f"synthetic voyage already exists: {target_voyage}")
            source_content = adapter.vessel_containers.get(source_voyage)
            if not isinstance(source_content, dict):
                raise ValueError(f"source voyage is missing: {source_voyage}")
            adapter.vessel_containers[target_voyage] = clone_export_content(
                source_content, target_voyage
            )
            export_takeover.append(target_voyage)
            mapping[source_voyage].append(target_voyage)
            cloned_berth_rows.append(
                clone_berth_rows(
                    adapter.vessel_berth_info,
                    source_voyage,
                    target_voyage,
                )
            )
            source_plan = large_plan.loc[
                plan_voyages.eq(source_voyage)
            ].copy(deep=True)
            if source_plan.empty:
                raise ValueError(
                    f"source export voyage has no large-plan row: {source_voyage}"
                )
            source_plan[voyage_column] = int(target_voyage)
            cloned_plan_rows.append(source_plan)

    adapter.take_over_vessel["E"] = list(dict.fromkeys(export_takeover))
    adapter.vessel_berth_info = pd.concat(
        [adapter.vessel_berth_info, *cloned_berth_rows],
        ignore_index=True,
    )
    expanded_plan = pd.concat(
        [large_plan, *cloned_plan_rows], ignore_index=True
    )

    declared_rows = 0
    for voyages in mapping.values():
        for voyage in voyages:
            documents = adapter.vessel_containers[voyage]["doc_cntrs"]
            declared_rows += int(len(documents))
    manifest = {
        "case": f"more_voyages_{copies}x",
        "base_input": portable_source_path(base_input),
        "base_large_plan": portable_source_path(base_large_plan),
        "yard_snapshot_policy": "unchanged_from_base_case",
        "import_commitment_policy": "unchanged_from_base_case",
        "source_export_voyages": list(SOURCE_EXPORT_VOYAGES),
        "export_voyage_mapping": mapping,
        "detailed_export_voyage_count": sum(len(values) for values in mapping.values()),
        "declared_export_container_rows": declared_rows,
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
