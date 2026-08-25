from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd
from adapters.planning_input import (
    normalize_code,
    normalize_container_size as normalize_size,
    normalize_voyage,
)

MODEL_SCHEMA_VERSION = "integrated_zone_v4"


@dataclass(frozen=True)
class ScenarioSpec:
    case_id: str
    seed: int
    export_voyage_count: int
    groups_per_voyage: int
    boxes_per_voyage: int
    import_fraction: float
    voyage_volume_spread: float = 0.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScenarioSpec":
        legacy_forecast_fields = {
            "forecast_extra_ratio",
            "voyage_forecast_spread",
        } & set(payload)
        if legacy_forecast_fields:
            raise ValueError(
                "Known-box scenario recipes must not contain forecast fields: "
                f"{sorted(legacy_forecast_fields)}"
            )
        spec = cls(
            case_id=str(payload["case_id"]),
            seed=int(payload["seed"]),
            export_voyage_count=int(payload["export_voyage_count"]),
            groups_per_voyage=int(payload["groups_per_voyage"]),
            boxes_per_voyage=int(payload["boxes_per_voyage"]),
            import_fraction=float(payload["import_fraction"]),
            voyage_volume_spread=float(
                payload.get("voyage_volume_spread", 0.0)
            ),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.case_id):
            raise ValueError(f"Invalid case_id: {self.case_id!r}")
        if self.export_voyage_count <= 0:
            raise ValueError("export_voyage_count must be positive.")
        if self.groups_per_voyage <= 0:
            raise ValueError("groups_per_voyage must be positive.")
        if self.boxes_per_voyage < self.groups_per_voyage:
            raise ValueError("boxes_per_voyage must be at least groups_per_voyage.")
        if not 0 <= self.import_fraction <= 1:
            raise ValueError("import_fraction must lie in [0, 1].")
        if not 0 <= self.voyage_volume_spread <= 0.5:
            raise ValueError("voyage_volume_spread must lie in [0, 0.5].")


def materialize_scenario(
    base: InputAdapterGd,
    spec: ScenarioSpec,
) -> tuple[InputAdapterGd, dict[str, Any]]:
    """Create one deterministic case while sharing the immutable yard snapshot."""

    spec.validate()
    rng = random.Random(spec.seed)
    export_sources = _export_sources_with_docs(base)
    export_targets = _build_export_targets(
        base,
        export_sources,
        spec.export_voyage_count,
    )
    selected_exports = [target["voyage"] for target in export_targets]
    group_palette = _build_group_palette(base, export_sources)
    if spec.groups_per_voyage > len(group_palette):
        raise ValueError(
            f"Case {spec.case_id} requests {spec.groups_per_voyage} groups per voyage, "
            f"but the base palette supports {len(group_palette)}."
        )

    generated = _shallow_adapter_copy(base)
    clone_berth_rows = _build_clone_berth_rows(base, export_targets)
    if clone_berth_rows:
        generated.vessel_berth_info = pd.concat(
            [base.vessel_berth_info, *clone_berth_rows],
            ignore_index=True,
        )
    vessel_containers: dict[str, dict[str, Any]] = {}
    voyage_profiles: dict[str, Any] = {}
    volume_weights = _heterogeneity_weights(
        len(export_targets),
        spec.voyage_volume_spread,
        rng,
    )
    voyage_box_totals = _allocate_weighted_total(
        volume_weights,
        spec.boxes_per_voyage * len(export_targets),
        minimum_each=spec.groups_per_voyage,
    )
    for target, voyage_boxes in zip(
        export_targets,
        voyage_box_totals,
    ):
        voyage = str(target["voyage"])
        source_voyage = str(target["source_voyage"])
        content = dict(base.vessel_containers[source_voyage])
        group_profile = _select_group_profile(group_palette, spec, rng)
        group_profile = _resize_group_profile(
            group_profile,
            voyage_boxes,
            rng,
        )
        document = _build_export_document(
            base,
            voyage,
            base.vessel_containers[source_voyage]["doc_cntrs"],
            group_profile,
            spec,
            rng,
        )
        snapshot_counts = _export_snapshot_size_counts(base.bay_slots_detail, voyage)
        document_counts = Counter(document["IYC_CSZ_CSIZECD"].map(normalize_size))
        content["doc_cntrs"] = document
        content.pop("predict_cntrs", None)
        content.pop("cntr_volume", None)
        content["type"] = "E"
        vessel_containers[voyage] = content
        voyage_profiles[voyage] = {
            "source_voyage": source_voyage,
            "copy_index": int(target["copy_index"]),
            "synthetic": bool(target["copy_index"]),
            "estimated_berth": str(target["berth"]),
            "declared_rows": int(len(document)),
            "group_count": len(group_profile),
            "size_totals": {size: int(document_counts[size]) for size in ("20", "40")},
            "snapshot_size_totals": {size: int(snapshot_counts[size]) for size in ("20", "40")},
            "group_profile": group_profile,
        }

    selected_imports, import_rows = _sample_import_documents(base, spec.import_fraction, rng)
    for voyage, document in selected_imports.items():
        content = dict(base.vessel_containers[voyage])
        content["doc_cntrs"] = document
        content["type"] = "I"
        vessel_containers[voyage] = content

    generated.take_over_vessel = {
        "E": list(selected_exports),
        "I": list(selected_imports),
    }
    generated.vessel_containers = vessel_containers
    manifest = {
        "schema_version": 3,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "case_id": spec.case_id,
        "generation_spec": asdict(spec),
        "input_contract": "one_materialized_InputAdapterGd_for_integrated_paper_model",
        "yard_snapshot_policy": "unchanged_from_base_input",
        "voyage_generation_policy": (
            "two_real_source_families_with_deterministic_volume_group_"
            "and_berth_diversification"
        ),
        "demand_policy": "yard_snapshot_plus_declared_documents_only",
        "prediction_policy": "not_generated_or_used",
        "tops_policy": "not_in_model_or_materialized_adapter_schema",
        "selected_export_voyages": list(selected_exports),
        "selected_import_voyages": list(selected_imports),
        "declared_export_rows": sum(profile["declared_rows"] for profile in voyage_profiles.values()),
        "declared_import_rows": import_rows,
        "export_group_count": sum(profile["group_count"] for profile in voyage_profiles.values()),
        "voyage_profiles": voyage_profiles,
    }
    _validate_materialized_scenario(generated, manifest)
    return generated, manifest


def _shallow_adapter_copy(base: InputAdapterGd) -> InputAdapterGd:
    generated = InputAdapterGd()
    generated.bay_slots_detail = base.bay_slots_detail
    generated.area_function_info = base.area_function_info
    generated.vessel_berth_info = base.vessel_berth_info
    generated.planning_time = base.planning_time
    generated.closed_area = set(base.closed_area)
    generated.berth_area_dist_matrix = base.berth_area_dist_matrix
    return generated


def _export_sources_with_docs(base: InputAdapterGd) -> list[str]:
    sources: list[str] = []
    for value in (base.take_over_vessel or {}).get("E", []):
        voyage = normalize_voyage(value)
        content = base.vessel_containers.get(voyage, {})
        document = content.get("doc_cntrs")
        if isinstance(document, pd.DataFrame) and not document.empty:
            sources.append(voyage)
    return sources


def _build_export_targets(
    base: InputAdapterGd,
    sources: Sequence[str],
    count: int,
) -> list[dict[str, Any]]:
    if not sources:
        raise ValueError("No export source voyage has declared containers.")
    complete_berths = _complete_distance_berths(base)
    existing_voyages = {
        normalize_voyage(value) for value in base.vessel_containers
    }
    targets: list[dict[str, Any]] = []
    for position in range(count):
        source_position = position % len(sources)
        copy_index = position // len(sources)
        source = str(sources[source_position])
        voyage = (
            source
            if copy_index == 0
            else _synthetic_voyage_id(source, copy_index)
        )
        if copy_index and voyage in existing_voyages:
            raise ValueError(f"Synthetic export voyage already exists: {voyage}")
        berth = (
            _source_voyage_berth(base, source)
            if copy_index == 0
            else _synthetic_berth(
                complete_berths,
                source_position,
                copy_index,
            )
        )
        targets.append(
            {
                "voyage": voyage,
                "source_voyage": source,
                "source_position": source_position,
                "copy_index": copy_index,
                "berth": berth,
            }
        )
    voyages = [str(target["voyage"]) for target in targets]
    if len(voyages) != len(set(voyages)):
        raise AssertionError("Generated export voyage IDs are not unique.")
    return targets


def _synthetic_voyage_id(source: str, copy_index: int) -> str:
    if source.isdigit():
        return str(int(source) + 100_000 * int(copy_index))
    return f"{source}C{int(copy_index)}"


def _complete_distance_berths(base: InputAdapterGd) -> list[str]:
    berths: list[str] = []
    for column in base.berth_area_dist_matrix.columns:
        berth = normalize_code(column)
        if not berth.startswith("B"):
            continue
        values = pd.to_numeric(
            base.berth_area_dist_matrix[column],
            errors="coerce",
        )
        if values.notna().all():
            berths.append(berth)
    if not berths:
        raise ValueError("No complete berth-distance column is available.")
    return sorted(
        berths,
        key=lambda value: (
            int(value[1:]) if value[1:].isdigit() else 10**9,
            value,
        ),
    )


def _source_voyage_berth(base: InputAdapterGd, voyage: str) -> str:
    frame = base.vessel_berth_info.copy()
    rows = frame[frame["VOY_ID"].map(normalize_voyage).eq(voyage)]
    if rows.empty:
        raise ValueError(f"Source export voyage has no berth row: {voyage}")
    row = rows.iloc[0]
    for column in ("VBT_BTH_ABTHNO", "VBT_BTH_PBTHNO"):
        berth = normalize_code(row.get(column))
        if berth:
            return berth if berth.startswith("B") else f"B{berth}"
    raise ValueError(f"Source export voyage has no usable berth: {voyage}")


def _synthetic_berth(
    berths: Sequence[str],
    source_position: int,
    copy_index: int,
) -> str:
    position = 1 + 3 * (copy_index - 1) + 2 * source_position
    return str(berths[position % len(berths)])


def _build_clone_berth_rows(
    base: InputAdapterGd,
    targets: Sequence[Mapping[str, Any]],
) -> list[pd.DataFrame]:
    normalized = base.vessel_berth_info["VOY_ID"].map(normalize_voyage)
    rows: list[pd.DataFrame] = []
    for target in targets:
        if not int(target["copy_index"]):
            continue
        source = str(target["source_voyage"])
        clone = base.vessel_berth_info.loc[
            normalized.eq(source)
        ].copy(deep=True)
        if clone.empty:
            raise ValueError(f"Source export voyage has no berth row: {source}")
        voyage = str(target["voyage"])
        clone["VOY_ID"] = int(voyage) if voyage.isdigit() else voyage
        clone["VOY_IEFG"] = "E"
        clone["VBT_BTH_PBTHNO"] = str(target["berth"]).removeprefix("B")
        if "VBT_BTH_ABTHNO" in clone:
            clone["VBT_BTH_ABTHNO"] = None
        for column in ("VSL_CNNAME", "VSL_ENNAME"):
            if column in clone:
                clone[column] = (
                    clone[column].fillna("").astype(str)
                    + f" SYN {voyage}"
                )
        rows.append(clone)
    return rows


def _heterogeneity_weights(
    count: int,
    spread: float,
    rng: random.Random,
) -> list[float]:
    if count <= 0:
        return []
    if count == 1 or spread <= 0:
        return [1.0] * count
    centered = [
        2.0 * position / (count - 1) - 1.0
        for position in range(count)
    ]
    rng.shuffle(centered)
    return [1.0 + float(spread) * value for value in centered]


def _allocate_weighted_total(
    weights: Sequence[float],
    total: int,
    *,
    minimum_each: int,
) -> list[int]:
    if total < minimum_each * len(weights):
        raise ValueError("Total is smaller than the per-voyage minimum.")
    remaining = total - minimum_each * len(weights)
    weight_total = sum(float(value) for value in weights)
    raw = [remaining * float(value) / weight_total for value in weights]
    allocation = [minimum_each + int(value) for value in raw]
    residue = total - sum(allocation)
    order = sorted(
        range(len(weights)),
        key=lambda index: (raw[index] - int(raw[index]), -index),
        reverse=True,
    )
    for index in order[:residue]:
        allocation[index] += 1
    return allocation


def _resize_group_profile(
    profile: Sequence[Mapping[str, Any]],
    total: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    del rng
    if sum(int(group["quantity"]) for group in profile) == int(total):
        return [dict(group) for group in profile]
    weights = [max(1, int(group["quantity"])) for group in profile]
    quantities = _allocate_weighted_total(
        weights,
        int(total),
        minimum_each=1,
    )
    resized = [dict(group) for group in profile]
    for group, quantity in zip(resized, quantities):
        group["quantity"] = int(quantity)
    return resized


def _build_group_palette(
    base: InputAdapterGd,
    export_sources: Sequence[str],
) -> list[tuple[str, str, str]]:
    ports: set[str] = set()
    heights_by_size: dict[str, set[str]] = {"20": set(), "40": set()}
    for voyage in export_sources:
        document = base.vessel_containers[voyage]["doc_cntrs"]
        for row in document.to_dict("records"):
            size = normalize_size(row.get("IYC_CSZ_CSIZECD"))
            height = normalize_code(row.get("IYC_CHEIGHTCD"))
            port = normalize_code(row.get("IYC_POT_UNLDPORT"))
            if size in heights_by_size and height:
                heights_by_size[size].add(height)
            if port:
                ports.add(port)
    if not ports or any(not heights_by_size[size] for size in ("20", "40")):
        raise ValueError("The base export documents do not provide a complete size-height-port palette.")
    return [
        (size, height, port)
        for size in ("20", "40")
        for height in sorted(heights_by_size[size])
        for port in sorted(ports)
    ]


def _select_group_profile(
    palette: Sequence[tuple[str, str, str]],
    spec: ScenarioSpec,
    rng: random.Random,
) -> list[dict[str, Any]]:
    by_size = {
        size: [item for item in palette if item[0] == size]
        for size in ("20", "40")
    }
    rng.shuffle(by_size["20"])
    rng.shuffle(by_size["40"])
    ordered: list[tuple[str, str, str]] = []
    while by_size["20"] or by_size["40"]:
        for size in ("20", "40"):
            if by_size[size]:
                ordered.append(by_size[size].pop())
    selected = ordered[: spec.groups_per_voyage]
    quantities = _allocate_positive(spec.boxes_per_voyage, len(selected), rng)
    return [
        {"size": size, "height": height, "port": port, "quantity": quantity}
        for (size, height, port), quantity in zip(selected, quantities)
    ]


def _allocate_positive(total: int, count: int, rng: random.Random) -> list[int]:
    base = [1] * count
    remaining = total - count
    weights = [rng.uniform(0.75, 1.25) for _ in range(count)]
    raw = [remaining * weight / sum(weights) for weight in weights]
    floors = [int(value) for value in raw]
    for index, value in enumerate(floors):
        base[index] += value
    remainder = total - sum(base)
    order = sorted(range(count), key=lambda index: raw[index] - floors[index], reverse=True)
    for index in order[:remainder]:
        base[index] += 1
    return base


def _build_export_document(
    base: InputAdapterGd,
    voyage: str,
    source_document: pd.DataFrame,
    profile: Sequence[Mapping[str, Any]],
    spec: ScenarioSpec,
    rng: random.Random,
) -> pd.DataFrame:
    templates = pd.concat(
        [
            content["doc_cntrs"]
            for content in base.vessel_containers.values()
            if isinstance(content, Mapping)
            and isinstance(content.get("doc_cntrs"), pd.DataFrame)
            and not content["doc_cntrs"].empty
            and normalize_code(content.get("type")) != "I"
        ],
        ignore_index=True,
    )
    records: list[dict[str, Any]] = []
    sequence = 0
    for group in profile:
        size = str(group["size"])
        height = str(group["height"])
        port = str(group["port"])
        compatible = templates[
            templates["IYC_CSZ_CSIZECD"].map(normalize_size).eq(size)
            & templates["IYC_CHEIGHTCD"].map(normalize_code).eq(height)
        ]
        template_rows = compatible if not compatible.empty else source_document
        template_records = template_rows.to_dict("records")
        for _ in range(int(group["quantity"])):
            sequence += 1
            row = dict(rng.choice(template_records))
            container_id = f"SYN_{spec.case_id}_{voyage}_{sequence:06d}"
            row.update(
                {
                    "IYC_CNTRID": container_id,
                    "IYC_CNTRNO": container_id,
                    "IYC_EVOY_ID": voyage,
                    "IYC_IVOY_ID": "",
                    "IYC_CSZ_CSIZECD": size,
                    "IYC_STS_CSTATUSCD": "OF",
                    "IYC_CHEIGHTCD": height,
                    "IYC_POT_UNLDPORT": port,
                    "IYC_EVESVOYAGE": f"SYN_{voyage}",
                    "REF_BILLNO": f"SYN_{spec.case_id}_{voyage}",
                }
            )
            records.append(row)
    return pd.DataFrame(records, columns=source_document.columns)


def _export_snapshot_size_counts(snapshot: pd.DataFrame, voyage: str) -> Counter[str]:
    occupied = snapshot[pd.to_numeric(snapshot["HAS_CONTAINER"], errors="coerce").fillna(0).eq(1)].copy()
    occupied = occupied[
        occupied["IYC_EVOY_ID"].map(normalize_voyage).eq(voyage)
        & occupied["IYC_IVOY_ID"].map(normalize_voyage).eq("")
    ].copy()
    occupied["cntr_id"] = occupied["IYC_CNTRID"].map(normalize_code)
    occupied = occupied[occupied["cntr_id"].ne("") & occupied["cntr_id"].ne("-1")]
    unique = occupied.sort_values("cntr_id").drop_duplicates("cntr_id", keep="first")
    return Counter(unique["IYC_CSZ_CSIZECD"].map(normalize_size))


def _sample_import_documents(
    base: InputAdapterGd,
    fraction: float,
    rng: random.Random,
) -> tuple[dict[str, pd.DataFrame], int]:
    selected: dict[str, pd.DataFrame] = {}
    total_rows = 0
    if fraction <= 0:
        return selected, total_rows
    for value in (base.take_over_vessel or {}).get("I", []):
        voyage = normalize_voyage(value)
        document = base.vessel_containers.get(voyage, {}).get("doc_cntrs")
        if not isinstance(document, pd.DataFrame) or document.empty:
            continue
        count = max(1, min(len(document), int(round(len(document) * fraction))))
        indices = list(range(len(document)))
        rng.shuffle(indices)
        sample = document.iloc[sorted(indices[:count])].copy().reset_index(drop=True)
        selected[voyage] = sample
        total_rows += len(sample)
    return selected, total_rows


def _validate_materialized_scenario(adapter: InputAdapterGd, manifest: Mapping[str, Any]) -> None:
    if hasattr(adapter, "tops_plan"):
        raise AssertionError("Materialized scenarios must not expose TOPS input.")
    export_ids: list[str] = []
    export_rows = 0
    for voyage in adapter.take_over_vessel["E"]:
        document = adapter.vessel_containers[voyage]["doc_cntrs"]
        export_rows += len(document)
        export_ids.extend(document["IYC_CNTRID"].map(normalize_code))
        if not document["IYC_EVOY_ID"].map(normalize_voyage).eq(voyage).all():
            raise AssertionError(f"Generated export document has mismatched voyage IDs: {voyage}")
    if len(export_ids) != len(set(export_ids)):
        raise AssertionError("Generated export container IDs are not unique.")
    if export_rows != int(manifest["declared_export_rows"]):
        raise AssertionError("Generated export row count differs from the manifest.")


def load_suite(path: str | Path) -> tuple[Path, list[ScenarioSpec]]:
    import hashlib
    import json

    suite_path = Path(path).resolve()
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 3:
        raise ValueError(
            "Preexperiment suite must use schema_version 3."
        )
    if str(payload.get("model_schema_version", "")) != MODEL_SCHEMA_VERSION:
        raise ValueError(
            "Preexperiment suite model schema is incompatible: "
            f"{payload.get('model_schema_version')!r}"
        )
    base_input = (suite_path.parent / payload["base_input"]).resolve()
    expected_digest = str(payload.get("base_input_sha256", "")).lower()
    if expected_digest:
        digest = hashlib.sha256()
        with base_input.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_digest = digest.hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(
                "Base input SHA-256 does not match pilot_suite.json: "
                f"actual={actual_digest}, expected={expected_digest}"
            )
    specs = [ScenarioSpec.from_dict(item) for item in payload["cases"]]
    case_ids = [spec.case_id for spec in specs]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Pilot suite contains duplicate case_id values.")
    return base_input, specs
