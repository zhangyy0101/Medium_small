"""Restricted integer master and validated incumbent recovery for V6."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Mapping, Sequence

from .models import ProblemData
from .row_aware_zones import (
    RowAwareZone,
    build_v6_row_aware_bay_atoms,
)
from .v6_column_generation import (
    V6ProjectedRestrictedMaster,
    V6RootCgConfig,
    V6RootCgResult,
    V6RootColumnGeneration,
)
from .v6_model import (
    V6_MODEL_SCHEMA_VERSION,
    V6ModelEvaluator,
    V6ObjectiveConfig,
    V6PeakUtilizationPolicy,
)


@dataclass(frozen=True)
class V6RestrictedIntegerConfig:
    """Execution controls for the V6 restricted integer master."""

    time_limit: float = 60.0
    mip_gap: float = 0.0
    solver_threads: int = 1
    solver_seed: int = 0
    verbose: bool = False
    objective: V6ObjectiveConfig = field(default_factory=V6ObjectiveConfig)

    def validate(self) -> None:
        if not math.isfinite(float(self.time_limit)) or float(
            self.time_limit
        ) <= 0.0:
            raise ValueError("V6 restricted-integer time limit must be positive")
        if not math.isfinite(float(self.mip_gap)) or not 0.0 <= float(
            self.mip_gap
        ) <= 1.0:
            raise ValueError("V6 restricted-integer MIP gap must lie in [0, 1]")
        if int(self.solver_threads) < 0:
            raise ValueError("V6 solver_threads must be nonnegative")
        self.objective.validate()


@dataclass(frozen=True)
class V6RestrictedIntegerResult:
    """One evaluator-validated V6 incumbent over a declared zone pool."""

    selected_zone_ids: tuple[int, ...]
    zone_bay_flow: Mapping[tuple[int, str], int]
    import_reservation: Mapping[tuple[str, str, str], int]
    peak_policy: V6PeakUtilizationPolicy
    certificate: Mapping[str, object]
    diagnostics: Mapping[str, object]
    zones: tuple[RowAwareZone, ...]

    @property
    def objective(self) -> float:
        return float(self.certificate["objective"])

    def as_dict(self) -> dict[str, object]:
        zones_by_id = {zone.zone_id: zone for zone in self.zones}
        return {
            "model_schema_version": V6_MODEL_SCHEMA_VERSION,
            "result_role": "validated_restricted_integer_incumbent",
            "peak_policy": self.peak_policy.as_dict(),
            "selected_zones": [
                {
                    "zone_id": zone_id,
                    "group_id": zones_by_id[zone_id].group_id,
                    "area_no": zones_by_id[zone_id].area_no,
                    "anchor_bay_keys": list(
                        zones_by_id[zone_id].anchor_bay_keys
                    ),
                    "rows_by_anchor_bay": [
                        {"bay_key": bay_key, "row_nos": list(row_nos)}
                        for bay_key, row_nos in zones_by_id[
                            zone_id
                        ].rows_by_anchor_bay
                    ],
                    "flow_by_anchor_bay": {
                        bay_key: int(
                            self.zone_bay_flow.get((zone_id, bay_key), 0)
                        )
                        for bay_key in zones_by_id[zone_id].anchor_bay_keys
                    },
                }
                for zone_id in self.selected_zone_ids
            ],
            "anonymous_import_reservation": [
                {
                    "flow": flow,
                    "size": size,
                    "bay_key": bay_key,
                    "reserved_boxes": int(quantity),
                }
                for (flow, size, bay_key), quantity in sorted(
                    self.import_reservation.items()
                )
                if int(quantity) > 0
            ],
            "certificate": dict(self.certificate),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class V6RestrictedIntegerWarmStart:
    """A validated zone solution mapped independently of transient zone IDs."""

    selected_zone_signatures: tuple[tuple[int, ...], ...]
    group_bay_flow: Mapping[tuple[str, str], int]
    import_reservation: Mapping[tuple[str, str, str], int]
    source_objective: float
    source: str = "v6_compact_primal_coverage"

    @classmethod
    def from_zone_solution(
        cls,
        zones: Sequence[RowAwareZone],
        selected_zone_ids: Sequence[int],
        zone_bay_flow: Mapping[tuple[int, str], int],
        import_reservation: Mapping[tuple[str, str, str], int],
        source_objective: float,
        *,
        source: str = "v6_compact_primal_coverage",
    ) -> "V6RestrictedIntegerWarmStart":
        zones_by_id = {int(zone.zone_id): zone for zone in zones}
        selected = tuple(sorted(int(value) for value in selected_zone_ids))
        missing = set(selected) - set(zones_by_id)
        if missing:
            raise ValueError(
                "V6 integer warm start references unknown source zones: "
                f"{sorted(missing)}"
            )
        flow: dict[tuple[str, str], int] = {}
        for (zone_id, bay_key), raw_quantity in zone_bay_flow.items():
            quantity = int(raw_quantity)
            if quantity <= 0:
                continue
            zone = zones_by_id.get(int(zone_id))
            if zone is None or int(zone_id) not in selected:
                raise ValueError(
                    "V6 integer warm-start flow references an unselected "
                    f"source zone: {zone_id}"
                )
            key = (str(zone.group_id), str(bay_key))
            flow[key] = flow.get(key, 0) + quantity
        return cls(
            selected_zone_signatures=tuple(
                tuple(zones_by_id[zone_id].candidate_indices)
                for zone_id in selected
            ),
            group_bay_flow=flow,
            import_reservation={
                tuple(map(str, key)): int(quantity)
                for key, quantity in import_reservation.items()
                if int(quantity) > 0
            },
            source_objective=float(source_objective),
            source=str(source),
        )


@dataclass(frozen=True)
class V6RootCgIntegerPipelineResult:
    root: V6RootCgResult
    integer: V6RestrictedIntegerResult
    diagnostics: Mapping[str, object]


def _candidate_bays_by_group(atoms) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for atom in atoms:
        output.setdefault(atom.group_id, set()).add(atom.anchor_bay_key)
    return output


class V6RestrictedIntegerSolver:
    """Solve one integer master over exactly the provided V6 zone pool."""

    def __init__(
        self,
        problem: ProblemData,
        zones: Sequence[RowAwareZone],
        peak_policy: V6PeakUtilizationPolicy,
        config: V6RestrictedIntegerConfig | None = None,
        warm_start: V6RestrictedIntegerWarmStart | None = None,
    ) -> None:
        self.problem = problem
        self.config = config or V6RestrictedIntegerConfig()
        self.config.validate()
        self.peak_policy = peak_policy
        self.warm_start = warm_start
        if not zones:
            raise ValueError("V6 restricted integer master requires a zone pool")
        signatures = [tuple(zone.candidate_indices) for zone in zones]
        if len(signatures) != len(set(signatures)):
            raise ValueError("V6 restricted integer zone pool has duplicates")
        self.zones = tuple(
            replace(zone, zone_id=index) for index, zone in enumerate(zones)
        )
        self.atoms, self.anchor_capacity_limits = build_v6_row_aware_bay_atoms(
            problem
        )
        self.evaluator = V6ModelEvaluator(
            problem,
            self.zones,
            self.config.objective,
            candidate_bays_by_group=_candidate_bays_by_group(self.atoms),
        )

    @staticmethod
    def _recover_zone_bay_flow(
        zones_by_id: Mapping[int, RowAwareZone],
        selected_zone_ids: Sequence[int],
        group_bay_flow: Mapping[tuple[str, str], int],
    ) -> dict[tuple[int, str], int]:
        owner_by_group_bay: dict[tuple[str, str], int] = {}
        for zone_id in selected_zone_ids:
            zone = zones_by_id[int(zone_id)]
            for bay_key in zone.anchor_bay_keys:
                key = (zone.group_id, bay_key)
                if key in owner_by_group_bay:
                    raise RuntimeError(
                        "V6 integer solution selected overlapping zones for one "
                        f"group-bay pair: key={key}, zones="
                        f"{owner_by_group_bay[key]},{zone_id}"
                    )
                owner_by_group_bay[key] = int(zone_id)

        recovered: dict[tuple[int, str], int] = {}
        for key, raw_quantity in group_bay_flow.items():
            quantity = int(raw_quantity)
            if quantity <= 0:
                continue
            zone_id = owner_by_group_bay.get(key)
            if zone_id is None:
                raise RuntimeError(
                    "positive V6 group-bay flow has no selected zone owner: "
                    f"key={key}, quantity={quantity}"
                )
            recovered[(zone_id, key[1])] = quantity

        for key, zone_id in owner_by_group_bay.items():
            if recovered.get((zone_id, key[1]), 0) <= 0:
                raise RuntimeError(
                    "selected V6 zone bay has no positive recovered flow: "
                    f"zone={zone_id}, group_bay={key}"
                )
        return recovered

    def solve(self) -> V6RestrictedIntegerResult:
        started = perf_counter()
        master = V6ProjectedRestrictedMaster(
            self.problem,
            self.atoms,
            self.anchor_capacity_limits,
            self.evaluator,
            self.peak_policy,
            phase="business",
            initial_zones=self.zones,
            solver_threads=self.config.solver_threads,
            solver_seed=self.config.solver_seed,
            verbose=self.config.verbose,
            integral=True,
            time_limit=self.config.time_limit,
            mip_gap=self.config.mip_gap,
        )
        try:
            warm_start_diagnostics: dict[str, object] = {
                "provided": False,
                "applied": False,
            }
            if self.warm_start is not None:
                warm_start_diagnostics = master.apply_integer_warm_start(
                    selected_zone_signatures=(
                        self.warm_start.selected_zone_signatures
                    ),
                    group_bay_flow=self.warm_start.group_bay_flow,
                    import_reservation=self.warm_start.import_reservation,
                )
                warm_start_diagnostics.update(
                    {
                        "source": self.warm_start.source,
                        "source_objective": (
                            self.warm_start.source_objective
                        ),
                    }
                )
            raw = master.solve_integer()
            zones = tuple(master.zones_by_id.values())
            zones_by_id = {zone.zone_id: zone for zone in zones}
            zone_bay_flow = self._recover_zone_bay_flow(
                zones_by_id,
                raw.selected_zone_ids,
                raw.group_bay_flow,
            )
            certificate_evaluator = V6ModelEvaluator(
                self.problem,
                zones,
                self.config.objective,
                candidate_bays_by_group=_candidate_bays_by_group(self.atoms),
            )
            certificate = certificate_evaluator.evaluate(
                raw.selected_zone_ids,
                zone_bay_flow,
                raw.import_reservation,
                self.peak_policy,
            )
            reconstructed = float(certificate["objective"])
            if not math.isclose(
                raw.objective,
                reconstructed,
                rel_tol=1e-8,
                abs_tol=1e-8,
            ):
                raise RuntimeError(
                    "V6 restricted integer objective differs from independent "
                    f"reconstruction: solver={raw.objective}, "
                    f"evaluator={reconstructed}"
                )
            diagnostics = {
                "algorithm": "v6_restricted_integer_master",
                "model_schema_version": V6_MODEL_SCHEMA_VERSION,
                "result_role": "validated_restricted_integer_incumbent",
                "zone_pool_count": len(zones),
                "selected_zone_count": len(raw.selected_zone_ids),
                "status": raw.status,
                "proven_optimal_over_restricted_pool": raw.proven_optimal,
                "solver_objective": raw.objective,
                "solver_bound": raw.best_bound,
                "solver_gap": raw.mip_gap,
                "solver_runtime_seconds": raw.runtime_seconds,
                "progress": raw.progress,
                "warm_start": warm_start_diagnostics,
                "warm_start_objective_preserved": (
                    self.warm_start is None
                    or reconstructed
                    <= self.warm_start.source_objective + 1e-8
                ),
                "solver_evaluator_objective_difference": (
                    raw.objective - reconstructed
                ),
                "independent_validation_passed": bool(
                    certificate["validation"]["passed"]
                ),
                "total_seconds": perf_counter() - started,
            }
            return V6RestrictedIntegerResult(
                selected_zone_ids=raw.selected_zone_ids,
                zone_bay_flow=zone_bay_flow,
                import_reservation=raw.import_reservation,
                peak_policy=self.peak_policy,
                certificate=certificate,
                diagnostics=diagnostics,
                zones=zones,
            )
        finally:
            master.dispose()


class V6RootCgIntegerPipeline:
    """Run exact root CG, then solve its generated-column integer master."""

    def __init__(
        self,
        problem: ProblemData,
        peak_policy: V6PeakUtilizationPolicy,
        root_config: V6RootCgConfig | None = None,
        integer_config: V6RestrictedIntegerConfig | None = None,
    ) -> None:
        self.problem = problem
        self.peak_policy = peak_policy
        self.root_config = root_config or V6RootCgConfig()
        self.integer_config = integer_config or V6RestrictedIntegerConfig(
            objective=self.root_config.objective
        )
        self.root_config.validate()
        self.integer_config.validate()
        if self.root_config.objective != self.integer_config.objective:
            raise ValueError(
                "V6 root CG and restricted integer master must share one "
                "objective configuration"
            )

    def solve(self) -> V6RootCgIntegerPipelineResult:
        started = perf_counter()
        root = V6RootColumnGeneration(
            self.problem,
            self.peak_policy,
            self.root_config,
        ).solve()
        integer = V6RestrictedIntegerSolver(
            self.problem,
            root.zones,
            self.peak_policy,
            self.integer_config,
        ).solve()
        root_bound = float(root.objective)
        incumbent = float(integer.objective)
        if incumbent + 1e-8 < root_bound:
            raise RuntimeError(
                "V6 restricted integer incumbent is below the closed root bound: "
                f"incumbent={incumbent}, root_bound={root_bound}"
            )
        absolute_gap = max(0.0, incumbent - root_bound)
        relative_gap = absolute_gap / max(1.0, abs(incumbent))
        diagnostics = {
            "algorithm": "v6_root_cg_then_restricted_integer",
            "model_schema_version": V6_MODEL_SCHEMA_VERSION,
            "root_lp_lower_bound": root_bound,
            "restricted_integer_upper_bound": incumbent,
            "absolute_root_gap": absolute_gap,
            "relative_root_gap": relative_gap,
            "generated_zone_count": len(root.zones),
            "integer_validation_passed": True,
            "total_seconds": perf_counter() - started,
        }
        return V6RootCgIntegerPipelineResult(
            root=root,
            integer=integer,
            diagnostics=diagnostics,
        )


__all__ = [
    "V6RestrictedIntegerWarmStart",
    "V6RestrictedIntegerConfig",
    "V6RestrictedIntegerResult",
    "V6RestrictedIntegerSolver",
    "V6RootCgIntegerPipeline",
    "V6RootCgIntegerPipelineResult",
]
