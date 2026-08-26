"""Fully enumerated V6 min-max and business MIPs for exact small cases."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Mapping, Sequence

from .gurobi_backend import GurobiModel, MipProgressRecorder
from .models import ProblemData
from .row_aware_zones import (
    RowAwareZone,
    build_complete_v6_zone_universe,
    v6_footprint,
)
from .v6_model import (
    V6_MODEL_SCHEMA_VERSION,
    V6ModelEvaluator,
    V6ObjectiveConfig,
    V6PeakUtilizationPolicy,
)


@dataclass(frozen=True)
class V6CompleteMipConfig:
    """Execution controls for the exact, fully enumerated V6 reference."""

    peak_time_limit: float = 60.0
    business_time_limit: float = 60.0
    business_mip_gap: float = 0.0
    solver_threads: int = 1
    solver_seed: int = 0
    verbose: bool = False
    maximum_zone_count: int = 250_000
    objective: V6ObjectiveConfig = field(default_factory=V6ObjectiveConfig)

    def validate(self) -> None:
        if not math.isfinite(float(self.peak_time_limit)) or float(
            self.peak_time_limit
        ) <= 0.0:
            raise ValueError("V6 peak_time_limit must be positive and finite")
        if not math.isfinite(float(self.business_time_limit)) or float(
            self.business_time_limit
        ) <= 0.0:
            raise ValueError("V6 business_time_limit must be positive and finite")
        if not math.isfinite(float(self.business_mip_gap)) or not 0.0 <= float(
            self.business_mip_gap
        ) <= 1.0:
            raise ValueError("V6 business_mip_gap must lie in [0, 1]")
        if int(self.solver_threads) < 0:
            raise ValueError("V6 solver_threads must be nonnegative")
        if int(self.maximum_zone_count) <= 0:
            raise ValueError("V6 maximum_zone_count must be positive")
        self.objective.validate()


@dataclass(frozen=True)
class V6CompleteMipResult:
    """Validated incumbent and exact-reference diagnostics."""

    selected_zone_ids: tuple[int, ...]
    zone_bay_flow: Mapping[tuple[int, str], int]
    import_reservation: Mapping[tuple[str, str, str], int]
    peak_policy: V6PeakUtilizationPolicy
    certificate: Mapping[str, object]
    diagnostics: Mapping[str, object]
    zones: Sequence[RowAwareZone]

    def as_dict(self) -> dict[str, object]:
        zones_by_id = {zone.zone_id: zone for zone in self.zones}
        selected_rows: list[dict[str, object]] = []
        for zone_id in self.selected_zone_ids:
            zone = zones_by_id[zone_id]
            selected_rows.append(
                {
                    "zone_id": int(zone_id),
                    "group_id": zone.group_id,
                    "area_no": zone.area_no,
                    "anchor_bay_keys": list(zone.anchor_bay_keys),
                    "anchor_bay_capacities": [
                        {"bay_key": bay_key, "capacity": int(capacity)}
                        for bay_key, capacity in zone.anchor_bay_capacities
                    ],
                    "rows_by_anchor_bay": [
                        {"bay_key": bay_key, "row_nos": list(row_nos)}
                        for bay_key, row_nos in zone.rows_by_anchor_bay
                    ],
                    "physical_row_resources": [
                        {"bay_key": bay_key, "row_no": row_no}
                        for bay_key, row_no in zone.resources
                    ],
                    "flow_by_anchor_bay": {
                        bay_key: int(
                            self.zone_bay_flow.get((zone_id, bay_key), 0)
                        )
                        for bay_key in zone.anchor_bay_keys
                    },
                }
            )
        imports = [
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
        ]
        return {
            "model_schema_version": V6_MODEL_SCHEMA_VERSION,
            "peak_policy": self.peak_policy.as_dict(),
            "selected_zones": selected_rows,
            "anonymous_import_reservation": imports,
            "certificate": dict(self.certificate),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class _V6MipArtifacts:
    model: GurobiModel
    zone_selection: dict[int, object]
    zone_bay_flow: dict[tuple[int, str], object]
    import_reservation: dict[tuple[str, str, str], object]
    peak_utilization: object | None
    constraint_count_by_family: dict[str, int]
    variable_count_by_family: dict[str, int]
    fingerprint: int


class V6CompleteMipSolver:
    """Solve the complete V6 zone model without V5 solver dependencies."""

    def __init__(
        self,
        problem: ProblemData,
        config: V6CompleteMipConfig | None = None,
    ) -> None:
        self.problem = problem
        self.config = config or V6CompleteMipConfig()
        self.config.validate()
        enumeration_started = perf_counter()
        self.zones = tuple(
            build_complete_v6_zone_universe(
                problem,
                maximum_zone_count=int(self.config.maximum_zone_count),
            )
        )
        self.enumeration_seconds = perf_counter() - enumeration_started
        if len(self.zones) > int(self.config.maximum_zone_count):
            raise ValueError(
                "provided complete V6 zone universe exceeds the execution "
                f"safety limit: zones={len(self.zones)}, "
                f"limit={self.config.maximum_zone_count}"
            )
        self.evaluator = V6ModelEvaluator(
            problem,
            self.zones,
            self.config.objective,
        )
        self.groups = self.evaluator.groups
        self.groups_by_id = self.evaluator.groups_by_id
        self.zones_by_id = self.evaluator.zones_by_id

    def solve(
        self,
        peak_policy: V6PeakUtilizationPolicy | None = None,
        peak_diagnostics: Mapping[str, object] | None = None,
    ) -> V6CompleteMipResult:
        total_started = perf_counter()
        if peak_policy is None:
            peak_policy, effective_peak_diagnostics = self.solve_peak_reference()
        else:
            if not math.isclose(
                float(peak_policy.headroom_fraction),
                float(
                    self.config.objective.peak_utilization_headroom_fraction
                ),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "complete V6 MIP peak policy differs from its objective config"
                )
            effective_peak_diagnostics = {
                "status": "externally_certified",
                "proven_optimal": bool(
                    peak_policy.as_dict()[
                        "minimum_feasible_utilization_proven"
                    ]
                ),
                "peak_policy": peak_policy.as_dict(),
                **dict(peak_diagnostics or {}),
            }
        (
            selected_zone_ids,
            zone_bay_flow,
            import_reservation,
            business_diagnostics,
        ) = self._solve_business_model(peak_policy)
        certificate = self.evaluator.evaluate(
            selected_zone_ids,
            zone_bay_flow,
            import_reservation,
            peak_policy,
        )
        solver_objective = float(business_diagnostics["solver_objective"])
        reconstructed = float(certificate["objective"])
        if not math.isclose(
            solver_objective,
            reconstructed,
            rel_tol=1e-8,
            abs_tol=1e-8,
        ):
            raise RuntimeError(
                "complete V6 MIP objective differs from independent "
                f"reconstruction: solver={solver_objective}, "
                f"evaluator={reconstructed}"
            )
        diagnostics = {
            "algorithm": "complete_enumerated_v6_zone_mip",
            "model_schema_version": V6_MODEL_SCHEMA_VERSION,
            "model_equivalence_scope": "same_v6_model_for_future_cg_baseline",
            "complete_zone_count": len(self.zones),
            "enumeration_seconds": round(self.enumeration_seconds, 6),
            "peak_reference": effective_peak_diagnostics,
            "business_model": business_diagnostics,
            "solver_evaluator_objective_difference": (
                solver_objective - reconstructed
            ),
            "independent_validation_passed": True,
            "total_seconds": round(perf_counter() - total_started, 6),
        }
        return V6CompleteMipResult(
            selected_zone_ids=tuple(sorted(selected_zone_ids)),
            zone_bay_flow=dict(sorted(zone_bay_flow.items())),
            import_reservation=dict(sorted(import_reservation.items())),
            peak_policy=peak_policy,
            certificate=certificate,
            diagnostics=diagnostics,
            zones=self.zones,
        )

    def solve_peak_reference(
        self,
    ) -> tuple[V6PeakUtilizationPolicy, dict[str, object]]:
        """Return the proven full-V6 peak policy without solving business MIP."""

        return self._solve_peak_reference()

    def _configure_model(
        self,
        model: GurobiModel,
        *,
        time_limit: float,
        mip_gap: float,
    ) -> None:
        if not self.config.verbose:
            model.hideOutput()
        model.setMinimize()
        model.setParam("Seed", int(self.config.solver_seed))
        if int(self.config.solver_threads) > 0:
            model.setParam("Threads", int(self.config.solver_threads))
        model.setParam("TimeLimit", float(time_limit))
        model.setParam("MIPGap", float(mip_gap))
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("IntFeasTol", 1e-9)

    def _build_model(
        self,
        *,
        objective_mode: str,
        peak_cap: float | None,
    ) -> _V6MipArtifacts:
        if objective_mode not in {"peak_minmax", "business"}:
            raise ValueError(f"unknown V6 complete-MIP objective: {objective_mode}")
        if objective_mode == "business" and peak_cap is None:
            raise ValueError("V6 business MIP requires a peak-utilization cap")

        from gurobipy import quicksum

        model = GurobiModel(f"v6_complete_{objective_mode}")
        self._configure_model(
            model,
            time_limit=(
                self.config.peak_time_limit
                if objective_mode == "peak_minmax"
                else self.config.business_time_limit
            ),
            mip_gap=(
                0.0
                if objective_mode == "peak_minmax"
                else self.config.business_mip_gap
            ),
        )
        business = objective_mode == "business"
        constraints: defaultdict[str, list[object]] = defaultdict(list)

        zone_selection = {
            zone.zone_id: model.addVar(
                vtype="B",
                obj=(
                    self.evaluator.zone_selection_objective_coefficient(zone)
                    if business
                    else 0.0
                ),
                name=f"x_zone_{zone.zone_id}",
            )
            for zone in self.zones
        }
        zone_bay_flow = {
            (zone.zone_id, bay_key): model.addVar(
                lb=0.0,
                ub=float(capacity),
                vtype="I",
                obj=(
                    self.evaluator.zone_bay_flow_objective_coefficient(
                        zone,
                        bay_key,
                    )
                    if business
                    else 0.0
                ),
                name=f"q_zone_{zone.zone_id}_bay_{position}",
            )
            for zone in self.zones
            for position, (bay_key, capacity) in enumerate(
                zone.anchor_bay_capacities
            )
        }
        import_candidate_rows = [
            (flow, size, bay_key, capacity)
            for (flow, size), candidates in sorted(
                self.evaluator.import_candidates.items()
            )
            for bay_key, capacity in candidates
        ]
        import_reservation = {
            (flow, size, bay_key): model.addVar(
                lb=0.0,
                ub=float(capacity),
                vtype="I",
                name=f"p_import_{index}",
            )
            for index, (flow, size, bay_key, capacity) in enumerate(
                import_candidate_rows
            )
        }
        peak_utilization = (
            model.addVar(
                lb=0.0,
                ub=1.0,
                obj=1.0,
                name="rho_peak",
            )
            if objective_mode == "peak_minmax"
            else None
        )
        if business:
            model.addVar(
                lb=1.0,
                ub=1.0,
                obj=self.evaluator.objective_constant(),
                name="business_objective_constant",
            )

        zones_by_group: defaultdict[str, list[RowAwareZone]] = defaultdict(list)
        zones_by_resource: defaultdict[tuple[str, str], list[RowAwareZone]] = (
            defaultdict(list)
        )
        zones_by_group_bay: defaultdict[tuple[str, str], list[RowAwareZone]] = (
            defaultdict(list)
        )
        zones_by_physical_bay: defaultdict[str, list[RowAwareZone]] = (
            defaultdict(list)
        )
        reserved_by_physical_bay: defaultdict[
            str, list[tuple[int, object]]
        ] = defaultdict(list)
        reserved_by_anchor_size: defaultdict[
            tuple[str, str], list[tuple[int, object]]
        ] = defaultdict(list)
        flow_by_group: defaultdict[str, list[object]] = defaultdict(list)
        flow_by_voyage_area: defaultdict[
            tuple[str, str], list[object]
        ] = defaultdict(list)
        planned_load_by_area: defaultdict[
            str, list[tuple[int, object]]
        ] = defaultdict(list)
        export_sizes_by_bay: defaultdict[str, set[str]] = defaultdict(set)
        export_heights_by_bay: defaultdict[str, set[str]] = defaultdict(set)

        for zone in self.zones:
            group = self.groups_by_id[zone.group_id]
            zones_by_group[group.group_id].append(zone)
            for resource in zone.resources:
                zones_by_resource[resource].append(zone)
            for physical_bay_key in zone.physical_bay_keys:
                zones_by_physical_bay[physical_bay_key].append(zone)
                export_sizes_by_bay[physical_bay_key].add(str(group.size))
                export_heights_by_bay[physical_bay_key].add(str(group.height))
            capacity_by_bay = dict(zone.anchor_bay_capacities)
            for bay_key in zone.anchor_bay_keys:
                variable = zone_bay_flow[(zone.zone_id, bay_key)]
                flow_by_group[group.group_id].append(variable)
                flow_by_voyage_area[(group.voyage_id, zone.area_no)].append(
                    variable
                )
                zones_by_group_bay[(group.group_id, bay_key)].append(zone)
                capacity = int(capacity_by_bay[bay_key])
                reserved_by_anchor_size[(bay_key, group.size)].append(
                    (capacity, zone_selection[zone.zone_id])
                )
                footprint = v6_footprint(self.problem, bay_key, group.size)
                for physical_bay_key in footprint:
                    reserved_by_physical_bay[physical_bay_key].append(
                        (capacity, zone_selection[zone.zone_id])
                    )
                planned_load_by_area[zone.area_no].append(
                    (len(footprint), variable)
                )
                constraints["zone_flow_lower"].append(
                    model.addConstr(
                        variable >= zone_selection[zone.zone_id],
                        name=f"zone_flow_lower_{zone.zone_id}_{bay_key}",
                    )
                )
                constraints["zone_flow_upper"].append(
                    model.addConstr(
                        variable
                        <= capacity * zone_selection[zone.zone_id],
                        name=f"zone_flow_upper_{zone.zone_id}_{bay_key}",
                    )
                )

        for group in self.groups:
            variables = flow_by_group.get(group.group_id, [])
            if not variables:
                raise ValueError(f"V6 group has no zone flow: {group.group_id}")
            constraints["export_group_demand"].append(
                model.addConstr(
                    quicksum(variables) == int(group.demand),
                    name=f"export_group_demand_{group.group_id}",
                )
            )
        for resource, zones in sorted(zones_by_resource.items()):
            constraints["physical_row_exclusive"].append(
                model.addConstr(
                    quicksum(zone_selection[zone.zone_id] for zone in zones)
                    <= 1,
                    name=f"physical_row_exclusive_{len(constraints['physical_row_exclusive'])}",
                )
            )
        for key, zones in sorted(zones_by_group_bay.items()):
            constraints["same_group_bay_nonoverlap"].append(
                model.addConstr(
                    quicksum(zone_selection[zone.zone_id] for zone in zones)
                    <= 1,
                    name=f"same_group_bay_nonoverlap_{len(constraints['same_group_bay_nonoverlap'])}",
                )
            )
        for bay_key, terms in sorted(reserved_by_physical_bay.items()):
            constraints["export_physical_capacity"].append(
                model.addConstr(
                    quicksum(coefficient * variable for coefficient, variable in terms)
                    <= int(self.problem.bays[bay_key].physical_capacity),
                    name=f"export_physical_capacity_{bay_key}",
                )
            )
        for (bay_key, size), terms in sorted(reserved_by_anchor_size.items()):
            constraints["export_anchor_size_capacity"].append(
                model.addConstr(
                    quicksum(coefficient * variable for coefficient, variable in terms)
                    <= int(self.problem.bays[bay_key].cap_by_size.get(size, 0)),
                    name=f"export_anchor_size_capacity_{bay_key}_{size}",
                )
            )

        size_state = {
            (bay_key, size): model.addVar(
                vtype="B",
                name=f"export_size_state_{index}",
            )
            for index, (bay_key, size) in enumerate(
                sorted(
                    (bay_key, size)
                    for bay_key, sizes in export_sizes_by_bay.items()
                    for size in sizes
                )
            )
        }
        height_state = {
            (bay_key, height): model.addVar(
                vtype="B",
                name=f"export_height_state_{index}",
            )
            for index, (bay_key, height) in enumerate(
                sorted(
                    (bay_key, height)
                    for bay_key, heights in export_heights_by_bay.items()
                    for height in heights
                )
            )
        }
        for bay_key, zones in sorted(zones_by_physical_bay.items()):
            for zone in zones:
                group = self.groups_by_id[zone.group_id]
                constraints["export_size_state_link"].append(
                    model.addConstr(
                        zone_selection[zone.zone_id]
                        <= size_state[(bay_key, group.size)],
                        name=f"export_size_state_link_{bay_key}_{zone.zone_id}",
                    )
                )
                constraints["export_height_state_link"].append(
                    model.addConstr(
                        zone_selection[zone.zone_id]
                        <= height_state[(bay_key, group.height)],
                        name=f"export_height_state_link_{bay_key}_{zone.zone_id}",
                    )
                )
            constraints["export_size_state_choice"].append(
                model.addConstr(
                    quicksum(
                        size_state[(bay_key, size)]
                        for size in export_sizes_by_bay[bay_key]
                    )
                    <= 1,
                    name=f"export_size_state_choice_{bay_key}",
                )
            )
            constraints["export_height_state_choice"].append(
                model.addConstr(
                    quicksum(
                        height_state[(bay_key, height)]
                        for height in export_heights_by_bay[bay_key]
                    )
                    <= 1,
                    name=f"export_height_state_choice_{bay_key}",
                )
            )

        import_by_flow_size: defaultdict[tuple[str, str], list[object]] = (
            defaultdict(list)
        )
        import_by_physical_bay: defaultdict[str, list[object]] = defaultdict(list)
        import_by_anchor_size: defaultdict[tuple[str, str], list[object]] = (
            defaultdict(list)
        )
        import_by_physical_size: defaultdict[tuple[str, str], list[object]] = (
            defaultdict(list)
        )
        import_sizes_by_bay: defaultdict[str, set[str]] = defaultdict(set)
        for (flow, size, bay_key), variable in import_reservation.items():
            import_by_flow_size[(flow, size)].append(variable)
            import_by_anchor_size[(bay_key, size)].append(variable)
            footprint = v6_footprint(self.problem, bay_key, size)
            for physical_bay_key in footprint:
                import_by_physical_bay[physical_bay_key].append(variable)
                import_by_physical_size[(physical_bay_key, size)].append(variable)
                import_sizes_by_bay[physical_bay_key].add(size)
            planned_load_by_area[self.problem.bays[bay_key].area_no].append(
                (len(footprint), variable)
            )
        for key, required in sorted(self.problem.import_demand_by_flow_size.items()):
            normalized_key = tuple(map(str, key))
            constraints["import_demand"].append(
                model.addConstr(
                    quicksum(import_by_flow_size.get(normalized_key, []))
                    == int(required),
                    name=f"import_demand_{normalized_key[0]}_{normalized_key[1]}",
                )
            )
        for (bay_key, size), variables in sorted(import_by_anchor_size.items()):
            constraints["import_anchor_size_capacity"].append(
                model.addConstr(
                    quicksum(variables)
                    <= int(self.problem.bays[bay_key].cap_by_size.get(size, 0)),
                    name=f"import_anchor_size_capacity_{bay_key}_{size}",
                )
            )

        allocation_bays = sorted(
            set(zones_by_physical_bay) | set(import_by_physical_bay)
        )
        export_bay_use = {
            bay_key: model.addVar(vtype="B", name=f"export_bay_use_{index}")
            for index, bay_key in enumerate(allocation_bays)
        }
        import_bay_use = {
            bay_key: model.addVar(vtype="B", name=f"import_bay_use_{index}")
            for index, bay_key in enumerate(allocation_bays)
        }
        import_size_state = {
            (bay_key, size): model.addVar(
                vtype="B",
                name=f"import_size_state_{index}",
            )
            for index, (bay_key, size) in enumerate(
                sorted(
                    (bay_key, size)
                    for bay_key, sizes in import_sizes_by_bay.items()
                    for size in sizes
                )
            )
        }
        for bay_key in allocation_bays:
            export_zones = zones_by_physical_bay.get(bay_key, [])
            export_sum = quicksum(
                zone_selection[zone.zone_id] for zone in export_zones
            )
            import_sum = quicksum(import_by_physical_bay.get(bay_key, []))
            if export_zones:
                constraints["export_bay_use_link"].append(
                    model.addConstr(
                        export_sum
                        <= len(export_zones) * export_bay_use[bay_key],
                        name=f"export_bay_use_link_{bay_key}",
                    )
                )
                constraints["export_bay_use_presence"].append(
                    model.addConstr(
                        export_bay_use[bay_key] <= export_sum,
                        name=f"export_bay_use_presence_{bay_key}",
                    )
                )
            else:
                constraints["export_bay_use_zero"].append(
                    model.addConstr(
                        export_bay_use[bay_key] == 0,
                        name=f"export_bay_use_zero_{bay_key}",
                    )
                )
            if import_by_physical_bay.get(bay_key):
                capacity = max(1, int(self.problem.bays[bay_key].physical_capacity))
                constraints["import_physical_capacity"].append(
                    model.addConstr(
                        import_sum <= capacity * import_bay_use[bay_key],
                        name=f"import_physical_capacity_{bay_key}",
                    )
                )
                constraints["import_bay_use_presence"].append(
                    model.addConstr(
                        import_bay_use[bay_key] <= import_sum,
                        name=f"import_bay_use_presence_{bay_key}",
                    )
                )
                for size in import_sizes_by_bay[bay_key]:
                    size_sum = quicksum(
                        import_by_physical_size[(bay_key, size)]
                    )
                    constraints["import_size_state_link"].append(
                        model.addConstr(
                            size_sum
                            <= capacity * import_size_state[(bay_key, size)],
                            name=f"import_size_state_link_{bay_key}_{size}",
                        )
                    )
                    constraints["import_size_state_presence"].append(
                        model.addConstr(
                            import_size_state[(bay_key, size)] <= size_sum,
                            name=f"import_size_state_presence_{bay_key}_{size}",
                        )
                    )
                constraints["import_size_state_choice"].append(
                    model.addConstr(
                        quicksum(
                            import_size_state[(bay_key, size)]
                            for size in import_sizes_by_bay[bay_key]
                        )
                        <= import_bay_use[bay_key],
                        name=f"import_size_state_choice_{bay_key}",
                    )
                )
            else:
                constraints["import_bay_use_zero"].append(
                    model.addConstr(
                        import_bay_use[bay_key] == 0,
                        name=f"import_bay_use_zero_{bay_key}",
                    )
                )
            constraints["export_import_bay_exclusive"].append(
                model.addConstr(
                    export_bay_use[bay_key] + import_bay_use[bay_key] <= 1,
                    name=f"export_import_bay_exclusive_{bay_key}",
                )
            )

        voyage_area_use: dict[tuple[str, str], object] = {}
        if business:
            demand_by_voyage: Counter[str] = Counter()
            for group in self.groups:
                demand_by_voyage[group.voyage_id] += int(group.demand)
            voyage_area_use = {
                key: model.addVar(
                    vtype="B",
                    obj=self.evaluator.voyage_area_objective_coefficient(),
                    name=f"voyage_area_use_{index}",
                )
                for index, key in enumerate(sorted(flow_by_voyage_area))
            }
            for key, variables in sorted(flow_by_voyage_area.items()):
                assigned = quicksum(variables)
                use = voyage_area_use[key]
                constraints["voyage_area_link"].append(
                    model.addConstr(
                        assigned <= int(demand_by_voyage[key[0]]) * use,
                        name=f"voyage_area_link_{key[0]}_{key[1]}",
                    )
                )
                constraints["voyage_area_presence"].append(
                    model.addConstr(
                        use <= assigned,
                        name=f"voyage_area_presence_{key[0]}_{key[1]}",
                    )
                )

        area_capacity: Counter[str] = Counter()
        for bay in self.problem.bays.values():
            area_capacity[str(bay.area_no)] += int(bay.physical_capacity)
        for area_no, terms in sorted(planned_load_by_area.items()):
            load = quicksum(
                coefficient * variable for coefficient, variable in terms
            )
            capacity = int(area_capacity[area_no])
            if capacity <= 0:
                raise ValueError(f"V6 used area has no residual capacity: {area_no}")
            right_hand_side = (
                capacity * peak_utilization
                if objective_mode == "peak_minmax"
                else capacity * float(peak_cap)
            )
            constraints["peak_utilization"].append(
                model.addConstr(
                    load <= right_hand_side,
                    name=f"peak_utilization_{area_no}",
                )
            )

        model.update()
        variable_count_by_family = {
            "zone_selection": len(zone_selection),
            "zone_bay_flow": len(zone_bay_flow),
            "anonymous_import_reservation": len(import_reservation),
            "export_size_state": len(size_state),
            "export_height_state": len(height_state),
            "export_bay_use": len(export_bay_use),
            "import_bay_use": len(import_bay_use),
            "import_size_state": len(import_size_state),
            "voyage_area_use": len(voyage_area_use),
            "peak_utilization": int(peak_utilization is not None),
            "total": len(model.getVars()),
        }
        return _V6MipArtifacts(
            model=model,
            zone_selection=zone_selection,
            zone_bay_flow=zone_bay_flow,
            import_reservation=import_reservation,
            peak_utilization=peak_utilization,
            constraint_count_by_family={
                key: len(values) for key, values in sorted(constraints.items())
            },
            variable_count_by_family=variable_count_by_family,
            fingerprint=model.getFingerprint(),
        )

    def _solve_peak_reference(
        self,
    ) -> tuple[V6PeakUtilizationPolicy, dict[str, object]]:
        artifacts = self._build_model(
            objective_mode="peak_minmax",
            peak_cap=None,
        )
        recorder = MipProgressRecorder(phase="v6_peak_minmax")
        try:
            artifacts.model.optimize(callback=recorder)
            progress = recorder.finalize(artifacts.model)
            status = artifacts.model.getStatusName()
            if status != "optimal":
                raise RuntimeError(
                    "V6 peak reference must be proven optimal before the "
                    f"business MIP can run: status={status}"
                )
            if artifacts.model.getSolutionCount() <= 0:
                raise RuntimeError("V6 peak reference has no feasible solution")
            raw_minimum = artifacts.model.getValue(artifacts.peak_utilization)
            if raw_minimum < -1e-8 or raw_minimum > 1.0 + 1e-8:
                raise RuntimeError(
                    f"V6 peak reference is outside [0, 1]: {raw_minimum}"
                )
            minimum = min(1.0, max(0.0, float(raw_minimum)))
            policy = V6PeakUtilizationPolicy(
                minimum_feasible_utilization=minimum,
                headroom_fraction=float(
                    self.config.objective.peak_utilization_headroom_fraction
                ),
            )
            diagnostics = {
                "status": status,
                "proven_optimal": True,
                "minimum_feasible_utilization": minimum,
                "solver_objective": artifacts.model.getObjectiveValue(),
                "solver_bound": artifacts.model.getBestBound(),
                "solver_gap": artifacts.model.getMipGap(),
                "runtime_seconds": artifacts.model.getRuntime(),
                "fingerprint": artifacts.fingerprint,
                "variable_count_by_family": artifacts.variable_count_by_family,
                "constraint_count_by_family": artifacts.constraint_count_by_family,
                "progress": progress,
            }
            return policy, diagnostics
        finally:
            artifacts.model.dispose()

    def _solve_business_model(
        self,
        peak_policy: V6PeakUtilizationPolicy,
    ) -> tuple[
        set[int],
        dict[tuple[int, str], int],
        dict[tuple[str, str, str], int],
        dict[str, object],
    ]:
        artifacts = self._build_model(
            objective_mode="business",
            peak_cap=peak_policy.epsilon_cap,
        )
        recorder = MipProgressRecorder(phase="v6_business_complete_mip")
        try:
            artifacts.model.optimize(callback=recorder)
            progress = recorder.finalize(artifacts.model)
            status = artifacts.model.getStatusName()
            if artifacts.model.getSolutionCount() <= 0:
                raise RuntimeError(
                    "complete V6 business MIP has no feasible incumbent: "
                    f"status={status}"
                )
            selected = {
                zone_id
                for zone_id, variable in artifacts.zone_selection.items()
                if artifacts.model.getValue(variable) > 0.5
            }
            flow = {
                key: int(round(artifacts.model.getValue(variable)))
                for key, variable in artifacts.zone_bay_flow.items()
                if artifacts.model.getValue(variable) > 0.5
            }
            imports = {
                key: int(round(artifacts.model.getValue(variable)))
                for key, variable in artifacts.import_reservation.items()
                if artifacts.model.getValue(variable) > 0.5
            }
            diagnostics = {
                "status": status,
                "proven_optimal": status == "optimal",
                "solver_objective": artifacts.model.getObjectiveValue(),
                "solver_bound": artifacts.model.getBestBound(),
                "solver_gap": artifacts.model.getMipGap(),
                "runtime_seconds": artifacts.model.getRuntime(),
                "fingerprint": artifacts.fingerprint,
                "variable_count_by_family": artifacts.variable_count_by_family,
                "constraint_count_by_family": artifacts.constraint_count_by_family,
                "selected_zone_count": len(selected),
                "positive_zone_bay_flow_count": len(flow),
                "positive_import_reservation_count": len(imports),
                "peak_policy": peak_policy.as_dict(),
                "progress": progress,
            }
            return selected, flow, imports, diagnostics
        finally:
            artifacts.model.dispose()


__all__ = [
    "V6CompleteMipConfig",
    "V6CompleteMipResult",
    "V6CompleteMipSolver",
]
