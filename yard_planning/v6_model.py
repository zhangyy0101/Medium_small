"""Canonical V6 mathematical-model contract and independent evaluator.

The solver implementations in the V5 modules are deliberately not imported
here.  Both the forthcoming fully enumerated MIP and column-generation master
must consume this module so that feasibility, objective normalization, and
objective reconstruction have one source of truth.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .models import (
    EXPORT_GROUP_IDENTITY_ATTRIBUTES,
    EXPORT_VOYAGE_ROW_NO_MIX_ATTR,
    ExportGroup,
    ProblemData,
    existing_export_group_key,
)
from .row_aware_zones import (
    RowAwareBayAtom,
    RowAwareZone,
    build_v6_row_aware_bay_atoms,
    v6_edge_large_bays as _edge_large_bays,
    v6_footprint as _footprint,
)


V6_MODEL_SCHEMA_VERSION = "row_aware_bay_zone_v6_1"
V6_OBJECTIVE_VERSION = "v6_three_category_normalized_v1"

ZoneBayFlowKey = tuple[int, str]
ImportReservationKey = tuple[str, str, str]


@dataclass(frozen=True)
class V6ObjectiveConfig:
    """Three declared objective categories and their internal composition.

    The defaults preserve the last evidence-backed primitive proportions while
    making the three business categories explicit.  They are the V6
    calibration baseline, not a terminal-issued policy.
    """

    spatial_concentration_weight: float = 0.6250
    berth_transport_weight: float = 0.1625
    reserved_capacity_efficiency_weight: float = 0.2125

    zone_dispersion_share: float = 0.56
    voyage_area_dispersion_share: float = 0.24
    existing_group_proximity_share: float = 0.20

    peak_utilization_headroom_fraction: float = 0.50

    def validate(self) -> None:
        category_weights = self.category_weights()
        concentration_shares = self.concentration_shares()
        for label, values in (
            ("category weights", category_weights),
            ("spatial-concentration shares", concentration_shares),
        ):
            if any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in values.values()
            ):
                raise ValueError(f"V6 {label} must be finite and nonnegative: {values}")
            if abs(sum(values.values()) - 1.0) > 1e-9:
                raise ValueError(f"V6 {label} must sum to one: {values}")
        headroom = float(self.peak_utilization_headroom_fraction)
        if not math.isfinite(headroom) or not 0.0 <= headroom <= 1.0:
            raise ValueError(
                "V6 peak-utilization headroom fraction must lie in [0, 1]"
            )

    def category_weights(self) -> dict[str, float]:
        return {
            "spatial_concentration": float(self.spatial_concentration_weight),
            "berth_transport": float(self.berth_transport_weight),
            "reserved_capacity_efficiency": float(
                self.reserved_capacity_efficiency_weight
            ),
        }

    def concentration_shares(self) -> dict[str, float]:
        return {
            "zone_dispersion": float(self.zone_dispersion_share),
            "voyage_area_dispersion": float(
                self.voyage_area_dispersion_share
            ),
            "existing_group_proximity": float(
                self.existing_group_proximity_share
            ),
        }

    def primitive_weights(self) -> dict[str, float]:
        category = self.category_weights()
        shares = self.concentration_shares()
        concentration = category["spatial_concentration"]
        return {
            "zone_dispersion": concentration * shares["zone_dispersion"],
            "voyage_area_dispersion": concentration
            * shares["voyage_area_dispersion"],
            "existing_group_proximity": concentration
            * shares["existing_group_proximity"],
            "berth_distance": category["berth_transport"],
            "unused_capacity": category["reserved_capacity_efficiency"],
        }

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "objective_version": V6_OBJECTIVE_VERSION,
            "aggregation": "normalized_weighted_sum_with_three_categories",
            "category_weights": self.category_weights(),
            "spatial_concentration_shares": self.concentration_shares(),
            "primitive_weights": self.primitive_weights(),
            "peak_utilization": {
                "role": "epsilon_constraint_not_objective",
                "headroom_fraction": float(
                    self.peak_utilization_headroom_fraction
                ),
            },
        }


@dataclass(frozen=True)
class V6ObjectiveScales:
    zone_dispersion: float
    voyage_area_dispersion: float
    existing_group_proximity: float
    berth_distance: float
    unused_capacity: float

    def as_dict(self) -> dict[str, float]:
        values = {
            "zone_dispersion": float(self.zone_dispersion),
            "voyage_area_dispersion": float(self.voyage_area_dispersion),
            "existing_group_proximity": float(self.existing_group_proximity),
            "berth_distance": float(self.berth_distance),
            "unused_capacity": float(self.unused_capacity),
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
            raise ValueError(f"V6 objective scales must be positive and finite: {values}")
        return values


@dataclass(frozen=True)
class V6PeakUtilizationPolicy:
    """A reproducible reference utilization and its epsilon headroom.

    ``minimum_feasible_utilization`` is retained as a serialized compatibility
    field.  For the production analytic policy it stores a valid workload
    lower bound, not a claimed optimal min-max value.  ``minimum_source`` and
    ``reference_role`` make that distinction explicit.
    """

    minimum_feasible_utilization: float
    headroom_fraction: float
    minimum_source: str = "auxiliary_full_v6_minmax_mip"

    def __post_init__(self) -> None:
        minimum = float(self.minimum_feasible_utilization)
        headroom = float(self.headroom_fraction)
        if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
            raise ValueError("minimum feasible V6 utilization must lie in [0, 1]")
        if not math.isfinite(headroom) or not 0.0 <= headroom <= 1.0:
            raise ValueError("V6 utilization headroom must lie in [0, 1]")
        if not str(self.minimum_source).strip():
            raise ValueError("V6 utilization minimum source must be declared")

    @property
    def epsilon_cap(self) -> float:
        minimum = float(self.minimum_feasible_utilization)
        return minimum + float(self.headroom_fraction) * (1.0 - minimum)

    @property
    def reference_role(self) -> str:
        return (
            "proven_minimum_feasible_utilization"
            if "minmax" in str(self.minimum_source)
            else "analytic_workload_lower_bound"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "type": "data_derived_epsilon_constraint",
            "minimum_source": str(self.minimum_source),
            "reference_role": self.reference_role,
            "reference_utilization": float(
                self.minimum_feasible_utilization
            ),
            "minimum_feasible_utilization": float(
                self.minimum_feasible_utilization
            ),
            "minimum_feasible_utilization_proven": (
                self.reference_role
                == "proven_minimum_feasible_utilization"
            ),
            "headroom_fraction": float(self.headroom_fraction),
            "epsilon_cap": float(self.epsilon_cap),
            "terminal_approved_threshold_used": False,
        }


def v6_export_group_key(group: ExportGroup) -> tuple[str, ...]:
    """Return the fixed serialized identity used by V6 incumbent anchors."""

    values = {
        "IYC_CSZ_CSIZECD": str(group.size),
        "IYC_POT_UNLDPORT": str(group.port),
        "IYC_CHEIGHTCD": str(group.height),
    }
    return (
        str(group.voyage_id),
        f"flow={group.status}",
        *(
            f"{attribute}={values[attribute]}"
            for attribute in EXPORT_GROUP_IDENTITY_ATTRIBUTES
        ),
    )


def v6_import_reservation_capacity(
    problem: ProblemData,
    flow: str,
    size: str,
    bay_key: str,
) -> int:
    """Return one legal anonymous-import anchor capacity, or zero."""

    bay = problem.bays.get(str(bay_key))
    if bay is None or str(size) not in {"20", "40"}:
        return 0
    if str(flow) not in problem.area_functions.get(str(bay.area_no), set()):
        return 0
    footprint = _footprint(problem, str(bay_key), str(size))
    if not footprint or any(
        str(problem.bays[key].area_no) != str(bay.area_no)
        for key in footprint
    ):
        return 0
    for footprint_key in footprint:
        existing_sizes = {
            str(value)
            for value in problem.bays[footprint_key].existing_size_modes
            if str(value)
        }
        if existing_sizes and existing_sizes != {str(size)}:
            return 0
    return max(
        0,
        min(
            int(bay.cap_by_size.get(str(size), 0)),
            *(int(problem.bays[key].physical_capacity) for key in footprint),
        ),
    )


def build_v6_import_candidates(
    problem: ProblemData,
) -> dict[tuple[str, str], tuple[tuple[str, int], ...]]:
    """Build every legal anonymous-import anchor for positive V6 demand."""

    output: dict[tuple[str, str], tuple[tuple[str, int], ...]] = {}
    ordered_bays = sorted(
        problem.bays,
        key=lambda key: (
            str(problem.bays[key].area_no),
            int(problem.bays[key].bay_order),
            str(key),
        ),
    )
    for raw_key, quantity in sorted(problem.import_demand_by_flow_size.items()):
        if int(quantity) <= 0:
            continue
        flow, size = map(str, raw_key)
        candidates = tuple(
            (bay_key, capacity)
            for bay_key in ordered_bays
            if (
                capacity := v6_import_reservation_capacity(
                    problem,
                    flow,
                    size,
                    bay_key,
                )
            )
            > 0
        )
        if not candidates:
            raise ValueError(
                "V6 anonymous import demand has no compatible bay: "
                f"flow={flow}, size={size}, demand={quantity}"
            )
        output[(flow, size)] = candidates
    return output


def derive_v6_analytic_peak_policy(
    problem: ProblemData,
    objective_config: V6ObjectiveConfig | None = None,
    *,
    atoms: Sequence[RowAwareBayAtom] | None = None,
) -> tuple[V6PeakUtilizationPolicy, dict[str, object]]:
    """Derive the production epsilon cap without solving a min-max MIP.

    This is the V6-native counterpart of the V5 workload/capacity policy.  It
    uses only reachability implied by legal V6 row atoms and anonymous-import
    candidates.  The result is a valid aggregate lower bound, not ``rho*``;
    a separate compact feasibility solve certifies the resulting cap.
    """

    config = objective_config or V6ObjectiveConfig()
    config.validate()
    if problem.area_guidance_target or problem.import_area_size_reference:
        raise ValueError("V6 analytic peak policy does not accept upstream area targets")
    if atoms is None:
        atoms, _anchor_limits = build_v6_row_aware_bay_atoms(problem)
    atoms = tuple(atoms)
    groups = tuple(
        sorted(
            (
                group
                for group in problem.export_groups
                if str(group.status) == "OF" and int(group.demand) > 0
            ),
            key=lambda group: str(group.group_id),
        )
    )
    groups_by_id = {str(group.group_id): group for group in groups}
    export_areas_by_group: defaultdict[str, set[str]] = defaultdict(set)
    export_areas_by_voyage: defaultdict[str, set[str]] = defaultdict(set)
    for atom in atoms:
        export_areas_by_group[str(atom.group_id)].add(str(atom.area_no))
        group = groups_by_id.get(str(atom.group_id))
        if group is not None:
            export_areas_by_voyage[str(group.voyage_id)].add(str(atom.area_no))
    missing_groups = [
        group.group_id
        for group in groups
        if not export_areas_by_group.get(str(group.group_id))
    ]
    if missing_groups:
        raise ValueError(
            f"V6 analytic peak policy has unreachable export groups: {missing_groups}"
        )

    import_candidates = build_v6_import_candidates(problem)
    import_areas_by_flow_size = {
        key: {
            str(problem.bays[bay_key].area_no)
            for bay_key, _capacity in candidates
        }
        for key, candidates in import_candidates.items()
    }
    area_capacity: Counter[str] = Counter()
    for bay in problem.bays.values():
        area_capacity[str(bay.area_no)] += int(bay.physical_capacity)

    def slot_units(size: str) -> int:
        if str(size) == "20":
            return 1
        if str(size) in {"40", "45"}:
            return 2
        raise ValueError(f"unsupported V6 size in peak policy: {size}")

    def capacity_of(areas: set[str]) -> int:
        return int(sum(area_capacity.get(area, 0) for area in areas))

    def discrete_load_floor(load: int, areas: set[str]) -> float:
        capacities = [
            int(area_capacity.get(area, 0))
            for area in sorted(areas)
            if int(area_capacity.get(area, 0)) > 0
        ]
        if not capacities or sum(capacities) < int(load):
            return math.inf
        lower = 0.0
        upper = 1.0
        for _iteration in range(60):
            midpoint = (lower + upper) / 2.0
            usable = sum(
                int(math.floor(midpoint * capacity + 1e-12))
                for capacity in capacities
            )
            if usable >= int(load):
                upper = midpoint
            else:
                lower = midpoint
        return min(1.0, upper + 1e-9)

    components: dict[str, float] = {}

    def add_component(name: str, load: int, areas: set[str]) -> None:
        if int(load) <= 0:
            return
        capacity = capacity_of(areas)
        if capacity <= 0:
            raise ValueError(
                "positive V6 workload has no reachable residual capacity: "
                f"scope={name}, load={load}"
            )
        components[name] = max(
            float(load) / float(capacity),
            discrete_load_floor(int(load), areas),
        )

    export_load = sum(
        int(group.demand) * slot_units(group.size) for group in groups
    )
    import_load = sum(
        int(quantity) * slot_units(str(size))
        for (_flow, size), quantity in problem.import_demand_by_flow_size.items()
        if int(quantity) > 0
    )
    export_areas = set().union(*export_areas_by_group.values()) if groups else set()
    import_areas = (
        set().union(*import_areas_by_flow_size.values())
        if import_areas_by_flow_size
        else set()
    )
    add_component(
        "all_workload",
        export_load + import_load,
        export_areas | import_areas,
    )
    add_component("all_exports", export_load, export_areas)
    add_component("all_imports", import_load, import_areas)
    for voyage_id, areas in sorted(export_areas_by_voyage.items()):
        load = sum(
            int(group.demand) * slot_units(group.size)
            for group in groups
            if str(group.voyage_id) == voyage_id
        )
        add_component(f"export_voyage:{voyage_id}", load, areas)
    for group in groups:
        add_component(
            f"export_group:{group.group_id}",
            int(group.demand) * slot_units(group.size),
            export_areas_by_group[str(group.group_id)],
        )
    for raw_key, quantity in sorted(problem.import_demand_by_flow_size.items()):
        if int(quantity) <= 0:
            continue
        flow, size = map(str, raw_key)
        add_component(
            f"import_flow_size:{flow}|{size}",
            int(quantity) * slot_units(size),
            import_areas_by_flow_size[(flow, size)],
        )

    lower_bound = max(components.values(), default=0.0)
    if not math.isfinite(lower_bound) or lower_bound > 1.0 + 1e-9:
        raise ValueError(
            "V6 workload exceeds analytically reachable residual capacity: "
            f"lower_bound={lower_bound}"
        )
    lower_bound = min(1.0, max(0.0, float(lower_bound)))
    policy = V6PeakUtilizationPolicy(
        minimum_feasible_utilization=lower_bound,
        headroom_fraction=float(
            config.peak_utilization_headroom_fraction
        ),
        minimum_source="v6_analytic_reachable_capacity_lower_bound",
    )
    binding = [
        name
        for name, value in sorted(components.items())
        if math.isclose(value, lower_bound, rel_tol=1e-10, abs_tol=1e-10)
    ]
    return policy, {
        "method": "v6_analytic_reachable_capacity_lower_bound",
        "reference_role": "analytic_workload_lower_bound",
        "load_lower_bound": lower_bound,
        "headroom_fraction": float(config.peak_utilization_headroom_fraction),
        "epsilon_cap": float(policy.epsilon_cap),
        "lower_bound_components": dict(sorted(components.items())),
        "binding_lower_bound_scopes": binding,
        "export_slot_units": int(export_load),
        "anonymous_import_slot_units": int(import_load),
        "reachable_area_capacity": capacity_of(export_areas | import_areas),
        "area_capacity": dict(sorted(area_capacity.items())),
        "integer_area_capacity_breakpoints_included": True,
        "terminal_approved_threshold_used": False,
        "minmax_mip_solved": False,
    }


def _row_capacity(
    problem: ProblemData,
    anchor_bay_key: str,
    row_no: str,
    size: str,
) -> int:
    """Return the exact residual capacity of one V6 anchor-bay row atom."""

    footprint = _footprint(problem, anchor_bay_key, size)
    if not footprint:
        return 0
    capacities: list[int] = [
        int(problem.bays[anchor_bay_key].cap_by_size.get(str(size), 0)),
        *(int(problem.bays[key].physical_capacity) for key in footprint),
    ]
    for footprint_key in footprint:
        bay = problem.bays[footprint_key]
        physical = bay.row_physical_capacity.get(str(row_no))
        sized = bay.row_cap_by_size.get(str(size), {}).get(str(row_no))
        if physical is None or sized is None:
            return 0
        capacities.append(min(int(physical), int(sized)))
    return max(0, min(capacities, default=0))


def v6_model_contract(
    objective_config: V6ObjectiveConfig | None = None,
) -> dict[str, object]:
    """Return solver-independent V6 decision and constraint metadata."""

    config = objective_config or V6ObjectiveConfig()
    config.validate()
    return {
        "model_schema_version": V6_MODEL_SCHEMA_VERSION,
        "export_group_identity": [
            "voyage",
            "size",
            "height",
            "discharge_port",
        ],
        "decision_families": {
            "zone_selection": "binary by group-specific row-aware zone",
            "zone_bay_flow": "nonnegative integer boxes by selected zone and anchor bay",
            "anonymous_import_reservation": "nonnegative integer boxes by flow, size, and anchor bay",
            "peak_utilization": "continuous auxiliary value in the min-max reference MIP only",
        },
        "hard_constraint_families": [
            "exact_export_group_demand",
            "positive_flow_in_every_selected_zone_bay",
            "zone_bay_flow_capacity",
            "physical_row_single_export_group",
            "same_group_zone_bay_nonoverlap",
            "bay_export_size_and_height_consistency",
            "existing_bay_and_row_state_nonworsening_compatibility",
            "area_function_compatibility",
            "20_40_45_footprint_consistency",
            "45ft_edge_large_bay_eligibility",
            "exact_anonymous_import_flow_size_demand",
            "anonymous_import_size_and_physical_capacity",
            "anonymous_import_single_size_per_physical_bay",
            "export_import_physical_bay_exclusivity",
            "data_derived_peak_utilization_epsilon_cap",
        ],
        "existing_state_policy": {
            "size": "strict_single_state_compatibility",
            "height": "new_height_must_belong_to_existing_set_and_new_boxes_use_one_height",
            "row_group": "new_export_group_must_belong_to_exact_existing_row_group_set_and_new_boxes_use_one_group",
        },
        "forbidden_inputs_or_terms": [
            "upstream_large_plan_area_target",
            "upstream_import_area_reference",
            "forecast_export_demand",
            "tops_plan",
            "group_area_target",
            "fixed_row_strip_zone",
            "row_number_continuity",
            "demand_plus_one_atom_zone_capacity_cap",
            "shortage_variable",
            "row_dispersion_objective",
        ],
        "objective": config.as_dict(),
    }


class V6ModelEvaluator:
    """Independently validate and score a complete V6 zone-flow decision."""

    def __init__(
        self,
        problem: ProblemData,
        zones: Sequence[RowAwareZone],
        objective_config: V6ObjectiveConfig | None = None,
        candidate_bays_by_group: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self.problem = problem
        self.config = objective_config or V6ObjectiveConfig()
        self.config.validate()
        self.groups = tuple(
            sorted(
                (
                    group
                    for group in problem.export_groups
                    if str(group.status) == "OF" and int(group.demand) > 0
                ),
                key=lambda group: str(group.group_id),
            )
        )
        self.groups_by_id = {group.group_id: group for group in self.groups}
        self.zones = tuple(zones)
        self.zones_by_id = {zone.zone_id: zone for zone in self.zones}
        self._validate_model_input()
        self.import_candidates = build_v6_import_candidates(problem)
        self._candidate_areas_by_group: defaultdict[str, set[str]] = defaultdict(set)
        self._candidate_bays_by_group: defaultdict[str, set[str]] = defaultdict(set)
        for group_id, bay_keys in (candidate_bays_by_group or {}).items():
            if str(group_id) not in self.groups_by_id:
                raise ValueError(
                    f"V6 objective candidates reference unknown group: {group_id}"
                )
            for raw_bay_key in bay_keys:
                bay_key = str(raw_bay_key)
                bay = self.problem.bays.get(bay_key)
                if bay is None:
                    raise ValueError(
                        f"V6 objective candidates reference unknown bay: {bay_key}"
                    )
                self._candidate_bays_by_group[str(group_id)].add(bay_key)
                self._candidate_areas_by_group[str(group_id)].add(
                    str(bay.area_no)
                )
        for zone in self.zones:
            self._candidate_areas_by_group[zone.group_id].add(zone.area_no)
            self._candidate_bays_by_group[zone.group_id].update(
                zone.anchor_bay_keys
            )
        self._anchors_by_group_area = self._existing_anchors()
        self._reachable_anchor_groups = self._find_reachable_anchor_groups()
        self._berth_bounds = self._prepare_berth_bounds()
        self.scales = self._derive_scales()

    def _validate_model_input(self) -> None:
        if self.problem.area_guidance_target:
            raise ValueError("V6 does not accept upstream large-plan area targets")
        if self.problem.import_area_size_reference:
            raise ValueError("V6 requires direct anonymous import demand, not area references")
        for (flow, size), quantity in (
            self.problem.import_demand_by_flow_size.items()
        ):
            if str(size) not in {"20", "40"}:
                raise ValueError(
                    "V6 anonymous import size must be 20 or physical 40 "
                    f"(45 must be pre-aggregated): flow={flow}, size={size}"
                )
            if int(quantity) <= 0 or float(quantity) != float(int(quantity)):
                raise ValueError(
                    f"V6 anonymous import demand must be a positive integer: "
                    f"flow={flow}, size={size}, quantity={quantity}"
                )
        if len(self.groups_by_id) != len(self.groups):
            raise ValueError("V6 export group_id values must be unique")
        identities: dict[tuple[str, ...], str] = {}
        target_voyages = {str(value) for value in self.problem.target_voyages}
        export_voyages = (
            {str(value) for value in self.problem.export_voyages}
            if self.problem.export_voyages is not None
            else None
        )
        for group in self.groups:
            if target_voyages and str(group.voyage_id) not in target_voyages:
                raise ValueError(
                    f"V6 export group is outside target voyages: {group.group_id}"
                )
            if (
                export_voyages is not None
                and str(group.voyage_id) not in export_voyages
            ):
                raise ValueError(
                    f"V6 export group is not classified as export: {group.group_id}"
                )
            key = v6_export_group_key(group)
            if key in identities:
                raise ValueError(
                    "V6 export demand must be aggregated by voyage, size, "
                    "height, and discharge port: "
                    f"groups={identities[key]},{group.group_id}"
                )
            identities[key] = group.group_id
        if len(self.zones_by_id) != len(self.zones):
            raise ValueError("V6 zone_id values must be unique")
        edge_large_bays = _edge_large_bays(self.problem)
        for zone in self.zones:
            group = self.groups_by_id.get(zone.group_id)
            if group is None:
                raise ValueError(f"V6 zone references unknown group: {zone.group_id}")
            if not zone.anchor_bay_keys:
                raise ValueError(f"V6 zone has no anchor bay: {zone.zone_id}")
            if len(zone.anchor_bay_keys) != len(set(zone.anchor_bay_keys)):
                raise ValueError(f"V6 zone repeats an anchor bay: {zone.zone_id}")
            capacity_by_bay = dict(zone.anchor_bay_capacities)
            if set(capacity_by_bay) != set(zone.anchor_bay_keys) or len(
                capacity_by_bay
            ) != len(zone.anchor_bay_capacities):
                raise ValueError(
                    f"V6 zone bay-capacity keys do not match its interval: {zone.zone_id}"
                )
            if any(int(value) <= 0 for _key, value in zone.anchor_bay_capacities):
                raise ValueError(f"V6 zone has nonpositive bay capacity: {zone.zone_id}")
            if sum(value for _key, value in zone.anchor_bay_capacities) != int(
                zone.capacity
            ):
                raise ValueError(f"V6 zone capacity is inconsistent: {zone.zone_id}")
            if len(zone.resources) != len(set(zone.resources)):
                raise ValueError(f"V6 zone repeats a physical row: {zone.zone_id}")
            if set(zone.physical_bay_keys) != {
                bay_key for bay_key, _row_no in zone.resources
            }:
                raise ValueError(f"V6 zone footprint is inconsistent: {zone.zone_id}")
            if "OF" not in self.problem.area_functions.get(zone.area_no, set()):
                raise ValueError(f"V6 zone violates export area function: {zone.zone_id}")
            expected_resources: set[tuple[str, str]] = set()
            row_map = dict(zone.rows_by_anchor_bay)
            if set(row_map) != set(zone.anchor_bay_keys) or len(row_map) != len(
                zone.rows_by_anchor_bay
            ):
                raise ValueError(f"V6 zone row-map keys are invalid: {zone.zone_id}")
            footprints: list[tuple[int, ...]] = []
            for bay_key, rows in zone.rows_by_anchor_bay:
                if not rows or len(rows) != len(set(rows)):
                    raise ValueError(f"V6 zone row map is invalid: {zone.zone_id}")
                bay = self.problem.bays.get(bay_key)
                if bay is None or str(bay.area_no) != str(zone.area_no):
                    raise ValueError(f"V6 zone bay is invalid: {zone.zone_id}, {bay_key}")
                footprint = _footprint(self.problem, bay_key, group.size)
                if not footprint:
                    raise ValueError(f"V6 zone has an invalid size footprint: {zone.zone_id}")
                if any(
                    str(self.problem.bays[key].area_no) != str(zone.area_no)
                    for key in footprint
                ):
                    raise ValueError(f"V6 zone crosses yard areas: {zone.zone_id}")
                footprints.append(
                    tuple(int(self.problem.bays[key].bay_order) for key in footprint)
                )
                if group.size == "45" and bay_key not in edge_large_bays:
                    raise ValueError(f"V6 45-ft zone is not at an edge bay: {zone.zone_id}")
                expected_capacity = sum(
                    _row_capacity(self.problem, bay_key, row_no, group.size)
                    for row_no in rows
                )
                if expected_capacity <= 0 or expected_capacity != int(
                    capacity_by_bay[bay_key]
                ):
                    raise ValueError(
                        "V6 zone anchor-bay row capacity is invalid: "
                        f"zone={zone.zone_id}, bay={bay_key}"
                    )
                for row_no in rows:
                    expected_resources.update(
                        (footprint_key, str(row_no)) for footprint_key in footprint
                    )
            if footprints != sorted(footprints):
                raise ValueError(f"V6 zone anchor bays are not ordered: {zone.zone_id}")
            if any(
                current[0] - previous[-1] != 2
                for previous, current in zip(footprints, footprints[1:])
            ):
                raise ValueError(f"V6 zone anchor bays are not contiguous: {zone.zone_id}")
            if expected_resources != set(zone.resources):
                raise ValueError(f"V6 zone row footprint is invalid: {zone.zone_id}")
            self._validate_zone_against_existing_state(zone, group)

    def _validate_zone_against_existing_state(
        self,
        zone: RowAwareZone,
        group: ExportGroup,
    ) -> None:
        for bay_key in zone.physical_bay_keys:
            bay = self.problem.bays[bay_key]
            sizes = {str(value) for value in bay.existing_size_modes if str(value)}
            heights = {str(value) for value in bay.existing_heights if str(value)}
            if sizes and sizes != {str(group.size)}:
                raise ValueError(
                    f"V6 zone conflicts with existing bay size: zone={zone.zone_id}, bay={bay_key}"
                )
            if heights and str(group.height) not in heights:
                raise ValueError(
                    f"V6 zone conflicts with existing bay height: zone={zone.zone_id}, bay={bay_key}"
                )
        for bay_key, row_no in zone.resources:
            bay = self.problem.bays[bay_key]
            exact_groups = {
                tuple(str(part) for part in value)
                for value in bay.existing_group_keys_by_row.get(row_no, set())
            }
            ports = {str(value) for value in bay.existing_ports_by_row.get(row_no, set())}
            voyages = {
                str(value)
                for value in bay.existing_attrs_by_row.get(row_no, {}).get(
                    EXPORT_VOYAGE_ROW_NO_MIX_ATTR,
                    set(),
                )
            }
            if exact_groups:
                if existing_export_group_key(group) not in exact_groups:
                    raise ValueError(
                        "V6 zone would add a new group to an existing row: "
                        f"zone={zone.zone_id}, resource={(bay_key, row_no)}"
                    )
            else:
                if ports and str(group.port) not in ports:
                    raise ValueError(
                        f"V6 zone conflicts with existing row port: zone={zone.zone_id}, "
                        f"resource={(bay_key, row_no)}"
                    )
                if voyages and str(group.voyage_id) not in voyages:
                    raise ValueError(
                        f"V6 zone conflicts with existing row voyage: zone={zone.zone_id}, "
                        f"resource={(bay_key, row_no)}"
                    )

    def _existing_anchors(self) -> defaultdict[tuple[str, str], set[str]]:
        anchors: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        for group in self.groups:
            prefix = v6_export_group_key(group)
            for key, quantity in self.problem.existing_group_bay_load.items():
                if int(quantity) <= 0 or tuple(key[:-2]) != prefix:
                    continue
                area_no, bay_key = str(key[-2]), str(key[-1])
                if bay_key in self.problem.bays:
                    anchors[(group.group_id, area_no)].add(bay_key)
        return anchors

    def _find_reachable_anchor_groups(self) -> set[str]:
        return {
            group.group_id
            for group in self.groups
            if any(
                self._anchors_by_group_area.get((group.group_id, area_no))
                for area_no in self._candidate_areas_by_group[group.group_id]
            )
        }

    def _prepare_berth_bounds(self) -> dict[str, tuple[float, float]]:
        bounds: dict[str, tuple[float, float]] = {}
        for group in self.groups:
            berth = self.problem.berth_by_voyage.get(group.voyage_id)
            if not berth:
                raise ValueError(f"V6 is missing berth for voyage {group.voyage_id}")
            distances: list[float] = []
            for area_no in sorted(self._candidate_areas_by_group[group.group_id]):
                value = self.problem.berth_distances.get((area_no, berth))
                if (
                    value is None
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                ):
                    raise ValueError(
                        f"V6 is missing berth distance: voyage={group.voyage_id}, "
                        f"berth={berth}, area={area_no}"
                    )
                distances.append(float(value))
            if not distances:
                raise ValueError(f"V6 group has no candidate zone: {group.group_id}")
            current = bounds.get(group.voyage_id)
            lower, upper = min(distances), max(distances)
            if current is None:
                bounds[group.voyage_id] = (lower, upper)
            else:
                bounds[group.voyage_id] = (
                    min(current[0], lower),
                    max(current[1], upper),
                )
        return bounds

    def _derive_scales(self) -> V6ObjectiveScales:
        demand_by_voyage: Counter[str] = Counter()
        areas_by_voyage: defaultdict[str, set[str]] = defaultdict(set)
        for group in self.groups:
            demand_by_voyage[group.voyage_id] += int(group.demand)
            areas_by_voyage[group.voyage_id].update(
                self._candidate_areas_by_group[group.group_id]
            )
        voyage_scale = sum(
            max(0, min(int(demand), len(areas_by_voyage[voyage_id])) - 1)
            for voyage_id, demand in demand_by_voyage.items()
        )
        zone_scale = sum(
            max(
                0,
                min(
                    int(group.demand),
                    len(self._candidate_bays_by_group[group.group_id]),
                )
                - 1,
            )
            for group in self.groups
        )
        total_demand = sum(int(group.demand) for group in self.groups)
        anchored_demand = sum(
            int(group.demand)
            for group in self.groups
            if group.group_id in self._reachable_anchor_groups
        )
        return V6ObjectiveScales(
            zone_dispersion=float(max(1, zone_scale)),
            voyage_area_dispersion=float(max(1, voyage_scale)),
            existing_group_proximity=float(max(1, anchored_demand)),
            berth_distance=float(max(1, total_demand)),
            unused_capacity=float(max(1, total_demand)),
        )

    def _normalized_existing_proximity(
        self,
        group: ExportGroup,
        bay_key: str,
    ) -> float:
        if group.group_id not in self._reachable_anchor_groups:
            return 0.0
        bay = self.problem.bays[bay_key]
        anchors = self._anchors_by_group_area.get(
            (group.group_id, str(bay.area_no)),
            set(),
        )
        if not anchors:
            return 1.0
        distance = min(
            abs(int(self.problem.bays[key].bay_order) - int(bay.bay_order))
            for key in anchors
        )
        area_orders = [
            int(candidate.bay_order)
            for candidate in self.problem.bays.values()
            if str(candidate.area_no) == str(bay.area_no)
        ]
        span = max(area_orders) - min(area_orders) if area_orders else 0
        return 0.0 if span <= 0 else min(1.0, float(distance) / float(span))

    def _normalized_berth_distance(self, voyage_id: str, area_no: str) -> float:
        berth = self.problem.berth_by_voyage[voyage_id]
        value = float(self.problem.berth_distances[(area_no, berth)])
        lower, upper = self._berth_bounds[voyage_id]
        return 0.0 if upper <= lower else (value - lower) / (upper - lower)

    def zone_selection_objective_coefficient(
        self,
        zone: RowAwareZone,
    ) -> float:
        """Return the complete V6 objective coefficient of ``x_z``."""

        weights = self.config.primitive_weights()
        scales = self.scales.as_dict()
        return (
            weights["zone_dispersion"] / scales["zone_dispersion"]
            + weights["unused_capacity"]
            * int(zone.capacity)
            / scales["unused_capacity"]
        )

    def zone_bay_flow_objective_coefficient(
        self,
        zone: RowAwareZone,
        bay_key: str,
    ) -> float:
        """Return the complete V6 objective coefficient of ``q_zb``."""

        if str(bay_key) not in zone.anchor_bay_keys:
            raise ValueError(
                f"bay {bay_key} is not in V6 zone {zone.zone_id}"
            )
        return self.group_bay_flow_objective_coefficient(
            zone.group_id,
            str(bay_key),
        )

    def group_bay_flow_objective_coefficient(
        self,
        group_id: str,
        bay_key: str,
    ) -> float:
        """Return the V6 objective coefficient of global ``q[group,bay]``."""

        group = self.groups_by_id[str(group_id)]
        bay = self.problem.bays[str(bay_key)]
        weights = self.config.primitive_weights()
        scales = self.scales.as_dict()
        return (
            weights["existing_group_proximity"]
            * self._normalized_existing_proximity(group, str(bay_key))
            / scales["existing_group_proximity"]
            + weights["berth_distance"]
            * self._normalized_berth_distance(
                group.voyage_id,
                str(bay.area_no),
            )
            / scales["berth_distance"]
            - weights["unused_capacity"] / scales["unused_capacity"]
        )

    def zone_fixed_activation_objective_coefficient(self) -> float:
        weights = self.config.primitive_weights()
        scales = self.scales.as_dict()
        return weights["zone_dispersion"] / scales["zone_dispersion"]

    def unused_capacity_unit_objective_coefficient(self) -> float:
        weights = self.config.primitive_weights()
        scales = self.scales.as_dict()
        return weights["unused_capacity"] / scales["unused_capacity"]

    def voyage_area_objective_coefficient(self) -> float:
        weights = self.config.primitive_weights()
        scales = self.scales.as_dict()
        return (
            weights["voyage_area_dispersion"]
            / scales["voyage_area_dispersion"]
        )

    def objective_constant(self) -> float:
        """Remove the unavoidable first zone and first area activations."""

        weights = self.config.primitive_weights()
        scales = self.scales.as_dict()
        voyage_count = len({group.voyage_id for group in self.groups})
        return -(
            len(self.groups)
            * weights["zone_dispersion"]
            / scales["zone_dispersion"]
            + voyage_count
            * weights["voyage_area_dispersion"]
            / scales["voyage_area_dispersion"]
        )

    def evaluate(
        self,
        selected_zone_ids: Iterable[int],
        zone_bay_flow: Mapping[ZoneBayFlowKey, int],
        import_reservation: Mapping[ImportReservationKey, int],
        peak_policy: V6PeakUtilizationPolicy,
    ) -> dict[str, object]:
        if abs(
            float(peak_policy.headroom_fraction)
            - float(self.config.peak_utilization_headroom_fraction)
        ) > 1e-12:
            raise ValueError(
                "V6 peak policy headroom differs from the objective/model config"
            )
        selected = {int(zone_id) for zone_id in selected_zone_ids}
        errors: list[str] = []
        unknown = sorted(selected - set(self.zones_by_id))
        if unknown:
            errors.append(f"unknown selected V6 zones: {unknown[:10]}")

        normalized_flow: Counter[ZoneBayFlowKey] = Counter()
        for raw_key, raw_quantity in zone_bay_flow.items():
            key = (int(raw_key[0]), str(raw_key[1]))
            quantity = int(raw_quantity)
            if quantity < 0 or float(raw_quantity) != float(quantity):
                errors.append(f"invalid V6 zone-bay flow: key={key}, value={raw_quantity}")
                continue
            if quantity > 0:
                normalized_flow[key] += quantity

        used_resources: dict[tuple[str, str], int] = {}
        group_bay_zones: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        export_groups_by_physical_bay: defaultdict[str, set[str]] = defaultdict(set)
        reserved_export_by_physical_bay: Counter[str] = Counter()
        reserved_export_by_anchor_size: Counter[tuple[str, str]] = Counter()
        assigned_by_group: Counter[str] = Counter()
        used_areas_by_voyage: defaultdict[str, set[str]] = defaultdict(set)
        selected_count_by_group: Counter[str] = Counter()
        planned_slot_load_by_area: Counter[str] = Counter()
        reserved_capacity = 0
        proximity_sum = 0.0
        berth_sum = 0.0

        for zone_id in sorted(selected & set(self.zones_by_id)):
            zone = self.zones_by_id[zone_id]
            group = self.groups_by_id[zone.group_id]
            selected_count_by_group[group.group_id] += 1
            reserved_capacity += int(zone.capacity)
            for resource in zone.resources:
                previous = used_resources.get(resource)
                if previous is not None:
                    errors.append(
                        f"physical row is selected by multiple zones: "
                        f"resource={resource}, zones={previous},{zone_id}"
                    )
                used_resources[resource] = zone_id
            for bay_key in zone.physical_bay_keys:
                export_groups_by_physical_bay[bay_key].add(group.group_id)
            capacity_by_bay = dict(zone.anchor_bay_capacities)
            for bay_key in zone.anchor_bay_keys:
                group_bay_zones[(group.group_id, bay_key)].append(zone_id)
                anchor_capacity = int(capacity_by_bay[bay_key])
                reserved_export_by_anchor_size[(bay_key, group.size)] += (
                    anchor_capacity
                )
                for footprint_key in _footprint(
                    self.problem,
                    bay_key,
                    group.size,
                ):
                    reserved_export_by_physical_bay[footprint_key] += (
                        anchor_capacity
                    )
                quantity = int(normalized_flow.get((zone_id, bay_key), 0))
                if quantity < 1:
                    errors.append(
                        f"selected V6 zone must carry positive flow in every bay: "
                        f"zone={zone_id}, bay={bay_key}"
                    )
                if quantity > int(capacity_by_bay[bay_key]):
                    errors.append(
                        f"V6 zone-bay capacity exceeded: zone={zone_id}, bay={bay_key}"
                    )
                assigned_by_group[group.group_id] += quantity
                if quantity > 0:
                    used_areas_by_voyage[group.voyage_id].add(zone.area_no)
                    proximity_sum += quantity * self._normalized_existing_proximity(
                        group,
                        bay_key,
                    )
                    berth_sum += quantity * self._normalized_berth_distance(
                        group.voyage_id,
                        zone.area_no,
                    )
                    planned_slot_load_by_area[zone.area_no] += quantity * len(
                        _footprint(self.problem, bay_key, group.size)
                    )

        for (zone_id, bay_key), quantity in normalized_flow.items():
            zone = self.zones_by_id.get(zone_id)
            if zone is None or zone_id not in selected or bay_key not in zone.anchor_bay_keys:
                errors.append(
                    f"positive flow is not supported by a selected V6 zone: "
                    f"zone={zone_id}, bay={bay_key}, quantity={quantity}"
                )
        for key, zone_ids in group_bay_zones.items():
            if len(zone_ids) > 1:
                errors.append(
                    f"one group uses overlapping zones in one bay: key={key}, zones={zone_ids}"
                )
        for group in self.groups:
            if assigned_by_group[group.group_id] != int(group.demand):
                errors.append(
                    f"V6 export demand mismatch: group={group.group_id}, "
                    f"assigned={assigned_by_group[group.group_id]}, demand={group.demand}"
                )
            if selected_count_by_group[group.group_id] <= 0:
                errors.append(f"V6 group has no selected zone: {group.group_id}")

        for bay_key, group_ids in export_groups_by_physical_bay.items():
            sizes = {self.groups_by_id[group_id].size for group_id in group_ids}
            heights = {self.groups_by_id[group_id].height for group_id in group_ids}
            if len(sizes) > 1 or len(heights) > 1:
                errors.append(
                    f"V6 export bay size/height mixing: bay={bay_key}, "
                    f"sizes={sorted(sizes)}, heights={sorted(heights)}"
                )
        for bay_key, capacity in reserved_export_by_physical_bay.items():
            if capacity > int(self.problem.bays[bay_key].physical_capacity):
                errors.append(
                    f"V6 reserved export physical capacity exceeded: bay={bay_key}"
                )
        for (bay_key, size), capacity in reserved_export_by_anchor_size.items():
            if capacity > int(self.problem.bays[bay_key].cap_by_size.get(size, 0)):
                errors.append(
                    f"V6 reserved export size capacity exceeded: "
                    f"bay={bay_key}, size={size}"
                )

        import_by_flow_size: Counter[tuple[str, str]] = Counter()
        import_by_physical_bay: Counter[str] = Counter()
        import_by_anchor_size: Counter[tuple[str, str]] = Counter()
        import_sizes_by_physical_bay: defaultdict[str, set[str]] = defaultdict(set)
        import_used_bays: set[str] = set()
        for raw_key, raw_quantity in import_reservation.items():
            flow, size, bay_key = map(str, raw_key)
            quantity = int(raw_quantity)
            if quantity < 0 or float(raw_quantity) != float(quantity):
                errors.append(f"invalid anonymous import quantity: key={raw_key}")
                continue
            if quantity <= 0:
                continue
            bay = self.problem.bays.get(bay_key)
            footprint = _footprint(self.problem, bay_key, size)
            if bay is None or size not in {"20", "40"} or not footprint:
                errors.append(f"invalid anonymous import candidate: key={raw_key}")
                continue
            if flow not in self.problem.area_functions.get(bay.area_no, set()):
                errors.append(f"anonymous import violates area function: key={raw_key}")
            if int(bay.cap_by_size.get(size, 0)) <= 0:
                errors.append(f"anonymous import violates anchor size capacity: key={raw_key}")
            for footprint_key in footprint:
                existing = {
                    str(value)
                    for value in self.problem.bays[footprint_key].existing_size_modes
                    if str(value)
                }
                if existing and existing != {size}:
                    errors.append(
                        f"anonymous import conflicts with existing size: "
                        f"bay={footprint_key}, size={size}"
                    )
                import_by_physical_bay[footprint_key] += quantity
                import_sizes_by_physical_bay[footprint_key].add(size)
                import_used_bays.add(footprint_key)
            import_by_anchor_size[(bay_key, size)] += quantity
            import_by_flow_size[(flow, size)] += quantity
            planned_slot_load_by_area[bay.area_no] += quantity * len(footprint)

        required_import = Counter(
            {
                (str(flow), str(size)): int(quantity)
                for (flow, size), quantity in self.problem.import_demand_by_flow_size.items()
                if int(quantity) > 0
            }
        )
        for key in sorted(set(required_import) | set(import_by_flow_size)):
            if import_by_flow_size[key] != required_import[key]:
                errors.append(
                    f"anonymous import demand mismatch: key={key}, "
                    f"reserved={import_by_flow_size[key]}, required={required_import[key]}"
                )
        for bay_key, load in import_by_physical_bay.items():
            if load > int(self.problem.bays[bay_key].physical_capacity):
                errors.append(f"anonymous import physical capacity exceeded: bay={bay_key}")
            if len(import_sizes_by_physical_bay[bay_key]) > 1:
                errors.append(f"anonymous import size mixing: bay={bay_key}")
        for (bay_key, size), load in import_by_anchor_size.items():
            if load > int(self.problem.bays[bay_key].cap_by_size.get(size, 0)):
                errors.append(
                    f"anonymous import size capacity exceeded: bay={bay_key}, size={size}"
                )
        shared_bays = sorted(set(export_groups_by_physical_bay) & import_used_bays)
        if shared_bays:
            errors.append(f"V6 import and export share physical bays: {shared_bays}")

        area_capacity: Counter[str] = Counter()
        for bay in self.problem.bays.values():
            area_capacity[str(bay.area_no)] += int(bay.physical_capacity)
        utilization_by_area = {
            area_no: (
                float(planned_slot_load_by_area[area_no]) / float(capacity)
                if int(capacity) > 0
                else math.inf
            )
            for area_no, capacity in sorted(area_capacity.items())
            if int(planned_slot_load_by_area.get(area_no, 0)) > 0
        }
        maximum_utilization = max(utilization_by_area.values(), default=0.0)
        if maximum_utilization > peak_policy.epsilon_cap + 1e-9:
            errors.append(
                f"V6 peak-utilization cap exceeded: actual={maximum_utilization}, "
                f"cap={peak_policy.epsilon_cap}"
            )
        if errors:
            raise ValueError("V6 solution validation failed: " + "; ".join(errors[:20]))

        extra_zones = sum(
            max(0, selected_count_by_group[group.group_id] - 1)
            for group in self.groups
        )
        extra_voyage_areas = sum(
            max(0, len(used_areas_by_voyage[voyage_id]) - 1)
            for voyage_id in {group.voyage_id for group in self.groups}
        )
        assigned_boxes = sum(assigned_by_group.values())
        unused_capacity = reserved_capacity - assigned_boxes
        raw = {
            "extra_contiguous_zones": float(extra_zones),
            "extra_voyage_areas": float(extra_voyage_areas),
            "existing_group_normalized_distance_sum": float(proximity_sum),
            "berth_normalized_distance_sum": float(berth_sum),
            "unused_reserved_capacity_boxes": float(unused_capacity),
        }
        scale = self.scales.as_dict()
        normalized = {
            "zone_dispersion": raw["extra_contiguous_zones"]
            / scale["zone_dispersion"],
            "voyage_area_dispersion": raw["extra_voyage_areas"]
            / scale["voyage_area_dispersion"],
            "existing_group_proximity": raw[
                "existing_group_normalized_distance_sum"
            ]
            / scale["existing_group_proximity"],
            "berth_distance": raw["berth_normalized_distance_sum"]
            / scale["berth_distance"],
            "unused_capacity": raw["unused_reserved_capacity_boxes"]
            / scale["unused_capacity"],
        }
        shares = self.config.concentration_shares()
        category_scores = {
            "spatial_concentration": sum(
                shares[key] * normalized[key] for key in shares
            ),
            "berth_transport": normalized["berth_distance"],
            "reserved_capacity_efficiency": normalized["unused_capacity"],
        }
        category_weights = self.config.category_weights()
        weighted_categories = {
            key: category_weights[key] * category_scores[key]
            for key in category_scores
        }
        primitive_weights = self.config.primitive_weights()
        weighted_primitives = {
            key: primitive_weights[key] * normalized[key]
            for key in primitive_weights
        }
        objective = float(sum(weighted_categories.values()))
        if not math.isclose(
            objective,
            sum(weighted_primitives.values()),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise RuntimeError("V6 category and primitive objective totals differ")
        return {
            "model_schema_version": V6_MODEL_SCHEMA_VERSION,
            "objective_version": V6_OBJECTIVE_VERSION,
            "objective": objective,
            "objective_categories": category_scores,
            "weighted_objective_categories": weighted_categories,
            "raw": raw,
            "normalized": normalized,
            "primitive_weights": primitive_weights,
            "weighted_primitives": weighted_primitives,
            "scales": scale,
            "selected_zone_count": len(selected),
            "reserved_export_capacity": int(reserved_capacity),
            "assigned_export_boxes": int(assigned_boxes),
            "unused_reserved_capacity_boxes": int(unused_capacity),
            "peak_utilization": {
                **peak_policy.as_dict(),
                "maximum": float(maximum_utilization),
                "slack": float(peak_policy.epsilon_cap - maximum_utilization),
                "planned_slot_load_by_area": dict(
                    sorted(planned_slot_load_by_area.items())
                ),
                "utilization_by_area": utilization_by_area,
                "anonymous_import_included": True,
            },
            "validation": {
                "passed": True,
                "physical_rows_checked": len(used_resources),
                "export_bays_checked": len(export_groups_by_physical_bay),
                "reserved_export_bays_checked": len(
                    reserved_export_by_physical_bay
                ),
                "import_bays_checked": len(import_used_bays),
            },
        }


__all__ = [
    "ImportReservationKey",
    "V6_MODEL_SCHEMA_VERSION",
    "V6_OBJECTIVE_VERSION",
    "V6ModelEvaluator",
    "V6ObjectiveConfig",
    "V6ObjectiveScales",
    "V6PeakUtilizationPolicy",
    "ZoneBayFlowKey",
    "build_v6_import_candidates",
    "derive_v6_analytic_peak_policy",
    "v6_export_group_key",
    "v6_import_reservation_capacity",
    "v6_model_contract",
]
