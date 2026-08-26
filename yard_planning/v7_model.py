"""Canonical zone-free V7 model contract and independent evaluator."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .models import EXPORT_GROUP_IDENTITY_ATTRIBUTES, ExportGroup, ProblemData
from .row_aware_zones import v6_footprint
from .v6_model import (
    V6ObjectiveConfig,
    build_v6_import_candidates,
    derive_v6_analytic_peak_policy,
    v6_export_group_key,
)
from .v7_atoms import (
    V7RowAtom,
    atoms_by_group_bay,
    build_v7_row_atoms,
    candidate_areas_by_group,
    candidate_bays_by_group,
)


V7_MODEL_SCHEMA_VERSION = "hierarchical_group_area_bay_pattern_v7"
V7_OBJECTIVE_VERSION = "v7_spatial_existing_berth_normalized_v1"

GroupBayKey = tuple[str, str]
ImportReservationKey = tuple[str, str, str]


@dataclass(frozen=True)
class V7ObjectiveConfig:
    """Provisional development weights for the three V7 business categories."""

    spatial_consolidation_weight: float = 0.60
    existing_group_proximity_weight: float = 0.20
    berth_transport_weight: float = 0.20

    area_count_share: float = 0.35
    bay_count_share: float = 0.40
    bay_span_share: float = 0.25

    peak_utilization_headroom_fraction: float = 0.50

    def validate(self) -> None:
        categories = self.category_weights()
        shares = self.spatial_shares()
        for label, values in (("category", categories), ("spatial", shares)):
            if any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in values.values()
            ):
                raise ValueError(f"V7 {label} weights must be finite and nonnegative")
            if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-9):
                raise ValueError(f"V7 {label} weights must sum to one: {values}")
        headroom = float(self.peak_utilization_headroom_fraction)
        if not math.isfinite(headroom) or not 0.0 <= headroom <= 1.0:
            raise ValueError("V7 peak-utilization headroom must lie in [0, 1]")

    def category_weights(self) -> dict[str, float]:
        return {
            "spatial_consolidation": float(self.spatial_consolidation_weight),
            "existing_group_proximity": float(
                self.existing_group_proximity_weight
            ),
            "berth_transport": float(self.berth_transport_weight),
        }

    def spatial_shares(self) -> dict[str, float]:
        return {
            "area_count": float(self.area_count_share),
            "bay_count": float(self.bay_count_share),
            "bay_span": float(self.bay_span_share),
        }

    def primitive_weights(self) -> dict[str, float]:
        spatial = float(self.spatial_consolidation_weight)
        shares = self.spatial_shares()
        return {
            "area_count": spatial * shares["area_count"],
            "bay_count": spatial * shares["bay_count"],
            "bay_span": spatial * shares["bay_span"],
            "existing_group_proximity": float(
                self.existing_group_proximity_weight
            ),
            "berth_distance": float(self.berth_transport_weight),
        }

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "objective_version": V7_OBJECTIVE_VERSION,
            "calibration_status": "provisional_development_baseline",
            "aggregation": "normalized_weighted_sum",
            "category_weights": self.category_weights(),
            "spatial_shares": self.spatial_shares(),
            "primitive_weights": self.primitive_weights(),
            "peak_utilization": {
                "role": "hard_epsilon_constraint_not_objective",
                "headroom_fraction": float(
                    self.peak_utilization_headroom_fraction
                ),
            },
        }


@dataclass(frozen=True)
class V7ObjectiveScales:
    area_count: float
    bay_count: float
    bay_span: float
    existing_group_proximity: float
    berth_distance: float

    def as_dict(self) -> dict[str, float]:
        values = {
            "area_count": float(self.area_count),
            "bay_count": float(self.bay_count),
            "bay_span": float(self.bay_span),
            "existing_group_proximity": float(self.existing_group_proximity),
            "berth_distance": float(self.berth_distance),
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
            raise ValueError(f"V7 objective scales must be positive: {values}")
        return values


@dataclass(frozen=True)
class V7PeakUtilizationPolicy:
    reference_utilization: float
    headroom_fraction: float
    reference_source: str = "v7_analytic_reachable_capacity_lower_bound"

    def __post_init__(self) -> None:
        reference = float(self.reference_utilization)
        headroom = float(self.headroom_fraction)
        if not math.isfinite(reference) or not 0.0 <= reference <= 1.0:
            raise ValueError("V7 reference utilization must lie in [0, 1]")
        if not math.isfinite(headroom) or not 0.0 <= headroom <= 1.0:
            raise ValueError("V7 utilization headroom must lie in [0, 1]")
        if not str(self.reference_source).strip():
            raise ValueError("V7 utilization reference source is required")

    @property
    def epsilon_cap(self) -> float:
        reference = float(self.reference_utilization)
        return reference + float(self.headroom_fraction) * (1.0 - reference)

    def as_dict(self) -> dict[str, object]:
        return {
            "type": "data_derived_hard_epsilon_constraint",
            "reference_role": "analytic_workload_lower_bound",
            "reference_source": str(self.reference_source),
            "reference_utilization": float(self.reference_utilization),
            "headroom_fraction": float(self.headroom_fraction),
            "epsilon_cap": float(self.epsilon_cap),
            "area_balance_secondary_objective_used": False,
        }


def derive_v7_analytic_peak_policy(
    problem: ProblemData,
    objective_config: V7ObjectiveConfig | None = None,
    *,
    atoms: Sequence[V7RowAtom] | None = None,
) -> tuple[V7PeakUtilizationPolicy, dict[str, object]]:
    """Derive the V7 cap from the full legal V7 atom domain."""

    config = objective_config or V7ObjectiveConfig()
    config.validate()
    full_atoms, _limits = build_v7_row_atoms(problem)
    if atoms is None:
        atoms = full_atoms
    elif tuple(atoms) != full_atoms:
        raise ValueError(
            "V7 peak policy must be derived from the complete legal atom domain"
        )
    atoms = tuple(atoms)
    legacy_config = V6ObjectiveConfig(
        peak_utilization_headroom_fraction=float(
            config.peak_utilization_headroom_fraction
        )
    )
    legacy_policy, diagnostics = derive_v6_analytic_peak_policy(
        problem,
        legacy_config,
        atoms=atoms,  # Both atom types expose group_id and area_no.
    )
    policy = V7PeakUtilizationPolicy(
        reference_utilization=float(
            legacy_policy.minimum_feasible_utilization
        ),
        headroom_fraction=float(config.peak_utilization_headroom_fraction),
    )
    return policy, {
        **dict(diagnostics),
        "method": "v7_analytic_reachable_capacity_lower_bound",
        "model_schema_version": V7_MODEL_SCHEMA_VERSION,
        "legal_atom_count": len(atoms),
        "epsilon_cap": float(policy.epsilon_cap),
        "height_no_mix_in_feasibility_domain": True,
        "max_three_groups_in_feasibility_witness_required": True,
    }


def v7_model_contract(
    objective_config: V7ObjectiveConfig | None = None,
) -> dict[str, object]:
    config = objective_config or V7ObjectiveConfig()
    config.validate()
    return {
        "model_schema_version": V7_MODEL_SCHEMA_VERSION,
        "objective_version": V7_OBJECTIVE_VERSION,
        "export_group_identity": [
            "voyage",
            "size",
            "height",
            "discharge_port",
        ],
        "formal_decisions": {
            "group_bay_flow": "integer q[g,b]",
            "group_bay_use": "binary u[g,b]",
            "group_area_use": "binary y[g,a]",
            "row_atom_selection": "binary z[i]",
            "physical_group_bay_use": "binary U_phy[g,beta]",
            "anonymous_import_reservation": "integer p[flow,size,b]",
        },
        "hard_constraints": [
            "exact_export_demand_no_shortage",
            "physical_row_exclusivity",
            "size_and_footprint_compatibility",
            "new_export_height_no_mix_per_physical_bay",
            "legacy_state_nonworsening",
            "maximum_three_new_export_groups_per_physical_bay",
            "anonymous_import_size_and_capacity",
            "export_import_physical_bay_exclusivity",
            "data_derived_peak_utilization_epsilon_cap",
        ],
        "removed_formal_objects": [
            "zone",
            "zone_selection",
            "zone_contiguity",
            "zone_recovery",
            "unused_zone_capacity",
        ],
        "forbidden_objectives": [
            "zone_dispersion",
            "unused_zone_capacity",
            "voyage_area_dispersion",
            "area_load_balance",
            "peak_minimization_secondary_objective",
        ],
        "objective": config.as_dict(),
    }


class V7ModelEvaluator:
    """Validate and score a complete zone-free V7 allocation independently."""

    def __init__(
        self,
        problem: ProblemData,
        atoms: Sequence[V7RowAtom] | None = None,
        objective_config: V7ObjectiveConfig | None = None,
        *,
        max_new_groups_per_physical_bay: int = 3,
    ) -> None:
        self.problem = problem
        self.config = objective_config or V7ObjectiveConfig()
        self.config.validate()
        if int(max_new_groups_per_physical_bay) != 3:
            raise ValueError("V7 baseline requires exactly three groups per physical bay")
        self.max_new_groups_per_physical_bay = 3
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
        self.groups_by_id = {str(group.group_id): group for group in self.groups}
        self._validate_problem()
        full_atoms, limits = build_v7_row_atoms(problem)
        if atoms is None:
            atoms = full_atoms
        elif tuple(atoms) != full_atoms:
            raise ValueError(
                "V7 evaluator requires the complete legal atom domain so that "
                "feasibility and objective scales remain invariant"
            )
        self.atoms = tuple(atoms)
        self.atoms_by_index = {atom.candidate_index: atom for atom in self.atoms}
        if len(self.atoms_by_index) != len(self.atoms):
            raise ValueError("V7 atom candidate_index values must be unique")
        self.anchor_capacity_limits = dict(limits)
        self.atoms_by_pair = atoms_by_group_bay(self.atoms)
        self.candidate_bays = candidate_bays_by_group(self.atoms)
        self.candidate_areas = candidate_areas_by_group(self.atoms)
        missing = [
            group.group_id
            for group in self.groups
            if not self.candidate_bays.get(group.group_id)
        ]
        if missing:
            raise ValueError(f"V7 groups have no legal row atoms: {missing}")
        self.import_candidates = build_v6_import_candidates(problem)
        self._anchors_by_group_area = self._existing_anchors()
        self._reachable_anchor_groups = self._find_reachable_anchor_groups()
        self._berth_bounds = self._prepare_berth_bounds()
        self._area_order_bounds = self._prepare_area_order_bounds()
        self.scales = self._derive_scales()

    def _validate_problem(self) -> None:
        if self.problem.area_guidance_target:
            raise ValueError("V7 does not accept upstream area quotas")
        if self.problem.import_area_size_reference:
            raise ValueError("V7 requires direct anonymous import demand")
        if len(self.groups_by_id) != len(self.groups):
            raise ValueError("V7 export group ids must be unique")
        identities: dict[tuple[str, ...], str] = {}
        targets = {str(value) for value in self.problem.target_voyages}
        exports = (
            {str(value) for value in self.problem.export_voyages}
            if self.problem.export_voyages is not None
            else None
        )
        for group in self.groups:
            if targets and str(group.voyage_id) not in targets:
                raise ValueError(f"V7 group is outside target voyages: {group.group_id}")
            if exports is not None and str(group.voyage_id) not in exports:
                raise ValueError(f"V7 group is not classified export: {group.group_id}")
            key = tuple(
                [str(group.voyage_id)]
                + [
                    {
                        "IYC_CSZ_CSIZECD": str(group.size),
                        "IYC_POT_UNLDPORT": str(group.port),
                        "IYC_CHEIGHTCD": str(group.height),
                    }[attribute]
                    for attribute in EXPORT_GROUP_IDENTITY_ATTRIBUTES
                ]
            )
            if key in identities:
                raise ValueError(
                    "V7 export demand must be aggregated by voyage, size, "
                    f"height and discharge port: {identities[key]},{group.group_id}"
                )
            identities[key] = group.group_id

    def _existing_anchors(self) -> defaultdict[tuple[str, str], set[str]]:
        output: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        for group in self.groups:
            prefix = v6_export_group_key(group)
            for key, quantity in self.problem.existing_group_bay_load.items():
                if int(quantity) <= 0 or tuple(key[:-2]) != prefix:
                    continue
                area_no, bay_key = str(key[-2]), str(key[-1])
                if bay_key in self.problem.bays:
                    output[(group.group_id, area_no)].add(bay_key)
        return output

    def _find_reachable_anchor_groups(self) -> set[str]:
        return {
            group.group_id
            for group in self.groups
            if any(
                self._anchors_by_group_area.get((group.group_id, area))
                for area in self.candidate_areas[group.group_id]
            )
        }

    def _prepare_berth_bounds(self) -> dict[str, tuple[float, float]]:
        values: defaultdict[str, list[float]] = defaultdict(list)
        for group in self.groups:
            berth = self.problem.berth_by_voyage.get(group.voyage_id)
            if not berth:
                raise ValueError(f"V7 missing berth for voyage {group.voyage_id}")
            for area in sorted(self.candidate_areas[group.group_id]):
                distance = self.problem.berth_distances.get((area, berth))
                if distance is None or not math.isfinite(float(distance)):
                    raise ValueError(
                        f"V7 missing berth distance: voyage={group.voyage_id}, area={area}"
                    )
                values[group.voyage_id].append(float(distance))
        return {
            voyage: (min(distances), max(distances))
            for voyage, distances in values.items()
        }

    def _prepare_area_order_bounds(self) -> dict[tuple[str, str], tuple[int, int]]:
        output: dict[tuple[str, str], tuple[int, int]] = {}
        for group in self.groups:
            by_area: defaultdict[str, list[int]] = defaultdict(list)
            for bay_key in self.candidate_bays[group.group_id]:
                bay = self.problem.bays[bay_key]
                by_area[str(bay.area_no)].append(int(bay.bay_order))
            for area, orders in by_area.items():
                output[(group.group_id, area)] = (min(orders), max(orders))
        return output

    def _derive_scales(self) -> V7ObjectiveScales:
        area_scale = 0
        bay_scale = 0
        span_scale = 0
        for group in self.groups:
            areas = self.candidate_areas[group.group_id]
            bays = self.candidate_bays[group.group_id]
            area_scale += max(0, min(int(group.demand), len(areas)) - 1)
            bay_scale += max(0, min(int(group.demand), len(bays)) - 1)
            span_scale += sum(
                1
                for area in areas
                if self._area_order_bounds[(group.group_id, area)][0]
                < self._area_order_bounds[(group.group_id, area)][1]
            )
        total_demand = sum(int(group.demand) for group in self.groups)
        anchored_demand = sum(
            int(group.demand)
            for group in self.groups
            if group.group_id in self._reachable_anchor_groups
        )
        return V7ObjectiveScales(
            area_count=float(max(1, area_scale)),
            bay_count=float(max(1, bay_scale)),
            bay_span=float(max(1, span_scale)),
            existing_group_proximity=float(max(1, anchored_demand)),
            berth_distance=float(max(1, total_demand)),
        )

    def normalized_bay_order(self, group_id: str, bay_key: str) -> float:
        bay = self.problem.bays[str(bay_key)]
        lower, upper = self._area_order_bounds[(str(group_id), str(bay.area_no))]
        if upper <= lower:
            return 0.0
        return (float(bay.bay_order) - lower) / (upper - lower)

    def normalized_existing_proximity(self, group_id: str, bay_key: str) -> float:
        group = self.groups_by_id[str(group_id)]
        if group.group_id not in self._reachable_anchor_groups:
            return 0.0
        bay = self.problem.bays[str(bay_key)]
        anchors = self._anchors_by_group_area.get(
            (group.group_id, str(bay.area_no)), set()
        )
        if not anchors:
            return 1.0
        area_orders = [
            int(self.problem.bays[key].bay_order)
            for key in self.candidate_bays[group.group_id]
            if str(self.problem.bays[key].area_no) == str(bay.area_no)
        ]
        span = max(area_orders) - min(area_orders) if area_orders else 0
        distance = min(
            abs(int(self.problem.bays[key].bay_order) - int(bay.bay_order))
            for key in anchors
        )
        return 0.0 if span <= 0 else min(1.0, float(distance) / float(span))

    def normalized_berth_distance(self, group_id: str, bay_key: str) -> float:
        group = self.groups_by_id[str(group_id)]
        area = str(self.problem.bays[str(bay_key)].area_no)
        berth = self.problem.berth_by_voyage[group.voyage_id]
        value = float(self.problem.berth_distances[(area, berth)])
        lower, upper = self._berth_bounds[group.voyage_id]
        return 0.0 if upper <= lower else (value - lower) / (upper - lower)

    def group_bay_flow_objective_coefficient(
        self, group_id: str, bay_key: str
    ) -> float:
        weights = self.config.primitive_weights()
        scales = self.scales.as_dict()
        return (
            weights["existing_group_proximity"]
            * self.normalized_existing_proximity(group_id, bay_key)
            / scales["existing_group_proximity"]
            + weights["berth_distance"]
            * self.normalized_berth_distance(group_id, bay_key)
            / scales["berth_distance"]
        )

    def group_bay_use_objective_coefficient(self) -> float:
        return (
            self.config.primitive_weights()["bay_count"]
            / self.scales.as_dict()["bay_count"]
        )

    def group_area_use_objective_coefficient(self) -> float:
        return (
            self.config.primitive_weights()["area_count"]
            / self.scales.as_dict()["area_count"]
        )

    def group_area_span_objective_coefficient(self) -> float:
        return (
            self.config.primitive_weights()["bay_span"]
            / self.scales.as_dict()["bay_span"]
        )

    def objective_constant(self) -> float:
        weights = self.config.primitive_weights()
        scales = self.scales.as_dict()
        return -len(self.groups) * (
            weights["area_count"] / scales["area_count"]
            + weights["bay_count"] / scales["bay_count"]
        )

    def area_coarse_cost(self, group_id: str, area_no: str) -> float:
        bays = [
            bay_key
            for bay_key in self.candidate_bays[str(group_id)]
            if str(self.problem.bays[bay_key].area_no) == str(area_no)
        ]
        if not bays:
            return math.inf
        return min(
            self.group_bay_flow_objective_coefficient(group_id, bay_key)
            for bay_key in bays
        )

    def evaluate(
        self,
        selected_atom_indices: Iterable[int],
        group_bay_flow: Mapping[GroupBayKey, int],
        import_reservation: Mapping[ImportReservationKey, int],
        peak_policy: V7PeakUtilizationPolicy,
    ) -> dict[str, object]:
        if not math.isclose(
            float(peak_policy.headroom_fraction),
            float(self.config.peak_utilization_headroom_fraction),
            abs_tol=1e-12,
        ):
            raise ValueError("V7 peak policy and objective config disagree")
        errors: list[str] = []
        selected = {int(index) for index in selected_atom_indices}
        unknown = selected - set(self.atoms_by_index)
        if unknown:
            errors.append(f"unknown V7 row atoms: {sorted(unknown)[:10]}")

        flow: Counter[GroupBayKey] = Counter()
        for raw_key, raw_quantity in group_bay_flow.items():
            key = (str(raw_key[0]), str(raw_key[1]))
            quantity = int(raw_quantity)
            if quantity < 0 or float(quantity) != float(raw_quantity):
                errors.append(f"invalid V7 group-bay flow: {raw_key}={raw_quantity}")
            elif quantity > 0:
                flow[key] += quantity

        selected_by_pair: defaultdict[GroupBayKey, list[V7RowAtom]] = defaultdict(list)
        used_resources: dict[tuple[str, str], int] = {}
        reserved_physical: Counter[str] = Counter()
        reserved_anchor_size: Counter[tuple[str, str]] = Counter()
        for index in sorted(selected & set(self.atoms_by_index)):
            atom = self.atoms_by_index[index]
            pair = (atom.group_id, atom.anchor_bay_key)
            selected_by_pair[pair].append(atom)
            for resource in atom.physical_resources:
                previous = used_resources.get(resource)
                if previous is not None:
                    errors.append(
                        f"V7 physical row selected twice: {resource}, atoms={previous},{index}"
                    )
                used_resources[resource] = index
            reserved_anchor_size[(atom.anchor_bay_key, atom.size)] += atom.capacity
            for physical in atom.physical_bays:
                reserved_physical[physical] += atom.capacity

        assigned: Counter[str] = Counter()
        used_bays_by_group: defaultdict[str, set[str]] = defaultdict(set)
        groups_by_physical: defaultdict[str, set[str]] = defaultdict(set)
        planned_load_by_area: Counter[str] = Counter()
        proximity_sum = 0.0
        berth_sum = 0.0
        for pair, quantity in flow.items():
            group_id, bay_key = pair
            group = self.groups_by_id.get(group_id)
            pair_atoms = selected_by_pair.get(pair, [])
            if group is None or bay_key not in self.problem.bays:
                errors.append(f"unknown V7 group-bay pair: {pair}")
                continue
            if not pair_atoms:
                errors.append(f"positive V7 flow has no selected row atom: {pair}")
                continue
            capacity = sum(atom.capacity for atom in pair_atoms)
            if quantity > capacity:
                errors.append(f"V7 group-bay atom capacity exceeded: {pair}")
            assigned[group_id] += quantity
            used_bays_by_group[group_id].add(bay_key)
            footprint = v6_footprint(self.problem, bay_key, group.size)
            for physical in footprint:
                groups_by_physical[physical].add(group_id)
            area = str(self.problem.bays[bay_key].area_no)
            planned_load_by_area[area] += quantity * len(footprint)
            proximity_sum += quantity * self.normalized_existing_proximity(
                group_id, bay_key
            )
            berth_sum += quantity * self.normalized_berth_distance(group_id, bay_key)

        for pair in selected_by_pair:
            if flow.get(pair, 0) <= 0:
                errors.append(f"selected V7 row atom has no positive bay flow: {pair}")
        for group in self.groups:
            if assigned[group.group_id] != int(group.demand):
                errors.append(
                    f"V7 export demand mismatch: group={group.group_id}, "
                    f"assigned={assigned[group.group_id]}, demand={group.demand}"
                )
        for (bay_key, size), capacity in reserved_anchor_size.items():
            if capacity > int(self.problem.bays[bay_key].cap_by_size.get(size, 0)):
                errors.append(f"V7 anchor size capacity exceeded: {(bay_key, size)}")
        for bay_key, capacity in reserved_physical.items():
            if capacity > int(self.problem.bays[bay_key].physical_capacity):
                errors.append(f"V7 physical capacity exceeded: {bay_key}")
        for bay_key, group_ids in groups_by_physical.items():
            sizes = {self.groups_by_id[group_id].size for group_id in group_ids}
            heights = {self.groups_by_id[group_id].height for group_id in group_ids}
            if len(sizes) > 1:
                errors.append(f"V7 new export size mixing: bay={bay_key}")
            if len(heights) > 1:
                errors.append(f"V7 new export height mixing: bay={bay_key}")
            if len(group_ids) > self.max_new_groups_per_physical_bay:
                errors.append(
                    f"V7 physical bay uses more than three new groups: bay={bay_key}"
                )

        import_by_flow_size: Counter[tuple[str, str]] = Counter()
        import_by_physical: Counter[str] = Counter()
        import_by_anchor_size: Counter[tuple[str, str]] = Counter()
        import_sizes_by_physical: defaultdict[str, set[str]] = defaultdict(set)
        import_used: set[str] = set()
        for raw_key, raw_quantity in import_reservation.items():
            flow_code, size, bay_key = map(str, raw_key)
            quantity = int(raw_quantity)
            if quantity < 0 or float(quantity) != float(raw_quantity):
                errors.append(f"invalid V7 import reservation: {raw_key}")
                continue
            if quantity <= 0:
                continue
            candidates = dict(self.import_candidates.get((flow_code, size), ()))
            if bay_key not in candidates or quantity > int(candidates.get(bay_key, 0)):
                errors.append(f"invalid V7 import candidate/capacity: {raw_key}")
                continue
            footprint = v6_footprint(self.problem, bay_key, size)
            for physical in footprint:
                import_by_physical[physical] += quantity
                import_sizes_by_physical[physical].add(size)
                import_used.add(physical)
            import_by_anchor_size[(bay_key, size)] += quantity
            import_by_flow_size[(flow_code, size)] += quantity
            area = str(self.problem.bays[bay_key].area_no)
            planned_load_by_area[area] += quantity * len(footprint)
        required_import = Counter(
            {
                tuple(map(str, key)): int(quantity)
                for key, quantity in self.problem.import_demand_by_flow_size.items()
                if int(quantity) > 0
            }
        )
        for key in sorted(set(required_import) | set(import_by_flow_size)):
            if required_import[key] != import_by_flow_size[key]:
                errors.append(f"V7 anonymous import demand mismatch: {key}")
        for bay_key, load in import_by_physical.items():
            if load > int(self.problem.bays[bay_key].physical_capacity):
                errors.append(f"V7 import physical capacity exceeded: {bay_key}")
            if len(import_sizes_by_physical[bay_key]) > 1:
                errors.append(f"V7 import size mixing: {bay_key}")
        for (bay_key, size), load in import_by_anchor_size.items():
            if load > int(self.problem.bays[bay_key].cap_by_size.get(size, 0)):
                errors.append(f"V7 import anchor capacity exceeded: {(bay_key, size)}")
        shared = sorted(set(groups_by_physical) & import_used)
        if shared:
            errors.append(f"V7 import and export share physical bays: {shared}")

        area_capacity: Counter[str] = Counter()
        for bay in self.problem.bays.values():
            area_capacity[str(bay.area_no)] += int(bay.physical_capacity)
        utilization = {
            area: float(load) / float(area_capacity[area])
            for area, load in sorted(planned_load_by_area.items())
            if int(load) > 0 and int(area_capacity[area]) > 0
        }
        maximum_utilization = max(utilization.values(), default=0.0)
        if maximum_utilization > float(peak_policy.epsilon_cap) + 1e-9:
            errors.append(
                f"V7 peak cap exceeded: actual={maximum_utilization}, "
                f"cap={peak_policy.epsilon_cap}"
            )
        if errors:
            raise ValueError("V7 solution validation failed: " + "; ".join(errors[:30]))

        used_areas_by_group: defaultdict[str, set[str]] = defaultdict(set)
        span_sum = 0.0
        span_by_group_area: dict[str, float] = {}
        for group_id, bay_keys in used_bays_by_group.items():
            by_area: defaultdict[str, list[float]] = defaultdict(list)
            for bay_key in bay_keys:
                area = str(self.problem.bays[bay_key].area_no)
                used_areas_by_group[group_id].add(area)
                by_area[area].append(self.normalized_bay_order(group_id, bay_key))
            for area, orders in by_area.items():
                span = max(orders) - min(orders) if len(orders) > 1 else 0.0
                span_by_group_area[f"{group_id}|{area}"] = float(span)
                span_sum += span

        extra_areas = sum(
            max(0, len(used_areas_by_group[group.group_id]) - 1)
            for group in self.groups
        )
        extra_bays = sum(
            max(0, len(used_bays_by_group[group.group_id]) - 1)
            for group in self.groups
        )
        raw = {
            "extra_group_areas": float(extra_areas),
            "extra_group_bays": float(extra_bays),
            "normalized_within_area_bay_span_sum": float(span_sum),
            "existing_group_normalized_distance_sum": float(proximity_sum),
            "berth_normalized_distance_sum": float(berth_sum),
        }
        scales = self.scales.as_dict()
        normalized = {
            "area_count": raw["extra_group_areas"] / scales["area_count"],
            "bay_count": raw["extra_group_bays"] / scales["bay_count"],
            "bay_span": raw["normalized_within_area_bay_span_sum"]
            / scales["bay_span"],
            "existing_group_proximity": raw[
                "existing_group_normalized_distance_sum"
            ]
            / scales["existing_group_proximity"],
            "berth_distance": raw["berth_normalized_distance_sum"]
            / scales["berth_distance"],
        }
        weights = self.config.primitive_weights()
        weighted = {key: weights[key] * normalized[key] for key in weights}
        shares = self.config.spatial_shares()
        categories = {
            "spatial_consolidation": sum(
                shares[key] * normalized[key] for key in shares
            ),
            "existing_group_proximity": normalized["existing_group_proximity"],
            "berth_transport": normalized["berth_distance"],
        }
        category_weights = self.config.category_weights()
        weighted_categories = {
            key: category_weights[key] * value for key, value in categories.items()
        }
        objective = float(sum(weighted.values()))
        if not math.isclose(
            objective, sum(weighted_categories.values()), rel_tol=1e-10, abs_tol=1e-10
        ):
            raise RuntimeError("V7 primitive and category objectives disagree")
        return {
            "model_schema_version": V7_MODEL_SCHEMA_VERSION,
            "objective_version": V7_OBJECTIVE_VERSION,
            "objective": objective,
            "raw": raw,
            "normalized": normalized,
            "primitive_weights": weights,
            "weighted_primitives": weighted,
            "objective_categories": categories,
            "weighted_objective_categories": weighted_categories,
            "scales": scales,
            "selected_atom_count": len(selected),
            "positive_group_bay_count": len(flow),
            "span_by_group_area": dict(sorted(span_by_group_area.items())),
            "groups_per_physical_bay": {
                key: len(value) for key, value in sorted(groups_by_physical.items())
            },
            "areas_per_group": {
                group.group_id: len(used_areas_by_group[group.group_id])
                for group in self.groups
            },
            "bays_per_group": {
                group.group_id: len(used_bays_by_group[group.group_id])
                for group in self.groups
            },
            "peak_utilization": {
                **peak_policy.as_dict(),
                "maximum": float(maximum_utilization),
                "slack": float(peak_policy.epsilon_cap - maximum_utilization),
                "planned_slot_load_by_area": dict(sorted(planned_load_by_area.items())),
                "utilization_by_area": utilization,
                "anonymous_import_included": True,
            },
            "validation": {
                "passed": True,
                "physical_rows_checked": len(used_resources),
                "physical_bays_checked": len(groups_by_physical),
                "max_new_groups_per_physical_bay": 3,
                "height_no_mix_checked": True,
                "legacy_compatibility_in_atom_domain": True,
            },
        }


__all__ = [
    "GroupBayKey",
    "ImportReservationKey",
    "V7_MODEL_SCHEMA_VERSION",
    "V7_OBJECTIVE_VERSION",
    "V7ModelEvaluator",
    "V7ObjectiveConfig",
    "V7ObjectiveScales",
    "V7PeakUtilizationPolicy",
    "derive_v7_analytic_peak_policy",
    "v7_model_contract",
]
