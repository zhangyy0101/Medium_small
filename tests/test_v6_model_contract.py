from __future__ import annotations

import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from yard_planning.direct_milp import DirectMilpPlanner
from yard_planning.models import (
    AttributeRules,
    Bay,
    ExportGroup,
    ProblemData,
    existing_export_group_key,
)
from yard_planning.planner import ColumnGenerationConfig, PlacementColumn
from yard_planning.output_validator import validate_output_files
from yard_planning.planner import write_rows
from yard_planning.row_aware_zones import (
    RowAwareZone,
    build_complete_row_aware_zone_universe,
    build_complete_v6_zone_universe,
    build_v6_row_aware_bay_atoms,
)
from yard_planning.v6_model import (
    V6ModelEvaluator,
    V6ObjectiveConfig,
    V6PeakUtilizationPolicy,
    derive_v6_analytic_peak_policy,
    v6_export_group_key,
    v6_model_contract,
)


def make_bay(
    bay_no: str,
    *,
    rows: tuple[str, ...] = ("1", "2"),
    row_capacity: int = 1,
) -> Bay:
    bay_key = f"A|{bay_no}"
    return Bay(
        area_no="A",
        bay_no=bay_no,
        bay_key=bay_key,
        bay_order=int(bay_no),
        cap_by_size={"20": len(rows) * row_capacity},
        physical_capacity=len(rows) * row_capacity,
        row_cap_by_size={
            "20": {row_no: row_capacity for row_no in rows}
        },
        row_physical_capacity={
            row_no: row_capacity for row_no in rows
        },
    )


def make_group(
    group_id: str,
    *,
    port: str,
    demand: int = 1,
) -> ExportGroup:
    return ExportGroup(
        group_id=group_id,
        voyage_id="V1",
        status="OF",
        port=port,
        size="20",
        height="96",
        demand=demand,
    )


def make_problem(
    groups: list[ExportGroup],
    bays: list[Bay],
    *,
    import_boxes: int = 0,
) -> ProblemData:
    functions = {"OF"}
    if import_boxes:
        functions.add("IF")
    return ProblemData(
        export_groups=groups,
        bays={bay.bay_key: bay for bay in bays},
        area_functions={"A": functions},
        target_voyages=["V1"],
        export_voyages={"V1"},
        import_demand_by_flow_size=(
            {("IF", "20"): import_boxes} if import_boxes else {}
        ),
        berth_distances={("A", "Q1"): 1.0},
        berth_by_voyage={"V1": "Q1"},
    )


def make_location(index: int, bay_no: str, row_no: str) -> PlacementColumn:
    bay_key = f"A|{bay_no}"
    return PlacementColumn(
        column_id=f"C{index}",
        group_id="G1",
        voyage_id="V1",
        flow="OF",
        port="P1",
        size="20",
        big_plan_size="20",
        height="96",
        attributes={},
        area_no="A",
        bay_key=bay_key,
        bay_no=bay_no,
        quantity=1,
        stack_units=1,
        row_allocation=((bay_key, row_no, 1),),
        quota_key=("V1", "OF", "A", "20"),
        group_key=("V1", "20", "96", "P1"),
        intrinsic_cost=0.0,
    )


class V6ModelContractTests(unittest.TestCase):
    def test_group_identity_cannot_be_redefined_by_dynamic_attributes(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed"):
            AttributeRules(
                group_attributes=("CUSTOM",),
                bay_no_mix_attributes=("IYC_CHEIGHTCD",),
                row_no_mix_attributes=("IYC_POT_UNLDPORT",),
            )

    def test_duplicate_fixed_group_identity_requires_preaggregation(self) -> None:
        first = make_group("G1", port="P1")
        second = ExportGroup(
            **{
                **first.__dict__,
                "group_id": "G2",
                "attributes": {"IGNORED_EXTRA_FIELD": "OTHER"},
            }
        )

        with self.assertRaisesRegex(ValueError, "must be aggregated"):
            DirectMilpPlanner(
                make_problem([first, second], [make_bay("01")])
            )

    def test_row_aware_zone_can_change_rows_between_adjacent_bays(self) -> None:
        locations = [
            make_location(0, "01", "1"),
            make_location(1, "01", "2"),
            make_location(2, "03", "1"),
            make_location(3, "03", "2"),
        ]
        zones = build_complete_row_aware_zone_universe(
            locations,
            {index: 1 for index in range(len(locations))},
            {"A|01": 1, "A|03": 3},
            {"G1": 3},
        )

        self.assertEqual(15, len(zones))
        self.assertTrue(
            any(zone.candidate_indices == (0, 3) for zone in zones)
        )
        self.assertTrue(
            any(zone.candidate_indices == (0, 1) for zone in zones)
        )
        self.assertTrue(
            all(len(zone.resources) == len(set(zone.resources)) for zone in zones)
        )
        self.assertTrue(all(zone.capacity <= 4 for zone in zones))

    def test_self_contained_v6_zone_universe_uses_problem_rows_directly(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1", demand=2)],
            [make_bay("01"), make_bay("03")],
        )

        zones = build_complete_v6_zone_universe(problem)

        self.assertTrue(
            any(
                dict(zone.rows_by_anchor_bay)
                == {"A|01": ("1",), "A|03": ("2",)}
                for zone in zones
            )
        )

    def test_v6_zone_universe_has_no_unproved_demand_plus_atom_cap(self) -> None:
        locations = [
            make_location(0, "01", "1"),
            make_location(1, "01", "2"),
            make_location(2, "03", "1"),
            make_location(3, "03", "2"),
        ]
        zones = build_complete_row_aware_zone_universe(
            locations,
            {index: 2 for index in range(len(locations))},
            {"A|01": 1, "A|03": 3},
            {"G1": 1},
        )

        largest = max(zones, key=lambda zone: zone.capacity)
        self.assertEqual(8, largest.capacity)
        self.assertEqual(
            (("A|01", 4), ("A|03", 4)),
            largest.anchor_bay_capacities,
        )

    def test_v6_objective_has_three_categories_and_expected_primitives(self) -> None:
        config = V6ObjectiveConfig()
        config.validate()

        self.assertEqual(
            {
                "spatial_concentration",
                "berth_transport",
                "reserved_capacity_efficiency",
            },
            set(config.category_weights()),
        )
        expected = {
            "zone_dispersion": 0.35,
            "voyage_area_dispersion": 0.15,
            "existing_group_proximity": 0.125,
            "berth_distance": 0.1625,
            "unused_capacity": 0.2125,
        }
        actual = config.primitive_weights()
        self.assertEqual(set(expected), set(actual))
        for key, value in expected.items():
            self.assertAlmostEqual(value, actual[key])

    def test_v6_contract_excludes_all_known_old_model_terms(self) -> None:
        contract = v6_model_contract()
        forbidden = set(contract["forbidden_inputs_or_terms"])

        self.assertIn("fixed_row_strip_zone", forbidden)
        self.assertIn("demand_plus_one_atom_zone_capacity_cap", forbidden)
        self.assertIn("row_dispersion_objective", forbidden)
        self.assertIn("upstream_large_plan_area_target", forbidden)
        self.assertNotIn("shortage_variable", contract["decision_families"])

    def test_v6_evaluator_independently_rejects_noncontiguous_zone(self) -> None:
        group = make_group("G1", port="P1", demand=2)
        bays = [
            make_bay("01", rows=("1",), row_capacity=1),
            make_bay("05", rows=("1",), row_capacity=1),
        ]
        forged = RowAwareZone(
            zone_id=0,
            group_id="G1",
            area_no="A",
            anchor_bay_keys=("A|01", "A|05"),
            anchor_bay_capacities=(("A|01", 1), ("A|05", 1)),
            candidate_indices=(0, 1),
            rows_by_anchor_bay=(("A|01", ("1",)), ("A|05", ("1",))),
            capacity=2,
            resources=(("A|01", "1"), ("A|05", "1")),
            physical_bay_keys=("A|01", "A|05"),
        )

        with self.assertRaisesRegex(ValueError, "not contiguous"):
            V6ModelEvaluator(make_problem([group], bays), [forged])

    def test_v6_selected_zone_cannot_use_an_empty_bridge_bay(self) -> None:
        group = make_group("G1", port="P1", demand=1)
        bays = [
            make_bay("01", rows=("1",)),
            make_bay("03", rows=("1",)),
        ]
        zones = build_complete_row_aware_zone_universe(
            [make_location(0, "01", "1"), make_location(1, "03", "1")],
            {0: 1, 1: 1},
            {"A|01": 1, "A|03": 3},
            {"G1": 1},
        )
        combined = next(zone for zone in zones if len(zone.anchor_bay_keys) == 2)
        evaluator = V6ModelEvaluator(make_problem([group], bays), zones)

        with self.assertRaisesRegex(ValueError, "positive flow in every bay"):
            evaluator.evaluate(
                {combined.zone_id},
                {(combined.zone_id, "A|01"): 1},
                {},
                V6PeakUtilizationPolicy(0.5, 0.5),
            )

    def test_v6_rejects_upstream_large_plan_targets(self) -> None:
        problem = make_problem([], [make_bay("01")])
        problem.area_guidance_target = {("V1", "OF", "A", "20"): 1}

        with self.assertRaisesRegex(ValueError, "large-plan"):
            V6ModelEvaluator(problem, [])

    def test_v6_large_box_zone_uses_complete_footprints_and_can_change_rows(self) -> None:
        group = ExportGroup(
            group_id="G1",
            voyage_id="V1",
            status="OF",
            port="P1",
            size="40",
            height="96",
            demand=2,
        )
        bays = [
            make_bay(code, rows=("1", "2"))
            for code in ("01", "03", "05", "07")
        ]
        for bay in bays:
            bay.cap_by_size = {"40": 2}
            bay.row_cap_by_size = {"40": {"1": 1, "2": 1}}
        bays[0].large_bay_partner_key = bays[1].bay_key
        bays[2].large_bay_partner_key = bays[3].bay_key

        def large_location(
            index: int,
            anchor: Bay,
            partner: Bay,
            row_no: str,
        ) -> PlacementColumn:
            return PlacementColumn(
                column_id=f"L{index}",
                group_id="G1",
                voyage_id="V1",
                flow="OF",
                port="P1",
                size="40",
                big_plan_size="40",
                height="96",
                attributes={},
                area_no="A",
                bay_key=anchor.bay_key,
                bay_no=anchor.bay_no,
                quantity=1,
                stack_units=1,
                row_allocation=(
                    (anchor.bay_key, row_no, 1),
                    (partner.bay_key, row_no, 1),
                ),
                quota_key=("V1", "OF", "A", "40"),
                group_key=("V1", "40", "96", "P1"),
                intrinsic_cost=0.0,
            )

        locations = [
            large_location(0, bays[0], bays[1], "1"),
            large_location(1, bays[2], bays[3], "2"),
        ]
        zones = build_complete_row_aware_zone_universe(
            locations,
            {0: 1, 1: 1},
            {bay.bay_key: bay.bay_order for bay in bays},
            {"G1": 2},
        )
        combined = next(
            zone for zone in zones if zone.candidate_indices == (0, 1)
        )
        evaluator = V6ModelEvaluator(make_problem([group], bays), zones)

        certificate = evaluator.evaluate(
            {combined.zone_id},
            {
                (combined.zone_id, bays[0].bay_key): 1,
                (combined.zone_id, bays[2].bay_key): 1,
            },
            {},
            V6PeakUtilizationPolicy(1.0, 0.5),
        )

        self.assertEqual(
            {bay.bay_key for bay in bays},
            set(combined.physical_bay_keys),
        )
        self.assertEqual(0.0, certificate["objective"])

    def test_v6_45ft_atoms_only_use_edge_large_bays(self) -> None:
        group = ExportGroup(
            group_id="G1",
            voyage_id="V1",
            status="OF",
            port="P1",
            size="45",
            height="96",
            demand=1,
        )
        bays = [
            make_bay(code, rows=("1",))
            for code in ("01", "03", "05", "07", "09", "11")
        ]
        for bay in bays:
            bay.cap_by_size = {"45": 1}
            bay.row_cap_by_size = {"45": {"1": 1}}
        for left, right in ((0, 1), (2, 3), (4, 5)):
            bays[left].large_bay_partner_key = bays[right].bay_key

        zones = build_complete_v6_zone_universe(make_problem([group], bays))

        anchors = {
            bay_key
            for zone in zones
            for bay_key in zone.anchor_bay_keys
        }
        self.assertEqual({"A|01", "A|09"}, anchors)

    def test_v6_objective_scores_true_disconnected_bay_zones(self) -> None:
        group = make_group("G1", port="P1", demand=2)
        bays = [
            make_bay("01", rows=("1",), row_capacity=1),
            make_bay("05", rows=("1",), row_capacity=1),
        ]
        locations = [
            make_location(0, "01", "1"),
            make_location(1, "05", "1"),
        ]
        zones = build_complete_row_aware_zone_universe(
            locations,
            {0: 1, 1: 1},
            {"A|01": 1, "A|05": 5},
            {"G1": 2},
        )
        evaluator = V6ModelEvaluator(make_problem([group], bays), zones)
        zone_by_bay = {
            zone.anchor_bay_keys[0]: zone.zone_id for zone in zones
        }
        selected = set(zone_by_bay.values())
        flow = {
            (zone_by_bay["A|01"], "A|01"): 1,
            (zone_by_bay["A|05"], "A|05"): 1,
        }

        certificate = evaluator.evaluate(
            selected,
            flow,
            {},
            V6PeakUtilizationPolicy(1.0, 0.5),
        )

        self.assertAlmostEqual(0.35, certificate["objective"])
        self.assertEqual(1.0, certificate["raw"]["extra_contiguous_zones"])
        self.assertEqual(0.0, certificate["raw"]["extra_voyage_areas"])
        self.assertEqual(0.0, certificate["raw"]["unused_reserved_capacity_boxes"])
        self.assertTrue(certificate["validation"]["passed"])

    def test_v6_unused_capacity_uses_selected_row_resource_capacity(self) -> None:
        group = make_group("G1", port="P1", demand=1)
        bay = make_bay("01", rows=("1", "2"), row_capacity=1)
        locations = [
            make_location(0, "01", "1"),
            make_location(1, "01", "2"),
        ]
        zones = build_complete_row_aware_zone_universe(
            locations,
            {0: 1, 1: 1},
            {"A|01": 1},
            {"G1": 1},
        )
        two_row_zone = next(
            zone for zone in zones if zone.candidate_indices == (0, 1)
        )
        evaluator = V6ModelEvaluator(make_problem([group], [bay]), zones)

        certificate = evaluator.evaluate(
            {two_row_zone.zone_id},
            {(two_row_zone.zone_id, "A|01"): 1},
            {},
            V6PeakUtilizationPolicy(0.5, 0.5),
        )

        self.assertEqual(1.0, certificate["raw"]["unused_reserved_capacity_boxes"])
        self.assertAlmostEqual(0.2125, certificate["objective"])

    def test_v6_reconstructs_area_anchor_and_berth_contributions(self) -> None:
        group = make_group("G1", port="P1", demand=2)
        bay_a = make_bay("01", rows=("1",))
        bay_b = make_bay("01", rows=("1",))
        bay_b.area_no = "B"
        bay_b.bay_key = "B|01"
        location_a = make_location(0, "01", "1")
        location_b = replace(
            make_location(1, "01", "1"),
            area_no="B",
            bay_key="B|01",
            row_allocation=(("B|01", "1", 1),),
        )
        problem = ProblemData(
            export_groups=[group],
            bays={bay_a.bay_key: bay_a, bay_b.bay_key: bay_b},
            area_functions={"A": {"OF"}, "B": {"OF"}},
            target_voyages=["V1"],
            export_voyages={"V1"},
            existing_group_bay_load={
                v6_export_group_key(group) + ("A", "A|01"): 1
            },
            berth_distances={("A", "Q1"): 1.0, ("B", "Q1"): 3.0},
            berth_by_voyage={"V1": "Q1"},
        )
        zones = build_complete_row_aware_zone_universe(
            [location_a, location_b],
            {0: 1, 1: 1},
            {"A|01": 1, "B|01": 1},
            {"G1": 2},
        )
        zone_by_area = {zone.area_no: zone.zone_id for zone in zones}
        evaluator = V6ModelEvaluator(problem, zones)

        certificate = evaluator.evaluate(
            set(zone_by_area.values()),
            {
                (zone_by_area["A"], "A|01"): 1,
                (zone_by_area["B"], "B|01"): 1,
            },
            {},
            V6PeakUtilizationPolicy(1.0, 0.5),
        )

        self.assertEqual(1.0, certificate["raw"]["extra_contiguous_zones"])
        self.assertEqual(1.0, certificate["raw"]["extra_voyage_areas"])
        self.assertEqual(
            1.0,
            certificate["raw"]["existing_group_normalized_distance_sum"],
        )
        self.assertEqual(1.0, certificate["raw"]["berth_normalized_distance_sum"])
        self.assertAlmostEqual(0.64375, certificate["objective"])

    def test_v6_peak_policy_marks_exact_minmax_reference(self) -> None:
        policy = V6PeakUtilizationPolicy(
            minimum_feasible_utilization=0.4,
            headroom_fraction=0.5,
        )

        self.assertAlmostEqual(0.7, policy.epsilon_cap)
        self.assertEqual(
            "auxiliary_full_v6_minmax_mip",
            policy.as_dict()["minimum_source"],
        )
        self.assertTrue(
            policy.as_dict()["minimum_feasible_utilization_proven"]
        )

    def test_v6_analytic_peak_policy_is_not_mislabeled_rho_star(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1", demand=1)],
            [make_bay("01")],
        )

        policy, diagnostics = derive_v6_analytic_peak_policy(problem)

        self.assertEqual("analytic_workload_lower_bound", policy.reference_role)
        self.assertFalse(
            policy.as_dict()["minimum_feasible_utilization_proven"]
        )
        self.assertFalse(diagnostics["minmax_mip_solved"])
        self.assertGreaterEqual(policy.epsilon_cap, 0.5)

    def test_bay_gap_breaks_zone_contiguity(self) -> None:
        locations = [
            make_location(0, "01", "1"),
            make_location(1, "05", "2"),
        ]
        zones = build_complete_row_aware_zone_universe(
            locations,
            {0: 1, 1: 1},
            {"A|01": 1, "A|05": 5},
            {"G1": 2},
        )

        self.assertFalse(
            any(len(zone.anchor_bay_keys) > 1 for zone in zones)
        )

    def test_existing_size_state_rejects_incompatible_import(self) -> None:
        bay = make_bay("01")
        bay.existing_size_modes = {"40"}
        planner = DirectMilpPlanner(
            make_problem([], [bay], import_boxes=1)
        )

        self.assertEqual([], planner.import_reservation_candidates[("IF", "20")])

    def test_mixed_existing_heights_allow_only_an_existing_height(self) -> None:
        height_96 = make_group("G96", port="P1")
        height_86 = replace(height_96, group_id="G86", height="86")
        new_height = replace(height_96, group_id="G106", height="106")
        bay = make_bay("01", rows=("1",))
        bay.existing_size_modes = {"20"}
        bay.existing_heights = {"86", "96"}

        atoms, _limits = build_v6_row_aware_bay_atoms(
            make_problem([height_96, height_86, new_height], [bay])
        )

        self.assertEqual(
            {"G86", "G96"},
            {atom.group_id for atom in atoms},
        )

    def test_mixed_existing_row_allows_only_an_exact_existing_group(self) -> None:
        first = make_group("G1", port="P1")
        second = replace(
            first,
            group_id="G2",
            voyage_id="V2",
            port="P2",
        )
        nonexistent_cross = replace(
            first,
            group_id="G3",
            port="P2",
        )
        bay = make_bay("01", rows=("1",))
        bay.existing_size_modes = {"20"}
        bay.existing_heights = {"96"}
        bay.existing_group_keys_by_row = {
            "1": {
                existing_export_group_key(first),
                existing_export_group_key(second),
            }
        }

        atoms, _limits = build_v6_row_aware_bay_atoms(
            make_problem([first, second, nonexistent_cross], [bay])
        )

        self.assertEqual(
            {"G1", "G2"},
            {atom.group_id for atom in atoms},
        )

    def test_existing_mixed_size_state_remains_strictly_closed(self) -> None:
        group = make_group("G1", port="P1")
        bay = make_bay("01", rows=("1",))
        bay.existing_size_modes = {"20", "40"}
        bay.existing_heights = {"96"}
        bay.existing_group_keys_by_row = {
            "1": {existing_export_group_key(group)}
        }

        atoms, _limits = build_v6_row_aware_bay_atoms(
            make_problem([group], [bay])
        )

        self.assertEqual((), atoms)

    def test_new_boxes_still_cannot_mix_heights_in_one_bay(self) -> None:
        first = make_group("G1", port="P1")
        second = replace(
            first,
            group_id="G2",
            port="P2",
            height="86",
        )
        bay = make_bay("01", rows=("1", "2"))
        bay.existing_size_modes = {"20"}
        bay.existing_heights = {"86", "96"}
        problem = make_problem([first, second], [bay])
        zones = build_complete_v6_zone_universe(problem)
        first_zone = next(
            zone
            for zone in zones
            if zone.group_id == "G1" and zone.resources == (("A|01", "1"),)
        )
        second_zone = next(
            zone
            for zone in zones
            if zone.group_id == "G2" and zone.resources == (("A|01", "2"),)
        )
        evaluator = V6ModelEvaluator(problem, zones)

        with self.assertRaisesRegex(ValueError, "size/height mixing"):
            evaluator.evaluate(
                {first_zone.zone_id, second_zone.zone_id},
                {
                    (first_zone.zone_id, "A|01"): 1,
                    (second_zone.zone_id, "A|01"): 1,
                },
                {},
                V6PeakUtilizationPolicy(0.5, 0.5),
            )

    def test_large_import_checks_every_footprint_bay_size_state(self) -> None:
        anchor = make_bay("02", rows=("1",), row_capacity=1)
        partner = make_bay("03", rows=("1",), row_capacity=1)
        anchor.cap_by_size = {"40": 1}
        anchor.row_cap_by_size = {"40": {"1": 1}}
        anchor.large_bay_partner_key = partner.bay_key
        partner.existing_size_modes = {"20"}
        problem = ProblemData(
            export_groups=[],
            bays={anchor.bay_key: anchor, partner.bay_key: partner},
            area_functions={"A": {"IF"}},
            target_voyages=[],
            export_voyages=set(),
            import_demand_by_flow_size={("IF", "40"): 1},
        )
        planner = DirectMilpPlanner(problem)

        self.assertEqual(0, planner._import_reservation_capacity(anchor.bay_key, "40"))
        self.assertEqual([], planner.import_reservation_candidates[("IF", "40")])

    def test_external_validator_rejects_import_export_bay_sharing(self) -> None:
        group = make_group("G1", port="P1")
        problem = make_problem(
            [group],
            [make_bay("01", rows=("1",), row_capacity=2)],
            import_boxes=1,
        )
        export_rows = [
            {
                "voyage_id": "V1",
                "group_id": "G1",
                "flow": "OF",
                "port": "P1",
                "size": "20",
                "height": "96",
                "area_no": "A",
                "bay_key": "A|01",
                "bay_no": "01",
                "row_no": "1",
                "row_allocation": "A|01:1:1",
                "planned_boxes": 1,
            }
        ]
        import_rows = [
            {
                "flow": "IF",
                "size": "20",
                "area_no": "A",
                "bay_key": "A|01",
                "bay_no": "01",
                "reserved_boxes": 1,
                "footprint_slot_units": 1,
                "reservation_scope": "anonymous_capacity",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            export_path = Path(directory) / "export.csv"
            import_path = Path(directory) / "import.csv"
            write_rows(export_path, export_rows)
            write_rows(import_path, import_rows)
            with self.assertRaisesRegex(ValueError, "import and export share"):
                validate_output_files(problem, export_path, import_path)

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy"),
        "gurobipy is unavailable",
    )
    def test_two_groups_can_share_bay_but_not_physical_row(self) -> None:
        result = DirectMilpPlanner(
            make_problem(
                [make_group("G1", port="P1"), make_group("G2", port="P2")],
                [make_bay("01")],
            ),
            ColumnGenerationConfig(
                total_time_limit=5.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
        ).solve()

        self.assertEqual({"A|01"}, {row["bay_key"] for row in result.export_rows})
        self.assertEqual(2, len({row["row_no"] for row in result.export_rows}))
        self.assertGreater(
            result.diagnostics["direct_model"]["group_row_owner_variable_count"],
            0,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy"),
        "gurobipy is unavailable",
    )
    def test_anonymous_import_and_export_are_segregated_by_bay(self) -> None:
        result = DirectMilpPlanner(
            make_problem(
                [make_group("G1", port="P1", demand=2)],
                [make_bay("01", rows=("1",), row_capacity=2),
                 make_bay("03", rows=("1",), row_capacity=2)],
                import_boxes=2,
            ),
            ColumnGenerationConfig(
                total_time_limit=5.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
        ).solve()

        export_bays = {row["bay_key"] for row in result.export_rows}
        import_bays = {row["bay_key"] for row in result.import_reservation_rows}
        self.assertTrue(export_bays)
        self.assertTrue(import_bays)
        self.assertTrue(export_bays.isdisjoint(import_bays))


if __name__ == "__main__":
    unittest.main()
