from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from yard_planning.direct_milp import DirectMilpPlanner
from yard_planning.models import AttributeRules, Bay, ExportGroup, ProblemData
from yard_planning.planner import ColumnGenerationConfig, PlacementColumn
from yard_planning.output_validator import validate_output_files
from yard_planning.planner import write_rows
from yard_planning.row_aware_zones import (
    build_complete_row_aware_zone_universe,
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
