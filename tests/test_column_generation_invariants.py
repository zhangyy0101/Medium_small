from __future__ import annotations

import unittest
import csv
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from yard_planning.models import Bay, ExportGroup
from yard_planning.planner import ColumnGenerationConfig, ColumnGenerationPlanner, PlacementColumn
from yard_planning.output_validator import validate_output_files


def make_group(size: str) -> ExportGroup:
    return ExportGroup(
        group_id=f"g-{size}",
        voyage_id="V1",
        status="OF",
        port="P1",
        size=size,
        height="96",
        demand=5,
    )


class ColumnGenerationInvariantTests(unittest.TestCase):
    def test_pricing_configuration_requires_negative_reduced_cost(self) -> None:
        config = ColumnGenerationConfig()
        self.assertGreater(config.max_columns_per_group_per_iteration, 0)
        self.assertGreater(config.reduced_cost_tolerance, 0.0)

    def test_normalized_policy_weights_sum_to_one_and_follow_priority(self) -> None:
        config = ColumnGenerationConfig()
        concentration = (
            config.area_dispersion_weight
            + config.row_dispersion_weight
            + config.existing_group_proximity_weight
        )
        weights = [
            config.area_dispersion_weight,
            config.row_dispersion_weight,
            config.existing_group_proximity_weight,
            config.area_guidance_weight,
            config.berth_distance_weight,
        ]
        self.assertAlmostEqual(1.0, sum(weights))
        self.assertGreater(concentration, config.area_guidance_weight)
        self.assertGreater(config.area_guidance_weight, config.berth_distance_weight)

    def test_45ft_is_edge_only_without_excluding_other_sizes(self) -> None:
        planner = ColumnGenerationPlanner.__new__(ColumnGenerationPlanner)
        planner.bays = {
            "edge": Bay(
                area_no="A",
                bay_no="01",
                bay_key="edge",
                bay_order=0,
                cap_by_size={"20": 10, "40": 10, "45": 10},
                physical_capacity=10,
                large_bay_partner_key="partner",
            ),
            "middle": Bay(
                area_no="A",
                bay_no="03",
                bay_key="middle",
                bay_order=1,
                cap_by_size={"20": 10, "40": 10, "45": 10},
                physical_capacity=10,
                large_bay_partner_key="partner",
            ),
            "partner": Bay(
                area_no="A",
                bay_no="02",
                bay_key="partner",
                bay_order=2,
                cap_by_size={"20": 10, "40": 10, "45": 10},
                physical_capacity=10,
            ),
        }
        planner.area_edge_bays = defaultdict(set, {"A": {"edge"}})
        planner._bay_existing_attrs_allow_group = lambda group, footprint: True
        planner._stack_count_for_group = lambda bay_key, size, group: 10
        planner._stack_unit_capacity_for_group = lambda bay_key, size, group: 1

        self.assertEqual(5, planner._max_quantity_in_bay(make_group("20"), "edge"))
        self.assertEqual(0, planner._max_quantity_in_bay(make_group("45"), "middle"))
        self.assertEqual(5, planner._max_quantity_in_bay(make_group("45"), "edge"))

    def test_pricing_enumerates_candidates_without_materializing_universe(self) -> None:
        planner = ColumnGenerationPlanner.__new__(ColumnGenerationPlanner)
        planner.groups = [make_group("20")]
        planner.config = ColumnGenerationConfig()
        planner.bays = {
            "A|01": Bay(
                area_no="A", bay_no="01", bay_key="A|01",
                bay_order=0, cap_by_size={"20": 9}, physical_capacity=9,
            )
        }
        planner._columns = []
        planner._column_keys = set()
        planner._column_indices_by_triplet = defaultdict(list)
        planner._candidate_bays_for_group = lambda group: [("A|01", 9, 0.0)]
        planner._placement_footprint_keys = lambda bay_key, size: (bay_key,)
        planner._row_capacity_items_for_group = lambda bay_key, size, group: [(str(i), 3) for i in range(1, 10)]
        planner._quota_key = lambda group, area: (group.voyage_id, group.status, area, group.size)
        planner._operational_group_key = lambda group: (group.voyage_id, group.status, group.port, group.size, group.height)
        planner._column_base_cost = lambda group, bay: 0.0
        planner._berth_distance_cost = lambda voyage, area, qty: 0.0

        candidates = list(planner._iter_feasible_unit_flow_columns(planner.groups[0]))

        self.assertEqual(9, len(candidates))
        self.assertTrue(all(column.quantity == 1 for column in candidates))
        self.assertEqual([], planner._columns)

        index = planner._append_generated_column(candidates[0])
        self.assertEqual(0, index)
        self.assertEqual(1, len(planner._columns))
        self.assertEqual("C0000001", planner._columns[0].column_id)

    def test_pricing_reduced_cost_uses_master_row_duals(self) -> None:
        planner = ColumnGenerationPlanner.__new__(ColumnGenerationPlanner)
        planner._placement_footprint_keys = lambda bay_key, size: (bay_key,)
        planner._row_mix_key_for_column = lambda column: "mix"
        planner._bay_no_mix_attrs_for_column = lambda column: ("height",)
        planner._row_no_mix_attrs_for_column = lambda column: ("voyage",)
        planner._attr_voyage_scope = lambda attr, voyage: "scope"
        planner._column_attr_value = lambda column, attr: f"{attr}-value"
        column = PlacementColumn(
            column_id="",
            group_id="G1",
            voyage_id="V1",
            flow="OF",
            port="P1",
            size="20",
            big_plan_size="20",
            height="96",
            attributes={},
            area_no="A",
            bay_key="B1",
            bay_no="01",
            quantity=1,
            stack_units=1,
            row_allocation=(("B1", "R1", 1),),
            quota_key=("V1", "OF", "A", "20"),
            group_key=("V1", "OF", "P1", "20", "96"),
            intrinsic_cost=100.0,
        )
        constraints = {
            "group_cover": {"G1": 2.0},
            "bay_capacity_limit": {"B1": 3.0},
            "bay_port_stack_link": {("B1", "mix", "20"): 5.0},
            "bay_attr_link": {("B1", "height", "scope", "height-value"): 7.0},
            "bay_size_limit": {("B1", "20"): 11.0},
            "row_capacity_limit": {("B1", "R1"): 13.0},
            "row_size_limit": {("B1", "R1", "20"): 17.0},
            "row_attr_link": {("B1", "R1", "voyage", "scope", "voyage-value"): 19.0},
            "area_guidance_balance": {("V1", "OF", "A", "20"): 23.0},
            "fixed_use_objective_limit": {
                ("group_area", column.group_key, "A"): 29.0,
                ("group_row", column.group_key, "B1", "R1"): 31.0,
                ("group_used_upper", *column.group_key): 37.0,
                ("group_used_lower", *column.group_key): 41.0,
            },
        }
        model = SimpleNamespace(getDualsolLinear=lambda constraint: constraint)

        reduced_cost = planner._column_reduced_cost(model, constraints, column, "full")

        self.assertAlmostEqual(-56.0, reduced_cost)

    def test_written_output_is_validated_independently(self) -> None:
        group = make_group("20")
        bay = Bay(
            area_no="A", bay_no="01", bay_key="A|01",
            bay_order=0, cap_by_size={"20": 4}, physical_capacity=4,
            row_cap_by_size={"20": {"1": 4}}, row_physical_capacity={"1": 4},
        )
        problem = SimpleNamespace(
            export_groups=[group], bays={"A|01": bay},
            import_area_size_reference={}, area_functions={"A": {"OF"}},
        )
        with TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.csv"
            unplaced_path = Path(directory) / "unplaced.csv"
            import_path = Path(directory) / "import.csv"
            with plan_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "group_id", "planned_boxes", "area_no", "bay_no", "row_no",
                    "size", "height", "voyage_id", "port",
                ])
                writer.writeheader()
                writer.writerow({
                    "group_id": group.group_id, "planned_boxes": 5, "area_no": "A",
                    "bay_no": "01", "row_no": "1", "size": "20", "height": "96",
                    "voyage_id": "V1", "port": "P1",
                })
            unplaced_path.touch()
            import_path.touch()
            with self.assertRaisesRegex(ValueError, "capacity"):
                validate_output_files(problem, plan_path, unplaced_path, import_path)

    def test_import_reservation_does_not_inherit_existing_size_no_mix(self) -> None:
        bay = Bay(
            area_no="A", bay_no="01", bay_key="A|01",
            bay_order=0, cap_by_size={"20": 4, "40": 4}, physical_capacity=4,
            row_cap_by_size={"20": {"1": 4}}, row_physical_capacity={"1": 4},
            existing_size_modes={"40"},
        )
        planner = ColumnGenerationPlanner.__new__(ColumnGenerationPlanner)
        planner.bays = {"A|01": bay}
        self.assertEqual(4, planner._import_reservation_capacity("A|01", "20"))
        problem = SimpleNamespace(
            export_groups=[], bays={"A|01": bay},
            import_area_size_reference={("IF", "A", "20"): 1},
            area_functions={"A": {"IF"}},
        )
        with TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.csv"
            unplaced_path = Path(directory) / "unplaced.csv"
            import_path = Path(directory) / "import.csv"
            plan_path.touch()
            unplaced_path.touch()
            with import_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "flow", "size", "area_no", "bay_key", "bay_no", "reserved_boxes",
                ])
                writer.writeheader()
                writer.writerow({
                    "flow": "IF", "size": "20", "area_no": "A",
                    "bay_key": "A|01", "bay_no": "01", "reserved_boxes": 1,
                })
            result = validate_output_files(problem, plan_path, unplaced_path, import_path)
            self.assertTrue(result["passed"])

    def test_import_reservation_requires_area_function(self) -> None:
        bay = Bay(
            area_no="A", bay_no="01", bay_key="A|01",
            bay_order=0, cap_by_size={"20": 4}, physical_capacity=4,
        )
        problem = SimpleNamespace(
            export_groups=[], bays={"A|01": bay},
            import_area_size_reference={("IF", "B", "20"): 1},
            area_functions={"A": {"OF"}},
        )
        with TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.csv"
            unplaced_path = Path(directory) / "unplaced.csv"
            import_path = Path(directory) / "import.csv"
            plan_path.touch()
            unplaced_path.touch()
            with import_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "flow", "size", "area_no", "bay_key", "bay_no", "reserved_boxes",
                ])
                writer.writeheader()
                writer.writerow({
                    "flow": "IF", "size": "20", "area_no": "A",
                    "bay_key": "A|01", "bay_no": "01", "reserved_boxes": 1,
                })
            with self.assertRaisesRegex(ValueError, "area-function"):
                validate_output_files(problem, plan_path, unplaced_path, import_path)


if __name__ == "__main__":
    unittest.main()
