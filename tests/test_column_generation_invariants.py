from __future__ import annotations

import unittest
import csv
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from block_bay_planning.models import Bay, SmallBoxGroup
from medium_small.column_generation_planner import ColumnGenerationConfig, ColumnGenerationPlanner
from medium_small.output_validator import validate_output_files


def make_group(size: str) -> SmallBoxGroup:
    return SmallBoxGroup(
        group_id=f"g-{size}",
        voyage_id="V1",
        status="OF",
        port="P1",
        size=size,
        height="96",
        weight_class="",
        demand=5,
    )


class ColumnGenerationInvariantTests(unittest.TestCase):
    def test_silent_solver_fallback_is_disabled_by_default(self) -> None:
        self.assertFalse(ColumnGenerationConfig().allow_greedy_fallback)

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
                block_id="",
                block_bays=(),
                block_bay_count=0,
                block_boundary_adjusted=False,
                bay_order=0,
                cap_by_size={"20": 10, "40": 10, "45": 10},
                physical_capacity=10,
                large_bay_partner_key="partner",
            ),
            "middle": Bay(
                area_no="A",
                bay_no="03",
                bay_key="middle",
                block_id="",
                block_bays=(),
                block_bay_count=0,
                block_boundary_adjusted=False,
                bay_order=1,
                cap_by_size={"20": 10, "40": 10, "45": 10},
                physical_capacity=10,
                large_bay_partner_key="partner",
            ),
            "partner": Bay(
                area_no="A",
                bay_no="02",
                bay_key="partner",
                block_id="",
                block_bays=(),
                block_bay_count=0,
                block_boundary_adjusted=False,
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

    def test_unit_flows_cover_every_feasible_row_without_pattern_cap(self) -> None:
        planner = ColumnGenerationPlanner.__new__(ColumnGenerationPlanner)
        planner.groups = [make_group("20")]
        planner.config = ColumnGenerationConfig(initial_columns_per_group=1)
        planner.bays = {
            "A|01": Bay(
                area_no="A", bay_no="01", bay_key="A|01", block_id="",
                block_bays=(), block_bay_count=0, block_boundary_adjusted=False,
                bay_order=0, cap_by_size={"20": 9}, physical_capacity=9,
            )
        }
        planner._columns = []
        planner._column_keys = set()
        planner._column_indices_by_triplet = defaultdict(list)
        planner._active_column_indices = set()
        planner._candidate_bays_for_group = lambda group: [("A|01", 9, 0.0)]
        planner._placement_footprint_keys = lambda bay_key, size: (bay_key,)
        planner._row_capacity_items_for_group = lambda bay_key, size, group: [(str(i), 3) for i in range(1, 10)]
        planner._quota_key = lambda group, area: (group.voyage_id, group.status, area, group.size)
        planner._operational_group_key = lambda group: (group.voyage_id, group.status, group.port, group.size, group.height)
        planner._berth_distance_cost = lambda voyage, area, qty: 0.0

        planner._build_unit_flow_column_universe()

        self.assertEqual(9, len(planner._columns))
        self.assertTrue(all(column.quantity == 1 for column in planner._columns))
        self.assertEqual(set(range(9)), planner._active_column_indices)

    def test_written_output_is_validated_independently(self) -> None:
        group = make_group("20")
        bay = Bay(
            area_no="A", bay_no="01", bay_key="A|01", block_id="",
            block_bays=(), block_bay_count=0, block_boundary_adjusted=False,
            bay_order=0, cap_by_size={"20": 4}, physical_capacity=4,
            row_cap_by_size={"20": {"1": 4}}, row_physical_capacity={"1": 4},
        )
        problem = SimpleNamespace(
            small_groups=[group], bays={"A|01": bay},
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
            area_no="A", bay_no="01", bay_key="A|01", block_id="",
            block_bays=(), block_bay_count=0, block_boundary_adjusted=False,
            bay_order=0, cap_by_size={"20": 4, "40": 4}, physical_capacity=4,
            row_cap_by_size={"20": {"1": 4}}, row_physical_capacity={"1": 4},
            existing_size_modes={"40"},
        )
        planner = ColumnGenerationPlanner.__new__(ColumnGenerationPlanner)
        planner.bays = {"A|01": bay}
        self.assertEqual(4, planner._import_reservation_capacity("A|01", "20"))
        problem = SimpleNamespace(
            small_groups=[], bays={"A|01": bay},
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
            area_no="A", bay_no="01", bay_key="A|01", block_id="",
            block_bays=(), block_bay_count=0, block_boundary_adjusted=False,
            bay_order=0, cap_by_size={"20": 4}, physical_capacity=4,
        )
        problem = SimpleNamespace(
            small_groups=[], bays={"A|01": bay},
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
