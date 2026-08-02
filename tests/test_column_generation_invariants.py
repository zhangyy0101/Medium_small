from __future__ import annotations

import unittest
from collections import defaultdict

from block_bay_planning.models import Bay, SmallBoxGroup
from medium_small.column_generation_planner import ColumnGenerationConfig, ColumnGenerationPlanner


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


class FakeVariable:
    def __init__(self, reduced_cost: float) -> None:
        self.RC = reduced_cost


class FakeModel:
    @staticmethod
    def getReducedCost(variable: FakeVariable) -> float:
        return variable.RC


class ColumnGenerationInvariantTests(unittest.TestCase):
    def test_silent_solver_fallback_is_disabled_by_default(self) -> None:
        self.assertFalse(ColumnGenerationConfig().allow_greedy_fallback)

    def test_exact_pricing_activates_every_negative_reduced_cost_column(self) -> None:
        planner = ColumnGenerationPlanner.__new__(ColumnGenerationPlanner)
        planner._active_column_indices = {0}
        variables = {
            0: FakeVariable(-100.0),  # already active; it is not priced again
            1: FakeVariable(-0.25),
            2: FakeVariable(0.0),
            3: FakeVariable(-1e-9),  # inside numerical tolerance
        }

        stats = planner._activate_negative_reduced_cost_columns(
            FakeModel(), {"column": variables}
        )

        self.assertEqual({0, 1}, planner._active_column_indices)
        self.assertEqual(1, stats["new_columns"])
        self.assertTrue(stats["exact_pricing"])

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


if __name__ == "__main__":
    unittest.main()
