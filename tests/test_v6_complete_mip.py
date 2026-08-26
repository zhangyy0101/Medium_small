from __future__ import annotations

import importlib.util
import unittest

from tests.test_v6_model_contract import make_bay, make_group, make_problem
from yard_planning.models import ProblemData
from yard_planning.v6_complete_mip import (
    V6CompleteMipConfig,
    V6CompleteMipSolver,
)
from yard_planning.v6_model import v6_export_group_key


GUROBI_AVAILABLE = importlib.util.find_spec("gurobipy") is not None


def exact_config(**overrides) -> V6CompleteMipConfig:
    values = {
        "peak_time_limit": 10.0,
        "business_time_limit": 10.0,
        "business_mip_gap": 0.0,
        "solver_threads": 1,
        "solver_seed": 0,
        "verbose": False,
        "maximum_zone_count": 10_000,
    }
    values.update(overrides)
    return V6CompleteMipConfig(**values)


class V6CompleteMipTests(unittest.TestCase):
    def test_enumeration_safety_limit_raises_instead_of_truncating(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1")],
            [make_bay("01")],
        )

        with self.assertRaisesRegex(RuntimeError, "was not truncated"):
            V6CompleteMipSolver(
                problem,
                exact_config(maximum_zone_count=1),
            )

    @unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is unavailable")
    def test_hand_computable_peak_and_zero_business_objective(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1")],
            [make_bay("01")],
        )

        result = V6CompleteMipSolver(problem, exact_config()).solve()

        self.assertAlmostEqual(
            0.5,
            result.peak_policy.minimum_feasible_utilization,
        )
        self.assertAlmostEqual(0.75, result.peak_policy.epsilon_cap)
        self.assertEqual(0.0, result.certificate["objective"])
        self.assertEqual("optimal", result.diagnostics["peak_reference"]["status"])
        self.assertEqual("optimal", result.diagnostics["business_model"]["status"])
        self.assertAlmostEqual(
            result.certificate["objective"],
            result.diagnostics["business_model"]["solver_objective"],
        )

    @unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is unavailable")
    def test_all_three_objective_categories_match_hand_value(self) -> None:
        group = make_group("G1", port="P1", demand=2)
        bay_a = make_bay("01", rows=("1",))
        bay_b = make_bay("01", rows=("1",))
        bay_b.area_no = "B"
        bay_b.bay_key = "B|01"
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

        result = V6CompleteMipSolver(problem, exact_config()).solve()

        self.assertAlmostEqual(1.0, result.peak_policy.minimum_feasible_utilization)
        self.assertAlmostEqual(0.64375, result.certificate["objective"])
        self.assertEqual(
            {
                "extra_contiguous_zones": 1.0,
                "extra_voyage_areas": 1.0,
                "existing_group_normalized_distance_sum": 1.0,
                "berth_normalized_distance_sum": 1.0,
                "unused_reserved_capacity_boxes": 0.0,
            },
            result.certificate["raw"],
        )

    @unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is unavailable")
    def test_exact_peak_reference_forces_balanced_business_allocation(self) -> None:
        group = make_group("G1", port="P1", demand=2)
        bay_a = make_bay("01", rows=("1",), row_capacity=2)
        bay_b = make_bay("01", rows=("1",), row_capacity=2)
        bay_b.area_no = "B"
        bay_b.bay_key = "B|01"
        problem = ProblemData(
            export_groups=[group],
            bays={bay_a.bay_key: bay_a, bay_b.bay_key: bay_b},
            area_functions={"A": {"OF"}, "B": {"OF"}},
            target_voyages=["V1"],
            export_voyages={"V1"},
            berth_distances={("A", "Q1"): 1.0, ("B", "Q1"): 1.0},
            berth_by_voyage={"V1": "Q1"},
        )

        result = V6CompleteMipSolver(problem, exact_config()).solve()

        self.assertAlmostEqual(0.5, result.peak_policy.minimum_feasible_utilization)
        self.assertAlmostEqual(0.75, result.peak_policy.epsilon_cap)
        self.assertEqual(
            {"A", "B"},
            {
                result.zones[zone_id].area_no
                for zone_id in result.selected_zone_ids
            },
        )
        self.assertAlmostEqual(0.7125, result.certificate["objective"])

    @unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is unavailable")
    def test_groups_share_bay_by_rows_while_import_uses_another_bay(self) -> None:
        problem = make_problem(
            [
                make_group("G1", port="P1"),
                make_group("G2", port="P2"),
            ],
            [make_bay("01"), make_bay("03")],
            import_boxes=1,
        )

        result = V6CompleteMipSolver(problem, exact_config()).solve()
        zones_by_id = {zone.zone_id: zone for zone in result.zones}
        export_bays = {
            bay_key
            for zone_id in result.selected_zone_ids
            for bay_key in zones_by_id[zone_id].physical_bay_keys
        }
        export_rows = {
            resource
            for zone_id in result.selected_zone_ids
            for resource in zones_by_id[zone_id].resources
        }
        import_bays = {
            bay_key
            for (_flow, _size, bay_key), quantity in result.import_reservation.items()
            if int(quantity) > 0
        }

        self.assertEqual(2, len(export_rows))
        self.assertEqual(1, len(export_bays))
        self.assertEqual(1, len(import_bays))
        self.assertTrue(export_bays.isdisjoint(import_bays))
        self.assertTrue(result.certificate["validation"]["passed"])

    @unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is unavailable")
    def test_40ft_complete_mip_reserves_both_footprint_members(self) -> None:
        group = make_group("G1", port="P1", demand=2)
        group = type(group)(
            **{
                **group.__dict__,
                "size": "40",
            }
        )
        bays = [
            make_bay(code, rows=("1", "2"))
            for code in ("01", "03", "05", "07")
        ]
        for bay in bays:
            bay.cap_by_size = {"40": 1}
            bay.row_cap_by_size = {"40": {"1": 1, "2": 1}}
        bays[0].large_bay_partner_key = bays[1].bay_key
        bays[2].large_bay_partner_key = bays[3].bay_key

        result = V6CompleteMipSolver(
            make_problem([group], bays),
            exact_config(),
        ).solve()
        zones_by_id = {zone.zone_id: zone for zone in result.zones}
        physical_bays = {
            bay_key
            for zone_id in result.selected_zone_ids
            for bay_key in zones_by_id[zone_id].physical_bay_keys
        }

        self.assertEqual(4, len(physical_bays))
        self.assertEqual(0.0, result.certificate["objective"])
        self.assertTrue(result.certificate["validation"]["passed"])


if __name__ == "__main__":
    unittest.main()
