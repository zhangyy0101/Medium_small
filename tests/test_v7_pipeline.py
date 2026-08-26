from __future__ import annotations

import importlib.util
import unittest

from tests.test_v6_model_contract import make_bay, make_group, make_problem
from yard_planning.v7_pipeline import (
    V7AlgorithmTimeBudget,
    V7Config,
    solve_v7,
)


GUROBI_AVAILABLE = importlib.util.find_spec("gurobipy") is not None


@unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is unavailable")
class V7PipelineTests(unittest.TestCase):
    def test_tiny_two_stage_pipeline_returns_valid_integer_incumbent(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1"), make_group("G2", port="P2")],
            [make_bay("01"), make_bay("03")],
            import_boxes=1,
        )

        result = solve_v7(
            problem,
            V7Config(
                algorithm_time_limit=25.0,
                minimum_integer_time=1.0,
                peak_feasibility_time_limit=5.0,
                stage1_time_limit=5.0,
                root_time_limit=10.0,
            ),
        )

        self.assertTrue(result.witness.certificate["validation"]["passed"])
        self.assertEqual("feasibility", result.witness.diagnostics["objective_mode"])
        self.assertIsNone(result.witness.diagnostics["certified_gap"])
        self.assertTrue(result.root.diagnostics["restricted_root_closed"])
        self.assertFalse(result.root.diagnostics["global_root_certified"])
        self.assertTrue(result.integer.certificate["validation"]["passed"])
        self.assertFalse(result.stage1.quota_fixed_in_stage2)
        self.assertFalse(result.diagnostics["complete_pattern_enumeration_used"])
        self.assertFalse(result.diagnostics["performance_experiments_run"])
        self.assertFalse(result.global_pricing_audit.executed)
        self.assertIsNone(result.global_lower_bound)
        self.assertIsNone(result.global_gap)
        self.assertIsNone(result.diagnostics["global_lower_bound"])
        self.assertIsNone(result.diagnostics["global_gap"])
        self.assertAlmostEqual(
            result.root.restricted_lp_bound,
            result.restricted_lp_bound,
            places=10,
        )
        self.assertAlmostEqual(
            result.integer.objective, result.objective, places=10
        )
        budget = result.diagnostics["algorithm_time_budget"]
        self.assertIsNone(budget["integer_optional_ceiling"])
        self.assertGreater(budget["allocated_time_limits"]["integer"], 0.0)
        self.assertTrue(
            result.integer.diagnostics["shared_wall_clock_deadline_used"]
        )

    def test_shared_budget_gives_final_stage_all_remaining_time(self) -> None:
        budget = V7AlgorithmTimeBudget(
            total_time_limit=120.0,
            minimum_integer_time=5.0,
        )
        budget.validate()

        witness = budget.soft_stage_limit(
            "witness",
            elapsed_seconds=0.0,
            soft_limit=10.0,
            future_reserve=5.2,
        )
        stage1 = budget.soft_stage_limit(
            "Stage 1",
            elapsed_seconds=3.5,
            soft_limit=10.0,
            future_reserve=5.1,
        )
        root = budget.soft_stage_limit(
            "root",
            elapsed_seconds=13.5,
            soft_limit=60.0,
            future_reserve=5.0,
        )
        integer = budget.integer_stage_limit(elapsed_seconds=22.0)

        self.assertEqual(10.0, witness)
        self.assertEqual(10.0, stage1)
        self.assertEqual(60.0, root)
        self.assertEqual(98.0, integer)

    def test_shared_budget_rejects_impossible_reservation(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot reserve"):
            V7AlgorithmTimeBudget(
                total_time_limit=5.0,
                minimum_integer_time=5.0,
            ).validate()


if __name__ == "__main__":
    unittest.main()
