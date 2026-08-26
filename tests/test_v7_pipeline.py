from __future__ import annotations

import importlib.util
import unittest

from tests.test_v6_model_contract import make_bay, make_group, make_problem
from yard_planning.v7_pipeline import V7Config, solve_v7


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
                peak_feasibility_time_limit=5.0,
                stage1_time_limit=5.0,
                root_time_limit=10.0,
                integer_time_limit=5.0,
            ),
        )

        self.assertTrue(result.witness.certificate["validation"]["passed"])
        self.assertTrue(result.root.diagnostics["root_closed"])
        self.assertTrue(result.integer.certificate["validation"]["passed"])
        self.assertFalse(result.stage1.quota_fixed_in_stage2)
        self.assertFalse(result.diagnostics["complete_pattern_enumeration_used"])
        self.assertFalse(result.diagnostics["performance_experiments_run"])
        self.assertAlmostEqual(
            result.integer.objective, result.objective, places=10
        )


if __name__ == "__main__":
    unittest.main()
