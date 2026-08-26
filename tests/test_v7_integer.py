from __future__ import annotations

import importlib.util
import unittest

from tests.test_v6_model_contract import make_bay, make_group, make_problem
from tests.test_v7_complete_mip import v7_exact_config
from yard_planning.v7_bay_patterns import enumerate_v7_bay_patterns
from yard_planning.v7_column_generation import build_full_v7_pattern_universe
from yard_planning.v7_atoms import build_v7_row_atoms, candidate_areas_by_group
from yard_planning.v7_complete_mip import V7CompleteMipSolver
from yard_planning.v7_integer import V7IntegerConfig, V7RestrictedIntegerSolver
from tests.test_v7_stage1_area import two_area_problem


GUROBI_AVAILABLE = importlib.util.find_spec("gurobipy") is not None


@unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is unavailable")
class V7IntegerTests(unittest.TestCase):
    def test_integer_incumbent_is_independently_validated(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1"), make_group("G2", port="P2")],
            [make_bay("01"), make_bay("03")],
            import_boxes=1,
        )
        compact_solver = V7CompleteMipSolver(problem, v7_exact_config())
        compact = compact_solver.solve()
        patterns = build_full_v7_pattern_universe(problem, compact_solver.atoms)

        result = V7RestrictedIntegerSolver(
            problem,
            compact.peak_policy,
            patterns,
            V7IntegerConfig(time_limit=10.0),
            atoms=compact_solver.atoms,
            restricted_areas_by_group=candidate_areas_by_group(
                compact_solver.atoms
            ),
        ).solve()

        self.assertTrue(result.certificate["validation"]["passed"])
        self.assertAlmostEqual(compact.objective, result.objective, places=8)
        self.assertAlmostEqual(
            0.0,
            result.diagnostics["solver_evaluator_objective_difference"],
            places=8,
        )
        self.assertIn("time_to_first_incumbent", result.diagnostics)
        self.assertIn("groups_per_physical_bay_distribution", result.diagnostics)
        self.assertIsNone(result.diagnostics["global_lower_bound"])
        self.assertIsNone(result.diagnostics["global_gap"])

    def test_integer_master_accepts_only_declared_proof_columns_outside_domain(self) -> None:
        problem = two_area_problem()
        atoms, _limits = build_v7_row_atoms(problem)
        compact_solver = V7CompleteMipSolver(problem, v7_exact_config())
        witness = compact_solver.solve(feasibility_only=True)
        search = enumerate_v7_bay_patterns(
            problem, atoms, "A|01", allowed_groups={"G1"}
        )
        proof = enumerate_v7_bay_patterns(
            problem, atoms, "B|01", allowed_groups={"G1"}
        )

        result = V7RestrictedIntegerSolver(
            problem,
            witness.peak_policy,
            [*search, *proof],
            V7IntegerConfig(time_limit=5.0),
            atoms=atoms,
            restricted_areas_by_group={"G1": {"A"}},
            feasibility_proof_patterns=proof,
        ).solve()

        self.assertTrue(result.certificate["validation"]["passed"])
        self.assertEqual(1, result.diagnostics["feasibility_proof_pattern_count"])
        self.assertFalse(result.diagnostics["proof_columns_expand_pricing_domain"])


if __name__ == "__main__":
    unittest.main()
