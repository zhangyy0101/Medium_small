from __future__ import annotations

import importlib.util
import unittest

from tests.test_v7_complete_mip import v7_exact_config
from tests.test_v7_stage1_area import two_area_problem
from preexperiment.v7_oracle_coverage import (
    audit_stage1_incumbent_coverage,
    select_stage1_domain_for_cap,
)
from yard_planning.v7_atoms import build_v7_row_atoms
from yard_planning.v7_complete_mip import V7CompleteMipSolver
from yard_planning.v7_model import derive_v7_analytic_peak_policy
from yard_planning.v7_stage1_area import V7Stage1AreaSolver, V7Stage1Config


GUROBI_AVAILABLE = importlib.util.find_spec("gurobipy") is not None


class V7OracleCoveragePureTests(unittest.TestCase):
    def test_cap_replay_keeps_best_and_caps_pool_before_bay_local(self) -> None:
        selection = select_stage1_domain_for_cap(
            {
                "mandatory_best_areas": ["A"],
                "pool_candidate_areas": ["B", "C", "D"],
                "bay_local_candidate_areas": ["D", "C", "B"],
            },
            additional_candidate_area_cap=3,
            maximum_pool_candidate_areas=1,
        )

        self.assertEqual(["B"], selection["selected_pool_alternatives"])
        self.assertEqual(["D", "C"], selection["selected_bay_local_areas"])
        self.assertEqual(
            ["A", "B", "C", "D"], selection["final_restricted_areas"]
        )


@unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is unavailable")
class V7OracleCoverageSolverTests(unittest.TestCase):
    def test_tiny_oracle_support_is_covered_and_ranked(self) -> None:
        problem = two_area_problem()
        atoms, _limits = build_v7_row_atoms(problem)
        peak, _diagnostics = derive_v7_analytic_peak_policy(problem, atoms=atoms)
        solver = V7CompleteMipSolver(problem, v7_exact_config())
        witness = solver.solve(peak, feasibility_only=True)
        complete = solver.solve(peak)
        stage1 = V7Stage1AreaSolver(
            problem,
            peak,
            atoms,
            V7Stage1Config(
                time_limit=5.0,
                maximum_pool_solutions=2,
                pool_gap=1.0,
                additional_candidate_area_cap=1,
                maximum_pool_candidate_areas=1,
            ),
        ).solve()

        audit = audit_stage1_incumbent_coverage(
            problem,
            atoms,
            stage1,
            complete,
            witness=witness,
            caps=(1,),
        )

        self.assertEqual(1, audit["complete_mip"]["group_area_pair_count"])
        self.assertEqual(1.0, audit["caps"]["1"]["pair_coverage_fraction"])
        self.assertEqual(1, len(audit["rows"]))
        self.assertTrue(audit["rows"][0]["cap_1_selected"])


if __name__ == "__main__":
    unittest.main()
