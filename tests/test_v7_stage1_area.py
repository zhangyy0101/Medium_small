from __future__ import annotations

import importlib.util
import unittest

from tests.test_v6_model_contract import make_bay, make_group, make_problem
from yard_planning.models import ProblemData
from yard_planning.v7_atoms import build_v7_row_atoms
from yard_planning.v7_model import derive_v7_analytic_peak_policy
from yard_planning.v7_stage1_area import (
    V7Stage1AreaSolver,
    V7Stage1Config,
    stage1_graph_diagnostics,
)


GUROBI_AVAILABLE = importlib.util.find_spec("gurobipy") is not None


def two_area_problem(demand: int = 1) -> ProblemData:
    group = make_group("G1", port="P1", demand=demand)
    bay_a = make_bay("01", rows=("1",))
    bay_b = make_bay("01", rows=("1",))
    bay_b.area_no = "B"
    bay_b.bay_key = "B|01"
    return ProblemData(
        export_groups=[group],
        bays={bay_a.bay_key: bay_a, bay_b.bay_key: bay_b},
        area_functions={"A": {"OF"}, "B": {"OF"}},
        target_voyages=["V1"],
        export_voyages={"V1"},
        berth_distances={("A", "Q1"): 1.0, ("B", "Q1"): 3.0},
        berth_by_voyage={"V1": "Q1"},
    )


@unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is unavailable")
class V7Stage1Tests(unittest.TestCase):
    def test_stage1_returns_guidance_not_fixed_quota(self) -> None:
        problem = two_area_problem()
        atoms, _limits = build_v7_row_atoms(problem)
        peak, _diagnostics = derive_v7_analytic_peak_policy(problem, atoms=atoms)

        result = V7Stage1AreaSolver(
            problem,
            peak,
            atoms,
            V7Stage1Config(
                time_limit=5.0,
                maximum_pool_solutions=4,
                pool_gap=1.0,
                initial_candidate_area_cap=1,
            ),
        ).solve()

        self.assertFalse(result.quota_fixed_in_stage2)
        self.assertFalse(result.diagnostics["stage2_quota_fixed"])
        self.assertFalse(result.diagnostics["candidate_cap_is_business_constraint"])
        self.assertGreaterEqual(len(result.pool_solutions), 1)
        self.assertEqual(1, len(result.active_areas_by_group["G1"]))

    def test_cap_never_removes_areas_required_by_best_solution(self) -> None:
        problem = two_area_problem(demand=2)
        atoms, _limits = build_v7_row_atoms(problem)
        peak, _diagnostics = derive_v7_analytic_peak_policy(problem, atoms=atoms)

        result = V7Stage1AreaSolver(
            problem,
            peak,
            atoms,
            V7Stage1Config(initial_candidate_area_cap=1, time_limit=5.0),
        ).solve()

        self.assertEqual({"A", "B"}, set(result.active_areas_by_group["G1"]))
        self.assertEqual(2, sum(result.best_group_area_quantity.values()))

    def test_graph_diagnostics_distinguish_full_and_active_domains(self) -> None:
        problem = two_area_problem()
        atoms, _limits = build_v7_row_atoms(problem)

        diagnostics = stage1_graph_diagnostics(atoms, {"G1": {"A"}})

        self.assertEqual(
            2, diagnostics["full_legal_domain"]["group_area_edge_count"]
        )
        self.assertEqual(
            1, diagnostics["stage1_active_domain"]["group_area_edge_count"]
        )
        self.assertAlmostEqual(0.5, diagnostics["active_edge_reduction_ratio"])
        self.assertIn("fraction_active_le_4", diagnostics)


if __name__ == "__main__":
    unittest.main()
