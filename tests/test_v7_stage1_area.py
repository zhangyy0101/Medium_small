from __future__ import annotations

import importlib.util
import unittest

from tests.test_v6_model_contract import make_bay, make_group, make_problem
from yard_planning.models import ProblemData
from yard_planning.v7_atoms import build_v7_row_atoms
from yard_planning.v7_model import V7ModelEvaluator, derive_v7_analytic_peak_policy
from yard_planning.v7_stage1_area import (
    V7Stage1AreaSolver,
    V7Stage1Config,
    V7Stage1IncompleteError,
    V7Stage1PoolSolution,
    build_v7_bay_local_area_candidates,
    build_v7_restricted_area_domain,
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
                additional_candidate_area_cap=1,
            ),
        ).solve()

        self.assertFalse(result.quota_fixed_in_stage2)
        self.assertFalse(result.diagnostics["stage2_quota_fixed"])
        self.assertFalse(result.diagnostics["candidate_cap_is_business_constraint"])
        self.assertGreaterEqual(len(result.pool_solutions), 1)
        self.assertEqual(2, len(result.restricted_areas_by_group["G1"]))
        self.assertEqual(2, result.diagnostics["unique_y_support_count"])
        self.assertTrue(result.diagnostics["pool_distinguished_by_y_support"])
        with self.assertRaises(TypeError):
            result.restricted_areas_by_group["G1"] = frozenset({"B"})

    def test_cap_never_removes_areas_required_by_best_solution(self) -> None:
        problem = two_area_problem(demand=2)
        atoms, _limits = build_v7_row_atoms(problem)
        peak, _diagnostics = derive_v7_analytic_peak_policy(problem, atoms=atoms)

        result = V7Stage1AreaSolver(
            problem,
            peak,
            atoms,
            V7Stage1Config(additional_candidate_area_cap=1, time_limit=5.0),
        ).solve()

        self.assertEqual(
            {"A", "B"}, set(result.restricted_areas_by_group["G1"])
        )
        self.assertEqual(2, sum(result.best_group_area_quantity.values()))

    def test_stage1_domain_has_no_witness_support_channel(self) -> None:
        problem = two_area_problem()
        atoms, _limits = build_v7_row_atoms(problem)
        peak, _diagnostics = derive_v7_analytic_peak_policy(problem, atoms=atoms)

        result = V7Stage1AreaSolver(
            problem,
            peak,
            atoms,
            V7Stage1Config(
                additional_candidate_area_cap=1,
                maximum_pool_solutions=1,
                time_limit=5.0,
            ),
        ).solve()

        group = result.diagnostics["candidate_area_diagnostics"]["groups"][0]
        self.assertNotIn("mandatory_witness_areas", group)
        self.assertEqual(
            "mandatory_stage1_best_plus_capped_pool_and_bay_local",
            result.diagnostics["candidate_area_policy"],
        )

    def test_cap_only_trims_optional_pool_areas(self) -> None:
        pool = (
            V7Stage1PoolSolution(
                solution_number=0,
                objective=0.0,
                group_area_quantity={("G1", "A"): 1},
                group_area_bay_incidence={("G1", "A"): 1},
                used_areas_by_group={"G1": ("A",)},
            ),
            V7Stage1PoolSolution(
                solution_number=1,
                objective=0.1,
                group_area_quantity={("G1", "C"): 1},
                group_area_bay_incidence={("G1", "C"): 1},
                used_areas_by_group={"G1": ("C",)},
            ),
            V7Stage1PoolSolution(
                solution_number=2,
                objective=0.2,
                group_area_quantity={("G1", "C"): 1, ("G1", "D"): 1},
                group_area_bay_incidence={("G1", "C"): 1, ("G1", "D"): 1},
                used_areas_by_group={"G1": ("C", "D")},
            ),
        )

        restricted, diagnostics = build_v7_restricted_area_domain(
            ["G1"],
            {"G1": {"A", "B", "C", "D"}},
            pool,
            additional_candidate_area_cap=1,
            maximum_pool_candidate_areas=1,
        )

        self.assertEqual({"A", "C"}, set(restricted["G1"]))
        group = diagnostics["groups"][0]
        self.assertEqual(["C", "D"], group["pool_candidate_areas"])
        self.assertEqual(["C"], group["selected_pool_alternatives"])

    def test_bay_local_area_fills_non_pool_channel(self) -> None:
        pool = (
            V7Stage1PoolSolution(
                solution_number=0,
                objective=0.0,
                group_area_quantity={("G1", "A"): 1},
                group_area_bay_incidence={("G1", "A"): 1},
                used_areas_by_group={"G1": ("A",)},
            ),
        )

        restricted, diagnostics = build_v7_restricted_area_domain(
            ["G1"],
            {"G1": {"A", "B", "C"}},
            pool,
            additional_candidate_area_cap=1,
            maximum_pool_candidate_areas=0,
            bay_local_areas_by_group={"G1": ("C", "B")},
        )

        self.assertEqual({"A", "C"}, set(restricted["G1"]))
        self.assertEqual(
            ["C"],
            diagnostics["groups"][0]["selected_bay_local_areas"],
        )

    def test_bay_local_ranking_uses_actual_bay_level_objective(self) -> None:
        problem = two_area_problem()
        atoms, _limits = build_v7_row_atoms(problem)
        evaluator = V7ModelEvaluator(problem, atoms)

        rankings, candidates = build_v7_bay_local_area_candidates(
            problem, atoms, evaluator
        )

        self.assertEqual(("A", "B"), rankings["G1"])
        self.assertEqual(0, candidates[("G1", "A")].shortage_boxes)
        self.assertLess(
            candidates[("G1", "A")].local_objective,
            candidates[("G1", "B")].local_objective,
        )

    def test_packing_proxy_rejects_four_groups_on_one_usable_bay(self) -> None:
        problem = make_problem(
            [make_group(f"G{i}", port=f"P{i}") for i in range(4)],
            [make_bay("01", rows=("1", "2", "3", "4"))],
        )
        atoms, _limits = build_v7_row_atoms(problem)
        peak, _diagnostics = derive_v7_analytic_peak_policy(problem, atoms=atoms)

        with self.assertRaises(V7Stage1IncompleteError) as raised:
            V7Stage1AreaSolver(
                problem,
                peak,
                atoms,
                V7Stage1Config(time_limit=5.0),
            ).solve()
        self.assertEqual(
            1,
            raised.exception.diagnostics["constraint_count_by_family"][
                "aggregate_three_groups_per_bay_proxy"
            ],
        )

    def test_graph_diagnostics_distinguish_full_and_active_domains(self) -> None:
        problem = two_area_problem()
        atoms, _limits = build_v7_row_atoms(problem)

        diagnostics = stage1_graph_diagnostics(atoms, {"G1": {"A"}})

        self.assertEqual(
            2, diagnostics["full_legal_domain"]["group_area_edge_count"]
        )
        self.assertEqual(
            1, diagnostics["stage1_restricted_domain"]["group_area_edge_count"]
        )
        self.assertAlmostEqual(0.5, diagnostics["active_edge_reduction_ratio"])
        self.assertIn("fraction_active_le_4", diagnostics)


if __name__ == "__main__":
    unittest.main()
