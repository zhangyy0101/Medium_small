from __future__ import annotations

import importlib.util
import unittest

from tests.test_v6_model_contract import make_bay, make_group, make_problem
from tests.test_v7_complete_mip import v7_exact_config
from tests.test_v7_stage1_area import two_area_problem
from yard_planning.v7_atoms import build_v7_row_atoms
from yard_planning.v7_bay_patterns import enumerate_v7_bay_patterns
from yard_planning.v7_column_generation import (
    V7GlobalPatternMaster,
    V7RootCgConfig,
    V7RootCgIncompleteError,
    V7RootColumnGeneration,
    patterns_from_atom_solution,
    solve_full_pattern_oracle,
)
from yard_planning.v7_complete_mip import V7CompleteMipSolver
from yard_planning.v7_model import derive_v7_analytic_peak_policy


GUROBI_AVAILABLE = importlib.util.find_spec("gurobipy") is not None


@unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is unavailable")
class V7ColumnGenerationTests(unittest.TestCase):
    def test_full_pattern_integer_oracle_equals_compact_mip(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1"), make_group("G2", port="P2")],
            [make_bay("01"), make_bay("03")],
            import_boxes=1,
        )
        compact = V7CompleteMipSolver(problem, v7_exact_config()).solve()
        atoms, _limits = build_v7_row_atoms(problem)

        pattern_solution, _patterns = solve_full_pattern_oracle(
            problem,
            compact.peak_policy,
            atoms=atoms,
            integral=True,
            time_limit=10.0,
        )

        self.assertAlmostEqual(compact.objective, pattern_solution.objective, places=8)

    def test_full_domain_certification_recovers_excluded_area(self) -> None:
        problem = two_area_problem()
        # Make B the globally preferred area while Stage 1 initially exposes A only.
        problem.berth_distances[("A", "Q1")] = 3.0
        problem.berth_distances[("B", "Q1")] = 1.0
        atoms, _limits = build_v7_row_atoms(problem)
        peak, _diagnostics = derive_v7_analytic_peak_policy(problem, atoms=atoms)
        initial = enumerate_v7_bay_patterns(
            problem, atoms, "A|01", allowed_groups={"G1"}
        )
        self.assertEqual(1, len(initial))

        result = V7RootColumnGeneration(
            problem,
            peak,
            {"G1": {"A"}},
            initial,
            V7RootCgConfig(root_time_limit=10.0),
            atoms=atoms,
        ).solve()
        full, _patterns = solve_full_pattern_oracle(
            problem, peak, atoms=atoms, integral=False
        )

        self.assertTrue(result.diagnostics["root_closed"])
        self.assertIn("B", result.active_areas_by_group["G1"])
        self.assertEqual(1, result.diagnostics["active_area_expansion_count"])
        self.assertGreaterEqual(
            result.diagnostics["full_domain_certification_count"], 1
        )
        self.assertAlmostEqual(full.objective, result.objective, places=8)

    def test_witness_patterns_make_initial_master_feasible(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1", demand=2)],
            [make_bay("01", rows=("1",)), make_bay("03", rows=("1",))],
        )
        solver = V7CompleteMipSolver(problem, v7_exact_config())
        witness = solver.solve(feasibility_only=True)
        patterns = patterns_from_atom_solution(
            problem, solver.atoms, witness.selected_atom_indices
        )
        master = V7GlobalPatternMaster(
            problem,
            solver.atoms,
            patterns,
            witness.peak_policy,
            integral=False,
        )
        try:
            solution = master.solve()
        finally:
            master.dispose()
        self.assertGreaterEqual(solution.objective, -1e-9)
        self.assertEqual(2, sum(solution.group_bay_flow.values()))

    def test_global_three_group_limit_catches_overlapping_40ft_anchors(self) -> None:
        groups = [
            type(make_group(f"G{i}", port=f"P{i}"))(
                **{**make_group(f"G{i}", port=f"P{i}").__dict__, "size": "40"}
            )
            for i in range(4)
        ]
        left = make_bay("01", rows=("1", "2"))
        shared = make_bay("03", rows=("1", "2", "3", "4"))
        right = make_bay("05", rows=("3", "4"))
        for bay in (left, shared, right):
            bay.cap_by_size = {"40": len(bay.row_physical_capacity)}
            bay.row_cap_by_size = {
                "40": {row: 1 for row in bay.row_physical_capacity}
            }
        left.large_bay_partner_key = shared.bay_key
        right.large_bay_partner_key = shared.bay_key
        problem = make_problem(groups, [left, shared, right])
        atoms, _limits = build_v7_row_atoms(problem)
        peak, _diagnostics = derive_v7_analytic_peak_policy(problem, atoms=atoms)
        patterns = [
            *enumerate_v7_bay_patterns(problem, atoms, left.bay_key),
            *enumerate_v7_bay_patterns(problem, atoms, right.bay_key),
        ]
        master = V7GlobalPatternMaster(
            problem, atoms, patterns, peak, integral=True, time_limit=5.0
        )
        try:
            with self.assertRaisesRegex(V7RootCgIncompleteError, "incomplete"):
                master.solve()
        finally:
            master.dispose()


if __name__ == "__main__":
    unittest.main()
