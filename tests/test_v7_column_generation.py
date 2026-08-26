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
    audit_full_domain_pricing,
    patterns_from_atom_solution,
    solve_restricted_pattern_oracle,
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

    def test_closed_restricted_cg_equals_exhaustive_restricted_lp(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1"), make_group("G2", port="P2")],
            [make_bay("01"), make_bay("03")],
        )
        solver = V7CompleteMipSolver(problem, v7_exact_config())
        witness = solver.solve(feasibility_only=True)
        initial = patterns_from_atom_solution(
            problem, solver.atoms, witness.selected_atom_indices
        )
        restricted = {"G1": {"A"}, "G2": {"A"}}

        root = V7RootColumnGeneration(
            problem,
            witness.peak_policy,
            restricted,
            initial,
            V7RootCgConfig(root_time_limit=10.0),
            atoms=solver.atoms,
        ).solve()
        exhaustive, _patterns = solve_restricted_pattern_oracle(
            problem,
            witness.peak_policy,
            restricted,
            atoms=solver.atoms,
            integral=False,
        )

        self.assertTrue(root.diagnostics["restricted_root_closed"])
        self.assertTrue(root.diagnostics["anchor_atom_index_used"])
        self.assertTrue(
            root.diagnostics["existing_signatures_indexed_by_anchor"]
        )
        self.assertGreater(root.diagnostics["pricing_anchor_count"], 0)
        self.assertTrue(
            all(
                row["excluded_signature_mode"] == "dual_feasible_filter"
                and row["no_good_constraint_count"] == 0
                for row in root.diagnostics["pricing_iterations"]
            )
        )
        self.assertAlmostEqual(
            exhaustive.objective, root.restricted_lp_bound, places=8
        )

    def test_restricted_cg_never_generates_an_excluded_area(self) -> None:
        problem = two_area_problem()
        # B is globally preferred, but V7.1 production is deliberately frozen to A.
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
        restricted, _patterns = solve_restricted_pattern_oracle(
            problem,
            peak,
            {"G1": {"A"}},
            atoms=atoms,
            integral=False,
        )

        self.assertTrue(result.diagnostics["restricted_root_closed"])
        self.assertFalse(result.diagnostics["global_root_certified"])
        self.assertEqual({"A"}, set(result.restricted_areas_by_group["G1"]))
        self.assertEqual(0, result.diagnostics["active_area_expansion_count"])
        self.assertEqual(0, result.diagnostics["full_domain_certification_count"])
        self.assertTrue(
            all(
                problem.bays[pattern.anchor_bay_key].area_no == "A"
                for pattern in result.patterns
            )
        )
        self.assertAlmostEqual(restricted.objective, result.objective, places=8)

    def test_feasibility_proof_column_does_not_expand_pricing_domain(self) -> None:
        problem = two_area_problem()
        atoms, _limits = build_v7_row_atoms(problem)
        peak, _diagnostics = derive_v7_analytic_peak_policy(problem, atoms=atoms)
        search = enumerate_v7_bay_patterns(
            problem, atoms, "A|01", allowed_groups={"G1"}
        )
        proof = enumerate_v7_bay_patterns(
            problem, atoms, "B|01", allowed_groups={"G1"}
        )

        result = V7RootColumnGeneration(
            problem,
            peak,
            {"G1": {"A"}},
            search,
            V7RootCgConfig(root_time_limit=10.0),
            atoms=atoms,
            feasibility_proof_patterns=proof,
        ).solve()

        proof_signatures = {pattern.signature for pattern in proof}
        outside = {
            pattern.signature
            for pattern in result.patterns
            if problem.bays[pattern.anchor_bay_key].area_no == "B"
        }
        self.assertEqual(proof_signatures, outside)
        self.assertEqual(
            proof_signatures, set(result.feasibility_proof_pattern_signatures)
        )
        self.assertEqual(
            1,
            result.diagnostics[
                "outside_restricted_domain_proof_pattern_count"
            ],
        )
        self.assertFalse(result.diagnostics["proof_columns_expand_pricing_domain"])

    def test_global_audit_finds_excluded_negative_column_without_mutation(self) -> None:
        problem = two_area_problem()
        problem.berth_distances[("A", "Q1")] = 3.0
        problem.berth_distances[("B", "Q1")] = 1.0
        atoms, _limits = build_v7_row_atoms(problem)
        peak, _diagnostics = derive_v7_analytic_peak_policy(problem, atoms=atoms)
        initial = enumerate_v7_bay_patterns(
            problem, atoms, "A|01", allowed_groups={"G1"}
        )
        root = V7RootColumnGeneration(
            problem,
            peak,
            {"G1": {"A"}},
            initial,
            V7RootCgConfig(root_time_limit=10.0),
            atoms=atoms,
        ).solve()
        areas_before = dict(root.restricted_areas_by_group)
        signatures_before = tuple(pattern.signature for pattern in root.patterns)

        audit = audit_full_domain_pricing(
            problem,
            peak,
            root,
            V7RootCgConfig(root_time_limit=10.0),
            atoms=atoms,
        )

        self.assertTrue(audit.executed)
        self.assertFalse(audit.globally_root_certified)
        self.assertGreater(audit.negative_pattern_count, 0)
        self.assertIn(("G1", "B"), audit.affected_group_area_pairs)
        self.assertEqual(areas_before, dict(root.restricted_areas_by_group))
        self.assertEqual(
            signatures_before,
            tuple(pattern.signature for pattern in root.patterns),
        )

    def test_global_audit_can_certify_when_restricted_domain_is_global(self) -> None:
        problem = two_area_problem()
        atoms, _limits = build_v7_row_atoms(problem)
        peak, _diagnostics = derive_v7_analytic_peak_policy(problem, atoms=atoms)
        initial = enumerate_v7_bay_patterns(
            problem, atoms, "A|01", allowed_groups={"G1"}
        )
        root = V7RootColumnGeneration(
            problem,
            peak,
            {"G1": {"A", "B"}},
            initial,
            V7RootCgConfig(root_time_limit=10.0),
            atoms=atoms,
        ).solve()

        audit = audit_full_domain_pricing(
            problem,
            peak,
            root,
            V7RootCgConfig(root_time_limit=10.0),
            atoms=atoms,
        )

        self.assertTrue(audit.globally_root_certified)
        self.assertEqual(0, audit.negative_pattern_count)

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
