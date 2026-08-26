from __future__ import annotations

import importlib.util
import unittest
from dataclasses import replace

from tests.test_v6_model_contract import make_bay, make_group, make_problem
from yard_planning.v7_complete_mip import (
    V7CompleteMipConfig,
    V7CompleteMipIncompleteError,
    V7CompleteMipSolver,
)
from yard_planning.v7_model import derive_v7_analytic_peak_policy


GUROBI_AVAILABLE = importlib.util.find_spec("gurobipy") is not None


def v7_exact_config(**overrides) -> V7CompleteMipConfig:
    values = {
        "time_limit": 10.0,
        "mip_gap": 0.0,
        "solver_threads": 1,
        "solver_seed": 0,
        "verbose": False,
    }
    values.update(overrides)
    return V7CompleteMipConfig(**values)


@unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is unavailable")
class V7CompleteMipTests(unittest.TestCase):
    def test_analytic_peak_policy_has_zone_free_integer_witness(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1"), make_group("G2", port="P2")],
            [make_bay("01", rows=("1", "2")), make_bay("03", rows=("1", "2"))],
            import_boxes=1,
        )
        solver = V7CompleteMipSolver(problem, v7_exact_config())
        policy, diagnostics = derive_v7_analytic_peak_policy(
            problem, atoms=solver.atoms
        )

        self.assertAlmostEqual(
            policy.epsilon_cap,
            policy.reference_utilization
            + policy.headroom_fraction * (1.0 - policy.reference_utilization),
        )
        self.assertEqual(len(solver.atoms), diagnostics["legal_atom_count"])
        witness = solver.solve(policy, feasibility_only=True)
        families = witness.diagnostics["constraint_count_by_family"]
        self.assertIn("export_height_state_choice", families)
        self.assertIn("max_three_groups_per_physical_bay", families)
        self.assertLessEqual(
            witness.certificate["peak_utilization"]["maximum"],
            policy.epsilon_cap + 1e-9,
        )
        self.assertTrue(witness.certificate["validation"]["passed"])

    def test_solver_objective_matches_independent_evaluator(self) -> None:
        group = make_group("G1", port="P1", demand=2)
        problem = make_problem(
            [group],
            [make_bay("01", rows=("1",)), make_bay("03", rows=("1",))],
        )

        result = V7CompleteMipSolver(problem, v7_exact_config()).solve()

        self.assertTrue(result.certificate["validation"]["passed"])
        self.assertAlmostEqual(
            0.0,
            result.diagnostics["solver_evaluator_objective_difference"],
            places=8,
        )
        self.assertFalse(result.diagnostics["zone_variables_used"])

    def test_complete_mip_allows_three_but_rejects_four_groups_in_one_bay(self) -> None:
        three = [make_group(f"G{i}", port=f"P{i}") for i in range(3)]
        three_problem = make_problem(
            three, [make_bay("01", rows=("1", "2", "3", "4"))]
        )
        result = V7CompleteMipSolver(three_problem, v7_exact_config()).solve()
        self.assertEqual(3, result.certificate["groups_per_physical_bay"]["A|01"])

        four = [make_group(f"H{i}", port=f"Q{i}") for i in range(4)]
        four_problem = make_problem(
            four, [make_bay("01", rows=("1", "2", "3", "4"))]
        )
        with self.assertRaisesRegex(V7CompleteMipIncompleteError, "no incumbent"):
            V7CompleteMipSolver(four_problem, v7_exact_config()).solve()

    def test_height_no_mix_and_import_export_exclusivity_are_in_model(self) -> None:
        first = replace(make_group("G1", port="P1"), height="H1")
        second = replace(make_group("G2", port="P2"), height="H2")
        impossible = make_problem(
            [first, second], [make_bay("01", rows=("1", "2"))]
        )
        with self.assertRaisesRegex(V7CompleteMipIncompleteError, "no incumbent"):
            V7CompleteMipSolver(impossible, v7_exact_config()).solve()

        feasible = make_problem(
            [make_group("G1", port="P1")],
            [make_bay("01"), make_bay("03")],
            import_boxes=1,
        )
        result = V7CompleteMipSolver(feasible, v7_exact_config()).solve()
        export_bays = {
            bay_key for (_group, bay_key), quantity in result.group_bay_flow.items() if quantity
        }
        import_bays = {
            bay_key
            for (_flow, _size, bay_key), quantity in result.import_reservation.items()
            if quantity
        }
        self.assertTrue(export_bays.isdisjoint(import_bays))

    def test_40ft_assignment_reserves_both_footprint_members(self) -> None:
        group = replace(make_group("G1", port="P1", demand=2), size="40")
        bays = [make_bay(code, rows=("1", "2")) for code in ("01", "03")]
        for bay in bays:
            bay.cap_by_size = {"40": 2}
            bay.row_cap_by_size = {"40": {"1": 1, "2": 1}}
        bays[0].large_bay_partner_key = bays[1].bay_key

        result = V7CompleteMipSolver(
            make_problem([group], bays), v7_exact_config()
        ).solve()

        self.assertEqual({("G1", "A|01"): 2}, dict(result.group_bay_flow))
        self.assertEqual(4, result.certificate["validation"]["physical_rows_checked"])


if __name__ == "__main__":
    unittest.main()
