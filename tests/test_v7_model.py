from __future__ import annotations

import unittest
from dataclasses import replace

from tests.test_v6_model_contract import make_bay, make_group, make_problem
from yard_planning.v6_model import v6_export_group_key
from yard_planning.v7_atoms import build_v7_row_atoms
from yard_planning.v7_model import (
    V7ModelEvaluator,
    V7ObjectiveConfig,
    derive_v7_analytic_peak_policy,
    v7_model_contract,
)


class V7ModelTests(unittest.TestCase):
    def test_contract_removes_zone_objective_and_declares_three_categories(self) -> None:
        contract = v7_model_contract()
        objective = contract["objective"]

        self.assertEqual(
            "hierarchical_group_area_bay_pattern_v7",
            contract["model_schema_version"],
        )
        self.assertIn("zone", contract["removed_formal_objects"])
        self.assertEqual(
            {
                "spatial_consolidation",
                "existing_group_proximity",
                "berth_transport",
            },
            set(objective["category_weights"]),
        )
        self.assertEqual(
            "provisional_development_baseline",
            objective["calibration_status"],
        )
        self.assertNotIn("unused_capacity", objective["primitive_weights"])

    def test_three_groups_are_feasible_but_four_are_rejected(self) -> None:
        groups = [make_group(f"G{i}", port=f"P{i}") for i in range(1, 5)]
        three_problem = make_problem(
            groups[:3], [make_bay("01", rows=("1", "2", "3", "4"))]
        )
        three_atoms, _limits = build_v7_row_atoms(three_problem)
        three_evaluator = V7ModelEvaluator(three_problem, three_atoms)
        three_policy, _diagnostics = derive_v7_analytic_peak_policy(
            three_problem, atoms=three_atoms
        )
        three_chosen = []
        for index, group in enumerate(groups[:3]):
            three_chosen.append(
                next(
                    atom.candidate_index
                    for atom in three_atoms
                    if atom.group_id == group.group_id and atom.row_no == str(index + 1)
                )
            )
        three_flow = {(group.group_id, "A|01"): 1 for group in groups[:3]}
        certificate = three_evaluator.evaluate(
            three_chosen, three_flow, {}, three_policy
        )
        self.assertTrue(certificate["validation"]["passed"])
        self.assertEqual(3, certificate["groups_per_physical_bay"]["A|01"])

        problem = make_problem(
            groups, [make_bay("01", rows=("1", "2", "3", "4"))]
        )
        atoms, _limits = build_v7_row_atoms(problem)
        evaluator = V7ModelEvaluator(problem, atoms)
        policy, _diagnostics = derive_v7_analytic_peak_policy(problem, atoms=atoms)
        chosen = [
            next(
                atom.candidate_index
                for atom in atoms
                if atom.group_id == group.group_id and atom.row_no == str(index + 1)
            )
            for index, group in enumerate(groups)
        ]
        four_flow = {(group.group_id, "A|01"): 1 for group in groups}
        with self.assertRaisesRegex(ValueError, "more than three"):
            evaluator.evaluate(chosen, four_flow, {}, policy)

    def test_new_height_no_mix_is_independent_of_legacy_mixed_state(self) -> None:
        bay = make_bay("01", rows=("1", "2"))
        bay.existing_heights = {"H1", "H2"}
        first = replace(make_group("G1", port="P1"), height="H1")
        second = replace(make_group("G2", port="P2"), height="H2")
        problem = make_problem([first, second], [bay])
        atoms, _limits = build_v7_row_atoms(problem)
        evaluator = V7ModelEvaluator(problem, atoms)
        policy, _diagnostics = derive_v7_analytic_peak_policy(problem, atoms=atoms)
        selected = [
            next(
                atom.candidate_index
                for atom in atoms
                if atom.group_id == "G1" and atom.row_no == "1"
            ),
            next(
                atom.candidate_index
                for atom in atoms
                if atom.group_id == "G2" and atom.row_no == "2"
            ),
        ]

        with self.assertRaisesRegex(ValueError, "height mixing"):
            evaluator.evaluate(
                selected,
                {("G1", "A|01"): 1, ("G2", "A|01"): 1},
                {},
                policy,
            )

    def test_objective_reconstructs_area_bay_span_existing_and_berth_terms(self) -> None:
        group = make_group("G1", port="P1", demand=2)
        bay_a1 = make_bay("01", rows=("1",))
        bay_a2 = make_bay("05", rows=("1",))
        bay_b = make_bay("01", rows=("1",))
        bay_b.area_no = "B"
        bay_b.bay_key = "B|01"
        problem = make_problem([group], [bay_a1, bay_a2, bay_b])
        problem.area_functions["B"] = {"OF"}
        problem.berth_distances[("B", "Q1")] = 3.0
        atoms, _limits = build_v7_row_atoms(problem)
        evaluator = V7ModelEvaluator(problem, atoms, V7ObjectiveConfig())
        policy, _diagnostics = derive_v7_analytic_peak_policy(problem, atoms=atoms)
        selected = [
            next(
                atom.candidate_index
                for atom in atoms
                if atom.anchor_bay_key == bay_a1.bay_key
            ),
            next(
                atom.candidate_index
                for atom in atoms
                if atom.anchor_bay_key == bay_a2.bay_key
            ),
        ]

        certificate = evaluator.evaluate(
            selected,
            {("G1", bay_a1.bay_key): 1, ("G1", bay_a2.bay_key): 1},
            {},
            policy,
        )

        self.assertEqual(0.0, certificate["raw"]["extra_group_areas"])
        self.assertEqual(1.0, certificate["raw"]["extra_group_bays"])
        self.assertEqual(
            1.0, certificate["raw"]["normalized_within_area_bay_span_sum"]
        )
        self.assertAlmostEqual(
            certificate["objective"],
            sum(certificate["weighted_primitives"].values()),
            places=10,
        )

    def test_area_existing_and_berth_raw_terms_are_separately_observable(self) -> None:
        group = make_group("G1", port="P1", demand=2)
        bay_a = make_bay("01", rows=("1",))
        bay_b = make_bay("01", rows=("1",))
        bay_b.area_no = "B"
        bay_b.bay_key = "B|01"
        problem = make_problem([group], [bay_a, bay_b])
        problem.area_functions["B"] = {"OF"}
        problem.berth_distances[("A", "Q1")] = 1.0
        problem.berth_distances[("B", "Q1")] = 3.0
        problem.existing_group_bay_load = {
            v6_export_group_key(group) + ("A", "A|01"): 1
        }
        atoms, _limits = build_v7_row_atoms(problem)
        evaluator = V7ModelEvaluator(problem, atoms)
        policy, _diagnostics = derive_v7_analytic_peak_policy(problem, atoms=atoms)
        selected = tuple(atom.candidate_index for atom in atoms)

        certificate = evaluator.evaluate(
            selected,
            {("G1", "A|01"): 1, ("G1", "B|01"): 1},
            {},
            policy,
        )

        self.assertEqual(1.0, certificate["raw"]["extra_group_areas"])
        self.assertEqual(1.0, certificate["raw"]["extra_group_bays"])
        self.assertEqual(
            1.0, certificate["raw"]["existing_group_normalized_distance_sum"]
        )
        self.assertEqual(1.0, certificate["raw"]["berth_normalized_distance_sum"])


if __name__ == "__main__":
    unittest.main()
