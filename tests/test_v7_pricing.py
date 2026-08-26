from __future__ import annotations

import importlib.util
import unittest

from tests.test_v6_model_contract import make_bay, make_group, make_problem
from yard_planning.v7_atoms import build_v7_row_atoms
from yard_planning.v7_bay_patterns import V7ExactBayPricing


GUROBI_AVAILABLE = importlib.util.find_spec("gurobipy") is not None


class V7PricingTests(unittest.TestCase):
    @unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is unavailable")
    def test_exact_pricing_mip_matches_exhaustive_pricing(self) -> None:
        problem = make_problem(
            [
                make_group("G1", port="P1"),
                make_group("G2", port="P2"),
                make_group("G3", port="P3"),
            ],
            [make_bay("01", rows=("1", "2", "3", "4"))],
        )
        atoms, _limits = build_v7_row_atoms(problem)
        pricing = V7ExactBayPricing(problem, atoms, columns_per_bay=5)

        def atom_cost(atom):
            return -0.13 * atom.capacity + 0.017 * atom.candidate_index

        def reduced_cost(pattern):
            return (
                0.37
                - 0.41 * len(pattern.active_groups)
                + sum(atom_cost(atoms[index]) for index in pattern.candidate_indices)
            )

        mip = pricing.price_bay_exact_mip(
            "A|01",
            atom_cost,
            lambda _anchor: 0.37,
            lambda _anchor, _size, _height, _group, _physical: -0.41,
            verify_reduced_cost=reduced_cost,
        )
        exhaustive = pricing.price_bay("A|01", reduced_cost, exhaustive=True)

        self.assertAlmostEqual(
            exhaustive.minimum_reduced_cost,
            mip.minimum_reduced_cost,
            places=10,
        )
        self.assertEqual(
            len(exhaustive.returned_patterns), len(mip.returned_patterns)
        )
        self.assertEqual(
            [round(reduced_cost(value), 12) for value in exhaustive.returned_patterns],
            [round(reduced_cost(value), 12) for value in mip.returned_patterns],
        )
        self.assertEqual(
            "exact_support_cardinality_row_assignment_mip",
            mip.diagnostics["method"],
        )

    def test_additive_row_dp_matches_exhaustive_pricing(self) -> None:
        problem = make_problem(
            [
                make_group("G1", port="P1"),
                make_group("G2", port="P2"),
                make_group("G3", port="P3"),
            ],
            [make_bay("01", rows=("1", "2", "3", "4"))],
        )
        atoms, _limits = build_v7_row_atoms(problem)
        pricing = V7ExactBayPricing(problem, atoms, columns_per_bay=5)

        def atom_cost(atom):
            return -0.13 * atom.capacity + 0.017 * atom.candidate_index

        def support_cost(_anchor, _size, _height, support, _physical):
            return 0.37 - 0.41 * len(support)

        def reduced_cost(pattern):
            return support_cost(
                pattern.anchor_bay_key,
                pattern.size_mode,
                pattern.height_mode,
                pattern.active_groups,
                pattern.physical_bays,
            ) + sum(atom_cost(atoms[index]) for index in pattern.candidate_indices)

        dynamic = pricing.price_bay_additive_dp(
            "A|01",
            atom_cost,
            support_cost,
            verify_reduced_cost=reduced_cost,
        )
        exhaustive = pricing.price_bay("A|01", reduced_cost, exhaustive=True)

        self.assertAlmostEqual(
            exhaustive.minimum_reduced_cost,
            dynamic.minimum_reduced_cost,
            places=10,
        )
        self.assertEqual(
            exhaustive.minimum_pattern.signature,
            dynamic.minimum_pattern.signature,
        )
        self.assertEqual(
            len(exhaustive.returned_patterns),
            len(dynamic.returned_patterns),
        )
        self.assertEqual(
            [
                round(reduced_cost(pattern), 12)
                for pattern in exhaustive.returned_patterns
            ],
            [
                round(reduced_cost(pattern), 12)
                for pattern in dynamic.returned_patterns
            ],
        )
        self.assertEqual(
            "joint_support_row_partition_dp",
            dynamic.diagnostics["method"],
        )

    def test_exact_support_pricing_matches_exhaustive_subset_pricing(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1"), make_group("G2", port="P2")],
            [make_bay("01", rows=("1", "2", "3"))],
        )
        atoms, _limits = build_v7_row_atoms(problem)
        pricing = V7ExactBayPricing(problem, atoms, columns_per_bay=4)

        def reduced_cost(pattern):
            return (
                0.37
                - 0.41 * len(pattern.active_groups)
                - 0.13 * sum(dict(pattern.group_capacities).values())
                + 0.017 * sum(pattern.candidate_indices)
            )

        exact = pricing.price_bay("A|01", reduced_cost)
        exhaustive = pricing.price_bay("A|01", reduced_cost, exhaustive=True)

        self.assertAlmostEqual(
            exhaustive.minimum_reduced_cost,
            exact.minimum_reduced_cost,
            places=10,
        )
        self.assertEqual(
            exhaustive.minimum_pattern.signature,
            exact.minimum_pattern.signature,
        )
        self.assertGreater(len(exact.returned_patterns), 0)
        self.assertTrue(exact.diagnostics["exact"])

    def test_active_group_filter_is_not_part_of_full_domain_pricing(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1"), make_group("G2", port="P2")],
            [make_bay("01", rows=("1", "2"))],
        )
        atoms, _limits = build_v7_row_atoms(problem)
        pricing = V7ExactBayPricing(problem, atoms)

        active = pricing.price_bay(
            "A|01", lambda pattern: -len(pattern.active_groups), allowed_groups={"G1"}
        )
        full = pricing.price_bay("A|01", lambda pattern: -len(pattern.active_groups))

        self.assertEqual(("G1",), active.minimum_pattern.active_groups)
        self.assertEqual(("G1", "G2"), full.minimum_pattern.active_groups)


if __name__ == "__main__":
    unittest.main()
