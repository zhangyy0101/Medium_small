from __future__ import annotations

import unittest

from preexperiment.comparison import aggregate_comparisons, compare_method_summaries


class PreexperimentComparisonTests(unittest.TestCase):
    def test_positive_improvement_means_full_method_wins(self) -> None:
        paired = compare_method_summaries(
            {
                "upper_bound": 8.0,
                "lower_bound": 6.0,
                "relative_gap": 0.25,
                "total_seconds": 10.0,
            },
            {
                "upper_bound": 10.0,
                "lower_bound": 6.0,
                "relative_gap": 0.40,
                "total_seconds": 10.0,
            },
        )

        self.assertEqual("full_fix_optimize", paired["winner"])
        self.assertAlmostEqual(0.2, paired["relative_improvement_from_fix_optimize"])
        self.assertAlmostEqual(0.0, paired["root_lower_bound_difference"])

    def test_aggregate_counts_wins_ties_and_correctness(self) -> None:
        cases = [
            {
                "full_method": {"valid": True},
                "no_fix_optimize": {"valid": True},
                "paired_comparison": {
                    "winner": "full_fix_optimize",
                    "relative_improvement_from_fix_optimize": 0.10,
                },
                "correctness": {"executed": True, "passed": True},
            },
            {
                "full_method": {"valid": True},
                "no_fix_optimize": {"valid": True},
                "paired_comparison": {
                    "winner": "tie",
                    "relative_improvement_from_fix_optimize": 0.0,
                },
                "correctness": {"executed": False},
            },
        ]

        aggregate = aggregate_comparisons(cases)

        self.assertEqual(1, aggregate["full_fix_optimize_win_count"])
        self.assertEqual(1, aggregate["tie_count"])
        self.assertEqual(1, aggregate["correctness_passed_count"])
        self.assertAlmostEqual(
            0.05,
            aggregate["mean_relative_improvement_from_fix_optimize"],
        )


if __name__ == "__main__":
    unittest.main()
