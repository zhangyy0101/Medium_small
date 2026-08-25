from __future__ import annotations

import unittest

from preexperiment.complete_mip_baseline import (
    aggregate_complete_mip_comparisons,
    compare_complete_mip_result,
)


class CompleteMipBaselineTests(unittest.TestCase):
    def test_lower_current_incumbent_wins_and_uses_direct_bound(self) -> None:
        comparison = compare_complete_mip_result(
            {
                "upper_bound": 0.8,
                "relative_gap": 0.25,
                "total_seconds": 60.0,
            },
            {
                "has_solution": True,
                "objective": 1.0,
                "bound": 0.7,
                "relative_gap": 0.3,
                "total_seconds": 60.0,
            },
        )

        self.assertEqual("current_algorithm", comparison["winner"])
        self.assertAlmostEqual(
            0.2,
            comparison["current_relative_incumbent_improvement"],
        )
        self.assertAlmostEqual(
            0.125,
            comparison["current_gap_using_complete_mip_bound"],
        )

    def test_invalid_direct_bound_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            compare_complete_mip_result(
                {
                    "upper_bound": 0.8,
                    "relative_gap": 0.25,
                    "total_seconds": 60.0,
                },
                {
                    "has_solution": True,
                    "objective": 1.0,
                    "bound": 0.9,
                    "relative_gap": 0.1,
                    "total_seconds": 60.0,
                },
            )

    def test_aggregate_counts_results(self) -> None:
        aggregate = aggregate_complete_mip_comparisons(
            [
                {
                    "passed": True,
                    "comparison": {
                        "winner": "current_algorithm",
                        "current_relative_incumbent_improvement": 0.2,
                    },
                },
                {
                    "passed": True,
                    "comparison": {
                        "winner": "tie",
                        "current_relative_incumbent_improvement": 0.0,
                    },
                },
            ]
        )

        self.assertEqual(1, aggregate["current_algorithm_win_count"])
        self.assertEqual(1, aggregate["tie_count"])
        self.assertAlmostEqual(
            0.1,
            aggregate["mean_current_relative_incumbent_improvement"],
        )


if __name__ == "__main__":
    unittest.main()
