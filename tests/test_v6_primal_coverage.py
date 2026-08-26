from __future__ import annotations

import importlib.util
import unittest
from dataclasses import replace

from tests.test_v6_complete_mip import exact_config
from tests.test_v6_model_contract import make_bay, make_group, make_problem
from yard_planning.models import ProblemData
from yard_planning.v6_column_generation import (
    V6RootCgConfig,
    V6RootColumnGeneration,
)
from yard_planning.v6_complete_mip import V6CompleteMipSolver
from yard_planning.v6_primal_coverage import (
    V6AnalyticPeakConfig,
    V6CompactPeakConfig,
    V6CompactPeakUtilizationSolver,
    V6CompactPrimalCoverageSolver,
    V6PrimalCoverageConfig,
    V6PrimalCoveredPipeline,
    V6ProductionPipeline,
)
from yard_planning.v6_restricted_integer import V6RestrictedIntegerConfig
from yard_planning.v6_model import v6_export_group_key


GUROBI_AVAILABLE = importlib.util.find_spec("gurobipy") is not None


def root_config() -> V6RootCgConfig:
    return V6RootCgConfig(
        maximum_phase_one_iterations=50,
        maximum_business_iterations=100,
        pricing_time_limit=10.0,
        solver_threads=1,
        solver_seed=0,
        verbose=False,
    )


def coverage_config() -> V6PrimalCoverageConfig:
    return V6PrimalCoverageConfig(
        time_limit=10.0,
        mip_gap=0.0,
        solver_threads=1,
        solver_seed=0,
        verbose=False,
    )


def integer_config() -> V6RestrictedIntegerConfig:
    return V6RestrictedIntegerConfig(
        time_limit=10.0,
        mip_gap=0.0,
        solver_threads=1,
        solver_seed=0,
        verbose=False,
    )


@unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is unavailable")
class V6PrimalCoverageTests(unittest.TestCase):
    def test_compact_peak_matches_complete_minmax(self) -> None:
        cases = {
            "single": make_problem(
                [make_group("G1", port="P1")],
                [make_bay("01")],
            ),
            "groups_and_import": make_problem(
                [
                    make_group("G1", port="P1", demand=2),
                    make_group("G2", port="P2", demand=1),
                ],
                [make_bay("01"), make_bay("03"), make_bay("05")],
                import_boxes=1,
            ),
        }
        for name, problem in cases.items():
            with self.subTest(name=name):
                complete = V6CompleteMipSolver(problem, exact_config()).solve()
                compact = V6CompactPeakUtilizationSolver(
                    problem,
                    V6CompactPeakConfig(time_limit=10.0),
                ).solve()
                self.assertAlmostEqual(
                    complete.peak_policy.minimum_feasible_utilization,
                    compact.peak_policy.minimum_feasible_utilization,
                    places=8,
                )
                self.assertAlmostEqual(
                    compact.peak_policy.minimum_feasible_utilization,
                    compact.certificate["peak_utilization"]["maximum"],
                    places=8,
                )
                self.assertTrue(compact.diagnostics["proven_optimal"])
                self.assertEqual(
                    "compact_full_v6_row_atom_minmax_mip",
                    compact.peak_policy.as_dict()["minimum_source"],
                )
                self.assertFalse(
                    compact.diagnostics["complete_zone_enumeration_used"]
                )

    def test_compact_primal_reconstructs_all_objective_categories(self) -> None:
        group = make_group("G1", port="P1", demand=2)
        bay_a = make_bay("01", rows=("1",))
        bay_b = make_bay("01", rows=("1",))
        bay_b.area_no = "B"
        bay_b.bay_key = "B|01"
        problem = ProblemData(
            export_groups=[group],
            bays={bay_a.bay_key: bay_a, bay_b.bay_key: bay_b},
            area_functions={"A": {"OF"}, "B": {"OF"}},
            target_voyages=["V1"],
            export_voyages={"V1"},
            existing_group_bay_load={
                v6_export_group_key(group) + ("A", "A|01"): 1
            },
            berth_distances={("A", "Q1"): 1.0, ("B", "Q1"): 3.0},
            berth_by_voyage={"V1": "Q1"},
        )
        complete = V6CompleteMipSolver(problem, exact_config()).solve()

        coverage = V6CompactPrimalCoverageSolver(
            problem,
            complete.peak_policy,
            coverage_config(),
        ).solve()

        self.assertAlmostEqual(0.64375, coverage.objective, places=8)
        self.assertAlmostEqual(
            complete.certificate["objective"], coverage.objective, places=8
        )
        self.assertEqual(complete.certificate["raw"], coverage.certificate["raw"])

    def test_compact_primal_matches_complete_mip_without_zone_enumeration(self) -> None:
        problem = make_problem(
            [
                make_group("G1", port="P1", demand=2),
                make_group("G2", port="P2", demand=1),
            ],
            [make_bay("01"), make_bay("03"), make_bay("05")],
            import_boxes=1,
        )
        complete = V6CompleteMipSolver(problem, exact_config()).solve()

        coverage = V6CompactPrimalCoverageSolver(
            problem,
            complete.peak_policy,
            coverage_config(),
        ).solve()

        self.assertAlmostEqual(
            complete.certificate["objective"], coverage.objective, places=8
        )
        self.assertFalse(
            coverage.diagnostics["complete_zone_enumeration_used"]
        )
        self.assertTrue(coverage.certificate["validation"]["passed"])
        self.assertAlmostEqual(
            0.0,
            coverage.diagnostics["solver_evaluator_objective_difference"],
            places=8,
        )

    def test_coverage_repairs_integer_infeasible_root_pool(self) -> None:
        problem = make_problem(
            [
                make_group("G1", port="P1", demand=2),
                make_group("G2", port="P2", demand=1),
            ],
            [make_bay("01"), make_bay("03"), make_bay("05")],
            import_boxes=1,
        )
        complete = V6CompleteMipSolver(problem, exact_config()).solve()
        standalone_root = V6RootColumnGeneration(
            problem,
            complete.peak_policy,
            root_config(),
        ).solve()

        result = V6PrimalCoveredPipeline(
            problem,
            complete.peak_policy,
            root_config(),
            coverage_config(),
            integer_config(),
        ).solve()

        self.assertAlmostEqual(
            standalone_root.objective, result.root.objective, places=8
        )
        self.assertAlmostEqual(
            complete.certificate["objective"],
            result.integer.objective,
            places=8,
        )
        self.assertGreater(result.diagnostics["coverage_added_zone_count"], 0)
        self.assertTrue(result.diagnostics["root_bound_preserved"])
        self.assertTrue(result.integer.certificate["validation"]["passed"])
        self.assertTrue(
            result.integer.diagnostics["warm_start"]["applied"]
        )
        self.assertTrue(
            result.integer.diagnostics["warm_start_objective_preserved"]
        )

    def test_production_pipeline_uses_analytic_cap_and_no_zone_enumeration(self) -> None:
        problem = make_problem(
            [
                make_group("G1", port="P1", demand=2),
                make_group("G2", port="P2", demand=1),
            ],
            [make_bay("01"), make_bay("03"), make_bay("05")],
            import_boxes=1,
        )
        result = V6ProductionPipeline(
            problem,
            V6AnalyticPeakConfig(feasibility_time_limit=10.0),
            root_config(),
            coverage_config(),
            integer_config(),
        ).solve()
        complete = V6CompleteMipSolver(problem, exact_config()).solve(
            result.peak.peak_policy,
            result.peak.diagnostics,
        )

        self.assertFalse(result.diagnostics["complete_zone_enumeration_used"])
        self.assertEqual(
            "analytic_workload_lower_bound",
            result.peak.peak_policy.reference_role,
        )
        self.assertTrue(result.peak.diagnostics["feasibility_certified"])
        self.assertTrue(
            result.planning.coverage.diagnostics["warm_start"]["applied"]
        )
        self.assertAlmostEqual(
            complete.certificate["objective"],
            result.planning.integer.objective,
            places=8,
        )
        self.assertTrue(
            result.planning.integer.certificate["validation"]["passed"]
        )

    def test_coverage_improves_suboptimal_feasible_root_pool(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1")],
            [make_bay("01")],
        )
        complete = V6CompleteMipSolver(problem, exact_config()).solve()

        result = V6PrimalCoveredPipeline(
            problem,
            complete.peak_policy,
            root_config(),
            coverage_config(),
            integer_config(),
        ).solve()

        self.assertAlmostEqual(0.0, result.integer.objective, places=8)
        self.assertAlmostEqual(
            complete.certificate["objective"],
            result.integer.objective,
            places=8,
        )
        self.assertLess(result.integer.objective, 0.2125)
        self.assertAlmostEqual(
            0.175,
            result.diagnostics["relative_root_gap"],
            places=8,
        )

    def test_compact_primal_preserves_40ft_footprint(self) -> None:
        group = replace(
            make_group("G1", port="P1", demand=2),
            size="40",
        )
        bays = [
            make_bay(code, rows=("1", "2"))
            for code in ("01", "03", "05", "07")
        ]
        for bay in bays:
            bay.cap_by_size = {"40": 1}
            bay.row_cap_by_size = {"40": {"1": 1, "2": 1}}
        bays[0].large_bay_partner_key = bays[1].bay_key
        bays[2].large_bay_partner_key = bays[3].bay_key
        problem = make_problem([group], bays)
        complete = V6CompleteMipSolver(problem, exact_config()).solve()

        coverage = V6CompactPrimalCoverageSolver(
            problem,
            complete.peak_policy,
            coverage_config(),
        ).solve()

        physical_bays = {
            bay_key for zone in coverage.zones for bay_key in zone.physical_bay_keys
        }
        self.assertEqual(4, len(physical_bays))
        self.assertAlmostEqual(
            complete.certificate["objective"], coverage.objective, places=8
        )
        self.assertTrue(coverage.certificate["validation"]["passed"])


if __name__ == "__main__":
    unittest.main()
