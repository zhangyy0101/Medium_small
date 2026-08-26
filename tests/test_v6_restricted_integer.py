from __future__ import annotations

import importlib.util
import unittest
from dataclasses import replace

from tests.test_v6_complete_mip import exact_config
from tests.test_v6_model_contract import make_bay, make_group, make_problem
from yard_planning.row_aware_zones import build_complete_v6_zone_universe
from yard_planning.v6_column_generation import V6RootCgConfig
from yard_planning.v6_complete_mip import V6CompleteMipSolver
from yard_planning.v6_restricted_integer import (
    V6RestrictedIntegerConfig,
    V6RestrictedIntegerSolver,
    V6RootCgIntegerPipeline,
)


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


def integer_config() -> V6RestrictedIntegerConfig:
    return V6RestrictedIntegerConfig(
        time_limit=10.0,
        mip_gap=0.0,
        solver_threads=1,
        solver_seed=0,
        verbose=False,
    )


@unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is unavailable")
class V6RestrictedIntegerTests(unittest.TestCase):
    def test_full_zone_integer_master_matches_complete_mip(self) -> None:
        problem = make_problem(
            [
                make_group("G1", port="P1", demand=2),
                make_group("G2", port="P2", demand=1),
            ],
            [make_bay("01"), make_bay("03"), make_bay("05")],
            import_boxes=1,
        )
        complete = V6CompleteMipSolver(problem, exact_config()).solve()
        full_zones = build_complete_v6_zone_universe(problem)

        result = V6RestrictedIntegerSolver(
            problem,
            full_zones,
            complete.peak_policy,
            integer_config(),
        ).solve()

        self.assertAlmostEqual(
            complete.certificate["objective"], result.objective, places=8
        )
        self.assertTrue(result.certificate["validation"]["passed"])
        self.assertTrue(
            result.diagnostics["proven_optimal_over_restricted_pool"]
        )
        self.assertAlmostEqual(
            0.0,
            result.diagnostics["solver_evaluator_objective_difference"],
            places=8,
        )

    def test_global_group_bay_flow_is_recovered_as_zone_bay_flow(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1", demand=2)],
            [make_bay("01"), make_bay("03")],
        )
        complete = V6CompleteMipSolver(problem, exact_config()).solve()
        result = V6RestrictedIntegerSolver(
            problem,
            build_complete_v6_zone_universe(problem),
            complete.peak_policy,
            integer_config(),
        ).solve()

        self.assertEqual(
            2,
            sum(result.zone_bay_flow.values()),
        )
        self.assertTrue(
            all(zone_id in result.selected_zone_ids for zone_id, _bay in result.zone_bay_flow)
        )
        self.assertTrue(result.certificate["validation"]["passed"])

    def test_full_zone_integer_equivalence_preserves_40ft_footprint(self) -> None:
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

        result = V6RestrictedIntegerSolver(
            problem,
            build_complete_v6_zone_universe(problem),
            complete.peak_policy,
            integer_config(),
        ).solve()

        self.assertAlmostEqual(
            complete.certificate["objective"], result.objective, places=8
        )
        selected_physical_bays = {
            bay_key
            for zone in result.zones
            if zone.zone_id in result.selected_zone_ids
            for bay_key in zone.physical_bay_keys
        }
        self.assertEqual(4, len(selected_physical_bays))
        self.assertTrue(result.certificate["validation"]["passed"])

    def test_batched_exact_root_pool_integer_incumbent_is_valid(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1")],
            [make_bay("01")],
        )
        complete = V6CompleteMipSolver(problem, exact_config()).solve()

        result = V6RootCgIntegerPipeline(
            problem,
            complete.peak_policy,
            root_config(),
            integer_config(),
        ).solve()

        self.assertTrue(result.integer.certificate["validation"]["passed"])
        self.assertLessEqual(result.root.objective, result.integer.objective)
        self.assertGreaterEqual(
            result.integer.objective + 1e-8,
            complete.certificate["objective"],
        )
        self.assertGreaterEqual(len(result.root.zones), 1)

    def test_diversified_root_pool_can_recover_coordinated_integer_solution(self) -> None:
        problem = make_problem(
            [
                make_group("G1", port="P1", demand=2),
                make_group("G2", port="P2", demand=1),
            ],
            [make_bay("01"), make_bay("03"), make_bay("05")],
            import_boxes=1,
        )
        complete = V6CompleteMipSolver(problem, exact_config()).solve()

        result = V6RootCgIntegerPipeline(
            problem,
            complete.peak_policy,
            root_config(),
            integer_config(),
        ).solve()

        self.assertTrue(result.integer.certificate["validation"]["passed"])
        self.assertLessEqual(result.root.objective, result.integer.objective)
        self.assertGreaterEqual(
            result.integer.objective + 1e-8,
            complete.certificate["objective"],
        )


if __name__ == "__main__":
    unittest.main()
