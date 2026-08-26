from __future__ import annotations

import importlib.util
import unittest

from tests.test_v6_complete_mip import exact_config
from tests.test_v6_model_contract import make_bay, make_group, make_problem
from yard_planning.row_aware_zones import (
    build_complete_v6_zone_universe,
    build_v6_row_aware_bay_atoms,
)
from yard_planning.v6_column_generation import (
    V6ExactZonePricing,
    V6ProjectedRestrictedMaster,
    V6RootCgConfig,
    V6RootColumnGeneration,
)
from yard_planning.v6_complete_mip import V6CompleteMipSolver
from yard_planning.v6_model import V6ModelEvaluator


GUROBI_AVAILABLE = importlib.util.find_spec("gurobipy") is not None


def cg_config() -> V6RootCgConfig:
    return V6RootCgConfig(
        maximum_phase_one_iterations=50,
        maximum_business_iterations=100,
        pricing_time_limit=10.0,
        solver_threads=1,
        solver_seed=0,
        verbose=False,
    )


@unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is unavailable")
class V6ColumnGenerationTests(unittest.TestCase):
    def _context(self, problem):
        complete = V6CompleteMipSolver(problem, exact_config()).solve()
        atoms, limits = build_v6_row_aware_bay_atoms(problem)
        candidate_bays: dict[str, set[str]] = {}
        for atom in atoms:
            candidate_bays.setdefault(atom.group_id, set()).add(
                atom.anchor_bay_key
            )
        evaluator = V6ModelEvaluator(
            problem,
            (),
            candidate_bays_by_group=candidate_bays,
        )
        return complete.peak_policy, atoms, limits, evaluator

    def test_exact_pricing_matches_exhaustive_oracle(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1", demand=2)],
            [make_bay("01"), make_bay("03")],
        )
        peak, atoms, limits, evaluator = self._context(problem)
        full_zones = build_complete_v6_zone_universe(problem)
        master = V6ProjectedRestrictedMaster(
            problem,
            atoms,
            limits,
            evaluator,
            peak,
            phase="phase_one",
        )
        pricing = V6ExactZonePricing(
            problem, atoms, limits, evaluator, cg_config()
        )
        try:
            solution = master.solve()
            exact = pricing.price_group(
                "G1", solution.duals, phase="phase_one"
            )
            exhaustive = pricing.exhaustive_price_group(
                "G1", full_zones, solution.duals, phase="phase_one"
            )
            self.assertIsNotNone(exact)
            self.assertIsNotNone(exhaustive)
            self.assertAlmostEqual(
                exhaustive.reduced_cost,
                exact.reduced_cost,
                places=8,
            )
            self.assertAlmostEqual(
                exact.reduced_cost,
                exact.solver_objective,
                places=8,
            )
            seed_zone = exact.zone
        finally:
            master.dispose()

        business_master = V6ProjectedRestrictedMaster(
            problem,
            atoms,
            limits,
            evaluator,
            peak,
            phase="business",
            initial_zones=(seed_zone,),
        )
        try:
            solution = business_master.solve()
            excluded = {seed_zone.candidate_indices}
            exact = pricing.price_group(
                "G1",
                solution.duals,
                phase="business",
                excluded_signatures=excluded,
            )
            exhaustive = pricing.exhaustive_price_group(
                "G1",
                full_zones,
                solution.duals,
                phase="business",
                excluded_signatures=excluded,
            )
            self.assertIsNotNone(exact)
            self.assertIsNotNone(exhaustive)
            self.assertAlmostEqual(
                exhaustive.reduced_cost,
                exact.reduced_cost,
                places=8,
            )
        finally:
            business_master.dispose()

    def test_exact_pricing_matches_oracle_for_synthetic_duals(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1", demand=2)],
            [make_bay("01"), make_bay("03"), make_bay("05")],
        )
        peak, atoms, limits, evaluator = self._context(problem)
        full_zones = build_complete_v6_zone_universe(problem)
        master = V6ProjectedRestrictedMaster(
            problem,
            atoms,
            limits,
            evaluator,
            peak,
            phase="phase_one",
        )
        pricing = V6ExactZonePricing(
            problem, atoms, limits, evaluator, cg_config()
        )
        try:
            shape = master.solve().duals
            synthetic = {
                family: {
                    key: ((index % 9) - 4) * 0.071
                    for index, key in enumerate(
                        sorted(values, key=repr), start=1
                    )
                }
                for family, values in shape.items()
            }
            exact = pricing.price_group(
                "G1", synthetic, phase="business"
            )
            exhaustive = pricing.exhaustive_price_group(
                "G1", full_zones, synthetic, phase="business"
            )
            self.assertIsNotNone(exact)
            self.assertIsNotNone(exhaustive)
            self.assertAlmostEqual(
                exhaustive.reduced_cost,
                exact.reduced_cost,
                places=8,
            )
        finally:
            master.dispose()

    def test_batched_dp_pricing_is_exact_valid_and_distinct(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1", demand=2)],
            [make_bay("01"), make_bay("03"), make_bay("05")],
        )
        peak, atoms, limits, evaluator = self._context(problem)
        full_zones = build_complete_v6_zone_universe(problem)
        full_signatures = {
            zone.candidate_indices
            for zone in full_zones
            if zone.group_id == "G1"
        }
        master = V6ProjectedRestrictedMaster(
            problem,
            atoms,
            limits,
            evaluator,
            peak,
            phase="phase_one",
        )
        pricing = V6ExactZonePricing(
            problem, atoms, limits, evaluator, cg_config()
        )
        try:
            solution = master.solve()
            batched = pricing.price_group_columns(
                "G1",
                solution.duals,
                phase="phase_one",
                maximum_columns=8,
                patterns_per_interval=3,
            )
            exhaustive = pricing.exhaustive_price_group(
                "G1", full_zones, solution.duals, phase="phase_one"
            )
            self.assertIsNotNone(exhaustive)
            self.assertGreater(len(batched), 1)
            self.assertAlmostEqual(
                exhaustive.reduced_cost,
                min(item.reduced_cost for item in batched),
                places=8,
            )
            signatures = [item.zone.candidate_indices for item in batched]
            self.assertEqual(len(signatures), len(set(signatures)))
            self.assertTrue(set(signatures) <= full_signatures)
            for item in batched:
                self.assertLess(item.reduced_cost, -1e-8)
                self.assertAlmostEqual(
                    item.reduced_cost, item.solver_objective, places=8
                )
        finally:
            master.dispose()
            pricing.dispose()

    def test_root_cg_matches_fully_enumerated_projected_lp(self) -> None:
        problem = make_problem(
            [
                make_group("G1", port="P1", demand=2),
                make_group("G2", port="P2", demand=1),
            ],
            [make_bay("01"), make_bay("03"), make_bay("05")],
            import_boxes=1,
        )
        peak, atoms, limits, evaluator = self._context(problem)
        result = V6RootColumnGeneration(
            problem, peak, cg_config()
        ).solve()
        full_zones = build_complete_v6_zone_universe(problem)
        full_master = V6ProjectedRestrictedMaster(
            problem,
            atoms,
            limits,
            evaluator,
            peak,
            phase="business",
            initial_zones=full_zones,
        )
        try:
            full_solution = full_master.solve()
            self.assertAlmostEqual(
                full_solution.objective,
                result.objective,
                places=8,
            )
            self.assertEqual(
                0.0,
                result.diagnostics["phase_one"]["final_artificial_deficit"],
            )
            self.assertTrue(
                result.diagnostics["business"]["closed_by_exact_pricing"]
            )
        finally:
            full_master.dispose()

    def test_feasible_seed_skips_phase_one_and_preserves_root_bound(self) -> None:
        problem = make_problem(
            [
                make_group("G1", port="P1", demand=2),
                make_group("G2", port="P2", demand=1),
            ],
            [make_bay("01"), make_bay("03"), make_bay("05")],
            import_boxes=1,
        )
        peak, _atoms, _limits, _evaluator = self._context(problem)
        unseeded = V6RootColumnGeneration(
            problem, peak, cg_config()
        ).solve()
        seeded = V6RootColumnGeneration(
            problem, peak, cg_config()
        ).solve(initial_zones=unseeded.zones)

        self.assertAlmostEqual(unseeded.objective, seeded.objective, places=8)
        self.assertTrue(
            seeded.diagnostics["phase_one"]["skipped_by_feasible_seed"]
        )
        self.assertEqual(0, seeded.diagnostics["phase_one"]["iterations"])
        self.assertEqual(
            len(unseeded.zones),
            seeded.diagnostics["phase_one"]["seed_zone_count"],
        )

    def test_phase_one_starts_without_zones_and_constructs_feasibility(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1", demand=2)],
            [make_bay("01"), make_bay("03")],
        )
        peak, _atoms, _limits, _evaluator = self._context(problem)

        result = V6RootColumnGeneration(
            problem, peak, cg_config()
        ).solve()

        trace = result.diagnostics["phase_one"]["trace"]
        self.assertGreater(trace[0]["artificial_deficit"], 0.0)
        self.assertEqual(0, trace[0]["column_count"])
        self.assertEqual(
            0.0,
            result.diagnostics["phase_one"]["final_artificial_deficit"],
        )
        self.assertGreater(len(result.zones), 0)

    def test_pricing_can_change_selected_row_between_adjacent_bays(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1", demand=2)],
            [make_bay("01"), make_bay("03")],
        )
        peak, atoms, limits, evaluator = self._context(problem)
        master = V6ProjectedRestrictedMaster(
            problem,
            atoms,
            limits,
            evaluator,
            peak,
            phase="phase_one",
        )
        pricing = V6ExactZonePricing(
            problem, atoms, limits, evaluator, cg_config()
        )
        full_zones = build_complete_v6_zone_universe(problem)
        cross_row_signatures = {
            zone.candidate_indices
            for zone in full_zones
            if len(zone.anchor_bay_keys) == 2
            and len({rows[0] for _bay, rows in zone.rows_by_anchor_bay}) == 2
        }
        self.assertTrue(cross_row_signatures)
        try:
            solution = master.solve()
            excluded = {
                zone.candidate_indices
                for zone in full_zones
                if zone.candidate_indices not in cross_row_signatures
            }
            priced = pricing.price_group(
                "G1",
                solution.duals,
                phase="phase_one",
                excluded_signatures=excluded,
            )
            self.assertIsNotNone(priced)
            self.assertIn(priced.zone.candidate_indices, cross_row_signatures)
        finally:
            master.dispose()


if __name__ == "__main__":
    unittest.main()
