from __future__ import annotations

import importlib.util
import unittest
from collections import defaultdict

from yard_planning.contiguous_zone_generation import (
    ContiguousZoneConfig,
    ContiguousZoneGenerationPlanner,
)
from yard_planning.models import Bay, ExportGroup, ProblemData
from yard_planning.planner import ColumnGenerationConfig


def make_bay(area: str, bay_no: str, row_capacity: int = 2) -> Bay:
    return Bay(
        area_no=area,
        bay_no=bay_no,
        bay_key=f"{area}|{bay_no}",
        bay_order=int(bay_no),
        cap_by_size={"20": row_capacity},
        physical_capacity=row_capacity,
        row_cap_by_size={"20": {"1": row_capacity}},
        row_physical_capacity={"1": row_capacity},
    )


def make_small_problem() -> ProblemData:
    groups = [
        ExportGroup(
            group_id="G1",
            voyage_id="V1",
            status="OF",
            port="P1",
            size="20",
            height="96",
            demand=3,
        ),
        ExportGroup(
            group_id="G2",
            voyage_id="V1",
            status="OF",
            port="P2",
            size="20",
            height="96",
            demand=2,
        ),
    ]
    bays = {
        bay.bay_key: bay
        for bay in (
            make_bay("A", "01"),
            make_bay("A", "03"),
            make_bay("B", "01"),
            make_bay("B", "03"),
        )
    }
    return ProblemData(
        export_groups=groups,
        bays=bays,
        area_guidance_target={
            ("V1", "OF", "A", "20"): 3,
            ("V1", "OF", "B", "20"): 2,
        },
        area_functions={"A": {"OF"}, "B": {"OF"}},
        target_voyages=["V1"],
        export_voyages={"V1"},
        berth_distances={("A", "Q1"): 1.0, ("B", "Q1"): 2.0},
        berth_by_voyage={"V1": "Q1"},
    )


class ContiguousZoneGenerationTests(unittest.TestCase):
    def test_configuration_rejects_invalid_adaptive_neighborhood(self) -> None:
        with self.assertRaises(ValueError):
            ContiguousZoneConfig(fix_optimize_objective_mass=0.0).validate()
        with self.assertRaises(ValueError):
            ContiguousZoneConfig(fix_optimize_zone_fraction=1.1).validate()
        with self.assertRaises(ValueError):
            ContiguousZoneConfig(
                peak_utilization_headroom_fraction=1.1
            ).validate()

    def test_integrated_objective_has_no_large_plan_term(self) -> None:
        planner = ContiguousZoneGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(verbose=False),
        )

        preparation = planner._prepare_zones()

        self.assertEqual(
            {
                "voyage_area_dispersion",
                "zone_dispersion",
                "existing_group_proximity",
                "unused_capacity",
                "berth_distance",
            },
            set(planner._zone_objective_weights()),
        )
        self.assertAlmostEqual(
            1.0,
            sum(planner._zone_objective_weights().values()),
            places=12,
        )
        policy = preparation["peak_utilization_policy"]
        self.assertAlmostEqual(0.75, policy["load_lower_bound"], places=8)
        self.assertAlmostEqual(0.875, policy["epsilon_cap"], places=8)
        self.assertFalse(policy["terminal_approved_threshold_used"])

    def test_unused_capacity_ablation_renormalizes_retained_weights(self) -> None:
        baseline = ContiguousZoneGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(verbose=False),
        )
        ablated = ContiguousZoneGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(verbose=False),
            ContiguousZoneConfig(
                unused_capacity_objective_enabled=False
            ),
        )

        baseline_weights = baseline._zone_objective_weights()
        ablated_weights = ablated._zone_objective_weights()
        self.assertEqual(0.0, ablated_weights["unused_capacity"])
        self.assertAlmostEqual(1.0, sum(ablated_weights.values()), places=12)
        retained = [
            key for key in baseline_weights if key != "unused_capacity"
        ]
        ratios = {
            ablated_weights[key] / baseline_weights[key]
            for key in retained
        }
        self.assertEqual(1, len({round(value, 12) for value in ratios}))

    def test_legacy_import_area_reference_is_not_accepted_as_demand(self) -> None:
        problem = make_small_problem()
        problem.import_area_size_reference = {("IF", "A", "20"): 1}

        with self.assertRaisesRegex(ValueError, "direct anonymous import"):
            ContiguousZoneGenerationPlanner(
                problem,
                ColumnGenerationConfig(verbose=False),
            )

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy"),
        "gurobipy is unavailable",
    )
    def test_root_matches_complete_zone_lp(self) -> None:
        planner = ContiguousZoneGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(
                total_time_limit=10.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
        )
        result = planner.analyze_root(compare_complete_lp=True)
        self.assertTrue(result["closed"])
        self.assertEqual(result["complete_zone_lp"]["status"], "optimal")
        self.assertAlmostEqual(result["root_complete_lp_difference"], 0.0, places=9)
        self.assertLess(result["active_zone_count"], result["zone_count"])

    def test_rmq_pricing_matches_complete_enumeration(self) -> None:
        planner = ContiguousZoneGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(verbose=False),
        )
        planner._prepare_zones()
        duals = {}
        for index, column in enumerate(planner._columns):
            duals[("flow_capacity", (column.group_id, column.bay_key))] = (
                -0.031 - 0.000137 * (index + 1)
            )
            duals[("area_zone_support", (column.group_id, column.area_no))] = (
                -0.007 * (index + 1)
            )
            duals[("area_flow_cover", (column.group_id, column.area_no))] = -0.011
        limit = 4
        priced = planner._price_zone_signatures(
            duals,
            per_group_limit=limit,
            active_zone_indices=set(),
            improving_only=False,
        )
        exhaustive = defaultdict(list)
        for strip_key, signature in planner._iter_zone_signatures():
            group_id, area_no, _row_no = strip_key
            capacity = sum(planner._atomic_capacity[index] for index in signature)
            reduced_cost = (
                planner._zone_activation_penalty()
                + duals[("area_zone_support", (group_id, area_no))]
                + sum(
                    planner._atomic_zone_reduced_cost(index, duals)
                    for index in signature
                )
                + min(capacity, planner.groups_by_id[group_id].demand)
                * duals[("area_flow_cover", (group_id, area_no))]
            )
            exhaustive[group_id].append(reduced_cost)
        expected = sorted(
            value
            for values in exhaustive.values()
            for value in sorted(values)[:limit]
        )
        actual = sorted(value for value, _strip, _signature in priced["selected"])
        self.assertEqual(len(expected), len(actual))
        for expected_value, actual_value in zip(expected, actual):
            self.assertAlmostEqual(expected_value, actual_value, places=12)
        self.assertAlmostEqual(
            min(value for values in exhaustive.values() for value in values),
            priced["minimum_reduced_cost"],
            places=12,
        )
        self.assertEqual(priced["pricing_method"], "exact_prefix_rmq_top_k")

    def test_capacity_rule_is_explicit_and_uniform(self) -> None:
        group = ExportGroup(
            group_id="G1",
            voyage_id="V1",
            status="OF",
            port="P1",
            size="20",
            height="96",
            demand=2,
        )
        bays = {
            bay.bay_key: bay
            for bay in (
                make_bay("A", "01"),
                make_bay("A", "03"),
                make_bay("A", "05"),
            )
        }
        problem = ProblemData(
            export_groups=[group],
            bays=bays,
            area_guidance_target={("V1", "OF", "A", "20"): 2},
            area_functions={"A": {"OF"}},
            target_voyages=["V1"],
            export_voyages={"V1"},
            berth_distances={("A", "Q1"): 1.0},
            berth_by_voyage={"V1": "Q1"},
        )
        planner = ContiguousZoneGenerationPlanner(
            problem,
            ColumnGenerationConfig(verbose=False),
        )
        preparation = planner._prepare_zones()
        signatures = list(planner._iter_zone_signatures())
        self.assertEqual(
            preparation["zone_candidate_policy"],
            "contiguous_intervals_with_one_atomic_row_capacity_slack",
        )
        self.assertEqual(preparation["zone_count"], 5)
        self.assertEqual(preparation["excluded_by_zone_capacity_rule_count"], 1)
        self.assertEqual(len(signatures), 5)
        self.assertTrue(
            all(
                sum(planner._atomic_capacity[index] for index in signature)
                <= planner._zone_capacity_limit_by_strip[strip]
                for strip, signature in signatures
            )
        )

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy"),
        "gurobipy is unavailable",
    )
    def test_anonymous_imports_only_affect_capacity_and_peak_constraint(self) -> None:
        problem = make_small_problem()
        problem.import_demand_by_flow_size = {("IF", "20"): 2}
        for functions in problem.area_functions.values():
            functions.add("IF")

        result = ContiguousZoneGenerationPlanner(
            problem,
            ColumnGenerationConfig(
                total_time_limit=10.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
        ).solve()

        certificate = result.diagnostics["zone_objective_certificate"]
        self.assertGreater(len(result.import_reservation_rows), 0)
        self.assertEqual(
            2,
            sum(
                int(row["reserved_boxes"])
                for row in result.import_reservation_rows
            ),
        )
        self.assertNotIn("area_guidance", certificate["weights"])
        self.assertLessEqual(
            certificate["peak_utilization"]["maximum"],
            certificate["peak_utilization"]["epsilon_cap"] + 1e-9,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy"),
        "gurobipy is unavailable",
    )
    def test_exact_fill_is_independently_valid(self) -> None:
        result = ContiguousZoneGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(
                total_time_limit=10.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
        ).solve()
        diagnostics = result.diagnostics
        self.assertTrue(diagnostics["independent_solution_validation"]["passed"])
        self.assertGreater(diagnostics["zone_candidate_reduction"], 0.0)
        self.assertAlmostEqual(
            diagnostics["zone_model_global_lower_bound"],
            diagnostics["zone_root"]["root_objective"],
            places=9,
        )
        self.assertEqual(
            diagnostics["zone_model_lower_bound_source"],
            "closed_exact_zone_pricing_root",
        )
        self.assertGreaterEqual(
            diagnostics["zone_model_upper_bound"] + 1e-9,
            diagnostics["zone_model_global_lower_bound"],
        )
        self.assertTrue(diagnostics["zone_fill"]["recourse_certificate"]["certified"])
        self.assertEqual(diagnostics["zone_fill"]["flow_realization_mismatch_count"], 0)
        certificate = diagnostics["zone_objective_certificate"]
        self.assertTrue(certificate["certified"])
        self.assertAlmostEqual(
            certificate["objective"],
            diagnostics["zone_model_upper_bound"],
            places=9,
        )
        self.assertAlmostEqual(sum(certificate["weights"].values()), 1.0, places=12)
        self.assertLessEqual(
            certificate["peak_utilization"]["maximum"],
            certificate["peak_utilization"]["epsilon_cap"] + 1e-9,
        )
        fix_optimize = diagnostics["zone_fix_optimize"]
        self.assertLessEqual(
            fix_optimize["final_objective"],
            fix_optimize["initial_objective"] + 1e-9,
        )
        selection = fix_optimize["local"]["neighborhood_selection"]
        self.assertEqual(
            selection["policy"],
            "objective_mass_under_candidate_zone_fraction",
        )
        self.assertIn(
            selection["binding_condition"],
            {"objective_mass_target", "candidate_zone_budget"},
        )
        self.assertFalse(
            diagnostics["zone_root_proof_cuts_retained_in_primal_search"]
        )
        self.assertNotIn("area_summary_big_plan_inheritance", diagnostics)
        self.assertFalse(
            diagnostics["import_capacity_reservation"]["area_reference_used"]
        )
        row_quality = diagnostics[
            "row_recourse_secondary_quality_components"
        ]
        self.assertFalse(row_quality["large_plan_guidance_used"])
        self.assertNotIn("area_guidance", row_quality["weighted"])

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy"),
        "gurobipy is unavailable",
    )
    def test_complete_zone_mip_has_valid_exact_recourse(self) -> None:
        result = ContiguousZoneGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(
                total_time_limit=10.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
        ).solve_complete_zone_mip()
        diagnostics = result.diagnostics

        self.assertTrue(diagnostics["independent_solution_validation"]["passed"])
        self.assertEqual(
            "complete_redefined_zone_model",
            diagnostics["master_bound_scope"],
        )
        self.assertTrue(
            diagnostics["zone_fill"]["recourse_certificate"]["certified"]
        )
        self.assertAlmostEqual(
            diagnostics["zone_objective_certificate"]["objective"],
            diagnostics["zone_model_upper_bound"],
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
