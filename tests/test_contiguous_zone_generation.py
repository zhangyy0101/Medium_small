from __future__ import annotations

import importlib.util
import unittest
from collections import defaultdict
from time import perf_counter
from unittest.mock import patch

from yard_planning.contiguous_zone_generation import (
    COMPLETE_MIP_BASELINE_POLICY,
    V5_MULTI_START_POLICY,
    ContiguousZoneConfig,
    ContiguousZoneGenerationPlanner,
    RootSnapshot,
)
from yard_planning.gurobi_backend import (
    StagnationStoppingMipProgressRecorder,
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


def make_two_bay_single_group_problem(*, import_boxes: int = 0) -> ProblemData:
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
        for bay in (make_bay("A", "01"), make_bay("A", "03"))
    }
    functions = {"OF"}
    if import_boxes:
        functions.add("IF")
    return ProblemData(
        export_groups=[group],
        bays=bays,
        area_functions={"A": functions},
        target_voyages=["V1"],
        export_voyages={"V1"},
        import_demand_by_flow_size=(
            {("IF", "20"): import_boxes} if import_boxes else {}
        ),
        berth_distances={("A", "Q1"): 1.0},
        berth_by_voyage={"V1": "Q1"},
    )


def make_three_group_conflict_problem() -> ProblemData:
    groups = [
        ExportGroup(
            group_id="G1",
            voyage_id="V1",
            status="OF",
            port="P1",
            size="20",
            height="96",
            demand=2,
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
        ExportGroup(
            group_id="G3",
            voyage_id="V2",
            status="OF",
            port="P3",
            size="20",
            height="86",
            demand=2,
        ),
    ]
    area_a_01 = make_bay("A", "01")
    area_a_03 = make_bay("A", "03")
    area_b_01 = make_bay("B", "01")
    area_b_03 = make_bay("B", "03")
    area_a_01.existing_heights = {"96"}
    area_a_03.existing_heights = {"96"}
    area_b_01.existing_heights = {"86"}
    area_b_03.existing_heights = {"86"}
    bays = {
        bay.bay_key: bay
        for bay in (
            area_a_01,
            area_a_03,
            area_b_01,
            area_b_03,
        )
    }
    return ProblemData(
        export_groups=groups,
        bays=bays,
        area_functions={"A": {"OF"}, "B": {"OF"}},
        target_voyages=["V1", "V2"],
        export_voyages={"V1", "V2"},
        berth_distances={
            ("A", "Q1"): 1.0,
            ("B", "Q1"): 2.0,
            ("A", "Q2"): 2.0,
            ("B", "Q2"): 1.0,
        },
        berth_by_voyage={"V1": "Q1", "V2": "Q2"},
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
        with self.assertRaises(ValueError):
            ContiguousZoneConfig(
                mip_start_total_time_fraction=0.05,
                mip_start_repair_total_fraction=0.08,
            ).validate()
        with self.assertRaises(ValueError):
            ContiguousZoneConfig(
                primal_pool_apply_lp_warm_start=1,
            ).validate()
        with self.assertRaises(ValueError):
            ContiguousZoneConfig(fix_optimize_max_rounds=0).validate()
        with self.assertRaises(ValueError):
            ContiguousZoneConfig(
                fix_optimize_max_rounds=3,
                fix_optimize_round_zone_fractions=(0.1, 0.2),
            ).validate()
        with self.assertRaises(ValueError):
            ContiguousZoneConfig(
                fix_optimize_round_zone_fractions=(0.2, 0.1, 0.3),
            ).validate()
        with self.assertRaises(ValueError):
            ContiguousZoneConfig(
                initial_mip_max_remaining_fraction=0.0
            ).validate()
        with self.assertRaises(ValueError):
            ContiguousZoneConfig(
                initial_mip_stagnation_total_fraction=0.0
            ).validate()
        with self.assertRaises(ValueError):
            ContiguousZoneConfig(
                initial_mip_min_relative_improvement=0.0
            ).validate()
        ContiguousZoneConfig(
            fix_optimize_policy="hybrid_multi_round"
        ).validate()
        ContiguousZoneConfig(
            fix_optimize_policy="directed_conflict_multi_round"
        ).validate()
        ContiguousZoneConfig(
            fix_optimize_policy="rotating_conflict_multi_round"
        ).validate()

    def test_initial_mip_stagnation_requires_feasible_incumbent(self) -> None:
        recorder = StagnationStoppingMipProgressRecorder(
            phase="test",
            minimum_run_seconds=6.0,
            stagnation_seconds=12.0,
            minimum_relative_improvement=1e-4,
        )
        self.assertFalse(
            recorder.observe_stopping_state(
                elapsed_seconds=100.0,
                incumbent=None,
            )
        )
        self.assertIsNone(recorder.termination_reason)

    def test_initial_mip_stagnation_resets_only_on_meaningful_gain(self) -> None:
        recorder = StagnationStoppingMipProgressRecorder(
            phase="test",
            minimum_run_seconds=6.0,
            stagnation_seconds=12.0,
            minimum_relative_improvement=1e-4,
        )
        self.assertFalse(
            recorder.observe_stopping_state(
                elapsed_seconds=1.0,
                incumbent=1.0,
            )
        )
        self.assertFalse(
            recorder.observe_stopping_state(
                elapsed_seconds=8.0,
                incumbent=0.99995,
            )
        )
        self.assertTrue(
            recorder.observe_stopping_state(
                elapsed_seconds=13.0,
                incumbent=0.99995,
            )
        )
        self.assertEqual("stagnation", recorder.termination_reason)

        reset = StagnationStoppingMipProgressRecorder(
            phase="test",
            minimum_run_seconds=6.0,
            stagnation_seconds=12.0,
            minimum_relative_improvement=1e-4,
        )
        reset.observe_stopping_state(elapsed_seconds=1.0, incumbent=1.0)
        self.assertFalse(
            reset.observe_stopping_state(
                elapsed_seconds=10.0,
                incumbent=0.9998,
            )
        )
        self.assertFalse(
            reset.observe_stopping_state(
                elapsed_seconds=21.9,
                incumbent=0.9998,
            )
        )
        self.assertTrue(
            reset.observe_stopping_state(
                elapsed_seconds=22.0,
                incumbent=0.9998,
            )
        )

    def test_hybrid_multiround_uses_objective_then_conflict(self) -> None:
        planner = ContiguousZoneGenerationPlanner(
            make_three_group_conflict_problem(),
            ColumnGenerationConfig(verbose=False),
            ContiguousZoneConfig(fix_optimize_policy="hybrid_multi_round"),
        )
        planner._prepare_zones()
        observed: list[tuple[int, str, float]] = []

        def objective_selection(*_args, **kwargs):
            return {"G1"}, {
                "policy": "objective_test",
                "selected_groups": ["G1"],
                "candidate_zone_budget": 1,
                "selected_candidate_zone_count": 1,
                "candidate_zone_fraction_limit": kwargs[
                    "candidate_zone_fraction"
                ],
            }

        def conflict_selection(*_args, **kwargs):
            return {"G2"}, {
                "policy": "conflict_test",
                "seed_group": "G2",
                "selected_groups": ["G2"],
                "candidate_zone_budget": 1,
                "selected_candidate_zone_count": 1,
                "candidate_zone_fraction": kwargs[
                    "candidate_zone_fraction"
                ],
            }

        def controlled_round(
            _model,
            _variables,
            zones,
            flow,
            imports,
            objective,
            **kwargs,
        ):
            selection = kwargs["neighborhood_selection"]
            fraction = selection.get(
                "candidate_zone_fraction_limit",
                selection.get("candidate_zone_fraction"),
            )
            observed.append(
                (
                    kwargs["round_id"],
                    selection["selection_family"],
                    fraction,
                )
            )
            return set(zones), dict(flow), dict(imports), {
                "round_id": kwargs["round_id"],
                "seed_group": selection.get("seed_group"),
                "selected_groups": selection["selected_groups"],
                "neighborhood_group_count": len(kwargs["neighborhood"]),
                "candidate_zone_budget": selection["candidate_zone_budget"],
                "candidate_zone_count": selection[
                    "selected_candidate_zone_count"
                ],
                "objective_before": objective,
                "local_objective": objective,
                "master_reoptimized_objective": objective,
                "objective_after": objective,
                "absolute_improvement": 0.0,
                "relative_improvement": 0.0,
                "selection_seconds": 0.0,
                "build_seconds": 0.0,
                "local_mip_seconds": 0.0,
                "master_reoptimization_seconds": 0.0,
                "local_nodes": 0.0,
                "local_time_to_first": 0.0,
                "local_time_to_best": 0.0,
                "added_zone_count": 0,
                "added_zone_indices": [],
                "improved": False,
                "neighborhood_selection": selection,
                "local": {},
                "master_reoptimization": {},
                "seconds": 0.0,
            }

        with (
            patch.object(
                planner,
                "_zone_objective_certificate",
                return_value={"objective": 1.0},
            ),
            patch.object(planner, "_gurobi_objective_value", return_value=1.0),
            patch.object(
                planner,
                "_objective_fix_optimize_neighborhood",
                side_effect=objective_selection,
            ),
            patch.object(
                planner,
                "_select_conflict_fix_optimize_neighborhood",
                side_effect=conflict_selection,
            ),
            patch.object(
                planner,
                "_run_fix_optimize_round",
                side_effect=controlled_round,
            ),
        ):
            _zones, _flow, _imports, diagnostics = (
                planner._run_objective_fix_optimize(
                    object(),
                    {},
                    set(),
                    {},
                    {},
                    perf_counter() + 5.0,
                )
            )

        self.assertEqual(3, diagnostics["round_count"])
        self.assertEqual(
            [
                (1, "objective", 0.12),
                (2, "conflict", 0.22),
                (3, "conflict", 0.35),
            ],
            observed,
        )

    def test_conflict_graph_prioritizes_shared_physical_resources(self) -> None:
        planner = ContiguousZoneGenerationPlanner(
            make_three_group_conflict_problem(),
            ColumnGenerationConfig(verbose=False),
        )
        planner._prepare_zones()

        scores, diagnostics = planner._build_group_conflict_scores(
            set(),
            {("G1", "A|01"): 2},
            {},
        )

        coupled = scores[("G1", "G2")]
        independent = scores[("G1", "G3")]
        self.assertGreater(coupled["score"], independent["score"])
        self.assertEqual(1.0, coupled["same_voyage"])
        self.assertEqual(1.0, coupled["shared_area"])
        self.assertEqual(1.0, coupled["resource_conflict"])
        self.assertGreater(coupled["peak_exchange"], 0.0)
        self.assertAlmostEqual(
            coupled["score"],
            sum(
                coupled[name]
                for name in (
                    "same_voyage",
                    "shared_area",
                    "resource_conflict",
                    "peak_exchange",
                )
            )
            / 4.0,
        )
        self.assertEqual(0.0, independent["score"])
        self.assertEqual(
            {
                "same_voyage": 0.25,
                "shared_area": 0.25,
                "resource_conflict": 0.25,
                "peak_exchange": 0.25,
            },
            diagnostics["component_weights"],
        )
        self.assertFalse(diagnostics["full_zone_pair_materialization_used"])

    def test_directed_conflict_graph_targets_current_resource_owner(self) -> None:
        planner = ContiguousZoneGenerationPlanner(
            make_three_group_conflict_problem(),
            ColumnGenerationConfig(verbose=False),
        )
        planner._prepare_zones()
        strip_key, signature = next(planner._iter_zone_signatures("G2"))
        g2_zone = planner._register_zone(strip_key, signature)

        scores, diagnostics = planner._build_directed_group_conflict_scores(
            {g2_zone},
            {
                ("G1", "A|03"): 2,
                ("G2", "A|01"): 2,
                ("G3", "B|01"): 2,
            },
            {},
        )

        release_owner = scores[("G1", "G2")]
        unrelated_owner = scores[("G1", "G3")]
        self.assertGreater(release_owner["score"], unrelated_owner["score"])
        self.assertEqual("G1->G2", release_owner["direction"])
        self.assertTrue(release_owner["retained"])
        self.assertEqual(1.0, release_owner["same_voyage"])
        self.assertEqual(1.0, release_owner["shared_area"])
        self.assertGreater(release_owner["resource_conflict"], 0.0)
        self.assertGreater(release_owner["peak_exchange"], 0.0)
        self.assertEqual(6, diagnostics["arc_count"])
        self.assertLessEqual(
            diagnostics["retained_arc_count"],
            len(planner.groups) * diagnostics["outgoing_arc_limit"],
        )
        self.assertEqual(
            "lowest_business_cost_sqrt_area_frontier",
            diagnostics["candidate_frontier"]["policy"],
        )

    def test_multiround_rotates_seed_and_accepts_later_improvement(self) -> None:
        planner = ContiguousZoneGenerationPlanner(
            make_three_group_conflict_problem(),
            ColumnGenerationConfig(verbose=False),
        )
        planner._prepare_zones()
        observed = []

        def controlled_round(
            _model,
            _variables,
            zones,
            flow,
            imports,
            objective,
            **kwargs,
        ):
            selection = kwargs["neighborhood_selection"]
            observed.append(
                (
                    kwargs["round_id"],
                    selection["seed_group"],
                    selection["candidate_zone_budget"],
                )
            )
            improved = kwargs["round_id"] == 2
            after = 0.8 if improved or objective <= 0.8 else objective
            return set(zones), dict(flow), dict(imports), {
                "round_id": kwargs["round_id"],
                "seed_group": selection["seed_group"],
                "selected_groups": selection["selected_groups"],
                "neighborhood_group_count": len(kwargs["neighborhood"]),
                "candidate_zone_budget": selection["candidate_zone_budget"],
                "candidate_zone_count": selection[
                    "selected_candidate_zone_count"
                ],
                "objective_before": objective,
                "local_objective": after,
                "master_reoptimized_objective": after,
                "objective_after": after,
                "absolute_improvement": max(0.0, objective - after),
                "relative_improvement": max(0.0, objective - after)
                / max(abs(objective), 1e-12),
                "selection_seconds": 0.0,
                "build_seconds": 0.0,
                "local_mip_seconds": 0.0,
                "master_reoptimization_seconds": 0.0,
                "local_nodes": 0.0,
                "local_time_to_first": 0.0,
                "local_time_to_best": 0.0,
                "added_zone_count": 0,
                "added_zone_indices": [],
                "improved": improved,
                "neighborhood_selection": selection,
                "local": {},
                "master_reoptimization": {},
                "seconds": 0.0,
            }

        with (
            patch.object(
                planner,
                "_zone_objective_certificate",
                return_value={"objective": 1.0},
            ),
            patch.object(planner, "_gurobi_objective_value", return_value=1.0),
            patch.object(
                planner,
                "_run_fix_optimize_round",
                side_effect=controlled_round,
            ),
        ):
            _zones, _flow, _imports, diagnostics = (
                planner._run_objective_fix_optimize(
                    object(),
                    {},
                    set(),
                    {},
                    {},
                    perf_counter() + 5.0,
                )
            )

        self.assertEqual(3, diagnostics["round_count"])
        self.assertEqual(1, diagnostics["successful_round_count"])
        self.assertLessEqual(
            diagnostics["final_objective"],
            diagnostics["initial_objective"],
        )
        self.assertAlmostEqual(0.8, diagnostics["final_objective"])
        self.assertNotEqual(observed[0][1], observed[1][1])
        self.assertLessEqual(observed[0][2], observed[1][2])
        self.assertLessEqual(observed[1][2], observed[2][2])
        self.assertLess(observed[0][2], observed[2][2])

    def test_meaningful_seed_rotation_excludes_successful_prior_seeds(
        self,
    ) -> None:
        planner = ContiguousZoneGenerationPlanner(
            make_three_group_conflict_problem(),
            ColumnGenerationConfig(verbose=False),
            ContiguousZoneConfig(
                fix_optimize_policy="rotating_conflict_multi_round"
            ),
        )
        planner._prepare_zones()
        observed_exclusions: list[set[str]] = []

        def rotating_selection(*_args, **kwargs):
            excluded = set(kwargs["excluded_seed_groups"])
            observed_exclusions.append(excluded)
            seed = next(
                group_id
                for group_id in ("G1", "G2", "G3")
                if group_id not in excluded
            )
            return {seed}, {
                "policy": "rotation_test",
                "seed_group": seed,
                "selected_groups": [seed],
                "selected_group_count": 1,
                "candidate_zone_budget": 1,
                "selected_candidate_zone_count": 1,
                "candidate_zone_fraction": kwargs[
                    "candidate_zone_fraction"
                ],
            }

        def improving_round(
            _model,
            _variables,
            zones,
            flow,
            imports,
            objective,
            **kwargs,
        ):
            after = objective - 0.1
            selection = kwargs["neighborhood_selection"]
            return set(zones), dict(flow), dict(imports), {
                "round_id": kwargs["round_id"],
                "seed_group": selection["seed_group"],
                "selected_groups": selection["selected_groups"],
                "neighborhood_group_count": 1,
                "candidate_zone_budget": 1,
                "candidate_zone_count": 1,
                "objective_before": objective,
                "local_objective": after,
                "master_reoptimized_objective": after,
                "objective_after": after,
                "absolute_improvement": 0.1,
                "relative_improvement": 0.1 / objective,
                "selection_seconds": 0.0,
                "build_seconds": 0.0,
                "local_mip_seconds": 0.0,
                "master_reoptimization_seconds": 0.0,
                "local_nodes": 0.0,
                "local_time_to_first": 0.0,
                "local_time_to_best": 0.0,
                "added_zone_count": 0,
                "added_zone_indices": [],
                "improved": True,
                "neighborhood_selection": selection,
                "local": {},
                "master_reoptimization": {},
                "seconds": 0.0,
            }

        with (
            patch.object(
                planner,
                "_zone_objective_certificate",
                return_value={"objective": 1.0},
            ),
            patch.object(planner, "_gurobi_objective_value", return_value=1.0),
            patch.object(
                planner,
                "_select_conflict_fix_optimize_neighborhood",
                side_effect=rotating_selection,
            ),
            patch.object(
                planner,
                "_run_fix_optimize_round",
                side_effect=improving_round,
            ),
        ):
            _zones, _flow, _imports, diagnostics = (
                planner._run_objective_fix_optimize(
                    object(),
                    {},
                    set(),
                    {},
                    {},
                    perf_counter() + 5.0,
                )
            )

        self.assertEqual([set(), {"G1"}, {"G1", "G2"}], observed_exclusions)
        self.assertEqual(["G1", "G2", "G3"], diagnostics["attempted_seed_groups"])
        self.assertEqual(3, diagnostics["distinct_attempted_seed_group_count"])
        self.assertEqual(
            "exclude_all_previously_attempted_seeds",
            diagnostics["seed_rotation_policy"],
        )

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

    def test_greedy_generation_evaluates_all_policies_before_selecting(self) -> None:
        planner = ContiguousZoneGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(verbose=False),
        )
        planner._prepare_zones()

        candidates, diagnostics = planner._generate_greedy_support_candidates({})
        selected, covered, protection = planner._on_demand_greedy_support({})

        self.assertEqual(20, diagnostics["generated_candidate_count"])
        generated = diagnostics["generated_candidate_summaries"]
        self.assertEqual(
            {
                "candidate_scarcity",
                "demand_descending",
                "group_id",
                "voyage_clustered",
            },
            {item["ordering_policy"] for item in generated},
        )
        self.assertEqual(
            {0.0, 0.25, 0.5, 0.75, 1.0},
            {item["import_protection"] for item in generated},
        )
        self.assertEqual(list(candidates[0]["support"]), selected)
        self.assertEqual(dict(candidates[0]["covered"]), covered)
        self.assertEqual(candidates[0]["import_protection"], protection)

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy"),
        "gurobipy is unavailable",
    )
    def test_joint_repair_rejects_export_only_feasible_support(self) -> None:
        planner = ContiguousZoneGenerationPlanner(
            make_two_bay_single_group_problem(import_boxes=2),
            ColumnGenerationConfig(
                total_time_limit=5.0,
                solver_threads=1,
                verbose=False,
            ),
        )
        planner._prepare_zones()
        planner._materialize_all_zones()
        oversized_support = {
            zone.zone_id for zone in planner._zones if zone.capacity == 4
        }
        self.assertEqual(1, len(oversized_support))

        repaired = planner._repair_zone_support(
            oversized_support,
            perf_counter() + 2.0,
        )

        self.assertFalse(repaired["feasible"])
        self.assertIn(repaired["status"], {"infeasible", "inforunbd"})

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy"),
        "gurobipy is unavailable",
    )
    def test_two_repaired_starts_are_submitted_and_bound_final_incumbent(self) -> None:
        planner = ContiguousZoneGenerationPlanner(
            make_two_bay_single_group_problem(),
            ColumnGenerationConfig(
                total_time_limit=10.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
        )
        planner._prepare_zones()
        one_bay_supports = [
            [(strip_key, signature)]
            for strip_key, signature in planner._iter_zone_signatures("G1")
            if sum(planner._atomic_capacity[index] for index in signature) == 2
        ]
        self.assertGreaterEqual(len(one_bay_supports), 2)
        candidates = [
            {
                "support": support,
                "covered": {"G1": 2},
                "support_key": tuple(sorted(support)),
                "ordering_policy": f"test_{index}",
                "import_protection": 0.0,
                "complete_export_cover": True,
                "generation_status": "complete",
                "covered_boxes": 2,
                "selected_zone_count": 1,
            }
            for index, support in enumerate(one_bay_supports[:2])
        ]
        generated = [
            (
                candidate,
                {
                    "ordering_policy": candidate["ordering_policy"],
                    "import_protection": 0.0,
                    "covered_boxes": 2,
                    "complete_export_cover": True,
                    "selected_zone_count": 1,
                    "generation_status": "complete",
                },
            )
            for candidate in candidates
        ]
        model, variables = planner._build_zone_master()
        try:
            planner._remove_proof_only_area_rows(model, variables)
            with (
                patch.object(
                    planner,
                    "_greedy_candidate_strategies",
                    return_value=[
                        ("family_a", ["G1"], 1.0),
                        ("family_b", ["G1"], 1.0),
                    ],
                ),
                patch.object(
                    planner,
                    "_generate_greedy_support_candidate",
                    side_effect=generated,
                ) as generate_spy,
            ):
                _zones, _flow, _imports, stats = planner._integerize_zone_master(
                    model,
                    variables,
                    perf_counter() + 5.0,
                    start_policy=V5_MULTI_START_POLICY,
                )
        finally:
            planner._free_gurobi_model(model)

        start = stats["mip_start"]
        self.assertEqual(2, generate_spy.call_count)
        self.assertGreaterEqual(start["feasible_repaired_count"], 2)
        self.assertGreaterEqual(start["provided_mip_start_count"], 2)
        self.assertEqual(
            start["provided_mip_start_count"],
            start["solver_submission"]["provided_mip_start_count"],
        )
        self.assertLessEqual(
            stats["objective"],
            start["best_repaired_start_objective"] + 1e-9,
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
            ContiguousZoneConfig(
                initial_mip_dynamic_stopping_enabled=True
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
        self.assertEqual(3, fix_optimize["round_count"])
        self.assertEqual(3, len(fix_optimize["rounds"]))
        selection = fix_optimize["rounds"][0]["neighborhood_selection"]
        self.assertEqual(
            selection["policy"],
            "conflict_aware_objective_seed_under_zone_budget",
        )
        self.assertEqual(
            {
                "same_voyage": 0.25,
                "shared_area": 0.25,
                "resource_conflict": 0.25,
                "peak_exchange": 0.25,
            },
            selection["conflict_graph"]["component_weights"],
        )
        required_round_fields = {
            "round_id",
            "seed_group",
            "selected_groups",
            "neighborhood_group_count",
            "candidate_zone_budget",
            "candidate_zone_count",
            "objective_before",
            "local_objective",
            "master_reoptimized_objective",
            "objective_after",
            "absolute_improvement",
            "relative_improvement",
            "build_seconds",
            "local_mip_seconds",
            "master_reoptimization_seconds",
            "local_nodes",
            "local_time_to_first",
            "local_time_to_best",
            "improved",
        }
        self.assertTrue(
            all(
                required_round_fields.issubset(round_values)
                for round_values in fix_optimize["rounds"]
            )
        )
        self.assertFalse(
            diagnostics["zone_root_proof_cuts_retained_in_primal_search"]
        )
        self.assertTrue(
            diagnostics["zone_mip"]["initial_mip_stopping"]["enabled"]
        )
        self.assertIn(
            diagnostics["zone_mip"]["termination_reason"],
            {"stagnation", "time_limit", "optimal"},
        )
        self.assertEqual(
            "integrated_zone_v5_seed_rotation_phase3_3",
            diagnostics["algorithm_version"],
        )
        mip_start = diagnostics["mip_start_diagnostics"]
        self.assertEqual(
            V5_MULTI_START_POLICY,
            mip_start["integer_search_policy"],
        )
        self.assertTrue(mip_start["v5_multi_start_enabled"])
        self.assertTrue(mip_start["candidate_generation_executed"])
        self.assertTrue(diagnostics["proof_primal_pool_separated"])
        self.assertGreater(
            diagnostics["proof_pool_zone_count"],
            diagnostics["primal_pool_zone_count"],
        )
        self.assertTrue(
            diagnostics["root_snapshot"][
                "proof_model_destroyed_before_primal_master"
            ]
        )
        primal_pool = diagnostics["primal_pool_diagnostics"]
        self.assertEqual(
            "group_specific_round_robin_diversified_columns",
            primal_pool["policy"],
        )
        self.assertTrue(primal_pool["lp_warm_start"]["enabled"])
        self.assertGreater(
            len(primal_pool["primal_pool_columns_by_origin"]),
            1,
        )
        anytime = diagnostics["mip_anytime"]
        self.assertIsNotNone(anytime["time_to_first_solution"])
        self.assertIsNotNone(anytime["time_to_best_solution"])
        self.assertTrue(
            any(
                event["event_type"] == "first_incumbent"
                for event in anytime["incumbent_trajectory"]
            )
        )
        self.assertTrue(
            any(
                event["event_type"] == "final"
                for event in anytime["incumbent_trajectory"]
            )
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
        planner = ContiguousZoneGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(
                total_time_limit=10.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
        )
        with patch.object(
            planner,
            "_generate_greedy_support_candidate",
            wraps=planner._generate_greedy_support_candidate,
        ) as v5_generation_spy, patch.object(
            planner,
            "_build_integrality_aware_primal_pool",
            wraps=planner._build_integrality_aware_primal_pool,
        ) as primal_pool_spy, patch.object(
            planner,
            "_build_group_conflict_scores",
            wraps=planner._build_group_conflict_scores,
        ) as conflict_graph_spy:
            result = planner.solve_complete_zone_mip()
        diagnostics = result.diagnostics

        self.assertEqual(0, v5_generation_spy.call_count)
        self.assertEqual(0, primal_pool_spy.call_count)
        self.assertEqual(0, conflict_graph_spy.call_count)
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
        zone_mip = diagnostics["zone_mip"]
        self.assertEqual(
            COMPLETE_MIP_BASELINE_POLICY,
            zone_mip["integer_search_policy"],
        )
        self.assertFalse(zone_mip["v5_multi_start_enabled"])
        self.assertFalse(zone_mip["candidate_generation_executed"])
        self.assertFalse(zone_mip["repair_executed"])
        self.assertEqual(
            "v4_single_partial_gurobi_repair_start",
            zone_mip["baseline_start_type"],
        )
        self.assertIn("mip_progress", zone_mip)

    def test_candidate_budget_interruption_is_not_infeasibility(self) -> None:
        planner = ContiguousZoneGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(verbose=False),
        )
        planner._prepare_zones()

        class FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                self.value += 1.0
                return self.value

        with patch(
            "yard_planning.contiguous_zone_generation.perf_counter",
            side_effect=FakeClock(),
        ):
            candidates, diagnostics = planner._generate_greedy_support_candidates(
                {},
                deadline=3.5,
            )

        self.assertEqual([], candidates)
        self.assertEqual(1, diagnostics["candidate_generation_interrupted_count"])
        self.assertEqual(1, diagnostics["budget_interrupted_candidate_count"])
        self.assertEqual(0, diagnostics["infeasible_candidate_count"])

    def test_candidate_and_repair_share_one_hard_deadline(self) -> None:
        planner = ContiguousZoneGenerationPlanner(
            make_two_bay_single_group_problem(),
            ColumnGenerationConfig(verbose=False),
        )
        planner._prepare_zones()
        support = [next(planner._iter_zone_signatures("G1"))]

        class MutableClock:
            def __init__(self) -> None:
                self.now = 0.0

            def __call__(self) -> float:
                return self.now

        clock = MutableClock()
        calls = {"candidate": 0, "repair": 0}

        class FakeModel:
            def update(self) -> None:
                return None

            def apply_mip_starts(self, starts, *, deadline=None):
                materialized = list(starts)
                return {
                    "requested_mip_start_count": len(materialized),
                    "provided_mip_start_count": len(materialized),
                    "assigned_value_counts": [len(start) for start in materialized],
                    "deadline_exhausted": False,
                    "solver_acceptance_observed": False,
                }

        def generate_candidate(*_args, deadline: float, **_kwargs):
            calls["candidate"] += 1
            if calls["candidate"] == 1:
                clock.now = 3.0
                candidate = {
                    "support": support,
                    "covered": {"G1": 2},
                    "support_key": tuple(sorted(support)),
                    "ordering_policy": "family_a",
                    "import_protection": 1.0,
                    "covered_boxes": 2,
                    "complete_export_cover": True,
                    "generation_status": "complete",
                    "selected_zone_count": 1,
                }
                return candidate, planner._greedy_candidate_summary(candidate)
            clock.now = deadline
            return None, {
                "ordering_policy": "family_b",
                "import_protection": 1.0,
                "covered_boxes": 0,
                "complete_export_cover": False,
                "selected_zone_count": 0,
                "generation_status": "budget_exhausted",
            }

        def repair_support(
            _selected,
            deadline: float,
            start_source_variables=None,
        ):
            del start_source_variables
            calls["repair"] += 1
            clock.now = deadline
            return {
                "status": "optimal",
                "feasible": True,
                "selected_zone_indices": set(_selected),
                "export_flow": {},
                "import_reserve": {},
                "objective": 1.0,
                "seconds": 0.5,
            }

        variables = {"active_zone_indices": set(), "zone": {}}

        def add_zone(_model, local_variables, zone_index):
            variable = ("zone", zone_index)
            local_variables["zone"][zone_index] = variable
            local_variables["active_zone_indices"].add(zone_index)
            return variable

        with (
            patch(
                "yard_planning.contiguous_zone_generation.perf_counter",
                side_effect=clock,
            ),
            patch.object(
                planner,
                "_greedy_candidate_strategies",
                return_value=[
                    ("family_a", ["G1"], 1.0),
                    ("family_b", ["G1"], 1.0),
                ],
            ),
            patch.object(
                planner,
                "_generate_greedy_support_candidate",
                side_effect=generate_candidate,
            ),
            patch.object(
                planner,
                "_repair_zone_support",
                side_effect=repair_support,
            ),
            patch.object(
                planner,
                "_add_zone_variable",
                side_effect=add_zone,
            ),
            patch.object(
                planner,
                "_repaired_start_values",
                return_value={"certified": 1.0},
            ),
        ):
            diagnostics = planner._greedy_zone_mip_start(
                FakeModel(),
                variables,
                start_deadline=6.0,
                start_budget_seconds=6.0,
                repair_time_limit=4.0,
            )

        self.assertEqual(2, calls["candidate"])
        self.assertEqual(1, calls["repair"])
        self.assertLessEqual(
            diagnostics["start_preparation_actual_seconds"],
            diagnostics["start_preparation_budget_seconds"],
        )
        self.assertLessEqual(
            diagnostics["candidate_generation_seconds"]
            + diagnostics["repair_seconds"],
            diagnostics["start_preparation_budget_seconds"],
        )
        self.assertEqual("budget_exhausted", diagnostics["start_termination_reason"])
        self.assertEqual(1, diagnostics["candidate_generation_interrupted_count"])
        self.assertEqual(1, diagnostics["provided_mip_start_count"])

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy"),
        "gurobipy is unavailable",
    )
    def test_zero_start_budget_falls_back_to_main_mip(self) -> None:
        result = ContiguousZoneGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(
                total_time_limit=10.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
            ContiguousZoneConfig(
                mip_start_total_time_fraction=1e-6,
                mip_start_repair_total_fraction=5e-7,
            ),
        ).solve()

        start = result.diagnostics["mip_start_diagnostics"]
        self.assertEqual(0, start["submitted_start_count"])
        self.assertEqual("budget_exhausted", start["start_termination_reason"])
        self.assertTrue(result.diagnostics["independent_solution_validation"]["passed"])

    def test_diversified_pool_is_multi_origin_and_deterministic(self) -> None:
        group = ExportGroup(
            group_id="G1",
            voyage_id="V1",
            status="OF",
            port="P1",
            size="20",
            height="96",
            demand=10,
        )
        problem = ProblemData(
            export_groups=[group],
            bays={
                bay.bay_key: bay
                for bay in (
                    make_bay("A", "01"),
                    make_bay("A", "03"),
                    make_bay("A", "05"),
                    make_bay("A", "07"),
                    make_bay("B", "01"),
                    make_bay("B", "03"),
                    make_bay("B", "05"),
                    make_bay("B", "07"),
                )
            },
            area_functions={"A": {"OF"}, "B": {"OF"}},
            target_voyages=["V1"],
            export_voyages={"V1"},
            berth_distances={("A", "Q1"): 1.0, ("B", "Q1"): 2.0},
            berth_by_voyage={"V1": "Q1"},
        )

        def build_once():
            planner = ContiguousZoneGenerationPlanner(
                problem,
                ColumnGenerationConfig(verbose=False),
            )
            planner._prepare_zones()
            snapshot = RootSnapshot(
                objective=0.0,
                duals={},
                zone_values={},
                export_flow_values={},
                import_values={},
                area_values={},
                voyage_area_values={},
                attr_values={},
                lp_warm_start=None,
                proof_zone_indices=frozenset(),
            )
            pool, provenance, diagnostics = (
                planner._build_integrality_aware_primal_pool(snapshot)
            )
            signatures = sorted(
                planner._zones[index].candidate_indices for index in pool
            )
            origins = {
                origin for index in pool for origin in provenance[index]
            }
            return signatures, origins, diagnostics

        first_signatures, first_origins, first_diagnostics = build_once()
        second_signatures, second_origins, second_diagnostics = build_once()

        self.assertEqual(first_signatures, second_signatures)
        self.assertEqual(first_origins, second_origins)
        self.assertEqual(
            [
                "root_support",
                "reduced_cost",
                "capacity_fit",
                "business_efficiency",
                "spatial_diversity",
            ],
            first_diagnostics["channel_order"],
        )
        self.assertTrue(
            {
                "reduced_cost",
                "capacity_fit",
                "business_efficiency",
                "spatial_diversity",
            }.issubset(first_origins)
        )
        self.assertEqual(
            first_diagnostics["base_primal_pool_zone_count"],
            second_diagnostics["base_primal_pool_zone_count"],
        )

    def test_mandatory_start_columns_may_overflow_group_budget(self) -> None:
        planner = ContiguousZoneGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(verbose=False),
        )
        planner._prepare_zones()
        planner._materialize_all_zones()
        by_group = {
            group.group_id: list(planner._zone_indices_by_group[group.group_id])
            for group in planner.groups
        }
        base = {indices[0] for indices in by_group.values()}
        mandatory = {indices[1] for indices in by_group.values()}
        actual = base | mandatory
        provenance = defaultdict(set)
        for zone_index in base:
            provenance[zone_index].add("capacity_fit")
        diagnostics = {
            "proof_pool_zone_count": len(planner._zones),
            "per_group": {
                group_id: {"nominal_budget": 1}
                for group_id in by_group
            },
        }

        planner._finalize_primal_pool_diagnostics(
            diagnostics,
            provenance,
            base,
            actual,
            mandatory,
        )

        self.assertTrue(mandatory.issubset(actual))
        self.assertTrue(diagnostics["budget_overflow"])
        self.assertEqual(len(mandatory), diagnostics["mandatory_overflow_count"])
        self.assertEqual(
            len(mandatory),
            diagnostics["primal_pool_columns_by_origin"]["greedy_start"],
        )

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy"),
        "gurobipy is unavailable",
    )
    def test_primal_lp_warm_start_does_not_change_solution(self) -> None:
        common = ColumnGenerationConfig(
            total_time_limit=10.0,
            mip_gap=0.0,
            solver_threads=1,
            verbose=False,
        )
        with_warm_start = ContiguousZoneGenerationPlanner(
            make_small_problem(),
            common,
            ContiguousZoneConfig(primal_pool_apply_lp_warm_start=True),
        ).solve()
        without_warm_start = ContiguousZoneGenerationPlanner(
            make_small_problem(),
            common,
            ContiguousZoneConfig(primal_pool_apply_lp_warm_start=False),
        ).solve()

        self.assertAlmostEqual(
            with_warm_start.diagnostics["zone_model_global_lower_bound"],
            without_warm_start.diagnostics["zone_model_global_lower_bound"],
            places=9,
        )
        self.assertAlmostEqual(
            with_warm_start.diagnostics["zone_model_upper_bound"],
            without_warm_start.diagnostics["zone_model_upper_bound"],
            places=9,
        )
        self.assertTrue(
            with_warm_start.diagnostics[
                "independent_solution_validation"
            ]["passed"]
        )
        self.assertTrue(
            without_warm_start.diagnostics[
                "independent_solution_validation"
            ]["passed"]
        )


if __name__ == "__main__":
    unittest.main()
