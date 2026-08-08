from __future__ import annotations

import unittest
import csv
import importlib.util
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

from yard_planning.models import AttributeRules, Bay, ExportGroup, ProblemData
from yard_planning.direct_milp import DirectMilpPlanner
from yard_planning.logic_benders import (
    LogicBendersConfig,
    LogicBendersPlanner,
)
from yard_planning.profile_resource_benders import (
    ProfileResourceBendersPlanner,
)
from yard_planning.selective_resource_benders import (
    SelectiveResourceBendersPlanner,
)
from yard_planning.planner import (
    ColumnGenerationConfig,
    YardPlanningBase,
)
from yard_planning.output_validator import _parse_integer, validate_output_files
from yard_planning.voyage_plan_column_generation import (
    VoyagePlanColumnGenerationPlanner,
    VoyagePlanPricingConfig,
)


def make_group(size: str) -> ExportGroup:
    return ExportGroup(
        group_id=f"g-{size}",
        voyage_id="V1",
        status="OF",
        port="P1",
        size=size,
        height="96",
        demand=5,
    )


def make_bay(
    area: str,
    bay_no: str,
    row_caps: dict[str, int],
    *,
    physical_capacity: int | None = None,
) -> Bay:
    capacity = int(physical_capacity if physical_capacity is not None else sum(row_caps.values()))
    return Bay(
        area_no=area,
        bay_no=bay_no,
        bay_key=f"{area}|{bay_no}",
        bay_order=int(bay_no),
        cap_by_size={"20": capacity},
        physical_capacity=capacity,
        row_cap_by_size={"20": dict(row_caps)},
        row_physical_capacity=dict(row_caps),
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
            make_bay("A", "01", {"1": 2}),
            make_bay("A", "03", {"1": 2}),
            make_bay("B", "01", {"1": 2}),
            make_bay("B", "03", {"1": 2}),
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


def make_two_voyage_problem() -> ProblemData:
    groups = [
        ExportGroup(
            group_id=f"G{index}",
            voyage_id=f"V{index}",
            status="OF",
            port="P1",
            size="20",
            height="96",
            demand=2,
        )
        for index in (1, 2)
    ]
    bays = {
        bay.bay_key: bay
        for bay in (
            make_bay("A", "01", {"1": 2}),
            make_bay("A", "03", {"1": 2}),
        )
    }
    return ProblemData(
        export_groups=groups,
        bays=bays,
        area_guidance_target={
            ("V1", "OF", "A", "20"): 2,
            ("V2", "OF", "A", "20"): 2,
        },
        area_functions={"A": {"OF"}},
        target_voyages=["V1", "V2"],
        export_voyages={"V1", "V2"},
        berth_distances={
            ("A", "Q1"): 1.0,
            ("A", "Q2"): 1.0,
        },
        berth_by_voyage={"V1": "Q1", "V2": "Q2"},
    )


class ColumnGenerationInvariantTests(unittest.TestCase):
    def test_reconstructed_objective_can_remove_l1_auxiliary_slack(self) -> None:
        slack = YardPlanningBase._absolute_deviation_auxiliary_slack(
            0.28979699,
            0.28970275,
            context="test",
        )
        self.assertAlmostEqual(0.00009424, slack)

    def test_reconstructed_objective_cannot_exceed_solver_objective(self) -> None:
        with self.assertRaises(RuntimeError):
            YardPlanningBase._absolute_deviation_auxiliary_slack(
                0.2,
                0.21,
                context="test",
            )

    def test_logic_benders_configuration_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            LogicBendersPlanner(
                make_small_problem(),
                ColumnGenerationConfig(verbose=False),
                LogicBendersConfig(max_iterations=0),
            )
        with self.assertRaises(ValueError):
            LogicBendersPlanner(
                make_small_problem(),
                ColumnGenerationConfig(verbose=False),
                LogicBendersConfig(voyage_time_limit=0.0),
            )

    def test_voyage_plan_pricing_configuration_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            VoyagePlanColumnGenerationPlanner(
                make_small_problem(),
                ColumnGenerationConfig(verbose=False),
                VoyagePlanPricingConfig(plans_per_pricing=0),
            )

    def test_decomposition_builds_one_complete_block_per_voyage(self) -> None:
        planner = VoyagePlanColumnGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(verbose=False),
        )
        voyages = planner._initialize_candidate_blocks()
        self.assertEqual(("V1",), voyages)
        self.assertEqual(0, len(planner._plans))
        self.assertTrue(
            all(
                candidate.voyage_id == voyage_id
                for voyage_id, candidates in planner._voyage_candidates.items()
                for candidate in candidates
            )
        )

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy"), "gurobipy is unavailable"
    )
    def test_complete_voyage_root_is_exact_on_small_case(self) -> None:
        result = VoyagePlanColumnGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(
                max_iterations=30,
                total_time_limit=20.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
        ).solve_root()
        self.assertEqual("optimal", result["status"])
        self.assertTrue(result["root_exact"])
        self.assertAlmostEqual(0.132, result["root_objective"], places=9)
        self.assertAlmostEqual(
            result["root_objective"], result["valid_lower_bound"], places=9
        )
        self.assertTrue(
            all(
                record["priced_voyage_count"] == result["voyage_count"]
                for record in result["records"]
            )
        )

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy"), "gurobipy is unavailable"
    )
    def test_restricted_row_recovery_uses_generated_plan_locations(self) -> None:
        planner = VoyagePlanColumnGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(
                max_iterations=30,
                total_time_limit=20.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
        )
        result = planner.solve_column_generation()
        self.assertIsNotNone(result["selected_locations"])
        self.assertEqual("restricted_row_milp", result["incumbent_source"])
        self.assertLess(
            result["restricted_row_recovery"]["candidate_count"],
            planner._base_feasible_placement_count + 1,
        )

    def test_column_generation_configuration_is_positive(self) -> None:
        config = ColumnGenerationConfig()
        self.assertGreater(config.reduced_cost_tolerance, 0.0)
        self.assertGreater(config.max_iterations, 0)

    def test_link_bound_uses_demand_capacity_minimum(self) -> None:
        self.assertEqual(4, YardPlanningBase._tight_link_bound(100, 4))
        self.assertEqual(3, YardPlanningBase._tight_link_bound(3, 20))
        with self.assertRaises(ValueError):
            YardPlanningBase._tight_link_bound(0, 20)

    def test_output_quantities_must_be_exact_integers(self) -> None:
        errors: list[str] = []
        self.assertIsNone(_parse_integer("1.5", "planned_boxes", errors))
        self.assertIn("non-integer field", errors[0])

    def test_normalized_policy_weights_sum_to_one_and_follow_policy_hierarchy(self) -> None:
        config = ColumnGenerationConfig()
        concentration = (
            config.area_dispersion_weight
            + config.row_dispersion_weight
            + config.existing_group_proximity_weight
        )
        weights = [
            config.area_dispersion_weight,
            config.row_dispersion_weight,
            config.existing_group_proximity_weight,
            config.area_guidance_weight,
            config.berth_distance_weight,
        ]
        self.assertEqual([0.290, 0.240, 0.070, 0.270, 0.130], weights)
        self.assertAlmostEqual(1.0, sum(weights))
        self.assertGreater(concentration, config.area_guidance_weight)
        self.assertGreater(config.area_guidance_weight, config.berth_distance_weight)
        self.assertGreater(config.area_dispersion_weight, config.row_dispersion_weight)
        self.assertGreater(
            config.row_dispersion_weight,
            config.existing_group_proximity_weight,
        )

    def test_45ft_is_edge_only_without_excluding_other_sizes(self) -> None:
        planner = YardPlanningBase.__new__(YardPlanningBase)
        planner.bays = {
            "edge": Bay(
                area_no="A",
                bay_no="01",
                bay_key="edge",
                bay_order=0,
                cap_by_size={"20": 10, "40": 10, "45": 10},
                physical_capacity=10,
                large_bay_partner_key="partner",
            ),
            "middle": Bay(
                area_no="A",
                bay_no="03",
                bay_key="middle",
                bay_order=1,
                cap_by_size={"20": 10, "40": 10, "45": 10},
                physical_capacity=10,
                large_bay_partner_key="partner",
            ),
            "partner": Bay(
                area_no="A",
                bay_no="02",
                bay_key="partner",
                bay_order=2,
                cap_by_size={"20": 10, "40": 10, "45": 10},
                physical_capacity=10,
            ),
        }
        planner.area_edge_bays = defaultdict(set, {"A": {"edge"}})
        planner._bay_existing_attrs_allow_group = lambda group, footprint: True
        planner._stack_count_for_group = lambda bay_key, size, group: 10
        planner._stack_unit_capacity_for_group = lambda bay_key, size, group: 1

        self.assertEqual(5, planner._max_quantity_in_bay(make_group("20"), "edge"))
        self.assertEqual(0, planner._max_quantity_in_bay(make_group("45"), "middle"))
        self.assertEqual(5, planner._max_quantity_in_bay(make_group("45"), "edge"))

    def test_big_plan_guidance_cannot_override_area_function(self) -> None:
        group = make_group("20")
        bays = {
            "A|01": make_bay("A", "01", {"1": 2}),
            "B|01": make_bay("B", "01", {"1": 2}),
        }
        problem = ProblemData(
            export_groups=[group],
            bays=bays,
            area_guidance_target={("V1", "OF", "B", "20"): 5},
            area_functions={"A": {"OF"}, "B": {"IF"}},
            target_voyages=["V1"],
            export_voyages={"V1"},
            berth_distances={("A", "Q1"): 1.0},
            berth_by_voyage={"V1": "Q1"},
        )
        planner = YardPlanningBase(
            problem, ColumnGenerationConfig(verbose=False)
        )
        self.assertEqual(["A"], planner._candidate_areas_for_group(group))

    def test_unreachable_existing_anchor_is_objective_neutral(self) -> None:
        group = make_group("20")
        anchor_key = (
            "V1",
            "flow=OF",
            "IYC_CSZ_CSIZECD=20",
            "IYC_POT_UNLDPORT=P1",
            "IYC_CHEIGHTCD=96",
        )
        open_bay = make_bay("A", "01", {"1": 5})
        full_anchor_bay = make_bay("Z", "01", {}, physical_capacity=0)
        problem = ProblemData(
            export_groups=[group],
            bays={open_bay.bay_key: open_bay, full_anchor_bay.bay_key: full_anchor_bay},
            area_guidance_target={("V1", "OF", "A", "20"): 5},
            area_functions={"A": {"OF"}, "Z": {"OF"}},
            target_voyages=["V1"],
            export_voyages={"V1"},
            existing_group_area_load={anchor_key + ("Z",): 1},
            existing_group_bay_load={anchor_key + ("Z", "Z|01"): 1},
            berth_distances={("A", "Q1"): 1.0},
            berth_by_voyage={"V1": "Q1"},
        )
        planner = YardPlanningBase(
            problem, ColumnGenerationConfig(verbose=False)
        )
        self.assertNotIn(group.group_id, planner.reachable_anchor_group_ids)
        self.assertEqual(0, planner._anchored_group_demand())
        self.assertEqual(0.0, planner._normalized_existing_proximity(group, "A|01"))

    def test_base_placement_pool_uses_feasible_row_capacity(self) -> None:
        group = ExportGroup(**{**make_group("20").__dict__, "demand": 10})
        bay = make_bay("A", "01", {"1": 3}, physical_capacity=100)
        bay.cap_by_size["20"] = 100
        problem = ProblemData(
            export_groups=[group],
            bays={bay.bay_key: bay},
            area_guidance_target={("V1", "OF", "A", "20"): 10},
            area_functions={"A": {"OF"}},
            target_voyages=["V1"],
            export_voyages={"V1"},
            berth_distances={("A", "Q1"): 1.0},
            berth_by_voyage={"V1": "Q1"},
        )
        planner = YardPlanningBase(
            problem, ColumnGenerationConfig(verbose=False)
        )
        planner._prepare_master_index_sets()
        candidates = planner._base_placements_for_group(group)
        self.assertEqual(1, len(candidates))
        self.assertEqual(
            3, planner._base_location_capacity(group, candidates[0])
        )

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_direct_milp_matches_column_generation_on_small_case(self) -> None:
        config = ColumnGenerationConfig(
            max_iterations=30,
            total_time_limit=20.0,
            mip_gap=0.0,
            verbose=False,
        )
        direct = DirectMilpPlanner(make_small_problem(), config).solve()
        generated = VoyagePlanColumnGenerationPlanner(
            make_small_problem(), config
        ).solve()
        self.assertTrue(
            direct.diagnostics["independent_solution_validation"]["passed"]
        )
        self.assertTrue(
            generated.diagnostics["independent_solution_validation"]["passed"]
        )
        self.assertEqual(
            generated.diagnostics["business_objective"],
            direct.diagnostics["business_objective"],
        )
        self.assertAlmostEqual(
            generated.diagnostics["final_business_objective"],
            direct.diagnostics["final_business_objective"],
            places=9,
        )

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_logic_benders_matches_m0_on_small_case(self) -> None:
        config = ColumnGenerationConfig(
            total_time_limit=20.0,
            mip_gap=0.0,
            solver_threads=1,
            verbose=False,
        )
        direct = DirectMilpPlanner(make_small_problem(), config).solve()
        decomposed = LogicBendersPlanner(
            make_small_problem(),
            config,
            LogicBendersConfig(
                max_iterations=20,
                master_time_limit=3.0,
                voyage_time_limit=3.0,
            ),
        ).solve()
        self.assertTrue(decomposed.diagnostics["lbbd_converged"])
        self.assertTrue(
            decomposed.diagnostics["independent_solution_validation"]["passed"]
        )
        self.assertGreater(
            decomposed.diagnostics["lbbd_cut_counts"][
                "initial_row_conflict_clique"
            ],
            0,
        )
        self.assertAlmostEqual(
            direct.diagnostics["final_business_objective"],
            decomposed.diagnostics["final_business_objective"],
            places=9,
        )

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_logic_benders_master_contains_voyage_row_resources(self) -> None:
        config = ColumnGenerationConfig(
            total_time_limit=20.0,
            mip_gap=0.0,
            solver_threads=1,
            verbose=False,
        )
        planner = LogicBendersPlanner(
            make_small_problem(),
            config,
            LogicBendersConfig(voyage_time_limit=3.0),
        )
        planner._prepare_decomposition()
        master, variables, master_stats = planner._build_master()
        try:
            self.assertGreater(
                master_stats["voyage_row_class_binary_count"], 0
            )
            self.assertEqual(
                master_stats["voyage_row_class_binary_count"],
                len(variables["owner"]),
            )
            self.assertLessEqual(
                master_stats["voyage_row_class_binary_count"],
                len(planner._columns),
            )
            self.assertLessEqual(
                master_stats["row_footprint_template_count"],
                master_stats["voyage_row_class_binary_count"],
            )
        finally:
            planner._free_gurobi_model(master)

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_profile_resource_benders_matches_m0_on_small_case(self) -> None:
        config = ColumnGenerationConfig(
            total_time_limit=20.0,
            mip_gap=0.0,
            solver_threads=1,
            verbose=False,
        )
        direct = DirectMilpPlanner(make_small_problem(), config).solve()
        profile = ProfileResourceBendersPlanner(
            make_small_problem(),
            config,
            LogicBendersConfig(
                max_iterations=10,
                master_time_limit=3.0,
                voyage_time_limit=3.0,
            ),
        ).solve()
        self.assertTrue(
            profile.diagnostics["independent_solution_validation"]["passed"]
        )
        self.assertAlmostEqual(
            direct.diagnostics["final_business_objective"],
            profile.diagnostics["final_business_objective"],
            places=9,
        )
        stats = profile.diagnostics["profile_lbbd_master"]
        self.assertLessEqual(
            stats["row_resource_profile_count"],
            stats["concrete_row_template_count"],
        )
        self.assertEqual(
            stats["profile_state_capacity_constraint_count"],
            stats["profile_owner_integer_count"],
        )
        self.assertTrue(
            profile.diagnostics["profile_lbbd_master_feasibility"].get(
                "exact_start_verified", False
            )
        )
        self.assertGreaterEqual(
            profile.diagnostics["profile_lbbd_master_round_count"], 1
        )
        self.assertTrue(
            profile.diagnostics["profile_lbbd_initial_disaggregation"]
        )
        self.assertNotIn("profile_lbbd_iterations", profile.diagnostics)
        self.assertIn(
            "profile_lbbd_aggregate_feasibility_cut_count",
            profile.diagnostics,
        )
        self.assertIn(
            "profile_lbbd_aggregate_optimality_cut_count",
            profile.diagnostics,
        )

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_selective_resource_benders_matches_m0_on_small_case(
        self,
    ) -> None:
        config = ColumnGenerationConfig(
            total_time_limit=20.0,
            mip_gap=0.0,
            solver_threads=1,
            verbose=False,
        )
        direct = DirectMilpPlanner(make_small_problem(), config).solve()
        selective = SelectiveResourceBendersPlanner(
            make_small_problem(),
            config,
            LogicBendersConfig(
                max_iterations=10,
                master_time_limit=3.0,
                voyage_time_limit=3.0,
            ),
        ).solve()
        self.assertTrue(
            selective.diagnostics["independent_solution_validation"][
                "passed"
            ]
        )
        self.assertAlmostEqual(
            direct.diagnostics["final_business_objective"],
            selective.diagnostics["final_business_objective"],
            places=9,
        )
        self.assertEqual(
            selective.diagnostics["algorithm"],
            "selective_resource_state_lbbd_gurobi",
        )
        stats = selective.diagnostics["selective_lbbd_master"]
        self.assertEqual(
            stats["selected_profile_state_integer_count"]
            + stats["relaxed_profile_state_count"],
            stats["profile_owner_integer_count"],
        )
        self.assertEqual(
            stats["integer_row_count_count"]
            + stats["relaxed_row_count_count"],
            stats["operational_bay_row_count_count"],
        )
        self.assertIn("tightened_area_activation_count", stats)
        self.assertGreaterEqual(stats["tightened_area_activation_count"], 0)
        self.assertGreater(stats["area_cardinality_cover_count"], 0)
        self.assertEqual(
            stats["area_cardinality_cover_count"],
            selective.diagnostics["planned_group_count"],
        )
        self.assertTrue(
            selective.diagnostics["selective_lbbd_master_start"][
                "formal_row_count_integrality_restored"
            ]
        )
        self.assertNotIn("primal_support_state_integer_count", stats)

    def test_selective_recourse_fixes_quantities_not_profile_states(
        self,
    ) -> None:
        planner = SelectiveResourceBendersPlanner.__new__(
            SelectiveResourceBendersPlanner
        )
        self.assertEqual(planner._global_fixed_profile_states(), ())
        self.assertEqual(planner._logic_cut_profile_states(), ())

    def test_selective_initial_states_record_conflict_hypergraph_coverage(
        self,
    ) -> None:
        planner = SelectiveResourceBendersPlanner(
            make_two_voyage_problem(),
            ColumnGenerationConfig(verbose=False),
        )
        planner._prepare_decomposition()
        planner._prepare_profiles()
        selected = set(planner._selected_profile_states)
        self.assertGreater(len(planner._conflict_hyperedges), 0)
        covered = sum(
            bool(selected.intersection(edge))
            for edge in planner._conflict_hyperedges
        )
        self.assertEqual(
            planner._initial_conflict_edge_coverage,
            covered,
        )
        self.assertGreater(covered, 0)

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_selective_iis_promotion_is_one_way(self) -> None:
        planner = SelectiveResourceBendersPlanner(
            make_two_voyage_problem(),
            ColumnGenerationConfig(verbose=False),
        )
        planner._prepare_decomposition()
        planner._prepare_profiles()
        master, variables, _stats = planner._build_profile_master()
        try:
            quantity_key = ("G1", "A|01")
            candidate_index = planner._candidate_indices_by_group_bay[
                quantity_key
            ][0]
            state = (
                planner._profile_by_template[
                    planner._template_by_candidate[candidate_index]
                ],
                planner._owner_key_by_candidate[candidate_index][1],
            )
            representative = planner._columns[
                planner._quantity_representative[quantity_key]
            ]
            row_key = (representative.group_key, quantity_key[1])
            variables["profile_use"][state].VType = "C"
            variables["row_count"][row_key].VType = "C"
            planner._selected_profile_states = tuple(
                candidate
                for candidate in planner._selected_profile_states
                if candidate != state
            )
            master.update()

            promotion = planner._promote_conflict_resources(
                master,
                variables,
                {quantity_key: 1},
                (quantity_key,),
            )
            self.assertTrue(promotion["promoted"])
            self.assertEqual(variables["profile_use"][state].VType, "I")
            self.assertEqual(variables["row_count"][row_key].VType, "I")
            repeated = planner._promote_conflict_resources(
                master,
                variables,
                {quantity_key: 1},
                (quantity_key,),
            )
            self.assertFalse(repeated["promoted"])
        finally:
            planner._free_gurobi_model(master)

    def test_selective_time_budget_scales_with_common_deadline(self) -> None:
        small = SelectiveResourceBendersPlanner(
            make_small_problem(),
            ColumnGenerationConfig(total_time_limit=60.0, verbose=False),
        )
        large = SelectiveResourceBendersPlanner(
            make_small_problem(),
            ColumnGenerationConfig(total_time_limit=180.0, verbose=False),
        )
        self.assertAlmostEqual(
            large._master_feasibility_slice(100.0),
            3.0 * small._master_feasibility_slice(100.0),
        )
        self.assertAlmostEqual(
            large._oracle_allowance(100.0),
            3.0 * small._oracle_allowance(100.0),
        )
        self.assertAlmostEqual(
            large._time_share("primal_polish"),
            3.0 * small._time_share("primal_polish"),
        )
        self.assertAlmostEqual(
            large._master_round_allowance(1, 100.0),
            3.0 * small._master_round_allowance(1, 100.0),
        )

    def test_selective_iis_fallback_limit_scales_sublinearly(self) -> None:
        planner = SelectiveResourceBendersPlanner.__new__(
            SelectiveResourceBendersPlanner
        )
        planner._quantity_upper = range(10_000)
        self.assertEqual(planner._monotone_iis_support_limit(), 300)
        planner._quantity_upper = range(1)
        self.assertEqual(planner._monotone_iis_support_limit(), 64)
        planner._quantity_upper = range(1_000_000)
        self.assertEqual(planner._monotone_iis_support_limit(), 512)

    def test_selective_large_iis_skips_auxiliary_capacity_mip(self) -> None:
        planner = SelectiveResourceBendersPlanner.__new__(
            SelectiveResourceBendersPlanner
        )
        planner._resource_capacity_cache = {}
        core = tuple((f"G{index}", "A|01") for index in range(65))
        result = planner._certify_core_resource_capacity(core, 1.0)
        self.assertFalse(result["certified"])
        self.assertEqual(result["status"], "skipped_large_IIS_core")

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_selective_monotone_iis_cut_is_deduplicated(self) -> None:
        planner = SelectiveResourceBendersPlanner(
            make_small_problem(),
            ColumnGenerationConfig(verbose=False),
        )
        planner._prepare_decomposition()
        planner._prepare_profiles()
        master, variables, _stats = planner._build_profile_master()
        try:
            keys = tuple(sorted(variables["quantity"]))[:2]
            quantities = {key: 1 for key in keys}
            signatures = set()
            added, binary_count = (
                planner._add_monotone_iis_feasibility_cut(
                    master,
                    variables,
                    quantities,
                    keys,
                    1,
                    signatures,
                )
            )
            self.assertTrue(added)
            self.assertEqual(binary_count, len(keys))
            duplicate, duplicate_binary_count = (
                planner._add_monotone_iis_feasibility_cut(
                    master,
                    variables,
                    quantities,
                    keys,
                    2,
                    signatures,
                )
            )
            self.assertFalse(duplicate)
            self.assertEqual(duplicate_binary_count, 0)
        finally:
            planner._free_gurobi_model(master)

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_selective_conflict_repair_relocates_an_infeasible_point(
        self,
    ) -> None:
        planner = SelectiveResourceBendersPlanner(
            make_two_voyage_problem(),
            ColumnGenerationConfig(
                total_time_limit=10.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
        )
        planner._prepare_decomposition()
        planner._prepare_profiles()
        quantities = {
            ("G1", "A|01"): 2,
            ("G2", "A|01"): 2,
        }
        repair = planner._solve_conflict_directed_repair(
            quantities,
            tuple(quantities),
            5.0,
        )
        self.assertTrue(repair["feasible"])
        self.assertEqual(repair["relocated_boxes"], 2)
        self.assertTrue(repair["validation"]["passed"])
        used_bays = {
            planner._columns[index].bay_key
            for index, value in repair["selected"].items()
            if value > 0
        }
        self.assertEqual(used_bays, {"A|01", "A|03"})
        improvement = planner._solve_exact_restricted_primal(
            repair["selected"],
            repair["imports"],
            tuple(quantities),
            5.0,
        )
        self.assertTrue(improvement["feasible"])
        self.assertEqual(improvement["optimized_group_count"], 1)
        self.assertLess(
            improvement["candidate_row_location_count"],
            len(planner._columns),
        )
        self.assertLessEqual(
            improvement["objective"], repair["objective"] + 1e-9
        )

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_selective_resource_capacity_certificate_is_valid(self) -> None:
        planner = SelectiveResourceBendersPlanner(
            make_two_voyage_problem(),
            ColumnGenerationConfig(
                solver_threads=1,
                verbose=False,
            ),
        )
        planner._prepare_decomposition()
        planner._prepare_profiles()
        core = (("G1", "A|01"), ("G2", "A|01"))
        certificate = planner._certify_core_resource_capacity(core, 3.0)
        self.assertTrue(certificate["certified"])
        self.assertEqual(certificate["capacity_upper_bound"], 2)
        self.assertGreater(certificate["typed_route_count"], 0)
        master, variables, _stats = planner._build_profile_master()
        try:
            cut = planner._add_certified_feasibility_cut(
                master,
                variables,
                {key: 2 for key in core},
                core,
                1,
                set(),
            )
            self.assertTrue(cut["added"])
            self.assertEqual(cut["kind"], "IIS_physical_resource_capacity")
            self.assertEqual(cut["binary_count"], 0)
            self.assertEqual(cut["capacity_upper_bound"], 2)
        finally:
            planner._free_gurobi_model(master)

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_selective_voyage_optimality_cut_has_local_support(self) -> None:
        planner = SelectiveResourceBendersPlanner(
            make_small_problem(),
            ColumnGenerationConfig(
                solver_threads=1,
                verbose=False,
            ),
        )
        planner._prepare_decomposition()
        planner._prepare_profiles()
        quantities = {
            ("G1", "A|01"): 1,
            ("G1", "A|03"): 2,
            ("G2", "B|01"): 2,
        }
        bound = planner._solve_voyage_recourse_bound(
            "V1", quantities, 3.0
        )
        self.assertTrue(bound["feasible"])
        self.assertTrue(bound["optimal"])
        master, variables, _stats = planner._build_profile_master()
        try:
            added, binary_count, support_count = (
                planner._add_voyage_optimality_cut(
                    master,
                    variables,
                    "V1",
                    quantities,
                    float(bound["bound"]),
                    1,
                    set(),
                )
            )
            self.assertTrue(added)
            self.assertEqual(support_count, len(quantities))
            self.assertEqual(binary_count, support_count)
            self.assertLess(
                support_count,
                len(variables["quantity"]),
            )
        finally:
            planner._free_gurobi_model(master)
            for subproblem in planner._voyage_subproblems.values():
                planner._free_gurobi_model(subproblem.model)

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_selective_sparse_oracle_exactly_fixes_positive_support(
        self,
    ) -> None:
        planner = SelectiveResourceBendersPlanner(
            make_small_problem(),
            ColumnGenerationConfig(
                total_time_limit=20.0,
                solver_threads=1,
                verbose=False,
            ),
        )
        planner._prepare_decomposition()
        planner._prepare_profiles()
        quantities = {
            ("G1", "A|01"): 1,
            ("G1", "A|03"): 2,
            ("G2", "B|01"): 2,
        }
        result = planner._solve_exact_recourse(quantities, 5.0)
        self.assertTrue(result["feasible"])
        self.assertTrue(result["optimal"])
        planner._validate_exact_quantities(
            quantities, result["selected"]
        )
        self.assertEqual(result["support_quantity_count"], 3)
        self.assertLess(
            result["support_row_location_count"], len(planner._columns)
        )
        self.assertIsNone(planner._global_profile_subproblem)

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_profile_global_oracle_exactly_disaggregates_master_point(
        self,
    ) -> None:
        config = ColumnGenerationConfig(
            total_time_limit=20.0,
            mip_gap=0.0,
            solver_threads=1,
            verbose=False,
        )
        planner = ProfileResourceBendersPlanner(
            make_small_problem(),
            config,
            LogicBendersConfig(voyage_time_limit=3.0),
        )
        planner._prepare_decomposition()
        planner._prepare_profiles()
        master, variables, _stats = planner._build_profile_master()
        try:
            planner._set_gurobi_param(master, "TimeLimit", 5.0)
            master.optimize()
            self.assertGreater(planner._gurobi_solution_count(master), 0)
            quantities, profile_counts, routing, imports, _theta = (
                planner._profile_master_assignment(master, variables)
            )
            fast = planner._verify_profile_assignment(
                quantities,
                profile_counts,
                routing,
                imports,
                perf_counter() + 5.0,
                {},
            )
            exact = planner._solve_global_profile_subproblem(
                quantities,
                profile_counts,
                5.0,
                fast["selected"] if fast["verified"] else None,
            )
            self.assertTrue(exact["feasible"])
            self.assertTrue(exact["optimal"])
            self.assertAlmostEqual(
                exact["objective"], fast["subproblem_objective"], places=9
            )
            self.assertAlmostEqual(
                planner._selected_solution_energy(exact["selected"]),
                planner._selected_solution_energy(fast["selected"]),
                places=9,
            )
        finally:
            planner._free_gurobi_model(master)
            for subproblem in planner._voyage_subproblems.values():
                planner._free_gurobi_model(subproblem.model)
            if planner._global_profile_subproblem is not None:
                planner._free_gurobi_model(
                    planner._global_profile_subproblem.model
                )

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_profile_aggregate_logic_cuts_bind_certified_point(self) -> None:
        config = ColumnGenerationConfig(
            total_time_limit=20.0,
            mip_gap=0.0,
            solver_threads=1,
            verbose=False,
        )
        planner = ProfileResourceBendersPlanner(
            make_small_problem(), config
        )
        planner._prepare_decomposition()
        planner._prepare_profiles()
        master, variables, _stats = planner._build_profile_master()
        try:
            planner._set_gurobi_param(master, "TimeLimit", 5.0)
            master.optimize()
            quantities, profile_counts, _routing, _imports, theta = (
                planner._profile_master_assignment(master, variables)
            )
            signatures: set[tuple] = set()
            certified_bound = sum(theta.values()) + 0.01
            added, binary_count = planner._add_aggregate_logic_cut(
                master,
                variables,
                quantities,
                profile_counts,
                1,
                "optimality",
                signatures,
                lower_bound=certified_bound,
            )
            self.assertTrue(added)
            self.assertGreater(binary_count, 0)
            for key, variable in variables["quantity"].items():
                variable.LB = float(quantities.get(key, 0))
                variable.UB = float(quantities.get(key, 0))
            for state, variable in variables["profile_use"].items():
                variable.LB = float(profile_counts.get(state, 0))
                variable.UB = float(profile_counts.get(state, 0))
            master.update()
            master.optimize()
            self.assertGreater(planner._gurobi_solution_count(master), 0)
            self.assertGreaterEqual(
                sum(
                    planner._gurobi_value(master, variable)
                    for variable in variables["theta"].values()
                ),
                certified_bound - 1e-8,
            )

            added, _binary_count = planner._add_aggregate_logic_cut(
                master,
                variables,
                quantities,
                profile_counts,
                1,
                "feasibility",
                signatures,
            )
            self.assertTrue(added)
            master.optimize()
            self.assertEqual(
                planner._gurobi_status_name(master), "infeasible"
            )
        finally:
            planner._free_gurobi_model(master)

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_profile_solve_uses_global_oracle_after_fast_failure(self) -> None:
        config = ColumnGenerationConfig(
            total_time_limit=20.0,
            mip_gap=0.0,
            solver_threads=1,
            verbose=False,
        )
        planner = ProfileResourceBendersPlanner(
            make_small_problem(),
            config,
            LogicBendersConfig(
                max_iterations=10,
                master_time_limit=3.0,
                voyage_time_limit=3.0,
            ),
        )
        original_verify = planner._verify_profile_assignment
        call_count = 0

        def force_one_fast_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = original_verify(*args, **kwargs)
            if call_count == 2:
                result["verified"] = False
            return result

        planner._verify_profile_assignment = force_one_fast_failure
        result = planner.solve()
        self.assertTrue(
            result.diagnostics["independent_solution_validation"]["passed"]
        )
        self.assertEqual(
            result.diagnostics[
                "profile_lbbd_global_subproblem_solve_count"
            ],
            1,
        )
        self.assertTrue(
            result.diagnostics["profile_lbbd_global_subproblem_records"][0][
                "feasible"
            ]
        )

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_profile_conditional_optimality_cut_is_enforced(self) -> None:
        config = ColumnGenerationConfig(
            total_time_limit=20.0,
            mip_gap=0.0,
            solver_threads=1,
            verbose=False,
        )
        planner = ProfileResourceBendersPlanner(
            make_small_problem(),
            config,
            LogicBendersConfig(voyage_time_limit=3.0),
        )
        planner._prepare_decomposition()
        planner._prepare_profiles()
        master, variables, _stats = planner._build_profile_master()
        try:
            planner._set_gurobi_param(master, "TimeLimit", 5.0)
            master.optimize()
            self.assertGreater(planner._gurobi_solution_count(master), 0)
            quantities, counts, _routing, _imports, theta = (
                planner._profile_master_assignment(master, variables)
            )
            incumbent_signature = (
                tuple(sorted(quantities.items())),
                tuple(sorted(counts.items())),
            )
            required_recourse = sum(theta.values()) + 0.25
            added, binary_count = planner._add_aggregate_logic_cut(
                master,
                variables,
                quantities,
                counts,
                1,
                "optimality",
                set(),
                lower_bound=required_recourse,
            )
            self.assertTrue(added)
            self.assertGreater(binary_count, 0)
            master.optimize()
            self.assertGreater(planner._gurobi_solution_count(master), 0)
            new_quantities, new_counts, _routing, _imports, new_theta = (
                planner._profile_master_assignment(master, variables)
            )
            new_signature = (
                tuple(sorted(new_quantities.items())),
                tuple(sorted(new_counts.items())),
            )
            self.assertTrue(
                new_signature != incumbent_signature
                or sum(new_theta.values()) >= required_recourse - 1e-7
            )
        finally:
            planner._free_gurobi_model(master)

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_profile_resource_benders_preserves_physical_row_separation(
        self,
    ) -> None:
        config = ColumnGenerationConfig(
            total_time_limit=20.0,
            mip_gap=0.0,
            solver_threads=1,
            verbose=False,
        )
        direct = DirectMilpPlanner(
            make_two_voyage_problem(), config
        ).solve()
        profile = ProfileResourceBendersPlanner(
            make_two_voyage_problem(),
            config,
            LogicBendersConfig(
                max_iterations=10,
                master_time_limit=3.0,
                voyage_time_limit=3.0,
            ),
        ).solve()
        self.assertAlmostEqual(
            direct.diagnostics["final_business_objective"],
            profile.diagnostics["final_business_objective"],
            places=9,
        )
        row_voyages: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        for row in profile.export_rows:
            row_voyages[(row["bay_key"], row["row_no"])].add(
                row["voyage_id"]
            )
        self.assertTrue(
            all(len(voyages) == 1 for voyages in row_voyages.values())
        )
        self.assertGreaterEqual(
            profile.diagnostics["profile_lbbd_master"][
                "overlap_physical_pool_constraint_count"
            ],
            0,
        )

    def test_profile_overlap_pool_interval_bound(self) -> None:
        footprints = {
            (("A|01", "01"), ("A|03", "01")),
            (("A|03", "01"), ("A|05", "01")),
        }
        self.assertEqual(
            ProfileResourceBendersPlanner._interval_packing_bound(
                footprints
            ),
            1,
        )

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_two_voyage_plans_preserve_global_row_separation(self) -> None:
        config = ColumnGenerationConfig(
            max_iterations=30,
            total_time_limit=20.0,
            mip_time_limit=5.0,
            mip_gap=0.0,
            solver_threads=1,
            verbose=False,
        )
        direct = DirectMilpPlanner(make_two_voyage_problem(), config).solve()
        generated = VoyagePlanColumnGenerationPlanner(
            make_two_voyage_problem(), config
        ).solve()
        self.assertAlmostEqual(
            direct.diagnostics["final_business_objective"],
            generated.diagnostics["final_business_objective"],
            places=9,
        )
        row_voyages: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        for row in generated.export_rows:
            row_voyages[(row["bay_key"], row["row_no"])].add(row["voyage_id"])
        self.assertTrue(all(len(voyages) == 1 for voyages in row_voyages.values()))

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_logic_benders_preserves_cross_voyage_row_separation(self) -> None:
        config = ColumnGenerationConfig(
            total_time_limit=20.0,
            mip_gap=0.0,
            solver_threads=1,
            verbose=False,
        )
        direct = DirectMilpPlanner(make_two_voyage_problem(), config).solve()
        decomposed = LogicBendersPlanner(
            make_two_voyage_problem(),
            config,
            LogicBendersConfig(
                max_iterations=20,
                master_time_limit=3.0,
                voyage_time_limit=3.0,
            ),
        ).solve()
        self.assertAlmostEqual(
            direct.diagnostics["final_business_objective"],
            decomposed.diagnostics["final_business_objective"],
            places=9,
        )
        row_voyages: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        for row in decomposed.export_rows:
            row_voyages[(row["bay_key"], row["row_no"])].add(
                row["voyage_id"]
            )
        self.assertTrue(
            all(len(voyages) == 1 for voyages in row_voyages.values())
        )

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_direct_and_column_generation_reject_infeasible_hard_demand(self) -> None:
        problem = make_small_problem()
        only_bay = make_bay("A", "01", {"1": 2})
        problem.bays = {only_bay.bay_key: only_bay}
        problem.area_functions = {"A": {"OF"}}
        problem.area_guidance_target = {("V1", "OF", "A", "20"): 5}
        problem.berth_distances = {("A", "Q1"): 1.0}
        config = ColumnGenerationConfig(
            max_iterations=30,
            total_time_limit=20.0,
            mip_gap=0.0,
            verbose=False,
        )
        with self.assertRaisesRegex(RuntimeError, "cannot assign all"):
            DirectMilpPlanner(problem, config).solve()
        with self.assertRaisesRegex(RuntimeError, "complete feasible plan"):
            VoyagePlanColumnGenerationPlanner(problem, config).solve()

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_small_case_finishes_exact_column_generation(self) -> None:
        diagnostics = VoyagePlanColumnGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(mip_gap=0.0, verbose=False),
        ).solve().diagnostics
        self.assertEqual("optimal", diagnostics["master_status"])
        self.assertEqual(
            "complete_voyage_dantzig_wolfe_root_relaxation",
            diagnostics["master_bound_scope"],
        )
        self.assertEqual(
            "exact_complete_voyage_plan_lp",
            diagnostics["complete_model_gap_source"],
        )
        self.assertAlmostEqual(
            0.0, diagnostics["complete_model_relative_gap"], places=10
        )


    def test_written_output_is_validated_independently(self) -> None:
        group = make_group("20")
        bay = Bay(
            area_no="A", bay_no="01", bay_key="A|01",
            bay_order=0, cap_by_size={"20": 4}, physical_capacity=4,
            row_cap_by_size={"20": {"1": 4}}, row_physical_capacity={"1": 4},
        )
        problem = SimpleNamespace(
            export_groups=[group], bays={"A|01": bay},
            import_area_size_reference={}, area_functions={"A": {"OF"}},
        )
        with TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.csv"
            import_path = Path(directory) / "import.csv"
            with plan_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "group_id", "planned_boxes", "area_no", "bay_key", "bay_no", "row_no",
                    "row_allocation", "flow", "size", "height", "voyage_id", "port",
                ])
                writer.writeheader()
                writer.writerow({
                    "group_id": group.group_id, "planned_boxes": 5, "area_no": "A",
                    "bay_key": "A|01", "bay_no": "01", "row_no": "1",
                    "row_allocation": "A|01:1:1", "flow": "OF", "size": "20", "height": "96",
                    "voyage_id": "V1", "port": "P1",
                })
            import_path.touch()
            with self.assertRaisesRegex(ValueError, "capacity"):
                validate_output_files(problem, plan_path, import_path)

    def test_output_validation_rejects_group_identity_tampering(self) -> None:
        group = make_group("20")
        bay = Bay(
            area_no="A", bay_no="01", bay_key="A|01",
            bay_order=0, cap_by_size={"20": 2}, physical_capacity=2,
            row_cap_by_size={"20": {"1": 2}}, row_physical_capacity={"1": 2},
        )
        problem = SimpleNamespace(
            export_groups=[group], bays={"A|01": bay}, export_voyages={"V1"},
            import_area_size_reference={}, area_functions={"A": {"OF"}},
            area_guidance_target={}, attribute_rules=AttributeRules(),
        )
        with TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.csv"
            import_path = Path(directory) / "import.csv"
            with plan_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "group_id", "planned_boxes", "area_no", "bay_key", "bay_no", "row_no",
                    "row_allocation", "flow", "size", "height", "voyage_id", "port",
                ])
                writer.writeheader()
                writer.writerow({
                    "group_id": group.group_id, "planned_boxes": 5, "area_no": "A",
                    "bay_key": "A|01", "bay_no": "01", "row_no": "1",
                    "row_allocation": "A|01:1:1", "flow": "OF", "size": "20",
                    "height": "86", "voyage_id": "V1", "port": "P1",
                })
            import_path.touch()
            with self.assertRaisesRegex(ValueError, "group identity mismatch"):
                validate_output_files(problem, plan_path, import_path)

    def test_output_validation_rejects_guided_area_function_violation(self) -> None:
        group = ExportGroup(**{**make_group("20").__dict__, "demand": 1})
        bay = make_bay("B", "01", {"1": 1})
        problem = ProblemData(
            export_groups=[group],
            bays={bay.bay_key: bay},
            area_guidance_target={("V1", "OF", "B", "20"): 1},
            area_functions={"B": {"IF"}},
            target_voyages=["V1"],
            export_voyages={"V1"},
        )
        with TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.csv"
            import_path = Path(directory) / "import.csv"
            with plan_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "group_id", "planned_boxes", "area_no", "bay_key", "bay_no",
                    "row_no", "row_allocation", "flow", "size", "height",
                    "voyage_id", "port",
                ])
                writer.writeheader()
                writer.writerow({
                    "group_id": group.group_id,
                    "planned_boxes": 1,
                    "area_no": "B",
                    "bay_key": "B|01",
                    "bay_no": "01",
                    "row_no": "1",
                    "row_allocation": "B|01:1:1",
                    "flow": "OF",
                    "size": "20",
                    "height": "96",
                    "voyage_id": "V1",
                    "port": "P1",
                })
            import_path.touch()
            with self.assertRaisesRegex(ValueError, "area-function"):
                validate_output_files(problem, plan_path, import_path)

    def test_output_validation_rejects_row_footprint_tampering(self) -> None:
        group = make_group("20")
        bay = Bay(
            area_no="A", bay_no="01", bay_key="A|01",
            bay_order=0, cap_by_size={"20": 5}, physical_capacity=5,
            row_cap_by_size={"20": {"1": 5}}, row_physical_capacity={"1": 5},
        )
        problem = SimpleNamespace(
            export_groups=[group], bays={"A|01": bay}, export_voyages={"V1"},
            import_area_size_reference={}, area_functions={"A": {"OF"}},
            area_guidance_target={}, attribute_rules=AttributeRules(),
        )
        with TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.csv"
            import_path = Path(directory) / "import.csv"
            with plan_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "group_id", "planned_boxes", "area_no", "bay_key", "bay_no", "row_no",
                    "row_allocation", "flow", "size", "height", "voyage_id", "port",
                ])
                writer.writeheader()
                writer.writerow({
                    "group_id": group.group_id, "planned_boxes": 5, "area_no": "A",
                    "bay_key": "A|01", "bay_no": "01", "row_no": "1",
                    "row_allocation": "A|99:1:1", "flow": "OF", "size": "20",
                    "height": "96", "voyage_id": "V1", "port": "P1",
                })
            import_path.touch()
            with self.assertRaisesRegex(ValueError, "row footprint mismatch"):
                validate_output_files(problem, plan_path, import_path)

    def test_output_validation_checks_configured_row_attribute_mix(self) -> None:
        group_a = make_group("20")
        group_a = ExportGroup(**{**group_a.__dict__, "group_id": "G-A", "demand": 1, "attributes": {"CUSTOM": "A"}})
        group_b = ExportGroup(**{**group_a.__dict__, "group_id": "G-B", "attributes": {"CUSTOM": "B"}})
        bay = Bay(
            area_no="A", bay_no="01", bay_key="A|01",
            bay_order=0, cap_by_size={"20": 2}, physical_capacity=2,
            row_cap_by_size={"20": {"1": 2}}, row_physical_capacity={"1": 2},
        )
        problem = SimpleNamespace(
            export_groups=[group_a, group_b], bays={"A|01": bay}, export_voyages={"V1"},
            import_area_size_reference={}, area_functions={"A": {"OF"}},
            area_guidance_target={},
            attribute_rules=AttributeRules(row_no_mix_attributes=("CUSTOM",)),
        )
        with TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.csv"
            import_path = Path(directory) / "import.csv"
            with plan_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "group_id", "planned_boxes", "area_no", "bay_key", "bay_no", "row_no",
                    "row_allocation", "flow", "size", "height", "voyage_id", "port", "CUSTOM",
                ])
                writer.writeheader()
                for group in (group_a, group_b):
                    writer.writerow({
                        "group_id": group.group_id, "planned_boxes": 1, "area_no": "A",
                        "bay_key": "A|01", "bay_no": "01", "row_no": "1",
                        "row_allocation": "A|01:1:1", "flow": "OF", "size": "20",
                        "height": "96", "voyage_id": "V1", "port": "P1",
                        "CUSTOM": group.attributes["CUSTOM"],
                    })
            import_path.touch()
            with self.assertRaisesRegex(ValueError, "row attribute mixing"):
                validate_output_files(problem, plan_path, import_path)

    def test_import_reservation_does_not_inherit_existing_size_no_mix(self) -> None:
        bay = Bay(
            area_no="A", bay_no="01", bay_key="A|01",
            bay_order=0, cap_by_size={"20": 4, "40": 4}, physical_capacity=4,
            row_cap_by_size={"20": {"1": 4}}, row_physical_capacity={"1": 4},
            existing_size_modes={"40"},
        )
        planner = YardPlanningBase.__new__(YardPlanningBase)
        planner.bays = {"A|01": bay}
        self.assertEqual(4, planner._import_reservation_capacity("A|01", "20"))
        problem = SimpleNamespace(
            export_groups=[], bays={"A|01": bay},
            import_area_size_reference={("IF", "A", "20"): 1},
            area_functions={"A": {"IF"}},
        )
        with TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.csv"
            import_path = Path(directory) / "import.csv"
            plan_path.touch()
            with import_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "flow", "size", "area_no", "bay_key", "bay_no", "reserved_boxes",
                ])
                writer.writeheader()
                writer.writerow({
                    "flow": "IF", "size": "20", "area_no": "A",
                    "bay_key": "A|01", "bay_no": "01", "reserved_boxes": 1,
                })
            result = validate_output_files(problem, plan_path, import_path)
            self.assertTrue(result["passed"])

    def test_import_reservation_requires_area_function(self) -> None:
        bay = Bay(
            area_no="A", bay_no="01", bay_key="A|01",
            bay_order=0, cap_by_size={"20": 4}, physical_capacity=4,
        )
        problem = SimpleNamespace(
            export_groups=[], bays={"A|01": bay},
            import_area_size_reference={("IF", "B", "20"): 1},
            area_functions={"A": {"OF"}},
        )
        with TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.csv"
            import_path = Path(directory) / "import.csv"
            plan_path.touch()
            with import_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "flow", "size", "area_no", "bay_key", "bay_no", "reserved_boxes",
                ])
                writer.writeheader()
                writer.writerow({
                    "flow": "IF", "size": "20", "area_no": "A",
                    "bay_key": "A|01", "bay_no": "01", "reserved_boxes": 1,
                })
            with self.assertRaisesRegex(ValueError, "area-function"):
                validate_output_files(problem, plan_path, import_path)


if __name__ == "__main__":
    unittest.main()
