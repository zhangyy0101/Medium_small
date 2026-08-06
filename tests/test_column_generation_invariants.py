from __future__ import annotations

import unittest
import csv
import importlib.util
from time import perf_counter
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from yard_planning.models import AttributeRules, Bay, ExportGroup, ProblemData
from yard_planning.direct_milp import DirectMilpPlanner
from yard_planning.area_configuration import (
    AdaptiveAreaPricingConfig,
    AreaConfigurationPlanner,
)
from yard_planning.area_branch_price import (
    AreaConfigurationBranchPricePlanner,
)
from yard_planning.planner import (
    BranchDecision,
    BranchPriceNode,
    ColumnGenerationConfig,
    YardPlanningBase,
)
from yard_planning.output_validator import _parse_integer, validate_output_files


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


class ColumnGenerationInvariantTests(unittest.TestCase):
    def test_adaptive_area_pricing_configuration_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            AreaConfigurationPlanner(
                make_small_problem(),
                ColumnGenerationConfig(verbose=False),
                AdaptiveAreaPricingConfig(direct_candidate_limit=0),
            )
        with self.assertRaises(ValueError):
            AreaConfigurationPlanner(
                make_small_problem(),
                ColumnGenerationConfig(verbose=False),
                AdaptiveAreaPricingConfig(nested_max_iterations=0),
            )
        with self.assertRaises(ValueError):
            AreaConfigurationPlanner(
                make_small_problem(),
                ColumnGenerationConfig(verbose=False),
                AdaptiveAreaPricingConfig(nested_time_fraction=1.0),
            )

    def test_area_pricing_strategy_uses_structure_not_area_name(self) -> None:
        planner = AreaConfigurationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(verbose=False),
            AdaptiveAreaPricingConfig(direct_candidate_limit=1),
        )
        planner._prepare_master_index_sets()
        planner._prepare_objective_normalization()
        profiles = {
            area_no: planner._area_pricing_profile(area_no)
            for area_no in planner._configuration_areas()
        }
        self.assertEqual({"A", "B"}, set(profiles))
        self.assertTrue(
            all(
                profile.strategy == "adaptive_block_guided_multicolumn"
                for profile in profiles.values()
            )
        )
        self.assertTrue(
            all(len(profile.footprint_blocks) == 2 for profile in profiles.values())
        )

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy"), "gurobipy is unavailable"
    )
    def test_equivalent_block_sharing_stops_for_location_branching(
        self,
    ) -> None:
        planner = AreaConfigurationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(verbose=False),
            AdaptiveAreaPricingConfig(direct_candidate_limit=1),
        )
        planner._initialize_area_configuration_pool()
        try:
            state = planner._nested_area_pricing_state("A")
            self.assertEqual(((0, 1),), state.equivalent_block_classes)
            self.assertEqual(
                ((0, 1),),
                planner._nested_block_classes_for_decisions(state, ()),
            )
            row_branch = BranchDecision(
                "branch_row_use", ("G1", "A|01", "1"), "L", 0
            )
            self.assertEqual(
                ((0,), (1,)),
                planner._nested_block_classes_for_decisions(
                    state, (row_branch,)
                ),
            )
        finally:
            planner._dispose_area_pricing_models()

    def test_area_pricing_schedule_periodically_restores_full_sweep(
        self,
    ) -> None:
        planner = AreaConfigurationBranchPricePlanner(
            make_small_problem(),
            ColumnGenerationConfig(verbose=False),
            AdaptiveAreaPricingConfig(
                full_sweep_frequency=3,
            ),
        )
        areas = ("A", "B")
        self.assertEqual(
            ("A",), planner._scheduled_pricing_areas(areas, {"A"}, 2)
        )
        self.assertEqual(
            areas, planner._scheduled_pricing_areas(areas, {"A"}, 3)
        )

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy"), "gurobipy is unavailable"
    )
    def test_active_area_sweep_cannot_issue_exact_certificate(self) -> None:
        planner = AreaConfigurationBranchPricePlanner(
            make_small_problem(),
            ColumnGenerationConfig(
                total_time_limit=20.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
        )
        areas = planner._initialize_area_configuration_pool()
        model, _variables, constraints = planner._build_area_master(areas)
        try:
            model.optimize()
            duals = planner._master_dual_snapshot(model, constraints)
            partial, _new_indices = planner._price_all_areas(
                areas,
                duals,
                "phase_one",
                perf_counter() + 10.0,
                pricing_areas=(areas[0],),
            )
            self.assertFalse(partial["all_areas_priced"])
            self.assertFalse(partial["exact"])
            self.assertIsNone(partial["valid_lower_bound_correction"])
            certified, _certificate_indices = (
                planner._complete_targeted_area_certificate(
                    partial,
                    duals,
                    "phase_one",
                    perf_counter() + 10.0,
                )
            )
            self.assertTrue(certified["all_areas_priced"])
            self.assertTrue(
                certified["exact"] or bool(_certificate_indices)
            )
        finally:
            planner._free_gurobi_model(model)
            for pricing in planner._area_pricing_models.values():
                planner._free_gurobi_model(pricing.model)

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy"), "gurobipy is unavailable"
    )
    def test_adaptive_area_root_is_exact_on_small_case(self) -> None:
        result = AreaConfigurationBranchPricePlanner(
            make_small_problem(),
            ColumnGenerationConfig(
                max_iterations=30,
                total_time_limit=20.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
            AdaptiveAreaPricingConfig(
                direct_candidate_limit=1,
                complex_area_pool_size=4,
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
                profile["strategy"]
                == "adaptive_block_guided_multicolumn"
                for profile in result["adaptive_pricing"]["profiles"]
            )
        )
        strategies = {
            area_result["pricing_strategy"]
            for record in result["records"]
            for area_result in record["area_results"]
        }
        self.assertIn(
            "nested_exact_block_column_generation", strategies
        )

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy"), "gurobipy is unavailable"
    )
    def test_area_row_recombination_preserves_global_column_pool(self) -> None:
        planner = AreaConfigurationBranchPricePlanner(
            make_small_problem(),
            ColumnGenerationConfig(
                max_iterations=30,
                total_time_limit=20.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
        )
        root = planner.solve_root()
        self.assertTrue(root["root_exact"])
        columns_before = tuple(planner._columns)
        keys_before = set(planner._column_keys)
        heuristic = planner._solve_area_row_recombination(
            planner._configuration_areas(), 5.0
        )
        self.assertIsNotNone(heuristic)
        self.assertEqual(columns_before, tuple(planner._columns))
        self.assertEqual(keys_before, planner._column_keys)

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_area_branch_rows_are_enforced_after_pricing(self) -> None:
        planner = AreaConfigurationBranchPricePlanner(
            make_small_problem(),
            ColumnGenerationConfig(
                max_iterations=30,
                total_time_limit=20.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
        )
        areas = planner._initialize_area_configuration_pool()
        try:
            for sense, rhs in (("L", 0), ("G", 1)):
                decision = BranchDecision(
                    "branch_row_use", ("G1", "A|01", "1"), sense, rhs
                )
                result = planner._solve_area_node_lp(
                    BranchPriceNode(
                        node_id=1 if sense == "L" else 2,
                        depth=1,
                        decisions=(decision,),
                    ),
                    areas,
                    perf_counter() + 10.0,
                )
                self.assertEqual("optimal", result["status"])
                row_use, _row_quantity, _imports = (
                    planner._aggregate_area_original_values(
                        result["configuration_values"]
                    )
                )
                value = row_use.get(("G1", "A|01", "1"), 0.0)
                if sense == "L":
                    self.assertLessEqual(value, 1e-7)
                else:
                    self.assertGreaterEqual(value, 1.0 - 1e-7)
        finally:
            for pricing in planner._area_pricing_models.values():
                planner._free_gurobi_model(pricing.model)

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_group_area_quantity_branch_is_enforced_after_pricing(self) -> None:
        planner = AreaConfigurationBranchPricePlanner(
            make_small_problem(),
            ColumnGenerationConfig(
                max_iterations=30,
                total_time_limit=20.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
        )
        areas = planner._initialize_area_configuration_pool()
        try:
            for sense, rhs in (("L", 0), ("G", 1)):
                decision = BranchDecision(
                    "branch_group_area_quantity",
                    ("G1", "A"),
                    sense,
                    rhs,
                )
                result = planner._solve_area_node_lp(
                    BranchPriceNode(
                        node_id=3 if sense == "L" else 4,
                        depth=1,
                        decisions=(decision,),
                    ),
                    areas,
                    perf_counter() + 10.0,
                )
                self.assertEqual("optimal", result["status"])
                value = sum(
                    float(weight)
                    * int(
                        dict(
                            planner._area_configurations[index].group_quantities
                        ).get("G1", 0)
                    )
                    for index, weight in result[
                        "configuration_values"
                    ].items()
                    if planner._area_configurations[index].area_no == "A"
                )
                if sense == "L":
                    self.assertLessEqual(value, 1e-7)
                else:
                    self.assertGreaterEqual(value, 1.0 - 1e-7)
        finally:
            for pricing in planner._area_pricing_models.values():
                planner._free_gurobi_model(pricing.model)

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_area_branch_and_price_is_exact_on_small_case(self) -> None:
        result = AreaConfigurationBranchPricePlanner(
            make_small_problem(),
            ColumnGenerationConfig(
                max_iterations=30,
                total_time_limit=20.0,
                mip_gap=0.0,
                solver_threads=1,
                verbose=False,
            ),
        ).solve_branch_and_price()
        self.assertEqual("optimal", result["status"])
        self.assertAlmostEqual(0.132, result["objective"], places=9)
        self.assertAlmostEqual(
            result["objective"], result["global_lower_bound"], places=9
        )

    def test_exact_branch_and_price_configuration_is_positive(self) -> None:
        config = ColumnGenerationConfig()
        self.assertGreater(config.reduced_cost_tolerance, 0.0)
        self.assertGreater(config.max_iterations, 0)
        self.assertGreater(config.max_branch_nodes, 0)

    def test_branch_node_limit_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            YardPlanningBase(
                make_small_problem(),
                ColumnGenerationConfig(max_branch_nodes=0, verbose=False),
            )

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
    def test_direct_milp_matches_branch_and_price_on_small_case(self) -> None:
        config = ColumnGenerationConfig(
            max_iterations=30,
            total_time_limit=20.0,
            mip_gap=0.0,
            verbose=False,
        )
        direct = DirectMilpPlanner(make_small_problem(), config).solve()
        generated = AreaConfigurationBranchPricePlanner(
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
    def test_direct_and_branch_and_price_reject_infeasible_hard_demand(self) -> None:
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
        with self.assertRaisesRegex(RuntimeError, "feasible incumbent"):
            AreaConfigurationBranchPricePlanner(problem, config).solve()

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_small_case_finishes_exact_branch_and_price(self) -> None:
        diagnostics = AreaConfigurationBranchPricePlanner(
            make_small_problem(),
            ColumnGenerationConfig(mip_gap=0.0, verbose=False),
        ).solve().diagnostics
        self.assertEqual("optimal", diagnostics["master_status"])
        self.assertEqual(
            "complete_branch_and_price_tree",
            diagnostics["master_bound_scope"],
        )
        self.assertEqual(
            "exact_area_pricing_and_branch_tree",
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
