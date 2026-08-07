from __future__ import annotations

import unittest
import csv
import importlib.util
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from yard_planning.models import AttributeRules, Bay, ExportGroup, ProblemData
from yard_planning.direct_milp import DirectMilpPlanner
from yard_planning.logic_benders import (
    LogicBendersConfig,
    LogicBendersPlanner,
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
                LogicBendersConfig(support_repair_fraction=1.1),
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
                area_time_limit=3.0,
                primal_seed_time_limit=0.0,
            ),
        ).solve()
        self.assertTrue(decomposed.diagnostics["lbbd_converged"])
        self.assertTrue(
            decomposed.diagnostics["independent_solution_validation"]["passed"]
        )
        self.assertGreater(
            decomposed.diagnostics["lbbd_cut_counts"][
                "initial_conflict_clique"
            ],
            0,
        )
        self.assertAlmostEqual(
            direct.diagnostics["final_business_objective"],
            decomposed.diagnostics["final_business_objective"],
            places=9,
        )

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_logic_benders_removes_temporary_repair_neighborhood(self) -> None:
        config = ColumnGenerationConfig(
            total_time_limit=20.0,
            mip_gap=0.0,
            solver_threads=1,
            verbose=False,
        )
        planner = LogicBendersPlanner(
            make_small_problem(),
            config,
            LogicBendersConfig(
                support_repair_iterations=2,
                support_repair_fraction=0.02,
            ),
        )
        planner._prepare_lbbd()
        selected, _stats = planner._solve_direct_milp()
        master, variables, _master_stats = planner._build_master()
        try:
            original_variable_count = len(master.getVars())
            export_by_area, import_by_area = planner._apply_master_start(
                variables,
                selected,
                planner._final_import_reservation,
            )
            neighborhood = planner._add_support_repair_neighborhood(
                master,
                variables,
                export_by_area,
                import_by_area,
            )
            self.assertIsNotNone(neighborhood)
            self.assertEqual(0.0, float(neighborhood.limit.RHS))
            self.assertGreater(len(master.getVars()), original_variable_count)
            planner._remove_support_repair_neighborhood(
                master, neighborhood
            )
            self.assertEqual(original_variable_count, len(master.getVars()))
        finally:
            planner._free_gurobi_model(master)

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
                area_time_limit=3.0,
                primal_seed_time_limit=0.0,
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
