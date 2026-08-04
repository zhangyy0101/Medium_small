from __future__ import annotations

import unittest
import csv
import importlib.util
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from yard_planning.models import AttributeRules, Bay, ExportGroup, ProblemData
from yard_planning.planner import (
    ColumnGenerationConfig,
    ColumnGenerationPlanner,
    PlacementColumn,
    _GurobiModelAdapter,
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
    def test_pricing_configuration_requires_negative_reduced_cost(self) -> None:
        config = ColumnGenerationConfig()
        self.assertGreater(config.min_columns_per_group_per_iteration, 0)
        self.assertGreater(config.max_columns_per_group_per_iteration, 0)
        self.assertGreaterEqual(
            config.max_columns_per_group_per_iteration,
            config.min_columns_per_group_per_iteration,
        )
        self.assertGreater(config.adaptive_pricing_fraction, 0.0)
        self.assertGreater(config.reduced_cost_tolerance, 0.0)

    def test_adaptive_pricing_batch_respects_fraction_and_bounds(self) -> None:
        planner = ColumnGenerationPlanner.__new__(ColumnGenerationPlanner)
        planner.config = ColumnGenerationConfig(
            min_columns_per_group_per_iteration=2,
            max_columns_per_group_per_iteration=8,
            adaptive_pricing_fraction=0.25,
        )
        self.assertEqual(0, planner._adaptive_pricing_batch_size(0))
        self.assertEqual(1, planner._adaptive_pricing_batch_size(1))
        self.assertEqual(2, planner._adaptive_pricing_batch_size(4))
        self.assertEqual(5, planner._adaptive_pricing_batch_size(20))
        self.assertEqual(8, planner._adaptive_pricing_batch_size(100))

    def test_stage_two_has_no_unplaced_penalty(self) -> None:
        planner = ColumnGenerationPlanner.__new__(ColumnGenerationPlanner)
        group = make_group("20")
        self.assertEqual(1.0, planner._unplaced_objective_for_group(group, "min_unplaced"))
        self.assertEqual(0.0, planner._unplaced_objective_for_group(group, "full"))

    def test_link_bound_uses_demand_capacity_minimum(self) -> None:
        self.assertEqual(4, ColumnGenerationPlanner._tight_link_bound(100, 4))
        self.assertEqual(3, ColumnGenerationPlanner._tight_link_bound(3, 20))
        with self.assertRaises(ValueError):
            ColumnGenerationPlanner._tight_link_bound(0, 20)

    def test_output_quantities_must_be_exact_integers(self) -> None:
        errors: list[str] = []
        self.assertIsNone(_parse_integer("1.5", "planned_boxes", errors))
        self.assertIn("non-integer field", errors[0])

    def test_normalized_policy_weights_sum_to_one_and_follow_priority(self) -> None:
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
        planner = ColumnGenerationPlanner.__new__(ColumnGenerationPlanner)
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
        planner = ColumnGenerationPlanner(problem, ColumnGenerationConfig(verbose=False))
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
        planner = ColumnGenerationPlanner(problem, ColumnGenerationConfig(verbose=False))
        self.assertNotIn(group.group_id, planner.reachable_anchor_group_ids)
        self.assertEqual(0, planner._anchored_group_demand())
        self.assertEqual(0.0, planner._normalized_existing_proximity(group, "A|01"))

    def test_group_area_big_m_uses_feasible_group_capacity(self) -> None:
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
        planner = ColumnGenerationPlanner(problem, ColumnGenerationConfig(verbose=False))
        planner._prepare_master_index_sets()
        key = (planner._operational_group_key(group), "A")
        self.assertEqual(3, planner._master_group_area_big_m[key])

    def test_pricing_enumerates_candidates_without_materializing_universe(self) -> None:
        planner = ColumnGenerationPlanner.__new__(ColumnGenerationPlanner)
        planner.groups = [make_group("20")]
        planner.config = ColumnGenerationConfig()
        planner.bays = {
            "A|01": Bay(
                area_no="A", bay_no="01", bay_key="A|01",
                bay_order=0, cap_by_size={"20": 9}, physical_capacity=9,
            )
        }
        planner._columns = []
        planner._column_keys = set()
        planner._column_indices_by_triplet = defaultdict(list)
        planner._candidate_bays_for_group = lambda group: [("A|01", 9, 0.0)]
        planner._placement_footprint_keys = lambda bay_key, size: (bay_key,)
        planner._row_capacity_items_for_group = lambda bay_key, size, group: [(str(i), 3) for i in range(1, 10)]
        planner._quota_key = lambda group, area: (group.voyage_id, group.status, area, group.size)
        planner._operational_group_key = lambda group: (group.voyage_id, group.status, group.port, group.size, group.height)
        planner._column_base_cost = lambda group, bay: 0.0
        planner._berth_distance_cost = lambda voyage, area, qty: 0.0

        candidates = list(planner._iter_feasible_unit_flow_columns(planner.groups[0]))

        self.assertEqual(9, len(candidates))
        self.assertTrue(all(column.quantity == 1 for column in candidates))
        self.assertEqual([], planner._columns)

        index = planner._append_generated_column(candidates[0])
        self.assertEqual(0, index)
        self.assertEqual(1, len(planner._columns))
        self.assertEqual("C0000001", planner._columns[0].column_id)

    def test_pricing_reduced_cost_uses_master_row_duals(self) -> None:
        planner = ColumnGenerationPlanner.__new__(ColumnGenerationPlanner)
        planner._placement_footprint_keys = lambda bay_key, size: (bay_key,)
        planner._row_mix_key_for_column = lambda column: "mix"
        planner._bay_no_mix_attrs_for_column = lambda column: ("height",)
        planner._row_no_mix_attrs_for_column = lambda column: ("voyage",)
        planner._attr_voyage_scope = lambda attr, voyage: "scope"
        planner._column_attr_value = lambda column, attr: f"{attr}-value"
        column = PlacementColumn(
            column_id="",
            group_id="G1",
            voyage_id="V1",
            flow="OF",
            port="P1",
            size="20",
            big_plan_size="20",
            height="96",
            attributes={},
            area_no="A",
            bay_key="B1",
            bay_no="01",
            quantity=1,
            stack_units=1,
            row_allocation=(("B1", "R1", 1),),
            quota_key=("V1", "OF", "A", "20"),
            group_key=("V1", "OF", "P1", "20", "96"),
            intrinsic_cost=100.0,
        )
        constraints = {
            "group_cover": {"G1": 2.0},
            "bay_capacity_limit": {"B1": 3.0},
            "bay_port_stack_link": {("B1", "mix", "20"): 5.0},
            "bay_attr_link": {("B1", "height", "scope", "height-value"): 7.0},
            "bay_size_limit": {("B1", "20"): 11.0},
            "row_capacity_limit": {("B1", "R1"): 13.0},
            "row_size_limit": {("B1", "R1", "20"): 17.0},
            "row_attr_link": {("B1", "R1", "voyage", "scope", "voyage-value"): 19.0},
            "area_guidance_balance": {("V1", "OF", "A", "20"): 23.0},
            "fixed_use_objective_limit": {
                ("group_area", column.group_key, "A"): 29.0,
                ("group_row", column.group_key, "B1", "R1"): 31.0,
                ("group_used_upper", *column.group_key): 37.0,
                ("group_used_lower", *column.group_key): 41.0,
            },
        }
        model = SimpleNamespace(getDualsolLinear=lambda constraint: constraint)

        reduced_cost = planner._column_reduced_cost(model, constraints, column, "full")

        self.assertAlmostEqual(-56.0, reduced_cost)

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_column_generation_lp_matches_complete_column_lp(self) -> None:
        from gurobipy import quicksum

        problem = make_small_problem()
        config = ColumnGenerationConfig(
            min_columns_per_group_per_iteration=1,
            max_columns_per_group_per_iteration=1,
            adaptive_pricing_fraction=1.0,
            max_iterations=30,
            mip_gap=0.0,
            complete_integer_verification_max_columns=0,
            verbose=False,
        )
        priced = ColumnGenerationPlanner(problem, config)
        priced_result = priced.solve()

        complete = ColumnGenerationPlanner(problem, config)
        complete._initialize_column_generation()
        complete._prepare_master_index_sets()
        complete._prepare_objective_normalization()
        complete._materialize_complete_unit_flow_universe()
        phase1, phase1_vars, _ = complete._build_restricted_master(
            _GurobiModelAdapter,
            quicksum,
            relax=True,
            objective_mode="min_unplaced",
        )
        try:
            phase1.optimize()
            self.assertEqual("optimal", complete._gurobi_status_name(phase1))
            phase1_unplaced = sum(
                complete._gurobi_value(phase1, var)
                for var in phase1_vars["unplaced"].values()
            )
        finally:
            complete._free_gurobi_model(phase1)

        phase2, _phase2_vars, _ = complete._build_restricted_master(
            _GurobiModelAdapter,
            quicksum,
            relax=True,
            objective_mode="full",
            fixed_unplaced_total=phase1_unplaced,
        )
        try:
            phase2.optimize()
            self.assertEqual("optimal", complete._gurobi_status_name(phase2))
            complete_objective = complete._gurobi_objective_value(phase2)
        finally:
            complete._free_gurobi_model(phase2)

        diagnostics = priced_result.diagnostics
        self.assertAlmostEqual(
            phase1_unplaced,
            diagnostics["pricing_phase1_lp_unplaced_boxes"],
            places=8,
        )
        self.assertAlmostEqual(
            complete_objective,
            diagnostics["pricing_phase2_lp_objective"],
            places=8,
        )
        self.assertEqual(
            "two_phase_reduced_cost_convergence",
            diagnostics["pricing_stop_reason"],
        )

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is unavailable")
    def test_small_case_complete_integer_verification_reports_full_scope(self) -> None:
        planner = ColumnGenerationPlanner(
            make_small_problem(),
            ColumnGenerationConfig(
                complete_integer_verification_max_columns=100,
                mip_gap=0.0,
                verbose=False,
            ),
        )
        diagnostics = planner.solve().diagnostics
        self.assertTrue(diagnostics["complete_integer_verification"]["performed"])
        self.assertEqual(
            diagnostics["potential_unit_flow_count"],
            diagnostics["complete_integer_verification"]["complete_column_count"],
        )
        self.assertEqual("complete_unit_flow_universe", diagnostics["master_bound_scope"])
        self.assertEqual(
            "complete_integer_master",
            diagnostics["complete_model_certified_gap_source"],
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
            unplaced_path = Path(directory) / "unplaced.csv"
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
            unplaced_path.touch()
            import_path.touch()
            with self.assertRaisesRegex(ValueError, "capacity"):
                validate_output_files(problem, plan_path, unplaced_path, import_path)

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
            unplaced_path = Path(directory) / "unplaced.csv"
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
            unplaced_path.touch()
            import_path.touch()
            with self.assertRaisesRegex(ValueError, "group identity mismatch"):
                validate_output_files(problem, plan_path, unplaced_path, import_path)

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
            unplaced_path = Path(directory) / "unplaced.csv"
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
            unplaced_path.touch()
            import_path.touch()
            with self.assertRaisesRegex(ValueError, "area-function"):
                validate_output_files(problem, plan_path, unplaced_path, import_path)

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
            unplaced_path = Path(directory) / "unplaced.csv"
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
            unplaced_path.touch()
            import_path.touch()
            with self.assertRaisesRegex(ValueError, "row footprint mismatch"):
                validate_output_files(problem, plan_path, unplaced_path, import_path)

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
            unplaced_path = Path(directory) / "unplaced.csv"
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
            unplaced_path.touch()
            import_path.touch()
            with self.assertRaisesRegex(ValueError, "row attribute mixing"):
                validate_output_files(problem, plan_path, unplaced_path, import_path)

    def test_import_reservation_does_not_inherit_existing_size_no_mix(self) -> None:
        bay = Bay(
            area_no="A", bay_no="01", bay_key="A|01",
            bay_order=0, cap_by_size={"20": 4, "40": 4}, physical_capacity=4,
            row_cap_by_size={"20": {"1": 4}}, row_physical_capacity={"1": 4},
            existing_size_modes={"40"},
        )
        planner = ColumnGenerationPlanner.__new__(ColumnGenerationPlanner)
        planner.bays = {"A|01": bay}
        self.assertEqual(4, planner._import_reservation_capacity("A|01", "20"))
        problem = SimpleNamespace(
            export_groups=[], bays={"A|01": bay},
            import_area_size_reference={("IF", "A", "20"): 1},
            area_functions={"A": {"IF"}},
        )
        with TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.csv"
            unplaced_path = Path(directory) / "unplaced.csv"
            import_path = Path(directory) / "import.csv"
            plan_path.touch()
            unplaced_path.touch()
            with import_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "flow", "size", "area_no", "bay_key", "bay_no", "reserved_boxes",
                ])
                writer.writeheader()
                writer.writerow({
                    "flow": "IF", "size": "20", "area_no": "A",
                    "bay_key": "A|01", "bay_no": "01", "reserved_boxes": 1,
                })
            result = validate_output_files(problem, plan_path, unplaced_path, import_path)
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
            unplaced_path = Path(directory) / "unplaced.csv"
            import_path = Path(directory) / "import.csv"
            plan_path.touch()
            unplaced_path.touch()
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
                validate_output_files(problem, plan_path, unplaced_path, import_path)


if __name__ == "__main__":
    unittest.main()
