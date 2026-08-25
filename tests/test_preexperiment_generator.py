from __future__ import annotations

import unittest

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd
from preexperiment import ScenarioSpec, materialize_scenario


def document(voyage: str, rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "IYC_CNTRID": f"{voyage}_{index}",
                "IYC_CNTRNO": f"{voyage}_{index}",
                "IYC_EVOY_ID": voyage,
                "IYC_IVOY_ID": "",
                "IYC_CSZ_CSIZECD": size,
                "IYC_STS_CSTATUSCD": "OF",
                "IYC_CHEIGHTCD": height,
                "IYC_POT_UNLDPORT": port,
                "IYC_EVESVOYAGE": voyage,
                "REF_BILLNO": voyage,
            }
            for index, (size, height, port) in enumerate(rows)
        ]
    )


def make_base() -> InputAdapterGd:
    base = InputAdapterGd()
    base.take_over_vessel = {"E": ["V1", "V2"], "I": ["I1"]}
    base.bay_slots_detail = pd.DataFrame(
        [
            {
                "HAS_CONTAINER": 1,
                "IYC_CNTRID": "SNAP1",
                "IYC_EVOY_ID": "V1",
                "IYC_IVOY_ID": "",
                "IYC_CSZ_CSIZECD": "20",
            },
            {
                "HAS_CONTAINER": 0,
                "IYC_CNTRID": "",
                "IYC_EVOY_ID": "",
                "IYC_IVOY_ID": "",
                "IYC_CSZ_CSIZECD": "",
            },
        ]
    )
    base.area_function_info = pd.DataFrame([{"area_no": "A", "cntr_type": "OF"}])
    base.vessel_berth_info = pd.DataFrame(
        [
            {
                "VOY_ID": "V1",
                "VOY_IEFG": "E",
                "VBT_BTH_PBTHNO": "1",
            },
            {
                "VOY_ID": "V2",
                "VOY_IEFG": "E",
                "VBT_BTH_PBTHNO": "1",
            },
        ]
    )
    base.berth_area_dist_matrix = pd.DataFrame([{"area_no": "A", "B1": 1}])
    base.planning_time = pd.Timestamp("2026-01-01")
    base.vessel_containers = {
        "V1": {
            "type": "E",
            "doc_cntrs": document("V1", [("20", "H1", "P1"), ("40", "H2", "P2")]),
            "predict_cntrs": {"20": {"total_volume": 100}},
        },
        "V2": {
            "type": "E",
            "doc_cntrs": document("V2", [("20", "H1", "P2"), ("40", "H2", "P1")]),
        },
        "I1": {
            "type": "I",
            "doc_cntrs": document("I1", [("20", "H1", "P1")] * 4).assign(IYC_EVOY_ID="", IYC_IVOY_ID="I1"),
        },
    }
    return base


class PreexperimentGeneratorTests(unittest.TestCase):
    def test_same_seed_materializes_identical_case(self) -> None:
        spec = ScenarioSpec("case_1", 7, 2, 4, 20, 0.5)

        first, first_manifest = materialize_scenario(make_base(), spec)
        second, second_manifest = materialize_scenario(make_base(), spec)

        self.assertEqual(first_manifest, second_manifest)
        for voyage in ("V1", "V2"):
            pd.testing.assert_frame_equal(
                first.vessel_containers[voyage]["doc_cntrs"],
                second.vessel_containers[voyage]["doc_cntrs"],
            )

    def test_manifest_contains_known_boxes_only(self) -> None:
        spec = ScenarioSpec("case_2", 11, 1, 4, 20, 0.5)
        base = make_base()
        generated, manifest = materialize_scenario(base, spec)
        self.assertEqual("integrated_zone_v4", manifest["model_schema_version"])
        self.assertEqual(20, manifest["declared_export_rows"])
        self.assertEqual(4, manifest["export_group_count"])
        self.assertEqual(2, manifest["declared_import_rows"])
        self.assertEqual("not_generated_or_used", manifest["prediction_policy"])
        self.assertNotIn("predict_cntrs", generated.vessel_containers["V1"])
        self.assertFalse(hasattr(generated, "tops_plan"))

    def test_invalid_case_dimensions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ScenarioSpec("bad case", 1, 1, 4, 3, 0.5).validate()

    def test_legacy_forecast_recipe_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain forecast"):
            ScenarioSpec.from_dict(
                {
                    "case_id": "legacy_forecast",
                    "seed": 1,
                    "export_voyage_count": 1,
                    "groups_per_voyage": 4,
                    "boxes_per_voyage": 20,
                    "import_fraction": 0.5,
                    "forecast_extra_ratio": 0.2,
                }
            )

    def test_synthetic_voyages_are_deterministic_and_heterogeneous(self) -> None:
        spec = ScenarioSpec(
            "scale_4v",
            19,
            4,
            4,
            20,
            0.50,
            voyage_volume_spread=0.20,
        )

        base = make_base()
        generated, manifest = materialize_scenario(base, spec)

        self.assertEqual(
            ["V1", "V2", "V1C1", "V2C1"],
            generated.take_over_vessel["E"],
        )
        profiles = manifest["voyage_profiles"]
        self.assertEqual(80, sum(item["declared_rows"] for item in profiles.values()))
        self.assertGreater(
            len({item["declared_rows"] for item in profiles.values()}),
            1,
        )
        self.assertGreater(len({str(item["group_profile"]) for item in profiles.values()}), 1)
        self.assertEqual(2, sum(item["synthetic"] for item in profiles.values()))
        self.assertTrue(
            all(
                "predict_cntrs" not in generated.vessel_containers[voyage]
                for voyage in generated.take_over_vessel["E"]
            )
        )
        self.assertIs(generated.bay_slots_detail, base.bay_slots_detail)


if __name__ == "__main__":
    unittest.main()
