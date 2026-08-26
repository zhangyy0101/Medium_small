from __future__ import annotations

import unittest
from datetime import datetime

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd
from adapters.planning_input import (
    active_large_container_shadow_slots,
    calculate_declared_import_demand,
    existing_bay_attributes,
    existing_large_pair_members_by_bay,
    existing_operational_group_loads,
    normalize_code,
)
from yard_planning.models import AttributeRules, EXISTING_EXPORT_GROUP_SCOPE


class PlanningInputPreprocessingTests(unittest.TestCase):
    def test_normalize_code_fast_path_preserves_numeric_codes(self) -> None:
        self.assertEqual("12", normalize_code(" 12.0 "))
        self.assertEqual("AB.0", normalize_code("ab.0"))
        self.assertEqual("", normalize_code(None))
        self.assertEqual("", normalize_code(float("nan")))

    def test_large_pair_index_and_shadow_slots(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "HAS_CONTAINER": 1,
                    "YAA_AREANO": "A",
                    "YBY_BAYNO": "01",
                    "YST_ROWNO": "1",
                    "YST_TIERNO": "1",
                    "YST_SLOTNO": "1",
                    "IYC_CSZ_CSIZECD": "40",
                    "IYC_CNTRID": "C1",
                    "YBY_ENABLECSIZECD": "40",
                },
                {
                    "HAS_CONTAINER": 1,
                    "YAA_AREANO": "A",
                    "YBY_BAYNO": "03",
                    "YST_ROWNO": "1",
                    "YST_TIERNO": "1",
                    "YST_SLOTNO": "1",
                    "IYC_CSZ_CSIZECD": "40",
                    "IYC_CNTRID": "C1",
                    "YBY_ENABLECSIZECD": "40",
                },
                {
                    "HAS_CONTAINER": 0,
                    "YAA_AREANO": "A",
                    "YBY_BAYNO": "05",
                    "YST_ROWNO": "1",
                    "YST_TIERNO": "1",
                    "YST_SLOTNO": "1",
                    "IYC_CSZ_CSIZECD": None,
                    "IYC_CNTRID": None,
                    "YBY_ENABLECSIZECD": "40",
                },
            ]
        )

        pairs = existing_large_pair_members_by_bay(frame)
        expected_pair = frozenset({"01", "03"})
        self.assertEqual({expected_pair}, pairs[("A", "01")])
        self.assertEqual({expected_pair}, pairs[("A", "03")])

        shadows = active_large_container_shadow_slots(
            frame,
            {("A", "01"): "03", ("A", "03"): "01"},
            pairs,
        )
        self.assertIn(("A", "03", "01", "01", "01"), shadows)
        self.assertIn(("A", "01", "01", "01", "01"), shadows)

    def test_existing_attributes_propagate_across_large_pair(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "HAS_CONTAINER": 1,
                    "YAA_AREANO": "A",
                    "YBY_BAYNO": "01",
                    "YST_ROWNO": "1",
                    "IYC_CSZ_CSIZECD": "40",
                    "IYC_CHEIGHTCD": "96",
                    "IYC_POT_UNLDPORT": "P1",
                    "IYC_EVOY_ID": "V1",
                    "IYC_IVOY_ID": "",
                }
            ]
        )
        attrs = existing_bay_attributes(
            frame,
            AttributeRules(),
            large_bay_partner_by_bay={("A", "01"): "03"},
        )

        for key in (("A", "01"), ("A", "03")):
            self.assertEqual({"40"}, attrs[key]["sizes"])
            self.assertEqual({"96"}, attrs[key]["heights"])
            self.assertEqual({"P1"}, attrs[key]["ports_by_row"]["01"])
            self.assertEqual(
                {
                    (
                        EXISTING_EXPORT_GROUP_SCOPE,
                        "V1",
                        "40",
                        "96",
                        "P1",
                    )
                },
                attrs[key]["group_keys_by_row"]["01"],
            )

    def test_operational_group_anchor_scan_deduplicates_container_slots(self) -> None:
        adapter = InputAdapterGd()
        adapter.take_over_vessel = {"E": ["V1"]}
        adapter.vessel_berth_info = pd.DataFrame(columns=["VOY_ID", "VOY_IEFG"])
        adapter.bay_slots_detail = pd.DataFrame(
            [
                {
                    "HAS_CONTAINER": 1,
                    "YAA_AREANO": "A",
                    "YBY_BAYNO": "01",
                    "IYC_CNTRNO": "C1",
                    "IYC_CNTRID": "1",
                    "IYC_EVOY_ID": "V1",
                    "IYC_IVOY_ID": "",
                    "IYC_STS_CSTATUSCD": "OF",
                    "IYC_CSZ_CSIZECD": "20",
                    "IYC_POT_UNLDPORT": "P1",
                    "IYC_CHEIGHTCD": "96",
                    "IYC_INYTM": "2025-01-01 00:00:00",
                },
                {
                    "HAS_CONTAINER": 1,
                    "YAA_AREANO": "A",
                    "YBY_BAYNO": "01",
                    "IYC_CNTRNO": "C1",
                    "IYC_CNTRID": "1",
                    "IYC_EVOY_ID": "V1",
                    "IYC_IVOY_ID": "",
                    "IYC_STS_CSTATUSCD": "OF",
                    "IYC_CSZ_CSIZECD": "20",
                    "IYC_POT_UNLDPORT": "P1",
                    "IYC_CHEIGHTCD": "96",
                    "IYC_INYTM": "2025-01-01 00:00:00",
                },
            ]
        )
        area_load, bay_load = existing_operational_group_loads(
            adapter,
            datetime(2025, 1, 2),
            {"V1"},
            {"A|01"},
            AttributeRules(),
        )
        group_key = (
            "V1",
            "flow=OF",
            "IYC_CSZ_CSIZECD=20",
            "IYC_POT_UNLDPORT=P1",
            "IYC_CHEIGHTCD=96",
        )
        self.assertEqual(1, area_load[group_key + ("A",)])
        self.assertEqual(1, bay_load[group_key + ("A", "A|01")])

    def test_declared_import_demand_is_anonymous_and_excludes_in_yard_ids(self) -> None:
        adapter = InputAdapterGd()
        adapter.take_over_vessel = {"I": ["I1"]}
        adapter.vessel_berth_info = pd.DataFrame(
            [{"VOY_ID": "I1", "VOY_IEFG": "I"}]
        )
        adapter.bay_slots_detail = pd.DataFrame(
            [
                {
                    "HAS_CONTAINER": 1,
                    "IYC_CNTRID": "C1",
                }
            ]
        )
        adapter.vessel_containers = {
            "I1": {
                "type": "I",
                "doc_cntrs": pd.DataFrame(
                    [
                        {
                            "IYC_CNTRID": "C1",
                            "IYC_IVOY_ID": "I1",
                            "IYC_CSZ_CSIZECD": "20",
                            "IYC_STS_CSTATUSCD": "IF",
                        },
                        {
                            "IYC_CNTRID": "C2",
                            "IYC_IVOY_ID": "I1",
                            "IYC_CSZ_CSIZECD": "20",
                            "IYC_STS_CSTATUSCD": "IF",
                        },
                        {
                            "IYC_CNTRID": "C3",
                            "IYC_IVOY_ID": "I1",
                            "IYC_CSZ_CSIZECD": "45",
                            "IYC_STS_CSTATUSCD": "IZ",
                        },
                        {
                            "IYC_CNTRID": "C3",
                            "IYC_IVOY_ID": "I1",
                            "IYC_CSZ_CSIZECD": "45",
                            "IYC_STS_CSTATUSCD": "IZ",
                        },
                    ]
                ),
            }
        }

        demand = calculate_declared_import_demand(adapter)

        self.assertEqual({("IF", "20"): 1, ("IZ", "40"): 1}, dict(demand))


if __name__ == "__main__":
    unittest.main()
