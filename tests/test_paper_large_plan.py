from __future__ import annotations

import unittest

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd
from paper_large_plan import allocation_frame, build_large_plan_data, solve_large_plan


def make_adapter() -> InputAdapterGd:
    adapter = InputAdapterGd()
    adapter.take_over_vessel = {"E": ["E1"], "I": []}
    adapter.planning_time = pd.Timestamp("2026-01-01 08:00:00")
    adapter.closed_area = set()
    slots = [
        {
            "HAS_CONTAINER": 1,
            "YAA_AREANO": "A",
            "YBY_ENABLECSIZECD": "20,40",
            "IYC_CNTRID": "C1",
            "IYC_EVOY_ID": "E1",
            "IYC_IVOY_ID": "",
            "IYC_CSZ_CSIZECD": "20",
            "IYC_STS_CSTATUSCD": "OF",
        }
    ]
    for area in ("A", "B"):
        for index in range(4):
            slots.append(
                {
                    "HAS_CONTAINER": 0,
                    "YAA_AREANO": area,
                    "YBY_ENABLECSIZECD": "20,40",
                    "IYC_CNTRID": "",
                    "IYC_EVOY_ID": "",
                    "IYC_IVOY_ID": "",
                    "IYC_CSZ_CSIZECD": "",
                    "IYC_STS_CSTATUSCD": "",
                    "slot": index,
                }
            )
    adapter.bay_slots_detail = pd.DataFrame(slots)
    adapter.area_function_info = pd.DataFrame(
        [{"area_no": "A", "cntr_type": "OF"}, {"area_no": "B", "cntr_type": "OF"}]
    )
    adapter.vessel_berth_info = pd.DataFrame(
        [{"VOY_ID": "E1", "VOY_IEFG": "E", "VBT_BTH_ABTHNO": "1", "VBT_BTH_PBTHNO": "1"}]
    )
    adapter.berth_area_dist_matrix = pd.DataFrame(
        [{"area_no": "A", "B1": 10}, {"area_no": "B", "B1": 20}]
    )
    adapter.vessel_containers = {
        "E1": {
            "type": "E",
            "doc_cntrs": pd.DataFrame(
                [
                    {
                        "IYC_CNTRID": "C1",
                        "IYC_EVOY_ID": "E1",
                        "IYC_IVOY_ID": "",
                        "IYC_CSZ_CSIZECD": "20",
                        "IYC_STS_CSTATUSCD": "OF",
                    },
                    {
                        "IYC_CNTRID": "C2",
                        "IYC_EVOY_ID": "E1",
                        "IYC_IVOY_ID": "",
                        "IYC_CSZ_CSIZECD": "20",
                        "IYC_STS_CSTATUSCD": "OF",
                    }
                ]
            ),
            "predict_cntrs": {"20": {"total_volume": 100}},
        }
    }
    return adapter


class PaperLargePlanTests(unittest.TestCase):
    def test_old_tops_field_is_ignored_by_shared_adapter(self) -> None:
        adapter = InputAdapterGd.from_dict({"tops_plan": {"columns": [], "index": [], "data": []}})
        self.assertFalse(hasattr(adapter, "tops_plan"))
        self.assertNotIn("tops_plan", adapter.to_dict())

    def test_prediction_payload_is_ignored(self) -> None:
        adapter = make_adapter()
        adapter.take_over_vessel["E"].append("PRED_ONLY")
        adapter.vessel_containers["PRED_ONLY"] = {
            "type": "E",
            "predict_cntrs": {"40": {"total_volume": 500}},
        }
        data = build_large_plan_data(adapter)

        self.assertNotIn("PRED_ONLY", data.voyages)
        self.assertEqual(2, data.demand20[("E1", "OF")])
        self.assertEqual(1, data.snapshot20[("E1", "OF", "A")])
        self.assertEqual(1, data.new_demand("20", "E1", "OF"))
        self.assertEqual(
            "yard_snapshot_plus_declared_documents_only",
            data.diagnostics["demand_policy"],
        )
        self.assertTrue(data.diagnostics["prediction_input_ignored"])

    def test_gurobi_solution_matches_downstream_large_plan_contract(self) -> None:
        data = build_large_plan_data(make_adapter())
        solution = solve_large_plan(data, time_limit=10, threads=1, verbose=False)
        output = allocation_frame(data, solution)

        self.assertTrue(solution.has_solution)
        self.assertEqual(0, solution.shortage_total)
        self.assertEqual(2, int(output["planned_qty"].sum()))
        self.assertEqual(1, int(output["snapshot_qty"].sum()))
        self.assertEqual(1, int(output["new_qty"].sum()))
        self.assertTrue(
            {"voy_id", "flow", "area_no", "size", "planned_qty", "snapshot_qty", "new_qty"}.issubset(output.columns)
        )


if __name__ == "__main__":
    unittest.main()
