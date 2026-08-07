from __future__ import annotations

from collections import Counter
from pathlib import Path
import unittest

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd, normalize_voyage_id
from adapters.planning_input import normalize_container_size, read_export_berths
from example.generate_diverse_voyages_case import (
    SOURCE_EXPORT_VOYAGES,
    build_case,
)
from example.generate_many_groups_case import build_case as build_many_groups_case


ROOT = Path(__file__).resolve().parents[1]


def demand_profile(frame: pd.DataFrame) -> Counter[tuple[str, str, str]]:
    return Counter(
        zip(
            frame["IYC_CSZ_CSIZECD"].astype(str),
            frame["IYC_CHEIGHTCD"].astype(str),
            frame["IYC_POT_UNLDPORT"].astype(str),
        )
    )


class DiverseCaseGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adapter, cls.large_plan, cls.manifest = build_case(
            ROOT / "example" / "input_data.json",
            ROOT / "example" / "large_plan.csv",
            copies=3,
        )

    def test_total_scale_is_exact_and_profiles_are_not_replicated(self) -> None:
        self.assertEqual(3.0, self.manifest["declared_scale"])
        self.assertEqual(2241, self.manifest["declared_export_container_rows"])
        for source in SOURCE_EXPORT_VOYAGES:
            voyages = self.manifest["export_voyage_mapping"][source]
            frames = [
                self.adapter.vessel_containers[voyage]["doc_cntrs"]
                for voyage in voyages
            ]
            self.assertEqual(3 * len(frames[0]), sum(map(len, frames)))
            profiles = [demand_profile(frame) for frame in frames]
            self.assertEqual(3, len({tuple(sorted(profile.items())) for profile in profiles}))

    def test_synthetic_plan_matches_documents_and_has_no_snapshot_load(self) -> None:
        plan = self.large_plan.copy()
        plan["_voyage"] = plan["voy_id"].map(normalize_voyage_id)
        synthetic_voyages = self.manifest["synthetic_voyage_profiles"]
        for voyage in synthetic_voyages:
            rows = plan.loc[plan["_voyage"].eq(voyage)]
            self.assertFalse(rows.empty)
            self.assertTrue(rows["snapshot_qty"].fillna(0).eq(0).all())
            self.assertTrue(rows["planned_qty"].eq(rows["new_qty"]).all())

            documents = self.adapter.vessel_containers[voyage]["doc_cntrs"]
            sizes = documents["IYC_CSZ_CSIZECD"].map(normalize_container_size)
            demand = sizes.map(lambda value: "40" if value == "45" else value).value_counts()
            planned = rows.groupby(rows["size"].astype(int).astype(str))["new_qty"].sum()
            self.assertEqual(demand.to_dict(), planned.astype(int).to_dict())

    def test_synthetic_berths_have_complete_distances(self) -> None:
        synthetic_voyages = list(self.manifest["synthetic_voyage_profiles"])
        berths = read_export_berths(self.adapter, synthetic_voyages)
        self.assertEqual(set(synthetic_voyages), set(berths))
        self.assertEqual(len(synthetic_voyages), len(set(berths.values())))
        matrix = self.adapter.berth_area_dist_matrix
        for berth in berths.values():
            self.assertIn(berth, matrix)
            self.assertTrue(pd.to_numeric(matrix[berth], errors="coerce").notna().all())


class ManyGroupsCaseGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adapter, cls.large_plan, cls.manifest = build_many_groups_case(
            ROOT / "example" / "input_data.json",
            ROOT / "example" / "large_plan.csv",
        )

    def test_every_detailed_voyage_has_twelve_positive_groups(self) -> None:
        self.assertEqual(6, self.manifest["detailed_export_voyage_count"])
        self.assertEqual(72, self.manifest["total_export_group_count"])
        self.assertEqual(2241, self.manifest["declared_export_container_rows"])
        for voyage in self.manifest["detailed_export_voyages"]:
            documents = self.adapter.vessel_containers[voyage]["doc_cntrs"]
            profile = demand_profile(documents)
            self.assertEqual(12, len(profile))
            self.assertTrue(all(quantity > 0 for quantity in profile.values()))

    def test_group_expansion_preserves_original_voyage_size_totals(self) -> None:
        for voyage in self.manifest["detailed_export_voyages"]:
            documents = self.adapter.vessel_containers[voyage]["doc_cntrs"]
            sizes = documents["IYC_CSZ_CSIZECD"].map(normalize_container_size)
            demand = sizes.value_counts().to_dict()
            profile = self.manifest["voyage_profiles"][voyage]
            self.assertEqual(
                profile["source_size_totals"],
                profile["expanded_size_totals"],
            )
            self.assertEqual(demand, profile["expanded_size_totals"])


if __name__ == "__main__":
    unittest.main()
