from __future__ import annotations

import unittest

from preexperiment.v6_scalability import merge_v6_zone_pools
from tests.test_v6_model_contract import make_bay, make_group, make_problem
from yard_planning.row_aware_zones import build_complete_v6_zone_universe


class V6ScalabilitySupportTests(unittest.TestCase):
    def test_proof_and_primal_pools_are_merged_by_atom_signature(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1", demand=1)],
            [make_bay("01"), make_bay("03")],
        )
        zones = build_complete_v6_zone_universe(problem)
        self.assertGreaterEqual(len(zones), 2)

        merged = merge_v6_zone_pools(
            (zones[0], zones[1]),
            (zones[0], zones[-1]),
        )

        signatures = [tuple(zone.candidate_indices) for zone in merged]
        self.assertEqual(len(signatures), len(set(signatures)))
        self.assertEqual(tuple(zones[0].candidate_indices), signatures[0])
        self.assertEqual(tuple(zones[1].candidate_indices), signatures[1])


if __name__ == "__main__":
    unittest.main()
