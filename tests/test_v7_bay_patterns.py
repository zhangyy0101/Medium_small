from __future__ import annotations

import unittest
from dataclasses import replace

from tests.test_v6_model_contract import make_bay, make_group, make_problem
from yard_planning.v7_atoms import build_v7_row_atoms
from yard_planning.v7_bay_patterns import (
    build_v7_pattern_from_atom_indices,
    enumerate_v7_bay_patterns,
    exhaustive_v7_bay_patterns,
)


class V7BayPatternTests(unittest.TestCase):
    def test_support_enumeration_equals_independent_exhaustive_oracle(self) -> None:
        groups = [make_group(f"G{i}", port=f"P{i}") for i in range(1, 4)]
        problem = make_problem(groups, [make_bay("01", rows=("1", "2", "3"))])
        atoms, _limits = build_v7_row_atoms(problem)

        exact = enumerate_v7_bay_patterns(problem, atoms, "A|01")
        exhaustive = exhaustive_v7_bay_patterns(problem, atoms, "A|01")

        self.assertEqual(
            {pattern.signature for pattern in exhaustive},
            {pattern.signature for pattern in exact},
        )
        self.assertTrue(any(len(pattern.active_groups) == 3 for pattern in exact))
        self.assertTrue(all(len(pattern.active_groups) <= 3 for pattern in exact))

    def test_four_group_pattern_and_mixed_height_pattern_do_not_exist(self) -> None:
        groups = [make_group(f"G{i}", port=f"P{i}") for i in range(1, 5)]
        problem = make_problem(groups, [make_bay("01", rows=("1", "2", "3", "4"))])
        atoms, _limits = build_v7_row_atoms(problem)
        patterns = enumerate_v7_bay_patterns(problem, atoms, "A|01")
        self.assertFalse(any(len(pattern.active_groups) == 4 for pattern in patterns))

        mixed = [
            replace(make_group("H1", port="P1"), height="H1"),
            replace(make_group("H2", port="P2"), height="H2"),
        ]
        mixed_problem = make_problem(mixed, [make_bay("01", rows=("1", "2"))])
        mixed_atoms, _limits = build_v7_row_atoms(mixed_problem)
        mixed_patterns = enumerate_v7_bay_patterns(
            mixed_problem, mixed_atoms, "A|01"
        )
        self.assertFalse(any(len(pattern.active_groups) == 2 for pattern in mixed_patterns))

    def test_40ft_pattern_records_both_physical_bays(self) -> None:
        group = replace(make_group("G1", port="P1"), size="40")
        bays = [make_bay(code, rows=("1",)) for code in ("01", "03")]
        for bay in bays:
            bay.cap_by_size = {"40": 1}
            bay.row_cap_by_size = {"40": {"1": 1}}
        bays[0].large_bay_partner_key = bays[1].bay_key
        problem = make_problem([group], bays)
        atoms, _limits = build_v7_row_atoms(problem)

        pattern = build_v7_pattern_from_atom_indices(
            problem,
            {atom.candidate_index: atom for atom in atoms},
            [atoms[0].candidate_index],
            pattern_id=0,
        )

        self.assertEqual(("A|01", "A|03"), pattern.physical_bays)
        self.assertEqual((("A|01", "1"), ("A|03", "1")), pattern.physical_resources)


if __name__ == "__main__":
    unittest.main()
