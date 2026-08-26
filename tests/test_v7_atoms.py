from __future__ import annotations

import unittest

from tests.test_v6_model_contract import make_bay, make_group, make_problem
from yard_planning.models import existing_export_group_key
from yard_planning.v7_atoms import build_v7_row_atoms


class V7AtomTests(unittest.TestCase):
    def test_atoms_are_zone_free_and_preserve_physical_resources(self) -> None:
        problem = make_problem(
            [make_group("G1", port="P1")],
            [make_bay("01", rows=("1", "2"))],
        )
        atoms, limits = build_v7_row_atoms(problem)

        self.assertEqual(2, len(atoms))
        self.assertEqual({"1", "2"}, {atom.row_no for atom in atoms})
        self.assertTrue(all(atom.physical_bays == ("A|01",) for atom in atoms))
        self.assertEqual(2, limits[("G1", "A|01")])
        self.assertFalse(any(hasattr(atom, "zone_id") for atom in atoms))

    def test_existing_mixed_height_is_nonworsening(self) -> None:
        bay = make_bay("01", rows=("1",))
        bay.existing_heights = {"H1", "H2"}
        groups = [
            make_group("G1", port="P1"),
            make_group("G2", port="P2"),
            make_group("G3", port="P3"),
        ]
        groups[0] = type(groups[0])(**{**groups[0].__dict__, "height": "H1"})
        groups[1] = type(groups[1])(**{**groups[1].__dict__, "height": "H2"})
        groups[2] = type(groups[2])(**{**groups[2].__dict__, "height": "H3"})

        atoms, _limits = build_v7_row_atoms(make_problem(groups, [bay]))

        self.assertEqual({"G1", "G2"}, {atom.group_id for atom in atoms})

    def test_exact_existing_row_group_is_required(self) -> None:
        bay = make_bay("01", rows=("1",))
        exact = make_group("G1", port="P1")
        only_voyage = make_group("G2", port="P2")
        only_port = make_group("G3", port="P1")
        only_port = type(only_port)(
            **{**only_port.__dict__, "voyage_id": "V2"}
        )
        bay.existing_group_keys_by_row = {
            "1": {existing_export_group_key(exact)}
        }
        problem = make_problem([exact, only_voyage, only_port], [bay])
        problem.target_voyages.append("V2")
        problem.export_voyages.add("V2")

        atoms, _limits = build_v7_row_atoms(problem)

        self.assertEqual({"G1"}, {atom.group_id for atom in atoms})

    def test_45ft_atoms_are_limited_to_edge_large_bays(self) -> None:
        group = make_group("G1", port="P1")
        group = type(group)(**{**group.__dict__, "size": "45"})
        bays = [make_bay(code, rows=("1",)) for code in ("01", "03", "05", "07")]
        for bay in bays:
            bay.cap_by_size = {"45": 1}
            bay.row_cap_by_size = {"45": {"1": 1}}
        bays[0].large_bay_partner_key = bays[1].bay_key
        bays[2].large_bay_partner_key = bays[3].bay_key

        atoms, _limits = build_v7_row_atoms(make_problem([group], bays))

        self.assertEqual({"A|01", "A|05"}, {atom.anchor_bay_key for atom in atoms})


if __name__ == "__main__":
    unittest.main()
