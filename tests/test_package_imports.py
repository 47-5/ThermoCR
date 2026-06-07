import unittest

from ThermoCR import (
    __all__ as thermocr_all,
    __version__,
    PointGroup,
    Rotation,
    atomic_mass,
    get_point_group as top_level_get_point_group,
    get_rotational_symmetry_number as top_level_get_rotational_symmetry_number,
    k_TST,
    q_trans,
    read_qm_out,
    sort_key as top_level_sort_key,
)
from ThermoCR.tools.about_orca import sort_key as orca_sort_key
from ThermoCR.tools.utils import get_point_group, get_rotational_symmetry_number


class PackageImportTests(unittest.TestCase):
    def setUp(self):
        self.symbols = ["C", "H", "H", "H", "H"]
        self.coords = [
            [0.000000, 0.000000, 0.000000],
            [0.629118, 0.629118, 0.629118],
            [-0.629118, -0.629118, 0.629118],
            [-0.629118, 0.629118, -0.629118],
            [0.629118, -0.629118, -0.629118],
        ]

    def test_pointgroup_uses_packaged_grid_module(self):
        point_group = get_point_group(self.coords, symbols=self.symbols)

        self.assertEqual(point_group, "Td")
        self.assertEqual(get_rotational_symmetry_number(point_group), 12)

    def test_top_level_exposes_package_version(self):
        self.assertEqual(__version__, "1.0")

    def test_top_level_exports_formal_api_and_legacy_aliases(self):
        point_group = top_level_get_point_group(self.coords, symbols=self.symbols)

        self.assertEqual(point_group, "Td")
        self.assertEqual(top_level_get_rotational_symmetry_number(point_group), 12)
        self.assertGreater(q_trans(M=28.0, T=298.15, P=101325.0), 0.0)
        self.assertGreater(k_TST(delta_G=0.0, delta_n=0, T=298.15), 0.0)
        self.assertTrue(callable(read_qm_out))
        self.assertAlmostEqual(atomic_mass(6), 12.0107, places=4)

    def test_top_level_all_keeps_common_public_names(self):
        self.assertIn("q_trans", thermocr_all)
        self.assertIn("k_TST", thermocr_all)
        self.assertIn("read_qm_out", thermocr_all)
        self.assertIn("PointGroup", thermocr_all)
        self.assertIn("get_point_group", thermocr_all)
        self.assertTrue(callable(PointGroup))
        self.assertTrue(callable(Rotation))

    def test_top_level_sort_key_keeps_legacy_orca_precedence(self):
        self.assertIs(top_level_sort_key, orca_sort_key)


if __name__ == "__main__":
    unittest.main()
