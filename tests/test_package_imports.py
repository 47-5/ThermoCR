import unittest

from ThermoCR import (
    get_point_group as top_level_get_point_group,
    get_rotational_symmetry_number as top_level_get_rotational_symmetry_number,
    k_TST,
    q_trans,
    read_qm_out,
)
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

    def test_top_level_exports_formal_api_and_legacy_aliases(self):
        point_group = top_level_get_point_group(self.coords, symbols=self.symbols)

        self.assertEqual(point_group, "Td")
        self.assertEqual(top_level_get_rotational_symmetry_number(point_group), 12)
        self.assertGreater(q_trans(M=28.0, T=298.15, P=101325.0), 0.0)
        self.assertGreater(k_TST(delta_G=0.0, delta_n=0, T=298.15), 0.0)
        self.assertTrue(callable(read_qm_out))


if __name__ == "__main__":
    unittest.main()