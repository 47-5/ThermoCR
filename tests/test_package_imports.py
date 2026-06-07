import unittest

from ThermoCR.tools.utils import get_point_group, get_rotational_symmetry_number


class PackageImportTests(unittest.TestCase):
    def test_pointgroup_uses_packaged_grid_module(self):
        symbols = ["C", "H", "H", "H", "H"]
        coords = [
            [0.000000, 0.000000, 0.000000],
            [0.629118, 0.629118, 0.629118],
            [-0.629118, -0.629118, 0.629118],
            [-0.629118, 0.629118, -0.629118],
            [0.629118, -0.629118, -0.629118],
        ]

        point_group = get_point_group(coords, symbols=symbols)

        self.assertEqual(point_group, "Td")
        self.assertEqual(get_rotational_symmetry_number(point_group), 12)


if __name__ == "__main__":
    unittest.main()
