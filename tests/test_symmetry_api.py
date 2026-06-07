import unittest

from ThermoCR.symmetry import (
    detect_point_group,
    is_linear,
    principal_moments,
    rotational_symmetry_number,
)
from ThermoCR.tools.utils import (
    check_linear,
    get_I,
    get_point_group,
    get_rotational_symmetry_number,
)


class SymmetryApiTests(unittest.TestCase):
    def test_new_and_legacy_pointgroup_interfaces_match(self):
        symbols = ["C", "H", "H", "H", "H"]
        coords = [
            [0.000000, 0.000000, 0.000000],
            [0.629118, 0.629118, 0.629118],
            [-0.629118, -0.629118, 0.629118],
            [-0.629118, 0.629118, -0.629118],
            [0.629118, -0.629118, -0.629118],
        ]

        self.assertEqual(detect_point_group(coords, symbols=symbols), "Td")
        self.assertEqual(get_point_group(coords, symbols=symbols), "Td")

    def test_rotational_symmetry_number_mapping(self):
        expected = {
            "C1": 1,
            "Cs": 1,
            "Ci": 1,
            "C3v": 3,
            "D3h": 6,
            "S4": 4,
            "Td": 12,
            "Oh": 24,
            "Dinfh": 2,
            "Cinfv": 1,
        }

        for point_group, sigma in expected.items():
            with self.subTest(point_group=point_group):
                self.assertEqual(rotational_symmetry_number(point_group), sigma)
                self.assertEqual(get_rotational_symmetry_number(point_group), sigma)

    def test_unknown_point_group_raises(self):
        with self.assertRaises(ValueError):
            rotational_symmetry_number("NotAGroup")

    def test_moments_and_linearity_wrappers_match(self):
        numbers = [8, 6, 8]
        coords = [
            [-1.16, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.16, 0.0, 0.0],
        ]

        moments = principal_moments(coords=coords, numbers=numbers)
        legacy_moments = get_I(coords=coords, numbers=numbers)

        self.assertTrue(is_linear(moments))
        self.assertTrue(check_linear(legacy_moments))
        self.assertEqual(len(moments), 3)


if __name__ == "__main__":
    unittest.main()
