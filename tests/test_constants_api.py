import unittest

from ThermoCR.constants import R, atomic_number_map, au2kj_mol
from ThermoCR.tools import R as tools_R
from ThermoCR.tools.constant import atomic_number_map as legacy_atomic_number_map


class ConstantsApiTests(unittest.TestCase):
    def test_new_and_legacy_constant_paths_match(self):
        self.assertEqual(R, tools_R)
        self.assertIs(atomic_number_map, legacy_atomic_number_map)
        self.assertEqual(atomic_number_map[0], "H")
        self.assertEqual(atomic_number_map[5], "C")
        self.assertGreater(au2kj_mol, 0.0)


if __name__ == "__main__":
    unittest.main()