import unittest

from ThermoCR.thermo import sort_key
from ThermoCR.thermo.solvation import (
    calculate_solvent_energy,
    sort_key as namespaced_sort_key,
)
from ThermoCR.tools.about_gaussian.calculate_solvent_energy import (
    calculate_solvent_energy as legacy_calculate_solvent_energy,
    sort_key as legacy_sort_key,
)


class SolvationApiTests(unittest.TestCase):
    def test_new_and_legacy_solvation_helpers_match(self):
        self.assertIs(calculate_solvent_energy, legacy_calculate_solvent_energy)
        self.assertIs(namespaced_sort_key, legacy_sort_key)
        self.assertIs(sort_key, namespaced_sort_key)

    def test_sort_key_uses_historical_gaussian_path_pattern(self):
        self.assertEqual(sort_key("01_02_path3_04.out"), (1, 2, 3, 4))
        self.assertEqual(sort_key("unmatched.out"), (float("inf"),) * 4)


if __name__ == "__main__":
    unittest.main()
