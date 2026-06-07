import unittest

from ThermoCR.elements import (
    atom_data,
    atomic_mass,
    atomic_masses,
    atomic_symbol,
    element_mass,
)
from ThermoCR.pointgroup.element_data import (
    atom_data as legacy_atom_data,
    element_mass as legacy_element_mass,
)


class ElementsApiTests(unittest.TestCase):
    def test_new_and_legacy_element_data_paths_match(self):
        self.assertIs(atom_data, legacy_atom_data)
        self.assertIs(element_mass, legacy_element_mass)

    def test_atomic_mass_helpers(self):
        self.assertEqual(atomic_symbol(6), "C")
        self.assertAlmostEqual(atomic_mass(1), 1.00794, places=5)
        self.assertAlmostEqual(atomic_mass("C"), 12.0107, places=4)
        self.assertEqual(atomic_masses([1, 6]), [atomic_mass(1), atomic_mass(6)])


if __name__ == "__main__":
    unittest.main()