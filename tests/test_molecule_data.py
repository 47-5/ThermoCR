from pathlib import Path
import unittest

from ThermoCR import MoleculeData as top_level_MoleculeData
from ThermoCR import read_molecule_data as top_level_read_molecule_data
from ThermoCR.io import read_electronic_energy, read_molecule_data
from ThermoCR.types import MoleculeData


class MoleculeDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example_path = Path(__file__).resolve().parents[1] / "example" / "CPD.out"

    def test_read_molecule_data_parses_link1_last_job(self):
        molecule = read_molecule_data(self.example_path)
        energy = read_electronic_energy(self.example_path)

        self.assertIsInstance(molecule, MoleculeData)
        self.assertEqual(molecule.n_atoms, 11)
        self.assertEqual(molecule.symbols.count("C"), 5)
        self.assertEqual(molecule.symbols.count("H"), 6)
        self.assertEqual(molecule.coordinates.shape, (11, 3))
        self.assertEqual(len(molecule.frequencies), 27)
        self.assertEqual(len(molecule.imaginary_frequencies), 0)
        self.assertEqual(molecule.electronic_energy_unit, "hartree")
        self.assertAlmostEqual(molecule.electronic_energy, float(energy), places=10)

    def test_read_molecule_data_can_keep_cclib_energy_unit(self):
        molecule = read_molecule_data(self.example_path, return_hartree=False)

        self.assertEqual(molecule.electronic_energy_unit, "eV")
        self.assertLess(molecule.electronic_energy, 0.0)

    def test_top_level_exports_molecule_data_api(self):
        self.assertIs(top_level_MoleculeData, MoleculeData)
        self.assertIs(top_level_read_molecule_data, read_molecule_data)

    def test_molecule_data_normalizes_sequence_inputs(self):
        molecule = MoleculeData(
            symbols=["H", "H"],
            atom_numbers=[1, 1],
            coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            frequencies=[4400.0],
            imaginary_frequencies=[],
        )

        self.assertEqual(molecule.symbols, ("H", "H"))
        self.assertEqual(molecule.atom_numbers.shape, (2,))
        self.assertEqual(molecule.coordinates.shape, (2, 3))
        self.assertEqual(molecule.frequencies.shape, (1,))


if __name__ == "__main__":
    unittest.main()
