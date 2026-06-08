from pathlib import Path
import unittest

from ThermoCR import ThermoOptions as top_level_ThermoOptions
from ThermoCR import calculate_thermo as top_level_calculate_thermo
from ThermoCR.io import read_molecule_data
from ThermoCR.thermo import ThermoOptions, ThermoResult, calculate_thermo, qm_thermo


class StructuredThermoApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example_path = Path(__file__).resolve().parents[1] / "example" / "CPD.out"

    def test_calculate_thermo_matches_legacy_qm_thermo(self):
        molecule = read_molecule_data(self.example_path)
        options = ThermoOptions(temperature=350.0, pressure=100000.0)

        result = calculate_thermo(molecule, options)
        legacy = qm_thermo(
            atom_numbers=molecule.atom_numbers,
            coords=molecule.coordinates,
            vibfreqs=molecule.frequencies,
            ee=molecule.electronic_energy,
            T=options.temperature,
            P=options.pressure,
            verbose=False,
        )

        self.assertIsInstance(result, ThermoResult)
        self.assertAlmostEqual(result.temperature, legacy["T/K"], places=10)
        self.assertAlmostEqual(result.pressure, legacy["P/Pa"], places=10)
        self.assertAlmostEqual(result.heat_capacity_cp, legacy["Cp/(J/mol/K)"], places=8)
        self.assertAlmostEqual(result.entropy, legacy["S/(J/mol/K)"], places=8)
        self.assertAlmostEqual(result.gibbs_free_energy, legacy["G/(J/mol)"], places=6)

    def test_calculate_thermo_rejects_non_hartree_energy(self):
        molecule = read_molecule_data(self.example_path, return_hartree=False)

        with self.assertRaises(ValueError):
            calculate_thermo(molecule)

    def test_thermo_options_validate_basic_inputs(self):
        with self.assertRaises(ValueError):
            ThermoOptions(temperature=0.0)
        with self.assertRaises(ValueError):
            ThermoOptions(pressure=0.0)
        with self.assertRaises(ValueError):
            ThermoOptions(electronic_energies=[0.0], electronic_degeneracies=None)
        with self.assertRaises(ValueError):
            ThermoOptions(electronic_energies=[0.0], electronic_degeneracies=[1, 2])

    def test_concentration_correction_is_preserved(self):
        molecule = read_molecule_data(self.example_path)
        result = calculate_thermo(molecule, ThermoOptions(concentration=1.0))

        self.assertIsNotNone(result.concentration_delta_g)

    def test_top_level_exports_structured_thermo_api(self):
        self.assertIs(top_level_calculate_thermo, calculate_thermo)
        self.assertIs(top_level_ThermoOptions, ThermoOptions)


if __name__ == "__main__":
    unittest.main()
