from pathlib import Path
import unittest

import numpy as np

from ThermoCR.constants import R
from ThermoCR import ThermoOptions as top_level_ThermoOptions
from ThermoCR import calculate_thermo as top_level_calculate_thermo
from ThermoCR import scan_thermo as top_level_scan_thermo
from ThermoCR.io import read_molecule_data
from ThermoCR.thermo import (
    ThermoOptions,
    ThermoResult,
    calculate_thermo,
    qm_thermo,
    qm_thermo_scan,
    scan_thermo,
)


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

    def test_scan_thermo_returns_dataframe_without_writing_files(self):
        molecule = read_molecule_data(self.example_path)

        df = scan_thermo(
            molecule,
            temperatures=[300.0, 400.0],
            pressure=100000.0,
        )

        self.assertEqual(list(df["temperature"]), [300.0, 400.0])
        self.assertEqual(list(df["pressure"]), [100000.0, 100000.0])
        self.assertIn("gibbs_free_energy", df.columns)
        self.assertEqual(len(df), 2)

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
            ThermoOptions(rotational_symmetry_number=0.0)
        with self.assertRaises(ValueError):
            ThermoOptions(electronic_energies=[0.0], electronic_degeneracies=None)
        with self.assertRaises(ValueError):
            ThermoOptions(electronic_energies=[0.0], electronic_degeneracies=[1, 2])

    def test_rotational_symmetry_number_override_changes_rotational_entropy(self):
        molecule = read_molecule_data(self.example_path)
        temperature = 350.0
        result_sigma_1 = calculate_thermo(
            molecule,
            ThermoOptions(
                temperature=temperature,
                pressure=100000.0,
                rotational_symmetry_number=1,
            ),
        )
        result_sigma_2 = calculate_thermo(
            molecule,
            ThermoOptions(
                temperature=temperature,
                pressure=100000.0,
                rotational_symmetry_number=2,
            ),
        )

        self.assertAlmostEqual(
            result_sigma_1.partition_function_v0 / result_sigma_2.partition_function_v0,
            2.0,
            places=10,
        )
        self.assertAlmostEqual(
            result_sigma_1.entropy - result_sigma_2.entropy,
            R * np.log(2.0),
            places=8,
        )
        self.assertAlmostEqual(
            result_sigma_2.gibbs_free_energy - result_sigma_1.gibbs_free_energy,
            temperature * R * np.log(2.0),
            places=6,
        )

    def test_point_group_override_matches_equivalent_symmetry_number(self):
        molecule = read_molecule_data(self.example_path)
        from_point_group = calculate_thermo(
            molecule,
            ThermoOptions(point_group="C2", pressure=100000.0),
        )
        from_sigma = calculate_thermo(
            molecule,
            ThermoOptions(rotational_symmetry_number=2, pressure=100000.0),
        )

        self.assertAlmostEqual(
            from_point_group.gibbs_free_energy,
            from_sigma.gibbs_free_energy,
            places=6,
        )

    def test_legacy_qm_thermo_scan_accepts_symmetry_override(self):
        molecule = read_molecule_data(self.example_path)
        structured = calculate_thermo(
            molecule,
            ThermoOptions(
                temperature=350.0,
                pressure=100000.0,
                rotational_symmetry_number=2,
            ),
        )

        legacy_scan = qm_thermo_scan(
            atom_numbers=molecule.atom_numbers,
            coords=molecule.coordinates,
            vibfreqs=molecule.frequencies,
            ee=molecule.electronic_energy,
            T=[350.0],
            P=[100000.0],
            rotational_symmetry_number=2,
            out_path=None,
        )

        self.assertAlmostEqual(
            float(legacy_scan["G/(J/mol)"].iloc[0]),
            structured.gibbs_free_energy,
            places=6,
        )

    def test_concentration_correction_is_preserved(self):
        molecule = read_molecule_data(self.example_path)
        result = calculate_thermo(molecule, ThermoOptions(concentration=1.0))

        self.assertIsNotNone(result.concentration_delta_g)

    def test_top_level_exports_structured_thermo_api(self):
        self.assertIs(top_level_calculate_thermo, calculate_thermo)
        self.assertIs(top_level_ThermoOptions, ThermoOptions)
        self.assertIs(top_level_scan_thermo, scan_thermo)


if __name__ == "__main__":
    unittest.main()
