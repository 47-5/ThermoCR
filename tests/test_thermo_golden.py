from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from ThermoCR.constants import au2j_mol
from ThermoCR.io import read_molecule_data
from ThermoCR.thermo import ThermoOptions, scan_thermo


THERMO_COLUMN_MAP = {
    "temperature": "T/K",
    "pressure": "P/Pa",
    "partition_function_v0": "q_tot_v_0",
    "partition_function_bottom": "q_tot_bot",
    "heat_capacity_cv": "Cv/(J/mol/K)",
    "heat_capacity_cp": "Cp/(J/mol/K)",
    "entropy": "S/(J/mol/K)",
    "zpe": "zpe/(J/mol)",
    "internal_energy_correction": "U_corr/(J/mol)",
    "enthalpy_correction": "H_corr/(J/mol)",
    "gibbs_energy_correction": "G_corr/(J/mol)",
    "electronic_energy": "ee/(J/mol)",
    "internal_energy": "U/(J/mol)",
    "enthalpy": "H/(J/mol)",
    "gibbs_free_energy": "G/(J/mol)",
}

THERMO_GOLDEN_CASES = [
    ("CPD.out", "QMthermoScan_CPD.xlsx"),
    ("DCPD.out", "QMthermoScan_DCPD.xlsx"),
    ("TS_CPD_DCPD.out", "QMthermoScan_TS.xlsx"),
    ("IRC_freq/01_02_path1_1_freq.out", "QMthermoScan_01_02_path1_1.xlsx"),
    ("IRC_freq/01_02_path2_1_freq.out", "QMthermoScan_01_02_path2_1.xlsx"),
]


class ThermoGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example_dir = Path(__file__).resolve().parents[1] / "example"

    def test_scan_thermo_matches_legacy_excel_golden_files(self):
        for output_name, golden_name in THERMO_GOLDEN_CASES:
            with self.subTest(output=output_name, golden=golden_name):
                golden = pd.read_excel(self.example_dir / golden_name)
                molecule = read_molecule_data(self.example_dir / output_name)
                molecule = replace(
                    molecule,
                    electronic_energy=float(golden["ee/(J/mol)"].iloc[0]) / au2j_mol,
                )
                pressure = float(golden["P/Pa"].iloc[0])

                result = scan_thermo(
                    molecule,
                    temperatures=golden["T/K"],
                    pressure=pressure,
                    options=ThermoOptions(pressure=pressure),
                )

                for result_column, golden_column in THERMO_COLUMN_MAP.items():
                    np.testing.assert_allclose(
                        result[result_column].to_numpy(dtype=float),
                        golden[golden_column].to_numpy(dtype=float),
                        rtol=1e-12,
                        atol=1e-6,
                        err_msg=f"{output_name}: {result_column}",
                    )


if __name__ == "__main__":
    unittest.main()
