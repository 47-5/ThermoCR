from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from ThermoCR import calculate_tst_rate_frame as top_level_calculate_tst_rate_frame
from ThermoCR.kinetics import calculate_tst_rate_frame, k_TST_scan


class StructuredKineticsApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example_dir = Path(__file__).resolve().parents[1] / "example"

    def test_calculate_tst_rate_frame_matches_legacy_scan(self):
        ts = pd.read_excel(self.example_dir / "QMthermoScan_TS.xlsx")
        reactant = pd.read_excel(self.example_dir / "QMthermoScan_CPD.xlsx")
        legacy = k_TST_scan(
            self.example_dir / "QMthermoScan_TS.xlsx",
            self.example_dir / "QMthermoScan_CPD.xlsx",
            self.example_dir / "QMthermoScan_CPD.xlsx",
            out_path=None,
        )

        result = calculate_tst_rate_frame(ts, [reactant, reactant])

        np.testing.assert_allclose(result["temperature"], legacy["T/K"])
        np.testing.assert_allclose(result["rate_constant"], legacy["k"], rtol=1e-12)
        self.assertEqual(list(result["delta_n"].unique()), [1])

    def test_calculate_tst_rate_frame_accepts_structured_columns(self):
        ts = pd.read_excel(self.example_dir / "QMthermoScan_TS.xlsx").rename(
            columns={
                "T/K": "temperature",
                "G/(J/mol)": "gibbs_free_energy",
                "ee/(J/mol)": "electronic_energy",
                "zpe/(J/mol)": "zpe",
            }
        )
        reactant = pd.read_excel(self.example_dir / "QMthermoScan_CPD.xlsx").rename(
            columns={
                "T/K": "temperature",
                "G/(J/mol)": "gibbs_free_energy",
                "ee/(J/mol)": "electronic_energy",
                "zpe/(J/mol)": "zpe",
            }
        )

        result = calculate_tst_rate_frame(ts, [reactant, reactant])

        self.assertIn("rate_constant", result.columns)
        self.assertGreater(float(result["rate_constant"].iloc[-1]), 0.0)

    def test_calculate_tst_rate_frame_validates_temperature_grid(self):
        ts = pd.read_excel(self.example_dir / "QMthermoScan_TS.xlsx")
        reactant = pd.read_excel(self.example_dir / "QMthermoScan_CPD.xlsx")
        shifted = reactant.copy()
        shifted.loc[0, "T/K"] += 1.0

        with self.assertRaises(ValueError):
            calculate_tst_rate_frame(ts, [reactant, shifted])

    def test_top_level_exports_structured_kinetics_api(self):
        self.assertIs(top_level_calculate_tst_rate_frame, calculate_tst_rate_frame)


if __name__ == "__main__":
    unittest.main()
