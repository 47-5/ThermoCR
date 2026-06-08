import unittest

import numpy as np
import pandas as pd

from ThermoCR import fit_thermo_frame as top_level_fit_thermo_frame
from ThermoCR.thermo import ThermoFitResult, fit_thermo_frame, nasa7


class ThermoFittingApiTests(unittest.TestCase):
    def setUp(self):
        self.temperatures = np.linspace(300.0, 1000.0, 12)
        self.parameters = (3.5, 1.0e-3, -2.0e-6, 1.0e-9, -1.0e-13, -1000.0, 5.0)
        cp, enthalpy, entropy = nasa7(self.temperatures, *self.parameters)
        self.frame = pd.DataFrame(
            {
                "temperature": self.temperatures,
                "heat_capacity_cp": cp,
                "enthalpy": enthalpy,
                "entropy": entropy,
            }
        )

    def test_fit_thermo_frame_returns_structured_result(self):
        result = fit_thermo_frame(
            self.frame,
            model_type="NASA7",
            guess=list(self.parameters),
            weight_strategy="uniform",
        )

        self.assertIsInstance(result, ThermoFitResult)
        self.assertEqual(result.model_type, "NASA7")
        self.assertEqual(result.temperature_range, (300.0, 1000.0))
        self.assertGreater(result.metrics["heat_capacity_cp"]["r2"], 0.999999)

        cp, enthalpy, entropy = result.predict(self.temperatures)
        np.testing.assert_allclose(cp, self.frame["heat_capacity_cp"], rtol=1e-9, atol=1e-6)
        np.testing.assert_allclose(enthalpy, self.frame["enthalpy"], rtol=1e-9, atol=1e-6)
        np.testing.assert_allclose(entropy, self.frame["entropy"], rtol=1e-9, atol=1e-6)

    def test_fit_thermo_frame_accepts_legacy_columns(self):
        legacy_frame = self.frame.rename(
            columns={
                "temperature": "T/K",
                "heat_capacity_cp": "Cp/(J/mol/K)",
                "enthalpy": "H/(J/mol)",
                "entropy": "S/(J/mol/K)",
            }
        )

        result = fit_thermo_frame(
            legacy_frame,
            model_type="nasa7",
            guess=list(self.parameters),
            weight_strategy="uniform",
        )

        self.assertEqual(result.model_type, "NASA7")
        self.assertGreater(result.metrics["entropy"]["r2"], 0.999999)

    def test_fit_thermo_frame_validates_columns(self):
        with self.assertRaises(ValueError):
            fit_thermo_frame(self.frame.drop(columns=["entropy"]))

    def test_top_level_exports_fit_api(self):
        self.assertIs(top_level_fit_thermo_frame, fit_thermo_frame)


if __name__ == "__main__":
    unittest.main()
