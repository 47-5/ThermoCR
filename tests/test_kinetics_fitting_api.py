import unittest

import numpy as np
import pandas as pd

from ThermoCR import fit_kinetics_frame as top_level_fit_kinetics_frame
from ThermoCR.kinetics import (
    KineticsFitResult,
    arrhenius,
    fit_kinetics_frame,
)


class KineticsFittingApiTests(unittest.TestCase):
    def setUp(self):
        self.temperatures = np.linspace(300.0, 1000.0, 12)
        self.parameters = (1.2e7, 35000.0, 0.5)
        self.rates = arrhenius(self.temperatures, *self.parameters)
        self.frame = pd.DataFrame({
            "temperature": self.temperatures,
            "rate_constant": self.rates,
        })

    def test_fit_kinetics_frame_returns_structured_result(self):
        result = fit_kinetics_frame(
            self.frame,
            model_type="Arrhenius",
            guess=list(self.parameters),
        )

        self.assertIsInstance(result, KineticsFitResult)
        self.assertEqual(result.model_type, "Arrhenius")
        self.assertGreater(result.metrics["rate_constant"]["r2"], 0.999999)
        self.assertEqual(set(result.named_parameters()), {"A", "Ea", "b"})
        np.testing.assert_allclose(
            result.predict(self.temperatures),
            self.rates,
            rtol=1e-9,
            atol=1e-12,
        )

    def test_fit_kinetics_frame_accepts_legacy_columns(self):
        legacy_frame = self.frame.rename(
            columns={
                "temperature": "T/K",
                "rate_constant": "k",
            }
        )

        result = fit_kinetics_frame(
            legacy_frame,
            model_type="arrhenius",
            guess=list(self.parameters),
        )

        self.assertEqual(result.model_type, "Arrhenius")
        self.assertGreater(result.metrics["rate_constant"]["r2"], 0.999999)

    def test_fit_kinetics_frame_validates_columns(self):
        with self.assertRaises(ValueError):
            fit_kinetics_frame(self.frame.drop(columns=["rate_constant"]))

    def test_top_level_exports_fit_api(self):
        self.assertIs(top_level_fit_kinetics_frame, fit_kinetics_frame)


if __name__ == "__main__":
    unittest.main()
