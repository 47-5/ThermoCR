import unittest

import numpy as np
import pandas as pd

from ThermoCR import fit_thermo_frame as top_level_fit_thermo_frame
from ThermoCR.constants import R
from ThermoCR.thermo import (
    ThermoFitResult,
    fit_thermo_frame,
    nasa7,
    nasa9,
    shomate,
)


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
                "pressure": 100000.0,
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
        self.assertEqual(result.reference_pressure_pa, 100000.0)
        self.assertEqual(result.as_dict()["reference_pressure_pa"], 100000.0)
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
                "pressure": "P/Pa",
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

    def test_fit_thermo_frame_rejects_multiple_reference_pressures(self):
        frame = self.frame.copy()
        frame.loc[0, "pressure"] = 101325.0

        with self.assertRaisesRegex(ValueError, "multiple reference pressures"):
            fit_thermo_frame(
                frame,
                guess=list(self.parameters),
                weight_strategy="uniform",
            )

    def test_fit_thermo_frame_rejects_pressure_conflict(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            fit_thermo_frame(
                self.frame,
                reference_pressure_pa=101325.0,
                guess=list(self.parameters),
                weight_strategy="uniform",
            )

    def test_fit_thermo_frame_uses_explicit_pressure_without_pressure_column(self):
        result = fit_thermo_frame(
            self.frame.drop(columns=["pressure"]),
            reference_pressure_pa=100000.0,
            guess=list(self.parameters),
            weight_strategy="uniform",
        )

        self.assertEqual(result.reference_pressure_pa, 100000.0)

    def test_fit_thermo_frame_warns_before_legacy_pressure_fallback(self):
        with self.assertWarnsRegex(DeprecationWarning, "101325"):
            result = fit_thermo_frame(
                self.frame.drop(columns=["pressure"]),
                guess=list(self.parameters),
                weight_strategy="uniform",
            )

        self.assertEqual(result.reference_pressure_pa, 101325.0)

    def test_fit_thermo_frame_preserves_attrs_only_reference_pressure(self):
        frame = self.frame.drop(columns=["pressure"])
        frame.attrs["reference_pressure_pa"] = 100000.0

        result = fit_thermo_frame(
            frame,
            guess=list(self.parameters),
            weight_strategy="uniform",
        )

        self.assertEqual(result.reference_pressure_pa, 100000.0)

    def test_fit_thermo_frame_rejects_pressure_column_attrs_conflict(self):
        frame = self.frame.copy()
        frame.attrs["reference_pressure_pa"] = 101325.0

        with self.assertRaisesRegex(ValueError, "multiple reference pressures"):
            fit_thermo_frame(
                frame,
                guess=list(self.parameters),
                weight_strategy="uniform",
            )


class ThermoModelEquationTests(unittest.TestCase):
    def test_thermo_models_accept_integer_temperature_arrays(self):
        temperatures = np.array([300, 1000], dtype=int)

        nasa9_values = nasa9(temperatures, *([0.0] * 9))
        shomate_values = shomate(
            temperatures,
            30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0,
        )

        self.assertTrue(all(np.all(np.isfinite(value)) for value in nasa9_values))
        self.assertTrue(all(np.all(np.isfinite(value)) for value in shomate_values))

    def test_thermo_models_reject_nonpositive_temperature(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            nasa7(0.0, *([0.0] * 7))
        with self.assertRaisesRegex(ValueError, "temperature"):
            nasa9(-1.0, *([0.0] * 9))
        with self.assertRaisesRegex(ValueError, "temperature"):
            shomate(0.0, *([0.0] * 7))

    def test_nasa7_uses_package_gas_constant(self):
        cp, enthalpy, entropy = nasa7(
            300.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )

        self.assertAlmostEqual(cp, R)
        self.assertAlmostEqual(enthalpy, R * 300.0)
        self.assertAlmostEqual(entropy, R * np.log(300.0))

    def test_nasa9_entropy_is_not_multiplied_by_temperature(self):
        temperature = np.array([300.0, 1000.0])
        parameters = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0)

        _, _, entropy = nasa9(temperature, *parameters)

        np.testing.assert_allclose(entropy, np.full(2, 2.0 * R))

    def test_shomate_enthalpy_is_returned_in_joules_per_mole(self):
        temperature = np.array([300.0, 1000.0])
        parameters = (0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0)

        _, enthalpy, _ = shomate(temperature, *parameters)

        np.testing.assert_allclose(enthalpy, np.full(2, 10_000.0))

    def test_thermo_models_satisfy_temperature_derivative_identities(self):
        cases = (
            (nasa7, (3.5, 1.0e-3, -2.0e-7, 0.0, 0.0, -1000.0, 2.0)),
            (nasa9, (0.0, 0.0, 3.5, 1.0e-3, -2.0e-7, 0.0, 0.0, -1000.0, 2.0)),
            (shomate, (30.0, 2.0, -0.5, 0.1, 0.0, -10.0, 200.0)),
        )
        temperature = 900.0
        delta_temperature = 1.0e-3

        for model, parameters in cases:
            with self.subTest(model=model.__name__):
                cp, _, _ = model(temperature, *parameters)
                _, h_plus, s_plus = model(
                    temperature + delta_temperature,
                    *parameters,
                )
                _, h_minus, s_minus = model(
                    temperature - delta_temperature,
                    *parameters,
                )
                numerical_dh_dt = (h_plus - h_minus) / (2.0 * delta_temperature)
                numerical_ds_dt = (s_plus - s_minus) / (2.0 * delta_temperature)

                self.assertAlmostEqual(numerical_dh_dt, cp, delta=1.0e-5)
                self.assertAlmostEqual(
                    numerical_ds_dt,
                    cp / temperature,
                    delta=1.0e-8,
                )


if __name__ == "__main__":
    unittest.main()
