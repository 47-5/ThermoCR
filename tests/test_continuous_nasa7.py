import math
import unittest

import numpy as np

from ThermoCR import (
    ContinuousNASA7Fit as top_level_continuous_nasa7_fit,
    GAS_CONSTANT_J_MOL_K as top_level_gas_constant,
    fit_continuous_nasa7 as top_level_fit_continuous_nasa7,
    nasa7_values as top_level_nasa7_values,
)
from ThermoCR.thermo import (
    ContinuousNASA7Fit,
    GAS_CONSTANT_J_MOL_K,
    fit_continuous_nasa7,
    nasa7_values,
)


class ContinuousNASA7FitTests(unittest.TestCase):
    def _exact_piecewise_data(self):
        midpoint = 1000.0
        low = np.array(
            [3.5, 1.0e-3, -2.0e-7, 4.0e-11, -3.0e-15, -1000.0, 2.0]
        )
        high = np.array(
            [0.0, 7.0e-4, -8.0e-8, 1.0e-11, -5.0e-16, 0.0, 0.0]
        )
        low_midpoint = nasa7_values([midpoint], low)
        high_cp_without_a1 = nasa7_values([midpoint], high)[0][0]
        high[0] = (
            low_midpoint[0][0] - high_cp_without_a1
        ) / GAS_CONSTANT_J_MOL_K
        high[5] = self._enthalpy_constant(
            high[:5],
            midpoint,
            low_midpoint[1][0],
        )
        high[6] = self._entropy_constant(
            high[:5],
            midpoint,
            low_midpoint[2][0],
        )
        temperatures = np.array(
            sorted(list(np.arange(200.0, 1500.1, 50.0)) + [298.15])
        )
        low_mask = temperatures <= midpoint
        cp = np.empty_like(temperatures)
        enthalpy = np.empty_like(temperatures)
        entropy = np.empty_like(temperatures)
        cp[low_mask], enthalpy[low_mask], entropy[low_mask] = nasa7_values(
            temperatures[low_mask],
            low,
        )
        cp[~low_mask], enthalpy[~low_mask], entropy[~low_mask] = nasa7_values(
            temperatures[~low_mask],
            high,
        )
        return temperatures, cp, enthalpy, entropy

    @staticmethod
    def _enthalpy_constant(coefficients, temperature, enthalpy):
        integral = sum(
            coefficient * temperature ** (power + 1) / (power + 1)
            for power, coefficient in enumerate(coefficients)
        )
        return enthalpy / GAS_CONSTANT_J_MOL_K - integral

    @staticmethod
    def _entropy_constant(coefficients, temperature, entropy):
        terms = coefficients[0] * math.log(temperature)
        terms += sum(
            coefficient * temperature**power / power
            for power, coefficient in enumerate(coefficients[1:], start=1)
        )
        return entropy / GAS_CONSTANT_J_MOL_K - terms

    def test_exact_piecewise_curve_is_recovered(self):
        temperatures, cp, enthalpy, entropy = self._exact_piecewise_data()

        result = fit_continuous_nasa7(
            temperatures,
            cp,
            enthalpy,
            entropy,
            midpoint_temperature_k=1000.0,
        )
        predicted_cp, predicted_h, predicted_s = result.predict(temperatures)

        self.assertIsInstance(result, ContinuousNASA7Fit)
        np.testing.assert_allclose(predicted_cp, cp, rtol=0.0, atol=1.0e-9)
        np.testing.assert_allclose(
            predicted_h,
            enthalpy,
            rtol=0.0,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            predicted_s,
            entropy,
            rtol=0.0,
            atol=1.0e-10,
        )
        self.assertLess(abs(result.continuity["cp_jump_j_mol_k"]), 1.0e-10)
        self.assertLess(abs(result.continuity["h_jump_j_mol"]), 1.0e-8)
        self.assertLess(abs(result.continuity["s_jump_j_mol_k"]), 1.0e-10)

    def test_anchor_values_are_exact(self):
        temperatures, cp, enthalpy, entropy = self._exact_piecewise_data()
        result = fit_continuous_nasa7(
            temperatures,
            cp,
            enthalpy,
            entropy,
            midpoint_temperature_k=1000.0,
        )
        _, anchor_h, anchor_s = result.predict([298.15])
        anchor_index = int(np.flatnonzero(np.isclose(temperatures, 298.15))[0])

        self.assertAlmostEqual(anchor_h[0], enthalpy[anchor_index], places=9)
        self.assertAlmostEqual(anchor_s[0], entropy[anchor_index], places=11)

    def test_missing_anchor_is_rejected(self):
        temperatures, cp, enthalpy, entropy = self._exact_piecewise_data()
        mask = ~np.isclose(temperatures, 298.15)

        with self.assertRaisesRegex(ValueError, "anchor_temperature_k"):
            fit_continuous_nasa7(
                temperatures[mask],
                cp[mask],
                enthalpy[mask],
                entropy[mask],
                midpoint_temperature_k=1000.0,
            )

    def test_each_region_requires_five_cp_points(self):
        temperatures, cp, enthalpy, entropy = self._exact_piecewise_data()
        mask = (temperatures <= 1000.0) | (temperatures >= 1400.0)

        with self.assertRaisesRegex(ValueError, "at least five"):
            fit_continuous_nasa7(
                temperatures[mask],
                cp[mask],
                enthalpy[mask],
                entropy[mask],
                midpoint_temperature_k=1000.0,
            )

    def test_cp_fit_mask_must_be_boolean(self):
        temperatures, cp, enthalpy, entropy = self._exact_piecewise_data()

        with self.assertRaisesRegex(ValueError, "boolean"):
            fit_continuous_nasa7(
                temperatures,
                cp,
                enthalpy,
                entropy,
                midpoint_temperature_k=1000.0,
                cp_fit_mask=np.ones(len(temperatures), dtype=int),
            )

    def test_prediction_refuses_extrapolation(self):
        temperatures, cp, enthalpy, entropy = self._exact_piecewise_data()
        result = fit_continuous_nasa7(
            temperatures,
            cp,
            enthalpy,
            entropy,
            midpoint_temperature_k=1000.0,
        )

        with self.assertRaisesRegex(ValueError, "valid only"):
            result.predict([temperatures[0] - 1.0])

    def test_top_level_exports_continuous_nasa7_api(self):
        self.assertIs(top_level_continuous_nasa7_fit, ContinuousNASA7Fit)
        self.assertIs(top_level_fit_continuous_nasa7, fit_continuous_nasa7)
        self.assertIs(top_level_nasa7_values, nasa7_values)
        self.assertIs(top_level_gas_constant, GAS_CONSTANT_J_MOL_K)


if __name__ == "__main__":
    unittest.main()
