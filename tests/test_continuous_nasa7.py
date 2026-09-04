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
    @classmethod
    def _valid_fit_kwargs(cls):
        midpoint = 1000.0
        anchor = 298.15
        low, high = cls._exact_piecewise_coefficients()
        _, anchor_h, anchor_s = nasa7_values([anchor], low)
        low_midpoint = nasa7_values([midpoint], low)
        high_midpoint = nasa7_values([midpoint], high)
        return {
            "temperature_range_k": (200.0, 1500.0),
            "midpoint_temperature_k": midpoint,
            "anchor_temperature_k": anchor,
            "anchor_enthalpy_j_mol": float(anchor_h[0]),
            "anchor_entropy_j_mol_k": float(anchor_s[0]),
            "reference_pressure_pa": 101325.0,
            "cp_fit_point_count": 28,
            "low_coefficients": tuple(low),
            "high_coefficients": tuple(high),
            "metrics": {"heat_capacity_cp": {"mae": 0.1}},
            "continuity": {
                name: float(high_value[0] - low_value[0])
                for name, low_value, high_value in zip(
                    (
                        "cp_jump_j_mol_k",
                        "h_jump_j_mol",
                        "s_jump_j_mol_k",
                    ),
                    low_midpoint,
                    high_midpoint,
                )
            },
            "diagnostics": {"algorithm": "test"},
        }

    @classmethod
    def _exact_piecewise_coefficients(cls):
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
        high[5] = cls._enthalpy_constant(
            high[:5],
            midpoint,
            low_midpoint[1][0],
        )
        high[6] = cls._entropy_constant(
            high[:5],
            midpoint,
            low_midpoint[2][0],
        )
        return low, high

    def _exact_piecewise_data(self):
        midpoint = 1000.0
        low, high = self._exact_piecewise_coefficients()
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
            reference_pressure_pa=101325.0,
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
            reference_pressure_pa=101325.0,
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
                reference_pressure_pa=101325.0,
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
                reference_pressure_pa=101325.0,
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
                reference_pressure_pa=101325.0,
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
            reference_pressure_pa=101325.0,
        )

        with self.assertRaisesRegex(ValueError, "valid only"):
            result.predict([temperatures[0] - 1.0])

    def test_fit_object_normalizes_core_numeric_sequences(self):
        values = self._valid_fit_kwargs()
        values["temperature_range_k"] = [200, 1500]
        values["low_coefficients"] = np.asarray(
            values["low_coefficients"],
            dtype=np.float64,
        )

        result = ContinuousNASA7Fit(**values)

        self.assertEqual(result.temperature_range_k, (200.0, 1500.0))
        self.assertEqual(
            result.low_coefficients,
            tuple(float(value) for value in values["low_coefficients"]),
        )
        self.assertEqual(result.reference_pressure_pa, 101325.0)
        self.assertEqual(result.to_dict()["reference_pressure_pa"], 101325.0)

    def test_fit_function_requires_explicit_reference_pressure(self):
        temperatures, cp, enthalpy, entropy = self._exact_piecewise_data()

        with self.assertRaisesRegex(TypeError, "reference_pressure_pa"):
            fit_continuous_nasa7(
                temperatures,
                cp,
                enthalpy,
                entropy,
                midpoint_temperature_k=1000.0,
            )

    def test_fit_object_rejects_invalid_reference_pressure(self):
        for invalid_value in (0.0, -1.0, float("nan")):
            with self.subTest(invalid_value=invalid_value):
                values = self._valid_fit_kwargs()
                values["reference_pressure_pa"] = invalid_value
                with self.assertRaisesRegex(ValueError, "reference pressure"):
                    ContinuousNASA7Fit(**values)

    def test_fit_object_rejects_anchor_not_reproduced_by_coefficients(self):
        for field_name in (
            "anchor_enthalpy_j_mol",
            "anchor_entropy_j_mol_k",
        ):
            with self.subTest(field_name=field_name):
                values = self._valid_fit_kwargs()
                values[field_name] += 1.0
                with self.assertRaisesRegex(ValueError, "do not reproduce"):
                    ContinuousNASA7Fit(**values)

    def test_fit_object_rejects_discontinuous_coefficients(self):
        values = self._valid_fit_kwargs()
        high = list(values["high_coefficients"])
        high[0] += 0.1
        values["high_coefficients"] = high

        with self.assertRaisesRegex(ValueError, "continuity tolerance"):
            ContinuousNASA7Fit(**values)

    def test_fit_object_rejects_false_continuity_declaration(self):
        values = self._valid_fit_kwargs()
        values["continuity"]["cp_jump_j_mol_k"] = 1.0

        with self.assertRaisesRegex(ValueError, "does not match"):
            ContinuousNASA7Fit(**values)

    def test_fit_object_rejects_invalid_temperature_structure(self):
        invalid_overrides = (
            {"temperature_range_k": (0.0, 1500.0)},
            {"temperature_range_k": (1500.0, 200.0)},
            {"temperature_range_k": (200.0, float("inf"))},
            {"temperature_range_k": (200.0, 1000.0, 1500.0)},
            {"midpoint_temperature_k": 200.0},
            {"midpoint_temperature_k": 1500.0},
            {"midpoint_temperature_k": float("nan")},
            {"anchor_temperature_k": 150.0},
            {"anchor_temperature_k": 1100.0},
            {"anchor_temperature_k": float("inf")},
        )
        for override in invalid_overrides:
            with self.subTest(override=override):
                values = self._valid_fit_kwargs()
                values.update(override)
                with self.assertRaises(ValueError):
                    ContinuousNASA7Fit(**values)

    def test_fit_object_rejects_invalid_anchor_properties(self):
        for field_name, invalid_value in (
            ("anchor_enthalpy_j_mol", float("nan")),
            ("anchor_entropy_j_mol_k", float("inf")),
        ):
            with self.subTest(field_name=field_name):
                values = self._valid_fit_kwargs()
                values[field_name] = invalid_value
                with self.assertRaisesRegex(ValueError, field_name):
                    ContinuousNASA7Fit(**values)

    def test_fit_object_requires_positive_integer_cp_fit_point_count(self):
        for invalid_value in (0, -1, 2.0, True):
            with self.subTest(invalid_value=invalid_value):
                values = self._valid_fit_kwargs()
                values["cp_fit_point_count"] = invalid_value
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    ContinuousNASA7Fit(**values)

    def test_fit_object_requires_seven_finite_coefficients_per_region(self):
        for field_name in ("low_coefficients", "high_coefficients"):
            for invalid_value in (
                (1.0,) * 6,
                (1.0,) * 6 + (float("nan"),),
            ):
                with self.subTest(field_name=field_name, invalid_value=invalid_value):
                    values = self._valid_fit_kwargs()
                    values[field_name] = invalid_value
                    with self.assertRaisesRegex(ValueError, field_name):
                        ContinuousNASA7Fit(**values)

    def test_fit_object_requires_metadata_mappings_with_finite_numbers(self):
        invalid_overrides = (
            {"metrics": []},
            {"continuity": None},
            {"diagnostics": "test"},
            {"metrics": {"future_metric": {"values": [1.0, float("nan")]}}},
            {"continuity": {"future_jump": float("inf")}},
            {"diagnostics": {"future": {"condition_number": float("nan")}}},
        )
        for override in invalid_overrides:
            with self.subTest(override=override):
                values = self._valid_fit_kwargs()
                values.update(override)
                with self.assertRaises(ValueError):
                    ContinuousNASA7Fit(**values)

    def test_fit_object_allows_unknown_finite_metadata_extensions(self):
        values = self._valid_fit_kwargs()
        values["metrics"]["future_metric"] = {
            "values": [1.0, 2.0],
            "method": "extension",
        }
        values["diagnostics"]["future_metadata"] = {"enabled": True}

        result = ContinuousNASA7Fit(**values)

        self.assertEqual(
            result.metrics["future_metric"]["method"],
            "extension",
        )

    def test_top_level_exports_continuous_nasa7_api(self):
        self.assertIs(top_level_continuous_nasa7_fit, ContinuousNASA7Fit)
        self.assertIs(top_level_fit_continuous_nasa7, fit_continuous_nasa7)
        self.assertIs(top_level_nasa7_values, nasa7_values)
        self.assertIs(top_level_gas_constant, GAS_CONSTANT_J_MOL_K)


if __name__ == "__main__":
    unittest.main()
