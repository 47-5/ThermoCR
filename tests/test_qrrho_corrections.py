import unittest

import numpy as np

from ThermoCR.constants import R, wave2freq
from ThermoCR.thermo.calculators import contribution_vib
from ThermoCR.thermo.corrections import (
    Cv_vib,
    S_vib,
    S_vib_FR_vec,
    U_vib_T,
    w_vec,
)


class QrrhoCorrectionTests(unittest.TestCase):
    frequency_cm1 = 50.0
    threshold_cm1 = 100.0
    temperature_k = 298.15

    @property
    def expected_weight(self):
        return 1.0 / (
            1.0 + (self.threshold_cm1 / self.frequency_cm1) ** 4
        )

    def test_grimme_entropy_uses_wavenumber_threshold(self):
        rrho_entropy = S_vib(
            [self.frequency_cm1],
            self.temperature_k,
            QRRHO=False,
        )
        free_rotor_entropy = S_vib_FR_vec(
            np.array([self.frequency_cm1 * wave2freq]),
            self.temperature_k,
        )[0]
        expected = (
            self.expected_weight * rrho_entropy
            + (1.0 - self.expected_weight) * free_rotor_entropy
        )

        actual = S_vib(
            [self.frequency_cm1],
            self.temperature_k,
            QRRHO=True,
        )

        self.assertAlmostEqual(actual, expected, places=12)
        self.assertNotAlmostEqual(actual, rrho_entropy, places=6)

    def test_qrrho_weight_matches_reference_wavenumbers(self):
        wavenumbers = np.array([20.0, 100.0, 1000.0])
        expected = 1.0 / (1.0 + (100.0 / wavenumbers) ** 4)

        weights_from_wavenumbers = w_vec(
            wavenumbers,
            convert_unit=False,
        )
        weights_from_hertz = w_vec(wavenumbers * wave2freq)

        np.testing.assert_allclose(
            weights_from_wavenumbers,
            expected,
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            weights_from_hertz,
            expected,
            rtol=0.0,
            atol=1.0e-15,
        )

    def test_qrrho_weight_rejects_nonpositive_or_nonfinite_frequencies(self):
        for frequency in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(frequency=frequency):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    w_vec([frequency], convert_unit=False)

    def test_qrrho_is_invariant_to_frequency_unit_input(self):
        for function in (S_vib, U_vib_T, Cv_vib):
            with self.subTest(function=function.__name__):
                value_from_wavenumber = function(
                    [self.frequency_cm1],
                    self.temperature_k,
                    convert_unit=True,
                    QRRHO=True,
                )
                value_from_hertz = function(
                    [self.frequency_cm1 * wave2freq],
                    self.temperature_k,
                    convert_unit=False,
                    QRRHO=True,
                )

                self.assertAlmostEqual(
                    value_from_wavenumber,
                    value_from_hertz,
                    places=12,
                )

    def test_minenkov_internal_energy_uses_wavenumber_threshold(self):
        rrho_energy = U_vib_T(
            [self.frequency_cm1],
            self.temperature_k,
            QRRHO=False,
        )
        expected = (
            self.expected_weight * rrho_energy
            + (1.0 - self.expected_weight) * R * self.temperature_k / 2.0
        )

        actual = U_vib_T(
            [self.frequency_cm1],
            self.temperature_k,
            QRRHO=True,
        )

        self.assertAlmostEqual(actual, expected, places=9)
        self.assertNotAlmostEqual(actual, rrho_energy, places=6)

    def test_minenkov_heat_capacity_matches_internal_energy_derivative(self):
        delta_temperature = 1.0e-3
        energy_plus = U_vib_T(
            [self.frequency_cm1],
            self.temperature_k + delta_temperature,
            QRRHO=True,
        )
        energy_minus = U_vib_T(
            [self.frequency_cm1],
            self.temperature_k - delta_temperature,
            QRRHO=True,
        )
        numerical_derivative = (
            energy_plus - energy_minus
        ) / (2.0 * delta_temperature)

        heat_capacity = Cv_vib(
            [self.frequency_cm1],
            self.temperature_k,
            QRRHO=True,
        )

        self.assertAlmostEqual(
            heat_capacity,
            numerical_derivative,
            delta=1.0e-6,
        )

    def test_minenkov_scaled_heat_capacity_matches_energy_derivative(self):
        scale_factor = 0.9
        delta_temperature = 1.0e-3
        energy_plus = U_vib_T(
            [self.frequency_cm1],
            self.temperature_k + delta_temperature,
            QRRHO=True,
            scale_factor_U_0_T=scale_factor,
        )
        energy_minus = U_vib_T(
            [self.frequency_cm1],
            self.temperature_k - delta_temperature,
            QRRHO=True,
            scale_factor_U_0_T=scale_factor,
        )
        numerical_derivative = (
            energy_plus - energy_minus
        ) / (2.0 * delta_temperature)

        heat_capacity = Cv_vib(
            [self.frequency_cm1],
            self.temperature_k,
            QRRHO=True,
            scale_factor=scale_factor,
        )

        self.assertAlmostEqual(
            heat_capacity,
            numerical_derivative,
            delta=1.0e-6,
        )

    def test_minimum_rejects_nonpositive_or_nonfinite_modes(self):
        for invalid_frequency in (-20.0, 0.0, np.nan, np.inf):
            with self.subTest(frequency=invalid_frequency):
                with self.assertRaisesRegex(
                    ValueError,
                    "minimum frequencies must be finite and positive",
                ):
                    contribution_vib(
                        [invalid_frequency, self.frequency_cm1],
                        self.temperature_k,
                    )

    def test_transition_state_requires_exactly_one_imaginary_mode(self):
        expected = contribution_vib(
            [self.frequency_cm1],
            self.temperature_k,
        )
        actual = contribution_vib(
            [-500.0, self.frequency_cm1],
            self.temperature_k,
            stationary_point_type="transition_state",
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

        for invalid_frequencies in (
            [self.frequency_cm1],
            [-500.0, -20.0, self.frequency_cm1],
            [-500.0, 0.0, self.frequency_cm1],
        ):
            with self.subTest(frequencies=invalid_frequencies):
                with self.assertRaisesRegex(
                    ValueError,
                    "exactly one imaginary frequency",
                ):
                    contribution_vib(
                        invalid_frequencies,
                        self.temperature_k,
                        stationary_point_type="transition_state",
                    )

    def test_minenkov_option_propagates_to_energy_and_heat_capacity(self):
        contributions = contribution_vib(
            [self.frequency_cm1],
            self.temperature_k,
            U_Minenkov=True,
            S_Grimme=True,
        )
        _, _, _, _, internal_energy, enthalpy, cv, cp, entropy, _ = (
            contributions
        )

        self.assertAlmostEqual(
            internal_energy,
            U_vib_T(
                [self.frequency_cm1],
                self.temperature_k,
                QRRHO=True,
            ),
            places=12,
        )
        self.assertAlmostEqual(enthalpy, internal_energy, places=12)
        self.assertAlmostEqual(
            cv,
            Cv_vib(
                [self.frequency_cm1],
                self.temperature_k,
                QRRHO=True,
            ),
            places=12,
        )
        self.assertAlmostEqual(cp, cv, places=12)
        self.assertAlmostEqual(
            entropy,
            S_vib(
                [self.frequency_cm1],
                self.temperature_k,
                QRRHO=True,
            ),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
