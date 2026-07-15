from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from ThermoCR import (
    EnergyConvention as top_level_EnergyConvention,
    SpeciesThermoArtifact as top_level_SpeciesThermoArtifact,
    anchor_enthalpy_curve as top_level_anchor_enthalpy_curve,
)
from ThermoCR.thermo import (
    EnergyConvention,
    SpeciesThermoArtifact,
    anchor_enthalpy_curve,
    assert_energy_conventions_compatible,
    fit_continuous_nasa7,
)
from ThermoCR.thermo.fitting import ThermoFitResult


class EnthalpyAnchoringTests(unittest.TestCase):
    def test_anchor_enthalpy_curve_preserves_increments_and_sets_reference(self):
        temperatures = np.array([298.15, 400.0, 500.0])
        absolute_enthalpies = np.array([-1000.0, -500.0, 0.0])

        anchored = anchor_enthalpy_curve(
            temperatures,
            absolute_enthalpies,
            target_hf298=20000.0,
        )

        np.testing.assert_allclose(anchored, [20000.0, 20500.0, 21000.0])
        np.testing.assert_allclose(
            np.diff(anchored),
            np.diff(absolute_enthalpies),
        )

    def test_anchor_enthalpy_curve_interpolates_without_reordering_output(self):
        temperatures = np.array([400.0, 300.0, 500.0])
        absolute_enthalpies = 2.0 * temperatures + 100.0

        anchored = anchor_enthalpy_curve(
            temperatures,
            absolute_enthalpies,
            target_hf298=-5000.0,
            reference_temperature=350.0,
        )

        expected = absolute_enthalpies - 800.0 - 5000.0
        np.testing.assert_allclose(anchored, expected)

    def test_anchor_enthalpy_curve_rejects_extrapolation_and_duplicate_temperatures(self):
        with self.assertRaisesRegex(ValueError, "inside the temperature range"):
            anchor_enthalpy_curve([300.0, 400.0], [1.0, 2.0], 0.0, 298.15)
        with self.assertRaisesRegex(ValueError, "unique"):
            anchor_enthalpy_curve([300.0, 300.0], [1.0, 2.0], 0.0, 300.0)


class SpeciesThermoArtifactTests(unittest.TestCase):
    @staticmethod
    def _continuous_nasa7_fit():
        temperatures = np.array(
            sorted(list(np.arange(200.0, 3000.1, 100.0)) + [298.15])
        )
        heat_capacity = np.full_like(temperatures, 35.0)
        enthalpy = -74600.0 + 35.0 * (temperatures - 298.15)
        entropy = 186.25 + 35.0 * np.log(temperatures / 298.15)
        return fit_continuous_nasa7(
            temperatures,
            heat_capacity,
            enthalpy,
            entropy,
            midpoint_temperature_k=1000.0,
            reference_pressure_pa=100000.0,
            anchor_temperature_k=298.15,
        )

    def _formation_artifact(self, **overrides):
        values = {
            "species_id": "methane",
            "cantera_name": "CH4",
            "composition": {"C": 1, "H": 4},
            "energy_convention": EnergyConvention.FORMATION_ENTHALPY,
            "formation_enthalpy_298_j_mol": -74600.0,
            "formation_enthalpy_uncertainty_j_mol": 100.0,
            "canonical_smiles": "C",
            "reference_pressure_pa": 100000.0,
            "temperature_grid_k": (298.15, 400.0, 500.0),
            "fit": {
                "model": "NASA7",
                "regions": [[298.15, 500.0]],
                "coefficients": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
            },
            "fit_metrics": {"enthalpy": {"mae": 1.0}},
            "source_sha256": "a" * 64,
        }
        values.update(overrides)
        return SpeciesThermoArtifact(**values)

    def test_artifact_json_round_trip_preserves_scientific_contract(self):
        artifact = self._formation_artifact()

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "CH4.thermo.json"
            artifact.write_json(path)
            restored = SpeciesThermoArtifact.read_json(path)

        self.assertEqual(restored, artifact)
        self.assertEqual(restored.energy_convention, EnergyConvention.FORMATION_ENTHALPY)
        self.assertEqual(restored.reference_pressure_pa, 100000.0)
        self.assertEqual(restored.table_units["enthalpy"], "J/mol")
        self.assertEqual(restored.to_dict()["energy_convention"], "formation_enthalpy")

    def test_artifact_can_be_constructed_from_thermo_fit_result(self):
        fit_result = ThermoFitResult(
            model_type="NASA7",
            parameters=(1, 2, 3, 4, 5, 6, 7),
            temperature_range=(298.15, 1000.0),
            metrics={"enthalpy": {"mae": 2.0}},
            covariance=np.eye(7),
            reference_pressure_pa=100000.0,
        )

        artifact = SpeciesThermoArtifact.from_fit_result(
            species_id="methane",
            cantera_name="CH4",
            composition={"C": 1, "H": 4},
            energy_convention="formation_enthalpy",
            fit_result=fit_result,
            formation_enthalpy_298_j_mol=-74600.0,
            temperature_grid_k=(298.15, 400.0, 1000.0),
        )

        self.assertEqual(artifact.reference_pressure_pa, 100000.0)
        self.assertEqual(artifact.fit["model"], "NASA7")
        self.assertEqual(artifact.fit["regions"], [[298.15, 1000.0]])
        self.assertEqual(artifact.fit["coefficients"], [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        self.assertEqual(artifact.fit["covariance"][0][0], 1.0)
        self.assertEqual(artifact.fit_metrics["enthalpy"]["mae"], 2.0)

    def test_artifact_can_be_constructed_from_continuous_nasa7_fit(self):
        fit_result = self._continuous_nasa7_fit()

        artifact = top_level_SpeciesThermoArtifact.from_continuous_nasa7_fit(
            species_id="methane",
            cantera_name="CH4",
            composition={"C": 1, "H": 4},
            energy_convention="formation_enthalpy",
            fit_result=fit_result,
            formation_enthalpy_298_j_mol=-74600.0,
            formation_enthalpy_uncertainty_j_mol=100.0,
            temperature_grid_k=(200.0, 298.15, 1000.0, 3000.0),
            source_sha256="a" * 64,
        )

        self.assertEqual(artifact.reference_temperature_k, 298.15)
        self.assertEqual(artifact.reference_pressure_pa, 100000.0)
        self.assertEqual(
            artifact.fit,
            {
                "model": "NASA7",
                "regions": [[200.0, 1000.0], [1000.0, 3000.0]],
                "coefficients": [
                    list(fit_result.low_coefficients),
                    list(fit_result.high_coefficients),
                ],
                "anchor_temperature_k": 298.15,
                "anchor_enthalpy_j_mol": -74600.0,
                "anchor_entropy_j_mol_k": 186.25,
                "cp_fit_point_count": fit_result.cp_fit_point_count,
                "diagnostics": fit_result.diagnostics,
            },
        )
        self.assertEqual(artifact.fit_metrics, fit_result.metrics)
        self.assertEqual(artifact.continuity_metrics, fit_result.continuity)
        self.assertEqual(artifact.source_sha256, "a" * 64)
        self.assertEqual(
            artifact.cantera_species_dict,
            {
                "name": "CH4",
                "composition": {"C": 1, "H": 4},
                "thermo": {
                    "model": "NASA7",
                    "reference-pressure": "100000 Pa",
                    "temperature-ranges": [200.0, 1000.0, 3000.0],
                    "data": [
                        list(fit_result.low_coefficients),
                        list(fit_result.high_coefficients),
                    ],
                },
            },
        )

    def test_continuous_nasa7_artifact_requires_exact_fit_type(self):
        fit_result = ThermoFitResult(
            model_type="NASA7",
            parameters=(1, 2, 3, 4, 5, 6, 7),
            temperature_range=(298.15, 1000.0),
            metrics={},
            covariance=None,
            reference_pressure_pa=100000.0,
        )

        with self.assertRaisesRegex(TypeError, "ContinuousNASA7Fit"):
            SpeciesThermoArtifact.from_continuous_nasa7_fit(
                species_id="methane",
                cantera_name="CH4",
                composition={"C": 1, "H": 4},
                energy_convention="formation_enthalpy",
                fit_result=fit_result,
                formation_enthalpy_298_j_mol=-74600.0,
            )

    def test_continuous_nasa7_artifact_rejects_pressure_override(self):
        with self.assertRaisesRegex(
            ValueError,
            "metadata cannot override fit-derived fields",
        ):
            SpeciesThermoArtifact.from_continuous_nasa7_fit(
                species_id="methane",
                cantera_name="CH4",
                composition={"C": 1, "H": 4},
                energy_convention="formation_enthalpy",
                fit_result=self._continuous_nasa7_fit(),
                formation_enthalpy_298_j_mol=-74600.0,
                reference_pressure_pa=101325.0,
            )

    def test_continuous_nasa7_artifact_protects_fit_derived_metadata(self):
        base_arguments = {
            "species_id": "methane",
            "cantera_name": "CH4",
            "composition": {"C": 1, "H": 4},
            "energy_convention": "formation_enthalpy",
            "fit_result": self._continuous_nasa7_fit(),
            "formation_enthalpy_298_j_mol": -74600.0,
        }

        for field_name in (
            "reference_temperature_k",
            "fit",
            "fit_metrics",
            "continuity_metrics",
            "cantera_species_dict",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(
                    ValueError,
                    "metadata cannot override fit-derived fields",
                ):
                    SpeciesThermoArtifact.from_continuous_nasa7_fit(
                        **base_arguments,
                        **{field_name: {}},
                    )

    def test_continuous_nasa7_formation_enthalpy_must_match_fit_anchor(self):
        with self.assertRaisesRegex(
            ValueError,
            "formation_enthalpy_298_j_mol must match",
        ):
            SpeciesThermoArtifact.from_continuous_nasa7_fit(
                species_id="methane",
                cantera_name="CH4",
                composition={"C": 1, "H": 4},
                energy_convention="formation_enthalpy",
                fit_result=self._continuous_nasa7_fit(),
                formation_enthalpy_298_j_mol=-74500.0,
            )

    def test_formation_enthalpy_convention_requires_anchor(self):
        with self.assertRaisesRegex(ValueError, "formation_enthalpy_298_j_mol"):
            self._formation_artifact(formation_enthalpy_298_j_mol=None)

    def test_absolute_qm_convention_requires_energy_reference_id(self):
        with self.assertRaisesRegex(ValueError, "energy_reference_id"):
            SpeciesThermoArtifact(
                species_id="methane",
                cantera_name="CH4",
                composition={"C": 1, "H": 4},
                energy_convention=EnergyConvention.ABSOLUTE_QM,
            )

    def test_nonformation_convention_rejects_formation_enthalpy_value(self):
        with self.assertRaisesRegex(ValueError, "must be omitted"):
            SpeciesThermoArtifact(
                species_id="methane",
                cantera_name="CH4",
                composition={"C": 1, "H": 4},
                energy_convention=EnergyConvention.THERMAL_INCREMENT_ONLY,
                formation_enthalpy_298_j_mol=-74600.0,
            )

    def test_incompatible_energy_conventions_are_rejected(self):
        formation = self._formation_artifact()
        absolute = SpeciesThermoArtifact(
            species_id="methane-qm",
            cantera_name="CH4_QM",
            composition={"C": 1, "H": 4},
            energy_convention=EnergyConvention.ABSOLUTE_QM,
            energy_reference_id="wb97m-v/def2-tzvppd|atoms-v1",
        )

        with self.assertRaisesRegex(ValueError, "incompatible energy conventions"):
            assert_energy_conventions_compatible([formation, absolute])

    def test_absolute_qm_artifacts_require_same_energy_reference(self):
        first = SpeciesThermoArtifact(
            species_id="A",
            cantera_name="A",
            composition={"C": 1},
            energy_convention="absolute_qm",
            energy_reference_id="method-a|atoms-v1",
        )
        second = SpeciesThermoArtifact(
            species_id="B",
            cantera_name="B",
            composition={"C": 2},
            energy_convention="absolute_qm",
            energy_reference_id="method-b|atoms-v1",
        )

        with self.assertRaisesRegex(ValueError, "energy_reference_id"):
            assert_energy_conventions_compatible([first, second])

    def test_compatible_absolute_qm_artifacts_can_share_one_zero(self):
        artifacts = [
            SpeciesThermoArtifact(
                species_id=name,
                cantera_name=name,
                composition={"C": carbon_count},
                energy_convention="absolute_qm",
                energy_reference_id="method-a|atoms-v1",
            )
            for name, carbon_count in (("A", 1), ("B", 2))
        ]

        self.assertIsNone(
            assert_energy_conventions_compatible(
                artifacts,
                require_equilibrium_ready=True,
            )
        )

    def test_artifacts_require_same_reference_pressure(self):
        one_bar = self._formation_artifact()
        one_atm = self._formation_artifact(
            species_id="methane-atm",
            cantera_name="CH4_ATM",
            reference_pressure_pa=101325.0,
        )

        with self.assertRaisesRegex(ValueError, "reference pressures"):
            assert_energy_conventions_compatible([one_bar, one_atm])

    def test_artifact_requires_si_table_units(self):
        with self.assertRaisesRegex(ValueError, "J/mol"):
            self._formation_artifact(
                table_units={
                    "temperature": "K",
                    "heat_capacity_cp": "J/mol/K",
                    "enthalpy": "kJ/mol",
                    "entropy": "J/mol/K",
                }
            )

    def test_thermal_increment_artifact_is_not_equilibrium_ready(self):
        artifact = SpeciesThermoArtifact(
            species_id="increment",
            cantera_name="increment",
            composition={"H": 2},
            energy_convention=EnergyConvention.THERMAL_INCREMENT_ONLY,
        )

        with self.assertRaisesRegex(ValueError, "cannot be used for equilibrium"):
            assert_energy_conventions_compatible(
                [artifact],
                require_equilibrium_ready=True,
            )

    def test_artifacts_for_one_phase_require_same_phase_label(self):
        gas = self._formation_artifact()
        liquid = self._formation_artifact(
            species_id="liquid-methane",
            cantera_name="CH4_L",
            phase="liquid",
        )

        with self.assertRaisesRegex(ValueError, "phase labels"):
            assert_energy_conventions_compatible([gas, liquid])

    def test_top_level_exports_artifact_api(self):
        self.assertIs(top_level_EnergyConvention, EnergyConvention)
        self.assertIs(top_level_SpeciesThermoArtifact, SpeciesThermoArtifact)
        self.assertIs(top_level_anchor_enthalpy_curve, anchor_enthalpy_curve)


if __name__ == "__main__":
    unittest.main()
