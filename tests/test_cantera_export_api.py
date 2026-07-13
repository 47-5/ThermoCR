from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util
import unittest

import numpy as np
import yaml

from ThermoCR.thermo import nasa7, nasa7_values, nasa9, shomate

from ThermoCR.export import (
    format_cantera_mechanism_yaml,
    format_cantera_reaction_yaml,
    format_cantera_species_yaml,
    format_cantera_yaml_thermo,
    make_cantera_mechanism_yaml,
    make_cantera_reaction_yaml,
    make_cantera_specie_name_yaml,
    write_cantera_yaml_thermo_NASA7,
    write_cantera_yaml_thermo_NASA9,
    write_cantera_yaml_thermo_Shomate,
)
from ThermoCR.tools.about_cantera import (
    format_cantera_mechanism_yaml as package_format_cantera_mechanism_yaml,
    format_cantera_species_yaml as package_format_cantera_species_yaml,
    make_cantera_mechanism_yaml as package_make_cantera_mechanism_yaml,
    make_cantera_reaction_yaml as package_make_cantera_reaction_yaml,
    make_cantera_specie_name_yaml as package_make_cantera_specie_name_yaml,
    write_cantera_yaml_thermo_NASA7 as package_write_cantera_yaml_thermo_NASA7,
)
from ThermoCR.tools.about_cantera.export_cantera_kinetics_yaml import (
    make_cantera_reaction_yaml as legacy_make_cantera_reaction_yaml,
)
from ThermoCR.tools.about_cantera.export_cantera_specie_name_yaml import (
    make_cantera_specie_name_yaml as legacy_make_cantera_specie_name_yaml,
)
from ThermoCR.tools.about_cantera.export_cantera_thermo_yaml import (
    write_cantera_yaml_thermo_NASA7 as legacy_write_cantera_yaml_thermo_NASA7,
)


CANTERA_AVAILABLE = importlib.util.find_spec("cantera") is not None


class CanteraExportApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example_path = Path(__file__).resolve().parents[1] / "example" / "CPD.out"

    def test_new_and_legacy_species_export_parse_qm_output(self):
        with TemporaryDirectory() as tmpdir:
            make_cantera_specie_name_yaml(
                specie_name="CPD",
                read_file_path=self.example_path,
                root_path=tmpdir,
            )
            yaml_text = (Path(tmpdir) / "CPD_head.yaml").read_text(encoding="utf-8")

        self.assertIn("- name: CPD", yaml_text)
        self.assertIn("C: 5", yaml_text)
        self.assertIn("H: 6", yaml_text)
        self.assertIs(make_cantera_specie_name_yaml, legacy_make_cantera_specie_name_yaml)
        self.assertIs(make_cantera_specie_name_yaml, package_make_cantera_specie_name_yaml)

    def test_reaction_export_does_not_mutate_reaction_lists(self):
        reactants = ["A", "B"]
        products = ["C"]

        with TemporaryDirectory() as tmpdir:
            make_cantera_reaction_yaml(
                reactants,
                products,
                A=1.2,
                b=0.5,
                Ea=10.0,
                reversible=False,
                write_mode="w",
                root_path=tmpdir,
            )
            yaml_text = (Path(tmpdir) / "reaction.yaml").read_text(encoding="utf-8")

        self.assertEqual(reactants, ["A", "B"])
        self.assertEqual(products, ["C"])
        self.assertIn("- equation: A + B => C", yaml_text)
        self.assertIn("rate-constant: {A: 1.2, b: 0.5, Ea: 10.0 }", yaml_text)
        self.assertIs(make_cantera_reaction_yaml, legacy_make_cantera_reaction_yaml)
        self.assertIs(make_cantera_reaction_yaml, package_make_cantera_reaction_yaml)

    def test_thermo_writer_legacy_import_points_to_new_api(self):
        with TemporaryDirectory() as tmpdir:
            write_cantera_yaml_thermo_NASA7(
                "CPD",
                T_range=(300.0, 1000.0),
                nasa7_parameters=[1, 2, 3, 4, 5, 6, 7],
                root_path=tmpdir,
            )
            yaml_text = (Path(tmpdir) / "CPD_thermo.yaml").read_text(encoding="utf-8")

        self.assertIn("model: NASA7", yaml_text)
        self.assertIn("temperature-ranges: [300.0, 1000.0]", yaml_text)
        self.assertIn("reference-pressure: 101325 Pa", yaml_text)
        self.assertIs(write_cantera_yaml_thermo_NASA7, legacy_write_cantera_yaml_thermo_NASA7)
        self.assertIs(write_cantera_yaml_thermo_NASA7, package_write_cantera_yaml_thermo_NASA7)

    def test_nasa7_writer_supports_two_regions(self):
        low = [3.5, 0.0, 0.0, 0.0, 0.0, -1000.0, 2.0]
        high = [3.0, 5.0e-4, 0.0, 0.0, 0.0, -750.0, 5.0]

        with TemporaryDirectory() as tmpdir:
            write_cantera_yaml_thermo_NASA7(
                "TWO_REGION",
                T_range=(300.0, 1000.0, 2000.0),
                nasa7_parameters=[low, high],
                root_path=tmpdir,
                reference_pressure_pa=100000.0,
            )
            payload = yaml.safe_load(
                (Path(tmpdir) / "TWO_REGION_thermo.yaml").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            payload["thermo"]["temperature-ranges"],
            [300.0, 1000.0, 2000.0],
        )
        self.assertEqual(payload["thermo"]["data"], [low, high])

    def test_format_cantera_reaction_yaml_matches_writer(self):
        reactants = ["A", "B"]
        products = ["C"]

        text = format_cantera_reaction_yaml(
            reactants,
            products,
            A=1.2,
            b=0.5,
            Ea=10.0,
            reversible=False,
        )

        self.assertEqual(reactants, ["A", "B"])
        self.assertEqual(products, ["C"])
        self.assertIn("- equation: A + B => C", text)
        self.assertIn("rate-constant: {A: 1.2, b: 0.5, Ea: 10.0 }", text)

    def test_format_cantera_yaml_thermo_supports_common_models(self):
        with self.assertWarns(DeprecationWarning):
            nasa9_text = format_cantera_yaml_thermo(
                "nasa9",
                T_range=(300.0, 2000.0),
                parameters=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                reference_p=1,
            )
        shomate_text = format_cantera_yaml_thermo(
            "Shomate",
            T_range=(300.0, 2000.0),
            parameters=[1, 2, 3, 4, 5, 6, 7],
        )

        self.assertIn("model: NASA9", nasa9_text)
        self.assertIn("reference-pressure: 100000 Pa\n   data:", nasa9_text)
        self.assertIn("model: Shomate", shomate_text)
        self.assertIn("reference-pressure: 101325 Pa", shomate_text)

    def test_format_cantera_yaml_thermo_accepts_explicit_pressure_in_pa(self):
        thermo_text = format_cantera_yaml_thermo(
            "NASA7",
            T_range=(300.0, 2000.0),
            parameters=[1, 2, 3, 4, 5, 6, 7],
            reference_pressure_pa=100000.0,
        )

        self.assertIn("reference-pressure: 100000 Pa", thermo_text)

    def test_format_cantera_yaml_thermo_supports_two_region_nasa7(self):
        low = [3.5, 0.0, 0.0, 0.0, 0.0, -1000.0, 2.0]
        high = [
            3.0,
            5.0e-4,
            0.0,
            0.0,
            0.0,
            -750.0,
            1.5 + 0.5 * np.log(1000.0),
        ]

        thermo_text = format_cantera_yaml_thermo(
            "NASA7",
            T_range=(300.0, 1000.0, 2000.0),
            parameters=[low, high],
            reference_pressure_pa=100000.0,
        )
        payload = yaml.safe_load(thermo_text)

        self.assertEqual(
            payload["thermo"]["temperature-ranges"],
            [300.0, 1000.0, 2000.0],
        )
        self.assertEqual(payload["thermo"]["data"], [low, high])

    def test_format_cantera_yaml_thermo_rejects_ambiguous_pressure_arguments(self):
        with self.assertRaisesRegex(ValueError, "reference pressure"):
            format_cantera_yaml_thermo(
                "NASA7",
                T_range=(300.0, 2000.0),
                parameters=[1, 2, 3, 4, 5, 6, 7],
                reference_p=1.0,
                reference_pressure_pa=100000.0,
            )

    def test_legacy_pressure_conversion_rejects_overflow(self):
        with self.assertWarns(DeprecationWarning):
            with self.assertRaisesRegex(ValueError, "finite positive"):
                format_cantera_yaml_thermo(
                    "NASA7",
                    T_range=(300.0, 2000.0),
                    parameters=[1, 2, 3, 4, 5, 6, 7],
                    reference_p=1.0e308,
                )

    def test_legacy_nasa9_and_shomate_writers_keep_one_bar_default(self):
        with TemporaryDirectory() as tmpdir:
            with self.assertWarns(DeprecationWarning):
                write_cantera_yaml_thermo_NASA9(
                    "NASA9_TEST",
                    (300.0, 2000.0),
                    [1] * 9,
                    root_path=tmpdir,
                )
            with self.assertWarns(DeprecationWarning):
                write_cantera_yaml_thermo_Shomate(
                    "SHOMATE_TEST",
                    (300.0, 2000.0),
                    [1] * 7,
                    root_path=tmpdir,
                )
            nasa9_text = (Path(tmpdir) / "NASA9_TEST_thermo.yaml").read_text(
                encoding="utf-8"
            )
            shomate_text = (Path(tmpdir) / "SHOMATE_TEST_thermo.yaml").read_text(
                encoding="utf-8"
            )

        self.assertIn("reference-pressure: 100000 Pa", nasa9_text)
        self.assertIn("reference-pressure: 100000 Pa", shomate_text)

    def test_format_cantera_yaml_thermo_validates_single_region_shape(self):
        with self.assertRaisesRegex(ValueError, "7 parameters"):
            format_cantera_yaml_thermo(
                "NASA7",
                T_range=(300.0, 2000.0),
                parameters=[1, 2, 3],
            )
        with self.assertRaisesRegex(ValueError, "two increasing"):
            format_cantera_yaml_thermo(
                "NASA9",
                T_range=(2000.0, 300.0),
                parameters=[1] * 9,
            )

    def test_format_cantera_yaml_thermo_validates_two_region_nasa7_shape(self):
        with self.assertRaisesRegex(ValueError, "two NASA7 coefficient regions"):
            format_cantera_yaml_thermo(
                "NASA7",
                T_range=(300.0, 1000.0, 2000.0),
                parameters=[1] * 7,
            )
        with self.assertRaisesRegex(ValueError, "three increasing"):
            format_cantera_yaml_thermo(
                "NASA7",
                T_range=(300.0, 2000.0, 1000.0),
                parameters=[[1] * 7, [1] * 7],
            )

    def test_format_cantera_mechanism_yaml_combines_fragments(self):
        species = format_cantera_species_yaml(
            "- name: CPD\n  composition: {C:5, H:6}\n",
            format_cantera_yaml_thermo(
                "NASA7",
                T_range=(300.0, 1000.0),
                parameters=[1, 2, 3, 4, 5, 6, 7],
            ),
        )
        reaction = format_cantera_reaction_yaml(
            ["CPD"],
            ["CPD"],
            A=1.0,
            b=0.0,
            Ea=0.0,
        )

        yaml_text = format_cantera_mechanism_yaml(
            [species],
            reaction_blocks=[reaction],
        )

        self.assertIn("phases:", yaml_text)
        self.assertIn("species: [CPD]", yaml_text)
        self.assertIn("elements: [C, H]", yaml_text)
        self.assertIn("- name: CPD", yaml_text)
        self.assertIn("reactions:", yaml_text)
        self.assertIn("- equation: CPD <=> CPD", yaml_text)
        payload = yaml.safe_load(yaml_text)
        self.assertEqual(payload["species"][0]["composition"], {"C": 5, "H": 6})
        self.assertIn("thermo", payload["species"][0])
        self.assertNotIn("thermo", payload)
        self.assertIs(format_cantera_mechanism_yaml, package_format_cantera_mechanism_yaml)
        self.assertIs(format_cantera_species_yaml, package_format_cantera_species_yaml)

    def test_make_cantera_mechanism_yaml_writes_output(self):
        species = format_cantera_species_yaml(
            "- name: A\n  composition: {C:1}\n",
            format_cantera_yaml_thermo(
                "NASA7",
                T_range=(300.0, 1000.0),
                parameters=[1, 2, 3, 4, 5, 6, 7],
            ),
        )

        with TemporaryDirectory() as tmpdir:
            make_cantera_mechanism_yaml(
                [species],
                yaml_name="mechanism.yaml",
                root_path=tmpdir,
            )
            yaml_text = (Path(tmpdir) / "mechanism.yaml").read_text(encoding="utf-8")

        self.assertIn("reactions: none", yaml_text)
        self.assertIn("species: [A]", yaml_text)
        self.assertIs(make_cantera_mechanism_yaml, package_make_cantera_mechanism_yaml)


@unittest.skipUnless(CANTERA_AVAILABLE, "Cantera is not installed")
class CanteraThermoRoundTripTests(unittest.TestCase):
    def _load_single_species(
        self,
        model,
        parameters,
        reference_pressure_pa,
        temperature_range=(300.0, 2000.0),
    ):
        import cantera as ct

        species = format_cantera_species_yaml(
            "- name: test-species\n  composition: {H: 2}\n",
            format_cantera_yaml_thermo(
                model,
                T_range=temperature_range,
                parameters=parameters,
                reference_pressure_pa=reference_pressure_pa,
            ),
        )
        mechanism = format_cantera_mechanism_yaml([species])
        with TemporaryDirectory() as tmpdir:
            mechanism_path = Path(tmpdir) / "mechanism.yaml"
            mechanism_path.write_text(mechanism, encoding="utf-8")
            return ct.Solution(str(mechanism_path))

    def test_nasa7_properties_survive_cantera_round_trip(self):
        parameters = (3.5, 1.0e-3, -2.0e-7, 0.0, 0.0, -1000.0, 2.0)
        reference_pressure_pa = 100000.0
        gas = self._load_single_species("NASA7", parameters, reference_pressure_pa)

        self.assertAlmostEqual(gas.reference_pressure, reference_pressure_pa)
        for temperature in (400.0, 900.0, 1500.0):
            with self.subTest(temperature=temperature):
                gas.TP = temperature, reference_pressure_pa
                expected_cp, expected_h, expected_s = nasa7(temperature, *parameters)
                self.assertAlmostEqual(gas.cp_mole / 1000.0, expected_cp, delta=1.0e-4)
                self.assertAlmostEqual(gas.enthalpy_mole / 1000.0, expected_h, delta=0.02)
                self.assertAlmostEqual(gas.entropy_mole / 1000.0, expected_s, delta=1.0e-4)

    def test_two_region_nasa7_survives_cantera_round_trip(self):
        low = (3.5, 0.0, 0.0, 0.0, 0.0, -1000.0, 2.0)
        high = (
            3.0,
            5.0e-4,
            0.0,
            0.0,
            0.0,
            -750.0,
            1.5 + 0.5 * np.log(1000.0),
        )
        reference_pressure_pa = 100000.0
        gas = self._load_single_species(
            "NASA7",
            [low, high],
            reference_pressure_pa,
            temperature_range=(300.0, 1000.0, 2000.0),
        )

        for temperature, coefficients in (
            (400.0, low),
            (1000.0, low),
            (1500.0, high),
        ):
            with self.subTest(temperature=temperature):
                gas.TP = temperature, reference_pressure_pa
                expected_cp, expected_h, expected_s = nasa7_values(
                    temperature,
                    coefficients,
                )
                self.assertAlmostEqual(
                    gas.cp_mole / 1000.0,
                    expected_cp[0],
                    delta=1.0e-4,
                )
                self.assertAlmostEqual(
                    gas.enthalpy_mole / 1000.0,
                    expected_h[0],
                    delta=0.02,
                )
                self.assertAlmostEqual(
                    gas.entropy_mole / 1000.0,
                    expected_s[0],
                    delta=1.0e-4,
                )

    def test_nasa9_properties_survive_cantera_round_trip(self):
        parameters = (1.0e4, -10.0, 3.5, 1.0e-3, -2.0e-7, 0.0, 0.0, -1000.0, 2.0)
        reference_pressure_pa = 100000.0
        gas = self._load_single_species("NASA9", parameters, reference_pressure_pa)

        self.assertAlmostEqual(gas.reference_pressure, reference_pressure_pa)
        for temperature in (400.0, 900.0, 1500.0):
            with self.subTest(temperature=temperature):
                gas.TP = temperature, reference_pressure_pa
                expected_cp, expected_h, expected_s = nasa9(temperature, *parameters)
                self.assertAlmostEqual(gas.cp_mole / 1000.0, expected_cp, delta=1.0e-4)
                self.assertAlmostEqual(gas.enthalpy_mole / 1000.0, expected_h, delta=0.02)
                self.assertAlmostEqual(gas.entropy_mole / 1000.0, expected_s, delta=1.0e-4)

    def test_shomate_properties_survive_cantera_round_trip(self):
        parameters = (30.0, 2.0, -0.5, 0.1, 0.0, -10.0, 200.0)
        reference_pressure_pa = 100000.0
        gas = self._load_single_species("Shomate", parameters, reference_pressure_pa)

        self.assertAlmostEqual(gas.reference_pressure, reference_pressure_pa)
        for temperature in (400.0, 900.0, 1500.0):
            with self.subTest(temperature=temperature):
                gas.TP = temperature, reference_pressure_pa
                expected_cp, expected_h, expected_s = shomate(temperature, *parameters)
                self.assertAlmostEqual(gas.cp_mole / 1000.0, expected_cp, places=10)
                self.assertAlmostEqual(gas.enthalpy_mole / 1000.0, expected_h, places=8)
                self.assertAlmostEqual(gas.entropy_mole / 1000.0, expected_s, places=10)

    def test_ideal_gas_entropy_changes_with_pressure_but_cp_and_h_do_not(self):
        import cantera as ct

        gas = self._load_single_species(
            "NASA7",
            (3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0),
            100000.0,
        )
        gas.TP = 900.0, gas.reference_pressure
        cp_reference = gas.cp_mole
        h_reference = gas.enthalpy_mole
        s_reference = gas.entropy_mole

        gas.TP = 900.0, 2.0 * gas.reference_pressure

        self.assertAlmostEqual(gas.cp_mole, cp_reference, places=8)
        self.assertAlmostEqual(gas.enthalpy_mole, h_reference, places=6)
        self.assertAlmostEqual(
            gas.entropy_mole,
            s_reference - ct.gas_constant * np.log(2.0),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
