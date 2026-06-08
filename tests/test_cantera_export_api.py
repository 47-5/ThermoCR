from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from ThermoCR.export import (
    format_cantera_mechanism_yaml,
    format_cantera_reaction_yaml,
    format_cantera_species_yaml,
    format_cantera_yaml_thermo,
    make_cantera_mechanism_yaml,
    make_cantera_reaction_yaml,
    make_cantera_specie_name_yaml,
    write_cantera_yaml_thermo_NASA7,
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
        self.assertIs(write_cantera_yaml_thermo_NASA7, legacy_write_cantera_yaml_thermo_NASA7)
        self.assertIs(write_cantera_yaml_thermo_NASA7, package_write_cantera_yaml_thermo_NASA7)

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
        self.assertIn("reference-pressure: 1 bar\n   data:", nasa9_text)
        self.assertIn("model: Shomate", shomate_text)

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


if __name__ == "__main__":
    unittest.main()
