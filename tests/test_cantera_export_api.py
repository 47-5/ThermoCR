from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ThermoCR.export import (
    make_cantera_reaction_yaml,
    make_cantera_specie_name_yaml,
    write_cantera_yaml_thermo_NASA7,
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
        self.assertIn("C:5", yaml_text)
        self.assertIn("H:6", yaml_text)
        self.assertIs(make_cantera_specie_name_yaml, legacy_make_cantera_specie_name_yaml)

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


if __name__ == "__main__":
    unittest.main()