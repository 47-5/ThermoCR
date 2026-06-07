from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ThermoCR.io import (
    read_atom_coordinates,
    read_ee,
    read_electronic_energy,
    read_qm_output,
    read_vibrational_frequencies,
)
from ThermoCR.tools.about_cantera.export_cantera_specie_name_yaml import (
    make_cantera_specie_name_yaml,
)
from ThermoCR.tools.read_qm_out import read_atom_coord, read_qm_out, read_vib


class IOApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example_path = Path(__file__).resolve().parents[1] / "example" / "CPD.out"

    def test_new_and_legacy_qm_readers_parse_link1_last_job(self):
        new_data = read_qm_output(self.example_path)
        legacy_data = read_qm_out(self.example_path)

        self.assertEqual(len(new_data.atomnos), 11)
        self.assertEqual(len(new_data.vibfreqs), 27)
        self.assertEqual(len(legacy_data.atomnos), 11)
        self.assertEqual(len(legacy_data.vibfreqs), 27)

    def test_new_and_legacy_atom_and_vibration_helpers_match(self):
        atom_numbers, coords = read_atom_coordinates(self.example_path)
        legacy_atom_numbers, legacy_coords = read_atom_coord(self.example_path)

        self.assertEqual(list(atom_numbers), list(legacy_atom_numbers))
        self.assertEqual(coords.shape, legacy_coords.shape)
        self.assertEqual(len(read_vibrational_frequencies(self.example_path)), 27)
        self.assertEqual(len(read_vib(self.example_path)), 27)

    def test_legacy_read_ee_accepts_original_keyword_names(self):
        new_energy = read_electronic_energy(self.example_path, energy_index=-1)
        legacy_energy = read_ee(self.example_path, ee_index=-1, return_Hartree=True)

        self.assertAlmostEqual(float(new_energy), float(legacy_energy), places=10)

    def test_cantera_species_name_export_uses_unified_reader(self):
        with TemporaryDirectory() as tmpdir:
            make_cantera_specie_name_yaml(
                specie_name="CPD",
                read_file_path=self.example_path,
                root_path=tmpdir,
            )

            yaml_text = (Path(tmpdir) / "CPD_head.yaml").read_text()

        self.assertIn("- name: CPD", yaml_text)
        self.assertIn("C:5", yaml_text)
        self.assertIn("H:6", yaml_text)


if __name__ == "__main__":
    unittest.main()
