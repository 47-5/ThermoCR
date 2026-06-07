from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ThermoCR.io import read_orca_final_single_point_energy
from ThermoCR.io.orca import sort_key
from ThermoCR.tools.about_orca import (
    read_orca_final_single_point_energy as package_read_orca_final_single_point_energy,
    read_orca_wB97Mp2_out as package_read_orca_wB97Mp2_out,
    sort_key as package_sort_key,
)
from ThermoCR.tools.about_orca.read_orca_wB97Mp2_out import (
    read_orca_final_single_point_energy as legacy_read_orca_final_single_point_energy,
    read_orca_wB97Mp2_out,
    sort_key as legacy_sort_key,
)


class OrcaApiTests(unittest.TestCase):
    def test_new_and_legacy_orca_energy_readers_match(self):
        self.assertIs(
            read_orca_final_single_point_energy,
            legacy_read_orca_final_single_point_energy,
        )
        self.assertIs(sort_key, legacy_sort_key)
        self.assertIs(read_orca_final_single_point_energy, package_read_orca_final_single_point_energy)
        self.assertIs(read_orca_wB97Mp2_out, package_read_orca_wB97Mp2_out)
        self.assertIs(sort_key, package_sort_key)

    def test_orca_reader_returns_last_final_single_point_energy(self):
        text = """
        FINAL SINGLE POINT ENERGY     -10.125000
        other line
        FINAL SINGLE POINT ENERGY     -11.250000
        """
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "orca.out"
            output_path.write_text(text)

            energy = read_orca_final_single_point_energy(output_path)
            legacy_energy = read_orca_wB97Mp2_out(output_path)

        self.assertEqual(energy, -11.25)
        self.assertEqual(legacy_energy, -11.25)

    def test_orca_sort_key_uses_historical_path_pattern(self):
        self.assertEqual(sort_key("01_02_path3_04.out"), (1, 2, 3, 4))
        self.assertEqual(sort_key("unmatched.out"), (float("inf"),) * 4)


if __name__ == "__main__":
    unittest.main()