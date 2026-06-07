import unittest

from ThermoCR import io
from ThermoCR.tools import (
    R,
    calculate_solvent_energy,
    gaussian_sort_key,
    get_point_group,
    read_orca_wB97Mp2_out,
    read_qm_out,
    select_gaussian_out,
    sort_key,
)
from ThermoCR.tools.about_gaussian import sort_key as gaussian_package_sort_key
from ThermoCR.tools.about_orca import sort_key as orca_package_sort_key


class ToolsPackageApiTests(unittest.TestCase):
    def test_tools_package_root_exports_legacy_helpers(self):
        self.assertGreater(R, 0.0)
        self.assertTrue(callable(read_qm_out))
        self.assertTrue(callable(get_point_group))
        self.assertTrue(callable(select_gaussian_out))
        self.assertTrue(callable(calculate_solvent_energy))
        self.assertTrue(callable(read_orca_wB97Mp2_out))

    def test_tools_sort_key_keeps_legacy_orca_precedence(self):
        self.assertIs(gaussian_sort_key, gaussian_package_sort_key)
        self.assertIs(sort_key, orca_package_sort_key)
        self.assertEqual(sort_key("01_02_path3_04.out"), (1, 2, 3, 4))

    def test_tools_qm_reader_matches_formal_io_reader(self):
        self.assertIs(read_qm_out, io.read_qm_out)


if __name__ == "__main__":
    unittest.main()