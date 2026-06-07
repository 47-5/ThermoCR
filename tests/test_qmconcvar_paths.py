from pathlib import Path
import unittest

from ThermoCR.QMconcvar.constant_temperature_simulator import ChemicalKineticsSimulator


class QMConcvarPathTests(unittest.TestCase):
    def test_resolves_relative_paths_against_config_directory(self):
        simulator = ChemicalKineticsSimulator.__new__(ChemicalKineticsSimulator)
        simulator.config_dir = Path(__file__).resolve().parents[1] / "example" / "QMconcvar"

        resolved = simulator._resolve_on_the_fly_paths({
            "atom_coord_path": "01.out",
            "vib_path": "01.out",
            "ee_path": None,
        })

        self.assertEqual(resolved["atom_coord_path"], str(simulator.config_dir / "01.out"))
        self.assertEqual(resolved["vib_path"], str(simulator.config_dir / "01.out"))
        self.assertIsNone(resolved["ee_path"])

    def test_keeps_absolute_paths(self):
        simulator = ChemicalKineticsSimulator.__new__(ChemicalKineticsSimulator)
        simulator.config_dir = Path(__file__).resolve().parents[1] / "example" / "QMconcvar"
        absolute_path = Path(__file__).resolve()

        resolved = simulator._resolve_config_path(str(absolute_path))

        self.assertEqual(resolved, str(absolute_path))


if __name__ == "__main__":
    unittest.main()
