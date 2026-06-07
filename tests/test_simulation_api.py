from pathlib import Path
import unittest

from ThermoCR import ChemicalKineticsSimulator as top_level_ChemicalKineticsSimulator
from ThermoCR.QMconcvar import ChemicalKineticsSimulator as legacy_ChemicalKineticsSimulator
from ThermoCR.QMconcvar.temperature_program_simulator import (
    parser_T_program as legacy_parser_T_program,
)
from ThermoCR.simulation import (
    ChemicalKineticsSimulator,
    parser_T_program,
    run_temperature_simulation,
)
from ThermoCR.simulation.reaction import (
    ChemicalKineticsSimulator as namespaced_ChemicalKineticsSimulator,
)
from ThermoCR.simulation.temperature_program import (
    parser_T_program as namespaced_parser_T_program,
)


class SimulationApiTests(unittest.TestCase):
    def test_new_and_legacy_simulation_class_paths_match(self):
        self.assertIs(ChemicalKineticsSimulator, legacy_ChemicalKineticsSimulator)
        self.assertIs(namespaced_ChemicalKineticsSimulator, legacy_ChemicalKineticsSimulator)
        self.assertIs(top_level_ChemicalKineticsSimulator, legacy_ChemicalKineticsSimulator)

    def test_new_and_legacy_temperature_program_helpers_match(self):
        self.assertIs(parser_T_program, legacy_parser_T_program)
        self.assertIs(namespaced_parser_T_program, legacy_parser_T_program)
        self.assertTrue(callable(run_temperature_simulation))

    def test_temperature_program_parser_handles_constant_and_linear_segments(self):
        T_list, t_start_list, t_end_list = parser_T_program([
            {"type": "constant", "T_start": 300.0, "t_start": 0.0, "t_end": 10.0},
            {
                "type": "linear",
                "T_start": 300.0,
                "T_end": 500.0,
                "segments": 2,
                "t_start": 10.0,
                "t_end": 30.0,
            },
        ])

        self.assertEqual(T_list, [300.0, 300.0, 500.0])
        self.assertEqual(t_start_list, [0.0, 10.0, 20.0])
        self.assertEqual(t_end_list, [10.0, 20.0, 30.0])

    def test_modern_simulation_class_resolves_relative_paths(self):
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


if __name__ == "__main__":
    unittest.main()