from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd

from ThermoCR.cli import build_parser, main
from ThermoCR.thermo import nasa7
from ThermoCR.types import ThermoOptions


LINK1_TEXT = """ Entering Gaussian System, Link 0=g16
 Copyright line
 ******************************************
 Gaussian 16
 ******************************************
 %nproc=8
 # opt b3lyp/6-31g(d)

 first job body
 Normal termination of Gaussian 16 at Wed Jul 23 23:12:17 2025.
 Link1:  Proceeding to internal job step number  2.
 ----------------------------------------------------------------------
 # freq b3lyp/6-31g(d)
 ----------------------------------------------------------------------
 second job body
 Normal termination of Gaussian 16 at Wed Jul 23 23:12:43 2025.
"""


class CliTests(unittest.TestCase):
    def _run_cli(self, argv):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(argv)
        return exit_code, stdout.getvalue()

    def test_parser_exposes_expected_commands(self):
        help_text = build_parser().format_help()

        self.assertIn("split-link1", help_text)
        self.assertIn("select-gaussian", help_text)
        self.assertIn("qm-energy", help_text)
        self.assertIn("orca-energy", help_text)
        self.assertIn("thermo", help_text)
        self.assertIn("kinetics", help_text)

    def test_split_link1_command_writes_job_files(self):
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "link1.out"
            output_dir = Path(tmpdir) / "jobs"
            input_path.write_text(LINK1_TEXT, encoding="utf-8")

            exit_code, stdout = self._run_cli([
                "split-link1",
                str(input_path),
                str(output_dir),
                "--prefix",
                "case",
            ])

            self.assertEqual(exit_code, 0)
            self.assertIn("case_job01.out", stdout)
            self.assertIn("case_job02.out", stdout)
            self.assertIn("first job body", (output_dir / "case_job01.out").read_text(encoding="utf-8"))
            self.assertIn("second job body", (output_dir / "case_job02.out").read_text(encoding="utf-8"))

    def test_select_gaussian_command_writes_selected_job(self):
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "link1.out"
            output_path = Path(tmpdir) / "selected.out"
            input_path.write_text(LINK1_TEXT, encoding="utf-8")

            exit_code, stdout = self._run_cli([
                "select-gaussian",
                str(input_path),
                str(output_path),
                "--task-id",
                "2",
                "--mode",
                "select",
            ])

            selected_text = output_path.read_text(encoding="utf-8")
            self.assertEqual(exit_code, 0)
            self.assertIn(str(output_path), stdout)
            self.assertIn("second job body", selected_text)
            self.assertNotIn("first job body", selected_text)

    @patch("ThermoCR.cli.read_electronic_energy")
    def test_qm_energy_command_dispatches_to_reader(self, read_electronic_energy):
        read_electronic_energy.return_value = -40.5

        exit_code, stdout = self._run_cli([
            "qm-energy",
            "calc.out",
            "--energy-index",
            "-2",
            "--gaussian-job-index",
            "2",
            "--unit",
            "ev",
        ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.strip(), "-40.5")
        read_electronic_energy.assert_called_once_with(
            filepath="calc.out",
            energy_index=-2,
            return_hartree=False,
            gaussian_job_index=2,
        )

    def test_orca_energy_command_prints_last_energy(self):
        text = """
        FINAL SINGLE POINT ENERGY     -10.125000
        FINAL SINGLE POINT ENERGY     -11.250000
        """
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "orca.out"
            output_path.write_text(text, encoding="utf-8")

            exit_code, stdout = self._run_cli(["orca-energy", str(output_path)])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.strip(), "-11.25")

    @patch("ThermoCR.cli.scan_thermo")
    @patch("ThermoCR.cli.read_molecule_data")
    def test_thermo_scan_command_writes_csv(self, read_molecule_data, scan_thermo):
        molecule = object()
        read_molecule_data.return_value = molecule
        scan_thermo.return_value = pd.DataFrame([
            {"temperature": 300.0, "pressure": 100000.0, "gibbs_free_energy": -1.0},
            {"temperature": 400.0, "pressure": 100000.0, "gibbs_free_energy": -2.0},
        ])

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "thermo.csv"
            exit_code, stdout = self._run_cli([
                "thermo",
                "scan",
                "calc.out",
                "--t-min",
                "300",
                "--t-max",
                "400",
                "--n-points",
                "2",
                "--pressure",
                "100000",
                "--gaussian-job-index",
                "-1",
                "--output",
                str(output_path),
            ])

            csv_text = output_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn(str(output_path), stdout)
        self.assertIn("temperature,pressure,gibbs_free_energy", csv_text)
        read_molecule_data.assert_called_once_with("calc.out", gaussian_job_index=-1)
        call = scan_thermo.call_args
        self.assertIs(call.args[0], molecule)
        self.assertEqual(call.kwargs["temperatures"], [300.0, 400.0])
        self.assertEqual(call.kwargs["pressure"], 100000.0)
        self.assertIsInstance(call.kwargs["options"], ThermoOptions)

    def test_thermo_fit_command_writes_json(self):
        temperatures = np.linspace(300.0, 1000.0, 12)
        parameters = [3.5, 1.0e-3, -2.0e-6, 1.0e-9, -1.0e-13, -1000.0, 5.0]
        heat_capacity, enthalpy, entropy = nasa7(temperatures, *parameters)

        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "thermo.csv"
            output_path = Path(tmpdir) / "fit.json"
            pd.DataFrame({
                "temperature": temperatures,
                "heat_capacity_cp": heat_capacity,
                "enthalpy": enthalpy,
                "entropy": entropy,
            }).to_csv(input_path, index=False)

            exit_code, stdout = self._run_cli([
                "thermo",
                "fit",
                str(input_path),
                "--model",
                "nasa7",
                "--weight-strategy",
                "uniform",
                "--guess",
                ",".join(str(parameter) for parameter in parameters),
                "--output",
                str(output_path),
            ])
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertIn(str(output_path), stdout)
        self.assertEqual(payload["model_type"], "NASA7")
        self.assertEqual(payload["temperature_range"], [300.0, 1000.0])
        self.assertGreater(payload["metrics"]["heat_capacity_cp"]["r2"], 0.999999)

    def test_thermo_fit_command_writes_cantera_yaml(self):
        temperatures = np.linspace(300.0, 1000.0, 12)
        parameters = [3.5, 1.0e-3, -2.0e-6, 1.0e-9, -1.0e-13, -1000.0, 5.0]
        heat_capacity, enthalpy, entropy = nasa7(temperatures, *parameters)

        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "thermo.csv"
            output_path = Path(tmpdir) / "fit.yaml"
            pd.DataFrame({
                "temperature": temperatures,
                "heat_capacity_cp": heat_capacity,
                "enthalpy": enthalpy,
                "entropy": entropy,
            }).to_csv(input_path, index=False)

            exit_code, stdout = self._run_cli([
                "thermo", "fit", str(input_path), "--guess",
                ",".join(str(parameter) for parameter in parameters),
                "--output", str(output_path),
            ])
            yaml_text = output_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn(str(output_path), stdout)
        self.assertIn("model: NASA7", yaml_text)
        self.assertIn("temperature-ranges: [300.0, 1000.0]", yaml_text)

    def test_kinetics_tst_command_writes_csv(self):
        temperatures = [300.0, 400.0]
        ts_frame = pd.DataFrame({
            "temperature": temperatures,
            "gibbs_free_energy": [20000.0, 22000.0],
            "electronic_energy": [0.0, 0.0],
            "zpe": [0.0, 0.0],
        })
        reactant_frame = pd.DataFrame({
            "temperature": temperatures,
            "gibbs_free_energy": [0.0, 0.0],
            "electronic_energy": [0.0, 0.0],
            "zpe": [0.0, 0.0],
        })

        with TemporaryDirectory() as tmpdir:
            ts_path = Path(tmpdir) / "ts.csv"
            r1_path = Path(tmpdir) / "r1.csv"
            r2_path = Path(tmpdir) / "r2.csv"
            output_path = Path(tmpdir) / "rates.csv"
            ts_frame.to_csv(ts_path, index=False)
            reactant_frame.to_csv(r1_path, index=False)
            reactant_frame.to_csv(r2_path, index=False)

            exit_code, stdout = self._run_cli([
                "kinetics",
                "tst",
                str(ts_path),
                "--reactant",
                str(r1_path),
                "--reactant",
                str(r2_path),
                "--output",
                str(output_path),
            ])
            output = pd.read_csv(output_path)

        self.assertEqual(exit_code, 0)
        self.assertIn(str(output_path), stdout)
        self.assertEqual(list(output["temperature"]), temperatures)
        self.assertEqual(list(output["delta_n"]), [1, 1])
        self.assertIn("rate_constant", output.columns)
        self.assertGreater(float(output["rate_constant"].iloc[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
