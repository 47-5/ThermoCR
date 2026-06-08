from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from ThermoCR.cli import build_parser, main


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

    def test_split_link1_command_writes_job_files(self):
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "link1.out"
            output_dir = Path(tmpdir) / "jobs"
            input_path.write_text(LINK1_TEXT)

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
            self.assertIn("first job body", (output_dir / "case_job01.out").read_text())
            self.assertIn("second job body", (output_dir / "case_job02.out").read_text())

    def test_select_gaussian_command_writes_selected_job(self):
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "link1.out"
            output_path = Path(tmpdir) / "selected.out"
            input_path.write_text(LINK1_TEXT)

            exit_code, stdout = self._run_cli([
                "select-gaussian",
                str(input_path),
                str(output_path),
                "--task-id",
                "2",
                "--mode",
                "select",
            ])

            selected_text = output_path.read_text()
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
            output_path.write_text(text)

            exit_code, stdout = self._run_cli(["orca-energy", str(output_path)])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.strip(), "-11.25")


if __name__ == "__main__":
    unittest.main()
