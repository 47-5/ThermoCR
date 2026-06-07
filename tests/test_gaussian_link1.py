from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ThermoCR.io import select_gaussian_output
from ThermoCR.tools.about_gaussian.link1 import (
    select_gaussian_link1_text,
    split_gaussian_link1_output,
    split_gaussian_link1_text,
)
from ThermoCR.tools.about_gaussian.select_gaussian_out import select_gaussian_out
from ThermoCR.tools.read_qm_out import read_qm_out


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


class GaussianLink1Tests(unittest.TestCase):
    def test_split_prepends_header_to_later_jobs(self):
        sections = split_gaussian_link1_text(LINK1_TEXT)

        self.assertEqual(len(sections), 2)
        self.assertTrue(sections[0].startswith(" Entering Gaussian System"))
        self.assertTrue(sections[1].startswith(" Entering Gaussian System"))
        self.assertIn("second job body", sections[1])
        self.assertNotIn("first job body", sections[1])

    def test_select_uses_gaussian_one_based_positive_index(self):
        second_job = select_gaussian_link1_text(LINK1_TEXT, job_index=2)

        self.assertIn("second job body", second_job)
        self.assertNotIn("first job body", second_job)

    def test_select_supports_last_job_index(self):
        second_job = select_gaussian_link1_text(LINK1_TEXT, job_index=-1)

        self.assertIn("second job body", second_job)

    def test_select_rejects_zero_index(self):
        with self.assertRaises(ValueError):
            select_gaussian_link1_text(LINK1_TEXT, job_index=0)

    def test_split_output_writes_single_job_files(self):
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "link1.out"
            output_dir = Path(tmpdir) / "jobs"
            input_path.write_text(LINK1_TEXT)

            output_paths = split_gaussian_link1_output(input_path, output_dir)

            self.assertEqual(len(output_paths), 2)
            self.assertIn("first job body", output_paths[0].read_text())
            self.assertIn("second job body", output_paths[1].read_text())

    def test_formal_select_gaussian_output_selects_readable_link1_job(self):
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "link1.out"
            output_path = Path(tmpdir) / "job2.out"
            input_path.write_text(LINK1_TEXT)

            select_gaussian_output(
                input_path,
                output_path,
                task_id=2,
                select_mode="select",
            )

            selected_text = output_path.read_text()
            self.assertTrue(selected_text.startswith(" Entering Gaussian System"))
            self.assertIn("second job body", selected_text)
            self.assertNotIn("first job body", selected_text)

    def test_legacy_select_gaussian_out_selects_readable_link1_job(self):
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "link1.out"
            output_path = Path(tmpdir) / "job2.out"
            input_path.write_text(LINK1_TEXT)

            select_gaussian_out(
                input_path,
                output_path,
                task_id=2,
                select_mode="select",
            )

            selected_text = output_path.read_text()
            self.assertTrue(selected_text.startswith(" Entering Gaussian System"))
            self.assertIn("second job body", selected_text)
            self.assertNotIn("first job body", selected_text)

    def test_read_qm_out_reads_last_job_from_real_link1_example(self):
        example_path = Path(__file__).resolve().parents[1] / "example" / "CPD.out"
        data = read_qm_out(example_path)

        self.assertEqual(len(data.atomnos), 11)
        self.assertEqual(len(data.vibfreqs), 27)


if __name__ == "__main__":
    unittest.main()