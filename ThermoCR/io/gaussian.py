"""Gaussian output helpers."""

from pathlib import Path
from tempfile import TemporaryDirectory

import cclib


NORMAL_TERMINATION_MARKER = "Normal termination of Gaussian"
LINK1_MARKER = "Link1:"


__all__ = [
    "LINK1_MARKER",
    "NORMAL_TERMINATION_MARKER",
    "is_gaussian_link1_output",
    "read_gaussian_link1_job",
    "select_gaussian_link1_text",
    "select_gaussian_out",
    "select_gaussian_output",
    "split_gaussian_link1_output",
    "split_gaussian_link1_text",
]


def _as_text(input_text):
    if isinstance(input_text, str):
        return input_text
    return "".join(input_text)


def _read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _write_text(path, text):
    Path(path).write_text(text, encoding="utf-8")


def _find_gaussian_header(lines):
    """Return the banner/header that cclib needs before a Link1 job body."""
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("#"):
            return lines[:index]
    return []


def _find_normal_termination_ends(lines):
    return [
        index + 1
        for index, line in enumerate(lines)
        if NORMAL_TERMINATION_MARKER in line
    ]


def _normalize_job_index(job_index, job_count):
    if job_count < 1:
        raise ValueError("No Gaussian job section was found.")

    if job_index == 0:
        raise ValueError(
            "Gaussian job_index is 1-based for positive values; use 1 for the "
            "first job or -1 for the last job."
        )

    if job_index < 0:
        normalized = job_count + job_index
    else:
        normalized = job_index - 1

    if normalized < 0 or normalized >= job_count:
        raise IndexError(
            f"Gaussian job_index {job_index} is out of range for "
            f"{job_count} job section(s)."
        )
    return normalized


def split_gaussian_link1_text(text):
    """Split a Gaussian Link1 output text into cclib-readable job texts.

    Gaussian Link1 output often prints the full Gaussian banner only once. Later
    internal job steps start after a ``Link1:`` line and are not recognized by
    cclib when naively sliced from the previous termination line. For those
    later sections, this function prepends the original Gaussian banner.
    """
    text = _as_text(text)
    lines = text.splitlines(keepends=True)
    termination_ends = _find_normal_termination_ends(lines)
    if not termination_ends:
        return [text]

    header = _find_gaussian_header(lines)
    sections = []
    start = 0
    for index, end in enumerate(termination_ends):
        section_lines = lines[start:end]
        if index > 0 and header:
            section_lines = header + section_lines
        sections.append("".join(section_lines))
        start = end
    return sections


def select_gaussian_link1_text(text, job_index=-1):
    """Select one Gaussian job text from a Link1 output."""
    sections = split_gaussian_link1_text(text)
    normalized = _normalize_job_index(job_index, len(sections))
    return sections[normalized]


def select_gaussian_output(input_path, output_path, task_id=2, select_mode="cut"):
    """Write a selected Gaussian Link1 job or prefix to an output file.

    ``task_id`` follows the legacy one-based Gaussian job numbering used by
    ``select_gaussian_out``. In ``select`` mode, only the selected Link1 job is
    written. In ``cut`` mode, the output is truncated through the selected
    normal termination.
    """
    if select_mode not in {"cut", "select"}:
        raise ValueError("select_mode must be 'cut' or 'select'")

    text = _read_text(input_path)

    if task_id == 1:
        select_mode = "cut"

    if select_mode == "select":
        output_text = select_gaussian_link1_text(text, job_index=task_id)
    else:
        lines = text.splitlines(keepends=True)
        end_line_indices = [
            line_index + 1
            for line_index, line in enumerate(lines)
            if "Normal termination" in line
        ]
        task_index = task_id - 1
        if task_index < 0 or task_index >= len(end_line_indices):
            raise ValueError(
                f"task_id {task_id} is out of range for "
                f"{len(end_line_indices)} Gaussian job section(s)."
            )
        output_text = "".join(lines[: end_line_indices[task_index]])

    _write_text(output_path, output_text)
    return None


# Backward-compatible name.
def select_gaussian_out(input_path, output_path, task_id=2, select_mode="cut"):
    return select_gaussian_output(
        input_path=input_path,
        output_path=output_path,
        task_id=task_id,
        select_mode=select_mode,
    )


def is_gaussian_link1_output(filepath):
    """Return True when a file looks like a multi-step Gaussian Link1 output."""
    text = _read_text(filepath)
    return (
        LINK1_MARKER in text
        and text.count(NORMAL_TERMINATION_MARKER) > 1
    )


def split_gaussian_link1_output(input_path, output_dir, prefix=None):
    """Write each Link1 job section to a separate output file."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = _read_text(input_path)
    sections = split_gaussian_link1_text(text)
    if prefix is None:
        prefix = input_path.stem

    output_paths = []
    width = max(2, len(str(len(sections))))
    for index, section in enumerate(sections, start=1):
        output_path = output_dir / f"{prefix}_job{index:0{width}d}.out"
        _write_text(output_path, section)
        output_paths.append(output_path)
    return output_paths


def read_gaussian_link1_job(input_path, job_index=-1):
    """Read one job section from a Gaussian Link1 output with cclib."""
    input_path = Path(input_path)
    text = _read_text(input_path)
    selected_text = select_gaussian_link1_text(text, job_index=job_index)

    with TemporaryDirectory() as tmpdir:
        selected_path = Path(tmpdir) / input_path.name
        _write_text(selected_path, selected_text)
        return cclib.io.ccread(str(selected_path))
