"""ORCA output helpers."""

from pathlib import Path
import re

_FINAL_SINGLE_POINT_PATTERN = re.compile(
    r"FINAL SINGLE POINT ENERGY\s+([-+]?\d+\.\d+)"
)


def read_orca_final_single_point_energy(orca_out_file_path):
    """Return the last final single-point energy from an ORCA output file."""
    matches = []
    with Path(orca_out_file_path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = _FINAL_SINGLE_POINT_PATTERN.search(line)
            if match:
                matches.append(float(match.group(1)))

    if not matches:
        raise ValueError(
            f"No final single-point energy was found in {orca_out_file_path!r}"
        )
    return matches[-1]


# Backward-compatible name retained from the original helper.
def read_orca_wB97Mp2_out(orca_out_file_path):
    return read_orca_final_single_point_energy(orca_out_file_path)


def sort_key(file_name):
    """Sort ORCA path filenames using the historical ThermoCR pattern."""
    match = re.search(r"(\d+)_(\d+)_path(\d+)_(\d+)", file_name)
    if match:
        return tuple(map(int, match.groups()))
    return float("inf"), float("inf"), float("inf"), float("inf")


__all__ = [
    "read_orca_final_single_point_energy",
    "read_orca_wB97Mp2_out",
    "sort_key",
]
