"""Compatibility namespace for legacy Gaussian helpers."""

from ThermoCR.tools.about_gaussian.calculate_solvent_energy import (
    calculate_solvent_energy,
    sort_key,
    standard_state_energy,
)
from ThermoCR.tools.about_gaussian.link1 import (
    LINK1_MARKER,
    NORMAL_TERMINATION_MARKER,
    is_gaussian_link1_output,
    read_gaussian_link1_job,
    select_gaussian_link1_text,
    split_gaussian_link1_output,
    split_gaussian_link1_text,
)
from ThermoCR.tools.about_gaussian.select_gaussian_out import (
    select_gaussian_out,
    select_gaussian_output,
)

__all__ = [
    "LINK1_MARKER",
    "NORMAL_TERMINATION_MARKER",
    "calculate_solvent_energy",
    "is_gaussian_link1_output",
    "read_gaussian_link1_job",
    "select_gaussian_link1_text",
    "select_gaussian_out",
    "select_gaussian_output",
    "sort_key",
    "split_gaussian_link1_output",
    "split_gaussian_link1_text",
    "standard_state_energy",
]