"""Backward-compatible Gaussian Link1 helpers."""

from ThermoCR.io.gaussian import (
    LINK1_MARKER,
    NORMAL_TERMINATION_MARKER,
    is_gaussian_link1_output,
    read_gaussian_link1_job,
    select_gaussian_link1_text,
    select_gaussian_out,
    select_gaussian_output,
    split_gaussian_link1_output,
    split_gaussian_link1_text,
)

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