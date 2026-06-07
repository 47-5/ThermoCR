"""Compatibility wrappers for legacy ORCA output helpers."""

from ThermoCR.io.orca import (
    read_orca_final_single_point_energy,
    read_orca_wB97Mp2_out,
    sort_key,
)

__all__ = [
    "read_orca_final_single_point_energy",
    "read_orca_wB97Mp2_out",
    "sort_key",
]
