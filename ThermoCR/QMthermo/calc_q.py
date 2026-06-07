"""Backward-compatible partition-function helpers."""

from ThermoCR.thermo.partition import (
    q,
    q_ele,
    q_rot,
    q_rot_linear,
    q_rot_nonlinear,
    q_rot_single_atom,
    q_trans,
    q_vib_V0,
    q_vib_bot,
)

__all__ = [
    "q",
    "q_ele",
    "q_rot",
    "q_rot_linear",
    "q_rot_nonlinear",
    "q_rot_single_atom",
    "q_trans",
    "q_vib_V0",
    "q_vib_bot",
]
