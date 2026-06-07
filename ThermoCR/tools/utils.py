"""Backward-compatible utility wrappers.

The symmetry-related helpers now live in ``ThermoCR.symmetry``. These names are
kept for existing ThermoCR scripts and older public imports.
"""

from ThermoCR.symmetry import (
    detect_point_group,
    is_linear,
    principal_moments,
    rotational_symmetry_number,
)

__all__ = [
    "check_linear",
    "get_I",
    "get_point_group",
    "get_rotational_symmetry_number",
]


def get_point_group(coords, symbols=None, numbers=None):
    return detect_point_group(coords=coords, symbols=symbols, numbers=numbers)


def get_I(coords, numbers):
    return principal_moments(coords=coords, numbers=numbers)


def check_linear(I, threshold=1e-3):
    return is_linear(I, threshold=threshold)


def get_rotational_symmetry_number(point_group):
    return rotational_symmetry_number(point_group)
