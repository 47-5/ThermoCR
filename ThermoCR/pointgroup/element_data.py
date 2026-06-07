"""Compatibility wrapper for legacy pointgroup element data imports."""

from ThermoCR.elements import (
    atom_data,
    atomic_mass,
    atomic_masses,
    atomic_symbol,
    element_data_by_atomic_number,
    element_mass,
)

__all__ = [
    "atom_data",
    "atomic_mass",
    "atomic_masses",
    "atomic_symbol",
    "element_data_by_atomic_number",
    "element_mass",
]