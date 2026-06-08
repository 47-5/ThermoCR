"""Shared data objects for ThermoCR public APIs."""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class MoleculeData:
    """Structured molecule data parsed from quantum-chemistry output."""

    symbols: Sequence[str]
    coordinates: np.ndarray
    atom_numbers: Optional[np.ndarray] = None
    electronic_energy: Optional[float] = None
    frequencies: Optional[np.ndarray] = None
    imaginary_frequencies: Optional[np.ndarray] = None
    charge: Optional[int] = None
    multiplicity: Optional[int] = None
    electronic_energy_unit: str = "hartree"
    coordinate_unit: str = "angstrom"
    frequency_unit: str = "cm^-1"

    def __post_init__(self):
        self.symbols = tuple(self.symbols)
        self.coordinates = np.asarray(self.coordinates, dtype=float)
        if self.atom_numbers is not None:
            self.atom_numbers = np.asarray(self.atom_numbers, dtype=int)
        if self.frequencies is not None:
            self.frequencies = np.asarray(self.frequencies, dtype=float)
        if self.imaginary_frequencies is not None:
            self.imaginary_frequencies = np.asarray(
                self.imaginary_frequencies,
                dtype=float,
            )

    @property
    def n_atoms(self):
        return len(self.symbols)


__all__ = ["MoleculeData"]
