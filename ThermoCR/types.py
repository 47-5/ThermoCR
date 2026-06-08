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


@dataclass
class ThermoOptions:
    """Options for structured thermochemistry calculations."""

    temperature: float = 298.15
    pressure: float = 101325.0
    zpe_scale_factor: float = 1.0
    internal_energy_scale_factor: float = 1.0
    heat_capacity_scale_factor: float = 1.0
    entropy_scale_factor: float = 1.0
    use_minenkov_internal_energy: bool = False
    use_grimme_entropy: bool = True
    electronic_energies: Optional[Sequence[float]] = None
    electronic_degeneracies: Optional[Sequence[int]] = None
    ignore_trans_and_rot: bool = False
    concentration: Optional[float] = None

    def __post_init__(self):
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.pressure <= 0.0:
            raise ValueError("pressure must be positive")
        if (self.electronic_energies is None) != (self.electronic_degeneracies is None):
            raise ValueError(
                "electronic_energies and electronic_degeneracies must be provided together"
            )
        if self.electronic_energies is not None:
            self.electronic_energies = tuple(float(value) for value in self.electronic_energies)
            self.electronic_degeneracies = tuple(
                int(value) for value in self.electronic_degeneracies
            )
            if len(self.electronic_energies) != len(self.electronic_degeneracies):
                raise ValueError(
                    "electronic_energies and electronic_degeneracies must have the same length"
                )


@dataclass
class ThermoResult:
    """Structured thermochemistry result in SI units."""

    temperature: float
    pressure: float
    partition_function_v0: float
    partition_function_bottom: float
    heat_capacity_cv: float
    heat_capacity_cp: float
    entropy: float
    zpe: float
    internal_energy_correction: float
    enthalpy_correction: float
    gibbs_energy_correction: float
    electronic_energy: float
    internal_energy: float
    enthalpy: float
    gibbs_free_energy: float
    concentration_delta_g: Optional[float] = None

    @classmethod
    def from_qm_thermo_dict(cls, data):
        return cls(
            temperature=float(data["T/K"]),
            pressure=float(data["P/Pa"]),
            partition_function_v0=float(data["q_tot_v_0"]),
            partition_function_bottom=float(data["q_tot_bot"]),
            heat_capacity_cv=float(data["Cv/(J/mol/K)"]),
            heat_capacity_cp=float(data["Cp/(J/mol/K)"]),
            entropy=float(data["S/(J/mol/K)"]),
            zpe=float(data["zpe/(J/mol)"]),
            internal_energy_correction=float(data["U_corr/(J/mol)"]),
            enthalpy_correction=float(data["H_corr/(J/mol)"]),
            gibbs_energy_correction=float(data["G_corr/(J/mol)"]),
            electronic_energy=float(data["ee/(J/mol)"]),
            internal_energy=float(data["U/(J/mol)"]),
            enthalpy=float(data["H/(J/mol)"]),
            gibbs_free_energy=float(data["G/(J/mol)"]),
            concentration_delta_g=data.get("delta_G_of_conc.(J/mol)"),
        )


__all__ = ["MoleculeData", "ThermoOptions", "ThermoResult"]
