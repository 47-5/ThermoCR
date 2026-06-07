"""Solvation thermochemistry helpers."""

import re

from ase.io import read
from ase.units import Hartree, kcal, mol

standard_state_energy = 1.84 * kcal / mol


def calculate_solvent_energy(
    gas_gaussian_out_path,
    sol_gaussian_out_path,
    add_standard_state=True,
    return_hartree=True,
):
    """Calculate the solvated-minus-gas electronic energy difference."""
    gas_atoms = read(gas_gaussian_out_path, format="gaussian-out")
    sol_atoms = read(sol_gaussian_out_path, format="gaussian-out")
    gas_energy = gas_atoms.get_potential_energy() / Hartree
    sol_energy = sol_atoms.get_potential_energy() / Hartree
    delta_energy = sol_energy - gas_energy
    if add_standard_state:
        delta_energy += standard_state_energy

    if return_hartree:
        delta_energy /= Hartree
    return delta_energy


def sort_key(file_name):
    """Sort Gaussian path filenames using the historical ThermoCR pattern."""
    match = re.search(r"(\d+)_(\d+)_path(\d+)_(\d+)", file_name)
    if match:
        return tuple(map(int, match.groups()))
    return float("inf"), float("inf"), float("inf"), float("inf")


__all__ = ["calculate_solvent_energy", "sort_key", "standard_state_energy"]
