"""Quantum-chemistry output readers."""

import cclib
import numpy as np
from ase.units import Hartree

from ThermoCR.elements import atomic_symbol
from ThermoCR.io.gaussian import is_gaussian_link1_output, read_gaussian_link1_job
from ThermoCR.types import MoleculeData


__all__ = [
    "read_atom_coordinates",
    "read_atom_coord",
    "read_electronic_energy",
    "read_ee",
    "read_imaginary_frequency",
    "read_imaginary_vib",
    "read_molecule_data",
    "read_qm_output",
    "read_qm_out",
    "read_vibrational_frequencies",
    "read_vib",
]


def read_qm_output(filepath, gaussian_job_index=None, prefer_link1_split=True):
    """Read a quantum-chemistry output file with cclib.

    Gaussian Link1 multi-step outputs are split before parsing by default. When
    ``gaussian_job_index`` is omitted, the last normally terminated Gaussian job
    is selected.
    """
    if prefer_link1_split and is_gaussian_link1_output(filepath):
        job_index = -1 if gaussian_job_index is None else gaussian_job_index
        return read_gaussian_link1_job(filepath, job_index=job_index)

    try:
        return cclib.io.ccread(filepath)
    except Exception:
        if prefer_link1_split:
            job_index = -1 if gaussian_job_index is None else gaussian_job_index
            return read_gaussian_link1_job(filepath, job_index=job_index)
        raise


def _optional_array(values, dtype=float):
    if values is None:
        return None
    return np.asarray(values, dtype=dtype)


def read_molecule_data(
    filepath,
    coord_index=-1,
    energy_index=-1,
    gaussian_job_index=None,
    prefer_link1_split=True,
    return_hartree=True,
):
    """Read a QM output file into a structured :class:`MoleculeData` object."""
    data = read_qm_output(
        filepath,
        gaussian_job_index=gaussian_job_index,
        prefer_link1_split=prefer_link1_split,
    )
    atom_numbers = np.asarray(data.atomnos, dtype=int)
    symbols = tuple(atomic_symbol(atomic_number) for atomic_number in atom_numbers)
    coordinates = np.asarray(data.atomcoords[coord_index], dtype=float)

    scfenergies = getattr(data, "scfenergies", None)
    electronic_energy = None
    electronic_energy_unit = "hartree" if return_hartree else "eV"
    if scfenergies is not None and len(scfenergies) > 0:
        electronic_energy = float(scfenergies[energy_index])
        if return_hartree:
            electronic_energy /= Hartree

    vibfreqs = getattr(data, "vibfreqs", None)
    frequencies = _optional_array(vibfreqs)
    imaginary_frequencies = None
    if frequencies is not None:
        imaginary_frequencies = frequencies[frequencies < 0.0]

    return MoleculeData(
        atom_numbers=atom_numbers,
        symbols=symbols,
        coordinates=coordinates,
        electronic_energy=electronic_energy,
        frequencies=frequencies,
        imaginary_frequencies=imaginary_frequencies,
        charge=getattr(data, "charge", None),
        multiplicity=getattr(data, "mult", None),
        electronic_energy_unit=electronic_energy_unit,
    )


def read_atom_coordinates(filepath, coord_index=-1, gaussian_job_index=None):
    """Return atomic numbers and one coordinate set from an output file."""
    data = read_qm_output(filepath, gaussian_job_index=gaussian_job_index)
    atom_numbers = data.atomnos
    coords = data.atomcoords[coord_index]
    return atom_numbers, coords


def read_vibrational_frequencies(filepath, gaussian_job_index=None):
    """Return vibrational frequencies from an output file."""
    data = read_qm_output(filepath, gaussian_job_index=gaussian_job_index)
    if len(data.atomnos) <= 1:
        return []
    return data.vibfreqs


def read_imaginary_frequency(filepath, vibfreqs=None, gaussian_job_index=None):
    """Return the largest-magnitude imaginary frequency, if present."""
    if vibfreqs is None:
        vibfreqs = read_vibrational_frequencies(
            filepath=filepath,
            gaussian_job_index=gaussian_job_index,
        )

    vibfreqs_float = [float(freq) for freq in vibfreqs]
    negative_freqs = [freq for freq in vibfreqs_float if freq < 0.0]

    if len(negative_freqs) > 1:
        return min(negative_freqs)
    if len(negative_freqs) == 1:
        return negative_freqs[0]
    return None


def read_electronic_energy(filepath, energy_index=-1, return_hartree=True,
                           gaussian_job_index=None):
    """Return one electronic energy from an output file."""
    data = read_qm_output(filepath, gaussian_job_index=gaussian_job_index)
    energy = data.scfenergies[energy_index]
    if return_hartree:
        energy /= Hartree
    return energy


# Backward-compatible wrappers.
def read_qm_out(filepath, gaussian_job_index=None, prefer_link1_split=True):
    return read_qm_output(
        filepath,
        gaussian_job_index=gaussian_job_index,
        prefer_link1_split=prefer_link1_split,
    )


def read_atom_coord(filepath, coord_index=-1, gaussian_job_index=None):
    return read_atom_coordinates(
        filepath,
        coord_index=coord_index,
        gaussian_job_index=gaussian_job_index,
    )


def read_vib(filepath, gaussian_job_index=None):
    return read_vibrational_frequencies(
        filepath,
        gaussian_job_index=gaussian_job_index,
    )


def read_imaginary_vib(filepath, vibfreqs=None, gaussian_job_index=None):
    return read_imaginary_frequency(
        filepath,
        vibfreqs=vibfreqs,
        gaussian_job_index=gaussian_job_index,
    )


def read_ee(filepath, ee_index=-1, return_Hartree=True, gaussian_job_index=None):
    return read_electronic_energy(
        filepath,
        energy_index=ee_index,
        return_hartree=return_Hartree,
        gaussian_job_index=gaussian_job_index,
    )
