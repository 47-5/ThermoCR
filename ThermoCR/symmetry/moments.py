"""Rotational moment helpers."""

import numpy as np
from ase import Atoms
from ase.units import Bohr


def principal_moments(coords, numbers, convert_to_atomic_units=True):
    """Return principal moments of inertia for a molecule.

    Coordinates are expected in Angstrom, matching ASE conventions. By default
    the result is normalized by ``Bohr ** 2`` to preserve ThermoCR's existing
    internal units.
    """
    atoms = Atoms(numbers=numbers, positions=coords)
    moments = atoms.get_moments_of_inertia()
    if convert_to_atomic_units:
        moments = moments / Bohr ** 2
    return moments


def is_linear(moments, threshold=1e-3):
    """Return True when principal moments indicate a linear molecule."""
    moments = np.asarray(moments)
    return bool(moments[0] < threshold and np.abs(moments[1] - moments[2]) < threshold)
