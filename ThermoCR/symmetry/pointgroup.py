"""Point-group detection helpers."""

from ase import Atoms

from ThermoCR.pointgroup import PointGroup


def detect_point_group(coords, symbols=None, numbers=None):
    """Detect the molecular point group from Cartesian coordinates.

    Parameters
    ----------
    coords : array-like
        Atomic Cartesian coordinates in Angstrom.
    symbols : list[str], optional
        Atomic symbols. If omitted, ``numbers`` must be provided.
    numbers : list[int], optional
        Atomic numbers used to infer symbols when ``symbols`` is omitted.
    """
    if symbols is None:
        if numbers is None:
            raise ValueError("Either symbols or numbers must be provided.")
        atoms = Atoms(numbers=numbers, positions=coords)
        symbols = atoms.symbols

    point_group = PointGroup(positions=coords, symbols=symbols)
    return point_group.get_point_group()
