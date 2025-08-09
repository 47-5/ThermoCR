from ase import Atoms
import numpy as np

from ThermoCR.pointgroup import PointGroup
from ase.units import Bohr
from ThermoCR.tools.constant import amu2kg


def get_point_group(coords, symbols=None, numbers=None):
    """
    Determine the point group of a given set of atomic coordinates.

    Summary:
    This function calculates the point group for a set of 3D coordinates
    representing atomic positions. It can take either atomic symbols or
    atomic numbers to identify the elements. The function uses the provided
    coordinates and symbols (or numbers) to create an Atoms object, which
    is then used to determine the point group of the structure.

    Parameters:
    coords: List[List[float]]
        A list of 3D Cartesian coordinates for each atom.
    symbols: Optional[List[str]]
        A list of atomic symbols corresponding to the atoms at the given
        coordinates. If not provided, `numbers` must be specified.
    numbers: Optional[List[int]]
        A list of atomic numbers corresponding to the atoms at the given
        coordinates. This is used if `symbols` are not provided.

    Returns:
    str
        The point group symbol as a string.

    Raises:
    ValueError
        If neither `symbols` nor `numbers` are provided, or if there is
        a mismatch in the length of `coords` and the provided `symbols`
        or `numbers`.

    Examples:
    None

    See Also:
    None

    Notes:
    The function internally creates an Atoms object using the provided
    data and then utilizes the PointGroup class to determine the point
    group of the molecular or crystal structure represented by the input
    coordinates and atomic identifiers.
    """
    if symbols is None:
        atoms = Atoms(numbers=numbers, positions=coords)
        symbols = atoms.symbols
    pg = PointGroup(positions=coords, symbols=symbols)
    return pg.get_point_group()


def get_I(coords, numbers):
    """
    Calculate the moments of inertia for a set of atoms.

    This function computes the moments of inertia for a given set of atomic coordinates and their corresponding atomic
    numbers. The result is normalized by the square of the Bohr radius.

    Parameters:
    coords (List[List[float]]): A list of 3D Cartesian coordinates for each atom.
    numbers (List[int]): A list of atomic numbers corresponding to each coordinate.

    Returns:
    numpy.ndarray: An array containing the three principal moments of inertia.

    Raises:
    ValueError: If the input lists do not have matching lengths or if any of the inputs are invalid.
    """
    atoms = Atoms(numbers=numbers, positions=coords)
    I = atoms.get_moments_of_inertia() / Bohr ** 2
    return I


def check_linear(I, threshold=1e-3):
    """
    Checks if the given array I of a molecule is considered linear based on a threshold.

    Summary:
    Evaluates whether the given array `I` (representing moments of inertia) is considered linear based on a specified
    threshold. The function returns `True` if the first element of `I` is below the threshold and the absolute
    difference between the second and third elements of `I` is also below this threshold; otherwise, it returns `False`.


    Args:
        I (List[float]): The input list or array to be checked.
        threshold (float, optional): The threshold value used for comparison. Defaults to 1e-3.

    Returns:
        bool: True if the array is considered linear, False otherwise.

    Raises:
        IndexError: If the input array I does not have at least three elements.
    """
    if I[0] < threshold and np.abs(I[1] - I[2]) < threshold:
        return True
    else:
        return False


def get_rotational_symmetry_number(point_group):
    """
    Calculate the rotational symmetry number for a given point group.

    The function supports various point groups, including cubic, cyclic (C), dihedral (D),
    special cases like C1, Cs, Ci, and specific high-symmetry groups such as Dinfh and Cinfv.
    For cubic groups (T, Th, Td, O, Oh, I, Ih), it returns predefined values. For cyclic
    and dihedral groups, it calculates the rotational symmetry based on the order of the group.
    Special cases and high-symmetry groups have fixed rotational symmetry numbers.

    :raises ValueError: If the provided point_group is not recognized by the function.
    :param str point_group: The point group symbol to calculate the rotational symmetry number for.
    :returns int: The rotational symmetry number corresponding to the given point group.
    """
    if point_group == 'Dinfh':
        return 2
    elif point_group == 'Cinfv':
        return 1

    cubic_groups = {
        'T': 12, 'Th': 12, 'Td': 12,
        'O': 24, 'Oh': 24,
        'I': 60, 'Ih': 60
    }
    if point_group in cubic_groups:
        return cubic_groups[point_group]

    if len(point_group) >= 2 and point_group[1].isdigit():
        n = int(''.join(filter(str.isdigit, point_group)))

        if point_group.startswith('C') or point_group.startswith('S'):
            return n
        elif point_group.startswith('D'):
            return 2 * n

    special_groups = {
        'C1': 1, 'Cs': 1, 'Ci': 1
    }
    if point_group in special_groups:
        return special_groups[point_group]

    raise ValueError(f"unknown: {point_group}")


# # 示例使用
# if __name__ == "__main__":
#     # 测试用例
#     test_groups = [
#         'C1', 'Cs', 'Ci', 'C2', 'C3', 'C3v', 'C4h', 'S4',
#         'D2', 'D3', 'D3h', 'D5d', 'Td', 'Oh', 'Ih', 'Cinfv', 'Dinfh'
#     ]
#
#     for pg in test_groups:
#         print(f"点群 {pg} 的旋转对称数: {get_rotational_symmetry_number(pg)}")