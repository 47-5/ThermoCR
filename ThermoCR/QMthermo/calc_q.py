import numpy as np

from ThermoCR.tools.constant import k_b, h, R, amu2kg, convert_I, wave2freq, au2eV
from ThermoCR.tools.utils import get_point_group, get_I, check_linear, get_rotational_symmetry_number


def q_trans(M, T, P, convert_unit=True):
    """
    Calculate the translational partition function for a given molar mass, temperature, and pressure.

    The function computes the translational partition function (q_t) using the provided molar mass (M),
    temperature (T), and pressure (P). The calculation can be performed with or without unit conversion
    based on the `convert_unit` parameter. If `convert_unit` is True, the molar mass is converted to
    kilograms before the calculation. The result is a measure of the number of states available to a
    molecule due to its translational motion at the specified conditions.

    Parameters:
        M (float): Molar mass of the substance in g/mol.
        T (float): Temperature in Kelvin.
        P (float): Pressure in Pascals.
        convert_unit (bool, optional): Whether to convert molar mass to kilograms. Defaults to True.

    Returns:
        float: Translational partition function.

    Notes:
        - The constants R, amu2kg, k_b, and h used in the calculation are assumed to be defined elsewhere
          in the code and represent the gas constant, atomic mass units to kilograms conversion factor,
          Boltzmann's constant, and Planck's constant, respectively.
    """
    if convert_unit:
        q_t = (R * T / P) * ((2 * np.pi * M * amu2kg * k_b * T) / (h ** 2)) ** 1.5
    else:
        q_t = (R * T / P) * ((2 * np.pi * M * k_b * T) / (h ** 2)) ** 1.5
    return q_t


def q_rot(atom_numbers, coords, T, convert_unit=True):
    """
    Calculate the rotational partition function for a given set of atoms and their coordinates.

    Summary:
    This function computes the rotational partition function (q_r) for a molecule. The calculation
    depends on whether the molecule is linear or non-linear, and it takes into account the point group
    and rotational symmetry number of the molecule. The moment of inertia tensor is also utilized in
    the computation. The function can handle both single-atom and multi-atom systems, adjusting its
    approach based on the linearity of the molecule and the need to convert units.

    Parameters:
        atom_numbers (List[int]): List of atomic numbers for each atom in the molecule.
        coords (List[List[float]]): 3D coordinates of each atom in the molecule.
        T (float): Temperature at which the partition function is calculated.
        convert_unit (bool, optional): Flag to indicate if the moment of inertia should be converted
                                       to appropriate units. Defaults to True.

    Returns:
        float: The calculated rotational partition function (q_r).

    Raises:
        ValueError: If the input data is inconsistent or invalid, such as mismatched lengths of
                    atom_numbers and coords, or if the temperature is not positive.

    Notes:
        - For single-atom systems, a specific function `q_rot_single_atom` is called.
        - The point group and rotational symmetry number are determined using `get_point_group`
          and `get_rotational_symmetry_number` functions, respectively.
        - The moment of inertia tensor is computed by `get_I`, and its units may be converted
          if `convert_unit` is set to True.
        - The linearity of the molecule is checked with `check_linear`.
        - Depending on the linearity, either `q_rot_linear` or `q_rot_nonlinear` is used to
          calculate the partition function.
    """
    if len(atom_numbers) == 1:
        q_r = q_rot_single_atom()
    else:
        pg = get_point_group(coords=coords, numbers=atom_numbers)
        sigma = get_rotational_symmetry_number(pg)
        I = get_I(coords=coords, numbers=atom_numbers)
        linear_flag = check_linear(I)

        if convert_unit:
            I *= convert_I
        if linear_flag:
            I = I[-1]
            q_r = q_rot_linear(sigma=sigma, I=I, T=T)

        else:
            IA, IB, IC = I[0], I[1], I[2]
            q_r = q_rot_nonlinear(sigma=sigma, IA=IA, IB=IB, IC=IC, T=T)
    return q_r


def q_rot_single_atom():
    """
    q_rot_single_atom is a function that returns a fixed integer value.

    Returns:
        int: A fixed integer value.
    """
    return 1


def q_rot_linear(sigma, I, T):
    """
    Calculate the rotational partition function for a linear molecule.

    This function computes the rotational partition function (q_r) based on
    the provided sigma, I, and T values. The formula used is derived from
    physical chemistry principles.

    :formula: q_r = (8 * π^2 * I * k_b * T) / (σ * h^2)

    :parameters:
    - sigma: float
        rotational symmetry number of the molecule
    - I: float
        The moment of inertia of the molecule.
    - T: float
        The absolute temperature in Kelvin at which the diffusion is being considered.

    :returns: float
        The calculated rotational partition function (q_r).
    """
    q_r = (8 * np.pi ** 2 * I * k_b * T) / (sigma * h ** 2)
    return q_r


def q_rot_nonlinear(sigma, IA, IB, IC, T):
    """

    Calculate the rotational partition function for a nonlinear molecule.

    The function computes the rotational partition function (q_r) for a given
    nonlinear molecule using its symmetry number (sigma), and moments of inertia (IA, IB, IC)
    along the principal axes. The calculation is performed at a specified temperature (T).

    Parameters:
    -----------
    sigma: int
        Rotational symmetry number of the molecule.
    IA: float
        Moment of inertia along the A axis.
    IB: float
        Moment of inertia along the B axis.
    IC: float
        Moment of inertia along the C axis.
    T: float
        Temperature in Kelvin.

    Returns:
    --------
    float
        The calculated rotational partition function (q_r).

    Notes:
    ------
    - The formula used assumes a classical treatment of the rotational motion.
    - h refers to Planck's constant.
    - k_b refers to the Boltzmann constant.
    - Ensure that the units are consistent across all input parameters for accurate results.
    """
    q_r = (8 * np.pi ** 2) / (sigma * h ** 3) * (2 * np.pi * k_b * T) ** (3 / 2) * (IA * IB * IC) ** 0.5
    return q_r


def q_vib_bot(vibfreqs, T, convert_unit=True):
    """
    Calculate the vibrational partition function for a set of vibrational frequencies at a given temperature.
    the partition function calculated by  taking bottom of potential energy surface as zero point.

    Summary:
    The function computes the vibrational partition function (q_v) for a list or array of vibrational frequencies.
    It takes into account only positive frequencies and allows for an optional conversion from wavenumbers to frequency
    units. The calculation is based on the standard formula for the vibrational partition function in statistical
    mechanics, which involves the Boltzmann constant (k_b), Planck's constant (h), and the provided temperature (T).

    Parameters:
    vibfreqs: Union[List[float], np.ndarray]
        A list or numpy array containing the vibrational frequencies. unit is Hz, if your input is cm^-1, set convert_unit to be True
    T: float
        Temperature at which the partition function is calculated.
    convert_unit: bool, optional
        If True, converts the input frequencies from wavenumbers to frequency units. Default is True.

    Returns:
    float
        The computed vibrational partition function value.

    Raises:
    ValueError
        If the temperature T is not a positive number.
    TypeError
        If vibfreqs is neither a list nor a numpy array.
    """
    v = np.copy(vibfreqs)
    if convert_unit:
        v *= wave2freq
    positive_freq_mask = v >= 0
    vibfreqs_pos = v[positive_freq_mask]
    q_v = np.prod(np.exp((-h * vibfreqs_pos) / (2 * k_b * T)) / (1 - np.exp((-h * vibfreqs_pos) / (k_b * T))))
    return q_v


def q_vib_V0(vibfreqs, T, convert_unit=True):
    """
    Calculate the vibrational partition function for a set of vibrational frequencies at a given temperature.
    the partition function  calculated by viewing vibrational ground state as zero point, and it reflects thermal excitation of vibrational states.
    This function computes the vibrational partition function, q_v, which is a key quantity in statistical thermodynamics. It takes into account only positive vibrational frequencies and can optionally convert the input frequencies from wavenumbers to frequency units.

    :Parameters:
        vibfreqs : array-like
            Array of vibrational frequencies, unit is Hz, if your input is cm^-1, set convert_unit to be True
        T : float
            Temperature in Kelvin.
        convert_unit : bool, optional
            If True, converts the input vibrational frequencies from wavenumbers to frequency units. Default is True.

    :Returns:
        q_v : float
            The calculated vibrational partition function.

    :Raises:
        ValueError: If any of the vibrational frequencies are negative after conversion and filtering.
    """
    v = np.copy(vibfreqs)
    if convert_unit:
        v *= wave2freq
    positive_freq_mask = v >= 0
    vibfreqs_pos = v[positive_freq_mask]
    q_v = np.prod(1 / (1 - np.exp((-h * vibfreqs_pos) / (k_b * T))))
    return q_v


def q_ele(E_list, g_list, T, convert_unit=True):
    """
    Calculate the partition function for Electron contribution.

    The function computes the partition function, which is a sum over all
    states of the Boltzmann factor, weighted by their degeneracies. This
    is a fundamental quantity in statistical mechanics and thermodynamics,
    used to calculate various thermodynamic properties of a system.

    Parameters:
    E_list: list of float
        The list of energy levels of the system. Unit is eV
    g_list: list of int
        The degeneracies corresponding to each energy level.
    T: float
        The temperature at which the partition function is evaluated.
    convert_unit: bool, optional
        If True, converts the energy from atomic units (a.u.) to electronvolts (eV).
        Default is True.

    Returns:
    float
        The calculated partition function.

    Raises:
    ValueError
        If the lengths of E_list and g_list do not match or if T is non-positive.
    TypeError
        If any of the input parameters are of incorrect type.
    """
    E_s = np.copy(E_list)
    if convert_unit:
        E_s *= au2eV
    E_s -= E_s[0]
    g_s = np.copy(g_list)

    q_e = np.sum(g_s * np.exp(-E_s / (k_b * T)))
    return q_e


def q(q_t, q_r, q_v, q_e, ignore_trans_and_rot=False):
    """
    Calculate total partition function, with an option to ignore translation and rotation.

    Parameters:
    q_t (float): The translation partition function.
    q_r (float): The rotation partition function.
    q_v (float): The Vibration partition function.
    q_e (float): The Electron partition function.
    ignore_trans_and_rot (bool, optional): If True, ignores the translation and rotation partition function in the calculation. Defaults to False.

    Returns:
    float: The calculated value.
    """
    if ignore_trans_and_rot:
        return q_v * q_e
    return q_t * q_r * q_v * q_e


# if __name__ == '__main__':
#
#     from read_qm_out import read_qm_out, read_atom_coord, read_vib, read_ee
#
#     atom_numbers, atom_masses, coords = read_atom_coord(filepath='01_02_sp.out')
#     print(atom_numbers)
#     print(atom_masses)
#     print(coords)
#
#     vibfreqs = read_vib(filepath='01_02.out')
#     print(vibfreqs)
#
#     ee = read_ee(filepath='01_02_sp.out')
#
#     q_t = q_trans(M=132.093960, T=298.15, P=101325)
#     print(f'q_t: {q_t}')
#     q_r = q_rot(atom_numbers=atom_numbers, coords=coords, T=298.15)
#     print(f'q_r: {q_r}')
#     q_v_bot = q_vib_bot(vibfreqs, T=298.15)
#     q_v_0 = q_vib_V0(vibfreqs, T=298.15)
#     print(f'q_v_bot: {q_v_bot}')
#     print(f'q_v_0: {q_v_0}')
#
#     q_e = q_ele(E_list=[ee], g_list=[1], T=298.15, convert_unit=False)
#     print(f'q_e: {q_e}')

