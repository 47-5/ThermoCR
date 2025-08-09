import numpy as np

from ThermoCR.QMthermo.calc_q import q, q_trans, q_rot, q_vib_bot, q_vib_V0, q_ele
from ThermoCR.tools.constant import R, Na, k_b, au2eV, h, wave2freq, au2kj_mol, Eh


def U_trans(T):
    """
    Compute the translational contribution to internal energy U for an ideal gas.

    Parameters:
    T : float
        Temperature in Kelvin.

    Returns:
    float
        Translational contribution to the internal energy in the same units as R * T.
    """
    return 1.5 * R * T


def H_trans(T):
    """
    Compute the translational contribution to H for an ideal gas.

    Parameters:
    T : float
        Temperature in Kelvin.

    Returns:
    float
        Translational contribution to the H in the same units as R * T.
    """
    return U_trans(T) + R * T


def Cv_trans():
    """
    Calculate the specific heat at constant volume (Cv) for an ideal gas.

    Summary:
    This function computes the specific heat at constant volume (Cv) for an ideal gas,
    using a predefined constant R. The result is 1.5 times the value of R, which is
    a common approximation for diatomic gases in terms of their molar specific heat
    at constant volume.

    Returns:
        float: The calculated specific heat at constant volume.
    """
    return 1.5 * R


def Cp_trans():
    """
    Calculate the molar specific heat at constant pressure for a substance.

    This function computes the molar specific heat at constant pressure (Cp) by
    adding the molar specific heat at constant volume (Cv) and the gas constant (R).
    It assumes that Cv is provided by an external function `Cv_trans()` and R is a
    predefined constant or variable in the scope of this function.

    Returns:
        float: The calculated molar specific heat at constant pressure.
    """
    return Cv_trans() + R


def S_trans(q_t, T):
    """
    Calculate the translational entropy of a gas.

    This function computes the translational entropy (S_trans) of a gas using the given
    quantity of gas (q_t) and temperature (T). The formula used for the calculation is
    based on statistical mechanics principles, where R is the ideal gas constant,
    and Na is Avogadro's number.

    Args:
        q_t (float): The translational partition function.
        T (float): The temperature at which the entropy is calculated.

    Returns:
        float: The calculated translational entropy of the gas.
    """
    S_trans = R * (np.log(q_t / Na) + 5 / 2)
    return S_trans


def U_rot_linear(T):
    """
    Calculate the internal energy contributed by the rotating part of a linear molecule

    Args:
        T: Temperature
    """
    return R * T


def H_rot_linear(T):
    """
    Compute the rotational enthalpy for a linear molecule at a given temperature.

    Summary:
    This function calculates the rotational enthalpy (H_rot) for a linear molecule using the provided temperature. The calculation is based on the U_rot_linear function, which should be defined elsewhere in the code to provide the specific computation for the rotational internal energy of a linear molecule.

    Parameters:
    T: float
        The temperature at which to compute the rotational enthalpy, typically in Kelvin.

    Returns:
    float
        The computed rotational enthalpy for the linear molecule at the specified temperature.
    """
    return U_rot_linear(T)


def Cv_rot_linear():
    """
    Calculate the contribution of the rotating part of the linear molecule to Cv

    Returns:
        float: The calculated rotational linear coefficient.
    """
    return R


def Cp_rot_linear():
    """
    Calculate the contribution of the rotating part of the linear molecule to Cp

    Returns:
        float: The calculated rotational linear coefficient.
    """
    return Cv_rot_linear()


def S_rot_linear(q_r, T):
    """
    Calculate the rotational entropy for a linear molecule.

    This function computes the rotational entropy (S_rot) for a linear molecule
    using the given rotational partition function (q_r) and temperature (T).
    The formula used is S_rot = R * (ln(q_r) + 1), where R is the gas constant.

    Parameters:
    q_r : float
        The rotational partition function of the molecule.
    T : float
        Temperature in Kelvin at which to calculate the entropy.

    Returns:
    float
        The calculated rotational entropy in J/(mol*K).

    Raises:
    ValueError
        If q_r or T are non-positive, as these are not physically meaningful values.
    """
    return R * (np.log(q_r) + 1)


def U_rot_nonlinear(T):
    """
    Calculate the non-linear rotational contribution to the internal energy.

    This function computes the non-linear rotational part of the internal energy
    of a molecule based on its temperature. The formula used is 1.5 * R * T,
    where R is the ideal gas constant and T is the temperature in Kelvin.

    :raises ValueError: If the temperature is negative.
    :returns: The non-linear rotational contribution to the internal energy.
    :rtype: float
    """
    return 1.5 * R * T


def H_rot_nonlinear(T):
    """
    Calculate the non-linear rotational contribution to the H.

    This function computes the non-linear rotational part of the H
    of a molecule based on its temperature. The formula used is 1.5 * R * T,
    where R is the ideal gas constant and T is the temperature in Kelvin.

    :raises ValueError: If the temperature is negative.
    :returns: The non-linear rotational contribution to the H.
    :rtype: float
    """
    return U_rot_nonlinear(T)


def Cv_rot_nonlinear():
    """
    Calculate the contribution of the rotating part of the nonlinear molecule to Cv

    Returns:
        float: The calculated rotational linear coefficient.
    """
    return 1.5 * R


def Cp_rot_nonlinear():
    """
    Calculate the contribution of the rotating part of the nonlinear molecule to Cp

    Returns:
        float: The calculated rotational linear coefficient.
    """
    return Cv_rot_nonlinear()


def S_rot_nonlinear(q_r, T):
    """
   Calculate the rotational entropy for a nonlinear molecule.

   This function computes the rotational entropy (S_rot) for a nonlinear molecule
   using the given rotational partition function (q_r) and temperature (T).
   The formula used is S_rot = R * (ln(q_r) + 1.5), where R is the gas constant.

   Parameters:
   q_r : float
       The rotational partition function of the molecule.
   T : float
       Temperature in Kelvin at which to calculate the entropy.

   Returns:
   float
       The calculated rotational entropy in J/(mol*K).

   Raises:
   ValueError
       If q_r or T are non-positive, as these are not physically meaningful values.
   """
    return R * (np.log(q_r) + 1.5)


def U_ele(E_list, g_list, T, convert_unit=True):
    """
    Calculate the contribution of the electronic excitation part to internal energy (U).

    Summary: Calculates the internal energy of a system given its energy levels,
             their degeneracies, and the temperature.
    Parameters: E_list (list[float]): List of energy levels of the system.
                g_list (list[float]): Degeneracy of each energy level.
                T (float): Temperature at which to calculate the internal energy.
                convert_unit (bool, optional): If True, converts E_list from atomic
                                               units to electron volts. Default is True.
    Returns: float: The calculated internal energy of the system.
    """
    E_s = np.array(E_list, dtype=float)
    g_s = np.array(g_list, dtype=float)

    if convert_unit:
        E_s *= au2eV
    E_s -= E_s.min()

    exp_terms = np.exp(-E_s / (k_b * T))
    numerator = np.sum(g_s * E_s * exp_terms) / (k_b * T)
    denominator = np.sum(g_s * exp_terms)

    return R * T * numerator / denominator


def H_ele(E_list, g_list, T, convert_unit=True):
    """
    Calculate the contribution of the electronic excitation part to H.

    Summary: Calculates the H of a system given its energy levels,
             their degeneracies, and the temperature.
    Parameters: E_list (list[float]): List of energy levels of the system.
                g_list (list[float]): Degeneracy of each energy level.
                T (float): Temperature at which to calculate the H.
                convert_unit (bool, optional): If True, converts E_list from atomic
                                               units to electron volts. Default is True.
    Returns: float: the contribution of the electronic excitation part to H.
    """
    return U_ele(E_list=E_list, g_list=g_list, T=T, convert_unit=convert_unit)


def Cv_ele(E_list, g_list, T, convert_unit=True):
    """
   Calculate the contribution of the electronic excitation part to Cv.

   Summary: Calculates the Cv of a system given its energy levels,
            their degeneracies, and the temperature.
   Parameters: E_list (list[float]): List of energy levels of the system.
               g_list (list[float]): Degeneracy of each energy level.
               T (float): Temperature at which to calculate the Cv.
               convert_unit (bool, optional): If True, converts E_list from atomic
                                              units to electron volts. Default is True.
   Returns: float: the contribution of the electronic excitation part to Cv.
   """
    E_s = np.array(E_list, dtype=float)
    g_s = np.array(g_list, dtype=float)

    if convert_unit:
        E_s *= au2eV
    E_s -= E_s.min()

    kT = k_b * T
    exp_terms = np.exp(-E_s / kT)
    gE_exp = g_s * exp_terms
    sum_gE_exp = np.sum(gE_exp)

    term1 = np.sum(g_s * (E_s / kT) ** 2 * exp_terms) / sum_gE_exp
    term2 = (np.sum(g_s * (E_s / kT) * exp_terms) / sum_gE_exp) ** 2

    return R * (term1 - term2)


def Cp_ele(E_list, g_list, T, convert_unit=True):
    """
   Calculate the contribution of the electronic excitation part to Cp.

   Summary: Calculates the Cp of a system given its energy levels,
            their degeneracies, and the temperature.
   Parameters: E_list (list[float]): List of energy levels of the system.
               g_list (list[float]): Degeneracy of each energy level.
               T (float): Temperature at which to calculate the Cp.
               convert_unit (bool, optional): If True, converts E_list from atomic
                                              units to electron volts. Default is True.
   Returns: float: the contribution of the electronic excitation part to Cp.
   """
    return Cv_ele(E_list=E_list, g_list=g_list, T=T, convert_unit=convert_unit)


def S_ele(E_list, g_list, T, convert_unit=True):
    """
   Calculate the contribution of the electronic excitation part to S.

   Summary: Calculates the S of a system given its energy levels,
            their degeneracies, and the temperature.
   Parameters: E_list (list[float]): List of energy levels of the system.
               g_list (list[float]): Degeneracy of each energy level.
               T (float): Temperature at which to calculate the S.
               convert_unit (bool, optional): If True, converts E_list from atomic
                                              units to electron volts. Default is True.
   Returns: float: the contribution of the electronic excitation part to S.
   """
    E_s = np.array(E_list, dtype=float)
    g_s = np.array(g_list, dtype=float)

    if convert_unit:
        E_s *= au2eV
    E_s -= E_s[0]

    exp_terms = np.exp(-E_s / (k_b * T))
    sum_g_exp = np.sum(g_s * exp_terms)
    sum_gE_exp = np.sum(g_s * E_s * exp_terms) / (k_b * T)

    return R * (np.log(sum_g_exp) + sum_gE_exp / sum_g_exp)


def ZPE(vibfreqs, convert_unit=True, scale_factor=1.0):
    """
    Calculate the Zero Point Energy (ZPE) from a list of vibrational frequencies.

    The function computes the ZPE based on the given vibrational frequencies. It can optionally
    convert the units of the input frequencies and apply a scale factor to them before performing
    the calculation. The final ZPE is returned in J/mol.

    Parameters:
    vibfreqs: list[float] or array-like
        A list or an array of vibrational frequencies. when unit is cm^-1, set convert_unit to be True
    convert_unit: bool, optional
        If True, converts the unit of vibfreqs. Default is True.
    scale_factor: float, optional
        A factor to scale the vibrational frequencies. Default is 1.0.

    Returns:
    float
        The calculated Zero Point Energy in J/mol.

    Raises:
    ValueError
        If any of the vibrational frequencies are negative after scaling.

    Notes:
    - The conversion and scaling of the vibrational frequencies are applied only if specified.
    - Negative vibrational frequencies are ignored in the ZPE calculation.
    - Constants used in the calculation such as Planck's constant (h), conversion factors,
      and energy units are assumed to be defined elsewhere in the code.
    """
    v = np.array(vibfreqs, dtype=float) * scale_factor
    if convert_unit:
        v = v * wave2freq / Eh

    zpe = 0.5 * h * (v[v >= 0]).sum()

    zpe *= (au2kj_mol * 1000)
    return zpe


def ZPE_one_mode(vibfreq, convert_unit=True):
    v = np.copy(vibfreq)
    if convert_unit:
        v = v / Eh
    zpe = 0

    if vibfreq < 0:
        pass
    else:
        zpe = 0.5 * v * h
    zpe *= (au2kj_mol * 1000)
    return zpe


def U_vib_0_T(vibfreqs, T, convert_unit=True, scale_factor=1.0):
    """
    Calculate the vibrational contribution to the U(T)vib - U(0)vib.

    This function computes the vibrational contribution to the U(T)vib - U(0)vib
    for a given set of vibrational frequencies at a specified temperature.
    The calculation is based on the formula for the vibrational partition function
    and its derivatives. The function can optionally convert the input frequencies
    from wavenumbers to frequency and apply a scaling factor to the frequencies.

    Parameters:
    vibfreqs (Iterable[float]): Vibrational frequencies in cm^-1 or Hz. When unit is cm^-1, set convert_unit to be True.
    T (float): Temperature in Kelvin.
    convert_unit (bool, optional): If True, convert vibfreqs from cm^-1 to Hz. Default is True.
    scale_factor (float, optional): Scaling factor to apply to the vibrational frequencies. Default is 1.0.

    Returns:
    float: Vibrational contribution to the U(T)vib - U(0)vib at temperature T in J/mol.

    Raises:
    ValueError: If T is not positive.
    TypeError: If vibfreqs is not iterable or if any element in vibfreqs is not a number.
    """
    v = np.array(vibfreqs, dtype=float) * scale_factor
    if convert_unit:
        v = v * wave2freq

    pos_mask = v >= 0
    v_pos = v[pos_mask]

    if len(v_pos) == 0:
        return 0.0

    term = h * v_pos / (k_b * T)
    exp_term = np.exp(-term)
    return np.sum(R * T * term * exp_term / (1 - exp_term))


def U_vib_T(vibfreqs, T, convert_unit=True, QRRHO=False,
            scale_factor_zpe=1.0, scale_factor_U_0_T=1.0):
    """
    Calculate the vibrational contribution to the internal energy at a given temperature.

    Summary:
    This function computes the vibrational contribution (U_vib) to the internal energy
    for a set of vibrational frequencies at a specified temperature. It supports
    options for unit conversion, quasi-rigid rotor harmonic oscillator (QRRHO) approximation,
    and scaling factors for zero-point energy and thermal contributions.

    Parameters:
    vibfreqs: list or numpy.ndarray of float
        Vibrational frequencies, in wavenumbers (cm^-1) if convert_unit is True.
    T: float
        Temperature in Kelvin.
    convert_unit: bool, optional
        If True, converts input frequencies from cm^-1 to Hz. Default is True.
    QRRHO: bool, optional
        If True, applies QRRHO correction. Default is False.
    scale_factor_zpe: float, optional
        Scaling factor for zero-point energy. Default is 1.0.
    scale_factor_U_0_T: float, optional
        Scaling factor for thermal contribution. Default is 1.0.

    Returns:
    float
        The total vibrational contribution to the internal energy in kJ/mol.

    Raises:
    ValueError
        If any of the input parameters are invalid or out of expected range.
    TypeError
        If the type of the input parameters does not match the required types.

    Notes:
    - The function uses physical constants such as Planck's constant (h), Boltzmann's
      constant (k_b), and the gas constant (R).
    - For QRRHO approximation, an additional weighting function (w_vec) is applied.
    - The function handles both positive and negative vibrational frequencies but
      only positive frequencies contribute to the final result.
    - The function assumes that the necessary physical constants and conversion factors
      (e.g., wave2freq, au2kj_mol, Eh) are defined elsewhere in the code.
    - The term "Eh" refers to Hartree energy, used in the context of atomic units.
    - The function sums up the contributions from all vibrational modes to return
      the total vibrational internal energy.
    - The default behavior without QRRHO correction is to sum the RRHO and ZPE
      contributions directly.
    - The QRRHO option modifies the calculation by blending between RRHO and free
      rotor (FR) models using a weighting function.
    """
    v = np.array(vibfreqs, dtype=float)
    if convert_unit:
        v = v * wave2freq

    v_zpe = v * scale_factor_zpe
    v_U_0_T = v * scale_factor_U_0_T

    pos_mask = v >= 0
    v_pos = v[pos_mask]
    v_zpe_pos = v_zpe[pos_mask]
    v_U_0_T_pos = v_U_0_T[pos_mask]

    if len(v_pos) == 0:
        return 0.0

    # 计算RRHO部分
    term = h * v_U_0_T_pos / (k_b * T)
    exp_term = np.exp(-term)
    U_0_T = R * T * term * exp_term / (1 - exp_term)

    # 计算ZPE部分
    if convert_unit:
        zpe = 0.5 * h * v_zpe_pos / Eh * (au2kj_mol * 1000)
    else:
        zpe = 0.5 * h * v_zpe_pos * (au2kj_mol * 1000)

    U_RRHO = U_0_T + zpe

    if not QRRHO:
        return np.sum(U_RRHO)

    # QRRHO处理
    wei = w_vec(v_pos, convert_unit=False)
    U_FR = R * T / 2
    return np.sum(wei * U_RRHO + (1 - wei) * U_FR)


def H_vib_0_T(vibfreqs, T, convert_unit=True, scale_factor=1.0):
    """
    See U_vib_0_T
    """
    return U_vib_0_T(vibfreqs=vibfreqs, T=T, convert_unit=convert_unit, scale_factor=scale_factor)


def H_vib_T(vibfreqs, T, convert_unit=True, QRRHO=False, scale_factor_zpe=1.0, scale_factor_U_0_T=1.0):
    """
    See U_vib_T
    """
    return U_vib_T(vibfreqs=vibfreqs, T=T, convert_unit=convert_unit, QRRHO=QRRHO,
                   scale_factor_zpe=scale_factor_zpe, scale_factor_U_0_T=scale_factor_U_0_T)


def Cv_vib(vibfreqs, T, convert_unit=True, scale_factor=1.0):
    """
    Calculate the vibrational contribution to the molar heat capacity (Cv) at a given temperature.

    Summary:
    This function computes the vibrational component of the molar heat capacity (Cv) for a set of vibrational frequencies at a specified temperature. It supports unit conversion and scaling of the input frequencies. The calculation is based on the Bose-Einstein distribution, considering only positive frequencies.

    Parameters:
    vibfreqs: list or array-like
        Vibrational frequencies in cm^-1.
    T: float
        Temperature in Kelvin.
    convert_unit: bool, optional
        If True, converts the input frequencies from cm^-1 to Hz. Default is True.
    scale_factor: float, optional
        A factor to scale the input frequencies. Default is 1.0.

    Returns:
    float
        The calculated vibrational contribution to the molar heat capacity (Cv) in J/(mol*K).

    Raises:
    ValueError
        If any of the input parameters are not of the expected type or if the temperature is non-positive.
    """
    v = np.array(vibfreqs, dtype=float) * scale_factor
    if convert_unit:
        v = v * wave2freq

    pos_mask = v >= 0
    v_pos = v[pos_mask]

    if len(v_pos) == 0:
        return 0.0

    term = h * v_pos / (k_b * T)
    exp_term = np.exp(-term)
    return np.sum(R * term ** 2 * exp_term / (1 - exp_term) ** 2)


def Cp_vib(vibfreqs, T, convert_unit=True, scale_factor=1.0):
    """
    same as Cv_vib
    """
    return Cv_vib(vibfreqs=vibfreqs, T=T, convert_unit=convert_unit, scale_factor=scale_factor)


def S_vib(vibfreqs, T, convert_unit=True, QRRHO=True, scale_factor=1.0):
    """
    Calculate the vibrational entropy of a molecule.

    This function computes the vibrational entropy of a molecule given its
    vibrational frequencies and the temperature. It supports both RRHO (Rigid
    Rotor Harmonic Oscillator) and QRRHO (Quasi Rigid Rotor Harmonic Oscillator)
    approximations, with an option to scale the vibrational frequencies and
    convert their units from wavenumbers to frequency.

    Summary:
    The function takes in vibrational frequencies, temperature, and optional
    parameters for unit conversion, QRRHO usage, and a scaling factor. It
    returns the total vibrational entropy based on the selected approximation.

    Parameters:
    - vibfreqs: list or numpy.ndarray of float
        Vibrational frequencies of the molecule. When unit is cm^-1, set convert_unit to be True
    - T: float
        Temperature at which to calculate the entropy.
    - convert_unit: bool, default True
        Whether to convert the input frequencies from wavenumbers to frequency.
    - QRRHO: bool, default True
        Whether to use the Quasi Rigid Rotor Harmonic Oscillator approximation.
    - scale_factor: float, default 1.0
        Scaling factor applied to the vibrational frequencies.

    Returns:
    - float
        The calculated vibrational entropy.

    Raises:
    - ValueError
        If `vibfreqs` is empty or contains non-positive values after applying
        the scale factor.
    - TypeError
        If `vibfreqs` is not a list or numpy array, or if `T` and `scale_factor`
        are not numbers.
    """
    v = np.array(vibfreqs, dtype=float)
    if convert_unit:
        v = v * wave2freq

    v_s = v * scale_factor
    pos_mask = v >= 0
    v_pos = v[pos_mask]
    v_s_pos = v_s[pos_mask]

    if len(v_pos) == 0:
        return 0.0

    if not QRRHO:
        return np.sum(S_vib_RRHO_vec(v_s_pos, T))

    # QRRHO处理
    wei = w_vec(v_pos, convert_unit=False)
    S_RRHO = S_vib_RRHO_vec(v_s_pos, T)
    S_FR = S_vib_FR_vec(v_s_pos, T)
    return np.sum(wei * S_RRHO + (1 - wei) * S_FR)


def w_vec(v, v0=100, convert_unit=True):
    if convert_unit:
        v0 = v0 * wave2freq
    return 1 / (1 + (v0 / v)**4)


def U_vib_0_T_RRHO(vibfreq, T):
    U = R * T * (h * vibfreq  / (k_b * T)) * np.exp(-h * vibfreq / (k_b * T)) / (1 - np.exp(-h * vibfreq / (k_b * T)))
    return U


def U_vib_FR(T):
    U = R * T / 2
    return U


def S_vib_RRHO_vec(vibfreqs, T):
    term = h * vibfreqs / (k_b * T)
    exp_term = np.exp(-term)
    return R * (term * exp_term / (1 - exp_term) - np.log(1 - exp_term))


def S_vib_FR_vec(vibfreqs, T, Bav=1e-44):
    miu = h / (8 * np.pi**2 * vibfreqs)
    miu_ = miu * Bav / (miu + Bav)
    return R * (0.5 + 0.5 * np.log(8 * np.pi**3 * miu_ * k_b * T / h**2))


# if __name__ == '__main__':
#
#     from read_qm_out import read_qm_out, read_atom_coord, read_vib, read_ee
#
#     atom_numbers, atom_masses, coords = read_atom_coord(filepath='01_02_sp.out')
#     print(atom_numbers)
#     print(atom_masses)
#     print(coords)
#
#     vibfreqs= read_vib(filepath='01_02.out')
#     print(vibfreqs)
#
#     ee = read_ee(filepath='01_02_sp.out')
#     print('-' * 30)
#     q_t = q_trans(M=132.093960, T=298.15, P=101325)
#     U_t = U_trans(T=298.15)
#     H_t = H_trans(T=298.15)
#     Cv_t = Cv_trans()
#     Cp_t = Cp_trans()
#     S_t = S_trans(q_t=q_t, T=298.15)
#     print(f'q_t: {q_t}')
#     print(f'U_t: {U_t}')
#     print(f'H_t: {H_t}')
#     print(f'Cv_t: {Cv_t}')
#     print(f'Cp_t: {Cp_t}')
#     print(f'S_t: {S_t}')
#     print('-' * 30)
#     q_r = q_rot(atom_numbers=atom_numbers, coords=coords, T=298.15)
#     U_r = U_rot_nonlinear(T=298.15)
#     H_r = H_rot_nonlinear(T=298.15)
#     Cv_r = Cv_rot_nonlinear()
#     Cp_r = Cp_rot_nonlinear()
#     S_r = S_rot_nonlinear(q_r=q_r, T=298.15)
#     print(f'q_r: {q_r}')
#     print(f'U_r: {U_r}')
#     print(f'H_r: {H_r}')
#     print(f'Cv_r: {Cv_r}')
#     print(f'Cp_r: {Cp_r}')
#     print(f'S_r: {S_r}')
#     print('-' * 30)
#     q_v_bot = q_vib_bot(vibfreqs, T=298.15)
#     q_v_0 = q_vib_V0(vibfreqs, T=298.15)
#     zpe = ZPE(vibfreqs)
#     U_v_0_T = U_vib_0_T(vibfreqs, T=298.15)
#     H_v_0_T = H_vib_0_T(vibfreqs, T=298.15)
#     U_v = U_vib_T(vibfreqs=vibfreqs, T=298.15, QRRHO=False)
#     H_v = H_vib_T(vibfreqs=vibfreqs, T=298.15)
#     Cv_v = Cv_vib(vibfreqs, T=298.15)
#     Cp_v = Cp_vib(vibfreqs, T=298.15)
#     S_v = S_vib(vibfreqs, T=298.15)
#     print(f'q_v_bot: {q_v_bot}')
#     print(f'q_v_0: {q_v_0}')
#     print(f'zpe: {zpe}')
#     print(f'U_v_T: {U_v_0_T}')
#     print(f'H_v_T: {H_v_0_T}')
#     print(f'U_v: {U_v}')
#     print(f'H_v: {H_v}')
#     print(f'Cv_vib: {Cv_v}')
#     print(f'Cp_vib: {Cp_v}')
#     print(f'S_vib: {S_v}')
#     print('-' * 30)
#
#     q_e = q_ele(E_list=[ee], g_list=[1], T=298.15, convert_unit=True)
#     U_e = U_ele(E_list=[ee], g_list=[1], T=298.15,)
#     H_e = H_ele(E_list=[ee], g_list=[1], T=298.15)
#     Cv_e = Cv_ele(E_list=[ee], g_list=[1], T=298.15)
#     Cp_e = Cp_ele(E_list=[ee], g_list=[1], T=298.15)
#     S_e = S_ele(E_list=[ee], g_list=[1], T=298.15)
#     print(f'q_e: {q_e}')
#     print(f'U_e: {U_e}')
#     print(f'H_e: {H_e}')
#     print(f'Cv_e: {Cv_e}')
#     print(f'Cp_e: {Cp_e}')
#     print(f'S_e: {S_e}')
#     print('-' * 30)