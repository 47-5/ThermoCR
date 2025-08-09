import numpy as np
from scipy.integrate import quad
from ThermoCR.tools.constant import h, k_b, R, cm_1_to_s_1, Na
from numpy import sqrt, cosh, pi, exp


__all__ = ['wigner_correction', 'eckart_correction', 'skodje_truhlar']


def wigner_correction(imaginary_freq, T, convert_unit=True):
    """
    Calculates the Wigner correction factor for a given imaginary frequency and temperature. The function
    optionally converts the imaginary frequency from cm^-1 to s^-1 based on the `convert_unit` parameter.

    Args:
        imaginary_freq: float, the imaginary frequency of the system, typically in units of cm^-1 unless
            `convert_unit` is set to False.
        T: float, the temperature at which the correction is being calculated, in Kelvin.
        convert_unit: bool, optional, a flag to determine whether to convert the imaginary frequency from
            cm^-1 to s^-1. Defaults to True.

    Returns:
        float, the Wigner correction factor, chi, which adjusts the partition function or other thermodynamic
        quantities to account for quantum effects at low temperatures.

    Raises:
        ValueError: If `T` is not a positive number, indicating an invalid temperature.
    """
    if convert_unit:
        chi = 1 + 1 / 24 * (h * imaginary_freq * cm_1_to_s_1/ (k_b * T)) ** 2
    else:
        chi = 1 + 1 / 24 * (h * imaginary_freq / (k_b * T)) ** 2
    return chi


def eckart_correction(imaginary_freq, T, delta_H_barrier_f_0K, delta_H_barrier_r_0K, nDOF=300,
                      convert_unit=True):
    """
    Computes the Eckart correction for a given set of parameters, which is used in
    transition state theory to correct the partition functions for reactions. The
    correction accounts for the anharmonicity and the coupling between the reaction
    coordinate and the vibrational modes of the molecule.

    Args:
        imaginary_freq: float, the imaginary frequency associated with the transition
            state. If convert_unit is True, it should be in cm^-1; otherwise, in s^-1.
        T: float, the temperature in Kelvin at which the correction is to be computed.
        delta_H_barrier_f_0K: float, the forward barrier height (enthalpy difference)
            at 0 K. If convert_unit is True, it should be in kJ/mol; otherwise, in J.
        delta_H_barrier_r_0K: float, the reverse barrier height (enthalpy difference)
            at 0 K. If convert_unit is True, it should be in kJ/mol; otherwise, in J.
        nDOF: int, optional, the number of degrees of freedom, default is 300.
        convert_unit: bool, optional, if True, converts the input units from kJ/mol
            and cm^-1 to J and s^-1 respectively, default is True.

    Returns:
        chi: float, the computed Eckart correction factor.

    Raises:
        ValueError: If any of the input parameters are outside their expected range
            or if the computation fails due to numerical instability.
    """
    nthEpsilon = 1e4  # 用于判断 Boltzmann 分布终止处
    if convert_unit:
        imaginary_freq *= cm_1_to_s_1
        delta_H_barrier_f_0K /= Na
        delta_H_barrier_r_0K /= Na

    A = delta_H_barrier_f_0K - delta_H_barrier_r_0K
    B = (sqrt(delta_H_barrier_f_0K) + sqrt(delta_H_barrier_r_0K)) ** 2
    cRoot = 6.6260696e-34 * imaginary_freq * sqrt(B ** 3 / (A ** 2 - B ** 2) ** 2)
    C = 1.0 / (2.0 * cRoot)
    D = C * sqrt(B - cRoot ** 2)

    # 能量积分上限估计
    if A >= 0:
        # Endothermic or athermic
        threeBarrierEnergy = 3 * delta_H_barrier_r_0K
        preFactor = exp(delta_H_barrier_r_0K / (k_b * T)) / (k_b * T)
        shiftE = A
    else:
        # Exothermic
        threeBarrierEnergy = 3 * delta_H_barrier_f_0K
        preFactor = exp(delta_H_barrier_f_0K / (k_b * T)) / (k_b * T)
        shiftE = 0.0

    energyUpperLimit = max(nDOF * k_b * T, threeBarrierEnergy)

    # 找到波尔兹曼分布的“峰值终止点”
    nStep = int(energyUpperLimit / (k_b * T))
    peakEnd = 0.0
    yMax = -1.0

    for i in range(1, nStep):
        x = i * k_b * T
        y = pE_exp(x, shiftE, C, A, D, T)
        if y > yMax:
            yMax = y
        elif y < yMax / nthEpsilon:
            peakEnd = x
            break

    # 使用 Simpson's Rule 进行积分
    a = 0.0
    b = peakEnd
    N = int(1 + (b - a) * 5 / (k_b * T))
    if N % 2 == 0:
        N += 1  # Simpson's rule 要求奇数点

    h = (b - a) / (N - 1)
    sum_y = (1.0 / 3.0) * (pE_exp(a, shiftE, C, A, D, T) + pE_exp(b, shiftE, C, A, D, T))

    for i in range(1, N - 1, 2):
        x = a + h * i
        sum_y += (4.0 / 3.0) * pE_exp(x, shiftE, C, A, D, T)

    for i in range(2, N - 1, 2):
        x = a + h * i
        sum_y += (2.0 / 3.0) * pE_exp(x, shiftE, C, A, D, T)

    chi = preFactor * sum_y * h

    return chi


def pE_exp(x, shiftE, C, A, D, T):
    alpha = C * sqrt(x + shiftE)
    beta = C * sqrt(x + shiftE - A)
    term1 = cosh(2 * pi * (alpha - beta)) + cosh(2 * pi * D)
    term2 = cosh(2 * pi * (alpha + beta)) + cosh(2 * pi * D)
    permeability = 1.0 - (term1 / term2)
    boltzmann = exp(-x / (k_b * T))
    return permeability * boltzmann


def skodje_truhlar(imaginary_freq, T, delta_H_barrier_f_0K, delta_H_barrier_r_0K,
                   convert_unit=True):
    """
    Calculates the Skodje-Truhlar tunneling correction factor for a chemical reaction.
    This function computes the tunneling correction factor using the Skodje-Truhlar
    approximation, which is particularly useful in reactions involving a small barrier
    or when the reaction occurs at low temperatures.

    Args:
        imaginary_freq (float): The imaginary frequency of the transition state. If convert_unit is True, it should be in cm^-1; otherwise, in s^-1.
        T (float): The temperature in Kelvin at which the reaction takes place.
        delta_H_barrier_f_0K (float): The forward activation enthalpy at 0 K.
        delta_H_barrier_r_0K (float): The reverse activation enthalpy at 0 K.
        convert_unit (bool, optional): A flag to indicate whether the imaginary
            frequency should be converted from cm^-1 to s^-1. Defaults to True.

    Returns:
        float: The tunneling correction factor (chi) that accounts for quantum
            mechanical tunneling through the reaction barrier.

    Raises:
        ValueError: If the provided temperature is not positive.
    """
    if convert_unit:
        imaginary_freq *= cm_1_to_s_1

    alpha = 2 * pi / (h * imaginary_freq)
    beta = 1 / (k_b * T)

    delta_H_reaction_0K = delta_H_barrier_f_0K - delta_H_barrier_r_0K
    V = 0 if delta_H_reaction_0K < 0 else delta_H_reaction_0K
    delta_V = delta_H_barrier_f_0K

    if alpha > beta:
        term1 = beta * pi / alpha
        chi = term1 / np.sin(term1) - beta / (alpha - beta) * np.exp((beta - alpha) * (delta_V - V))
    else:
        chi = beta / (beta - alpha) * (np.exp((beta - alpha) - (delta_V - V)) - 1)
    return chi


# if __name__ == '__main__':
#
#     chi = wigner_correction(438.7819, 500)
#     print(chi)
#
#     chi = eckart_correction(438.7819, 500, delta_H_barrier_f_0K=64180, delta_H_barrier_r_0K=160960, convert_unit=True)
#     print(chi)
#
#     chi = skodje_truhlar(438.7819, 500, delta_H_barrier_f_0K=64180, delta_H_barrier_r_0K=160960, convert_unit=True)
#     print(chi)

