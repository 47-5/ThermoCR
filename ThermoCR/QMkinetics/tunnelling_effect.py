import numpy as np
from scipy.integrate import quad
from ThermoCR.tools.constant import h, k_b, R, cm_1_to_s_1, Na
from numpy import sqrt, cosh, pi, exp


__all__ = ['wigner_correction', 'eckart_correction']


def wigner_correction(imaginary_freq, T, convert_unit=True):
    if convert_unit:
        chi = 1 + 1 / 24 * (h * imaginary_freq * cm_1_to_s_1/ (k_b * T)) ** 2
    else:
        chi = 1 + 1 / 24 * (h * imaginary_freq / (k_b * T)) ** 2
    return chi


def eckart_correction(imaginary_freq, T, delta_H_barrier_f_0K, delta_H_barrier_r_0K, nDOF=300,
                      convert_unit=True):
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

