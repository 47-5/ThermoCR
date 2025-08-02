import numpy as np

from constant import k_b, h, R, amu2kg, convert_I, wave2freq, au2eV
from utils import get_point_group, get_I, check_linear, get_rotational_symmetry_number


def q_trans(M, T, P, convert_unit=True):
    if convert_unit:
        q_t = (R * T / P) * ((2 * np.pi * M * amu2kg * k_b * T) / (h ** 2)) ** 1.5
    else:
        q_t = (R * T / P) * ((2 * np.pi * M * k_b * T) / (h ** 2)) ** 1.5
    return q_t


def q_rot(atom_numbers, coords, T, convert_unit=True):
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
    return 1


def q_rot_linear(sigma, I, T):
    q_r = (8 * np.pi ** 2 * I * k_b * T) / (sigma * h ** 2)
    return q_r


def q_rot_nonlinear(sigma, IA, IB, IC, T):
    q_r = (8 * np.pi ** 2) / (sigma * h ** 3) * (2 * np.pi * k_b * T) ** (3 / 2) * (IA * IB * IC) ** 0.5
    return q_r


def q_vib_bot(vibfreqs, T, convert_unit=True):
    v = np.copy(vibfreqs)
    if convert_unit:
        v *= wave2freq
    q_v = 1
    for vibfreq in v:
        if vibfreq < 0:
            pass
        else:
            q_v *= (np.exp((-h * vibfreq) / (2 * k_b * T)) / (1 - np.exp((-h * vibfreq) / (k_b * T))))
    return q_v


def q_vib_V0(vibfreqs, T, convert_unit=True):
    v = np.copy(vibfreqs)
    if convert_unit:
        v *= wave2freq
    q_v = 1
    for vibfreq in v:
        if vibfreq < 0:
            pass
        else:
            q_v *= 1 / (1 - np.exp((-h * vibfreq) / (k_b * T)))
    return q_v


def q_ele(E_list, g_list, T, convert_unit=True):
    E_s = np.copy(E_list)
    if convert_unit:
        E_s *= au2eV
    E_s -= E_s[0]
    g_s = np.copy(g_list)

    q_e = 0
    for E, g in zip(E_s, g_s):
        q_e += g * np.exp(-E / (k_b * T))
    return q_e


def q(q_t, q_r, q_v, q_e):
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

