import numpy as np

from ThermoCR.QMthermo.calc_q import q, q_trans, q_rot, q_vib_bot, q_vib_V0, q_ele
from ThermoCR.tools.constant import R, Na, k_b, au2eV, h, wave2freq, au2kj_mol, Eh


def U_trans(T):
    return 1.5 * R * T


def H_trans(T):
    return U_trans(T) + R * T


def Cv_trans():
    return 1.5 * R


def Cp_trans():
    return Cv_trans() + R


def S_trans(q_t, T):
    S_trans = R * (np.log(q_t / Na) + 5 / 2)
    return S_trans


def U_rot_linear(T):
    return R * T


def H_rot_linear(T):
    return U_rot_linear(T)


def Cv_rot_linear():
    return R


def Cp_rot_linear():
    return Cv_rot_linear()


def S_rot_linear(q_r, T):
    return R * (np.log(q_r) + 1)


def U_rot_nonlinear(T):
    return 1.5 * R * T


def H_rot_nonlinear(T):
    return U_rot_nonlinear(T)


def Cv_rot_nonlinear():
    return 1.5 * R


def Cp_rot_nonlinear():
    return Cv_rot_nonlinear()


def S_rot_nonlinear(q_r, T):
    return R * (np.log(q_r) + 1.5)


def U_ele(E_list, g_list, T, convert_unit=True):
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
    return U_ele(E_list=E_list, g_list=g_list, T=T, convert_unit=convert_unit)


def Cv_ele(E_list, g_list, T, convert_unit=True):
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
    return Cv_ele(E_list=E_list, g_list=g_list, T=T, convert_unit=convert_unit)


def S_ele(E_list, g_list, T, convert_unit=True):
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
    return U_vib_0_T(vibfreqs=vibfreqs, T=T, convert_unit=convert_unit, scale_factor=scale_factor)


def H_vib_T(vibfreqs, T, convert_unit=True, QRRHO=False, scale_factor_zpe=1.0, scale_factor_U_0_T=1.0):
    return U_vib_T(vibfreqs=vibfreqs, T=T, convert_unit=convert_unit, QRRHO=QRRHO,
                   scale_factor_zpe=scale_factor_zpe, scale_factor_U_0_T=scale_factor_U_0_T)


def Cv_vib(vibfreqs, T, convert_unit=True, scale_factor=1.0):
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
    return Cv_vib(vibfreqs=vibfreqs, T=T, convert_unit=convert_unit, scale_factor=scale_factor)


def S_vib(vibfreqs, T, convert_unit=True, QRRHO=True, scale_factor=1.0):
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