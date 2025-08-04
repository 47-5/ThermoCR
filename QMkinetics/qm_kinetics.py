import numpy as np
import pandas as pd
from tools.constant import k_b, h, R
from QMkinetics.tunnelling_effect import wigner_correction, eckart_correction


__all__ = ['k_TST', 'k_VTST', 'k_TST_scan']


def k_TST(delta_G, delta_n, T=298.15, P0=100000, sigma=1, tunnelling_effect=None, imaginary_freq=None,
          delta_H_barrier_f_0K=None, delta_H_barrier_r_0K=None):
    """
    calculate k base on TST

    :param delta_G: G_TS - G_IS / (J/mol)
    :param delta_n: 双分子反应取1，单分子反应取0
    :param T: K
    :param P0: 标准压力 100000 Pa = 1 bar
    :param sigma: 反应路径简并度 = sigma_rot_TS / sigma_rot_IS 如果计算G时已经考虑转动对称数，这里应该设置为1
    :param delta_H_barrier_f_0K:
    :param delta_H_barrier_r_0K:
    :param imaginary_freq:
    :param tunnelling_effect:
    :return: k:双分子反应单位: s^-1 * (molecule / m^3)
             k:单分子反应单位: s^-1
    """

    k = sigma * k_b * T / h * (k_b * T / P0) ** delta_n * np.exp(-delta_G / (R * T))
    if tunnelling_effect is not None:
        assert imaginary_freq is not None, 'imaginary_freq must set when considering tunnelling effect'
        if tunnelling_effect == 'wigner':
            chi = wigner_correction(imaginary_freq=imaginary_freq, T=T, convert_unit=True)
        elif tunnelling_effect == 'eckart':
            assert delta_H_barrier_r_0K is not None, 'delta_H_barrier_r_0K must set when considering tunnelling effect by eckart method'
            assert delta_H_barrier_f_0K is not None, 'delta_H_barrier_f_0K must set when considering tunnelling effect by eckart method'
            chi = eckart_correction(imaginary_freq=imaginary_freq, T=T,
                                    delta_H_barrier_f_0K=delta_H_barrier_f_0K, delta_H_barrier_r_0K=delta_H_barrier_r_0K)
        else:
            raise NotImplemented(f'{tunnelling_effect} is not a right value')

        k *= chi

    return k


def k_TST_scan(thermo_ts_path, thermo_r1_path, thermo_r2_path=None, thermo_p_path=None,
               tunnelling_effect=None, imaginary_freq=None,
               sigma=1):
    ts_thermo_df = pd.read_excel(thermo_ts_path)
    r1_thermo_df = pd.read_excel(thermo_r1_path)
    if thermo_r2_path is not None:
        r2_thermo_df = pd.read_excel(thermo_r2_path)
        delta_n = 1
        delta_G = ts_thermo_df['G/(J/mol)'] - r1_thermo_df['G/(J/mol)'] - r2_thermo_df['G/(J/mol)']
    else:
        delta_n = 0
        delta_G = ts_thermo_df['G/(J/mol)'] - r1_thermo_df['G/(J/mol)']
    T = ts_thermo_df['T/K']



    if tunnelling_effect is not None:
        assert imaginary_freq is not None, 'imaginary_freq must set when considering tunnelling effect'

        if tunnelling_effect == 'wigner':
            k_scan = np.array(
                [
                    k_TST(delta_G=delta_g, delta_n=delta_n, T=t, sigma=sigma, tunnelling_effect='wigner', imaginary_freq=imaginary_freq)
                    for delta_g, t in zip(delta_G, T)
                ]
            )

        elif tunnelling_effect == 'eckart':
            assert thermo_p_path is not None, 'thermo_p_path must be set when considering tunneling effect by eckart method'
            p_thermo_df = pd.read_excel(thermo_p_path)
            if thermo_r2_path is not None:
                delta_H_barrier_f_0K_scan = ts_thermo_df['ee/(J/mol)'] + ts_thermo_df['zpe/(J/mol)'] - \
                                        (r1_thermo_df['ee/(J/mol)'] + r1_thermo_df['zpe/(J/mol)']) - \
                                        (r2_thermo_df['ee/(J/mol)'] + r2_thermo_df['zpe/(J/mol)'])
            else:
                delta_H_barrier_f_0K_scan = ts_thermo_df['ee/(J/mol)'] + ts_thermo_df['zpe/(J/mol)'] - \
                                         (r1_thermo_df['ee/(J/mol)'] + r1_thermo_df['zpe/(J/mol)'])
            delta_H_barrier_r_0K_scan = ts_thermo_df['ee/(J/mol)'] + ts_thermo_df['zpe/(J/mol)'] - \
                                   (p_thermo_df['ee/(J/mol)'] + p_thermo_df['zpe/(J/mol)'])
            k_scan = np.array(
                [
                    k_TST(delta_G=delta_g, delta_n=delta_n, T=t, sigma=sigma, tunnelling_effect=tunnelling_effect,
                          imaginary_freq=imaginary_freq, delta_H_barrier_f_0K=f, delta_H_barrier_r_0K=r)
                    for delta_g, t, f, r in zip(delta_G, T, delta_H_barrier_f_0K_scan, delta_H_barrier_r_0K_scan)
                ]
            )

        else:
            raise NotImplemented(f'{tunnelling_effect} is not a right value')

    # 不考虑隧道效应时
    else:
        k_scan = np.array(
            [
                k_TST(delta_G=delta_g, delta_n=delta_n, T=t, sigma=sigma, tunnelling_effect=tunnelling_effect,
                      imaginary_freq=imaginary_freq)
                for delta_g, t in zip(delta_G, T)
            ]
        )

    return k_scan



def k_VTST(delta_G_list, delta_n, T=298.15, P0=100000, sigma=1, also_get_k_tst=False):
    if isinstance(delta_G_list, list):
        delta_G_list = np.array(delta_G_list)
    k_tst_list = k_TST(delta_G=delta_G_list, delta_n=delta_n, T=T, P0=P0, sigma=sigma)
    k_vtst = np.min(k_tst_list)
    if also_get_k_tst:
        return k_vtst, k_tst_list
    return k_vtst


if __name__ == '__main__':

    k = k_TST(delta_G=23000, delta_n=0)
    print(k)

    k_vtst = k_VTST(delta_G_list=[23000, 24000], delta_n=0)
    print(k_vtst)