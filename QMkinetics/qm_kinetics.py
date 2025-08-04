import numpy as np
from tools.constant import k_b, h, R


def k_TST(delta_G, delta_n, T=298.15, P0=100000, sigma=1):
    """
    calculate k base on TST
    :param delta_G: G_TS - G_IS / (J/mol)
    :param delta_n: 双分子反应取1，单分子反应取0
    :param T: K
    :param P0: 标准压力 100000 Pa = 1 bar
    :param sigma: 反应路径简并度 = sigma_rot_TS / sigma_rot_IS 如果计算G时已经考虑转动对称数，这里应该设置为1
    :return: k:双分子反应单位: s^-1 * (molecule / m^3)
             k:单分子反应单位: s^-1
    """

    k = sigma * k_b * T / h * (k_b * T / P0) ** delta_n * np.exp(-delta_G / (R * T))
    return k


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