import numpy as np
from ThermoCR.tools.constant import R


def k_equilibrium_constants(delta_G, T):
    k_eq = np.exp(-delta_G / (R * T))
    return k_eq