import numpy as np
from ThermoCR.tools.constant import R


def k_equilibrium_constants(delta_G, T):
    """
    Calculates the equilibrium constant (k_eq) for a chemical reaction based on the
    Gibbs free energy change (delta_G) and temperature (T). The formula used is
    k_eq = exp(-delta_G / (R * T)), where R is the ideal gas constant.

    Parameters:
    - delta_G (float): Gibbs free energy change of the reaction in joules per mole.
    - T (float): Temperature at which the reaction occurs in Kelvin.

    Returns:
        float, The equilibrium constant (k_eq) of the reaction.
    """
    k_eq = np.exp(-delta_G / (R * T))
    return k_eq