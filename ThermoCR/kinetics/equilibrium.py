"""Equilibrium constant helpers."""

import numpy as np

from ThermoCR.constants import R


def k_equilibrium_constants(delta_G, T):
    """Calculate the equilibrium constant from Gibbs free energy and temperature."""
    return np.exp(-delta_G / (R * T))


__all__ = ["k_equilibrium_constants"]
