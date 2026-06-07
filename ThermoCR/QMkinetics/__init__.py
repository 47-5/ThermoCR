"""Backward-compatible kinetics namespace."""

from ThermoCR.kinetics.equilibrium import k_equilibrium_constants
from ThermoCR.kinetics.fitting import (
    A_nplus1,
    Arrhenius,
    Arrhenius2Piecewise,
    arrhenius,
    arrhenius_2piecewise,
    cal_metric,
    convert_k_unit_from_ThermoCR_to_Cantera,
    export_data,
    fit,
    fit_kinetics_model,
    plot_fit,
)
from ThermoCR.kinetics.rate_constants import k_TST, k_TST_scan, k_VTST, k_VTST_scan
from ThermoCR.kinetics.tunneling import (
    eckart_correction,
    skodje_truhlar,
    wigner_correction,
)

__all__ = [
    "A_nplus1",
    "Arrhenius",
    "Arrhenius2Piecewise",
    "arrhenius",
    "arrhenius_2piecewise",
    "cal_metric",
    "convert_k_unit_from_ThermoCR_to_Cantera",
    "eckart_correction",
    "export_data",
    "fit",
    "fit_kinetics_model",
    "k_TST",
    "k_TST_scan",
    "k_VTST",
    "k_VTST_scan",
    "k_equilibrium_constants",
    "plot_fit",
    "skodje_truhlar",
    "wigner_correction",
]
