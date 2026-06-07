"""Backward-compatible kinetics namespace."""

from ThermoCR.QMkinetics.equilibrium_constants import k_equilibrium_constants
from ThermoCR.QMkinetics.fit_kinetics import (
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
from ThermoCR.QMkinetics.qm_kinetics import k_TST, k_TST_scan, k_VTST, k_VTST_scan
from ThermoCR.QMkinetics.tunnelling_effect import (
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
