"""Modern kinetics API for ThermoCR."""

from ThermoCR.kinetics.equilibrium import k_equilibrium_constants
from ThermoCR.kinetics.fitting import (
    A_nplus1,
    Arrhenius,
    Arrhenius2Piecewise,
    arrhenius,
    arrhenius_2piecewise,
    convert_k_unit_from_ThermoCR_to_Cantera,
    fit_kinetics_model,
)
from ThermoCR.kinetics.rate_constants import (
    calculate_tst_rate_frame,
    k_TST,
    k_TST_scan,
    k_VTST,
    k_VTST_scan,
)
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
    "convert_k_unit_from_ThermoCR_to_Cantera",
    "calculate_tst_rate_frame",
    "eckart_correction",
    "fit_kinetics_model",
    "k_TST",
    "k_TST_scan",
    "k_VTST",
    "k_VTST_scan",
    "k_equilibrium_constants",
    "skodje_truhlar",
    "wigner_correction",
]