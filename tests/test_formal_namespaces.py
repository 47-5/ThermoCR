import math
import unittest

from ThermoCR.QMkinetics import (
    k_TST as legacy_package_k_TST,
    k_equilibrium_constants as legacy_package_k_equilibrium_constants,
)
from ThermoCR.QMkinetics.equilibrium_constants import (
    k_equilibrium_constants as legacy_k_equilibrium_constants,
)
from ThermoCR.QMkinetics.qm_kinetics import k_TST as legacy_k_TST
from ThermoCR.QMkinetics.tunnelling_effect import (
    eckart_correction as legacy_eckart_correction,
    skodje_truhlar as legacy_skodje_truhlar,
    wigner_correction as legacy_wigner_correction,
)
from ThermoCR.QMthermo import (
    S_trans as legacy_package_S_trans,
    ZPE as legacy_package_ZPE,
    q_trans as legacy_package_q_trans,
)
from ThermoCR.QMthermo.calc_q import (
    q_rot_single_atom as legacy_q_rot_single_atom,
    q_trans as legacy_q_trans,
    q_vib_V0 as legacy_q_vib_V0,
)
from ThermoCR.QMthermo.calc_thermo_corr import (
    S_trans as legacy_S_trans,
    S_vib_RRHO_vec as legacy_S_vib_RRHO_vec,
    ZPE as legacy_ZPE,
)
from ThermoCR.kinetics import (
    eckart_correction,
    k_TST,
    k_equilibrium_constants,
    skodje_truhlar,
    wigner_correction,
)
from ThermoCR.kinetics.equilibrium import (
    k_equilibrium_constants as namespaced_k_equilibrium_constants,
)
from ThermoCR.kinetics.rate_constants import k_TST as namespaced_k_TST
from ThermoCR.kinetics.tunneling import (
    eckart_correction as namespaced_eckart_correction,
    skodje_truhlar as namespaced_skodje_truhlar,
    wigner_correction as namespaced_wigner_correction,
)
from ThermoCR.thermo import (
    S_trans,
    S_vib_RRHO_vec,
    ZPE,
    q_rot_single_atom,
    q_trans,
    q_vib_V0,
    qm_thermo,
)
from ThermoCR.thermo.calculators import qm_thermo as namespaced_qm_thermo
from ThermoCR.thermo.corrections import (
    S_trans as namespaced_S_trans,
    S_vib_RRHO_vec as namespaced_S_vib_RRHO_vec,
    ZPE as namespaced_ZPE,
)
from ThermoCR.thermo.partition import (
    q_rot_single_atom as namespaced_q_rot_single_atom,
    q_trans as namespaced_q_trans,
    q_vib_V0 as namespaced_q_vib_V0,
)


class FormalNamespaceApiTests(unittest.TestCase):
    def test_thermo_namespace_reexports_legacy_functions(self):
        self.assertIs(q_trans, legacy_q_trans)
        self.assertIs(q_trans, legacy_package_q_trans)
        self.assertIs(namespaced_q_trans, legacy_q_trans)
        self.assertIs(q_rot_single_atom, legacy_q_rot_single_atom)
        self.assertIs(q_rot_single_atom, namespaced_q_rot_single_atom)
        self.assertIs(q_vib_V0, legacy_q_vib_V0)
        self.assertIs(q_vib_V0, namespaced_q_vib_V0)
        self.assertIs(S_trans, legacy_S_trans)
        self.assertIs(S_trans, legacy_package_S_trans)
        self.assertIs(S_trans, namespaced_S_trans)
        self.assertIs(ZPE, legacy_ZPE)
        self.assertIs(ZPE, legacy_package_ZPE)
        self.assertIs(ZPE, namespaced_ZPE)
        self.assertIs(S_vib_RRHO_vec, legacy_S_vib_RRHO_vec)
        self.assertIs(S_vib_RRHO_vec, namespaced_S_vib_RRHO_vec)
        self.assertIs(qm_thermo, namespaced_qm_thermo)
        self.assertGreater(q_trans(M=28.0, T=298.15, P=101325.0), 0.0)
        self.assertEqual(q_rot_single_atom(), 1.0)
        self.assertGreater(ZPE([1000.0, 1500.0]), 0.0)

    def test_kinetics_namespace_reexports_legacy_functions(self):
        self.assertIs(k_TST, legacy_k_TST)
        self.assertIs(k_TST, legacy_package_k_TST)
        self.assertIs(namespaced_k_TST, legacy_k_TST)
        self.assertIs(k_equilibrium_constants, legacy_k_equilibrium_constants)
        self.assertIs(k_equilibrium_constants, legacy_package_k_equilibrium_constants)
        self.assertIs(namespaced_k_equilibrium_constants, legacy_k_equilibrium_constants)
        self.assertIs(wigner_correction, legacy_wigner_correction)
        self.assertIs(wigner_correction, namespaced_wigner_correction)
        self.assertIs(eckart_correction, legacy_eckart_correction)
        self.assertIs(eckart_correction, namespaced_eckart_correction)
        self.assertIs(skodje_truhlar, legacy_skodje_truhlar)
        self.assertIs(skodje_truhlar, namespaced_skodje_truhlar)
        rate_constant = k_TST(delta_G=0.0, delta_n=0, T=298.15)
        self.assertTrue(math.isfinite(rate_constant))
        self.assertGreater(rate_constant, 0.0)
        self.assertGreater(wigner_correction(imaginary_freq=500.0, T=298.15), 1.0)
        self.assertAlmostEqual(k_equilibrium_constants(delta_G=0.0, T=298.15), 1.0)


if __name__ == "__main__":
    unittest.main()
