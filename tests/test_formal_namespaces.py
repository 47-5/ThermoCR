import math
import unittest

from ThermoCR.QMkinetics.qm_kinetics import k_TST as legacy_k_TST
from ThermoCR.QMthermo.calc_q import q_trans as legacy_q_trans
from ThermoCR.kinetics import k_TST, wigner_correction
from ThermoCR.kinetics.rate_constants import k_TST as namespaced_k_TST
from ThermoCR.thermo import q_trans, qm_thermo
from ThermoCR.thermo.calculators import qm_thermo as namespaced_qm_thermo
from ThermoCR.thermo.partition import q_trans as namespaced_q_trans


class FormalNamespaceApiTests(unittest.TestCase):
    def test_thermo_namespace_reexports_legacy_functions(self):
        self.assertIs(q_trans, legacy_q_trans)
        self.assertIs(namespaced_q_trans, legacy_q_trans)
        self.assertIs(qm_thermo, namespaced_qm_thermo)
        self.assertGreater(q_trans(M=28.0, T=298.15, P=101325.0), 0.0)

    def test_kinetics_namespace_reexports_legacy_functions(self):
        self.assertIs(k_TST, legacy_k_TST)
        self.assertIs(namespaced_k_TST, legacy_k_TST)
        rate_constant = k_TST(delta_G=0.0, delta_n=0, T=298.15)
        self.assertTrue(math.isfinite(rate_constant))
        self.assertGreater(rate_constant, 0.0)
        self.assertGreater(wigner_correction(imaginary_freq=500.0, T=298.15), 1.0)


if __name__ == "__main__":
    unittest.main()