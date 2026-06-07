import math
import unittest

from ThermoCR.QMkinetics import (
    Arrhenius as legacy_package_Arrhenius,
    arrhenius as legacy_package_arrhenius,
    fit_kinetics_model as legacy_package_fit_kinetics_model,
    k_TST as legacy_package_k_TST,
    k_equilibrium_constants as legacy_package_k_equilibrium_constants,
)
from ThermoCR.QMkinetics.equilibrium_constants import (
    k_equilibrium_constants as legacy_k_equilibrium_constants,
)
from ThermoCR.QMkinetics.fit_kinetics import (
    Arrhenius as legacy_Arrhenius,
    arrhenius as legacy_arrhenius,
    convert_k_unit_from_ThermoCR_to_Cantera as legacy_convert_k_unit,
    fit_kinetics_model as legacy_fit_kinetics_model,
)
from ThermoCR.QMkinetics.qm_kinetics import (
    k_TST as legacy_k_TST,
    k_TST_scan as legacy_k_TST_scan,
    k_VTST as legacy_k_VTST,
    k_VTST_scan as legacy_k_VTST_scan,
)
from ThermoCR.QMkinetics.tunnelling_effect import (
    eckart_correction as legacy_eckart_correction,
    skodje_truhlar as legacy_skodje_truhlar,
    wigner_correction as legacy_wigner_correction,
)
from ThermoCR.QMthermo import (
    NASA7 as legacy_package_NASA7,
    S_trans as legacy_package_S_trans,
    ZPE as legacy_package_ZPE,
    fit_thermo_model as legacy_package_fit_thermo_model,
    nasa7 as legacy_package_nasa7,
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
from ThermoCR.QMthermo.fit_thermo import (
    NASA7 as legacy_NASA7,
    fit_thermo_model as legacy_fit_thermo_model,
    nasa7 as legacy_nasa7,
)
from ThermoCR.kinetics import (
    Arrhenius,
    arrhenius,
    convert_k_unit_from_ThermoCR_to_Cantera,
    eckart_correction,
    fit_kinetics_model,
    k_TST,
    k_TST_scan,
    k_VTST,
    k_VTST_scan,
    k_equilibrium_constants,
    skodje_truhlar,
    wigner_correction,
)
from ThermoCR.kinetics.equilibrium import (
    k_equilibrium_constants as namespaced_k_equilibrium_constants,
)
from ThermoCR.kinetics.fitting import (
    Arrhenius as namespaced_Arrhenius,
    arrhenius as namespaced_arrhenius,
    convert_k_unit_from_ThermoCR_to_Cantera as namespaced_convert_k_unit,
    fit_kinetics_model as namespaced_fit_kinetics_model,
)
from ThermoCR.kinetics.rate_constants import (
    k_TST as namespaced_k_TST,
    k_TST_scan as namespaced_k_TST_scan,
    k_VTST as namespaced_k_VTST,
    k_VTST_scan as namespaced_k_VTST_scan,
)
from ThermoCR.kinetics.tunneling import (
    eckart_correction as namespaced_eckart_correction,
    skodje_truhlar as namespaced_skodje_truhlar,
    wigner_correction as namespaced_wigner_correction,
)
from ThermoCR.thermo import (
    NASA7,
    S_trans,
    S_vib_RRHO_vec,
    ZPE,
    fit_thermo_model,
    nasa7,
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
from ThermoCR.thermo.fitting import (
    NASA7 as namespaced_NASA7,
    fit_thermo_model as namespaced_fit_thermo_model,
    nasa7 as namespaced_nasa7,
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
        self.assertIs(NASA7, legacy_NASA7)
        self.assertIs(NASA7, legacy_package_NASA7)
        self.assertIs(NASA7, namespaced_NASA7)
        self.assertIs(nasa7, legacy_nasa7)
        self.assertIs(nasa7, legacy_package_nasa7)
        self.assertIs(nasa7, namespaced_nasa7)
        self.assertIs(fit_thermo_model, legacy_fit_thermo_model)
        self.assertIs(fit_thermo_model, legacy_package_fit_thermo_model)
        self.assertIs(fit_thermo_model, namespaced_fit_thermo_model)
        self.assertIs(qm_thermo, namespaced_qm_thermo)
        self.assertGreater(q_trans(M=28.0, T=298.15, P=101325.0), 0.0)
        self.assertEqual(q_rot_single_atom(), 1.0)
        self.assertGreater(ZPE([1000.0, 1500.0]), 0.0)
        cp, h, s = nasa7(300.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(cp, 8.314)
        self.assertAlmostEqual(h, 8.314 * 300.0)
        self.assertAlmostEqual(s, 8.314 * math.log(300.0))

    def test_kinetics_namespace_reexports_legacy_functions(self):
        self.assertIs(k_TST, legacy_k_TST)
        self.assertIs(k_TST, legacy_package_k_TST)
        self.assertIs(namespaced_k_TST, legacy_k_TST)
        self.assertIs(k_TST_scan, legacy_k_TST_scan)
        self.assertIs(k_TST_scan, namespaced_k_TST_scan)
        self.assertIs(k_VTST, legacy_k_VTST)
        self.assertIs(k_VTST, namespaced_k_VTST)
        self.assertIs(k_VTST_scan, legacy_k_VTST_scan)
        self.assertIs(k_VTST_scan, namespaced_k_VTST_scan)
        self.assertIs(k_equilibrium_constants, legacy_k_equilibrium_constants)
        self.assertIs(k_equilibrium_constants, legacy_package_k_equilibrium_constants)
        self.assertIs(namespaced_k_equilibrium_constants, legacy_k_equilibrium_constants)
        self.assertIs(wigner_correction, legacy_wigner_correction)
        self.assertIs(wigner_correction, namespaced_wigner_correction)
        self.assertIs(eckart_correction, legacy_eckart_correction)
        self.assertIs(eckart_correction, namespaced_eckart_correction)
        self.assertIs(skodje_truhlar, legacy_skodje_truhlar)
        self.assertIs(skodje_truhlar, namespaced_skodje_truhlar)
        self.assertIs(Arrhenius, legacy_Arrhenius)
        self.assertIs(Arrhenius, legacy_package_Arrhenius)
        self.assertIs(Arrhenius, namespaced_Arrhenius)
        self.assertIs(arrhenius, legacy_arrhenius)
        self.assertIs(arrhenius, legacy_package_arrhenius)
        self.assertIs(arrhenius, namespaced_arrhenius)
        self.assertIs(fit_kinetics_model, legacy_fit_kinetics_model)
        self.assertIs(fit_kinetics_model, legacy_package_fit_kinetics_model)
        self.assertIs(fit_kinetics_model, namespaced_fit_kinetics_model)
        self.assertIs(convert_k_unit_from_ThermoCR_to_Cantera, legacy_convert_k_unit)
        self.assertIs(convert_k_unit_from_ThermoCR_to_Cantera, namespaced_convert_k_unit)
        rate_constant = k_TST(delta_G=0.0, delta_n=0, T=298.15)
        self.assertTrue(math.isfinite(rate_constant))
        self.assertGreater(rate_constant, 0.0)
        self.assertGreater(wigner_correction(imaginary_freq=500.0, T=298.15), 1.0)
        self.assertAlmostEqual(k_equilibrium_constants(delta_G=0.0, T=298.15), 1.0)
        expected_vtst = rate_constant * math.exp(-1000.0 / (8.3144648 * 298.15))
        self.assertAlmostEqual(k_VTST(delta_G_list=[0.0, 1000.0], delta_n=0, T=298.15), expected_vtst)
        self.assertAlmostEqual(arrhenius(T=300.0, A=1.0, Ea=0.0), 300.0)


if __name__ == "__main__":
    unittest.main()
