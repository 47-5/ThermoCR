import numpy as np

from QMthermo.qm_thermo import qm_thermo, qm_thermo_scan, qm_thermo_conformation_weighting
from QMkinetics.qm_kinetics import k_TST, k_VTST, k_TST_scan
from QMkinetics.tunnelling_effect import eckart_correction, wigner_correction
from tools.read_qm_out import read_imaginary_vib


if __name__ == '__main__':

    s1 = qm_thermo(atom_coord_path='01.out', T=500, P=100000, verbose=True)
    s2 = qm_thermo(atom_coord_path='02.out', T=500, P=100000, verbose=True)
    ts_1_to_2 = qm_thermo(atom_coord_path='01_02.out', T=500, P=100000, verbose=True)

    T = list(range(50, 1000, 50))
    s1_df = qm_thermo_scan(atom_coord_path='01.out', T=T, out_path='s1.xlsx')
    s2_df = qm_thermo_scan(atom_coord_path='02.out', T=T, out_path='s2.xlsx')
    ts_1_to_2_df = qm_thermo_scan(atom_coord_path='01_02.out', T=T, out_path='ts_1_to_2.xlsx')

    k_tst_scan = k_TST_scan(thermo_ts_path='ts_1_to_2.xlsx', thermo_r1_path='s1.xlsx', thermo_r2_path='s1.xlsx', sigma=1)
    print(k_tst_scan)

    imaginary_freq = read_imaginary_vib(filepath='01_02.out')
    print(imaginary_freq)
    chi_wigner_scan = np.array([wigner_correction(imaginary_freq=-imaginary_freq, T=i) for i in T])
    print(chi_wigner_scan)

    k_tst_wigner_scan = k_TST_scan(thermo_ts_path='ts_1_to_2.xlsx',
                                   thermo_r1_path='s1.xlsx', thermo_r2_path='s1.xlsx',
                                   sigma=1,
                                   tunnelling_effect='wigner', imaginary_freq=imaginary_freq)
    print(k_tst_wigner_scan)

    k_tst_eckart_scan = k_TST_scan(thermo_ts_path='ts_1_to_2.xlsx',
                                   thermo_r1_path='s1.xlsx', thermo_r2_path='s1.xlsx',
                                   thermo_p_path='s2.xlsx',
                                   sigma=1,
                                   tunnelling_effect='eckart', imaginary_freq=imaginary_freq)
    print(k_tst_eckart_scan)