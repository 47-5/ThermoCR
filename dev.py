from QMthermo.qm_thermo import qm_thermo, qm_thermo_scan, qm_thermo_conformation_weighting
from QMkinetics.qm_kinetics import k_TST, k_VTST, k_TST_scan
from QMkinetics.tunnelling_effect import eckart_correction, wigner_correction


if __name__ == '__main__':

    s1 = qm_thermo(atom_coord_path='01.out', T=500, P=100000, verbose=True)
    s2 = qm_thermo(atom_coord_path='02.out', T=500, P=100000, verbose=True)
    ts_1_to_2 = qm_thermo(atom_coord_path='01_02.out', T=500, P=100000, verbose=True)

    T = list(range(50, 1000, 50))
    s1_df = qm_thermo_scan(atom_coord_path='01.out', T=T, out_path='s1.xlsx')
    s2_df = qm_thermo_scan(atom_coord_path='02.out', T=T, out_path='s2.xlsx')
    ts_1_to_2_df = qm_thermo_scan(atom_coord_path='01_02.out', T=T, out_path='ts_1_to_2.xlsx')

    # delta_G = ts_1_to_2['G/(J/mol)'] - 2 * s1['G/(J/mol)']
    # print(delta_G)
    # k_tst = k_TST(delta_G=delta_G, delta_n=1, T=500, sigma=1)
    # print(k_tst)
    delta_G_scan = ts_1_to_2_df['G/(J/mol)'] - 2 * s1_df['G/(J/mol)']
    for delta_G, t in zip(delta_G_scan, T):
        k_tst = k_TST(delta_G=delta_G, delta_n=1, T=t, sigma=1)
        print(k_tst)

    k_scan = k_TST_scan(thermo_ts_path='ts_1_to_2.xlsx', thermo_r1_path='s1.xlsx', thermo_r2_path='s1.xlsx', sigma=1)
    print(k_scan)