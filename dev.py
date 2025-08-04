from QMthermo.qm_thermo import qm_thermo, qm_thermo_scan, qm_thermo_conformation_weighting


if __name__ == '__main__':

    qm_thermo(atom_coord_path='01.out', vib_path=None, verbose=True, ee=-194.283781614283, g_list=None, ignore_trans_and_rot=True)

    qm_thermo_scan(atom_coord_path='01.out', vib_path=None, ee_path=None,
                   T=list(range(100, 3050, 50)), P=[101325],
                   sclZPE=1.0, sclU=1.0, sclCv=1.0, sclS=1.0,
                   U_Minenkov=False, S_Grimme=True,
                   ee=-194.283781614283,
                   g_list=None
                   )
    s2 = qm_thermo(atom_coord_path='02.out', verbose=True)
    s3 = qm_thermo(atom_coord_path='03.out', verbose=True)

    U_list = [s2['U/(J/mol)'], s3['U/(J/mol)']]
    H_list = [s2['H/(J/mol)'], s3['H/(J/mol)']]
    G_list = [s2['G/(J/mol)'], s3['G/(J/mol)']]
    S_list = [s2['S/(J/mol/K)'], s3['S/(J/mol/K)']]
    Cp_list = [s2['Cp/(J/mol/K)'], s3['Cp/(J/mol/K)']]
    Cv_list = [s2['Cv/(J/mol/K)'], s3['Cv/(J/mol/K)']]
    conformation_weighting_result = qm_thermo_conformation_weighting(U_list, H_list, G_list, S_list, Cv_list, Cp_list, T=298.15)
    print(conformation_weighting_result)