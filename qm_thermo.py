from calc_q import q_trans, q_rot_single_atom, q_rot_linear, q_rot_nonlinear, q_vib_V0, q_vib_bot, q_ele, q
from calc_thermo_corr import *
from read_qm_out import read_qm_out, read_atom_coord, read_vib, read_ee
from pointgroup.element_data import atom_data
from utils import get_I, check_linear, get_rotational_symmetry_number, get_point_group
from constant import convert_I, au2j_mol
from typing import List
import pandas as pd


def qm_thermo(atom_coord_path, vib_path=None, ee_path=None, T=298.15, P=101325,
              sclZPE=1.0, sclU=1.0, sclCv=1.0, sclS=1.0,
              U_Minenkov=False, S_Grimme=True, verbose=True,
              read_ee_index=-1,
              E_list=None, g_list=None):
    # load data
    if vib_path is None:
        vib_path = atom_coord_path
    if ee_path is None:
        ee_path = atom_coord_path
    atom_numbers, atom_masses, coords = read_atom_coord(atom_coord_path)
    atom_masses = np.array([atom_data[i][3] for i in atom_numbers])
    M = np.sum(atom_masses)
    vibfreqs = read_vib(vib_path)
    ee = read_ee(ee_path, ee_index=read_ee_index)
    if E_list is None:
        E_list = [ee]
    if g_list is None:
        g_list = [1]

    # calculate
    # trans
    q_t, U_t, H_t, Cv_t, Cp_t, S_t = contribution_trans(M=M, T=T, P=P)
    if verbose:
        print('======= Translation =======')
        print(f'q_t: {q_t}')
        print(f'U_t: {U_t}')
        print(f'H_t: {H_t}')
        print(f'Cv_t: {Cv_t}')
        print(f'Cp_t: {Cp_t}')
        print(f'S_t: {S_t}')
    # rot
    q_r, U_r, H_r, Cv_r, Cp_r, S_r = contribution_rot(atom_numbers=atom_numbers, coords=coords, T=T, convert_unit=True)
    if verbose:
        print('========= Rotation ========')
        print(f'q_r: {q_r}')
        print(f'U_r: {U_r}')
        print(f'H_r: {H_r}')
        print(f'Cv_r: {Cv_r}')
        print(f'Cp_r: {Cp_r}')
        print(f'S_r: {S_r}')
    # vib
    q_v_0, q_v_bot, U_v_0_T, H_v_0_T, U_v, H_v, Cv_v, Cp_v, S_v, zpe = contribution_vib(vibfreqs=vibfreqs, T=T, convert_unit=True,
                                                                                        sclZPE=sclZPE, sclU=sclU, sclCv=sclCv, sclS=sclS,
                                                                                        U_Minenkov=U_Minenkov, S_Grimme=S_Grimme)
    if verbose:
        print('======== Vibration ========')
        print(f'q_v_bot: {q_v_bot}')
        print(f'q_v_0: {q_v_0}')
        print(f'zpe: {zpe}')
        print(f'U_v_0_T: {U_v_0_T}')
        print(f'H_v_0_T: {H_v_0_T}')
        print(f'U_v: {U_v}')
        print(f'H_v: {H_v}')
        print(f'Cv_v: {Cv_v}')
        print(f'Cp_v: {Cp_v}')
        print(f'S_v: {S_v}')
    # ele
    q_e, U_e, H_e, Cv_e, Cp_e, S_e = contribution_ele(E_list=E_list, g_list=g_list, T=T, convert_unit=True)
    if verbose:
        print('======== Electron excitation ========')
        print(f'q_e: {q_e}')
        print(f'U_e: {U_e}')
        print(f'H_e: {H_e}')
        print(f'Cv_e: {Cv_e}')
        print(f'Cp_e: {Cp_e}')
        print(f'S_e: {S_e}')

    # total
    q_tot_v_0 = q(q_t=q_t, q_r=q_r, q_v=q_v_0, q_e=q_e)
    q_tot_bot = q(q_t=q_t, q_r=q_r, q_v=q_v_bot, q_e=q_e)
    Cv_tot = Cv_t + Cv_r + Cv_v + Cv_e
    Cp_tot = Cp_t + Cp_r + Cp_v + Cp_e
    S_tot = S_t + S_r + S_v + S_e
    U_corr = U_t + U_r + U_v + U_e
    H_corr = H_t + H_r + H_v + H_e
    G_corr = H_corr - T * S_tot

    zpe_plus_ee = ee * au2j_mol  + zpe
    U = ee * au2j_mol + U_corr
    H = ee * au2j_mol  + H_corr
    G = ee * au2j_mol  + G_corr
    if verbose:
        print('========== Total ==========')
        print(f'q(V=0): {q_tot_v_0}')
        print(f'q(bot): {q_tot_bot}')
        print(f'Cv_tot: {Cv_tot} J/mol/K')
        print(f'Cp_tot: {Cp_tot} J/mol/K')
        print(f'S_tot: {S_tot} J/mol/K')
        print(f'ZPE: {zpe} J/mol')
        print(f'U_corr: {U_corr} J/mol')
        print(f'H_corr: {H_corr} J/mol')
        print(f'G_corr: {G_corr} J/mol')
        print(f'EE: {ee * au2j_mol} J/mol {ee} a.u.')
        print(f'Sum of electronic energy and ZPE, namely U/H/G at 0 K: {zpe_plus_ee} J/mol {zpe_plus_ee/au2j_mol} a.u.')
        print(f'Sum of electronic energy and thermal correction to U: {U}  J/mol {U/au2j_mol} a.u.')
        print(f'Sum of electronic energy and thermal correction to H: {H}  J/mol {H/au2j_mol} a.u.')
        print(f'Sum of electronic energy and thermal correction to G: {G}  J/mol {G/au2j_mol} a.u.')
    return {
        'T/K': T, 'P/Pa': P,
        'q_tot_v_0': q_tot_v_0, 'q_tot_bot': q_tot_bot,
        'Cv/(J/mol/K)': Cv_tot, 'Cp/(J/mol/K)': Cp_tot, 'S/(J/mol/K)': S_tot,
        'zpe/J': zpe, 'U_corr/J': U_corr, 'H_corr/J': H_corr, 'G_corr/J': G_corr,
        'ee/J': ee, 'U/J': U, 'H/J': H, 'G/J': G
    }


def qm_thermo_scan(
        atom_coord_path, vib_path=None, ee_path=None,
        T:List[int|float]=[298.15], P:List[int|float]=[101325],
        sclZPE=1.0, sclU=1.0, sclCv=1.0, sclS=1.0,
        U_Minenkov=False, S_Grimme=True,
        read_ee_index=-1,
        E_list=None, g_list=None,
        out_path='QMthermoScan.xlsx'
        ):
    results = []
    for t in T:
        print(f'T={t} K')
        for p in P:
            print(f'p={p} Pa')
            result = qm_thermo(atom_coord_path=atom_coord_path, vib_path=vib_path, ee_path=ee_path,
                               T=t, P=p,
                               sclZPE=sclZPE, sclU=sclU, sclCv=sclCv, sclS=sclS,
                               U_Minenkov=U_Minenkov, S_Grimme=S_Grimme,
                               read_ee_index=read_ee_index, E_list=E_list, g_list=g_list,
                               verbose=False
                               )
            results.append(result)
    df = pd.DataFrame(results)
    if out_path is not None:
        df.to_excel(out_path, index=False)
    return df


def contribution_trans(M, T, P):
    q_t = q_trans(M=M, T=T, P=P)
    U_t = U_trans(T=T)
    H_t = H_trans(T=T)
    Cv_t = Cv_trans()
    Cp_t = Cp_trans()
    S_t = S_trans(q_t=q_t, T=T)
    return q_t, U_t, H_t, Cv_t, Cp_t, S_t


def contribution_rot(atom_numbers, coords, T, convert_unit=True):
    if len(atom_numbers) == 1:
        q_r = q_rot_single_atom()
        U_r = 0
        H_r = 0
        Cv_r = 0
        Cp_r = 0
        S_r = 0
    else:
        I = get_I(coords=coords, numbers=atom_numbers)
        linear_flag = check_linear(I)
        pg = get_point_group(coords=coords, numbers=atom_numbers)
        sigma = get_rotational_symmetry_number(pg)
        if convert_unit:
            I *= convert_I
        if linear_flag:
            I = I[-1]
            q_r = q_rot_linear(sigma=sigma, I=I, T=T)
            U_r = U_rot_linear(T=T)
            H_r = H_rot_linear(T=T)
            Cv_r = Cv_rot_linear()
            Cp_r = Cp_rot_linear()
            S_r = S_rot_linear(q_r=q_r, T=T)
        else:
            IA, IB, IC = I[0], I[1], I[2]
            q_r = q_rot_nonlinear(sigma=sigma, IA=IA, IB=IB, IC=IC, T=T)
            U_r = U_rot_nonlinear(T=T)
            H_r = H_rot_nonlinear(T=T)
            Cv_r = Cv_rot_nonlinear()
            Cp_r = Cp_rot_nonlinear()
            S_r = S_rot_nonlinear(q_r=q_r, T=T)
    return q_r, U_r, H_r, Cv_r, Cp_r, S_r


def contribution_vib(vibfreqs, T, convert_unit=True, sclZPE=1.0, sclU=1.0, sclCv=1.0, sclS=1.0, U_Minenkov=False, S_Grimme=True):
    q_v_0 = q_vib_V0(vibfreqs=vibfreqs, T=T, convert_unit=convert_unit)
    q_v_bot = q_vib_bot(vibfreqs=vibfreqs, T=T, convert_unit=convert_unit)

    zpe = ZPE(vibfreqs=vibfreqs, convert_unit=convert_unit, scale_factor=sclZPE)
    U_v_0_T = U_vib_0_T(vibfreqs=vibfreqs, T=T, convert_unit=convert_unit, scale_factor=sclU)
    H_v_0_T = H_vib_0_T(vibfreqs=vibfreqs, T=T, convert_unit=convert_unit, scale_factor=sclU)
    U_v = U_vib_T(vibfreqs=vibfreqs, T=T, QRRHO=U_Minenkov, convert_unit=convert_unit, scale_factor_U_0_T=sclU, scale_factor_zpe=sclZPE)
    H_v = H_vib_T(vibfreqs=vibfreqs, T=T, QRRHO=U_Minenkov, convert_unit=convert_unit, scale_factor_U_0_T=sclU, scale_factor_zpe=sclZPE)
    Cv_v = Cv_vib(vibfreqs=vibfreqs, T=T, convert_unit=convert_unit, scale_factor=sclCv)
    Cp_v = Cp_vib(vibfreqs=vibfreqs, T=T, convert_unit=convert_unit, scale_factor=sclCv)
    S_v = S_vib(vibfreqs=vibfreqs, T=T, convert_unit=convert_unit, QRRHO=S_Grimme, scale_factor=sclS)
    return q_v_0, q_v_bot, U_v_0_T, H_v_0_T, U_v, H_v, Cv_v, Cp_v, S_v, zpe


def contribution_ele(E_list, g_list, T, convert_unit=True):
    q_e = q_ele(E_list=E_list, g_list=g_list, T=T, convert_unit=convert_unit)
    U_e = U_ele(E_list=E_list, g_list=g_list, T=T, convert_unit=convert_unit)
    H_e = H_ele(E_list=E_list, g_list=g_list, T=T, convert_unit=convert_unit)
    Cv_e = Cv_ele(E_list=E_list, g_list=g_list, T=T, convert_unit=convert_unit)
    Cp_e = Cp_ele(E_list=E_list, g_list=g_list, T=T, convert_unit=convert_unit)
    S_e = S_ele(E_list=E_list, g_list=g_list, T=T, convert_unit=convert_unit)
    return q_e, U_e, H_e, Cv_e, Cp_e, S_e


if __name__ == '__main__':

    qm_thermo(atom_coord_path='01_02.out', verbose=True, read_ee_index=-3)

    qm_thermo_scan(atom_coord_path='01_02.out', vib_path=None, ee_path='01_02_sp.out',
                   T=[150, 200, 250, 298.15, 300], P=[101325, 150000, 300000],
                   sclZPE=1.0, sclU=1.0, sclCv=1.0, sclS=1.0,
                   U_Minenkov=False, S_Grimme=True,
                   )