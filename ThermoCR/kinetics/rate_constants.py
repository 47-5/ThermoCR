"""Reaction rate constant helpers."""

import numpy as np
import pandas as pd
from ThermoCR.constants import k_b, h, R
from ThermoCR.kinetics.tunneling import wigner_correction, eckart_correction, skodje_truhlar
from typing import List
from os.path import basename


__all__ = ['calculate_tst_rate_frame', 'k_TST', 'k_VTST', 'k_TST_scan', 'k_VTST_scan']


_STRUCTURED_THERMO_COLUMNS = {
    "T": "temperature",
    "G": "gibbs_free_energy",
    "EE": "electronic_energy",
    "ZPE": "zpe",
}
_LEGACY_THERMO_COLUMNS = {
    "T": "T/K",
    "G": "G/(J/mol)",
    "EE": "ee/(J/mol)",
    "ZPE": "zpe/(J/mol)",
}


def _as_frame(data):
    return pd.DataFrame(data)


def _as_frame_list(data, name):
    if data is None:
        return []
    if isinstance(data, (pd.DataFrame, dict)):
        return [_as_frame(data)]
    frames = [_as_frame(item) for item in data]
    if not frames:
        raise ValueError(f"{name} must not be empty")
    return frames


def _resolve_thermo_columns(data_frame):
    for columns in (_STRUCTURED_THERMO_COLUMNS, _LEGACY_THERMO_COLUMNS):
        if all(column in data_frame.columns for column in columns.values()):
            return columns
    raise ValueError(
        "could not infer thermo columns; expected structured or legacy thermo columns"
    )


def _thermo_values(data_frame, key):
    columns = _resolve_thermo_columns(data_frame)
    return data_frame[columns[key]].to_numpy(dtype=float)


def _require_matching_temperatures(reference, frames):
    for frame in frames:
        values = _thermo_values(frame, "T")
        if len(values) != len(reference) or not np.allclose(values, reference):
            raise ValueError("all thermo frames must use the same temperature grid")


def _sum_thermo_values(frames, key):
    total = np.zeros(len(frames[0]), dtype=float)
    for frame in frames:
        total += _thermo_values(frame, key)
    return total


def calculate_tst_rate_frame(
        transition_state_frame,
        reactant_frames,
        product_frames=None,
        delta_n=None,
        liquid=False,
        tunnelling_effect=None,
        imaginary_freq=None,
        sigma=1,
        reference_pressure=100000,
):
    """Calculate a TST rate scan from thermo tables without file I/O."""
    ts_frame = _as_frame(transition_state_frame)
    reactants = _as_frame_list(reactant_frames, "reactant_frames")
    products = _as_frame_list(product_frames, "product_frames")
    temperatures = _thermo_values(ts_frame, "T")
    _require_matching_temperatures(temperatures, reactants)
    _require_matching_temperatures(temperatures, products)

    if delta_n is None:
        delta_n = len(reactants) - 1
    delta_g = _thermo_values(ts_frame, "G") - _sum_thermo_values(reactants, "G")

    forward_barriers = None
    reverse_barriers = None
    if tunnelling_effect in {"eckart", "skodje_truhlar"}:
        if not products:
            raise ValueError(
                "product_frames must be provided for eckart or skodje_truhlar tunnelling"
            )
        ts_h0 = _thermo_values(ts_frame, "EE") + _thermo_values(ts_frame, "ZPE")
        reactant_h0 = _sum_thermo_values(reactants, "EE") + _sum_thermo_values(reactants, "ZPE")
        product_h0 = _sum_thermo_values(products, "EE") + _sum_thermo_values(products, "ZPE")
        forward_barriers = ts_h0 - reactant_h0
        reverse_barriers = ts_h0 - product_h0

    rate_constants = []
    for index, (temperature, delta_g_value) in enumerate(zip(temperatures, delta_g)):
        kwargs = {}
        if forward_barriers is not None:
            kwargs["delta_H_barrier_f_0K"] = forward_barriers[index]
            kwargs["delta_H_barrier_r_0K"] = reverse_barriers[index]
        rate_constants.append(
            k_TST(
                delta_G=delta_g_value,
                delta_n=delta_n,
                T=temperature,
                P0=reference_pressure,
                sigma=sigma,
                liquid=liquid,
                tunnelling_effect=tunnelling_effect,
                imaginary_freq=imaginary_freq,
                **kwargs,
            )
        )

    data = {
        "temperature": temperatures,
        "rate_constant": np.asarray(rate_constants, dtype=float),
        "delta_g": delta_g,
        "delta_n": np.full(len(temperatures), delta_n, dtype=int),
    }
    if forward_barriers is not None:
        data["delta_h_barrier_forward_0k"] = forward_barriers
        data["delta_h_barrier_reverse_0k"] = reverse_barriers
    return pd.DataFrame(data)


def k_TST(delta_G, delta_n, T=298.15, P0=100000, sigma=1,
          liquid=False,
          tunnelling_effect=None, imaginary_freq=None,
          delta_H_barrier_f_0K=None, delta_H_barrier_r_0K=None):
    """
    Calculates the rate constant using the Transition State Theory (TST) with optional corrections for tunnelling and liquid phase.

    Parameters:
    - delta_G: float, Gibbs free energy change of the reaction. G_TS - G_IS / (J/mol).
    - delta_n: int, difference in the number of gas molecules between products and reactants. Take 1 for bimolecular reactions and 0 for single-molecule reactions
    - T: float, temperature in Kelvin. Default is 298.15 K.
    - P0: float, reference pressure in Pascals. Default is 100000 Pa (1 bar).
    - sigma: Reaction path degeneracy = sigma_rot_TS/sigma_rot_IS This should be set to 1 if the rotational symmetry number is already taken into account when calculating G
    - liquid: bool, indicates if the reaction occurs in a liquid phase. Default is False. When set to True, P0 is useless.
    - tunnelling_effect: Optional[str], specifies the type of tunnelling correction to apply. Can be 'wigner', 'eckart', or 'skodje_truhlar'. Default is None.
    - imaginary_freq: Optional[float], frequency of the imaginary mode associated with the transition state. Required if tunnelling_effect is not None.
    - delta_H_barrier_f_0K: Optional[float], enthalpy barrier of the forward reaction at 0 K. Required for 'eckart' and 'skodje_truhlar' tunnelling effects.
    - delta_H_barrier_r_0K: Optional[float], enthalpy barrier of the reverse reaction at 0 K. Required for 'eckart' and 'skodje_truhlar' tunnelling effects.

    Returns:
        float, the calculated rate constant k.
        if liquid==True: k: (mol/m^3)^(-delta_n) * s^-1
        if liquid==False: k: (molecule/m^3)^(-delta_n) * s^-1

    Raises:
    - AssertionError: If required parameters for specific tunnelling corrections are not provided.
    - NotImplementedError: If an unsupported tunnelling effect is specified.
    """
    if liquid:
        k = sigma * k_b * T / h * np.exp(-delta_G / (R * T))
    else:
        k = sigma * k_b * T / h * (k_b * T / P0) ** delta_n * np.exp(-delta_G / (R * T))

    if tunnelling_effect is not None:
        assert imaginary_freq is not None, 'imaginary_freq must set when considering tunnelling effect'
        if tunnelling_effect == 'wigner':
            chi = wigner_correction(imaginary_freq=imaginary_freq, T=T, convert_unit=True)
        elif tunnelling_effect == 'eckart':
            assert delta_H_barrier_r_0K is not None, 'delta_H_barrier_r_0K must set when considering tunnelling effect by eckart method'
            assert delta_H_barrier_f_0K is not None, 'delta_H_barrier_f_0K must set when considering tunnelling effect by eckart method'
            chi = eckart_correction(imaginary_freq=imaginary_freq, T=T,
                                    delta_H_barrier_f_0K=delta_H_barrier_f_0K, delta_H_barrier_r_0K=delta_H_barrier_r_0K)
        elif tunnelling_effect == 'skodje_truhlar':
            assert delta_H_barrier_r_0K is not None, 'delta_H_barrier_r_0K must set when considering tunnelling effect by skodje_truhlar method'
            assert delta_H_barrier_f_0K is not None, 'delta_H_barrier_f_0K must set when considering tunnelling effect by skodje_truhlar method'
            chi = skodje_truhlar(imaginary_freq=imaginary_freq, T=T,
                                 delta_H_barrier_f_0K=delta_H_barrier_f_0K, delta_H_barrier_r_0K=delta_H_barrier_r_0K)
        else:
            raise NotImplemented(f'{tunnelling_effect} is not a right value')

        k *= chi

    return k


def k_TST_scan(thermo_ts_path, thermo_r1_path, thermo_r2_path=None, thermo_p_path=None,
               liquid=False,
               tunnelling_effect=None, imaginary_freq=None,
               sigma=1, out_path='QMkineticsScan.xlsx'):
    """
    Calculates the kinetic rate constants using the Transition State Theory (TST) method. The function supports
    consideration of tunneling effects through different models and can handle both gas-phase and liquid-phase reactions.
    The function reads thermodynamic data from provided Excel files, calculates the rate constants for a range of temperatures,
    and outputs the results to an Excel file.

    Parameters:
    - thermo_ts_path: str, path to the Excel file containing thermodynamic data for the transition state.
    - thermo_r1_path: str, path to the Excel file containing thermodynamic data for reactant 1.
    - thermo_r2_path: Optional[str], path to the Excel file containing thermodynamic data for reactant 2. Default is None.
    - thermo_p_path: Optional[str], path to the Excel file containing thermodynamic data for the product. Required if 'eckart' or 'skodje_truhlar' tunnelling effects are considered. Default is None.
    - liquid: bool, indicates whether the reaction occurs in a liquid phase. Default is False.
    - tunnelling_effect: Optional[str], specifies the type of tunnelling effect to consider. Options are 'wigner', 'eckart', and 'skodje_truhlar'. Default is None, meaning no tunnelling effect is considered.
    - imaginary_freq: Optional[float], the value of the imaginary frequency at the transition state, required when considering any tunnelling effect. Default is None.
    - sigma: int, symmetry number for the transition state. Default is 1.
    - out_path: Optional[str], path to the output Excel file where the calculated rate constants will be saved. Default is 'QMkineticsScan.xlsx'.

    Returns:
        pandas.DataFrame, contains the temperature ('T/K') and corresponding rate constant ('k') values.

    Raises:
    - AssertionError: If `imaginary_freq` is not provided when a tunnelling effect is specified.
    - AssertionError: If `thermo_p_path` is not provided when 'eckart' or 'skodje_truhlar' tunnelling effects are considered.
    - NotImplementedError: If an unsupported tunnelling effect is specified.
    """
    ts_thermo_df = pd.read_excel(thermo_ts_path)
    r1_thermo_df = pd.read_excel(thermo_r1_path)
    if thermo_r2_path is not None:
        r2_thermo_df = pd.read_excel(thermo_r2_path)
        delta_n = 1
        delta_G = ts_thermo_df['G/(J/mol)'] - r1_thermo_df['G/(J/mol)'] - r2_thermo_df['G/(J/mol)']
    else:
        delta_n = 0
        delta_G = ts_thermo_df['G/(J/mol)'] - r1_thermo_df['G/(J/mol)']
    T = ts_thermo_df['T/K']

    if tunnelling_effect is not None:
        assert imaginary_freq is not None, 'imaginary_freq must set when considering tunnelling effect'

        if tunnelling_effect == 'wigner':
            k_scan = np.array(
                [
                    k_TST(delta_G=delta_g, delta_n=delta_n, T=t, sigma=sigma, liquid=liquid,
                          tunnelling_effect='wigner', imaginary_freq=imaginary_freq)
                    for delta_g, t in zip(delta_G, T)
                ]
            )

        elif tunnelling_effect == 'eckart':
            assert thermo_p_path is not None, 'thermo_p_path must be set when considering tunneling effect by eckart method'
            p_thermo_df = pd.read_excel(thermo_p_path)
            if thermo_r2_path is not None:
                delta_H_barrier_f_0K_scan = ts_thermo_df['ee/(J/mol)'] + ts_thermo_df['zpe/(J/mol)'] - \
                                        (r1_thermo_df['ee/(J/mol)'] + r1_thermo_df['zpe/(J/mol)']) - \
                                        (r2_thermo_df['ee/(J/mol)'] + r2_thermo_df['zpe/(J/mol)'])
            else:
                delta_H_barrier_f_0K_scan = ts_thermo_df['ee/(J/mol)'] + ts_thermo_df['zpe/(J/mol)'] - \
                                         (r1_thermo_df['ee/(J/mol)'] + r1_thermo_df['zpe/(J/mol)'])
            delta_H_barrier_r_0K_scan = ts_thermo_df['ee/(J/mol)'] + ts_thermo_df['zpe/(J/mol)'] - \
                                   (p_thermo_df['ee/(J/mol)'] + p_thermo_df['zpe/(J/mol)'])
            k_scan = np.array(
                [
                    k_TST(delta_G=delta_g, delta_n=delta_n, T=t, sigma=sigma, liquid=liquid,
                          tunnelling_effect=tunnelling_effect,
                          imaginary_freq=imaginary_freq, delta_H_barrier_f_0K=f, delta_H_barrier_r_0K=r)
                    for delta_g, t, f, r in zip(delta_G, T, delta_H_barrier_f_0K_scan, delta_H_barrier_r_0K_scan)
                ]
            )

        elif tunnelling_effect == 'skodje_truhlar':
            assert thermo_p_path is not None, 'thermo_p_path must be set when considering tunneling effect by eckart method'
            p_thermo_df = pd.read_excel(thermo_p_path)
            if thermo_r2_path is not None:
                delta_H_barrier_f_0K_scan = ts_thermo_df['ee/(J/mol)'] + ts_thermo_df['zpe/(J/mol)'] - \
                                        (r1_thermo_df['ee/(J/mol)'] + r1_thermo_df['zpe/(J/mol)']) - \
                                        (r2_thermo_df['ee/(J/mol)'] + r2_thermo_df['zpe/(J/mol)'])
            else:
                delta_H_barrier_f_0K_scan = ts_thermo_df['ee/(J/mol)'] + ts_thermo_df['zpe/(J/mol)'] - \
                                         (r1_thermo_df['ee/(J/mol)'] + r1_thermo_df['zpe/(J/mol)'])
            delta_H_barrier_r_0K_scan = ts_thermo_df['ee/(J/mol)'] + ts_thermo_df['zpe/(J/mol)'] - \
                                   (p_thermo_df['ee/(J/mol)'] + p_thermo_df['zpe/(J/mol)'])
            k_scan = np.array(
                [
                    k_TST(delta_G=delta_g, delta_n=delta_n, T=t, sigma=sigma, liquid=liquid,
                          tunnelling_effect=tunnelling_effect,
                          imaginary_freq=imaginary_freq, delta_H_barrier_f_0K=f, delta_H_barrier_r_0K=r)
                    for delta_g, t, f, r in zip(delta_G, T, delta_H_barrier_f_0K_scan, delta_H_barrier_r_0K_scan)
                ]
            )

        else:
            raise NotImplemented(f'{tunnelling_effect} is not a right value')

    # 不考虑隧道效应时
    else:
        k_scan = np.array(
            [
                k_TST(delta_G=delta_g, delta_n=delta_n, T=t, sigma=sigma, liquid=liquid,
                      tunnelling_effect=tunnelling_effect,
                      imaginary_freq=imaginary_freq)
                for delta_g, t in zip(delta_G, T)
            ]
        )

    df = pd.DataFrame({'T/K': T, 'k': k_scan})
    if out_path is not None:
        df.to_excel(out_path, index=False)
    return df


def k_VTST(delta_G_list, delta_n, T=298.15, P0=100000, liquid=False, sigma=1,
           tunnelling_effect=None, imaginary_freq=None,
           delta_H_barrier_f_0K=None, delta_H_barrier_r_0K=None,
           also_get_k_tst=False):
    """
    Calculates the Variational Transition State Theory (VTST) rate constant for a given set of Gibbs free energy
    differences. The function also provides an option to return the Transition State Theory (TST) rate constants
    used in the VTST calculation.

    Args:
    - delta_G_list: List or array of floats representing the Gibbs free energy differences.
    - delta_n: Integer, change in the number of gas-phase molecules.
    - T: Float, temperature in Kelvin. Default is 298.15 K.
    - P0: Float, reference pressure in Pascals. Default is 100000 Pa.
    - liquid: Boolean, indicates if the reaction occurs in a liquid phase. Default is False. When set to True, P0 is useless.
    - sigma: Reaction path degeneracy = sigma_rot_TS/sigma_rot_IS This should be set to 1 if the rotational symmetry number is already taken into account when calculating G
    - liquid: bool, indicates if the reaction occurs in a liquid phase. Default is False.
    - tunnelling_effect: Optional[str], specifies the type of tunnelling correction to apply. Can be 'wigner', 'eckart', or 'skodje_truhlar'. Default is None.
    - imaginary_freq: Optional[float], frequency of the imaginary mode associated with the transition state. Required if tunnelling_effect is not None.
    - delta_H_barrier_f_0K: Optional[float], enthalpy barrier of the forward reaction at 0 K. Required for 'eckart' and 'skodje_truhlar' tunnelling effects.
    - delta_H_barrier_r_0K: Optional[float], enthalpy barrier of the reverse reaction at 0 K. Required for 'eckart' and 'skodje_truhlar' tunnelling effects.

    Returns:
    - k_vtst: Float, the calculated VTST rate constant.
    - k_tst_list: List of floats, the TST rate constants used in the VTST calculation. This is returned only if `also_get_k_tst` is True.
    """
    if isinstance(delta_G_list, list):
        delta_G_list = np.array(delta_G_list)
    k_tst_list = k_TST(delta_G=delta_G_list, delta_n=delta_n, T=T, P0=P0, sigma=sigma, liquid=liquid, tunnelling_effect=tunnelling_effect,
                       imaginary_freq=imaginary_freq, delta_H_barrier_f_0K=delta_H_barrier_f_0K, delta_H_barrier_r_0K=delta_H_barrier_r_0K)
    k_vtst = np.min(k_tst_list)
    if also_get_k_tst:
        return k_vtst, k_tst_list
    return k_vtst


def k_VTST_scan(thermo_irc_path_list: List[str],
                thermo_r1_path, thermo_r2_path=None, thermo_p_path=None,
                liquid=False,
                tunnelling_effect=None, imaginary_freq=None,
                sigma=1, also_get_k_tst_scan=False, out_path='QMkineticsScanVTST.xlsx',
                out_TST_path='QMkineticsScanTST.xlsx'):
    """
    Calculates the variational transition state theory (VTST) rate constants for a given set of thermodynamic paths.
    This function computes the VTST rate constants by first calculating the TST (Transition State Theory) rate
    constants for each path in the provided list, then taking the minimum of these TST rate constants at each
    temperature. Optionally, it can also return the TST scan data.

    Parameters:
    - thermo_irc_path_list: List of strings representing the file paths to the IRC (Intrinsic Reaction Coordinate)
            thermodynamic data for each reaction path.
    - thermo_r1_path: String representing the file path to the thermodynamic data for reactant 1.
    - thermo_r2_path: Optional string representing the file path to the thermodynamic data for reactant 2.
    - thermo_p_path: Optional string representing the file path to the thermodynamic data for the product.
    - liquid: Boolean indicating if the reaction is in a liquid phase. Default is False.
    - tunnelling_effect: Optional parameter to account for the tunnelling effect in the calculations.
    - imaginary_freq: Optional parameter specifying the imaginary frequency used in the calculations.
    - sigma: Reaction path degeneracy = sigma_rot_TS/sigma_rot_IS This should be set to 1 if the rotational symmetry number is already taken into account when calculating G
    - also_get_k_tst_scan: Boolean indicating whether to also calculate and return the TST scan data.
            Default is False.
    - out_path: String representing the output file path for the VTST scan data. Default is 'QMkineticsScanVTST.xlsx'.
    - out_TST_path: String representing the output file path for the TST scan data, if requested.
            Default is 'QMkineticsScanTST.xlsx'.

    Returns:
        A pandas DataFrame containing the VTST scan data with columns 'T/K' and 'k', representing temperature and
        the VTST rate constant, respectively. If `also_get_k_tst_scan` is True, a second DataFrame containing
        the TST scan data is also returned, with each column (except 'T/K') named after the corresponding
        IRC path's basename, and values being the TST rate constants for that path.

    Raises:
    - ValueError: If any of the required input files are not found or cannot be read.
    - TypeError: If the types of the arguments do not match their expected types.
    """
    k_tst_scan_list = [
        k_TST_scan(thermo_ts_path=i, thermo_r1_path=thermo_r1_path, thermo_r2_path=thermo_r2_path,
                   thermo_p_path=thermo_p_path,
                   liquid=liquid,
                   tunnelling_effect=tunnelling_effect, imaginary_freq=imaginary_freq,
                   sigma=sigma)['k'].values for i in thermo_irc_path_list
    ]

    T = pd.read_excel(thermo_r1_path)['T/K']

    k_vtst_scan = np.minimum.reduce(k_tst_scan_list)
    df_vtst = pd.DataFrame({'T/K': T, 'k': k_vtst_scan})
    if out_path is not None:
        df_vtst.to_excel(out_path, index=False)

    if also_get_k_tst_scan:
        df_tst = {basename(i): j for i, j in zip(thermo_irc_path_list, k_tst_scan_list)}
        df_tst['T/K'] = T
        df_tst = pd.DataFrame(df_tst)
        if out_TST_path is not None:
            df_tst.to_excel(out_TST_path, index=False)
        return df_vtst, df_tst


    return df_vtst



# if __name__ == '__main__':
#
#     k = k_TST(delta_G=23000, delta_n=0)
#     print(k)
#
#     k_vtst = k_VTST(delta_G_list=[23000, 24000], delta_n=0)
#     print(k_vtst)