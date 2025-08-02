"""
根据Shermo输出的扫描热力学数据，根据NASA等模型拟合公式，用于制作cantera的输入
"""
import os
from os.path import isfile, join
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from matplotlib import rcParams
import matplotlib.pyplot as plt

# from read_shermo_scan import read_shermo_scan
from convert_shermo_scan_to_cantera_yaml import write_cantera_yaml_species_NASA7


kj_to_j = 1000
kcal_to_kj = 4.184
cal_to_j = 4.184
hartree_to_kj = 2625.5


class NASA7:
    def __init__(self, a0, a1, a2, a3, a4, a5, a6, return_mode='fit'):
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.a6 = a6
        self.return_mode = return_mode

    def __call__(self, T):
        if self.return_mode == 'fit':
            return nasa7_for_fit(T, self.a0, self.a1, self.a2, self.a3, self.a4, self.a5, self.a6)
        else:
            out = nasa7(T, self.a0, self.a1, self.a2, self.a3, self.a4, self.a5, self.a6)
            if self.return_mode == 'all':
                return out
            elif self.return_mode == 'Cp_T':
                return out[0]
            elif self.return_mode == 'H_T':
                return out[1]
            elif self.return_mode == 'S_T':
                return out[2]
            else:
                raise NotImplementedError


def nasa7(T, a0, a1, a2, a3, a4, a5, a6):
    R = 8.314
    Cp_T = R * (a0 + a1 * T + a2 * T ** 2 + a3 * T ** 3 + a4 * T ** 4)
    H_T = R * T * (a0 + a1 / 2 * T + a2 / 3 * T ** 2 + a3 / 4 * T ** 3 + a4 / 5 * T ** 4 + a5 / T)
    S_T = R * (a0 * np.log(T) + a1 * T + a2 / 2 * T ** 2 + a3 / 3 * T ** 3 + a4 / 4 * T ** 4 + a6)
    return Cp_T, H_T, S_T


def nasa7_for_fit(T, a0, a1, a2, a3, a4, a5, a6):
    Cp_T, H_T, S_T = nasa7(T, a0, a1, a2, a3, a4, a5, a6)
    out = np.hstack([Cp_T, H_T, S_T])
    return out


class NASA9:
    def __init__(self, a0, a1, a2, a3, a4, a5, a6, a7, a8, return_mode='fit'):
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.a6 = a6
        self.a7 = a7
        self.a8 = a8
        self.return_mode = return_mode

    def __call__(self, T):
        if self.return_mode == 'fit':
            return nasa9_for_fit(T, self.a0, self.a1, self.a2, self.a3, self.a4, self.a5, self.a6, self.a7, self.a8)
        else:
            out = nasa9(T, self.a0, self.a1, self.a2, self.a3, self.a4, self.a5, self.a6, self.a7, self.a8)
            if self.return_mode == 'all':
                return out
            elif self.return_mode == 'Cp_T':
                return out[0]
            elif self.return_mode == 'H_T':
                return out[1]
            elif self.return_mode == 'S_T':
                return out[2]
            else:
                raise NotImplementedError


def nasa9(T, a0, a1, a2, a3, a4, a5, a6, a7, a8):
    R = 8.314
    Cp_T = R * (a0 * T ** -2 + a1 * T ** -1 +
                a2 + a3 * T + a4 * T ** 2 + a5 * T ** 3 + a6 * T ** 4)
    H_T = R * T * (-a0 * T ** -2 + a1 * np.log(T) / T +
                   a2 + a3 / 2 * T + a4 / 3 * T ** 2 + a5 / 4 * T ** 3 + a6 / 5 * T ** 4 + a7 / T)
    S_T = R * T * (-a0 / 2 * T ** -2 - a1 * T ** -1 +
                   a2 * np.log(T) + a3 * T + a4 / 2 * T ** 2 + a5 / 3 * T ** 3 + a6 / 4 * T ** 4 + a8)
    return Cp_T, H_T, S_T


def nasa9_for_fit(T, a0, a1, a2, a3, a4, a5, a6, a7, a8):
    Cp_T, H_T, S_T = nasa9(T, a0, a1, a2, a3, a4, a5, a6, a7, a8)
    out = np.hstack([Cp_T, H_T, S_T])
    return out


class Shomate:
    def __init__(self, A, B, C, D, E, F, G, return_mode='fit'):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F
        self.G = G
        self.return_mode = return_mode

    def __call__(self, T):
        if self.return_mode == 'fit':
            return shomate_for_fit(T, self.A, self.B, self.C, self.D, self.E, self.F, self.G)
        else:
            out = shomate(T, self.A, self.B, self.C, self.D, self.E, self.F, self.G)
            if self.return_mode == 'all':
                return out
            elif self.return_mode == 'Cp_T':
                return out[0]
            elif self.return_mode == 'H_T':
                return out[1]
            elif self.return_mode == 'S_T':
                return out[2]
            else:
                raise NotImplementedError


def shomate(T, A, B, C, D, E, F, G):
    t = T / 1000
    Cp_T = A + B * t + C * t ** 2 + D * t ** 3 + E * t ** -2
    H_T = A * t + B / 2 * t ** 2 + C / 3 * t ** 3 + D / 4 * t ** 4 - E / t + F
    S_T = A * np.log(t) + B * t + C / 2 * t ** 2 + D / 3 * t ** 3 - E / 2 * t ** -2 + G
    return Cp_T, H_T, S_T


def shomate_for_fit(T, A, B, C, D, E, F, G):
    Cp_T, H_T, S_T = shomate(T, A, B, C, D, E, F, G)
    out = np.hstack([Cp_T, H_T, S_T])
    return out


def fit(fun, xdata, ydata, sigma, p0, bounds, maxfev=10000):
    print('-' * 20 + 'START' + '-' * 20)
    popt, pcov = curve_fit(f=fun, xdata=xdata, ydata=ydata, sigma=sigma, p0=p0, bounds=bounds, maxfev=maxfev)
    print(f'fitted model parameters: \n{popt}')
    print(f'cov of fitted model parameters: \n{pcov}')
    return popt, pcov


def cal_metric(y_label, y_pred, key='model_performance', save=None, save_root_path='.'):
    r2 = round(r2_score(y_label, y_pred), 3)
    mse = round(mean_squared_error(y_label, y_pred), 3)
    mae = round(mean_absolute_error(y_label, y_pred), 3)
    mape = round(mean_absolute_percentage_error(y_label, y_pred), 3)
    print('-' * 50)
    print(f' {key}\n r2: {r2} \n mse: {mse} \n mae: {mae} \n mape: {mape}')
    if save:
        with open(join(save_root_path, f'{key}.txt'), 'w') as f:
            f.write(f'{key} \n')
            f.write(f' r2: {r2} \n mse: {mse} \n mae: {mae} \n mape: {mape}')
    return r2, mse, mae, mape


def plot_fit(x, y, F, x_label='x', y_label='y', save=None, save_root_path='.'):
    # ['train $\mathrm{R^2}$', 'val $\mathrm{R^2}$', 'test $\mathrm{R^2}$']  # 若不写\mathrm{}则会是斜体的效果 上下标的写法
    # 首先配置字体信息
    config = {
        "font.family": 'serif',
        "font.size": 16,
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman'],
    }
    rcParams.update(config)
    # experiment data
    fig, ax = plt.subplots(layout='constrained')
    ax.scatter(x, y, label='Experiment data')
    # model prediction curve
    x_plot = np.linspace(start=x[0], stop=x[-1], num=500)
    y_plot = F(x_plot)
    ax.plot(x_plot, y_plot, label='Model prediction')

    # plot settings
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    if save:
        plt.savefig(join(save_root_path, save), dpi=1000)
        plt.close()
        export_data(x_data=x, y_data=y, export_path=join(save_root_path, f'{save}_experiment_data_scatter.txt'))
        export_data(x_data=x_plot, y_data=y_plot, export_path=join(save_root_path, f'{save}_fit_data_curve.txt'))
    else:
        plt.show()
        plt.close()
    return None


def export_data(x_data, y_data, export_path):
    with open(export_path, 'w') as f:
        for x, y in zip(x_data, y_data):
            f.write(f'{x:.5f}  {y:.5f}\n')
    f.close()
    return None


def calculate_scan_formation_thermo(n_C, n_H, df_target, level='M062X_def2TZVP'):
    # 读取H2和石墨的数据，用于计算生成焓
    H2_df = load_qm_scan_data(SCq_path=f'data/H2_scan_SCq_{level}.txt', UHG_path=f'data/H2_scan_UHG_{level}.txt')
    C_df = load_qm_scan_data(SCq_path=f'data/C_scan_SCq_{level}.txt', UHG_path=f'data/C_scan_UHG_{level}.txt')
    H2_df = H2_df[['T(K)', 'H', 'S']].rename(columns={'H': 'H_H2', 'S': 'S_H2'})
    C_df = C_df[['T(K)', 'H', 'S']].rename(columns={'H': 'H_C', 'S': 'S_C'})
    df_data = pd.merge(H2_df, C_df, left_on='T(K)', right_on='T(K)')

    df = pd.merge(df_data, df_target, left_on='T(K)', right_on='T(K)')

    T = df['T(K)']
    # target molecule
    H_T = df['H']
    S_T = df['S']

    # H2
    H2_H_T = df['H_H2']
    H2_S_T = df['S_H2']

    # 石墨
    C_H_T = df['H_C']
    C_S_T = df['S_C']

    # 升华
    H_sublimation = df['Hf°']
    S_sublimation = df['Sf°']

    H_formation = H_T - n_C * C_H_T - n_H / 2 * H2_H_T + n_C * H_sublimation
    S_formation = S_T - n_C * C_S_T - n_H / 2 * H2_S_T + n_C * S_sublimation
    return T, H_formation, S_formation


def load_sublimation_data_of_graphite(data_path='data/C.xlsx'):
    df = pd.read_excel(data_path)
    df['Hf°'] *= kj_to_j
    df['Gf°'] *= kj_to_j
    return df

def load_qm_scan_data(SCq_path, UHG_path):
    SCq_df = read_shermo_scan(read_path=SCq_path, column_name_list=None, skip_first_row=3, write_path=None)
    UHG_df = read_shermo_scan(read_path=UHG_path, column_name_list=None, skip_first_row=3, write_path=None)
    qm_df = pd.merge(SCq_df, UHG_df, left_on='T(K)', right_on='T(K)')
    qm_df['CP'] *= cal_to_j
    qm_df['CV'] *= cal_to_j
    qm_df['U'] *= (hartree_to_kj * kj_to_j)
    qm_df['H'] *= (hartree_to_kj * kj_to_j)
    qm_df['G'] *= (hartree_to_kj * kj_to_j)
    qm_df['Ucorr'] *= (kcal_to_kj * kj_to_j)
    qm_df['Hcorr'] *= (kcal_to_kj * kj_to_j)
    qm_df['Gcorr'] *= (kcal_to_kj * kj_to_j)
    qm_df['S'] *= cal_to_j
    return qm_df


if __name__ == "__main__":

    # load experiment data
    name = '43'
    n_C = 20
    n_H = 24
    df_qm = load_qm_scan_data(SCq_path='scan_SCq.txt', UHG_path='scan_UHG.txt')
    df_sublimation = load_sublimation_data_of_graphite(data_path='data/C.xlsx')
    df = pd.merge(df_sublimation, df_qm, left_on='T(K)', right_on='T(K)')

    Cp_T = df['CP']
    T, H_T, S_T = calculate_scan_formation_thermo(n_C=n_C, n_H=n_H, df_target=df, level='wB97M(2)_def2TZVP')

    result = pd.DataFrame({'T': T, 'Cp': Cp_T, 'H': H_T, 'S': S_T})
    result.to_csv('result.csv')

    # preprocess data
    start = 2
    end = 15
    T, H_T, S_T, Cp_T = T.to_numpy(), H_T.to_numpy(), S_T.to_numpy(), Cp_T.to_numpy()
    T, H_T, S_T, Cp_T = T[start:end], H_T[start:end], S_T[start:end], Cp_T[start:end]
    n_data = len(T)

    # 计算每个物理量的量级（绝对值均值）
    scale_cp = np.mean(np.abs(Cp_T))
    scale_h = np.mean(np.abs(H_T)) / 2
    scale_s = np.mean(np.abs(S_T)) / 10
    # 设置权重：与量级成反比（量级越小权重越大）
    weight_cp = 1.0 / scale_cp
    weight_h = 1.0 / scale_h
    weight_s = 1.0 / scale_s
    print(f'weight: Cp:{weight_cp}, H:{weight_h}, S:{weight_s}')
    # 构造权重向量（对每个数据点应用相应权重）
    weights = np.concatenate([
        np.full(n_data, weight_cp),  # Cp部分
        np.full(n_data, weight_h),  # H部分
        np.full(n_data, weight_s)  # S部分
    ])

    # preprocess data
    X = np.array(T)
    Y = np.hstack([Cp_T, H_T, S_T])
    print(f'X data shape: {X.shape}')
    print(f'Y data shape: {Y.shape}')

    # settings
    FUN = nasa7_for_fit
    FUN_CLASS = NASA7
    n_parm = 7

    GUESS = None
    SIGMA = 1 / weights  # todo 搞清楚这个SIGMA的原理
    BOUNDS = ([-np.inf] * n_parm, [np.inf] * n_parm)
    MAXFEV = 100000

    # fit
    popt, pcov = fit(fun=FUN, xdata=X, ydata=Y, sigma=SIGMA, p0=GUESS, bounds=BOUNDS, maxfev=MAXFEV)
    fitted_model = FUN_CLASS(*popt, return_mode='all')

    Cp_T_pre, H_T_pre, S_T_pre = fitted_model(T=T)

    cal_metric(y_label=Cp_T, y_pred=Cp_T_pre, key='Cp_T', save=True, save_root_path='.')
    cal_metric(y_label=H_T, y_pred=H_T_pre, key='H_T', save=True, save_root_path='.')
    cal_metric(y_label=S_T, y_pred=S_T_pre, key='S_T', save=True, save_root_path='.')

    plot_fit(x=X, y=Cp_T, F=FUN_CLASS(*popt, return_mode='Cp_T'), x_label='T', y_label='Cp_T',
             save='Cp_T.png', save_root_path='.')
    plot_fit(x=X, y=H_T, F=FUN_CLASS(*popt, return_mode='H_T'), x_label='T', y_label='H_T',
             save='H_T.png', save_root_path='.')
    plot_fit(x=X, y=S_T, F=FUN_CLASS(*popt, return_mode='S_T'), x_label='T', y_label='S_T',
             save='S_T.png', save_root_path='.')


    write_cantera_yaml_species_NASA7(specie_name=name, n_C=n_C, n_H=n_H,
                                     T_range=[float(T[0]), float(T[-1])],
                                     nasa7_parameters=[float(i) for i in popt])