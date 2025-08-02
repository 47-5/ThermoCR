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

from convert_shermo_scan_to_cantera_yaml import write_cantera_yaml_species_NASA7


kj_to_j = 1000


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
        export_data(x_data=x, y_data=y, export_path=join(save_root_path, f'{save}_experiment_data_scatter.xlsx'))
        export_data(x_data=x_plot, y_data=y_plot, export_path=join(save_root_path, f'{save}_fit_data_curve.xlsx'))
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


if __name__ == "__main__":

    # load experiment data
    name = '43'
    n_C = 5
    n_H = 6
    df = pd.read_excel('QMThermoScan.xlsx')

    Cp_T = df['Cp/(J/mol/K)']
    T, H_T, S_T = df['T/K'], df['H/(J/mol)'], df['S/(J/mol/K)']


    # preprocess data
    start = 0
    end = 21
    T, H_T, S_T, Cp_T = T.to_numpy(), H_T.to_numpy(), S_T.to_numpy(), Cp_T.to_numpy()
    T, H_T, S_T, Cp_T = T[start:end], H_T[start:end], S_T[start:end], Cp_T[start:end]
    n_data = len(T)

    # 计算每个物理量的量级（绝对值均值）
    scale_cp = np.mean(np.abs(Cp_T))
    scale_h = np.mean(np.abs(H_T))
    scale_s = np.mean(np.abs(S_T))
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