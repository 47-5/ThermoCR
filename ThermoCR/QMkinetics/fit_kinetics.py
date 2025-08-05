import os
from os.path import isfile, join
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from matplotlib import rcParams
import matplotlib.pyplot as plt

from ThermoCR.tools.constant import R
from ThermoCR.QMkinetics.export_cantera_kinetics_yaml import make_cantera_reaction_yaml


class Arrhenius:
    def __init__(self, A, Ea, b=1):
        self.A = A
        self.Ea = Ea
        self.b = b

    def __call__(self, T):
        return arrhenius(T=T, A=self.A, Ea=self.Ea, b=self.b)


def arrhenius(T, A, Ea, b=1):
    k = A * T ** b * np.exp(-Ea / (R * T))
    return k


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
        plt.savefig(join(save_root_path, f'{save}.png'), dpi=1000)
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
            f.write(f'{x:.5f}  {y:.5e}\n')
    f.close()
    return None


def fit_kinetics_model(
        data_path: str,
        r_name_list, p_name_list, reversible=True,
        model_type: str = 'Arrhenius',
        data_columns=None,

        output_dir: str = ".",
        save_plots: bool = True,
        save_metrics: bool = True,
        write_yaml: bool = True,
        guess: list = None,
        bounds: tuple = None,
        maxfev: int = 100000,
):
    """
    根据热力学数据拟合NASA/Shomate模型参数

    参数:
    data_path -- 热力学数据文件路径 (Excel格式)
    model_type -- 模型类型: Arrhenius
    data_columns -- 数据列名映射字典 (默认: Shermo输出格式)
    output_dir -- 输出目录 (默认: 当前目录)
    save_plots -- 是否保存拟合图表 (默认: True)
    save_metrics -- 是否保存评估指标 (默认: True)
    write_yaml -- 是否输出Cantera YAML文件 (默认: True)
    guess -- 初始参数猜测值 (默认: None)
    bounds -- 参数边界 (默认: None表示无界)
    maxfev -- 最大函数评估次数 (默认: 100000)

    返回:
    拟合参数和模型对象
    """
    # 确保输出目录存在
    if data_columns is None:
        data_columns = {
            "T": "T/K",
            "k": "k",
        }
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    df = pd.read_excel(data_path)

    # 提取列数据
    T = df[data_columns["T"]].to_numpy()
    T = np.array(T, dtype=float)
    k = df[data_columns['k']].to_numpy()

    n_data = len(T)
    X = T
    Y = k

    # 选择模型函数
    model_info = {
        "Arrhenius": {
            "fit_func": arrhenius,
            "model_class": Arrhenius,
            "n_params": 3,
            "yaml_writer": make_cantera_reaction_yaml,
        },
    }

    if model_type not in model_info:
        raise ValueError(f"不支持的模型类型: {model_type}")

    model = model_info[model_type]
    FUN = model["fit_func"]
    FUN_CLASS = model["model_class"]
    n_params = model["n_params"]


    SIGMA = None
    # 设置默认边界
    if bounds is None:
        bounds = ([-np.inf] * n_params, [np.inf] * n_params)

    # 拟合模型
    popt, pcov = curve_fit(
        f=FUN,
        xdata=X,
        ydata=Y,
        sigma=SIGMA,
        p0=guess,
        bounds=bounds,
        maxfev=maxfev
    )

    # 创建拟合模型对象
    fitted_model = FUN_CLASS(*popt)

    # 计算预测值
    k_pre = fitted_model(T)

    # 评估模型
    if save_metrics:
        cal_metric(k, k_pre, "k", save=True, save_root_path=output_dir)


    # 绘制图表
    if save_plots:
        plot_fit(
            X, k, FUN_CLASS(*popt),
            "T", "k", "k", output_dir
        )

    # 输出Cantera YAML
    if write_yaml and model["yaml_writer"]:
        if model_type == "Arrhenius":
            model["yaml_writer"](
                r_name_list=r_name_list, p_name_list=p_name_list, A=popt[0], Ea=popt[1], b=popt[2],
                reversible=reversible, root_path=output_dir
            )

    return popt, fitted_model


# if __name__ == '__main__':
#
#     fit_kinetics_model(
#         data_path='../QMkineticsScan.xlsx',
#         r_name_list=['S01', 'S01'],
#         p_name_list=['S02'],
#         output_dir='../Arrhenius_result'
#     )