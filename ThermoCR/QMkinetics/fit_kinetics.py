import os
from os.path import join
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from matplotlib import rcParams
import matplotlib.pyplot as plt

from ThermoCR.tools.constant import R
from ThermoCR.tools.about_cantera.export_cantera_kinetics_yaml import make_cantera_reaction_yaml


class Arrhenius:
    def __init__(self, A, Ea, b=1):
        self.A = A
        self.Ea = Ea
        self.b = b

    def __call__(self, T):
        return arrhenius(T=T, A=self.A, Ea=self.Ea, b=self.b)

    def get_parameters(self):
        return self.A, self.Ea, self.b


def arrhenius(T, A, Ea, b=1):
    k = A * T ** b * np.exp(-Ea / (R * T))
    return k


class Arrhenius2Piecewise:
    def __init__(self, A1, b1, b2, Ea1, Ea2, T_cut):
        self.A1 = A1
        self.b1 = b1
        self.b2 = b2
        self.Ea1 = Ea1
        self.Ea2 = Ea2
        self.T_cut = T_cut
        self.A2 = None

    def __call__(self, T):
        return arrhenius_2piecewise(T, A1=self.A1, b1=self.b1, b2=self.b2, Ea1=self.Ea1, Ea2=self.Ea2, T_cut=self.T_cut)

    def get_parameters(self):
        self.A2 = A_nplus1(A_n=self.A1, Ea_n=self.Ea1, Ea_nplus1=self.Ea2, b_n=self.b1, b_nplus1=self.b2, T_cut_n=self.T_cut)
        return {'A1':self.A1, 'b1': self.b1, 'Ea1': self.Ea1, 'A2': self.A2, 'b2': self.b2, 'Ea2':self.Ea2, 'Tcut':self.T_cut}


def arrhenius_2piecewise(T, A1, b1, b2, Ea1, Ea2, T_cut):
    A2 = A_nplus1(A_n=A1, Ea_n=Ea1, Ea_nplus1=Ea2, b_n=b1, b_nplus1=b2, T_cut_n=T_cut)

    k = np.zeros_like(T)

    mask_low = T < T_cut
    mask_high = ~mask_low

    k[mask_low] = A1 * T[mask_low] ** b1 * np.exp(-Ea1 / (R * T[mask_low]))
    k[mask_high] = A2 * T[mask_high] ** b2 * np.exp(-Ea2 / (R * T[mask_high]))
    return k


def A_nplus1(A_n, Ea_n, Ea_nplus1, b_n, b_nplus1, T_cut_n):
    A_nplus1 = A_n * T_cut_n ** (b_n - b_nplus1) * np.exp((Ea_nplus1 - Ea_n) / (R * T_cut_n))
    return A_nplus1


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


def convert_k_unit_from_ThermoCR_to_Cantera(k):
    """
    convert k unit: (mol/m^3)^(-delta_n) * s^-1  -->  (kmol/m^3)^(-delta_n) * s^-1
    Args:
        k:

    Returns: k

    """
    print('convert k unit: (mol/m^3)^(-delta_n) * s^-1  -->  (kmol/m^3)^(-delta_n) * s^-1')
    k = k * 1000
    return k


def fit_kinetics_model(
        data_path: str,
        r_name_list, p_name_list, reversible=True,
        model_type: str = 'Arrhenius',
        data_columns=None,
        start_index: int = 0,
        end_index: int = None,
        output_dir: str = ".",
        save_plots: bool = True,
        save_metrics: bool = True,
        write_yaml: bool = True,
        guess: list = None,
        bounds: tuple = None,
        maxfev: int = 100000,
        convert_k_unit_fun = convert_k_unit_from_ThermoCR_to_Cantera
):
    """
    Fits a kinetic model to the provided experimental data. The function supports
    fitting of different types of models, saving plots and metrics, and writing
    the results to a Cantera YAML file.

    Parameters:
    - data_path: Path to the Excel file containing the experimental data.
    - r_name_list: List of reactant names for the reaction.
    - p_name_list: List of product names for the reaction.
    - reversible: Boolean indicating if the reaction is reversible.
    - model_type: Type of the kinetic model to be fitted. Default is 'Arrhenius'.
    - data_columns: Dictionary mapping column names in the data file to the
            corresponding data (e.g., temperature, rate constant). If not provided,
            default values are used.
    - start_index: Index of the first data point to be used for fitting. Default is 0.
    - end_index: Index of the last data point to be used for fitting. If None, all
            data points from start_index to the end of the dataset are used.
    - output_dir: Directory where the output files (plots, metrics, YAML) will be saved.
            Default is the current directory.
    - save_plots: Boolean indicating whether to save the plots of the fitted model.
            Default is True.
    - save_metrics: Boolean indicating whether to save the metrics of the fitted model.
            Default is True.
    - write_yaml: Boolean indicating whether to write the fitted parameters to a
            Cantera YAML file. Default is True.
    - guess: Initial guess for the parameters of the model. If None, the solver
            will use its own initial guess. See document of Scipy for more details.
    - bounds: Tuple of lower and upper bounds for the model parameters. If None,
            no bounds are applied. See document of Scipy for more details.
    - maxfev: Maximum number of function evaluations for the curve fitting. Default is 100000. See document of Scipy for more details.

    Returns:
        A tuple containing the optimized parameters (popt) and the fitted model object.
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
    T = df[data_columns["T"]].to_numpy()[start_index:end_index]
    T = np.array(T, dtype=float)
    k = df[data_columns['k']].to_numpy()[start_index:end_index]

    if convert_k_unit_fun is not None:
        k = convert_k_unit_fun(k)

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
        "Arrhenius2Piecewise": {
            "fit_func": arrhenius_2piecewise,
            "model_class": Arrhenius2Piecewise,
            "n_params": 6,
            "yaml_writer": None,
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
    params = fitted_model.get_parameters()
    print('Fitted parameters:')
    print(params)

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
                r_name_list=r_name_list, p_name_list=p_name_list, A=params[0], Ea=params[1], b=params[2],
                reversible=reversible, root_path=output_dir
            )

    return popt, fitted_model


# if __name__ == '__main__':
#
#     bound = ([0, -np.inf, -np.inf, 0, 0, 150], [np.inf, np.inf, np.inf, np.inf, np.inf, 301])
#     fit_kinetics_model(
#         model_type='Arrhenius2Piecewise',
#         bounds=bound,
#         data_path='../../03_09_vtst.xlsx',
#         r_name_list=['S01', 'S01'],
#         p_name_list=['S02'],
#         output_dir='../../Arrhenius_result2',
#         start_index=0,
#         end_index=None,
#         maxfev=1000000
#     )
#
#     fit_kinetics_model(
#         model_type='Arrhenius',
#         data_path='../../03_09_vtst.xlsx',
#         r_name_list=['S01', 'S01'],
#         p_name_list=['S02'],
#         output_dir='../../Arrhenius_result',
#         start_index=0,
#         end_index=None
#     )