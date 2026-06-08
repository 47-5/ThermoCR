"""Thermodynamic model fitting helpers."""

from dataclasses import dataclass
import os
from os.path import join
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from matplotlib import rcParams
import matplotlib.pyplot as plt

from ThermoCR.export.cantera import write_cantera_yaml_thermo_NASA7, write_cantera_yaml_thermo_NASA9, write_cantera_yaml_thermo_Shomate


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

_STRUCTURED_THERMO_COLUMNS = {
    "T": "temperature",
    "Cp": "heat_capacity_cp",
    "H": "enthalpy",
    "S": "entropy",
}
_LEGACY_THERMO_COLUMNS = {
    "T": "T/K",
    "Cp": "Cp/(J/mol/K)",
    "H": "H/(J/mol)",
    "S": "S/(J/mol/K)",
}
_THERMO_MODEL_INFO = {
    "NASA7": {
        "fit_func": nasa7_for_fit,
        "model_class": NASA7,
        "n_params": 7,
        "yaml_writer": write_cantera_yaml_thermo_NASA7,
    },
    "NASA9": {
        "fit_func": nasa9_for_fit,
        "model_class": NASA9,
        "n_params": 9,
        "yaml_writer": write_cantera_yaml_thermo_NASA9,
    },
    "Shomate": {
        "fit_func": shomate_for_fit,
        "model_class": Shomate,
        "n_params": 7,
        "yaml_writer": write_cantera_yaml_thermo_Shomate,
    },
}


@dataclass
class ThermoFitResult:
    """Structured thermodynamic model fit result."""

    model_type: str
    parameters: tuple
    temperature_range: tuple
    metrics: dict
    covariance: np.ndarray = None

    def __post_init__(self):
        self.model_type = _canonical_model_type(self.model_type)
        self.parameters = tuple(float(parameter) for parameter in self.parameters)
        self.temperature_range = tuple(float(value) for value in self.temperature_range)
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance, dtype=float)

    def model(self, return_mode="all"):
        """Return a callable model object for the fitted parameters."""
        model_class = _get_thermo_model_info(self.model_type)["model_class"]
        return model_class(*self.parameters, return_mode=return_mode)

    def predict(self, temperatures):
        """Predict Cp, H, and S at the requested temperatures."""
        return self.model(return_mode="all")(np.asarray(temperatures, dtype=float))

    def as_dict(self):
        covariance = None
        if self.covariance is not None:
            covariance = self.covariance.tolist()
        return {
            "model_type": self.model_type,
            "parameters": list(self.parameters),
            "temperature_range": list(self.temperature_range),
            "metrics": self.metrics,
            "covariance": covariance,
        }


def _canonical_model_type(model_type):
    for candidate in _THERMO_MODEL_INFO:
        if str(model_type).lower() == candidate.lower():
            return candidate
    raise ValueError(f"unsupported thermo model type: {model_type}")


def _get_thermo_model_info(model_type):
    return _THERMO_MODEL_INFO[_canonical_model_type(model_type)]


def _resolve_thermo_columns(data_frame, data_columns):
    if data_columns is not None:
        required = {"T", "Cp", "H", "S"}
        missing_keys = required.difference(data_columns)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"data_columns is missing keys: {missing}")
        return data_columns

    for columns in (_STRUCTURED_THERMO_COLUMNS, _LEGACY_THERMO_COLUMNS):
        if all(column in data_frame.columns for column in columns.values()):
            return columns

    raise ValueError(
        "could not infer thermo columns; provide data_columns with T, Cp, H, and S"
    )


def _extract_thermo_fit_arrays(data_frame, data_columns, start_index, end_index):
    columns = _resolve_thermo_columns(data_frame, data_columns)
    selected = data_frame.iloc[start_index:end_index]
    arrays = tuple(
        selected[columns[key]].to_numpy(dtype=float)
        for key in ("T", "Cp", "H", "S")
    )

    if len(arrays[0]) == 0:
        raise ValueError("no thermo rows were selected for fitting")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("thermo fitting data must contain only finite numbers")
    return arrays


def _build_fit_sigma(Cp_T, H_T, S_T, weight_strategy):
    if weight_strategy == "uniform":
        return None
    if weight_strategy != "inverse_mean_abs":
        raise ValueError(
            "weight_strategy must be 'inverse_mean_abs' or 'uniform'"
        )

    sigma_parts = []
    for values in (Cp_T, H_T, S_T):
        scale = float(np.mean(np.abs(values)))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        sigma_parts.append(np.full(len(values), scale))
    return np.concatenate(sigma_parts)


def _fit_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def fit_thermo_frame(
        data_frame,
        model_type: str = "NASA7",
        data_columns=None,
        start_index: int = 0,
        end_index: int = None,
        weight_strategy: str = "inverse_mean_abs",
        T_range: list = None,
        guess: list = None,
        bounds: tuple = None,
        maxfev: int = 100000,
):
    """Fit NASA/Shomate thermo parameters from a table without file I/O.

    By default this accepts the snake_case columns returned by ``scan_thermo``:
    ``temperature``, ``heat_capacity_cp``, ``enthalpy``, and ``entropy``. It
    also accepts the legacy Excel column names used by ``fit_thermo_model``.
    """
    df = pd.DataFrame(data_frame)
    T, Cp_T, H_T, S_T = _extract_thermo_fit_arrays(
        df,
        data_columns=data_columns,
        start_index=start_index,
        end_index=end_index,
    )
    if T_range is None:
        T_range = [float(np.min(T)), float(np.max(T))]

    model = _get_thermo_model_info(model_type)
    n_params = model["n_params"]
    if bounds is None:
        bounds = ([-np.inf] * n_params, [np.inf] * n_params)

    y_data = np.hstack([Cp_T, H_T, S_T])
    sigma = _build_fit_sigma(Cp_T, H_T, S_T, weight_strategy)
    popt, pcov = curve_fit(
        f=model["fit_func"],
        xdata=T,
        ydata=y_data,
        sigma=sigma,
        p0=guess,
        bounds=bounds,
        maxfev=maxfev,
    )

    fitted_model = model["model_class"](*popt, return_mode="all")
    Cp_T_pre, H_T_pre, S_T_pre = fitted_model(T)
    metrics = {
        "heat_capacity_cp": _fit_metrics(Cp_T, Cp_T_pre),
        "enthalpy": _fit_metrics(H_T, H_T_pre),
        "entropy": _fit_metrics(S_T, S_T_pre),
    }

    return ThermoFitResult(
        model_type=_canonical_model_type(model_type),
        parameters=tuple(float(parameter) for parameter in popt),
        temperature_range=tuple(float(value) for value in T_range),
        metrics=metrics,
        covariance=pcov,
    )



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
            f.write(f'{x:.5f}  {y:.5f}\n')
    f.close()
    return None


def fit_thermo_model(
        data_path: str,
        name: str,
        model_type: str = "NASA7",
        data_columns=None,
        start_index: int = 0,
        end_index: int = None,
        weight_strategy: str = "inverse_mean_abs",
        output_dir: str = ".",
        save_plots: bool = True,
        save_metrics: bool = True,
        write_yaml: bool = True,
        T_range: list = None,
        guess: list = None,
        bounds: tuple = None,
        maxfev: int = 100000,
):
    """
    Fits a thermodynamic model to experimental data and evaluates the fit.

    The function supports fitting different types of models (e.g., NASA7, NASA9, Shomate) to the
    thermodynamic data. It reads the data from an Excel file, fits the specified model,
    and optionally saves plots and metrics of the fit. The fitted parameters can be written
    to a Cantera YAML file for further use in chemical kinetic simulations.

    Parameters:
    - data_path (str):
            Path to the Excel file containing the thermodynamic data.
    - name (str):
            Name of the species or reaction for which the model is being fitted.
    - model_type (str, optional):
            Type of the thermodynamic model to fit. Default is "NASA7".
    - data_columns (dict, optional):
            Dictionary mapping column names in the Excel file to their corresponding
            thermodynamic properties. Default maps are provided if not specified.
    - start_index (int, optional):
            Index of the first row to include in the fit. Default is 0.
    - end_index (int, optional):
            Index of the last row to include in the fit. If None, all rows after
            `start_index` are included. Default is None.
    - weight_strategy (str, optional):
            Strategy for weighting the data points during the fit. Options are
            "inverse_mean_abs" and "uniform". Default is "inverse_mean_abs".
    - output_dir (str, optional):
            Directory where output files (plots, metrics, YAML) will be saved. Default is ".".
    - save_plots (bool, optional):
            Whether to save plots of the fitted data. Default is True.
    - save_metrics (bool, optional):
            Whether to save metrics of the fit. Default is True.
    - write_yaml (bool, optional):
            Whether to write the fitted parameters to a Cantera YAML file. Default is True.
    - T_range (list, optional):
            Temperature range [T_min, T_max] for the fitted model. If None, the range
            is determined from the data. Default is None.
    - guess (list, optional):
            Initial guess for the model parameters. If None, default guesses are used.
            Default is None.
    - bounds (tuple, optional):
            Bounds on the model parameters. If None, default bounds are used. Default is None. See scipy document.
    - maxfev (int, optional):
            Maximum number of function evaluations for the fitting process. Default is 100000.

    Returns:
    - popt (list):
            List of optimized parameters for the fitted model.
    - fitted_model (object):
            Fitted model object that can be used to evaluate the model at different temperatures.

    Raises:
    - ValueError: If an unsupported model type is specified.
    """
    # 确保输出目录存在
    if data_columns is None:
        data_columns = {
            "T": "T/K",
            "Cp": "Cp/(J/mol/K)",
            "H": "H/(J/mol)",
            "S": "S/(J/mol/K)",
        }
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    df = pd.read_excel(data_path)

    # 提取列数据
    T = df[data_columns["T"]].to_numpy()[start_index:end_index]
    T = np.array(T, dtype=float)
    H_T = df[data_columns["H"]].to_numpy()[start_index:end_index]
    S_T = df[data_columns["S"]].to_numpy()[start_index:end_index]
    Cp_T = df[data_columns["Cp"]].to_numpy()[start_index:end_index]

    # 确定温度范围
    if T_range is None:
        T_range = [float(T[0]), float(T[-1])]

    n_data = len(T)
    X = T
    Y = np.hstack([Cp_T, H_T, S_T])

    # 选择模型函数
    model_type = _canonical_model_type(model_type)
    model = _get_thermo_model_info(model_type)
    FUN = model["fit_func"]
    FUN_CLASS = model["model_class"]
    n_params = model["n_params"]

    # 设置权重
    if weight_strategy == "inverse_mean_abs":
        scale_cp = np.mean(np.abs(Cp_T))
        scale_h = np.mean(np.abs(H_T))
        scale_s = np.mean(np.abs(S_T))
        weight_cp = 1.0 / scale_cp
        weight_h = 1.0 / scale_h
        weight_s = 1.0 / scale_s
        weights = np.concatenate([
            np.full(n_data, weight_cp),
            np.full(n_data, weight_h),
            np.full(n_data, weight_s)
        ])
        SIGMA = 1 / weights
    else:  # uniform weighting
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
    fitted_model = FUN_CLASS(*popt, return_mode="all")

    # 计算预测值
    Cp_T_pre, H_T_pre, S_T_pre = fitted_model(T)

    # 评估模型
    if save_metrics:
        cal_metric(Cp_T, Cp_T_pre, "Cp_T", save=True, save_root_path=output_dir)
        cal_metric(H_T, H_T_pre, "H_T", save=True, save_root_path=output_dir)
        cal_metric(S_T, S_T_pre, "S_T", save=True, save_root_path=output_dir)

    # 绘制图表
    if save_plots:
        plot_fit(
            X, Cp_T, FUN_CLASS(*popt, return_mode="Cp_T"),
            "T", "Cp_T", "Cp_T", output_dir
        )
        plot_fit(
            X, H_T, FUN_CLASS(*popt, return_mode="H_T"),
            "T", "H_T", "H_T", output_dir
        )
        plot_fit(
            X, S_T, FUN_CLASS(*popt, return_mode="S_T"),
            "T", "S_T", "S_T", output_dir
        )

    # 输出Cantera YAML
    if write_yaml and model["yaml_writer"]:
        if model_type == "NASA7":
            model["yaml_writer"](
                name, T_range, [float(p) for p in popt], root_path=output_dir
            )
        elif model_type == "NASA9":
            model["yaml_writer"](
                name, T_range, [float(p) for p in popt], root_path=output_dir
            )
        elif model_type == 'Shomate':
            model["yaml_writer"](
                name, T_range, [float(p) for p in popt], root_path=output_dir
            )

    return popt, fitted_model


# if __name__ == "__main__":
#
#
#     fit_thermo_model(
#         data_path='QMthermoScan.xlsx',
#         name='S1',
#         model_type="NASA7",
#         start_index=0,
#         end_index=21,
#         output_dir="NASA7_results",
#         weight_strategy="inverse_mean_abs",
#     )

__all__ = [
    'NASA7',
    'NASA9',
    'Shomate',
    'cal_metric',
    'export_data',
    'fit',
    'ThermoFitResult',
    'fit_thermo_frame',
    'fit_thermo_model',
    'nasa7',
    'nasa7_for_fit',
    'nasa9',
    'nasa9_for_fit',
    'plot_fit',
    'shomate',
    'shomate_for_fit',
]
