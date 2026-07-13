"""Deterministic continuous two-region NASA7 fitting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


# Cantera's molar gas constant in J/(mol K). Keeping the fitting and export
# convention identical avoids a systematic coefficient interpretation offset.
GAS_CONSTANT_J_MOL_K = 8.31446261815324
_TEMPERATURE_SCALE_K = 1000.0


@dataclass(frozen=True)
class ContinuousNASA7Fit:
    """A two-region NASA7 fit with exact Cp, H, and S continuity."""

    temperature_range_k: tuple[float, float]
    midpoint_temperature_k: float
    anchor_temperature_k: float
    anchor_enthalpy_j_mol: float
    anchor_entropy_j_mol_k: float
    cp_fit_point_count: int
    low_coefficients: tuple[float, ...]
    high_coefficients: tuple[float, ...]
    metrics: dict
    continuity: dict
    diagnostics: dict

    def predict(self, temperatures):
        """Evaluate Cp, H, and S using Cantera's midpoint convention."""

        temperature = _temperature_array(temperatures)
        lower, upper = self.temperature_range_k
        if np.any(temperature < lower) or np.any(temperature > upper):
            raise ValueError(
                f"NASA7 fit is valid only within [{lower:g}, {upper:g}] K"
            )
        return _predict_piecewise(
            temperature,
            self.midpoint_temperature_k,
            self.low_coefficients,
            self.high_coefficients,
        )

    def to_dict(self):
        """Return a JSON-serializable representation."""

        return {
            "temperature_range_k": list(self.temperature_range_k),
            "midpoint_temperature_k": self.midpoint_temperature_k,
            "anchor_temperature_k": self.anchor_temperature_k,
            "anchor_enthalpy_j_mol": self.anchor_enthalpy_j_mol,
            "anchor_entropy_j_mol_k": self.anchor_entropy_j_mol_k,
            "cp_fit_point_count": self.cp_fit_point_count,
            "low_coefficients": list(self.low_coefficients),
            "high_coefficients": list(self.high_coefficients),
            "metrics": self.metrics,
            "continuity": self.continuity,
            "diagnostics": self.diagnostics,
        }


def fit_continuous_nasa7(
    temperatures,
    heat_capacities,
    enthalpies,
    entropies,
    *,
    midpoint_temperature_k,
    anchor_temperature_k=298.15,
    cp_fit_mask=None,
):
    """Fit two NASA7 regions from Cp with exact Cp/H/S continuity.

    The five Cp coefficients in each region are fit by uniform linear least
    squares under one equality constraint enforcing Cp continuity at the
    midpoint. The low-region H and S integration constants are fixed by the
    requested anchor. The high-region constants are fixed by H and S
    continuity at the midpoint.
    """

    temperature, cp, enthalpy, entropy, fit_mask = _validated_fit_arrays(
        temperatures,
        heat_capacities,
        enthalpies,
        entropies,
        cp_fit_mask=cp_fit_mask,
    )
    midpoint = _finite_positive(
        midpoint_temperature_k,
        "midpoint_temperature_k",
    )
    anchor = _finite_positive(anchor_temperature_k, "anchor_temperature_k")
    if not temperature[0] < midpoint < temperature[-1]:
        raise ValueError("midpoint_temperature_k must lie inside the fit range")

    low_mask = temperature <= midpoint
    fit_low_mask = fit_mask & low_mask
    fit_high_mask = fit_mask & ~low_mask
    if np.count_nonzero(fit_low_mask) < 5 or np.count_nonzero(fit_high_mask) < 5:
        raise ValueError("each NASA7 region requires at least five temperature points")

    anchor_indices = np.flatnonzero(
        np.isclose(temperature, anchor, rtol=0.0, atol=1.0e-9)
    )
    if len(anchor_indices) != 1:
        raise ValueError("the fit table must contain anchor_temperature_k exactly once")
    anchor_index = int(anchor_indices[0])
    if anchor > midpoint:
        raise ValueError("anchor_temperature_k must lie in the low NASA7 region")

    fit_temperature = temperature[fit_mask]
    fit_cp = cp[fit_mask]
    scaled_temperature = fit_temperature / _TEMPERATURE_SCALE_K
    fit_low = fit_temperature <= midpoint
    design = np.zeros((len(fit_temperature), 10), dtype=float)
    design[fit_low, :5] = _cp_design(scaled_temperature[fit_low])
    design[~fit_low, 5:] = _cp_design(scaled_temperature[~fit_low])

    scaled_midpoint = midpoint / _TEMPERATURE_SCALE_K
    midpoint_basis = np.array(
        [scaled_midpoint**power for power in range(5)],
        dtype=float,
    )
    # Eliminate the Cp-continuity equality analytically. Solving the reduced
    # least-squares problem avoids squaring the condition number through normal
    # equations and keeps the constrained fit deterministic.
    null_space = np.zeros((10, 9), dtype=float)
    null_space[:5, :5] = np.eye(5)
    null_space[5, :5] = midpoint_basis
    null_space[5, 5:] = -midpoint_basis[1:]
    null_space[6:, 5:] = np.eye(4)
    reduced_design = design @ null_space
    reduced_coefficients, _, rank, singular_values = np.linalg.lstsq(
        reduced_design,
        fit_cp / GAS_CONSTANT_J_MOL_K,
        rcond=None,
    )
    if rank != reduced_design.shape[1]:
        raise ValueError("NASA7 constrained Cp fit is rank deficient")

    scaled_coefficients = null_space @ reduced_coefficients
    low = _unscale_cp_coefficients(scaled_coefficients[:5])
    high = _unscale_cp_coefficients(scaled_coefficients[5:])
    anchor_enthalpy = float(enthalpy[anchor_index])
    anchor_entropy = float(entropy[anchor_index])
    low[5] = _enthalpy_integration_constant(
        low[:5],
        anchor,
        anchor_enthalpy,
    )
    low[6] = _entropy_integration_constant(
        low[:5],
        anchor,
        anchor_entropy,
    )
    _, midpoint_enthalpy, midpoint_entropy = nasa7_values([midpoint], low)
    high[5] = _enthalpy_integration_constant(
        high[:5],
        midpoint,
        float(midpoint_enthalpy[0]),
    )
    high[6] = _entropy_integration_constant(
        high[:5],
        midpoint,
        float(midpoint_entropy[0]),
    )

    low_values = nasa7_values([midpoint], low)
    high_values = nasa7_values([midpoint], high)
    continuity = {
        "cp_jump_j_mol_k": float(high_values[0][0] - low_values[0][0]),
        "h_jump_j_mol": float(high_values[1][0] - low_values[1][0]),
        "s_jump_j_mol_k": float(high_values[2][0] - low_values[2][0]),
    }
    predicted_cp, predicted_h, predicted_s = _predict_piecewise(
        temperature,
        midpoint,
        low,
        high,
    )
    scaled_residual = (
        reduced_design @ reduced_coefficients
        - fit_cp / GAS_CONSTANT_J_MOL_K
    )
    return ContinuousNASA7Fit(
        temperature_range_k=(float(temperature[0]), float(temperature[-1])),
        midpoint_temperature_k=midpoint,
        anchor_temperature_k=anchor,
        anchor_enthalpy_j_mol=anchor_enthalpy,
        anchor_entropy_j_mol_k=anchor_entropy,
        cp_fit_point_count=int(np.count_nonzero(fit_mask)),
        low_coefficients=tuple(float(value) for value in low),
        high_coefficients=tuple(float(value) for value in high),
        metrics={
            "heat_capacity_cp": _error_metrics(cp, predicted_cp),
            "enthalpy": _error_metrics(enthalpy, predicted_h),
            "entropy": _error_metrics(entropy, predicted_s),
        },
        continuity=continuity,
        diagnostics={
            "algorithm": "analytic_constraint_elimination_numpy_lstsq",
            "reduced_design_rank": int(rank),
            "reduced_design_columns": int(reduced_design.shape[1]),
            "reduced_design_condition_number": float(
                singular_values[0] / singular_values[-1]
            ),
            "cp_fit_rmse_j_mol_k": float(
                np.sqrt(np.mean(scaled_residual**2))
                * GAS_CONSTANT_J_MOL_K
            ),
        },
    )


def nasa7_values(temperatures, coefficients):
    """Evaluate one NASA7 region in J, mol, and K units."""

    temperature = _temperature_array(temperatures)
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (7,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("NASA7 coefficients must contain seven finite values")
    a1, a2, a3, a4, a5, a6, a7 = coefficients
    cp = GAS_CONSTANT_J_MOL_K * (
        a1
        + a2 * temperature
        + a3 * temperature**2
        + a4 * temperature**3
        + a5 * temperature**4
    )
    enthalpy = GAS_CONSTANT_J_MOL_K * temperature * (
        a1
        + a2 * temperature / 2.0
        + a3 * temperature**2 / 3.0
        + a4 * temperature**3 / 4.0
        + a5 * temperature**4 / 5.0
        + a6 / temperature
    )
    entropy = GAS_CONSTANT_J_MOL_K * (
        a1 * np.log(temperature)
        + a2 * temperature
        + a3 * temperature**2 / 2.0
        + a4 * temperature**3 / 3.0
        + a5 * temperature**4 / 4.0
        + a7
    )
    return cp, enthalpy, entropy


def _predict_piecewise(temperature, midpoint, low, high):
    low_mask = temperature <= midpoint
    outputs = [np.empty_like(temperature) for _ in range(3)]
    for mask, coefficients in ((low_mask, low), (~low_mask, high)):
        if not np.any(mask):
            continue
        values = nasa7_values(temperature[mask], coefficients)
        for output, property_values in zip(outputs, values):
            output[mask] = property_values
    return tuple(outputs)


def _validated_fit_arrays(temperatures, *properties, cp_fit_mask=None):
    arrays = [_temperature_array(temperatures)]
    for values in properties:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError(
                "thermodynamic property arrays must be finite and one-dimensional"
            )
        arrays.append(array)
    if any(len(array) != len(arrays[0]) for array in arrays[1:]):
        raise ValueError("thermodynamic fit arrays must have equal lengths")

    if cp_fit_mask is None:
        fit_mask = np.ones(len(arrays[0]), dtype=bool)
    else:
        fit_mask = np.asarray(cp_fit_mask)
        if fit_mask.ndim != 1 or len(fit_mask) != len(arrays[0]):
            raise ValueError("cp_fit_mask must match the thermodynamic table")
        if fit_mask.dtype != np.bool_:
            raise ValueError("cp_fit_mask must contain boolean values")

    order = np.argsort(arrays[0])
    arrays = [array[order] for array in arrays]
    fit_mask = fit_mask[order]
    if np.any(np.diff(arrays[0]) <= 0.0):
        raise ValueError("fit temperatures must be unique")
    return (*arrays, fit_mask)


def _temperature_array(values):
    temperature = np.asarray(values, dtype=float)
    if temperature.ndim == 0:
        temperature = temperature.reshape(1)
    if (
        temperature.ndim != 1
        or not np.all(np.isfinite(temperature))
        or np.any(temperature <= 0.0)
    ):
        raise ValueError("temperatures must be finite, positive, and one-dimensional")
    return temperature


def _finite_positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _cp_design(scaled_temperature):
    return np.column_stack(
        [scaled_temperature**power for power in range(5)]
    )


def _unscale_cp_coefficients(values):
    coefficients = np.zeros(7, dtype=float)
    for power, value in enumerate(values):
        coefficients[power] = value / _TEMPERATURE_SCALE_K**power
    return coefficients


def _enthalpy_integration_constant(cp_coefficients, temperature, enthalpy):
    integral = sum(
        coefficient * temperature ** (power + 1) / (power + 1)
        for power, coefficient in enumerate(cp_coefficients)
    )
    return enthalpy / GAS_CONSTANT_J_MOL_K - integral


def _entropy_integration_constant(cp_coefficients, temperature, entropy):
    terms = cp_coefficients[0] * math.log(temperature)
    terms += sum(
        coefficient * temperature**power / power
        for power, coefficient in enumerate(cp_coefficients[1:], start=1)
    )
    return entropy / GAS_CONSTANT_J_MOL_K - terms


def _error_metrics(reference, predicted):
    error = np.asarray(predicted) - np.asarray(reference)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
    }


__all__ = [
    "ContinuousNASA7Fit",
    "GAS_CONSTANT_J_MOL_K",
    "fit_continuous_nasa7",
    "nasa7_values",
]
