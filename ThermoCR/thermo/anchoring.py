"""Reference-state anchoring for thermochemical enthalpy curves."""

import numpy as np


def anchor_enthalpy_curve(
    temperatures,
    absolute_enthalpies,
    target_hf298,
    reference_temperature=298.15,
):
    """Anchor an absolute enthalpy curve to a formation enthalpy in J/mol.

    The returned values preserve every enthalpy increment from the input curve
    while setting the linearly interpolated value at ``reference_temperature``
    to ``target_hf298``. Extrapolation is intentionally rejected.
    """
    temperature = np.asarray(temperatures, dtype=float)
    enthalpy = np.asarray(absolute_enthalpies, dtype=float)
    if temperature.ndim != 1 or enthalpy.ndim != 1:
        raise ValueError("temperatures and absolute_enthalpies must be one-dimensional")
    if temperature.shape != enthalpy.shape or temperature.size == 0:
        raise ValueError(
            "temperatures and absolute_enthalpies must have the same nonzero length"
        )
    if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperatures must be finite and positive")
    if not np.all(np.isfinite(enthalpy)):
        raise ValueError("absolute_enthalpies must contain only finite values")

    reference_temperature = float(reference_temperature)
    target_hf298 = float(target_hf298)
    if not np.isfinite(reference_temperature) or reference_temperature <= 0.0:
        raise ValueError("reference_temperature must be finite and positive")
    if not np.isfinite(target_hf298):
        raise ValueError("target_hf298 must be finite")

    order = np.argsort(temperature)
    sorted_temperature = temperature[order]
    sorted_enthalpy = enthalpy[order]
    if np.any(np.diff(sorted_temperature) <= 0.0):
        raise ValueError("temperatures must be unique")
    if not sorted_temperature[0] <= reference_temperature <= sorted_temperature[-1]:
        raise ValueError("reference_temperature must lie inside the temperature range")

    reference_enthalpy = np.interp(
        reference_temperature,
        sorted_temperature,
        sorted_enthalpy,
    )
    return enthalpy - reference_enthalpy + target_hf298


__all__ = ["anchor_enthalpy_curve"]
