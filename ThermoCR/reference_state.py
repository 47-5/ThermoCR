"""Reference-state validation shared by thermo fitting and Cantera export."""

import math
import warnings


DEFAULT_REFERENCE_PRESSURE_PA = 101325.0
PA_PER_BAR = 100000.0


def validate_reference_pressure_pa(value):
    """Return a validated finite, positive reference pressure in Pa."""
    try:
        pressure_pa = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("reference pressure must be a finite positive number") from exc
    if not math.isfinite(pressure_pa) or pressure_pa <= 0.0:
        raise ValueError("reference pressure must be a finite positive number")
    return pressure_pa


def resolve_export_reference_pressure_pa(
    reference_p=None,
    reference_pressure_pa=None,
    *,
    warning_stacklevel=2,
):
    """Resolve the canonical Pa value while supporting legacy bar input."""
    if reference_p is not None and reference_pressure_pa is not None:
        raise ValueError(
            "reference pressure is ambiguous; provide reference_pressure_pa "
            "or the legacy reference_p argument, not both"
        )
    if reference_pressure_pa is not None:
        return validate_reference_pressure_pa(reference_pressure_pa)
    if reference_p is not None:
        warnings.warn(
            "reference_p is deprecated and is interpreted in bar; use "
            "reference_pressure_pa with an explicit pressure in Pa instead",
            DeprecationWarning,
            stacklevel=warning_stacklevel,
        )
        pressure_pa = validate_reference_pressure_pa(reference_p) * PA_PER_BAR
        return validate_reference_pressure_pa(pressure_pa)
    return DEFAULT_REFERENCE_PRESSURE_PA


__all__ = [
    "DEFAULT_REFERENCE_PRESSURE_PA",
    "PA_PER_BAR",
    "resolve_export_reference_pressure_pa",
    "validate_reference_pressure_pa",
]
