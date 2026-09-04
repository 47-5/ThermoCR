"""Versioned thermochemistry artifacts and energy-zero conventions."""

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
import re

import numpy as np

from ThermoCR.reference_state import validate_reference_pressure_pa


THERMO_ARTIFACT_SCHEMA_VERSION = "1.0"
_FORMATION_REFERENCE_TEMPERATURE_K = 298.15
_FORMATION_ENTHALPY_ANCHOR_ATOL_J_MOL = 1.0e-6
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_DEFAULT_TABLE_UNITS = {
    "temperature": "K",
    "heat_capacity_cp": "J/mol/K",
    "enthalpy": "J/mol",
    "entropy": "J/mol/K",
}


class EnergyConvention(str, Enum):
    """Energy-zero convention carried by a thermochemistry artifact."""

    FORMATION_ENTHALPY = "formation_enthalpy"
    ABSOLUTE_QM = "absolute_qm"
    THERMAL_INCREMENT_ONLY = "thermal_increment_only"


def _normalize_energy_convention(value):
    if isinstance(value, EnergyConvention):
        return value
    try:
        return EnergyConvention(str(value))
    except ValueError as exc:
        choices = ", ".join(item.value for item in EnergyConvention)
        raise ValueError(f"energy_convention must be one of: {choices}") from exc


def _finite_float(value, field_name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _json_mapping(value, field_name):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary")
    try:
        return json.loads(json.dumps(value, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain JSON-serializable values") from exc


def _normalize_composition(composition):
    if not isinstance(composition, dict) or not composition:
        raise ValueError("composition must be a nonempty dictionary")
    normalized = {}
    for element, amount in composition.items():
        element = str(element).strip()
        if not element:
            raise ValueError("composition element names must not be empty")
        numeric_amount = _finite_float(amount, f"composition[{element!r}]")
        integer_amount = int(numeric_amount)
        if numeric_amount <= 0.0 or numeric_amount != integer_amount:
            raise ValueError("composition amounts must be positive integers")
        normalized[element] = integer_amount
    return normalized


def _normalize_temperature_grid(values):
    if values is None:
        return ()
    grid = tuple(_finite_float(value, "temperature_grid_k") for value in values)
    if any(value <= 0.0 for value in grid):
        raise ValueError("temperature_grid_k must contain positive values")
    if any(right <= left for left, right in zip(grid, grid[1:])):
        raise ValueError("temperature_grid_k must be strictly increasing")
    return grid


@dataclass(frozen=True)
class SpeciesThermoArtifact:
    """JSON-serializable scientific contract for one species thermo model."""

    species_id: str
    cantera_name: str
    composition: dict
    energy_convention: EnergyConvention
    schema_version: str = THERMO_ARTIFACT_SCHEMA_VERSION
    canonical_smiles: str = None
    inchikey: str = None
    charge: int = 0
    multiplicity: int = 1
    phase: str = "gas"
    reference_temperature_k: float = _FORMATION_REFERENCE_TEMPERATURE_K
    reference_pressure_pa: float = 100000.0
    formation_enthalpy_298_j_mol: float = None
    formation_enthalpy_uncertainty_j_mol: float = None
    energy_reference_id: str = None
    qm_source_bundle: dict = field(default_factory=dict)
    thermo_options: dict = field(default_factory=dict)
    temperature_grid_k: tuple = field(default_factory=tuple)
    table_units: dict = field(default_factory=lambda: dict(_DEFAULT_TABLE_UNITS))
    fit: dict = field(default_factory=dict)
    fit_metrics: dict = field(default_factory=dict)
    continuity_metrics: dict = field(default_factory=dict)
    source_sha256: str = None
    thermocr_version: str = None
    git_commit: str = None
    dependency_versions: dict = field(default_factory=dict)
    cantera_species_dict: dict = field(default_factory=dict)

    def __post_init__(self):
        species_id = str(self.species_id).strip()
        cantera_name = str(self.cantera_name).strip()
        phase = str(self.phase).strip()
        if not species_id or not cantera_name:
            raise ValueError("species_id and cantera_name must not be empty")
        if not phase:
            raise ValueError("phase must not be empty")
        if str(self.schema_version) != THERMO_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported thermo artifact schema_version: {self.schema_version}"
            )

        convention = _normalize_energy_convention(self.energy_convention)
        reference_temperature = _finite_float(
            self.reference_temperature_k,
            "reference_temperature_k",
        )
        if reference_temperature <= 0.0:
            raise ValueError("reference_temperature_k must be positive")
        reference_pressure = validate_reference_pressure_pa(
            self.reference_pressure_pa
        )

        try:
            charge = int(self.charge)
            multiplicity = int(self.multiplicity)
        except (TypeError, ValueError) as exc:
            raise ValueError("charge and multiplicity must be integers") from exc
        if charge != self.charge or multiplicity != self.multiplicity:
            raise ValueError("charge and multiplicity must be integers")
        if multiplicity <= 0:
            raise ValueError("multiplicity must be positive")

        formation_enthalpy = self.formation_enthalpy_298_j_mol
        uncertainty = self.formation_enthalpy_uncertainty_j_mol
        energy_reference_id = self.energy_reference_id
        if convention is EnergyConvention.FORMATION_ENTHALPY:
            if formation_enthalpy is None:
                raise ValueError(
                    "formation_enthalpy_298_j_mol is required for "
                    "formation_enthalpy convention"
                )
            formation_enthalpy = _finite_float(
                formation_enthalpy,
                "formation_enthalpy_298_j_mol",
            )
            if not np.isclose(
                reference_temperature,
                _FORMATION_REFERENCE_TEMPERATURE_K,
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise ValueError(
                    "formation_enthalpy_298_j_mol requires "
                    "reference_temperature_k=298.15"
                )
            if energy_reference_id is not None:
                raise ValueError(
                    "energy_reference_id must be omitted for formation_enthalpy convention"
                )
        else:
            if formation_enthalpy is not None or uncertainty is not None:
                raise ValueError(
                    "formation enthalpy and uncertainty must be omitted for "
                    "nonformation energy conventions"
                )
            if convention is EnergyConvention.ABSOLUTE_QM:
                if energy_reference_id is None or not str(energy_reference_id).strip():
                    raise ValueError(
                        "energy_reference_id is required for absolute_qm convention"
                    )
                energy_reference_id = str(energy_reference_id).strip()
            elif energy_reference_id is not None:
                raise ValueError(
                    "energy_reference_id must be omitted for thermal_increment_only"
                )

        if uncertainty is not None:
            uncertainty = _finite_float(
                uncertainty,
                "formation_enthalpy_uncertainty_j_mol",
            )
            if uncertainty < 0.0:
                raise ValueError(
                    "formation_enthalpy_uncertainty_j_mol must be nonnegative"
                )

        table_units = _json_mapping(self.table_units, "table_units")
        for key, expected_unit in _DEFAULT_TABLE_UNITS.items():
            if table_units.get(key) != expected_unit:
                raise ValueError(f"table_units[{key!r}] must be {expected_unit!r}")

        source_sha256 = self.source_sha256
        if source_sha256 is not None and not _SHA256_PATTERN.fullmatch(
            str(source_sha256)
        ):
            raise ValueError("source_sha256 must contain 64 hexadecimal characters")

        object.__setattr__(self, "species_id", species_id)
        object.__setattr__(self, "cantera_name", cantera_name)
        object.__setattr__(self, "composition", _normalize_composition(self.composition))
        object.__setattr__(self, "energy_convention", convention)
        object.__setattr__(self, "schema_version", THERMO_ARTIFACT_SCHEMA_VERSION)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "charge", charge)
        object.__setattr__(self, "multiplicity", multiplicity)
        object.__setattr__(self, "reference_temperature_k", reference_temperature)
        object.__setattr__(self, "reference_pressure_pa", reference_pressure)
        object.__setattr__(self, "formation_enthalpy_298_j_mol", formation_enthalpy)
        object.__setattr__(self, "formation_enthalpy_uncertainty_j_mol", uncertainty)
        object.__setattr__(self, "energy_reference_id", energy_reference_id)
        object.__setattr__(self, "qm_source_bundle", _json_mapping(self.qm_source_bundle, "qm_source_bundle"))
        object.__setattr__(self, "thermo_options", _json_mapping(self.thermo_options, "thermo_options"))
        object.__setattr__(self, "temperature_grid_k", _normalize_temperature_grid(self.temperature_grid_k))
        object.__setattr__(self, "table_units", table_units)
        object.__setattr__(self, "fit", _json_mapping(self.fit, "fit"))
        object.__setattr__(self, "fit_metrics", _json_mapping(self.fit_metrics, "fit_metrics"))
        object.__setattr__(self, "continuity_metrics", _json_mapping(self.continuity_metrics, "continuity_metrics"))
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "dependency_versions", _json_mapping(self.dependency_versions, "dependency_versions"))
        object.__setattr__(self, "cantera_species_dict", _json_mapping(self.cantera_species_dict, "cantera_species_dict"))

    def to_dict(self):
        """Return a JSON-ready representation of the artifact."""
        return {
            "schema_version": self.schema_version,
            "species_id": self.species_id,
            "cantera_name": self.cantera_name,
            "canonical_smiles": self.canonical_smiles,
            "inchikey": self.inchikey,
            "composition": dict(self.composition),
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "phase": self.phase,
            "energy_convention": self.energy_convention.value,
            "reference_temperature_k": self.reference_temperature_k,
            "reference_pressure_pa": self.reference_pressure_pa,
            "formation_enthalpy_298_j_mol": self.formation_enthalpy_298_j_mol,
            "formation_enthalpy_uncertainty_j_mol": self.formation_enthalpy_uncertainty_j_mol,
            "energy_reference_id": self.energy_reference_id,
            "qm_source_bundle": self.qm_source_bundle,
            "thermo_options": self.thermo_options,
            "temperature_grid_k": list(self.temperature_grid_k),
            "table_units": self.table_units,
            "fit": self.fit,
            "fit_metrics": self.fit_metrics,
            "continuity_metrics": self.continuity_metrics,
            "source_sha256": self.source_sha256,
            "thermocr_version": self.thermocr_version,
            "git_commit": self.git_commit,
            "dependency_versions": self.dependency_versions,
            "cantera_species_dict": self.cantera_species_dict,
        }

    @classmethod
    def from_dict(cls, data):
        """Construct an artifact from a JSON-ready dictionary."""
        if not isinstance(data, dict):
            raise ValueError("thermo artifact data must be a dictionary")
        return cls(**dict(data))

    @classmethod
    def from_fit_result(
        cls,
        *,
        species_id,
        cantera_name,
        composition,
        energy_convention,
        fit_result,
        formation_enthalpy_298_j_mol=None,
        formation_enthalpy_uncertainty_j_mol=None,
        energy_reference_id=None,
        temperature_grid_k=(),
        **metadata,
    ):
        """Build an artifact from a validated ``ThermoFitResult``."""
        from ThermoCR.thermo.fitting import ThermoFitResult

        if not isinstance(fit_result, ThermoFitResult):
            raise TypeError("fit_result must be a ThermoFitResult instance")
        reserved = {
            "reference_pressure_pa",
            "fit",
            "fit_metrics",
        }.intersection(metadata)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"metadata cannot override fit-derived fields: {names}")

        covariance = None
        if fit_result.covariance is not None:
            covariance = np.asarray(fit_result.covariance, dtype=float).tolist()
        fit = {
            "model": fit_result.model_type,
            "regions": [list(fit_result.temperature_range)],
            "coefficients": [list(fit_result.parameters)],
            "covariance": covariance,
        }
        return cls(
            species_id=species_id,
            cantera_name=cantera_name,
            composition=composition,
            energy_convention=energy_convention,
            reference_pressure_pa=fit_result.reference_pressure_pa,
            formation_enthalpy_298_j_mol=formation_enthalpy_298_j_mol,
            formation_enthalpy_uncertainty_j_mol=(
                formation_enthalpy_uncertainty_j_mol
            ),
            energy_reference_id=energy_reference_id,
            temperature_grid_k=temperature_grid_k,
            fit=fit,
            fit_metrics=fit_result.metrics,
            **metadata,
        )

    @classmethod
    def from_continuous_nasa7_fit(
        cls,
        *,
        species_id,
        cantera_name,
        composition,
        energy_convention,
        fit_result,
        formation_enthalpy_298_j_mol=None,
        formation_enthalpy_uncertainty_j_mol=None,
        energy_reference_id=None,
        temperature_grid_k=(),
        **metadata,
    ):
        """Build an artifact from a validated two-region NASA7 fit."""
        from ThermoCR.thermo.continuous_nasa7 import ContinuousNASA7Fit

        if not isinstance(fit_result, ContinuousNASA7Fit):
            raise TypeError(
                "fit_result must be a ContinuousNASA7Fit instance"
            )
        reserved = {
            "reference_pressure_pa",
            "reference_temperature_k",
            "fit",
            "fit_metrics",
            "continuity_metrics",
            "cantera_species_dict",
        }.intersection(metadata)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(
                f"metadata cannot override fit-derived fields: {names}"
            )

        convention = _normalize_energy_convention(energy_convention)
        if (
            convention is EnergyConvention.FORMATION_ENTHALPY
            and formation_enthalpy_298_j_mol is not None
        ):
            formation_enthalpy = _finite_float(
                formation_enthalpy_298_j_mol,
                "formation_enthalpy_298_j_mol",
            )
            anchor_enthalpy = _finite_float(
                fit_result.anchor_enthalpy_j_mol,
                "fit_result.anchor_enthalpy_j_mol",
            )
            if not np.isclose(
                formation_enthalpy,
                anchor_enthalpy,
                rtol=0.0,
                atol=_FORMATION_ENTHALPY_ANCHOR_ATOL_J_MOL,
            ):
                raise ValueError(
                    "formation_enthalpy_298_j_mol must match "
                    "fit_result.anchor_enthalpy_j_mol"
                )

        lower, upper = fit_result.temperature_range_k
        midpoint = fit_result.midpoint_temperature_k
        reference_pressure = fit_result.reference_pressure_pa
        normalized_composition = _normalize_composition(composition)
        fit = {
            "model": "NASA7",
            "regions": [
                [lower, midpoint],
                [midpoint, upper],
            ],
            "coefficients": [
                list(fit_result.low_coefficients),
                list(fit_result.high_coefficients),
            ],
            "anchor_temperature_k": fit_result.anchor_temperature_k,
            "anchor_enthalpy_j_mol": fit_result.anchor_enthalpy_j_mol,
            "anchor_entropy_j_mol_k": fit_result.anchor_entropy_j_mol_k,
            "cp_fit_point_count": fit_result.cp_fit_point_count,
            "diagnostics": fit_result.diagnostics,
        }
        cantera_species_dict = {
            "name": str(cantera_name).strip(),
            "composition": normalized_composition,
            "thermo": {
                "model": "NASA7",
                "reference-pressure": f"{reference_pressure:.17g} Pa",
                "temperature-ranges": [lower, midpoint, upper],
                "data": [
                    list(fit_result.low_coefficients),
                    list(fit_result.high_coefficients),
                ],
            },
        }
        return cls(
            species_id=species_id,
            cantera_name=cantera_name,
            composition=normalized_composition,
            energy_convention=energy_convention,
            reference_temperature_k=fit_result.anchor_temperature_k,
            reference_pressure_pa=reference_pressure,
            formation_enthalpy_298_j_mol=formation_enthalpy_298_j_mol,
            formation_enthalpy_uncertainty_j_mol=(
                formation_enthalpy_uncertainty_j_mol
            ),
            energy_reference_id=energy_reference_id,
            temperature_grid_k=temperature_grid_k,
            fit=fit,
            fit_metrics=fit_result.metrics,
            continuity_metrics=fit_result.continuity,
            cantera_species_dict=cantera_species_dict,
            **metadata,
        )

    def write_json(self, path):
        """Write the artifact as a deterministic UTF-8 JSON sidecar."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def read_json(cls, path):
        """Read and validate an artifact JSON sidecar."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def assert_energy_conventions_compatible(
    artifacts,
    require_equilibrium_ready=False,
):
    """Raise when artifacts cannot safely share one thermo phase."""
    artifacts = tuple(artifacts)
    if not artifacts:
        raise ValueError("at least one SpeciesThermoArtifact is required")
    if not all(isinstance(item, SpeciesThermoArtifact) for item in artifacts):
        raise TypeError("all items must be SpeciesThermoArtifact instances")

    conventions = {item.energy_convention for item in artifacts}
    if len(conventions) != 1:
        values = ", ".join(sorted(item.value for item in conventions))
        raise ValueError(f"incompatible energy conventions: {values}")
    convention = next(iter(conventions))

    first = artifacts[0]
    for item in artifacts[1:]:
        if item.phase != first.phase:
            raise ValueError("artifacts use incompatible phase labels")
        if not np.isclose(
            item.reference_temperature_k,
            first.reference_temperature_k,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError("artifacts use incompatible reference temperatures")
        if not np.isclose(
            item.reference_pressure_pa,
            first.reference_pressure_pa,
            rtol=1.0e-12,
            atol=1.0e-9,
        ):
            raise ValueError("artifacts use incompatible reference pressures")

    if convention is EnergyConvention.ABSOLUTE_QM:
        reference_ids = {item.energy_reference_id for item in artifacts}
        if len(reference_ids) != 1:
            raise ValueError("absolute_qm artifacts use different energy_reference_id values")
    if (
        convention is EnergyConvention.THERMAL_INCREMENT_ONLY
        and require_equilibrium_ready
    ):
        raise ValueError(
            "thermal_increment_only artifacts cannot be used for equilibrium"
        )
    return None


__all__ = [
    "EnergyConvention",
    "SpeciesThermoArtifact",
    "THERMO_ARTIFACT_SCHEMA_VERSION",
    "assert_energy_conventions_compatible",
]
