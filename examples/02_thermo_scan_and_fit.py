"""Scan thermochemistry and fit a NASA7 model with the modern API."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ThermoCR.export import format_cantera_yaml_thermo
from ThermoCR.io import read_molecule_data
from ThermoCR.thermo import ThermoOptions, fit_thermo_frame, scan_thermo


EXAMPLE_DIR = ROOT / "example"
OUTPUT_DIR = ROOT / "examples" / "output"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    molecule = read_molecule_data(EXAMPLE_DIR / "CPD.out")
    temperatures = np.linspace(300.0, 1500.0, 16)
    thermo_frame = scan_thermo(
        molecule,
        temperatures=temperatures,
        pressure=100000.0,
        options=ThermoOptions(
            pressure=100000.0,
            rotational_symmetry_number=1,
        ),
    )
    thermo_csv = OUTPUT_DIR / "CPD_thermo_scan.csv"
    thermo_frame.to_csv(thermo_csv, index=False)

    fit = fit_thermo_frame(
        thermo_frame,
        model_type="NASA7",
        weight_strategy="uniform",
        T_range=(300.0, 1500.0),
    )
    thermo_yaml = OUTPUT_DIR / "CPD_thermo.yaml"
    thermo_yaml.write_text(
        format_cantera_yaml_thermo(
            fit.model_type,
            fit.temperature_range,
            fit.parameters,
        ),
        encoding="utf-8",
    )

    print(f"wrote: {thermo_csv}")
    print(f"wrote: {thermo_yaml}")
    print(f"NASA7 Cp R2: {fit.metrics['heat_capacity_cp']['r2']:.6f}")


if __name__ == "__main__":
    main()
