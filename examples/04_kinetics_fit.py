"""Fit Arrhenius kinetics from a rate scan."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ThermoCR.export import format_cantera_reaction_yaml
from ThermoCR.kinetics import calculate_vtst_rate_frame, fit_kinetics_frame


EXAMPLE_DIR = ROOT / "example"
OUTPUT_DIR = ROOT / "examples" / "output"


def _load_or_create_vtst_rates():
    existing = OUTPUT_DIR / "VTST_2CPD_to_DCPD.csv"
    if existing.exists():
        return pd.read_csv(existing)

    reactant = pd.read_excel(EXAMPLE_DIR / "QMthermoScan_CPD.xlsx")
    path1 = pd.read_excel(EXAMPLE_DIR / "QMthermoScan_01_02_path1_1.xlsx")
    path2 = pd.read_excel(EXAMPLE_DIR / "QMthermoScan_01_02_path2_1.xlsx")
    return calculate_vtst_rate_frame(
        [path1, path2],
        [reactant, reactant],
        path_names=["irc_path1", "irc_path2"],
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rates = _load_or_create_vtst_rates()
    fit = fit_kinetics_frame(rates, model_type="Arrhenius")
    parameters = fit.named_parameters()

    reaction_yaml = OUTPUT_DIR / "2CPD_to_DCPD_reaction.yaml"
    reaction_yaml.write_text(
        format_cantera_reaction_yaml(
            ["CPD", "CPD"],
            ["DCPD"],
            A=parameters["A"],
            b=parameters["b"],
            Ea=parameters["Ea"],
        ),
        encoding="utf-8",
    )

    print(f"wrote: {reaction_yaml}")
    print(f"Arrhenius A: {parameters['A']:.6e}")
    print(f"Arrhenius b: {parameters['b']:.6f}")
    print(f"Arrhenius Ea / J mol-1: {parameters['Ea']:.6f}")


if __name__ == "__main__":
    main()
