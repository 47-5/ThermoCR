"""Build a complete Cantera YAML mechanism from modern export helpers."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ThermoCR.export import (
    format_cantera_mechanism_yaml,
    format_cantera_reaction_yaml,
    format_cantera_species_yaml,
    format_cantera_yaml_thermo,
)
from ThermoCR.kinetics import fit_kinetics_frame
from ThermoCR.thermo import fit_thermo_frame


EXAMPLE_DIR = ROOT / "example"
OUTPUT_DIR = ROOT / "examples" / "output"


def _fit_species_block(species_name, composition, thermo_scan_name):
    thermo_frame = pd.read_excel(EXAMPLE_DIR / thermo_scan_name)
    fit = fit_thermo_frame(
        thermo_frame,
        model_type="NASA7",
        weight_strategy="uniform",
    )
    composition_text = ", ".join(f"{element}: {count}" for element, count in composition.items())
    species_head = f"- name: {species_name}\n  composition: {{{composition_text}}}\n"
    thermo_block = format_cantera_yaml_thermo(
        fit.model_type,
        fit.temperature_range,
        fit.parameters,
    )
    return format_cantera_species_yaml(species_head, thermo_block)


def _fit_reaction_block():
    rates = pd.read_excel(EXAMPLE_DIR / "VTST_scan_2CPD_to_DCPD.xlsx")
    fit = fit_kinetics_frame(rates, model_type="Arrhenius")
    parameters = fit.named_parameters()
    return format_cantera_reaction_yaml(
        ["CPD", "CPD"],
        ["DCPD"],
        A=parameters["A"],
        b=parameters["b"],
        Ea=parameters["Ea"],
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cpd = _fit_species_block("CPD", {"C": 5, "H": 6}, "QMthermoScan_CPD.xlsx")
    dcpd = _fit_species_block("DCPD", {"C": 10, "H": 12}, "QMthermoScan_DCPD.xlsx")
    reaction = _fit_reaction_block()
    mechanism = format_cantera_mechanism_yaml(
        [cpd, dcpd],
        reaction_blocks=[reaction],
        phase_name="gas",
    )

    mechanism_yaml = OUTPUT_DIR / "CPD_DCPD_mechanism.yaml"
    mechanism_yaml.write_text(mechanism, encoding="utf-8")
    print(f"wrote: {mechanism_yaml}")


if __name__ == "__main__":
    main()
