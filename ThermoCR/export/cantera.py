"""Cantera YAML export helpers for ThermoCR."""

from collections import Counter
from pathlib import Path

from ThermoCR.io import read_qm_output
from ThermoCR.constants import atomic_number_map

au_to_kcal_per_mol = 627.51
au_to_kJ_per_mol = 2625.5


def _output_path(root_path, filename):
    return Path(root_path) / filename


def format_cantera_yaml_thermo(model_type, T_range, parameters, reference_p=None):
    """Return a Cantera YAML thermo block for fitted parameters."""
    model_names = {
        "nasa7": "NASA7",
        "nasa9": "NASA9",
        "shomate": "Shomate",
    }
    model = model_names.get(str(model_type).lower())
    if model is None:
        raise ValueError(f"unsupported thermo model type: {model_type}")

    lines = [
        "  thermo:",
        f"   model: {model}",
        f"   temperature-ranges: {list(T_range)}",
    ]
    if reference_p is not None:
        lines.append(f"   reference-pressure: {reference_p} bar")
    lines.extend([
        "   data:",
        f"   - {list(parameters)}",
    ])
    return "\n".join(lines) + "\n"


def write_cantera_yaml_thermo_piecewise_Gibbs(
    specie_name,
    T=None,
    H_formation=None,
    G_formation=None,
    root_path=".",
):
    """Write piecewise-Gibbs thermodynamic data in Cantera YAML format."""
    T298_index = T.tolist().index(298.15)
    data = {str(t): str(g) for t, g in zip(T, G_formation)}

    yaml_path = _output_path(root_path, f"{specie_name}_thermo.yaml")
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write("  thermo:\n")
        f.write("   model: piecewise-Gibbs\n")
        f.write(f"   h0: {H_formation[T298_index]} kJ/mol\n")
        f.write("   dimensionless: False\n")
        f.write(f"   data: {data}")
    return None


def write_cantera_yaml_thermo_NASA7(specie_name, T_range, nasa7_parameters, root_path="."):
    """Write NASA7 thermodynamic data in Cantera YAML format."""
    yaml_path = _output_path(root_path, f"{specie_name}_thermo.yaml")
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(format_cantera_yaml_thermo("NASA7", T_range, nasa7_parameters))
    return None


def write_cantera_yaml_thermo_NASA9(
    specie_name,
    T_range,
    nasa9_parameters,
    reference_p=1,
    root_path=".",
):
    """Write NASA9 thermodynamic data in Cantera YAML format."""
    yaml_path = _output_path(root_path, f"{specie_name}_thermo.yaml")
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(format_cantera_yaml_thermo(
            "NASA9",
            T_range,
            nasa9_parameters,
            reference_p=reference_p,
        ))
    return None


def write_cantera_yaml_thermo_Shomate(
    specie_name,
    T_range,
    Shomate_parameters,
    reference_p=1,
    root_path=".",
):
    """Write Shomate thermodynamic data in Cantera YAML format."""
    yaml_path = _output_path(root_path, f"{specie_name}_thermo.yaml")
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(format_cantera_yaml_thermo(
            "Shomate",
            T_range,
            Shomate_parameters,
            reference_p=reference_p,
        ))
    return None


def format_cantera_reaction_yaml(
    r_name_list,
    p_name_list,
    A,
    b,
    Ea,
    reversible=True,
    convert_A_unit_fun=None,
):
    """Return one elementary reaction entry in Cantera YAML format."""
    left = " + ".join(r_name_list)
    middle = "<=>" if reversible else "=>"
    right = " + ".join(p_name_list)
    if convert_A_unit_fun is not None:
        A = convert_A_unit_fun(A)
    return (
        f"- equation: {left} {middle} {right}\n"
        "  type: elementary\n"
        f"  rate-constant: {{A: {A}, b: {b}, Ea: {Ea} }}\n"
    )


def make_cantera_reaction_yaml(
    r_name_list,
    p_name_list,
    A,
    b,
    Ea,
    reversible=True,
    yaml_name="reaction.yaml",
    write_mode="a",
    root_path=".",
    convert_A_unit_fun=None,
):
    """Write one elementary reaction entry in Cantera YAML format."""
    yaml_path = _output_path(root_path, yaml_name)
    with yaml_path.open(write_mode, encoding="utf-8") as f:
        f.write(format_cantera_reaction_yaml(
            r_name_list,
            p_name_list,
            A,
            b,
            Ea,
            reversible=reversible,
            convert_A_unit_fun=convert_A_unit_fun,
        ))
    return None


def make_cantera_specie_name_yaml(
    specie_name,
    composition_dict=None,
    read_file_path=None,
    root_path=".",
):
    """Write a Cantera species header from a composition dict or QM output."""
    yaml_path = _output_path(root_path, f"{specie_name}_head.yaml")

    if read_file_path is not None:
        data = read_qm_output(read_file_path)
        count_dict = Counter(data.atomnos)
        composition_dict = {
            atomic_number_map[atomic_number - 1]: count
            for atomic_number, count in count_dict.items()
        }

    if composition_dict is None:
        raise ValueError("composition_dict or read_file_path must be provided")

    formatted_string = "{" + ", ".join(
        f"{element}:{count}" for element, count in composition_dict.items()
    ) + "}"

    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(f"- name: {specie_name}\n")
        f.write(f"  composition: {formatted_string}\n")
    return None