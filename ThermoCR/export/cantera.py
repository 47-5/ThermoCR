"""Cantera YAML export helpers for ThermoCR."""

from collections import Counter
from pathlib import Path
import re

from ThermoCR.io import read_qm_output
from ThermoCR.constants import atomic_number_map

au_to_kcal_per_mol = 627.51
au_to_kJ_per_mol = 2625.5

__all__ = [
    "au_to_kJ_per_mol",
    "au_to_kcal_per_mol",
    "format_cantera_mechanism_yaml",
    "format_cantera_reaction_yaml",
    "format_cantera_species_yaml",
    "format_cantera_yaml_thermo",
    "make_cantera_mechanism_yaml",
    "make_cantera_reaction_yaml",
    "make_cantera_specie_name_yaml",
    "write_cantera_yaml_thermo_NASA7",
    "write_cantera_yaml_thermo_NASA9",
    "write_cantera_yaml_thermo_Shomate",
    "write_cantera_yaml_thermo_piecewise_Gibbs",
]


def _output_path(root_path, filename):
    return Path(root_path) / filename


def _clean_yaml_fragment(text):
    lines = [line.rstrip() for line in str(text).strip().splitlines()]
    return _normalize_composition_flow_mapping(
        "\n".join(line for line in lines if line.strip())
    )


def _normalize_composition_flow_mapping(text):
    def replace_match(match):
        items = []
        for item in match.group(2).split(","):
            if ":" not in item:
                items.append(item.strip())
                continue
            element, count = item.split(":", 1)
            items.append(f"{element.strip()}: {count.strip()}")
        return f"{match.group(1)}{{{', '.join(items)}}}"

    return re.sub(r"(composition:\s*)\{([^}]*)\}", replace_match, text)


def _flow_list(values):
    return "[" + ", ".join(str(value) for value in values) + "]"


def _species_name_from_block(species_block):
    match = re.search(r"^\s*-\s*name:\s*(.+?)\s*$", species_block, re.MULTILINE)
    if match is None:
        raise ValueError("each species block must contain a '- name:' line")
    return match.group(1).strip().strip("'\"")


def _elements_from_species_blocks(species_blocks):
    elements = []
    for block in species_blocks:
        for match in re.finditer(r"composition:\s*\{([^}]*)\}", block):
            for item in match.group(1).split(","):
                if ":" not in item:
                    continue
                element = item.split(":", 1)[0].strip().strip("'\"")
                if element and element not in elements:
                    elements.append(element)
    return elements


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


def format_cantera_species_yaml(species_head, thermo_block=None):
    """Return one Cantera species entry from a species header and optional thermo block."""
    species_text = _clean_yaml_fragment(species_head)
    if thermo_block is not None:
        species_text = species_text + "\n" + _clean_yaml_fragment(thermo_block)
    return species_text + "\n"


def format_cantera_mechanism_yaml(
    species_blocks,
    reaction_blocks=None,
    phase_name="gas",
    elements=None,
    species_names=None,
    thermo_model="ideal-gas",
    kinetics_model="gas",
    state=None,
):
    """Return a complete Cantera YAML mechanism from species and reaction fragments."""
    species_blocks = [_clean_yaml_fragment(block) for block in species_blocks]
    species_blocks = [block for block in species_blocks if block]
    if not species_blocks:
        raise ValueError("at least one species block is required")

    if species_names is None:
        species_names = [_species_name_from_block(block) for block in species_blocks]
    else:
        species_names = [str(name) for name in species_names]
        if len(species_names) != len(species_blocks):
            raise ValueError("species_names must have the same length as species_blocks")

    if elements is None:
        elements = _elements_from_species_blocks(species_blocks)
    else:
        elements = [str(element) for element in elements]
    if not elements:
        raise ValueError("elements must be provided or derivable from species composition blocks")

    reaction_blocks = [] if reaction_blocks is None else [
        _clean_yaml_fragment(block) for block in reaction_blocks
    ]
    reaction_blocks = [block for block in reaction_blocks if block]
    reactions_value = "all" if reaction_blocks else "none"
    state = {"T": 300.0, "P": "1 atm"} if state is None else dict(state)

    lines = [
        "phases:",
        f"- name: {phase_name}",
        f"  thermo: {thermo_model}",
        f"  elements: {_flow_list(elements)}",
        f"  species: {_flow_list(species_names)}",
        f"  kinetics: {kinetics_model}",
        f"  reactions: {reactions_value}",
        "  state:",
    ]
    for key, value in state.items():
        lines.append(f"    {key}: {value}")

    lines.extend(["", "species:"])
    lines.extend(species_blocks)

    if reaction_blocks:
        lines.extend(["", "reactions:"])
        lines.extend(reaction_blocks)

    return "\n".join(lines).rstrip() + "\n"


def make_cantera_mechanism_yaml(
    species_blocks,
    reaction_blocks=None,
    yaml_name="mechanism.yaml",
    root_path=".",
    phase_name="gas",
    elements=None,
    species_names=None,
    thermo_model="ideal-gas",
    kinetics_model="gas",
    state=None,
):
    """Write a complete Cantera YAML mechanism from species and reaction fragments."""
    yaml_path = _output_path(root_path, yaml_name)
    yaml_text = format_cantera_mechanism_yaml(
        species_blocks,
        reaction_blocks=reaction_blocks,
        phase_name=phase_name,
        elements=elements,
        species_names=species_names,
        thermo_model=thermo_model,
        kinetics_model=kinetics_model,
        state=state,
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return None


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
        f"{element}: {count}" for element, count in composition_dict.items()
    ) + "}"

    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(f"- name: {specie_name}\n")
        f.write(f"  composition: {formatted_string}\n")
    return None
