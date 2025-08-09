import os.path

au_to_kcal_per_mol = 627.51
au_to_kJ_per_mol = 2625.5


def write_cantera_yaml_thermo_piecewise_Gibbs(specie_name,
                               T=None, H_formation=None, G_formation=None,
                                               root_path='.'):
    """
    Write a piecewise Gibbs free energy thermodynamic data for a species in Cantera YAML format.

    Summary:
    This function takes the name of a chemical species and its thermodynamic data (temperature, enthalpy of formation,
    and Gibbs free energy of formation) to generate a Cantera-compatible YAML file. The file is saved with the species
    name in the specified root path. The function assumes that the temperature array includes 298.15 K, which is used
    as a reference point for calculating the standard enthalpy of formation.

    Args:
        specie_name (str): The name of the chemical species.
        T (Optional[ArrayLike]): An array of temperatures in Kelvin.
        H_formation (Optional[ArrayLike]): An array of enthalpies of formation at the given temperatures in kJ/mol.
        G_formation (Optional[ArrayLike]): An array of Gibbs free energies of formation at the given temperatures in kJ/mol.
        root_path (str): The directory where the YAML file will be saved. Defaults to the current working directory.

    Returns:
        None

    Raises:
        IndexError: If 298.15 K is not found in the provided temperature array.
        ValueError: If any of the input arrays (T, H_formation, G_formation) are not of the same length.
    """
    T298_index = T.tolist().index(298.15)

    data = {str(t): str(g) for t, g in zip(T, G_formation)}

    yaml_path = os.path.join(root_path, f'{specie_name}.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'  thermo:\n')
        f.write(f'   model: piecewise-Gibbs\n')
        f.write(f'   h0: {H_formation[T298_index]} kJ/mol\n')
        f.write(f'   dimensionless: False\n')
        f.write(f'   data: {data}')
    return None


def write_cantera_yaml_thermo_NASA7(specie_name, T_range, nasa7_parameters,
                                     root_path='.'):
    """
    Write a Cantera YAML format thermodynamic data for a species using NASA7 polynomial.

    This function takes the name of a chemical species, a temperature range, and NASA7
    polynomial coefficients to generate a YAML file that can be used with Cantera
    for thermodynamic properties. The generated file is saved in the specified root
    directory or the current working directory if no path is provided.

    Parameters:
    specie_name (str): The name of the chemical species.
    T_range (Tuple[float, float]): A tuple containing the minimum and maximum
    temperature for which the NASA7 polynomial is valid.
    nasa7_parameters (List[float]): A list of 14 coefficients for the NASA7
    polynomial.
    root_path (str, optional): The root directory where the YAML file will be
    saved. Defaults to the current working directory ('.').

    Raises:
    ValueError: If the length of nasa7_parameters is not exactly 14.
    TypeError: If T_range is not a tuple or if nasa7_parameters is not a list.

    Returns:
    None: This function does not return any value. It writes the output to a file.
    """
    yaml_path = os.path.join(root_path, f'{specie_name}.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'  thermo:\n')
        f.write(f'   model: NASA7\n')
        f.write(f'   temperature-ranges: {list(T_range)}\n')
        f.write(f'   data:\n')
        f.write(f'   - {list(nasa7_parameters)}\n')
    return None


def write_cantera_yaml_thermo_NASA9(specie_name, T_range, nasa9_parameters, reference_p=1,
                                     root_path='.'):
    """
    Writes the thermodynamic data of a species in Cantera's YAML format using NASA9 polynomial coefficients.

    The function creates a YAML file for a given species, containing its thermodynamic properties
    defined by the NASA9 polynomial. The file is saved in the specified root directory with the
    species name as the filename.

    Parameters:
    specie_name (str): The name of the species to write the thermodynamic data for.
    T_range (Tuple[float, float, float]): A tuple containing three temperature points that define
    the temperature ranges over which the NASA9 polynomials are valid.
    nasa9_parameters (List[float]): A list of 14 or 15 NASA9 polynomial coefficients.
    reference_p (float, optional): The reference pressure in bar. Defaults to 1 bar.
    root_path (str, optional): The directory where the YAML file will be saved. Defaults to the
    current working directory.

    Returns:
    None

    Raises:
    FileNotFoundError: If the specified root path does not exist or is not writable.
    TypeError: If any of the input parameters do not match their expected types.
    ValueError: If the length of nasa9_parameters is not 14 or 15, or if T_range does not contain
    exactly three elements.

    Notes:
    - The function assumes that the provided `nasa9_parameters` and `T_range` are correct and
    appropriate for the species in question.
    - The output file will be named as `<specie_name>.yaml` and placed in the `root_path`.
    - No checks are performed on the validity of the NASA9 coefficients or the temperature range.
    """
    yaml_path = os.path.join(root_path, f'{specie_name}.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'  thermo:\n')
        f.write(f'   model: NASA9\n')
        f.write(f'   temperature-ranges: {list(T_range)}\n')
        f.write(f'   reference-pressure: {reference_p} bar')
        f.write(f'   data:\n')
        f.write(f'   - {list(nasa9_parameters)}\n')
    return None


def write_cantera_yaml_thermo_Shomate(specie_name, T_range, Shomate_parameters, reference_p=1,
                                     root_path='.'):
    """
    Write a Cantera YAML file for thermodynamic data using Shomate parameters.

    Detailed summary:
    This function generates a Cantera-compatible YAML file containing thermodynamic
    data based on the provided Shomate parameters. The generated file includes
    temperature ranges and reference pressure, suitable for use in chemical
    kinetics and thermodynamics simulations with Cantera.

    Parameters:
    specie_name (str): Name of the species for which the thermodynamic data is being
                       written.
    T_range (tuple): A tuple specifying the temperature range over which the
                     Shomate parameters are valid.
    Shomate_parameters (list): List of Shomate parameters used to define the
                               thermodynamic properties.
    reference_p (float, optional): Reference pressure in bar. Default is 1 bar.
    root_path (str, optional): Root directory where the YAML file will be saved.
                               Default is the current working directory.

    Returns:
    None

    Raises:
    FileNotFoundError: If the specified root path does not exist.
    TypeError: If `specie_name` is not a string or if `T_range` or
               `Shomate_parameters` are not provided as expected types.
    ValueError: If `T_range` does not contain exactly two elements.
    """
    yaml_path = os.path.join(root_path, f'{specie_name}.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'  thermo:\n')
        f.write(f'   model: NASA9\n')
        f.write(f'   temperature-ranges: {list(T_range)}\n')
        f.write(f'   reference-pressure: {reference_p} bar')
        f.write(f'   data:\n')
        f.write(f'   - {list(Shomate_parameters)}\n')
    return None









