import os.path


def make_cantera_reaction_yaml(r_name_list, p_name_list, A, b, Ea, reversible=True,
                               yaml_name='reaction.yaml', write_mode='a',
                               root_path='.'):
    """
    Generate a Cantera reaction in YAML format and write it to a file.

    The function constructs a reaction string from given reactant and product names, along with the Arrhenius
    parameters A, b, and Ea. The reaction is written in a specified format to a YAML file, which can be used for
    chemical kinetics simulations with Cantera.

    Parameters:
    - r_name_list (List[str]): List of reactant names.
    - p_name_list (List[str]): List of product names.
    - A (float): Pre-exponential factor in the Arrhenius equation.
    - b (float): Temperature exponent in the Arrhenius equation.
    - Ea (float): Activation energy in the Arrhenius equation.
    - reversible (bool, optional): Indicates if the reaction is reversible. Defaults to True.
    - yaml_name (str, optional): Name of the output YAML file. Defaults to 'reaction.yaml'.
    - write_mode (str, optional): Mode in which to open the file ('w' for write, 'a' for append). Defaults to 'a'.
    - root_path (str, optional): Path where the YAML file will be saved. Defaults to the current directory.

    Returns:
    None

    Raises:
    - FileNotFoundError: If the root path does not exist.
    - IOError: If there is an issue writing to the file.
    """
    left = ''
    middle = '<=>' if reversible else '=>'
    right = ''
    n_left = 0
    n_right = 0
    while r_name_list:
        if n_left > 0:
            left += ' + '
        left += f'{r_name_list.pop(0)}'
        n_left += 1

    while p_name_list:
        if n_right > 0:
            right += ' + '
        right += f'{p_name_list.pop(0)}'
        n_right += 1

    yaml_path = os.path.join(root_path, yaml_name)
    with open(yaml_path, write_mode) as f:
        f.write(f'- equation: {left} {middle} {right}\n')
        f.write(f'  type: elementary\n')
        f.write(f'  rate-constant: {{A: {A}, b: {b}, Ea: {Ea} }}\n')