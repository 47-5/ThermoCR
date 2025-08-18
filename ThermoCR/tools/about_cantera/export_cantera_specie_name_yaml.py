import os.path
import cclib
from collections import Counter

from ThermoCR.tools.constant import atomic_number_map


def make_cantera_specie_name_yaml(specie_name, composition_dict=None, read_file_path=None, root_path='.'):
    """
    Generates a YAML file for Cantera species with the specified name and composition.
    The function creates a YAML file that defines a chemical species for use in Cantera simulations.
    It can automatically determine the composition from a given computational chemistry output file or use a provided dictionary of elemental compositions.

    Parameters:
    - specie_name: (str)
            The name of the chemical species to be defined in the YAML file.
    - composition_dict: (Optional[Dict[str, int]])
            A dictionary where keys are element symbols (e.g., 'C', 'H') and values are the number of atoms of each element in the species. If not provided, it will be inferred from the `read_file_path` if available.
    - read_file_path: (Optional[str])
            Path to a computational chemistry output file (supported by cclib) from which the elemental composition of the species can be extracted. If provided, overrides `composition_dict`.
    - root_path: (str, default='.')
            The directory where the generated YAML file will be saved. Defaults to the current working directory.

    Returns:
        None

    Raises:
        FileNotFoundError: If `read_file_path` is provided but the file does not exist.
        ValueError: If neither `composition_dict` nor `read_file_path` is provided, making it impossible to determine the species' composition.
    """
    yaml_path = os.path.join(root_path, f'{specie_name}_head.yaml')

    if read_file_path is not None:
        data = cclib.io.ccread(read_file_path)
        atom_numbers = data.atomnos
        count_dict = Counter(atom_numbers)
        composition_dict = {atomic_number_map[key - 1]: value for key, value in count_dict.items()}
        print(composition_dict)

    formatted_string = '{' + ', '.join(f'{key}:{value}' for key, value in composition_dict.items()) + '}'

    with open(yaml_path, 'w') as f:
        f.write(f'- name: {specie_name}\n')
        f.write(f'  composition: {formatted_string}\n')
    return None



# if __name__ == '__main__':
#
#     make_cantera_specie_name_yaml(specie_name='S01', read_file_path='../../../01.out')