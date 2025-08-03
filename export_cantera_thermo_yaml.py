import os.path

au_to_kcal_per_mol = 627.51
au_to_kJ_per_mol = 2625.5


def write_cantera_yaml_thermo_piecewise_Gibbs(specie_name,
                               T=None, H_formation=None, G_formation=None,
                                               root_path='.'):

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
    yaml_path = os.path.join(root_path, f'{specie_name}.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'  thermo:\n')
        f.write(f'   model: NASA9\n')
        f.write(f'   temperature-ranges: {list(T_range)}\n')
        f.write(f'   reference-pressure: {reference_p} bar')
        f.write(f'   data:\n')
        f.write(f'   - {list(Shomate_parameters)}\n')
    return None









