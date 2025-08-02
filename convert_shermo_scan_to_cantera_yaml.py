from read_shermo_scan import read_shermo_scan

au_to_kcal_per_mol = 627.51
au_to_kJ_per_mol = 2625.5


def write_cantera_yaml_species_piecewise_Gibbs(specie_name, n_C, n_H,
                               T=None, H_formation=None, G_formation=None):

    T298_index = T.tolist().index(298.15)

    data = {str(t): str(g) for t, g in zip(T, G_formation)}
    with open(f'{specie_name}.yaml', 'w') as f:
        f.write(f'- name: {specie_name}\n')
        f.write(f'  composition: {{C: {n_C}, H: {n_H}}}\n')
        f.write(f'  thermo:\n')
        f.write(f'   model: piecewise-Gibbs\n')
        f.write(f'   h0: {H_formation[T298_index]} kJ/mol\n')
        f.write(f'   dimensionless: False\n')
        f.write(f'   data: {data}')
    return None


def write_cantera_yaml_species_NASA7(specie_name, n_C, n_H, T_range, nasa7_parameters):
    with open(f'{specie_name}.yaml', 'w') as f:
        f.write(f'- name: {specie_name}\n')
        f.write(f'  composition: {{C: {n_C}, H: {n_H}}}\n')
        f.write(f'  thermo:\n')
        f.write(f'   model: NASA7\n')
        f.write(f'   temperature-ranges: {list(T_range)}\n')
        f.write(f'   data:\n')
        f.write(f'   - {list(nasa7_parameters)}\n')
    return None









