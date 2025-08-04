import os.path


def make_cantera_reaction_yaml(r_name_list, p_name_list, A, b, Ea, reversible=True,
                               yaml_name='reaction.yaml', write_mode='a',
                               root_path='.'):
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