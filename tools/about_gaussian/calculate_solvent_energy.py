import glob
import ase
from ase.io import read, write
from ase.units import Hartree, kcal, mol
import re
import os
import pandas as pd


standard_state_energy = 1.84 * kcal / mol


def calculate_solvent_energy(gas_gaussian_out_path, sol_gaussian_out_path, add_standard_state=True,
                             return_hartree=True):
    gas_atoms = read(gas_gaussian_out_path, format='gaussian-out')
    sol_atoms = read(sol_gaussian_out_path, format='gaussian-out')
    gas_energy = gas_atoms.get_potential_energy() / Hartree
    sol_energy = sol_atoms.get_potential_energy() / Hartree
    delta_energy = sol_energy - gas_energy
    if add_standard_state:
        delta_energy += standard_state_energy

    if return_hartree:
        delta_energy /= Hartree
    return delta_energy


# 自定义排序键函数
def sort_key(file_name):
    # 使用正则表达式提取文件名中的数字部分
    match = re.search(r'(\d+)_(\d+)_path(\d+)_(\d+)', file_name)
    if match:
        # 提取匹配到的数字部分
        part1, part2, part3, part4 = map(int, match.groups())
        return part1, part2, part3, part4
    else:
        # 如果没有匹配到，返回一个较大的值以便排在最后
        return float('inf'), float('inf'), float('inf'), float('inf')


# if __name__ == '__main__':
#
#     run_mode = 'batch'
#
#     add_standard_state = True
#     return_hartree = True
#     sort = True
#
#     if run_mode == 'single':
#         delta = calculate_solvent_energy(gas_gaussian_out_path='species_dft_smd/01_sp.out',
#                                          sol_gaussian_out_path='species_dft_smd/01_sp_smd.out',
#                                          add_standard_state=add_standard_state,
#                                          return_hartree=return_hartree)
#         print(delta)
#
#     elif run_mode == 'batch':
#
#         gas_gaussian_out_path_list = glob.glob('VTST_sol_smd/*_sp.out')
#         sol_gaussian_out_path_list = glob.glob('VTST_sol_smd/*_sp_smd.out')
#         if sort:
#             gas_gaussian_out_path_list.sort(key=sort_key)
#             sol_gaussian_out_path_list.sort(key=sort_key)
#         print(gas_gaussian_out_path_list)
#         print(sol_gaussian_out_path_list)
#
#         # todo name怎么设置根据实际情况修改
#         name = [os.path.basename(i) for i in gas_gaussian_out_path_list]
#         name = [i.split('_sol')[0] for i in name]
#         print(name)
#
#         delta_energies = []
#         for n, gas, sol in zip(name, gas_gaussian_out_path_list, sol_gaussian_out_path_list):
#             delta = calculate_solvent_energy(gas, sol,
#                                              add_standard_state=add_standard_state,
#                                              return_hartree=return_hartree)
#             delta_energies.append(delta)
#
#         df = pd.DataFrame({'name': name, 'delta_energy': delta_energies})
#         df.to_excel('delta_energy.xlsx', index=False)
