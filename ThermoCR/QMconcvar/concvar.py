import yaml
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from collections import defaultdict
import re

from ThermoCR.QMthermo import qm_thermo
from ThermoCR.QMkinetics import k_TST


class ChemicalKineticsSimulator:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)

        self.system_config = self.config['system']
        self.T = self.system_config['T']
        self.P = self.system_config['P']
        self.R = 8.314462618  # J/(mol·K)

        self.species_config = self.config['species']
        self.species = list(self.species_config.keys())  # 物种ID列表
        self.thermo_data = self.calculate_thermodynamic_data()

        self.reaction_data = {}  # 存储反应数据

        # 加载反应数据
        self.parse_reactions()

    def load_config(self, config_file):
        """加载YAML配置文件"""
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

    def calculate_thermodynamic_data(self):
        """为每个物种计算热力学数据"""
        all_thermo_data = {}
        for species_id, species_info in self.species_config.items():
            # print(species_id, species_info)
            thermo_type = species_info.get('thermo_type')

            if thermo_type == 'on_the_fly':
                # print(f"Calculating thermo for {species_info['name']}...")
                params = species_info['on_the_fly_params']
                thermo_result = self.thermo_on_the_fly(on_the_fly_params=params)
                all_thermo_data[species_id] = thermo_result
                # print(f"  G = {thermo_result['G/(J/mol)']:.2f} J/mol")
        return all_thermo_data

    def parse_reactions(self):
        """解析所有反应，包括可逆反应"""
        self.parsed_reactions = []

        for i, reaction in enumerate(self.config['reactions']):
            equation = reaction['equation']
            rate_type = reaction.get('rate_type')

            # 解析可逆反应
            reaction_parts = self.parse_reaction_equation(equation)

            if rate_type == 'TST':
                TST_params = reaction.get('TST_params', {})

                # 计算正向反应速率常数
                k_forward = self.calculate_TST_rate_constants(
                    reaction_parts['forward'], TST_params
                )

                # 计算逆向反应速率常数 - 基于平衡常数而不是TST
                if reaction_parts['reverse'] is not None:
                    # 计算平衡常数
                    K_eq = self.calculate_equilibrium_constant(
                        reaction_parts['forward']['reactants'],
                        reaction_parts['forward']['products']
                    )

                    # 根据平衡常数计算逆反应速率常数
                    k_reverse = k_forward / K_eq

                    print(
                        f"Reaction {i + 1}: k_forward = {k_forward:.2e}, k_reverse = {k_reverse:.2e}, K_eq = {K_eq:.2e}")
                else:
                    k_reverse = 0.0  # 不可逆反应
                    print(f"Reaction {i + 1}: k_forward = {k_forward:.2e} (irreversible)")

                # 存储正向反应
                self.parsed_reactions.append({
                    'reaction_id': f"R{i + 1}_forward",
                    'reactants': self.parse_chemical_species(reaction_parts['forward']['reactants']),
                    'products': self.parse_chemical_species(reaction_parts['forward']['products']),
                    'rate_constant': k_forward,
                    'is_reversible': (reaction_parts['reverse'] is not None)
                })

                # 如果是可逆反应，也存储逆向反应
                if reaction_parts['reverse'] is not None:
                    self.parsed_reactions.append({
                        'reaction_id': f"R{i + 1}_reverse",
                        'reactants': self.parse_chemical_species(reaction_parts['reverse']['reactants']),
                        'products': self.parse_chemical_species(reaction_parts['reverse']['products']),
                        'rate_constant': k_reverse,
                        'is_reversible': True
                    })

    def parse_reaction_equation(self, equation):
        """解析可逆反应方程式，返回正向和逆向反应"""
        if '<->' in equation:
            reactants_str, products_str = equation.split('<->')
            return {
                'forward': {'reactants': reactants_str.strip(), 'products': products_str.strip()},
                'reverse': {'reactants': products_str.strip(), 'products': reactants_str.strip()}
            }
        elif '->' in equation:
            reactants_str, products_str = equation.split('->')
            return {
                'forward': {'reactants': reactants_str.strip(), 'products': products_str.strip()},
                'reverse': None
            }
        else:
            raise ValueError(f"Invalid reaction equation: {equation}")

    def parse_chemical_species(self, species_str):
        """解析化学物种和计量系数"""
        species_dict = {}
        pattern = r'(\d*\.?\d*)\s*([A-Za-z][A-Za-z0-9]*)'

        for match in re.finditer(pattern, species_str):
            coeff_str, species = match.groups()
            coefficient = float(coeff_str) if coeff_str else 1.0
            species_dict[species] = coefficient

        return species_dict

    def thermo_on_the_fly(self, on_the_fly_params):
        thermo_result = qm_thermo(
            atom_coord_path=on_the_fly_params.get('atom_coord_path', None),
            atom_numbers=on_the_fly_params.get('atom_numbers', None),
            coords=on_the_fly_params.get('coords', None),
            vib_path=on_the_fly_params.get('vib_path', None),
            vibfreqs=on_the_fly_params.get('vibfreqs', None),
            ee_path=on_the_fly_params.get('ee_path', None),
            ee=on_the_fly_params.get('ee', None),
            T=self.T,
            P=self.P,
            sclZPE=on_the_fly_params.get('sclZPE', 1.0),
            sclU=on_the_fly_params.get('sclU', 1.0),
            sclCv=on_the_fly_params.get('sclCv', 1.0),
            sclS=on_the_fly_params.get('sclS', 1.0),
            U_Minenkov=on_the_fly_params.get('U_Minenkov', False),
            S_Grimme=on_the_fly_params.get('S_Grimme', True),
            verbose=on_the_fly_params.get('verbose', False),
            read_ee_index=on_the_fly_params.get('read_ee_index', -1),
            E_list=on_the_fly_params.get('E_list', None),
            g_list=on_the_fly_params.get('g_list', None),
            ignore_trans_and_rot=on_the_fly_params.get('ignore_trans_and_rot', False),
            c=on_the_fly_params.get('c', None)
        )

        return thermo_result

    def calculate_TST_rate_constants(self, reaction_info, TST_params):
        """基于TST计算速率常数"""
        reactants = self.parse_chemical_species(reaction_info['reactants'])
        products = self.parse_chemical_species(reaction_info['products'])
        # print(f'r:{reactants} | p:{products}')

        # 计算Δn（气相分子数变化）
        n_reactants = sum(reactants.values())
        n_products = sum(products.values())
        delta_n = n_reactants - n_products

        # 计算自由能垒
        if TST_params.get('TS_on_the_fly_params'):
            # 如果有过渡态数据，计算实际的自由能垒
            TS_thermo = self.thermo_on_the_fly(TST_params['TS_on_the_fly_params'])
            if TS_thermo:
                G_TS = TS_thermo['G/(J/mol)']
                G_reactants = sum(coeff * self.thermo_data[species]['G/(J/mol)']
                                  for species, coeff in reactants.items()
                                  if species in self.thermo_data)
                delta_G = G_TS - G_reactants
            else:
                delta_G = TST_params.get('delta_G')
        else:
            delta_G = TST_params.get('delta_G')

        # 调用k_TST函数
        k_forward = k_TST(
            delta_G=delta_G,
            delta_n=delta_n,
            T=TST_params.get('T', self.T),
            P0=TST_params.get('P0', 100000),
            sigma=TST_params.get('sigma', 1),
            liquid=TST_params.get('liquid', False),
            tunnelling_effect=TST_params.get('tunnelling_effect'),
            imaginary_freq=TST_params.get('imaginary_freq'),
            delta_H_barrier_f_0K=TST_params.get('delta_H_barrier_f_0K'),
            delta_H_barrier_r_0K=TST_params.get('delta_H_barrier_r_0K')
        )

        # 单位换算：确保速率常数使用 mol/m³ 作为浓度单位
        liquid = TST_params.get('liquid', False)
        if not liquid:
            # 对于气相反应，k_TST 返回的单位是 (molecule/m³)^(-delta_n) * s^-1
            # 需要转换为 (mol/m³)^(-delta_n) * s^-1
            avogadro = 6.02214076e23  # 阿伏伽德罗常数
            k_forward = k_forward * (avogadro ** delta_n)

        # 注意：现在 k_forward 的单位是 (mol/m³)^(-delta_n) * s^-1
        # 如果你希望使用 mol/L (M) 作为浓度单位，需要进一步转换：
        k_forward = k_forward * (1000 ** delta_n)  # 从 mol/m³ 转换为 mol/L

        return k_forward

    def calculate_equilibrium_constant(self, reactants_str, products_str):
        """计算反应的平衡常数"""
        reactants = self.parse_chemical_species(reactants_str)
        products = self.parse_chemical_species(products_str)

        # 计算反应的ΔG
        delta_G = 0.0

        # 产物 - 反应物
        for species, coeff in products.items():
            if species in self.thermo_data:
                delta_G += coeff * self.thermo_data[species]['G/(J/mol)']

        for species, coeff in reactants.items():
            if species in self.thermo_data:
                delta_G -= coeff * self.thermo_data[species]['G/(J/mol)']

        # 计算平衡常数 K = exp(-ΔG/RT)
        K_eq = np.exp(-delta_G / (self.R * self.T))

        return K_eq

    def reaction_rate(self, concentrations, reaction):
        """计算单个反应的反应速率"""
        rate = reaction['rate_constant']

        # 反应物浓度乘积
        for species, coeff in reaction['reactants'].items():
            rate *= concentrations[species] ** coeff

        return rate

    def dydt(self, t, y):
        """定义微分方程组"""
        # 将向量y转换为浓度字典
        concentrations = {species: y[i] for i, species in enumerate(self.species)}
        dydt_dict = {species: 0.0 for species in self.species}

        # 对每个反应计算浓度变化
        for reaction in self.parsed_reactions:
            rate = self.reaction_rate(concentrations, reaction)

            # 反应物浓度减少
            for species, coeff in reaction['reactants'].items():
                dydt_dict[species] -= coeff * rate

            # 产物浓度增加
            for species, coeff in reaction['products'].items():
                dydt_dict[species] += coeff * rate

        # 返回导数向量
        return np.array([dydt_dict[species] for species in self.species])

    def simulate(self, method='BDF', rtol=1e-6, atol=1e-8):
        """执行模拟"""
        # 初始条件
        y0 = np.array([self.config['system']['initial_concentrations'][species]
                       for species in self.species])

        # 时间范围
        t_span = (self.config['system']['time_span']['start'], self.config['system']['time_span']['end'])

        # 求解微分方程
        solution = solve_ivp(
            self.dydt, t_span, y0,
            method=method, rtol=rtol, atol=atol,
            dense_output=True
        )

        self.solution = solution
        return solution

    def plot_results(self, save_path=None):
        """绘制结果"""
        if not hasattr(self, 'solution'):
            print("请先运行模拟!")
            return

        t_eval = np.linspace(self.config['system']['time_span']['start'],
                             self.config['system']['time_span']['end'], 1000)
        concentrations = self.solution.sol(t_eval)

        plt.figure(figsize=(10, 6))
        for i, species_id in enumerate(self.species):
            species_name = self.species_config[species_id]['name']
            plt.plot(t_eval, concentrations[i], label=species_name, linewidth=2)

        plt.xlabel('time')
        plt.ylabel('C')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self):
        """生成模拟报告"""
        print("\n" + "=" * 60)
        print("CHEMICAL KINETICS SIMULATION REPORT")
        print("=" * 60)

        print("\nSPECIES THERMODYNAMIC DATA:")
        print("-" * 40)
        for species_id, thermo in self.thermo_data.items():
            species_name = self.species_config[species_id]['name']
            if 'G/(J/mol)' in thermo:
                print(f"{species_name} ({species_id}): G = {thermo['G/(J/mol)']:>10.2f} J/mol")

        print("\nREACTION RATE CONSTANTS:")
        print("-" * 40)
        for reaction in self.parsed_reactions:
            reactants = " + ".join([f"{coeff if coeff != 1 else ''}{self.species_config[s]['name']}"
                                    for s, coeff in reaction['reactants'].items()])
            products = " + ".join([f"{coeff if coeff != 1 else ''}{self.species_config[s]['name']}"
                                   for s, coeff in reaction['products'].items()])
            print(f"{reaction['reaction_id']}: {reactants} -> {products}")
            print(f"  k = {reaction['rate_constant']:.2e}")


# 使用示例
if __name__ == "__main__":
    simulator = ChemicalKineticsSimulator('reaction_system.yaml')
    simulator.generate_report()
    results = simulator.simulate()
    simulator.plot_results('kinetics_simulation.png')

    # from ThermoCR.tools import read_atom_coord, read_vib
    # print(read_atom_coord('01.out'))

    # qm_thermo(atom_coord_path='01.out', vib_path='01.out')
    # print(read_vib(filepath='01.out'))
    # print(simulator.config)
    # print(simulator.species_config)
    # print(simulator.species)
    # print(simulator.thermo_data)
    # print(simulator.system_config)
    # print(simulator.parsed_reactions)