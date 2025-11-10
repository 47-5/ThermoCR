import yaml
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from collections import defaultdict
import re


from ThermoCR.QMthermo import qm_thermo
from ThermoCR.QMkinetics import k_TST

class AdvancedChemicalKineticsSimulator:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.species_config = self.config['species']
        self.species = list(self.species_config.keys())  # 物种ID列表
        self.thermo_data = {}  # 存储所有物种的热力学数据
        self.reaction_data = {}  # 存储反应数据

        # 加载热力学数据和反应数据
        self.load_thermodynamic_data()
        self.parse_reactions()

    def load_config(self, config_file):
        """加载YAML配置文件"""
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

    def load_thermodynamic_data(self):
        """为每个物种计算热力学数据"""
        print("Loading thermodynamic data...")

        for species_id, species_info in self.species_config.items():
            print(species_id, species_info)
            thermo_type = species_info.get('thermo_type', 'constant')

            if thermo_type == 'on_the_fly':
                print(f"Calculating thermo for {species_info['name']}...")
                params = species_info['on_the_fly_params']

                # 调用你的qm_thermo函数
                thermo_result = qm_thermo(
                    atom_coord_path=params.get('atom_coord_path'),
                    atom_numbers=params.get('atom_numbers'),
                    coords=params.get('coords'),
                    vib_path=params.get('vib_path'),
                    vibfreqs=params.get('vibfreqs'),
                    ee_path=params.get('ee_path'),
                    ee=params.get('ee'),
                    T=params.get('T', 298.15),
                    P=params.get('P', 101325),
                    sclZPE=params.get('sclZPE', 1.0),
                    sclU=params.get('sclU', 1.0),
                    sclCv=params.get('sclCv', 1.0),
                    sclS=params.get('sclS', 1.0),
                    U_Minenkov=params.get('U_Minenkov', False),
                    S_Grimme=params.get('S_Grimme', True),
                    verbose=params.get('verbose', False),
                    read_ee_index=params.get('read_ee_index', -1),
                    E_list=params.get('E_list'),
                    g_list=params.get('g_list'),
                    ignore_trans_and_rot=params.get('ignore_trans_and_rot', False),
                    c=params.get('c')
                )

                self.thermo_data[species_id] = thermo_result
                print(f"  G = {thermo_result['G/(J/mol)']:.2f} J/mol")

            elif thermo_type == 'constant':
                # 对于恒定浓度的物种，只需要存储基本信息
                self.thermo_data[species_id] = {
                    'name': species_info['name'],
                    'concentration': species_info.get('concentration', 0.0)
                }

    def parse_reversible_reaction(self, equation):
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

    def calculate_reaction_delta_G(self, reactants, products):         # todo 这里是错的，应该是反应和过渡态的热力学量
        """计算反应的ΔG"""
        delta_G = 0.0

        # 产物 - 反应物
        for species, coeff in products.items():
            if species in self.thermo_data:
                delta_G += coeff * self.thermo_data[species]['G/(J/mol)']

        for species, coeff in reactants.items():
            if species in self.thermo_data:
                delta_G -= coeff * self.thermo_data[species]['G/(J/mol)']

        return delta_G

    def calculate_TS_thermo(self, TS_params):
        """计算过渡态的热力学数据"""
        if TS_params is None:
            return None

        print("Calculating transition state thermodynamics...")
        thermo_result = qm_thermo(
            atom_coord_path=TS_params.get('atom_coord_path'),
            atom_numbers=TS_params.get('atom_numbers'),
            coords=TS_params.get('coords'),
            vib_path=TS_params.get('vib_path'),
            vibfreqs=TS_params.get('vibfreqs'),
            ee_path=TS_params.get('ee_path'),
            ee=TS_params.get('ee'),
            T=TS_params.get('T', 298.15),
            P=TS_params.get('P', 101325),
            sclZPE=TS_params.get('sclZPE', 1.0),
            sclU=TS_params.get('sclU', 1.0),
            sclCv=TS_params.get('sclCv', 1.0),
            sclS=TS_params.get('sclS', 1.0),
            U_Minenkov=TS_params.get('U_Minenkov', False),
            S_Grimme=TS_params.get('S_Grimme', True),
            verbose=TS_params.get('verbose', False),
            read_ee_index=TS_params.get('read_ee_index', -1),
            E_list=TS_params.get('E_list'),
            g_list=TS_params.get('g_list'),
            ignore_trans_and_rot=TS_params.get('ignore_trans_and_rot', False),
            c=TS_params.get('c')
        )

        return thermo_result

    def calculate_TST_rate_constants(self, reaction_info, TST_params):
        """基于TST计算速率常数"""
        reactants = self.parse_chemical_species(reaction_info['reactants'])
        products = self.parse_chemical_species(reaction_info['products'])

        # 计算Δn（气相分子数变化）
        n_reactants = sum(reactants.values())
        n_products = sum(products.values())
        delta_n = n_products - n_reactants

        # 计算自由能垒
        if TST_params.get('TS_on_the_fly_params'):
            # 如果有过渡态数据，计算实际的自由能垒
            TS_thermo = self.calculate_TS_thermo(TST_params['TS_on_the_fly_params'])
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

        # 如果既没有过渡态数据也没有提供delta_G，则从反应物和产物计算
        if delta_G is None:
            delta_G = self.calculate_reaction_delta_G(reactants, products)
            # 对于TST，我们需要的是活化自由能，这里简单假设为反应自由能的一半
            # 这是一个近似，实际应该使用过渡态数据
            delta_G = delta_G * 0.5

        # 调用你的k_TST函数
        k_forward = k_TST(
            delta_G=delta_G,
            delta_n=delta_n,
            T=TST_params.get('T', 298.15),
            P0=TST_params.get('P0', 100000),
            sigma=TST_params.get('sigma', 1),
            liquid=TST_params.get('liquid', False),
            tunnelling_effect=TST_params.get('tunnelling_effect'),
            imaginary_freq=TST_params.get('imaginary_freq'),
            delta_H_barrier_f_0K=TST_params.get('delta_H_barrier_f_0K'),
            delta_H_barrier_r_0K=TST_params.get('delta_H_barrier_r_0K')
        )

        return k_forward

    def parse_reactions(self):
        """解析所有反应，包括可逆反应"""
        self.parsed_reactions = []

        for i, reaction in enumerate(self.config['reactions']):
            equation = reaction['equation']
            rate_type = reaction.get('rate_type', 'constant')

            # 解析可逆反应
            reaction_parts = self.parse_reversible_reaction(equation)

            if rate_type == 'TST':
                TST_params = reaction.get('TST_params', {})

                # 计算正向反应速率常数
                k_forward = self.calculate_TST_rate_constants(
                    reaction_parts['forward'], TST_params
                )

                # 计算逆向反应速率常数
                if reaction_parts['reverse'] is not None:
                    k_reverse = self.calculate_TST_rate_constants(
                        reaction_parts['reverse'], TST_params
                    )
                else:
                    k_reverse = 0.0  # 不可逆反应

                print(f"Reaction {i + 1}: k_forward = {k_forward:.2e}, k_reverse = {k_reverse:.2e}")

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

            elif rate_type == 'constant':
                # 处理常数速率常数的反应
                k_forward = reaction.get('rate_constant', 0.0)
                k_reverse = reaction.get('reverse_rate_constant', 0.0)

                self.parsed_reactions.append({
                    'reaction_id': f"R{i + 1}_forward",
                    'reactants': self.parse_chemical_species(reaction_parts['forward']['reactants']),
                    'products': self.parse_chemical_species(reaction_parts['forward']['products']),
                    'rate_constant': k_forward,
                    'is_reversible': (k_reverse > 0)
                })

                if k_reverse > 0:
                    self.parsed_reactions.append({
                        'reaction_id': f"R{i + 1}_reverse",
                        'reactants': self.parse_chemical_species(reaction_parts['reverse']['reactants']),
                        'products': self.parse_chemical_species(reaction_parts['reverse']['products']),
                        'rate_constant': k_reverse,
                        'is_reversible': True
                    })

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
        y0 = np.array([self.config['initial_concentrations'][species]
                       for species in self.species])

        # 时间范围
        t_span = (self.config['time_span']['start'], self.config['time_span']['end'])

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

        t_eval = np.linspace(self.config['time_span']['start'],
                             self.config['time_span']['end'], 1000)
        concentrations = self.solution.sol(t_eval)

        plt.figure(figsize=(10, 6))
        for i, species_id in enumerate(self.species):
            species_name = self.species_config[species_id]['name']
            plt.plot(t_eval, concentrations[i], label=species_name, linewidth=2)

        plt.xlabel('时间')
        plt.ylabel('浓度')
        plt.title('化学物种浓度随时间变化')
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
    simulator = AdvancedChemicalKineticsSimulator('reaction_system.yaml')
    simulator.generate_report()
    results = simulator.simulate()
    simulator.plot_results('kinetics_simulation.png')

    # from ThermoCR.tools import read_atom_coord, read_vib
    # print(read_atom_coord('01.out'))

    # qm_thermo(atom_coord_path='01.out', vib_path='01.out')
    # print(read_vib(filepath='01.out'))