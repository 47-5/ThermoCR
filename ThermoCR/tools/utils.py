from ase import Atoms
import numpy as np

from ThermoCR.pointgroup import PointGroup
from ase.units import Bohr
from ThermoCR.tools.constant import amu2kg


def get_point_group(coords, symbols=None, numbers=None):
    if symbols is None:
        atoms = Atoms(numbers=numbers, positions=coords)
        symbols = atoms.symbols
    pg = PointGroup(positions=coords, symbols=symbols)
    return pg.get_point_group()


def get_I(coords, numbers):
    atoms = Atoms(numbers=numbers, positions=coords)
    I = atoms.get_moments_of_inertia() / Bohr ** 2
    return I


def check_linear(I, threshold=1e-3):
    if I[0] < threshold and np.abs(I[1] - I[2]) < threshold:
        return True
    else:
        return False


def get_rotational_symmetry_number(point_group):
    """
    根据点群名称计算旋转对称数

    参数:
    point_group (str): 点群符号（来自PointGroup.get_point_group()的输出）

    返回:
    int: 旋转对称数
    """
    # 处理线性分子
    if point_group == 'Dinfh':
        return 2
    elif point_group == 'Cinfv':
        return 1

    # 处理球对称群
    cubic_groups = {
        'T': 12, 'Th': 12, 'Td': 12,
        'O': 24, 'Oh': 24,
        'I': 60, 'Ih': 60
    }
    if point_group in cubic_groups:
        return cubic_groups[point_group]

    # 处理带数字的点群（Cn, Dn等）
    if len(point_group) >= 2 and point_group[1].isdigit():
        # 提取旋转轴阶数n
        n = int(''.join(filter(str.isdigit, point_group)))

        # 根据点群类型确定旋转对称数
        if point_group.startswith('C') or point_group.startswith('S'):
            return n
        elif point_group.startswith('D'):
            return 2 * n

    # 处理特殊点群（无数字）
    special_groups = {
        'C1': 1, 'Cs': 1, 'Ci': 1
    }
    if point_group in special_groups:
        return special_groups[point_group]

    # 默认情况（理论上不应到达这里）
    raise ValueError(f"未知的点群符号: {point_group}")


# # 示例使用
# if __name__ == "__main__":
#     # 测试用例
#     test_groups = [
#         'C1', 'Cs', 'Ci', 'C2', 'C3', 'C3v', 'C4h', 'S4',
#         'D2', 'D3', 'D3h', 'D5d', 'Td', 'Oh', 'Ih', 'Cinfv', 'Dinfh'
#     ]
#
#     for pg in test_groups:
#         print(f"点群 {pg} 的旋转对称数: {get_rotational_symmetry_number(pg)}")