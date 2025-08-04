import cclib
from ase.units import Hartree


def read_qm_out(filepath):
    data = cclib.io.ccread(filepath)
    return data


def read_atom_coord(filepath, coord_index=-1):
    data = cclib.io.ccread(filepath)
    atom_numbers = data.atomnos
    coords = data.atomcoords[coord_index]
    return atom_numbers, coords


def read_vib(filepath):
    data = cclib.io.ccread(filepath)
    if len(data.atomnos) <= 1:
        return []
    vibfreqs = data.vibfreqs
    return vibfreqs


def read_imaginary_vib(filepath, vibfreqs=None):
    if vibfreqs is None:
        vibfreqs = read_vib(filepath=filepath)

    # 将所有元素转换为浮点数
    vibfreqs_float = [float(freq) for freq in vibfreqs]

    # 筛选出所有的负数频率
    negative_freqs = [freq for freq in vibfreqs_float if freq < 0.0]

    if len(negative_freqs) > 1:
        # 当存在多个负数频率时，选取绝对值最大的那一个
        selected_freq = min(negative_freqs)
        print("警告：检测到多个虚频，已选择绝对值最大的一个。")
    elif len(negative_freqs) == 1:
        # 只有一个负数频率的情况
        selected_freq = negative_freqs[0]
    else:
        # 没有找到负数频率
        selected_freq = None
        print("注意：没有检测到虚频。")

    return selected_freq


def read_ee(filepath, ee_index=-1, return_Hartree=True):
    data = cclib.io.ccread(filepath)
    ee = data.scfenergies[ee_index]
    if return_Hartree:
        ee /= Hartree
    return ee



# if __name__ == "__main__":
#
#     vib = read_imaginary_vib(filepath=None, vibfreqs=[-200, 100, 300])
#     print(vib)