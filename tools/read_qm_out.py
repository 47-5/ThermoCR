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


def read_ee(filepath, ee_index=-1, return_Hartree=True):
    data = cclib.io.ccread(filepath)
    ee = data.scfenergies[ee_index]
    if return_Hartree:
        ee /= Hartree
    return ee



# if __name__ == "__main__":
#
#     # print(data.scfenergies)
#
#     data = read_qm_out('01_02.out')
#
#     print(read_atom_coord('01_02_sp.out'))