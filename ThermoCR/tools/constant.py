from ase.units import Bohr, Hartree


k_b = 1.3806503e-23 # J/K
h = 6.6260696e-34 # J⋅s
R = 8.3144648 # J⋅(mol⋅K)-1
Na = 6.02214179e23

amu2kg = 1.66053878e-27

convert_I = amu2kg * (Bohr * 1e-10) ** 2

wave2freq = 2.99792458e10 # cm^-1 to s^-1 (Hz)

au2eV = Hartree
au2kj_mol = 2625.5e0
au2j_mol = au2kj_mol * 1000
au2cm_1=219474.6363e0

Eh = 4.3597447222071e-18  # 1 Hartree = ? J

cm_1_to_s_1 = 3e10  # cm^-1 -> s^-1
