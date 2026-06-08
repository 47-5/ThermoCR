"""Read QM output with the modern ThermoCR IO API."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ThermoCR.io import read_electronic_energy, read_molecule_data
from ThermoCR.symmetry import detect_point_group, rotational_symmetry_number

EXAMPLE_DIR = ROOT / "example"


def main():
    output_path = EXAMPLE_DIR / "CPD.out"
    molecule = read_molecule_data(output_path)
    energy = read_electronic_energy(output_path, return_hartree=True)
    point_group = detect_point_group(
        coords=molecule.coordinates,
        numbers=molecule.atom_numbers,
    )

    print(f"file: {output_path}")
    print(f"atoms: {molecule.n_atoms}")
    print(f"electronic energy / hartree: {energy:.12f}")
    print(f"frequencies: {len(molecule.frequencies)}")
    print(f"point group: {point_group}")
    print(f"rotational symmetry number: {rotational_symmetry_number(point_group)}")


if __name__ == "__main__":
    main()
