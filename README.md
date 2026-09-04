# ThermoCR

[简体中文](https://github.com/47-5/ThermoCR/blob/main/README.zh.md) · [中文完整教程](https://github.com/47-5/ThermoCR/blob/main/doc/tutorials.zh.md)

ThermoCR is a Python package for generating molecular thermodynamic data and reaction-rate parameters from quantum-chemistry calculation outputs. It can:

- parse Gaussian and ORCA output files;
- calculate ideal-gas $C_p(T)$, $H(T)$, $S(T)$, and $G(T)$;
- apply vibrational-frequency scaling, QRRHO low-frequency corrections, point-group assignments, and rotational symmetry numbers;
- fit NASA7, NASA9, Shomate, and continuous two-region NASA7 models;
- generate thermochemistry JSON sidecars containing the energy convention, reference pressure, fit metrics, and provenance information;
- calculate TST and VTST rate constants and fit Arrhenius parameters; and
- export Cantera species, reactions, and complete mechanism YAML files.

ThermoCR outputs can be used by Cantera, [`calculate_heat_sink`](https://github.com/47-5/calculate_heat_sink), and other equilibrium or reactor programs. ThermoCR does not provide Benson group-contribution estimates, BSR reference-reaction construction, or experimental thermochemistry database retrieval. The user must supply the enthalpy-of-formation anchor required by the selected thermochemistry protocol.

For the complete Chinese-language guide, see [`doc/tutorials.zh.md`](https://github.com/47-5/ThermoCR/blob/main/doc/tutorials.zh.md).

## Installation

Python 3.11 in a dedicated Conda environment is recommended:

```bash
conda create -n thermocr python=3.11
conda activate thermocr
git clone https://github.com/47-5/ThermoCR.git
cd ThermoCR
pip install .
```

For an editable installation:

```bash
pip install -e .
```

To install Cantera for cross-validation and run the complete test suite:

```bash
pip install -e ".[test]"
```

Verify the installation:

```bash
python -c "import ThermoCR; print(ThermoCR.__version__)"
thermocr --help
```

If the console entry point is unavailable, replace `thermocr` with `python -m ThermoCR`.

## Five-minute CLI quickstart

The following commands use `example/CPD.out`, which is included in the repository. They are suitable for verifying the installation and becoming familiar with the file flow.

### 1. Inspect the electronic energy

```bash
thermocr qm-energy example/CPD.out --unit hartree
```

For an ORCA single-point calculation, use the dedicated command to read the last `FINAL SINGLE POINT ENERGY`:

```bash
thermocr orca-energy path/to/single_point.out
```

### 2. Generate a thermodynamic scan

```bash
thermocr thermo scan example/CPD.out --t-min 300 --t-max 1500 --n-points 49 --pressure 100000 --output CPD_thermo_scan.csv
```

The output table contains temperature, pressure, $C_v$, $C_p$, entropy, ZPE, thermal corrections, and total $U/H/G$. Energies are reported in J/mol, while heat capacities and entropy are reported in J/(mol K).

### 3. Fit a single-region NASA7 model

```bash
thermocr thermo fit CPD_thermo_scan.csv --model NASA7 --weight-strategy uniform --t-range 300 1500 --reference-pressure-pa 100000 --output CPD_thermo.yaml
```

`CPD_thermo.yaml` is a Cantera thermo fragment. To assemble it into a complete mechanism, first create the species header `CPD_head.yaml`:

```yaml
- name: CPD
  composition: {C: 5, H: 6}
```

Then run:

```bash
thermocr cantera mechanism --species-head CPD_head.yaml --species-thermo CPD_thermo.yaml --output mechanism.yaml
```

These CLI commands are intended for quick checks and simple single-file tasks. They do not constitute a production high-accuracy thermochemistry protocol.

## Production thermochemistry workflows

The following tasks require the Python API:

- reading geometry and frequencies from `opt_freq.out` while taking a higher-level electronic energy from a separate `single_point.out`;
- applying a frequency scale factor of 0.9838 consistently to ZPE, thermal internal energy, heat capacity, and entropy;
- enabling both the Grimme entropy correction and the Minenkov internal-energy/heat-capacity QRRHO treatment;
- selecting custom QRRHO reference wavenumbers and interpolation exponents;
- anchoring the enthalpy curve to a supplied $\Delta_f H^\circ(298.15\ \mathrm{K})$;
- fitting a continuous two-region NASA7 model that preserves $C_p$, $H$, and $S$ continuity at the midpoint temperature;
- recording the energy convention, reference pressure, computational protocol, and fit-audit information in a `SpeciesThermoArtifact`; and
- validating fit errors, continuity, units, and agreement with Cantera.

`SpeciesThermoArtifact` is an audit sidecar, not a replacement for the Cantera YAML file. Species used in one equilibrium pool must have compatible energy conventions, phases, reference temperatures, and reference pressures.

The minimal set of interfaces is:

```python
from ThermoCR.io import (
    read_molecule_data,
    read_orca_final_single_point_energy,
)
from ThermoCR.thermo import (
    EnergyConvention,
    SpeciesThermoArtifact,
    ThermoOptions,
    anchor_enthalpy_curve,
    assert_energy_conventions_compatible,
    fit_continuous_nasa7,
    scan_thermo,
)
from ThermoCR.export import (
    format_cantera_mechanism_yaml,
    format_cantera_species_yaml,
    format_cantera_yaml_thermo,
)
```

A calculation using the 0.9838 frequency scale factor and both QRRHO treatments can be configured as follows:

```python
options = ThermoOptions(
    pressure=100000.0,
    zpe_scale_factor=0.9838,
    internal_energy_scale_factor=0.9838,
    heat_capacity_scale_factor=0.9838,
    entropy_scale_factor=0.9838,
    use_minenkov_internal_energy=True,
    use_grimme_entropy=True,
    qrrho_reference_wavenumber_cm1=100.0,
    qrrho_interpolation_exponent=4.0,
    stationary_point_type="minimum",
)
```

The complete template from QM outputs to a continuous two-region NASA7 fit and Cantera YAML, together with parameter explanations and validation procedures, is provided in the [Chinese user guide](https://github.com/47-5/ThermoCR/blob/main/doc/tutorials.zh.md). The user must supply the actual QM outputs, molecular identity, rotational symmetry number, and enthalpy-of-formation anchor.

The ThermoCR CLI does not currently define a YAML configuration format covering the complete `ThermoOptions` interface. Production calculations should therefore use a short, version-controlled Python driver script.

## Python examples

Five example scripts are included:

```bash
python examples/01_read_qm_output.py
python examples/02_thermo_scan_and_fit.py
python examples/03_tst_vtst_rates.py
python examples/04_kinetics_fit.py
python examples/05_cantera_mechanism_export.py
```

Example outputs are written to `examples/output/`.

## Tests

Run the test suite from the repository root:

```bash
python -m unittest discover -s tests
```

Before generating a production batch of species data, verify at least the following:

1. All unit tests pass.
2. A minimum has no imaginary frequencies, while a transition state has exactly one.
3. The ORCA single-point energy agrees with the last `FINAL SINGLE POINT ENERGY`.
4. The same reference pressure is used during scanning, fitting, and export.
5. NASA7 fit errors and midpoint continuity satisfy the project requirements.
6. All artifacts in a batch use compatible energy conventions, phases, and reference pressures.
7. The exported YAML can be loaded by Cantera and reproduces $C_p/H/S$.

## Repository layout

| Path | Contents |
| --- | --- |
| `ThermoCR/io/` | Gaussian, ORCA, and generic QM-output readers |
| `ThermoCR/thermo/` | Partition functions, thermal corrections, QRRHO, enthalpy-of-formation anchoring, thermodynamic fitting, and audit artifacts |
| `ThermoCR/kinetics/` | TST, VTST, tunneling corrections, and kinetics fitting |
| `ThermoCR/export/` | Cantera YAML export |
| `ThermoCR/symmetry/` | Point groups, moments of inertia, and rotational symmetry numbers |
| `examples/` | Recommended Python example scripts |
| `example/` | Reference inputs and tabular data used by the examples |
| `tests/` | Unit tests and numerical regression tests |
| `README.zh.md` | Chinese README |
| `doc/tutorials.zh.md` | Complete Chinese-language guide |

## License

ThermoCR is distributed under the MIT License.
