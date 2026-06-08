# ThermoCR

This repository corresponds to the paper:

ThermoCR: A Python program for calculating molecular thermochemistry properties and reaction rate constants

# Install
Download the source code:

`git clone https://github.com/47-5/ThermoCR.git`

One may create an environment using Anaconda:

`conda create -n ThermoCR_env python=3.11`

`conda activate ThermoCR_env`

Change the working directory to the code files of ThermoCR:

`cd ThermoCR`

Install from the repository root:

`pip install .`

ThermoCR uses `pyproject.toml` as the package configuration. Legacy `python setup.py install` and `python setup.py develop` workflows are not supported.

For development, install the package in editable mode:

`pip install -e .`

# Public API
ThermoCR now provides modern, lower-case namespaces for common workflows:

```python
from ThermoCR.elements import atomic_mass, atomic_masses
from ThermoCR.io import read_qm_output, read_molecule_data, read_atom_coordinates
from ThermoCR.thermo import calculate_thermo, scan_thermo, fit_thermo_frame, ThermoOptions
from ThermoCR.kinetics import calculate_tst_rate_frame, k_TST, k_VTST
from ThermoCR.simulation import ChemicalKineticsSimulator
from ThermoCR.export import format_cantera_mechanism_yaml, make_cantera_specie_name_yaml
```

Legacy imports such as `ThermoCR.QMthermo`, `ThermoCR.QMkinetics`, `ThermoCR.QMconcvar`, and `ThermoCR.tools` remain available for existing scripts.

# Command Line
After installation, ThermoCR provides a small command-line entry point for common file utilities:

```bash
thermocr split-link1 example/CPD.out split_jobs
thermocr select-gaussian example/CPD.out selected.out --task-id 2 --mode select
thermocr qm-energy example/CPD.out --gaussian-job-index -1
thermocr thermo scan example/CPD.out --t-min 300 --t-max 3000 --n-points 100 --output thermo.csv
thermocr thermo scan example/CPD.out --point-group C2v --output thermo_with_symmetry_override.csv
thermocr thermo fit thermo.csv --model NASA7 --output CPD_thermo.yaml
thermocr kinetics tst thermo_ts.csv --reactant thermo_r1.csv --reactant thermo_r2.csv --output rates.csv
thermocr kinetics vtst path1.csv path2.csv --reactant thermo_r1.csv --reactant thermo_r2.csv --output vtst_rates.csv
thermocr kinetics fit rates.csv --model Arrhenius --reactant-name CPD --reactant-name CPD --product-name DCPD --output reaction.yaml
thermocr cantera mechanism --species-head CPD_head.yaml --species-thermo CPD_thermo.yaml --reaction reaction.yaml --output mechanism.yaml
thermocr orca-energy path/to/orca.out
```

The same commands are available through `python -m ThermoCR` before installing the console script.
# Get Started Quickly

We have prepared some reference examples for new users. Users can view the `examples.ipynb` file stored in the `example` directory using Jupyter Notebook. All the necessary input files for `examples.ipynb` are also stored in the `example` directory.

The documents are saved in the `doc` directory. Users can view them to obtain the API descriptions of all the functions in ThermoCR.

# Note
The point group implementation is bundled under `ThermoCR.pointgroup`; the recommended public symmetry helpers are available from `ThermoCR.symmetry`.

# Development
Run the test suite from the repository root:

`python -m unittest discover -s tests`
