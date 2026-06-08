# ThermoCR Modern API Examples

These examples use the modern public API and CLI added during the refactor.
Run them from the repository root after installing ThermoCR in editable mode:

```bash
pip install -e .
python examples/01_read_qm_output.py
python examples/02_thermo_scan_and_fit.py
python examples/03_tst_vtst_rates.py
python examples/04_kinetics_fit.py
python examples/05_cantera_mechanism_export.py
```

The scripts read reference data from `example/` and write generated artifacts to
`examples/output/`. The legacy `example/` directory is kept as the data source
and backward-compatibility reference.
