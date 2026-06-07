# QMconcvar Example

This directory contains a minimal reaction-system example for the `ThermoCR.QMconcvar` reaction simulation module.

```python
from ThermoCR.QMconcvar import ChemicalKineticsSimulator

simulator = ChemicalKineticsSimulator("example/QMconcvar/reaction_system.yaml")
results = simulator.simulate()
simulator.export_result("qmconcvar_result.csv")
```

Relative paths in `reaction_system.yaml` are resolved against this directory.
