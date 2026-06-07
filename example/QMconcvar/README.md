# QMconcvar Example

This directory contains a minimal reaction-system example for the reaction simulation module.

Recommended modern import:

```python
from ThermoCR.simulation import ChemicalKineticsSimulator

simulator = ChemicalKineticsSimulator("example/QMconcvar/reaction_system.yaml")
result = simulator.simulate()
simulator.export_result("qmconcvar_result.csv")
```

The legacy import remains supported:

```python
from ThermoCR.QMconcvar import ChemicalKineticsSimulator
```