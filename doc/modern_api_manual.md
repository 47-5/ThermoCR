# ThermoCR 现代接口手册

本文档面向重构后的 ThermoCR。推荐新项目优先使用 `ThermoCR.io`、
`ThermoCR.thermo`、`ThermoCR.kinetics`、`ThermoCR.export` 和
`ThermoCR.symmetry`。旧命名空间仍保留，用于兼容已有脚本。

## 安装

```bash
git clone https://github.com/47-5/ThermoCR.git
cd ThermoCR
pip install -e .
python -m unittest discover -s tests
```

安装后可以使用 Python API，也可以使用命令行入口 `thermocr`。

## 目录约定

- `example/`: 旧示例数据和回归参考数据。
- `examples/`: 新接口示例脚本。
- `examples/output/`: 示例脚本生成的输出目录。
- `tests/`: 单元测试和回归测试。
- `doc/modern_api_manual.md`: 本手册。

## QM 输出读取

```python
from ThermoCR.io import read_molecule_data, read_electronic_energy

molecule = read_molecule_data("example/CPD.out")
energy = read_electronic_energy("example/CPD.out", return_hartree=True)

print(molecule.symbols)
print(molecule.coordinates)
print(molecule.frequencies)
print(energy)
```

Gaussian `--Link1--` 多步输出可以先拆分或选择单步任务：

```bash
thermocr split-link1 calc.out split_jobs
thermocr select-gaussian calc.out selected.out --task-id 2 --mode select
```

读取电子能量时也可以指定 Link1 job：

```bash
thermocr qm-energy calc.out --gaussian-job-index -1
```

## 热力学扫描

```python
import numpy as np

from ThermoCR.io import read_molecule_data
from ThermoCR.thermo import ThermoOptions, scan_thermo

molecule = read_molecule_data("example/CPD.out")
df = scan_thermo(
    molecule,
    temperatures=np.linspace(300.0, 1500.0, 16),
    pressure=100000.0,
    options=ThermoOptions(pressure=100000.0),
)
df.to_csv("thermo.csv", index=False)
```

命令行等价入口：

```bash
thermocr thermo scan example/CPD.out --t-min 300 --t-max 1500 --n-points 16 --output thermo.csv
```

## Point Group 和对称数覆盖

默认情况下，ThermoCR 会根据几何结构自动检测点群并计算转动对称数。对于复杂分子或
自动检测不稳定的情况，可以手动覆盖：

```python
from ThermoCR.thermo import ThermoOptions, scan_thermo

options = ThermoOptions(point_group="C2v")
df = scan_thermo(molecule, temperatures=[298.15], options=options)

options = ThermoOptions(rotational_symmetry_number=2)
df = scan_thermo(molecule, temperatures=[298.15], options=options)
```

`rotational_symmetry_number` 的优先级高于 `point_group`。

命令行入口：

```bash
thermocr thermo scan example/CPD.out --point-group C2v --output thermo.csv
thermocr thermo scan example/CPD.out --rotational-symmetry-number 2 --output thermo.csv
```

## 热力学拟合

`fit_thermo_frame` 支持新版 `scan_thermo` 列名，也兼容旧 Excel 结果中的
`T/K`、`Cp/(J/mol/K)`、`H/(J/mol)` 和 `S/(J/mol/K)`。

```python
from ThermoCR.export import format_cantera_yaml_thermo
from ThermoCR.thermo import fit_thermo_frame

fit = fit_thermo_frame(df, model_type="NASA7", weight_strategy="uniform")
yaml_text = format_cantera_yaml_thermo(
    fit.model_type,
    fit.temperature_range,
    fit.parameters,
)
```

命令行入口：

```bash
thermocr thermo fit thermo.csv --model NASA7 --output CPD_thermo.yaml
```

## TST 和 VTST

结构化 TST 接口直接接收 DataFrame，不再强制从文件读写：

```python
import pandas as pd

from ThermoCR.kinetics import calculate_tst_rate_frame, calculate_vtst_rate_frame

reactant = pd.read_excel("example/QMthermoScan_CPD.xlsx")
ts = pd.read_excel("example/QMthermoScan_TS.xlsx")
path1 = pd.read_excel("example/QMthermoScan_01_02_path1_1.xlsx")
path2 = pd.read_excel("example/QMthermoScan_01_02_path2_1.xlsx")

tst = calculate_tst_rate_frame(ts, [reactant, reactant])
vtst = calculate_vtst_rate_frame(
    [path1, path2],
    [reactant, reactant],
    path_names=["path1", "path2"],
)
```

VTST 的当前口径是：对每个温度点，计算各路径的 TST 速率常数，然后取最小值作为
VTST 速率，并输出 `limiting_path`。

命令行入口：

```bash
thermocr kinetics tst thermo_ts.csv --reactant thermo_r1.csv --reactant thermo_r2.csv --output rates.csv
thermocr kinetics vtst path1.csv path2.csv --reactant thermo_r1.csv --reactant thermo_r2.csv --output vtst_rates.csv
```

## 动力学拟合

```python
from ThermoCR.kinetics import fit_kinetics_frame

fit = fit_kinetics_frame(vtst, model_type="Arrhenius")
print(fit.named_parameters())
```

命令行入口：

```bash
thermocr kinetics fit vtst_rates.csv --model Arrhenius --reactant-name CPD --reactant-name CPD --product-name DCPD --output reaction.yaml
```

## Cantera YAML 导出

Species、reaction 和 mechanism 都可以从新导出接口生成：

```python
from ThermoCR.export import (
    format_cantera_mechanism_yaml,
    format_cantera_reaction_yaml,
    format_cantera_species_yaml,
    format_cantera_yaml_thermo,
)

species = format_cantera_species_yaml(
    "- name: CPD\n  composition: {C: 5, H: 6}\n",
    format_cantera_yaml_thermo("NASA7", (300.0, 1500.0), fit.parameters),
)
reaction = format_cantera_reaction_yaml(
    ["CPD", "CPD"],
    ["DCPD"],
    A=1.0e10,
    b=0.0,
    Ea=50000.0,
)
mechanism = format_cantera_mechanism_yaml([species], reaction_blocks=[reaction])
```

命令行入口：

```bash
thermocr cantera mechanism --species-head CPD_head.yaml --species-thermo CPD_thermo.yaml --reaction reaction.yaml --output mechanism.yaml
```

## 旧接口迁移表

| 旧接口 | 推荐新接口 |
| --- | --- |
| `ThermoCR.QMthermo.qm_thermo` | `ThermoCR.thermo.calculate_thermo` 或 `scan_thermo` |
| `ThermoCR.QMthermo.qm_thermo_scan` | `ThermoCR.thermo.scan_thermo` |
| `ThermoCR.QMkinetics.k_TST_scan` | `ThermoCR.kinetics.calculate_tst_rate_frame` |
| `ThermoCR.QMkinetics.k_VTST_scan` | `ThermoCR.kinetics.calculate_vtst_rate_frame` |
| `ThermoCR.tools.about_cantera` | `ThermoCR.export` |
| `ThermoCR.tools.utils.get_point_group` | `ThermoCR.symmetry.detect_point_group` |

## 建议工作流

1. 用 `ThermoCR.io` 读取 QM 输出。
2. 用 `ThermoCR.thermo.scan_thermo` 扫描热力学表。
3. 用 `ThermoCR.thermo.fit_thermo_frame` 拟合 NASA/Shomate 参数。
4. 用 `ThermoCR.kinetics.calculate_tst_rate_frame` 或
   `calculate_vtst_rate_frame` 计算速率表。
5. 用 `ThermoCR.kinetics.fit_kinetics_frame` 拟合 Arrhenius 参数。
6. 用 `ThermoCR.export` 生成 Cantera YAML。
7. 在 `calculate_heat_sink` 等上层项目中调用这些新接口，不再依赖旧脚本式入口。

