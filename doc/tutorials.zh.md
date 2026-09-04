# ThermoCR 使用手册

## 1. 程序用途

ThermoCR 用量子化学计算结果构建分子热力学和反应动力学数据。最常用的任务是：

1. 从 Gaussian 或 ORCA 输出文件读取分子几何、振动频率和电子能；
2. 在给定温度与压力下计算理想气体的平动、转动、振动和电子贡献；
3. 扫描得到 $C_p(T)$、$H(T)$、$S(T)$ 和 $G(T)$；
4. 对低频振动采用 QRRHO 修正；
5. 用给定的 $\Delta_fH^\circ(298.15\ \mathrm K)$ 锚定生成焓曲线；
6. 拟合单区 NASA7、NASA9、Shomate，或连续双区 NASA7；
7. 导出 Cantera 可读取的 species、reaction 和完整 mechanism YAML；
8. 根据热力学表计算 TST/VTST 速率并拟合 Arrhenius 参数。

ThermoCR 负责从量子化学结果到热化学/动力学参数这一层。平衡、反应器和热沉计算应在 Cantera 或 `calculate_heat_sink` 等上层程序中完成。

ThermoCR 不会自动判断不同来源的数据是否处于同一电子结构水平，也不提供 Benson 基团贡献、BSR 参考反应构造或实验数据库检索。若需要生成焓锚点，使用者必须自行提供数值及其来源。

## 2. 安装

### 2.1 建议环境

项目声明支持 Python 3.8 及以上版本；建议使用 Python 3.11 创建独立环境：

```powershell
conda create -n thermocr python=3.11
conda activate thermocr
```

在仓库根目录安装：

```powershell
pip install .
```

若要边修改代码边使用：

```powershell
pip install -e .
```

Cantera 是可选依赖。若要执行 Cantera 导入验证和完整测试，可安装测试依赖：

```powershell
pip install -e ".[test]"
```

### 2.2 检查安装

```powershell
python -c "import ThermoCR; print(ThermoCR.__version__)"
thermocr --help
python -m unittest discover -s tests
```

若尚未安装控制台入口，可用下列形式代替 `thermocr`：

```powershell
python -m ThermoCR --help
```

## 3. 推荐使用的命名空间

```python
from ThermoCR.io import read_molecule_data
from ThermoCR.thermo import ThermoOptions, calculate_thermo, scan_thermo
from ThermoCR.thermo import anchor_enthalpy_curve, fit_continuous_nasa7
from ThermoCR.export import format_cantera_yaml_thermo
```

主要模块的职责如下。

| 模块 | 用途 |
| --- | --- |
| `ThermoCR.io` | 读取 Gaussian/ORCA 等量子化学输出 |
| `ThermoCR.thermo` | 热力学计算、温度扫描、生成焓锚定和多项式拟合 |
| `ThermoCR.symmetry` | 点群、主惯性矩、线性判断和转动对称数 |
| `ThermoCR.kinetics` | TST、VTST、隧穿修正和动力学拟合 |
| `ThermoCR.export` | 生成 Cantera YAML 文本 |
| `ThermoCR.simulation` | 简单动力学模拟接口 |

本文后续示例均从以上命名空间导入所需接口。

## 4. 热力学计算所需的量子化学数据

一个普通气相分子的热力学计算至少需要：

- 元素和原子坐标；
- 一个经过检查的稳定构型；
- 振动频率，单位为 $\mathrm{cm^{-1}}$；
- 电子能，单位为 Hartree；
- 正确的电荷、多重度和转动对称数；
- 与研究体系相符的频率缩放与低频处理方案。

对于极小值，频率分析必须与优化后的同一结构相对应，并且所有振动频率都应为有限正数。若出现虚频，应先检查优化是否收敛以及该振动模式的物理意义，再重新优化。不能通过删除虚频把一个未收敛结构伪装成极小值。

对于过渡态，必须明确使用 `stationary_point_type="transition_state"`。ThermoCR 要求恰好一个虚频，并只从振动热力学中排除该模式。

### 4.1 同一文件与分离文件

若优化、频率和最终电子能都来自同一个输出文件，可以直接读取：

```python
from ThermoCR.io import read_molecule_data

molecule = read_molecule_data("molecule.out")
```

更常见的高精度复合流程是：

- `opt_freq.out` 提供几何和频率；
- `single_point.out` 提供更高水平的单点电子能。

这种情况下应显式组合两部分数据，见第 5.3 节。

## 5. 读取量子化学输出

### 5.1 结构化读取

```python
from ThermoCR.io import read_molecule_data

molecule = read_molecule_data("opt_freq.out")

print(molecule.symbols)
print(molecule.coordinates)
print(molecule.atom_numbers)
print(molecule.frequencies)
print(molecule.electronic_energy)
print(molecule.electronic_energy_unit)
print(molecule.charge, molecule.multiplicity)
```

`read_molecule_data` 返回 `MoleculeData`。默认行为包括：

- 使用最后一组坐标；
- 使用最后一个 `cclib.scfenergies` 值；
- 将 `cclib` 的电子能转换成 Hartree；
- 保留频率的 $\mathrm{cm^{-1}}$ 单位；
- 对 Gaussian Link1 输出默认选取最后一个正常终止的任务。

`calculate_thermo` 要求 `molecule.electronic_energy_unit == "hartree"`。不要把 eV 或 J/mol 数值直接写入该字段。

### 5.2 Gaussian Link1 文件

查看指定 Link1 任务的电子能：

```powershell
thermocr qm-energy calculation.out --gaussian-job-index -1 --unit hartree
```

拆分所有正常终止的任务：

```powershell
thermocr split-link1 calculation.out split_jobs
```

选择一个任务写入独立文件：

```powershell
thermocr select-gaussian calculation.out selected.out --task-id 2 --mode select
```

Python API 中也可指定任务编号：

```python
molecule = read_molecule_data("calculation.out", gaussian_job_index=-1)
```

### 5.3 ORCA 单点能

ORCA 正式单点能应读取输出中的最后一行：

```text
FINAL SINGLE POINT ENERGY      -156.120526966514
```

专用读取器会搜索全部匹配项并返回最后一个值：

```python
from ThermoCR.io import (
    read_molecule_data,
    read_orca_final_single_point_energy,
)

molecule = read_molecule_data("opt_freq.out")
molecule.electronic_energy = read_orca_final_single_point_energy(
    "single_point.out"
)
molecule.electronic_energy_unit = "hartree"
```

命令行检查：

```powershell
thermocr orca-energy single_point.out
```

当几何/频率与高水平单点能来自不同文件时，必须使用上面的显式组合。还应确认：

- 两个计算对应同一构型和原子顺序；
- 单点计算的电荷和多重度正确；
- 单点能采用预定的电子结构方法、基组和色散方案；
- 输出确实正常终止；
- 选中的数值确实是最后一个 `FINAL SINGLE POINT ENERGY`。

## 6. `ThermoOptions` 参数

`ThermoOptions` 集中描述一次热力学计算的数值和物理选项。

```python
from ThermoCR.thermo import ThermoOptions

options = ThermoOptions(
    temperature=298.15,
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
    rotational_symmetry_number=1,
)
```

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `temperature` | `298.15` | 温度，K；扫描时由各格点覆盖 |
| `pressure` | `101325.0` | 理想气体参考压力，Pa |
| `zpe_scale_factor` | `1.0` | ZPE 中的振动频率缩放因子 |
| `internal_energy_scale_factor` | `1.0` | 振动内能/焓热校正中的频率缩放因子 |
| `heat_capacity_scale_factor` | `1.0` | 振动 $C_v/C_p$ 中的频率缩放因子 |
| `entropy_scale_factor` | `1.0` | 振动熵中的频率缩放因子 |
| `use_minenkov_internal_energy` | `False` | 对振动内能、焓和热容使用 Minenkov 低频插值 |
| `use_grimme_entropy` | `True` | 对振动熵使用 Grimme QRRHO 插值 |
| `qrrho_reference_wavenumber_cm1` | `100.0` | QRRHO 参考波数 $\nu_0$，$\mathrm{cm^{-1}}$ |
| `qrrho_interpolation_exponent` | `4.0` | QRRHO 权重指数 $\alpha$ |
| `point_group` | `None` | 覆盖自动点群判断，例如 `"C2v"` |
| `rotational_symmetry_number` | `None` | 直接指定转动对称数；优先级高于 `point_group` |
| `stationary_point_type` | `"minimum"` | `"minimum"` 或 `"transition_state"` |
| `electronic_energies` | `None` | 电子激发能级列表，Hartree |
| `electronic_degeneracies` | `None` | 与电子能级一一对应的简并度 |
| `ignore_trans_and_rot` | `False` | 是否排除平动和转动贡献 |
| `concentration` | `None` | 指定浓度，$\mathrm{mol\,m^{-3}}$，用于浓度态 Gibbs 修正 |

一般气相物种不应设置 `ignore_trans_and_rot=True`。该选项只适用于明确不包含平动/转动自由度的特殊模型。

## 7. 频率缩放因子 0.9838

若采用统一的 0.9838 频率校正协议，应将四个振动频率缩放参数都设为 0.9838：

```python
FREQUENCY_SCALE = 0.9838

options = ThermoOptions(
    pressure=100000.0,
    zpe_scale_factor=FREQUENCY_SCALE,
    internal_energy_scale_factor=FREQUENCY_SCALE,
    heat_capacity_scale_factor=FREQUENCY_SCALE,
    entropy_scale_factor=FREQUENCY_SCALE,
)
```

这四个参数之所以分别存在，是因为有些热化学协议会对 ZPE、热内能、热容和熵采用不同处理。若协议规定同一个频率因子，就应明确给出四次，避免某些项仍使用默认值 1.0。

这些参数缩放的是各热力学公式中的振动频率，而不是在计算结束后把 ZPE、$H$、$C_p$ 或 $S$ 整体乘以 0.9838。

## 8. QRRHO 低频处理

低频谐振子会给出过大的振动熵，并可能使柔性分子的 Gibbs 能对很小的频率变化异常敏感。ThermoCR 使用下式在 RRHO 和自由转子极限之间插值：

$$
w(\nu)=\frac{1}{1+\left(\nu_0/\nu\right)^\alpha}.
$$

其中默认 $\nu_0=100\ \mathrm{cm^{-1}}$，$\alpha=4$。高频时 $w\to1$，结果接近 RRHO；低频时 $w\to0$，结果逐渐转向自由转子近似。

ThermoCR 将低频处理拆成两个独立开关：

- `use_grimme_entropy=True`：对振动熵进行 Grimme QRRHO 插值；
- `use_minenkov_internal_energy=True`：对振动内能、焓以及热容进行 Minenkov 插值。

默认只启用 Grimme 熵修正。若正式协议要求热力学各项都采用 QRRHO，应显式启用两个开关：

```python
options = ThermoOptions(
    use_grimme_entropy=True,
    use_minenkov_internal_energy=True,
    qrrho_reference_wavenumber_cm1=100.0,
    qrrho_interpolation_exponent=4.0,
)
```

当前实现以未缩放的正频率计算插值权重。RRHO 端使用相应的缩放频率；Grimme 熵修正的自由转子端也使用熵缩放后的频率，而 Minenkov 内能和热容的自由转子极限不依赖频率。因此，频率缩放参数和 QRRHO 参数必须作为同一个计算协议一起记录。

若要得到纯 RRHO 结果：

```python
options = ThermoOptions(
    use_grimme_entropy=False,
    use_minenkov_internal_energy=False,
)
```

## 9. 极小值、过渡态与对称性

### 9.1 极小值

```python
options = ThermoOptions(stationary_point_type="minimum")
```

ThermoCR 要求所有频率均为有限正数。零频、虚频和非有限值都会直接报错。

### 9.2 一阶鞍点

```python
options = ThermoOptions(stationary_point_type="transition_state")
```

此时必须恰好有一个虚频，且其余频率均为有限正数。唯一的虚频会从振动配分函数中排除。

### 9.3 转动对称数

转动对称数会改变转动熵。自动点群判断可用于初步计算，但正式数据应核对分子几何和对称性。

```python
# 用点群覆盖自动判断
options = ThermoOptions(point_group="C2")

# 或直接指定转动对称数；这一项优先
options = ThermoOptions(rotational_symmetry_number=2)
```

## 10. 单个温压点的热力学计算

```python
from ThermoCR.io import read_molecule_data
from ThermoCR.thermo import ThermoOptions, calculate_thermo

molecule = read_molecule_data("opt_freq.out")

result = calculate_thermo(
    molecule,
    ThermoOptions(
        temperature=298.15,
        pressure=100000.0,
        zpe_scale_factor=0.9838,
        internal_energy_scale_factor=0.9838,
        heat_capacity_scale_factor=0.9838,
        entropy_scale_factor=0.9838,
        use_minenkov_internal_energy=True,
        use_grimme_entropy=True,
        rotational_symmetry_number=1,
    ),
)

print(result.heat_capacity_cp)       # J mol-1 K-1
print(result.enthalpy)               # J mol-1
print(result.entropy)                # J mol-1 K-1
print(result.gibbs_free_energy)      # J mol-1
```

`ThermoResult` 中常用字段如下。

| 字段 | 单位 | 含义 |
| --- | --- | --- |
| `temperature` | K | 温度 |
| `pressure` | Pa | 参考压力 |
| `heat_capacity_cv` | J mol$^{-1}$ K$^{-1}$ | 定容热容 |
| `heat_capacity_cp` | J mol$^{-1}$ K$^{-1}$ | 定压热容 |
| `entropy` | J mol$^{-1}$ K$^{-1}$ | 总熵 |
| `zpe` | J mol$^{-1}$ | 零点能 |
| `internal_energy_correction` | J mol$^{-1}$ | 内能热校正 |
| `enthalpy_correction` | J mol$^{-1}$ | 焓热校正 |
| `gibbs_energy_correction` | J mol$^{-1}$ | Gibbs 能热校正 |
| `electronic_energy` | J mol$^{-1}$ | 电子能转换后的摩尔量 |
| `internal_energy` | J mol$^{-1}$ | 电子能加内能校正 |
| `enthalpy` | J mol$^{-1}$ | 电子能加焓校正 |
| `gibbs_free_energy` | J mol$^{-1}$ | 电子能加 Gibbs 校正 |

注意：`MoleculeData.electronic_energy` 使用 Hartree，而 `ThermoResult.electronic_energy` 使用 J/mol。

## 11. 温度扫描

温度扫描返回一个 `pandas.DataFrame`，不会自动写文件：

```python
import numpy as np

from ThermoCR.io import read_molecule_data
from ThermoCR.thermo import ThermoOptions, scan_thermo

molecule = read_molecule_data("opt_freq.out")

temperatures = np.unique(
    np.concatenate(
        [
            np.arange(200.0, 2000.0 + 1.0, 25.0),
            np.array([298.15]),
        ]
    )
)

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
    rotational_symmetry_number=1,
)

table = scan_thermo(
    molecule,
    temperatures=temperatures,
    pressure=100000.0,
    options=options,
)
table.to_csv("thermo_scan.csv", index=False)
```

常用列为：

```text
temperature
pressure
heat_capacity_cp
enthalpy
entropy
gibbs_free_energy
```

表中能量使用 J/mol，热容和熵使用 J/(mol K)。`DataFrame.attrs["reference_pressure_pa"]` 也会记录参考压力；保存为 CSV 后属性不会保留，但 `pressure` 列仍在。

命令行可完成基本扫描：

```powershell
thermocr thermo scan opt_freq.out `
  --t-min 300 --t-max 1500 --n-points 49 `
  --pressure 100000 `
  --rotational-symmetry-number 1 `
  --stationary-point-type minimum `
  --output thermo_scan.csv
```

命令行扫描不暴露四个频率缩放因子、Minenkov 开关以及 QRRHO 阈值/指数。因此，采用 0.9838 和完整 QRRHO 的正式计算应使用 Python API。

## 12. 从绝对量子化学焓到生成焓曲线

`scan_thermo` 给出的 `enthalpy` 是电子能加热校正：

$$
H_{\mathrm{QM}}(T)=E_{\mathrm{elec}}+H_{\mathrm{corr}}(T).
$$

它不是标准生成焓，不能直接与实验生成焓或来自其他能量零点的 Cantera 物种混合。

若已知该物种的标准生成焓 $\Delta_fH^\circ(298.15\ \mathrm K)$，可保留量子化学计算得到的温度增量，同时将 298.15 K 的零点平移到目标生成焓：

$$
\Delta_fH^\circ(T)
=
H_{\mathrm{QM}}(T)
-H_{\mathrm{QM}}(298.15\ \mathrm K)
+\Delta_fH^\circ(298.15\ \mathrm K).
$$

代码如下：

```python
from ThermoCR.thermo import anchor_enthalpy_curve

hf298_j_mol = -74600.0

table["enthalpy_absolute_qm"] = table["enthalpy"]
table["gibbs_free_energy_absolute_qm"] = table["gibbs_free_energy"]
table["enthalpy"] = anchor_enthalpy_curve(
    temperatures=table["temperature"].to_numpy(),
    absolute_enthalpies=table["enthalpy"].to_numpy(),
    target_hf298=hf298_j_mol,
    reference_temperature=298.15,
)
table["gibbs_free_energy"] = (
    table["enthalpy"]
    - table["temperature"] * table["entropy"]
)
```

`anchor_enthalpy_curve` 只施加一个常数平移，因此所有 $H(T_2)-H(T_1)$ 保持不变。若后续还要使用或保存 `gibbs_free_energy` 列，应像上例一样按锚定后的 $H-TS$ 重新计算；否则该列仍保留原来的绝对 QM 能量零点。参考温度必须位于扫描温区内；函数允许线性插值，但后续连续双区 NASA7 拟合要求 298.15 K 在表中恰好出现一次，所以推荐一开始就把 298.15 K 加入温度格点。

生成焓锚点可以来自实验、经验证的高水平复合方法或使用者认可的其他来源。ThermoCR 不会计算或查询该锚点，也不会替使用者判断多个物种的参考能级是否一致。

## 13. 热力学多项式拟合

### 13.1 单区 NASA7、NASA9 和 Shomate

```python
from ThermoCR.thermo import fit_thermo_frame

fit_table = table.loc[
    table["temperature"].between(200.0, 1000.0)
].copy()

fit = fit_thermo_frame(
    fit_table,
    model_type="NASA7",
    weight_strategy="uniform",
    T_range=(200.0, 1000.0),
    reference_pressure_pa=100000.0,
)

print(fit.parameters)
print(fit.metrics)
```

`T_range` 写入拟合结果的有效温区，但不会替使用者筛选输入行，因此必须像上例一样先建立所需温区的子表。`fit_thermo_frame` 可识别 `scan_thermo` 的列名，也可识别 `T/K`、`Cp/(J/mol/K)`、`H/(J/mol)` 和 `S/(J/mol/K)` 这些列名。

命令行入口用于单区拟合：

```powershell
thermocr thermo fit thermo_scan_300_1500.csv `
  --model NASA7 `
  --weight-strategy uniform `
  --t-range 300 1500 `
  --reference-pressure-pa 100000 `
  --output species_thermo.yaml
```

这里的 `thermo_scan_300_1500.csv` 应已经只包含 300--1500 K 的数据。`--t-range` 声明输出模型的有效温区，不负责筛选 CSV。若输出扩展名为 `.json`，程序写出参数、协方差和误差指标；若为 `.yaml` 或 `.yml`，程序写出 Cantera thermo 片段。

### 13.2 连续双区 NASA7

宽温区推荐使用 `fit_continuous_nasa7`：

```python
from ThermoCR.thermo import fit_continuous_nasa7

fit = fit_continuous_nasa7(
    temperatures=table["temperature"].to_numpy(),
    heat_capacities=table["heat_capacity_cp"].to_numpy(),
    enthalpies=table["enthalpy"].to_numpy(),
    entropies=table["entropy"].to_numpy(),
    midpoint_temperature_k=1000.0,
    anchor_temperature_k=298.15,
    reference_pressure_pa=100000.0,
)
```

该拟合器采用如下结构：

1. 低温区和高温区各拟合五个 $C_p/R$ 系数；
2. 在中间温度精确约束 $C_p$ 连续；
3. 低温区的焓、熵积分常数由锚点值确定；
4. 高温区的积分常数由中间温度处的 $H$、$S$ 连续性确定。

因此拟合结果在中间温度处同时满足 $C_p$、$H$ 和 $S$ 连续。输入要求为：

- 温度必须有限、为正且互不重复；输入顺序可以任意，拟合器会排序；
- 中间温度必须严格位于总温区内部；
- 锚点温度必须位于低温区，并在输入表中恰好出现一次；
- 两个区域各至少有五个参与 $C_p$ 拟合的温度点；
- `reference_pressure_pa` 必须明确给出；
- 全部 $C_p$、$H$ 和 $S$ 数值必须有限。

结果对象提供：

```python
print(fit.low_coefficients)
print(fit.high_coefficients)
print(fit.metrics)
print(fit.continuity)
print(fit.diagnostics)

cp_fit, h_fit, s_fit = fit.predict(table["temperature"].to_numpy())
```

`metrics` 分别报告 $C_p$、$H$、$S$ 的 MAE、RMSE 和最大绝对误差；`continuity` 报告中点两侧的跳变量；`diagnostics` 包括拟合矩阵秩、条件数和 $C_p$ RMSE。

连续双区 NASA7 当前只通过 Python API 提供。`thermocr thermo fit` 是单区拟合入口，不能替代该函数。

## 14. 导出 Cantera YAML

### 14.1 导出双区 NASA7 thermo 块

```python
from ThermoCR.export import format_cantera_yaml_thermo

t_low, t_high = fit.temperature_range_k
t_mid = fit.midpoint_temperature_k

thermo_block = format_cantera_yaml_thermo(
    "NASA7",
    (t_low, t_mid, t_high),
    (fit.low_coefficients, fit.high_coefficients),
    reference_pressure_pa=fit.reference_pressure_pa,
)
```

必须把低温区和高温区系数按此顺序传入，并显式传递拟合时的参考压力。

### 14.2 构建 species 块

```python
from ThermoCR.export import format_cantera_species_yaml

species_block = format_cantera_species_yaml(
    "- name: fuel\n  composition: {C: 10, H: 16}\n",
    thermo_block,
)
```

### 14.3 构建完整机制

```python
from pathlib import Path
from ThermoCR.export import format_cantera_mechanism_yaml

mechanism_text = format_cantera_mechanism_yaml(
    [species_block],
    phase_name="gas",
    thermo_model="ideal-gas",
    kinetics_model="gas",
    state={"T": 300.0, "P": "1 bar"},
)

Path("fuel_mechanism.yaml").write_text(mechanism_text, encoding="utf-8")
```

若已有多个 species 或 reaction 片段，也可用命令行合并：

```powershell
thermocr cantera mechanism `
  --species-head fuel_head.yaml `
  --species-thermo fuel_thermo.yaml `
  --species other_species.yaml `
  --reaction reaction.yaml `
  --phase-name gas `
  --thermo-model ideal-gas `
  --output mechanism.yaml
```

`--species-head` 与 `--species-thermo` 可重复使用，但数量和顺序必须一致。`--species` 用于已经完整包含名称、组成和 thermo 的 species 片段。

## 15. 从 ORCA 输出到 Cantera 的完整代码模板

下面的脚本展示一个分离 `opt_freq.out` 与 `single_point.out` 的完整流程。它假定输入对应 CH4，并以 1 bar 下的示例生成焓锚点演示接口。实际使用时必须提供真实 QM 文件，并按数据来源核对物种名称、组成、转动对称数和生成焓锚点。

```python
from pathlib import Path

import numpy as np

from ThermoCR import __version__ as thermocr_version
from ThermoCR.export import (
    format_cantera_mechanism_yaml,
    format_cantera_species_yaml,
    format_cantera_yaml_thermo,
)
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

PRESSURE_PA = 100000.0
FREQUENCY_SCALE = 0.9838
HF298_J_MOL = -74600.0

molecule = read_molecule_data("opt_freq.out")
molecule.electronic_energy = read_orca_final_single_point_energy(
    "single_point.out"
)
molecule.electronic_energy_unit = "hartree"

temperatures = np.unique(
    np.concatenate(
        [np.arange(200.0, 2000.0 + 1.0, 25.0), [298.15]]
    )
)

options = ThermoOptions(
    pressure=PRESSURE_PA,
    zpe_scale_factor=FREQUENCY_SCALE,
    internal_energy_scale_factor=FREQUENCY_SCALE,
    heat_capacity_scale_factor=FREQUENCY_SCALE,
    entropy_scale_factor=FREQUENCY_SCALE,
    use_minenkov_internal_energy=True,
    use_grimme_entropy=True,
    qrrho_reference_wavenumber_cm1=100.0,
    qrrho_interpolation_exponent=4.0,
    stationary_point_type="minimum",
    rotational_symmetry_number=12,
)

table = scan_thermo(
    molecule,
    temperatures=temperatures,
    pressure=PRESSURE_PA,
    options=options,
)

table["enthalpy_absolute_qm"] = table["enthalpy"]
table["gibbs_free_energy_absolute_qm"] = table["gibbs_free_energy"]
table["enthalpy"] = anchor_enthalpy_curve(
    table["temperature"].to_numpy(),
    table["enthalpy"].to_numpy(),
    target_hf298=HF298_J_MOL,
    reference_temperature=298.15,
)
table["gibbs_free_energy"] = (
    table["enthalpy"]
    - table["temperature"] * table["entropy"]
)

fit = fit_continuous_nasa7(
    table["temperature"].to_numpy(),
    table["heat_capacity_cp"].to_numpy(),
    table["enthalpy"].to_numpy(),
    table["entropy"].to_numpy(),
    midpoint_temperature_k=1000.0,
    anchor_temperature_k=298.15,
    reference_pressure_pa=PRESSURE_PA,
)

artifact = SpeciesThermoArtifact.from_continuous_nasa7_fit(
    species_id="CH4",
    cantera_name="CH4",
    composition={"C": 1, "H": 4},
    energy_convention=EnergyConvention.FORMATION_ENTHALPY,
    fit_result=fit,
    formation_enthalpy_298_j_mol=HF298_J_MOL,
    temperature_grid_k=tuple(table["temperature"]),
    qm_source_bundle={
        "opt_freq": "opt_freq.out",
        "single_point": "single_point.out",
    },
    thermo_options={
        "frequency_scale": FREQUENCY_SCALE,
        "use_minenkov_internal_energy": True,
        "use_grimme_entropy": True,
        "qrrho_reference_wavenumber_cm1": 100.0,
        "qrrho_interpolation_exponent": 4.0,
        "rotational_symmetry_number": 12,
    },
    thermocr_version=thermocr_version,
)
assert_energy_conventions_compatible(
    [artifact],
    require_equilibrium_ready=True,
)

thermo_block = format_cantera_yaml_thermo(
    "NASA7",
    (
        fit.temperature_range_k[0],
        fit.midpoint_temperature_k,
        fit.temperature_range_k[1],
    ),
    (fit.low_coefficients, fit.high_coefficients),
    reference_pressure_pa=fit.reference_pressure_pa,
)

species_block = format_cantera_species_yaml(
    "- name: CH4\n  composition: {C: 1, H: 4}\n",
    thermo_block,
)
mechanism = format_cantera_mechanism_yaml([species_block])

table.to_csv("CH4_thermo_scan.csv", index=False)
artifact.write_json("CH4.thermo.json")
Path("CH4.yaml").write_text(mechanism, encoding="utf-8")
```

## 16. 计算参数的保存

正式计算建议把第 15 节脚本、输入文件、Cantera YAML 和 `*.thermo.json` 一同保存。`SpeciesThermoArtifact` 是版本化的审计 sidecar，记录能量零点、参考温度与压力、相态、温度格点、拟合与连续性指标、Cantera species 字典以及可选的来源哈希、ThermoCR 版本、Git commit 和依赖版本。它不替代供 Cantera 载入的 YAML。

批量计算完成后，应在送入 CCGE 前统一检查所有物种：

```python
from pathlib import Path

from ThermoCR.thermo import (
    SpeciesThermoArtifact,
    assert_energy_conventions_compatible,
)

artifacts = [
    SpeciesThermoArtifact.read_json(path)
    for path in sorted(Path("artifacts").glob("*.thermo.json"))
]
assert_energy_conventions_compatible(
    artifacts,
    require_equilibrium_ready=True,
)
```

该检查拒绝混用不同能量约定、相态、参考温度或参考压力；`thermal_increment_only` 也不能直接进入平衡计算。建议同时在每批 manifest 中记录驱动脚本哈希、ThermoCR Git commit 和生成时间。ThermoCR 没有定义独立的热化学 YAML 配置格式；如果研究项目需要用 YAML 管理参数，可在自己的驱动脚本中读取这些数值并构造 `ThermoOptions`，但该 YAML 属于项目输入，不是 ThermoCR 命令行接口。

## 17. Cantera 交叉验证

安装 Cantera 后，应检查 YAML 能否载入，并比较若干温度点的 $C_p$、$H$ 和 $S$。下列代码接续第 15 节，并使用其中生成的 `fit` 对象和 `CH4.yaml`：

```python
import cantera as ct
import numpy as np

gas = ct.Solution("CH4.yaml")
species_index = gas.species_index("CH4")

for temperature in (298.15, 500.0, 1000.0, 1500.0):
    gas.TP = temperature, 100000.0

    cp_j_mol_k = (
        gas.standard_cp_R[species_index]
        * ct.gas_constant
        / 1000.0
    )
    h_j_mol = (
        gas.standard_enthalpies_RT[species_index]
        * ct.gas_constant
        * temperature
        / 1000.0
    )
    s_j_mol_k = (
        gas.standard_entropies_R[species_index]
        * ct.gas_constant
        / 1000.0
    )

    cp_ref, h_ref, s_ref = fit.predict([temperature])
    np.testing.assert_allclose(cp_j_mol_k, cp_ref[0], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(h_j_mol, h_ref[0], rtol=0.0, atol=1e-3)
    np.testing.assert_allclose(s_j_mol_k, s_ref[0], rtol=0.0, atol=1e-6)
```

Cantera 的 `ct.gas_constant` 使用 J/(kmol K)，因此上例除以 1000 后再与 ThermoCR 的 J/(mol K) 或 J/mol 比较。

## 18. 动力学计算

### 18.1 TST

```python
import pandas as pd
from ThermoCR.kinetics import calculate_tst_rate_frame

reactant_1 = pd.read_csv("reactant_1_thermo.csv")
reactant_2 = pd.read_csv("reactant_2_thermo.csv")
transition_state = pd.read_csv("transition_state_thermo.csv")

rates = calculate_tst_rate_frame(
    transition_state,
    [reactant_1, reactant_2],
)
rates.to_csv("tst_rates.csv", index=False)
```

命令行：

```powershell
thermocr kinetics tst transition_state.csv `
  --reactant reactant_1.csv `
  --reactant reactant_2.csv `
  --output tst_rates.csv
```

### 18.2 VTST

```python
import pandas as pd
from ThermoCR.kinetics import calculate_vtst_rate_frame

path_1 = pd.read_csv("path_1_thermo.csv")
path_2 = pd.read_csv("path_2_thermo.csv")
path_3 = pd.read_csv("path_3_thermo.csv")

rates = calculate_vtst_rate_frame(
    [path_1, path_2, path_3],
    [reactant_1, reactant_2],
    path_names=["path_1", "path_2", "path_3"],
)
```

当前 VTST 接口在每个温度点计算各路径的 TST 速率，并取最小值作为 VTST 速率，同时输出 `limiting_path`。

### 18.3 Arrhenius 拟合和 reaction YAML

```python
from ThermoCR.kinetics import fit_kinetics_frame
from ThermoCR.export import format_cantera_reaction_yaml

kinetics_fit = fit_kinetics_frame(rates, model_type="Arrhenius")
parameters = kinetics_fit.named_parameters()

# fit_kinetics_frame 中 Ea 的单位是 J/mol。当前 YAML 导出器写入裸数值，
# 而 Cantera 默认把裸活化能解释为 J/kmol，因此这里乘以 1000。
ea_cantera_j_kmol = parameters["Ea"] * 1000.0

reaction_yaml = format_cantera_reaction_yaml(
    ["R1", "R2"],
    ["P"],
    A=parameters["A"],
    b=parameters["b"],
    Ea=ea_cantera_j_kmol,
)
```

这个换算只处理活化能。$A$ 的量纲取决于反应级数和拟合速率常数所用的浓度单位，必须在调用导出器前单独换算到目标 Cantera 机制的单位体系。当前 `format_cantera_reaction_yaml` 不写显式单位，因此在单位未核对前，不应把该片段直接用于生产机理。

## 19. CLI 与 Python API 的边界

命令行适合简单、单文件、少参数的任务；Python API 适合正式热化学生产流程。

| 任务 | CLI | Python API |
| --- | --- | --- |
| Gaussian Link1 拆分/选择 | 支持 | 支持 |
| 查看 Gaussian/通用 QM 电子能 | 支持 | 支持 |
| 精确读取 ORCA 最后一个 `FINAL SINGLE POINT ENERGY` | 支持 | 支持 |
| 单文件、默认热化学选项的温度扫描 | 支持 | 支持 |
| 独立 opt/freq 与单点能组合 | 不支持 | 支持 |
| 四项 0.9838 频率缩放 | 不支持 | 支持 |
| Minenkov 热内能/热容 QRRHO | 不支持 | 支持 |
| 自定义 QRRHO 阈值和指数 | 不支持 | 支持 |
| 生成焓锚定 | 不支持 | 支持 |
| 单区 NASA7/NASA9/Shomate 拟合 | 支持 | 支持 |
| 连续双区 NASA7 | 不支持 | 支持 |
| TST、VTST 和 Arrhenius 拟合 | 支持 | 支持 |
| 组合 Cantera YAML 片段 | 支持 | 支持 |

因此，正式的“高水平单点能 + 频率缩放 + QRRHO + 生成焓锚定 + 连续双区 NASA7”应写成一个短的 Python 驱动脚本，而不是堆叠多个命令行调用。

## 20. 科学与数值验证清单

### 20.1 量子化学输入

- 优化和频率计算正常终止；
- 极小值无虚频；过渡态恰好一个虚频；
- 单点能取自正确文件和正确方法；
- ORCA 单点能确实是最后一个 `FINAL SINGLE POINT ENERGY`；
- 几何、频率和单点能对应同一构型、电荷、多重度和原子顺序；
- 转动对称数经过人工核对。

### 20.2 热化学协议

- 温度和压力单位分别为 K 和 Pa；
- 四个频率缩放参数都符合既定协议；
- QRRHO 两个开关、$\nu_0$ 和指数已经明确；
- 参考压力在扫描、拟合和导出过程中保持一致；
- 298.15 K 在温度格点中恰好出现一次；
- 生成焓锚点的单位为 J/mol，来源已记录。

### 20.3 NASA7 拟合

- 中点位于温区内部；
- 两个区间各有至少五个拟合点；
- 检查 `fit.metrics` 的 MAE、RMSE 和最大绝对误差；
- 检查 `fit.continuity` 中 $C_p$、$H$、$S$ 跳变接近零；
- 检查 `fit.diagnostics` 的矩阵秩和条件数；
- 不在拟合温区外外推；
- 用 Cantera 在多个温度点交叉验证。

### 20.4 物种集合

- 同一 Cantera 相中的物种使用兼容的能量零点；
- 不把绝对量子化学能与实验生成焓直接混用；
- species 名称、元素组成和化学式一致；
- NASA7 温区覆盖上层计算所需的完整温度范围。

## 21. 常见错误

### 21.1 `minimum frequencies must be finite and positive`

原因：极小值输入包含虚频、零频或非有限频率。

处理：检查优化收敛和对应振动模式，重新优化后再计算。不要直接删除该频率。

### 21.2 `transition_state requires exactly one imaginary frequency`

原因：过渡态没有虚频、存在多个虚频，或其他频率非法。

处理：重新定位一阶鞍点并检查虚频是否对应目标反应坐标。

### 21.3 `molecule.electronic_energy must be in hartree`

原因：把 eV 或 J/mol 写入了 `MoleculeData.electronic_energy`，或调用 `read_molecule_data(..., return_hartree=False)` 后直接计算。

处理：使用默认 `return_hartree=True`，或显式提供 Hartree 数值。

### 21.4 ORCA 电子能不正确

原因：从优化/频率文件读到了低水平能量，或输出中有多个能量值。

处理：用 `read_orca_final_single_point_energy("single_point.out")`，并核对返回值对应最后一个 `FINAL SINGLE POINT ENERGY`。

### 21.5 `the fit table must contain anchor_temperature_k exactly once`

原因：温度表没有 298.15 K，或重复包含该温度。

处理：构造温度数组时追加 `[298.15]` 后使用 `np.unique`。

### 21.6 `each NASA7 region requires at least five temperature points`

原因：中点某一侧有效 $C_p$ 点不足。

处理：增加温度格点，或调整中点；不要用少量点拟合宽温区。

### 21.7 参考压力不一致

原因：扫描表的压力、拟合参数和导出压力不同，或一个表混有多个压力。

处理：每个标准压力独立生成热化学表，并在 `scan_thermo`、`fit_continuous_nasa7` 和 `format_cantera_yaml_thermo` 中传递同一个 Pa 数值。

### 21.8 Cantera 能载入，但焓差很大

常见原因：

- 未进行生成焓锚定；
- 把 kJ/mol 当成 J/mol；
- 低、高温 NASA7 系数顺序颠倒；
- Cantera 的 J/kmol 与 ThermoCR 的 J/mol 未换算；
- 物种混用了不同能量零点。

### 21.9 CLI 结果没有采用 0.9838 或完整 QRRHO

原因：`thermocr thermo scan` 当前只暴露基础扫描参数。

处理：使用 `ThermoOptions` 和 `scan_thermo` Python API。

## 22. 与 `calculate_heat_sink` 的衔接

推荐的数据流为：

```text
QM 输出
  -> ThermoCR 读取与热力学扫描
  -> 生成焓锚定
  -> 连续双区 NASA7
  -> Cantera species/mechanism YAML
  -> calculate_heat_sink 执行 CCGE、热沉分解和绘图
```

ThermoCR 的交付物是经过检查的 Cantera YAML。`calculate_heat_sink` 从该 YAML 开始，不需要再次包装量子化学读取、QRRHO 或 NASA7 拟合过程。

## 23. 最小 API 速查

```python
# 读取
from ThermoCR.io import read_molecule_data
from ThermoCR.io import read_orca_final_single_point_energy

# 热力学
from ThermoCR.thermo import EnergyConvention, SpeciesThermoArtifact
from ThermoCR.thermo import ThermoOptions
from ThermoCR.thermo import calculate_thermo, scan_thermo
from ThermoCR.thermo import anchor_enthalpy_curve
from ThermoCR.thermo import assert_energy_conventions_compatible
from ThermoCR.thermo import fit_thermo_frame, fit_continuous_nasa7
from ThermoCR.thermo import nasa7_values

# Cantera 导出
from ThermoCR.export import format_cantera_yaml_thermo
from ThermoCR.export import format_cantera_species_yaml
from ThermoCR.export import format_cantera_reaction_yaml
from ThermoCR.export import format_cantera_mechanism_yaml

# 动力学
from ThermoCR.kinetics import calculate_tst_rate_frame
from ThermoCR.kinetics import calculate_vtst_rate_frame
from ThermoCR.kinetics import fit_kinetics_frame
```

仓库中的可运行示例位于：

```text
examples/01_read_qm_output.py
examples/02_thermo_scan_and_fit.py
examples/03_tst_vtst_rates.py
examples/04_kinetics_fit.py
examples/05_cantera_mechanism_export.py
```

从仓库根目录运行：

```powershell
python examples/01_read_qm_output.py
python examples/02_thermo_scan_and_fit.py
python examples/03_tst_vtst_rates.py
python examples/04_kinetics_fit.py
python examples/05_cantera_mechanism_export.py
```
