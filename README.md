# ThermoCR

ThermoCR 是一个从量子化学计算结果生成分子热力学数据和反应速率参数的 Python 程序。它可以：

- 读取 Gaussian 和 ORCA 输出；
- 计算理想气体 $C_p(T)$、$H(T)$、$S(T)$ 和 $G(T)$；
- 处理振动频率缩放、QRRHO 低频修正、点群和转动对称数；
- 拟合 NASA7、NASA9、Shomate 以及连续双区 NASA7；
- 生成带能量零点、参考压力、拟合指标和来源信息的热化学 JSON sidecar；
- 计算 TST/VTST 速率并拟合 Arrhenius 参数；
- 导出 Cantera species、reaction 和完整 mechanism YAML。

ThermoCR 的输出可交给 Cantera、`calculate_heat_sink` 或其他平衡与反应器程序继续使用。ThermoCR 不提供 Benson 基团贡献、BSR 参考反应构造或实验热化学数据库检索；生成焓锚点需要由使用者提供。

完整中文手册见 [`doc/tutorials.zh.md`](doc/tutorials.zh.md)。

## 安装

建议使用 Python 3.11 和独立 Conda 环境：

```bash
conda create -n thermocr python=3.11
conda activate thermocr
git clone https://github.com/47-5/ThermoCR.git
cd ThermoCR
pip install .
```

若要修改源码并立即生效：

```bash
pip install -e .
```

若需要 Cantera 交叉验证和完整测试：

```bash
pip install -e ".[test]"
```

检查安装：

```bash
python -c "import ThermoCR; print(ThermoCR.__version__)"
thermocr --help
```

控制台入口不可用时，可将 `thermocr` 替换为 `python -m ThermoCR`。

## 五分钟 CLI 示例

以下命令使用仓库自带的 `example/CPD.out`，适合检查安装和熟悉文件流。

### 1. 查看电子能

```bash
thermocr qm-energy example/CPD.out --unit hartree
```

ORCA 单点计算可用专用命令读取最后一个 `FINAL SINGLE POINT ENERGY`：

```bash
thermocr orca-energy path/to/single_point.out
```

### 2. 扫描热力学表

```bash
thermocr thermo scan example/CPD.out --t-min 300 --t-max 1500 --n-points 49 --pressure 100000 --output CPD_thermo_scan.csv
```

输出表包含温度、压力、$C_v$、$C_p$、熵、ZPE、热校正和总 $U/H/G$。热量单位为 J/mol，热容和熵单位为 J/(mol K)。

### 3. 拟合一个单区 NASA7

```bash
thermocr thermo fit CPD_thermo_scan.csv --model NASA7 --weight-strategy uniform --t-range 300 1500 --reference-pressure-pa 100000 --output CPD_thermo.yaml
```

`CPD_thermo.yaml` 是 Cantera thermo 片段。若要把它组合成完整机制，先建立物种头文件 `CPD_head.yaml`：

```yaml
- name: CPD
  composition: {C: 5, H: 6}
```

再运行：

```bash
thermocr cantera mechanism --species-head CPD_head.yaml --species-thermo CPD_thermo.yaml --output mechanism.yaml
```

这组 CLI 命令用于快速检查和简单单文件任务，不代表正式高精度热化学协议。

## 正式热化学计算

下列任务需要 Python API：

- 从 `opt_freq.out` 读取几何和频率，再从独立 `single_point.out` 读取高水平电子能；
- 对 ZPE、热内能、热容和熵统一采用 0.9838 频率缩放；
- 同时启用 Grimme 熵修正和 Minenkov 内能/热容 QRRHO；
- 自定义 QRRHO 参考波数和插值指数；
- 用给定的 $\Delta_fH^\circ(298.15\ \mathrm K)$ 锚定生成焓曲线；
- 拟合在中间温度处保证 $C_p$、$H$、$S$ 连续的双区 NASA7；
- 用 `SpeciesThermoArtifact` 保存能量约定、参考压力、计算协议和拟合审计信息；
- 对拟合结果执行误差、连续性、单位和 Cantera 交叉验证。

最小接口如下：

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

0.9838 与完整 QRRHO 的设置形式为：

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

从 QM 输出到连续双区 NASA7 和 Cantera YAML 的完整代码模板、参数说明及验证方法见 [`doc/tutorials.zh.md`](doc/tutorials.zh.md)。该模板需要使用者提供实际 QM 输出、分子身份、转动对称数和生成焓锚点。ThermoCR CLI 当前不定义完整 `ThermoOptions` 的 YAML 配置格式；正式计算直接使用一个短 Python 驱动脚本即可。

## Python 示例

仓库提供五个示例脚本：

```bash
python examples/01_read_qm_output.py
python examples/02_thermo_scan_and_fit.py
python examples/03_tst_vtst_rates.py
python examples/04_kinetics_fit.py
python examples/05_cantera_mechanism_export.py
```

示例输出写入 `examples/output/`。

## 测试

在仓库根目录运行：

```bash
python -m unittest discover -s tests
```

正式生成一批物种数据前，建议至少完成：

1. 全部单元测试通过；
2. 极小值无虚频，过渡态恰好一个虚频；
3. ORCA 单点能与最后一个 `FINAL SINGLE POINT ENERGY` 一致；
4. 参考压力在扫描、拟合和导出中一致；
5. NASA7 的拟合误差和中点连续性满足要求；
6. 同批 artifact 的能量约定、相态和参考压力相容；
7. 导出的 YAML 能被 Cantera 载入并重现 $C_p/H/S$。

## 仓库导航

| 路径 | 内容 |
| --- | --- |
| `ThermoCR/io/` | Gaussian、ORCA 和通用 QM 输出读取 |
| `ThermoCR/thermo/` | 配分函数、热校正、QRRHO、生成焓锚定、热力学拟合和审计 artifact |
| `ThermoCR/kinetics/` | TST、VTST、隧穿修正和动力学拟合 |
| `ThermoCR/export/` | Cantera YAML 导出 |
| `ThermoCR/symmetry/` | 点群、惯性矩和转动对称数 |
| `examples/` | 推荐的 Python 示例脚本 |
| `example/` | 示例所用的参考输入和表格数据 |
| `tests/` | 单元测试与数值回归测试 |
| `doc/tutorials.zh.md` | 完整中文使用手册 |

## 许可证

ThermoCR 使用 MIT License。
