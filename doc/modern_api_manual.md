# ThermoCR 现代接口手册（兼容入口）

原现代接口手册已经合并进更完整的[中文使用教程](tutorials.zh.md)。本文件保留在原路径，用于兼容已有书签和外部链接。

新项目优先使用 `ThermoCR.io`、`ThermoCR.thermo`、`ThermoCR.kinetics`、`ThermoCR.export` 和 `ThermoCR.symmetry`。旧命名空间仍保留，用于兼容已有脚本。

## 旧接口迁移表

| 旧接口 | 推荐新接口 |
| --- | --- |
| `ThermoCR.QMthermo.qm_thermo` | `ThermoCR.thermo.calculate_thermo` 或 `scan_thermo` |
| `ThermoCR.QMthermo.qm_thermo_scan` | `ThermoCR.thermo.scan_thermo` |
| `ThermoCR.QMkinetics.k_TST_scan` | `ThermoCR.kinetics.calculate_tst_rate_frame` |
| `ThermoCR.QMkinetics.k_VTST_scan` | `ThermoCR.kinetics.calculate_vtst_rate_frame` |
| `ThermoCR.tools.about_cantera` | `ThermoCR.export` |
| `ThermoCR.tools.utils.get_point_group` | `ThermoCR.symmetry.detect_point_group` |

安装、QM 输出读取、RRHO/QRRHO、生成焓锚定、连续双区 NASA7、热化学 artifact、Cantera 导出和验证流程均见[中文使用教程](tutorials.zh.md)。
