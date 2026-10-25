# M7-C 审计副本

本目录保存 M7-C 表格所依赖的紧凑结果副本，便于在论文提交时核对数值与追溯来源；完整文件仍在 `outputs/metrics/m7c/`。

| 文件 | 内容 |
| --- | --- |
| `table_c1_overall.csv` | M4/M6-B/M7-A 三场景正式性能 |
| `table_c1_gains.csv` | DoseSyn 相对各内部基线的逐场景提升 |
| `table_c1_external_common.csv` | comboKR 可比公共测试矩阵子集 |
| `table_c2_m6a_ablation.csv` | M6-A 控制变量消融（T0–T4） |
| `table_c2_m6b_film.csv` | M6-B B1 vs B5 正式结果 |
| `table_c3_center_holdout.csv` | 中心 2×2 剂量留出 |
| `table_c4_sparsity.csv` | 75%/50%/25% 观测稀疏度 |
| `table_c4_boundary.csv` | 高剂量边界外推 |
| `table_c5_symmetry.csv` | 药物顺序交换误差 |
| `table_c7_complexity.csv` | 参数量、训练与推理开销 |
| `symmetry_runs.csv` | 逐场景对称性统计 |
| `inference_timing.json` | CPU 推理计时（4 线程） |
| `surface_cases.csv` / `.json` | 曲面案例选取规则与误差 |
| `m7c_summary.json` | 章节/报告引用的全部数值汇总 |

复现：

```bash
.venv/bin/python scripts/21_build_m7c_tables.py
.venv/bin/python scripts/22_m7c_symmetry.py
.venv/bin/python scripts/22b_measure_inference.py
.venv/bin/python scripts/25_write_chapter4.py
```

数据来源：M4 `outputs/runs/m4`、M6-A `outputs/runs/m6a`、M6-B `outputs/runs/m6b`、M7-A `outputs/metrics/m7a`、M7-B `outputs/runs/m7b`。M0–M7-B 结果未被修改或覆盖。
