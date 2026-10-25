# DoseSyn M7-C 报告：第四章实验完善与可视化

**结论：M7-C PASS。**

## 1. 完成范围

- C1 总体性能对比：内部 B0–B5 与外部 comboKR 2.0、DeepSynBa adapted。
- C2 模块消融：M6-A 控制变量（T0–T4）、M6-B B1 vs B5、M5 组件附录。
- C3 中心 2×2 剂量位置留出（11 split）。
- C4 50% 稀疏观测；25% 与边界结果作为补充，不用于正面结论。
- C5 冻结 B5 的交换对称性（S1/S2/S3 全部测试矩阵）。
- C6 曲面案例：分位数规则选取良好/中位/困难案例 + 中心插值案例。
- C7 参数量、训练时间、CPU 推理耗时。
- C8 第四章中文正文草稿（4.1–4.4，含 4.3.1–4.3.8）。

## 2. 核心数值

| 模型 | S1 | S2 | S3 | 等权平均 |
| --- | ---: | ---: | ---: | ---: |
| DoseSyn B5 | 0.0650 | 0.0768 | 0.1937 | 0.1118 |
| M4 B3 GINE | 0.0795 | 0.0897 | 0.1994 | 0.1228 |
| M4 B2 MLP | 0.0764 | 0.1022 | 0.2033 | 0.1273 |
| M6-B B1 T3 | 0.0793 | 0.1085 | 0.1966 | 0.1281 |
| comboKR 2.0 | 0.1341 | 0.1361 | — | 0.1351 (S1/S2) |
| DeepSynBa adapted | 0.2353 | 0.2670 | 0.2754 | 0.2592 |

- B5 相对 GINE 平均降低 9.0%；S1/S2/S3 分别 18.2%/14.4%/2.8%。
- 中心留出：B5 S1/S2 为 0.0702/0.0811，相对 GINE 降低 8.3%/9.6%。
- 50% 观测：B5 S1/S2/S3 0.0710/0.0838/0.2091；25% 极端稀疏下 B5 弱于 GINE，已如实保留。
- 交换对称性：三场景最大绝对误差均为 0.0e+00。
- 参数/推理：B5 535,682；CPU 0.18 ms/矩阵（B3 0.12）。

## 3. 交付物

- `docs/paper/chapter4_experiments.md`（中文正文草稿）
- `docs/paper/tables/table4_*.md`（10 个表）
- `outputs/figures/m7c/`（13 个主图，SVG/PDF/PNG）
- `outputs/metrics/m7c/`（CSV + JSON + 对称性逐矩阵结果）
- `docs/audit/m7c/`（紧凑审计副本）

## 4. 复现命令

```bash
.venv/bin/python scripts/21_build_m7c_tables.py
.venv/bin/python scripts/22b_measure_inference.py
.venv/bin/python scripts/22_m7c_symmetry.py
/tmp/m7venv/bin/python scripts/23_make_m7c_figures.py
/tmp/m7venv/bin/python scripts/24_plot_m7c_surfaces.py
/tmp/m7venv/bin/python scripts/25_write_chapter4.py
.venv/bin/python -m pytest -q tests
```

## 5. 限制

1. S3 上 B5 仅略优于 GINE（2.8%），不能宣称全面领先。
2. DeepSynBa 为 adapted 版本，不属忠实复现。
3. M4 部分历史训练日志的列名与实际指标不一致，本章统一使用 `compute_metrics` 重算结果。
4. 25% 极端稀疏与高剂量边界结果不作为 B5 优势证据。
5. 图表数值全部由脚本生成；正文数值与表格由同一脚本读取，避免手抄误差。

## 6. Git

M7-C 独立 commit；详见 `git log -1` 与最终交付说明。
