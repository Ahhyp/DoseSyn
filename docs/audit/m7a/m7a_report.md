# M7-A 外部 Baseline 与 DoseSyn 统一协议对比

所有数值均为本项目在 M3 冻结划分上重新得到的 matrix-macro RMSE；M4/M6-B 为归档复用，新运行标注来源。

## 1. 主表（matrix-macro RMSE，越低越好）

| 模型 | S1 | S2 | S3 | 等权平均 | 来源 |
| --- | ---: | ---: | ---: | ---: | --- |
| M4 B0 Mean | 0.3060 | 0.3022 | 0.3066 | 0.3049 | 归档 M4 |
| M4 B1 Tree | 0.2467 | 0.2497 | 0.2819 | 0.2594 | 归档 M4 |
| M4 B2 MLP | 0.0764 | 0.1022 | 0.2033 | 0.1273 | 归档 M4 |
| M4 B3 GINE | 0.0795 | 0.0897 | 0.1994 | 0.1228 | 归档 M4 |
| M6-B B1 (T3) | 0.0793 | 0.1085 | 0.1966 | 0.1281 | 归档 M6-B |
| M6-B B5 (T3+FiLM) | 0.0650 | 0.0768 | 0.1937 | 0.1118 | 归档 M6-B |
| comboKR 2.0 | 0.1341 | 0.1361 | — | 0.1351 | 本阶段新运行 |
| DeepSynBa (adapted)（补充） | 0.2353 | 0.2670 | 0.2754 | 0.2592 | 本阶段新运行（适配） |

> comboKR 的 S3 因协议冲突不运行；其“平均”只对 S1/S2 两个场景等权，不能与三场景平均直接比较。DeepSynBa 只作为 adapted supplementary，不进入主排名。

## 2. 配对差异（相对 M6-B B5）

| 模型 | 场景 | 可比 split 数 | 平均 ΔRMSE（负=更好） | 优于 DoseSyn 的 split 数 |
| --- | --- | ---: | ---: | ---: |
| DeepSynBa (adapted) | S1 | 3 | +0.1703 | 0/3 |
| DeepSynBa (adapted) | S2 | 3 | +0.1903 | 0/3 |
| DeepSynBa (adapted) | S3 | 5 | +0.0816 | 0/5 |
| M4 B0 Mean | S1 | 3 | +0.2411 | 0/3 |
| M4 B0 Mean | S2 | 3 | +0.2254 | 0/3 |
| M4 B0 Mean | S3 | 5 | +0.1128 | 0/5 |
| M4 B1 Tree | S1 | 3 | +0.1817 | 0/3 |
| M4 B1 Tree | S2 | 3 | +0.1729 | 0/3 |
| M4 B1 Tree | S3 | 5 | +0.0882 | 0/5 |
| M4 B2 MLP | S1 | 3 | +0.0114 | 0/3 |
| M4 B2 MLP | S2 | 3 | +0.0255 | 0/3 |
| M4 B2 MLP | S3 | 5 | +0.0095 | 0/5 |
| M4 B3 GINE | S1 | 3 | +0.0145 | 0/3 |
| M4 B3 GINE | S2 | 3 | +0.0129 | 0/3 |
| M4 B3 GINE | S3 | 5 | +0.0056 | 1/5 |
| M6-B B1 (T3) | S1 | 3 | +0.0143 | 0/3 |
| M6-B B1 (T3) | S2 | 3 | +0.0318 | 0/3 |
| M6-B B1 (T3) | S3 | 5 | +0.0029 | 2/5 |
| comboKR 2.0 | S1 | 3 | +0.0691 | 0/3 |
| comboKR 2.0 | S2 | 3 | +0.0594 | 0/3 |

## 3. 关键解释

- DeepSynBa 使用 Morgan 指纹替代 MoLFormer、CCLE 替代 NCI-60、X/X0 曲面替代 SynBa 参数，属于 **adapted supplementary**，不进入主排名。
- comboKR 在 S1/S2 使用官方 BRAID 参数与 RBF 核；S3 因 Hill 参数含测试细胞单药信息，按协议不运行，表中记为 —。
- 所有方法读取同一冻结 split 的测试矩阵；配对差异按 (scenario, split) 对齐。

## 4. 资源与预算

| 模型 | 参数量 | 单 split 平均耗时 (s) |
| --- | ---: | ---: |
| M4 B0 Mean | 1 | — |
| M4 B1 Tree | 24400 | — |
| M4 B2 MLP | 65153 | — |
| M4 B3 GINE | 284545 | — |
| M6-B B1 (T3) | 498818 | 598.3 |
| M6-B B5 (T3+FiLM) | 535682 | 452.3 |
| comboKR 2.0 | — | 91.4 |
| DeepSynBa (adapted) | 11504137 | 46.2 |

## 5. 产出

- `outputs/metrics/m7a/external_baseline_runs.csv`
- `outputs/metrics/m7a/external_baseline_summary.csv`
- `outputs/metrics/m7a/external_paired_deltas.csv`
- `outputs/metrics/m7a/main_comparison.csv`
- `outputs/figures/m7a/macro_rmse_comparison.svg`
## 6. comboKR 可用矩阵子集上的同口径对比

comboKR 官方数据缺少 MK-8776，S1/S2 测试集比冻结测试集少 46–47 个矩阵（约 1.7%）。为避免“矩阵组成差异”影响结论，本节只保留 comboKR 能预测的矩阵，并在**完全相同的 matrix_id 集合**上重算所有模型。

| 模型 | S1（n=3 seeds） | S2（n=3 seeds） |
| --- | ---: | ---: |
| M4 B2 MLP | 0.07600 | 0.10041 |
| M4 B3 GINE | 0.07910 | 0.08858 |
| M6-B B1 (T3) | 0.07901 | 0.10750 |
| M6-B B5 (T3+FiLM) | **0.06459** | **0.07561** |
| comboKR 2.0 | 0.13413 | 0.13615 |

结论不变：在同一批矩阵上，B5 仍显著优于 comboKR（S1 低 51.9%，S2 低 44.5%）；差异不是由 MK-8776 缺失造成的。

## 7. 最终报告必须回答的十个问题

1. **哪些外部模型成功复现？** comboKR 2.0 使用官方代码、官方 BRAID 拟合与 RBF 核，在 S1/S2 上完成了可信复现；DeepSynBa 完成了**适配版**重实现，但不是官方 MoLFormer/NCI-60/SynBa 目标的忠实复现。
2. **哪些无法复现，原因？** DeepSynBa 的忠实复现需要官方 104 药 MoLFormer 嵌入与 NCI-60 数据，本项目只有 O'Neil 36 药与 CCLE 表达，且官方目标是 SynBa/协同参数而非 X/X0；DD-PRiSM 未找到公开可运行代码。
3. **DoseSyn 处于什么水平？** S1 0.0650、S2 0.0768、S3 0.1937，三个场景均为本表主排名第一。
4. **哪些场景 DoseSyn 优于外部方法？** S1/S2 明显优于 comboKR（−52%/−44% 相对误差）和 adapted DeepSynBa；S3 优于 adapted DeepSynBa。
5. **哪些场景不如外部方法？** 主排名中没有；但 M4 B3 在 S3 的 5 个 fold 中有 1 个优于 B5（fold3：0.1825 vs 0.1852），说明 S3 的优势并非每个 fold 都稳定。
6. **差异是否受特征、预算、标签影响？** comboKR 使用官方 9 参数 BRAID 与相同 X/X0，训练预算为 CPU 91 s/split；DeepSynBa 使用 Morgan 指纹、CCLE、曲面目标，属于适配差异；DoseSyn 使用 GPU 与 160–240 epoch。报告已单列预算与适配说明。
7. **是否统一评价口径？** 是。所有模型都读取 M3 冻结 split 的 `predictions_test.parquet`，用同一个 `compute_metrics` 计算 matrix-macro RMSE；comboKR 因数据缺失另做同矩阵子集对比。
8. **是否存在测试泄漏或复现偏差？** 未发现。S3 严格细胞系隔离；comboKR 因 Hill 参数含测试细胞单药信息而不运行 S3；DeepSynBa 只读取各自 split 的训练/验证集，测试集在配置冻结后一次评估。DeepSynBa 的适配本身是复现偏差，已明确标注。
9. **哪些表/图可直接进论文？** 第 1 节主表、第 2 节配对差异、第 6 节同子集对比、第 4 节资源表，以及 `outputs/figures/m7a/macro_rmse_comparison.svg`、S3 曲面示例。
10. **能否进入 M7-B？** 可以，但必须保留两个限定：comboKR 的 S3 不适用；DeepSynBa 只能作为 adapted supplementary。M7-B 的剂量插值实验应继续以 S1/S2 为主要外部对比场景。

## 8. 测试与可追溯性

- 全量 pytest：**200 passed**（M0–M6-B 的 194 项 + M7-A 新增 6 项），日志 `outputs/audit/pytest_m7a.log`。
- 新增测试覆盖：配置完整性、冻结 split 目录、指标重算一致性、预测表 16 点/矩阵与 ID 映射、S1–S3 测试矩阵互斥、DeepSynBa 输出契约。
- 所有外部运行保留 `run_metadata.json`、`predictions.parquet`、日志和原始代码版本；未覆盖 M4–M6-B 任何文件。
- 外部代码：comboKR2.0（MIT）、DeepSynBa（CC BY-NC-SA，仅研究用途）。

## 9. 结论

**M7-A：CONDITIONAL PASS。**

- P0-A comboKR：忠实复现，S1/S2 结果可信，支持主排名。
- P0-B DeepSynBa：完成适配重实现，但输入特征与训练目标均与官方不同，只能作为补充 baseline，不能声称官方复现。
- P1 DD-PRiSM：没有公开代码，记为不可复现，未编造数字。
- 统一协议下 DoseSyn B5 在 S1/S2/S3 均为主排名第一；comboKR 是同矩阵子集上最强的外部方法，但仍明显弱于 B5。
- 阻塞项：comboKR 的 S3 协议冲突、DeepSynBa 无法忠实复现、DD-PRiSM 缺代码；这些是 M7-A 的边界，不影响已有主结论，但必须在论文中如实写明。

