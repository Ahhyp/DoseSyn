# DoseSyn M6-B 正式模型与多场景评估报告

生成时间：2026-10-24T18:13:44+00:00  
Git 提交：`750270014691f1b310759cc8fb68fd07747bf2b8`  
验证集选型：`b5_t3_film_reg`

## 1. 结论摘要

- 正式测试的最终模型为 **b5_t3_film_reg**；三个场景测试 matrix-macro RMSE 均值 0.1118。
- 共 22 个正式 run，覆盖 11 个冻结 split。
- 相对 B1 T3 参考：S1 −18.0%、S2 −29.2%、S3 −1.5%；S3 的 5 折方向不一致，1.5% 的均值优势落在折间波动内，不能声称 FiLM 改善跨细胞系泛化。
- 测试集只在 checkpoint 冻结后读取一次；验证集选型见 `m6b_model_selection.md`。
- 所有标签保持原始 `X/X0`，未裁剪；负预测未截断。

### 1.1 各冻结 split 的实际规模（train / val / test 矩阵数）

| scenario | split | train | val | test |
| --- | --- | ---: | ---: | ---: |
| s1_random_condition | seed_13 | 13026 | 2788 | 2785 |
| s1_random_condition | seed_2026 | 13029 | 2777 | 2793 |
| s1_random_condition | seed_42 | 13022 | 2791 | 2786 |
| s2_unseen_pair | seed_13 | 12974 | 2785 | 2840 |
| s2_unseen_pair | seed_2026 | 13013 | 2759 | 2827 |
| s2_unseen_pair | seed_42 | 12922 | 2798 | 2879 |
| s3_unseen_cell | fold_0 | 10927 | 3836 | 3836 |
| s3_unseen_cell | fold_1 | 10969 | 3794 | 3836 |
| s3_unseen_cell | fold_2 | 10990 | 3815 | 3794 |
| s3_unseen_cell | fold_3 | 11466 | 3318 | 3815 |
| s3_unseen_cell | fold_4 | 11445 | 3836 | 3318 |


## 2. 协议与隔离

- 数据：M3 冻结的 11 个 split；未重新划分、未改筛选规则。
- 训练：AdamW + Huber，CosineAnnealing，早停看 val matrix-macro RMSE。
- 单药：S1/S2 为 `known_monotherapy` 先验；S3 为 `strict_cell_isolation`，只允许训练细胞系的可训练曲线。
- 零剂量：显式 zero flag，禁止 `log10(0)`；boundary 为模型自洽误差，不是实测零剂量误差。
- 测试集：仅在验证选型冻结后运行 `--eval-test`，不回传梯度/调参。

## 3. 验证集选型

选定：**b5_t3_film_reg**。完整表见 `m6b_model_selection.md` 与 `outputs/metrics/m6b/validation_runs.csv`。

理由：B5 T3 + Regularized FiLM improves mean by 0.0067 and S3 gap is +0.0015; selected for formal M6-B.

## 4. 正式测试结果

### 4.1 分场景汇总（test matrix-macro RMSE）

| model | scenario | n | mean | std | min | max | params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| b1_t3_fourier | s1_random_condition | 3 | 0.0793 | 0.0007 | 0.0789 | 0.0800 | 498818 |
| b1_t3_fourier | s2_unseen_pair | 3 | 0.1085 | 0.0006 | 0.1080 | 0.1092 | 498818 |
| b1_t3_fourier | s3_unseen_cell | 5 | 0.1966 | 0.0175 | 0.1758 | 0.2187 | 498818 |
| b5_t3_film_reg | s1_random_condition | 3 | 0.0650 | 0.0011 | 0.0640 | 0.0662 | 535682 |
| b5_t3_film_reg | s2_unseen_pair | 3 | 0.0768 | 0.0037 | 0.0740 | 0.0810 | 535682 |
| b5_t3_film_reg | s3_unseen_cell | 5 | 0.1937 | 0.0125 | 0.1827 | 0.2135 | 535682 |

### 4.2 逐 split 结果（最终模型）

| scenario | split | val macro | test macro | point RMSE | Pearson | R² | best epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s1_random_condition | seed_13 | 0.0662 | 0.0640 | 0.0775 | 0.970 | 0.941 | 169 |
| s1_random_condition | seed_2026 | 0.0652 | 0.0662 | 0.0805 | 0.968 | 0.936 | 149 |
| s1_random_condition | seed_42 | 0.0665 | 0.0648 | 0.0803 | 0.968 | 0.936 | 155 |
| s2_unseen_pair | seed_13 | 0.0805 | 0.0740 | 0.0904 | 0.960 | 0.921 | 169 |
| s2_unseen_pair | seed_2026 | 0.0776 | 0.0753 | 0.0940 | 0.953 | 0.909 | 159 |
| s2_unseen_pair | seed_42 | 0.0742 | 0.0810 | 0.1005 | 0.947 | 0.894 | 183 |
| s3_unseen_cell | fold_0 | 0.1673 | 0.1981 | 0.2323 | 0.709 | 0.471 | 6 |
| s3_unseen_cell | fold_1 | 0.2192 | 0.1827 | 0.2052 | 0.772 | 0.594 | 29 |
| s3_unseen_cell | fold_2 | 0.1814 | 0.2135 | 0.2412 | 0.684 | 0.407 | 16 |
| s3_unseen_cell | fold_3 | 0.2154 | 0.1852 | 0.2070 | 0.760 | 0.567 | 5 |
| s3_unseen_cell | fold_4 | 0.1906 | 0.1893 | 0.2121 | 0.771 | 0.571 | 22 |

### 4.3 与 M4 B3 的同 split 对照

| scenario | split | M4 B3 | final | Δ(final−B3) |
| --- | --- | ---: | ---: | ---: |
| s1_random_condition | seed_13 | 0.0805 | 0.0640 | -0.0165 |
| s1_random_condition | seed_2026 | 0.0799 | 0.0662 | -0.0138 |
| s1_random_condition | seed_42 | 0.0779 | 0.0648 | -0.0132 |
| s2_unseen_pair | seed_13 | 0.0902 | 0.0740 | -0.0161 |
| s2_unseen_pair | seed_2026 | 0.0870 | 0.0753 | -0.0117 |
| s2_unseen_pair | seed_42 | 0.0919 | 0.0810 | -0.0109 |
| s3_unseen_cell | fold_0 | 0.2070 | 0.1981 | -0.0089 |
| s3_unseen_cell | fold_1 | 0.1849 | 0.1827 | -0.0023 |
| s3_unseen_cell | fold_2 | 0.2269 | 0.2135 | -0.0134 |
| s3_unseen_cell | fold_3 | 0.1825 | 0.1852 | 0.0027 |
| s3_unseen_cell | fold_4 | 0.1956 | 0.1893 | -0.0063 |

### 4.4 配对差异（相对 T3 基础模型）

| model | scenario | pairs | mean Δ | std | wins | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| b5_t3_film_reg | s1_random_condition | 3 | -0.0143 | 0.0005 | 3 | 0 |
| b5_t3_film_reg | s2_unseen_pair | 3 | -0.0318 | 0.0040 | 3 | 0 |
| b5_t3_film_reg | s3_unseen_cell | 5 | -0.0029 | 0.0069 | 3 | 2 |

## 5. 药理一致性诊断（验证集）

| model | scenario | swap max | E_boundary(a0) | E_boundary(0b) | mono legal RMSE | smooth |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| b1_t3_fourier | s1_random_condition | 3.58e-07 | — | — | — | — |
| b1_t3_fourier | s2_unseen_pair | 4.77e-07 | — | — | — | — |
| b1_t3_fourier | s3_unseen_cell | 2.98e-07 | — | — | — | — |
| b5_t3_film_reg | s1_random_condition | 4.17e-07 | — | — | — | — |
| b5_t3_film_reg | s2_unseen_pair | 3.58e-07 | — | — | — | — |
| b5_t3_film_reg | s3_unseen_cell | 2.38e-07 | — | — | — | — |

## 6. 效率

| model | params | mean wall (s) | GPU peak (MB) |
| --- | ---: | ---: | ---: |
| b1_t3_fourier | 498818 | 1284 | 124 |
| b1_t3_fourier | 498818 | 188 | 124 |
| b1_t3_fourier | 498818 | 322 | 125 |
| b5_t3_film_reg | 535682 | 929 | 154 |
| b5_t3_film_reg | 535682 | 293 | 154 |
| b5_t3_film_reg | 535682 | 135 | 154 |

## 7. 失败与局限

- 训练预算不一致：B1 为 200 epoch，B5 S1/S2 为 160 epoch，B5 S3 为 80 epoch；S3 的最佳 epoch 多为 5–29，说明未见细胞系场景训练不足。
- FiLM 的 S3 泛化需诚实报告；不因 S1 最优而选跨细胞系较差模型。
- 边界误差是模型自洽诊断，不是实测零剂量实验。
- 未做大规模超参数搜索；测试集未参与任何选择。

## 8. 产物

- `outputs/metrics/m6b/formal_runs.csv`, `formal_summary.csv`, `paired_deltas.csv`
- `outputs/figures/m6b/formal_test_macro.svg`, `formal_val_curves.svg`, `surface_examples.svg`
- `outputs/metrics/m6b/surface_examples.csv`
- 每个 run 的 `outputs/runs/m6b/<run_id>/{best.pt,train_log.csv,val_metrics.json,test_metrics.json,predictions_test.parquet}`

## 9. Git 与复现

- 生成时 Git：`750270014691f1b310759cc8fb68fd07747bf2b8`
- 冻结数据：`data/processed/oneil_primary_v1_*.parquet` 与 `data/splits/` 均由 M3 manifest 哈希锁定，本阶段未修改。
- 复现：`scripts/11_run_m6b.py --stage formal --model <model> --epochs 200 --eval-test`；汇总：`scripts/13_aggregate_m6b.py --write`。