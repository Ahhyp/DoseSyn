# DoseSyn M6-0 前置审计报告
生成时间：2026-10-24T11:56:59+00:00  
基线 commit：`820dfcb4fb4dd6ca4baaf212f9f16f2ed64fa662`  
**关卡结论：PASS**

## 1. 结论摘要
- **S1 seed42 数量异常是报告文字错误，不是训练用错 split。** M3 `seed42` 元数据为 13,022 / 2,791 / 2,786；M5 全部 7 个 S1 seed42 run 的 val 预测矩阵集合与 seed42 val 完全相等（2,791 个），与 seed13（2,788 个）不相等。
- M5 训练/预测/指标没有混用 seed13；错误仅出现在 M5 对外摘要把 seed42 写成了 seed13 的 13,026 / 2,788 / 2,785。原始 M5 产物未改动。
- 在**同一组** S1 seed42 val 上，M4 B3 `matrix_macro_RMSE=0.0801`，M5 C0 `=0.1820`，C3 `=0.1797`，C6 `=0.1870`。这不是 0.0795 test 与 0.1820 val 的“直接比较”，而是同口径验证集比较。
- C0 不是 B3 的复现：参数量 559,299 vs 284,545，剂量表示、融合、激活、训练预算与早停指标均不同。M6-A 若需要对 B3 的公平参照，应实现/使用 `b3_exact`，不能拿 C0 代替。
- M5 S1 的 C0/C3/C5/C6 最佳 epoch 均在 60/60 附近，验证曲线仍在下降；预算偏短。但仅延长预算不足以解释与 B3 的 0.10 差距——B3 在第 60 epoch 的验证 point RMSE 已约 0.096，而 C0 的 matrix-macro 仍是 0.182。

## 2. 冻结划分与文件哈希
| split | path | train | val | test | assignment sha256 |
| --- | --- | ---: | ---: | ---: | --- |
| S1 seed42 | `/home/kouzheng_2/AH_YeP/DoseSyn/data/splits/s1_random_condition/seed_42` | 13022 | 2791 | 2786 | `b2219079dad06796…` |
| S1 seed13 | `/home/kouzheng_2/AH_YeP/DoseSyn/data/splits/s1_random_condition/seed_13` | 13026 | 2788 | 2785 | `e54c56f0999d348a…` |
| S3 fold0 | `/home/kouzheng_2/AH_YeP/DoseSyn/data/splits/s3_unseen_cell/fold_0` | 10927 | 3836 | 3836 | `e83dc95b1c40db9f…` |

### 2.1 seed42 vs seed13
- val 交集：423 / seed42 2791 / seed13 2788；集合完全相同：False。
- 两者是不同随机划分，val 矩阵数不同（2,791 vs 2,788），不能互相替代。
- M3 冻结输入/输出哈希校验：18/18 个文件与 M3 manifest 一致。

### 2.2 M5 实际读取
| run | split_id | val matrices | == seed42 | == seed13 | missing | extra |
| --- | --- | ---: | --- | --- | ---: | ---: |
| `m5__s1_random_condition__seed_42__c0_control` | seed_42 | 2791 | True | False | 0 | 0 |
| `m5__s1_random_condition__seed_42__c1_symmetric` | seed_42 | 2791 | True | False | 0 | 0 |
| `m5__s1_random_condition__seed_42__c2_fourier` | seed_42 | 2791 | True | False | 0 | 0 |
| `m5__s1_random_condition__seed_42__c3_film` | seed_42 | 2791 | True | False | 0 | 0 |
| `m5__s1_random_condition__seed_42__c4_mono` | seed_42 | 2791 | True | False | 0 | 0 |
| `m5__s1_random_condition__seed_42__c5_boundary` | seed_42 | 2791 | True | False | 0 | 0 |
| `m5__s1_random_condition__seed_42__c6_full` | seed_42 | 2791 | True | False | 0 | 0 |

## 3. B3 与 M5 同口径验证指标
| 场景 | 模型 | matrix-macro RMSE | point RMSE | Pearson | R² | matrices |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S1 seed42 | m4_b3 | 0.0801 | 0.0950 | 0.9546 | 0.9109 | 2791 |
| S1 seed42 | m5_c0 | 0.1820 | 0.2067 | 0.7608 | 0.5787 | 2791 |
| S1 seed42 | m5_c3 | 0.1797 | 0.2049 | 0.7655 | 0.5859 | 2791 |
| S1 seed42 | m5_c5 | 0.1803 | 0.2051 | 0.7649 | 0.5850 | 2791 |
| S1 seed42 | m5_c6 | 0.1870 | 0.2090 | 0.7565 | 0.5690 | 2791 |
| S3 fold0 | m4_b3 | 0.1681 | 0.1855 | 0.8200 | 0.6682 | 3836 |
| S3 fold0 | m5_c0 | 0.2289 | 0.2480 | 0.6381 | 0.4068 | 3836 |
| S3 fold0 | m5_c6 | 0.2331 | 0.2504 | 0.6365 | 0.3953 | 3836 |

- M4 B3 的 `val_metrics.json` 与用保存的 `predictions_val.parquet` 重算结果一致（最大差 < 1e-9），说明指标复算无误。
- M4/M5 预测对齐：共同键 44656，标签不一致 0，剂量最大差 0.0/0.0。

## 4. 结构、训练预算与早停差异
| 维度 | M4 B3 GINE | M5 C0 control | 对齐 | 说明 |
| --- | --- | --- | --- | --- |
| 参数 | 284545 | 559299 | False | C0 更大，不是 B3 复现 |
| 药物编码 | 共享 GINE，原子/键类别嵌入，3 层 | 共享 GINE，同类特征，3 层 | True | 同一 M2 图特征 |
| 细胞编码 | 原始 50 维 Hallmark | 原始 50 维（ordered 分支） | True | 标准化由 M3 scaler 提供 |
| 剂量表示 | 标准化标量 + zero flag | Linear(2,64) 剂量嵌入 | False | C0 多一层剂量投影 |
| 药物对融合 | 有序 concat(cell, ha, hb, da, db) | 有序 concat(cell, ha, hb, ea64, eb64) | False | C0 非对称对照 |
| 激活函数 | GELU + LayerNorm(GINE) | SiLU + GINE LayerNorm | False | 结构差异 |
| 优化器 | AdamW lr=1e-3 wd=1e-4 | AdamW lr=1e-3 wd=1e-4 | True | 相同 |
| batch size | 64 | 64 | True |  |
| 训练预算 | 80 epochs, patience 15 | 60 epochs, patience 12 | False | M5 预算短 25% |
| 早停指标 | validation POINT RMSE | validation MATRIX-MACRO RMSE | False | 评价口径不同；B3 利好 point 指标 |
| 标签 | 原始 X/X0，未裁剪 | 原始 X/X0，未裁剪 | True |  |
| 验证集 | M3 S1 seed42 val (2,791) | M3 S1 seed42 val (2,791) | True | 本轮已验证集合相等 |

### 4.1 训练曲线
| run | epochs | best epoch | best val (log) | last val | last-10 slope | final lr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m4_b3_s1 | 80 | 77 | 0.0950 | 0.0951 | -0.0000 | 0.0 |
| m5_c0_s1 | 60 | 60 | 0.1820 | 0.1820 | -0.0013 | 0.00025 |
| m5_c3_s1 | 60 | 59 | 0.1797 | 0.1797 | -0.0006 | 0.00025 |
| m5_c5_s1 | 60 | 59 | 0.1803 | 0.1803 | -0.0004 | 6.25e-05 |
| m5_c6_s1 | 60 | 59 | 0.1870 | 0.1871 | -0.0042 | 0.00025 |
| m4_b3_s3 | 37 | 22 | 0.1855 | 0.1902 | 0.0014 | 0.0005587686987289 |

- M4 `train_log.csv` 的 `val_matrix_macro_rmse` 列实际记录的是 validation **point RMSE**（checkpoint `epoch=77, val_macro_rmse=0.09502` 与 train_log 第 77 行一致，而 `val_metrics.json` 的 matrix-macro 是 0.0801）。这是 M4 的日志命名/早停口径问题，不是本轮新产生的数据错误；M6-A 必须显式记录。

## 5. 判定与 M6-A 前置条件
| 检查 | 状态 | 说明 |
| --- | --- | --- |
| seed42_metadata_counts | PASS | {'train': 13022, 'val': 2791, 'test': 2786} |
| seed42_val_matrix_set_size | PASS | 2791 |
| m5_seed42_uses_seed42 | PASS | 7/7 runs |
| m5_seed42_not_seed13 | PASS | no run val set equals seed13 val set |
| m4_seed42_uses_seed42 | PASS | 2791 |
| m4_m5_same_labels_doses | PASS | {'n_m4_rows': 44656, 'n_m5_rows': 44656, 'n_common_keys': 44656, 'n_label_mismatch': 0, 'max_abs_dose_a_diff': 0.0, 'max_abs_dose_b_diff': 0.0, 'same_matrix_set': True} |
| metric_recompute_matches_stored | PASS | 0.0 |
| m5_s3_uses_fold0 | PASS | 5/5 |
| m4_s3_uses_fold0 | PASS | 3836 |
| m3_frozen_hashes_unchanged | PASS | 18/18 files |
| c0_is_not_b3_reproduction | WARN | params 559,299 vs 284,545; different dose/fusion/activation/budget/early-stop metric |
| m4_train_log_metric_label | WARN | M4 train_log val column is point RMSE despite its name; stored val_metrics/checkpoint are authoritative |

- **放行条件满足**：split 对齐、B3 同口径 val 参照建立、冻结哈希未变、M4/M5 产物未覆盖。
- **必须带上的修正**：M6-A 使用 `b3_exact`（或 M4 B3 原始 checkpoint/重训）作为 B3 参照；M5 C0 只能标为“M5 ordered control”；训练预算至少 160 epoch，patience 25–40；早停统一用 matrix-macro RMSE；M4 train_log 列名问题在报告中注明。
- 本轮没有读取任何 test 标签，没有运行正式 11 split，没有修改 M0–M3 或覆盖 M4/M5 产物。

## 6. 产物、测试与 Git
- `outputs/audit/m6_gate0_split_audit.json`（完整，本地；含逐 split matrix id）
- `outputs/audit/m6_gate0_baseline_alignment.json`（完整，本地）
- `outputs/metrics/m6/gate0_validation_metrics.csv`（本地）
- Git 跟踪副本：`docs/audit/m6_gate0_split_audit.json`、`docs/audit/m6_gate0_baseline_alignment.json`、`docs/audit/gate0_validation_metrics.csv`
- `m6_gate0_report.md`
- 自动测试：`tests/test_m6_gate0.py`（split 集合、矩阵数、B3/M5 标签与剂量一致性）。
- 完整 pytest：**165 passed in 41.52s**（日志 `outputs/audit/pytest_m6.log`）。
- 本报告与审计脚本不修改 M4/M5 产物；M3 冻结哈希 18/18 一致。
