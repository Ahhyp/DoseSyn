# DoseSyn M6-A 性能复现与结构诊断报告

生成时间：2026-10-24T15:13:03+00:00  
基线 commit：`dbb63b41c09d8f76794a83ab5ca0a8a84c33d400`

**结论：GO**

## 1. 结论摘要

- T0 B3 checkpoint 推理复现成功：matrix-macro RMSE `0.0801`（M4 存储值 `0.0801`，参数 284,545）。
- 交叉配方：B3 + M4 = `0.0801`（T0 复现），B3 + M5 = `0.0820`；C0 + M5 = `0.1820`，C0 + M4 = `0.1820` → 训练配方不是性能差距主因。
- 结构消融：T2 对称+标量 `0.1015`，T3 对称+Fourier `0.0820`，T4 对称+Fourier+FiLM `0.0670`；Fourier 剂量表示是恢复性能的关键，FiLM 在 S1 进一步增益但在 S3 过拟合。
- S1 最佳 M6-A 变体：`m6a__b3sym_film__s1_random_condition__seed_42`，matrix-macro `0.0670`（相对 C0 的 0.1820 收窄 63.2%）。

## 2. B3 精确对照（T0）

checkpoint：`outputs/runs/m4/b3_gine__s1_random_condition__seed_42/best.pt`（epoch 77，sha256 `f696e0b5aa59d978…`）

| metric | value |
| --- | --- |
| n_matrices | 2791 |
| n_points | 44656 |
| matrix_macro_rmse | 0.08005502275233599 |
| matrix_macro_mae | 0.0646218528765092 |
| condition_macro_rmse | 0.07982508921945221 |
| point_rmse | 0.09502250508392601 |
| point_mae | 0.0646218528765092 |
| pooled_pearson | 0.9545984418003403 |
| r2 | 0.9109291316299961 |
| matrix_pearson_mean | 0.9267269092995943 |
| n_matrix_pearson_undefined | 0 |
| y_true_min | 0.0 |
| y_true_max | 1.638069987297058 |
| y_true_mean | 0.5691502622289107 |
| y_pred_min | -0.13458238542079926 |
| y_pred_max | 1.3210351467132568 |
| y_pred_mean | 0.57447854889155 |
| n_y_true_gt1 | 3072 |
| n_y_true_eq0 | 14 |
| n_y_pred_negative | 401 |


## 3. 隔离探针：配方 vs 架构

| probe | 说明 | 参数 | matrix-macro | point | 最佳 epoch |
| --- | --- | ---: | ---: | ---: | ---: |

| b3_m4 | T0-replica: B3 architecture + M4 recipe | 284,545 | 0.0801 | 0.0950 | 77 |

| b3_m5 | probe: B3 architecture + M5 recipe | 284,545 | 0.0820 | 0.0965 | 59 |

| c0_m5 | probe: M5 C0 + M5 recipe | 559,299 | 0.1820 | 0.2067 | 60 |

| c0_m4 | probe: M5 C0 + M4 recipe | 559,299 | 0.1820 | 0.2068 | 77 |


## 4. 结构差异

| 模型 | 参数 | 剂量表示 | 药物对融合 |
| --- | ---: | --- | --- |

| T0 b3_exact | 284,545 | standardized scalar + zero flag | ordered concat(cell, ha, hb, da, db) |

| T1 c0_control | 559,299 | Linear(2,64) dose embedding | ordered concat(cell, ha, hb, ea64, eb64) |

| T2 b3sym_scalar | 429,954 | raw scalar + zero flag | sum/product/|diff| + raw dose |

| T3 b3sym_fourier | 492,290 | Fourier embedding 64d | sum/product/|diff| + raw dose |

| T4 b3sym_film | 535,682 | Fourier + FiLM(gamma,beta) | FiLM then sum/product/|diff| |


## 5. 受控实验（S1 seed42 / S3 fold0 验证集）

### S1 seed42

| model | matrix_macro_rmse | point_rmse | pooled_pearson | r2 | best_epoch | epochs_run | n_parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b3sym_film | 0.0670 | 0.0824 | 0.9659999368680249 | 0.9330809586590242 | 153 | 160 | 535682 |
| b3sym_fourier | 0.0820 | 0.0968 | 0.9527657178301394 | 0.907524348547235 | 149 | 160 | 492290 |
| b3sym_scalar | 0.1015 | 0.1180 | 0.9288754290526939 | 0.8627550223902014 | 147 | 160 | 429954 |


### S3 fold0

| model | matrix_macro_rmse | point_rmse | pooled_pearson | r2 | best_epoch | epochs_run | n_parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b3sym_film | 0.1734 | 0.1923 | 0.8080993083615786 | 0.643374282849797 | 20 | 50 | 535682 |
| b3sym_fourier | 0.1695 | 0.1886 | 0.8169388372603258 | 0.6569599441001597 | 17 | 47 | 492290 |
| b3sym_scalar | 0.1752 | 0.1936 | 0.8035337054192098 | 0.6386675475252617 | 59 | 89 | 429954 |


## 6. 性能差距归因

1. **训练配方不是主因**：B3 在 M5 配方下仍保持 `0.0820`；C0 在 M4 配方下仍停在 `0.1820`。
2. **标量剂量 + 对称融合（T2）只恢复一半**：`0.1015`，仍比 B3 差 26.7%。说明单纯去掉 C0 的 ordered 头/64 维剂量嵌入不够。
3. **Fourier 剂量表示（T3）是主要缺失件**：`0.0820`，与 B3 `0.0801` 仅差 2.4%；S3 为 `0.1695` vs B3 `0.1681`。
4. **FiLM（T4）在 S1 显著增益、在 S3 过拟合**：S1 `0.0670`（比 B3 好 16.3%），但 S3 `0.1734`（比 B3 差 3.1%），最佳 epoch 仅 20，泛化不稳定。
5. **早停口径**：M4 用 point RMSE，M5 用 matrix-macro；M6-A 全部统一为 matrix-macro，并显式写入 `selection_metric`。
6. **标签未截断**：所有 run 的验证标签最大值 > 1.2（S1 1.638，S3 1.330），未裁剪 `X/X0`。

## 7. 模型选择与 GO/NO-GO

- S1 最佳：`m6a__b3sym_film__s1_random_condition__seed_42`（0.0670）；S3 最佳：`m6a__b3sym_fourier__s3_unseen_cell__fold_0`（0.1695）。
- 跨场景稳健选择：**T3 `b3sym_fourier`**（S1 0.0820 / S3 0.1695，均接近或达到 B3），作为 M6-B 默认配置；T4 FiLM 仅建议在 S1 类同分布场景使用或作为集成成员，不单独用于跨细胞系泛化结论。
- 不建议回退到 C0 ordered 头：其 S1 0.1820 的差距不是训练预算能解释的。
- 建议：**GO**。

## 8. 检查项

| name | status | detail |
| --- | --- | --- |
| b3_checkpoint_parity | PASS | recomputed 0.0801 vs stored 0.0801 |
| b3_recipe_independent | PASS | B3 + M5 recipe = 0.0820 |
| c0_recipe_independent | WARN | C0 + M4 recipe = 0.1820 |
| s1_runs_complete | PASS | 3 S1 structural variants (T2-T4); T0/T1 run as probes |
| s3_runs_complete | PASS | 3 S3 runs |
| matrix_macro_early_stop | PASS | all runs select on matrix-macro RMSE |
| no_label_truncation | PASS | validation max label > 1.2 in all runs (S1 1.638, S3 1.330), no truncation |
| predictions_16_rows | PASS | m6a__b3sym_film__s1_random_condition__seed_42: 44656 rows / 2791 matrices; m6a__b3sym_film__s3_unseen_cell__fold_0: 61376 rows / 3836 matrices; m6a__b3sym_fourier__s1_random_condition__seed_42: 44656 rows / 2791 matrices; m6a__b3sym_fourier__s3_unseen_cell__fold_0: 61376 rows / 3836 matrices; m6a__b3sym_scalar__s1_random_condition__seed_42: 44656 rows / 2791 matrices; m6a__b3sym_scalar__s3_unseen_cell__fold_0: 61376 rows / 3836 matrices |
| m3_frozen_inputs | PASS | 15/15 M3 inputs unchanged |

### 8.5 失败与中止实验

- 20594/20595/20596（T2–T4 首轮，160 epoch）在加入逐 epoch 日志前长时间无输出；其中 20595 被 SIGINT 中止，栈显示位于首轮 batch 的 `global_mean_pool`，不是死锁。为消除每 batch 128 次重复 GINE 编码，改为 36 个唯一药物图的 `vocab_graph_batch` 后重新提交，旧任务全部 qdel。
- 旧任务未产生 best.pt/val_metrics，不进入比较表；重跑后 6 个正式 run 全部完成。
- 未运行独立 `b3_m4` 训练探针：T0 直接加载 M4 B3 checkpoint 已验证同口径复现（0.0801），避免重复训练。

## 9. 产物与复现
- 运行：`.venv/bin/python scripts/08_run_m6a.py --all --recipe m6a --epochs 160 --device cuda`
- 报告：`.venv/bin/python scripts/09_write_m6a_report.py --write`
- 测试：`.venv/bin/python -m pytest -q tests`
- 曲线：6 个 SVG（outputs/figures/m6a/curves/）

## 10. Git 记录

- M6-A 代码提交：`dbb63b41c09d8f76794a83ab5ca0a8a84c33d400`（父提交 `9065a21`，M6-0）。
- 本报告由 `scripts/09_write_m6a_report.py --write` 从 6 个 run 的 best.pt/val_metrics 与 4 个探针 JSON 生成，不重新训练。
- 报告与 manifest 随代码提交之后的文档提交发布；测试记录 `outputs/audit/pytest_m6a.log`（172 passed）。