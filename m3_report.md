# DoseSyn M3 数据集划分报告

生成时间：2026-10-24T06:42:14+00:00　|　数据集版本：`oneil-primary-v1`　|　Git：`ccc75669fc6f`

## 1. 结论

| 验收项 | 结果 |
| --- | --- |
| Primary-v1 冻结 | PASS（18,599 矩阵 / 1,782 单药曲线） |
| S1/S2/S3 生成 | PASS（11 个 split 目录） |
| 泄漏检查 | PASS（11/11） |
| 单药隔离披露 | PASS（S3 严格封存 val/test 细胞曲线） |
| 训练折专属变换 | PASS（fit 实体 == train 实体） |
| 自动测试 | PASS（114 passed / 0 failed / 0 errors） |
| 未训练神经网络 | 确认（本阶段无任何模型代码） |

## 2. 主数据集冻结

纳入规则：M2 `feature_complete` ＋ 细胞表达/通路可用 ＋ 4×4 完整矩阵 ＋ 药物图可建模；标签为 M1 原始 `X/X0`（不裁剪）。Carboplatin/Oxaliplatin 因 RDKit 无法表达铂配位键而在一期排除，原始数据、映射与 SMILES 全部保留，并逐矩阵记录排除原因。

| 实体 | 数量 |
| --- | ---: |
| 矩阵 | 18,599 |
| 条件（细胞×无序对） | 18,326 |
| 细胞系 | 34 |
| 药物 | 36 |
| 无序药物对 | 539 |
| 响应点 | 297,584 |
| 单药曲线 | 1,782 |

排除原因（行数，可能重叠）：

| exclusion_reason | 行数 |
| --- | ---: |
| `cell_features_incomplete` | 2,953 |
| `drug_graph_excluded` | 1,812 |

## 3. 三种实验场景

S1 以 `condition_id` 为原子、S2 以 `pair_id` 为原子、S3 以 `cell_id` 为原子；任何矩阵、批次重复与组键都不会跨 split。S1/S2 各 3 个种子，S3 为 5 折。

### 3.1 全部 split 规模

| 场景 | 种子/折 | train 条件 | val 条件 | test 条件 | train 矩阵 | val 矩阵 | test 矩阵 | mono trainable | mono held |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1_random_condition | seed 13 | 12,828 | 2,748 | 2,750 | 13,026 | 2,788 | 2,785 | 1,782 | 0 |
| s1_random_condition | seed 42 | 12,828 | 2,748 | 2,750 | 13,022 | 2,791 | 2,786 | 1,782 | 0 |
| s1_random_condition | seed 2026 | 12,828 | 2,748 | 2,750 | 13,029 | 2,777 | 2,793 | 1,782 | 0 |
| s2_unseen_pair | seed 13 | 12,818 | 2,720 | 2,788 | 12,974 | 2,785 | 2,840 | 1,782 | 0 |
| s2_unseen_pair | seed 42 | 12,818 | 2,720 | 2,788 | 12,922 | 2,798 | 2,879 | 1,782 | 0 |
| s2_unseen_pair | seed 2026 | 12,818 | 2,720 | 2,788 | 13,013 | 2,759 | 2,827 | 1,782 | 0 |
| s3_unseen_cell | fold 0 | 10,780 | 3,773 | 3,773 | 10,927 | 3,836 | 3,836 | 990 | 792 |
| s3_unseen_cell | fold 1 | 10,780 | 3,773 | 3,773 | 10,969 | 3,794 | 3,836 | 1,062 | 720 |
| s3_unseen_cell | fold 2 | 10,780 | 3,773 | 3,773 | 10,990 | 3,815 | 3,794 | 1,152 | 630 |
| s3_unseen_cell | fold 3 | 11,319 | 3,234 | 3,773 | 11,466 | 3,318 | 3,815 | 1,134 | 648 |
| s3_unseen_cell | fold 4 | 11,319 | 3,773 | 3,234 | 11,445 | 3,836 | 3,318 | 1,008 | 774 |

### 3.2 代表划分的细胞/药物对覆盖

| 场景 | split | 细胞 | 药物对 |
| --- | --- | ---: | ---: |
| S1 seed13 | train | 34 | 539 |
| S1 seed13 | val | 34 | 537 |
| S1 seed13 | test | 34 | 533 |
| S2 seed13 | train | 34 | 377 |
| S2 seed13 | val | 34 | 80 |
| S2 seed13 | test | 34 | 82 |
| S3 fold0 | train | 20 | 539 |
| S3 fold0 | val | 7 | 539 |
| S3 fold0 | test | 7 | 539 |

### 3.3 代表划分的癌症类型分布（test）

**S1 seed13 test**（6 类）

| 癌症类型 | test 矩阵 | 全集矩阵 | 保留比例 |
| --- | ---: | ---: | ---: |
| Lung Cancer | 663 | 4,396 | 15.1% |
| Ovarian Cancer | 589 | 3,836 | 15.4% |
| Colon/Colorectal Cancer | 572 | 3,857 | 14.8% |
| Skin Cancer | 466 | 3,276 | 14.2% |
| Breast Cancer | 349 | 2,156 | 16.2% |
| Prostate Cancer | 146 | 1,078 | 13.5% |

**S2 seed13 test**（6 类）

| 癌症类型 | test 矩阵 | 全集矩阵 | 保留比例 |
| --- | ---: | ---: | ---: |
| Lung Cancer | 672 | 4,396 | 15.3% |
| Colon/Colorectal Cancer | 590 | 3,857 | 15.3% |
| Ovarian Cancer | 586 | 3,836 | 15.3% |
| Skin Cancer | 500 | 3,276 | 15.3% |
| Breast Cancer | 328 | 2,156 | 15.2% |
| Prostate Cancer | 164 | 1,078 | 15.2% |

**S3 fold0 test**（6 类）

| 癌症类型 | test 矩阵 | 全集矩阵 | 保留比例 |
| --- | ---: | ---: | ---: |
| Colon/Colorectal Cancer | 1,099 | 3,857 | 28.5% |
| Ovarian Cancer | 1,099 | 3,836 | 28.6% |
| Lung Cancer | 560 | 4,396 | 12.7% |
| Breast Cancer | 539 | 2,156 | 25.0% |
| Skin Cancer | 539 | 3,276 | 16.5% |
| Prostate Cancer | 0 | 1,078 | 0.0% |

## 4. 单药监督范围

| 场景 | 策略 | 可训练曲线 | 封存曲线 | 说明 |
| --- | --- | ---: | ---: | --- |
| S1 | `known_monotherapy` | 1,782 | 0 | 已知先验，不是独立测试集 |
| S2 | `known_monotherapy` | 1,782 | 0 | 同上，报告中明确披露 |
| S3 | `strict_cell_isolation` | 990 | 792 | 仅训练细胞；val/test 细胞标签封存 |

S3 每折的 `mono_assignments.parquet` 以 `label_access`（`train` / `held_out` / `known_prior`）与 `trainable` 布尔列显式标注；任何训练脚本读取 `held_out` 曲线即违反协议。

## 5. 训练折专属特征变换

| 变换 | 拟合数据 | 参数文件 | 关键校验 |
| --- | --- | --- | --- |
| Hallmark 50 维标准化 | 仅 train 细胞（S1/S2）/ 仅该折 train 细胞（S3） | `scalers.joblib` | fit cells == train cells |
| 表达谱标准化（可选） | 仅 train 细胞 | `scalers.joblib` | 同上 |
| 组合药物剂量 log10(M) | 仅 train 矩阵的非零剂量 | `scalers.joblib` | fit_matrix_ids == train matrices |
| 单药剂量 log10(M) | 仅可训练单药曲线 | `scalers.joblib` | S3 仅 train 细胞 |

- **S1 seed13**：cells=34，conditions=12,828，mono_curves=1,782，scaler_sha256=`d3da31e95c965294…`
- **S2 seed13**：cells=34，conditions=12,818，mono_curves=1,782，scaler_sha256=`a94dab4de268c398…`
- **S3 fold0**：cells=20，conditions=10,780，mono_curves=990，scaler_sha256=`acd05d89bf36f058…`

## 6. 泄漏检查

| split | 状态 | 问题数 |
| --- | --- | ---: |
| s1_random_condition/seed_13 | PASS | 0 |
| s1_random_condition/seed_2026 | PASS | 0 |
| s1_random_condition/seed_42 | PASS | 0 |
| s2_unseen_pair/seed_13 | PASS | 0 |
| s2_unseen_pair/seed_2026 | PASS | 0 |
| s2_unseen_pair/seed_42 | PASS | 0 |
| s3_unseen_cell/fold_0 | PASS | 0 |
| s3_unseen_cell/fold_1 | PASS | 0 |
| s3_unseen_cell/fold_2 | PASS | 0 |
| s3_unseen_cell/fold_3 | PASS | 0 |
| s3_unseen_cell/fold_4 | PASS | 0 |

执行的检查：`batch_repeats_stay_together`；`cell_id_never_spans_splits`；`cell_scalers_fit_on_train_cells_only`；`cells_disjoint_across_splits`；`condition_id_never_spans_splits`；`conditions_partition_exactly`；`counts_match_metadata`；`every_matrix_assigned`；`fold_cells_match_files`；`labels_finite_for_all_points`；`matrix_never_spans_splits`；`mono_assignments_cover_curves`；`mono_known_prior_fully_trainable`；`mono_scaler_fit_matches_trainable`；`mono_trainable_only_train_cells`；`mono_val_test_labels_sealed`；`no_cell_in_multiple_splits`；`pair_id_never_spans_splits`；`scaler_fit_cells_match_train`；`scaler_fit_conditions_match_train`；`train_groups_disjoint`。

## 7. 文件、哈希与数据加载

| 文件 | SHA256（前 16 位） | 大小 |
| --- | --- | ---: |
| `/home/kouzheng_2/AH_YeP/DoseSyn/data/processed/oneil_primary_v1_matrix_index.parquet` | `ae9254bb490762d4` | 605,003 B |
| `/home/kouzheng_2/AH_YeP/DoseSyn/data/processed/oneil_primary_v1_mono_index.parquet` | `f16efe9995d7bab2` | 60,138 B |
| `/home/kouzheng_2/AH_YeP/DoseSyn/data/processed/oneil_primary_v1_exclusions.parquet` | `1e4b2268c9036dc5` | 152,784 B |
| `/home/kouzheng_2/AH_YeP/DoseSyn/outputs/audit/m3_input_manifest.json` | `e9176574d61c4831` | 3,182 B |
| `/home/kouzheng_2/AH_YeP/DoseSyn/outputs/audit/m3_leakage_report.json` | `2a194b239eb79247` | 36,155 B |

输入哈希（M1/M2 未修改）：

| 输入 | SHA256 前缀 |
| --- | --- |
| `data/processed/matrix_feature_index.parquet` | `84880290f093aa91` |
| `data/processed/condition_feature_index.parquet` | `a0e1667925519b0a` |
| `data/processed/mono_feature_index.parquet` | `3ea2df807b6bb6bb` |
| `data/processed/oneil_combo_matrix_metadata.parquet` | `beaa0126365c09e1` |
| `data/processed/oneil_combo_points.parquet` | `906c922d673c0d69` |
| `data/processed/oneil_mono_curve_metadata.parquet` | `284f4518f5165d56` |
| `data/processed/oneil_monotherapy.parquet` | `acabe2c2c3f61f07` |
| `data/processed/drug_mapping.csv` | `60c1ed25499471a1` |
| `data/processed/cell_mapping.csv` | `7a392eb67cf1cbf2` |
| `data/processed/drug_graph_features.pt` | `f579729c945a3cfb` |
| `data/processed/drug_graph_metadata.json` | `85cadcbfdeadce36` |
| `data/processed/cell_expression.parquet` | `8d6cc51d7940cbfc` |
| `data/processed/cell_features.parquet` | `031f4fc91181f63e` |
| `data/processed/pathway_features.parquet` | `7b41b1ca720eb1f3` |
| `data/processed/pathway_metadata.csv` | `1c4d6a72ed2d129e` |

数据加载接口 `DoseSynDataset` 已在 S1/S2/S3 真实分组上做 smoke test：每个 batch 返回 4×4 标签、标准化剂量、零剂量标记、细胞特征与药物图；`collate_matrices` 能正确拼接异构图并给出 `drug_batch`/`drug_role`。本阶段未训练任何网络。

## 8. 自动测试

- 命令：`.venv/bin/python -u -m pytest -q tests`
- 结果：**PASS**，114 passed / 0 failed / 0 errors
- 末行：`114 passed in 33.00s`
- 日志：`outputs/audit/pytest.log`（SHA256 `7a6cb9a54efee9fa…`）

覆盖：M0/M1 原有 53 项 ＋ M3 新增 33 项（冻结、三种划分、单药策略、训练折变换、接口 smoke）。

## 9. Git 与可复现性

- 基线：`cdd9c8b`
- 当前 commit：`ccc75669fc6f1f85ae40330da914d07eec26d51b`
- 工作区：clean
- 生成命令：

```bash
.venv/bin/python scripts/03_generate_splits.py --scenario all
.venv/bin/python scripts/03_verify_splits.py
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/04_write_m3_report.py
```

同一 commit + 配置 + 种子产生相同 `assignment_sha256`/`content_sha256`；所有 split 的 `split_metadata.json` 记录 assignment、scaler 与内容哈希。

## 10. 遗留问题

1. Carboplatin/Oxaliplatin 图不可用（配位键），影响 1,716 个矩阵；需专门的铂配合物图表示或 2D 描述符后才能纳入。
2. EFM192B、UWB1289BRCA1 无 DepMap 映射；COLO320DM 仅有 COLO320 需要人工确认；OCUBM、UWB1289 无 CCLE 表达。
3. 数据集细胞系来自单一 O'Neil 面板（34 个），S3 每折 test 仅 6–7 个细胞，跨癌种泛化结论需谨慎。
4. S1/S2 使用 known-monotherapy 先验；若 M4 要做严格单药泛化，应改用 S3 的 strict 策略。
5. 未生成同源细胞系（lineage）分组划分；若 M4 需评估 lineage 泛化，可新增 S4。
6. 本阶段未定义缺失标签的掩码训练损失；M4 需在模型侧处理 NaN（当前 Primary-v1 全部标签有限）。

## 11. 本轮文件清单

```
configs/dataset_primary_v1.yaml
configs/m3.yaml
configs/splits/s1_random_condition.yaml
configs/splits/s2_unseen_pair.yaml
configs/splits/s3_unseen_cell.yaml
src/dosesyn/features/transform_train_only.py
src/dosesyn/splits/build.py
src/dosesyn/splits/mono_policy.py
src/dosesyn/splits/verify.py
src/dosesyn/data/dataset.py
scripts/03_generate_splits.py
scripts/03_verify_splits.py
scripts/04_write_m3_report.py
tests/test_m3_primary.py
tests/test_m3_split_integrity.py
tests/test_m3_mono_policy.py
tests/test_m3_transforms.py
tests/test_m3_dataset.py
docs/decisions/m3_primary_dataset.md
docs/decisions/m3_split_protocols.md
m3_report.md
```
