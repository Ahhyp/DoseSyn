# DoseSyn M7-B：剂量位置插值与稀疏曲面泛化

**结论：M7-B PASS**（中心留出 36 个正式运行，4 个模型；稀疏实验 完成；边界实验 完成）。

## 1. 实验协议

- 冻结 M3 的 11 个 split（S1 3 seeds、S2 3 seeds、S3 5 folds），不重新划分。
- 4×4 矩阵按 row-major 展平；掩码对转置对称，因此交换药物顺序后观测/留出位置保持一致。
- 训练损失仅使用 observed 位置；验证/测试指标仅使用 held-out 位置；测试只在 checkpoint 选定后读取一次。
- 训练预算统一 160 epoch、patience 30、cosine、batch 64；未根据测试结果调参。
- 主指标：held-out matrix-macro RMSE（先算每个矩阵 RMSE，再对矩阵取平均）。

### 掩码定义

| mask | n_observed | n_held | held_indices | 说明 |
| --- | --- | --- | --- | --- |
| center_holdout_75 | 12 | 4 | [5, 6, 9, 10] | 中心 2x2 四点留出，其余 12 点可观测（插值主实验） |
| checkerboard_holdout_50 | 8 | 8 | [1, 3, 4, 6, 9, 11, 12, 14] | 棋盘交错留出 8 点，观测 8 点（50% observed） |
| corners_holdout_25 | 4 | 12 | [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14] | 仅保留四角 4 点观测，留出 12 点（25% observed） |
| boundary_holdout_50 | 9 | 7 | [3, 7, 11, 12, 13, 14, 15] | 最高剂量 A 整行 + 最高剂量 B 整列，留出 7 点（边界外推） |

## 2. B1 中心 2×2 留出（主实验）

中心四点 (1,1),(1,2),(2,1),(2,2) 在训练中完全不可见；其余 12 点参与损失。

| display | scenario | n_runs | macro_mean | macro_std | mae_mean | point_rmse_mean | pearson_mean | r2_mean | epoch_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M4 B2 MLP-Dose | s1_random_condition | 1 | 0.0739 | nan | 0.0641 | 0.0924 | 0.9473 | 0.8974 | 147.0000 |
| M4 B3 GINE | s1_random_condition | 3 | 0.0765 | 0.0015 | 0.0664 | 0.0962 | 0.9433 | 0.8894 | 146.0000 |
| M6-B B1 (T3) | s1_random_condition | 3 | 0.0988 | 0.0011 | 0.0849 | 0.1212 | 0.9099 | 0.8245 | 128.6667 |
| M6-B B5 (DoseSyn) | s1_random_condition | 3 | 0.0702 | 0.0005 | 0.0601 | 0.0898 | 0.9507 | 0.9038 | 142.3333 |
| M4 B2 MLP-Dose | s2_unseen_pair | 1 | 0.0921 | nan | 0.0803 | 0.1174 | 0.9096 | 0.8233 | 129.0000 |
| M4 B3 GINE | s2_unseen_pair | 3 | 0.0898 | 0.0053 | 0.0787 | 0.1139 | 0.9174 | 0.8399 | 134.3333 |
| M6-B B1 (T3) | s2_unseen_pair | 3 | 0.1185 | 0.0005 | 0.1029 | 0.1441 | 0.8667 | 0.7453 | 94.3333 |
| M6-B B5 (DoseSyn) | s2_unseen_pair | 3 | 0.0811 | 0.0058 | 0.0700 | 0.1051 | 0.9298 | 0.8632 | 137.0000 |
| M4 B2 MLP-Dose | s3_unseen_cell | 1 | 0.2192 | nan | 0.2022 | 0.2592 | 0.5516 | 0.2030 | 101.0000 |
| M4 B3 GINE | s3_unseen_cell | 5 | 0.2032 | 0.0197 | 0.1876 | 0.2412 | 0.6045 | 0.2964 | 32.2000 |
| M6-B B1 (T3) | s3_unseen_cell | 5 | 0.2126 | 0.0252 | 0.1950 | 0.2476 | 0.5927 | 0.2587 | 18.6000 |
| M6-B B5 (DoseSyn) | s3_unseen_cell | 5 | 0.2051 | 0.0191 | 0.1882 | 0.2432 | 0.5923 | 0.2883 | 27.8000 |

![center](outputs/figures/m7b/b1_center_rmse.svg)

## 3. 稀疏观测（B2）

| mask | display | scenario | n_runs | macro_mean | macro_std | mae_mean |
| --- | --- | --- | --- | --- | --- | --- |
| center_holdout_75 | M4 B2 MLP-Dose | s1_random_condition | 1 | 0.0739 | nan | 0.0641 |
| center_holdout_75 | M4 B3 GINE | s1_random_condition | 3 | 0.0765 | 0.0015 | 0.0664 |
| checkerboard_holdout_50 | M4 B3 GINE | s1_random_condition | 1 | 0.0784 | nan | 0.0639 |
| corners_holdout_25 | M4 B3 GINE | s1_random_condition | 1 | 0.1211 | nan | 0.0990 |
| center_holdout_75 | M6-B B1 (T3) | s1_random_condition | 3 | 0.0988 | 0.0011 | 0.0849 |
| center_holdout_75 | M6-B B5 (DoseSyn) | s1_random_condition | 3 | 0.0702 | 0.0005 | 0.0601 |
| checkerboard_holdout_50 | M6-B B5 (DoseSyn) | s1_random_condition | 1 | 0.0710 | nan | 0.0567 |
| corners_holdout_25 | M6-B B5 (DoseSyn) | s1_random_condition | 1 | 0.1572 | nan | 0.1248 |
| center_holdout_75 | M4 B2 MLP-Dose | s2_unseen_pair | 1 | 0.0921 | nan | 0.0803 |
| center_holdout_75 | M4 B3 GINE | s2_unseen_pair | 3 | 0.0898 | 0.0053 | 0.0787 |
| checkerboard_holdout_50 | M4 B3 GINE | s2_unseen_pair | 1 | 0.0894 | nan | 0.0727 |
| corners_holdout_25 | M4 B3 GINE | s2_unseen_pair | 1 | 0.1273 | nan | 0.1039 |
| center_holdout_75 | M6-B B1 (T3) | s2_unseen_pair | 3 | 0.1185 | 0.0005 | 0.1029 |
| center_holdout_75 | M6-B B5 (DoseSyn) | s2_unseen_pair | 3 | 0.0811 | 0.0058 | 0.0700 |
| checkerboard_holdout_50 | M6-B B5 (DoseSyn) | s2_unseen_pair | 1 | 0.0838 | nan | 0.0668 |
| corners_holdout_25 | M6-B B5 (DoseSyn) | s2_unseen_pair | 1 | 0.1839 | nan | 0.1498 |
| center_holdout_75 | M4 B2 MLP-Dose | s3_unseen_cell | 1 | 0.2192 | nan | 0.2022 |
| center_holdout_75 | M4 B3 GINE | s3_unseen_cell | 5 | 0.2032 | 0.0197 | 0.1876 |
| checkerboard_holdout_50 | M4 B3 GINE | s3_unseen_cell | 1 | 0.2095 | nan | 0.1838 |
| corners_holdout_25 | M4 B3 GINE | s3_unseen_cell | 1 | 0.2305 | nan | 0.2055 |
| center_holdout_75 | M6-B B1 (T3) | s3_unseen_cell | 5 | 0.2126 | 0.0252 | 0.1950 |
| center_holdout_75 | M6-B B5 (DoseSyn) | s3_unseen_cell | 5 | 0.2051 | 0.0191 | 0.1882 |
| checkerboard_holdout_50 | M6-B B5 (DoseSyn) | s3_unseen_cell | 1 | 0.2091 | nan | 0.1811 |
| corners_holdout_25 | M6-B B5 (DoseSyn) | s3_unseen_cell | 1 | 0.2354 | nan | 0.2046 |

## 4. 高剂量边界外推（B3）

留出最高剂量 A 整行与最高剂量 B 整列（7 点），训练仅用其余 9 点。该任务与中心插值不可直接混合排名。

| display | scenario | n_runs | macro_mean | macro_std | bias_mean | pred_min | pred_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M4 B3 GINE | s1_random_condition | 1 | 0.1206 | nan | 0.0169 | -0.3315 | 1.4057 |
| M6-B B1 (T3) | s1_random_condition | 1 | 0.2406 | nan | 0.1051 | -0.3830 | 1.0971 |
| M6-B B5 (DoseSyn) | s1_random_condition | 1 | 0.2340 | nan | 0.1646 | -0.1213 | 1.3213 |
| M4 B3 GINE | s2_unseen_pair | 1 | 0.1229 | nan | 0.0321 | -0.3863 | 1.2874 |
| M6-B B1 (T3) | s2_unseen_pair | 1 | 0.2367 | nan | 0.1438 | -0.1791 | 1.1441 |
| M6-B B5 (DoseSyn) | s2_unseen_pair | 1 | 0.2336 | nan | 0.1671 | -0.1877 | 1.3819 |
| M4 B3 GINE | s3_unseen_cell | 1 | 0.2072 | nan | -0.0251 | -0.4917 | 1.2633 |
| M6-B B1 (T3) | s3_unseen_cell | 1 | 0.2555 | nan | 0.0601 | -0.1302 | 1.1362 |
| M6-B B5 (DoseSyn) | s3_unseen_cell | 1 | 0.2640 | nan | 0.0916 | -0.0801 | 1.0096 |

## 5. 配对差异

| mask | scenario | model | n | mean_delta | std_delta |
| --- | --- | --- | --- | --- | --- |
| boundary_holdout_50 | s1_random_condition | M4 B3 GINE | 1 | -0.1134 | nan |
| boundary_holdout_50 | s1_random_condition | M6-B B1 (T3) | 1 | 0.0065 | nan |
| boundary_holdout_50 | s1_random_condition | M6-B B5 (DoseSyn) | 1 | 0.0000 | nan |
| boundary_holdout_50 | s2_unseen_pair | M4 B3 GINE | 1 | -0.1106 | nan |
| boundary_holdout_50 | s2_unseen_pair | M6-B B1 (T3) | 1 | 0.0031 | nan |
| boundary_holdout_50 | s2_unseen_pair | M6-B B5 (DoseSyn) | 1 | 0.0000 | nan |
| boundary_holdout_50 | s3_unseen_cell | M4 B3 GINE | 1 | -0.0568 | nan |
| boundary_holdout_50 | s3_unseen_cell | M6-B B1 (T3) | 1 | -0.0085 | nan |
| boundary_holdout_50 | s3_unseen_cell | M6-B B5 (DoseSyn) | 1 | 0.0000 | nan |
| center_holdout_75 | s1_random_condition | M4 B2 MLP-Dose | 1 | 0.0038 | nan |
| center_holdout_75 | s1_random_condition | M4 B3 GINE | 3 | 0.0063 | 0.0010 |
| center_holdout_75 | s1_random_condition | M6-B B1 (T3) | 3 | 0.0286 | 0.0010 |
| center_holdout_75 | s1_random_condition | M6-B B5 (DoseSyn) | 3 | 0.0000 | 0.0000 |
| center_holdout_75 | s2_unseen_pair | M4 B2 MLP-Dose | 1 | 0.0042 | nan |
| center_holdout_75 | s2_unseen_pair | M4 B3 GINE | 3 | 0.0087 | 0.0010 |
| center_holdout_75 | s2_unseen_pair | M6-B B1 (T3) | 3 | 0.0374 | 0.0063 |
| center_holdout_75 | s2_unseen_pair | M6-B B5 (DoseSyn) | 3 | 0.0000 | 0.0000 |
| center_holdout_75 | s3_unseen_cell | M4 B2 MLP-Dose | 1 | 0.0035 | nan |
| center_holdout_75 | s3_unseen_cell | M4 B3 GINE | 5 | -0.0018 | 0.0103 |
| center_holdout_75 | s3_unseen_cell | M6-B B1 (T3) | 5 | 0.0075 | 0.0116 |
| center_holdout_75 | s3_unseen_cell | M6-B B5 (DoseSyn) | 5 | 0.0000 | 0.0000 |
| checkerboard_holdout_50 | s1_random_condition | M4 B3 GINE | 1 | 0.0074 | nan |
| checkerboard_holdout_50 | s1_random_condition | M6-B B5 (DoseSyn) | 1 | 0.0000 | nan |
| checkerboard_holdout_50 | s2_unseen_pair | M4 B3 GINE | 1 | 0.0056 | nan |
| checkerboard_holdout_50 | s2_unseen_pair | M6-B B5 (DoseSyn) | 1 | 0.0000 | nan |
| checkerboard_holdout_50 | s3_unseen_cell | M4 B3 GINE | 1 | 0.0004 | nan |
| checkerboard_holdout_50 | s3_unseen_cell | M6-B B5 (DoseSyn) | 1 | 0.0000 | nan |
| corners_holdout_25 | s1_random_condition | M4 B3 GINE | 1 | -0.0361 | nan |
| corners_holdout_25 | s1_random_condition | M6-B B5 (DoseSyn) | 1 | 0.0000 | nan |
| corners_holdout_25 | s2_unseen_pair | M4 B3 GINE | 1 | -0.0566 | nan |
| corners_holdout_25 | s2_unseen_pair | M6-B B5 (DoseSyn) | 1 | 0.0000 | nan |
| corners_holdout_25 | s3_unseen_cell | M4 B3 GINE | 1 | -0.0049 | nan |
| corners_holdout_25 | s3_unseen_cell | M6-B B5 (DoseSyn) | 1 | 0.0000 | nan |

## 6. 结论与局限

1. 中心插值最佳模型按场景：s1_random_condition = M6-B B5 (DoseSyn) (0.0702); s2_unseen_pair = M6-B B5 (DoseSyn) (0.0811); s3_unseen_cell = M4 B3 GINE (0.2032)。
2. 中心留出衡量的是网格位置插值，不是绝对浓度外推；不能与原始 16 点 RMSE 直接比较。
作为参考（**不同任务，不可直接排名**）：M6-B 全 16 点测试与本次中心 4 点留出的差异如下。中心留出普遍变差，符合“训练时未观测中心位置”的预期。

| model | scenario | full_grid_16pt | center_holdout_4pt | relative_change |
| --- | --- | --- | --- | --- |
| M6-B B5 (DoseSyn) | S1 | 0.0650 | 0.0702 | +7.9% |
| M6-B B5 (DoseSyn) | S2 | 0.0768 | 0.0811 | +5.7% |
| M6-B B5 (DoseSyn) | S3 | 0.1937 | 0.2051 | +5.8% |
| M6-B B1 (T3) | S1 | 0.0793 | 0.0988 | +24.6% |
| M6-B B1 (T3) | S2 | 0.1085 | 0.1185 | +9.2% |
| M6-B B1 (T3) | S3 | 0.1966 | 0.2126 | +8.1% |
3. FiLM/B5 与 T3/B1、GINE/B3 的差异见第 2、5 节；S3 未见细胞场景需按 fold 报告离散度。
4. B1（T3）的中心留出退化显著大于 B5（S1：+24.6% vs +7.9%）：Fourier 编码在缺少 FiLM 剂量调制时更易记住观测位置；FiLM 是 Fourier 插值的关键条件机制，而非可有可无的装饰。
5. 边界实验为高剂量压力测试，模型允许输出物理范围外的原始预测，报告未裁剪；边界与中心插值不可混合排名。
6. 50%/25% 观测比例仅在 3 个 dev split 上运行，用于趋势说明，不纳入主排名。
7. S3 训练未使用测试细胞系的单药或组合标签；单药边界项未在本阶段启用。
8. 稀疏与边界条件下排序会反转：50% 观测时 B5 与 B3 接近，25% 观测时 B5 明显变差；高剂量边界 B3 显著优于 B5。FiLM/Fourier 的收益依赖观测密度与目标区域，不能宣称在所有插值/外推条件下全面优于 GINE。

### 资源开销

| display | n_runs | params | wall_s | best_epoch |
| --- | --- | --- | --- | --- |
| M4 B2 MLP-Dose | 3 | 65153.0000 | 75.9927 | 129.0000 |
| M4 B3 GINE | 20 | 284545.0000 | 148.2845 | 98.5000 |
| M6-B B1 (T3) | 14 | 498818.0000 | 109.2178 | 47.0000 |
| M6-B B5 (DoseSyn) | 20 | 535682.0000 | 90.8433 | 26.5000 |

## 7. 复现

```bash
cd /home/kouzheng_2/AH_YeP/DoseSyn
.venv/bin/python scripts/18_run_m7b.py --group b1_center_all --force
.venv/bin/python scripts/18_run_m7b.py --group sparse_dev --force
.venv/bin/python scripts/18_run_m7b.py --group boundary_dev --force
.venv/bin/python scripts/19_aggregate_m7b.py
```

运行数：57；掩码版本：center_holdout_75=371a327b4ec46091, checkerboard_holdout_50=27b94d92a92ae7d4, corners_holdout_25=51575f98b2b555ec, boundary_holdout_50=846e59818a571c76。
## 8. Git 与环境

- 实验基线：`973fba6d5f95dba2e2f6ad71acecf38dadc1d8f3`（M7-A 完成时的 main）。
- 掩码版本：`center_holdout_75=371a327b4ec46091`、`checkerboard_holdout_50=27b94d92a92ae7d4`、`corners_holdout_25=51575f98b2b555ec`、`boundary_holdout_50=846e59818a571c76`。
- 运行环境：Python 3.10.21、pandas 1.3.5、numpy 1.21.6、torch 1.13.1、PyG 2.3.1、RDKit 2023.03.2。
- 计算节点：node4（2×RTX 4090，中心 B5/B1/B3）与 node1（2×V100，部分 B5 运行）；训练预算 160 epoch、patience 30、cosine、batch 64。
- 本阶段交付提交与工作区状态见阶段回复中的 `commit` 字段；未推送 force，未覆盖 M3–M7-A 产物。
