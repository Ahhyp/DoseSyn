# DoseSyn M4 基线训练报告
生成时间：2026-10-24T07:43:14+00:00  
配置：`configs/m4_training.yaml`（m4-v1）
## 1. 结论
- 成功运行 **44** 个（失败 0 个）。按 scenario 等权平均的测试 `matrix_macro_RMSE` 最优模型为 **B3 GINE-Base**（0.1228）。
- B2 等权平均 0.1273，B3 0.1228；B2 在 S1/S2（已见条件/已见药物对）更好，B3 在 S3（未见细胞系）略优。
- 主指标：`matrix_macro_RMSE`（每个 4×4 矩阵先算 RMSE，再对矩阵取宏平均）；模型选择/早停仅用验证集。
- 测试标签为原始 `X/X0`，未裁剪；所有预测逐点保存。

## 2. 固定流程与防泄漏
- 直接读取 M3 冻结的 `train/val/test_conditions.parquet`，未重新划分、未做点级随机拆分。
- S1/S2 用 3 个种子，S3 用 5 折；同一 condition/batch 的矩阵不跨集合。
- 特征标准化只使用训练折拟合的 scaler（M3 产物），测试集不参与任何统计。
- B0/B1 不训练梯度；B1 仅在训练折上拟合。
- 单药监督范围：M4 的四个基线均不使用单药标签，单药曲线实际使用 0 条；S3 的 strict_cell_isolation 分配保持封存。

### 模型配置
- B0：训练折均值常数；B1：HistGradientBoosting（max_iter=400, lr=0.05, leaves=31）。
- B2：细胞特征 50 + 两个药物 embedding 32 + 剂量/零标志 → MLP 256；B3：3 层 GINE（hidden 128）+ 均值池化 + 同构 MLP 头。
- 优化：AdamW lr=0.001, wd=0.0001, Huber δ=1.0, batch=64, epochs≤80, patience=15。

## 3. 数据完整性检查
- 数据源: M3 冻结 splits（未重新划分）
- split 数: 11
- 成功运行数: 44
- 失败运行数: 0
- 预测行数/matrix: 每矩阵 16 行（runner 内断言）
- 测试标签: 原始 X/X0 最大 2.237，最小 0.000；未裁剪
- 各运行测试点 X/X0>1 合计: 160064
- 各运行测试点 X/X0=0 合计: 1308
- 负预测点数: 10379
- 早停依据: 仅验证集 matrix_macro_RMSE
- 单药监督使用: 0 条单药曲线（四个基线均不使用单药标签）
- 随机种子: [13, 42, 2026]
- Git commit: b736b6c3b9612ca22e023c0d297ae8e80692b8d9

## 4. 各模型 × 场景测试结果
| 模型 | 场景 | runs | macro RMSE | ±std | point RMSE | Pearson | R² | 参数量 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 Mean | s1_random_condition | 3 | 0.3060 | 0.0004 | 0.3185 | — | -0.0002 | 1 |
| B0 Mean | s2_unseen_pair | 3 | 0.3022 | 0.0082 | 0.3147 | — | -0.0047 | 1 |
| B0 Mean | s3_unseen_cell | 5 | 0.3066 | 0.0049 | 0.3188 | — | -0.0010 | 1 |
| B1 Tree-Dose | s1_random_condition | 3 | 0.2467 | 0.0020 | 0.2629 | 0.5655 | 0.3185 | 24400 |
| B1 Tree-Dose | s2_unseen_pair | 3 | 0.2497 | 0.0022 | 0.2658 | 0.5355 | 0.2833 | 24400 |
| B1 Tree-Dose | s3_unseen_cell | 5 | 0.2819 | 0.0058 | 0.2979 | 0.3703 | 0.1254 | 24400 |
| B2 MLP-Dose | s1_random_condition | 3 | 0.0764 | 0.0002 | 0.0890 | 0.9606 | 0.9220 | 65153 |
| B2 MLP-Dose | s2_unseen_pair | 3 | 0.1022 | 0.0143 | 0.1215 | 0.9222 | 0.8488 | 65153 |
| B2 MLP-Dose | s3_unseen_cell | 5 | 0.2033 | 0.0127 | 0.2262 | 0.7213 | 0.4936 | 65153 |
| B3 GINE-Base | s1_random_condition | 3 | 0.0795 | 0.0014 | 0.0936 | 0.9561 | 0.9136 | 284545 |
| B3 GINE-Base | s2_unseen_pair | 3 | 0.0897 | 0.0025 | 0.1053 | 0.9421 | 0.8873 | 284545 |
| B3 GINE-Base | s3_unseen_cell | 5 | 0.1994 | 0.0182 | 0.2231 | 0.7287 | 0.5054 | 284545 |

## 5. 逐 split 明细
| 模型 | 场景 | split | macro RMSE | point RMSE | Pearson | R² | 最佳 epoch | wall s | GPU MB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 Mean | s1_random_condition | seed_13 | 0.3065 | 0.3188 | — | -0.0000 | — | 4.4 | 0.0 |
| B0 Mean | s1_random_condition | seed_2026 | 0.3057 | 0.3184 | — | -0.0005 | — | 4.5 | 0.0 |
| B0 Mean | s1_random_condition | seed_42 | 0.3059 | 0.3183 | — | -0.0000 | — | 4.5 | 0.0 |
| B1 Tree-Dose | s1_random_condition | seed_13 | 0.2487 | 0.2646 | 0.5590 | 0.3111 | — | 11.7 | 0.0 |
| B1 Tree-Dose | s1_random_condition | seed_2026 | 0.2447 | 0.2612 | 0.5724 | 0.3263 | — | 11.2 | 0.0 |
| B1 Tree-Dose | s1_random_condition | seed_42 | 0.2466 | 0.2628 | 0.5652 | 0.3181 | — | 11.2 | 0.0 |
| B2 MLP-Dose | s1_random_condition | seed_13 | 0.0763 | 0.0877 | 0.9620 | 0.9243 | — | 45.7 | 29.4 |
| B2 MLP-Dose | s1_random_condition | seed_2026 | 0.0766 | 0.0898 | 0.9596 | 0.9203 | — | 47.2 | 31.5 |
| B2 MLP-Dose | s1_random_condition | seed_42 | 0.0762 | 0.0893 | 0.9601 | 0.9213 | — | 47.5 | 29.4 |
| B3 GINE-Base | s1_random_condition | seed_13 | 0.0805 | 0.0954 | 0.9545 | 0.9105 | — | 529.9 | 111.1 |
| B3 GINE-Base | s1_random_condition | seed_2026 | 0.0799 | 0.0936 | 0.9561 | 0.9135 | — | 364.3 | 98.6 |
| B3 GINE-Base | s1_random_condition | seed_42 | 0.0779 | 0.0918 | 0.9577 | 0.9167 | — | 239.2 | 111.1 |
| B0 Mean | s2_unseen_pair | seed_13 | 0.3116 | 0.3234 | — | -0.0107 | — | 2.9 | 0.0 |
| B0 Mean | s2_unseen_pair | seed_2026 | 0.2988 | 0.3117 | — | -0.0000 | — | 2.7 | 0.0 |
| B0 Mean | s2_unseen_pair | seed_42 | 0.2963 | 0.3091 | — | -0.0035 | — | 2.9 | 0.0 |
| B1 Tree-Dose | s2_unseen_pair | seed_13 | 0.2522 | 0.2691 | 0.5530 | 0.3003 | — | 13.2 | 0.0 |
| B1 Tree-Dose | s2_unseen_pair | seed_2026 | 0.2478 | 0.2642 | 0.5305 | 0.2812 | — | 12.5 | 0.0 |
| B1 Tree-Dose | s2_unseen_pair | seed_42 | 0.2491 | 0.2640 | 0.5232 | 0.2683 | — | 13.0 | 0.0 |
| B2 MLP-Dose | s2_unseen_pair | seed_13 | 0.1173 | 0.1449 | 0.8945 | 0.7972 | — | 70.7 | 31.5 |
| B2 MLP-Dose | s2_unseen_pair | seed_2026 | 0.1006 | 0.1150 | 0.9298 | 0.8639 | — | 68.3 | 31.5 |
| B2 MLP-Dose | s2_unseen_pair | seed_42 | 0.0888 | 0.1045 | 0.9423 | 0.8853 | — | 115.8 | 31.5 |
| B3 GINE-Base | s2_unseen_pair | seed_13 | 0.0902 | 0.1053 | 0.9449 | 0.8929 | — | 537.6 | 98.6 |
| B3 GINE-Base | s2_unseen_pair | seed_2026 | 0.0870 | 0.1039 | 0.9430 | 0.8889 | — | 514.5 | 123.7 |
| B3 GINE-Base | s2_unseen_pair | seed_42 | 0.0919 | 0.1069 | 0.9385 | 0.8801 | — | 305.1 | 98.6 |
| B0 Mean | s3_unseen_cell | fold_0 | 0.3054 | 0.3196 | — | -0.0007 | — | 3.1 | 0.0 |
| B0 Mean | s3_unseen_cell | fold_1 | 0.3100 | 0.3220 | — | -0.0000 | — | 3.0 | 0.0 |
| B0 Mean | s3_unseen_cell | fold_2 | 0.3009 | 0.3131 | — | -0.0000 | — | 3.1 | 0.0 |
| B0 Mean | s3_unseen_cell | fold_3 | 0.3035 | 0.3152 | — | -0.0034 | — | 3.0 | 0.0 |
| B0 Mean | s3_unseen_cell | fold_4 | 0.3130 | 0.3241 | — | -0.0010 | — | 3.0 | 0.0 |
| B1 Tree-Dose | s3_unseen_cell | fold_0 | 0.2893 | 0.3081 | 0.3109 | 0.0701 | — | 12.1 | 0.0 |
| B1 Tree-Dose | s3_unseen_cell | fold_1 | 0.2742 | 0.2891 | 0.4416 | 0.1942 | — | 11.3 | 0.0 |
| B1 Tree-Dose | s3_unseen_cell | fold_2 | 0.2822 | 0.2973 | 0.3423 | 0.0984 | — | 12.6 | 0.0 |
| B1 Tree-Dose | s3_unseen_cell | fold_3 | 0.2788 | 0.2938 | 0.3819 | 0.1279 | — | 11.0 | 0.0 |
| B1 Tree-Dose | s3_unseen_cell | fold_4 | 0.2851 | 0.3010 | 0.3749 | 0.1366 | — | 11.4 | 0.0 |
| B2 MLP-Dose | s3_unseen_cell | fold_0 | 0.2053 | 0.2330 | 0.6935 | 0.4681 | — | 52.1 | 31.5 |
| B2 MLP-Dose | s3_unseen_cell | fold_1 | 0.1935 | 0.2167 | 0.7716 | 0.5472 | — | 35.3 | 29.4 |
| B2 MLP-Dose | s3_unseen_cell | fold_2 | 0.2243 | 0.2470 | 0.6499 | 0.3776 | — | 41.0 | 31.5 |
| B2 MLP-Dose | s3_unseen_cell | fold_3 | 0.1942 | 0.2142 | 0.7510 | 0.5366 | — | 22.3 | 29.4 |
| B2 MLP-Dose | s3_unseen_cell | fold_4 | 0.1991 | 0.2200 | 0.7407 | 0.5387 | — | 21.1 | 29.4 |
| B3 GINE-Base | s3_unseen_cell | fold_0 | 0.2070 | 0.2375 | 0.6987 | 0.4475 | — | 268.5 | 98.6 |
| B3 GINE-Base | s3_unseen_cell | fold_1 | 0.1849 | 0.2021 | 0.7874 | 0.6062 | — | 109.6 | 111.1 |
| B3 GINE-Base | s3_unseen_cell | fold_2 | 0.2269 | 0.2508 | 0.6366 | 0.3585 | — | 172.1 | 98.6 |
| B3 GINE-Base | s3_unseen_cell | fold_3 | 0.1825 | 0.2063 | 0.7681 | 0.5700 | — | 126.3 | 102.8 |
| B3 GINE-Base | s3_unseen_cell | fold_4 | 0.1956 | 0.2186 | 0.7527 | 0.5445 | — | 97.1 | 98.6 |

## 7. 资源与复现
- GPU：NVIDIA V100-PCIE-32GB（32 GB）；作业未独占 GPU，最多 4 个作业共享一张卡，显存占用约 0.1 GB。
- 调度：Torque/PBS `qsub`，节点 node1；作业脚本自重定向日志（PBS `-o` 在本集群不生效）。
- 命令：
```bash
.venv/bin/python scripts/04_run_baseline.py --scenario s1_random_condition --seed 42 --models b0_mean b1_tree b2_mlp b3_gine --device cuda
.venv/bin/python scripts/05_aggregate_m4.py
.venv/bin/python -m pytest -q tests
```
- 生成图：`outputs/figures/m4/curves/b2_mlp__s1_random_condition__seed_13.svg`, `outputs/figures/m4/curves/b2_mlp__s1_random_condition__seed_2026.svg`, `outputs/figures/m4/curves/b2_mlp__s1_random_condition__seed_42.svg`, `outputs/figures/m4/curves/b2_mlp__s2_unseen_pair__seed_13.svg`, `outputs/figures/m4/curves/b2_mlp__s2_unseen_pair__seed_2026.svg`, `outputs/figures/m4/curves/b2_mlp__s2_unseen_pair__seed_42.svg`, `outputs/figures/m4/curves/b2_mlp__s3_unseen_cell__fold_0.svg`, `outputs/figures/m4/curves/b2_mlp__s3_unseen_cell__fold_1.svg`, `outputs/figures/m4/curves/b2_mlp__s3_unseen_cell__fold_2.svg`, `outputs/figures/m4/curves/b2_mlp__s3_unseen_cell__fold_3.svg`, `outputs/figures/m4/curves/b2_mlp__s3_unseen_cell__fold_4.svg`, `outputs/figures/m4/curves/b3_gine__s1_random_condition__seed_13.svg`, `outputs/figures/m4/curves/b3_gine__s1_random_condition__seed_2026.svg`, `outputs/figures/m4/curves/b3_gine__s1_random_condition__seed_42.svg`, `outputs/figures/m4/curves/b3_gine__s2_unseen_pair__seed_13.svg`, `outputs/figures/m4/curves/b3_gine__s2_unseen_pair__seed_2026.svg`, `outputs/figures/m4/curves/b3_gine__s2_unseen_pair__seed_42.svg`, `outputs/figures/m4/curves/b3_gine__s3_unseen_cell__fold_0.svg`, `outputs/figures/m4/curves/b3_gine__s3_unseen_cell__fold_1.svg`, `outputs/figures/m4/curves/b3_gine__s3_unseen_cell__fold_2.svg`, `outputs/figures/m4/curves/b3_gine__s3_unseen_cell__fold_3.svg`, `outputs/figures/m4/curves/b3_gine__s3_unseen_cell__fold_4.svg`, `outputs/figures/m4/macro_rmse_by_model_scenario.svg`
- 自动测试：131 passed（outputs/audit/pytest.log）

## 8. 尚未解决的问题
- B3 GINE-Base 仅用全局均值池化 + 拼接，不含 FiLM、单药边界、平滑正则或协同重建（按任务书留给 M5）。
- 本阶段四个基线均不使用单药监督标签：实际进入训练/验证/测试的单药曲线为 0 条；S1/S2 的 known_monotherapy 仅是标签访问协议，S3 的 strict_cell_isolation 分配保持封存，未用于 M4。
- S2 的 unseen pair 场景中，药物对完全未在训练中出现；若未来启用单药先验，也不能当作独立测试证据。
- S3 某些折的训练/测试细胞癌症类型不均衡（如 fold0 test 无前列腺），指标方差较大。
- 本阶段未做超参搜索；S1/S2 上 B2 优于 B3、S3 上 B3 略优，差距不足以断言图结构无用，需在 M5 消融。
- 预测允许为负、未裁剪到 [0, 2.5]；后续若需要物理约束应在模型侧处理，而不是后处理测试标签。
- 未使用 AMP；V100 显存占用很低，训练瓶颈在 Python/小批量，而非显存。
- 正式矩阵训练前有 2 个作业（20548/20549）因尝试安装 matplotlib 时 numpy 目录损坏而失败；修复后重跑，结果均来自成功作业。
