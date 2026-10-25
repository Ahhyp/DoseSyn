# M7-A 统一实验协议

## 1. 数据与标签

- 数据：M3 Primary-v1，18,599 个矩阵 / 18,326 个条件 / 34 个细胞系 / 36 个图可用药物。
- 划分：11 个冻结 split（S1 3 seeds、S2 3 seeds、S3 5 folds），本阶段不做任何重新划分。
- 标签：原始 `X/X0`，不裁剪、不取整、不替换。
- 单位：药物浓度 µM；外部适配层内部按需转换（comboKR 官方为 mol/L，已显式除以 1e6）。

## 2. 评价口径

主指标：

```
matrix_macro_RMSE = mean_i( sqrt( mean_{p=1..16} (y_ip - yhat_ip)^2 ) )
```

即先算每个矩阵 16 点 RMSE，再对矩阵等权平均；与 M4/M6-B 完全一致。

补充指标：

- matrix-macro MAE：先算矩阵 16 点 MAE，再等权平均。
- point RMSE / MAE：所有点合并计算。
- pooled Pearson：所有点合并计算。
- R²：所有点合并计算（常数标签时记为 None）。
- condition-macro RMSE：同一 condition 内矩阵 RMSE 平均，再对 condition 等权平均。

缺失值规则：冻结数据无缺失标签；预测不得出现 NaN/Inf。不插补测试标签。

## 3. 选择与测试隔离

1. 外部模型配置、超参数仅依据 S1 seed42 与 S3 fold0 的验证集确定。
2. 配置冻结后，才在 11 个 split 上读取测试集。
3. 测试集只用于最终报告，不参与早停、模型选择或超参数调整。
4. 训练/验证/测试矩阵集合由 M3 split parquet 决定，不允许点级随机拆分。
5. S3：测试细胞系的单药与组合标签均不可见；任何依赖测试细胞单药曲线的步骤均禁止。

## 4. 各模型适配说明

### comboKR 2.0

- 官方 BRAID 9 参数拟合与 RBF Nystrom 预测。
- 使用官方数据中的 18,235 个可匹配矩阵（364 个 MK-8776 组合缺失）。
- S3 不运行：官方 Hills 参数包含测试细胞单药曲线，违反严格隔离。
- 结果进入主排名（S1/S2）。

### DeepSynBa（adapted）

- 保留双分支 drug-cell 编码器 + 9 个 SynBa 参数头 + 曲面重建。
- 药物输入：Morgan 2048 fingerprint → 768（官方为 MoLFormer）。
- 细胞输入：CCLE 950/976 landmark 基因，缺失补 0（官方为 NCI-60）。
- 目标：X/X0 16 点曲面（官方为 SynBa 参数/协同标签）。
- 结论只能作为 supplementary，不进入主排名。

### DD-PRiSM

- 未找到公开可运行代码，本阶段不复现、不报告数字。

## 5. 预算与资源

| 模型 | 训练预算 | 硬件 |
| --- | --- | --- |
| comboKR | 官方默认参数，无 epoch；每 split 单次拟合 | CPU，隔离环境 `/tmp/m7venv` |
| DeepSynBa (adapted) | 60 epoch，bs512，patience 10 | 单张 V100/4090，batch 512 |
| M4/M6-B | 归档结果（200–240 epoch 预算） | V100/4090 |

预算不等价是已知限制；报告单列，不据此宣称方法优劣。

## 6. 可追溯性

每个外部模型 run 目录包含：

- `run_metadata.json`：模型、场景、seed/fold、split、训练/评估矩阵数量、指标、参数、耗时、特征适配说明。
- `predictions.parquet`：matrix_id / row_idx / col_idx / dose_a_um / dose_b_um / y_true / y_pred，16 行/矩阵。
- 原始代码路径、数据路径、依赖环境记录在配置与元数据中。

## 7. 运行命令

```bash
# comboKR（隔离环境）
/tmp/m7venv/bin/python src/dosesyn/baselines/external/combokr_adapter.py \
  --scenario s1_random_condition --seed 42 --split test \
  --out outputs/runs/m7a/combokr/s1_random_condition/seed_42/predictions.parquet

# DeepSynBa adapted（项目环境）
.venv/bin/python src/dosesyn/baselines/external/deepsynba_adapter.py \
  --scenario s1_random_condition --seed 42 --epochs 60 --patience 10 \
  --batch-size 512 --device cuda \
  --out outputs/runs/m7a/deepsynba/s1_random_condition/seed_42/predictions.parquet

# 汇总
.venv/bin/python scripts/16_aggregate_m7a.py
```
