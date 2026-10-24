# M6-B 正式实验协议

本文件冻结 M6-B 的数据、模型与评估协议。所有正式结果必须按此执行；任何偏离都需要在
`m6b_report.md` 中记录原因。

## 1. 数据与划分

- 使用 M3 冻结的 11 个 split，不重新划分、不改变筛选规则：
  - S1 Random Condition：seed 13 / 42 / 2026
  - S2 Unseen Pair：seed 13 / 42 / 2026
  - S3 Unseen Cell：fold 0–4
- 主数据集为 `oneil-primary-v1`：36 个可用药物（排除 Carboplatin、Oxaliplatin）、
  34 个细胞系、18,599 个完整 4×4 矩阵。
- 训练/验证/测试的矩阵与条件互斥；同一 `condition_id` 不跨 split；批次重复不拆开。

## 2. 开发阶段与测试隔离

阶段 1（验证集开发）只允许：

- S1 seed42 验证集
- S3 fold0 验证集

阶段 1 严禁读取任何 test 标签。`run_m6b` 仅在 `eval_test=True` 时加载 test split，
且 `eval_test` 只在模型结构与训练协议冻结后由正式脚本传入。

阶段 2（正式 11 split）才允许每个 split 读取一次 test：

- 测试标签只用于计算最终指标与保存预测；
- 不参与 epoch 选择、损失权重、结构选择或早停；
- 每个 run 的 `best.pt` 由验证集 matrix-macro RMSE 选出，然后才创建 test 预测。

## 3. 模型变体

核心主干为 M6-A T3 `b3sym_fourier`：

| 编号 | 配置 |
| --- | --- |
| B0 | M4 B3 exact，仅作参考 |
| B1 | T3 symmetric Fourier |
| B2 | T3 + Monotherapy Head |
| B3 | T3 + Mono + Boundary |
| B4 | T3 + Mono + Boundary + Smooth |
| B5 | T3 + Regularized FiLM |
| B6 | T3 + Regularized FiLM + Mono + Boundary |

- 所有变体共享药物 GINE、细胞 MLP、Fourier 剂量编码器、隐藏维度、数据与损失实现；
- 每个模块均可独立开关，实际生效配置写入每个 run 的 `config.json`；
- 不使用 M5 C0 ordered 头作为正式主干；
- B6 只在 B5 与 B3/B4 的结果支持时运行。

## 4. 损失与约束

- 组合损失：逐点 Huber，`delta=1.0`；不裁剪标签。
- 单药损失：共享 MonoHead 的逐曲线平均 Huber，`λ_m` 主实验取 0.1。
- 边界损失：显式 zero flag，`F(A,x,B,0)` 与 `M(A,x)`、`F(A,0,B,y)` 与 `M(B,y)`；
  不是实测零剂量误差。
- 平滑损失：同一矩阵内部、非均匀 log-dose 网格二阶差分；不假设单调、不跨矩阵；
  B4 在 epoch 10 后启用 `λ_s=0.01`。
- FiLM：零初始化；B5 将 gamma/beta 调制减半，dropout 0.15，weight decay 5e-4。

## 5. 训练与选择

- 上限 240 epochs，patience 35，min 10；早停指标为验证集 matrix-macro RMSE。
- AdamW，lr 1e-3，weight decay 1e-4，CosineAnnealing（T_max=epochs）。
- batch 64，eval batch 128；图编码使用 `vocab_graph_batch`，每批只编码 36 个唯一药物图。
- 每个 run 保存 `best.pt`、`train_log.csv`、`val_metrics.json`、`predictions_val.parquet`、
  `resources.json`；正式 run 额外保存 `test_metrics.json` 与 `predictions_test.parquet`。

## 6. 模型选择规则

1. 先比较 B1–B5 在 S1 seed42 与 S3 fold0 的验证集 matrix-macro RMSE。
2. 选择均值最低者，但若相对 B1 的均值增益 < 0.001 或 S3 恶化 > 0.002，则保留 B1。
3. 不允许仅凭 S1 最优选择 S3 明显退化的模型。
4. 边界误差下降不能抵消组合 RMSE 的显著退化。
5. 选择结果写入 `m6b_model_selection.md` 与 `docs/audit/m6b_model_selection.json`。

## 7. 正式评估

- 正式模型：B1 + 验证集选出的最终模型（若不同）。
- 每个模型在每个 split 上训练一次、验证选 checkpoint、测试一次。
- 报告 test matrix-macro RMSE、point RMSE/MAE、Pearson、R²、参数量、耗时、显存。
- 与 M4 B3 同 split 结果配对比较；不把不同 split 的 test 与 val 混在一起。

## 8. 诚实性约束

- 不因结果不好而删除 run、改测试集或改划分。
- 失败/中止实验保留在报告中并说明原因。
- 药理一致性误差是模型自洽诊断，不能表述为真实零剂量实验误差。
- 测试集未参与任何调参；所有结论限定在 11 个冻结 split 上。
