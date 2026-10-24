# DoseSyn M4 基线训练与评估体系执行任务书 V1.0
**项目路径**：`/home/kouzheng_2/AH_YeP/DoseSyn`  
**GitHub**：`https://github.com/Ahhyp/DoseSyn`  
**M3 基线提交**：`5d8a3097c469fe54cd6258619cee562a08675373`（执行前记录实际 `git rev-parse HEAD`）。  
**数据集**：`oneil-primary-v1`，M3 冻结子集：18,599 张 4×4 响应矩阵，18,326 条独立 `condition_id`，297,584 个响应点，34 细胞系、36 药物、539 无序药物对。  
**目标**：验证训练管线，获得公平、可复现的剂量响应预测基线，为 M5 DoseSyn 的真实创新模块建立参照。  
**禁止事项**：本轮**不实现完整 DoseSyn**，不引入 FiLM/复杂 cross-attention/单药边界损失/曲面平滑/协同指标重建；不修改 M0–M3 的数据或固定 split；不因为测试集成绩而重新选择超参数或数据清洗方案。

## 1. 开始前的验收门槛（P0）

1. 读取 `AGENTS.md`、总规划、`m3_report.md`、`docs/decisions/m3_split_protocols.md`、`src/dosesyn/data/dataset.py` 和 M3 配置；不要自创与 M3 不兼容的数据结构。
2. 运行 `git status --short`、`git rev-parse HEAD`、`.venv/bin/python -m pytest -q tests`、M3 划分校验；保存真实日志。对 114 项旧测试失败或哈希不匹配的情况立刻停下调查。
3. 记录实际 GPU 型号、显存、PyTorch、PyG、CUDA、RDKit、NumPy、sklearn 版本；不要假定有多少显存，也不要随意更新现有基础包。
4. 验证 `DoseSynDataset` 和 `collate_matrices` 输出的键、真实 dtype 与形状。每条训练记录是 4×4 矩阵而非一个剂量点，图的 batching 可直接复用。
5. 验证 M3 的 `S1 seed42`、`S2 seed42`、`S3 fold0` 可以正确加载 train/val/test；禁止在调参前加载测试集标签，允许最终评估脚本从只读测试分区读取真值。
6. **检查必要标签列与特征的独立性**：单药 `X/X0`、目标矩阵真实 `X/X0`、技术重复摘要、其他分区的 `condition_id` 不可成为预测输入；ID 只供检索与审计，不能作为可训练 ID embedding 代替药物结构或细胞表达。

## 2. 统一预测任务与评估口径（必须先实现并锁定）

输入：
- Drug A / Drug B：结构图（GINE）或 Morgan 指纹（RF/MLP），使用同一分子处理和映射；
- Cell：M3 的 50 维 Hallmark（主版），可选基因表达扩展另报；
- Dose：每张矩阵 4×4 浓度坐标；输入使用 M3 `log10(molar)` 训练折 scaler；
- 输出：每张 4×4 矩阵对应的 `X/X0` 实值预测。

**主标签 `X/X0` 不裁剪、不转二分类、不加 sigmoid；保留 0 与大于 1 的数值。**
- 为每张矩阵预测张量 `[B,4,4]`，或 `[B,16]` 再 reshape，保证 `(row_idx,col_idx)` 没有转置。
- `loss=Huber(delta=可配置)`，训练 loss 默认对每张矩阵内的 16 点求均值后再对 batch 内矩阵求均值。可以另报 MSE；不可用测试集选损失。
- 主评估指标固定为 **matrix_macro_RMSE**（各矩阵先算 RMSE，再等权平均；适用于模型早停），同时输出 **point_RMSE、point_MAE、pooled_Pearson、pooled_R²、condition_macro_RMSE**。
- 各指标明确单位为 `X/X0` 的无量纲原始尺度；模型可以在验证集比较 Huber 与 MSE，但最终真实评估必须使用原始标签。
- Pearson 使用全测试点整体汇总；按矩阵 Pearson 可作为附录，常值矩阵计算未定义时报告数量并不应填成 0。
- `condition_macro_RMSE`：同一 condition 的若干矩阵先均值再跨 condition 平均，用于检查重复批次权重偏移。
- S1/S2 为 3 个 seeds 的均值±标准差，S3 为 5 个 outer folds 的均值±标准差；不要把 11 个 split 合在一起平均成一个不分场景的分数。
- 保存逐剂量点预测：`model, scenario, seed_or_fold, split, matrix_id, condition_id, pair_id, cell_id, batch_id, drug_a_id, drug_b_id, row_idx, col_idx, dose_a_um, dose_b_um, y_true, y_pred`。
- **每个模型要和其余基线共享同一个 split、标签、特征覆盖和相同评估实现。** 禁止只比较不相同场景的最优指标。

## 3. 要完成的模型（M4 仅这些模型）

### B0：Train-only Mean Baseline（P0）
- 用当前 split 的训练点拟合全局 `X/X0` 均值（可另加 train-only dose-bin 均值作增强参考）。
- 对所有验证/测试样本返回常值（或 train-only dose bin 输出）。
- 作用：查错的下限参照；如果深度学习连 B0 都无法稳定改善，必须检查管线而不是马上堆网络。
- 明确不能用 val/test 的标签均值或测试剂量点标签构造预测。

### B1：传统机器学习剂量基线（P0）
- 采用 `RandomForestRegressor`、`ExtraTreesRegressor`、`HistGradientBoostingRegressor` 或能在现有环境稳定运行的同类方法；至少成功一个树基线，若环境允许可增加 Ridge 作为线性下限。
- 输入为 drug A/B 的 Morgan fingerprint + 50 维 Hallmark + 两药标准化 `log-dose`。
- 建议先用 256/512 bits fingerprint 防止 16× 大表造成内存膨胀，具体位数、radius、实现版本记录在配置里。
- 为每个剂量点构造独立数值行，但仍通过矩阵 ID / condition ID 确保 split 不泄漏；不可按剂量点重新切分。
- 按原始 drug_A/B 顺序和剂量对应输出；可以用规范药物顺序消除任意方向差异，但必须同时交换对应剂量，并与 MLP 采用可比较的输入规则。
- 大数据内存不够：先报告峰值内存和样本量，再使用训练集内部的**可复现抽样**或降低指纹维度/树数；必须标明实际训练点数量，避免悄悄只用部分训练集。

### B2：MLP-Dose（P0、最关键深度学习基线）
- 结构输入采用 Morgan fingerprint（推荐 512 bits ×2）+ Hallmark 50 维 + 两个标准化 log-dose。
- 两药可先独立使用**共享药物 MLP**输出 drug embeddings，细胞使用轻量 MLP，拼接 dose 后交给回归 MLP。
- 第一版可直接 concat，不强制交换不变；为避免弱基线，另加 `MLP-SymAvg` 作为可选强对照：对 drug/dose 同步交换的两种输入预测取平均。
- 参考起点：drug hidden=128，cell hidden=64，预测头 hidden=256→128→1，dropout=0.1，AdamW LR=1e-3，weight_decay=1e-4；以 val 指标选择，可调整但必须记录。
- 模型处理一个 `[B,4,4]` 任务，使用广播的 drug/cell 上下文，计算 16 个剂量点并 reshape；不得用 matrix_id 的可训练 embedding。
- `y_pred` 最后一层为 Linear，不得 Sigmoid 截断正响应。

### B3：GINE-Base（P0、M5 直接的比较基准）
- 使用 M2 缓存的 PyG 分子图，Drug A/B 由**共享的** GINE/GINEConv 结构编码器构造向量，建议 2–3 层、hidden=128、global mean/sum pooling（具体通过验证集选择）。
- 使用同一 Hallmark 50 维、相同剂量输入、相同 MLP 预测头；先采用普通 concat，**不引入 DoseSyn 的连续 Fourier/RBF、FiLM、单药边界和 smoothness**。
- 特别核查 M2 原子/键特征究竟是整数类别编码还是连续向量，选用适当 embedding/投影层；不能直接把高类别整数看成有物理顺序的可连续数值。确保 `edge_attr` 的维度符合 `GINEConv`。
- 复用 M3 `drug_batch` 与 `drug_role`，验证 `[2B,D]` 分子级向量能够按矩阵 A/B 顺序还原。
- 第一版不使用单药有标签监督。S1/S2 的 known-monotherapy 标签可用但**M4 基线均不读取这些标签作为训练特征或辅助 loss**，这保证四个主基线处于相同监督预算；单药辅助头属于 M5 的新增研究模块，应另做 same-supervision 消融。
- 在训练曲线上确认非 NaN、反向传播图/梯度有效、一个小数据子集能被过拟合。

## 4. 核心训练管线（P0）

请先写与模型无关的公共训练与评估接口：
- `train_one_epoch(model, loader, optimizer, loss_fn, device)`
- `evaluate(model, loader, metrics, save_predictions=False)`
- `fit_with_early_stopping(...)`
- `save_checkpoint_atomic`, `load_best_checkpoint`, `set_all_seeds`.
- 使用 M3 scaler，不重新在 val/test 拟合。
- 首轮保守训练建议 max_epochs=150、patience=20、batch_size=32 或 64 个矩阵；OOM 时改为 16 或梯度累积；GPU 可用就用 GPU，CPU 也要能 smoke。
- 以 `val_matrix_macro_RMSE` 选择 best checkpoint。训练期间绝不通过 test 指标优化。
- 早停之后载入 best checkpoint，**只进行一次最终 test 推理与汇总**；完整 11 split 跑法同理。
- 除 checkpoint 外，每次 run 必须保存 `config.yaml, split_id, git_commit, upstream_manifest_hash, seed, optimizer/lr_schedule, best_epoch, train_log.csv, val_metrics.json, test_metrics.json, predictions_test.parquet, resources.json`。
- 输出唯一 `run_id`，避免覆盖；失败 run 同样保存错误日志和已完成的 epoch，不能静默跳过。
- 运行时启用梯度裁剪（例如 1.0）与有限 loss 检查；异常终止需保存具体 batch 信息（不包含敏感数据）。
- 测量参数量、每 epoch 耗时、峰值显存；先确保准确、可比较，再制作图表。
- 记录标准化剂量 `dose_a/b` 是 `[B,4]` 各自轴还是 `[B,4,4]` 广播视图，防止矩阵轴混淆。

## 5. 执行次序与阶段停止点（不要一次盲目启动所有训练）

### M4-A｜公共框架、B0/B1
1. 确认 M3 数据接口与列类型。
2. 编写指标库和输出 schema，使用人工小矩阵验证 point RMSE、matrix_macro RMSE、Pearson 和 condition_macro。
3. B0 train only 训练/验证/测试。
4. B1 在 `S1 seed42` 完成树模型基线。
5. 输出训练代码/配置/性能和测试通过情况；此步通过再进入 M4-B。

### M4-B｜B2、B3 的最小可行性
1. 用 `S1 seed42` 只取**训练集合内部**的一个小规模固定子集，做 1 个 batch 前向/后向测试。
2. 对固定例如 16–64 张矩阵执行过拟合检查；验证训练损失显著下降（不要设不合理的绝对阈值）。
3. B2/B3 在 `S1 seed42` 正式训练，val early stopping 后输出 test 指标与预测文件。
4. 给出 B0/B1/B2/B3 的同 split 对比表；排查所有指标、残差分布、特别高的标签及 0 标签。
5. 如果 GINE-Base 不如 MLP-Dose，要先排查图编码/过拟合/学习率，不应直接宣称 GINE 必然优于 Morgan 指纹。

### M4-C｜扩大到 11 个 split
- 仅在 M4-B 完成、相同协议跑通后，按**预先确定**的验证集选择规则执行 S1 三种子、S2 三种子、S3 五折。
- 预算不够时可先跑 B0/B1/B2 的全 split，再逐步增加 B3；明确未完成项目，不伪造数字。
- S3 严格遵守 M3 mono 隔离（虽然 M4 不训练 mono head，也应运行访问审计）。
- 每个 split 中所有模型均使用对应训练折 scaler/同一数据索引。
- 汇总 `results_m4.csv` 和 `results_m4.md`，给出每折数值、均值±标准差和失败试验说明。
- 如果 S3 个别组织分布不均，应在结果讨论中提醒“仅 34 细胞、部分测试折缺少某癌种”。

## 6. 建议代码和配置结构

```text
configs/
  m4_common.yaml
  m4_baseline_mean.yaml
  m4_baseline_tree.yaml
  m4_mlp_dose.yaml
  m4_gine_base.yaml
scripts/
  04_run_baseline.py
  04_run_experiments.py
  04_evaluate_m4.py
  04_make_m4_report.py
src/dosesyn/
  models/
    baseline_mean.py
    baseline_tree.py
    mlp_dose.py
    gine_base.py
    encoders.py
  training/
    trainer.py
    metrics.py
    callbacks.py
    reproducibility.py
  data/
    baseline_features.py
tests/
  test_m4_metrics.py
  test_m4_baseline_features.py
  test_m4_mlp.py
  test_m4_gine.py
  test_m4_trainer.py
  test_m4_reproducibility.py
outputs/
  runs/<run_id>/
  tables/m4/
  figures/m4/
m4_report.md
```

不要因建议文件名与既有仓库脚本号重合而覆盖任何原有文件；保持既有 public API 向后兼容。若新增功能可以依赖现有训练接口，则优先复用。

## 7. 必须执行的 M4 测试

1. 所有既有 `114` 项测试继续通过。
2. 真实 batch 前向输出确认为 `[B,4,4]`；剂量 A 行、剂量 B 列均与真实矩阵对齐。
3. 手工简单数组的 RMSE、MAE、R²、Pearson 和 matrix-macro 计算值正确；R² 常数标签时有清楚的处理。
4. `X/X0>1` 能被模型输出与指标接受（输出不受 sigmoid 限制）；0 标签的损失有限。
5. B2/B3 的小训练循环 loss 有限、梯度非空且至少能对极小样本拟合。
6. B3 `edge_attr` 与原子类别特征处理方式正确，batch 重排不影响对应分子 embedding。
7. Drug A/B 同步交换输入应产生对应的等价标签位置；不对非对称基线强制“交换误差为 0”，但必须能正确计算其差异。固定顺序和 `SymAvg` 的逻辑单测。
8. train 只访问 train split；early stopping 仅访问 val；test 不参与任何 hyperparameter selection。
9. 验证与测试预测文件有完整 condition/matrix/dose 标识，逐点 16 行/矩阵，无丢行或重复。
10. M3 的 scaler 仅加载，不重新 fit；S3 严格没有监督测试/验证单药标签的训练访问。
11. 相同 seed、同环境运行两次获得一致或数值容差内相近的指标，并记录 GPU 非确定性。
12. 检查模型是否使用任何 train-only 数据派生的 test response 平均、matrix 聚合标签或测试统计量作为推理输入。

## 8. 主要成果及验收标准

需要输出至少以下三张主表：

**表 M4-1：S1(seed42) 基线完整对比**
`model | val_matrix_macro_RMSE | test_matrix_macro_RMSE | test_point_RMSE | test_MAE | test_Pearson | test_R2 | params | train_time`。

**表 M4-2：全划分基线结果**
`scenario | seed/fold | model | n_test_matrices | matrix_macro_RMSE | condition_macro_RMSE | point_RMSE | MAE | Pearson | R2`。

**表 M4-3：模型复杂度**
`model | n_parameters | total_train_seconds | seconds_per_epoch | peak_GPU_memory_MB | CPU_RAM_peak_MB | prediction_ms_per_matrix`。

**额外产物**：
- 至少一张 GT/Pred/Error 的 4×4 heatmap 和真实值-预测值散点图，所有图英语标注、仅使用最终测试预测；
- 测试集剂量大小和响应极端值分层误差分析（可选）；不要将图作为调参理由。
- `m4_report.md` 报告全部已完成 run、失败 run、代码版本、表格与运行命令。
- 自动测试全绿；GitHub main 推送源码、配置、测试、报告摘要。大型 checkpoints / predictions / processed tables 继续留本地，不提交普通 Git 历史。

**M4 PASS 最低条件**：
- B0/B1/B2/B3 在相同 S1 seed42 上均成功完成最终评估；
- B2/B3 训练曲线、early stopping、best checkpoint 和 test 预测都真实存在；
- S1/S2/S3 实验至少有完整且清楚标记的覆盖情况，全量 11 split 建议完成后再宣布“正式 M4 完成”；
- 不允许因 baseline 性能未超过别的 baseline 而伪造 PASS；PASS 意味训练评估**正确且可重现**，不是必须达到特定 RMSE；
- 指标口径固定且无泄漏测试通过；
- M0–M3 数据、特征、分组哈希保持不变；
- 本阶段没有启动 DoseSyn 完整架构 M5 或协同重建 M6/M7。

## 9. Agent 交付回复格式

1. M4-A/B/C 的实际完成情况（YES/NO，不能把“计划”写成已完成）
2. 本轮改动代码与 Git commit
3. 环境和硬件资源
4. 数据/分组/监督预算核对
5. B0–B3 实际指标表（附每种 scenario 与 fold）
6. 训练曲线、early stopping、checkpoint 和预测路径
7. 测试数/失败数、哈希核验
8. 尚未解决问题及是否建议进入 M5

**最后一句硬约束：当前只执行 M4。完成后停止，等待后续独立的 M5 模型任务书。**
