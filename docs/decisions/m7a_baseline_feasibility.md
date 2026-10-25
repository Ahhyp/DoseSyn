# M7-A 外部 Baseline 复现可行性审计

生成时间：2026-10-25
基线 commit：`14dd019897cfcf162e2ba5e12c1b1c4fbb17300d`
冻结数据：M3 Primary-v1，11 个 split（S1×3、S2×3、S3×5），18,599 个矩阵 / 18,326 个条件。

## 1. 结论

| 方法 | 官方代码 | 许可 | 与本项目协议兼容性 | 处理 |
| --- | --- | --- | --- | --- |
| comboKR 2.0 | 有，`aalto-ics-kepaco/comboKR2.0` | MIT | S1/S2 可公平评估；S3 因 Hill 参数含测试细胞单药信息而**不可公平** | 主排名（S1/S2） |
| DeepSynBa | 有，`hikuru/DeepSynBa` | CC BY-NC-SA | 药物嵌入（MoLFormer）与细胞基因集（NCI-60）无法直接用于 O'Neil + CCLE；目标为 SynBa 参数 | **adapted supplementary** |
| DD-PRiSM | 未找到公开可运行代码 | 未确认 | 无法审计 | 记为不可复现 |

本阶段不引用原论文 RMSE 作为同协议结果；所有数字由本项目在冻结划分上重新训练/预测得到。

## 2. comboKR 2.0

### 2.1 代码与环境

- 上游：https://github.com/aalto-ics-kepaco/comboKR2.0 ，MIT License。
- 运行环境：`/tmp/m7venv`（Python 3.11.15，numpy 1.26.4，scipy 1.13.1，scikit-learn 1.5.2，pandas 2.2.3，synergy 0.5.1，RLScore 0.8.2a0，comboKR2）。
- 官方 `requirements.txt` 要求 numpy≥1.26 / sklearn≥1.5，与项目主环境（numpy 1.21 / torch 1.13）冲突，因此外部 baseline 使用隔离环境。

### 2.2 数据与输入

- 官方 `data.zip`：22,527 个 O'Neil 曲面，39 细胞系，38 药物；**没有 MK-8776**。
- `cdd_full.npy`、`conc1.npy`、`conc2.npy`、`measurementsMedian.npy`（22,527×16）、`braids.pickle`（9 参数/曲面）、`Hills.csv`（1,482 条单药曲线）。
- 浓度单位为 **mol/L**；项目 M1 为 µM，适配层统一除以 `1e6`。
- 本项目 Primary-v1 中 **18,235 / 18,599** 个矩阵能在官方数据中匹配（98.0%）；未匹配的 364 个全部含 MK-8776（13 个细胞系 × 28 个组合）。
- 匹配后剂量网格一致性检查：18,235/18,235 完全一致（按 4×4 展开后比较，误差 <1e-12）。

### 2.3 方法

- 每个训练曲面拟合官方 BRAID 9 参数（`fit_braid_generic`，官方默认设置），得到 confidence features。
- 预测使用官方 `predict_with_candidates` + RBF Nystrom 差异核；不使用深度学习，不依赖 PyTorch。
- 训练标签只来自该 split 的训练矩阵；验证集用于选择超参数；测试集只在配置冻结后评估一次。
- 预测值不裁剪。

### 2.4 S3 协议阻塞

comboKR 官方流程使用 `Hills.csv`，其 Hill 参数由包含全部 O'Neil 细胞系（含 S3 测试细胞系）的单药曲线拟合。S3 要求测试细胞系的单药标签完全不可见，因此：

- comboKR **不参与 S3 主排名**；
- 若强行运行，会使用测试细胞系的真实单药信息，属于协议泄漏；
- 在报告中记为 `S3 不适用（协议冲突）`，不作为失败或劣于 DoseSyn 的证据。

### 2.5 预算

- 无 epoch 概念；S1 单 seed 训练 + 2,737 个验证曲面预测约 8–10 分钟（CPU）。
- 与 DoseSyn 的 GPU 预算不同，报告中记录 wall time 与硬件。

## 3. DeepSynBa（adapted）

### 3.1 代码与许可

- 上游：https://github.com/hikuru/DeepSynBa ，许可 CC BY-NC-SA（仅限非商业研究）。
- 官方输入：NCI-60 细胞系的 976 个 landmark 基因 + 104 个 NCI 药物的 MoLFormer 768 维嵌入。
- 官方输出：SynBa 参数（9 个）或由其重建的曲面；训练目标包含协同标签。

### 3.2 与 O'Neil 的不可对齐点

1. 药物集合不同：官方 104 个 NCI 药物，与本项目 36 个可用药物仅少量重合；MoLFormer 嵌入无法离线获得。
2. 细胞系不同：官方 NCI-60，本项目 CCLE 34 个可用细胞系。
3. 标签不同：官方为 SynBa/协同参数，本项目为原始 `X/X0`。
4. 因此**不能声称“复现 DeepSynBa”**。

### 3.3 采用的适配（明确记录）

- 药物特征：RDKit Morgan 指纹 2048 bit（count），线性投影到 768 维，替代 MoLFormer。
- 细胞特征：CCLE 表达中 950/976 个 landmark 基因可匹配，缺失 26 个补 0；保留 976 维。
- 架构：保留 DeepSynBa 的双分支 drug-cell 编码器（976+768→256）与 9 个 SynBa 参数头，由 9 参数重建 4×4 曲面；损失为曲面 Huber。
- 训练：60 epoch，batch 512，AdamW lr 1e-3，weight decay 1e-4，patience 10，早停看验证集 matrix-macro RMSE。
- 结果只能列为 **adapted supplementary baseline**，不进入主排名。

## 4. DD-PRiSM

- 参考论文 DOI: 10.1093/bib/bbae717。
- 快速检索未找到官方公开仓库；论文补充材料是否含完整训练代码未确认。
- 本阶段不实现、不报告数字；在报告中记为 `not reproducible (no public code)`。

## 5. 统一协议

- 数据：M3 冻结的 11 个 split，绝不重新划分。
- 标签：原始 `X/X0`，不裁剪、不取整。
- 主指标：matrix-macro RMSE（先算每个矩阵 16 点 RMSE，再对矩阵等权平均）。
- 补充：matrix-macro MAE、point RMSE、pooled Pearson、R²、condition-macro RMSE。
- 测试集只在配置冻结后读取一次；外部模型的超参数只依据验证集。
- S3：任何使用测试细胞系单药/组合标签的步骤都禁止；comboKR 因此不适用。
- 训练预算差异、特征适配、目标差异均在报告中单列。

## 6. 实施计划

1. 适配层：`src/dosesyn/baselines/external/{combokr_adapter,deepsynba_adapter}.py`。
2. 验证试运行：S1 seed42 与 S3 fold0（DeepSynBa）；comboKR S1 seed42。
3. 配置冻结：`configs/m7a/m7a.yaml`。
4. 正式运行：comboKR S1/S2 各 3 seeds；DeepSynBa 11 splits。
5. 汇总：M4 B0–B3、M6-B B1/B5、comboKR、DeepSynBa；配对差异与资源表。
6. 测试：形状、16 点完整性、测试隔离、映射一致性、指标复算、缺失处理、可追溯性。
