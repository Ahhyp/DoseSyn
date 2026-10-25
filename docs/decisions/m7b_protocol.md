# M7-B 剂量位置插值与稀疏曲面泛化：实验协议

## 1. 问题与定位

M6-B 的正式指标是在完整 4×4 网格上计算的，不能单独证明模型对训练时未观测的剂量位置具有插值能力。M7-B 在**不改变 M3 划分、不改变主模型结构**的前提下，系统性地隐藏训练矩阵中的剂量位置，只在这些位置评价模型。

## 2. 掩码定义

平面索引 `k = row_idx * 4 + col_idx`，row 为药物 A 剂量秩，col 为药物 B 剂量秩，均从 0 开始。

| 掩码 | 观测 | 留出 | 用途 |
| --- | ---: | ---: | --- |
| `center_holdout_75` | 12 | 4: (1,1),(1,2),(2,1),(2,2) | 主实验：中心插值 |
| `checkerboard_holdout_50` | 8 | 8 | 稀疏性 |
| `corners_holdout_25` | 4 | 12 | 稀疏性压力测试 |
| `boundary_holdout_50` | 9 | 7: 最高 A 行 ∪ 最高 B 列 | 高剂量边界外推 |

所有掩码在转置下不变，因此交换药物顺序不会改变观测/留出集合。掩码由 `src/dosesyn/train/m7b_masks.py` 固定生成，版本哈希写入每个 run 的 metadata。

## 3. 训练与选择规则

- 训练损失只对 observed 位置计算 Huber（逐矩阵平均后取 batch 平均）；held-out 标签不参与损失、不参与预处理统计、不参与曲面拟合。
- 验证指标只在 held-out 位置计算；早停和 checkpoint 选择只用该指标。
- 测试只在验证集选好 checkpoint 后读取一次；测试结果不回流到任何配置。
- 统一预算：160 epoch、patience 30、cosine、AdamW、lr 1e-3、batch 64、Huber δ=1。所有模型共享相同掩码与预算。
- B2 MLP 仅作为 P1 在 3 个 dev split（S1 seed42、S2 seed42、S3 fold0）上运行。

## 4. 评价

主指标：held-out matrix-macro RMSE（每个矩阵先算 held-out RMSE，再对矩阵等权平均）。

补充：held-out matrix-macro MAE、point RMSE、Pearson（仅当方差非零）、R²、mean bias、负预测比例、按剂量区域误差。指标先过滤到 held-out 位置再计算，不与原始 16 点 RMSE 混合排名。

## 5. 场景与模型

- S1 seed13/42/2026、S2 seed13/42/2026、S3 fold0–4（M3 冻结）。
- M4 B3、M6-B B1(T3)、M6-B B5(DoseSyn) 全 11 split；M4 B2 仅 dev。
- S3 严格细胞系隔离，测试细胞系的单药/组合标签不参与训练。

## 6. 复现

```bash
.venv/bin/python scripts/18_run_m7b.py --group b1_center_all --force
.venv/bin/python scripts/18_run_m7b.py --group b2_dev_mlp --force
.venv/bin/python scripts/18_run_m7b.py --group sparse_dev --force
.venv/bin/python scripts/18_run_m7b.py --group boundary_dev --force
.venv/bin/python scripts/19_aggregate_m7b.py
```
