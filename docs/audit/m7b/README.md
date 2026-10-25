# M7-B 审计文件索引

本目录保存 M7-B 的可追溯性材料，均不包含原始响应数据或大体积 checkpoint。

| 文件 | 内容 |
| --- | --- |
| `masks.json` / `mask_definition.json` | 四个剂量位置掩码的完整定义（版本、观测位置、留出位置、说明），由 `src/dosesyn/train/m7b_masks.py` 生成。 |
| `b1_center_summary.csv` | M7-B1 中心 2×2 留出实验按模型、场景聚合的 held-out matrix-macro RMSE / MAE / Pearson / R²。 |
| `m7b_runs.csv` | 全部 M7-B 正式运行逐项记录（模型、掩码、split、指标、预算、参数量）。 |
| `paired_deltas.csv` | 中心留出逐 split 相对 B5 的配对 RMSE 差值。 |
| `sparsity_summary.csv` | 50% / 25% 观测比例实验（S1 seed42、S2 seed42、S3 fold0）。 |
| `boundary_summary.csv` | 高剂量边界外推实验（B5 / B1 / B3）。 |
| `m7b_manifest.json` | 实验版本、掩码版本、Git commit、环境、文件大小。 |

掩码版本固定为 `m7b-masks-v1`；训练/测试脚本只接受该版本，避免事后修改掩码。
所有指标使用统一 `compute_metrics`，且先过滤到 held-out 位置再对矩阵做宏平均。
