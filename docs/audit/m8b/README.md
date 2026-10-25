# M8-B 审计副本索引

本目录只保存 M8-B 的**小体积**审计材料；完整产物在 `outputs/metrics/m8b/`、
`outputs/figures/m8b/` 与 `outputs/runs/m6b/`（后者为冻结 B5，未修改）。

| 文件 | 说明 |
| --- | --- |
| `checkpoint_sha256.csv` | ConfSyn 五折 checkpoint 的路径、大小与 SHA256（只记录，不复制大文件） |
| `confsyn_checkpoint_validation.csv` | 每折 2,000 条 DrugCombDB 记录的标签重建 AUC/ACC；**不是泛化指标** |
| `confsyn_unscoreable_candidates.csv` | A 级候选中药物/细胞不在 DrugCombDB 图内的 180 个条件及原因 |
| `confsyn_scores_s2_seed42.csv` | 120 个可评分条件的五折协同概率 |
| `bridge_candidates.csv` | 双模型联合主表（ConfSyn 概率 + DoseSyn B5 曲面统计/排名） |
| `joint_ranking_top30.csv` | 按 ConfSyn 概率排序的 Top-30 |
| `m8b_summary.json` | 推理摘要、候选计数、诊断 |
| `m8b_audit_manifest.json` | 所有指标/图文件与 checkpoint 的 SHA256 |

## 语义提醒

- `p(synergy >= 0)` 是 DrugCombDB 二分类标签下的概率，不是 Bliss/HSA/Loewe/ZIP 分数。
- `y_pred_*` 是 O'Neil `X/X0` 剂量响应，不是协同分数。
- 两列在联合表中保持分离，没有合成单一评分。
- ConfSyn 逐样本划分未持久化，A 级只表示“条件不在 DrugCombDB 表中”，不能证明不在某个训练折。
