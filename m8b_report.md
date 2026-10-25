# M8-B：ConfSyn–DoseSyn 冻结模型联合筛选与剂量响应可视化

## 1. 结论

本阶段在 **s2_unseen_pair / seed_42** 的 **A 级候选（DrugCombDB 中不存在的组合条件）** 上完成双模型推理。
A 级候选共 **300** 个矩阵；其中 **180** 个包含 DrugCombDB 图中不存在的药物（主要是仅在 OncologyScreen 出现的 SN-38 等），无法由 DrugCombDB 五折 ConfSyn 评分；最终 **120** 个可评分组合条件进入联合表。
ConfSyn 五折协同概率均值 `0.0015`–`0.9995`（中位数 `0.6692`），DoseSyn B5 冻结 4×4 曲面均值响应 `0.2978`–`0.8834`。

> ⚠️ 语义边界：ConfSyn 输出的是 DrugCombDB 定义下 `synergy >= 0` 的正类概率，**不是** Bliss/HSA/Loewe/ZIP 协同分数；DoseSyn 输出的是原始 `X/X0` 剂量响应。本阶段只做“联合筛选展示”，不构建新的协同评分器，也不把 `X/X0` 解释为协同。
>
> 本阶段**不重新训练** ConfSyn 或 DoseSyn，不微调、不使用测试标签选择候选；只加载冻结 checkpoint 做推理。

## 2. 冻结 ConfSyn 资产恢复

| fold | checkpoint | n | AUC（标签重建） | ACC | p(正类均值) | p(负类均值) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `cv_DrugCombDB_mode_1_fold_num_0.pth` | 2000 | 0.9636 | 0.9030 | 0.8197 | 0.1326 |
| 1 | `cv_DrugCombDB_mode_1_fold_num_1.pth` | 2000 | 0.9519 | 0.8860 | 0.7870 | 0.1511 |
| 2 | `cv_DrugCombDB_mode_1_fold_num_2.pth` | 2000 | 0.9609 | 0.9145 | 0.8462 | 0.1142 |
| 3 | `cv_DrugCombDB_mode_1_fold_num_3.pth` | 2000 | 0.9672 | 0.9245 | 0.8559 | 0.0975 |
| 4 | `cv_DrugCombDB_mode_1_fold_num_4.pth` | 2000 | 0.9741 | 0.9255 | 0.8516 | 0.0980 |

复现要点：ConfSyn 使用 DrugCombDB 的 68 个细胞系、764 个药物、15,970 个蛋白节点；图特征为 `torch.manual_seed(42)` 生成的 300 维随机特征；HGT 为 2 层、4 头、768 维；药物侧使用仓库缓存的 ChemBERTa 256 维投影，细胞侧使用 16,383 维表达谱。五折 checkpoint 均以 `strict=False` 加载，除冻结的 RoBERTa 权重外无缺失/多余参数。

> 这些随机 DrugCombDB 行并非独立测试集（官方逐样本划分索引未持久化），AUC 只说明**预处理、实体索引与 checkpoint 恢复一致**，不能当作泛化性能。

## 3. 联合筛选表（按 ConfSyn 概率排序）

| # | 矩阵 | 细胞系 | 药物对 | ConfSyn p (mean±std) | DoseSyn 曲面均值 | 曲面最小 |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | `MTX_0789c8f4f5a3` | A2780 | Zolinza + Lapatinib | 0.9995 ± 0.0002 | 0.6411 | 0.0867 |
| 2 | `MTX_7216e36d184c` | NCIH2122 | Zolinza + Lapatinib | 0.9993 ± 0.0005 | 0.5684 | 0.1582 |
| 3 | `MTX_f8a05122d84e` | A2780 | Zolinza + Bortezomib | 0.9991 ± 0.0005 | 0.3845 | 0.0271 |
| 4 | `MTX_ade7cd844b8b` | A2780 | Etoposide + ABT-888 | 0.9952 ± 0.0023 | 0.4414 | 0.2580 |
| 5 | `MTX_5e3c11acf13b` | A2058 | Zolinza + Lapatinib | 0.9952 ± 0.0047 | 0.6891 | 0.1392 |
| 6 | `MTX_2ccfd01b9f5a` | SKMEL30 | Zolinza + Lapatinib | 0.9945 ± 0.0059 | 0.7450 | 0.3676 |
| 7 | `MTX_7ac4e638d5ff` | A2780 | ABT-888 + Sorafenib | 0.9905 ± 0.0111 | 0.3943 | 0.0534 |
| 8 | `MTX_6dc8ed1c3ab0` | SKMEL30 | Zolinza + Bortezomib | 0.9848 ± 0.0096 | 0.4145 | 0.0201 |
| 9 | `MTX_0f775619e2c3` | A2058 | ABT-888 + Sorafenib | 0.9835 ± 0.0190 | 0.6815 | 0.1386 |
| 10 | `MTX_4742a750c822` | RPMI7951 | Zolinza + Lapatinib | 0.9824 ± 0.0110 | 0.6037 | 0.0555 |
| 11 | `MTX_b1e1b6795f4b` | DLD1 | ABT-888 + Sorafenib | 0.9822 ± 0.0127 | 0.6227 | 0.2218 |
| 12 | `MTX_7ea63564027e` | SKMES1 | ABT-888 + Sorafenib | 0.9811 ± 0.0170 | 0.6869 | 0.1395 |
| 13 | `MTX_8fb327944bd4` | ES2 | Zolinza + Bortezomib | 0.9807 ± 0.0136 | 0.5297 | 0.0555 |
| 14 | `MTX_cc6d978a94e9` | A375 | ABT-888 + Sorafenib | 0.9790 ± 0.0230 | 0.5866 | 0.1172 |
| 15 | `MTX_f2fabd03aab7` | DLD1 | Zolinza + Bortezomib | 0.9785 ± 0.0121 | 0.4770 | 0.0776 |

ConfSyn p 与 DoseSyn 曲面均值响应的 Spearman 相关为 `0.033`，与曲面最小值的 Spearman 相关为 `-0.118`。相关接近 0，说明两个模型对“组合优先级”的看法并不一致，这本身是需要如实报告的结果，而不是可以合并成一个分数的证据。

### 3.1 二维象限与语义一致性

以 ConfSyn `p_mean > 0.5` 作为“高协同概率”、以 DoseSyn `曲面最小 X/X0 < 0.2` 作为“高抑制”划分四个象限：

| 象限 | 含义 | 数量 |
| --- | --- | ---: |
| p>0.5 且 min X/X0<0.2 | 两模型同向：既可能协同、又有强抑制 | 56 |
| p>0.5 且 min X/X0≥0.2 | ConfSyn 高协同，但 DoseSyn 未预测强抑制 | 23 |
| p≤0.5 且 min X/X0<0.2 | ConfSyn 低协同，但 DoseSyn 预测强抑制 | 25 |
| p≤0.5 且 min X/X0≥0.2 | 两模型同向：低协同、弱抑制 | 16 |

`p_mean > 0.9` 且 `min X/X0 < 0.2` 的高置信交集有 **27** 个；`p_mean > 0.9` 但 `min X/X0 ≥ 0.5` 的强分歧有 **1** 个。最典型的分歧是 **Zolinza + Lapatinib**：ConfSyn 在 A2780/NCIH2122/A2058/SKMEL30 上给出 0.994–0.999，但 B5 曲面均值 X/X0 为 0.57–0.75，提示该组合在 O'Neil 响应尺度上并未产生强抑制。这类候选可以被标记为“高协同概率—弱响应”，但不能据此断言“假阳性”，因为 ConfSyn 的输出语义本来就不是 X/X0。

### 3.2 语义保持

两个模型在联合表中保持**分离的两列**：`p_mean` 只表示 DrugCombDB 定义下 `synergy >= 0` 的概率，`y_pred_*` 只表示 O'Neil `X/X0` 响应。本阶段没有把它们相乘、加权或校准成单一“协同分数”，也没有把 `X/X0` 当作协同标签。`joint_ranking_top30.csv` 只按 ConfSyn 概率排序，DoseSyn 响应作为独立对照列展示。

## 4. 图表

- `outputs/figures/m8b/fig8_1_dual_dimension_scatter.{svg,pdf,png}`：ConfSyn 概率 × DoseSyn 曲面均值，颜色为曲面最小值；四象限参考线。
- `outputs/figures/m8b/fig8_2_top_candidates.{svg,pdf,png}`：Top-15 候选的 ConfSyn 五折均值与范围。
- `outputs/figures/m8b/fig8_3_top_surface_heatmaps.{svg,pdf,png}`：Top-6 候选的 DoseSyn B5 冻结 4×4 `X/X0` 曲面。

## 5. 交付物

- `outputs/metrics/m8b/confsyn_checkpoint_validation.csv`：五折标签重建检查
- `outputs/metrics/m8b/confsyn_scores_s2_seed42.csv`：A 级候选五折概率
- `outputs/metrics/m8b/bridge_candidates.csv`：联合筛选主表（含曲面统计与排名）
- `outputs/metrics/m8b/joint_ranking_top30.csv`：Top-30 排序
- `outputs/metrics/m8b/dosesyn_surfaces.json`：Top 候选 4×4 冻结曲面
- `outputs/figures/m8b/`：双维度散点图、Top 候选剂量响应热力图
- `outputs/audit/m8b/`：紧凑审计副本、环境与来源哈希

## 6. 局限

1. ConfSyn 的五折逐样本划分未持久化，A 级判定只能基于“该条件不在 DrugCombDB 表中”，无法进一步证明它不在某个训练折。
2. 本阶段只展示 S2 seed42 的 300 个 A 级候选（其中 120 个可被 DrugCombDB 五折评分），未做 S1/S3 的联合筛选；跨场景结论需在 M8-C 另行验证。
3. 有 180 个 A 级候选因药物不在 DrugCombDB 图（或仅有 OncologyScreen 覆盖）而无法评分，已单列 `confsyn_unscoreable_candidates.csv`，不视为阴性。
4. ConfSyn 概率是组合级二分类概率，不含剂量；DoseSyn 曲面是剂量级响应，不含协同标签，二者不是同一目标。
5. 未重新训练、未微调、未使用测试标签选择候选。
6. ConfSyn 的 68 个 DrugCombDB 细胞系与 DoseSyn 的 34 个细胞系只部分重叠；A 级定义只说明该组合条件不在 DrugCombDB 表中，不代表细胞系层面的完全独立。

_生成时间：2026-10-25T12:49:56+00:00_
