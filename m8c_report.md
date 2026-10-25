# M8-C：冻结桥接核验、代表性曲面分析与第四章整合

## 1. 结论

本阶段对 M8-B 的 300 个 A 级候选（S2 seed42 测试矩阵）做了统一口径复核：**120** 个可由 ConfSyn 五折评分，**180** 个因药物不在 DrugCombDB 图中而明确不可评分。ConfSyn 的 BERT stub 经三项探针确认**不参与前向计算**，BERT 权重缺失不影响预测；真正的药物语义来自训练时缓存并冻结的 ChemBERTa 256 维投影。

原先“强分歧”的 27 个候选统一改称为**高协同倾向且存在低 X/X0 剂量点的双信号候选**：它们同时满足 ConfSyn 五折平均概率 ≥ 0.9 与 DoseSyn B5 曲面最小 X/X0 < 0.2。这一命名只描述两条信号同时出现，不表示 ConfSyn 与 DoseSyn 在预测同一目标，也不表示二者互相验证。

**判定：PASS（技术流程与语义边界核验通过；桥接本身的独立验证强度仍受 M8-A 的 A 级定义限制）。**

## 2. BERT stub 与冻结推理可靠性

| 探针 | 结果 | 含义 |
| --- | --- | --- |
| `drug_semantic_enc.forward` 调用次数 | 0 | 提供 `drug1_emb/drug2_emb` 时 BERT 分支被完全绕过 |
| 随机重初始化 `proj` 后预测变化 | 0.000e+00 | 预测逐位不变，`proj` 参数不参与推理 |
| stub 前向替换后预测变化 | 0.000e+00 | 替换实现不改变输出 |
| 缓存 ChemBERTa 投影置零 | 平均 |Δp|=0.3306，Spearman=0.495 | 真正承载药物语义的是缓存投影，不是缺失的 BERT 本体 |
| checkpoint 中 BERT 参数 | 104 个张量 / 44,104,450 参数 | 占 checkpoint 的 49.9%，全部为冻结项 |
| fold 0 标签重建 AUC | 0.9636 | 仅证明恢复一致，**不是泛化性能** |

五折标签重建 AUC 分别为 fold0=0.9636、fold1=0.9519、fold2=0.9609、fold3=0.9672、fold4=0.9741。这些数值来自随机抽取的 DrugCombDB 行，不能作为独立测试集指标。

> 需要如实说明：我们无法用真实 RoBERTa 权重重跑一次对照，因为 checkpoint 本身不包含 BERT 本体；但 ConfSyn 在提供了预计算药物 embedding 时根本不会调用该分支，因此“缺失 BERT 是否影响预测”在本推理路径上可判定为不影响。

## 3. 候选宇宙：300 → 120 + 180

| 状态 | 数量 | 说明 |
| --- | ---: | --- |
| A 级候选矩阵 | 300 | S2 seed42 test，`evidence_level=A` |
| ConfSyn 可评分 | 120 | 药物与细胞均在 DrugCombDB 图中，五折均有输出 |
| 不可评分 | 180 | 全部因 `drug_or_cell_absent_from_DrugCombDB_graph` |
| 双信号候选 | 27 | p≥0.9 且 min X/X0<0.2 |
| 高概率且高最低响应 | 2 | p≥0.95 且 min X/X0≥0.35 |

180 个不可评分候选都由 **SN-38** 参与：SN-38 只出现在 OncologyScreen 数据中，不在当前 DrugCombDB 的 764 药物图内，而现有 ConfSyn checkpoint 是 DrugCombDB 模型。这些候选没有伪造概率，也未从 300 个 A 级矩阵中删除。

## 4. 代表性案例（规则先于结果固定）

案例选择规则写入 `configs/m8c/m8c.yaml`，在查看曲面之前确定：最高 ConfSyn 概率、中位概率、高概率且高最低响应、双信号最大间距各一个。

| 案例 | 细胞系 | 药物对 | p (mean±std) | B5 范围 | B5 min | 原始 X/X0 min | RMSE | 源行 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| top_p | A2780 | Zolinza+Lapatinib | 0.9995±0.0002 | 0.087–0.991 | 0.087 | 0.173 | 0.0525 | 18306–18321 |
| median_p | A2058 | Etoposide+ABT-888 | 0.6711±0.3097 | 0.147–0.948 | 0.147 | 0.206 | 0.0464 | 2162–2177 |
| high_p_high_min | NCIH2122 | Etoposide+ABT-888 | 0.9655±0.0380 | 0.388–0.881 | 0.388 | 0.376 | 0.0344 | 158658–158673 |
| dual_signal | A2780 | Zolinza+Bortezomib | 0.9991±0.0005 | 0.027–0.969 | 0.027 | 0.024 | 0.0333 | 18242–18257 |

四个案例的 16 个预测点均与 O'Neil 原始 `source_row` 对齐：`y_true` 与原始 `X/X0` 逐点一致，`dose_a_um/dose_b_um` 与 M1 浓度逐点一致，源行连续。案例 1（top_p）是 ConfSyn 概率最高、但 B5 曲面同时存在低响应角的代表；案例 2（median_p）代表 120 个候选的典型水平；案例 3（high_p_high_min）是“高协同概率但整体抑制较弱”的对照；案例 4（dual_signal）是两条信号间距最大的候选。

## 5. 与公开文献的关系（只做背景核对，不作为标签）

- Vorinostat（Zolinza）+ Lapatinib 有 I 期实体瘤/乳腺癌研究（[NCT01118975](https://classic.clinicaltrials.gov/show/NCT01118975)），说明该组合并非凭空构造；但文献中的瘤种、剂量与终点均与本项目 O'Neil 4×4 网格不同，**不能把文献结论当作本项目样本的协同标签**。
- Vorinostat + Bortezomib 有 I 期实体瘤研究（[ASCO 2007 摘要 3573](https://ascopubs.org/doi/abs/10.1200/jco.2007.25.18_suppl.3573)），HDAC 与蛋白酶体抑制的联合在机制上被讨论；本项目中的低 X/X0 只表示 O'Neil 响应尺度上的低存活，不等于文献定义的协同。
- Etoposide + Veliparib（ABT-888）存在以依托泊苷/铂类为骨架的临床试验（[NCT01642251](https://clinicaltrials.gov/study/NCT01642251)）；本项目的 ABT-888 是 O'Neil 原始名称，映射到 veliparib 结构。

因此，文献核对只支持“组合具有生物学/临床背景”，不支持“ConfSyn 概率被独立实验验证”或“DoseSyn 曲面被文献验证”。本阶段没有新的湿实验。

## 6. 图表与论文整合

- 新增 4.3.9 节，流程图（图 4-14）、双信号景观（图 4-15）、代表案例曲面（图 4-16）与候选漏斗（图 4-17）。
- 表 4-9 为 BERT stub 探针；表 4-10 为 300/120/180 与双信号计数；表 4-11 为四个案例的源行与指标。
- 4.4 小结新增第 6 条，明确 ConfSyn 与 DoseSyn 是两条互补但不可互相替代的预测通道。

## 7. 数值一致性检查

- 候选宇宙 300 = 120 + 180：300 = 120 + 180。
- 双信号候选 27 个与 `m8c_dual_signal_candidates.csv` 行数一致。
- 四个案例 `y_true` 与 M1 原始 `X/X0` 逐点一致（atol=1e-6；实际最大差 2.9×10⁻⁸，来自预测文件 float32），剂量逐点一致（atol=1e-6；实际最大差 4.3×10⁻⁸）。
- 300/120/180 与 27/2 的计数均由同一 `m8c_summary.json` 驱动，报告数字不手工改写。
- 案例源行连续且每例 16 行；每行保留 `viability_1..4`，未做插补或裁剪。
- 冻结输入哈希在 `m8c_frozen_hashes.json` 中，脚本不写回 M0–M8-B 产物。

## 8. 仍需解决的问题

1. ConfSyn 的逐样本折划分未持久化，A 级仍只能保证“条件不在 DrugCombDB 表中”，不能逐样本证明未参与训练。
2. 180 个 SN-38 候选无法由当前 DrugCombDB checkpoint 评分；若要覆盖，需要重新训练含 SN-38 的药物图模型，超出本阶段约束。
3. BERT 本体不在 checkpoint 中；虽然推理路径绕过该分支，但无法对“若使用真实 BERT 会怎样”做数值对照。
4. 双信号只说明两个模型给出不同侧面的信号，不构成协同验证；M8-C 不产生新的协同分数。
5. 代表性案例是规则抽取，不是随机抽样；不能据此估计 120 个候选的整体错误率。

## 9. 交付物

| 文件 | 内容 |
| --- | --- |
| `m8c_report.md` | 本报告 |
| `m8_final_report.md` | M8 整体总结 |
| `outputs/metrics/m8c/m8c_candidate_universe.csv` | 300 个 A 级候选及状态 |
| `outputs/metrics/m8c/m8c_scored_candidates.csv` | 120 个可评分候选 |
| `outputs/metrics/m8c/m8c_dual_signal_candidates.csv` | 双信号候选 |
| `outputs/metrics/m8c/m8c_representative_cases.csv` | 4 个案例与选择规则 |
| `outputs/metrics/m8c/m8c_case_source_rows.csv` | 64 行原始源行追溯 |
| `outputs/metrics/m8c/m8c_stub_audit.json` | BERT stub 三项探针 |
| `outputs/metrics/m8c/m8c_frozen_hashes.json` | 冻结输入哈希 |
| `outputs/figures/m8c/` | 流程图、景观图、案例曲面、漏斗图 |

_生成时间：2026-10-25T14:45:30+00:00_
