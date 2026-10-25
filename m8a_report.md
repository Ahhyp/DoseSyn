# DoseSyn M8-A：ConfSyn–DoseSyn 数据关联与独立性审计

**结论：CONDITIONAL PASS**

本阶段只做只读审计，不重训 DoseSyn B5，不修改 M0–M7-C 的冻结数据、checkpoint、split 或正式结果；不启动 M8-B/C。

## 1. 审计对象与可用性

| 项目 | 结果 |
| --- | --- |
| DoseSyn 基线 | `a6cfe9df89a21577e60952b2a690bec51fdf677a` |
| DoseSyn 冻结矩阵 / 条件 | 18,599 / 18,326 |
| ConfSyn 本地路径 | `/home/kouzheng_2/AH_YeP/confsyn/Consyn` |
| ConfSyn 训练数据 | DrugCombDB（63,010 行原始组合记录） |
| ConfSyn 可用 checkpoint | 仅 DrugCombDB 5-fold，无 OncologyScreen checkpoint |
| ConfSyn 逐样本预测 / split 索引 | **不存在**（只保存聚合指标与权重） |
| ConfSyn 标签语义 | `P(synergy >= 0)`，标签来自 DrugCombDB 的 synergy 定义 |

## 2. 实体映射

| 实体 | 总数 | EXACT | REVIEWED_ALIAS | UNMATCHED | 覆盖率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 药物 | 38 | 22 | 0 | 16 | 57.9% |
| 细胞系 | 39 | 29 | 2 | 8 | 79.5% |
| 条件 | 18326 | 4274 | 0 | 14052 | 23.3% |

- 药物：22/38 有 ConfSyn 结构；缺失 16 个：AZD1775, BEZ-235, Carboplatin, Cyclophosphamide, Dinaciclib, L778123, MK-2206, MK-4541, MK-4827, MK-5108, MK-8669, MK-8776, MRK-003, Mitomycine, Oxaliplatin, Temozolomide。
- 细胞系：31/39 名称可映射；缺失 8 个：A427, EFM192B, MSTO, OCUBM, OVCAR3, PA1, UWB1289, UWB1289BRCA1。
- `COLO320DM→COLO320`、`LNCAP→LNCAPCLONEFGC` 记录为 REVIEWED_ALIAS；其余为精确匹配。MSTO/OVCAR3 不在 ConfSyn 语料中（不得强行别名映射）。
- 仅按名称相同不构成实验独立；药物匹配使用 InChIKey，细胞使用显式名称/别名表。

## 3. 数据来源与标签重叠

- ConfSyn OncologyScreen 组合表：4,176 行、21 药、29 细胞；DrugCombDB 中可匹配到 **2940** 个 DoseSyn 条件。
- OncologyScreen 与 DrugCombDB 的同一条件（细胞+无序药对）重叠：**2842** 个；两套 synergy 标签完全相等的条件：**0** 个。
- OS vs DB label 的 Pearson = 0.42，Spearman = 0.36（n=2842）；仅作描述，不构成可互换标签。
- OS synergy vs DoseSyn 条件均值 X/X0：Pearson = -0.06，Spearman = 0.07（n=4176）。
- **结论：OS 的 synergy、DrugCombDB 的 synergy、DoseSyn 的 X/X0 是三种不同的派生量；不能把 X/X0 直接当作协同分数，也不能把 ConfSyn 概率直接映射为 X/X0。**

## 4. 训练/测试独立性：A/B/C/D

| 等级 | 定义 |
| --- | --- |
| A | ConfSyn-independent condition and DoseSyn held-out test matrix |
| B | ConfSyn-independent condition but DoseSyn train/val matrix |
| C | condition present in ConfSyn DrugCombDB corpus; split membership unknown |
| D | not bridgeable (entity/condition missing or matrix excluded) |

判定规则：

1. 条件在 ConfSyn DrugCombDB 中出现 → **C**（逐样本 train/test 归属未持久化，不能声称独立）。
2. 条件不在 ConfSyn 语料中、且 ConfSyn 覆盖其药物/细胞 → DoseSyn test 为 **A**，train/val 为 **B**。
3. 实体或条件不可映射、矩阵被 Primary-v1 排除 → **D**。
4. ConfSyn 没有 OncologyScreen checkpoint；即使 OS 与 O'Neil 同源，ConfSyn 的 DrugCombDB 模型并未在 OS 上训练（但 OS 行本身仍与 DoseSyn 同源，不能当作独立重复实验）。

### 4.1 各场景 held-out test 的 A/B/C 覆盖

| 场景 | test 矩阵（并集） | A 矩阵 | A 条件 | C 矩阵 |
| --- | ---: | ---: | ---: | ---: |
| s1_random_condition | 7131 | 742 | 742 | 1140 |
| s2_unseen_pair | 7194 | 750 | 750 | 1200 |
| s3_unseen_cell | 18599 | 2010 | 2010 | 2940 |

### 4.2 逐 split 证据

| 场景 | split | test 矩阵 | A | B | C | D |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| s1_random_condition | seed_13 | 2785 | 272 | 0 | 446 | 2067 |
| s1_random_condition | seed_2026 | 2793 | 284 | 0 | 402 | 2107 |
| s1_random_condition | seed_42 | 2786 | 297 | 0 | 457 | 2032 |
| s2_unseen_pair | seed_13 | 2840 | 390 | 0 | 390 | 2060 |
| s2_unseen_pair | seed_2026 | 2827 | 240 | 0 | 510 | 2077 |
| s2_unseen_pair | seed_42 | 2879 | 300 | 0 | 420 | 2159 |
| s3_unseen_cell | fold_0 | 3836 | 469 | 0 | 686 | 2681 |
| s3_unseen_cell | fold_1 | 3836 | 402 | 0 | 588 | 2846 |
| s3_unseen_cell | fold_2 | 3794 | 402 | 0 | 588 | 2804 |
| s3_unseen_cell | fold_3 | 3815 | 402 | 0 | 588 | 2825 |
| s3_unseen_cell | fold_4 | 3318 | 335 | 0 | 490 | 2493 |

A 级条件共 **2010** 个（按场景去重），全部为 ConfSyn 语料未覆盖且 DoseSyn test 的条件。

## 5. 联合推理的语义兼容性

- ConfSyn 输出 `p_synergy` 是 DrugCombDB 标签 `synergy >= 0` 的概率；它不是 O'Neil X/X0 的预测值，也不是 Bliss/HSA/Loewe/ZIP 分数。
- DoseSyn 输出 16 点 X/X0 响应面；X/X0 是相对 DMSO 的 viability 派生量，本身不等于协同。
- 两者的量纲、标签定义和训练数据均不同；**未发现可验证的单调映射或校准关系**。
- 因此 M8-A 不支持“联合排序”或“融合分数”的正式结论；M8-B 若做桥接，只能做 A 级条件上的**条件级一致性分析**，且必须明确标注为探索性。

## 6. 测试与可复现性

- 全量测试：`226 passed`（M7-C 基线）；本阶段新增 `tests/test_m8a_*.py` 后运行全量 pytest。
- 审计脚本：`scripts/26_audit_m8a.py`；映射/证据函数：`src/dosesyn/m8a/{mapping,evidence}.py`。
- 所有输入只读；输出在 `outputs/metrics/m8a/`、`docs/audit/m8a/`、`m8a_report.md`。
- 文件哈希、Git 状态、环境见 `outputs/metrics/m8a/m8a_manifest.json`。

## 7. M8-B 建议

1. 只使用 **A 级 test 矩阵** 作为桥接候选；C 级只能做描述性对照，不能作为独立测试。
2. 先验证 ConfSyn 在 22 个可映射药物 / 31 个可映射细胞上的前向推理；当前没有逐样本预测文件。
3. 将 ConfSyn 概率与 DoseSyn X/X0 分别报告，不构造单一“联合分数”；若必须构造，需先做外部校准并明确不确定性。
4. 由于 ConfSyn 逐样本 split 不可恢复，M8-B 的独立性声明只能建立在“条件不在 DrugCombDB 语料”之上，不能建立在“ConfSyn test fold”之上。
5. 报告 A 级覆盖率（相对 11 个 split 的 test 集），不要用 18,599 全量矩阵作分母。

## 8. 验收判定

| 检查 | 结果 |
| --- | --- |
| drug_mapping_at_least_20 | PASS |
| cell_mapping_at_least_25 | PASS |
| condition_matches_at_least_1000 | PASS |
| A_bridge_available_in_all_scenarios | PASS |
| label_semantics_documented | PASS |
| confsyn_per_sample_split_persisted | FAIL |
| confsyn_oncology_screen_checkpoint_available | FAIL |
| dosesyn_test_labels_untouched | PASS |

**CONDITIONAL PASS**：实体映射与 A 级桥接样本存在，但 ConfSyn 缺少逐样本 split、缺少 OncologyScreen checkpoint，且标签语义不可直接对齐；因此不能给出 FULL PASS，也不能在 M8-A 直接做联合排序。

