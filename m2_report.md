# DoseSyn M2 报告：多模态特征构建与实体映射

生成时间：2026-10-24T05:56:47+00:00　配置：`/home/kouzheng_2/AH_YeP/DoseSyn/configs/features_m2.yaml`

> 本阶段只构建特征，**未训练任何 DoseSyn 神经网络，也未创建 train/val/test 划分**。

## 1. 结论

- 药物：38 个，SMILES 38 个，RDKit 构图 38 个（36 OK / 2 NEEDS_REVIEW / 0 FAILED）。
- 细胞系：39 个，DepMap 映射 37 个，表达谱 35 个，通路特征 34 个。
- Hallmark：50 个基因集，有效 50 个；基因覆盖率 96.5%–100.0%
- 组合矩阵：23052 个，特征齐全 18599 个（80.7%）；独立条件 22737 个，齐全 18326 个。
- 单药曲线：2094 条，特征齐全 1782 条（85.1%）。

## 2. 药物映射与分子图

| 项目 | 数量 |
| --- | ---: |
| O'Neil 药物全集 | 38 |
| SMILES 覆盖 | 38 |
| RDKit 构图 | 38 |
| 图可直接用于 GINE | 36 |
| 图需人工复核 | 2 |
| 构图失败 | 0 |

需复核的结构：`Carboplatin`、`Oxaliplatin`

Carboplatin / Oxaliplatin 的 PubChem SMILES 中铂配位键不被 RDKit 表示为共价键，因此图由多个连通分量组成。图仍被保存，但 `graph_status=NEEDS_REVIEW`，其参与的矩阵不计入“特征齐全”，避免把不完整图静默送入模型。

ABT-888 的 PubChem 同名条目是二盐酸盐；映射使用游离碱 Veliparib（CID 11960529），并用 RDKit InChIKey 与 DrugBank/ConfSyn 结构交叉验证。所有 38 个药物的 InChIKey 均由 RDKit 重算并与 PubChem 比对。

## 3. 细胞系映射

| 项目 | 数量 |
| --- | ---: |
| O'Neil 细胞系 | 39 |
| DepMap 映射 | 37 |
| VERIFIED/VERIFIED_ALIAS | 36 |
| 有 CCLE 表达 | 35 |
| 有通路特征 | 34 |
| 细胞特征齐全 | 34 |

未映射：`EFM192B`、`UWB1289BRCA1`

需复核：`COLO320DM`

已映射但 CCLE 表达缺失：`OCUBM`、`UWB1289`

## 4. 通路特征

MSigDB Hallmark 50 个集合；按精确大写 gene symbol 匹配，要求每集合 ≥10 个基因且覆盖率 ≥50%。有效通路 50 个，基因数 32–200，覆盖率最低 96.5%。

打分使用逐细胞 rank-based weighted enrichment（ssGSEA 风格，alpha=0.25），只依赖单个细胞的基因排序，不使用队列均值/方差，因此不依赖 M3 的训练集划分，可安全缓存。

## 5. 特征覆盖率

| 指标 | 数值 |
| --- | ---: |
| 特征齐全组合矩阵 | 18599 / 23052 |
| 特征齐全独立条件 | 18326 / 22737 |
| 特征齐全单药曲线 | 1782 / 2094 |

缺失原因（矩阵计数；同一矩阵可能命中多个原因，故合计大于未覆盖矩阵数）：

| 原因 | 矩阵数 |
| --- | ---: |
| `drug_graph_not_model_ready` | 1716 |
| `cell_unmapped` | 1187 |
| `cell_expression_missing` | 1166 |
| `cell_pathway_missing` | 604 |

按癌症类型的细胞与矩阵覆盖：

| 癌症类型 | 细胞数 | 特征齐全细胞 | 矩阵数 | 特征齐全矩阵 | 矩阵保留率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Colon/Colorectal Cancer | 8 | 7 | 4769 | 3857 | 80.9% |
| Lung Cancer | 8 | 8 | 4748 | 4396 | 92.6% |
| Ovarian Cancer | 8 | 7 | 4727 | 3836 | 81.2% |
| Skin Cancer | 6 | 6 | 3540 | 3276 | 92.5% |
| Breast Cancer | 5 | 4 | 2915 | 2156 | 74.0% |
| UNKNOWN | 2 | 0 | 1187 | 0 | 0.0% |
| Prostate Cancer | 2 | 2 | 1166 | 1078 | 92.5% |

## 6. 无法映射的实体与受影响矩阵

| 实体 | 类型 | 原因 |
| --- | --- | --- |
| `EFM192B` | 细胞系 | DepMap 无对应模型（UFM/衍生系），不填充 |
| `UWB1289BRCA1` | 细胞系 | DepMap 无对应模型（UFM/衍生系），不填充 |
| `COLO320DM` | 细胞系 | 变体状态无法确认，保守排除 |
| `Carboplatin` | 药物 | 金属配位图不完整，标记复核 |
| `Oxaliplatin` | 药物 | 金属配位图不完整，标记复核 |

受影响矩阵数量见上表缺失原因；未映射细胞的数据点仍保留在 M1，不会因特征缺失而删除。

## 7. M2 验收

| 验收项 | 结果 |
| --- | --- |
| 所有无法映射实体有明确原因 | PASS |
| 无静默丢弃（M1 表未被修改） | PASS |
| 药物图可正常读取 | PASS |
| 表达矩阵与通路特征维度一致 | PASS |
| 每个矩阵可追溯药物/细胞特征 | PASS |
| 原始 M1 文件哈希保持不变 | PASS |

自动测试：`81 passed in 11.80s`

## 8. 复现命令

```bash
cd /home/kouzheng_2/AH_YeP/DoseSyn
.venv/bin/python scripts/02_build_features.py
.venv/bin/python -m pytest -q tests
```

## 9. 尚未解决的问题

- 2 个金属配合物的分子图不完整，需在 M3 前决定是否手工补铂配位键或排除。
- EFM192B / UWB1289BRCA1 无 DepMap 模型；COLO320DM 仅有 COLO320，均保守排除。
- OCUBM / UWB1289 有 DepMap ID 但该版 CCLE 表达谱无数据。
- 未做基因级标准化，避免全数据拟合；M3 需在训练集上 fit。
- 未创建训练/验证/测试划分（按规划留给 M3）。

