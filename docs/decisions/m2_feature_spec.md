# DoseSyn M2 特征构建说明（v1）

本文档说明 M2 阶段（药物结构、分子图、细胞系、表达谱、Hallmark 通路、特征连接）的
数据来源、编码规则与可复现命令。M2 **不训练模型、不创建 train/val/test 划分**。

## 1. 输入

| 输入 | 路径 | 用途 |
| --- | --- | --- |
| M1 双药点表 | `data/processed/oneil_combo_points.parquet` | 药物/细胞实体全集与连接 |
| M1 双药矩阵元数据 | `data/processed/oneil_combo_matrix_metadata.parquet` | 23,052 个矩阵索引 |
| M1 单药点表 / 曲线元数据 | `data/processed/oneil_monotherapy.parquet`、`oneil_mono_curve_metadata.parquet` | 2,094 条曲线索引 |
| ConfSyn 药物结构 | `Consyn/Data/OncologyScreen/drugsmiles.csv` | 21 个 DrugBank SMILES |
| ConfSyn 药物名映射 | `Consyn/Data/OncologyScreen/drug_combinations.csv` | DrugBank ID ↔ O'Neil 药物名 |
| CCLE 表达 | `Consyn/Data/CCLE_expression.csv` | 1,270 细胞系 × 19,144 基因（log2(TPM+1)） |
| DepMap 样本信息 | `Consyn/Data/sample_info.csv` | DepMap ID、谱系、别名 |
| MSigDB Hallmark | `Consyn/Data/h.all.v2026.1.Hs.symbols.gmt` | 50 个通路基因集 |

## 2. 药物映射规则

最终映射表 `data/processed/drug_mapping.csv`（38 行）：

| 列 | 说明 |
| --- | --- |
| `oneil_drug_name` | O'Neil 原始名称（A、B、单药三列并集） |
| `canonical_drug_name` | 标准化名称 |
| `drug_id` | 沿用 M1 的稳定 ID（`DRUG_*`） |
| `pubchem_cid` | PubChem CID |
| `canonical_smiles` | PubChem ConnectivitySMILES（去立体） |
| `isomeric_smiles` | PubChem SMILES（含立体），用于构图 |
| `inchikey` | RDKit 由 isomeric SMILES 重算 |
| `mapping_source` | 来源（ConfSyn/PubChem/别名） |
| `mapping_status` | `VERIFIED` / `VERIFIED_ALIAS` / `NEEDS_REVIEW` / `UNMAPPED` |

规则：

1. 只接受 **RDKit InChIKey 完全一致** 的结构；名称相同但结构不同即拒绝。
2. `ABT-888` 的 PubChem 同名条目是二盐酸盐，改用游离碱 `Veliparib`（CID 11960529，
   InChIKey `JNAHVYVRKWKWKQ-CYBMUJFWSA-N`），状态 `VERIFIED_ALIAS`。
3. `Carboplatin` / `Oxaliplatin` 的铂配位键无法被 SMILES 完整表达，RDKit 得到
   多个连通分量；结构保留但状态 `NEEDS_REVIEW`，其参与的矩阵不计入“特征齐全”。
4. 38 个药物全部有 SMILES、CID、InChIKey；无 `UNMAPPED`。

## 3. 分子图编码

缓存：`data/processed/drug_graph_features.pt`（`torch.save({"graphs", "metadata"})`），
元数据：`data/processed/drug_graph_metadata.json`。

- 原子特征 9 维（离散整数，供 embedding）：原子序数、形式电荷（截断 ±3）、芳香性、
  杂化、总氢数（截断 4）、手性、是否成环、环大小（截断 8）、度（截断 6）。
- 化学键特征 4 维：键类型、共轭、成环、键立体。
- `edge_index` 为无向双向边，`edge_attr` 与边一一对应；无自环。
- 两种药物、单药与双药共用同一 `featurize` 实现。
- 校验：SMILES 可解析、非空、至少一根键、特征范围、边对称性、无自环。

## 4. 细胞系映射

`data/processed/cell_mapping.csv`（39 行）：

- 精确匹配顺序：DepMap `stripped_cell_line_name` → `CCLE_Name` 前缀 → `alias`；
  人工别名优先于自动匹配。
- 状态：`VERIFIED`（33）、`VERIFIED_ALIAS`（3：LNCAP→LNCAPCLONEFGC、
  MSTO→MSTO211H、OVCAR3→NIHOVCAR3）、`NEEDS_REVIEW`（1：COLO320DM→COLO320）、
  `UNMAPPED`（2：EFM192B、UWB1289BRCA1）。
- 已映射但该版 CCLE 表达谱无数据：`OCUBM`（ACH-002179）、`UWB1289`（ACH-001418）。
- 最终特征可用细胞 34 个。

## 5. 表达与 Hallmark 通路

- 表达矩阵 `cell_expression.parquet`：34 行 × 19,144 基因，保留原始 log2(TPM+1)，
  **不做任何数据集级标准化或填补**。
- 基因匹配规则：DepMap 列名去掉 ` (entrez)` 后的大写符号，与 GMT 符号精确匹配。
- 打分：逐细胞 rank-based weighted enrichment（ssGSEA 风格），
  `hit weight=(N-rank+1)^0.25`，取 signed max |cumsum(hit−miss)|。
  只用单个细胞内部的排序，不依赖队列均值/方差，可缓存。
- 每个通路保留 `n_genes`、`n_present`、`coverage`、`valid`；
  50 个通路全部有效，覆盖率 96.5%–100%。

## 6. 特征连接

- `oneil_combo_feature_index.parquet`：23,052 行，与 M1 矩阵一一对应。
  每行包含 A/B 药物图状态、细胞映射/表达/通路状态、`depmap_id`、
  `feature_complete` 与 `missing_reasons`。
- `oneil_condition_feature_index.parquet`：22,737 个独立条件，
  汇总矩阵数、批次数与特征齐全矩阵数。
- `oneil_mono_feature_index.parquet`：2,094 条单药曲线。
- **不修改、不删除任何 M1 行**；特征缺失只记录 `missing_reasons`。

覆盖率（v1）：

| 指标 | 数量 |
| --- | ---: |
| 特征齐全组合矩阵 | 18,599 / 23,052（80.7%） |
| 特征齐全独立条件 | 18,326 / 22,737 |
| 特征齐全单药曲线 | 1,782 / 2,094（85.1%） |
| 涉及药物对（有齐全矩阵） | 539 / 583 |

缺失原因（矩阵计数，一行可多因）：
`drug_graph_not_model_ready` 1,716、`cell_unmapped` 1,187、
`cell_expression_missing` 1,166、`cell_pathway_missing` 604。

## 7. 复现

```bash
cd /home/kouzheng_2/AH_YeP/DoseSyn
.venv/bin/python scripts/02_build_features.py            # 全流程 + 测试 + 报告
.venv/bin/python scripts/02_build_features.py --stage drugs
.venv/bin/python scripts/02_build_features.py --stage cells
.venv/bin/python scripts/02_build_features.py --stage coverage
.venv/bin/python -m pytest -q tests
```

产物：

- `data/processed/drug_mapping.csv`
- `data/processed/drug_graph_features.pt`、`drug_graph_metadata.json`
- `data/processed/cell_mapping.csv`
- `data/processed/cell_expression.parquet`
- `data/processed/cell_features.parquet`、`pathway_features.parquet`、`pathway_metadata.csv`
- `data/processed/oneil_combo_feature_index.parquet` 等 3 张索引表
- `outputs/audit/feature_coverage.json`、`m2_drug_build.json`、`m2_cell_pathway_build.json`、`m2_manifest.json`
- `m2_report.md`
