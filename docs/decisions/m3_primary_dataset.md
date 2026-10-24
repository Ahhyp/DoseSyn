# M3 决策：Oneil-Primary-v1 数据集冻结

日期：2026-10-24  
决策状态：已执行（M3）  
上游基线：M2 commit `cdd9c8b`

## 1. 目的

M3 需要一个**不可变、可追溯、特征齐全**的主数据集，供 S1/S2/S3 三种划分复用。
本文件记录冻结范围、排除规则和每个被排除实体的去向。冻结产物：

| 文件 | 内容 |
| --- | --- |
| `data/processed/oneil_primary_v1_matrix_index.parquet` | 18,599 个 4×4 组合矩阵（36 药、34 细胞系、539 无序对、297,584 点） |
| `data/processed/oneil_primary_v1_mono_index.parquet` | 1,782 条单药曲线 |
| `data/processed/oneil_primary_v1_exclusions.parquet` | 所有被排除矩阵/曲线的逐行原因 |
| `outputs/audit/m3_input_manifest.json` | 输入哈希、计数、排除汇总、生成信息 |

上游 M1/M2 文件在冻结后**只读且哈希不变**（测试 `test_m1_m2_inputs_unchanged_by_m3` 逐文件复核）。

## 2. 纳入条件

一个组合矩阵进入 Primary-v1，必须同时满足：

1. M2 `feature_complete == True`（两个药物图 OK + 细胞具备表达/Hallmark 特征）；
2. 细胞映射状态为 `VERIFIED`/`VERIFIED_ALIAS` 且 `feature_eligible`；
3. 矩阵结构完整：`complete_matrix == True`、16 个点、4×4 剂量；
4. 药物不在一期排除清单（见 §3）；
5. 标签列为 M1 原始 `X/X0`，**不裁剪、不取 sigmoid、不重算**。

单药曲线额外要求 8 个剂量点且药物图 OK。

## 3. 排除 Carboplatin / Oxaliplatin 的理由

M2 的 RDKit 图基于共价 SMILES：铂类药物的配位键无法由 RDKit 常规键型表达，
Carboplatin 被解析为 4 个断开的片段，Oxaliplatin 为 3 个片段（`graph_status=NEEDS_REVIEW`）。
把这种图交给 GINE 会把“一个铂配合物”当成多个无关分子，属于结构性错误。

因此：

- **不删除原始数据**：M0/M1/M2 的原始行、映射、SMILES、CID、InChIKey 全部保留；
- 只从 Primary-v1 建模集合中排除，并把每个受影响矩阵写入 `oneil_primary_v1_exclusions.parquet`；
- 若后续补上配位键表示或改用 2D 描述符，可重新纳入，无需重跑 M0–M2。

## 4. 其他未进入 Primary-v1 的实体

| 实体 | 状态 | 影响 |
| --- | --- | --- |
| EFM192B、UWB1289BRCA1 | `UNMAPPED`（DepMap 无对应） | 其条件不可用 |
| COLO320DM | `NEEDS_REVIEW`（仅 COLO320） | 其条件不可用 |
| OCUBM、UWB1289 | 已映射但 CCLE 表达缺失 | 其条件不可用 |
| Carboplatin、Oxaliplatin | 图 `NEEDS_REVIEW` | 1,716 个矩阵被排除 |

所有排除均记录 `exclusion_reason`，没有任何静默丢弃。排除原因计数写入
`m3_input_manifest.json` 与 `m3_report.md`。

## 5. 冻结校验

`freeze_primary()` 在生成时断言以下计数，任一不符立即失败：
18,599 矩阵 / 18,326 条件 / 34 细胞 / 36 药 / 539 对 / 1,782 单药曲线 / 297,584 点。
测试 `tests/test_m3_primary.py` 再次独立复核，并校验：
- `matrix_id` 唯一、每矩阵 16 点；
- 排除实体存在于 `drug_mapping.csv` 和 exclusions 表；
- 输入哈希与 `m3_input_manifest.json` 一致。

## 6. 复现

```bash
cd /home/kouzheng_2/AH_YeP/DoseSyn
.venv/bin/python scripts/03_generate_splits.py --scenario all
.venv/bin/python scripts/03_verify_splits.py
.venv/bin/python -m pytest -q tests
```
