# DoseSyn — 双药组合数据、特征与基准划分工程

本项目已完成 **M0（原始数据审计）**、**M1（标准化预处理）**、
**M2（药物结构、细胞系、表达谱与 Hallmark 特征构建）** 与
**M3（Primary-v1 数据集冻结、S1/S2/S3 划分、单药监督隔离、训练折专属变换与加载接口）**。
当前**尚未实现或训练任何神经网络**，也未进入 M4。M2 见
[`docs/decisions/m2_feature_spec.md`](docs/decisions/m2_feature_spec.md) 与
[`m2_report.md`](m2_report.md)；M3 见
[`docs/decisions/m3_primary_dataset.md`](docs/decisions/m3_primary_dataset.md)、
[`docs/decisions/m3_split_protocols.md`](docs/decisions/m3_split_protocols.md) 与
[`m3_report.md`](m3_report.md)。

## 1. 结论速览

| 项目 | 结果 |
| --- | --- |
| M0 审计 | **PASS**（SHA256 一致、真实容器识别、行数对账、字段核对） |
| M1 预处理 | **PASS**（368,832 双药点 / 23,052 矩阵 / 2,094 单药曲线） |
| 自动测试 | **53 passed / 0 failed** |
| M2 准入 | **可进入 M2**（无阻塞项） |
| 原始文件 | 只读；未修改、未覆盖、未重命名 |

## 2. 输入数据

| 角色 | 文件 | 大小 | SHA256（前 16 位） | 真实容器 |
| --- | --- | ---: | --- | --- |
| 双药响应 | `15357163mct150843-sup-156849_1_supp_1_w2lrww.xls` | 38,829,879 B | `a0e2e002dc06c34e` | OOXML/ZIP |
| 单药响应 | `15357163mct150843-sup-156849_1_supp_0_w2lh45.xlsx` | 1,678,111 B | `4a0622c074b9c505` | OOXML/ZIP |
| 字段说明 | `15357163mct150843-sup-156849_1_supp_1_g2lr29.xlsx` | 10,430 B | `4a85ebffc9a3c94b` | OOXML/ZIP |

> ⚠️ 双药文件后缀是 `.xls`，但 magic bytes 为 `504b0304`，实际是 OOXML 容器。
> 若交给旧版 BIFF/xlrd 读取器会直接失败。流程会在
> `data/interim/format_work/` 生成字节级一致的 `.xlsx` 工作副本，再交给 openpyxl；
> 原始文件在 `data/raw/ONEIL/` 中保持不变，并在每次审计时重新计算 SHA256。

## 3. 环境与安装

宿主机无可用 pandas 的系统 Python，因此使用 conda 的 `confsyn` 解释器创建项目级 venv：

```bash
cd /home/kouzheng_2/AH_YeP/DoseSyn

# 1) 已执行（记录在案）
/home/kouzheng_2/miniconda3/envs/confsyn/bin/python -m venv \
    --without-pip --system-site-packages .venv
.venv/bin/python -m pip install pyarrow==10.0.1 pytest==7.4.4

# 2) 后续复现
.venv/bin/python -c "import pandas, pyarrow, pytest; print('ok')"
```

已安装的关键版本：Python 3.10.21、pandas 1.3.5、numpy 1.21.6、
openpyxl 3.1.2、pyarrow 10.0.1、pytest 7.4.4。

## 4. 执行命令

```bash
cd /home/kouzheng_2/AH_YeP/DoseSyn

# M0：原始文件审计（只读原始文件，生成清单/字段/缓存）
.venv/bin/python scripts/00_audit_raw.py

# M1：标准化、矩阵重建、单药曲线、QC、异常清单、数据字典、自动测试
.venv/bin/python scripts/01_prepare_oneil.py

# 仅运行测试
.venv/bin/python -m pytest -q tests
```

命令在终端约 20 秒会切到后台执行，完整日志见 `outputs/logs/`。

## 5. 数据产物

### 5.1 标准 Parquet 表（`data/processed/`）

| 文件 | 行数 | 说明 |
| --- | ---: | --- |
| `oneil_combo_points.parquet` | 368,832 | 双药逐点响应，含 `matrix_id/row_idx/col_idx`、µM 浓度、全部技术重复、发布值 `mu_raw/x_x0_raw` |
| `oneil_combo_matrix_metadata.parquet` | 23,052 | 4×4 矩阵元数据：剂量网格、完整性、缺失、异常、批次 |
| `oneil_combo_replicates.parquet` | 1,475,328 | 技术重复长表（含缺失槽位，不插补） |
| `oneil_monotherapy.parquet` | 16,752 | 单药逐点响应，含 `mono_curve_id/row_idx` |
| `oneil_mono_curve_metadata.parquet` | 2,094 | 单药曲线元数据 |
| `oneil_mono_replicates.parquet` | 100,512 | 单药技术重复长表 |

所有响应值逐列保留：`mu_raw` 与 `x_x0_raw` 是发布值原样；重算列
`x_x0_recalc = 4**(mu-1)` 仅用于审计，最大绝对残差 5×10⁻⁶。
**没有**裁剪、winsorize、去重或按均值合并任何响应。

### 5.2 审计与 QC（`outputs/audit/`）

- `raw_file_manifest.json` — 哈希、真实容器、字段、行数、实体、审计结论
- `raw_schema.md` — 逐字段类型/范围/缺失
- `oneil_entity_registry.json` — 39 细胞系、38 药物、583 无序药物对
- `oneil_qc.json` / `oneil_qc.md` — 机器可读 + 人类可读 QC 报告（含 M2 准入表）
- `oneil_anomalies.csv` + `anomalies_by_type/*.csv` — 22,605 条异常
- `batch_repeats.csv` / `mono_batch_repeats.csv` — 跨批次重复审计
- `oneil_entity_index.csv` — 每个实体的点数/矩阵数索引
- `sample_matrices/` — 5 个代表性矩阵的 `x_x0`、`viability_mean`、原始点表
- `pytest.log` — 测试日志

### 5.3 数据字典

`docs/data_dictionary/oneil_datasets.md`，包含每张表的字段、类型、缺失率、
单位与公式、ID 规则、QC 阈值及理由、已知限制。

## 6. 关键数据事实

- 双药：368,832 行 = 23,052 个 16 点矩阵；每矩阵 4×4、16 个唯一剂量坐标；
  22,737 个 `(细胞系, 药物对)` 条件；583 个无序药物对；39 细胞系；38 药物。
- 单药：16,752 行 = 2,094 条 8 点曲线；1,482 个 `(细胞系, 药物)` 条件。
- 行顺序为 A 慢 / B 快，`row_idx` 为 A 剂量秩、`col_idx` 为 B 剂量秩（0 起）。
- 浓度全部 > 0（双药 1.1×10⁻⁴–250 µM；单药 2.2×10⁻⁶–250 µM），
  **没有真正的零剂量锚点**；LG0/协同基线需在 M2 另行定义。
- 跨批次重复：315 个条件；其中 105 个剂量网格相同、210 个不同。
  `BatchID=3` 是部分重复验证，不是完整独立重复。
- 缺失技术重复：双药 6,018 个值、单药 15,600 个值，保留为 NaN。
- 双药 `X/X0 > 1` 共 25,247 点，`> 2` 共 7 点（6 个 T47D、1 个 ZR751，
  最大 2.34333）；单药 `> 1` 共 5,512 点（最大 1.71937），全部保留。
- 双药 213 个 `X/X0` 精确为 0（公式下限，四舍五入），保留原值并标记。
- 技术重复极差：双药最大 2.182、767 个点超过阈值 1.0；
  单药 107 个点超过 1.0。

## 7. 工程结构

```
DoseSyn/
├── configs/data_oneil.yaml          # 路径、哈希、字段、QC 阈值、单位
├── scripts/
│   ├── 00_audit_raw.py              # M0 审计
│   └── 01_prepare_oneil.py          # M1 标准化 + QC + 测试 + 文档
├── src/dosesyn/
│   ├── data/inspect.py              # 哈希、容器检测、Excel 读取、legend 解析
│   ├── data/parse_oneil.py          # 列标准化、ID 映射、重复统计
│   ├── data/normalize.py            # 单位换算、log10、响应重算、重复统计
│   ├── data/ids.py                  # 稳定 ID 与剂量网格 token
│   ├── data/make_matrices.py        # 4×4 矩阵/单药曲线/重复长表/结构校验
│   ├── data/qc.py                   # 异常、批次重复、示例矩阵、验收、报告
│   ├── data/schema.py               # 数据字典生成
│   └── utils/{config,io,logging,seed}.py
├── tests/                           # 5 个测试文件 + conftest
├── data/{raw,interim,processed,external,mappings,splits}
├── outputs/{audit,logs,runs,metrics,figures,tables,bridge}
└── docs/{data_dictionary,decisions,experiment_log,chapter4}
```

## 8. 尚未解决的问题（进入 M2 前需知）

1. `.xls` 后缀与真实 OOXML 格式不符；外部工具若按后缀解析会失败。
2. 7 个 `X/X0 > 2` 响应点保留并标记，M2 应做敏感性分析。
3. 213 个 `X/X0 = 0` 点使用 `log` 类特征时需偏移或改用 `mu`。
4. 315 个跨批次重复条件（210 个网格不同）不可平均合并；M2 需批次分层。
5. 源数据无零剂量锚点，协同零基线需要外部定义。
6. Batch 3 为部分验证批次，不能视为等价随机重复。
7. M1 未生成 train/val/test 划分；须在 M2 按 `condition_id` 分组，防止同条件泄漏。
8. 缺失重复值未插补，M2 建模需显式处理 NaN 或使用聚合统计量。

## 9. M2 准入判定

对照总规划验收标准：

| 条件 | 结果 |
| --- | --- |
| 三个文件可读、SHA256 不变、`.xls` 未走旧版 XLS 引擎 | PASS |
| 行数与源表对账一致（368,832 / 16,752），无静默丢行 | PASS |
| 矩阵/曲线唯一且完整（4×4=16，单药 8 点） | PASS |
| 单位统一 µM、响应公式可复算（残差 ≤ 1×10⁻⁵） | PASS |
| `X/X0 > 1`（含 >2）原值保留 | PASS |
| 技术重复逐列保留、缺失不插补 | PASS |
| 每个响应点可追溯至 `source_row` | PASS |
| 跨批次重复已识别、未合并 | PASS |
| 自动化测试全绿 | PASS（53/53） |

**结论：满足进入 M2 的条件。** 本轮未创建任何模型或训练脚本。

## 10. M2 阶段（多模态特征构建）

执行：

```bash
cd /home/kouzheng_2/AH_YeP/DoseSyn
.venv/bin/python scripts/02_build_features.py      # 全流程 + pytest + m2_report.md
```

产物：

- `data/processed/drug_mapping.csv`（38 药物，SMILES/CID/InChIKey/状态）
- `data/processed/drug_graph_features.pt` + `drug_graph_metadata.json`（PyG 图缓存）
- `data/processed/cell_mapping.csv`（39 细胞系 → DepMap）
- `data/processed/cell_expression.parquet`（34 × 19,144 基因）
- `data/processed/cell_features.parquet`、`pathway_features.parquet`、`pathway_metadata.csv`
  （34 × 50 Hallmark）
- `data/processed/oneil_combo_feature_index.parquet`、
  `oneil_condition_feature_index.parquet`、`oneil_mono_feature_index.parquet`
- `outputs/audit/feature_coverage.json`、`m2_manifest.json`
- `m2_report.md`

M2 关键结果：

| 指标 | 结果 |
| --- | --- |
| 药物 SMILES / RDKit 图 | 38 / 38（36 OK，2 NEEDS_REVIEW） |
| 细胞系 DepMap 映射 | 37 / 39（34 个特征齐全） |
| Hallmark 有效通路 | 50 / 50，覆盖率 96.5%–100% |
| 特征齐全组合矩阵 | 18,599 / 23,052（80.7%） |
| 特征齐全独立条件 | 18,326 / 22,737 |
| 特征齐全单药曲线 | 1,782 / 2,094（85.1%） |
| 自动测试 | 81 passed（含 M0/M1 原有 53 项） |

无法映射的实体、受影响矩阵数量及处理理由见 `m2_report.md` 第 6 节。



## 11. M3：数据集冻结与基准划分

### 11.1 命令

```bash
cd /home/kouzheng_2/AH_YeP/DoseSyn

.venv/bin/python scripts/03_generate_splits.py --scenario all  # 冻结 + 11 个 split
.venv/bin/python scripts/03_verify_splits.py                  # 泄漏复核
.venv/bin/python -m pytest -q tests                            # 全量测试
.venv/bin/python scripts/04_write_m3_report.py                 # m3_report.md
```

### 11.2 产物

- `data/processed/oneil_primary_v1_{matrix,mono}_index.parquet`
- `data/processed/oneil_primary_v1_exclusions.parquet`
- `data/splits/{s1_random_condition,s2_unseen_pair,s3_unseen_cell}/...`
  （每个目录含 `train/val/test_conditions.parquet`、`mono_assignments.parquet`、
  `scalers.joblib`、`split_metadata.json`）
- `outputs/audit/m3_{input_manifest,manifest,leakage_report}.json`
- `m3_report.md`

### 11.3 关键结果

| 指标 | 结果 |
| --- | --- |
| Primary-v1 矩阵 / 单药曲线 | 18,599 / 1,782 |
| 条件 / 细胞系 / 药物 / 无序对 | 18,326 / 34 / 36 / 539 |
| S1（condition，3 seeds） | 70/15/15，matrix 约 13.0k/2.8k/2.8k |
| S2（unseen pair，3 seeds） | 70/15/15，pair 完全隔离 |
| S3（unseen cell，5 folds） | test 6–7 细胞/折，val 6–7，train 20–21 |
| 单药监督 | S1/S2 已知先验（1,782）；S3 严格隔离（990 trainable / 792 sealed） |
| 泄漏检查 | 11/11 split PASS |
| 自动测试 | 114 passed（M0/M1/M2 81 + M3 33） |

M3 未创建任何模型代码，也未生成训练/验证/测试集之外的调参结果。
