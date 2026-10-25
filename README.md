# DoseSyn — 双药组合数据、特征与基准划分工程

本项目已完成 **M0（原始数据审计）**、**M1（标准化预处理）**、
**M2（药物结构、细胞系、表达谱与 Hallmark 特征构建）**、
**M3（Primary-v1 数据集冻结、S1/S2/S3 划分、单药监督隔离、训练折专属变换与加载接口）**
与 **M4（B0/B1/B2/B3 基线训练与统一评测框架）**。
M4 仅是基线对照，**尚未实现完整 DoseSyn**（无 FiLM、单药边界、平滑正则或协同重建）。M2 见
[`docs/decisions/m2_feature_spec.md`](docs/decisions/m2_feature_spec.md) 与
[`m2_report.md`](m2_report.md)；M3 见
[`docs/decisions/m3_primary_dataset.md`](docs/decisions/m3_primary_dataset.md)、
[`docs/decisions/m3_split_protocols.md`](docs/decisions/m3_split_protocols.md) 与
[`m3_report.md`](m3_report.md)；M4 见
[`docs/decisions/m4_training_protocol.md`](docs/decisions/m4_training_protocol.md) 与
[`m4_report.md`](m4_report.md)。

## 1. 结论速览

| 项目 | 结果 |
| --- | --- |
| M0 审计 | **PASS**（SHA256 一致、真实容器识别、行数对账、字段核对） |
| M1 预处理 | **PASS**（368,832 双药点 / 23,052 矩阵 / 2,094 单药曲线） |
| M2 特征 | **PASS**（38 药物 / 39 细胞 / 50 Hallmark；18,599 特征齐全矩阵） |
| M3 数据冻结与划分 | **PASS**（11 个 split，泄漏检查 11/11） |
| M4 基线 | **PASS**（44/44 运行成功；B3 等权平均 0.1228，B2 0.1273） |
| 自动测试 | **131 passed / 0 failed** |
| 下一阶段 | M5（完整 DoseSyn），尚未开始 |
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

## 12. M4 基线训练

M4 的目标是建立统一训练/验证/测试、指标、checkpoint 与资源统计框架，并比较
**B0 Mean**、**B1 Tree-Dose**、**B2 MLP-Dose**、**B3 GINE-Base** 四类基线。
不包含 FiLM、单药边界、平滑正则或协同重建（留给 M5）。

### 12.1 命令

```bash
cd /home/kouzheng_2/AH_YeP/DoseSyn

# 单个 split（GPU 节点）
/opt/software/torque-6.1.0/bin/qsub \
  -v scenario=s1_random_condition,kind=seed,value=42,gpu=0 \
  scripts/m4_gpu_job.pbs

# S1 seed42 四模型（前台调试）
.venv/bin/python scripts/04_run_baseline.py \
  --scenario s1_random_condition --seed 42 \
  --models b0_mean b1_tree b2_mlp b3_gine --device cuda

# 汇总、作图、报告
.venv/bin/python scripts/05_aggregate_m4.py

# 全量测试（M0–M4）
.venv/bin/python -m pytest -q tests
```

集群注意：node1 有 2×V100-32GB；Torque 不跟踪 GPU（`qnodes gres=` 为空），
需人工看 `nvidia-smi`；PBS `-o` 不生效，作业脚本用 `exec > log 2>&1` 自重定向。

### 12.2 产物

- `outputs/runs/m4/<run_id>/`：`best.pt`、`train_log.csv`、`val/test_metrics.json`、
  `predictions_{val,test}.parquet`、`resources.json`、`run_metadata.json`
- `outputs/metrics/m4/{runs_summary,aggregate_summary,overall_by_model}.csv`、
  `m4_summary.json`、`m4_report_manifest.json`
- `outputs/figures/m4/`：逐 run 训练曲线 SVG、`macro_rmse_by_model_scenario.svg`
- `m4_report.md`

`predictions_test.parquet` 每矩阵 16 行，含 `matrix_id/condition_id/cell_id/drug_a_id/
drug_b_id/row_idx/col_idx/dose_a_um/dose_b_um/y_true/y_pred/residual`；测试标签为原始
`X/X0`，未裁剪，允许负预测。

### 12.3 结果（测试集，11 个 split，44 个运行全部成功）

| 模型 | S1 已见条件 | S2 未见药物对 | S3 未见细胞系 | 等权平均 |
| --- | ---: | ---: | ---: | ---: |
| B0 Mean | 0.3060 | 0.3022 | 0.3066 | 0.3049 |
| B1 Tree-Dose | 0.2467 | 0.2497 | 0.2819 | 0.2594 |
| B2 MLP-Dose | **0.0764** | **0.1022** | 0.2033 | 0.1273 |
| B3 GINE-Base | 0.0795 | 0.0897 | **0.1994** | **0.1228** |

数值为 `matrix_macro_RMSE`（先逐 4×4 矩阵算 RMSE，再对矩阵取宏平均，越低越好）。
S1 跨 3 个种子，S2 跨 3 个种子，S3 跨 5 折。

- **S1**：B2 略优于 B3；**S2/S3**：B3 略优于 B2。两者都远优于 B1/B0。
- **S3**：相对 B0/B1 的优势明显缩小——未见细胞系是主要难点。
- B0 的 R² 约为 0，说明简单均值基线没有解释力；B1 的 R² 在 S1/S2 约 0.28–0.32，
  S3 降到约 0.13。

### 12.4 防泄漏与可追溯

- 直接读取 M3 冻结的 `train/val/test_conditions.parquet`；没有重新划分，
  没有点级随机拆分，没有删除或改动任何测试矩阵。
- 同 condition/batch 的 16 点矩阵始终在同一 split；S2 按无序药物对隔离；
  S3 按细胞系隔离。
- 特征标准化只使用 M3 训练折拟合的 `scalers.joblib`（路径和 fit 集合在
  `split_metadata.json` 中可查）。
- 早停与 LR 调度只看验证集；测试集只在训练结束后评估一次。
- 每个 run 记录 `git_commit`、split/scaler 哈希、配置哈希、环境、参数量、
  训练时间与 GPU 峰值显存。

### 12.5 已知问题

- 本阶段未做超参搜索；S1 下 B2 略胜 B3，S2/S3 下 B3 略胜 B2，不能据此断言图结构无用。
- S3 各折癌症类型不均衡（如 fold0 test 无前列腺细胞），指标方差较大。
- 预测未做 `[0, 2.5]` 物理裁剪；如需约束应在模型侧实现，不能改测试标签。
- B3 仅用均值池化 + 拼接，未使用 FiLM、单药边界、平滑正则或协同重建。
- 正式训练前 2 个作业因安装 matplotlib 时损坏 numpy 失败，修复后重跑；
  最终 44 个结果均来自成功作业。

---

## 13. M5：完整 DoseSyn 网络（验证集开发）

### 13.1 模型组件

- **药物编码**：共享 GINE（9 类原子特征 + 4 类键特征），两种药物同一编码器。
- **细胞编码**：50 维 Hallmark ssGSEA → MLP。
- **连续剂量编码**：`scalar` / `mlp` / `fourier` 三种可选，零剂量用显式 flag，不取 `log10(0)`。
- **剂量条件交互**：每个药物独立的 drug-cell 交互 MLP + FiLM `(γ,β)=MLP(dose)`，
  `t=(1+γ)·u+β`，FiLM 末层零初始化。
- **严格交换对称融合**：`[t1+t2, t1⊙t2, |t1−t2|, h_cell]`，不含任何有序项；
  C0 保留有序对照用于验证“对称不是表面改名”。
- **共享 Monotherapy Head**：`M(d,x,c)=MonoHead(GINE(d), CellEncoder(c), DoseEncoder(x))`。
- **零剂量边界**：虚拟查询 `F(d1,x1,d2,0)≈M(d1,x1)`、`F(d1,0,d2,x2)≈M(d2,x2)`。
- **局部曲面平滑**：在矩阵自身对数剂量网格上做非均匀二阶差分，不跨矩阵、不假设单调性。

### 13.2 消融开关

`DoseSynCore` 与配置支持：`symmetric_pair`、`dose_encoding`、`film`、
`mono_weight`、`boundary_weight`、`smooth_weight`。C0–C6 依次打开，
为 M6 的正式消融准备入口。

### 13.3 单药标签协议

- S1/S2 使用 `known_monotherapy`：1,782 条曲线全部可训练，仅作为已知先验披露，
  不能当作独立测试证据。
- S3 使用 `strict_cell_isolation`：fold0 仅 990 条训练细胞曲线可训练，
  792 条验证/测试细胞曲线封存；读取封存标签会抛 `MonoAccessError`。
- 单药损失与 combo 损失联合优化，系数显式写入配置与 run 元数据。

### 13.4 协同语义

`X/X0` 到 Bliss/HSA/Loewe/ZIP 的转换需要零剂量锚点与可逆单药效应，
本数据不满足，全部标记 `NOT_VALIDATED`。M5 **不输出任何“标准协同分数”**，
只报告 `E_swap`、`E_boundary`、`R_smooth` 等自洽性诊断。
详见 `docs/decisions/m5_synergy_reconstruction_semantics.md`。

### 13.5 命令

```bash
# 单元测试（151 项）
.venv/bin/python -m pytest -q tests

# S1 seed42 验证集：C0–C6
.venv/bin/python scripts/06_run_dosesyn.py --scenario s1_random_condition --seed 42 \
  --variants c0_control c1_symmetric c2_fourier c3_film c4_mono c5_boundary c6_full --device cuda --force

# S3 fold0 验证集：C0/C1/C2/C3/C6
.venv/bin/python scripts/06_run_dosesyn.py --scenario s3_unseen_cell --fold 0 \
  --variants c0_control c1_symmetric c2_fourier c3_film c6_full --device cuda --force

# 报告与验收
.venv/bin/python scripts/06_validate_m5.py --pytest-log outputs/audit/pytest_m5.log
```

### 13.6 结果

真实验证指标、收敛曲线、参数量、失败实验与逐 run 产物见
[`m5_report.md`](m5_report.md)。M5 只使用验证集，未读取测试集；
技术验收标准是组件/协议/数值正确，而不是必须超过 M4 B3。
## 14. M6-A 性能复现与结构诊断

M6-A 只做诊断，不读测试集、不跑 11 split、不修改 M0–M3 冻结数据或 M4/M5 产物。

### 14.1 目的

解释 M5 C0 在 S1 seed42 验证集（matrix-macro RMSE 0.1820）为何明显差于 M4 B3（0.0801），并建立同口径、可复现的对照。

### 14.2 关键结论

| 对照 | matrix-macro RMSE | 说明 |
| --- | ---: | --- |
| T0 B3 checkpoint 复现 | 0.0801 | 与 M4 存储值一致 |
| B3 架构 + M5 配方 | 0.0820 | 配方本身不导致退化 |
| C0 架构 + M4 配方 | 0.1820 | 换配方也救不回 C0 |
| T2 对称 + 标量剂量 | 0.1015 | 去掉 ordered 头只恢复一半 |
| T3 对称 + Fourier 剂量 | 0.0820 | 恢复 B3 水平；S3 0.1695 |
| T4 对称 + Fourier + FiLM | **0.0670** | S1 最佳；S3 0.1734 过拟合 |

结论：M4→M5 的性能下降不是训练配方或早停口径造成的，而是 C0 的结构问题；Fourier 剂量表示是恢复性能的关键组件，FiLM 只在同分布 S1 上增益。M6-B 默认建议 T3 `b3sym_fourier`，T4 作为 S1 场景/集成候选。

### 14.3 复现命令

```bash
# 6 个正式验证 run（S1 seed42：T2/T3/T4；S3 fold0：T2/T3/T4）
.venv/bin/python scripts/08_run_m6a.py --all --recipe m6a --epochs 160 --device cuda

# 从 best.pt 导出 16 行/矩阵的验证预测
.venv/bin/python scripts/10_export_m6a_predictions.py --device cpu

# 报告 / 检查项 / 产物清单
.venv/bin/python scripts/09_write_m6a_report.py --write

# 全量测试（172 项）
.venv/bin/python -m pytest -q tests
```

报告：[`m6a_report.md`](m6a_report.md)，
结构差异：[`m6a_architecture_diff.md`](m6a_architecture_diff.md)，
模型选择：[`m6a_model_selection.md`](m6a_model_selection.md)。

### 14.4 未解决问题

- FiLM 在 S3 过拟合（最佳 epoch 20/50），跨细胞系泛化不稳定。
- 未做超参数搜索；M6-A 的结论仅基于 S1 seed42 / S3 fold0 验证集。
- 测试集未读取；M6-B 才允许在验证集选型后做正式测试评估。

## 15. M6-B 正式模型与多场景评估（M6-B）

### 15.1 协议

- 候选来自 M6-A：T1/T2/T3/T4 + 单药/边界/平滑组合，共 5 个变体（B1–B5）；
  全部在 **验证集** S1 seed42 + S3 fold0 上比较，规则预先固定：
  均值至少改善 0.001，且 S3 退化不超过 0.002，否则保留 B1。
- 验证结果（matrix-macro RMSE，S1 / S3 / 均值）：

| 变体 | S1 seed42 | S3 fold0 | 均值 | 参数量 |
| --- | ---: | ---: | ---: | ---: |
| B1 T3 Fourier | 0.0798 | 0.1662 | 0.1230 | 498,818 |
| B2 T3 + Mono | 0.0818 | 0.1687 | 0.1252 | 548,227 |
| B3 T3 + Mono + Boundary | 0.0809 | 0.1665 | 0.1237 | 548,227 |
| B4 T3 + Mono + Boundary + Smooth | 0.1124 | 0.1693 | 0.1409 | 548,227 |
| B5 T3 + Regularized FiLM | **0.0649** | 0.1677 | **0.1163** | 535,682 |

- 选型只使用验证集，选中 **B5 `b5_t3_film_reg`**；随后才在 M3 冻结的
  11 个 split 上一次性读取测试集，未用测试标签调参。

### 15.2 正式测试结果

11 个冻结 split：S1 3 seeds、S2 3 seeds、S3 5 folds；每场景取 run 均值的再平均。

| 场景 | B1 T3 Fourier | B5 + Reg-FiLM | 相对变化 |
| --- | ---: | ---: | ---: |
| S1 随机条件 | 0.0793 | **0.0650** | −18.0% |
| S2 未见药物对 | 0.1085 | **0.0768** | −29.2% |
| S3 未见细胞系 | 0.1966 | **0.1937** | −1.5% |
| 三场景等权平均 | 0.1281 | **0.1118** | −12.7% |

S3 的 5 折测试 RMSE 分别为
0.1981 / 0.1827 / 0.2135 / 0.1852 / 0.1893（B5），
0.2087 / 0.1758 / 0.2187 / 0.1840 / 0.1958（B1）。
逐折差异有正有负，S3 的 1.5% 均值优势落在折间波动范围内，**不能**声称
FiLM 改善了跨细胞系泛化。

### 15.3 解释

- FiLM 的收益主要来自同分布/同药物对场景（S1、S2），符合“剂量条件调制”
  的预期；在未见细胞系上收益很小。
- 单药辅助、边界损失、平滑损失在本预算下没有带来验证集收益：
  B2/B3 与 B1 基本持平，B4 明显变差（平滑项抑制了曲面细节）。
- 对称性由结构保证：B1/B5 的 swap max abs error ≈ 3×10⁻⁷；
  E_boundary 是模型自洽诊断，不是实测零剂量误差。
- 所有测试预测未裁剪：负预测保留，`X/X0 > 1` 原值保留。

### 15.4 复现命令

```bash
# 验证集选型（5 变体 × S1 seed42/S3 fold0）
.venv/bin/python scripts/11_run_m6b.py --stage validation --epochs 240 --device cuda
.venv/bin/python scripts/12_select_m6b.py

# 正式 11 split（B1 参考 + 选中的 B5）
.venv/bin/python scripts/11_run_m6b.py --stage formal --model b1_t3_fourier --epochs 200 --eval-test --device cuda
.venv/bin/python scripts/11_run_m6b.py --stage formal --model b5_t3_film_reg --epochs 160 --eval-test --device cuda

# 汇总 / 图表 / 报告 / 全量测试
.venv/bin/python scripts/13_aggregate_m6b.py --write
.venv/bin/python -m pytest -q tests
```

报告：[`m6b_report.md`](m6b_report.md)；选型：[`m6b_model_selection.md`](m6b_model_selection.md)；
协议：[`docs/decisions/m6b_protocol.md`](docs/decisions/m6b_protocol.md)。

### 15.5 未解决问题

- FiLM 在 S3 没有稳定收益，5 折方向不一致；跨细胞系泛化仍是主要瓶颈。
- B5 的 S1 正式 run 使用 160 epoch（B1 200），最佳 epoch 接近上限，仍有欠训练可能。
- 未做超参数搜索；B5 的收益是单次选型结果，需在 M7 外部数据上复核。
- 边界/平滑项的权重未系统调参；B4 的负结果说明当前平滑权重过强。

---

## 16. M7-A 外部 Baseline 复现与统一协议对比

M7-A 在 M3 冻结的 11 个 split 上复现/适配了两个外部方法，并与 M4、M6-B 在统一 matrix-macro RMSE 口径下比较。

| 模型 | S1 | S2 | S3 | 等权平均 | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| M4 B0 Mean | 0.3060 | 0.3022 | 0.3066 | 0.3049 | 归档 |
| M4 B1 Tree-Dose | 0.2467 | 0.2497 | 0.2819 | 0.2594 | 归档 |
| M4 B2 MLP-Dose | 0.0764 | 0.1022 | 0.2033 | 0.1273 | 归档 |
| M4 B3 GINE | 0.0795 | 0.0897 | 0.1994 | 0.1228 | 归档 |
| M6-B B1 (T3) | 0.0793 | 0.1085 | 0.1966 | 0.1281 | 归档 |
| **M6-B B5 (T3+FiLM)** | **0.0650** | **0.0768** | **0.1937** | **0.1118** | 归档，最终模型 |
| comboKR 2.0 | 0.1341 | 0.1361 | — | 0.1351* | 本阶段，忠实复现 |
| DeepSynBa (adapted) | 0.2353 | 0.2670 | 0.2754 | 0.2592 | 本阶段，仅补充 |

\* comboKR 的 S3 因 Hill 参数包含测试细胞系单药信息而按协议不运行，其平均只覆盖 S1/S2。

- comboKR：官方代码 + 官方 BRAID 拟合 + RBF 核；本项目 Primary-v1 有 18,235/18,599 个矩阵可映射（MK-8776 缺失）。为排除矩阵组成影响，另做了 comboKR 可用矩阵子集上的同口径对比：B5 仍为 0.0646/0.0756，comboKR 为 0.1341/0.1361。
- DeepSynBa：官方结构保留，但药物输入由 MoLFormer 换为 Morgan 指纹、细胞由 NCI-60 换为 CCLE、目标由 SynBa 参数换为 X/X0 曲面，只能作为 adapted supplementary。
- DD-PRiSM：未找到公开可运行代码，记为不可复现，未编造数字。

```bash
# comboKR 适配层（/tmp/m7venv）
/tmp/m7venv/bin/python src/dosesyn/baselines/external/combokr_adapter.py \
  --scenario s1_random_condition --seed 42 --split test \
  --out outputs/runs/m7a/combokr/s1_random_condition/seed_42/predictions.parquet

# DeepSynBa 适配层（项目 .venv）
.venv/bin/python src/dosesyn/baselines/external/deepsynba_adapter.py \
  --scenario s1_random_condition --seed 42 --epochs 60 --patience 10 \
  --batch-size 512 --device cuda \
  --out outputs/runs/m7a/deepsynba/s1_random_condition/seed_42/predictions.parquet

# 汇总、同子集对比、测试
.venv/bin/python scripts/16_aggregate_m7a.py
.venv/bin/python scripts/17_m7a_common_subset.py
.venv/bin/python -m pytest -q tests
```

报告：[`m7a_report.md`](m7a_report.md)；可行性：[`docs/decisions/m7a_baseline_feasibility.md`](docs/decisions/m7a_baseline_feasibility.md)；
协议：[`docs/decisions/m7a_experiment_protocol.md`](docs/decisions/m7a_experiment_protocol.md)。

**M7-A 结论：CONDITIONAL PASS。** 主排名结论成立，但 comboKR 的 S3 不适用、DeepSynBa 非忠实复现、DD-PRiSM 无代码，须在论文中明确标注。

## 18. M7-B：剂量位置插值与稀疏曲面泛化

M7-B 在**不改变 M3 划分、不改变 B5 结构**的前提下，将每个 4×4 矩阵的中心 2×2（`(1,1),(1,2),(2,1),(2,2)`）从训练监督中完全移除，只用其余 12 点训练，并在留出的 4 点测试。

- 主实验：B5、B3、B1 三个模型在 11 个冻结 split 上重新训练；测试标签只在 checkpoint 选定后读取一次。
- 稀疏实验：50% 交错掩码、25% 四角掩码，S1/S2 seed42、S3 fold0。
- 边界实验：留出最高剂量 A 行与最高剂量 B 列（7 点），检验高剂量外推，不与中心插值混合排名。
- 训练预算：160 epoch、patience 30、cosine、batch 64；所有模型共享同一掩码。

B5 中心留出测试（held-out matrix-macro RMSE）：S1 0.0702、S2 0.0811、S3 0.2050。
与 M6-B 全 16 点结果（0.0650/0.0768/0.1937）相比，中心留出分别变差约 7.9%/5.7%/5.8%，说明模型具备有限但可测量的网格插值能力，S3 未见细胞仍是主要瓶颈。

同口径下，B3 GINE 为 0.0765/0.0898/0.2032，B1 T3 为 0.0988/0.1185/0.2126：B5 在 S1/S2 领先，S3 与 B3 基本持平（B3 低 0.0018）。
稀疏与边界实验出现排序反转：50% 观测时 B5/B3 接近，25% 四角时 B3 明显更好（S1 0.121 vs B5 0.157）；
高剂量边界外推 B3 为 0.121/0.123/0.207，显著优于 B5 的 0.234/0.234/0.264。
因此 FiLM/Fourier 的收益依赖观测密度和目标区域，不能宣称在所有插值/外推条件下全面优于 GINE。

完整协议、逐 split 配对差异、稀疏/边界结果、曲面案例与失败案例见
[`m7b_report.md`](m7b_report.md)；协议细节见
[`docs/decisions/m7b_protocol.md`](docs/decisions/m7b_protocol.md)。

```bash
.venv/bin/python scripts/18_run_m7b.py --group b1_center_all --force
.venv/bin/python scripts/18_run_m7b.py --group sparse_dev --force
.venv/bin/python scripts/18_run_m7b.py --group boundary_dev --force
.venv/bin/python scripts/19_aggregate_m7b.py
.venv/bin/python scripts/20_plot_m7b_surfaces.py
.venv/bin/python -m pytest -q tests
```

**M7-B 结论：以 `m7b_report.md` 的自动判定为准。** 不因测试结果修改 B5 结构或掩码；所有负结果保留。
