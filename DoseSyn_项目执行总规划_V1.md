# DoseSyn 第四章项目执行总规划 V1.0
**项目定位**：ConfSyn 组合级协同筛选 → DoseSyn 剂量级响应曲面预测  
**用途**：毕业论文研究实施方案 / AI Agent 工程任务书 / 实验验收清单  
**状态**：规划版；未完成的结果不能写成已完成结果  
**核心原则**：先验证原始数据与划分，再跑基线，最后实现完整模型；任何结论必须有可复现的实验支持。

---

## 0. 项目目标与边界

### 0.1 科学问题
- 第三章 ConfSyn：输入 `(drug_A, drug_B, cell)`，预测组合级协同分类/评分，侧重多模态置信度融合。
- 第四章 DoseSyn：输入 `(drug_A, drug_B, cell, dose_A, dose_B)`，预测连续组合响应 `X/X0`，重建二维剂量响应矩阵，利用单药曲线与药理一致性约束，进一步分析剂量相关协同。
- 最终桥接实验：**ConfSyn 候选组合排序 → DoseSyn 预测有效剂量范围与剂量级协同区域**。
- 不把“首次使用剂量”或“首次使用连续曲面”写成创新；需要靠对称性、边界一致性、可靠的数据划分和充分消融支撑创新性。

### 0.2 主研究交付物
1. 具备来源记录的 O'Neil 清洗数据和特征映射。
2. 不泄漏的固定数据划分，以及自动化校验程序。
3. 至少 3 种可重现的基线模型。
4. DoseSyn 完整模型和可解释的模块消融。
5. 主性能、剂量插值/外推、药理一致性、协同景观和桥接实验。
6. 论文第四章的图、表、方法描述与完整复现实验说明。

### 0.3 非目标
- 暂不做临床处方推荐、真实患者用药剂量建议、体内疗效推断。
- 暂不引入 3D 构象、扩散模型或额外巨型异构图。
- ALMANAC 是扩展/外部验证目标，不作为 O'Neil 主流程上线的前置条件。
- 不修改原始 ConfSyn 训练代码；可读取和导出其已有特征与预测，须记录来源。

---

## 1. 当前已获得的数据与需要确认的事实

### 1.1 已上传的三份 O'Neil 附件
将以下原始文件拷贝至 `data/raw/ONEIL/`，**保留原文件名及 SHA256，不修改原始字节**。

| 原文件名 | 实际内容 | 原始文件记录 | 主要字段 |
|---|---|---:|---|
| `15357163mct150843-sup-156849_1_supp_0_w2lh45.xlsx` | 单药响应 | 16,752 | `BatchID, cell_line, drug_name, Drug_concentration (µM), viability1–6, mu/muMax, X/X0` |
| `15357163mct150843-sup-156849_1_supp_1_g2lr29.xlsx` | 字段说明 | 13 行说明 | 单药/组合字段定义 |
| `15357163mct150843-sup-156849_1_supp_1_w2lrww.xls` | 双药响应 | 368,832 | `BatchID, cell_line, drugA_name, drugA Conc (µM), drugB_name, drugB Conc (µM), combination_name, viability1–4, mu/muMax, X/X0` |

**特殊兼容性要求**：最后一个文件的扩展名为 `.xls`，但文件内是 OOXML/ZIP 结构。Agent 应先依据文件签名检测格式；若需要，创建一个**工作副本**并改名 `.xlsx` 或采用能读取 OOXML 的读取器，**禁止覆盖原文件**。请勿仅凭扩展名决定使用旧版 XLS 读取引擎。

初步数量核对：368,832 / 16 = 23,052 个 4×4 矩阵**候选实例**；16,752 / 8 = 2,094 条 8 剂量单药曲线**候选实例**。这是由记录数推算，并非已经验证的唯一 `(药物对,细胞,批次)` 数。必须由 M1 的分组审计确认，并区分 Batch 3 重复验证实验。原论文的 583 个组合、39 个细胞系是研究设计背景，不应代替清洗后实测覆盖率。

### 1.2 目标变量定义——必须写入 README
官方图例：
- `mu/muMax`：相对于 DMSO 对照的生长速率。
- `X/X0 = EXP(LN(4) × (mu/muMax - 1))`：相对生长/分数响应指标；0 表示生长完全受抑，1 表示无抑制。
- 实际记录可 `X/X0 > 1`（反映相对对照的促生长或归一化波动）；**主训练标签保留原始值，不剪裁到 [0,1]，输出层不强制 Sigmoid**。
- 原始 `viability1–4/6` 是技术重复测量；`X/X0` 是归一化衍生列，二者不能混用为同一标签。
- `inhibition = 1 - X/X0` 可作为**派生研究量**，但允许负值；如协同计算算法要求范围约束，必须单独标记其转换规则、使用范围和敏感性分析，不能静默裁剪。

### 1.3 还需补齐的特征
1. 38 种左右药物的标准化 SMILES、稳定药物 ID、RDKit 分子图（来源与映射状态可追溯）。
2. O'Neil 细胞系到已有 ConfSyn/CCLE 表达矩阵的映射（来源细胞名、标准名、DepMap ID、表达覆盖状态）。
3. Hallmark pathway gene sets 和从表达谱产生的通路特征（训练集相关归一化参数必须在训练折拟合）。
4. ConfSyn 在共同药物-细胞集合上的**样本外预测**，供最后桥接实验使用；不允许直接把训练集上的 ConfSyn 预测当测试性能。

### 1.4 数据来源参考
- O'Neil 原始论文：https://doi.org/10.1158/1535-7163.MCT-15-0843
- 已有 ConfSyn 公开仓库：https://github.com/Ahhyp/Consyn
- O'Neil 官方补充数据见论文 Supplementary Data 栏目。任何未来补充数据均保存链接、下载日期、版本/许可证及 SHA256。

---

## 2. 目标代码仓库结构（Agent 依此创建）

```text
DoseSyn/
├── README.md
├── AGENTS.md
├── pyproject.toml                 # 或 requirements.txt，锁定实际版本
├── configs/
│   ├── data_oneil.yaml
│   ├── baseline_mlp.yaml
│   ├── baseline_gine.yaml
│   ├── dosesyn.yaml
│   └── experiments/
├── data/
│   ├── raw/ONEIL/                # 只读、原样保存，不纳入大型公开 Git 提交
│   ├── raw/ALMANAC/              # 后续阶段
│   ├── external/                 # ConfSyn/CCLE/Hallmark 等来源文件
│   ├── mappings/                 # 映射表，留存人工审核记录
│   ├── interim/                  # 中间格式
│   ├── processed/                # 可重建数据及特征
│   └── splits/                   # 固定分组、train/val/test manifest
├── src/dosesyn/
│   ├── data/{inspect,parse_oneil,normalize,make_matrices,qc}.py
│   ├── features/{drug,cell,pathway,dose}.py
│   ├── splits/{build,verify}.py
│   ├── models/{mlp,gine,mono,dosesyn}.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils/{seed,logging,io}.py
├── scripts/
│   ├── 00_audit_raw.py
│   ├── 01_prepare_oneil.py
│   ├── 02_build_mappings.py
│   ├── 03_generate_splits.py
│   ├── 04_train_baseline.py
│   ├── 05_train_dosesyn.py
│   ├── 06_evaluate.py
│   ├── 07_reconstruct_synergy.py
│   └── 08_confsyn_bridge.py
├── tests/
│   ├── test_raw_schema.py
│   ├── test_data_transform.py
│   ├── test_matrix_identity.py
│   ├── test_split_leakage.py
│   ├── test_dose_zero.py
│   ├── test_swap_invariance.py
│   └── test_train_smoke.py
├── outputs/{audit,runs,metrics,figures,tables,bridge}/
└── docs/{data_dictionary,decisions,experiment_log,chapter4}/
```

`data/raw` 和大体量 `processed` 默认排除 Git 提交，保留 `data/README.md`、下载说明和数据清单；报告、配置、切分 manifest、代码、少量匿名化单元测试样本可以版本控制。

---

## 3. 里程碑与 Agent 工作包（严格按依赖顺序）

### M0｜工程初始化与文件来源审计（P0）
**任务**：创建独立 DoseSyn 项目、环境清单、配置读取、日志框架、随机数种子管理、原始文件哈希和格式探测程序。

**输入**：3 个 O'Neil 文件；ConfSyn 仓库地址。  
**产物**：
- `outputs/audit/raw_file_manifest.json`（文件路径、大小、SHA256、检测格式）
- `outputs/audit/raw_schema.md`（每列名称、解释、示例值、缺失定义）
- 可执行 `python scripts/00_audit_raw.py --config configs/data_oneil.yaml`
- `README.md` 环境搭建说明与自动化测试入口。

**验收**：全部文件可被读取；双药文件格式特殊情况有清晰日志；原文件哈希保持不变；没有把 `.xls` 强行交给旧版 XLS 引擎。  
**禁止**：训练任何神经网络，改写原始实验文件。

### M1｜O'Neil 数据解析、矩阵构建、质量审计（P0）
**任务分解**：
1. 逐列解析单药、双药并转换为规范列名；保存所有原始响应列与源行号。
2. 统一浓度：`dose_um` → `dose_molar = dose_um × 1e-6`；正剂量 `log10_dose_molar = log10(dose_molar)`，0 剂量标记为零剂量，不计算 `log10(0)`。
3. 识别 `(BatchID, cell_line, combination_name, drugA, drugB, dose_A 网格, dose_B 网格)` 构成的**实际实验矩阵**；允许相同药物对/细胞存在不同批次的复测。
4. 核对每张矩阵的 16 个网格位置、四个 A 剂量、四个 B 剂量；遇到缺行/重复坐标先列入异常表，**不要静默丢弃**。
5. 原始 Drug A/B 方向保留，同时构造规范化**无序药物对**（稳定 ID 排序）供数据切分；剂量及矩阵轴必须随药物顺序一起同步交换。
6. 为单药曲线建立 `mono_curve_id`，按实际批次/细胞/药物/浓度集合辨识 8 点曲线；单药浓度可不等于双药矩阵浓度，不直接按浮点数相等合并。
7. 每一行保留 `source_file, source_sheet, source_row, batch_id`；所有派生对象都能追溯到原始数据。

**规范化组合表 `oneil_combo_points.parquet`（建议列）**：
`study, source_row, batch_id, matrix_id, condition_id, drug_a_raw, drug_b_raw, drug_a_id, drug_b_id, unordered_pair_id, cell_raw, cell_id, dose_a_um, dose_b_um, dose_a_molar, dose_b_molar, log10_dose_a, log10_dose_b, viability1, viability2, viability3, viability4, mu_over_mu_max, x_over_x0, row_idx, col_idx, qc_status`

**规范化单药表 `oneil_monotherapy.parquet`（建议列）**：
`study, source_row, batch_id, mono_curve_id, drug_raw, drug_id, cell_raw, cell_id, dose_um, dose_molar, log10_dose, viability1–6, mu_over_mu_max, x_over_x0, qc_status`

**还需输出**：
- `oneil_matrix_metadata.parquet`
- `oneil_combo_replicates.parquet`（技术重复长表；不要把 viability1–4 误认为 4 张独立剂量矩阵）
- `outputs/audit/qc_oneil.json`、`outputs/audit/qc_oneil.md`
- 异常清单：重复剂量坐标、非正浓度、无响应、响应异常值、不同批次重叠、缺失技术重复。

**自动验收**：
- 处理前后主响应记录数必须能逐条对账，丢弃条目及原因必须计数；不能凭空改变 368,832 和 16,752 的基准。
- `matrix_id + row_idx + col_idx` 在“每个实验矩阵”粒度上唯一，否则必须明确定义第二级复测 ID。
- 所有被标记 `complete_4x4` 的矩阵恰有 16 个**唯一**坐标；不完整矩阵有明确数量。
- 值超过 1 的 `X/X0` 保留下来且审计报告记录。
- 产生两张典型样本真实/规范化对照矩阵（只作检查，不作为训练数据）。
- 对 QC 结果不满意时不得进入 M2。

### M2｜药物、细胞、通路的可复用特征（P0）
**药物**：
- 从 ConfSyn 现有药物 SMILES 表提取可对应药物；其余从可信数据库补齐。
- `drug_mapping.csv` 必须有 `source_name, canonical_name, drug_id, canonical_smiles, inchikey, mapping_method, review_status, source_url`。
- 使用 RDKit 检查 SMILES 可解析、分子图构造是否成功；盐形式/立体信息变化写入映射日志。
- 不允许 Agent 在不确定映射下按字符串“猜一个最像的药”；设 `NEEDS_REVIEW` 等人工确认。

**细胞**：
- 构建 `cell_mapping.csv`：`oneil_name, depmap_id, canonical_name, lineage, expression_available, review_status`。
- 对接当前 ConfSyn 对应的 CCLE 表达文件，按稳定细胞 ID 获取基因表达；不可用细胞单列审计。
- 使用 Hallmark 基因集得到通路向量。优先采用可重现的简单方法（例如规范化 gene-set 聚合）；升级为 ssGSEA/GSVA 必须写明实现与版本。
- 对表达和通路特征的缺失值处理、标准化，仅从当前训练折学习统计量（验证/测试只能 transform）。

**产物**：`drug_features.parquet`/缓存分子图、`cell_features.parquet`、`pathway_features.parquet`、两张映射表、映射覆盖率报告。  
**验收**：逐个列出缺失实体；主训练集仅包含有可用特征的样本，删除比例有报告；禁止混淆细胞系名称和药物异名。

### M3｜无泄漏数据划分与固定实验协议（P0）
**共同原则**：先按实体/条件**分组**后划分，再扩展到每个 dose point。**禁止对 368,832 条剂量点普通随机划分**。一个 `(无序药物对, 细胞系)` 对应的全部剂量、批次复测必须同属一个主划分，除专门研究“剂量点补全”的实验。

| ID | 实验设置 | 固定规则 | 研究意图 |
|---|---|---|---|
| S1 | Random-condition | 按 `condition_id=(unordered_pair_id, cell_id)` 分组随机划分，例如 70/15/15；同一条件的不同批次一起分配 | 主性能 |
| S2 | Unseen-pair | 按无序药物对分组；测试药物对在训练/验证不出现 | 未见药物对 |
| S3 | Unseen-cell | 按细胞系分组；测试细胞在训练/验证不出现 | 未见细胞 |
| S4 | Dose interpolation | 在指定且彼此隔离的矩阵内隐藏**中间**剂量坐标；训练同一矩阵其他坐标；明确是补全评测、不能称为新组合泛化 | 连续插值 |
| S5 | Dose extrapolation | 训练较低/较窄范围剂量，测试更高/范围外剂量；明确训练剂量覆盖 | 外推 |
| S6 | Cross-study | O'Neil 主流程稳定后再引入 ALMANAC，并做名称/标签/实验协议可比性审计 | 外部验证，条件允许 |
| S7 | Strict unseen-drug | 可选增强：测试药物实体在训练/验证的组合**和监督单药曲线**均不出现 | 强归纳测试 |

**特别关注监督泄漏**：
- 在 S3 中，如果用测试细胞的带标签单药曲线训练 monotherapy head，必须明确称为“有单药先验的未知细胞场景”，不能冒充完全未见细胞；严格版须同时隔离这些单药标签。
- 在 S7 中同理隔离未见药物的单药监督；纯 SMILES/外部无标签表达数据可以使用，但需记录。
- 禁止测试集统计量用于训练归一化、特征选择、调参、early stopping。
- 同一条件的 Batch 3 复测必须随主批次进入同一组，不得跨 train/test。

**产物**：每个协议的 `train.parquet / val.parquet / test.parquet` 或条件 ID manifest、划分统计、`verify_splits.py`。  
**验收**：泄漏单元测试全部通过；至少 3 个固定随机种子（如 13/42/2026）的划分可重建；测试集只评一次最终选定模型。

### M4｜最小训练链路与基线（P0）
先训练简单模型，**不允许直接跳到完整 DoseSyn**：
1. **B0 Mean/curve prior**：训练数据的合理平均/单药效应先验，只作为参考下界。
2. **B1 RF/XGBoost-Dose**：Morgan 指纹 + 细胞表达/通路 + `log10(dose_A/B)`；显式处理双药顺序。
3. **B2 MLP-Dose**：同类输入，纯拼接回归；固定标准化、优化器、验证集 early stopping。
4. **B3 GINE-Base**：共享 GINE 药物编码 + pathway cell encoder + 普通剂量 MLP + 回归头；不加药理约束。

**统一训练行为**：
- 优先回归 `X/X0`，采用 Huber 或 MSE（在验证集选择）；输出不裁剪。
- 保存 `config.yaml, seed, git_commit, dataset_hash, split_id, best.ckpt, metrics.json, predictions.parquet`。
- 报告点级 RMSE、MAE、Pearson、R²；同时报告按矩阵宏平均的误差（每张矩阵等权），不能仅用所有点混合计算。
- 一个 smoke test：1 个 batch 前向/反向、loss 有限、100 个小样本可过拟合、固定 seed 可重跑。

**验收**：B2、B3 在 S1 固定划分跑通，训练和评估入口有效；没有因误设输出 Sigmoid 而截断 `X/X0 > 1` 的标签。先记录真实指标，**不虚构提升**。

### M5｜DoseSyn 完整模型（P0）
设计尽量集中于“剂量条件响应 + 药理一致性”，不堆叠无关模块：

**模块 A：共享药物图编码器**
\[
h_A=f_\mathrm{GINE}(G_A),\quad h_B=f_\mathrm{GINE}(G_B)
\]
同一个 encoder 处理两种药物。

**模块 B：细胞通路上下文**
\[
P_c=f_\mathrm{pathway}(x_c),\quad h_c=f_\mathrm{cell}(P_c)
\]
可以采用通路 token/轻量注意力或简洁 MLP；先保证稳定性和可追溯性。

**模块 C：连续剂量编码**
正剂量先转 mol/L 的 `log10`，再使用 Fourier/RBF：
\[
e_A=\phi(\log_{10}d_A),\quad e_B=\phi(\log_{10}d_B)
\]
注意 `dose=0` 是独立特殊输入，使用 `(zero_flag, encoded_positive_dose)`，绝不执行 `log10(0)`。

**模块 D：药物-细胞-剂量条件交互**
对两药使用共享调制器：`u_A=FiLM(h_A,h_c,e_A)`、`u_B=FiLM(h_B,h_c,e_B)`。
用交换不变算子（例如 `[u_A+u_B, u_A*u_B, abs(u_A-u_B)]`）生成组合表示，注意在交换药物时同时交换其剂量。

**模块 E：连续响应曲面解码器**
\[
\hat v_{AB}=F_\theta(G_A,G_B,c,d_A,d_B)
\]
对任意所选连续正剂量坐标生成响应；不保证超出已测剂量域的外推准确性。

**模块 F：单药头与边界条件**
共享单药头输出 `m_A(d_A,c)`、`m_B(d_B,c)`。
采用先易后难的两级方案：
- V1（必须）：辅助单药监督 + `F(d_A,0)≈m_A(d_A)`、`F(0,d_B)≈m_B(d_B)` 的边界损失。
- V2（仅在 V1 结果支持时）：构造 `F = m_A*m_B + g(d_A,d_B)*interaction_correction`，其中 `g` 在任一剂量为零时恰为零，以**结构性保证**单药边界；需要检验协同参考模型和输出范围，不预设一定更优。

**建议主损失**：
\[
L=L_\mathrm{combo}+\lambda_mL_\mathrm{mono}
+\lambda_bL_\mathrm{boundary}+\lambda_sL_\mathrm{smooth}
\]
- `L_combo`：实测矩阵点的 Huber。
- `L_mono`：实测单药 X/X0 的 Huber。
- `L_boundary`：以模型产生的**零剂量虚拟坐标**检查/约束边界，不能把虚拟零剂量混入原始“实测标签”。
- `L_smooth`：在训练剂量域内，对临近网格或随机采样的剂量点做适度平滑（优先 log-dose 坐标）；小权重，并通过消融检查过平滑。
- 不把强制剂量单调性作为默认假设；如添加则做可选敏感性实验。

**验收**：
- 正向 / 反向传播正常、无 NaN。
- 药物和相应剂量互换后结果在数值容差内一致；若不一致，停止并修复实现。
- 零剂量边界无 log(0) 错误；测试单药头与联合头边界误差。
- 无法在 S1 上稳定优于 B3 时，先定位模块问题并消融，禁止通过观察测试集反复调参。

### M6｜完整实验矩阵（P0/P1）
**E1 主结果（P0）**：S1 上 B0/B1/B2/B3/DoseSyn 比较，3 个随机种子，报告均值±标准差；同样条件下使用相同数据、输入、评估口径。

**E2 泛化（P0）**：S2 Unseen-pair、S3 Unseen-cell；确认严格/有单药先验两种情形的标签可用性后，明确命名，避免误导。

**E3 消融（P0）**：
`Full / w/o pathway / raw-dose instead Fourier / concat instead symmetric / w/o mono / w/o boundary / w/o smoothness / w/o dose-conditioned FiLM`；
控制其他配置相同。

**E4 剂量插值与外推（P0）**：S4/S5；报告不同隐去剂量比例、剂量跨度、单药剂量是否覆盖；分别展示插值和外推而不是合成一个分数。

**E5 药理一致性（P0）**：
- `swap_error`（交换误差）
- `boundary_error`（单药边界误差）
- `surface_roughness`（统一网格上的二阶差分）
- 主回归误差，避免只追求“光滑”而降低真实性。

**E6 剂量响应曲面（P0）**：选 3–5 个测试矩阵展示 Ground Truth / Prediction / Absolute Error，保持统一色标，明确测量点与插值网格的区别。

**E7 协同景观重建（P1）**：
- 使用与实测组合相配套的单药预测/曲线得到 Bliss/HSA 等指标。
- 若以**存活/相对生长分数** \(v\) 表示，Bliss 参考 \(v_\mathrm{Bliss}=v_Av_B\)，可定义正协同 \(S_\mathrm{Bliss}=v_Av_B-v_{AB}\)；HSA 在相同响应方向下可写 `min(v_A,v_B)-v_AB`。**必须统一正负号、量纲和规范化规则**。
- Loewe/ZIP 不应通过任意公式“硬凑”；需调用经校验的规范方法，确认必要的完整单药曲线、剂量范围与可辨识性。信息不足则只报告可信的 Bliss/HSA，Loewe/ZIP 列为扩展，不伪造结果。
- 协同景观应与相同协议计算的真实协同结果比较，不能把不同工具定义混用。

**E8 复杂度与稳定性（P1）**：参数量、训练耗时、推理耗时、峰值显存、seed 稳定性；按同一硬件和计量单位。

**E9 批次/重复噪声（P1）**：原始技术重复缺失分布，响应重复方差与预测误差关联；Batch 3 外部重复测试必须确保条件泄漏被控制，不能冒充独立未知组合验证。

**E10 跨数据集（P2）**：ALMANAC / DrugComb。前置步骤：确认响应量纲及控制定义一致、剂量单位可转、药物和细胞映射可用；如果不满足则记录为“不能直接比较”，不能报貌似精确的跨研究 AUC/RMSE。

**报告统计规范**：优先按 `condition_id`/矩阵进行聚类 bootstrap 置信区间或条件粒度检验，不把共享同一实验条件的 16 个剂量点当成 16 个独立受试对象。明确指出单药数据是否参与监督训练。

### M7｜ConfSyn → DoseSyn 桥接（P1）
**桥接问题**：第三章筛选出的高置信组合是否在第四章被定位到明确的剂量响应与可能协同区域？

**步骤**：
1. 核对两个项目的药物 ID、无序药物对、细胞系 ID 和单位映射。
2. 选择**双方均有观测、且 ConfSyn 预测为样本外**的共同测试条件；必要时重新训练 ConfSyn 以产生与桥接划分一致的 OOF/OOS 预测。
3. 导出 ConfSyn 概率、标签/参考协同分数及 DoseSyn 预测矩阵，不允许在模型选择时使用桥接测试标签。
4. 比较 ConfSyn 按组合层排序与 DoseSyn 按区域级效应/Bliss/HSA 归纳的排序，量化相关性和互补性。
5. 选 2–3 种典型案例：整体一致、仅局部剂量协同、组合级信号一般但局部剂量响应强。
6. 只得出“**体外实验数据范围内的计算结果**”，不得宣称安全剂量、最优临床联合方案或真实药效验证。

**产物**：`outputs/bridge/bridge_predictions.parquet`、案例图、统计表、实验说明。  
**验收**：每个例子的 ConfSyn/DoseSyn 数据划分及样本外状态可追踪，案例不是依据测试结果反向选择用于模型调参。

### M8｜论文图表、写作与复现实验打包（P0）
**建议图**：
- 图4-1 总体模型架构
- 图4-2 O'Neil 原始数据与清洗流程
- 图4-3 S1–S5 划分示意
- 图4-4 回归预测散点和残差
- 图4-5 剂量编码/损失权重敏感性
- 图4-6 真值/预测/误差矩阵
- 图4-7 剂量插值/外推
- 图4-8 单药边界与交换一致性
- 图4-9 真实/预测协同景观
- 图4-10 ConfSyn → DoseSyn 桥接案例

**建议表**：
- 表4-1 数据集统计与特征覆盖
- 表4-2 实验配置与训练超参数
- 表4-3 主基线比较
- 表4-4 泛化实验
- 表4-5 消融实验
- 表4-6 剂量插值与外推
- 表4-7 药理一致性
- 表4-8 协同重建
- 表4-9 跨数据集（若可行）
- 表4-10 时间/参数/显存和桥接汇总

**建议论文目录**：
- 4.1 问题定义与 DoseSyn 总体框架
- 4.2 分子图与细胞通路表征
- 4.3 连续剂量条件交互与对称融合
- 4.4 响应曲面预测、单药辅助监督与一致性约束
- 4.5 数据集、划分、基线与评价指标
- 4.6 主对比、泛化与消融
- 4.7 插值/外推、药理一致性和协同景观
- 4.8 ConfSyn 桥接与典型案例
- 4.9 本章小结
最终章节编号可按学校模板调整。

**最终交付**：
- 完整 `configs/` 与运行命令
- `requirements.lock`/环境信息 + git commit + 数据哈希
- 各实验指标 CSV/JSON、预测 Parquet、必要 checkpoint
- 所有论文图表的自动生成脚本
- 运行复现实验的 `README_REPRODUCE.md`
- `docs/chapter4/` 中论文正文草稿及图表说明

---

## 4. 关键训练细则（强制标准）

1. **所有数据处理可重建**：从 raw 到 processed 只通过脚本，绝不手动修改原始 Excel。
2. **组标识是核心**：矩阵/条件/批次/技术重复必须是不同级别的 ID；不以行号当药物对 ID。
3. **固定 split manifest**：先生成切分，再训练；同一表格里的全部模型用同一分组划分。
4. **训练折拟合变换**：标准化、缺失值填补、基因特征选择均 train-fit/val-test-transform。
5. **metric 要清楚**：
   - `point_RMSE`：所有剂量点的总体 RMSE。
   - `matrix_macro_RMSE`：每张实验矩阵先算 RMSE 后等权平均。
   - `condition_macro_RMSE`：对同一 pair-cell 条件的批次复测先汇总再等权平均。
   - `Pearson`、`Spearman`、`R²` 需说明按点还是按条件统计；常数向量导致 Pearson 未定义时不应强行填零。
6. **超参数只看验证集**；测试集只用于最终报告。需要开发调试时使用单独 smoke 子集。
7. **随机种子**固定数据切分、NumPy、PyTorch、dataloader 等；记录仍存在的非确定性。
8. **保存逐点预测**：`study, condition_id, matrix_id, doseA, doseB, y_true, y_pred, split, seed, model`，方便任何审计与图表重绘。
9. **代码先有测试，再大规模跑**：每新增一个模块，先单测 shape、梯度、NaN、对称性与零剂量。
10. **性能负结果也记录**：每次实验写清“假设—配置—指标—观察—下一步”，禁止只保留最好一次。

---

## 5. 各阶段的停止条件与替代方案

| 风险 | 判断条件 | 处置 |
|---|---|---|
| O'Neil 分组异常 | 无法重建可靠 4×4 实验矩阵 | 停在 M1，生成异常样本和定位报告，不继续训练 |
| 药物 SMILES 映射不充分 | 主样本大量缺药物结构 | 先用 Morgan/已有 ChemBERTa 做 B2，不凭空补 SMILES |
| CCLE 匹配不充分 | 主样本大量缺细胞表达 | 先量化覆盖率，明确有限子集；可考虑精简细胞特征，不冒充全数据结果 |
| Monotherapy 对不上浓度 | 单药曲线与组合浓度不同 | 用可训练连续单药头/曲线内插；不得将“最近邻剂量测量”冒充相同剂量真实值 |
| Dense 模型未提升 | Full DoseSyn 未稳定优于 B3 | 优先消融剂量条件/正则权重，缩减结构而非继续堆模块 |
| 超出剂量域预测发散 | S5 大误差明显 | 如实报告外推局限；限制论文的有效预测区域 |
| Loewe/ZIP 信息不足 | 曲线拟合失败、参考指标不稳定 | 只报告标准化后的可信指标并说明限制 |
| Cross-study 不同口径 | 无法统一响应或样本覆盖率不足 | 将 ALMANAC 改为独立实验附录/未来工作，不影响主章完成 |

---

## 6. 项目管理与建议顺序

### 优先级定义
- **P0（毕业论文最小闭环）**：M0–M5、E1–E6、M8；必须完成。
- **P1（增强论文说服力）**：E7–E9、M7；应力争完成。
- **P2（数据风险较高的扩展）**：ALMANAC 大规模跨研究、严格双药未知、额外公开模型复现；前置稳定后再做。

### 交付节奏（建议，不是 Agent 自动后台执行）
- 阶段一：M0 → M1（原始数据和 QC）
- 阶段二：M2 → M3（特征与防泄漏 split）
- 阶段三：M4（基线 + 训练工具链）
- 阶段四：M5（DoseSyn 模型开发）
- 阶段五：M6（完整实验及误差分析）
- 阶段六：M7 → M8（桥接与论文成稿）

**每阶段结束必须提交**：
1. 新增或修改的文件清单；
2. 复现实验命令与运行环境；
3. 自动测试报告；
4. 核心产物绝对/相对路径；
5. QC/实验摘要；
6. 未解决问题与下一阶段建议；
7. 是否满足阶段验收（YES/NO）。

### 推荐首轮 Agent 请求
> 现在只执行 M0 与 M1，禁止训练任何模型。读取我提供的三个 O'Neil 附件，先用文件签名确认 OOXML 格式并计算 SHA256，再检查真实字段、行数、缺失、药物和细胞数量以及 4×4 矩阵分组逻辑。生成规范化的双药点表、单药点表、矩阵元数据、技术重复长表，以及 QC Markdown/JSON 报告；对任何映射不确定或分组失败的记录留下异常文件。不要覆盖原始数据，不要平均技术重复，不要裁剪 `X/X0`，不要随机拆分单个剂量点。写自动化测试并运行；只有全部验收通过才建议进入 M2。每次报告附上实际命令、真实输出数值、文件路径和测试结果。不得伪造结果。

---

## 7. 可复现命令契约（需由 Agent 实现）

建议最终支持以下命令（**此处是接口规范，不表示现在已经存在对应脚本**）：

```bash
python scripts/00_audit_raw.py --config configs/data_oneil.yaml
python scripts/01_prepare_oneil.py --config configs/data_oneil.yaml
python scripts/02_build_mappings.py --config configs/data_oneil.yaml
python scripts/03_generate_splits.py --config configs/data_oneil.yaml --scenario S1 --seed 42
python scripts/04_train_baseline.py --config configs/baseline_mlp.yaml --scenario S1 --seed 42
python scripts/05_train_dosesyn.py --config configs/dosesyn.yaml --scenario S1 --seed 42
python scripts/06_evaluate.py --checkpoint outputs/runs/<run_id>/best.ckpt --split test
python scripts/07_reconstruct_synergy.py --predictions outputs/runs/<run_id>/predictions.parquet
python scripts/08_confsyn_bridge.py --config configs/experiments/bridge.yaml
python -m pytest tests/ -q
```

CLI 的实际参数名可在第一阶段确定，确定后应保持向后兼容，并同步更新 `README_REPRODUCE.md`。

---

## 8. 项目验收总清单

- [ ] O'Neil 原始三个附件有来源、哈希、字段字典
- [ ] 原始单药/组合记录数能对账
- [ ] 正确识别 4×4 矩阵、Batch 3 复测与技术重复
- [ ] 主标签 `X/X0` 定义和超出 1 的值被保留
- [ ] SMILES / CCLE / Hallmark 映射覆盖率有报告
- [ ] S1–S3 分组划分通过泄漏测试
- [ ] 数据统计与训练/验证归一化规则可复现
- [ ] B2 与 B3 跑通并得到真实基线指标
- [ ] DoseSyn 模型无 NaN，满足交换一致性和零剂量测试
- [ ] 主实验包含多个 seed 与矩阵级指标
- [ ] 消融逐项支撑/否定预设模块作用
- [ ] 插值、外推单独评估
- [ ] Bliss/HSA 等协同重建指标的定义与正负方向统一
- [ ] ConfSyn 桥接使用有效的样本外预测
- [ ] 所有图表可由 CSV/Parquet 一键再生成
- [ ] 正文不声称未经验证的“首次”“显著提升”或临床有效性

**执行判据：先保证数据与实验真实，再追求模型性能与论文体量。**
