# DoseSyn M5 完整模型开发执行任务书 V1.0

**项目**：DoseSyn（毕业论文第四章）  
**阶段**：M5 — 完整模型开发、模块验证和小规模验证集试验  
**前置基线**：M4 `632ba0ce0105b4e1ba560c2f4bbfe83c122d57c3`  
**仓库**：https://github.com/Ahhyp/DoseSyn  
**服务器工程目录**：`/home/kouzheng_2/AH_YeP/DoseSyn`  
**阶段边界**：M5 完成后停止；**不自行进入 M6、M7、M8**。

> 本文是给 AI Agent 的工程执行规范。任何指标必须来自真实运行，不得填入目标值冒充结果。保持 M0–M4 的可追溯性和可复现性。

## 0. M4 验收与进入 M5 的理由

M4 书面报告：44/44 run 成功，131 passed、0 failed，11/11 split 泄漏检查通过，2,272,576 行测试预测；以下数字均来自 M4 报告，非本文独立复算。

| 模型 | S1 RMSE | S2 RMSE | S3 RMSE | 场景等权平均 |
|---|---:|---:|---:|---:|
| B0 Mean | 0.3060 | 0.3022 | 0.3066 | 0.3049 |
| B1 Tree-Dose | 0.2467 | 0.2497 | 0.2819 | 0.2594 |
| B2 MLP-Dose | **0.0764** | 0.1022 | 0.2033 | 0.1273 |
| B3 GINE-Base | 0.0795 | **0.0897** | **0.1994** | **0.1228** |

**重要纠错**：M4 结论中的“S1、S2 上 B2 略优”与表格相矛盾。正确口径为：S1 B2 优于 B3；S2、S3 B3 优于 B2。若修改 M4 报告，采用单独的文档勘误提交，不改训练与预测产物。

**研究动机**：
- M4 已有很强的 B2/B3 基线，M5 不应仅做更大网络或 GINE 超参数调整。
- 未见细胞系 S3 误差显著高于 S1；该问题属于第四章的泛化验证场景，但第四章创新主线仍是**剂量条件作用机制 + 单药边界一致性 + 连续曲面结构**。
- B2/B3 模型性能差距小，必须用受控消融识别结构收益，不能预设 FiLM、GINE、平滑正则必然提高 RMSE。

## 1. 强制约束（不可违反）

1. **M0–M3 全部冻结**：不修改原始 Excel、Primary-v1 选样、drug/cell 特征、任何 split、mono policy、train-only scaler 和对应冻结清单。复核 `m3_manifest.json` SHA256 前缀 `7974091b4d1b1fb9`。
2. **M4 产物不覆盖**：保留所有 B0–B3 训练代码、最佳 checkpoint、报告、预测与指标；新模型单独写到 M5 目录。
3. **标签原义不变**：主回归目标为原始 `X/X0`，是相对生长响应，不直接等同于通常定义的细胞存活概率或 inhibition；允许大于 1，不裁剪真实标签，默认不强制裁剪预测。
4. **拆分仍以矩阵/condition/pair/cell 为单位**：不得点级随机拆分或让重复批次跨 split；验证只用 M3 冻结的 val，最终测试只用冻结的 test。
5. **单药访问协议严格一致**：S1/S2 为 `known_monotherapy`（已有的所有允许标签作为已知先验，需如实报告）；S3 为 `strict_cell_isolation`（仅训练细胞曲线可用于监督，val/test 细胞单药标签禁止读取、拟合、监控、调参或评估）。
6. **只在训练集拟合任何新转换**：新标准化、阈值、重采样分布等不得接触 val/test 标签；可复用已有训练折 scaler。
7. **测试集不能成为模型选择集**：M5 的开发和模块比较仅看验证集；在结构/权重/超参数冻结以前不进行反复 test 评估。M5 允许 S1 seed42 与 S3 fold0 的验证集试运行，以覆盖普通数据场景和严格单药隔离场景；M6 才运行 11 个正式拆分。
8. **环境优先稳定性**：在现有 `.venv` 内开发；不得为了画图或依赖升级破坏 numpy/RDKit/PyTorch 环境，记录所有环境变化。图可以用无依赖 SVG。
9. **不得虚构协同指标**：没有经过科学验证的 `X/X0` → effect/viability 转换及药理参考模型实现之前，不得将 `mono1 × mono2` 等表达式直接标成“标准 Bliss/ZIP/Loewe 预测”。
10. **每项结果可溯源**：记录代码 commit、配置 SHA256、M3 数据清单哈希、种子、训练日志、最优 epoch、资源、失败作业、GPU 等信息。

## 2. M5 研究假设与模型接口

**研究假设 H1**：药物作用表示应同时依赖药物、细胞背景和**连续剂量**，而非仅将剂量标量附加在最终回归层。

**研究假设 H2**：组合响应函数的药物交换对称性应由网络结构保证：
`F(d1, x1, d2, x2, c) = F(d2, x2, d1, x1, c)`。

**研究假设 H3**：利用合法的单药监督和零剂量虚拟边界，可以让组合曲面的端点与单药预测一致；局部平滑可以抑制不合理振荡，但效果须验证。

**统一接口**：

- 输入：`(drug_graph_1, drug_graph_2, cell_features_50d, dose_1, dose_2, zero_mask_1, zero_mask_2)`；
- 组合输出：每矩阵 4×4 共 16 个 `X/X0` 预测，另支持任意连续、非负的剂量查询；
- 单药输出：共享单药头 `M(drug_graph, cell_features, dose, zero_mask)`；
- 单位：底层沿用 M3 冻结协议的 µM、log10(M) 和训练折专属 scaler，模型不能混用原始单位与规范化单位；
- 对于零剂量：显式 zero flag / sentinel 分支，**禁止 log10(0)**，且 sentinel 不应与任何真实非零剂量混淆。

### 2.1 共享药物与细胞编码器

复用 B3 的 GINE + 50 维 Hallmark 表征作为**可比起点**。两药共享同一个药物编码器；细胞通路编码器从已冻结的 M3 50 维特征开始，暂不扩展外部生物学数据。

`h_i = GINE(G_i)`；`h_c = CellEncoder(p_c)`。

在首次复现对照时维持 B3 的关键训练预算和基础隐藏维度，新增参数与运行耗时单独统计。

### 2.2 连续剂量编码器

对真实正剂量先按照 M3 的 scaler 得到标准化 log-dose，然后比较下述编码：
- 原基线标量编码（保留 B3 对照）；
- `log-dose + MLP`；
- 固定频率 Fourier/RBF 特征再经 MLP（M5 可先实现 Fourier；完整对照留到 M6）。

`e_i = DoseEncoder(z_i, is_zero_i)`。

**注意**：傅里叶周期尺度和输入标准化需要合理初始化；不能默认越高频越好。每种编码的 feature dimension、参数量和前向耗时都要可记录。

### 2.3 剂量条件 Drug–Cell Interaction（FiLM）

定义每个药物的共享交互函数：  
`u_i = DrugCellInteract(h_i, h_c)`，  
`(gamma_i, beta_i) = MLP_dose(e_i)`，  
`t_i = (1 + gamma_i) ⊙ u_i + beta_i`。

采用 `(1 + gamma)` 或等效的近似恒等初始化，以减少初始优化的不稳定性。不得为 drug A 与 drug B 分别引入不共享参数的网络。

### 2.4 结构性保证药物交换对称性

组合表征（例）：
`p = concat(t1 + t2, t1 ⊙ t2, abs(t1 - t2), h_c)`。

`F = SurfaceDecoder(p)`。

**核心实现要求**：用于预测当前点的所有剂量信息必须已被注入 `t1/t2`，或通过同样严格对称的路径进入 Decoder。若解码器再拼接有序的 `e1,e2`，对称性就被破坏，不允许这样实现却仍宣称结构严格对称。

对于输入药物与剂量一起交换，预测应在浮点误差范围内相等。若一次输入整张 4×4 网格，交换药物并转置网格后，输出应与原网格的转置对应。

### 2.5 组合连续响应曲面解码器

选择轻量 residual MLP 或相似的**条件坐标隐式函数**，实现按任意剂量点前向查询；不同矩阵共享模型参数。矩阵采样要保持每张矩阵独立，避免在不同 condition 间施加平滑损失。

可选固定矩阵 context 缓存用于减少重复 GINE 前向。不要依赖固定 4×4 输出层作为唯一接口（M6 需要 unseen dose 插值测试）。

### 2.6 Monotherapy Head

实现共享单药函数：
`M(d, x, c) = MonoHead(GINE(G_d), CellEncoder(c), DoseEncoder(x))`。

单药监督从 M3 的 `mono_assignments.parquet` 和 `label_access/trainable` 列加载：
- S1/S2 只能加载协议明确允许的 `known_prior/trainable`；
- S3 只允许 train-cell mono；
- 单药头不能在验证/测试细胞的单药标签上做早停、损失、模型选择或“辅助评估”。

**优化策略**：先检查 mono-only 的可训练性与验证集以外的合法诊断，再联合训练。防止单药头与边界损失共同漂移，必须由真实的可训练单药曲线 `L_mono` 锚定。

### 2.7 零剂量边界一致性

对于给定 cell，构造**虚拟零剂量查询**：
`F(d1,x1,d2,0,c) ≈ M(d1,x1,c)`，
`F(d1,0,d2,x2,c) ≈ M(d2,x2,c)`。

约束：
`L_boundary = mean Huber(F(x1,0), M1(x1)) + mean Huber(F(0,x2), M2(x2))`。

这两个零剂量组合点是**虚拟边界查询**，不是 O'Neil 测量到的双药零剂量 ground truth。二者在文档与输出中必须明确区分。

切勿无依据地对 `F(0,0)` 直接添加固定“等于 1”的硬约束。若需要加入空白对照锚点，必须先证实该数据中的归一化实验定义和单位一致。

### 2.8 局部平滑（非强制单调）

`L_smooth` 优先采用相同 log-dose 间隔的内插网格上的二阶有限差分 / 曲率惩罚；若在原始非均匀网格计算，则必须按真实剂量间隔使用适当的非均匀有限差分系数。

仅对每个矩阵自身做差分；不得把其他矩阵的点拼成邻居。不要强制剂量响应单调下降，也不要把响应直接投影到 [0,1]，以免覆盖真实非单调现象和 `X/X0>1` 的数据。

### 2.9 总损失

建议先实现：

`L_total = L_combo + λ_mono L_mono + λ_boundary L_boundary + λ_smooth L_smooth`。

- `L_combo`: 对原始 `X/X0` 做每矩阵等权的 Huber；留出可切换 MSE 的配置。
- `L_mono`: 仅合法可训练的单药曲线，按曲线或药物-细胞条件归一化，避免某一批次样本数偏大导致权重异常。
- `L_boundary`: 按矩阵平均，虚拟零剂量边界与单药头预测一致。
- `L_smooth`: 轻量并带可调系数；先确保量级不会压倒真实标签损失。
- 结构严格对称时默认**不加**额外 symmetry loss；仍保留独立交换误差诊断 `E_swap`。

各项系数必须显式出现在配置和日志中；禁止隐藏固定权重。若单药、平滑损失与主目标明显冲突，需用验证集证据减弱或关闭，不可为凑“药理模块齐全”而牺牲科学可靠性。

## 3. M5 阶段实施顺序和停止点

### M5-A — 复用 B3、建立可切换模块的训练基础

**任务**：
1. 先阅读 M4 的 B3 源码、训练入口、数据加载及 `AGENTS.md`。
2. 新建 M5 config + module，并构建实验开关，不修改 M4 原有预测和最优 checkpoint。
3. 在相同 S1 seed42、验证集口径下，确认新接口的“全部新增模块关闭”能复现 B3 的输出或定位不可避免的实现差异；记录随机性和误差。
4. 先完成 CPU 小样本前后向、GPU smoke test 和小样本过拟合检查。

**交付**：新目录、配置、兼容验证记录、初始单测；本小节不做 11 个正式 split。

### M5-B — 对称融合 + 连续剂量 + FiLM

**任务**：
1. 建立 `use_symmetric_pair`、`dose_encoding`、`use_film` 等可配置开关。
2. 逐个验证 C1：对称融合；C2：连续 dose encoding；C3：FiLM 剂量条件交互。
3. S1 seed42 的训练只用于检查收敛及 val 性能；在接口稳定后再增加一次 S3 fold0 的 val 试运行，确认严格细胞隔离下能够正常训练。禁止参考 test 结果做修改。
4. 单独验证药物互换测试，输出 `E_swap` 最大值 / 平均值、16 格转置误差。

**交付**：Dosesyn-Core 能训练、预测连续坐标、通过对称性 property test。

### M5-C — 单药头、合法标签访问与边界一致性

**任务**：
1. 实现独立单药数据加载器及访问策略钩子。
2. 对每个 split 验证：允许加载的 mono curve 数量与 M3 policy 一致（S3 严禁 held-out 访问）。
3. 验证 `dose=0` 的前向、反向、混合 batch，无 NaN/Inf。
4. 构建 C4：加 mono 真实监督、C5：加 boundary；观察训练集损失与 val matrix-macro RMSE，不得混淆“增加外部监督”和“结构改进”。

**交付**：单药头能通过真实训练曲线的监督，边界一致性误差可记录。

### M5-D — 平滑正则、损失诊断与完整集成

**任务**：
1. 实现 C6：加 smoothness；报告不同正则项的量级和梯度范围。
2. 验证非均匀 dose 网格、局部内插点、数值梯度稳定性。
3. 用相同验证口径比较 C0–C6 及完整版（必要时保留开关，不要求 M5 做足所有正式消融）。
4. 导出每张矩阵 GT / predicted / absolute error 的 SVG 示例（仅展示观测点；连续查询曲面明确标注为模型插值，不冒充测量值）。

**交付**：集成模型完整，模块开关可供 M6 使用；S1 seed42 和 S3 fold0 验证集试验得到可复现 checkpoint。

### M5-E — 协同重建可行性审计（仅诊断，不冒险宣称完成）

**任务**：
1. 阅读 O'Neil 发布源数据对 `X/X0`、对照归一化、`mu` 定义与重复实验的说明，形成书面语义审计。
2. 明确 Bliss / HSA / Loewe / ZIP 各自需要的响应变量、单位、零剂量参考点、单药曲线拟合条件。
3. 判断能否得到严格合法的 response → effect 映射；不满足条件则输出 `NOT_VALIDATED`，并在 M6 作为待解决研究问题。
4. 若计算探索性的“相对独立作用偏差”，必须使用独立名称、写清假设、标注非标准协同分数；不可和 SynergyFinder 官方 Bliss/ZIP 混用。

**交付**：`docs/decisions/m5_synergy_reconstruction_semantics.md`，包含可计算/不可计算项和下一步证据需求。

## 4. 推荐消融编码（先可运行，正式多场景留给 M6）

| ID | 改动 | 目的 |
|---|---|---|
| C0 | B3 对照 / M5 兼容实现 | 隔离实现差异 |
| C1 | C0 + symmetric pair | 检验结构交换一致性及性能代价 |
| C2 | C1 + Fourier dose | 检验非线性连续剂量编码 |
| C3 | C2 + FiLM | 检验剂量调制交互 |
| C4 | C3 + mono supervision | 量化额外单药数据效应 |
| C5 | C4 + boundary loss | 检验药理边界一致性 |
| C6 | C5 + smoothness | 检验局部曲面形状 |

上述是**开发顺序**，不意味着所有模块作用可由相邻差分完全归因。M6 需额外做以固定完整架构为底的 leave-one-module-out 消融，并对单药额外监督做信息量匹配对照。

## 5. 模型与统计检验要求

**主指标**：沿用 M4 `matrix_macro_RMSE`，同时保留 point RMSE、MAE、Pearson、R²。为了与 M4 可比，**不能更改矩阵先算 RMSE 再平均**的主指标定义。

**辅助诊断**：
- `E_swap`：交换两个药物及其剂量后的平均/最大预测绝对差；
- `E_boundary`：虚拟零剂量边界与单药头预测的误差，**不是实测边界误差**；
- 单药训练监督误差；
- `R_smooth`：曲面局部二阶差分摘要；
- out-of-range 预测比例（不做后处理裁剪）；
- 参数量、训练时长、吞吐、峰值显存、最佳 epoch；
- `X/X0 > 1` 标签子集的误差，避免在平均 RMSE 中隐藏难例。

**统计策略（正式 M6 执行）**：
- B3 vs 新模型逐矩阵配对比较，不做不同测试集间的简单非配对 t 检验；
- S1/S2 的不同随机种子并非完全独立样本；优先以 condition 组级 bootstrap、并报告每个 split 的差值；
- S3 只有 34 个细胞、每折 6–7 个 test 细胞，优先细胞系级的分层/层次 bootstrap；报告癌种组成和不确定区间；
- 不把 S1/S2/S3 的 11 组 run 当成 11 个同分布独立重复来声称统计显著。

## 6. 自动测试：必须新增，不少于以下覆盖点

1. **交换对称性**：单点、batch、矩阵转置、随机初始化与训练后；在允许的数值容忍度内相等。
2. **连续剂量**：任意有效正剂量及边界 `0` 都可前向；0 不发生 log(0)/NaN；不同输入单位绝不被静默混用。
3. **单药数据访问**：S1/S2 协议与数量一致；S3 任意 held-out 曲线读入应自动报错（负向测试）。
4. **标签保护**：`y_true >1`、`y_true ==0`、预测小于 0 等值原样保存、不裁剪。
5. **矩阵维度**：完整 4×4、每矩阵 16 点，支持不同矩阵的剂量范围、单独曲面查询。
6. **平滑数值**：等间距 / 非等间距 logdose 两种情况；不会跨矩阵拼错邻居。
7. **损失开关**：每个 λ=0 应正确关闭对应项，所有 loss 与梯度 finite；可反向传播。
8. **checkpoint**：恢复后前向一致；模型参数、配置、scaler 哈希及版本信息可追溯；best checkpoint 原子写入。
9. **冻结清单**：M0–M3 数据/分组/归一化产物前后 SHA256 未变。
10. **回归测试**：M4 原有 131 项（基于其报告口径）仍通过；不得通过删测试维持通过率。

建议增加 property-based 随机输入测试，但不得为了安装新库破坏当前环境。

## 7. M5 验收准则（通过不等于发表级胜利）

M5 通过应同时满足：
- **技术完整**：DoseSyn-Core、mono、boundary、smooth 四类机制能按配置逐个启停，有完整训练和验证流水线。
- **协议正确**：S1/S2/S3 的 mono policy 可自动审计；11 个 split 文件仅用于正确读取和政策测试，不擅自重划分。
- **数值正确**：输入交换对称性等式成立；虚拟边界数值/梯度正常；训练与验证没有 NaN/Inf。
- **结果真实**：至少 S1 seed42 与 S3 fold0 验证集上有真实训练运行、best checkpoint、预测、可复现指标及详细日志；记录不收敛和未超越 B3 的真实情况。
- **无回归**：全量原测试 + 新测试 PASS，M3 manifest 哈希不变，M4 预测和 checkpoint 不覆盖。
- **语义合规**：若协同重建的数学条件不足，明确留作 M6，不宣称已完成正式 Bliss/HSA/Loewe/ZIP。
- **Git 交付**：新增文件、配置、测试、报告提交到远端 main；若远端推送失败，说明原因，不能声称成功。

**注意**：不以“新模型一定优于 B3”作为 M5 强制通过条件；M5 是方法实现与可验证性阶段。若验证性能没有改进，必须忠实披露并给出 M6 待检验原因。

## 8. 目标工程产物（可根据现有实际包结构微调）

推荐文件：
- `configs/m5_dosesyn.yaml`
- `src/dosesyn/models/dosesyn.py`
- `src/dosesyn/models/dose_encoder.py`
- `src/dosesyn/models/mono_head.py`
- `src/dosesyn/losses/pharmacology.py`
- `src/dosesyn/train/m5_trainer.py`
- `scripts/06_run_dosesyn.py`
- `scripts/06_validate_m5.py`
- `tests/test_m5_symmetry.py`
- `tests/test_m5_dose_boundary.py`
- `tests/test_m5_mono_policy.py`
- `tests/test_m5_losses.py`
- `tests/test_m5_training_checkpoint.py`
- `docs/decisions/m5_synergy_reconstruction_semantics.md`
- `m5_report.md`

产物目录：
- `outputs/runs/m5/<run_id>/{config.yaml,best.pt,run_metadata.json,train_log.csv,val_metrics.json,predictions_val.parquet,resources.json}`
- `outputs/metrics/m5/`
- `outputs/figures/m5/`
- `outputs/audit/m5_manifest.json` 及 `m5_test_report.json`

**不创建也不覆盖** `outputs/runs/m4/...` 与 `outputs/metrics/m4/...`。

## 9. 调度、资源和故障处理

- 现有 V100 32GB；M4 B3 峰值显存约 76–111MB。M5 应优先检查 CPU dataloader / Python 调度开销，而非盲目启用 AMP。
- GPU 由多个作业共享，训练时长只能作为条件相近时的资源参考；不得把共享造成的波动解释为模块复杂度的确定差异。
- PBS `-o` 曾未投递日志：继续在脚本中自行重定向 stdout/stderr，并保留 qsub id、退出码及资源记录。
- 不在稳定的 `.venv` 中随意重新安装 matplotlib/numpy/RDKit。绘图尽量沿用现有无依赖 SVG。
- 失败作业必须在 M5 报告中列出原因、是否重跑及影响范围，不能从审计记录里消失。

## 10. 交付报告格式

`m5_report.md` 至少包含：
1. 执行总结：完成与未完成项、阶段准入结论。
2. 模型结构与公式、张量形状、训练/推理接口。
3. C0–C6 各版本定义、参数量和实际运行状态。
4. 单药数据访问审计（按 S1/S2/S3 和 fold/seed）。
5. 所有真实的 S1 seed42、S3 fold0 验证集指标与训练曲线。
6. 交换对称性、零剂量、边界一致性和正则项检查。
7. `X/X0`/协同公式语义判断（可行、不确定、不可行）。
8. 测试数量与 pytest 输出、失败项、GPU/PBS/环境问题。
9. SHA256、Git commit、远端状态、完整复现命令。
10. 尚未解决的问题和进入 M6 的明确建议。

## 11. 直接交给 AI Agent 的主指令

你是 DoseSyn 项目的研究工程师。请先读取项目 `AGENTS.md`、M0–M4 冻结协议和本文件全文，再开始 M5。

**只执行 M5，不执行 M6/M7/M8。**

按 M5-A → M5-B → M5-C → M5-D → M5-E 顺序工作。先以 M4 B3 对照实现为起点，建立可开关的共享 GINE、连续剂量编码、FiLM、严格对称融合、隐式剂量响应曲面、单药头、虚拟零剂量边界及局部平滑损失。先做小样本/单元测试；开发试验优先限于 S1 seed42 验证集和 S3 fold0 验证集；不要在调参过程中访问正式 test 指标，也不要提前运行 M6 的 11 个正式 split。

保证 M3 冻结数据和全部拆分不变，S3 的 held-out 单药标签绝不能被训练或验证读取。保留原始 X/X0、不裁剪，严格区分推断的连续曲面与实测响应点。对无法证明合法的标准协同重建保留 `NOT_VALIDATED` 结论。

输出真实代码、pytest、checkpoint、验证指标、诊断图、模块对照、模型复杂度、失败日志、Git 提交与完整的 `m5_report.md`。所有结果必须可复现，不得隐去未改善的模型表现。完成 M5 后停止并提交验收报告。
