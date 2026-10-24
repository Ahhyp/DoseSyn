# DoseSyn M6 正式实验与模型验证执行任务书 V1.0

**研究定位**：第四章基于剂量条件交互与药理一致性约束的药物组合响应曲面预测  
**承接阶段**：M5 完成报告，最终 commit `820dfcb4fb4dd6ca4baaf212f9f16f2ed64fa662`  
**工程目录**：`/home/kouzheng_2/AH_YeP/DoseSyn`  
**仓库**：https://github.com/Ahhyp/DoseSyn  
**执行边界**：仅执行 M6；M6 前置审计不通过时暂停，不自动进入 M7 或 M8。

## 0. 研究目标、当前事实与科学风险

M5 已实现：
- 共享 GINE / 50维 Hallmark 细胞编码器；
- scalar / MLP / Fourier 剂量编码；
- FiLM 剂量条件交互；
- 严格药物交换对称融合；
- 共享单药头、虚拟零剂量边界一致性、非均匀剂量网格平滑损失；
- 151/151 pytest 通过，仅报告 S1 seed42 和 S3 fold0 **验证集**实验，无本阶段 test 评估。

M5 val `matrix_macro_RMSE`：
- S1：C0=0.1820，C3 FiLM=0.1797，C6 Full=0.1870；
- S3 fold0：C0=0.2289，C3=0.2300，C6=0.2331。
- S1 所有 best epochs 靠近 60 epoch 预算终点；不能把 M5 验证结果当作最终优化结果。
- S1 加 boundary 后 `E_boundary`=0.0289（是模型-单药头自洽差，不是实测单药边界 RMSE）；加 smooth 后主指标变差。

M4 B3 `matrix_macro_RMSE` 为 S1 test 三种子平均 0.0795、S2 0.0897、S3 0.1994。**不能将这些 M4 test 指标直接与 M5 单个验证集指标比较**；必须用相同验证矩阵/指标重算。

M4 B3 参数量为 284,545；M5 C0 559,299。**C0 不是已证明的“B3 原样复现”**。M6 必须保留 M4 B3 的真实评估，M5 C0 仅作为同架构内部消融起点；新增模块的收益不能直接用 C0 代替 B3 来声称。

## 1. M6-0 硬性关卡：冻结划分和 B3 兼容性审计（先执行，不通过立即停止）

### 1.1 S1 seed42 数量冲突

M4 正式记录：
- S1 seed13：train=13,026，val=2,788，test=2,785 矩阵；
- **S1 seed42：train=13,022，val=2,791，test=2,786 矩阵**。

M5 报告将 “S1 seed42” 写为 train=13,026、val=2,788、test=2,785，**恰好和 M4 seed13 一致**。

操作：
1. 只读检查 M3 `s1_random_condition/seed42` 与 `seed13` 的冻结划分 manifest/parquet、路径、SHA256；
2. 对每个 M5 S1 run 记录实际加载的 split 文件绝对路径、SHA256、train/val condition_id、matrix_id 集合哈希与记录数；核查 `run_metadata.json`、`predictions_val.parquet`、训练日志及配置；
3. 将 M5 预测矩阵 ID 集合与 M3 seed42 的 val matrix ID 做集合相等比较，同时与 seed13 比较；输出缺失、多余、交集计数和 checksum；
4. 若仅报告计数错误：单独提交勘误，不篡改原产物；
5. 若实际加载 seed13：将错误标注的 M5 seed42 结果标记 `INVALID_FOR_SEED42`，**重新在真正 seed42 上运行必要的验证集实验**，不改动 M3 冻结文件；
6. 如实际是新的数据逻辑错误，暂停 M6，优先修复并复测，禁止继续完整 benchmark。

### 1.2 在相同验证集重新计算 M4 B3

1. 找到 M4 `S1 seed42` 和 `S3 fold0` 已保存的原始 `best.pt` 与 `predictions_val.parquet`；
2. 用统一评估器复核 M4 的 val `matrix_macro_RMSE`、point RMSE、Pearson、R²；必要时用 checkpoint 在冻结 val 重算；
3. 验证 matrix_id/condition_id 与 M5 评估使用同一组 val；同一标签 `X/X0`、剂量单位、矩阵宏平均公式；
4. 检查 M4 B3 和 M5 C0 的输入特征、剂量缩放、训练/早停策略、参数量、激活函数与网络结构；
5. 若架构不同，**明确把 M5 C0 标成“M5 ordered control”**，不可写成“严格复现 B3”。如需精确 B3 同管线桥接，可实现一个配置上完全对齐的 `b3_exact`，做相同 val 复测。
6. 检查 M5 的 val 0.18 vs M4 val 的差距究竟来自 train budget、实现、网络差异、split 混用还是评估口径；不得在未知原因情况下宣称 M5 更强。

### 1.3 资源、测试与冻结审计

- 不修改 M0–M3 冻结数据/特征/split/scaler/monotherapy 访问协议；
- 对 M4 和 M5 既有产物仅只读，不覆写旧 checkpoint、预测或报告；
- M4 原测试 + M5 20 项新测试必须继续通过，新增数据路径一致性回归测试；
- 所有结果都存到 M6 单独目录；
- **关卡输出**：`outputs/audit/m6_gate0_split_audit.json`、`outputs/audit/m6_gate0_baseline_alignment.json`、`m6_gate0_report.md`；
- **放行准则**：split 对齐或错误重跑完成；B3 同口径 val 参照已建立；冻结哈希未变；无法解释的异常已记录且不会导致虚假结论。

## 2. M6-A：只用 train/val 的训练充分性与小范围优化

### 2.1 首轮预算

以 S1 seed42 和 S3 fold0 开发；不要在这里使用 test 标签或 test 指标。重点考察：
- B3（原框架或等价实现）
- M5 C0
- M5 C3（当前最好的 S1 FiLM）
- M5 C5（mono + boundary）
- M5 C6（full，带 smooth）

在已冻结数据上延长训练预算：建议 `max_epochs` 先测试 **160、240**（不是要求一定跑满）；`patience` 由 60 epoch 的短预算改成约 25–40 epochs，可从验证曲线制定预注册小网格；优化器和学习率调度保持可追溯。需要检查 val 曲线是否出现明确的最优点/平台，并报告因预算终止还是 early-stop。

若 B3 收敛好而 C0/C3 长期较差，应先定位输入/训练逻辑，而非盲目增加 epoch 到几百。

### 2.2 正则强度与 warm-up

以已实现的 `mono_weight`、`boundary_weight`、`smooth_weight` 开关为基础，进行**有限且事先写明的验证集搜索**：
- 对单药监督与边界项，比较关闭、原始权重、弱化权重（例如原始权重的 0.1 倍）；
- 对 smooth 项，比较 0、原始权重的 0.1 倍和原始权重；必要时添加前 10%–20% epochs warm-up（仅少量方案）；
- 保留所有损失项的未加权与加权量级及梯度范数；检查负迁移是否来自单药数据、边界项或平滑项；
- 使用同一训练和验证矩阵，同一初始化种子/训练预算比较关键变体；
- 搜索只用 val；记录所有失败与未改善候选；**不要为了保留故事上的 Full 版而强制用验证集更差的参数**。

### 2.3 预选候选

选择不超过两个由 val 决定的正式候选，例如：
- `DoseSyn-Core`：symmetric + Fourier + FiLM，不加单药/边界/平滑；
- `DoseSyn-Pharm`：在 Core 上加入经验证不严重损害预测的 mono/boundary/可能的 smooth。

名称与保留条件取决于真实结果；允许 `smooth_weight=0` 为最终选择。若复杂完整网络持续劣于 Core，就将完整版本保留为消融反例，而非主模型。

### 2.4 M6-A 产物

`outputs/metrics/m6/validation_search.csv`、val 曲线 SVG、训练日志、逐实验配置和 commit、训练预算/资源、`m6_model_selection.md`，明示最终配置是在 test 前冻结。

## 3. M6-B：同口径的消融与机制检验

开发阶段至少需要两种互补消融：

1. **递进消融**：ordered C0 → symmetric → Fourier → FiLM → mono → boundary → smooth；
2. **基于最终候选的 leave-one-out**：w/o FiLM、w/o Fourier、w/o symmetry、w/o mono、w/o boundary、w/o smooth（模型确有此模块时）；
3. **监督信息公平控制**：S1/S2 单药已知先验可能给 DoseSyn 更多标签信息；区分 `no_mono` 和 `with_mono`，报告额外监督来源、曲线数及 S3 封存策略；
4. **诊断**：`E_swap_max/mean`、`E_boundary`（模型-模型自洽，不是实测边界）、Mono training RMSE、局部曲率、`X/X0>1` 的验证误差；
5. **统计**：比较同一验证矩阵预测的配对差值，避免用不同矩阵的总体指标简单相减而夸大模块收益。

正则和剂量编码对照不要默认带来性能收益；用结果选择最合理模型配置。

## 4. M6-C：外部文献 Baseline 的可复现性审计

M4 的 B0 Mean、B1 Tree-Dose、B2 MLP-Dose、B3 GINE-Base 为必保留基线。

外部文献模型按优先级尝试：
1. **DD-PRiSM**：单药作用 + 组合响应交互分解；
2. **comboKR 2.0**：连续剂量曲面核方法；
3. **DeepSynBa**：药理参数化剂量响应曲面；
4. **comboFM**：高阶因子分解模型；
5. **comboLTR**：潜在张量重建（可复现性受限时如实说明）。

实施要求：
- 优先官方代码及许可证，记录 paper/version/commit，独立虚拟环境，不污染 DoseSyn `.venv`；
- 先做一张 `baseline_compatibility.csv`：原始输入、输出定义、单药需求、对测试实体/测试细胞有无标签依赖、能否处理 `X/X0>1`、4×4 网格、任意剂量、S3 未见细胞、依赖与预估成本；
- **无法在 S3 严格细胞隔离下合法使用的 baseline 不列入该场景的严格主表**；可在协议不同的补充分析报告，不混为一谈；
- 不复制文献中的已发表测试数字到 DoseSyn 主表，所有主表指标必须在冻结 Primary-v1 同一 split、同一标签尺度上重新评价；
- 对复现失败的模型提供原因和 logs，不因凑数量而作无法解释的重写；
- 如果 M6 资源有限，优先 3 个外部方法；comboFM 和 comboLTR 可作为扩展。

## 5. M6-D：冻结配置后的正式 11 split 评估

**只有 M6-0 与 M6-A 放行，并且最终候选/主要基线配置已经预先冻结，才执行 test。**

正式测试：
- S1 `seed13/42/2026`：3 组；
- S2 `seed13/42/2026`：3 组；
- S3 `fold0/1/2/3/4`：5 组；
- 共 11 个 M3 冻结 split（不得重新拆分）。
- 对每个最终入选模型在每个 split 训练一次，保存 best val checkpoint；
- 训练结束且最佳配置确定以后读取 test 一次；不得用 test 结果做下一轮超参选择；
- 如发现训练/数据 bug，修复后要说明被污染的版本/结果并将修复作为新版本重新评估，不能悄悄替换好看结果。

指标：
- 主：`matrix_macro_RMSE`（每矩阵先 RMSE、再矩阵宏平均）；
- 辅：point RMSE、MAE、Pearson、R²；按场景等权平均；
- 质量：预测越界/负值比例，`X/X0>1` 误差，交换误差、虚拟边界误差、曲率；
- 成本：参数量、训练 wall clock、GPU 峰值显存、推理吞吐；
- 数据：每预测文件严格 `16 × test_matrices` 行，无缺失/重复 matrix-dose ID，保持未裁剪标签与预测。

统计：
- 对相同 split/矩阵做模型预测误差配对比较，至少报告每 split 改善量；
- S1/S2 种子不是独立实体全集，置信区间可按 condition 作为簇估计；
- S3 以独立 cell 为簇做谨慎的 bootstrap/层次统计，说明只有 34 个细胞、每折 6–7 个 test cell；
- 不将 11 个不同划分当成 11 次完全独立重复来声明显著性；
- S3 fold0 无前列腺测试细胞，癌症亚型探索只做描述性分析。

## 6. M6-E：剂量曲面与未见剂量实验

**原则**：4×4 离散观测可以检验“已测但训练时隐藏的剂量点”，不能验证真正在未采样的任意连续坐标的生物学真值。

建议另建 `derived_m6_unseen_dose` 评估协议，不修改 M3 的 S1/S2/S3 固定矩阵归属：
- **矩阵内插值**：每张训练矩阵只使用指定的 3×3=9 个剂量观测，隐藏一个内侧剂量行和列共 7 个点；按完全相同的剂量位置规则对 val/test 矩阵做 9 已知 / 7 隐藏掩码；预测误差仅在被隐藏的 7 个真实观测点计算。
- **边界外推**：可选隐藏极值剂量行与列，单独报告，不和插值混成一个分数；
- **分组隔离不变**：train/val/test 的矩阵/condition/cell 分组完全沿用 M3，不许从 val/test 抽任何隐藏观测训练；
- 与至少 B2、B3、comboKR（若可用）公平比较；
- 对 query 的连续插值区域绘制曲面，只标记已实测坐标对应的 ground truth，不把模型插值得到的密集网格值当作实测标签。

剂量曲面示例：
- 3 个场景分别选有代表性的 case（非仅挑表现最佳）；
- 每例展示真实 4×4、预测 4×4、绝对误差 4×4；
- 另可展示连续查询曲面与虚拟边界，但必须明确属于模型推断。

## 7. 协同重建科学边界

M5 将标准 Bliss/HSA/Loewe/ZIP 重建标记为 `NOT_VALIDATED`。M6 **不得**把未经证实的 `X/X0` 直接代入经典 viability/inhibition 公式后称为标准协同分数。

如果补足证据/定义：
- 另立语义决策文件与独立可复现的计算模块；
- 检查零剂量单药/vehicle 锚点、response→effect 转换、Loewe/ZIP 曲线拟合可辨识性；
- 如上述证据仍不足，则把“标准协同重建”移至未来工作，不作为 M6 验收必须达成项；
- 仍可完成第四章核心“组合剂量响应预测”研究和 ConfSyn→DoseSyn 后续桥接。

## 8. M6 验收门槛

M6 的交付必须包括：
1. **M6-0 检查明确 PASS**：seed42/seed13 疑点查清，M4 B3 与 M5 同口径 val 对齐，报告是否修正；
2. 模型预选来自 train/val，收敛预算充分；所有已试配置和失败/负结果有记录；
3. 内部 B0–B3、最终 DoseSyn 的 11 split test 完整；外部 baseline 的进度、可执行性与实际复现结果透明；
4. 一张 S1/S2/S3 的主性能表、逐 split 明细、配对增益和谨慎不确定性分析；
5. 至少一张完整消融表（后续 M7/M8 可整理图表），并分析 S3 为什么难；
6. 剂量曲面真实/预测/误差示例和未见剂量协议，若未见剂量实验尚未完成必须显式标为“部分完成”；
7. S1/S2/S3 单药访问无违规；未裁剪 `X/X0`；测试只在冻结配置后评价；
8. 所有旧/新 pytest PASS、冻结 manifest 哈希未变、M4/M5 产物不覆盖、资源/环境/失败审计完备；
9. `m6_report.md` 说明是否明显超过 B3、超出的不确定性、模块是否真的有增益、相对外部 baseline 的位置；
10. 提交代码/文档/配置/报告到远端 Git；如未成功同步远端，明确写为未推送。

**不强制要求模型胜过全部 baseline，也不强制保留全部损失项**；质量要求在于独立复现、同协议和对负结果的诚实解释。

## 9. 推荐目录

- `configs/m6_experiments.yaml`
- `scripts/07_audit_m6_gate0.py`
- `scripts/07_validate_train_m6.py`
- `scripts/07_run_m6_final.py`
- `scripts/07_evaluate_m6.py`
- `scripts/07_plot_m6.py`
- `src/dosesyn/evaluation/m6_metrics.py`
- `src/dosesyn/baselines/external/`（隔离依赖/适配）
- `tests/test_m6_split_identity.py`
- `tests/test_m6_baseline_alignment.py`
- `tests/test_m6_unseen_dose.py`
- `outputs/audit/m6_gate0_*.json`
- `outputs/metrics/m6/validation_search.csv`
- `outputs/metrics/m6/final_by_split.csv`
- `outputs/metrics/m6/final_summary.csv`
- `outputs/metrics/m6/baseline_compatibility.csv`
- `outputs/figures/m6/*.svg`
- `m6_gate0_report.md`、`m6_model_selection.md`、`m6_report.md`

路径可依仓库实际结构微调，务必保存独立 run_id + config SHA + checkpoint + val/test metrics + 预测。

## 10. 第一轮 AI Agent 指令：只运行 M6-0

请以 commit `820dfcb4fb4dd6ca4baaf212f9f16f2ed64fa662` 为基线，先读 `AGENTS.md`、M3、M4、M5 相关代码/协议/报告。**这次先只执行 M6-0，不进行完整 M6-B/C/D/E，也不要读正式 test 标签。**

尤其核实 M5 S1 seed42 的矩阵数为何等于 M4 seed13，核对实际 split 路径、真实 train/val ID 集合和 M3 SHA。读取 M4 B3 S1 seed42、S3 fold0 的保存 checkpoint/val 预测，以与 M5 一致的指标在相同 val 上比较并审计 B3 与 C0 架构差异。用真实证据说明 0.0795 test 与 0.1820 val 的差距不能直接当成性能变化，但何处需要进一步调查。

发现不一致时修复错误路径或报告，若错误实验需要重跑，仅在正确的验证集重跑并保留旧错误产物审计；严禁改动冻结划分、覆盖 M4/M5 产物和查看 test 标签。完成 `m6_gate0_report.md`、审计 JSON、测试、Git 记录后即停止，交付我们审阅；不擅自继续到所有 11 个 split。
