# DoseSyn M6-A 性能复现与结构诊断执行任务书 V1.0

**基线 commit**：`9065a21e52edf363369f6b96220db4fc3ffa7203`
**工程目录**：`/home/kouzheng_2/AH_YeP/DoseSyn`
**阶段边界**：只执行 M6-A。不得擅自进入 M6-B/C/D/E，不得读取 test 标签或执行 11 split 正式评估。

## 1. 已确认事实

- M6-0 审计 9 PASS + 2 WARN。S1 seed42 实际划分正确，train/val/test 为 13,022/2,791/2,786；M5 对外摘要中的 seed13 数量属于笔误。
- S1 seed42 同一 val：M4 B3 macro RMSE **0.0801**；M5 C0 **0.1820**、C3 **0.1797**、C6 **0.1870**。
- S3 fold0 同一 val：M4 B3 **0.1681**；M5 C0 **0.2289**、C3 **0.2300**、C6 **0.2331**。
- S1 val 44,656 共同键，标签无差异、剂量最大差为零。B3 与 C0 架构不同（284,545 vs 559,299 参数）；C0 不能称为 B3 原样复现。
- M4 训练日志里 `val_matrix_macro_rmse` 实际为 point RMSE，早停按 point RMSE；独立计算的 B3 matrix-macro RMSE 可靠。
- M5 最佳 epoch 接近 60/60，尚未充分收敛；但单纯延长训练不一定能解释性能差距。
- 目前 165 项自动测试通过。M0–M3 冻结；不得覆盖 M4/M5 产物。

## 2. 第一道关卡：B3 精确参照与差距归因（优先执行）

1. 只读加载 M4 B3 在 S1 seed42、S3 fold0 的原 checkpoint 与 val 预测，按统一评估器复核 S1≈0.0801、S3≈0.1681。报告完整精度、矩阵集合、每矩阵 16 点。
2. 如移植 `b3_exact` 到新管线，必须复用 M4 原模型结构、前向、loss 与特征处理，检查 checkpoint 是否可以加载并实现 batch 预测数值一致；否则注明“非精确移植”。区分 checkpoint 推理一致与随机重训的精度差异。
3. 对照 M4 B3 与 M5 C0，逐项审计药物图节点/边特征、GINE/池化、50维 Hallmark 顺序、剂量原始单位与 log/scaler/zero flag、输入张量、输出范围、标签、损失（MSE/Huber，point/matrix 权重）、batch、优化器、scheduler、激活、参数量、早停指标。
4. 用固定真实 train batch 检查数值尺度、模型初始预测、梯度是否异常；单独比较 `X/X0>1` 的预测与残差，不得裁剪。
5. 若发现损坏输入、单位转换、错误 loss、错误模型行为等严重 bug，先修复并做 regression test，**不要启动耗时的超参搜索**。
6. 交付 `m6a_architecture_diff.md`、`outputs/audit/m6a_parity.json`，给出初步根因结论。

**关卡要求**：原 B3 的 val 参考可信，同口径确认；每项结构差异有代码或实验依据；无未处理的严重数据/单位 bug。

## 3. 第二道关卡：逐层恢复核心预测能力

仅在冻结的 **S1 seed42 train/val** 上建立下面的小规模梯级，必要时用 **S3 fold0 train/val** 确认泛化；正式 test 仍封存。

| 试验 | 结构 | 目的 |
|---|---|---|
| T0 | 原 M4 B3 / 真正 b3_exact | 高性能参照 |
| T1 | M5 ordered control (原 C0) | 定位新框架带来的损失 |
| T2 | B3-like 主干 + 对称融合 | 单独分析对称性 |
| T3 | T2 + Fourier dose | 单独分析剂量编码 |
| T4 | T3 + FiLM | 单独分析剂量条件交互 |

实施规则：
- 架构变化逐项引入，避免同时改变尺寸、激活、loss 和 pooling 后又把差异归因到 FiLM。
- 首轮预设训练上限 `max_epochs=160`、early stopping patience 25–40，统一 checkpoint 依据 **val matrix-macro RMSE**；可以正常提前停止。
- 若 160 epoch 时确有持续改善且没有明显过拟合，才考虑另立配置增加到 240，并记录原因。不得不受约束地不断追加 epoch。
- 记录每 epoch train loss、val macro/point RMSE、val Pearson、学习率、最佳 epoch、时间、GPU 资源；模型参数量也需列出。
- 优先运行 T0–T4 的 S1 seed42，选择少数有诊断价值的版本再上 S3 fold0，避免不必要的全量计算。
- 不能默认 T1 延长训练后就会追上 T0；若仍差很多，优先诊断主干或优化实现。
- 严格区分原 M4 B3 的 point-RMSE 选模结果与新实验按 matrix-macro 选模的结果。

## 4. 第三道关卡：单药边界与平滑约束（仅当核心问题已解释）

从表现合理的 Core 版本出发，在相同训练预算、相同 val 下有限比较：
- P0 = Core；
- P1 = Core + 单药真实监督；
- P2 = P1 + 虚拟零剂量边界；
- P3 = P2 + 平滑（保留 smooth=0 对照）。

预定义的弱化梯度如权重 0、原配置×0.1、原配置；只对最关键的项做有限网格，不进行无上限组合搜索。若平滑仍损害宏平均 RMSE，允许主模型不使用平滑，作为负结果写入消融。必要时检查平滑 warm-up。

**注意**：
- `E_boundary` 是两路模型预测之间的虚拟边界自洽差，不等于实测零剂量药效误差。
- S1/S2 的 known monotherapy 是额外可用监督，报告必须说明信息来源；S3 只能用 train-cell 曲线，禁止读取 val/test 细胞单药标签。
- 不对 `X/X0` 强制裁剪，不未经语义论证计算“标准 Bliss/ZIP/Loewe/HSA”。

## 5. 选择机制与停止条件

在没有读取 test 的前提下：
1. 对 S1 seed42 与 S3 fold0 的 val matrix-macro RMSE、与 B3 的差值、`E_swap`、`E_boundary`、曲率、参数量和训练成本作清晰对照。
2. 最多冻结两个候选：`DoseSyn-Core`（模型本身的剂量交互/对称设计）和可选的 `DoseSyn-Pharm`（加入经证实合理的单药边界项）。
3. 不强制保留 FiLM、mono、boundary、smooth；需要根据证据选择，保留负面结果。
4. 若仍明显劣于 B3，说明为何失败，给出 **NO-GO** 建议，不自动开始所有 11 split 测试；如果差距已经解释，收敛得到验证，给出 **GO** 建议和事先冻结的配置。
5. 严格不使用 test 进行模型选择，也不以测试集结果反复改权重。

## 6. 自动化测试、工程产物和报告

- M0–M3 全部冻结 hash 不变，M4/M5 checkpoint/预测只读，新 run 单独保存 `outputs/runs/m6a/`。
- 保留全部原有 165 项测试，增加真实 B3 parity、早停 metric 校验、剂量单位一致性、预测取值范围、S3 mono seal 的测试。
- PBS 作业自行重定向 stdout/stderr，记录失败/重跑、GPU 共享条件，不为了可视化修改已有 numpy/PyTorch 环境。
- 每 run 保存 config SHA256、Git commit、split/scaler hash、seed、best checkpoint、train_log、val metrics、val predictions 和资源信息。
- 输出：`m6a_report.md`、`m6a_architecture_diff.md`、`m6a_model_selection.md`、`outputs/audit/m6a_parity.json`、`outputs/metrics/m6a/validation_runs.csv`、收敛曲线/残差 SVG、测试日志、Git 提交与远端状态。

## 7. 直接交给 AI Agent 的最终指令

从 commit `9065a21e52edf363369f6b96220db4fc3ffa7203` 开始，先阅读 `AGENTS.md`、M4 B3、M5、`m6_gate0_report.md` 及本任务书。

**只执行 M6-A。** 第一优先级是解释为何原始 B3 的 S1 val RMSE 为 0.0801，而 M5 C0 仍为 0.1820；不能仅仅增加 epoch，也不能把 C0 当作 B3 的精确复现。建立 `b3_exact`，分析输入、结构、剂量、损失与训练口径，先排除严重 bug。

通过诊断后，在 S1 seed42 和 S3 fold0 的合法 train/val 上进行受控的 160 epoch 训练与有限的 240 epoch 复核，以真正的 val matrix-macro RMSE 选 checkpoint。逐项加入对称融合、Fourier、FiLM；基础性能问题有解释后再少量调优 mono/boundary/smooth。

报告所有真实验证结果、失败与负结果、best checkpoint、消融差异、资源、测试和 Git 状态。提出是否进入正式 M6-B/D 的 GO/NO-GO 判定，随后**停止**；不要提前读取 test 或启动全部 11 split。
