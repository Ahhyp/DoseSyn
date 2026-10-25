# DoseSyn M8 总报告：ConfSyn–DoseSyn 冻结联合筛选与论文整合

## 1. 总体结论

**M8 总判定：PASS（分阶段为 M8-A CONDITIONAL PASS、M8-B PASS、M8-C PASS）。**

M8 完成了两项独立训练模型在**不重新训练、不微调、不使用测试标签选模型**前提下的条件级连接：

- **M8-A**：完成 ConfSyn checkpoint/数据/划分/预测接口审计，建立药物、细胞、条件三级映射，并把 4,274 个可匹配 O'Neil 条件按 A/B/C/D 分级；因 ConfSyn 逐样本折划分未持久化、且缺少 OncologyScreen checkpoint，判 CONDITIONAL PASS。
- **M8-B**：恢复 ConfSyn 的 5 个 DrugCombDB 冻结 checkpoint，对 S2 seed42 的 300 个 A 级候选做五折推理；120 个可评分、180 个因 SN-38 不在药物图中明确不可评分；用 DoseSyn B5 冻结 checkpoint 生成 4×4 `X/X0` 曲面并完成联合排序与可视化。
- **M8-C**：核验 BERT stub 不参与前向计算；统一 300→120+180 口径；把“27 个强分歧”修正为“高协同倾向且存在低 `X/X0` 剂量点的双信号候选”；对 4 个预设案例完成原始行追溯；把 4.3.9 节、4.4 衔接说明和 4 张新图写入第四章。

M8 **没有**产生新的协同分数，**没有**把 `X/X0` 当作协同标签，**没有**声称桥接提升了任何未实际测量的性能指标。

## 2. 联合筛选流程

```text
O'Neil Primary-v1 (18,599 matrices)
        │  M3 冻结 split + train-only scalers
        ▼
DoseSyn B5 (frozen, 535,682 params) ──► X/X0 4×4 surface statistics
        │
        │  条件键 = (cell_id, 无序药物对)
        ▼
A-level candidates (S2 seed42 test, 300)
        │  ConfSyn DrugCombDB graph (68 cells / 764 drugs)
        ▼
ConfSyn 5 folds (frozen, 88.4M params) ──► p(synergy >= 0) mean/std/min/max
        │
        ▼
120 scored + 180 unscoreable + 27 dual-signal + 4 traced cases
```

流程只做推理与连接，不更新任何参数。完整流程图见 `docs/paper/figures/m8c/fig4_14_workflow.svg`。

## 3. 关键结果

| 项目 | 数值 |
| --- | ---: |
| ConfSyn checkpoint 数 | 5（DrugCombDB CV folds 0–4） |
| ConfSyn 参数量 | 88,442,425 |
| B5 参数量 | 535,682 |
| S2 seed42 A 级候选 | 300 |
| ConfSyn 可评分 | 120 |
| 不可评分（SN-38 不在图中） | 180 |
| 双信号候选（p≥0.9 且 min X/X0<0.2） | 27 |
| 高概率且高最低响应（p≥0.95 且 min X/X0≥0.35） | 2 |
| ConfSyn p 与 DoseSyn 平均响应 Spearman | 0.033 |
| ConfSyn p 与 DoseSyn 最小响应 Spearman | −0.118 |
| ConfSyn 五折标签重建 AUC | 0.9519–0.9741 |
| 代表案例 B5 RMSE | 0.0333–0.0525 |
| 案例源行追溯 | 64 行（4×16），全部连续 |

四个预设案例的选择规则在查看曲面前固定：最高概率、中位概率、高概率且高最低响应、双信号最大间距。它们分别覆盖“高协同 + 低响应角”“典型候选”“高协同 + 整体弱抑制”“双信号最强”四种情形，而不是事后挑选的好看案例。

## 4. 语义边界

1. **ConfSyn 输出**是 DrugCombDB 定义下 `synergy >= 0` 的正类概率。DrugCombDB 的 synergy 是聚合来源的混合标度，不是本项目重新计算的 Bliss/HSA/Loewe/ZIP。
2. **DoseSyn 输出**是 O'Neil 的 `X/X0` 响应，0 表示接近完全抑制，1 表示接近 DMSO 对照；它是响应，不是协同分数。
3. 两者不能相乘、加权或校准成单一分数。M8 的联合表只并排放置两列，不生成“联合协同分数”。
4. “双信号”只表示两条独立通道同时给出的信息，**不是**相互验证，也**不是**湿实验结论。

## 5. BERT stub 核验

ConfSyn checkpoint 含 104 个 BERT 张量（44.1M 参数，占 49.9%），但推理时提供预计算 ChemBERTa 投影，模型走 embedding 分支，不调用 `drug_semantic_enc.forward`：

- BERT 前向调用次数：0；
- 未使用的 `proj` 随机重初始化后预测变化：0.000e+00；
- stub 替换后预测变化：0.000e+00；
- 缓存投影置零后平均概率变化：0.3306，Spearman 降至 0.495。

因此，“缺失 BERT 本体是否影响预测”在本推理路径上可判定为不影响；但无法用真实 BERT 权重做数值对照，因为 checkpoint 本身不含 BERT 本体。报告如实保留这一限制。

## 6. 论文整合

第四章新增/修改如下：

- 新增 **4.3.9 ConfSyn–DoseSyn 联合筛选与剂量响应分析**，含 4.3.9.1 流程、4.3.9.2 stub 可靠性、4.3.9.3 候选宇宙、4.3.9.4 代表案例、4.3.9.5 互补性与边界。
- 4.4 本章小结新增第 6 条，并补充“两条互补研究主线”的衔接说明。
- 新增图 4-14（流程图）、4-15（双信号景观）、4-16（四案例曲面）、4-17（候选漏斗）。
- 新增表 4-9（stub 探针）、4-10（候选宇宙）、4-11（代表案例）。
- 图文件同时提供 SVG/PDF/PNG，位于 `docs/paper/figures/m8c/`。

可直接用于论文的表述已避免三类越界：(1) 不把 A 级写成“完全独立测试”；(2) 不把高概率+低响应写成“模型分歧”或“相互验证”；(3) 不用文献中的组合背景替代本项目标签。

## 7. 交付物

| 路径 | 内容 |
| --- | --- |
| `m8c_report.md` | M8-C 专项报告（stub、候选、案例、图表、限制） |
| `m8_final_report.md` | 本文件，M8 总报告 |
| `docs/paper/chapter4_experiments.md` | 第四章正文（含 4.3.9 与 4.4 更新） |
| `docs/paper/figures/m8c/` | 图 4-14–4-17（SVG/PDF/PNG） |
| `docs/audit/m8c/` | 冻结哈希、候选表、案例源行、图表审计 |
| `outputs/metrics/m8c/` | M8-C 全量 CSV/JSON |
| `outputs/metrics/m8b/` | M8-B 评分、排名、checkpoint 审计 |
| `outputs/metrics/m8a/` | M8-A 映射、证据等级、split 独立性 |
| `scripts/29_build_m8c.py` | M8-C 审计/案例构建 |
| `scripts/30_plot_m8c.py` | 图 4-14–4-17 |
| `tests/test_m8c_outputs.py` | 输出契约、数值一致性、来源追溯测试 |

M8-A/B 的完整交付物见 `m8a_report.md`、`m8b_report.md` 与各自 `docs/audit/m8{a,b}/`。

## 8. 复现命令

```bash
cd /home/kouzheng_2/AH_YeP/DoseSyn

# M8-A：映射/证据分级（只读 ConfSyn 与 M0–M7 产物）
.venv/bin/python scripts/26_audit_m8a.py

# M8-B：冻结 ConfSyn 五折推理 + B5 曲面
.venv/bin/python scripts/27_run_m8b.py
/tmp/m7venv/bin/python scripts/28_plot_m8b.py

# M8-C：stub 核验、候选统一、案例追溯、图 4-14–4-17
.venv/bin/python scripts/29_build_m8c.py
/tmp/m7venv/bin/python scripts/30_plot_m8c.py

# 全量测试
.venv/bin/python -m pytest -q tests
```

`/tmp/m7venv` 只用于 matplotlib 绘图；M8 的推理与审计在项目 `.venv` 中完成。ConfSyn 代码以只读方式加入 `sys.path`，不改其仓库。

## 9. 限制与后续

1. ConfSyn 的逐样本训练/测试折划分未持久化，A 级只能保证“条件不在 DrugCombDB 表中”，不能逐样本证明未参与训练。
2. 180 个 SN-38 候选无法由当前 DrugCombDB checkpoint 评分；若要覆盖，需要重新训练含 SN-38 的药物图模型，超出 M8 约束。
3. ConfSyn checkpoint 不含 BERT 本体；虽然推理绕过该分支，但无法做“真实 BERT”数值对照。
4. 27 个双信号候选是优先级列表，不是湿实验验证；下一步应做独立剂量响应实验或外部数据集验证，而不是继续在测试集上调参。
5. 联合流程没有新的性能指标；任何“联合提升”表述都需要未来的前瞻性实验支撑。

## 10. Git

- M8-A：`4212a82e438276336032cd3410525a7a8d25ad65`
- M8-B：`c3b4039d4aeb33d45ff8452047a9264feb18e1c9`
- M8-C：本报告与第四章更新提交后的 commit 见最终回复与 `outputs/metrics/m8c/m8c_manifest.json`。

所有提交只包含源码、配置、测试、文档与小体积审计表；不包含原始数据、checkpoint、虚拟环境或大体积预测文件。
