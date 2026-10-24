# DoseSyn M3：数据冻结与无泄漏实验划分（AI Agent 任务书 V1.0）

**仓库**：https://github.com/Ahhyp/DoseSyn  
**项目路径**：`/home/kouzheng_2/AH_YeP/DoseSyn`  
**上游版本**：M2 报告提交 `cdd9c8b`，执行前记录当前 HEAD。  
**边界**：本阶段只处理数据集冻结、实验划分、train-only 变换、数据加载和自动测试，**禁止训练模型**。

## 一、不可更改的决策

- 以 M2 `feature_complete=True` 的 O'Neil 子集构建 `ONEIL_PRIMARY_V1`，预期 18,599 张矩阵、18,326 个独立 `(unordered_pair_id, cell_id)` 条件，运行后重新核对真实数量。
- 首版只使用 36 个图状态为 `OK` 的药物。Carboplatin / Oxaliplatin 的图尚需复核，禁止自动补造铂配位键或将多个碎片视为验证通过的配位结构；保留原始数据，后续作为单独敏感性分析。
- 细胞系须有经审核的映射、CCLE 表达和 Hallmark 通路。不可自动用相似名称替代缺失实体。
- 主标签始终是原始 `X/X0`，允许 >1，不能裁剪或用 sigmoid 强行约束。原始 M1/M2 文件只读，哈希不能变化。
- `condition_id = (无序药物对, 细胞系)`；同一条件的不同 Batch、复测、不同剂量网格必须进入同一主划分。
- 每个 split 的基因/通路/剂量标准化参数只能在训练折上拟合，val/test 只能 transform。
- 保存随机种子、所有输入哈希、规则版本、代码 commit、分组分配结果，支持重复生成。

## 二、M3.0 冻结数据集

读取并核对 M1 双药点表、矩阵元数据、单药点表、单药元数据，以及 M2 的药物/细胞映射、图缓存、表达/通路表和三张特征连接索引。

输出 `configs/dataset_primary_v1.yaml`、`data/processed/oneil_primary_v1_matrix_index.parquet`、`data/processed/oneil_primary_v1_mono_index.parquet`、`docs/decisions/m3_primary_dataset.md`、`outputs/audit/m3_input_manifest.json`。

检查主键唯一性、16 点矩阵完整性、特征均可检索、排除原因可追溯。不得直接改写已处理的 M1/M2 数据。

## 三、M3.1 主实验划分

| 场景 | 分组 key | 推荐协议 | 说明 |
|---|---|---|---|
| S1 Random-condition | `condition_id` | train/val/test=70/15/15，seed=13/42/2026 | 新的 pair-cell 条件，不等于新药物对 |
| S2 Unseen-pair | `unordered_pair_id` | train/val/test≈70/15/15，seed=13/42/2026 | 测试无序药物对不出现在训练或验证，但组成药物可见 |
| S3 Unseen-cell | `cell_id` / DepMap ID | 34 个可用细胞建议分成 5 个外层测试折；每次另选一折验证，剩余训练 | 严格未知细胞评估，输出每折 held-out cell ID 与癌种分布 |

必须保证 S1 的 condition 集合互不相交、S2 的 pair 集合互不相交、S3 的 cell 集合互不相交。所有同条件复测不能跨划分。允许 S1 不同分区共享药物/细胞，S2 不同分区共享药物，不能误报为泄漏。主任务绝不能对 16 个剂量点随机分组。

**单药监督协议**：生成独立 `mono_assignments`。S1/S2 可定义 `known_monotherapy`（允许使用药物-细胞已测单药监督，但必须披露；已训练的单药曲线不可再称独立单药测试样本）。S3 必须提供严格版本：测试及验证细胞的单药标签不得进入训练或任何训练归一化；无标签细胞表达可以用于预测。后续若做已知单药的未知细胞实验，必须另命名为 `mono-informed`。禁止把单药记录与双药记录混成一个随机划分。

**可选场景 S4/S5**：分别建立剂量插值与外推遮蔽规则。S4 隐藏 4×4 中间剂量行/列，S5 隐藏最高剂量边缘行/列，保存物理浓度、观测/隐藏 mask，并防止相同条件不同 batch 泄露隐藏响应。明确 S4/S5 是“同一矩阵部分剂量可见的补全/外推任务”，绝不能当作新组合测试。M3 可实现生成器，训练留到 M6。

## 四、M3.2 训练折专属特征变换

- 50 维 Hallmark scores：仅在训练折的**唯一细胞系**上拟合 scaler，不能用剂量点出现频次对细胞加权。
- 19,144 维基因表达：保留原始 `log2(TPM+1)`；如额外构建 z-score 版本，也仅使用训练细胞拟合，并安全处理零方差。34 个细胞不多，严防不稳定的逐基因标准化。
- 浓度统一保留 µM、molar、log10(molar)，零剂量使用独立 `zero_flag`，严禁 `log10(0)`；剂量 scaler 只用训练剂量拟合。
- 缺失值策略透明可追踪，不静默填零。保存每个 split 的 scaler、训练细胞 ID、训练剂量范围和 SHA256。
- 所有标准化/插补/特征选择在 train-fit、val-test-transform；M2 的逐细胞 rank-based 通路分数本身不使用全数据统计，可以缓存。

## 五、M3.3 模型无关的数据加载接口

提供后续所有 baseline / DoseSyn 统一读取的 Dataset 或索引接口，输出：
`drug_a_id, drug_b_id, cell_id, graph_a, graph_b, pathway_vector, (optional expression_vector), dose_a_um, dose_b_um, log10_dose_a, log10_dose_b, x_x0_raw, matrix_id, condition_id, pair_id, batch_id, split`。
当前预测点的技术重复和真实标签统计不得作为推理时的输入特征。

执行 smoke test：载入一个 batch，检查图张量与 pathway 维度、药物/剂量轴同步、标签有限值、矩阵归属；不需要训练模型。

## 六、必须新增的自动测试

1. S1 `condition_id` train/val/test 三者交集为空。
2. S2 `unordered_pair_id` 三者交集为空；A/B 交换不改变 pair ID。
3. S3 `cell_id` 三者交集为空，测试及验证细胞的单药曲线没有进入训练池。
4. 所有非 S4/S5 场景每张 4×4 矩阵不跨分区，同条件跨批次复测不跨分区。
5. 输入 M1/M2 文件 SHA256 不变，全部规范化索引行数对账。
6. 全部可训练药物图 `graph_status=OK`；禁止异常铂类图混入 Primary-v1。
7. 真实 `X/X0` 不裁剪，超过 1 和等于 0 的值（若落入子集）能加载。
8. scaler 的 fit 实体和训练实体完全一致；val/test 不更新均值/方差或填补参数。
9. 指定 seed 重跑产生内容等价、哈希稳定的 manifest。
10. 训练/验证/测试每组非空且有合理的独立条件；不能失败后悄悄降级点级随机划分。
11. 既有 81 项 M0/M1/M2 测试全部通过。

## 七、交付代码、报告与目录

新增或扩展：
`configs/dataset_primary_v1.yaml`
`configs/splits/s1_random_condition.yaml`
`configs/splits/s2_unseen_pair.yaml`
`configs/splits/s3_unseen_cell.yaml`
`scripts/03_generate_splits.py`
`scripts/03_verify_splits.py`
`src/dosesyn/splits/{build,verify,mono_policy}.py`
`src/dosesyn/features/transform_train_only.py`
`src/dosesyn/data/dataset.py`
`tests/test_m3_*.py`
`docs/decisions/{m3_primary_dataset,m3_split_protocols}.md`
`m3_report.md`

生成：
`data/splits/<scenario>/<seed_or_fold>/{train,val,test}_conditions.parquet`
`data/splits/<scenario>/<seed_or_fold>/mono_assignments.parquet`
`data/splits/<scenario>/<seed_or_fold>/scalers.joblib`
`data/splits/<scenario>/<seed_or_fold>/split_metadata.json`
`outputs/audit/{m3_manifest,m3_leakage_report}.json`

每个协议/种子/折输出：训练/验证/测试的细胞数、药物数、药物对数、独立条件数、矩阵数、剂量点数、可用单药曲线数、癌种分布、剂量分布与 `X/X0` 分布；报告 condition/pair/cell 跨分区交集以及哪些必须为零。另报告批次复测条件和排除原因。

**重要**：Git 只推源码、配置、测试、报告摘要、必要的小体积分组 manifest/校验哈希；大型 Parquet、原始 Excel、图缓存和 `.venv` 不提交。

## 八、验收与结束

M3 PASS 必须同时满足：Primary-v1 冻结、S1/S2/S3 分组正确、单药标签访问协议明确、训练折专属变换正确、原始哈希不变、所有新增与既有测试通过、可装载真实批次、结果可用 seed/commit 复现。不能达标则输出 blocker，不得伪造 PASS 或偷换实验协议。

建议运行命令（Agent 实现后确保有效）：
```bash
cd /home/kouzheng_2/AH_YeP/DoseSyn
.venv/bin/python scripts/03_generate_splits.py --scenario S1 --seed 42
.venv/bin/python scripts/03_generate_splits.py --scenario S2 --seed 42
.venv/bin/python scripts/03_generate_splits.py --scenario S3 --fold all
.venv/bin/python scripts/03_verify_splits.py --all
.venv/bin/python -m pytest -q tests
git status
git diff --stat
```
完成后提交真实执行命令、实际统计、测试报告、变更文件、Git commit hash、尚未解决风险，**到 M3 结束就停止，不开始 M4 训练**。
