# M3 决策：S1/S2/S3 划分协议与单药监督隔离

日期：2026-10-24  
决策状态：已执行（M3）  
主数据集：`oneil-primary-v1`（18,599 矩阵 / 18,326 条件 / 1,782 单药曲线）

## 1. 为什么不能做点级随机划分

一个 4×4 矩阵内 16 个点共享同一细胞系、同一药物对、同一批次，响应高度相关。
若按点随机划分，测试点会与训练点处于同一矩阵，模型只需记住矩阵即可“泄漏”。
因此本阶段所有划分都以**组**为最小单位，且同一条件的全部批次矩阵一起移动。

## 2. S1：Random-condition

- **原子**：`condition_id`（细胞系 × 无序药物对）
- **比例**：70/15/15，对排序后的条件 ID 用 `numpy.random.default_rng(seed).permutation` 抽样
- **种子**：13、42、2026
- **回答的问题**：在已见细胞系 + 已见药物对的新组合上，模型能否预测响应？
- 由于同一药物对会出现在多个细胞系中，S1 不测试“新药对泛化”。

规模（seed 13）：train 12,828 / val 2,748 / test 2,750 条件；
对应 13,026 / 2,787 / 2,786 个矩阵。

## 3. S2：Unseen-pair

- **原子**：`pair_id`（无序药物对）
- **比例**：70/15/15，种子同上
- **回答的问题**：面对训练集中完全没出现过的药物对，能否泛化？
- 同一药物对的所有细胞系、所有批次必须整体进入同一 split；测试按 `pair_id` 组不重叠。
- 规模（seed 13）：train 12,818 / val 2,720 / test 2,788 条件。

## 4. S3：Unseen-cell（5 折）

- **原子**：`cell_id`
- **折数**：5；按 `(disease, cell_id)` 排序后逐个分配给当前细胞数最少的折，使疾病类型尽量分散
- 对第 k 折：`test = fold k`，`val = fold (k+1) % 5`，其余为 train
- **回答的问题**：面对完全没见过的细胞系，药物-细胞响应能否泛化？
- 34 个细胞系分配到 5 折时并非严格等分：fold 0–2 为 test 7 / val 7 / train 20，fold 3 为 test 7 / val 6 / train 21，fold 4 为 test 6 / val 7 / train 21；5 折的 test 集合恰好覆盖全部 34 个细胞系一次。
- 规模（fold 0）：train 10,780 / val 3,773 / test 3,773 条件。

## 5. 单药监督：known vs strict

| 场景 | 策略 | 可训练单药曲线 | 说明 |
| --- | --- | --- | --- |
| S1 | `known_monotherapy` | 全部 1,782 | 单药标签是**已知先验**，不是独立单药测试集 |
| S2 | `known_monotherapy` | 全部 1,782 | 同上；正式报告必须披露 |
| S3 | `strict_cell_isolation` | 仅训练细胞（fold 0 为 990） | val/test 细胞的 792 条曲线完全封存 |

S3 的封存是强制性的：`mono_assignments.parquet` 中 `trainable=False`，
`label_access=held_out`；训练脚本若读取这些曲线即违反协议。
S3 的每折单药剂量 scaler 也只在训练细胞的曲线上拟合。

## 6. 训练折专属变换

所有统计量只允许在**训练实体**上估计：

- Hallmark 50 维：train 细胞的 mean/std（ddof=0），val/test 只 transform；
- 表达谱（可选）：同一批 train 细胞；
- 药物剂量：train 非零剂量的 `log10(molar)` 的 mean/std；
- 零剂量：不参与 log 拟合，用显式 `zero_flag` 表示；
- 缺失值：train 中位数填补（Primary-v1 实际无缺失，但接口支持）。

`split_metadata.json` 记录 `fit_cells`、`fit_conditions`、`fit_mono_curves` 与统计量指纹；
`scalers.joblib` 保存完整参数。验证器与测试逐项复核：
fit 实体集合 == train 实体集合，且 val/test 实体集合与 fit 集合不相交。

## 7. 泄漏检查（每次生成后运行）

`scripts/03_verify_splits.py` 对每个 split 目录重算：

1. 三份条件文件恰好划分全部 18,326 个条件，两两无交集；
2. 每个矩阵不跨 split；
3. 场景原子（condition/pair/cell）不跨 split；
4. 多批次条件不跨 split；
5. S3 细胞三集合互斥且 fold 的 test 集合与 metadata 一致；
6. 训练组与验证/测试组不相交；
7. scaler fit cells == train cells；
8. S3 单药可训练曲线只属于训练细胞；
9. 计数与 metadata 一致；
10. 所有标签有限。

结果写入 `outputs/audit/m3_leakage_report.json`，当前为 11/11 split 全部 PASS。

## 8. 复现

```bash
.venv/bin/python scripts/03_generate_splits.py --scenario all
.venv/bin/python scripts/03_verify_splits.py
.venv/bin/python -m pytest -q tests
```

同一 commit + 同一配置 + 同一种子应产生相同的 `assignment_sha256` 与
`content_sha256`；任何变化都必须来自显式配置修改。
