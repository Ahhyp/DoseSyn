# M4 基线训练协议（DoseSyn）

## 1. 目标与边界

M4 只建立可信的基线训练/评估框架，不实现完整 DoseSyn。没有引入 FiLM、
单药边界约束、平滑正则或协同重建；这些属于 M5。

- 不重新划分数据：直接读取 M3 冻结的 `data/splits/`。
- 不使用测试集调参：模型选择与早停只看验证集 `matrix_macro_RMSE`。
- 标签为原始 `X/X0`，不裁剪、不截断、不做后处理。
- 所有预测逐点保存（每个 4×4 矩阵 16 行）。

## 2. 模型

| 模型 | 输入 | 说明 |
| --- | --- | --- |
| B0 Mean | 无（训练集均值） | 常数预测；下界基线 |
| B1 Tree-Dose | 细胞统计 + 药物 ID + 剂量 | HistGradientBoosting，400 棵树 |
| B2 MLP-Dose | 细胞 50 维 + 药物 embedding + 剂量/零剂量标记 | 两个 256 维隐层 |
| B3 GINE-Base | 分子图（9 原子特征/4 键特征）+ 细胞 + 剂量 | 3 层 GINE + 全局均值池化 + MLP 头 |

B2/B3 都是 4×4 共享头：16 个点通过 `row=k//4, col=k%4` 从
`dose_a[4]`/`dose_b[4]` 取值，一次前向输出 16 个响应。

## 3. 训练与早停

- 优化器 AdamW，lr=1e-3，weight decay=1e-4，Huber(delta=1.0)。
- batch=64，eval batch=128，最多 80 epoch，patience=15，最少 5 epoch。
- 梯度裁剪 5.0；ReduceLROnPlateau(factor=0.5, patience=5)。
- 每个 run 独立种子；B0/B1 不涉及随机初始化。
- 显存：V100 32 GB，单 run 占用 < 100 MB，瓶颈在 Python 侧而非显存。

## 4. 指标

主指标 `matrix_macro_RMSE`：先对每个矩阵的 16 个点算 RMSE，再对所有矩阵取宏平均。
同时报告 `point_RMSE/MAE`、`pooled_Pearson`、`pooled_R²`、`condition_macro_RMSE`。
所有指标由 `src/dosesyn/training/metrics.py` 统一计算，测试中有手算校验。

## 5. 泄漏控制

- M3 已保证 condition/batch/细胞系不跨集合；M4 不重新划分。
- S1/S2 的已知单药曲线只作先验披露；S3 中验证/测试细胞的单药曲线被 sealed。
- 特征标准化参数来自 M3 训练折 scaler，不在 M4 重新拟合。
- B1 只在训练折拟合；早停只用验证折。
- 测试折只在最终评估时调用一次。

## 6. 资源与调度

- 集群为 Torque/PBS；GPU 不纳入调度，需在作业内指定 `CUDA_VISIBLE_DEVICES`。
- PBS 的 `-o` 在本集群不生效，作业脚本开头自行 `exec > log 2>&1`。
- 一个作业请求 `nodes=node1:ppn=4`，并在同一节点上按 GPU 0/1 分配。

## 7. 复现

```bash
cd /home/kouzheng_2/AH_YeP/DoseSyn
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/04_run_baseline.py --scenario s1_random_condition --seed 42 \
    --models b0_mean b1_tree b2_mlp b3_gine --device cuda
.venv/bin/python scripts/05_aggregate_m4.py
```

单次运行目录：`outputs/runs/m4/<model>__<scenario>__<split>/`，包含
`config.yaml`、`run_metadata.json`、`best.pt`、`train_log.csv`、
`val_metrics.json`、`test_metrics.json`、`predictions_{val,test}.parquet`、
`resources.json`。

## 8. 已知限制

- 未做超参搜索；所有模型用同一配置。
- B3 只是 GINE 基线，不含 pair-aware 交互。
- 预测未约束到物理范围；测试标签也未裁剪。
- S3 各折癌症类型覆盖不均，macro 指标方差较大。

## 8. 结果摘要（2026-10-24）

11 个 split、4 个模型，共 44 个运行全部成功。测试集 `matrix_macro_RMSE`：

| 模型 | S1（3 seeds） | S2（3 seeds） | S3（5 folds） | 等权平均 |
| --- | ---: | ---: | ---: | ---: |
| B0 Mean | 0.3060 | 0.3022 | 0.3066 | 0.3049 |
| B1 Tree-Dose | 0.2467 | 0.2497 | 0.2819 | 0.2594 |
| B2 MLP-Dose | 0.0764 | 0.1022 | 0.2033 | 0.1273 |
| B3 GINE-Base | 0.0795 | 0.0897 | 0.1994 | 0.1228 |

- S1/S2 上 B2 略优；S3 上 B3 略优，说明图结构在未见细胞系时更有帮助，但两者
  相对 B0/B1 的增益都明显缩小。
- 所有运行早停只看验证集；测试集未参与调参，预测未裁剪（负预测 10,379 个，
  `X/X0` 最大 2.237 被完整保留）。
- 详细逐 run 指标见 `outputs/metrics/m4/runs_summary.csv`，报告见 `m4_report.md`。
