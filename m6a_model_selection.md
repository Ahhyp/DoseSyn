# M6-A 模型选择



- S1 最佳：`m6a__b3sym_film__s1_random_condition__seed_42`（0.0670）；S3 最佳：`m6a__b3sym_fourier__s3_unseen_cell__fold_0`（0.1695）。
- 跨场景稳健选择：**T3 `b3sym_fourier`**（S1 0.0820 / S3 0.1695，均接近或达到 B3），作为 M6-B 默认配置；T4 FiLM 仅建议在 S1 类同分布场景使用或作为集成成员，不单独用于跨细胞系泛化结论。
- 不建议回退到 C0 ordered 头：其 S1 0.1820 的差距不是训练预算能解释的。
- 建议：**GO**。

