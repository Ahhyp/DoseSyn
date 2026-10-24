# DoseSyn M5 完整模型开发报告
生成时间：2026-10-24T09:47:38+00:00  
配置：`configs/m5_dosesyn.yaml`（m5-v1）  
## 1. 结论
- 完成 DoseSyn-Core（对称融合 + 连续剂量 + FiLM + 共享单药头 + 边界/平滑损失），共 **12** 个验证运行；本阶段只做 S1 seed42 / S3 fold0 的**验证集**实验，未触碰测试集。
- 严格交换对称在 C1–C6 上 `E_swap_max < 1e-6`（float32 容差内为零）；C0 有序对照的 E_swap 非零，说明对照确实非对称。
- 协同语义审计结论：`X/X0` 不可无假设转换为 Bliss/HSA/Loewe/ZIP，这些量在本仓库统一标记 `NOT_VALIDATED`，未作为标签或标准分数输出。
- S1 seed42：最优 **c3_film**，val macro RMSE=0.1797；相对 C0 的提升仅约 0.002，属于单种子下的微小差异。
- S3 fold0：最优 **c0_control**，val macro RMSE=0.2289；对称/连续剂量/边界/平滑在该折没有带来稳定增益。
- 零剂量边界损失把 S1 的 E_boundary 从无边界约束的 0.3347 降到 0.0289–0.0302；共享单药头在 S1 的 mono train RMSE 为 0.058–0.063，说明单药监督确实被优化。
- 以上是单种子/单折验证结果，未做显著性检验；60 epoch 下 S1 的最佳 epoch 多在 57–60，模型仍欠训练，本报告不声称 M5 已优于 M4 B3。

## 2. 模型与开关
| 组件 | 实现 | 开关 |
| --- | --- | --- |
| 药物编码 | GINE（9 原子 + 4 键类别嵌入） | 固定共享 |
| 细胞编码 | 50 维 Hallmark → MLP | 固定 |
| 剂量编码 | scalar / MLP / Fourier | `dose_encoding` |
| 药物-细胞交互 | shared MLP + FiLM(γ,β) | `film` |
| 药物对融合 | 对称 `[t1+t2, t1*t2, |t1−t2|, cell]`；C0 为有序对照 | `symmetric_pair` |
| 单药头 | 共享 MonoHead | `mono_weight>0` |
| 零剂量边界 | F(d,0)≈M(d) | `boundary_weight>0` |
| 曲面平滑 | 自身 log 剂量网格二阶差分 | `smooth_weight>0` |

## 3. 协议与防泄漏
- 直接读取 M3 冻结 split；S1 seed42 与 S3 fold0；**只用 val**，test 未读取。
- 早停/选型只用 `val matrix_macro_RMSE`；学习率调度同指标。
- 单药标签：S1 使用 `known_monotherapy`（全部可训练，仅作已知先验）；S3 使用 `strict_cell_isolation`，封存 792 条，只训练 990 条训练细胞曲线。
- 标准化只加载 M3 训练折 scaler；未在全数据上重拟合。
- 零剂量是显式 sentinel/flag，不对 0 取 log10。

## 4. 实验结果（验证集）
| 场景 | 变体 | macro RMSE | point RMSE | Pearson | R² | E_swap max | E_boundary | mono train RMSE | 参数量 | epochs | 时间(s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 已见条件 | c0_control | 0.1820 | 0.2067 | 0.7608 | 0.5787 | 6.62e-01 | — | — | 559299 | 60 | 213.6 |
| S1 已见条件 | c1_symmetric | 0.1809 | 0.2054 | 0.7640 | 0.5837 | 1.79e-07 | — | — | 579267 | 60 | 274.4 |
| S1 已见条件 | c2_fourier | 0.1805 | 0.2052 | 0.7647 | 0.5847 | 1.79e-07 | — | — | 583939 | 60 | 324.9 |
| S1 已见条件 | c3_film | 0.1797 | 0.2049 | 0.7655 | 0.5859 | 2.38e-07 | — | — | 600579 | 60 | 380.2 |
| S1 已见条件 | c4_mono | 0.1800 | 0.2052 | 0.7647 | 0.5847 | 1.79e-07 | 0.3347 | 0.0583 | 600579 | 60 | 585.1 |
| S1 已见条件 | c5_boundary | 0.1803 | 0.2051 | 0.7649 | 0.5850 | 1.79e-07 | 0.0289 | 0.0618 | 600579 | 60 | 1069.9 |
| S1 已见条件 | c6_full | 0.1870 | 0.2090 | 0.7565 | 0.5690 | 2.38e-07 | 0.0302 | 0.0633 | 600579 | 60 | 813.9 |
| S3 未见细胞系 | c0_control | 0.2289 | 0.2480 | 0.6381 | 0.4068 | 5.56e-01 | — | — | 559299 | 20 | 63.5 |
| S3 未见细胞系 | c1_symmetric | 0.2325 | 0.2526 | 0.6266 | 0.3848 | 2.98e-07 | — | — | 579267 | 28 | 99.5 |
| S3 未见细胞系 | c2_fourier | 0.2324 | 0.2516 | 0.6346 | 0.3897 | 1.79e-07 | — | — | 583939 | 17 | 64.1 |
| S3 未见细胞系 | c3_film | 0.2300 | 0.2479 | 0.6470 | 0.4072 | 1.19e-07 | — | — | 600579 | 16 | 60.8 |
| S3 未见细胞系 | c6_full | 0.2331 | 0.2504 | 0.6365 | 0.3953 | 1.79e-07 | 0.1006 | 0.1360 | 600579 | 16 | 168.6 |

### 4.1 变体增益（相对 C0，验证 macro RMSE）
| 场景 | 变体 | macro RMSE | 相对 C0 变化 |
| --- | --- | ---: | ---: |
| S1 已见条件 | c0_control | 0.1820 | +0.0000 |
| S1 已见条件 | c1_symmetric | 0.1809 | -0.0011 |
| S1 已见条件 | c2_fourier | 0.1805 | -0.0015 |
| S1 已见条件 | c3_film | 0.1797 | -0.0023 |
| S1 已见条件 | c4_mono | 0.1800 | -0.0020 |
| S1 已见条件 | c5_boundary | 0.1803 | -0.0017 |
| S1 已见条件 | c6_full | 0.1870 | +0.0050 |
| S3 未见细胞系 | c0_control | 0.2289 | +0.0000 |
| S3 未见细胞系 | c1_symmetric | 0.2325 | +0.0037 |
| S3 未见细胞系 | c2_fourier | 0.2324 | +0.0035 |
| S3 未见细胞系 | c3_film | 0.2300 | +0.0011 |
| S3 未见细胞系 | c6_full | 0.2331 | +0.0042 |

- 以上仅为单种子/单折验证集结果，不构成显著性结论；M6 才做多种子/全折/消融。
- 与 M4 的 B3 GINE-Base 相比：M4 S1 三种子平均 test macro RMSE=0.0795，M5 当前最好的 S1 val macro RMSE=0.1797（c3_film）。二者集合角色不同（test vs val），不能直接做显著性比较；M5 的 C0–C6 在 60 epoch 预算下验证损失仍缓慢下降，属于欠训练状态，本报告不声称 M5 已超过 M4。

### 4.2 失败实验与修复记录
- 首次提交的 GPU 作业（20569 S1 / 20570 S3）在 `c0_control`、`c1_symmetric` 立即失败：scalar 剂量编码返回 2 维特征，导致有序解码器输入维度 310、`dose_proj` 输入 2（期望 64）。
- 修复：scalar 分支改为 `Linear(2, out_dim)`，所有编码器统一输出 `out_dim`；随后 7 个变体的前向/反向手工检查通过，作业 20573/20574 重跑成功。没有修改数据、划分或测试集。
- 另外，早期运行缺少 `E_boundary_mean` / `mono_train_rmse`，用 `scripts/06_m5_posthoc_diagnostics.py` 从 `best.pt` 在验证集上补算，未重新训练、未读取测试集。

## 5. 收敛与资源
- `m5__s1_random_condition__seed_42__c0_control`：best epoch 60 / 60，Tesla V100-PCIE-32GB，峰值 reserved 169.9 MB。
- `m5__s1_random_condition__seed_42__c1_symmetric`：best epoch 59 / 60，Tesla V100-PCIE-32GB，峰值 reserved 192.9 MB。
- `m5__s1_random_condition__seed_42__c2_fourier`：best epoch 57 / 60，Tesla V100-PCIE-32GB，峰值 reserved 195.0 MB。
- `m5__s1_random_condition__seed_42__c3_film`：best epoch 59 / 60，Tesla V100-PCIE-32GB，峰值 reserved 197.1 MB。
- `m5__s1_random_condition__seed_42__c4_mono`：best epoch 60 / 60，Tesla V100-PCIE-32GB，峰值 reserved 220.2 MB。
- `m5__s1_random_condition__seed_42__c5_boundary`：best epoch 59 / 60，Tesla V100-PCIE-32GB，峰值 reserved 396.4 MB。
- `m5__s1_random_condition__seed_42__c6_full`：best epoch 59 / 60，Tesla V100-PCIE-32GB，峰值 reserved 398.5 MB。
- `m5__s3_unseen_cell__fold_0__c0_control`：best epoch 8 / 20，Tesla V100-PCIE-32GB，峰值 reserved 157.3 MB。
- `m5__s3_unseen_cell__fold_0__c1_symmetric`：best epoch 16 / 28，Tesla V100-PCIE-32GB，峰值 reserved 161.5 MB。
- `m5__s3_unseen_cell__fold_0__c2_fourier`：best epoch 5 / 17，Tesla V100-PCIE-32GB，峰值 reserved 163.6 MB。
- `m5__s3_unseen_cell__fold_0__c3_film`：best epoch 4 / 16，Tesla V100-PCIE-32GB，峰值 reserved 167.8 MB。
- `m5__s3_unseen_cell__fold_0__c6_full`：best epoch 4 / 16，Tesla V100-PCIE-32GB，峰值 reserved 346.0 MB。

## 6. 协同语义审计
详见 `docs/decisions/m5_synergy_reconstruction_semantics.md`。
- Bliss / HSA / Loewe / ZIP 均需要组合剂量处的单药效应或可逆单药曲线；本数据无零剂量锚点、存在 `X/X0>1` 与四舍五入的 0，未通过验证。
- 因此全部标 `NOT_VALIDATED`；M5 不输出任何“标准协同分数”，协同重建留待 M6。

## 7. 自动检查
| 检查 | 状态 | 详情 |
| --- | --- | --- |
| S1 seed42 变体齐全 | PASS | ['c0_control', 'c1_symmetric', 'c2_fourier', 'c3_film', 'c4_mono', 'c5_boundary', 'c6_full'] |
| S3 fold0 变体齐全 | PASS | ['c0_control', 'c1_symmetric', 'c2_fourier', 'c3_film', 'c6_full'] |
| 自动测试 | PASS | 151 passed in 42.37s |
| M0–M3 冻结文件哈希未变 | PASS | {'checked': 15, 'failures': [], 'status': 'PASS'} |
| 预测逐点保存（16 行/矩阵、无 NaN） | PASS | 12/12 |
| 严格交换对称（C1–C6 E_swap≤1e-6） | PASS | max=2.98e-07 |
| C0 有序对照不同（E_swap>1e-6） | PASS | [0.6617252230644226, 0.5557067394256592] |
| 零剂量边界已测量 | PASS | 3 runs |
| 单药头已训练/评估 | PASS | 4 runs |
| 协同语义审计为 NOT_VALIDATED | PASS | docs/decisions/m5_synergy_reconstruction_semantics.md |
| M4 运行结果未被覆盖 | PASS | 44/44 successful dirs |

## 8. 测试与复现
- 完整测试：**151 passed / 0 failed / 0 errors**（151 passed in 42.37s）。
- 命令：
```bash
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/06_run_dosesyn.py --scenario s1_random_condition --seed 42 --variants c0_control c1_symmetric c2_fourier c3_film c4_mono c5_boundary c6_full --device cuda --force
.venv/bin/python scripts/06_run_dosesyn.py --scenario s3_unseen_cell --fold 0 --variants c0_control c1_symmetric c2_fourier c3_film c6_full --device cuda --force
.venv/bin/python scripts/06_validate_m5.py --pytest-log outputs/audit/pytest_m5.log
```

## 9. 已知问题与 M6 准备
- M5 只在 S1 seed42 与 S3 fold0 的验证集上做模块验证；没有测试集调参，也没有跑 M6 的 11-split 全量实验。
- 单药监督依赖 M3 冻结的 mono_assignments；S1/S2 的 known_monotherapy 是已知先验，不能当作独立单药测试。
- 零剂量边界使用虚拟查询，不是实测 vehicle；它对真实基线漂移的鲁棒性未知。
- 平滑损失只使用矩阵内部相邻剂量点，不跨矩阵，也不假设单调性。
- X/X0→effect 的协同指标语义尚未验证，报告不使用 Bliss/HSA/Loewe/ZIP 作为标准协同分数。
- 未做完整超参搜索；M5 目标是接口/协议/数值正确性，不保证优于 M4 B3。
- S1 的最佳 epoch 多在 57–60，验证损失仍在下降，60 epoch 预算下模型欠训练；M6 应延长训练并做 LR/早停消融。
- M5 只有单种子/单折结果，没有置信区间或显著性检验；不能据此判断组件真实增益。
- M5 的验证 macro RMSE 与 M4 的测试 macro RMSE 集合角色不同，不可直接比较优劣。
- 消融入口已就绪：`symmetric_pair`、`dose_encoding`、`film`、`mono_weight`、`boundary_weight`、`smooth_weight` 均为配置开关。
- M6 应扩到 S1/S2 三种子、S3 五折，并固定测试集后再报告 test；M5 未做。

## 10. Git 记录
- 仓库：https://github.com/Ahhyp/DoseSyn（main 分支）。
- M4 基线：`632ba0ce0105b4e1ba560c2f4bbfe83c122d57c3`。
- M4 报告表述更正：`e03b92e`、`6d17122`（仅改文字，未改实验数据）。
- M5 代码与报告提交：
  - `94fedc0 M5: add DoseSyn core with symmetric fusion, FiLM, mono head and boundary/smooth losses`
  - `6d17122 M4: correct S2 model comparison statement`
  - `e03b92e M4: fix S2 model-comparison wording in report`
  - `632ba0c M4: include parameter counts in aggregate report`
- 复现命令见第 8 节；输入/报告哈希见 `outputs/audit/m5_manifest.json`。
