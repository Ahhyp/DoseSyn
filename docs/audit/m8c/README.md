# M8-C 审计归档

本目录是 `outputs/metrics/m8c/` 中 M8-C 关键结果的 Git 跟踪副本，便于在论文/答辩中直接核对，不依赖大体积输出目录。

| 文件 | 内容 |
| --- | --- |
| `m8c_candidate_universe.csv` | 300 个 A 级候选（120 可评分 + 180 不可评分） |
| `m8c_scored_candidates.csv` | 120 个 ConfSyn 五折可评分候选 |
| `m8c_unscoreable_summary.csv` | 180 个不可评分候选按药物对汇总（全部含 SN-38） |
| `m8c_dual_signal_candidates.csv` | 27 个双信号候选 |
| `m8c_representative_cases.csv` | 4 个预设案例及选择规则 |
| `m8c_case_source_rows.csv` | 4×16 原始行、重复值、浓度与残差 |
| `m8c_stub_audit.json` | BERT stub 三项探针与五折 AUC |
| `m8c_frozen_hashes.json` | 本阶段读取的 7 个冻结输入哈希 |
| `m8c_summary.json` | 计数、案例、追溯、stub 审计汇总 |
| `m8c_manifest.json` | 输入/输出哈希清单 |

复现命令：

```bash
cd /home/kouzheng_2/AH_YeP/DoseSyn
.venv/bin/python scripts/29_build_m8c.py
/tmp/m7venv/bin/python scripts/30_plot_m8c.py
.venv/bin/python -m pytest -q tests
```

注意：`29_build_m8c.py` 会重写 `m8c_report.md`。若已人工润色报告，请只重跑脚本中的表格生成部分，或先备份报告；本目录的 CSV/JSON 是最终审计快照。
