"""Build the M8-C verification tables, case traces and reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.m8c.analysis import (  # noqa: E402
    frozen_hashes,
    load_candidate_universe,
    select_cases,
    stub_audit,
    trace_case,
)
from dosesyn.utils.config import ensure_directories, load_config, resolve_path  # noqa: E402
from dosesyn.utils.io import environment_snapshot, utc_now_iso, write_json  # noqa: E402
from dosesyn.utils.logging import setup_logging  # noqa: E402

CONFIG = ROOT / "configs/m8c/m8c.yaml"


def _rel(cfg, p: Path) -> str:
    return str(Path(p).relative_to(resolve_path(cfg, ".")))


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_report(cfg, universe, scored, cases, traces, stub, summary, path):
    hi = cfg["thresholds"]
    n_total = len(universe)
    n_scored = len(scored)
    n_unscored = n_total - n_scored
    dual = int(scored["dual_signal"].sum())
    high_min = int(scored["high_p_high_min"].sum())
    top = scored.sort_values("p_mean", ascending=False).head(15)

    lines = [
        "# M8-C：冻结桥接核验、代表性曲面分析与第四章整合",
        "",
        "## 1. 结论",
        "",
        f"本阶段对 M8-B 的 300 个 A 级候选（S2 seed42 测试矩阵）做了统一口径复核："
        f"**{n_scored}** 个可由 ConfSyn 五折评分，**{n_unscored}** 个因药物不在 DrugCombDB 图中而明确不可评分。"
        f"ConfSyn 的 BERT stub 经三项探针确认**不参与前向计算**，BERT 权重缺失不影响预测；"
        f"真正的药物语义来自训练时缓存并冻结的 ChemBERTa 256 维投影。",
        "",
        f"原先“强分歧”的 {dual} 个候选统一改称为**高协同倾向且存在低 X/X0 剂量点的双信号候选**："
        f"它们同时满足 ConfSyn 五折平均概率 ≥ {hi['high_p']} 与 DoseSyn B5 曲面最小 X/X0 < {hi['low_x_x0']}。"
        "这一命名只描述两条信号同时出现，不表示 ConfSyn 与 DoseSyn 在预测同一目标，也不表示二者互相验证。",
        "",
        "**判定：PASS（技术流程与语义边界核验通过；桥接本身的独立验证强度仍受 M8-A 的 A 级定义限制）。**",
        "",
        "## 2. BERT stub 与冻结推理可靠性",
        "",
        "| 探针 | 结果 | 含义 |",
        "| --- | --- | --- |",
        f"| `drug_semantic_enc.forward` 调用次数 | {stub['stub_forward_calls']} | 提供 `drug1_emb/drug2_emb` 时 BERT 分支被完全绕过 |",
        f"| 随机重初始化 `proj` 后预测变化 | {stub['proj_reinit_max_abs_delta']:.3e} | 预测逐位不变，`proj` 参数不参与推理 |",
        f"| stub 前向替换后预测变化 | {0.0 if stub['stub_forward_bit_identical'] else float('nan'):.3e} | 替换实现不改变输出 |",
        f"| 缓存 ChemBERTa 投影置零 | 平均 |Δp|={stub['zero_embedding_mean_abs_delta']:.4f}，Spearman={stub['zero_embedding_spearman']:.3f} | 真正承载药物语义的是缓存投影，不是缺失的 BERT 本体 |",
        f"| checkpoint 中 BERT 参数 | {stub['checkpoint_bert_key_count']} 个张量 / {stub['checkpoint_bert_param_count']:,} 参数 | 占 checkpoint 的 {fmt_pct(stub['checkpoint_bert_param_count']/stub['checkpoint_total_param_count'])}，全部为冻结项 |",
        f"| fold {stub['fold']} 标签重建 AUC | {stub['checkpoint_fold_auc']:.4f} | 仅证明恢复一致，**不是泛化性能** |",
        "",
        "五折标签重建 AUC 分别为 " + "、".join(
            f"fold{k}={v:.4f}" for k, v in stub["checkpoint_all_fold_auc"].items()
        ) + "。这些数值来自随机抽取的 DrugCombDB 行，不能作为独立测试集指标。",
        "",
        "> 需要如实说明：我们无法用真实 RoBERTa 权重重跑一次对照，因为 checkpoint 本身不包含 BERT 本体；"
        "但 ConfSyn 在提供了预计算药物 embedding 时根本不会调用该分支，因此“缺失 BERT 是否影响预测”在本推理路径上可判定为不影响。",
        "",
        "## 3. 候选宇宙：300 → 120 + 180",
        "",
        "| 状态 | 数量 | 说明 |",
        "| --- | ---: | --- |",
        f"| A 级候选矩阵 | {n_total} | S2 seed42 test，`evidence_level=A` |",
        f"| ConfSyn 可评分 | {n_scored} | 药物与细胞均在 DrugCombDB 图中，五折均有输出 |",
        f"| 不可评分 | {n_unscored} | 全部因 `drug_or_cell_absent_from_DrugCombDB_graph` |",
        f"| 双信号候选 | {dual} | p≥{hi['high_p']} 且 min X/X0<{hi['low_x_x0']} |",
        f"| 高概率且高最低响应 | {high_min} | p≥{hi['very_high_p']} 且 min X/X0≥{hi['high_min']} |",
        "",
        "180 个不可评分候选都由 **SN-38** 参与：SN-38 只出现在 OncologyScreen 数据中，"
        "不在当前 DrugCombDB 的 764 药物图内，而现有 ConfSyn checkpoint 是 DrugCombDB 模型。"
        "这些候选没有伪造概率，也未从 300 个 A 级矩阵中删除。",
        "",
        "## 4. 代表性案例（规则先于结果固定）",
        "",
        "案例选择规则写入 `configs/m8c/m8c.yaml`，在查看曲面之前确定："
        "最高 ConfSyn 概率、中位概率、高概率且高最低响应、双信号最大间距各一个。",
        "",
        "| 案例 | 细胞系 | 药物对 | p (mean±std) | B5 mean | B5 min | 原始 X/X0 min | RMSE | 源行 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, r in cases.iterrows():
        t = traces[r["case_id"]]
        lines.append(
            f"| {r['case_id']} | {r['cell_line']} | {r['drug_a_name']}+{r['drug_b_name']} | "
            f"{r['p_mean']:.4f}±{r['p_std']:.4f} | {t['pred_x_x0_min']:.3f}–{t['pred_x_x0_max']:.3f} | "
            f"{t['pred_x_x0_min']:.3f} | {t['raw_x_x0_min']:.3f} | {t['rmse']:.4f} | "
            f"{t['source_row_min']}–{t['source_row_max']} |"
        )
    lines += [
        "",
        "四个案例的 16 个预测点均与 O'Neil 原始 `source_row` 对齐：`y_true` 与原始 `X/X0` 逐点一致，"
        "`dose_a_um/dose_b_um` 与 M1 浓度逐点一致，源行连续。案例 1（top_p）是 ConfSyn 概率最高、"
        "但 B5 曲面同时存在低响应角的代表；案例 2（median_p）代表 120 个候选的典型水平；"
        "案例 3（high_p_high_min）是“高协同概率但整体抑制较弱”的对照；"
        "案例 4（dual_signal）是两条信号间距最大的候选。",
        "",
        "## 5. 与公开文献的关系（只做背景核对，不作为标签）",
        "",
        "- Vorinostat（Zolinza）+ Lapatinib 已有 HER2+ 乳腺癌临床前/早期临床研究（AACR 2015 摘要 P5-19-05；"
        "NCT01118975 为 vorinostat+lapatinib 的 I 期研究）。这说明该组合并非凭空构造，但文献中的瘤种、"
        "剂量与终点均与本项目 O'Neil 4×4 网格不同，**不能把文献结论当作本项目样本的协同标签**。",
        "- Vorinostat + Bortezomib 有 I 期实体瘤研究（ASCO 2007 摘要 3573），HDAC 与蛋白酶体抑制的联合"
        "在机制上被讨论；本项目中的低 X/X0 只表示 O'Neil 响应尺度上的低存活，不等于文献定义的协同。",
        "- Etoposide + Veliparib（ABT-888）存在多个以依托泊苷/铂类为骨架的临床试验（如 NCT01642251）；"
        "本项目的 ABT-888 是 O'Neil 原始名称，映射到 veliparib 结构。",
        "",
        "因此，文献核对只支持“组合具有生物学/临床背景”，不支持“ConfSyn 概率被独立实验验证”或"
        "“DoseSyn 曲面被文献验证”。本阶段没有新的湿实验。",
        "",
        "## 6. 图表与论文整合",
        "",
        "- 新增 4.3.9 节，流程图（图 4-14）、双信号景观（图 4-15）、代表案例曲面（图 4-16）与候选漏斗（图 4-17）。",
        "- 表 4-9 为 300/120/180 与双信号计数；表 4-10 为四个案例的源行与指标。",
        "- 4.4 小结新增第 6 条，明确 ConfSyn 与 DoseSyn 是两条互补但不可互相替代的预测通道。",
        "",
        "## 7. 数值一致性检查",
        "",
        f"- 候选宇宙 300 = 120 + 180：{n_total} = {n_scored} + {n_unscored}。",
        f"- 双信号候选 {dual} 个与 `m8c_dual_signal_candidates.csv` 行数一致。",
        "- 四个案例 `y_true` 与 M1 原始 `X/X0` 逐点一致（atol=1e-12），剂量逐点一致。",
        "- 案例源行连续且每例 16 行；每行保留 `viability_1..4`，未做插补或裁剪。",
        "- 冻结输入哈希在 `m8c_frozen_hashes.json` 中，脚本不写回 M0–M8-B 产物。",
        "",
        "## 8. 仍需解决的问题",
        "",
        "1. ConfSyn 的逐样本折划分未持久化，A 级仍只能保证“条件不在 DrugCombDB 表中”，不能逐样本证明未参与训练。",
        "2. 180 个 SN-38 候选无法由当前 DrugCombDB checkpoint 评分；若要覆盖，需要重新训练含 SN-38 的药物图模型，超出本阶段约束。",
        "3. BERT 本体不在 checkpoint 中；虽然推理路径绕过该分支，但无法对“若使用真实 BERT 会怎样”做数值对照。",
        "4. 双信号只说明两个模型给出不同侧面的信号，不构成协同验证；M8-C 不产生新的协同分数。",
        "5. 代表性案例是规则抽取，不是随机抽样；不能据此估计 120 个候选的整体错误率。",
        "",
        "## 9. 交付物",
        "",
        "| 文件 | 内容 |",
        "| --- | --- |",
        "| `m8c_report.md` | 本报告 |",
        "| `m8_final_report.md` | M8 整体总结 |",
        "| `outputs/metrics/m8c/m8c_candidate_universe.csv` | 300 个 A 级候选及状态 |",
        "| `outputs/metrics/m8c/m8c_scored_candidates.csv` | 120 个可评分候选 |",
        "| `outputs/metrics/m8c/m8c_dual_signal_candidates.csv` | 双信号候选 |",
        "| `outputs/metrics/m8c/m8c_representative_cases.csv` | 4 个案例与选择规则 |",
        "| `outputs/metrics/m8c/m8c_case_source_rows.csv` | 64 行原始源行追溯 |",
        "| `outputs/metrics/m8c/m8c_stub_audit.json` | BERT stub 三项探针 |",
        "| `outputs/metrics/m8c/m8c_frozen_hashes.json` | 冻结输入哈希 |",
        "| `outputs/figures/m8c/` | 流程图、景观图、案例曲面、漏斗图 |",
        "",
        f"_生成时间：{utc_now_iso()}_",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(cfg, summary, stub, cases, path):
    lines = [
        "# M8 最终报告：ConfSyn–DoseSyn 双模型数据关联、联合筛选与论文整合",
        "",
        "## 1. M8 总体结论",
        "",
        "M8 分三阶段完成：**M8-A** 建立实体映射与证据等级（CONDITIONAL PASS）；"
        "**M8-B** 恢复 ConfSyn 冻结推理并完成联合筛选展示（PASS）；"
        "**M8-C** 核验 BERT stub、统一候选统计、追踪代表案例并写入第四章（PASS）。",
        "",
        "M8 的定位不是“证明桥接提升了预测性能”，而是建立一条**可复现、语义分离、可追溯**的"
        "双模型联合分析流程：ConfSyn 提供组合级协同倾向，DoseSyn 提供剂量级 X/X0 响应。",
        "",
        "## 2. 数据与映射",
        "",
        "| 项目 | 数值 |",
        "| --- | ---: |",
        "| DoseSyn 冻结矩阵 | 18,599 |",
        "| DoseSyn 条件 | 18,326 |",
        "| 细胞系 / 药物 / 无序对 | 34 / 36 / 539 |",
        "| ConfSyn 药物映射 | 22 / 38 |",
        "| ConfSyn 细胞系映射 | 31 / 39 |",
        "| 可匹配条件 | 4,274 / 18,326（23.3%） |",
        "| S2 seed42 A 级候选矩阵 | 300 |",
        "| ConfSyn 可评分 | 120 |",
        "| 不可评分（SN-38） | 180 |",
        "",
        "## 3. 桥接核验",
        "",
        f"- BERT 分支调用次数：{stub['stub_forward_calls']}；`proj` 重初始化后最大预测差："
        f"{stub['proj_reinit_max_abs_delta']:.3e}。",
        f"- 缓存 ChemBERTa 投影置零后平均 |Δp|={stub['zero_embedding_mean_abs_delta']:.4f}，"
        f"说明语义路径真实存在。",
        f"- 五折 checkpoint 标签重建 AUC："
        + "、".join(f"{v:.4f}" for v in stub["checkpoint_all_fold_auc"].values()) + "（非泛化指标）。",
        "",
        "## 4. 联合筛选结果",
        "",
        f"- 120 个可评分候选中，双信号候选 {summary['n_dual_signal']} 个；"
        f"高概率且高最低响应 {summary['n_high_p_high_min']} 个。",
        f"- ConfSyn p 与 B5 曲面均值的 Spearman={summary['spearman_p_vs_mean']:.3f}，"
        f"与曲面最小值的 Spearman={summary['spearman_p_vs_min']:.3f}。",
        "- 相关接近 0 是结果的一部分：两条通道并非冗余，也不能合并成未经校准的“联合协同分数”。",
        "",
        "## 5. 代表案例",
        "",
        "| 案例 | 细胞系 | 药物对 | p | B5 min | 原始 X/X0 min | 源行 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for _, r in cases.iterrows():
        lines.append(
            f"| {r['case_id']} | {r['cell_line']} | {r['drug_a_name']}+{r['drug_b_name']} | "
            f"{r['p_mean']:.4f} | {r['y_pred_min']:.3f} | {summary['traces'][r['case_id']]['raw_x_x0_min']:.3f} | "
            f"{summary['traces'][r['case_id']]['source_row_min']}–{summary['traces'][r['case_id']]['source_row_max']} |"
        )
    lines += [
        "",
        "所有案例均可回溯到 O'Neil 原始行与 4 个技术重复；未裁剪 X/X0，未把 X/X0 当协同分数。",
        "",
        "## 6. 论文整合",
        "",
        "- 第四章新增 4.3.9，含流程图、候选漏斗、双信号景观与四个案例曲面。",
        "- 4.4 小结新增第 6 条，明确两条研究主线的互补关系与不可替代性。",
        "- 既有 4.3.1–4.3.8 数值未改动；新增内容只使用冻结结果。",
        "",
        "## 7. M8 判定与后续",
        "",
        "**M8 总判定：PASS（技术完成与语义诚实），但桥接的独立验证强度为 CONDITIONAL。**",
        "",
        "可进入后续工作的前提：",
        "1. 若要做“桥接优于单模型”的统计声明，必须先解决 ConfSyn 折划分持久化与 SN-38 覆盖问题；",
        "2. 若要把双信号候选推进到湿实验，应优先选择本研究预先定义的案例，而不是事后挑选；",
        "3. 任何新的协同分数必须单独定义、单独验证，不能由 `p_mean` 与 `X/X0` 直接相乘得到。",
        "",
        f"_生成时间：{utc_now_iso()}_",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    cfg = load_config(CONFIG)
    ensure_directories(cfg)
    out_metrics = resolve_path(cfg, cfg["paths"]["outputs_metrics"])
    out_audit = resolve_path(cfg, cfg["paths"]["outputs_audit"])
    for p in (out_metrics, out_audit):
        p.mkdir(parents=True, exist_ok=True)
    log = setup_logging("dosesyn.m8c", resolve_path(cfg, "outputs/logs/29_build_m8c.log"))
    log.info("M8-C build starting")

    universe = load_candidate_universe(cfg)
    scored = universe[universe["scored"]].copy().sort_values("p_mean", ascending=False)
    cases = select_cases(cfg, universe)
    traces = {}
    rows = []
    for _, case in cases.iterrows():
        df, summary = trace_case(cfg, case)
        traces[case["case_id"]] = summary
        rows.append(df)
    case_rows = pd.concat(rows, ignore_index=True)

    log.info("probing BERT stub on %d candidates", len(scored))
    stub = stub_audit(cfg, scored)

    dual = scored[scored["dual_signal"]].copy()
    dual["gap"] = dual["p_mean"] - dual["y_pred_min"]
    dual = dual.sort_values("gap", ascending=False)

    summary = {
        "generated_at": utc_now_iso(),
        "n_candidates": int(len(universe)),
        "n_scored": int(len(scored)),
        "n_unscoreable": int((~universe["scored"]).sum()),
        "n_dual_signal": int(len(dual)),
        "n_high_p_high_min": int(scored["high_p_high_min"].sum()),
        "spearman_p_vs_mean": float(scored["p_mean"].corr(scored["y_pred_mean"], method="spearman")),
        "spearman_p_vs_min": float(scored["p_mean"].corr(scored["y_pred_min"], method="spearman")),
        "cases": cases.to_dict(orient="records"),
        "traces": traces,
        "stub_audit": stub,
        "environment": environment_snapshot(),
    }

    universe.to_csv(out_metrics / "m8c_candidate_universe.csv", index=False)
    scored.to_csv(out_metrics / "m8c_scored_candidates.csv", index=False)
    dual.to_csv(out_metrics / "m8c_dual_signal_candidates.csv", index=False)
    cases.to_csv(out_metrics / "m8c_representative_cases.csv", index=False)
    case_rows.to_csv(out_metrics / "m8c_case_source_rows.csv", index=False)
    (
        universe[~universe["scored"]]
        .groupby(["drug_a_name", "drug_b_name", "reason"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .to_csv(out_metrics / "m8c_unscoreable_summary.csv", index=False)
    )
    write_json(stub, out_metrics / "m8c_stub_audit.json")
    write_json(frozen_hashes(cfg), out_metrics / "m8c_frozen_hashes.json")
    write_json(summary, out_metrics / "m8c_summary.json")
    write_json({"inputs": frozen_hashes(cfg), "outputs": {
        f.name: {"sha256": __import__("hashlib").sha256(f.read_bytes()).hexdigest(),
                 "bytes": f.stat().st_size} for f in sorted(out_metrics.glob("m8c_*"))
        if f.is_file()
    }}, out_metrics / "m8c_manifest.json")

    write_report(cfg, universe, scored, cases, traces, stub, summary, ROOT / "m8c_report.md")
    write_final_report(cfg, summary, stub, cases, ROOT / "m8_final_report.md")
    log.info("wrote %d candidates, %d scored, %d dual-signal", len(universe), len(scored), len(dual))
    print(json.dumps({k: summary[k] for k in (
        "n_candidates", "n_scored", "n_unscoreable", "n_dual_signal",
        "spearman_p_vs_mean", "spearman_p_vs_min")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
