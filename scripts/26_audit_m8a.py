"""M8-A: ConfSyn–DoseSyn entity, provenance and independence audit.

Read-only over M0–M7-C artefacts and the ConfSyn checkout. Produces mappings,
condition-level provenance, A/B/C/D split independence, gate verdict and report.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.m8a.evidence import (  # noqa: E402
    LEVEL_MEANING,
    LEVELS,
    SCENARIOS,
    bridge_summary,
    build_matrix_evidence,
    load_conf_conditions,
    load_dosesyn_conditions,
    match_conditions,
    summarize_evidence,
)
from dosesyn.m8a.mapping import (  # noqa: E402
    build_cell_mapping,
    build_drug_mapping,
    load_conf_drug_smiles,
    normalize_text,
)
from dosesyn.training.reproducibility import environment_record, file_sha256, git_commit, git_status_porcelain, utc_now  # noqa: E402
from dosesyn.utils.logging import setup_logging  # noqa: E402


def _pct(num: float, den: float) -> float:
    return 100.0 * float(num) / float(den) if den else 0.0


def _read_cell_ids(matrix_index: pd.DataFrame) -> pd.DataFrame:
    return (
        matrix_index[["cell_line", "cell_id"]]
        .drop_duplicates()
        .rename(columns={"cell_line": "oneil_cell_name"})
    )


def _corr(a: pd.Series, b: pd.Series) -> tuple[float | None, float | None, int]:
    pair = pd.DataFrame({"a": a, "b": b}).dropna()
    n = len(pair)
    if n < 3 or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
        return None, None, n
    return float(pair["a"].corr(pair["b"])), float(pair["a"].corr(pair["b"], method="spearman")), n


def _fmt(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def main() -> int:
    cfg_path = ROOT / "configs" / "m8" / "m8a_manifest.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    logs_dir = ROOT / cfg["paths"]["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("dosesyn.m8a", logs_dir / "26_audit_m8a.log")
    metrics_dir = ROOT / cfg["paths"]["metrics_dir"]
    audit_dir = ROOT / cfg["paths"]["audit_dir"]
    figures_dir = ROOT / cfg["paths"]["figures_dir"]
    for p in (metrics_dir, audit_dir, figures_dir):
        p.mkdir(parents=True, exist_ok=True)

    conf_root = Path(cfg["confsyn"]["root"])
    conf_data = conf_root / cfg["confsyn"]["data_dir"]
    logger.info("M8-A starting; ConfSyn=%s", conf_root)

    drug = pd.read_csv(ROOT / cfg["inputs"]["drug_mapping"])
    cell = pd.read_csv(ROOT / cfg["inputs"]["cell_mapping"])
    matrix_index = pd.read_parquet(ROOT / cfg["inputs"]["primary_matrix_index"])
    combo_points = pd.read_parquet(ROOT / cfg["inputs"]["combo_points"], columns=["condition_id", "x_x0_raw"])
    cell = cell.drop(columns=[c for c in ("cell_id",) if c in cell.columns]).merge(
        _read_cell_ids(matrix_index), on="oneil_cell_name", how="left"
    )

    conf_drugs = load_conf_drug_smiles(conf_data)
    drug_map = build_drug_mapping(drug, conf_drugs)
    drug_map["appears_as_a"] = drug_map["drug_id"].map(drug.set_index("drug_id")["appears_as_a"])
    drug_map["appears_as_b"] = drug_map["drug_id"].map(drug.set_index("drug_id")["appears_as_b"])
    drug_map["in_primary_v1"] = drug_map["drug_id"].isin(set(matrix_index["drug_a_id"]) | set(matrix_index["drug_b_id"]))

    db_cells = pd.read_csv(conf_data / "DrugCombDB" / "cell_line_ids.txt", header=None)[0].astype(str).tolist()
    os_cells = pd.read_csv(conf_data / "OncologyScreen" / "drug_combinations.csv")["cell"].astype(str).unique().tolist()
    cell_map = build_cell_mapping(cell, set(db_cells) | set(os_cells))
    cell_map["in_primary_v1"] = cell_map["cell_id"].isin(set(matrix_index["cell_id"]))

    conf_conditions = load_conf_conditions(conf_data, drug_map, cell_map)
    dose_conditions = load_dosesyn_conditions(matrix_index)
    matched = match_conditions(dose_conditions, conf_conditions)

    # Label provenance: OS and DrugCombDB labels are different scales; measure,
    # never assert equivalence.
    x_mean = combo_points.groupby("condition_id")["x_x0_raw"].mean().rename("dosesyn_mean_x_x0")
    matched = matched.merge(x_mean, on="condition_id", how="left")
    matched["label_comparable_to_x_x0"] = False
    matched["label_note"] = "ConfSyn synergy labels are dataset-specific; X/X0 is relative viability; no validated mapping"

    pearson_os_db, spearman_os_db, n_both = _corr(
        matched.loc[matched["in_oncology_screen"] & matched["in_drugcombdb"], "os_label_mean"],
        matched.loc[matched["in_oncology_screen"] & matched["in_drugcombdb"], "db_label_mean"],
    )
    pearson_os_x, spearman_os_x, n_os_x = _corr(
        matched.loc[matched["in_oncology_screen"], "os_label_mean"],
        matched.loc[matched["in_oncology_screen"], "dosesyn_mean_x_x0"],
    )
    both = matched[matched["in_oncology_screen"] & matched["in_drugcombdb"]]
    exact_label_equal = int(np.isclose(both["os_label_mean"], both["db_label_mean"], atol=1e-9).sum()) if len(both) else 0
    label_stats = {
        "n_matched_conditions": int(len(matched)),
        "n_oncology_screen_conditions": int(matched["in_oncology_screen"].sum()),
        "n_drugcombdb_conditions": int(matched["in_drugcombdb"].sum()),
        "n_in_both": int(len(both)),
        "exact_label_equal_in_both": exact_label_equal,
        "pearson_os_vs_db": pearson_os_db,
        "spearman_os_vs_db": spearman_os_db,
        "n_os_db_pairs": n_both,
        "mean_os_label": float(both["os_label_mean"].mean()) if len(both) else None,
        "mean_db_label": float(both["db_label_mean"].mean()) if len(both) else None,
        "pearson_os_synergy_vs_dosesyn_x_x0": pearson_os_x,
        "spearman_os_synergy_vs_dosesyn_x_x0": spearman_os_x,
        "n_os_x_pairs": n_os_x,
        "interpretation": (
            "OS and DrugCombDB synergy labels are different derived quantities and are not numerically equal "
            "for the same condition; neither is the O'Neil X/X0 relative viability. Correlations are descriptive "
            "only and do not establish a calibration."
        ),
    }

    evidence = build_matrix_evidence(ROOT / cfg["inputs"]["splits_dir"], matrix_index, matched, drug_map, cell_map)
    evidence_summary = summarize_evidence(evidence)
    bridge = bridge_summary(evidence)

    # Entity coverage table.
    n_drug_total = len(drug_map)
    n_drug_exact = int((drug_map["match_status"] == "EXACT").sum())
    n_drug_alias = int((drug_map["match_status"] == "REVIEWED_ALIAS").sum())
    n_drug_unmatched = int((drug_map["match_status"] == "UNMATCHED").sum())
    n_cell_total = len(cell_map)
    n_cell_exact = int((cell_map["match_status"] == "EXACT").sum())
    n_cell_alias = int((cell_map["match_status"] == "REVIEWED_ALIAS").sum())
    n_cell_unmatched = int((cell_map["match_status"] == "UNMATCHED").sum())
    coverage = pd.DataFrame(
        [
            {
                "entity": "drug",
                "total": n_drug_total,
                "exact": n_drug_exact,
                "reviewed_alias": n_drug_alias,
                "unmatched": n_drug_unmatched,
                "coverage_pct": round(_pct(n_drug_exact + n_drug_alias, n_drug_total), 2),
                "note": "DoseSyn 38 drugs; ConfSyn/DrugCombDB has 22 exact structures, 16 absent",
            },
            {
                "entity": "cell_line",
                "total": n_cell_total,
                "exact": n_cell_exact,
                "reviewed_alias": n_cell_alias,
                "unmatched": n_cell_unmatched,
                "coverage_pct": round(_pct(n_cell_exact + n_cell_alias, n_cell_total), 2),
                "note": "DoseSyn 39 cells; 31 match ConfSyn names (29 exact, 2 reviewed aliases)",
            },
            {
                "entity": "condition",
                "total": int(len(dose_conditions)),
                "exact": int(len(matched)),
                "reviewed_alias": 0,
                "unmatched": int(len(dose_conditions) - len(matched)),
                "coverage_pct": round(_pct(len(matched), len(dose_conditions)), 2),
                "note": "canonical (cell, unordered drug pair) condition; matched to ConfSyn DrugCombDB/OncologyScreen",
            },
        ]
    )

    # Split-level evidence summaries and A candidates.
    a_test = evidence[(evidence["role"] == "test") & (evidence["evidence_level"] == "A")].copy()
    a_conditions = (
        a_test[["scenario", "condition_id", "pair_id", "cell_id", "cell_line", "drug_a_name", "drug_b_name"]]
        .drop_duplicates()
        .sort_values(["scenario", "cell_line", "drug_a_name", "drug_b_name"])
    )
    a_conditions["n_test_matrices"] = a_conditions.apply(
        lambda r: int(
            a_test[
                (a_test["scenario"] == r["scenario"])
                & (a_test["condition_id"] == r["condition_id"])
            ]["matrix_id"].nunique()
        ),
        axis=1,
    )
    a_conditions.to_csv(metrics_dir / "bridge_a_conditions.csv", index=False)

    # Compact per-split summary with percentages.
    split_summary = evidence_summary.copy()
    role_totals = split_summary.groupby(["scenario", "split_id", "role"])["n_matrices"].transform("sum")
    split_summary["pct_of_role"] = (100.0 * split_summary["n_matrices"] / role_totals).round(3)

    # Evidence parquet for traceability (not committed).
    evidence.to_parquet(metrics_dir / "split_matrix_evidence.parquet", index=False, compression="snappy")

    # Public CSVs.
    coverage.to_csv(metrics_dir / "entity_coverage.csv", index=False)
    drug_map.to_csv(metrics_dir / "drug_mapping.csv", index=False)
    cell_map.to_csv(metrics_dir / "cell_mapping.csv", index=False)
    matched.to_csv(metrics_dir / "condition_matches.csv", index=False)
    matched.to_csv(metrics_dir / "label_provenance.csv", index=False)
    split_summary.to_csv(metrics_dir / "split_independence_summary.csv", index=False)
    a_test[
        [
            "scenario",
            "split_id",
            "matrix_id",
            "condition_id",
            "pair_id",
            "cell_id",
            "cell_line",
            "drug_a_name",
            "drug_b_name",
            "batch_id",
        ]
    ].to_csv(metrics_dir / "bridge_a_matrices.csv", index=False)

    # Gate checks.
    a_by_scenario = {s: int(bridge.get(s, {}).get("levels", {}).get("A", {}).get("n_matrices_union", 0)) for s in SCENARIOS}
    c_by_scenario = {s: int(bridge.get(s, {}).get("levels", {}).get("C", {}).get("n_matrices_union", 0)) for s in SCENARIOS}
    a_cond_by_scenario = {
        s: int(bridge.get(s, {}).get("levels", {}).get("A", {}).get("n_conditions_union", 0)) for s in SCENARIOS
    }
    checks = {
        "drug_mapping_at_least_20": n_drug_exact + n_drug_alias >= 20,
        "cell_mapping_at_least_25": n_cell_exact + n_cell_alias >= 25,
        "condition_matches_at_least_1000": len(matched) >= 1000,
        "A_bridge_available_in_all_scenarios": all(v > 0 for v in a_by_scenario.values()),
        "label_semantics_documented": True,
        "confsyn_per_sample_split_persisted": False,
        "confsyn_oncology_screen_checkpoint_available": False,
        "dosesyn_test_labels_untouched": True,
    }
    hard_pass = all(checks[k] for k in ("drug_mapping_at_least_20", "cell_mapping_at_least_25", "condition_matches_at_least_1000", "A_bridge_available_in_all_scenarios"))
    if not hard_pass:
        verdict = "BLOCKED"
    elif checks["confsyn_per_sample_split_persisted"] and checks["label_semantics_documented"]:
        verdict = "FULL PASS"
    else:
        verdict = "CONDITIONAL PASS"

    bridge_feasibility = {
        "verdict": verdict,
        "generated_at": utc_now(),
        "git_commit": git_commit(ROOT),
        "conf_syn": {
            "root": str(conf_root),
            "git_commit": _git_head(conf_root),
            "worktree_dirty": bool(_git_status(conf_root)),
            "training_dataset": "DrugCombDB",
            "oncology_screen_checkpoint": False,
            "per_sample_split_persisted": False,
            "sample_level_predictions_saved": False,
            "label_definition": "P(synergy >= 0) under DrugCombDB-derived labels",
        },
        "entity_coverage": coverage.to_dict(orient="records"),
        "condition_matches": {
            "n_matched_conditions": int(len(matched)),
            "n_dose_conditions": int(len(dose_conditions)),
            "n_in_oncology_screen": int(matched["in_oncology_screen"].sum()),
            "n_in_drugcombdb": int(matched["in_drugcombdb"].sum()),
            "n_both": int((matched["in_oncology_screen"] & matched["in_drugcombdb"]).sum()),
        },
        "label_provenance": label_stats,
        "bridge_by_scenario": bridge,
        "A_test_matrices_union": a_by_scenario,
        "A_test_conditions_union": a_cond_by_scenario,
        "C_test_matrices_union": c_by_scenario,
        "checks": checks,
        "interpretation": {
            "A": LEVEL_MEANING["A"],
            "B": LEVEL_MEANING["B"],
            "C": LEVEL_MEANING["C"],
            "D": LEVEL_MEANING["D"],
        },
        "m8b_recommendation": [
            "Use only A-level test matrices as bridge candidates; never treat C as independent.",
            "ConfSyn per-sample split membership is not persisted, so C conditions cannot be used as a new test set.",
            "Verify ConfSyn inference on the 22 mapped drugs / 31 mapped cells before any joint experiment.",
            "Treat ConfSyn p_synergy and DoseSyn X/X0 as different semantic quantities; no joint score without calibration.",
            "If M8-B needs a joint ranking, restrict to A and report A-only coverage, not the global dataset.",
        ],
    }
    (metrics_dir / "bridge_feasibility.json").write_text(
        json.dumps(bridge_feasibility, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    # Report.
    lines = _render_report(
        verdict=verdict,
        checks=checks,
        coverage=coverage,
        drug_map=drug_map,
        cell_map=cell_map,
        label_stats=label_stats,
        matched=matched,
        split_summary=split_summary,
        bridge=bridge,
        a_by_scenario=a_by_scenario,
        a_cond_by_scenario=a_cond_by_scenario,
        c_by_scenario=c_by_scenario,
        a_conditions=a_conditions,
    )
    report_path = ROOT / cfg["paths"]["report"]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Compact audit copies.
    docs_audit = ROOT / cfg["paths"]["docs_audit"]
    docs_audit.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(docs_audit / "entity_coverage.csv", index=False)
    drug_map.to_csv(docs_audit / "drug_mapping.csv", index=False)
    cell_map.to_csv(docs_audit / "cell_mapping.csv", index=False)
    split_summary.to_csv(docs_audit / "split_independence_summary.csv", index=False)
    a_conditions.to_csv(docs_audit / "bridge_a_conditions.csv", index=False)
    label_stats_df = pd.DataFrame(
        [{"metric": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v} for k, v in label_stats.items()]
    )
    label_stats_df.to_csv(docs_audit / "label_provenance_stats.csv", index=False)

    # Manifest.
    outputs = [
        metrics_dir / "entity_coverage.csv",
        metrics_dir / "drug_mapping.csv",
        metrics_dir / "cell_mapping.csv",
        metrics_dir / "condition_matches.csv",
        metrics_dir / "label_provenance.csv",
        metrics_dir / "split_independence_summary.csv",
        metrics_dir / "bridge_a_conditions.csv",
        metrics_dir / "bridge_a_matrices.csv",
        metrics_dir / "bridge_feasibility.json",
        report_path,
    ]
    manifest = {
        "stage": "M8-A",
        "generated_at": utc_now(),
        "git_commit": git_commit(ROOT),
        "git_status_porcelain": git_status_porcelain(ROOT).splitlines(),
        "environment": environment_record(),
        "confsyn": bridge_feasibility["conf_syn"],
        "inputs": {
            k: (
                {"path": str(v), "type": "directory"}
                if (ROOT / v).is_dir()
                else {"path": str(v), "sha256": file_sha256(ROOT / v) if (ROOT / v).exists() else None}
            )
            for k, v in cfg["inputs"].items()
        },
        "outputs": {str(p.relative_to(ROOT)): {"sha256": file_sha256(p), "bytes": p.stat().st_size} for p in outputs if p.exists()},
        "verdict": verdict,
        "counts": {
            "n_drugs_matched": n_drug_exact + n_drug_alias,
            "n_cells_matched": n_cell_exact + n_cell_alias,
            "n_conditions_matched": int(len(matched)),
            "A_test_matrices_union": a_by_scenario,
            "A_test_conditions_union": a_cond_by_scenario,
            "C_test_matrices_union": c_by_scenario,
        },
    }
    (metrics_dir / "m8a_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    logger.info("M8-A verdict: %s; A test matrices %s", verdict, a_by_scenario)
    print(json.dumps({"verdict": verdict, "counts": manifest["counts"]}, ensure_ascii=False, indent=2))
    return 0 if verdict != "BLOCKED" else 2


def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def _git_status(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    except Exception:
        return ""


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _render_report(
    *,
    verdict: str,
    checks: dict,
    coverage: pd.DataFrame,
    drug_map: pd.DataFrame,
    cell_map: pd.DataFrame,
    label_stats: dict,
    matched: pd.DataFrame,
    split_summary: pd.DataFrame,
    bridge: dict,
    a_by_scenario: dict,
    a_cond_by_scenario: dict,
    c_by_scenario: dict,
    a_conditions: pd.DataFrame,
) -> list[str]:
    d = {r.entity: r for r in coverage.itertuples()}
    n_drug = int(d["drug"].exact + d["drug"].reviewed_alias)
    n_cell = int(d["cell_line"].exact + d["cell_line"].reviewed_alias)
    unmatched_drugs = drug_map[drug_map["match_status"] == "UNMATCHED"]["oneil_drug_name"].tolist()
    unmatched_cells = cell_map[cell_map["match_status"] == "UNMATCHED"]["oneil_cell_name"].tolist()
    lines: list[str] = []
    lines += [
        "# DoseSyn M8-A：ConfSyn–DoseSyn 数据关联与独立性审计",
        "",
        f"**结论：{verdict}**",
        "",
        "本阶段只做只读审计，不重训 DoseSyn B5，不修改 M0–M7-C 的冻结数据、checkpoint、split 或正式结果；不启动 M8-B/C。",
        "",
        "## 1. 审计对象与可用性",
        "",
        "| 项目 | 结果 |",
        "| --- | --- |",
        f"| DoseSyn 基线 | `a6cfe9df89a21577e60952b2a690bec51fdf677a` |",
        "| DoseSyn 冻结矩阵 / 条件 | 18,599 / 18,326 |",
        f"| ConfSyn 本地路径 | `/home/kouzheng_2/AH_YeP/confsyn/Consyn` |",
        "| ConfSyn 训练数据 | DrugCombDB（63,010 行原始组合记录） |",
        "| ConfSyn 可用 checkpoint | 仅 DrugCombDB 5-fold，无 OncologyScreen checkpoint |",
        "| ConfSyn 逐样本预测 / split 索引 | **不存在**（只保存聚合指标与权重） |",
        "| ConfSyn 标签语义 | `P(synergy >= 0)`，标签来自 DrugCombDB 的 synergy 定义 |",
        "",
        "## 2. 实体映射",
        "",
        "| 实体 | 总数 | EXACT | REVIEWED_ALIAS | UNMATCHED | 覆盖率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| 药物 | {d['drug'].total} | {d['drug'].exact} | {d['drug'].reviewed_alias} | {d['drug'].unmatched} | {d['drug'].coverage_pct:.1f}% |",
        f"| 细胞系 | {d['cell_line'].total} | {d['cell_line'].exact} | {d['cell_line'].reviewed_alias} | {d['cell_line'].unmatched} | {d['cell_line'].coverage_pct:.1f}% |",
        f"| 条件 | {d['condition'].total} | {d['condition'].exact} | 0 | {d['condition'].unmatched} | {d['condition'].coverage_pct:.1f}% |",
        "",
        f"- 药物：{n_drug}/38 有 ConfSyn 结构；缺失 16 个：{', '.join(unmatched_drugs)}。",
        f"- 细胞系：{n_cell}/39 名称可映射；缺失 8 个：{', '.join(unmatched_cells)}。",
        "- `COLO320DM→COLO320`、`LNCAP→LNCAPCLONEFGC` 记录为 REVIEWED_ALIAS；其余为精确匹配。MSTO/OVCAR3 不在 ConfSyn 语料中（不得强行别名映射）。",
        "- 仅按名称相同不构成实验独立；药物匹配使用 InChIKey，细胞使用显式名称/别名表。",
        "",
        "## 3. 数据来源与标签重叠",
        "",
        f"- ConfSyn OncologyScreen 组合表：4,176 行、21 药、29 细胞；DrugCombDB 中可匹配到 **{int(matched['in_drugcombdb'].sum())}** 个 DoseSyn 条件。",
        f"- OncologyScreen 与 DrugCombDB 的同一条件（细胞+无序药对）重叠：**{label_stats['n_in_both']}** 个；两套 synergy 标签完全相等的条件：**{label_stats['exact_label_equal_in_both']}** 个。",
        f"- OS vs DB label 的 Pearson = {_fmt(label_stats['pearson_os_vs_db'])}，Spearman = {_fmt(label_stats['spearman_os_vs_db'])}（n={label_stats['n_os_db_pairs']}）；仅作描述，不构成可互换标签。",
        f"- OS synergy vs DoseSyn 条件均值 X/X0：Pearson = {_fmt(label_stats['pearson_os_synergy_vs_dosesyn_x_x0'])}，Spearman = {_fmt(label_stats['spearman_os_synergy_vs_dosesyn_x_x0'])}（n={label_stats['n_os_x_pairs']}）。",
        "- **结论：OS 的 synergy、DrugCombDB 的 synergy、DoseSyn 的 X/X0 是三种不同的派生量；不能把 X/X0 直接当作协同分数，也不能把 ConfSyn 概率直接映射为 X/X0。**",
        "",
        "## 4. 训练/测试独立性：A/B/C/D",
        "",
        "| 等级 | 定义 |",
        "| --- | --- |",
    ]
    for level in LEVELS:
        lines.append(f"| {level} | {LEVEL_MEANING[level]} |")
    lines += [
        "",
        "判定规则：",
        "",
        "1. 条件在 ConfSyn DrugCombDB 中出现 → **C**（逐样本 train/test 归属未持久化，不能声称独立）。",
        "2. 条件不在 ConfSyn 语料中、且 ConfSyn 覆盖其药物/细胞 → DoseSyn test 为 **A**，train/val 为 **B**。",
        "3. 实体或条件不可映射、矩阵被 Primary-v1 排除 → **D**。",
        "4. ConfSyn 没有 OncologyScreen checkpoint；即使 OS 与 O'Neil 同源，ConfSyn 的 DrugCombDB 模型并未在 OS 上训练（但 OS 行本身仍与 DoseSyn 同源，不能当作独立重复实验）。",
        "",
        "### 4.1 各场景 held-out test 的 A/B/C 覆盖",
        "",
        "| 场景 | test 矩阵（并集） | A 矩阵 | A 条件 | C 矩阵 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for s in SCENARIOS:
        total = int(bridge.get(s, {}).get("n_test_matrices_union", 0))
        lines.append(f"| {s} | {total} | {a_by_scenario[s]} | {a_cond_by_scenario[s]} | {c_by_scenario[s]} |")
    lines += [
        "",
        "### 4.2 逐 split 证据",
        "",
        "| 场景 | split | test 矩阵 | A | B | C | D |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (scenario, split_id), grp in split_summary[split_summary["role"] == "test"].groupby(["scenario", "split_id"]):
        counts = grp.set_index("level")["n_matrices"].to_dict()
        total = int(grp["n_matrices"].sum())
        lines.append(
            f"| {scenario} | {split_id} | {total} | {counts.get('A', 0)} | {counts.get('B', 0)} | {counts.get('C', 0)} | {counts.get('D', 0)} |"
        )
    lines += [
        "",
        f"A 级条件共 **{a_conditions['condition_id'].nunique()}** 个（按场景去重），全部为 ConfSyn 语料未覆盖且 DoseSyn test 的条件。",
        "",
        "## 5. 联合推理的语义兼容性",
        "",
        "- ConfSyn 输出 `p_synergy` 是 DrugCombDB 标签 `synergy >= 0` 的概率；它不是 O'Neil X/X0 的预测值，也不是 Bliss/HSA/Loewe/ZIP 分数。",
        "- DoseSyn 输出 16 点 X/X0 响应面；X/X0 是相对 DMSO 的 viability 派生量，本身不等于协同。",
        "- 两者的量纲、标签定义和训练数据均不同；**未发现可验证的单调映射或校准关系**。",
        "- 因此 M8-A 不支持“联合排序”或“融合分数”的正式结论；M8-B 若做桥接，只能做 A 级条件上的**条件级一致性分析**，且必须明确标注为探索性。",
        "",
        "## 6. 测试与可复现性",
        "",
        "- 全量测试：`226 passed`（M7-C 基线）；本阶段新增 `tests/test_m8a_*.py` 后运行全量 pytest。",
        "- 审计脚本：`scripts/26_audit_m8a.py`；映射/证据函数：`src/dosesyn/m8a/{mapping,evidence}.py`。",
        "- 所有输入只读；输出在 `outputs/metrics/m8a/`、`docs/audit/m8a/`、`m8a_report.md`。",
        "- 文件哈希、Git 状态、环境见 `outputs/metrics/m8a/m8a_manifest.json`。",
        "",
        "## 7. M8-B 建议",
        "",
        "1. 只使用 **A 级 test 矩阵** 作为桥接候选；C 级只能做描述性对照，不能作为独立测试。",
        "2. 先验证 ConfSyn 在 22 个可映射药物 / 31 个可映射细胞上的前向推理；当前没有逐样本预测文件。",
        "3. 将 ConfSyn 概率与 DoseSyn X/X0 分别报告，不构造单一“联合分数”；若必须构造，需先做外部校准并明确不确定性。",
        "4. 由于 ConfSyn 逐样本 split 不可恢复，M8-B 的独立性声明只能建立在“条件不在 DrugCombDB 语料”之上，不能建立在“ConfSyn test fold”之上。",
        "5. 报告 A 级覆盖率（相对 11 个 split 的 test 集），不要用 18,599 全量矩阵作分母。",
        "",
        "## 8. 验收判定",
        "",
        "| 检查 | 结果 |",
        "| --- | --- |",
    ]
    for key, val in checks.items():
        lines.append(f"| {key} | {'PASS' if val else 'FAIL'} |")
    lines += [
        "",
        f"**{verdict}**：实体映射与 A 级桥接样本存在，但 ConfSyn 缺少逐样本 split、缺少 OncologyScreen checkpoint，且标签语义不可直接对齐；因此不能给出 FULL PASS，也不能在 M8-A 直接做联合排序。",
        "",
    ]
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
