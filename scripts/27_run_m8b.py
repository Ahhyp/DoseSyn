"""Build M8-A A-level S2 candidates, score them with frozen ConfSyn folds, and
join frozen DoseSyn B5 4x4 predictions.

Inference only. No model is trained or fine-tuned; no M0--M8-A artefact is edited.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.utils.config import ensure_directories, load_config, resolve_path
from dosesyn.utils.io import environment_snapshot, sha256_file, utc_now_iso, write_json
from dosesyn.utils.logging import setup_logging


def load_inputs(cfg: dict):
    paths = cfg["paths"]
    evidence = pd.read_parquet(resolve_path(cfg, paths["m8a_evidence"]))
    drugs = pd.read_csv(resolve_path(cfg, paths["m8a_drug_mapping"]))
    cells = pd.read_csv(resolve_path(cfg, paths["m8a_cell_mapping"]))
    return evidence, drugs, cells


def select_candidates(cfg: dict, evidence: pd.DataFrame) -> pd.DataFrame:
    q = evidence[
        (evidence["scenario"] == cfg["scenario"])
        & (evidence["split_id"] == cfg["split_id"])
        & (evidence["role"] == "test")
        & (evidence["evidence_level"] == cfg["evidence_level"])
    ].copy()
    return q


def attach_entities(q, drugs, cells):
    db = {}
    for _, r in drugs.iterrows():
        if isinstance(r.get("conf_dbname"), str) and str(r["conf_dbname"]).startswith("DB"):
            db[r["oneil_drug_name"]] = r["conf_dbname"]
    cm = {}
    for _, r in cells.iterrows():
        if isinstance(r.get("conf_cell_name"), str) and r["conf_cell_name"]:
            cm[r["oneil_cell_name"]] = r["conf_cell_name"]

    q["conf_db_a"] = q["drug_a_name"].map(db)
    q["conf_db_b"] = q["drug_b_name"].map(db)
    q["conf_cell_name"] = q["cell_line"].map(cm)
    missing = q[["conf_db_a", "conf_db_b", "conf_cell_name"]].isna().any(axis=1)
    if missing.any():
        raise RuntimeError(f"{int(missing.sum())} candidates lack a ConfSyn entity:\n"
                           f"{q.loc[missing, ['matrix_id', 'drug_a_name', 'drug_b_name', 'cell_line']].head()}")
    return q


def dosesyn_surface_stats(cfg: dict, matrix_ids):
    run_dir = resolve_path(cfg, cfg["paths"]["dosesyn_run"])
    pred_path = run_dir / "predictions_test.parquet"
    pred = pd.read_parquet(pred_path)
    pred = pred[pred["matrix_id"].isin(matrix_ids)].copy()
    if pred.empty:
        raise RuntimeError("no frozen DoseSyn predictions for A-level candidates")
    pred["held"] = True
    stats = (
        pred.groupby("matrix_id")
        .agg(
            n_points=("y_pred", "size"),
            y_pred_mean=("y_pred", "mean"),
            y_pred_median=("y_pred", "median"),
            y_pred_min=("y_pred", "min"),
            y_pred_max=("y_pred", "max"),
            y_true_mean=("y_true", "mean"),
            y_true_max=("y_true", "max"),
            y_pred_std=("y_pred", "std"),
        )
        .reset_index()
    )
    # 4x4 row-major surface, retained for the heat-map figure.
    surfaces = []
    for mid, g in pred.groupby("matrix_id"):
        g = g.sort_values(["row_idx", "col_idx"])
        if len(g) != 16 or g[["row_idx", "col_idx"]].drop_duplicates().shape[0] != 16:
            raise RuntimeError(f"matrix {mid} does not have a unique 4x4 surface")
        surfaces.append({
            "matrix_id": mid,
            "surface": g["y_pred"].to_numpy(dtype=float).reshape(4, 4).tolist(),
            "truth": g["y_true"].to_numpy(dtype=float).reshape(4, 4).tolist(),
            "dose_a_um": np.sort(g["dose_a_um"].unique()).tolist(),
            "dose_b_um": np.sort(g["dose_b_um"].unique()).tolist(),
        })
    return stats, pd.DataFrame(surfaces)


def main() -> int:
    cfg = load_config(ROOT / "configs" / "m8b" / "m8b.yaml")
    for key in ("m8a_evidence", "m8a_drug_mapping", "m8a_cell_mapping", "dosesyn_run",
                "outputs_metrics", "outputs_figures", "outputs_audit", "report"):
        cfg["paths"][key] = str(resolve_path(cfg, cfg["paths"][key]))
    ensure_directories(cfg)
    for key in ("outputs_metrics", "outputs_figures", "outputs_audit"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    logger = setup_logging("dosesyn.m8b", ROOT / "outputs/logs/27_run_m8b.log")

    evidence, drugs, cells = load_inputs(cfg)
    candidates = attach_entities(select_candidates(cfg, evidence), drugs, cells)
    candidates = candidates.drop_duplicates("matrix_id").reset_index(drop=True)
    logger.info("A-level %s %s candidates: %d matrices", cfg["scenario"], cfg["split_id"], len(candidates))

    from dosesyn.m8b.confsyn_bridge import ConfSynBridge

    bridge = ConfSynBridge(cfg["confsyn_root"], folds=cfg["folds"], device="cpu")
    logger.info("node counts: %s; folds: %s", bridge.node_counts, cfg["folds"])

    scoreable = (
        candidates["conf_db_a"].isin(bridge.drug_index)
        & candidates["conf_db_b"].isin(bridge.drug_index)
        & candidates["conf_cell_name"].isin(bridge.cell_index)
    )
    excluded = candidates[~scoreable].copy()
    excluded["exclusion_reason"] = "drug_or_cell_absent_from_DrugCombDB_graph"
    excluded_path = Path(cfg["paths"]["outputs_metrics"]) / "confsyn_unscoreable_candidates.csv"
    excluded.to_csv(excluded_path, index=False)
    candidates = candidates[scoreable].reset_index(drop=True)
    logger.info("scoreable candidates: %d; excluded (not in DrugCombDB graph): %d", len(candidates), len(excluded))
    if candidates.empty:
        raise RuntimeError("no A-level candidate is scoreable by the DrugCombDB ConfSyn checkpoint")

    validation = bridge.sanity_check(n=cfg["sanity_n"], seed=cfg["sanity_seed"])
    out_metrics = Path(cfg["paths"]["outputs_metrics"])
    validation.to_csv(out_metrics / "confsyn_checkpoint_validation.csv", index=False)
    logger.info("checkpoint label-reconstruction AUC: %s", validation["auc"].round(4).tolist())

    scored = bridge.score_conditions(candidates[[
        "matrix_id", "condition_id", "cell_line", "drug_a_name", "drug_b_name",
        "conf_cell_name", "conf_db_a", "conf_db_b",
    ]])
    scored.to_csv(out_metrics / "confsyn_scores_s2_seed42.csv", index=False)
    logger.info("ConfSyn scored %d candidates x %d folds", len(scored), len(cfg["folds"]))

    stats, surfaces = dosesyn_surface_stats(cfg, set(scored["matrix_id"]))
    joined = scored.merge(stats, on="matrix_id", how="left", validate="one_to_one")
    joined["scenario"] = cfg["scenario"]
    joined["split_id"] = cfg["split_id"]
    joined["evidence_level"] = cfg["evidence_level"]
    if joined["n_points"].isna().any():
        raise RuntimeError("some candidates have no DoseSyn surface")
    joined["conf_rank"] = joined["p_mean"].rank(ascending=False, method="min").astype(int)
    joined["dosesyn_response_rank"] = joined["y_pred_mean"].rank(ascending=True, method="min").astype(int)
    joined["dosesyn_surface_rank"] = joined["y_pred_min"].rank(ascending=True, method="min").astype(int)
    joined = joined.sort_values("p_mean", ascending=False).reset_index(drop=True)
    joined.to_csv(out_metrics / "bridge_candidates.csv", index=False)

    top = joined.head(30).copy()
    top.to_csv(out_metrics / "joint_ranking_top30.csv", index=False)
    surfaces.to_json(out_metrics / "dosesyn_surfaces.json", orient="records", indent=2)

    corr = joined[["p_mean", "y_pred_mean", "y_pred_min", "p_std"]].corr(method="spearman")
    summary = {
        "generated_at": utc_now_iso(),
        "stage": "M8-B",
        "scenario": cfg["scenario"],
        "split_id": cfg["split_id"],
        "candidate_matrices": int(len(joined)),
        "a_candidates_before_entity_filter": int(len(candidates) + len(excluded)),
        "unscoreable_candidates": int(len(excluded)),
        "confsyn_folds": cfg["folds"],
        "confsyn_checkpoint_validation": validation.to_dict(orient="records"),
        "p_mean": {
            "min": float(joined["p_mean"].min()),
            "median": float(joined["p_mean"].median()),
            "max": float(joined["p_mean"].max()),
            "n_ge_0_5": int((joined["p_mean"] >= 0.5).sum()),
            "n_ge_0_9": int((joined["p_mean"] >= 0.9).sum()),
        },
        "p_std": {
            "mean": float(joined["p_std"].mean()),
            "max": float(joined["p_std"].max()),
        },
        "dosesyn": {
            "y_pred_mean_min": float(joined["y_pred_mean"].min()),
            "y_pred_mean_median": float(joined["y_pred_mean"].median()),
            "y_pred_mean_max": float(joined["y_pred_mean"].max()),
            "y_pred_min_min": float(joined["y_pred_min"].min()),
        },
        "spearman": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in corr.to_dict().items()},
        "sources": {
            "evidence": cfg["paths"]["m8a_evidence"],
            "evidence_sha256": sha256_file(resolve_path(cfg, cfg["paths"]["m8a_evidence"])),
            "dosesyn_predictions": str(resolve_path(cfg, cfg["paths"]["dosesyn_run"]) / "predictions_test.parquet"),
            "dosesyn_predictions_sha256": sha256_file(resolve_path(cfg, cfg["paths"]["dosesyn_run"]) / "predictions_test.parquet"),
            "confsyn_checkpoints": [str(bridge.checkpoint_path(f)) for f in cfg["folds"]],
        },
        "environment": environment_snapshot(),
        "notes": [
            "ConfSyn p is the positive-class probability for DrugCombDB label synergy>=0; it is not X/X0 and not a standard synergy score.",
            "All candidates are M8-A evidence level A (absent from DrugCombDB), on the DoseSyn S2 seed42 test split.",
            "DoseSyn B5 is frozen; predictions come from the existing M6-B run, not a new inference pass.",
            "ConfSyn official per-sample split indices are not persisted; checkpoint validation is a label-reconstruction check, not an independent test metric.",
        ],
    }
    write_json(summary, out_metrics / "m8b_summary.json")

    report = build_report(cfg, summary, joined, validation)
    (ROOT / cfg["paths"]["report"]).write_text(report, encoding="utf-8")
    logger.info("M8-B wrote %d candidates; summary saved", len(joined))
    print(json.dumps({k: summary[k] for k in ("candidate_matrices", "p_mean", "p_std", "dosesyn", "spearman")}, indent=2))
    return 0


def build_report(cfg, summary, joined, validation) -> str:
    v = validation
    top = joined.head(15)
    lines = [
        "# M8-B：ConfSyn–DoseSyn 冻结模型联合筛选与剂量响应可视化",
        "",
        "## 1. 结论",
        "",
        f"本阶段在 **{cfg['scenario']} / {cfg['split_id']}** 的 **A 级候选（DrugCombDB 中不存在的组合条件）** 上完成双模型推理：",
        f"共 **{summary['candidate_matrices']}** 个组合条件进入联合表，ConfSyn 五折协同概率均值 "
        f"`{summary['p_mean']['min']:.4f}`–`{summary['p_mean']['max']:.4f}`（中位数 `{summary['p_mean']['median']:.4f}`），"
        f"DoseSyn B5 冻结 4×4 曲面均值响应 `{summary['dosesyn']['y_pred_mean_min']:.4f}`–`{summary['dosesyn']['y_pred_mean_max']:.4f}`。",
        "",
        "> ⚠️ 语义边界：ConfSyn 输出的是 DrugCombDB 定义下 `synergy >= 0` 的正类概率，"
        "**不是** Bliss/HSA/Loewe/ZIP 协同分数；DoseSyn 输出的是原始 `X/X0` 剂量响应。"
        "本阶段只做“联合筛选展示”，不构建新的协同评分器，也不把 `X/X0` 解释为协同。",
        "",
        "## 2. 冻结 ConfSyn 资产恢复",
        "",
        "| fold | checkpoint | n | AUC（标签重建） | ACC | p(正类均值) | p(负类均值) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in v.iterrows():
        lines.append(
            f"| {int(r.fold)} | `{Path(r.checkpoint).name}` | {int(r.n)} | {r.auc:.4f} | {r.accuracy:.4f} | "
            f"{r.mean_p_positive:.4f} | {r.mean_p_negative:.4f} |"
        )
    lines += [
        "",
        "复现要点：ConfSyn 使用 DrugCombDB 的 68 个细胞系、764 个药物、15,970 个蛋白节点；"
        "图特征为 `torch.manual_seed(42)` 生成的 300 维随机特征；HGT 为 2 层、4 头、768 维；"
        "药物侧使用仓库缓存的 ChemBERTa 256 维投影，细胞侧使用 16,383 维表达谱。"
        "五折 checkpoint 均以 `strict=False` 加载，除冻结的 RoBERTa 权重外无缺失/多余参数。",
        "",
        "> 这些随机 DrugCombDB 行并非独立测试集（官方逐样本划分索引未持久化），"
        "AUC 只说明**预处理、实体索引与 checkpoint 恢复一致**，不能当作泛化性能。",
        "",
        "## 3. 联合筛选表（按 ConfSyn 概率排序）",
        "",
        "| # | 矩阵 | 细胞系 | 药物对 | ConfSyn p (mean±std) | DoseSyn 曲面均值 | 曲面最小 |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for i, r in top.iterrows():
        lines.append(
            f"| {i + 1} | `{r.matrix_id}` | {r.cell_line} | {r.drug_a_name} + {r.drug_b_name} | "
            f"{r.p_mean:.4f} ± {r.p_std:.4f} | {r.y_pred_mean:.4f} | {r.y_pred_min:.4f} |"
        )
    s = summary["spearman"]
    lines += [
        "",
        f"ConfSyn p 与 DoseSyn 曲面均值响应的 Spearman 相关为 `{s['p_mean']['y_pred_mean']:.3f}`，"
        f"与曲面最小值的 Spearman 相关为 `{s['p_mean']['y_pred_min']:.3f}`。"
        "相关较弱时，说明两个模型对“组合优先级”的看法并不一致，这本身就是需要如实报告的结果。",
        "",
        "## 4. 交付物",
        "",
        "- `outputs/metrics/m8b/confsyn_checkpoint_validation.csv`：五折标签重建检查",
        "- `outputs/metrics/m8b/confsyn_scores_s2_seed42.csv`：A 级候选五折概率",
        "- `outputs/metrics/m8b/bridge_candidates.csv`：联合筛选主表（含曲面统计与排名）",
        "- `outputs/metrics/m8b/joint_ranking_top30.csv`：Top-30 排序",
        "- `outputs/metrics/m8b/dosesyn_surfaces.json`：Top 候选 4×4 冻结曲面",
        "- `outputs/figures/m8b/`：双维度散点图、Top 候选剂量响应热力图",
        "- `outputs/audit/m8b/`：紧凑审计副本、环境与来源哈希",
        "",
        "## 5. 局限",
        "",
        "1. ConfSyn 的五折逐样本划分未持久化，A 级判定只能基于“该条件不在 DrugCombDB 表中”，无法进一步证明它不在某个训练折。",
        "2. 本阶段只展示 S2 seed42 的 750 个 A 级候选，未做 S1/S3 的联合筛选；跨场景结论需在 M8-C 另行验证。",
        "3. ConfSyn 概率是组合级二分类概率，不含剂量；DoseSyn 曲面是剂量级响应，不含协同标签，二者不是同一目标。",
        "4. 未重新训练、未微调、未使用测试标签选择候选。",
        "",
        f"_生成时间：{summary['generated_at']}_",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
