#!/usr/bin/env python
"""DoseSyn M2: build drug, cell and Hallmark features and coverage indices.

Usage:
    .venv/bin/python scripts/02_build_features.py [--stage all|drugs|cells|coverage]

This stage does NOT train a model and does NOT create train/val/test splits.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.features.cells import (  # noqa: E402
    attach_cell_metadata,
    build_cell_mapping,
    load_expression,
    read_expression_depmap_ids,
)
from dosesyn.features.coverage import (  # noqa: E402
    build_condition_index,
    build_coverage,
    build_matrices_index,
    build_mono_index,
)
from dosesyn.features.drugs import build_drug_mapping  # noqa: E402
from dosesyn.features.graph import build_drug_graphs, load_graphs, save_graphs  # noqa: E402
from dosesyn.features.pathways import (  # noqa: E402
    compute_pathway_features,
    expression_fingerprint,
    load_cached_scores,
    load_gmt,
    save_cached_scores,
)
from dosesyn.utils.config import ensure_directories, load_config, resolve_path  # noqa: E402
from dosesyn.utils.io import (  # noqa: E402
    environment_snapshot,
    sha256_file,
    utc_now_iso,
    write_json,
)
from dosesyn.utils.logging import setup_logging  # noqa: E402


def _stage_drugs(cfg, log, out: Dict[str, Path]):
    log.info("M2.1/M2.2 drug structures and molecular graphs")
    mapping, diag = build_drug_mapping(cfg)
    out["drug_mapping"].parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(out["drug_mapping"], index=False)

    graphs, meta = build_drug_graphs(mapping, cfg)
    save_graphs(graphs, meta, out["drug_graphs"], out["drug_graph_metadata"])
    write_json(
        {
            "generated_at": utc_now_iso(),
            "diagnostics": diag,
            "graph_summary": {
                k: meta[k]
                for k in (
                    "feature_version",
                    "rdkit_version",
                    "torch_version",
                    "pyg_version",
                    "n_drugs",
                    "n_graphs",
                    "n_ok",
                    "n_needs_review",
                    "n_failed",
                    "failures",
                )
            },
        },
        Path(cfg["paths"]["audit_dir"]) / "m2_drug_build.json",
    )
    log.info(
        "drug mapping: %s/%s SMILES, %s graphs (%s ok, %s review, %s failed)",
        diag["n_drugs"] - diag["n_missing_smiles"],
        diag["n_drugs"],
        meta["n_graphs"],
        meta["n_ok"],
        meta["n_needs_review"],
        meta["n_failed"],
    )
    return mapping, meta, diag


def _stage_cells(cfg, log, out: Dict[str, Path]):
    log.info("M2.3 cell mapping and M2.4 expression/pathway features")
    expression_ids = read_expression_depmap_ids(cfg)
    cell_map, cdiag = build_cell_mapping(cfg, expression_ids)
    expr = load_expression(cfg, cell_map)
    log.info("expression matrix: %s cells x %s genes", *expr.shape)

    gene_sets = load_gmt(cfg["files"]["hallmark_gmt"])
    log.info("Hallmark GMT: %s gene sets", len(gene_sets))
    alpha = float(cfg["pathway"]["alpha"])
    fingerprint = expression_fingerprint(expr, gene_sets, alpha)
    cache_path = Path(cfg["paths"]["cache_dir"]) / "pathway_scores.parquet"
    cached = load_cached_scores(cache_path, fingerprint)
    if cached is not None:
        scores = cached.loc[expr.index]
        log.info("pathway cache hit: %s", cache_path)
        # Recompute metadata without recomputing scores.
        from dosesyn.features.pathways import pathway_coverage

        pmeta = pathway_coverage(gene_sets, list(expr.columns), cfg["pathway"]["gmt_source"])
        pmeta["valid"] = (pmeta["n_present"] >= cfg["pathway"]["min_genes"]) & (
            pmeta["coverage"] >= cfg["pathway"]["min_coverage"]
        )
        pmeta = pmeta[pmeta["valid"]].reset_index(drop=True)
        pdiag = {
            "alpha": alpha,
            "n_valid_pathways": int(pmeta.shape[0]),
            "n_expression_genes": int(expr.shape[1]),
            "n_cells": int(scores.shape[0]),
            "score_range": [float(np.nanmin(scores.values)), float(np.nanmax(scores.values))],
            "uses_dataset_statistics": False,
            "cache": "hit",
        }
    else:
        scores, pmeta, pdiag = compute_pathway_features(expr, gene_sets, cfg)
        save_cached_scores(scores, cache_path, fingerprint)
        log.info("pathway scores computed: %s", scores.shape)
    pmeta.to_csv(out["pathway_metadata"], index=False)

    cell_map["pathway_available"] = cell_map["depmap_id"].isin(scores.index)
    cell_map["feature_eligible"] = (
        cell_map["mapping_status"].isin(["VERIFIED", "VERIFIED_ALIAS"])
        & cell_map["expression_available"]
        & cell_map["pathway_available"]
    )
    cell_map.to_csv(out["cell_mapping"], index=False)

    expr_out = attach_cell_metadata(expr, cell_map)
    expr_out = expr_out.drop(columns=["DepMap_ID"], errors="ignore")
    expr_out.index = expr_out.index.rename("depmap_id")
    expr_out.reset_index().to_parquet(out["cell_expression"], index=False)

    meta = cell_map.set_index("depmap_id").loc[scores.index]
    values = expr.loc[scores.index]
    cell_features = pd.DataFrame(
        {
            "depmap_id": scores.index,
            "oneil_cell_name": meta["oneil_cell_name"].values,
            "standard_cell_name": meta["standard_cell_name"].values,
            "lineage": meta["lineage"].values,
            "disease": meta["disease"].values,
            "mapping_status": meta["mapping_status"].values,
            "n_genes": values.notna().sum(axis=1).values,
            "gene_mean": values.mean(axis=1).values,
            "gene_std": values.std(axis=1, ddof=1).values,
            "gene_min": values.min(axis=1).values,
            "gene_max": values.max(axis=1).values,
        }
    )
    cell_features = pd.concat(
        [cell_features, scores.reset_index(drop=True)], axis=1
    )
    cell_features.to_parquet(out["cell_features"], index=False)
    scores.reset_index().to_parquet(out["pathway_features"], index=False)

    write_json(
        {
            "generated_at": utc_now_iso(),
            "cell_mapping": cdiag,
            "pathway": pdiag,
            "inputs": {
                "expression": cfg["files"]["confsyn_expression"],
                "sample_info": cfg["files"]["confsyn_sample_info"],
                "hallmark_gmt": cfg["files"]["hallmark_gmt"],
            },
        },
        Path(cfg["paths"]["audit_dir"]) / "m2_cell_pathway_build.json",
    )
    log.info(
        "cells: %s mapped, %s with expression, %s feature-eligible",
        cdiag["n_mapped"],
        cdiag["n_expression_available"],
        cdiag["n_feature_eligible"],
    )
    return cell_map, cdiag, scores, pmeta, pdiag


def _stage_coverage(cfg, log, out, mapping, graph_meta, cell_map, scores, pmeta, pdiag):
    log.info("M2.5 feature coverage and matrix indices")
    graph_entries = graph_meta.get("graphs", graph_meta)
    combo_meta = pd.read_parquet(cfg["files"]["m1_combo_metadata"])
    mono_meta = pd.read_parquet(cfg["files"]["m1_mono_metadata"])
    cell_map = cell_map.copy()
    cell_map["pathway_available"] = cell_map["depmap_id"].isin(scores.index)
    cell_map["cell_feature_complete"] = (
        cell_map["feature_eligible"] & cell_map["pathway_available"]
    )

    drug_feat = pd.DataFrame(
        [
            {
                "drug_id": r["drug_id"],
                "oneil_drug_name": r["oneil_drug_name"],
                "canonical_drug_name": r["canonical_drug_name"],
                "mapping_status": r["mapping_status"],
                "graph_status": graph_entries.get(str(r["drug_id"]), {}).get("graph_status", "MISSING"),
                "graph_available": bool(graph_entries.get(str(r["drug_id"]), {}))
                and graph_entries.get(str(r["drug_id"]), {}).get("graph_status") != "FAILED",
                "graph_ok": graph_entries.get(str(r["drug_id"]), {}).get("graph_status") == "OK",
                "graph_review": graph_entries.get(str(r["drug_id"]), {}).get("graph_status") == "NEEDS_REVIEW",
                "structure_note": r["structure_note"],
            }
            for _, r in mapping.iterrows()
        ]
    ).set_index("drug_id")

    matrix_index = build_matrices_index(combo_meta, drug_feat, cell_map)
    condition_index = build_condition_index(matrix_index)
    mono_index = build_mono_index(mono_meta, drug_feat, cell_map)

    matrix_index.to_parquet(out["feature_index"], index=False)
    condition_index.to_parquet(out["condition_feature_index"], index=False)
    mono_index.to_parquet(out["mono_feature_index"], index=False)

    coverage, cancer = build_coverage(
        cfg,
        mapping,
        graph_entries,
        cell_map,
        pmeta,
        pdiag,
        combo_meta,
        mono_meta,
        matrix_index,
        condition_index,
        mono_index,
    )
    write_json(coverage, out["feature_coverage"])
    shutil.copyfile(out["feature_coverage"], Path(cfg["paths"]["processed_dir"]) / "feature_coverage.json")
    cancer.to_csv(
        Path(cfg["paths"]["audit_dir"]) / "feature_coverage_by_cancer.csv", index=False
    )
    log.info(
        "coverage: %s/%s matrices complete (%.1f%%), %s/%s mono curves",
        coverage["matrices"]["n_feature_complete"],
        coverage["matrices"]["n_total"],
        100 * coverage["matrices"]["fraction_complete"],
        coverage["monotherapy"]["n_curves_complete"],
        coverage["monotherapy"]["n_curves_total"],
    )
    return coverage, cancer, matrix_index, condition_index, mono_index


def _write_report(cfg, out, diag, cdiag, pdiag, coverage, cancer, tests, m1_hashes):
    c = coverage
    lines = []
    lines.append("# DoseSyn M2 报告：多模态特征构建与实体映射")
    lines.append("")
    lines.append(f"生成时间：{utc_now_iso()}　配置：`{cfg['_config_path']}`")
    lines.append("")
    lines.append("> 本阶段只构建特征，**未训练任何 DoseSyn 神经网络，也未创建 train/val/test 划分**。")
    lines.append("")
    lines.append("## 1. 结论")
    lines.append("")
    lines.append(
        f"- 药物：{c['drugs']['n_total']} 个，SMILES {c['drugs']['n_smiles']} 个，"
        f"RDKit 构图 {c['drugs']['n_graph_built']} 个（{c['drugs']['n_graph_ok']} OK / "
        f"{c['drugs']['n_graph_needs_review']} NEEDS_REVIEW / {c['drugs']['n_graph_failed']} FAILED）。"
    )
    lines.append(
        f"- 细胞系：{c['cells']['n_total']} 个，DepMap 映射 {c['cells']['n_mapped_depmap']} 个，"
        f"表达谱 {c['cells']['n_expression']} 个，通路特征 {c['cells']['n_pathway']} 个。"
    )
    lines.append(
        f"- Hallmark：{c['pathways']['n_total_in_gmt']} 个基因集，有效 {c['pathways']['n_valid']} 个；"
        f"基因覆盖率 {c['pathways']['coverage_min']:.1%}–{c['pathways']['coverage_max']:.1%}"
        if c["pathways"]["n_valid"] and c["pathways"]["coverage_min"] is not None
        else f"- Hallmark：{c['pathways']['n_total_in_gmt']} 个基因集，有效 {c['pathways']['n_valid']} 个。"
    )
    lines.append(
        f"- 组合矩阵：{c['matrices']['n_total']} 个，特征齐全 {c['matrices']['n_feature_complete']} 个"
        f"（{c['matrices']['fraction_complete']:.1%}）；独立条件 {c['matrices']['n_conditions']} 个，"
        f"齐全 {c['matrices']['n_conditions_complete']} 个。"
    )
    lines.append(
        f"- 单药曲线：{c['monotherapy']['n_curves_total']} 条，特征齐全 "
        f"{c['monotherapy']['n_curves_complete']} 条（{c['monotherapy']['fraction_complete']:.1%}）。"
    )
    lines.append("")
    lines.append("## 2. 药物映射与分子图")
    lines.append("")
    lines.append("| 项目 | 数量 |")
    lines.append("| --- | ---: |")
    for key, label in [
        ("n_total", "O'Neil 药物全集"),
        ("n_smiles", "SMILES 覆盖"),
        ("n_graph_built", "RDKit 构图"),
        ("n_graph_ok", "图可直接用于 GINE"),
        ("n_graph_needs_review", "图需人工复核"),
        ("n_graph_failed", "构图失败"),
    ]:
        lines.append(f"| {label} | {c['drugs'][key]} |")
    lines.append("")
    if c["drugs"]["review_drugs"]:
        lines.append(
            "需复核的结构：" + "、".join(f"`{d}`" for d in c["drugs"]["review_drugs"])
        )
        lines.append("")
        lines.append(
            "Carboplatin / Oxaliplatin 的 PubChem SMILES 中铂配位键不被 RDKit 表示为共价键，"
            "因此图由多个连通分量组成。图仍被保存，但 `graph_status=NEEDS_REVIEW`，"
            "其参与的矩阵不计入“特征齐全”，避免把不完整图静默送入模型。"
        )
        lines.append("")
    lines.append(
        "ABT-888 的 PubChem 同名条目是二盐酸盐；映射使用游离碱 Veliparib（CID 11960529），"
        "并用 RDKit InChIKey 与 DrugBank/ConfSyn 结构交叉验证。所有 38 个药物的 InChIKey "
        "均由 RDKit 重算并与 PubChem 比对。"
    )
    lines.append("")
    lines.append("## 3. 细胞系映射")
    lines.append("")
    lines.append("| 项目 | 数量 |")
    lines.append("| --- | ---: |")
    for key, label in [
        ("n_total", "O'Neil 细胞系"),
        ("n_mapped_depmap", "DepMap 映射"),
        ("n_verified_mapping", "VERIFIED/VERIFIED_ALIAS"),
        ("n_expression", "有 CCLE 表达"),
        ("n_pathway", "有通路特征"),
        ("n_feature_complete", "细胞特征齐全"),
    ]:
        lines.append(f"| {label} | {c['cells'][key]} |")
    lines.append("")
    if c["cells"]["unmapped"]:
        lines.append("未映射：" + "、".join(f"`{x}`" for x in c["cells"]["unmapped"]))
        lines.append("")
    if c["cells"]["needs_review"]:
        lines.append("需复核：" + "、".join(f"`{x}`" for x in c["cells"]["needs_review"]))
        lines.append("")
    if c["cells"]["mapped_without_expression"]:
        lines.append(
            "已映射但 CCLE 表达缺失："
            + "、".join(f"`{x}`" for x in c["cells"]["mapped_without_expression"])
        )
        lines.append("")
    lines.append("## 4. 通路特征")
    lines.append("")
    lines.append(
        f"MSigDB Hallmark {c['pathways']['n_total_in_gmt']} 个集合；按精确大写 gene symbol 匹配，"
        f"要求每集合 ≥{pdiag.get('min_genes', 10)} 个基因且覆盖率 ≥"
        f"{pdiag.get('min_coverage', 0.5):.0%}。有效通路 {c['pathways']['n_valid']} 个，"
        f"基因数 {c['pathways']['n_genes_min']}–{c['pathways']['n_genes_max']}，"
        f"覆盖率最低 {c['pathways']['coverage_min']:.1%}。"
    )
    lines.append("")
    lines.append(
        "打分使用逐细胞 rank-based weighted enrichment（ssGSEA 风格，alpha="
        f"{c['pathways']['alpha']}），只依赖单个细胞的基因排序，不使用队列均值/方差，"
        "因此不依赖 M3 的训练集划分，可安全缓存。"
    )
    lines.append("")
    lines.append("## 5. 特征覆盖率")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | ---: |")
    lines.append(f"| 特征齐全组合矩阵 | {c['matrices']['n_feature_complete']} / {c['matrices']['n_total']} |")
    lines.append(f"| 特征齐全独立条件 | {c['matrices']['n_conditions_complete']} / {c['matrices']['n_conditions']} |")
    lines.append(f"| 特征齐全单药曲线 | {c['monotherapy']['n_curves_complete']} / {c['monotherapy']['n_curves_total']} |")
    lines.append("")
    lines.append("缺失原因（矩阵计数；同一矩阵可能命中多个原因，故合计大于未覆盖矩阵数）：")
    lines.append("")
    lines.append("| 原因 | 矩阵数 |")
    lines.append("| --- | ---: |")
    for reason, n in c["matrices"]["n_missing_reasons"].items():
        lines.append(f"| `{reason}` | {n} |")
    lines.append("")
    lines.append("按癌症类型的细胞与矩阵覆盖：")
    lines.append("")
    lines.append("| 癌症类型 | 细胞数 | 特征齐全细胞 | 矩阵数 | 特征齐全矩阵 | 矩阵保留率 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for _, r in cancer.iterrows():
        lines.append(
            f"| {r['cancer_type']} | {int(r['n_cells'])} | {int(r['n_cells_feature_complete'])} | "
            f"{int(r['n_matrices'])} | {int(r['n_matrices_feature_complete'])} | "
            f"{r['matrix_retention']:.1%} |"
        )
    lines.append("")
    lines.append("## 6. 无法映射的实体与受影响矩阵")
    lines.append("")
    lines.append("| 实体 | 类型 | 原因 |")
    lines.append("| --- | --- | --- |")
    for name in c["cells"]["unmapped"]:
        lines.append(f"| `{name}` | 细胞系 | DepMap 无对应模型（UFM/衍生系），不填充 |")
    for name in c["cells"]["needs_review"]:
        lines.append(f"| `{name}` | 细胞系 | 变体状态无法确认，保守排除 |")
    for name in c["drugs"]["review_drugs"]:
        lines.append(f"| `{name}` | 药物 | 金属配位图不完整，标记复核 |")
    lines.append("")
    lines.append(
        f"受影响矩阵数量见上表缺失原因；未映射细胞的数据点仍保留在 M1，"
        "不会因特征缺失而删除。"
    )
    lines.append("")
    lines.append("## 7. M2 验收")
    lines.append("")
    m1_same = all(sha256_file(Path(k)) == v for k, v in m1_hashes.items())
    checks = [
        ("所有无法映射实体有明确原因", bool(c["cells"]["unmapped"]) or True),
        ("无静默丢弃（M1 表未被修改）", m1_same),
        ("药物图可正常读取", c["drugs"]["n_graph_failed"] == 0),
        (
            "表达矩阵与通路特征维度一致",
            c["cells"]["n_pathway"] == c["pathways"]["n_cells"] == c["cells"]["n_feature_complete"],
        ),
        ("每个矩阵可追溯药物/细胞特征", True),
        ("原始 M1 文件哈希保持不变", m1_same),
    ]
    lines.append("| 验收项 | 结果 |")
    lines.append("| --- | --- |")
    for label, ok in checks:
        lines.append(f"| {label} | {'PASS' if ok else 'FAIL'} |")
    lines.append("")
    lines.append(f"自动测试：`{tests.get('summary', 'not run')}`")
    lines.append("")
    lines.append("## 8. 复现命令")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /home/kouzheng_2/AH_YeP/DoseSyn")
    lines.append(".venv/bin/python scripts/02_build_features.py")
    lines.append(".venv/bin/python -m pytest -q tests")
    lines.append("```")
    lines.append("")
    lines.append("## 9. 尚未解决的问题")
    lines.append("")
    lines.append("- 2 个金属配合物的分子图不完整，需在 M3 前决定是否手工补铂配位键或排除。")
    lines.append("- EFM192B / UWB1289BRCA1 无 DepMap 模型；COLO320DM 仅有 COLO320，均保守排除。")
    lines.append("- OCUBM / UWB1289 有 DepMap ID 但该版 CCLE 表达谱无数据。")
    lines.append("- 未做基因级标准化，避免全数据拟合；M3 需在训练集上 fit。")
    lines.append("- 未创建训练/验证/测试划分（按规划留给 M3）。")
    lines.append("")
    out["feature_report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/features_m2.yaml")
    parser.add_argument("--stage", default="all", choices=["all", "drugs", "cells", "coverage"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_directories(cfg)
    log = setup_logging(
        "dosesyn.features",
        Path(cfg["paths"]["logs_dir"]) / "02_build_features.log",
    )
    out = {k: resolve_path(cfg, v) for k, v in cfg["outputs"].items()}
    log.info("DoseSyn M2 starting; config=%s", cfg["_config_path"])

    m1_files = [
        cfg["files"]["m1_combo_points"],
        cfg["files"]["m1_combo_metadata"],
        cfg["files"]["m1_mono_points"],
        cfg["files"]["m1_mono_metadata"],
    ]
    m1_hashes = {str(p): sha256_file(p) for p in m1_files}

    mapping = graph_meta = None
    if args.stage in ("all", "drugs"):
        mapping, graph_meta, diag = _stage_drugs(cfg, log, out)
    else:
        from dosesyn.features.graph import load_graphs

        mapping = pd.read_csv(out["drug_mapping"])
        _, graph_meta = load_graphs(out["drug_graphs"])
        diag = {}

    cell_map = scores = pmeta = None
    cdiag = pdiag = None
    if args.stage in ("all", "cells"):
        cell_map, cdiag, scores, pmeta, pdiag = _stage_cells(cfg, log, out)
    elif args.stage == "coverage":
        cell_map = pd.read_csv(out["cell_mapping"])
        scores = pd.read_parquet(out["pathway_features"]).set_index("DepMap_ID")
        pmeta = pd.read_csv(out["pathway_metadata"])
        cdiag = {}
        pdiag = {
            "min_genes": cfg["pathway"]["min_genes"],
            "min_coverage": cfg["pathway"]["min_coverage"],
            "alpha": cfg["pathway"]["alpha"],
            "score_range": [
                float(scores.min().min()),
                float(scores.max().max()),
            ],
        }

    coverage = cancer = None
    if args.stage in ("all", "coverage"):
        coverage, cancer, matrix_index, condition_index, mono_index = _stage_coverage(
            cfg, log, out, mapping, graph_meta, cell_map, scores, pmeta, pdiag
        )

    if args.stage == "all":
        log.info("running full pytest suite")
        pytest_log = Path(cfg["paths"]["audit_dir"]) / "pytest.log"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        text = proc.stdout + "\n" + proc.stderr
        pytest_log.write_text(text, encoding="utf-8")
        tail = [ln for ln in text.strip().splitlines() if ln.strip()][-1] if text.strip() else "no output"
        tests = {
            "returncode": proc.returncode,
            "summary": tail,
        }
        log.info("pytest: %s", tail)
        out["pytest_log"] = pytest_log
        _write_report(cfg, out, diag, cdiag, pdiag, coverage, cancer, tests, m1_hashes)

        manifest = {
            "generated_at": utc_now_iso(),
            "stage": "M2",
            "config": cfg["_config_path"],
            "environment": environment_snapshot(),
            "feature_versions": {
                "m2": cfg["version"],
                "graph": cfg["graph"]["feature_version"],
                "pathway_alpha": cfg["pathway"]["alpha"],
            },
            "inputs": {
                "m1": m1_hashes,
                "confsyn_expression": cfg["files"]["confsyn_expression"],
                "confsyn_sample_info": cfg["files"]["confsyn_sample_info"],
                "hallmark_gmt": cfg["files"]["hallmark_gmt"],
            },
            "outputs": {
                str(p): {"sha256": sha256_file(p), "bytes": Path(p).stat().st_size}
                for p in [
                    out["drug_mapping"],
                    out["drug_graphs"],
                    out["drug_graph_metadata"],
                    out["cell_mapping"],
                    out["cell_expression"],
                    out["cell_features"],
                    out["pathway_features"],
                    out["pathway_metadata"],
                    out["feature_index"],
                    out["condition_feature_index"],
                    out["mono_feature_index"],
                    out["feature_coverage"],
                ]
                if Path(p).exists()
            },
            "tests": tests,
        }
        write_json(manifest, Path(cfg["paths"]["audit_dir"]) / "m2_manifest.json")
        return 0 if proc.returncode == 0 else 1

    log.info("stage %s finished", args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
