"""M2.5 feature coverage: join drug graphs, cell expression, pathways to M1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.config import resolve_path
from ..utils.io import sha256_file


def _drug_feature_table(
    mapping: pd.DataFrame, graph_meta: Dict[str, Any]
) -> pd.DataFrame:
    rows = []
    for _, r in mapping.iterrows():
        g = graph_meta.get(str(r["drug_id"]), {})
        rows.append(
            {
                "drug_id": r["drug_id"],
                "oneil_drug_name": r["oneil_drug_name"],
                "canonical_drug_name": r["canonical_drug_name"],
                "mapping_status": r["mapping_status"],
                "graph_status": g.get("graph_status", "MISSING"),
                "graph_available": bool(g) and g.get("graph_status") != "FAILED",
                "graph_ok": g.get("graph_status") == "OK",
                "graph_review": g.get("graph_status") == "NEEDS_REVIEW",
                "structure_note": r["structure_note"],
            }
        )
    return pd.DataFrame(rows).set_index("drug_id")


def build_matrices_index(
    combo_meta: pd.DataFrame,
    drug_feat: pd.DataFrame,
    cell_map_by_name: pd.DataFrame,
) -> pd.DataFrame:
    out = combo_meta[
        [
            "matrix_id",
            "condition_id",
            "batch_id",
            "cell_id",
            "cell_line",
            "drug_a_id",
            "drug_a_name",
            "drug_b_id",
            "drug_b_name",
        ]
    ].copy()
    out["drug_a_graph_available"] = out["drug_a_id"].map(drug_feat["graph_available"])
    out["drug_b_graph_available"] = out["drug_b_id"].map(drug_feat["graph_available"])
    out["drug_a_graph_ok"] = out["drug_a_id"].map(drug_feat["graph_ok"])
    out["drug_b_graph_ok"] = out["drug_b_id"].map(drug_feat["graph_ok"])
    out["drug_a_graph_review"] = out["drug_a_id"].map(drug_feat["graph_review"])
    out["drug_b_graph_review"] = out["drug_b_id"].map(drug_feat["graph_review"])

    cm = cell_map_by_name.set_index("oneil_cell_name")
    out["cell_mapping_status"] = out["cell_line"].map(cm["mapping_status"])
    out["cell_expression"] = out["cell_line"].map(cm["expression_available"]).fillna(False)
    out["cell_pathway"] = out["cell_line"].map(cm["pathway_available"]).fillna(False)
    out["cell_feature_complete"] = out["cell_line"].map(cm["cell_feature_complete"]).fillna(False)
    out["depmap_id"] = out["cell_line"].map(cm["depmap_id"])

    for col in [
        "drug_a_graph_available",
        "drug_b_graph_available",
        "drug_a_graph_ok",
        "drug_b_graph_ok",
        "drug_a_graph_review",
        "drug_b_graph_review",
    ]:
        out[col] = out[col].fillna(False).astype(bool)

    out["feature_complete"] = (
        out["drug_a_graph_ok"]
        & out["drug_b_graph_ok"]
        & out["cell_feature_complete"]
    )

    def reasons(row) -> str:
        r = []
        if not row["cell_feature_complete"]:
            if pd.isna(row["depmap_id"]):
                r.append("cell_unmapped")
            elif not row["cell_expression"]:
                r.append("cell_expression_missing")
            elif not row["cell_pathway"]:
                r.append("cell_pathway_missing")
            else:
                r.append("cell_mapping_not_eligible")
        if not (row["drug_a_graph_ok"] and row["drug_b_graph_ok"]):
            r.append("drug_graph_not_model_ready")
        return ";".join(r)

    out["missing_reasons"] = out.apply(reasons, axis=1)
    return out


def build_condition_index(matrix_index: pd.DataFrame) -> pd.DataFrame:
    g = matrix_index.groupby("condition_id", sort=False)
    out = g.agg(
        cell_line=("cell_line", "first"),
        cell_id=("cell_id", "first"),
        drug_a_name=("drug_a_name", "first"),
        drug_b_name=("drug_b_name", "first"),
        n_matrices=("matrix_id", "nunique"),
        n_batches=("batch_id", "nunique"),
        n_feature_complete=("feature_complete", "sum"),
    ).reset_index()
    out["feature_complete"] = out["n_feature_complete"] > 0
    out["all_matrices_complete"] = out["n_feature_complete"] == out["n_matrices"]
    return out


def build_mono_index(
    mono_meta: pd.DataFrame,
    drug_feat: pd.DataFrame,
    cell_map_by_name: pd.DataFrame,
) -> pd.DataFrame:
    out = mono_meta[
        [
            "mono_curve_id",
            "mono_condition_id",
            "batch_id",
            "cell_id",
            "cell_line",
            "drug_id",
            "drug_name",
        ]
    ].copy()
    out["drug_graph_available"] = out["drug_id"].map(drug_feat["graph_available"])
    out["drug_graph_ok"] = out["drug_id"].map(drug_feat["graph_ok"])
    out["drug_graph_review"] = out["drug_id"].map(drug_feat["graph_review"])
    cm = cell_map_by_name.set_index("oneil_cell_name")
    out["cell_mapping_status"] = out["cell_line"].map(cm["mapping_status"])
    out["cell_expression"] = out["cell_line"].map(cm["expression_available"]).fillna(False)
    out["cell_pathway"] = out["cell_line"].map(cm["pathway_available"]).fillna(False)
    out["cell_feature_complete"] = out["cell_line"].map(cm["cell_feature_complete"]).fillna(False)
    out["depmap_id"] = out["cell_line"].map(cm["depmap_id"])
    for col in ["drug_graph_available", "drug_graph_ok", "drug_graph_review"]:
        out[col] = out[col].fillna(False).astype(bool)
    out["feature_complete"] = (
        out["drug_graph_ok"] & out["cell_feature_complete"]
    )

    def reasons(row) -> str:
        r = []
        if not row["cell_feature_complete"]:
            if pd.isna(row["depmap_id"]):
                r.append("cell_unmapped")
            elif not row["cell_expression"]:
                r.append("cell_expression_missing")
            elif not row["cell_pathway"]:
                r.append("cell_pathway_missing")
            else:
                r.append("cell_mapping_not_eligible")
        if not row["drug_graph_ok"]:
            r.append("drug_graph_not_model_ready")
        return ";".join(r)

    out["missing_reasons"] = out.apply(reasons, axis=1)
    return out


def build_cancer_type_table(
    cell_mapping: pd.DataFrame, matrix_index: pd.DataFrame, mono_index: pd.DataFrame
) -> pd.DataFrame:
    cm = cell_mapping.copy()
    cm["cancer_type"] = cm["disease"].fillna(cm["lineage"]).fillna("UNKNOWN")
    per_cell = cm[
        ["oneil_cell_name", "cancer_type", "feature_eligible", "pathway_available"]
    ].copy()
    per_cell["cell_complete"] = per_cell["feature_eligible"] & per_cell["pathway_available"]

    m = matrix_index.groupby("cell_line").agg(
        n_matrices=("matrix_id", "nunique"),
        n_matrices_complete=("feature_complete", "sum"),
    )
    mo = mono_index.groupby("cell_line").agg(
        n_curves=("mono_curve_id", "nunique"),
        n_curves_complete=("feature_complete", "sum"),
    )
    per_cell = per_cell.join(m, on="oneil_cell_name").join(mo, on="oneil_cell_name")
    per_cell = per_cell.fillna(0)
    for c in ["n_matrices", "n_matrices_complete", "n_curves", "n_curves_complete"]:
        per_cell[c] = per_cell[c].astype(int)

    g = per_cell.groupby("cancer_type", sort=True)
    out = g.agg(
        n_cells=("oneil_cell_name", "nunique"),
        n_cells_feature_complete=("cell_complete", "sum"),
        n_matrices=("n_matrices", "sum"),
        n_matrices_feature_complete=("n_matrices_complete", "sum"),
        n_mono_curves=("n_curves", "sum"),
        n_mono_curves_feature_complete=("n_curves_complete", "sum"),
    ).reset_index()
    out["cell_retention"] = out["n_cells_feature_complete"] / out["n_cells"].replace(0, np.nan)
    out["matrix_retention"] = out["n_matrices_feature_complete"] / out["n_matrices"].replace(0, np.nan)
    out = out.sort_values(["n_matrices", "cancer_type"], ascending=[False, True])
    return out


def _reason_counts(index: pd.DataFrame) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in index["missing_reasons"]:
        if not value:
            continue
        for reason in str(value).split(";"):
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def build_coverage(
    cfg: Dict[str, Any],
    drug_mapping: pd.DataFrame,
    drug_graph_meta: Dict[str, Any],
    cell_mapping: pd.DataFrame,
    pathway_meta: pd.DataFrame,
    pathway_diag: Dict[str, Any],
    combo_meta: pd.DataFrame,
    mono_meta: pd.DataFrame,
    matrix_index: pd.DataFrame,
    condition_index: pd.DataFrame,
    mono_index: pd.DataFrame,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    cancer = build_cancer_type_table(cell_mapping, matrix_index, mono_index)

    drugs = pd.DataFrame(
        [
            {
                "oneil_drug_name": r["oneil_drug_name"],
                "mapping_status": r["mapping_status"],
                "has_smiles": isinstance(r["isomeric_smiles"], str) and bool(r["isomeric_smiles"]),
                "graph_status": drug_graph_meta.get(str(r["drug_id"]), {}).get("graph_status", "MISSING"),
            }
            for _, r in drug_mapping.iterrows()
        ]
    )
    n_drugs = int(drugs.shape[0])
    n_smiles = int(drugs["has_smiles"].sum())
    n_graph = int((drugs["graph_status"] != "MISSING").sum())
    n_graph_ok = int((drugs["graph_status"] == "OK").sum())
    n_graph_review = int((drugs["graph_status"] == "NEEDS_REVIEW").sum())

    n_cells = int(len(cell_mapping))
    n_mapped = int(cell_mapping["depmap_id"].notna().sum())
    n_verified = int(cell_mapping["mapping_status"].isin(["VERIFIED", "VERIFIED_ALIAS"]).sum())
    n_expr = int(cell_mapping["expression_available"].sum())
    n_pathway_cells = int(cell_mapping["pathway_available"].sum())
    n_cell_complete = int(
        (cell_mapping["feature_eligible"] & cell_mapping["pathway_available"]).sum()
    )

    valid_pathways = pathway_meta.loc[pathway_meta["valid"]]
    coverage = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "feature_version": cfg["version"],
        "drugs": {
            "n_total": n_drugs,
            "n_smiles": n_smiles,
            "n_smiles_fraction": n_smiles / n_drugs if n_drugs else None,
            "n_graph_built": n_graph,
            "n_graph_ok": n_graph_ok,
            "n_graph_needs_review": n_graph_review,
            "n_graph_failed": n_drugs - n_graph,
            "review_drugs": drugs.loc[drugs["graph_status"] == "NEEDS_REVIEW", "oneil_drug_name"].tolist(),
            "unmapped_drugs": drugs.loc[drugs["mapping_status"] == "UNMAPPED", "oneil_drug_name"].tolist(),
        },
        "cells": {
            "n_total": n_cells,
            "n_mapped_depmap": n_mapped,
            "n_verified_mapping": n_verified,
            "n_expression": n_expr,
            "n_pathway": n_pathway_cells,
            "n_feature_complete": n_cell_complete,
            "unmapped": cell_mapping.loc[
                cell_mapping["mapping_status"] == "UNMAPPED", "oneil_cell_name"
            ].tolist(),
            "needs_review": cell_mapping.loc[
                cell_mapping["mapping_status"] == "NEEDS_REVIEW", "oneil_cell_name"
            ].tolist(),
            "mapped_without_expression": cell_mapping.loc[
                cell_mapping["depmap_id"].notna() & ~cell_mapping["expression_available"],
                "oneil_cell_name",
            ].tolist(),
        },
        "pathways": {
            "n_total_in_gmt": int(pathway_meta.shape[0]),
            "n_valid": int(pathway_meta["valid"].sum()),
            "n_genes_min": int(valid_pathways["n_genes"].min()) if len(valid_pathways) else 0,
            "n_genes_max": int(valid_pathways["n_genes"].max()) if len(valid_pathways) else 0,
            "coverage_min": float(valid_pathways["coverage"].min()) if len(valid_pathways) else None,
            "coverage_max": float(valid_pathways["coverage"].max()) if len(valid_pathways) else None,
            "coverage_mean": float(valid_pathways["coverage"].mean()) if len(valid_pathways) else None,
            "n_cells": int(cell_mapping["pathway_available"].sum()),
            "alpha": pathway_diag["alpha"],
            "score_range": pathway_diag["score_range"],
            "gene_mapping_rule": "exact uppercase HGNC symbol match between DepMap column prefix and MSigDB symbol GMT",
        },
        "matrices": {
            "n_total": int(matrix_index.shape[0]),
            "n_feature_complete": int(matrix_index["feature_complete"].sum()),
            "fraction_complete": float(matrix_index["feature_complete"].mean()),
            "n_missing_reasons": _reason_counts(matrix_index),
            "n_conditions": int(condition_index.shape[0]),
            "n_conditions_complete": int(condition_index["feature_complete"].sum()),
            "n_conditions_all_matrices_complete": int(condition_index["all_matrices_complete"].sum()),
            "n_unique_pairs_with_features": int(
                matrix_index.loc[matrix_index["feature_complete"], ["drug_a_id", "drug_b_id"]]
                .drop_duplicates()
                .shape[0]
            ),
        },
        "monotherapy": {
            "n_curves_total": int(mono_meta.shape[0]),
            "n_curves_complete": int(mono_index["feature_complete"].sum()),
            "fraction_complete": float(mono_index["feature_complete"].mean()),
            "n_missing_reasons": _reason_counts(mono_index),
        },
        "by_cancer_type": cancer.to_dict(orient="records"),
        "provenance": {
            "m1_combo_metadata_sha256": sha256_file(resolve_path(cfg, cfg["files"]["m1_combo_metadata"])),
            "m1_mono_metadata_sha256": sha256_file(resolve_path(cfg, cfg["files"]["m1_mono_metadata"])),
            "drug_mapping_sha256": sha256_file(resolve_path(cfg, cfg["outputs"]["drug_mapping"])),
            "cell_mapping_sha256": sha256_file(resolve_path(cfg, cfg["outputs"]["cell_mapping"])),
        },
    }
    return coverage, cancer
