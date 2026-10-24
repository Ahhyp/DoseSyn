"""M2.3 cell-line entity mapping to DepMap and M2.4 expression loading."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SAMPLE_COLUMNS = [
    "DepMap_ID",
    "stripped_cell_line_name",
    "CCLE_Name",
    "alias",
    "lineage",
    "disease",
    "disease_subtype",
]

CELL_COLUMNS = [
    "oneil_cell_name",
    "standard_cell_name",
    "depmap_id",
    "lineage",
    "disease",
    "expression_available",
    "pathway_available",
    "feature_eligible",
    "mapping_source",
    "mapping_status",
    "mapping_reason",
    "n_combo_matrices",
    "n_mono_curves",
]


def _norm(name: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def _norm_loose(name: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(name).upper())


def load_sample_info(cfg: Dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(cfg["files"]["confsyn_sample_info"], usecols=SAMPLE_COLUMNS)
    df["DepMap_ID"] = df["DepMap_ID"].astype(str).str.strip()
    df["stripped_cell_line_name"] = df["stripped_cell_line_name"].astype(str).str.strip()
    df["_stripped_key"] = df["stripped_cell_line_name"].map(_norm)
    df["_ccle_prefix_key"] = (
        df["CCLE_Name"].astype(str).str.split("_").str[0].map(_norm)
    )
    return df.drop_duplicates("DepMap_ID").reset_index(drop=True)


def _build_indices(sample: pd.DataFrame) -> Dict[str, Dict[str, List[int]]]:
    idx: Dict[str, Dict[str, List[int]]] = {"stripped": {}, "ccle": {}, "alias": {}}

    def add(bucket: str, key: str, i: int) -> None:
        if not key:
            return
        idx[bucket].setdefault(key, []).append(i)

    for i, row in sample.iterrows():
        add("stripped", row["_stripped_key"], i)
        add("ccle", row["_ccle_prefix_key"], i)
        if isinstance(row["alias"], str):
            for token in row["alias"].split(","):
                add("alias", _norm(token), i)
    return idx


def _oneil_cells(cfg: Dict[str, Any]) -> pd.DataFrame:
    combo = pd.read_parquet(
        cfg["files"]["m1_combo_metadata"],
        columns=["cell_line", "cell_id"],
    )
    mono = pd.read_parquet(
        cfg["files"]["m1_mono_metadata"],
        columns=["cell_line", "cell_id"],
    )
    rows: Dict[str, Dict[str, Any]] = {}
    for (name, cid), n in combo.groupby(["cell_line", "cell_id"]).size().items():
        r = rows.setdefault(name, {"cell_id": cid, "n_combo": 0, "n_mono": 0})
        if r["cell_id"] != cid:
            raise ValueError(f"inconsistent cell_id for {name}")
        r["n_combo"] = int(n)
    for (name, cid), n in mono.groupby(["cell_line", "cell_id"]).size().items():
        r = rows.setdefault(name, {"cell_id": cid, "n_combo": 0, "n_mono": 0})
        if r["cell_id"] != cid:
            raise ValueError(f"inconsistent cell_id for {name}")
        r["n_mono"] = int(n)
    return pd.DataFrame(
        [{"oneil_cell_name": k, **v} for k, v in sorted(rows.items())]
    )


def build_cell_mapping(
    cfg: Dict[str, Any], expression_ids: Optional[set[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    sample = load_sample_info(cfg)
    indices = _build_indices(sample)
    if expression_ids is None:
        expression_ids = read_expression_depmap_ids(cfg)
    curated = cfg["cell_mapping"].get("aliases", {}) or {}
    unmapped_reasons = cfg["cell_mapping"].get("unmapped", {}) or {}
    cells = _oneil_cells(cfg)

    rows: List[Dict[str, Any]] = []
    for _, cell in cells.iterrows():
        name = str(cell["oneil_cell_name"])
        depmap_id: Optional[str] = None
        status = "UNMAPPED"
        source = "unmapped"
        reason = unmapped_reasons.get(name, "no exact DepMap stripped-name/CCLE/alias match")
        standard_name = name
        lineage = None
        disease = None

        if name in curated:
            spec = curated[name]
            depmap_id = str(spec["depmap_id"])
            status = str(spec["status"])
            source = "curated_alias"
            reason = str(spec["reason"])
            hit = sample[sample["DepMap_ID"] == depmap_id]
            if len(hit) != 1:
                raise ValueError(f"curated DepMap ID {depmap_id} not found for {name}")
            standard_name = str(hit.iloc[0]["stripped_cell_line_name"])
            lineage = hit.iloc[0]["lineage"]
            disease = hit.iloc[0]["disease"]
        else:
            key = _norm(name)
            for bucket in ("stripped", "ccle", "alias"):
                hits = indices[bucket].get(key, [])
                if len(hits) == 1:
                    row = sample.iloc[hits[0]]
                    depmap_id = str(row["DepMap_ID"])
                    standard_name = str(row["stripped_cell_line_name"])
                    lineage = row["lineage"]
                    disease = row["disease"]
                    source = f"sample_info_{bucket}_exact"
                    status = "VERIFIED" if bucket in ("stripped", "ccle") else "VERIFIED_ALIAS"
                    reason = (
                        "exact DepMap stripped_cell_line_name / CCLE name"
                        if bucket in ("stripped", "ccle")
                        else "exact DepMap alias field"
                    )
                    break
                if len(hits) > 1:
                    status = "NEEDS_REVIEW"
                    source = f"sample_info_{bucket}_ambiguous"
                    reason = f"{len(hits)} DepMap candidates for exact key {key}"
                    break

        expr_ok = bool(depmap_id and depmap_id in expression_ids)
        if status in ("VERIFIED", "VERIFIED_ALIAS") and not expr_ok and depmap_id:
            reason = f"{reason}; DepMap model present but absent from CCLE_expression release"
        eligible = status in ("VERIFIED", "VERIFIED_ALIAS") and expr_ok
        if status == "NEEDS_REVIEW":
            eligible = False
        rows.append(
            {
                "oneil_cell_name": name,
                "standard_cell_name": standard_name,
                "depmap_id": depmap_id,
                "lineage": lineage,
                "disease": disease,
                "expression_available": expr_ok,
                "pathway_available": False,
                "feature_eligible": eligible,
                "mapping_source": source,
                "mapping_status": status,
                "mapping_reason": reason,
                "n_combo_matrices": int(cell["n_combo"]),
                "n_mono_curves": int(cell["n_mono"]),
            }
        )

    df = pd.DataFrame(rows)[CELL_COLUMNS].sort_values("oneil_cell_name").reset_index(drop=True)
    diagnostics = {
        "n_cells": int(len(df)),
        "n_mapped": int(df["depmap_id"].notna().sum()),
        "n_unmapped": int(df["depmap_id"].isna().sum()),
        "n_unique_depmap": int(df["depmap_id"].nunique(dropna=True)),
        "status_counts": df["mapping_status"].value_counts().to_dict(),
        "n_expression_available": int(df["expression_available"].sum()),
        "n_feature_eligible": int(df["feature_eligible"].sum()),
        "unmapped_cells": df.loc[df["mapping_status"] == "UNMAPPED", "oneil_cell_name"].tolist(),
        "review_cells": df.loc[df["mapping_status"] == "NEEDS_REVIEW", "oneil_cell_name"].tolist(),
        "mapped_without_expression": df.loc[
            df["depmap_id"].notna() & ~df["expression_available"], "oneil_cell_name"
        ].tolist(),
    }
    return df, diagnostics


def read_expression_depmap_ids(cfg: Dict[str, Any]) -> set[str]:
    first = pd.read_csv(cfg["files"]["confsyn_expression"], usecols=[0])
    return set(first.iloc[:, 0].astype(str).str.strip())


def load_expression(
    cfg: Dict[str, Any], cell_mapping: pd.DataFrame
) -> pd.DataFrame:
    """Load CCLE expression for eligible cells; columns are gene symbols.

    Values are the published log2(TPM+1); no dataset-level transformation is
    applied at this stage.
    """
    raw = pd.read_csv(cfg["files"]["confsyn_expression"])
    raw = raw.rename(columns={raw.columns[0]: "DepMap_ID"})
    raw["DepMap_ID"] = raw["DepMap_ID"].astype(str).str.strip()
    if raw["DepMap_ID"].duplicated().any():
        raise ValueError("duplicated DepMap_ID in CCLE_expression.csv")

    gene_cols = list(raw.columns[1:])
    symbols = [re.sub(r"\s*\(.*\)$", "", c).strip() for c in gene_cols]
    if len(set(symbols)) != len(symbols):
        raise ValueError("duplicated gene symbols after stripping Entrez suffix")
    expr = raw.set_index("DepMap_ID")[gene_cols].copy()
    expr.columns = symbols

    eligible = cell_mapping.loc[cell_mapping["feature_eligible"], "depmap_id"].tolist()
    missing = [d for d in eligible if d not in expr.index]
    if missing:
        raise ValueError(f"eligible cells missing from expression: {missing}")
    return expr.loc[eligible].sort_index()


def attach_cell_metadata(
    expr: pd.DataFrame, cell_mapping: pd.DataFrame
) -> pd.DataFrame:
    meta = cell_mapping.set_index("depmap_id")
    out = expr.copy()
    out.insert(0, "oneil_cell_name", meta.loc[out.index, "oneil_cell_name"].values)
    out.insert(1, "standard_cell_name", meta.loc[out.index, "standard_cell_name"].values)
    out.insert(2, "lineage", meta.loc[out.index, "lineage"].values)
    out.insert(3, "disease", meta.loc[out.index, "disease"].values)
    return out
