"""M2.4 Hallmark pathway activity from CCLE expression.

Scores are computed per cell with a rank-based weighted enrichment score
(ssGSEA-style, alpha configurable).  No cohort mean/variance is used, so the
transform is independent of the M3 train/validation/test split and can be
cached safely.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PATHWAY_COLUMNS = ["pathway", "n_genes", "n_present", "coverage", "valid", "source"]


def load_gmt(path: str | Path) -> Dict[str, List[str]]:
    sets: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            name = fields[0].strip()
            genes = []
            seen = set()
            for g in fields[2:]:
                key = g.strip().upper()
                if key and key not in seen:
                    seen.add(key)
                    genes.append(key)
            if genes:
                sets[name] = genes
    return sets


def pathway_coverage(
    gene_sets: Dict[str, List[str]], gene_universe: List[str], source: str
) -> pd.DataFrame:
    universe = {str(g).upper() for g in gene_universe}
    rows = []
    for name, genes in gene_sets.items():
        present = [g for g in genes if g in universe]
        rows.append(
            {
                "pathway": name,
                "n_genes": len(genes),
                "n_present": len(present),
                "coverage": len(present) / len(genes) if genes else 0.0,
                "valid": False,
                "source": source,
            }
        )
    return pd.DataFrame(rows)[PATHWAY_COLUMNS]


def _enrichment_scores(
    values: np.ndarray,
    gene_sets: Dict[str, List[str]],
    gene_index: Dict[str, int],
    alpha: float,
) -> np.ndarray:
    """Signed max-deviation enrichment score for one cell.

    Genes are ranked by expression (highest first).  Hit weights are
    ``(N - rank + 1) ** alpha`` normalised over the set; miss weights are
    uniform over the complement.  The score is the cumulative-walk deviation
    with the largest absolute value.  This uses only the current cell's ranks.
    """
    n = values.shape[0]
    order = np.argsort(-values, kind="stable")
    rank_pos = np.empty(n, dtype=np.int64)
    rank_pos[order] = np.arange(1, n + 1)
    weights = (n - rank_pos + 1).astype(np.float64) ** alpha

    scores = np.zeros(len(gene_sets), dtype=np.float64)
    for k, (name, genes) in enumerate(gene_sets.items()):
        idx = np.array(
            [gene_index[g] for g in genes if g in gene_index], dtype=np.int64
        )
        if idx.size == 0:
            scores[k] = np.nan
            continue
        hit = np.zeros(n, dtype=bool)
        hit[idx] = True
        w_hit = weights[idx]
        p_hit = w_hit / w_hit.sum()
        n_miss = n - idx.size
        if n_miss <= 0:
            scores[k] = 0.0
            continue
        delta = np.full(n, -1.0 / n_miss, dtype=np.float64)
        delta[idx] = p_hit
        walk = np.cumsum(delta[order])
        scores[k] = walk[np.argmax(np.abs(walk))]
    return scores


def compute_pathway_features(
    expr: pd.DataFrame,
    gene_sets: Dict[str, List[str]],
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    alpha = float(cfg["pathway"]["alpha"])
    min_genes = int(cfg["pathway"]["min_genes"])
    min_cov = float(cfg["pathway"]["min_coverage"])
    source = str(cfg["pathway"]["gmt_source"])

    genes = [str(g).upper() for g in expr.columns]
    if len(set(genes)) != len(genes):
        raise ValueError("duplicated gene symbols in expression matrix")
    gene_index = {g: i for i, g in enumerate(genes)}
    coverage = pathway_coverage(gene_sets, genes, source)
    coverage["valid"] = (coverage["n_present"] >= min_genes) & (
        coverage["coverage"] >= min_cov
    )

    valid_sets = {
        name: gene_sets[name]
        for name in coverage.loc[coverage["valid"], "pathway"].tolist()
    }
    if not valid_sets:
        raise ValueError("no valid Hallmark pathway after coverage filtering")

    values = expr.to_numpy(dtype=np.float64)
    scores = np.vstack(
        [_enrichment_scores(row, valid_sets, gene_index, alpha) for row in values]
    )
    out = pd.DataFrame(scores, index=expr.index, columns=list(valid_sets))
    out.index.name = "DepMap_ID"

    diagnostics = {
        "alpha": alpha,
        "min_genes": min_genes,
        "min_coverage": min_cov,
        "n_gene_sets_in_gmt": len(gene_sets),
        "n_valid_pathways": int(coverage["valid"].sum()),
        "n_invalid_pathways": int((~coverage["valid"]).sum()),
        "min_coverage_observed": float(coverage["coverage"].min()),
        "max_coverage_observed": float(coverage["coverage"].max()),
        "n_expression_genes": int(len(genes)),
        "n_cells": int(out.shape[0]),
        "score_range": [float(np.nanmin(scores)), float(np.nanmax(scores))],
        "formula": "signed max |cumsum(hit-miss)| over genes ranked by expression; hit weights (N-rank+1)^alpha",
        "uses_dataset_statistics": False,
    }
    return out, coverage, diagnostics


def expression_fingerprint(expr: pd.DataFrame, gene_sets: Dict[str, List[str]], alpha: float) -> str:
    h = hashlib.sha256()
    h.update(str(alpha).encode())
    h.update("|".join(sorted(gene_sets)).encode())
    h.update("|".join(str(g) for g in expr.columns).encode())
    h.update("|".join(str(x) for x in expr.index).encode())
    h.update(np.ascontiguousarray(expr.to_numpy(dtype=np.float32)).tobytes())
    return h.hexdigest()


def load_cached_scores(cache_path: str | Path, fingerprint: str) -> pd.DataFrame | None:
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(".meta.json")
    if not (cache_path.exists() and meta_path.exists()):
        return None
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("fingerprint") != fingerprint:
        return None
    return pd.read_parquet(cache_path).set_index("DepMap_ID")


def save_cached_scores(
    scores: pd.DataFrame, cache_path: str | Path, fingerprint: str
) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    scores.reset_index().to_parquet(cache_path, index=False)
    meta_path = cache_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"fingerprint": fingerprint}, fh, indent=2)
        fh.write("\n")
