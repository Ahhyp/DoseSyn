"""M2.3/M2.4: cell-line mapping, expression and Hallmark pathway features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dosesyn.utils.config import load_config


@pytest.fixture(scope="module")
def m2cfg():
    return load_config("configs/features_m2.yaml")


@pytest.fixture(scope="module")
def cell_mapping(m2cfg):
    p = Path(m2cfg["outputs"]["cell_mapping"])
    if not p.exists():
        pytest.fail(f"missing {p}; run scripts/02_build_features.py --stage cells")
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def cell_features(m2cfg):
    return pd.read_parquet(m2cfg["outputs"]["cell_features"])


@pytest.fixture(scope="module")
def pathway_features(m2cfg):
    return pd.read_parquet(m2cfg["outputs"]["pathway_features"])


@pytest.fixture(scope="module")
def pathway_meta(m2cfg):
    return pd.read_csv(m2cfg["outputs"]["pathway_metadata"])


def test_every_oneil_cell_is_accounted_for(cell_mapping, combo_meta, mono_meta):
    names = set(combo_meta["cell_line"]) | set(mono_meta["cell_line"])
    assert len(names) == 39
    assert set(cell_mapping["oneil_cell_name"]) == names
    assert cell_mapping["oneil_cell_name"].is_unique
    assert cell_mapping["depmap_id"].dropna().is_unique


def test_unmapped_and_review_cells_have_reasons(cell_mapping):
    bad = cell_mapping[cell_mapping["mapping_status"] != "VERIFIED"]
    assert len(bad) == 6
    assert set(bad["oneil_cell_name"]) == {
        "COLO320DM", "EFM192B", "LNCAP", "MSTO", "OVCAR3", "UWB1289BRCA1"
    }
    assert bad["mapping_reason"].notna().all()
    assert (bad["mapping_reason"].str.len() > 10).all()
    unmapped = cell_mapping[cell_mapping["mapping_status"] == "UNMAPPED"]
    assert set(unmapped["oneil_cell_name"]) == {"EFM192B", "UWB1289BRCA1"}
    assert unmapped["depmap_id"].isna().all()


def test_curated_aliases_are_explicit(cell_mapping):
    m = cell_mapping.set_index("oneil_cell_name")
    assert m.loc["LNCAP", "depmap_id"] == "ACH-000977"
    assert m.loc["MSTO", "depmap_id"] == "ACH-000335"
    assert m.loc["OVCAR3", "depmap_id"] == "ACH-000001"
    assert m.loc["LNCAP", "mapping_status"] == "VERIFIED_ALIAS"
    assert m.loc["COLO320DM", "depmap_id"] == "ACH-000202"
    assert m.loc["COLO320DM", "mapping_status"] == "NEEDS_REVIEW"
    assert not bool(m.loc["COLO320DM", "feature_eligible"])


def test_cells_missing_expression_are_flagged(cell_mapping):
    m = cell_mapping.set_index("oneil_cell_name")
    for name in ("OCUBM", "UWB1289"):
        assert pd.notna(m.loc[name, "depmap_id"])
        assert not bool(m.loc[name, "expression_available"])
        assert not bool(m.loc[name, "feature_eligible"])
    assert int(cell_mapping["expression_available"].sum()) == 35
    assert int(cell_mapping["feature_eligible"].sum()) == 34


def test_expression_and_features_share_cells(cell_mapping, cell_features, pathway_features):
    eligible = set(cell_mapping.loc[cell_mapping["feature_eligible"], "depmap_id"])
    assert set(cell_features["depmap_id"]) == eligible
    assert set(pathway_features["DepMap_ID"]) == eligible
    assert len(cell_features) == 34
    assert cell_features["depmap_id"].is_unique


def test_gene_matrix_is_consistent(m2cfg, cell_features):
    expr = pd.read_parquet(m2cfg["outputs"]["cell_expression"])
    meta_cols = {
        "DepMap_ID",
        "depmap_id",
        "oneil_cell_name",
        "standard_cell_name",
        "lineage",
        "disease",
    }
    gene_cols = [c for c in expr.columns if c not in meta_cols]
    assert len(gene_cols) == 19144
    assert expr[gene_cols].isna().sum().sum() == 0
    assert np.isfinite(expr[gene_cols].to_numpy()).all()
    assert set(expr["depmap_id"]) == set(cell_features["depmap_id"])
    summary = cell_features.set_index("depmap_id")[["n_genes", "gene_mean", "gene_std"]]
    assert (summary["n_genes"] == 19144).all()
    assert np.allclose(summary["gene_mean"], expr.set_index("depmap_id")[gene_cols].mean(axis=1))


def test_pathway_matrix_dimensions(cell_features, pathway_features, pathway_meta):
    assert pathway_features.shape == (34, 51)  # DepMap_ID + 50 Hallmark sets
    assert pathway_features["DepMap_ID"].is_unique
    score_cols = [c for c in pathway_features.columns if c.startswith("HALLMARK_")]
    assert len(score_cols) == 50
    assert set(score_cols) == set(cell_features.columns[11:])
    assert set(score_cols) == set(pathway_meta["pathway"])
    assert pathway_meta["pathway"].is_unique
    assert np.isfinite(pathway_features[score_cols].to_numpy()).all()


def test_pathway_gene_coverage_is_preserved(pathway_meta):
    assert len(pathway_meta) == 50
    assert pathway_meta["valid"].all()
    assert (pathway_meta["n_genes"] >= 10).all()
    assert (pathway_meta["n_present"] <= pathway_meta["n_genes"]).all()
    assert (pathway_meta["coverage"] >= 0.5).all()
    recomputed = pathway_meta["n_present"] / pathway_meta["n_genes"]
    assert np.allclose(pathway_meta["coverage"], recomputed)
