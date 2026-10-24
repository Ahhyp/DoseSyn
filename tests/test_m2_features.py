"""M2.5: feature joins to M1 matrices/curves, coverage arithmetic, traceability."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dosesyn.utils.config import load_config
from dosesyn.utils.io import read_json, sha256_file


@pytest.fixture(scope="module")
def m2cfg():
    return load_config("configs/features_m2.yaml")


@pytest.fixture(scope="module")
def matrix_index(m2cfg):
    p = Path(m2cfg["outputs"]["feature_index"])
    if not p.exists():
        pytest.fail(f"missing {p}; run scripts/02_build_features.py --stage coverage")
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def condition_index(m2cfg):
    return pd.read_parquet(m2cfg["outputs"]["condition_feature_index"])


@pytest.fixture(scope="module")
def mono_index(m2cfg):
    return pd.read_parquet(m2cfg["outputs"]["mono_feature_index"])


@pytest.fixture(scope="module")
def coverage(m2cfg):
    return read_json(m2cfg["outputs"]["feature_coverage"])


def test_no_m1_matrix_is_dropped(matrix_index, combo_meta):
    assert len(matrix_index) == 23052
    assert set(matrix_index["matrix_id"]) == set(combo_meta["matrix_id"])
    assert matrix_index["matrix_id"].is_unique


def test_no_m1_mono_curve_is_dropped(mono_index, mono_meta):
    assert len(mono_index) == 2094
    assert set(mono_index["mono_curve_id"]) == set(mono_meta["mono_curve_id"])
    assert mono_index["mono_curve_id"].is_unique


def test_complete_rows_have_all_feature_sources(matrix_index):
    complete = matrix_index[matrix_index["feature_complete"]]
    assert (complete["drug_a_graph_ok"] & complete["drug_b_graph_ok"]).all()
    assert complete["cell_feature_complete"].all()
    assert complete["depmap_id"].notna().all()
    assert (complete["missing_reasons"] == "").all()


def test_incomplete_rows_have_explicit_reasons(matrix_index):
    incomplete = matrix_index[~matrix_index["feature_complete"]]
    assert len(incomplete) > 0
    assert (incomplete["missing_reasons"].str.len() > 0).all()
    allowed = {
        "cell_unmapped",
        "cell_expression_missing",
        "cell_pathway_missing",
        "cell_mapping_not_eligible",
        "drug_graph_not_model_ready",
    }
    seen = set()
    for value in incomplete["missing_reasons"]:
        for reason in value.split(";"):
            if reason:
                seen.add(reason)
    assert seen <= allowed


def test_condition_index_arithmetic(condition_index, matrix_index):
    assert len(condition_index) == 22737
    assert condition_index["condition_id"].is_unique
    assert condition_index["n_matrices"].sum() == len(matrix_index)
    assert (condition_index["n_feature_complete"] <= condition_index["n_matrices"]).all()
    assert condition_index["feature_complete"].equals(condition_index["n_feature_complete"] > 0)
    assert condition_index["all_matrices_complete"].equals(
        condition_index["n_feature_complete"] == condition_index["n_matrices"]
    )


def test_mono_index_arithmetic(mono_index, mono_meta):
    assert mono_index["feature_complete"].sum() == 1782
    assert mono_index["depmap_id"].notna().sum() >= mono_index["feature_complete"].sum()
    assert len(mono_index) == len(mono_meta)


def test_coverage_json_matches_tables(coverage, matrix_index, mono_index, condition_index):
    assert coverage["drugs"]["n_total"] == 38
    assert coverage["drugs"]["n_smiles"] == 38
    assert coverage["drugs"]["n_graph_ok"] == 36
    assert coverage["drugs"]["n_graph_failed"] == 0
    assert coverage["cells"]["n_total"] == 39
    assert coverage["cells"]["n_mapped_depmap"] == 37
    assert coverage["cells"]["n_feature_complete"] == 34
    assert coverage["pathways"]["n_valid"] == 50
    assert coverage["matrices"]["n_total"] == len(matrix_index)
    assert coverage["matrices"]["n_feature_complete"] == int(matrix_index["feature_complete"].sum())
    assert coverage["matrices"]["n_conditions"] == len(condition_index)
    assert coverage["monotherapy"]["n_curves_total"] == len(mono_index)
    assert coverage["monotherapy"]["n_curves_complete"] == int(mono_index["feature_complete"].sum())
    assert coverage["pathways"]["n_cells"] == coverage["cells"]["n_pathway"]


def test_cancer_type_retention(coverage):
    rows = coverage["by_cancer_type"]
    assert len(rows) >= 6
    df = pd.DataFrame(rows)
    assert df["n_cells"].sum() == 39
    assert (df["n_cells_feature_complete"] <= df["n_cells"]).all()
    assert ((df["cell_retention"] >= 0) & (df["cell_retention"] <= 1)).all()
    assert ((df["matrix_retention"] >= 0) & (df["matrix_retention"] <= 1)).all()


def test_provenance_hashes_match(m2cfg, coverage):
    prov = coverage["provenance"]
    assert prov["drug_mapping_sha256"] == sha256_file(m2cfg["outputs"]["drug_mapping"])
    assert prov["cell_mapping_sha256"] == sha256_file(m2cfg["outputs"]["cell_mapping"])
    assert prov["m1_combo_metadata_sha256"] == sha256_file(m2cfg["files"]["m1_combo_metadata"])
    assert prov["m1_mono_metadata_sha256"] == sha256_file(m2cfg["files"]["m1_mono_metadata"])
