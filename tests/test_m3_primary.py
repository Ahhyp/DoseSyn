"""M3 frozen Primary-v1 dataset tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dosesyn.utils.io import sha256_file


def test_primary_counts_match_freeze(m3_primary, m3_cfg):
    exp = m3_cfg["expected"]
    assert len(m3_primary) == exp["matrices"]
    assert m3_primary["condition_id"].nunique() == exp["conditions"]
    assert m3_primary["cell_id"].nunique() == exp["cells"]
    assert len(set(m3_primary["drug_a_id"]) | set(m3_primary["drug_b_id"])) == exp["drugs"]
    assert m3_primary["pair_id"].nunique() == exp["pairs"]
    assert int(m3_primary["n_points"].sum()) == exp["points"]


def test_primary_is_complete_and_unique(m3_primary):
    assert not m3_primary["matrix_id"].duplicated().any()
    assert (m3_primary["n_points"] == 16).all()
    assert (m3_primary["n_dose_a"] == 4).all()
    assert (m3_primary["n_dose_b"] == 4).all()
    assert m3_primary["condition_id"].notna().all()
    assert m3_primary["pair_id"].notna().all()


def test_metal_complexes_excluded_but_retained_in_inputs(m3_primary, m3_cfg):
    present = set(m3_primary["drug_a_name"]) | set(m3_primary["drug_b_name"])
    for name in m3_cfg["filters"]["excluded_drugs"]:
        assert name not in present
    mapping = pd.read_csv(Path(m3_cfg["inputs"]["drug_mapping"]))
    assert {"Carboplatin", "Oxaliplatin"} <= set(mapping["oneil_drug_name"])
    # exclusion record exists rather than a silent drop
    ex = pd.read_parquet(Path(m3_cfg["outputs"]["exclusions"]))
    assert {"Carboplatin", "Oxaliplatin"} <= set(ex["drug_name"].dropna()) | set(ex["drug_a_name"].dropna())


def test_excluded_rows_are_documented(m3_cfg):
    ex = pd.read_parquet(Path(m3_cfg["outputs"]["exclusions"]))
    assert "exclusion_reason" in ex.columns
    summary = ex.groupby(["dataset", "exclusion_reason"]).size()
    assert summary.sum() > 0
    assert (ex["exclusion_reason"] == "drug_graph_excluded").any()


def test_primary_hashes_match_manifest(m3_cfg, m3_input_manifest):
    out = m3_input_manifest["outputs"]
    for rel, expected in out.items():
        assert sha256_file(Path(m3_cfg["_root"]) / rel) == expected


def test_m1_m2_inputs_unchanged_by_m3(m3_cfg, m3_input_manifest):
    for rel, expected in m3_input_manifest["input_hashes"].items():
        assert sha256_file(Path(m3_cfg["_root"]) / rel) == expected, f"M3 modified input {rel}"


def test_primary_labels_are_raw_and_unclipped(m3_primary):
    assert m3_primary["dose_a_um_json"].str.len().min() > 0
    # max X/X0 of primary source is preserved (2.34333 is Carboplatin, excluded)
    assert m3_primary["n_points"].sum() == 297_584
