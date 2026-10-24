"""M2.1/M2.2: drug entity mapping and feature-complete structure checks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dosesyn.utils.config import load_config
from dosesyn.utils.io import read_json

EXPECTED_DRUGS = 38
METAL_DRUGS = {"Carboplatin", "Oxaliplatin"}


@pytest.fixture(scope="module")
def m2cfg():
    return load_config("configs/features_m2.yaml")


@pytest.fixture(scope="module")
def drug_mapping(m2cfg):
    p = Path(m2cfg["outputs"]["drug_mapping"])
    if not p.exists():
        pytest.fail(f"missing {p}; run scripts/02_build_features.py --stage drugs")
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def m2_diag(m2cfg):
    return read_json(Path(m2cfg["paths"]["audit_dir"]) / "m2_drug_build.json")


def test_all_m1_drugs_are_mapped(drug_mapping, combo, mono):
    m1_ids = set(combo["drug_a_id"]) | set(combo["drug_b_id"]) | set(mono["drug_id"])
    assert set(drug_mapping["drug_id"]) == m1_ids
    assert len(drug_mapping) == EXPECTED_DRUGS
    assert drug_mapping["drug_id"].is_unique
    assert set(drug_mapping["oneil_drug_name"]) == {
        "5-FU", "ABT-888", "AZD1775", "BEZ-235", "Bortezomib", "Carboplatin",
        "Cyclophosphamide", "Dasatinib", "Dexamethasone", "Dinaciclib",
        "Doxorubicin", "Erlotinib", "Etoposide", "Gemcitabine", "L778123",
        "Lapatinib", "MK-2206", "MK-4541", "MK-4827", "MK-5108", "MK-8669",
        "MK-8776", "MRK-003", "Metformin", "Methotrexate", "Mitomycine",
        "Oxaliplatin", "PD325901", "Paclitaxel", "SN-38", "Sorafenib",
        "Sunitinib", "Temozolomide", "Topotecan", "Vinblastine", "Vinorelbine",
        "Zolinza", "geldanamycin",
    }


def test_mapping_status_and_structure_coverage(drug_mapping, m2_diag):
    assert drug_mapping["isomeric_smiles"].notna().all()
    assert drug_mapping["canonical_smiles"].notna().all()
    assert drug_mapping["inchikey"].notna().all()
    assert drug_mapping["pubchem_cid"].notna().all()
    assert drug_mapping["mapping_source"].notna().all()
    assert drug_mapping["mapping_status"].isin(
        ["VERIFIED", "VERIFIED_ALIAS", "NEEDS_REVIEW", "UNMAPPED"]
    ).all()
    status = drug_mapping["mapping_status"].value_counts().to_dict()
    assert status.get("UNMAPPED", 0) == 0
    assert status.get("NEEDS_REVIEW", 0) == 2
    review = set(drug_mapping.loc[drug_mapping["mapping_status"] == "NEEDS_REVIEW", "oneil_drug_name"])
    assert review == METAL_DRUGS
    assert m2_diag["graph_summary"]["n_graphs"] == EXPECTED_DRUGS
    assert m2_diag["graph_summary"]["n_failed"] == 0
    assert m2_diag["graph_summary"]["n_needs_review"] == 2


def test_ambiguous_entities_are_documented(drug_mapping):
    abt = drug_mapping.set_index("oneil_drug_name").loc["ABT-888"]
    assert abt["mapping_status"] == "VERIFIED_ALIAS"
    assert abt["pubchem_cid"] == 11960529
    assert abt["inchikey"] == "JNAHVYVRKWKWKQ-CYBMUJFWSA-N"
    assert "veliparib" in abt["structure_note"].lower()

    metal = drug_mapping.set_index("oneil_drug_name").loc[["Carboplatin", "Oxaliplatin"]]
    assert (metal["n_fragments"] > 1).all()
    assert metal["structure_note"].str.contains("disconnected").all()


def test_unique_structures(drug_mapping):
    assert drug_mapping["inchikey"].is_unique
    assert drug_mapping["canonical_smiles"].is_unique


def test_side_specific_drugs_are_in_the_union(combo, drug_mapping):
    ids = set(drug_mapping["drug_id"])
    assert "MK-4541" in set(drug_mapping["oneil_drug_name"])
    assert "MK-8776" in set(drug_mapping["oneil_drug_name"])
    # Union of A and B, not just one side.
    assert set(combo["drug_a_id"]) <= ids
    assert set(combo["drug_b_id"]) <= ids
