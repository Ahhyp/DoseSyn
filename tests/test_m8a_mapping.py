"""M8-A entity mapping contracts (artifact-based, read-only)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "outputs" / "metrics" / "m8a"


@pytest.fixture(scope="module")
def drug_map():
    return pd.read_csv(M / "drug_mapping.csv")


@pytest.fixture(scope="module")
def cell_map():
    return pd.read_csv(M / "cell_mapping.csv")


def test_drug_coverage_and_unmatched(drug_map):
    assert len(drug_map) == 38
    matched = drug_map[drug_map.match_status.isin(["EXACT", "REVIEWED_ALIAS"])]
    assert len(matched) == 22
    assert "MK-8776" in set(drug_map.loc[drug_map.match_status == "UNMATCHED", "oneil_drug_name"])
    # PD325901 is in DrugCombDB but not in the OncologyScreen 21-drug table.
    assert "PD325901" in set(matched.oneil_drug_name)


def test_matched_inchikeys_are_unique(drug_map):
    keys = drug_map.loc[drug_map.match_status.isin(["EXACT", "REVIEWED_ALIAS"]), "conf_inchikey"]
    assert keys.notna().all()
    assert keys.nunique() == len(keys)


def test_cell_aliases_and_unmatched(cell_map):
    assert len(cell_map) == 39
    matched = cell_map[cell_map.match_status != "UNMATCHED"]
    assert len(matched) == 31
    aliases = cell_map[cell_map.match_status == "REVIEWED_ALIAS"]
    assert set(aliases.oneil_cell_name) == {"COLO320DM", "LNCAP"}
    # These are not present in either ConfSyn corpus; no fuzzy mapping allowed.
    assert cell_map.set_index("oneil_cell_name").loc["MSTO", "match_status"] == "UNMATCHED"
    assert cell_map.set_index("oneil_cell_name").loc["OVCAR3", "match_status"] == "UNMATCHED"


def test_condition_key_is_order_invariant():
    sys_path = ROOT / "src"
    import sys

    sys.path.insert(0, str(sys_path))
    from dosesyn.m8a.mapping import condition_key

    assert condition_key("C", "DRUG_A", "DRUG_B") == condition_key("C", "DRUG_B", "DRUG_A")
