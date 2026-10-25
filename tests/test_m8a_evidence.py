"""M8-A evidence-grade invariants over the frozen DoseSyn splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "outputs" / "metrics" / "m8a"


@pytest.fixture(scope="module")
def evidence():
    return pd.read_parquet(M / "split_matrix_evidence.parquet")


@pytest.fixture(scope="module")
def summary():
    return pd.read_csv(M / "split_independence_summary.csv")


@pytest.fixture(scope="module")
def full_metrics():
    return json.loads((M / "bridge_feasibility.json").read_text(encoding="utf-8"))


def test_grades_are_disjoint_and_cover_all_rows(evidence):
    assert set(evidence.evidence_level) <= {"A", "B", "C", "D"}
    # A matrix can be test in one seed and train in another, so uniqueness is
    # per (scenario, split, matrix), not global.
    keyed = evidence.groupby(["scenario", "split_id", "matrix_id"]).evidence_level.nunique()
    assert keyed.max() == 1
    assert evidence.evidence_level.notna().all()


def test_grade_a_requires_test_role_and_absent_from_conf(evidence):
    a = evidence[evidence.evidence_level == "A"]
    assert len(a) == 3793
    assert (a.role == "test").all()
    assert a.in_drugcombdb.eq(False).all()
    assert a.supported.all()
    # Per-scenario A test coverage.
    per = a.groupby("scenario").size().to_dict()
    assert per == {
        "s1_random_condition": 853,
        "s2_unseen_pair": 930,
        "s3_unseen_cell": 2010,
    }


def test_grade_b_is_train_or_val_only(evidence):
    b = evidence[evidence.evidence_level == "B"]
    assert len(b) == 18317
    assert b.role.isin(["train", "val"]).all()
    assert not (b.role == "test").any()
    assert b.in_drugcombdb.eq(False).all()


def test_grade_c_is_condition_seen_by_conf(evidence):
    c = evidence[evidence.evidence_level == "C"]
    assert len(c) == 32340
    assert c.in_drugcombdb.all()


def test_grade_d_is_unsupported(evidence):
    d = evidence[evidence.evidence_level == "D"]
    assert len(d) == 150139
    assert (~d.supported).all()


def test_summary_reconciles_with_row_level(evidence, summary):
    per = evidence.groupby(["scenario", "split_id", "role", "evidence_level"]).size()
    for row in summary.itertuples(index=False):
        key = (row.scenario, row.split_id, row.role, row.level)
        assert int(per.loc[key]) == int(row.n_matrices)


def test_a_condition_bridge(evidence):
    a = evidence[evidence.evidence_level == "A"]
    conds = a[["scenario", "condition_id"]].drop_duplicates()
    assert len(conds) == 3502
    assert conds.condition_id.nunique() == 2010
    per = conds.groupby("scenario").size().to_dict()
    assert per == {
        "s1_random_condition": 742,
        "s2_unseen_pair": 750,
        "s3_unseen_cell": 2010,
    }


def test_label_provenance_is_not_interchangeable(full_metrics):
    labels = full_metrics["label_provenance"]
    assert labels["n_in_both"] == 2842
    assert labels["exact_label_equal_in_both"] == 0
    assert labels["n_os_db_pairs"] > 0
    assert -1.0 <= labels["pearson_os_vs_db"] <= 1.0
    assert -1.0 <= labels["pearson_os_synergy_vs_dosesyn_x_x0"] <= 1.0


def test_verdict_is_conditional_and_blockers_recorded(full_metrics):
    assert full_metrics["verdict"] == "CONDITIONAL PASS"
    checks = full_metrics["checks"]
    assert checks["drug_mapping_at_least_20"] is True
    assert checks["cell_mapping_at_least_25"] is True
    assert checks["condition_matches_at_least_1000"] is True
    assert checks["A_bridge_available_in_all_scenarios"] is True
    assert checks["confsyn_per_sample_split_persisted"] is False
    assert checks["confsyn_oncology_screen_checkpoint_available"] is False
