"""M4 frozen-data integrity, leakage and prediction-artefact tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dosesyn.training import m4_data

ROOT = Path(__file__).resolve().parents[1]
S1 = "s1_random_condition"
S3 = "s3_unseen_cell"


@pytest.fixture(scope="module")
def s1_train():
    return m4_data.load_matrix_table(S1, "train", seed=42)


@pytest.fixture(scope="module")
def s1_val():
    return m4_data.load_matrix_table(S1, "val", seed=42)


@pytest.fixture(scope="module")
def s1_test():
    return m4_data.load_matrix_table(S1, "test", seed=42)


def test_matrix_shapes_and_raw_labels(s1_train):
    assert s1_train.y.shape[1:] == (4, 4)
    assert s1_train.dose_a_std.shape[1] == 4
    assert s1_train.dose_b_std.shape[1] == 4
    assert s1_train.cell_features.shape[1] == 50
    assert np.isfinite(s1_train.y).all()
    assert s1_train.y.min() >= 0.0
    assert s1_train.y.max() <= 2.5
    assert len(set(s1_train.matrix_ids)) == len(s1_train)


def test_split_matrix_sets_are_disjoint(s1_train, s1_val, s1_test):
    a, b, c = set(s1_train.matrix_ids), set(s1_val.matrix_ids), set(s1_test.matrix_ids)
    assert not (a & b)
    assert not (a & c)
    assert not (b & c)
    assert len(a | b | c) == len(a) + len(b) + len(c)


def test_split_conditions_are_disjoint(s1_train, s1_val, s1_test):
    a, b, c = set(s1_train.condition_ids), set(s1_val.condition_ids), set(s1_test.condition_ids)
    assert not (a & b) and not (a & c) and not (b & c)


def test_s3_scaler_fit_cells_exclude_val_test_cells():
    t = m4_data.load_matrix_table(S3, "train", fold=0)
    meta = json.loads((t.split_dir / "split_metadata.json").read_text(encoding="utf-8"))
    fit_cells = set(meta["scaler_fit"]["fingerprints"]["fit_cells"])
    assert fit_cells == set(t.depmap_ids)
    val = m4_data.load_matrix_table(S3, "val", fold=0)
    test = m4_data.load_matrix_table(S3, "test", fold=0)
    assert not (fit_cells & set(val.depmap_ids))
    assert not (fit_cells & set(test.depmap_ids))


def test_s3_cells_fully_disjoint():
    train = m4_data.load_matrix_table(S3, "train", fold=0)
    val = m4_data.load_matrix_table(S3, "val", fold=0)
    test = m4_data.load_matrix_table(S3, "test", fold=0)
    assert not (set(train.cell_ids) & set(val.cell_ids))
    assert not (set(train.cell_ids) & set(test.cell_ids))
    assert not (set(val.cell_ids) & set(test.cell_ids))


def test_b0_predictions_have_16_rows_per_matrix():
    run = ROOT / "outputs/runs/m4/b0_mean__s1_random_condition__seed_42"
    path = run / "predictions_test.parquet"
    if not path.exists():
        pytest.skip("run b0_mean s1 seed42 first")
    frame = pd.read_parquet(path)
    assert frame.groupby("matrix_id").size().eq(16).all()
    assert {"model", "scenario", "seed_or_fold", "matrix_id", "row_idx", "col_idx", "y_true", "y_pred"} <= set(frame.columns)
    assert frame["y_true"].between(0, 2.5).all()


def test_test_predictions_match_frozen_test_matrices():
    run = ROOT / "outputs/runs/m4/b0_mean__s1_random_condition__seed_42"
    path = run / "predictions_test.parquet"
    if not path.exists():
        pytest.skip("run b0_mean s1 seed42 first")
    frame = pd.read_parquet(path)
    test = m4_data.load_matrix_table(S1, "test", seed=42)
    assert set(frame["matrix_id"]) == set(test.matrix_ids)
