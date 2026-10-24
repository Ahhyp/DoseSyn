"""M4 metric hand-checks and prediction-frame schema."""

import numpy as np
import pandas as pd
import pytest

from dosesyn.training.metrics import (
    build_prediction_frame,
    compute_metrics,
    condition_macro_rmse,
    matrix_rmse,
    pooled_pearson,
    r2_score,
)


def test_matrix_rmse_hand_value():
    y = np.zeros((1, 16))
    p = np.full((1, 16), 0.5)
    assert matrix_rmse(y, p)[0] == pytest.approx(0.5)


def test_macro_condition_weighting():
    # condition A has two matrices, B one; macro must weight A and B equally.
    vals = [1.0, 3.0, 2.0]
    conds = ["A", "A", "B"]
    assert condition_macro_rmse(vals, conds) == pytest.approx((2.0 + 2.0) / 2)


def test_pooled_pearson_and_r2():
    y = np.arange(16.0).reshape(1, 16)
    assert pooled_pearson(y, y) == pytest.approx(1.0)
    assert r2_score(y, y) == pytest.approx(1.0)
    assert pooled_pearson(np.zeros((1, 16)), np.zeros((1, 16))) is None


def test_metrics_accept_high_and_zero_labels():
    y = np.array([[0.0] + [2.3] * 15])
    p = np.array([[0.1] + [2.0] * 15])
    m = compute_metrics(y, p, matrix_ids=["m"], condition_ids=["c"])
    assert m["n_y_true_gt1"] == 15
    assert m["n_y_true_eq0"] == 1
    assert np.isfinite(m["matrix_macro_rmse"])


def test_prediction_frame_has_16_rows_per_matrix():
    n = 2
    frame = build_prediction_frame(
        model="b0_mean",
        scenario="s1_random_condition",
        seed_or_fold="seed_42",
        split="test",
        matrix_ids=["m1", "m2"],
        condition_ids=["c1", "c2"],
        pair_ids=["p1", "p2"],
        cell_ids=["x", "y"],
        batch_ids=[1, 2],
        drug_a_ids=["a", "b"],
        drug_b_ids=["c", "d"],
        dose_a_um=np.array([[1, 2, 3, 4]] * n, dtype=float),
        dose_b_um=np.array([[5, 6, 7, 8]] * n, dtype=float),
        y_true=np.zeros((n, 4, 4)),
        y_pred=np.ones((n, 4, 4)),
    )
    assert len(frame) == 32
    assert frame.groupby("matrix_id").size().eq(16).all()
    assert set(frame["row_idx"]) == {0, 1, 2, 3}
    assert {"model", "scenario", "seed_or_fold", "matrix_id", "y_true", "y_pred"} <= set(frame.columns)
    assert np.allclose(frame["residual"], 1.0)
