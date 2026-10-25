"""M7-A external baseline contract tests (no GPU, no training)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.training.metrics import compute_metrics  # noqa: E402

PRIMARY = ROOT / "data/processed/oneil_primary_v1_matrix_index.parquet"
COMBO_VAL = ROOT / "outputs/runs/m7a/combokr/s1_seed42_val/predictions.parquet"


def test_config_declares_main_and_supplementary():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/m7a/m7a.yaml").read_text())
    models = cfg["models"]
    assert models["combokr"]["status"] == "main_ranking"
    assert models["combokr"]["s3_status"] == "blocked_protocol"
    assert models["combokr"]["scenarios"] == ["s1_random_condition", "s2_unseen_pair"]
    assert models["deepsynba_adapted"]["status"] == "supplementary"
    assert "MoLFormer" in models["deepsynba_adapted"]["protocol_difference"][0]


def test_frozen_split_unchanged():
    meta = json.loads((ROOT / "data/splits/s1_random_condition/seed_42/split_metadata.json").read_text())
    counts = meta["counts"]
    assert counts["train"]["n_matrices"] == 13022
    assert counts["val"]["n_matrices"] == 2791
    assert counts["test"]["n_matrices"] == 2786


def test_metric_suite_manual_macro():
    rng = np.random.default_rng(0)
    y = rng.random((5, 16))
    p = y + rng.normal(0, 0.1, size=y.shape)
    m = compute_metrics(y, p, matrix_ids=[f"m{i}" for i in range(5)],
                        condition_ids=[f"c{i}" for i in range(5)])
    per = np.sqrt(((y - p) ** 2).mean(axis=1))
    assert m["matrix_macro_rmse"] == pytest.approx(float(per.mean()))
    assert m["point_rmse"] == pytest.approx(float(np.sqrt(((y - p) ** 2).mean())))


@pytest.mark.skipif(not COMBO_VAL.exists(), reason="comboKR validation predictions not available")
def test_combokr_prediction_contract_and_mapping():
    pred = pd.read_parquet(COMBO_VAL)
    assert {"matrix_id", "row_idx", "col_idx", "dose_a_um", "dose_b_um",
            "y_true", "y_pred"}.issubset(pred.columns)
    assert len(pred) % 16 == 0
    assert pred.matrix_id.nunique() == len(pred) // 16
    assert pred[["y_true", "y_pred"]].notna().all().all()
    assert pred.row_idx.between(0, 3).all() and pred.col_idx.between(0, 3).all()
    primary = pd.read_parquet(PRIMARY, columns=["matrix_id"])
    assert set(pred.matrix_id.unique()).issubset(set(primary.matrix_id))


@pytest.mark.skipif(not (ROOT / "outputs/runs/m7a/combokr/s1_seed42_val/run_metadata.json").exists(),
                    reason="comboKR metadata missing")
def test_combokr_train_eval_disjoint():
    meta = json.loads((ROOT / "outputs/runs/m7a/combokr/s1_seed42_val/run_metadata.json").read_text())
    train = set(meta["train_matrix_ids"])
    ev = set(meta["eval_matrix_ids"])
    assert train and ev
    assert not (train & ev)


@pytest.mark.skipif(not (ROOT / "outputs/runs/m7a/deepsynba").exists(),
                    reason="DeepSynBa runs not started")
def test_deepsynba_predictions_have_full_16_rows():
    files = list((ROOT / "outputs/runs/m7a/deepsynba").glob("*/*/predictions.parquet"))
    if not files:
        pytest.skip("no DeepSynBa predictions")
    for f in files:
        df = pd.read_parquet(f)
        assert len(df) % 16 == 0
        assert df.y_pred.notna().all()
        assert df.matrix_id.nunique() == len(df) // 16
