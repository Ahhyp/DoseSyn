"""M7-B run contract tests (skipped until a run exists)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = sorted((ROOT / "outputs/runs/m7b").glob("m7b__*"))


@pytest.mark.skipif(not RUNS, reason="no M7-B runs yet")
def test_every_run_has_16_predictions_per_matrix():
    for run in RUNS:
        for split in ("val", "test"):
            p = run / f"predictions_{split}.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p, columns=["matrix_id", "is_held", "row_idx", "col_idx"])
            counts = df.groupby("matrix_id").size()
            assert (counts == 16).all(), run.name
            assert df.isna().sum().sum() == 0
            assert df.is_held.isin([0, 1]).all()


@pytest.mark.skipif(not RUNS, reason="no M7-B runs yet")
def test_run_metadata_traceable():
    for run in RUNS:
        p = run / "run_metadata.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        assert m["mask_version"]
        assert m["selection_metric"] == "val_held_matrix_macro_rmse"
        assert m["test_policy"] == "single_read_after_checkpoint_selection"
        assert m["mask_held"]
        assert len(m["mask_held"]) + len(m["mask_observed"]) == 16


@pytest.mark.skipif(not RUNS, reason="no M7-B runs yet")
def test_held_prediction_metric_matches_saved():
    for run in RUNS:
        p = run / "predictions_test.parquet"
        m = run / "test_metrics.json"
        if not (p.exists() and m.exists()):
            continue
        df = pd.read_parquet(p)
        held = df[df.is_held.astype(bool)]
        per = held.groupby("matrix_id").apply(lambda d: float((d.residual ** 2).mean() ** 0.5))
        saved = json.loads(m.read_text())["held"]["matrix_macro_rmse"]
        assert abs(per.mean() - saved) < 1e-5
