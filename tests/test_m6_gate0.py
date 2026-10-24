"""M6-0 gate tests: split identity, M4/M5 same-validation alignment, frozen artefacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def primary() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data/processed/oneil_primary_v1_matrix_index.parquet")


def _matrix_ids_for(primary: pd.DataFrame, split_dir: Path, split: str) -> set:
    cond = pd.read_parquet(split_dir / f"{split}_conditions.parquet")
    return set(
        primary.loc[primary["condition_id"].isin(set(cond["condition_id"])), "matrix_id"].astype(str)
    )


@pytest.fixture(scope="module")
def seed42(primary) -> dict:
    d = ROOT / "data/splits/s1_random_condition/seed_42"
    meta = json.loads((d / "split_metadata.json").read_text())
    return {
        "dir": d,
        "meta": meta,
        "val": _matrix_ids_for(primary, d, "val"),
        "train": _matrix_ids_for(primary, d, "train"),
        "test": _matrix_ids_for(primary, d, "test"),
    }


@pytest.fixture(scope="module")
def seed13(primary) -> dict:
    d = ROOT / "data/splits/s1_random_condition/seed_13"
    meta = json.loads((d / "split_metadata.json").read_text())
    return {"dir": d, "meta": meta, "val": _matrix_ids_for(primary, d, "val")}


def test_seed42_metadata_counts(seed42):
    counts = seed42["meta"]["counts"]
    assert counts["train"]["n_matrices"] == 13022
    assert counts["val"]["n_matrices"] == 2791
    assert counts["test"]["n_matrices"] == 2786


def test_seed42_val_actual_size(seed42):
    assert len(seed42["val"]) == 2791


def test_seed42_and_seed13_are_different_splits(seed42, seed13):
    assert seed42["val"] != seed13["val"]
    assert len(seed13["val"]) == 2788
    # 同一批 condition 可以重叠，但矩阵集合不应完全相等
    assert len(seed42["val"] & seed13["val"]) < len(seed42["val"])


@pytest.mark.parametrize(
    "run",
    [
        "m5__s1_random_condition__seed_42__c0_control",
        "m5__s1_random_condition__seed_42__c1_symmetric",
        "m5__s1_random_condition__seed_42__c2_fourier",
        "m5__s1_random_condition__seed_42__c3_film",
        "m5__s1_random_condition__seed_42__c4_mono",
        "m5__s1_random_condition__seed_42__c5_boundary",
        "m5__s1_random_condition__seed_42__c6_full",
    ],
)
def test_m5_seed42_uses_seed42_val(run, seed42, seed13):
    path = ROOT / "outputs/runs/m5" / run / "predictions_val.parquet"
    if not path.exists():
        pytest.skip(f"missing M5 artefact {path}")
    pred = pd.read_parquet(path, columns=["matrix_id"])
    ids = set(pred["matrix_id"].astype(str))
    assert ids == seed42["val"]
    assert ids != seed13["val"]
    assert pred.shape[0] == 2791 * 16


def test_m4_b3_seed42_uses_seed42_val(seed42):
    path = ROOT / "outputs/runs/m4/b3_gine__s1_random_condition__seed_42/predictions_val.parquet"
    if not path.exists():
        pytest.skip("missing M4 B3 artefact")
    pred = pd.read_parquet(path, columns=["matrix_id"])
    assert set(pred["matrix_id"].astype(str)) == seed42["val"]
    assert pred.shape[0] == 2791 * 16


def test_m4_m5_same_labels_and_doses(seed42):
    m4 = pd.read_parquet(
        ROOT / "outputs/runs/m4/b3_gine__s1_random_condition__seed_42/predictions_val.parquet"
    )
    m5 = pd.read_parquet(
        ROOT / "outputs/runs/m5/m5__s1_random_condition__seed_42__c0_control/predictions_val.parquet"
    )
    keys = ["matrix_id", "row_idx", "col_idx"]
    merged = m4[keys + ["y_true", "dose_a_um", "dose_b_um"]].merge(
        m5[keys + ["y_true", "dose_a_um", "dose_b_um"]], on=keys, suffixes=("_m4", "_m5")
    )
    assert len(merged) == len(m4) == len(m5)
    assert (merged["y_true_m4"] == merged["y_true_m5"]).all()
    assert (merged["dose_a_um_m4"] == merged["dose_a_um_m5"]).all()
    assert (merged["dose_b_um_m4"] == merged["dose_b_um_m5"]).all()


def test_s3_fold0_same_validation():
    m4 = ROOT / "outputs/runs/m4/b3_gine__s3_unseen_cell__fold_0/predictions_val.parquet"
    m5 = ROOT / "outputs/runs/m5/m5__s3_unseen_cell__fold_0__c0_control/predictions_val.parquet"
    if not (m4.exists() and m5.exists()):
        pytest.skip("missing S3 artefacts")
    a = set(pd.read_parquet(m4, columns=["matrix_id"])["matrix_id"].astype(str))
    b = set(pd.read_parquet(m5, columns=["matrix_id"])["matrix_id"].astype(str))
    assert a == b
    assert len(a) == 3836


def test_metric_json_matches_recomputation():
    from dosesyn.training.metrics import compute_metrics

    pred = pd.read_parquet(
        ROOT / "outputs/runs/m4/b3_gine__s1_random_condition__seed_42/predictions_val.parquet"
    )
    pred = pred.sort_values(["matrix_id", "row_idx", "col_idx"])
    groups = list(pred.groupby("matrix_id", sort=True))
    import numpy as np

    y_true = np.stack([g["y_true"].to_numpy(dtype=float).reshape(4, 4) for _, g in groups])
    y_pred = np.stack([g["y_pred"].to_numpy(dtype=float).reshape(4, 4) for _, g in groups])
    metrics = compute_metrics(
        y_true,
        y_pred,
        matrix_ids=[str(m) for m, _ in groups],
        condition_ids=[str(g["condition_id"].iloc[0]) for _, g in groups],
    )
    stored = json.loads(
        (ROOT / "outputs/runs/m4/b3_gine__s1_random_condition__seed_42/val_metrics.json").read_text()
    )
    assert abs(metrics["matrix_macro_rmse"] - stored["matrix_macro_rmse"]) < 1e-12
