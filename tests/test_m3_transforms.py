"""M3 train-fold-only transform tests."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from dosesyn.features.transform_train_only import (
    assert_train_only,
    concentration_columns,
    fit_standardizer,
    scaler_fingerprint,
    transform_dose,
    transform_standardizer,
)

SPLITS = ("train", "val", "test")


def _load(split_dir):
    parts = {s: pd.read_parquet(split_dir / f"{s}_conditions.parquet") for s in SPLITS}
    scalers = joblib.load(split_dir / "scalers.joblib")
    return parts, scalers


def test_scalers_are_fit_on_train_cells_only(m3_split_dirs):
    for _, sdir in m3_split_dirs:
        parts, sc = _load(sdir)
        train = sorted(set(parts["train"]["depmap_id"]))
        assert_train_only(sc["pathway"], train)
        assert_train_only(sc["expression"], train)
        assert sorted(sc["meta"]["fit_cells"]) == train


def test_pathway_mean_and_scale_are_the_train_statistics(m3_split_dirs, m3_cfg):
    pathway = pd.read_parquet(Path(m3_cfg["inputs"]["pathway_features"])).set_index("DepMap_ID")
    for _, sdir in m3_split_dirs:
        parts, sc = _load(sdir)
        train = sorted(set(parts["train"]["depmap_id"]))
        cols = sc["pathway"]["columns"]
        raw = pathway.loc[train, cols].to_numpy()
        np.testing.assert_allclose(sc["pathway"]["mean"], raw.mean(axis=0), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(sc["pathway"]["scale"], raw.std(axis=0, ddof=0), rtol=1e-10, atol=1e-10)
        assert sc["pathway"]["n_fit_rows"] == len(train)


def test_validation_transform_uses_train_parameters(m3_split_dirs, m3_cfg):
    pathway = pd.read_parquet(Path(m3_cfg["inputs"]["pathway_features"])).set_index("DepMap_ID")
    checked = 0
    for _, sdir in m3_split_dirs:
        parts, sc = _load(sdir)
        val = sorted(set(parts["val"]["depmap_id"]))
        cols = sc["pathway"]["columns"]
        raw = pathway.loc[val, cols].to_numpy()
        got = transform_standardizer(raw, sc["pathway"])
        want = (raw - np.asarray(sc["pathway"]["mean"])) / np.asarray(sc["pathway"]["scale"])
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
        checked += 1
    assert checked >= 3


def test_dose_scaler_uses_train_matrices(m3_cfg, m3_input_manifest):
    from dosesyn.splits.build import SCENARIOS  # noqa: F401

    root = Path(m3_cfg["outputs"]["splits_dir"]) / "s1_random_condition" / "seed_13"
    parts, sc = _load(root)
    points = pd.read_parquet(
        Path(m3_cfg["inputs"]["combo_points"]),
        columns=["matrix_id", "condition_id", "conc_a_um"],
    )
    train_mids = sorted(points.loc[points["condition_id"].isin(set(parts["train"]["condition_id"])), "matrix_id"].unique())
    assert sorted(sc["drug_a_dose"]["fit_matrix_ids"]) == train_mids
    assert sc["meta"]["fit_dose_matrix_ids"] == len(train_mids)
    # scaling must use the train mean, not a validation statistic
    train_vals = points.loc[points["condition_id"].isin(set(parts["train"]["condition_id"])), "conc_a_um"].to_numpy()
    mean_log = np.log10(train_vals[train_vals > 0] * 1e-6).mean()
    assert abs(sc["drug_a_dose"]["mean"] - mean_log) < 1e-9


def test_s3_scaler_excludes_test_cells(m3_split_dirs):
    for scenario, sdir in m3_split_dirs:
        if scenario != "s3_unseen_cell":
            continue
        parts, sc = _load(sdir)
        fit = set(sc["meta"]["fit_cells"])
        assert fit == set(parts["train"]["depmap_id"])
        assert not fit & set(parts["test"]["depmap_id"])
        assert not fit & set(parts["val"]["depmap_id"])


def test_zero_dose_and_log10_handling():
    out = concentration_columns(np.array([0.0, 1e-6, 1.0]))
    assert out["zero_flag"].tolist() == [1, 0, 0]
    assert np.isnan(out["log10_molar"][0])
    assert out["log10_molar"][1] == pytest.approx(-12.0)  # 1e-6 µM = 1e-12 M
    assert out["log10_molar"][2] == pytest.approx(-6.0)   # 1 µM = 1e-6 M


def test_fit_imputes_train_median_only():
    x = np.array([[1.0, np.nan], [3.0, 5.0], [5.0, 7.0]])
    sc = fit_standardizer(x, ["a", "b"], ["c1", "c2", "c3"])
    assert sc["median"] == [3.0, 6.0]
    assert sc["mean"] == [3.0, 6.0]
    assert sc["n_missing_values"] == 1
    assert scaler_fingerprint(sc) == scaler_fingerprint({**sc})
    assert "fit_ids" in sc and sc["fit_ids"] == ["c1", "c2", "c3"]
