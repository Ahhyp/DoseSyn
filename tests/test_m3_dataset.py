"""M3 data loader smoke tests on real splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dosesyn.data.dataset import DoseSynDataset, collate_matrices, make_loader, smoke_test


def _ds(scenario, split="train", **kw):
    return DoseSynDataset(scenario=scenario, split=split, **kw)


def test_loader_smoke_all_scenarios():
    cases = [
        ("s1_random_condition", {"seed": 13}),
        ("s2_unseen_pair", {"seed": 42}),
        ("s3_unseen_cell", {"fold": 2}),
    ]
    for scenario, kw in cases:
        report = smoke_test(scenario, batch_size=4, **kw)
        assert report["passed"], (scenario, report["checks"])


def test_loader_matches_raw_labels_and_ids(m3_cfg):
    ds = _ds("s1_random_condition", seed=13)
    item = ds[0]
    mid = item["matrix_id"]
    pts = pd.read_parquet(
        Path(m3_cfg["inputs"]["combo_points"]),
        columns=["matrix_id", "row_idx", "col_idx", "x_x0_raw", "conc_a_um", "conc_b_um"],
    )
    sub = pts[pts["matrix_id"] == mid].sort_values(["row_idx", "col_idx"])
    y = item["y"].numpy()
    np.testing.assert_allclose(y, sub["x_x0_raw"].to_numpy().reshape(4, 4), rtol=1e-6)
    # raw X/X0 is preserved; no clipping and no sigmoid
    assert float(y.max()) <= 2.2
    assert float(y.min()) >= 0.0
    assert item["cell_line"] in set(ds.conditions["cell_line"])


def test_loader_cell_features_use_fitted_scaler(m3_cfg):
    ds = _ds("s1_random_condition", seed=13)
    cell = ds.conditions.iloc[0]
    vec = ds.cell_vectors[str(cell["depmap_id"])]
    assert len(vec) == len(ds.cell_feature_names) == 50
    path = pd.read_parquet(Path(m3_cfg["inputs"]["pathway_features"])).set_index("DepMap_ID")
    raw = path.loc[str(cell["depmap_id"]), ds.cell_feature_names].to_numpy(dtype=float)
    want = (raw - np.asarray(ds.scalers["pathway"]["mean"])) / np.asarray(ds.scalers["pathway"]["scale"])
    np.testing.assert_allclose(vec, want, rtol=1e-6)


def test_collate_batch_has_two_graphs_per_matrix():
    ds = _ds("s2_unseen_pair", split="val", seed=13)
    batch = next(iter(make_loader(ds, batch_size=3, shuffle=False)))
    b = len(batch["matrix_id"])
    assert b == 3
    assert batch["drug_batch"].max().item() + 1 == 2 * b
    assert batch["drug_role"].shape[0] == 2 * b
    assert batch["cell_features"].shape[0] == b
    assert batch["y"].shape == (b, 4, 4)
    assert batch["dose_a"].shape == (b, 4)


def test_dataset_lengths_match_condition_files():
    for scenario, lab, kw in [
        ("s1_random_condition", "seed_13", {"seed": 13}),
        ("s3_unseen_cell", "fold_0", {"fold": 0}),
    ]:
        ds = _ds(scenario, split="test", **kw)
        n_cond = pd.read_parquet(ds.split_dir / "test_conditions.parquet")["condition_id"].nunique()
        assert len(ds) > 0
        assert ds.conditions["condition_id"].nunique() == n_cond
        assert len(ds) * 16 == len(ds.points)


def test_dataset_does_not_require_full_expression_by_default():
    ds = _ds("s1_random_condition", seed=13)
    assert ds.include_expression is False
    assert len(ds.cell_feature_names) == 50
