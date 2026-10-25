"""M7-B mask, loss-isolation and held-out metric contract tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.train.m7b_masks import MASK_SPECS, all_masks, indices, mask_matrix, mask_version
from dosesyn.train.m7b_trainer import _held_metrics, masked_huber


def test_mask_partition_and_symmetry():
    for name in MASK_SPECS:
        held = set(indices(name))
        obs = set(indices(name, observed=True))
        assert held.isdisjoint(obs)
        assert len(held) + len(obs) == 16
        # transpose maps held -> held and observed -> observed
        assert all(((i % 4) * 4 + i // 4) in held for i in held)
        assert all(((i % 4) * 4 + i // 4) in obs for i in obs)


def test_center_mask_is_exact_four():
    held = set(indices("center_holdout_75"))
    assert held == {5, 6, 9, 10}
    assert len(indices("center_holdout_75", observed=True)) == 12


def test_boundary_mask_is_row_and_column():
    held = set(indices("boundary_holdout_50"))
    assert held == {12, 13, 14, 15, 3, 7, 11}
    assert len(indices("boundary_holdout_50", observed=True)) == 9


def test_mask_matrix_shape_and_version():
    for name in MASK_SPECS:
        m = mask_matrix(name)
        assert m.shape == (4, 4)
        assert int(m.sum()) in (4, 7, 8, 12)
        assert len(mask_version(name)) == 16
    d = all_masks()
    assert set(d) == set(MASK_SPECS)


def test_held_target_cannot_change_training_loss():
    torch.manual_seed(0)
    pred = torch.randn(3, 16, requires_grad=True)
    y1 = torch.randn(3, 16)
    y2 = y1.clone()
    held = indices("center_holdout_75")
    y2[:, held] = torch.randn(3, 4) * 100
    obs = torch.zeros(3, 16)
    obs[:, indices("center_holdout_75", observed=True)] = 1
    l1 = masked_huber(pred, y1, obs)
    l2 = masked_huber(pred, y2, obs)
    assert torch.allclose(l1, l2)
    l1.backward()
    assert pred.grad[:, held].abs().sum() == 0


def test_observed_target_changes_training_loss():
    pred = torch.randn(2, 16)
    y1 = torch.zeros(2, 16)
    y2 = y1.clone()
    y2[:, indices("center_holdout_75", observed=True)] += 1.0
    obs = torch.zeros(2, 16)
    obs[:, indices("center_holdout_75", observed=True)] = 1
    assert not torch.allclose(masked_huber(pred, y1, obs), masked_huber(pred, y2, obs))


def test_held_metrics_uses_only_held_positions():
    n = 5
    rng = np.random.default_rng(0)
    y = rng.normal(size=(n, 16))
    pred = y.copy()
    held = indices("center_holdout_75")
    pred[:, held] += 1.0
    table = SimpleNamespace(
        y=y.reshape(n, 4, 4),
        matrix_ids=np.array([f"m{i}" for i in range(n)]),
        condition_ids=np.array([f"c{i}" for i in range(n)]),
    )
    m = _held_metrics(table, pred, "center_holdout_75")
    assert m["n_points_evaluated"] == n * 4
    assert abs(m["matrix_macro_rmse"] - 1.0) < 1e-9
    assert abs(m["mean_bias"] - 1.0) < 1e-9


def test_shared_mask_across_models_in_config():
    import yaml

    with open(ROOT / "configs/m7b/m7b.yaml", encoding="utf-8") as fh:
        c = yaml.safe_load(fh)
    assert set(c["runs"]["b1_center_all"]["models"]) >= {"m6b_b5_film", "m6b_b1_t3", "m4_b3_gine"}
    assert c["runs"]["b1_center_all"]["mask"] == "center_holdout_75"
    for name in c["runs"]["b1_center_all"]["models"]:
        assert name in c["models"]


def test_s2_and_s3_isolation_still_holds():
    root = ROOT / "data/splits"
    s2 = root / "s2_unseen_pair/seed_42"
    tr = pd.read_parquet(s2 / "train_conditions.parquet", columns=["pair_id"])
    te = pd.read_parquet(s2 / "test_conditions.parquet", columns=["pair_id"])
    assert set(tr.pair_id).isdisjoint(set(te.pair_id))
    s3 = root / "s3_unseen_cell/fold_0"
    tr = pd.read_parquet(s3 / "train_conditions.parquet", columns=["cell_id"])
    te = pd.read_parquet(s3 / "test_conditions.parquet", columns=["cell_id"])
    assert set(tr.cell_id).isdisjoint(set(te.cell_id))
