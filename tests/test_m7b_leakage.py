"""M7-B leakage/consistency tests beyond the run-contract checks."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from dosesyn.train.m7b_masks import MASK_SPECS, all_masks, indices
from dosesyn.train.m7b_trainer import masked_huber

ROOT = Path(__file__).resolve().parents[1]


def test_masked_huber_ignores_held_targets():
    pred = torch.zeros(2, 16)
    target = torch.zeros(2, 16)
    obs = torch.zeros(2, 16)
    obs[:, indices("center_holdout_75", observed=True)] = 1.0
    # Held-out targets are huge; loss must not change because mask is zero there.
    target[:, indices("center_holdout_75")] = 100.0
    a = masked_huber(pred, target, obs, delta=1.0)
    target2 = target.clone()
    target2[:, indices("center_holdout_75")] = -50.0
    b = masked_huber(pred, target2, obs, delta=1.0)
    assert torch.isfinite(a) and abs(float(a - b)) < 1e-8


def test_masks_are_transpose_symmetric():
    for name, spec in all_masks().items():
        held = set(spec["held_indices"])
        for k in held:
            r, c = divmod(k, 4)
            assert r * 4 + c in held
            assert c * 4 + r in held, (name, k)
            assert (c * 4 + r) in held


def test_all_center_runs_share_mask_version():
    run_dirs = [p for p in (ROOT / "outputs/runs/m7b").glob("m7b__center_holdout_75__*") if p.is_dir()]
    if not run_dirs:
        return
    versions = set()
    for p in run_dirs:
        meta_p = p / "run_metadata.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        versions.add(meta["mask_version"])
        assert meta["mask_held"] == indices("center_holdout_75")
        assert meta["test_policy"] == "single_read_after_checkpoint_selection"
    assert versions == {all_masks()["center_holdout_75"]["version"]}
