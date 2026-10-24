"""M3 monotherapy supervision policy tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

SPLITS = ("train", "val", "test")


def _parts(split_dir):
    parts = {s: pd.read_parquet(split_dir / f"{s}_conditions.parquet") for s in SPLITS}
    mono = pd.read_parquet(split_dir / "mono_assignments.parquet")
    meta = json.loads((split_dir / "split_metadata.json").read_text(encoding="utf-8"))
    return parts, mono, meta


def test_s1_s2_use_known_monotherapy(m3_split_dirs):
    for scenario, sdir in m3_split_dirs:
        if scenario == "s3_unseen_cell":
            continue
        parts, mono, meta = _parts(sdir)
        assert meta["mono_policy"] == "known_monotherapy"
        assert mono["trainable"].all()
        assert set(mono["label_access"]) == {"known_prior"}
        assert mono["mono_curve_id"].is_unique
        assert "not an independent test" in mono["reason"].iloc[0]


def test_s3_seals_val_test_cell_monotherapy(m3_split_dirs):
    for scenario, sdir in m3_split_dirs:
        if scenario != "s3_unseen_cell":
            continue
        parts, mono, meta = _parts(sdir)
        assert meta["mono_policy"] == "strict_cell_isolation"
        role = {cid: s for s, p in parts.items() for cid in p["condition_id"]}
        cell_role = {cid: s for s, p in parts.items() for cid in p["cell_id"]}
        mono_role = mono["cell_id"].map(cell_role)
        assert mono.loc[mono["trainable"], "cell_id"].map(cell_role).eq("train").all()
        assert not mono.loc[mono_role.isin(["val", "test"]), "trainable"].any()
        assert set(mono.loc[~mono["trainable"], "cell_id"]) == set(parts["val"]["cell_id"]) | set(parts["test"]["cell_id"])
        assert meta["mono"]["n_held_out_curves"] == int((~mono["trainable"]).sum())


def test_s3_trainable_curves_belong_to_train_cells(m3_split_dirs):
    for scenario, sdir in m3_split_dirs:
        if scenario != "s3_unseen_cell":
            continue
        parts, mono, _ = _parts(sdir)
        train_cells = set(parts["train"]["cell_id"])
        assert set(mono.loc[mono["trainable"], "cell_id"]) <= train_cells
        assert not (set(mono.loc[mono["trainable"], "cell_id"]) & (set(parts["test"]["cell_id"]) | set(parts["val"]["cell_id"])))


def test_mono_assignments_cover_every_primary_curve(m3_split_dirs, m3_cfg):
    n_primary = int(pd.read_parquet(Path(m3_cfg["outputs"]["mono_index"])).shape[0])
    for _, sdir in m3_split_dirs:
        _, mono, _ = _parts(sdir)
        assert len(mono) == n_primary
        assert mono["mono_curve_id"].is_unique
