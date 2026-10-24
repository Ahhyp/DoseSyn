"""M5 monotherapy label protocol: known-prior vs strict cell isolation."""

from __future__ import annotations

import pytest

from dosesyn.train.m5_data import (
    MonoAccessError,
    load_mono_assignments,
    load_mono_labels_for_audit,
    load_mono_table,
    mono_access_summary,
)


def test_s3_fold0_heldout_labels_are_sealed():
    assignments = load_mono_assignments("s3_unseen_cell", fold=0)
    held = assignments[~assignments["trainable"].astype(bool)]["mono_curve_id"].astype(str).tolist()
    assert len(held) == 792
    with pytest.raises(MonoAccessError):
        load_mono_labels_for_audit("s3_unseen_cell", held[:3], fold=0)


def test_s3_fold0_trainable_only():
    table = load_mono_table("s3_unseen_cell", fold=0)
    assert len(table) == 990
    assignments = load_mono_assignments("s3_unseen_cell", fold=0)
    trainable = set(
        assignments.loc[assignments["trainable"].astype(bool), "mono_curve_id"].astype(str)
    )
    assert set(table.curve_ids) <= trainable
    assert table.dose_std.shape == (990, 8)
    assert table.y.shape == (990, 8)
    assert table.cell_features.shape == (990, 50)


def test_s1_known_monotherapy_all_trainable():
    table = load_mono_table("s1_random_condition", seed=42)
    assert len(table) == 1782
    summary = mono_access_summary(load_mono_assignments("s1_random_condition", seed=42))
    assert summary["n_held_out"] == 0
    assert summary["n_trainable"] == 1782


def test_trainable_labels_can_be_read():
    assignments = load_mono_assignments("s3_unseen_cell", fold=0)
    ok = assignments[assignments["trainable"].astype(bool)]["mono_curve_id"].astype(str).tolist()
    df = load_mono_labels_for_audit("s3_unseen_cell", ok[:2], fold=0)
    assert len(df) == 16  # two 8-point curves
    assert df["x_x0_raw"].notna().all()
