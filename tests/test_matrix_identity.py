"""Matrix/curve identity tests: 4x4 completeness, uniqueness, ranking, batch repeats."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dosesyn.data.ids import matrix_id, mono_curve_id


def test_combo_matrix_count_and_uniqueness(combo, combo_meta, cfg):
    assert combo_meta["matrix_id"].is_unique
    assert len(combo_meta) == cfg["expected_counts"]["combo_matrices"] == 23052
    assert combo["matrix_id"].nunique() == 23052
    # One matrix per (batch, cell, ordered pair, dose grid A, dose grid B).
    keys = ["batch_id", "cell_id", "pair_id", "dose_a_um_json", "dose_b_um_json"]
    assert not combo_meta.duplicated(keys).any()
    # matrix_id is a deterministic function of those keys.
    sample = combo_meta.head(200)
    for r in sample.itertuples(index=False):
        expected = matrix_id(
            int(r.batch_id), r.cell_id, r.drug_a_id, r.drug_b_id,
            tuple(json.loads(r.dose_a_um_json)), tuple(json.loads(r.dose_b_um_json)),
        )
        assert expected == r.matrix_id


def test_every_combo_matrix_is_complete_4x4(combo, combo_meta):
    assert combo_meta["complete_matrix"].all(), (
        combo_meta.loc[~combo_meta["complete_matrix"], ["matrix_id", "n_points", "n_dose_a", "n_dose_b"]]
        .head(10)
        .to_string()
    )
    sizes = combo.groupby("matrix_id").size()
    assert (sizes == 16).all()
    assert (combo.groupby(["matrix_id", "conc_a_um", "conc_b_um"]).size() == 1).all()
    # Every matrix covers all 4x4 rank combinations exactly once.
    ranks = combo.groupby("matrix_id")[["row_idx", "col_idx"]].apply(
        lambda d: len(set(zip(d["row_idx"], d["col_idx"])))
    )
    assert (ranks == 16).all()
    assert combo["row_idx"].between(0, 3).all()
    assert combo["col_idx"].between(0, 3).all()
    for mid in combo_meta["matrix_id"].head(100):
        sub = combo.loc[combo.matrix_id == mid]
        assert sub["row_idx"].nunique() == 4
        assert sub["col_idx"].nunique() == 4


def test_row_and_col_indices_match_sorted_doses(combo):
    sub = combo.loc[combo.matrix_id == combo["matrix_id"].iloc[0]]
    dose_a_rank = sub["conc_a_um"].rank(method="dense").astype(int) - 1
    dose_b_rank = sub["conc_b_um"].rank(method="dense").astype(int) - 1
    assert (sub["row_idx"] == dose_a_rank).all()
    assert (sub["col_idx"] == dose_b_rank).all()


def test_source_rows_are_contiguous_blocks(combo, mono):
    for frame, key in ((combo, "matrix_id"), (mono, "mono_curve_id")):
        g = frame.groupby(key)["source_row"]
        span = g.max() - g.min() + 1
        assert (span == g.size()).all(), f"{key} is not a contiguous source block"


def test_mono_curve_count_and_structure(mono, mono_meta, cfg):
    assert mono_meta["mono_curve_id"].is_unique
    assert len(mono_meta) == cfg["expected_counts"]["mono_curves"] == 2094
    assert mono_meta["complete_curve"].all()
    assert (mono.groupby("mono_curve_id").size() == 8).all()
    assert (mono.groupby(["mono_curve_id", "conc_um"]).size() == 1).all()
    assert mono["row_idx"].between(0, 7).all()
    # deterministic ID
    r = mono_meta.iloc[0]
    expected = mono_curve_id(
        int(r.batch_id), r.cell_id, r.drug_id,
        tuple(json.loads(r.dose_um_json)),
    )
    assert expected == r.mono_curve_id


def test_condition_and_pair_cardinality(combo, cfg):
    assert combo["condition_id"].nunique() == 22737
    assert combo["pair_id"].nunique() == cfg["expected_counts"]["n_unordered_pairs"] == 583
    # One condition_id per (pair, cell).
    pairs = combo[["condition_id", "pair_id", "cell_id"]].drop_duplicates()
    assert not pairs.duplicated(["condition_id"]).any()
    assert not pairs.duplicated(["pair_id", "cell_id"]).any()


def test_pair_id_is_symmetric_and_canonical(combo):
    # The source lists each unordered pair once, so no both-orientation case is
    # expected; verify the ID itself is orientation-free.
    from dosesyn.data.ids import unordered_pair_id

    a = combo["drug_a_id"].iloc[0]
    b = combo["drug_b_id"].iloc[0]
    p1, low1, high1 = unordered_pair_id(a, b)
    p2, low2, high2 = unordered_pair_id(b, a)
    assert p1 == p2 and low1 == low2 and high1 == high2
    both = combo[["pair_id", "drug_a_id", "drug_b_id"]].drop_duplicates()
    assert both.groupby("pair_id").size().max() == 1


def test_batch_repeat_audit_matches_originals(combo, batch_repeats):
    assert len(batch_repeats) == 315
    assert int(batch_repeats["grids_identical"].sum()) == 105
    assert int((~batch_repeats["grids_identical"]).sum()) == 210
    # Every condition flagged as repeated really has >1 batch; all others have 1.
    n_batches = combo.groupby("condition_id")["batch_id"].nunique()
    assert int((n_batches > 1).sum()) == len(batch_repeats)
    assert set(batch_repeats["condition_id"]) == set(n_batches.index[n_batches > 1])
    detail = json.loads(batch_repeats.iloc[0]["batch_detail_json"])
    assert len(detail) == int(batch_repeats.iloc[0]["n_batches"])


def test_grids_are_4_by_4_in_metadata(combo_meta):
    for r in combo_meta.head(500).itertuples(index=False):
        assert len(json.loads(r.dose_a_um_json)) == 4
        assert len(json.loads(r.dose_b_um_json)) == 4
        assert len(json.loads(r.dose_low_um_json)) == 4
        assert len(json.loads(r.dose_high_um_json)) == 4
