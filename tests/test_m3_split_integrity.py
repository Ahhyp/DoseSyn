"""M3 split partition and leakage tests (S1/S2/S3)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

KEY = {
    "s1_random_condition": "condition_id",
    "s2_unseen_pair": "pair_id",
    "s3_unseen_cell": "cell_id",
}
SPLITS = ("train", "val", "test")


def _load(split_dir: Path):
    parts = {s: pd.read_parquet(split_dir / f"{s}_conditions.parquet") for s in SPLITS}
    meta = json.loads((split_dir / "split_metadata.json").read_text(encoding="utf-8"))
    return parts, meta


def test_every_split_dir_is_well_formed(m3_split_dirs, m3_primary):
    ids = set(m3_primary["condition_id"])
    for scenario, sdir in m3_split_dirs:
        parts, meta = _load(sdir)
        union = set().union(*(set(p["condition_id"]) for p in parts.values()))
        assert union == ids, f"{sdir}: conditions not partitioned"
        assert sum(len(p) for p in parts.values()) == len(ids)
        for a in SPLITS:
            for b in SPLITS:
                if a < b:
                    assert not (set(parts[a]["condition_id"]) & set(parts[b]["condition_id"]))
        assert meta["scenario"] == scenario
        assert meta["content_sha256"]


def test_matrix_never_spans_splits(m3_split_dirs, m3_primary):
    for _, sdir in m3_split_dirs:
        parts, _ = _load(sdir)
        role = {}
        for s, p in parts.items():
            for cid in p["condition_id"]:
                role[cid] = s
        sub = m3_primary.assign(split=m3_primary["condition_id"].map(role))
        assert sub["split"].notna().all()
        assert (sub.groupby("matrix_id")["split"].nunique() == 1).all()


def test_group_key_never_spans_splits(m3_split_dirs, m3_primary):
    for scenario, sdir in m3_split_dirs:
        parts, _ = _load(sdir)
        role = {cid: s for s, p in parts.items() for cid in p["condition_id"]}
        sub = m3_primary.assign(split=m3_primary["condition_id"].map(role))
        key = KEY[scenario]
        assert (sub.groupby(key)["split"].nunique() == 1).all(), f"{sdir}: {key} leakage"


def test_multi_batch_conditions_stay_together(m3_split_dirs, m3_primary):
    n_multi = int((m3_primary.groupby("condition_id")["batch_id"].nunique() > 1).sum())
    assert n_multi > 0
    for _, sdir in m3_split_dirs:
        parts, _ = _load(sdir)
        role = {cid: s for s, p in parts.items() for cid in p["condition_id"]}
        sub = m3_primary.assign(split=m3_primary["condition_id"].map(role))
        assert (sub.groupby("condition_id")["split"].nunique() == 1).all()


def test_s1_and_s2_are_group_disjoint(m3_split_dirs):
    seen = {}
    for scenario, sdir in m3_split_dirs:
        if scenario == "s3_unseen_cell":
            continue
        parts, _ = _load(sdir)
        key = KEY[scenario]
        tr, va, te = (set(parts[s][key]) for s in SPLITS)
        assert not (tr & va) and not (tr & te) and not (va & te)
        seen.setdefault(scenario, set()).add(
            tuple(sorted([(k, "train" if k in tr else "val" if k in va else "test") for k in tr | va | te]))
        )
    for scenario, variants in seen.items():
        assert len(variants) >= 2, f"{scenario}: seeds produced identical assignments"


def test_s1_condition_fractions(m3_split_dirs):
    for scenario, sdir in m3_split_dirs:
        if scenario != "s1_random_condition":
            continue
        parts, _ = _load(sdir)
        n = sum(len(p) for p in parts.values())
        assert abs(len(parts["train"]) / n - 0.70) < 0.01
        assert abs(len(parts["val"]) / n - 0.15) < 0.01
        assert abs(len(parts["test"]) / n - 0.15) < 0.01


def test_s2_every_pair_is_split_atom(m3_split_dirs, m3_primary):
    for scenario, sdir in m3_split_dirs:
        if scenario != "s2_unseen_pair":
            continue
        parts, _ = _load(sdir)
        role = {cid: s for s, p in parts.items() for cid in p["condition_id"]}
        sub = m3_primary.assign(split=m3_primary["condition_id"].map(role))
        assert (sub.groupby("pair_id")["split"].nunique() == 1).all()
        # every condition of a pair (across cell lines) lands in the same split
        assert sub.groupby(["pair_id", "cell_id"])["split"].nunique().max() == 1


def test_s3_cells_are_disjoint_and_cover_all(m3_split_dirs):
    folds = []
    all_cells = set()
    for scenario, sdir in m3_split_dirs:
        if scenario != "s3_unseen_cell":
            continue
        parts, meta = _load(sdir)
        sets = {s: set(parts[s]["cell_id"]) for s in SPLITS}
        assert not (sets["train"] & sets["val"])
        assert not (sets["train"] & sets["test"])
        assert not (sets["val"] & sets["test"])
        assert sets["test"] == set(meta["fold_cells"]["test"])
        assert sets["val"] == set(meta["fold_cells"]["val"])
        assert sets["train"] == set(meta["fold_cells"]["train"])
        folds.append(sets["test"])
        all_cells |= sets["train"] | sets["val"] | sets["test"]
    assert len(folds) == 5
    assert set().union(*folds) == all_cells
    assert sum(len(f) for f in folds) == len(all_cells)  # every cell is test exactly once


def test_counts_match_metadata(m3_split_dirs):
    for _, sdir in m3_split_dirs:
        parts, meta = _load(sdir)
        for s in SPLITS:
            assert len(parts[s]) == meta["counts"][s]["n_conditions"]
            assert parts[s]["condition_id"].is_unique
