"""Independent leakage / integrity verifier for M3 split directories.

This module re-reads the frozen primary index and every split artefact.  It does
not trust the split metadata: counts and group integrity are recomputed from
the emitted condition files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd

from ..features.transform_train_only import assert_train_only
from ..utils.config import load_config
from ..utils.io import read_json, sha256_file, utc_now_iso, write_json

SCENARIO_KEYS = {
    "s1_random_condition": "condition_id",
    "s2_unseen_pair": "pair_id",
    "s3_unseen_cell": "cell_id",
}
SPLITS = ("train", "val", "test")


def _resolve(cfg: Dict[str, Any], value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def _read_split(split_dir: Path) -> Dict[str, pd.DataFrame]:
    out = {}
    for s in SPLITS:
        p = split_dir / f"{s}_conditions.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        out[s] = pd.read_parquet(p)
    return out


def _check(name: str, ok: bool, detail: Any = None) -> Dict[str, Any]:
    return {"name": name, "status": "PASS" if ok else "FAIL", "detail": detail}


def verify_split_dir(
    split_dir: Path,
    scenario: str,
    primary: pd.DataFrame,
    cond_all: pd.DataFrame,
    cfg: Dict[str, Any],
    points: pd.DataFrame,
) -> Dict[str, Any]:
    meta = read_json(split_dir / "split_metadata.json")
    parts = _read_split(split_dir)
    mono = pd.read_parquet(split_dir / "mono_assignments.parquet")
    scalers = joblib.load(split_dir / "scalers.joblib")
    checks: List[Dict[str, Any]] = []

    # 1) exact partition of conditions
    all_ids = set(cond_all["condition_id"])
    union = set().union(*(set(p["condition_id"]) for p in parts.values()))
    dup = sum(len(set(parts[a]["condition_id"]) & set(parts[b]["condition_id"])) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1 :])
    checks.append(_check("conditions_partition_exactly", union == all_ids and dup == 0, {"n_expected": len(all_ids), "n_seen": len(union), "duplicates": dup}))

    # 2) matrix-level partition (every matrix of a condition moves together)
    c2s = {r.condition_id: r.split for r in pd.concat([p.assign(split=s) for s, p in parts.items()]).itertuples(index=False)}
    primary = primary.copy()
    primary["split"] = primary["condition_id"].map(c2s)
    checks.append(_check("every_matrix_assigned", primary["split"].notna().all(), {"n_unassigned": int(primary["split"].isna().sum())}))
    mat_span = primary.groupby("matrix_id")["split"].nunique()
    checks.append(_check("matrix_never_spans_splits", bool((mat_span == 1).all()), {"n_span": int((mat_span > 1).sum())}))

    # 3) group integrity
    key = SCENARIO_KEYS[scenario]
    group_span = primary.groupby(key)["split"].nunique()
    checks.append(_check(f"{key}_never_spans_splits", bool((group_span == 1).all()), {"n_groups": int(len(group_span)), "n_span": int((group_span > 1).sum())}))

    # 4) batch repeats of a condition stay together (condition is the split atom)
    multi_batch = primary.groupby("condition_id")["batch_id"].nunique()
    cond_span = primary.groupby("condition_id")["split"].nunique()
    multi_ids = multi_batch[multi_batch > 1].index
    bad_multi = int((cond_span.reindex(multi_ids) > 1).sum())
    checks.append(
        _check(
            "batch_repeats_stay_together",
            bad_multi == 0,
            {"n_multi_batch_conditions": int(len(multi_ids)), "n_split": bad_multi},
        )
    )

    # 5) S3 cells are mutually exclusive
    if scenario == "s3_unseen_cell":
        cells = {s: set(p["cell_id"]) for s, p in parts.items()}
        overlap = (len(cells["train"] & cells["val"]), len(cells["train"] & cells["test"]), len(cells["val"] & cells["test"]))
        checks.append(_check("cells_disjoint_across_splits", all(x == 0 for x in overlap), {"n_cells": {k: len(v) for k, v in cells.items()}, "overlap": overlap}))
        checks.append(_check("no_cell_in_multiple_splits", len(cells["train"] | cells["val"] | cells["test"]) == sum(len(v) for v in cells.values()), None))
        fold_cells = meta.get("fold_cells") or {}
        checks.append(_check("fold_cells_match_files", {k: sorted(v) for k, v in cells.items()} == {k: sorted(fold_cells.get(k, [])) for k in SPLITS}, None))

    # 6) sealed test/val groups never leak into training rows
    train_groups = set(parts["train"][key])
    val_groups = set(parts["val"][key])
    test_groups = set(parts["test"][key])
    checks.append(_check("train_groups_disjoint", not (train_groups & val_groups) and not (train_groups & test_groups) and not (val_groups & test_groups), {"n_train": len(train_groups), "n_val": len(val_groups), "n_test": len(test_groups)}))

    # 7) scalers were fitted on training entities only
    train_cells = sorted(set(parts["train"]["cell_id"]))
    train_depmaps = sorted(set(parts["train"]["depmap_id"]))
    try:
        assert_train_only(scalers["pathway"], train_depmaps)
        assert_train_only(scalers["expression"], train_depmaps)
        checks.append(_check("cell_scalers_fit_on_train_cells_only", True, {"n_cells": len(train_depmaps)}))
    except AssertionError as exc:
        checks.append(_check("cell_scalers_fit_on_train_cells_only", False, str(exc)))
    checks.append(_check("scaler_fit_conditions_match_train", int(scalers["meta"]["fit_conditions"]) == len(parts["train"]), {"fit": scalers["meta"]["fit_conditions"], "train": len(parts["train"])}))
    checks.append(_check("scaler_fit_cells_match_train", sorted(scalers["meta"]["fit_cells"]) == train_depmaps, {"fit": len(scalers["meta"]["fit_cells"]), "train": len(train_depmaps)}))

    # 8) mono labels
    trainable = mono[mono["trainable"]]
    role = {r.cell_id: r.split for r in pd.concat([p.assign(split=s) for s, p in parts.items()]).itertuples(index=False)}
    if scenario == "s3_unseen_cell":
        bad = trainable["cell_id"].map(role).ne("train")
        checks.append(_check("mono_trainable_only_train_cells", not bad.any(), {"n_bad": int(bad.sum())}))
        checks.append(_check("mono_val_test_labels_sealed", not bool(mono.loc[mono["cell_id"].map(role).isin(["val", "test"]), "trainable"].any()), None))
    else:
        checks.append(_check("mono_known_prior_fully_trainable", bool(mono["trainable"].all()), {"policy": meta.get("mono_policy")}))
    checks.append(_check("mono_assignments_cover_curves", len(mono) == int(meta["frozen_primary"]["n_mono_curves"]), {"n": len(mono)}))
    checks.append(_check("mono_scaler_fit_matches_trainable", scalers["meta"]["fit_mono_curves"] == int(mono["trainable"].sum()), {"fit": scalers["meta"]["fit_mono_curves"], "trainable": int(mono["trainable"].sum())}))

    # 9) counts recomputed
    recomputed = {s: int(len(parts[s])) for s in SPLITS}
    meta_counts = {s: int(meta["counts"][s]["n_conditions"]) for s in SPLITS}
    checks.append(_check("counts_match_metadata", recomputed == meta_counts, {"files": recomputed, "metadata": meta_counts}))

    # 10) labels present and finite
    pts = points[points["matrix_id"].isin(set(primary["matrix_id"]))]
    merged = pts.merge(primary[["matrix_id", "split"]], on="matrix_id", how="left")
    checks.append(_check("labels_finite_for_all_points", bool(np.isfinite(merged["x_x0_raw"]).all()), {"n": len(merged)}))

    return {
        "scenario": scenario,
        "split_dir": str(split_dir),
        "generated_at": utc_now_iso(),
        "checks": checks,
        "passed": all(c["status"] == "PASS" for c in checks),
    }


def verify_all(cfg_path: str, logger=None) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    primary = pd.read_parquet(_resolve(cfg, cfg["outputs"]["matrix_index"]))
    # Condition universe is taken from the frozen matrix index (independent of splits).
    cond_all = primary[["condition_id", "pair_id", "cell_id"]].drop_duplicates("condition_id")
    points = pd.read_parquet(
        _resolve(cfg, cfg["inputs"]["combo_points"]),
        columns=["matrix_id", "condition_id", "x_x0_raw"],
    )
    points = points[points["matrix_id"].isin(set(primary["matrix_id"]))].reset_index(drop=True)

    split_root = _resolve(cfg, cfg["outputs"]["splits_dir"])
    results = []
    for scenario in SCENARIO_KEYS:
        sdir = split_root / scenario
        if not sdir.exists():
            results.append({"scenario": scenario, "passed": False, "checks": [_check("split_dir_exists", False, str(sdir))]})
            continue
        for sub in sorted(p for p in sdir.iterdir() if p.is_dir()):
            if not (sub / "split_metadata.json").exists():
                continue
            results.append(verify_split_dir(sub, scenario, primary, cond_all, cfg, points))
    report = {
        "version": "m3-leakage-v1",
        "generated_at": utc_now_iso(),
        "git_commit": read_json(_resolve(cfg, cfg["outputs"]["manifest"]))["git_commit"] if _resolve(cfg, cfg["outputs"]["manifest"]).exists() else None,
        "n_splits": len(results),
        "n_passed": sum(1 for r in results if r["passed"]),
        "overall": "PASS" if results and all(r["passed"] for r in results) else "FAIL",
        "splits": results,
    }
    report_path = _resolve(cfg, cfg["outputs"]["leakage_report"])
    write_json(report, report_path)
    if logger:
        logger.info("leakage report: %s (%d/%d splits pass)", report_path, report["n_passed"], report["n_splits"])
    return report
