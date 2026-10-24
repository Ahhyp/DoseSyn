"""Build the frozen primary dataset and leakage-safe M3 splits.

The module never writes into data/raw, data/processed M0/M1/M2 artefacts, or
the feature cache.  It reads those as immutable inputs and writes new files
under data/processed (primary index only) and data/splits.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from ..features.transform_train_only import (
    concentration_columns,
    fit_dose_scaler,
    fit_standardizer,
    scaler_fingerprint,
)
from ..utils.config import load_config
from ..utils.io import sha256_file, utc_now_iso, write_json
from .mono_policy import build_mono_assignments, trainable_curve_ids

SCENARIOS = {
    "s1_random_condition": {
        "key": "condition_id",
        "unit": "condition",
        "mono_policy": "known_monotherapy",
    },
    "s2_unseen_pair": {
        "key": "pair_id",
        "unit": "drug_pair",
        "mono_policy": "known_monotherapy",
    },
    "s3_unseen_cell": {
        "key": "cell_id",
        "unit": "cell_line",
        "mono_policy": "strict_cell_isolation",
    },
}


# ---------------------------------------------------------------------------
# Paths / hashing
# ---------------------------------------------------------------------------
def resolve(cfg: Dict[str, Any], value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def git_commit(root: Path) -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def assignment_fingerprint(condition_splits: pd.DataFrame, mono: pd.DataFrame, extra: Any = None) -> str:
    rows = sorted(
        (str(r.condition_id), str(r.split)) for r in condition_splits.itertuples(index=False)
    )
    mono_rows = sorted(
        (str(r.mono_curve_id), int(bool(r.trainable))) for r in mono.itertuples(index=False)
    )
    return canonical_hash({"conditions": rows, "mono": mono_rows, "extra": extra})


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------
def _cell_lookup(cell_mapping: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if cell_mapping["oneil_cell_name"].duplicated().any():
        raise ValueError("duplicate oneil_cell_name in cell_mapping")
    lookup = {}
    for r in cell_mapping.itertuples(index=False):
        lookup[str(r.oneil_cell_name)] = {
            "cell_line": str(r.oneil_cell_name),
            "standard_cell_name": str(r.standard_cell_name),
            "depmap_id": None if pd.isna(r.depmap_id) else str(r.depmap_id),
            "lineage": None if pd.isna(r.lineage) else str(r.lineage),
            "disease": None if pd.isna(r.disease) else str(r.disease),
            "mapping_status": str(r.mapping_status),
            "feature_eligible": bool(r.feature_eligible),
        }
    return lookup


def _exclusion_reason_matrix(row: pd.Series, excluded_drugs: set) -> Optional[str]:
    if str(row["drug_a_name"]) in excluded_drugs or str(row["drug_b_name"]) in excluded_drugs:
        return "drug_graph_excluded"
    if not bool(row["drug_a_graph_ok"]) or not bool(row["drug_b_graph_ok"]):
        return "drug_graph_not_model_ready"
    if not bool(row["cell_feature_complete"]):
        return "cell_features_incomplete"
    if not bool(row["complete_matrix"]) or int(row["n_points"]) != 16:
        return "incomplete_matrix"
    return None


def _exclusion_reason_mono(row: pd.Series, excluded_drugs: set) -> Optional[str]:
    if str(row["drug_name"]) in excluded_drugs:
        return "drug_graph_excluded"
    if not bool(row["drug_graph_ok"]):
        return "drug_graph_not_model_ready"
    if not bool(row["cell_feature_complete"]):
        return "cell_features_incomplete"
    if int(row["n_points"]) != 8:
        return "incomplete_curve"
    return None


def freeze_primary(cfg: Dict[str, Any], logger) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Freeze the primary combo matrix index and mono index (v1)."""
    inp = cfg["inputs"]
    out = cfg["outputs"]
    excluded = set(cfg["filters"]["excluded_drugs"])

    mi = pd.read_parquet(resolve(cfg, inp["matrix_feature_index"]))
    meta = pd.read_parquet(
        resolve(cfg, inp["combo_matrix_metadata"]),
        columns=[
            "matrix_id",
            "pair_id",
            "complete_matrix",
            "n_points",
            "n_dose_a",
            "n_dose_b",
            "dose_a_um_json",
            "dose_b_um_json",
        ],
    )
    if meta["matrix_id"].duplicated().any():
        raise ValueError("duplicate matrix_id in M1 matrix metadata")
    mi = mi.merge(meta, on="matrix_id", how="left", validate="one_to_one")
    if mi["pair_id"].isna().any():
        raise ValueError("some primary matrices have no pair_id")
    lookup = _cell_lookup(pd.read_csv(resolve(cfg, inp["cell_mapping"])))
    mi["disease"] = mi["cell_line"].map(lambda c: lookup[c]["disease"])
    mi["lineage"] = mi["cell_line"].map(lambda c: lookup[c]["lineage"])
    mi["standard_cell_name"] = mi["cell_line"].map(lambda c: lookup[c]["standard_cell_name"])

    reasons = []
    for _, row in mi.iterrows():
        reasons.append(_exclusion_reason_matrix(row, excluded))
    mi["exclusion_reason"] = reasons
    mi["included"] = mi["exclusion_reason"].isna()

    mono = pd.read_parquet(resolve(cfg, inp["mono_feature_index"]))
    mono_meta = pd.read_parquet(
        resolve(cfg, inp["mono_curve_metadata"]),
        columns=["mono_curve_id", "n_points", "dose_um_json"],
    )
    if mono_meta["mono_curve_id"].duplicated().any():
        raise ValueError("duplicate mono_curve_id in M1 mono metadata")
    mono = mono.merge(mono_meta, on="mono_curve_id", how="left", validate="one_to_one")
    mono["disease"] = mono["cell_line"].map(lambda c: lookup[c]["disease"])
    mono["lineage"] = mono["cell_line"].map(lambda c: lookup[c]["lineage"])
    mono["standard_cell_name"] = mono["cell_line"].map(lambda c: lookup[c]["standard_cell_name"])
    mono_reasons = []
    for _, row in mono.iterrows():
        mono_reasons.append(_exclusion_reason_mono(row, excluded))
    mono["exclusion_reason"] = mono_reasons
    mono["included"] = mono["exclusion_reason"].isna()

    primary = mi[mi["included"]].reset_index(drop=True)
    primary_mono = mono[mono["included"]].reset_index(drop=True)

    exp = cfg["expected"]
    checks = {
        "matrices": int(len(primary)),
        "conditions": int(primary["condition_id"].nunique()),
        "cells": int(primary["cell_id"].nunique()),
        "drugs": int(len(set(primary["drug_a_id"]) | set(primary["drug_b_id"]))),
        "pairs": int(primary["pair_id"].nunique()),
        "mono_curves": int(len(primary_mono)),
        "points": int(primary["n_points"].sum()),
    }
    expected = {
        "matrices": int(exp["matrices"]),
        "conditions": int(exp["conditions"]),
        "cells": int(exp["cells"]),
        "drugs": int(exp["drugs"]),
        "pairs": int(exp["pairs"]),
        "mono_curves": int(exp["mono_curves"]),
        "points": int(exp["points"]),
    }
    for k, want in expected.items():
        if checks[k] != want:
            raise ValueError(f"primary {k} mismatch: got {checks[k]}, expected {want}")
    if (primary["n_points"] != 16).any():
        raise ValueError("primary contains a matrix without 16 points")

    matrix_out = resolve(cfg, out["matrix_index"])
    mono_out = resolve(cfg, out["mono_index"])
    excl_out = resolve(cfg, out["exclusions"])
    for p in (matrix_out, mono_out, excl_out):
        p.parent.mkdir(parents=True, exist_ok=True)

    keep_matrix = [
        "matrix_id",
        "condition_id",
        "pair_id",
        "batch_id",
        "cell_id",
        "cell_line",
        "standard_cell_name",
        "depmap_id",
        "disease",
        "lineage",
        "drug_a_id",
        "drug_a_name",
        "drug_b_id",
        "drug_b_name",
        "drug_a_graph_ok",
        "drug_b_graph_ok",
        "cell_feature_complete",
        "feature_complete",
        "n_points",
        "n_dose_a",
        "n_dose_b",
        "dose_a_um_json",
        "dose_b_um_json",
        "included",
        "exclusion_reason",
    ]
    primary[keep_matrix].to_parquet(matrix_out, index=False)
    keep_mono = [
        "mono_curve_id",
        "mono_condition_id",
        "batch_id",
        "cell_id",
        "cell_line",
        "standard_cell_name",
        "depmap_id",
        "disease",
        "lineage",
        "drug_id",
        "drug_name",
        "drug_graph_ok",
        "cell_feature_complete",
        "feature_complete",
        "n_points",
        "included",
        "exclusion_reason",
    ]
    primary_mono[keep_mono].to_parquet(mono_out, index=False)
    exclusions = pd.concat(
        [
            mi.loc[~mi["included"], ["matrix_id", "condition_id", "cell_id", "cell_line", "drug_a_name", "drug_b_name", "exclusion_reason"]].assign(dataset="combination"),
            mono.loc[~mono["included"], ["mono_curve_id", "mono_condition_id", "cell_id", "cell_line", "drug_name", "exclusion_reason"]].assign(dataset="monotherapy"),
        ],
        ignore_index=True,
    )
    exclusions.to_parquet(excl_out, index=False)

    summary = {
        "dataset_version": cfg["dataset_version"],
        "expected": expected,
        "observed": checks,
        "excluded_drugs": sorted(excluded),
        "exclusion_counts": exclusions.groupby(["dataset", "exclusion_reason"]).size().reset_index(name="n").to_dict(orient="records"),
        "outputs": {
            str(matrix_out.relative_to(cfg["_root"])): sha256_file(matrix_out),
            str(mono_out.relative_to(cfg["_root"])): sha256_file(mono_out),
            str(excl_out.relative_to(cfg["_root"])): sha256_file(excl_out),
        },
        "input_hashes": {
            p: sha256_file(resolve(cfg, p))
            for p in inp.values()
            if isinstance(p, str) and resolve(cfg, p).exists()
        },
        "generated_at": utc_now_iso(),
        "git_commit": git_commit(Path(cfg["_root"])),
    }
    return primary, primary_mono, summary


def build_condition_table(primary: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell line, unordered drug pair) condition present in v1."""
    rows = []
    for cid, g in primary.groupby("condition_id", sort=True):
        drug_ids = sorted(set(g["drug_a_id"]) | set(g["drug_b_id"]))
        if len(drug_ids) != 2:
            raise ValueError(f"condition {cid} does not have exactly 2 drugs")
        batches = sorted(set(str(b) for b in g["batch_id"]))
        rows.append(
            {
                "condition_id": cid,
                "pair_id": g["pair_id"].iloc[0],
                "cell_id": g["cell_id"].iloc[0],
                "cell_line": g["cell_line"].iloc[0],
                "standard_cell_name": g["standard_cell_name"].iloc[0],
                "depmap_id": g["depmap_id"].iloc[0],
                "disease": g["disease"].iloc[0],
                "lineage": g["lineage"].iloc[0],
                "drug_ids": "|".join(drug_ids),
                "n_drugs": len(drug_ids),
                "n_matrices": int(g["matrix_id"].nunique()),
                "n_points": int(g["n_points"].sum()),
                "n_batches": len(batches),
                "batch_ids": ",".join(batches),
            }
        )
    cond = pd.DataFrame(rows)
    if cond["condition_id"].duplicated().any():
        raise ValueError("duplicate condition_id in condition table")
    if not (cond["n_drugs"] == 2).all():
        raise ValueError("condition must contain exactly two distinct drugs")
    return cond


# ---------------------------------------------------------------------------
# Assignments
# ---------------------------------------------------------------------------
def _split_sizes(n: int, fractions: Sequence[float]) -> Tuple[int, int, int]:
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must be three numbers summing to 1")
    n_train = int(np.floor(fractions[0] * n))
    n_val = int(np.floor(fractions[1] * n))
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"split too small: n={n}")
    return n_train, n_val, n_test


def assign_by_key(keys: Iterable[str], seed: int, fractions: Sequence[float]) -> Dict[str, str]:
    """Deterministic group-level 70/15/15 split over sorted unique keys."""
    ids = np.array(sorted(set(map(str, keys))), dtype=object)
    n_train, n_val, _ = _split_sizes(len(ids), fractions)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(ids))
    mapping = {}
    for pos, idx in enumerate(order):
        mapping[str(ids[idx])] = "train" if pos < n_train else ("val" if pos < n_train + n_val else "test")
    return mapping


def cell_fold_table(conditions: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Balance cell lines across folds, spreading related diseases."""
    cells = (
        conditions[["cell_id", "cell_line", "standard_cell_name", "depmap_id", "disease", "lineage"]]
        .drop_duplicates("cell_id")
        .sort_values(["disease", "cell_id"], na_position="last")
        .reset_index(drop=True)
    )
    counts = np.zeros(n_folds, dtype=int)
    folds = []
    for _, row in cells.iterrows():
        fold = int(np.argmin(counts))
        counts[fold] += 1
        folds.append(fold)
    cells["fold"] = folds
    return cells


# ---------------------------------------------------------------------------
# Monotherapy and scalers
# ---------------------------------------------------------------------------
def _load_points_with_features(cfg: Dict[str, Any], primary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "matrix_id",
        "condition_id",
        "cell_id",
        "drug_a_id",
        "drug_b_id",
        "conc_a_um",
        "conc_b_um",
        "x_x0_raw",
        "row_idx",
        "col_idx",
    ]
    points = pd.read_parquet(resolve(cfg, cfg["inputs"]["combo_points"]), columns=cols)
    return points[points["matrix_id"].isin(set(primary["matrix_id"]))].reset_index(drop=True)


def _read_feature_matrices(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pathway = pd.read_parquet(resolve(cfg, cfg["inputs"]["pathway_features"]))
    if "DepMap_ID" not in pathway.columns:
        raise ValueError("pathway_features lacks DepMap_ID")
    expression = pd.read_parquet(resolve(cfg, cfg["inputs"]["cell_expression"]))
    if "depmap_id" not in expression.columns:
        raise ValueError("cell_expression lacks depmap_id")
    if expression["depmap_id"].duplicated().any():
        raise ValueError("duplicate depmap_id in cell_expression")
    return pathway, expression


def fit_csv_scalers(
    split_dir: Path,
    cond_train: pd.DataFrame,
    points: pd.DataFrame,
    mono_points: pd.DataFrame,
    mono_train_curve_ids: Sequence[str],
    pathway: pd.DataFrame,
    expression: pd.DataFrame,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Fit every transform on training entities only and persist them."""
    depmaps = sorted(set(str(d) for d in cond_train["depmap_id"].dropna()))
    pathway_cols = [c for c in pathway.columns if c != "DepMap_ID"]
    expr_meta = {"depmap_id", "oneil_cell_name", "standard_cell_name", "lineage", "disease"}
    expr_cols = [c for c in expression.columns if c not in expr_meta]
    pathway_index = pathway.set_index("DepMap_ID")
    expr_index = expression.set_index("depmap_id")
    missing_path = [d for d in depmaps if d not in pathway_index.index]
    missing_expr = [d for d in depmaps if d not in expr_index.index]
    if missing_path or missing_expr:
        raise ValueError(f"missing features for train cells: pathway={missing_path} expr={missing_expr}")

    pathway_scaler = fit_standardizer(
        pathway_index.loc[depmaps, pathway_cols].to_numpy(), pathway_cols, depmaps
    )
    expression_scaler = fit_standardizer(
        expr_index.loc[depmaps, expr_cols].to_numpy(), expr_cols, depmaps
    )

    conc_a = concentration_columns(points["conc_a_um"].to_numpy())
    conc_b = concentration_columns(points["conc_b_um"].to_numpy())
    dose_a = fit_dose_scaler(conc_a["log10_molar"], conc_a["zero_flag"], sorted(points["matrix_id"].unique()))
    dose_b = fit_dose_scaler(conc_b["log10_molar"], conc_b["zero_flag"], sorted(points["matrix_id"].unique()))

    trainable = mono_points[mono_points["mono_curve_id"].isin(set(mono_train_curve_ids))]
    mono_dose = None
    if len(trainable):
        mc = concentration_columns(trainable["conc_um"].to_numpy())
        mono_dose = fit_dose_scaler(
            mc["log10_molar"], mc["zero_flag"], sorted(trainable["mono_curve_id"].unique())
        )

    payload = {
        "meta": {
            **meta,
            "fit_cells": depmaps,
            "fit_conditions": int(len(cond_train)),
            "fit_dose_matrix_ids": int(points["matrix_id"].nunique()),
            "fit_mono_curves": int(trainable["mono_curve_id"].nunique()),
            "pathway_fingerprint": scaler_fingerprint(pathway_scaler),
            "expression_fingerprint": scaler_fingerprint(expression_scaler),
            "drug_a_dose_fingerprint": scaler_fingerprint(dose_a),
            "drug_b_dose_fingerprint": scaler_fingerprint(dose_b),
            "mono_dose_fingerprint": scaler_fingerprint(mono_dose) if mono_dose else None,
        },
        "pathway": pathway_scaler,
        "expression": expression_scaler,
        "drug_a_dose": dose_a,
        "drug_b_dose": dose_b,
        "mono_dose": mono_dose,
    }
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / "scalers.joblib"
    joblib.dump(payload, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "fit_cells": depmaps,
        "fit_conditions": int(len(cond_train)),
        "fit_mono_curves": int(trainable["mono_curve_id"].nunique()),
        "fingerprints": payload["meta"],
    }


# ---------------------------------------------------------------------------
# Split metadata
# ---------------------------------------------------------------------------
def _dose_stats(values: np.ndarray) -> Dict[str, Any]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    return {
        "min_um": float(v.min()),
        "median_um": float(np.median(v)),
        "max_um": float(v.max()),
        "min_log10_molar": float(np.log10(v.min() * 1e-6)),
        "max_log10_molar": float(np.log10(v.max() * 1e-6)),
    }


def split_statistics(
    cond: pd.DataFrame,
    primary: pd.DataFrame,
    points: pd.DataFrame,
    mono: pd.DataFrame,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        cs = cond[cond["split"] == split]
        mids = set(primary.loc[primary["condition_id"].isin(set(cs["condition_id"])), "matrix_id"])
        ps = points[points["matrix_id"].isin(mids)]
        ys = ps["x_x0_raw"].to_numpy(dtype=float)
        sub = primary[primary["condition_id"].isin(set(cs["condition_id"]))]
        n_drugs = len(set(sub["drug_a_id"]) | set(sub["drug_b_id"]))
        mono_s = mono[mono["cell_id"].isin(set(cs["cell_id"]))] if len(mono) else mono
        out[split] = {
            "n_conditions": int(len(cs)),
            "n_matrices": int(len(mids)),
            "n_points": int(len(ps)),
            "n_cells": int(cs["cell_id"].nunique()),
            "n_drug_pairs": int(cs["pair_id"].nunique()),
            "n_drugs": int(n_drugs),
            "mono_curves_cell_overlap": int(len(mono_s)),
            "label": {
                "mean": float(np.mean(ys)),
                "std": float(np.std(ys)),
                "min": float(np.min(ys)),
                "max": float(np.max(ys)),
                "q05": float(np.quantile(ys, 0.05)),
                "q95": float(np.quantile(ys, 0.95)),
                "n_gt1": int((ys > 1).sum()),
                "n_eq0": int((ys == 0).sum()),
            },
            "dose_a": _dose_stats(ps["conc_a_um"].to_numpy()),
            "dose_b": _dose_stats(ps["conc_b_um"].to_numpy()),
            "cancer_types": cs.groupby("disease").agg(n_conditions=("condition_id", "size")).reset_index().to_dict(orient="records"),
        }
    return out


def _condition_columns() -> List[str]:
    return [
        "condition_id",
        "pair_id",
        "cell_id",
        "cell_line",
        "standard_cell_name",
        "depmap_id",
        "disease",
        "lineage",
        "drug_ids",
        "n_drugs",
        "n_matrices",
        "n_points",
        "n_batches",
        "batch_ids",
        "split",
    ]


def write_split(
    split_dir: Path,
    cond: pd.DataFrame,
    mono: pd.DataFrame,
    scaler_info: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        cond[cond["split"] == split][_condition_columns()].to_parquet(
            split_dir / f"{split}_conditions.parquet", index=False
        )
    mono.to_parquet(split_dir / "mono_assignments.parquet", index=False)
    meta = dict(metadata)
    meta["assignment_sha256"] = assignment_fingerprint(cond, mono, metadata.get("fold_cells"))
    meta["scaler_sha256"] = scaler_info["sha256"]
    meta["generated_at"] = utc_now_iso()
    meta["content_sha256"] = canonical_hash(
        {k: v for k, v in meta.items() if k not in ("content_sha256",)}
    )
    write_json(meta, split_dir / "split_metadata.json")
    return meta


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------
def build_random_condition(
    cond: pd.DataFrame,
    seed: int,
    fractions: Sequence[float],
    split_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    mapping = assign_by_key(cond["condition_id"], seed, fractions)
    cond = cond.copy()
    cond["split"] = cond["condition_id"].map(mapping)
    if cond["split"].isna().any():
        raise ValueError("unassigned condition")
    return cond, {"group_key": "condition_id", "seed": int(seed), "fractions": list(fractions), "fold": None, "fold_cells": None}


def build_unseen_pair(
    cond: pd.DataFrame,
    seed: int,
    fractions: Sequence[float],
    split_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    mapping = assign_by_key(cond["pair_id"], seed, fractions)
    cond = cond.copy()
    cond["split"] = cond["pair_id"].map(mapping)
    if cond["split"].isna().any():
        raise ValueError("unassigned pair")
    return cond, {"group_key": "pair_id", "seed": int(seed), "fractions": list(fractions), "fold": None, "fold_cells": None}


def build_unseen_cell(
    cond: pd.DataFrame,
    cells: pd.DataFrame,
    fold: int,
    n_folds: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    val_fold = (fold + 1) % n_folds
    role = {}
    for r in cells.itertuples(index=False):
        if int(r.fold) == fold:
            role[str(r.cell_id)] = "test"
        elif int(r.fold) == val_fold:
            role[str(r.cell_id)] = "val"
        else:
            role[str(r.cell_id)] = "train"
    cond = cond.copy()
    cond["split"] = cond["cell_id"].map(role)
    if cond["split"].isna().any():
        raise ValueError("unassigned cell")
    fold_cells = {
        "test": sorted(cells.loc[cells["fold"] == fold, "cell_id"].tolist()),
        "val": sorted(cells.loc[cells["fold"] == val_fold, "cell_id"].tolist()),
        "train": sorted(cells.loc[~cells["fold"].isin([fold, val_fold]), "cell_id"].tolist()),
    }
    return cond, {
        "group_key": "cell_id",
        "seed": None,
        "fractions": None,
        "fold": int(fold),
        "n_folds": int(n_folds),
        "fold_cells": fold_cells,
    }


def generate(
    cfg_path: str,
    scenarios: Sequence[str],
    seeds: Optional[Sequence[int]] = None,
    folds: Optional[Sequence[int]] = None,
    logger=None,
) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    root = Path(cfg["_root"])
    if logger is None:
        import logging

        logger = logging.getLogger("dosesyn.splits")
    seeds = list(seeds or cfg["split"]["seeds"])
    n_folds = int(cfg["split"]["s3_folds"])
    folds = list(folds if folds is not None else range(n_folds))

    logger.info("freezing primary dataset %s", cfg["dataset_version"])
    primary, primary_mono, freeze_summary = freeze_primary(cfg, logger)
    write_json(freeze_summary, resolve(cfg, cfg["outputs"]["input_manifest"]))
    cond = build_condition_table(primary)
    points = _load_points_with_features(cfg, primary)
    mono_points = pd.read_parquet(
        resolve(cfg, cfg["inputs"]["mono_points"]), columns=["mono_curve_id", "cell_id", "conc_um", "x_x0_raw", "row_idx"]
    )
    mono_points = mono_points[
        mono_points["mono_curve_id"].isin(set(primary_mono["mono_curve_id"]))
    ].reset_index(drop=True)
    pathway, expression = _read_feature_matrices(cfg)
    cells = cell_fold_table(cond, n_folds)

    split_root = resolve(cfg, cfg["outputs"]["splits_dir"])
    records = []
    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario}")
        spec = SCENARIOS[scenario]
        if scenario == "s3_unseen_cell":
            jobs = [(f"fold_{f}", f) for f in folds]
        else:
            jobs = [(f"seed_{s}", s) for s in seeds]
        for label, value in jobs:
            logger.info("building %s/%s", scenario, label)
            if scenario == "s1_random_condition":
                c, extra = build_random_condition(cond, int(value), cfg["split"]["fractions"], split_root / scenario / label)
            elif scenario == "s2_unseen_pair":
                c, extra = build_unseen_pair(cond, int(value), cfg["split"]["fractions"], split_root / scenario / label)
            else:
                c, extra = build_unseen_cell(cond, cells, int(value), n_folds)

            mono = build_mono_assignments(primary_mono, c, spec["mono_policy"], seed=extra.get("seed"))
            split_dir = split_root / scenario / label
            train_cond = c[c["split"] == "train"]
            train_pts = points[points["condition_id"].isin(set(train_cond["condition_id"]))]
            scaler_meta = {
                "dataset_version": cfg["dataset_version"],
                "scenario": scenario,
                "split": label,
                "mono_policy": spec["mono_policy"],
                "group_key": spec["key"],
            }
            scaler_info = fit_csv_scalers(
                split_dir,
                train_cond,
                train_pts,
                mono_points,
                trainable_curve_ids(mono),
                pathway,
                expression,
                scaler_meta,
            )
            stats = split_statistics(c, primary, points, mono)
            metadata = {
                "scenario": scenario,
                "split_label": label,
                "dataset_version": cfg["dataset_version"],
                "seed": extra.get("seed"),
                "fold": extra.get("fold"),
                "n_folds": extra.get("n_folds"),
                "fractions": extra.get("fractions"),
                "group_key": spec["key"],
                "split_unit": spec["unit"],
                "mono_policy": spec["mono_policy"],
                "fold_cells": extra.get("fold_cells"),
                "counts": stats,
                "mono": {
                    "n_trainable_curves": int(mono["trainable"].sum()),
                    "n_held_out_curves": int((~mono["trainable"]).sum()),
                    "held_out_cells": sorted(mono.loc[~mono["trainable"], "cell_id"].unique().tolist()),
                },
                "scaler_fit": {
                    "n_cells": len(scaler_info["fit_cells"]),
                    "n_conditions": scaler_info["fit_conditions"],
                    "n_mono_curves": scaler_info["fit_mono_curves"],
                    "fingerprints": scaler_info["fingerprints"],
                },
                "frozen_primary": {
                    "n_matrices": int(len(primary)),
                    "n_conditions": int(len(cond)),
                    "n_mono_curves": int(len(primary_mono)),
                },
                "input_hashes": freeze_summary["input_hashes"],
                "git_commit": freeze_summary["git_commit"],
                "rule_notes": [
                    "All matrices/curves of one group are assigned together; no point-level split.",
                    "Batch repeats of one condition remain together.",
                    "Transforms are fitted on training cells/doses only.",
                    "Known monotherapy is auxiliary prior supervision, not an independent test set." if spec["mono_policy"] == "known_monotherapy" else "Validation/test cell monotherapy labels are held out from training and scaler fitting.",
                ],
            }
            meta = write_split(split_dir, c, mono, scaler_info, metadata)
            records.append(meta)
            logger.info(
                "%s/%s: train=%d val=%d test=%d conditions, mono_trainable=%d",
                scenario,
                label,
                meta["counts"]["train"]["n_conditions"],
                meta["counts"]["val"]["n_conditions"],
                meta["counts"]["test"]["n_conditions"],
                meta["mono"]["n_trainable_curves"],
            )

    manifest = {
        "version": "m3-v1",
        "dataset_version": cfg["dataset_version"],
        "generated_at": utc_now_iso(),
        "git_commit": freeze_summary["git_commit"],
        "freeze": freeze_summary,
        "scenarios": [{k: r.get(k) for k in ("scenario", "split_label", "seed", "fold", "assignment_sha256", "content_sha256", "scaler_sha256", "mono_policy")} for r in records],
        "config_hashes": {
            "dataset": sha256_file(Path(cfg_path)),
            "m3_config": sha256_file(root / "configs" / "m3.yaml") if (root / "configs" / "m3.yaml").exists() else None,
        },
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    write_json(manifest, resolve(cfg, cfg["outputs"]["manifest"]))
    logger.info("wrote %d split directories and manifest", len(records))
    return {"freeze": freeze_summary, "splits": records, "manifest": manifest}
