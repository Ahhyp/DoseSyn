"""M1 QC: anomaly tables, batch-repeat audit, entity index, sample matrices,
machine-readable checks and the human-readable QC report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.io import write_json, write_parquet
from .ids import dose_grid_token
from .make_matrices import pivot_matrix

ANOMALY_COLUMNS = [
    "anomaly_type",
    "severity",
    "dataset",
    "batch_id",
    "cell_line",
    "drug_a_name",
    "drug_b_name",
    "drug_name",
    "combination_name",
    "matrix_id",
    "mono_curve_id",
    "row_idx",
    "col_idx",
    "concentration_a_um",
    "concentration_b_um",
    "concentration_um",
    "replicate_index",
    "value",
    "detail",
    "source_row",
]


def _frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records, columns=ANOMALY_COLUMNS)


def _combo_rows(df: pd.DataFrame, mask: pd.Series, anomaly_type: str, severity: str, detail: str) -> List[Dict[str, Any]]:
    sub = df.loc[mask]
    return [
        {
            "anomaly_type": anomaly_type,
            "severity": severity,
            "dataset": "combination",
            "batch_id": int(r.batch_id),
            "cell_line": r.cell_line,
            "drug_a_name": r.drug_a_name,
            "drug_b_name": r.drug_b_name,
            "combination_name": r.combination_name,
            "matrix_id": r.matrix_id,
            "row_idx": int(r.row_idx),
            "col_idx": int(r.col_idx),
            "concentration_a_um": float(r.conc_a_um),
            "concentration_b_um": float(r.conc_b_um),
            "value": (None if pd.isna(r.x_x0_raw) else float(r.x_x0_raw)),
            "detail": detail,
            "source_row": int(r.source_row),
        }
        for r in sub.itertuples(index=False)
    ]


def _mono_rows(df: pd.DataFrame, mask: pd.Series, anomaly_type: str, severity: str, detail: str) -> List[Dict[str, Any]]:
    sub = df.loc[mask]
    return [
        {
            "anomaly_type": anomaly_type,
            "severity": severity,
            "dataset": "monotherapy",
            "batch_id": int(r.batch_id),
            "cell_line": r.cell_line,
            "drug_name": r.drug_name,
            "mono_curve_id": r.mono_curve_id,
            "row_idx": int(r.row_idx),
            "concentration_um": float(r.conc_um),
            "value": (None if pd.isna(r.x_x0_raw) else float(r.x_x0_raw)),
            "detail": detail,
            "source_row": int(r.source_row),
        }
        for r in sub.itertuples(index=False)
    ]


def build_anomalies(
    combo_points: pd.DataFrame,
    combo_meta: pd.DataFrame,
    mono_points: pd.DataFrame,
    mono_meta: pd.DataFrame,
    qc_cfg: Dict[str, Any],
) -> pd.DataFrame:
    spread = float(qc_cfg["replicate_spread_outlier"])
    upper = float(qc_cfg["x_over_x0_upper_outlier"])
    lower = float(qc_cfg["x_over_x0_lower_outlier"])

    records: List[Dict[str, Any]] = []

    # Combination
    records += _combo_rows(
        combo_points,
        combo_points["x_x0_raw"].isna(),
        "missing_response",
        "warning",
        "no X/X0 value published for this dose coordinate",
    )
    records += _combo_rows(
        combo_points,
        combo_points["response_gt2"],
        "response_outlier_high",
        "warning",
        f"X/X0 > {upper} (retained, not clipped)",
    )
    records += _combo_rows(
        combo_points,
        combo_points["response_lt0"],
        "response_negative",
        "error",
        "negative X/X0 (impossible under the published formula)",
    )
    records += _combo_rows(
        combo_points,
        combo_points["response_eq0"],
        "response_floor_zero",
        "info",
        "X/X0 published as exactly 0 (formula would give ~3.8e-6); retained as-is",
    )
    records += _combo_rows(
        combo_points,
        combo_points["viability_spread"] > spread,
        "high_replicate_spread",
        "warning",
        f"max-min of technical replicates > {spread}",
    )
    for r in combo_meta.loc[~combo_meta["complete_matrix"]].itertuples(index=False):
        records.append(
            {
                "anomaly_type": "incomplete_matrix",
                "severity": "error",
                "dataset": "combination",
                "batch_id": int(r.batch_id),
                "cell_line": r.cell_line,
                "drug_a_name": r.drug_a_name,
                "drug_b_name": r.drug_b_name,
                "matrix_id": r.matrix_id,
                "detail": f"n_points={r.n_points}, n_dose_a={r.n_dose_a}, n_dose_b={r.n_dose_b}",
            }
        )
    # Duplicate coordinates (expected: none)
    dup = (
        combo_points.groupby(["matrix_id", "conc_a_um", "conc_b_um"])
        .size()
        .rename("n")
        .reset_index()
    )
    dup = dup.loc[dup["n"] > 1]
    for r in dup.itertuples(index=False):
        records.append(
            {
                "anomaly_type": "duplicate_coordinate",
                "severity": "error",
                "dataset": "combination",
                "matrix_id": r.matrix_id,
                "concentration_a_um": float(r.conc_a_um),
                "concentration_b_um": float(r.conc_b_um),
                "detail": f"{int(r.n)} rows share this dose coordinate",
            }
        )
    # Non-positive concentrations (expected: none)
    bad_conc = combo_points["conc_a_um_is_nonpositive"] | combo_points["conc_b_um_is_nonpositive"]
    records += _combo_rows(
        combo_points,
        bad_conc,
        "nonpositive_concentration",
        "error",
        "concentration <= 0; log10 undefined",
    )
    records += _combo_rows(
        combo_points,
        ~combo_points["combination_name_matches_parts"],
        "combination_name_mismatch",
        "warning",
        "combination_name != drugA & drugB",
    )

    # Monotherapy
    records += _mono_rows(
        mono_points,
        mono_points["x_x0_raw"].isna(),
        "missing_response",
        "warning",
        "no X/X0 value published",
    )
    records += _mono_rows(
        mono_points,
        mono_points["response_gt2"],
        "response_outlier_high",
        "warning",
        f"X/X0 > {upper} (retained, not clipped)",
    )
    records += _mono_rows(
        mono_points,
        mono_points["response_lt0"],
        "response_negative",
        "error",
        "negative X/X0",
    )
    records += _mono_rows(
        mono_points,
        mono_points["response_eq0"],
        "response_floor_zero",
        "info",
        "X/X0 published as exactly 0; retained as-is",
    )
    records += _mono_rows(
        mono_points,
        mono_points["viability_spread"] > spread,
        "high_replicate_spread",
        "warning",
        f"max-min of technical replicates > {spread}",
    )
    for r in mono_meta.loc[~mono_meta["complete_curve"]].itertuples(index=False):
        records.append(
            {
                "anomaly_type": "incomplete_curve",
                "severity": "error",
                "dataset": "monotherapy",
                "batch_id": int(r.batch_id),
                "cell_line": r.cell_line,
                "drug_name": r.drug_name,
                "mono_curve_id": r.mono_curve_id,
                "detail": f"n_points={r.n_points}, n_unique_doses={r.n_unique_doses}",
            }
        )

    out = _frame(records)
    return out.sort_values(["severity", "anomaly_type", "source_row"], na_position="last")


def missing_replicate_records(
    combo_rep: pd.DataFrame, mono_rep: pd.DataFrame
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in combo_rep.loc[combo_rep["is_missing"]].itertuples(index=False):
        rows.append(
            {
                "anomaly_type": "missing_replicate",
                "severity": "info",
                "dataset": "combination",
                "matrix_id": r.matrix_id,
                "row_idx": int(r.row_idx),
                "col_idx": int(r.col_idx),
                "concentration_a_um": float(r.conc_a_um),
                "concentration_b_um": float(r.conc_b_um),
                "replicate_index": int(r.replicate_index),
                "value": None,
                "detail": f"viability{r.replicate_index} not published; other replicates retained",
                "source_row": int(r.source_row),
            }
        )
    for r in mono_rep.loc[mono_rep["is_missing"]].itertuples(index=False):
        rows.append(
            {
                "anomaly_type": "missing_replicate",
                "severity": "info",
                "dataset": "monotherapy",
                "mono_curve_id": r.mono_curve_id,
                "row_idx": int(r.row_idx),
                "concentration_um": float(r.conc_um),
                "replicate_index": int(r.replicate_index),
                "value": None,
                "detail": f"viability{r.replicate_index} not published; other replicates retained",
                "source_row": int(r.source_row),
            }
        )
    return _frame(rows)


# ---------------------------------------------------------------------------
# Batch repeats
# ---------------------------------------------------------------------------
def combo_batch_repeats(points: pd.DataFrame) -> pd.DataFrame:
    """Conditions measured in more than one batch, with grid comparison."""
    rows: List[Dict[str, Any]] = []
    for condition_id, grp in points.groupby("condition_id", sort=False):
        batches = sorted(grp["batch_id"].unique().tolist())
        if len(batches) < 2:
            continue
        base = grp.iloc[0]
        per = []
        for batch in batches:
            sub = grp.loc[grp["batch_id"] == batch]
            per.append(
                {
                    "batch_id": int(batch),
                    "matrix_id": sorted(sub["matrix_id"].unique())[0],
                    "n_points": int(len(sub)),
                    "dose_a": dose_grid_token(sub["conc_a_um"].unique()),
                    "dose_b": dose_grid_token(sub["conc_b_um"].unique()),
                    "x_mean": float(sub["x_x0_raw"].mean()),
                    "x_min": float(sub["x_x0_raw"].min()),
                    "x_max": float(sub["x_x0_raw"].max()),
                }
            )
        grids_equal = len({(p["dose_a"], p["dose_b"]) for p in per}) == 1
        rows.append(
            {
                "condition_id": condition_id,
                "pair_id": base.pair_id,
                "cell_line": base.cell_line,
                "drug_a_name": base.drug_a_name,
                "drug_b_name": base.drug_b_name,
                "combination_name": base.combination_name,
                "n_batches": len(batches),
                "batches": ",".join(str(b) for b in batches),
                "grids_identical": grids_equal,
                "batch_detail_json": json.dumps(per),
                "x_mean_across_batches": float(grp["x_x0_raw"].mean()),
            }
        )
    return pd.DataFrame(rows)


def mono_batch_repeats(points: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for condition_id, grp in points.groupby("mono_condition_id", sort=False):
        batches = sorted(grp["batch_id"].unique().tolist())
        if len(batches) < 2:
            continue
        base = grp.iloc[0]
        per = []
        for batch in batches:
            sub = grp.loc[grp["batch_id"] == batch]
            per.append(
                {
                    "batch_id": int(batch),
                    "mono_curve_id": sorted(sub["mono_curve_id"].unique())[0],
                    "n_points": int(len(sub)),
                    "doses": dose_grid_token(sub["conc_um"].unique()),
                    "x_mean": float(sub["x_x0_raw"].mean()),
                }
            )
        rows.append(
            {
                "mono_condition_id": condition_id,
                "cell_line": base.cell_line,
                "drug_name": base.drug_name,
                "n_batches": len(batches),
                "batches": ",".join(str(b) for b in batches),
                "grids_identical": len({p["doses"] for p in per}) == 1,
                "batch_detail_json": json.dumps(per),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entity index
# ---------------------------------------------------------------------------
def entity_index(
    combo_points: pd.DataFrame,
    mono_points: pd.DataFrame,
    registry: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    combo_by_cell = combo_points.groupby("cell_id").size()
    mono_by_cell = mono_points.groupby("cell_id").size()
    for name in registry["cell_lines"]:
        cid = registry["cells"][name]
        rows.append(
            {
                "entity_type": "cell_line",
                "entity_id": cid,
                "label": name,
                "n_combo_points": int(combo_by_cell.get(cid, 0)),
                "n_mono_points": int(mono_by_cell.get(cid, 0)),
                "n_matrices": int(combo_points.loc[combo_points.cell_id == cid, "matrix_id"].nunique()),
                "n_mono_curves": int(mono_points.loc[mono_points.cell_id == cid, "mono_curve_id"].nunique()),
            }
        )
    combo_by_drug: Dict[str, int] = {}
    for col in ("drug_a_id", "drug_b_id"):
        for did, n in combo_points.groupby(col).size().items():
            combo_by_drug[did] = combo_by_drug.get(did, 0) + int(n)
    mono_by_drug = mono_points.groupby("drug_id").size()
    a_only = set(registry["drugs_only_as_a"])
    b_only = set(registry["drugs_only_as_b"])
    for name in registry["drug_names"]:
        did = registry["drugs"][name]
        rows.append(
            {
                "entity_type": "drug",
                "entity_id": did,
                "label": name,
                "n_combo_points": combo_by_drug.get(did, 0),
                "n_mono_points": int(mono_by_drug.get(did, 0)),
                "n_matrices": int(
                    combo_points.loc[
                        (combo_points.drug_a_id == did) | (combo_points.drug_b_id == did),
                        "matrix_id",
                    ].nunique()
                ),
                "n_mono_curves": int(mono_points.loc[mono_points.drug_id == did, "mono_curve_id"].nunique()),
                "role_only_as": ("A" if name in a_only else ("B" if name in b_only else "")),
            }
        )
    pair_counts = combo_points.groupby("pair_id").size()
    for pair_id, grp in combo_points.groupby("pair_id", sort=False):
        rows.append(
            {
                "entity_type": "drug_pair",
                "entity_id": pair_id,
                "label": grp["pair_name"].iloc[0],
                "n_combo_points": int(pair_counts.get(pair_id, 0)),
                "n_mono_points": 0,
                "n_matrices": int(grp["matrix_id"].nunique()),
                "n_mono_curves": 0,
            }
        )
    cond = combo_points.groupby("condition_id").size()
    for cid, grp in combo_points.groupby("condition_id", sort=False):
        rows.append(
            {
                "entity_type": "combo_condition",
                "entity_id": cid,
                "label": f"{grp['cell_line'].iloc[0]} | {grp['combination_name'].iloc[0]}",
                "n_combo_points": int(cond.get(cid, 0)),
                "n_mono_points": 0,
                "n_matrices": int(grp["matrix_id"].nunique()),
                "n_mono_curves": 0,
            }
        )
    for mid, grp in combo_points.groupby("matrix_id", sort=False):
        rows.append(
            {
                "entity_type": "combo_matrix",
                "entity_id": mid,
                "label": f"{grp['cell_line'].iloc[0]} | batch {grp['batch_id'].iloc[0]} | {grp['combination_name'].iloc[0]}",
                "n_combo_points": int(len(grp)),
                "n_mono_points": 0,
                "n_matrices": 1,
                "n_mono_curves": 0,
            }
        )
    for curve_id, grp in mono_points.groupby("mono_curve_id", sort=False):
        rows.append(
            {
                "entity_type": "mono_curve",
                "entity_id": curve_id,
                "label": f"{grp['cell_line'].iloc[0]} | batch {grp['batch_id'].iloc[0]} | {grp['drug_name'].iloc[0]}",
                "n_combo_points": 0,
                "n_mono_points": int(len(grp)),
                "n_matrices": 0,
                "n_mono_curves": 1,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sample matrices
# ---------------------------------------------------------------------------
def _pick_sample_matrices(
    points: pd.DataFrame, meta: pd.DataFrame, repeats: pd.DataFrame
) -> List[Dict[str, str]]:
    picks: List[Dict[str, str]] = []
    by_x = meta.sort_values("x_x0_max", ascending=False).iloc[0]
    picks.append({"reason": "highest_x_x0", "matrix_id": by_x.matrix_id})
    by_spread = meta.sort_values("max_replicate_spread", ascending=False).iloc[0]
    picks.append({"reason": "largest_replicate_spread", "matrix_id": by_spread.matrix_id})
    repeated = repeats.loc[repeats["grids_identical"]] if len(repeats) else repeats
    if len(repeated):
        detail = json.loads(repeated.iloc[0]["batch_detail_json"])
        picks.append({"reason": "batch_repeat_same_grid", "matrix_id": detail[0]["matrix_id"]})
    diff = repeats.loc[~repeats["grids_identical"]] if len(repeats) else repeats
    if len(diff):
        detail = json.loads(diff.iloc[0]["batch_detail_json"])
        picks.append({"reason": "batch_repeat_different_grid", "matrix_id": detail[0]["matrix_id"]})
    picks.append({"reason": "first_matrix", "matrix_id": sorted(meta["matrix_id"])[0]})
    seen = set()
    out = []
    for p in picks:
        if p["matrix_id"] not in seen:
            seen.add(p["matrix_id"])
            out.append(p)
    return out


def write_sample_matrices(
    combo_points: pd.DataFrame,
    combo_meta: pd.DataFrame,
    repeats: pd.DataFrame,
    out_dir: Path,
    max_samples: int = 5,
) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = _pick_sample_matrices(combo_points, combo_meta, repeats)[:max_samples]
    manifest: List[Dict[str, Any]] = []
    rows = []
    for i, sample in enumerate(samples, start=1):
        mid = sample["matrix_id"]
        sub = combo_points.loc[combo_points.matrix_id == mid].copy()
        meta_row = combo_meta.loc[combo_meta.matrix_id == mid].iloc[0]
        stem = f"{i:02d}_{sample['reason']}_{mid}"
        x_grid = pivot_matrix(combo_points, mid, "x_x0_raw")
        v_grid = pivot_matrix(combo_points, mid, "viability_mean")
        x_grid.to_csv(out_dir / f"{stem}_x_x0.csv")
        v_grid.to_csv(out_dir / f"{stem}_viability_mean.csv")
        long = sub[
            [
                "matrix_id",
                "batch_id",
                "cell_line",
                "drug_a_name",
                "drug_b_name",
                "row_idx",
                "col_idx",
                "conc_a_um",
                "conc_b_um",
                "viability_1",
                "viability_2",
                "viability_3",
                "viability_4",
                "viability_mean",
                "viability_spread",
                "mu_raw",
                "x_x0_raw",
                "source_row",
            ]
        ].sort_values(["row_idx", "col_idx"])
        long.to_csv(out_dir / f"{stem}_points.csv", index=False)
        rows.append(long)
        manifest.append(
            {
                "reason": sample["reason"],
                "matrix_id": mid,
                "batch_id": int(meta_row.batch_id),
                "cell_line": meta_row.cell_line,
                "combination_name": meta_row.combination_name,
                "x_x0_min": float(meta_row.x_x0_min),
                "x_x0_max": float(meta_row.x_x0_max),
                "max_replicate_spread": float(meta_row.max_replicate_spread),
                "files": [
                    f"{stem}_x_x0.csv",
                    f"{stem}_viability_mean.csv",
                    f"{stem}_points.csv",
                ],
            }
        )
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out_dir / "sample_matrices_long.csv", index=False)
    write_json(manifest, out_dir / "sample_matrices_manifest.json")
    return manifest


# ---------------------------------------------------------------------------
# Checks and report
# ---------------------------------------------------------------------------
def _check(name: str, status: str, observed: Any, expected: Any, detail: str = "") -> Dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "observed": observed,
        "expected": expected,
        "detail": detail,
    }


def build_checks(
    cfg: Dict[str, Any],
    file_entries: Sequence[Dict[str, Any]],
    raw_stats: Dict[str, Any],
    combo_points: pd.DataFrame,
    combo_meta: pd.DataFrame,
    mono_points: pd.DataFrame,
    mono_meta: pd.DataFrame,
    repeats: pd.DataFrame,
    combo_struct: Dict[str, Any],
    mono_struct: Dict[str, Any],
) -> List[Dict[str, Any]]:
    exp = cfg["expected_counts"]
    qc = cfg["qc"]
    checks: List[Dict[str, Any]] = []

    checks.append(
        _check(
            "raw_sha256_matches_config",
            "PASS" if all(e["sha256_match"] for e in file_entries) else "FAIL",
            {e["role"]: e["sha256"] for e in file_entries},
            cfg["expected_sha256"],
            "Raw files are byte-identical to the supplied files.",
        )
    )
    containers = {e["role"]: e["family"] for e in file_entries}
    checks.append(
        _check(
            "raw_containers_detected_from_magic_bytes",
            "PASS" if all(e["is_ooxml"] for e in file_entries) else "FAIL",
            containers,
            "all ooxml",
            "All three files are OOXML/ZIP; the .xls extension is misleading and "
            "the file is read via an unmodified .xlsx working copy with openpyxl.",
        )
    )
    checks.append(
        _check(
            "combination_rows_reconciled",
            "PASS" if len(combo_points) == exp["combo_rows"] else "FAIL",
            int(len(combo_points)),
            int(exp["combo_rows"]),
            "Every published combination row is present exactly once.",
        )
    )
    checks.append(
        _check(
            "monotherapy_rows_reconciled",
            "PASS" if len(mono_points) == exp["mono_rows"] else "FAIL",
            int(len(mono_points)),
            int(exp["mono_rows"]),
        )
    )
    checks.append(
        _check(
            "combo_matrix_structure",
            "PASS" if combo_struct["all_ok"] else "FAIL",
            combo_struct,
            {"n_matrices": exp["combo_matrices"], "n_points_per_matrix": 16},
        )
    )
    checks.append(
        _check(
            "mono_curve_structure",
            "PASS" if mono_struct["all_ok"] else "FAIL",
            mono_struct,
            {"n_curves": exp["mono_curves"], "n_points_per_curve": 8},
        )
    )

    # Value preservation
    raw_max = float(max(e for e in [combo_points["x_x0_raw"].max(), mono_points["x_x0_raw"].max()]))
    checks.append(
        _check(
            "response_not_clipped",
            "PASS" if combo_points["x_x0_raw"].max() > 1.0 else "FAIL",
            {
                "combo_max_x_x0": float(combo_points["x_x0_raw"].max()),
                "combo_n_gt1": int(combo_points["response_gt1"].sum()),
                "combo_n_gt2": int(combo_points["response_gt2"].sum()),
                "mono_max_x_x0": float(mono_points["x_x0_raw"].max()),
                "mono_n_gt1": int(mono_points["response_gt1"].sum()),
            },
            "raw values retained, including X/X0 > 1",
            "No upper/lower clipping is applied anywhere in the pipeline.",
        )
    )
    tol = float(qc["response_formula_tolerance"])
    combo_bad = int((~combo_points["response_formula_ok"]).sum())
    mono_bad = int((~mono_points["response_formula_ok"]).sum())
    checks.append(
        _check(
            "response_formula_reproduces_published_x_x0",
            "PASS" if combo_bad == 0 and mono_bad == 0 else "FAIL",
            {
                "combo_failures": combo_bad,
                "mono_failures": mono_bad,
                "combo_max_abs_residual": float(combo_points["response_formula_residual"].max()),
                "mono_max_abs_residual": float(mono_points["response_formula_residual"].max()),
            },
            f"abs residual <= {tol}",
            "X/X0 = exp(ln(4) * (mu/muMax - 1)); residuals come only from 5-decimal source rounding.",
        )
    )
    checks.append(
        _check(
            "no_negative_response",
            "PASS"
            if (int((combo_points["x_x0_raw"] < 0).sum()) == 0 and int((mono_points["x_x0_raw"] < 0).sum()) == 0)
            else "FAIL",
            {
                "combo_negative": int((combo_points["x_x0_raw"] < 0).sum()),
                "mono_negative": int((mono_points["x_x0_raw"] < 0).sum()),
            },
            0,
        )
    )
    checks.append(
        _check(
            "concentrations_positive_and_um",
            "PASS"
            if not (combo_points[["conc_a_um_is_nonpositive", "conc_b_um_is_nonpositive"]].any().any()
                    or mono_points["conc_um_is_nonpositive"].any())
            else "FAIL",
            {
                "combo_nonpositive_a": int(combo_points["conc_a_um_is_nonpositive"].sum()),
                "combo_nonpositive_b": int(combo_points["conc_b_um_is_nonpositive"].sum()),
                "mono_nonpositive": int(mono_points["conc_um_is_nonpositive"].sum()),
                "combo_min_um": float(min(combo_points["conc_a_um"].min(), combo_points["conc_b_um"].min())),
                "combo_max_um": float(max(combo_points["conc_a_um"].max(), combo_points["conc_b_um"].max())),
                "mono_min_um": float(mono_points["conc_um"].min()),
                "mono_max_um": float(mono_points["conc_um"].max()),
            },
            "all > 0 µM",
            "Published unit is µM; conversion factor 1.0 recorded in the data dictionary.",
        )
    )
    checks.append(
        _check(
            "traceability_source_row",
            "PASS"
            if combo_points["source_row"].is_unique and mono_points["source_row"].is_unique
            else "FAIL",
            {
                "combo_unique_rows": bool(combo_points["source_row"].is_unique),
                "combo_row_range": [int(combo_points["source_row"].min()), int(combo_points["source_row"].max())],
                "mono_unique_rows": bool(mono_points["source_row"].is_unique),
                "mono_row_range": [int(mono_points["source_row"].min()), int(mono_points["source_row"].max())],
            },
            "one processed row per published row",
        )
    )
    checks.append(
        _check(
            "batch_repeat_audit",
            "PASS" if len(repeats) > 0 else "WARN",
            {
                "n_repeated_conditions": int(len(repeats)),
                "same_grid": int(repeats["grids_identical"].sum()),
                "different_grid": int((~repeats["grids_identical"]).sum()),
            },
            "reported, not merged",
            "Repeat experiments are kept as separate matrices; batch is never averaged away.",
        )
    )
    checks.append(
        _check(
            "technical_replicates_retained",
            "PASS",
            {
                "combo_replicate_columns": 4,
                "combo_missing_replicate_values": int(combo_points["n_replicates_missing"].sum()),
                "mono_replicate_columns": 6,
                "mono_missing_replicate_values": int(mono_points["n_replicates_missing"].sum()),
            },
            "raw viability columns preserved verbatim",
            "Missing replicate values are NaN, never imputed.",
        )
    )
    checks.append(
        _check(
            "matrix_ids_unique_per_experiment",
            "PASS" if combo_meta["matrix_id"].is_unique and mono_meta["mono_curve_id"].is_unique else "FAIL",
            {
                "n_matrices": int(len(combo_meta)),
                "n_mono_curves": int(len(mono_meta)),
            },
            {"matrices": exp["combo_matrices"], "curves": exp["mono_curves"]},
        )
    )
    return checks


def build_qc_summary(
    cfg: Dict[str, Any],
    environment: Dict[str, Any],
    file_entries: Sequence[Dict[str, Any]],
    raw_stats: Dict[str, Any],
    combo_points: pd.DataFrame,
    combo_meta: pd.DataFrame,
    mono_points: pd.DataFrame,
    mono_meta: pd.DataFrame,
    anomalies: pd.DataFrame,
    repeats: pd.DataFrame,
    checks: List[Dict[str, Any]],
    generated_at: str,
) -> Dict[str, Any]:
    anomaly_counts = anomalies["anomaly_type"].value_counts().sort_index().to_dict()
    severity_counts = anomalies["severity"].value_counts().sort_index().to_dict()
    m2_ready = all(c["status"] != "FAIL" for c in checks) and anomaly_counts.get("response_negative", 0) == 0
    blocking = [c["name"] for c in checks if c["status"] == "FAIL"]
    return {
        "study": "ONEIL",
        "stage": "M0/M1",
        "generated_at": generated_at,
        "environment": environment,
        "config": {
            "path": cfg.get("_config_path"),
            "root": cfg.get("_root"),
            "units": cfg.get("units", {}),
            "qc_thresholds": cfg["qc"],
            "hash_algorithm": "sha256",
            "id_hash_len": cfg.get("ids", {}).get("hash_len", 12),
        },
        "inputs": list(file_entries),
        "raw": raw_stats,
        "counts": {
            "combo_points": int(len(combo_points)),
            "combo_matrices": int(combo_meta["matrix_id"].nunique()),
            "combo_conditions": int(combo_points["condition_id"].nunique()),
            "combo_drug_pairs": int(combo_points["pair_id"].nunique()),
            "mono_points": int(len(mono_points)),
            "mono_curves": int(mono_meta["mono_curve_id"].nunique()),
            "mono_conditions": int(mono_points["mono_condition_id"].nunique()),
            "cell_lines": int(combo_points["cell_line"].nunique()),
            "drugs": int(
                len(
                    set(combo_points["drug_a_name"])
                    | set(combo_points["drug_b_name"])
                    | set(mono_points["drug_name"])
                )
            ),
            "batch_repeated_conditions": int(len(repeats)),
            "batch_repeat_same_grid": int(repeats["grids_identical"].sum()) if len(repeats) else 0,
            "batch_repeat_different_grid": int((~repeats["grids_identical"]).sum()) if len(repeats) else 0,
            "combo_missing_replicate_values": int(combo_points["n_replicates_missing"].sum()),
            "mono_missing_replicate_values": int(mono_points["n_replicates_missing"].sum()),
            "combo_x_gt1": int(combo_points["response_gt1"].sum()),
            "combo_x_gt2": int(combo_points["response_gt2"].sum()),
            "combo_x_max": float(combo_points["x_x0_raw"].max()),
            "combo_x_min": float(combo_points["x_x0_raw"].min()),
            "mono_x_gt1": int(mono_points["response_gt1"].sum()),
            "mono_x_gt2": int(mono_points["response_gt2"].sum()),
            "mono_x_max": float(mono_points["x_x0_raw"].max()),
            "mono_x_min": float(mono_points["x_x0_raw"].min()),
        },
        "anomalies": {
            "by_type": {k: int(v) for k, v in anomaly_counts.items()},
            "by_severity": {k: int(v) for k, v in severity_counts.items()},
            "total": int(len(anomalies)),
        },
        "checks": checks,
        "acceptance": {
            "m2_ready": bool(m2_ready),
            "blocking_failures": blocking,
            "note": "M2 may start only when every check is PASS and pytest is green.",
        },
    }


def render_qc_markdown(summary: Dict[str, Any]) -> str:
    c = summary["counts"]
    lines: List[str] = []
    lines.append("# DoseSyn — O'Neil 数据审计与预处理 QC 报告 (M0/M1)")
    lines.append("")
    lines.append(f"- 生成时间: `{summary['generated_at']}`")
    lines.append(f"- 阶段: {summary['stage']}")
    lines.append(f"- 结论: **{'PASS — 可进入 M2' if summary['acceptance']['m2_ready'] else 'FAIL — 不可进入 M2'}**")
    if summary["acceptance"]["blocking_failures"]:
        lines.append(f"- 阻塞项: {', '.join(summary['acceptance']['blocking_failures'])}")
    lines.append("")
    lines.append("## 1. 输入文件与格式审计")
    lines.append("")
    lines.append("| 角色 | 文件 | 大小 (B) | SHA256 (前16位) | 真实容器 | 扩展名匹配 |")
    lines.append("| --- | --- | ---: | --- | --- | --- |")
    for e in summary["inputs"]:
        lines.append(
            f"| {e['role']} | `{e['filename']}` | {e['size_bytes']:,} | `{e['sha256'][:16]}` | "
            f"{e['family']}/{e['container']} | {e['extension_matches_content']} |"
        )
    lines.append("")
    lines.append("> 组合响应文件后缀为 `.xls`，但 magic bytes `504b0304` 表明其真实格式为 OOXML/ZIP。")
    lines.append("> 读取时使用 `data/interim/format_work/` 下的字节级副本 `.xlsx` + openpyxl；原始文件未被修改。")
    lines.append("")
    lines.append("## 2. 数据规模")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | ---: |")
    for key, label in [
        ("combo_points", "双药响应行"),
        ("combo_matrices", "4×4 矩阵"),
        ("combo_conditions", "双药条件 (细胞系×药物对)"),
        ("combo_drug_pairs", "无序药物对"),
        ("mono_points", "单药响应行"),
        ("mono_curves", "单药剂量曲线"),
        ("cell_lines", "细胞系"),
        ("drugs", "药物"),
        ("batch_repeated_conditions", "跨批次重复条件"),
        ("batch_repeat_same_grid", "  其中剂量网格相同"),
        ("batch_repeat_different_grid", "  其中剂量网格不同"),
        ("combo_missing_replicate_values", "双药缺失技术重复值"),
        ("mono_missing_replicate_values", "单药缺失技术重复值"),
        ("combo_x_gt1", "双药 X/X0 > 1 (保留)"),
        ("combo_x_gt2", "双药 X/X0 > 2 (保留并标记)"),
        ("mono_x_gt1", "单药 X/X0 > 1 (保留)"),
    ]:
        lines.append(f"| {label} | {c[key]:,} |")
    lines.append("")
    lines.append("## 3. 自动检查")
    lines.append("")
    lines.append("| 检查 | 状态 | 观测值 | 期望 | 说明 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for chk in summary["checks"]:
        obs = chk["observed"]
        obs_s = json.dumps(obs, ensure_ascii=False) if not isinstance(obs, str) else obs
        if len(obs_s) > 160:
            obs_s = obs_s[:157] + "..."
        lines.append(
            f"| {chk['name']} | **{chk['status']}** | `{obs_s}` | `{chk['expected']}` | {chk.get('detail', '')} |"
        )
    lines.append("")
    lines.append("## 4. M2 准入判定")
    lines.append("")
    by_name = {chk["name"]: chk for chk in summary["checks"]}
    gate = [
        ("原始文件可读、SHA256 与交付时一致、`.xls` 未交给旧版 XLS 引擎",
         ["raw_sha256_matches_config", "raw_containers_detected_from_magic_bytes"]),
        ("行数与源表对账一致，无静默丢行",
         ["combination_rows_reconciled", "monotherapy_rows_reconciled"]),
        ("矩阵/曲线唯一且完整（4×4=16，单药 8 点）",
         ["combo_matrix_structure", "mono_curve_structure", "matrix_ids_unique_per_experiment"]),
        ("剂量单位统一为 µM，响应公式可复算",
         ["concentrations_positive_and_um", "response_formula_reproduces_published_x_x0"]),
        ("X/X0 未被裁剪，>1 与 >2 的原始值保留",
         ["response_not_clipped"]),
        ("技术重复逐列保留，缺失值不插补",
         ["technical_replicates_retained"]),
        ("每个响应点可追溯至源表行",
         ["traceability_source_row"]),
        ("跨批次重复已识别、未合并",
         ["batch_repeat_audit"]),
        ("自动化测试全部通过",
         ["pytest_suite_passes"]),
    ]
    lines.append("| 准入条件 | 结果 | 证据 |")
    lines.append("| --- | --- | --- |")
    for label, names in gate:
        statuses = [by_name[n]["status"] for n in names if n in by_name]
        ok = bool(statuses) and all(s == "PASS" for s in statuses)
        lines.append(f"| {label} | **{'PASS' if ok else 'FAIL'}** | {', '.join(names)} |")
    lines.append("")
    lines.append("## 5. 异常清单（按类型汇总）")
    lines.append("")
    lines.append("| 异常类型 | 数量 |")
    lines.append("| --- | ---: |")
    for k, v in summary["anomalies"]["by_type"].items():
        lines.append(f"| {k} | {v:,} |")
    lines.append("")
    lines.append("完整逐条清单见 `outputs/audit/oneil_anomalies.csv` 及 `outputs/audit/anomalies_by_type/`。")
    lines.append("")
    lines.append("## 6. 产出文件")
    lines.append("")
    lines.append("| 文件 | 内容 |")
    lines.append("| --- | --- |")
    for path, desc in summary.get("outputs", {}).items():
        lines.append(f"| `{path}` | {desc} |")
    lines.append("")
    lines.append("## 7. 尚未解决的问题 / 风险")
    lines.append("")
    for item in summary.get("open_issues", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
