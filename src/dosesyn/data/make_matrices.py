"""Reconstruct 4x4 combination matrices and 8-point monotherapy curves.

The published files store one row per (cell line, drug pair, dose coordinate)
with 4 (combination) or 6 (single agent) technical replicates.  This module
groups rows back into experimental units:

* combination: ``batch_id x cell_line x drugA x drugB`` -> `matrix_id`,
  with ``row_idx`` = rank of drug A dose and ``col_idx`` = rank of drug B dose;
* monotherapy: ``batch_id x cell_line x drug`` -> `mono_curve_id`, with
  ``row_idx`` = rank of the single-agent dose.

No row is dropped: incomplete / duplicated units are kept and flagged so the
QC layer can report them.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .ids import (
    combo_condition_id,
    dose_grid_token,
    matrix_id,
    mono_condition_id,
    mono_curve_id,
    unordered_pair_id,
)

COMBO_KEYS = ["batch_id", "cell_id", "drug_a_id", "drug_b_id"]
MONO_KEYS = ["batch_id", "cell_id", "drug_id"]


# ---------------------------------------------------------------------------
# Pair / canonical orientation
# ---------------------------------------------------------------------------
def add_pair_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add pair_id, canonical low/high drug and dose columns."""
    out = df
    a = out["drug_a_id"].to_numpy()
    b = out["drug_b_id"].to_numpy()
    a_is_low = a < b
    low_id = np.where(a_is_low, a, b)
    high_id = np.where(a_is_low, b, a)

    unique_pairs = pd.DataFrame({"low": low_id, "high": high_id}).drop_duplicates()
    pair_map = {
        (lo, hi): unordered_pair_id(lo, hi)
        for lo, hi in zip(unique_pairs["low"], unique_pairs["high"])
    }
    out["pair_id"] = [pair_map[(lo, hi)][0] for lo, hi in zip(low_id, high_id)]
    out["pair_drug_low_id"] = low_id
    out["pair_drug_high_id"] = high_id
    out["is_drug_a_low_id"] = a_is_low

    out["dose_low_um"] = np.where(a_is_low, out["conc_a_um"], out["conc_b_um"])
    out["dose_high_um"] = np.where(a_is_low, out["conc_b_um"], out["conc_a_um"])
    out["log10_dose_low"] = np.where(
        a_is_low, out["log10_conc_a"], out["log10_conc_b"]
    )
    out["log10_dose_high"] = np.where(
        a_is_low, out["log10_conc_b"], out["log10_conc_a"]
    )
    name_map = {}
    for lo, hi, ia, na, nb in zip(
        low_id,
        high_id,
        a,
        out["drug_a_name"].to_numpy(),
        out["drug_b_name"].to_numpy(),
    ):
        if (lo, hi) not in name_map:
            first, second = (na, nb) if ia == lo else (nb, na)
            name_map[(lo, hi)] = f"{first} & {second}"
    out["pair_name"] = [name_map[(lo, hi)] for lo, hi in zip(low_id, high_id)]
    return out


def _join_grids(
    df: pd.DataFrame, keys: Sequence[str], value_col: str, name: str
) -> pd.DataFrame:
    """Attach a canonical sorted-grid token for each experimental unit."""
    grids = (
        df[list(keys) + [value_col]]
        .drop_duplicates()
        .groupby(list(keys), sort=False)[value_col]
        .apply(lambda s: dose_grid_token(s.to_numpy()))
        .rename(name)
        .reset_index()
    )
    return df.merge(grids, on=list(keys), how="left", validate="many_to_one")


# ---------------------------------------------------------------------------
# Combination matrices
# ---------------------------------------------------------------------------
def build_combo_matrices(
    points: pd.DataFrame, spread_threshold: float = 1.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (points with matrix/row/col ids, per-matrix metadata)."""
    out = points.copy()
    out = add_pair_columns(out)
    out = _join_grids(out, COMBO_KEYS, "conc_a_um", "grid_a_token")
    out = _join_grids(out, COMBO_KEYS, "conc_b_um", "grid_b_token")

    units = out[COMBO_KEYS + ["grid_a_token", "grid_b_token"]].drop_duplicates()
    unit_ids = [
        matrix_id(b, c, a, bb, ga.split("|"), gb.split("|"))
        for b, c, a, bb, ga, gb in zip(
            units["batch_id"],
            units["cell_id"],
            units["drug_a_id"],
            units["drug_b_id"],
            units["grid_a_token"],
            units["grid_b_token"],
        )
    ]
    units = units.assign(matrix_id=unit_ids)
    out = out.merge(
        units,
        on=COMBO_KEYS + ["grid_a_token", "grid_b_token"],
        how="left",
        validate="many_to_one",
    )

    conditions = out[["pair_id", "cell_id"]].drop_duplicates()
    conditions["condition_id"] = [
        combo_condition_id(p, c)
        for p, c in zip(conditions["pair_id"], conditions["cell_id"])
    ]
    out = out.merge(conditions, on=["pair_id", "cell_id"], how="left", validate="many_to_one")

    # Dense ranks are sort-based, so row/col 0 is always the lowest dose and
    # the layout is reproducible even if source rows are shuffled.
    out["row_idx"] = (
        out.groupby("matrix_id")["conc_a_um"].rank(method="dense").astype("int16") - 1
    )
    out["col_idx"] = (
        out.groupby("matrix_id")["conc_b_um"].rank(method="dense").astype("int16") - 1
    )

    meta = _combo_metadata(out, spread_threshold=spread_threshold)
    return out, meta


def _json_grid(values: Sequence[float]) -> str:
    return json.dumps([float(v) for v in sorted(values)])


def _combo_metadata(df: pd.DataFrame, spread_threshold: float = 1.0) -> pd.DataFrame:
    g = df.groupby("matrix_id", sort=False)
    meta = g.agg(
        batch_id=("batch_id", "first"),
        cell_line=("cell_line", "first"),
        cell_id=("cell_id", "first"),
        pair_id=("pair_id", "first"),
        drug_a_id=("drug_a_id", "first"),
        drug_a_name=("drug_a_name", "first"),
        drug_b_id=("drug_b_id", "first"),
        drug_b_name=("drug_b_name", "first"),
        combination_name=("combination_name", "first"),
        condition_id=("condition_id", "first"),
        n_points=("source_row", "size"),
        n_dose_a=("conc_a_um", "nunique"),
        n_dose_b=("conc_b_um", "nunique"),
        n_unique_coords=("source_row", "size"),  # overwritten below
        source_row_min=("source_row", "min"),
        source_row_max=("source_row", "max"),
        n_missing_response=("x_x0_raw", lambda s: int(s.isna().sum())),
        n_missing_replicates=("n_replicates_missing", "sum"),
        n_points_reduced_reps=("has_all_replicates", lambda s: int((~s).sum())),
        x_x0_min=("x_x0_raw", "min"),
        x_x0_max=("x_x0_raw", "max"),
        x_x0_mean=("x_x0_raw", "mean"),
        n_x_gt1=("response_gt1", "sum"),
        n_x_gt2=("response_gt2", "sum"),
        n_x_lt0=("response_lt0", "sum"),
        max_replicate_spread=("viability_spread", "max"),
        n_high_replicate_spread=(
            "viability_spread",
            lambda s: int((s > spread_threshold).sum()),
        ),
        all_mu_present=("mu_raw", lambda s: bool(s.notna().all())),
    ).reset_index()

    # unique coordinates per matrix
    coords = (
        df[["matrix_id", "conc_a_um", "conc_b_um"]]
        .drop_duplicates()
        .groupby("matrix_id")
        .size()
        .rename("n_unique_coords_calc")
    )
    meta = meta.drop(columns=["n_unique_coords"]).merge(
        coords, left_on="matrix_id", right_index=True, how="left"
    )
    meta = meta.rename(columns={"n_unique_coords_calc": "n_unique_coords"})

    grids = (
        df.groupby("matrix_id", sort=False)
        .agg(
            dose_a_um_json=("conc_a_um", lambda s: _json_grid(s.unique())),
            dose_b_um_json=("conc_b_um", lambda s: _json_grid(s.unique())),
            dose_low_um_json=("dose_low_um", lambda s: _json_grid(s.unique())),
            dose_high_um_json=("dose_high_um", lambda s: _json_grid(s.unique())),
        )
        .reset_index()
    )
    meta = meta.merge(grids, on="matrix_id", how="left")
    meta["source_rows_contiguous"] = (
        meta["source_row_max"] - meta["source_row_min"] + 1 == meta["n_points"]
    )
    meta["complete_matrix"] = (
        (meta["n_points"] == 16)
        & (meta["n_dose_a"] == 4)
        & (meta["n_dose_b"] == 4)
        & (meta["n_unique_coords"] == 16)
    )
    return meta


def pivot_matrix(
    points: pd.DataFrame, mid: str, value_col: str = "x_x0_raw"
) -> pd.DataFrame:
    """Return one 4x4 matrix with dose labels as index/columns."""
    sub = points.loc[points["matrix_id"] == mid]
    grid = sub.pivot_table(
        index="conc_a_um",
        columns="conc_b_um",
        values=value_col,
        aggfunc="mean",
        dropna=False,
    )
    grid.index.name = "drugA_conc_uM"
    grid.columns.name = "drugB_conc_uM"
    return grid.sort_index(axis=0).sort_index(axis=1)


# ---------------------------------------------------------------------------
# Monotherapy curves
# ---------------------------------------------------------------------------
def build_mono_curves(
    points: pd.DataFrame, spread_threshold: float = 1.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = points.copy()
    out = _join_grids(out, MONO_KEYS, "conc_um", "grid_token")
    units = out[MONO_KEYS + ["grid_token"]].drop_duplicates()
    units = units.assign(
        mono_curve_id=[
            mono_curve_id(b, c, d, tok.split("|"))
            for b, c, d, tok in zip(
                units["batch_id"], units["cell_id"], units["drug_id"], units["grid_token"]
            )
        ]
    )
    out = out.merge(
        units, on=MONO_KEYS + ["grid_token"], how="left", validate="many_to_one"
    )

    conditions = out[["drug_id", "cell_id"]].drop_duplicates()
    conditions["mono_condition_id"] = [
        mono_condition_id(d, c)
        for d, c in zip(conditions["drug_id"], conditions["cell_id"])
    ]
    out = out.merge(
        conditions, on=["drug_id", "cell_id"], how="left", validate="many_to_one"
    )
    out["row_idx"] = (
        out.groupby("mono_curve_id")["conc_um"].rank(method="dense").astype("int16") - 1
    )
    meta = _mono_metadata(out, spread_threshold=spread_threshold)
    return out, meta


def _mono_metadata(
    df: pd.DataFrame, spread_threshold: float = 1.0
) -> pd.DataFrame:
    meta = df.groupby("mono_curve_id", sort=False).agg(
        batch_id=("batch_id", "first"),
        cell_line=("cell_line", "first"),
        cell_id=("cell_id", "first"),
        drug_id=("drug_id", "first"),
        drug_name=("drug_name", "first"),
        mono_condition_id=("mono_condition_id", "first"),
        n_points=("source_row", "size"),
        n_unique_doses=("conc_um", "nunique"),
        source_row_min=("source_row", "min"),
        source_row_max=("source_row", "max"),
        n_missing_response=("x_x0_raw", lambda s: int(s.isna().sum())),
        total_missing_replicates=("n_replicates_missing", "sum"),
        n_points_reduced_reps=("has_all_replicates", lambda s: int((~s).sum())),
        x_x0_min=("x_x0_raw", "min"),
        x_x0_max=("x_x0_raw", "max"),
        x_x0_mean=("x_x0_raw", "mean"),
        n_x_gt1=("response_gt1", "sum"),
        n_x_gt2=("response_gt2", "sum"),
        n_x_lt0=("response_lt0", "sum"),
        max_replicate_spread=("viability_spread", "max"),
        n_high_replicate_spread=(
            "viability_spread",
            lambda s: int((s > spread_threshold).sum()),
        ),
    ).reset_index()
    grids = (
        df.groupby("mono_curve_id", sort=False)
        .agg(dose_um_json=("conc_um", lambda s: _json_grid(s.unique())))
        .reset_index()
    )
    meta = meta.merge(grids, on="mono_curve_id", how="left")
    meta["source_rows_contiguous"] = (
        meta["source_row_max"] - meta["source_row_min"] + 1 == meta["n_points"]
    )
    meta["complete_curve"] = (meta["n_points"] == 8) & (meta["n_unique_doses"] == 8)
    return meta


# ---------------------------------------------------------------------------
# Replicate long tables
# ---------------------------------------------------------------------------
def combo_replicates(points: pd.DataFrame, nominal: int = 4) -> pd.DataFrame:
    id_cols = [
        "matrix_id",
        "condition_id",
        "source_row",
        "row_idx",
        "col_idx",
        "conc_a_um",
        "conc_b_um",
        "x_x0_raw",
    ]
    rep_cols = [f"viability_{i}" for i in range(1, nominal + 1)]
    long = points[id_cols + rep_cols].melt(
        id_vars=id_cols,
        value_vars=rep_cols,
        var_name="replicate_label",
        value_name="viability",
    )
    long["replicate_index"] = (
        long["replicate_label"].str.extract(r"(\d+)$")[0].astype("int16")
    )
    long["is_missing"] = long["viability"].isna()
    return long.sort_values(["matrix_id", "row_idx", "col_idx", "replicate_index"])


def mono_replicates(points: pd.DataFrame, nominal: int = 6) -> pd.DataFrame:
    id_cols = [
        "mono_curve_id",
        "mono_condition_id",
        "source_row",
        "row_idx",
        "conc_um",
        "x_x0_raw",
    ]
    rep_cols = [f"viability_{i}" for i in range(1, nominal + 1)]
    long = points[id_cols + rep_cols].melt(
        id_vars=id_cols,
        value_vars=rep_cols,
        var_name="replicate_label",
        value_name="viability",
    )
    long["replicate_index"] = (
        long["replicate_label"].str.extract(r"(\d+)$")[0].astype("int16")
    )
    long["is_missing"] = long["viability"].isna()
    return long.sort_values(["mono_curve_id", "row_idx", "replicate_index"])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_combo_structure(points: pd.DataFrame) -> Dict[str, Any]:
    g = points.groupby("matrix_id", sort=False)
    sizes = g.size()
    dose_a = g["conc_a_um"].nunique()
    dose_b = g["conc_b_um"].nunique()
    coords = points.groupby(["matrix_id", "conc_a_um", "conc_b_um"]).size()
    duplicated_coords = int((coords > 1).sum())
    bad = {
        "n_points_not_16": int((sizes != 16).sum()),
        "n_dose_a_not_4": int((dose_a != 4).sum()),
        "n_dose_b_not_4": int((dose_b != 4).sum()),
        "duplicated_coordinates": duplicated_coords,
        "n_matrices": int(points["matrix_id"].nunique()),
        "n_conditions": int(points["condition_id"].nunique()),
        "n_pairs": int(points["pair_id"].nunique()),
    }
    bad["all_ok"] = (
        bad["n_points_not_16"] == 0
        and bad["n_dose_a_not_4"] == 0
        and bad["n_dose_b_not_4"] == 0
        and bad["duplicated_coordinates"] == 0
    )
    return bad


def validate_mono_structure(points: pd.DataFrame) -> Dict[str, Any]:
    g = points.groupby("mono_curve_id", sort=False)
    sizes = g.size()
    doses = g["conc_um"].nunique()
    out = {
        "n_curves": int(points["mono_curve_id"].nunique()),
        "n_conditions": int(points["mono_condition_id"].nunique()),
        "n_points_not_8": int((sizes != 8).sum()),
        "n_doses_not_8": int((doses != 8).sum()),
        "duplicated_doses": int((points.groupby(["mono_curve_id", "conc_um"]).size() > 1).sum()),
    }
    out["all_ok"] = (
        out["n_points_not_8"] == 0
        and out["n_doses_not_8"] == 0
        and out["duplicated_doses"] == 0
    )
    return out
