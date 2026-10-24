"""Train-fold-only feature transformations.

Every statistic in this module is fitted on rows supplied by the caller, which
must be the *training* cells/doses only. Validation and test data are only ever
transformed. No function here reads a full dataset by itself.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd

SCALER_VERSION = "m3-train-only-v1"


def _as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def fit_standardizer(
    matrix: np.ndarray,
    columns: Sequence[str],
    fit_ids: Sequence[str],
    *,
    impute: bool = True,
) -> Dict[str, Any]:
    """Fit mean/std/median on the provided (training) rows only."""
    arr = _as_2d(matrix)
    if arr.shape[1] != len(columns):
        raise ValueError(f"matrix has {arr.shape[1]} columns, expected {len(columns)}")
    if arr.shape[0] == 0:
        raise ValueError("cannot fit a scaler on zero rows")

    nan_mask = np.isnan(arr)
    n_missing = int(nan_mask.sum())
    if impute and n_missing:
        median = np.nanmedian(arr, axis=0)
        median = np.where(np.isnan(median), 0.0, median)
        filled = np.where(nan_mask, median, arr)
    elif n_missing:
        raise ValueError(f"{n_missing} missing values and impute=False")
    else:
        median = np.zeros(arr.shape[1], dtype=np.float64)
        filled = arr

    mean = filled.mean(axis=0)
    std = filled.std(axis=0, ddof=0)
    zero_var = std < 1e-12
    scale = np.where(zero_var, 1.0, std)
    return {
        "version": SCALER_VERSION,
        "type": "standard",
        "columns": list(columns),
        "fit_ids": list(fit_ids),
        "n_fit_rows": int(arr.shape[0]),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "median": median.tolist(),
        "zero_variance_columns": [columns[i] for i in np.flatnonzero(zero_var)],
        "n_missing_values": n_missing,
    }


def transform_standardizer(matrix: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    """Apply a fitted scaler; never re-estimates statistics."""
    arr = _as_2d(matrix).copy()
    median = np.asarray(scaler["median"], dtype=np.float64)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        arr = np.where(nan_mask, median, arr)
    return (arr - mean) / scale


def fit_dose_scaler(
    log10_molar: np.ndarray,
    zero_flag: np.ndarray,
    fit_matrix_ids: Sequence[str],
) -> Dict[str, Any]:
    """Fit a dose scaler on training *non-zero* doses (log10 molar)."""
    vals = np.asarray(log10_molar, dtype=np.float64)
    zeros = np.asarray(zero_flag, dtype=bool)
    usable = vals[~zeros & np.isfinite(vals)]
    if usable.size == 0:
        raise ValueError("no positive training doses to fit")
    return {
        "version": SCALER_VERSION,
        "type": "standard",
        "unit": "log10_molar",
        "mean": float(usable.mean()),
        "scale": float(usable.std(ddof=0) or 1.0),
        "fit_matrix_ids": list(fit_matrix_ids),
        "n_fit_doses": int(usable.size),
        "min": float(usable.min()),
        "max": float(usable.max()),
        "zero_rule": "zero dose is excluded from the log fit and encoded by zero_flag",
    }


def transform_dose(log10_molar: np.ndarray, zero_flag: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    vals = np.asarray(log10_molar, dtype=np.float64)
    zeros = np.asarray(zero_flag, dtype=bool)
    out = np.full(vals.shape, np.nan, dtype=np.float64)
    pos = ~zeros & np.isfinite(vals)
    out[pos] = (vals[pos] - float(scaler["mean"])) / float(scaler["scale"])
    return out


def concentration_columns(dose_um: np.ndarray) -> Dict[str, np.ndarray]:
    """Convert µM to molar and log10(molar) with an explicit zero flag."""
    um = np.asarray(dose_um, dtype=np.float64)
    positive = um > 0
    molar = np.where(positive, um * 1e-6, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log10_molar = np.where(positive, np.log10(np.where(positive, um, 1.0)) - 6.0, np.nan)
    return {
        "dose_um": um,
        "dose_molar": molar,
        "log10_molar": log10_molar,
        "zero_flag": (~positive).astype(np.int8),
        "nonpositive_flag": (~positive).astype(np.int8),
    }


def scaler_fingerprint(scaler: Dict[str, Any]) -> str:
    """Stable SHA256 over fitted entities and statistics (ignores nothing)."""
    payload = {
        "version": scaler.get("version"),
        "fit_ids": scaler.get("fit_ids") or scaler.get("fit_matrix_ids"),
        "mean": scaler.get("mean"),
        "scale": scaler.get("scale"),
        "median": scaler.get("median"),
        "columns": scaler.get("columns"),
        "n_fit_rows": scaler.get("n_fit_rows"),
        "n_fit_doses": scaler.get("n_fit_doses"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def assert_train_only(
    scaler: Dict[str, Any],
    expected_fit_ids: Iterable[str],
    *,
    key: str = "fit_ids",
) -> None:
    """Guard used by tests and the verifier: fit entities must equal train entities."""
    got = sorted(map(str, scaler.get(key, [])))
    want = sorted(map(str, expected_fit_ids))
    if got != want:
        raise AssertionError(
            f"scaler fit entities != training entities: {len(got)} vs {len(want)}"
        )
