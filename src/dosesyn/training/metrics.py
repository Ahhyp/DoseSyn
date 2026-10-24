"""Metrics for DoseSyn M4 matrix-level dose-response prediction.

All metrics are computed on the raw, unclipped ``X/X0`` scale.  A prediction
item is one 4x4 matrix; metrics are never computed after pooling away the
matrix structure.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


def _as_2d_arrays(y_true: np.ndarray, y_pred: np.ndarray):
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")
    if true.ndim == 1:
        true = true.reshape(1, -1)
        pred = pred.reshape(1, -1)
    if true.ndim != 2:
        raise ValueError("expected [n_matrices, n_points] arrays")
    return true, pred


def matrix_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-matrix RMSE over the 16 dose points."""
    true, pred = _as_2d_arrays(y_true, y_pred)
    return np.sqrt(np.mean((true - pred) ** 2, axis=1))


def matrix_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true, pred = _as_2d_arrays(y_true, y_pred)
    return np.mean(np.abs(true - pred), axis=1)


def pooled_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if true.size < 2 or np.std(true) == 0 or np.std(pred) == 0:
        return None
    return float(np.corrcoef(true, pred)[0, 1])


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    if ss_tot == 0.0:
        return None
    ss_res = float(np.sum((true - pred) ** 2))
    return float(1.0 - ss_res / ss_tot)


def per_matrix_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true, pred = _as_2d_arrays(y_true, y_pred)
    out = np.full(true.shape[0], np.nan, dtype=np.float64)
    for i in range(true.shape[0]):
        if np.std(true[i]) > 0 and np.std(pred[i]) > 0:
            out[i] = np.corrcoef(true[i], pred[i])[0, 1]
    return out


def condition_macro_rmse(
    matrix_rmse_values: Sequence[float], condition_ids: Sequence[str]
) -> Optional[float]:
    """Mean of per-condition mean matrix RMSE.

    Each condition contributes equally regardless of how many repeated batches
    it has, which exposes batch-weighting shifts.
    """
    df = pd.DataFrame(
        {"condition_id": list(condition_ids), "matrix_rmse": list(matrix_rmse_values)}
    )
    if df.empty:
        return None
    return float(df.groupby("condition_id")["matrix_rmse"].mean().mean())


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    matrix_ids: Sequence[str],
    condition_ids: Sequence[str],
) -> Dict[str, Any]:
    """Compute the locked M4 metric suite.

    Parameters
    ----------
    y_true, y_pred:
        ``[N, 16]`` or ``[N, 4, 4]`` raw ``X/X0`` values.
    matrix_ids, condition_ids:
        One entry per matrix, in the same order.
    """
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.ndim == 3:
        true = true.reshape(true.shape[0], -1)
        pred = pred.reshape(pred.shape[0], -1)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")
    if len(matrix_ids) != true.shape[0] or len(condition_ids) != true.shape[0]:
        raise ValueError("matrix_ids/condition_ids length mismatch")

    rmse_per_matrix = matrix_rmse(true, pred)
    mae_per_matrix = matrix_mae(true, pred)
    pm_p = per_matrix_pearson(true, pred)
    finite_pm = pm_p[np.isfinite(pm_p)]

    return {
        "n_matrices": int(true.shape[0]),
        "n_points": int(true.size),
        "matrix_macro_rmse": float(np.mean(rmse_per_matrix)),
        "matrix_macro_mae": float(np.mean(mae_per_matrix)),
        "condition_macro_rmse": condition_macro_rmse(rmse_per_matrix, condition_ids),
        "point_rmse": float(np.sqrt(np.mean((true - pred) ** 2))),
        "point_mae": float(np.mean(np.abs(true - pred))),
        "pooled_pearson": pooled_pearson(true, pred),
        "r2": r2_score(true, pred),
        "matrix_pearson_mean": float(np.mean(finite_pm)) if finite_pm.size else None,
        "n_matrix_pearson_undefined": int(pm_p.size - finite_pm.size),
        "y_true_min": float(true.min()),
        "y_true_max": float(true.max()),
        "y_true_mean": float(true.mean()),
        "y_pred_min": float(pred.min()),
        "y_pred_max": float(pred.max()),
        "y_pred_mean": float(pred.mean()),
        "n_y_true_gt1": int((true > 1.0).sum()),
        "n_y_true_eq0": int((true == 0.0).sum()),
        "n_y_pred_negative": int((pred < 0.0).sum()),
    }


def build_prediction_frame(
    *,
    model: str,
    scenario: str,
    seed_or_fold: str,
    split: str,
    matrix_ids: Sequence[str],
    condition_ids: Sequence[str],
    pair_ids: Sequence[str],
    cell_ids: Sequence[str],
    batch_ids: Sequence[int],
    drug_a_ids: Sequence[str],
    drug_b_ids: Sequence[str],
    dose_a_um: np.ndarray,
    dose_b_um: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cell_lines: Optional[Sequence[str]] = None,
    drug_a_names: Optional[Sequence[str]] = None,
    drug_b_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build the long per-dose-point prediction table (16 rows per matrix)."""
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.ndim == 3:
        n, r, c = true.shape
        if (r, c) != (4, 4):
            raise ValueError(f"expected 4x4, got {true.shape}")
        true = true.reshape(n, 16)
        pred = pred.reshape(n, 16)
    n = true.shape[0]
    if true.shape[1] != 16:
        raise ValueError("each matrix must have exactly 16 points")
    rows = []
    for i in range(n):
        for k in range(16):
            ri, ci = divmod(k, 4)
            rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "seed_or_fold": str(seed_or_fold),
                    "split": split,
                    "matrix_id": matrix_ids[i],
                    "condition_id": condition_ids[i],
                    "pair_id": pair_ids[i],
                    "cell_id": cell_ids[i],
                    "cell_line": None if cell_lines is None else cell_lines[i],
                    "batch_id": int(batch_ids[i]),
                    "drug_a_id": drug_a_ids[i],
                    "drug_b_id": drug_b_ids[i],
                    "drug_a_name": None if drug_a_names is None else drug_a_names[i],
                    "drug_b_name": None if drug_b_names is None else drug_b_names[i],
                    "row_idx": ri,
                    "col_idx": ci,
                    "dose_a_um": float(dose_a_um[i][ri]),
                    "dose_b_um": float(dose_b_um[i][ci]),
                    "y_true": float(true[i, k]),
                    "y_pred": float(pred[i, k]),
                    "residual": float(pred[i, k] - true[i, k]),
                }
            )
    return pd.DataFrame(rows)


def json_safe(value: Any) -> Any:
    """Convert NaN/inf/numpy scalars into strict-JSON-safe values."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value
