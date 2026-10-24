"""Concentration-unit normalisation, response arithmetic and replicate stats.

Design rules (see AGENTS.md / plan V1, M0/M1):
  * ``X/X0`` and ``mu/muMax`` are stored exactly as published -- never clipped.
  * Missing technical replicates stay missing (NaN); they are never imputed,
    and the mean is computed from the observed replicates only.
  * Concentrations are converted to µM exactly once, with the original value
    retained alongside the converted one.
  * A zero / negative concentration is flagged, not silently replaced: log10
    is undefined there and is emitted as NaN.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

LN4 = math.log(4.0)

#: multiplicative factor to convert the unit in `from_unit` into µM.
_UNIT_TO_UM: Dict[str, float] = {
    "m": 1.0e6,
    "mm": 1.0e3,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "micromolar": 1.0,
    "micromole/l": 1.0,
    "nm": 1.0e-3,
    "pm": 1.0e-6,
}


def unit_factor(from_unit: str, to_unit: str = "uM") -> float:
    """Return the multiplicative factor converting *from_unit* to *to_unit*.

    Only molar concentration units are supported; anything else raises so a
    unit mismatch can never pass silently.
    """
    src = from_unit.strip()
    dst = to_unit.strip()
    src_key, dst_key = src.lower(), dst.lower()
    if src_key not in _UNIT_TO_UM:
        raise ValueError(f"unsupported concentration unit: {from_unit!r}")
    if dst_key not in _UNIT_TO_UM:
        raise ValueError(f"unsupported target concentration unit: {to_unit!r}")
    return _UNIT_TO_UM[src_key] / _UNIT_TO_UM[dst_key]


def to_micromolar(values: pd.Series | np.ndarray, from_unit: str = "uM") -> np.ndarray:
    """Convert a numeric concentration vector to µM (no clipping)."""
    factor = unit_factor(from_unit, "uM")
    return np.asarray(values, dtype="float64") * factor


def safe_log10(values: pd.Series | np.ndarray) -> np.ndarray:
    """log10 that yields NaN (never -inf / an exception) for values <= 0."""
    arr = np.asarray(values, dtype="float64")
    out = np.full(arr.shape, np.nan, dtype="float64")
    positive = np.isfinite(arr) & (arr > 0)
    out[positive] = np.log10(arr[positive])
    return out


def recompute_response(mu: pd.Series | np.ndarray) -> np.ndarray:
    """X/X0 = exp(ln(4) * (mu/muMax - 1)) = 4 ** (mu/muMax - 1)."""
    arr = np.asarray(mu, dtype="float64")
    with np.errstate(over="ignore", invalid="ignore"):
        return np.power(4.0, arr - 1.0)


def response_residual(stored: pd.Series | np.ndarray, mu: pd.Series | np.ndarray) -> np.ndarray:
    """abs(stored - recomputed) in the linear X/X0 space."""
    return np.abs(np.asarray(stored, dtype="float64") - recompute_response(mu))


def concentration_flags(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Add ``*_is_nonpositive`` flags for the given concentration columns."""
    out = df
    for col in cols:
        out[f"{col}_is_nonpositive"] = ~(out[col] > 0)
    return out


def annotate_response(
    df: pd.DataFrame,
    response_col: str = "x_x0_raw",
    mu_col: str = "mu_raw",
    tolerance: float = 1e-5,
    upper_outlier: float = 2.0,
) -> pd.DataFrame:
    """Add derived, non-destructive response columns.

    Adds: recomputed response, formula residual + pass flag, >1 / >2 / ==0
    flags, negative flag, outlier flag.  The raw response column is untouched.
    """
    out = df
    out["x_x0_recalc"] = recompute_response(out[mu_col])
    out["response_formula_residual"] = response_residual(out[response_col], out[mu_col])
    # NaN residual means the stored or recomputed value was missing; treat as
    # "not checkable" rather than a failure.
    out["response_formula_ok"] = out["response_formula_residual"].isna() | (
        out["response_formula_residual"] <= tolerance
    )
    out["response_gt1"] = out[response_col] > 1.0
    out["response_gt2"] = out[response_col] > upper_outlier
    out["response_lt0"] = out[response_col] < 0.0
    out["response_eq0"] = out[response_col] == 0.0
    out["response_isna"] = out[response_col].isna()
    return out


def replicate_columns(df: pd.DataFrame) -> List[str]:
    """Return the technical-replicate viability columns in file order."""
    cols = [c for c in df.columns if c.startswith("viability_")]
    cols.sort(key=lambda c: int(c.rsplit("_", 1)[1]))
    return cols


def add_replicate_stats(
    df: pd.DataFrame,
    rep_cols: Sequence[str] | None = None,
    nominal: int | None = None,
) -> pd.DataFrame:
    """Summarise technical replicates *without* dropping or imputing any.

    Adds n_obs, n_missing, viability_mean, viability_std (sample, ddof=1),
    viability_spread (max - min), viability_min/max, and a boolean
    ``has_all_replicates`` when *nominal* is given.
    """
    out = df
    cols = list(rep_cols) if rep_cols is not None else replicate_columns(out)
    values = out[cols].to_numpy(dtype="float64")
    observed = np.isfinite(values)
    n_obs = observed.sum(axis=1)
    # All-missing rows legitimately produce NaN; suppress the empty-slice
    # RuntimeWarning that numpy would otherwise emit for them.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(values, axis=1)
        std = np.nanstd(values, axis=1, ddof=1)
        vmin = np.nanmin(values, axis=1)
        vmax = np.nanmax(values, axis=1)
    out["n_replicates_observed"] = n_obs.astype("int16")
    out["n_replicates_missing"] = (len(cols) - n_obs).astype("int16")
    out["viability_mean"] = mean
    out["viability_std"] = std
    out["viability_min"] = vmin
    out["viability_max"] = vmax
    out["viability_spread"] = vmax - vmin
    if nominal is not None:
        out["has_all_replicates"] = n_obs == nominal
    return out


def effective_concentration(
    df: pd.DataFrame, conc_col: str, response_col: str
) -> pd.Series:
    """Per-row concentration after unit conversion (identity guard for M1).

    M1 stores one concentration per row; this helper exists so downstream code
    never re-derives units ad hoc.
    """
    return df[conc_col].astype("float64")
