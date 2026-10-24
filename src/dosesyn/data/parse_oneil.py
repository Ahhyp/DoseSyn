"""Read the O'Neil workbooks and emit canonical, traceable data frames.

The public entry points are :func:`load_raw_frames` (raw parquet cache written
by ``00_audit_raw.py``) and :func:`standardize_combo` / :func:`standardize_mono`
which add derived columns but never modify, clip, impute or drop a source value.

Every output row keeps ``source_row`` (the 1-based Excel row, header = row 1) so
any processed value can be traced back to the published workbook.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..utils.io import detect_container, sha256_file
from . import normalize
from .ids import cell_ids, drug_ids
from .inspect import check_columns, expected_columns


# ---------------------------------------------------------------------------
# Workbook resolution
# ---------------------------------------------------------------------------
def resolve_workbook(cfg: Dict[str, Any], role: str) -> Tuple[Path, str]:
    """Return (readable_path, engine) for a raw file.

    The published combination file has a ``.xls`` suffix but is really an
    OOXML/ZIP workbook.  openpyxl refuses it by suffix and xlrd cannot parse
    it, so an unmodified byte-for-byte copy with a ``.xlsx`` suffix is made in
    the interim work directory.  The raw file itself is only ever opened in
    binary read mode.
    """
    raw = Path(cfg["files"][f"{role}_path"])
    info = detect_container(raw)
    suffix = raw.suffix.lower()

    if info["is_ooxml"] and suffix not in (".xlsx", ".xlsm"):
        work_dir = Path(cfg["paths"]["work_dir"])
        work_dir.mkdir(parents=True, exist_ok=True)
        copy = work_dir / f"{raw.stem}_working_copy.xlsx"
        if not copy.exists() or sha256_file(copy) != sha256_file(raw):
            shutil.copyfile(raw, copy)
        return copy, "openpyxl"

    if info["is_ooxml"]:
        return raw, "openpyxl"
    if info["is_legacy_xls"]:
        return raw, "xlrd"
    raise ValueError(f"unsupported input container for {raw}: {info['note']}")


def read_raw_from_workbook(cfg: Dict[str, Any], role: str) -> pd.DataFrame:
    path, engine = resolve_workbook(cfg, role)
    sheet = cfg["raw_columns"][role]["sheet"]
    return pd.read_excel(path, sheet_name=sheet, engine=engine)


def raw_cache_path(cfg: Dict[str, Any], role: str) -> Path:
    return Path(cfg["paths"]["interim_dir"]) / "raw_oneil" / f"{role}.parquet"


def load_raw_frames(cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Load all three raw frames from the parquet cache."""
    cache = raw_cache_path(cfg, "combo").parent
    frames: Dict[str, pd.DataFrame] = {}
    for role in ("mono", "combo"):
        p = cache / f"{role}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"raw cache {p} missing; run scripts/00_audit_raw.py first"
            )
        frames[role] = pd.read_parquet(p)
    frames["legend"] = pd.read_parquet(cache / "legend.parquet")
    return frames


# ---------------------------------------------------------------------------
# Text / dtype hygiene
# ---------------------------------------------------------------------------
def clean_text(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    if out.isna().any():
        raise ValueError("empty cell in a required text column")
    return out.astype("object")


def _viability_columns(cfg: Dict[str, Any], role: str) -> List[str]:
    return list(cfg["raw_columns"][role]["viability"])


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------
def _base_trace_columns(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    out["source_index"] = raw.index.astype("int64")
    out["source_row"] = out["source_index"] + 2  # Excel header occupies row 1
    return out


def standardize_combo(
    raw: pd.DataFrame,
    cfg: Dict[str, Any],
    cell_map: Dict[str, str],
    drug_map: Dict[str, str],
    logger=None,
) -> pd.DataFrame:
    spec = cfg["raw_columns"]["combo"]
    check = check_columns(raw, cfg, "combo")
    if not check["ok"]:
        raise ValueError(
            f"combination sheet columns changed: missing={check['missing']} extra={check['extra']}"
        )
    qc = cfg["qc"]
    out = _base_trace_columns(raw)
    out["batch_id"] = raw[spec["batch"]].astype("int64")
    out["cell_line"] = clean_text(raw[spec["cell"]])
    out["drug_a_name"] = clean_text(raw[spec["drug_a"]])
    out["drug_b_name"] = clean_text(raw[spec["drug_b"]])
    out["combination_name"] = clean_text(raw[spec["combination"]])
    out["cell_id"] = out["cell_line"].map(cell_map)
    out["drug_a_id"] = out["drug_a_name"].map(drug_map)
    out["drug_b_id"] = out["drug_b_name"].map(drug_map)
    if out[["cell_id", "drug_a_id", "drug_b_id"]].isna().any().any():
        missing = sorted(
            set(out.loc[out["cell_id"].isna(), "cell_line"])
            | set(out.loc[out["drug_a_id"].isna(), "drug_a_name"])
            | set(out.loc[out["drug_b_id"].isna(), "drug_b_name"])
        )
        raise ValueError(f"entity registry is missing: {missing}")

    raw_unit = cfg.get("units", {}).get("raw_concentration", "uM")
    out["conc_a_source"] = raw[spec["dose_a_um"]].astype("float64")
    out["conc_b_source"] = raw[spec["dose_b_um"]].astype("float64")
    out["conc_a_um"] = normalize.to_micromolar(out["conc_a_source"], raw_unit)
    out["conc_b_um"] = normalize.to_micromolar(out["conc_b_source"], raw_unit)
    out["log10_conc_a"] = normalize.safe_log10(out["conc_a_um"])
    out["log10_conc_b"] = normalize.safe_log10(out["conc_b_um"])
    normalize.concentration_flags(out, ["conc_a_um", "conc_b_um"])

    rep_cols = _viability_columns(cfg, "combo")
    for i, col in enumerate(rep_cols, start=1):
        out[f"viability_{i}"] = raw[col].astype("float64")
    out = normalize.add_replicate_stats(
        out,
        [f"viability_{i}" for i in range(1, len(rep_cols) + 1)],
        nominal=int(qc["nominal_combo_replicates"]),
    )
    assert len(rep_cols) == int(qc["nominal_combo_replicates"])

    out["mu_raw"] = raw[spec["mu"]].astype("float64")
    out["x_x0_raw"] = raw[spec["response"]].astype("float64")
    out = normalize.annotate_response(
        out,
        tolerance=float(qc["response_formula_tolerance"]),
        upper_outlier=float(qc["x_over_x0_upper_outlier"]),
    )
    out["combination_name_matches_parts"] = out["combination_name"] == (
        out["drug_a_name"] + " & " + out["drug_b_name"]
    )
    if logger is not None:
        logger.info(
            "standardised combination sheet: %d rows, %d missing responses, "
            "%d X/X0>1 preserved, %d X/X0>2",
            len(out),
            int(out["x_x0_raw"].isna().sum()),
            int(out["response_gt1"].sum()),
            int(out["response_gt2"].sum()),
        )
    return out


def standardize_mono(
    raw: pd.DataFrame,
    cfg: Dict[str, Any],
    cell_map: Dict[str, str],
    drug_map: Dict[str, str],
    logger=None,
) -> pd.DataFrame:
    spec = cfg["raw_columns"]["mono"]
    check = check_columns(raw, cfg, "mono")
    if not check["ok"]:
        raise ValueError(
            f"single-agent sheet columns changed: missing={check['missing']} extra={check['extra']}"
        )
    qc = cfg["qc"]
    out = _base_trace_columns(raw)
    out["batch_id"] = raw[spec["batch"]].astype("int64")
    out["cell_line"] = clean_text(raw[spec["cell"]])
    out["drug_name"] = clean_text(raw[spec["drug"]])
    out["cell_id"] = out["cell_line"].map(cell_map)
    out["drug_id"] = out["drug_name"].map(drug_map)
    if out[["cell_id", "drug_id"]].isna().any().any():
        raise ValueError("entity registry is missing a cell line or drug in the single-agent sheet")

    raw_unit = cfg.get("units", {}).get("raw_concentration", "uM")
    out["conc_source"] = raw[spec["dose_um"]].astype("float64")
    out["conc_um"] = normalize.to_micromolar(out["conc_source"], raw_unit)
    out["log10_conc"] = normalize.safe_log10(out["conc_um"])
    normalize.concentration_flags(out, ["conc_um"])

    rep_cols = _viability_columns(cfg, "mono")
    for i, col in enumerate(rep_cols, start=1):
        out[f"viability_{i}"] = raw[col].astype("float64")
    out = normalize.add_replicate_stats(
        out,
        [f"viability_{i}" for i in range(1, len(rep_cols) + 1)],
        nominal=int(qc["nominal_mono_replicates"]),
    )
    assert len(rep_cols) == int(qc["nominal_mono_replicates"])

    out["mu_raw"] = raw[spec["mu"]].astype("float64")
    out["x_x0_raw"] = raw[spec["response"]].astype("float64")
    out = normalize.annotate_response(
        out,
        tolerance=float(qc["response_formula_tolerance"]),
        upper_outlier=float(qc["x_over_x0_upper_outlier"]),
    )
    if logger is not None:
        logger.info(
            "standardised single-agent sheet: %d rows, %d missing responses, "
            "%d X/X0>1 preserved",
            len(out),
            int(out["x_x0_raw"].isna().sum()),
            int(out["response_gt1"].sum()),
        )
    return out


# ---------------------------------------------------------------------------
# Entity registry
# ---------------------------------------------------------------------------
def build_entity_registry(
    mono: pd.DataFrame, combo: pd.DataFrame
) -> Dict[str, Any]:
    """Build one canonical name -> ID registry from both sheets combined."""
    cell_names = sorted(set(mono["cell_line"]) | set(combo["cell_line"]))
    drug_names = sorted(
        set(mono["drug_name"])
        | set(combo["drugA_name"])
        | set(combo["drugB_name"])
    )
    pairs = set()
    for a, b in zip(combo["drugA_name"], combo["drugB_name"]):
        pairs.add(tuple(sorted((a, b))))
    return {
        "cells": cell_ids(cell_names),
        "drugs": drug_ids(drug_names),
        "cell_lines": cell_names,
        "drug_names": drug_names,
        "n_cells": len(cell_names),
        "n_drugs": len(drug_names),
        "n_unordered_pairs": len(pairs),
        "unordered_pairs": sorted([" & ".join(p) for p in pairs]),
        "drugs_only_as_a": sorted(set(combo["drugA_name"]) - set(combo["drugB_name"])),
        "drugs_only_as_b": sorted(set(combo["drugB_name"]) - set(combo["drugA_name"])),
    }


def raw_stats(frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Row/column/missing-value inventory for the audit manifest."""
    stats: Dict[str, Any] = {}
    for role in ("mono", "combo", "legend"):
        df = frames[role]
        stats[role] = {
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "columns": [str(c) for c in df.columns],
            "missing_by_column": {str(c): int(df[c].isna().sum()) for c in df.columns},
            "batch_counts": (
                {int(k): int(v) for k, v in df[cfg_batch_col(role)].value_counts().sort_index().items()}
                if role != "legend"
                else {}
            ),
        }
    return stats


def cfg_batch_col(role: str) -> str:
    return "BatchID" if role in ("mono", "combo") else "unused"
