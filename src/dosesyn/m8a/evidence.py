"""M8-A: provenance and A/B/C/D evidence classification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .mapping import add_condition_key, normalize_text


SCENARIOS = ("s1_random_condition", "s2_unseen_pair", "s3_unseen_cell")
LEVELS = ("A", "B", "C", "D")

LEVEL_MEANING = {
    "A": "ConfSyn-independent condition and DoseSyn held-out test matrix",
    "B": "ConfSyn-independent condition but DoseSyn train/val matrix",
    "C": "condition present in ConfSyn DrugCombDB corpus; split membership unknown",
    "D": "not bridgeable (entity/condition missing or matrix excluded)",
}


def load_conf_conditions(
    confsyn_data: Path,
    drug_map: pd.DataFrame,
    cell_map: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per ConfSyn condition with dataset provenance flags.

    Only conditions whose two drugs and cell map to DoseSyn entities are kept.
    """
    data = Path(confsyn_data)
    dbname_to_drug = {
        str(r.conf_dbname): r.drug_id
        for r in drug_map.itertuples()
        if isinstance(r.conf_dbname, str) and r.match_status in ("EXACT", "REVIEWED_ALIAS")
    }
    conf_cell_to_dose = {
        normalize_text(r.conf_cell_name): r.cell_id
        for r in cell_map.itertuples()
        if isinstance(r.conf_cell_name, str) and r.match_status in ("EXACT", "REVIEWED_ALIAS")
    }

    frames: List[pd.DataFrame] = []
    db_path = data / "DrugCombDB" / "drug_combinations.csv"
    if db_path.exists():
        db = pd.read_csv(db_path)
        db["dataset"] = "DrugCombDB"
        db["drug_1_id"] = db["drug1_db"].map(dbname_to_drug)
        db["drug_2_id"] = db["drug2_db"].map(dbname_to_drug)
        db["cell_id"] = db["cell"].map(lambda x: conf_cell_to_dose.get(normalize_text(x)))
        db = db.dropna(subset=["drug_1_id", "drug_2_id", "cell_id"])
        db = add_condition_key(db, "cell_id", "drug_1_id", "drug_2_id")
        db["label"] = pd.to_numeric(db["synergy"], errors="coerce")
        frames.append(db[["dataset", "condition_key", "cell_id", "drug_low_id", "drug_high_id", "label"]])

    os_path = data / "OncologyScreen" / "drug_combinations.csv"
    if os_path.exists():
        os = pd.read_csv(os_path)
        os["dataset"] = "OncologyScreen"
        os["drug_1_id"] = os["drug1_db"].map(dbname_to_drug)
        os["drug_2_id"] = os["drug2_db"].map(dbname_to_drug)
        os["cell_id"] = os["cell"].map(lambda x: conf_cell_to_dose.get(normalize_text(x)))
        os = os.dropna(subset=["drug_1_id", "drug_2_id", "cell_id"])
        os = add_condition_key(os, "cell_id", "drug_1_id", "drug_2_id")
        os["label"] = pd.to_numeric(os["synergy"], errors="coerce")
        frames.append(os[["dataset", "condition_key", "cell_id", "drug_low_id", "drug_high_id", "label"]])

    if not frames:
        return pd.DataFrame(
            columns=["condition_key", "cell_id", "drug_low_id", "drug_high_id", "in_drugcombdb", "in_oncology_screen", "db_n", "os_n", "db_label_mean", "os_label_mean"]
        )

    long = pd.concat(frames, ignore_index=True)
    agg = (
        long.groupby(["condition_key", "cell_id", "drug_low_id", "drug_high_id", "dataset"])["label"]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
    )
    out = long[["condition_key", "cell_id", "drug_low_id", "drug_high_id"]].drop_duplicates()
    db = agg[agg.dataset == "DrugCombDB"].rename(
        columns={"count": "db_n", "mean": "db_label_mean", "min": "db_label_min", "max": "db_label_max"}
    )
    os = agg[agg.dataset == "OncologyScreen"].rename(
        columns={"count": "os_n", "mean": "os_label_mean", "min": "os_label_min", "max": "os_label_max"}
    )
    out = out.merge(db.drop(columns=["dataset"]), on="condition_key", how="left")
    out = out.merge(os.drop(columns=["dataset"]), on="condition_key", how="left")
    out["in_drugcombdb"] = out["db_n"].notna()
    out["in_oncology_screen"] = out["os_n"].notna()
    for col in ("db_n", "os_n"):
        out[col] = out[col].fillna(0).astype(int)
    return out


def load_dosesyn_conditions(matrix_index: pd.DataFrame) -> pd.DataFrame:
    """Aggregate frozen Primary-v1 matrices to one row per condition."""
    work = matrix_index.copy()
    work["cell_id"] = work["cell_id"].astype(str)
    work["drug_a_id"] = work["drug_a_id"].astype(str)
    work["drug_b_id"] = work["drug_b_id"].astype(str)
    work = add_condition_key(work, "cell_id", "drug_a_id", "drug_b_id")
    agg = (
        work.groupby("condition_key")
        .agg(
            condition_id=("condition_id", "first"),
            pair_id=("pair_id", "first"),
            cell_id=("cell_id", "first"),
            cell_line=("cell_line", "first"),
            drug_a_name=("drug_a_name", "first"),
            drug_b_name=("drug_b_name", "first"),
            n_matrices=("matrix_id", "nunique"),
            n_batches=("batch_id", "nunique"),
        )
        .reset_index()
    )
    return agg


def match_conditions(dose_conditions: pd.DataFrame, conf_conditions: pd.DataFrame) -> pd.DataFrame:
    """Inner-join DoseSyn and ConfSyn conditions on the canonical key."""
    conf_cols = [
        "condition_key",
        "in_drugcombdb",
        "in_oncology_screen",
        "db_n",
        "os_n",
        "db_label_mean",
        "os_label_mean",
        "db_label_min",
        "db_label_max",
        "os_label_min",
        "os_label_max",
    ]
    matched = dose_conditions.merge(conf_conditions[conf_cols], on="condition_key", how="inner")
    matched["in_drugcombdb"] = matched["in_drugcombdb"].fillna(False).astype(bool)
    matched["in_oncology_screen"] = matched["in_oncology_screen"].fillna(False).astype(bool)
    matched["label_source_class"] = np.select(
        [matched["in_drugcombdb"] & matched["in_oncology_screen"], matched["in_drugcombdb"], matched["in_oncology_screen"]],
        ["both_conf_datasets", "drugcombdb_only", "oncology_screen_only"],
        default="none",
    )
    return matched


def classify_condition(
    in_drugcombdb: bool,
    supported: bool,
    role: str,
    included: bool = True,
) -> Tuple[str, str]:
    """Return (evidence_level, reason) for one DoseSyn matrix/split role."""
    if not included:
        return "D", "matrix excluded from Primary-v1"
    if not supported:
        return "D", "ConfSyn has no drug/cell coverage for this condition"
    if in_drugcombdb:
        return "C", "condition is in ConfSyn DrugCombDB; per-sample train/test membership not persisted"
    if role == "test":
        return "A", "condition absent from ConfSyn corpus; DoseSyn test matrix"
    if role in ("train", "val"):
        return "B", f"condition absent from ConfSyn corpus; DoseSyn {role} matrix"
    return "D", f"unknown DoseSyn role: {role}"


def build_matrix_evidence(
    split_root: Path,
    matrix_index: pd.DataFrame,
    matched: pd.DataFrame,
    drug_map: pd.DataFrame,
    cell_map: pd.DataFrame,
) -> pd.DataFrame:
    """Per (matrix, split, role) evidence rows for the 11 frozen splits."""
    conf_drugs = {
        str(r.drug_id)
        for r in drug_map.itertuples()
        if r.match_status in ("EXACT", "REVIEWED_ALIAS")
    }
    conf_cells = {
        str(r.cell_id)
        for r in cell_map.itertuples()
        if r.match_status in ("EXACT", "REVIEWED_ALIAS")
    }
    flags = matched.set_index("condition_key")[["in_drugcombdb", "in_oncology_screen"]]

    base = matrix_index.copy()
    base["cell_id"] = base["cell_id"].astype(str)
    base["drug_a_id"] = base["drug_a_id"].astype(str)
    base["drug_b_id"] = base["drug_b_id"].astype(str)
    base = add_condition_key(base, "cell_id", "drug_a_id", "drug_b_id")
    base["supported"] = (
        base["drug_a_id"].isin(conf_drugs)
        & base["drug_b_id"].isin(conf_drugs)
        & base["cell_id"].isin(conf_cells)
    )

    rows = []
    split_root = Path(split_root)
    for scenario in SCENARIOS:
        scenario_dir = split_root / scenario
        if not scenario_dir.exists():
            continue
        for split_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            split_id = split_dir.name
            for role in ("train", "val", "test"):
                path = split_dir / f"{role}_conditions.parquet"
                if not path.exists():
                    continue
                cond = pd.read_parquet(path, columns=["condition_id"])
                sub = base[base["condition_id"].isin(set(cond["condition_id"].astype(str)))]
                if sub.empty:
                    continue
                sub = sub.copy()
                sub["in_drugcombdb"] = sub["condition_key"].map(flags["in_drugcombdb"]).fillna(False).astype(bool)
                sub["in_oncology_screen"] = sub["condition_key"].map(flags["in_oncology_screen"]).fillna(False).astype(bool)
                levels, reasons = zip(
                    *[
                        classify_condition(db, sup, role, inc)
                        for db, sup, inc in zip(
                            sub["in_drugcombdb"].tolist(),
                            sub["supported"].tolist(),
                            sub["included"].astype(bool).tolist(),
                        )
                    ]
                )
                sub["evidence_level"] = levels
                sub["evidence_reason"] = reasons
                sub["scenario"] = scenario
                sub["split_id"] = split_id
                sub["role"] = role
                rows.append(
                    sub[
                        [
                            "scenario",
                            "split_id",
                            "role",
                            "matrix_id",
                            "condition_id",
                            "condition_key",
                            "pair_id",
                            "cell_id",
                            "cell_line",
                            "batch_id",
                            "drug_a_id",
                            "drug_b_id",
                            "drug_a_name",
                            "drug_b_name",
                            "in_drugcombdb",
                            "in_oncology_screen",
                            "supported",
                            "included",
                            "evidence_level",
                            "evidence_reason",
                        ]
                    ]
                )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty:
        return pd.DataFrame(columns=["scenario", "split_id", "role", "level", "n_matrices", "n_conditions"])
    out = (
        evidence.groupby(["scenario", "split_id", "role", "evidence_level"])
        .agg(n_matrices=("matrix_id", "nunique"), n_conditions=("condition_id", "nunique"))
        .reset_index()
        .rename(columns={"evidence_level": "level"})
    )
    return out.sort_values(["scenario", "split_id", "role", "level"]).reset_index(drop=True)


def bridge_summary(evidence: pd.DataFrame) -> Dict[str, object]:
    """Aggregate test-set bridge counts by scenario and evidence level."""
    if evidence.empty:
        return {"verdict": "BLOCKED", "reason": "no evidence rows"}
    test = evidence[evidence["role"] == "test"]
    per_scenario: Dict[str, Dict[str, object]] = {}
    for scenario in SCENARIOS:
        sub = test[test["scenario"] == scenario]
        if sub.empty:
            continue
        per_scenario[scenario] = {
            "n_test_matrices_union": int(sub["matrix_id"].nunique()),
            "n_test_conditions_union": int(sub["condition_id"].nunique()),
            "levels": {
                level: {
                    "n_matrices_union": int(sub[sub["evidence_level"] == level]["matrix_id"].nunique()),
                    "n_conditions_union": int(sub[sub["evidence_level"] == level]["condition_id"].nunique()),
                }
                for level in LEVELS
            },
        }
    return per_scenario
