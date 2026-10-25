"""M8-A: entity mapping between ConfSyn and DoseSyn.

All functions are read-only with respect to frozen artefacts. ConfSyn files are
read from the local checkout; nothing under the ConfSyn repository is written.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

try:  # RDKit is a project dependency (M2 graph construction used it)
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover - only used for a clear error message
    Chem = None


CONF_DRUGDB = "DrugCombDB"
CONF_ONCOLOGY = "OncologyScreen"


def normalize_text(value: object) -> str:
    """Upper-case, strip and collapse separators for name comparisons."""
    text = str(value).strip().upper()
    text = re.sub(r"[\s\-_/]+", "", text)
    return text


def inchikey_of(smiles: object) -> Optional[str]:
    if Chem is None or not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def canonical_smiles_of(smiles: object) -> Optional[str]:
    if Chem is None or not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def load_conf_drug_smiles(confsyn_data: Path) -> pd.DataFrame:
    """Load DrugCombDB and OncologyScreen DBname->SMILES tables."""
    frames = []
    for dataset, rel in (
        (CONF_DRUGDB, "DrugCombDB/drugsmiles.csv"),
        (CONF_ONCOLOGY, "OncologyScreen/drugsmiles.csv"),
    ):
        path = Path(confsyn_data) / rel
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["dataset"] = dataset
        df["inchikey"] = df["Smiles"].map(inchikey_of)
        df["canonical_smiles"] = df["Smiles"].map(canonical_smiles_of)
        frames.append(df.rename(columns={"DBname": "dbname", "Smiles": "smiles"}))
    if not frames:
        return pd.DataFrame(columns=["dataset", "dbname", "smiles", "inchikey", "canonical_smiles"])
    return pd.concat(frames, ignore_index=True)


def build_drug_mapping(dose_drugs: pd.DataFrame, conf_drugs: pd.DataFrame) -> pd.DataFrame:
    """Map every DoseSyn drug to a ConfSyn SMILES by InChIKey, then skeleton.

    Statuses follow the M8 evidence vocabulary:
      EXACT          full InChIKey match
      REVIEWED_ALIAS same InChIKey skeleton (first 14 chars), documented reason
      UNMATCHED      no ConfSyn structure
    """
    if dose_drugs.empty or conf_drugs.empty:
        return pd.DataFrame(
            columns=[
                "drug_id",
                "oneil_drug_name",
                "conf_dbname",
                "conf_dataset",
                "conf_inchikey",
                "match_type",
                "match_status",
                "match_reason",
            ]
        )

    conf = conf_drugs.dropna(subset=["inchikey"]).copy()
    conf["skeleton"] = conf["inchikey"].str[:14]
    by_key: Dict[str, pd.Series] = {}
    by_skel: Dict[str, pd.Series] = {}
    for _, row in conf.iterrows():
        by_key.setdefault(row["inchikey"], row)
        by_skel.setdefault(row["skeleton"], row)

    records = []
    for _, row in dose_drugs.iterrows():
        key = row.get("inchikey")
        candidate = by_key.get(key) if isinstance(key, str) else None
        match_type = "EXACT"
        reason = "full InChIKey match to ConfSyn SMILES"
        if candidate is None and isinstance(key, str):
            candidate = by_skel.get(key[:14])
            if candidate is not None:
                match_type = "SKELETON"
                reason = "InChIKey skeleton match (stereo/salt-level difference); review required"
        status = "UNMATCHED"
        if candidate is not None:
            status = "EXACT" if match_type == "EXACT" else "REVIEWED_ALIAS"
        records.append(
            {
                "drug_id": row.get("drug_id"),
                "oneil_drug_name": row.get("oneil_drug_name"),
                "conf_dbname": None if candidate is None else candidate.get("dbname"),
                "conf_dataset": None if candidate is None else candidate.get("dataset"),
                "conf_inchikey": None if candidate is None else candidate.get("inchikey"),
                "match_type": match_type if candidate is not None else "NONE",
                "match_status": status,
                "match_reason": reason if candidate is not None else "no ConfSyn structure with this InChIKey",
            }
        )
    return pd.DataFrame.from_records(records)


def build_cell_mapping(dose_cells: pd.DataFrame, conf_cells: Iterable[str]) -> pd.DataFrame:
    """Map DoseSyn cell lines to ConfSyn cell names.

    ConfSyn uses stripped CCLE-style names. We accept exact normalized matches
    on the O'Neil name or the standardized DepMap name, plus explicitly listed
    reviewed aliases. No fuzzy matching is accepted.
    """
    aliases = {
        "LNCAPCLONEFGC": "LNCAP",
        "MSTO211H": "MSTO",
        "NIHOVCAR3": "OVCAR3",
        "COLO320": "COLO320DM",
    }
    wanted: Dict[str, str] = {}
    for name in conf_cells:
        if name is None or (isinstance(name, float) and pd.isna(name)):
            continue
        norm = normalize_text(name)
        if not norm:
            continue
        target = aliases.get(norm, norm)
        wanted.setdefault(target, str(name))

    records = []
    for _, row in dose_cells.iterrows():
        candidates = [row.get("oneil_cell_name"), row.get("standard_cell_name")]
        match = None
        via = None
        for value in candidates:
            if not isinstance(value, str):
                continue
            norm = normalize_text(value)
            if norm in aliases:
                norm = aliases[norm]
            if norm in wanted:
                match = wanted[norm]
                via = value
                break
        # Also accept canonical alias targets such as COLO320DM.
        status = "UNMATCHED"
        reason = "no exact/aliased ConfSyn cell name"
        if match is not None:
            status = "EXACT" if normalize_text(match) == normalize_text(via) else "REVIEWED_ALIAS"
            reason = f"matched via {via}"
        records.append(
            {
                "cell_id": row.get("cell_id"),
                "oneil_cell_name": row.get("oneil_cell_name"),
                "standard_cell_name": row.get("standard_cell_name"),
                "depmap_id": row.get("depmap_id"),
                "conf_cell_name": match,
                "match_status": status,
                "match_reason": reason,
            }
        )
    return pd.DataFrame.from_records(records)


def condition_key(cell_id: str, drug_a_id: str, drug_b_id: str) -> Tuple[str, str, str]:
    """Canonical unordered condition key (cell, sorted drug pair)."""
    a, b = sorted([str(drug_a_id), str(drug_b_id)])
    return str(cell_id), a, b


def add_condition_key(df: pd.DataFrame, cell_col: str, a_col: str, b_col: str) -> pd.DataFrame:
    keys = [condition_key(c, a, b) for c, a, b in zip(df[cell_col], df[a_col], df[b_col])]
    out = df.copy()
    out["cell_id"] = [k[0] for k in keys]
    out["drug_low_id"] = [k[1] for k in keys]
    out["drug_high_id"] = [k[2] for k in keys]
    out["condition_key"] = [f"{k[0]}|{k[1]}|{k[2]}" for k in keys]
    return out
