"""M2.1 drug entity standardisation.

The O'Neil screen contains 38 drugs.  Structures are taken from ConfSyn's
DrugBank-derived ``drugsmiles.csv`` when available (21 drugs) and from PubChem
for the rest.  Every structure is verified by recomputing its InChIKey with
RDKit and comparing it with PubChem; a name hit alone is never accepted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, inchi, rdMolDescriptors

from ..utils.io import slugify
from .pubchem import PubChemClient, PubChemError

RDLogger.DisableLog("rdApp.*")

# CIDs captured during the audited probe (used only as a fallback when the
# name endpoint is unavailable); all are re-fetched and verified at build time.
KNOWN_CIDS = {
    "AZD1775": 24856436,
    "BEZ-235": 11977753,
    "Carboplatin": 426756,
    "Cyclophosphamide": 2907,
    "Dinaciclib": 46926350,
    "L778123": 216454,
    "MK-2206": 24964624,
    "MK-4541": 59691338,
    "MK-4827": 24958200,
    "MK-5108": 24748204,
    "MK-8669": 11520894,
    "MK-8776": 46239015,
    "MRK-003": 15953832,
    "Mitomycine": 5746,
    "Oxaliplatin": 9887053,
    "PD325901": 9826528,
    "Temozolomide": 5394,
}

DRUG_COLUMNS = [
    "oneil_drug_name",
    "canonical_drug_name",
    "drug_id",
    "pubchem_cid",
    "canonical_smiles",
    "isomeric_smiles",
    "inchikey",
    "mapping_source",
    "mapping_status",
    "molecular_formula",
    "molecular_weight",
    "n_fragments",
    "n_heavy_atoms",
    "appears_as_a",
    "appears_as_b",
    "n_combo_points",
    "n_mono_points",
    "structure_note",
]


def _key(name: str) -> str:
    return str(name).strip().upper()


def load_m1_drug_entities(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Union of drug names/ids in the combination and monotherapy tables."""
    combo = pd.read_parquet(
        cfg["files"]["m1_combo_points"],
        columns=["drug_a_name", "drug_b_name", "drug_a_id", "drug_b_id"],
    )
    mono = pd.read_parquet(
        cfg["files"]["m1_mono_points"], columns=["drug_name", "drug_id"]
    )

    a = combo.groupby(["drug_a_name", "drug_a_id"]).size().rename("n_as_a")
    b = combo.groupby(["drug_b_name", "drug_b_id"]).size().rename("n_as_b")
    m = mono.groupby(["drug_name", "drug_id"]).size().rename("n_mono")

    rows: Dict[str, Dict[str, Any]] = {}
    for (name, did), n in a.items():
        r = rows.setdefault(name, {"drug_id": did, "n_as_a": 0, "n_as_b": 0, "n_mono": 0})
        if r["drug_id"] != did:
            raise ValueError(f"inconsistent drug_id for {name}: {r['drug_id']} vs {did}")
        r["n_as_a"] = int(n)
    for (name, did), n in b.items():
        r = rows.setdefault(name, {"drug_id": did, "n_as_a": 0, "n_as_b": 0, "n_mono": 0})
        if r["drug_id"] != did:
            raise ValueError(f"inconsistent drug_id for {name}: {r['drug_id']} vs {did}")
        r["n_as_b"] = int(n)
    for (name, did), n in m.items():
        r = rows.setdefault(name, {"drug_id": did, "n_as_a": 0, "n_as_b": 0, "n_mono": 0})
        if r["drug_id"] != did:
            raise ValueError(f"inconsistent drug_id for {name}: {r['drug_id']} vs {did}")
        r["n_mono"] = int(n)

    entities = pd.DataFrame(
        [{"oneil_drug_name": k, **v} for k, v in sorted(rows.items())]
    )
    return entities, rows


def load_confsyn_structures(cfg: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Map O'Neil drug name -> {db_id, smiles} from ConfSyn (case-insensitive)."""
    combos = pd.read_csv(
        cfg["files"]["confsyn_drug_combinations"],
        usecols=["drug_a_name", "drug_b_name", "drug1_db", "drug2_db"],
    )
    name2db: Dict[str, str] = {}
    for _, row in combos.iterrows():
        name2db.setdefault(_key(row["drug_a_name"]), str(row["drug1_db"]))
        name2db.setdefault(_key(row["drug_b_name"]), str(row["drug2_db"]))
    smiles = pd.read_csv(cfg["files"]["confsyn_drug_smiles"])
    db2smiles = {
        str(r["DBname"]).strip(): str(r["Smiles"]).strip() for _, r in smiles.iterrows()
    }
    out: Dict[str, Dict[str, str]] = {}
    for name, db in name2db.items():
        if db in db2smiles:
            out[name] = {"db_id": db, "smiles": db2smiles[db]}
    return out


def _mol(smiles: str) -> Optional[Chem.Mol]:
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _inchikey(mol: Chem.Mol) -> Optional[str]:
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def _rdkit_props(mol: Chem.Mol) -> Dict[str, Any]:
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mw": round(float(Descriptors.MolWt(mol)), 2),
        "n_fragments": len(Chem.GetMolFrags(mol)),
        "n_heavy": int(mol.GetNumHeavyAtoms()),
    }


def resolve_drug(
    name: str,
    entity: Dict[str, Any],
    confsyn: Optional[Dict[str, str]],
    client: PubChemClient,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    mp = cfg["drug_mapping"]
    query = mp.get("name_queries", {}).get(name, name)
    canonical_name = mp.get("canonical_names", {}).get(name, name)
    metal = name in set(mp.get("metal_complexes", []))
    notes: List[str] = []

    conf_mol = _mol(confsyn["smiles"]) if confsyn else None
    conf_key = _inchikey(conf_mol) if conf_mol is not None else None

    props: Optional[Dict[str, Any]] = None
    pc_error: Optional[str] = None
    try:
        props = client.fetch_by_name(query)
    except PubChemError as exc:
        pc_error = str(exc)
        try:
            if conf_key:
                props = client.fetch_by_inchikey(conf_key)
                notes.append("name query failed; resolved through ConfSyn InChIKey")
        except PubChemError:
            props = None
        if props is None and name in KNOWN_CIDS:
            try:
                props = client.properties_by_cid(KNOWN_CIDS[name])
                notes.append(f"name query failed; used audited CID {KNOWN_CIDS[name]}")
            except PubChemError:
                props = None

    chosen_smiles: Optional[str] = None
    source = ""
    status = "VERIFIED"
    pc_key = None
    if props is not None:
        pc_key = props.get("InChIKey")
        pc_smiles = props.get("SMILES")
        pc_mol = _mol(pc_smiles) if pc_smiles else None
        pc_calc_key = _inchikey(pc_mol) if pc_mol is not None else None
        if pc_calc_key and pc_key and pc_calc_key != pc_key:
            notes.append(f"PubChem InChIKey inconsistent with its own SMILES ({pc_calc_key})")
            status = "NEEDS_REVIEW"

        if conf_mol is not None and conf_key is not None:
            if pc_key and conf_key == pc_key:
                chosen_smiles = confsyn["smiles"]
                source = f"confsyn_drugsmiles({confsyn['db_id']})+pubchem(CID {props['CID']})"
            elif pc_mol is not None:
                chosen_smiles = pc_smiles
                source = f"pubchem(CID {props['CID']})"
                status = "NEEDS_REVIEW"
                notes.append(
                    f"ConfSyn structure InChIKey {conf_key} != PubChem {pc_key}; used PubChem"
                )
            else:
                chosen_smiles = confsyn["smiles"]
                source = f"confsyn_drugsmiles({confsyn['db_id']})"
                status = "NEEDS_REVIEW"
        elif pc_mol is not None:
            chosen_smiles = pc_smiles
            source = f"pubchem(CID {props['CID']})"
        else:
            chosen_smiles = confsyn["smiles"] if confsyn else None
    else:
        chosen_smiles = confsyn["smiles"] if confsyn else None
        source = f"confsyn_drugsmiles({confsyn['db_id']})" if confsyn else "none"
        status = "NEEDS_REVIEW"
        notes.append(f"PubChem unavailable: {pc_error}")

    if chosen_smiles is None:
        status = "UNMAPPED"
        notes.append("no structural source")
        return {
            "oneil_drug_name": name,
            "canonical_drug_name": canonical_name,
            "drug_id": entity["drug_id"],
            "pubchem_cid": None,
            "canonical_smiles": None,
            "isomeric_smiles": None,
            "inchikey": None,
            "mapping_source": source or "none",
            "mapping_status": status,
            "molecular_formula": None,
            "molecular_weight": None,
            "n_fragments": None,
            "n_heavy_atoms": None,
            "appears_as_a": bool(entity["n_as_a"]),
            "appears_as_b": bool(entity["n_as_b"]),
            "n_combo_points": int(entity["n_as_a"] + entity["n_as_b"]),
            "n_mono_points": int(entity["n_mono"]),
            "structure_note": "; ".join(notes),
        }

    mol = _mol(chosen_smiles)
    if mol is None:
        status = "NEEDS_REVIEW"
        notes.append("chosen SMILES failed RDKit parsing")
        return {
            "oneil_drug_name": name,
            "canonical_drug_name": canonical_name,
            "drug_id": entity["drug_id"],
            "pubchem_cid": props["CID"] if props else None,
            "canonical_smiles": None,
            "isomeric_smiles": None,
            "inchikey": None,
            "mapping_source": source,
            "mapping_status": status,
            "molecular_formula": None,
            "molecular_weight": None,
            "n_fragments": None,
            "n_heavy_atoms": None,
            "appears_as_a": bool(entity["n_as_a"]),
            "appears_as_b": bool(entity["n_as_b"]),
            "n_combo_points": int(entity["n_as_a"] + entity["n_as_b"]),
            "n_mono_points": int(entity["n_mono"]),
            "structure_note": "; ".join(notes),
        }

    rprops = _rdkit_props(mol)
    key = _inchikey(mol)
    n_frag = rprops["n_fragments"]
    if n_frag > 1 and not metal:
        status = "NEEDS_REVIEW"
        notes.append(f"{n_frag} disconnected fragments (salt/hydrate/mixture?)")
    if metal:
        status = "NEEDS_REVIEW"
        notes.append(
            f"metal-organic complex; RDKit yields {n_frag} disconnected fragment(s) "
            "because Pt coordination bonds are not encoded"
        )
    if query != name:
        notes.append(f"structure resolved via PubChem synonym {query!r}")
        if status == "VERIFIED":
            status = "VERIFIED_ALIAS"
    if name == "ABT-888":
        notes.append(
            "PubChem page for 'ABT-888' is the dihydrochloride salt; free base "
            "veliparib CID used"
        )
    if pc_key and key and key != pc_key:
        status = "NEEDS_REVIEW"
        notes.append(f"final InChIKey {key} != PubChem {pc_key}")

    canonical_smiles = (
        props.get("ConnectivitySMILES")
        if props and props.get("ConnectivitySMILES")
        else Chem.MolToSmiles(mol, isomericSmiles=False)
    )
    isomeric_smiles = (
        props.get("SMILES") if props and props.get("SMILES") else Chem.MolToSmiles(mol, isomericSmiles=True)
    )

    return {
        "oneil_drug_name": name,
        "canonical_drug_name": canonical_name,
        "drug_id": entity["drug_id"],
        "pubchem_cid": int(props["CID"]) if props else None,
        "canonical_smiles": canonical_smiles,
        "isomeric_smiles": isomeric_smiles,
        "inchikey": key,
        "mapping_source": source,
        "mapping_status": status,
        "molecular_formula": (
            props.get("MolecularFormula") if props else rprops["formula"]
        ),
        "molecular_weight": (
            float(props["MolecularWeight"]) if props else rprops["mw"]
        ),
        "n_fragments": n_frag,
        "n_heavy_atoms": rprops["n_heavy"],
        "appears_as_a": bool(entity["n_as_a"]),
        "appears_as_b": bool(entity["n_as_b"]),
        "n_combo_points": int(entity["n_as_a"] + entity["n_as_b"]),
        "n_mono_points": int(entity["n_mono"]),
        "structure_note": "; ".join(notes),
    }


def build_drug_mapping(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    entities, _ = load_m1_drug_entities(cfg)
    confsyn = load_confsyn_structures(cfg)
    cache_dir = Path(cfg["paths"]["external_dir"]) / cfg["pubchem"]["cache_subdir"]
    client = PubChemClient(
        cache_dir=cache_dir,
        base_url=cfg["pubchem"]["base_url"],
        timeout=cfg["pubchem"]["timeout_seconds"],
        sleep_seconds=cfg["pubchem"]["sleep_seconds"],
        user_agent=cfg["pubchem"]["user_agent"],
    )

    rows = []
    for _, ent in entities.iterrows():
        name = ent["oneil_drug_name"]
        rows.append(
            resolve_drug(
                name,
                ent.to_dict(),
                confsyn.get(_key(name)),
                client,
                cfg,
            )
        )
    df = pd.DataFrame(rows)[DRUG_COLUMNS]
    df = df.sort_values("oneil_drug_name").reset_index(drop=True)

    diagnostics = {
        "n_drugs": int(len(df)),
        "n_confsyn_smiles_available": int(len(confsyn)),
        "n_using_confsyn_structure": int(df["mapping_source"].str.startswith("confsyn").sum()),
        "n_pubchem_queries": int(client.n_requests),
        "n_cache_hits": int(client.n_cache_hits),
        "status_counts": df["mapping_status"].value_counts().to_dict(),
        "n_missing_smiles": int(df["isomeric_smiles"].isna().sum()),
        "n_missing_inchikey": int(df["inchikey"].isna().sum()),
        "n_review": int((df["mapping_status"] == "NEEDS_REVIEW").sum()),
        "review_drugs": df.loc[df["mapping_status"] == "NEEDS_REVIEW", "oneil_drug_name"].tolist(),
        "n_duplicate_cid": int(df["pubchem_cid"].duplicated().sum()),
        "generated_features": ["canonical_smiles", "isomeric_smiles", "inchikey"],
    }
    return df, diagnostics
