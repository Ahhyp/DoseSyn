"""M2.2 molecular graph featurisation (RDKit -> PyTorch Geometric Data).

The same featuriser is used for both drug columns and for monotherapy drugs.
Only categorical integer features are stored (embedding-ready); no GINE layer
is implemented here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

ATOM_FEATURES: List[Dict[str, Any]] = [
    {"name": "atomic_num", "size": 119, "kind": "categorical"},
    {"name": "formal_charge", "size": 7, "kind": "categorical"},
    {"name": "aromatic", "size": 2, "kind": "categorical"},
    {"name": "hybridization", "size": 7, "kind": "categorical"},
    {"name": "total_hs", "size": 5, "kind": "categorical"},
    {"name": "chiral_tag", "size": 4, "kind": "categorical"},
    {"name": "in_ring", "size": 2, "kind": "categorical"},
    {"name": "ring_size", "size": 9, "kind": "categorical"},
    {"name": "degree", "size": 7, "kind": "categorical"},
]
BOND_FEATURES: List[Dict[str, Any]] = [
    {"name": "bond_type", "size": 5, "kind": "categorical"},
    {"name": "conjugated", "size": 2, "kind": "categorical"},
    {"name": "in_ring", "size": 2, "kind": "categorical"},
    {"name": "stereo", "size": 5, "kind": "categorical"},
]

_HYBRID = {
    Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
    Chem.rdchem.HybridizationType.S: 1,
    Chem.rdchem.HybridizationType.SP: 2,
    Chem.rdchem.HybridizationType.SP2: 3,
    Chem.rdchem.HybridizationType.SP3: 4,
    Chem.rdchem.HybridizationType.SP3D: 5,
    Chem.rdchem.HybridizationType.SP3D2: 6,
}
_CHIRAL = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
}
_BOND_TYPE = {
    Chem.rdchem.BondType.UNSPECIFIED: 0,
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}
_BOND_STEREO = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOANY: 1,
    Chem.rdchem.BondStereo.STEREOZ: 2,
    Chem.rdchem.BondStereo.STEREOE: 3,
}


def _ring_size(atom: Chem.Atom, max_size: int = 8) -> int:
    if not atom.IsInRing():
        return 0
    for size in range(3, max_size + 1):
        if atom.IsInRingSize(size):
            return size
    return max_size


def atom_features(atom: Chem.Atom, cfg: Dict[str, Any]) -> List[int]:
    clip = int(cfg["graph"]["charge_clip"])
    hs_clip = int(cfg["graph"]["total_hs_clip"])
    deg_clip = int(cfg["graph"]["degree_clip"])
    ring_clip = int(cfg["graph"]["ring_size_clip"])
    charge = max(-clip, min(clip, int(atom.GetFormalCharge()))) + clip
    return [
        int(atom.GetAtomicNum()),
        int(charge),
        int(atom.GetIsAromatic()),
        _HYBRID.get(atom.GetHybridization(), 6),
        0 if atom.GetTotalNumHs() == 0 else min(hs_clip, int(atom.GetTotalNumHs())),
        _CHIRAL.get(atom.GetChiralTag(), 3),
        int(atom.IsInRing()),
        min(ring_clip, _ring_size(atom, ring_clip)),
        min(deg_clip, int(atom.GetDegree())),
    ]


def bond_features(bond: Chem.Bond) -> List[int]:
    return [
        _BOND_TYPE.get(bond.GetBondType(), 0),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        _BOND_STEREO.get(bond.GetStereo(), 4),
    ]


def mol_to_data(
    mol: Chem.Mol,
    drug_id: str,
    drug_name: str,
    cfg: Dict[str, Any],
) -> Data:
    x = torch.tensor(
        [atom_features(a, cfg) for a in mol.GetAtoms()], dtype=torch.long
    )
    src: List[int] = []
    dst: List[int] = []
    attr: List[List[int]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        src.extend([i, j])
        dst.extend([j, i])
        attr.extend([feats, feats])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attr, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = int(mol.GetNumAtoms())
    data.drug_id = drug_id
    data.drug_name = drug_name
    return data


def validate_graph(data: Data) -> List[str]:
    problems: List[str] = []
    if data.x is None or data.x.numel() == 0:
        problems.append("empty atom feature matrix")
    if data.edge_index is None or data.edge_index.numel() == 0:
        problems.append("molecule has no bonds")
        return problems
    n = int(data.num_nodes)
    if data.x.shape[0] != n:
        problems.append("x rows != num_nodes")
    if data.edge_attr.shape[0] != data.edge_index.shape[1]:
        problems.append("edge_attr rows != edge_index columns")
    if data.edge_index.min() < 0 or data.edge_index.max() >= n:
        problems.append("edge_index out of range")
    pairs = {(int(a), int(b)) for a, b in data.edge_index.t().tolist()}
    if any((b, a) not in pairs for a, b in pairs):
        problems.append("edge_index is not symmetric")
    if any(a == b for a, b in pairs):
        problems.append("self-loop found")
    for col, spec in enumerate(ATOM_FEATURES):
        vals = data.x[:, col]
        if int(vals.min()) < 0 or int(vals.max()) >= spec["size"]:
            problems.append(f"atom feature {spec['name']} out of range")
    for col, spec in enumerate(BOND_FEATURES):
        vals = data.edge_attr[:, col]
        if int(vals.min()) < 0 or int(vals.max()) >= spec["size"]:
            problems.append(f"bond feature {spec['name']} out of range")
    return problems


def build_drug_graphs(
    mapping: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Data], Dict[str, Any]]:
    graphs: Dict[str, Data] = {}
    graph_meta: Dict[str, Any] = {}
    failures: List[Dict[str, str]] = []

    for _, row in mapping.iterrows():
        drug_id = str(row["drug_id"])
        name = str(row["oneil_drug_name"])
        smiles = row["isomeric_smiles"]
        entry: Dict[str, Any] = {
            "oneil_drug_name": name,
            "canonical_drug_name": row["canonical_drug_name"],
            "drug_id": drug_id,
            "isomeric_smiles": smiles,
            "inchikey": row["inchikey"],
            "graph_status": "FAILED",
            "problems": [],
            "n_nodes": 0,
            "n_edges": 0,
            "n_components": 0,
            "n_rings": 0,
            "n_aromatic_rings": 0,
            "molecular_formula": row["molecular_formula"],
        }
        if not isinstance(smiles, str) or not smiles:
            entry["problems"].append("missing SMILES")
            failures.append({"drug_id": drug_id, "drug": name, "reason": "missing SMILES"})
            graph_meta[drug_id] = entry
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            entry["problems"].append("invalid SMILES")
            failures.append({"drug_id": drug_id, "drug": name, "reason": "invalid SMILES"})
            graph_meta[drug_id] = entry
            continue
        if mol.GetNumAtoms() == 0:
            entry["problems"].append("empty molecule")
            failures.append({"drug_id": drug_id, "drug": name, "reason": "empty molecule"})
            graph_meta[drug_id] = entry
            continue
        if mol.GetNumBonds() == 0:
            entry["problems"].append("molecule has no bonds")
            failures.append({"drug_id": drug_id, "drug": name, "reason": "no bonds"})
            graph_meta[drug_id] = entry
            continue

        data = mol_to_data(mol, drug_id, name, cfg)
        problems = validate_graph(data)
        n_components = len(Chem.GetMolFrags(mol))
        entry.update(
            n_nodes=int(data.num_nodes),
            n_edges=int(data.edge_index.shape[1]),
            n_components=n_components,
            n_rings=int(rdMolDescriptors.CalcNumRings(mol)),
            n_aromatic_rings=int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            problems=problems,
        )
        if problems:
            entry["graph_status"] = "FAILED"
            failures.append(
                {"drug_id": drug_id, "drug": name, "reason": "; ".join(problems)}
            )
        elif n_components > 1:
            entry["graph_status"] = "NEEDS_REVIEW"
            entry["problems"].append(
                f"{n_components} disconnected components (metal coordination or salt)"
            )
        else:
            entry["graph_status"] = "OK"
        graphs[drug_id] = data
        graph_meta[drug_id] = entry

    import rdkit

    metadata = {
        "feature_version": cfg["graph"]["feature_version"],
        "rdkit_version": rdkit.__version__,
        "torch_version": torch.__version__,
        "pyg_version": __import__("torch_geometric").__version__,
        "n_drugs": int(len(mapping)),
        "n_graphs": int(len(graphs)),
        "n_failed": int(sum(1 for v in graph_meta.values() if v["graph_status"] == "FAILED")),
        "n_needs_review": int(
            sum(1 for v in graph_meta.values() if v["graph_status"] == "NEEDS_REVIEW")
        ),
        "n_ok": int(sum(1 for v in graph_meta.values() if v["graph_status"] == "OK")),
        "atom_features": ATOM_FEATURES,
        "bond_features": BOND_FEATURES,
        "graph_encoding": "undirected edge_index with both (i,j) and (j,i); edge_attr duplicated",
        "failures": failures,
        "graphs": graph_meta,
    }
    return graphs, metadata


def save_graphs(
    graphs: Dict[str, Data],
    metadata: Dict[str, Any],
    pt_path: str | Path,
    meta_path: str | Path,
) -> None:
    pt_path = Path(pt_path)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"graphs": graphs, "metadata": metadata}, pt_path)
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2, default=str)
        fh.write("\n")


def load_graphs(pt_path: str | Path) -> Tuple[Dict[str, Data], Dict[str, Any]]:
    payload = torch.load(Path(pt_path), map_location="cpu")
    return payload["graphs"], payload["metadata"]
