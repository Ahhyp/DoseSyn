"""M4 tensor preparation from frozen M3 splits.

The loader below is deliberately explicit so that every tensor used by a model
can be traced back to a frozen M1/M2 artefact and a frozen M3 split file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch

from ..features.transform_train_only import (
    concentration_columns,
    transform_dose,
    transform_standardizer,
)
from ..utils.config import load_config

LABEL = "x_x0_raw"
N_DOSES = 4
N_POINTS = 16


def resolve(cfg: Dict[str, Any], value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def split_dir(cfg: Dict[str, Any], scenario: str, seed: Optional[int], fold: Optional[int]) -> Path:
    if scenario == "s3_unseen_cell":
        if fold is None:
            raise ValueError("S3 requires fold")
        name = f"fold_{int(fold)}"
    else:
        if seed is None:
            raise ValueError("S1/S2 require seed")
        name = f"seed_{int(seed)}"
    return resolve(cfg, cfg["outputs"]["splits_dir"]) / scenario / name


@dataclass
class MatrixTable:
    scenario: str
    seed: Optional[int]
    fold: Optional[int]
    split: str
    split_dir: Path
    matrix_ids: List[str]
    condition_ids: List[str]
    pair_ids: List[str]
    cell_ids: List[str]
    cell_lines: List[str]
    depmap_ids: List[str]
    batch_ids: np.ndarray
    drug_a_ids: List[str]
    drug_b_ids: List[str]
    drug_a_names: List[str]
    drug_b_names: List[str]
    drug_a_index: np.ndarray
    drug_b_index: np.ndarray
    dose_a_std: np.ndarray
    dose_b_std: np.ndarray
    dose_a_zero: np.ndarray
    dose_b_zero: np.ndarray
    dose_a_um: np.ndarray
    dose_b_um: np.ndarray
    cell_features: np.ndarray
    y: np.ndarray
    drug_vocab: List[str]
    scaler_metadata: Dict[str, Any]

    def __len__(self) -> int:
        return len(self.matrix_ids)

    def subset(self, indices: Sequence[int]) -> "MatrixTable":
        idx = np.asarray(list(indices), dtype=np.int64)
        return MatrixTable(
            **{k: (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx])
               for k, v in self.__dict__.items() if k not in ("scenario", "seed", "fold", "split", "split_dir", "drug_vocab", "scaler_metadata")},
            scenario=self.scenario,
            seed=self.seed,
            fold=self.fold,
            split=self.split,
            split_dir=self.split_dir,
            drug_vocab=self.drug_vocab,
            scaler_metadata=self.scaler_metadata,
        )


def _reconstruct(points: pd.DataFrame, matrix_ids: Sequence[str]) -> tuple:
    """Return (y[n,4,4], dose_a_um[n,4], dose_b_um[n,4], row_order_valid)."""
    wanted = set(matrix_ids)
    sub = points[points["matrix_id"].isin(wanted)]
    sub = sub.sort_values(["matrix_id", "row_idx", "col_idx"]).reset_index(drop=True)
    if len(sub) != len(matrix_ids) * N_POINTS:
        raise ValueError(f"expected {len(matrix_ids)*N_POINTS} points, got {len(sub)}")
    mids = sub["matrix_id"].to_numpy().reshape(-1, N_POINTS)
    if not (mids == mids[:, :1]).all():
        raise ValueError("points are not contiguous per matrix")
    ordered = [str(m) for m in mids[:, 0]]
    if ordered != list(matrix_ids):
        # matrix_ids is sorted; keep the sorted order as canonical
        pass
    vals = sub[["conc_a_um", "conc_b_um", LABEL]].to_numpy(dtype=np.float64).reshape(-1, N_POINTS, 3)
    row_idx = sub["row_idx"].to_numpy().reshape(-1, N_POINTS)
    col_idx = sub["col_idx"].to_numpy().reshape(-1, N_POINTS)
    expect_r = np.tile(np.repeat(np.arange(4), 4), (len(matrix_ids), 1))
    expect_c = np.tile(np.tile(np.arange(4), 4), (len(matrix_ids), 1))
    if not (row_idx == expect_r).all() or not (col_idx == expect_c).all():
        raise ValueError("points are not in 4x4 row-major order")
    y = vals[:, :, 2].reshape(-1, 4, 4)
    dose_a = vals[:, 0::4, 0]
    dose_b = vals[:, :4, 1]
    if not np.isfinite(y).all():
        raise ValueError("non-finite labels")
    return y, dose_a, dose_b


def load_matrix_table(
    scenario: str,
    split: str,
    *,
    seed: Optional[int] = None,
    fold: Optional[int] = None,
    config_path: str = "configs/dataset_primary_v1.yaml",
    m4_config_path: str = "configs/m4_training.yaml",
) -> MatrixTable:
    cfg = load_config(config_path)
    m4 = load_config(m4_config_path)
    sdir = split_dir(cfg, scenario, seed, fold)
    conditions = pd.read_parquet(sdir / f"{split}_conditions.parquet")
    wanted_conditions = set(conditions["condition_id"].astype(str))
    primary = pd.read_parquet(resolve(cfg, cfg["outputs"]["matrix_index"]))
    primary = primary[primary["condition_id"].astype(str).isin(wanted_conditions)].copy()
    if primary.empty:
        raise ValueError(f"{scenario}/{split} is empty")
    primary = primary.sort_values("matrix_id").reset_index(drop=True)
    matrix_ids = primary["matrix_id"].astype(str).tolist()

    points = pd.read_parquet(
        resolve(cfg, cfg["inputs"]["combo_points"]),
        columns=["matrix_id", "row_idx", "col_idx", "conc_a_um", "conc_b_um", LABEL],
    )
    y, dose_a_um, dose_b_um = _reconstruct(points, matrix_ids)

    scalers = joblib.load(sdir / "scalers.joblib")
    pathway = pd.read_parquet(resolve(cfg, cfg["inputs"]["pathway_features"])).set_index("DepMap_ID")
    pcols = scalers["pathway"]["columns"]
    pmat = transform_standardizer(pathway[pcols].to_numpy(), scalers["pathway"]).astype(np.float32)
    pos = {str(d): i for i, d in enumerate(pathway.index)}
    cf = np.stack([pmat[pos[str(d)]] for d in primary["depmap_id"]], axis=0)

    dose_a_std = np.zeros((len(matrix_ids), 4), dtype=np.float32)
    dose_b_std = np.zeros((len(matrix_ids), 4), dtype=np.float32)
    dose_a_zero = np.zeros((len(matrix_ids), 4), dtype=np.float32)
    dose_b_zero = np.zeros((len(matrix_ids), 4), dtype=np.float32)
    for i in range(len(matrix_ids)):
        ca = concentration_columns(dose_a_um[i])
        cb = concentration_columns(dose_b_um[i])
        dose_a_std[i] = transform_dose(ca["log10_molar"], ca["zero_flag"], scalers["drug_a_dose"]).astype(np.float32)
        dose_b_std[i] = transform_dose(cb["log10_molar"], cb["zero_flag"], scalers["drug_b_dose"]).astype(np.float32)
        dose_a_zero[i] = ca["zero_flag"].astype(np.float32)
        dose_b_zero[i] = cb["zero_flag"].astype(np.float32)

    drug_vocab = sorted(set(primary["drug_a_id"].astype(str)) | set(primary["drug_b_id"].astype(str)))
    dpos = {d: i for i, d in enumerate(drug_vocab)}
    meta = json.loads((sdir / "split_metadata.json").read_text(encoding="utf-8"))
    return MatrixTable(
        scenario=scenario,
        seed=seed,
        fold=fold,
        split=split,
        split_dir=sdir,
        matrix_ids=matrix_ids,
        condition_ids=primary["condition_id"].astype(str).tolist(),
        pair_ids=primary["pair_id"].astype(str).tolist(),
        cell_ids=primary["cell_id"].astype(str).tolist(),
        cell_lines=primary["cell_line"].astype(str).tolist(),
        depmap_ids=primary["depmap_id"].astype(str).tolist(),
        batch_ids=primary["batch_id"].astype(np.int64).to_numpy(),
        drug_a_ids=primary["drug_a_id"].astype(str).tolist(),
        drug_b_ids=primary["drug_b_id"].astype(str).tolist(),
        drug_a_names=primary["drug_a_name"].astype(str).tolist(),
        drug_b_names=primary["drug_b_name"].astype(str).tolist(),
        drug_a_index=np.array([dpos[d] for d in primary["drug_a_id"].astype(str)], dtype=np.int64),
        drug_b_index=np.array([dpos[d] for d in primary["drug_b_id"].astype(str)], dtype=np.int64),
        dose_a_std=dose_a_std,
        dose_b_std=dose_b_std,
        dose_a_zero=dose_a_zero,
        dose_b_zero=dose_b_zero,
        dose_a_um=dose_a_um.astype(np.float32),
        dose_b_um=dose_b_um.astype(np.float32),
        cell_features=cf,
        y=y.astype(np.float32),
        drug_vocab=drug_vocab,
        scaler_metadata={
            "split_dir": str(sdir),
            "scaler_sha256": meta.get("scaler_sha256"),
            "content_sha256": meta.get("content_sha256"),
            "mono_policy": meta.get("mono_policy"),
            "train_cells": meta.get("train_cells"),
        },
    )


def load_graphs(m4_config_path: str = "configs/m4_training.yaml") -> tuple:
    m4 = load_config(m4_config_path)
    p = resolve(m4, m4["inputs"]["drug_graphs"])
    payload = torch.load(p, map_location="cpu")
    return payload["graphs"], payload.get("metadata", {})


def graph_batch(
    table: MatrixTable,
    indices: Sequence[int],
    graphs: Dict[str, Any],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Collate 2*graphs atom tensors with duplicated undirected edges."""
    xs, eis, eas, batch_vec, graph_drug = [], [], [], [], []
    offset = 0
    drug_a_local, drug_b_local = [], []
    for local, gi in enumerate(indices):
        for role, drug_id in ((0, table.drug_a_ids[gi]), (1, table.drug_b_ids[gi])):
            g = graphs[drug_id]
            x = g.x.long()
            ei = g.edge_index.long() + offset
            ea = g.edge_attr.long()
            xs.append(x)
            eis.append(ei)
            eas.append(ea)
            batch_vec.append(torch.full((x.shape[0],), len(graph_drug), dtype=torch.long))
            graph_drug.append(drug_id)
            offset += x.shape[0]
            if role == 0:
                drug_a_local.append(len(graph_drug) - 1)
            else:
                drug_b_local.append(len(graph_drug) - 1)
    return {
        "drug_x": torch.cat(xs, dim=0).to(device),
        "drug_edge_index": torch.cat(eis, dim=1).to(device),
        "drug_edge_attr": torch.cat(eas, dim=0).to(device),
        "drug_batch": torch.cat(batch_vec, dim=0).to(device),
        "drug_a_graph": torch.tensor(drug_a_local, dtype=torch.long, device=device),
        "drug_b_graph": torch.tensor(drug_b_local, dtype=torch.long, device=device),
        "n_graphs": len(graph_drug),
    }


def table_batch(table: MatrixTable, indices: Sequence[int], *, device: torch.device) -> Dict[str, torch.Tensor]:
    idx = np.asarray(list(indices), dtype=np.int64)
    return {
        "cell_features": torch.from_numpy(table.cell_features[idx]).to(device),
        "dose_a": torch.from_numpy(table.dose_a_std[idx]).to(device),
        "dose_b": torch.from_numpy(table.dose_b_std[idx]).to(device),
        "dose_a_zero": torch.from_numpy(table.dose_a_zero[idx]).to(device),
        "dose_b_zero": torch.from_numpy(table.dose_b_zero[idx]).to(device),
        "drug_a_index": torch.from_numpy(table.drug_a_index[idx]).to(device),
        "drug_b_index": torch.from_numpy(table.drug_b_index[idx]).to(device),
        "y": torch.from_numpy(table.y[idx]).to(device),
    }
