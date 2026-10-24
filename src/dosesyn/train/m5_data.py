"""M5 monotherapy data access with explicit label-protocol guards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch

from ..features.transform_train_only import concentration_columns, transform_dose, transform_standardizer
from ..training.m4_data import resolve, split_dir
from ..utils.config import load_config


class MonoAccessError(RuntimeError):
    """Raised when a caller tries to read held-out monotherapy labels."""


@dataclass
class MonoTable:
    scenario: str
    seed: Optional[int]
    fold: Optional[int]
    curve_ids: List[str]
    condition_ids: List[str]
    cell_ids: List[str]
    depmap_ids: List[str]
    drug_ids: List[str]
    dose_std: np.ndarray
    zero: np.ndarray
    cell_features: np.ndarray
    y: np.ndarray
    policy: str

    def __len__(self) -> int:
        return len(self.curve_ids)

    def subset(self, indices: Sequence[int]) -> "MonoTable":
        idx = np.asarray(list(indices), dtype=np.int64)
        return MonoTable(
            scenario=self.scenario,
            seed=self.seed,
            fold=self.fold,
            curve_ids=[self.curve_ids[i] for i in idx],
            condition_ids=[self.condition_ids[i] for i in idx],
            cell_ids=[self.cell_ids[i] for i in idx],
            depmap_ids=[self.depmap_ids[i] for i in idx],
            drug_ids=[self.drug_ids[i] for i in idx],
            dose_std=self.dose_std[idx],
            zero=self.zero[idx],
            cell_features=self.cell_features[idx],
            y=self.y[idx],
            policy=self.policy,
        )


def load_mono_assignments(
    scenario: str,
    *,
    seed: Optional[int] = None,
    fold: Optional[int] = None,
    config_path: str = "configs/dataset_primary_v1.yaml",
) -> pd.DataFrame:
    cfg = load_config(config_path)
    sdir = split_dir(cfg, scenario, seed, fold)
    path = sdir / "mono_assignments.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_mono_table(
    scenario: str,
    *,
    seed: Optional[int] = None,
    fold: Optional[int] = None,
    allow_heldout: bool = False,
    config_path: str = "configs/dataset_primary_v1.yaml",
) -> MonoTable:
    """Load trainable monotherapy curves only.

    ``allow_heldout`` exists ONLY for audit code that needs to count sealed
    labels; it never returns labels to the training loop.  Any attempt to load
    labels for a non-trainable curve raises ``MonoAccessError``.
    """
    cfg = load_config(config_path)
    sdir = split_dir(cfg, scenario, seed, fold)
    assignments = pd.read_parquet(sdir / "mono_assignments.parquet")
    trainable = assignments[assignments["trainable"].astype(bool)]
    if len(trainable) == 0:
        raise MonoAccessError(f"{scenario} has no trainable monotherapy curves")
    if not allow_heldout and (assignments["label_access"] == "held_out").any():
        # Sealed labels may exist in the split, but they must not enter any tensor.
        sealed = int((assignments["label_access"] == "held_out").sum())
        if sealed == 0:
            raise MonoAccessError("inconsistent mono assignments")
    allowed = set(trainable["mono_curve_id"].astype(str))

    mono = pd.read_parquet(
        resolve(cfg, cfg["inputs"]["mono_points"]),
        columns=[
            "mono_curve_id",
            "mono_condition_id",
            "cell_id",
            "drug_id",
            "conc_um",
            "x_x0_raw",
            "row_idx",
        ],
    )
    mono = mono[mono["mono_curve_id"].astype(str).isin(allowed)].copy()
    if mono.empty:
        raise MonoAccessError("no monotherapy points for trainable curves")
    mono = mono.sort_values(["mono_curve_id", "row_idx"]).reset_index(drop=True)
    if (mono.groupby("mono_curve_id").size() != 8).any():
        bad = mono.groupby("mono_curve_id").size()
        raise ValueError(f"expected 8 points per curve, got {sorted(set(bad))}")
    if mono[["conc_um", "x_x0_raw"]].isna().any().any():
        raise ValueError("monotherapy labels/concentrations contain NaN")

    first = mono.groupby("mono_curve_id", sort=True).first().reset_index()
    curve_ids = first["mono_curve_id"].astype(str).tolist()
    y = mono["x_x0_raw"].to_numpy(dtype=np.float32).reshape(len(curve_ids), 8)
    conc = mono["conc_um"].to_numpy(dtype=np.float64).reshape(len(curve_ids), 8)
    zero = np.zeros_like(conc, dtype=np.float32)
    dose_std = np.zeros_like(conc, dtype=np.float32)
    scaler = joblib.load(sdir / "scalers.joblib")["mono_dose"]
    for i in range(len(curve_ids)):
        cols = concentration_columns(conc[i])
        zero[i] = cols["zero_flag"]
        dose_std[i] = transform_dose(cols["log10_molar"], cols["zero_flag"], scaler).astype(np.float32)

    pathway = pd.read_parquet(resolve(cfg, cfg["inputs"]["pathway_features"])).set_index("DepMap_ID")
    pscaler = joblib.load(sdir / "scalers.joblib")["pathway"]
    pmat = transform_standardizer(pathway[pscaler["columns"]].to_numpy(), pscaler).astype(np.float32)
    pos = {str(d): i for i, d in enumerate(pathway.index)}
    depmaps = _depmaps_for_cells(cfg, first["cell_id"].astype(str).tolist())
    cell_features = np.stack([pmat[pos[str(d)]] for d in depmaps], axis=0)

    return MonoTable(
        scenario=scenario,
        seed=seed,
        fold=fold,
        curve_ids=curve_ids,
        condition_ids=first["mono_condition_id"].astype(str).tolist(),
        cell_ids=first["cell_id"].astype(str).tolist(),
        depmap_ids=depmaps,
        drug_ids=first["drug_id"].astype(str).tolist(),
        dose_std=dose_std,
        zero=zero,
        cell_features=cell_features,
        y=y,
        policy=str(assignments["policy"].iloc[0]),
    )


def _depmaps_for_cells(cfg: Dict[str, Any], cell_ids: Sequence[str]) -> List[str]:
    mapping = pd.read_parquet(resolve(cfg, cfg["outputs"]["matrix_index"]), columns=["cell_id", "depmap_id"])
    m = mapping.drop_duplicates("cell_id").set_index("cell_id")["depmap_id"].to_dict()
    return [str(m[c]) for c in cell_ids]


def mono_batch(
    table: MonoTable,
    indices: Sequence[int],
    graphs: Dict[str, Any],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """One graph per curve; all curves use the `drug_a_graph` slot."""
    idx = np.asarray(list(indices), dtype=np.int64)
    xs, eis, eas, batch_vec, graph_idx = [], [], [], [], []
    offset = 0
    for local, gi in enumerate(idx):
        g = graphs[table.drug_ids[gi]]
        x = g.x.long()
        xs.append(x)
        eis.append(g.edge_index.long() + offset)
        eas.append(g.edge_attr.long())
        batch_vec.append(torch.full((x.shape[0],), local, dtype=torch.long))
        graph_idx.append(local)
        offset += x.shape[0]
    return {
        "drug_x": torch.cat(xs, dim=0).to(device),
        "drug_edge_index": torch.cat(eis, dim=1).to(device),
        "drug_edge_attr": torch.cat(eas, dim=0).to(device),
        "drug_batch": torch.cat(batch_vec, dim=0).to(device),
        "drug_a_graph": torch.tensor(graph_idx, dtype=torch.long, device=device),
        "cell_features": torch.from_numpy(table.cell_features[idx]).to(device),
        "dose_a": torch.from_numpy(table.dose_std[idx]).to(device),
        "dose_a_zero": torch.from_numpy(table.zero[idx]).to(device),
        "y": torch.from_numpy(table.y[idx]).to(device),
    }


def load_mono_labels_for_audit(
    scenario: str,
    curve_ids: Sequence[str],
    *,
    seed: Optional[int] = None,
    fold: Optional[int] = None,
    config_path: str = "configs/dataset_primary_v1.yaml",
) -> pd.DataFrame:
    """Return labels for *trainable* curves only; raise on any sealed curve.

    This is the explicit negative guard required by the M5 mono protocol: the
    training path filters silently, while audit code that names a held-out
    curve gets a hard error instead of a quiet empty frame.
    """
    assignments = load_mono_assignments(
        scenario, seed=seed, fold=fold, config_path=config_path
    )
    trainable = set(assignments.loc[assignments["trainable"].astype(bool), "mono_curve_id"].astype(str))
    requested = [str(c) for c in curve_ids]
    sealed = sorted(set(requested) - trainable)
    if sealed:
        raise MonoAccessError(
            f"refusing to read {len(sealed)} held-out monotherapy curve(s): {sealed[:5]}"
        )
    cfg = load_config(config_path)
    mono = pd.read_parquet(
        resolve(cfg, cfg["inputs"]["mono_points"]),
        columns=["mono_curve_id", "row_idx", "conc_um", "x_x0_raw"],
    )
    mono = mono[mono["mono_curve_id"].astype(str).isin(requested)]
    return mono.sort_values(["mono_curve_id", "row_idx"]).reset_index(drop=True)


def mono_access_summary(assignments: pd.DataFrame) -> Dict[str, Any]:
    return {
        "n_curves": int(len(assignments)),
        "n_trainable": int(assignments["trainable"].astype(bool).sum()),
        "n_held_out": int((~assignments["trainable"].astype(bool)).sum()),
        "policies": sorted(assignments["policy"].astype(str).unique().tolist()),
        "label_access": assignments["label_access"].value_counts().to_dict(),
    }
