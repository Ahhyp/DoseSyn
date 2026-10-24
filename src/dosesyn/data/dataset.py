"""PyTorch dataset / loader for DoseSyn M3 splits.

The loader is intentionally model-agnostic: it returns per-sample tensors for
the two molecular graphs, cell features, standardized dose coordinates, raw
X/X0 labels, and all traceability IDs.  No neural network is defined here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..features.transform_train_only import (
    concentration_columns,
    transform_dose,
    transform_standardizer,
)
from ..utils.config import load_config
from ..utils.io import read_json

LABEL_COLUMN = "x_x0_raw"
N_DOSES = 4


def _resolve(cfg: Dict[str, Any], value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


class DoseSynDataset(Dataset):
    """One item = one 4x4 combination matrix."""

    def __init__(
        self,
        dataset_config: str = "configs/dataset_primary_v1.yaml",
        scenario: str = "s1_random_condition",
        split: str = "train",
        *,
        seed: Optional[int] = None,
        fold: Optional[int] = None,
        label: Optional[str] = None,
        include_expression: bool = False,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train/val/test")
        self.cfg = load_config(dataset_config)
        self.scenario = scenario
        self.split = split
        self.include_expression = include_expression
        self.split_dir = self._split_dir(seed=seed, fold=fold, label=label)

        self.conditions = pd.read_parquet(self.split_dir / f"{split}_conditions.parquet")
        self.primary = pd.read_parquet(_resolve(self.cfg, self.cfg["outputs"]["matrix_index"]))
        self.primary = self.primary[self.primary["condition_id"].isin(set(self.conditions["condition_id"]))]
        if self.primary.empty:
            raise ValueError(f"{scenario}/{split} has no matrices")
        self.matrix_ids = sorted(self.primary["matrix_id"].unique())

        points = pd.read_parquet(
            _resolve(self.cfg, self.cfg["inputs"]["combo_points"]),
            columns=[
                "matrix_id",
                "row_idx",
                "col_idx",
                "conc_a_um",
                "conc_b_um",
                LABEL_COLUMN,
            ],
        )
        self.points = points[points["matrix_id"].isin(set(self.matrix_ids))].reset_index(drop=True)
        if len(self.points) != len(self.matrix_ids) * 16:
            raise ValueError("split points are not 16 per matrix")

        self.scalers = joblib.load(self.split_dir / "scalers.joblib")
        self.graphs, self.graph_meta = self._load_graphs()
        self.cell_vectors, self.cell_feature_names = self._cell_matrix()
        self.matrix_records = self._matrix_records()
        self.meta = read_json(self.split_dir / "split_metadata.json")

    # ------------------------------------------------------------------
    def _split_dir(self, *, seed: Optional[int], fold: Optional[int], label: Optional[str]) -> Path:
        if label:
            name = label
        elif self.scenario == "s3_unseen_cell":
            if fold is None:
                raise ValueError("s3 requires fold")
            name = f"fold_{int(fold)}"
        else:
            if seed is None:
                seed = int(self.cfg["split"]["seeds"][0])
            name = f"seed_{int(seed)}"
        path = _resolve(self.cfg, self.cfg["outputs"]["splits_dir"]) / self.scenario / name
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _load_graphs(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        payload = torch.load(
            _resolve(self.cfg, self.cfg["inputs"]["drug_graphs"]), map_location="cpu"
        )
        return payload["graphs"], payload.get("metadata", {})

    def _cell_matrix(self) -> Tuple[Dict[str, np.ndarray], List[str]]:
        pathway = pd.read_parquet(_resolve(self.cfg, self.cfg["inputs"]["pathway_features"]))
        pathway = pathway.set_index("DepMap_ID")
        cols = [c for c in pathway.columns]
        path_scaler = self.scalers["pathway"]
        missing = [c for c in path_scaler["columns"] if c not in cols]
        if missing:
            raise ValueError(f"pathway feature mismatch: {missing[:5]}")
        matrix = transform_standardizer(
            pathway[path_scaler["columns"]].to_numpy(), path_scaler
        )
        names = list(path_scaler["columns"])

        if self.include_expression:
            expr = pd.read_parquet(_resolve(self.cfg, self.cfg["inputs"]["cell_expression"]))
            expr = expr.set_index("depmap_id")
            expr_scaler = self.scalers["expression"]
            expr_matrix = transform_standardizer(
                expr[expr_scaler["columns"]].to_numpy(), expr_scaler
            )
            matrix = np.concatenate([matrix, expr_matrix], axis=1)
            names += list(expr_scaler["columns"])

        # Cells absent from the pathway table cannot reach this point (primary requires features).
        depmaps = pathway.index.tolist()
        pos = {str(d): i for i, d in enumerate(depmaps)}
        vectors = {}
        for d in self.conditions["depmap_id"].dropna().unique():
            if str(d) not in pos:
                raise ValueError(f"cell {d} lacks pathway features")
            vectors[str(d)] = matrix[pos[str(d)]].astype(np.float32)
        return vectors, names

    def _matrix_records(self) -> List[Dict[str, Any]]:
        recs = []
        primary = self.primary.set_index("matrix_id")
        records = primary.to_dict(orient="index")
        ordered = self.points.sort_values(["matrix_id", "row_idx", "col_idx"]).reset_index(drop=True)
        if len(ordered) % 16 != 0:
            raise ValueError("split points are not a multiple of 16")
        n = len(ordered) // 16
        mids = ordered["matrix_id"].to_numpy().reshape(n, 16)
        if not (mids == mids[:, :1]).all():
            raise ValueError("matrix rows are not contiguous after sorting")
        vals = ordered[["conc_a_um", "conc_b_um", LABEL_COLUMN]].to_numpy(dtype=np.float64).reshape(n, 16, 3)
        row_idx = ordered["row_idx"].to_numpy().reshape(n, 16)
        col_idx = ordered["col_idx"].to_numpy().reshape(n, 16)
        expect_r = np.tile(np.repeat(np.arange(4), 4), (n, 1))
        expect_c = np.tile(np.tile(np.arange(4), 4), (n, 1))
        if not (row_idx == expect_r).all() or not (col_idx == expect_c).all():
            raise ValueError("matrix rows are not in 4x4 row-major order")
        y_all = vals[:, :, 2].reshape(n, 4, 4)
        if not np.isfinite(y_all).all():
            raise ValueError("split contains missing labels")
        a_all = vals[:, 0::4, 0]
        b_all = vals[:, :4, 1]

        for i in range(n):
            mid = str(mids[i, 0])
            row = records[mid]
            a_um = a_all[i]
            b_um = b_all[i]
            ca = concentration_columns(a_um)
            cb = concentration_columns(b_um)
            recs.append(
                {
                    "matrix_id": mid,
                    "condition_id": str(row["condition_id"]),
                    "pair_id": str(row["pair_id"]),
                    "cell_id": str(row["cell_id"]),
                    "cell_line": str(row["cell_line"]),
                    "depmap_id": str(row["depmap_id"]),
                    "disease": None if pd.isna(row["disease"]) else str(row["disease"]),
                    "drug_a_id": str(row["drug_a_id"]),
                    "drug_b_id": str(row["drug_b_id"]),
                    "drug_a_name": str(row["drug_a_name"]),
                    "drug_b_name": str(row["drug_b_name"]),
                    "batch_id": int(row["batch_id"]),
                    "dose_a_um": a_um,
                    "dose_b_um": b_um,
                    "dose_a_std": transform_dose(ca["log10_molar"], ca["zero_flag"], self.scalers["drug_a_dose"]).astype(np.float32),
                    "dose_b_std": transform_dose(cb["log10_molar"], cb["zero_flag"], self.scalers["drug_b_dose"]).astype(np.float32),
                    "dose_a_zero": ca["zero_flag"].astype(np.float32),
                    "dose_b_zero": cb["zero_flag"].astype(np.float32),
                    "y": y_all[i],
                }
            )
        return recs

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.matrix_records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        r = self.matrix_records[index]
        ga = self.graphs[r["drug_a_id"]]
        gb = self.graphs[r["drug_b_id"]]
        return {
            "cell_features": torch.from_numpy(self.cell_vectors[r["depmap_id"]]),
            "drug_a_x": ga.x.float(),
            "drug_a_edge_index": ga.edge_index.long(),
            "drug_a_edge_attr": ga.edge_attr.float(),
            "drug_b_x": gb.x.float(),
            "drug_b_edge_index": gb.edge_index.long(),
            "drug_b_edge_attr": gb.edge_attr.float(),
            "dose_a": torch.from_numpy(r["dose_a_std"]),
            "dose_b": torch.from_numpy(r["dose_b_std"]),
            "dose_a_zero": torch.from_numpy(r["dose_a_zero"]),
            "dose_b_zero": torch.from_numpy(r["dose_b_zero"]),
            "dose_a_um": torch.from_numpy(r["dose_a_um"].astype(np.float32)),
            "dose_b_um": torch.from_numpy(r["dose_b_um"].astype(np.float32)),
            "y": torch.from_numpy(r["y"].astype(np.float32)),
            "batch_id": torch.tensor(r["batch_id"], dtype=torch.long),
            "matrix_id": r["matrix_id"],
            "condition_id": r["condition_id"],
            "pair_id": r["pair_id"],
            "cell_id": r["cell_id"],
            "cell_line": r["cell_line"],
            "depmap_id": r["depmap_id"],
            "disease": r["disease"],
            "drug_a_id": r["drug_a_id"],
            "drug_b_id": r["drug_b_id"],
            "drug_a_name": r["drug_a_name"],
            "drug_b_name": r["drug_b_name"],
        }

    def summary(self) -> Dict[str, Any]:
        y = np.concatenate([r["y"].ravel() for r in self.matrix_records])
        return {
            "scenario": self.scenario,
            "split": self.split,
            "split_dir": str(self.split_dir),
            "n_matrices": len(self),
            "n_points": int(y.size),
            "n_conditions": int(self.conditions["condition_id"].nunique()),
            "n_cells": int(self.conditions["cell_id"].nunique()),
            "n_drugs": int(len(set(self.primary["drug_a_id"]) | set(self.primary["drug_b_id"]))),
            "cell_feature_dim": len(self.cell_feature_names),
            "label": {
                "min": float(y.min()),
                "max": float(y.max()),
                "mean": float(y.mean()),
                "n_gt1": int((y > 1).sum()),
            },
            "mono_policy": self.meta.get("mono_policy"),
        }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------
def collate_matrices(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate molecular graphs and stack 4x4 matrices."""
    xs, eis, eas, graph_batch, roles = [], [], [], [], []
    offset = 0
    graph_id = 0
    for i, item in enumerate(batch):
        for role, xkey, eikey, eakey in (
            (0, "drug_a_x", "drug_a_edge_index", "drug_a_edge_attr"),
            (1, "drug_b_x", "drug_b_edge_index", "drug_b_edge_attr"),
        ):
            x = item[xkey]
            ei = item[eikey] + offset
            xs.append(x)
            eis.append(ei)
            eas.append(item[eakey])
            graph_batch.append(torch.full((x.shape[0],), graph_id, dtype=torch.long))
            roles.append(role)
            offset += x.shape[0]
            graph_id += 1

    out = {
        "cell_features": torch.stack([b["cell_features"] for b in batch]),
        "drug_x": torch.cat(xs, dim=0),
        "drug_edge_index": torch.cat(eis, dim=1),
        "drug_edge_attr": torch.cat(eas, dim=0),
        "drug_batch": torch.cat(graph_batch, dim=0),
        "drug_role": torch.tensor(roles, dtype=torch.long),
        "dose_a": torch.stack([b["dose_a"] for b in batch]),
        "dose_b": torch.stack([b["dose_b"] for b in batch]),
        "dose_a_zero": torch.stack([b["dose_a_zero"] for b in batch]),
        "dose_b_zero": torch.stack([b["dose_b_zero"] for b in batch]),
        "dose_a_um": torch.stack([b["dose_a_um"] for b in batch]),
        "dose_b_um": torch.stack([b["dose_b_um"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "batch_id": torch.stack([b["batch_id"] for b in batch]),
        "matrix_id": [b["matrix_id"] for b in batch],
        "condition_id": [b["condition_id"] for b in batch],
        "pair_id": [b["pair_id"] for b in batch],
        "cell_id": [b["cell_id"] for b in batch],
        "cell_line": [b["cell_line"] for b in batch],
        "depmap_id": [b["depmap_id"] for b in batch],
        "disease": [b["disease"] for b in batch],
        "drug_a_id": [b["drug_a_id"] for b in batch],
        "drug_b_id": [b["drug_b_id"] for b in batch],
        "drug_a_name": [b["drug_a_name"] for b in batch],
        "drug_b_name": [b["drug_b_name"] for b in batch],
    }
    return out


def make_loader(
    dataset: DoseSynDataset,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_matrices,
        pin_memory=False,
    )


def smoke_test(
    scenario: str = "s1_random_condition",
    seed: int = 13,
    fold: Optional[int] = None,
    batch_size: int = 4,
) -> Dict[str, Any]:
    """Materialize one real batch and check shapes/finiteness/label range."""
    ds = DoseSynDataset(scenario=scenario, split="train", seed=seed, fold=fold)
    loader = make_loader(ds, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    expected_cell_dim = len(ds.cell_feature_names)
    checks = {
        "cell_features_shape": tuple(batch["cell_features"].shape) == (min(batch_size, len(ds)), expected_cell_dim),
        "dose_a_shape": tuple(batch["dose_a"].shape) == (min(batch_size, len(ds)), 4),
        "dose_b_shape": tuple(batch["dose_b"].shape) == (min(batch_size, len(ds)), 4),
        "y_shape": tuple(batch["y"].shape) == (min(batch_size, len(ds)), 4, 4),
        "finite": bool(
            torch.isfinite(batch["cell_features"]).all()
            and torch.isfinite(batch["drug_x"]).all()
            and torch.isfinite(batch["dose_a"]).all()
            and torch.isfinite(batch["y"]).all()
        ),
        "labels_are_raw_x_x0": float(batch["y"].max()) <= 2.5 and float(batch["y"].min()) >= 0.0,
        "graphs_are_shaped": batch["drug_x"].shape[1] == 9 and batch["drug_edge_attr"].shape[1] == 4,
    }
    return {
        "dataset": ds.summary(),
        "batch": {
            "cell_features": list(batch["cell_features"].shape),
            "drug_x": list(batch["drug_x"].shape),
            "drug_edge_index": list(batch["drug_edge_index"].shape),
            "drug_edge_attr": list(batch["drug_edge_attr"].shape),
            "y": list(batch["y"].shape),
            "n_graphs": int(batch["drug_batch"].max()) + 1,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
