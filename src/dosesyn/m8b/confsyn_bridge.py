"""Restore ConfSyn's frozen DrugCombDB checkpoints and score conditions.

This is intentionally an *inference-only* adapter. It reconstructs the exact
entity index, homogeneous graph, HGT view, gene-expression matrix, and precomputed
ChemBERTa embeddings used by ConfSyn training, loads each fold checkpoint with
strict=False (the frozen RoBERTa weights are not stored in the checkpoint), and
returns the fusion-head positive-class probability.

The adapter never trains and never touches ConfSyn files.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class _DrugSemanticStub(nn.Module):
    """Key-compatible replacement for DrugSemanticEncoder.

    ConfSyn's checkpoints store only ``drug_semantic_enc.proj.*`` because the
    RoBERTa encoder is frozen.  Inference here uses precomputed ChemBERTa
    embeddings, so the 125M-parameter BERT body is unnecessary.
    """

    def __init__(self, proj_dim: int = 256, **kwargs):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(768, proj_dim * 2),
            nn.GELU(),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, *args, **kwargs):  # pragma: no cover - never called
        raise RuntimeError("drug semantic encoder is bypassed via precomputed embeddings")


class ConfSynBridge:
    def __init__(self, confsyn_root: str | Path, folds: Sequence[int] = (0, 1, 2, 3, 4), device: str = "cpu"):
        self.root = Path(confsyn_root).resolve()
        self.folds = list(folds)
        self.device = torch.device(device)
        self.logger = logging.getLogger("dosesyn.m8b.confsyn")

        os.chdir(self.root)
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))

        import Model.confsyn_architecture as arch
        from Utils.data_prep import load_smiles_and_genes
        from Utils.graph_constrct import get_graph
        from Utils.hgt_utils import NODE_TYPES, convert_homogeneous_to_hgt

        arch.DrugSemanticEncoder = _DrugSemanticStub
        self.arch = arch
        self.NODE_TYPES = NODE_TYPES

        (self.data, self.entity_mapping, self.synergy, self.node_counts,
         self.reg, self.reg_mean, self.reg_std) = get_graph("DrugCombDB")
        self.smiles, self.gene_matrix = load_smiles_and_genes(self.entity_mapping)
        self.x_dict, self.edge_index_dict = convert_homogeneous_to_hgt(
            self.data, self.node_counts, verbose=False
        )
        self.embeddings = torch.load(self.root / "cache" / "chemberta_emb_proj256.pt", map_location="cpu")

        # global index maps
        self.cell_index = {k: v for k, v in self.entity_mapping.items()
                           if isinstance(k, str) and not k.startswith("DB") and k not in ("", None)}
        self.drug_index = {k: v for k, v in self.entity_mapping.items()
                           if isinstance(k, str) and k.startswith("DB")}
        self._models: dict[int, nn.Module] = {}

    # ------------------------------------------------------------------
    def checkpoint_path(self, fold: int) -> Path:
        return self.root / (
            "result/paper_reproduction/2026-09-10_15-44-09_19269.mgt/random_cv/"
            f"weight_DrugCombDB/cv_DrugCombDB_mode_1_fold_num_{fold}.pth"
        )

    def _build_model(self) -> nn.Module:
        model = self.arch.UncertaintyAwareSynergyNet(
            in_feats=300,
            hid_feats=600,
            out_feats=300,
            logger=self.logger,
            num_relations=6,
            proj_dim=256,
            hgt_dropout=0.2,
            hgt_layers=2,
            hgt_heads=4,
            hgt_out_channels=768,
            cell_dropout=0.2,
            cell_hidden_dim=1024,
            use_cell_ae_decoder=False,
            node_types=self.NODE_TYPES,
            edge_types=list(self.edge_index_dict.keys()),
            node_feat_dims={"cell": 300, "drug": 300, "protein": 300},
            fusion_type="gate",
            use_pathway=False,
            use_cond_readout=False,
            use_graph_anchor=False,
            drug_encoder_unfreeze_last_n=0,
            lambda_entropy=1.0,
        )
        return model.to(self.device)

    def load_fold(self, fold: int) -> nn.Module:
        if fold in self._models:
            return self._models[fold]
        path = self.checkpoint_path(fold)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu")
        state = payload["model_state_dict"] if "model_state_dict" in payload else payload
        model = self._build_model()
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Only the frozen BERT body may be absent; every task parameter must load.
        bad_missing = [k for k in missing if not k.startswith("drug_semantic_enc.bert.")]
        bad_unexpected = [k for k in unexpected if not k.startswith("drug_semantic_enc.bert.")]
        if bad_missing or bad_unexpected:
            raise RuntimeError(f"checkpoint mismatch fold {fold}: missing={bad_missing[:5]} unexpected={bad_unexpected[:5]}")
        model.eval()
        self._models[fold] = model
        return model

    # ------------------------------------------------------------------
    def _tensors(self, rows: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """rows: cell_line, drug_a_name, drug_b_name, db_a, db_b."""
        cell_idx = [self.cell_index[c] for c in rows["conf_cell_name"]]
        g1 = [self.drug_index[d] for d in rows["conf_db_a"]]
        g2 = [self.drug_index[d] for d in rows["conf_db_b"]]
        e1 = torch.stack([self.embeddings[int(g)] for g in g1])
        e2 = torch.stack([self.embeddings[int(g)] for g in g2])
        cf = torch.from_numpy(self.gene_matrix[np.asarray(cell_idx, dtype=int)]).float()
        index = torch.tensor(np.column_stack([g1, g2, cell_idx]), dtype=torch.long)
        return e1, e2, cf, index, torch.tensor(g1)

    @torch.no_grad()
    def score_rows(self, rows: pd.DataFrame, fold: int) -> np.ndarray:
        model = self.load_fold(fold)
        e1, e2, cf, index, _ = self._tensors(rows)
        e1, e2, cf, index = e1.to(self.device), e2.to(self.device), cf.to(self.device), index.to(self.device)
        out = model(
            graph=self.data,
            index=index,
            drug1_ids=None,
            drug1_mask=None,
            drug2_ids=None,
            drug2_mask=None,
            drug1_emb=e1,
            drug2_emb=e2,
            cline_features=cf,
            label=None,
            x_dict=self.x_dict,
            edge_index_dict=self.edge_index_dict,
            node_counts=self.node_counts,
        )
        return torch.softmax(out["logits_fusion"], dim=1)[:, 1].cpu().numpy()

    def score_conditions(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Score each condition with every fold; returns mean/std/min/max."""
        rows = rows.copy()
        for fold in self.folds:
            rows[f"p_fold{fold}"] = self.score_rows(rows, fold)
        cols = [f"p_fold{f}" for f in self.folds]
        rows["p_mean"] = rows[cols].mean(axis=1)
        rows["p_std"] = rows[cols].std(axis=1, ddof=0)
        rows["p_min"] = rows[cols].min(axis=1)
        rows["p_max"] = rows[cols].max(axis=1)
        return rows

    # ------------------------------------------------------------------
    def sanity_check(self, n: int = 2000, seed: int = 0) -> pd.DataFrame:
        """Label-reconstruction check on random DrugCombDB rows, all folds.

        These rows are not an independent test set (the official split indices
        are not persisted); the numbers demonstrate that the checkpoint and
        preprocessing were restored consistently, not generalization.
        """
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.synergy), size=min(n, len(self.synergy)), replace=False)
        raw = self.synergy[idx].astype(int)
        labels = raw[:, 3]
        from sklearn.metrics import roc_auc_score

        frame = pd.DataFrame({"g1": raw[:, 0], "g2": raw[:, 1], "cell": raw[:, 2], "label": labels})
        # build mapping-free batch directly (indices are already global)
        e1 = torch.stack([self.embeddings[int(g)] for g in frame["g1"]])
        e2 = torch.stack([self.embeddings[int(g)] for g in frame["g2"]])
        cf = torch.from_numpy(self.gene_matrix[frame["cell"].to_numpy()]).float()
        index = torch.tensor(frame[["g1", "g2", "cell"]].to_numpy(), dtype=torch.long)
        out = []
        for fold in self.folds:
            model = self.load_fold(fold)
            with torch.no_grad():
                logits = model(
                    graph=self.data, index=index, drug1_ids=None, drug1_mask=None,
                    drug2_ids=None, drug2_mask=None, drug1_emb=e1, drug2_emb=e2,
                    cline_features=cf, label=None, x_dict=self.x_dict,
                    edge_index_dict=self.edge_index_dict, node_counts=self.node_counts,
                )["logits_fusion"]
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            out.append({
                "fold": fold,
                "n": int(len(labels)),
                "positive_rate": float(labels.mean()),
                "auc": float(roc_auc_score(labels, p)),
                "accuracy": float(((p > 0.5) == labels).mean()),
                "mean_p_positive": float(p[labels == 1].mean()),
                "mean_p_negative": float(p[labels == 0].mean()),
                "checkpoint": str(self.checkpoint_path(fold).relative_to(self.root)),
            })
        return pd.DataFrame(out)
