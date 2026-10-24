"""M4 baselines: Mean, MLP-Dose and GINE-Base (no M5 components)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ATOM_FEATURE_SIZES: List[int] = [119, 7, 2, 7, 5, 4, 2, 9, 7]
BOND_FEATURE_SIZES: List[int] = [5, 2, 2, 5]


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class MeanBaseline(nn.Module):
    """B0: constant train-set mean for every dose point."""

    needs_graphs = False
    needs_drug_ids = False
    model_name = "b0_mean"

    def __init__(self, mean: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(float(mean), dtype=torch.float32))

    def set_mean(self, value: float) -> None:
        self.mean.fill_(float(value))

    def forward(self, batch: Dict[str, torch.Tensor], graphs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        n = batch["dose_a"].shape[0]
        return self.mean.expand(n, 16)


class MLPDose(nn.Module):
    """B2: shared per-dose-point MLP conditioned on cell and drug embeddings."""

    needs_graphs = False
    needs_drug_ids = True
    model_name = "b2_mlp"

    def __init__(
        self,
        n_drugs: int,
        cell_dim: int = 50,
        drug_dim: int = 32,
        hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drug_embedding = nn.Embedding(n_drugs, drug_dim)
        in_dim = cell_dim + 2 * drug_dim + 2 + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor], graphs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        cell = batch["cell_features"]
        ea = self.drug_embedding(batch["drug_a_index"])
        eb = self.drug_embedding(batch["drug_b_index"])
        b = cell.shape[0]
        # index of the a-dose (row) and b-dose (column) used by each of the 16 points
        ai = torch.arange(16, device=cell.device) // 4
        bi = torch.arange(16, device=cell.device) % 4
        f = torch.cat(
            [
                cell.unsqueeze(1).expand(b, 16, cell.shape[1]),
                ea.unsqueeze(1).expand(b, 16, ea.shape[1]),
                eb.unsqueeze(1).expand(b, 16, eb.shape[1]),
                batch["dose_a"].index_select(1, ai).unsqueeze(-1),
                batch["dose_b"].index_select(1, bi).unsqueeze(-1),
                batch["dose_a_zero"].index_select(1, ai).unsqueeze(-1),
                batch["dose_b_zero"].index_select(1, bi).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(f).squeeze(-1)


class AtomEncoder(nn.Module):
    def __init__(self, sizes: List[int], hidden: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(s, hidden) for s in sizes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0
        for i, emb in enumerate(self.embeddings):
            out = out + emb(x[:, i])
        return out


class BondEncoder(nn.Module):
    def __init__(self, sizes: List[int], hidden: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(s, hidden) for s in sizes])

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        out = 0
        for i, emb in enumerate(self.embeddings):
            out = out + emb(edge_attr[:, i])
        return out


class GINEBase(nn.Module):
    """B3: GINE encoder for both drugs + cell features + dose-coordinate head."""

    needs_graphs = True
    needs_drug_ids = False
    model_name = "b3_gine"

    def __init__(
        self,
        cell_dim: int = 50,
        hidden: int = 128,
        layers: int = 3,
        dropout: float = 0.1,
        atom_sizes: Optional[List[int]] = None,
        bond_sizes: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GINEConv

        atom_sizes = atom_sizes or ATOM_FEATURE_SIZES
        bond_sizes = bond_sizes or BOND_FEATURE_SIZES
        self.atom_encoder = AtomEncoder(atom_sizes, hidden)
        self.bond_encoder = BondEncoder(bond_sizes, hidden)
        self.convs = nn.ModuleList(
            [
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.GELU(),
                        nn.Linear(hidden, hidden),
                    ),
                    edge_dim=hidden,
                )
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        point_dim = cell_dim + 2 * hidden + 4
        self.head = nn.Sequential(
            nn.Linear(point_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def encode_graphs(self, graphs: Dict[str, torch.Tensor]) -> torch.Tensor:
        from torch_geometric.nn import global_mean_pool

        x = self.atom_encoder(graphs["drug_x"])
        edge_attr = self.bond_encoder(graphs["drug_edge_attr"])
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, graphs["drug_edge_index"], edge_attr)
            h = F.gelu(norm(h))
            x = x + self.dropout(h)
        return global_mean_pool(x, graphs["drug_batch"])

    def forward(self, batch: Dict[str, torch.Tensor], graphs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        if graphs is None:
            raise ValueError("GINEBase requires graph tensors")
        node_vec = self.encode_graphs(graphs)  # [2B, hidden]
        ha = node_vec[graphs["drug_a_graph"]]
        hb = node_vec[graphs["drug_b_graph"]]
        cell = batch["cell_features"]
        b = cell.shape[0]
        ai = torch.arange(16, device=cell.device) // 4
        bi = torch.arange(16, device=cell.device) % 4
        f = torch.cat(
            [
                cell.unsqueeze(1).expand(b, 16, cell.shape[1]),
                ha.unsqueeze(1).expand(b, 16, ha.shape[1]),
                hb.unsqueeze(1).expand(b, 16, hb.shape[1]),
                batch["dose_a"].index_select(1, ai).unsqueeze(-1),
                batch["dose_b"].index_select(1, bi).unsqueeze(-1),
                batch["dose_a_zero"].index_select(1, ai).unsqueeze(-1),
                batch["dose_b_zero"].index_select(1, bi).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.head(f).squeeze(-1)


def build_dl_model(
    name: str,
    *,
    n_drugs: int,
    cell_dim: int,
    hidden: int,
    drug_dim: int,
    mlp_hidden: int,
    gine_layers: int,
    dropout: float,
    mean: float = 0.0,
) -> nn.Module:
    if name == "b0_mean":
        return MeanBaseline(mean)
    if name == "b2_mlp":
        return MLPDose(
            n_drugs=n_drugs,
            cell_dim=cell_dim,
            drug_dim=drug_dim,
            hidden=mlp_hidden,
            dropout=dropout,
        )
    if name == "b3_gine":
        return GINEBase(cell_dim=cell_dim, hidden=hidden, layers=gine_layers, dropout=dropout)
    raise ValueError(f"unknown DL model {name}")
