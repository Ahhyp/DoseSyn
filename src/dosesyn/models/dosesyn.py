"""DoseSyn-Core: GINE drug encoder + dose encoder + FiLM interaction +
strictly symmetric pair fusion + continuous surface decoder + mono head.

Symmetry argument
-----------------
For a pair of drugs ``(a, b)`` the decoder never sees the ordered pair
``(h_a, h_b)``.  It sees only permutation-invariant aggregates
``t_a+t_b``, ``t_a*t_b``, ``|t_a-t_b|`` and the cell state.  Swapping the two
drugs (and their dose grids) therefore maps the 4x4 output matrix to its
transpose, including at arbitrary continuous dose queries.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from .baseline_models import GINEBase
from .dose_encoder import DoseEncoder
from .mono_head import MonoHead


class DoseSynCore(nn.Module):
    needs_graphs = True
    model_name = "dosesyn_core"

    def __init__(
        self,
        *,
        cell_dim: int = 50,
        hidden_dim: int = 128,
        cell_hidden: int = 128,
        dose_hidden: int = 64,
        decoder_hidden: int = 256,
        gine_layers: int = 3,
        dropout: float = 0.1,
        dose_encoding: str = "fourier",
        fourier_bands: int = 4,
        symmetric_pair: bool = True,
        film: bool = True,
    ) -> None:
        super().__init__()
        self.symmetric_pair = bool(symmetric_pair)
        self.use_film = bool(film)
        self.hidden_dim = hidden_dim
        self.dose_encoding = dose_encoding

        self.drug_encoder = GINEBase(
            cell_dim=cell_dim, hidden=hidden_dim, layers=gine_layers, dropout=dropout
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_dim, cell_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(cell_hidden, hidden_dim),
        )
        self.dose_encoder = DoseEncoder(
            mode=dose_encoding,
            out_dim=dose_hidden,
            hidden_dim=dose_hidden,
            fourier_bands=fourier_bands,
            dropout=dropout,
        )
        self.dose_proj = nn.Linear(dose_hidden, hidden_dim)
        self.interact = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if self.use_film:
            self.film = nn.Linear(dose_hidden, 2 * hidden_dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

        if self.symmetric_pair:
            point_dim = 4 * hidden_dim
        else:
            point_dim = cell_dim + 2 * hidden_dim + 2 * dose_hidden
        self.decoder = nn.Sequential(
            nn.Linear(point_dim, decoder_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, decoder_hidden // 2),
            nn.SiLU(),
            nn.Linear(decoder_hidden // 2, 1),
        )
        self.mono_head = MonoHead(hidden_dim=hidden_dim, dose_dim=dose_hidden, dropout=dropout)

    # ------------------------------------------------------------------
    def encode_drugs(self, graphs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        node_vec = self.drug_encoder.encode_graphs(graphs)
        return node_vec[graphs["drug_a_graph"]], node_vec[graphs["drug_b_graph"]]

    def encode_cell(self, cell_features: torch.Tensor) -> torch.Tensor:
        return self.cell_encoder(cell_features)

    # ------------------------------------------------------------------
    @staticmethod
    def expand_grid(values: torch.Tensor, n_points: int = 16) -> torch.Tensor:
        """[B,4] grid -> [B,16] row-major point values, or pass [B,N] through."""
        if values.shape[1] == 4 and n_points == 16:
            ai = torch.arange(16, device=values.device) // 4
            return values.index_select(1, ai)
        return values

    def _point_doses(
        self,
        batch: Dict[str, torch.Tensor],
        dose_a: Optional[torch.Tensor],
        dose_b: Optional[torch.Tensor],
        zero_a: Optional[torch.Tensor],
        zero_b: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = 16 if dose_a is None else dose_a.shape[1]
        if dose_a is None:
            dose_a = self.expand_grid(batch["dose_a"], n)
        if dose_b is None:
            dose_b = self.expand_grid(batch["dose_b"], n)
        if zero_a is None:
            zero_a = self.expand_grid(batch["dose_a_zero"], n)
        if zero_b is None:
            zero_b = self.expand_grid(batch["dose_b_zero"], n)
        if dose_a.shape != dose_b.shape or dose_a.shape != zero_a.shape or dose_a.shape != zero_b.shape:
            raise ValueError("dose/zero tensors must share shape [B,N]")
        return dose_a, dose_b, zero_a, zero_b

    def surface_embeddings(
        self,
        graphs: Dict[str, torch.Tensor],
        cell_features: torch.Tensor,
        dose_a: torch.Tensor,
        dose_b: torch.Tensor,
        zero_a: torch.Tensor,
        zero_b: torch.Tensor,
    ) -> torch.Tensor:
        ha, hb = self.encode_drugs(graphs)
        cell_h = self.encode_cell(cell_features)
        ea = self.dose_encoder(dose_a, zero_a)
        eb = self.dose_encoder(dose_b, zero_b)
        if not self.symmetric_pair:
            b, n, _ = ea.shape
            # Ordered control: deliberately not symmetric, used only for C0.
            return torch.cat(
                [
                    cell_features.unsqueeze(1).expand(b, n, cell_features.shape[-1]),
                    ha.unsqueeze(1).expand(b, n, self.hidden_dim),
                    hb.unsqueeze(1).expand(b, n, self.hidden_dim),
                    ea,
                    eb,
                ],
                dim=-1,
            )
        ua = self.interact(torch.cat([ha, cell_h], dim=-1))
        ub = self.interact(torch.cat([hb, cell_h], dim=-1))
        if self.use_film:
            ga, ba = self.film(ea).chunk(2, dim=-1)
            gb, bb = self.film(eb).chunk(2, dim=-1)
            ta = (1.0 + ga) * ua.unsqueeze(1) + ba
            tb = (1.0 + gb) * ub.unsqueeze(1) + bb
        else:
            ta = ua.unsqueeze(1) + self.dose_proj(ea)
            tb = ub.unsqueeze(1) + self.dose_proj(eb)
        b, n, _ = ta.shape
        cell_ctx = cell_h.unsqueeze(1).expand(b, n, self.hidden_dim)
        return torch.cat([ta + tb, ta * tb, torch.abs(ta - tb), cell_ctx], dim=-1)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        graphs: Optional[Dict[str, torch.Tensor]] = None,
        *,
        dose_a: Optional[torch.Tensor] = None,
        dose_b: Optional[torch.Tensor] = None,
        zero_a: Optional[torch.Tensor] = None,
        zero_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if graphs is None:
            raise ValueError("DoseSynCore requires graph tensors")
        dose_a, dose_b, zero_a, zero_b = self._point_doses(batch, dose_a, dose_b, zero_a, zero_b)
        emb = self.surface_embeddings(
            graphs, batch["cell_features"], dose_a, dose_b, zero_a, zero_b
        )
        return self.decoder(emb).squeeze(-1)

    def predict_mono(
        self,
        drug_h: torch.Tensor,
        cell_h: torch.Tensor,
        dose_std: torch.Tensor,
        zero_flag: torch.Tensor,
    ) -> torch.Tensor:
        return self.mono_head(drug_h, cell_h, self.dose_encoder(dose_std, zero_flag))

    def forward_mono_batch(self, batch: Dict[str, torch.Tensor], graphs: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_vec = self.drug_encoder.encode_graphs(graphs)
        drug_h = node_vec[graphs["drug_a_graph"]]
        cell_h = self.encode_cell(batch["cell_features"])
        return self.predict_mono(drug_h, cell_h, batch["dose_a"], batch["dose_a_zero"])
