"""M6-A diagnostic models: exact B3 reference and B3-like symmetric variants."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from .baseline_models import GINEBase
from .dose_encoder import DoseEncoder


class B3Exact(GINEBase):
    """Byte-for-byte M4 B3 architecture (GINEBase) with an explicit name."""

    model_name = "m6a_b3_exact"


class B3LikeSymmetric(nn.Module):
    """B3 trunk with a strictly symmetric pair head.

    ``dose_mode``:
      * ``scalar``  : raw standardized dose + zero flag only (T2)
      * ``fourier`` : scalar + Fourier dose embedding (T3)
      * ``film``    : Fourier embedding + per-drug FiLM before fusion (T4)
    """

    needs_graphs = True
    model_name = "m6a_b3sym"

    def __init__(
        self,
        *,
        cell_dim: int = 50,
        hidden: int = 128,
        layers: int = 3,
        dropout: float = 0.1,
        dose_mode: str = "scalar",
        dose_hidden: int = 64,
        fourier_bands: int = 4,
        decoder_hidden: int = 256,
    ) -> None:
        super().__init__()
        if dose_mode not in {"scalar", "fourier", "film"}:
            raise ValueError(f"unknown dose_mode {dose_mode}")
        self.dose_mode = dose_mode
        self.trunk = GINEBase(cell_dim=cell_dim, hidden=hidden, layers=layers, dropout=dropout)
        self.use_dose_embedding = dose_mode in {"fourier", "film"}
        if self.use_dose_embedding:
            self.dose_encoder = DoseEncoder(
                mode="fourier", out_dim=dose_hidden, hidden_dim=dose_hidden,
                fourier_bands=fourier_bands, dropout=dropout,
            )
            self.dose_proj = nn.Linear(dose_hidden, hidden)
        if dose_mode == "film":
            self.cell_proj = nn.Linear(cell_dim, hidden)
            self.film = nn.Linear(dose_hidden, 2 * hidden)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            self.interact = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
            )
            in_dim = 4 * hidden + 4  # ta aggregates(3H) + cell H + raw dose terms
        else:
            in_dim = 3 * hidden + cell_dim + 4
            if self.use_dose_embedding:
                in_dim += 3 * dose_hidden
        self.head = nn.Sequential(
            nn.Linear(in_dim, decoder_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, decoder_hidden // 2),
            nn.SiLU(),
            nn.Linear(decoder_hidden // 2, 1),
        )

    def encode_drugs(self, graphs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.trunk.encode_graphs(graphs)

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
            raise ValueError("B3LikeSymmetric requires graphs")
        if dose_a is not None or dose_b is not None:
            batch = dict(batch)
            if dose_a is not None:
                batch["dose_a"] = dose_a
            if dose_b is not None:
                batch["dose_b"] = dose_b
            if zero_a is not None:
                batch["dose_a_zero"] = zero_a
            if zero_b is not None:
                batch["dose_b_zero"] = zero_b
        node = self.encode_drugs(graphs)
        ha = node[graphs["drug_a_graph"]]
        hb = node[graphs["drug_b_graph"]]
        cell = batch["cell_features"]
        da, db = batch["dose_a"], batch["dose_b"]
        za, zb = batch["dose_a_zero"], batch["dose_b_zero"]
        if da.shape[1] == 4:
            ai = torch.arange(16, device=da.device) // 4
            bi = torch.arange(16, device=da.device) % 4
            da = da.index_select(1, ai)
            db = db.index_select(1, bi)
            za = za.index_select(1, ai)
            zb = zb.index_select(1, bi)
        b, n = da.shape
        h = ha.shape[-1]
        HA = ha.unsqueeze(1).expand(b, n, h)
        HB = hb.unsqueeze(1).expand(b, n, h)
        raw_dose = torch.stack([da + db, da * db, torch.abs(da - db), za + zb], dim=-1)

        if self.dose_mode == "film":
            ea = self.dose_encoder(da, za)
            eb = self.dose_encoder(db, zb)
            cell_h = self.cell_proj(cell)
            ua = self.interact(torch.cat([ha, cell_h], dim=-1)).unsqueeze(1).expand(b, n, h)
            ub = self.interact(torch.cat([hb, cell_h], dim=-1)).unsqueeze(1).expand(b, n, h)
            ga, ba = self.film(ea).chunk(2, dim=-1)
            gb, bb = self.film(eb).chunk(2, dim=-1)
            ta = (1.0 + ga) * ua + ba
            tb = (1.0 + gb) * ub + bb
            feats = [ta + tb, ta * tb, torch.abs(ta - tb),
                     cell_h.unsqueeze(1).expand(b, n, h), raw_dose]
        else:
            feats = [HA + HB, HA * HB, torch.abs(HA - HB),
                     cell.unsqueeze(1).expand(b, n, cell.shape[-1])]
            if self.use_dose_embedding:
                ea = self.dose_encoder(da, za)
                eb = self.dose_encoder(db, zb)
                feats += [ea + eb, ea * eb, torch.abs(ea - eb)]
            feats.append(raw_dose)
        return self.head(torch.cat(feats, dim=-1)).squeeze(-1)


def build_m6a_model(kind: str, *, cell_dim: int = 50, hidden: int = 128,
                    layers: int = 3, dropout: float = 0.1,
                    dose_hidden: int = 64, fourier_bands: int = 4) -> nn.Module:
    if kind == "b3_exact":
        return B3Exact(cell_dim=cell_dim, hidden=hidden, layers=layers, dropout=dropout)
    if kind in {"b3sym_scalar", "b3sym_fourier", "b3sym_film"}:
        mode = kind.split("_", 1)[1]
        return B3LikeSymmetric(cell_dim=cell_dim, hidden=hidden, layers=layers,
                               dropout=dropout, dose_mode=mode,
                               dose_hidden=dose_hidden, fourier_bands=fourier_bands)
    raise ValueError(f"unknown M6-A model kind {kind}")
