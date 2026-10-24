"""M6-B models: T3 symmetric Fourier core plus optional mono/boundary/FiLM.

The M6-A ``B3LikeSymmetric`` class is deliberately left untouched; this module
extends the same trunk with an explicit shared monotherapy head and configurable
FiLM modulation strength.  No test label or split logic lives here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from .baseline_models import GINEBase, count_parameters
from .dose_encoder import DoseEncoder
from .mono_head import MonoHead


class M6BDoseSyn(nn.Module):
    """DoseSyn-Core candidate on the frozen M3 splits.

    ``dose_mode`` is ``fourier`` (T3) or ``film`` (T4/B5).  ``film=True`` is
    only used with ``dose_mode='film'``.  ``mono=True`` adds the shared
    monotherapy head; it does not change the combination surface head.
    """

    needs_graphs = True
    model_name = "m6b_dosesyn"

    def __init__(
        self,
        *,
        cell_dim: int = 50,
        hidden: int = 128,
        layers: int = 3,
        dropout: float = 0.1,
        dose_mode: str = "fourier",
        dose_hidden: int = 64,
        fourier_bands: int = 4,
        decoder_hidden: int = 256,
        mono: bool = False,
        mono_hidden: int = 128,
        film_gamma_scale: float = 1.0,
        film_beta_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if dose_mode not in {"fourier", "film"}:
            raise ValueError(f"unknown dose_mode {dose_mode}")
        self.dose_mode = dose_mode
        self.mono_enabled = bool(mono)
        self.film_gamma_scale = float(film_gamma_scale)
        self.film_beta_scale = float(film_beta_scale)

        self.trunk = GINEBase(cell_dim=cell_dim, hidden=hidden, layers=layers, dropout=dropout)
        self.cell_proj = nn.Linear(cell_dim, hidden)
        self.dose_encoder = DoseEncoder(
            mode="fourier", out_dim=dose_hidden, hidden_dim=dose_hidden,
            fourier_bands=fourier_bands, dropout=dropout,
        )
        self.dose_proj = nn.Linear(dose_hidden, hidden)
        self.use_film = dose_mode == "film"
        if self.use_film:
            self.film = nn.Linear(dose_hidden, 2 * hidden)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            self.interact = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
            )
            in_dim = 4 * hidden + 4
        else:
            in_dim = 3 * hidden + cell_dim + 3 * dose_hidden + 4
        self.head = nn.Sequential(
            nn.Linear(in_dim, decoder_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, decoder_hidden // 2),
            nn.SiLU(),
            nn.Linear(decoder_hidden // 2, 1),
        )
        self.mono_head = (
            MonoHead(hidden_dim=hidden, dose_dim=dose_hidden, dropout=dropout)
            if self.mono_enabled else None
        )

    # ------------------------------------------------------------------
    def encode_drugs(self, graphs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.trunk.encode_graphs(graphs)

    def _expand_grid(self, values: torch.Tensor, axis: str = "a", n: int = 16) -> torch.Tensor:
        if values.shape[1] == 4 and n == 16:
            base = torch.arange(16, device=values.device)
            idx = base // 4 if axis == "a" else base % 4
            return values.index_select(1, idx)
        return values

    def forward_all(
        self,
        batch: Dict[str, torch.Tensor],
        graphs: Dict[str, torch.Tensor],
        *,
        dose_a: Optional[torch.Tensor] = None,
        dose_b: Optional[torch.Tensor] = None,
        zero_a: Optional[torch.Tensor] = None,
        zero_b: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return surface plus optional mono predictions at the combo doses."""
        if graphs is None:
            raise ValueError("M6BDoseSyn requires graphs")
        node = self.encode_drugs(graphs)
        ha = node[graphs["drug_a_graph"]]
        hb = node[graphs["drug_b_graph"]]
        cell = batch["cell_features"]
        da = batch["dose_a"] if dose_a is None else dose_a
        db = batch["dose_b"] if dose_b is None else dose_b
        za = batch["dose_a_zero"] if zero_a is None else zero_a
        zb = batch["dose_b_zero"] if zero_b is None else zero_b
        da = self._expand_grid(da, "a")
        db = self._expand_grid(db, "b")
        za = self._expand_grid(za, "a")
        zb = self._expand_grid(zb, "b")
        b, n = da.shape
        h = ha.shape[-1]
        HA = ha.unsqueeze(1).expand(b, n, h)
        HB = hb.unsqueeze(1).expand(b, n, h)
        cell_h = self.cell_proj(cell)
        raw_dose = torch.stack([da + db, da * db, torch.abs(da - db), za + zb], dim=-1)
        ea = self.dose_encoder(da, za)
        eb = self.dose_encoder(db, zb)

        if self.use_film:
            ua = self.interact(torch.cat([ha, cell_h], dim=-1)).unsqueeze(1).expand(b, n, h)
            ub = self.interact(torch.cat([hb, cell_h], dim=-1)).unsqueeze(1).expand(b, n, h)
            ga, ba = self.film(ea).chunk(2, dim=-1)
            gb, bb = self.film(eb).chunk(2, dim=-1)
            ga = ga * self.film_gamma_scale
            gb = gb * self.film_gamma_scale
            ba = ba * self.film_beta_scale
            bb = bb * self.film_beta_scale
            ta = (1.0 + ga) * ua + ba
            tb = (1.0 + gb) * ub + bb
            feats = [ta + tb, ta * tb, torch.abs(ta - tb),
                     cell_h.unsqueeze(1).expand(b, n, h), raw_dose]
        else:
            feats = [HA + HB, HA * HB, torch.abs(HA - HB),
                     cell.unsqueeze(1).expand(b, n, cell.shape[-1]),
                     ea + eb, ea * eb, torch.abs(ea - eb), raw_dose]
        surface = self.head(torch.cat(feats, dim=-1)).squeeze(-1)

        out = {"surface": surface, "ha": ha, "hb": hb, "cell_h": cell_h,
               "dose_a": da, "dose_b": db, "zero_a": za, "zero_b": zb,
               "ea": ea, "eb": eb}
        if self.mono_head is not None:
            out["mono_a"] = self.mono_head(ha, cell_h, ea).squeeze(-1)
            out["mono_b"] = self.mono_head(hb, cell_h, eb).squeeze(-1)
        return out

    def forward(self, batch, graphs=None, **kwargs) -> torch.Tensor:
        return self.forward_all(batch, graphs, **kwargs)["surface"]

    def forward_mono_vocab(self, batch: Dict[str, torch.Tensor], graphs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mono_head is None:
            raise RuntimeError("monotherapy head is disabled")
        node = self.encode_drugs(graphs)
        h = node[batch["drug_local"]]
        cell_h = self.cell_proj(batch["cell_features"])
        e = self.dose_encoder(batch["dose_a"], batch["dose_a_zero"])
        return self.mono_head(h, cell_h, e).squeeze(-1)

    def forward_mono_batch(self, batch: Dict[str, torch.Tensor], graphs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mono_head is None:
            raise RuntimeError("monotherapy head is disabled")
        node = self.encode_drugs(graphs)
        h = node[graphs["drug_a_graph"]]
        cell_h = self.cell_proj(batch["cell_features"])
        e = self.dose_encoder(batch["dose_a"], batch["dose_a_zero"])
        return self.mono_head(h, cell_h, e).squeeze(-1)


def build_m6b_model(variant: Dict[str, Any], model_cfg: Dict[str, Any]) -> nn.Module:
    kind = str(variant["kind"])
    if kind == "b3_exact":
        from .m6a_models import B3Exact
        return B3Exact(
            cell_dim=50, hidden=int(model_cfg["hidden_dim"]),
            layers=int(model_cfg["gine_layers"]), dropout=float(model_cfg["dropout"]),
        )
    if kind not in {"b3sym_fourier", "b3sym_film"}:
        raise ValueError(f"unknown M6-B model kind {kind}")
    return M6BDoseSyn(
        cell_dim=50,
        hidden=int(model_cfg["hidden_dim"]),
        layers=int(model_cfg["gine_layers"]),
        dropout=float(variant.get("dropout", model_cfg["dropout"])),
        dose_mode="film" if kind == "b3sym_film" else "fourier",
        dose_hidden=int(model_cfg["dose_hidden"]),
        fourier_bands=int(model_cfg["fourier_bands"]),
        decoder_hidden=int(model_cfg["decoder_hidden"]),
        mono=bool(variant.get("mono", False)),
        mono_hidden=int(model_cfg.get("mono_hidden", 128)),
        film_gamma_scale=float(variant.get("film_gamma_scale", model_cfg.get("film_gamma_scale", 1.0))),
        film_beta_scale=float(variant.get("film_beta_scale", model_cfg.get("film_beta_scale", 1.0))),
    )


__all__ = ["M6BDoseSyn", "build_m6b_model", "count_parameters"]
