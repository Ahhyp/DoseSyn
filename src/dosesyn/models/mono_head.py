"""Shared monotherapy head M(d, x, c).

The head is deliberately separate from the combination surface decoder but
shares the *same* drug encoder and dose encoder.  That is what makes the
virtual zero-dose boundary loss meaningful: both paths interpret a zero dose
through the identical sentinel representation.
"""

from __future__ import annotations

import torch
from torch import nn


class MonoHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, dose_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim + dose_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        drug_h: torch.Tensor,
        cell_h: torch.Tensor,
        dose_e: torch.Tensor,
    ) -> torch.Tensor:
        """drug_h/cell_h: [B,H]; dose_e: [B,N,De] -> [B,N]."""
        if dose_e.dim() != 3:
            raise ValueError("dose_e must be [B,N,De]")
        b, n, _ = dose_e.shape
        d = drug_h.unsqueeze(1).expand(b, n, drug_h.shape[-1])
        c = cell_h.unsqueeze(1).expand(b, n, cell_h.shape[-1])
        return self.net(torch.cat([d, c, dose_e], dim=-1)).squeeze(-1)
