"""Continuous dose encoders.

All encoders receive the *standardized* log10 molar dose produced by the frozen
M3 train-only scaler plus an explicit ``zero_flag``.  A zero dose never goes
through ``log10``: its continuous part is replaced by 0 and the flag carries the
information.  The same encoder is shared by both drugs and by the monotherapy
head, so the two pathways cannot silently learn different dose semantics.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

DoseMode = Literal["scalar", "mlp", "fourier"]


def sanitize_dose(dose_std: torch.Tensor, zero_flag: torch.Tensor) -> torch.Tensor:
    """Replace non-finite / zero-dose entries by a safe 0 before any operation."""
    if dose_std.shape != zero_flag.shape:
        raise ValueError(f"dose shape {tuple(dose_std.shape)} != zero shape {tuple(zero_flag.shape)}")
    safe = torch.nan_to_num(dose_std.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return safe * (1.0 - zero_flag.float())


class DoseEncoder(nn.Module):
    def __init__(
        self,
        mode: DoseMode = "scalar",
        out_dim: int = 64,
        hidden_dim: int = 64,
        fourier_bands: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in ("scalar", "mlp", "fourier"):
            raise ValueError(f"unknown dose encoding: {mode}")
        self.mode = mode
        self.out_dim = out_dim
        self.fourier_bands = fourier_bands
        in_dim = 2
        if mode == "fourier":
            in_dim = 2 * fourier_bands + 2
        if mode == "scalar":
            # Still project to out_dim so the shared decoder/FiLM interfaces
            # see the same width for every dose_encoding switch.
            self.net = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

    def features(self, dose_std: torch.Tensor, zero_flag: torch.Tensor) -> torch.Tensor:
        zero = zero_flag.float()
        safe = sanitize_dose(dose_std, zero)
        if self.mode == "scalar":
            return torch.stack([safe, zero], dim=-1)
        parts = [safe.unsqueeze(-1), zero.unsqueeze(-1)]
        if self.mode == "fourier":
            for band in range(self.fourier_bands):
                freq = math.pi * (2.0 ** band)
                parts.append(torch.sin(freq * safe).unsqueeze(-1))
                parts.append(torch.cos(freq * safe).unsqueeze(-1))
        return torch.cat(parts, dim=-1)

    def forward(self, dose_std: torch.Tensor, zero_flag: torch.Tensor) -> torch.Tensor:
        feats = self.features(dose_std, zero_flag)
        if self.net is None:
            return feats
        return self.net(feats)
