"""M5 pharmacology losses: matrix-equal Huber, mono Huber, zero-dose boundary,
and non-uniform local surface smoothness.

No monotonicity or [0,1] projection is applied.  No loss uses a batch-level
average that would let a matrix with many points dominate.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=delta, reduction="none")


def combo_matrix_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
    reduction: str = "matrix_mean",
) -> torch.Tensor:
    """pred/target [B,16] -> scalar.  Each matrix contributes equally."""
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
    per_point = huber(pred, target, delta)
    per_matrix = per_point.reshape(pred.shape[0], -1).mean(dim=1)
    if reduction == "matrix_mean":
        return per_matrix.mean()
    if reduction == "sum":
        return per_matrix.sum()
    raise ValueError(f"unknown combo reduction: {reduction}")


def mono_curve_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
    reduction: str = "curve_mean",
) -> torch.Tensor:
    """pred/target [B,N] -> scalar; every curve contributes equally."""
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
    per_curve = huber(pred, target, delta).mean(dim=1)
    if reduction == "curve_mean":
        return per_curve.mean()
    if reduction == "sum":
        return per_curve.sum()
    raise ValueError(f"unknown mono reduction: {reduction}")


def _second_difference(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Non-uniform second difference along dim=1 of y [B,L]; x [B,L] positions."""
    y0, y1, y2 = y[:, :-2], y[:, 1:-1], y[:, 2:]
    x0, x1, x2 = x[:, :-2], x[:, 1:-1], x[:, 2:]
    d01 = x1 - x0
    d12 = x2 - x1
    d02 = x2 - x0
    safe = (d01.abs() > 1e-6) & (d12.abs() > 1e-6)
    denom = torch.where(safe, d01 * d12 * d02, torch.ones_like(d01))
    w0 = 2.0 * d12 / denom
    w1 = -2.0 * d02 / denom
    w2 = 2.0 * d01 / denom
    return w0.unsqueeze(-1) * y0 + w1.unsqueeze(-1) * y1 + w2.unsqueeze(-1) * y2


def smoothness_loss(
    pred: torch.Tensor,
    dose_a_std: torch.Tensor,
    dose_b_std: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean |second difference| along rows and columns of each 4x4 surface.

    pred: [B,4,4]; dose_*_std: [B,4] standardized log10 molar coordinates.
    Only the matrix's own grid is used; no cross-matrix neighbour and no
    assumption of equal spacing.
    """
    if pred.dim() != 3 or pred.shape[1:] != (4, 4):
        raise ValueError("pred must be [B,4,4]")
    if dose_a_std.shape != (pred.shape[0], 4) or dose_b_std.shape != (pred.shape[0], 4):
        raise ValueError("dose grids must be [B,4]")
    row = _second_difference(pred, dose_a_std)  # [B,2,4]
    col = _second_difference(pred.transpose(1, 2), dose_b_std).transpose(1, 2)  # [B,4,2]
    return row.abs().mean() + col.abs().mean()


def boundary_loss(
    surface_a: torch.Tensor,
    surface_b: torch.Tensor,
    mono_a: torch.Tensor,
    mono_b: torch.Tensor,
    delta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Virtual zero-dose consistency.

    surface_a: F(d_a, 0) [B,4]; mono_a: M(d_a) [B,4]; analogous for b.
    Returns (total, raw_mean_abs_difference).
    """
    la = huber(surface_a, mono_a, delta).mean()
    lb = huber(surface_b, mono_b, delta).mean()
    gap = (surface_a - mono_a).abs().mean() + (surface_b - mono_b).abs().mean()
    return 0.5 * (la + lb), 0.5 * gap


def out_of_range_fraction(pred: torch.Tensor, lo: float = 0.0, hi: float = 2.5) -> float:
    with torch.no_grad():
        return float(((pred < lo) | (pred > hi)).float().mean().item())
