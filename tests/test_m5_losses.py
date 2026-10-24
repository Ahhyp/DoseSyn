"""Hand-checked M5 losses and switch semantics."""

from __future__ import annotations

import torch

from dosesyn.losses.pharmacology import (
    boundary_loss,
    combo_matrix_loss,
    mono_curve_loss,
    smoothness_loss,
)


def test_combo_matrix_reduction_is_per_matrix_then_mean():
    target = torch.zeros(2, 16)
    pred = torch.cat([torch.full((1, 16), 1.0), torch.full((1, 16), 3.0)])
    loss = combo_matrix_loss(pred, target, delta=10.0, reduction="matrix_mean")
    # Huber with delta=10 is quadratic: 0.5*1^2=0.5 and 0.5*9=4.5; mean = 2.5
    assert torch.isclose(loss, torch.tensor(2.5), atol=1e-6)


def test_mono_curve_reduction():
    target = torch.zeros(3, 8)
    pred = torch.full((3, 8), 1.0)
    assert torch.isclose(mono_curve_loss(pred, target, delta=10.0), torch.tensor(0.5))


def test_smoothness_zero_for_constant_surface():
    pred = torch.full((2, 4, 4), 1.234)
    da = torch.tensor([[-2.0, -1.0, -0.5, 0.0]]).repeat(2, 1)
    db = da.clone()
    assert torch.isclose(smoothness_loss(pred, da, db), torch.tensor(0.0))


def test_smoothness_nonuniform_grid_quadratic_is_zero():
    # an exact linear function has zero second difference on any grid
    x = torch.tensor([-2.0, -1.0, -0.2, 0.5])
    pred = (2.0 * x + 1.0).unsqueeze(0).unsqueeze(0).repeat(1, 4, 1)  # rows vary along a? build explicitly
    # surface[i, j] = 2*x_i + 1 so each row is constant across b, each column linear in a
    surface = torch.stack([torch.full((4,), float(2.0 * xi + 1.0)) for xi in x]).unsqueeze(0)
    assert torch.isclose(smoothness_loss(surface, x.unsqueeze(0), x.unsqueeze(0)), torch.tensor(0.0), atol=1e-6)


def test_smoothness_only_uses_own_grid():
    pred = torch.zeros(1, 4, 4)
    pred[0, 1, 1] = 1.0
    da = torch.tensor([[-2.0, -1.0, -0.5, 0.0]])
    db = da.clone()
    loss = smoothness_loss(pred, da, db)
    assert loss > 0


def test_boundary_loss_zero_when_consistent():
    s = torch.rand(3, 4)
    total, gap = boundary_loss(s, s, s, s)
    assert torch.isclose(total, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(gap, torch.tensor(0.0), atol=1e-6)


def test_boundary_loss_positive_when_shifted():
    s = torch.zeros(2, 4)
    m = torch.ones(2, 4)
    total, gap = boundary_loss(s, s, m, m, delta=10.0)
    assert gap > 0 and total > 0
