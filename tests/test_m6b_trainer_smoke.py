"""M6-B trainer loss composition tests on a tiny synthetic batch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.losses.pharmacology import boundary_loss, smoothness_loss  # noqa: E402
from dosesyn.models.m6b_models import M6BDoseSyn  # noqa: E402

B = 3


def _batch():
    return {
        "cell_features": torch.randn(B, 50),
        "dose_a": torch.randn(B, 4),
        "dose_b": torch.randn(B, 4),
        "dose_a_zero": torch.zeros(B, 4),
        "dose_b_zero": torch.zeros(B, 4),
    }


def _graphs():
    n_atoms = 6
    sizes = [20, 5, 2, 5, 3, 4, 2, 5, 6]
    x = torch.stack([torch.randint(0, n, (n_atoms,)) for n in sizes], dim=1)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])
    edge_attr = torch.stack([torch.randint(0, n, (10,)) for n in (3, 2, 2, 3)], dim=1)
    return {
        "drug_x": x, "drug_edge_index": edge_index, "drug_edge_attr": edge_attr,
        "drug_batch": torch.arange(3, dtype=torch.long).repeat_interleave(2),
        "drug_a_graph": torch.arange(B), "drug_b_graph": torch.arange(B),
    }


def test_t3_forward_and_symmetry():
    model = M6BDoseSyn(hidden=32, layers=1, decoder_hidden=64, dose_hidden=16, dropout=0.0)
    b = _batch()
    g = _graphs()
    out = model.forward_all(b, g)
    assert out["surface"].shape == (B, 16)
    # swap doses -> transposed 4x4 surfaces
    swapped = model.forward_all(b, g, dose_a=b["dose_b"], dose_b=b["dose_a"],
                                zero_a=b["dose_b_zero"], zero_b=b["dose_a_zero"])["surface"]
    a = out["surface"].view(B, 4, 4)
    s = swapped.view(B, 4, 4).transpose(1, 2)
    assert torch.max(torch.abs(a - s)).item() < 1e-5


def test_boundary_and_smooth_losses_backward_finite():
    model = M6BDoseSyn(hidden=32, layers=1, decoder_hidden=64, dose_hidden=16,
                       dropout=0.0, mono=True)
    b = _batch()
    out = model.forward_all(b, _graphs())
    surface = out["surface"].view(B, 4, 4)
    smooth = smoothness_loss(surface, b["dose_a"], b["dose_b"])
    # virtual zero boundary surfaces reuse the real mono head outputs
    a0 = surface  # placeholder shape check only
    lb, gap = boundary_loss(a0, surface, out["mono_a"].view(B, 4, 4), out["mono_b"].view(B, 4, 4))
    (smooth + lb).backward()
    assert np.isfinite(smooth.item()) and np.isfinite(lb.item())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_film_zero_init_and_scaled_output():
    model = M6BDoseSyn(hidden=32, layers=1, decoder_hidden=64, dose_hidden=16,
                       dropout=0.0, dose_mode="film", film_gamma_scale=0.5)
    # zero-initialised FiLM must start as identity modulation
    assert torch.allclose(model.film.weight, torch.zeros_like(model.film.weight))
    out = model.forward_all(_batch(), _graphs())
    assert torch.isfinite(out["surface"]).all()


def test_model_has_no_mono_head_when_disabled():
    model = M6BDoseSyn(hidden=32, layers=1, decoder_hidden=64, dose_hidden=16)
    assert model.mono_head is None
