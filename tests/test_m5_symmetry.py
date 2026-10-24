"""M5 symmetry: strict drug-pair exchange symmetry and 4x4 transpose identity."""

from __future__ import annotations

import torch

from dosesyn.models.dosesyn import DoseSynCore


def _swap(batch, g):
    sb = dict(batch)
    sb["dose_a"], sb["dose_b"] = batch["dose_b"], batch["dose_a"]
    sb["dose_a_zero"], sb["dose_b_zero"] = batch["dose_b_zero"], batch["dose_a_zero"]
    sg = dict(g)
    sg["drug_a_graph"], sg["drug_b_graph"] = g["drug_b_graph"], g["drug_a_graph"]
    return sb, sg


def test_symmetric_model_single_matrix(small_model, batch_pair):
    batch, g = batch_pair
    model = small_model.eval()
    with torch.no_grad():
        pred = model(batch, g)
        sb, sg = _swap(batch, g)
        swapped = model(sb, sg)
    assert pred.shape == (8, 16)
    pred_t = pred.reshape(-1, 4, 4).transpose(1, 2).reshape(-1, 16)
    assert torch.allclose(pred_t, swapped, atol=1e-5, rtol=1e-5)


def test_symmetric_model_batch(small_model, batch_pair):
    batch, g = batch_pair
    model = small_model.eval()
    with torch.no_grad():
        pred = model(batch, g)
        sb, sg = _swap(batch, g)
        swapped = model(sb, sg)
    pred_t = pred.reshape(-1, 4, 4).transpose(1, 2).reshape(-1, 16)
    assert (pred_t - swapped).abs().max().item() < 1e-5


def test_ordered_control_is_a_separate_model(batch_pair):
    """C0 must be built with symmetric_pair=False; its decoder input differs."""
    batch, g = batch_pair
    torch.manual_seed(1)
    control = DoseSynCore(
        cell_dim=50,
        hidden_dim=32,
        cell_hidden=32,
        dose_hidden=16,
        decoder_hidden=64,
        gine_layers=1,
        dropout=0.0,
        dose_encoding="fourier",
        symmetric_pair=False,
        film=True,
    ).eval()
    with torch.no_grad():
        pred = control(batch, g)
        sb, sg = _swap(batch, g)
        swapped = control(sb, sg)
    pred_t = pred.reshape(-1, 4, 4).transpose(1, 2).reshape(-1, 16)
    diff = (pred_t - swapped).abs().max().item()
    # no symmetry claim: the control is explicitly order-dependent
    assert diff > 0.0
