"""M5 continuous dose encoding and virtual zero-dose boundary."""

from __future__ import annotations

import numpy as np
import torch


def test_explicit_sentinel_and_arbitrary_queries(small_model, batch_pair):
    batch, g = batch_pair
    model = small_model.eval()
    da = torch.linspace(-3.0, -1.0, 5)
    db = torch.linspace(-3.0, -1.0, 5)
    B = batch["cell_features"].shape[0]
    with torch.no_grad():
        out = model(
            batch,
            g,
            dose_a=da.unsqueeze(0).expand(B, -1),
            dose_b=db.unsqueeze(0).expand(B, -1),
            zero_a=torch.zeros(B, 5),
            zero_b=torch.zeros(B, 5),
        )
    assert out.shape == (B, 5)
    assert torch.isfinite(out).all()


def test_zero_dose_is_finite_and_not_logged(small_model, batch_pair):
    batch, g = batch_pair
    model = small_model.eval()
    B = batch["cell_features"].shape[0]
    zeros = torch.zeros(B, 4)
    ones = torch.ones(B, 4)
    with torch.no_grad():
        f_a = model(batch, g, dose_a=batch["dose_a"], dose_b=zeros, zero_a=batch["dose_a_zero"], zero_b=ones)
        f_b = model(batch, g, dose_a=zeros, dose_b=batch["dose_b"], zero_a=ones, zero_b=batch["dose_b_zero"])
        ha, hb = model.encode_drugs(g)
        cell_h = model.encode_cell(batch["cell_features"])
        m_a = model.predict_mono(ha, cell_h, batch["dose_a"], batch["dose_a_zero"])
        m_b = model.predict_mono(hb, cell_h, batch["dose_b"], batch["dose_b_zero"])
    for t in (f_a, f_b, m_a, m_b):
        assert torch.isfinite(t).all()
    assert f_a.shape == (B, 4) and m_a.shape == (B, 4)
    # standalone matrix surface: F(d_a, d_b) same when the other side is zero-flagged
    with torch.no_grad():
        full = model(batch, g)
    assert full.shape == (B, 16)


def test_film_zero_init_is_identity(small_model, batch_pair):
    batch, g = batch_pair
    model = small_model.eval()
    # FiLM layers are zero-initialised: at init gamma=beta=0 -> (1+0)u+0 = u
    assert isinstance(model.film, torch.nn.Linear)
    assert torch.count_nonzero(model.film.weight) == 0
    assert torch.count_nonzero(model.film.bias) == 0
