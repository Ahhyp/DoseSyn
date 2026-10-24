"""M6-B model tests: symmetry, mono head, explicit zero-dose safety."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.models.m6b_models import build_m6b_model  # noqa: E402
from dosesyn.train.m6a_trainer import slice_graph_batch, vocab_graph_batch  # noqa: E402
from dosesyn.training.m4_data import load_graphs, load_matrix_table, table_batch  # noqa: E402
from dosesyn.utils.config import load_config  # noqa: E402

CFG = load_config("configs/m6b_experiments.yaml")
VARIANTS = [v for v in CFG["variants"] if v != "b0_b3_exact"]


@pytest.fixture(scope="module")
def data():
    graphs, _ = load_graphs("configs/m4_training.yaml")
    table = load_matrix_table("s1_random_condition", "val", seed=42)
    idx = list(range(6))
    b = table_batch(table, idx, device=torch.device("cpu"))
    full = vocab_graph_batch(table, graphs, torch.device("cpu"))
    g = slice_graph_batch(full, table, idx, torch.device("cpu"))
    return graphs, table, b, g


@pytest.mark.parametrize("name", VARIANTS)
def test_shape_and_finite(name, data):
    _, _, b, g = data
    model = build_m6b_model(CFG["variants"][name], CFG["model"]).eval()
    with torch.no_grad():
        out = model.forward_all(b, g)
    assert out["surface"].shape == (6, 16)
    assert torch.isfinite(out["surface"]).all()
    if name in {"b2_t3_mono", "b3_t3_mono_boundary", "b4_t3_mono_boundary_smooth", "b6_t3_film_reg_pharm"}:
        assert out["mono_a"].shape == (6, 16)
        assert torch.isfinite(out["mono_a"]).all()


@pytest.mark.parametrize("name", VARIANTS)
def test_permutation_symmetry(name, data):
    _, _, b, g = data
    model = build_m6b_model(CFG["variants"][name], CFG["model"]).eval()
    swapped = dict(g)
    swapped["drug_a_graph"], swapped["drug_b_graph"] = g["drug_b_graph"], g["drug_a_graph"]
    with torch.no_grad():
        p = model.forward_all(b, g)["surface"]
        q = model.forward_all(
            {**b, "dose_a": b["dose_b"], "dose_b": b["dose_a"],
             "dose_a_zero": b["dose_b_zero"], "dose_b_zero": b["dose_a_zero"]},
            swapped,
        )["surface"].reshape(-1, 4, 4).transpose(1, 2).reshape(-1, 16)
    max_err = float((p - q).abs().max())
    assert max_err < 1e-5, (name, max_err)


def test_film_is_identity_at_init(data):
    _, _, b, g = data
    model = build_m6b_model(CFG["variants"]["b5_t3_film_reg"], CFG["model"]).eval()
    # zero-initialized FiLM means the film branch starts from the unmodulated cell/drug state.
    assert torch.count_nonzero(model.film.weight) == 0
    assert torch.count_nonzero(model.film.bias) == 0
    with torch.no_grad():
        out = model.forward_all(b, g)["surface"]
    assert torch.isfinite(out).all()


def test_explicit_zero_dose_is_safe(data):
    _, _, b, g = data
    model = build_m6b_model(CFG["variants"]["b4_t3_mono_boundary_smooth"], CFG["model"]).eval()
    zeros = torch.zeros_like(b["dose_a"])
    ones = torch.ones_like(b["dose_a"])
    with torch.no_grad():
        a0 = model.forward_all(b, g, dose_b=zeros, zero_b=ones)["surface"]
        b0 = model.forward_all(b, g, dose_a=zeros, zero_a=ones)["surface"]
    assert torch.isfinite(a0).all() and torch.isfinite(b0).all()
    assert not torch.isnan(a0).any()
