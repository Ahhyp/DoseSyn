"""M6-A diagnostics: parity, architecture isolation, symmetry and metric discipline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dosesyn.models.baseline_models import GINEBase, count_parameters
from dosesyn.models.dosesyn import DoseSynCore
from dosesyn.models.m6a_models import B3LikeSymmetric, build_m6a_model
from dosesyn.training.m4_data import graph_batch, table_batch
from dosesyn.training.metrics import compute_metrics

M4_B3_CKPT = Path("outputs/runs/m4/b3_gine__s1_random_condition__seed_42/best.pt")


def _predict(model, table, graphs, device="cpu"):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(table), 128):
            idx = np.arange(s, min(s + 128, len(table)))
            b = table_batch(table, idx, device=device)
            g = graph_batch(table, idx, graphs, device=device)
            out.append(model(b, g).cpu().numpy())
    return np.concatenate(out)


def test_m4_b3_checkpoint_reproduces_validation_metric(s1_val, m5_graphs):
    graphs, _ = m5_graphs
    payload = torch.load(M4_B3_CKPT, map_location="cpu")
    model = GINEBase(cell_dim=50, hidden=128, layers=3, dropout=0.1)
    model.load_state_dict(payload["model_state"])
    pred = _predict(model, s1_val, graphs).reshape(-1, 4, 4)
    metrics = compute_metrics(
        s1_val.y, pred, matrix_ids=s1_val.matrix_ids, condition_ids=s1_val.condition_ids
    )
    assert abs(metrics["matrix_macro_rmse"] - 0.08005502275233599) < 1e-6
    assert abs(metrics["point_rmse"] - 0.09502250508392601) < 1e-6
    assert metrics["n_matrices"] == 2791


def test_m4_b3_and_m5_c0_are_not_the_same_model():
    b3 = GINEBase(cell_dim=50, hidden=128, layers=3)
    c0 = DoseSynCore(
        symmetric_pair=False, dose_encoding="scalar", film=False,
        hidden_dim=128, cell_hidden=128, dose_hidden=64, decoder_hidden=256,
        gine_layers=3,
    )
    assert count_parameters(b3) == 284545
    assert count_parameters(c0) == 559299
    assert count_parameters(c0) > count_parameters(b3)


def test_b3sym_parameter_counts():
    expected = {"b3sym_scalar": 429954, "b3sym_fourier": 492290, "b3sym_film": 535682}
    for mode, n in expected.items():
        model = build_m6a_model(mode)
        assert count_parameters(model) == n, (mode, count_parameters(model))


def test_symmetric_models_have_zero_swap_error(s1_val, m5_graphs):
    graphs, _ = m5_graphs
    idx = np.arange(8)
    b = table_batch(s1_val, idx, device="cpu")
    g = graph_batch(s1_val, idx, graphs, device="cpu")
    for mode in ("scalar", "fourier", "film"):
        model = build_m6a_model(f"b3sym_{mode}").eval()
        with torch.no_grad():
            pred = model(b, g)
            swapped = DictSwap(s1_val, idx, graphs)
            pred_swap = model(swapped.batch, swapped.graphs)
        # F(d1,d2,x1,x2) must equal transpose of F(d2,d1,x2,x1)
        err = (pred.reshape(-1, 4, 4) - pred_swap.reshape(-1, 4, 4).transpose(1, 2)).abs().max().item()
        assert err < 1e-5, (mode, err)


class DictSwap:
    """Build swapped drug/dose batches by exchanging A and B roles."""

    def __init__(self, table, idx, graphs):
        b = table_batch(table, idx, device="cpu")
        g = graph_batch(table, idx, graphs, device="cpu")
        self.batch = dict(b)
        self.batch["dose_a"], self.batch["dose_b"] = b["dose_b"], b["dose_a"]
        self.batch["dose_a_zero"], self.batch["dose_b_zero"] = b["dose_b_zero"], b["dose_a_zero"]
        self.graphs = dict(g)
        self.graphs["drug_a_graph"], self.graphs["drug_b_graph"] = g["drug_b_graph"], g["drug_a_graph"]


def test_symmetric_zero_dose_is_finite(s1_val, m5_graphs):
    graphs, _ = m5_graphs
    idx = np.arange(4)
    b = table_batch(s1_val, idx, device="cpu")
    g = graph_batch(s1_val, idx, graphs, device="cpu")
    model = build_m6a_model("b3sym_fourier").eval()
    with torch.no_grad():
        out = model(b, g, dose_a=torch.zeros(4, 5), dose_b=torch.zeros(4, 5),
                    zero_a=torch.ones(4, 5), zero_b=torch.ones(4, 5))
    assert out.shape == (4, 5)
    assert torch.isfinite(out).all()


def test_early_stop_metric_is_matrix_macro():
    from dosesyn.train import m6a_trainer

    src = Path(m6a_trainer.__file__).read_text(encoding="utf-8")
    assert 'score = float(m["matrix_macro_rmse"])' in src
    assert "selection_metric" in src


def test_m3_frozen_hashes_still_match(m3_input_manifest):
    from dosesyn.training.reproducibility import file_sha256

    manifest = json.loads(Path("outputs/audit/m3_input_manifest.json").read_text(encoding="utf-8"))
    mismatches = []
    for rel, digest in manifest.get("input_hashes", {}).items():
        p = Path(rel)
        if p.exists() and file_sha256(p) != digest:
            mismatches.append(rel)
    assert not mismatches, mismatches
