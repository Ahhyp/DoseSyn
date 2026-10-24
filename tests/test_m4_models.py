"""M4 model shape/category tests without training."""

from pathlib import Path

import numpy as np
import torch

from dosesyn.models.baseline_models import (
    ATOM_FEATURE_SIZES,
    BOND_FEATURE_SIZES,
    GINEBase,
    MLPDose,
    MeanBaseline,
    build_dl_model,
    count_parameters,
)
from dosesyn.training import m4_data


def _table(n=3, cell_dim=50):
    n_drugs = 4
    return m4_data.MatrixTable(
        scenario="s1_random_condition",
        seed=42,
        fold=None,
        split="train",
        split_dir=Path("."),
        matrix_ids=[f"m{i}" for i in range(n)],
        condition_ids=[f"c{i}" for i in range(n)],
        pair_ids=[f"p{i}" for i in range(n)],
        cell_ids=[f"cell{i}" for i in range(n)],
        cell_lines=[f"CL{i}" for i in range(n)],
        depmap_ids=[f"ACH{i}" for i in range(n)],
        batch_ids=np.ones(n, dtype=np.int64),
        drug_a_ids=["d0"] * n,
        drug_b_ids=["d1"] * n,
        drug_a_names=["A"] * n,
        drug_b_names=["B"] * n,
        drug_a_index=np.zeros(n, dtype=np.int64),
        drug_b_index=np.ones(n, dtype=np.int64),
        dose_a_std=np.linspace(-2, 1, 4).repeat(n).reshape(n, 4).astype(np.float32),
        dose_b_std=np.linspace(-1, 2, 4).repeat(n).reshape(n, 4).astype(np.float32),
        dose_a_zero=np.zeros((n, 4), dtype=np.float32),
        dose_b_zero=np.zeros((n, 4), dtype=np.float32),
        dose_a_um=np.tile(np.array([1e-4, 1e-2, 1.0, 100.0], dtype=np.float32), (n, 1)),
        dose_b_um=np.tile(np.array([1e-3, 0.1, 10.0, 200.0], dtype=np.float32), (n, 1)),
        cell_features=np.random.RandomState(0).randn(n, cell_dim).astype(np.float32),
        y=np.random.RandomState(1).rand(n, 4, 4).astype(np.float32) * 2,
        drug_vocab=["d0", "d1", "d2", "d3"],
        scaler_metadata={},
    )


def _graphs():
    from torch_geometric.data import Data

    g = {}
    for name in ("d0", "d1"):
        g[name] = Data(
            x=torch.tensor(
                [
                    [6, 3, 0, 3, 0, 0, 1, 6, 2],
                    [7, 3, 0, 3, 1, 0, 1, 6, 2],
                    [8, 3, 0, 3, 0, 0, 0, 0, 1],
                ],
                dtype=torch.long,
            ),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
            edge_attr=torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.long),
        )
    return g


def test_feature_category_sizes_are_consistent():
    assert len(ATOM_FEATURE_SIZES) == 9
    assert len(BOND_FEATURE_SIZES) == 4
    assert ATOM_FEATURE_SIZES[0] >= 119


def test_mean_baseline_shape():
    t = _table(2)
    b = m4_data.table_batch(t, [0, 1], device=torch.device("cpu"))
    m = MeanBaseline(0.7)
    out = m(b, None)
    assert out.shape == (2, 16)
    assert torch.allclose(out, torch.full((2, 16), 0.7))


def test_mlp_dose_shape_and_parameters():
    t = _table(3)
    b = m4_data.table_batch(t, [0, 1, 2], device=torch.device("cpu"))
    m = MLPDose(n_drugs=4, cell_dim=50)
    out = m(b, None)
    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()
    assert count_parameters(m) > 0


def test_gine_forward_shape_and_edges():
    t = _table(2)
    b = m4_data.table_batch(t, [0, 1], device=torch.device("cpu"))
    g = _graphs()
    gb = m4_data.graph_batch(t, [0, 1], g, device=torch.device("cpu"))
    assert gb["drug_x"].shape[1] == 9
    assert gb["drug_edge_attr"].shape[1] == 4
    assert gb["drug_a_graph"].shape == (2,)
    assert gb["n_graphs"] == 4
    m = GINEBase(cell_dim=50, hidden=32, layers=2)
    out = m(b, gb)
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_build_dl_model_names():
    assert isinstance(build_dl_model("b0_mean", n_drugs=3, cell_dim=50, hidden=16, drug_dim=8, mlp_hidden=32, gine_layers=1, dropout=0.0), MeanBaseline)
    assert isinstance(build_dl_model("b2_mlp", n_drugs=3, cell_dim=50, hidden=16, drug_dim=8, mlp_hidden=32, gine_layers=1, dropout=0.0), MLPDose)
