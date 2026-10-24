"""M2.2: molecular graph cache validity and shared feature logic."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dosesyn.features.graph import ATOM_FEATURES, BOND_FEATURES, load_graphs
from dosesyn.utils.config import load_config


@pytest.fixture(scope="module")
def m2cfg():
    return load_config("configs/features_m2.yaml")


@pytest.fixture(scope="module")
def graphs(m2cfg):
    p = Path(m2cfg["outputs"]["drug_graphs"])
    if not p.exists():
        pytest.fail(f"missing {p}; run scripts/02_build_features.py --stage drugs")
    graphs, meta = load_graphs(p)
    return graphs, meta


def _entry(meta, name):
    return next(g for g in meta["graphs"].values() if g["oneil_drug_name"] == name)


def test_graph_cache_roundtrip(graphs):
    g, meta = graphs
    assert meta["n_graphs"] == 38
    assert len(g) == 38
    assert set(g) == set(meta["graphs"])
    for drug_id, data in g.items():
        assert data.x.shape[1] == len(ATOM_FEATURES), drug_id
        assert data.edge_index.shape[0] == 2
        assert data.edge_attr.shape[1] == len(BOND_FEATURES), drug_id
        assert data.x.shape[0] == data.num_nodes
        assert data.edge_index.shape[1] == data.edge_attr.shape[0]
        assert torch.isfinite(data.x.float()).all()
        assert torch.isfinite(data.edge_attr.float()).all()


def test_graphs_have_no_self_loops(graphs):
    g, _ = graphs
    for drug_id, data in g.items():
        src, dst = data.edge_index
        assert not (src == dst).any(), drug_id


def test_graphs_are_undirected(graphs):
    g, _ = graphs
    for drug_id, data in g.items():
        src, dst = data.edge_index.numpy()
        forward = set(zip(src.tolist(), dst.tolist()))
        backward = set(zip(dst.tolist(), src.tolist()))
        assert forward == backward, drug_id


def test_feature_ranges_match_spec(graphs):
    g, meta = graphs
    specs = {
        "atom": meta["atom_features"],
        "bond": meta["bond_features"],
    }
    for drug_id, data in g.items():
        for j, spec in enumerate(specs["atom"]):
            assert int(data.x[:, j].max()) < spec["size"], (drug_id, spec["name"])
            assert int(data.x[:, j].min()) >= 0
        for j, spec in enumerate(specs["bond"]):
            assert int(data.edge_attr[:, j].max()) < spec["size"]
            assert int(data.edge_attr[:, j].min()) >= 0


def test_metal_complexes_are_flagged(graphs):
    g, meta = graphs
    for name in ("Carboplatin", "Oxaliplatin"):
        entry = _entry(meta, name)
        assert entry["graph_status"] == "NEEDS_REVIEW"
        assert entry["n_components"] > 1
        assert g[entry["drug_id"]].num_nodes > 0


def test_all_other_drugs_have_ok_graphs(graphs):
    _, meta = graphs
    review = [d for d in meta["graphs"].values() if d["graph_status"] == "NEEDS_REVIEW"]
    assert {d["oneil_drug_name"] for d in review} == {"Carboplatin", "Oxaliplatin"}
    assert meta["n_ok"] == 36
    assert meta["n_failed"] == 0
