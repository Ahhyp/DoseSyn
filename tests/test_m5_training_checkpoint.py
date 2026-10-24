"""M5 training plumbing: finite gradients, checkpoint round-trip, frozen inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from dosesyn.models.dosesyn import DoseSynCore
from dosesyn.training.reproducibility import save_torch_atomic, utc_now
from dosesyn.utils.config import load_config, resolve_path


def test_all_loss_switches_backward(small_model, batch_pair):
    batch, g = batch_pair
    model = small_model.train()
    pred = model(batch, g)
    from dosesyn.losses.pharmacology import (
        boundary_loss,
        combo_matrix_loss,
        smoothness_loss,
    )

    loss = combo_matrix_loss(pred, batch["y"].reshape(pred.shape[0], 16), delta=1.0)
    surfaces = pred.reshape(-1, 4, 4)
    loss = loss + 0.01 * smoothness_loss(surfaces, batch["dose_a"], batch["dose_b"])
    ha, hb = model.encode_drugs(g)
    cell_h = model.encode_cell(batch["cell_features"])
    ma = model.predict_mono(ha, cell_h, batch["dose_a"], batch["dose_a_zero"])
    mb = model.predict_mono(hb, cell_h, batch["dose_b"], batch["dose_b_zero"])
    b, _ = boundary_loss(surfaces[:, 0, :], surfaces[:, :, 0], ma, mb)
    loss = loss + 0.02 * b
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g_ is not None and torch.isfinite(g_).all() for g_ in grads)


def test_checkpoint_roundtrip(small_model, batch_pair, tmp_path: Path):
    batch, g = batch_pair
    model = small_model.eval()
    with torch.no_grad():
        before = model(batch, g)
    path = tmp_path / "best.pt"
    save_torch_atomic({"state_dict": model.state_dict(), "epoch": 1}, path)
    payload = torch.load(path, map_location="cpu")
    clone = DoseSynCore(
        cell_dim=50,
        hidden_dim=32,
        cell_hidden=32,
        dose_hidden=16,
        decoder_hidden=64,
        gine_layers=1,
        dropout=0.0,
        dose_encoding="fourier",
        symmetric_pair=True,
        film=True,
    ).eval()
    clone.load_state_dict(payload["state_dict"])
    with torch.no_grad():
        after = clone(batch, g)
    assert torch.allclose(before, after, atol=1e-6)
    assert payload["epoch"] == 1


def test_m3_frozen_input_hashes_unchanged():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "outputs/audit/m3_input_manifest.json").read_text())
    for path, expected in manifest["input_hashes"].items():
        p = Path(path)
        if not p.is_absolute():
            p = root / p
        assert p.exists(), p
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == expected, f"frozen input changed: {path}"
