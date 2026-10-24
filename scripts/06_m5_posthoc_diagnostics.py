"""Recompute M5 boundary and monotherapy diagnostics from saved best checkpoints.

Runs launched before the diagnostic-logging update lack `E_boundary_mean` and
`mono_train_rmse` in `val_metrics.json`. This script recomputes those two
quantities on the validation split using the frozen M3 data and the saved
`best.pt`; it never trains, never touches test data, and only augments the
existing metrics JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.losses.pharmacology import boundary_loss  # noqa: E402
from dosesyn.train.m5_data import load_mono_table, mono_batch  # noqa: E402
from dosesyn.train.m5_trainer import (  # noqa: E402
    _batch_iter,
    build_model,
    model_boundary,
    predict_table,
)
from dosesyn.training.m4_data import graph_batch, load_graphs, load_matrix_table, table_batch  # noqa: E402
from dosesyn.utils.config import load_config  # noqa: E402
from dosesyn.utils.io import write_json  # noqa: E402


def process(run_dir: Path, cfg, graphs, device) -> dict:
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    val_metrics = json.loads((run_dir / "val_metrics.json").read_text(encoding="utf-8"))
    variant = meta["variant"]
    if not variant.get("symmetric_pair"):
        return {"run": run_dir.name, "skipped": "not symmetric"}
    if "E_boundary_mean" in val_metrics and "mono_train_rmse" in val_metrics:
        return {"run": run_dir.name, "skipped": "already present"}

    model = build_model(variant, cfg["model"])
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    scenario = meta["scenario"]
    split_id = meta["split_id"]
    seed = meta.get("seed")
    fold = meta.get("fold")
    val = load_matrix_table(
        scenario, "val", seed=seed, fold=fold,
        config_path=cfg["inputs"]["dataset_config"],
        m4_config_path=cfg["inputs"]["m4_config"],
    )
    gaps = []
    with torch.no_grad():
        for bidx in _batch_iter(len(val), min(256, len(val)), np.random.default_rng(0)):
            g = graph_batch(val, bidx, graphs, device=device)
            sa, sb, ma, mb = model_boundary(model, table_batch(val, bidx, device=device), g)
            gaps.append(float(boundary_loss(sa, sb, ma, mb, float(cfg["training"]["huber_delta"]))[1]))
    val_metrics["E_boundary_mean"] = float(np.mean(gaps)) if gaps else 0.0

    mono = load_mono_table(
        scenario, seed=seed, fold=fold, allow_heldout=False,
        config_path=cfg["inputs"]["dataset_config"],
    )
    if mono is not None and len(mono):
        midx = np.arange(min(len(mono), 1024))
        mb = mono_batch(mono, midx, graphs, device=device)
        with torch.no_grad():
            mp = model.forward_mono_batch(mb, mb)
            val_metrics["mono_train_rmse"] = float(
                torch.sqrt(torch.mean((mp - mb["y"]) ** 2)).detach().cpu()
            )
        val_metrics["mono_train_curves"] = len(mono)
    write_json(val_metrics, run_dir / "val_metrics.json")
    return {
        "run": run_dir.name,
        "E_boundary_mean": val_metrics.get("E_boundary_mean"),
        "mono_train_rmse": val_metrics.get("mono_train_rmse"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/m5_dosesyn.yaml")
    ap.add_argument("--runs", nargs="*", default=None, help="run dir names (default: c5/c6)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = load_config(args.config)
    runs_dir = Path(cfg["outputs"]["runs_dir"])
    graphs, _ = load_graphs(cfg["inputs"]["m4_config"])
    device = torch.device(args.device)
    targets = args.runs or [
        d.name for d in sorted(runs_dir.iterdir())
        if d.is_dir() and (
            d.name.endswith("__c5_boundary") or d.name.endswith("__c6_full")
        )
    ]
    out = []
    for name in targets:
        out.append(process(runs_dir / name, cfg, graphs, device))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
