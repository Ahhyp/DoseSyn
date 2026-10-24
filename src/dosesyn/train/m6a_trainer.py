"""M6-A diagnostic trainer: isolated architecture comparisons on frozen splits."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..models.baseline_models import GINEBase, count_parameters
from ..models.dosesyn import DoseSynCore
from ..models.m6a_models import build_m6a_model
from ..training.m4_data import graph_batch, load_graphs, load_matrix_table, split_dir, table_batch
from ..training.metrics import compute_metrics
from ..training.reproducibility import (
    environment_record,
    save_json_atomic,
    save_torch_atomic,
    set_global_seed,
    utc_now,
)
from ..utils.config import load_config


def vocab_graph_batch(table, graphs, device):
    """One graph per unique drug (36), with per-matrix drug indices.

    Equivalent to duplicating each drug graph per matrix, but avoids O(B)
    python-side graph collation and repeated GINE forwards.
    """
    xs, eis, eas, batch_vec = [], [], [], []
    offset = 0
    for gi, drug_id in enumerate(table.drug_vocab):
        g = graphs[drug_id]
        x = g.x.long()
        xs.append(x)
        eis.append(g.edge_index.long() + offset)
        eas.append(g.edge_attr.long())
        batch_vec.append(torch.full((x.shape[0],), gi, dtype=torch.long))
        offset += x.shape[0]
    return {
        "drug_x": torch.cat(xs, dim=0).to(device),
        "drug_edge_index": torch.cat(eis, dim=1).to(device),
        "drug_edge_attr": torch.cat(eas, dim=0).to(device),
        "drug_batch": torch.cat(batch_vec, dim=0).to(device),
        "drug_a_graph": torch.as_tensor(table.drug_a_index, dtype=torch.long, device=device),
        "drug_b_graph": torch.as_tensor(table.drug_b_index, dtype=torch.long, device=device),
        "n_graphs": len(table.drug_vocab),
    }


def slice_graph_batch(cache, table, idx, device):
    g = dict(cache)
    g["drug_a_graph"] = torch.as_tensor(table.drug_a_index[idx], dtype=torch.long, device=device)
    g["drug_b_graph"] = torch.as_tensor(table.drug_b_index[idx], dtype=torch.long, device=device)
    return g


def build_model(name: str, cfg: Dict[str, Any]) -> nn.Module:
    m = cfg["model"]
    if name == "b3_exact":
        return build_m6a_model("b3_exact", cell_dim=50, hidden=int(m["hidden_dim"]),
                               layers=int(m["gine_layers"]), dropout=float(m["dropout"]))
    if name == "c0_control":
        return DoseSynCore(
            cell_dim=50, hidden_dim=int(m["hidden_dim"]), cell_hidden=int(m["cell_hidden"]),
            dose_hidden=int(m["dose_hidden"]), decoder_hidden=int(m["decoder_hidden"]),
            gine_layers=int(m["gine_layers"]), dropout=float(m["dropout"]),
            dose_encoding="scalar", fourier_bands=int(m["fourier_bands"]),
            symmetric_pair=False, film=False,
        )
    if name in {"b3sym_scalar", "b3sym_fourier", "b3sym_film"}:
        return build_m6a_model(name, cell_dim=50, hidden=int(m["hidden_dim"]),
                               layers=int(m["gine_layers"]), dropout=float(m["dropout"]),
                               dose_hidden=int(m["dose_hidden"]),
                               fourier_bands=int(m["fourier_bands"]))
    raise ValueError(f"unknown model {name}")


@torch.no_grad()
def predict_table(model: nn.Module, table, graphs, device, bs: int = 128) -> np.ndarray:
    model.eval()
    out = []
    g = vocab_graph_batch(table, graphs, device)
    for s in range(0, len(table), bs):
        idx = np.arange(s, min(s + bs, len(table)))
        b = table_batch(table, idx, device=device)
        gb = slice_graph_batch(g, table, idx, device)
        out.append(model(b, gb).detach().float().cpu().numpy().reshape(-1, 4, 4))
    return np.concatenate(out).astype(np.float64)


def _metrics(table, pred) -> Dict[str, Any]:
    return compute_metrics(table.y, pred, matrix_ids=table.matrix_ids,
                           condition_ids=table.condition_ids)


def run_m6a(
    *,
    model_name: str,
    scenario: str,
    seed: Optional[int],
    fold: Optional[int],
    cfg: Dict[str, Any],
    recipe: str = "m6a",
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
    device_name: str = "cuda",
    force: bool = False,
) -> Dict[str, Any]:
    split_id = f"fold_{fold}" if scenario == "s3_unseen_cell" else f"seed_{seed}"
    run_id = f"m6a__{model_name}__{scenario}__{split_id}"
    out = Path(cfg["outputs"]["runs_dir"]) / run_id
    if (out / "val_metrics.json").exists() and not force:
        return {"run_id": run_id, "status": "exists", "run_dir": str(out)}
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    t = cfg["training"]
    epochs = int(epochs or (80 if recipe == "m4" else t["epochs"]))
    patience = int(patience if patience is not None else (15 if recipe == "m4" else t["patience"]))
    set_global_seed(seed if seed is not None else 42)
    torch.set_num_threads(int(t["num_threads"]))

    train = load_matrix_table(scenario, "train", seed=seed, fold=fold)
    val = load_matrix_table(scenario, "val", seed=seed, fold=fold)
    graphs, _ = load_graphs(cfg["inputs"]["m4_config"])
    if device.type == "cuda":
        graphs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in graphs.items()}

    model = build_model(model_name, cfg).to(device)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=float(t["learning_rate"]),
                            weight_decay=float(t["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.HuberLoss(delta=float(t["huber_delta"]))
    rng = np.random.default_rng(seed if seed is not None else 42)
    bs = int(t["batch_size"])
    graph_cache = vocab_graph_batch(train, graphs, device)
    best, best_epoch, best_state = float("inf"), -1, None
    bad = 0
    log: List[Dict[str, Any]] = []
    start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        order = rng.permutation(len(train))
        for s in range(0, len(train), bs):
            idx = order[s:s + bs]
            b = table_batch(train, idx, device=device)
            g = slice_graph_batch(graph_cache, train, idx, device)
            opt.zero_grad(set_to_none=True)
            pred = model(b, g)
            loss = loss_fn(pred, b["y"].reshape(len(idx), 16))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(t["grad_clip"]))
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        vp = predict_table(model, val, graphs, device, int(t["eval_batch_size"]))
        m = _metrics(val, vp)
        score = float(m["matrix_macro_rmse"])
        select = score if recipe == "m6a" else float(m["point_rmse"])
        log.append({"epoch": ep, "train_loss": float(np.mean(losses)),
                    "val_matrix_macro_rmse": score, "val_point_rmse": float(m["point_rmse"]),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "epoch_seconds": round(time.time() - start, 1)})
        print(f"ep {ep:3d} loss {np.mean(losses):.5f} val_macro {score:.5f} "
              f"val_point {m['point_rmse']:.5f} lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        if select < best - 1e-6:
            best, best_epoch = select, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep >= int(t["min_epochs"]) and bad >= patience:
            break

    model.load_state_dict(best_state)
    vp = predict_table(model, val, graphs, device, int(t["eval_batch_size"]))
    m = _metrics(val, vp)
    m.update({"best_epoch": best_epoch, "epochs_run": len(log), "n_parameters": n_params,
              "selection_metric": "point_rmse" if recipe == "m4" else "matrix_macro_rmse"})
    save_json_atomic(cfg, out / "config.json")
    save_torch_atomic({"state_dict": model.state_dict(), "epoch": best_epoch,
                       "val_metrics": m, "model_name": model_name}, out / "best.pt")
    with open(out / "train_log.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)
    save_json_atomic(m, out / "val_metrics.json")
    save_json_atomic({
        "run_id": run_id, "model": model_name, "scenario": scenario, "split_id": split_id,
        "seed": seed, "fold": fold, "recipe": recipe, "epochs_cap": epochs,
        "patience": patience, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_peak_allocated_mb": (torch.cuda.max_memory_allocated() / 1e6
                                  if device.type == "cuda" else 0.0),
        "wall_seconds": round(time.time() - start, 1), "n_train": len(train),
        "n_val": len(val), "n_parameters": n_params, "environment": environment_record(),
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "split_scaler_sha256": json.loads(
            (split_dir(load_config(cfg["inputs"]["dataset_config"]), scenario, seed, fold)
             / "split_metadata.json").read_text()
        ).get("scaler_sha256"),
        "created_at": utc_now(),
    }, out / "run_metadata.json")
    return {"run_id": run_id, "status": "ok", "run_dir": str(out), "val": m,
            "n_parameters": n_params, "best_epoch": best_epoch}
