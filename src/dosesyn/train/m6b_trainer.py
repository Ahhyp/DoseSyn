"""M6-B trainer: T3 trunk + optional mono, boundary and smoothness terms.

All model selection uses validation matrix-macro RMSE.  Test labels are only
touched by ``run_formal`` after the checkpoint has been selected on validation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from dosesyn.losses.pharmacology import (
    boundary_loss,
    combo_matrix_loss,
    huber,
    mono_curve_loss,
    out_of_range_fraction,
    smoothness_loss,
)
from dosesyn.models.baseline_models import count_parameters
from dosesyn.models.m6b_models import build_m6b_model
from dosesyn.training.m4_data import (
    graph_batch,
    load_graphs,
    load_matrix_table,
    table_batch,
)
from dosesyn.train.m5_data import MonoTable, load_mono_table
from dosesyn.training.metrics import compute_metrics, json_safe
from dosesyn.training.runner import _save_predictions
from dosesyn.training.reproducibility import (
    environment_record,
    git_commit,
    save_json_atomic,
    save_torch_atomic,
    set_global_seed,
    utc_now,
)
from dosesyn.utils.config import load_config, resolve_path


# ---------------------------------------------------------------------------
# graph / mono helpers
# ---------------------------------------------------------------------------
def _graphs_for_ids(
    drug_ids: Sequence[str], graphs: Dict[str, Any], device: torch.device
) -> Dict[str, torch.Tensor]:
    xs, eis, eas, batch_vec = [], [], [], []
    offset = 0
    for local, drug_id in enumerate(drug_ids):
        g = graphs[drug_id]
        x = g.x.long()
        xs.append(x)
        eis.append(g.edge_index.long() + offset)
        eas.append(g.edge_attr.long())
        batch_vec.append(torch.full((x.shape[0],), local, dtype=torch.long))
        offset += x.shape[0]
    return {
        "drug_x": torch.cat(xs, dim=0).to(device),
        "drug_edge_index": torch.cat(eis, dim=1).to(device),
        "drug_edge_attr": torch.cat(eas, dim=0).to(device),
        "drug_batch": torch.cat(batch_vec, dim=0).to(device),
    }


def _mono_vocab(table: MonoTable, graphs: Dict[str, Any], device: torch.device):
    unique = sorted(set(table.drug_ids))
    index = {d: i for i, d in enumerate(unique)}
    local = np.asarray([index[d] for d in table.drug_ids], dtype=np.int64)
    return _graphs_for_ids(unique, graphs, device), local


def mono_vocab_batch(
    table: MonoTable,
    indices: Sequence[int],
    graphs: Dict[str, Any],
    local: np.ndarray,
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    idx = np.asarray(list(indices), dtype=np.int64)
    return {
        "cell_features": torch.as_tensor(table.cell_features[idx], dtype=torch.float32, device=device),
        "dose_a": torch.as_tensor(table.dose_std[idx], dtype=torch.float32, device=device),
        "dose_a_zero": torch.as_tensor(table.zero[idx], dtype=torch.float32, device=device),
        "y": torch.as_tensor(table.y[idx], dtype=torch.float32, device=device),
        "drug_local": torch.as_tensor(local[idx], dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------------------
# inference / metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(self_model, table, graphs, device, *, batch_size: int = 256):
    from dosesyn.train.m6a_trainer import vocab_graph_batch  # reuse cached 36-graph batching
    from dosesyn.train.m6a_trainer import slice_graph_batch

    preds, i = [], 0
    self_model.eval()
    while i < len(table.matrix_ids):
        idx = list(range(i, min(i + batch_size, len(table.matrix_ids))))
        b = table_batch(table, idx, device=device)
        full = vocab_graph_batch(table, graphs, device)
        g = slice_graph_batch(full, table, idx, device)
        out = self_model.forward_all(b, g)
        preds.append(out["surface"].detach().cpu().numpy())
        i += batch_size
    return np.concatenate(preds, axis=0)


def boundary_diagnostics(self_model, table, graphs, device, *, max_matrices: int = 1024) -> Dict[str, float]:
    from dosesyn.train.m6a_trainer import slice_graph_batch, vocab_graph_batch

    n = min(len(table.matrix_ids), max_matrices)
    idx = list(range(n))
    b = table_batch(table, idx, device=device)
    g = slice_graph_batch(vocab_graph_batch(table, graphs, device), table, idx, device)
    with torch.no_grad():
        base = self_model.forward_all(b, g)
        if "mono_a" not in base:
            return {"E_boundary_mean": float("nan")}
        zero_b = torch.ones_like(b["dose_b"])
        zero_a = torch.ones_like(b["dose_a"])
        a0 = self_model.forward_all(b, g, dose_b=torch.zeros_like(b["dose_b"]), zero_b=zero_b)
        b0 = self_model.forward_all(b, g, dose_a=torch.zeros_like(b["dose_a"]), zero_a=zero_a)
        e1 = (a0["surface"] - base["mono_a"]).abs()
        e2 = (b0["surface"] - base["mono_b"]).abs()
        return {
            "E_boundary_mean": float((0.5 * (e1 + e2)).mean().cpu()),
            "E_boundary_a0": float(e1.mean().cpu()),
            "E_boundary_0b": float(e2.mean().cpu()),
        }


def _swap_error(self_model, table, graphs, device, *, max_matrices: int = 512) -> Dict[str, float]:
    from dosesyn.train.m6a_trainer import slice_graph_batch, vocab_graph_batch

    n = min(len(table.matrix_ids), max_matrices)
    idx = list(range(n))
    b = table_batch(table, idx, device=device)
    g = slice_graph_batch(vocab_graph_batch(table, graphs, device), table, idx, device)
    with torch.no_grad():
        p = self_model.forward_all(b, g)["surface"]
        # swapped views: use the opposite drug graph for each slot
        hb = torch.arange(len(idx))
        node_g = {k: v for k, v in g.items()}
        node_g["drug_a_graph"], node_g["drug_b_graph"] = g["drug_b_graph"], g["drug_a_graph"]
        q = self_model.forward_all(
            {**b, "dose_a": b["dose_b"], "dose_b": b["dose_a"],
             "dose_a_zero": b["dose_b_zero"], "dose_b_zero": b["dose_a_zero"]}, node_g
        )["surface"].reshape(-1, 4, 4).transpose(1, 2).reshape(-1, 16)
        err = (p - q).abs().reshape(-1).cpu().numpy()
    return {"swap_max_abs": float(np.max(err)), "swap_mean_abs": float(np.mean(err))}


def _mono_legal_metrics(self_model, mono: Optional[MonoTable], graphs, device, local) -> Dict[str, Any]:
    if mono is None or self_model.mono_head is None:
        return {}
    preds, ys, n = [], [], len(mono.curve_ids)
    for i in range(0, n, 128):
        idx = list(range(i, min(i + 128, n)))
        mb = mono_vocab_batch(mono, idx, graphs, local, device=device)
        with torch.no_grad():
            p = self_model.forward_mono_vocab(mb, graphs)
        preds.append(p.cpu().numpy()); ys.append(mono.y[idx])
    p = np.concatenate(preds); y = np.concatenate(ys)
    return {
        "mono_rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mono_mae": float(np.mean(np.abs(p - y))),
        "mono_curves": int(n),
        "mono_points": int(p.size),
    }


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def _build_optimizer(model, cfg):
    tr = cfg["training"]
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(tr["learning_rate"]), weight_decay=float(tr["weight_decay"]),
    )


def run_m6b(
    variant_name: str,
    scenario: str,
    *,
    seed: Optional[int] = None,
    fold: Optional[int] = None,
    split_id: Optional[str] = None,
    device: str = "cuda",
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
    force: bool = False,
    eval_test: bool = False,
    m6b_config: str = "configs/m6b_experiments.yaml",
) -> Dict[str, Any]:
    cfg = load_config(m6b_config)
    variant = dict(cfg["variants"][variant_name])
    kind = str(variant["kind"])
    dev = torch.device(device)
    epochs = int(epochs or cfg["training"]["epochs"])
    patience = int(patience or cfg["training"]["patience"])
    split_id = split_id or (f"seed_{seed}" if seed is not None else f"fold_{fold}")
    run_id = f"m6b__{variant_name}__{scenario}__{split_id}"
    runs_dir = resolve_path(cfg, cfg["outputs"]["runs_dir"])
    out = runs_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_metadata.json").exists() and not force:
        return {"run_id": run_id, "status": "skip_exists"}

    set_global_seed(int(cfg["training"]["seed"]))
    graphs, _ = load_graphs(cfg["inputs"]["m4_config"])
    train = load_matrix_table(scenario, "train", seed=seed, fold=fold)
    val = load_matrix_table(scenario, "val", seed=seed, fold=fold)
    mono = None
    mono_local = None
    mono_graphs = None
    if bool(variant.get("mono", False)):
        mono = load_mono_table(scenario, seed=seed, fold=fold)
        mono_graphs, mono_local = _mono_vocab(mono, graphs, dev)

    from dosesyn.train.m6a_trainer import slice_graph_batch, vocab_graph_batch

    train_graph_full = vocab_graph_batch(train, graphs, dev)
    val_graph_full = vocab_graph_batch(val, graphs, dev)
    model = build_m6b_model(variant, cfg["model"]).to(dev)
    opt = _build_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    bs = int(cfg["training"]["batch_size"])
    lam_m = float(variant.get("mono_weight", 0.0))
    lam_b = float(variant.get("boundary_weight", 0.0))
    lam_s = float(variant.get("smooth_weight", 0.0))
    smooth_warm = int(variant.get("smooth_warmup_epochs", 0))

    best, best_ep, best_state, wait = float("inf"), 0, None, 0
    history: List[Dict[str, Any]] = []
    run_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        rng = np.random.default_rng(1000 * ep + seed if seed is not None else 1000 * ep)
        order = rng.permutation(len(train.matrix_ids))
        total, nb = 0.0, 0
        mono_iter = None
        if mono is not None and len(mono.curve_ids):
            mono_order = rng.permutation(len(mono.curve_ids))
            mono_iter = [mono_order[i:i + bs] for i in range(0, len(mono_order), bs)]
        for bi, start in enumerate(range(0, len(order), bs)):
            idx = order[start:start + bs].tolist()
            b = table_batch(train, idx, device=dev)
            g = slice_graph_batch(train_graph_full, train, idx, dev)
            o = model.forward_all(b, g)
            loss = combo_matrix_loss(o["surface"], b["y"].reshape(b["y"].shape[0], -1),
                                     delta=float(cfg["training"]["huber_delta"]), reduction="matrix_mean")
            if mono is not None and mono_iter is not None:
                mi = mono_iter[bi % len(mono_iter)].tolist()
                mb = mono_vocab_batch(mono, mi, graphs, mono_local, device=dev)
                mp = model.forward_mono_vocab(mb, mono_graphs)
                loss = loss + lam_m * mono_curve_loss(mp, mb["y"], delta=float(cfg["training"]["huber_delta"]))
            if lam_b > 0 and "mono_a" in o:
                zero_b = torch.ones_like(b["dose_b"])
                zero_a = torch.ones_like(b["dose_a"])
                a0 = model.forward_all(b, g, dose_b=torch.zeros_like(b["dose_b"]), zero_b=zero_b)
                b0 = model.forward_all(b, g, dose_a=torch.zeros_like(b["dose_a"]), zero_a=zero_a)
                lb, _ = boundary_loss(a0["surface"], b0["surface"], o["mono_a"], o["mono_b"])
                loss = loss + lam_b * lb
            if lam_s > 0 and ep > smooth_warm:
                loss = loss + lam_s * smoothness_loss(o["surface"].reshape(-1, 4, 4), b["dose_a"], b["dose_b"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip"]))
            opt.step()
            opt.zero_grad(set_to_none=True)
            total += float(loss.detach().cpu())
            nb += 1

        val_pred = predict(model, val, graphs, dev, batch_size=int(cfg["training"]["eval_batch_size"]))
        vm = compute_metrics(val.y, val_pred, matrix_ids=val.matrix_ids,
                             condition_ids=val.condition_ids)
        macro = float(vm["matrix_macro_rmse"])
        sched.step()
        rec = {"epoch": ep, "train_loss": total / max(nb, 1), "val_matrix_macro_rmse": macro,
               "val_point_rmse": float(vm["point_rmse"]), "lr": opt.param_groups[0]["lr"]}
        if mono is not None:
            mm = _mono_legal_metrics(model, mono, mono_graphs, dev, mono_local)
            rec.update({"mono_train_rmse": mm.get("mono_rmse")})
        history.append(rec)
        if macro < best - 1e-6:
            best, best_ep, wait = macro, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        print(f"[{run_id}] ep {ep:03d} train {rec['train_loss']:.5f} val_macro {macro:.5f} "
              f"best {best:.5f}@{best_ep} lr {rec['lr']:.2g}", flush=True)
        if ep >= int(cfg["training"]["min_epochs"]) and wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    wall = time.time() - run_start
    val_pred = predict(model, val, graphs, dev, batch_size=int(cfg["training"]["eval_batch_size"]))
    val_m = compute_metrics(val.y, val_pred, matrix_ids=val.matrix_ids,
                            condition_ids=val.condition_ids)
    diag = _swap_error(model, val, graphs, dev)
    diag.update(boundary_diagnostics(model, val, graphs, dev))
    diag.update(_mono_legal_metrics(model, mono, mono_graphs, dev, mono_local))
    diag["out_of_range_fraction"] = float(out_of_range_fraction(torch.from_numpy(val_pred)))
    if lam_s > 0:
        with torch.no_grad():
            b = table_batch(val, list(range(min(256, len(val.matrix_ids)))), device=dev)
            g = slice_graph_batch(val_graph_full, val, list(range(min(256, len(val.matrix_ids)))), dev)
            o = model.forward_all(b, g)
            from dosesyn.losses.pharmacology import smoothness_loss as _sl
            diag["smoothness_loss"] = float(_sl(o["surface"].reshape(-1, 4, 4), b["dose_a"], b["dose_b"]).cpu())

    result = {
        "run_id": run_id, "variant": variant_name, "scenario": scenario, "split_id": split_id,
        "best_epoch": best_ep, "epochs_run": len(history), "wall_seconds": wall,
        "n_parameters": int(count_parameters(model)), "n_train_matrices": len(train.matrix_ids),
        "n_val_matrices": len(val.matrix_ids), "best_val_macro": best,
        "val_metrics": json_safe(val_m), "diagnostics": json_safe(diag),
        "selection_metric": "matrix_macro_rmse",
    }
    save_torch_atomic({"state_dict": model.state_dict(), "meta": result,
                       "variant": variant, "config": cfg}, out / "best.pt")
    pd.DataFrame(history).to_csv(out / "train_log.csv", index=False)
    save_json_atomic(result, out / "val_metrics.json")
    save_json_atomic(result, out / "run_metadata.json")
    _save_predictions(out / "predictions_val.parquet", val, val_pred,
                      model=variant_name, scenario=scenario, seed_or_fold=split_id, split="val")
    resources = {"wall_seconds": wall, "host": environment_record().get("hostname"),
                 "gpu_peak_mb": float(torch.cuda.max_memory_allocated(dev) / 1024 ** 2) if dev.type == "cuda" else 0.0,
                 "n_parameters": int(count_parameters(model))}
    save_json_atomic(resources, out / "resources.json")

    if eval_test:
        test = load_matrix_table(scenario, "test", seed=seed, fold=fold)
        test_pred = predict(model, test, graphs, dev, batch_size=int(cfg["training"]["eval_batch_size"]))
        test_m = compute_metrics(test.y, test_pred, matrix_ids=test.matrix_ids,
                                 condition_ids=test.condition_ids)
        save_json_atomic(json_safe(test_m), out / "test_metrics.json")
        _save_predictions(out / "predictions_test.parquet", test, test_pred,
                          model=variant_name, scenario=scenario, seed_or_fold=split_id, split="test")
        result["test_metrics"] = json_safe(test_m)
    print(f"[{run_id}] DONE best {best:.5f}@{best_ep} wall {wall:.0f}s", flush=True)
    return result
