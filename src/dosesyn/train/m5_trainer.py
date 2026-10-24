"""M5 trainer: one run id, one frozen M3 split, validation-only model selection."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..losses.pharmacology import (
    boundary_loss,
    combo_matrix_loss,
    mono_curve_loss,
    out_of_range_fraction,
    smoothness_loss,
)
from ..models.dosesyn import DoseSynCore
from ..training.m4_data import (
    MatrixTable,
    graph_batch,
    load_graphs,
    load_matrix_table,
    split_dir,
    table_batch,
)
from ..training.metrics import build_prediction_frame, compute_metrics, json_safe
from ..training.reproducibility import (
    environment_record,
    save_json_atomic,
    save_torch_atomic,
    set_global_seed,
    utc_now,
)
from ..utils.config import load_config, resolve_path
from .m5_data import MonoTable, load_mono_table, mono_access_summary, mono_batch


def build_model(variant: Dict[str, Any], model_cfg: Dict[str, Any]) -> DoseSynCore:
    return DoseSynCore(
        cell_dim=50,
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        cell_hidden=int(model_cfg.get("cell_hidden", 128)),
        dose_hidden=int(model_cfg.get("dose_hidden", 64)),
        decoder_hidden=int(model_cfg.get("decoder_hidden", 256)),
        gine_layers=int(model_cfg.get("gine_layers", 3)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        dose_encoding=str(variant.get("dose_encoding", "fourier")),
        fourier_bands=int(model_cfg.get("fourier_bands", 4)),
        symmetric_pair=bool(variant.get("symmetric_pair", True)),
        film=bool(variant.get("film", True)),
    )


def _batch_iter(n: int, batch_size: int, rng: np.random.Generator) -> List[np.ndarray]:
    order = rng.permutation(n)
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


@torch.no_grad()
def predict_table(
    model: nn.Module,
    table: MatrixTable,
    graphs: Dict[str, Any],
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    outs = []
    for start in range(0, len(table), batch_size):
        idx = np.arange(start, min(start + batch_size, len(table)))
        batch = table_batch(table, idx, device=device)
        g = graph_batch(table, idx, graphs, device=device)
        outs.append(model(batch, g).detach().cpu())
    return torch.cat(outs, dim=0).numpy()


@torch.no_grad()
def diagnostic_metrics(
    model: DoseSynCore,
    table: MatrixTable,
    graphs: Dict[str, Any],
    device: torch.device,
    *,
    n: int = 256,
) -> Dict[str, float]:
    idx = np.arange(min(n, len(table)))
    batch = table_batch(table, idx, device=device)
    g = graph_batch(table, idx, graphs, device=device)
    model.eval()
    pred = model(batch, g)
    swapped = dict(g)
    swapped["drug_a_graph"], swapped["drug_b_graph"] = g["drug_b_graph"], g["drug_a_graph"]
    sb = dict(batch)
    sb["dose_a"], sb["dose_b"] = batch["dose_b"], batch["dose_a"]
    sb["dose_a_zero"], sb["dose_b_zero"] = batch["dose_b_zero"], batch["dose_a_zero"]
    pred_swap = model(sb, swapped)
    pred_t = pred.reshape(-1, 4, 4).transpose(1, 2).reshape(-1, 16)
    e_swap = (pred - pred_swap).abs()
    out = {
        "E_swap_mean": float(e_swap.mean().item()),
        "E_swap_max": float(e_swap.max().item()),
        "out_of_range_fraction": out_of_range_fraction(pred),
    }
    # boundary gap and smoothness on the same batch (surface only; no gradients)
    surfaces = pred.reshape(-1, 4, 4)
    if model.symmetric_pair:
        out["R_smooth"] = float(
            smoothness_loss(surfaces, batch["dose_a"], batch["dose_b"]).item()
        )
    else:
        out["R_smooth"] = float("nan")
    return out


def run_m5(
    *,
    scenario: str,
    seed: Optional[int],
    fold: Optional[int],
    variant_name: str,
    variant: Dict[str, Any],
    cfg: Dict[str, Any],
    device: str = "cuda",
    epochs: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    split_id = f"fold_{fold}" if scenario == "s3_unseen_cell" else f"seed_{seed}"
    run_id = f"m5__{scenario}__{split_id}__{variant_name}"
    runs_dir = Path(cfg["outputs"]["runs_dir"])
    run_dir = runs_dir / run_id
    if run_dir.exists() and (run_dir / "val_metrics.json").exists() and not force:
        return {"run_id": run_id, "status": "exists", "run_dir": str(run_dir)}

    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    set_global_seed(int(seed if seed is not None else (fold or 0) + 1000))
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(int(train_cfg.get("num_threads", 4)))

    train = load_matrix_table(
        scenario, "train", seed=seed, fold=fold, config_path=cfg["inputs"]["dataset_config"]
    )
    val = load_matrix_table(
        scenario, "val", seed=seed, fold=fold, config_path=cfg["inputs"]["dataset_config"]
    )
    graphs, graph_meta = load_graphs(cfg["inputs"]["m4_config"])
    mono_weight = float(variant.get("mono_weight", 0.0))
    boundary_weight = float(variant.get("boundary_weight", 0.0))
    smooth_weight = float(variant.get("smooth_weight", 0.0))
    mono: Optional[MonoTable] = None
    mono_summary = {}
    if mono_weight > 0:
        mono = load_mono_table(
            scenario, seed=seed, fold=fold, config_path=cfg["inputs"]["dataset_config"]
        )
        mono_summary = mono_access_summary(
            pd.read_parquet(split_dir(
                load_config(cfg["inputs"]["dataset_config"]), scenario, seed, fold
            ) / "mono_assignments.parquet")
        )

    model = build_model(variant, cfg["model"]).to(dev)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    max_epochs = int(epochs if epochs is not None else train_cfg["epochs"])
    patience = int(train_cfg["patience"])
    min_epochs = int(train_cfg["min_epochs"])
    delta = float(train_cfg["huber_delta"])
    batch_size = int(train_cfg["batch_size"])
    rng = np.random.default_rng(int(seed if seed is not None else (fold or 0) + 1000))
    mono_rng = np.random.default_rng(int(seed if seed is not None else (fold or 0) + 2000))
    best = float("inf")
    best_epoch = -1
    best_state = None
    log_rows: List[Dict[str, Any]] = []
    run_start = time.time()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: List[float] = []
        mono_losses: List[float] = []
        boundary_losses: List[float] = []
        smooth_losses: List[float] = []
        for bidx in _batch_iter(len(train), batch_size, rng):
            batch = table_batch(train, bidx, device=dev)
            g = graph_batch(train, bidx, graphs, device=dev)
            pred = model(batch, g)
            loss = float(loss_cfg["combo_weight"]) * combo_matrix_loss(
                pred, batch["y"].reshape(pred.shape[0], 16), delta, loss_cfg["combo_reduction"]
            )
            if mono is not None and mono_weight > 0:
                midx = mono_rng.integers(0, len(mono), size=min(batch_size, len(mono)))
                mb = mono_batch(mono, midx, graphs, device=dev)
                mp = model.forward_mono_batch(mb, mb)
                ml = mono_curve_loss(mp, mb["y"], delta, loss_cfg["mono_reduction"])
                loss = loss + mono_weight * ml
                mono_losses.append(float(ml.detach()))
            if boundary_weight > 0 and model.symmetric_pair:
                sa, sb, ma, mby = model_boundary(model, batch, g)
                bl, _ = boundary_loss(sa, sb, ma, mby, delta)
                loss = loss + boundary_weight * bl
                boundary_losses.append(float(bl.detach()))
            if smooth_weight > 0 and model.symmetric_pair:
                surf = pred.reshape(-1, 4, 4)
                sl = smoothness_loss(surf, batch["dose_a"], batch["dose_b"], cfg["model"]["smooth_eps"])
                loss = loss + smooth_weight * sl
                smooth_losses.append(float(sl.detach()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            opt.step()
            losses.append(float(loss.detach()))
        val_pred = predict_table(model, val, graphs, dev, int(train_cfg["eval_batch_size"]))
        val_m = compute_metrics(
            val.y.reshape(len(val), 16),
            val_pred.reshape(len(val), 16),
            matrix_ids=val.matrix_ids,
            condition_ids=val.condition_ids,
        )
        val_score = float(val_m["matrix_macro_rmse"])
        sched.step(val_score)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "mono_loss": float(np.mean(mono_losses)) if mono_losses else "",
                "boundary_loss": float(np.mean(boundary_losses)) if boundary_losses else "",
                "smooth_loss": float(np.mean(smooth_losses)) if smooth_losses else "",
                "val_matrix_macro_rmse": val_score,
                "val_point_rmse": float(val_m["point_rmse"]),
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )
        if val_score < best:
            best = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch >= min_epochs and epoch - best_epoch >= patience:
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    val_pred = predict_table(model, val, graphs, dev, int(train_cfg["eval_batch_size"]))
    val_metrics = compute_metrics(
        val.y.reshape(len(val), 16),
        val_pred.reshape(len(val), 16),
        matrix_ids=val.matrix_ids,
        condition_ids=val.condition_ids,
    )
    diag = diagnostic_metrics(model, val, graphs, dev)
    if mono is not None and len(mono) > 0:
        midx = np.arange(min(len(mono), 1024))
        mb = mono_batch(mono, midx, graphs, device=dev)
        mp = model.forward_mono_batch(mb, mb)
        diag["mono_train_rmse"] = float(torch.sqrt(torch.mean((mp - mb["y"]) ** 2)).detach().cpu())
    if model.symmetric_pair:
        gaps = []
        for bidx in _batch_iter(len(val), min(256, len(val)), np.random.default_rng(0)):
            g = graph_batch(val, bidx, graphs, device=dev)
            sa, sb, ma, mby = model_boundary(model, table_batch(val, bidx, device=dev), g)
            gaps.append(float(boundary_loss(sa, sb, ma, mby, float(train_cfg["huber_delta"]))[1]))
        diag["E_boundary_mean"] = float(np.mean(gaps)) if gaps else 0.0
    val_metrics.update(diag)
    val_metrics["best_epoch"] = best_epoch
    val_metrics["n_parameters"] = n_params
    val_metrics["mono_access"] = mono_summary
    val_metrics["mono_train_curves"] = len(mono) if mono is not None else 0

    pred_df = build_prediction_frame(
        model=f"m5_{variant_name}",
        scenario=scenario,
        seed_or_fold=split_id,
        split="val",
        matrix_ids=val.matrix_ids,
        condition_ids=val.condition_ids,
        pair_ids=val.pair_ids,
        cell_ids=val.cell_ids,
        batch_ids=val.batch_ids.tolist(),
        drug_a_ids=val.drug_a_ids,
        drug_b_ids=val.drug_b_ids,
        dose_a_um=val.dose_a_um,
        dose_b_um=val.dose_b_um,
        y_true=val.y,
        y_pred=val_pred.reshape(-1, 4, 4),
        cell_lines=val.cell_lines,
        drug_a_names=val.drug_a_names,
        drug_b_names=val.drug_b_names,
    )
    if len(pred_df) != len(val) * 16:
        raise AssertionError("prediction table must have 16 rows per matrix")
    if pred_df[["y_true", "y_pred"]].isna().any().any():
        raise AssertionError("NaN in predictions")

    run_dir.mkdir(parents=True, exist_ok=True)
    save_json_atomic(
        {**cfg, "variant": {"name": variant_name, **variant}, "run": {"run_id": run_id, "scenario": scenario, "split_id": split_id}},
        run_dir / "config.yaml",
    )
    save_torch_atomic({"state_dict": model.state_dict(), "epoch": best_epoch, "val_metrics": val_metrics}, run_dir / "best.pt")
    with open(run_dir / "train_log.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    pred_df.to_parquet(run_dir / "predictions_val.parquet", index=False)
    save_json_atomic(json_safe(val_metrics), run_dir / "val_metrics.json")
    resources = {
        "device": str(dev),
        "hostname": os.uname().nodename,
        "wall_seconds": time.time() - run_start,
        "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(dev) / 1e6 if dev.type == "cuda" else 0.0,
        "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(dev) / 1e6 if dev.type == "cuda" else 0.0,
        "n_parameters": n_params,
        "epochs_run": len(log_rows),
        "n_train": len(train),
        "n_val": len(val),
        "n_mono_train": len(mono) if mono is not None else 0,
        "variant": variant_name,
        "graph_metadata_keys": sorted(graph_meta.keys()),
    }
    save_json_atomic(json_safe(resources), run_dir / "resources.json")
    save_json_atomic(
        {
            "run_id": run_id,
            "model": f"m5_{variant_name}",
            "scenario": scenario,
            "split_id": split_id,
            "seed": seed,
            "fold": fold,
            "status": "ok",
            "generated_at": utc_now(),
            "git_commit": os.popen("git rev-parse HEAD").read().strip(),
            "environment": environment_record(),
            "variant": {"name": variant_name, **variant},
        },
        run_dir / "run_metadata.json",
    )
    return {"run_id": run_id, "status": "ok", "run_dir": str(run_dir), "val": val_metrics, "resources": resources}


def model_boundary(model: DoseSynCore, batch: Dict[str, torch.Tensor], graphs: Dict[str, torch.Tensor]):
    """Return F(d_a,0), F(0,d_b), M(d_a), M(d_b) for one batch of matrices."""
    ha, hb = model.encode_drugs(graphs)
    cell_h = model.encode_cell(batch["cell_features"])
    zeros = torch.zeros_like(batch["dose_a"])
    ones = torch.ones_like(batch["dose_a_zero"])
    sa = model(
        batch,
        graphs,
        dose_a=batch["dose_a"],
        dose_b=zeros,
        zero_a=batch["dose_a_zero"],
        zero_b=ones,
    )
    sb = model(
        batch,
        graphs,
        dose_a=zeros,
        dose_b=batch["dose_b"],
        zero_a=ones,
        zero_b=batch["dose_b_zero"],
    )
    ma = model.predict_mono(ha, cell_h, batch["dose_a"], batch["dose_a_zero"])
    mb = model.predict_mono(hb, cell_h, batch["dose_b"], batch["dose_b_zero"])
    return sa, sb, ma, mb
