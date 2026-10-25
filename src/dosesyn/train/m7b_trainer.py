"""M7-B: strict dose-position holdout trainer.

Only observed grid positions contribute to the training loss.  Validation and
test metrics are computed only on the held-out positions.  Checkpoints and all
hyper-parameters are selected on validation held-out RMSE.
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

from dosesyn.models.baseline_models import build_dl_model, count_parameters
from dosesyn.models.m6b_models import build_m6b_model
from dosesyn.training.m4_data import (
    load_graphs,
    load_matrix_table,
    table_batch,
)
from dosesyn.training.metrics import build_prediction_frame, compute_metrics, json_safe
from dosesyn.training.reproducibility import (
    environment_record,
    git_commit,
    save_json_atomic,
    save_torch_atomic,
    set_global_seed,
    utc_now,
)
from dosesyn.train.m6a_trainer import slice_graph_batch, vocab_graph_batch
from dosesyn.train.m7b_masks import MASK_SPECS, indices, mask_version
from dosesyn.utils.config import load_config, resolve_path


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Per-matrix mean Huber over observed entries only."""
    err = pred - target
    abs_err = err.abs()
    hub = torch.where(abs_err <= delta, 0.5 * err * err, delta * (abs_err - 0.5 * delta))
    denom = mask.sum(dim=1).clamp_min(1.0)
    return ((hub * mask).sum(dim=1) / denom).mean()


def _build_model(model_key: str, cfg: Dict[str, Any], n_drugs: int, cell_dim: int):
    spec = cfg["models"][model_key]
    if spec["source"] == "m4":
        m4 = load_config(cfg["inputs"]["m4_config"])
        m = m4["model"]
        return build_dl_model(
            spec["name"],
            n_drugs=n_drugs,
            cell_dim=cell_dim,
            hidden=int(m["hidden_dim"]),
            drug_dim=int(m["drug_embedding_dim"]),
            mlp_hidden=int(m["mlp_hidden"]),
            gine_layers=int(m["gine_layers"]),
            dropout=float(m["dropout"]),
        ), "m4"
    m6b = load_config(cfg["inputs"]["m6b_config"])
    variant = dict(m6b["variants"][spec["variant"]])
    return build_m6b_model(variant, m6b["model"]), "m6b", variant


def _surface(model, kind: str, b, g) -> torch.Tensor:
    if kind == "m4":
        return model(b, g)
    return model.forward_all(b, g)["surface"]


def _held_metrics(table, pred: np.ndarray, mask_name: str) -> Dict[str, Any]:
    held = indices(mask_name)
    p = np.asarray(pred)[:, held]
    y = np.asarray(table.y).reshape(len(table.matrix_ids), -1)[:, held]
    m = compute_metrics(y, p, matrix_ids=table.matrix_ids, condition_ids=table.condition_ids)
    m["n_held_positions"] = len(held)
    m["n_points_evaluated"] = int(p.size)
    m["mean_bias"] = float(np.mean(p - y))
    m["mean_abs_bias"] = float(np.mean(np.abs(p - y)))
    return m


def _full_metrics(table, pred: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(table.y).reshape(len(table.matrix_ids), -1)
    return compute_metrics(y, np.asarray(pred), matrix_ids=table.matrix_ids,
                           condition_ids=table.condition_ids)


def _predict(model, kind: str, table, graphs, device, mask_name: str, batch_size: int = 256) -> np.ndarray:
    full = vocab_graph_batch(table, graphs, device)
    preds, i = [], 0
    model.eval()
    with torch.no_grad():
        while i < len(table.matrix_ids):
            idx = list(range(i, min(i + batch_size, len(table.matrix_ids))))
            b = table_batch(table, idx, device=device)
            g = slice_graph_batch(full, table, idx, device)
            preds.append(_surface(model, kind, b, g).detach().cpu().numpy())
            i += batch_size
    return np.concatenate(preds, axis=0)


def _save_predictions(path: Path, table, pred: np.ndarray, mask_name: str,
                      model: str, scenario: str, split_id: str, split: str) -> None:
    df = build_prediction_frame(
        model=model,
        scenario=scenario,
        seed_or_fold=split_id,
        split=split,
        matrix_ids=table.matrix_ids,
        condition_ids=table.condition_ids,
        pair_ids=table.pair_ids,
        cell_ids=table.cell_ids,
        cell_lines=table.cell_lines,
        batch_ids=table.batch_ids,
        drug_a_ids=table.drug_a_ids,
        drug_b_ids=table.drug_b_ids,
        drug_a_names=table.drug_a_names,
        drug_b_names=table.drug_b_names,
        dose_a_um=table.dose_a_um,
        dose_b_um=table.dose_b_um,
        y_true=np.asarray(table.y).reshape(len(table.matrix_ids), -1),
        y_pred=np.asarray(pred),
    )
    held = set(indices(mask_name))
    df["is_held"] = df.apply(lambda r: int(r["row_idx"] * 4 + r["col_idx"]) in held, axis=1)
    df["mask_name"] = mask_name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run_m7b(
    mask_name: str,
    model_key: str,
    scenario: str,
    *,
    seed: Optional[int] = None,
    fold: Optional[int] = None,
    split_id: Optional[str] = None,
    device: str = "cuda",
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
    force: bool = False,
    eval_test: bool = True,
    config_path: str = "configs/m7b/m7b.yaml",
) -> Dict[str, Any]:
    if mask_name not in MASK_SPECS:
        raise KeyError(mask_name)
    cfg = load_config(config_path)
    dev = torch.device(device)
    epochs = int(epochs or cfg["training"]["epochs"])
    patience = int(patience or cfg["training"]["patience"])
    split_id = split_id or (f"seed_{seed}" if seed is not None else f"fold_{fold}")
    run_id = f"m7b__{mask_name}__{model_key}__{scenario}__{split_id}"
    out = resolve_path(cfg, cfg["outputs"]["runs_dir"]) / run_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_metadata.json").exists() and not force:
        return {"run_id": run_id, "status": "skip_exists"}

    set_global_seed(int(cfg["training"]["seed"]))
    graphs, _ = load_graphs(cfg["inputs"]["m4_config"])
    train = load_matrix_table(scenario, "train", seed=seed, fold=fold)
    val = load_matrix_table(scenario, "val", seed=seed, fold=fold)
    train_g = vocab_graph_batch(train, graphs, dev)
    val_g = vocab_graph_batch(val, graphs, dev)

    built = _build_model(model_key, cfg, n_drugs=len(train.drug_vocab), cell_dim=train.cell_features.shape[1])
    model, kind = built[0], built[1]
    variant = built[2] if len(built) > 2 else None
    model = model.to(dev)

    obs_np = np.zeros((1, 16), dtype=np.float32)
    obs_np[0, indices(mask_name, observed=True)] = 1.0
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=dev)

    tr = cfg["training"]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=float(tr["learning_rate"]), weight_decay=float(tr["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    bs = int(tr["batch_size"])
    best, best_ep, best_state, wait = float("inf"), 0, None, 0
    history: List[Dict[str, Any]] = []
    start = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(1000 * ep + int(tr["seed"])).permutation(len(train.matrix_ids))
        total, nb = 0.0, 0
        for s in range(0, len(order), bs):
            idx = order[s:s + bs].tolist()
            b = table_batch(train, idx, device=dev)
            g = slice_graph_batch(train_g, train, idx, dev)
            pred = _surface(model, kind, b, g)
            m = obs.expand(pred.shape[0], -1)
            loss = masked_huber(pred, b["y"].reshape(pred.shape[0], -1), m,
                                delta=float(tr["huber_delta"]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr["grad_clip"]))
            opt.step()
            opt.zero_grad(set_to_none=True)
            total += float(loss.detach().cpu())
            nb += 1

        val_pred = _predict(model, kind, val, graphs, dev, mask_name, int(tr["eval_batch_size"]))
        vm = _held_metrics(val, val_pred, mask_name)
        macro = float(vm["matrix_macro_rmse"])
        sched.step()
        rec = {"epoch": ep, "train_loss": total / max(nb, 1), "val_held_macro_rmse": macro,
               "val_held_point_rmse": float(vm["point_rmse"]), "val_held_mae": float(vm["matrix_macro_mae"]),
               "lr": opt.param_groups[0]["lr"]}
        history.append(rec)
        if macro < best - 1e-6:
            best, best_ep, wait = macro, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        print(f"[{run_id}] ep {ep:03d} train {rec['train_loss']:.5f} val_held {macro:.5f} "
              f"best {best:.5f}@{best_ep}", flush=True)
        if ep >= int(tr["min_epochs"]) and wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    wall = time.time() - start
    val_pred = _predict(model, kind, val, graphs, dev, mask_name, int(tr["eval_batch_size"]))
    val_held = _held_metrics(val, val_pred, mask_name)
    val_full = _full_metrics(val, val_pred)

    result = {
        "run_id": run_id, "stage": "M7-B", "mask": mask_name,
        "mask_version": mask_version(mask_name),
        "mask_observed": indices(mask_name, observed=True),
        "mask_held": indices(mask_name),
        "model": model_key, "scenario": scenario, "split_id": split_id,
        "seed": seed, "fold": fold,
        "best_epoch": best_ep, "epochs_run": len(history), "wall_seconds": wall,
        "n_parameters": int(count_parameters(model)),
        "n_train_matrices": len(train.matrix_ids), "n_val_matrices": len(val.matrix_ids),
        "best_val_held_macro": best,
        "val_held_metrics": json_safe(val_held), "val_full_metrics": json_safe(val_full),
        "selection_metric": "val_held_matrix_macro_rmse",
        "test_policy": "single_read_after_checkpoint_selection",
    }
    if variant is not None:
        result["variant"] = variant

    save_torch_atomic({"state_dict": model.state_dict(), "meta": result}, out / "best.pt")
    pd.DataFrame(history).to_csv(out / "train_log.csv", index=False)
    save_json_atomic(result, out / "val_metrics.json")
    save_json_atomic(result, out / "run_metadata.json")
    _save_predictions(out / "predictions_val.parquet", val, val_pred, mask_name,
                      model_key, scenario, split_id, "val")

    if eval_test:
        test = load_matrix_table(scenario, "test", seed=seed, fold=fold)
        test_pred = _predict(model, kind, test, graphs, dev, mask_name, int(tr["eval_batch_size"]))
        test_held = _held_metrics(test, test_pred, mask_name)
        test_full = _full_metrics(test, test_pred)
        save_json_atomic({"held": json_safe(test_held), "full": json_safe(test_full)},
                         out / "test_metrics.json")
        _save_predictions(out / "predictions_test.parquet", test, test_pred, mask_name,
                          model_key, scenario, split_id, "test")
        result["test_held_metrics"] = json_safe(test_held)
        result["test_full_metrics"] = json_safe(test_full)

    env = environment_record()
    resources = {
        "wall_seconds": wall, "host": env.get("hostname"),
        "gpu": env.get("gpu"), "torch": env.get("torch"),
        "gpu_peak_mb": float(torch.cuda.max_memory_allocated(dev) / 1024 ** 2) if dev.type == "cuda" else 0.0,
        "n_parameters": int(count_parameters(model)),
    }
    save_json_atomic(resources, out / "resources.json")
    print(f"[{run_id}] DONE best {best:.5f}@{best_ep} wall {wall:.0f}s", flush=True)
    return result
