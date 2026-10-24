"""M4 experiment runner: one frozen split x one model = one traceable run."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ..models.baseline_models import build_dl_model, count_parameters
from ..models.tree_baseline import TreeDoseBaseline
from ..utils.config import load_config, project_root
from . import m4_data
from .metrics import build_prediction_frame, compute_metrics, json_safe
from .reproducibility import (
    environment_record,
    file_sha256,
    git_commit,
    git_status_porcelain,
    save_json_atomic,
    save_torch_atomic,
    set_global_seed,
    utc_now,
)

MODEL_NAMES = ["b0_mean", "b1_tree", "b2_mlp", "b3_gine"]


@dataclass
class RunSpec:
    model: str
    scenario: str
    seed: Optional[int] = None
    fold: Optional[int] = None
    max_seconds: Optional[float] = None
    max_epochs: Optional[int] = None

    @property
    def split_tag(self) -> str:
        if self.scenario == "s3_unseen_cell":
            return f"fold_{self.fold}"
        return f"seed_{self.seed}"

    @property
    def run_id(self) -> str:
        return f"{self.model}__{self.scenario}__{self.split_tag}"


def run_dir_for(cfg: Dict[str, Any], spec: RunSpec) -> Path:
    return Path(cfg["outputs"]["runs_dir"]) / spec.run_id


def _device(name: str = "cuda") -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _subset(table: m4_data.MatrixTable, idx: np.ndarray) -> m4_data.MatrixTable:
    out = object.__new__(m4_data.MatrixTable)
    for k, v in table.__dict__.items():
        if isinstance(v, np.ndarray) and v.shape[0] == len(table):
            out.__dict__[k] = v[idx]
        elif isinstance(v, list) and len(v) == len(table):
            out.__dict__[k] = [v[i] for i in idx]
        else:
            out.__dict__[k] = v
    return out


def _dl_predict(
    model: nn.Module,
    table: m4_data.MatrixTable,
    graphs: Optional[Dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(table), batch_size):
            idx = list(range(start, min(start + batch_size, len(table))))
            batch = m4_data.table_batch(table, idx, device=device)
            gb = (
                m4_data.graph_batch(table, idx, graphs, device=device)
                if getattr(model, "needs_graphs", False)
                else None
            )
            out = model(batch, gb)
            preds.append(out.detach().float().cpu().numpy().reshape(len(idx), 4, 4))
    return np.concatenate(preds, axis=0).astype(np.float64)


def _metrics_for(table: m4_data.MatrixTable, pred: np.ndarray) -> Dict[str, Any]:
    return compute_metrics(
        table.y,
        pred,
        matrix_ids=table.matrix_ids,
        condition_ids=table.condition_ids,
    )


def _save_predictions(
    path: Path,
    table: m4_data.MatrixTable,
    pred: np.ndarray,
    *,
    model: str,
    scenario: str,
    seed_or_fold: str,
    split: str,
) -> None:
    frame = build_prediction_frame(
        model=model,
        scenario=scenario,
        seed_or_fold=seed_or_fold,
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
        y_true=table.y,
        y_pred=pred,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _resources(device: torch.device, started: float, **extra: Any) -> Dict[str, Any]:
    rec = {
        "device": str(device),
        "wall_seconds": round(time.time() - started, 3),
        "hostname": environment_record().get("hostname"),
    }
    if device.type == "cuda":
        rec["gpu_name"] = torch.cuda.get_device_name(0)
        rec["gpu_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        rec["gpu_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1e6, 1)
    rec.update(extra)
    return rec


def _metadata(
    cfg: Dict[str, Any],
    spec: RunSpec,
    table: m4_data.MatrixTable,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    root = Path(cfg["_root"])
    m3_manifest = Path(cfg["inputs"]["m3_manifest"])
    split_meta = json.loads((table.split_dir / "split_metadata.json").read_text(encoding="utf-8"))
    meta = {
        "run_id": spec.run_id,
        "model": spec.model,
        "scenario": spec.scenario,
        "seed": spec.seed,
        "fold": spec.fold,
        "split": table.split,
        "split_dir": str(table.split_dir),
        "n_matrices": len(table),
        "label": "x_x0_raw (raw, unclipped)",
        "git_commit": git_commit(root),
        "git_dirty": bool(git_status_porcelain(root)),
        "m3_manifest_sha256": file_sha256(m3_manifest) if m3_manifest.exists() else None,
        "split_content_sha256": split_meta.get("content_sha256"),
        "split_assignment_sha256": split_meta.get("assignment_sha256"),
        "scaler_sha256": split_meta.get("scaler_sha256"),
        "mono_policy": split_meta.get("mono_policy"),
        "m4_config_sha256": file_sha256(cfg["_config_path"]),
        "environment": environment_record(),
        "created_at": utc_now(),
    }
    if extra:
        meta.update(extra)
    return meta


def run_experiment(
    spec: RunSpec,
    *,
    config_path: str = "configs/m4_training.yaml",
    device_name: str = "cuda",
    resume: bool = False,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    root = Path(cfg["_root"])
    out = run_dir_for(cfg, spec)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if (out / "test_metrics.json").exists() and not resume:
        return json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))

    # Immutable provenance snapshot.
    (out / "config.yaml").write_text(Path(cfg["_config_path"]).read_text(encoding="utf-8"), encoding="utf-8")
    set_global_seed(spec.seed if spec.seed is not None else 42)
    torch.set_num_threads(int(cfg["training"]["num_threads"]))

    train = m4_data.load_matrix_table(
        spec.scenario, "train", seed=spec.seed, fold=spec.fold, config_path=cfg["inputs"]["dataset_config"]
    )
    val = m4_data.load_matrix_table(
        spec.scenario, "val", seed=spec.seed, fold=spec.fold, config_path=cfg["inputs"]["dataset_config"]
    )
    test = m4_data.load_matrix_table(
        spec.scenario, "test", seed=spec.seed, fold=spec.fold, config_path=cfg["inputs"]["dataset_config"]
    )
    graphs = None
    if spec.model == "b3_gine":
        graphs, _ = m4_data.load_graphs(config_path)
    device = _device(device_name)

    try:
        if spec.model == "b0_mean":
            model = build_dl_model(
                "b0_mean",
                n_drugs=len(train.drug_vocab),
                cell_dim=train.cell_features.shape[1],
                hidden=cfg["model"]["hidden_dim"],
                drug_dim=cfg["model"]["drug_embedding_dim"],
                mlp_hidden=cfg["model"]["mlp_hidden"],
                gine_layers=cfg["model"]["gine_layers"],
                dropout=cfg["model"]["dropout"],
                mean=float(train.y.mean()),
            )
            val_pred = np.full_like(val.y, float(train.y.mean()), dtype=np.float64)
            test_pred = np.full_like(test.y, float(train.y.mean()), dtype=np.float64)
            save_torch_atomic(
                {"model_state": model.state_dict(), "model_name": "b0_mean", "mean": float(train.y.mean())},
                out / "best.pt",
            )
            train_log = pd.DataFrame(
                [{"epoch": 0, "train_loss": None, "val_matrix_macro_rmse": None}]
            )
            epochs_run = 0
            params = 1
        elif spec.model == "b1_tree":
            tree = TreeDoseBaseline(
                max_iter=int(cfg["model"]["tree_max_iter"]),
                learning_rate=float(cfg["model"]["tree_learning_rate"]),
                max_leaf_nodes=int(cfg["model"]["tree_max_leaf_nodes"]),
                random_state=spec.seed if spec.seed is not None else 42,
            ).fit(train)
            val_pred = tree.predict(val)
            test_pred = tree.predict(test)
            params = tree.n_parameters()
            train_log = pd.DataFrame([{"epoch": 0, "train_loss": None, "val_matrix_macro_rmse": None}])
            epochs_run = 1
            save_torch_atomic({"leaf_nodes": params, "model_name": "b1_tree"}, out / "best.pt")
        else:
            model, val_pred, test_pred, train_log, epochs_run = _train_dl(
                cfg, spec, train, val, test, graphs, device
            )
            params = count_parameters(model)

        val_metrics = _metrics_for(val, val_pred)
        test_metrics = _metrics_for(test, test_pred)
        _save_predictions(out / "predictions_val.parquet", val, val_pred, model=spec.model, scenario=spec.scenario, seed_or_fold=spec.split_tag, split="val")
        _save_predictions(out / "predictions_test.parquet", test, test_pred, model=spec.model, scenario=spec.scenario, seed_or_fold=spec.split_tag, split="test")
        train_log.to_csv(out / "train_log.csv", index=False)

        resources = _resources(
            device,
            started,
            n_parameters=params,
            epochs_run=int(epochs_run),
            n_train=len(train),
            n_val=len(val),
            n_test=len(test),
        )
        meta = _metadata(
            cfg,
            spec,
            train,
            extra={
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                "n_parameters": params,
                "label_min": float(train.y.min()),
                "label_max": float(train.y.max()),
                "label_mean": float(train.y.mean()),
            },
        )
        save_json_atomic(meta, out / "run_metadata.json")
        save_json_atomic(resources, out / "resources.json")
        save_json_atomic(json_safe(val_metrics), out / "val_metrics.json")
        save_json_atomic(json_safe({**test_metrics, "run_id": spec.run_id, "model": spec.model}), out / "test_metrics.json")
        return json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))
    except Exception as exc:  # keep real logs for failed runs
        save_json_atomic(
            {
                "run_id": spec.run_id,
                "model": spec.model,
                "scenario": spec.scenario,
                "seed": spec.seed,
                "fold": spec.fold,
                "error": f"{type(exc).__name__}: {exc}",
                "created_at": utc_now(),
            },
            out / "error.json",
        )
        raise


def _train_dl(
    cfg: Dict[str, Any],
    spec: RunSpec,
    train: m4_data.MatrixTable,
    val: m4_data.MatrixTable,
    test: m4_data.MatrixTable,
    graphs: Optional[Dict[str, Any]],
    device: torch.device,
) -> Tuple[nn.Module, np.ndarray, np.ndarray, pd.DataFrame, int]:
    model = build_dl_model(
        spec.model,
        n_drugs=len(train.drug_vocab),
        cell_dim=train.cell_features.shape[1],
        hidden=int(cfg["model"]["hidden_dim"]),
        drug_dim=int(cfg["model"]["drug_embedding_dim"]),
        mlp_hidden=int(cfg["model"]["mlp_hidden"]),
        gine_layers=int(cfg["model"]["gine_layers"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    epochs = int(spec.max_epochs or cfg["training"]["epochs_dl"])
    bs = int(cfg["training"]["batch_size"])
    eval_bs = int(cfg["training"]["eval_batch_size"])
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.HuberLoss(delta=float(cfg["training"]["huber_delta"]))
    patience = int(cfg["training"]["patience"])
    min_epochs = int(cfg["training"]["min_epochs"])
    rng = np.random.default_rng(spec.seed if spec.seed is not None else 42)
    best = float("inf")
    best_state = None
    best_epoch = -1
    bad = 0
    log: List[Dict[str, Any]] = []
    elapsed_limit = spec.max_seconds
    run_start = time.time()

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for start in range(0, len(train), bs):
            idx = order[start : start + bs].tolist()
            batch = m4_data.table_batch(train, idx, device=device)
            gb = m4_data.graph_batch(train, idx, graphs, device=device) if model.needs_graphs else None
            opt.zero_grad(set_to_none=True)
            pred = model(batch, gb)
            loss = loss_fn(pred, batch["y"].reshape(len(idx), 16))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip"]))
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        val_pred = _dl_predict(model, val, graphs, device=device, batch_size=eval_bs)
        val_rmse = float(np.sqrt(np.mean((val.y - val_pred) ** 2)))
        log.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_matrix_macro_rmse": val_rmse,
                "lr": float(opt.param_groups[0]["lr"]),
                "epoch_seconds": round(time.time() - t0, 2),
            }
        )
        if val_rmse < best - 1e-6:
            best = val_rmse
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_torch_atomic(
                {
                    "model_state": best_state,
                    "model_name": spec.model,
                    "epoch": best_epoch,
                    "val_macro_rmse": best,
                    "n_drugs": len(train.drug_vocab),
                    "cell_dim": int(train.cell_features.shape[1]),
                    "config": cfg["model"],
                },
                run_dir_for(cfg, spec) / "best.pt",
            )
            bad = 0
        else:
            bad += 1
        if epoch + 1 >= min_epochs and bad >= patience:
            break
        if elapsed_limit and (time.time() - run_start) > elapsed_limit:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred = _dl_predict(model, val, graphs, device=device, batch_size=eval_bs)
    test_pred = _dl_predict(model, test, graphs, device=device, batch_size=eval_bs)
    return model, val_pred, test_pred, pd.DataFrame(log), len(log)
