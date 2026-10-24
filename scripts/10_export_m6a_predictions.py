#!/usr/bin/env python3
"""Export M6-A validation predictions from best.pt (no training, no test)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.models.m6a_models import build_m6a_model
from dosesyn.training.m4_data import load_graphs, load_matrix_table
from dosesyn.train.m6a_trainer import predict_table, vocab_graph_batch
from dosesyn.utils.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m6a_diagnostic.yaml")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    graphs, _ = load_graphs("configs/m4_training.yaml")
    runs_dir = ROOT / cfg["outputs"]["runs_dir"]
    exported = []
    for d in sorted(runs_dir.glob("m6a__*")):
        metrics_path = d / "val_metrics.json"
        ckpt = d / "best.pt"
        if not metrics_path.exists() or not ckpt.exists():
            continue
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_id = d.name
        _, model_name, scenario, split_id = run_id.split("__")
        if scenario == "s1_random_condition":
            seed, fold = int(split_id.split("_")[1]), None
        else:
            seed, fold = None, int(split_id.split("_")[1])
        model = build_m6a_model(model_name)
        payload = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        model.eval()
        table = load_matrix_table(scenario, "val", seed=seed, fold=fold)
        pred = predict_table(model, table, graphs, device, 128).reshape(-1, 16)
        n = len(table)
        assert pred.shape == (n, 16)
        frame = pd.DataFrame({
            "run_id": run_id,
            "model": model_name,
            "matrix_id": np.repeat(table.matrix_ids, 16),
            "condition_id": np.repeat(table.condition_ids, 16),
            "pair_id": np.repeat(table.pair_ids, 16),
            "cell_id": np.repeat(table.cell_ids, 16),
            "batch_id": np.repeat(table.batch_ids, 16),
            "drug_a_id": np.repeat(table.drug_a_ids, 16),
            "drug_b_id": np.repeat(table.drug_b_ids, 16),
            "row_idx": np.tile(np.arange(4), n * 4),
            "col_idx": np.tile(np.repeat(np.arange(4), 4), n),
            "dose_a_um": np.repeat(table.dose_a_um, 4, axis=1).reshape(-1),
            "dose_b_um": np.tile(table.dose_b_um, (1, 4)).reshape(-1),
            "y_true": table.y.reshape(-1),
            "y_pred": pred.reshape(-1),
        })
        frame["residual"] = frame["y_pred"] - frame["y_true"]
        assert len(frame) == n * 16 and not frame[["y_true", "y_pred"]].isna().any().any()
        out = d / "predictions_val.parquet"
        frame.to_parquet(out, index=False)
        exported.append({"run_id": run_id, "rows": int(len(frame)),
                         "n_matrices": int(frame.matrix_id.nunique())})
        print(f"wrote {out.relative_to(ROOT)} rows={len(frame)}", flush=True)
    summary = ROOT / cfg["outputs"]["metrics_dir"] / "prediction_export.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({"runs": exported, "device": args.device}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
