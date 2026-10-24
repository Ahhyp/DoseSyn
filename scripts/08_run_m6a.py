#!/usr/bin/env python3
"""Run M6-A diagnostic experiments on frozen train/val splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.train.m6a_trainer import run_m6a
from dosesyn.utils.config import ensure_directories, load_config
from dosesyn.utils.logging import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    choices=["b3_exact", "c0_control", "b3sym_scalar", "b3sym_fourier", "b3sym_film"])
    ap.add_argument("--scenario", default="s1_random_condition")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--recipe", default="m6a", choices=["m4", "m6a"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    cfg = load_config("configs/m6a_diagnostic.yaml")
    ensure_directories(cfg)
    for key in ("runs_dir", "metrics_dir", "figures_dir", "audit_dir"):
        Path(cfg["outputs"][key]).mkdir(parents=True, exist_ok=True)
    log = setup_logging("dosesyn.m6a", Path(cfg["paths"]["logs_dir"]) / "08_run_m6a.log")
    device = "cuda" if args.device == "auto" else args.device

    specs = cfg["runs"] if args.all else [{
        "model": args.model, "scenario": args.scenario,
        "seed": args.seed if args.scenario != "s3_unseen_cell" else None,
        "fold": args.fold if args.scenario == "s3_unseen_cell" else None,
    }]
    if not args.all and args.model is None:
        ap.error("--model is required unless --all")

    results = []
    for spec in specs:
        try:
            res = run_m6a(
                model_name=spec["model"], scenario=spec["scenario"],
                seed=spec.get("seed"), fold=spec.get("fold"),
                cfg=cfg, recipe=args.recipe, epochs=args.epochs,
                patience=args.patience, device_name=device, force=args.force,
            )
            results.append(res)
            v = res.get("val", {})
            log.info("%s status=%s macro=%.6f point=%.6f best_epoch=%s",
                     res["run_id"], res["status"], v.get("matrix_macro_rmse", float("nan")),
                     v.get("point_rmse", float("nan")), v.get("best_epoch"))
        except Exception as exc:  # noqa: BLE001
            log.exception("run failed: %s", spec)
            results.append({"spec": spec, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps(results, indent=2, default=str))
    return 0 if all(r.get("status") in {"ok", "exists"} for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
