#!/usr/bin/env python3
"""Run M6-B validation or formal experiments on frozen M3 splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.train.m6b_trainer import run_m6b
from dosesyn.utils.config import load_config
from dosesyn.utils.logging import setup_logging


def _validation_runs(cfg: Dict[str, Any], variants: List[str]) -> List[Dict[str, Any]]:
    vcfg = cfg["validation"]
    jobs = []
    for sc in vcfg["scenarios"]:
        for variant in variants:
            jobs.append({
                "variant": variant, "scenario": sc["name"],
                "seed": sc.get("seed"), "fold": sc.get("fold"),
                "split_id": sc["split_id"], "eval_test": False,
            })
    return jobs


def _formal_runs(cfg: Dict[str, Any], models: List[str]) -> List[Dict[str, Any]]:
    jobs = []
    for sc in cfg["formal"]["scenarios"]:
        name = sc["name"]
        if "seeds" in sc:
            for seed in sc["seeds"]:
                for model in models:
                    jobs.append({"variant": model, "scenario": name, "seed": seed,
                                 "fold": None, "split_id": f"seed_{seed}", "eval_test": True})
        else:
            for fold in sc["folds"]:
                for model in models:
                    jobs.append({"variant": model, "scenario": name, "seed": None,
                                 "fold": fold, "split_id": f"fold_{fold}", "eval_test": True})
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m6b_experiments.yaml")
    ap.add_argument("--stage", choices=["validation", "formal"], default="validation")
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--model", action="append", default=[], help="formal model variant")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--eval-test", action="store_true", help="formal: allow one test read")
    ap.add_argument("--scenario", action="append", default=[])
    ap.add_argument("--seed", type=int, action="append", default=[])
    ap.add_argument("--fold", type=int, action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ROOT.joinpath(cfg["paths"]["logs_dir"]).mkdir(parents=True, exist_ok=True)
    log_path = ROOT / cfg["paths"]["logs_dir"] / f"11_m6b_{args.stage}.log"
    log = setup_logging("dosesyn.m6b", log_path, "INFO")

    if args.stage == "validation":
        variants = args.variant or list(cfg["validation"]["variants"])
        jobs = _validation_runs(cfg, variants)
    else:
        models = args.model or list(cfg["formal"]["models"])
        jobs = _formal_runs(cfg, models)
    if args.scenario:
        jobs = [j for j in jobs if j["scenario"] in args.scenario]
    if args.seed:
        jobs = [j for j in jobs if j["seed"] in args.seed]
    if args.fold:
        jobs = [j for j in jobs if j["fold"] in args.fold]
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return 0

    results = []
    for job in jobs:
        variant = job["variant"]
        if args.stage == "formal" and not args.eval_test:
            log.warning("formal stage without --eval-test: %s will only validate", variant)
        try:
            r = run_m6b(
                variant, job["scenario"], seed=job["seed"], fold=job["fold"],
                split_id=job["split_id"], device=f"cuda:{args.gpu}" if args.device == "cuda" else "cpu",
                epochs=args.epochs, patience=args.patience, force=args.force,
                eval_test=bool(args.eval_test and job["eval_test"]),
                m6b_config=args.config,
            )
            results.append(r)
            log.info("%s %s %s macro=%s", r.get("run_id"), r.get("status", "ok"),
                     r.get("best_epoch"), r.get("best_val_macro"))
        except Exception as exc:  # noqa: BLE001
            log.exception("run failed: %s", job)
            results.append({**job, "status": "failed", "error": repr(exc)})
    print(json.dumps(results, indent=2, default=str))
    return 0 if all(r.get("status") != "failed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
