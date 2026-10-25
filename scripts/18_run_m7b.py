"""M7-B: run dose-position holdout experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.train.m7b_trainer import run_m7b  # noqa: E402
from dosesyn.utils.config import ensure_directories, load_config
from dosesyn.utils.logging import setup_logging  # noqa: E402


def expand(groups):
    jobs = []
    for gname in groups:
        g = CFG["runs"][gname]
        masks = g.get("masks") or [g["mask"]]
        for mask in masks:
            for model in g["models"]:
                for scenario, spec in g["scenarios"].items():
                    kind = spec["kind"]
                    for v in spec["values"]:
                        jobs.append(dict(group=gname, mask=mask, model=model, scenario=scenario,
                                         **{kind: int(v)}))
    return jobs


def main() -> int:
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", default=None,
                    choices=["b1_center_all", "b2_dev_mlp", "sparse_dev", "boundary_dev"])
    ap.add_argument("--mask", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default="configs/m7b/m7b.yaml")
    args = ap.parse_args()

    CFG = load_config(args.config)
    ensure_directories(CFG)
    setup_logging("dosesyn.m7b", ROOT / "outputs/logs/18_run_m7b.log")
    groups = args.group or ["b1_center_all"]
    jobs = expand(groups)
    if args.mask:
        jobs = [j for j in jobs if j["mask"] == args.mask]
    if args.model:
        jobs = [j for j in jobs if j["model"] == args.model]
    if args.scenario:
        jobs = [j for j in jobs if j["scenario"] == args.scenario]
    if args.seed is not None:
        jobs = [j for j in jobs if j.get("seed") == args.seed]
    if args.fold is not None:
        jobs = [j for j in jobs if j.get("fold") == args.fold]

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps({"n_jobs": len(jobs), "device": device, "jobs": jobs[:3]}, ensure_ascii=False))
    if args.dry_run:
        print(json.dumps(jobs, ensure_ascii=False, indent=2))
        return 0

    results = []
    failed = 0
    for j in jobs:
        try:
            r = run_m7b(
                j["mask"], j["model"], j["scenario"],
                seed=j.get("seed"), fold=j.get("fold"),
                device=device, epochs=args.epochs, patience=args.patience,
                force=args.force, eval_test=not args.no_test,
                config_path=args.config,
            )
            results.append({"job": j, "status": r.get("status", "ok"), "run_id": r.get("run_id")})
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAILED] {j}: {exc}", flush=True)
            results.append({"job": j, "status": "failed", "error": repr(exc)})
    print(json.dumps({"n": len(results), "failed": failed, "results": results}, ensure_ascii=False,
                     default=str)[:4000])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
