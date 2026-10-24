"""Run one M4 experiment matrix (all requested models on one frozen split)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dosesyn.training.reproducibility import save_json_atomic
from dosesyn.training.runner import MODEL_NAMES, RunSpec, run_experiment
from dosesyn.utils.config import load_config
from dosesyn.utils.logging import setup_logging

SCENARIOS = ["s1_random_condition", "s2_unseen_pair", "s3_unseen_cell"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", required=True, choices=SCENARIOS)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--models", nargs="+", default=["all"], choices=MODEL_NAMES + ["all"])
    p.add_argument("--config", default="configs/m4_training.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true", help="rerun even if test_metrics.json exists")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    run_root = Path(cfg["outputs"]["runs_dir"])
    run_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        name="dosesyn.m4",
        log_file=Path(cfg["_root"]) / "outputs/logs" / f"04_run_{args.scenario}.log",
    )
    if args.scenario == "s3_unseen_cell":
        if args.fold is None:
            raise SystemExit("S3 requires --fold")
    elif args.seed is None:
        raise SystemExit("S1/S2 require --seed")

    models = MODEL_NAMES if args.models == ["all"] else args.models
    results = []
    for model in models:
        spec = RunSpec(
            model=model,
            scenario=args.scenario,
            seed=args.seed,
            fold=args.fold,
            max_seconds=args.max_seconds,
            max_epochs=args.epochs,
        )
        if args.force:
            for name in ("test_metrics.json", "error.json"):
                p = run_root / spec.run_id / name
                if p.exists():
                    p.unlink()
        logger.info("running %s (%s %s)", spec.run_id, args.device, spec.split_tag)
        try:
            metrics = run_experiment(
                spec, config_path=args.config, device_name=args.device, resume=args.resume
            )
            results.append({"run_id": spec.run_id, "status": "ok", **metrics})
            logger.info(
                "%s: test macro RMSE %.4f | point RMSE %.4f | R2 %s",
                spec.run_id,
                metrics["matrix_macro_rmse"],
                metrics["point_rmse"],
                metrics["r2"],
            )
        except Exception as exc:
            logger.exception("run failed: %s", spec.run_id)
            results.append({"run_id": spec.run_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        save_json_atomic(results, run_root / f"_last_{args.scenario}.json")

    print(json.dumps(results, indent=2))
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
