"""Run DoseSyn M5 variants on frozen M3 splits (validation only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.train.m5_trainer import run_m5  # noqa: E402
from dosesyn.utils.config import ensure_directories, load_config  # noqa: E402
from dosesyn.utils.logging import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/m5_dosesyn.yaml")
    ap.add_argument("--scenario", choices=["s1_random_condition", "s2_unseen_pair", "s3_unseen_cell"], default="s1_random_condition")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=["c6_full"])
    ap.add_argument("--epochs", type=int, default=None, help="debug override (final runs use config)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_directories(cfg)
    log_dir = Path(cfg["paths"]["logs_dir"])
    logger = setup_logging("dosesyn.m5", log_dir / "06_run_dosesyn.log")
    device = args.device
    if device == "auto":
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning("CUDA not available; running on CPU")
    results = []
    failures = 0
    for name in args.variants:
        if name not in cfg["variants"]:
            raise SystemExit(f"unknown variant {name}; choose from {sorted(cfg['variants'])}")
        variant = dict(cfg["variants"][name])
        logger.info("M5 variant %s on %s", name, args.scenario)
        try:
            result = run_m5(
                scenario=args.scenario,
                seed=args.seed if args.scenario != "s3_unseen_cell" else None,
                fold=args.fold if args.scenario == "s3_unseen_cell" else None,
                variant_name=name,
                variant=variant,
                cfg=cfg,
                device=device,
                epochs=args.epochs,
                force=args.force,
            )
            results.append(result)
            if result.get("status") == "ok":
                logger.info(
                    "%s: val macro RMSE %.4f | E_swap max %.3g | best epoch %s",
                    result["run_id"],
                    result["val"]["matrix_macro_rmse"],
                    result["val"]["E_swap_max"],
                    result["val"]["best_epoch"],
                )
        except Exception as exc:  # keep the other variants running; log the real error
            failures += 1
            logger.exception("variant %s failed", name)
            results.append({"variant": name, "status": "failed", "error": repr(exc)})
    print(json.dumps(results, indent=2, default=str))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
