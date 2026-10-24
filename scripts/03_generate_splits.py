#!/usr/bin/env python
"""Generate the frozen Primary-v1 index and all M3 split directories.

Examples
--------
python scripts/03_generate_splits.py --scenario all
python scripts/03_generate_splits.py --scenario s1_random_condition --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.splits.build import SCENARIOS, generate  # noqa: E402
from dosesyn.utils.logging import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "dataset_primary_v1.yaml"))
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all"] + list(SCENARIOS),
        help="scenario to build (default: all)",
    )
    parser.add_argument("--seed", type=int, action="append", default=None, help="repeatable; default all seeds in config")
    parser.add_argument("--fold", type=int, action="append", default=None, help="repeatable; default 0..4")
    args = parser.parse_args()

    logger = setup_logging(
        "dosesyn.splits",
        log_file=str(ROOT / "outputs" / "logs" / "03_generate_splits.log"),
    )
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    result = generate(
        args.config,
        scenarios,
        seeds=args.seed,
        folds=args.fold,
        logger=logger,
    )
    summary = {
        "freeze": result["freeze"]["observed"],
        "n_splits": len(result["splits"]),
        "manifest_sha256": result["manifest"]["manifest_sha256"],
        "splits": [
            {
                "scenario": r["scenario"],
                "label": r["split_label"],
                "counts": {s: r["counts"][s]["n_conditions"] for s in ("train", "val", "test")},
                "mono_trainable": r["mono"]["n_trainable_curves"],
                "assignment_sha256": r["assignment_sha256"],
            }
            for r in result["splits"]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
