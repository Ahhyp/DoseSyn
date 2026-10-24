#!/usr/bin/env python
"""Recompute leakage/integrity checks for every emitted M3 split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.splits.verify import verify_all  # noqa: E402
from dosesyn.utils.logging import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "dataset_primary_v1.yaml"))
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    logger = setup_logging(
        "dosesyn.verify",
        log_file=str(ROOT / "outputs" / "logs" / "03_verify_splits.log"),
    )
    report = verify_all(args.config, logger=logger)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    summary = {
        "overall": report["overall"],
        "n_splits": report["n_splits"],
        "n_passed": report["n_passed"],
        "failures": [
            {"scenario": s.get("scenario"), "dir": s.get("split_dir"), "failed": [c for c in s["checks"] if c["status"] != "PASS"]}
            for s in report["splits"]
            if not s["passed"]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
