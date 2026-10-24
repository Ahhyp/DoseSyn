#!/usr/bin/env python3
"""M6-B validation-stage model selection (no test labels)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.utils.config import load_config
from dosesyn.training.reproducibility import file_sha256, git_commit, utc_now

VARIANTS = [
    "b1_t3_fourier", "b2_t3_mono", "b3_t3_mono_boundary",
    "b4_t3_mono_boundary_smooth", "b5_t3_film_reg",
]
LABEL = {
    "b1_t3_fourier": "B1 T3 Fourier",
    "b2_t3_mono": "B2 T3 + Mono",
    "b3_t3_mono_boundary": "B3 T3 + Mono + Boundary",
    "b4_t3_mono_boundary_smooth": "B4 T3 + Mono + Boundary + Smooth",
    "b5_t3_film_reg": "B5 T3 + Regularized FiLM",
}
SCEN = {"s1_random_condition": "S1 seed42", "s3_unseen_cell": "S3 fold0"}


def _rows(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = ROOT / cfg["outputs"]["runs_dir"]
    out = []
    for v in VARIANTS:
        for scenario in SCEN:
            split = "seed_42" if scenario.startswith("s1") else "fold_0"
            run_id = f"m6b__{v}__{scenario}__{split}"
            path = runs / run_id / "val_metrics.json"
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            m = d["val_metrics"]
            diag = d.get("diagnostics", {})
            out.append({
                "variant": v, "variant_label": LABEL[v], "scenario": scenario,
                "scenario_label": SCEN[scenario], "run_id": run_id,
                "matrix_macro_rmse": m["matrix_macro_rmse"],
                "point_rmse": m["point_rmse"], "point_mae": m["point_mae"],
                "pooled_pearson": m["pooled_pearson"], "r2": m["r2"],
                "swap_max_abs": diag.get("swap_max_abs"),
                "E_boundary_mean": diag.get("E_boundary_mean"),
                "mono_rmse": diag.get("mono_rmse"),
                "smoothness_loss": diag.get("smoothness_loss"),
                "n_parameters": d["n_parameters"], "best_epoch": d["best_epoch"],
                "epochs_run": d["epochs_run"], "wall_seconds": d["wall_seconds"],
            })
    return out


def select(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by = {(r["variant"], r["scenario"]): r for r in rows}
    summary = []
    for v in VARIANTS:
        s1 = by.get((v, "s1_random_condition"))
        s3 = by.get((v, "s3_unseen_cell"))
        if not s1 or not s3:
            continue
        summary.append({
            "variant": v, "label": LABEL[v],
            "s1_macro": s1["matrix_macro_rmse"], "s3_macro": s3["matrix_macro_rmse"],
            "mean_macro": 0.5 * (s1["matrix_macro_rmse"] + s3["matrix_macro_rmse"]),
            "params": s1["n_parameters"],
        })
    if not summary:
        raise SystemExit("no complete validation runs found")
    base = next((s for s in summary if s["variant"] == "b1_t3_fourier"), None)
    best = min(summary, key=lambda s: s["mean_macro"])
    chosen = best
    reason = f"{best['label']} has the lowest mean S1/S3 validation macro RMSE."
    if base and best["variant"] != "b1_t3_fourier":
        gain = base["mean_macro"] - best["mean_macro"]
        s3_gap = best["s3_macro"] - base["s3_macro"]
        if gain < 0.001 or s3_gap > 0.002:
            chosen = base
            reason = (f"{best['label']} best mean but gain {gain:.4f} < 0.001 or S3 gap "
                      f"{s3_gap:+.4f} > 0.002; parsimony/robustness selects B1 T3.")
        else:
            reason = (f"{best['label']} improves mean by {gain:.4f} and S3 gap is "
                      f"{s3_gap:+.4f}; selected for formal M6-B.")
    return {
        "selected": chosen["variant"], "selected_label": chosen["label"],
        "criterion": "mean of S1 seed42 and S3 fold0 validation matrix-macro RMSE; "
                     "B1 retained unless a candidate improves mean >=0.001 and S3 worsens <=0.002",
        "reason": reason, "summary": summary, "rows": rows,
        "generated_at": utc_now(), "git_commit": git_commit(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m6b_experiments.yaml")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rows = _rows(cfg)
    if len(rows) < 2:
        raise SystemExit(f"only {len(rows)} validation runs found; run 11_run_m6b.py first")
    result = select(rows)
    if args.write:
        out = ROOT / cfg["outputs"]["metrics_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "validation_runs.csv").write_text(
            pd.DataFrame(rows).to_csv(index=False), encoding="utf-8")
        audit = ROOT / "docs/audit"
        audit.mkdir(parents=True, exist_ok=True)
        (audit / "m6b_model_selection.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        lines = ["# M6-B 验证集模型选择", "",
                 f"生成时间：{result['generated_at']}  ",
                 f"基线 commit：`{result['git_commit']}`", "",
                 f"**选择：{result['selected_label']} (`{result['selected']}`)**", "",
                 f"理由：{result['reason']}", "",
                 "| 变体 | S1 seed42 | S3 fold0 | 均值 | 参数量 |",
                 "| --- | ---: | ---: | ---: | ---: |"]
        for s in result["summary"]:
            lines.append(f"| {s['label']} | {s['s1_macro']:.4f} | {s['s3_macro']:.4f} | "
                         f"{s['mean_macro']:.4f} | {s['params']:,} |")
        lines += ["", "选择只使用验证集，未读取测试标签。", ""]
        (ROOT / "m6b_model_selection.md").write_text("\n".join(lines), encoding="utf-8")
        print(json.dumps({k: result[k] for k in ("selected", "reason", "summary")},
                         indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
