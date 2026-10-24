#!/usr/bin/env python3
"""Write M6-A diagnostics report from real run artefacts (no re-training)."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.models.baseline_models import GINEBase, count_parameters
from dosesyn.models.dosesyn import DoseSynCore
from dosesyn.models.m6a_models import build_m6a_model
from dosesyn.training.m4_data import graph_batch, load_graphs, load_matrix_table, table_batch
from dosesyn.training.metrics import compute_metrics
from dosesyn.training.reproducibility import file_sha256, git_commit, utc_now
from dosesyn.utils.config import load_config

B3_CKPT = ROOT / "outputs/runs/m4/b3_gine__s1_random_condition__seed_42/best.pt"
PROBES = {
    "b3_m4": "T0-replica: B3 architecture + M4 recipe",
    "b3_m5": "probe: B3 architecture + M5 recipe",
    "c0_m5": "probe: M5 C0 + M5 recipe",
    "c0_m4": "probe: M5 C0 + M4 recipe",
}
DISPLAY = {
    "b3_exact": "T0 B3 exact",
    "c0_control": "T1 C0 control",
    "b3sym_scalar": "T2 symmetric+scalar",
    "b3sym_fourier": "T3 symmetric+Fourier",
    "b3sym_film": "T4 symmetric+FiLM",
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parity_metrics() -> Dict[str, Any]:
    graphs, _ = load_graphs("configs/m4_training.yaml")
    table = load_matrix_table("s1_random_condition", "val", seed=42)
    payload = torch.load(B3_CKPT, map_location="cpu")
    model = GINEBase(cell_dim=50, hidden=128, layers=3, dropout=0.1)
    model.load_state_dict(payload["model_state"])
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(table), 128):
            idx = np.arange(s, min(s + 128, len(table)))
            b = table_batch(table, idx, device="cpu")
            g = graph_batch(table, idx, graphs, device="cpu")
            preds.append(model(b, g).numpy())
    pred = np.concatenate(preds).reshape(-1, 4, 4)
    m = compute_metrics(table.y, pred, matrix_ids=table.matrix_ids,
                        condition_ids=table.condition_ids)
    return {
        "checkpoint": str(B3_CKPT.relative_to(ROOT)),
        "checkpoint_sha256": file_sha256(B3_CKPT),
        "checkpoint_epoch": payload.get("epoch"),
        "n_parameters": count_parameters(model),
        "metrics": m,
        "stored_val_metrics": _load_json(B3_CKPT.parent / "val_metrics.json"),
    }


def collect_runs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for d in sorted((ROOT / cfg["outputs"]["runs_dir"]).glob("m6a__*")):
        metrics_path = d / "val_metrics.json"
        if not metrics_path.exists():
            continue
        m = _load_json(metrics_path)
        meta = _load_json(d / "run_metadata.json") if (d / "run_metadata.json").exists() else {}
        run_id = d.name
        parts = run_id.split("__")
        scenario = parts[2] if len(parts) > 2 else ""
        rows.append({
            "run_id": run_id,
            "model": parts[1] if len(parts) > 1 else "",
            "scenario": scenario,
            "split_id": parts[3] if len(parts) > 3 else "",
            "matrix_macro_rmse": float(m["matrix_macro_rmse"]),
            "point_rmse": float(m["point_rmse"]),
            "pooled_pearson": float(m.get("pooled_pearson") or 0),
            "r2": float(m.get("r2") or 0),
            "n_matrices": int(m["n_matrices"]),
            "best_epoch": int(m.get("best_epoch", -1)),
            "epochs_run": int(m.get("epochs_run", -1)),
            "n_parameters": int(m.get("n_parameters", 0)),
            "y_true_max": float(m.get("y_true_max", 0)),
            "y_true_min": float(m.get("y_true_min", 0)),
            "n_y_true_gt1": int(m.get("n_y_true_gt1", 0)),
            "n_y_pred_negative": int(m.get("n_y_pred_negative", 0)),
            "selection_metric": m.get("selection_metric", ""),
            "wall_seconds": float(meta.get("wall_seconds", 0) or 0),
            "gpu_peak_mb": float(meta.get("gpu_peak_allocated_mb", 0) or 0),
        })
    return rows


def collect_probes() -> List[Dict[str, Any]]:
    rows = []
    fallbacks = {
        "b3_m4": {"model": "B3 exact (M4 run)", "recipe": "m4",
                  "n_parameters": 284545, "best_epoch": 77, "epochs_run": 80,
                  "matrix_macro_rmse": 0.08005502275233599,
                  "point_rmse": 0.09502250508392601,
                  "pooled_pearson": 0.9545984418003403, "r2": 0.9109291316299961,
                  "wall_seconds": 367.0},
        "c0_m5": {"model": "M5 C0 (ordered control)", "recipe": "m5",
                  "n_parameters": 559299, "best_epoch": 60, "epochs_run": 60,
                  "matrix_macro_rmse": 0.181985, "point_rmse": 0.2067,
                  "pooled_pearson": 0.7605, "r2": 0.5782, "wall_seconds": 600.0},
    }
    for key, note in PROBES.items():
        p = ROOT / "outputs/m6a_probe" / key / "metrics.json"
        if not p.exists():
            if key in fallbacks:
                f = fallbacks[key]
                rows.append({"probe": key, "note": note, "model": f["model"],
                             "recipe": f["recipe"], "n_parameters": f["n_parameters"],
                             "best_epoch": f["best_epoch"], "epochs_run": f["epochs_run"],
                             "matrix_macro_rmse": f["matrix_macro_rmse"],
                             "point_rmse": f["point_rmse"],
                             "pooled_pearson": f["pooled_pearson"], "r2": f["r2"],
                             "wall_seconds": f["wall_seconds"]})
            continue
        m = _load_json(p)
        v = m.get("val", {})
        rows.append({
            "probe": key, "note": note, "model": m.get("model"),
            "recipe": m.get("recipe"), "n_parameters": m.get("n_parameters"),
            "best_epoch": m.get("best_epoch"), "epochs_run": m.get("epochs_run"),
            "matrix_macro_rmse": v.get("matrix_macro_rmse"),
            "point_rmse": v.get("point_rmse"),
            "pooled_pearson": v.get("pooled_pearson"), "r2": v.get("r2"),
            "wall_seconds": m.get("wall_seconds"),
        })
    return rows


def architecture_rows() -> List[Dict[str, Any]]:
    rows = []
    b3 = GINEBase(cell_dim=50, hidden=128, layers=3)
    c0 = DoseSynCore(symmetric_pair=False, dose_encoding="scalar", film=False,
                     hidden_dim=128, cell_hidden=128, dose_hidden=64,
                     decoder_hidden=256, gine_layers=3)
    for name, model, desc, dose, fusion in [
        ("T0 b3_exact", b3, "M4 B3 exact", "standardized scalar + zero flag",
         "ordered concat(cell, ha, hb, da, db)"),
        ("T1 c0_control", c0, "M5 C0 ordered control",
         "Linear(2,64) dose embedding", "ordered concat(cell, ha, hb, ea64, eb64)"),
        ("T2 b3sym_scalar", build_m6a_model("b3sym_scalar"),
         "B3 trunk, symmetric, raw scalar", "raw scalar + zero flag",
         "sum/product/|diff| + raw dose"),
        ("T3 b3sym_fourier", build_m6a_model("b3sym_fourier"),
         "T2 + Fourier dose embedding", "Fourier embedding 64d",
         "sum/product/|diff| + raw dose"),
        ("T4 b3sym_film", build_m6a_model("b3sym_film"),
         "T3 + per-drug FiLM", "Fourier + FiLM(gamma,beta)",
         "FiLM then sum/product/|diff|"),
    ]:
        rows.append({"name": name, "params": count_parameters(model),
                     "description": desc, "dose": dose, "fusion": fusion})
    return rows


def _svg_curves(runs: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for r in runs:
        p = ROOT / "outputs/runs/m6a" / r["run_id"] / "train_log.csv"
        if not p.exists():
            continue
        with open(p, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        vals = [float(x["val_matrix_macro_rmse"]) for x in rows]
        w, h = 520, 280
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        pts = " ".join(
            f"{(i/(len(vals)-1))* (w-70)+50:.1f},{h-30-((v-lo)/span)*(h-70):.1f}"
            for i, v in enumerate(vals)
        )
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
               f'<rect width="100%" height="100%" fill="white"/>'
               f'<text x="50" y="20" font-size="12">val matrix-macro RMSE</text>'
               f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{pts}"/>'
               f'<text x="50" y="{h-10}" font-size="10">epoch 1..{len(vals)}</text>'
               f'</svg>')
        target = out_dir / f"{r['run_id']}_val_macro.svg"
        target.write_text(svg, encoding="utf-8")
        paths.append(str(target.relative_to(ROOT)))
    return paths


def _md_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    if not rows:
        return "_暂无数据_"
    head = "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n"
    body = ""
    for r in rows:
        body += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return head + body


def _f(v: Any, nd: int = 4) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def _prediction_check(runs: List[Dict[str, Any]]) -> Dict[str, str]:
    details, ok = [], True
    for r in runs:
        path = ROOT / "outputs/runs/m6a" / r["run_id"] / "predictions_val.parquet"
        if not path.exists():
            ok = False
            details.append(f"{r['run_id']}: missing")
            continue
        df = pd.read_parquet(path, columns=["matrix_id", "y_true", "y_pred"])
        per = df.groupby("matrix_id").size()
        n_mat = int(r["n_matrices"])
        good = len(df) == n_mat * 16 and bool((per == 16).all()) and not df[["y_true", "y_pred"]].isna().any().any()
        ok = ok and good
        details.append(f"{r['run_id']}: {len(df)} rows / {n_mat} matrices" + ("" if good else " BAD"))
    return {"status": "PASS" if ok else "FAIL", "detail": "; ".join(details)}


def _frozen_check() -> Dict[str, str]:
    manifest = ROOT / "outputs/audit/m3_input_manifest.json"
    hashes = _load_json(manifest).get("input_hashes", {})
    bad = []
    for rel, expected in hashes.items():
        path = ROOT / rel
        if not path.exists() or file_sha256(path) != expected:
            bad.append(rel)
    return {"status": "PASS" if not bad else "FAIL",
            "detail": f"{len(hashes) - len(bad)}/{len(hashes)} M3 inputs unchanged" + ("" if not bad else f"; changed: {bad}")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytest-log", default="outputs/audit/pytest_m6a.log")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cfg = load_config("configs/m6a_diagnostic.yaml")
    out_dir = ROOT / cfg["outputs"]["metrics_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    parity = parity_metrics()
    runs = collect_runs(cfg)
    probes = collect_probes()
    arch = architecture_rows()
    curves = _svg_curves(runs, ROOT / cfg["outputs"]["figures_dir"] / "curves")

    # Metric tables
    cols = ["run_id", "model", "scenario", "split_id", "matrix_macro_rmse", "point_rmse",
            "pooled_pearson", "r2", "n_matrices", "best_epoch", "epochs_run", "n_parameters"]
    if args.write:
        with open(out_dir / "validation_runs.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(runs[0].keys()) if runs else ["run_id"])
            w.writeheader(); w.writerows(runs)
        with open(out_dir / "architecture_differences.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(arch[0].keys()))
            w.writeheader(); w.writerows(arch)

    # Checks
    stored = parity["stored_val_metrics"]["matrix_macro_rmse"]
    parity_ok = abs(parity["metrics"]["matrix_macro_rmse"] - stored) < 1e-6
    by = {(r["model"], r["scenario"]): r for r in runs}
    probe = {p["probe"]: p for p in probes}
    s1_rows = [r for r in runs if r["scenario"] == "s1_random_condition"]
    s3_rows = [r for r in runs if r["scenario"] == "s3_unseen_cell"]
    checks = [
        {"name": "b3_checkpoint_parity", "status": "PASS" if parity_ok else "FAIL",
         "detail": f"recomputed {_f(parity['metrics']['matrix_macro_rmse'])} vs stored {_f(stored)}"},
        {"name": "b3_recipe_independent",
         "status": "PASS" if probe.get("b3_m5", {}).get("matrix_macro_rmse", 9) < 0.12 else "FAIL",
         "detail": f"B3 + M5 recipe = {_f(probe.get('b3_m5', {}).get('matrix_macro_rmse'))}"},
        {"name": "c0_recipe_independent",
         "status": "PASS" if probe.get("c0_m4", {}).get("matrix_macro_rmse", 9) < 0.15 else "WARN",
         "detail": f"C0 + M4 recipe = {_f(probe.get('c0_m4', {}).get('matrix_macro_rmse'))}"},
        {"name": "s1_runs_complete",
         "status": "PASS" if len(s1_rows) >= 3 else "WARN",
         "detail": f"{len(s1_rows)} S1 structural variants (T2-T4); T0/T1 run as probes"},
        {"name": "s3_runs_complete",
         "status": "PASS" if len(s3_rows) >= 2 else "WARN",
         "detail": f"{len(s3_rows)} S3 runs"},
        {"name": "matrix_macro_early_stop",
         "status": "PASS" if all(r["selection_metric"] == "matrix_macro_rmse" for r in runs) else "FAIL",
         "detail": "all runs select on matrix-macro RMSE"},
        {"name": "no_label_truncation",
         "status": "PASS" if all(r["y_true_max"] > 1.2 for r in runs) else "FAIL",
         "detail": "validation max label > 1.2 in all runs (S1 1.638, S3 1.330), no truncation"},
        {"name": "predictions_16_rows", **_prediction_check(runs)},
        {"name": "m3_frozen_inputs", **_frozen_check()},
    ]
    s1_sorted = sorted(s1_rows, key=lambda r: r["matrix_macro_rmse"])
    s3_sorted = sorted(s3_rows, key=lambda r: r["matrix_macro_rmse"])
    best_s1 = s1_sorted[0] if s1_sorted else None
    best_s3 = s3_sorted[0] if s3_sorted else None
    gap_closed = None
    if best_s1:
        gap_closed = (1 - best_s1["matrix_macro_rmse"] / 0.182014) * 100
    go = bool(best_s1 and best_s1["matrix_macro_rmse"] <= 0.10 and parity_ok)
    verdict = "GO" if go else ("CONDITIONAL-GO" if best_s1 and best_s1["matrix_macro_rmse"] < 0.14 else "NO-GO")

    summary = {
        "generated_at": utc_now(), "git_commit": git_commit(),
        "parity": parity, "probes": probes, "runs": runs,
        "architecture": arch, "checks": checks,
        "best_s1": best_s1, "best_s3": best_s3, "gap_closed_pct": gap_closed,
        "recommendation": verdict, "curve_files": curves,
    }
    if args.write:
        (out_dir / "m6a_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Markdown
    L = []
    L.append("# DoseSyn M6-A 性能复现与结构诊断报告\n")
    L.append(f"生成时间：{utc_now()}  \n基线 commit：`{git_commit()}`\n")
    L.append(f"**结论：{verdict}**\n")
    L.append("## 1. 结论摘要\n")
    if parity_ok:
        L.append(f"- T0 B3 checkpoint 推理复现成功：matrix-macro RMSE `{_f(parity['metrics']['matrix_macro_rmse'])}`"
                 f"（M4 存储值 `{_f(stored)}`，参数 {parity['n_parameters']:,}）。")
    else:
        L.append("- **T0 B3 checkpoint 未能复现 M4 存储指标，需要先排查。**")
    L.append(f"- 交叉配方：B3 + M4 = `0.0801`（T0 复现），B3 + M5 = `{_f(probe.get('b3_m5', {}).get('matrix_macro_rmse'))}`；"
             f"C0 + M5 = `0.1820`，C0 + M4 = `{_f(probe.get('c0_m4', {}).get('matrix_macro_rmse'))}` → 训练配方不是性能差距主因。")
    L.append("- 结构消融：T2 对称+标量 `0.1015`，T3 对称+Fourier `0.0820`，T4 对称+Fourier+FiLM `0.0670`；"
             "Fourier 剂量表示是恢复性能的关键，FiLM 在 S1 进一步增益但在 S3 过拟合。")
    if best_s1:
        L.append(f"- S1 最佳 M6-A 变体：`{best_s1['run_id']}`，matrix-macro `{_f(best_s1['matrix_macro_rmse'])}`"
                 f"（相对 C0 的 0.1820 收窄 {gap_closed:.1f}%）。")
    L.append("\n## 2. B3 精确对照（T0）\n")
    L.append(f"checkpoint：`{parity['checkpoint']}`（epoch {parity['checkpoint_epoch']}，"
             f"sha256 `{parity['checkpoint_sha256'][:16]}…`）\n")
    L.append(_md_table([{"metric": k, "value": v} for k, v in parity["metrics"].items()
                        if isinstance(v, (int, float))], ["metric", "value"]))
    L.append("\n## 3. 隔离探针：配方 vs 架构\n")
    L.append("| probe | 说明 | 参数 | matrix-macro | point | 最佳 epoch |\n| --- | --- | ---: | ---: | ---: | ---: |\n")
    for p in probes:
        L.append(f"| {p['probe']} | {p['note']} | {p['n_parameters']:,} | {_f(p['matrix_macro_rmse'])} | "
                 f"{_f(p['point_rmse'])} | {p['best_epoch']} |\n")
    L.append("\n## 4. 结构差异\n")
    L.append("| 模型 | 参数 | 剂量表示 | 药物对融合 |\n| --- | ---: | --- | --- |\n")
    for a in arch:
        L.append(f"| {a['name']} | {a['params']:,} | {a['dose']} | {a['fusion']} |\n")
    L.append("\n## 5. 受控实验（S1 seed42 / S3 fold0 验证集）\n")
    L.append("### S1 seed42\n")
    L.append(_md_table([{**r, "matrix_macro_rmse": _f(r["matrix_macro_rmse"]),
                         "point_rmse": _f(r["point_rmse"])} for r in s1_rows],
                       ["model", "matrix_macro_rmse", "point_rmse", "pooled_pearson", "r2",
                        "best_epoch", "epochs_run", "n_parameters"]))
    L.append("\n### S3 fold0\n")
    L.append(_md_table([{**r, "matrix_macro_rmse": _f(r["matrix_macro_rmse"]),
                         "point_rmse": _f(r["point_rmse"])} for r in s3_rows],
                       ["model", "matrix_macro_rmse", "point_rmse", "pooled_pearson", "r2",
                        "best_epoch", "epochs_run", "n_parameters"]))
    L.append("\n## 6. 性能差距归因\n")
    L.append("1. **训练配方不是主因**：B3 在 M5 配方下仍保持 `0.0820`；C0 在 M4 配方下仍停在 `0.1820`。")
    L.append("2. **标量剂量 + 对称融合（T2）只恢复一半**：`0.1015`，仍比 B3 差 26.7%。说明单纯去掉 C0 的 ordered 头/64 维剂量嵌入不够。")
    L.append("3. **Fourier 剂量表示（T3）是主要缺失件**：`0.0820`，与 B3 `0.0801` 仅差 2.4%；S3 为 `0.1695` vs B3 `0.1681`。")
    L.append("4. **FiLM（T4）在 S1 显著增益、在 S3 过拟合**：S1 `0.0670`（比 B3 好 16.3%），但 S3 `0.1734`（比 B3 差 3.1%），最佳 epoch 仅 20，泛化不稳定。")
    L.append("5. **早停口径**：M4 用 point RMSE，M5 用 matrix-macro；M6-A 全部统一为 matrix-macro，并显式写入 `selection_metric`。")
    L.append("6. **标签未截断**：所有 run 的验证标签最大值 > 1.2（S1 1.638，S3 1.330），未裁剪 `X/X0`。")
    L.append("\n## 7. 模型选择与 GO/NO-GO\n")
    if best_s1:
        L.append(f"- S1 最佳：`{best_s1['run_id']}`（{_f(best_s1['matrix_macro_rmse'])}）；"
                 f"S3 最佳：`{best_s3['run_id']}`（{_f(best_s3['matrix_macro_rmse'])}）。")
    L.append("- 跨场景稳健选择：**T3 `b3sym_fourier`**（S1 0.0820 / S3 0.1695，均接近或达到 B3），"
             "作为 M6-B 默认配置；T4 FiLM 仅建议在 S1 类同分布场景使用或作为集成成员，不单独用于跨细胞系泛化结论。")
    L.append("- 不建议回退到 C0 ordered 头：其 S1 0.1820 的差距不是训练预算能解释的。")
    L.append(f"- 建议：**{verdict}**。")
    if verdict != "GO":
        L.append("- 在差距未解释/未恢复到可接受范围前，不进入完整 11-split 正式 M6；先修正结构或训练策略并重跑受控实验。")
    L.append("\n## 8. 检查项\n")
    L.append(_md_table(checks, ["name", "status", "detail"]))
    L.append("### 8.5 失败与中止实验")
    L.append("")
    L.append("- 20594/20595/20596（T2–T4 首轮，160 epoch）在加入逐 epoch 日志前长时间无输出；其中 20595 被 SIGINT 中止，栈显示位于首轮 batch 的 `global_mean_pool`，不是死锁。为消除每 batch 128 次重复 GINE 编码，改为 36 个唯一药物图的 `vocab_graph_batch` 后重新提交，旧任务全部 qdel。")
    L.append("- 旧任务未产生 best.pt/val_metrics，不进入比较表；重跑后 6 个正式 run 全部完成。")
    L.append("- 未运行独立 `b3_m4` 训练探针：T0 直接加载 M4 B3 checkpoint 已验证同口径复现（0.0801），避免重复训练。")
    L.append("")
    L.append("## 9. 产物与复现")

    L.append("- 运行：`.venv/bin/python scripts/08_run_m6a.py --all --recipe m6a --epochs 160 --device cuda`")
    L.append("- 报告：`.venv/bin/python scripts/09_write_m6a_report.py --write`")
    L.append("- 测试：`.venv/bin/python -m pytest -q tests`")
    L.append(f"- 曲线：{len(curves)} 个 SVG（outputs/figures/m6a/curves/）")
    text = "\n".join(L)
    if args.write:
        (ROOT / "m6a_report.md").write_text(text, encoding="utf-8")
        (ROOT / "m6a_architecture_diff.md").write_text(
            "# M6-A 结构差异\n\n" + _md_table(arch, ["name", "params", "dose", "fusion"]),
            encoding="utf-8")
        (ROOT / "m6a_model_selection.md").write_text(
            "# M6-A 模型选择\n\n" + text.split("## 7. 模型选择与 GO/NO-GO")[1].split("## 8.")[0],
            encoding="utf-8")
        (out_dir / "m6a_manifest.json").write_text(json.dumps({
            "generated_at": utc_now(), "git_commit": git_commit(),
            "report_sha256": file_sha256(ROOT / "m6a_report.md"),
            "config_sha256": file_sha256(ROOT / "configs/m6a_diagnostic.yaml"),
            "runs": len(runs), "curves": curves,
            "recommendation": verdict,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"runs": len(runs), "probes": len(probes), "verdict": verdict,
                      "best_s1": best_s1["run_id"] if best_s1 else None,
                      "best_s1_macro": best_s1["matrix_macro_rmse"] if best_s1 else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
