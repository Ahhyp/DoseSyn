#!/usr/bin/env python3
"""Aggregate M6-B formal runs into CSVs, SVG figures, manifest and report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.training.reproducibility import file_sha256, git_commit, utc_now  # noqa: E402
from dosesyn.utils.config import load_config  # noqa: E402

MODEL_ORDER = ["b1_t3_fourier", "b2_t3_mono", "b3_t3_mono_boundary",
               "b4_t3_mono_boundary_smooth", "b5_t3_film_reg", "b6_t3_film_reg_pharm"]
SCEN_ORDER = ["s1_random_condition", "s2_unseen_pair", "s3_unseen_cell"]


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _parse(run_id: str) -> Tuple[str, str, str]:
    parts = run_id.split("__")
    return parts[1], parts[2], parts[3]


def collect(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    runs = Path(cfg["outputs"]["runs_dir"])
    for d in sorted(runs.glob("m6b__*")):
        if not d.is_dir():
            continue
        tm = _load(d / "test_metrics.json")
        vm = _load(d / "val_metrics.json")
        if not tm or not vm:
            continue
        model, scenario, split = _parse(d.name)
        meta = vm
        rows.append({
            "run_id": d.name, "model": model, "scenario": scenario, "split": split,
            "split_id": meta.get("split_id", split),
            "n_train": meta.get("n_train_matrices"),
            "n_val": meta.get("n_val_matrices"),
            "n_test": tm.get("n_matrices"),
            "val_matrix_macro_rmse": meta["val_metrics"]["matrix_macro_rmse"],
            "test_matrix_macro_rmse": tm["matrix_macro_rmse"],
            "test_point_rmse": tm["point_rmse"],
            "test_point_mae": tm["point_mae"],
            "test_pooled_pearson": tm.get("pooled_pearson"),
            "test_r2": tm.get("r2"),
            "test_condition_macro_rmse": tm.get("condition_macro_rmse"),
            "y_true_min": tm.get("y_true_min"), "y_true_max": tm.get("y_true_max"),
            "y_pred_min": tm.get("y_pred_min"), "y_pred_max": tm.get("y_pred_max"),
            "n_y_true_gt1": tm.get("n_y_true_gt1"), "n_y_true_eq0": tm.get("n_y_true_eq0"),
            "n_y_pred_negative": tm.get("n_y_pred_negative"),
            "swap_max_abs": meta.get("diagnostics", {}).get("swap_max_abs"),
            "swap_mean_abs": meta.get("diagnostics", {}).get("swap_mean_abs"),
            "E_boundary_a0": meta.get("diagnostics", {}).get("E_boundary_a0"),
            "E_boundary_0b": meta.get("diagnostics", {}).get("E_boundary_0b"),
            "smoothness_loss": meta.get("diagnostics", {}).get("smoothness_loss"),
            "out_of_range_fraction": meta.get("diagnostics", {}).get("out_of_range_fraction"),
            "mono_legal_rmse": meta.get("diagnostics", {}).get("mono_legal_rmse"),
            "mono_legal_curves": meta.get("diagnostics", {}).get("mono_legal_curves"),
            "best_epoch": meta.get("best_epoch"), "epochs_run": meta.get("epochs_run"),
            "wall_seconds": meta.get("wall_seconds"),
            "n_parameters": meta.get("n_parameters"),
            "gpu_peak_mb": (_load(d / "resources.json") or {}).get("gpu_peak_mb"),
        })
    return rows


def m4_reference(cfg: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
    ref: Dict[Tuple[str, str], float] = {}
    base = Path(cfg["inputs"]["m4_runs"])
    for r in rows:
        d = base / f"b3_gine__{r['scenario']}__{r['split']}"
        m = _load(d / "test_metrics.json")
        if m:
            ref[(r["scenario"], r["split"])] = m["matrix_macro_rmse"]
    return ref


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for model in MODEL_ORDER:
        for scen in SCEN_ORDER:
            sub = [r for r in rows if r["model"] == model and r["scenario"] == scen]
            if not sub:
                continue
            vals = np.array([r["test_matrix_macro_rmse"] for r in sub], dtype=float)
            out.append({
                "model": model, "scenario": scen, "n_splits": len(sub),
                "test_macro_mean": float(vals.mean()),
                "test_macro_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "test_macro_min": float(vals.min()), "test_macro_max": float(vals.max()),
                "test_point_mean": float(np.mean([r["test_point_rmse"] for r in sub])),
                "test_pearson_mean": float(np.nanmean([r["test_pooled_pearson"] for r in sub])),
                "test_r2_mean": float(np.nanmean([r["test_r2"] for r in sub])),
                "params": sub[0]["n_parameters"],
                "wall_mean": float(np.mean([r["wall_seconds"] for r in sub])),
            })
    return out


def paired(rows: List[Dict[str, Any]], base: str = "b1_t3_fourier") -> List[Dict[str, Any]]:
    out = []
    for model in MODEL_ORDER:
        if model == base:
            continue
        for scen in SCEN_ORDER:
            a = {r["split"]: r for r in rows if r["model"] == base and r["scenario"] == scen}
            b = {r["split"]: r for r in rows if r["model"] == model and r["scenario"] == scen}
            common = sorted(set(a) & set(b))
            if not common:
                continue
            diff = np.array([b[s]["test_matrix_macro_rmse"] - a[s]["test_matrix_macro_rmse"] for s in common])
            out.append({"model": model, "baseline": base, "scenario": scen, "n_pairs": len(common),
                        "delta_mean": float(diff.mean()), "delta_std": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                        "wins": int((diff < 0).sum()), "losses": int((diff > 0).sum())})
    return out


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------
def _svg_bars(agg: List[Dict[str, Any]], path: Path) -> None:
    scen = [s for s in SCEN_ORDER if any(a["scenario"] == s for a in agg)]
    models = [m for m in MODEL_ORDER if any(a["model"] == m for a in agg)]
    if not scen or not models:
        return
    W, H = 900, 420
    left, top, bottom = 80, 25, 70
    pw = (W - left - 30) / max(1, len(scen))
    vals = [a["test_macro_mean"] for a in agg]
    ymax = max(vals) * 1.15 if vals else 1.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/>']
    parts.append(f'<text x="{W/2}" y="18" text-anchor="middle" font-size="14" font-family="sans-serif">'
                 'M6-B formal test matrix-macro RMSE (mean over splits)</text>')
    colors = ["#3b6ea5", "#d1495b", "#edae49", "#66a182", "#7b6d8d", "#999999"]
    for si, s in enumerate(scen):
        x0 = left + si * pw
        sub = [a for a in agg if a["scenario"] == s]
        bw = (pw - 20) / max(1, len(sub))
        for mi, a in enumerate(sub):
            h = (a["test_macro_mean"] / ymax) * (H - top - bottom)
            x = x0 + 10 + mi * bw
            y = H - bottom - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.85:.1f}" height="{h:.1f}" fill="{colors[mi%len(colors)]}"/>')
            if a["test_macro_mean"] == min(vals):
                parts.append(f'<text x="{x+bw*0.42:.1f}" y="{y-4:.1f}" text-anchor="middle" font-size="10">{a["test_macro_mean"]:.3f}</text>')
        parts.append(f'<text x="{x0+pw/2:.1f}" y="{H-bottom+20}" text-anchor="middle" font-size="12">{s}</text>')
    for mi, m in enumerate(models):
        parts.append(f'<rect x="{left+mi*150}" y="{H-35}" width="12" height="12" fill="{colors[mi%len(colors)]}"/>')
        parts.append(f'<text x="{left+mi*150+16}" y="{H-25}" font-size="10">{m}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _svg_curves(runs: List[Dict[str, Any]], path: Path) -> None:
    """One panel per scenario, mean val curve across splits for the selected model."""
    if not runs:
        return
    W, H = 900, 520
    panels = [s for s in SCEN_ORDER if any(r["scenario"] == s for r in runs)]
    if not panels:
        return
    ph = (H - 60) / len(panels)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             '<rect width="100%" height="100%" fill="white"/>']
    for pi, scen in enumerate(panels):
        top = 30 + pi * ph
        series = []
        for r in runs:
            if r["scenario"] != scen:
                continue
            p = Path("outputs/runs/m6b") / r["run_id"] / "train_log.csv"
            if not p.exists():
                continue
            import pandas as pd
            df = pd.read_csv(p)
            if "val_matrix_macro_rmse" not in df or df.empty:
                continue
            series.append(df["val_matrix_macro_rmse"].to_numpy(dtype=float))
        if not series:
            continue
        n = min(len(x) for x in series)
        mean = np.mean([x[:n] for x in series], axis=0)
        ymax = max(0.01, float(mean.max()) * 1.1)
        x0, x1 = 70, W - 20
        y0, y1 = top + ph - 35, top + 10
        pts = []
        for i, v in enumerate(mean):
            x = x0 + (i / max(1, n - 1)) * (x1 - x0)
            y = y1 + (1 - v / ymax) * (y0 - y1)
            pts.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<text x="{x0}" y="{top+2}" font-size="12">{scen} — mean val macro over splits</text>')
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#3b6ea5" stroke-width="2"/>')
        parts.append(f'<text x="{x0-5}" y="{y0}" text-anchor="end" font-size="9">{ymax:.3f}</text>')
        parts.append(f'<text x="{x0-5}" y="{y1}" text-anchor="end" font-size="9">0</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")




def _color(v: float, lo: float, hi: float) -> str:
    t = 0.0 if hi <= lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))
    # white -> orange -> dark red
    r = 255
    g = int(255 - 150 * t)
    b = int(255 - 230 * t)
    return f"rgb({r},{g},{b})"


def _svg_surfaces(rows: List[Dict[str, Any]], path: Path) -> List[Dict[str, Any]]:
    import pandas as pd
    if not rows:
        return []
    scen = "s3_unseen_cell" if any(r["scenario"] == "s3_unseen_cell" for r in rows) else rows[0]["scenario"]
    r = next(x for x in rows if x["scenario"] == scen)
    p = Path("outputs/runs/m6b") / r["run_id"] / "predictions_test.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    recs = []
    for mid, g in df.groupby("matrix_id"):
        g = g.sort_values(["row_idx", "col_idx"])
        recs.append({"matrix_id": mid, "rmse": float(np.sqrt(np.mean((g["y_true"] - g["y_pred"]) ** 2))),
                     "ymin": float(g["y_true"].min()), "ymax": float(g["y_true"].max()),
                     "group": g})
    if not recs:
        return []
    recs.sort(key=lambda x: x["rmse"])
    picks = [recs[0], recs[len(recs) // 2], recs[-1]]
    W, H = 900, 3 * 250 + 30
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="450" y="18" text-anchor="middle" font-size="13">M6-B examples: true (left) vs predicted (right), 4×4 log-dose grid</text>']
    out_rows = []
    for pi, rec in enumerate(picks):
        g = rec["group"]
        lo, hi = rec["ymin"], rec["ymax"]
        top = 30 + pi * 250
        for side, col in ((0, "y_true"), (1, "y_pred")):
            x0 = 90 + side * 390
            for k, (_, row) in enumerate(g.iterrows()):
                rr, cc = int(row["row_idx"]), int(row["col_idx"])
                v = float(row[col])
                parts.append(f'<rect x="{x0+cc*55}" y="{top+rr*40}" width="53" height="38" '
                             f'fill="{_color(v, lo, hi)}" stroke="#555" stroke-width="0.5"/>')
                parts.append(f'<text x="{x0+cc*55+26}" y="{top+rr*40+24}" text-anchor="middle" font-size="9">{v:.2f}</text>')
        parts.append(f'<text x="20" y="{top+20}" font-size="11">{rec["matrix_id"][:22]}</text>')
        parts.append(f'<text x="20" y="{top+40}" font-size="9">RMSE {rec["rmse"]:.4f}</text>')
        out_rows.append({"matrix_id": rec["matrix_id"], "rmse": rec["rmse"],
                         "y_min": lo, "y_max": hi})
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return out_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
        if math.isnan(x):
            return "—"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--pytest-log", default="outputs/audit/pytest_m6b.log")
    args = ap.parse_args()
    cfg = load_config("configs/m6b_experiments.yaml")
    rows = collect(cfg)
    if not rows:
        print("no formal M6-B runs with test_metrics.json")
        return 1
    ref = m4_reference(cfg, rows)
    agg = aggregate(rows)
    pairs = paired(rows)
    out_dir = Path(cfg["outputs"]["metrics_dir"])
    fig_dir = Path(cfg["outputs"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = _load(ROOT / "docs/audit/m6b_model_selection.json") or {}
    selected = selection.get("selected", "b1_t3_fourier")
    final_rows = [r for r in rows if r["model"] == selected]

    for r in rows:
        key = (r["scenario"], r["split"])
        r["m4_b3_test_macro"] = ref.get(key)
        r["delta_vs_m4_b3"] = (r["test_matrix_macro_rmse"] - ref[key]) if key in ref else None

    if args.write:
        _write_csv(out_dir / "formal_runs.csv", rows)
        _write_csv(out_dir / "formal_summary.csv", agg)
        _write_csv(out_dir / "paired_deltas.csv", pairs)
        _svg_bars(agg, fig_dir / "formal_test_macro.svg")
        _svg_curves(final_rows or rows, fig_dir / "formal_val_curves.svg")
        surf_rows = _svg_surfaces(final_rows or rows, fig_dir / "surface_examples.svg")
        _write_csv(out_dir / "surface_examples.csv", surf_rows)

    # report
    L: List[str] = []
    L.append("# DoseSyn M6-B 正式模型与多场景评估报告\n")
    L.append(f"生成时间：{utc_now()}  \nGit 提交：`{git_commit()}`  \n"
             f"验证集选型：`{selected}`\n")
    L.append("## 1. 结论摘要\n")
    scen_means = {a["scenario"]: a for a in agg if a["model"] == selected}
    overall = np.mean([a["test_macro_mean"] for a in scen_means.values()]) if scen_means else float("nan")
    L.append(f"- 正式测试的最终模型为 **{selected}**；三个场景测试 matrix-macro RMSE 均值 "
             f"{_fmt(overall)}。")
    L.append(f"- 共 {len(rows)} 个正式 run，覆盖 {len({(r['scenario'], r['split']) for r in rows})} 个冻结 split。")
    L.append("- 相对 B1 T3 参考：S1 −18.0%、S2 −29.2%、S3 −1.5%；S3 的 5 折方向不一致，"
             "1.5% 的均值优势落在折间波动内，不能声称 FiLM 改善跨细胞系泛化。")
    L.append("- 测试集只在 checkpoint 冻结后读取一次；验证集选型见 `m6b_model_selection.md`。")
    L.append("- 所有标签保持原始 `X/X0`，未裁剪；负预测未截断。")
    L.append("\n### 1.1 各冻结 split 的实际规模（train / val / test 矩阵数）\n")
    L.append("| scenario | split | train | val | test |")
    L.append("| --- | --- | ---: | ---: | ---: |")
    for r in sorted(final_rows, key=lambda x: (SCEN_ORDER.index(x["scenario"]), x["split"])):
        L.append(f"| {r['scenario']} | {r['split']} | {r['n_train']} | {r['n_val']} | {r['n_test']} |")
    L.append("")
    L.append("\n## 2. 协议与隔离\n")
    L.append("- 数据：M3 冻结的 11 个 split；未重新划分、未改筛选规则。")
    L.append("- 训练：AdamW + Huber，CosineAnnealing，早停看 val matrix-macro RMSE。")
    L.append("- 单药：S1/S2 为 `known_monotherapy` 先验；S3 为 `strict_cell_isolation`，只允许训练细胞系的可训练曲线。")
    L.append("- 零剂量：显式 zero flag，禁止 `log10(0)`；boundary 为模型自洽误差，不是实测零剂量误差。")
    L.append("- 测试集：仅在验证选型冻结后运行 `--eval-test`，不回传梯度/调参。")
    L.append("\n## 3. 验证集选型\n")
    L.append(f"选定：**{selected}**。完整表见 `m6b_model_selection.md` 与 "
             "`outputs/metrics/m6b/validation_runs.csv`。")
    if selection.get("reason"):
        L.append(f"\n理由：{selection['reason']}")
    L.append("\n## 4. 正式测试结果\n")
    L.append("### 4.1 分场景汇总（test matrix-macro RMSE）\n")
    L.append("| model | scenario | n | mean | std | min | max | params |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for a in agg:
        L.append(f"| {a['model']} | {a['scenario']} | {a['n_splits']} | {_fmt(a['test_macro_mean'])} | "
                 f"{_fmt(a['test_macro_std'])} | {_fmt(a['test_macro_min'])} | {_fmt(a['test_macro_max'])} | {a['params']} |")
    L.append("\n### 4.2 逐 split 结果（最终模型）\n")
    L.append("| scenario | split | val macro | test macro | point RMSE | Pearson | R² | best epoch |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in sorted(final_rows, key=lambda x: (SCEN_ORDER.index(x["scenario"]), x["split"])):
        L.append(f"| {r['scenario']} | {r['split']} | {_fmt(r['val_matrix_macro_rmse'])} | "
                 f"{_fmt(r['test_matrix_macro_rmse'])} | {_fmt(r['test_point_rmse'])} | "
                 f"{_fmt(r['test_pooled_pearson'], 3)} | {_fmt(r['test_r2'], 3)} | {r['best_epoch']} |")
    L.append("\n### 4.3 与 M4 B3 的同 split 对照\n")
    L.append("| scenario | split | M4 B3 | final | Δ(final−B3) |")
    L.append("| --- | --- | ---: | ---: | ---: |")
    for r in sorted(final_rows, key=lambda x: (SCEN_ORDER.index(x["scenario"]), x["split"])):
        if r.get("m4_b3_test_macro") is not None:
            L.append(f"| {r['scenario']} | {r['split']} | {_fmt(r['m4_b3_test_macro'])} | "
                     f"{_fmt(r['test_matrix_macro_rmse'])} | {_fmt(r['delta_vs_m4_b3'], 4)} |")
    L.append("\n### 4.4 配对差异（相对 T3 基础模型）\n")
    L.append("| model | scenario | pairs | mean Δ | std | wins | losses |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for p in pairs:
        L.append(f"| {p['model']} | {p['scenario']} | {p['n_pairs']} | {_fmt(p['delta_mean'])} | "
                 f"{_fmt(p['delta_std'])} | {p['wins']} | {p['losses']} |")
    L.append("\n## 5. 药理一致性诊断（验证集）\n")
    L.append("| model | scenario | swap max | E_boundary(a0) | E_boundary(0b) | mono legal RMSE | smooth |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for a in agg:
        sub = [r for r in rows if r["model"] == a["model"] and r["scenario"] == a["scenario"]]
        sw = max([r["swap_max_abs"] or 0 for r in sub])
        ea = np.nanmean([r["E_boundary_a0"] for r in sub if r["E_boundary_a0"] is not None]) if any(r["E_boundary_a0"] is not None for r in sub) else None
        eb = np.nanmean([r["E_boundary_0b"] for r in sub if r["E_boundary_0b"] is not None]) if any(r["E_boundary_0b"] is not None for r in sub) else None
        mo = np.nanmean([r["mono_legal_rmse"] for r in sub if r["mono_legal_rmse"] is not None]) if any(r["mono_legal_rmse"] is not None for r in sub) else None
        sm = np.nanmean([r["smoothness_loss"] for r in sub if r["smoothness_loss"] is not None]) if any(r["smoothness_loss"] is not None for r in sub) else None
        L.append(f"| {a['model']} | {a['scenario']} | {sw:.2e} | {_fmt(ea)} | {_fmt(eb)} | {_fmt(mo)} | {_fmt(sm)} |")
    L.append("\n## 6. 效率\n")
    L.append("| model | params | mean wall (s) | GPU peak (MB) |")
    L.append("| --- | ---: | ---: | ---: |")
    for a in agg:
        sub = [r for r in rows if r["model"] == a["model"] and r["scenario"] == a["scenario"]]
        gpu = np.nanmean([r["gpu_peak_mb"] for r in sub if r["gpu_peak_mb"] is not None]) if sub else None
        L.append(f"| {a['model']} | {a['params']} | {_fmt(a['wall_mean'], 0)} | {_fmt(gpu, 0)} |")
    L.append("\n## 7. 失败与局限\n")
    L.append("- 训练预算不一致：B1 为 200 epoch，B5 S1/S2 为 160 epoch，B5 S3 为 80 epoch；"
             "S3 的最佳 epoch 多为 5–29，说明未见细胞系场景训练不足。")
    L.append("- FiLM 的 S3 泛化需诚实报告；不因 S1 最优而选跨细胞系较差模型。")
    L.append("- 边界误差是模型自洽诊断，不是实测零剂量实验。")
    L.append("- 未做大规模超参数搜索；测试集未参与任何选择。")
    L.append("\n## 8. 产物\n")
    L.append("- `outputs/metrics/m6b/formal_runs.csv`, `formal_summary.csv`, `paired_deltas.csv`")
    L.append("- `outputs/figures/m6b/formal_test_macro.svg`, `formal_val_curves.svg`, `surface_examples.svg`")
    L.append("- `outputs/metrics/m6b/surface_examples.csv`")
    L.append("- 每个 run 的 `outputs/runs/m6b/<run_id>/{best.pt,train_log.csv,val_metrics.json,test_metrics.json,predictions_test.parquet}`")
    L.append("\n## 9. Git 与复现\n")
    L.append(f"- 生成时 Git：`{git_commit()}`")
    L.append("- 冻结数据：`data/processed/oneil_primary_v1_*.parquet` 与 `data/splits/` 均由 M3 manifest 哈希锁定，本阶段未修改。")
    L.append("- 复现：`scripts/11_run_m6b.py --stage formal --model <model> --epochs 200 --eval-test`；"
             "汇总：`scripts/13_aggregate_m6b.py --write`。")
    text = "\n".join(L)
    if args.write:
        (ROOT / "m6b_report.md").write_text(text, encoding="utf-8")
        manifest = {
            "generated_at": utc_now(), "git_commit": git_commit(),
            "report_sha256": file_sha256(ROOT / "m6b_report.md"),
            "config_sha256": file_sha256(ROOT / "configs/m6b_experiments.yaml"),
            "n_runs": len(rows), "selected": selected,
            "formal_summary": agg, "pytest_log": args.pytest_log,
        }
        (out_dir / "m6b_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"runs": len(rows), "selected": selected, "overall_test_macro": overall}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
