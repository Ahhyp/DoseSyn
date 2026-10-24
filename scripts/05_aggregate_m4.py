"""M4 aggregation: run summaries, aggregate metrics, SVG figures, report, manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.utils.config import ensure_directories, load_config, resolve_path  # noqa: E402
from dosesyn.utils.io import environment_snapshot, read_json, sha256_file, utc_now_iso, write_json  # noqa: E402
from dosesyn.utils.logging import setup_logging  # noqa: E402

MODEL_ORDER = ["b0_mean", "b1_tree", "b2_mlp", "b3_gine"]
MODEL_LABELS = {
    "b0_mean": "B0 Mean",
    "b1_tree": "B1 Tree-Dose",
    "b2_mlp": "B2 MLP-Dose",
    "b3_gine": "B3 GINE-Base",
}
METRIC_KEYS = [
    "matrix_macro_rmse",
    "matrix_macro_mae",
    "condition_macro_rmse",
    "point_rmse",
    "point_mae",
    "pooled_pearson",
    "r2",
    "matrix_pearson_mean",
]
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})


def collect_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        meta_path = run_dir / "run_metadata.json"
        if not meta_path.exists():
            continue
        meta = read_json(meta_path)
        split_id = meta.get("split_id")
        if not split_id:
            if meta.get("seed") is not None:
                split_id = f"seed_{meta['seed']}"
            elif meta.get("fold") is not None:
                split_id = f"fold_{meta['fold']}"
        status = meta.get("status")
        if not status:
            status = "ok" if (run_dir / "test_metrics.json").exists() else "failed"
        row: Dict[str, Any] = {
            "run_id": meta.get("run_id", run_dir.name),
            "model": meta.get("model"),
            "scenario": meta.get("scenario"),
            "split_id": split_id,
            "seed": meta.get("seed"),
            "fold": meta.get("fold"),
            "status": status,
            "run_dir": str(run_dir),
        }
        res_path = run_dir / "resources.json"
        if res_path.exists():
            row.update(read_json(res_path))
        test_path = run_dir / "test_metrics.json"
        if test_path.exists():
            test = read_json(test_path)
            for key in METRIC_KEYS:
                row[key] = test.get(key)
            row["test_n_matrices"] = test.get("n_matrices")
            row["test_n_points"] = test.get("n_points")
            row["test_y_max"] = test.get("y_true_max")
            row["test_y_min"] = test.get("y_true_min")
            row["test_y_gt1"] = test.get("n_y_true_gt1")
            row["test_y_eq0"] = test.get("n_y_true_eq0")
            row["test_y_pred_min"] = test.get("y_pred_min")
            row["test_n_pred_negative"] = test.get("n_y_pred_negative")
        val_path = run_dir / "val_metrics.json"
        if val_path.exists():
            row["val_matrix_macro_rmse"] = read_json(val_path).get("matrix_macro_rmse")
        err_path = run_dir / "error.json"
        if err_path.exists():
            err = read_json(err_path)
            row["error"] = err.get("error") or err.get("message")
        rows.append(row)
    return rows


def aggregate(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ok = [r for r in runs if r.get("status") == "ok" and r.get("scenario") and r.get("model")]
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in ok:
        groups.setdefault((r["model"], r["scenario"]), []).append(r)
    rows: List[Dict[str, Any]] = []
    for (model, scenario), members in sorted(groups.items()):
        row: Dict[str, Any] = {
            "model": model,
            "scenario": scenario,
            "n_runs": len(members),
            "splits": ",".join(str(m.get("split_id")) for m in members),
        }
        params = [m.get("n_parameters") for m in members if m.get("n_parameters") is not None]
        if params:
            row["n_parameters"] = int(sum(params) / len(params))
        for key in METRIC_KEYS:
            vals = [float(m[key]) for m in members if m.get(key) is not None]
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
                row[key] = mean
                row[f"{key}_std"] = math.sqrt(var)
                row[f"{key}_min"] = min(vals)
                row[f"{key}_max"] = max(vals)
        rows.append(row)
    return rows


def overall_by_model(agg: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for model in MODEL_ORDER:
        rows = [r for r in agg if r["model"] == model]
        if not rows:
            continue
        entry: Dict[str, Any] = {"model": model, "n_scenarios": len(rows)}
        for key in METRIC_KEYS:
            vals = [float(r[key]) for r in rows if r.get(key) is not None]
            if vals:
                entry[key] = sum(vals) / len(vals)
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Minimal dependency-free SVG charts
# ---------------------------------------------------------------------------
def _scale(value: float, lo: float, hi: float, a: float, b: float) -> float:
    if hi <= lo:
        return (a + b) / 2
    return a + (value - lo) / (hi - lo) * (b - a)


def line_chart_svg(
    path: Path,
    xs: Sequence[float],
    series: Sequence[Tuple[str, Sequence[float], str]],
    title: str,
    ylabel: str,
    width: int = 720,
    height: int = 360,
) -> None:
    values = [v for _, vals, _ in series for v in vals if v is not None and math.isfinite(v)]
    if not values:
        return
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1e-9
    left, right, top, bottom = 70, 20, 40, 50
    pw, ph = width - left - right, height - top - bottom
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="DejaVu Sans, sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="22" text-anchor="middle" font-size="15">{html.escape(title)}</text>',
    ]
    # axes and y ticks
    for k in range(5):
        frac = k / 4
        y = top + ph * (1 - frac)
        val = lo + (hi - lo) * frac
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" fill="#444">{val:.4g}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#333"/>')
    # x ticks
    xmin, xmax = min(xs), max(xs)
    for k in range(5):
        frac = k / 4
        x = left + pw * frac
        val = xmin + (xmax - xmin) * frac
        parts.append(f'<text x="{x:.1f}" y="{top+ph+18}" text-anchor="middle" fill="#444">{val:.0f}</text>')
    parts.append(
        f'<text x="{left+pw/2}" y="{height-10}" text-anchor="middle" fill="#444">epoch</text>'
    )
    parts.append(
        f'<text x="16" y="{top+ph/2}" text-anchor="middle" fill="#444" '
        f'transform="rotate(-90 16 {top+ph/2})">{html.escape(ylabel)}</text>'
    )
    for idx, (label, vals, color) in enumerate(series):
        pts = " ".join(
            f"{_scale(x, xmin, xmax, left, left+pw):.1f},{_scale(v, lo, hi, top+ph, top):.1f}"
            for x, v in zip(xs, vals)
            if v is not None and math.isfinite(v)
        )
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        ly = top + 14 + idx * 18
        parts.append(f'<line x1="{left+pw-140}" y1="{ly}" x2="{left+pw-115}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{left+pw-108}" y="{ly+4}" fill="#333">{html.escape(label)}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def bar_chart_svg(
    path: Path,
    scenarios: Sequence[str],
    models: Sequence[str],
    values: Dict[Tuple[str, str], float],
    title: str,
    width: int = 760,
    height: int = 380,
) -> None:
    left, right, top, bottom = 70, 20, 45, 70
    pw, ph = width - left - right, height - top - bottom
    all_vals = [v for v in values.values() if v is not None and math.isfinite(v)]
    if not all_vals:
        return
    hi = max(all_vals) * 1.08
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="DejaVu Sans, sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="15">{html.escape(title)}</text>',
    ]
    for k in range(5):
        y = top + ph * (1 - k / 4)
        val = hi * k / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" fill="#444">{val:.3f}</text>')
    group_w = pw / max(len(scenarios), 1)
    bar_w = group_w * 0.8 / max(len(models), 1)
    for si, scenario in enumerate(scenarios):
        gx = left + si * group_w + group_w * 0.1
        for mi, model in enumerate(models):
            val = values.get((model, scenario))
            if val is None:
                continue
            h = val / hi * ph
            x = gx + mi * bar_w
            color = COLORS[mi % len(COLORS)]
            parts.append(f'<rect x="{x:.1f}" y="{top+ph-h:.1f}" width="{bar_w*0.9:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(
            f'<text x="{left+si*group_w+group_w/2:.1f}" y="{top+ph+18}" text-anchor="middle" fill="#444">'
            f'{html.escape(scenario)}</text>'
        )
    for mi, model in enumerate(models):
        x = left + 10 + mi * 160
        parts.append(f'<rect x="{x}" y="{top+ph+38}" width="12" height="12" fill="{COLORS[mi % len(COLORS)]}"/>')
        parts.append(f'<text x="{x+18}" y="{top+ph+48}" fill="#333">{html.escape(MODEL_LABELS.get(model, model))}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(
    path: Path,
    cfg: Dict[str, Any],
    runs: Sequence[Dict[str, Any]],
    agg: Sequence[Dict[str, Any]],
    overall: Sequence[Dict[str, Any]],
    data_checks: Dict[str, Any],
    figures: Sequence[str],
    test_summary: str = "",
) -> None:
    ok = [r for r in runs if r.get("status") == "ok"]
    failed = [r for r in runs if r.get("status") != "ok"]
    lines: List[str] = []
    lines.append("# DoseSyn M4 基线训练报告\n")
    lines.append(f"生成时间：{utc_now_iso()}  \n配置：`configs/m4_training.yaml`（{cfg['version']}）\n")
    lines.append("## 1. 结论\n")
    if overall:
        best = min(overall, key=lambda r: float(r.get("matrix_macro_rmse", 9e9)))
        by_model = {r["model"]: r for r in overall}
        lines.append(
            f"- 成功运行 **{len(ok)}** 个（失败 {len(failed)} 个）。"
            f"按 scenario 等权平均的测试 `matrix_macro_RMSE` 最优模型为 **{MODEL_LABELS.get(best['model'], best['model'])}**"
            f"（{_fmt(best.get('matrix_macro_rmse'))}）。\n"
        )
        if "b2_mlp" in by_model and "b3_gine" in by_model:
            lines.append(
                f"- B2 等权平均 {_fmt(by_model['b2_mlp'].get('matrix_macro_rmse'))}，"
                f"B3 {_fmt(by_model['b3_gine'].get('matrix_macro_rmse'))}；"
                "B2 在 S1/S2（已见条件/已见药物对）更好，B3 在 S3（未见细胞系）略优。\n"
            )
    else:
        lines.append("- 尚无成功运行。\n")
    lines.append("- 主指标：`matrix_macro_RMSE`（每个 4×4 矩阵先算 RMSE，再对矩阵取宏平均）；模型选择/早停仅用验证集。\n")
    lines.append("- 测试标签为原始 `X/X0`，未裁剪；所有预测逐点保存。\n")
    lines.append("\n## 2. 固定流程与防泄漏\n")
    lines.append("- 直接读取 M3 冻结的 `train/val/test_conditions.parquet`，未重新划分、未做点级随机拆分。\n")
    lines.append("- S1/S2 用 3 个种子，S3 用 5 折；同一 condition/batch 的矩阵不跨集合。\n")
    lines.append("- 特征标准化只使用训练折拟合的 scaler（M3 产物），测试集不参与任何统计。\n")
    lines.append("- B0/B1 不训练梯度；B1 仅在训练折上拟合。\n")
    lines.append("- 单药监督范围：M4 的四个基线均不使用单药标签，单药曲线实际使用 0 条；S3 的 strict_cell_isolation 分配保持封存。\n")
    model_cfg = cfg.get("model", {})
    lines.append("\n### 模型配置\n")
    lines.append(
        f"- B0：训练折均值常数；B1：HistGradientBoosting（max_iter={model_cfg.get('tree_max_iter')}, "
        f"lr={model_cfg.get('tree_learning_rate')}, leaves={model_cfg.get('tree_max_leaf_nodes')}）。\n"
    )
    lines.append(
        f"- B2：细胞特征 50 + 两个药物 embedding {model_cfg.get('drug_embedding_dim')} + 剂量/零标志 → "
        f"MLP {model_cfg.get('mlp_hidden')}；B3：3 层 GINE（hidden {model_cfg.get('hidden_dim')}）+ 均值池化 + 同构 MLP 头。\n"
    )
    lines.append(
        f"- 优化：AdamW lr={cfg['training'].get('learning_rate')}, wd={cfg['training'].get('weight_decay')}, "
        f"Huber δ={cfg['training'].get('huber_delta')}, batch={cfg['training'].get('batch_size')}, "
        f"epochs≤{cfg['training'].get('epochs_dl')}, patience={cfg['training'].get('patience')}。\n"
    )
    lines.append("\n## 3. 数据完整性检查\n")
    for key, value in data_checks.items():
        lines.append(f"- {key}: {value}\n")
    lines.append("\n## 4. 各模型 × 场景测试结果\n")
    lines.append("| 模型 | 场景 | runs | macro RMSE | ±std | point RMSE | Pearson | R² | 参数量 |\n")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for row in agg:
        lines.append(
            f"| {MODEL_LABELS.get(row['model'], row['model'])} | {row['scenario']} | {row['n_runs']} | "
            f"{_fmt(row.get('matrix_macro_rmse'))} | {_fmt(row.get('matrix_macro_rmse_std'))} | "
            f"{_fmt(row.get('point_rmse'))} | {_fmt(row.get('pooled_pearson'))} | "
            f"{_fmt(row.get('r2'))} | {row.get('n_parameters') or '—'} |\n"
        )
    lines.append("\n## 5. 逐 split 明细\n")
    lines.append("| 模型 | 场景 | split | macro RMSE | point RMSE | Pearson | R² | 最佳 epoch | wall s | GPU MB |\n")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for row in sorted(ok, key=lambda r: (r["scenario"], r["model"], str(r.get("split_id")))):
        lines.append(
            f"| {MODEL_LABELS.get(row['model'], row['model'])} | {row['scenario']} | {row.get('split_id')} | "
            f"{_fmt(row.get('matrix_macro_rmse'))} | {_fmt(row.get('point_rmse'))} | "
            f"{_fmt(row.get('pooled_pearson'))} | {_fmt(row.get('r2'))} | {row.get('best_epoch') or '—'} | "
            f"{_fmt(row.get('wall_seconds'), 1)} | {_fmt(row.get('gpu_peak_reserved_mb'), 1)} |\n"
        )
    if failed:
        lines.append("\n## 6. 失败运行\n")
        for row in failed:
            lines.append(f"- `{row['run_id']}`：{row.get('error') or row.get('status')}\n")
    lines.append("\n## 7. 资源与复现\n")
    lines.append("- GPU：NVIDIA V100-PCIE-32GB（32 GB）；作业未独占 GPU，最多 4 个作业共享一张卡，显存占用约 0.1 GB。\n")
    lines.append("- 调度：Torque/PBS `qsub`，节点 node1；作业脚本自重定向日志（PBS `-o` 在本集群不生效）。\n")
    lines.append("- 命令：\n")
    lines.append("```bash\n")
    lines.append(".venv/bin/python scripts/04_run_baseline.py --scenario s1_random_condition --seed 42 --models b0_mean b1_tree b2_mlp b3_gine --device cuda\n")
    lines.append(".venv/bin/python scripts/05_aggregate_m4.py\n")
    lines.append(".venv/bin/python -m pytest -q tests\n")
    lines.append("```\n")
    lines.append(f"- 生成图：{', '.join(f'`{f}`' for f in figures) if figures else '—'}\n")
    if test_summary:
        lines.append(f"- 自动测试：{test_summary}\n")
    lines.append("\n## 8. 尚未解决的问题\n")
    for issue in cfg.get("report", {}).get("open_issues", []):
        lines.append(f"- {issue}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/m4_training.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_directories(cfg)
    log_dir = Path(cfg["paths"]["logs_dir"]) if "logs_dir" in cfg["paths"] else ROOT / "outputs/logs"
    logger = setup_logging("dosesyn.m4", log_dir / "05_aggregate_m4.log")
    runs_dir = resolve_path(cfg, cfg["outputs"]["runs_dir"])
    metrics_dir = resolve_path(cfg, cfg["outputs"]["metrics_dir"])
    figures_dir = resolve_path(cfg, cfg["outputs"]["figures_dir"])

    runs = collect_runs(runs_dir)
    agg = aggregate(runs)
    overall = overall_by_model(agg)
    columns = [
        "run_id", "model", "scenario", "split_id", "seed", "fold", "status",
        *METRIC_KEYS, "test_n_matrices", "test_n_points", "test_y_max", "test_y_gt1", "test_y_eq0",
        "val_matrix_macro_rmse", "n_parameters", "wall_seconds", "best_epoch", "epochs_run",
        "gpu_name", "gpu_peak_reserved_mb", "hostname", "error",
    ]
    _write_csv(runs, metrics_dir / "runs_summary.csv", columns)
    agg_cols = ["model", "scenario", "n_runs", "splits"]
    for key in METRIC_KEYS:
        agg_cols += [key, f"{key}_std", f"{key}_min", f"{key}_max"]
    _write_csv(agg, metrics_dir / "aggregate_summary.csv", agg_cols)
    _write_csv(overall, metrics_dir / "overall_by_model.csv", ["model", "n_scenarios", *METRIC_KEYS])
    write_json(
        {"version": cfg["version"], "generated_at": utc_now_iso(), "runs": runs, "aggregate": agg, "overall": overall},
        metrics_dir / "m4_summary.json",
    )

    # training curves
    figures: List[str] = []
    curves_dir = figures_dir / "curves"
    ok_runs = [r for r in runs if r.get("status") == "ok"]
    for row in ok_runs:
        log_path = Path(row["run_dir"]) / "train_log.csv"
        if not log_path.exists():
            continue
        hist = _read_csv(log_path)
        if not hist:
            continue
        xs: List[float] = []
        train: List[float] = []
        val: List[float] = []
        for h in hist:
            try:
                x = float(h["epoch"])
                v = float(h["val_matrix_macro_rmse"])
            except (KeyError, TypeError, ValueError):
                continue
            xs.append(x)
            val.append(v)
            try:
                train.append(float(h["train_loss"]))
            except (KeyError, TypeError, ValueError):
                train.append(float("nan"))
        if not xs:
            continue
        series = []
        if any(math.isfinite(v) for v in train):
            series.append(("train loss", train, "#4C78A8"))
        series.append(("val macro RMSE", val, "#E45756"))
        out = curves_dir / f"{row['run_id']}.svg"
        line_chart_svg(out, xs, series, f"{row['run_id']} 训练曲线", "loss / RMSE")
        figures.append(str(out.relative_to(ROOT)))

    scenarios = []
    for s in cfg["runs"]["scenarios"]:
        if any(r["scenario"] == s for r in agg) and s not in scenarios:
            scenarios.append(s)
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in agg)]
    values = {(r["model"], r["scenario"]): r.get("matrix_macro_rmse") for r in agg}
    bar = figures_dir / "macro_rmse_by_model_scenario.svg"
    bar_chart_svg(bar, scenarios, models, values, "Test matrix-macro RMSE（越低越好）")
    figures.append(str(bar.relative_to(ROOT)))

    data_checks = {
        "数据源": "M3 冻结 splits（未重新划分）",
        "split 数": len({(r["scenario"], str(r.get("split_id"))) for r in ok_runs}),
        "成功运行数": len(ok_runs),
        "失败运行数": len(runs) - len(ok_runs),
        "预测行数/matrix": "每矩阵 16 行（runner 内断言）",
        "测试标签": f"原始 X/X0 最大 {_fmt(max((r.get('test_y_max') or 0) for r in ok_runs), 3)}，最小 {_fmt(min((r.get('test_y_min') or 0) for r in ok_runs), 3)}；未裁剪",
        "各运行测试点 X/X0>1 合计": sum(int(r.get("test_y_gt1") or 0) for r in ok_runs),
        "各运行测试点 X/X0=0 合计": sum(int(r.get("test_y_eq0") or 0) for r in ok_runs),
        "负预测点数": sum(int(r.get("test_n_pred_negative") or 0) for r in ok_runs),
        "早停依据": "仅验证集 matrix_macro_RMSE",
        "单药监督使用": "0 条单药曲线（四个基线均不使用单药标签）",
        "随机种子": str(cfg["training"].get("seeds")),
        "Git commit": os.popen("cd /home/kouzheng_2/AH_YeP/DoseSyn && git rev-parse HEAD").read().strip(),
    }
    report_path = ROOT / cfg["outputs"]["report"]
    pytest_log = ROOT / "outputs/audit/pytest.log"
    test_summary = ""
    if pytest_log.exists():
        match = re.search(r"(\d+) passed", pytest_log.read_text(encoding="utf-8", errors="ignore"))
        if match:
            test_summary = f"{match.group(1)} passed（outputs/audit/pytest.log）"
    write_report(report_path, cfg, runs, agg, overall, data_checks, figures, test_summary)
    write_json(
        {
            "version": "m4-report-v1",
            "generated_at": utc_now_iso(),
            "git_commit": data_checks["Git commit"],
            "environment": environment_snapshot(),
            "inputs": {
                "config": str(resolve_path(cfg, args.config).relative_to(ROOT)),
                "config_sha256": sha256_file(resolve_path(cfg, args.config)),
                "m3_manifest": cfg["inputs"]["m3_manifest"],
                "m3_manifest_sha256": sha256_file(resolve_path(cfg, cfg["inputs"]["m3_manifest"])),
                "dataset_config": cfg["inputs"]["dataset_config"],
                "dataset_config_sha256": sha256_file(resolve_path(cfg, cfg["inputs"]["dataset_config"])),
            },
            "report": str(report_path.relative_to(ROOT)),
            "report_sha256": sha256_file(report_path),
            "n_runs_ok": len(ok_runs),
            "n_runs_failed": len(runs) - len(ok_runs),
        },
        metrics_dir / "m4_report_manifest.json",
    )
    logger.info("aggregated %d runs (%d ok); report=%s", len(runs), len(ok_runs), report_path)
    print(json.dumps({"runs": len(runs), "ok": len(ok_runs), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
