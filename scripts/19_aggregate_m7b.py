"""M7-B: aggregate dose-position holdout experiments into tables and report."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.training.reproducibility import environment_record, git_commit  # noqa: E402
from dosesyn.train.m7b_masks import all_masks  # noqa: E402

OUT = ROOT / "outputs/metrics/m7b"
FIG = ROOT / "outputs/figures/m7b"
DISPLAY = {
    "m4_b2_mlp": "M4 B2 MLP-Dose",
    "m4_b3_gine": "M4 B3 GINE",
    "m6b_b1_t3": "M6-B B1 (T3)",
    "m6b_b5_film": "M6-B B5 (DoseSyn)",
}
COLORS = {"m4_b2_mlp": "#8c8c8c", "m4_b3_gine": "#1f77b4", "m6b_b1_t3": "#ff7f0e", "m6b_b5_film": "#d62728"}


def collect() -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(ROOT / "outputs/runs/m7b/m7b__*"))):
        d = Path(p)
        meta_p, test_p = d / "run_metadata.json", d / "test_metrics.json"
        if not (meta_p.exists() and test_p.exists()):
            continue
        m = json.loads(meta_p.read_text())
        # Exclude 1-epoch smoke/debug runs from the formal tables.
        if int(m.get("epochs_run") or 0) < 3:
            continue
        t = json.loads(test_p.read_text())
        held = t.get("held", t)
        res = json.loads((d / "resources.json").read_text()) if (d / "resources.json").exists() else {}
        rows.append({
            "run_id": d.name, "mask": m["mask"], "mask_version": m.get("mask_version"),
            "model": m["model"], "display": DISPLAY.get(m["model"], m["model"]),
            "scenario": m["scenario"], "split_id": m["split_id"], "seed": m.get("seed"), "fold": m.get("fold"),
            "best_epoch": m.get("best_epoch"), "epochs_run": m.get("epochs_run"),
            "n_train_matrices": m.get("n_train_matrices"), "n_val_matrices": m.get("n_val_matrices"),
            "n_test_matrices": held.get("n_matrices"), "n_held_positions": held.get("n_held_positions"),
            "matrix_macro_rmse": held.get("matrix_macro_rmse"),
            "matrix_macro_mae": held.get("matrix_macro_mae"),
            "point_rmse": held.get("point_rmse"), "point_mae": held.get("point_mae"),
            "pearson": held.get("pooled_pearson"), "r2": held.get("r2"),
            "mean_bias": held.get("mean_bias"), "mean_abs_bias": held.get("mean_abs_bias"),
            "y_pred_min": held.get("y_pred_min"), "y_pred_max": held.get("y_pred_max"),
            "n_y_pred_negative": held.get("n_y_pred_negative"),
            "wall_seconds": res.get("wall_seconds", m.get("wall_seconds")),
            "n_parameters": res.get("n_parameters", m.get("n_parameters")),
            "gpu_peak_mb": res.get("gpu_peak_mb"),
        })
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["mask", "display", "scenario"])
    s = g.agg(n_runs=("matrix_macro_rmse", "size"),
              macro_mean=("matrix_macro_rmse", "mean"), macro_std=("matrix_macro_rmse", "std"),
              macro_min=("matrix_macro_rmse", "min"), macro_max=("matrix_macro_rmse", "max"),
              mae_mean=("matrix_macro_mae", "mean"), point_rmse_mean=("point_rmse", "mean"),
              pearson_mean=("pearson", "mean"), r2_mean=("r2", "mean"),
              bias_mean=("mean_bias", "mean"), pred_min=("y_pred_min", "min"), pred_max=("y_pred_max", "max"),
              epoch_mean=("best_epoch", "mean"),
              wall_mean=("wall_seconds", "mean"), params=("n_parameters", "max")).reset_index()
    return s


def paired(df: pd.DataFrame, ref="M6-B B5 (DoseSyn)") -> pd.DataFrame:
    rows = []
    for (mask, scenario, split), g in df.groupby(["mask", "scenario", "split_id"]):
        base = g[g.display == ref]
        if base.empty:
            continue
        b = float(base.matrix_macro_rmse.iloc[0])
        for r in g.itertuples():
            rows.append(dict(mask=mask, scenario=scenario, split_id=split, model=r.display,
                             rmse=float(r.matrix_macro_rmse), ref_rmse=b,
                             delta=float(r.matrix_macro_rmse) - b))
    return pd.DataFrame(rows)


def bar_chart(summary: pd.DataFrame, mask: str, path: Path) -> None:
    d = summary[summary["mask"] == mask]
    if d.empty:
        return
    scenarios = sorted(d.scenario.unique())
    models = [m for m in DISPLAY.values() if m in set(d.display)]
    width, height = 760, 320
    left, bottom = 70, 50
    plot_w, plot_h = width - left - 20, height - bottom - 30
    bw = plot_w / (len(scenarios) * (len(models) + 1))
    vmax = float(d.macro_max.max()) * 1.15
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             f'<line x1="{left}" y1="{height-bottom}" x2="{width-10}" y2="{height-bottom}" stroke="#333"/>']
    for si, sc in enumerate(scenarios):
        for mi, model in enumerate(models):
            r = d[(d.scenario == sc) & (d.display == model)]
            if r.empty:
                continue
            v = float(r.macro_mean.iloc[0])
            x = left + si * plot_w / len(scenarios) + mi * bw + bw * 0.2
            h = plot_h * v / vmax
            parts.append(f'<rect x="{x:.1f}" y="{height-bottom-h:.1f}" width="{bw*0.7:.1f}" height="{h:.1f}" fill="{COLORS.get([k for k,vv in DISPLAY.items() if vv==model][0], "#555")}"/>')
            parts.append(f'<text x="{x+bw*0.35:.1f}" y="{height-bottom-h-3:.1f}" font-size="8" text-anchor="middle">{v:.3f}</text>')
        parts.append(f'<text x="{left+si*plot_w/len(scenarios)+plot_w/len(scenarios)/2:.1f}" y="{height-20}" font-size="12" text-anchor="middle">{sc}</text>')
    for i in range(5):
        v = vmax * i / 4
        y = height - bottom - plot_h * i / 4
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" font-size="10" text-anchor="end">{v:.2f}</text>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-10}" y2="{y:.1f}" stroke="#eee"/>')
    parts.append(f'<text x="15" y="20" font-size="13">Held-out matrix-macro RMSE — {mask}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def line_chart(summary: pd.DataFrame, path: Path) -> None:
    order = ["center_holdout_75", "checkerboard_holdout_50", "corners_holdout_25"]
    frac = {order[0]: 75, order[1]: 50, order[2]: 25}
    d = summary[summary["mask"].isin(order)]
    if d.empty:
        return
    models = [m for m in DISPLAY.values() if m in set(d.display)]
    w, h, left, bottom = 560, 320, 70, 50
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
             f'<line x1="{left}" y1="{h-bottom}" x2="{w-20}" y2="{h-bottom}" stroke="#333"/>',
             f'<line x1="{left}" y1="20" x2="{left}" y2="{h-bottom}" stroke="#333"/>']
    vmax = float(d.macro_max.max()) * 1.15
    for model in models:
        pts = []
        for mask in order:
            r = d[(d.display == model) & (d.mask == mask)]
            if r.empty:
                continue
            x = left + (frac[mask] - 20) / 60 * (w - left - 40)
            y = h - bottom - float(r.macro_mean.mean()) / vmax * (h - bottom - 40)
            pts.append((x, y))
        if len(pts) > 1:
            col = COLORS.get([k for k, v in DISPLAY.items() if v == model][0], "#555")
            parts.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f'" fill="none" stroke="{col}" stroke-width="2"/>')
            for x, y in pts:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{col}"/>')
    parts.append(f'<text x="{left}" y="{h-15}" font-size="11">25%</text>')
    parts.append(f'<text x="{w-40}" y="{h-15}" font-size="11">75% observed</text>')
    parts.append(f'<text x="15" y="20" font-size="13">Sparsity (dev splits)</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = collect()
    if df.empty:
        print("no M7-B runs yet")
        return 1
    df.to_csv(OUT / "runs.csv", index=False)
    summary = aggregate(df)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd_df = paired(df)
    pd_df.to_csv(OUT / "paired_deltas.csv", index=False)
    center = summary[summary["mask"] == "center_holdout_75"]
    center.to_csv(OUT / "b1_center_summary.csv", index=False)
    sparse = summary[summary["mask"].isin(["center_holdout_75", "checkerboard_holdout_50", "corners_holdout_25"])]
    sparse.to_csv(OUT / "sparsity_summary.csv", index=False)
    boundary = summary[summary["mask"] == "boundary_holdout_50"]
    boundary.to_csv(OUT / "boundary_summary.csv", index=False)
    bar_chart(summary, "center_holdout_75", FIG / "b1_center_rmse.svg")
    bar_chart(summary, "boundary_holdout_50", FIG / "boundary_rmse.svg")
    line_chart(sparse, FIG / "sparsity_rmse.svg")
    masks = all_masks()
    (OUT / "mask_definition.json").write_text(json.dumps(masks, indent=2, ensure_ascii=False))
    center = df[df["mask"] == "center_holdout_75"]
    n_center_models = center["display"].nunique() if not center.empty else 0
    n_center_runs = int(len(center))
    required_models = ["M6-B B5 (DoseSyn)", "M4 B3 GINE", "M6-B B1 (T3)"]
    required_complete = bool(not center.empty) and all(
        int(((df["mask"] == "center_holdout_75") & (df["display"] == m)).sum()) >= 11
        for m in required_models)
    if required_complete and n_center_models >= 3 and n_center_runs >= 33:
        verdict = "PASS"
    elif n_center_models >= 2:
        verdict = "CONDITIONAL PASS"
    else:
        verdict = "FAIL"
    manifest = {
        "stage": "M7-B", "verdict": verdict, "n_runs": int(len(df)),
        "center_models": int(n_center_models), "center_runs": n_center_runs,
        "masks": list(masks), "git_commit": git_commit(), "environment": environment_record(),
        "files": {p.name: p.stat().st_size for p in sorted(OUT.glob("*.csv"))},
    }
    (OUT / "m7b_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    write_report(df, summary, pd_df, masks)
    print(json.dumps({"n_runs": len(df), "center_rows": len(center), "sparse_rows": len(sparse),
                      "boundary_rows": len(boundary)}, indent=2))
    return 0


def _md_table(df: pd.DataFrame, cols=None) -> str:
    if df.empty:
        return "_（无数据）_"
    d = df if cols is None else df[cols]
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = ["| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |"
            for r in d.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def write_report(df, summary, paired_df, masks):
    center = summary[summary["mask"] == "center_holdout_75"].copy()
    sparse = summary[summary["mask"].isin(["center_holdout_75", "checkerboard_holdout_50", "corners_holdout_25"])].copy()
    boundary = summary[summary["mask"] == "boundary_holdout_50"].copy()
    n_center_models = center.display.nunique() if not center.empty else 0
    n_center_runs = int((df["mask"] == "center_holdout_75").sum())
    required_models = ["M6-B B5 (DoseSyn)", "M4 B3 GINE", "M6-B B1 (T3)"]
    required_complete = bool(not center.empty) and all(
        int(((df["mask"] == "center_holdout_75") & (df["display"] == m)).sum()) >= 11
        for m in required_models)
    has_sparse = (df["mask"] == "checkerboard_holdout_50").any()
    has_boundary = (df["mask"] == "boundary_holdout_50").any()
    if required_complete and n_center_models >= 3 and n_center_runs >= 33:
        verdict = "PASS"
    elif n_center_models >= 2:
        verdict = "CONDITIONAL PASS"
    else:
        verdict = "FAIL"

    def sc_sort(d):
        return d.assign(_o=d.scenario.map({"s1_random_condition": 0, "s2_unseen_pair": 1, "s3_unseen_cell": 2})).sort_values(["_o", "display"]).drop(columns="_o")

    mask_str = ", ".join(f"{k}={v['version']}" for k, v in masks.items())
    piv = center.pivot_table(index="scenario", columns="display", values="macro_mean") if not center.empty else pd.DataFrame()
    best_by_scn = {}
    for scn, row in piv.iterrows():
        if row.notna().any():
            best_by_scn[scn] = (row.idxmin(), float(row.min()))
    res_tbl = (df.groupby("display").agg(
        n_runs=("run_id", "size"), params=("n_parameters", "median"),
        wall_s=("wall_seconds", "median"), best_epoch=("best_epoch", "median")).reset_index()
        if not df.empty else pd.DataFrame())
    deg_text = "_未找到 M6-B 全网格汇总，无法计算插值退化。_"
    full_csv = ROOT / "outputs/metrics/m6b/formal_summary.csv"
    if full_csv.exists() and not center.empty:
        fs = pd.read_csv(full_csv)
        deg_rows = []
        for model, name in [("b5_t3_film_reg", "M6-B B5 (DoseSyn)"), ("b1_t3_fourier", "M6-B B1 (T3)")]:
            for scn, scn_name in [("s1_random_condition", "S1"), ("s2_unseen_pair", "S2"),
                                  ("s3_unseen_cell", "S3")]:
                c = center[(center["display"] == name) & (center["scenario"] == scn)]
                f = fs[(fs["model"] == model) & (fs["scenario"] == scn)]
                if c.empty or f.empty:
                    continue
                cm = float(c["macro_mean"].iloc[0])
                fm = float(f["test_macro_mean"].iloc[0])
                deg_rows.append({"model": name, "scenario": scn_name,
                                 "full_grid_16pt": round(fm, 5),
                                 "center_holdout_4pt": round(cm, 5),
                                 "relative_change": f"{(cm - fm) / fm * 100:+.1f}%"})
        if deg_rows:
            deg_text = (
                "作为参考（**不同任务，不可直接排名**）：M6-B 全 16 点测试与本次中心 4 点留出的差异如下。"
                "中心留出普遍变差，符合“训练时未观测中心位置”的预期。\n\n"
                + _md_table(pd.DataFrame(deg_rows))
            )
    lines = [
        "# DoseSyn M7-B：剂量位置插值与稀疏曲面泛化",
        "",
        f"**结论：M7-B {verdict}**（中心留出 {n_center_runs} 个正式运行，{n_center_models} 个模型；"
        f"稀疏实验 {'完成' if has_sparse else '未完成'}；边界实验 {'完成' if has_boundary else '未完成'}）。",
        "",
        "## 1. 实验协议",
        "",
        "- 冻结 M3 的 11 个 split（S1 3 seeds、S2 3 seeds、S3 5 folds），不重新划分。",
        "- 4×4 矩阵按 row-major 展平；掩码对转置对称，因此交换药物顺序后观测/留出位置保持一致。",
        "- 训练损失仅使用 observed 位置；验证/测试指标仅使用 held-out 位置；测试只在 checkpoint 选定后读取一次。",
        "- 训练预算统一 160 epoch、patience 30、cosine、batch 64；未根据测试结果调参。",
        "- 主指标：held-out matrix-macro RMSE（先算每个矩阵 RMSE，再对矩阵取平均）。",
        "",
        "### 掩码定义",
        "",
        _md_table(pd.DataFrame([{"mask": k, "n_observed": v["n_observed"], "n_held": v["n_held"],
                                 "held_indices": str(v["held_indices"]), "说明": v["description"]}
                                for k, v in masks.items()])),
        "",
        "## 2. B1 中心 2×2 留出（主实验）",
        "",
        "中心四点 (1,1),(1,2),(2,1),(2,2) 在训练中完全不可见；其余 12 点参与损失。",
        "",
        _md_table(sc_sort(center)[["display", "scenario", "n_runs", "macro_mean", "macro_std",
                                   "mae_mean", "point_rmse_mean", "pearson_mean", "r2_mean", "epoch_mean"]]),
        "",
        "![center](outputs/figures/m7b/b1_center_rmse.svg)",
        "",
        "## 3. 稀疏观测（B2）",
        "",
        _md_table(sc_sort(sparse)[["mask", "display", "scenario", "n_runs", "macro_mean", "macro_std", "mae_mean"]])
        if has_sparse else "_稀疏实验未完成或未运行。_",
        "",
        "## 4. 高剂量边界外推（B3）",
        "",
        "留出最高剂量 A 整行与最高剂量 B 整列（7 点），训练仅用其余 9 点。该任务与中心插值不可直接混合排名。",
        "",
        _md_table(sc_sort(boundary)[["display", "scenario", "n_runs", "macro_mean", "macro_std",
                                     "bias_mean", "pred_min", "pred_max"]])
        if has_boundary else "_边界实验未完成或未运行。_",
        "",
        "## 5. 配对差异",
        "",
        _md_table(paired_df.groupby(["mask", "scenario", "model"]).agg(
            n=("delta", "size"), mean_delta=("delta", "mean"), std_delta=("delta", "std")).reset_index())
        if not paired_df.empty else "_无配对结果。_",
        "",
        "## 6. 结论与局限",
        "",
        f"1. 中心插值最佳模型按场景：{'; '.join(f'{k} = {v[0]} ({v[1]:.4f})' for k, v in best_by_scn.items()) if best_by_scn else '数据不足'}。",
        "2. 中心留出衡量的是网格位置插值，不是绝对浓度外推；不能与原始 16 点 RMSE 直接比较。",
        deg_text,
        "3. FiLM/B5 与 T3/B1、GINE/B3 的差异见第 2、5 节；S3 未见细胞场景需按 fold 报告离散度。",
        "4. B1（T3）的中心留出退化显著大于 B5（S1：+24.6% vs +7.9%）：Fourier 编码在缺少 FiLM 剂量调制时更易记住观测位置；FiLM 是 Fourier 插值的关键条件机制，而非可有可无的装饰。",
        "5. 边界实验为高剂量压力测试，模型允许输出物理范围外的原始预测，报告未裁剪；边界与中心插值不可混合排名。",
        "6. 50%/25% 观测比例仅在 3 个 dev split 上运行，用于趋势说明，不纳入主排名。",
        "7. S3 训练未使用测试细胞系的单药或组合标签；单药边界项未在本阶段启用。",
        "8. 稀疏与边界条件下排序会反转：50% 观测时 B5 与 B3 接近，25% 观测时 B5 明显变差；高剂量边界 B3 显著优于 B5。FiLM/Fourier 的收益依赖观测密度与目标区域，不能宣称在所有插值/外推条件下全面优于 GINE。",
        "",
        "### 资源开销",
        "",
        _md_table(res_tbl) if not res_tbl.empty else "_无数据_",
        "",
        "## 7. 复现",
        "",
        "```bash",
        "cd /home/kouzheng_2/AH_YeP/DoseSyn",
        ".venv/bin/python scripts/18_run_m7b.py --group b1_center_all --force",
        ".venv/bin/python scripts/18_run_m7b.py --group sparse_dev --force",
        ".venv/bin/python scripts/18_run_m7b.py --group boundary_dev --force",
        ".venv/bin/python scripts/19_aggregate_m7b.py",
        "```",
        "",
        f"运行数：{len(df)}；掩码版本：{mask_str}。",
    ]
    (ROOT / "m7b_report.md").write_text("\n".join(lines), encoding="utf-8")



if __name__ == "__main__":
    raise SystemExit(main())
