"""Aggregate M7-A external baselines with archived M4/M6-B results."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCENARIO_SHORT = {
    "s1_random_condition": "S1",
    "s2_unseen_pair": "S2",
    "s3_unseen_cell": "S3",
}
DISPLAY = {
    "m4_b0_mean": "M4 B0 Mean",
    "m4_b1_tree": "M4 B1 Tree",
    "m4_b2_mlp": "M4 B2 MLP",
    "m4_b3_gine": "M4 B3 GINE",
    "m6b_b1_t3": "M6-B B1 (T3)",
    "m6b_b5_film": "M6-B B5 (T3+FiLM)",
    "combokr": "comboKR 2.0",
    "deepsynba_adapted": "DeepSynBa (adapted)",
}
MAIN = ["m4_b0_mean", "m4_b1_tree", "m4_b2_mlp", "m4_b3_gine",
        "m6b_b1_t3", "m6b_b5_film", "combokr"]
SUPP = ["deepsynba_adapted"]


def _metrics(d):
    return {k: d.get(k) for k in (
        "matrix_macro_rmse", "matrix_macro_mae", "point_rmse", "point_mae",
        "pooled_pearson", "r2", "condition_macro_rmse", "n_matrices")}


def _split_id(path):
    name = path.parent.name
    if name.startswith("seed_"):
        return name
    if name.startswith("fold_"):
        return name
    return name


def collect_archived_m4(rows):
    for p in sorted((ROOT / "outputs/runs/m4").glob("*/test_metrics.json")):
        run = p.parent.name
        parts = run.split("__")
        if len(parts) < 3:
            continue
        model = parts[0]
        if model not in ("b0_mean", "b1_tree", "b2_mlp", "b3_gine"):
            continue
        meta_path = p.parent / "run_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        m = _metrics(json.loads(p.read_text()))
        rows.append(dict(
            model=f"m4_{model}", display=DISPLAY[f"m4_{model}"], source="archived_m4",
            scenario=parts[1], split_id=parts[2], split_role="test", **m,
            params=meta.get("n_parameters"),
            runtime_seconds=meta.get("wall_seconds"),
            notes="reused M4 archived run"))


def collect_archived_m6b(rows):
    for p in sorted((ROOT / "outputs/runs/m6b").glob("*/test_metrics.json")):
        run = p.parent.name
        parts = run.split("__")
        if len(parts) < 3 or not parts[0].startswith("m6b"):
            continue
        model = parts[1]
        key = {"b1_t3_fourier": "m6b_b1_t3", "b5_t3_film_reg": "m6b_b5_film"}.get(model)
        if key is None:
            continue
        meta_path = p.parent / "run_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        m = _metrics(json.loads(p.read_text()))
        rows.append(dict(
            model=key, display=DISPLAY[key], source="archived_m6b",
            scenario=parts[2], split_id=parts[3], split_role="test", **m,
            params=meta.get("n_parameters"),
            runtime_seconds=meta.get("wall_seconds"),
            notes="reused M6-B archived run"))


def collect_external(rows, kind):
    model_key = {"combokr": "combokr", "deepsynba": "deepsynba_adapted"}[kind]
    base = ROOT / "outputs/runs/m7a" / kind
    for meta_path in sorted(base.glob("*/*/run_metadata.json")):
        meta = json.loads(meta_path.read_text())
        if "metrics" not in meta and "test_metrics" not in meta:
            continue
        m = _metrics(meta.get("metrics") or meta.get("test_metrics"))
        scen = meta.get("scenario")
        split = f"seed_{meta['seed']}" if meta.get("seed") is not None else f"fold_{meta['fold']}"
        rows.append(dict(
            model=model_key, display=DISPLAY[model_key], source="m7a_new",
            scenario=scen, split_id=split, split_role=meta.get("split", "test"),
            **m,
            params=meta.get("params"),
            runtime_seconds=meta.get("runtime_seconds"),
            notes=meta.get("notes", "")))


def summary_table(df):
    g = df.groupby(["model", "display", "scenario"], sort=False)
    out = g.agg(
        n_runs=("matrix_macro_rmse", "size"),
        macro_rmse_mean=("matrix_macro_rmse", "mean"),
        macro_rmse_std=("matrix_macro_rmse", "std"),
        macro_rmse_min=("matrix_macro_rmse", "min"),
        macro_rmse_max=("matrix_macro_rmse", "max"),
        macro_mae_mean=("matrix_macro_mae", "mean"),
        point_rmse_mean=("point_rmse", "mean"),
        pearson_mean=("pooled_pearson", "mean"),
        r2_mean=("r2", "mean"),
        params=("params", "max"),
        runtime_seconds=("runtime_seconds", "mean"),
    ).reset_index()
    out["scenario_short"] = out["scenario"].map(SCENARIO_SHORT)
    return out


def paired_deltas(df):
    out = []
    base = df[(df.model == "m6b_b5_film")][["scenario", "split_id", "matrix_macro_rmse"]]
    base = base.rename(columns={"matrix_macro_rmse": "dosesyn_rmse"})
    for model in df.model.unique():
        if model == "m6b_b5_film":
            continue
        sub = df[df.model == model][["scenario", "split_id", "matrix_macro_rmse", "display"]]
        merged = sub.merge(base, on=["scenario", "split_id"], how="inner")
        if merged.empty:
            continue
        merged["delta_vs_dosesyn"] = merged["matrix_macro_rmse"] - merged["dosesyn_rmse"]
        merged["scenario_short"] = merged["scenario"].map(SCENARIO_SHORT)
        out.append(merged)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def svg_bars(summary, path, models):
    sub = summary[summary.model.isin(models)].copy()
    scen = ["S1", "S2", "S3"]
    sub = sub[sub.scenario_short.isin(scen)]
    if sub.empty:
        return
    models = [m for m in models if m in set(sub.model)]
    width, height = 900, 360
    left, top, bottom = 95, 20, 50
    bar_w = (width - left - 20) / max(1, len(scen))
    vmax = max(0.35, float(np.nanmax(sub.macro_rmse_mean)) * 1.1)
    colors = ["#6c757d", "#a5a58d", "#457b9d", "#1d3557", "#2a9d8f", "#e76f51", "#f4a261"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="Helvetica">']
    parts.append(f'<text x="{width/2}" y="16" text-anchor="middle" font-size="14">M7-A matrix-macro RMSE</text>')
    for i, sc in enumerate(scen):
        x0 = left + i * bar_w
        parts.append(f'<text x="{x0 + bar_w/2}" y="{height - 20}" text-anchor="middle" font-size="13">{sc}</text>')
        n = max(1, len(models))
        for j, model in enumerate(models):
            r = sub[(sub.model == model) & (sub.scenario_short == sc)]
            if r.empty:
                continue
            v = float(r.macro_rmse_mean.iloc[0])
            bh = (height - top - bottom) * v / vmax
            x = x0 + 10 + j * (bar_w - 20) / n
            w = max(2.0, (bar_w - 20) / n - 3)
            y = height - bottom - bh
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bh:.1f}" fill="{colors[j % len(colors)]}"/>')
    for j, model in enumerate(models):
        x = left + j * 110
        parts.append(f'<rect x="{x}" y="{height-12}" width="10" height="8" fill="{colors[j % len(colors)]}"/>')
        parts.append(f'<text x="{x+14}" y="{height-4}" font-size="9">{html.escape(DISPLAY.get(model, model))}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m7a/m7a.yaml")
    args = ap.parse_args()
    metrics_dir = ROOT / "outputs/metrics/m7a"
    fig_dir = ROOT / "outputs/figures/m7a"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    collect_archived_m4(rows)
    collect_archived_m6b(rows)
    collect_external(rows, "combokr")
    collect_external(rows, "deepsynba")
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no runs found")
    df.to_csv(metrics_dir / "external_baseline_runs.csv", index=False)

    summ = summary_table(df)
    summ.to_csv(metrics_dir / "external_baseline_summary.csv", index=False)
    paired = paired_deltas(df)
    if not paired.empty:
        paired.to_csv(metrics_dir / "external_paired_deltas.csv", index=False)

    svg_bars(summ, fig_dir / "macro_rmse_comparison.svg", MAIN + SUPP)

    # pivot main table
    pivot = summ.pivot_table(index=["display"], columns="scenario_short",
                             values="macro_rmse_mean", aggfunc="mean")
    pivot = pivot.reindex(columns=["S1", "S2", "S3"])
    order = [DISPLAY[m] for m in MAIN + SUPP if DISPLAY[m] in pivot.index]
    pivot = pivot.reindex(order)
    pivot["mean"] = pivot[["S1", "S2", "S3"]].mean(axis=1, skipna=True)
    pivot.to_csv(metrics_dir / "main_comparison.csv")

    lines = ["# M7-A 外部 Baseline 与 DoseSyn 统一协议对比", ""]
    lines.append("所有数值均为本项目在 M3 冻结划分上重新得到的 matrix-macro RMSE；"
                 "M4/M6-B 为归档复用，新运行标注来源。")
    lines.append("")
    lines.append("## 1. 主表（matrix-macro RMSE，越低越好）")
    lines.append("")
    header = "| 模型 | S1 | S2 | S3 | 等权平均 | 来源 |"
    lines.append(header)
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    src = {"m4_b0_mean": "归档 M4", "m4_b1_tree": "归档 M4", "m4_b2_mlp": "归档 M4",
           "m4_b3_gine": "归档 M4", "m6b_b1_t3": "归档 M6-B", "m6b_b5_film": "归档 M6-B",
           "combokr": "本阶段新运行", "deepsynba_adapted": "本阶段新运行（适配）"}
    for model in MAIN + SUPP:
        name = DISPLAY[model]
        if name not in pivot.index:
            continue
        r = pivot.loc[name]
        def fmt(v):
            return "—" if pd.isna(v) else f"{v:.4f}"
        lines.append(f"| {name}{'（补充）' if model in SUPP else ''} | {fmt(r.get('S1'))} | "
                     f"{fmt(r.get('S2'))} | {fmt(r.get('S3'))} | {fmt(r.get('mean'))} | {src[model]} |")
    lines.append("")
    lines.append("## 2. 配对差异（相对 M6-B B5）")
    lines.append("")
    if not paired.empty:
        pg = paired.groupby(["display", "scenario_short"]).agg(
            n=("delta_vs_dosesyn", "size"), mean_delta=("delta_vs_dosesyn", "mean"),
            wins=("delta_vs_dosesyn", lambda s: int((s < 0).sum()))).reset_index()
        lines.append("| 模型 | 场景 | 可比 split 数 | 平均 ΔRMSE（负=更好） | 优于 DoseSyn 的 split 数 |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for r in pg.itertuples():
            lines.append(f"| {r.display} | {r.scenario_short} | {r.n} | {r.mean_delta:+.4f} | {r.wins}/{r.n} |")
    else:
        lines.append("暂无可配对结果。")
    lines.append("")
    lines.append("## 3. 关键解释")
    lines.append("")
    lines.append("- DeepSynBa 使用 Morgan 指纹替代 MoLFormer、CCLE 替代 NCI-60、X/X0 曲面替代 SynBa 参数，"
                 "属于 **adapted supplementary**，不进入主排名。")
    lines.append("- comboKR 在 S1/S2 使用官方 BRAID 参数与 RBF 核；S3 因 Hill 参数含测试细胞单药信息，"
                 "按协议不运行，表中记为 —。")
    lines.append("- 所有方法读取同一冻结 split 的测试矩阵；配对差异按 (scenario, split) 对齐。")
    lines.append("")
    lines.append("## 4. 资源与预算")
    lines.append("")
    lines.append("| 模型 | 参数量 | 单 split 平均耗时 (s) |")
    lines.append("| --- | ---: | ---: |")
    for model in MAIN + SUPP:
        r = summ[summ.model == model]
        if r.empty:
            continue
        p = r.params.dropna().max()
        t = r.runtime_seconds.dropna().mean()
        lines.append(f"| {DISPLAY[model]} | {'—' if pd.isna(p) else int(p)} | {'—' if pd.isna(t) else f'{t:.1f}'} |")
    lines.append("")
    lines.append("## 5. 产出")
    lines.append("")
    for f in ["external_baseline_runs.csv", "external_baseline_summary.csv",
              "external_paired_deltas.csv", "main_comparison.csv"]:
        lines.append(f"- `outputs/metrics/m7a/{f}`")
    lines.append("- `outputs/figures/m7a/macro_rmse_comparison.svg`")
    (ROOT / "m7a_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_json = dict(
        n_runs=int(len(df)),
        models=sorted(df.model.unique().tolist()),
        scenarios=sorted(df.scenario.unique().tolist()),
        main_table=pivot.reset_index().to_dict(orient="records"),
        generated_at=pd.Timestamp.utcnow().isoformat(),
    )
    (metrics_dir / "m7a_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False))
    print(json.dumps({k: summary_json[k] for k in ("n_runs", "models")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
