"""M7-C: build the chapter-4 result tables from frozen experiment artefacts.

Reads only existing M4/M5/M6/M7-A/M7-B outputs; writes new M7-C tables.
No model training, no modification of upstream results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs/metrics/m7c"
TAB = ROOT / "docs/paper/tables"
FIG = ROOT / "outputs/figures/m7c"


def read(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def fmt(v, nd=4):
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.{nd}f}"


def md_table(df: pd.DataFrame, cols: Dict[str, str], nd: int = 4) -> str:
    header = "| " + " | ".join(cols.values()) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for key in cols:
            v = row.get(key)
            if key in ("n_parameters", "params") and v is not None and not pd.isna(v):
                cells.append(f"{int(v):,}")
            elif isinstance(v, float):
                if key in ("S1", "S2", "S3", "mean", "mean_rmse", "S1_rmse", "s1_val_macro"):
                    cells.append(fmt(v, 4))
                elif "wall" in key or "ms_per" in key:
                    cells.append(fmt(v, 2))
                else:
                    cells.append(fmt(v, nd))
            else:
                cells.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def c1_overall() -> pd.DataFrame:
    main = read("outputs/metrics/m7a/main_comparison.csv")
    family = {
        "M4 B0 Mean": "internal / trivial",
        "M4 B1 Tree": "internal / tree",
        "M4 B2 MLP": "internal / MLP",
        "M4 B3 GINE": "internal / GINE",
        "M6-B B1 (T3)": "internal / T3",
        "M6-B B5 (T3+FiLM)": "proposed",
        "comboKR 2.0": "external / faithful",
        "DeepSynBa (adapted)": "external / adapted",
    }
    params = {
        "M4 B0 Mean": 1,
        "M4 B1 Tree": 24_400,
        "M4 B2 MLP": 65_153,
        "M4 B3 GINE": 284_545,
        "M6-B B1 (T3)": 498_818,
        "M6-B B5 (T3+FiLM)": 535_682,
        "DeepSynBa (adapted)": 11_560_833,
    }
    order = list(family)
    main["_order"] = main["display"].map({k: i for i, k in enumerate(order)})
    main = main.sort_values("_order").drop(columns="_order")
    main["family"] = main["display"].map(family)
    main["n_parameters"] = main["display"].map(params)
    main["note"] = main["display"].map(
        {
            "comboKR 2.0": "S3 blocked (test-cell monotherapy leakage)",
            "DeepSynBa (adapted)": "feature/objective adaptation, supplementary",
            "M6-B B5 (T3+FiLM)": "final DoseSyn model",
        }
    ).fillna("")
    main = main.rename(columns={"display": "model", "mean": "mean_rmse"})
    main = main[["model", "family", "S1", "S2", "S3", "mean_rmse", "n_parameters", "note"]]
    main.to_csv(OUT / "table_c1_overall.csv", index=False)
    return main


def c1_gains(c1: pd.DataFrame) -> pd.DataFrame:
    b5 = c1[c1.model == "M6-B B5 (T3+FiLM)"].iloc[0]
    rows = []
    for _, r in c1.iterrows():
        if r.model == "M6-B B5 (T3+FiLM)":
            continue
        if r.model.startswith("comboKR") or r.model.startswith("DeepSynBa"):
            continue
        for sc in ["S1", "S2", "S3"]:
            if pd.isna(r[sc]) or pd.isna(b5[sc]):
                continue
            rows.append(
                {
                    "baseline": r.model,
                    "scenario": sc,
                    "baseline_rmse": float(r[sc]),
                    "dosesyn_rmse": float(b5[sc]),
                    "abs_gain": float(r[sc] - b5[sc]),
                    "rel_gain_pct": float((r[sc] - b5[sc]) / r[sc] * 100.0),
                }
            )
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "table_c1_gains.csv", index=False)
    return d


def c1_external() -> pd.DataFrame:
    common = read("outputs/metrics/m7a/common_subset_summary.csv")
    scen = {"s1_random_condition": "S1", "s2_unseen_pair": "S2"}
    common["scenario"] = common["scenario"].map(scen).fillna(common["scenario"])
    p = common.pivot_table(index="model", columns="scenario", values="macro_rmse_mean")
    n = common.groupby("model")["n_runs"].max()
    p["n_runs"] = n
    p = p.reset_index().rename(columns={"model": "model"})
    order = [
        "M6-B B5 (T3+FiLM)",
        "M6-B B1 (T3)",
        "M4 B3 GINE",
        "M4 B2 MLP",
        "comboKR 2.0",
    ]
    p["_o"] = p["model"].map({k: i for i, k in enumerate(order)})
    p = p.sort_values("_o").drop(columns="_o")
    p = p[["model", "S1", "S2", "n_runs"]]
    p.to_csv(OUT / "table_c1_external_common.csv", index=False)
    return p


def c2_m6a() -> pd.DataFrame:
    d = read("outputs/metrics/m6a/validation_runs.csv")
    p = d.pivot_table(index="model", columns="scenario", values="matrix_macro_rmse")
    p = p.reset_index()
    labels = {
        "b3sym_scalar": "T2 B3-trunk + symmetric + scalar",
        "b3sym_fourier": "T3 B3-trunk + symmetric + Fourier",
        "b3sym_film": "T4 B3-trunk + symmetric + Fourier + FiLM",
    }
    p["label"] = p["model"].map(labels).fillna(p["model"])
    p["S1"] = p.get("s1_random_condition")
    p["S3"] = p.get("s3_unseen_cell")
    # T0 parity and T1 control are single-scenario references.
    t0 = 0.08005502275233599
    t1 = 0.181985
    rows = [
        {"model": "t0_b3_exact", "label": "T0 M4 B3 exact checkpoint", "S1": t0, "S3": None},
        {"model": "t1_c0_m4", "label": "T1 C0 ordered control + M4 recipe", "S1": t1, "S3": None},
    ]
    for _, r in p.iterrows():
        rows.append({"model": r["model"], "label": r["label"], "S1": r["S1"], "S3": r["S3"]})
    out = pd.DataFrame(rows)
    order = ["t0_b3_exact", "t1_c0_m4", "b3sym_scalar", "b3sym_fourier", "b3sym_film"]
    out["_order"] = out["model"].map({k: i for i, k in enumerate(order)}).fillna(99)
    out = out.sort_values("_order").drop(columns="_order")
    out.to_csv(OUT / "table_c2_m6a_ablation.csv", index=False)
    return out


def c2_m6b() -> pd.DataFrame:
    d = read("outputs/metrics/m6b/formal_summary.csv")
    d = d[d.model.isin(["b1_t3_fourier", "b5_t3_film_reg"])]
    p = d.pivot_table(index="model", columns="scenario", values="test_macro_mean")
    p = p.reset_index()
    p["S1"] = p["s1_random_condition"]
    p["S2"] = p["s2_unseen_pair"]
    p["S3"] = p["s3_unseen_cell"]
    p["mean"] = p[["S1", "S2", "S3"]].mean(axis=1)
    p["label"] = p.model.map(
        {
            "b1_t3_fourier": "B1 T3 (unregularized, no FiLM)",
            "b5_t3_film_reg": "B5 DoseSyn (regularized FiLM)",
        }
    )
    p["params"] = p.model.map({"b1_t3_fourier": 498_818, "b5_t3_film_reg": 535_682})
    p = p[["model", "label", "S1", "S2", "S3", "mean", "params"]]
    p.to_csv(OUT / "table_c2_m6b_film.csv", index=False)
    return p


def c2_m5() -> pd.DataFrame:
    rows = []
    for p in sorted((ROOT / "outputs/runs/m5").glob("m5__s1_random_condition__seed_42__*/val_metrics.json")):
        d = json.loads(p.read_text())
        variant = p.parent.name.split("__")[-1]
        rows.append({"variant": variant, "s1_val_macro": d.get("matrix_macro_rmse")})
    out = pd.DataFrame(rows)
    if not out.empty:
        order = ["c0_control", "c1_symmetric", "c2_fourier", "c3_film", "c4_mono", "c5_boundary", "c6_full"]
        out["_o"] = out.variant.map({k: i for i, k in enumerate(order)}).fillna(99)
        out = out.sort_values("_o").drop(columns="_o")
    out.to_csv(OUT / "table_c2_m5_components.csv", index=False)
    return out


def scenario_pivot(df: pd.DataFrame, models: list[str], value="macro_mean") -> pd.DataFrame:
    p = df.pivot_table(index="display", columns="scenario", values=value)
    p = p.reindex([m for m in models if m in p.index])
    p = p.rename(columns={"s1_random_condition": "S1", "s2_unseen_pair": "S2", "s3_unseen_cell": "S3"})
    return p.reset_index()


def c3_center() -> pd.DataFrame:
    d = read("outputs/metrics/m7b/b1_center_summary.csv")
    d = d[d["mask"] == "center_holdout_75"]
    models = ["M6-B B5 (DoseSyn)", "M4 B3 GINE", "M6-B B1 (T3)", "M4 B2 MLP-Dose"]
    p = scenario_pivot(d, models)
    p["mean"] = p[["S1", "S2", "S3"]].mean(axis=1, skipna=True)
    p = p.rename(columns={"display": "model"})
    p.to_csv(OUT / "table_c3_center_holdout.csv", index=False)
    return p


def c4_sparsity() -> pd.DataFrame:
    d = read("outputs/metrics/m7b/sparsity_summary.csv")
    d = d[d["display"].isin(["M6-B B5 (DoseSyn)", "M4 B3 GINE"])]
    d = d[d["mask"].isin(["center_holdout_75", "checkerboard_holdout_50", "corners_holdout_25"])]
    p = d.pivot_table(index=["mask", "display"], columns="scenario", values="macro_mean").reset_index()
    p = p.rename(
        columns={
            "display": "model",
            "s1_random_condition": "S1",
            "s2_unseen_pair": "S2",
            "s3_unseen_cell": "S3",
        }
    )
    p["mean"] = p[["S1", "S2", "S3"]].mean(axis=1)
    p.to_csv(OUT / "table_c4_sparsity.csv", index=False)
    return p


def c4_boundary() -> pd.DataFrame:
    d = read("outputs/metrics/m7b/boundary_summary.csv")
    models = ["M4 B3 GINE", "M6-B B5 (DoseSyn)", "M6-B B1 (T3)"]
    p = scenario_pivot(d, models)
    p["mean"] = p[["S1", "S2", "S3"]].mean(axis=1, skipna=True)
    p = p.rename(columns={"display": "model"})
    p.to_csv(OUT / "table_c4_boundary.csv", index=False)
    return p


def c7_complexity(c1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in c1.iterrows():
        if r.model.startswith(("comboKR", "DeepSynBa")):
            wall = None
            if r.model.startswith("comboKR"):
                wall = None
            else:
                wall = None
        else:
            wall = None
        rows.append(
            {
                "model": r.model,
                "n_parameters": r.n_parameters if not pd.isna(r.n_parameters) else None,
                "S1_rmse": r.S1,
                "mean_rmse": r.mean_rmse,
            }
        )
    # measured S1 seed42 wall time from the respective run records
    wall_map = {
        "M4 B0 Mean": 4.439,
        "M4 B1 Tree": None,
        "M4 B2 MLP": 47.517,
        "M4 B3 GINE": 239.218,
        "M6-B B1 (T3)": 1063.0,
        "M6-B B5 (T3+FiLM)": 928.4,
    }
    for r in rows:
        r["train_wall_s_s1_seed42"] = wall_map.get(r["model"])
    infer = {}
    ipath = OUT / "inference_timing.json"
    if ipath.exists():
        for rec in json.loads(ipath.read_text()):
            infer[rec["model"]] = rec["ms_per_matrix_median"]
    for r in rows:
        r["inference_ms_per_matrix_cpu"] = {
            "M6-B B5 (T3+FiLM)": infer.get("B5 DoseSyn"),
            "M4 B3 GINE": infer.get("B3 GINE"),
        }.get(r["model"])
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table_c7_complexity.csv", index=False)
    return out


def main() -> int:
    for d in (OUT, TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)
    c1 = c1_overall()
    gains = c1_gains(c1)
    ext = c1_external()
    m6a = c2_m6a()
    m6b = c2_m6b()
    m5 = c2_m5()
    c3 = c3_center()
    c4 = c4_sparsity()
    b4 = c4_boundary()
    c7 = c7_complexity(c1)

    # Markdown tables for direct chapter use.
    (TAB / "table4_1_overall.md").write_text(
        md_table(c1, {"model": "Model", "S1": "S1 RMSE", "S2": "S2 RMSE", "S3": "S3 RMSE", "mean_rmse": "Mean", "n_parameters": "Params"}),
        encoding="utf-8",
    )
    (TAB / "table4_1_gains.md").write_text(
        md_table(gains, {"baseline": "Baseline", "scenario": "Scenario", "rel_gain_pct": "RMSE reduction vs baseline (%)"}, nd=2),
        encoding="utf-8",
    )
    (TAB / "table4_2_external.md").write_text(
        md_table(ext, {"model": "Model", "S1": "S1 RMSE", "S2": "S2 RMSE"}),
        encoding="utf-8",
    )
    (TAB / "table4_3_m6a_ablation.md").write_text(
        md_table(m6a, {"label": "Configuration", "S1": "S1 val RMSE", "S3": "S3 val RMSE"}),
        encoding="utf-8",
    )
    (TAB / "table4_4_m6b_film.md").write_text(
        md_table(m6b, {"label": "Model", "S1": "S1 RMSE", "S2": "S2 RMSE", "S3": "S3 RMSE", "mean": "Mean", "params": "Params"}),
        encoding="utf-8",
    )
    (TAB / "table4_5_center_holdout.md").write_text(
        md_table(c3, {"model": "Model", "S1": "S1 center", "S2": "S2 center", "S3": "S3 center", "mean": "Mean"}),
        encoding="utf-8",
    )
    (TAB / "table4_4b_m5_components.md").write_text(
        md_table(m5, {"variant": "Variant", "s1_val_macro": "S1 validation RMSE"}),
        encoding="utf-8",
    )
    (TAB / "table4_6_sparsity.md").write_text(
        md_table(c4, {"mask": "Mask", "model": "Model", "S1": "S1", "S2": "S2", "S3": "S3", "mean": "Mean"}),
        encoding="utf-8",
    )
    (TAB / "table4_7_boundary.md").write_text(
        md_table(b4, {"model": "Model", "S1": "S1", "S2": "S2", "S3": "S3", "mean": "Mean"}),
        encoding="utf-8",
    )
    (TAB / "table4_8_complexity.md").write_text(
        md_table(c7, {"model": "Model", "n_parameters": "Params", "S1_rmse": "S1 RMSE", "train_wall_s_s1_seed42": "S1 train wall (s)", "inference_ms_per_matrix_cpu": "Inference (ms/matrix, CPU)"}, nd=2),
        encoding="utf-8",
    )

    summary = {
        "generated_from": {
            "m7a_main": "outputs/metrics/m7a/main_comparison.csv",
            "m7a_common": "outputs/metrics/m7a/common_subset_summary.csv",
            "m6a": "outputs/metrics/m6a/validation_runs.csv",
            "m6b": "outputs/metrics/m6b/formal_summary.csv",
            "m7b_center": "outputs/metrics/m7b/b1_center_summary.csv",
            "m7b_sparsity": "outputs/metrics/m7b/sparsity_summary.csv",
            "m7b_boundary": "outputs/metrics/m7b/boundary_summary.csv",
        },
        "rows": {
            "c1_overall": len(c1),
            "c1_gains": len(gains),
            "c1_external_common": len(ext),
            "c2_m6a": len(m6a),
            "c2_m6b": len(m6b),
            "c2_m5": len(m5),
            "c3_center": len(c3),
            "c4_sparsity": len(c4),
            "c4_boundary": len(b4),
            "c7_complexity": len(c7),
        },
    }
    (OUT / "m7c_tables_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
