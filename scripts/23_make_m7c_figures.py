"""M7-C: publication-style figures for the chapter-4 results (English labels)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "outputs/metrics/m7c"
FIG = ROOT / "outputs/figures/m7c"
FIG.mkdir(parents=True, exist_ok=True)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    }
)


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(v):
    return float(v) if v not in ("", None, "nan") else np.nan


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)


def bar_panel(ax, labels, values, colors=None, title="", ylabel="Matrix-macro RMSE", rotate=25):
    x = np.arange(len(labels))
    colors = colors or [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    bars = ax.bar(x, values, color=colors, width=0.68, edgecolor="white", linewidth=0.6)
    for b, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(0.02, np.nanmax(values) * 1.18))


def fig_overall():
    rows = read(M / "table_c1_overall.csv")
    labels = ["B0", "B1 Tree", "B2 MLP", "B3 GINE", "T3", "B5"]
    models = ["M4 B0 Mean", "M4 B1 Tree", "M4 B2 MLP", "M4 B3 GINE", "M6-B B1 (T3)", "M6-B B5 (T3+FiLM)"]
    by = {r["model"]: r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), sharey=True)
    for ax, sc in zip(axes, ["S1", "S2", "S3"]):
        vals = [f(by[m][sc]) for m in models]
        bar_panel(ax, labels, vals, title=f"{sc}", rotate=20)
    axes[0].set_ylabel("Matrix-macro RMSE (lower is better)")
    fig.suptitle("Overall performance of DoseSyn and internal baselines", y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "fig4_1_overall_performance")


def fig_external():
    rows = read(M / "table_c1_external_common.csv")
    labels = ["B5 DoseSyn", "T3", "B3 GINE", "B2 MLP", "comboKR 2.0"]
    mapping = {
        "M6-B B5 (T3+FiLM)": "B5 DoseSyn",
        "M6-B B1 (T3)": "T3",
        "M4 B3 GINE": "B3 GINE",
        "M4 B2 MLP": "B2 MLP",
        "comboKR 2.0": "comboKR 2.0",
    }
    by = {mapping[r["model"]]: r for r in rows if r["model"] in mapping}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    x = np.arange(len(labels))
    w = 0.38
    s1 = [f(by[k]["S1"]) for k in labels]
    s2 = [f(by[k]["S2"]) for k in labels]
    axes[0].bar(x - w / 2, s1, w, label="S1 Random", color="#4C72B0")
    axes[0].bar(x + w / 2, s2, w, label="S2 Unseen Pair", color="#DD8452")
    for i, (a, b) in enumerate(zip(s1, s2)):
        axes[0].text(i - w / 2, a + 0.004, f"{a:.3f}", ha="center", va="bottom", rotation=90, fontsize=6)
        axes[0].text(i + w / 2, b + 0.004, f"{b:.3f}", ha="center", va="bottom", rotation=90, fontsize=6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylabel("Matrix-macro RMSE")
    axes[0].set_title("Common test matrices (S1/S2)")
    axes[0].set_ylim(0, 0.165)
    axes[0].legend(frameon=False, loc="upper center", ncol=2, fontsize=7)

    allrows = read(M / "table_c1_overall.csv")
    by2 = {r["model"]: r for r in allrows}
    ex_labels = ["B5 DoseSyn", "B3 GINE", "comboKR 2.0", "DeepSynBa (adapted)"]
    ex_models = ["M6-B B5 (T3+FiLM)", "M4 B3 GINE", "comboKR 2.0", "DeepSynBa (adapted)"]
    vals = [f(by2[m]["mean_rmse"]) for m in ex_models]
    bar_panel(axes[1], ex_labels, vals, title="Overall (mean of available scenarios)", rotate=25)
    fig.tight_layout()
    save(fig, "fig4_2_external_comparison")


def fig_ablation():
    rows = read(M / "table_c2_m6a_ablation.csv")
    labels = ["T0 B3", "T1 C0", "T2 scalar", "T3 Fourier", "T4 FiLM"]
    keys = ["t0_b3_exact", "t1_c0_m4", "b3sym_scalar", "b3sym_fourier", "b3sym_film"]
    by = {r["model"]: r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    s1 = [f(by[k]["S1"]) for k in keys]
    s3 = [f(by[k]["S3"]) if by[k]["S3"] not in ("", None, "None") else np.nan for k in keys]
    x = np.arange(len(labels))
    axes[0].bar(x, s1, color="#4C72B0", width=0.65)
    for i, v in enumerate(s1):
        axes[0].text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_title("S1 seed42 (validation)")
    axes[0].set_ylabel("Matrix-macro RMSE")
    axes[0].set_ylim(0, 0.22)

    m6b = read(M / "table_c2_m6b_film.csv")
    by2 = {r["model"]: r for r in m6b}
    labs = ["B1 T3", "B5 DoseSyn"]
    scs = ["S1", "S2", "S3"]
    w = 0.25
    for j, sc in enumerate(scs):
        vals = [f(by2["b1_t3_fourier"][sc]), f(by2["b5_t3_film_reg"][sc])]
        axes[1].bar(np.arange(2) + (j - 1) * w, vals, w, label=sc, color=PALETTE[j])
        for i, v in enumerate(vals):
            axes[1].text(i + (j - 1) * w, v + 0.004, f"{v:.3f}", ha="center", fontsize=6)
    axes[1].set_xticks(np.arange(2))
    axes[1].set_xticklabels(labs)
    axes[1].set_title("Regularized FiLM on full data")
    axes[1].set_ylim(0, 0.23)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig4_3_ablation")


def fig_center():
    rows = read(M / "table_c3_center_holdout.csv")
    labels = ["B5 DoseSyn", "B3 GINE", "T3", "B2 MLP"]
    mapping = {"M6-B B5 (DoseSyn)": "B5 DoseSyn", "M4 B3 GINE": "B3 GINE", "M6-B B1 (T3)": "T3", "M4 B2 MLP-Dose": "B2 MLP"}
    by = {mapping[r["model"]]: r for r in rows if r["model"] in mapping}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(labels))
    w = 0.26
    for j, sc in enumerate(["S1", "S2", "S3"]):
        vals = [f(by[k][sc]) for k in labels]
        ax.bar(x + (j - 1) * w, vals, w, label=sc, color=PALETTE[j])
        for i, v in enumerate(vals):
            ax.text(i + (j - 1) * w, v + 0.004, f"{v:.3f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Held-out matrix-macro RMSE")
    ax.set_title("Center 2x2 dose-position holdout")
    ax.legend(frameon=False, ncol=3)
    ax.set_ylim(0, 0.25)
    fig.tight_layout()
    save(fig, "fig4_4_center_holdout")


def fig_sparse():
    rows = read(M / "table_c4_sparsity.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), sharey=True)
    masks = [("center_holdout_75", "75%"), ("checkerboard_holdout_50", "50%"), ("corners_holdout_25", "25%")]
    for ax, sc in zip(axes, ["S1", "S2", "S3"]):
        for model, color, marker in [("M6-B B5 (DoseSyn)", "#C44E52", "o"), ("M4 B3 GINE", "#4C72B0", "s")]:
            vals = []
            for mask, _ in masks:
                r = [x for x in rows if x["mask"] == mask and x["model"] == model][0]
                vals.append(f(r[sc]))
            ax.plot([1, 2, 3], vals, marker=marker, color=color, label=model.replace("M6-B ", "").replace(" (DoseSyn)", ""))
            for xi, v in zip([1, 2, 3], vals):
                ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=6)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([m[1] for m in masks])
        ax.set_xlabel("Observed dose positions")
        ax.set_title(sc)
    axes[0].set_ylabel("Held-out matrix-macro RMSE")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Performance under increasing dose-observation sparsity", y=1.03, fontsize=10)
    fig.tight_layout()
    save(fig, "fig4_5_sparsity")


def fig_symmetry():
    path = M / "table_c5_symmetry.csv"
    if not path.exists():
        print("symmetry table not found, skip")
        return
    rows = read(path)
    labels = [r["scenario"] for r in rows]
    mx = [f(r["max_abs_error"]) for r in rows]
    mn = [f(r["mean_abs_error"]) for r in rows]
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, mx, 0.36, label="max", color="#4C72B0")
    ax.bar(x + 0.18, mn, 0.36, label="mean", color="#DD8452")
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1e-4)
    ax.axhline(1e-5, color="gray", ls="--", lw=0.8, label="1e-5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Absolute swap error")
    ax.set_title("Drug-order exchange symmetry (B5)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig4_6_symmetry")


def fig_complexity():
    rows = read(M / "table_c7_complexity.csv")
    rows = [r for r in rows if r["n_parameters"] not in ("", None, "None")]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for i, r in enumerate(rows):
        x = float(r["n_parameters"])
        y = f(r["mean_rmse"])
        ax.scatter(x, y, s=45, color=PALETTE[i % len(PALETTE)], zorder=3)
        ax.annotate(r["model"].replace("M6-B ", "").replace("M4 ", ""), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Mean matrix-macro RMSE")
    ax.set_title("Accuracy vs model size")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    save(fig, "fig4_7_complexity")


def fig_runtime():
    rows = read(M / "table_c7_complexity.csv")
    rows = [r for r in rows if r["train_wall_s_s1_seed42"] not in ("", None, "None")]
    labels = [r["model"].replace("M6-B ", "").replace("M4 ", "") for r in rows]
    vals = [f(r["train_wall_s_s1_seed42"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    bars = ax.bar(labels, vals, color="#55A868", width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 15, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_ylabel("Training wall time (s)")
    ax.set_title("S1 seed42 training cost (same node1 V100)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save(fig, "fig4_8_training_cost")


def main():
    fig_overall()
    fig_external()
    fig_ablation()
    fig_center()
    fig_sparse()
    fig_symmetry()
    fig_complexity()
    fig_runtime()
    files = sorted(p.name for p in FIG.glob("fig4_*"))
    (FIG / "figures_manifest.json").write_text(json.dumps({"figures": files}, indent=2) + "\n", encoding="utf-8")
    print("\n".join(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
