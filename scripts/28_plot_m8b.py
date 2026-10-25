"""M8-B figures: dual-dimension scatter, top-candidate ranking, 4x4 heatmaps.

Run with /tmp/m7venv/bin/python (matplotlib 3.8.4). Reads only M8-B CSVs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "outputs/metrics/m8b"
FIG = ROOT / "outputs/figures/m8b"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 220,
})


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(name)


def scatter():
    d = pd.read_csv(MET / "bridge_candidates.csv")
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        d["p_mean"], d["y_pred_mean"],
        c=d["y_pred_min"], cmap="viridis_r", s=18 + 90 * d["p_std"] / max(d["p_std"].max(), 1e-9),
        alpha=0.8, edgecolor="white", linewidth=0.4,
    )
    ax.axvline(0.5, color="#888", ls="--", lw=0.8)
    ax.axhline(d["y_pred_mean"].median(), color="#888", ls=":", lw=0.8)
    # annotate the 5 highest-ConfSyn and 5 lowest-response candidates
    ann = pd.concat([d.nlargest(5, "p_mean"), d.nsmallest(5, "y_pred_mean")]).drop_duplicates("matrix_id")
    ax.set_xlim(-0.03, 1.12)
    for _, r in ann.iterrows():
        right = r.p_mean > 0.80
        ax.annotate(f"{r.drug_a_name}+{r.drug_b_name}", (r.p_mean, r.y_pred_mean),
                    fontsize=5.5, xytext=(-4 if right else 4, 3), textcoords="offset points",
                    ha="right" if right else "left", clip_on=True)
    ax.set_xlabel("ConfSyn ensemble probability  p(synergy >= 0)")
    ax.set_ylabel("DoseSyn B5 predicted mean X/X0")
    ax.set_title("Frozen-model joint landscape (S2 unseen pair, A-level)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("DoseSyn surface minimum X/X0")
    save(fig, "fig8_1_dual_dimension_scatter")


def ranking():
    d = pd.read_csv(MET / "joint_ranking_top30.csv").head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    y = np.arange(len(d))
    ax.errorbar(d["p_mean"], y, xerr=[d["p_mean"] - d["p_min"], d["p_max"] - d["p_mean"]],
                fmt="o", color="#2b6cb0", ecolor="#90cdf4", capsize=3, ms=5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.cell_line}: {r.drug_a_name}+{r.drug_b_name}" for _, r in d.iterrows()], fontsize=7)
    ax.axvline(0.5, color="#888", ls="--", lw=0.8)
    ax.set_xlabel("ConfSyn p(synergy >= 0), five-fold mean and range")
    ax.set_title("Top-15 candidates by frozen ConfSyn")
    save(fig, "fig8_2_top_candidates")


def heatmaps():
    surf = pd.read_json(MET / "dosesyn_surfaces.json")
    d = pd.read_csv(MET / "bridge_candidates.csv").head(6)
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.8))
    fig.subplots_adjust(hspace=0.62, wspace=0.38, top=0.88)
    for ax, (_, r) in zip(axes.ravel(), d.iterrows()):
        row = surf[surf.matrix_id == r.matrix_id].iloc[0]
        arr = np.asarray(row["surface"], dtype=float)
        im = ax.imshow(arr, origin="lower", cmap="viridis", aspect="auto", vmin=0, vmax=max(1.0, arr.max()))
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"{x:.2g}" for x in row["dose_b_um"]], fontsize=6, rotation=35)
        ax.set_yticks(range(4))
        ax.set_yticklabels([f"{x:.2g}" for x in row["dose_a_um"]], fontsize=6)
        ax.set_xlabel("Drug B (µM)", fontsize=7)
        ax.set_ylabel("Drug A (µM)", fontsize=7)
        ax.set_title(f"{r.cell_line}\n{r.drug_a_name}+{r.drug_b_name}  p={r.p_mean:.2f}", fontsize=7)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if arr[i, j] < 0.55 * max(1.0, arr.max()) else "black")
    fig.suptitle("Frozen DoseSyn B5 4x4 X/X0 surfaces for top ConfSyn candidates", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.6, label="predicted X/X0")
    save(fig, "fig8_3_top_surface_heatmaps")


if __name__ == "__main__":
    scatter()
    ranking()
    heatmaps()
