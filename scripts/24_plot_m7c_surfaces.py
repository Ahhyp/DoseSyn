"""M7-C6: dose-response surface case studies from frozen B5 test predictions.

Case-selection rule (fixed before looking at errors): among test matrices with
y_true std >= 0.05 and range >= 0.2, choose the matrices whose per-matrix RMSE
is closest to the 10th, 50th and 90th percentile of the eligible distribution.
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
M = ROOT / "outputs/metrics/m7c"
FIG = ROOT / "outputs/figures/m7c"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "figure.dpi": 120,
    }
)

FULL = ROOT / "outputs/runs/m6b/m6b__b5_t3_film_reg__s1_random_condition__seed_42/predictions_test.parquet"
INTERP = ROOT / "outputs/runs/m7b/m7b__center_holdout_75__m6b_b5_film__s1_random_condition__seed_42/predictions_test.parquet"


def per_matrix_rmse(df):
    g = df.groupby("matrix_id")
    rmse = g.apply(lambda x: float(np.sqrt(np.mean((x.y_true - x.y_pred) ** 2))), include_groups=False)
    std = g["y_true"].std()
    rng = g["y_true"].apply(lambda s: float(s.max() - s.min()))
    meta = df.groupby("matrix_id")[["cell_line", "drug_a_name", "drug_b_name"]].first()
    out = meta.join(rmse.rename("rmse")).join(std.rename("std")).join(rng.rename("range"))
    return out.reset_index()


def choose(df, q):
    elig = df[(df["std"] >= 0.05) & (df["range"] >= 0.2)].copy()
    target = np.percentile(elig["rmse"], q)
    idx = (elig["rmse"] - target).abs().idxmin()
    return elig.loc[idx]


def grid(df, mid, col):
    d = df[df.matrix_id == mid].sort_values(["row_idx", "col_idx"])
    return d.pivot(index="row_idx", columns="col_idx", values=col).values, d


def surface_panel(ax, mat, doses_a, doses_b, title, cmap="viridis", vmin=None, vmax=None):
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([f"{x:.4g}" for x in doses_b], rotation=30, fontsize=6)
    ax.set_yticklabels([f"{x:.4g}" for x in doses_a], fontsize=6)
    ax.set_xlabel("Drug B concentration (µM)")
    ax.set_ylabel("Drug A concentration (µM)")
    ax.set_title(title)
    return im


def plot_case(df, row, name, tag):
    mid = row.matrix_id
    yt, d = grid(df, mid, "y_true")
    yp, _ = grid(df, mid, "y_pred")
    err = np.abs(yt - yp)
    da = sorted(d.dose_a_um.unique())
    db = sorted(d.dose_b_um.unique())
    vmax = max(yt.max(), yp.max())
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
    im0 = surface_panel(axes[0], yt, da, db, "Ground truth", vmin=0, vmax=vmax)
    im1 = surface_panel(axes[1], yp, da, db, "DoseSyn prediction", vmin=0, vmax=vmax)
    im2 = surface_panel(axes[2], err, da, db, "Absolute error", cmap="magma")
    for ax, im in zip(axes, [im0, im1, im2]):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="X/X0")
    fig.suptitle(
        f"{row.cell_line} | {row.drug_a_name} + {row.drug_b_name} | RMSE={row.rmse:.3f} ({tag})",
        y=1.03,
        fontsize=9,
    )
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)


def plot_interpolation_mask():
    df = pd.read_parquet(INTERP)
    rm = per_matrix_rmse(df)
    elig = rm[(rm["std"] >= 0.05) & (rm["range"] >= 0.2)]
    med = elig.iloc[(elig["rmse"] - elig["rmse"].median()).abs().argsort().iloc[0]]
    mid = med.matrix_id
    yt, d = grid(df, mid, "y_true")
    yp, _ = grid(df, mid, "y_pred")
    err = np.abs(yt - yp)
    mask = d.pivot(index="row_idx", columns="col_idx", values="is_held").values.astype(bool)
    da = sorted(d.dose_a_um.unique())
    db = sorted(d.dose_b_um.unique())
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 2.8))
    vmax = max(yt.max(), yp.max())
    im0 = surface_panel(axes[0], yt, da, db, "Ground truth", vmin=0, vmax=vmax)
    im1 = surface_panel(axes[1], yp, da, db, "Prediction (12 observed)", vmin=0, vmax=vmax)
    im2 = surface_panel(axes[2], err, da, db, "Absolute error", cmap="magma")
    axes[3].imshow(mask, origin="lower", aspect="auto", cmap=matplotlib.colors.ListedColormap(["#BBDEFB", "#E57373"]))
    for i in range(4):
        for j in range(4):
            axes[3].text(j, i, "held" if mask[i, j] else "obs", ha="center", va="center", fontsize=6)
    axes[3].set_title("Observation mask")
    axes[3].set_xticks(range(4)); axes[3].set_yticks(range(4))
    axes[3].set_xticklabels([f"{x:.4g}" for x in db], rotation=30, fontsize=6)
    axes[3].set_yticklabels([f"{x:.4g}" for x in da], fontsize=6)
    axes[3].set_xlabel("Drug B (µM)"); axes[3].set_ylabel("Drug A (µM)")
    for ax, im in zip(axes[:3], [im0, im1, im2]):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="X/X0")
    fig.suptitle(f"Center-holdout interpolation case | {med.cell_line} | RMSE={med.rmse:.3f}", y=1.04, fontsize=9)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(FIG / f"fig4_12_interpolation_mask.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)
    return med


def plot_pair_curves(df, row):
    mid = row.matrix_id
    d = df[df.matrix_id == mid].sort_values(["row_idx", "col_idx"])
    da = sorted(d.dose_a_um.unique())
    a_level = da[1] if len(da) > 1 else da[0]
    sub = d[np.isclose(d.dose_a_um, a_level)]
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    ax.plot(sub.dose_b_um, sub.y_true, "o-", color="#4C72B0", label="Ground truth")
    ax.plot(sub.dose_b_um, sub.y_pred, "s--", color="#C44E52", label="DoseSyn")
    ax.set_xscale("log")
    ax.set_xlabel("Drug B concentration (µM, log scale)")
    ax.set_ylabel("X/X0")
    ax.set_title(f"{row.cell_line}: {row.drug_a_name} + {row.drug_b_name}\n(A = {a_level:.4g} µM)", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(FIG / f"fig4_13_pair_curves.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)


def main():
    df = pd.read_parquet(FULL)
    rm = per_matrix_rmse(df)
    med = plot_interpolation_mask() if INTERP.exists() else None
    rows = []
    for q, tag, name in [(10, "good (p10)", "fig4_9_surface_good"), (50, "median (p50)", "fig4_10_surface_median"), (90, "hard (p90)", "fig4_11_surface_hard")]:
        r = choose(rm, q)
        plot_case(df, r, name, tag)
        rows.append({"figure": name, "quantile": q, "matrix_id": r.matrix_id, "cell_line": r.cell_line,
                     "drug_a_name": r.drug_a_name, "drug_b_name": r.drug_b_name,
                     "rmse": float(r.rmse), "std": float(r["std"]), "range": float(r["range"])})
    good = choose(rm, 10)
    plot_pair_curves(df, good)
    out = pd.DataFrame(rows)
    out.to_csv(M / "surface_cases.csv", index=False)
    meta = {
        "source": str(FULL.relative_to(ROOT)),
        "selection_rule": "test matrices with y_true std>=0.05 and range>=0.2; nearest to p10/p50/p90 of per-matrix RMSE",
        "n_eligible": int(((rm["std"] >= 0.05) & (rm["range"] >= 0.2)).sum()),
        "n_test": len(rm),
        "cases": rows,
        "interpolation_case": None if med is None else {"matrix_id": med.matrix_id, "cell_line": med.cell_line, "rmse": float(med.rmse)},
    }
    (M / "surface_cases.json").write_text(json.dumps(meta, indent=2, default=float) + "\n", encoding="utf-8")
    (ROOT / "docs/audit/m7c").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs/audit/m7c/surface_cases.csv").write_text(out.to_csv(index=False), encoding="utf-8")
    print(out.to_string(index=False))
    print(json.dumps(meta["interpolation_case"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
