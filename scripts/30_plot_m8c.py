"""Render M8-C / chapter-4 figures in English labels (SVG + PDF + PNG)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.utils.config import load_config, resolve_path  # noqa: E402

CONFIG = ROOT / "configs/m8c/m8c.yaml"
COLORS = ["#3a6ea5", "#c44e52", "#55a868", "#8172b2", "#ccb974", "#64b5cd"]


def _save(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(base.with_suffix("." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, w, h, text, fc, ec="#333333", fs=8.0):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.0, edgecolor=ec, facecolor=fc))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def workflow(path: Path):
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.0); ax.axis("off")
    ax.text(0.5, 0.97, "Frozen two-model screening workflow", ha="center", va="top", fontsize=11)

    xs = [0.02, 0.27, 0.52, 0.77]
    headers = ["Frozen inputs", "Model inference", "Bridge layer", "Auditable output"]
    for x, h in zip(xs, headers):
        _box(ax, (x, 0.80), 0.21, 0.09, h, "#dddddd", fs=9.2)
    boxes = [
        ["O'Neil S2 seed42 test\n2,879 matrices", "DoseSyn Primary-v1\n18,599 matrices", "M3 splits + scalers\n(train-fitted)"],
        ["DoseSyn B5\n5.36e5 params\nX/X0 surface", "ConfSyn 5 folds\nDrugCombDB\n88.4M params", "Frozen checkpoints\n+ cached ChemBERTa"],
        ["A-level candidates\n300 matrices", "Condition key\ncell + unordered pair", "O'Neil raw trace\nsource_row + reps"],
        ["120 scored / 180 SN-38\nexplicitly unscoreable", "p(synergy>=0)\nX/X0 kept separate", "4 cases\n+ source rows"],
    ]
    ys = [0.60, 0.39, 0.18]
    for x, col in zip(xs, boxes):
        for y, text in zip(ys, col):
            _box(ax, (x, y), 0.21, 0.15, text, "#f4f7fb", fs=7.1)
    # arrows
    for x0, x1 in [(0.23, 0.27), (0.48, 0.52), (0.73, 0.77)]:
        ax.add_patch(FancyArrowPatch((x0, 0.5), (x1, 0.5), arrowstyle="-|>", mutation_scale=10,
                                     linewidth=1.0, color="#333333"))
    ax.text(0.5, 0.07, "No retraining. No test-label selection. X/X0 is a response, not a synergy score.",
            ha="center", fontsize=7.6, color="#444444")
    _save(fig, path / "fig4_14_workflow")


def landscape(scored: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    s = scored.copy()
    sc = ax.scatter(s["p_mean"], s["y_pred_mean"], c=s["y_pred_min"], s=28,
                    cmap="viridis_r", vmin=0, vmax=0.6, alpha=0.9, edgecolor="none")
    ax.axvline(0.9, ls="--", lw=0.8, color="#666666")
    ax.axhline(s["y_pred_mean"].median(), ls=":", lw=0.8, color="#666666")
    ax.set_xlabel("ConfSyn five-fold mean probability  p(synergy >= 0)")
    ax.set_ylabel("DoseSyn B5 predicted mean X/X0")
    ax.set_title("Dual-signal landscape (S2 unseen pair, A-level, 120 scored)")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("DoseSyn surface minimum X/X0")
    ax.set_xlim(-0.02, 1.14)
    ax.grid(alpha=0.15)
    # annotate the four pre-specified cases
    for _, r in s[s["matrix_id"].isin([
        "MTX_0789c8f4f5a3", "MTX_19a1a1a84643", "MTX_0c18c12f877c", "MTX_f8a05122d84e"
    ])].iterrows():
        right = r["p_mean"] > 0.88
        ax.annotate(f"{r['cell_line']} {r['drug_a_name']}+{r['drug_b_name']}",
                    (r["p_mean"], r["y_pred_mean"]), fontsize=5.8,
                    xytext=(-5 if right else 4, 4), textcoords="offset points",
                    ha="right" if right else "left", clip_on=True)
    _save(fig, path / "fig4_15_dual_signal_landscape")


def funnel(path: Path, n_total=300, n_scored=120, n_dual=27):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    stages = ["A-level candidates", "ConfSyn-scoreable", "Dual-signal", "Traced cases"]
    vals = [n_total, n_scored, n_dual, 4]
    colors = ["#b7c9e2", "#7fa8d4", "#c44e52", "#55a868"]
    y = np.arange(len(stages))[::-1]
    ax.barh(y, vals, color=colors, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 5, yi, str(v), va="center", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels(stages, fontsize=8.5)
    ax.set_xlabel("Number of matrices / cases")
    ax.set_xlim(0, 330)
    ax.set_title("Candidate funnel (explicit exclusions, no silent drops)")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path / "fig4_17_candidate_funnel")


def cases_figure(rows: pd.DataFrame, cases: pd.DataFrame, path: Path):
    n = len(cases)
    fig, axes = plt.subplots(n, 3, figsize=(10.0, 2.6 * n))
    if n == 1:
        axes = np.asarray([axes])
    for i, (_, c) in enumerate(cases.iterrows()):
        d = rows[rows["case_id"] == c["case_id"]]
        a_vals = np.sort(d["dose_a_um"].unique())
        b_vals = np.sort(d["dose_b_um"].unique())
        raw = d.pivot(index="row_idx", columns="col_idx", values="raw_x_x0").to_numpy()
        pred = d.pivot(index="row_idx", columns="col_idx", values="y_pred").to_numpy()
        err = np.abs(pred - raw)
        vmin = min(raw.min(), pred.min()); vmax = max(raw.max(), pred.max())
        for j, (mat, title, cmap, lim) in enumerate([
            (raw, "Observed X/X0", "viridis", (vmin, vmax)),
            (pred, "DoseSyn B5 prediction", "viridis", (vmin, vmax)),
            (err, "Absolute error", "magma", (0, 0.25)),
        ]):
            ax = axes[i, j]
            im = ax.imshow(mat, origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1], aspect="auto")
            ax.set_xticks(range(4)); ax.set_xticklabels([f"{v:g}" for v in b_vals], fontsize=6.2, rotation=35)
            ax.set_yticks(range(4)); ax.set_yticklabels([f"{v:g}" for v in a_vals], fontsize=6.2)
            ax.set_title(title, fontsize=7.6)
            if j == 0:
                ax.set_ylabel(f"{c['case_id']}\nA: {c['drug_a_name']}\n(µM)", fontsize=6.5)
            if i == n - 1:
                ax.set_xlabel(f"B: {c['drug_b_name']} (µM)", fontsize=6.5)
            if j == 2:
                for r in range(4):
                    for cc in range(4):
                        ax.text(cc, r, f"{err[r, cc]:.2f}", ha="center", va="center", fontsize=5.2,
                                color="white" if err[r, cc] > 0.14 else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=5.5)
        axes[i, 2].text(1.02, -0.22, f"p={c['p_mean']:.3f}, RMSE={c['rmse'] if 'rmse' in c else 0:.3f}",
                        transform=axes[i, 2].transAxes, fontsize=6.0, ha="right")
    fig.suptitle("Pre-specified representative cases: raw, prediction and error", y=1.005, fontsize=10)
    fig.subplots_adjust(hspace=0.55, wspace=0.35)
    _save(fig, path / "fig4_16_case_surfaces")


def main():
    cfg = load_config(CONFIG)
    metrics = resolve_path(cfg, cfg["paths"]["outputs_metrics"])
    figs = resolve_path(cfg, cfg["paths"]["outputs_figures"])
    docs = resolve_path(cfg, cfg["paths"]["docs_figures"])
    figs.mkdir(parents=True, exist_ok=True); docs.mkdir(parents=True, exist_ok=True)

    scored = pd.read_csv(metrics / "m8c_scored_candidates.csv")
    cases = pd.read_csv(metrics / "m8c_representative_cases.csv")
    rows = pd.read_csv(metrics / "m8c_case_source_rows.csv")
    traces = pd.read_json(metrics / "m8c_summary.json").loc["traces"].to_dict() if False else None
    # attach RMSE from case_source_rows for annotations
    rmse = rows.groupby("case_id").apply(lambda d: float(np.sqrt(np.mean((d["y_pred"] - d["raw_x_x0"]) ** 2))))
    cases["rmse"] = cases["case_id"].map(rmse)

    workflow(figs); workflow(docs)
    landscape(scored, figs); landscape(scored, docs)
    funnel(figs, 300, len(scored), int(scored["dual_signal"].sum()) if "dual_signal" in scored else 27)
    funnel(docs, 300, len(scored), int(scored["dual_signal"].sum()) if "dual_signal" in scored else 27)
    cases_figure(rows, cases, figs); cases_figure(rows, cases, docs)
    print("m8c figures:", sorted(p.name for p in docs.glob("*.png")))


if __name__ == "__main__":
    main()
