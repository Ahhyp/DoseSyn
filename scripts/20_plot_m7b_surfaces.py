"""M7-B: dose-surface panels for representative matrices.

Selection rule is fixed and reported: for the requested run, rank test matrices
by held-out RMSE and pick the 10th, 50th and 90th percentiles (plus the worst
high-dose boundary case when mask=boundary_holdout_50).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/figures/m7b/surfaces"


def color(v: float, vmax: float = 2.0) -> str:
    t = max(0.0, min(1.0, v / vmax))
    # blue -> white -> red
    if t < 0.5:
        r = int(40 + 2 * t * 215); g = int(70 + 2 * t * 185); b = int(160 + 2 * t * 80)
    else:
        r = 255; g = int(255 - (t - 0.5) * 2 * 200); b = int(240 - (t - 0.5) * 2 * 180)
    return f"rgb({r},{g},{b})"


def panel(x0, y0, grid, title, vmax=None, err=False):
    s = 46
    parts = [f'<text x="{x0}" y="{y0-6}" font-size="12">{title}</text>']
    for i in range(4):
        for j in range(4):
            v = float(grid[i, j])
            if err:
                fill = f"rgb({int(min(255, 255*v/0.5))},{int(220 - min(220, 440*v))},80)"
            else:
                fill = color(v, vmax or 2.0)
            parts.append(f'<rect x="{x0+j*s}" y="{y0+i*s}" width="{s-1}" height="{s-1}" fill="{fill}" stroke="#ddd"/>')
            parts.append(f'<text x="{x0+j*s+s/2}" y="{y0+i*s+s/2+4}" font-size="9" text-anchor="middle" fill="#111">{v:.2f}</text>')
    return parts


def render(path, df, y_true, y_pred, mask, title):
    err = np.abs(y_pred - y_true)
    w, h = 920, 520
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
             f'<text x="20" y="24" font-size="15">{title}</text>']
    parts += panel(40, 55, y_true, "Ground Truth X/X0")
    parts += panel(280, 55, y_pred, "Prediction X/X0")
    parts += panel(520, 55, err, "|Error|", err=True)
    # mask panel
    s = 46
    parts += [f'<text x="760" y="49" font-size="12">Observation mask</text>']
    for i in range(4):
        for j in range(4):
            held = mask[i, j] > 0.5
            fill = "#f2f2f2" if held else "#c0392b"
            parts.append(f'<rect x="{760+j*s}" y="{55+i*s}" width="{s-1}" height="{s-1}" fill="{fill}" stroke="#ddd"/>')
            parts.append(f'<text x="{760+j*s+s/2}" y="{55+i*s+s/2+4}" font-size="9" text-anchor="middle" fill="#111">{"held" if held else "obs"}</text>')
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    parts.append(f'<text x="40" y="270" font-size="12">point RMSE = {rmse:.4f}; matrix_id = {df.matrix_id.iloc[0]}</text>')
    parts.append(f'<text x="40" y="290" font-size="11">cell = {df.cell_line.iloc[0]} | {df.drug_a_name.iloc[0]} + {df.drug_b_name.iloc[0]}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="outputs/runs/m7b/<run_id>")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    run = (ROOT / args.run) if not Path(args.run).is_absolute() else Path(args.run)
    pred_p = run / "predictions_test.parquet"
    if not pred_p.exists():
        print("missing", pred_p)
        return 1
    df = pd.read_parquet(pred_p)
    from dosesyn.train.m7b_masks import mask_matrix

    mask_name = str(df.mask_name.iloc[0])
    mask = mask_matrix(mask_name)
    held = df.is_held.astype(bool)
    rm = df[held].groupby("matrix_id").apply(lambda d: float(np.sqrt(np.mean(d.residual ** 2))))
    rm = rm.sort_values()
    picks = {f"p{int(q*100)}": rm.index[int((len(rm) - 1) * q)] for q in (0.1, 0.5, 0.9)}
    if mask_name == "boundary_holdout_50":
        picks["worst"] = rm.index[-1]
    tag = args.tag or run.name
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for label, mid in picks.items():
        d = df[df.matrix_id == mid].sort_values(["row_idx", "col_idx"])
        gt = d.y_true.to_numpy().reshape(4, 4)
        pr = d.y_pred.to_numpy().reshape(4, 4)
        p = OUT / f"{tag}_{label}_{mid}.svg"
        render(p, d, gt, pr, mask, f"{tag} — {label} (held-out quantile)")
        manifest.append({"label": label, "matrix_id": mid, "svg": str(p.relative_to(ROOT)),
                         "held_rmse": float(rm.loc[mid])})
    (OUT / f"{tag}_cases.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
