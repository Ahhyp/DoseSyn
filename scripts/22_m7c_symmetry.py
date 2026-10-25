"""M7-C5: verify exchange symmetry of the frozen DoseSyn B5 checkpoint.

f(A, B, x, y, c) vs f(B, A, y, x, c) on held-out test matrices of S1/S2/S3.
No training, no adaptation, no test-label use beyond forward evaluation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.models.m6b_models import build_m6b_model  # noqa: E402
from dosesyn.train.m6a_trainer import slice_graph_batch, vocab_graph_batch  # noqa: E402
from dosesyn.training.m4_data import load_graphs, load_matrix_table, table_batch  # noqa: E402

OUT_M = ROOT / "outputs/metrics/m7c"
OUT_A = ROOT / "docs/audit/m7c"
CFG_PATH = ROOT / "configs/m6b_experiments.yaml"

CASES = [
    ("S1", "s1_random_condition", {"seed": 42}, "m6b__b5_t3_film_reg__s1_random_condition__seed_42"),
    ("S2", "s2_unseen_pair", {"seed": 42}, "m6b__b5_t3_film_reg__s2_unseen_pair__seed_42"),
    ("S3", "s3_unseen_cell", {"fold": 0}, "m6b__b5_t3_film_reg__s3_unseen_cell__fold_0"),
]


def load_model(cfg):
    variant = cfg["variants"]["b5_t3_film_reg"]
    model = build_m6b_model(variant, cfg["model"])
    ckpt = ROOT / "outputs/runs/m6b" / "m6b__b5_t3_film_reg__s1_random_condition__seed_42" / "best.pt"
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, ckpt


def swap_batch(batch):
    out = dict(batch)
    out["dose_a"], out["dose_b"] = batch["dose_b"], batch["dose_a"]
    out["dose_a_zero"], out["dose_b_zero"] = batch["dose_b_zero"], batch["dose_a_zero"]
    return out


def swap_graphs(graphs):
    out = dict(graphs)
    out["drug_a_graph"], out["drug_b_graph"] = graphs["drug_b_graph"], graphs["drug_a_graph"]
    return out


def forward_surface(model, batch, graphs):
    with torch.no_grad():
        return model.forward_all(batch, graphs)["surface"]


def main() -> int:
    for d in (OUT_M, OUT_A):
        d.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cfg = yaml.safe_load(CFG_PATH.read_text())
    model, ckpt = load_model(cfg)
    graphs_all, _ = load_graphs(cfg["inputs"]["m4_config"])

    rows = []
    examples = []
    curves = []
    t0 = time.time()
    for label, scenario, kwargs, run_id in CASES:
        table = load_matrix_table(scenario, "test", **kwargs)
        n = len(table.matrix_ids)
        p_all = np.zeros((n, 16), dtype=np.float64)
        q_all = np.zeros((n, 16), dtype=np.float64)
        for start in range(0, n, 256):
            idx = list(range(start, min(start + 256, n)))
            b = table_batch(table, idx, device=torch.device("cpu"))
            g = slice_graph_batch(vocab_graph_batch(table, graphs_all, torch.device("cpu")), table, idx, torch.device("cpu"))
            p = forward_surface(model, b, g)
            q = forward_surface(model, swap_batch(b), swap_graphs(g))
            p_all[start : start + len(idx)] = p.cpu().numpy()
            q_all[start : start + len(idx)] = q.cpu().numpy()

        # f(A,B) should equal transpose of f(B,A)
        p = p_all.reshape(n, 4, 4)
        q = q_all.reshape(n, 4, 4).transpose(0, 2, 1)
        diff = np.abs(p - q).reshape(n, 16)
        per_matrix_max = diff.max(axis=1)
        per_matrix_mean = diff.mean(axis=1)

        for mid, cell, da, db, mx, mn in zip(
            table.matrix_ids, table.cell_lines, table.drug_a_names, table.drug_b_names, per_matrix_max, per_matrix_mean
        ):
            rows.append(
                {
                    "scenario": label,
                    "split_id": table.split_id if hasattr(table, "split_id") else str(kwargs),
                    "run_id": run_id,
                    "matrix_id": mid,
                    "cell_line": cell,
                    "drug_a_name": da,
                    "drug_b_name": db,
                    "max_abs_swap_error": float(mx),
                    "mean_abs_swap_error": float(mn),
                }
            )
        order = np.argsort(-per_matrix_max)[:5]
        for i in order:
            examples.append(
                {
                    "scenario": label,
                    "matrix_id": table.matrix_ids[i],
                    "cell_line": table.cell_lines[i],
                    "drug_a_name": table.drug_a_names[i],
                    "drug_b_name": table.drug_b_names[i],
                    "max_abs_swap_error": float(per_matrix_max[i]),
                    "mean_abs_swap_error": float(per_matrix_mean[i]),
                    **{f"pred_AB_{k:02d}": float(p[i].reshape(-1)[k]) for k in range(16)},
                    **{f"pred_BA_{k:02d}": float(q[i].reshape(-1)[k]) for k in range(16)},
                }
            )
        # a representative curve: median-error matrix, row 2 across columns
        med = int(np.argsort(per_matrix_max)[n // 2])
        for j in range(4):
            curves.append(
                {
                    "scenario": label,
                    "matrix_id": table.matrix_ids[med],
                    "dose_a_um": float(table.dose_a_um[med, 2]),
                    "dose_b_um": float(table.dose_b_um[med, j]),
                    "pred_AB": float(p[med, 2, j]),
                    "pred_BA": float(q[med, 2, j]),
                }
            )
        print(
            f"{label}: n={n} max={per_matrix_max.max():.3e} mean={per_matrix_mean.mean():.3e} "
            f"p99={np.percentile(per_matrix_max, 99):.3e}"
        )

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT_M / "symmetry_runs.csv", index=False)
    pd.DataFrame(examples).to_csv(OUT_M / "symmetry_examples.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT_M / "symmetry_curve.csv", index=False)

    summary = []
    for label, grp in runs.groupby("scenario", sort=False):
        summary.append(
            {
                "scenario": label,
                "n_matrices": len(grp),
                "max_abs_error": float(grp.max_abs_swap_error.max()),
                "mean_abs_error": float(grp.mean_abs_swap_error.mean()),
                "p99_matrix_max": float(np.percentile(grp.max_abs_swap_error, 99)),
                "median_matrix_max": float(np.median(grp.max_abs_swap_error)),
                "n_matrices_gt_1e-5": int((grp.max_abs_swap_error > 1e-5).sum()),
            }
        )
    s = pd.DataFrame(summary)
    s.to_csv(OUT_M / "table_c5_symmetry.csv", index=False)
    (OUT_A / "table_c5_symmetry.csv").write_text(s.to_csv(index=False), encoding="utf-8")
    meta = {
        "checkpoint": str(ckpt.relative_to(ROOT)),
        "checkpoint_sha256": __import__("hashlib").sha256(ckpt.read_bytes()).hexdigest(),
        "scenarios": [c[0] for c in CASES],
        "definition": "max over 16 grid points of |f(A,B,x,y,c) - f(B,A,y,x,c)| on held-out test matrices",
        "wall_seconds": round(time.time() - t0, 2),
        "torch_threads": 4,
    }
    (OUT_M / "symmetry_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(s.to_string(index=False))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
