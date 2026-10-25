"""comboKR 2.0 adapter for the frozen O'Neil M3 splits.

Runs ONLY in the isolated Python 3.11 environment (numpy 1.26 / synergy 0.5.1 /
RLScore 0.8.2a0), because the official code needs modern sklearn/numpy and the
project environment is pinned to numpy 1.21 for M0--M6.

The adapter does not retrain comboKR from raw values: training labels are the
official BRAID parameter fits distributed with comboKR 2.0 (9 parameters per
surface).  Only surfaces whose matrix belongs to the frozen train split are
used for fitting.  Test surfaces are predicted from their (cell, drug1, drug2)
identity and evaluated at the exact 4x4 dose grid of this project.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
REF = ROOT / "data/external/m7a_src/comboKR2.0-main"
DATA = Path("/tmp/combokr_data/data")


def load_reference():
    cdd = np.load(DATA / "cdd_full.npy", allow_pickle=True).astype(str)
    braids = np.load(DATA / "fitted_braid_params_full.npy")
    c1 = np.load(DATA / "conc1.npy")
    c2 = np.load(DATA / "conc2.npy")
    y = np.load(DATA / "measurementsMedian.npy")
    hills = pickle.load(open(DATA / "fitted_hills_full.df.pkl", "rb"))
    Kc = np.load(DATA / "Kc.npy")
    Kd = np.load(DATA / "Kd.npy")
    cell_ids = np.load(DATA / "cellfeat_ids.npy", allow_pickle=True).astype(str)
    drug_ids = np.load(DATA / "drugfeat_ids.npy", allow_pickle=True).astype(str)
    return dict(cdd=cdd, braids=braids, c1=c1, c2=c2, y=y, hills=hills,
                Kc=Kc, Kd=Kd, cell_ids=cell_ids, drug_ids=drug_ids)


def build_matrix_map(ref, primary):
    """Map matrix_id -> (cdd_index, swapped, grid_equal)."""
    cdd = ref["cdd"]
    index = {}
    for i, (cell, d1, d2) in enumerate(cdd):
        index.setdefault((cell, d1, d2), i)
    return index


def split_ids(scenario: str, seed: int | None, fold: int | None, split: str):
    split_id = f"fold_{fold}" if fold is not None else f"seed_{seed}"
    return ROOT / "data/splits" / scenario / split_id


def load_matrix_table(scenario, seed, fold):
    split_id = f"fold_{fold}" if fold is not None else f"seed_{seed}"
    base = ROOT / "data/splits" / scenario / split_id
    meta = json.loads((base / "split_metadata.json").read_text())
    return base, meta


def order_braids(braids, swapped):
    out = np.array(braids, copy=True)
    mask = np.asarray(swapped, dtype=bool)
    if mask.any():
        out[mask] = out[mask][:, [0, 2, 1, 3, 5, 4, 7, 6, 8]]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n-iters", type=int, default=5)
    ap.add_argument("--tol", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REF))
    from comboKR2.scalable_comboKR import (
        ScalablecomboKRdiff,
        SurfaceKernelNystromDifference,
    )

    t_start = time.time()
    ref = load_reference()
    primary_idx = pd.read_parquet(ROOT / "data/processed/oneil_primary_v1_matrix_index.parquet").set_index("matrix_id")
    base = ROOT / "data/splits" / args.scenario / (
        f"fold_{args.fold}" if args.fold is not None else f"seed_{args.seed}"
    )
    primary = pd.read_parquet(ROOT / "data/processed/oneil_primary_v1_matrix_index.parquet")
    points = pd.read_parquet(
        ROOT / "data/processed/oneil_combo_points.parquet",
        columns=["matrix_id", "row_idx", "col_idx", "conc_a_um", "conc_b_um", "x_x0_raw"],
    )
    points = points.sort_values(["matrix_id", "row_idx", "col_idx"], kind="mergesort")
    doses_a = points.pivot_table(index="matrix_id", columns="row_idx", values="conc_a_um", aggfunc="first")
    doses_b = points.pivot_table(index="matrix_id", columns="col_idx", values="conc_b_um", aggfunc="first")
    labels = points.pivot_table(index="matrix_id", columns=["row_idx", "col_idx"], values="x_x0_raw", aggfunc="first")
    labels = labels[sorted(labels.columns, key=lambda t: (t[0], t[1]))]

    cdd = ref["cdd"]
    lookup = {}
    for i, (cell, d1, d2) in enumerate(cdd):
        lookup.setdefault((cell, d1, d2), i)
        lookup.setdefault((cell, d2, d1), i)

    rows = []
    missing = []
    for r in primary.itertuples():
        key = (str(r.cell_line), str(r.drug_a_name).lower(), str(r.drug_b_name).lower())
        i = lookup.get(key)
        if i is None:
            missing.append(r.matrix_id)
            continue
        # orientation: cdd stores (drug1, drug2); our columns are (a, b)
        swapped = str(r.drug_a_name).lower() != cdd[i, 1]
        # grid comparison (cdd conc vectors are row-major 4x4 with A slow, B fast)
        grid_a = np.asarray(doses_a.loc[r.matrix_id].values, dtype=float) / 1e6
        grid_b = np.asarray(doses_b.loc[r.matrix_id].values, dtype=float) / 1e6
        cdd_a = c1 = ref["c1"][i].reshape(4, 4)[:, 0]
        cdd_b = ref["c2"][i].reshape(4, 4)[0, :]
        if swapped:
            cdd_a, cdd_b = cdd_b, cdd_a
        grid_equal = np.allclose(grid_a, cdd_a, rtol=1e-6, atol=1e-12) and np.allclose(
            grid_b, cdd_b, rtol=1e-6, atol=1e-12
        )
        rows.append(dict(matrix_id=r.matrix_id, cdd_index=i, swapped=bool(swapped),
                         grid_equal=bool(grid_equal)))
    m = pd.DataFrame(rows)

    train_ids = set(pd.read_parquet(base / "train_conditions.parquet").condition_id)
    eval_ids = set(pd.read_parquet(base / f"{args.split}_conditions.parquet").condition_id)
    pm = primary.set_index("matrix_id")
    m["split"] = np.where(m.matrix_id.map(pm.condition_id).isin(train_ids), "train",
                  np.where(m.matrix_id.map(pm.condition_id).isin(eval_ids), args.split, "other"))
    train = m[(m.split == "train") & m.grid_equal].copy()
    test = m[(m.split == args.split) & m.grid_equal].copy()
    print(f"mapped={len(m)} train={len(train)} {args.split}={len(test)} "
          f"grid_mismatch={int((~m.grid_equal).sum())} missing={len(missing)}", flush=True)

    cdd_tr = cdd[train.cdd_index.values]
    braids_tr = order_braids(ref["braids"][train.cdd_index.values], train.swapped.values)
    # double the training set for pair-order symmetry, exactly as in demo.py
    cdd_tr2 = np.stack([cdd_tr[:, 0], cdd_tr[:, 2], cdd_tr[:, 1]], axis=1)
    braids_tr2 = braids_tr[:, [0, 2, 1, 3, 5, 4, 7, 6, 8]]
    cdd_tr = np.vstack([cdd_tr, cdd_tr2])
    braids_tr = np.vstack([braids_tr, braids_tr2])

    out_nystrom = SurfaceKernelNystromDifference(tol=args.tol, kernel="rbf", max_is_100=False)
    out_nystrom.create_approximation()
    model = ScalablecomboKRdiff(
        ref["Kc"], ref["Kd"], ref["cell_ids"], ref["drug_ids"], out_nystrom,
        hills=ref["hills"], max_is_100=False,
    )
    model.train(cdd_tr, braids_tr, args.n_iters)

    test_triples = cdd[test.cdd_index.values]
    ca4 = np.vstack([np.asarray(doses_a.loc[mid].values, dtype=float) / 1e6 for mid in test.matrix_id])
    cb4 = np.vstack([np.asarray(doses_b.loc[mid].values, dtype=float) / 1e6 for mid in test.matrix_id])
    ca = np.repeat(ca4, 4, axis=1)          # A slow: each row dose repeated 4x
    cb = np.tile(cb4, (1, 4))               # B fast: 4-dose row tiled 4x
    if test.swapped.any():
        sw = test.swapped.values
        ca[sw], cb[sw] = cb[sw].copy(), ca[sw].copy()

    pred_vals, params = model.predict_with_candidates(test_triples, ca, cb)
    pred_vals = np.asarray(pred_vals, dtype=float)
    print("predicted", pred_vals.shape, flush=True)

    records = []
    t = test.reset_index(drop=True)
    for k, row in t.iterrows():
        mid = row.matrix_id
        lab = labels.loc[mid].values
        pm = primary_idx.loc[mid]
        for p in range(16):
            rec = dict(
                matrix_id=mid,
                condition_id=pm["condition_id"],
                pair_id=pm["pair_id"],
                cell_id=pm["cell_id"],
                drug_a_id=pm["drug_a_id"],
                drug_b_id=pm["drug_b_id"],
                row_idx=p // 4,
                col_idx=p % 4,
                dose_a_um=float(ca[k, p]),
                dose_b_um=float(cb[k, p]),
                y_true=float(lab[p]),
                y_pred=float(pred_vals[k, p]),
            )
            records.append(rec)
    df = pd.DataFrame(records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    sys.path.insert(0, str(ROOT / "src"))
    from dosesyn.training.metrics import compute_metrics
    y_eval = labels.loc[list(test.matrix_id)].values
    cond_ids = [primary_idx.loc[m, "condition_id"] for m in test.matrix_id]
    metrics = compute_metrics(y_eval, pred_vals, matrix_ids=list(test.matrix_id), condition_ids=cond_ids)
    meta = dict(
        model="combokr2.0",
        metrics=metrics,
        scenario=args.scenario, seed=args.seed, fold=args.fold, split=args.split,
        n_iters=args.n_iters, tolerance=args.tol,
        n_train_surfaces=len(train), n_train_rows=int(len(cdd_tr)),
        n_eval_matrices=len(test), n_eval_points=len(df),
        n_mapped=len(m), n_grid_mismatch=int((~m.grid_equal).sum()),
        n_missing_mapping=len(missing),
        train_matrix_ids=[str(x) for x in train.matrix_id],
        eval_matrix_ids=[str(x) for x in test.matrix_id],
        runtime_seconds=time.time() - t_start,
        reference_data=str(DATA), reference_code=str(REF),
        notes=("Official BRAID parameter labels and RBF Nystrom output kernel; "
               "grid-equal subsets only; predictions unclipped."),
    )
    (out.parent / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("n_train_surfaces", "n_eval_matrices",
                                          "n_eval_points", "runtime_seconds")}), flush=True)


if __name__ == "__main__":
    main()
