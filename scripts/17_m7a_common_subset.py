"""M7-A: compare all models on the matrix subset where comboKR is available.

comboKR has no MK-8776 surfaces, so its S1/S2 test sets are a strict subset of
the frozen test matrices. Aggregate means alone could be confounded by matrix
composition; this script recomputes every model on exactly comboKR's matrices.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.training.metrics import compute_metrics  # noqa: E402

OUT = ROOT / "outputs/metrics/m7a"


def frame_metric(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["point"] = df["row_idx"].astype(int) * 4 + df["col_idx"].astype(int)
    yt = df.pivot_table(index="matrix_id", columns="point", values="y_true").sort_index()
    yp = df.pivot_table(index="matrix_id", columns="point", values="y_pred").sort_index(axis=0).loc[yt.index]
    yp = yp[yt.columns]
    mids = yt.index.to_numpy()
    return compute_metrics(
        yt.to_numpy(), yp.to_numpy(),
        matrix_ids=mids, condition_ids=mids,
    )


def ref_run(model: str, scenario: str, seed: int | None, fold: int | None) -> Path | None:
    if model.startswith("m6b"):
        prefix = "m6b__b5_t3_film_reg" if model == "m6b_b5_film" else "m6b__b1_t3_fourier"
        key = f"seed_{seed}" if seed is not None else f"fold_{fold}"
        hits = glob.glob(str(ROOT / "outputs/runs/m6b" / f"{prefix}__{scenario}__{key}" / "predictions_test.parquet"))
    elif model == "m4_b3_gine":
        key = f"seed_{seed}" if seed is not None else f"fold_{fold}"
        hits = glob.glob(str(ROOT / "outputs/runs/m4" / f"b3_gine__{scenario}__{key}" / "predictions_test.parquet"))
    elif model == "m4_b2_mlp":
        key = f"seed_{seed}" if seed is not None else f"fold_{fold}"
        hits = glob.glob(str(ROOT / "outputs/runs/m4" / f"b2_mlp__{scenario}__{key}" / "predictions_test.parquet"))
    else:
        return None
    return Path(hits[0]) if hits else None


def main() -> int:
    rows = []
    for meta_path in sorted(glob.glob(str(ROOT / "outputs/runs/m7a/combokr/*/*/run_metadata.json"))):
        meta = json.loads(Path(meta_path).read_text())
        scenario = meta["scenario"]
        if scenario not in ("s1_random_condition", "s2_unseen_pair"):
            continue
        seed, fold = meta.get("seed"), meta.get("fold")
        combo = pd.read_parquet(Path(meta_path).parent / "predictions.parquet")
        common = set(combo["matrix_id"].unique())
        for model, label in [
            ("combokr", "comboKR 2.0"),
            ("m6b_b5_film", "M6-B B5 (T3+FiLM)"),
            ("m6b_b1_t3", "M6-B B1 (T3)"),
            ("m4_b3_gine", "M4 B3 GINE"),
            ("m4_b2_mlp", "M4 B2 MLP"),
        ]:
            if model == "combokr":
                df = combo
            else:
                p = ref_run(model, scenario, seed, fold)
                if p is None:
                    continue
                df = pd.read_parquet(p)
                df = df[df["matrix_id"].isin(common)]
            m = frame_metric(df)
            rows.append(dict(model=label, scenario=scenario, seed=seed, fold=fold,
                             n_matrices=m["n_matrices"], n_points=m["n_points"],
                             matrix_macro_rmse=m["matrix_macro_rmse"],
                             matrix_macro_mae=m["matrix_macro_mae"],
                             point_rmse=m["point_rmse"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "common_subset_comparison.csv", index=False)
    summary = (df.groupby(["model", "scenario"])["matrix_macro_rmse"]
                 .agg(["count", "mean", "std"]).reset_index()
                 .rename(columns={"count": "n_runs", "mean": "macro_rmse_mean", "std": "macro_rmse_std"}))
    summary.to_csv(OUT / "common_subset_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
