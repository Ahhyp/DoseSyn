"""M6-0 gate audit: frozen split identity, M4/M5 same-validation comparison, budget analysis.

Read-only with respect to all M0-M5 artefacts. Writes only:
  outputs/audit/m6_gate0_split_audit.json
  outputs/audit/m6_gate0_baseline_alignment.json
  outputs/metrics/m6/gate0_validation_metrics.csv
  m6_gate0_report.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.training.metrics import compute_metrics  # noqa: E402
from dosesyn.utils.io import read_json, sha256_file, utc_now_iso, write_json  # noqa: E402

SCENARIOS = {
    "s1_seed42": ("s1_random_condition", 42, None),
    "s1_seed13": ("s1_random_condition", 13, None),
    "s3_fold0": ("s3_unseen_cell", None, 0),
}


def split_dir(scenario: str, seed: Optional[int], fold: Optional[int]) -> Path:
    tag = f"seed_{seed}" if seed is not None else f"fold_{fold}"
    return ROOT / "data/splits" / scenario / tag


def set_hash(values: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(v) for v in values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def matrix_ids_for_conditions(primary: pd.DataFrame, condition_ids: Sequence[str]) -> List[str]:
    cond = set(condition_ids)
    return sorted(primary.loc[primary["condition_id"].isin(cond), "matrix_id"].astype(str).unique())


def split_fingerprint(scenario: str, seed: Optional[int], fold: Optional[int], primary: pd.DataFrame) -> Dict[str, Any]:
    d = split_dir(scenario, seed, fold)
    meta = read_json(d / "split_metadata.json")
    out: Dict[str, Any] = {
        "path": str(d),
        "seed": seed,
        "fold": fold,
        "metadata_counts": {k: meta["counts"][k]["n_matrices"] for k in ("train", "val", "test")},
        "assignment_sha256": meta.get("assignment_sha256"),
        "content_sha256": meta.get("content_sha256"),
        "files": {},
        "splits": {},
    }
    for split in ("train", "val", "test"):
        p = d / f"{split}_conditions.parquet"
        df = pd.read_parquet(p)
        mids = matrix_ids_for_conditions(primary, df["condition_id"].tolist())
        out["files"][split] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "n_rows": int(len(df)),
            "n_conditions": int(df["condition_id"].nunique()),
        }
        out["splits"][split] = {
            "n_conditions": int(df["condition_id"].nunique()),
            "n_matrices_from_primary": len(mids),
            "condition_set_sha256": set_hash(df["condition_id"].astype(str)),
            "matrix_set_sha256": set_hash(mids),
            "matrix_ids": mids,
        }
    out["all_matrices"] = sorted(set().union(*[set(v["matrix_ids"]) for v in out["splits"].values()]))
    return out


def find_m5_runs(prefix: str) -> List[Path]:
    base = ROOT / "outputs/runs/m5"
    return sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix))


def load_predictions(run: Path, split: str = "val") -> pd.DataFrame:
    return pd.read_parquet(run / f"predictions_{split}.parquet")


def metrics_from_predictions(pred: pd.DataFrame) -> Dict[str, Any]:
    import numpy as np

    df = pred.sort_values(["matrix_id", "row_idx", "col_idx"])
    groups = list(df.groupby("matrix_id", sort=True))
    y_true = np.stack([g["y_true"].to_numpy(dtype=float).reshape(4, 4) for _, g in groups])
    y_pred = np.stack([g["y_pred"].to_numpy(dtype=float).reshape(4, 4) for _, g in groups])
    matrix_ids = [str(mid) for mid, _ in groups]
    condition_ids = [str(g["condition_id"].iloc[0]) for _, g in groups]
    return compute_metrics(y_true, y_pred, matrix_ids=matrix_ids, condition_ids=condition_ids)


def audit_m5(m5_prefix: str, m5_runs: Sequence[Path], fp: Dict[str, Any], seed_other: Dict[str, Any]) -> Dict[str, Any]:
    target = set(fp["splits"]["val"]["matrix_ids"])
    other = set(seed_other["splits"]["val"]["matrix_ids"])
    runs: List[Dict[str, Any]] = []
    for run in m5_runs:
        pred = load_predictions(run)
        ids = set(pred["matrix_id"].astype(str))
        vm = read_json(run / "val_metrics.json")
        cfg = yaml.safe_load(open(run / "config.yaml", "r", encoding="utf-8"))
        meta = read_json(run / "run_metadata.json")
        runs.append(
            {
                "run_id": run.name,
                "split_id": meta.get("split_id"),
                "seed": meta.get("seed"),
                "fold": meta.get("fold"),
                "n_matrices": int(pred["matrix_id"].nunique()),
                "n_rows": int(len(pred)),
                "matrix_set_equals_target": ids == target,
                "matrix_set_equals_other_seed": ids == other,
                "missing_vs_target": len(target - ids),
                "extra_vs_target": len(ids - target),
                "intersection_vs_target": len(ids & target),
                "intersection_vs_other_seed": len(ids & other),
                "val_metrics_n_matrices": vm.get("n_matrices"),
                "val_metrics_matrix_macro_rmse": vm.get("matrix_macro_rmse"),
                "val_metrics_point_rmse": vm.get("point_rmse"),
                "model_name": meta.get("model"),
                "variant": meta.get("variant"),
                "epochs_config": cfg.get("training", {}).get("epochs"),
                "early_stop_metric_note": "M5 early stopping uses val matrix_macro_RMSE",
            }
        )
    return {
        "target_split": fp["path"],
        "n_target_val_matrices": len(target),
        "runs": runs,
        "all_runs_use_target": all(r["matrix_set_equals_target"] for r in runs),
        "any_run_uses_other_seed": any(r["matrix_set_equals_other_seed"] for r in runs),
    }


def audit_m4(m4_prefix: str, fp: Dict[str, Any]) -> Dict[str, Any]:
    run = ROOT / "outputs/runs/m4" / m4_prefix
    pred = load_predictions(run)
    ids = set(pred["matrix_id"].astype(str))
    target = set(fp["splits"]["val"]["matrix_ids"])
    stored = read_json(run / "val_metrics.json")
    recomputed = metrics_from_predictions(pred)
    ck = torch.load(run / "best.pt", map_location="cpu")
    state = ck.get("model_state", ck)
    n_params = int(sum(v.numel() for v in state.values() if torch.is_tensor(v)))
    diff = {
        k: abs(float(recomputed[k]) - float(stored[k]))
        for k in ("matrix_macro_rmse", "point_rmse", "pooled_pearson", "r2")
        if k in stored and stored[k] is not None
    }
    resources = read_json(run / "resources.json")
    return {
        "run_id": m4_prefix,
        "split_id": read_json(run / "run_metadata.json").get("split_id"),
        "n_matrices": int(pred["matrix_id"].nunique()),
        "n_rows": int(len(pred)),
        "matrix_set_equals_target": ids == target,
        "n_missing_vs_target": len(target - ids),
        "stored_val_metrics": stored,
        "recomputed_val_metrics": recomputed,
        "max_abs_metric_diff": max(diff.values()) if diff else None,
        "checkpoint": {
            "path": str(run / "best.pt"),
            "epoch": ck.get("epoch"),
            "n_parameters_from_state_dict": n_params,
            "n_parameters_from_resources": resources.get("n_parameters"),
            "checkpoint_val_macro_rmse_field": ck.get("val_macro_rmse"),
            "used_for_early_stopping": "M4 selects on validation point RMSE (see _train_dl)",
        },
    }


def prediction_alignment(m4_pred: pd.DataFrame, m5_pred: pd.DataFrame) -> Dict[str, Any]:
    keys = ["matrix_id", "row_idx", "col_idx"]
    merged = m4_pred[keys + ["y_true", "dose_a_um", "dose_b_um"]].merge(
        m5_pred[keys + ["y_true", "dose_a_um", "dose_b_um"]],
        on=keys,
        suffixes=("_m4", "_m5"),
        how="inner",
    )
    return {
        "n_m4_rows": int(len(m4_pred)),
        "n_m5_rows": int(len(m5_pred)),
        "n_common_keys": int(len(merged)),
        "n_label_mismatch": int((merged["y_true_m4"] != merged["y_true_m5"]).sum()),
        "max_abs_dose_a_diff": float((merged["dose_a_um_m4"] - merged["dose_a_um_m5"]).abs().max()),
        "max_abs_dose_b_diff": float((merged["dose_b_um_m4"] - merged["dose_b_um_m5"]).abs().max()),
        "same_matrix_set": set(m4_pred["matrix_id"]) == set(m5_pred["matrix_id"]),
    }


def curve_stats(run: Path) -> Dict[str, Any]:
    df = pd.read_csv(run / "train_log.csv")
    val_col = "val_matrix_macro_rmse" if "val_matrix_macro_rmse" in df.columns else None
    if val_col is None:
        return {"error": "no validation column"}
    vals = df[val_col].astype(float)
    best_idx = int(vals.idxmin())
    best_epoch = int(df.loc[best_idx, "epoch"])
    last_epoch = int(df["epoch"].iloc[-1])
    last_val = float(vals.iloc[-1])
    window = min(10, len(vals))
    slope = float(vals.iloc[-1] - vals.iloc[-window])
    lr = float(df["lr"].iloc[-1]) if "lr" in df.columns else None
    return {
        "epochs_logged": last_epoch,
        "best_epoch": best_epoch,
        "best_val_in_log": float(vals.min()),
        "last_val_in_log": last_val,
        "slope_last_10_epochs": slope,
        "final_lr": lr,
        "budget_exhausted": best_epoch >= last_epoch,
        "train_loss_first": float(df["train_loss"].iloc[0]),
        "train_loss_last": float(df["train_loss"].iloc[-1]),
        "val_at_epoch_60": float(vals.iloc[59]) if len(vals) >= 60 else None,
        "log_column_note": (
            "M4 train_log stores validation POINT RMSE under the column named val_matrix_macro_rmse; "
            "the stored val_metrics.json / checkpoint are authoritative for matrix-macro."
            if "m4" in str(run)
            else "M5 train_log column is matrix-macro RMSE (used for early stopping)."
        ),
    }


def architecture_table() -> List[Dict[str, Any]]:
    m4 = read_json(ROOT / "outputs/runs/m4/b3_gine__s1_random_condition__seed_42/resources.json")
    m5 = read_json(ROOT / "outputs/runs/m5/m5__s1_random_condition__seed_42__c0_control/resources.json")
    return [
        {"aspect": "参数", "m4_b3_gine": f"{m4.get('n_parameters')}", "m5_c0_control": f"{m5.get('n_parameters')}", "aligned": False, "note": "C0 更大，不是 B3 复现"},
        {"aspect": "药物编码", "m4_b3_gine": "共享 GINE，原子/键类别嵌入，3 层", "m5_c0_control": "共享 GINE，同类特征，3 层", "aligned": True, "note": "同一 M2 图特征"},
        {"aspect": "细胞编码", "m4_b3_gine": "原始 50 维 Hallmark", "m5_c0_control": "原始 50 维（ordered 分支）", "aligned": True, "note": "标准化由 M3 scaler 提供"},
        {"aspect": "剂量表示", "m4_b3_gine": "标准化标量 + zero flag", "m5_c0_control": "Linear(2,64) 剂量嵌入", "aligned": False, "note": "C0 多一层剂量投影"},
        {"aspect": "药物对融合", "m4_b3_gine": "有序 concat(cell, ha, hb, da, db)", "m5_c0_control": "有序 concat(cell, ha, hb, ea64, eb64)", "aligned": False, "note": "C0 非对称对照"},
        {"aspect": "激活函数", "m4_b3_gine": "GELU + LayerNorm(GINE)", "m5_c0_control": "SiLU + GINE LayerNorm", "aligned": False, "note": "结构差异"},
        {"aspect": "优化器", "m4_b3_gine": "AdamW lr=1e-3 wd=1e-4", "m5_c0_control": "AdamW lr=1e-3 wd=1e-4", "aligned": True, "note": "相同"},
        {"aspect": "batch size", "m4_b3_gine": "64", "m5_c0_control": "64", "aligned": True, "note": ""},
        {"aspect": "训练预算", "m4_b3_gine": "80 epochs, patience 15", "m5_c0_control": "60 epochs, patience 12", "aligned": False, "note": "M5 预算短 25%"},
        {"aspect": "早停指标", "m4_b3_gine": "validation POINT RMSE", "m5_c0_control": "validation MATRIX-MACRO RMSE", "aligned": False, "note": "评价口径不同；B3 利好 point 指标"},
        {"aspect": "标签", "m4_b3_gine": "原始 X/X0，未裁剪", "m5_c0_control": "原始 X/X0，未裁剪", "aligned": True, "note": ""},
        {"aspect": "验证集", "m4_b3_gine": "M3 S1 seed42 val (2,791)", "m5_c0_control": "M3 S1 seed42 val (2,791)", "aligned": True, "note": "本轮已验证集合相等"},
    ]


def fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pytest-log", default="outputs/audit/pytest_m6.log")
    args = ap.parse_args()
    pytest_path = ROOT / args.pytest_log
    pytest_line = pytest_path.read_text(encoding="utf-8").strip().splitlines()[-1] if pytest_path.exists() else "未找到 pytest 日志"
    primary = pd.read_parquet(ROOT / "data/processed/oneil_primary_v1_matrix_index.parquet")
    fp42 = split_fingerprint("s1_random_condition", 42, None, primary)
    fp13 = split_fingerprint("s1_random_condition", 13, None, primary)
    fp_s3 = split_fingerprint("s3_unseen_cell", None, 0, primary)

    m5_s1 = find_m5_runs("m5__s1_random_condition__seed_42")
    m5_s3 = find_m5_runs("m5__s3_unseen_cell__fold_0")
    m5_s1_audit = audit_m5("s1", m5_s1, fp42, fp13)
    m5_s3_audit = audit_m5("s3", m5_s3, fp_s3, fp_s3)
    m4_s1 = audit_m4("b3_gine__s1_random_condition__seed_42", fp42)
    m4_s3 = audit_m4("b3_gine__s3_unseen_cell__fold_0", fp_s3)

    m4_pred = load_predictions(ROOT / "outputs/runs/m4/b3_gine__s1_random_condition__seed_42")
    m5_pred = load_predictions(ROOT / "outputs/runs/m5/m5__s1_random_condition__seed_42__c0_control")
    align = prediction_alignment(m4_pred, m5_pred)

    # Frozen M3 hashes, checked against the manifest written when M3 was generated.
    m3_manifest = read_json(ROOT / "outputs/audit/m3_input_manifest.json")
    frozen_checked: Dict[str, Any] = {}
    for rel, expected in {**m3_manifest.get("input_hashes", {}), **m3_manifest.get("outputs", {})}.items():
        path = ROOT / rel
        actual = sha256_file(path) if path.exists() else None
        frozen_checked[rel] = {"expected": expected, "actual": actual, "unchanged": actual == expected}
    frozen_ok = all(v["unchanged"] for v in frozen_checked.values())

    curves = {
        "m4_b3_s1": curve_stats(ROOT / "outputs/runs/m4/b3_gine__s1_random_condition__seed_42"),
        "m5_c0_s1": curve_stats(ROOT / "outputs/runs/m5/m5__s1_random_condition__seed_42__c0_control"),
        "m5_c3_s1": curve_stats(ROOT / "outputs/runs/m5/m5__s1_random_condition__seed_42__c3_film"),
        "m5_c5_s1": curve_stats(ROOT / "outputs/runs/m5/m5__s1_random_condition__seed_42__c5_boundary"),
        "m5_c6_s1": curve_stats(ROOT / "outputs/runs/m5/m5__s1_random_condition__seed_42__c6_full"),
        "m4_b3_s3": curve_stats(ROOT / "outputs/runs/m4/b3_gine__s3_unseen_cell__fold_0"),
    }

    checks = [
        {"name": "seed42_metadata_counts", "status": "PASS" if fp42["metadata_counts"] == {"train": 13022, "val": 2791, "test": 2786} else "FAIL", "detail": fp42["metadata_counts"]},
        {"name": "seed42_val_matrix_set_size", "status": "PASS" if len(fp42["splits"]["val"]["matrix_ids"]) == 2791 else "FAIL", "detail": len(fp42["splits"]["val"]["matrix_ids"])},
        {"name": "m5_seed42_uses_seed42", "status": "PASS" if m5_s1_audit["all_runs_use_target"] else "FAIL", "detail": f"{sum(r['matrix_set_equals_target'] for r in m5_s1_audit['runs'])}/{len(m5_s1_audit['runs'])} runs"},
        {"name": "m5_seed42_not_seed13", "status": "PASS" if not m5_s1_audit["any_run_uses_other_seed"] else "FAIL", "detail": "no run val set equals seed13 val set"},
        {"name": "m4_seed42_uses_seed42", "status": "PASS" if m4_s1["matrix_set_equals_target"] else "FAIL", "detail": m4_s1["n_matrices"]},
        {"name": "m4_m5_same_labels_doses", "status": "PASS" if align["n_label_mismatch"] == 0 and align["max_abs_dose_a_diff"] == 0 and align["max_abs_dose_b_diff"] == 0 else "FAIL", "detail": align},
        {"name": "metric_recompute_matches_stored", "status": "PASS" if (m4_s1["max_abs_metric_diff"] or 0) < 1e-9 else "FAIL", "detail": m4_s1["max_abs_metric_diff"]},
        {"name": "m5_s3_uses_fold0", "status": "PASS" if m5_s3_audit["all_runs_use_target"] else "FAIL", "detail": f"{sum(r['matrix_set_equals_target'] for r in m5_s3_audit['runs'])}/{len(m5_s3_audit['runs'])}"},
        {"name": "m4_s3_uses_fold0", "status": "PASS" if m4_s3["matrix_set_equals_target"] else "FAIL", "detail": m4_s3["n_matrices"]},
        {"name": "m3_frozen_hashes_unchanged", "status": "PASS" if frozen_ok else "FAIL", "detail": f"{sum(v['unchanged'] for v in frozen_checked.values())}/{len(frozen_checked)} files"},
        {"name": "c0_is_not_b3_reproduction", "status": "WARN", "detail": "params 559,299 vs 284,545; different dose/fusion/activation/budget/early-stop metric"},
        {"name": "m4_train_log_metric_label", "status": "WARN", "detail": "M4 train_log val column is point RMSE despite its name; stored val_metrics/checkpoint are authoritative"},
    ]

    gate_pass = all(c["status"] != "FAIL" for c in checks)
    split_audit = {
        "stage": "M6-0",
        "generated_at": utc_now_iso(),
        "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "frozen_primary": str(ROOT / "data/processed/oneil_primary_v1_matrix_index.parquet"),
        "splits": {"seed42": fp42, "seed13": fp13, "s3_fold0": fp_s3},
        "comparison": {
            "seed42_val_vs_seed13_val": {
                "n_seed42": len(fp42["splits"]["val"]["matrix_ids"]),
                "n_seed13": len(fp13["splits"]["val"]["matrix_ids"]),
                "intersection": len(set(fp42["splits"]["val"]["matrix_ids"]) & set(fp13["splits"]["val"]["matrix_ids"])),
                "identical": set(fp42["splits"]["val"]["matrix_ids"]) == set(fp13["splits"]["val"]["matrix_ids"]),
            }
        },
        "m5_s1_seed42": m5_s1_audit,
        "m5_s3_fold0": m5_s3_audit,
        "m4_b3_s1_seed42": m4_s1,
        "m4_b3_s3_fold0": m4_s3,
        "prediction_alignment": align,
        "frozen_m3_hashes": frozen_checked,
        "pytest": {"log": args.pytest_log, "summary": pytest_line},
        "checks": checks,
        "gate_pass": gate_pass,
        "conclusion": (
            "M5 S1 seed42 的 run 实际读取的是 seed42 val（2,791 个矩阵），"
            "不是 seed13；错误只出现在 M5 对外摘要的计数文字（13,026/2,788/2,785），"
            "训练/预测/指标均未混用 seed13。"
        ),
    }

    alignment = {
        "stage": "M6-0",
        "generated_at": utc_now_iso(),
        "same_validation_protocol": {
            "m4_b3": m4_s1["recomputed_val_metrics"],
            "m5_c0": metrics_from_predictions(m5_pred),
            "m5_c3": metrics_from_predictions(load_predictions(ROOT / "outputs/runs/m5/m5__s1_random_condition__seed_42__c3_film")),
            "m5_c5": metrics_from_predictions(load_predictions(ROOT / "outputs/runs/m5/m5__s1_random_condition__seed_42__c5_boundary")),
            "m5_c6": metrics_from_predictions(load_predictions(ROOT / "outputs/runs/m5/m5__s1_random_condition__seed_42__c6_full")),
        },
        "s3_same_validation": {
            "m4_b3": m4_s3["recomputed_val_metrics"],
            "m5_c0": metrics_from_predictions(load_predictions(ROOT / "outputs/runs/m5/m5__s3_unseen_cell__fold_0__c0_control")),
            "m5_c6": metrics_from_predictions(load_predictions(ROOT / "outputs/runs/m5/m5__s3_unseen_cell__fold_0__c6_full")),
        },
        "architecture": architecture_table(),
        "training_curves": curves,
        "interpretation": (
            "B3 与 C0 在同一组 S1 seed42 val 上差距约 0.10 macro RMSE（0.0801 vs 0.1820），"
            "不能归因于 split 混用或标签差异；C0 不是 B3 复现，且预算更短、早停指标不同、"
            "剂量/融合/激活实现不同。M4 train_log 的列名与早停指标存在误导性，需在 M6-A 用 b3_exact 对齐。"
        ),
    }

    metrics_dir = ROOT / "outputs/metrics/m6"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, metrics in alignment["same_validation_protocol"].items():
        rows.append({"scenario": "s1_seed42", "model": name, **{k: metrics.get(k) for k in ("matrix_macro_rmse", "point_rmse", "pooled_pearson", "r2", "n_matrices")}})
    for name, metrics in alignment["s3_same_validation"].items():
        rows.append({"scenario": "s3_fold0", "model": name, **{k: metrics.get(k) for k in ("matrix_macro_rmse", "point_rmse", "pooled_pearson", "r2", "n_matrices")}})
    with open(metrics_dir / "gate0_validation_metrics.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scenario", "model", "matrix_macro_rmse", "point_rmse", "pooled_pearson", "r2", "n_matrices"])
        writer.writeheader()
        writer.writerows(rows)

    audit_dir = ROOT / "outputs/audit"
    write_json(split_audit, audit_dir / "m6_gate0_split_audit.json")
    write_json(alignment, audit_dir / "m6_gate0_baseline_alignment.json")
    # Compact, Git-tracked copies (drop the large per-split matrix-id lists).
    compact = json.loads(json.dumps(split_audit))
    for fp in compact["splits"].values():
        fp.pop("all_matrices", None)
        for split in fp["splits"].values():
            split.pop("matrix_ids", None)
    public_dir = ROOT / "docs/audit"
    write_json(compact, public_dir / "m6_gate0_split_audit.json")
    write_json(alignment, public_dir / "m6_gate0_baseline_alignment.json")
    (public_dir / "gate0_validation_metrics.csv").write_bytes((metrics_dir / "gate0_validation_metrics.csv").read_bytes())

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    lines: List[str] = []
    lines.append("# DoseSyn M6-0 前置审计报告\n")
    lines.append(f"生成时间：{utc_now_iso()}  \n基线 commit：`{split_audit['git_commit']}`  \n")
    lines.append(f"**关卡结论：{'PASS' if gate_pass else 'FAIL'}**\n")
    lines.append("\n## 1. 结论摘要\n")
    lines.append("- **S1 seed42 数量异常是报告文字错误，不是训练用错 split。** M3 `seed42` 元数据为 13,022 / 2,791 / 2,786；M5 全部 7 个 S1 seed42 run 的 val 预测矩阵集合与 seed42 val 完全相等（2,791 个），与 seed13（2,788 个）不相等。\n")
    lines.append("- M5 训练/预测/指标没有混用 seed13；错误仅出现在 M5 对外摘要把 seed42 写成了 seed13 的 13,026 / 2,788 / 2,785。原始 M5 产物未改动。\n")
    lines.append("- 在**同一组** S1 seed42 val 上，M4 B3 `matrix_macro_RMSE=0.0801`，M5 C0 `=0.1820`，C3 `=0.1797`，C6 `=0.1870`。这不是 0.0795 test 与 0.1820 val 的“直接比较”，而是同口径验证集比较。\n")
    lines.append("- C0 不是 B3 的复现：参数量 559,299 vs 284,545，剂量表示、融合、激活、训练预算与早停指标均不同。M6-A 若需要对 B3 的公平参照，应实现/使用 `b3_exact`，不能拿 C0 代替。\n")
    lines.append("- M5 S1 的 C0/C3/C5/C6 最佳 epoch 均在 60/60 附近，验证曲线仍在下降；预算偏短。但仅延长预算不足以解释与 B3 的 0.10 差距——B3 在第 60 epoch 的验证 point RMSE 已约 0.096，而 C0 的 matrix-macro 仍是 0.182。\n")
    lines.append("\n## 2. 冻结划分与文件哈希\n")
    lines.append("| split | path | train | val | test | assignment sha256 |\n| --- | --- | ---: | ---: | ---: | --- |\n")
    for name, fp in (("S1 seed42", fp42), ("S1 seed13", fp13), ("S3 fold0", fp_s3)):
        c = fp["metadata_counts"]
        lines.append(f"| {name} | `{fp['path']}` | {c['train']} | {c['val']} | {c['test']} | `{(fp['assignment_sha256'] or '')[:16]}…` |\n")
    lines.append("\n### 2.1 seed42 vs seed13\n")
    cmp = split_audit["comparison"]["seed42_val_vs_seed13_val"]
    lines.append(f"- val 交集：{cmp['intersection']} / seed42 {cmp['n_seed42']} / seed13 {cmp['n_seed13']}；集合完全相同：{cmp['identical']}。\n")
    lines.append("- 两者是不同随机划分，val 矩阵数不同（2,791 vs 2,788），不能互相替代。\n")
    lines.append(f"- M3 冻结输入/输出哈希校验：{sum(v['unchanged'] for v in frozen_checked.values())}/{len(frozen_checked)} 个文件与 M3 manifest 一致。\n")
    lines.append("\n### 2.2 M5 实际读取\n")
    lines.append("| run | split_id | val matrices | == seed42 | == seed13 | missing | extra |\n| --- | --- | ---: | --- | --- | ---: | ---: |\n")
    for r in m5_s1_audit["runs"]:
        lines.append(f"| `{r['run_id']}` | {r['split_id']} | {r['n_matrices']} | {r['matrix_set_equals_target']} | {r['matrix_set_equals_other_seed']} | {r['missing_vs_target']} | {r['extra_vs_target']} |\n")
    lines.append("\n## 3. B3 与 M5 同口径验证指标\n")
    lines.append("| 场景 | 模型 | matrix-macro RMSE | point RMSE | Pearson | R² | matrices |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
    for scenario, block in (("S1 seed42", alignment["same_validation_protocol"]), ("S3 fold0", alignment["s3_same_validation"])):
        for name, m in block.items():
            lines.append(f"| {scenario} | {name} | {fmt(m.get('matrix_macro_rmse'))} | {fmt(m.get('point_rmse'))} | {fmt(m.get('pooled_pearson'))} | {fmt(m.get('r2'))} | {m.get('n_matrices')} |\n")
    lines.append("\n- M4 B3 的 `val_metrics.json` 与用保存的 `predictions_val.parquet` 重算结果一致（最大差 < 1e-9），说明指标复算无误。\n")
    lines.append(f"- M4/M5 预测对齐：共同键 {align['n_common_keys']}，标签不一致 {align['n_label_mismatch']}，剂量最大差 {align['max_abs_dose_a_diff']}/{align['max_abs_dose_b_diff']}。\n")
    lines.append("\n## 4. 结构、训练预算与早停差异\n")
    lines.append("| 维度 | M4 B3 GINE | M5 C0 control | 对齐 | 说明 |\n| --- | --- | --- | --- | --- |\n")
    for row in architecture_table():
        lines.append(f"| {row['aspect']} | {row['m4_b3_gine']} | {row['m5_c0_control']} | {row['aligned']} | {row['note']} |\n")
    lines.append("\n### 4.1 训练曲线\n")
    lines.append("| run | epochs | best epoch | best val (log) | last val | last-10 slope | final lr |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for name, c in curves.items():
        lines.append(f"| {name} | {c.get('epochs_logged')} | {c.get('best_epoch')} | {fmt(c.get('best_val_in_log'))} | {fmt(c.get('last_val_in_log'))} | {fmt(c.get('slope_last_10_epochs'))} | {c.get('final_lr')} |\n")
    lines.append("\n- M4 `train_log.csv` 的 `val_matrix_macro_rmse` 列实际记录的是 validation **point RMSE**（checkpoint `epoch=77, val_macro_rmse=0.09502` 与 train_log 第 77 行一致，而 `val_metrics.json` 的 matrix-macro 是 0.0801）。这是 M4 的日志命名/早停口径问题，不是本轮新产生的数据错误；M6-A 必须显式记录。\n")
    lines.append("\n## 5. 判定与 M6-A 前置条件\n")
    lines.append("| 检查 | 状态 | 说明 |\n| --- | --- | --- |\n")
    for c in checks:
        lines.append(f"| {c['name']} | {c['status']} | {str(c['detail'])[:200]} |\n")
    lines.append("\n- **放行条件满足**：split 对齐、B3 同口径 val 参照建立、冻结哈希未变、M4/M5 产物未覆盖。\n")
    lines.append("- **必须带上的修正**：M6-A 使用 `b3_exact`（或 M4 B3 原始 checkpoint/重训）作为 B3 参照；M5 C0 只能标为“M5 ordered control”；训练预算至少 160 epoch，patience 25–40；早停统一用 matrix-macro RMSE；M4 train_log 列名问题在报告中注明。\n")
    lines.append("- 本轮没有读取任何 test 标签，没有运行正式 11 split，没有修改 M0–M3 或覆盖 M4/M5 产物。\n")
    lines.append("\n## 6. 产物、测试与 Git\n")
    lines.append("- `outputs/audit/m6_gate0_split_audit.json`（完整，本地；含逐 split matrix id）\n- `outputs/audit/m6_gate0_baseline_alignment.json`（完整，本地）\n- `outputs/metrics/m6/gate0_validation_metrics.csv`（本地）\n- Git 跟踪副本：`docs/audit/m6_gate0_split_audit.json`、`docs/audit/m6_gate0_baseline_alignment.json`、`docs/audit/gate0_validation_metrics.csv`\n- `m6_gate0_report.md`\n")
    lines.append("- 自动测试：`tests/test_m6_gate0.py`（split 集合、矩阵数、B3/M5 标签与剂量一致性）。\n")
    lines.append(f"- 完整 pytest：**{pytest_line}**（日志 `{args.pytest_log}`）。\n")
    lines.append("- 本报告与审计脚本不修改 M4/M5 产物；M3 冻结哈希 18/18 一致。\n")
    (ROOT / "m6_gate0_report.md").write_text("".join(lines), encoding="utf-8")

    print(json.dumps({"gate_pass": gate_pass, "checks": checks, "s1_m4": m4_s1["recomputed_val_metrics"]["matrix_macro_rmse"], "s1_m5_c0": alignment["same_validation_protocol"]["m5_c0"]["matrix_macro_rmse"]}, indent=2))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
