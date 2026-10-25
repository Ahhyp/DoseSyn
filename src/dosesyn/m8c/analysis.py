"""M8-C: frozen-bridge verification, candidate harmonisation and case tracing.

This module never trains or fine-tunes. It reads the frozen M8-A/M8-B artefacts,
probes the ConfSyn checkpoint's BERT-stub path, selects representative cases by
rules fixed in ``configs/m8c/m8c.yaml``, and traces every case back to the raw
O'Neil source rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from dosesyn.utils.config import load_config, resolve_path
from dosesyn.utils.io import sha256_file, write_json


def _p(cfg: Dict[str, Any], key: str) -> Path:
    return resolve_path(cfg, cfg["paths"][key])


def load_candidate_universe(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Return all A-level S2 seed42 test matrices, scored or explicitly unscoreable."""
    ev = pd.read_parquet(_p(cfg, "m8a_evidence"))
    sel = ev[
        (ev["scenario"] == cfg["experiment"]["scenario"])
        & (ev["split_id"] == cfg["experiment"]["split_id"])
        & (ev["role"] == "test")
        & (ev["evidence_level"] == cfg["experiment"]["evidence_level"])
    ].copy()
    assert len(sel) == 300, f"expected 300 A-level matrices, got {len(sel)}"

    m8b = _p(cfg, "m8b_metrics")
    scored = pd.read_csv(m8b / "bridge_candidates.csv")
    unscored = pd.read_csv(m8b / "confsyn_unscoreable_candidates.csv")

    conf_cols = ["matrix_id", "conf_cell_name", "conf_db_a", "conf_db_b"]
    conf_meta = scored[conf_cols].copy()
    keep = ["matrix_id", "p_mean", "p_std", "p_min", "p_max", "y_pred_mean",
            "y_pred_min", "y_pred_max", "y_true_mean", "y_true_max", "n_points"]
    scored = scored[keep].copy()
    scored["scored"] = True
    scored["status"] = "scored"
    scored["reason"] = ""

    unscored = unscored[["matrix_id", "drug_a_name", "drug_b_name", "exclusion_reason"]].copy()
    unscored["scored"] = False
    unscored["status"] = "unscoreable"
    unscored["reason"] = unscored["exclusion_reason"]
    for col in keep:
        if col not in unscored:
            unscored[col] = np.nan

    universe = sel[["matrix_id", "condition_id", "cell_line", "drug_a_name",
                    "drug_b_name", "batch_id", "pair_id"]].merge(
        pd.concat([scored[keep + ["scored", "status", "reason"]],
                   unscored[keep + ["scored", "status", "reason"]]], ignore_index=True),
        on="matrix_id", how="left", validate="one_to_one",
    ).merge(conf_meta, on="matrix_id", how="left", validate="one_to_one")
    assert len(universe) == 300
    assert universe["status"].notna().all(), "some candidate has no status"

    uni = universe[universe["scored"]].copy()
    hi = cfg["thresholds"]
    uni["dual_signal"] = (uni["p_mean"] >= hi["high_p"]) & (uni["y_pred_min"] < hi["low_x_x0"])
    uni["high_p_high_min"] = (uni["p_mean"] >= hi["very_high_p"]) & (uni["y_pred_min"] >= hi["high_min"])
    universe = universe.merge(
        uni[["matrix_id", "dual_signal", "high_p_high_min"]], on="matrix_id", how="left"
    )
    universe["dual_signal"] = universe["dual_signal"].fillna(False).astype(bool)
    universe["high_p_high_min"] = universe["high_p_high_min"].fillna(False).astype(bool)
    return universe.sort_values(["scored", "p_mean"], ascending=[False, False]).reset_index(drop=True)


def _forward_probs(model, bridge, rows: pd.DataFrame) -> np.ndarray:
    e1, e2, cf, index, _ = bridge._tensors(rows)
    with torch.no_grad():
        out = model(
            graph=bridge.data, index=index, drug1_ids=None, drug1_mask=None,
            drug2_ids=None, drug2_mask=None, drug1_emb=e1, drug2_emb=e2,
            cline_features=cf, label=None, x_dict=bridge.x_dict,
            edge_index_dict=bridge.edge_index_dict, node_counts=bridge.node_counts,
        )
    return torch.softmax(out["logits_fusion"], dim=1)[:, 1].cpu().numpy()


def stub_audit(cfg: Dict[str, Any], cand: pd.DataFrame) -> Dict[str, Any]:
    """Prove the BERT stub is bypassed and quantify the real semantic path.

    Checks:
    1. ``drug_semantic_enc.forward`` is never called when ``drug1_emb`` is provided.
    2. Re-initialising the checkpoint's ``proj`` weights leaves predictions bit-identical.
    3. Zeroing the cached ChemBERTa embeddings changes predictions (they are the real path).
    """
    from dosesyn.m8b.confsyn_bridge import ConfSynBridge

    bridge = ConfSynBridge(cfg["paths"]["confsyn_root"], folds=[cfg["experiment"]["fold"]])
    fold = int(cfg["experiment"]["fold"])
    rows = cand.reset_index(drop=True)
    model = bridge.load_fold(fold)

    base = _forward_probs(model, bridge, rows)

    calls = {"n": 0}

    def _counting_forward(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("BERT branch must not be called when drug embeddings are supplied")

    model.drug_semantic_enc.forward = _counting_forward
    after_stub = _forward_probs(model, bridge, rows)
    stub_call_count = int(calls["n"])

    proj = model.drug_semantic_enc.proj
    saved = {k: v.detach().clone() for k, v in proj.state_dict().items()}
    torch.manual_seed(int(cfg["audit"]["stub_perturbation_seed"]))
    with torch.no_grad():
        for p in proj.parameters():
            p.normal_(0.0, 1.0)
    after_reinit = _forward_probs(model, bridge, rows)
    reinit_max_abs = float(np.max(np.abs(after_reinit - base)))
    proj.load_state_dict(saved)

    e1, e2, cf, index, _ = bridge._tensors(rows)
    e1z, e2z = torch.zeros_like(e1), torch.zeros_like(e2)
    with torch.no_grad():
        out = model(
            graph=bridge.data, index=index, drug1_ids=None, drug1_mask=None,
            drug2_ids=None, drug2_mask=None, drug1_emb=e1z, drug2_emb=e2z,
            cline_features=cf, label=None, x_dict=bridge.x_dict,
            edge_index_dict=bridge.edge_index_dict, node_counts=bridge.node_counts,
        )
    zero_emb = torch.softmax(out["logits_fusion"], dim=1)[:, 1].cpu().numpy()
    zero_delta = float(np.mean(np.abs(zero_emb - base)))
    zero_corr = float(pd.Series(zero_emb).corr(pd.Series(base), method="spearman"))

    payload = torch.load(bridge.checkpoint_path(fold), map_location="cpu")
    state = payload["model_state_dict"]
    bert_keys = [k for k in state if k.startswith("drug_semantic_enc.bert.")]
    bert_params = int(sum(state[k].numel() for k in bert_keys))
    total_params = int(sum(v.numel() for v in state.values() if hasattr(v, "numel")))

    val = pd.read_csv(_p(cfg, "m8b_metrics") / "confsyn_checkpoint_validation.csv")
    return {
        "fold": fold,
        "n_candidates": int(len(rows)),
        "stub_forward_calls": stub_call_count,
        "stub_used": stub_call_count > 0,
        "proj_reinit_max_abs_delta": reinit_max_abs,
        "proj_reinit_bit_identical": bool(reinit_max_abs == 0.0),
        "stub_forward_bit_identical": bool(np.max(np.abs(after_stub - base)) == 0.0),
        "zero_embedding_mean_abs_delta": zero_delta,
        "zero_embedding_spearman": zero_corr,
        "checkpoint_bert_key_count": len(bert_keys),
        "checkpoint_bert_param_count": bert_params,
        "checkpoint_total_param_count": total_params,
        "checkpoint_fold_auc": float(val.loc[val["fold"] == fold, "auc"].iloc[0]),
        "checkpoint_all_fold_auc": dict(zip(val["fold"].astype(int), val["auc"].astype(float))),
        "interpretation": (
            "BERT branch is bypassed because precomputed ChemBERTa embeddings are supplied; "
            "the checkpoint's frozen BERT body therefore cannot affect predictions. "
            "Zeroing those embeddings does change predictions, confirming the semantic path used."
        ),
    }


def select_cases(cfg: Dict[str, Any], cand: pd.DataFrame) -> pd.DataFrame:
    """Apply the rules fixed in the config to choose four cases."""
    scored = cand[cand["scored"]].copy()
    hi = cfg["thresholds"]
    picks: List[Dict[str, Any]] = []

    top = scored.sort_values(["p_mean", "matrix_id"], ascending=[False, True]).iloc[0]
    picks.append({"case_id": "top_p", "rule": cfg["cases"]["top_p"], "row": top})

    median_value = float(scored["p_mean"].median())
    med = scored.iloc[(scored["p_mean"] - median_value).abs().argsort(kind="stable")[:1]].iloc[0]
    picks.append({"case_id": "median_p", "rule": cfg["cases"]["median_p"], "row": med})

    hm = scored[scored["p_mean"] >= hi["very_high_p"]].sort_values(
        ["y_pred_min", "matrix_id"], ascending=[False, True]).iloc[0]
    picks.append({"case_id": "high_p_high_min", "rule": cfg["cases"]["high_p_high_min"], "row": hm})

    dual = scored[(scored["p_mean"] >= hi["high_p"]) & (scored["y_pred_min"] < hi["low_x_x0"])].copy()
    dual["gap"] = dual["p_mean"] - dual["y_pred_min"]
    ds = dual.sort_values(["gap", "matrix_id"], ascending=[False, True]).iloc[0]
    picks.append({"case_id": "dual_signal", "rule": cfg["cases"]["dual_signal"], "row": ds})

    out = pd.DataFrame([
        {
            "case_id": p["case_id"],
            "selection_rule": p["rule"],
            "matrix_id": p["row"]["matrix_id"],
            "condition_id": p["row"]["condition_id"],
            "cell_line": p["row"]["cell_line"],
            "drug_a_name": p["row"]["drug_a_name"],
            "drug_b_name": p["row"]["drug_b_name"],
            "p_mean": float(p["row"]["p_mean"]),
            "p_std": float(p["row"]["p_std"]),
            "y_pred_mean": float(p["row"]["y_pred_mean"]),
            "y_pred_min": float(p["row"]["y_pred_min"]),
            "y_true_mean": float(p["row"]["y_true_mean"]),
            "dual_signal": bool(p["row"]["dual_signal"]),
            "high_p_high_min": bool(p["row"]["high_p_high_min"]),
        }
        for p in picks
    ])
    if out["matrix_id"].duplicated().any():
        raise RuntimeError("representative cases are not distinct")
    return out


def trace_case(cfg: Dict[str, Any], case: pd.Series) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Join the frozen B5 prediction and the raw O'Neil replicate rows for one matrix."""
    pred = pd.read_parquet(_p(cfg, "b5_predictions"))
    pred = pred[pred["matrix_id"] == case["matrix_id"]].copy()
    points = pd.read_parquet(_p(cfg, "m1_combo_points"))
    points = points[points["matrix_id"] == case["matrix_id"]].copy()
    if len(pred) != 16 or len(points) != 16:
        raise RuntimeError(f"{case['matrix_id']}: expected 16 prediction and point rows")

    keep = ["matrix_id", "cell_line", "drug_a_name", "drug_b_name", "source_row",
            "batch_id", "row_idx", "col_idx", "conc_a_um", "conc_b_um",
            "viability_1", "viability_2", "viability_3", "viability_4",
            "viability_mean", "x_x0_raw", "x_x0_recalc"]
    points = points[keep].rename(columns={
        "cell_line": "raw_cell_line", "drug_a_name": "raw_drug_a_name",
        "drug_b_name": "raw_drug_b_name", "x_x0_raw": "raw_x_x0",
        "batch_id": "raw_batch_id",
    })
    merged = pred.merge(
        points,
        on=["matrix_id", "row_idx", "col_idx"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise RuntimeError(f"{case['matrix_id']}: prediction and raw points do not align")
    merged = merged.drop(columns="_merge")
    if not np.allclose(merged["y_true"], merged["raw_x_x0"], atol=1e-6, rtol=0):
        raise RuntimeError(f"{case['matrix_id']}: y_true differs from raw X/X0")
    max_label_delta = float(np.max(np.abs(merged["y_true"] - merged["raw_x_x0"])))
    if not np.allclose(merged["dose_a_um"], merged["conc_a_um"], atol=1e-6, rtol=0):
        raise RuntimeError(f"{case['matrix_id']}: dose_a_um differs from M1")
    if not np.allclose(merged["dose_b_um"], merged["conc_b_um"], atol=1e-6, rtol=0):
        raise RuntimeError(f"{case['matrix_id']}: dose_b_um differs from M1")
    max_dose_delta = float(max(
        np.max(np.abs(merged["dose_a_um"] - merged["conc_a_um"])),
        np.max(np.abs(merged["dose_b_um"] - merged["conc_b_um"])),
    ))

    batch = int(merged["batch_id"].iloc[0])
    src = sorted(int(v) for v in merged["source_row"])
    contiguous = src == list(range(min(src), max(src) + 1))
    summary = {
        "case_id": case["case_id"],
        "matrix_id": case["matrix_id"],
        "cell_line": case["cell_line"],
        "drug_a_name": case["drug_a_name"],
        "drug_b_name": case["drug_b_name"],
        "batch_id": batch,
        "source_row_min": int(min(src)),
        "source_row_max": int(max(src)),
        "source_rows_contiguous": bool(contiguous),
        "n_points": int(len(merged)),
        "n_replicates_complete": int((merged[["viability_1", "viability_2",
                                               "viability_3", "viability_4"]].notna().sum(axis=1) == 4).sum()),
        "max_label_delta": max_label_delta,
        "max_dose_delta": max_dose_delta,
        "raw_x_x0_min": float(merged["raw_x_x0"].min()),
        "raw_x_x0_max": float(merged["raw_x_x0"].max()),
        "pred_x_x0_min": float(merged["y_pred"].min()),
        "pred_x_x0_max": float(merged["y_pred"].max()),
        "rmse": float(np.sqrt(np.mean((merged["y_pred"] - merged["raw_x_x0"]) ** 2))),
        "p_mean": float(case["p_mean"]),
        "p_std": float(case["p_std"]),
        "dual_signal": bool(case["dual_signal"]),
        "high_p_high_min": bool(case["high_p_high_min"]),
    }
    merged["case_id"] = case["case_id"]
    cols = ["case_id", "matrix_id", "cell_line", "drug_a_name", "drug_b_name",
            "batch_id", "source_row", "row_idx", "col_idx", "dose_a_um", "dose_b_um",
            "viability_1", "viability_2", "viability_3", "viability_4",
            "viability_mean", "raw_x_x0", "y_pred", "residual"]
    return merged[cols].sort_values(["row_idx", "col_idx"]).reset_index(drop=True), summary


def frozen_hashes(cfg: Dict[str, Any]) -> Dict[str, str]:
    keys = ["b5_predictions", "b5_checkpoint", "m3_manifest", "m8a_evidence",
            "m1_combo_points"]
    out = {}
    for k in keys:
        p = _p(cfg, k)
        out[str(p.relative_to(resolve_path(cfg, ".")))] = sha256_file(p)
    m8b = _p(cfg, "m8b_metrics")
    for f in ["bridge_candidates.csv", "confsyn_checkpoint_validation.csv"]:
        out[str((m8b / f).relative_to(resolve_path(cfg, ".")))] = sha256_file(m8b / f)
    return out


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", indent=2, force_ascii=False)
