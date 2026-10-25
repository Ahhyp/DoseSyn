"""M8-C output-contract tests: counts, traces, stub audit, chapter integration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "outputs/metrics/m8c"
FIG = ROOT / "outputs/figures/m8c"


@pytest.fixture(scope="module")
def summary():
    return json.loads((MET / "m8c_summary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def universe():
    return pd.read_csv(MET / "m8c_candidate_universe.csv")


def test_candidate_count_conservation(universe, summary):
    assert len(universe) == 300
    assert universe["matrix_id"].is_unique
    assert int(universe["scored"].sum()) == 120
    assert int((~universe["scored"]).sum()) == 180
    assert summary["n_candidates"] == 300
    assert summary["n_scored"] + summary["n_unscoreable"] == 300


def test_unscoreable_all_sn38(universe):
    bad = universe[~universe["scored"]]
    assert len(bad) == 180
    texts = bad.apply(lambda r: f"{r.drug_a_name} {r.drug_b_name}", axis=1)
    assert texts.str.contains("SN-38").all()


def test_dual_signal_definition(universe, summary):
    scored = universe[universe["scored"]].copy()
    dual = (scored["p_mean"] >= 0.9) & (scored["y_pred_min"] < 0.2)
    assert int(dual.sum()) == summary["n_dual_signal"] == 27
    d = pd.read_csv(MET / "m8c_dual_signal_candidates.csv")
    assert len(d) == 27
    assert (d["p_mean"] >= 0.9).all()
    assert (d["y_pred_min"] < 0.2).all()


def test_high_p_high_min_definition(universe, summary):
    scored = universe[universe["scored"]]
    n = int(((scored["p_mean"] >= 0.95) & (scored["y_pred_min"] >= 0.35)).sum())
    assert n == summary["n_high_p_high_min"]


def test_four_cases_are_prespecified_and_distinct():
    cases = pd.read_csv(MET / "m8c_representative_cases.csv")
    assert len(cases) == 4
    assert cases["case_id"].is_unique
    assert set(cases["case_id"]) == {"top_p", "median_p", "high_p_high_min", "dual_signal"}
    for _, r in cases.iterrows():
        assert str(r.selection_rule).strip()
        assert r.matrix_id.startswith("MTX_")


def test_source_rows_traceable_and_complete():
    rows = pd.read_csv(MET / "m8c_case_source_rows.csv")
    assert len(rows) == 64
    for case_id, g in rows.groupby("case_id"):
        assert len(g) == 16, case_id
        g = g.sort_values("source_row")
        assert (g["source_row"].diff().dropna() == 1).all(), case_id
        for col in ["viability_1", "viability_2", "viability_3", "viability_4"]:
            assert g[col].notna().all(), (case_id, col)
        assert set(g["row_idx"]) == set(range(4))
        assert set(g["col_idx"]) == set(range(4))


def test_labels_match_raw_m1_within_float32():
    rows = pd.read_csv(MET / "m8c_case_source_rows.csv")
    assert (rows["residual"] - (rows["y_pred"] - rows["raw_x_x0"])).abs().max() < 1e-6


def test_stub_audit_is_decisive():
    audit = json.loads((MET / "m8c_stub_audit.json").read_text(encoding="utf-8"))
    assert audit["stub_forward_calls"] == 0
    assert audit["stub_used"] is False
    assert audit["proj_reinit_max_abs_delta"] == 0.0
    assert audit["stub_forward_bit_identical"] is True
    assert audit["zero_embedding_mean_abs_delta"] > 0.1
    assert len(audit["checkpoint_all_fold_auc"]) == 5
    assert all(0.8 <= v <= 1.0 for v in audit["checkpoint_all_fold_auc"].values())


def test_chapter_integration_and_wording():
    text = (ROOT / "docs/paper/chapter4_experiments.md").read_text(encoding="utf-8")
    assert "### 4.3.9 ConfSyn–DoseSyn 联合筛选与剂量响应分析" in text
    assert "4.3.9.5 互补性与适用边界" in text
    assert "双信号候选" in text
    assert "高协同倾向且存在低 `X/X0` 剂量点" in text
    assert "两条互补主线" in text
    assert "fig4_14_workflow" in text and "fig4_17_candidate_funnel" in text
    # the old incorrect framing may only appear in the correction sentence
    assert text.count("强分歧") <= 1


def test_figures_exist_in_three_formats():
    for stem in ["fig4_14_workflow", "fig4_15_dual_signal_landscape",
                 "fig4_16_case_surfaces", "fig4_17_candidate_funnel"]:
        for ext in ["svg", "pdf", "png"]:
            p = FIG / f"{stem}.{ext}"
            assert p.exists() and p.stat().st_size > 1000, p


def test_reports_state_verdict_and_boundaries():
    c = (ROOT / "m8c_report.md").read_text(encoding="utf-8")
    f = (ROOT / "m8_final_report.md").read_text(encoding="utf-8")
    assert "PASS" in c and "PASS" in f
    for t in [c, f]:
        assert "X/X0" in t
        assert "DoseSyn" in t and "ConfSyn" in t
    assert "不重新训练" in f or "不重新训练" in c


def test_frozen_hashes_unchanged():
    data = json.loads((MET / "m8c_frozen_hashes.json").read_text(encoding="utf-8"))
    files = data.get("files") or data
    checked = 0
    for rel, expected in files.items():
        if not isinstance(expected, str):
            continue
        p = ROOT / rel
        if not p.exists():
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == expected, rel
        checked += 1
    assert checked >= 5
