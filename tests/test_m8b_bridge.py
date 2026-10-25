"""Contract tests for the M8-B frozen-model bridge audit."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "outputs/metrics/m8b"


@pytest.fixture(scope="module")
def cfg():
    with open(ROOT / "configs/m8b/m8b.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def cand():
    path = MET / "bridge_candidates.csv"
    if not path.exists():
        pytest.skip("run scripts/27_run_m8b.py first")
    return pd.read_csv(path)


def test_five_frozen_checkpoints_exist(cfg):
    ckpt_dir = Path(cfg["confsyn_root"]) / (
        "result/paper_reproduction/2026-09-10_15-44-09_19269.mgt/"
        "random_cv/weight_DrugCombDB"
    )
    for fold in cfg["folds"]:
        p = ckpt_dir / f"cv_DrugCombDB_mode_1_fold_num_{fold}.pth"
        assert p.exists(), f"missing checkpoint {p}"
        assert p.stat().st_size > 300_000_000
    val = pd.read_csv(MET / "confsyn_checkpoint_validation.csv")
    assert len(val) == 5
    assert val["auc"].between(0.8, 1.0).all()


def test_bridge_candidates_are_a_level_s2_test(cand):
    assert len(cand) > 0
    assert (cand["evidence_level"] == "A").all()
    assert (cand["scenario"] == "s2_unseen_pair").all()
    assert (cand["split_id"] == "seed_42").all()
    assert cand["matrix_id"].is_unique


def test_probability_columns_are_five_fold_and_bounded(cand):
    pcols = [f"p_fold{k}" for k in range(5)]
    for c in pcols:
        assert c in cand.columns
        assert cand[c].between(0, 1).all()
    assert (cand["p_min"] <= cand["p_mean"]).all()
    assert (cand["p_mean"] <= cand["p_max"]).all()
    assert (cand["p_std"] >= 0).all()


def test_dosesyn_surfaces_are_4x4(cand):
    surf = pd.read_json(MET / "dosesyn_surfaces.json").merge(
        cand[["matrix_id", "y_pred_min", "y_pred_mean"]], on="matrix_id", how="left"
    )
    assert len(surf) == len(cand)
    for s in surf["surface"]:
        assert len(s) == 4 and all(len(row) == 4 for row in s)
    assert (surf["y_pred_min"] <= surf["y_pred_mean"]).all()


def test_unscoreable_candidates_are_documented():
    p = MET / "confsyn_unscoreable_candidates.csv"
    assert p.exists()
    ex = pd.read_csv(p)
    assert len(ex) > 0
    assert (ex["exclusion_reason"] == "drug_or_cell_absent_from_DrugCombDB_graph").all()


def test_no_retraining_in_bridge_scripts():
    for name in ("27_run_m8b.py", "confsyn_bridge.py"):
        src = (ROOT / ("scripts" if name.startswith("27") else "src/dosesyn/m8b") / name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"backward", "step"}:
                    pytest.fail(f"{name} contains optimizer/backward call")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "train":
                pytest.fail(f"{name} calls train()")


def test_report_states_frozen_and_caveats():
    text = (ROOT / "m8b_report.md").read_text(encoding="utf-8")
    assert "冻结" in text or "frozen" in text.lower()
    assert "不重新训练" in text
    assert "synergy >= 0" in text
    # The required semantic caveat must be present, not silently omitted.
    assert ("不代表" in text) or ("不可" in text)
