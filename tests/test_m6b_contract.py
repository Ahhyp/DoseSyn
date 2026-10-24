"""M6-B protocol contract tests: variants, selection rule, test isolation."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dosesyn.utils.config import load_config  # noqa: E402
import importlib.util  # noqa: E402


def _load_selector():
    spec = importlib.util.spec_from_file_location("m6b_select", ROOT / "scripts" / "12_select_m6b.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_all_variants_defined_and_switches_independent():
    cfg = load_config("configs/m6b_experiments.yaml")
    variants = cfg["variants"]
    assert set(variants) == {
        "b0_b3_exact", "b1_t3_fourier", "b2_t3_mono", "b3_t3_mono_boundary",
        "b4_t3_mono_boundary_smooth", "b5_t3_film_reg", "b6_t3_film_reg_pharm",
    }
    assert variants["b1_t3_fourier"].get("mono", False) is False
    assert variants["b2_t3_mono"]["mono"] is True
    assert variants["b2_t3_mono"].get("boundary", False) is False
    assert variants["b3_t3_mono_boundary"]["boundary"] is True
    assert variants["b4_t3_mono_boundary_smooth"]["smooth"] is True
    assert variants["b5_t3_film_reg"]["kind"] == "b3sym_film"
    assert variants["b6_t3_film_reg_pharm"]["mono"] is True


def test_validation_path_does_not_read_test_split():
    src = (ROOT / "src/dosesyn/train/m6b_trainer.py").read_text()
    tree = ast.parse(src)
    # `load_matrix_table(..., "test")` is only allowed inside `if eval_test:`
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            cond = ast.unparse(node.test)
            if "eval_test" in cond:
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and getattr(child.func, "id", "") == "load_matrix_table":
                    args = [ast.unparse(a) for a in child.args]
                    assert '"test"' not in args, "unconditional test split read"
    assert "if eval_test:" in src


def test_selection_prefers_parsimony_when_gain_small():
    mod = _load_selector()
    rows = []
    for v, s1, s3, p in [
        ("b1_t3_fourier", 0.080, 0.169, 492290),
        ("b2_t3_mono", 0.0798, 0.1692, 600000),
        ("b3_t3_mono_boundary", 0.0799, 0.1691, 610000),
        ("b4_t3_mono_boundary_smooth", 0.0797, 0.1690, 620000),
        ("b5_t3_film_reg", 0.078, 0.171, 530000),
    ]:
        for sc, val in (("s1_random_condition", s1), ("s3_unseen_cell", s3)):
            rows.append({"variant": v, "scenario": sc, "matrix_macro_rmse": val,
                         "point_rmse": val, "point_mae": val, "pooled_pearson": 0.9,
                         "r2": 0.8, "swap_max_abs": 1e-7, "n_parameters": p,
                         "best_epoch": 100, "epochs_run": 120, "wall_seconds": 1})
    out = mod.select(rows)
    assert out["selected"] == "b1_t3_fourier"
    assert "0.001" in out["reason"] or "S3" in out["reason"]


def test_formal_test_policy_declared():
    cfg = load_config("configs/m6b_experiments.yaml")
    assert cfg["formal"]["test_policy"] == "single_read_after_checkpoint_selection"
    assert cfg["training"]["epochs"] >= 200
