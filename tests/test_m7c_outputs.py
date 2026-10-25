"""M7-C deliverable tests: tables, numbers, symmetry, figures, chapter."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "outputs/metrics/m7c"
T = ROOT / "docs/paper/tables"
FIG = ROOT / "outputs/figures/m7c"


def rows(name):
    with open(M / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_required_tables_exist():
    names = [
        "table_c1_overall.csv",
        "table_c1_gains.csv",
        "table_c1_external_common.csv",
        "table_c2_m6a_ablation.csv",
        "table_c2_m6b_film.csv",
        "table_c3_center_holdout.csv",
        "table_c4_sparsity.csv",
        "table_c4_boundary.csv",
        "table_c5_symmetry.csv",
        "table_c7_complexity.csv",
    ]
    for n in names:
        p = M / n
        assert p.exists() and p.stat().st_size > 0, n
    assert len(list(T.glob("table4_*.md"))) >= 9


def test_overall_numbers_and_ordering():
    d = {r["model"]: r for r in rows("table_c1_overall.csv")}
    b5 = d["M6-B B5 (T3+FiLM)"]
    b3 = d["M4 B3 GINE"]
    assert abs(float(b5["S1"]) - 0.064991868274251) < 1e-9
    assert abs(float(b5["S2"]) - 0.0767734033898119) < 1e-9
    assert abs(float(b5["S3"]) - 0.1937350280841562) < 1e-9
    assert abs(float(b5["mean_rmse"]) - 0.1118334332494064) < 1e-9
    assert float(b5["S1"]) < float(b3["S1"])
    assert float(b5["S2"]) < float(b3["S2"])
    assert float(b5["mean_rmse"]) < float(b3["mean_rmse"])


def test_gain_arithmetic_matches_table():
    d = {r["model"]: r for r in rows("table_c1_overall.csv")}
    b5 = float(d["M6-B B5 (T3+FiLM)"]["S1"])
    for r in rows("table_c1_gains.csv"):
        if r["baseline"] == "M4 B3 GINE" and r["scenario"] == "S1":
            base = float(r["baseline_rmse"])
            assert abs(base - float(d["M4 B3 GINE"]["S1"])) < 1e-12
            expected = (base - b5) / base * 100
            assert abs(float(r["rel_gain_pct"]) - expected) < 1e-9
            return
    pytest.fail("gain row missing")


def test_ablation_order_and_values():
    d = rows("table_c2_m6a_ablation.csv")
    assert [r["model"] for r in d] == [
        "t0_b3_exact",
        "t1_c0_m4",
        "b3sym_scalar",
        "b3sym_fourier",
        "b3sym_film",
    ]
    by = {r["model"]: r for r in d}
    assert abs(float(by["b3sym_fourier"]["S1"]) - 0.08201616) < 1e-4
    assert abs(float(by["b3sym_film"]["S1"]) - 0.06696632) < 1e-4
    assert float(by["b3sym_film"]["S1"]) < float(by["b3sym_fourier"]["S1"])


def test_center_holdout_improvement():
    d = {r["model"]: r for r in rows("table_c3_center_holdout.csv")}
    b5, b3 = d["M6-B B5 (DoseSyn)"], d["M4 B3 GINE"]
    gain_s1 = (float(b3["S1"]) - float(b5["S1"])) / float(b3["S1"]) * 100
    gain_s2 = (float(b3["S2"]) - float(b5["S2"])) / float(b3["S2"]) * 100
    assert gain_s1 > 7.0
    assert gain_s2 > 8.0
    # S3 is intentionally not a positive claim; both directions allowed but tiny.
    assert abs(float(b5["S3"]) - float(b3["S3"])) < 0.01


def test_sparsity_and_boundary_negatives_are_reported():
    d = {(r["mask"], r["model"]): r for r in rows("table_c4_sparsity.csv")}
    b5 = d[("checkerboard_holdout_50", "M6-B B5 (DoseSyn)")]
    b3 = d[("checkerboard_holdout_50", "M4 B3 GINE")]
    assert float(b5["S1"]) < float(b3["S1"])
    assert float(b5["S2"]) < float(b3["S2"])
    c5 = d[("corners_holdout_25", "M6-B B5 (DoseSyn)")]
    c3 = d[("corners_holdout_25", "M4 B3 GINE")]
    assert float(c5["S1"]) > float(c3["S1"]), "25% corner weakness must be preserved"
    bound = {(r["model"]): r for r in rows("table_c4_boundary.csv")}
    assert float(bound["M4 B3 GINE"]["S1"]) < 0.15
    assert float(bound["M6-B B5 (DoseSyn)"]["S1"]) > 0.20


def test_symbolic_symmetry_is_exact():
    d = rows("table_c5_symmetry.csv")
    assert len(d) == 3
    assert sum(int(r["n_matrices"]) for r in d) == 9501
    for r in d:
        assert float(r["max_abs_error"]) == 0.0
        assert float(r["mean_abs_error"]) == 0.0
        assert int(r["n_matrices_gt_1e-5"]) == 0


def test_complexity_values():
    d = {r["model"]: r for r in rows("table_c7_complexity.csv")}
    assert int(float(d["M6-B B5 (T3+FiLM)"]["n_parameters"])) == 535682
    assert int(float(d["M4 B3 GINE"]["n_parameters"])) == 284545
    assert abs(float(d["M6-B B5 (T3+FiLM)"]["inference_ms_per_matrix_cpu"]) - 0.1837) < 0.02
    assert abs(float(d["M4 B3 GINE"]["inference_ms_per_matrix_cpu"]) - 0.1193) < 0.02
    # Training walls are not comparable at equal epochs; both must be reported.
    assert float(d["M4 B3 GINE"]["train_wall_s_s1_seed42"]) > 0
    assert float(d["M6-B B5 (T3+FiLM)"]["train_wall_s_s1_seed42"]) > 0


def test_chapter_sections_and_numbers():
    text = (ROOT / "docs/paper/chapter4_experiments.md").read_text(encoding="utf-8")
    for sec in ["4.3.1", "4.3.2", "4.3.3", "4.3.4", "4.3.5", "4.3.6", "4.3.7", "4.3.8"]:
        assert f"### {sec}" in text
    for token in ["0.0650", "0.0768", "0.1937", "0.1118", "0.0702", "0.0811", "535,682"]:
        assert token in text, token
    assert "标准 Bliss" not in text
    assert "NaN" not in text


def test_figures_and_surface_cases_exist():
    for stem in ["fig4_1_overall_performance", "fig4_4_center_holdout", "fig4_9_surface_good"]:
        for ext in (".svg", ".pdf", ".png"):
            p = FIG / f"{stem}{ext}"
            assert p.exists() and p.stat().st_size > 500, p
    cases = rows("surface_cases.csv")
    assert len(cases) >= 3
    assert all(0.0 <= float(c["rmse"]) < 1.0 for c in cases)
    meta = json.loads((M / "surface_cases.json").read_text())
    assert meta["n_eligible"] > 100
    assert meta["interpolation_case"]["rmse"] < 0.2


def test_summary_json_consistent():
    d = json.loads((M / "m7c_summary.json").read_text())
    h = d["headline"]
    assert math.isclose(h["B5_S1"], 0.064991868274251, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(h["B5_mean"], 0.1118334332494064, rel_tol=0, abs_tol=1e-12)
    assert d["chapter_sha256"] == __import__("hashlib").sha256(
        (ROOT / "docs/paper/chapter4_experiments.md").read_bytes()
    ).hexdigest()
