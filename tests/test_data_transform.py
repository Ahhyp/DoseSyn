"""M1 format-conversion tests: values preserved, units, formulas, no clipping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dosesyn.data.normalize import recompute_response, response_residual


def test_all_published_rows_are_kept(combo, mono, raw):
    assert len(combo) == len(raw["combo"]) == 368832
    assert len(mono) == len(raw["mono"]) == 16752
    assert combo["source_row"].is_unique
    assert mono["source_row"].is_unique
    assert combo["source_index"].tolist() == list(range(len(raw["combo"])))
    assert mono["source_index"].tolist() == list(range(len(raw["mono"])))
    # Excel header is row 1, so data starts at row 2.
    assert combo["source_row"].min() == 2
    assert combo["source_row"].max() == len(raw["combo"]) + 1
    assert mono["source_row"].max() == len(raw["mono"]) + 1


@pytest.mark.parametrize("i", [1, 2, 3, 4])
def test_combo_technical_replicates_preserved(combo, raw, i):
    pd.testing.assert_series_equal(
        combo[f"viability_{i}"],
        raw["combo"][f"viability{i}"],
        check_names=False,
        check_dtype=False,
    )
    assert combo[f"viability_{i}"].isna().sum() == raw["combo"][f"viability{i}"].isna().sum()


@pytest.mark.parametrize("i", [1, 2, 3, 4, 5, 6])
def test_mono_technical_replicates_preserved(mono, raw, i):
    pd.testing.assert_series_equal(
        mono[f"viability_{i}"],
        raw["mono"][f"viability{i}"],
        check_names=False,
        check_dtype=False,
    )


def test_combo_responses_and_mu_preserved(combo, raw):
    pd.testing.assert_series_equal(combo["x_x0_raw"], raw["combo"]["X/X0"], check_names=False)
    pd.testing.assert_series_equal(combo["mu_raw"], raw["combo"]["mu/muMax"], check_names=False)


def test_mono_responses_and_mu_preserved(mono, raw):
    pd.testing.assert_series_equal(mono["x_x0_raw"], raw["mono"]["X/X0"], check_names=False)
    pd.testing.assert_series_equal(mono["mu_raw"], raw["mono"]["mu/muMax"], check_names=False)


def test_no_clipping_and_high_responses_retained(combo, mono):
    assert combo["x_x0_raw"].max() == pytest.approx(2.34333)
    assert int((combo["x_x0_raw"] > 1).sum()) == 25247
    assert int((combo["x_x0_raw"] > 2).sum()) == 7
    assert combo["x_x0_raw"].min() == 0.0
    assert int((combo["x_x0_raw"] < 0).sum()) == 0
    assert int(combo["response_gt1"].sum()) == 25247
    assert int(combo["response_gt2"].sum()) == 7
    assert len(combo.loc[combo["response_gt2"]]) == 7
    assert mono["x_x0_raw"].max() == pytest.approx(1.71937)
    assert int((mono["x_x0_raw"] > 1).sum()) == 5512
    assert mono["x_x0_raw"].min() >= 0


def test_unit_conversion_is_identity_for_micromolar(combo, mono):
    assert np.allclose(combo["conc_a_um"], combo["conc_a_source"], rtol=0, atol=0)
    assert np.allclose(combo["conc_b_um"], combo["conc_b_source"], rtol=0, atol=0)
    assert np.allclose(mono["conc_um"], mono["conc_source"], rtol=0, atol=0)


def test_concentrations_positive_and_log_defined(combo, mono):
    assert (combo["conc_a_um"] > 0).all()
    assert (combo["conc_b_um"] > 0).all()
    assert (mono["conc_um"] > 0).all()
    assert not combo[["conc_a_um_is_nonpositive", "conc_b_um_is_nonpositive"]].any().any()
    assert not mono["conc_um_is_nonpositive"].any()
    assert np.isfinite(combo["log10_conc_a"]).all()
    assert np.isfinite(combo["log10_conc_b"]).all()
    assert np.isfinite(mono["log10_conc"]).all()


def test_response_formula_reproduced_within_rounding_tolerance(combo, mono, cfg):
    tol = float(cfg["qc"]["response_formula_tolerance"])
    for frame, label in ((combo, "combo"), (mono, "mono")):
        residual = response_residual(frame["x_x0_raw"], frame["mu_raw"])
        assert float(residual.max()) <= tol, f"{label} formula residual too large"
        assert frame["response_formula_ok"].all()
        assert (frame["response_formula_residual"] <= tol).all()
    # spot-check exact formula
    mu = np.array([-7.99858, 0.0, 0.5, 1.0, 1.61428])
    expected = np.exp(np.log(4.0) * (mu - 1.0))
    assert np.allclose(recompute_response(mu), expected, rtol=1e-12)


def test_pair_canonicalisation_is_consistent(combo):
    low_is_a = combo["is_drug_a_low_id"]
    expected_low = np.where(low_is_a, combo["conc_a_um"], combo["conc_b_um"])
    expected_high = np.where(low_is_a, combo["conc_b_um"], combo["conc_a_um"])
    assert np.allclose(combo["dose_low_um"], expected_low)
    assert np.allclose(combo["dose_high_um"], expected_high)
    assert (combo["pair_drug_low_id"] < combo["pair_drug_high_id"]).all()


def test_replicate_summary_uses_observed_values_only(combo, raw):
    cols = [f"viability_{i}" for i in range(1, 5)]
    expected_mean = raw["combo"][[f"viability{i}" for i in range(1, 5)]].mean(axis=1)
    pd.testing.assert_series_equal(
        combo["viability_mean"].reset_index(drop=True),
        expected_mean.reset_index(drop=True),
        check_names=False,
    )
    assert combo["viability_mean"].isna().sum() == 0
    # No value is invented for missing replicates.
    missing = combo[cols].isna().sum(axis=1)
    assert (combo["n_replicates_missing"] == missing).all()
    assert (combo["n_replicates_observed"] == 4 - missing).all()


def test_mono_meta_grid_matches_points(mono, mono_meta):
    import json

    m = mono_meta.set_index("mono_curve_id")
    sample = mono["mono_curve_id"].drop_duplicates().head(20)
    for curve_id in sample:
        doses = sorted(mono.loc[mono.mono_curve_id == curve_id, "conc_um"].unique())
        encoded = json.loads(m.loc[curve_id, "dose_um_json"])
        assert np.allclose(doses, encoded)
