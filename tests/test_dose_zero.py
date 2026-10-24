"""Dose-zero / missing-data tests: no imputation, no clipping, floor behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_no_zero_or_negative_concentrations(combo, mono):
    assert combo["conc_a_source"].gt(0).all()
    assert combo["conc_b_source"].gt(0).all()
    assert mono["conc_source"].gt(0).all()
    assert not combo["conc_a_um_is_nonpositive"].any()
    assert not combo["conc_b_um_is_nonpositive"].any()
    assert not mono["conc_um_is_nonpositive"].any()
    # Hence no NaN log10 was introduced by a dose of zero.
    assert not combo["log10_conc_a"].isna().any()
    assert not combo["log10_conc_b"].isna().any()
    assert not mono["log10_conc"].isna().any()
    assert combo["log10_conc_a"].between(-6, 3).all()
    assert mono["log10_conc"].between(-6, 3).all()


def test_published_zero_responses_are_retained_and_flagged(combo, mono):
    zeros = combo.loc[combo["x_x0_raw"] == 0.0]
    assert len(zeros) == 213
    assert zeros["response_eq0"].all()
    assert (zeros["x_x0_recalc"] > 0).all()
    assert (zeros["x_x0_recalc"] < 1e-3).all()
    # They were not replaced by the formula value.
    assert zeros["x_x0_raw"].eq(0.0).all()
    assert int((combo["x_x0_raw"] == 0).sum()) == int(combo["response_eq0"].sum())
    assert int((mono["x_x0_raw"] == 0).sum()) == int(mono["response_eq0"].sum())


def test_high_responses_are_not_winsorised(combo, mono, cfg):
    upper = float(cfg["qc"]["x_over_x0_upper_outlier"])
    high = combo.loc[combo["x_x0_raw"] > upper]
    assert len(high) == 7
    assert (high["x_x0_raw"] > 2).all()
    assert high["response_gt2"].all()
    # No value was capped at the threshold.
    assert not np.isclose(high["x_x0_raw"], upper).any()
    assert combo["x_x0_raw"].max() > upper
    assert mono["x_x0_raw"].max() > 1


def test_missing_replicates_are_nan_not_imputed(combo, mono, combo_rep, mono_rep):
    assert int(combo["n_replicates_missing"].sum()) == 6018
    assert int(mono["n_replicates_missing"].sum()) == 15600
    assert int(combo_rep["is_missing"].sum()) == 6018
    assert int(mono_rep["is_missing"].sum()) == 15600
    assert len(combo_rep) == len(combo) * 4
    assert len(mono_rep) == len(mono) * 6
    # Long tables carry the same values as the wide columns, and no fill value.
    assert combo_rep["viability"].isna().sum() == 6018
    assert mono_rep["viability"].isna().sum() == 15600
    assert not np.isinf(combo_rep["viability"]).any()
    assert not np.isinf(mono_rep["viability"]).any()


def test_mean_is_nanmean_and_not_zero_filled(combo):
    vals = combo[["viability_1", "viability_2", "viability_3", "viability_4"]]
    expected = vals.mean(axis=1, skipna=True)
    assert np.allclose(combo["viability_mean"], expected, equal_nan=True)
    # Rows with any missing replicate must not have a zero substituted.
    reduced = combo.loc[combo["n_replicates_missing"] > 0]
    assert not (reduced["viability_mean"] == 0).all()


def test_replicate_stats_consistency(combo):
    vals = combo[["viability_1", "viability_2", "viability_3", "viability_4"]].to_numpy(float)
    spread = np.nanmax(vals, axis=1) - np.nanmin(vals, axis=1)
    assert np.allclose(combo["viability_spread"], spread, equal_nan=True)
    std = np.nanstd(vals, axis=1, ddof=1)
    assert np.allclose(combo["viability_std"], std, equal_nan=True)
    # std undefined for a single observation
    single = combo["n_replicates_observed"] < 2
    assert combo.loc[single, "viability_std"].isna().all()


def test_no_synthetic_rows_or_duplicated_source_rows(combo, mono):
    assert combo["source_row"].is_unique
    assert mono["source_row"].is_unique
    assert combo["source_row"].between(2, 368833).all()
    assert mono["source_row"].between(2, 16753).all()
    # Source index is a complete permutation of range(n).
    assert sorted(combo["source_index"]) == list(range(368832))
    assert sorted(mono["source_index"]) == list(range(16752))
