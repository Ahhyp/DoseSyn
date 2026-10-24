"""Traceability tests: processed rows must equal their published source rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dosesyn.data.make_matrices import pivot_matrix


def test_manifest_verdict(manifest):
    assert manifest["audit"]["verdict"] == "PASS"
    assert manifest["audit"]["hash_ok"] is True
    assert manifest["audit"]["row_counts_ok"] is True
    assert manifest["audit"]["columns_ok"] is True


def _sample_indices(n: int, k: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, size=min(k, n), replace=False))


def test_combo_rows_match_published_rows(combo, raw):
    src = raw["combo"].reset_index(drop=True)
    for i in _sample_indices(len(combo)):
        got = combo.iloc[i]
        want = src.iloc[int(got["source_index"])]
        assert int(got["source_row"]) == i + 2
        assert int(got["batch_id"]) == int(want["BatchID"])
        assert got["cell_line"] == want["cell_line"]
        assert got["drug_a_name"] == want["drugA_name"]
        assert got["drug_b_name"] == want["drugB_name"]
        assert got["combination_name"] == want["combination_name"]
        assert float(got["conc_a_source"]) == float(want["drugA Conc (µM)"])
        assert float(got["conc_b_source"]) == float(want["drugB Conc (µM)"])
        assert float(got["mu_raw"]) == float(want["mu/muMax"])
        assert float(got["x_x0_raw"]) == float(want["X/X0"])
        for r in range(1, 5):
            got_v = got[f"viability_{r}"]
            want_v = want[f"viability{r}"]
            if pd.isna(want_v):
                assert pd.isna(got_v)
            else:
                assert float(got_v) == float(want_v)


def test_mono_rows_match_published_rows(mono, raw):
    src = raw["mono"].reset_index(drop=True)
    for i in _sample_indices(len(mono)):
        got = mono.iloc[i]
        want = src.iloc[int(got["source_index"])]
        assert int(got["batch_id"]) == int(want["BatchID"])
        assert got["cell_line"] == want["cell_line"]
        assert got["drug_name"] == want["drug_name"]
        assert float(got["conc_source"]) == float(want["Drug_concentration (µM)"])
        assert float(got["mu_raw"]) == float(want["mu/muMax"])
        assert float(got["x_x0_raw"]) == float(want["X/X0"])
        for r in range(1, 7):
            got_v = got[f"viability_{r}"]
            want_v = want[f"viability{r}"]
            if pd.isna(want_v):
                assert pd.isna(got_v)
            else:
                assert float(got_v) == float(want_v)


def test_entity_registry_covers_all_published_names(combo, mono, registry):
    assert registry["n_cells"] == 39
    assert registry["n_drugs"] == 38
    assert len(set(registry["drugs"].values())) == 38
    assert len(set(registry["cells"].values())) == 39
    assert set(combo["cell_line"]) == set(registry["cells"])
    assert set(mono["cell_line"]) == set(registry["cells"])
    assert set(combo["drug_a_name"]) | set(combo["drug_b_name"]) == set(registry["drugs"])
    assert set(mono["drug_name"]) == set(registry["drugs"])
    id_to_name = {v: k for k, v in registry["cells"].items()}
    assert combo["cell_id"].map(id_to_name).eq(combo["cell_line"]).all()


def test_sample_matrix_csv_matches_points(cfg, combo):
    sample_dir = Path(cfg["paths"]["audit_dir"]) / "sample_matrices"
    manifest = json.loads((sample_dir / "sample_matrices_manifest.json").read_text())
    assert len(manifest) >= 3
    for entry in manifest:
        grid = pd.read_csv(sample_dir / entry["files"][0], index_col=0)
        expected = pivot_matrix(combo, entry["matrix_id"], "x_x0_raw")
        assert grid.shape == (4, 4)
        assert np.allclose(grid.to_numpy(float), expected.to_numpy(float), equal_nan=True)
        assert float(np.nanmin(grid.to_numpy(float))) == pytest.approx(entry["x_x0_min"])
        assert float(np.nanmax(grid.to_numpy(float))) == pytest.approx(entry["x_x0_max"])


def test_anomalies_reference_real_source_rows(combo, mono, anomalies):
    assert set(anomalies["anomaly_type"]) >= {
        "missing_replicate", "response_outlier_high", "response_floor_zero",
        "high_replicate_spread",
    }
    combo_rows = set(combo["source_row"].astype(int))
    mono_rows = set(mono["source_row"].astype(int))
    for r in anomalies.itertuples(index=False):
        sr = int(r.source_row)
        if r.dataset == "combination":
            assert sr in combo_rows
        else:
            assert sr in mono_rows
    # Missing-replicate records correspond to genuine NaNs in the wide table.
    miss = anomalies.loc[anomalies.anomaly_type == "missing_replicate"]
    assert len(miss) == 6018 + 15600
    one = miss.loc[miss.dataset == "combination"].iloc[0]
    assert pd.isna(
        combo.loc[combo.source_row == one.source_row, f"viability_{int(one.replicate_index)}"].iloc[0]
    )
    one = miss.loc[miss.dataset == "monotherapy"].iloc[0]
    assert pd.isna(
        mono.loc[mono.source_row == one.source_row, f"viability_{int(one.replicate_index)}"].iloc[0]
    )


def test_entity_index_counts(combo, mono, cfg):
    idx = pd.read_csv(Path(cfg["paths"]["audit_dir"]) / "oneil_entity_index.csv")
    drugs = idx.loc[idx.entity_type == "drug"]
    cells = idx.loc[idx.entity_type == "cell_line"]
    pairs = idx.loc[idx.entity_type == "drug_pair"]
    matrices = idx.loc[idx.entity_type == "combo_matrix"]
    curves = idx.loc[idx.entity_type == "mono_curve"]
    assert len(drugs) == 38
    assert len(cells) == 39
    assert len(pairs) == 583
    assert len(matrices) == 23052
    assert len(curves) == 2094
    assert int(matrices["n_combo_points"].sum()) == len(combo)
    assert int(curves["n_mono_points"].sum()) == len(mono)
    # Drug A/B usage is recorded for every drug.
    assert int(drugs["n_combo_points"].sum()) == 2 * len(combo)
