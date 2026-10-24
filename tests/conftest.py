"""Shared pytest fixtures for the DoseSyn M1 test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.utils.config import load_config  # noqa: E402
from dosesyn.utils.io import read_json  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "data_oneil.yaml")


@pytest.fixture(scope="session")
def manifest(cfg):
    return read_json(Path(cfg["paths"]["audit_dir"]) / "raw_file_manifest.json")


@pytest.fixture(scope="session")
def registry(cfg):
    return read_json(Path(cfg["paths"]["audit_dir"]) / "oneil_entity_registry.json")


@pytest.fixture(scope="session")
def raw(cfg):
    cache = Path(cfg["paths"]["interim_dir"]) / "raw_oneil"
    frames = {}
    for role in ("combo", "mono"):
        p = cache / f"{role}.parquet"
        if not p.exists():
            pytest.fail(f"raw cache missing: {p} (run scripts/00_audit_raw.py)")
        frames[role] = pd.read_parquet(p)
    return frames


@pytest.fixture(scope="session")
def combo(cfg):
    p = Path(cfg["paths"]["processed_dir"]) / "oneil_combo_points.parquet"
    if not p.exists():
        pytest.fail(f"processed table missing: {p} (run scripts/01_prepare_oneil.py)")
    return pd.read_parquet(p)


@pytest.fixture(scope="session")
def combo_meta(cfg):
    return pd.read_parquet(
        Path(cfg["paths"]["processed_dir"]) / "oneil_combo_matrix_metadata.parquet"
    )


@pytest.fixture(scope="session")
def combo_rep(cfg):
    return pd.read_parquet(
        Path(cfg["paths"]["processed_dir"]) / "oneil_combo_replicates.parquet"
    )


@pytest.fixture(scope="session")
def mono(cfg):
    p = Path(cfg["paths"]["processed_dir"]) / "oneil_monotherapy.parquet"
    if not p.exists():
        pytest.fail(f"processed table missing: {p} (run scripts/01_prepare_oneil.py)")
    return pd.read_parquet(p)


@pytest.fixture(scope="session")
def mono_meta(cfg):
    return pd.read_parquet(
        Path(cfg["paths"]["processed_dir"]) / "oneil_mono_curve_metadata.parquet"
    )


@pytest.fixture(scope="session")
def mono_rep(cfg):
    return pd.read_parquet(
        Path(cfg["paths"]["processed_dir"]) / "oneil_mono_replicates.parquet"
    )


@pytest.fixture(scope="session")
def anomalies(cfg):
    return pd.read_csv(Path(cfg["paths"]["audit_dir"]) / "oneil_anomalies.csv")


@pytest.fixture(scope="session")
def batch_repeats(cfg):
    return pd.read_csv(Path(cfg["paths"]["audit_dir"]) / "batch_repeats.csv")


# ---------------------------------------------------------------------------
# M3 fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def m3_cfg():
    return load_config(ROOT / "configs" / "dataset_primary_v1.yaml")


@pytest.fixture(scope="session")
def m3_primary(m3_cfg):
    p = Path(m3_cfg["outputs"]["matrix_index"])
    if not p.exists():
        pytest.fail(f"primary index missing: {p} (run scripts/03_generate_splits.py)")
    return pd.read_parquet(p)


@pytest.fixture(scope="session")
def m3_input_manifest(m3_cfg):
    return read_json(Path(m3_cfg["outputs"]["input_manifest"]))


@pytest.fixture(scope="session")
def m3_split_dirs(m3_cfg):
    root = Path(m3_cfg["outputs"]["splits_dir"])
    dirs = []
    for scenario in ("s1_random_condition", "s2_unseen_pair", "s3_unseen_cell"):
        for sub in sorted((root / scenario).glob("*")):
            if (sub / "split_metadata.json").exists():
                dirs.append((scenario, sub))
    if not dirs:
        pytest.fail("no M3 splits found (run scripts/03_generate_splits.py)")
    return dirs

