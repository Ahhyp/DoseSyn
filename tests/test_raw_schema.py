"""M0 regression tests: file identity, true container format, schema, counts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dosesyn.data import inspect as insp
from dosesyn.data.parse_oneil import resolve_workbook
from dosesyn.utils.io import detect_container, sha256_file


def test_raw_files_are_byte_identical_to_provenance(cfg, manifest):
    for entry in manifest["files"]:
        role = entry["role"]
        raw_path = Path(cfg["files"][f"{role}_path"])
        assert raw_path.exists(), f"missing raw file {raw_path}"
        assert sha256_file(raw_path) == entry["sha256"], f"{role}: file changed on disk"
        assert entry["sha256"] == cfg["expected_sha256"][role]
        assert entry["sha256_match"] is True


def test_combo_file_is_ooxml_despite_xls_suffix(cfg):
    """The most important M0 trap: `.xls` is actually an OOXML/ZIP workbook."""
    path = Path(cfg["files"]["combo_path"])
    assert path.suffix == ".xls"
    info = detect_container(path)
    assert info["is_ooxml"] is True
    assert info["is_legacy_xls"] is False
    assert info["container"] == "zip"
    assert info["extension_matches_content"] is False
    assert info["signature_hex"].startswith("504b0304")
    # The exact bytes are copied to a .xlsx working copy; the copy must match.
    work, engine = resolve_workbook(cfg, "combo")
    assert work.exists(), "working copy missing; rerun scripts/00_audit_raw.py"
    assert engine == "openpyxl"
    assert sha256_file(work) == sha256_file(path)


def test_mono_and_legend_are_ooxml(cfg):
    for role in ("mono", "legend"):
        info = detect_container(Path(cfg["files"][f"{role}_path"]))
        assert info["is_ooxml"] is True
        assert info["extension_matches_content"] is True


def test_expected_columns_present(cfg, raw):
    """Every configured column exists exactly once (sheet order is audited separately)."""
    for role in ("combo", "mono"):
        expected = insp.expected_columns(cfg, role)
        actual = [str(c) for c in raw[role].columns]
        assert sorted(actual) == sorted(expected), f"{role} column set changed"
        assert len(actual) == len(set(actual)), f"{role} has duplicate columns"


def test_row_counts_reconcile(cfg, raw):
    assert len(raw["combo"]) == cfg["expected_counts"]["combo_rows"] == 368832
    assert len(raw["mono"]) == cfg["expected_counts"]["mono_rows"] == 16752


def test_published_missing_replicate_counts(raw):
    combo_missing = {f"viability{i}": int(raw["combo"][f"viability{i}"].isna().sum()) for i in range(1, 5)}
    mono_missing = {f"viability{i}": int(raw["mono"][f"viability{i}"].isna().sum()) for i in range(1, 7)}
    assert combo_missing == {"viability1": 0, "viability2": 78, "viability3": 198, "viability4": 5742}
    assert mono_missing == {
        "viability1": 0, "viability2": 0, "viability3": 0,
        "viability4": 5200, "viability5": 5200, "viability6": 5200,
    }


def test_batch_counts(raw):
    assert raw["combo"]["BatchID"].value_counts().sort_index().to_dict() == {
        1: 358560, 2: 5232, 3: 5040,
    }
    assert raw["mono"]["BatchID"].value_counts().sort_index().to_dict() == {
        1: 11856, 2: 2736, 3: 2160,
    }


def test_published_ranges_and_entities(raw):
    x = raw["combo"]["X/X0"]
    assert x.min() == 0.0
    assert round(float(x.max()), 5) == 2.34333
    assert int((x > 1).sum()) == 25247
    assert int((x > 2).sum()) == 7
    assert int((x < 0).sum()) == 0
    assert raw["mono"]["X/X0"].max() > 1
    assert raw["combo"]["combination_name"].nunique() == 583
    assert raw["combo"]["cell_line"].nunique() == 39
    assert len(set(raw["combo"]["drugA_name"]) | set(raw["combo"]["drugB_name"])) == 38
    assert set(raw["mono"]["drug_name"]) == (
        set(raw["combo"]["drugA_name"]) | set(raw["combo"]["drugB_name"])
    )


def test_combination_name_matches_drug_parts(raw):
    parts = raw["combo"]["drugA_name"] + " & " + raw["combo"]["drugB_name"]
    assert (raw["combo"]["combination_name"] == parts).all()


def test_no_duplicate_columns(raw):
    for role in ("combo", "mono"):
        assert not insp.duplicate_column_names(raw[role])
