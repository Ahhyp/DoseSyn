#!/usr/bin/env python
"""M0 — O'Neil raw data audit.

Reads the three supplied workbooks, detects their true container format from
magic bytes, verifies SHA256 against the recorded provenance, reconciles row
counts and columns, and writes:

  outputs/audit/raw_file_manifest.json   machine-readable audit
  outputs/audit/raw_schema.md            human-readable schema
  data/interim/raw_oneil/*.parquet       raw cache for the M1 pipeline

The raw files are opened read-only and are never rewritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.data import inspect as insp  # noqa: E402
from dosesyn.data.parse_oneil import (  # noqa: E402
    build_entity_registry,
    load_raw_frames,
    raw_cache_path,
    read_raw_from_workbook,
    resolve_workbook,
)
from dosesyn.utils.config import ensure_directories, load_config  # noqa: E402
from dosesyn.utils.io import (  # noqa: E402
    environment_snapshot,
    utc_now_iso,
    write_json,
    write_parquet,
)
from dosesyn.utils.logging import setup_logging  # noqa: E402


def _batch_counts(df: pd.DataFrame) -> dict:
    if "BatchID" not in df.columns:
        return {}
    return {int(k): int(v) for k, v in df["BatchID"].value_counts().sort_index().items()}


def _text_schema(df: pd.DataFrame) -> list:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "name": str(col),
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "missing_pct": round(100.0 * float(s.isna().mean()), 4),
                "min": (None if s.dropna().empty or not pd.api.types.is_numeric_dtype(s)
                        else float(s.min())),
                "max": (None if s.dropna().empty or not pd.api.types.is_numeric_dtype(s)
                        else float(s.max())),
                "n_unique": int(s.nunique(dropna=True)),
            }
        )
    return rows


def render_schema_md(manifest: dict) -> str:
    lines = ["# O'Neil raw schema audit (M0)", ""]
    lines.append(f"- Generated: `{manifest['generated_at']}`")
    lines.append(f"- Python: `{manifest['environment']['python']}` / pandas `{manifest['environment']['pandas']}`")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| Role | File | Bytes | SHA256 | Container | Suffix matches content |")
    lines.append("| --- | --- | ---: | --- | --- | --- |")
    for e in manifest["files"]:
        lines.append(
            f"| {e['role']} | `{e['filename']}` | {e['size_bytes']:,} | `{e['sha256']}` | "
            f"{e['family']} / {e['container']} | {e['extension_matches_content']} |"
        )
    lines.append("")
    lines.append("The combination file carries a `.xls` suffix but its magic bytes are "
                 "`504b0304` (OOXML/ZIP). It is therefore NOT a BIFF workbook and must not be "
                 "opened with xlrd; the pipeline uses a byte-identical `.xlsx` copy under "
                 "`data/interim/format_work/` read by openpyxl.")
    lines.append("")
    for role in ("combo", "mono", "legend"):
        r = manifest["tables"].get(role)
        if not r:
            continue
        lines.append(f"## Table `{role}`")
        lines.append("")
        lines.append(f"- Sheet: `{r['sheet']}`")
        lines.append(f"- Rows: {r['n_rows']:,}  Columns: {r['n_columns']}")
        lines.append(f"- Rows expected by config: {r.get('expected_rows')}")
        lines.append(f"- Row count match: **{r.get('row_count_match')}**")
        lines.append(f"- Column set exact match: **{r.get('columns_ok')}**")
        lines.append(f"- Batch counts: `{r.get('batch_counts')}`")
        lines.append("")
        lines.append("| Column | dtype | Non-null | Missing | Missing % | Min | Max | Unique |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for c in r["columns"]:
            lines.append(
                f"| `{c['name']}` | {c['dtype']} | {c['non_null']:,} | {c['missing']:,} | "
                f"{c['missing_pct']} | {c['min']} | {c['max']} | {c['n_unique']:,} |"
            )
        lines.append("")
    lines.append("## Entity inventory")
    lines.append("")
    ent = manifest["entities"]
    lines.append(f"- Cell lines: {ent['n_cells']}")
    lines.append(f"- Drugs: {ent['n_drugs']}")
    lines.append(f"- Unordered drug pairs: {ent['n_unordered_pairs']}")
    lines.append(f"- Drugs appearing only as drug A: {ent['drugs_only_as_a']}")
    lines.append(f"- Drugs appearing only as drug B: {ent['drugs_only_as_b']}")
    lines.append("")
    lines.append("## Legend (field definitions published with the data)")
    lines.append("")
    for role, fields in manifest["legend"]["fields"].items():
        lines.append(f"### {role}")
        lines.append("")
        for key, desc in fields.items():
            lines.append(f"- **`{key}`** — {desc}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="DoseSyn M0 raw data audit")
    parser.add_argument("--config", default="configs/data_oneil.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_directories(cfg)
    log_path = Path(cfg["paths"]["logs_dir"]) / "00_audit_raw.log"
    logger = setup_logging("dosesyn.audit", log_path)
    logger.info("DoseSyn M0 audit starting; config=%s", cfg["_config_path"])

    entries = [insp.file_entry(cfg, role) for role in ("combo", "mono", "legend")]
    for e in entries:
        logger.info(
            "file %-6s %-55s %12d B  sha256=%s  container=%s/%s  sha_match=%s",
            e["role"], e["filename"], e["size_bytes"], e["sha256"][:16],
            e["family"], e["container"], e["sha256_match"],
        )
    hash_ok = all(e["sha256_match"] for e in entries)

    tables = {}
    frames = {}
    expected = {
        "combo": int(cfg["expected_counts"]["combo_rows"]),
        "mono": int(cfg["expected_counts"]["mono_rows"]),
    }
    for role in ("combo", "mono"):
        readable, engine = resolve_workbook(cfg, role)
        sheets = insp.workbook_sheets(readable)
        logger.info("%s workbook: path=%s engine=%s sheets=%s", role, readable.name, engine, sheets)
        df = read_raw_from_workbook(cfg, role)
        check = insp.check_columns(df, cfg, role)
        dup_cols = insp.duplicate_column_names(df)
        frames[role] = df
        tables[role] = {
            "sheet": cfg["raw_columns"][role]["sheet"],
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "expected_rows": expected[role],
            "row_count_match": int(df.shape[0]) == expected[role],
            "columns_ok": bool(check["ok"]),
            "columns_missing": check["missing"],
            "columns_extra": check["extra"],
            "columns_exact_order": check["exact_order"],
            "duplicate_column_names": dup_cols,
            "batch_counts": _batch_counts(df),
            "columns": _text_schema(df),
            "workbook_sheets": sheets,
            "reader": engine,
            "read_path_relative": str(readable.relative_to(Path(cfg["_root"]))),
        }
        logger.info(
            "%s rows=%d expected=%d columns_ok=%s batch_counts=%s",
            role, df.shape[0], expected[role], check["ok"], tables[role]["batch_counts"],
        )

    legend_df = pd.read_excel(
        cfg["files"]["legend_path"],
        sheet_name=cfg["raw_columns"]["legend"]["sheet"],
        header=None,
        engine="openpyxl",
    )
    legend_df.columns = [f"col_{i}" for i in range(legend_df.shape[1])]
    legend = insp.parse_legend(cfg)
    registry = build_entity_registry(frames["mono"], frames["combo"])

    cache = raw_cache_path(cfg, "combo").parent
    write_parquet(frames["combo"], cache / "combo.parquet")
    write_parquet(frames["mono"], cache / "mono.parquet")
    write_parquet(legend_df, cache / "legend.parquet")
    write_json(registry, Path(cfg["paths"]["audit_dir"]) / "oneil_entity_registry.json")
    logger.info("raw cache written to %s", cache)

    manifest = {
        "study": "ONEIL",
        "stage": "M0",
        "generated_at": utc_now_iso(),
        "environment": environment_snapshot(),
        "config": {
            "path": cfg["_config_path"],
            "root": cfg["_root"],
            "units": cfg.get("units", {}),
            "expected_counts": cfg["expected_counts"],
        },
        "files": entries,
        "tables": tables,
        "legend": legend,
        "entities": {
            "n_cells": registry["n_cells"],
            "n_drugs": registry["n_drugs"],
            "n_unordered_pairs": registry["n_unordered_pairs"],
            "drugs_only_as_a": registry["drugs_only_as_a"],
            "drugs_only_as_b": registry["drugs_only_as_b"],
            "cell_lines": registry["cell_lines"],
            "drugs": registry["drug_names"],
            "cell_id_map": registry["cells"],
            "drug_id_map": registry["drugs"],
        },
        "audit": {
            "hash_ok": hash_ok,
            "row_counts_ok": all(
                tables[r]["row_count_match"] for r in ("combo", "mono")
            ),
            "columns_ok": all(tables[r]["columns_ok"] for r in ("combo", "mono")),
            "combo_is_ooxml_not_biff": next(
                e["is_ooxml"] for e in entries if e["role"] == "combo"
            ),
            "n_issues": sum(
                1
                for r in ("combo", "mono")
                for flag in (
                    tables[r]["row_count_match"],
                    tables[r]["columns_ok"],
                )
                if not flag
            ),
        },
    }
    manifest["audit"]["verdict"] = (
        "PASS"
        if hash_ok
        and manifest["audit"]["row_counts_ok"]
        and manifest["audit"]["columns_ok"]
        else "FAIL"
    )

    audit_dir = Path(cfg["paths"]["audit_dir"])
    write_json(manifest, audit_dir / "raw_file_manifest.json")
    (audit_dir / "raw_schema.md").write_text(render_schema_md(manifest), encoding="utf-8")
    logger.info("manifest -> %s", audit_dir / "raw_file_manifest.json")
    logger.info("schema   -> %s", audit_dir / "raw_schema.md")
    logger.info("M0 verdict: %s", manifest["audit"]["verdict"])

    if manifest["audit"]["verdict"] != "PASS":
        logger.error("M0 audit failed; refusing to continue")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
