#!/usr/bin/env python
"""M1 — normalise the O'Neil data into analysis-ready tables.

Inputs (produced by ``scripts/00_audit_raw.py``):
  data/interim/raw_oneil/{combo,mono,legend}.parquet
  outputs/audit/{raw_file_manifest.json,oneil_entity_registry.json}

Outputs:
  data/processed/oneil_combo_points.parquet
  data/processed/oneil_combo_matrix_metadata.parquet
  data/processed/oneil_combo_replicates.parquet
  data/processed/oneil_monotherapy.parquet
  data/processed/oneil_mono_curve_metadata.parquet
  data/processed/oneil_mono_replicates.parquet
  outputs/audit/oneil_qc.json / .md, anomalies, batch repeats, entity index,
  sample matrices, pytest log
  docs/data_dictionary/oneil_datasets.md

Nothing is imputed, clipped, averaged across batches or split into train/test.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.data import qc  # noqa: E402
from dosesyn.data.make_matrices import (  # noqa: E402
    build_combo_matrices,
    build_mono_curves,
    combo_replicates,
    mono_replicates,
    validate_combo_structure,
    validate_mono_structure,
)
from dosesyn.data.parse_oneil import (  # noqa: E402
    load_raw_frames,
    standardize_combo,
    standardize_mono,
)
from dosesyn.data.schema import render_data_dictionary  # noqa: E402
from dosesyn.utils.config import ensure_directories, load_config  # noqa: E402
from dosesyn.utils.io import (  # noqa: E402
    environment_snapshot,
    read_json,
    utc_now_iso,
    write_json,
    write_parquet,
)
from dosesyn.utils.logging import setup_logging  # noqa: E402


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_pytest(cfg: Dict[str, Any], logger) -> Dict[str, Any]:
    """Run the M1 test suite, persist its log and return a summary."""
    audit_dir = Path(cfg["paths"]["audit_dir"])
    cmd = [sys.executable, "-m", "pytest", "-q", "tests"]
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    duration = round(time.time() - started, 1)
    log = (proc.stdout or "") + (proc.stderr or "")
    (audit_dir / "pytest.log").write_text(log, encoding="utf-8")
    passed = int(re.search(r"(\d+) passed", log).group(1)) if re.search(r"(\d+) passed", log) else 0
    failed = int(re.search(r"(\d+) failed", log).group(1)) if re.search(r"(\d+) failed", log) else 0
    errors = int(re.search(r"(\d+) error", log).group(1)) if re.search(r"(\d+) error", log) else 0
    logger.info(
        "pytest: rc=%d passed=%d failed=%d errors=%d in %.1fs",
        proc.returncode, passed, failed, errors, duration,
    )
    if proc.returncode != 0:
        logger.error("pytest failed; tail:\n%s", "\n".join(log.splitlines()[-40:]))
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "duration_s": duration,
        "log": str((audit_dir / "pytest.log").relative_to(ROOT)),
        "ok": proc.returncode == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DoseSyn M1 normalisation")
    parser.add_argument("--config", default="configs/data_oneil.yaml")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_directories(cfg)
    logger = setup_logging(
        "dosesyn.prepare", Path(cfg["paths"]["logs_dir"]) / "01_prepare_oneil.log"
    )
    logger.info("DoseSyn M1 starting; config=%s", cfg["_config_path"])

    audit_dir = Path(cfg["paths"]["audit_dir"])
    processed = Path(cfg["paths"]["processed_dir"])
    manifest = read_json(audit_dir / "raw_file_manifest.json")
    registry = read_json(audit_dir / "oneil_entity_registry.json")

    raw = load_raw_frames(cfg)
    logger.info("raw cache loaded: combo=%s mono=%s", raw["combo"].shape, raw["mono"].shape)

    qc_cfg = cfg["qc"]
    spread_threshold = float(qc_cfg["replicate_spread_outlier"])
    tol = float(qc_cfg["response_formula_tolerance"])
    upper = float(qc_cfg["x_over_x0_upper_outlier"])
    lower = float(qc_cfg["x_over_x0_lower_outlier"])

    combo_points = standardize_combo(
        raw["combo"], cfg, registry["cells"], registry["drugs"], logger
    )
    mono_points = standardize_mono(
        raw["mono"], cfg, registry["cells"], registry["drugs"], logger
    )
    logger.info("standardised: combo=%s mono=%s", combo_points.shape, mono_points.shape)

    combo_points, combo_meta = build_combo_matrices(
        combo_points, spread_threshold=spread_threshold
    )
    mono_points, mono_meta = build_mono_curves(
        mono_points, spread_threshold=spread_threshold
    )
    combo_struct = validate_combo_structure(combo_points)
    mono_struct = validate_mono_structure(mono_points)
    logger.info("combo structure: %s", combo_struct)
    logger.info("mono structure: %s", mono_struct)

    combo_rep = combo_replicates(
        combo_points, nominal=int(qc_cfg["nominal_combo_replicates"])
    )
    mono_rep = mono_replicates(
        mono_points, nominal=int(qc_cfg["nominal_mono_replicates"])
    )
    logger.info("replicate long tables: combo=%s mono=%s", combo_rep.shape, mono_rep.shape)

    anomalies = qc.build_anomalies(
        combo_points, combo_meta, mono_points, mono_meta, qc_cfg
    )
    anomalies = pd.concat(
        [anomalies, qc.missing_replicate_records(combo_rep, mono_rep)],
        ignore_index=True,
    ).sort_values(["severity", "anomaly_type", "source_row"], na_position="last")
    repeats = qc.combo_batch_repeats(combo_points)
    mono_repeats = qc.mono_batch_repeats(mono_points)
    entities = qc.entity_index(combo_points, mono_points, registry)
    logger.info(
        "anomalies=%d (types=%s) repeated conditions=%d mono repeats=%d entities=%d",
        len(anomalies), anomalies["anomaly_type"].value_counts().to_dict(),
        len(repeats), len(mono_repeats), len(entities),
    )

    # ---- processed tables -------------------------------------------------
    outputs: Dict[str, str] = {}
    combo_path = write_parquet(combo_points, processed / "oneil_combo_points.parquet")
    outputs[str(combo_path.relative_to(ROOT))] = "双药响应点（4×4 矩阵元素级，含原始重复与审计列）"
    meta_path = write_parquet(combo_meta, processed / "oneil_combo_matrix_metadata.parquet")
    outputs[str(meta_path.relative_to(ROOT))] = "双药 4×4 矩阵元数据（批次/细胞/药物对/剂量网格）"
    rep_path = write_parquet(combo_rep, processed / "oneil_combo_replicates.parquet")
    outputs[str(rep_path.relative_to(ROOT))] = "双药技术重复长表（缺失以 NaN 保留）"
    mono_path = write_parquet(mono_points, processed / "oneil_monotherapy.parquet")
    outputs[str(mono_path.relative_to(ROOT))] = "单药剂量-响应点（含原始重复与审计列）"
    mono_meta_path = write_parquet(mono_meta, processed / "oneil_mono_curve_metadata.parquet")
    outputs[str(mono_meta_path.relative_to(ROOT))] = "单药曲线元数据"
    mono_rep_path = write_parquet(mono_rep, processed / "oneil_mono_replicates.parquet")
    outputs[str(mono_rep_path.relative_to(ROOT))] = "单药技术重复长表"

    # ---- audit artefacts --------------------------------------------------
    anom_path = audit_dir / "oneil_anomalies.csv"
    _write_csv(anomalies, anom_path)
    outputs[str(anom_path.relative_to(ROOT))] = "全部异常/标记逐条清单"
    by_type_dir = audit_dir / "anomalies_by_type"
    for name, grp in anomalies.groupby("anomaly_type", sort=True):
        p = by_type_dir / f"{name}.csv"
        _write_csv(grp, p)
        outputs[str(p.relative_to(ROOT))] = f"异常类型 {name}"
    _write_csv(
        anomalies.groupby(["anomaly_type", "severity"]).size().rename("count").reset_index(),
        audit_dir / "anomaly_summary.csv",
    )
    _write_csv(repeats, audit_dir / "batch_repeats.csv")
    outputs[str((audit_dir / "batch_repeats.csv").relative_to(ROOT))] = "跨批次重复的双药条件（315）"
    _write_csv(mono_repeats, audit_dir / "mono_batch_repeats.csv")
    outputs[str((audit_dir / "mono_batch_repeats.csv").relative_to(ROOT))] = "跨批次重复的单药条件"
    _write_csv(entities, audit_dir / "oneil_entity_index.csv")
    outputs[str((audit_dir / "oneil_entity_index.csv").relative_to(ROOT))] = "药物/细胞系/药对/条件/矩阵/曲线索引"

    sample_manifest = qc.write_sample_matrices(
        combo_points, combo_meta, repeats, audit_dir / "sample_matrices"
    )
    logger.info("sample matrices written: %s", [s["matrix_id"] for s in sample_manifest])
    outputs[str((audit_dir / "sample_matrices").relative_to(ROOT))] = "5 个代表性矩阵的原始 4×4 CSV 与逐点明细"

    checks = qc.build_checks(
        cfg, manifest["files"], manifest.get("raw", {}),
        combo_points, combo_meta, mono_points, mono_meta,
        repeats, combo_struct, mono_struct,
    )

    tests: Dict[str, Any] = {"ok": None, "note": "skipped by --skip-tests"}
    if not args.skip_tests:
        tests = run_pytest(cfg, logger)
        checks.append(
            qc._check(
                "pytest_suite_passes",
                "PASS" if tests["ok"] else "FAIL",
                {
                    "passed": tests["passed"],
                    "failed": tests["failed"],
                    "errors": tests["errors"],
                    "log": tests["log"],
                },
                "0 failed / 0 errors",
                "Automated integrity, uniqueness, conversion and traceability tests.",
            )
        )

    summary = qc.build_qc_summary(
        cfg, environment_snapshot(), manifest["files"], manifest.get("raw", {}),
        combo_points, combo_meta, mono_points, mono_meta,
        anomalies, repeats, checks, utc_now_iso(),
    )
    summary["tests"] = tests
    summary["sample_matrices"] = sample_manifest
    summary["matrix_structure"] = {"combo": combo_struct, "mono": mono_struct}
    summary["outputs"] = outputs
    summary["open_issues"] = [
        "组合文件后缀 `.xls` 与实际 OOXML 格式不符：已用字节级 `.xlsx` 工作副本读取，原文件未改；后续外部工具若按后缀解析会失败。",
        "7 个 X/X0 > 2 的响应点（6 个 T47D、1 个 ZR751）已保留并标记；M2 应对其做敏感性分析，不得静默裁剪。",
        "213 个 X/X0 精确为 0 的点已按发布值保留；使用 log(X/X0) 或比值类特征时需加偏移或改用原始 mu。",
        "315 个条件跨批次重复，其中 210 个剂量网格不同；批次不可被平均合并，M2 应考虑批次分层或批次作协变量。",
        "缺失技术重复值：双药 6018 个、单药 15600 个，全部保留为 NaN，未插补。",
        "源数据所有浓度均 > 0，没有真正的零剂量/vehicle 锚点；LG0 或协同零基线需在 M2 另行定义，不能从本表直接得到。",
        "Batch 3 只是部分重复验证，不是完整独立重复；不能把 Batch 3 当作与其他批次等价的随机样本。",
        "M1 未生成 train/val/test 划分：按 condition_id 分组拆分留待 M2，避免同条件泄漏。",
    ]

    outputs["outputs/audit/raw_schema.md"] = "M0 原始字段、行数、缺失与容器审计"
    outputs["outputs/audit/raw_file_manifest.json"] = "M0 输入文件哈希、真实容器与对账清单"
    outputs["outputs/audit/oneil_entity_registry.json"] = "细胞系/药物/无序药物对注册表"
    outputs["outputs/audit/sample_matrices/sample_matrices_manifest.json"] = "5 个示例矩阵的选取理由与文件索引"
    write_json(summary, audit_dir / "oneil_qc.json")
    (audit_dir / "oneil_qc.md").write_text(
        qc.render_qc_markdown(summary), encoding="utf-8"
    )
    outputs[str((audit_dir / "oneil_qc.json").relative_to(ROOT))] = "机器可读 QC 汇总（含验收结论）"
    outputs[str((audit_dir / "oneil_qc.md").relative_to(ROOT))] = "人类可读 QC 报告"
    outputs["outputs/logs/00_audit_raw.log"] = "M0 审计日志"
    outputs["outputs/logs/01_prepare_oneil.log"] = "M1 处理日志"

    frames_for_docs = {
        "oneil_combo_points.parquet": combo_points,
        "oneil_combo_matrix_metadata.parquet": combo_meta,
        "oneil_combo_replicates.parquet": combo_rep,
        "oneil_monotherapy.parquet": mono_points,
        "oneil_mono_curve_metadata.parquet": mono_meta,
        "oneil_mono_replicates.parquet": mono_rep,
    }
    docs_dir = Path(cfg["paths"]["docs_dir"]) / "data_dictionary"
    docs_dir.mkdir(parents=True, exist_ok=True)
    dict_path = docs_dir / "oneil_datasets.md"
    dict_path.write_text(
        render_data_dictionary(cfg, manifest, frames_for_docs, summary), encoding="utf-8"
    )
    outputs[str(dict_path.relative_to(ROOT))] = "数据字典（自动生成）"
    # Re-write QC JSON/MD so the output index is complete.
    write_json(summary, audit_dir / "oneil_qc.json")
    (audit_dir / "oneil_qc.md").write_text(qc.render_qc_markdown(summary), encoding="utf-8")

    logger.info("M1 verdict: m2_ready=%s", summary["acceptance"]["m2_ready"])
    print(json.dumps({
        "m2_ready": summary["acceptance"]["m2_ready"],
        "blocking_failures": summary["acceptance"]["blocking_failures"],
        "counts": summary["counts"],
        "anomalies": summary["anomalies"],
        "tests": tests,
        "qc_json": str(audit_dir / "oneil_qc.json"),
    }, ensure_ascii=False, indent=2))
    return 0 if summary["acceptance"]["m2_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
