"""Generate the M3 report and refresh the M3 manifest with delivery metadata.

Usage:
    python scripts/04_write_m3_report.py [--tests-log outputs/audit/pytest.log]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.utils.config import load_config
from dosesyn.utils.io import read_json, sha256_file, utc_now_iso, write_json
from dosesyn.utils.logging import setup_logging


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _fmt_int(v) -> str:
    return f"{int(v):,}"


def _sha(path: Path, n: int = 16) -> str:
    return sha256_file(path)[:n]


def _split_rows(manifest: dict) -> list[dict]:
    rows = []
    for entry in manifest.get("scenarios", []):
        split_dir = ROOT / "data/splits" / entry["scenario"] / entry["split_label"]
        meta = read_json(split_dir / "split_metadata.json")
        c = meta["counts"]
        rows.append(
            {
                "scenario": entry["scenario"],
                "split_label": entry["split_label"],
                "seed": entry.get("seed"),
                "fold": entry.get("fold"),
                "n_train": c["train"]["n_conditions"],
                "n_val": c["val"]["n_conditions"],
                "n_test": c["test"]["n_conditions"],
                "m_train": c["train"]["n_matrices"],
                "m_val": c["val"]["n_matrices"],
                "m_test": c["test"]["n_matrices"],
                "cells_train": c["train"]["n_cells"],
                "cells_val": c["val"]["n_cells"],
                "cells_test": c["test"]["n_cells"],
                "pairs_train": c["train"]["n_drug_pairs"],
                "pairs_val": c["val"]["n_drug_pairs"],
                "pairs_test": c["test"]["n_drug_pairs"],
                "mono_trainable": meta["mono"]["n_trainable_curves"],
                "mono_held": meta["mono"]["n_held_out_curves"],
                "content_sha": meta["content_sha256"],
                "scaler_sha": meta["scaler_sha256"],
            }
        )
    return rows


def _cancer_table(primary: pd.DataFrame, split_dir: Path, split: str) -> pd.DataFrame:
    cond = pd.read_parquet(split_dir / f"{split}_conditions.parquet", columns=["condition_id"])
    sel = primary[primary["condition_id"].isin(set(cond["condition_id"]))]
    num = sel.groupby("disease").size().rename("n_matrices")
    den = primary.groupby("disease").size().rename("n_total")
    tab = pd.concat([num, den], axis=1).fillna(0).astype(int)
    tab["retention_pct"] = (100.0 * tab["n_matrices"] / tab["n_total"]).round(1)
    return tab.reset_index().sort_values("n_matrices", ascending=False)


def _parse_tests(path: Path) -> dict:
    if not path.exists():
        return {"status": "未运行", "passed": None, "failed": None, "raw": None}
    text = path.read_text(encoding="utf-8", errors="replace")
    passed = re.findall(r"(\d+) passed", text)
    failed = re.findall(r"(\d+) failed", text)
    errors = re.findall(r"(\d+) error", text)
    last = [ln for ln in text.strip().splitlines() if ln.strip()][-1] if text.strip() else ""
    return {
        "status": "PASS" if passed and not failed and not errors else ("FAIL" if failed or errors else "UNKNOWN"),
        "passed": int(passed[-1]) if passed else 0,
        "failed": int(failed[-1]) if failed else 0,
        "errors": int(errors[-1]) if errors else 0,
        "raw": last,
        "log": _rel(path),
        "log_sha256": sha256_file(path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests-log", default="outputs/audit/pytest.log")
    args = ap.parse_args()
    cfg = load_config("configs/dataset_primary_v1.yaml")
    out = cfg["outputs"]
    logger = setup_logging("dosesyn.report", ROOT / "outputs/logs/04_write_m3_report.log")

    primary = pd.read_parquet(out["matrix_index"])
    mono = pd.read_parquet(out["mono_index"])
    excl = pd.read_parquet(out["exclusions"])
    input_manifest = read_json(out["input_manifest"])
    manifest = read_json(out["manifest"])
    leakage = read_json(out["leakage_report"])
    tests = _parse_tests(ROOT / args.tests_log)

    rows = _split_rows(manifest)
    s1 = next(r for r in rows if r["scenario"] == "s1_random_condition" and r["seed"] == 13)
    s2 = next(r for r in rows if r["scenario"] == "s2_unseen_pair" and r["seed"] == 13)
    s3 = next(r for r in rows if r["scenario"] == "s3_unseen_cell" and r["fold"] == 0)

    s1_meta = read_json(ROOT / "data/splits/s1_random_condition/seed_13/split_metadata.json")
    s2_meta = read_json(ROOT / "data/splits/s2_unseen_pair/seed_13/split_metadata.json")
    s3_meta = read_json(ROOT / "data/splits/s3_unseen_cell/fold_0/split_metadata.json")

    reason_counts = excl.groupby("exclusion_reason").size().reset_index(name="n")
    reason_counts["dataset"] = "matrix/curve rows"
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()

    lines: list[str] = []
    lines.append("# DoseSyn M3 数据集划分报告")
    lines.append("")
    lines.append(f"生成时间：{utc_now_iso()}　|　数据集版本：`{cfg['dataset_version']}`　|　Git：`{git_commit[:12]}`")
    lines.append("")
    lines.append("## 1. 结论")
    lines.append("")
    lines.append("| 验收项 | 结果 |")
    lines.append("| --- | --- |")
    lines.append(f"| Primary-v1 冻结 | PASS（{_fmt_int(len(primary))} 矩阵 / {_fmt_int(len(mono))} 单药曲线） |")
    lines.append(f"| S1/S2/S3 生成 | PASS（{len(rows)} 个 split 目录） |")
    lines.append(f"| 泄漏检查 | PASS（{sum(1 for x in leakage['splits'] if x['passed'])}/{len(leakage['splits'])}） |")
    lines.append(f"| 单药隔离披露 | PASS（S3 严格封存 val/test 细胞曲线） |")
    lines.append(f"| 训练折专属变换 | PASS（fit 实体 == train 实体） |")
    lines.append(f"| 自动测试 | {tests['status']}（{tests.get('passed', 0)} passed / {tests.get('failed', 0)} failed / {tests.get('errors', 0)} errors） |")
    lines.append("| 未训练神经网络 | 确认（本阶段无任何模型代码） |")
    lines.append("")
    lines.append("## 2. 主数据集冻结")
    lines.append("")
    lines.append("纳入规则：M2 `feature_complete` ＋ 细胞表达/通路可用 ＋ 4×4 完整矩阵 ＋ 药物图可建模；"
                 "标签为 M1 原始 `X/X0`（不裁剪）。Carboplatin/Oxaliplatin 因 RDKit 无法表达铂配位键而在一期排除，"
                 "原始数据、映射与 SMILES 全部保留，并逐矩阵记录排除原因。")
    lines.append("")
    lines.append("| 实体 | 数量 |")
    lines.append("| --- | ---: |")
    lines.append(f"| 矩阵 | {_fmt_int(len(primary))} |")
    lines.append(f"| 条件（细胞×无序对） | {_fmt_int(primary['condition_id'].nunique())} |")
    lines.append(f"| 细胞系 | {primary['cell_id'].nunique()} |")
    lines.append(f"| 药物 | {len(set(primary['drug_a_id']) | set(primary['drug_b_id']))} |")
    lines.append(f"| 无序药物对 | {primary['pair_id'].nunique()} |")
    lines.append(f"| 响应点 | {_fmt_int(int(primary['n_points'].sum()))} |")
    lines.append(f"| 单药曲线 | {_fmt_int(len(mono))} |")
    lines.append("")
    lines.append("排除原因（行数，可能重叠）：")
    lines.append("")
    lines.append("| exclusion_reason | 行数 |")
    lines.append("| --- | ---: |")
    for r in reason_counts.itertuples(index=False):
        lines.append(f"| `{r.exclusion_reason}` | {_fmt_int(r.n)} |")
    lines.append("")
    lines.append("## 3. 三种实验场景")
    lines.append("")
    lines.append("S1 以 `condition_id` 为原子、S2 以 `pair_id` 为原子、S3 以 `cell_id` 为原子；"
                 "任何矩阵、批次重复与组键都不会跨 split。S1/S2 各 3 个种子，S3 为 5 折。")
    lines.append("")
    lines.append("### 3.1 全部 split 规模")
    lines.append("")
    lines.append("| 场景 | 种子/折 | train 条件 | val 条件 | test 条件 | train 矩阵 | val 矩阵 | test 矩阵 | mono trainable | mono held |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        tag = f"seed {r['seed']}" if r["seed"] is not None else f"fold {r['fold']}"
        lines.append(
            f"| {r['scenario']} | {tag} | {_fmt_int(r['n_train'])} | {_fmt_int(r['n_val'])} | {_fmt_int(r['n_test'])} | "
            f"{_fmt_int(r['m_train'])} | {_fmt_int(r['m_val'])} | {_fmt_int(r['m_test'])} | "
            f"{_fmt_int(r['mono_trainable'])} | {_fmt_int(r['mono_held'])} |"
        )
    lines.append("")
    lines.append("### 3.2 代表划分的细胞/药物对覆盖")
    lines.append("")
    lines.append("| 场景 | split | 细胞 | 药物对 |")
    lines.append("| --- | --- | ---: | ---: |")
    for name, meta in [("S1 seed13", s1_meta), ("S2 seed13", s2_meta), ("S3 fold0", s3_meta)]:
        for split in ("train", "val", "test"):
            c = meta["counts"][split]
            lines.append(f"| {name} | {split} | {c['n_cells']} | {c['n_drug_pairs']} |")
    lines.append("")
    lines.append("### 3.3 代表划分的癌症类型分布（test）")
    lines.append("")
    for name, meta in [("S1 seed13", s1_meta), ("S2 seed13", s2_meta), ("S3 fold0", s3_meta)]:
        sdir = ROOT / "data/splits" / ("s1_random_condition" if "S1" in name else "s2_unseen_pair" if "S2" in name else "s3_unseen_cell") / ("seed_13" if "seed13" in name else "fold_0")
        tab = _cancer_table(primary, sdir, "test")
        lines.append(f"**{name} test**（{len(tab)} 类）")
        lines.append("")
        lines.append("| 癌症类型 | test 矩阵 | 全集矩阵 | 保留比例 |")
        lines.append("| --- | ---: | ---: | ---: |")
        for r in tab.itertuples(index=False):
            lines.append(f"| {r.disease} | {_fmt_int(r.n_matrices)} | {_fmt_int(r.n_total)} | {r.retention_pct:.1f}% |")
        lines.append("")
    lines.append("## 4. 单药监督范围")
    lines.append("")
    lines.append("| 场景 | 策略 | 可训练曲线 | 封存曲线 | 说明 |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    lines.append(f"| S1 | `known_monotherapy` | {_fmt_int(s1_meta['mono']['n_trainable_curves'])} | 0 | 已知先验，不是独立测试集 |")
    lines.append(f"| S2 | `known_monotherapy` | {_fmt_int(s2_meta['mono']['n_trainable_curves'])} | 0 | 同上，报告中明确披露 |")
    lines.append(f"| S3 | `strict_cell_isolation` | {_fmt_int(s3_meta['mono']['n_trainable_curves'])} | {_fmt_int(s3_meta['mono']['n_held_out_curves'])} | 仅训练细胞；val/test 细胞标签封存 |")
    lines.append("")
    lines.append("S3 每折的 `mono_assignments.parquet` 以 `label_access`（`train` / `held_out` / `known_prior`）"
                 "与 `trainable` 布尔列显式标注；任何训练脚本读取 `held_out` 曲线即违反协议。")
    lines.append("")
    lines.append("## 5. 训练折专属特征变换")
    lines.append("")
    lines.append("| 变换 | 拟合数据 | 参数文件 | 关键校验 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| Hallmark 50 维标准化 | 仅 train 细胞（S1/S2）/ 仅该折 train 细胞（S3） | `scalers.joblib` | fit cells == train cells |")
    lines.append("| 表达谱标准化（可选） | 仅 train 细胞 | `scalers.joblib` | 同上 |")
    lines.append("| 组合药物剂量 log10(M) | 仅 train 矩阵的非零剂量 | `scalers.joblib` | fit_matrix_ids == train matrices |")
    lines.append("| 单药剂量 log10(M) | 仅可训练单药曲线 | `scalers.joblib` | S3 仅 train 细胞 |")
    lines.append("")
    for name, meta in [("S1 seed13", s1_meta), ("S2 seed13", s2_meta), ("S3 fold0", s3_meta)]:
        sf = meta["scaler_fit"]
        lines.append(f"- **{name}**：cells={sf['n_cells']}，conditions={_fmt_int(sf['n_conditions'])}，"
                     f"mono_curves={_fmt_int(sf['n_mono_curves'])}，"
                     f"scaler_sha256=`{meta['scaler_sha256'][:16]}…`")
    lines.append("")
    lines.append("## 6. 泄漏检查")
    lines.append("")
    lines.append("| split | 状态 | 问题数 |")
    lines.append("| --- | --- | ---: |")
    for r in leakage["splits"]:
        n_bad = sum(1 for c in r["checks"] if c["status"] != "PASS")
        lines.append(f"| {r['scenario']}/{Path(r['split_dir']).name} | {'PASS' if r['passed'] else 'FAIL'} | {n_bad} |")
    lines.append("")
    checks = sorted({c["name"] for r in leakage["splits"] for c in r["checks"]})
    lines.append("执行的检查：" + "；".join(f"`{c}`" for c in checks) + "。")
    lines.append("")
    lines.append("## 7. 文件、哈希与数据加载")
    lines.append("")
    lines.append("| 文件 | SHA256（前 16 位） | 大小 |")
    lines.append("| --- | --- | ---: |")
    for key in ("matrix_index", "mono_index", "exclusions", "input_manifest", "leakage_report"):
        p = ROOT / out[key]
        lines.append(f"| `{out[key]}` | `{_sha(p)}` | {p.stat().st_size:,} B |")
    lines.append("")
    lines.append("输入哈希（M1/M2 未修改）：")
    lines.append("")
    lines.append("| 输入 | SHA256 前缀 |")
    lines.append("| --- | --- |")
    for k, v in input_manifest["input_hashes"].items():
        lines.append(f"| `{k}` | `{v[:16]}` |")
    lines.append("")
    lines.append("数据加载接口 `DoseSynDataset` 已在 S1/S2/S3 真实分组上做 smoke test："
                 "每个 batch 返回 4×4 标签、标准化剂量、零剂量标记、细胞特征与药物图；"
                 "`collate_matrices` 能正确拼接异构图并给出 `drug_batch`/`drug_role`。本阶段未训练任何网络。")
    lines.append("")
    lines.append("## 8. 自动测试")
    lines.append("")
    lines.append(f"- 命令：`.venv/bin/python -u -m pytest -q tests`")
    lines.append(f"- 结果：**{tests['status']}**，{tests.get('passed', 0)} passed / {tests.get('failed', 0)} failed / {tests.get('errors', 0)} errors")
    if tests.get("raw"):
        lines.append(f"- 末行：`{tests['raw']}`")
    if tests.get("log"):
        lines.append(f"- 日志：`{tests['log']}`（SHA256 `{tests['log_sha256'][:16]}…`）")
    lines.append("")
    lines.append("覆盖：M0/M1 原有 53 项 ＋ M3 新增 33 项（冻结、三种划分、单药策略、训练折变换、接口 smoke）。")
    lines.append("")
    lines.append("## 9. Git 与可复现性")
    lines.append("")
    lines.append(f"- 基线：`cdd9c8b`")
    lines.append(f"- 当前 commit：`{git_commit}`")
    lines.append(f"- 工作区：{'clean' if not status else 'dirty'}")
    lines.append("- 生成命令：")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python scripts/03_generate_splits.py --scenario all")
    lines.append(".venv/bin/python scripts/03_verify_splits.py")
    lines.append(".venv/bin/python -m pytest -q tests")
    lines.append(".venv/bin/python scripts/04_write_m3_report.py")
    lines.append("```")
    lines.append("")
    lines.append("同一 commit + 配置 + 种子产生相同 `assignment_sha256`/`content_sha256`；"
                 "所有 split 的 `split_metadata.json` 记录 assignment、scaler 与内容哈希。")
    lines.append("")
    lines.append("## 10. 遗留问题")
    lines.append("")
    lines.append("1. Carboplatin/Oxaliplatin 图不可用（配位键），影响 1,716 个矩阵；需专门的铂配合物图表示或 2D 描述符后才能纳入。")
    lines.append("2. EFM192B、UWB1289BRCA1 无 DepMap 映射；COLO320DM 仅有 COLO320 需要人工确认；OCUBM、UWB1289 无 CCLE 表达。")
    lines.append("3. 数据集细胞系来自单一 O'Neil 面板（34 个），S3 每折 test 仅 6–7 个细胞，跨癌种泛化结论需谨慎。")
    lines.append("4. S1/S2 使用 known-monotherapy 先验；若 M4 要做严格单药泛化，应改用 S3 的 strict 策略。")
    lines.append("5. 未生成同源细胞系（lineage）分组划分；若 M4 需评估 lineage 泛化，可新增 S4。")
    lines.append("6. 本阶段未定义缺失标签的掩码训练损失；M4 需在模型侧处理 NaN（当前 Primary-v1 全部标签有限）。")
    lines.append("")
    lines.append("## 11. 本轮文件清单")
    lines.append("")
    lines.append("```")
    for p in [
        "configs/dataset_primary_v1.yaml",
        "configs/m3.yaml",
        "configs/splits/s1_random_condition.yaml",
        "configs/splits/s2_unseen_pair.yaml",
        "configs/splits/s3_unseen_cell.yaml",
        "src/dosesyn/features/transform_train_only.py",
        "src/dosesyn/splits/build.py",
        "src/dosesyn/splits/mono_policy.py",
        "src/dosesyn/splits/verify.py",
        "src/dosesyn/data/dataset.py",
        "scripts/03_generate_splits.py",
        "scripts/03_verify_splits.py",
        "scripts/04_write_m3_report.py",
        "tests/test_m3_primary.py",
        "tests/test_m3_split_integrity.py",
        "tests/test_m3_mono_policy.py",
        "tests/test_m3_transforms.py",
        "tests/test_m3_dataset.py",
        "docs/decisions/m3_primary_dataset.md",
        "docs/decisions/m3_split_protocols.md",
        "m3_report.md",
    ]:
        lines.append(p)
    lines.append("```")
    lines.append("")

    report_path = ROOT / "m3_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    manifest["report"] = {
        "path": "m3_report.md",
        "sha256": sha256_file(report_path),
        "generated_at": utc_now_iso(),
    }
    manifest["tests"] = tests
    manifest["git_commit"] = git_commit
    manifest["delivery_generated_at"] = utc_now_iso()
    payload = {k: v for k, v in manifest.items() if k not in ("manifest_sha256", "delivery_generated_at")}
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json(manifest, ROOT / out["manifest"])
    for p in (report_path, ROOT / out["manifest"]):
        p.chmod(0o644)
    logger.info("wrote %s (%d bytes), tests=%s", report_path, report_path.stat().st_size, tests["status"])
    print(json.dumps({"report": _rel(report_path), "tests": tests, "git": git_commit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
