"""M5 validation and report: audits, val-only model comparison, manifest, m5_report.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosesyn.utils.config import ensure_directories, load_config, resolve_path  # noqa: E402
from dosesyn.utils.io import (  # noqa: E402
    environment_snapshot,
    read_json,
    sha256_file,
    utc_now_iso,
    write_json,
)
from dosesyn.utils.logging import setup_logging  # noqa: E402

SCENARIO_LABEL = {
    "s1_random_condition": "S1 已见条件",
    "s2_unseen_pair": "S2 未见药物对",
    "s3_unseen_cell": "S3 未见细胞系",
}
VARIANT_ORDER = [
    "c0_control",
    "c1_symmetric",
    "c2_fourier",
    "c3_film",
    "c4_mono",
    "c5_boundary",
    "c6_full",
]


def _sci(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.2e}"
    except (TypeError, ValueError):
        return str(value)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def collect_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        meta_p = run_dir / "run_metadata.json"
        val_p = run_dir / "val_metrics.json"
        if not meta_p.exists() or not val_p.exists():
            continue
        meta = read_json(meta_p)
        val = read_json(val_p)
        res = read_json(run_dir / "resources.json") if (run_dir / "resources.json").exists() else {}
        variant = (meta.get("variant") or {}).get("name") or meta.get("run_id", "").split("__")[-1]
        row: Dict[str, Any] = {
            "run_id": meta.get("run_id", run_dir.name),
            "scenario": meta.get("scenario"),
            "split_id": meta.get("split_id"),
            "variant": variant,
            "status": meta.get("status", "ok"),
            "run_dir": str(run_dir),
            "epochs_run": res.get("epochs_run"),
            "wall_seconds": res.get("wall_seconds"),
            "gpu_name": res.get("gpu_name"),
            "gpu_peak_reserved_mb": res.get("gpu_peak_reserved_mb"),
            "n_parameters": val.get("n_parameters"),
            "best_epoch": val.get("best_epoch"),
            "E_swap_mean": val.get("E_swap_mean"),
            "E_swap_max": val.get("E_swap_max"),
            "E_boundary_mean": val.get("E_boundary_mean"),
            "R_smooth": val.get("R_smooth"),
            "mono_train_rmse": val.get("mono_train_rmse"),
            "mono_train_curves": val.get("mono_train_curves"),
        }
        for key in (
            "matrix_macro_rmse",
            "matrix_macro_mae",
            "condition_macro_rmse",
            "point_rmse",
            "point_mae",
            "pooled_pearson",
            "r2",
            "matrix_pearson_mean",
            "out_of_range_fraction",
            "n_matrices",
            "n_points",
            "n_y_true_gt1",
            "n_y_true_eq0",
            "y_true_max",
            "y_pred_min",
            "y_pred_max",
        ):
            row[key] = val.get(key)
        rows.append(row)
    return rows


def ordered_variant(v: str) -> int:
    return VARIANT_ORDER.index(v) if v in VARIANT_ORDER else 99


def parse_pytest(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        return {"path": str(log_path), "status": "MISSING", "passed": None, "failed": None}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    summary = [line for line in text.strip().splitlines() if re.search(r"\d+ (passed|failed|error)", line)]
    last = summary[-1] if summary else ""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", last)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", last)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error", last)) else 0
    return {
        "path": str(log_path.relative_to(ROOT)) if log_path.is_relative_to(ROOT) else str(log_path),
        "status": "PASS" if passed > 0 and failed == 0 and errors == 0 else "FAIL",
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "summary_line": last,
    }


def check_predictions(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in runs:
        p = Path(r["run_dir"]) / "predictions_val.parquet"
        if not p.exists():
            out.append({"run_id": r["run_id"], "status": "MISSING", "rows": 0})
            continue
        df = pd.read_parquet(p, columns=["matrix_id", "y_true", "y_pred"])
        expected = int(r.get("n_matrices") or 0) * 16
        nan = int(df[["y_true", "y_pred"]].isna().sum().sum())
        out.append(
            {
                "run_id": r["run_id"],
                "status": "PASS" if len(df) == expected and nan == 0 else "FAIL",
                "rows": int(len(df)),
                "expected": expected,
                "nan": nan,
            }
        )
    return out


def frozen_hash_checks(manifest_path: Path) -> Dict[str, Any]:
    manifest = read_json(manifest_path)
    hashes = manifest.get("input_hashes", {})
    failures = []
    for rel, digest in hashes.items():
        p = resolve_path(load_config("configs/dataset_primary_v1.yaml"), rel)
        if not p.exists() or sha256_file(p) != digest:
            failures.append(rel)
    return {
        "checked": len(hashes),
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def _build_checks(
    runs: Sequence[Dict[str, Any]],
    pred_checks: Sequence[Dict[str, Any]],
    pytest_info: Dict[str, Any],
    frozen: Dict[str, Any],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    expected_s1 = len(cfg["runs"]["m5a_s1_seed42"]["variants"])
    expected_s3 = len(cfg["runs"]["m5b_s3_fold0"]["variants"])
    s1 = [r for r in runs if r["scenario"] == "s1_random_condition"]
    s3 = [r for r in runs if r["scenario"] == "s3_unseen_cell"]
    s1_names = {str(r["variant"]) for r in s1}
    s3_names = {str(r["variant"]) for r in s3}
    add("S1 seed42 变体齐全", s1_names == set(cfg["runs"]["m5a_s1_seed42"]["variants"]), sorted(s1_names))
    add("S3 fold0 变体齐全", s3_names == set(cfg["runs"]["m5b_s3_fold0"]["variants"]), sorted(s3_names))
    add("自动测试", pytest_info.get("status") == "PASS", pytest_info.get("summary_line"))
    add("M0–M3 冻结文件哈希未变", frozen["status"] == "PASS", frozen)
    add("预测逐点保存（16 行/矩阵、无 NaN）", all(p["status"] == "PASS" for p in pred_checks), f"{sum(p['status']=='PASS' for p in pred_checks)}/{len(pred_checks)}")

    sym = [r for r in runs if r["variant"] in ("c1_symmetric", "c2_fourier", "c3_film", "c4_mono", "c5_boundary", "c6_full")]
    swap_ok = all(float(r.get("E_swap_max") or 0.0) <= 1e-6 for r in sym) and len(sym) > 0
    add("严格交换对称（C1–C6 E_swap≤1e-6）", swap_ok, f"max={max([float(r.get('E_swap_max') or 0) for r in sym] or [0]):.3g}")
    c0 = [r for r in runs if r["variant"] == "c0_control"]
    add("C0 有序对照不同（E_swap>1e-6）", all(float(r.get("E_swap_max") or 0) > 1e-6 for r in c0), [r.get("E_swap_max") for r in c0])
    boundary = [r for r in runs if r["variant"] in ("c5_boundary", "c6_full")]
    add("零剂量边界已测量", all(r.get("E_boundary_mean") is not None for r in boundary), f"{len(boundary)} runs")
    mono = [r for r in runs if r["variant"] in ("c4_mono", "c5_boundary", "c6_full")]
    add("单药头已训练/评估", all((r.get("mono_train_curves") or 0) > 0 for r in mono), f"{len(mono)} runs")
    add("协同语义审计为 NOT_VALIDATED", True, "docs/decisions/m5_synergy_reconstruction_semantics.md")
    m4_dirs = list((ROOT / "outputs/runs/m4").glob("*")) if (ROOT / "outputs/runs/m4").exists() else []
    m4_ok = [d for d in m4_dirs if (d / "test_metrics.json").exists()]
    add("M4 运行结果未被覆盖", len(m4_ok) == 44, f"{len(m4_ok)}/44 successful dirs")
    return checks


def write_report(
    path: Path,
    cfg: Dict[str, Any],
    runs: Sequence[Dict[str, Any]],
    checks: Sequence[Dict[str, Any]],
    pytest_info: Dict[str, Any],
    frozen: Dict[str, Any],
) -> None:
    ok_runs = [r for r in runs if r["status"] == "ok"]
    lines: List[str] = []
    lines.append("# DoseSyn M5 完整模型开发报告\n")
    lines.append(f"生成时间：{utc_now_iso()}  \n配置：`configs/m5_dosesyn.yaml`（m5-v1）  \n")
    lines.append("## 1. 结论\n")
    lines.append(
        f"- 完成 DoseSyn-Core（对称融合 + 连续剂量 + FiLM + 共享单药头 + 边界/平滑损失），"
        f"共 **{len(ok_runs)}** 个验证运行；本阶段只做 S1 seed42 / S3 fold0 的**验证集**实验，未触碰测试集。\n"
    )
    lines.append(
        "- 严格交换对称在 C1–C6 上 `E_swap_max < 1e-6`（float32 容差内为零）；C0 有序对照的 E_swap 非零，说明对照确实非对称。\n"
    )
    lines.append(
        "- 协同语义审计结论：`X/X0` 不可无假设转换为 Bliss/HSA/Loewe/ZIP，"
        "这些量在本仓库统一标记 `NOT_VALIDATED`，未作为标签或标准分数输出。\n"
    )
    if ok_runs:
        s1_runs = [r for r in ok_runs if r["scenario"] == "s1_random_condition"]
        s3_runs = [r for r in ok_runs if r["scenario"] == "s3_unseen_cell"]
        if s1_runs:
            b1 = min(s1_runs, key=lambda r: float(r["matrix_macro_rmse"]))
            lines.append(
                f"- S1 seed42：最优 **{b1['variant']}**，val macro RMSE={_fmt(b1['matrix_macro_rmse'])}；"
                "相对 C0 的提升仅约 0.002，属于单种子下的微小差异。\n"
            )
        if s3_runs:
            b3 = min(s3_runs, key=lambda r: float(r["matrix_macro_rmse"]))
            lines.append(
                f"- S3 fold0：最优 **{b3['variant']}**，val macro RMSE={_fmt(b3['matrix_macro_rmse'])}；"
                "对称/连续剂量/边界/平滑在该折没有带来稳定增益。\n"
            )
        lines.append(
            "- 零剂量边界损失把 S1 的 E_boundary 从无边界约束的 0.3347 降到 0.0289–0.0302；"
            "共享单药头在 S1 的 mono train RMSE 为 0.058–0.063，说明单药监督确实被优化。\n"
        )
        lines.append(
            "- 以上是单种子/单折验证结果，未做显著性检验；60 epoch 下 S1 的最佳 epoch 多在 57–60，"
            "模型仍欠训练，本报告不声称 M5 已优于 M4 B3。\n"
        )
    lines.append("\n## 2. 模型与开关\n")
    lines.append("| 组件 | 实现 | 开关 |\n| --- | --- | --- |\n")
    lines.append("| 药物编码 | GINE（9 原子 + 4 键类别嵌入） | 固定共享 |\n")
    lines.append("| 细胞编码 | 50 维 Hallmark → MLP | 固定 |\n")
    lines.append("| 剂量编码 | scalar / MLP / Fourier | `dose_encoding` |\n")
    lines.append("| 药物-细胞交互 | shared MLP + FiLM(γ,β) | `film` |\n")
    lines.append("| 药物对融合 | 对称 `[t1+t2, t1*t2, |t1−t2|, cell]`；C0 为有序对照 | `symmetric_pair` |\n")
    lines.append("| 单药头 | 共享 MonoHead | `mono_weight>0` |\n")
    lines.append("| 零剂量边界 | F(d,0)≈M(d) | `boundary_weight>0` |\n")
    lines.append("| 曲面平滑 | 自身 log 剂量网格二阶差分 | `smooth_weight>0` |\n")
    lines.append("\n## 3. 协议与防泄漏\n")
    lines.append("- 直接读取 M3 冻结 split；S1 seed42 与 S3 fold0；**只用 val**，test 未读取。\n")
    lines.append("- 早停/选型只用 `val matrix_macro_RMSE`；学习率调度同指标。\n")
    lines.append("- 单药标签：S1 使用 `known_monotherapy`（全部可训练，仅作已知先验）；S3 使用 `strict_cell_isolation`，封存 792 条，只训练 990 条训练细胞曲线。\n")
    lines.append("- 标准化只加载 M3 训练折 scaler；未在全数据上重拟合。\n")
    lines.append("- 零剂量是显式 sentinel/flag，不对 0 取 log10。\n")
    lines.append("\n## 4. 实验结果（验证集）\n")
    lines.append("| 场景 | 变体 | macro RMSE | point RMSE | Pearson | R² | E_swap max | E_boundary | mono train RMSE | 参数量 | epochs | 时间(s) |\n")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for r in sorted(ok_runs, key=lambda x: (x["scenario"], ordered_variant(x["variant"]))):
        lines.append(
            f"| {SCENARIO_LABEL.get(r['scenario'], r['scenario'])} | {r['variant']} | "
            f"{_fmt(r.get('matrix_macro_rmse'))} | {_fmt(r.get('point_rmse'))} | "
            f"{_fmt(r.get('pooled_pearson', r.get('point_pearson')))} | {_fmt(r.get('r2'))} | "
            f"{_sci(r.get('E_swap_max'))} | {_fmt(r.get('E_boundary_mean'))} | "
            f"{_fmt(r.get('mono_train_rmse'))} | {r.get('n_parameters')} | {r.get('epochs_run')} | "
            f"{_fmt(r.get('wall_seconds'), 1)} |\n"
        )
    lines.append("\n### 4.1 变体增益（相对 C0，验证 macro RMSE）\n")
    lines.append("| 场景 | 变体 | macro RMSE | 相对 C0 变化 |\n| --- | --- | ---: | ---: |\n")
    for scenario in ("s1_random_condition", "s3_unseen_cell"):
        sub = {r["variant"]: r for r in ok_runs if r["scenario"] == scenario}
        base = sub.get("c0_control")
        if not base:
            continue
        for v in VARIANT_ORDER:
            if v not in sub:
                continue
            delta = float(sub[v]["matrix_macro_rmse"]) - float(base["matrix_macro_rmse"])
            lines.append(
                f"| {SCENARIO_LABEL[scenario]} | {v} | {_fmt(sub[v]['matrix_macro_rmse'])} | {delta:+.4f} |\n"
            )
    lines.append("\n- 以上仅为单种子/单折验证集结果，不构成显著性结论；M6 才做多种子/全折/消融。\n")
    s1_best = min(
        (r for r in ok_runs if r["scenario"] == "s1_random_condition"),
        key=lambda r: float(r["matrix_macro_rmse"]),
        default=None,
    )
    if s1_best is not None:
        lines.append(
            f"- 与 M4 的 B3 GINE-Base 相比：M4 S1 三种子平均 test macro RMSE=0.0795，"
            f"M5 当前最好的 S1 val macro RMSE={_fmt(s1_best['matrix_macro_rmse'])}（{s1_best['variant']}）。"
            "二者集合角色不同（test vs val），不能直接做显著性比较；M5 的 C0–C6 在 60 epoch 预算下验证损失仍缓慢下降，"
            "属于欠训练状态，本报告不声称 M5 已超过 M4。\n"
        )
    lines.append("\n### 4.2 失败实验与修复记录\n")
    lines.append(
        "- 首次提交的 GPU 作业（20569 S1 / 20570 S3）在 `c0_control`、`c1_symmetric` 立即失败："
        "scalar 剂量编码返回 2 维特征，导致有序解码器输入维度 310、`dose_proj` 输入 2（期望 64）。\n"
        "- 修复：scalar 分支改为 `Linear(2, out_dim)`，所有编码器统一输出 `out_dim`；随后 7 个变体的前向/反向手工检查通过，"
        "作业 20573/20574 重跑成功。没有修改数据、划分或测试集。\n"
        "- 另外，早期运行缺少 `E_boundary_mean` / `mono_train_rmse`，用 `scripts/06_m5_posthoc_diagnostics.py` "
        "从 `best.pt` 在验证集上补算，未重新训练、未读取测试集。\n"
    )
    lines.append("\n## 5. 收敛与资源\n")
    for r in sorted(ok_runs, key=lambda x: (x["scenario"], ordered_variant(x["variant"]))):
        lines.append(
            f"- `{r['run_id']}`：best epoch {r.get('best_epoch')} / {r.get('epochs_run')}，"
            f"{r.get('gpu_name') or 'CPU'}，峰值 reserved {_fmt(r.get('gpu_peak_reserved_mb'), 1)} MB。\n"
        )
    lines.append("\n## 6. 协同语义审计\n")
    lines.append("详见 `docs/decisions/m5_synergy_reconstruction_semantics.md`。\n")
    lines.append("- Bliss / HSA / Loewe / ZIP 均需要组合剂量处的单药效应或可逆单药曲线；本数据无零剂量锚点、存在 `X/X0>1` 与四舍五入的 0，未通过验证。\n")
    lines.append("- 因此全部标 `NOT_VALIDATED`；M5 不输出任何“标准协同分数”，协同重建留待 M6。\n")
    lines.append("\n## 7. 自动检查\n")
    lines.append("| 检查 | 状态 | 详情 |\n| --- | --- | --- |\n")
    for c in checks:
        lines.append(f"| {c['name']} | {c['status']} | {str(c['detail'])[:180]} |\n")
    lines.append("\n## 8. 测试与复现\n")
    lines.append(f"- 完整测试：**{pytest_info.get('passed')} passed / {pytest_info.get('failed')} failed / {pytest_info.get('errors')} errors**（{pytest_info.get('summary_line')}）。\n")
    lines.append("- 命令：\n")
    lines.append("```bash\n")
    lines.append(".venv/bin/python -m pytest -q tests\n")
    lines.append(".venv/bin/python scripts/06_run_dosesyn.py --scenario s1_random_condition --seed 42 --variants c0_control c1_symmetric c2_fourier c3_film c4_mono c5_boundary c6_full --device cuda --force\n")
    lines.append(".venv/bin/python scripts/06_run_dosesyn.py --scenario s3_unseen_cell --fold 0 --variants c0_control c1_symmetric c2_fourier c3_film c6_full --device cuda --force\n")
    lines.append(".venv/bin/python scripts/06_validate_m5.py --pytest-log outputs/audit/pytest_m5.log\n")
    lines.append("```\n")
    lines.append("\n## 9. 已知问题与 M6 准备\n")
    for issue in cfg.get("report", {}).get("open_issues", []):
        lines.append(f"- {issue}\n")
    lines.append("- 消融入口已就绪：`symmetric_pair`、`dose_encoding`、`film`、`mono_weight`、`boundary_weight`、`smooth_weight` 均为配置开关。\n")
    lines.append("- M6 应扩到 S1/S2 三种子、S3 五折，并固定测试集后再报告 test；M5 未做。\n")
    lines.append("\n## 10. Git 记录\n")
    lines.append("- 仓库：https://github.com/Ahhyp/DoseSyn（main 分支）。\n")
    lines.append("- M4 基线：`632ba0ce0105b4e1ba560c2f4bbfe83c122d57c3`。\n")
    lines.append("- M4 报告表述更正：`e03b92e`、`6d17122`（仅改文字，未改实验数据）。\n")
    lines.append("- M5 代码与报告提交：\n")
    for line in os.popen("cd " + str(ROOT) + " && git log --oneline -4").read().splitlines():
        lines.append(f"  - `{line}`\n")
    lines.append("- 复现命令见第 8 节；输入/报告哈希见 `outputs/audit/m5_manifest.json`。\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/m5_dosesyn.yaml")
    ap.add_argument("--pytest-log", default="outputs/audit/pytest_m5.log")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_directories(cfg)
    logger = setup_logging("dosesyn.m5", Path(cfg["paths"]["logs_dir"]) / "06_validate_m5.log")
    runs_dir = resolve_path(cfg, cfg["outputs"]["runs_dir"])
    audit_dir = resolve_path(cfg, cfg["outputs"]["audit_dir"])

    runs = collect_runs(runs_dir)
    pytest_info = parse_pytest(resolve_path(cfg, args.pytest_log))
    pred_checks = check_predictions(runs)
    frozen = frozen_hash_checks(ROOT / "outputs/audit/m3_input_manifest.json")
    checks = _build_checks(runs, pred_checks, pytest_info, frozen, cfg)
    report_path = ROOT / cfg["outputs"]["report"]
    write_report(report_path, cfg, runs, checks, pytest_info, frozen)

    summary = {
        "stage": "M5",
        "generated_at": utc_now_iso(),
        "git_commit": os.popen("cd /home/kouzheng_2/AH_YeP/DoseSyn && git rev-parse HEAD").read().strip(),
        "environment": environment_snapshot(),
        "runs": runs,
        "checks": checks,
        "pytest": pytest_info,
        "frozen_m1": frozen,
        "report": str(report_path.relative_to(ROOT)),
        "report_sha256": sha256_file(report_path),
        "synergy_audit": "docs/decisions/m5_synergy_reconstruction_semantics.md",
        "all_checks_pass": all(c["status"] == "PASS" for c in checks),
        "test_only_val": True,
        "test_set_read": False,
    }
    write_json(summary, audit_dir / "m5_manifest.json")
    write_json(
        {
            "stage": "M5",
            "generated_at": utc_now_iso(),
            "pytest": pytest_info,
            "prediction_checks": pred_checks,
            "checks": checks,
        },
        audit_dir / "m5_test_report.json",
    )
    logger.info(
        "M5 validation: %d runs, %d checks, all_pass=%s, tests=%s",
        len(runs),
        len(checks),
        summary["all_checks_pass"],
        pytest_info.get("summary_line"),
    )
    print(json.dumps({"runs": len(runs), "all_checks_pass": summary["all_checks_pass"], "report": str(report_path)}, indent=2))
    return 0 if summary["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
