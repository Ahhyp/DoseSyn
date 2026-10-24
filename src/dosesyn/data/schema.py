"""Column-level documentation and the generated data dictionary."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd

COLUMN_DESCRIPTIONS: Dict[str, str] = {
    # provenance / keys
    "source_index": "0-based row position in the published sheet (traceability)",
    "source_row": "1-based Excel row in the published sheet (header = row 1)",
    "batch_id": "Published BatchID: 1/2 = primary large screen, 3 = partial repeat",
    "cell_line": "Cell line name exactly as published",
    "drug_a_name": "Drug A name exactly as published (row axis of the matrix)",
    "drug_b_name": "Drug B name exactly as published (column axis of the matrix)",
    "drug_name": "Single-agent drug name exactly as published",
    "combination_name": "Published combination label (`drugA & drugB`)",
    "cell_id": "DoseSyn cell-line identifier (CELL_<SLUG>)",
    "drug_a_id": "DoseSyn drug identifier for drug A (DRUG_<SLUG>)",
    "drug_b_id": "DoseSyn drug identifier for drug B (DRUG_<SLUG>)",
    "drug_id": "DoseSyn drug identifier (DRUG_<SLUG>)",
    "pair_id": "Unordered drug-pair identifier (PAIR_<hash>)",
    "pair_drug_low_id": "Lexicographically lower drug ID of the unordered pair",
    "pair_drug_high_id": "Lexicographically higher drug ID of the unordered pair",
    "pair_name": "Pair label in canonical ID order",
    "condition_id": "Combination condition = pair x cell line (COND_<hash>)",
    "mono_condition_id": "Single-agent condition = drug x cell line (MCOND_<hash>)",
    "matrix_id": "4x4 matrix = batch x cell line x drug pair x dose grid (MTX_<hash>)",
    "mono_curve_id": "Single-agent curve = batch x cell x drug x dose grid (MONO_<hash>)",
    "row_idx": "Dense rank (0-based) of the dose within the unit",
    "col_idx": "Dense rank (0-based) of the drug B dose within the matrix",
    # concentrations
    "conc_a_source": "Drug A concentration exactly as published (source unit)",
    "conc_b_source": "Drug B concentration exactly as published (source unit)",
    "conc_source": "Single-agent concentration exactly as published (source unit)",
    "conc_a_um": "Drug A concentration converted to µM",
    "conc_b_um": "Drug B concentration converted to µM",
    "conc_um": "Single-agent concentration converted to µM",
    "log10_conc_a": "log10(conc_a_um); NaN only if concentration <= 0",
    "log10_conc_b": "log10(conc_b_um); NaN only if concentration <= 0",
    "log10_conc": "log10(conc_um); NaN only if concentration <= 0",
    "conc_a_um_is_nonpositive": "Flag: drug A concentration is <= 0 (none expected)",
    "conc_b_um_is_nonpositive": "Flag: drug B concentration is <= 0 (none expected)",
    "conc_um_is_nonpositive": "Flag: single-agent concentration is <= 0 (none expected)",
    "dose_low_um": "Dose of the lexicographically lower drug ID (pair-canonical orientation)",
    "dose_high_um": "Dose of the lexicographically higher drug ID",
    "log10_dose_low": "log10 of dose_low_um",
    "log10_dose_high": "log10 of dose_high_um",
    "is_drug_a_low_id": "True when drug A is the lexicographically lower ID",
    "grid_a_token": "Canonical sorted dose grid of drug A (µM, '|' separated)",
    "grid_b_token": "Canonical sorted dose grid of drug B (µM, '|' separated)",
    "grid_token": "Canonical sorted single-agent dose grid (µM, '|' separated)",
    # replicates
    "n_replicates_observed": "Number of published technical replicates (non-NaN)",
    "n_replicates_missing": "Number of missing technical-replicate slots",
    "has_all_replicates": "True when all nominal replicate slots are published",
    "viability_mean": "Mean viability across observed replicates (missing values skipped, not imputed)",
    "viability_std": "Sample SD (ddof=1) across observed replicates",
    "viability_min": "Minimum observed replicate viability",
    "viability_max": "Maximum observed replicate viability",
    "viability_spread": "max - min across observed replicates (technical variability)",
    "replicate_index": "Technical-replicate number (1-based)",
    "replicate_label": "Source column name (viability1 ... viabilityN)",
    "is_missing": "True when this replicate slot is NaN in the published data",
    # responses
    "mu_raw": "Published mu/muMax exactly as supplied (never recomputed in place)",
    "x_x0_raw": "Published X/X0 exactly as supplied (never clipped, never winsorised)",
    "x_x0_recalc": "Recomputed exp(ln(4)*(mu/muMax-1)) for auditing only",
    "response_formula_residual": "abs(x_x0_raw - x_x0_recalc)",
    "response_formula_ok": "Residual within the configured tolerance (1e-5)",
    "response_gt1": "Flag: published X/X0 > 1 (retained)",
    "response_gt2": "Flag: published X/X0 > 2.0 (retained and reported as outlier)",
    "response_lt0": "Flag: published X/X0 < 0 (none expected)",
    "response_eq0": "Flag: published X/X0 == 0 (formula floor; retained as published)",
    "response_isna": "Flag: published X/X0 is missing",
    "combination_name_matches_parts": "combination_name equals `drugA & drugB`",
    # metadata tables
    "n_points": "Number of dose coordinates in the unit",
    "n_dose_a": "Number of unique drug A doses",
    "n_dose_b": "Number of unique drug B doses",
    "n_unique_doses": "Number of unique single-agent doses",
    "n_unique_coords": "Number of unique (drug A dose, drug B dose) coordinates",
    "source_row_min": "First Excel row of the unit",
    "source_row_max": "Last Excel row of the unit",
    "source_rows_contiguous": "True when the unit occupies a contiguous block of source rows",
    "complete_matrix": "True for a full 4x4 grid with 16 unique coordinates",
    "complete_curve": "True for a full 8-point curve",
    "n_missing_response": "Dose coordinates with missing X/X0",
    "n_missing_replicates": "Total missing replicate values in the unit",
    "n_points_reduced_reps": "Coordinates with fewer than the nominal replicates",
    "total_missing_replicates": "Total missing replicate values in the curve",
    "x_x0_min": "Minimum published X/X0 in the unit",
    "x_x0_max": "Maximum published X/X0 in the unit",
    "x_x0_mean": "Mean published X/X0 in the unit (descriptive only)",
    "n_x_gt1": "Number of coordinates with X/X0 > 1",
    "n_x_gt2": "Number of coordinates with X/X0 > 2",
    "n_x_lt0": "Number of coordinates with X/X0 < 0",
    "max_replicate_spread": "Largest replicate max-min in the unit",
    "n_high_replicate_spread": "Coordinates whose replicate spread exceeds the threshold",
    "all_mu_present": "True when every coordinate has a published mu/muMax",
    "dose_a_um_json": "JSON list of drug A doses (µM)",
    "dose_b_um_json": "JSON list of drug B doses (µM)",
    "dose_low_um_json": "JSON list of pair-canonical low-ID drug doses (µM)",
    "dose_high_um_json": "JSON list of pair-canonical high-ID drug doses (µM)",
    "dose_um_json": "JSON list of single-agent doses (µM)",
    "batches": "Comma-separated batch IDs for this condition",
    "n_batches": "Number of batches in which this condition was measured",
    "grids_identical": "True when repeated batches used identical dose grids",
    "batch_detail_json": "Per-batch matrix/curve IDs, grids and response summaries",
    "x_mean_across_batches": "Mean response pooled across batches (descriptive only)",
}

for _i in range(1, 7):
    COLUMN_DESCRIPTIONS.setdefault(
        f"viability_{_i}",
        f"Technical replicate {_i} exactly as published (NaN when not published)",
    )

ENTITY_SCHEME = [
    ("Drug", "DRUG_<SLUG>", "name"),
    ("Cell line", "CELL_<SLUG>", "name"),
    ("Unordered pair", "PAIR_<sha1_12>", "sorted drug IDs"),
    ("Combination condition", "COND_<sha1_12>", "pair_id + cell_id"),
    ("Matrix", "MTX_<sha1_12>", "batch + cell + drugA + drugB + both dose grids"),
    ("Mono condition", "MCOND_<sha1_12>", "drug_id + cell_id"),
    ("Mono curve", "MONO_<sha1_12>", "batch + cell + drug + dose grid"),
]


def _table_schema(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "null_pct": round(100.0 * float(s.isna().mean()), 3),
                "description": COLUMN_DESCRIPTIONS.get(col, "—"),
            }
        )
    return rows


def render_data_dictionary(
    cfg: Dict[str, Any],
    manifest: Dict[str, Any],
    frames: Dict[str, pd.DataFrame],
    qc_summary: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# DoseSyn — O'Neil 数据集数据字典 (M1)")
    lines.append("")
    lines.append("本文档由 `scripts/01_prepare_oneil.py` 从实际产出自动生成；数值统计与 "
                 "`outputs/audit/oneil_qc.json` 同源。")
    lines.append("")
    lines.append("## 1. 数据来源")
    lines.append("")
    lines.append("O'Neil et al. (2016) 双药组合筛选补充数据，共三个 Excel 文件：")
    lines.append("")
    lines.append("| 角色 | 文件 | SHA256 | 真实格式 |")
    lines.append("| --- | --- | --- | --- |")
    for e in manifest["files"]:
        lines.append(
            f"| {e['role']} | `{e['filename']}` | `{e['sha256']}` | "
            f"{e['family']}/{e['container']} |"
        )
    lines.append("")
    lines.append("> **格式陷阱**：组合响应文件扩展名为 `.xls`，但文件头为 `504b0304`，"
                 "实际是 OOXML/ZIP。若按后缀交给 xlrd/BIFF 解析会失败或产生错误结果。"
                 "本工程将其字节级复制为 `.xlsx` 工作副本后由 openpyxl 读取，原始文件保持不变。")
    lines.append("")
    lines.append("## 2. 实体与 ID 规则")
    lines.append("")
    lines.append("| 实体 | ID 形式 | 生成依据 |")
    lines.append("| --- | --- | --- |")
    for name, form, basis in ENTITY_SCHEME:
        lines.append(f"| {name} | `{form}` | {basis} |")
    lines.append("")
    lines.append("所有 ID 均为内容的确定性函数（SHA1 前 12 位），与行顺序无关，重跑稳定。")
    lines.append("")
    lines.append("## 3. 表结构")
    lines.append("")
    counts = qc_summary["counts"]
    lines.append(f"- 双药响应点: **{counts['combo_points']:,}** 行 / "
                 f"**{counts['combo_matrices']:,}** 个 4×4 矩阵")
    lines.append(f"- 单药响应点: **{counts['mono_points']:,}** 行 / "
                 f"**{counts['mono_curves']:,}** 条 8 点曲线")
    lines.append(f"- 细胞系 **{counts['cell_lines']}**，药物 **{counts['drugs']}**，"
                 f"无序药物对 **{counts['combo_drug_pairs']}**")
    lines.append("")
    for table_name, df in frames.items():
        lines.append(f"### `{table_name}` ({len(df):,} rows × {df.shape[1]} cols)")
        lines.append("")
        lines.append("| 列 | dtype | 非空 | 空值% | 含义 |")
        lines.append("| --- | --- | ---: | ---: | --- |")
        for row in _table_schema(df):
            lines.append(
                f"| `{row['column']}` | {row['dtype']} | {row['non_null']:,} | "
                f"{row['null_pct']} | {row['description']} |"
            )
        lines.append("")
    lines.append("## 4. 单位与公式")
    lines.append("")
    lines.append("- 浓度单位：原始/目标均为 **µM**（转换因子 1.0，记录于配置 `units`）。")
    lines.append("- 生长率：`mu/muMax = ((LN(median viability) / (muMax*96)) + 1)`（原文定义）。")
    lines.append("- 响应：`X/X0 = EXP(LN(4)*((mu/muMax)-1))`；0 = 完全抑制，1 = 无效应。")
    lines.append("- 审计公式复算时残差仅来自源文件 5 位小数舍入，容差 1e-5。")
    lines.append("- `X/X0` **不做任何裁剪**：>1 与 >2 的值原样保留并单独标记。")
    lines.append("")
    lines.append("## 5. QC 阈值与理由")
    lines.append("")
    lines.append("| 阈值 | 值 | 理由 |")
    lines.append("| --- | ---: | --- |")
    qc = cfg["qc"]
    lines.append(f"| 每矩阵名义重复数 | {qc['nominal_combo_replicates']} | 图例：组合每板最多 4 个技术重复 |")
    lines.append(f"| 每单药曲线名义重复数 | {qc['nominal_mono_replicates']} | 图例：单药每板最多 6 个技术重复 |")
    lines.append(f"| X/X0 高异常阈值 | {qc['x_over_x0_upper_outlier']} | 仅标记，不裁剪 |")
    lines.append(f"| X/X0 低异常阈值 | {qc['x_over_x0_lower_outlier']} | 负响应 = 公式不可能值 |")
    lines.append(f"| 技术重复极差阈值 | {qc['replicate_spread_outlier']} | >1.0 的板内波动会主导建模信号 |")
    lines.append(f"| 公式复算容差 | {qc['response_formula_tolerance']} | 源数据 5 位小数舍入 |")
    lines.append(f"| 每轴剂量数 | {qc['combo_doses_per_axis']} | 4×4 矩阵 |")
    lines.append(f"| 每单药曲线剂量数 | {qc['mono_doses_per_curve']} | 8 点曲线 |")
    lines.append("")
    lines.append("## 6. 已知注意事项")
    lines.append("")
    lines.append("- **批次不可合并**：同一条件在 batch 3 中重复时保留为独立矩阵；"
                 f"共 {counts['batch_repeated_conditions']} 个条件跨批次，其中 "
                 f"{counts['batch_repeat_same_grid']} 个网格相同、"
                 f"{counts['batch_repeat_different_grid']} 个网格不同。")
    lines.append("- **缺失技术重复**：以 NaN 保留，不插补；均值仅由观测重复计算。")
    lines.append("- **X/X0 为 0**：源文件存在 213 个精确 0（公式下限约 3.8e-6），"
                 "按发布值保留并标记 `response_floor_zero`。")
    lines.append("- **M2 拆分约束**：同一 `condition_id` 的矩阵在任何划分中只能落入一个 split；"
                 "M1 未生成 split 文件，避免信息泄漏。")
    lines.append("")
    lines.append("## 7. 复现命令")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /home/kouzheng_2/AH_YeP/DoseSyn")
    lines.append(".venv/bin/python scripts/00_audit_raw.py")
    lines.append(".venv/bin/python scripts/01_prepare_oneil.py")
    lines.append(".venv/bin/python -m pytest -q")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)
