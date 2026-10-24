# DoseSyn — O'Neil 数据集数据字典 (M1)

本文档由 `scripts/01_prepare_oneil.py` 从实际产出自动生成；数值统计与 `outputs/audit/oneil_qc.json` 同源。

## 1. 数据来源

O'Neil et al. (2016) 双药组合筛选补充数据，共三个 Excel 文件：

| 角色 | 文件 | SHA256 | 真实格式 |
| --- | --- | --- | --- |
| combo | `15357163mct150843-sup-156849_1_supp_1_w2lrww.xls` | `a0e2e002dc06c34e8aedf3206877a93bab7e45190ad8bb8e70ed79ff24c14818` | ooxml/zip |
| mono | `15357163mct150843-sup-156849_1_supp_0_w2lh45.xlsx` | `4a0622c074b9c505661fb6e5edf8dbbb018f1be33b2bec8058fb7d933ed66dec` | ooxml/zip |
| legend | `15357163mct150843-sup-156849_1_supp_1_g2lr29.xlsx` | `4a85ebffc9a3c94b7cb3c0f26a8d90accb363282913bfbbf708e56cb6c6b3d6b` | ooxml/zip |

> **格式陷阱**：组合响应文件扩展名为 `.xls`，但文件头为 `504b0304`，实际是 OOXML/ZIP。若按后缀交给 xlrd/BIFF 解析会失败或产生错误结果。本工程将其字节级复制为 `.xlsx` 工作副本后由 openpyxl 读取，原始文件保持不变。

## 2. 实体与 ID 规则

| 实体 | ID 形式 | 生成依据 |
| --- | --- | --- |
| Drug | `DRUG_<SLUG>` | name |
| Cell line | `CELL_<SLUG>` | name |
| Unordered pair | `PAIR_<sha1_12>` | sorted drug IDs |
| Combination condition | `COND_<sha1_12>` | pair_id + cell_id |
| Matrix | `MTX_<sha1_12>` | batch + cell + drugA + drugB + both dose grids |
| Mono condition | `MCOND_<sha1_12>` | drug_id + cell_id |
| Mono curve | `MONO_<sha1_12>` | batch + cell + drug + dose grid |

所有 ID 均为内容的确定性函数（SHA1 前 12 位），与行顺序无关，重跑稳定。

## 3. 表结构

- 双药响应点: **368,832** 行 / **23,052** 个 4×4 矩阵
- 单药响应点: **16,752** 行 / **2,094** 条 8 点曲线
- 细胞系 **39**，药物 **38**，无序药物对 **583**

### `oneil_combo_points.parquet` (368,832 rows × 56 cols)

| 列 | dtype | 非空 | 空值% | 含义 |
| --- | --- | ---: | ---: | --- |
| `source_index` | int64 | 368,832 | 0.0 | 0-based row position in the published sheet (traceability) |
| `source_row` | int64 | 368,832 | 0.0 | 1-based Excel row in the published sheet (header = row 1) |
| `batch_id` | int64 | 368,832 | 0.0 | Published BatchID: 1/2 = primary large screen, 3 = partial repeat |
| `cell_line` | object | 368,832 | 0.0 | Cell line name exactly as published |
| `drug_a_name` | object | 368,832 | 0.0 | Drug A name exactly as published (row axis of the matrix) |
| `drug_b_name` | object | 368,832 | 0.0 | Drug B name exactly as published (column axis of the matrix) |
| `combination_name` | object | 368,832 | 0.0 | Published combination label (`drugA & drugB`) |
| `cell_id` | object | 368,832 | 0.0 | DoseSyn cell-line identifier (CELL_<SLUG>) |
| `drug_a_id` | object | 368,832 | 0.0 | DoseSyn drug identifier for drug A (DRUG_<SLUG>) |
| `drug_b_id` | object | 368,832 | 0.0 | DoseSyn drug identifier for drug B (DRUG_<SLUG>) |
| `conc_a_source` | float64 | 368,832 | 0.0 | Drug A concentration exactly as published (source unit) |
| `conc_b_source` | float64 | 368,832 | 0.0 | Drug B concentration exactly as published (source unit) |
| `conc_a_um` | float64 | 368,832 | 0.0 | Drug A concentration converted to µM |
| `conc_b_um` | float64 | 368,832 | 0.0 | Drug B concentration converted to µM |
| `log10_conc_a` | float64 | 368,832 | 0.0 | log10(conc_a_um); NaN only if concentration <= 0 |
| `log10_conc_b` | float64 | 368,832 | 0.0 | log10(conc_b_um); NaN only if concentration <= 0 |
| `conc_a_um_is_nonpositive` | bool | 368,832 | 0.0 | Flag: drug A concentration is <= 0 (none expected) |
| `conc_b_um_is_nonpositive` | bool | 368,832 | 0.0 | Flag: drug B concentration is <= 0 (none expected) |
| `viability_1` | float64 | 368,832 | 0.0 | Technical replicate 1 exactly as published (NaN when not published) |
| `viability_2` | float64 | 368,754 | 0.021 | Technical replicate 2 exactly as published (NaN when not published) |
| `viability_3` | float64 | 368,634 | 0.054 | Technical replicate 3 exactly as published (NaN when not published) |
| `viability_4` | float64 | 363,090 | 1.557 | Technical replicate 4 exactly as published (NaN when not published) |
| `n_replicates_observed` | int16 | 368,832 | 0.0 | Number of published technical replicates (non-NaN) |
| `n_replicates_missing` | int16 | 368,832 | 0.0 | Number of missing technical-replicate slots |
| `viability_mean` | float64 | 368,832 | 0.0 | Mean viability across observed replicates (missing values skipped, not imputed) |
| `viability_std` | float64 | 368,754 | 0.021 | Sample SD (ddof=1) across observed replicates |
| `viability_min` | float64 | 368,832 | 0.0 | Minimum observed replicate viability |
| `viability_max` | float64 | 368,832 | 0.0 | Maximum observed replicate viability |
| `viability_spread` | float64 | 368,832 | 0.0 | max - min across observed replicates (technical variability) |
| `has_all_replicates` | bool | 368,832 | 0.0 | True when all nominal replicate slots are published |
| `mu_raw` | float64 | 368,832 | 0.0 | Published mu/muMax exactly as supplied (never recomputed in place) |
| `x_x0_raw` | float64 | 368,832 | 0.0 | Published X/X0 exactly as supplied (never clipped, never winsorised) |
| `x_x0_recalc` | float64 | 368,832 | 0.0 | Recomputed exp(ln(4)*(mu/muMax-1)) for auditing only |
| `response_formula_residual` | float64 | 368,832 | 0.0 | abs(x_x0_raw - x_x0_recalc) |
| `response_formula_ok` | bool | 368,832 | 0.0 | Residual within the configured tolerance (1e-5) |
| `response_gt1` | bool | 368,832 | 0.0 | Flag: published X/X0 > 1 (retained) |
| `response_gt2` | bool | 368,832 | 0.0 | Flag: published X/X0 > 2.0 (retained and reported as outlier) |
| `response_lt0` | bool | 368,832 | 0.0 | Flag: published X/X0 < 0 (none expected) |
| `response_eq0` | bool | 368,832 | 0.0 | Flag: published X/X0 == 0 (formula floor; retained as published) |
| `response_isna` | bool | 368,832 | 0.0 | Flag: published X/X0 is missing |
| `combination_name_matches_parts` | bool | 368,832 | 0.0 | combination_name equals `drugA & drugB` |
| `pair_id` | object | 368,832 | 0.0 | Unordered drug-pair identifier (PAIR_<hash>) |
| `pair_drug_low_id` | object | 368,832 | 0.0 | Lexicographically lower drug ID of the unordered pair |
| `pair_drug_high_id` | object | 368,832 | 0.0 | Lexicographically higher drug ID of the unordered pair |
| `is_drug_a_low_id` | bool | 368,832 | 0.0 | True when drug A is the lexicographically lower ID |
| `dose_low_um` | float64 | 368,832 | 0.0 | Dose of the lexicographically lower drug ID (pair-canonical orientation) |
| `dose_high_um` | float64 | 368,832 | 0.0 | Dose of the lexicographically higher drug ID |
| `log10_dose_low` | float64 | 368,832 | 0.0 | log10 of dose_low_um |
| `log10_dose_high` | float64 | 368,832 | 0.0 | log10 of dose_high_um |
| `pair_name` | object | 368,832 | 0.0 | Pair label in canonical ID order |
| `grid_a_token` | object | 368,832 | 0.0 | Canonical sorted dose grid of drug A (µM, '|' separated) |
| `grid_b_token` | object | 368,832 | 0.0 | Canonical sorted dose grid of drug B (µM, '|' separated) |
| `matrix_id` | object | 368,832 | 0.0 | 4x4 matrix = batch x cell line x drug pair x dose grid (MTX_<hash>) |
| `condition_id` | object | 368,832 | 0.0 | Combination condition = pair x cell line (COND_<hash>) |
| `row_idx` | int16 | 368,832 | 0.0 | Dense rank (0-based) of the dose within the unit |
| `col_idx` | int16 | 368,832 | 0.0 | Dense rank (0-based) of the drug B dose within the matrix |

### `oneil_combo_matrix_metadata.parquet` (23,052 rows × 35 cols)

| 列 | dtype | 非空 | 空值% | 含义 |
| --- | --- | ---: | ---: | --- |
| `matrix_id` | object | 23,052 | 0.0 | 4x4 matrix = batch x cell line x drug pair x dose grid (MTX_<hash>) |
| `batch_id` | int64 | 23,052 | 0.0 | Published BatchID: 1/2 = primary large screen, 3 = partial repeat |
| `cell_line` | object | 23,052 | 0.0 | Cell line name exactly as published |
| `cell_id` | object | 23,052 | 0.0 | DoseSyn cell-line identifier (CELL_<SLUG>) |
| `pair_id` | object | 23,052 | 0.0 | Unordered drug-pair identifier (PAIR_<hash>) |
| `drug_a_id` | object | 23,052 | 0.0 | DoseSyn drug identifier for drug A (DRUG_<SLUG>) |
| `drug_a_name` | object | 23,052 | 0.0 | Drug A name exactly as published (row axis of the matrix) |
| `drug_b_id` | object | 23,052 | 0.0 | DoseSyn drug identifier for drug B (DRUG_<SLUG>) |
| `drug_b_name` | object | 23,052 | 0.0 | Drug B name exactly as published (column axis of the matrix) |
| `combination_name` | object | 23,052 | 0.0 | Published combination label (`drugA & drugB`) |
| `condition_id` | object | 23,052 | 0.0 | Combination condition = pair x cell line (COND_<hash>) |
| `n_points` | int64 | 23,052 | 0.0 | Number of dose coordinates in the unit |
| `n_dose_a` | int64 | 23,052 | 0.0 | Number of unique drug A doses |
| `n_dose_b` | int64 | 23,052 | 0.0 | Number of unique drug B doses |
| `source_row_min` | int64 | 23,052 | 0.0 | First Excel row of the unit |
| `source_row_max` | int64 | 23,052 | 0.0 | Last Excel row of the unit |
| `n_missing_response` | int64 | 23,052 | 0.0 | Dose coordinates with missing X/X0 |
| `n_missing_replicates` | int16 | 23,052 | 0.0 | Total missing replicate values in the unit |
| `n_points_reduced_reps` | int64 | 23,052 | 0.0 | Coordinates with fewer than the nominal replicates |
| `x_x0_min` | float64 | 23,052 | 0.0 | Minimum published X/X0 in the unit |
| `x_x0_max` | float64 | 23,052 | 0.0 | Maximum published X/X0 in the unit |
| `x_x0_mean` | float64 | 23,052 | 0.0 | Mean published X/X0 in the unit (descriptive only) |
| `n_x_gt1` | int64 | 23,052 | 0.0 | Number of coordinates with X/X0 > 1 |
| `n_x_gt2` | int64 | 23,052 | 0.0 | Number of coordinates with X/X0 > 2 |
| `n_x_lt0` | int64 | 23,052 | 0.0 | Number of coordinates with X/X0 < 0 |
| `max_replicate_spread` | float64 | 23,052 | 0.0 | Largest replicate max-min in the unit |
| `n_high_replicate_spread` | int64 | 23,052 | 0.0 | Coordinates whose replicate spread exceeds the threshold |
| `all_mu_present` | bool | 23,052 | 0.0 | True when every coordinate has a published mu/muMax |
| `n_unique_coords` | int64 | 23,052 | 0.0 | Number of unique (drug A dose, drug B dose) coordinates |
| `dose_a_um_json` | object | 23,052 | 0.0 | JSON list of drug A doses (µM) |
| `dose_b_um_json` | object | 23,052 | 0.0 | JSON list of drug B doses (µM) |
| `dose_low_um_json` | object | 23,052 | 0.0 | JSON list of pair-canonical low-ID drug doses (µM) |
| `dose_high_um_json` | object | 23,052 | 0.0 | JSON list of pair-canonical high-ID drug doses (µM) |
| `source_rows_contiguous` | bool | 23,052 | 0.0 | True when the unit occupies a contiguous block of source rows |
| `complete_matrix` | bool | 23,052 | 0.0 | True for a full 4x4 grid with 16 unique coordinates |

### `oneil_combo_replicates.parquet` (1,475,328 rows × 12 cols)

| 列 | dtype | 非空 | 空值% | 含义 |
| --- | --- | ---: | ---: | --- |
| `matrix_id` | object | 1,475,328 | 0.0 | 4x4 matrix = batch x cell line x drug pair x dose grid (MTX_<hash>) |
| `condition_id` | object | 1,475,328 | 0.0 | Combination condition = pair x cell line (COND_<hash>) |
| `source_row` | int64 | 1,475,328 | 0.0 | 1-based Excel row in the published sheet (header = row 1) |
| `row_idx` | int16 | 1,475,328 | 0.0 | Dense rank (0-based) of the dose within the unit |
| `col_idx` | int16 | 1,475,328 | 0.0 | Dense rank (0-based) of the drug B dose within the matrix |
| `conc_a_um` | float64 | 1,475,328 | 0.0 | Drug A concentration converted to µM |
| `conc_b_um` | float64 | 1,475,328 | 0.0 | Drug B concentration converted to µM |
| `x_x0_raw` | float64 | 1,475,328 | 0.0 | Published X/X0 exactly as supplied (never clipped, never winsorised) |
| `replicate_label` | object | 1,475,328 | 0.0 | Source column name (viability1 ... viabilityN) |
| `viability` | float64 | 1,469,310 | 0.408 | — |
| `replicate_index` | int16 | 1,475,328 | 0.0 | Technical-replicate number (1-based) |
| `is_missing` | bool | 1,475,328 | 0.0 | True when this replicate slot is NaN in the published data |

### `oneil_monotherapy.parquet` (16,752 rows × 39 cols)

| 列 | dtype | 非空 | 空值% | 含义 |
| --- | --- | ---: | ---: | --- |
| `source_index` | int64 | 16,752 | 0.0 | 0-based row position in the published sheet (traceability) |
| `source_row` | int64 | 16,752 | 0.0 | 1-based Excel row in the published sheet (header = row 1) |
| `batch_id` | int64 | 16,752 | 0.0 | Published BatchID: 1/2 = primary large screen, 3 = partial repeat |
| `cell_line` | object | 16,752 | 0.0 | Cell line name exactly as published |
| `drug_name` | object | 16,752 | 0.0 | Single-agent drug name exactly as published |
| `cell_id` | object | 16,752 | 0.0 | DoseSyn cell-line identifier (CELL_<SLUG>) |
| `drug_id` | object | 16,752 | 0.0 | DoseSyn drug identifier (DRUG_<SLUG>) |
| `conc_source` | float64 | 16,752 | 0.0 | Single-agent concentration exactly as published (source unit) |
| `conc_um` | float64 | 16,752 | 0.0 | Single-agent concentration converted to µM |
| `log10_conc` | float64 | 16,752 | 0.0 | log10(conc_um); NaN only if concentration <= 0 |
| `conc_um_is_nonpositive` | bool | 16,752 | 0.0 | Flag: single-agent concentration is <= 0 (none expected) |
| `viability_1` | float64 | 16,752 | 0.0 | Technical replicate 1 exactly as published (NaN when not published) |
| `viability_2` | float64 | 16,752 | 0.0 | Technical replicate 2 exactly as published (NaN when not published) |
| `viability_3` | float64 | 16,752 | 0.0 | Technical replicate 3 exactly as published (NaN when not published) |
| `viability_4` | float64 | 11,552 | 31.041 | Technical replicate 4 exactly as published (NaN when not published) |
| `viability_5` | float64 | 11,552 | 31.041 | Technical replicate 5 exactly as published (NaN when not published) |
| `viability_6` | float64 | 11,552 | 31.041 | Technical replicate 6 exactly as published (NaN when not published) |
| `n_replicates_observed` | int16 | 16,752 | 0.0 | Number of published technical replicates (non-NaN) |
| `n_replicates_missing` | int16 | 16,752 | 0.0 | Number of missing technical-replicate slots |
| `viability_mean` | float64 | 16,752 | 0.0 | Mean viability across observed replicates (missing values skipped, not imputed) |
| `viability_std` | float64 | 16,752 | 0.0 | Sample SD (ddof=1) across observed replicates |
| `viability_min` | float64 | 16,752 | 0.0 | Minimum observed replicate viability |
| `viability_max` | float64 | 16,752 | 0.0 | Maximum observed replicate viability |
| `viability_spread` | float64 | 16,752 | 0.0 | max - min across observed replicates (technical variability) |
| `has_all_replicates` | bool | 16,752 | 0.0 | True when all nominal replicate slots are published |
| `mu_raw` | float64 | 16,752 | 0.0 | Published mu/muMax exactly as supplied (never recomputed in place) |
| `x_x0_raw` | float64 | 16,752 | 0.0 | Published X/X0 exactly as supplied (never clipped, never winsorised) |
| `x_x0_recalc` | float64 | 16,752 | 0.0 | Recomputed exp(ln(4)*(mu/muMax-1)) for auditing only |
| `response_formula_residual` | float64 | 16,752 | 0.0 | abs(x_x0_raw - x_x0_recalc) |
| `response_formula_ok` | bool | 16,752 | 0.0 | Residual within the configured tolerance (1e-5) |
| `response_gt1` | bool | 16,752 | 0.0 | Flag: published X/X0 > 1 (retained) |
| `response_gt2` | bool | 16,752 | 0.0 | Flag: published X/X0 > 2.0 (retained and reported as outlier) |
| `response_lt0` | bool | 16,752 | 0.0 | Flag: published X/X0 < 0 (none expected) |
| `response_eq0` | bool | 16,752 | 0.0 | Flag: published X/X0 == 0 (formula floor; retained as published) |
| `response_isna` | bool | 16,752 | 0.0 | Flag: published X/X0 is missing |
| `grid_token` | object | 16,752 | 0.0 | Canonical sorted single-agent dose grid (µM, '|' separated) |
| `mono_curve_id` | object | 16,752 | 0.0 | Single-agent curve = batch x cell x drug x dose grid (MONO_<hash>) |
| `mono_condition_id` | object | 16,752 | 0.0 | Single-agent condition = drug x cell line (MCOND_<hash>) |
| `row_idx` | int16 | 16,752 | 0.0 | Dense rank (0-based) of the dose within the unit |

### `oneil_mono_curve_metadata.parquet` (2,094 rows × 25 cols)

| 列 | dtype | 非空 | 空值% | 含义 |
| --- | --- | ---: | ---: | --- |
| `mono_curve_id` | object | 2,094 | 0.0 | Single-agent curve = batch x cell x drug x dose grid (MONO_<hash>) |
| `batch_id` | int64 | 2,094 | 0.0 | Published BatchID: 1/2 = primary large screen, 3 = partial repeat |
| `cell_line` | object | 2,094 | 0.0 | Cell line name exactly as published |
| `cell_id` | object | 2,094 | 0.0 | DoseSyn cell-line identifier (CELL_<SLUG>) |
| `drug_id` | object | 2,094 | 0.0 | DoseSyn drug identifier (DRUG_<SLUG>) |
| `drug_name` | object | 2,094 | 0.0 | Single-agent drug name exactly as published |
| `mono_condition_id` | object | 2,094 | 0.0 | Single-agent condition = drug x cell line (MCOND_<hash>) |
| `n_points` | int64 | 2,094 | 0.0 | Number of dose coordinates in the unit |
| `n_unique_doses` | int64 | 2,094 | 0.0 | Number of unique single-agent doses |
| `source_row_min` | int64 | 2,094 | 0.0 | First Excel row of the unit |
| `source_row_max` | int64 | 2,094 | 0.0 | Last Excel row of the unit |
| `n_missing_response` | int64 | 2,094 | 0.0 | Dose coordinates with missing X/X0 |
| `total_missing_replicates` | int16 | 2,094 | 0.0 | Total missing replicate values in the curve |
| `n_points_reduced_reps` | int64 | 2,094 | 0.0 | Coordinates with fewer than the nominal replicates |
| `x_x0_min` | float64 | 2,094 | 0.0 | Minimum published X/X0 in the unit |
| `x_x0_max` | float64 | 2,094 | 0.0 | Maximum published X/X0 in the unit |
| `x_x0_mean` | float64 | 2,094 | 0.0 | Mean published X/X0 in the unit (descriptive only) |
| `n_x_gt1` | int64 | 2,094 | 0.0 | Number of coordinates with X/X0 > 1 |
| `n_x_gt2` | int64 | 2,094 | 0.0 | Number of coordinates with X/X0 > 2 |
| `n_x_lt0` | int64 | 2,094 | 0.0 | Number of coordinates with X/X0 < 0 |
| `max_replicate_spread` | float64 | 2,094 | 0.0 | Largest replicate max-min in the unit |
| `n_high_replicate_spread` | int64 | 2,094 | 0.0 | Coordinates whose replicate spread exceeds the threshold |
| `dose_um_json` | object | 2,094 | 0.0 | JSON list of single-agent doses (µM) |
| `source_rows_contiguous` | bool | 2,094 | 0.0 | True when the unit occupies a contiguous block of source rows |
| `complete_curve` | bool | 2,094 | 0.0 | True for a full 8-point curve |

### `oneil_mono_replicates.parquet` (100,512 rows × 10 cols)

| 列 | dtype | 非空 | 空值% | 含义 |
| --- | --- | ---: | ---: | --- |
| `mono_curve_id` | object | 100,512 | 0.0 | Single-agent curve = batch x cell x drug x dose grid (MONO_<hash>) |
| `mono_condition_id` | object | 100,512 | 0.0 | Single-agent condition = drug x cell line (MCOND_<hash>) |
| `source_row` | int64 | 100,512 | 0.0 | 1-based Excel row in the published sheet (header = row 1) |
| `row_idx` | int16 | 100,512 | 0.0 | Dense rank (0-based) of the dose within the unit |
| `conc_um` | float64 | 100,512 | 0.0 | Single-agent concentration converted to µM |
| `x_x0_raw` | float64 | 100,512 | 0.0 | Published X/X0 exactly as supplied (never clipped, never winsorised) |
| `replicate_label` | object | 100,512 | 0.0 | Source column name (viability1 ... viabilityN) |
| `viability` | float64 | 84,912 | 15.521 | — |
| `replicate_index` | int16 | 100,512 | 0.0 | Technical-replicate number (1-based) |
| `is_missing` | bool | 100,512 | 0.0 | True when this replicate slot is NaN in the published data |

## 4. 单位与公式

- 浓度单位：原始/目标均为 **µM**（转换因子 1.0，记录于配置 `units`）。
- 生长率：`mu/muMax = ((LN(median viability) / (muMax*96)) + 1)`（原文定义）。
- 响应：`X/X0 = EXP(LN(4)*((mu/muMax)-1))`；0 = 完全抑制，1 = 无效应。
- 审计公式复算时残差仅来自源文件 5 位小数舍入，容差 1e-5。
- `X/X0` **不做任何裁剪**：>1 与 >2 的值原样保留并单独标记。

## 5. QC 阈值与理由

| 阈值 | 值 | 理由 |
| --- | ---: | --- |
| 每矩阵名义重复数 | 4 | 图例：组合每板最多 4 个技术重复 |
| 每单药曲线名义重复数 | 6 | 图例：单药每板最多 6 个技术重复 |
| X/X0 高异常阈值 | 2.0 | 仅标记，不裁剪 |
| X/X0 低异常阈值 | 0.0 | 负响应 = 公式不可能值 |
| 技术重复极差阈值 | 1.0 | >1.0 的板内波动会主导建模信号 |
| 公式复算容差 | 1e-05 | 源数据 5 位小数舍入 |
| 每轴剂量数 | 4 | 4×4 矩阵 |
| 每单药曲线剂量数 | 8 | 8 点曲线 |

## 6. 已知注意事项

- **批次不可合并**：同一条件在 batch 3 中重复时保留为独立矩阵；共 315 个条件跨批次，其中 105 个网格相同、210 个网格不同。
- **缺失技术重复**：以 NaN 保留，不插补；均值仅由观测重复计算。
- **X/X0 为 0**：源文件存在 213 个精确 0（公式下限约 3.8e-6），按发布值保留并标记 `response_floor_zero`。
- **M2 拆分约束**：同一 `condition_id` 的矩阵在任何划分中只能落入一个 split；M1 未生成 split 文件，避免信息泄漏。

## 7. 复现命令

```bash
cd /home/kouzheng_2/AH_YeP/DoseSyn
.venv/bin/python scripts/00_audit_raw.py
.venv/bin/python scripts/01_prepare_oneil.py
.venv/bin/python -m pytest -q
```
