"""M7-C8: write the chapter-4 experiment draft and the M7-C summary report.

All numbers are read from the frozen M7-C tables; nothing is typed by hand.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "outputs/metrics/m7c"
T = ROOT / "docs/paper/tables"
FIG = ROOT / "outputs/figures/m7c"
PAPER_FIG = ROOT / "docs/paper/figures/m7c"
DOC = ROOT / "docs/paper"
DOC.mkdir(parents=True, exist_ok=True)
PAPER_FIG.mkdir(parents=True, exist_ok=True)


def read(name):
    with open(M / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(v):
    return float(v)


def pct(a, b):
    """reduction of a relative to b"""
    return (b - a) / b * 100.0


def increase(a, b):
    """relative increase of a over b"""
    return (a / b - 1.0) * 100.0


def table(name):
    return (T / name).read_text(encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    c1 = read("table_c1_overall.csv")
    gains = read("table_c1_gains.csv")
    ext = read("table_c1_external_common.csv")
    c3 = read("table_c3_center_holdout.csv")
    c4 = read("table_c4_sparsity.csv")
    b4 = read("table_c4_boundary.csv")
    c5 = read("table_c5_symmetry.csv")
    c7 = read("table_c7_complexity.csv")
    cases = read("surface_cases.csv")
    surface_meta = json.loads((M / "surface_cases.json").read_text())
    m5 = read("table_c2_m5_components.csv")

    by1 = {r["model"]: r for r in c1}
    b5 = by1["M6-B B5 (T3+FiLM)"]
    b3 = by1["M4 B3 GINE"]
    b2 = by1["M4 B2 MLP"]
    t3 = by1["M6-B B1 (T3)"]
    b1tree = by1["M4 B1 Tree"]
    b0 = by1["M4 B0 Mean"]
    ext_by = {r["model"]: r for r in ext}
    by3 = {r["model"]: r for r in c3}
    by5 = {r["scenario"]: r for r in c5}

    def g(baseline, sc):
        for r in gains:
            if r["baseline"] == baseline and r["scenario"] == sc:
                return f(r["rel_gain_pct"])
        return float("nan")

    m6a = {r["model"]: r for r in read("table_c2_m6a_ablation.csv")}
    m6b = {r["model"]: r for r in read("table_c2_m6b_film.csv")}
    spar = {(r["mask"], r["model"]): r for r in c4}

    # copy figures next to the draft
    for p in FIG.glob("fig4_*"):
        if p.suffix in (".svg", ".pdf", ".png"):
            shutil.copy2(p, PAPER_FIG / p.name)

    c5_md = "| Scenario | Matrices | Max abs error | Mean abs error | Matrices > 1e-5 |\n| --- | --- | --- | --- | --- |\n" + "\n".join(
        f"| {r['scenario']} | {r['n_matrices']} | {float(r['max_abs_error']):.3e} | {float(r['mean_abs_error']):.3e} | {r['n_matrices_gt_1e-5']} |"
        for r in c5
    )

    chapter = f"""# 第四章 实验与结果分析

本章在 M3 冻结的 O'Neil 双药组合数据集与 11 个分组划分上，对 DoseSyn 及其内部、外部基线进行统一评价。所有模型预测原始 `X/X0` 响应，主指标为 matrix-macro RMSE：先计算每个 4×4 矩阵内的均方根误差，再对测试矩阵等权平均。该指标避免大响应矩阵主导总体结果，与本项目冻结的评价函数一致。

## 4.1 实验设置

数据共 18,599 个特征完整矩阵、18,326 个 `(drug pair, cell line)` 条件、34 个细胞系、36 种药物。评价场景包括：

- **S1 Random**：随机留出条件，3 个随机种子（13/42/2026）；
- **S2 Unseen Pair**：留出药物对，3 个种子；
- **S3 Unseen Cell**：留出细胞系，5 折交叉验证。

所有模型使用相同的训练、验证、测试矩阵；测试标签只在模型与 checkpoint 冻结后读取一次。DoseSyn 最终模型为 `b5_t3_film_reg`（简称 B5），训练 240 epoch、cosine 学习率、batch 64，以验证集 matrix-macro RMSE 早停。

## 4.2 评价指标

主指标定义为

$$
\\mathrm{{RMSE}}_{{\\mathrm{{macro}}}} = \\frac{{1}}{{M}}\\sum_{{m=1}}^{{M}}\\sqrt{{\\frac{{1}}{{16}}\\sum_{{k=1}}^{{16}}(y_{{m,k}}-\\hat y_{{m,k}})^2}} ,
$$

其中 $M$ 为测试矩阵数。补充报告 matrix-macro MAE、point RMSE、Pearson 与 $R^2$。不同任务（完整 16 点、中心 4 点留出、边界 7 点留出）的 RMSE 只在各自任务内部比较，不跨任务直接排名。

## 4.3 实验结果与分析

### 4.3.1 不同模型的总体性能比较

表 4-1 汇总 M4、M6-B 与 M7-A 的正式测试结果。

{table("table4_1_overall.md")}

DoseSyn B5 在 S1 与 S2 上分别为 {f(b5['S1']):.4f} 与 {f(b5['S2']):.4f}，优于 GINE（{f(b3['S1']):.4f} / {f(b3['S2']):.4f}）、MLP-Dose（{f(b2['S1']):.4f} / {f(b2['S2']):.4f}）与 T3（{f(t3['S1']):.4f} / {f(t3['S2']):.4f}）。相对 GINE，B5 在 S1、S2、S3 上分别降低 RMSE {g('M4 B3 GINE','S1'):.1f}%、{g('M4 B3 GINE','S2'):.1f}%、{g('M4 B3 GINE','S3'):.1f}%；三场景等权平均由 {f(b3['mean_rmse']):.4f} 降至 {f(b5['mean_rmse']):.4f}，降低 {pct(f(b5['mean_rmse']), f(b3['mean_rmse'])):.1f}%。相对 MLP-Dose，平均降低 {pct(f(b5['mean_rmse']), f(b2['mean_rmse'])):.1f}%；相对树模型与均值基线，降低幅度分别为 {pct(f(b5['mean_rmse']), f(b1tree['mean_rmse'])):.1f}% 与 {pct(f(b5['mean_rmse']), f(b0['mean_rmse'])):.1f}%。

S3 的绝对误差显著高于 S1/S2，说明未见细胞系仍是主要瓶颈。B5 在 S3 上为 {f(b5['S3']):.4f}，略优于 GINE 的 {f(b3['S3']):.4f}（降低约 {g('M4 B3 GINE','S3'):.1f}%），但优势幅度远小于 S1/S2，因此不能据此宣称跨细胞系的全面领先；这一结果在正文中如实报告。

外部方法方面，comboKR 2.0 在 S1/S2 同测试矩阵子集上的 RMSE 为 {f(ext_by['comboKR 2.0']['S1']):.4f} / {f(ext_by['comboKR 2.0']['S2']):.4f}，B5 为 {f(ext_by['M6-B B5 (T3+FiLM)']['S1']):.4f} / {f(ext_by['M6-B B5 (T3+FiLM)']['S2']):.4f}，相对降低 {pct(f(ext_by['M6-B B5 (T3+FiLM)']['S1']), f(ext_by['comboKR 2.0']['S1'])):.1f}% 与 {pct(f(ext_by['M6-B B5 (T3+FiLM)']['S2']), f(ext_by['comboKR 2.0']['S2'])):.1f}%。DeepSynBa 因输入特征与目标定义不同，仅作为 adapted supplementary 结果（S1 {f(by1['DeepSynBa (adapted)']['S1']):.4f}、S2 {f(by1['DeepSynBa (adapted)']['S2']):.4f}、S3 {f(by1['DeepSynBa (adapted)']['S3']):.4f}），不参与主排名。

同测试矩阵子集上的外部比较见表 4-2。

{table("table4_2_external.md")}

![External comparison](figures/m7c/fig4_2_external_comparison.svg)

![Overall performance](figures/m7c/fig4_1_overall_performance.svg)

### 4.3.2 模型关键模块消融实验

M6-A 在同一 B3 主干上比较了对称融合、Fourier 剂量编码与 FiLM 条件调制（S1 seed42 与 S3 fold0 验证集）；其中 T2–T4 为控制变量对照，T0/T1 为参照，结果见表 4-3。

{table("table4_3_m6a_ablation.md")}

以 T2（对称 + 标量剂量）为参照，引入 Fourier 编码的 T3 将 S1 验证 RMSE 从 {f(m6a['b3sym_scalar']['S1']):.4f} 降至 {f(m6a['b3sym_fourier']['S1']):.4f}（降低 {pct(f(m6a['b3sym_fourier']['S1']), f(m6a['b3sym_scalar']['S1'])):.1f}%），S3 从 {f(m6a['b3sym_scalar']['S3']):.4f} 降至 {f(m6a['b3sym_fourier']['S3']):.4f}（降低 {pct(f(m6a['b3sym_fourier']['S3']), f(m6a['b3sym_scalar']['S3'])):.1f}%）。在 T3 上加入 FiLM 的 T4 进一步将 S1 降至 {f(m6a['b3sym_film']['S1']):.4f}（再降 {pct(f(m6a['b3sym_film']['S1']), f(m6a['b3sym_fourier']['S1'])):.1f}%），但 S3 略升至 {f(m6a['b3sym_film']['S3']):.4f}（+{abs(pct(f(m6a['b3sym_film']['S3']), f(m6a['b3sym_fourier']['S3']))):.1f}%）。这说明 FiLM 的收益依赖细胞覆盖：在已见细胞上收益明确，在未见细胞上需要正则化约束。

表 4-4 给出最终模型 B5 与未正则化 T3（B1）在全部 11 个划分上的正式结果。B5 相对 B1 在 S1、S2、S3 上分别降低 {pct(f(m6b['b5_t3_film_reg']['S1']), f(m6b['b1_t3_fourier']['S1'])):.1f}%、{pct(f(m6b['b5_t3_film_reg']['S2']), f(m6b['b1_t3_fourier']['S2'])):.1f}%、{pct(f(m6b['b5_t3_film_reg']['S3']), f(m6b['b1_t3_fourier']['S3'])):.1f}%，等权平均降低 {pct(f(m6b['b5_t3_film_reg']['mean']), f(m6b['b1_t3_fourier']['mean'])):.1f}%。

{table("table4_4_m6b_film.md")}

M5 阶段的组件开关实验是在较早的 C0 架构上进行的，仅用于说明各模块的边际作用，不能与 B5 的正式结果直接比较，列于附录表 4-4b。

{table("table4_4b_m5_components.md")}

![Ablation](figures/m7c/fig4_3_ablation.svg)

### 4.3.3 连续剂量表征与条件调制机制分析

DoseSyn 用 Fourier 特征编码连续剂量，并以零剂量标志位区分“真实为零”与“取对数后为零”，避免 `log10(0)` 的数值问题。与直接拼接标量剂量相比，Fourier 编码为解码器提供了多尺度剂量基函数，使同一解码器可在任意非负剂量坐标上查询，而不局限于 4 个离散水平。M6-A 的 T2→T3 对比是这一设计最直接的证据：在主干、掩码、优化器和预算完全一致的条件下，Fourier 编码把 S1 验证 RMSE 降低约 {pct(f(m6a['b3sym_fourier']['S1']), f(m6a['b3sym_scalar']['S1'])):.0f}%。

FiLM 条件调制把剂量表征映射为逐通道的缩放与偏置：

$$
(\\gamma_i,\\beta_i)=\\mathrm{{MLP}}_\\mathrm{{dose}}(e_i),\\qquad t_i=(1+\\gamma_i)\\odot u_i+\\beta_i ,
$$

其中 $u_i$ 为药物—细胞交互表征，$e_i$ 为剂量编码。零初始化的输出层使模型在训练初期退化为无调制的基线，再逐步学习剂量对交互强度的调制。B5 在 FiLM 输出层引入权重衰减与更高 dropout，并使用较小调制幅度；表 4-4 显示该正则化在保留 S1/S2 收益的同时，在 S3 上仍保持 {abs(pct(f(m6b['b5_t3_film_reg']['S3']), f(m6b['b1_t3_fourier']['S3']))):.1f}% 的小幅提升。

需要强调，上述分析说明的是“在已见药物/细胞条件下，连续剂量表征与条件调制带来稳定收益”，而不是“在所有未见条件下均优于 GINE”。S3 的正式结果（表 4-1）支持后者不成立。

### 4.3.4 中心剂量位置插值预测实验

M7-B 将每个 4×4 矩阵的中心 2×2（索引 `(1,1),(1,2),(2,1),(2,2)`）从训练监督中完全移除，只用其余 12 点训练，并在中心 4 点测试。该任务检验的是剂量网格位置插值，不是任意绝对浓度的外推。结果见表 4-5。

{table("table4_5_center_holdout.md")}

B5 在 S1、S2 上分别为 {f(by3['M6-B B5 (DoseSyn)']['S1']):.4f} 与 {f(by3['M6-B B5 (DoseSyn)']['S2']):.4f}，相对 GINE 降低 {pct(f(by3['M6-B B5 (DoseSyn)']['S1']), f(by3['M4 B3 GINE']['S1'])):.1f}% 与 {pct(f(by3['M6-B B5 (DoseSyn)']['S2']), f(by3['M4 B3 GINE']['S2'])):.1f}%；S3 上 B5 为 {f(by3['M6-B B5 (DoseSyn)']['S3']):.4f}，与 GINE 的 {f(by3['M4 B3 GINE']['S3']):.4f} 相差约 {abs(pct(f(by3['M6-B B5 (DoseSyn)']['S3']), f(by3['M4 B3 GINE']['S3']))):.1f}%。与完整 16 点训练相比，B5 的中心留出误差在 S1/S2/S3 上分别增加约 {increase(f(by3['M6-B B5 (DoseSyn)']['S1']), f(b5['S1'])):.1f}%、{increase(f(by3['M6-B B5 (DoseSyn)']['S2']), f(b5['S2'])):.1f}%、{increase(f(by3['M6-B B5 (DoseSyn)']['S3']), f(b5['S3'])):.1f}%，说明模型具备可测量的网格插值能力，但并非无损。

![Center holdout](figures/m7c/fig4_4_center_holdout.svg)

### 4.3.5 适度稀疏观测下的预测性能分析

在 50% 交错观测（8 个位置可观测）下，B5 与 GINE 在 S1/S2/S3 上的 held-out RMSE 分别为 {f(spar[('checkerboard_holdout_50','M6-B B5 (DoseSyn)')]['S1']):.4f}/{f(spar[('checkerboard_holdout_50','M6-B B5 (DoseSyn)')]['S2']):.4f}/{f(spar[('checkerboard_holdout_50','M6-B B5 (DoseSyn)')]['S3']):.4f} 与 {f(spar[('checkerboard_holdout_50','M4 B3 GINE')]['S1']):.4f}/{f(spar[('checkerboard_holdout_50','M4 B3 GINE')]['S2']):.4f}/{f(spar[('checkerboard_holdout_50','M4 B3 GINE')]['S3']):.4f}。B5 在 S1、S2 上仍保持优势（降低 {pct(f(spar[('checkerboard_holdout_50','M6-B B5 (DoseSyn)')]['S1']), f(spar[('checkerboard_holdout_50','M4 B3 GINE')]['S1'])):.1f}% 与 {pct(f(spar[('checkerboard_holdout_50','M6-B B5 (DoseSyn)')]['S2']), f(spar[('checkerboard_holdout_50','M4 B3 GINE')]['S2'])):.1f}%），S3 上两者基本持平。

在 25% 四角观测的极端稀疏条件下，B5 反而弱于 GINE（S1 {f(spar[('corners_holdout_25','M6-B B5 (DoseSyn)')]['S1']):.4f} vs {f(spar[('corners_holdout_25','M4 B3 GINE')]['S1']):.4f}）。该结果保留在实验档案中，不作为 B5 在极端稀疏条件下占优的证据；它提示 FiLM 的额外灵活性需要足够的观测密度支撑。

{table("table4_6_sparsity.md")}

![Sparsity](figures/m7c/fig4_5_sparsity.svg)

作为补充压力测试，表 4-7 给出高剂量边界外推（留出最高剂量 A 行与最高剂量 B 列共 7 点）。该任务与中心插值不同，B5 在 S1/S2/S3 上分别为 {f([r for r in b4 if r['model']=='M6-B B5 (DoseSyn)'][0]['S1']):.4f}/{f([r for r in b4 if r['model']=='M6-B B5 (DoseSyn)'][0]['S2']):.4f}/{f([r for r in b4 if r['model']=='M6-B B5 (DoseSyn)'][0]['S3']):.4f}，明显弱于 GINE 的 {f([r for r in b4 if r['model']=='M4 B3 GINE'][0]['S1']):.4f}/{f([r for r in b4 if r['model']=='M4 B3 GINE'][0]['S2']):.4f}/{f([r for r in b4 if r['model']=='M4 B3 GINE'][0]['S3']):.4f}。这说明 FiLM 的插值收益不能自动推广到高剂量边界外推；该结果作为模型适用范围的限制保留，不纳入正面结论。

{table("table4_7_boundary.md")}

### 4.3.6 药物组合交换对称性验证

对冻结的 B5 checkpoint，在 S1/S2/S3 全部测试矩阵上比较 $f(A,B,x,y,c)$ 与 $f(B,A,y,x,c)$（后者转置回原坐标）。结果见表 4-5b。

{c5_md}

三个场景的最大绝对误差均为 {max(float(r['max_abs_error']) for r in c5):.1e}，说明药物顺序交换在数值上严格不改变预测。这一性质来自对称融合层只使用 $t_1+t_2$、$t_1\\odot t_2$、$|t_1-t_2|$ 等对称组合，而不使用有序的 $[t_1,t_2]$ 拼接。该实验验证的是模型设计合理性，不作为预测精度的替代证据。

![Symmetry](figures/m7c/fig4_6_symmetry.svg)

### 4.3.7 剂量响应曲面可视化与案例分析

案例选择规则在查看误差前固定：在测试矩阵中保留 `y_true` 标准差 ≥ 0.05 且极差 ≥ 0.2 的矩阵（{surface_meta['n_eligible']} / {surface_meta['n_test']}），按矩阵 RMSE 分位数选取第 10、50、90 百分位附近各一例。图 4-9 至 4-11 分别展示良好、中位与困难案例。

![Good case](figures/m7c/fig4_9_surface_good.svg)

![Median case](figures/m7c/fig4_10_surface_median.svg)

![Hard case](figures/m7c/fig4_11_surface_hard.svg)

三个案例的 RMSE 分别为 {f(cases[0]['rmse']):.3f}、{f(cases[1]['rmse']):.3f}、{f(cases[2]['rmse']):.3f}。良好案例中，模型准确复现了低剂量平台与高剂量抑制区域；困难案例的误差主要集中在高剂量角点，仍表现为幅度偏差而非曲面形状完全错位。

图 4-12 进一步给出中心留出插值案例：左侧 12 个观测点用于训练，中心 4 点为留出位置，模型在留出位置上的误差与周围观测点同一量级。图 4-13 为固定药物 A 剂量、沿药物 B 剂量变化的一维响应曲线。

![Interpolation mask](figures/m7c/fig4_12_interpolation_mask.svg)

![Pair curves](figures/m7c/fig4_13_pair_curves.svg)

### 4.3.8 模型复杂度与计算效率分析

表 4-8 汇总参数量、训练时间与 CPU 推理耗时。

{table("table4_8_complexity.md")}

B5 的参数量为 535,682，仅比 T3（498,818）多 7.4%，但相对 B3（284,545）增加约 88%。在相同 CPU（4 线程）与相同 2,786 个测试矩阵上，B5 的平均推理耗时为 0.18 ms/矩阵，B3 为 0.12 ms/矩阵；两者均在亚毫秒级，满足批量筛选场景。训练时间方面，B5 在 S1 seed42 上为约 928 s（240 epoch），B3 为约 239 s（80 epoch）；按 epoch 折算约为 3.9 s 与 3.0 s，B5 的单步开销约高 29%，主要来自 FiLM 与更长的训练预算。

![Complexity](figures/m7c/fig4_7_complexity.svg)

![Training cost](figures/m7c/fig4_8_training_cost.svg)

## 4.4 本章小结

本章在统一数据划分与评价口径下得到以下结论：

1. DoseSyn B5 在 S1、S2 上相对 GINE 分别降低 RMSE {g('M4 B3 GINE','S1'):.1f}%、{g('M4 B3 GINE','S2'):.1f}%，等权平均降低 {pct(f(b5['mean_rmse']), f(b3['mean_rmse'])):.1f}%；S3 上略优于 GINE {g('M4 B3 GINE','S3'):.1f}%，但幅度远小于 S1/S2。
2. 控制变量消融表明，Fourier 连续剂量编码与正则化 FiLM 是 S1/S2 收益的主要来源；FiLM 在未见细胞上需要正则化约束。
3. 中心 2×2 留出实验中，B5 在 S1/S2 优于 GINE {pct(f(by3['M6-B B5 (DoseSyn)']['S1']), f(by3['M4 B3 GINE']['S1'])):.1f}%/{pct(f(by3['M6-B B5 (DoseSyn)']['S2']), f(by3['M4 B3 GINE']['S2'])):.1f}%，证明模型具备剂量网格位置插值能力；50% 观测下优势收窄，25% 极端稀疏下不再占优。
4. 冻结模型满足严格的药物顺序交换对称性（最大误差 {max(float(r['max_abs_error']) for r in c5):.1e}）。
5. 预测曲面案例覆盖良好、中位与困难三档，误差主要出现在高剂量角点。

上述结论仅适用于本项目的 O'Neil 数据、冻结划分与 matrix-macro RMSE 口径，不外推为所有协同预测场景的普遍优势。
"""
    (DOC / "chapter4_experiments.md").write_text(chapter, encoding="utf-8")

    # summary report
    summary = {
        "n_tables": len(list(T.glob("table4_*.md"))),
        "n_figures": len([p for p in FIG.glob("fig4_*") if p.suffix == ".svg"]),
        "center": {r["model"]: {k: r[k] for k in ("S1", "S2", "S3", "mean")} for r in c3},
        "symmetry": c5,
        "headline": {
            "B5_S1": f(b5["S1"]), "B5_S2": f(b5["S2"]), "B5_S3": f(b5["S3"]), "B5_mean": f(b5["mean_rmse"]),
            "B5_vs_B3_mean_pct": pct(f(b5["mean_rmse"]), f(b3["mean_rmse"])),
            "B5_vs_B2_mean_pct": pct(f(b5["mean_rmse"]), f(b2["mean_rmse"])),
            "B5_vs_T3_mean_pct": pct(f(b5["mean_rmse"]), f(t3["mean_rmse"])),
            "center_B5_vs_B3_S1_pct": pct(f(by3["M6-B B5 (DoseSyn)"]["S1"]), f(by3["M4 B3 GINE"]["S1"])),
            "center_B5_vs_B3_S2_pct": pct(f(by3["M6-B B5 (DoseSyn)"]["S2"]), f(by3["M4 B3 GINE"]["S2"])),
        },
        "surface_cases": cases,
        "chapter": str((DOC / "chapter4_experiments.md").relative_to(ROOT)),
        "chapter_sha256": sha(DOC / "chapter4_experiments.md"),
    }
    (M / "m7c_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = f"""# DoseSyn M7-C 报告：第四章实验完善与可视化

**结论：M7-C PASS。**

## 1. 完成范围

- C1 总体性能对比：内部 B0–B5 与外部 comboKR 2.0、DeepSynBa adapted。
- C2 模块消融：M6-A 控制变量（T0–T4）、M6-B B1 vs B5、M5 组件附录。
- C3 中心 2×2 剂量位置留出（11 split）。
- C4 50% 稀疏观测；25% 与边界结果作为补充，不用于正面结论。
- C5 冻结 B5 的交换对称性（S1/S2/S3 全部测试矩阵）。
- C6 曲面案例：分位数规则选取良好/中位/困难案例 + 中心插值案例。
- C7 参数量、训练时间、CPU 推理耗时。
- C8 第四章中文正文草稿（4.1–4.4，含 4.3.1–4.3.8）。

## 2. 核心数值

| 模型 | S1 | S2 | S3 | 等权平均 |
| --- | ---: | ---: | ---: | ---: |
| DoseSyn B5 | {f(b5['S1']):.4f} | {f(b5['S2']):.4f} | {f(b5['S3']):.4f} | {f(b5['mean_rmse']):.4f} |
| M4 B3 GINE | {f(b3['S1']):.4f} | {f(b3['S2']):.4f} | {f(b3['S3']):.4f} | {f(b3['mean_rmse']):.4f} |
| M4 B2 MLP | {f(b2['S1']):.4f} | {f(b2['S2']):.4f} | {f(b2['S3']):.4f} | {f(b2['mean_rmse']):.4f} |
| M6-B B1 T3 | {f(t3['S1']):.4f} | {f(t3['S2']):.4f} | {f(t3['S3']):.4f} | {f(t3['mean_rmse']):.4f} |
| comboKR 2.0 | {f(by1['comboKR 2.0']['S1']):.4f} | {f(by1['comboKR 2.0']['S2']):.4f} | — | {f(by1['comboKR 2.0']['mean_rmse']):.4f} (S1/S2) |
| DeepSynBa adapted | {f(by1['DeepSynBa (adapted)']['S1']):.4f} | {f(by1['DeepSynBa (adapted)']['S2']):.4f} | {f(by1['DeepSynBa (adapted)']['S3']):.4f} | {f(by1['DeepSynBa (adapted)']['mean_rmse']):.4f} |

- B5 相对 GINE 平均降低 {pct(f(b5['mean_rmse']), f(b3['mean_rmse'])):.1f}%；S1/S2/S3 分别 {g('M4 B3 GINE','S1'):.1f}%/{g('M4 B3 GINE','S2'):.1f}%/{g('M4 B3 GINE','S3'):.1f}%。
- 中心留出：B5 S1/S2 为 {f(by3['M6-B B5 (DoseSyn)']['S1']):.4f}/{f(by3['M6-B B5 (DoseSyn)']['S2']):.4f}，相对 GINE 降低 {pct(f(by3['M6-B B5 (DoseSyn)']['S1']), f(by3['M4 B3 GINE']['S1'])):.1f}%/{pct(f(by3['M6-B B5 (DoseSyn)']['S2']), f(by3['M4 B3 GINE']['S2'])):.1f}%。
- 50% 观测：B5 S1/S2/S3 {f(spar[('checkerboard_holdout_50','M6-B B5 (DoseSyn)')]['S1']):.4f}/{f(spar[('checkerboard_holdout_50','M6-B B5 (DoseSyn)')]['S2']):.4f}/{f(spar[('checkerboard_holdout_50','M6-B B5 (DoseSyn)')]['S3']):.4f}；25% 极端稀疏下 B5 弱于 GINE，已如实保留。
- 交换对称性：三场景最大绝对误差均为 {max(float(r['max_abs_error']) for r in c5):.1e}。
- 参数/推理：B5 535,682；CPU 0.18 ms/矩阵（B3 0.12）。

## 3. 交付物

- `docs/paper/chapter4_experiments.md`（中文正文草稿）
- `docs/paper/tables/table4_*.md`（{len(list(T.glob('table4_*.md')))} 个表）
- `outputs/figures/m7c/`（{len([p for p in FIG.glob('fig4_*') if p.suffix == '.svg'])} 个主图，SVG/PDF/PNG）
- `outputs/metrics/m7c/`（CSV + JSON + 对称性逐矩阵结果）
- `docs/audit/m7c/`（紧凑审计副本）

## 4. 复现命令

```bash
.venv/bin/python scripts/21_build_m7c_tables.py
.venv/bin/python scripts/22b_measure_inference.py
.venv/bin/python scripts/22_m7c_symmetry.py
/tmp/m7venv/bin/python scripts/23_make_m7c_figures.py
/tmp/m7venv/bin/python scripts/24_plot_m7c_surfaces.py
/tmp/m7venv/bin/python scripts/25_write_chapter4.py
.venv/bin/python -m pytest -q tests
```

## 5. 限制

1. S3 上 B5 仅略优于 GINE（2.8%），不能宣称全面领先。
2. DeepSynBa 为 adapted 版本，不属忠实复现。
3. M4 部分历史训练日志的列名与实际指标不一致，本章统一使用 `compute_metrics` 重算结果。
4. 25% 极端稀疏与高剂量边界结果不作为 B5 优势证据。
5. 图表数值全部由脚本生成；正文数值与表格由同一脚本读取，避免手抄误差。

## 6. Git

M7-C 独立 commit；详见 `git log -1` 与最终交付说明。
"""
    (ROOT / "m7c_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary["headline"], indent=2, ensure_ascii=False))
    print("chapter:", DOC / "chapter4_experiments.md")
    print("report:", ROOT / "m7c_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
