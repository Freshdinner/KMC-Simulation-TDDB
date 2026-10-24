"""最小间隔 0.42 nm、E_A 高斯分布（1.45 ± 0.15 eV）、t₀ = 2.4 eV（r_c = 0.68 nm）的 3.6 V 测试汇总（其余同 examples/zro2_tddb.yaml，
seed 1、2，`drive.py`，各 10 h 机时）。对照是 E_A 标准差 0.05 eV、其余相同的同种子轨迹（`../../10.23/sep042/`）。

每条轨迹分段，时间原点不变（从应力开始计）：
- pre-stress 阶段：t < T_PRE = 0.1 s（实验的 I-t 从 0.1 s 起），只在 summary.txt 里记生成的空位数与 J、ΣG 的首末值，不画进图线；
- 幂律阶段：0.1 s 到击穿过程之前（没有击穿时到终点），I-t 拟合只用这一段的
  内层输运电流，N(t) 拟合只用其中 N ≥ 1 的记录。两次生成之间状态不变：0.1 s 这一点、拟合与局域斜率的取样点都取此前最后一条记录的值，
  (a)(b)(d) 的图线画成台阶；
- 击穿过程：负极导体最高点最后一轮连续上升的起点（悬空导电簇接上负极，或负极导体从尖端开始逐个空位长高，见 `segments`）到贯穿通路
  形成（第一条有细丝电流的记录，给出 t_BD）；
- 击穿后：再跑的 post_breakdown_steps 条记录（停跑时可能不足 20 条）。

统计写到 summary.txt。出图 it_sigma15.pdf / .png，版式同 `../../10.21/sep040/it_sep040`（Nature 版式：双栏 183 mm、7 pt；
Times New Roman + stix 公式、蓝红对比色、坐标轴框线与刻度线 1.5 pt），图线从 0.1 s 画起：(a) 总电流密度 J(t)（内层输运 + 间隙隧穿 + 细丝，
有击穿时空心圈标击穿点）；(b) N(t)（N ≥ 1）；(c) 幂律阶段的局域斜率 d lnJ/d ln t（内层输运电流，log t 上每个数量级 4 个点）；(d) ΣG(t)；(e) 接负极导电簇的成员数与最高点；(f) 幂律阶段新空位的出生高度分布。图中应力时间是模拟原值，不平移。
usage: analyze.py
"""
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.family": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix", "font.size": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 1.5, "xtick.major.width": 1.5,
    "ytick.major.width": 1.5, "xtick.minor.width": 1.5, "ytick.minor.width": 1.5, "xtick.major.size": 3,
    "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.direction": "out",
    "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
    "pdf.fonttype": 42, "savefig.dpi": 600, "lines.linewidth": 0.8})
MM = 1.0 / 25.4
HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, "..", "..", "10.23", "sep042")
SEEDS = (1, 2)
COLORS = {1: "#1f77b4", 2: "#d62728"}
T_PRE = 0.1  # pre-stress 阶段的终点 [s]
POST_BREAKDOWN_STEPS = 20  # examples/zro2_tddb.yaml 的 post_breakdown_steps
AMBIENT = 298.15  # 25 °C [K]
MAX_PAUSE = 3  # 击穿过程里负极导体最高点相邻两次抬高之间至多隔几条记录
STOPS = {"wall": "机时到点停跑", "memory": "内存保护停跑", "breakdown": "击穿后记录跑满停"}


def steps(folder, seed):
    """逐步数据：`steps_s{SEED}.csv` 里续跑起点之前的行 + `steps_s{SEED}_from{N}.csv`。"""
    rows = list(csv.DictReader(open(os.path.join(folder, "steps_s{}.csv".format(seed)), encoding="utf-8")))
    for resumed in sorted(glob.glob(os.path.join(folder, "steps_s{}_from*.csv".format(seed)))):
        start = int(resumed.rsplit("from", 1)[1][:4])
        rows = [r for r in rows if int(r["n_generated"]) < start] + list(csv.DictReader(open(resumed, encoding="utf-8")))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def held(t, y, grid):
    """grid 各时刻的值：取此前最后一条记录（两次生成之间状态不变）。"""
    return y[np.searchsorted(t, grid, side="right") - 1]


def log_fit(t, y, lo, hi):
    """[lo, hi] 上按 log t 等间距取 60 点（取值见 `held`）的双对数直线拟合：(斜率, R²)；区间里 y 不变时 R² 记 nan。"""
    grid = np.logspace(np.log10(lo), np.log10(hi), 60)
    ly = np.log(held(t, y, grid))
    slope = np.polyfit(np.log(grid), ly, 1)[0]
    return slope, np.corrcoef(np.log(grid), ly)[0, 1] ** 2 if np.ptp(ly) > 0 else float("nan")


def rows(d, start, stop=None):
    """第 start 到 stop（不含）条记录。"""
    return {k: v[start:stop].copy() for k, v in d.items()}


def from_pre_stress_end(d):
    """t ≥ T_PRE 的记录：第一行是 t = T_PRE 时的状态，即此前最后一条记录（两次生成之间状态不变），时间改记 T_PRE。"""
    stage = rows(d, int(np.searchsorted(d["t_s"], T_PRE, side="right")) - 1)
    stage["t_s"][0] = T_PRE
    return stage


def segments(d):
    """(幂律阶段, 击穿记录序号, 击穿过程起点记录序号)。击穿记录是第一条有细丝电流的记录。击穿过程是负极导体最高点一路升到贯穿的
    最后一轮连续上升：从击穿记录往回找抬高最高点的记录，相邻两条之间至多隔 MAX_PAUSE 条其他记录，最早的那条是起点（悬空导电簇接上
    负极时是接上的那条，负极导体从尖端逐个空位长高时是开始长高的那条）。没有击穿时后两个为 None，幂律阶段到终点。"""
    spanning = np.flatnonzero(d["J_filament_A_cm2"] > 0)
    if spanning.size == 0:
        return from_pre_stress_end(d), None, None
    b = int(spanning[0])
    raises = np.flatnonzero(np.diff(d["negative_front_nm"][:b + 1]) > 0) + 1
    k = b
    for j in raises[::-1]:
        if k - j > MAX_PAUSE + 1:
            break
        k = int(j)
    return from_pre_stress_end(rows(d, 0, k)), b, k


def it_fit(stage):
    """I-t（内层输运电流）的双对数拟合：(n, R², 前半段 n, 后半段 n)，前后半段在 log t 上对分。"""
    t, J = stage["t_s"], stage["J_transport_A_cm2"]
    n, r2 = log_fit(t, J, t[0], t[-1])
    half = np.sqrt(t[0] * t[-1])
    return n, r2, log_fit(t, J, t[0], half)[0], log_fit(t, J, half, t[-1])[0]


def elapsed(t, start, stop):
    """第 start 到第 stop 条记录经过的应力时间（文字）。steps CSV 的时间存 10 位有效数字，分辨不出的记为"< 分辨率"。"""
    resolution = 10.0 ** (np.floor(np.log10(t[stop])) - 9)
    return "{:.2g}".format(t[stop] - t[start]) if t[stop] - t[start] >= resolution else "< {:.0e}".format(resolution)


def births(stage):
    """这一段里出生的新空位的高度 [nm]：第一行之后各条记录的出生位置（续跑起点那条记录没有出生位置，记 nan，去掉）。"""
    z = stage["birth_z_nm"][1:]
    return z[np.isfinite(z)]


def stop_text(seed):
    """这个种子最后一段日志末行给出的停跑原因；还在跑时记"运行中"。"""
    last = open(sorted(glob.glob(os.path.join(HERE, "logs", "run_s{}*.log".format(seed))))[-1],
                encoding="utf-8").read().splitlines()[-1]
    return STOPS.get(last.split()[1], last.split()[1]) if last.startswith("done ") else "运行中"


new = {s: steps(HERE, s) for s in SEEDS}
old = {s: steps(OLD, s) for s in SEEDS}
seg = {s: segments(new[s]) for s in SEEDS}

lines = []
for s in SEEDS:
    d = new[s]
    p, b, k = seg[s]
    t, N, J = p["t_s"], p["n_generated"], p["J_transport_A_cm2"]
    m = N >= 1
    nN, rN = log_fit(t[m], N[m], t[m][0], t[m][-1])
    nJ, r2, n1, n2 = it_fit(p)
    onJ, or2, on1, on2 = it_fit(segments(old[s])[0])
    G, Gt = p["G_total_s"], p["G_trap_s"]
    z = births(p)
    steps_wall = np.diff(d["wall_s"])
    steps_wall = steps_wall[steps_wall > 0]  # 去掉续跑起点（机时从 0 重新计）
    if b is None:
        lines.append("== seed {}: {} 个空位，t = {:.4g} s，未击穿（{}）".format(s, int(d["n_generated"][-1]), d["t_s"][-1], stop_text(s)))
    else:
        lines.append("== seed {}: 击穿，t_BD = {:.6g} s，第 {} 个空位；击穿后有 {} / {} 条记录（{}）".format(
            s, d["t_s"][b], int(d["n_generated"][b]), len(d["t_s"]) - 1 - b, POST_BREAKDOWN_STEPS, stop_text(s)))
    lines += [
        "  pre-stress（t < {:g} s）：生成 {} 个空位；J {:.3g} -> {:.3g} A/cm2；ΣG {:.3g} -> {:.3g} 1/s".format(
            T_PRE, int(N[0]), d["J_transport_A_cm2"][0], J[0], d["G_total_s"][0], G[0]),
        "  幂律阶段（{:g} s 到 {:.6g} s{}）：J {:.3g} -> {:.3g} A/cm2；ΣG {:.3g} -> {:.3g} 1/s（陷阱项占比 {:.2f} -> {:.2f}）".format(
            T_PRE, t[-1], "，击穿过程之前" if b is not None else "", J[0], J[-1], G[0], G[-1], Gt[0] / G[0], Gt[-1] / G[-1]),
        "  N ∝ t^{:.2f}（R² = {:.3f}，第一个空位 t = {:.3g} s）；I-t n = {:.2f}，R² = {:.3f}；前半段 n = {:.2f}，后半段 n = {:.2f}".format(
            nN, rN, d["t_s"][np.argmax(d["n_generated"] >= 1)], nJ, r2, n1, n2),
    ]
    if b is not None:
        lines += [
            "  击穿过程：从第 {} 个空位（z = {:.2f} nm，接上负极导体，t = {:.6g} s）起，负极导体最高点 {:.2f} -> {:.2f} nm（这一步）"
            "-> {:.2f} nm（贯穿前），其间抬高最高点的空位 {} 个；接上时最大悬空导电簇 {} -> {} 个空位，ΣG {:.3g} -> {:.3g} 1/s；"
            "再过 {} 个空位、{} s 贯穿；其间间隙隧穿最大 {:.2g} A/cm2，缺陷处最高 {:.0f} K".format(
                int(d["n_generated"][k]), d["birth_z_nm"][k], d["t_s"][k], d["negative_front_nm"][k - 1],
                d["negative_front_nm"][k], d["negative_front_nm"][b - 1],
                int(np.sum(np.diff(d["negative_front_nm"][k - 1:b]) > 0)), int(d["max_floating"][k - 1]),
                int(d["max_floating"][k]), d["G_total_s"][k - 1], d["G_total_s"][k], b - k, elapsed(d["t_s"], k, b),
                d["J_gap_A_cm2"][k:b].max(), d["T_defect_max_K"][k:b].max()),
            "  击穿：J {:.3g} -> {:.3g} A/cm2（内层输运 {:.3g}、间隙隧穿 {:.3g}、细丝 {:.3g}）；接负极导体 {} 个成员、接正极 {} 个；"
            "击穿后缺陷处最高 {:.3g} K".format(
                d["J_A_cm2"][k - 1], d["J_A_cm2"][b], d["J_transport_A_cm2"][b], d["J_gap_A_cm2"][b], d["J_filament_A_cm2"][b],
                int(d["n_negative_members"][b]), int(d["n_positive_members"][b]), d["T_defect_max_K"][b:].max()),
        ]
    lines += [
        "  幂律阶段末：接负极导电簇 {} 个成员，最高点 {:.2f} nm；导体间隙 {:.2f} nm；最大悬空导电簇 {} 个空位".format(
            int(p["n_negative_members"][-1]), p["negative_front_nm"][-1], p["gap_nm"][-1], int(p["max_floating"].max())),
        "  对照（E_A 标准差 0.05 eV，sep042，同样从 {:g} s 到击穿过程之前）：n = {:.2f}，R² = {:.3f}；前半段 n = {:.2f}，"
        "后半段 n = {:.2f}".format(T_PRE, onJ, or2, on1, on2),
        "  幂律阶段出生高度：中位数 {:.2f} nm，90% 分位 {:.2f} nm，最高 {:.2f} nm；负极 0.6 nm 以内 {:.1%}，z > 2 nm {:.1%}".format(
            np.median(z), np.quantile(z, 0.9), z.max(), np.mean(z < 0.6), np.mean(z > 2.0)),
        "  幂律阶段缺陷处最高升温 {:.3f} K；间隙隧穿最大 {:.2g} A/cm2".format(
            p["T_defect_max_K"].max() - AMBIENT, p["J_gap_A_cm2"].max()),
        "  平均场：没收敛 {} 条，用牛顿法 {} 条；热迭代没收敛 {} 条".format(
            int((d["mf_converged"] == 0).sum()), int((d["mf_newton_steps"] > 0).sum()), int((d["thermal_converged"] == 0).sum())),
        "  每步耗时：中位数 {:.0f} s，最后 20 步平均 {:.0f} s；峰值工作集 {:.0f} MB".format(
            np.median(steps_wall), steps_wall[-20:].mean(), d["ws_peak_MB"].max()),
    ]
text = "\n".join(lines)
print(text)
open(os.path.join(HERE, "summary.txt"), "w", encoding="utf-8").write(text + "\n")

fig, axes = plt.subplots(2, 3, figsize=(183 * MM, 100 * MM), constrained_layout=True)
shown = {s: from_pre_stress_end(new[s]) for s in SEEDS}  # 0.1 s 到终点（含击穿过程与击穿后）

ax = axes[0, 0]
for s in SEEDS:
    q = shown[s]
    ax.loglog(q["t_s"], q["J_A_cm2"], color=COLORS[s], label="seed {}".format(s), drawstyle="steps-post")
    b = seg[s][1]
    if b is not None:
        ax.loglog(new[s]["t_s"][b], new[s]["J_A_cm2"][b], ls="none", marker="o", ms=4, mfc="none", mec=COLORS[s], mew=1.0)
if any(seg[s][1] is not None for s in SEEDS):
    ax.plot([], [], ls="none", marker="o", ms=4, mfc="none", mec="k", mew=1.0, label="Breakdown")
ax.set_xlabel("Stress time (s)")
ax.set_ylabel("J (A cm$^{-2}$)")
ax.legend(frameon=False, loc="upper left")

ax = axes[0, 1]
for s in SEEDS:
    q = shown[s]
    m = q["n_generated"] >= 1
    ax.loglog(q["t_s"][m], q["n_generated"][m], color=COLORS[s], label="seed {}".format(s), drawstyle="steps-post")
ax.set_xlabel("Stress time (s)")
ax.set_ylabel("Generated vacancies N")
ax.legend(frameon=False)

ax = axes[0, 2]
for s in SEEDS:
    p = seg[s][0]
    tt, J = p["t_s"], p["J_transport_A_cm2"]
    grid = 10.0 ** np.arange(np.log10(tt[0]), np.log10(tt[-1]), 0.25)
    lj = np.log(held(tt, J, grid))
    ax.semilogx(np.sqrt(grid[1:] * grid[:-1]), np.diff(lj) / np.diff(np.log(grid)), color=COLORS[s], marker="o", ms=1.5,
                lw=0.6, label="seed {}".format(s))
ax.set_xlabel("Stress time (s)")
ax.set_ylabel(r"Local slope d ln$J$/d ln$t$")
ax.legend(frameon=False)

ax = axes[1, 0]
for s in SEEDS:
    q = shown[s]
    finite = np.isfinite(q["G_total_s"])  # 击穿后最后一条不求生成率
    ax.loglog(q["t_s"][finite], q["G_total_s"][finite], color=COLORS[s], label="seed {}".format(s), drawstyle="steps-post")
ax.set_xlabel("Stress time (s)")
ax.set_ylabel(r"$\Sigma G$ (s$^{-1}$)")
ax.legend(frameon=False, loc="upper left")

ax = axes[1, 1]
for s in SEEDS:
    q = shown[s]
    ax.plot(q["n_generated"], q["n_negative_members"], color=COLORS[s], label="seed {}".format(s))
ax.set_xlabel("Generated vacancies N")
ax.set_ylabel("Negative-conductor members")
twin = ax.twinx()
for s in SEEDS:
    q = shown[s]
    twin.plot(q["n_generated"], q["negative_front_nm"], color=COLORS[s], ls="--", lw=0.6)
twin.set_ylabel("Front height (nm, dashed)")
twin.spines["right"].set_visible(True)
ax.legend(frameon=False, loc="upper left")

ax = axes[1, 2]
bins = np.arange(0.0, 8.01, 0.2)
for s in SEEDS:
    ax.hist(births(seg[s][0]), bins=bins, histtype="step", color=COLORS[s], label="seed {}".format(s), lw=0.8)
ax.set_xlabel("Birth height z (nm)")
ax.set_ylabel("Count")
ax.legend(frameon=False)

for a, letter in zip(axes.flat, "abcdef"):
    a.text(-0.02, 1.02, letter, transform=a.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "it_sigma15.{}".format(ext)))
print("wrote it_sigma15.pdf, it_sigma15.png, summary.txt")
