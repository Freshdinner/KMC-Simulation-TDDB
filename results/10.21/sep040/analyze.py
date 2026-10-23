"""最小间隔 0.4 nm、t₀ = 2.0 eV（r_c = 0.65 nm）的 3.6 V 测试汇总（其余同 examples/zro2_tddb.yaml，seed 1、2，`drive.py`；各跑 10 h 后
从存档续跑到击穿）。对照是最小间隔 0.26 nm、t₀ = 0.5 eV（r_c = 0.42 nm）的同种子轨迹（`../../10.19/accept/`，ADR-063/064，未击穿）。

每条轨迹分四段，时间原点不变（从应力开始计）：
- pre-stress 阶段：t < T_PRE = 0.1 s（实验的 I-t 从 0.1 s 起），只在 summary.txt 里记生成的空位数与 J、ΣG 的首末值，不画进图线；
- 幂律阶段：0.1 s 到悬空导电簇接上负极之前（负极导体最高点单步跳升最大的那条记录之前），I-t 拟合只用这一段的内层输运电流。
  0.1 s 这一点取此前最后一条记录的状态（两次生成之间状态不变）；
- 击穿过程：悬空导电簇接上负极到贯穿通路形成（第一条有细丝电流的记录，给出 t_BD），与 t_BD 只差 10⁻⁵ s 量级；
- 击穿后：再跑的 post_breakdown_steps 条记录（内存保护停跑时不足 20 条）。

统计写到 summary.txt。出图 it_sep040.pdf / .png，版式同 `../../10.19/accept/it_accept`（Nature 版式：双栏 183 mm、7 pt；
Times New Roman + stix 公式、蓝红对比色、坐标轴框线与刻度线 1.5 pt），图线从 0.1 s 画起：(a) 总电流密度 J(t)（内层输运 + 间隙隧穿 + 细丝，
空心圈标击穿点）；(b) N(t)；(c) 幂律阶段的局域斜率 d lnJ/d ln t（内层输运电流，log t 上每个数量级 4 个点）；(d) ΣG(t)；
(e) 接负极导电簇的成员数与最高点；(f) 幂律阶段新空位的出生高度分布。图中应力时间是模拟原值，不平移。
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
OLD = os.path.join(HERE, "..", "..", "10.19", "accept")
SEEDS = (1, 2)
COLORS = {1: "#1f77b4", 2: "#d62728"}
T_PRE = 0.1  # pre-stress 阶段的终点 [s]
POST_BREAKDOWN_STEPS = 20  # examples/zro2_tddb.yaml 的 post_breakdown_steps
AMBIENT = 298.15  # 25 °C [K]


def steps(folder, seed):
    """逐步数据：`steps_s{SEED}.csv` 里续跑起点之前的行 + `steps_s{SEED}_from{N}.csv`。"""
    rows = list(csv.DictReader(open(os.path.join(folder, "steps_s{}.csv".format(seed)), encoding="utf-8")))
    for resumed in sorted(glob.glob(os.path.join(folder, "steps_s{}_from*.csv".format(seed)))):
        start = int(resumed.rsplit("from", 1)[1][:4])
        rows = [r for r in rows if int(r["n_generated"]) < start] + list(csv.DictReader(open(resumed, encoding="utf-8")))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def log_fit(t, y):
    """log t 上等间距取样后的双对数直线拟合：(斜率, R²)。"""
    grid = np.logspace(np.log10(t[0]), np.log10(t[-1]), 60)
    yi = np.exp(np.interp(np.log(grid), np.log(t), np.log(y)))
    p = np.polyfit(np.log(grid), np.log(yi), 1)
    r2 = np.corrcoef(np.log(grid), np.log(yi))[0, 1] ** 2
    return p[0], r2


def rows(d, start, stop=None):
    """第 start 到 stop（不含）条记录。"""
    return {k: v[start:stop].copy() for k, v in d.items()}


def from_pre_stress_end(d):
    """t ≥ T_PRE 的记录：第一行是 t = T_PRE 时的状态，即此前最后一条记录（两次生成之间状态不变），时间改记 T_PRE。"""
    stage = rows(d, int(np.searchsorted(d["t_s"], T_PRE, side="right")) - 1)
    stage["t_s"][0] = T_PRE
    return stage


def it_fit(stage):
    """I-t（内层输运电流）的双对数拟合：(n, R², 前半段 n, 后半段 n)，前后半段在 log t 上对分。"""
    t, J = stage["t_s"], stage["J_transport_A_cm2"]
    n, r2 = log_fit(t, J)
    half = np.sqrt(t[0] * t[-1])
    return n, r2, log_fit(t[t <= half], J[t <= half])[0], log_fit(t[t >= half], J[t >= half])[0]


def births(stage):
    """这一段里出生的新空位的高度 [nm]：第一行之后各条记录的出生位置（续跑起点那条记录没有出生位置，记 nan，去掉）。"""
    z = stage["birth_z_nm"][1:]
    return z[np.isfinite(z)]


def last_log_line(seed):
    """这个种子最后一段日志的最后一行。"""
    return open(sorted(glob.glob(os.path.join(HERE, "logs", "run_s{}*.log".format(seed))))[-1],
                encoding="utf-8").read().splitlines()[-1]


new = {s: steps(HERE, s) for s in SEEDS}
old = {s: steps(OLD, s) for s in SEEDS}
breakdown = {s: int(np.argmax(new[s]["J_filament_A_cm2"] > 0)) for s in SEEDS}  # 第一条有细丝电流的记录
# 悬空导电簇接上负极：击穿前负极导体最高点单步跳升最大的那条记录
connect = {s: int(np.argmax(np.diff(new[s]["negative_front_nm"][:breakdown[s] + 1]))) + 1 for s in SEEDS}
power_law = {s: from_pre_stress_end(rows(new[s], 0, connect[s])) for s in SEEDS}

lines = []
for s in SEEDS:
    d, p, b, k = new[s], power_law[s], breakdown[s], connect[s]
    t, N, J = p["t_s"], p["n_generated"], p["J_transport_A_cm2"]
    nN, _ = log_fit(t, N)
    nJ, r2, n1, n2 = it_fit(p)
    onJ, or2, on1, on2 = it_fit(from_pre_stress_end(old[s]))
    G, Gt = p["G_total_s"], p["G_trap_s"]
    z = births(p)
    after = len(d["t_s"]) - 1 - b
    stop = ("击穿后再跑 {} 条记录后停".format(after) if after >= POST_BREAKDOWN_STEPS
            else "击穿后跑到第 {} / {} 条记录时内存保护停跑".format(after, POST_BREAKDOWN_STEPS))
    steps_wall = np.diff(d["wall_s"])
    steps_wall = steps_wall[steps_wall > 0]  # 去掉续跑起点（机时从 0 重新计）
    lines += [
        "== seed {}: 击穿，t_BD = {:.6g} s，第 {} 个空位；{}（{}）".format(
            s, d["t_s"][b], int(d["n_generated"][b]), stop, last_log_line(s).split(" n_generated")[0].split(" end_time")[0]),
        "  pre-stress（t < {:g} s）：生成 {} 个空位；J {:.3g} -> {:.3g} A/cm2；ΣG {:.3g} -> {:.3g} 1/s".format(
            T_PRE, int(N[0]), d["J_transport_A_cm2"][0], J[0], d["G_total_s"][0], G[0]),
        "  幂律阶段（{:g} s 到 {:.6g} s，悬空导电簇接上负极之前）：J {:.3g} -> {:.3g} A/cm2；ΣG {:.3g} -> {:.3g} 1/s（陷阱项占比 {:.2f} -> {:.2f}）".format(
            T_PRE, t[-1], J[0], J[-1], G[0], G[-1], Gt[0] / G[0], Gt[-1] / G[-1]),
        "  N ∝ t^{:.2f}；I-t n = {:.2f}，R² = {:.3f}；前半段 n = {:.2f}，后半段 n = {:.2f}".format(nN, nJ, r2, n1, n2),
        "  击穿过程：第 {} 个空位（z = {:.2f} nm）把悬空导电簇接到负极，负极导体最高点 {:.2f} -> {:.2f} nm，导体间隙 {:.2f} -> {:.2f} nm，"
        "ΣG {:.3g} -> {:.3g} 1/s；再过 {} 个空位、{:.2g} s 贯穿；其间间隙隧穿最大 {:.2g} A/cm2，缺陷处最高 {:.0f} K".format(
            int(d["n_generated"][k]), d["birth_z_nm"][k], d["negative_front_nm"][k - 1], d["negative_front_nm"][k],
            d["gap_nm"][k - 1], d["gap_nm"][k], d["G_total_s"][k - 1], d["G_total_s"][k], b - k, d["t_s"][b] - d["t_s"][k],
            d["J_gap_A_cm2"][k:b].max(), d["T_defect_max_K"][k:b].max()),
        "  击穿：J {:.3g} -> {:.3g} A/cm2（内层输运 {:.3g}、间隙隧穿 {:.3g}、细丝 {:.3g}）；接负极导体 {} 个成员、接正极 {} 个；"
        "击穿后缺陷处最高 {:.3g} K".format(
            d["J_A_cm2"][k - 1], d["J_A_cm2"][b], d["J_transport_A_cm2"][b], d["J_gap_A_cm2"][b], d["J_filament_A_cm2"][b],
            int(d["n_negative_members"][b]), int(d["n_positive_members"][b]), d["T_defect_max_K"][b:].max()),
        "  幂律阶段末：接负极导电簇 {} 个成员，最高点 {:.2f} nm；最大悬空导电簇 {} 个空位".format(
            int(p["n_negative_members"][-1]), p["negative_front_nm"][-1], int(p["max_floating"].max())),
        "  对照（最小间隔 0.26 nm、t₀ = 0.5 eV，10.19，未击穿，同样从 {:g} s 起）：n = {:.2f}，R² = {:.3f}；前半段 n = {:.2f}，后半段 n = {:.2f}；"
        "{} 个空位用了 {:.3g} s".format(T_PRE, onJ, or2, on1, on2, int(old[s]["n_generated"][-1]), old[s]["t_s"][-1]),
        "  幂律阶段出生高度：中位数 {:.2f} nm，90% 分位 {:.2f} nm，最高 {:.2f} nm；负极 0.6 nm 以内 {:.1%}，z > 2 nm {:.1%}".format(
            np.median(z), np.quantile(z, 0.9), z.max(), np.mean(z < 0.6), np.mean(z > 2.0)),
        "  幂律阶段缺陷处最高升温 {:.3f} K；间隙隧穿最大 {:.2g} A/cm2".format(
            p["T_defect_max_K"].max() - AMBIENT, p["J_gap_A_cm2"].max()),
        "  平均场：没收敛 {} 条，用牛顿法 {} 条；热迭代没收敛 {} 条".format(
            int((d["mf_converged"] == 0).sum()), int((d["mf_newton_steps"] > 0).sum()), int((d["thermal_converged"] == 0).sum())),
        "  每步耗时：中位数 {:.0f} s，击穿过程与击穿后平均 {:.0f} s；峰值工作集 {:.0f} MB".format(
            np.median(steps_wall), steps_wall[-(len(d["t_s"]) - k):].mean(), d["ws_peak_MB"].max()),
    ]
text = "\n".join(lines)
print(text)
open(os.path.join(HERE, "summary.txt"), "w", encoding="utf-8").write(text + "\n")

fig, axes = plt.subplots(2, 3, figsize=(183 * MM, 100 * MM), constrained_layout=True)
shown = {s: from_pre_stress_end(new[s]) for s in SEEDS}  # 0.1 s 到终点（含击穿过程与击穿后）

ax = axes[0, 0]
for s in SEEDS:
    q = shown[s]
    ax.loglog(q["t_s"], q["J_A_cm2"], color=COLORS[s], label="seed {}".format(s))
    b = breakdown[s]
    ax.loglog(new[s]["t_s"][b], new[s]["J_A_cm2"][b], ls="none", marker="o", ms=4, mfc="none", mec=COLORS[s], mew=1.0)
ax.plot([], [], ls="none", marker="o", ms=4, mfc="none", mec="k", mew=1.0, label="Breakdown")
ax.set_xlabel("Stress time (s)")
ax.set_ylabel("J (A cm$^{-2}$)")
ax.legend(frameon=False, loc="upper left")

ax = axes[0, 1]
for s in SEEDS:
    q = shown[s]
    ax.loglog(q["t_s"], q["n_generated"], color=COLORS[s], label="seed {}".format(s))
end = power_law[SEEDS[0]]
t = np.logspace(np.log10(T_PRE), np.log10(end["t_s"][-1]), 10)
ax.loglog(t, end["n_generated"][-1] * (t / end["t_s"][-1]) ** (0.0257 / 0.086), color="k", lw=0.5, ls=":",
          label=r"$t^{k_BT/E_0}$ = $t^{0.30}$")
ax.set_xlabel("Stress time (s)")
ax.set_ylabel("Generated vacancies N")
ax.legend(frameon=False)

ax = axes[0, 2]
for s in SEEDS:
    p = power_law[s]
    tt, J = p["t_s"], p["J_transport_A_cm2"]
    grid = 10.0 ** np.arange(np.log10(tt[0]), np.log10(tt[-1]), 0.25)
    lj = np.interp(np.log(grid), np.log(tt), np.log(J))
    ax.semilogx(np.sqrt(grid[1:] * grid[:-1]), np.diff(lj) / np.diff(np.log(grid)), color=COLORS[s], marker="o", ms=1.5,
                lw=0.6, label="seed {}".format(s))
ax.axhline(0.30, color="k", lw=0.5, ls=":")
ax.set_xlabel("Stress time (s)")
ax.set_ylabel(r"Local slope d ln$J$/d ln$t$")
ax.legend(frameon=False)

ax = axes[1, 0]
for s in SEEDS:
    q = shown[s]
    finite = np.isfinite(q["G_total_s"])  # 击穿后最后一条不求生成率
    ax.loglog(q["t_s"][finite], q["G_total_s"][finite], color=COLORS[s], label="seed {}".format(s))
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
    ax.hist(births(power_law[s]), bins=bins, histtype="step", color=COLORS[s], label="seed {}".format(s), lw=0.8)
ax.set_xlabel("Birth height z (nm)")
ax.set_ylabel("Count")
ax.legend(frameon=False)

for a, letter in zip(axes.flat, "abcdef"):
    a.text(-0.02, 1.02, letter, transform=a.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "it_sep040.{}".format(ext)))
print("wrote it_sep040.pdf, it_sep040.png, summary.txt")
