"""ADR-063/064（E_A 指数尾 E₀ = 0.086 eV、导电簇按电子离域判据连边 t₀ = 0.5 eV）之后的 3.6 V 验收汇总（examples/zro2_tddb.yaml 原样，
seed 1、2，各 10 h 机时，`drive.py`）。对照是 E_A 全部取 1.6 eV、r_c = 0.3 nm 的同种子轨迹（`../../10.17/accept/`，ADR-062）。

统计写到 summary.txt：空位数与应力时间、J 与 ΣG 的首末值、N(t) 指数、I-t 双对数拟合（内层输运电流，第一次生成到终点，按 log t 等间距取样；
前后半段斜率看弯曲方向）、接负极导电簇的成员数与最高点、导体间隙、新空位出生高度、平均场与热迭代收敛、每步耗时与峰值工作集。
出图 it_accept.pdf / .png（Nature 版式：双栏 183 mm、7 pt；Times New Roman + stix 公式、蓝红对比色、坐标轴框线与刻度线 1.5 pt），
只画指数尾的结果：(a) J(t)；(b) N(t)；(c) 局域斜率 d lnJ/d ln t；(d) ΣG(t)；(e) 接负极导电簇的成员数与最高点；
(f) 新空位出生高度分布。图中应力时间统一乘 T_SHIFT = 1e3（向右平移 3 个数量级），summary.txt 里的时间是模拟原值。
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
OLD = os.path.join(HERE, "..", "..", "10.17", "accept")
SEEDS = (1, 2)
COLORS = {1: "#1f77b4", 2: "#d62728"}
T_SHIFT = 1e3  # 出图时应力时间乘的倍数（向右平移 3 个数量级）


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


new = {s: steps(HERE, s) for s in SEEDS}
old = {s: steps(OLD, s) for s in SEEDS}

lines = []
for s in SEEDS:
    d = new[s]
    m = d["n_generated"] >= 1
    t, N, J = d["t_s"][m], d["n_generated"][m], d["J_transport_A_cm2"][m]
    nN, _ = log_fit(t, N)
    nJ, r2 = log_fit(t, J)
    half = np.sqrt(t[0] * t[-1])
    n1, _ = log_fit(t[t <= half], J[t <= half])
    n2, _ = log_fit(t[t >= half], J[t >= half])
    o = old[s]
    om = o["n_generated"] >= 1
    onJ, or2 = log_fit(o["t_s"][om], o["J_transport_A_cm2"][om])
    G, Gt = d["G_total_s"], d["G_trap_s"]
    z = d["birth_z_nm"][1:]
    steps_wall = np.diff(d["wall_s"])
    lines += [
        "== seed {}: {} 个空位，t = {:.3g} s（10 h 机时到点停，未击穿）".format(s, int(d["n_generated"][-1]), d["t_s"][-1]),
        "  J {:.3g} -> {:.3g} A/cm2；ΣG {:.3g} -> {:.3g} 1/s（陷阱项占比 {:.2f} -> {:.2f}）".format(
            J[0], J[-1], G[0], G[-1], Gt[0] / G[0], Gt[-1] / G[-1]),
        "  N ∝ t^{:.2f}；I-t n = {:.2f}，R² = {:.3f}；前半段 n = {:.2f}，后半段 n = {:.2f}".format(nN, nJ, r2, n1, n2),
        "  对照（E_A 全部 1.6 eV、r_c = 0.3 nm，10.17）：n = {:.2f}，R² = {:.3f}，{} 个空位用了 {:.3g} s".format(
            onJ, or2, int(o["n_generated"][-1]), o["t_s"][-1]),
        "  接负极导电簇：成员 {} 个，最高点 {:.2f} nm；接正极 {} 个；导体间隙 {:.2f} nm；最大悬空导电簇 {} 个空位".format(
            int(d["n_negative_members"][-1]), d["negative_front_nm"].max(), int(d["n_positive_members"][-1]),
            d["gap_nm"].min(), int(d["max_floating"].max())),
        "  出生高度：中位数 {:.2f} nm，90% 分位 {:.2f} nm，最高 {:.2f} nm；负极 0.6 nm 以内 {:.1%}，z > 2 nm {:.1%}".format(
            np.median(z), np.quantile(z, 0.9), z.max(), np.mean(z < 0.6), np.mean(z > 2.0)),
        "  缺陷处最高升温 {:.3f} K；间隙隧穿最大 {:.2g} A/cm2".format(d["T_defect_max_K"].max() - 298.15, d["J_gap_A_cm2"].max()),
        "  平均场：没收敛 {} 条，用牛顿法 {} 条；热迭代没收敛 {} 条".format(
            int((d["mf_converged"] == 0).sum()), int((d["mf_newton_steps"] > 0).sum()), int((d["thermal_converged"] == 0).sum())),
        "  每步耗时：中位数 {:.0f} s，最后 20 步平均 {:.0f} s；峰值工作集 {:.0f} MB".format(
            np.median(steps_wall), steps_wall[-20:].mean(), d["ws_peak_MB"].max()),
    ]
text = "\n".join(lines)
print(text)
open(os.path.join(HERE, "summary.txt"), "w", encoding="utf-8").write(text + "\n")

fig, axes = plt.subplots(2, 3, figsize=(183 * MM, 100 * MM), constrained_layout=True)
ax = axes[0, 0]
for s in SEEDS:
    d = new[s]
    m = d["n_generated"] >= 1
    ax.loglog(d["t_s"][m] * T_SHIFT, d["J_transport_A_cm2"][m], color=COLORS[s], label="seed {}".format(s))
ax.set_xlabel("Stress time (s)")
ax.set_ylabel("J (A cm$^{-2}$)")
ax.legend(frameon=False)

ax = axes[0, 1]
for s in SEEDS:
    d = new[s]
    m = d["n_generated"] >= 1
    ax.loglog(d["t_s"][m] * T_SHIFT, d["n_generated"][m], color=COLORS[s], label="seed {}".format(s))
t = np.logspace(-5, 1, 10) * T_SHIFT
ax.loglog(t, 900 * (t / (8.0 * T_SHIFT)) ** (0.0257 / 0.086), color="k", lw=0.5, ls=":",
          label=r"$t^{k_BT/E_0}$ = $t^{0.30}$")
ax.set_xlabel("Stress time (s)")
ax.set_ylabel("Generated vacancies N")
ax.legend(frameon=False)

ax = axes[0, 2]
for s in SEEDS:
    d = new[s]
    m = d["n_generated"] >= 1
    tt, J = d["t_s"][m], d["J_transport_A_cm2"][m]
    grid = np.logspace(np.log10(tt[0]), np.log10(tt[-1]), 40)
    lj = np.interp(np.log(grid), np.log(tt), np.log(J))
    ax.semilogx(np.sqrt(grid[1:] * grid[:-1]) * T_SHIFT, np.diff(lj) / np.diff(np.log(grid)), color=COLORS[s], marker="o", ms=1.5,
                lw=0.6, label="seed {}".format(s))
ax.axhline(0.30, color="k", lw=0.5, ls=":")
ax.set_xlabel("Stress time (s)")
ax.set_ylabel(r"Local slope d ln$J$/d ln$t$")
ax.legend(frameon=False)

ax = axes[1, 0]
for s in SEEDS:
    d = new[s]
    m = d["n_generated"] >= 1
    ax.loglog(d["t_s"][m] * T_SHIFT, d["G_total_s"][m], color=COLORS[s], label="seed {}".format(s))
ax.set_xlabel("Stress time (s)")
ax.set_ylabel(r"$\Sigma G$ (s$^{-1}$)")
ax.legend(frameon=False, loc="lower left")

ax = axes[1, 1]
for s in SEEDS:
    d = new[s]
    ax.plot(d["n_generated"], d["n_negative_members"], color=COLORS[s], label="seed {}".format(s))
ax.set_xlabel("Generated vacancies N")
ax.set_ylabel("Negative-conductor members")
twin = ax.twinx()
for s in SEEDS:
    d = new[s]
    twin.plot(d["n_generated"], d["negative_front_nm"], color=COLORS[s], ls="--", lw=0.6)
twin.set_ylabel("Front height (nm, dashed)")
twin.set_ylim(0, 8)
twin.spines["right"].set_visible(True)
ax.legend(frameon=False, loc="upper left")

ax = axes[1, 2]
bins = np.arange(0.0, 8.01, 0.2)
for s in SEEDS:
    ax.hist(new[s]["birth_z_nm"][1:], bins=bins, histtype="step", color=COLORS[s], label="seed {}".format(s), lw=0.8)
ax.set_xlabel("Birth height z (nm)")
ax.set_ylabel("Count")
ax.legend(frameon=False)

for a, letter in zip(axes.flat, "abcdef"):
    a.text(-0.02, 1.02, letter, transform=a.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "it_accept.{}".format(ext)))
print("wrote it_accept.pdf, it_accept.png, summary.txt")
