"""ADR-055 验收汇总：前驱位点按最小间隔 0.26 nm 铺满（不设浓度）、r_c = 0.3 nm，其余同 ADR-054（3.6 V / 25 °C，E_A = 1.30 eV）。

统计：终止原因与 t_BD、击穿时的空位数、击穿前后的电流与跳升倍数、I-t 双对数拟合（内层输运电流，从第一次生成起到终点，按 log t 等间距
取样；前后半段斜率看弯曲方向）、N(t) 的幂律指数、导体间隙降到 3 / 2 / 1 nm 的时刻、从 ΣG 最后一次低于 10·ΣG(0) 到击穿的时间与空位数、
新空位深度与接到阴极导电簇的比例、ΣG 的首末值、没收敛的平均场记录，以及 t_BD = 1000 s 需要的 E_A（P_bb 未饱和时 E_A 平移 ΔE
让时刻乘 exp(ΔE/k_BT)）。最后一个快照上另求：现有缺陷形成贯穿通路所需的最小 r_c（二分），与 r_c 取 0.3–0.8 nm 时接阴极导电簇的最高点。
日志里没有 done 行的轨迹记为手动停止（ADR-055 的三条约 13 h 后停）。
出图 it_dense.png：J(t)（含击穿时的细丝电流）、导体间隙 gap(t)、ΣG(t)、J 对空位数；path_dense.png：击穿状态（或最后一个快照）的缺陷
x–z、y–z 投影，贯穿通路标红、接阴极 / 阳极的导电簇分色。
usage: analyze.py
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.tddb import conducting_clusters, fit_power_law  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False
SEEDS = (1, 2, 3)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # 与 ../../10.12/percolation/analyze.py 相同，按种子固定
KT_EV = 298.15 * 8.617333262e-5  # 25 °C 下的 k_BT [eV]
E_A = 1.30  # [eV]
LZ = 8.0  # [nm]
R_C = 0.3  # [nm]
NM = 1e-9
DEVICE = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml")).device


def load(seed):
    path = os.path.join(HERE, "steps_s{}.csv".format(seed))
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    column = lambda name: np.array([float(r[name]) for r in rows])  # noqa: E731
    log = os.path.join(HERE, "logs", "run_s{}.log".format(seed))
    lines = open(log, encoding="utf-8").read().splitlines() if os.path.exists(log) else []
    done = [line for line in lines if line.startswith("done")]
    sites = [line for line in lines if line.startswith("precursor sites")]
    return dict(t=column("t_s"), J=column("J_A_cm2"), J_tat=column("J_transport_A_cm2"), G=column("G_total_s"),
                G_trap=column("G_trap_s"), gap=column("gap_nm"), front=column("cathode_front_nm"),
                n_cath=column("n_cathode_members"), z=column("birth_z_nm")[1:], side=column("birth_side")[1:],
                conv=column("mf_converged"), newton=column("mf_newton_steps"), residual=column("mf_residual"),
                wall=column("wall_s"), rates_wall=column("rates_wall_s"), dT=column("T_peak_K") - 298.15,
                peak=column("ws_peak_MB"), stop=done[0].split()[1] if done else "stopped by hand",
                end=float(done[0].split("end_time=")[1].split()[0]) if done else None,
                sites=sites[0] if sites else "")


def log_t_fit(t, J, start, end):
    """分段常数 J(t) 按 log t 等间距取样（与 fit_transient 同口径），窗口 [start, end]；返回 (整段, 前半, 后半) 拟合。"""
    times = np.logspace(np.log10(start), np.log10(end), max(4, int(np.ceil(np.log10(end / start) * 20)) + 1))
    sample = J[np.clip(np.searchsorted(t, times, side="right") - 1, 0, None)]
    half = times.size // 2
    return fit_power_law(times, sample), fit_power_law(times[:half], sample[:half]), fit_power_law(times[half:], sample[half:])


lines, data = [], {}
for seed in SEEDS:
    d = load(seed)
    if d is None:
        continue
    data[seed] = d
    t, J, J_tat, G, gap = d["t"], d["J"], d["J_tat"], d["G"], d["gap"]
    end = d["end"] if d["end"] is not None else t[-1]
    n = t.size - 1
    broke = d["stop"] == "breakdown"
    lines.append("== seed {}: {} at t_end = {:.4g} s, {} vacancies; J0 {:.3f} A/cm2; {}".format(
        seed, d["stop"], end, n, J[0], d["sites"]))
    if broke:
        lines.append("  breakdown: J before {:.3f} A/cm2 (TAT), at t_BD {:.4g} A/cm2 (TAT + G0 V), jump x{:.3g}".format(
            J_tat[-2], J[-1], J[-1] / J_tat[-2]))
        quiet = np.flatnonzero(G[:-1] < 10.0 * G[0])
        if quiet.size:
            k = quiet[-1]
            lines.append("  last stretch: from the last record with sum G < 10 x sum G(0) ({:.3g} 1/s) to breakdown: {:.3g} s, "
                         "{} vacancies, sum G up to {:.3g} 1/s".format(G[k], end - t[k], n - k, np.nanmax(G[k:-1])))
    reach = []
    for level in (3.0, 2.0, 1.0):
        k = np.flatnonzero(gap <= level)
        reach.append("{:.4g} s (n = {})".format(t[k[0]], k[0]) if k.size else "-")
    lines.append("  conductor gap: t = 0 {:.2f} nm, last before end {:.2f} nm; gap <= 3 / 2 / 1 nm first at: {}".format(
        gap[0], gap[-2] if broke else gap[-1], " / ".join(reach)))
    lines.append("  cathode conductor: members {} -> {}, front {:.2f} -> {:.2f} nm".format(
        int(d["n_cath"][0]), int(d["n_cath"][-1]), d["front"][0], d["front"][-1]))
    G_last = G[-2] if broke else G[-1]
    if n >= 1:
        lines.append("  sum G first {:.3g}, last {:.3g} 1/s (trap share last {:.2f}); births: z median {:.2f} nm, 90% {:.2f} nm, "
                     "max {:.2f} nm; joined cathode conductor {:.0f}%".format(
                         G[0], G_last, (d["G_trap"][-2] if broke else d["G_trap"][-1]) / G_last,
                         *np.percentile(d["z"], [50, 90, 100]), 100 * np.mean((d["side"] == 0) | (d["side"] == 2))))
    lines.append("  mean field converged {}/{} (Newton used in {} records); median step wall {:.1f} s (rates {:.1f} s); "
                 "max dT {:.1e} K; peak working set {:.0f} MB".format(
                     int(d["conv"].sum()), d["conv"].size, int(np.sum(d["newton"] > 0)), float(np.median(np.diff(d["wall"]))),
                     float(np.nanmedian(d["rates_wall"])), d["dT"].max(), d["peak"].max()))
    if n >= 4:
        whole, first, second = log_t_fit(t, J_tat, t[1], end)
        lines.append("  I-t fit (transport current, first birth to t_end): n = {:.2f}, R² = {:.3f}; first half n = {:.2f}, "
                     "second half n = {:.2f}".format(whole.exponent, whole.r_squared, first.exponent, second.exponent))
        N = np.arange(t.size)
        fit = fit_power_law(t[1:], N[1:])
        lines.append("  N(t) ∝ t^{:.2f} (R² = {:.3f})".format(fit.exponent, fit.r_squared))
    if broke:
        lines.append("  t_BD = {:.4g} s -> E_A for t_BD = 1000 s: {:.3f} eV".format(end, E_A - KT_EV * np.log(end / 1000.0)))
    bad = np.flatnonzero(d["conv"] == 0)
    if bad.size:
        lines.append("  not converged: records {} (residual {})".format(
            bad.tolist(), ", ".join("{:.1e}".format(v) for v in d["residual"][bad])))
    if np.any(J >= 1.0) and np.any(J >= 10.0):
        up, ten = np.flatnonzero(J >= 1.0)[0], np.flatnonzero(J >= 10.0)[0]
        lines.append("  J reaches 1 A/cm2 at t = {:.3g} s (n = {}), 10 A/cm2 at t = {:.3g} s (n = {})".format(t[up], up, t[ten], ten))
    snaps = sorted(f for f in os.listdir(HERE) if f.startswith("snap_s{}_n".format(seed)))
    snap = np.load(os.path.join(HERE, snaps[-1]))
    positions = snap["positions"]
    lo, hi = R_C * NM, 4.0 * NM
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if np.any(conducting_clusters(positions, DEVICE, mid) == 2) else (mid, hi)
    fronts = []
    for r in (0.3, 0.4, 0.5, 0.6, 0.8):
        sides = conducting_clusters(positions, DEVICE, r * NM)
        cathode = (sides == 0) | (sides == 2)
        fronts.append("{:.1f}: {:.2f}".format(r, positions[cathode, 2].max() / NM if cathode.any() else 0.0))
    lines.append("  last snapshot ({} vacancies): spanning needs r_c >= {:.2f} nm; cathode conductor front [nm] for r_c [nm] = {}".format(
        positions.shape[0] - int(snap["n_initial"]), hi / NM, ", ".join(fronts)))
print("\n".join(lines))
with open(os.path.join(HERE, "summary.txt"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
(ax_j, ax_gap), (ax_g, ax_jn) = axes
for seed, color in zip(SEEDS, COLORS):
    d = data.get(seed)
    if d is None:
        continue
    t, J = d["t"], d["J"]
    end = d["end"] if d["end"] is not None else t[-1]
    ax_j.step(np.append(t[1:], end), np.append(J[1:], J[-1]), where="post", color=color, lw=1.6, label="seed {}".format(seed))
    if d["stop"] == "breakdown":
        ax_j.plot([end], [J[-1]], ls="none", marker="X", ms=9, color=color)
    ax_gap.step(t[1:], d["gap"][1:], where="post", color=color, lw=1.6)
    ax_g.step(t[1:], d["G"][1:], where="post", color=color, lw=1.6)
    ax_jn.plot(np.arange(t.size), J, color=color, lw=1.6)
for ax, title, xlabel, ylabel, ylog in (
        (ax_j, "J(t), filament current added at breakdown", "t (s)", "J (A/cm²)", True),
        (ax_gap, "gap between cathode-side and anode-side conductors", "t (s)", "gap (nm)", False),
        (ax_g, "total generation rate ΣG", "t (s)", "ΣG (1/s)", True),
        (ax_jn, "J vs generated vacancies", "N", "J (A/cm²)", True)):
    ax.set_xscale("log" if xlabel == "t (s)" else "linear")
    ax.set_yscale("log" if ylog else "linear")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", color="#e4e4e2", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
ax_gap.axhline(R_C, color="#8a8a8a", lw=1, ls=":")
ax_gap.text(0.02, 0.04, "r_c = 0.3 nm", transform=ax_gap.transAxes, color="#5a5a5a", fontsize=9)
ax_j.legend(frameon=False, loc="upper left", fontsize=9)
fig.suptitle("ADR-055: 3.6 V / 25 °C, 50×50 nm, precursor sites packed at d = 0.26 nm, r_c = 0.3 nm, E_A = 1.30 eV")
fig.savefig(os.path.join(HERE, "it_dense.png"), dpi=150)

# 击穿状态（或最后一个快照）的缺陷投影
fig, axes = plt.subplots(len(data), 2, figsize=(12, 2.9 * len(data)), constrained_layout=True, squeeze=False)
for row, (seed, d) in enumerate(sorted(data.items())):
    snaps = sorted(f for f in os.listdir(HERE) if f.startswith("snap_s{}_n".format(seed)))
    snap = np.load(os.path.join(HERE, snaps[-1]))
    pos, sides, n_initial = snap["positions"] / 1e-9, snap["sides"], int(snap["n_initial"])
    for col, (axis, name) in enumerate(((0, "x"), (1, "y"))):
        ax = axes[row, col]
        for value, color, label in ((-1, "#b9b9b6", "not connected"), (0, "#2a78d6", "cathode conductor"),
                                    (1, "#eb6834", "anode conductor"), (2, "#d01c1c", "spanning path")):
            m = sides == value
            ax.scatter(pos[m, axis], pos[m, 2], s=7 if value < 0 else 12, color=color, label=label, lw=0)
        ax.scatter(pos[:n_initial, axis], pos[:n_initial, 2], s=22, facecolors="none", edgecolors="k", lw=0.6,
                   label="initial defect")
        ax.set_xlim(0, 50)
        ax.set_ylim(0, LZ)
        ax.set_xlabel("{} (nm)".format(name))
        ax.set_ylabel("z (nm)")
        ax.set_title("seed {}, {} vacancies generated, t = {:.3g} s".format(seed, int(pos.shape[0] - n_initial), float(snap["time"])),
                     fontsize=10)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9, loc="outside lower center", ncol=5)
fig.savefig(os.path.join(HERE, "path_dense.png"), dpi=150)
print("wrote summary.txt, it_dense.png, path_dense.png")
