"""ADR-056 验收汇总：悬空导电簇也等电位，seed 1 下 r_c = 0.3 / 0.35 / 0.4 nm 各一条（3.6 V / 25 °C，E_A = 1.30 eV，前驱位点按 0.26 nm
铺满）。三条都在停跑前的快照处续跑过一次（ADR-057，`make_state.py`，换了新的随机数流），之后手动停。

每条的逐步数据 = `steps_{TAG}.csv` 里续跑起点之前的行 + `steps_{TAG}_from{N}.csv`。统计：空位数与应力时间、J 与 ΣG 的首末值、接负极导电簇的
成员数与最高点、导体间隙降到 3 / 2 / 1 nm 的时刻、最大悬空导电簇、新空位深度与落在负极 0.6 nm 以内的比例、I-t 双对数拟合（内层输运电流，
第一次生成到终点，按 log t 等间距取样；前后半段斜率看弯曲方向）、平均场收敛与每步耗时；终态（最后一个快照加上之后各步的出生位置）上
现有空位形成贯穿通路所需的最小 r_c（二分）。
出图 it_floating.png：J(t)、J、ΣG、导体间隙对已生成空位数；path_floating.png：终态缺陷的 x–z、y–z 投影，接负极 / 正极的导电簇分色。
usage: analyze.py
"""
import csv
import glob
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
from kmc_leakage.tddb import conducting_clusters, distance_clusters, fit_power_law  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False
RADII = (0.3, 0.35, 0.4)  # [nm]
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # 按 r_c 固定
LZ = 8.0  # [nm]
NM = 1e-9
DEVICE = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml")).device


def tag_of(r_c):
    return "s1_rc{:03d}".format(int(round(100 * r_c)))


def load(r_c):
    """拼接停跑前与续跑后的逐步数据；返回各列与续跑起点。"""
    tag = tag_of(r_c)
    resumed = sorted(glob.glob(os.path.join(HERE, "steps_{}_from*.csv".format(tag))))
    rows = list(csv.DictReader(open(os.path.join(HERE, "steps_{}.csv".format(tag)), encoding="utf-8")))
    starts = []
    for path in resumed:
        start = int(path.rsplit("from", 1)[1][:4])
        starts.append(start)
        rows = [r for r in rows if int(r["n_generated"]) < start] + list(csv.DictReader(open(path, encoding="utf-8")))
    column = lambda name: np.array([float(r[name]) for r in rows])  # noqa: E731
    d = {name: column(name) for name in (
        "n_generated", "t_s", "J_A_cm2", "J_transport_A_cm2", "G_total_s", "G_trap_s", "gap_nm", "negative_front_nm",
        "n_negative_members", "max_floating", "birth_x_nm", "birth_y_nm", "birth_z_nm", "birth_side", "mf_converged",
        "mf_newton_steps", "mf_residual", "wall_s", "rates_wall_s", "T_peak_K", "ws_peak_MB")}
    assert np.array_equal(d["n_generated"], np.arange(len(rows))), tag
    d["starts"] = starts
    return d


def final_positions(r_c, d):
    """终态缺陷位置 [nm]：最后一个快照，加上它之后各步的出生位置。"""
    snaps = sorted(glob.glob(os.path.join(HERE, "snap_{}_n*.npz".format(tag_of(r_c)))))
    snap = np.load(snaps[-1])
    n_snap = snap["positions"].shape[0] - int(snap["n_initial"])
    later = np.column_stack([d["birth_x_nm"], d["birth_y_nm"], d["birth_z_nm"]])[n_snap + 1:]
    return np.vstack([snap["positions"] / NM, later]), int(snap["n_initial"])


def log_t_fit(t, J, start, end):
    """分段常数 J(t) 按 log t 等间距取样（与 fit_transient 同口径），窗口 [start, end]；返回 (整段, 前半, 后半) 拟合。"""
    times = np.logspace(np.log10(start), np.log10(end), max(4, int(np.ceil(np.log10(end / start) * 20)) + 1))
    sample = J[np.clip(np.searchsorted(t, times, side="right") - 1, 0, None)]
    half = times.size // 2
    return fit_power_law(times, sample), fit_power_law(times[:half], sample[:half]), fit_power_law(times[half:], sample[half:])


lines, data, finals = [], {}, {}
for r_c in RADII:
    d = load(r_c)
    data[r_c] = d
    t, J, G, gap = d["t_s"], d["J_A_cm2"], d["G_total_s"], d["gap_nm"]
    n = t.size - 1
    positions, n_initial = final_positions(r_c, d)
    finals[r_c] = positions, n_initial
    assert positions.shape[0] - n_initial == n
    z = d["birth_z_nm"][1:]
    z = z[np.isfinite(z)]  # 续跑起点的记录没有出生
    lines.append("== r_c {:.2f} nm: stopped by hand at {} vacancies, t = {:.6g} s (resumed from n = {}); J {:.3f} -> {:.1f} A/cm2".format(
        r_c, n, t[-1], ", ".join(str(s) for s in d["starts"]) or "-", J[0], J[-1]))
    lines.append("  sum G first {:.3g}, last {:.3g}, max {:.3g} 1/s (trap share last {:.2f})".format(
        G[0], G[-1], G.max(), d["G_trap_s"][-1] / G[-1]))
    reach = []
    for level in (3.0, 2.0, 1.0):
        k = np.flatnonzero(gap <= level)
        reach.append("t = {:.6g} s (n = {})".format(t[k[0]], k[0]) if k.size else "-")
    lines.append("  negative conductor: members {} -> {}, front {:.2f} -> {:.2f} nm; conductor gap {:.2f} -> {:.2f} nm; "
                 "gap <= 3 / 2 / 1 nm first at: {}".format(
                     int(d["n_negative_members"][0]), int(d["n_negative_members"][-1]), d["negative_front_nm"][0],
                     d["negative_front_nm"][-1], gap[0], gap[-1], " / ".join(reach)))
    joined = np.flatnonzero(d["n_negative_members"] >= 10)
    if joined.size:
        k = joined[0]
        lines.append("  negative conductor reached >= 10 members at n = {} (t = {:.6g} s): {} members, front {:.2f} nm; "
                     "births n = {}-{} at z [nm] {}".format(
                         k, t[k], int(d["n_negative_members"][k]), d["negative_front_nm"][k], k - 8, k,
                         " ".join("{:.2f}".format(v) for v in d["birth_z_nm"][k - 8:k + 1])))
        lines.append("  after that: t advanced {:.3g} s over {} vacancies, sum G {:.3g}-{:.3g} 1/s".format(
            t[-1] - t[k], n - k, G[k:].min(), G[k:].max()))
    lines.append("  largest floating cluster: max over run {} vacancies, at end {}".format(
        int(d["max_floating"].max()), int(d["max_floating"][-1])))
    lines.append("  births: z median {:.2f} nm, 90% {:.2f} nm, max {:.2f} nm; within 0.6 nm of negative electrode {:.1f}%; "
                 "z > 2 nm {:.0f}%".format(*np.percentile(z, [50, 90, 100]), 100 * np.mean(z < 0.6), 100 * np.mean(z > 2.0)))
    whole, first, second = log_t_fit(t, d["J_transport_A_cm2"], t[1], t[-1])
    lines.append("  I-t fit (transport current, first birth to end): n = {:.2f}, R² = {:.3f}; first half n = {:.2f}, "
                 "second half n = {:.2f}".format(whole.exponent, whole.r_squared, first.exponent, second.exponent))
    step = np.diff(d["wall_s"])
    step = step[step > 0]  # 续跑起点处 wall 从头计
    bad = np.flatnonzero(d["mf_converged"] == 0)
    lines.append("  mean field converged {}/{} (Newton used in {} records){}; step wall median {:.0f} s, last 10 mean {:.0f} s "
                 "(rates {:.0f} s); max dT {:.2g} K; peak working set {:.0f} MB".format(
                     int(d["mf_converged"].sum()), t.size, int(np.sum(d["mf_newton_steps"] > 0)),
                     "; not converged at n = {} (residual up to {:.1e})".format(bad.tolist(), d["mf_residual"][bad].max())
                     if bad.size else "", float(np.median(step)), float(step[-10:].mean()),
                     float(np.nanmedian(d["rates_wall_s"])), d["T_peak_K"].max() - 298.15, d["ws_peak_MB"].max()))
    lo, hi = 0.25 * NM, 4.0 * NM
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if np.any(conducting_clusters(positions * NM, DEVICE, mid) == 2) else (mid, hi)
    labels = distance_clusters(positions * NM, DEVICE, r_c * NM)
    sides = conducting_clusters(positions * NM, DEVICE, r_c * NM)
    lines.append("  final state: spanning needs r_c >= {:.2f} nm; clusters with >= 2 vacancies: {} (floating {})".format(
        hi / NM, int(np.sum(np.bincount(labels) >= 2)),
        len(set(labels[(np.bincount(labels)[labels] >= 2) & (sides < 0)].tolist()))))
print("\n".join(lines))
with open(os.path.join(HERE, "summary.txt"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
(ax_jt, ax_jn), (ax_g, ax_gap) = axes
for r_c, color in zip(RADII, COLORS):
    d = data[r_c]
    t, N = d["t_s"], d["n_generated"]
    label = "r_c = {:.2f} nm".format(r_c)
    ax_jt.step(t[1:], d["J_A_cm2"][1:], where="post", color=color, lw=2, label=label)
    ax_jn.plot(N, d["J_A_cm2"], color=color, lw=2, label=label)
    ax_g.plot(N, d["G_total_s"], color=color, lw=2, label=label)
    ax_gap.plot(N, d["gap_nm"], color=color, lw=2, label=label)
    for ax, y in ((ax_jn, d["J_A_cm2"]), (ax_g, d["G_total_s"]), (ax_gap, d["gap_nm"])):
        ax.annotate("{:.2f} nm".format(r_c), (N[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=9,
                    color="#3a3a3a", va="center")
for ax, title, xlabel, ylabel, xlog, ylog in (
        (ax_jt, "J(t)", "t (s)", "J (A/cm²)", True, True),
        (ax_jn, "J vs generated vacancies", "N", "J (A/cm²)", False, True),
        (ax_g, "total generation rate ΣG", "N", "ΣG (1/s)", False, True),
        (ax_gap, "gap between negative-side and positive-side conductors", "N", "gap (nm)", False, False)):
    ax.set_xscale("log" if xlog else "linear")
    ax.set_yscale("log" if ylog else "linear")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", color="#e4e4e2", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
ax_jt.legend(frameon=False, loc="upper left", fontsize=9)
fig.suptitle("ADR-056: floating conductors equipotential; seed 1, 3.6 V / 25 °C, 50×50 nm, sites packed at 0.26 nm, E_A = 1.30 eV")
fig.savefig(os.path.join(HERE, "it_floating.png"), dpi=150)

fig, axes = plt.subplots(len(RADII), 2, figsize=(12, 2.9 * len(RADII)), constrained_layout=True, squeeze=False)
for row, r_c in enumerate(RADII):
    pos, n_initial = finals[r_c]
    sides = conducting_clusters(pos * NM, DEVICE, r_c * NM)
    labels = distance_clusters(pos * NM, DEVICE, r_c * NM)
    floating = (sides < 0) & (np.bincount(labels)[labels] >= 2)
    for col, (axis, name) in enumerate(((0, "x"), (1, "y"))):
        ax = axes[row, col]
        for mask, color, size, label in (
                ((sides < 0) & ~floating, "#b9b9b6", 7, "isolated vacancy"),
                (floating, "#8a5cc2", 10, "floating conductor (≥ 2)"),
                ((sides == 0) | (sides == 2), "#2a78d6", 12, "negative-side conductor"),
                ((sides == 1) | (sides == 2), "#eb6834", 12, "positive-side conductor")):
            ax.scatter(pos[mask, axis], pos[mask, 2], s=size, color=color, label=label, lw=0)
        ax.scatter(pos[:n_initial, axis], pos[:n_initial, 2], s=22, facecolors="none", edgecolors="k", lw=0.6,
                   label="initial defect")
        ax.set_xlim(0, 50)
        ax.set_ylim(0, LZ)
        ax.set_xlabel("{} (nm)".format(name))
        ax.set_ylabel("z (nm)")
        ax.set_title("r_c = {:.2f} nm, {} vacancies generated, t = {:.4g} s".format(
            r_c, pos.shape[0] - n_initial, data[r_c]["t_s"][-1]), fontsize=10)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9, loc="outside lower center", ncol=5)
fig.savefig(os.path.join(HERE, "path_floating.png"), dpi=150)
print("wrote summary.txt, it_floating.png, path_floating.png")
