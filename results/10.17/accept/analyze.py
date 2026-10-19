"""ADR-062（E_A = 1.6 eV）之后的 3.6 V 验收汇总（examples/zro2_tddb.yaml 原样，seed 1、2，`drive.py`）。正在跑的轨迹按已写出的行统计。

每条的逐步数据 = `steps_s{SEED}.csv` 里续跑起点之前的行 + `steps_s{SEED}_from{N}.csv`。统计：空位数与应力时间、停跑原因、J 与 ΣG 的首末值、
击穿（第一条有贯穿通路的记录：t_BD、前后 J 的跳升倍数、击穿前最后一段的用时）、接负极导电簇的成员数与最高点、导体间隙降到 3 / 2 / 1 nm 的时刻、
间隙隧穿电流的最大值、缺陷处最高温度（击穿前 / 后）与热迭代收敛、新空位深度、击穿前 I-t 双对数拟合（内层输运电流，第一次生成到 t_BD 或终点，
按 log t 等间距取样；前后半段斜率看弯曲方向）、平均场收敛、每步耗时与峰值工作集。
出图 it_accept.pdf / .png（按 Nature 版式：双栏 183 mm、Arial 7 pt）与 E_A = 1.30 eV 的同种子轨迹（`../../10.15/accept/`）对比：
(a) J(t)；(b) J 对已生成空位数；(c) ΣG 对已生成空位数，E_A = 1.30 eV 的乘以 exp(−0.3 eV/k_BT)；(d) 导体间隙；(e) 新空位出生高度；
(f) 达到同样空位数所用应力时间之比。
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
from scipy import constants  # noqa: E402

from kmc_leakage.tddb import fit_power_law  # noqa: E402

# Nature 版式：Arial 5–7 pt，线宽 0.5–1 pt，刻度朝外，去掉上、右边框
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False, "pdf.fonttype": 42,
    "savefig.dpi": 600})
MM = 1.0 / 25.4  # [inch]
SEEDS = (1, 2)
COLORS = ("#0072B2", "#D55E00")  # Okabe–Ito，按种子固定
AMBIENT = 298.15  # [K]
REFERENCE = os.path.join(HERE, "..", "..", "10.15", "accept")  # E_A = 1.30 eV 的同种子轨迹（ADR-060/061 验收）
DELTA_EA = 0.3  # [eV]，1.6 − 1.30
NAMES = ("n_generated", "t_s", "J_A_cm2", "J_transport_A_cm2", "J_gap_A_cm2", "J_filament_A_cm2", "G_total_s", "G_trap_s",
         "gap_nm", "negative_front_nm", "n_negative_members", "n_positive_members", "max_floating", "birth_z_nm",
         "mf_converged", "mf_newton_steps", "mf_residual", "thermal_iterations", "thermal_converged", "T_defect_max_K",
         "T_peak_K", "wall_s", "ws_peak_MB")


def load(seed, here=HERE):
    """拼接续跑前后的逐步数据；还没开跑时返回 None。"""
    path = os.path.join(here, "steps_s{}.csv".format(seed))
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    starts = []
    for resumed in sorted(glob.glob(os.path.join(here, "steps_s{}_from*.csv".format(seed)))):
        start = int(resumed.rsplit("from", 1)[1][:4])
        starts.append(start)
        rows = [r for r in rows if int(r["n_generated"]) < start] + list(csv.DictReader(open(resumed, encoding="utf-8")))
    if not rows:
        return None
    d = {name: np.array([float(r[name]) for r in rows]) for name in NAMES}
    assert np.array_equal(d["n_generated"], np.arange(len(rows))), seed
    d["starts"] = starts
    log = os.path.join(here, "logs", "run_s{}{}.log".format(seed, "_from{:04d}".format(starts[-1]) if starts else ""))
    done = [line for line in open(log, encoding="utf-8") if line.startswith("done")] if os.path.exists(log) else []
    d["status"] = done[-1].split()[1] if done else "running"
    return d


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
        lines.append("== seed {}: not started".format(seed))
        continue
    data[seed] = d
    t, J, G, gap, N = d["t_s"], d["J_A_cm2"], d["G_total_s"], d["gap_nm"], d["n_generated"]
    n = t.size - 1
    spanning = np.flatnonzero(d["J_filament_A_cm2"] > 0)
    k_bd = spanning[0] if spanning.size else None
    lines.append("== seed {}: {} ({}), {} vacancies, t = {:.6g} s; J {:.3f} -> {:.4g} A/cm2".format(
        seed, d["status"], "resumed from n = " + ", ".join(str(s) for s in d["starts"]) if d["starts"] else "not resumed",
        n, t[-1], J[0], J[-1]))
    finite = np.isfinite(G)
    lines.append("  sum G first {:.3g}, last {:.3g}, max {:.3g} 1/s (trap share last {:.2f})".format(
        G[0], G[finite][-1], G[finite].max(), d["G_trap_s"][finite][-1] / G[finite][-1]))
    if k_bd is not None:
        slow = np.flatnonzero(G[:k_bd] < 10.0)
        lines.append("  BREAKDOWN at n = {}, t_BD = {:.6g} s: J {:.4g} -> {:.4g} A/cm2 (x {:.3g}); last sum G < 10 1/s at "
                     "n = {} ({:.3g} s before t_BD); {} records after".format(
                         k_bd, t[k_bd], J[k_bd - 1], J[k_bd], J[k_bd] / J[k_bd - 1],
                         slow[-1] if slow.size else -1, t[k_bd] - t[slow[-1]] if slow.size else float("nan"), n - k_bd))
    else:
        lines.append("  no breakdown")
    reach = []
    for level in (3.0, 2.0, 1.0):
        k = np.flatnonzero(gap <= level)
        reach.append("t = {:.6g} s (n = {})".format(t[k[0]], k[0]) if k.size else "-")
    lines.append("  negative conductor: members {} -> {} (max {}), front max {:.2f} nm; positive members max {}; conductor gap "
                 "{:.2f} -> {:.2f} nm (min {:.2f}); gap <= 3 / 2 / 1 nm first at: {}".format(
                     int(d["n_negative_members"][0]), int(d["n_negative_members"][-1]), int(d["n_negative_members"].max()),
                     d["negative_front_nm"].max(), int(d["n_positive_members"].max()), gap[0], gap[-1], gap.min(),
                     " / ".join(reach)))
    pre = slice(0, k_bd if k_bd is not None else t.size)
    jgap = d["J_gap_A_cm2"]
    kg = int(np.argmax(jgap[pre]))
    lines.append("  gap tunneling before breakdown: max {:.3g} A/cm2 (= {:.3g} A) at n = {} (gap {:.2f} nm), share of J {:.2g}".format(
        jgap[kg], jgap[kg] * 2500e-14, kg, gap[kg], jgap[kg] / J[kg]))
    Td = d["T_defect_max_K"]
    lines.append("  defect temperature rise: before breakdown max {:.3g} K (n = {}){}; thermal not converged in {} records; "
                 "thermal iterations max {}".format(
                     Td[pre].max() - AMBIENT, int(np.argmax(Td[pre])),
                     "; after breakdown max {:.3g} K, field peak {:.3g} K".format(Td[k_bd:].max(), d["T_peak_K"][k_bd:].max())
                     if k_bd is not None else "", int(np.sum(d["thermal_converged"] == 0)), int(d["thermal_iterations"].max())))
    z = d["birth_z_nm"][1:]
    z = z[np.isfinite(z)]
    if z.size:
        lines.append("  births: z median {:.2f} nm, 90% {:.2f} nm, max {:.2f} nm; within 0.6 nm of negative electrode {:.1f}%; "
                     "z > 2 nm {:.0f}%".format(*np.percentile(z, [50, 90, 100]), 100 * np.mean(z < 0.6), 100 * np.mean(z > 2.0)))
    end = t[k_bd] if k_bd is not None else t[-1]
    if n >= 10 and end > t[1]:
        whole, first, second = log_t_fit(t, d["J_transport_A_cm2"], t[1], end)
        lines.append("  I-t fit before breakdown (transport current, first birth to {}): n = {:.2f}, R² = {:.3f}; first half "
                     "n = {:.2f}, second half n = {:.2f}".format(
                         "t_BD" if k_bd is not None else "end", whole.exponent, whole.r_squared, first.exponent,
                         second.exponent))
    step = np.diff(d["wall_s"])
    step = step[step > 0]  # 续跑起点处 wall 从头计
    bad = np.flatnonzero(d["mf_converged"] == 0)
    lines.append("  mean field converged {}/{} (Newton used in {} records){}; step wall median {:.0f} s, last 10 mean {:.0f} s; "
                 "peak working set {:.0f} MB".format(
                     int(d["mf_converged"].sum()), t.size, int(np.sum(d["mf_newton_steps"] > 0)),
                     "; not converged at n = {} (residual up to {:.1e})".format(bad.tolist(), d["mf_residual"][bad].max())
                     if bad.size else "", float(np.median(step)) if step.size else 0.0,
                     float(step[-10:].mean()) if step.size else 0.0, d["ws_peak_MB"].max()))
# ---- 与 E_A = 1.30 eV 的同种子轨迹对比 ----
scale = np.exp(-DELTA_EA * constants.electron_volt / (constants.Boltzmann * AMBIENT))
reference = {seed: load(seed, REFERENCE) for seed in SEEDS}
lines.append("== comparison with E_A = 1.30 eV ({}), same seeds; exp(-{} eV/kT) = {:.3g} (1/{:.3g})".format(
    os.path.relpath(REFERENCE, HERE), DELTA_EA, scale, 1.0 / scale))
for seed in SEEDS:
    d, r = data.get(seed), reference.get(seed)
    if d is None or r is None:
        continue
    m = min(d["t_s"].size, r["t_s"].size)
    k = np.arange(50, m)  # 前 50 个空位的出生次数太少，比值噪声大
    ratio_t = d["t_s"][k] / r["t_s"][k]
    ratio_J = d["J_A_cm2"][k] / r["J_A_cm2"][k]
    ratio_G = d["G_total_s"][k] / r["G_total_s"][k]
    zr = r["birth_z_nm"][1:m]
    zd = d["birth_z_nm"][1:m]
    lines.append("  seed {}: N = 50-{}: t ratio median {:.3g} (IQR {:.3g}-{:.3g}); sum G ratio median {:.3g}; J ratio median {:.3g} "
                 "(range {:.2f}-{:.2f}); births up to N = {}: z median {:.2f} vs {:.2f} nm, z > 2 nm {:.0f}% vs {:.0f}%; "
                 "gap min {:.2f} vs {:.2f} nm".format(
                     seed, m - 1, np.median(ratio_t), *np.percentile(ratio_t, [25, 75]), np.median(ratio_G), np.median(ratio_J),
                     ratio_J.min(), ratio_J.max(), m - 1, np.nanmedian(zd), np.nanmedian(zr), 100 * np.nanmean(zd > 2.0),
                     100 * np.nanmean(zr > 2.0), d["gap_nm"][:m].min(), r["gap_nm"][:m].min()))
print("\n".join(lines))
with open(os.path.join(HERE, "summary.txt"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")

fig, axes = plt.subplots(2, 3, figsize=(183 * MM, 105 * MM), constrained_layout=True)
(ax_jt, ax_jn, ax_g), (ax_gap, ax_z, ax_ratio) = axes
bins = np.arange(0.2, 8.0 + 1e-9, 0.25)
for seed, color in zip(SEEDS, COLORS):
    for d, ls, ea in ((reference.get(seed), (0, (3, 1.5)), 1.30), (data.get(seed), "-", 1.6)):
        if d is None:
            continue
        t, N = d["t_s"], d["n_generated"]
        label = "seed {}, $E_A$ = {:.2f} eV".format(seed, ea)
        lw = 0.6 if ea < 1.5 else 0.9
        ax_jt.step(t[1:], d["J_A_cm2"][1:], where="post", color=color, lw=lw, ls=ls, label=label)
        ax_jn.plot(N, d["J_A_cm2"], color=color, lw=lw, ls=ls, label=label)
        ax_g.plot(N, d["G_total_s"] * (scale if ea < 1.5 else 1.0), color=color, lw=lw, ls=ls, label=label)
        ax_gap.plot(N, d["gap_nm"], color=color, lw=lw, ls=ls, label=label)
        z = d["birth_z_nm"][1:]
        ax_z.hist(z[np.isfinite(z)], bins=bins, histtype="step", density=True, color=color, lw=lw, ls=ls, label=label)
    d, r = data.get(seed), reference.get(seed)
    if d is not None and r is not None:
        m = min(d["t_s"].size, r["t_s"].size)
        ax_ratio.plot(np.arange(1, m), d["t_s"][1:m] / r["t_s"][1:m], color=color, lw=0.9, label="seed {}".format(seed))
ax_ratio.axhline(1.0 / scale, color="0.4", lw=0.5, ls=(0, (1, 1)))
ax_ratio.annotate("exp(0.3 eV/$k_BT$) = {:.2g}".format(1.0 / scale), (1.0, 1.0 / scale), xycoords=("axes fraction", "data"),
                  xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom", fontsize=6, color="0.3")
for ax, xlabel, ylabel, xlog, ylog in (
        (ax_jt, "Stress time $t$ (s)", "$J$ (A cm$^{-2}$)", True, True),
        (ax_jn, "Generated vacancies $N$", "$J$ (A cm$^{-2}$)", False, True),
        (ax_g, "Generated vacancies $N$", "$\\Sigma G$ (s$^{-1}$)", False, True),
        (ax_gap, "Generated vacancies $N$", "Conductor gap (nm)", False, False),
        (ax_z, "Birth height $z$ (nm)", "Probability density (nm$^{-1}$)", False, False),
        (ax_ratio, "Generated vacancies $N$", "$t_{1.6\\,\\mathrm{eV}}/t_{1.30\\,\\mathrm{eV}}$", False, True)):
    ax.set_xscale("log" if xlog else "linear")
    ax.set_yscale("log" if ylog else "linear")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
ax_gap.set_ylim(0, 8.3)
ax_z.set_xlim(0, 8.0)
ax_g.annotate("$E_A$ = 1.30 eV scaled by exp($-$0.3 eV/$k_BT$)", (0.98, 0.03), xycoords="axes fraction", ha="right", va="bottom", fontsize=6, color="0.3")
ax_ratio.set_ylim(1e4, 1e6)
for ax, letter in zip(axes.flat, "abcdef"):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")
ax_jn.legend(frameon=False, loc="lower right", handlelength=2.2)
ax_ratio.legend(frameon=False, loc="lower left")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "it_accept.{}".format(ext)))
print("wrote summary.txt, it_accept.pdf, it_accept.png")
