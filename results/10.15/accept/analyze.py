"""ADR-060/061 之后的 3.6 V 验收汇总（examples/zro2_tddb.yaml 原样，seed 1–4，`drive.py`）。正在跑的轨迹按已写出的行统计。

每条的逐步数据 = `steps_s{SEED}.csv` 里续跑起点之前的行 + `steps_s{SEED}_from{N}.csv`。统计：空位数与应力时间、停跑原因、J 与 ΣG 的首末值、
击穿（第一条有贯穿通路的记录：t_BD、前后 J 的跳升倍数、击穿前最后一段的用时）、接负极导电簇的成员数与最高点、导体间隙降到 3 / 2 / 1 nm 的时刻、
间隙隧穿电流的最大值、缺陷处最高温度（击穿前 / 后）与热迭代收敛、新空位深度、击穿前 I-t 双对数拟合（内层输运电流，第一次生成到 t_BD 或终点，
按 log t 等间距取样；前后半段斜率看弯曲方向）、平均场收敛、每步耗时与峰值工作集。
出图 it_accept.png：J(t)、J、ΣG、导体间隙、缺陷处最高温度对已生成空位数。
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
from kmc_leakage.tddb import fit_power_law  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False
SEEDS = (1, 2, 3, 4)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2")  # 按种子固定
AMBIENT = 298.15  # [K]
NAMES = ("n_generated", "t_s", "J_A_cm2", "J_transport_A_cm2", "J_gap_A_cm2", "J_filament_A_cm2", "G_total_s", "G_trap_s",
         "gap_nm", "negative_front_nm", "n_negative_members", "n_positive_members", "max_floating", "birth_z_nm",
         "mf_converged", "mf_newton_steps", "mf_residual", "thermal_iterations", "thermal_converged", "T_defect_max_K",
         "T_peak_K", "wall_s", "ws_peak_MB")


def load(seed):
    """拼接续跑前后的逐步数据；还没开跑时返回 None。"""
    path = os.path.join(HERE, "steps_s{}.csv".format(seed))
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    starts = []
    for resumed in sorted(glob.glob(os.path.join(HERE, "steps_s{}_from*.csv".format(seed)))):
        start = int(resumed.rsplit("from", 1)[1][:4])
        starts.append(start)
        rows = [r for r in rows if int(r["n_generated"]) < start] + list(csv.DictReader(open(resumed, encoding="utf-8")))
    if not rows:
        return None
    d = {name: np.array([float(r[name]) for r in rows]) for name in NAMES}
    assert np.array_equal(d["n_generated"], np.arange(len(rows))), seed
    d["starts"] = starts
    log = os.path.join(HERE, "logs", "run_s{}.log".format(seed))
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
print("\n".join(lines))
with open(os.path.join(HERE, "summary.txt"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")

fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
(ax_jt, ax_jn, ax_g), (ax_gap, ax_T, ax_off) = axes
ax_off.axis("off")
for seed, color in zip(SEEDS, COLORS):
    if seed not in data:
        continue
    d = data[seed]
    t, N = d["t_s"], d["n_generated"]
    label = "seed {}".format(seed)
    ax_jt.step(t[1:], d["J_A_cm2"][1:], where="post", color=color, lw=2, label=label)
    ax_jn.plot(N, d["J_A_cm2"], color=color, lw=2, label=label)
    ax_g.plot(N, d["G_total_s"], color=color, lw=2, label=label)
    ax_gap.plot(N, d["gap_nm"], color=color, lw=2, label=label)
    ax_T.plot(N, d["T_defect_max_K"] - AMBIENT, color=color, lw=2, label=label)
for ax, title, xlabel, ylabel, xlog, ylog in (
        (ax_jt, "J(t)", "t (s)", "J (A/cm²)", True, True),
        (ax_jn, "J vs generated vacancies", "N", "J (A/cm²)", False, True),
        (ax_g, "total generation rate ΣG", "N", "ΣG (1/s)", False, True),
        (ax_gap, "gap between negative-side and positive-side conductors", "N", "gap (nm)", False, False),
        (ax_T, "max temperature rise at defects", "N", "ΔT (K)", False, False)):
    ax.set_xscale("log" if xlog else "linear")
    ax.set_yscale("log" if ylog else "linear")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", color="#e4e4e2", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
ax_T.set_yscale("symlog", linthresh=1e-3)
ax_jt.legend(frameon=False, loc="upper left", fontsize=9)
fig.suptitle("ADR-060/061 acceptance: examples/zro2_tddb.yaml, 3.6 V / 25 °C, 50×50 nm, r_c = 0.3 nm, seed 1–2 (10 h wall each)")
fig.savefig(os.path.join(HERE, "it_accept.png"), dpi=150)
print("wrote summary.txt, it_accept.png")
