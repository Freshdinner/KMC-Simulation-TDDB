"""ADR-053 验收汇总：E_A 全部前驱相同（本目录）与 ADR-052 的 E_A 高斯分布（../../10.11/discrete/long）并排，seed 1–4。

同一种子两组的初始缺陷与前驱位置相同，只差 E_A 是否有分布。统计：终止原因与 t_BD、生成空位数、J₀、I-t 双对数拟合（从第一次
生成起；[t_end/1000, t_end]；前后半段斜率看弯曲方向）、N(t) 的幂律指数、ΣG 的首末值、新空位深度、按深度分层的前驱用掉比例，
以及 t_BD = 1000 s 需要的 E_A（P_bb 未饱和时 E_A 平移 ΔE 让时刻乘 exp(ΔE/k_BT)）。本目录 seed 3 到 1.1×10⁷ s 仍未击穿，
与 ADR-052 长轨迹的应力时间上限 10⁷ s 对齐后手动停掉，没有 done 行与 final 文件，记为 unfinished。
出图 it_fixed_ea.png：J(t)、N(t)、ΣG(t)、J(N) 四个面板，颜色按种子，实线 ADR-053、虚线 ADR-052。
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
from kmc_leakage.tddb import fit_power_law  # noqa: E402

SEEDS = (1, 2, 3, 4)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # 与 ../../10.11/discrete/analyze.py 相同，按种子固定
RUNS = (("ADR-052 (E_A 1.30 ± 0.10 eV)", os.path.join(ROOT, "results", "10.11", "discrete", "long")),
        ("ADR-053 (E_A 1.30 eV fixed)", HERE))
KT_EV = 298.15 * 8.617333262e-5  # 25 °C 下的 k_BT [eV]
E_A = 1.30  # 两组的 E_A（均值）[eV]
LAYERS = np.arange(0.0, 9.0, 1.0)  # nm


def load(directory, seed):
    path = os.path.join(directory, "steps_s{}.csv".format(seed))
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    column = lambda name: np.array([float(r[name]) for r in rows])  # noqa: E731
    log = os.path.join(directory, "logs", "run_s{}.log".format(seed))
    done = [line for line in open(log, encoding="utf-8") if line.startswith("done")] if os.path.exists(log) else []
    initial = np.load(os.path.join(directory, "snap_s{}_n0000.npz".format(seed)))["precursor_positions"][:, 2] / 1e-9
    last = max(int(r["n_generated"]) for r in rows) // 25 * 25
    left = np.load(os.path.join(directory, "snap_s{}_n{:04d}.npz".format(seed, last)))["precursor_positions"][:, 2] / 1e-9
    return dict(t=column("t_s"), J=column("J_A_cm2"), G=column("G_total_s"), G_trap=column("G_trap_s"),
                z=column("birth_z_nm")[1:], conv=column("mf_converged"), wall=column("wall_s"),
                dT=column("T_peak_K") - 298.15,
                used=1.0 - np.histogram(left, bins=LAYERS)[0] / np.maximum(np.histogram(initial, bins=LAYERS)[0], 1),
                used_at=last, stop=done[0].split()[1] if done else "unfinished",
                end=float(done[0].split("end_time=")[1].split()[0]) if done else None)


def log_t_fit(t, J, start, end):
    """分段常数 J(t) 按 log t 等间距取样（与 fit_transient 同口径），窗口 [start, end]；返回 (整段, 前半, 后半) 拟合。"""
    times = np.logspace(np.log10(start), np.log10(end), max(4, int(np.ceil(np.log10(end / start) * 20)) + 1))
    sample = J[np.clip(np.searchsorted(t, times, side="right") - 1, 0, None)]
    half = times.size // 2
    return fit_power_law(times, sample), fit_power_law(times[:half], sample[:half]), fit_power_law(times[half:], sample[half:])


lines, data = [], {}
for label, directory in RUNS:
    lines.append("######## {}".format(label))
    data[label] = {}
    for seed in SEEDS:
        d = load(directory, seed)
        if d is None:
            continue
        data[label][seed] = d
        t, J, G = d["t"], d["J"], d["G"]
        end = d["end"] if d["end"] is not None else t[-1]
        n = t.size - 1
        lines.append("== seed {}: {} at t_end = {:.4g} s, {} vacancies; J0 {:.3f}, J_end {:.3f} A/cm2; G first {:.3g}, last {:.3g} 1/s "
                     "(trap share last {:.2f})".format(seed, d["stop"], end, n, J[0], J[-1], G[0], G[-1], d["G_trap"][-1] / G[-1]))
        lines.append("  birth z median {:.2f} nm, 90% {:.2f} nm, max {:.2f} nm; precursors used per nm layer at n = {}: {} %".format(
            *np.percentile(d["z"], [50, 90, 100]), d["used_at"], ",".join("{:.0f}".format(100 * u) for u in d["used"])))
        lines.append("  mean field converged {}/{}; median step wall {:.1f} s; max dT {:.1e} K".format(
            int(d["conv"].sum()), d["conv"].size, float(np.median(np.diff(d["wall"]))), d["dT"].max()))
        for name, start in (("first birth", t[1]), ("[t_end/1e3, t_end]", end / 1e3)):
            whole, first, second = log_t_fit(t, J, start, end)
            lines.append("  I-t fit {:20s}: n = {:.2f}, R² = {:.3f}; first half n = {:.2f}, second half n = {:.2f}".format(
                name, whole.exponent, whole.r_squared, first.exponent, second.exponent))
        N = np.arange(t.size)
        fit = fit_power_law(t[1:], N[1:])
        lines.append("  N(t) ∝ t^{:.2f} (R² = {:.3f}); t at N = 25 / 100 / 200: {}".format(
            fit.exponent, fit.r_squared, " / ".join("{:.3g}".format(t[k]) if k < t.size else "-" for k in (25, 100, 200))))
        if d["stop"] == "breakdown":
            lines.append("  t_BD = {:.4g} s -> E_A for t_BD = 1000 s: {:.3f} eV".format(end, E_A - KT_EV * np.log(end / 1000.0)))
print("\n".join(lines))
with open(os.path.join(HERE, "summary.txt"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
(ax_j, ax_n), (ax_g, ax_jn) = axes
for (label, _), style, width in zip(RUNS, ("--", "-"), (1.2, 2.0)):
    for seed, color in zip(SEEDS, COLORS):
        d = data[label].get(seed)
        if d is None:
            continue
        t, J, G = d["t"], d["J"], d["G"]
        end = d["end"] if d["end"] is not None else t[-1]
        name = "seed {}, {}".format(seed, label.split(" ")[0])
        ax_j.step(np.append(t[1:], end), np.append(J[1:], J[-1]), where="post", color=color, ls=style, lw=width, label=name)
        ax_n.step(t[1:], np.arange(1, t.size), where="post", color=color, ls=style, lw=width)
        ax_g.step(t[1:], G[1:], where="post", color=color, ls=style, lw=width)
        ax_jn.plot(np.arange(t.size), J, color=color, ls=style, lw=width)
ax_j.axhline(10.0, color="#8a8a8a", lw=1, ls=":")
ax_j.text(0.02, 0.95, "breakdown 10 A/cm²", transform=ax_j.transAxes, color="#5a5a5a", fontsize=9, va="top")
for ax, title, xlabel, ylabel in (
        (ax_j, "J(t)", "t (s)", "J (A/cm²)"), (ax_n, "generated vacancies N(t)", "t (s)", "N"),
        (ax_g, "total generation rate ΣG", "t (s)", "ΣG (1/s)"), (ax_jn, "J vs generated vacancies", "N", "J (A/cm²)")):
    ax.set_xscale("log" if xlabel == "t (s)" else "linear")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", color="#e4e4e2", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
ax_j.legend(frameon=False, loc="lower right", fontsize=8, ncol=2)
fig.suptitle("3.6 V / 25 °C, 50×50 nm, N_T = 1e20 cm⁻³: solid = E_A fixed 1.30 eV (ADR-053), dashed = 1.30 ± 0.10 eV (ADR-052)")
fig.savefig(os.path.join(HERE, "it_fixed_ea.png"), dpi=150)
print("wrote summary.txt, it_fixed_ea.png")
