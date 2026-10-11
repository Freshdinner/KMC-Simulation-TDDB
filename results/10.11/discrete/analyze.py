"""ADR-052 验证汇总：离散前驱位点下 seed 1–4 的 I-t、生成空位数 N(t) 与 ΣG(t)，与 ADR-051（模型 C）并排给出。

统计：终止原因与 t_BD、击穿前生成次数、J₀、I-t 双对数拟合（从第一次生成起；[t_end/1000, t_end] 三个数量级窗口；[1 s, t_end]），
窗口前半段 / 后半段各自的斜率（判断弯曲方向），N(t) 的幂律指数 α，用掉的前驱的 E_A 与深度，平均场收敛，每步耗时。
出图 it_discrete.png：J(t)、N(t)、ΣG(t) 三个双对数面板，每条曲线一个种子。
给出子目录（run_long.sh 的 long/）时读那里的轨迹，出图 it_discrete_<子目录>.png，并核对它与本目录 10⁴ s 的轨迹在
10⁴ s 之前逐条相同；击穿的轨迹另给 t_BD = 1000 s 需要的 E_A 均值（P_bb 未饱和时 E_A 平移 ΔE 让时刻乘 exp(ΔE/k_BT)）。
usage: analyze.py [SUBDIR]
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
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # dataviz 参考调色板分类色 1–4，按种子固定（已过校验脚本）
SUBDIR = sys.argv[1] if len(sys.argv) > 1 else None
RUN_DIR = os.path.join(HERE, SUBDIR) if SUBDIR else HERE
RUNS = (("ADR-051 (model C)", os.path.join(ROOT, "results", "10.09", "accept50")),
        ("ADR-052 (discrete precursors)", RUN_DIR))
KT_EV = 298.15 * 8.617333262e-5  # 25 °C 下的 k_BT [eV]
E_A_MEAN = 1.30  # 这批轨迹的 E_A 均值 [eV]


def load(directory, seed):
    path = os.path.join(directory, "steps_s{}.csv".format(seed))
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    column = lambda name: np.array([float(r[name]) for r in rows])  # noqa: E731
    log = os.path.join(directory, "logs", "run_s{}.log".format(seed))
    done = [line for line in open(log, encoding="utf-8") if line.startswith("done")] if os.path.exists(log) else []
    data = dict(t=column("t_s"), J=column("J_A_cm2"), G=column("G_total_s"), conv=column("mf_converged"),
                newton=column("mf_newton_steps"), wall=column("wall_s"), med=column("generated_median_z_nm"),
                top=column("generated_top_nm"), stop=done[0].split()[1] if done else "running",
                end=float(done[0].split("end_time=")[1].split()[0]) if done else None)
    if "birth_E_A_eV" in rows[0]:
        data["E_A"] = column("birth_E_A_eV")[1:]
        data["z"] = column("birth_z_nm")[1:]
        data["left"] = column("n_precursors")
    return data


def log_t_fit(t, J, start, end):
    """分段常数 J(t) 按 log t 等间距取样（与 fit_transient 同口径），窗口 [start, end]；返回 (整段, 前半, 后半) 拟合。"""
    times = np.logspace(np.log10(start), np.log10(end), max(4, int(np.ceil(np.log10(end / start) * 20)) + 1))
    sample = J[np.clip(np.searchsorted(t, times, side="right") - 1, 0, None)]
    half = times.size // 2
    return fit_power_law(times, sample), fit_power_law(times[:half], sample[:half]), fit_power_law(times[half:], sample[half:])


data = {}
for label, directory in RUNS:
    print("######## {}".format(label))
    data[label] = {}
    for seed in SEEDS:
        d = load(directory, seed)
        if d is None:
            continue
        data[label][seed] = d
        t, J, G = d["t"], d["J"], d["G"]
        end = d["end"] if d["end"] is not None else t[-1]
        n = t.size - 1
        print("== seed {}: {} at t_end = {:.4g} s, {} vacancies; J0 {:.3f}, J_end {:.3f} A/cm2; G first {:.3g}, last {:.3g} 1/s".format(
            seed, d["stop"], end, n, J[0], J[-1], G[0], G[-1]))
        print("  mean field converged {}/{}, newton records {}; median step wall {:.1f} s; generated z median {:.2f} nm, top {:.2f} nm".format(
            int(d["conv"].sum()), d["conv"].size, int(np.count_nonzero(d["newton"])), float(np.median(np.diff(d["wall"]))),
            d["med"][-1], d["top"][-1]))
        if "E_A" in d:
            print("  consumed precursors: E_A mean {:.3f} eV (min {:.3f}, max {:.3f}); birth z median {:.2f} nm; precursors left {:.0f} of {:.0f}".format(
                d["E_A"].mean(), d["E_A"].min(), d["E_A"].max(), float(np.median(d["z"])), d["left"][-1], d["left"][0]))
        if n >= 3:
            for name, start in (("first birth", t[1]), ("[t_end/1e3, t_end]", end / 1e3), ("[1 s, t_end]", 1.0)):
                if start >= end:
                    continue
                whole, first, second = log_t_fit(t, J, start, end)
                print("  I-t fit {:20s}: n = {:.2f}, R² = {:.3f}; first half n = {:.2f}, second half n = {:.2f}".format(
                    name, whole.exponent, whole.r_squared, first.exponent, second.exponent))
            N = np.arange(t.size)
            fit = fit_power_law(t[1:], N[1:])
            print("  N(t) ∝ t^{:.2f} (R² = {:.3f}) from first birth".format(fit.exponent, fit.r_squared))
        if d["stop"] == "breakdown" and "E_A" in d:
            print("  t_BD = {:.4g} s -> E_A mean for t_BD = 1000 s: {:.3f} eV".format(
                end, E_A_MEAN - KT_EV * np.log(end / 1000.0)))
        if SUBDIR and "E_A" in d:
            short = load(HERE, seed)
            k = short["t"].size
            same = np.array_equal(short["t"], d["t"][:k]) and np.array_equal(short["J"], d["J"][:k])
            print("  first {} records identical to the 1e4 s run: {}".format(k, same))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
label = RUNS[1][0]
for seed, color in zip(SEEDS, COLORS):
    d = data[label].get(seed)
    if d is None:
        continue
    t, J, G = d["t"], d["J"], d["G"]
    end = d["end"] if d["end"] is not None else t[-1]
    # 第 k 条记录的 J 保持在 [t_k, t_{k+1})，最后一条保持到 t_end；t = 0 的 J₀ 画成第一次生成时刻上的点
    axes[0].step(np.append(t[1:], end), np.append(J[1:], J[-1]), where="post", color=color, lw=2,
                 label="seed {}".format(seed))
    axes[0].plot([t[1]], [J[0]], "o", ms=8, color=color, mec="white", mew=2)
    axes[1].step(t[1:], np.arange(1, t.size), where="post", color=color, lw=2, label="seed {}".format(seed))
    axes[2].step(t[1:], G[1:], where="post", color=color, lw=2, label="seed {}".format(seed))
axes[0].axhline(10.0, color="#8a8a8a", lw=1, ls="--")
axes[0].text(0.02, 0.95, "breakdown 10 A/cm²", transform=axes[0].transAxes, color="#5a5a5a", fontsize=9, va="top")
for ax, title, ylabel in zip(axes, ("J(t) (dot: J at t = 0)", "generated vacancies N(t)", "total generation rate ΣG"),
                             ("J (A/cm²)", "N", "ΣG (1/s)")):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("t (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", color="#e4e4e2", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, loc="lower right")
fig.suptitle("3.6 V / 25 °C, 50×50 nm, discrete precursors (N_T = 1e20 cm⁻³, E_A = 1.30 ± 0.10 eV, ADR-052)")
name = "it_discrete_{}.png".format(SUBDIR) if SUBDIR else "it_discrete.png"
fig.savefig(os.path.join(HERE, name), dpi=150)
print("wrote " + name)
