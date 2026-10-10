"""ADR-051 验收汇总：50×50 nm（本目录）与 ADR-048 的 20×20 nm（../accept/）用同一套统计并排给出——ΣG、平均场收敛、
I-t 双对数拟合（n、R²）、J 对生成空位数的依赖、击穿时刻与击穿前生成次数、每步耗时与峰值工作集；出图 compare_it.png。

usage: analyze.py            （由 `results/10.09/accept/analyze.py` 改写：两种盒子同一套统计，拟合起点加 n = 10、25，
                               公共 log t 网格按各自的轨迹取，加每步耗时与内存）
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
from kmc_leakage.tddb import fit_power_law  # noqa: E402

BOXES = (("20×20 nm (ADR-048)", os.path.join(HERE, "..", "accept"), 20.0),
         ("50×50 nm (ADR-051)", HERE, 50.0))
SEEDS = (1, 2, 3, 4)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # dataviz 参考调色板分类色 1–4，按种子固定
BREAKDOWN = 10.0  # A/cm²


def load(directory, seed):
    path = os.path.join(directory, "steps_s{}.csv".format(seed))
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    column = lambda name, kind=float: np.array([kind(r[name]) for r in rows])  # noqa: E731
    log = os.path.join(directory, "logs", "run_s{}.log".format(seed))
    done = [line for line in open(log, encoding="utf-8") if line.startswith("done")] if os.path.exists(log) else []
    return dict(t=column("t_s"), J=column("J_A_cm2"), G=column("G_total_s"), conv=column("mf_converged", int),
                newton=column("mf_newton_steps", int), it=column("mf_iterations", int), tat=column("J_tat_A_cm2"),
                pf=column("J_pf_A_cm2"), top=column("generated_top_nm"), med=column("generated_median_z_nm"),
                wall=column("wall_s"), ws_peak=column("ws_peak_MB"), kinds=[r["birth_event"] for r in rows[1:]],
                stop=done[0].split()[1] if done else "running")


def log_t_fit(t, J, start, end):
    """分段常数 J(t) 按 log t 等间距取样（与 fit_transient 同口径），窗口 [start, end]。"""
    times = np.logspace(np.log10(start), np.log10(end), max(2, int(np.ceil(np.log10(end / start) * 20)) + 1))
    idx = np.clip(np.searchsorted(t, times, side="right") - 1, 0, None)
    return fit_power_law(times, J[idx])


data = {}
for label, directory, side in BOXES:
    print("######## {}".format(label))
    data[label] = {}
    for seed in SEEDS:
        d = load(directory, seed)
        if d is None:
            continue
        data[label][seed] = d
        t, J, G = d["t"], d["J"], d["G"]
        n_last = t.size - 1
        step_wall = np.diff(d["wall"])
        print("== seed {}: {}, {} records, t_end = {:.4g} s at n = {}".format(seed, d["stop"], t.size, t[-1], n_last))
        print("  J(t=0) {:.3g} A/cm²; ΣG min {:.0f} max {:.0f} s^-1, first {:.0f} last {:.0f}; ΣG/A first {:.3g} last {:.3g} "
              "s^-1 nm^-2".format(J[0], G.min(), G.max(), G[0], G[-1], G[0] / side ** 2, G[-1] / side ** 2))
        print("  mean field converged {}/{}, newton {} records, max it {}".format(
            d["conv"].sum(), d["conv"].size, np.count_nonzero(d["newton"]), d["it"].max()))
        print("  TAT share at end {:.2f}, PF {:.2f}; generated top z {:.2f} nm, median z {:.2f} nm; trap-term births {}/{}".format(
            d["tat"][-1] / J[-1], d["pf"][-1] / J[-1], d["top"][-1], d["med"][-1], d["kinds"].count("trap"), len(d["kinds"])))
        if step_wall.size:
            print("  step wall: first {:.0f} s, median {:.0f} s, last {:.0f} s, max {:.0f} s; total {:.2f} h; peak ws {:.0f} MB".format(
                step_wall[0], np.median(step_wall), step_wall[-1], step_wall.max(), d["wall"][-1] / 3600, d["ws_peak"].max()))
        for k in (1, 2, 3, 5, 10, 25):
            if k < n_last:
                fit = log_t_fit(t, J, t[k], t[-1])
                print("  fit from n={} (t={:.3g} s): n = {:.2f}, R² = {:.3f}".format(k, t[k], fit.exponent, fit.r_squared))
        if n_last >= 2:
            N = np.arange(J.size)
            fit = fit_power_law(N[1:], J[1:])
            print("  J vs N: J ∝ N^{:.2f}, R² = {:.3f}; J(N=1) {:.3g}, J(N_end) {:.3g} A/cm²".format(
                fit.exponent, fit.r_squared, J[1], J[-1]))
        # 相邻两条记录之间 J 的最大跳变（台阶）
        if n_last >= 1:
            jumps = J[1:] / J[:-1]
            print("  largest single-vacancy jump ×{:.2f} (at n = {}); jumps > ×1.5: {}".format(
                jumps.max(), int(np.argmax(jumps)) + 1, int(np.count_nonzero(jumps > 1.5))))
    # 4 条轨迹在公共 log t 网格上的几何平均：从各条第一次生成的最晚时刻到最早的终点
    runs = [d for d in data[label].values() if d["t"].size > 2]
    if len(runs) == len(SEEDS):
        start = max(d["t"][1] for d in runs)
        end = min(d["t"][-1] for d in runs)
        grid = np.logspace(np.log10(start), np.log10(end), 40)
        logs = [np.log10(d["J"][np.clip(np.searchsorted(d["t"], grid, side="right") - 1, 0, None)]) for d in runs]
        fit = fit_power_law(grid, 10 ** np.mean(logs, axis=0))
        print("== 4-seed geometric-mean J(t), t = {:.3g} … {:.3g} s: n = {:.2f}, R² = {:.3f}".format(
            start, end, fit.exponent, fit.r_squared))
        j0 = np.array([d["J"][0] for d in runs])
        print("== t = 0: J {} A/cm², max/min {:.3g}".format(" ".join("{:.3g}".format(v) for v in j0), j0.max() / j0.min()))

# ---- 图：上行 J(t)，下行 J 对生成空位面密度；左 20 nm、右 50 nm；纵轴共用 ----
figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.5), sharey=True, sharex="row")
for column, (label, _, side) in enumerate(BOXES):
    area_cm2 = (side * 1e-7) ** 2
    for seed, color in zip(SEEDS, COLORS):
        d = data[label].get(seed)
        if d is None or d["t"].size < 2:
            continue
        t, J = d["t"], d["J"]
        edges = np.append(t[1:], t[-1])
        axes[0, column].step(edges, np.append(J[1:], J[-1]), where="post", color=color, lw=2.0,
                             label="seed {}".format(seed))
        axes[0, column].plot([t[1]], [J[0]], ls="none", marker="o", ms=8, color=color, markeredgecolor="white")
        density = np.arange(J.size) / area_cm2
        axes[1, column].step(density[1:], J[1:], where="post", color=color, lw=2.0, label="seed {}".format(seed))
    axes[0, column].set_xscale("log")
    axes[0, column].set_xlabel("t (s)")
    axes[0, column].set_title(label, fontsize=11)
    axes[1, column].set_xscale("log")
    axes[1, column].set_xlabel(r"generated vacancies per area (cm$^{-2}$)")
for ax in axes.flat:
    ax.set_yscale("log")
    ax.axhline(BREAKDOWN, color="0.55", ls="--", lw=1.0)
    ax.grid(True, which="major", ls=":", lw=0.5, color="0.8")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
axes[0, 0].set_ylabel(r"J (A/cm$^2$)")
axes[1, 0].set_ylabel(r"J (A/cm$^2$)")
axes[0, 0].text(axes[0, 0].get_xlim()[0], BREAKDOWN * 1.15, " breakdown 10 A/cm²", color="0.4", fontsize=8, va="bottom")
axes[0, 1].legend(loc="lower right", fontsize=9, frameon=False)
figure.suptitle("3.6 V / 25 °C, J(t) to breakdown (dot: J at t = 0, drawn at the first generation)", fontsize=11)
figure.tight_layout()
figure.savefig(os.path.join(HERE, "compare_it.png"), dpi=130)
