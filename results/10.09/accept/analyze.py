"""ADR-048 验收汇总：ΣG 是否自加速、平均场是否每步收敛、I-t 双对数拟合（n、R²）、击穿时刻。"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
from kmc_leakage.tddb import fit_power_law  # noqa: E402

for seed in (1, 2, 3, 4):
    rows = list(csv.DictReader(open(os.path.join(HERE, "steps_s{}.csv".format(seed)))))
    t = np.array([float(r["t_s"]) for r in rows])
    J = np.array([float(r["J_A_cm2"]) for r in rows])
    G = np.array([float(r["G_total_s"]) for r in rows])
    conv = np.array([int(r["mf_converged"]) for r in rows])
    newton = np.array([int(r["mf_newton_steps"]) for r in rows])
    it = np.array([int(r["mf_iterations"]) for r in rows])
    tat = np.array([float(r["J_tat_A_cm2"]) for r in rows])
    pf = np.array([float(r["J_pf_A_cm2"]) for r in rows])
    top = np.array([float(r["generated_top_nm"]) for r in rows])
    med = np.array([float(r["generated_median_z_nm"]) for r in rows])
    kinds = [r["birth_event"] for r in rows[1:]]
    print("== seed {}: {} records, t_BD = {:.4f} s at n = {}".format(seed, len(rows), t[-1], len(rows) - 1))
    print("  ΣG: min {:.0f} max {:.0f} s^-1, first {:.0f} last {:.0f}; mean field converged {}/{}, newton {} records, max it {}".format(
        G.min(), G.max(), G[0], G[-1], conv.sum(), conv.size, np.count_nonzero(newton), it.max()))
    print("  J: " + " ".join("{:.2g}".format(v) for v in J))
    print("  TAT share at end {:.2f}, PF {:.2f}; generated top z {:.2f} nm, median z {:.2f} nm; trap-term births {}/{}".format(
        tat[-1] / J[-1], pf[-1] / J[-1], top[-1], med[-1], kinds.count("trap"), len(kinds)))
    # 分段常数 J(t)：按 log t 等间距取样（与 fit_transient 同口径），窗口从第 k 次生成起到击穿
    end = t[-1]
    for k in (1, 2, 3, 5):
        if k >= t.size - 1:
            continue
        start = t[k]
        times = np.logspace(np.log10(start), np.log10(end), max(2, int(np.ceil(np.log10(end / start) * 20)) + 1))
        idx = np.clip(np.searchsorted(t, times, side="right") - 1, 0, None)
        fit = fit_power_law(times, J[idx])
        print("  fit from n={} (t={:.3g} s): n = {:.2f}, R² = {:.3f}".format(k, start, fit.exponent, fit.r_squared))

# J 对生成空位数 N 的双对数（ΣG 近似恒定时 N ∝ t）与 4 条轨迹在公共 log t 网格上的平均
print("== J vs N (generated vacancies, N >= 1)")
grid = np.logspace(-3, np.log10(0.0396), 40)
logs = []
for seed in (1, 2, 3, 4):
    rows = list(csv.DictReader(open(os.path.join(HERE, "steps_s{}.csv".format(seed)))))
    t = np.array([float(r["t_s"]) for r in rows])
    J = np.array([float(r["J_A_cm2"]) for r in rows])
    N = np.arange(J.size)
    fit = fit_power_law(N[1:], J[1:])
    print("  seed {}: J ∝ N^{:.2f}, R² = {:.3f}; J(N=1) {:.3g}, J(N_BD) {:.3g} A/cm²".format(seed, fit.exponent, fit.r_squared, J[1], J[-1]))
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, None)
    logs.append(np.log10(J[idx]))
mean_log = np.mean(logs, axis=0)
fit = fit_power_law(grid, 10 ** mean_log)
print("  4-seed geometric-mean J(t), t = 1e-3 … 0.0396 s: n = {:.2f}, R² = {:.3f}".format(fit.exponent, fit.r_squared))
