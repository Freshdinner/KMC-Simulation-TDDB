"""what-if：把 depletion_scan.py 的 N(t)（前驱消耗 + E_A 高斯分布，E_A 均值平移到盒内 N_BD 个空位落在 1000 s）
与 ADR-051 四条 50 nm 轨迹的 J(N) 拼起来，看 I-t 在双对数下是否变直。忽略消耗对 J(N) 的影响（两种情形下
新空位的平均深度都约 1.06 nm）。
usage: compose_it.py
"""
import csv
import os
import sys

import numpy as np
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from kmc_leakage.tddb import fit_power_law  # noqa: E402

d = np.load(os.path.join(HERE, "spectrum_s1.npz"))
kT, eV = constants.Boltzmann * float(d["T"]), constants.electron_volt
lr = 0.5 * (d["log_rate"][:-1] + d["log_rate"][1:]).astype(float)
g0 = 0.5 * (d["barrier"][:-1] + d["barrier"][1:]).astype(float)
vol = float(d["column_area"] * d["dz"])
H, er, eg = np.histogram2d(lr.ravel(), g0.ravel(), bins=(240, 120))
ii, jj = np.nonzero(H)
w_vol, w_lr, w_g = H[ii, jj] * vol, 0.5 * (er[1:] + er[:-1])[ii], 0.5 * (eg[1:] + eg[:-1])[jj]
xq, wq = np.polynomial.hermite_e.hermegauss(64)
wq = wq / wq.sum()
times = np.logspace(-40, 12, 1041)


def n_of_t(N_T, sigma):
    ln_k = w_lr[:, None] - np.maximum(0.0, w_g[:, None] + sigma * xq[None, :] * eV / kT)
    return np.array([N_T * np.sum(w_vol[:, None] * wq[None, :] * -np.expm1(-np.exp(ln_k) * t)) for t in times])


def load(seed):
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "10.09", "accept50", "steps_s{}.csv".format(seed)),
                                    encoding="utf-8")))
    return np.array([float(r["J_A_cm2"]) for r in rows])


def fit(ts, Js):
    f = fit_power_law(ts, Js)
    h = ts.size // 2
    return f, fit_power_law(ts[:h], Js[:h]).exponent, fit_power_law(ts[h:], Js[h:]).exponent


curves = {}
for N_T, sigma in ((1e20, 0.0), (1e20, 0.1), (3e20, 0.1), (1e20, 0.15), (1e21, 0.1)):
    curves[(N_T, sigma)] = n_of_t(N_T * 1e6, sigma)
for seed in (1, 2, 3, 4):
    J = load(seed)
    N_BD = J.size - 1
    print("== seed {}: N_BD = {}, J0 = {:.2f} A/cm²".format(seed, N_BD, J[0]))
    for (N_T, sigma), N in curves.items():
        tb = np.exp(np.interp(np.log(N_BD), np.log(np.maximum(N, 1e-300)), np.log(times)))
        t = times * 1000.0 / tb  # 平移到 t(N_BD) = 1000 s（E_A 均值平移）
        shift = kT * np.log(1000.0 / tb) / eV
        out = []
        for lo in (0.1, 1.0, 10.0):
            ts = np.logspace(np.log10(lo), 3.0, int(20 * (3 - np.log10(lo))) + 1)
            Nk = np.floor(np.interp(np.log(ts), np.log(t), N) + 1e-9).astype(int)  # 每个时刻已生成的空位数
            Js = J[np.clip(Nk, 0, N_BD)]
            f, a, b = fit(ts, Js)
            out.append("[{:g},1000] n={:.2f} R²={:.2f} 前/后 {:.2f}/{:.2f}".format(lo, f.exponent, f.r_squared, a, b))
        n1 = np.interp(np.log(1.0), np.log(t), N)
        print("  N_T={:.0e} σ={:.2f} ΔE_A={:+.2f} eV, N(1 s)={:.1f}: ".format(N_T, sigma, shift, n1) + "; ".join(out))
