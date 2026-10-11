"""depletion.py 的扫描版：E_A 均值只平移时间标尺（P_bb 未饱和时严格成立），所以对每组 (N_T, σ_A) 先在 E_A 均值不变
时算 N(t)，再给出 (1) N 从 1 到 N_BD 的时间跨度与这段的 α = dlnN/dlnt，(2) t(N_BD) = 1000 s 需要的 E_A 均值平移。
usage: depletion_scan.py [SEED] [N_BD]
"""
import os
import sys

import numpy as np
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N_BD = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
d = np.load(os.path.join(HERE, "spectrum_s{}.npz".format(seed)))
kT = constants.Boltzmann * float(d["T"])
eV = constants.electron_volt
log_rate, barrier = d["log_rate"].astype(float), d["barrier"].astype(float)
ln_r = 0.5 * (log_rate[:-1] + log_rate[1:])
g0 = 0.5 * (barrier[:-1] + barrier[1:])
z_mid = (0.5 * (d["levels"][:-1] + d["levels"][1:]) / constants.nano)[:, None] * np.ones_like(ln_r)
vol = float(d["column_area"] * d["dz"])
H, er, eg = np.histogram2d(ln_r.ravel(), g0.ravel(), bins=(240, 120))
Hz, _, _ = np.histogram2d(ln_r.ravel(), g0.ravel(), bins=(er, eg), weights=z_mid.ravel())
ii, jj = np.nonzero(H)
w_vol = H[ii, jj] * vol
w_lr = 0.5 * (er[1:] + er[:-1])[ii]
w_g = 0.5 * (eg[1:] + eg[:-1])[jj]
w_z = Hz[ii, jj] / H[ii, jj]
x, wq = np.polynomial.hermite_e.hermegauss(64)
wq = wq / wq.sum()
times = np.logspace(-40, 12, 521)


def curves(N_T, sigma):
    dE = sigma * x
    ln_k = w_lr[:, None] - np.maximum(0.0, w_g[:, None] + dE[None, :] * eV / kT)
    N = np.empty(times.size)
    zbar = np.empty(times.size)
    for m, t in enumerate(times):
        p = w_vol[:, None] * wq[None, :] * -np.expm1(-np.exp(ln_k) * t)
        N[m] = N_T * p.sum()
        zbar[m] = (p.sum(axis=1) * w_z).sum() / max(p.sum(), 1e-300)
    return N, zbar


print("seed {}, N_BD = {:.0f}（50×50 nm 盒子内的生成空位数）".format(seed, N_BD))
print("{:>8s} {:>6s} | {:>9s} {:>13s} {:>14s} {:>14s} | {:>12s}".format(
    "N_T", "σ_A", "跨度[dec]", "α(N=1→N_BD)", "α 在 N=10/50", "<z>(N_BD)[nm]", "ΔE_A[eV]"))
for N_T in (1e20, 3e20, 1e21, 3e21, 1e22):
    for sigma in (0.0, 0.05, 0.1, 0.15, 0.2):
        N, zbar = curves(N_T * 1e6, sigma)
        lnN = np.log(np.maximum(N, 1e-300))
        alpha = np.gradient(lnN, np.log(times))
        if N[-1] < N_BD:
            print("{:8.0e} {:6.2f} | 盒内前驱不够 N_BD（最多 {:.0f}）".format(N_T, sigma, N[-1]))
            continue
        t1 = np.interp(0.0, lnN, np.log(times))
        tb = np.interp(np.log(N_BD), lnN, np.log(times))
        seg = (lnN >= 0.0) & (lnN <= np.log(N_BD))
        a10 = np.interp(np.log(10.0), lnN, alpha)
        a50 = np.interp(np.log(50.0), lnN, alpha)
        shift = kT * (np.log(1000.0) - tb) / eV
        print("{:8.0e} {:6.2f} | {:9.2f} {:6.2f}–{:<6.2f} {:6.2f}/{:<6.2f}  {:12.2f}   | {:+10.3f}".format(
            N_T, sigma, (tb - t1) / np.log(10), alpha[seg].min(), alpha[seg].max(), a10, a50,
            np.interp(np.log(N_BD), lnN, zbar), shift))
