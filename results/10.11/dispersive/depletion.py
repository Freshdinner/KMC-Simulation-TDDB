"""估算"前驱逐个消耗 + E_A 高斯分布"下的生成空位数 N(t)（不改 src/）。

用 rate_spectrum.py 存下的 t = 0 单前驱生成速率谱 k₀(x) = R_1e,电极(x)·P_bb(x)（E_A 取配置均值），假定新空位
不改变场与占据（忽略反馈、陷阱项与禁区），E_A 在前驱之间按 N(μ_A, σ_A) 分布、每个前驱断键一次即消耗：

    N(t) = N_T·∫d³x ∫dE_A g(E_A)·[1 − exp(−k(x, E_A)·t)],  k = R_1e·exp(−max(0, g₀ + (E_A − E_A,0)/k_BT))

输出 N(t) 的局部斜率 α = dlnN/dlnt 与到达 N_BD 的时刻。
usage: depletion.py [SEED]
"""
import os
import sys

import numpy as np
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
d = np.load(os.path.join(HERE, "spectrum_s{}.npz".format(seed)))
kT = constants.Boltzmann * float(d["T"])
eV = constants.electron_volt
log_rate, barrier = d["log_rate"].astype(float), d["barrier"].astype(float)
levels = d["levels"] / constants.nano
# 段中点（对数线性的几何平均），段体积 = 列面积 × dz
ln_r = 0.5 * (log_rate[:-1] + log_rate[1:])
g0 = 0.5 * (barrier[:-1] + barrier[1:])
z_mid = 0.5 * (levels[:-1] + levels[1:])
vol = float(d["column_area"] * d["dz"])
N_T0, area_nm2 = float(d["N_T"]), float(d["area"]) / constants.nano ** 2
print("check: N_T*sum k0*V = {:.4g} 1/s (cells table {:.4g}); J0 = {:.3g} A/cm2".format(
    N_T0 * vol * np.exp(ln_r - np.maximum(0.0, g0)).sum(), float(d["weights_sum"]), float(d["J0"]) * 1e-4))
print("z profile (lateral median): z [nm], ln R_1e, E_eff [eV], k0 per precursor [1/s]")
for z in (0.55, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0):
    k = int(np.argmin(np.abs(z_mid - z)))
    print("  {:.2f}  {:7.2f}  {:.3f}  {:.3g}".format(z_mid[k], np.median(ln_r[k]), np.median(g0[k]) * kT / eV,
                                                     np.median(np.exp(ln_r[k] - np.maximum(0, g0[k])))))

# 体积按 (ln R_1e, g₀) 二维直方图归并，降低计算量
lr, gg = ln_r.ravel(), g0.ravel()
H, er, eg = np.histogram2d(lr, gg, bins=(240, 120))
cr, cg = 0.5 * (er[1:] + er[:-1]), 0.5 * (eg[1:] + eg[:-1])
ii, jj = np.nonzero(H)
w_vol, w_lr, w_g = H[ii, jj] * vol, cr[ii], cg[jj]  # m³


def n_of_t(times, N_T, dE_mean, sigma):
    """dE_mean：E_A 均值相对配置值的平移 [eV]；sigma：E_A 标准差 [eV]。"""
    if sigma > 0:
        x, wq = np.polynomial.hermite_e.hermegauss(48)
        dE, wq = dE_mean + sigma * x, wq / wq.sum()
    else:
        dE, wq = np.array([dE_mean]), np.array([1.0])
    ln_k = w_lr[:, None] - np.maximum(0.0, w_g[:, None] + dE[None, :] * eV / kT)  # (bins, quad)
    out = np.empty(times.size)
    for m, t in enumerate(times):
        out[m] = N_T * np.sum(w_vol[:, None] * wq[None, :] * -np.expm1(-np.exp(ln_k) * t))
    return out


times = np.logspace(-6, 8, 141)
N_BD = 100.0
cases = [
    ("当前：N_T=3e18，σ=0，无消耗（N=ΣG·t）", None),
    ("N_T=3e18, σ=0", (3e18, 0.0, 0.0)),
    ("N_T=1e20, σ=0", (1e20, 0.0, 0.0)),
    ("N_T=1e21, σ=0", (1e21, 0.0, 0.0)),
    ("N_T=1e20, σ=0.1", (1e20, 0.0, 0.1)),
    ("N_T=1e21, σ=0.1", (1e21, 0.0, 0.1)),
    ("N_T=1e21, σ=0.2", (1e21, 0.0, 0.2)),
    ("N_T=1e22, σ=0.2", (1e22, 0.0, 0.2)),
    ("N_T=1e21, σ=0.3", (1e21, 0.0, 0.3)),
    ("N_T=1e22, σ=0.3", (1e22, 0.0, 0.3)),
]
for label, p in cases:
    if p is None:
        N = float(d["weights_sum"]) * times
    else:
        N = n_of_t(times, p[0] * 1e6, p[1], p[2])
    N_box = N  # 50×50 nm 盒子里的个数
    alpha = np.gradient(np.log(np.maximum(N_box, 1e-300)), np.log(times))
    reach = np.flatnonzero(N_box >= N_BD)
    t_bd = times[reach[0]] if reach.size else float("nan")
    # N 从 1 到 N_BD 的时间跨度与这段的 α 范围
    one = np.flatnonzero(N_box >= 1.0)
    span = ""
    if reach.size and one.size:
        seg = slice(one[0], reach[0] + 1)
        span = "N:1→{:.0f} 跨 {:.1f} 个数量级时间，α {:.2f}–{:.2f}".format(
            N_BD, np.log10(times[reach[0]] / times[one[0]]), alpha[seg].min(), alpha[seg].max())
    total = p[0] * 1e6 * float(d["area"]) * 7e-9 if p else float("inf")
    print("{:34s} t(N=1)={:.3g} s, t(N={:.0f})={:.3g} s; {}; 盒内前驱总数 {:.3g}".format(
        label, times[one[0]] if one.size else float("nan"), N_BD, t_bd, span, total))
