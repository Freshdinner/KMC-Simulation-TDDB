"""平均场 Jacobian 的结构探查：为"按稀疏结构求 Jacobian"做依据。

usage: probe.py STATE_NPZ R_C_NM

在续跑状态（`tddb.save_state`）的缺陷集合上建内层（输运温度取状态里的值），记录：
1. 基线：`solve_mean_field` 从状态里的占据热启动的耗时、求值次数、牛顿步数与残差；
2. 一次 F（`mean_field_flux`）的耗时与 cProfile 前 15 项；
3. 不动点 x* = ln p* 处的完整有限差分 Jacobian ∂G/∂x（与 `mean_field._jacobian` 相同），按缺陷间距分箱统计 |J_ij| 的最大值
   与 Frobenius 范数的占比，以及只留间距 ≤ R 的元素时，牛顿方向与完整 Jacobian 给出的方向相差多少。
结果写 `probe_{TAG}.npz`（Jacobian、距离、x*）与标准输出。
"""
import cProfile
import io
import os
import pstats
import sys
import time
from dataclasses import replace

import numpy as np
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage import mean_field as M  # noqa: E402
from kmc_leakage.defects import minimum_image_delta  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import KMCSimulation  # noqa: E402
from kmc_leakage.tddb import load_state  # noqa: E402

path, r_c_nm = sys.argv[1], float(sys.argv[2])
state = load_state(path)
tag = os.path.splitext(os.path.basename(path))[0]
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
config = replace(config, simulation=replace(config.simulation, temperature=state.transport_temperature),
                 tddb=replace(config.tddb, percolation_distance=r_c_nm * constants.nano))
began = time.time()
sim = KMCSimulation(config, defects=state.defects)
n = sim.n_defects
print("{}: {} defects, inner built in {:.0f} s; clusters {} (max size {})".format(
    tag, n, time.time() - began, len(sim.cluster_members), max((m.size for m in sim.cluster_members), default=1)), flush=True)

began = time.time()
mf = M.solve_mean_field(sim, state.occupancy)
print("baseline solve: {:.0f} s, evaluations {}, newton steps {}, residual {:.1e}, converged {}".format(
    time.time() - began, mf.iterations, mf.newton_steps, mf.residual, mf.converged), flush=True)

x = M._log_occupancy(mf.occupancy)
began = time.time()
current = M._evaluate(sim, x)
print("one F: {:.2f} s".format(time.time() - began), flush=True)
profile = cProfile.Profile()
profile.enable()
M._evaluate(sim, x)
profile.disable()
stream = io.StringIO()
pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(15)
print(stream.getvalue(), flush=True)

began = time.time()
jacobian = M._jacobian(sim, current, [np.array([j]) for j in range(n)])  # 每组一个缺陷：逐列有限差分
print("full FD Jacobian: {:.0f} s ({} evaluations)".format(time.time() - began, n), flush=True)
pos = sim.defects.positions
delta = minimum_image_delta(pos[:, None, :] - pos[None, :, :], sim.device.Lx, sim.device.Ly)
distance = np.linalg.norm(delta, axis=-1) / constants.nano
np.savez(os.path.join(HERE, "probe_{}.npz".format(tag)), jacobian=jacobian, distance=distance, x=x, gap=current.gap)

off = ~np.eye(n, dtype=bool)
total = np.sum(jacobian[off] ** 2)
print("off-diagonal |J_ij| by distance (nm): bin, count, max, share of off-diagonal Frobenius^2")
edges = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 40]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = off & (distance >= lo) & (distance < hi)
    if m.any():
        print("  [{:>4}, {:>4}) {:6d}  {:.2e}  {:.2e}".format(lo, hi, int(m.sum()), np.abs(jacobian[m]).max(),
                                                              np.sum(jacobian[m] ** 2) / total))
print("diagonal: min {:.3f} median {:.3f} max {:.3f}".format(*np.percentile(np.diag(jacobian), [0, 50, 100])))
full = np.linalg.solve(jacobian, -current.gap) if current.residual > 0 else None
for radius in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
    kept = np.where(distance <= radius, jacobian, 0.0)
    neighbours = (distance <= 2 * radius).sum(axis=1).max()
    cond = np.linalg.cond(kept)
    line = "  R = {:.1f} nm: kept {:.1f}% of entries, Frobenius^2 kept {:.4f}; max defects within 2R {} ; cond {:.3g}".format(
        radius, 100 * np.mean(distance <= radius), np.sum(kept[off] ** 2) / total, int(neighbours), cond)
    # 截断 Jacobian 作牛顿迭代矩阵时的线性收敛因子：I − J_R⁻¹ J 的谱半径
    rho = np.max(np.abs(np.linalg.eigvals(np.eye(n) - np.linalg.solve(kept, jacobian))))
    line += "; spectral radius of I - J_R^-1 J {:.3g}".format(rho)
    print(line, flush=True)
