"""加大盒子对 t = 0 电流散布的作用：3.6 V / 25 °C 下初始缺陷集合的稳态电流（簇级平均场，等于 TDDB 第 0 条记录）
在 L×L×8 nm 盒子（box.box_config：hx = 0.25 nm，其余同 examples/zro2_tddb.yaml）里 seed 1–M 的分布。

usage: j0_box_size.py L_NM M

逐种子记 J、承载最多的缺陷占的份额与它的 z；结束时打印统计量并存 j0_L{L}.npz。
"""
import os
import sys
import time
from dataclasses import replace

import numpy as np
from scipy import constants

from box import HERE, box_config
from kmc_leakage.electrostatics import PoissonSolver
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.mean_field import solve_mean_field
from memory import working_set_mb

PER_CM2 = constants.centi ** 2
side, count = float(sys.argv[1]), int(sys.argv[2])
config = box_config(side)
solver = PoissonSolver(config.device, config.simulation.grid)
J, share, z_top, n_defects = [], [], [], []
start = time.time()
for seed in range(1, count + 1):
    sim = KMCSimulation(replace(config, simulation=replace(config.simulation, seed=seed)), solver=solver)
    mf = solve_mean_field(sim)
    assert mf.converged
    contrib = mf.by_defect / mf.area * PER_CM2
    top = int(np.argmax(np.abs(contrib)))
    J.append(mf.current_density * PER_CM2)
    share.append(contrib[top] / contrib.sum())
    z_top.append(sim.defects.positions[top, 2] / constants.nano)
    n_defects.append(len(sim.defects))
    print("seed {:3d}: N {} J {:.4g} A/cm2, top trap {:.0%} at z {:.2f} nm | wall {:.0f} s, ws {:.0f}/{:.0f} MB".format(
        seed, n_defects[-1], J[-1], share[-1], z_top[-1], time.time() - start, *working_set_mb()), flush=True)

J = np.array(J)
log_j = np.log10(J)
print("{:g}x{:g} nm, 3.6 V: M={} N={} | min {:.2g} p10 {:.2g} median {:.2g} p90 {:.2g} max {:.2g} | arithmetic mean {:.2g} "
      "geometric mean {:.2g} sigma(log10 J) {:.2f} | top-trap share median {:.0%}".format(
          side, side, J.size, n_defects[0], J.min(), np.percentile(J, 10), np.median(J), np.percentile(J, 90), J.max(),
          J.mean(), 10 ** log_j.mean(), log_j.std(), np.median(share)), flush=True)
np.savez(os.path.join(HERE, "j0_L{:g}.npz".format(side)), J=J, share=np.array(share), z_top=np.array(z_top),
         n_defects=np.array(n_defects))
