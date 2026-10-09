"""t = 0 的 J 为什么在种子之间差几个数量级：3.6 V / 25 °C 下初始缺陷集合的稳态电流（簇级平均场，等于 TDDB 第 0 条记录）。

- 20×20×8 nm 小盒子（examples/zro2_tddb.yaml，10 个缺陷）seed 1–128，与默认盒子 50×50×8 nm（examples/single_run.yaml，
  60 个缺陷）seed 1–32 的分布；
- seed 1–4（TDDB 验收用的四个种子）逐缺陷拆开电流：哪几个缺陷承载、它们的 z、能级 E_D、到最近邻的距离、所在强耦合簇。
"""
import os
import sys
from dataclasses import replace

import numpy as np
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage.defects import minimum_image_delta  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import KMCSimulation  # noqa: E402
from kmc_leakage.mean_field import solve_mean_field  # noqa: E402

PER_CM2 = constants.centi ** 2
VOLTAGE = 3.6


def sweep(path, seeds, temperature=None):
    config = load_config(os.path.join(ROOT, "examples", path))
    sim_cfg = replace(config.simulation, voltage=VOLTAGE)
    solver = PoissonSolver(config.device, sim_cfg.grid)
    out = []
    for seed in seeds:
        c = replace(config, simulation=replace(sim_cfg, seed=seed))
        sim = KMCSimulation(c, solver=solver)
        mf = solve_mean_field(sim)
        assert mf.converged
        out.append((seed, sim, mf))
    return config, out


def summary(label, results):
    J = np.array([mf.current_density * PER_CM2 for _, _, mf in results])
    print("{}: M={} min {:.2g} p10 {:.2g} median {:.2g} p90 {:.2g} max {:.2g} | arithmetic mean {:.2g} "
          "geometric mean {:.2g} sigma(log10 J) {:.2f}".format(
              label, J.size, J.min(), np.percentile(J, 10), np.median(J), np.percentile(J, 90), J.max(), J.mean(),
              10 ** np.mean(np.log10(J)), np.std(np.log10(J))))
    return J


config_small, small = sweep("zro2_tddb.yaml", range(1, 129))
J_small = summary("20x20 nm, 3.6 V", small)
config_big, big = sweep("single_run.yaml", range(1, 33))
J_big = summary("50x50 nm, 3.6 V", big)
np.savez(os.path.join(HERE, "j0_3p6V.npz"), J_small=J_small, J_big=J_big)

device = config_small.device
nm, ev = constants.nano, constants.electron_volt
for seed, sim, mf in small[:4]:
    d = sim.defects
    contrib = mf.by_defect / mf.area * PER_CM2
    order = np.argsort(-np.abs(contrib))
    delta = minimum_image_delta(d.positions[:, None, :] - d.positions[None, :, :], device.Lx, device.Ly)
    dist = np.linalg.norm(delta, axis=-1) + np.eye(len(d)) * 1e9
    cluster_of = np.full(len(d), -1)
    for k, m in enumerate(sim.cluster_members):
        cluster_of[m] = k
    print("== seed {}: J(0) = {:.3g} A/cm² (TAT {:.3g}, PF {:.3g}, R1 {:.3g}); clusters {}".format(
        seed, mf.current_density * PER_CM2, mf.by_mechanism["tat"] / mf.area * PER_CM2,
        mf.by_mechanism["pf"] / mf.area * PER_CM2,
        (mf.by_mechanism["direct_fn"] + mf.by_mechanism["thermionic"]) / mf.area * PER_CM2,
        [m.tolist() for m in sim.cluster_members]))
    for i in order[:4]:
        j = int(np.argmin(dist[i]))
        print("   trap {:2d}: {:.3g} A/cm² ({:.0%})  z {:.2f} nm  E_D {:.3f} eV  p {:.3f}  nearest #{} at {:.2f} nm (z {:.2f}, E_D {:.3f})  cluster {}".format(
            i, contrib[i], contrib[i] / contrib.sum(), d.positions[i, 2] / nm, d.ED[i] / ev, mf.occupancy[i], j,
            dist[i, j] / nm, d.positions[j, 2] / nm, d.ED[j] / ev, cluster_of[i]))
    print("   all z (nm):", np.round(np.sort(d.positions[:, 2]) / nm, 2).tolist())
