"""ADR-049：缺陷能级改为定值 1.15 eV 后，3.6 V / 25 °C 下 t = 0 的 J（初始缺陷集合的稳态电流，簇级平均场）在种子之间的散布，
与高斯能级（σ_E = 0.15 eV，`j0_spread.log`）对比。小盒子用 examples/zro2_tddb.yaml（现为定值）；默认盒子用 single_run.yaml
把能级换成同一个定值作参照。
"""
import os
import sys
from dataclasses import replace

import numpy as np
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage.config import DefectLevelConfig  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import KMCSimulation  # noqa: E402
from kmc_leakage.mean_field import solve_mean_field  # noqa: E402

PER_CM2 = constants.centi ** 2


def fixed_level(config):
    stack = config.device.stack
    layer = stack.layers[0]
    level = DefectLevelConfig("fixed", layer.defects.level.ED)
    layers = (replace(layer, defects=replace(layer.defects, level=level)),) + tuple(stack.layers[1:])
    return replace(config, device=replace(config.device, stack=replace(stack, layers=layers)))


def sweep(config, seeds):
    config = replace(config, simulation=replace(config.simulation, voltage=3.6))
    solver = PoissonSolver(config.device, config.simulation.grid)
    J = []
    for seed in seeds:
        sim = KMCSimulation(replace(config, simulation=replace(config.simulation, seed=seed)), solver=solver)
        mf = solve_mean_field(sim)
        assert mf.converged
        J.append(mf.current_density * PER_CM2)
    return np.array(J)


def summary(label, J):
    print("{}: M={} min {:.2g} p10 {:.2g} median {:.2g} p90 {:.2g} max {:.2g} | arithmetic mean {:.2g} "
          "geometric mean {:.2g} sigma(log10 J) {:.2f}".format(
              label, J.size, J.min(), np.percentile(J, 10), np.median(J), np.percentile(J, 90), J.max(), J.mean(),
              10 ** np.mean(np.log10(J)), np.std(np.log10(J))), flush=True)


small = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
assert small.defects.level.mode == "fixed"
J_small = sweep(small, range(1, 129))
summary("20x20 nm, fixed E_D = 1.15 eV, 3.6 V", J_small)
print("  seed 1-4:", " ".join("{:.3g}".format(v) for v in J_small[:4]))
J_big = sweep(fixed_level(load_config(os.path.join(ROOT, "examples", "single_run.yaml"))), range(1, 33))
summary("50x50 nm, fixed E_D = 1.15 eV, 3.6 V", J_big)
np.savez(os.path.join(HERE, "j0_fixed_level_3p6V.npz"), J_small=J_small, J_big=J_big)
