"""PF 发射主导电流（ADR-045）对泊松网格的依赖：同一缺陷组态在泊松 80³ 与 160³ 上各解一次平均场，比较各机制电流。

usage: pf_grid_check.py DIR SEED N [N ...]
（DIR 为快照所在目录，相对仓库根，如 results/10.03/rerun）

R8 用缺陷处的局域场（含近邻带电空位的 CIC 近场），PF 势垒降低 ∝ √F，电流随它指数变化；ADR-043 实测 +e 在 0.5 nm 处的
CIC 场在 0.25 nm 网格上横向偏强 45%、0.125 nm 网格上 +13%。平均场从快照的占据热启动，输运温度取快照记录值。
另列 R8 用到的局域场：生成空位里按 z < 1 nm / ≥ 1 nm 分开的中位数（80³）。
"""
import os
import sys
import time
from dataclasses import replace

import numpy as np
from scipy import constants

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage.config import GridConfig  # noqa: E402
from kmc_leakage.defects import DefectEnsemble  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import MECHANISMS, KMCSimulation  # noqa: E402
from kmc_leakage.mean_field import solve_mean_field  # noqa: E402

nm, per_cm2 = constants.nano, constants.centi ** 2
directory, seed = sys.argv[1], int(sys.argv[2])
base = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
device = base.device
material = device.stack.single_material

for n in (int(v) for v in sys.argv[3:]):
    snap = np.load(os.path.join(ROOT, directory, "snap_s{}_n{:04d}.npz".format(seed, n)))
    n_initial = int(snap["n_initial"])
    defects = DefectEnsemble.uniform(snap["positions"], snap["ED"], material,
                                     band_reference=float(device.layer_band_references()[0]),
                                     charge_state=device.stack.layers[0].defects.charge_state)
    for grid in (base.simulation.grid, GridConfig(160, 160, 160)):
        config = replace(base, simulation=replace(base.simulation, seed=seed, grid=grid,
                                                  temperature=float(snap["transport_temperature"])))
        tick = time.time()
        inner = KMCSimulation(config, defects=defects)
        mf = solve_mean_field(inner, snap["occupancy"])
        shares = {m: mf.by_mechanism[m] / mf.current for m in MECHANISMS}
        line = "s{} n={} grid {}^3: J {:.4e} A/cm2 | ".format(seed, n, grid.nx, mf.current_density * per_cm2) + \
            " ".join("{} {:.3f}".format(m, shares[m]) for m in MECHANISMS) + \
            " | empty generated {:.1f} | it {} res {:.1e} | {:.0f} s".format(
                float(np.sum(1.0 - mf.occupancy[n_initial:])), mf.iterations, mf.residual, time.time() - tick)
        if grid is base.simulation.grid:
            _, field, _ = inner._electrostatics(mf.occupancy)
            z = snap["positions"][n_initial:, 2]
            f = field[n_initial:] * nm  # V/m -> V/nm
            line += " | local |F| median z<1 nm {:.2f} V/nm, z>=1 nm {:.2f} V/nm".format(
                float(np.median(f[z < 1 * nm])), float(np.median(f[z >= 1 * nm])) if np.any(z >= 1 * nm) else float("nan"))
        print(line, flush=True)
