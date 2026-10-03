"""ADR-045 数值抽查：在轨迹的记录点上重建状态，查生成事件表的网格依赖。

usage: check_states.py SEED N [N ...]

状态取 `drive.py` 每 50 个空位存的快照 `snap_s{SEED}_n{NNNN}.npz`（缺陷位置、能级、这一步的平均场占据、
输运温度、已满事件标记）。占据直接用快照里的不动点，放热与温度场按它重算（与 run_tddb 同一套函数），再求事件表：
1. 重建自洽：生成网格 16×16×70、按快照的已满标记置零后的 ΣG，与轨迹逐步记录（steps_s{SEED}.csv）的 ΣG 比；
2. 生成网格无关：16 / 32 格（z 70 格）的 ΣG_单元、ΣG_陷阱 与电极项沿 z 的份额（z < 1、1–2、> 2 nm）；
3. 泊松网格：80³ → 160³（温度场在 160³ 上重解；内层占据与陷阱的平均场能级仍取 80³ 上的解，同 ADR-044 验收）。
"""
import os
import sys
import time
from dataclasses import replace

import numpy as np
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage.config import GridConfig  # noqa: E402
from kmc_leakage.defects import DefectEnsemble  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import KMCSimulation  # noqa: E402
from kmc_leakage.tddb import generation_cells, generation_rates  # noqa: E402
from kmc_leakage.thermal import heat_release, power_map, solve_temperature  # noqa: E402

nm = constants.nano
seed = int(sys.argv[1])
base = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
base = replace(base, simulation=replace(base.simulation, seed=seed))
device, tddb, ambient = base.device, base.tddb, base.simulation.temperature
steps = np.genfromtxt(os.path.join(HERE, "steps_s{}.csv".format(seed)), delimiter=",", names=True,
                      dtype=None, encoding="utf-8")


def table_sums(inner, p, grid, solver=None):
    """(ΣG_单元 未置零, ΣG_陷阱 未置零, 单元事件 + 源陷阱事件的整张速率表, 电极项沿 z 的三段份额, 墙钟)。"""
    solver = inner.solver if solver is None else solver
    heat = heat_release(inner, p)
    temperature = solve_temperature(solver, power_map(solver, inner.defects.positions, heat.defect),
                                    tddb.thermal_conductivity, ambient)
    original = inner.solver
    inner.solver = solver
    tick = time.time()
    try:
        table = generation_rates(inner, p, tddb, generation_cells(device, grid), temperature)
    finally:
        inner.solver = original
    wall = time.time() - tick
    cells = table.cells
    layers = cells._levels.size - 1
    middle = cells._levels[:-1] + 0.5 * cells._dz
    by_layer = cells._weights.reshape(layers, -1).sum(axis=1)
    total = by_layer.sum()
    shares = [by_layer[middle < 1 * nm].sum() / total, by_layer[(middle >= 1 * nm) & (middle < 2 * nm)].sum() / total,
              by_layer[middle >= 2 * nm].sum() / total]
    return (float(cells.rates.sum()), float(table.traps.rates.sum()),
            np.concatenate([cells.rates, table.traps.rates]), shares, wall)


for n in (int(v) for v in sys.argv[2:]):
    snap = np.load(os.path.join(HERE, "snap_s{}_n{:04d}.npz".format(seed, n)))
    material = device.stack.single_material
    defects = DefectEnsemble.uniform(snap["positions"], snap["ED"], material,
                                     band_reference=float(device.layer_band_references()[0]),
                                     charge_state=device.stack.layers[0].defects.charge_state)
    config = replace(base, simulation=replace(base.simulation, temperature=float(snap["transport_temperature"])))
    inner = KMCSimulation(config, defects=defects)
    p = snap["occupancy"]
    recorded = steps[steps["n_generated"] == n][0]

    cells16, traps16, events16, shares16, wall16 = table_sums(inner, p, (16, 16, 70))
    live = float(np.where(snap["full"], 0.0, events16).sum())
    print("s{} n={} N={} T_transport={:.6f} K | ΣG rebuilt (masked) {:.9e} vs recorded {:.9e} (rel {:+.1e}); "
          "full events {}".format(seed, n, inner.n_defects, float(snap["transport_temperature"]), live,
                                  recorded["G_total_s"], live / recorded["G_total_s"] - 1.0, int(snap["full"].sum())),
          flush=True)
    print("  gen 16x16x70 : cells {:.9e} traps {:.9e} total {:.9e} | cell share z<1 {:.6f} 1-2 {:.6f} >2 {:.3e} | "
          "wall {:.1f}s".format(cells16, traps16, cells16 + traps16, *shares16, wall16), flush=True)
    cells32, traps32, _, shares32, wall32 = table_sums(inner, p, (32, 32, 70))
    print("  gen 32x32x70 : cells {:.9e} traps {:.9e} total {:.9e} | cell share z<1 {:.6f} 1-2 {:.6f} >2 {:.3e} | "
          "wall {:.1f}s".format(cells32, traps32, cells32 + traps32, *shares32, wall32), flush=True)
    print("  16 vs 32     : cells rel {:+.1e}, traps rel {:+.1e} (bitwise {}), total rel {:+.1e}, max share diff {:.1e}"
          .format(cells32 / cells16 - 1.0, traps32 / traps16 - 1.0 if traps16 > 0 else 0.0, traps32 == traps16,
                  (cells32 + traps32) / (cells16 + traps16) - 1.0, np.max(np.abs(np.subtract(shares32, shares16)))),
          flush=True)
    fine = PoissonSolver(device, GridConfig(160, 160, 160))
    cells160, traps160, _, shares160, wall160 = table_sums(inner, p, (16, 16, 70), solver=fine)
    print("  poisson 160^3: cells {:.9e} ({:+.2%}) traps {:.9e} ({:+.2%}) total {:.9e} ({:+.2%}) | "
          "cell share z<1 {:.6f} 1-2 {:.6f} >2 {:.3e} | wall {:.1f}s".format(
              cells160, cells160 / cells16 - 1.0, traps160, traps160 / traps16 - 1.0 if traps16 > 0 else 0.0,
              cells160 + traps160, (cells160 + traps160) / (cells16 + traps16) - 1.0, *shares160, wall160), flush=True)
