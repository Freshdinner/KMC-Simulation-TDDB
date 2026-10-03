"""ADR-044 验收：单元电极项改为对单元内可放置区域的积分之后，生成事件表的网格依赖与求积收敛。

usage: acceptance.py STATE

状态同 ADR-042（`../adr042/common.py`），等温，σ = 0.5 nm（ADR-043），全部走生产代码 `tddb.generation_rates`。
每行给 ΣG_单元、ΣG_陷阱、合计、电极项沿 z 的份额（z < 1、1–2、> 2 nm）与按事件表抽 4000 次的新空位 z 中位数、
90% 分位与 z > 2 nm 的比例，以及求一次事件表的墙钟时间：
1. 生成网格横向 16 / 32 / 48 格（z 70 格）——验收；
2. 生成网格 16 格下求积参数逐项加密：列间距 0.08 → 0.0625 nm（列间距的趋势另见 `column_trend.py`）、z 节点间距
   与插值表的能级、温度步长减半；
3. 生成网格 16 格下泊松网格 80³ → 160³（内层占据与陷阱能级仍取 80³ 上的解）。
ADR-043 的单元中心规则在 128×128×140 上的值见 `../adr043/acceptance_<状态>.log`，作对照。
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adr042"))

import numpy as np
from scipy import constants

from common import load_state
from kmc_leakage import tddb as tddb_module
from kmc_leakage.config import GridConfig
from kmc_leakage.electrostatics import PoissonSolver
from kmc_leakage.tddb import generation_cells, generation_rates

nm = constants.nano
name = sys.argv[1]
config, inner, p, mf_wall = load_state(name)
device, tddb = config.device, config.tddb
print("{}: N={} defects, empty (p<0.5) {}, mean-field {:.1f}s".format(
    name, inner.n_defects, int(np.sum(p < 0.5)), mf_wall), flush=True)


def report(label, grid=(16, 16, 70), solver=None):
    cells = generation_cells(device, grid)
    original = inner.solver
    if solver is not None:
        inner.solver = solver
    tick = time.time()
    try:
        table = generation_rates(inner, p, tddb, cells)
    finally:
        inner.solver = original
    wall = time.time() - tick
    electrode, traps = table.cells, table.traps
    total_cells = electrode.rates.sum()
    layers = electrode._levels.size - 1
    middle = electrode._levels[:-1] + 0.5 * electrode._dz
    by_layer = electrode._weights.reshape(layers, -1).sum(axis=1)
    shares = [by_layer[middle < 1 * nm].sum(), by_layer[(middle >= 1 * nm) & (middle < 2 * nm)].sum(),
              by_layer[middle >= 2 * nm].sum()]
    rng = np.random.default_rng(7)
    events = np.concatenate([electrode.rates, traps.rates])
    cumulative = np.cumsum(events)
    z = []
    for _ in range(4000):
        event = min(int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right")), events.size - 1)
        n = electrode.rates.size
        point = electrode.sample(event, rng) if event < n else traps.sample(event - n, rng)
        if point is not None:
            z.append(point[2])
    z = np.array(z)
    print("{} {:32s} | cells={:.5e} traps={:.4e} total={:.5e} | cell share z<1 {:.4f} 1-2 {:.4f} >2 {:.2e} | "
          "births z median={:.3f} q90={:.3f} z>2nm={:.4f} (n={}) | wall={:.1f}s".format(
              name, label, total_cells, traps.rates.sum(), total_cells + traps.rates.sum(),
              *(s / total_cells if total_cells > 0 else np.nan for s in shares),
              np.median(z) / nm, np.quantile(z, 0.9) / nm, np.mean(z > 2 * nm), z.size, wall), flush=True)


for nxy in (16, 32, 48):
    report("gen {}x{}x70".format(nxy, nxy), (nxy, nxy, 70))

defaults = {key: getattr(tddb_module, key) for key in
            ("_COLUMN_STEP", "_LEVEL_STEP", "_TABLE_ENERGY_STEP", "_TABLE_TEMPERATURE_STEP")}
for label, keys, factor in (("column step 0.0625 nm", ("_COLUMN_STEP",), 0.0625 / 0.08),
                            ("level step /2", ("_LEVEL_STEP",), 0.5),
                            ("table steps /2", ("_TABLE_ENERGY_STEP", "_TABLE_TEMPERATURE_STEP"), 0.5)):
    for key in keys:
        setattr(tddb_module, key, defaults[key] * factor)
    report(label)
    for key in keys:
        setattr(tddb_module, key, defaults[key])

report("poisson 160^3", solver=PoissonSolver(device, GridConfig(160, 160, 160)))
