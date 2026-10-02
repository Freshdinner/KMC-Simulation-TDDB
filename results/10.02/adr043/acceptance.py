"""ADR-043 验收：P_bb 改用 Gaussian 窗口平均的宏观场（σ = 0.5 nm）之后，生成事件表的网格依赖与物理影响。

usage: acceptance.py STATE

状态同 ADR-042（`../adr042/common.py`：t = 0 初始缺陷、ADR-037 的 410 个缺陷终态、ADR-041 的 1010 个缺陷终态），
等温，全部走生产代码 `tddb.generation_rates`。输出：
1. 生成网格：横向 16 / 32 / 48 格（验收）与 64 / 96 / 128 格、z 70 格，再加 128 格、z 140 格；
2. 泊松网格：生成网格 32 格下把泊松网格从 80³（0.25 / 0.1 nm）换成 160³（0.125 / 0.05 nm），σ = 0.5 nm 与
   σ = 0（局域场）各一次。内层平均场占据与陷阱能级仍取 80³ 上的解，只换生成率求值用的电势与场；
3. σ 的敏感性：生成网格 64 格下 σ = 0.35 / 0.7 nm；
每行给 ΣG_单元、ΣG_陷阱、合计，单元事件沿 z 的份额，按事件表抽 4000 次的新空位 z 中位数、90% 分位与 z > 2 nm 的比例。
"""
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adr042"))

import numpy as np
from scipy import constants

from common import load_state
from kmc_leakage.config import GridConfig
from kmc_leakage.electrostatics import PoissonSolver
from kmc_leakage.tddb import generation_cells, generation_rates

nm = constants.nano
name = sys.argv[1]
config, inner, p, mf_wall = load_state(name)
device, tddb = config.device, config.tddb
print("{}: N={} defects, empty (p<0.5) {}, mean-field {:.1f}s, sigma={:.2f} nm".format(
    name, inner.n_defects, int(np.sum(p < 0.5)), mf_wall, tddb.field_averaging_width / nm), flush=True)


def report(label, cells, width, solver=None):
    original = inner.solver
    if solver is not None:
        inner.solver = solver
    tick = time.time()
    try:
        table = generation_rates(inner, p, replace(tddb, field_averaging_width=width), cells)
    finally:
        inner.solver = original
    wall = time.time() - tick
    rates, traps = table.cells, table.traps
    zc = cells.centers[:, 2]
    total_cells = rates.sum()
    shares = [rates[zc < 1 * nm].sum(), rates[(zc >= 1 * nm) & (zc < 2 * nm)].sum(), rates[zc >= 2 * nm].sum()]
    rng = np.random.default_rng(len(cells))
    events = np.concatenate([rates, traps.rates])
    cumulative = np.cumsum(events)
    z = []
    for _ in range(4000):
        event = min(int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right")), events.size - 1)
        if event < len(cells):
            z.append(cells.lower[event, 2] + cells.size[2] * rng.random())
        else:
            point = traps.sample(event - len(cells), rng)
            if point is not None:
                z.append(point[2])
    z = np.array(z)
    print("{} {:28s} sigma={:.2f} | cells={:.4e} traps={:.4e} total={:.4e} | cell share z<1 {:.3f} 1-2 {:.3f} >2 {:.2e} | "
          "births z median={:.3f} q90={:.3f} z>2nm={:.4f} | wall={:.0f}s".format(
              name, label, width / nm, total_cells, traps.rates.sum(), total_cells + traps.rates.sum(),
              *(s / total_cells if total_cells > 0 else np.nan for s in shares),
              np.median(z) / nm, np.quantile(z, 0.9) / nm, np.mean(z > 2 * nm), wall), flush=True)


sigma = tddb.field_averaging_width
for nxy, nz in ((16, 70), (32, 70), (48, 70), (64, 70), (96, 70), (128, 70), (128, 140)):
    report("gen {}x{}x{}".format(nxy, nxy, nz), generation_cells(device, (nxy, nxy, nz)), sigma)

cells32 = generation_cells(device, (32, 32, 70))
fine_poisson = PoissonSolver(device, GridConfig(160, 160, 160))
for width in (sigma, 0.0):
    report("gen 32 poisson 80^3", cells32, width)
    report("gen 32 poisson 160^3", cells32, width, fine_poisson)

cells64 = generation_cells(device, (64, 64, 70))
for width in (0.35 * nm, 0.7 * nm):
    report("gen 64x64x70", cells64, width)
