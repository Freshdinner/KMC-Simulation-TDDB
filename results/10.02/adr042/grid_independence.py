"""ADR-042 的网格无关性验证：R_1e 陷阱项改为逐源陷阱积分之后，生成率表对生成网格的依赖。

usage: grid_independence.py STATE [N_SAMPLES]

每个状态（common.STATES：t = 0 初始缺陷、ADR-037 的 410 个缺陷终态、ADR-041 的 1010 个缺陷终态）在
横向 16 / 32 / 48 格（z 70 格）的生成网格上各求一次事件表，输出：
1. 新算法：ΣG_单元（电极项）、ΣG_陷阱（源陷阱积分）、陷阱项占比；三种网格下陷阱项是否逐位相同；
2. 旧算法（ADR-039，单元中心求和 N_T·V_cell·P_bb·Σ_i p_i·R7）在同一网格上的陷阱项，用新代码的被积函数
   逐 (单元, 源陷阱) 求值重建（只取 2.5 nm 以内的对，之外每对 < e^{−24}）；
3. 落点：按事件表抽 N_SAMPLES 次生成事件，给出新空位 z 的中位数、90% 分位与 z > 2 nm 的比例（单元事件
   在单元内均匀落点，源陷阱事件按 `TrapCaptureGeneration.sample`）；
4. 逐源陷阱的 Cartesian 加密对照：取速率最大的 6 个与随机 6 个源陷阱，单元中心求和在 0.05 / 0.025 nm
   上相对源陷阱积分的偏差。
"""
import sys
import time

import numpy as np
from scipy import constants

from common import load_state
from kmc_leakage.defects import _global_standoff, minimum_image_delta
from kmc_leakage.tddb import admissible_cells, generation_cells, generation_rates

name = sys.argv[1]
n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
config, inner, p, mf_wall = load_state(name)
device, tddb = config.device, config.tddb
standoff = _global_standoff(device)
positions = inner.defects.positions
print("{}: N={} defects, mean-field {:.1f}s".format(name, inner.n_defects, mf_wall), flush=True)


def old_trap_term(traps, cells):
    """ADR-039 的单元中心求和：Σ_可放置单元 N_T·V_cell·P_bb(c)·Σ_i p_i·R7(i → c)。"""
    live = np.flatnonzero(admissible_cells(cells, positions, device))
    centers = cells.centers[live]
    total = np.zeros(len(cells))
    sources = np.flatnonzero(p > 0.0)
    for start in range(0, live.size, 2048):
        rows = slice(start, start + 2048)
        delta = minimum_image_delta(centers[rows][:, None, :] - positions[None, sources, :], device.Lx, device.Ly)
        distance = np.linalg.norm(delta, axis=-1)
        cell, src = np.nonzero(distance < 2.5e-9)
        values, _, _ = traps._integrand(sources[src], centers[rows][cell], distance[cell, src])
        rate = p[sources[src]] * values / distance[cell, src] ** 2
        total[live[rows]] = np.bincount(cell, rate, minlength=centers[rows].shape[0])
    return tddb.precursor_concentration * cells.volume * total


def sample_z(table, cells, rng):
    rates = np.concatenate([table.cells, table.traps.rates])
    cumulative = np.cumsum(rates)
    z = np.empty(n_samples)
    for m in range(n_samples):
        event = min(int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right")), rates.size - 1)
        if event < len(cells):
            z[m] = cells.lower[event, 2] + cells.size[2] * rng.random()
        else:
            point = table.traps.sample(event - len(cells), rng)
            z[m] = np.nan if point is None else point[2]
    return z[np.isfinite(z)]


reference = None
for nxy in (16, 32, 48):
    cells = generation_cells(device, (nxy, nxy, 70))
    tick = time.time()
    table = generation_rates(inner, p, tddb, cells)
    wall = time.time() - tick
    traps = table.traps
    same = "reference" if reference is None else ("bitwise same" if np.array_equal(traps.rates, reference) else "DIFFERENT")
    if reference is None:
        reference = traps.rates
    tick = time.time()
    old = old_trap_term(traps, cells)
    old_wall = time.time() - tick
    z = sample_z(table, cells, np.random.default_rng(nxy))
    cell_sum, trap_sum = table.cells.sum(), traps.rates.sum()
    print("{} grid {}x{}x70 | new: sum_G cells={:.4e} traps={:.4e} (trap share {:.4f}, traps {}) wall={:.1f}s | "
          "old cell-center trap term={:.4e} ({:.3e} x new) wall={:.1f}s | births z median={:.2f} q90={:.2f} "
          "z>2nm={:.4f} (n={})".format(
              name, nxy, nxy, cell_sum, trap_sum, trap_sum / (cell_sum + trap_sum), same, wall,
              old.sum(), old.sum() / trap_sum if trap_sum > 0 else np.nan, old_wall,
              np.median(z) * 1e9, np.quantile(z, 0.9) * 1e9, np.mean(z > 2e-9), z.size), flush=True)

# 逐源陷阱的 Cartesian 加密对照
rng = np.random.default_rng(7)
order = np.argsort(traps.rates)[::-1]
eligible = np.flatnonzero(traps.rates > 0.0)
chosen = np.unique(np.concatenate([order[:6], rng.choice(eligible, size=min(6, eligible.size), replace=False)]))
reach = 1.6e-9
for i in chosen:
    errors = []
    for h in (0.05e-9, 0.025e-9):
        axis = np.arange(-reach + 0.5 * h, reach, h)
        offsets = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        distance = np.linalg.norm(offsets, axis=-1)
        outside = distance >= standoff
        points, distance = positions[i] + offsets[outside], distance[outside]
        points[:, :2] = np.mod(points[:, :2], [device.Lx, device.Ly])
        keep, _ = traps._admissible(points, np.full(points.shape[0], i))
        values, _, _ = traps._integrand(np.full(int(keep.sum()), i), points[keep], distance[keep])
        cell_sum = tddb.precursor_concentration * p[i] * h ** 3 * np.sum(values / distance[keep] ** 2)
        errors.append(cell_sum / traps.rates[i] - 1.0)
    print("{} source {:4d} z={:.2f} nm p={:.3f} G_i={:.3e} /s ({:.2e} of sum) | Cartesian 0.05 nm {:+.3e}, "
          "0.025 nm {:+.3e}".format(name, i, positions[i, 2] * 1e9, p[i], traps.rates[i],
                                    traps.rates[i] / traps.rates.sum(), *errors), flush=True)
