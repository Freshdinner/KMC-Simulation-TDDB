"""陷阱→前驱俘获通道（ADR-039）的生成网格敏感度：横向 16 格与 32 格（z 70 格不变）。

usage: generation_grid_check.py SEED [initial]
默认取 Phase 4 终态（../phase4/p4_thermal_s{SEED}.npz，410 个缺陷），加 initial 取 t = 0 的初始缺陷。
输出每种网格一行：ΣG、陷阱项占 ΣG 的比例（与截断半径置零、只剩电极项的表对比）、生成落在 z > 2 nm 的比例、
按 G 加权的 z 中位数、单次求表耗时。
"""
import sys
import time
from dataclasses import replace

import numpy as np

from kmc_leakage import tddb as tddb_module
from kmc_leakage.defects import DefectEnsemble, generate_defects
from kmc_leakage.io import load_config
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.mean_field import solve_mean_field
from kmc_leakage.tddb import generation_cells, generation_rates

seed = int(sys.argv[1])
initial = len(sys.argv) > 2 and sys.argv[2] == "initial"
config = load_config("examples/zro2_tddb.yaml")
config = replace(config, simulation=replace(config.simulation, seed=seed))
material = config.device.stack.single_material
if initial:
    defects = generate_defects(config.device, np.random.default_rng(seed))
    start = None
else:
    data = np.load("results/9.30/phase4/p4_thermal_s{}.npz".format(seed))
    defects = DefectEnsemble.uniform(data["positions"], data["ED"], material,
                                     band_reference=float(config.device.layer_band_references()[0]))
    start = data["occupancy"]
inner = KMCSimulation(config, defects=defects)
occupancy = solve_mean_field(inner, start).occupancy
cutoff = tddb_module._trap_capture_cutoff


def weighted_median(z, w):
    order = np.argsort(z)
    c = np.cumsum(w[order])
    return z[order][np.searchsorted(c, 0.5 * c[-1])]


for nxy in (16, 32):
    cells = generation_cells(config.device, (nxy, nxy, 70))
    tick = time.time()
    tddb_module._trap_capture_cutoff = cutoff
    rates = generation_rates(inner, occupancy, config.tddb, cells)
    wall = time.time() - tick
    tddb_module._trap_capture_cutoff = lambda *args: 0.0
    electrode = generation_rates(inner, occupancy, config.tddb, cells)
    z = cells.centers[:, 2] * 1e9
    total = rates.sum()
    print("seed={} {} N={} grid {}x{}x70: sum_G={:.4e} /s (electrode only {:.4e}) trap_share={:.4f} "
          "G(z>2nm)/sum={:.4e} G(z>4nm)/sum={:.4e} z_median={:.2f} nm (electrode only {:.2f}) wall={:.1f}s".format(
              seed, "t0" if initial else "final", len(defects), nxy, nxy, total, electrode.sum(),
              1.0 - electrode.sum() / total, rates[z > 2].sum() / total, rates[z > 4].sum() / total,
              weighted_median(z, rates), weighted_median(z, electrode), wall), flush=True)
