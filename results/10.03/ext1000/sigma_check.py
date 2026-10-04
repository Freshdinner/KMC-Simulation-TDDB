"""P_bb 平均窗口 σ（ADR-043）的敏感性：同一状态在 σ = 0.35 / 0.5 / 0.7 nm 下的 ΣG 与生成的 z 分布。

usage: sigma_check.py STATE_NPZ SEED Z_GENERATED
- STATE_NPZ：快照 snap_s*_n*.npz 或终态 final_s*.npz（相对仓库根）；
- Z_GENERATED：生成空位的 charge_state（主轨迹 1；O_i⁻ 与空位重合的 what-if 为 0，见 results/10.03/oi_colocated）。

占据取文件里的平均场不动点，温度取文件里的输运温度（终态文件没有记录时取环境温度），等温求事件表（不置零已满事件）。
"""
import os
import sys
from dataclasses import replace

import numpy as np
from scipy import constants

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage.defects import DefectEnsemble  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import KMCSimulation  # noqa: E402
from kmc_leakage.tddb import generation_cells, generation_rates  # noqa: E402

nm = constants.nano
path, seed, z_generated = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
base = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
device, tddb = base.device, base.tddb
state = np.load(os.path.join(ROOT, path))
n_initial = int(state["n_initial"])
temperature = float(state["transport_temperature"]) if "transport_temperature" in state.files else \
    base.simulation.temperature
charge_state = np.full(state["positions"].shape[0], float(device.stack.layers[0].defects.charge_state))
charge_state[n_initial:] = z_generated
defects = replace(DefectEnsemble.uniform(state["positions"], state["ED"], device.stack.single_material,
                                         band_reference=float(device.layer_band_references()[0])),
                  charge_state=charge_state)
config = replace(base, simulation=replace(base.simulation, seed=seed, temperature=temperature))
inner = KMCSimulation(config, defects=defects)
p = state["occupancy"]
cells = generation_cells(device, tddb.generation_grid)
print("{} | N = {} ({} generated, Z_generated = {:g}) | net charge {:+.1f} e".format(
    path, len(defects), len(defects) - n_initial, z_generated, float(np.sum(charge_state - p))), flush=True)
for width in (0.35, 0.5, 0.7):
    table = generation_rates(inner, p, replace(tddb, field_averaging_width=width * nm), cells)
    events = table.cells
    layers = events._levels.size - 1
    middle = events._levels[:-1] + 0.5 * events._dz
    by_layer = events._weights.reshape(layers, -1).sum(axis=1)
    g_cells, g_traps = float(events.rates.sum()), float(table.traps.rates.sum())
    total = g_cells + g_traps
    shares = [by_layer[middle < 1 * nm].sum(), by_layer[(middle >= 1 * nm) & (middle < 2 * nm)].sum(),
              by_layer[middle >= 2 * nm].sum()]
    print("  sigma {:.2f} nm: ΣG {:.4e} (cells {:.4e}, traps {:.4e}, trap share {:.1%}) | electrode term by z "
          "<1 / 1-2 / >2 nm: {:.3f} / {:.3f} / {:.3f}".format(
              width, total, g_cells, g_traps, g_traps / total, *(s / sum(shares) for s in shares)), flush=True)
