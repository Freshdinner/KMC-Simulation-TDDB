"""ADR-044：致密状态下电极项 ΣG 随列间距的变化（生成网格 16 格，z 节点 0.1 nm），用来定默认列间距。

usage: column_trend.py STATE
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adr042"))

from scipy import constants

from common import load_state
from kmc_leakage import tddb as tddb_module
from kmc_leakage.tddb import ElectrodeCaptureGeneration, generation_cells, top_boundary_potential

nm = constants.nano
name = sys.argv[1]
config, inner, p, _ = load_state(name)
solver, defects = inner.solver, inner.defects
rho = solver.deposit(defects.positions, constants.elementary_charge * (defects.charge_state - p))
phi = solver.solve(rho, top_boundary_potential(inner.device, inner.sim.voltage))
import numpy as np
bond = solver.gaussian_average(np.stack(solver.electric_field(phi), axis=-1), config.tddb.field_averaging_width)
cells = generation_cells(config.device, (16, 16, 70))
for step in (0.25, 0.2, 0.16, 0.125, 0.1, 0.08, 0.0625):
    tddb_module._COLUMN_STEP = step * nm
    tick = time.time()
    total = ElectrodeCaptureGeneration(inner, config.tddb, cells, phi, bond, None).rates.sum()
    print("{} column step {:.4f} nm: sum_G_cells={:.5e} wall={:.1f}s".format(name, step, total, time.time() - tick), flush=True)
