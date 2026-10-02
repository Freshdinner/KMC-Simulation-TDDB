"""陷阱项源陷阱积分对自身求积参数的收敛性（ADR-042）。usage: quadrature_convergence.py STATE

基准取代码默认（Δr 0.04 nm、极角 24、方位角 24、尾部容差 1e-3），逐项加密 / 放宽一倍，
报告 ΣG_陷阱 的相对变化与逐源陷阱相对变化（按速率加权的中位数、最大值）。
"""
import sys
import time

import numpy as np
from scipy import constants

from common import load_state
from kmc_leakage import tddb as T
from kmc_leakage.bands import top_boundary_potential

config, inner, p, _ = load_state(sys.argv[1])
solver, defects, device = inner.solver, inner.defects, inner.device
rho = solver.deposit(defects.positions, constants.elementary_charge * (defects.charge_state - p))
phi = solver.solve(rho, top_boundary_potential(device, inner.sim.voltage))
DEFAULT = dict(_TRAP_RADIAL_STEP=T._TRAP_RADIAL_STEP, _TRAP_POLAR_BINS=T._TRAP_POLAR_BINS,
               _TRAP_AZIMUTH_BINS=T._TRAP_AZIMUTH_BINS, _TRAP_TAIL_TOLERANCE=T._TRAP_TAIL_TOLERANCE)
VARIANTS = [
    ("default", {}),
    ("dr x2 (0.08 nm)", dict(_TRAP_RADIAL_STEP=0.08e-9)),
    ("dr /2 (0.02 nm)", dict(_TRAP_RADIAL_STEP=0.02e-9)),
    ("polar 12", dict(_TRAP_POLAR_BINS=12)),
    ("polar 48", dict(_TRAP_POLAR_BINS=48)),
    ("azimuth 12", dict(_TRAP_AZIMUTH_BINS=12)),
    ("azimuth 48", dict(_TRAP_AZIMUTH_BINS=48)),
    ("tail 1e-2", dict(_TRAP_TAIL_TOLERANCE=1e-2)),
    ("tail 1e-5", dict(_TRAP_TAIL_TOLERANCE=1e-5)),
    ("all fine (0.02 nm, 48, 48, 1e-5)", dict(_TRAP_RADIAL_STEP=0.02e-9, _TRAP_POLAR_BINS=48,
                                               _TRAP_AZIMUTH_BINS=48, _TRAP_TAIL_TOLERANCE=1e-5)),
]
only = sys.argv[2:]  # 可选：只跑列出的变体
reference = None
for name, override in VARIANTS:
    if only and name != "default" and name not in only:
        continue
    for key, value in {**DEFAULT, **override}.items():
        setattr(T, key, value)
    tick = time.time()
    rates = T.TrapCaptureGeneration(inner, p, config.tddb, phi, None).rates
    wall = time.time() - tick
    if reference is None:
        reference = rates
    weight = reference / reference.sum()
    rel = np.abs(rates / np.where(reference > 0, reference, 1.0) - 1.0)[reference > 0]
    order = np.argsort(rel)
    weighted_median = rel[order][np.searchsorted(np.cumsum(weight[reference > 0][order]), 0.5)]
    print("{} | {:34s} sum_G_trap={:.6e} /s  d(sum)={:+.2e}  per-source: weighted median {:.2e}, max {:.2e}  "
          "wall={:.1f}s".format(sys.argv[1], name, rates.sum(), rates.sum() / reference.sum() - 1.0,
                                weighted_median, rel.max(), wall), flush=True)
