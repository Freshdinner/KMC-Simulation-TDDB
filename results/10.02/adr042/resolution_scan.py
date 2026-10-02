"""t = 0 状态（10 个源陷阱，真实占据）上逐个求积参数的影响，参考取最细一级 0.01 nm / 96 / 96。

usage: resolution_scan.py STATE
"""
import sys

import numpy as np
from scipy import constants

from common import load_state
from kmc_leakage import tddb as T
from kmc_leakage.bands import top_boundary_potential

config, inner, p, _ = load_state(sys.argv[1])
s, d = inner.solver, inner.defects
phi = s.solve(s.deposit(d.positions, constants.elementary_charge * (d.charge_state - p)),
              top_boundary_potential(inner.device, inner.sim.voltage))
D = dict(_TRAP_RADIAL_STEP=T._TRAP_RADIAL_STEP, _TRAP_POLAR_BINS=T._TRAP_POLAR_BINS,
         _TRAP_AZIMUTH_BINS=T._TRAP_AZIMUTH_BINS)
variants = [("finest 0.01/96/96", dict(_TRAP_RADIAL_STEP=0.01e-9, _TRAP_POLAR_BINS=96, _TRAP_AZIMUTH_BINS=96)),
            ("0.04/24/24", {}), ("0.02/24/24", dict(_TRAP_RADIAL_STEP=0.02e-9)),
            ("0.04/48/24", dict(_TRAP_POLAR_BINS=48)), ("0.04/24/48", dict(_TRAP_AZIMUTH_BINS=48)),
            ("0.04/24/96", dict(_TRAP_AZIMUTH_BINS=96)), ("0.04/48/48", dict(_TRAP_POLAR_BINS=48, _TRAP_AZIMUTH_BINS=48)),
            ("0.02/48/48", dict(_TRAP_RADIAL_STEP=0.02e-9, _TRAP_POLAR_BINS=48, _TRAP_AZIMUTH_BINS=48)),
            ("0.04/32/64", dict(_TRAP_POLAR_BINS=32, _TRAP_AZIMUTH_BINS=64)),
            ("0.02/32/64", dict(_TRAP_RADIAL_STEP=0.02e-9, _TRAP_POLAR_BINS=32, _TRAP_AZIMUTH_BINS=64)),
            ("0.02/48/96", dict(_TRAP_RADIAL_STEP=0.02e-9, _TRAP_POLAR_BINS=48, _TRAP_AZIMUTH_BINS=96))]
reference = None
print("{} p = {}".format(sys.argv[1], " ".join("{:.2e}".format(x) for x in p)))
for label, override in variants:
    for key, value in {**D, **override}.items():
        setattr(T, key, value)
    rates = T.TrapCaptureGeneration(inner, p, config.tddb, phi, None).rates
    if reference is None:
        reference = rates
        print("{} {:18s} sum={:.6e} share by source: {}".format(sys.argv[1], label, rates.sum(), " ".join(
            "{:.1e}".format(x) for x in rates / rates.sum())))
        continue
    rel = rates / np.where(reference > 0, reference, 1) - 1
    print("{} {:18s} d(sum)={:+.2e} per source: {}".format(sys.argv[1], label, rates.sum() / reference.sum() - 1,
                                                          " ".join("{:+.1e}".format(x) for x in rel)), flush=True)
