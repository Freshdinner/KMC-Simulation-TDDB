"""带电（空着、p 小）源陷阱上的精度：源陷阱积分逐级加密 vs Cartesian 单元中心求和逐级加密。

usage: charged_source_check.py STATE SOURCE [SOURCE ...]（t = 0 这类小体系：每级都积全部源陷阱）
带电源陷阱自身的 +e 库仑场让近旁 P_bb 与前驱能级在 0.25 nm 的泊松网格尺度上变化，看两种求积是否收敛到
同一个值。
"""
import sys

import numpy as np
from scipy import constants

from common import load_state
from kmc_leakage import tddb as T
from kmc_leakage.bands import top_boundary_potential
from kmc_leakage.defects import _global_standoff

name, sources = sys.argv[1], [int(x) for x in sys.argv[2:]]
config, inner, p, _ = load_state(name)
device, tddb = config.device, config.tddb
solver, defects = inner.solver, inner.defects
phi = solver.solve(solver.deposit(defects.positions, constants.elementary_charge * (defects.charge_state - p)),
                   top_boundary_potential(device, inner.sim.voltage))
standoff = _global_standoff(device)
DEFAULT = dict(_TRAP_RADIAL_STEP=T._TRAP_RADIAL_STEP, _TRAP_POLAR_BINS=T._TRAP_POLAR_BINS,
               _TRAP_AZIMUTH_BINS=T._TRAP_AZIMUTH_BINS)
levels = [("default 0.04nm/24/24", {}),
          ("0.02nm/48/48", dict(_TRAP_RADIAL_STEP=0.02e-9, _TRAP_POLAR_BINS=48, _TRAP_AZIMUTH_BINS=48)),
          ("0.01nm/96/96", dict(_TRAP_RADIAL_STEP=0.01e-9, _TRAP_POLAR_BINS=96, _TRAP_AZIMUTH_BINS=96))]
results = {}
for label, override in levels:
    for key, value in {**DEFAULT, **override}.items():
        setattr(T, key, value)
    obj = T.TrapCaptureGeneration(inner, p, tddb, phi, None)  # 真实占据：源陷阱的平均场能级依赖全体占据
    results[label] = obj.rates[sources]
    print("{} | quadrature {:22s}: {}".format(name, label, " ".join("{:.6e}".format(x) for x in obj.rates[sources])),
          flush=True)
for key, value in DEFAULT.items():
    setattr(T, key, value)
base = results["0.01nm/96/96"]
for h in (0.05e-9, 0.025e-9, 0.0125e-9):
    reach = 1.2e-9
    axis = np.arange(-reach + 0.5 * h, reach, h)
    offsets = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    distance0 = np.linalg.norm(offsets, axis=-1)
    outside = distance0 >= standoff
    sums = []
    for i in sources:
        points, distance = defects.positions[i] + offsets[outside], distance0[outside]
        points[:, :2] = np.mod(points[:, :2], [device.Lx, device.Ly])
        keep, _ = obj._admissible(points, np.full(points.shape[0], i))
        values, _, _ = obj._integrand(np.full(int(keep.sum()), i), points[keep], distance[keep])
        sums.append(tddb.precursor_concentration * p[i] * h ** 3 * np.sum(values / distance[keep] ** 2))
    print("{} | Cartesian h={:.4f} nm: {}".format(name, h * 1e9, " ".join("{:.6e}".format(x) for x in sums)), flush=True)
print("{} | default / finest quadrature − 1: {}".format(
    name, " ".join("{:+.2e}".format(x) for x in results["default 0.04nm/24/24"] / base - 1.0)))
