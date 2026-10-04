"""ADR-046 热反馈说明（由 ../rerun/heat_breakdown.py 复制，读本目录的快照）：在轨迹快照上把 T13 的放热 ΔE·R 按去处拆开（阴极、阳极、介质里各缺陷按 z 分段）。

usage: heat_breakdown.py SEED N [N ...]

状态与 check_states.py 相同（快照的不动点占据与输运温度）。U·I_陷阱 = 阴极 + 阳极 + 介质放热（能量守恒，ADR-038）；
介质部分再按缺陷所在 z 分段，并分开放热（> 0）与吸热（< 0）的缺陷。
"""
import os
import sys
from dataclasses import replace

import numpy as np
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage.defects import DefectEnsemble  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import KMCSimulation  # noqa: E402
from kmc_leakage.mean_field import solve_mean_field  # noqa: E402
from kmc_leakage.thermal import heat_release  # noqa: E402

nm = constants.nano
seed = int(sys.argv[1])
base = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
base = replace(base, simulation=replace(base.simulation, seed=seed))
device = base.device
BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 8.0))

for n in (int(v) for v in sys.argv[2:]):
    snap = np.load(os.path.join(HERE, "snap_s{}_n{:04d}.npz".format(seed, n)))
    defects = DefectEnsemble.uniform(snap["positions"], snap["ED"], device.stack.single_material,
                                     band_reference=float(device.layer_band_references()[0]),
                                     charge_state=device.stack.layers[0].defects.charge_state)
    config = replace(base, simulation=replace(base.simulation, temperature=float(snap["transport_temperature"])))
    inner = KMCSimulation(config, defects=defects)
    mf = solve_mean_field(inner, snap["occupancy"])
    heat = heat_release(inner, mf.occupancy)
    i_trap = mf.current - mf.by_mechanism["direct_fn"] - mf.by_mechanism["thermionic"]
    ui = config.simulation.voltage * i_trap
    z = defects.positions[:, 2] / nm
    print("s{} n={} N={} | U*I_trap {:.4e} W | cathode {:+.4f} anode {:+.4f} dielectric {:+.4f} (of U*I) | "
          "balance {:.1e}".format(seed, n, len(defects), ui, heat.cathode / ui, heat.anode / ui, heat.dielectric / ui,
                                  (heat.cathode + heat.anode + heat.dielectric) / ui - 1.0), flush=True)
    for lo, hi in BINS:
        inside = (z >= lo) & (z < hi)
        h = heat.defect[inside]
        print("  defects at z in [{:.0f}, {:.0f}) nm: {:4d} | net {:+.4f} | releasing {:+.4f} ({}) | "
              "absorbing {:+.4f} ({}) (of U*I)".format(lo, hi, int(inside.sum()), h.sum() / ui, h[h > 0].sum() / ui,
                                                       int(np.sum(h > 0)), h[h < 0].sum() / ui, int(np.sum(h < 0))),
              flush=True)
