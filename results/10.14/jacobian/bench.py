"""分组有限差分 Jacobian（ADR-058）的实测对比：同一个续跑状态上，分组与逐列（每组一个缺陷，即 ADR-058 之前）各解一次平均场。

usage: bench.py STATE_NPZ R_C_NM

内层与 probe.py 相同（输运温度取状态里的值），都从状态里的占据热启动。输出两种做法的耗时、求 F 次数、牛顿步数、残差，以及
电流与占据的相对差。
"""
import os
import sys
import time
from dataclasses import replace

import numpy as np
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage import mean_field as M  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import KMCSimulation  # noqa: E402
from kmc_leakage.tddb import load_state  # noqa: E402

path, r_c_nm = sys.argv[1], float(sys.argv[2])
state = load_state(path)
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
config = replace(config, simulation=replace(config.simulation, temperature=state.transport_temperature),
                 tddb=replace(config.tddb, percolation_distance=r_c_nm * constants.nano))
sim = KMCSimulation(config, defects=state.defects)
print("{}: {} defects, {} Jacobian groups".format(os.path.basename(path), sim.n_defects, len(M._jacobian_groups(sim))),
      flush=True)
grouped_rule = M._jacobian_groups
results = {}
for name, rule in (("grouped", grouped_rule), ("column", lambda sim: [np.array([j]) for j in range(sim.n_defects)])):
    M._jacobian_groups = rule
    began = time.time()
    results[name] = M.solve_mean_field(sim, state.occupancy)
    r = results[name]
    print("{:8s}: {:.0f} s, evaluations {}, newton steps {}, residual {:.1e}, converged {}, J {:.10g} A/cm2".format(
        name, time.time() - began, r.iterations, r.newton_steps, r.residual, r.converged, r.current_density * 1e-4),
        flush=True)
M._jacobian_groups = grouped_rule
a, b = results["grouped"], results["column"]
print("relative difference: current {:.1e}, occupancy max {:.1e}".format(
    abs(a.current / b.current - 1.0), np.max(np.abs(a.occupancy - b.occupancy) / np.maximum(b.occupancy, 1e-300))))
