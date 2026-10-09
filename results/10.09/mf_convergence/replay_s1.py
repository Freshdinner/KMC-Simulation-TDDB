"""按新代码（O_i⁻ 与空位重合，q_fix = −e）重放 ADR-046 what-if 的 seed 1 轨迹（3.0 V、zro2_tddb.yaml 其余不变）到第 30 个空位，
核对出生位置与 results/10.03/oi_colocated/steps_s1.csv 相同，并存下第 29、30 个空位出生后交给平均场的状态
（what-if 里这两条记录 Anderson 200 次不收敛），给 tests/data/ 用。
"""
import csv
import os
import sys
from dataclasses import replace

import numpy as np
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage import tddb as tddb_module  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402

config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
config = replace(
    config,
    simulation=replace(config.simulation, voltage=3.0, seed=1),
    tddb=replace(config.tddb, max_generated_vacancies=30, breakdown_current_density=1.0e12),
)
calls = []
original = tddb_module.solve_mean_field


def recording(sim, occupancy=None):
    result = original(sim, occupancy)
    calls.append((sim, None if occupancy is None else occupancy.copy(), result))
    n = sim.n_defects - 10
    print("n={} N={} converged={} it={} newton={} res={:.2e} J={:.4e} A/cm2".format(
        n, sim.n_defects, result.converged, result.iterations, result.newton_steps, result.residual,
        result.current_density * 1e-4), flush=True)
    return result


tddb_module.solve_mean_field = recording
result = tddb_module.run_tddb(config)
rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "10.03", "oi_colocated", "steps_s1.csv"))))
whatif = np.array([[float(r["birth_x_nm"]), float(r["birth_y_nm"]), float(r["birth_z_nm"])] for r in rows[1:31]])
mine = result.defects.positions[result.n_initial:] / constants.nano
print("max |Δ birth position| vs what-if: {:.3e} nm".format(float(np.max(np.abs(mine - whatif)))))
for n in (29, 30):
    sim, start, _ = calls[n]
    np.savez(os.path.join(HERE, "state_s1_n{}.npz".format(n)), positions=sim.defects.positions, ED=sim.defects.ED,
             fixed_charge=sim.defects.fixed_charge, occupancy=start, temperature=sim.sim.temperature,
             n_initial=result.n_initial)
