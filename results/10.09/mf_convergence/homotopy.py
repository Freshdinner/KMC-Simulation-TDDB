"""O_i⁻ 电荷的延拓：生成空位 q_fix = −λ·e，λ 从 0 走到 1，每步从上一步的不动点热启动求平均场（当前 src 的
solve_mean_field：Anderson）。看在哪个 λ 开始不收敛、占据怎么随 λ 变。

usage: homotopy.py SEED N STEPS
"""
import os, sys, time
from dataclasses import replace
import numpy as np
from scipy import constants
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repro import load_state  # noqa
from kmc_leakage import mean_field
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.tddb import IncrementalCoupling
from kmc_leakage.electrostatics import PoissonSolver

seed, n, steps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
config, defects, snap = load_state(seed, n)
n_initial = int(snap["n_initial"])
defects = replace(defects, charge_state=np.ones(len(defects)))
solver = PoissonSolver(config.device, config.simulation.grid)
coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
generated = np.arange(len(defects)) >= n_initial
p = None
start = time.time()
for lam in np.linspace(0.0, 1.0, steps + 1):
    fixed = np.where(generated, -lam * constants.elementary_charge, 0.0)
    sim = KMCSimulation(config, defects=replace(defects, fixed_charge=fixed), coupling=coupling, solver=solver)
    r = mean_field.solve_mean_field(sim, p)
    print("lambda %.3f converged %d it %3d res %.2e J %.4e A/cm2 sum p_gen %.3f clusters %d wall %.0f" % (
        lam, r.converged, r.iterations, r.residual, r.current_density * 1e-4, r.occupancy[generated].sum(),
        len(sim.cluster_members), time.time() - start), flush=True)
    p = r.occupancy
    np.save(os.path.join(HERE, "homotopy_s{}_n{}_lam{:.3f}.npy".format(seed, n, lam)), p)
