"""稠密有限差分 Jacobian 牛顿（回溯线搜索），初值可选：empty（p = 0 处的一次 F）| file:PATH.npy（占据）。
usage: newton_from.py SEED N START"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repro import load_state  # noqa
from kmc_leakage import mean_field
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.tddb import IncrementalCoupling
from kmc_leakage.electrostatics import PoissonSolver

seed, n, start_kind = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
config, defects, snap = load_state(seed, n)
solver = PoissonSolver(config.device, config.simulation.grid)
coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
sim = KMCSimulation(config, defects=defects, coupling=coupling, solver=solver)
N = sim.n_defects
def G(x):
    return mean_field._log_occupancy(mean_field.mean_field_flux(sim, np.exp(np.minimum(x, 0.0)))[2]) - x
if start_kind == "empty":
    x = mean_field._log_occupancy(mean_field.mean_field_flux(sim, np.zeros(N))[2])
else:
    x = mean_field._log_occupancy(np.load(start_kind.split(":", 1)[1]))
g = G(x)
t0 = time.time()
for it in range(25):
    print("newton %d max|g| %.3e |g|2 %.3e wall %.0f" % (it, np.abs(g).max(), np.linalg.norm(g), time.time() - t0), flush=True)
    if np.abs(g).max() <= 1e-10:
        break
    J = np.empty((N, N))
    for j in range(N):
        e = np.zeros(N); e[j] = -1e-7
        J[:, j] = (G(x + e) - g) / -1e-7
    dx = -np.linalg.solve(J, g)
    t = 1.0
    while True:
        xt = np.clip(x + t * dx, mean_field._LOG_FLOOR, 0.0)
        gt = G(xt)
        if np.linalg.norm(gt) < (1 - 1e-4 * t) * np.linalg.norm(g) or t < 1e-3:
            break
        t *= 0.5
    print("   t=%.4g max|dx| %.3e smallest |eig(J)| %.3e" % (t, np.abs(dx).max(), np.abs(np.linalg.eigvals(J)).min()), flush=True)
    x, g = xt, gt
np.save(os.path.join(HERE, "x_newton_{}_s{}_n{}.npy".format(start_kind.split(":")[0], seed, n)), x)
