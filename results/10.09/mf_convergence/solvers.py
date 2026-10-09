"""在不收敛快照上对比几种不动点求解方式（ln p 变量，残差 g = ln p̂(p) − ln p，判据 max|g| ≤ 1e-10）。

usage: solvers.py SEED N METHOD
METHOD: anderson-m{M}（历史 M 步）| anderson-m{M}-b{beta}（带阻尼）| nk（scipy newton_krylov）
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repro import load_state  # noqa
from kmc_leakage import mean_field
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.tddb import IncrementalCoupling
from kmc_leakage.electrostatics import PoissonSolver

seed, n, method = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
config, defects, snap = load_state(seed, n)
solver = PoissonSolver(config.device, config.simulation.grid)
coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
sim = KMCSimulation(config, defects=defects, coupling=coupling, solver=solver)
count = [0]


def F(x):
    count[0] += 1
    return mean_field._log_occupancy(mean_field.mean_field_flux(sim, np.exp(np.minimum(x, 0.0)))[2])


x = mean_field._log_occupancy(snap["occupancy"])
start = time.time()
if method.startswith("anderson"):
    parts = method.split("-")
    m = int(parts[1][1:])
    beta = float(parts[2][1:]) if len(parts) > 2 else 1.0
    xs, gs = [], []
    for it in range(1, 401):
        g = F(x) - x
        res = float(np.max(np.abs(g)))
        print(it, "%.3e" % res, flush=True)
        if res <= 1e-10:
            break
        xs.append(x); gs.append(g)
        if len(xs) > m + 1:
            xs.pop(0); gs.pop(0)
        step = x + beta * g
        if len(xs) > 1:
            dx = np.diff(np.array(xs), axis=0).T
            dg = np.diff(np.array(gs), axis=0).T
            gamma = np.linalg.lstsq(dg, g, rcond=None)[0]
            step = step - (dx + beta * dg) @ gamma
        x = np.clip(step, mean_field._LOG_FLOOR, 0.0)
elif method == "nk":
    from scipy.optimize import newton_krylov
    def G(x):
        g = F(x) - x
        print(count[0], "%.3e" % np.max(np.abs(g)), flush=True)
        return g
    try:
        x = newton_krylov(G, x, f_tol=1e-10, maxiter=100, verbose=False)
    except Exception as exc:
        print("nk failed", exc)
print("method", method, "evals", count[0], "wall %.0f" % (time.time() - start))
np.save(os.path.join(HERE, "x_s{}_n{}_{}.npy".format(seed, n, method)), x)
