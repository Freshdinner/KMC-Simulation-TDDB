"""原型：有限差分 Jacobian + Broyden 秩一更新的牛顿迭代（ln p 变量，x ≤ 0，回溯线搜索），统计 F 求值次数。

起点依次试：START 指定的占据（snap = 快照里的未收敛迭代值），停住了再从 p̂(0)（全空占据的一次 F）重来。
usage: broyden.py SEED N [snap|empty]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repro import load_state  # noqa: E402
from kmc_leakage import mean_field  # noqa: E402
from kmc_leakage.kmc import KMCSimulation  # noqa: E402
from kmc_leakage.tddb import IncrementalCoupling  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402

TOL = 1e-10
H = 1e-7

seed, n = int(sys.argv[1]), int(sys.argv[2])
first = sys.argv[3] if len(sys.argv) > 3 else "snap"
config, defects, snap = load_state(seed, n)
solver = PoissonSolver(config.device, config.simulation.grid)
coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
sim = KMCSimulation(config, defects=defects, coupling=coupling, solver=solver)
N = sim.n_defects
evals = [0]


def G(x):
    evals[0] += 1
    return mean_field._log_occupancy(mean_field.mean_field_flux(sim, np.exp(x))[2]) - x


def fd_jacobian(x, g):
    J = np.empty((N, N))
    for j in range(N):
        e = np.zeros(N)
        e[j] = -H
        J[:, j] = (G(x + e) - g) / -H
    return J


def newton(x, max_steps=60):
    g = G(x)
    J = fd_jacobian(x, g)
    fresh = True
    for step in range(max_steps):
        res = float(np.max(np.abs(g)))
        print("  step %2d max|g| %.3e evals %d fresh %d" % (step, res, evals[0], fresh), flush=True)
        if res <= TOL:
            return x, True
        dx = np.linalg.solve(J, -g)
        norm = np.linalg.norm(g)
        t, accepted = 1.0, False
        while t >= 2.0 ** -10:
            xt = np.clip(x + t * dx, mean_field._LOG_FLOOR, 0.0)
            gt = G(xt)
            if np.linalg.norm(gt) <= (1.0 - 1e-4 * t) * norm:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            if fresh:
                print("  stalled (line search failed with a fresh Jacobian)", flush=True)
                return x, False
            J = fd_jacobian(x, g)
            fresh = True
            continue
        s, y = xt - x, gt - g
        J += np.outer(y - J @ s, s) / (s @ s)
        x, g, fresh = xt, gt, False
    return x, False


start = time.time()
starts = []
if first == "snap":
    starts.append(("snap", mean_field._log_occupancy(snap["occupancy"])))
starts.append(("empty", mean_field._log_occupancy(mean_field.mean_field_flux(sim, np.zeros(N))[2])))
for name, x0 in starts:
    print("start", name, flush=True)
    x, ok = newton(x0)
    print("start {} converged {} evals {} wall {:.0f}".format(name, ok, evals[0], time.time() - start), flush=True)
    if ok:
        np.save(os.path.join(HERE, "x_broyden_s{}_n{}.npy".format(seed, n)), x)
        break
