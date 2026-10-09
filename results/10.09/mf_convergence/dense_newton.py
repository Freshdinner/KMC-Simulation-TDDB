"""参考解：x = ln p 上对 G(x) = ln F(e^x) − x 做稠密有限差分 Jacobian 的牛顿迭代（回溯线搜索、x ≤ 0），
确认不收敛快照上不动点存在、方程本身良定。保存 x 与不动点处的 Jacobian 谱。"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repro import load_state  # noqa
from kmc_leakage import mean_field
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.tddb import IncrementalCoupling
from kmc_leakage.electrostatics import PoissonSolver

seed, n = int(sys.argv[1]), int(sys.argv[2])
config, defects, snap = load_state(seed, n)
solver = PoissonSolver(config.device, config.simulation.grid)
coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
sim = KMCSimulation(config, defects=defects, coupling=coupling, solver=solver)
N = sim.n_defects

def G(x):
    return mean_field._log_occupancy(mean_field.mean_field_flux(sim, np.exp(x))[2]) - x

def jac(x, g):
    J = np.empty((N, N))
    for j in range(N):
        h = 1e-7 * max(1.0, abs(x[j]))
        e = np.zeros(N); e[j] = -h
        J[:, j] = (G(x + e) - g) / (-h)
    return J

x = mean_field._log_occupancy(snap["occupancy"])
g = G(x)
start = time.time()
for it in range(40):
    res = np.max(np.abs(g))
    print("newton", it, "max|g| %.3e" % res, "|g|2 %.3e" % np.linalg.norm(g), "wall %.0f" % (time.time() - start), flush=True)
    if res <= 1e-10:
        break
    J = jac(x, g)
    dx = -np.linalg.solve(J, g)
    t = 1.0
    while True:
        xt = np.clip(x + t * dx, mean_field._LOG_FLOOR, 0.0)
        gt = G(xt)
        if np.linalg.norm(gt) < (1 - 1e-4 * t) * np.linalg.norm(g) or t < 1e-4:
            break
        t *= 0.5
    print("   step t=%.4g max|dx|=%.3e" % (t, np.max(np.abs(dx))), flush=True)
    x, g = xt, gt
np.save(os.path.join(HERE, "xstar_s{}_n{}.npy".format(seed, n)), x)
J = jac(x, g) + np.eye(N)  # F 的 Jacobian = G 的 + I
w = np.linalg.eigvals(J)
order = np.argsort(-np.abs(w))
print("eig(J_F) at fixed point, largest |λ|:", np.round(w[order[:12]], 4))
print("Re>0.9:", np.round(np.sort(w.real[w.real > 0.9])[::-1], 6))
print("cond(J_G) %.3e" % np.linalg.cond(J - np.eye(N)))
