"""快照处的一个牛顿步：dx = −(J − I)⁻¹·g（J 为 jacobian.py 存的稠密有限差分 Jacobian），看哪些分量大、沿 dx 残差怎么变。"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repro import load_state  # noqa
from kmc_leakage import mean_field
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.tddb import IncrementalCoupling
from kmc_leakage.electrostatics import PoissonSolver

config, defects, snap = load_state(2, 150)
solver = PoissonSolver(config.device, config.simulation.grid)
coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
sim = KMCSimulation(config, defects=defects, coupling=coupling, solver=solver)
N = sim.n_defects
cluster_of = np.full(N, -1)
for k, m in enumerate(sim.cluster_members):
    cluster_of[m] = k

def g_of(x):
    return mean_field._log_occupancy(mean_field.mean_field_flux(sim, np.exp(np.minimum(x, 0.0)))[2]) - x

x0 = mean_field._log_occupancy(snap["occupancy"])
g0 = g_of(x0)
J = np.load(os.path.join(HERE, "J_s2_n150.npy"))
dx = -np.linalg.solve(J - np.eye(N), g0)
big = np.argsort(-np.abs(dx))[:10]
print("max|g0| %.3e |g0|2 %.3e" % (np.abs(g0).max(), np.linalg.norm(g0)))
for i in big:
    print("dx[%3d] = %+.3e  x0 %.3f  g0 %+.3e  cl %d  z %.2f nm  Z %g" % (i, dx[i], x0[i], g0[i], cluster_of[i], defects.positions[i, 2] * 1e9, defects.charge_state[i]))
for t in (1.0, 0.3, 0.1, 0.03, 0.01, 0.001):
    g = g_of(x0 + t * dx)
    lin = g0 + t * (J - np.eye(N)) @ dx
    print("t=%-6g max|g| %.3e |g|2 %.3e   linear-model |g|2 %.3e  worst trap %d" % (t, np.abs(g).max(), np.linalg.norm(g), np.linalg.norm(lin), int(np.argmax(np.abs(g)))))
# x 被夹在 0 的分量
print("components with x0 > -1e-12:", np.flatnonzero(x0 > -1e-12).tolist(), "dx there", np.round(dx[x0 > -1e-12], 3).tolist())
