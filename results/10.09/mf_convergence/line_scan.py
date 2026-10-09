"""F 是否连续：沿快照处的一条方向（Anderson 第一步给的方向）细扫 g(x0 + t·d)，找相邻两点之间跳变最大的分量。"""
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
cluster_of = np.full(sim.n_defects, -1)
for k, m in enumerate(sim.cluster_members):
    cluster_of[m] = k

def g_of(x):
    return mean_field._log_occupancy(mean_field.mean_field_flux(sim, np.exp(x))[2]) - x

x0 = mean_field._log_occupancy(snap["occupancy"])
g0 = g_of(x0)
d = np.minimum(x0 + g0, 0.0) - x0   # Picard 方向
ts = np.linspace(0.0, 1.0, 201)
G = np.array([g_of(np.minimum(x0 + t * d, 0.0)) for t in ts])
jump = np.abs(np.diff(G, axis=0))            # (200, N)
second = np.abs(np.diff(G, n=2, axis=0))     # 光滑时 ~ O(dt²)
k, i = np.unravel_index(np.argmax(second), second.shape)
print("max |Δg| between neighbours %.3e at t=%.3f trap %d (cluster %d)" % (jump.max(), ts[np.unravel_index(np.argmax(jump), jump.shape)[0]], np.unravel_index(np.argmax(jump), jump.shape)[1], cluster_of[np.unravel_index(np.argmax(jump), jump.shape)[1]]))
print("max |Δ²g| %.3e at t=%.3f trap %d (cluster %d) p=%.3g" % (second.max(), ts[k + 1], i, cluster_of[i], np.exp(x0[i])))
worst = np.argsort(-second.max(axis=0))[:10]
for i in worst:
    kk = int(np.argmax(second[:, i]))
    print("trap %3d cl %2d max|Δ²g| %.3e at t=%.3f  g around: %s" % (i, cluster_of[i], second[kk, i], ts[kk + 1], np.round(G[max(kk - 2, 0):kk + 4, i], 4)))
np.save(os.path.join(HERE, "line_scan_G.npy"), G)
