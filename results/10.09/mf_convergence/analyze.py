"""逐次残差与振荡陷阱的诊断（repro.py 的同一个状态）。"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repro import load_state, solve_logged, ROOT  # noqa
from kmc_leakage import mean_field
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.tddb import IncrementalCoupling
from kmc_leakage.electrostatics import PoissonSolver

seed, n = int(sys.argv[1]), int(sys.argv[2])
config, defects, snap = load_state(seed, n)
solver = PoissonSolver(config.device, config.simulation.grid)
coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
sim = KMCSimulation(config, defects=defects, coupling=coupling, solver=solver)
print("N", sim.n_defects, "clusters", [m.size for m in sim.cluster_members])
p, hist = solve_logged(sim, snap["occupancy"], max_iter=60, log=False)
for k, (r, w) in enumerate(hist):
    print(k + 1, "%.3e" % r, w, "z=%.2f" % (defects.positions[w, 2] * 1e9), "Z=%g" % defects.charge_state[w])
# 简单迭代 p -> p_hat 的局部 Jacobian 谱半径（ln p 空间），在快照 p 上用有限差分估计主特征值（幂迭代）
x0 = mean_field._log_occupancy(snap["occupancy"])
def F(x):
    return mean_field._log_occupancy(mean_field.mean_field_flux(sim, np.exp(x))[2])
f0 = F(x0)
v = np.random.default_rng(0).normal(size=x0.size); v /= np.linalg.norm(v)
eps = 1e-6
for k in range(15):
    w = (F(np.minimum(x0 + eps * v, 0.0)) - f0) / eps
    lam = float(v @ w)
    nv = np.linalg.norm(w)
    print("power it", k, "rayleigh", lam, "|Jv|", nv)
    v = w / nv
top = np.argsort(-np.abs(v))[:12]
print("dominant mode traps", top, "z", np.round(defects.positions[top, 2] * 1e9, 2), "Z", defects.charge_state[top], "p", snap["occupancy"][top])
