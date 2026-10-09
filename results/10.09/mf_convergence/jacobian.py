"""不收敛快照上不动点映射 G(x) = F(x) − x（x = ln p）的有限差分 Jacobian：特征值、主模态涉及的陷阱及其分簇归属。"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repro import load_state  # noqa
from kmc_leakage import mean_field
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.tddb import IncrementalCoupling, _cluster_diagnostics
from kmc_leakage.electrostatics import PoissonSolver
from kmc_leakage.clusters import coupling_strength

seed, n = int(sys.argv[1]), int(sys.argv[2])
xfile = sys.argv[3] if len(sys.argv) > 3 else None
config, defects, snap = load_state(seed, n)
solver = PoissonSolver(config.device, config.simulation.grid)
coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
sim = KMCSimulation(config, defects=defects, coupling=coupling, solver=solver)
N = sim.n_defects
x0 = np.load(xfile) if xfile else mean_field._log_occupancy(snap["occupancy"])
p0 = np.exp(x0)

def F(x):
    return mean_field._log_occupancy(mean_field.mean_field_flux(sim, np.exp(np.minimum(x, 0.0)))[2])

f0 = F(x0)
J = np.empty((N, N))
h = 1e-6
for j in range(N):
    e = np.zeros(N); e[j] = -h  # 向下扰动，保持 x ≤ 0
    J[:, j] = (F(x0 + e) - f0) / (-h)
np.save(os.path.join(HERE, "J_s{}_n{}.npy".format(seed, n)), J)
w, V = np.linalg.eig(J)
order = np.argsort(-np.abs(w))
print("N", N, "clusters", [m.size for m in sim.cluster_members], "diag", _cluster_diagnostics(sim))
cluster_of = np.full(N, -1)
for k, m in enumerate(sim.cluster_members):
    cluster_of[m] = k
blocks, hop = sim._rates(np.zeros(N, dtype=bool))
strength = coupling_strength(sim._pairs.a, sim._pairs.b, hop, blocks.sum(axis=0))
strong = strength >= 1e-3
a, b = sim._pairs.a[strong], sim._pairs.b[strong]
cut = cluster_of[a] != cluster_of[b]
cut_traps = np.unique(np.concatenate([a[cut], b[cut]]))
print("strong pairs", strong.sum(), "cut strong pairs (across clusters / mean-field)", cut.sum(), "max strength on cut", strength[strong][cut].max() if cut.any() else 0)
for k in order[:10]:
    v = V[:, k]
    top = np.argsort(-np.abs(v))[:8]
    print("eig %.4f%+.4fj" % (w[k].real, w[k].imag), "traps", top.tolist(), "|v|", np.round(np.abs(v[top]), 2).tolist(),
          "cluster", cluster_of[top].tolist(), "on cut", np.isin(top, cut_traps).astype(int).tolist(), "p", np.round(p0[top], 3).tolist())
print("eigs of J with Re>0.9:", np.sort(w.real[w.real > 0.9])[::-1][:20])
print("eigs with Re < -1:", np.sort(w.real[w.real < -1])[:20])
