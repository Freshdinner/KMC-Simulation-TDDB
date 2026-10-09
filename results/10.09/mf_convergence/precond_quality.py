"""预条件质量：快照处稠密有限差分 Jacobian J（jacobian.py 存的 J_s2_n150.npy）与冻结速率跃迁 Jacobian J_T 的比较。
看 (I − J_T)⁻¹(I − J) 的特征值，以及 J − J_T 里库仑（远程）与近邻两部分的大小。"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repro import load_state  # noqa
from precond import transport_jacobian  # noqa
from kmc_leakage import mean_field
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.tddb import IncrementalCoupling
from kmc_leakage.electrostatics import PoissonSolver

config, defects, snap = load_state(2, 150)
solver = PoissonSolver(config.device, config.simulation.grid)
coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
sim = KMCSimulation(config, defects=defects, coupling=coupling, solver=solver)
N = sim.n_defects
p = snap["occupancy"]
J = np.load(os.path.join(HERE, "J_s2_n150.npy"))
_, JT = transport_jacobian(sim, p)
JT = JT.toarray()
I = np.eye(N)
M = np.linalg.solve(I - JT, I - J)
w = np.linalg.eigvals(M)
print("eig of (I-JT)^-1 (I-J): min Re %.3g max Re %.3g max |Im| %.3g" % (w.real.min(), w.real.max(), np.abs(w.imag).max()))
print("sorted by |w-1| (largest 15):", np.round(w[np.argsort(-np.abs(w - 1))][:15], 3))
print("eig(I-J) smallest |.|:", np.round(np.sort(np.abs(np.linalg.eigvals(I - J)))[:8], 6))
print("eig(I-JT) smallest |.|:", np.round(np.sort(np.abs(np.linalg.eigvals(I - JT)))[:8], 6))
D = J - JT
pairs = np.zeros((N, N), dtype=bool)
pairs[sim._pairs.a, sim._pairs.b] = True
pairs |= pairs.T
print("|J-JT| on neighbour pairs: %.3g, off pairs: %.3g, diag: %.3g" % (np.abs(D[pairs]).sum(), np.abs(D[~pairs & ~np.eye(N, dtype=bool)]).sum(), np.abs(np.diag(D)).sum()))
print("|J| on pairs %.3g off %.3g" % (np.abs(J[pairs]).sum(), np.abs(J[~pairs & ~np.eye(N, dtype=bool)]).sum()))
# 库仑近似：∂ln p̂_i/∂E_i ≈ −(1−p̂_i)/kT，∂E_i/∂p_j = −q·phi_matrix[j,i]
cluster_of = np.full(N, -1)
for k, m in enumerate(sim.cluster_members):
    cluster_of[m] = k
order = np.dstack(np.unravel_index(np.argsort(-np.abs(D).ravel())[:15], D.shape))[0]
p_hat = np.exp(mean_field._log_occupancy(mean_field.mean_field_flux(sim, p)[2]))
for i, j in order:
    print("i %3d j %3d  J %.3g  JT %.3g  cl %d/%d  p_i %.3g p_hat_i %.3g p_j %.3g" % (i, j, J[i, j], JT[i, j], cluster_of[i], cluster_of[j], p[i], p_hat[i], p[j]))
