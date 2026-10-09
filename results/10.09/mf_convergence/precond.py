"""原型：用冻结速率的跃迁 Jacobian 做预条件的 Anderson（ln p 变量），在不收敛快照上试。

冻结速率时 p̂_i = A_i/B_i，A_i = 俘获总速率（电极 + Σ_j k_ji·p_j），B_i = A_i + 逃逸总速率（电极 + PF + Σ_j k_ij·(1 − p_j)），
  ∂ln p̂_i/∂ln p_j = k_ji·p_j·(1 − p̂_i)/A_i + k_ij·p_j/B_i（j 与 i 不在同一个簇里；簇内由簇内主方程精确处理）。
簇成员与簇外陷阱之间的速率按簇内稳态分布 π 平均；簇成员的 A、B 也按 π 平均。
预条件残差 r = (I − J_x)⁻¹·g，在 r 上做 Anderson。

usage: precond.py SEED N [M]
"""
import os
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import spsolve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repro import load_state  # noqa: E402
from kmc_leakage import mean_field  # noqa: E402
from kmc_leakage.kmc import _B_R2A, _B_R2C, _B_R3A, _B_R3C, _B_R4A, _B_R4C, _B_R5A, _B_R5C, _B_R8, KMCSimulation  # noqa
from kmc_leakage.tddb import IncrementalCoupling  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402


def transport_jacobian(sim, p):
    """(g, J_x)：ln 残差与冻结速率下 ∂ln p̂/∂ln p 的稀疏近似（含 p̂ 本身）。"""
    rates = mean_field.mean_field_rates(sim, p)
    n = sim.n_defects
    b = rates.blocks
    capture = b[_B_R2C] + b[_B_R4C] + b[_B_R2A] + b[_B_R4A]
    escape = b[_B_R3C] + b[_B_R5C] + b[_B_R3A] + b[_B_R5A] + b[_B_R8]
    A = capture + rates.hop_in
    B = A + escape + rates.hop_out
    pair_rate = rates.hop.copy()
    a_idx, b_idx = sim._pairs.a, sim._pairs.b
    cluster_of = rates.cluster_of
    for table, pi in zip(rates.tables, rates.distributions):
        c = table.cluster
        blk = table.block
        cap_s = blk[_B_R2C] + blk[_B_R4C] + blk[_B_R2A] + blk[_B_R4A] + table.hop_in
        esc_s = blk[_B_R3C] + blk[_B_R5C] + blk[_B_R3A] + blk[_B_R5A] + blk[_B_R8] + table.hop_out
        occ = c.states.astype(float)
        A[c.members] = pi @ cap_s
        B[c.members] = pi @ (cap_s + esc_s)
        boundary = np.concatenate([c.in_cols, c.out_cols])
        pair_rate[c.pairs[boundary]] = pi @ table.hop[:, boundary]
    _, _, p_hat = mean_field.mean_field_flux(sim, p)
    same = (cluster_of[a_idx] == cluster_of[b_idx]) & (cluster_of[a_idx] >= 0)
    k = np.where(same, 0.0, pair_rate)
    # 有序对 a→b（速率 k）：b 的俘获里有 k·p_a，a 的逃逸里有 k·(1 − p_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        rows = np.concatenate([b_idx, a_idx])
        cols = np.concatenate([a_idx, b_idx])
        vals = np.concatenate([
            np.where(A[b_idx] > 0, k * p[a_idx] * (1.0 - p_hat[b_idx]) / A[b_idx], 0.0),
            np.where(B[a_idx] > 0, k * p[b_idx] / B[a_idx], 0.0),
        ])
    J = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    return p_hat, J


if __name__ == "__main__":
    seed, n = int(sys.argv[1]), int(sys.argv[2])
    m = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    config, defects, snap = load_state(seed, n)
    solver = PoissonSolver(config.device, config.simulation.grid)
    coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
    sim = KMCSimulation(config, defects=defects, coupling=coupling, solver=solver)
    N = sim.n_defects
    x = mean_field._log_occupancy(snap["occupancy"])
    xs, rs = [], []
    start = time.time()
    for it in range(1, 201):
        p = np.exp(x)
        p_hat, J = transport_jacobian(sim, p)
        g = mean_field._log_occupancy(p_hat) - x
        res = float(np.max(np.abs(g)))
        print(it, "%.3e" % res, "wall %.0f" % (time.time() - start), flush=True)
        if res <= 1e-10:
            break
        r = spsolve(identity(N, format="csc") - J, g)
        xs.append(x)
        rs.append(r)
        if len(xs) > m + 1:
            xs.pop(0)
            rs.pop(0)
        step = x + r
        if len(xs) > 1 and m > 0:
            dx = np.diff(np.array(xs), axis=0).T
            dr = np.diff(np.array(rs), axis=0).T
            gamma = np.linalg.lstsq(dr, r, rcond=None)[0]
            step = step - (dx + dr) @ gamma
        x = np.clip(step, mean_field._LOG_FLOOR, 0.0)
    np.save(os.path.join(HERE, "x_precond_s{}_n{}_m{}.npy".format(seed, n, m)), x)
