"""weak_tip seed 1 后期平均场为什么总要牛顿法：在一个续跑状态上重求这一步的平均场不动点，记下每次求 F 的残差与残差最大的缺陷，
再在不动点处求不动点映射 F（ln p 上）的 Jacobian，看收敛慢的方向（|λ| 接近或大于 1 的特征值）落在哪些缺陷上。

设置同 `../../10.27/weak_tip/drive.py`（σ = 0.6 nm、p₀ = 0.9 eÅ，其余同 ea160）。输运按 `tddb.solve_transport` 的口径建在陷阱上，
温度取续跑状态里的缺陷处温度（这一步热迭代的起点，seed 1 后期只迭代 1 次，就是这一步用的温度）。
结果写到 diag_{LABEL}.npz 与 diag_{LABEL}.txt。

usage: diag.py STATE_NPZ LABEL
"""
import os
import sys
import time
from dataclasses import replace

import numpy as np
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage import mean_field  # noqa: E402
from kmc_leakage import tddb as tddb_module  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import KMCSimulation, Conductors  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402
from kmc_leakage.tddb import IncrementalCoupling, conductor_layout, load_state  # noqa: E402

nm = constants.nano
path, label = sys.argv[1], sys.argv[2]
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
stack = config.device.stack
layers = tuple(replace(layer, defects=replace(layer.defects, min_separation=0.42 * nm)) for layer in stack.layers)
config = replace(config, device=replace(config.device, stack=replace(stack, layers=layers)),
                 simulation=replace(config.simulation, seed=1),
                 tddb=replace(config.tddb, transfer_integral=2.4 * constants.electron_volt,
                              activation_energy=1.60 * constants.electron_volt, activation_energy_tail=0.0,
                              field_averaging_width=0.6 * nm,
                              dipole_moment=0.9 * constants.elementary_charge * constants.angstrom))
state = load_state(path)
device, defects = config.device, state.defects
solver = PoissonSolver(device, config.simulation.grid)
coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
layout = conductor_layout(defects, coupling, solver, config.simulation.voltage, config.tddb.transfer_integral)
traps, members = layout.traps, layout.members
phi_at, field_at, phi_bar = layout.background
conductors = Conductors(positions=defects.positions[members], fermi_level=layout.fermi_level,
                        positive_share=layout.positive_share, r_D=defects.r_D[members],
                        temperature=state.temperature[members], phi_at=phi_at, field_at=field_at, phi_bar=phi_bar)
sim = KMCSimulation(config, defects=tddb_module._take(defects, traps), coupling=layout.coupling, solver=solver,
                    temperature=state.temperature[traps], conductors=conductors)
n = sim.n_defects
lines = ["{}：{} 个缺陷，其中陷阱 {}、导体成员 {}；强耦合簇 {} 个".format(
    label, len(defects), n, members.size, len(sim.cluster_members))]
n_clusters, max_cluster, largest, n_split = tddb_module._cluster_diagnostics(sim)
lines.append("  最大簇 {}，切分前最大强耦合分量 {}，被切开的分量 {}".format(max_cluster, largest, n_split))

# 每次求 F 的残差与残差最大的缺陷
history = []
flux = mean_field.mean_field_flux


def recorded(s, p):
    out = flux(s, p)
    gap = mean_field._log_occupancy(out[2]) - mean_field._log_occupancy(p)
    k = int(np.argmax(np.abs(gap)))
    history.append((float(np.abs(gap[k])), k))
    return out


mean_field.mean_field_flux = recorded
began = time.time()
start = state.occupancy[traps]
result = mean_field.solve_mean_field(sim, start)
mean_field.mean_field_flux = flux
lines.append("  求解：{} 次求 F、牛顿 {} 步、残差 {:.1e}、收敛 {}，{:.0f} s".format(
    result.iterations, result.newton_steps, result.residual, result.converged, time.time() - began))
residuals = np.array([h[0] for h in history])
worst = np.array([h[1] for h in history])
anderson = residuals[:mean_field._ANDERSON_MAX_ITER]
lines.append("  Anderson 残差：" + " ".join("{:.0e}".format(r) for r in anderson))
lines.append("  Anderson 残差最大的缺陷（陷阱编号）：" + " ".join(str(k) for k in worst[:mean_field._ANDERSON_MAX_ITER]))

# 不动点处 F 的 Jacobian：J_F = I + ∂G/∂x（分组有限差分，ADR-058）
p = result.occupancy
x = mean_field._log_occupancy(p)
current = mean_field._evaluate(sim, x)
groups = mean_field._jacobian_groups(sim)
jac_g = mean_field._jacobian(sim, current, groups)
jac_f = np.eye(n) + jac_g
values, vectors = np.linalg.eig(jac_f)
order = np.argsort(-np.abs(values))
values, vectors = values[order], vectors[:, order]
lines.append("  J_F（{} 组）模最大的特征值：".format(len(groups)) + " ".join(
    "{:.3f}{:+.3f}i".format(v.real, v.imag) for v in values[:12]))
g_values = np.linalg.eigvals(jac_g)
lines.append("  J_G 模最小的特征值：" + " ".join(
    "{:.3f}{:+.3f}i".format(v.real, v.imag) for v in g_values[np.argsort(np.abs(g_values))][:6]))

cluster_of = np.full(n, -1)
for c, m in enumerate(sim.cluster_members):
    cluster_of[m] = c
z = sim.defects.positions[:, 2] / nm
pairs = sim._pairs
nearest = np.full(n, np.inf)
np.minimum.at(nearest, pairs.a, pairs.r / nm)
sizes = np.array([m.size for m in sim.cluster_members])
generated = traps >= state.n_initial
for mode in range(4):
    weight = np.abs(vectors[:, mode]) ** 2
    weight /= weight.sum()
    top = np.argsort(-weight)[:10]
    lines.append("  特征值 {:.3f}{:+.3f}i：权重前 10 的陷阱占 {:.0%}".format(
        values[mode].real, values[mode].imag, weight[top].sum()))
    for k in top:
        c = cluster_of[k]
        lines.append("    #{:4d} 权重 {:.2f} z {:.2f} nm p {:.3e} 簇 {}（{} 个成员）最近邻 {:.2f} nm 生成空位 {}".format(
            k, weight[k], z[k], p[k], c, sizes[c] if c >= 0 else 1, nearest[k], bool(generated[k])))

np.savez(os.path.join(HERE, "diag_{}.npz".format(label)), residuals=residuals, worst=worst, occupancy=p,
         values=values, vectors=vectors, jacobian_g=jac_g, cluster_of=cluster_of, z=z, nearest=nearest,
         traps=traps, positions=sim.defects.positions, iterations=result.iterations, newton=result.newton_steps)
text = "\n".join(lines)
print(text)
open(os.path.join(HERE, "diag_{}.txt".format(label)), "w", encoding="utf-8").write(text + "\n")
