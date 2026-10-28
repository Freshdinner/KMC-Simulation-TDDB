"""平均场一步的耗时剖析：在 `../../10.27/weak_tip/state_s2.npz`（seed 2 第 950 个空位，1010 个缺陷）上按 weak_tip 的参数
求一次导体划分与输运–热稳态（`tddb._steady_state`）和生成率，记录：
- cProfile 的函数耗时；
- 每次求 F（`mean_field.mean_field_flux`）的耗时与相对残差 max|ln p̂ − ln p|，残差最大的陷阱在哪；
- 陷阱数、近邻对数、强耦合簇的大小分布与组态总数 R = Σ2^k；
- 簇内组态静电叠加（`_electrostatics_batch`）现做法与"只改簇成员的低秩增量"做法的耗时与差值；
- 一次求 F、生成率各自的峰值内存（tracemalloc）。
结果写到 profile.txt。

usage: python profile_step.py
"""
import cProfile
import io
import os
import pstats
import sys
import time
import tracemalloc
from dataclasses import replace

import numpy as np
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage import mean_field as MF  # noqa: E402
from kmc_leakage import tddb as T  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402

nm, eV = constants.nano, constants.electron_volt
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
stack = config.device.stack
layers = tuple(replace(layer, defects=replace(layer.defects, min_separation=0.42 * nm)) for layer in stack.layers)
config = replace(config, device=replace(config.device, stack=replace(stack, layers=layers)),
                 simulation=replace(config.simulation, seed=2),
                 tddb=replace(config.tddb, transfer_integral=2.4 * eV, activation_energy=1.60 * eV,
                              activation_energy_tail=0.0, field_averaging_width=0.6 * nm,
                              dipole_moment=0.9 * constants.elementary_charge * constants.angstrom))
device, sim_config, tddb = config.device, config.simulation, config.tddb
out = open(os.path.join(HERE, "profile.txt"), "w", encoding="utf-8")


def say(text=""):
    print(text, flush=True)
    out.write(text + "\n")


state = T.load_state(os.path.join(ROOT, "results", "10.27", "weak_tip", "state_s2.npz"))
defects = state.defects
solver = PoissonSolver(device, sim_config.grid)
began = time.time()
coupling = T.IncrementalCoupling.build(solver, defects.positions, defects.field_side)
say("{} 个缺陷（初始 {}），耦合矩阵从头建 {:.0f} s".format(len(defects), state.n_initial, time.time() - began))

calls = []
original_flux = MF.mean_field_flux


def flux(sim, occupancy):
    began = time.time()
    result = original_flux(sim, occupancy)
    calls.append((time.time() - began, np.array(occupancy, dtype=float), result[2]))
    return result


MF.mean_field_flux = flux

profile = cProfile.Profile()
profile.enable()
began = time.time()
layout = T.conductor_layout(defects, coupling, solver, sim_config.voltage, tddb.transfer_integral)
t_layout = time.time() - began
began = time.time()
step = T._steady_state(config, layout, defects, coupling, solver, state.occupancy, state.temperature)
t_steady = time.time() - began
began = time.time()
generation = T.generation_rates(step.transport, tddb, state.precursors, step.field)
t_rates = time.time() - began
profile.disable()
MF.mean_field_flux = original_flux

mf = step.transport.result
sim = step.transport.sim
say("导体划分 {:.1f} s；输运–热稳态 {:.1f} s（热迭代 {} 次，平均场求 F {} 次、牛顿 {} 步、残差 {:.1e}）；生成率 {:.1f} s".format(
    t_layout, t_steady, step.iterations, mf.iterations, mf.newton_steps, mf.residual, t_rates))

# ---- 规模 ----
sizes = np.array([c.members.size for c in sim._clusters])
states = 2 ** sizes
say("\n陷阱 F = {}，接电极导体成员 M = {}，近邻有序对 P = {}".format(sim.n_defects, layout.members.size, sim._pairs.a.size))
say("强耦合簇 {} 个，成员共 {} 个；簇大小分布 {}；组态总数 R = Σ2^k = {}，组态 × 成员 = {}".format(
    sizes.size, int(sizes.sum()), dict(zip(*np.unique(sizes, return_counts=True))), int(states.sum()),
    int(np.sum(states * sizes))))

# ---- 每次求 F ----
log = MF._log_occupancy
history = [float(np.max(np.abs(log(p_hat) - log(p)))) for _, p, p_hat in calls]
say("\n求 F {} 次（含热放热记账以外的全部），每次 {:.2f} s（中位数），合计 {:.1f} s".format(
    len(calls), float(np.median([c[0] for c in calls])), float(sum(c[0] for c in calls))))
say("逐次相对残差：" + " ".join("{:.1e}".format(r) for r in history))
ratios = np.array(history[2:]) / np.array(history[1:-1])
say("相邻两次残差之比（中位数）{:.2f}".format(float(np.median(ratios))))
cluster_of = np.full(sim.n_defects, -1)
for label, c in enumerate(sim._clusters):
    cluster_of[c.members] = label
z = sim.defects.positions[:, 2] / nm
for k in (1, len(calls) // 2, len(calls) - 2):
    _, p, p_hat = calls[k]
    g = np.abs(log(p_hat) - log(p))
    worst = np.argsort(g)[::-1][:5]
    say("第 {} 次残差最大的 5 个陷阱：".format(k + 1) + "；".join(
        "#{} z={:.2f} nm p={:.2e} 簇{} |g|={:.1e}".format(
            int(i), z[i], p[i], "（{} 个）".format(sizes[cluster_of[i]]) if cluster_of[i] >= 0 else "外", g[i])
        for i in worst))

# ---- 簇内组态的静电叠加：现做法与低秩增量 ----
p = mf.occupancy
clusters = sim._clusters
occ_states = np.concatenate([np.repeat(p[None, :], c.states.shape[0], axis=0) for c in clusters])
row_start = np.cumsum([0] + [c.states.shape[0] for c in clusters])
for c, start, stop in zip(clusters, row_start[:-1], row_start[1:]):
    occ_states[start:stop, c.members] = c.states
began = time.time()
phi_ref, field_ref, bar_ref = sim._electrostatics_batch(occ_states)
t_batch = time.time() - began

began = time.time()
cpl = sim.coupling
weight = p - sim.defects.charge_state
phi0 = sim._phi_applied_at + sim._phi_fixed_at + weight @ cpl.phi_matrix - weight * cpl.phi_self
field0 = sim._field_applied_at + sim._field_fixed_at + np.einsum("i,cij->cj", weight, cpl.field_matrix) \
    - weight * cpl.field_self
bar0 = sim._phi_bar_applied + sim._phi_bar_fixed + weight @ cpl.phi_bar
phi_low, field_low, bar_low = [], [], []
for c in clusters:
    m = c.members
    delta = c.states - p[m]  # (S, k)：组态与平均占据只在成员上不同
    phi_s = phi0 + delta @ cpl.phi_matrix[m]
    phi_s[:, m] -= delta * cpl.phi_self[m]
    vec = field0[:, None, :] + np.einsum("sk,ckj->csj", delta, cpl.field_matrix[:, m, :])
    vec[:, :, m] -= delta[None, :, :] * cpl.field_self[:, m][:, None, :]
    phi_low.append(phi_s)
    field_low.append(np.linalg.norm(vec, axis=0))
    bar_low.append(bar0 + delta @ cpl.phi_bar[m])
phi_low, field_low, bar_low = np.concatenate(phi_low), np.concatenate(field_low), np.concatenate(bar_low)
t_low = time.time() - began
say("\n簇内组态静电叠加：现做法 {:.2f} s，低秩增量（只改成员、仍算全部 N 列）{:.2f} s；最大差 电势 {:.1e} V、"
    "场 {:.1e}（相对）、剖面 {:.1e} V".format(
        t_batch, t_low, float(np.max(np.abs(phi_low - phi_ref))),
        float(np.max(np.abs(field_low - field_ref) / np.maximum(field_ref, 1.0))), float(np.max(np.abs(bar_low - bar_ref)))))
needed = [np.union1d(c.members, np.concatenate([sim._pairs.a[c.pairs], sim._pairs.b[c.pairs]])) for c in clusters]
say("每个簇真正用到的列（成员 + 与簇有关的近邻对端点）：中位数 {:.0f} 列、最多 {} 列（全部 N = {} 列）".format(
    float(np.median([n.size for n in needed])), max(n.size for n in needed), sim.n_defects))

# ---- 峰值内存 ----
tracemalloc.start()
MF.mean_field_flux(sim, p)
peak_flux = tracemalloc.get_traced_memory()[1] / 2 ** 20
tracemalloc.stop()
tracemalloc.start()
T.generation_rates(step.transport, tddb, state.precursors, step.field)
peak_rates = tracemalloc.get_traced_memory()[1] / 2 ** 20
tracemalloc.stop()
say("\n峰值内存增量：一次求 F {:.0f} MB，生成率 {:.0f} MB".format(peak_flux, peak_rates))

stream = io.StringIO()
pstats.Stats(profile, stream=stream).sort_stats("tottime").print_stats(30)
say("\n==== cProfile（按自身耗时）====")
say(stream.getvalue())
stream = io.StringIO()
pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(40)
say("\n==== cProfile（按累计耗时）====")
say(stream.getvalue())
out.close()
