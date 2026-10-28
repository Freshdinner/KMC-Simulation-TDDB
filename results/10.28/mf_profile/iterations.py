"""平均场迭代次数的检验：在 `../../10.27/weak_tip/state_s{SEED}.npz` 上按 weak_tip 的参数求一次导体划分与内层输运
（`tddb.solve_transport`，温度取续跑状态里的缺陷处温度），比较：
- 基准（判据 1e-10、Anderson 历史 5 步、新空位初值 1）：逐次相对残差、电流 J 与占据相对基准不动点的偏差；
- 判据放宽到 1e-6、1e-8：迭代次数、J 与 ΣG（`tddb.generation_rates`，等温）相对基准的偏差；
- Anderson 历史 10、20 步：迭代次数；
- 初值的上限检验：新空位（或新空位 2 nm 以内的陷阱）的初值直接取基准不动点的值，其余照旧，看迭代次数还剩多少。
结果写到 iterations_s{SEED}.txt。

usage: python iterations.py SEED
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

from kmc_leakage import mean_field as MF  # noqa: E402
from kmc_leakage import tddb as T  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402

seed = int(sys.argv[1])
nm, eV, q = constants.nano, constants.electron_volt, constants.elementary_charge
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
stack = config.device.stack
layers = tuple(replace(layer, defects=replace(layer.defects, min_separation=0.42 * nm)) for layer in stack.layers)
config = replace(config, device=replace(config.device, stack=replace(stack, layers=layers)),
                 simulation=replace(config.simulation, seed=seed),
                 tddb=replace(config.tddb, transfer_integral=2.4 * eV, activation_energy=1.60 * eV,
                              activation_energy_tail=0.0, field_averaging_width=0.6 * nm,
                              dipole_moment=0.9 * constants.elementary_charge * constants.angstrom))
device, sim_config, tddb = config.device, config.simulation, config.tddb
out = open(os.path.join(HERE, "iterations_s{}.txt".format(seed)), "w", encoding="utf-8")


def say(text=""):
    print(text, flush=True)
    out.write(text + "\n")


state = T.load_state(os.path.join(ROOT, "results", "10.27", "weak_tip", "state_s{}.npz".format(seed)))
defects = state.defects
solver = PoissonSolver(device, sim_config.grid)
began = time.time()
coupling = T.IncrementalCoupling.build(solver, defects.positions, defects.field_side)
layout = T.conductor_layout(defects, coupling, solver, sim_config.voltage, tddb.transfer_integral)
say("seed {}：{} 个缺陷（生成 {}），陷阱 {} 个，耦合矩阵与导体划分 {:.0f} s".format(
    seed, len(defects), len(defects) - state.n_initial, layout.traps.size, time.time() - began))

calls = []
original_flux = MF.mean_field_flux


def flux(sim, occupancy):
    result = original_flux(sim, occupancy)
    calls.append((np.array(occupancy, dtype=float), result[0].sum() + result[1].sum(), result[2]))
    return result


MF.mean_field_flux = flux


def solve(occupancy, tol=1e-10, history=5):
    MF._TOL, MF._HISTORY = tol, history
    del calls[:]
    began = time.time()
    transport = T.solve_transport(config, layout, defects, coupling, solver, state.temperature, occupancy)
    MF._TOL, MF._HISTORY = 1e-10, 5
    return transport, time.time() - began


def total_current(transport):
    return transport.result.current / device.area * constants.centi ** 2  # A/cm²（内层输运）


# ---- 基准 ----
ref, wall = solve(state.occupancy)
r1 = sum(ref.sim._r1_current)
p_ref = ref.result.occupancy
J_ref = total_current(ref)
G_ref = T.generation_rates(ref, tddb, state.precursors).rates
log = MF._log_occupancy
say("\n基准：求 F {} 次、{:.0f} s，残差 {:.1e}，J = {:.9e} A/cm²，ΣG = {:.9e} 1/s".format(
    ref.result.iterations, wall, ref.result.residual, J_ref, G_ref.sum()))
say("  逐次：相对残差 | 占据相对基准 max|ln p − ln p*| | J 相对基准的偏差")
for k, (p, net, p_hat) in enumerate(calls):
    J = (q * net + r1) / device.area * constants.centi ** 2
    say("  {:2d} {:8.1e} {:8.1e} {:8.1e}".format(
        k + 1, float(np.max(np.abs(log(p_hat) - log(p)))), float(np.max(np.abs(log(p) - log(p_ref)))),
        abs(J / J_ref - 1.0)))

# ---- 判据放宽 ----
for tol in (1e-6, 1e-8):
    transport, wall = solve(state.occupancy, tol=tol)
    G = T.generation_rates(transport, tddb, state.precursors).rates
    say("判据 {:.0e}：求 F {} 次、{:.0f} s；J 偏差 {:.1e}，ΣG 偏差 {:.1e}，逐位点 G 最大相对偏差 {:.1e}".format(
        tol, transport.result.iterations, wall, abs(total_current(transport) / J_ref - 1.0),
        abs(G.sum() / G_ref.sum() - 1.0), float(np.max(np.abs(G - G_ref) / np.maximum(G_ref, 1e-300)))))

# ---- Anderson 历史 ----
for history in (10, 20):
    transport, wall = solve(state.occupancy, history=history)
    say("Anderson 历史 {} 步：求 F {} 次、{:.0f} s，J 偏差 {:.1e}".format(
        history, transport.result.iterations, wall, abs(total_current(transport) / J_ref - 1.0)))

# ---- 初值的上限检验 ----
newborn = len(defects) - 1
traps = layout.traps
if layout.sides[newborn] >= 0:
    say("新空位接在导体上，跳过初值检验")
else:
    distance = np.linalg.norm(T.minimum_image_delta(
        defects.positions - defects.positions[newborn], device.Lx, device.Ly), axis=1)
    for label, near in (("新空位", np.arange(len(defects)) == newborn), ("新空位 2 nm 以内", distance <= 2.0 * nm)):
        start = state.occupancy.copy()
        chosen = near.copy()
        chosen[layout.members] = False
        start[chosen] = ref.occupancy[chosen]
        transport, wall = solve(start)
        say("初值 {} 的 {} 个陷阱取基准不动点：求 F {} 次、{:.0f} s，J 偏差 {:.1e}".format(
            label, int(chosen.sum()), transport.result.iterations, wall, abs(total_current(transport) / J_ref - 1.0)))
out.close()
