"""评估第 4 条：把强耦合簇的大小上限（`kmc._CLUSTER_MAX_SIZE`，现为 8）调大，能不能消掉 weak_tip seed 1 后期平均场的慢方向，代价多大。

- `cap.py static`：第 1000 / 1225 个空位的续跑状态上，上限取 8、10、12、16 时的分簇（按全空组态的耦合强度 ≥ 1e-3 连边，超限的分量
  在内部提高阈值切开，同 `clusters.coupled_clusters`）：簇大小分布、仍被切开的分量里的陷阱数与其中成了孤立陷阱的数目、簇内组态总数
  Σ2^k（组态表的计算量）与 Σ(2^k)³（GTH 约化的计算量），写到 static.txt；
- `cap.py dynamic CAP`：第 1225 个空位的状态上，上限取 CAP 建引擎，记每次求 F 的耗时，从续跑状态的占据起求平均场不动点（Anderson 混合，
  外推不到判据就改用牛顿法，ADR-066），不动点处求 F 的分组有限差分 Jacobian 与特征值，写到 cap{CAP}.txt / cap{CAP}.npz。

设置同 `../../10.27/weak_tip/drive.py`（σ = 0.6 nm、p₀ = 0.9 eÅ），输运按 `tddb.solve_transport` 的口径建在陷阱上。

usage: cap.py static | cap.py dynamic CAP
"""
import os
import sys
import time
from dataclasses import replace

import numpy as np
from scipy import constants
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
WEAK_TIP = os.path.join(ROOT, "results", "10.27", "weak_tip")
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage import kmc as kmc_module  # noqa: E402
from kmc_leakage import mean_field  # noqa: E402
from kmc_leakage import tddb as tddb_module  # noqa: E402
from kmc_leakage.clusters import coupled_clusters, coupling_strength  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import _CLUSTER_COUPLING, Conductors, KMCSimulation  # noqa: E402
from kmc_leakage.tddb import IncrementalCoupling, conductor_layout, load_state  # noqa: E402

nm = constants.nano
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
stack = config.device.stack
layers = tuple(replace(layer, defects=replace(layer.defects, min_separation=0.42 * nm)) for layer in stack.layers)
config = replace(config, device=replace(config.device, stack=replace(stack, layers=layers)),
                 simulation=replace(config.simulation, seed=1),
                 tddb=replace(config.tddb, transfer_integral=2.4 * constants.electron_volt,
                              activation_energy=1.60 * constants.electron_volt, activation_energy_tail=0.0,
                              field_averaging_width=0.6 * nm,
                              dipole_moment=0.9 * constants.elementary_charge * constants.angstrom))


def build(n_generated):
    """第 n_generated 个空位的续跑状态上、按当前 `kmc._CLUSTER_MAX_SIZE` 分簇的陷阱引擎与热启动占据。"""
    state = load_state(os.path.join(WEAK_TIP, "state_s1_n{:04d}.npz".format(n_generated)))
    defects = state.defects
    solver = PoissonSolver(config.device, config.simulation.grid)
    coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
    layout = conductor_layout(defects, coupling, solver, config.simulation.voltage, config.tddb.transfer_integral)
    traps, members = layout.traps, layout.members
    phi_at, field_at, phi_bar = layout.background
    conductors = Conductors(positions=defects.positions[members], fermi_level=layout.fermi_level,
                            positive_share=layout.positive_share, r_D=defects.r_D[members],
                            temperature=state.temperature[members], phi_at=phi_at, field_at=field_at, phi_bar=phi_bar)
    sim = KMCSimulation(config, defects=tddb_module._take(defects, traps), coupling=layout.coupling, solver=solver,
                        temperature=state.temperature[traps], conductors=conductors)
    return sim, state.occupancy[traps]


def static():
    lines = []
    for n_generated in (1000, 1225):
        sim, _ = build(n_generated)
        n, pairs = sim.n_defects, sim._pairs
        blocks, hop = sim._rates(np.zeros(n, dtype=bool))
        strength = coupling_strength(pairs.a, pairs.b, hop, blocks.sum(axis=0))
        keep = strength >= _CLUSTER_COUPLING
        graph = coo_matrix((np.ones(int(keep.sum())), (pairs.a[keep], pairs.b[keep])), shape=(n, n))
        component = connected_components(graph, directed=False)[1]
        component_size = np.bincount(component)[component]
        lines.append("第 {} 个空位：陷阱 {}，强耦合分量最大 {}，大于 8 的分量 {} 个、共 {} 个陷阱".format(
            n_generated, n, component_size.max(), np.unique(component[component_size > 8]).size,
            int(np.sum(component_size > 8))))
        base = None
        for cap in (8, 10, 12, 16):
            clusters = coupled_clusters(n, pairs.a, pairs.b, strength, _CLUSTER_COUPLING, cap)
            sizes = np.array([c.size for c in clusters])
            in_cluster = np.zeros(n, dtype=bool)
            for c in clusters:
                in_cluster[c] = True
            cut = component_size > cap
            configurations = float(np.sum(2.0 ** sizes))
            gth = float(np.sum(8.0 ** sizes))
            if base is None:
                base = (configurations, gth)
            lines.append("  上限 {:2d}：簇 {} 个（大小 {}），仍被切开的分量里的陷阱 {}、其中孤立 {}；Σ2^k = {:.3g}（{:.1f} 倍），"
                         "Σ(2^k)³ = {:.3g}（{:.0f} 倍）".format(
                             cap, sizes.size, np.bincount(sizes, minlength=cap + 1)[2:].tolist(), int(cut.sum()),
                             int(np.sum(cut & ~in_cluster)), configurations, configurations / base[0], gth, gth / base[1]))
    text = "\n".join(lines)
    print(text)
    open(os.path.join(HERE, "static.txt"), "w", encoding="utf-8").write(text + "\n")


def dynamic(cap):
    kmc_module._CLUSTER_MAX_SIZE = cap
    began = time.time()
    sim, start = build(1225)
    lines = ["上限 {}：第 1225 个空位，陷阱 {}，簇 {} 个，最大 {}；建引擎 {:.0f} s".format(
        cap, sim.n_defects, len(sim.cluster_members), max(m.size for m in sim.cluster_members), time.time() - began)]
    times = []
    for _ in range(3):
        began = time.time()
        mean_field.mean_field_flux(sim, start)
        times.append(time.time() - began)
    lines.append("  每次求 F {:.1f} s（3 次的中位数）".format(np.median(times)))

    residuals = []
    flux = mean_field.mean_field_flux

    def recorded(s, p):
        out = flux(s, p)
        residuals.append(float(np.max(np.abs(mean_field._log_occupancy(out[2]) - mean_field._log_occupancy(p)))))
        return out

    mean_field.mean_field_flux = recorded
    began = time.time()
    result = mean_field.solve_mean_field(sim, start)
    mean_field.mean_field_flux = flux
    density = result.current_density * constants.centi ** 2
    lines.append("  求解：{} 次求 F、牛顿 {} 步、残差 {:.1e}、收敛 {}，{:.0f} s；J = {:.9e} A/cm²".format(
        result.iterations, result.newton_steps, result.residual, result.converged, time.time() - began, density))
    lines.append("  前 50 次求 F 的残差：" + " ".join("{:.0e}".format(r) for r in residuals[:50]))

    x = mean_field._log_occupancy(result.occupancy)
    current = mean_field._evaluate(sim, x)
    groups = mean_field._jacobian_groups(sim)
    values = np.linalg.eigvals(np.eye(sim.n_defects) + mean_field._jacobian(sim, current, groups))
    values = values[np.argsort(-np.abs(values))]
    lines.append("  J_F（{} 组）|λ| ≥ 0.6 的 {} 个、≥ 0.7 的 {} 个、≥ 0.8 的 {} 个；最大的 12 个：".format(
        len(groups), int(np.sum(np.abs(values) >= 0.6)), int(np.sum(np.abs(values) >= 0.7)),
        int(np.sum(np.abs(values) >= 0.8))) + " ".join("{:.3f}{:+.3f}i".format(v.real, v.imag) for v in values[:12]))
    np.savez(os.path.join(HERE, "cap{}.npz".format(cap)), values=values, residuals=np.array(residuals),
             occupancy=result.occupancy, current_density=result.current_density, by_defect=result.by_defect)
    text = "\n".join(lines)
    print(text)
    open(os.path.join(HERE, "cap{}.txt".format(cap)), "w", encoding="utf-8").write(text + "\n")


if sys.argv[1] == "static":
    static()
else:
    dynamic(int(sys.argv[2]))
