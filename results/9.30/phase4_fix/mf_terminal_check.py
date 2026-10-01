"""Phase 4 终态上的内层平均场自洽检查（ADR-037 → ADR-038）。

usage: mf_terminal_check.py SEED [VOLTAGE ...]
重建：examples/zro2_tddb.yaml + DefectEnsemble.uniform(positions, ED, material, band_reference=E_B)，
从 npz 的 occupancy 热启动。代码版本由 PYTHONPATH 决定（HEAD 归档 = 改动前，工作树 = 改动后）。
输出每个电压一行：电流、阴极净注入、阳极净收集、放热记账（介质、两电极、总和 / U·I_陷阱）、诊断。
"""
import sys
import time
from dataclasses import replace

import numpy as np
from scipy import constants

from kmc_leakage import mean_field
from kmc_leakage.clusters import cluster_generator, stationary_distribution_gth
from kmc_leakage.defects import DefectEnsemble
from kmc_leakage.io import load_config
from kmc_leakage.kmc import _B_R2C, _B_R3C, _B_R4C, _B_R5C, _B_R2A, _B_R3A, _B_R4A, _B_R5A, _B_R8, KMCSimulation
from kmc_leakage.mean_field import solve_mean_field
from kmc_leakage.thermal import heat_release
from kmc_leakage.tddb import _cluster_diagnostics

NEW = hasattr(mean_field, "mean_field_hop_rates")
Q = constants.elementary_charge

seed = int(sys.argv[1])
voltages = [float(v) for v in sys.argv[2:]] or [3.0, 0.0]
data = np.load("results/9.30/phase4/p4_thermal_s{}.npz".format(seed))
base = load_config("examples/zro2_tddb.yaml")
device = base.device
material = device.stack.single_material
defects = DefectEnsemble.uniform(data["positions"], data["ED"], material,
                                 band_reference=float(device.layer_band_references()[0]))


def electrode_net(sim, p):
    """各陷阱的 (阴极净注入, 阳极净收集) [1/s]：孤立陷阱按平均场，簇成员按簇内稳态分布。"""
    energy, field, bc, ba = sim._trap_state(p)
    blocks = sim._electrode_blocks(energy, field, bc, ba)
    cathode = (1 - p) * (blocks[_B_R2C] + blocks[_B_R4C]) - p * (blocks[_B_R3C] + blocks[_B_R5C])
    anode = p * (blocks[_B_R3A] + blocks[_B_R5A] + blocks[_B_R8]) - (1 - p) * (blocks[_B_R2A] + blocks[_B_R4A])
    if NEW:
        rates = mean_field.mean_field_rates(sim, p)
        pairs = zip(rates.tables, rates.distributions)
    else:
        pairs = []
        for t in sim._cluster_tables(p):
            c, b = t.cluster, t.block
            capture = b[_B_R2A] + b[_B_R4A] + b[_B_R2C] + b[_B_R4C] + t.hop_in
            escape = b[_B_R3A] + b[_B_R5A] + b[_B_R8] + b[_B_R3C] + b[_B_R5C] + t.hop_out
            pairs.append((t, stationary_distribution_gth(cluster_generator(
                c.states, capture, escape, c.intra_a, c.intra_b, t.hop[:, c.intra_cols]))))
    for t, pi in pairs:
        c, b = t.cluster, t.block
        n = c.states.astype(float)
        cathode[c.members] = pi @ ((1 - n) * (b[_B_R2C] + b[_B_R4C]) - n * (b[_B_R3C] + b[_B_R5C]))
        anode[c.members] = pi @ (n * (b[_B_R3A] + b[_B_R5A] + b[_B_R8]) - (1 - n) * (b[_B_R2A] + b[_B_R4A]))
    return cathode, anode


for voltage in voltages:
    start = time.time()
    config = replace(base, simulation=replace(base.simulation, voltage=voltage, seed=seed))
    sim = KMCSimulation(config, defects=defects)
    mf = solve_mean_field(sim, data["occupancy"])
    p = mf.occupancy
    cathode, anode = electrode_net(sim, p)
    heat = heat_release(sim, p)
    i_trap = (mf.by_mechanism["tat"] + mf.by_mechanism["pf"])
    total = heat.dielectric + heat.cathode + heat.anode
    n_clusters, max_cluster, largest, n_split = _cluster_diagnostics(sim)
    print("{} seed={} U={:.2f} N={} J={:.6e} A/cm2 conv={} it={} | cathode_in={:.4e} anode_out={:.4e} /s | "
          "P_diel={:.4e} P_cat={:.4e} P_an={:.4e} sum={:.4e} W  U*I_trap={:.4e} W  sum/UI={:.6g} diel/UI={:.4g} | "
          "clusters={} max={} largest={} split={} wall={:.0f}s".format(
              "new" if NEW else "head", seed, voltage, sim.n_defects, mf.current_density * 1e-4, mf.converged,
              mf.iterations, cathode.sum(), anode.sum(), heat.dielectric, heat.cathode, heat.anode, total,
              voltage * i_trap, total / (voltage * i_trap) if voltage else float("nan"),
              heat.dielectric / (voltage * i_trap) if voltage else float("nan"),
              n_clusters, max_cluster, largest, n_split, time.time() - start), flush=True)
