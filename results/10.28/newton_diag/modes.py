"""`diag.py` 找到的慢方向落在哪：J_F 模 ≥ 0.6 的特征值个数，各慢方向的权重有多少落在"被切开的强耦合分量"（全空组态下耦合强度
≥ 1e-3 连边、大小超过簇上限 8、被 `coupled_clusters` 在内部提高阈值切开）上，其中又有多少在切开后成了孤立陷阱或小簇的成员。
结果写到 modes.txt。

usage: modes.py LABEL STATE_NPZ [LABEL STATE_NPZ ...]
"""
import os
import sys
from dataclasses import replace

import numpy as np
from scipy import constants
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage import tddb as tddb_module  # noqa: E402
from kmc_leakage.clusters import coupling_strength  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import _CLUSTER_COUPLING, _CLUSTER_MAX_SIZE, Conductors, KMCSimulation  # noqa: E402
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
lines = []
for label, path in zip(sys.argv[1::2], sys.argv[2::2]):
    state = load_state(path)
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
    n, pairs = sim.n_defects, sim._pairs
    blocks, hop = sim._rates(np.zeros(n, dtype=bool))
    keep = coupling_strength(pairs.a, pairs.b, hop, blocks.sum(axis=0)) >= _CLUSTER_COUPLING
    graph = coo_matrix((np.ones(int(keep.sum())), (pairs.a[keep], pairs.b[keep])), shape=(n, n))
    component = connected_components(graph, directed=False)[1]
    size = np.bincount(component)[component]
    split = size > _CLUSTER_MAX_SIZE  # 在被切开的分量里
    cluster_size = np.ones(n, dtype=int)
    for m in sim.cluster_members:
        cluster_size[m] = m.size

    d = np.load(os.path.join(HERE, "diag_{}.npz".format(label)))
    values, vectors = d["values"], d["vectors"]
    slow = np.abs(values) >= 0.6
    lines.append("{}：陷阱 {}，在被切开分量里的 {}（{:.0%}），其中切开后成了孤立陷阱的 {}".format(
        label, n, split.sum(), split.mean(), int(np.sum(split & (cluster_size == 1)))))
    lines.append("  |λ| ≥ 0.6 的特征值 {} 个，≥ 0.7 的 {} 个，≥ 0.8 的 {} 个".format(
        int(slow.sum()), int(np.sum(np.abs(values) >= 0.7)), int(np.sum(np.abs(values) >= 0.8))))
    for mode in np.flatnonzero(slow):
        weight = np.abs(vectors[:, mode]) ** 2
        weight /= weight.sum()
        lines.append("  λ = {:+.3f}{:+.3f}i：权重在被切开分量上 {:.0%}（其中孤立陷阱 {:.0%}），分量外的孤立陷阱 {:.0%}".format(
            values[mode].real, values[mode].imag, weight[split].sum(), weight[split & (cluster_size == 1)].sum(),
            weight[~split & (cluster_size == 1)].sum()))
text = "\n".join(lines)
print(text)
open(os.path.join(HERE, "modes.txt"), "w", encoding="utf-8").write(text + "\n")
