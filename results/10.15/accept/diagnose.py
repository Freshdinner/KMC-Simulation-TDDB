"""接负极导电簇长不起来的诊断：从停跑时存下的续跑状态 `state_s{SEED}.npz` 出发，按 `run_tddb` 的同一套步骤（导体划分、输运与热迭代、
生成率）重算一次各前驱位点的生成率，统计：

- 按出生高度 z 分层（0.1 nm 一层）的位点数、ΣG（电极项 / 陷阱项）、位点平均 R_1e 与 P_bb；
- 能把负极导体接长的位点：z < r_c（直接接负极，前驱窗口从 z = min_separation 起，这一层厚 r_c − min_separation），或离接负极导体成员
  < r_c；它们的个数、ΣG 占比与期望等待时间 1/ΣG_接长；
- 同样口径下"接到最大悬空导电簇"的位点作对照。
结果写到 `diagnose_s{SEED}.txt` 与 `diagnose_s{SEED}.npz`（逐层数据，供 plot.py 出图）。
usage: diagnose.py SEED
"""
import os
import sys

import numpy as np
from scipy import constants
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage import tddb as T  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402

nm = constants.nano
seed = int(sys.argv[1])
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
device, sim, tddb = config.device, config.simulation, config.tddb
state = T.load_state(os.path.join(HERE, "state_s{}.npz".format(seed)))
defects = state.defects
solver = PoissonSolver(device, sim.grid)
coupling = T.IncrementalCoupling.build(solver, defects.positions, defects.field_side)
layout = T.conductor_layout(defects, coupling, solver, sim.voltage, tddb.percolation_distance)
step = T._steady_state(config, layout, defects, coupling, solver, state.occupancy, state.temperature)
rates = T.generation_rates(step.transport, tddb, state.precursors, step.field)
G = rates.rates
G_el = rates.electrode * rates.bond_breaking
G_tr = rates.trap * rates.bond_breaking
sites = state.precursors.positions
z = sites[:, 2] / nm
r_c = tddb.percolation_distance
period = np.array([device.Lx, device.Ly])


def near(points):
    """各位点离 `points`（(K, 3) [m]）的最近距离 < r_c（xy 周期）。"""
    if len(points) == 0:
        return np.zeros(len(sites), dtype=bool)
    tree = cKDTree(points[:, :2] % period, boxsize=period)
    hit = np.zeros(len(sites), dtype=bool)
    for i, neighbors in enumerate(tree.query_ball_point(sites[:, :2] % period, r_c)):
        if neighbors:
            dz = points[neighbors, 2] - sites[i, 2]
            dxy = (points[neighbors, :2] - sites[i, :2] + 0.5 * period) % period - 0.5 * period
            hit[i] = np.any(np.hypot(np.hypot(dxy[:, 0], dxy[:, 1]), dz) < r_c)
    return hit


sides, labels = layout.sides, layout.labels
negative_members = defects.positions[(sides == 0) | (sides == 2)]
direct = sites[:, 2] < r_c
extend = direct | near(negative_members)
sizes = np.bincount(labels)
floating = np.where(sides < 0, sizes[labels], 0)
largest = labels[int(np.argmax(floating))]
join_floating = near(defects.positions[labels == largest])

edges = np.arange(0.2, 8.0 + 1e-9, 0.1)
index = np.clip(np.digitize(z, edges) - 1, 0, edges.size - 2)
count = np.bincount(index, minlength=edges.size - 1)
G_bin = np.bincount(index, G, minlength=edges.size - 1)
G_el_bin = np.bincount(index, G_el, minlength=edges.size - 1)
G_tr_bin = np.bincount(index, G_tr, minlength=edges.size - 1)
safe = np.maximum(count, 1)
R1e_bin = np.bincount(index, rates.electrode + rates.trap, minlength=edges.size - 1) / safe
Pbb_bin = np.bincount(index, rates.bond_breaking, minlength=edges.size - 1) / safe
R_el_bin = np.bincount(index, rates.electrode, minlength=edges.size - 1) / safe
R_tr_bin = np.bincount(index, rates.trap, minlength=edges.size - 1) / safe

total = G.sum()
lines = ["seed {}: state at n = {} generated, t = {:.6g} s; {} precursor sites; sum G = {:.4g} 1/s (electrode {:.3g}, trap {:.3g})".format(
    seed, len(defects) - state.n_initial, state.time, len(sites), total, G_el.sum(), G_tr.sum())]
lines.append("negative conductor members {} (z = {} nm)".format(
    len(negative_members), " ".join("{:.3f}".format(v) for v in negative_members[:, 2] / nm)))
for name, mask in (("direct contact (z < r_c = {:.2f} nm)".format(r_c / nm), direct),
                   ("extend negative conductor (direct or < r_c of a member)", extend),
                   ("join largest floating cluster ({} vacancies, z {:.2f}-{:.2f} nm)".format(
                       sizes[largest], defects.positions[labels == largest, 2].min() / nm,
                       defects.positions[labels == largest, 2].max() / nm), join_floating)):
    g = G[mask].sum()
    lines.append("  {}: {} sites, sum G = {:.3g} 1/s ({:.2g} of total), expected wait {:.3g} s; per-site mean R_1e {:.3g} 1/s, "
                 "P_bb {:.3g}".format(name, int(mask.sum()), g, g / total, 1.0 / g if g > 0 else float("inf"),
                                       float(np.mean(rates.electrode[mask] + rates.trap[mask])) if mask.any() else 0.0,
                                       float(np.mean(rates.bond_breaking[mask])) if mask.any() else 0.0))
lines.append("  z bin [nm]   sites   sum G   share   G_el    G_trap   <R_1e,el>  <R_1e,trap>  <P_bb>")
for k in range(edges.size - 1):
    if edges[k] < 3.0 or k % 5 == 0:
        lines.append("  {:.1f}-{:.1f}  {:7d}  {:8.3g}  {:5.3f}  {:8.3g}  {:8.3g}  {:9.3g}  {:9.3g}  {:9.3g}".format(
            edges[k], edges[k + 1], count[k], G_bin[k], G_bin[k] / total, G_el_bin[k], G_tr_bin[k], R_el_bin[k],
            R_tr_bin[k], Pbb_bin[k]))
text = "\n".join(lines)
print(text)
with open(os.path.join(HERE, "diagnose_s{}.txt".format(seed)), "w", encoding="utf-8") as handle:
    handle.write(text + "\n")
np.savez(os.path.join(HERE, "diagnose_s{}.npz".format(seed)), edges=edges, count=count, G=G_bin, G_el=G_el_bin, G_tr=G_tr_bin,
         R_el=R_el_bin, R_tr=R_tr_bin, P_bb=Pbb_bin, G_extend=G[extend].sum(), G_direct=G[direct].sum(),
         n_extend=int(extend.sum()), n_direct=int(direct.sum()), G_total=total)
