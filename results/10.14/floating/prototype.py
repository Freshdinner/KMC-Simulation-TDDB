"""ADR-056 原型：悬空导电簇等电位前后，同一个快照上的生成率对比。

usage: prototype.py SEED N [r_c_nm ...]

在 `../../10.13/dense_sites/snap_s{SEED}_n{N:04d}.npz`（ADR-055 验收，r_c = 0.3 nm 时没有击穿）上重建缺陷集合（初始缺陷按同一种子
重新生成并核对位置，生成空位按快照的位置、能级与 O_i⁻ 放回）与前驱位点（同一种子 `place_precursors` 重新铺、去掉已出生空位的位置），
用快照的平均场占据。对每个 r_c 比较：
- old：只有接电极的导电簇等电位（ADR-054/055，悬空导电簇的编号全部拆开即得）；
- new：悬空导电簇也等电位（ADR-056）。
输出导电簇规模、线性方程的条件数、成员处电势的离散、悬空导电簇的电势、ΣG 与生成加权深度、生成落在悬空导电簇下方缺口（簇最低点与
负极之间、横向 1 nm 内）的占比，以及 ΣG 最大的位点。
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

from kmc_leakage import tddb as T  # noqa: E402
from kmc_leakage.defects import append_defects, generate_defects, layer_bulk_defects, minimum_image_delta  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import KMCSimulation  # noqa: E402

nm = constants.nano
seed, n_snap = int(sys.argv[1]), int(sys.argv[2])
radii = [float(r) for r in sys.argv[3:]] or [0.3, 0.35, 0.4]
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
config = replace(config, simulation=replace(config.simulation, seed=seed))
device = config.device
snap = np.load(os.path.join(ROOT, "results", "10.13", "dense_sites", "snap_s{}_n{:04d}.npz".format(seed, n_snap)))

rng = np.random.default_rng(seed)
initial = generate_defects(device, rng)
n0 = len(initial)
assert n0 == int(snap["n_initial"]) and np.allclose(initial.positions, snap["positions"][:n0], rtol=0, atol=1e-15)
start = time.time()
precursors = T.place_precursors(device, initial.positions, rng)
born = snap["positions"][n0:]
keep = np.ones(len(precursors), dtype=bool)
index = {tuple(p): k for k, p in enumerate(precursors.positions)}
for p in born:
    keep[index[tuple(p)]] = False
precursors = T.Precursors(precursors.positions[keep])
print("seed {} n {}: {} defects, {} precursors (placed in {:.0f} s)".format(
    seed, n_snap, len(snap["positions"]), len(precursors), time.time() - start), flush=True)

vacancies = layer_bulk_defects(device, 0, born, rng)
vacancies = replace(vacancies, ED=snap["ED"][n0:], fixed_charge=snap["fixed_charge"][n0:])
defects = append_defects(initial, vacancies)
start = time.time()
inner = KMCSimulation(config, defects=defects)
print("inner built in {:.0f} s".format(time.time() - start), flush=True)
occupancy = snap["occupancy"]
z = defects.positions[:, 2]


def gap_share(rates, labels, sides):
    """生成落在悬空导电簇（≥ 2 个空位）下方缺口里的占比：位点在簇某个成员横向 1 nm 内、z 低于簇最低点。"""
    points = precursors.positions
    inside = np.zeros(len(points), dtype=bool)
    for label in np.unique(labels[sides < 0]):
        member = np.flatnonzero(labels == label)
        if member.size < 2:
            continue
        bottom = z[member].min()
        candidate = np.flatnonzero(points[:, 2] < bottom)
        lateral = np.linalg.norm(minimum_image_delta(
            points[candidate, None, :2] - defects.positions[None, member, :2], device.Lx, device.Ly), axis=-1).min(axis=1)
        inside[candidate[lateral < 1.0 * nm]] = True
    return rates[inside].sum() / rates.sum()


for r_c in radii:
    tddb = replace(config.tddb, percolation_distance=r_c * nm)
    labels = T.distance_clusters(defects.positions, device, tddb.percolation_distance)
    sides = T.conducting_clusters(defects.positions, device, tddb.percolation_distance)
    sizes = np.bincount(labels)
    floating = (sizes[labels] >= 2) & (sides < 0)
    big = np.unique(labels[floating])
    print("\n== r_c {:.2f} nm: {} floating clusters with >= 2 vacancies ({} members, largest {}); electrode members {}".format(
        r_c, big.size, int(floating.sum()), int(sizes[big].max(initial=0)), int(np.sum(sides >= 0))), flush=True)
    for mode in ("old", "new"):
        original = T.distance_clusters
        if mode == "old":
            T.distance_clusters = lambda positions, device, distance: np.arange(len(positions))
        start = time.time()
        g = T.generation_rates(inner, occupancy, tddb, precursors)
        T.distance_clusters = original
        rates = g.rates
        phi, shift = T._conductor_potential(inner, occupancy, sides, labels if mode == "new" else np.arange(len(z)))
        at = inner.solver.interpolate(phi, defects.positions)
        spread = max((np.ptp(at[labels == label]) for label in big), default=0.0)
        top = np.argsort(rates)[::-1][:5]
        depth = np.sum(rates * precursors.positions[:, 2]) / rates.sum() / nm
        print("  {}: sum G {:.4g} 1/s (electrode {:.3g}, trap {:.3g}) in {:.0f} s; G-weighted z {:.2f} nm; gap share {:.3f}; "
              "max potential spread in floating clusters {:.3g} V".format(
                  mode, rates.sum(), np.sum(g.electrode * g.bond_breaking), np.sum(g.trap * g.bond_breaking),
                  time.time() - start, depth, gap_share(rates, labels, sides), spread), flush=True)
        print("     top sites z [nm] {} G [1/s] {}".format(
            np.round(precursors.positions[top, 2] / nm, 2).tolist(), ["{:.3g}".format(v) for v in rates[top]]))
    if big.size:
        members = np.flatnonzero(floating | (sides >= 0))
        sub = inner.coupling.phi_matrix[np.ix_(members, members)]
        print("  cond(G on members) {:.3g}".format(np.linalg.cond(sub)))
        cluster_potential = [at[labels == label][0] for label in big]
        tall = big[np.argsort([np.ptp(z[labels == label]) for label in big])[::-1][:3]]
        for label in tall:
            member = labels == label
            print("  cluster {} members, z {:.2f}-{:.2f} nm, potential {:.3f} V (applied at mid-height {:.3f} V)".format(
                int(member.sum()), z[member].min() / nm, z[member].max() / nm, at[member][0],
                config.simulation.voltage * z[member].mean() / device.Lz))
