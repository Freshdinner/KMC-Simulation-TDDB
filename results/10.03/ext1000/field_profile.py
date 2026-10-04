"""阴极侧空位层的正空间电荷对膜内场与 P_bb 的影响：在快照上重建电势，沿 z 列出横向平均的场与 P_bb。

usage: field_profile.py DIR SEED N [N ...]
（DIR 为快照所在目录，相对仓库根，如 results/10.03/rerun）

每个快照：缺陷净电荷 q·(Z − p) 与外加偏压解一次泊松方程（同 `generation_rates`），输出
- 若干高度上横向平均的 E_z [V/nm]、Gaussian 平均场 |Ē| 的横向平均、横向平均的 P_bb（同式 T2，环境温度）；
- 净电荷、电荷的平均高度，以及一维面电荷估算的阴极侧 / 体内场（s₁ = s₂ + σ/(κε₀)，s₁·z̄ + s₂·(Lz − z̄) = U + V_bi）；
- 生成空位按高度分段的空态比例。
"""
import os
import sys

import numpy as np
from scipy import constants

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage.bands import top_boundary_potential  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.rates.generation import effective_activation_energy  # noqa: E402

nm, q = constants.nano, constants.elementary_charge
directory, seed = sys.argv[1], int(sys.argv[2])
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
device, tddb, sim = config.device, config.tddb, config.simulation
material = device.stack.single_material
Z = device.stack.layers[0].defects.charge_state
solver = PoissonSolver(device, sim.grid)
phi_top = top_boundary_potential(device, sim.voltage)
kT = constants.Boltzmann * sim.temperature
probe = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5])
nodes = [int(np.argmin(np.abs(solver.z - z * nm))) for z in probe]
print("seed {} | U + V_bi = {:.3f} V | z [nm] = {}".format(seed, phi_top, " ".join("{:.2f}".format(v) for v in probe)))

for n in (int(v) for v in sys.argv[3:]):
    snap = np.load(os.path.join(ROOT, directory, "snap_s{}_n{:04d}.npz".format(seed, n)))
    positions, p, n_initial = snap["positions"], snap["occupancy"], int(snap["n_initial"])
    charge = q * (Z - p)
    phi = solver.solve(solver.deposit(positions, charge), phi_top)
    field = np.stack(solver.electric_field(phi), axis=-1)
    bond = solver.gaussian_average(field, tddb.field_averaging_width)
    barrier = effective_activation_energy(tddb.activation_energy, tddb.dipole_moment, material.kappa,
                                          np.linalg.norm(bond, axis=-1))
    p_bb = np.exp(-np.maximum(barrier, 0.0) / kT)
    ez = solver.lateral_average(field[..., 2])  # E_z > 0 指向阳极；电子受力指向阴极 → 外加偏压下 E_z < 0
    bond_mag = solver.lateral_average(np.linalg.norm(bond, axis=-1))
    pbb = solver.lateral_average(p_bb)
    net = float(charge.sum())
    zbar = float(np.sum(charge * positions[:, 2]) / net) if net > 0 else float("nan")
    sheet = net / device.area / (material.kappa * constants.epsilon_0)
    s2 = (phi_top - sheet * zbar) / device.Lz
    generated = slice(n_initial, None)
    zg, empty = positions[generated, 2] / nm, 1.0 - p[generated]
    bins = [(0, 1), (1, 2), (2, 8)]
    print("n={:4d} t={:.4g} s | net +{:.1f} e at mean z {:.2f} nm | 1D sheet: cathode side {:.2f} V/nm, bulk {:.3f} V/nm"
          .format(n, float(snap["time"]), net / q, zbar / nm, (s2 + sheet) * nm, s2 * nm))
    print("   -E_z  [V/nm] " + " ".join("{:7.3f}".format(-ez[k] * nm) for k in nodes))
    print("   |Ē|  [V/nm] " + " ".join("{:7.3f}".format(bond_mag[k] * nm) for k in nodes))
    print("   P_bb        " + " ".join("{:7.1e}".format(pbb[k]) for k in nodes))
    print("   generated vacancies empty fraction by z: " + ", ".join(
        "{}–{} nm {:.2f} ({} of them)".format(a, b, empty[(zg >= a) & (zg < b)].mean() if np.any((zg >= a) & (zg < b))
                                              else float("nan"), int(np.sum((zg >= a) & (zg < b)))) for a, b in bins),
          flush=True)
