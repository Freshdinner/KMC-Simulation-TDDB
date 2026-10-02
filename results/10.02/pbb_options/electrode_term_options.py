"""Phase 4 问题 2 的 (a)(b) 评估：生成事件表在两种 P_bb 口径下随生成网格加密的收敛，以及新空位落点的分布。

usage: electrode_term_options.py STATE [NXY ...]

状态同 ADR-042（`../adr042/common.py`），等温。两种口径：
- local（现状，式 T2 原文）：P_bb 取含点电荷近场的局域场 |E(x)|；
- macro（方案 (a)）：P_bb 取横向平均电势的 z 向场 |dφ̄/dz|(z)；R_1e 不变（前驱能级仍取局域电势，
  R4 的 c_MP 仍取局域场）。陷阱项同样只换 P_bb 里的场。

单元电极项在横向 NXY 格（默认 16 32 48 64 96 128）、z 70 格上按现行的单元中心规则求值；
最后一档另做 z 140 格。现行规则在单元无限细时的极限就是方案 (b)（对单元内可放置区域的积分），
所以加密序列的末端就是 (b) 的值。陷阱项按源陷阱积分（ADR-042），与网格无关，每种口径算一次。

输出：每档网格两种口径的 ΣG_单元；ΣG_陷阱；ΣG_单元 沿 z 的分布（z < 1、1–2、> 2 nm 的份额与加权中位数）；
按事件表抽 4000 次的新空位 z 中位数、90% 分位与 z > 2 nm 的比例。
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adr042"))

import numpy as np
from scipy import constants

from common import load_state
from kmc_leakage.bands import anode_fermi_level, electrode_band_bottom
from kmc_leakage.defects import _global_standoff
from kmc_leakage.rates.generation import CaptureElectrode, bond_break_probability, single_electron_capture_rate
from kmc_leakage import tddb as tddb_module
from kmc_leakage.tddb import (
    TrapCaptureGeneration,
    _PRECURSOR_N_PHONON,
    _TRAP_PATH_POINTS,
    _band_on_path_steps,
    _interface_step_nodes,
    admissible_cells,
    conduction_band_edge,
    generation_cells,
    top_boundary_potential,
)

name = sys.argv[1]
grids = [int(a) for a in sys.argv[2:]] or [16, 32, 48, 64, 96, 128]
config, inner, p, mf_wall = load_state(name)
device, tddb = config.device, config.tddb
solver, defects = inner.solver, inner.defects
material = device.stack.single_material
q = constants.elementary_charge
T = inner.sim.temperature
kT = constants.Boltzmann * T
nm = constants.nano
print("{}: N={} defects, empty (p<0.5) {}, mean-field {:.1f}s".format(
    name, inner.n_defects, int(np.sum(p < 0.5)), mf_wall), flush=True)

rho = solver.deposit(defects.positions, q * (defects.charge_state - p))
phi = solver.solve(rho, top_boundary_potential(device, inner.sim.voltage))
phi_bar = solver.lateral_average(phi)
ez_bar = -np.gradient(phi_bar, solver.hz)  # 宏观场 −dφ̄/dz，z 节点上
band = conduction_band_edge(device, solver.z, phi_bar)
iface_idx, delta_Ec = _interface_step_nodes(device, solver.z)
applied = inner.sim.voltage / device.Lz
for z in (0.6, 1.0, 1.5, 2.0, 3.0, 4.0):
    e = abs(np.interp(z * nm, solver.z, ez_bar))
    print("  macro field z={:.1f} nm: |E|/E_applied={:.3f}, P_bb={:.3e}".format(
        z, e / applied, float(bond_break_probability(tddb.activation_energy, tddb.dipole_moment, material.kappa, e, T))),
        flush=True)


def electrode_density(centers, z_values, layer):
    """各点 N_T·R_1e,电极 与局域场、宏观场的大小。z_values 是各层中心，layer 是点所在层号。"""
    t = np.linspace(0.0, 1.0, _TRAP_PATH_POINTS)
    zl = z_values[:, None]
    sides = []
    for path, electrode, fermi in (
        (zl * t[None, :], device.cathode, 0.0),
        (zl + (device.Lz - zl) * t[None, :], device.anode, anode_fermi_level(inner.sim.voltage)),
    ):
        sides.append(CaptureElectrode(
            path=path, barrier=_band_on_path_steps(path, solver.z, band, iface_idx, delta_Ec),
            m_di_path=device.stack.m_di_at(path), fermi_level=fermi,
            band_bottom=electrode_band_bottom(fermi, electrode.fermi_energy), mass=electrode.effective_mass))
    local_field = np.empty(centers.shape[0])
    capture = np.empty(centers.shape[0])
    chunk = 4096
    for start in range(0, centers.shape[0], chunk):
        rows = slice(start, start + chunk)
        c = centers[rows]
        local_field[rows] = np.linalg.norm(solver.field_vectors_at(phi, c, np.zeros(c.shape[0], dtype=int)), axis=0)
        conduction = device.band_reference_profile(c[:, 2]) - q * solver.interpolate(phi, c)
        lay = layer[rows]
        for sub in range(0, c.shape[0], 64):
            s = slice(sub, sub + 64)
            electrodes = [e._replace(path=e.path[lay[s]], barrier=e.barrier[lay[s]], m_di_path=e.m_di_path[lay[s]])
                          for e in sides]
            capture[rows][s] = single_electron_capture_rate(
                conduction[s], local_field[rows][s], electrodes, tddb.precursor_ionization_energy,
                tddb.precursor_relaxation_energy, material.phonon_energy, material.band_gap, material.m_di, T,
                _PRECURSOR_N_PHONON)
    macro_field = np.abs(np.interp(centers[:, 2], solver.z, ez_bar))
    return tddb.precursor_concentration * capture, local_field, macro_field


def trap_term(mode):
    if mode == "local":
        return TrapCaptureGeneration(inner, p, tddb, phi, None)
    original = solver.electric_field
    zeros = np.zeros_like(phi)
    solver.electric_field = lambda _phi: (zeros, zeros, np.broadcast_to(ez_bar, phi.shape).copy())
    try:
        return TrapCaptureGeneration(inner, p, tddb, phi, None)
    finally:
        solver.electric_field = original


def z_summary(z_lo, dz, rates):
    """单元事件沿 z 的分布：z < 1、1–2、> 2 nm 的份额与加权中位数（单元内均匀）。"""
    total = rates.sum()
    if not total > 0.0:
        return "empty"
    zc = z_lo + 0.5 * dz
    share = [rates[zc < 1 * nm].sum(), rates[(zc >= 1 * nm) & (zc < 2 * nm)].sum(), rates[zc >= 2 * nm].sum()]
    order = np.argsort(zc)
    cum = np.cumsum(rates[order])
    k = int(np.searchsorted(cum, 0.5 * total))
    median = z_lo[order][k] + dz * (0.5 * total - (cum[k] - rates[order][k])) / rates[order][k]
    return "z<1 {:.3f} 1-2 {:.3f} >2 {:.2e} median {:.3f} nm".format(*(s / total for s in share), median / nm)


standoff = _global_standoff(device)
traps = {}
for mode in ("local", "macro"):
    tick = time.time()
    traps[mode] = trap_term(mode)
    print("{} trap term {:5s}: sum_G={:.4e} /s  wall={:.1f}s".format(name, mode, traps[mode].rates.sum(), time.time() - tick),
          flush=True)

schedule = [(n, 70) for n in grids] + [(grids[-1], 140)]
for nxy, nz in schedule:
    cells = generation_cells(device, (nxy, nxy, nz))
    tick = time.time()
    live = np.flatnonzero(admissible_cells(cells, defects.positions, device))
    density, local_field, macro_field = electrode_density(cells.centers[live], cells.z_centers, cells.layer[live])
    result = {}
    for mode, field in (("local", local_field), ("macro", macro_field)):
        rates = np.zeros(len(cells))
        rates[live] = cells.volume * density * bond_break_probability(
            tddb.activation_energy, tddb.dipole_moment, material.kappa, field, T)
        result[mode] = rates
    wall = time.time() - tick
    saturated = np.mean(local_field * tddb.dipole_moment * (2.0 + material.kappa) / 3.0 >= tddb.activation_energy)
    for mode, rates in result.items():
        trap = traps[mode]
        rng = np.random.default_rng(nxy * 1000 + nz)
        table = np.concatenate([rates, trap.rates])
        cum = np.cumsum(table)
        z = []
        for _ in range(4000):
            event = min(int(np.searchsorted(cum, rng.random() * cum[-1], side="right")), table.size - 1)
            if event < len(cells):
                z.append(cells.lower[event, 2] + cells.size[2] * rng.random())
            else:
                point = trap.sample(event - len(cells), rng)
                if point is not None:
                    z.append(point[2])
        z = np.array(z)
        print("{} grid {:3d}x{:3d}x{:3d} {:5s} | sum_G cells={:.4e} traps={:.4e} total={:.4e} | cells: {} | "
              "births z median={:.2f} q90={:.2f} z>2nm={:.4f} | live {:.3f} sat {:.2e} wall={:.0f}s".format(
                  name, nxy, nxy, nz, mode, rates.sum(), trap.rates.sum(), rates.sum() + trap.rates.sum(),
                  z_summary(cells.lower[:, 2], cells.size[2], rates), np.median(z) / nm, np.quantile(z, 0.9) / nm,
                  np.mean(z > 2 * nm), live.size / len(cells), saturated, wall), flush=True)
