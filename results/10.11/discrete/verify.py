"""ADR-052 数值验证：离散前驱位点的生成速率与引擎、直接求和对拍，给出实际误差（单元测试只断言上界）。

A. 20×20 nm、96 个初始缺陷（带固定电荷）、3 V：280 个前驱位点的电极项与引擎在该点放前驱参数探针时的两侧 R2 + R4
   （前驱声子项数设成引擎的 12 项）；陷阱项与对全部已占据陷阱的直接求和（不截断）；
B. examples/zro2_tddb.yaml 默认工况（50×50 nm、3.6 V，seed 1）t = 0：前驱位点数与 E_A 统计、ΣG 的电极项/陷阱项份额、
   生成率一步的耗时，以及按 z 分段的 ΣG。
usage: verify.py
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
sys.path.insert(0, os.path.join(ROOT, "tests"))
from conftest import make_config, make_defect_config, make_device  # noqa: E402
from kmc_leakage import tddb as tddb_module  # noqa: E402
from kmc_leakage.config import TDDBConfig  # noqa: E402
from kmc_leakage.defects import DefectEnsemble, append_defects, generate_defects, localization_radius, minimum_image_delta  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import _B_R2A, _B_R2C, _B_R4A, _B_R4C, _N_PHONON, KMCSimulation  # noqa: E402
from kmc_leakage.mean_field import solve_mean_field  # noqa: E402
from kmc_leakage.rates import trap_precursor_capture_rate  # noqa: E402
from kmc_leakage.tddb import IncrementalCoupling, Precursors, generation_rates, place_precursors  # noqa: E402

EV, Q, NM = constants.electron_volt, constants.elementary_charge, constants.nano

# ---- A ----
tddb_module._PRECURSOR_N_PHONON = _N_PHONON
tddb = TDDBConfig(
    activation_energy=4.0 * EV, activation_energy_sigma=0.1 * EV, precursor_ionization_energy=1.15 * EV,
    precursor_relaxation_energy=1.0 * EV, dipole_moment=7.1 * Q * constants.angstrom, field_averaging_width=0.5e-9,
    precursor_concentration=1.0e26, max_generated_vacancies=4, stop_time=1.0e4, breakdown_current_density=1.0e8,
)
device = make_device(defects=make_defect_config(concentration_cm3=3.0e19))
config = replace(make_config(voltage=3.0, seed=3, device=device), tddb=tddb)
material = device.stack.single_material
defects = KMCSimulation(config).defects
fixed = np.where(np.arange(len(defects)) % 3 == 0, -Q, 0.0)
inner = KMCSimulation(config, defects=replace(defects, fixed_charge=fixed))
p = solve_mean_field(inner).occupancy
precursors = place_precursors(device, tddb, inner.defects.positions, np.random.default_rng(5))
rng = np.random.default_rng(9)
sources = np.flatnonzero(p >= 0.5)
near = []
while len(near) < 60:  # 再加 60 个贴着已占据缺陷（0.5–0.8 nm）的位点，陷阱项在这些位点上显著
    d = rng.normal(size=3)
    point = inner.defects.positions[rng.choice(sources)] + rng.uniform(0.5, 0.8) * NM * d / np.linalg.norm(d)
    point[:2] = np.mod(point[:2], [device.Lx, device.Ly])
    r = np.linalg.norm(minimum_image_delta(inner.defects.positions - point, device.Lx, device.Ly), axis=1)
    if 0.5 * NM <= point[2] <= device.Lz - 0.5 * NM and r.min() >= 0.5 * NM:
        near.append(point)
sites = Precursors(np.vstack([precursors.positions, near]),
                   np.concatenate([precursors.activation_energy, np.full(len(near), tddb.activation_energy)]))
rates = generation_rates(inner, p, tddb, sites)

probes = DefectEnsemble.uniform(sites.positions, np.full(len(sites), tddb.precursor_ionization_energy),
                                replace(material, huang_rhys=tddb.precursor_relaxation_energy / material.phonon_energy),
                                band_reference=float(device.layer_band_references()[0]), charge_state=0.0)
extended = KMCSimulation(config, defects=append_defects(inner.defects, probes))
occ = np.concatenate([p, np.zeros(len(sites))])
energy, field, barrier_c, barrier_a = extended._trap_state(occ)
blocks = extended._electrode_blocks(energy, field, barrier_c, barrier_a)
probe = np.arange(inner.n_defects, extended.n_defects)
capture = (blocks[_B_R2C] + blocks[_B_R2A] + blocks[_B_R4C] + blocks[_B_R4A])[probe]
rel = np.abs(rates.electrode / capture - 1.0)
print("A. {} sites ({} near occupied traps); electrode term vs engine probe R2+R4: max rel err {:.2e}, median {:.2e}; "
      "R_1e,electrode range {:.2e} – {:.2e} 1/s".format(len(sites), len(near), rel.max(), np.median(rel),
                                                       capture.min(), capture.max()))

trap_energy = inner._trap_state(p)[0]
delta = minimum_image_delta(sites.positions[:, None, :] - inner.defects.positions[None, :, :], device.Lx, device.Ly)
direct = trap_precursor_capture_rate(
    inner.defects.attempt_frequency[None, :], np.linalg.norm(delta, axis=-1), inner.defects.r_D[None, :],
    localization_radius(tddb.precursor_ionization_energy, material.m_di), trap_energy[None, :],
    energy[probe][:, None], p[None, :], config.simulation.temperature).sum(axis=1)
big = direct > 1e-6
print("   trap term vs direct sum over all occupied traps: max |diff| {:.2e} 1/s (floor {:.0e}); max rel err where > 1e-6 1/s "
      "{:.2e} ({} sites); trap term on near sites {:.2e} – {:.2e} 1/s, electrode term there {:.2e} – {:.2e} 1/s".format(
          np.abs(rates.trap - direct).max(), tddb_module._TRAP_TERM_FLOOR, np.abs(rates.trap[big] / direct[big] - 1).max(),
          int(big.sum()), direct[-len(near):].min(), direct[-len(near):].max(),
          rates.electrode[-len(near):].min(), rates.electrode[-len(near):].max()))

# ---- B ----
tddb_module._PRECURSOR_N_PHONON = 100
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
config = replace(config, simulation=replace(config.simulation, seed=1))
device, tddb = config.device, config.tddb
rng = np.random.default_rng(1)
defects = generate_defects(device, rng)
precursors = place_precursors(device, tddb, defects.positions, rng)
window = device.Lz - 2.0 * 0.5 * NM
print("B. default config seed 1: N_T·V = {:.1f}, {} sites kept ({} within 0.5 nm of the {} initial defects); "
      "E_A mean {:.4f} eV, std {:.4f} eV".format(
          tddb.precursor_concentration * device.Lx * device.Ly * window, len(precursors),
          int(round(tddb.precursor_concentration * device.Lx * device.Ly * window)) - len(precursors), len(defects),
          precursors.activation_energy.mean() / EV, precursors.activation_energy.std() / EV))
solver = PoissonSolver(device, config.simulation.grid)
coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
inner = KMCSimulation(config, defects=defects, coupling=coupling, solver=solver)
mf = solve_mean_field(inner, None)
start = time.time()
g = generation_rates(inner, mf.occupancy, tddb, precursors)
wall = time.time() - start
G = g.rates
z = precursors.positions[:, 2] / NM
print("   J0 = {:.3f} A/cm2; sum G = {:.4g} 1/s (electrode {:.4f}, trap {:.2e}); generation_rates wall {:.2f} s".format(
    mf.current_density * 1e-4, G.sum(), (g.electrode * g.bond_breaking).sum() / G.sum(),
    (g.trap * g.bond_breaking).sum() / G.sum(), wall))
for lo, hi in ((0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 7.5)):
    m = (z >= lo) & (z < hi)
    print("   z {:.1f}–{:.1f} nm: {:4d} sites, share of sum G {:.3f}, max G {:.3g} 1/s".format(
        lo, hi, int(m.sum()), G[m].sum() / G.sum(), G[m].max()))
