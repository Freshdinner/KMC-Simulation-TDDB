"""孤立中性源陷阱（p = 1、Z = +1，不带电）在均匀外场里：源陷阱积分 vs scipy 自适应积分的精确值。

电势严格线性，前驱能级 E(z) = E_C(z_i) − e·F·(z − z_i) − E_t1，P_bb 是常数；被积函数只依赖 (r, u)：
G = N_T·2π·∫∫ r²·ν·e^{−α r}·Θ(E(z_i + r·u) − E_i)·P_bb du dr，窗口 z ∈ [standoff, Lz − standoff] 截掉一部分。
usage: isolated_reference.py
"""
import time
from dataclasses import replace

import numpy as np
from scipy import constants, integrate

from kmc_leakage import tddb as T
from kmc_leakage.bands import top_boundary_potential
from kmc_leakage.defects import DefectEnsemble, _global_standoff, localization_radius
from kmc_leakage.io import load_config
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.rates import bond_break_probability

EV, Q, KB = constants.electron_volt, constants.elementary_charge, constants.Boltzmann
config = load_config("examples/zro2_tddb.yaml")
device, tddb = config.device, config.tddb
material = device.stack.single_material
temperature = config.simulation.temperature
standoff = _global_standoff(device)
DEFAULT = dict(_TRAP_RADIAL_STEP=T._TRAP_RADIAL_STEP, _TRAP_POLAR_BINS=T._TRAP_POLAR_BINS,
               _TRAP_AZIMUTH_BINS=T._TRAP_AZIMUTH_BINS)


def reference(sim, z_i, energy_i, r_D_i, nu):
    phi_top = top_boundary_potential(device, sim.sim.voltage)
    F = phi_top / device.Lz
    E_C_i = float(device.band_reference_profile(np.array([z_i]))[0]) - Q * phi_top * z_i / device.Lz
    alpha = 1.0 / r_D_i + 1.0 / localization_radius(tddb.precursor_ionization_energy, material.m_di)
    p_bb = float(bond_break_probability(tddb.activation_energy, tddb.dipole_moment, material.kappa, np.array(F), temperature))
    kT = KB * temperature

    def radial(u):
        lo, hi = standoff, 6e-9
        if u < 0:
            hi = min(hi, (z_i - standoff) / -u)
        elif u > 0:
            hi = min(hi, (device.Lz - standoff - z_i) / u)
        if hi <= lo:
            return 0.0
        delta0 = E_C_i - tddb.precursor_ionization_energy - energy_i  # r = 0 处的 ΔE
        f = lambda r: r * r * np.exp(-alpha * r - max(0.0, delta0 - Q * F * r * u) / kT)
        points = []
        if u > 0 and delta0 > 0:
            kink = delta0 / (Q * F * u)
            if lo < kink < hi:
                points = [kink]
        return integrate.quad(f, lo, hi, points=points or None, epsabs=0, epsrel=1e-12, limit=200)[0]

    breaks = [(standoff - z_i) / standoff, -1e-12, 1e-12]
    value = integrate.quad(radial, -1.0, 1.0, points=[b for b in breaks if -1 < b < 1], epsabs=0, epsrel=1e-10, limit=400)[0]
    return tddb.precursor_concentration * 2.0 * np.pi * nu * p_bb * value


for z_i, ED in ((4.0e-9, 1.15), (4.0e-9, 1.45), (0.8e-9, 1.15), (0.8e-9, 1.45), (0.62e-9, 0.9)):
    defects = DefectEnsemble.uniform(np.array([[10e-9, 10e-9, z_i]]), np.array([ED * EV]), material,
                                     band_reference=float(device.layer_band_references()[0]))
    sim = KMCSimulation(config, defects=defects)
    p = np.ones(1)
    rho = sim.solver.deposit(defects.positions, Q * (defects.charge_state - p))
    phi = sim.solver.solve(rho, top_boundary_potential(device, sim.sim.voltage))
    energy_i = sim._trap_state(p)[0][0]
    ref = reference(sim, z_i, energy_i, defects.r_D[0], defects.attempt_frequency[0])
    out = []
    for name, override in (("default", {}), ("dr/2", dict(_TRAP_RADIAL_STEP=0.02e-9)), ("polar48", dict(_TRAP_POLAR_BINS=48)),
                           ("az48", dict(_TRAP_AZIMUTH_BINS=48))):
        for key, value in {**DEFAULT, **override}.items():
            setattr(T, key, value)
        rate = T.TrapCaptureGeneration(sim, p, tddb, phi, None).rates[0]
        out.append("{} {:+.2e}".format(name, rate / ref - 1.0))
    for key, value in DEFAULT.items():
        setattr(T, key, value)
    print("z_i={:.2f} nm ED={:.2f} eV  ref G={:.6e} /s  rel. error: {}".format(z_i * 1e9, ED, ref, ", ".join(out)), flush=True)
