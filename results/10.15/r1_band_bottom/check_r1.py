# -*- coding: utf-8 -*-
"""ADR-061 的依据：R1 的能量积分从电极导带底积起（完整的 Tsu–Esaki 积分）之后，平面 R1 与间隙隧穿的变化。

0. 能量网格：费米能级以下按与 [0, 势垒顶] 相同的间距取点，每段 128 / 256 点时与精细网格的相对差（线性势垒，均匀场 3.6 V / 8 nm）；
1. 默认器件（examples/zro2_tddb.yaml，50×50×8 nm）上的平面 R1：2.0 / 3.6 V、25 °C，下限取导带底与取费米能级的直接/FN 部分；
2. 与 results/10.15/hotspot/check_hotspot.py 第 2 节相同的体系（20×20×8 nm、网格 80³、3.6 V / 25 °C、k_TH = 2 W/(m·K)、
   r_c = 0.3 nm）：一列接负极的空位顶端离正极 g 时的间隙电流、放热与热迭代后的温度，两种下限对照。

运行：PYTHONPATH=src PYTHONIOENCODING=utf-8 python results/10.15/r1_band_bottom/check_r1.py
"""

import sys
from dataclasses import replace

import numpy as np
from scipy import constants
from scipy.integrate import trapezoid

sys.path.insert(0, "tests")
from conftest import make_config, make_defect_config, make_device  # noqa: E402

from kmc_leakage import tddb as T  # noqa: E402
from kmc_leakage.bands import r1_band_bottom, top_boundary_potential  # noqa: E402
from kmc_leakage.config import GridConfig, TDDBConfig  # noqa: E402
from kmc_leakage.defects import DefectEnsemble, append_defects  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import _R1_ENERGY_POINTS, _R1_PATH_POINTS, _band_on_path_steps, _interface_step_nodes  # noqa: E402
from kmc_leakage.rates import r1_electrode_electrode, supply_function, wkb_transmission  # noqa: E402

EV = constants.electron_volt
Q = constants.elementary_charge
NM = constants.nano
VOLTAGE = 3.6

print("== 0. 能量网格（势垒 1.74 eV，m_di = 1.2 m_e，下限 −5 eV）==")
m, m_di, bottom = constants.electron_mass, 1.2 * constants.electron_mass, -5.0 * EV
for label, d in (("8 nm 平面", 8.0), ("间隙 2 nm", 2.0), ("间隙 1 nm", 1.0), ("间隙 0.5 nm", 0.5), ("间隙 0.3 nm", 0.3)):
    z = np.linspace(0.0, d * NM, 257)
    band = 1.74 * EV - EV * VOLTAGE * z / (8.0 * NM)
    for temperature in (298.15, 2000.0):
        kT = constants.Boltzmann * temperature
        prefactor = m * kT / (2.0 * np.pi ** 2 * constants.hbar ** 3)

        def integral(lo, hi, n):
            e = np.linspace(lo, hi, n)
            return prefactor * trapezoid(wkb_transmission(z, band, e, m_di) * supply_function(e, VOLTAGE * d / 8.0, temperature), e)

        fine = integral(bottom, 0.0, 200001) + integral(0.0, band.max(), 20001)
        errors = [r1_electrode_electrode(z, band, 1.0, VOLTAGE * d / 8.0, temperature, m, m_di, bottom, n_points=n)[0] / fine - 1.0
                  for n in (128, 256)]
        fermi = integral(0.0, band.max(), 20001)
        print("  {}，{:.0f} K：完整积分是下限取费米能级的 {:.3g} 倍；每段 128 / 256 点的相对差 {:.1e} / {:.1e}".format(
            label, temperature, fine / fermi, *errors))

print("\n== 1. 默认器件的平面 R1（下限：导带底 vs 费米能级）==")
config = load_config("examples/zro2_tddb.yaml")
device = config.device
solver = T.PoissonSolver(device, config.simulation.grid)
per_cm2 = constants.centi ** 2 / device.area
for voltage in (2.0, 3.6):
    phi_bar = solver.lateral_average(solver.solve(np.zeros(solver.grid.shape), top_boundary_potential(device, voltage)))
    z = np.linspace(0.0, device.Lz, _R1_PATH_POINTS)
    iface_idx, delta_Ec = _interface_step_nodes(device, solver.z)
    band = _band_on_path_steps(z, solver.z, T.conduction_band_edge(device, solver.z, phi_bar), iface_idx, delta_Ec)
    args = (z, band, device.area, voltage, config.simulation.temperature, device.negative.effective_mass,
            device.stack.m_di_at(z))
    full, thermionic = r1_electrode_electrode(*args, r1_band_bottom(device, voltage), n_points=_R1_ENERGY_POINTS)
    fermi, _ = r1_electrode_electrode(*args, 0.0, n_points=_R1_ENERGY_POINTS)
    print("  U = {} V：直接/FN 下限取导带底 {:.3e} A/cm²，取费米能级 {:.3e} A/cm²（{:.2f} 倍）；热发射 {:.3e} A/cm²".format(
        voltage, Q * full * per_cm2, Q * fermi * per_cm2, full / fermi, Q * thermionic * per_cm2))


def make(concentration, r_c, thermal=2.0):
    device = make_device(defects=make_defect_config(concentration_cm3=concentration))
    tddb = TDDBConfig(
        activation_energy=1.3 * EV, precursor_ionization_energy=1.15 * EV, precursor_relaxation_energy=1.0 * EV,
        dipole_moment=1.0 * Q * 1e-10, field_averaging_width=0.5e-9, max_generated_vacancies=4, stop_time=1e4,
        percolation_distance=r_c, thermal_conductivity=thermal,
    )
    config = replace(make_config(voltage=VOLTAGE, seed=3, device=device), tddb=tddb)
    return replace(config, simulation=replace(config.simulation, grid=GridConfig(80, 80, 80)))


print("\n== 2. 一列接负极的空位，顶端离正极 g（20×20×8 nm，网格 80³）==")
config = make(3.0e18, 0.3e-9)
device = config.device
material = device.stack.single_material
background = T.generate_defects(device, np.random.default_rng(2))


def column(top):
    """(10.03, 10.07) nm 处从 z = 0.2 nm 到 `top` 的一列空位（间距约 0.28 nm），加上横向离它 > 1 nm 的 3×10¹⁸ cm⁻³ 初始缺陷。"""
    z = np.arange(0.2e-9, top + 1e-15, 0.28e-9)
    z = z * (top / z[-1])
    pos = np.column_stack([np.full(z.size, 10.03e-9), np.full(z.size, 10.07e-9), z])
    chain = DefectEnsemble.uniform(pos, np.full(z.size, 1.15 * EV), material,
                                   band_reference=float(device.layer_band_references()[0]), charge_state=1.0)
    keep = np.hypot(background.positions[:, 0] - pos[0, 0], background.positions[:, 1] - pos[0, 1]) > 1.0e-9
    return append_defects(T._take(background, np.flatnonzero(keep)), chain)


def steady(defects):
    solver = T.PoissonSolver(config.device, config.simulation.grid)
    coupling = T.IncrementalCoupling.build(solver, defects.positions, defects.field_side)
    layout = T.conductor_layout(defects, coupling, solver, VOLTAGE, config.tddb.percolation_distance)
    return T._steady_state(config, layout, defects, coupling, solver, None, np.full(len(defects), 298.15))


band_bottom = T.r1_band_bottom
for gap in (0.32, 0.4, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
    defects = column(device.Lz - gap * NM)
    T.r1_band_bottom = band_bottom
    step = steady(defects)
    T.r1_band_bottom = lambda device, voltage: 0.0  # 原先的下限：负极费米能级
    old = steady(defects)
    T.r1_band_bottom = band_bottom
    print("  g = {:.2f} nm：间隙电流 {:.2e} A（下限取费米能级时 {:.2e} A，{:.0f} 倍），放热 {:.2e} W；缺陷处最高 {:.2f} K、"
          "温度场最高 {:.2f} K，热迭代 {} 次".format(
              gap, step.gap_current, old.gap_current, step.gap_current / old.gap_current, step.gap_current * VOLTAGE,
              step.temperature.max(), step.field.max(), step.iterations))
