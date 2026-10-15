"""ADR-060 的依据：接电极导电簇在输运里是导体、间隙隧穿、局域温度与热迭代、击穿后的细丝功率。

20×20×8 nm、网格 80×80×80（hx = hy = 0.25 nm、hz = 0.1 nm），3.6 V / 25 °C，k_TH = 2 W/(m·K)，r_c = 0.3 nm：
1. 等效耦合：3×10¹⁹ cm⁻³ 的初始缺陷、r_c = 1.2 nm（两侧都有接电极的导电簇），陷阱处引擎电势与直接解泊松方程的差、成员处电势、能量守恒；
2. 一列接负极的空位（间距约 0.28 nm）顶端离正极 g 时的间隙隧穿电流、放热与热迭代后的缺陷处温度；
3. 这一列接通正极（贯穿通路）后：细丝功率 G₀·V² 平分到通路的空位上，热迭代后的温度与生成率；
4. R1 的能量积分下限取发射极费米能级（项目约定）与取发射极导带底（完整的 Tsu–Esaki 积分）在平板间隙上的差别。

运行：PYTHONPATH=src python results/10.15/hotspot/check_hotspot.py
"""

import sys
from dataclasses import replace

import numpy as np
from scipy import constants
from scipy.integrate import trapezoid

sys.path.insert(0, "tests")
from conftest import make_config, make_defect_config, make_device  # noqa: E402

from kmc_leakage import tddb as T  # noqa: E402
from kmc_leakage.config import GridConfig, TDDBConfig  # noqa: E402
from kmc_leakage.defects import DefectEnsemble, append_defects  # noqa: E402
from kmc_leakage.rates import r1_electrode_electrode, supply_function, wkb_transmission  # noqa: E402

EV = constants.electron_volt
Q = constants.elementary_charge
NM = constants.nano
VOLTAGE = 3.6


def make(concentration, r_c, thermal=2.0):
    device = make_device(defects=make_defect_config(concentration_cm3=concentration))
    tddb = TDDBConfig(
        activation_energy=1.3 * EV, precursor_ionization_energy=1.15 * EV, precursor_relaxation_energy=1.0 * EV,
        dipole_moment=1.0 * Q * 1e-10, field_averaging_width=0.5e-9, max_generated_vacancies=4, stop_time=1e4,
        percolation_distance=r_c, thermal_conductivity=thermal,
    )
    config = replace(make_config(voltage=VOLTAGE, seed=3, device=device), tddb=tddb)
    return replace(config, simulation=replace(config.simulation, grid=GridConfig(80, 80, 80)))


def setup(config, defects):
    solver = T.PoissonSolver(config.device, config.simulation.grid)
    coupling = T.IncrementalCoupling.build(solver, defects.positions, defects.field_side)
    layout = T.conductor_layout(defects, coupling, solver, VOLTAGE, config.tddb.percolation_distance)
    return solver, coupling, layout


print("== 1. 等效耦合（3e19 cm^-3，r_c = 1.2 nm）==")
config = make(3.0e19, 1.2e-9, thermal=None)
defects = T.initial_state(config).defects
solver, coupling, layout = setup(config, defects)
temperature = np.full(len(defects), config.simulation.temperature)
transport = T.solve_transport(config, layout, defects, coupling, solver, temperature)
p = transport.result.occupancy
traps, members = layout.traps, layout.members
at = solver.interpolate(transport.phi, defects.positions)
own = p - defects.charge_state[traps] - defects.colocated_fixed_charge[traps] / Q
engine = transport.sim._electrostatics(p)[0]
heat = transport.heat
trap_current = transport.result.by_mechanism["tat"] + transport.result.by_mechanism["pf"]
print("  缺陷 {}：接负极 {}、接正极 {}、陷阱 {}".format(len(defects), np.sum(layout.sides == 0), np.sum(layout.sides == 1), traps.size))
print("  成员处电势与导体电势之差 ≤ {:.1e} V".format(np.abs(at[members] - layout.potential).max()))
print("  陷阱处：直接解泊松方程扣掉自身电荷 vs 引擎电势，差 ≤ {:.1e} V（电势最大 {:.2f} V）".format(
    np.abs(at[traps] - own * transport.sim.coupling.phi_self - engine).max(), np.abs(engine).max()))
print("  能量守恒：介质 + 两侧电极 = {:.6e} W，U·I_陷阱 = {:.6e} W，相对差 {:.1e}；其中导体成员处 {:.2e} W".format(
    heat.dielectric + heat.negative + heat.positive, VOLTAGE * trap_current,
    (heat.dielectric + heat.negative + heat.positive) / (VOLTAGE * trap_current) - 1.0, heat.conductor.sum()))

print("\n== 2. 一列接负极的空位，顶端离正极 g ==")
config = make(3.0e18, 0.3e-9)
device = config.device
material = device.stack.single_material
rng = np.random.default_rng(2)
background = T.generate_defects(device, rng)


def column(top):
    """(10.03, 10.07) nm 处从 z = 0.2 nm 到 `top` 的一列空位（间距约 0.28 nm），加上横向离它 > 1 nm 的 3×10¹⁸ cm⁻³ 初始缺陷。"""
    z = np.arange(0.2e-9, top + 1e-15, 0.28e-9)
    z = z * (top / z[-1])
    pos = np.column_stack([np.full(z.size, 10.03e-9), np.full(z.size, 10.07e-9), z])
    chain = DefectEnsemble.uniform(pos, np.full(z.size, 1.15 * EV), material,
                                   band_reference=float(device.layer_band_references()[0]), charge_state=1.0)
    keep = np.hypot(background.positions[:, 0] - pos[0, 0], background.positions[:, 1] - pos[0, 1]) > 1.0e-9
    return append_defects(T._take(background, np.flatnonzero(keep)), chain), pos


for gap in (0.4, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
    defects, pos = column(device.Lz - gap * NM)
    solver, coupling, layout = setup(config, defects)
    step = T._steady_state(config, layout, defects, coupling, solver, None, np.full(len(defects), 298.15))
    line = np.column_stack([np.full(5, pos[-1, 0]), np.full(5, pos[-1, 1]), np.linspace(pos[-1, 2], device.Lz, 5)])
    print("  g = {:.1f} nm：导体 {} 个空位，间隙电流 {:.2e} A、放热 {:.2e} W；缺陷处最高 {:.2f} K、温度场最高 {:.2f} K，"
          "热迭代 {} 次；顶端到正极 1/4、1/2、3/4 处电势 {}".format(
              gap, layout.members.size, step.gap_current, step.gap_current * VOLTAGE, step.temperature.max(),
              step.field.max(), step.iterations, np.round(solver.interpolate(step.transport.phi, line)[1:4], 2)))

print("\n== 3. 贯穿通路（顶端离正极 0.2 nm < r_c）==")
defects, _ = column(device.Lz - 0.2 * NM)
solver, coupling, layout = setup(config, defects)
step = T._steady_state(config, layout, defects, coupling, solver, None, np.full(len(defects), 298.15))
spanning = layout.sides == 2
print("  通路 {} 个空位，细丝 {:.3e} A、功率 {:.3e} W；热迭代 {} 次、收敛 {}".format(
    np.sum(spanning), step.filament_current, step.heat.sum(), step.iterations, step.converged))
print("  温度场最高 {:.0f} K，通路空位 {:.0f}–{:.0f} K，陷阱最高 {:.0f} K".format(
    step.field.max(), step.temperature[spanning].min(), step.temperature[spanning].max(), step.temperature[layout.traps].max()))
precursors = T.place_precursors(device, defects.positions, np.random.default_rng(5))
rates = T.generation_rates(step.transport, config.tddb, precursors, step.field)
hot = solver.interpolate(step.field, precursors.positions) > 400.0
print("  生成率 ΣG = {:.2e} 1/s（温度 > 400 K 的位点 {} 个，占 ΣG 的 {:.3f}），P_bb 饱和的位点 {} 个".format(
    rates.rates.sum(), np.sum(hot), rates.rates[hot].sum() / rates.rates.sum(), np.sum(rates.bond_breaking >= 1.0)))

print("\n== 4. R1 能量积分的下限（平板间隙，势垒 1.74 eV，m_di = 1.2 m_e，面积 (0.26 nm)²）==")
m, m_di = constants.electron_mass, 1.2 * constants.electron_mass
temperature = 298.15
kT = constants.Boltzmann * temperature
area = (0.26 * NM) ** 2
for gap in (0.5, 1.0, 2.0):
    z = np.linspace(0.0, gap * NM, 129)
    band = 1.74 * EV - EV * VOLTAGE * z / z[-1]
    r1 = sum(r1_electrode_electrode(z, band, area, VOLTAGE, temperature, m, m_di))
    prefactor = area * m * kT / (2.0 * np.pi ** 2 * constants.hbar ** 3)
    energies = np.linspace(-5.0 * EV, 1.74 * EV + 30 * kT, 20001)  # 下限取 TiN 导带底（费米能级下 5 eV）
    full = prefactor * trapezoid(wkb_transmission(z, band, energies, m_di) * supply_function(energies, VOLTAGE, temperature), energies)
    print("  g = {} nm：下限取费米能级 {:.2e} A，取导带底 {:.2e} A，相差 {:.0f} 倍".format(gap, Q * r1, Q * full, full / r1))
