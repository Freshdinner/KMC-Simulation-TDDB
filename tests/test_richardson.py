"""R1 热电子发射子部分与 Richardson-Schottky 解析式的独立对比。

docs/rate_equations.md 要求把 R1 按势垒顶切分出的"热电子发射"部分与
J_TE = A*·T²·... 做独立对比，作为该部分实现是否正确的检验。两式的势垒
模型一致（零场 E_B、不含镜像力降低），偏压依赖也一致；按规格，前置因子
的质量选择不同——R1 用发射极自由电子质量 m_e，A* 用介质导带底有效质量
m_cb——所以判据是"两者之比恒等于 m_e/m_cb，且不随温度、偏压、缺陷浓度
漂移"。
"""

import numpy as np
import pytest
from scipy import constants

from conftest import make_config, make_defect_config, make_device, make_material
from kmc_leakage.bands import conduction_band_edge, top_boundary_potential
from kmc_leakage.config import (
    DeviceConfig,
    DielectricStack,
    ElectrodeMaterial,
    GridConfig,
    Interface,
    Layer,
)
from kmc_leakage.electrostatics import PoissonSolver
from kmc_leakage.kmc import (
    KMCSimulation,
    _band_on_path_steps,
    _interface_step_nodes,
    _R1_PATH_POINTS,
    r1_electrode_current,
)
from kmc_leakage.rates import (
    barrier_maximum,
    r1_electrode_electrode,
    richardson_schottky_current_density,
)

EV = constants.electron_volt
M_E = constants.electron_mass


def _analytic(device, voltage, temperature):
    return richardson_schottky_current_density(
        temperature, voltage, device.barrier(device.negative), device.stack.single_material.m_cb
    )


def _mass_corrected_deviation(result, device):
    """把规格指定的前置因子质量差折算掉之后的相对偏差。"""
    j_kmc = result.by_mechanism["thermionic"] / result.area
    j_analytic = _analytic(device, result.voltage, result.temperature)
    mass_ratio = device.stack.single_material.m_cb / device.negative.effective_mass
    return j_kmc * mass_ratio / j_analytic - 1.0


def test_kmc_thermionic_matches_richardson_schottky():
    """KMC 侧算出的热发射电流密度与解析式一致（折算质量比之后）。"""
    device = make_device()
    for voltage in (2.0, 3.0, 4.0):
        for temperature in (273.15, 398.15):
            result = KMCSimulation(
                make_config(voltage=voltage, temperature=temperature, n_events=200)
            ).run()
            assert abs(_mass_corrected_deviation(result, device)) < 1.5e-2


def test_deviation_does_not_grow_as_defect_concentration_drops():
    """docs 的判据：偏差随缺陷浓度趋于零而收敛到零。

    R1 在**外加偏压剖面**（零空间电荷的理想梯形势垒）上求值一次，按构造就与
    缺陷实现无关，所以这条判据是严格满足的——三个浓度下的热发射电流密度逐位
    相同，剩下的偏差只有能量积分的离散误差（<1%），与浓度无关。

    这里断言的是"严格相等"而不是"不变差"：R1 与缺陷解耦是一个实现决定，这条
    测试就是钉住它。空间电荷若重新进入 R1 的势垒剖面，这里会立刻红。
    """
    device = make_device()
    densities = []
    deviations = []
    for concentration in (3.0e17, 3.0e18, 3.0e19):
        result = KMCSimulation(
            make_config(
                voltage=3.0,
                n_events=200,
                defects=make_defect_config(concentration_cm3=concentration),
            )
        ).run()
        densities.append(result.by_mechanism["thermionic"] / result.area)
        deviations.append(abs(_mass_corrected_deviation(result, device)))
    assert max(deviations) < 1.5e-2
    assert densities[0] == densities[1] == densities[2]


def test_ratio_is_constant_across_operating_points():
    """比值必须是常数（= m_e/m_cb），不随温度、偏压漂移。"""
    device = make_device()
    ratios = []
    for voltage in (1.0, 2.5, 4.0):
        for temperature in (273.15, 323.15, 398.15):
            result = KMCSimulation(
                make_config(voltage=voltage, temperature=temperature, n_events=100)
            ).run()
            j_kmc = result.by_mechanism["thermionic"] / result.area
            ratios.append(j_kmc / _analytic(device, voltage, temperature))
    expected = device.negative.effective_mass / device.stack.single_material.m_cb
    assert np.allclose(ratios, expected, rtol=1.5e-2)
    assert np.std(ratios) / np.mean(ratios) < 5e-3  # 漂移远小于偏差本身


def test_barrier_bump_suppresses_thermionic_part():
    """势垒顶被空间电荷推进体内时，阈值切分必须跟着变，热发射部分下降。

    这条确认 R1 的阈值是按当前势垒剖面动态取的，而不是写死成 E_B。要让
    势垒顶离开注入界面，体内抬升的斜率必须压过外加偏压造成的倾斜，所以
    这里用一个小偏压配一个 50 meV 的鼓包。
    """
    device = make_device()
    E_B = device.barrier(device.negative)
    voltage = 0.1
    z = np.linspace(0.0, device.Lz, 129)
    flat = E_B - constants.elementary_charge * voltage * z / device.Lz
    bump = flat + 0.05 * EV * np.sin(np.pi * z / device.Lz)
    assert barrier_maximum(flat) == pytest.approx(E_B)  # 平直时势垒顶就在界面
    assert barrier_maximum(bump) > barrier_maximum(flat)  # 鼓包把势垒顶推进体内

    args = (device.area, voltage, 298.15, M_E, device.stack.single_material.m_di, -device.negative.fermi_energy)
    _, thermionic_flat = r1_electrode_electrode(z, flat, *args)
    _, thermionic_bump = r1_electrode_electrode(z, bump, *args)
    kT = constants.Boltzmann * 298.15
    expected_ratio = np.exp(-(barrier_maximum(bump) - barrier_maximum(flat)) / kT)
    assert thermionic_bump / thermionic_flat == pytest.approx(expected_ratio, rel=5e-2)
    assert thermionic_bump < thermionic_flat


# --------------------------------------------------------------------------
# 叠层：以 E_max 为参考的 R1 供给函数导出式（ADR-012；单层比值测试见上，退化用例）
# --------------------------------------------------------------------------
def _zaz_device():
    """ZrO₂/Al₂O₃/ZrO₂ 对称叠层（4/1/4 nm，亲和能对齐带阶）。"""
    zro2 = make_material("ZrO2")
    al2o3 = make_material(
        "Al2O3", kappa=9.0, eps_opt=3.0, band_gap=6.4 * EV, electron_affinity=1.4 * EV,
        m_cb=0.3 * M_E, m_vb=0.4 * M_E, m_di=0.3 * M_E, phonon_energy=0.09 * EV,
    )
    stack = DielectricStack(
        materials={"ZrO2": zro2, "Al2O3": al2o3},
        layers=(Layer("L0", "ZrO2", 4e-9), Layer("L1", "Al2O3", 1e-9), Layer("L2", "ZrO2", 4e-9)),
        interfaces=(Interface(), Interface()),
    )
    electrode = ElectrodeMaterial("TiN", 4.6 * EV, M_E, 5.0 * EV)
    return DeviceConfig(50e-9, 50e-9, electrode, electrode, stack, barrier_height=1.74 * EV)


def test_stack_thermionic_referenced_to_barrier_top():
    """叠层里 R1 热发射子部分 = 以 E_max（分段势垒最高点，落在 Al₂O₃）为参考的
    R1 供给函数玻尔兹曼导出式（ADR-012）。

    阈值以上 P=1，供给函数在玻尔兹曼尾里 S(E)≈e^{-E/kT}(1−e^{-eU/kT})，故
    R1_TE → [A·m_e·(kT)²/(2π²ℏ³)]·e^{-E_max/kT}·(1−e^{-eU/kT})。E_max 取分段势垒
    的真实台阶顶（叠层里在 Al₂O₃ 内），单层退回 E_B 即上面保留的比值测试口径。
    """
    device = _zaz_device()
    solver = PoissonSolver(device, GridConfig(8, 8, 36))
    for voltage, temperature in ((2.0, 298.15), (3.0, 398.15)):
        phi_bar = solver.lateral_average(
            solver.solve(np.zeros(solver.grid.shape), top_boundary_potential(device, voltage))
        )
        z_nodes = solver.z
        tunnel, thermionic = r1_electrode_current(device, voltage, temperature, z_nodes, phi_bar)

        # E_max：分段势垒（真实台阶）的最高点，与 r1 内部 barrier_maximum 同口径
        band_nodes = conduction_band_edge(device, z_nodes, phi_bar)
        iface_idx, delta_Ec = _interface_step_nodes(device, z_nodes)
        z = np.linspace(0.0, device.Lz, _R1_PATH_POINTS)
        barrier = _band_on_path_steps(z, z_nodes, band_nodes, iface_idx, delta_Ec)
        e_max = barrier_maximum(barrier)
        assert e_max / EV > 1.74  # 势垒顶抬到了 Al₂O₃ 内（比负极 E_B 高）

        kT = constants.Boltzmann * temperature
        prefactor = device.area * M_E * kT ** 2 / (2.0 * np.pi ** 2 * constants.hbar ** 3)
        eU = constants.elementary_charge * voltage
        expected = prefactor * np.exp(-e_max / kT) * (1.0 - np.exp(-eU / kT))
        # thermionic 是电流 q·R1_TE；换回速率与解析式比
        assert thermionic / constants.elementary_charge == pytest.approx(expected, rel=2e-2)
