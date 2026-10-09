"""F03 —— PF 收集概率 P_A 与带偏压势垒一致（ADR-021 F03 / ADR-009 位置感知）。

这一批盯住 Phase 5 的 F03 交付：
- P_A 的 WKB 透射在带偏压的 Al₂O₃ 势垒（矩形/梯形/三角）上算，与解析闭式一致；
- 0 ≤ P_A ≤ 1；
- 位置感知：只有与收集电极之间隔着阻挡层的陷阱用 P_A，同侧立即收集 P_A=1；
- 收集方向随偏压定（正向阳极、反偏阴极），非正向工况标"未通过验证"；
- 单层无阻挡层时 P_A≡1、R8_eff=R8，逐位退回原路径。
"""

import numpy as np
import pytest
from scipy import constants

from kmc_leakage.config import (
    Config,
    DeviceConfig,
    DielectricMaterial,
    DielectricStack,
    ElectrodeMaterial,
    GridConfig,
    Interface,
    Layer,
    SimulationConfig,
)
from kmc_leakage.defects import DefectEnsemble, localization_radius
from kmc_leakage.kmc import KMCSimulation, _blocking_layer_indices, pf_collection_transmission

EV = constants.electron_volt
M_E = constants.electron_mass
HBAR = constants.hbar
Q = constants.elementary_charge


def _material(name, kappa, eps_opt=5.6, band_gap_eV=5.4, chi_eV=2.86, m_cb=1.2, phonon_eV=0.04):
    return DielectricMaterial(
        name=name, kappa=kappa, eps_opt=eps_opt, band_gap=band_gap_eV * EV,
        electron_affinity=chi_eV * EV, m_cb=m_cb * M_E, m_vb=2.5 * M_E, m_di=m_cb * M_E,
        phonon_energy=phonon_eV * EV, huang_rhys=1.0, attempt_frequency=1.0e13,
    )


ZRO2 = _material("ZrO2", 38.0)
AL2O3 = _material("Al2O3", 9.0, eps_opt=3.0, band_gap_eV=6.4, chi_eV=1.4, m_cb=0.3, phonon_eV=0.09)
_ELECTRODE = ElectrodeMaterial("TiN", 4.6 * EV, M_E, 5.0 * EV)


def _zaz_device(barrier_height=1.74 * EV):
    stack = DielectricStack(
        materials={"ZrO2": ZRO2, "Al2O3": AL2O3},
        layers=(Layer("L0", "ZrO2", 4e-9), Layer("L1", "Al2O3", 1e-9), Layer("L2", "ZrO2", 4e-9)),
        interfaces=(Interface(), Interface()),
    )
    return DeviceConfig(50e-9, 50e-9, _ELECTRODE, _ELECTRODE, stack, barrier_height=barrier_height)


def _analytic_transmission(delta_Ec, voltage_drop, thickness, m_di, temperature):
    """P_A 的解析闭式：直线势垒 V=a−b·x 的 WKB 积分有闭式，+ 热跃因子，封顶 1。"""
    a = delta_Ec
    b = Q * voltage_drop / thickness
    if b == 0.0:
        integral = np.sqrt(a) * thickness
    else:
        x_turn = min(thickness, a / b)  # 势垒落到 0 的位置（三角势垒在层内）
        integral = (2.0 / (3.0 * b)) * (a ** 1.5 - (a - b * x_turn) ** 1.5)
    exponent = (2.0 / HBAR) * np.sqrt(2.0 * m_di) * integral
    thermal = np.exp(-a / (constants.Boltzmann * temperature))
    return min(1.0, np.exp(-exponent) + thermal)


@pytest.mark.parametrize("drop_V, label", [(0.0, "rectangle"), (0.5, "trapezoid"), (2.5, "triangle")])
def test_pf_transmission_matches_analytic(drop_V, label):
    """矩形/梯形/三角势垒：数值 WKB 与解析闭式一致（含封顶）。"""
    delta_Ec, t, m_di, T = 1.46 * EV, 1e-9, 0.3 * M_E, 298.15
    got = pf_collection_transmission(delta_Ec, drop_V, t, m_di, T)
    expected = _analytic_transmission(delta_Ec, drop_V, t, m_di, T)
    assert got == pytest.approx(expected, rel=5e-3)
    assert 0.0 <= got <= 1.0


def test_pf_transmission_bounds_and_monotonic_in_bias():
    """0 ≤ P_A ≤ 1；偏压越高（Al₂O₃ 电压降越大）势垒越薄、P_A 单调增。"""
    delta_Ec, t, m_di, T = 1.46 * EV, 1e-9, 0.3 * M_E, 298.15
    drops = np.linspace(0.0, 1.4, 8)
    vals = [pf_collection_transmission(delta_Ec, d, t, m_di, T) for d in drops]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert np.all(np.diff(vals) > 0.0)


def test_blocking_layer_detection():
    """阻挡层 = 导带基准高于两侧的内部层；ZAZ 里是中间层，单层无。"""
    zaz = _zaz_device().layer_band_references()
    assert list(_blocking_layer_indices(zaz)) == [1]
    single = np.array([1.74 * EV])
    assert _blocking_layer_indices(single).size == 0


def _zaz_ensemble():
    """ZAZ 上手搭 4 个缺陷：层0体、层2体、界面0（阴极侧）、界面1（阳极侧）。"""
    positions = np.array([
        [10e-9, 10e-9, 2e-9],   # 层 0 体（阴极侧 ZrO₂）
        [10e-9, 10e-9, 7e-9],   # 层 2 体（阳极侧 ZrO₂）
        [20e-9, 20e-9, 4e-9],   # 界面 0（z₁），宿主层 0
        [30e-9, 30e-9, 5e-9],   # 界面 1（z₂），宿主层 2
    ])
    ED = np.array([1.15, 1.15, 1.15, 1.15]) * EV
    m_di = np.full(4, 1.2 * M_E)
    return DefectEnsemble(
        positions=positions, ED=ED, r_D=localization_radius(ED, m_di),
        charge_state=np.ones(4), fixed_charge=np.zeros(4), fixed_charge_offset=np.zeros((4, 3)),
        m_di=m_di, eps_opt=np.full(4, 5.6),
        band_gap=np.full(4, 5.4 * EV), phonon_energy=np.full(4, 0.04 * EV),
        huang_rhys=np.ones(4), attempt_frequency=np.full(4, 1.0e13),
        band_reference=np.full(4, 1.74 * EV),
        layer_id=np.array([0, 2, -1, -1]), interface_id=np.array([-1, -1, 0, 1]),
        field_side=np.array([0, 0, -1, 1]),
    )


def _zaz_sim(voltage):
    config = Config(
        device=_zaz_device(),
        simulation=SimulationConfig(voltage=voltage, temperature=298.15, n_events=5, seed=1,
                                    grid=GridConfig(16, 16, 36)),
    )
    return KMCSimulation(config, defects=_zaz_ensemble())


def test_pf_collection_is_position_aware_forward():
    """正向：阴极侧（层0体 + 界面0）穿 Al₂O₃ → P_A<1；阳极侧（层2体 + 界面1）→ P_A=1。"""
    sim = _zaz_sim(voltage=3.0)
    p = sim._pf_collection
    assert p[0] < 1.0 and p[2] < 1.0  # 阴极侧
    assert p[1] == 1.0 and p[3] == 1.0  # 阳极侧
    assert np.all((p >= 0.0) & (p <= 1.0))
    # 宿主层判定正确
    assert list(sim._defect_host_layers()) == [0, 2, 0, 2]


def test_pf_collection_flips_under_reverse_bias():
    """反偏：收集电极换到阴极，阳极侧陷阱改用 P_A、阴极侧立即收集（角色互换）。"""
    sim = _zaz_sim(voltage=-3.0)  # 对称 ZAZ 下 V_bi=0，驱动电势 <0
    assert sim._pf_forward is False
    p = sim._pf_collection
    assert p[0] == 1.0 and p[2] == 1.0  # 阴极侧现在立即收集
    assert p[1] < 1.0 and p[3] < 1.0    # 阳极侧现在要穿 Al₂O₃


def test_r8_eff_folds_collection_probability():
    """R8_eff = P_A·R8 折进事件表：阴极侧 R8 被压低，阳极侧不变。"""
    from kmc_leakage.kmc import _B_R8
    from kmc_leakage.rates import r8_pf_emission

    sim = _zaz_sim(voltage=3.0)
    occ = np.zeros(4, dtype=bool)
    blocks, _ = sim._rates(occ)
    _, field_at, _ = sim._electrostatics(occ)
    raw = r8_pf_emission(
        sim.defects.attempt_frequency, sim.defects.ED, field_at,
        sim.defects.charge_state, sim.defects.eps_opt, 298.15,
    )
    assert np.allclose(blocks[_B_R8], sim._pf_collection * raw, rtol=1e-12, atol=0.0)


def test_single_layer_pf_collection_is_unity():
    """单层无阻挡层：P_A≡1，R8_eff=R8，与 F03 前逐位相同。"""
    from conftest import make_config

    sim = KMCSimulation(make_config(voltage=3.0))
    assert np.all(sim._pf_collection == 1.0)
    assert sim._pf_forward is True
