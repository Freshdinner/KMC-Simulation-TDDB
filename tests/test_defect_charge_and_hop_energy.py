"""缺陷固定电荷（F01）与 R7 多体事件能量（F02）的检验。

两项都是 ADR-021 记录、要在冻结单层基准前修好的 Critical Bug：

- **F01**：空施主陷阱带固定正电荷 +Z·q，必须进泊松源项。净电荷 q·(Z_i−n_i)，
  耦合矩阵按 −q 电子预计算，故组态叠加权重取 n_i−Z_i（见 kmc._electrostatics）。
- **F02**：陷阱间跃迁 R7 的 MA 能量因子要用完整事件能量差
  ΔH_{a→b}=(E_b−E_a)−U_ab（U_ab=q²·G_ab 是两处电子的库仑互作用能），而不是
  跳跃前组态里的单点能级差。正反向速率因此满足细致平衡。
- **固定电荷 q_fix**（ADR-048，TDDB 生成空位带的 O_i⁻）：净电荷 q·(Z_i−n_i) + q_fix,i。与缺陷重合的
  q_fix 是该缺陷净电荷的一部分（等价于把 Z 换成 Z + q_fix/q）；离开缺陷位置的 q_fix 是独立的点电荷。

数值锚点取自 docs/reviews/zaz-stack-review-2026-09-15.md（同一套静电接口的独立
复核）：50×50×9 nm、32×32×36 网格、300 K、零偏压、陷阱在 (25,25,4) 与
(25,25,5) nm 时 U_ab = 0.03476324 eV，F02 修好后正反比 = 0.5105069。
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy import constants

from kmc_leakage.bands import trap_levels
from kmc_leakage.config import (
    Config,
    DefectConfig,
    DefectLevelConfig,
    DefectSpatialConfig,
    DeviceConfig,
    DielectricMaterial,
    DielectricStack,
    ElectrodeMaterial,
    GridConfig,
    SimulationConfig,
)
from kmc_leakage.defects import DefectEnsemble
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.rates import r7_inelastic_hop

EV = constants.electron_volt
M_E = constants.electron_mass
Q = constants.elementary_charge

# 参考预设 + 复核报告当时的默认器件（9 nm 单层），复核报告即用此配置
_DIELECTRIC = DielectricMaterial(
    name="reference-preset",
    kappa=38.0,
    eps_opt=5.6,
    band_gap=5.4 * EV,
    electron_affinity=2.86 * EV,
    m_cb=1.2 * M_E,
    m_vb=2.5 * M_E,
    m_di=1.2 * M_E,
    phonon_energy=0.04 * EV,
    huang_rhys=1.0,
    attempt_frequency=1.0e13,
)
_ELECTRODE = ElectrodeMaterial("TiN", 4.6 * EV, M_E, 5.0 * EV)
# 单层退化叠层：参考预设材料 + 9 nm 单层。层里挂一份体缺陷配置只为让 stack
# 完整——本测试用显式 DefectEnsemble 跑 KMC，不经缺陷生成，config.defects 不被消费。
_LAYER_DEFECTS = DefectConfig(
    concentration=3.0e18 * constants.centi ** -3,
    level=DefectLevelConfig(mode="fixed", ED=1.15 * EV),
    spatial=DefectSpatialConfig(mode="uniform"),
    charge_state=1.0,
    min_separation=0.0,
)
_DEVICE = DeviceConfig(
    50e-9,
    50e-9,
    _ELECTRODE,
    _ELECTRODE,
    DielectricStack.single(_DIELECTRIC, 9e-9, _LAYER_DEFECTS),
    barrier_height=1.74 * EV,
)


def _two_trap_sim(ED_a_eV, ED_b_eV, charge_state=1.0, temperature=300.0, voltage=0.0):
    """两个陷阱、参考网格的 KMCSimulation，位置取复核报告的 (25,25,4)/(25,25,5) nm。"""
    positions = np.array([[25e-9, 25e-9, 4e-9], [25e-9, 25e-9, 5e-9]])
    ED = np.array([ED_a_eV, ED_b_eV]) * EV
    ensemble = DefectEnsemble.uniform(
        positions, ED, _DIELECTRIC, band_reference=1.74 * EV, charge_state=charge_state
    )
    config = Config(
        device=_DEVICE,
        simulation=SimulationConfig(
            voltage=voltage, temperature=temperature, n_events=10, seed=1, grid=GridConfig(32, 32, 36)
        ),
    )
    return KMCSimulation(config, defects=ensemble)


def _pair_index(pairs, a, b):
    return int(np.where((pairs.a == a) & (pairs.b == b))[0][0])


# --------------------------------------------------------------------------
# F01：空施主的固定 +Z·q 进入泊松源项
# --------------------------------------------------------------------------
def test_empty_donor_carries_fixed_positive_charge():
    """空施主 +Z·q、占据后中性，与复核报告的电势逐位吻合。

    两陷阱全空（均为施主）时每个位置的缺陷电势 = +0.03476324 V，全占据（中性）
    时 = 0 V；旧实现给的是 0 V 与 −0.03476324 V（只投占据电子 −q·n）。
    """
    sim = _two_trap_sim(1.15, 1.15)
    applied = sim._phi_applied_at

    phi_empty, _, _ = sim._electrostatics(np.array([False, False]))
    phi_occ, _, _ = sim._electrostatics(np.array([True, True]))

    defect_empty = phi_empty - applied
    defect_occ = phi_occ - applied
    # 空施主：正电势，量级与复核报告一致，且等于对方 −q 电子响应的相反数
    assert np.all(defect_empty > 0.0)
    assert defect_empty == pytest.approx(0.03476324, rel=1e-6)
    assert defect_empty[0] == pytest.approx(-sim.coupling.phi_matrix[1, 0], rel=1e-12)
    # 占据（中性）：净电荷为零，缺陷电势归零
    assert np.allclose(defect_occ, 0.0, atol=1e-12)


def test_net_charge_weight_reaches_field_and_band_profile():
    """净电荷权重 n_i−Z_i 同时作用于电势、电场与横向平均剖面，不只是电势。

    复核报告 F08 特别点出：局域场也必须用净电荷读回（R4/R5/R8 都吃场），只给电势
    改权重会让场继续用错电荷。这里按一个占据、一个空的组态逐位核对三个返回量。
    """
    sim = _two_trap_sim(1.15, 1.10)
    occ = np.array([True, False])
    weight = occ.astype(float) - sim.defects.charge_state  # [0, -1]

    phi, field_mag, phi_bar = sim._electrostatics(occ)
    phi_expected = (
        sim._phi_applied_at + weight @ sim.coupling.phi_matrix - weight * sim.coupling.phi_self
    )
    field_expected = (
        sim._field_applied_at + weight @ sim.coupling.field_matrix - weight * sim.coupling.field_self
    )
    bar_expected = sim._phi_bar_applied + weight @ sim.coupling.phi_bar

    assert np.array_equal(phi, phi_expected)
    assert np.array_equal(field_mag, np.linalg.norm(field_expected, axis=0))
    assert np.array_equal(phi_bar, bar_expected)
    # 空施主真的在近旁产生了非零场（否则等于没进源项）
    assert field_mag[0] > 0.0


# --------------------------------------------------------------------------
# 固定电荷 q_fix（ADR-048）：净电荷 q·(Z−n) + q_fix
# --------------------------------------------------------------------------
_OCCUPATIONS = [np.array(occ) for occ in ([False, False], [True, False], [False, True], [True, True])]


def _field_vectors(sim, occ):
    """`_electrostatics` 里的场矢量 (3, N)（它只返回大小）。"""
    weight = occ.astype(float) - sim.defects.charge_state
    return (
        sim._field_applied_at + sim._field_fixed_at
        + weight @ sim.coupling.field_matrix - weight * sim.coupling.field_self
    )


def test_colocated_fixed_charge_is_part_of_the_defect_net_charge():
    """与缺陷重合的 q_fix 是该缺陷净电荷 q·(Z−n) + q_fix 的一部分：Z = 1、q_fix = −e 的缺陷与 Z = 0、q_fix = 0
    的缺陷在每个组态下给出相同的电势、电场、横向平均剖面与全部速率（自作用整项扣除，R8 的 β 取 Z + q_fix/q = 0），
    只差固定电荷那一次泊松求解的舍入误差。"""
    base = _two_trap_sim(1.15, 1.10, voltage=2.0)
    fixed = KMCSimulation(base.config, defects=replace(base.defects, fixed_charge=np.array([-Q, 0.0])))
    shifted = KMCSimulation(base.config, defects=replace(base.defects, charge_state=np.array([0.0, 1.0])))
    assert fixed._emission_charge.tolist() == [0.0, 1.0]
    for occ in _OCCUPATIONS:
        for a, b in zip(fixed._electrostatics(occ), shifted._electrostatics(occ)):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15)
        blocks_a, hop_a = fixed._rates(occ)
        blocks_b, hop_b = shifted._rates(occ)
        np.testing.assert_allclose(blocks_a, blocks_b, rtol=1e-9)
        np.testing.assert_allclose(hop_a, hop_b, rtol=1e-9)
    # 固定电荷确实进了静电：两种组态记账都不同于不带 q_fix 的原缺陷
    assert not np.allclose(fixed._electrostatics(_OCCUPATIONS[0])[0], base._electrostatics(_OCCUPATIONS[0])[0])


def test_displaced_fixed_charge_is_an_independent_point_charge():
    """离开缺陷位置的 q_fix 是另一个点电荷：它对每个缺陷（包括它所属的缺陷）的电势与场就是它单独的泊松解在缺陷处
    的读数，横向平均剖面同样；R8 的 β 仍取 Z（发射后缺陷处只留下 +Z·q）。"""
    base = _two_trap_sim(1.15, 1.10, voltage=2.0)
    offset = np.array([[0.3e-9, 0.0, 0.6e-9], [0.0, 0.0, 0.0]])
    defects = replace(base.defects, fixed_charge=np.array([-Q, 0.0]), fixed_charge_offset=offset)
    sim = KMCSimulation(base.config, defects=defects)
    assert sim._emission_charge.tolist() == [1.0, 1.0]
    solver = sim.solver
    point = solver.solve(solver.deposit(defects.positions[:1] + offset[:1], np.array([-Q])), 0.0)
    for occ in _OCCUPATIONS:
        phi, field, bar = sim._electrostatics(occ)
        phi0, _, bar0 = base._electrostatics(occ)
        np.testing.assert_allclose(phi, phi0 + solver.interpolate(point, defects.positions), rtol=1e-12)
        np.testing.assert_allclose(bar, bar0 + solver.lateral_average(point), rtol=1e-12, atol=1e-15)
        expected = _field_vectors(base, occ) + solver.field_vectors_at(point, defects.positions, defects.field_side)
        np.testing.assert_allclose(field, np.linalg.norm(expected, axis=0), rtol=1e-12)


# --------------------------------------------------------------------------
# F02：R7 用完整事件能量差 ΔH = (E_b − E_a) − U_ab
# --------------------------------------------------------------------------
def test_r7_forward_reverse_satisfy_detailed_balance():
    """正反 R7 速率比在各自出发组态求值后严格满足细致平衡。

    R(a→b|n)/R(b→a|n') = exp[−(H(n')−H(n))/k_BT]。两陷阱一个电子时
    H(n')−H(n) = ε_b^0 − ε_a^0（无互作用项，单电子），直接用代码的能级机器算。
    只更换几何重叠因子、不修事件能量差是过不了这条的（复核报告 F02）。
    """
    sim = _two_trap_sim(1.30, 1.00)  # 深度不对称，ΔH ≠ 0
    pairs = sim._pairs

    _, hop_forward_cfg = sim._rates(np.array([True, False]))  # n = [1,0]
    _, hop_reverse_cfg = sim._rates(np.array([False, True]))  # n' = [0,1]
    r_forward = hop_forward_cfg[_pair_index(pairs, 0, 1)]
    r_reverse = hop_reverse_cfg[_pair_index(pairs, 1, 0)]

    # ε_a^0 / ε_b^0：各自单独占据时的单点能级（另一陷阱为空、无贡献）
    eps_a0 = trap_levels(sim.defects.band_reference, sim._electrostatics(np.array([True, False]))[0], sim.defects.ED)[0]
    eps_b0 = trap_levels(sim.defects.band_reference, sim._electrostatics(np.array([False, True]))[0], sim.defects.ED)[1]
    kT = constants.Boltzmann * sim.sim.temperature
    assert r_forward / r_reverse == pytest.approx(np.exp(-(eps_b0 - eps_a0) / kT), rel=1e-9)


def test_r7_ratio_matches_review_anchor():
    """复现复核报告的正反比 0.5105069（第二陷阱比第一浅 U_ab/2）。"""
    u_ab = float(_two_trap_sim(1.15, 1.15)._hop_U_ab[0])  # 先取 U_ab
    sim = _two_trap_sim(1.15, 1.15 - 0.5 * u_ab / EV)
    pairs = sim._pairs
    r_forward = sim._rates(np.array([True, False]))[1][_pair_index(pairs, 0, 1)]
    r_reverse = sim._rates(np.array([False, True]))[1][_pair_index(pairs, 1, 0)]
    assert r_forward / r_reverse == pytest.approx(0.5105069, rel=1e-6)


def test_r7_uses_many_body_energy_not_single_point():
    """R7 用的是 E_b − U_ab，不是跳跃前组态的单点 E_b——两者数值上确实不同。

    这条钉住 F02 真的落地了：把终态能级换回单点 E_b（旧写法）会得到不同的速率。
    """
    u_ab = float(_two_trap_sim(1.15, 1.15)._hop_U_ab[0])
    sim = _two_trap_sim(1.15, 1.15 - 0.5 * u_ab / EV)
    pairs = sim._pairs
    k = _pair_index(pairs, 0, 1)
    _, hop = sim._rates(np.array([True, False]))

    energy = trap_levels(sim.defects.band_reference, sim._electrostatics(np.array([True, False]))[0], sim.defects.ED)
    diel = sim.device.stack.single_material
    args = (diel.attempt_frequency, pairs.r[k], sim.defects.r_D[0], sim.defects.r_D[1])
    many_body = r7_inelastic_hop(*args, energy[0], energy[1] - sim._hop_U_ab[k], sim.sim.temperature)
    single_point = r7_inelastic_hop(*args, energy[0], energy[1], sim.sim.temperature)

    assert hop[k] == pytest.approx(many_body, rel=1e-12)
    assert not np.isclose(hop[k], single_point, rtol=1e-6)
