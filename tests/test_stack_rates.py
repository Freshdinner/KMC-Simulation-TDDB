"""分段 WKB、按材料取速率参数、分段 G_ab 与近邻截断重推（Phase 4 = ADR-012/010/013）。

这一批盯住 Phase 4 的速率交付：
- 分段 WKB（ADR-012）：m_di 沿路径按层取；界面台阶用双值采样、不插成斜坡；
  R1 势垒顶落在最高的那段（叠层里可在阻挡层内）；
- 分段几何因子 G_ab（ADR-010）：α 沿路径按所在层的 m_di 取（路径传播质量，
  review F08）；同质材料退回闭式；跨层用 Al₂O₃ 低隧穿质量算衰减；
- 事件能量差仍是 F02 的多体口径（ΔH=(E_b−E_a)−U_ab），正反向细致平衡不变；
- 近邻截断按叠层所有材料里最慢的衰减推、判据用被丢弃的总速率上界（ADR-013/F05）；
- 按材料/物种取速率参数（R2/R3 前置因子、R4/R5 的 c_MP、R8 的 ε_opt/ν）；
- 退化用例（ADR-018）：三层同材料 + ΔE_C=0 的速率表与轨迹逐位退回单层。
"""

import numpy as np
import pytest
from scipy import constants

from kmc_leakage.bands import conduction_band_edge, trap_levels
from kmc_leakage.config import (
    Config,
    DefectLevelConfig,
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
from kmc_leakage.kmc import (
    MECHANISMS,
    KMCSimulation,
    _band_on_path_steps,
    _interface_step_nodes,
    r1_electrode_current,
)
from kmc_leakage.rates import r6_elastic_hop, segmented_overlap_factor, wkb_transmission

EV = constants.electron_volt
M_E = constants.electron_mass
HBAR = constants.hbar
Q = constants.elementary_charge


def _material(name, kappa, eps_opt=5.6, band_gap_eV=5.4, chi_eV=2.86, m_cb=1.2, m_di=None,
              phonon_eV=0.04):
    return DielectricMaterial(
        name=name, kappa=kappa, eps_opt=eps_opt, band_gap=band_gap_eV * EV,
        electron_affinity=chi_eV * EV, m_cb=m_cb * M_E, m_vb=2.5 * M_E,
        m_di=(m_cb if m_di is None else m_di) * M_E,
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


# --------------------------------------------------------------------------
# 分段 WKB（ADR-012）
# --------------------------------------------------------------------------
def test_wkb_per_segment_mass_matches_piecewise_analytic():
    """矩形势垒分两段、每段不同 m_di：P = exp[-(2/ℏ)Σ√(2·m_seg·U0)·d_seg]。

    m_di 传成沿积分轴逐样点给出的数组（路径传播质量），常数 m_di 时逐位退回标量。
    """
    d, U0 = 2.0e-9, 1.2 * EV
    z = np.linspace(0.0, d, 65)
    barrier = np.full_like(z, U0)
    m_lo, m_hi = 1.2 * M_E, 0.3 * M_E
    m_path = np.where(z < d / 2, m_lo, m_hi)  # 前半 m_lo、后半 m_hi

    # 解析：两半各 d/2，√ 项各自的质量。界面处质量台阶落在一个格子内混合，
    # 65 点梯形求积在过渡格上有 ~0.5% 的量级误差（固定网格上台阶不可避免）
    expected_exponent = (2.0 / HBAR) * (
        np.sqrt(2.0 * m_lo * U0) * (d / 2) + np.sqrt(2.0 * m_hi * U0) * (d / 2)
    )
    got = wkb_transmission(z, barrier, np.array(0.0), m_path)
    assert np.log(got) == pytest.approx(-expected_exponent, rel=1.5e-2)
    # 结果确实落在两个纯质量极限之间（真的按段取质量，不是取了某一个）
    all_lo = wkb_transmission(z, barrier, np.array(0.0), m_lo)
    all_hi = wkb_transmission(z, barrier, np.array(0.0), m_hi)
    assert all_lo < got < all_hi

    # 常数 m_di 数组与标量逐位相同
    const_arr = wkb_transmission(z, barrier, np.array(0.0), np.full_like(z, m_lo))
    const_scalar = wkb_transmission(z, barrier, np.array(0.0), m_lo)
    assert const_arr == const_scalar


def test_band_on_path_keeps_true_step_not_ramp():
    """界面双值采样：界面下侧一格内仍取下层 E_C，不被插成斜坡（ADR-012 / review F06）。

    直接对比：普通 np.interp 会把台阶插成斜坡（界面前一格给出居中的错值），
    分段插值给出真实的下层值。单层（无界面）时两者逐位相同。
    """
    device = _zaz_device()
    z_nodes = np.arange(37) * 0.25e-9  # nz=36, hz=0.25nm
    band_nodes = conduction_band_edge(device, z_nodes, np.zeros_like(z_nodes))  # 零场：B_ℓ 台阶
    iface_idx, delta_Ec = _interface_step_nodes(device, z_nodes)
    assert list(iface_idx) == [16, 20]  # 4nm、5nm 界面对齐到节点
    assert np.allclose(delta_Ec / EV, [1.46, -1.46])  # 亲和能对齐 ΔE_C

    z_probe = np.array([3.9e-9])  # 界面前一格（3.75–4.0 nm），仍在 ZrO₂ 里
    stepped = _band_on_path_steps(z_probe, z_nodes, band_nodes, iface_idx, delta_Ec)
    plain = np.interp(z_probe, z_nodes, band_nodes)
    assert stepped[0] / EV == pytest.approx(1.74, rel=1e-9)  # 真实台阶：下层 ZrO₂ 值
    assert plain[0] / EV > 2.3  # 普通插值把 1.74→3.20 的台阶插成斜坡，给出居中错值

    # 单层（无界面）时分段插值逐位退回 np.interp
    empty_idx, empty_d = np.zeros(0, dtype=int), np.zeros(0)
    single_nodes = np.linspace(0.0, 9e-9, 37)
    single_band = 1.74 * EV - Q * 3.0 * single_nodes / 9e-9
    probe = np.linspace(0.0, 9e-9, 41)
    got = _band_on_path_steps(probe, single_nodes, single_band, empty_idx, empty_d)
    assert np.array_equal(got, np.interp(probe, single_nodes, single_band))


def test_r1_barrier_top_sits_in_blocking_layer_at_low_bias():
    """低偏压下 ZAZ 的 R1 势垒顶落在 Al₂O₃ 内（比负极 E_B 高一个 ΔE_C）。

    分段势垒保留台阶后，np.max(barrier) 自动落到阻挡层顶——阈值切分与 RS 对照式
    同口径（ADR-012）。这里用 R1 的分段势垒直接检查最高点确实抬到了 3.2 eV 附近。
    """
    device = _zaz_device()
    z_nodes = np.arange(37) * 0.25e-9
    phi_bar = np.zeros_like(z_nodes)  # 零偏压零电荷：纯 B_ℓ 台阶
    z = np.linspace(0.0, device.Lz, 129)
    band_nodes = conduction_band_edge(device, z_nodes, phi_bar)
    iface_idx, delta_Ec = _interface_step_nodes(device, z_nodes)
    barrier = _band_on_path_steps(z, z_nodes, band_nodes, iface_idx, delta_Ec)
    assert np.max(barrier) / EV == pytest.approx(3.20, rel=1e-6)  # 1.74 + 1.46（Al₂O₃ 内）
    # R1 电流能在叠层上求值（分段 m_di + 分段势垒），非负且有限
    tunnel, thermionic = r1_electrode_current(device, 3.0, 298.15, z_nodes, phi_bar)
    assert np.isfinite(tunnel) and np.isfinite(thermionic)
    assert tunnel >= 0.0 and thermionic >= 0.0


# --------------------------------------------------------------------------
# 分段几何因子 G_ab（ADR-010 / review F08）
# --------------------------------------------------------------------------
def test_segmented_overlap_reduces_to_closed_form_when_homogeneous():
    """同质材料（单层）里分段 G_ab = exp[−r_ab(1/r_D,a+1/r_D,b)]（rtol 1e-12）。"""
    layer_edges = np.array([0.0, 9e-9])
    layer_m_di = np.array([1.2 * M_E])
    layer_band = np.array([1.74 * EV])
    ED_a, ED_b = 1.15 * EV, 0.9 * EV
    trap_a, trap_b = 1.74 * EV - ED_a, 1.74 * EV - ED_b
    z_a, z_b = np.array([2e-9]), np.array([5e-9])
    r_ab = np.array([3e-9])  # 同 xy，纯 z 方向

    got = segmented_overlap_factor(z_a, z_b, r_ab, np.array([trap_a]), np.array([trap_b]),
                                   layer_edges, layer_m_di, layer_band)
    rD_a = localization_radius(np.array([ED_a]), 1.2 * M_E)
    rD_b = localization_radius(np.array([ED_b]), 1.2 * M_E)
    closed = np.exp(-r_ab * (1.0 / rD_a + 1.0 / rD_b))
    assert got == pytest.approx(closed, rel=1e-12)

    # 同 z 平面对（Δz=0，如两个同界面缺陷）也不出错，单段积分
    flat = segmented_overlap_factor(z_a, z_a, r_ab, np.array([trap_a]), np.array([trap_b]),
                                    layer_edges, layer_m_di, layer_band)
    assert flat == pytest.approx(closed, rel=1e-12)


def test_segmented_overlap_uses_path_mass_in_blocking_layer():
    """跨 Al₂O₃ 的 G_ab 用**路径所在层**的 m_di（review F08），与纯 ZrO₂ 闭式不同。

    低隧穿质量的 Al₂O₃ 衰减更慢，跨层重叠因子比"全用 ZrO₂ 质量"的闭式**大**
    （阻挡层不必然抑制陷阱间隧穿，review F10）。用独立的分段解析式对照。
    """
    device = _zaz_device()
    edges = device.stack.layer_edges  # [0,4,5,9] nm
    m_layer = device.stack.layer_m_di  # [1.2, 0.3, 1.2] m_e
    band_layer = device.layer_band_references()  # [1.74, 3.20, 1.74] eV
    # 两个 ZrO₂ 陷阱（ED=1.15，band_ref=1.74 → trap_energy=0.59 eV），跨 Al₂O₃
    trap = np.array([1.74 * EV - 1.15 * EV])
    z_a, z_b, r_ab = np.array([2e-9]), np.array([7e-9]), np.array([5e-9])

    got = segmented_overlap_factor(z_a, z_b, r_ab, trap, trap, edges, m_layer, band_layer)

    # 独立分段解析：三段（ZrO₂ / Al₂O₃ / ZrO₂），各段 α 用该层 m_di
    def alpha(m, B):
        return np.sqrt(2.0 * m * max(B - trap[0], 0.0)) / HBAR
    a_zro2 = alpha(1.2 * M_E, 1.74 * EV)
    a_al2o3 = alpha(0.3 * M_E, 3.20 * EV)
    arc = 5e-9
    integral = 2.0 * (a_zro2 * arc * 0.4 + a_al2o3 * arc * 0.2 + a_zro2 * arc * 0.4)
    assert got[0] == pytest.approx(np.exp(-integral), rel=1e-9)

    # 若错用物种（ZrO₂）质量贯穿整条路径，会得到更强的抑制——两者必须显著不同。
    # 注意用比值判断：两值都 ~1e-25，np.isclose 默认 atol=1e-8 会把它们都当成 0
    rD = localization_radius(np.array([1.15 * EV]), 1.2 * M_E)
    species_closed = np.exp(-r_ab * 2.0 / rD)[0]
    assert got[0] / species_closed > 5.0  # Al₂O₃ 低质量 → 衰减更慢 → 重叠大一个量级以上


def test_zaz_hop_detailed_balance_preserved():
    """ZAZ 里 R7 正反向仍满足细致平衡：只换几何因子、事件能量差仍是 F02 多体口径。

    R(a→b|n)/R(b→a|n') = exp[−(H(n')−H(n))/k_BT]。取两个 ZrO₂ 陷阱、深度不对称，
    几何因子对称（同一对 a↔b），比值只由多体事件能量差决定。
    """
    device = _zaz_device()
    positions = np.array([[25e-9, 25e-9, 2e-9], [25e-9, 25e-9, 3e-9]])  # 同层 ZrO₂ 邻近对
    ED = np.array([1.30, 1.00]) * EV
    ensemble = DefectEnsemble.uniform(positions, ED, ZRO2, band_reference=1.74 * EV)
    config = Config(
        device=device,
        simulation=SimulationConfig(voltage=0.0, temperature=300.0, n_events=10, seed=1,
                                    grid=GridConfig(16, 16, 36)),
    )
    sim = KMCSimulation(config, defects=ensemble)
    pairs = sim._pairs

    def idx(a, b):
        return int(np.where((pairs.a == a) & (pairs.b == b))[0][0])

    r_fwd = sim._rates(np.array([True, False]))[1][idx(0, 1)]  # n=[1,0]
    r_rev = sim._rates(np.array([False, True]))[1][idx(1, 0)]  # n'=[0,1]
    eps_a0 = trap_levels(ensemble.band_reference, sim._electrostatics(np.array([True, False]))[0], ensemble.ED)[0]
    eps_b0 = trap_levels(ensemble.band_reference, sim._electrostatics(np.array([False, True]))[0], ensemble.ED)[1]
    kT = constants.Boltzmann * 300.0
    assert r_fwd / r_rev == pytest.approx(np.exp(-(eps_b0 - eps_a0) / kT), rel=1e-9)


# --------------------------------------------------------------------------
# 近邻截断重推（ADR-013 / review F05）
# --------------------------------------------------------------------------
def test_stack_neighbor_cutoff_uses_slowest_material():
    """截断半径按叠层最慢衰减（Al₂O₃ 低质量）推，比只看 ZrO₂ 的半径更大（ADR-013）。

    同一套缺陷分别挂到 ZAZ 与纯 ZrO₂ 单层上：ZAZ 的 α_min 落在 Al₂O₃（衰减更慢），
    r_cut 因此更大。判据（被丢弃总速率上界，F05）在两边相同（同缺陷、同 R8_min/
    ν_max/N），所以差异纯粹来自 α_min 换成了最慢材料。
    """
    positions = np.array([[25e-9, 25e-9, 2e-9], [25e-9, 25e-9, 7e-9]])
    ED = np.array([1.15, 1.15]) * EV
    ensemble = DefectEnsemble.uniform(positions, ED, ZRO2, band_reference=1.74 * EV)

    def cutoff(device):
        cfg = Config(
            device=device,
            simulation=SimulationConfig(voltage=0.0, temperature=298.15, n_events=5, seed=1,
                                        grid=GridConfig(16, 16, 36)),
        )
        return KMCSimulation(cfg, defects=ensemble)._hop_cutoff_radius()

    zaz = cutoff(_zaz_device())
    single = DeviceConfig(
        50e-9, 50e-9, _ELECTRODE, _ELECTRODE,
        DielectricStack.single(ZRO2, 9e-9), barrier_height=1.74 * EV,
    )
    zro2_only = cutoff(single)
    # Al₂O₃ 低隧穿质量压过高势垒 → 衰减更慢 → 截断半径更大（ADR-013）
    assert zaz > zro2_only
    # 量级：α_Al2O3 / α_ZrO2 ≈ √(0.3·2.61 / 1.2·1.15) 的倒数，r_cut 比约 1.3×
    assert zaz / zro2_only == pytest.approx(1.33, rel=0.1)
    # 半径落在物理合理范围内（未爆到整盒）
    assert 0.0 < zaz < np.hypot(np.hypot(50e-9, 50e-9), 9e-9)


# --------------------------------------------------------------------------
# 按材料/物种取速率参数（R8 的 ε_opt / ν 逐缺陷）
# --------------------------------------------------------------------------
def test_rates_consume_per_defect_material_params():
    """R8 逐缺陷取自己的 ε_opt / ν / E_D，并折进 F03 收集概率（R8_eff = P_A·R8）。"""
    from kmc_leakage.rates import r8_pf_emission

    device = _zaz_device()
    positions = np.array([[20e-9, 20e-9, 2e-9], [30e-9, 30e-9, 7e-9]])
    ED = np.array([1.15, 1.15]) * EV
    m_di = np.array([1.2, 1.2]) * M_E
    # 两个缺陷物性不同：ε_opt 与 ν 各异
    ensemble = DefectEnsemble(
        positions=positions, ED=ED, r_D=localization_radius(ED, m_di),
        charge_state=np.array([1.0, 1.0]), fixed_charge=np.zeros(2), fixed_charge_offset=np.zeros((2, 3)),
        m_di=m_di,
        eps_opt=np.array([5.6, 3.0]), band_gap=np.array([5.4, 6.4]) * EV,
        phonon_energy=np.array([0.04, 0.09]) * EV, huang_rhys=np.array([1.0, 1.0]),
        attempt_frequency=np.array([1.0e13, 2.0e13]),
        band_reference=np.array([1.74, 1.74]) * EV,
        layer_id=np.array([0, 2]), interface_id=np.array([-1, -1]),
        field_side=np.array([0, 0]),
    )
    config = Config(
        device=device,
        simulation=SimulationConfig(voltage=3.0, temperature=298.15, n_events=5, seed=1,
                                    grid=GridConfig(16, 16, 36)),
    )
    sim = KMCSimulation(config, defects=ensemble)
    occ = np.array([False, False])
    blocks, _ = sim._rates(occ)
    _, field_at, _ = sim._electrostatics(occ)

    expected = r8_pf_emission(
        ensemble.attempt_frequency, ensemble.ED, field_at, ensemble.charge_state,
        ensemble.eps_opt, 298.15,
    )
    from kmc_leakage.kmc import _B_R8

    # R8_eff = P_A·R8：负极侧缺陷（层 0，穿 Al₂O₃）P_A<1，正极侧缺陷（层 2）P_A=1
    assert sim._pf_collection[0] < 1.0 and sim._pf_collection[1] == 1.0
    assert np.allclose(blocks[_B_R8], sim._pf_collection * expected, rtol=1e-12, atol=0.0)
    # 两缺陷 ε_opt/ν 不同 -> R8 不同（真的按缺陷取，不是全体一份）
    assert not np.isclose(blocks[_B_R8][0], blocks[_B_R8][1], rtol=1e-6)


# --------------------------------------------------------------------------
# 退化用例（ADR-018）：三层同材料 + ΔE_C=0 逐位退回单层
# --------------------------------------------------------------------------
def _degenerate_pair():
    """同一套注入缺陷、同网格，分别挂到单层 9nm 与三层同材料 4/1/4 ZrO₂ 上。"""
    positions = np.array([
        [25e-9, 25e-9, 2.0e-9], [25e-9, 25e-9, 3.0e-9],
        [24e-9, 26e-9, 6.0e-9], [26e-9, 24e-9, 7.0e-9],
    ])
    ED = np.array([1.15, 1.30, 1.05, 1.20]) * EV
    ensemble = DefectEnsemble.uniform(positions, ED, ZRO2, band_reference=1.74 * EV)

    single = DeviceConfig(
        50e-9, 50e-9, _ELECTRODE, _ELECTRODE,
        DielectricStack.single(ZRO2, 9e-9), barrier_height=1.74 * EV,
    )
    multi_stack = DielectricStack(
        materials={"ZrO2": ZRO2},
        layers=(Layer("L0", "ZrO2", 4e-9), Layer("L1", "ZrO2", 1e-9), Layer("L2", "ZrO2", 4e-9)),
        interfaces=(Interface(delta_Ec=0.0), Interface(delta_Ec=0.0)),
    )
    multi = DeviceConfig(50e-9, 50e-9, _ELECTRODE, _ELECTRODE, multi_stack, barrier_height=1.74 * EV)

    def make(device):
        cfg = Config(
            device=device,
            simulation=SimulationConfig(voltage=3.0, temperature=298.15, n_events=300, seed=7,
                                        grid=GridConfig(16, 16, 36)),
        )
        return KMCSimulation(cfg, defects=ensemble)

    return make(single), make(multi)


def test_uniform_multilayer_rate_table_matches_single_layer():
    """固定组态下完整速率表：三层同材料 + ΔE_C=0 与单层逐位退回（rtol ≤ 1e-12，ADR-018）。"""
    sim_single, sim_multi = _degenerate_pair()
    assert sim_single.device.stack.n_materials == sim_multi.device.stack.n_materials == 1
    for occ in (np.array([False, False, False, False]), np.array([True, False, True, False])):
        b1, h1 = sim_single._rates(occ)
        b2, h2 = sim_multi._rates(occ)
        assert np.allclose(b1, b2, rtol=1e-12, atol=0.0)
        assert np.allclose(h1, h2, rtol=1e-12, atol=0.0)
    # R1 两分量也逐位退回
    assert sim_single._r1_current == pytest.approx(sim_multi._r1_current, rel=1e-12)


def test_uniform_multilayer_trajectory_matches_single_layer():
    """同种子完整轨迹：三层同材料 + ΔE_C=0 与单层的 J 与机制分解退回（rtol ≤ 1e-9，ADR-018）。"""
    sim_single, sim_multi = _degenerate_pair()
    a, b = sim_single.run(), sim_multi.run()
    assert a.current == pytest.approx(b.current, rel=1e-9)
    for name in MECHANISMS:
        assert a.by_mechanism[name] == pytest.approx(b.by_mechanism[name], rel=1e-9, abs=0.0)
