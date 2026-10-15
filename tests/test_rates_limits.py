"""速率函数的解析极限与物理约束检验（R1–R8）。

每个速率函数至少配一个验证：能对照解析极限的对照解析极限，没有解析解的
用物理约束（细致平衡、互补占据因子等）。
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy import constants
from scipy.integrate import quad

from conftest import make_device
from kmc_leakage.bands import r1_band_bottom
from kmc_leakage.rates import (
    barrier_maximum,
    effective_richardson_constant,
    elastic_prefactor,
    miller_abrahams_factor,
    multiphonon_coupling,
    pf_beta,
    phonon_transition_probability,
    r1_electrode_electrode,
    r2_elastic_capture,
    r3_elastic_emission,
    r4_multiphonon_capture,
    r5_multiphonon_emission,
    r6_elastic_hop,
    r7_inelastic_hop,
    r8_pf_emission,
    richardson_schottky_current_density,
    supply_function,
    wkb_transmission,
)
from kmc_leakage.rates.pf_emission import pf_bracket

EV = constants.electron_volt
M_E = constants.electron_mass
KB = constants.Boltzmann
HBAR = constants.hbar

NU0 = 1.0e13
ED = 1.15 * EV
EPS_OPT = 5.6
M_DI = 1.2 * M_E
T = 298.15
BOTTOM = -5.0 * EV  # R1 的积分下限：费米能 5 eV 的同种电极、U ≥ 0 时就是负极导带底


# ----------------------------------------------------------------------
# WKB
# ----------------------------------------------------------------------
def test_wkb_rectangular_barrier_matches_analytic():
    """矩形势垒：P = exp(-2d√(2mU0)/ℏ)，梯形积分对常数被积函数是精确的。"""
    d, U0 = 1.5e-9, 1.2 * EV
    z = np.linspace(0.0, d, 33)
    barrier = np.full_like(z, U0)
    expected = np.exp(-2.0 * d * np.sqrt(2.0 * M_DI * U0) / HBAR)
    assert wkb_transmission(z, barrier, np.array(0.0), M_DI) == pytest.approx(expected, rel=1e-12)


def test_wkb_above_barrier_is_unity_and_direction_independent():
    z = np.linspace(0.0, 2e-9, 17)
    barrier = 1.0 * EV * (1.0 - z / z[-1])
    assert wkb_transmission(z, barrier, np.array(1.5 * EV), M_DI) == pytest.approx(1.0)
    forward = wkb_transmission(z, barrier, np.array(0.2 * EV), M_DI)
    backward = wkb_transmission(z[::-1], barrier[::-1], np.array(0.2 * EV), M_DI)
    assert forward == pytest.approx(backward, rel=1e-12)


def test_wkb_triangular_barrier_converges_to_analytic():
    """三角势垒（含转折点）：∫√(U0(1-z/d))dz = (2/3)·d·√U0。"""
    d, U0 = 2.0e-9, 1.5 * EV
    expected = np.exp(-2.0 / HBAR * np.sqrt(2.0 * M_DI) * (2.0 / 3.0) * d * np.sqrt(U0))
    errors = []
    for n in (65, 257, 1025):
        z = np.linspace(0.0, d, n)
        P = wkb_transmission(z, U0 * (1.0 - z / d), np.array(0.0), M_DI)
        errors.append(abs(np.log(P) - np.log(expected)))
    assert errors[0] > errors[1] > errors[2]  # 转折点处的 √ 奇性，单调收敛
    assert errors[-1] < 1e-3


# ----------------------------------------------------------------------
# R1 + Richardson-Schottky
# ----------------------------------------------------------------------
def _linear_band(E_B, voltage, n=257, Lz=8e-9):
    z = np.linspace(0.0, Lz, n)
    return z, E_B - constants.elementary_charge * voltage * z / Lz


def test_supply_function_vanishes_at_zero_bias_and_is_positive_otherwise():
    energies = np.linspace(0.0, 2.0 * EV, 51)
    assert np.allclose(supply_function(energies, 0.0, T), 0.0)
    assert np.all(supply_function(energies, 1.0, T) > 0.0)


def test_r1_vanishes_at_zero_bias():
    z, band = _linear_band(1.74 * EV, 0.0)
    tunnel, thermionic = r1_electrode_electrode(z, band, 2.5e-15, 0.0, T, M_E, M_DI, BOTTOM)
    assert tunnel == pytest.approx(0.0, abs=1e-30)
    assert thermionic == pytest.approx(0.0, abs=1e-30)


def test_r1_thermionic_part_matches_boltzmann_limit():
    """E_t > 势垒顶时 P=1，玻尔兹曼极限下热发射子部分有解析式。

    R1_TE -> [A·m_e·(k_BT)²/(2π²ℏ³)]·exp(-E_B/k_BT)·[1-exp(-eU/k_BT)]
    """
    E_B, area, U = 1.74 * EV, 2.5e-15, 2.0
    z, band = _linear_band(E_B, U)
    _, thermionic = r1_electrode_electrode(z, band, area, U, T, M_E, M_DI, BOTTOM, n_points=512)
    kT = KB * T
    expected = (
        area
        * M_E
        * kT ** 2
        / (2.0 * np.pi ** 2 * HBAR ** 3)
        * np.exp(-barrier_maximum(band) / kT)
        * (1.0 - np.exp(-constants.elementary_charge * U / kT))
    )
    assert thermionic == pytest.approx(expected, rel=1e-2)


def test_r1_tunnel_part_grows_with_bias_and_thinner_barrier():
    z, band = _linear_band(1.74 * EV, 2.0)
    low, _ = r1_electrode_electrode(z, band, 2.5e-15, 2.0, T, M_E, M_DI, BOTTOM)
    z4, band4 = _linear_band(1.74 * EV, 4.0)
    high, _ = r1_electrode_electrode(z4, band4, 2.5e-15, 4.0, T, M_E, M_DI, BOTTOM)
    assert high > low > 0.0
    z_thin, band_thin = _linear_band(1.74 * EV, 2.0, Lz=4e-9)
    thin, _ = r1_electrode_electrode(z_thin, band_thin, 2.5e-15, 2.0, T, M_E, M_DI, BOTTOM)
    assert thin > low


def test_r1_tunnel_part_counts_electrons_down_to_band_bottom():
    """低温极限下 R1 的隧穿部分就是完整的 Tsu–Esaki 积分（ADR-061）。

    T → 0 时 k_BT·S(E) 在 E < −eU 为 eU、在 −eU < E < 0 为 −E、在 E > 0 为 0，于是
    R1 → [A·m_e/(2π²ℏ³)]·[eU·∫_{E_c}^{−eU} P dE + ∫_{−eU}^{0} (−E)·P dE]，费米能级以下直到导带底 E_c 的电子
    全部计入；原先从费米能级积起时这个极限为 0。矩形势垒 P(E) = exp[−2d√(2m(V₀ − E))/ℏ]，参照值用 quad 求。
    """
    d, V0, U, area, temperature = 1.0e-9, 1.0 * EV, 0.5, 1.0e-18, 10.0
    z = np.linspace(0.0, d, 33)
    tunnel, _ = r1_electrode_electrode(z, np.full_like(z, V0), area, U, temperature, M_E, M_DI, BOTTOM)

    def transmission(energy):
        return np.exp(-2.0 * d * np.sqrt(2.0 * M_DI * (V0 - energy)) / HBAR)

    eU = constants.elementary_charge * U
    deep, _ = quad(transmission, BOTTOM, -eU, epsabs=0.0, epsrel=1e-12)
    shallow, _ = quad(lambda e: -e * transmission(e), -eU, 0.0, epsabs=0.0, epsrel=1e-12)
    expected = area * M_E / (2.0 * np.pi ** 2 * HBAR ** 3) * (eU * deep + shallow)
    assert tunnel == pytest.approx(expected, rel=1e-3)


def test_r1_band_bottom_is_the_higher_electrode_band_bottom():
    """R1 的积分下限取两侧电极导带底中较高的那个：同种电极时是负极导带底，正极费米能小到它的导带底更高时取正极的。"""
    device = make_device()
    assert r1_band_bottom(device, 3.6) == pytest.approx(-device.negative.fermi_energy, rel=1e-15)
    shallow = replace(device, positive=replace(device.positive, fermi_energy=1.0 * EV))
    assert r1_band_bottom(shallow, 0.5) == pytest.approx(-0.5 * EV - 1.0 * EV, rel=1e-15)
    assert r1_band_bottom(shallow, 4.5) == pytest.approx(-device.negative.fermi_energy, rel=1e-15)


def test_richardson_schottky_independent_comparison():
    """R1 的热发射子部分 vs Richardson-Schottky 解析式（docs 要求的独立检验）。

    两式的势垒模型相同（零场 E_B、不含镜像力降低），偏压依赖也相同，
    差别只在指定的前置因子质量：R1 用发射极自由电子质量 m_e，A* 用介质
    导带底有效质量 m_cb。因此比值应当恒等于 m_e/m_cb，与温度、偏压无关
    ——这比"量级差不多"更适合作为判据。
    """
    E_B, area, m_cb = 1.74 * EV, 2.5e-15, 1.2 * M_E
    ratios = []
    for temperature in (273.15, 323.15, 398.15):
        for U in (1.0, 2.5, 4.0):
            z, band = _linear_band(E_B, U)
            _, thermionic = r1_electrode_electrode(
                z, band, area, U, temperature, M_E, M_DI, BOTTOM, n_points=512
            )
            j_kmc = constants.elementary_charge * thermionic / area
            j_rs = richardson_schottky_current_density(temperature, U, E_B, m_cb)
            ratios.append(j_kmc / j_rs)
    assert np.allclose(ratios, M_E / m_cb, rtol=1.5e-2)


def test_effective_richardson_constant_value():
    """A* = 4π·m·q·k_B²/h³，自由电子时约 1.20e6 A/(m²·K²)。"""
    assert effective_richardson_constant(M_E) == pytest.approx(1.2017e6, rel=1e-3)


# ----------------------------------------------------------------------
# R2 / R3
# ----------------------------------------------------------------------
def test_r2_r3_ratio_is_complementary_occupation():
    """R2/R3 共享前置因子与 P(E)，比值必须等于 f/(1-f)。"""
    E1 = np.array([3.0, 5.0, 7.0]) * EV
    f = np.array([0.2, 0.5, 0.9])
    P = np.array([1e-6, 1e-8, 1e-3])
    r2 = r2_elastic_capture(E1, ED, M_E, M_DI, f, P)
    r3 = r3_elastic_emission(E1, ED, M_E, M_DI, f, P)
    assert np.allclose(r2 / r3, f / (1.0 - f), rtol=1e-12)


def test_elastic_prefactor_scaling_and_cutoff():
    """前置因子 ∝ E1^1.5·(m_e/m_di)^2.5/√ED；E1 ≤ 0 时电极侧无态可用。"""
    base = elastic_prefactor(np.array(4.0 * EV), ED, M_E, M_DI)
    assert elastic_prefactor(np.array(16.0 * EV), ED, M_E, M_DI) == pytest.approx(8.0 * base)
    assert elastic_prefactor(np.array(4.0 * EV), 4.0 * ED, M_E, M_DI) == pytest.approx(0.5 * base)
    heavier = elastic_prefactor(np.array(4.0 * EV), ED, M_E, 2.0 * M_DI)
    assert heavier == pytest.approx(base * 2.0 ** -2.5)
    assert elastic_prefactor(np.array([-1.0 * EV, 0.0]), ED, M_E, M_DI) == pytest.approx([0.0, 0.0])


# ----------------------------------------------------------------------
# R4 / R5
# ----------------------------------------------------------------------
def test_r4_r5_detailed_balance():
    """R5/R4 = exp((E-E_F)/k_BT)：热平衡下陷阱占据回到费米-狄拉克分布。

    这一关系对每个声子分量 p 都成立，因此对求和也成立，与 N_C、P、L_p 的
    具体取值无关——系数写错会立刻破坏它。
    """
    phonon_energy = 0.04 * EV
    p = np.arange(12)
    L = phonon_transition_probability(p, 1.0, phonon_energy, T)
    trap_energy = 0.35 * EV
    fermi_level = 0.0
    energies = trap_energy + p * phonon_energy
    f = 1.0 / (1.0 + np.exp((energies - fermi_level) / (KB * T)))
    dos = np.sqrt(energies)  # 任意（正）态密度，比值与之无关
    P = np.exp(-energies / EV)  # 任意（正）透射系数
    coupling = multiphonon_coupling(0.17e-9, 2.5e8, 5.4 * EV, M_DI)

    r4 = r4_multiphonon_capture(coupling, dos, f, P, L)
    r5 = r5_multiphonon_emission(coupling, dos, f, P, L, p, phonon_energy, T)
    assert r5 / r4 == pytest.approx(np.exp((trap_energy - fermi_level) / (KB * T)), rel=1e-10)


def test_multiphonon_coupling_scaling():
    """c_MP ∝ r_D³·F²，F=0 时多声子跃迁关闭。"""
    base = multiphonon_coupling(0.2e-9, 1e8, 5.4 * EV, M_DI)
    assert multiphonon_coupling(0.4e-9, 1e8, 5.4 * EV, M_DI) == pytest.approx(8.0 * base)
    assert multiphonon_coupling(0.2e-9, 2e8, 5.4 * EV, M_DI) == pytest.approx(4.0 * base)
    assert multiphonon_coupling(0.2e-9, 0.0, 5.4 * EV, M_DI) == 0.0


def test_phonon_weights_poisson_limit_at_low_temperature():
    """低温极限 n̄→0 时 L_p 退化为泊松分布 S^p·e^{-S}/p!。"""
    S, phonon_energy, low_T = 1.0, 0.04 * EV, 60.0
    p = np.arange(6)
    L = phonon_transition_probability(p, S, phonon_energy, low_T)
    from math import factorial

    poisson = np.array([S ** k * np.exp(-S) / factorial(k) for k in p])
    assert np.allclose(L, poisson, rtol=2e-2)


def test_phonon_weights_are_positive_and_subnormalized():
    """只对 p ≥ 0 求和，总和必须落在 (0,1)：其余归于声子吸收分支 p<0。"""
    for temperature in (273.15, 398.15):
        L = phonon_transition_probability(np.arange(40), 1.0, 0.04 * EV, temperature)
        assert np.all(L > 0.0)
        assert 0.5 < L.sum() < 1.0
        assert L[12:].sum() / L.sum() < 1e-4  # 截断到 p=11 足够


# ----------------------------------------------------------------------
# R6 / R7
# ----------------------------------------------------------------------
def test_r6_symmetry_and_exponential_decay():
    r_D = np.array([0.17e-9, 0.25e-9])
    r_ab = 1.0e-9
    forward = r6_elastic_hop(NU0, r_ab, r_D[0], r_D[1])
    backward = r6_elastic_hop(NU0, r_ab, r_D[1], r_D[0])
    assert forward == pytest.approx(backward)
    assert r6_elastic_hop(NU0, 0.0, r_D[0], r_D[1]) == pytest.approx(NU0)
    doubled = r6_elastic_hop(NU0, 2.0 * r_ab, r_D[0], r_D[1])
    assert doubled == pytest.approx(forward ** 2 / NU0)


def test_r7_is_r6_downhill_and_boltzmann_uphill():
    r_D = 0.17e-9
    r_ab = 0.8e-9
    r6 = r6_elastic_hop(NU0, r_ab, r_D, r_D)
    E_a, E_b = 0.5 * EV, 0.6 * EV
    downhill = r7_inelastic_hop(NU0, r_ab, r_D, r_D, E_b, E_a, T)
    uphill = r7_inelastic_hop(NU0, r_ab, r_D, r_D, E_a, E_b, T)
    assert downhill == pytest.approx(r6)  # 下坡不受能量惩罚，R7 ≡ R6
    assert uphill == pytest.approx(r6 * np.exp(-(E_b - E_a) / (KB * T)))
    assert uphill < downhill
    # 细致平衡：正反跃迁之比只由能量差决定
    assert uphill / downhill == pytest.approx(np.exp(-(E_b - E_a) / (KB * T)))


def test_miller_abrahams_factor_bounds():
    assert miller_abrahams_factor(1.0 * EV, 1.0 * EV, T) == pytest.approx(1.0)
    assert miller_abrahams_factor(1.0 * EV, 0.5 * EV, T) == pytest.approx(1.0)
    assert 0.0 < miller_abrahams_factor(0.5 * EV, 1.0 * EV, T) < 1.0


# ----------------------------------------------------------------------
# R8
# ----------------------------------------------------------------------
def test_r8_zero_field_limit():
    """F→0 时 β→0，花括号项→1，R8 退化为无场热发射 ν·exp(-ED/k_BT)。

    并检验展开的一阶修正：花括号项 = 1 + β/3 + O(β²)。
    """
    expected = NU0 * np.exp(-ED / (KB * T))
    assert r8_pf_emission(NU0, ED, 0.0, 1.0, EPS_OPT, T) == pytest.approx(expected, rel=1e-12)
    for field in (1.0, 1.0e-2, 1.0e-4):
        beta = pf_beta(field, 1.0, EPS_OPT, T)
        ratio = r8_pf_emission(NU0, ED, field, 1.0, EPS_OPT, T) / expected
        assert ratio - 1.0 == pytest.approx(beta / 3.0, rel=1e-3)


def test_r8_bracket_series_matches_direct_evaluation():
    """小 β 的级数展开与直接求值在阈值附近必须衔接。"""
    beta = np.array([9.0e-4, 1.1e-3])
    direct = (1.0 + (beta - 1.0) * np.exp(beta)) / beta ** 2 + 0.5
    assert np.allclose(pf_bracket(beta), direct, rtol=1e-6)


def test_r8_beta_is_barrier_lowering_over_kT():
    """β·k_BT = √(Z·e³·F/(π·ε0·ε_opt))，即 PF 场致势垒降低量 Δ。"""
    F, Z = 3.0e8, 1.0
    beta = pf_beta(F, Z, EPS_OPT, T)
    delta = np.sqrt(
        Z * constants.elementary_charge ** 3 * F / (np.pi * constants.epsilon_0 * EPS_OPT)
    )
    assert beta * KB * T == pytest.approx(delta, rel=1e-12)


def test_r8_high_field_limit_approaches_exponential_pf():
    """β≫1 时 R8 → ν·exp[-(ED-Δ)/k_BT]/β，相对偏差按 1/β 收敛。"""
    deviations = []
    for beta_target in (10.0, 20.0, 40.0):
        # 由目标 β 反推所需场强
        F = (
            (beta_target * KB * T) ** 2
            * np.pi
            * constants.epsilon_0
            * EPS_OPT
            / constants.elementary_charge ** 3
        )
        beta = pf_beta(F, 1.0, EPS_OPT, T)
        delta = beta * KB * T
        asymptote = NU0 * np.exp(-(ED - delta) / (KB * T)) / beta
        ratio = r8_pf_emission(NU0, ED, F, 1.0, EPS_OPT, T) / asymptote
        deviations.append(abs(ratio - 1.0))
        assert abs(ratio - 1.0) < 1.5 / beta
    assert deviations[0] > deviations[1] > deviations[2]


def test_r8_uses_optical_permittivity_only():
    """R8 的场依赖只经 ε_opt 进入：换 ε_opt 结果变，与静态 κ 无关。"""
    F = 2.5e8
    with_opt = r8_pf_emission(NU0, ED, F, 1.0, EPS_OPT, T)
    with_static = r8_pf_emission(NU0, ED, F, 1.0, 38.0, T)
    assert with_opt > with_static  # ε 越大势垒降低越小
    assert not np.isclose(with_opt, with_static)
