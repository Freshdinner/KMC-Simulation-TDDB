"""TDDB 缺陷生成速率（式 T1–T4，rates/generation.py）的解析极限与口径检验。

对应 PROJECT_CONTEXT_TDDB.md 第 9 节的三条：
1. G 的零场极限退化为 R_1e·exp(−E_A/k_BT)；
2. 场加速项线性压低势垒，log G 对 E 成线性（有效势垒降到 0 以下时 P_bb 饱和为 1）；
3. R_1e 的电极项等于同一前驱参数下两侧电极的 R2 + R4 之和，与输运引擎口径一致；
另加 R_1e 的陷阱项（ADR-039）：每个来源是 p·R7(陷阱 → 前驱)，源陷阱空着时为 0；逐前驱位点对近旁陷阱的求和见
test_tddb.py（ADR-052）。
"""

import numpy as np
import pytest
from scipy import constants

from conftest import make_config, make_material
from kmc_leakage.bands import conduction_band_edge
from kmc_leakage.config import DeviceConfig, DielectricStack, ElectrodeMaterial
from kmc_leakage.defects import DefectEnsemble
from kmc_leakage.kmc import _B_R2A, _B_R2C, _B_R4A, _B_R4C, _N_PHONON, KMCSimulation
from kmc_leakage.rates import (
    CaptureElectrode,
    bond_break_generation_rate,
    bond_break_probability,
    effective_activation_energy,
    single_electron_capture_parts,
    single_electron_capture_rate,
    trap_precursor_capture_rate,
)

EV = constants.electron_volt
M_E = constants.electron_mass
KB = constants.Boltzmann
Q = constants.elementary_charge
E_ANGSTROM = constants.elementary_charge * constants.angstrom  # 1 e·Å [C·m]

E_A = 4.0 * EV
KAPPA = 38.0
T = 298.15


def test_generation_rate_zero_field_limit():
    """E = 0 时场加速项消失，G = R_1e·exp(−E_A/k_BT)（式 T1/T2）。"""
    capture = np.array([1.0e3, 2.5e7])
    G = bond_break_generation_rate(capture, E_A, 3.0 * E_ANGSTROM, KAPPA, np.zeros(2), T)
    assert G == pytest.approx(capture * np.exp(-E_A / (KB * T)), rel=1e-12)


def test_log_generation_rate_is_linear_in_field():
    """有效势垒为正的范围内 ln G 对 E 严格线性，斜率 p₀·(2+κ)/(3·k_BT)。"""
    p0 = 5.0 * E_ANGSTROM
    field = np.linspace(1.0e8, 5.0e8, 9)
    assert np.all(effective_activation_energy(E_A, p0, KAPPA, field) > 0.0)
    log_G = np.log(bond_break_generation_rate(np.full(field.size, 1.0e5), E_A, p0, KAPPA, field, T))
    slope = np.diff(log_G) / np.diff(field)
    assert slope == pytest.approx(np.full(slope.size, p0 * (2.0 + KAPPA) / (3.0 * KB * T)), rel=1e-9)


def test_bond_break_probability_saturates_when_barrier_vanishes():
    """有效势垒 ≤ 0 时 P_bb = 1，G 只剩俘获速率（§8.1：3 V/8 nm 下 p₀ ≥ 8 eÅ）。"""
    field = np.array([3.0 / 8e-9])
    assert effective_activation_energy(E_A, 8.0 * E_ANGSTROM, KAPPA, field) == pytest.approx(0.0, abs=1e-30)
    assert bond_break_probability(E_A, 9.0 * E_ANGSTROM, KAPPA, field, T) == pytest.approx(1.0)
    G = bond_break_generation_rate(np.array([4.2e6]), E_A, 9.0 * E_ANGSTROM, KAPPA, field, T)
    assert G == pytest.approx([4.2e6], rel=1e-15)


def test_field_lowering_uses_static_kappa_lorentz_factor():
    """场加速项的 k 是静态 κ：κ=38 时 (2+κ)/3 = 40/3，3 V/8 nm 下每 1 eÅ 压低 0.5 eV（§8.1）。"""
    lowering = E_A - effective_activation_energy(E_A, 1.0 * E_ANGSTROM, KAPPA, np.array([3.0 / 8e-9]))
    assert lowering == pytest.approx([0.5 * EV], rel=1e-12)


def _precursor_probe(ionization_energy, relaxation_energy, voltage):
    """把前驱参数当成真实陷阱放进引擎：单层 ZrO₂ 参考预设、三个位置、全空组态。

    引擎侧陷阱的 Huang-Rhys 因子取 E_rel1/ℏω₀，只是为了让两边的多声子线型相同、逐项
    对拍，不代表 E_rel1 与输运的 S 有关联（ADR-030）。
    """
    phonon_energy = make_material().phonon_energy
    material = make_material(huang_rhys=relaxation_energy / phonon_energy)
    stack = DielectricStack.single(material, thickness=8e-9)
    electrode = ElectrodeMaterial("TiN", 4.6 * EV, M_E, 5.0 * EV)
    device = DeviceConfig(20e-9, 20e-9, electrode, electrode, stack, barrier_height=1.74 * EV)
    positions = np.array([[4e-9, 4e-9, 2e-9], [10e-9, 10e-9, 4e-9], [16e-9, 16e-9, 6e-9]])
    ensemble = DefectEnsemble.uniform(
        positions, np.full(3, ionization_energy), material, band_reference=1.74 * EV
    )
    sim = KMCSimulation(make_config(voltage=voltage, n_events=10, device=device), defects=ensemble)
    return sim, material


@pytest.mark.parametrize("voltage", [3.0, -3.0])
def test_single_electron_capture_rate_matches_engine_r2_plus_r4(voltage):
    """R_1e = 两侧电极的 R2 + R4 之和：前驱参数当成真实陷阱时与引擎的块速率逐项一致。

    前驱取 E_t1 = 1.6 eV、E_rel1 = 0.12 eV（E_rel1/ℏω₀ = 3），正反偏压各测一次，两侧
    电极都占主导的情形都覆盖到。能级、局域场、WKB 路径与势垒全部取自引擎的同一组态。
    """
    ionization_energy, relaxation_energy = 1.6 * EV, 0.12 * EV
    sim, material = _precursor_probe(ionization_energy, relaxation_energy, voltage)
    device = sim.device
    occ = np.zeros(len(sim.defects), dtype=bool)
    blocks, _ = sim._rates(occ)
    phi_at, field_at, phi_bar = sim._electrostatics(occ)
    barrier_c, barrier_a = sim._path_barriers(conduction_band_edge(device, sim._z_nodes, phi_bar))
    electrodes = (
        CaptureElectrode(
            sim._path_negative, barrier_c, sim._m_di_path_negative,
            sim._fermi_negative, sim._band_bottom[0], device.negative.effective_mass,
        ),
        CaptureElectrode(
            sim._path_positive, barrier_a, sim._m_di_path_positive,
            sim._fermi_positive, sim._band_bottom[1], device.positive.effective_mass,
        ),
    )

    r_1e = single_electron_capture_rate(
        sim.defects.band_reference - Q * phi_at,
        field_at,
        electrodes,
        ionization_energy,
        relaxation_energy,
        material.phonon_energy,
        material.band_gap,
        material.m_di,
        sim.sim.temperature,
        _N_PHONON,
    )
    expected = blocks[_B_R2C] + blocks[_B_R2A] + blocks[_B_R4C] + blocks[_B_R4A]
    assert np.all(expected > 0.0)
    assert np.all(blocks[_B_R4C] + blocks[_B_R4A] > 0.0)  # 多声子通道确实参与
    assert r_1e == pytest.approx(expected, rel=1e-12)
    # 按场依赖拆开的两部分（ADR-044）：弹性部分 = 两侧 R2，F² × 多声子部分 = 两侧 R4
    elastic, multiphonon = single_electron_capture_parts(
        sim.defects.band_reference - Q * phi_at,
        electrodes,
        ionization_energy,
        relaxation_energy,
        material.phonon_energy,
        material.band_gap,
        material.m_di,
        sim.sim.temperature,
        _N_PHONON,
    )
    assert elastic == pytest.approx(blocks[_B_R2C] + blocks[_B_R2A], rel=1e-12)
    assert field_at ** 2 * multiphonon == pytest.approx(blocks[_B_R4C] + blocks[_B_R4A], rel=1e-12)


def test_trap_term_is_occupancy_weighted_miller_abrahams_hop():
    """R_1e 的陷阱项 = p·R7(陷阱 → 前驱)：下坡（前驱能级不高于陷阱）只剩 R6 的几何因子
    p·ν0·exp[−r·(1/r_D,陷阱 + 1/r_D,前驱)]，上坡再乘 exp(−ΔE/k_BT)；源陷阱空着（p = 0）时为 0（ADR-039）。"""
    nu, r = 1.0e13, np.array([0.6e-9, 0.6e-9, 0.9e-9, 0.6e-9])
    r_trap, r_precursor = np.array([0.15e-9, 0.15e-9, 0.17e-9, 0.15e-9]), 0.166e-9
    trap_energy = np.array([0.1, 0.1, -0.2, 0.1]) * EV
    precursor_energy = np.array([-0.05, 0.25, -0.2, 0.25]) * EV
    p = np.array([0.7, 0.7, 1.0, 0.0])
    rate = trap_precursor_capture_rate(np.full(4, nu), r, r_trap, r_precursor, trap_energy, precursor_energy, p, T)
    geometric = nu * np.exp(-r / r_trap - r / r_precursor)
    uphill = np.exp(-(precursor_energy - trap_energy) / (KB * T))
    expected = p * geometric * np.array([1.0, uphill[1], 1.0, uphill[3]])
    assert rate == pytest.approx(expected, rel=1e-12)
    assert rate[3] == 0.0
