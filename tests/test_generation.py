"""TDDB 缺陷生成速率（式 T1–T4，rates/generation.py）的解析极限与口径检验。

对应 PROJECT_CONTEXT_TDDB.md 第 9 节的三条：
1. G 的零场极限退化为 R_1e·exp(−E_A/k_BT)；
2. 场加速项线性压低势垒，log G 对 E 成线性（有效势垒降到 0 以下时 P_bb 饱和为 1）；
3. R_1e 等于同一前驱参数下两侧电极的 R2 + R4 之和，与输运引擎口径一致。
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
    single_electron_capture_rate,
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


def _precursor_probe(ionization_energy, huang_rhys, voltage):
    """把前驱参数当成真实陷阱放进引擎：单层 ZrO₂ 参考预设、三个位置、全空组态。"""
    material = make_material(huang_rhys=huang_rhys)
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

    前驱取 E_t1 = 1.6 eV、S₁ = 3（都不同于 O 空位预设），正反偏压各测一次，两侧电极
    都占主导的情形都覆盖到。能级、局域场、WKB 路径与势垒全部取自引擎的同一组态。
    """
    ionization_energy, huang_rhys = 1.6 * EV, 3.0
    sim, material = _precursor_probe(ionization_energy, huang_rhys, voltage)
    device = sim.device
    occ = np.zeros(len(sim.defects), dtype=bool)
    blocks, _ = sim._rates(occ)
    phi_at, field_at, phi_bar = sim._electrostatics(occ)
    barrier_c, barrier_a = sim._path_barriers(conduction_band_edge(device, sim._z_nodes, phi_bar))
    electrodes = (
        CaptureElectrode(
            sim._path_cathode, barrier_c, sim._m_di_path_cathode,
            sim._fermi_cathode, sim._band_bottom[0], device.cathode.effective_mass,
        ),
        CaptureElectrode(
            sim._path_anode, barrier_a, sim._m_di_path_anode,
            sim._fermi_anode, sim._band_bottom[1], device.anode.effective_mass,
        ),
    )

    r_1e = single_electron_capture_rate(
        sim.defects.band_reference - Q * phi_at,
        field_at,
        electrodes,
        ionization_energy,
        huang_rhys * material.phonon_energy,
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
