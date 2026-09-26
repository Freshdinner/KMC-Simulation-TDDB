"""小状态空间主方程稳态解——条件通量估计量的独立算法基准（ADR-021 F04）。

这一批把 review F04 的反例与一致极限钉成测试：
- 复现 review 附录的 4 态反例：条件通量比真实稳态低 25%（局部稳态近似不严格无偏）；
- 单陷阱（无耦合）时条件通量 = 真实稳态（p̂ 对两态系统精确）；
- 稳态分布 π 是合法概率分布、满足 πQ=0；
- 长 KMC 轨迹的条件通量估计量收敛到主方程给出的同一期望（引擎与独立基准互证）。
"""

import numpy as np
import pytest
from scipy import constants

from conftest import make_config, make_material
from kmc_leakage.config import DeviceConfig, DielectricStack, ElectrodeMaterial
from kmc_leakage.defects import DefectEnsemble
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.master_equation import (
    enumerate_occupations,
    steady_state_flux,
    stationary_distribution,
    transition_generator,
)

EV = constants.electron_volt
M_E = constants.electron_mass
Q = constants.elementary_charge


def test_review_four_state_counterexample_reproduced():
    """复现 review F04 附录：条件通量比真实稳态阳极通量低 25%（0.3 vs 0.4）。

    两陷阱、强正向驱动极限，三条通道速率均 1/s：阴极→陷阱0、陷阱0→陷阱1、
    陷阱1→阳极。手搭生成矩阵解稳态，直接调 `KMCSimulation._conditional_flux`，
    证明条件稳态近似在耦合网络里存在系统偏差——这正是 F04 要求独立验证的原因。
    """
    states = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    generator = np.zeros((4, 4))
    for source, target in [(0, 1), (1, 2), (2, 0), (2, 3), (3, 1)]:
        generator[source, target] = 1.0
    generator[np.diag_indices(4)] = -generator.sum(axis=1)
    pi = stationary_distribution(generator)

    assert pi == pytest.approx([0.2, 0.4, 0.2, 0.2], abs=1e-12)

    blocks = np.zeros((9, 2))
    blocks[0, 0] = 1.0  # R2C：阴极向陷阱 0 注入
    blocks[5, 1] = 1.0  # R3A：陷阱 1 向阳极逃逸

    conditional = []
    for occ_a, occ_b in states:
        hop_in = np.array([0.0, float(occ_a)])
        hop_out = np.array([1.0 - float(occ_b), 0.0])
        tat, pf, _ = KMCSimulation._conditional_flux(blocks, hop_in, hop_out)
        conditional.append(float(np.sum(tat + pf)))

    exact_flux = float(pi @ states[:, 1])  # 真实阳极通量 = 陷阱1 的平均占据 × 1/s
    conditional_flux = float(pi @ np.array(conditional))
    assert exact_flux == pytest.approx(0.4, abs=1e-12)
    assert conditional_flux == pytest.approx(0.3, abs=1e-12)
    assert conditional_flux / exact_flux == pytest.approx(0.75, abs=1e-12)  # 低 25%


def _two_trap_sim(voltage=3.0, n_events=200, seed=3, z=(3e-9, 6e-9), ED=(1.0, 1.15)):
    """单层器件 + 两个体缺陷的小体系（2^2=4 个组态，主方程可解）。"""
    material = make_material()
    stack = DielectricStack.single(material, thickness=9e-9)
    electrode = ElectrodeMaterial("TiN", 4.6 * EV, M_E, 5.0 * EV)
    device = DeviceConfig(20e-9, 20e-9, electrode, electrode, stack, barrier_height=1.74 * EV)
    positions = np.array([[10e-9, 10e-9, z[0]], [10e-9, 10e-9, z[1]]])
    ensemble = DefectEnsemble.uniform(positions, np.array(ED) * EV, material, band_reference=1.74 * EV)
    config = make_config(voltage=voltage, n_events=n_events, seed=seed, device=device)
    return KMCSimulation(config, defects=ensemble)


def test_stationary_distribution_is_valid_and_solves_generator():
    """π 是合法概率分布（≥0、Σ=1）且满足 πQ=0。"""
    sim = _two_trap_sim()
    occs, Q = transition_generator(sim)
    assert occs.shape == (4, 2)
    pi = stationary_distribution(Q)
    assert np.all(pi >= -1e-15)
    assert pi.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(pi @ Q, 0.0, atol=1e-12)


def test_single_trap_conditional_flux_equals_exact():
    """单陷阱（无耦合）：两态系统的 p̂ 精确，条件通量 = 真实稳态（rtol 1e-12）。"""
    material = make_material()
    stack = DielectricStack.single(material, thickness=9e-9)
    electrode = ElectrodeMaterial("TiN", 4.6 * EV, M_E, 5.0 * EV)
    device = DeviceConfig(20e-9, 20e-9, electrode, electrode, stack, barrier_height=1.74 * EV)
    ensemble = DefectEnsemble.uniform(
        np.array([[10e-9, 10e-9, 4.5e-9]]), np.array([1.1 * EV]), material, band_reference=1.74 * EV
    )
    sim = KMCSimulation(make_config(voltage=3.0, n_events=10, device=device), defects=ensemble)
    flux = steady_state_flux(sim)
    assert flux["exact_flux"] == pytest.approx(flux["conditional_flux"], rel=1e-12)


def test_enumerate_occupations_covers_all_states():
    occs = enumerate_occupations(3)
    assert occs.shape == (8, 3)
    # 每个组态互异
    assert len({tuple(row) for row in occs.tolist()}) == 8


def test_kmc_conditional_flux_converges_to_master_equation():
    """长 KMC 轨迹的条件通量估计量收敛到主方程的同一期望（引擎 × 独立基准互证）。

    KMC 的 `current`（陷阱部分，去掉 R1）随事件数趋于 Σ_s π_s·条件通量_i(s)，正是
    主方程算出的 conditional_current。低方差，取足够事件即可收敛。
    """
    sim = _two_trap_sim(voltage=3.0, n_events=40000, seed=11)
    flux = steady_state_flux(sim)
    result = sim.run()
    r1 = result.by_mechanism["direct_fn"] + result.by_mechanism["thermionic"]
    trap_current = result.current - r1
    assert flux["conditional_current"] > 0.0
    assert trap_current == pytest.approx(flux["conditional_current"], rel=0.1)
