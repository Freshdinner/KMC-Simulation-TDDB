"""TDDB 功率耗散与稳态热流（Phase 4，式 T13/T14，ADR-036，PROJECT_CONTEXT_TDDB.md §9）。

- 热流（T14）：均匀功率 + 两端恒温下 T 剖面与解析抛物线一致；零功率退回环境温度；
- 功率（T13）按 CIC 分到网格，积分等于各缺陷放热之和；
- 放热记账（T13）的能量守恒：介质 + 两侧电极的放热 = U·I_陷阱。孤立陷阱（逐陷阱平均场）、
  整个体系就是一个强耦合簇（主方程精确稳态）、强耦合簇与簇外陷阱混合、强耦合分量被切开
  四种情形都严格成立（ADR-038）；
- 零偏压热平衡下没有净放热（细致平衡），只剩 R8 无反向通道的极小项；
- 输运温度是缺陷处温度按放热加权的平均，没有放热时就是环境温度。
"""

import numpy as np
import pytest
from scipy import constants

from conftest import make_config, make_defect_config, make_device
from test_fast_traps import _three_trap_sim
from test_master_equation import _two_trap_sim
from kmc_leakage.bands import fermi_dirac
from kmc_leakage.config import GridConfig
from kmc_leakage.electrostatics import PoissonSolver
from kmc_leakage.kmc import _B_R4A, _B_R4C, _B_R8, KMCSimulation
from kmc_leakage.mean_field import solve_mean_field
from kmc_leakage.thermal import heat_release, power_map, solve_temperature, transport_temperature

K_TH = 1.0  # W/(m·K)
T_AMB = 298.15


def _solver(grid=GridConfig(8, 8, 40)):
    return PoissonSolver(make_device(), grid)


def _trap_current(result):
    return result.by_mechanism["tat"] + result.by_mechanism["pf"]


def test_uniform_power_gives_parabolic_profile():
    """均匀体功率 P、两电极恒温：T(z) = T_amb + P·z·(L − z)/(2·k_TH)，与 xy 无关。

    二阶差分对二次函数精确，所以离散解在节点上逐点等于解析抛物线（到舍入误差）。
    """
    solver = _solver()
    power = 3.0e15  # W/m^3
    temperature = solve_temperature(solver, np.full(solver.grid.shape, power), K_TH, T_AMB)
    z, L = solver.z, solver.device.Lz
    expected = T_AMB + power * z * (L - z) / (2.0 * K_TH)
    np.testing.assert_allclose(temperature, np.broadcast_to(expected, temperature.shape), rtol=1e-12)
    assert temperature[..., [0, -1]] == pytest.approx(T_AMB)


def test_zero_power_returns_ambient_temperature():
    solver = _solver()
    temperature = solve_temperature(solver, np.zeros(solver.grid.shape), K_TH, T_AMB)
    assert np.all(temperature == T_AMB)


def test_power_map_conserves_total_power():
    """CIC 把各缺陷的放热分到网格，体功率密度的积分等于放热之和（缺陷离电极足够远）。"""
    solver = _solver()
    rng = np.random.default_rng(3)
    positions = np.column_stack([rng.uniform(0, 20e-9, 6), rng.uniform(0, 20e-9, 6), rng.uniform(1e-9, 7e-9, 6)])
    heat = rng.uniform(-1e-12, 3e-12, 6)
    density = power_map(solver, positions, heat)
    assert density.sum() * solver.hx * solver.hy * solver.hz == pytest.approx(heat.sum(), rel=1e-12)


@pytest.mark.parametrize(
    "build",
    [
        lambda: _two_trap_sim(z=(3e-9, 3.9e-9)),  # 一个强耦合簇覆盖全部陷阱
        lambda: KMCSimulation(make_config(voltage=2.0, seed=2)),  # 没有强耦合簇
        lambda: KMCSimulation(make_config(voltage=3.0, seed=3)),
    ],
    ids=["single-cluster", "per-trap-2V", "per-trap-3V"],
)
def test_heat_release_conserves_energy(build):
    """介质 + 两侧电极的放热 = U·I_陷阱：电子从阴极费米能级走到阳极费米能级，每个放掉 qU。

    逐陷阱平均场里每个陷阱在不动点处粒子数平衡，能级项逐陷阱相消；强耦合簇内由主方程稳态
    分布保证能量平衡。两者都严格成立，剩下的只是平均场收敛残差（相对 1e-10）。
    """
    sim = build()
    assert (len(sim.cluster_members) > 0) == (sim.n_defects == 2)
    result = solve_mean_field(sim)
    heat = heat_release(sim, result.occupancy)
    total = heat.dielectric + heat.cathode + heat.anode
    assert total == pytest.approx(sim.sim.voltage * _trap_current(result), rel=1e-8)


@pytest.mark.parametrize(
    "build",
    [
        _three_trap_sim,  # 一个簇 + 一个簇外陷阱
        lambda: KMCSimulation(  # 3×10¹⁹ cm⁻³：17 个簇，没有被切开的分量
            make_config(voltage=3.0, seed=1, device=make_device(defects=make_defect_config(concentration_cm3=3.0e19)))
        ),
        lambda: KMCSimulation(  # 1×10²⁰ cm⁻³、10×10 nm：60 个陷阱的强耦合分量被切开，有簇–簇边界
            make_config(
                voltage=3.0,
                seed=1,
                device=make_device(Lx=10e-9, Ly=10e-9, defects=make_defect_config(concentration_cm3=1.0e20)),
            )
        ),
    ],
    ids=["cluster+free", "17-clusters", "split-component"],
)
def test_heat_release_with_clusters_and_environment_conserves_energy(build):
    """强耦合簇与簇外陷阱混合、强耦合分量被切开时，能量守恒同样严格成立（ADR-038）：每条陷阱间跃迁的
    速率只算一次、两端共用，放热按两端各自分布下的能级记账，不动点处每个陷阱粒子数守恒。原来簇一侧
    按组态、簇外一侧按平均占据各算一遍边界通量，Phase 4 终态上放热总和是 U·I 的 3–14 倍（ADR-037）。"""
    sim = build()
    assert len(sim.cluster_members) > 0
    result = solve_mean_field(sim)
    heat = heat_release(sim, result.occupancy)
    total = heat.dielectric + heat.cathode + heat.anode
    assert total == pytest.approx(sim.sim.voltage * _trap_current(result), rel=1e-8)


def test_no_net_phonon_heat_in_zero_bias_equilibrium():
    """零偏压、热平衡占据 p = f(E)：R4/R5 逐声子项满足细致平衡，陷阱处俘获放出的声子能量与
    逃逸吸收的相消，剩下的只有 R8 的吸热 p·R8·E_D。

    占据直接取费米分布而不取平均场不动点：R8 没有反向通道，会把不动点占据推离平衡（相对
    1e-6 量级），那部分放热是真实的非平衡效应。两个陷阱相距 4.5 nm，在跃迁截断半径之外；局域
    场来自另一个陷阱的电荷。
    """
    sim = _two_trap_sim(voltage=0.0, z=(2e-9, 6.5e-9))
    assert len(sim._pairs) == 0
    energy = sim._trap_state(solve_mean_field(sim).occupancy)[0]
    p = fermi_dirac(energy, 0.0, sim.sim.temperature)  # 零偏压下两侧电极费米能级都在 0
    heat = heat_release(sim, p)
    energy, field, barrier_c, barrier_a = sim._trap_state(p)
    blocks = sim._electrode_blocks(energy, field, barrier_c, barrier_a)
    phonon = sim._electrode_blocks(
        energy, field, barrier_c, barrier_a, weights=sim._phonon_weights * sim._phonon_offsets
    )
    one_way = (1.0 - p) * (phonon[_B_R4C] + phonon[_B_R4A])
    assert np.all(one_way > 0.0)
    pf_absorbed = p * blocks[_B_R8] * sim.defects.ED
    np.testing.assert_allclose(heat.defect + pf_absorbed, 0.0, atol=1e-9 * one_way.min())


def test_transport_temperature_is_heat_weighted_mean_at_defects():
    """缺陷处温度按放热加权（只取放热部分），介质里没有放热时就是环境温度。"""
    solver = _solver()
    positions = np.array([[5e-9, 5e-9, 2e-9], [15e-9, 12e-9, 6e-9], [9e-9, 3e-9, 4e-9]])
    field = T_AMB + np.random.default_rng(0).uniform(0.0, 30.0, solver.grid.shape)
    at_defects = solver.interpolate(field, positions)
    heat = np.array([2e-12, 1e-12, -5e-12])
    expected = (2.0 * at_defects[0] + 1.0 * at_defects[1]) / 3.0
    assert transport_temperature(solver, field, positions, heat, T_AMB) == pytest.approx(expected, rel=1e-14)
    assert transport_temperature(solver, field, positions, np.array([0.0, 0.0, -1e-12]), T_AMB) == T_AMB
    assert transport_temperature(solver, field, np.zeros((0, 3)), np.zeros(0), T_AMB) == T_AMB


def test_point_source_heats_its_neighbourhood_and_electrodes_stay_ambient():
    """一个放热缺陷：温度最高处在它附近，离它越远越低，两电极节点恒为环境温度。"""
    solver = _solver(GridConfig(16, 16, 40))
    position = np.array([[10e-9, 10e-9, 4e-9]])
    temperature = solve_temperature(solver, power_map(solver, position, np.array([1e-9])), K_TH, T_AMB)
    assert np.all(temperature[..., 0] == T_AMB) and np.all(temperature[..., -1] == T_AMB)
    assert np.all(temperature >= T_AMB)
    near = solver.interpolate(temperature, position)[0]
    far = solver.interpolate(temperature, np.array([[10e-9, 10e-9, 1e-9], [0.0, 0.0, 4e-9]]))
    assert near > 0.99 * (temperature.max() - T_AMB) + T_AMB
    assert np.all(far < near)
