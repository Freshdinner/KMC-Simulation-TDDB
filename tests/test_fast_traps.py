"""快陷阱移出事件预算（ADR-023，kmc 模块文档说明 6）。

- 没有陷阱达到 R_cut 时与不配置逐位相同（回归锚点）；
- 三陷阱小体系：贴负极的快陷阱退出事件表后，电流仍等于主方程精确稳态（快陷阱照样
  进强耦合簇），测量窗口延长 8 个数量级；
- 快陷阱与慢陷阱之间的跃迁仍逐事件记账，全局守恒改为
  (I_C−I_A)·Δt = q·(ΔN_inside − fast_exchange)，逐事件精确成立；
- 全部陷阱都是快陷阱时没有事件可推进，判未通过验证；时间尺度分离不足时同样不通过。
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy import constants

from conftest import make_config, make_material
from kmc_leakage.config import DeviceConfig, DielectricStack, ElectrodeMaterial
from kmc_leakage.defects import DefectEnsemble
from kmc_leakage.kmc import _FAST_SEPARATION_MIN, KMCSimulation, _validate_result
from kmc_leakage.master_equation import steady_state_flux

EV = constants.electron_volt
M_E = constants.electron_mass
Q = constants.elementary_charge


def _with_fast_rate(config, rate):
    return replace(config, simulation=replace(config.simulation, fast_trap_rate=rate))


def _three_trap_sim(fast_trap_rate=None, n_events=3000, seed=1):
    """9 nm 单层 + 三个陷阱：#0 贴负极（Γ≈4e12 s⁻¹）、#1 在它旁边 0.8 nm（Γ≈2e8）、#2 在膜中部。

    #0 与 #1 之间的跃迁 5.5e8 s⁻¹，两者连成一个强耦合簇；2^3 个组态，主方程可解。
    """
    material = make_material()
    stack = DielectricStack.single(material, thickness=9e-9)
    electrode = ElectrodeMaterial("TiN", 4.6 * EV, M_E, 5.0 * EV)
    device = DeviceConfig(20e-9, 20e-9, electrode, electrode, stack, barrier_height=1.74 * EV)
    positions = np.array([[10e-9, 10e-9, 0.7e-9], [10.4e-9, 10e-9, 1.4e-9], [10e-9, 10e-9, 4.0e-9]])
    ensemble = DefectEnsemble.uniform(
        positions, np.array([1.15, 1.2, 1.0]) * EV, material, band_reference=1.74 * EV
    )
    config = make_config(voltage=3.0, n_events=n_events, seed=seed, device=device)
    return KMCSimulation(_with_fast_rate(config, fast_trap_rate), defects=ensemble)


def _trap_current(result):
    return result.current - (result.by_mechanism["direct_fn"] + result.by_mechanism["thermionic"])


def test_unreached_fast_trap_rate_is_bitwise_identical():
    """R_cut 高到没有陷阱达到：与不配置逐位相同（电流、逐缺陷分解、占据、交叉估计量）。"""
    config = make_config(voltage=3.0, n_events=800, seed=5)
    base = KMCSimulation(config).run()
    same = KMCSimulation(_with_fast_rate(config, 1e30)).run()
    assert same.n_fast_traps == 0
    assert same.current == base.current
    assert np.array_equal(same.by_defect, base.by_defect)
    assert np.array_equal(same.occupancy, base.occupancy)
    assert same.current_direct == base.current_direct
    assert same.current_counts_total == base.current_counts_total


@pytest.mark.parametrize("rate, n_fast", [(1e9, 1), (1e7, 2)])
def test_fast_traps_keep_exact_current_and_extend_window(rate, n_fast):
    """快陷阱退出事件表后，陷阱电流仍等于主方程精确稳态（rtol 1e-4）。

    R_cut=1e7 时 #0、#1 都退出，剩下的事件只有 #2 的慢过程，窗口从 ~4e-5 s 延长到
    ~2e4 s。快陷阱照样作为簇成员精确枚举，近似只进入慢陷阱感受到的平均电荷。
    """
    exact = steady_state_flux(_three_trap_sim())["exact_current"]
    base = _three_trap_sim().run()
    result = _three_trap_sim(fast_trap_rate=rate).run()
    assert result.n_fast_traps == n_fast
    assert result.validated
    assert _trap_current(result) == pytest.approx(exact, rel=1e-4)
    if n_fast == 2:
        assert result.elapsed_time > 1e6 * base.elapsed_time


def test_global_conservation_includes_fast_exchange():
    """快↔慢跃迁不经过电极：(I_C−I_A)·Δt = q·(ΔN_inside − fast_exchange)，逐事件精确。"""
    result = _three_trap_sim(fast_trap_rate=1e7).run()
    assert result.fast_exchange != 0
    lhs = (result.current_counts_negative - result.current_counts) * result.elapsed_time
    assert lhs == pytest.approx(Q * (result.delta_n_inside - result.fast_exchange), rel=1e-9, abs=1e-30)


def test_all_traps_fast_leaves_nothing_to_sample():
    """全部陷阱都退出事件表：没有事件可推进，窗口为 0，判未通过验证。"""
    result = _three_trap_sim(fast_trap_rate=1e5).run()
    assert result.n_fast_traps == 3
    assert result.elapsed_time == 0.0
    assert result.validated is False
    assert any("测量窗口不足" in note for note in result.validation_notes)


def test_insufficient_timescale_separation_fails_validation():
    """min Γ_F 与平均事件速率之比低于下限 → 未通过验证；没有快陷阱（∞）不受影响。"""
    args = dict(
        n_defects=10, forward=True, elapsed=1e-6, measured_events=3200,
        occ_mean_abs_dev=0.0, n_positive_events=0, current=1e-15, current_counts_total=0.0,
        batch_rel_error=0.0,
    )
    ok, notes = _validate_result(timescale_separation=np.inf, **args)
    assert ok is True and notes == ()
    ok, notes = _validate_result(timescale_separation=0.5 * _FAST_SEPARATION_MIN, **args)
    assert ok is False
    assert len(notes) == 1 and "时间尺度分离不足" in notes[0]
