"""簇级平均场内层求解器（TDDB Phase 2，ADR-031，PROJECT_CONTEXT_TDDB.md §6.2、§6.9、§9）。

- 孤立的强耦合簇上等于主方程精确稳态（与 test_master_equation.py 同一体系，rtol 1e-10）；
- 强耦合簇之外还有一个弱耦合陷阱（test_fast_traps.py 的三陷阱体系）：簇级平均场与精确稳态
  差 <1e-9，去掉分簇的逐陷阱平均场高 2.4%——那是簇内 Pauli/库仑关联被平均掉的结果；
- 没有强耦合时就是逐陷阱平均场：电流逐位等于单陷阱条件通量在不动点处的值，不动点与独立
  写的简单阻尼迭代一致；
- 收敛判据是占据的相对残差：占据只有 1e-10 量级、经快跃迁供给邻居的陷阱也要收敛到位；
- 不动点与初值无关，从不动点热启动一次求值即收敛；未收敛时如实标出；
- 按机制、按缺陷分解之和 = 总电流；无缺陷时只剩 R1。
"""

import numpy as np
import pytest
from scipy import constants

from conftest import make_config, make_defect_config, make_device
from test_fast_traps import _three_trap_sim
from test_master_equation import _two_trap_sim
from kmc_leakage import mean_field
from kmc_leakage.kmc import MECHANISMS, KMCSimulation
from kmc_leakage.master_equation import stationary_distribution, steady_state_flux, transition_generator
from kmc_leakage.mean_field import solve_mean_field

Q = constants.elementary_charge


def _trap_current(result):
    return result.current - (result.by_mechanism["direct_fn"] + result.by_mechanism["thermionic"])


def _exact(sim):
    """主方程精确稳态：(陷阱电流 [A], 各陷阱占据期望)。"""
    occs, generator = transition_generator(sim)
    pi = stationary_distribution(generator)
    return steady_state_flux(sim)["exact_current"], pi @ occs


@pytest.fixture(scope="module")
def clustered():
    """20×20×8 nm、3×10¹⁹ cm⁻³（96 个陷阱，17 个强耦合簇）、3 V：有分数占据与多陷阱簇。"""
    device = make_device(defects=make_defect_config(concentration_cm3=3.0e19))
    sim = KMCSimulation(make_config(voltage=3.0, seed=1, device=device))
    return sim, solve_mean_field(sim)


def test_isolated_cluster_equals_master_equation():
    """相距 0.9 nm 的两个陷阱连成一个簇、覆盖全部陷阱：不动点就是主方程精确稳态。"""
    sim = _two_trap_sim(z=(3e-9, 3.9e-9))
    assert [c.members.tolist() for c in sim._clusters] == [[0, 1]]
    exact_current, exact_occupancy = _exact(sim)
    result = solve_mean_field(sim)
    assert result.converged
    assert _trap_current(result) == pytest.approx(exact_current, rel=1e-10)
    assert result.occupancy == pytest.approx(exact_occupancy, rel=1e-8)


def test_cluster_keeps_correlations_that_per_trap_mean_field_drops():
    """三陷阱体系：#0/#1 是强耦合簇，#2 在簇外、占据约 1e-8。

    簇级平均场只对 #2 取平均电荷，与精确稳态差 <1e-9。去掉分簇就是逐陷阱平均场，#1 的
    占据 0.136、与 #0 的库仑作用被按平均电荷计，电流高 2.4%（与 ADR-023 把快陷阱排除
    出簇时的偏差同源）。
    """
    sim = _three_trap_sim()
    assert [c.members.tolist() for c in sim._clusters] == [[0, 1]]
    exact_current, exact_occupancy = _exact(sim)
    result = solve_mean_field(sim)
    assert result.converged
    assert _trap_current(result) == pytest.approx(exact_current, rel=1e-9)
    assert result.occupancy[:2] == pytest.approx(exact_occupancy[:2], rel=1e-8)

    sim._clusters = []
    per_trap = solve_mean_field(sim)
    assert per_trap.converged
    assert _trap_current(per_trap) / exact_current == pytest.approx(1.024, abs=0.001)


def test_without_coupled_clusters_it_is_the_per_trap_mean_field():
    """没有强耦合簇时逐位退回逐陷阱平均场：电流就是单陷阱条件通量在不动点处的值。"""
    sim = KMCSimulation(make_config(voltage=2.0, seed=2))
    assert sim._clusters == []
    result = solve_mean_field(sim)
    assert result.converged
    p = result.occupancy
    assert np.any((p > 0.05) & (p < 0.95))  # 有分数占据，不是平凡的全空/全满

    blocks, hop = sim._rates(p)
    tat, pf, p_hat = sim._conditional_flux(blocks, *sim._hop_sums(hop, p, 1.0 - p))
    assert np.array_equal(result.by_defect, (tat + pf) * Q)
    assert np.all(np.abs(p_hat - p) <= 1e-9 * p)

    # 独立写的简单阻尼迭代收敛到同一个不动点
    damped = np.zeros(sim.n_defects)
    for _ in range(1000):
        blocks, hop = sim._rates(damped)
        target = sim._conditional_flux(blocks, *sim._hop_sums(hop, damped, 1.0 - damped))[2]
        if np.all(np.abs(target - damped) <= 1e-13 * target):
            break
        damped = 0.5 * damped + 0.5 * target
    assert p == pytest.approx(damped, rel=1e-9)


def test_tiny_supplying_occupancy_is_converged_relatively():
    """占据很小的陷阱经快跃迁供给邻居，收敛要看相对残差。

    0.9 nm 强耦合对去掉分簇后，陷阱 #0 的占据只有 9e-11，却经 ~1e9 s⁻¹ 的跃迁供给 #1。
    绝对残差 1e-10 的判据在全空初值处就满足，会把这条通路整个漏掉（电流只剩 1.4%）；
    相对判据下逐陷阱平均场的 #0 占据与精确值差 2e-4，电流差 2e-7。
    """
    sim = _two_trap_sim(z=(3e-9, 3.9e-9))
    exact_current, exact_occupancy = _exact(sim)
    assert exact_occupancy[0] < 1e-10
    sim._clusters = []
    result = solve_mean_field(sim)
    assert result.converged and result.iterations > 1
    assert result.occupancy[0] == pytest.approx(exact_occupancy[0], rel=1e-3)
    assert _trap_current(result) == pytest.approx(exact_current, rel=1e-6)


def test_fixed_point_does_not_depend_on_initial_occupancy(clustered):
    """全空、全满两种初值收敛到同一不动点；从不动点热启动一次求值即收敛、结果逐位相同。"""
    sim, empty = clustered
    assert len(sim._clusters) > 0 and max(c.members.size for c in sim._clusters) > 2
    full = solve_mean_field(sim, occupancy=np.ones(sim.n_defects))
    assert empty.converged and full.converged
    assert full.occupancy == pytest.approx(empty.occupancy, rel=1e-8)
    assert full.current == pytest.approx(empty.current, rel=1e-8)

    warm = solve_mean_field(sim, occupancy=empty.occupancy)
    assert warm.iterations == 1 and warm.converged
    assert warm.current == empty.current
    assert np.array_equal(warm.occupancy, empty.occupancy)


def test_unconverged_result_is_flagged(clustered, monkeypatch):
    """迭代次数用完仍未收敛：标 converged=False，电流与占据取最后一次求值处。"""
    sim, reference = clustered
    monkeypatch.setattr(mean_field, "_MAX_ITER", 2)
    result = solve_mean_field(sim)
    assert result.iterations == 2
    assert not result.converged and result.residual > mean_field._TOL
    blocks, hop = sim._rates(result.occupancy)
    tat, pf, _ = sim._trap_flux(result.occupancy, blocks, hop)
    assert np.array_equal(result.by_defect, (tat + pf) * Q)
    assert result.current != reference.current


def test_decompositions_sum_to_total_current(clustered):
    """按机制四项之和 = 总电流；按缺陷之和 + R1 两项 = 总电流 = TAT + PF + R1。"""
    sim, result = clustered
    assert set(result.by_mechanism) == set(MECHANISMS)
    assert sum(result.by_mechanism.values()) == pytest.approx(result.current, rel=1e-12)
    r1 = result.by_mechanism["direct_fn"] + result.by_mechanism["thermionic"]
    assert result.by_defect.sum() + r1 == pytest.approx(result.current, rel=1e-12)
    assert result.by_defect.sum() == pytest.approx(
        result.by_mechanism["tat"] + result.by_mechanism["pf"], rel=1e-12
    )
    assert result.current_density == pytest.approx(result.current / result.area, rel=1e-15)
    assert r1 == pytest.approx(sum(sim._r1_current), rel=1e-15)


def test_empty_device_is_r1_only():
    """无缺陷：一次求值即收敛，电流只剩 R1 两项。"""
    device = make_device(defects=make_defect_config(concentration_cm3=0.0))
    sim = KMCSimulation(make_config(voltage=3.0, device=device))
    result = solve_mean_field(sim)
    assert result.converged and result.iterations == 1
    assert result.by_defect.shape == (0,)
    assert result.by_mechanism["tat"] == 0.0 and result.by_mechanism["pf"] == 0.0
    assert result.current == pytest.approx(sum(sim._r1_current), rel=1e-15) and result.current > 0.0
