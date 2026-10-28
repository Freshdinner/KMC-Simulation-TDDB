"""簇级平均场内层求解器（TDDB Phase 2，ADR-031/038，PROJECT_CONTEXT_TDDB.md §6.2、§6.9、§9）。

- 孤立的强耦合簇上等于主方程精确稳态（与 test_master_equation.py 同一体系，rtol 1e-10）；
- 强耦合簇之外还有一个弱耦合陷阱（test_fast_traps.py 的三陷阱体系）：簇级平均场与精确稳态
  差 <1e-9，去掉分簇的逐陷阱平均场高 2.4%——那是簇内 Pauli/库仑关联被平均掉的结果；
- 没有强耦合时就是逐陷阱平均场：陷阱间跃迁的 MA 因子取两端平均场能级之差，电流逐位等于
  单陷阱条件通量在不动点处的值，不动点与独立写的简单阻尼迭代一致；
- 收敛判据是占据的相对残差：占据只有 1e-10 量级、经快跃迁供给邻居的陷阱也要收敛到位；
- 不动点与初值无关，从不动点热启动一次求值即收敛；未收敛时如实标出；
- Anderson 混合走不动时牛顿法接手（ADR-048）：收敛到同一个不动点；ADR-046 what-if 里 Anderson 不收敛的状态
  （膜体内致密的空位网络）现在收敛；
- 分组有限差分 Jacobian（ADR-058）：同一组的缺陷两两间距 > 1.25·L_z，全部缺陷各属一组；与逐列有限差分相对差 ≤ 5%，
  作牛顿迭代矩阵时线性收敛因子 ≤ 0.1；
- 按机制、按缺陷分解之和 = 总电流；无缺陷时只剩 R1；
- 电流连续（ADR-038）：强耦合分量被切开的致密体系上，负极净注入 = 正极净收集；
- 零偏压平衡（ADR-038）：逐陷阱平均场里每一对陷阱的正反向通量相等（KMC 的 −U_ab 口径做不到）；
  簇–孤立边界按簇内组态求速率，零偏压下没有净通量。
"""

import os
from dataclasses import replace

import numpy as np
import pytest
from scipy import constants

from conftest import M_E, make_config, make_defect_config, make_device, make_material
from test_fast_traps import _three_trap_sim
from test_master_equation import _two_trap_sim
from kmc_leakage import kmc as kmc_module
from kmc_leakage import mean_field
from kmc_leakage.config import DeviceConfig, DielectricStack, ElectrodeMaterial
from kmc_leakage.defects import DefectEnsemble, layer_bulk_defects, localization_radius, minimum_image_delta
from kmc_leakage.io import load_config
from kmc_leakage.kmc import (
    _B_R2A,
    _B_R2C,
    _B_R3A,
    _B_R3C,
    _B_R4A,
    _B_R4C,
    _B_R5A,
    _B_R5C,
    _B_R8,
    _CLUSTER_MAX_SIZE,
    MECHANISMS,
    KMCSimulation,
)
from kmc_leakage.master_equation import stationary_distribution, steady_state_flux, transition_generator
from kmc_leakage.mean_field import mean_field_flux, mean_field_rates, solve_mean_field
from kmc_leakage.rates import miller_abrahams_factor
from kmc_leakage.tddb import _cluster_diagnostics
from kmc_leakage.thermal import heat_release

Q = constants.elementary_charge
EV = constants.electron_volt


def _trap_current(result):
    return result.current - (result.by_mechanism["direct_fn"] + result.by_mechanism["thermionic"])


def _exact(sim):
    """主方程精确稳态：(陷阱电流 [A], 各陷阱占据期望)。"""
    occs, generator = transition_generator(sim)
    pi = stationary_distribution(generator)
    return steady_state_flux(sim)["exact_current"], pi @ occs


def _per_trap_flux(sim, p):
    """逐陷阱平均场的 F(p)，按定义独立写：MA 因子取两端平均场能级之差 ΔH = E_b − E_a。"""
    energy, field, barrier_c, barrier_a = sim._trap_state(p)
    blocks = sim._electrode_blocks(energy, field, barrier_c, barrier_a)
    pairs = sim._pairs
    hop = sim._hop_geometric * miller_abrahams_factor(energy[pairs.a], energy[pairs.b], sim.sim.temperature)
    return sim._conditional_flux(blocks, *sim._hop_sums(hop, p, 1.0 - p))


def _electrode_net_flux(sim, p):
    """各陷阱的 (负极净注入, 正极净收集) [1/s]：孤立陷阱按平均占据，簇成员按簇内稳态分布。"""
    rates = mean_field_rates(sim, p)
    b = rates.blocks
    negative = (1.0 - p) * (b[_B_R2C] + b[_B_R4C]) - p * (b[_B_R3C] + b[_B_R5C])
    positive = p * (b[_B_R3A] + b[_B_R5A] + b[_B_R8]) - (1.0 - p) * (b[_B_R2A] + b[_B_R4A])
    for table, pi in zip(rates.tables, rates.distributions):
        members, n, block = table.cluster.members, table.cluster.states.astype(float), table.block
        negative[members] = pi @ ((1.0 - n) * (block[_B_R2C] + block[_B_R4C]) - n * (block[_B_R3C] + block[_B_R5C]))
        positive[members] = pi @ (n * (block[_B_R3A] + block[_B_R5A] + block[_B_R8]) - (1.0 - n) * (block[_B_R2A] + block[_B_R4A]))
    return negative, positive


@pytest.fixture(scope="module")
def clustered():
    """20×20×8 nm、3×10¹⁹ cm⁻³（96 个陷阱，17 个强耦合簇）、3 V：有分数占据与多陷阱簇。"""
    device = make_device(defects=make_defect_config(concentration_cm3=3.0e19))
    sim = KMCSimulation(make_config(voltage=3.0, seed=1, device=device))
    return sim, solve_mean_field(sim)


@pytest.fixture(scope="module")
def split():
    """10×10×8 nm、1×10²⁰ cm⁻³（80 个陷阱）、3 V：60 个陷阱的强耦合分量超过簇大小上限、被切开，
    簇–簇与簇–孤立两类边界都有（Phase 4 致密空位层的小样）。"""
    device = make_device(Lx=10e-9, Ly=10e-9, defects=make_defect_config(concentration_cm3=1.0e20))
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
    """没有强耦合簇时逐位退回逐陷阱平均场：电流就是单陷阱条件通量在不动点处的值，陷阱间跃迁的
    MA 因子取两端平均场能级之差（不是 KMC 事件用的 E_b − U_ab − E_a）。"""
    sim = KMCSimulation(make_config(voltage=2.0, seed=2))
    assert sim._clusters == []
    result = solve_mean_field(sim)
    assert result.converged
    p = result.occupancy
    assert np.any((p > 0.05) & (p < 0.95))  # 有分数占据，不是平凡的全空/全满

    tat, pf, p_hat = _per_trap_flux(sim, p)
    assert np.array_equal(result.by_defect, (tat + pf) * Q)
    assert np.all(np.abs(p_hat - p) <= 1e-9 * p)
    energy = sim._trap_state(p)[0]
    assert not np.array_equal(mean_field.mean_field_hop_rates(sim, energy), sim._hop_rates(energy))

    # 独立写的简单阻尼迭代收敛到同一个不动点
    damped = np.zeros(sim.n_defects)
    for _ in range(1000):
        target = _per_trap_flux(sim, damped)[2]
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
    """Anderson 与牛顿法的步数都用完仍未收敛：标 converged=False，电流与占据取残差最小的那次求值处。
    Anderson 2 次、牛顿法 0 步时求 F 4 次：Anderson 2 次、p̂(0) 1 次、从 p̂(0) 出发的起点 1 次。"""
    sim, reference = clustered
    monkeypatch.setattr(mean_field, "_ANDERSON_MAX_ITER", 2)
    monkeypatch.setattr(mean_field, "_NEWTON_MAX_STEPS", 0)
    result = solve_mean_field(sim)
    assert result.iterations == 4 and result.newton_steps == 0
    assert not result.converged and result.residual > mean_field._TOL
    tat, pf, p_hat = mean_field_flux(sim, result.occupancy)
    assert np.array_equal(result.by_defect, (tat + pf) * Q)
    gap = mean_field._log_occupancy(p_hat) - mean_field._log_occupancy(result.occupancy)
    assert result.residual == pytest.approx(np.max(np.abs(gap)), abs=1e-12)
    assert result.current != reference.current


def _whatif_state():
    """ADR-046 what-if（O_i⁻ 与空位重合，seed 1、3.0 V，`zro2_tddb.yaml`）第 29 个空位出生后交给平均场的状态：
    39 个缺陷（10 个初始、29 个生成空位带 q_fix = −e）与热启动占据（`tests/data/whatif_s1_n29.npz`，按 ADR-048 的
    代码重放 what-if 轨迹得到，前 29 个出生位置与 what-if 相同）。what-if 的能级按当时的高斯分布抽样，数据里存了全部
    缺陷的位置与能级，这里直接用。what-if 跑在 20×20 nm 盒子上，盒子与泊松网格固定为当时的 20×20 nm、nx = ny = 80，
    与 `zro2_tddb.yaml` 现在的盒子（ADR-051 起 50×50 nm）无关。"""
    data = np.load(os.path.join(os.path.dirname(__file__), "data", "whatif_s1_n29.npz"))
    config = load_config(os.path.join(os.path.dirname(__file__), "..", "examples", "zro2_tddb.yaml"))
    config = replace(
        config, device=replace(config.device, Lx=20e-9, Ly=20e-9),
        simulation=replace(config.simulation, voltage=3.0, seed=1, temperature=float(data["temperature"]),
                           grid=replace(config.simulation.grid, nx=80, ny=80)))
    n_initial = int(data["n_initial"])
    defects = layer_bulk_defects(config.device, 0, data["positions"], np.random.default_rng(1))
    defects = replace(
        defects, ED=data["ED"], r_D=localization_radius(data["ED"], defects.m_di),
        fixed_charge=np.where(np.arange(len(defects)) >= n_initial, -Q, 0.0),
    )
    return KMCSimulation(config, defects=defects), data["occupancy"]


def test_state_that_defeated_anderson_converges_with_newton(monkeypatch):
    """ADR-046 what-if 里平均场不收敛的状态：Anderson 混合 200 次仍不收敛（what-if 把这条记录的 12.6 A/cm² 当成了
    击穿），牛顿法接手后收敛到 6.48 A/cm²（ADR-048）。"""
    sim, start = _whatif_state()
    with monkeypatch.context() as patch:
        patch.setattr(mean_field, "_ANDERSON_MAX_ITER", 200)
        patch.setattr(mean_field, "_NEWTON_MAX_STEPS", 0)
        anderson = solve_mean_field(sim, start)
    assert not anderson.converged and anderson.residual > 1e-2
    result = solve_mean_field(sim, start)
    assert result.converged and result.newton_steps > 0
    assert result.current_density * 1e-4 == pytest.approx(6.4786, rel=1e-4)
    tat, pf, p_hat = mean_field_flux(sim, result.occupancy)
    assert np.allclose(p_hat, result.occupancy, rtol=1e-9, atol=0.0)


def test_newton_takes_over_when_anderson_stops(clustered, monkeypatch):
    """Anderson 只给 2 次时牛顿法接手（分组有限差分 Jacobian + Broyden 更新，ADR-048/058），收敛到同一个不动点；
    Anderson 自己收敛时不走牛顿法。"""
    sim, reference = clustered
    assert reference.converged and reference.newton_steps == 0
    monkeypatch.setattr(mean_field, "_ANDERSON_MAX_ITER", 2)
    result = solve_mean_field(sim)
    assert result.converged and result.newton_steps > 0
    assert result.iterations > len(mean_field._jacobian_groups(sim))  # 至少求过一次有限差分 Jacobian
    assert result.occupancy == pytest.approx(reference.occupancy, rel=1e-8, abs=1e-14)
    assert result.current == pytest.approx(reference.current, rel=1e-8)


def test_grouped_jacobian_is_close_enough_for_newton(clustered):
    """分组有限差分（ADR-058）：同一组的缺陷两两间距 > 1.25·L_z（xy 最小镜像），全部缺陷各属一组，组数少于缺陷数；
    在不动点处与逐列有限差分（每组一个缺陷）的相对差 ≤ 5%（实测 2.6%），作牛顿迭代矩阵时 I − J_组⁻¹·J_逐列 的谱半径
    ≤ 0.1（实测 0.038），牛顿法照样收敛。"""
    sim, reference = clustered
    device, positions, n = sim.device, sim.defects.positions, sim.n_defects
    groups = mean_field._jacobian_groups(sim)
    assert sorted(np.concatenate(groups).tolist()) == list(range(n)) and len(groups) < n
    for members in groups:
        distance = np.linalg.norm(
            minimum_image_delta(positions[members, None, :] - positions[None, members, :], device.Lx, device.Ly), axis=-1
        )
        assert np.all(distance[~np.eye(members.size, dtype=bool)] > mean_field._JACOBIAN_GROUP_SEPARATION * device.Lz)

    current = mean_field._evaluate(sim, mean_field._log_occupancy(reference.occupancy))
    grouped = mean_field._jacobian(sim, current, groups)
    column = mean_field._jacobian(sim, current, [np.array([j]) for j in range(n)])
    assert np.linalg.norm(grouped - column) / np.linalg.norm(column) <= 0.05
    assert np.max(np.abs(np.linalg.eigvals(np.eye(n) - np.linalg.solve(grouped, column)))) <= 0.1


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


def test_cluster_tables_equal_configuration_by_configuration_rates(split, monkeypatch):
    """簇内组态表（`KMCSimulation._cluster_tables`）：能级与成员的电极速率块（静电按低秩增量叠加）等于把这个组态
    整体代进 `_trap_state` 后求的值；跃迁的流入 / 流出总速率等于逐对累加；电极速率按小块求（`_RATE_CHUNK`）与
    一次求完相同。"""
    sim, result = split
    p = result.occupancy
    tables = sim._cluster_tables(p)
    pairs = sim._pairs
    for table in tables:
        c = table.cluster
        for s, state in enumerate(c.states):
            occupancy = p.copy()
            occupancy[c.members] = state
            energy, field, barrier_c, barrier_a = sim._trap_state(occupancy)
            assert table.energy[s] == pytest.approx(energy, rel=1e-12, abs=1e-12 * EV)
            blocks = sim._electrode_blocks(
                energy[c.members], field[c.members], barrier_c[c.members], barrier_a[c.members], c.members
            )
            assert table.block[:, s, :] == pytest.approx(blocks, rel=1e-12, abs=0.0)
            hop_in = np.zeros(c.members.size)
            for col, member in zip(c.in_cols, c.in_member):
                hop_in[member] += p[pairs.a[c.pairs[col]]] * table.hop[s, col]
            assert table.hop_in[s] == pytest.approx(hop_in, rel=1e-13)
    monkeypatch.setattr(kmc_module, "_RATE_CHUNK", 7)
    for whole, chunked in zip(tables, sim._cluster_tables(p)):
        assert chunked.block == pytest.approx(whole.block, rel=1e-14, abs=0.0)


def test_current_is_continuous_across_split_components(split):
    """强耦合分量超过簇大小上限被切开后，簇–簇、簇–孤立边界上的跃迁仍是两侧共用的同一组速率：
    不动点处每个陷阱粒子数守恒，负极净注入 = 正极净收集 = I_陷阱/q（ADR-038）。

    原来簇一侧按组态、簇外一侧按平均占据各算一遍边界通量，Phase 4 终态上两者差 258 倍（ADR-037）。
    """
    sim, result = split
    assert result.converged
    _, _, largest, n_split = _cluster_diagnostics(sim)
    assert largest > _CLUSTER_MAX_SIZE and n_split >= 1
    cluster_of = mean_field_rates(sim, result.occupancy).cluster_of
    a, b = cluster_of[sim._pairs.a], cluster_of[sim._pairs.b]
    assert np.any((a >= 0) & (b >= 0) & (a != b)) and np.any((a >= 0) != (b >= 0))

    negative, positive = _electrode_net_flux(sim, result.occupancy)
    trap_current = result.by_mechanism["tat"] + result.by_mechanism["pf"]
    assert positive.sum() * Q == pytest.approx(trap_current, rel=1e-12)
    assert negative.sum() == pytest.approx(positive.sum(), rel=1e-8)


def test_per_trap_mean_field_has_no_circulation_at_zero_bias():
    """零偏压、去掉分簇：MA 因子取两端平均场能级之差时，不动点就是平均场平衡 p_i = f(E_i(p))，每一对
    陷阱的正反向通量逐对相等。同一组能级换成 KMC 事件的 ΔH = E_b − U_ab − E_a，正反两个方向都多减了
    U_ab，同一占据下净通量是总通量的 19–33%（ADR-037）。

    陷阱取深能级（E_D ≈ 1.75 eV）：R8 没有反向通道，浅能级陷阱零偏压下占据只有 1e-10、陷阱间通量与
    R8 的伪电流同量级，会盖住这里要看的平衡。
    """
    sim = _negative_cluster_with_free_neighbour(0.0)
    sim._clusters = []
    result = solve_mean_field(sim)
    assert result.converged
    p = result.occupancy
    assert np.all((p > 0.3) & (p < 0.95))
    energy = sim._trap_state(p)[0]
    a, b = sim._pairs.a, sim._pairs.b
    order = np.lexsort((b, a))
    reverse = order[np.lexsort((a[order], b[order]))]  # 第 k 对的反向对在近邻表里的位置
    assert np.array_equal(a[reverse], b) and np.array_equal(b[reverse], a)

    def net_over_gross(hop):
        forward = hop * p[a] * (1.0 - p[b])
        backward = forward[reverse]
        return np.abs(forward - backward) / (forward + backward)

    assert np.max(net_over_gross(mean_field.mean_field_hop_rates(sim, energy))) < 1e-8
    assert np.max(net_over_gross(sim._hop_rates(energy))) > 0.1


def _negative_cluster_with_free_neighbour(voltage):
    """负极侧两个强耦合陷阱 #0/#1（0.8 nm）成簇，#2 在 #0 旁 1.0 nm、按耦合阈值留在簇外，三者都是深能级
    （E_D ≈ 1.75 eV，零偏压下占据 0.4–0.9），簇–孤立边界跃迁约 3e5 s⁻¹。"""
    material = make_material()
    stack = DielectricStack.single(material, thickness=8e-9)
    electrode = ElectrodeMaterial("TiN", 4.6 * EV, M_E, 5.0 * EV)
    device = DeviceConfig(20e-9, 20e-9, electrode, electrode, stack, barrier_height=1.74 * EV)
    positions = np.array([[10e-9, 10e-9, 0.7e-9], [10.4e-9, 10e-9, 1.4e-9], [9.0e-9, 10e-9, 0.6e-9]])
    defects = DefectEnsemble.uniform(positions, np.array([1.75, 1.72, 1.78]) * EV, material, band_reference=1.74 * EV)
    return KMCSimulation(make_config(voltage=voltage, device=device), defects=defects)


def test_cluster_to_free_trap_boundary_is_detailed_balanced_at_zero_bias():
    """簇–孤立边界的速率按成员在每个簇内组态下的能级求、孤立陷阱一侧按 π 平均（ADR-038）：零偏压下这条
    边界对簇就是费米能级处的库，簇内链逐组态满足细致平衡，边界没有净通量、介质不放热。

    速率若取与组态无关的平均场常数，同一体系上边界净通量是总通量的 1%、介质吸热 1.5e-16 W（"电极 →
    成员 → 边界"环流）。测试把缺陷电荷对横向平均剖面的贡献置零：引擎的电极→陷阱势垒取横向平均剖面，
    里面含成员自身的电荷，同一成员的俘获（空、带 +e）与逃逸（占据、中性）走的势垒略有不同，在簇内链里
    留下 1e-3 量级的非平衡——这是引擎原有的口径（KMC 也是），与边界跃迁无关，这里把它单独拿掉。
    """
    sim = _negative_cluster_with_free_neighbour(0.0)
    sim.coupling.phi_bar[:] = 0.0
    assert sim.cluster_members[0].tolist() == [0, 1] and len(sim.cluster_members) == 1
    result = solve_mean_field(sim)
    assert result.converged
    p = result.occupancy
    assert np.all((p > 0.3) & (p < 0.95))

    rates = mean_field_rates(sim, p)
    table, pi = rates.tables[0], rates.distributions[0]
    c = table.cluster
    n = c.states.astype(float)
    into = pi @ ((1.0 - n[:, c.in_member]) * p[c.in_source] * table.hop[:, c.in_cols])
    out = pi @ (n[:, c.out_member] * (1.0 - p[c.out_target]) * table.hop[:, c.out_cols])
    assert into.max() > 1e5
    np.testing.assert_allclose(into, out, rtol=1e-9)

    # 介质放热的量级标尺：边界总通量 × 0.01 eV（常数速率时实测 1.5e-16 W，约为这个标尺的 0.35 倍）
    assert abs(heat_release(sim, p).dielectric) < 1e-9 * into.sum() * 0.01 * EV
