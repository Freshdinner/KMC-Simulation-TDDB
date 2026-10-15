"""KMC 引擎的一致性检验。

重点是分解统计的自洽（各分量之和 = 总电流）、可复现性，以及三个电流
估计量之间的相互印证。
"""

import numpy as np
import pytest
from scipy import constants

from conftest import make_config, make_defect_config
from kmc_leakage import kmc
from kmc_leakage.kmc import MECHANISMS, KMCSimulation, run_single
from kmc_leakage.rates import r1_electrode_electrode, r7_inelastic_hop

EV = constants.electron_volt


def _run(**kwargs):
    return KMCSimulation(make_config(**kwargs)).run()


def test_mechanisms_sum_to_total_current():
    """按机制分解的四项之和必须等于总电流——天然的一致性检验。"""
    result = _run()
    assert sum(result.by_mechanism.values()) == pytest.approx(result.current, rel=1e-12)
    assert set(result.by_mechanism) == set(MECHANISMS)


def test_defect_decomposition_sums_to_total_current():
    """按缺陷分解之和 + R1 的两项 = 总电流（R1 不经过任何缺陷）。"""
    result = _run()
    electrode_only = result.by_mechanism["direct_fn"] + result.by_mechanism["thermionic"]
    assert result.by_defect.sum() + electrode_only == pytest.approx(result.current, rel=1e-12)
    assert result.by_defect.size == len(result.defects)


def test_defect_decomposition_matches_tat_plus_pf():
    """缺陷分解的口径与机制分解一致：陷阱相关的两类机制之和。"""
    result = _run()
    trap_mechanisms = result.by_mechanism["tat"] + result.by_mechanism["pf"]
    assert result.by_defect.sum() == pytest.approx(trap_mechanisms, rel=1e-12)


def test_r1_contribution_is_the_ideal_barrier_formula():
    """R1 的两项就是理想梯形势垒上的公式值 q·R1，不掺任何 KMC 统计。

    conftest 显式给了 barrier_height，两电极同材料，内建电势为 0，所以势垒
    剖面严格是 E_B - eU·z/Lz。手写这条直线、直接调纯函数，结果必须与
    by_mechanism 的两项吻合——残差只来自求解器给出的 φ 与解析直线之间的浮点
    差（实测 <1e-14 V）。
    """
    config = make_config(voltage=3.0)
    device = config.device
    result = KMCSimulation(config).run()

    z = np.linspace(0.0, device.Lz, kmc._R1_PATH_POINTS)
    band = device.barrier(device.negative) - constants.elementary_charge * 3.0 * z / device.Lz
    tunnel, thermionic = r1_electrode_electrode(
        z,
        band,
        device.area,
        3.0,
        config.simulation.temperature,
        device.negative.effective_mass,
        device.stack.single_material.m_di,
        -device.negative.fermi_energy,
        n_points=kmc._R1_ENERGY_POINTS,
    )
    q = constants.elementary_charge
    assert result.by_mechanism["direct_fn"] == pytest.approx(q * tunnel, rel=1e-9)
    assert result.by_mechanism["thermionic"] == pytest.approx(q * thermionic, rel=1e-9)


def test_r1_contribution_is_independent_of_kmc_sampling():
    """R1 在外加偏压剖面上求值一次，与轨迹无关：换种子、换事件数都逐位相同。

    这是"R1 脱离 KMC 循环"这条实现决定的直接可观察后果。同一工作点下 TAT/PF
    会随种子涨落，R1 的两项必须一动不动。
    """
    base = _run(voltage=3.0, n_events=400, seed=13)
    other_seed = _run(voltage=3.0, n_events=400, seed=99)
    longer = _run(voltage=3.0, n_events=900, seed=13)
    for other in (other_seed, longer):
        assert other.by_mechanism["direct_fn"] == base.by_mechanism["direct_fn"]
        assert other.by_mechanism["thermionic"] == base.by_mechanism["thermionic"]
    assert other_seed.by_mechanism["tat"] != base.by_mechanism["tat"]  # 轨迹项确实在动


def test_mechanism_fractions_sum_to_one():
    result = _run()
    assert sum(result.mechanism_fractions.values()) == pytest.approx(1.0, rel=1e-12)


def test_same_seed_reproduces_result():
    a, b = _run(seed=31), _run(seed=31)
    assert a.current == b.current
    assert np.allclose(a.by_defect, b.by_defect)
    c = _run(seed=32)
    assert c.current != a.current  # 换种子结果应当不同（统计涨落）


def test_occupancy_is_a_probability_and_matches_mean_field():
    result = _run(n_events=1500)
    assert np.all(result.occupancy >= 0.0) and np.all(result.occupancy <= 1.0)
    assert np.all(result.occupancy_mean_field >= 0.0)
    assert np.all(result.occupancy_mean_field <= 1.0)
    # KMC 采样的占据概率与条件稳态 p̂ 应当接近：这是条件期望估计量成立的前提
    assert np.abs(result.occupancy - result.occupancy_mean_field).mean() < 0.1


def test_estimators_agree_when_direct_statistics_are_sufficient():
    """输运事件足够多时，三个估计量必须相互印证。

    低偏压下直接估计量被正极俘获/逃逸的相消淹没（这正是主输出改用条件
    期望估计量的原因），所以这一检验放在事件充足的工况下做。注意本测试
    对种子敏感：漏电流由"位置最合适的那一个缺陷"主导，不同缺陷实现之间
    可以差几个数量级，这里固定一个事件数充足的实现。
    """
    result = _run(voltage=4.0, n_events=2000, seed=13)
    assert result.n_positive_events > 50
    assert result.current > 0.0
    assert result.current_direct == pytest.approx(result.current, rel=0.3)
    assert result.current_counts == pytest.approx(result.current, rel=0.3)


def test_current_increases_with_voltage_and_temperature():
    low = _run(voltage=2.0)
    high = _run(voltage=3.0)
    assert 0.0 < low.current < high.current
    cold = _run(voltage=3.0, temperature=273.15)
    hot = _run(voltage=3.0, temperature=398.15)
    assert 0.0 < cold.current < hot.current


def test_zero_bias_current_is_negligible():
    """零偏压下 R1 恒为零，TAT 的正反通量解析抵消，残余量可忽略。

    残余来自两处，都记录在案：
    1. 本项目的机制表里 PF 发射没有反向通道（导带→陷阱俘获未建模），零场
       下 R8 = ν·exp(-ED/k_BT) 仍非零，留下一个伪零偏压电流；
    2. 条件期望估计量对陷阱间跃迁网络不严格满足细致平衡。
    两者合计比 3 V 工作点小十几个数量级。
    """
    result = _run(voltage=0.0)
    assert result.by_mechanism["direct_fn"] == 0.0
    assert result.by_mechanism["thermionic"] == 0.0
    biased = _run(voltage=3.0)
    assert abs(result.current) < 1e-12 * abs(biased.current)


def test_hopping_channel_is_r7_not_r6_plus_r7():
    """陷阱间跃迁在事件表里只走 R7，避免下坡跃迁被算两倍。"""
    config = make_config(defects=make_defect_config(concentration_cm3=3.0e19))
    sim = KMCSimulation(config)
    occ = np.zeros(sim.n_defects, dtype=bool)
    _, hop = sim._rates(occ)

    phi, _, _ = sim._electrostatics(occ)
    from kmc_leakage.bands import trap_levels

    energy = trap_levels(sim.defects.band_reference, phi, sim.defects.ED)
    # 取近邻表里最近的一对做逐元素核对
    pairs = sim._pairs
    k = int(np.argmin(pairs.r))
    a, b = int(pairs.a[k]), int(pairs.b[k])
    expected = r7_inelastic_hop(
        config.device.stack.single_material.attempt_frequency,
        pairs.r[k],
        sim.defects.r_D[a],
        sim.defects.r_D[b],
        energy[a],
        energy[b] - sim._hop_U_ab[k],  # F02：终态 b 去掉出发陷阱 a 的库仑贡献
        config.simulation.temperature,
    )
    assert hop[k] == pytest.approx(expected, rel=1e-12)
    assert np.all(pairs.a != pairs.b)  # 不允许自跃迁
    assert hop.shape == (len(pairs),)


def test_hop_neighbour_table_reproduces_the_dense_matrix():
    """近邻表在保留对上与稠密 (N,N) 矩阵逐位相同，截掉的总速率可忽略。

    这是"只列截断半径内的缺陷对"这条实现（模块文档说明 4）的正确性判据：
    保留的必须一字不差，丢掉的必须在双精度噪声以下。聚合量（流入/流出总
    速率）也要与稠密 matvec 对上，否则条件期望估计量的分母就错了。
    """
    config = make_config(defects=make_defect_config(concentration_cm3=3.0e19))
    device, diel = config.device, config.device.stack.single_material
    sim = KMCSimulation(config)
    occ = np.zeros(sim.n_defects, dtype=bool)
    occ[::2] = True
    _, hop = sim._rates(occ)

    phi, _, _ = sim._electrostatics(occ)
    from kmc_leakage.bands import trap_levels
    from kmc_leakage.defects import pair_distances

    energy = trap_levels(sim.defects.band_reference, phi, sim.defects.ED)
    # F02：终态 b 的能级要去掉出发陷阱 a 的库仑贡献 U_ab（对称化的 q²·G_ab），
    # 与 kmc._rates 里 energy[b] − self._hop_U_ab 同口径
    u_dense = -constants.elementary_charge * 0.5 * (
        sim.coupling.phi_matrix + sim.coupling.phi_matrix.T
    )
    dense = r7_inelastic_hop(
        diel.attempt_frequency,
        pair_distances(sim.defects.positions, device.Lx, device.Ly),
        sim.defects.r_D[:, None],
        sim.defects.r_D[None, :],
        energy[:, None],
        energy[None, :] - u_dense,
        config.simulation.temperature,
    )
    np.fill_diagonal(dense, 0.0)

    pairs = sim._pairs
    assert 0 < len(pairs) < sim.n_defects ** 2  # 确实截掉了一部分
    assert np.array_equal(hop, dense[pairs.a, pairs.b])
    # 截掉的那部分总速率在双精度噪声以下
    assert dense.sum() - hop.sum() <= 1e-15 * dense.sum()

    # 聚合量与稠密 matvec 等价（把稠密矩阵按同一张表屏蔽后逐位比较）
    occf, empty = occ.astype(float), ~occ
    kept = np.zeros_like(dense)
    kept[pairs.a, pairs.b] = dense[pairs.a, pairs.b]
    hop_in, hop_out = sim._hop_sums(hop, occf, empty)
    assert np.allclose(hop_in, occf @ kept, rtol=1e-12, atol=0.0)
    assert np.allclose(hop_out, kept @ empty, rtol=1e-12, atol=0.0)


def test_run_single_matches_class_api():
    config = make_config()
    assert run_single(config).current == KMCSimulation(config).run().current


def test_empty_device_runs_r1_only():
    """全器件无缺陷时走 R1-only 完整路径，不触发空数组异常（review F07）。

    陷阱相关机制恒为零、条件通量窗口为空，总电流等于 R1 两项；实际执行事件数为
    0（不是配置值），逐缺陷分解是空数组。整条链（生成→静电→KMC→结果）容空集合。
    """
    from conftest import make_device

    device = make_device(defects=make_defect_config(concentration_cm3=0.0))
    result = KMCSimulation(make_config(device=device, voltage=3.0, n_events=500)).run()
    assert len(result.defects) == 0
    assert result.by_mechanism["tat"] == 0.0 and result.by_mechanism["pf"] == 0.0
    r1 = result.by_mechanism["direct_fn"] + result.by_mechanism["thermionic"]
    assert result.current == pytest.approx(r1, rel=1e-12) and result.current > 0.0
    assert result.n_events == 0  # 实际执行事件数，不是配置值
    assert result.elapsed_time == 0.0
    assert result.by_defect.shape == (0,)


def test_result_energetics_are_consistent():
    """统计窗口应当是正的，占据概率、电流密度换算一致。"""
    result = _run()
    assert result.elapsed_time > 0.0
    assert result.current_density == pytest.approx(result.current / result.area)
    assert result.n_events == 400
