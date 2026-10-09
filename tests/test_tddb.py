"""TDDB 瞬态驱动（Phase 3/4，ADR-032–036，PROJECT_CONTEXT_TDDB.md §5、§6、§9）。

- 增量静电耦合（§6.4）：逐个追加缺陷后的耦合矩阵与对全体缺陷从头计算的一致；
- 新空位出生（§6.7、ADR-048）：出生时被一个电子占据、与初始缺陷同一物种，带着与它重合的 O_i⁻
  （q_fix = −e）；出生时介质总电荷正好多一个 −e，其余缺陷处的电势与电场的变化就是这个 −e 的贡献；
  外层每步从上一步的占据热启动；
- G = 0 时 I(t) 恒定，且逐位等于同一初始实现下的稳态结果（等温与开热反馈都成立）；
- 每步的强耦合簇与对当前缺陷集合新建 `KMCSimulation` 的分簇相同，终态电流与对终态缺陷
  从头求解的稳态一致；
- 单元生成率就是模型 C 的电极项：N_T·V_cell·R_1e,电极·P_bb，R_1e,电极 与引擎在单元中心放一个前驱参数
  陷阱时的两侧 R2 + R4 之和一致（含空间电荷）；按单元局域温度求值时逐单元等于该温度下的生成率；
- 陷阱项按源陷阱积分（ADR-042）：孤立源陷阱等于 scipy 自适应积分的精确值（含上坡的 MA 折点与窗口
  截断），有近旁缺陷挡住一部分区域时是单元中心求和在单元加密下的极限，与生成网格无关，源陷阱
  全空时为 0；落点按被积函数分布、都在可放置区域内；
- N_T 只改变时间标尺；放不下新空位的单元被跳过；
- 热反馈（§6.6、§8.4）：每步的输运温度是上一步功率给出的缺陷处加权温度，生成率取同一步的温度场；
- 击穿判据（§6.3，ADR-047）：轨迹停在电流密度第一次超过击穿电流密度的记录上，之前的记录与不停的
  轨迹逐位相同；
- 致密团簇标志：间距 < 截断的缺陷连边，xy 取最小镜像；
- I-t 幂律拟合对纯幂律精确还原 n，对带噪声的幂律给出合理的拟合优度。
"""

import csv
from dataclasses import fields, replace

import numpy as np
import pytest
from scipy import constants, integrate

from conftest import make_config, make_defect_config, make_device
from kmc_leakage import tddb as tddb_module
from kmc_leakage.config import GridConfig, TDDBConfig
from kmc_leakage.defects import (
    DefectEnsemble,
    _global_standoff,
    append_defects,
    localization_radius,
    minimum_image_delta,
)
from kmc_leakage.electrostatics import PoissonSolver
from kmc_leakage.kmc import _B_R2A, _B_R2C, _B_R4A, _B_R4C, _N_PHONON, MECHANISMS, ElectrostaticCoupling, KMCSimulation
from kmc_leakage.mean_field import solve_mean_field
from kmc_leakage.rates import bond_break_probability, single_electron_capture_rate
from kmc_leakage.tddb import (
    IncrementalCoupling,
    _draw_generation,
    distance_clusters,
    fit_power_law,
    fit_transient,
    generation_cells,
    generation_rates,
    run_tddb,
    write_tddb_csv,
)
from kmc_leakage.thermal import heat_release, power_map, solve_temperature, transport_temperature

EV = constants.electron_volt
Q = constants.elementary_charge
E_ANGSTROM = Q * constants.angstrom
PER_CM3 = constants.centi ** -3


def _tddb(**overrides) -> TDDBConfig:
    params = dict(
        activation_energy=4.0 * EV,
        precursor_ionization_energy=1.15 * EV,
        precursor_relaxation_energy=1.0 * EV,
        dipole_moment=7.1 * E_ANGSTROM,
        field_averaging_width=0.5e-9,
        precursor_concentration=4.0e19 * PER_CM3,
        generation_grid=(2, 2, 14),
        max_generated_vacancies=4,
        stop_time=1.0e4,
        breakdown_current_density=1.0e8,  # 1e4 A/cm²，远高于这些小体系的电流
    )
    params.update(overrides)
    return TDDBConfig(**params)


def _config(seed=3, concentration_cm3=3.0e18, **overrides):
    """20×20×8 nm、网格 8/8/16、3 V / 25 °C，生成单元 2×2×14；3×10¹⁸ cm⁻³ 即 10 个初始缺陷。"""
    device = make_device(defects=make_defect_config(concentration_cm3=concentration_cm3))
    return replace(make_config(voltage=3.0, seed=seed, device=device), tddb=_tddb(**overrides))


def _subset(defects: DefectEnsemble, n: int) -> DefectEnsemble:
    return DefectEnsemble(**{item.name: getattr(defects, item.name)[:n] for item in fields(DefectEnsemble)})


@pytest.fixture(scope="module")
def trajectory():
    """4 次生成的短轨迹，并记下每步交给平均场的缺陷实现与初值。

    初始缺陷取 3×10¹⁹ cm⁻³（96 个），一开始就有多个强耦合簇，逐步对比分簇才有内容。
    """
    config = _config(seed=1, concentration_cm3=3.0e19)
    calls = []
    original = tddb_module.solve_mean_field

    def recording(sim, occupancy=None):
        result = original(sim, occupancy)
        calls.append((sim, None if occupancy is None else occupancy.copy(), result))
        return result

    tddb_module.solve_mean_field = recording
    try:
        result = run_tddb(config)
    finally:
        tddb_module.solve_mean_field = original
    return config, result, calls


def test_incremental_coupling_matches_full_recomputation():
    """初始 4 个缺陷一次算好、再逐个追加 3 个：与对 7 个缺陷从头算的耦合矩阵一致（§6.4）。

    电势列按互易性取、场列用伴随求解，两者都只差舍入误差；初始块与 ElectrostaticCoupling 逐位相同。
    """
    config = make_config()
    solver = PoissonSolver(config.device, GridConfig(8, 8, 16))
    rng = np.random.default_rng(11)
    positions = np.column_stack([rng.uniform(0, 20e-9, 7), rng.uniform(0, 20e-9, 7), rng.uniform(0.6e-9, 7.4e-9, 7)])
    incremental = IncrementalCoupling.build(solver, positions[:4], np.zeros(4, dtype=int))
    initial = ElectrostaticCoupling(solver, positions[:4], np.zeros(4, dtype=int))
    for name in ("phi_matrix", "field_matrix", "phi_bar", "phi_self", "field_self"):
        assert np.array_equal(getattr(incremental, name), getattr(initial, name)), name
    first = incremental
    for point in positions[4:]:
        incremental = incremental.extended(point)
    assert first.phi_matrix.shape == (4, 4)  # 追加返回新对象，原对象不变
    full = ElectrostaticCoupling(solver, positions, np.zeros(7, dtype=int))

    phi_scale = np.abs(full.phi_matrix).max()
    field_scale = np.abs(full.field_matrix).max()
    np.testing.assert_allclose(incremental.phi_matrix, full.phi_matrix, rtol=1e-10, atol=1e-12 * phi_scale)
    np.testing.assert_allclose(incremental.field_matrix, full.field_matrix, rtol=1e-9, atol=1e-11 * field_scale)
    np.testing.assert_allclose(incremental.phi_bar, full.phi_bar, rtol=1e-12, atol=1e-14 * phi_scale)
    np.testing.assert_allclose(incremental.phi_self, full.phi_self, rtol=1e-12)
    np.testing.assert_allclose(incremental.field_self, full.field_self, rtol=1e-9, atol=1e-11 * field_scale)


def _field_vectors(sim, occupancy):
    """`KMCSimulation._electrostatics` 里的场矢量 (3, N)（它只返回大小）。"""
    weight = occupancy - sim.defects.charge_state
    return (
        sim._field_applied_at + sim._field_fixed_at
        + weight @ sim.coupling.field_matrix - weight * sim.coupling.field_self
    )


def test_new_vacancy_is_born_occupied_with_its_interstitial_charge(trajectory):
    """新空位出生占据 = 1（其余缺陷从上一步不动点热启动），与初始缺陷同一物种，带着与它重合的 O_i⁻
    （q_fix = −e，ADR-048）；初始缺陷 q_fix = 0。新空位连同 O_i⁻ 出生时带 −e：其余缺陷处的电势、电场与
    横向平均剖面在出生前后的变化，就是出生位置上一个 −e 点电荷的贡献（耦合矩阵新的一行）。"""
    config, result, calls = trajectory
    for (_, _, previous), (sim, start, _) in zip(calls[:-1], calls[1:]):
        assert start[-1] == 1.0
        assert np.array_equal(start[:-1], previous.occupancy)

    defects = result.defects
    initial = np.arange(result.n_initial)
    for name in ("charge_state", "m_di", "eps_opt", "band_gap", "phonon_energy", "huang_rhys",
                 "attempt_frequency", "band_reference", "layer_id", "interface_id", "field_side"):
        values = getattr(defects, name)
        assert np.all(values[result.n_initial:] == values[initial[0]]), name
    np.testing.assert_allclose(defects.r_D, localization_radius(defects.ED, defects.m_di), rtol=1e-15)
    assert np.all(defects.fixed_charge[initial] == 0.0)
    assert np.all(defects.fixed_charge[result.n_initial:] == -Q)
    assert np.all(defects.fixed_charge_offset == 0.0)

    for (before, _, solved), (after, start, _) in zip(calls[:-1], calls[1:]):
        n = before.n_defects
        coupling = after.coupling
        phi0, _, bar0 = before._electrostatics(solved.occupancy)
        phi1, field1, bar1 = after._electrostatics(start)
        np.testing.assert_allclose(phi1[:n], phi0 + coupling.phi_matrix[n, :n], rtol=1e-11)
        np.testing.assert_allclose(bar1, bar0 + coupling.phi_bar[n], rtol=1e-11, atol=1e-15)
        expected = _field_vectors(before, solved.occupancy) + coupling.field_matrix[:, n, :n]
        np.testing.assert_allclose(field1[:n], np.linalg.norm(expected, axis=0), rtol=1e-10)


def test_generation_conserves_charge(trajectory):
    """生成反应按电荷守恒处理（ADR-048）：介质里全部电荷 Σ[q·(Z − p) + q_fix] 在每次出生时正好多一个 −e，
    就是前驱俘获的那个电子；记录的净空间电荷就是这个总和。O_i 的电荷取 0（旧的处理方式）时出生前后总电荷不变，
    那个 −e 丢掉了。"""
    config, result, calls = trajectory

    def total(defects, occupancy):
        return Q * np.sum(defects.charge_state - occupancy) + np.sum(defects.fixed_charge)

    for k, (sim, _, solved) in enumerate(calls):
        assert result.space_charge[k] == pytest.approx(total(sim.defects, solved.occupancy), rel=1e-12)
    for (before, _, solved), (after, start, _) in zip(calls[:-1], calls[1:]):
        born = total(after.defects, start) - total(before.defects, solved.occupancy)
        assert born == pytest.approx(-Q, rel=1e-9)
        # 只算随占据变的 q·(Z − p)：新空位出生占据 1、Z = 1，这部分不变
        occupancy_part = np.sum(after.defects.charge_state - start) - np.sum(before.defects.charge_state - solved.occupancy)
        assert occupancy_part == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("thermal_conductivity", [None, 1.0])
def test_zero_generation_reproduces_steady_state_bitwise(thermal_conductivity):
    """N_T = 0（G = 0）：没有生成事件，I(t) 在 [0, stop_time] 上恒定，且逐位等于同一初始实现下的
    稳态平均场结果。开热反馈时第一条记录的输运仍取环境温度，结论不变。"""
    config = _config(precursor_concentration=0.0, thermal_conductivity=thermal_conductivity)
    result = run_tddb(config)
    steady = solve_mean_field(KMCSimulation(config))
    assert result.stop_reason == "no_generation"
    assert result.end_time == config.tddb.stop_time
    assert result.time.tolist() == [0.0] and result.generation_rate.tolist() == [0.0]
    assert result.current[0] == steady.current
    for name in MECHANISMS:
        assert result.by_mechanism[name][0] == steady.by_mechanism[name]
    assert np.array_equal(result.occupancy, steady.occupancy)
    assert np.all(result.current_at([0.0, 1.0, config.tddb.stop_time]) == steady.current)


def test_each_step_matches_a_fresh_simulation_of_the_current_defects(trajectory):
    """每步的强耦合簇与对当前缺陷集合新建的 KMCSimulation 相同；终态电流与从头求解的稳态一致。"""
    config, result, calls = trajectory
    assert result.stop_reason == "max_generated"
    assert any(len(sim.cluster_members) > 0 for sim, _, _ in calls)
    for sim, _, _ in calls:
        fresh = KMCSimulation(config, defects=_subset(result.defects, sim.n_defects))
        assert [m.tolist() for m in sim.cluster_members] == [m.tolist() for m in fresh.cluster_members]

    final = solve_mean_field(KMCSimulation(config, defects=result.defects))
    assert final.converged and result.converged[-1]
    assert result.current[-1] == pytest.approx(final.current, rel=1e-8)
    assert result.occupancy == pytest.approx(final.occupancy, rel=1e-7, abs=1e-12)


def test_trajectory_bookkeeping(trajectory):
    """时间递增、出生时刻 = 对应记录的时刻、新空位落在可采样窗口里且满足 min_separation，
    分解之和 = 总电流，诊断量自洽。"""
    config, result, _ = trajectory
    device = config.device
    n_initial = result.n_initial
    assert np.all(np.diff(result.time) > 0.0)
    assert result.end_time == result.time[-1]
    assert result.n_generated.tolist() == list(range(config.tddb.max_generated_vacancies + 1))
    assert np.array_equal(result.birth_time[:n_initial], np.zeros(n_initial))
    assert np.array_equal(result.birth_time[n_initial:], result.time[1:])

    standoff = _global_standoff(device)
    positions = result.defects.positions
    new = positions[n_initial:]
    assert np.all((new[:, 2] >= standoff) & (new[:, 2] <= device.Lz - standoff))
    for k, point in enumerate(new):
        others = positions[: n_initial + k]
        distance = np.linalg.norm(minimum_image_delta(others - point, device.Lx, device.Ly), axis=1)
        assert distance.min() >= standoff

    total = sum(result.by_mechanism[name] for name in MECHANISMS)
    np.testing.assert_allclose(total, result.current, rtol=1e-12)
    assert np.all(result.generation_rate > 0.0)
    assert np.all(result.max_cluster_size <= result.largest_component)
    final_charge = Q * np.sum(result.defects.charge_state - result.occupancy) + np.sum(result.defects.fixed_charge)
    assert result.space_charge[-1] == pytest.approx(final_charge, rel=1e-12)
    assert result.empty_generated[-1] == pytest.approx(np.sum(1.0 - result.occupancy[n_initial:]), rel=1e-12)
    assert result.empty_generated[0] == 0.0

    # 等温：温度全是环境值；电流远低于击穿电流密度，没有击穿
    ambient = config.simulation.temperature
    assert not result.thermal
    for name in ("transport_temperature", "effective_temperature", "peak_temperature"):
        assert np.all(getattr(result, name) == ambient), name
    assert result.breakdown_time is None and result.stop_reason == "max_generated"
    for k, size in enumerate(result.dense_cluster_size):
        assert size == np.bincount(distance_clusters(positions[: n_initial + k], device, 0.6e-9)).max()


@pytest.mark.parametrize("width", [0.0, 0.5e-9])
def test_electrode_generation_nodes_match_engine_capture_rates(monkeypatch, width):
    """单元事件的被积函数 N_T·R_1e,电极·P_bb（ADR-044）：在求积节点上，R_1e,电极 与引擎在该点放一个前驱参数陷阱时
    的两侧 R2 + R4 之和一致（插值表误差内）；局域电势、局域场都含当前占据下的空间电荷（§6.1）。P_bb 取泊松解的场在
    Gaussian 窗口上的平均（ADR-043）；窗口宽度 0 时就是引擎在探针处读到的局域场。

    引擎的 R4 用 12 个声子项，这里把前驱的声子项数也设成 12 再逐项对拍。探针陷阱 Z = 0、
    占据 0，不带电，不改变别的缺陷感受到的静电环境。三分之一的缺陷带与之重合的固定电荷 −e，另有一个
    离开缺陷 0.6 nm：生成率的泊松源项（固定电荷放在各自的位置）与引擎的静电背景一致（ADR-048）。
    """
    monkeypatch.setattr(tddb_module, "_PRECURSOR_N_PHONON", _N_PHONON)
    config = _config(concentration_cm3=3.0e19, field_averaging_width=width)
    tddb = config.tddb
    material = config.device.stack.single_material
    band_reference = float(config.device.layer_band_references()[0])
    cells = generation_cells(config.device, tddb.generation_grid)
    defects = KMCSimulation(config).defects
    fixed = np.where(np.arange(len(defects)) % 3 == 0, -Q, 0.0)
    offset = np.zeros((len(defects), 3))
    fixed[1] = -Q
    offset[1, 2] = 0.6e-9 if defects.positions[1, 2] < 0.5 * config.device.Lz else -0.6e-9
    inner = KMCSimulation(config, defects=replace(defects, fixed_charge=fixed, fixed_charge_offset=offset))
    occupancy = solve_mean_field(inner).occupancy
    electrode = generation_rates(inner, occupancy, tddb, cells).cells

    levels = np.array([0, 3, electrode._levels.size // 2, electrode._levels.size - 1])
    columns = np.array([0, 777, electrode._columns.shape[0] // 2, electrode._columns.shape[0] - 1])
    positions = np.column_stack([electrode._columns[columns], electrode._levels[levels]])
    probes = DefectEnsemble.uniform(
        positions,
        np.full(levels.size, tddb.precursor_ionization_energy),
        replace(material, huang_rhys=tddb.precursor_relaxation_energy / material.phonon_energy),
        band_reference=band_reference,
        charge_state=0.0,
    )
    extended = KMCSimulation(config, defects=append_defects(inner.defects, probes))
    occ = np.concatenate([occupancy, np.zeros(levels.size)])
    energy, field, barrier_c, barrier_a = extended._trap_state(occ)
    blocks = extended._electrode_blocks(energy, field, barrier_c, barrier_a)
    probe = np.arange(inner.n_defects, extended.n_defects)
    capture = (blocks[_B_R2C] + blocks[_B_R2A] + blocks[_B_R4C] + blocks[_B_R4A])[probe]
    bond_field = field[probe]
    if width > 0.0:
        solver = inner.solver
        phi = solver.solve(
            solver.deposit(inner.defects.positions, Q * (inner.defects.charge_state - occupancy))
            + solver.deposit(inner.defects.fixed_charge_positions, inner.defects.fixed_charge),
            tddb_module.top_boundary_potential(config.device, config.simulation.voltage),
        )
        mean = solver.gaussian_average(np.stack(solver.electric_field(phi), axis=-1), width)
        bond_field = np.linalg.norm(solver.interpolate(mean, positions), axis=-1)
        assert not np.array_equal(bond_field, field[probe])
    p_bb = bond_break_probability(
        tddb.activation_energy, tddb.dipole_moment, material.kappa, bond_field, config.simulation.temperature
    )
    expected = tddb.precursor_concentration * capture * p_bb
    actual = tddb.precursor_concentration * np.exp(
        electrode._log_rate[levels, columns] - np.maximum(0.0, electrode._barrier[levels, columns])
    )
    assert np.all(expected > 0.0)
    np.testing.assert_allclose(actual, expected, rtol=2e-3)


def _uniform_field_density(config, inner, z):
    """均匀外场里（缺陷都不带电）单元事件的被积函数 N_T·R_1e,电极·P_bb，只依赖高度 z，直接求值。"""
    device, tddb = config.device, config.tddb
    material = device.stack.single_material
    voltage, temperature = config.simulation.voltage, config.simulation.temperature
    solver = inner.solver
    phi = solver.solve(np.zeros(solver.grid.shape), tddb_module.top_boundary_potential(device, voltage))
    F = abs(tddb_module.top_boundary_potential(device, voltage)) / device.Lz
    z = np.atleast_1d(z)
    conduction = device.band_reference_profile(z) - Q * F * z
    rate = single_electron_capture_rate(
        conduction,
        np.full(z.size, F),
        tddb_module._capture_electrodes(device, solver, phi, voltage, z),
        tddb.precursor_ionization_energy,
        tddb.precursor_relaxation_energy,
        material.phonon_energy,
        material.band_gap,
        material.m_di,
        temperature,
        tddb_module._PRECURSOR_N_PHONON,
    )
    p_bb = bond_break_probability(tddb.activation_energy, tddb.dipole_moment, material.kappa, F, temperature)
    return tddb.precursor_concentration * rate * p_bb


def test_electrode_generation_is_the_admissible_volume_integral(monkeypatch):
    """单元事件是可放置区域上的体积分（ADR-044）。均匀外场里被积函数 f 只依赖 z，与被积函数的直接求值对照：
    (1) 远离阴极的中性缺陷挡住的部分可以忽略时，ΣG_单元 = Lx·Ly·∫_窗口 f dz。误差来自 z 向对数线性插值（ln f 在阴极
        附近有弯曲），随节点间距二阶下降：默认 0.1 nm 时实测 −0.78%（判据 1%），0.05 nm 时 −0.20%（判据 0.3%）；
    (2) 同一个中性缺陷放在 z₀ = 1 nm 与贴近阳极处，ΣG_单元 之差就是禁区球挡住的 ∫ f(z)·π·(s² − (z−z₀)²) dz。
        禁区圆盘的横截面由列中点给出，默认列间距 0.08 nm 下实测 −0.25%（判据 2%）。"""
    config = _config()
    device = config.device
    standoff = _global_standoff(device)
    cells = generation_cells(device, config.tddb.generation_grid)
    z_far, z_near = 7.4e-9, 1.0e-9

    def total(z_defect):
        inner = KMCSimulation(config, defects=_neutral_sources(config, [10.03e-9, 9.97e-9, z_defect], 1.15))
        return generation_rates(inner, np.ones(1), config.tddb, cells).cells.rates.sum(), inner

    far, inner = total(z_far)
    window = device.Lx * device.Ly * integrate.quad(
        lambda z: float(_uniform_field_density(config, inner, z)[0]),
        standoff, device.Lz - standoff, epsabs=0.0, epsrel=1e-10, limit=400,
    )[0]
    assert far == pytest.approx(window, rel=1e-2)
    blocked = integrate.quad(
        lambda z: float(_uniform_field_density(config, inner, z)[0]) * np.pi * (standoff ** 2 - (z - z_near) ** 2),
        z_near - standoff, z_near + standoff, epsabs=0.0, epsrel=1e-10,
    )[0]
    assert far - total(z_near)[0] == pytest.approx(blocked, rel=2e-2)
    monkeypatch.setattr(tddb_module, "_LEVEL_STEP", tddb_module._LEVEL_STEP / 2.0)
    assert total(z_far)[0] == pytest.approx(window, rel=3e-3)


def test_electrode_generation_is_independent_of_generation_grid():
    """单元事件的求积与生成网格无关（ADR-044）：生成单元只是事件分组，ΣG 在几种网格下相同；细网格单元按
    粗网格分组求和后逐个等于粗网格单元的速率（空间分布也相同）。"""
    config = _config(concentration_cm3=3.0e19)
    inner = KMCSimulation(config)
    occupancy = solve_mean_field(inner).occupancy
    coarse = generation_rates(inner, occupancy, config.tddb, generation_cells(config.device, (2, 2, 14))).cells.rates
    fine = generation_rates(inner, occupancy, config.tddb, generation_cells(config.device, (4, 4, 28))).cells.rates
    other = generation_rates(inner, occupancy, config.tddb, generation_cells(config.device, (3, 5, 7))).cells.rates
    assert np.all(coarse > 0.0)
    assert fine.sum() == pytest.approx(coarse.sum(), rel=1e-12)
    assert other.sum() == pytest.approx(coarse.sum(), rel=1e-12)
    grouped = fine.reshape(2, 2, 2, 2, 14, 2).sum(axis=(1, 3, 5)).ravel()
    np.testing.assert_allclose(grouped, coarse, rtol=1e-12)


def test_electrode_generation_samples_follow_the_integrand():
    """单元事件的落点（ADR-044）：都在可放置区域里，沿 z 的分布与被积函数一致——均匀外场里一个中性缺陷挡住单元的
    一部分，3000 个落点的 ⟨z⟩ 等于 ∫ z·f(z)·A(z) dz / ∫ f(z)·A(z) dz（A 是该高度上单元的可放置面积）。"""
    config = _config()
    device = config.device
    standoff = _global_standoff(device)
    center = np.array([5.03e-9, 4.97e-9, 1.0e-9])
    inner = KMCSimulation(config, defects=_neutral_sources(config, center, 1.15))
    cells = generation_cells(device, (2, 2, 14))  # 单元 10×10×0.5 nm，z 第 1 层是 [1.0, 1.5] nm
    electrode = generation_rates(inner, np.ones(1), config.tddb, cells).cells
    cell = 1
    lower, upper = cells.lower[cell, 2], cells.lower[cell, 2] + cells.size[2]
    rng = np.random.default_rng(3)
    samples = np.array([electrode.sample(cell, rng) for _ in range(3000)])
    distance = np.linalg.norm(minimum_image_delta(samples - center, device.Lx, device.Ly), axis=-1)
    assert np.all(distance >= standoff)
    assert np.all((samples[:, 2] >= lower) & (samples[:, 2] <= upper))

    def weight(z, moment):
        area = cells.size[0] * cells.size[1] - np.pi * max(0.0, standoff ** 2 - (z - center[2]) ** 2)
        return z ** moment * float(_uniform_field_density(config, inner, z)[0]) * area

    norm = integrate.quad(weight, lower, upper, args=(0,), points=[center[2] + standoff], epsrel=1e-10)[0]
    mean = integrate.quad(weight, lower, upper, args=(1,), points=[center[2] + standoff], epsrel=1e-10)[0] / norm
    z = samples[:, 2]
    assert z.mean() == pytest.approx(mean, abs=4.0 * z.std() / np.sqrt(z.size))


def _neutral_sources(config, positions, ED_eV):
    """中性源陷阱（Z = +1、占据 1 时净电荷 0）。"""
    material = config.device.stack.single_material
    return DefectEnsemble.uniform(
        np.atleast_2d(positions),
        np.atleast_1d(ED_eV) * EV,
        material,
        band_reference=float(config.device.layer_band_references()[0]),
    )


def _isolated_source_reference(config, inner, moment=None):
    """孤立中性源陷阱（p = 1）在均匀外场里的 N_T·∫_A R7·P_bb·w d³x，scipy 自适应积分到 1e-10。

    电势就是外加偏压的线性解，被积函数只依赖 (r, u = cosθ)：Θ = exp(−max(0, ΔE₀ − e·F·r·u)/k_BT)，P_bb 是
    常数，A 是 r ≥ min_separation 且 z ∈ [standoff, Lz − standoff]。权重 w 取 1（`moment` 为 None）、r（"r"）或 u（"u"）。
    """
    device, tddb = config.device, config.tddb
    material = device.stack.single_material
    temperature = config.simulation.temperature
    standoff = _global_standoff(device)
    z_i = float(inner.defects.positions[0, 2])
    F = tddb_module.top_boundary_potential(device, config.simulation.voltage) / device.Lz
    E_C_i = float(device.band_reference_profile(np.array([z_i]))[0]) - Q * F * z_i
    delta0 = E_C_i - tddb.precursor_ionization_energy - inner._trap_state(np.ones(1))[0][0]
    alpha = 1.0 / inner.defects.r_D[0] + 1.0 / localization_radius(tddb.precursor_ionization_energy, material.m_di)
    kT = constants.Boltzmann * temperature
    p_bb = float(bond_break_probability(tddb.activation_energy, tddb.dipole_moment, material.kappa, F, temperature))

    def radial(u):
        hi = 4e-9
        if u < 0.0:
            hi = min(hi, (z_i - standoff) / -u)
        elif u > 0.0:
            hi = min(hi, (device.Lz - standoff - z_i) / u)
        if hi <= standoff:
            return 0.0

        def f(r):
            weight = r if moment == "r" else (u if moment == "u" else 1.0)
            return weight * r * r * np.exp(-alpha * r - max(0.0, delta0 - Q * F * r * u) / kT)

        kink = delta0 / (Q * F * u) if u > 0.0 and delta0 > 0.0 else None
        points = [kink] if kink is not None and standoff < kink < hi else None
        return integrate.quad(f, standoff, hi, points=points, epsabs=0.0, epsrel=1e-12, limit=200)[0]

    breaks = [b for b in ((standoff - z_i) / standoff, 0.0) if -1.0 < b < 1.0]
    value = integrate.quad(radial, -1.0, 1.0, points=breaks, epsabs=0.0, epsrel=1e-10, limit=400)[0]
    return tddb.precursor_concentration * 2.0 * np.pi * inner.defects.attempt_frequency[0] * p_bb * value


@pytest.mark.parametrize(
    "z_nm, ED_eV, rtol",
    [
        (4.0, 1.15, 2e-3),  # 能级与前驱相当，被积函数集中在 min_separation 那一圈
        (4.0, 1.45, 6e-3),  # 深能级：近处上坡，峰值落在下游 MA 因子刚到 1 的折点上
        (0.62, 0.90, 2e-3),  # 贴着可采样窗口下沿：球的下半部分被窗口平面截掉
    ],
)
def test_trap_generation_matches_adaptive_reference_for_isolated_source(z_nm, ED_eV, rtol):
    """陷阱项（ADR-042）：一个中性源陷阱（p = 1）在均匀外场里的事件速率等于 N_T·∫_A R7·P_bb d³x 的
    自适应积分精确值。误差来自径向步长，随步长二阶下降（ADR-042 的收敛性表）。"""
    config = _config()
    inner = KMCSimulation(config, defects=_neutral_sources(config, [10e-9, 10e-9, z_nm * 1e-9], ED_eV))
    cells = generation_cells(config.device, config.tddb.generation_grid)
    traps = generation_rates(inner, np.ones(1), config.tddb, cells).traps
    assert traps.rates[0] == pytest.approx(_isolated_source_reference(config, inner), rel=rtol)


def test_trap_generation_is_the_fine_cell_limit_of_the_cell_center_sum(monkeypatch):
    """有近旁缺陷挡住一部分区域（一个中性、一个空着带 +e）时，源陷阱积分就是 ADR-039 的单元中心求和
    Σ_单元 N_T·V_cell·p·R7·P_bb 在单元加密下的极限：两种独立的求积——Cartesian 单元中心求和（0.025 nm）与加密一级的
    源陷阱积分（0.02 nm / 48 / 48）——相差 ≤ 1%；默认求积参数与加密一级相差 ≤ 2%（ADR-042 的收敛性表）。"""
    config = _config()
    device = config.device
    standoff = _global_standoff(device)
    center = np.array([10e-9, 10e-9, 3e-9])
    positions = np.array([center, center + [0.6e-9, 0.0, 0.1e-9], center + [-0.2e-9, 0.7e-9, -0.3e-9]])
    inner = KMCSimulation(config, defects=_neutral_sources(config, positions, [1.15, 1.15, 1.15]))
    occupancy = np.array([1.0, 1.0, 0.0])  # 第三个空着、带 +e，不是源陷阱
    cells = generation_cells(device, config.tddb.generation_grid)
    default = generation_rates(inner, occupancy, config.tddb, cells).traps
    assert default.rates[2] == 0.0 and default.rates[0] > 0.0
    monkeypatch.setattr(tddb_module, "_TRAP_RADIAL_STEP", 0.02e-9)
    monkeypatch.setattr(tddb_module, "_TRAP_POLAR_BINS", 48)
    monkeypatch.setattr(tddb_module, "_TRAP_AZIMUTH_BINS", 48)
    fine = generation_rates(inner, occupancy, config.tddb, cells).traps

    h, reach = 0.025e-9, 1.6e-9  # 被积函数在 min_separation 外每 0.083 nm 衰减 e 倍，1.6 nm 处已低于 1e-5
    axis = np.arange(-reach + 0.5 * h, reach, h)
    offsets = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    distance = np.linalg.norm(offsets, axis=-1)
    outside = distance >= standoff
    points, distance = center + offsets[outside], distance[outside]
    keep, _ = fine._admissible(points, np.zeros(points.shape[0], dtype=int))
    values, _, _ = fine._integrand(np.zeros(int(keep.sum()), dtype=int), points[keep], distance[keep])
    cell_sum = config.tddb.precursor_concentration * h ** 3 * np.sum(values / distance[keep] ** 2)
    assert cell_sum == pytest.approx(fine.rates[0], rel=1e-2)
    assert default.rates[0] == pytest.approx(fine.rates[0], rel=2e-2)


def test_trap_generation_is_independent_of_generation_grid_and_vanishes_for_empty_sources():
    """陷阱项不经过生成单元：两种生成网格下逐位相同；源陷阱全空（p = 0）时全部为 0。"""
    config = _config(concentration_cm3=3.0e19)
    inner = KMCSimulation(config)
    occupancy = solve_mean_field(inner).occupancy
    coarse = generation_rates(inner, occupancy, config.tddb, generation_cells(config.device, (2, 2, 14)))
    fine = generation_rates(inner, occupancy, config.tddb, generation_cells(config.device, (4, 4, 20)))
    assert np.any(coarse.traps.rates > 0.0)
    assert np.array_equal(coarse.traps.rates, fine.traps.rates)
    cells = generation_cells(config.device, (2, 2, 14))
    empty = generation_rates(inner, np.zeros(inner.n_defects), config.tddb, cells)
    assert np.all(empty.traps.rates == 0.0)


def test_trap_generation_samples_follow_the_integrand():
    """源陷阱事件的落点（ADR-042）：都在可放置区域里；深能级源陷阱（近处上坡）的落点偏向下游，⟨r⟩、⟨u⟩
    与自适应积分给出的被积函数一阶矩一致（3000 个样本的统计误差内）。"""
    config = _config()
    device = config.device
    inner = KMCSimulation(config, defects=_neutral_sources(config, [10e-9, 10e-9, 4e-9], 1.40))
    cells = generation_cells(device, config.tddb.generation_grid)
    traps = generation_rates(inner, np.ones(1), config.tddb, cells).traps
    rng = np.random.default_rng(3)
    samples = np.array([traps.sample(0, rng) for _ in range(3000)])
    delta = minimum_image_delta(samples - inner.defects.positions[0], device.Lx, device.Ly)
    r = np.linalg.norm(delta, axis=-1)
    u = delta[:, 2] / r
    standoff = _global_standoff(device)
    assert np.all(r >= standoff)
    assert np.all((samples[:, 2] >= standoff) & (samples[:, 2] <= device.Lz - standoff))

    norm = _isolated_source_reference(config, inner)
    mean_r = _isolated_source_reference(config, inner, moment="r") / norm
    mean_u = _isolated_source_reference(config, inner, moment="u") / norm
    assert mean_u > 0.3  # 下游（阳极方向）占优
    assert r.mean() == pytest.approx(mean_r, abs=4.0 * r.std() / np.sqrt(r.size) + 2e-12)
    assert u.mean() == pytest.approx(mean_u, abs=4.0 * u.std() / np.sqrt(u.size) + 0.01)


def test_precursor_concentration_only_rescales_time():
    """N_T 只是生成率的前置因子：同一种子下 N_T 翻 10 倍，事件序列与电流逐位相同、时刻缩为 1/10。"""
    base = run_tddb(_config(max_generated_vacancies=3))
    scaled = run_tddb(_config(max_generated_vacancies=3, precursor_concentration=4.0e20 * PER_CM3))
    assert np.array_equal(scaled.current, base.current)
    assert np.array_equal(scaled.defects.positions, base.defects.positions)
    np.testing.assert_allclose(scaled.time * 10.0, base.time, rtol=1e-12)
    np.testing.assert_allclose(scaled.generation_rate, 10.0 * base.generation_rate, rtol=1e-12)


def test_generation_rates_follow_local_temperature():
    """局域温度（§6.6）：温度场处处取环境温度时单元事件与源陷阱事件都退回缺省（按工作点温度求值）；温度场不均匀时
    单元事件的每个求积节点按该节点的温度求值，等于整张温度场取这个节点温度时同一节点的值（插值表误差内）。"""
    config = _config()
    inner = KMCSimulation(config)
    occupancy = solve_mean_field(inner).occupancy
    cells = generation_cells(config.device, config.tddb.generation_grid)
    ambient = config.simulation.temperature
    shape = inner.solver.grid.shape
    base = generation_rates(inner, occupancy, config.tddb, cells)
    flat = generation_rates(inner, occupancy, config.tddb, cells, np.full(shape, ambient))
    assert np.any(base.traps.rates > 0.0)
    np.testing.assert_allclose(flat.cells.rates, base.cells.rates, rtol=1e-12)
    np.testing.assert_allclose(flat.traps.rates, base.traps.rates, rtol=1e-12)

    field = ambient + np.random.default_rng(4).uniform(0.0, 80.0, shape)
    local = generation_rates(inner, occupancy, config.tddb, cells, field).cells
    assert not np.allclose(local.rates, base.cells.rates)
    for level, column in ((2, 0), (20, 9000), (60, 25000)):
        point = np.append(local._columns[column], local._levels[level])
        node_temperature = float(inner.solver.interpolate(field, point)[0])
        uniform = generation_rates(inner, occupancy, config.tddb, cells, np.full(shape, node_temperature)).cells
        assert local._log_rate[level, column] == pytest.approx(uniform._log_rate[level, column], abs=1e-3)
        assert local._barrier[level, column] == pytest.approx(uniform._barrier[level, column], rel=1e-12)


def test_thermal_feedback_feeds_previous_power_into_transport_and_generation(monkeypatch):
    """开热反馈（§6.6、§8.4）：第 k 步的内层输运用第 k−1 步功率给出的缺陷处加权温度（第 0 步取
    环境温度）；每步记录的功率、加权温度与温度场最高值就是该步稳态的 ΔE·R 解热方程的结果；
    生成率取同一步的温度场。k_TH 取得很小，升温到开尔文量级，数值上看得出来。"""
    k_th = 1.0e-6
    config = _config(seed=1, concentration_cm3=3.0e19, thermal_conductivity=k_th, max_generated_vacancies=3)
    calls, temperature_fields = [], []
    original_mf, original_rates = tddb_module.solve_mean_field, tddb_module.generation_rates

    def recording_mf(sim, occupancy=None):
        result = original_mf(sim, occupancy)
        calls.append((sim, result))
        return result

    def recording_rates(inner, occupancy, tddb, cells, temperature_field=None):
        temperature_fields.append(temperature_field)
        return original_rates(inner, occupancy, tddb, cells, temperature_field)

    monkeypatch.setattr(tddb_module, "solve_mean_field", recording_mf)
    monkeypatch.setattr(tddb_module, "generation_rates", recording_rates)
    result = run_tddb(config)

    ambient = config.simulation.temperature
    assert result.thermal and result.stop_reason == "max_generated"
    assert result.transport_temperature[0] == ambient
    assert np.array_equal(result.transport_temperature[1:], result.effective_temperature[:-1])
    assert np.all(result.effective_temperature > ambient + 1.0)
    assert np.all(result.peak_temperature >= result.effective_temperature)
    for k, ((sim, mf), temperature_field) in enumerate(zip(calls, temperature_fields)):
        assert sim.sim.temperature == result.transport_temperature[k]
        heat = heat_release(sim, mf.occupancy)
        field = solve_temperature(sim.solver, power_map(sim.solver, sim.defects.positions, heat.defect), k_th, ambient)
        assert result.power[k] == heat.dielectric
        assert result.effective_temperature[k] == transport_temperature(
            sim.solver, field, sim.defects.positions, heat.defect, ambient
        )
        assert result.peak_temperature[k] == field.max()
        assert np.array_equal(temperature_field, field)


def test_distance_clusters_link_defects_closer_than_the_cutoff():
    """几何团簇（致密团簇标志用）：间距 < 截断的缺陷连边，xy 取最小镜像。"""
    device = make_device()  # 20×20×8 nm
    z = np.linspace(0.6e-9, 7.4e-9, 10)  # 相邻间距 0.756 nm
    x = np.where(np.arange(10) % 2 == 0, 19.9e-9, 0.1e-9)  # 横向来回跨过周期边界，最小镜像下只差 0.2 nm
    chain = np.column_stack([x, np.full(10, 10e-9), z])
    labels = distance_clusters(chain, device, 0.9e-9)
    assert np.all(labels == labels[0])
    # 截断小于链上的间距：每个缺陷各成一个分量
    assert np.unique(distance_clusters(chain, device, 0.7e-9)).size == 10
    # 去掉中间一个：断成两段
    assert np.unique(distance_clusters(np.delete(chain, 5, axis=0), device, 0.9e-9)).size == 2
    assert distance_clusters(np.zeros((0, 3)), device, 0.9e-9).size == 0


def test_trajectory_breaks_down_when_current_density_exceeds_the_threshold(trajectory):
    """击穿判据（ADR-047）：轨迹停在电流密度第一次超过击穿电流密度的记录上，t_BD 就是这条记录的
    时刻，之前的记录与不停的轨迹逐位相同；J(0) 就超过时停在 t = 0；等于阈值不算超过。"""
    config, reference, _ = trajectory
    density = reference.current_density
    threshold = 0.5 * (density[1] + density[2])
    assert np.all(density[:2] < threshold) and density[2] > threshold
    result = run_tddb(replace(config, tddb=replace(config.tddb, breakdown_current_density=threshold)))
    assert result.stop_reason == "breakdown"
    assert result.breakdown_time == result.end_time == reference.time[2]
    assert np.array_equal(result.time, reference.time[:3])
    assert np.array_equal(result.current, reference.current[:3])

    at_start = run_tddb(replace(config, tddb=replace(config.tddb, breakdown_current_density=0.5 * density[0])))
    assert at_start.stop_reason == "breakdown"
    assert at_start.time.tolist() == [0.0] and at_start.breakdown_time == 0.0

    at_peak = run_tddb(replace(config, tddb=replace(config.tddb, breakdown_current_density=density.max())))
    assert at_peak.stop_reason == "max_generated" and at_peak.breakdown_time is None
    assert np.array_equal(at_peak.current, reference.current)


class _StubEvents:
    """只有速率与落点的事件源：`places[i]` 为 None 表示第 i 个事件放不下新空位。"""

    def __init__(self, places):
        self.rates = np.ones(len(places))
        self._places = places

    def sample(self, index, rng):
        return self._places[index]


def test_full_event_is_skipped():
    """选中的事件放不下新空位：记为已满，改从其余事件抽；全部已满或总速率为 0 时返回 None。"""
    point = np.array([2e-9, 2e-9, 2e-9])
    table = tddb_module.GenerationTable(_StubEvents([None, point]), _StubEvents([]))
    full = np.zeros(2, dtype=bool)
    rng = np.random.default_rng(0)
    dt, position = _draw_generation(np.array([1e12, 1.0]), full, table, rng)
    assert full.tolist() == [True, False]
    assert np.array_equal(position, point)
    assert dt > 0.0
    assert _draw_generation(np.zeros(2), full, table, rng) is None


def test_source_event_places_by_its_sampler_and_is_marked_full_when_it_cannot():
    """事件表 = 单元事件 + 源陷阱事件（ADR-042）：选中源陷阱时落点由它的抽样给出（在它周围、满足
    min_separation）；源陷阱放不下新空位时记为已满，剩下没有事件时返回 None。"""
    config = _config()
    device = config.device
    inner = KMCSimulation(config, defects=_neutral_sources(config, [10e-9, 10e-9, 4e-9], 1.15))
    cells = generation_cells(device, config.tddb.generation_grid)
    table = generation_rates(inner, np.ones(1), config.tddb, cells)
    traps = table.traps
    standoff = _global_standoff(device)
    rates = np.concatenate([np.zeros(len(cells)), traps.rates])
    full = np.zeros(rates.size, dtype=bool)
    rng = np.random.default_rng(0)
    dt, position = _draw_generation(rates, full, table, rng)
    assert dt > 0.0 and not full.any()
    assert standoff <= np.linalg.norm(position - inner.defects.positions[0]) < 2e-9

    traps.sample = lambda source, rng: None  # 源陷阱周围放不下
    assert _draw_generation(rates, full, table, rng) is None
    assert full.tolist() == [False] * len(cells) + [True]


def test_trajectory_driven_by_source_events_keeps_bookkeeping(monkeypatch):
    """只留源陷阱事件（单元事件置零）跑轨迹：每个新空位都落在某个已有缺陷的近旁、满足 min_separation 与
    可采样窗口，记录的 ΣG 就是源陷阱事件之和；事件表随缺陷数增长（每个新空位多一条源陷阱事件）。"""
    original = tddb_module.generation_rates
    tables = []

    def traps_only(inner, occupancy, tddb, cells, temperature_field=None):
        table = original(inner, occupancy, tddb, cells, temperature_field)
        tables.append(table)
        table.cells.rates = np.zeros_like(table.cells.rates)
        return table

    monkeypatch.setattr(tddb_module, "generation_rates", traps_only)
    config = _config(seed=1, concentration_cm3=3.0e19, max_generated_vacancies=3)
    result = run_tddb(config)
    device = config.device
    assert result.stop_reason == "max_generated"
    standoff = _global_standoff(device)
    positions = result.defects.positions
    for k in range(3):
        n = result.n_initial + k
        distance = np.linalg.norm(minimum_image_delta(positions[:n] - positions[n], device.Lx, device.Ly), axis=1)
        assert standoff <= distance.min() < 2e-9
        assert standoff <= positions[n, 2] <= device.Lz - standoff
        assert tables[k].traps.rates.size == n
        assert result.generation_rate[k] == pytest.approx(tables[k].traps.rates.sum(), rel=1e-12)


def test_fit_power_law_recovers_exponent_and_rates_noise():
    """纯幂律 I = A·tⁿ 精确还原 n 与 A、R² = 1；log I 加 5% 量级噪声时 n 仍在 ±0.05 内、R² 略低于 1。"""
    t = np.logspace(-3, 3, 61)
    exact = fit_power_law(t, 3.0e-8 * t ** 0.7)
    assert exact.exponent == pytest.approx(0.7, abs=1e-12)
    assert exact.prefactor == pytest.approx(3.0e-8, rel=1e-10)
    assert exact.r_squared == pytest.approx(1.0, abs=1e-12)
    assert fit_power_law(t, np.full(t.size, 2.0)).r_squared == 1.0

    noisy = 3.0e-8 * t ** 0.7 * 10.0 ** np.random.default_rng(5).normal(0.0, 0.05, t.size)
    fit = fit_power_law(t, noisy)
    assert fit.exponent == pytest.approx(0.7, abs=0.05)
    assert 0.9 < fit.r_squared < 1.0
    with pytest.raises(ValueError):
        fit_power_law(np.array([0.0, 1.0]), np.array([1.0, 2.0]))


def test_fit_transient_samples_log_time_between_first_event_and_end(trajectory):
    _, result, _ = trajectory
    fit, times = fit_transient(result, points_per_decade=10)
    assert times[0] == pytest.approx(result.time[1], rel=1e-12)
    assert times[-1] == pytest.approx(result.end_time, rel=1e-12)
    assert np.allclose(np.diff(np.log10(times)), np.log10(times[1] / times[0]))
    reference = fit_power_law(times, result.current_at(times) / result.area)
    assert fit == reference


def test_write_tddb_csv(tmp_path, trajectory):
    _, result, _ = trajectory
    path = tmp_path / "it.csv"
    write_tddb_csv(result, str(path))
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result.time.size
    assert [float(r["t_end_s"]) for r in rows] == pytest.approx(result.segment_end.tolist(), rel=1e-6)
    per_cm2 = constants.centi ** 2
    for k, (row, density) in enumerate(zip(rows, result.current_density)):
        assert float(row["J_total_A_cm2"]) == pytest.approx(density * per_cm2, rel=1e-5)
        parts = sum(float(row["J_{}_A_cm2".format(name)]) for name in MECHANISMS)
        assert parts == pytest.approx(float(row["J_total_A_cm2"]), rel=1e-5)
        assert float(row["power_W"]) == pytest.approx(result.power[k], rel=1e-5)
        assert float(row["T_effective_K"]) == pytest.approx(result.effective_temperature[k], abs=1e-6)
        assert int(row["dense_cluster_size"]) == result.dense_cluster_size[k]
        assert int(row["mf_newton_steps"]) == result.newton_steps[k]


def test_cli_tddb_writes_csv_and_figure(tmp_path, capsys):
    """CLI `tddb`：读 YAML 跑一条开热反馈的轨迹，落 CSV 与 I-t / 温度图。"""
    import yaml

    from kmc_leakage.__main__ import tddb as tddb_command

    raw = {
        "materials": {"ZrO2": {
            "kappa": 38.0, "eps_opt": 5.6, "band_gap_eV": 5.4, "electron_affinity_eV": 2.86,
            "m_cb_me": 1.2, "m_vb_me": 2.5, "phonon_energy_eV": 0.04, "huang_rhys_S": 1.0,
        }},
        "layers": [{"material": "ZrO2", "thickness_nm": 8.0, "defects": {
            "concentration_cm3": 3.0e18, "min_separation_nm": 0.5, "level": {"mode": "fixed", "ED_eV": 1.15},
        }}],
        "device": {"Lx_nm": 20.0, "Ly_nm": 20.0, "barrier_height_eV": 1.74},
        "electrodes": {"cathode": {"work_function_eV": 4.6}, "anode": {"work_function_eV": 4.6}},
        "simulation": {"voltage_V": 3.0, "temperature_C": 25.0, "seed": 3, "grid": {"nx": 8, "ny": 8, "nz": 16}},
        "tddb": {
            "activation_energy_eV": 4.0, "precursor_ionization_energy_eV": 1.15,
            "precursor_relaxation_energy_eV": 1.0, "dipole_moment_eA": 7.1, "field_averaging_width_nm": 0.5,
            "precursor_concentration_cm3": 4.0e19, "generation_grid": {"nx": 2, "ny": 2, "nz": 14},
            "max_generated_vacancies": 2, "stop_time_s": 1.0e4,
            "breakdown_current_density_A_cm2": 1.0e4,
            "thermal_conductivity_W_mK": 1.0,
        },
    }
    config_path = tmp_path / "tddb.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    tddb_command(config=str(config_path), seed=None, out=str(tmp_path / "it.csv"), plot=str(tmp_path / "it.png"))
    assert (tmp_path / "it.png").stat().st_size > 0
    with open(tmp_path / "it.csv", newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    assert "n = " in capsys.readouterr().out
