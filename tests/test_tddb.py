"""TDDB 瞬态驱动（Phase 3/4，ADR-032–036、ADR-052，PROJECT_CONTEXT_TDDB.md §5、§6、§9）。

- 增量静电耦合（§6.4）：逐个追加缺陷后的耦合矩阵与对全体缺陷从头计算的一致；
- 新空位出生（§6.7、ADR-048）：出生时被一个电子占据、与初始缺陷同一物种，带着与它重合的 O_i⁻
  （q_fix = −e）；出生时介质总电荷正好多一个 −e，其余缺陷处的电势与电场的变化就是这个 −e 的贡献；
  外层每步从上一步的占据热启动；
- G = 0 时 I(t) 恒定，且逐位等于同一初始实现下的稳态结果（等温与开热反馈都成立）；
- 每步的强耦合簇与对当前缺陷集合新建 `KMCSimulation` 的分簇相同，终态电流与对终态缺陷
  从头求解的稳态一致；
- 离散前驱位点（§6.1，ADR-052/055）：t = 0 在可采样窗口里做最小间隔 min_separation 的最大随机堆积，位点两两之间、
  与初始缺陷之间都 ≥ min_separation，且再也放不下一个位点；新空位落在某个位点上，只用掉这一个位点；
- 每个位点的生成速率 G_i = R_1e,i·P_bb,i：电极项逐点求值时等于引擎在该点放一个前驱参数陷阱时的两侧 R2 + R4（含空间
  电荷），插值表与逐点求值的差别在表的精度内（ADR-055）；陷阱项等于对全部已占据陷阱的直接求和（截断误差
  ≤ `_TRAP_TERM_FLOOR`），P_bb 取配置的 E_A（全部位点相同，ADR-053）与 Gaussian 平均场；三项都按位点处的局域温度求值；
- P_bb 未饱和时 E_A 整体平移只改时间标尺；Gillespie 按速率选位点；
- 热反馈（§6.6、§8.4）：每步的输运温度是上一步功率给出的缺陷处加权温度，生成率取同一步的温度场；
- 导电簇与击穿判据（§6.3，ADR-054/056）：间距 < r_c 的空位连成导电簇，离电极 < r_c 的接到该电极；接电极的导电簇
  在生成率的电势里与该电极等电位，并作为电极费米能级处占满的电子源进陷阱项；悬空导电簇自身等电位、总电荷守恒；
  轨迹停在第一条出现贯穿通路的记录上，
  这条记录的电流加上细丝的 G₀·V；
- 续跑（ADR-057）：每步开始时的状态存盘后逐位还原，从中途状态续跑与不停跑的同一条轨迹只差舍入误差；
- 致密团簇标志：间距 < 截断的缺陷连边，xy 取最小镜像；
- I-t 幂律拟合对纯幂律精确还原 n，对带噪声的幂律给出合理的拟合优度。
"""

import copy
import csv
from dataclasses import fields, replace

import numpy as np
import pytest
from scipy import constants
from scipy.spatial import cKDTree

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
from kmc_leakage.rates import bond_break_probability, trap_precursor_capture_rate
from kmc_leakage.tddb import (
    _QUANTUM_CONDUCTANCE,
    GenerationRates,
    IncrementalCoupling,
    Precursors,
    _draw_generation,
    conducting_clusters,
    conductor_gap,
    distance_clusters,
    fit_power_law,
    fit_transient,
    generation_rates,
    initial_state,
    load_state,
    place_precursors,
    run_tddb,
    save_state,
    write_tddb_csv,
)
from kmc_leakage.thermal import heat_release, power_map, solve_temperature, transport_temperature

EV = constants.electron_volt
Q = constants.elementary_charge
E_ANGSTROM = Q * constants.angstrom


def _tddb(**overrides) -> TDDBConfig:
    params = dict(
        activation_energy=4.0 * EV,
        precursor_ionization_energy=1.15 * EV,
        precursor_relaxation_energy=1.0 * EV,
        dipole_moment=7.1 * E_ANGSTROM,
        field_averaging_width=0.5e-9,
        max_generated_vacancies=4,
        stop_time=1.0e4,
        # r_c 小于 min_separation（0.5 nm）：空位之间、空位与电极都连不上，没有导电簇，也不会击穿
        percolation_distance=0.4e-9,
    )
    params.update(overrides)
    return TDDBConfig(**params)


def _config(seed=3, concentration_cm3=3.0e18, **overrides):
    """20×20×8 nm、网格 8/8/16、3 V / 25 °C；3×10¹⁸ cm⁻³ 即 10 个初始缺陷，前驱位点按 min_separation = 0.5 nm
    铺满窗口，约 1.7 万个。"""
    device = make_device(defects=make_defect_config(concentration_cm3=concentration_cm3))
    return replace(make_config(voltage=3.0, seed=seed, device=device), tddb=_tddb(**overrides))


def _subset(defects: DefectEnsemble, n: int) -> DefectEnsemble:
    return DefectEnsemble(**{item.name: getattr(defects, item.name)[:n] for item in fields(DefectEnsemble)})


@pytest.fixture(scope="module")
def trajectory():
    """4 次生成的短轨迹，并记下每步交给平均场的缺陷实现与初值、交给生成率的前驱位点表与算出的速率。

    初始缺陷取 3×10¹⁹ cm⁻³（96 个），一开始就有多个强耦合簇，逐步对比分簇才有内容。
    """
    config = _config(seed=1, concentration_cm3=3.0e19)
    calls, generations = [], []
    original_mf, original_rates = tddb_module.solve_mean_field, tddb_module.generation_rates

    def recording(sim, occupancy=None):
        result = original_mf(sim, occupancy)
        calls.append((sim, None if occupancy is None else occupancy.copy(), result))
        return result

    def recording_rates(inner, occupancy, tddb, precursors, temperature_field=None):
        result = original_rates(inner, occupancy, tddb, precursors, temperature_field)
        generations.append((precursors, result))
        return result

    tddb_module.solve_mean_field = recording
    tddb_module.generation_rates = recording_rates
    try:
        result = run_tddb(config)
    finally:
        tddb_module.solve_mean_field = original_mf
        tddb_module.generation_rates = original_rates
    return config, result, calls, generations


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
    config, result, calls, _ = trajectory
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
    config, result, calls, _ = trajectory

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
    """G = 0（E_A 取 10³ eV，P_bb 下溢为 0）：没有生成事件，I(t) 在 [0, stop_time] 上恒定，且逐位等于同一初始实现下的
    稳态平均场结果。开热反馈时第一条记录的输运仍取环境温度，结论不变。"""
    config = _config(activation_energy=1.0e3 * EV, thermal_conductivity=thermal_conductivity)
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
    config, result, calls, _ = trajectory
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
    config, result, _, _ = trajectory
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
    assert np.all(result.filament_current == 0.0) and np.array_equal(result.transport_current, result.current)
    assert np.all(result.conductor_gap == device.Lz)  # r_c < min_separation：没有导电簇
    assert np.all(result.generation_rate > 0.0)
    assert np.all(np.diff(result.n_precursors) == -1)  # 每次出生用掉一个前驱位点
    assert np.all(result.max_cluster_size <= result.largest_component)
    final_charge = Q * np.sum(result.defects.charge_state - result.occupancy) + np.sum(result.defects.fixed_charge)
    assert result.space_charge[-1] == pytest.approx(final_charge, rel=1e-12)
    assert result.empty_generated[-1] == pytest.approx(np.sum(1.0 - result.occupancy[n_initial:]), rel=1e-12)
    assert result.empty_generated[0] == 0.0

    # 等温：温度全是环境值；没有导电簇，不会击穿
    ambient = config.simulation.temperature
    assert not result.thermal
    for name in ("transport_temperature", "effective_temperature", "peak_temperature"):
        assert np.all(getattr(result, name) == ambient), name
    assert result.breakdown_time is None and result.stop_reason == "max_generated"
    for k, size in enumerate(result.dense_cluster_size):
        assert size == np.bincount(distance_clusters(positions[: n_initial + k], device, 0.6e-9)).max()


def test_precursors_fill_the_sampling_window():
    """t = 0 的前驱位点（ADR-055）：可采样窗口里最小间隔 d = min_separation 的最大随机堆积。位点都在窗口内，两两之间、
    与初始缺陷之间都 ≥ d（xy 最小镜像）；窗口里随机取的点离某个位点或初始缺陷都 < d，即再也放不下一个位点；密度在
    随机顺序堆积的饱和值（0.73/d³）附近。同一个随机数流给出同一组位点；min_separation = 0 时没有间隔可用，报错。"""
    config = _config(concentration_cm3=3.0e19)
    device = config.device
    defects = KMCSimulation(config).defects.positions
    d = _global_standoff(device)
    positions = place_precursors(device, defects, np.random.default_rng(7)).positions
    assert np.all((positions[:, 2] >= d) & (positions[:, 2] <= device.Lz - d))
    assert np.all((positions[:, :2] >= 0.0) & (positions[:, :2] < [device.Lx, device.Ly]))

    box = np.array([device.Lx, device.Ly, 4.0 * device.Lz])
    sites = cKDTree(positions, boxsize=box)
    assert sites.query(positions, k=2)[0][:, 1].min() >= d
    assert cKDTree(np.mod(defects, box), boxsize=box).query(positions)[0].min() >= d
    window = (device.Lz - 2.0 * d) * device.Lx * device.Ly
    assert 0.70 < len(positions) * d ** 3 / window < 0.80

    rng = np.random.default_rng(8)
    probes = np.column_stack([rng.uniform(0.0, device.Lx, 200000), rng.uniform(0.0, device.Ly, 200000),
                              rng.uniform(d, device.Lz - d, 200000)])
    nearest = np.minimum(sites.query(probes)[0], cKDTree(np.mod(defects, box), boxsize=box).query(probes)[0])
    assert nearest.max() < d

    assert np.array_equal(place_precursors(device, defects, np.random.default_rng(7)).positions, positions)
    flat = make_device(defects=make_defect_config(concentration_cm3=3.0e19, min_separation=0.0))
    with pytest.raises(ValueError):
        place_precursors(flat, defects, np.random.default_rng(7))


def test_converted_precursor_is_the_only_site_taken_out():
    """断键的位点用掉，其余位点原样保留、顺序不变：位点两两之间本来就 ≥ min_separation（ADR-055）。"""
    positions = 1e-9 * np.arange(15.0).reshape(5, 3)
    precursors = Precursors(positions)
    left = precursors.converted(2)
    assert len(left) == 4 and len(precursors) == 5
    assert np.array_equal(left.positions, positions[[0, 1, 3, 4]])


def test_each_birth_converts_one_precursor_site(trajectory):
    """新空位就落在上一步的某个前驱位点上（ADR-052）：只用掉这一个位点，其余位点原样留到下一步（ADR-055）；记录的
    ΣG 与剩余位点数就是该步的生成速率与位点表。"""
    config, result, _, generations = trajectory
    assert len(generations) == result.time.size
    for k, (precursors, rates) in enumerate(generations):
        assert result.n_precursors[k] == len(precursors)
        assert result.generation_rate[k] == pytest.approx(rates.rates.sum(), rel=1e-12)
    for k in range(result.time.size - 1):
        before, after = generations[k][0], generations[k + 1][0]
        vacancy = result.defects.positions[result.n_initial + k]
        used = np.flatnonzero(np.all(before.positions == vacancy, axis=1))
        assert used.size == 1
        assert np.array_equal(after.positions, np.delete(before.positions, used[0], axis=0))


def _admissible(device, positions, point):
    """`point` 在可采样窗口内，且离 `positions` 里每个缺陷都 ≥ min_separation（xy 最小镜像）。"""
    standoff = _global_standoff(device)
    if not standoff <= point[2] <= device.Lz - standoff:
        return False
    distance = np.linalg.norm(minimum_image_delta(positions - point, device.Lx, device.Ly), axis=1)
    return bool(distance.min() >= standoff)


def _sites_near_sources(inner, occupancy, rng, n_near=40, n_uniform=40, sources=None):
    """前驱位点：前 `n_near` 个放在 `sources`（缺省为占据 ≥ 0.5 的缺陷）旁 0.5–0.8 nm 处（陷阱项显著），其余 `n_uniform`
    个在可采样窗口里均匀撒；都在窗口内且离全部缺陷 ≥ min_separation。"""
    device = inner.device
    positions = inner.defects.positions
    standoff = _global_standoff(device)
    sources = np.flatnonzero(occupancy >= 0.5) if sources is None else sources
    assert sources.size > 0
    points = []
    while len(points) < n_near:
        direction = rng.normal(size=3)
        point = positions[rng.choice(sources)] + rng.uniform(0.5e-9, 0.8e-9) * direction / np.linalg.norm(direction)
        point[:2] = np.mod(point[:2], [device.Lx, device.Ly])
        if _admissible(device, positions, point):
            points.append(point)
    while len(points) < n_near + n_uniform:
        point = np.array([rng.uniform(0.0, device.Lx), rng.uniform(0.0, device.Ly),
                          rng.uniform(standoff, device.Lz - standoff)])
        if _admissible(device, positions, point):
            points.append(point)
    return Precursors(np.array(points))


def _probe(config, inner, occupancy, positions):
    """在 `positions` 放前驱参数的探针陷阱（E_D = E_t1，线型参数 E_rel1/ℏω₀，Z = 0、占据 0，不带电，不改变别的
    缺陷感受到的静电环境），返回 (扩展后的 KMCSimulation, 扩展后的占据, 探针编号)。"""
    tddb = config.tddb
    material = config.device.stack.single_material
    probes = DefectEnsemble.uniform(
        positions,
        np.full(len(positions), tddb.precursor_ionization_energy),
        replace(material, huang_rhys=tddb.precursor_relaxation_energy / material.phonon_energy),
        band_reference=float(config.device.layer_band_references()[0]),
        charge_state=0.0,
    )
    extended = KMCSimulation(config, defects=append_defects(inner.defects, probes))
    return extended, np.concatenate([occupancy, np.zeros(len(positions))]), np.arange(inner.n_defects, extended.n_defects)


@pytest.mark.parametrize("width", [0.0, 0.5e-9])
def test_electrode_term_and_bond_breaking_match_engine_probe(monkeypatch, width):
    """电极项（式 T4）：逐点求值（单个位点，插值表每个方向只有一个节点）时每个前驱位点的 R_1e,电极 等于引擎在该点放
    一个前驱参数陷阱时的两侧 R2 + R4；局域电势、局域场都含当前占据下的空间电荷。引擎的 R4 用 12 个声子项，这里把
    前驱的声子项数也设成 12 再逐项对拍。一组位点一起求值时走插值表（ADR-055），与逐点求值相比按生成率加权的相对
    误差 ≤ 10⁻³、ΣG 的相对误差 ≤ 10⁻³。三分之一的缺陷带与之重合的固定电荷 −e，另有一个离开缺陷 0.6 nm：生成率的
    泊松源项（固定电荷放在各自的位置）与引擎的静电背景一致（ADR-048）。P_bb 用配置的 E_A（全部位点相同，ADR-053），
    场取泊松解在 Gaussian 窗口上的平均（ADR-043），窗口宽度 0 时就是引擎在探针处读到的局域场。"""
    monkeypatch.setattr(tddb_module, "_PRECURSOR_N_PHONON", _N_PHONON)
    config = _config(concentration_cm3=3.0e19, field_averaging_width=width)
    tddb = config.tddb
    material = config.device.stack.single_material
    defects = KMCSimulation(config).defects
    fixed = np.where(np.arange(len(defects)) % 3 == 0, -Q, 0.0)
    offset = np.zeros((len(defects), 3))
    fixed[1] = -Q
    offset[1, 2] = 0.6e-9 if defects.positions[1, 2] < 0.5 * config.device.Lz else -0.6e-9
    inner = KMCSimulation(config, defects=replace(defects, fixed_charge=fixed, fixed_charge_offset=offset))
    occupancy = solve_mean_field(inner).occupancy
    # 铺满窗口的位点每 60 个取一个，约 290 个
    precursors = Precursors(place_precursors(config.device, inner.defects.positions, np.random.default_rng(5)).positions[::60])
    rates = generation_rates(inner, occupancy, tddb, precursors)
    single = np.array([
        generation_rates(inner, occupancy, tddb, Precursors(point[None, :])).electrode[0] for point in precursors.positions
    ])

    extended, occ, probe = _probe(config, inner, occupancy, precursors.positions)
    energy, field, barrier_c, barrier_a = extended._trap_state(occ)
    blocks = extended._electrode_blocks(energy, field, barrier_c, barrier_a)
    capture = (blocks[_B_R2C] + blocks[_B_R2A] + blocks[_B_R4C] + blocks[_B_R4A])[probe]
    assert len(precursors) > 200 and np.all(capture > 0.0)
    np.testing.assert_allclose(single, capture, rtol=1e-8)
    generation = single * rates.bond_breaking
    assert np.sum(np.abs(rates.electrode / single - 1.0) * generation) <= 1e-3 * generation.sum()
    assert rates.electrode @ rates.bond_breaking == pytest.approx(generation.sum(), rel=1e-3)

    bond_field = field[probe]
    if width > 0.0:
        solver = inner.solver
        phi = solver.solve(
            solver.deposit(inner.defects.positions, Q * (inner.defects.charge_state - occupancy))
            + solver.deposit(inner.defects.fixed_charge_positions, inner.defects.fixed_charge),
            tddb_module.top_boundary_potential(config.device, config.simulation.voltage),
        )
        mean = solver.gaussian_average(np.stack(solver.electric_field(phi), axis=-1), width)
        bond_field = np.linalg.norm(solver.interpolate(mean, precursors.positions), axis=-1)
        assert not np.allclose(bond_field, field[probe])
    p_bb = bond_break_probability(
        tddb.activation_energy, tddb.dipole_moment, material.kappa, bond_field, config.simulation.temperature
    )
    np.testing.assert_allclose(rates.bond_breaking, p_bb, rtol=1e-9)


def test_trap_term_is_the_direct_sum_over_occupied_traps():
    """陷阱项（式 T5 的陷阱来源，ADR-039）：每个前驱位点的 R_1e,陷阱 等于对全部已占据陷阱的直接求和 Σ_j p_j·R7(j → i)
    （xy 最小镜像、不截断），截断误差不超过 `_TRAP_TERM_FLOOR`。前驱能级取引擎在该点放一个前驱参数探针时的陷阱能级，
    源陷阱能级取平均场能级（ADR-038）。源陷阱全空时陷阱项为 0。"""
    config = _config(concentration_cm3=3.0e19)
    tddb, device = config.tddb, config.device
    inner = KMCSimulation(config)
    occupancy = solve_mean_field(inner).occupancy
    precursors = _sites_near_sources(inner, occupancy, np.random.default_rng(9))
    rates = generation_rates(inner, occupancy, tddb, precursors)

    extended, occ, probe = _probe(config, inner, occupancy, precursors.positions)
    precursor_energy = extended._trap_state(occ)[0][probe]
    trap_energy = inner._trap_state(occupancy)[0]
    defects = inner.defects
    delta = minimum_image_delta(precursors.positions[:, None, :] - defects.positions[None, :, :], device.Lx, device.Ly)
    expected = trap_precursor_capture_rate(
        defects.attempt_frequency[None, :],
        np.linalg.norm(delta, axis=-1),
        defects.r_D[None, :],
        localization_radius(tddb.precursor_ionization_energy, device.stack.single_material.m_di),
        trap_energy[None, :],
        precursor_energy[:, None],
        occupancy[None, :],
        config.simulation.temperature,
    ).sum(axis=1)
    assert np.all(expected[:40] > 1.0)
    np.testing.assert_allclose(rates.trap, expected, rtol=1e-9, atol=tddb_module._TRAP_TERM_FLOOR)
    assert np.all(generation_rates(inner, np.zeros(inner.n_defects), tddb, precursors).trap == 0.0)


def test_generation_rates_follow_local_temperature():
    """局域温度（§6.6）：温度场处处取环境温度时三个分量都等于缺省（按工作点温度求值）；温度场不均匀时每个位点按它
    所在处的温度求值：单个位点求值时等于整张温度场取这个温度时的值；一组位点一起求值时陷阱项与 P_bb 与单个位点
    相同，电极项的差别在插值表的精度内（ADR-055）。"""
    config = _config(concentration_cm3=3.0e19)
    tddb = config.tddb
    inner = KMCSimulation(config)
    occupancy = solve_mean_field(inner).occupancy
    precursors = _sites_near_sources(inner, occupancy, np.random.default_rng(4))
    ambient = config.simulation.temperature
    shape = inner.solver.grid.shape
    base = generation_rates(inner, occupancy, tddb, precursors)
    flat = generation_rates(inner, occupancy, tddb, precursors, np.full(shape, ambient))
    assert np.all(base.trap[:40] > 0.0)
    for name in GenerationRates._fields:
        np.testing.assert_allclose(getattr(flat, name), getattr(base, name), rtol=1e-12)

    field = ambient + np.random.default_rng(4).uniform(0.0, 80.0, shape)
    local = generation_rates(inner, occupancy, tddb, precursors, field)
    temperature = inner.solver.interpolate(field, precursors.positions)
    assert not np.allclose(local.rates, base.rates)
    for i in (0, 17, 39, 40, len(precursors) - 1):
        single = Precursors(precursors.positions[i:i + 1])
        alone = generation_rates(inner, occupancy, tddb, single, field)
        uniform = generation_rates(inner, occupancy, tddb, single, np.full(shape, temperature[i]))
        for name in GenerationRates._fields:
            assert getattr(alone, name)[0] == pytest.approx(getattr(uniform, name)[0], rel=1e-10), name
        # 一组位点一起求值：陷阱项与 P_bb 与单个位点相同，电极项走插值表（ADR-055）
        assert local.trap[i] == pytest.approx(alone.trap[0], rel=1e-10)
        assert local.bond_breaking[i] == pytest.approx(alone.bond_breaking[0], rel=1e-10)
        assert local.electrode[i] == pytest.approx(alone.electrode[0], rel=1e-2)


def test_activation_energy_shift_only_rescales_time(monkeypatch):
    """P_bb 未饱和时，E_A 整体平移 ΔE 只把每个位点的生成速率乘 exp(−ΔE/k_BT)（ADR-052）：同一种子下前驱位点、
    事件序列与电流逐位相同，时刻放大 exp(ΔE/k_BT) 倍。随机数的消耗与 E_A 无关。"""
    saturated = []
    original = tddb_module.generation_rates

    def recording(inner, occupancy, tddb, precursors, temperature_field=None):
        result = original(inner, occupancy, tddb, precursors, temperature_field)
        saturated.append(bool(np.any(result.bond_breaking >= 1.0)))
        return result

    monkeypatch.setattr(tddb_module, "generation_rates", recording)
    shift = 0.1 * EV
    config = _config(max_generated_vacancies=3, activation_energy=4.3 * EV, stop_time=1.0e30)
    base = run_tddb(config)
    shifted = run_tddb(replace(config, tddb=replace(config.tddb, activation_energy=4.3 * EV + shift)))
    assert len(saturated) == 8 and not any(saturated)
    factor = np.exp(shift / (constants.Boltzmann * config.simulation.temperature))
    assert base.stop_reason == shifted.stop_reason == "max_generated"
    assert np.array_equal(shifted.defects.positions, base.defects.positions)
    assert np.array_equal(shifted.current, base.current)
    assert np.array_equal(shifted.n_precursors, base.n_precursors)
    np.testing.assert_allclose(shifted.time, factor * base.time, rtol=1e-9)
    np.testing.assert_allclose(shifted.generation_rate, base.generation_rate / factor, rtol=1e-9)


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

    def recording_rates(inner, occupancy, tddb, precursors, temperature_field=None):
        temperature_fields.append(temperature_field)
        return original_rates(inner, occupancy, tddb, precursors, temperature_field)

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


def test_conducting_clusters_join_vacancies_and_electrodes_closer_than_r_c():
    """导电簇（ADR-054）：间距 < r_c 的空位连边（xy 最小镜像），离电极平面 < r_c 的空位接到该电极，整个簇随之接上；两侧
    都接即贯穿通路。导体间隙是负极侧导体（负极平面与接负极的簇）到正极侧导体的最短距离，有贯穿通路时为 0。"""
    device = make_device()  # 20×20×8 nm
    z = np.linspace(0.6e-9, 7.4e-9, 10)  # 相邻间距 0.756 nm，两端离电极 0.6 nm
    x = np.where(np.arange(10) % 2 == 0, 19.9e-9, 0.1e-9)  # 横向来回跨过周期边界，最小镜像下只差 0.2 nm
    chain = np.column_stack([x, np.full(10, 10e-9), z])
    sides = conducting_clusters(chain, device, 0.9e-9)
    assert np.all(sides == 2) and conductor_gap(chain, sides, device) == 0.0

    broken = np.delete(chain, 5, axis=0)  # 去掉中间一个：下半段接负极、上半段接正极，间隙是断口两端的距离
    sides = conducting_clusters(broken, device, 0.9e-9)
    assert sides.tolist() == [0] * 5 + [1] * 4
    gap = conductor_gap(broken, sides, device)
    assert gap == pytest.approx(np.linalg.norm(broken[4] - broken[5]), rel=1e-12) and gap >= 0.9e-9

    sides = conducting_clusters(chain, device, 0.7e-9)  # r_c 小于链上间距：只剩两端各自接电极
    assert sides.tolist() == [0] + [-1] * 8 + [1]
    across = np.linalg.norm(minimum_image_delta(chain[9] - chain[0], device.Lx, device.Ly))
    assert conductor_gap(chain, sides, device) == pytest.approx(across, rel=1e-12)
    assert conducting_clusters(chain[4:5], device, 0.9e-9).tolist() == [-1]
    assert conductor_gap(chain[4:5], np.array([-1]), device) == device.Lz
    assert conducting_clusters(np.zeros((0, 3)), device, 0.9e-9).size == 0


def test_conductors_are_equipotential_and_electrode_conductors_supply_electrons_at_the_fermi_level():
    """导电簇是等电位的导体（ADR-054/056）。生成率的电势里：接电极的导电簇成员处的电势等于所接电极的边界值（负极 0、
    正极 U）；悬空导电簇（≥ 2 个空位）成员处的电势彼此相同，导体电荷之和等于这些成员在引擎里的电荷之和；成员的能级取
    导体电势；其余缺陷处的电势扣掉自身电荷的贡献，等于引擎的电势加上导电簇带来的改变。陷阱项把接电极的成员当作所接
    电极费米能级处占满（p = 1）的电子源，其余陷阱（含悬空导电簇的成员）按平均场占据、能级取同一份电势；P_bb 的场取
    这份电势的 Gaussian 平均。没有导电簇时电势就是只放缺陷电荷的解。三分之一的缺陷带与之重合的固定电荷 −e，另有一个
    离开缺陷 0.6 nm。"""
    config = _config(concentration_cm3=3.0e19, percolation_distance=1.2e-9)
    tddb, device = config.tddb, config.device
    material = device.stack.single_material
    defects = KMCSimulation(config).defects
    fixed = np.where(np.arange(len(defects)) % 3 == 0, -Q, 0.0)
    offset = np.zeros((len(defects), 3))
    fixed[1] = -Q
    offset[1, 2] = 0.6e-9 if defects.positions[1, 2] < 0.5 * device.Lz else -0.6e-9
    inner = KMCSimulation(config, defects=replace(defects, fixed_charge=fixed, fixed_charge_offset=offset))
    occupancy = solve_mean_field(inner).occupancy
    positions = inner.defects.positions
    labels = distance_clusters(positions, device, tddb.percolation_distance)
    sides = conducting_clusters(positions, device, tddb.percolation_distance)
    floating = (np.bincount(labels)[labels] >= 2) & (sides < 0)
    members = np.flatnonzero(sides >= 0)
    conductor = np.flatnonzero((sides >= 0) | floating)
    free = np.flatnonzero((sides < 0) & ~floating)
    groups = np.unique(labels[floating])
    assert np.any(sides == 0) and np.any(sides == 1) and not np.any(sides == 2)
    assert groups.size >= 2 and np.any(fixed[members] != 0.0) and np.any(fixed[floating] != 0.0)

    solver = inner.solver
    top = tddb_module.top_boundary_potential(device, config.simulation.voltage)
    phi, shift = tddb_module._conductor_potential(inner, occupancy, sides, labels)
    at = solver.interpolate(phi, positions)
    np.testing.assert_allclose(at[members], np.where(sides[members] == 0, 0.0, top), rtol=0.0, atol=1e-10)
    for group in groups:
        assert np.ptp(at[labels == group]) < 1e-10
    electrons = occupancy - inner.defects.charge_state - inner.defects.colocated_fixed_charge / Q
    engine = inner._electrostatics(occupancy)[0]
    np.testing.assert_allclose(
        (at - electrons * inner.coupling.phi_self)[free], (engine + shift)[free], rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose((engine + shift)[conductor], at[conductor], rtol=0.0, atol=1e-12)
    assert np.abs(shift[free]).max() > 1e-3
    # 导体电荷：从电势里扣掉其余电荷的解，在成员上解耦合矩阵即得；悬空导电簇的总电子数守恒
    outside = np.setdiff1d(np.arange(len(positions)), conductor)
    rest = solver.solve(
        solver.deposit(positions[outside], Q * (inner.defects.charge_state - occupancy)[outside])
        + solver.deposit(
            inner.defects.fixed_charge_positions,
            np.where(np.isin(np.arange(len(positions)), conductor), fixed - inner.defects.colocated_fixed_charge, fixed),
        ),
        top,
    )
    green = inner.coupling.phi_matrix[np.ix_(conductor, conductor)]
    carried = np.linalg.solve(green, at[conductor] - solver.interpolate(rest, positions[conductor]))
    for group in groups:
        inside = labels[conductor] == group
        assert carried[inside].sum() == pytest.approx(electrons[conductor][inside].sum(), rel=1e-8, abs=1e-8)
        assert not np.allclose(carried[inside], electrons[conductor][inside], atol=1e-3)

    no_clusters = np.arange(len(positions))
    plain, zero = tddb_module._conductor_potential(inner, occupancy, np.full(len(sides), -1), no_clusters)
    reference = solver.solve(
        solver.deposit(positions, Q * (inner.defects.charge_state - occupancy))
        + solver.deposit(inner.defects.fixed_charge_positions, inner.defects.fixed_charge),
        top,
    )
    np.testing.assert_allclose(plain, reference, rtol=1e-12, atol=1e-14)
    assert np.all(zero == 0.0)

    precursors = _sites_near_sources(inner, occupancy, np.random.default_rng(9), sources=members)
    rates = generation_rates(inner, occupancy, tddb, precursors)
    points = precursors.positions
    precursor_energy = (
        device.band_reference_profile(points[:, 2]) - Q * solver.interpolate(phi, points) - tddb.precursor_ionization_energy
    )
    distance = np.linalg.norm(
        minimum_image_delta(points[:, None, :] - inner.defects.positions[None, :, :], device.Lx, device.Ly), axis=-1
    )

    def direct_sum(source_energy, source_occupancy):
        return trap_precursor_capture_rate(
            inner.defects.attempt_frequency[None, :], distance, inner.defects.r_D[None, :],
            localization_radius(tddb.precursor_ionization_energy, material.m_di),
            source_energy[None, :], precursor_energy[:, None], source_occupancy[None, :], config.simulation.temperature,
        ).sum(axis=1)

    trap_energy = inner._trap_state(occupancy)[0] - Q * shift
    fermi = np.where(sides == 0, 0.0, -Q * config.simulation.voltage)
    expected = direct_sum(np.where(sides >= 0, fermi, trap_energy), np.where(sides >= 0, 1.0, occupancy))
    np.testing.assert_allclose(rates.trap, expected, rtol=1e-9, atol=tddb_module._TRAP_TERM_FLOOR)
    # 成员若仍按平均场能级与占据计入，紧挨成员的位点陷阱项明显不同
    assert not np.allclose(direct_sum(trap_energy, occupancy)[:40], expected[:40], rtol=0.1)

    mean = solver.gaussian_average(np.stack(solver.electric_field(phi), axis=-1), tddb.field_averaging_width)
    p_bb = bond_break_probability(
        tddb.activation_energy, tddb.dipole_moment, material.kappa,
        np.linalg.norm(solver.interpolate(mean, points), axis=-1), config.simulation.temperature,
    )
    np.testing.assert_allclose(rates.bond_breaking, p_bb, rtol=1e-9)


def test_breakdown_at_start_when_the_initial_defects_already_span_the_film():
    """初始缺陷已经连成贯穿通路（r_c = 4 nm）：轨迹停在 t = 0，t_BD = 0；这条记录的电流 = 同一实现的稳态内层输运电流 +
    细丝的 G₀·V（G₀ = 2e²/h ≈ 77.5 µS），不求生成率（NaN），导体间隙为 0（ADR-054）。"""
    config = _config(concentration_cm3=3.0e19, percolation_distance=4.0e-9)
    result = run_tddb(config)
    steady = solve_mean_field(KMCSimulation(config))
    assert result.stop_reason == "breakdown" and result.breakdown_time == 0.0 and result.time.tolist() == [0.0]
    assert _QUANTUM_CONDUCTANCE == pytest.approx(7.748091729e-5, rel=1e-9)
    assert result.filament_current[0] == _QUANTUM_CONDUCTANCE * config.simulation.voltage
    assert result.current[0] == steady.current + result.filament_current[0]
    assert result.transport_current[0] == steady.current
    assert np.isnan(result.generation_rate[0]) and result.conductor_gap[0] == 0.0


def test_trajectory_stops_at_the_first_spanning_path(monkeypatch):
    """击穿判据（§6.3，ADR-054）：每条记录按当时的缺陷集合找导电簇；没有贯穿通路的记录细丝电流为 0，记录的导体间隙就是
    `conductor_gap`、不小于 r_c；轨迹停在第一条出现贯穿通路的记录上，t_BD 就是它的时刻，电流 = 内层输运电流 + G₀·V，
    间隙为 0、不求生成率。进度回调拿到的是总电流密度；I-t 拟合只用内层输运电流。r_c = 1.95 nm 时初始缺陷还没连通
    （阈值 2.017 nm），第 4 次生成接通。"""
    config = _config(seed=1, concentration_cm3=3.0e19, percolation_distance=1.95e-9, max_generated_vacancies=12)
    device, r_c = config.device, config.tddb.percolation_distance
    calls, reported = [], []
    original = tddb_module.solve_mean_field

    def recording(sim, occupancy=None):
        result = original(sim, occupancy)
        calls.append((sim, result))
        return result

    monkeypatch.setattr(tddb_module, "solve_mean_field", recording)
    result = run_tddb(config, progress=lambda n, t, density: reported.append(density))

    assert result.stop_reason == "breakdown" and result.breakdown_time == result.end_time == result.time[-1]
    assert result.n_generated[-1] == 4 and len(calls) == len(reported) == result.time.size
    for k, (sim, mf) in enumerate(calls):
        sides = conducting_clusters(sim.defects.positions, device, r_c)
        last = k == len(calls) - 1
        assert bool(np.any(sides == 2)) == last
        assert result.conductor_gap[k] == conductor_gap(sim.defects.positions, sides, device)
        assert result.filament_current[k] == (_QUANTUM_CONDUCTANCE * config.simulation.voltage if last else 0.0)
        assert result.transport_current[k] == mf.current
        assert result.current[k] == mf.current + result.filament_current[k]
        assert reported[k] == pytest.approx(result.current[k] / result.area, rel=1e-15)
        if not last:
            assert result.conductor_gap[k] >= r_c and result.generation_rate[k] > 0.0
    assert result.conductor_gap[-1] == 0.0 and np.isnan(result.generation_rate[-1])
    assert result.current[-1] > 1.0e3 * result.transport_current[-1]
    fit, times = fit_transient(result)
    index = np.searchsorted(result.time, times, side="right") - 1
    assert fit == fit_power_law(times, result.transport_current[index] / result.area)


def test_resuming_from_a_checkpoint_continues_the_same_trajectory(tmp_path):
    """续跑（ADR-057）：`checkpoint` 每步开始时拿到一次状态，第 0 个就是 `initial_state`；状态经 `save_state` / `load_state`
    逐位还原（含 t = 0 时没有占据的情形与随机数流）。从第 2 步的状态续跑（开热反馈），记录与不停跑的同一条轨迹从这一步起
    一致：新空位落在同样的位置，时刻、电流与输运温度只差舍入误差（静电耦合从头算，与逐个追加相对差约 10⁻¹⁰）；
    `n_initial` 与出生时刻接着原轨迹，交给 `start` 的状态不被改动。"""
    config = _config(seed=1, concentration_cm3=3.0e19, max_generated_vacancies=5, thermal_conductivity=2.0)
    states = []
    full = run_tddb(config, checkpoint=states.append)
    assert full.stop_reason == "max_generated" and len(states) == full.time.size == 6
    first = initial_state(config)
    assert np.array_equal(states[0].defects.positions, first.defects.positions)
    assert np.array_equal(states[0].precursors.positions, first.precursors.positions)
    assert states[0].occupancy is None and states[0].transport_temperature == config.simulation.temperature

    for k in (0, 2):
        path = str(tmp_path / "state_{}.npz".format(k))
        save_state(states[k], path)
        loaded = load_state(path)
        for item in fields(DefectEnsemble):
            assert np.array_equal(getattr(loaded.defects, item.name), getattr(states[k].defects, item.name)), item.name
        assert loaded.n_initial == states[k].n_initial and loaded.time == states[k].time
        assert np.array_equal(loaded.birth_time, states[k].birth_time)
        assert (loaded.occupancy is None) == (k == 0)
        if k:
            assert np.array_equal(loaded.occupancy, states[k].occupancy)
        assert np.array_equal(loaded.precursors.positions, states[k].precursors.positions)
        assert loaded.transport_temperature == states[k].transport_temperature
        assert np.array_equal(copy.deepcopy(loaded.rng).random(4), copy.deepcopy(states[k].rng).random(4))

    start = load_state(str(tmp_path / "state_2.npz"))
    before = start.rng.bit_generator.state
    assert start.time == full.time[2] and len(start.defects) == full.n_defects[2]
    assert start.transport_temperature != config.simulation.temperature
    resumed = run_tddb(config, start=start)
    assert start.rng.bit_generator.state == before and len(start.defects) == full.n_defects[2]
    assert resumed.stop_reason == full.stop_reason and resumed.n_initial == full.n_initial
    assert resumed.time.size == full.time.size - 2 and resumed.end_time == pytest.approx(full.end_time, rel=1e-9)
    assert np.array_equal(resumed.defects.positions, full.defects.positions)
    np.testing.assert_allclose(resumed.birth_time, full.birth_time, rtol=1e-9)
    np.testing.assert_allclose(resumed.time, full.time[2:], rtol=1e-9)
    np.testing.assert_allclose(resumed.current, full.current[2:], rtol=1e-8)
    np.testing.assert_allclose(resumed.generation_rate, full.generation_rate[2:], rtol=1e-8)
    np.testing.assert_allclose(resumed.transport_temperature, full.transport_temperature[2:], rtol=1e-12)


def test_draw_generation_picks_sites_by_rate():
    """Gillespie 抽事件（ADR-052）：Δt = −ln(r)/ΣG，按速率选位点，速率为 0 的位点不会被选中；总速率为 0 或没有
    位点时返回 None。"""
    rng = np.random.default_rng(0)
    rates = np.array([0.0, 1.0, 0.0, 3.0])
    draws = [_draw_generation(rates, rng) for _ in range(4000)]
    dt = np.array([d[0] for d in draws])
    picks = np.array([d[1] for d in draws])
    assert set(picks.tolist()) == {1, 3}
    assert np.mean(picks == 3) == pytest.approx(0.75, abs=4.0 * np.sqrt(0.75 * 0.25 / picks.size))
    assert np.all(dt > 0.0) and dt.mean() == pytest.approx(0.25, abs=4.0 * 0.25 / np.sqrt(dt.size))
    assert _draw_generation(np.zeros(3), rng) is None
    assert _draw_generation(np.zeros(0), rng) is None


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
    _, result, _, _ = trajectory
    fit, times = fit_transient(result, points_per_decade=10)
    assert times[0] == pytest.approx(result.time[1], rel=1e-12)
    assert times[-1] == pytest.approx(result.end_time, rel=1e-12)
    assert np.allclose(np.diff(np.log10(times)), np.log10(times[1] / times[0]))
    reference = fit_power_law(times, result.current_at(times) / result.area)
    assert fit == reference


def test_write_tddb_csv(tmp_path, trajectory):
    _, result, _, _ = trajectory
    path = tmp_path / "it.csv"
    write_tddb_csv(result, str(path))
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result.time.size
    assert [float(r["t_end_s"]) for r in rows] == pytest.approx(result.segment_end.tolist(), rel=1e-6)
    per_cm2 = constants.centi ** 2
    for k, (row, density) in enumerate(zip(rows, result.current_density)):
        assert float(row["J_total_A_cm2"]) == pytest.approx(density * per_cm2, rel=1e-5)
        parts = sum(float(row["J_{}_A_cm2".format(name)]) for name in MECHANISMS) + float(row["J_filament_A_cm2"])
        assert parts == pytest.approx(float(row["J_total_A_cm2"]), rel=1e-5)
        assert float(row["conductor_gap_nm"]) == pytest.approx(result.conductor_gap[k] / constants.nano, abs=1e-6)
        assert float(row["power_W"]) == pytest.approx(result.power[k], rel=1e-5)
        assert float(row["T_effective_K"]) == pytest.approx(result.effective_temperature[k], abs=1e-6)
        assert int(row["dense_cluster_size"]) == result.dense_cluster_size[k]
        assert int(row["mf_newton_steps"]) == result.newton_steps[k]
        assert int(row["n_precursors"]) == result.n_precursors[k]


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
        "electrodes": {"negative": {"work_function_eV": 4.6}, "positive": {"work_function_eV": 4.6}},
        "simulation": {"voltage_V": 3.0, "temperature_C": 25.0, "seed": 3, "grid": {"nx": 8, "ny": 8, "nz": 16}},
        "tddb": {
            "activation_energy_eV": 4.0, "precursor_ionization_energy_eV": 1.15,
            "precursor_relaxation_energy_eV": 1.0, "dipole_moment_eA": 7.1, "field_averaging_width_nm": 0.5,
            "max_generated_vacancies": 2, "stop_time_s": 1.0e4,
            "percolation_distance_nm": 0.4,
            "thermal_conductivity_W_mK": 1.0,
        },
    }
    config_path = tmp_path / "tddb.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    tddb_command(config=str(config_path), seed=None, out=str(tmp_path / "it.csv"), plot=str(tmp_path / "it.png"))
    assert (tmp_path / "it.png").stat().st_size > 0
    with open(tmp_path / "it.csv", newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    output = capsys.readouterr().out
    assert "n = " in output and "前驱位点" in output
