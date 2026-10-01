"""TDDB 瞬态驱动（Phase 3/4，ADR-032–036，PROJECT_CONTEXT_TDDB.md §5、§6、§9）。

- 增量静电耦合（§6.4）：逐个追加缺陷后的耦合矩阵与对全体缺陷从头计算的一致；
- 新空位出生（§6.7）：出生时被一个电子占据（电荷态 0）、与初始缺陷同一物种，出生前后其余
  缺陷处的电势与电场不变；外层每步从上一步的占据热启动；
- G = 0 时 I(t) 恒定，且逐位等于同一初始实现下的稳态结果（等温与开热反馈都成立）；
- 每步的强耦合簇与对当前缺陷集合新建 `KMCSimulation` 的分簇相同，终态电流与对终态缺陷
  从头求解的稳态一致；
- 单元生成率就是模型 C：N_T·V_cell·R_1e·P_bb，R_1e 与引擎在单元中心放一个前驱参数陷阱时的
  两侧 R2 + R4 之和一致（含空间电荷）；按单元局域温度求值时逐单元等于该温度下的生成率；
- N_T 只改变时间标尺；放不下新空位的单元被跳过；
- 热反馈（§6.6、§8.4）：每步的输运温度是上一步功率给出的缺陷处加权温度，生成率取温度场在
  单元中心的值；
- 击穿判据（§6.3）：渗流团簇同时触到两个电极即击穿，轨迹停在第一条连通的记录上；电流密度
  达到限流值是第二停止条件；
- I-t 幂律拟合对纯幂律精确还原 n，对带噪声的幂律给出合理的拟合优度。
"""

import csv
from dataclasses import fields, replace

import numpy as np
import pytest
from scipy import constants

from conftest import make_config, make_defect_config, make_device
from kmc_leakage import tddb as tddb_module
from kmc_leakage.config import GridConfig, TDDBConfig
from kmc_leakage.defects import (
    DefectEnsemble,
    _global_standoff,
    append_defects,
    layer_bulk_defects,
    localization_radius,
    minimum_image_delta,
)
from kmc_leakage.electrostatics import PoissonSolver
from kmc_leakage.kmc import _B_R2A, _B_R2C, _B_R4A, _B_R4C, _N_PHONON, MECHANISMS, ElectrostaticCoupling, KMCSimulation
from kmc_leakage.mean_field import mean_field_hop_rates, solve_mean_field
from kmc_leakage.rates import bond_break_probability
from kmc_leakage.tddb import (
    GenerationCells,
    IncrementalCoupling,
    _draw_generation,
    admissible_cells,
    fit_power_law,
    fit_transient,
    generation_cells,
    generation_rates,
    percolation_clusters,
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
        precursor_concentration=4.0e19 * PER_CM3,
        generation_grid=(2, 2, 14),
        max_generated_vacancies=4,
        stop_time=1.0e4,
        percolation_distance=0.9e-9,
        compliance_current_density=1.0e8,  # 1e4 A/cm²，远高于这些小体系的电流
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


def test_new_vacancy_is_born_occupied_same_species_and_neutral(trajectory):
    """新空位出生占据 = 1（其余缺陷从上一步不动点热启动），与初始缺陷同一物种；
    出生前后其余缺陷处的电势、电场与横向平均剖面只差舍入误差（新空位 q·(Z−n) = 0，§6.7）。"""
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

    for (before, _, solved), (after, start, _) in zip(calls[:-1], calls[1:]):
        n = before.n_defects
        phi0, field0, bar0 = before._electrostatics(solved.occupancy)
        phi1, field1, bar1 = after._electrostatics(start)
        np.testing.assert_allclose(phi1[:n], phi0, rtol=1e-12)
        np.testing.assert_allclose(field1[:n], field0, rtol=1e-11)
        np.testing.assert_allclose(bar1, bar0, rtol=1e-12, atol=1e-15)


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
    final_charge = Q * np.sum(result.defects.charge_state - result.occupancy)
    assert result.space_charge[-1] == pytest.approx(final_charge, rel=1e-12)
    assert result.empty_generated[-1] == pytest.approx(np.sum(1.0 - result.occupancy[n_initial:]), rel=1e-12)
    assert result.empty_generated[0] == 0.0

    # 等温：温度全是环境值；没有击穿，渗流团簇没有连通两个电极
    ambient = config.simulation.temperature
    assert not result.thermal
    for name in ("transport_temperature", "effective_temperature", "peak_temperature"):
        assert np.all(getattr(result, name) == ambient), name
    assert result.breakdown_time is None
    for k, size in enumerate(result.percolation_size):
        labels, spanning = percolation_clusters(positions[: n_initial + k], device, config.tddb.percolation_distance)
        assert size == np.bincount(labels).max() and not spanning.any()
    assert np.all(result.dense_cluster_size <= result.percolation_size)


def test_generation_rate_is_model_c_with_engine_capture_rates(monkeypatch):
    """G_cell = N_T·V_cell·R_1e·P_bb：R_1e 与引擎在单元中心放一个前驱参数陷阱时一致——两侧 R2 + R4 之和，
    加上近旁陷阱跳向探针的 R7 按源陷阱占据加权之和（R7 取内层平均场的口径，ADR-038/039）；局域场、
    电势与能级都含当前占据下的空间电荷（§6.1）。

    引擎的 R4 用 12 个声子项，这里把前驱的声子项数也设成 12 再逐项对拍。探针陷阱 Z = 0、
    占据 0，不带电，不改变别的缺陷感受到的静电环境。另在一个单元中心阴极一侧 0.6 nm 处放一个
    深能级（1.5 eV、几乎全占）的缺陷：它在那个单元上是上坡跃迁、被 MA 因子压低，同时把均匀取的
    探针里的一个变成陷阱项压过电极项的单元。
    """
    monkeypatch.setattr(tddb_module, "_PRECURSOR_N_PHONON", _N_PHONON)
    config = _config()
    tddb = config.tddb
    material = config.device.stack.single_material
    band_reference = float(config.device.layer_band_references()[0])
    cells = generation_cells(config.device, tddb.generation_grid)
    base = KMCSimulation(config)
    admissible = np.flatnonzero(admissible_cells(cells, base.defects.positions, config.device))
    target = admissible[cells.layer[admissible] == 0][1]  # 最靠阴极的一层：1.5 eV 的能级在阴极费米能级以下
    deep = DefectEnsemble.uniform(
        cells.centers[target] + np.array([0.0, 0.0, 0.6e-9]), np.array([1.5 * EV]), material, band_reference
    )
    inner = KMCSimulation(config, defects=append_defects(base.defects, deep))
    occupancy = solve_mean_field(inner).occupancy
    assert occupancy[-1] > 0.99
    rates = generation_rates(inner, occupancy, tddb, cells)

    probe_material = replace(material, huang_rhys=tddb.precursor_relaxation_energy / material.phonon_energy)
    live = np.flatnonzero(admissible_cells(cells, inner.defects.positions, config.device))
    chosen = np.append(live[[0, live.size // 3, 2 * live.size // 3, -1]], target)
    probes = DefectEnsemble.uniform(
        cells.centers[chosen],
        np.full(chosen.size, tddb.precursor_ionization_energy),
        probe_material,
        band_reference=band_reference,
        charge_state=0.0,
    )
    extended = KMCSimulation(config, defects=append_defects(inner.defects, probes))
    occ = np.concatenate([occupancy, np.zeros(chosen.size)])
    energy, field, barrier_c, barrier_a = extended._trap_state(occ)
    blocks = extended._electrode_blocks(energy, field, barrier_c, barrier_a)
    n = inner.n_defects
    probe = np.arange(n, extended.n_defects)
    electrode = (blocks[_B_R2C] + blocks[_B_R2A] + blocks[_B_R4C] + blocks[_B_R4A])[probe]
    hop = mean_field_hop_rates(extended, energy)
    a, b = extended._pairs.a, extended._pairs.b
    into = (a < n) & (b >= n)
    from_traps = np.bincount(b[into] - n, hop[into] * occ[a[into]], chosen.size)
    assert np.any(from_traps > 1e3 * electrode)  # 陷阱项压过电极项的单元
    assert 0.0 < from_traps[-1] < 1e-3 * electrode[-1]  # 上坡：被 MA 因子压低
    p_bb = bond_break_probability(
        tddb.activation_energy, tddb.dipole_moment, material.kappa, field[probe], config.simulation.temperature
    )
    expected = tddb.precursor_concentration * cells.volume * (electrode + from_traps) * p_bb
    assert np.all(expected > 0.0)
    np.testing.assert_allclose(rates[chosen], expected, rtol=1e-9)


def test_trap_channel_vanishes_for_empty_or_distant_traps(monkeypatch):
    """R_1e 的陷阱项按源陷阱占据加权、只列截断半径内的 (陷阱, 前驱) 对（判据同近邻表，ADR-039）。

    两个深能级陷阱（1.6 eV，阴极侧，几乎全占）放在一列生成单元的中心线上（单元横向 10 nm，中心在
    x, y = 5 nm）。对照取"一条陷阱来源都不列"的生成率表（截断半径置零，R_1e 只剩电极项）：陷阱全空时
    两者逐位相同；陷阱占据时，离两个陷阱都超过截断半径的单元逐位相同，截断半径内的单元变大，紧挨着
    陷阱的单元以陷阱项为主。
    """
    config = _config()
    material = config.device.stack.single_material
    defects = DefectEnsemble.uniform(
        np.array([[5.0e-9, 5.0e-9, 1.5e-9], [5.6e-9, 5.0e-9, 1.8e-9]]),
        np.array([1.6, 1.6]) * EV,
        material,
        band_reference=float(config.device.layer_band_references()[0]),
    )
    inner = KMCSimulation(config, defects=defects)
    occupancy = solve_mean_field(inner).occupancy
    assert np.all(occupancy > 0.9)
    cells = generation_cells(config.device, config.tddb.generation_grid)
    live = np.flatnonzero(admissible_cells(cells, defects.positions, config.device))
    r_cut = tddb_module._trap_capture_cutoff(
        defects,
        localization_radius(config.tddb.precursor_ionization_energy, material.m_di),
        live.size,
        config.simulation.temperature,
    )
    rates = generation_rates(inner, occupancy, config.tddb, cells)
    rates_empty = generation_rates(inner, np.zeros(2), config.tddb, cells)
    monkeypatch.setattr(tddb_module, "_trap_capture_cutoff", lambda *args: 0.0)
    electrode_only = generation_rates(inner, occupancy, config.tddb, cells)
    assert np.array_equal(rates_empty, generation_rates(inner, np.zeros(2), config.tddb, cells))

    distance = np.min(
        [np.linalg.norm(minimum_image_delta(cells.centers[live] - point, config.device.Lx, config.device.Ly), axis=1)
         for point in defects.positions],
        axis=0,
    )
    far = distance >= r_cut
    assert far.any() and (~far).any()
    assert np.array_equal(rates[live[far]], electrode_only[live[far]])
    assert np.all(rates[live[~far]] >= electrode_only[live[~far]])
    assert np.any(rates[live[~far]] > 2.0 * electrode_only[live[~far]])


def test_cells_within_min_separation_of_a_defect_have_zero_rate():
    """单元中心落在已有缺陷 min_separation 之内的单元生成率为 0，其余单元不受影响（§6.1）。"""
    config = _config()
    inner = KMCSimulation(config)
    occupancy = solve_mean_field(inner).occupancy
    cells = generation_cells(config.device, config.tddb.generation_grid)
    before = generation_rates(inner, occupancy, config.tddb, cells)
    assert np.all(before > 0.0)

    target = 20
    offset = np.array([0.3e-9, 0.0, 0.0])  # 离单元中心 0.3 nm < min_separation，离别的单元中心都远
    extra = layer_bulk_defects(config.device, 0, cells.centers[target] + offset, np.random.default_rng(0))
    extra = replace(extra, charge_state=np.zeros(1))  # 不带电，其余单元的静电环境不变
    grown = KMCSimulation(config, defects=append_defects(inner.defects, extra))
    after = generation_rates(grown, np.append(occupancy, 0.0), config.tddb, cells)
    assert after[target] == 0.0
    assert np.sum(after == 0.0) == 1
    others = np.arange(len(cells)) != target
    np.testing.assert_allclose(after[others], before[others], rtol=1e-9)


def test_precursor_concentration_only_rescales_time():
    """N_T 只是生成率的前置因子：同一种子下 N_T 翻 10 倍，事件序列与电流逐位相同、时刻缩为 1/10。"""
    base = run_tddb(_config(max_generated_vacancies=3))
    scaled = run_tddb(_config(max_generated_vacancies=3, precursor_concentration=4.0e20 * PER_CM3))
    assert np.array_equal(scaled.current, base.current)
    assert np.array_equal(scaled.defects.positions, base.defects.positions)
    np.testing.assert_allclose(scaled.time * 10.0, base.time, rtol=1e-12)
    np.testing.assert_allclose(scaled.generation_rate, 10.0 * base.generation_rate, rtol=1e-12)


def test_generation_rates_follow_local_cell_temperature():
    """逐单元温度（§6.6）：每个单元的生成率等于整张表都取该单元温度时的值；全表取工作点温度时
    逐位退回缺省（按工作点温度求值）。"""
    config = _config()
    inner = KMCSimulation(config)
    occupancy = solve_mean_field(inner).occupancy
    cells = generation_cells(config.device, config.tddb.generation_grid)
    ambient = config.simulation.temperature
    base = generation_rates(inner, occupancy, config.tddb, cells)
    assert np.array_equal(generation_rates(inner, occupancy, config.tddb, cells, np.full(len(cells), ambient)), base)
    temperature = ambient + np.random.default_rng(4).uniform(0.0, 80.0, len(cells))
    local = generation_rates(inner, occupancy, config.tddb, cells, temperature)
    assert not np.allclose(local, base)
    for cell in (0, 17, len(cells) // 2, len(cells) - 1):
        uniform = generation_rates(inner, occupancy, config.tddb, cells, np.full(len(cells), temperature[cell]))
        assert local[cell] == pytest.approx(uniform[cell], rel=1e-12)


def test_thermal_feedback_feeds_previous_power_into_transport_and_generation(monkeypatch):
    """开热反馈（§6.6、§8.4）：第 k 步的内层输运用第 k−1 步功率给出的缺陷处加权温度（第 0 步取
    环境温度）；每步记录的功率、加权温度与温度场最高值就是该步稳态的 ΔE·R 解热方程的结果；
    生成率按温度场在单元中心的值求。k_TH 取得很小，升温到开尔文量级，数值上看得出来。"""
    k_th = 1.0e-6
    config = _config(seed=1, concentration_cm3=3.0e19, thermal_conductivity=k_th, max_generated_vacancies=3)
    calls, cell_temperatures = [], []
    original_mf, original_rates = tddb_module.solve_mean_field, tddb_module.generation_rates

    def recording_mf(sim, occupancy=None):
        result = original_mf(sim, occupancy)
        calls.append((sim, result))
        return result

    def recording_rates(inner, occupancy, tddb, cells, temperature=None):
        cell_temperatures.append(temperature)
        return original_rates(inner, occupancy, tddb, cells, temperature)

    monkeypatch.setattr(tddb_module, "solve_mean_field", recording_mf)
    monkeypatch.setattr(tddb_module, "generation_rates", recording_rates)
    result = run_tddb(config)

    ambient = config.simulation.temperature
    assert result.thermal and result.stop_reason == "max_generated"
    assert result.transport_temperature[0] == ambient
    assert np.array_equal(result.transport_temperature[1:], result.effective_temperature[:-1])
    assert np.all(result.effective_temperature > ambient + 1.0)
    assert np.all(result.peak_temperature >= result.effective_temperature)
    cells = generation_cells(config.device, config.tddb.generation_grid)
    for k, ((sim, mf), cell_temperature) in enumerate(zip(calls, cell_temperatures)):
        assert sim.sim.temperature == result.transport_temperature[k]
        heat = heat_release(sim, mf.occupancy)
        field = solve_temperature(sim.solver, power_map(sim.solver, sim.defects.positions, heat.defect), k_th, ambient)
        assert result.power[k] == heat.dielectric
        assert result.effective_temperature[k] == transport_temperature(
            sim.solver, field, sim.defects.positions, heat.defect, ambient
        )
        assert result.peak_temperature[k] == field.max()
        assert np.array_equal(cell_temperature, sim.solver.interpolate(field, cells.centers))


def test_percolation_clusters_detect_a_chain_spanning_both_electrodes():
    """渗流团簇（§6.3）：间距 < d_c 连边、xy 取最小镜像；分量里有缺陷离阴极、阳极界面都 < d_c
    才算连通两个电极。"""
    device = make_device()  # 20×20×8 nm
    z = np.linspace(0.6e-9, 7.4e-9, 10)  # 相邻间距 0.756 nm
    x = np.where(np.arange(10) % 2 == 0, 19.9e-9, 0.1e-9)  # 横向来回跨过周期边界，最小镜像下只差 0.2 nm
    chain = np.column_stack([x, np.full(10, 10e-9), z])
    labels, spanning = percolation_clusters(chain, device, 0.9e-9)
    assert np.all(labels == labels[0]) and spanning.tolist() == [True]
    # 截断小于链上的间距：每个缺陷各成一个分量
    labels, spanning = percolation_clusters(chain, device, 0.7e-9)
    assert np.unique(labels).size == 10 and not spanning.any()
    # 去掉中间一个：断成两段，各自只触到一个电极
    labels, spanning = percolation_clusters(np.delete(chain, 5, axis=0), device, 0.9e-9)
    assert np.unique(labels).size == 2 and not spanning.any()
    # 整条链上移 0.35 nm：最下端离阴极 0.95 nm ≥ d_c，没触到阴极
    assert not percolation_clusters(chain + np.array([0.0, 0.0, 0.35e-9]), device, 0.9e-9)[1].any()
    labels, spanning = percolation_clusters(np.zeros((0, 3)), device, 0.9e-9)
    assert labels.size == 0 and spanning.size == 0


def test_trajectory_stops_at_first_spanning_record(monkeypatch, trajectory):
    """渗流判据触发即停：轨迹停在第一条连通两个电极的记录上，t_BD 就是这条记录的时刻，之前的
    记录与不停的轨迹逐位相同。这里由包装触发"连通"：缺陷数达到初始 + 2 起，判据的截断换成
    膜厚，每个分量都触到两个电极（判据本身由上一个测试单独验证）。"""
    config, reference, _ = trajectory
    original = tddb_module.percolation_clusters

    def spanning_from_second_generation(positions, device, distance):
        if distance == config.tddb.percolation_distance and positions.shape[0] >= reference.n_initial + 2:
            distance = device.Lz
        return original(positions, device, distance)

    monkeypatch.setattr(tddb_module, "percolation_clusters", spanning_from_second_generation)
    result = run_tddb(config)
    assert result.stop_reason == "breakdown"
    assert result.breakdown_time == result.end_time == reference.time[2]
    assert np.array_equal(result.time, reference.time[:3])
    assert np.array_equal(result.current, reference.current[:3])


def test_compliance_stops_when_current_density_reaches_the_limit(trajectory):
    """限流是第二停止条件：电流密度第一次达到限流值的那条记录停下；J(0) 就达到时停在 t = 0。"""
    config, reference, _ = trajectory
    density = reference.current_density
    limit = 0.5 * (density[1] + density[2])
    assert np.all(density[:2] < limit) and density[2] >= limit
    result = run_tddb(replace(config, tddb=replace(config.tddb, compliance_current_density=limit)))
    assert result.stop_reason == "compliance" and result.breakdown_time is None
    assert result.end_time == reference.time[2]
    assert np.array_equal(result.current, reference.current[:3])

    at_start = run_tddb(replace(config, tddb=replace(config.tddb, compliance_current_density=density[0])))
    assert at_start.stop_reason == "compliance"
    assert at_start.time.tolist() == [0.0] and at_start.end_time == 0.0


def test_full_cell_is_skipped():
    """选中的单元整个落在已有缺陷的 min_separation 之内：记为已满，改从其余单元抽。"""
    config = make_config()
    device = config.device
    existing = np.array([[10e-9, 10e-9, 4e-9]])
    cells = GenerationCells(
        lower=np.array([[9.95e-9, 9.95e-9, 3.95e-9], [2e-9, 2e-9, 2e-9]]),
        size=np.array([0.1e-9, 0.1e-9, 0.1e-9]),
        layer=np.array([0, 1]),
        z_centers=np.array([4e-9, 2.05e-9]),
    )
    full = np.zeros(2, dtype=bool)
    rng = np.random.default_rng(0)
    dt, position = _draw_generation(np.array([1e12, 1.0]), full, cells, existing, 0.5e-9, device, rng)
    assert full.tolist() == [True, False]
    assert np.all((position >= cells.lower[1]) & (position < cells.lower[1] + cells.size))
    assert dt > 0.0
    assert _draw_generation(np.zeros(2), full, cells, existing, 0.5e-9, device, rng) is None


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
        assert int(row["percolation_size"]) == result.percolation_size[k]
        assert int(row["dense_cluster_size"]) == result.dense_cluster_size[k]


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
            "precursor_relaxation_energy_eV": 1.0, "dipole_moment_eA": 7.1,
            "precursor_concentration_cm3": 4.0e19, "generation_grid": {"nx": 2, "ny": 2, "nz": 14},
            "max_generated_vacancies": 2, "stop_time_s": 1.0e4,
            "percolation_distance_nm": 0.9, "compliance_current_density_A_cm2": 1.0e4,
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
