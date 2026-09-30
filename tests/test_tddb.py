"""TDDB 瞬态驱动（Phase 3，ADR-032–034，PROJECT_CONTEXT_TDDB.md §5、§6、§9）。

- 增量静电耦合（§6.4）：逐个追加缺陷后的耦合矩阵与对全体缺陷从头计算的一致；
- 新空位出生（§6.7）：出生占据 = 1、与初始缺陷同一物种，出生前后其余缺陷处的电势与电场不变；
  外层每步从上一步的占据热启动；
- G = 0 时 I(t) 恒定，且逐位等于同一初始实现下的稳态结果；
- 每步的强耦合簇与对当前缺陷集合新建 `KMCSimulation` 的分簇相同，终态电流与对终态缺陷
  从头求解的稳态一致；
- 单元生成率就是模型 C：N_T·V_cell·R_1e·P_bb，R_1e 与引擎在单元中心放一个前驱参数陷阱时的
  两侧 R2 + R4 之和一致（含空间电荷）；
- N_T 只改变时间标尺；放不下新空位的单元被跳过；
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
from kmc_leakage.mean_field import solve_mean_field
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
    run_tddb,
    write_tddb_csv,
)

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


def test_zero_generation_reproduces_steady_state_bitwise():
    """N_T = 0（G = 0）：没有生成事件，I(t) 在 [0, stop_time] 上恒定，且逐位等于同一初始实现下的
    稳态平均场结果。"""
    config = _config(precursor_concentration=0.0)
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


def test_generation_rate_is_model_c_with_engine_capture_rates(monkeypatch):
    """G_cell = N_T·V_cell·R_1e·P_bb：R_1e 与引擎在单元中心放一个前驱参数陷阱时的两侧 R2 + R4 之和
    一致，局域场与电势都含当前占据下的空间电荷（§6.1）。

    引擎的 R4 用 12 个声子项，这里把前驱的声子项数也设成 12 再逐项对拍。探针陷阱 Z = 0、
    占据 0，不带电，不改变别的缺陷感受到的静电环境。
    """
    monkeypatch.setattr(tddb_module, "_PRECURSOR_N_PHONON", _N_PHONON)
    config = _config()
    tddb = config.tddb
    inner = KMCSimulation(config)
    occupancy = solve_mean_field(inner).occupancy
    cells = generation_cells(config.device, tddb.generation_grid)
    rates = generation_rates(inner, occupancy, tddb, cells)

    material = config.device.stack.single_material
    probe_material = replace(material, huang_rhys=tddb.precursor_relaxation_energy / material.phonon_energy)
    live = np.flatnonzero(admissible_cells(cells, inner.defects.positions, config.device))
    chosen = live[[0, live.size // 3, 2 * live.size // 3, -1]]
    probes = DefectEnsemble.uniform(
        cells.centers[chosen],
        np.full(chosen.size, tddb.precursor_ionization_energy),
        probe_material,
        band_reference=float(config.device.layer_band_references()[0]),
        charge_state=0.0,
    )
    extended = KMCSimulation(config, defects=append_defects(inner.defects, probes))
    occ = np.concatenate([occupancy, np.zeros(chosen.size)])
    blocks, _ = extended._rates(occ)
    probe = np.arange(inner.n_defects, extended.n_defects)
    capture = (blocks[_B_R2C] + blocks[_B_R2A] + blocks[_B_R4C] + blocks[_B_R4A])[probe]
    field = extended._electrostatics(occ)[1][probe]
    p_bb = bond_break_probability(
        tddb.activation_energy, tddb.dipole_moment, material.kappa, field, config.simulation.temperature
    )
    expected = tddb.precursor_concentration * cells.volume * capture * p_bb
    assert np.all(expected > 0.0)
    np.testing.assert_allclose(rates[chosen], expected, rtol=1e-9)


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
    for row, density in zip(rows, result.current_density):
        assert float(row["J_total_A_cm2"]) == pytest.approx(density * per_cm2, rel=1e-5)
        parts = sum(float(row["J_{}_A_cm2".format(name)]) for name in MECHANISMS)
        assert parts == pytest.approx(float(row["J_total_A_cm2"]), rel=1e-5)


def test_cli_tddb_writes_csv_and_figure(tmp_path, capsys):
    """CLI `tddb`：读 YAML 跑一条轨迹，落 CSV 与 I-t 图。"""
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
        },
    }
    config_path = tmp_path / "tddb.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    tddb_command(config=str(config_path), seed=None, out=str(tmp_path / "it.csv"), plot=str(tmp_path / "it.png"))
    assert (tmp_path / "it.png").stat().st_size > 0
    with open(tmp_path / "it.csv", newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    assert "n = " in capsys.readouterr().out
