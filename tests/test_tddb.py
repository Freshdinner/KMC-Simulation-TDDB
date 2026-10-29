"""TDDB 瞬态驱动（Phase 3/4，ADR-032–036、ADR-052，PROJECT_CONTEXT_TDDB.md §5、§6、§9）。

- 增量静电耦合（§6.4）：逐个追加缺陷后的耦合矩阵与对全体缺陷从头计算的一致；
- 新空位出生（§6.7、ADR-048）：出生时被一个电子占据、与初始缺陷同一物种，带着与它重合的 O_i⁻
  （q_fix = −e）；出生时介质总电荷正好多一个 −e，其余缺陷处的电势与电场的变化就是这个 −e 的贡献；
  外层每步从上一步的占据热启动；
- G = 0 时 I(t) 恒定，且逐位等于同一初始实现下的稳态结果（等温与开热反馈都成立）；
- 每步的强耦合簇与对当前缺陷集合新建 `KMCSimulation` 的分簇相同，终态电流与对终态缺陷
  从头求解的稳态一致；
- 离散前驱位点（§6.1，ADR-052/055/063）：t = 0 在可采样窗口里做最小间隔 min_separation 的最大随机堆积，位点两两之间、
  与初始缺陷之间都 ≥ min_separation，且再也放不下一个位点；每个位点的 E_A,i 从上限 E_A 往下按指数尾抽取；新空位落在某个位点上，只用掉这一个位点；
- 每个位点的生成速率 G_i = R_1e,i·P_bb,i：电极项逐点求值时等于引擎在该点放一个前驱参数陷阱时的两侧 R2 + R4（含空间
  电荷），插值表与逐点求值的差别在表的精度内（ADR-055）；陷阱项等于对全部已占据陷阱的直接求和（截断误差
  ≤ `_TRAP_TERM_FLOOR`），P_bb 取位点自己的 E_A,i（指数尾，ADR-063）与 Gaussian 平均场；三项都按位点处的局域温度求值；
- P_bb 未饱和时 E_A 整体平移只改时间标尺；Gillespie 按速率选位点；
- 局域温度（§6.6，ADR-060）：逐缺陷温度全取工作点温度时引擎逐位不变；不均匀时每个缺陷的电极速率取它自己的温度，
  陷阱间跃迁的 MA 因子取两端平均；
- 热反馈（§6.6、§8.4，ADR-060）：输运按缺陷处的局域温度求值，与热方程迭代到缺陷处温度的变化 ≤ 0.01 K；生成率取同一份
  温度场，下一步从这份温度场起；
- 导电簇与击穿判据（§6.3，ADR-054/056/060/064）：转移积分的两倍大于重组能的空位对连成导电簇，离电极小于自身离域距离的
  空位接到该电极；接电极的导电簇
  在输运里是导体：陷阱之间的等效耦合与背景让陷阱处电势等于直接解泊松方程的结果，导体与近旁陷阱按陷阱间通道的形式
  交换电子、细致平衡；导体与对面电极之间的间隙隧穿按 R1 的能量积分；生成率的电势里接电极的导体保持导体电势、并作为
  所在处费米能级占满的电子源进陷阱项，悬空导电簇自身等电位、总电荷守恒；第一条出现贯穿通路的记录给出 t_BD，
  电流加上细丝的 G₀·V，功率 G₀·V² 平分到通路的空位上，之后再跑 `post_breakdown_steps` 步；
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
from kmc_leakage import mean_field
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
from kmc_leakage.kmc import (
    _B_R2A,
    _B_R2C,
    _B_R3A,
    _B_R3C,
    _B_R4A,
    _B_R4C,
    _N_PHONON,
    MECHANISMS,
    ElectrostaticCoupling,
    KMCSimulation,
)
from kmc_leakage.mean_field import solve_mean_field
from kmc_leakage.rates import (
    bond_break_probability,
    miller_abrahams_factor,
    r1_electrode_electrode,
    r6_elastic_hop,
    trap_precursor_capture_rate,
)
from kmc_leakage.tddb import (
    _QUANTUM_CONDUCTANCE,
    CarriedJacobian,
    GenerationRates,
    IncrementalCoupling,
    Precursors,
    _draw_generation,
    _remap_jacobian,
    conducting_clusters,
    conductor_gap,
    conductor_layout,
    distance_clusters,
    filament,
    fit_power_law,
    fit_transient,
    gap_tunneling,
    generation_rates,
    initial_state,
    load_state,
    place_precursors,
    run_tddb,
    save_state,
    solve_transport,
    write_tddb_csv,
)
from kmc_leakage.thermal import power_map, solve_temperature

EV = constants.electron_volt
Q = constants.elementary_charge
E_ANGSTROM = Q * constants.angstrom


def _transfer_integral(r_c: float) -> float:
    """让测试材料里 E_D = 1.15 eV 的空位（r_D 相同、S·ℏω 相同）离域距离正好是 `r_c` 的转移积分前因子 t₀：
    两个这样的空位间距 < r_c 时连通、离电极 < r_c 时接上（ADR-064）。"""
    material = make_device().stack.single_material
    r_D = float(localization_radius(np.array([1.15 * EV]), material.m_di)[0])
    return material.huang_rhys * material.phonon_energy * float(np.exp(r_c / r_D))


def _tddb(r_c=0.4e-9, **overrides) -> TDDBConfig:
    params = dict(
        activation_energy=4.0 * EV,
        activation_energy_tail=0.0,
        precursor_ionization_energy=1.15 * EV,
        precursor_relaxation_energy=1.0 * EV,
        dipole_moment=7.1 * E_ANGSTROM,
        field_averaging_width=0.5e-9,
        max_generated_vacancies=4,
        stop_time=1.0e4,
        # 离域距离 r_c 小于 min_separation（0.5 nm）：空位之间、空位与电极都连不上，没有导电簇，也不会击穿
        transfer_integral=_transfer_integral(r_c),
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


def _transport(config, defects=None, temperature=None):
    """`defects`（缺省为 `KMCSimulation(config)` 的初始缺陷）上的内层输运稳态（`solve_transport`），缺陷处温度缺省取工作点
    温度。没有接电极的导体时 `sim` 就是全体缺陷上的引擎。"""
    defects = KMCSimulation(config).defects if defects is None else defects
    solver = PoissonSolver(config.device, config.simulation.grid)
    coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
    layout = conductor_layout(defects, coupling, solver, config.simulation.voltage, config.tddb.transfer_integral)
    if temperature is None:
        temperature = np.full(len(defects), config.simulation.temperature)
    return solve_transport(config, layout, defects, coupling, solver, temperature)


@pytest.fixture(scope="module")
def trajectory():
    """4 次生成的短轨迹，并记下每步交给平均场的缺陷实现与初值、交给生成率的前驱位点表与算出的速率。

    初始缺陷取 3×10¹⁹ cm⁻³（96 个），一开始就有多个强耦合簇，逐步对比分簇才有内容。
    """
    config = _config(seed=1, concentration_cm3=3.0e19)
    calls, generations = [], []
    original_mf, original_rates = tddb_module.solve_mean_field, tddb_module.generation_rates

    def recording(sim, occupancy=None, jacobian=None, refresh=None):
        result = original_mf(sim, occupancy, jacobian, refresh)
        calls.append((sim, None if occupancy is None else occupancy.copy(), result))
        return result

    def recording_rates(transport, tddb, precursors, temperature_field=None):
        result = original_rates(transport, tddb, precursors, temperature_field)
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


def test_uniform_defect_temperature_reproduces_the_scalar_engine_bitwise():
    """逐缺陷温度（ADR-060）全部取工作点温度时，引擎的近邻表、强耦合簇、全部速率与平均场不动点与不传温度时逐位相同。"""
    config = make_config(voltage=3.0, seed=1, device=make_device(defects=make_defect_config(concentration_cm3=3.0e19)))
    plain = KMCSimulation(config)
    uniform = KMCSimulation(
        config, defects=plain.defects, coupling=plain.coupling, temperature=np.full(plain.n_defects, config.simulation.temperature)
    )
    assert np.array_equal(plain._pairs.a, uniform._pairs.a) and np.array_equal(plain._pairs.b, uniform._pairs.b)
    assert [m.tolist() for m in plain.cluster_members] == [m.tolist() for m in uniform.cluster_members]
    reference = solve_mean_field(plain)
    occupied = reference.occupancy > 0.5
    for expected, got in zip(plain._rates(occupied), uniform._rates(occupied)):
        assert np.array_equal(expected, got)
    result = solve_mean_field(uniform)
    assert result.current == reference.current and np.array_equal(result.occupancy, reference.occupancy)


def test_local_temperature_enters_each_defect_and_each_pair():
    """逐缺陷温度不均匀时（ADR-060）：缺陷 i 的 9 块电极速率等于整个引擎取温度 T_i 时第 i 行的值（声子线型、费米分布、R5 的
    玻尔兹曼因子、R8 都取 T_i）；陷阱间跃迁的 MA 因子取两端温度的平均，引擎（KMC 组态）与平均场两种跃迁速率都是。"""
    from kmc_leakage.mean_field import mean_field_hop_rates

    config = make_config(voltage=3.0, seed=1, device=make_device(defects=make_defect_config(concentration_cm3=3.0e19)))
    plain = KMCSimulation(config)
    temperature = config.simulation.temperature + np.random.default_rng(3).uniform(0.0, 400.0, plain.n_defects)
    hot = KMCSimulation(config, defects=plain.defects, coupling=plain.coupling, temperature=temperature)
    occupancy = solve_mean_field(plain).occupancy
    energy, field, barrier_c, barrier_a = hot._trap_state(occupancy)
    blocks = hot._electrode_blocks(energy, field, barrier_c, barrier_a)
    for i in (0, 7, 31, plain.n_defects - 1):
        single = KMCSimulation(
            replace(config, simulation=replace(config.simulation, temperature=float(temperature[i]))),
            defects=plain.defects, coupling=plain.coupling,
        )
        expected = single._electrode_blocks(energy, field, barrier_c, barrier_a)[:, i]
        np.testing.assert_allclose(blocks[:, i], expected, rtol=1e-12)
    a, b = hot._pairs.a, hot._pairs.b
    pair_temperature = 0.5 * (temperature[a] + temperature[b])
    np.testing.assert_allclose(
        hot._hop_rates(energy),
        hot._hop_geometric * miller_abrahams_factor(energy[a], energy[b] - hot._hop_U_ab, pair_temperature),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        mean_field_hop_rates(hot, energy),
        hot._hop_geometric * miller_abrahams_factor(energy[a], energy[b], pair_temperature),
        rtol=1e-14,
    )
    assert not np.allclose(blocks, plain._electrode_blocks(energy, field, barrier_c, barrier_a))


def _conductor_config():
    """r_c = 1.2 nm、3×10¹⁹ cm⁻³：两侧都有接电极的导电簇。三分之一的缺陷带与之重合的固定电荷 −e，一个接电极导体的成员带
    离开它 0.6 nm 的固定电荷 −e（导体背景里单独补上的那部分）。"""
    config = _config(concentration_cm3=3.0e19, r_c=1.2e-9)
    defects = KMCSimulation(config).defects
    _, sides = conducting_clusters(defects, config.device, config.tddb.transfer_integral)
    member = int(np.flatnonzero(sides == 0)[0])
    fixed = np.where(np.arange(len(defects)) % 3 == 0, -Q, 0.0)
    offset = np.zeros((len(defects), 3))
    fixed[member] = -Q
    offset[member, 2] = 0.6e-9
    return config, replace(defects, fixed_charge=fixed, fixed_charge_offset=offset)


def test_conductor_coupling_reproduces_the_poisson_potential_at_traps():
    """接电极的导体在输运里是导体（ADR-060）：输运不动点上，成员处的电势等于导体电势（负极 0、正极 U）；陷阱处的引擎电势、
    电场矢量与横向平均剖面，等于把陷阱电荷、固定电荷、导体电荷与外加偏压一起直接解泊松方程的结果扣掉陷阱自身电荷的贡献
    （等效耦合的对角元，含导体对它的像）。导体电荷由耦合矩阵在成员上解出（`ConductorLayout`），成员上离开成员位置的固定电荷
    照常计入。"""
    config, defects = _conductor_config()
    transport = _transport(config, defects)
    layout, sim, solver = transport.layout, transport.sim, transport.sim.solver
    sides, members, traps = layout.sides, layout.members, layout.traps
    assert np.any(sides == 0) and np.any(sides == 1) and sim.n_defects == traps.size
    assert np.any(defects.fixed_charge[members] != defects.colocated_fixed_charge[members])
    top = tddb_module.top_boundary_potential(config.device, config.simulation.voltage)
    phi = transport.phi
    np.testing.assert_allclose(solver.interpolate(phi, defects.positions[members]), np.where(sides[members] == 0, 0.0, top),
                               rtol=0.0, atol=1e-12)
    p = transport.result.occupancy
    own = p - defects.charge_state[traps] - defects.colocated_fixed_charge[traps] / Q
    engine, _, engine_bar = sim._electrostatics(p)
    np.testing.assert_allclose(
        solver.interpolate(phi, defects.positions[traps]) - own * sim.coupling.phi_self, engine, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(solver.lateral_average(phi), engine_bar, rtol=0.0, atol=1e-12)
    field = solver.field_vectors_at(phi, defects.positions[traps], defects.field_side[traps]) - own * sim.coupling.field_self
    np.testing.assert_allclose(field, _field_vectors(sim, p), rtol=1e-9, atol=1e-3)
    # 占据等价：q·(Z − 占据) + q_fix,重合 在成员上就是导体电荷
    rest = solver.solve(
        solver.deposit(defects.positions[traps], Q * (defects.charge_state[traps] - p))
        + solver.deposit(defects.fixed_charge_positions,
                         np.where(np.isin(np.arange(len(defects)), members), defects.fixed_charge - defects.colocated_fixed_charge,
                                  defects.fixed_charge)),
        top,
    )
    green = transport.coupling.phi_matrix[np.ix_(members, members)]
    carried = np.linalg.solve(green, solver.interpolate(phi, defects.positions[members]) - solver.interpolate(rest, defects.positions[members]))
    equivalent = transport.occupancy[members] - defects.charge_state[members] - defects.colocated_fixed_charge[members] / Q
    np.testing.assert_allclose(equivalent, carried, rtol=1e-9, atol=1e-9)


def test_conductor_exchange_enters_the_electrode_blocks_with_detailed_balance():
    """陷阱与接电极导体成员交换电子（`kmc.Conductors`，ADR-060）：引擎的 R2/R3 块比去掉交换通道（成员表置空、背景不变）
    多出的部分，等于逐对直接求和的 ν·exp[−r·(1/r_D,陷阱 + 1/r_D,成员)]·Θ（成员 → 陷阱取 Θ(μ, E)、陷阱 → 成员取 Θ(E, μ)），
    接负极的记进 R2C/R3C、接正极的记进 R2A/R3A，其余块不变。每个陷阱与负极侧导体的俘获与逃逸之比是 exp[−(E − 0)/k_BT]，
    只与这个电子库交换时平衡在费米分布。"""
    config, defects = _conductor_config()
    transport = _transport(config, defects)
    sim = transport.sim
    conductors = sim.conductors
    bare = KMCSimulation(
        config, defects=sim.defects, coupling=sim.coupling, solver=sim.solver, temperature=sim._temperature,
        conductors=replace(conductors, positions=np.zeros((0, 3)), fermi_level=np.zeros(0), positive_share=np.zeros(0),
                           r_D=np.zeros(0), temperature=np.zeros(0)),
    )
    p = transport.result.occupancy
    state = sim._trap_state(p)
    blocks, reference = sim._electrode_blocks(*state), bare._electrode_blocks(*state)
    energy = state[0]
    distance = np.linalg.norm(minimum_image_delta(
        conductors.positions[None, :, :] - sim.defects.positions[:, None, :], config.device.Lx, config.device.Ly
    ), axis=-1)
    geometric = r6_elastic_hop(sim.defects.attempt_frequency[:, None], distance, sim.defects.r_D[:, None], conductors.r_D[None, :])
    temperature = config.simulation.temperature
    capture = geometric * miller_abrahams_factor(conductors.fermi_level[None, :], energy[:, None], temperature)
    escape = geometric * miller_abrahams_factor(energy[:, None], conductors.fermi_level[None, :], temperature)
    share = conductors.positive_share[None, :]
    floor = 1e-12 * np.abs(blocks[[_B_R2C, _B_R2A, _B_R3C, _B_R3A]]).max()
    for block, expected in (
        (_B_R2C, (capture * (1.0 - share)).sum(axis=1)), (_B_R2A, (capture * share).sum(axis=1)),
        (_B_R3C, (escape * (1.0 - share)).sum(axis=1)), (_B_R3A, (escape * share).sum(axis=1)),
    ):
        np.testing.assert_allclose(blocks[block] - reference[block], expected, rtol=1e-9, atol=floor)
    untouched = [b for b in range(9) if b not in (_B_R2C, _B_R2A, _B_R3C, _B_R3A)]
    assert np.array_equal(blocks[untouched], reference[untouched])
    exchanged = (blocks[_B_R2C] - reference[_B_R2C]) > 1e3
    assert np.any(exchanged)
    ratio = (blocks[_B_R2C] - reference[_B_R2C])[exchanged] / (blocks[_B_R3C] - reference[_B_R3C])[exchanged]
    np.testing.assert_allclose(ratio, np.exp(-energy[exchanged] / (constants.Boltzmann * temperature)), rtol=1e-9)


def _chain(device, x, y, z_from, z_to, spacing=0.28e-9):
    """(x, y) 处从 z_from 到 z_to 的一列 O 空位，间距不超过 `spacing`。"""
    n = int(np.ceil(abs(z_to - z_from) / spacing)) + 1
    return np.column_stack([np.full(n, x), np.full(n, y), np.linspace(z_from, z_to, n)])


def test_gap_tunneling_follows_the_R1_integral_on_the_gap_path():
    """间隙隧穿（ADR-060）：一列接负极的空位（顶端离正极 1 nm）与一列接正极的空位（底端离负极 1.5 nm、横向离得远）。
    每个接负极成员一条竖直路径到正极平面，每个接正极成员一条从负极平面上来的竖直路径；每条路径的速率等于 R1 的能量积分
    （`r1_electrode_electrode`，从负极导带底积起，ADR-061）在这条路径上的值，势垒取输运不动点的三维电势，面积取
    `gap_tunneling_area`。总电流是各路径之和乘 e，各成员的放热是自己那条路径的 eU 倍；顶端成员的路径最短、尖端前方电势
    降得快，比身后的成员大得多。间隙从 1 nm 拉大到 2 nm 时电流变小。"""
    config = _config(concentration_cm3=3.0e18, r_c=0.3e-9)
    config = replace(config, simulation=replace(config.simulation, voltage=3.6, grid=replace(config.simulation.grid, nx=80, ny=80, nz=80)))
    device, area, voltage = config.device, config.tddb.gap_tunneling_area, 3.6
    material = device.stack.single_material
    solver = PoissonSolver(device, config.simulation.grid)
    currents = []
    for gap in (1.0e-9, 2.0e-9):
        positions = np.vstack([
            _chain(device, 10.03e-9, 10.07e-9, 0.2e-9, device.Lz - gap),
            _chain(device, 2.03e-9, 2.07e-9, device.Lz - 0.2e-9, 1.5e-9),
        ])
        defects = DefectEnsemble.uniform(
            positions, np.full(len(positions), 1.15 * EV), material,
            band_reference=float(device.layer_band_references()[0]), charge_state=1.0,
        )
        temperature = np.full(len(defects), config.simulation.temperature)
        transport = _transport(config, defects, temperature)
        layout = transport.layout
        assert set(layout.sides.tolist()) == {0, 1} and layout.traps.size == 0
        current, heat = gap_tunneling(transport, temperature, area)
        points = positions[layout.members]
        sides = layout.sides[layout.members]
        s = np.linspace(0.0, 1.0, tddb_module._GAP_PATH_POINTS)
        expected = np.zeros(points.shape[0])
        for m, point in enumerate(points):
            start, end = (point, np.append(point[:2], device.Lz)) if sides[m] == 0 else (np.append(point[:2], 0.0), point)
            line = start + s[:, None] * (end - start)
            band = device.band_reference_profile(line[:, 2]) - Q * solver.interpolate(transport.phi, line)
            expected[m] = sum(r1_electrode_electrode(
                s * float(np.linalg.norm(end - start)), band, area, voltage, config.simulation.temperature,
                device.negative.effective_mass, device.stack.m_di_at(line[:, 2]), -device.negative.fermi_energy,
            ))
        assert current == pytest.approx(Q * expected.sum(), rel=1e-12)
        np.testing.assert_allclose(heat, Q * voltage * expected, rtol=1e-12)
        negative = np.flatnonzero(sides == 0)
        tip = negative[np.argmax(points[negative, 2])]
        assert expected[tip] > 10.0 * np.sort(expected[negative])[-2]
        currents.append(current)
    assert currents[0] > currents[1] > 0.0


def test_gap_tunneling_counts_a_mutual_pair_of_tips_once():
    """接负极与接正极的两列空位上下相对、间隙 1 nm：顶端两个成员互为最近，这条路径只算一次、放热两端各一半；接负极的其余
    成员连到对面最近的接正极成员（底端），接正极的其余成员从对面最近的接负极成员（顶端）连上来（ADR-060）。"""
    config = _config(concentration_cm3=3.0e18, r_c=0.3e-9)
    config = replace(config, simulation=replace(config.simulation, voltage=3.6, grid=replace(config.simulation.grid, nx=80, ny=80, nz=80)))
    device, area, voltage = config.device, config.tddb.gap_tunneling_area, 3.6
    material = device.stack.single_material
    solver = PoissonSolver(device, config.simulation.grid)
    positions = np.vstack([
        _chain(device, 10.03e-9, 10.07e-9, 0.2e-9, 3.5e-9),
        _chain(device, 10.04e-9, 10.06e-9, device.Lz - 0.2e-9, 4.5e-9),
    ])
    defects = DefectEnsemble.uniform(
        positions, np.full(len(positions), 1.15 * EV), material,
        band_reference=float(device.layer_band_references()[0]), charge_state=1.0,
    )
    temperature = np.full(len(defects), config.simulation.temperature)
    transport = _transport(config, defects, temperature)
    layout = transport.layout
    points = positions[layout.members]
    sides = layout.sides[layout.members]
    negative, positive = np.flatnonzero(sides == 0), np.flatnonzero(sides == 1)
    tip, bottom = negative[np.argmax(points[negative, 2])], positive[np.argmin(points[positive, 2])]
    s = np.linspace(0.0, 1.0, tddb_module._GAP_PATH_POINTS)

    def rate(start, end):
        line = start + s[:, None] * (end - start)
        band = device.band_reference_profile(line[:, 2]) - Q * solver.interpolate(transport.phi, line)
        return sum(r1_electrode_electrode(
            s * float(np.linalg.norm(end - start)), band, area, voltage, config.simulation.temperature,
            device.negative.effective_mass, device.stack.m_di_at(line[:, 2]), -device.negative.fermi_energy,
        ))

    heat = np.zeros(points.shape[0])
    total = 0.0
    for a in negative:
        r = rate(points[a], points[bottom])
        total += r
        heat[[a, bottom]] += 0.5 * Q * voltage * r
    for b in positive:
        if b != bottom:
            r = rate(points[tip], points[b])
            total += r
            heat[[tip, b]] += 0.5 * Q * voltage * r
    current, got = gap_tunneling(transport, temperature, area)
    assert current == pytest.approx(Q * total, rel=1e-12)
    np.testing.assert_allclose(got, heat, rtol=1e-12)


def test_thermal_iteration_that_does_not_converge_is_flagged(monkeypatch):
    """输运与热的迭代次数用满仍不收敛时记为热失控（ADR-060）：上限设为 1、k_TH 取得很小（升温到开尔文量级）时，
    记录的迭代次数是 1、不收敛。"""
    monkeypatch.setattr(tddb_module, "_THERMAL_MAX_ITERATIONS", 1)
    config = _config(seed=1, concentration_cm3=3.0e19, thermal_conductivity=1.0e-6, max_generated_vacancies=1)
    result = run_tddb(config)
    assert np.all(result.thermal_iterations == 1) and not np.any(result.thermal_converged)


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
    稳态平均场结果。开热反馈时升温不到 0.01 K，输运与热迭代一次即收敛，输运仍按环境温度求值，结论不变。"""
    config = _config(activation_energy=1.0e3 * EV, thermal_conductivity=thermal_conductivity)
    result = run_tddb(config)
    steady = solve_mean_field(KMCSimulation(config))
    assert result.stop_reason == "no_generation" and result.breakdown_time is None
    assert result.end_time == config.tddb.stop_time
    assert result.time.tolist() == [0.0] and result.generation_rate.tolist() == [0.0]
    assert result.thermal_iterations.tolist() == [1] and result.thermal_converged.tolist() == [True]
    assert np.all(result.temperature_at_defects == config.simulation.temperature)
    assert result.gap_current[0] == 0.0 and result.filament_current[0] == 0.0
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
    assert np.all(result.filament_current == 0.0) and np.all(result.gap_current == 0.0)
    assert np.array_equal(result.transport_current, result.current)
    assert np.all(result.conductor_gap == device.Lz)  # r_c < min_separation：没有导电簇
    assert np.all(result.generation_rate > 0.0)
    assert np.all(np.diff(result.n_precursors) == -1)  # 每次出生用掉一个前驱位点
    assert np.all(result.max_cluster_size <= result.largest_component)
    final_charge = Q * np.sum(result.defects.charge_state - result.occupancy) + np.sum(result.defects.fixed_charge)
    assert result.space_charge[-1] == pytest.approx(final_charge, rel=1e-12)
    assert result.empty_generated[-1] == pytest.approx(np.sum(1.0 - result.occupancy[n_initial:]), rel=1e-12)
    assert result.empty_generated[0] == 0.0

    # 等温：温度全是环境值，输运只求一次；没有导电簇，不会击穿
    ambient = config.simulation.temperature
    assert not result.thermal
    for name in ("defect_temperature", "peak_temperature"):
        assert np.all(getattr(result, name) == ambient), name
    assert np.all(result.thermal_iterations == 1) and np.all(result.thermal_converged)
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
    positions = place_precursors(device, defects, 4.0 * EV, 0.0, np.random.default_rng(7)).positions
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

    assert np.array_equal(place_precursors(device, defects, 4.0 * EV, 0.0, np.random.default_rng(7)).positions, positions)
    flat = make_device(defects=make_defect_config(concentration_cm3=3.0e19, min_separation=0.0))
    with pytest.raises(ValueError):
        place_precursors(flat, defects, 4.0 * EV, 0.0, np.random.default_rng(7))


def test_precursor_activation_energies_follow_the_exponential_tail():
    """E_A 指数尾（ADR-063）：每个位点 E_A,i = E_A − E₀·X_i，X_i 服从单位指数分布。全部 ≤ E_A，(E_A − E_A,i)/E₀ 的均值与
    标准差都接近 1、分位数与指数分布一致；同一个随机数流下位点坐标与 E₀ 无关，E₀ = 0 时全部取 E_A。"""
    config = _config(concentration_cm3=3.0e19)
    device = config.device
    defects = KMCSimulation(config).defects.positions
    upper, tail = 1.6 * EV, 0.086 * EV
    sites = place_precursors(device, defects, upper, tail, np.random.default_rng(7))
    x = (upper - sites.activation_energy) / tail
    assert len(sites) > 10000 and np.all(x >= 0.0)
    assert np.mean(x) == pytest.approx(1.0, abs=0.03) and np.std(x) == pytest.approx(1.0, abs=0.03)
    np.testing.assert_allclose(np.quantile(x, [0.5, 0.9, 0.99]), -np.log([0.5, 0.1, 0.01]), rtol=0.05)
    flat = place_precursors(device, defects, upper, 0.0, np.random.default_rng(7))
    assert np.array_equal(flat.positions, sites.positions) and np.all(flat.activation_energy == upper)


def test_converted_precursor_is_the_only_site_taken_out():
    """断键的位点用掉，其余位点原样保留、顺序不变：位点两两之间本来就 ≥ min_separation（ADR-055）。"""
    positions = 1e-9 * np.arange(15.0).reshape(5, 3)
    activation = EV * np.array([1.0, 1.1, 1.2, 1.3, 1.4])
    precursors = Precursors(positions, activation)
    left = precursors.converted(2)
    assert len(left) == 4 and len(precursors) == 5
    assert np.array_equal(left.positions, positions[[0, 1, 3, 4]])
    assert np.array_equal(left.activation_energy, activation[[0, 1, 3, 4]])


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


def _sites_near_sources(transport, rng, n_near=40, n_uniform=40, sources=None):
    """前驱位点：前 `n_near` 个放在 `sources`（缺省为占据 ≥ 0.5 的缺陷，编号取全体缺陷）旁 0.5–0.8 nm 处（陷阱项显著），
    其余 `n_uniform` 个在可采样窗口里均匀撒；都在窗口内且离全部缺陷 ≥ min_separation。"""
    device = transport.sim.device
    positions = transport.defects.positions
    standoff = _global_standoff(device)
    sources = np.flatnonzero(transport.occupancy >= 0.5) if sources is None else sources
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
    return Precursors(np.array(points), np.full(len(points), 4.0 * EV))


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
    泊松源项（固定电荷放在各自的位置）与引擎的静电背景一致（ADR-048）。P_bb 用位点自己的 E_A,i（指数尾，ADR-063），
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
    transport = _transport(config, replace(defects, fixed_charge=fixed, fixed_charge_offset=offset))
    inner, occupancy = transport.sim, transport.occupancy
    # 铺满窗口的位点每 60 个取一个，约 290 个
    full = place_precursors(config.device, inner.defects.positions, tddb.activation_energy, 0.2 * EV, np.random.default_rng(5))
    precursors = Precursors(full.positions[::60], full.activation_energy[::60])
    rates = generation_rates(transport, tddb, precursors)
    single = np.array([
        generation_rates(transport, tddb, Precursors(point[None, :], activation[None])).electrode[0]
        for point, activation in zip(precursors.positions, precursors.activation_energy)
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
        precursors.activation_energy, tddb.dipole_moment, material.kappa, bond_field, config.simulation.temperature
    )
    assert np.ptp(precursors.activation_energy) > 0.2 * EV
    np.testing.assert_allclose(rates.bond_breaking, p_bb, rtol=1e-9)


def test_trap_term_is_the_direct_sum_over_occupied_traps():
    """陷阱项（式 T5 的陷阱来源，ADR-039）：每个前驱位点的 R_1e,陷阱 等于对全部已占据陷阱的直接求和 Σ_j p_j·R7(j → i)
    （xy 最小镜像、不截断），截断误差不超过 `_TRAP_TERM_FLOOR`。前驱能级取引擎在该点放一个前驱参数探针时的陷阱能级，
    源陷阱能级取平均场能级（ADR-038）。源陷阱全空时陷阱项为 0。"""
    config = _config(concentration_cm3=3.0e19)
    tddb, device = config.tddb, config.device
    transport = _transport(config)
    inner, occupancy = transport.sim, transport.occupancy
    precursors = _sites_near_sources(transport, np.random.default_rng(9))
    rates = generation_rates(transport, tddb, precursors)

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
    empty = replace(transport, occupancy=np.zeros(inner.n_defects))
    assert np.all(generation_rates(empty, tddb, precursors).trap == 0.0)


def test_generation_rates_follow_local_temperature():
    """局域温度（§6.6）：温度场处处取环境温度时三个分量都等于缺省（按工作点温度求值）；温度场不均匀时每个位点按它
    所在处的温度求值：单个位点求值时等于整张温度场取这个温度时的值；一组位点一起求值时陷阱项与 P_bb 与单个位点
    相同，电极项的差别在插值表的精度内（ADR-055）。"""
    config = _config(concentration_cm3=3.0e19)
    tddb = config.tddb
    transport = _transport(config)
    inner, occupancy = transport.sim, transport.occupancy
    precursors = _sites_near_sources(transport, np.random.default_rng(4))
    ambient = config.simulation.temperature
    shape = inner.solver.grid.shape
    base = generation_rates(transport, tddb, precursors)
    flat = generation_rates(transport, tddb, precursors, np.full(shape, ambient))
    assert np.all(base.trap[:40] > 0.0)
    for name in GenerationRates._fields:
        np.testing.assert_allclose(getattr(flat, name), getattr(base, name), rtol=1e-12)

    field = ambient + np.random.default_rng(4).uniform(0.0, 80.0, shape)
    local = generation_rates(transport, tddb, precursors, field)
    temperature = inner.solver.interpolate(field, precursors.positions)
    assert not np.allclose(local.rates, base.rates)
    for i in (0, 17, 39, 40, len(precursors) - 1):
        single = Precursors(precursors.positions[i:i + 1], precursors.activation_energy[i:i + 1])
        alone = generation_rates(transport, tddb, single, field)
        uniform = generation_rates(transport, tddb, single, np.full(shape, temperature[i]))
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

    def recording(transport, tddb, precursors, temperature_field=None):
        result = original(transport, tddb, precursors, temperature_field)
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


def test_thermal_feedback_iterates_local_temperature_to_self_consistency(monkeypatch):
    """开热反馈（§6.6、§8.4，ADR-060）：内层输运按缺陷处的局域温度求值（引擎的逐缺陷温度就是传进去的温度）。每一步里
    "输运 → 放热 → 温度场 → 缺陷处温度"迭代：下一次输运的温度就是上一次温度场在缺陷处的值，最后一次与它给出的温度场
    在缺陷处相差 ≤ 0.01 K（收敛）。记录的功率、缺陷处最高温度与温度场最高值都取最后一次，生成率取同一份温度场；
    下一步从这份温度场在新缺陷集合处的值起（第 0 步取环境温度）。k_TH 取得很小，升温到开尔文量级，第 0 步迭代不止一次。"""
    k_th = 1.0e-6
    config = _config(seed=1, concentration_cm3=3.0e19, thermal_conductivity=k_th, max_generated_vacancies=3)
    calls, temperature_fields = [], []
    original_transport, original_rates = tddb_module.solve_transport, tddb_module.generation_rates

    def recording_transport(config, layout, defects, coupling, solver, temperature, occupancy=None, jacobian=None):
        result = original_transport(config, layout, defects, coupling, solver, temperature, occupancy, jacobian)
        calls.append((defects, temperature.copy(), result))
        return result

    def recording_rates(transport, tddb, precursors, temperature_field=None):
        temperature_fields.append(temperature_field)
        return original_rates(transport, tddb, precursors, temperature_field)

    monkeypatch.setattr(tddb_module, "solve_transport", recording_transport)
    monkeypatch.setattr(tddb_module, "generation_rates", recording_rates)
    result = run_tddb(config)

    ambient = config.simulation.temperature
    assert result.thermal and result.stop_reason == "max_generated"
    assert np.all(result.thermal_converged) and result.thermal_iterations[0] >= 2
    assert len(calls) == result.thermal_iterations.sum() and len(temperature_fields) == result.time.size
    assert np.all(calls[0][1] == ambient)
    assert np.all(result.defect_temperature > ambient + 1.0)
    solver = PoissonSolver(config.device, config.simulation.grid)

    def field_of(defects, transport):
        heat = np.zeros(len(defects))
        heat[transport.layout.traps] = transport.heat.defect
        return solve_temperature(solver, power_map(solver, defects.positions, heat), k_th, ambient)

    last = np.cumsum(result.thermal_iterations) - 1
    for j, (defects, temperature, transport) in enumerate(calls):
        assert np.array_equal(transport.sim._temperature, temperature)
        field = field_of(defects, transport)
        updated = solver.interpolate(field, defects.positions)
        if j in last:
            k = int(np.flatnonzero(last == j)[0])
            assert np.max(np.abs(updated - temperature)) <= 0.01
            assert result.power[k] == pytest.approx(transport.heat.dielectric, rel=1e-12)
            assert result.defect_temperature[k] == temperature.max()
            assert result.peak_temperature[k] == pytest.approx(field.max(), rel=1e-12)
            np.testing.assert_allclose(temperature_fields[k], field, rtol=1e-12)
        if j + 1 < len(calls):
            next_defects, next_temperature, _ = calls[j + 1]
            np.testing.assert_allclose(next_temperature, solver.interpolate(field, next_defects.positions), rtol=1e-12)


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


def _vacancies(device, positions, ED=1.15 * EV):
    """`positions` 处的一组 O 空位（测试材料，能级 `ED`）。"""
    ED = np.broadcast_to(np.asarray(ED, dtype=float), (len(positions),)).copy()
    return DefectEnsemble.uniform(
        positions, ED, device.stack.single_material, band_reference=float(device.layer_band_references()[0]),
        charge_state=1.0,
    )


def test_conducting_clusters_join_vacancies_and_electrodes_closer_than_r_c():
    """导电簇（ADR-054/064）：离域距离为 r_c 的同种空位间距 < r_c 时连边（xy 最小镜像），离电极平面 < r_c 的空位接到该电极，
    整个簇随之接上；两侧都接即贯穿通路。导体间隙是负极侧导体（负极平面与接负极的簇）到正极侧导体的最短距离，有贯穿通路时为 0。"""
    device = make_device()  # 20×20×8 nm
    z = np.linspace(0.6e-9, 7.4e-9, 10)  # 相邻间距 0.756 nm，两端离电极 0.6 nm
    x = np.where(np.arange(10) % 2 == 0, 19.9e-9, 0.1e-9)  # 横向来回跨过周期边界，最小镜像下只差 0.2 nm
    chain = np.column_stack([x, np.full(10, 10e-9), z])
    t9, t7 = _transfer_integral(0.9e-9), _transfer_integral(0.7e-9)
    labels, sides = conducting_clusters(_vacancies(device, chain), device, t9)
    assert np.all(sides == 2) and np.all(labels == labels[0]) and conductor_gap(chain, sides, device) == 0.0

    broken = np.delete(chain, 5, axis=0)  # 去掉中间一个：下半段接负极、上半段接正极，间隙是断口两端的距离
    _, sides = conducting_clusters(_vacancies(device, broken), device, t9)
    assert sides.tolist() == [0] * 5 + [1] * 4
    gap = conductor_gap(broken, sides, device)
    assert gap == pytest.approx(np.linalg.norm(broken[4] - broken[5]), rel=1e-12) and gap >= 0.9e-9

    labels, sides = conducting_clusters(_vacancies(device, chain), device, t7)  # 离域距离小于链上间距：只剩两端各自接电极
    assert sides.tolist() == [0] + [-1] * 8 + [1] and np.unique(labels).size == 10
    across = np.linalg.norm(minimum_image_delta(chain[9] - chain[0], device.Lx, device.Ly))
    assert conductor_gap(chain, sides, device) == pytest.approx(across, rel=1e-12)
    assert conducting_clusters(_vacancies(device, chain[4:5]), device, t9)[1].tolist() == [-1]
    assert conductor_gap(chain[4:5], np.array([-1]), device) == device.Lz
    assert conducting_clusters(_vacancies(device, np.zeros((0, 3))), device, t9)[1].size == 0


def test_conducting_clusters_follow_the_delocalization_criterion():
    """离域判据（ADR-064）：空位 a、b 的转移积分 t₀·exp[−r·(1/r_D,a + 1/r_D,b)/2] 的两倍大于 (S_a + S_b)·ℏω 时连通，能级不同
    （r_D 不同）的一对按两者的平均衰减常数取连通距离；接电极按自身离域距离 r_D·ln[t₀/(S·ℏω)]。2t₀ 不超过重组能时没有边，
    贴着电极的空位也不接电极。"""
    device = make_device()
    material = device.stack.single_material
    relaxation = material.huang_rhys * material.phonon_energy
    t0 = 0.5 * EV
    ED = np.array([1.15, 0.9]) * EV
    r_D = localization_radius(ED, material.m_di)
    pair = 2.0 * np.log(t0 / relaxation) / (1.0 / r_D[0] + 1.0 / r_D[1])
    assert r_D[0] * np.log(t0 / relaxation) < pair < r_D[1] * np.log(t0 / relaxation)
    for r, linked in ((0.999 * pair, True), (1.001 * pair, False)):
        positions = np.array([[10e-9, 10e-9, 3e-9], [10e-9, 10e-9, 3e-9 + r]])
        labels, sides = conducting_clusters(_vacancies(device, positions, ED), device, t0)
        assert (labels[0] == labels[1]) == linked and np.all(sides == -1)
    reach = r_D[0] * np.log(t0 / relaxation)
    near = np.array([[10e-9, 10e-9, 0.999 * reach], [5e-9, 5e-9, device.Lz - 1.001 * reach]])
    assert conducting_clusters(_vacancies(device, near), device, t0)[1].tolist() == [0, -1]
    weak = 0.99 * relaxation  # 2t₀ < 2S·ℏω：不离域
    touching = np.array([[10e-9, 10e-9, 0.01e-9], [10e-9, 10e-9, 0.02e-9]])
    labels, sides = conducting_clusters(_vacancies(device, touching), device, weak)
    assert labels[0] != labels[1] and np.all(sides == -1)


def test_conductors_are_equipotential_and_electrode_conductors_supply_electrons_at_the_fermi_level():
    """导电簇是等电位的导体（ADR-054/056/060）。生成率的电势里：接电极的导体成员处的电势等于导体电势（负极 0、正极 U）；
    悬空导电簇（≥ 2 个空位）成员处的电势彼此相同，导体电荷之和等于这些成员在输运里的电荷之和；成员的能级取导体电势；
    其余陷阱处的电势扣掉自身电荷的贡献（等效耦合的对角元，含接电极导体对自身电荷的像），等于输运的电势加上悬空导电簇带来的
    改变。陷阱项把接电极的成员当作所在处费米能级（负极 0、正极 −eU）占满（p = 1）的电子源，其余陷阱（含悬空导电簇的成员）
    按平均场占据、能级取同一份电势；P_bb 的场取这份电势的 Gaussian 平均。没有导电簇时电势就是只放缺陷电荷的解。三分之一的
    缺陷带与之重合的固定电荷 −e，另有一个离开缺陷 0.6 nm。"""
    config = _config(concentration_cm3=3.0e19, r_c=1.2e-9)
    tddb, device = config.tddb, config.device
    material = device.stack.single_material
    defects = KMCSimulation(config).defects
    fixed = np.where(np.arange(len(defects)) % 3 == 0, -Q, 0.0)
    offset = np.zeros((len(defects), 3))
    fixed[1] = -Q
    offset[1, 2] = 0.6e-9 if defects.positions[1, 2] < 0.5 * device.Lz else -0.6e-9
    defects = replace(defects, fixed_charge=fixed, fixed_charge_offset=offset)
    transport = _transport(config, defects)
    layout, occupancy, positions = transport.layout, transport.occupancy, defects.positions
    labels, sides, members = layout.labels, layout.sides, layout.members
    floating = (np.bincount(labels)[labels] >= 2) & (sides < 0)
    conductor = np.flatnonzero((sides >= 0) | floating)
    free = np.flatnonzero((sides < 0) & ~floating)
    groups = np.unique(labels[floating])
    assert np.any(sides == 0) and np.any(sides == 1) and not np.any(sides == 2)
    assert groups.size >= 2 and np.any(fixed[members] != 0.0) and np.any(fixed[floating] != 0.0)

    solver = transport.sim.solver
    top = tddb_module.top_boundary_potential(device, config.simulation.voltage)
    phi, shift = tddb_module._conductor_potential(transport)
    at = solver.interpolate(phi, positions)
    np.testing.assert_allclose(at[members], np.where(sides[members] == 0, 0.0, top), rtol=0.0, atol=1e-10)
    for group in groups:
        assert np.ptp(at[labels == group]) < 1e-10
    electrons = occupancy - defects.charge_state - defects.colocated_fixed_charge / Q
    self_term = transport.sim.coupling.phi_self[np.searchsorted(layout.traps, free)]
    np.testing.assert_allclose(
        (at[free] - electrons[free] * self_term), (transport.potential + shift)[free], rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose((transport.potential + shift)[conductor], at[conductor], rtol=0.0, atol=1e-12)
    assert np.abs(shift[free]).max() > 1e-4
    # 导体电荷：从电势里扣掉其余电荷的解，在成员上解耦合矩阵即得；悬空导电簇的总电子数守恒
    outside = np.setdiff1d(np.arange(len(positions)), conductor)
    rest = solver.solve(
        solver.deposit(positions[outside], Q * (defects.charge_state - occupancy)[outside])
        + solver.deposit(
            defects.fixed_charge_positions,
            np.where(np.isin(np.arange(len(positions)), conductor), fixed - defects.colocated_fixed_charge, fixed),
        ),
        top,
    )
    green = transport.coupling.phi_matrix[np.ix_(conductor, conductor)]
    carried = np.linalg.solve(green, at[conductor] - solver.interpolate(rest, positions[conductor]))
    for group in groups:
        inside = labels[conductor] == group
        assert carried[inside].sum() == pytest.approx(electrons[conductor][inside].sum(), rel=1e-8, abs=1e-8)
        assert not np.allclose(carried[inside], electrons[conductor][inside], atol=1e-3)

    plain_config = replace(config, tddb=replace(tddb, transfer_integral=_transfer_integral(0.1e-9)))
    plain = _transport(plain_config, defects)
    assert plain.layout.members.size == 0
    plain_phi, zero = tddb_module._conductor_potential(plain)
    reference = solver.solve(
        solver.deposit(positions, Q * (defects.charge_state - plain.occupancy))
        + solver.deposit(defects.fixed_charge_positions, defects.fixed_charge),
        top,
    )
    np.testing.assert_allclose(plain_phi, reference, rtol=1e-12, atol=1e-14)
    assert np.all(zero == 0.0)

    precursors = _sites_near_sources(transport, np.random.default_rng(9), sources=members)
    rates = generation_rates(transport, tddb, precursors)
    points = precursors.positions
    precursor_energy = (
        device.band_reference_profile(points[:, 2]) - Q * solver.interpolate(phi, points) - tddb.precursor_ionization_energy
    )
    distance = np.linalg.norm(
        minimum_image_delta(points[:, None, :] - positions[None, :, :], device.Lx, device.Ly), axis=-1
    )

    def direct_sum(source_energy, source_occupancy):
        return trap_precursor_capture_rate(
            defects.attempt_frequency[None, :], distance, defects.r_D[None, :],
            localization_radius(tddb.precursor_ionization_energy, material.m_di),
            source_energy[None, :], precursor_energy[:, None], source_occupancy[None, :], config.simulation.temperature,
        ).sum(axis=1)

    fermi = np.where(sides == 0, 0.0, -Q * config.simulation.voltage)
    np.testing.assert_allclose(transport.energy[members], fermi[members], rtol=0.0, atol=1e-30)
    source = np.where(sides >= 0, fermi, transport.energy - Q * shift)
    expected = direct_sum(source, np.where(sides >= 0, 1.0, occupancy))
    np.testing.assert_allclose(rates.trap, expected, rtol=1e-9, atol=tddb_module._TRAP_TERM_FLOOR)
    # 接电极成员不作电子源时，紧挨成员的位点陷阱项明显不同
    assert not np.allclose(direct_sum(source, np.where(sides >= 0, 0.0, occupancy))[:40], expected[:40], rtol=0.1)

    mean = solver.gaussian_average(np.stack(solver.electric_field(phi), axis=-1), tddb.field_averaging_width)
    p_bb = bond_break_probability(
        tddb.activation_energy, tddb.dipole_moment, material.kappa,
        np.linalg.norm(solver.interpolate(mean, points), axis=-1), config.simulation.temperature,
    )
    np.testing.assert_allclose(rates.bond_breaking, p_bb, rtol=1e-9)


def test_breakdown_at_start_when_the_initial_defects_already_span_the_film():
    """初始缺陷已经连成贯穿通路（r_c = 4 nm）：t_BD = 0，击穿后不再跑（post_breakdown_steps = 0）时轨迹停在 t = 0。这条记录
    的电流 = 同一实现的内层输运电流 + 间隙隧穿 + 细丝的 G₀·V（G₀ = 2e²/h ≈ 77.5 µS），细丝功率 G₀·V² 平分到贯穿通路的
    空位上、记进介质功率，不求生成率（NaN），导体间隙为 0（ADR-054/060）。"""
    config = _config(concentration_cm3=3.0e19, r_c=4.0e-9)
    result = run_tddb(config)
    steady = _transport(config)
    voltage = config.simulation.voltage
    gap, gap_heat = gap_tunneling(steady, np.full(len(steady.defects), config.simulation.temperature),
                                  config.tddb.gap_tunneling_area)
    current, heat = filament(steady.layout, voltage)
    spanning = steady.layout.sides[steady.layout.members] == 2
    assert np.any(spanning)
    assert current == _QUANTUM_CONDUCTANCE * voltage and _QUANTUM_CONDUCTANCE == pytest.approx(7.748091729e-5, rel=1e-9)
    np.testing.assert_allclose(heat[spanning], current * voltage / np.count_nonzero(spanning), rtol=1e-15)
    assert np.all(heat[~spanning] == 0.0)

    assert result.stop_reason == "breakdown" and result.breakdown_time == 0.0 and result.time.tolist() == [0.0]
    assert result.filament_current[0] == current and result.gap_current[0] == gap
    assert result.transport_current[0] == steady.result.current
    assert result.current[0] == steady.result.current + gap + current
    assert result.power[0] == pytest.approx(steady.heat.dielectric + gap_heat.sum() + current * voltage, rel=1e-12)
    assert np.isnan(result.generation_rate[0]) and result.conductor_gap[0] == 0.0


def test_breakdown_time_and_steps_after_breakdown(monkeypatch):
    """击穿判据（§6.3，ADR-054/060）：每条记录按当时的缺陷集合找导电簇；第一条出现贯穿通路的记录给出 t_BD，之后再跑
    `post_breakdown_steps` = 2 步生成，第 2 步之后停（stop_reason = "breakdown"，终点是最后一条记录）。贯穿通路形成之前的
    记录细丝电流为 0、导体间隙就是 `conductor_gap` 且不小于 r_c；之后每条记录的电流都含 G₀·V、间隙为 0。电流 = 内层输运 +
    间隙隧穿 + 细丝，进度回调拿到的是总电流密度；只有最后一条不求生成率。I-t 拟合的缺省窗口到 t_BD 为止、只用内层输运
    电流。r_c = 1.95 nm 时初始缺陷还没连通（阈值 2.017 nm），第 3 次生成接通。"""
    config = _config(
        seed=1, concentration_cm3=3.0e19, r_c=1.95e-9, max_generated_vacancies=12, post_breakdown_steps=2
    )
    device, r_c, t0 = config.device, 1.95e-9, config.tddb.transfer_integral
    calls, reported = [], []
    original = tddb_module.solve_transport

    def recording(config, layout, defects, coupling, solver, temperature, occupancy=None, jacobian=None):
        result = original(config, layout, defects, coupling, solver, temperature, occupancy, jacobian)
        calls.append((defects, result))
        return result

    monkeypatch.setattr(tddb_module, "solve_transport", recording)
    result = run_tddb(config, progress=lambda n, t, density: reported.append(density))

    assert result.stop_reason == "breakdown" and result.end_time == result.time[-1]
    assert len(calls) == len(reported) == result.time.size
    spanning = np.array([np.any(conducting_clusters(d, device, t0)[1] == 2) for d, _ in calls])
    first = int(np.argmax(spanning))
    assert result.n_generated[first] == 3 and result.breakdown_time == result.time[first]
    assert np.array_equal(spanning, np.arange(len(calls)) >= first) and len(calls) == first + 3
    for k, (defects, transport) in enumerate(calls):
        _, sides = conducting_clusters(defects, device, t0)
        assert result.conductor_gap[k] == conductor_gap(defects.positions, sides, device)
        assert result.filament_current[k] == (_QUANTUM_CONDUCTANCE * config.simulation.voltage if spanning[k] else 0.0)
        assert result.transport_current[k] == transport.result.current
        assert result.current[k] == transport.result.current + result.gap_current[k] + result.filament_current[k]
        assert reported[k] == pytest.approx(result.current[k] / result.area, rel=1e-15)
        if not spanning[k]:
            assert result.conductor_gap[k] >= r_c
    assert np.all(result.conductor_gap[first:] == 0.0)
    assert np.all(result.generation_rate[:-1] > 0.0) and np.isnan(result.generation_rate[-1])
    assert result.current[first] > 1.0e3 * result.transport_current[first]
    fit, times = fit_transient(result)
    assert times[-1] == pytest.approx(result.breakdown_time, rel=1e-12)
    index = np.searchsorted(result.time, times, side="right") - 1
    assert fit == fit_power_law(times, result.transport_current[index] / result.area)


def test_resuming_from_a_checkpoint_continues_the_same_trajectory(tmp_path):
    """续跑（ADR-057）：`checkpoint` 每步开始时拿到一次状态，第 0 个就是 `initial_state`；状态经 `save_state` / `load_state`
    逐位还原（含 t = 0 时没有占据的情形、击穿之后的状态与随机数流）。从第 2 步的状态续跑（开热反馈），记录与不停跑的同一条
    轨迹从这一步起一致：新空位落在同样的位置，时刻、电流与缺陷处温度只差舍入误差（静电耦合从头算，与逐个追加相对差约
    10⁻¹⁰）；`n_initial` 与出生时刻接着原轨迹，交给 `start` 的状态不被改动。"""
    config = _config(seed=1, concentration_cm3=3.0e19, max_generated_vacancies=5, thermal_conductivity=2.0)
    ambient = config.simulation.temperature
    states = []
    full = run_tddb(config, checkpoint=states.append)
    assert full.stop_reason == "max_generated" and len(states) == full.time.size == 6
    first = initial_state(config)
    assert np.array_equal(states[0].defects.positions, first.defects.positions)
    assert np.array_equal(states[0].precursors.positions, first.precursors.positions)
    assert states[0].occupancy is None and np.all(states[0].temperature == ambient)
    assert all(state.breakdown_time is None and state.steps_after_breakdown == 0 for state in states)

    broken = replace(states[2], breakdown_time=1.5e-3, steps_after_breakdown=3)
    for k, state in ((0, states[0]), (2, states[2]), (3, broken)):
        path = str(tmp_path / "state_{}.npz".format(k))
        save_state(state, path)
        loaded = load_state(path)
        for item in fields(DefectEnsemble):
            assert np.array_equal(getattr(loaded.defects, item.name), getattr(state.defects, item.name)), item.name
        assert loaded.n_initial == state.n_initial and loaded.time == state.time
        assert np.array_equal(loaded.birth_time, state.birth_time)
        assert (loaded.occupancy is None) == (k == 0)
        if k:
            assert np.array_equal(loaded.occupancy, state.occupancy)
        assert np.array_equal(loaded.precursors.positions, state.precursors.positions)
        assert np.array_equal(loaded.temperature, state.temperature)
        assert loaded.breakdown_time == state.breakdown_time
        assert loaded.steps_after_breakdown == state.steps_after_breakdown
        assert np.array_equal(copy.deepcopy(loaded.rng).random(4), copy.deepcopy(state.rng).random(4))

    start = load_state(str(tmp_path / "state_2.npz"))
    before = start.rng.bit_generator.state
    assert start.time == full.time[2] and len(start.defects) == full.n_defects[2]
    assert np.all(start.temperature != ambient)
    resumed = run_tddb(config, start=start)
    assert start.rng.bit_generator.state == before and len(start.defects) == full.n_defects[2]
    assert resumed.stop_reason == full.stop_reason and resumed.n_initial == full.n_initial
    assert resumed.time.size == full.time.size - 2 and resumed.end_time == pytest.approx(full.end_time, rel=1e-9)
    assert np.array_equal(resumed.defects.positions, full.defects.positions)
    np.testing.assert_allclose(resumed.birth_time, full.birth_time, rtol=1e-9)
    np.testing.assert_allclose(resumed.time, full.time[2:], rtol=1e-9)
    np.testing.assert_allclose(resumed.current, full.current[2:], rtol=1e-8)
    np.testing.assert_allclose(resumed.generation_rate, full.generation_rate[2:], rtol=1e-8)
    np.testing.assert_allclose(resumed.defect_temperature, full.defect_temperature[2:], rtol=1e-12)
    np.testing.assert_allclose(resumed.temperature_at_defects, full.temperature_at_defects, rtol=1e-12)


def test_remap_jacobian_keeps_entries_between_surviving_traps():
    """沿用的 Jacobian 排到新的陷阱上（ADR-066）：两次都是陷阱的缺陷之间照搬，离开的陷阱的行列去掉，新出现的陷阱的行列取
    −I、列在要重求的列表里。"""
    previous = np.array([0, 2, 3, 5])
    matrix = np.arange(16.0).reshape(4, 4)
    remapped, refresh = _remap_jacobian(CarriedJacobian(matrix, previous), np.array([0, 3, 5, 6]))
    expected = -np.eye(4)
    expected[:3, :3] = matrix[np.ix_([0, 2, 3], [0, 2, 3])]
    assert np.array_equal(remapped, expected) and refresh.tolist() == [3]


def test_newton_jacobian_is_carried_to_the_next_step(monkeypatch):
    """外层每次输运求解把平均场牛顿法的 Jacobian 交给下一次（ADR-066，开热反馈）：Anderson 只给 2 次、第一次求解就走
    牛顿法，之后每次都带着上一次的 Jacobian 直接走牛顿法，新生空位的列标为要重求；轨迹与每次都从头求解的相同（同样的空位
    位置，时刻与电流只差收敛判据以内的误差），求 F 的总次数更少。"""
    monkeypatch.setattr(mean_field, "_ANDERSON_MAX_ITER", 2)
    config = _config(seed=1, concentration_cm3=3.0e19, max_generated_vacancies=4, thermal_conductivity=2.0)
    calls = []
    original = tddb_module.solve_mean_field

    def recording(sim, occupancy=None, jacobian=None, refresh=None):
        calls.append((sim.n_defects, None if jacobian is None else jacobian.shape, refresh))
        return original(sim, occupancy, jacobian, refresh)

    monkeypatch.setattr(tddb_module, "solve_mean_field", recording)
    carried = run_tddb(config)
    monkeypatch.setattr(tddb_module, "carried_jacobian", lambda transport: None)
    fresh = run_tddb(config)

    n_calls = int(carried.thermal_iterations.sum())
    assert calls[0][1] is None and len(calls) == n_calls + int(fresh.thermal_iterations.sum())
    assert all(shape == (n, n) for n, shape, _ in calls[1:n_calls])
    step_starts = np.cumsum(carried.thermal_iterations)[:-1]  # 每个外层步的第一次求解：多了一个新生空位
    assert all(calls[k][2].size == 1 for k in step_starts)
    assert all(refresh is None for _, _, refresh in calls[n_calls:])
    assert np.all(carried.converged) and np.all(fresh.converged)
    assert np.array_equal(carried.defects.positions, fresh.defects.positions)
    np.testing.assert_allclose(carried.time, fresh.time, rtol=1e-8)
    np.testing.assert_allclose(carried.current, fresh.current, rtol=1e-8)
    assert carried.iterations.sum() < fresh.iterations.sum()


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
        parts = sum(float(row["J_{}_A_cm2".format(name)]) for name in MECHANISMS)
        parts += float(row["J_gap_A_cm2"]) + float(row["J_filament_A_cm2"])
        assert parts == pytest.approx(float(row["J_total_A_cm2"]), rel=1e-5)
        assert float(row["conductor_gap_nm"]) == pytest.approx(result.conductor_gap[k] / constants.nano, abs=1e-6)
        assert float(row["power_W"]) == pytest.approx(result.power[k], rel=1e-5)
        assert float(row["T_defect_max_K"]) == pytest.approx(result.defect_temperature[k], abs=1e-6)
        assert int(row["thermal_iterations"]) == result.thermal_iterations[k]
        assert int(row["thermal_converged"]) == int(result.thermal_converged[k])
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
            "activation_energy_tail_eV": 0.0, "transfer_integral_eV": _transfer_integral(0.4e-9) / EV,
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
