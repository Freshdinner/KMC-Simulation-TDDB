"""t₂ 结构对照配置与退化用例全套（Phase 6 = ADR-006/018/019）。

这一批盯住 Phase 6 的配置交付与退化判据：
- 三份 t₂ 示例配置（ADR-006/019）：固定总厚 8 nm、ZrO₂ 两侧对称、层厚对齐到
  hz 的整数倍，三份之间只有层结构不同；
- t₂=0 是单层退化点：无 Al₂O₃、无 ZA 界面、界面缺陷同时关闭，且与单层基准
  配置 `examples/single_run.yaml` **逐位相同**（ADR-006 的"兼作回归校验"）；
- 退化用例两级判据（ADR-018）：确定性量（φ、E_C(z)、固定组态下完整速率表、
  R1 两分量）rtol ≤ 1e-12，同种子完整轨迹 rtol ≤ 1e-9。三层同材料 + ΔE_C=0
  的速率表与轨迹在 `test_stack_rates.py` 已有，这里补齐它的场量与 t₂=0 一路。
"""

import dataclasses
import pathlib

import numpy as np
import pytest
from scipy import constants

from kmc_leakage.bands import conduction_band_edge, top_boundary_potential
from kmc_leakage.config import (
    DeviceConfig,
    DielectricStack,
    ElectrodeMaterial,
    GridConfig,
    Interface,
    Layer,
    SimulationConfig,
)
from kmc_leakage.defects import generate_defects
from kmc_leakage.electrostatics import PoissonSolver
from kmc_leakage.io import load_config
from kmc_leakage.kmc import MECHANISMS, KMCSimulation, _blocking_layer_indices

EV = constants.electron_volt
M_E = constants.electron_mass
ROOT = pathlib.Path(__file__).resolve().parents[1]
T2_NM = (0, 2, 4)

# 退化用例跑的小配置：缺陷来自配置（N=60），只把网格与事件数压下来
_DEGENERACY_SIM = SimulationConfig(
    voltage=3.0, temperature=298.15, n_events=200, burn_in_fraction=0.2, seed=12345,
    grid=GridConfig(16, 16, 80),  # hz = 8/80 = 0.1 nm，层厚仍然对齐
)


def _load(t2: int):
    return load_config(str(ROOT / "examples" / "stack_t2_{}nm.yaml".format(t2)))


@pytest.fixture(scope="module")
def configs():
    return {t2: _load(t2) for t2 in T2_NM}


def _with_small_sim(config):
    return dataclasses.replace(config, simulation=_DEGENERACY_SIM)


# --------------------------------------------------------------------------
# 三份 t₂ 配置的结构（ADR-006/019）
# --------------------------------------------------------------------------
def test_t2_series_fixes_total_thickness_and_keeps_zro2_symmetric(configs):
    """固定总厚 8 nm、ZrO₂ 两侧对称 t₁=t₃=(8−t₂)/2，层厚都是 hz 的整数倍。"""
    for t2, config in configs.items():
        layers = config.device.stack.layers
        thickness_nm = np.array([layer.thickness for layer in layers]) / constants.nano
        assert config.device.Lz == pytest.approx(8e-9)
        assert thickness_nm.sum() == pytest.approx(8.0)
        if t2 == 0:
            assert thickness_nm.size == 1  # 单层退化：没有 Al₂O₃ 也就没有第二、三层
        else:
            assert thickness_nm.size == 3
            assert thickness_nm[1] == pytest.approx(float(t2))
            assert thickness_nm[0] == pytest.approx(thickness_nm[2])  # 两侧对称
            assert thickness_nm[0] == pytest.approx((8.0 - t2) / 2.0)
        # 层厚是 hz 的整数倍（变 κ 泊松要求界面落在网格节点上，ADR-011）
        hz_nm = 8.0 / config.simulation.grid.nz
        assert np.allclose(thickness_nm / hz_nm, np.rint(thickness_nm / hz_nm))
        PoissonSolver(config.device, config.simulation.grid)  # 界面对齐校验不报错


def test_t2_series_differs_only_in_layer_structure(configs):
    """三份配置除层结构外完全一致：材料、体缺陷、器件、电极、扫描范围、事件数。"""
    reference = configs[0]
    for config in configs.values():
        assert config.device.stack.materials["ZrO2"] == reference.device.stack.materials["ZrO2"]
        assert config.device.stack.layers[0].defects == reference.device.stack.layers[0].defects
        assert config.device.Lx == reference.device.Lx
        assert config.device.Ly == reference.device.Ly
        assert config.device.barrier_height == reference.device.barrier_height
        assert config.device.cathode == reference.device.cathode
        assert config.device.anode == reference.device.anode
        assert config.simulation.grid == reference.simulation.grid
        assert config.simulation.n_events == reference.simulation.n_events
        assert np.array_equal(config.sweep.voltages, reference.sweep.voltages)
        assert np.array_equal(config.sweep.temperatures, reference.sweep.temperatures)
        assert len(config.seeds) == len(reference.seeds) == 5


def test_t2_zero_turns_off_al2o3_interfaces_and_interface_defects(configs):
    """t₂=0：无 Al₂O₃、无内部界面、无界面缺陷、无阻挡层——纯单层 ZrO₂（ADR-006）。"""
    device = configs[0].device
    stack = device.stack
    assert len(stack.layers) == 1 and len(stack.interfaces) == 0
    assert stack.n_materials == 1
    assert _blocking_layer_indices(device.layer_band_references()).size == 0
    defects = generate_defects(device, np.random.default_rng(12345))
    assert np.all(defects.interface_id < 0)  # 一个界面缺陷都没有
    assert np.all(defects.layer_id == 0)


@pytest.mark.parametrize("t2", [2, 4])
def test_positive_t2_has_blocking_layer_empty_al2o3_and_interface_defects(configs, t2):
    """t₂>0：ZrO₂/Al₂O₃/ZrO₂ + 两个对称界面，Al₂O₃ 体内无缺陷（ADR-005/008）。"""
    device = configs[t2].device
    stack = device.stack
    assert [layer.material for layer in stack.layers] == ["ZrO2", "Al2O3", "ZrO2"]
    assert stack.layers[1].defects is None  # 阻挡层体相为空
    assert stack.n_materials == 2
    # 亲和能对齐给出 B_ℓ = [1.74, 3.20, 1.74] eV，中间层就是阻挡层
    assert np.allclose(device.layer_band_references() / EV, [1.74, 3.20, 1.74])
    assert list(_blocking_layer_indices(device.layer_band_references())) == [1]
    # 两个界面同参数、同 Nₛ；v1 界面固定面电荷为 0（ADR-007）
    first, second = stack.interfaces
    assert first.defects == second.defects
    assert first.sigma == second.sigma == 0.0
    assert first.defects.surface_density == pytest.approx(1.0e12 * constants.centi ** -2)

    defects = generate_defects(device, np.random.default_rng(12345))
    # 每个界面 Nₛ·A = 1e12 cm^-2 × 50×50 nm² = 25 个，放在界面平面上
    for j, z_interface in enumerate(stack.interface_positions):
        on_interface = defects.interface_id == j
        assert int(on_interface.sum()) == 25
        assert np.allclose(defects.positions[on_interface, 2], z_interface)
    assert set(np.unique(defects.layer_id[defects.layer_id >= 0])) == {0, 2}  # 体缺陷只在两层 ZrO₂


# --------------------------------------------------------------------------
# 退化用例（ADR-018）：t₂=0 回到单层基准
# --------------------------------------------------------------------------
def test_t2_zero_config_matches_the_single_layer_baseline_config():
    """t₂=0 与 `examples/single_run.yaml` 是同一个器件（材料名之外逐项相同）。"""
    t2_zero = _load(0).device
    baseline = load_config(str(ROOT / "examples" / "single_run.yaml")).device
    material = t2_zero.stack.single_material
    reference = baseline.stack.single_material
    # 材料名不同（ZrO2 / reference-preset），物性必须逐项相同
    assert dataclasses.replace(material, name=reference.name) == reference
    assert t2_zero.stack.layers[0].defects == baseline.stack.layers[0].defects
    assert t2_zero.stack.layers[0].thickness == baseline.stack.layers[0].thickness
    assert (t2_zero.Lx, t2_zero.Ly) == (baseline.Lx, baseline.Ly)
    assert t2_zero.barrier_height == baseline.barrier_height
    assert t2_zero.cathode == baseline.cathode and t2_zero.anode == baseline.anode


def test_t2_zero_reproduces_the_single_layer_baseline_numerically():
    """t₂=0 与单层基准的确定性量 rtol ≤ 1e-12、同种子轨迹 rtol ≤ 1e-9（ADR-018）。"""
    a = KMCSimulation(_with_small_sim(_load(0)))
    b = KMCSimulation(_with_small_sim(load_config(str(ROOT / "examples" / "single_run.yaml"))))

    # 同一份缺陷实现（同器件 + 同种子）：位置与能级逐位相同
    assert np.array_equal(a.defects.positions, b.defects.positions)
    assert np.array_equal(a.defects.ED, b.defects.ED)

    # 确定性量：固定组态下的完整速率表与 R1 两分量
    for occ in (np.zeros(a.n_defects, dtype=bool), a.defects.ED > np.median(a.defects.ED)):
        blocks_a, hop_a = a._rates(occ)
        blocks_b, hop_b = b._rates(occ)
        assert np.allclose(blocks_a, blocks_b, rtol=1e-12, atol=0.0)
        assert np.allclose(hop_a, hop_b, rtol=1e-12, atol=0.0)
    assert a._r1_current == pytest.approx(b._r1_current, rel=1e-12)
    assert np.all(a._pf_collection == 1.0)  # 无阻挡层 → P_A ≡ 1，R8_eff = R8

    # 同种子完整轨迹：J 与机制分解 rtol ≤ 1e-9
    ra, rb = a.run(), b.run()
    assert ra.current == pytest.approx(rb.current, rel=1e-9)
    for name in MECHANISMS:
        assert ra.by_mechanism[name] == pytest.approx(rb.by_mechanism[name], rel=1e-9, abs=0.0)


def test_t2_zero_potential_and_band_edge_match_the_single_layer_baseline():
    """t₂=0 与单层基准的 φ(z) 与 E_C(z) 逐位相同（ADR-018 的确定性量一栏）。"""
    devices = [_load(0).device, load_config(str(ROOT / "examples" / "single_run.yaml")).device]
    grid = _DEGENERACY_SIM.grid
    phi_bar, band = [], []
    for device in devices:
        solver = PoissonSolver(device, grid)
        phi = solver.solve(np.zeros(grid.shape), top_boundary_potential(device, 3.0))
        phi_bar.append(solver.lateral_average(phi))
        band.append(conduction_band_edge(device, solver.z, phi_bar[-1]))
    assert np.allclose(phi_bar[0], phi_bar[1], rtol=1e-12, atol=1e-15)
    assert np.allclose(band[0], band[1], rtol=1e-12, atol=0.0)


# --------------------------------------------------------------------------
# 退化用例（ADR-018）：三层同材料 + ΔE_C=0 回到单层
# --------------------------------------------------------------------------
def test_uniform_three_layer_potential_and_band_edge_match_single_layer():
    """三层同材料 + ΔE_C=0 的 φ(z) 与 E_C(z) 逐位退回单层（rtol ≤ 1e-12）。

    同一用例的固定组态速率表与同种子轨迹见 `test_stack_rates.py`；这里补的是
    ADR-018 确定性量里的场量一栏，用 t₂=2 那份配置的材料与器件参数。
    """
    reference = _load(2).device
    zro2 = reference.stack.materials["ZrO2"]
    electrode = reference.cathode
    grid = _DEGENERACY_SIM.grid

    single = DeviceConfig(
        reference.Lx, reference.Ly, electrode, electrode,
        DielectricStack.single(zro2, 8e-9), barrier_height=reference.barrier_height,
    )
    layered = DeviceConfig(
        reference.Lx, reference.Ly, electrode, electrode,
        DielectricStack(
            materials={"ZrO2": zro2},
            layers=(Layer("L0", "ZrO2", 3e-9), Layer("L1", "ZrO2", 2e-9), Layer("L2", "ZrO2", 3e-9)),
            interfaces=(Interface(delta_Ec=0.0), Interface(delta_Ec=0.0)),
        ),
        barrier_height=reference.barrier_height,
    )

    rng = np.random.default_rng(11)
    rho = rng.normal(0.0, 1e3, grid.shape)
    rho[:, :, 0] = rho[:, :, -1] = 0.0
    fields = []
    for device in (single, layered):
        solver = PoissonSolver(device, grid)
        phi = solver.solve(rho, top_boundary_potential(device, 3.0))
        phi_bar = solver.lateral_average(phi)
        fields.append((phi, phi_bar, conduction_band_edge(device, solver.z, phi_bar)))
    for a, b in zip(*fields):
        assert np.allclose(a, b, rtol=1e-12, atol=1e-15)
