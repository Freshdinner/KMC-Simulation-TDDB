"""缺陷生成、空间/能级分布与几何工具的检验。

分层生成（ADR-005/008）：体缺陷按层、界面缺陷按面密度放在界面平面上、独立物种
取势阱侧材料；全局最小间距同时管缺陷两两、到电极、体缺陷到内部界面；空器件返回
空实现（review F07）。单层退化与旧的整器件生成逐位相同。
"""

import numpy as np
import pytest
from scipy import constants
from scipy.integrate import trapezoid

from conftest import make_defect_config, make_device, make_material
from kmc_leakage.config import (
    DefectLevelConfig,
    DefectSpatialConfig,
    DeviceConfig,
    DielectricStack,
    ElectrodeMaterial,
    Interface,
    InterfaceDefectConfig,
    Layer,
)
from kmc_leakage import defects as defects_module
from kmc_leakage.defects import (
    defect_count,
    empty_ensemble,
    expected_depth_density,
    generate_defects,
    hop_cutoff_radius,
    interface_profile,
    localization_radius,
    minimum_image_delta,
    neighbor_pairs,
    pair_distances,
)

EV = constants.electron_volt
M_E = constants.electron_mass


def _zaz_device(t2_nm=1.0, ns_cm2=1.0e12, concentration_cm3=3.0e18, min_sep=0.5e-9,
                Lx=50e-9, Ly=50e-9, both_interfaces=True):
    """对称 ZrO₂/Al₂O₃/ZrO₂ 器件：两 ZrO₂ 层有体缺陷、Al₂O₃ 空、界面有界面缺陷。"""
    zro2 = make_material("ZrO2")
    al2o3 = make_material(
        "Al2O3", kappa=9.0, eps_opt=3.0, band_gap=6.4 * EV, electron_affinity=1.4 * EV,
        m_cb=0.3 * M_E, m_vb=0.4 * M_E, m_di=0.3 * M_E, phonon_energy=0.09 * EV,
    )
    t_side = (9e-9 - t2_nm * 1e-9) / 2.0
    bulk = make_defect_config(concentration_cm3=concentration_cm3, min_separation=min_sep)
    idef = InterfaceDefectConfig(
        surface_density=ns_cm2 * constants.centi ** -2,
        level=DefectLevelConfig(mode="fixed", ED=1.15 * EV),
        charge_state=1.0,
    )
    stack = DielectricStack(
        materials={"ZrO2": zro2, "Al2O3": al2o3},
        layers=(
            Layer("L0", "ZrO2", t_side, bulk),
            Layer("L1", "Al2O3", t2_nm * 1e-9, None),
            Layer("L2", "ZrO2", t_side, bulk),
        ),
        interfaces=(
            Interface(defects=idef),
            Interface(defects=idef if both_interfaces else None),
        ),
    )
    electrode = ElectrodeMaterial("TiN", 4.6 * EV, M_E, 5.0 * EV)
    return DeviceConfig(Lx, Ly, electrode, electrode, stack, barrier_height=1.74 * EV)


# --------------------------------------------------------------------------
# 单层生成（与旧整器件口径一致）
# --------------------------------------------------------------------------
def test_defect_count_follows_concentration_times_volume():
    device = make_device(defects=make_defect_config(concentration_cm3=3.0e18))  # 20×20×8 nm^3
    ensemble = generate_defects(device, np.random.default_rng(0))
    assert len(ensemble) == defect_count(device, make_defect_config(concentration_cm3=3.0e18))


def test_localization_radius_formula():
    """r_D = ℏ/√(2·m_di·E_D)，用隧穿质量；m_di 可逐缺陷。"""
    ED = np.array([1.15, 4.6]) * EV
    m_di = 1.2 * M_E
    r_D = localization_radius(ED, m_di)
    assert r_D[0] == pytest.approx(constants.hbar / np.sqrt(2.0 * m_di * ED[0]))
    assert r_D[1] == pytest.approx(0.5 * r_D[0])  # 能级深 4 倍 -> r_D 减半
    assert r_D[0] == pytest.approx(0.166e-9, rel=0.02)
    # 逐缺陷 m_di：轻质量 -> 大 r_D
    per_defect = localization_radius(np.array([1.15, 1.15]) * EV, np.array([1.2, 0.3]) * M_E)
    assert per_defect[1] == pytest.approx(per_defect[0] * np.sqrt(1.2 / 0.3))
    with pytest.raises(ValueError):
        localization_radius(np.array([0.0]), m_di)


def test_min_separation_applies_to_pairs_and_to_interfaces():
    device = make_device(defects=make_defect_config(concentration_cm3=3.0e18, min_separation=0.6e-9))
    ensemble = generate_defects(device, np.random.default_rng(3))
    distances = pair_distances(ensemble.positions, device.Lx, device.Ly)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= 0.6e-9
    z = ensemble.positions[:, 2]
    assert z.min() >= 0.6e-9
    assert z.max() <= device.Lz - 0.6e-9


def test_positions_stay_inside_the_box():
    device = make_device()
    ensemble = generate_defects(device, np.random.default_rng(1))
    assert np.all(ensemble.positions[:, 0] >= 0.0) and np.all(ensemble.positions[:, 0] <= device.Lx)
    assert np.all(ensemble.positions[:, 1] >= 0.0) and np.all(ensemble.positions[:, 1] <= device.Ly)


def test_fixed_level_mode_gives_identical_levels():
    ensemble = generate_defects(make_device(), np.random.default_rng(2))
    assert np.allclose(ensemble.ED, 1.15 * EV)
    assert np.allclose(ensemble.r_D, ensemble.r_D[0])


def test_gaussian_level_mode_statistics_and_positivity():
    device = make_device(
        defects=make_defect_config(
            concentration_cm3=6.0e19,
            level=DefectLevelConfig(mode="gaussian", ED=1.15 * EV, sigma=0.15 * EV),
        )
    )
    ensemble = generate_defects(device, np.random.default_rng(5))
    assert np.all(ensemble.ED > 0.0)  # 截断保证 r_D 有限
    assert ensemble.ED.mean() / EV == pytest.approx(1.15, abs=0.05)
    assert ensemble.ED.std() / EV == pytest.approx(0.15, abs=0.04)
    assert ensemble.r_D.std() > 0.0  # 能级分散 -> r_D 分散


def test_generation_is_reproducible():
    device = make_device()
    a = generate_defects(device, np.random.default_rng(9))
    b = generate_defects(device, np.random.default_rng(9))
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.ED, b.ED)


def test_impossible_min_separation_raises():
    device = make_device(defects=make_defect_config(concentration_cm3=3.0e18, min_separation=5e-9))
    with pytest.raises((RuntimeError, ValueError)):
        generate_defects(device, np.random.default_rng(0))


# --------------------------------------------------------------------------
# 分层 + 界面缺陷生成（ADR-005/008）
# --------------------------------------------------------------------------
def test_layered_generation_places_bulk_only_in_defect_layers():
    """体缺陷只落在配置了 defects 的层里；Al₂O₃（空层）体内没有缺陷。"""
    device = _zaz_device(t2_nm=1.0)
    ensemble = generate_defects(device, np.random.default_rng(7))
    bulk = ensemble.layer_id >= 0
    assert set(np.unique(ensemble.layer_id[bulk])) == {0, 2}  # 只有两 ZrO₂ 层
    # 体缺陷 z 落在各自 ZrO₂ 层内（避让内部界面 min_sep）
    z_bulk = ensemble.positions[bulk, 2]
    edges = device.stack.layer_edges
    in_l0 = (z_bulk >= edges[0]) & (z_bulk <= edges[1])
    in_l2 = (z_bulk >= edges[2]) & (z_bulk <= edges[3])
    assert np.all(in_l0 | in_l2)
    assert not np.any((z_bulk > edges[1]) & (z_bulk < edges[2]))  # Al₂O₃ 里没有体缺陷


def test_interface_defects_sit_on_planes_with_host_species_params():
    """界面缺陷在界面平面上、独立物种取势阱（ZrO₂）侧材料、能级基准取该侧 B。"""
    device = _zaz_device(t2_nm=1.0)
    ensemble = generate_defects(device, np.random.default_rng(7))
    iface = ensemble.interface_id >= 0
    assert set(np.unique(ensemble.interface_id[iface])) == {0, 1}
    assert np.all(ensemble.layer_id[iface] == -1)

    z_iface = device.stack.interface_positions  # [4nm, 5nm]
    for j, z_plane in enumerate(z_iface):
        on_j = ensemble.interface_id == j
        assert np.allclose(ensemble.positions[on_j, 2], z_plane)
    # 物性取 ZrO₂ 侧（m_di=1.2 m_e），不是 Al₂O₃（0.3 m_e）
    assert np.allclose(ensemble.m_di[iface], 1.2 * M_E)
    assert np.allclose(ensemble.eps_opt[iface], 5.6)
    # 两个界面的 ZrO₂ 侧导带基准都是 1.74 eV
    assert np.allclose(ensemble.band_reference[iface], 1.74 * EV)


def test_interface_defect_field_side_points_to_the_well():
    """界面读场侧取势阱（ZrO₂）侧：界面 0 在下侧(−1)、界面 1 在上侧(+1)。"""
    device = _zaz_device(t2_nm=1.0)
    ensemble = generate_defects(device, np.random.default_rng(7))
    assert np.all(ensemble.field_side[ensemble.interface_id == 0] == -1)
    assert np.all(ensemble.field_side[ensemble.interface_id == 1] == +1)
    assert np.all(ensemble.field_side[ensemble.layer_id >= 0] == 0)  # 体缺陷用常规读场


def test_layered_counts_match_per_layer_volume_and_surface_density():
    """体缺陷 = 浓度×层体积、界面缺陷 = Nₛ×A，与体缺陷预算分开计（ADR-005）。"""
    device = _zaz_device(t2_nm=1.0, ns_cm2=1.0e12, concentration_cm3=3.0e18)
    ensemble = generate_defects(device, np.random.default_rng(7))
    t_side = (9e-9 - 1e-9) / 2.0
    n_bulk_each = round(3.0e24 * device.area * t_side)
    n_iface_each = round(1.0e16 * device.area)
    assert np.sum(ensemble.layer_id == 0) == n_bulk_each
    assert np.sum(ensemble.layer_id == 2) == n_bulk_each
    assert np.sum(ensemble.interface_id == 0) == n_iface_each
    assert np.sum(ensemble.interface_id == 1) == n_iface_each
    assert len(ensemble) == 2 * n_bulk_each + 2 * n_iface_each


def test_global_min_separation_spans_layers_and_interfaces():
    """全局最小间距同时管跨层缺陷对、体缺陷到内部界面、缺陷到电极。"""
    device = _zaz_device(t2_nm=1.0, min_sep=0.5e-9)
    ensemble = generate_defects(device, np.random.default_rng(13))
    distances = pair_distances(ensemble.positions, device.Lx, device.Ly)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= 0.5e-9  # 含跨层对
    # 体缺陷避让内部界面与电极
    bulk_z = ensemble.positions[ensemble.layer_id >= 0, 2]
    for z_plane in np.concatenate(([0.0, device.Lz], device.stack.interface_positions)):
        assert np.all(np.abs(bulk_z - z_plane) >= 0.5e-9 - 1e-15)


def test_empty_device_returns_empty_ensemble():
    """所有层空、无界面缺陷 -> 空实现（R1-only 路径的地基，review F07）。"""
    device = _zaz_device(t2_nm=1.0, ns_cm2=0.0, concentration_cm3=0.0)
    ensemble = generate_defects(device, np.random.default_rng(1))
    assert len(ensemble) == 0
    assert ensemble.positions.shape == (0, 3)
    assert ensemble.charge_state.shape == (0,)
    assert len(empty_ensemble()) == 0


# --------------------------------------------------------------------------
# 空间/能级轮廓与几何工具
# --------------------------------------------------------------------------
def test_interface_profile_shape():
    """n(z) ∝ 1+(g-1)[exp(-z/λ)+exp(-(Lz-z)/λ)]：界面处 ≈ g，中间衰减到 ≈ 1。"""
    Lz, g, lam = 8e-9, 5.0, 1e-9
    assert interface_profile(np.array(0.0), Lz, g, lam) == pytest.approx(g, rel=1e-3)
    assert interface_profile(np.array(Lz), Lz, g, lam) == pytest.approx(g, rel=1e-3)
    middle = interface_profile(np.array(Lz / 2), Lz, g, lam)
    assert middle == pytest.approx(1.0 + 2.0 * (g - 1.0) * np.exp(-0.5 * Lz / lam), rel=1e-12)
    assert 1.0 < middle < 1.2
    assert interface_profile(np.array(Lz / 2), Lz, 1.0, lam) == pytest.approx(1.0)


def test_interface_mode_enriches_near_electrodes_without_changing_count():
    common = dict(concentration_cm3=1.0e20, min_separation=0.0)
    enrichment, decay = 5.0, 1e-9
    uniform = generate_defects(make_device(defects=make_defect_config(**common)), np.random.default_rng(11))
    enriched = generate_defects(
        make_device(
            defects=make_defect_config(
                spatial=DefectSpatialConfig(mode="interface", enrichment=enrichment, decay_length=decay),
                **common,
            )
        ),
        np.random.default_rng(11),
    )
    device = make_device()
    assert len(enriched) == len(uniform)  # 轮廓只改形状，不改总数

    def near_interface_fraction(ensemble):
        z = ensemble.positions[:, 2]
        return np.mean((z < decay) | (z > device.Lz - decay))

    z = np.linspace(0.0, device.Lz, 20001)
    profile = interface_profile(z, device.Lz, enrichment, decay)
    inside = (z < decay) | (z > device.Lz - decay)
    expected = trapezoid(profile * inside, z) / trapezoid(profile, z)
    assert expected == pytest.approx(0.44, abs=0.01)
    assert near_interface_fraction(enriched) == pytest.approx(expected, abs=0.09)
    assert near_interface_fraction(uniform) == pytest.approx(0.25, abs=0.05)
    assert near_interface_fraction(enriched) > 1.5 * near_interface_fraction(uniform)


def test_interface_enrichment_survives_standoff_shorter_than_lambda():
    """λ ≲ standoff 时富集仍可见：轮廓锚定到可采样窗口，峰落在距界面 standoff 处。

    旧写法把峰锚在界面上，λ < min_separation 时峰整个落进被禁区，采样窗口上
    近乎平坦、直方图看不出富集（本测试会失败）。锚到窗口后峰移到 z≈standoff。
    """
    standoff, decay = 0.5e-9, 0.2e-9  # λ < standoff
    device = make_device(
        defects=make_defect_config(
            concentration_cm3=1.0e21,  # 大浓度压低泊松噪声，单次实现即可判形
            min_separation=standoff,
            spatial=DefectSpatialConfig(mode="interface", enrichment=8.0, decay_length=decay),
        )
    )
    z = generate_defects(device, np.random.default_rng(3)).positions[:, 2]
    lo, hi = standoff, device.Lz - standoff  # 可采样窗口

    # 峰落在距界面 standoff（=窗口边界）处，而不是界面 z=0/Lz 上
    edge_zone = decay  # 一个 λ 宽的边带，恰好罩住尖峰
    near_boundary = np.mean((z < lo + edge_zone) | (z > hi - edge_zone))
    # 均匀分布在同一可采样窗口下落进这两条边带的期望占比
    uniform_frac = 2.0 * edge_zone / (hi - lo)
    assert near_boundary > 2.5 * uniform_frac  # 明显富集
    # standoff 内确实空（数值稳定性约束）
    assert z.min() >= lo - 1e-15
    assert z.max() <= hi + 1e-15


def test_expected_depth_density_conserves_count_and_tracks_profile():
    """解析深度轮廓与采样口径同源：数量守恒 ∫ρ dz = 浓度·层厚；峰锚在可采样窗口边界。"""
    standoff = 0.5e-9
    # uniform：平台 ρ = 浓度·层厚/窗口宽（standoff 把缺陷挤进更窄窗口，平台略高于浓度）
    dev = make_device(defects=make_defect_config(min_separation=standoff))
    conc = dev.stack.layers[0].defects.concentration
    z, rho = expected_depth_density(dev)
    window = dev.Lz - 2.0 * standoff
    assert trapezoid(rho, z) == pytest.approx(conc * dev.Lz)  # 单层：层厚 = Lz
    assert np.allclose(rho, conc * dev.Lz / window)  # 均匀平台
    assert z[0] == pytest.approx(standoff) and z[-1] == pytest.approx(dev.Lz - standoff)

    # interface：数量守恒不变，峰落在窗口两端而非层中心
    dev2 = make_device(
        defects=make_defect_config(
            min_separation=standoff,
            spatial=DefectSpatialConfig(mode="interface", enrichment=5.0, decay_length=1e-9),
        )
    )
    z2, rho2 = expected_depth_density(dev2)
    assert trapezoid(rho2, z2) == pytest.approx(conc * dev2.Lz)  # 轮廓只改形状不改数量
    assert rho2[0] > 2.0 * rho2[len(rho2) // 2]  # 边缘显著高于中心


def test_minimum_image_in_periodic_xy_only():
    Lx = Ly = 20e-9
    delta = np.array([[18e-9, -19e-9, 6e-9]])
    wrapped = minimum_image_delta(delta, Lx, Ly)
    assert wrapped[0, 0] == pytest.approx(-2e-9)
    assert wrapped[0, 1] == pytest.approx(1e-9)
    assert wrapped[0, 2] == pytest.approx(6e-9)  # z 不周期


def test_pair_distances_symmetric_with_zero_diagonal():
    device = make_device()
    ensemble = generate_defects(device, np.random.default_rng(4))
    distances = pair_distances(ensemble.positions, device.Lx, device.Ly)
    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diag(distances), 0.0)
    assert distances.max() <= np.sqrt((device.Lx / 2) ** 2 + (device.Ly / 2) ** 2 + device.Lz ** 2)


def test_hop_cutoff_radius_hits_the_requested_tolerance():
    """截断半径取最慢衰减（最小 α_min = 1/max(r_D)）：其余组合衰减更快。"""
    r_D = np.array([0.10e-9, 0.1662e-9])
    tolerance = 1e-24
    alpha_min = 1.0 / r_D.max()
    r_cut = hop_cutoff_radius(alpha_min, tolerance)
    slowest = np.exp(-r_cut * (1.0 / r_D.max() + 1.0 / r_D.max()))
    assert slowest == pytest.approx(tolerance)
    assert np.exp(-r_cut * (1.0 / r_D[0] + 1.0 / r_D[1])) < tolerance
    assert r_cut == pytest.approx(4.60e-9, rel=0.01)  # 参考预设（E_D=1.15 eV）下的取值
    for bad in (0.0, 1.0, -1e-3):
        with pytest.raises(ValueError):
            hop_cutoff_radius(alpha_min, bad)
    with pytest.raises(ValueError):
        hop_cutoff_radius(0.0, tolerance)


def test_neighbor_pairs_match_the_dense_matrix(monkeypatch):
    """近邻表必须与"稠密矩阵 + 掩码"给出完全相同的对、相同的距离。"""
    monkeypatch.setattr(defects_module, "_PAIR_CHUNK_ELEMENTS", 200)
    device = make_device(defects=make_defect_config(concentration_cm3=3.0e19))
    ensemble = generate_defects(device, np.random.default_rng(5))
    r_cut = 3.0e-9
    distances = pair_distances(ensemble.positions, device.Lx, device.Ly)
    expected = np.nonzero((distances < r_cut) & ~np.eye(len(ensemble), dtype=bool))

    pairs = neighbor_pairs(ensemble.positions, device.Lx, device.Ly, r_cut)
    assert len(pairs) == expected[0].size > 0
    assert np.array_equal(pairs.a, expected[0])
    assert np.array_equal(pairs.b, expected[1])
    assert np.array_equal(pairs.r, distances[expected])
    assert len(pairs) % 2 == 0
    with pytest.raises(ValueError):
        neighbor_pairs(ensemble.positions, device.Lx, device.Ly, 0.0)


def test_neighbor_pairs_keep_everything_when_the_cutoff_covers_the_box():
    device = make_device()
    ensemble = generate_defects(device, np.random.default_rng(6))
    n = len(ensemble)
    pairs = neighbor_pairs(ensemble.positions, device.Lx, device.Ly, 1.0)
    assert len(pairs) == n * (n - 1)


def test_neighbor_pairs_tolerate_empty_and_single():
    """空 / 单个缺陷时没有任何有序对，返回空表而不是抛异常（review F07）。"""
    assert len(neighbor_pairs(np.zeros((0, 3)), 20e-9, 20e-9, 3e-9)) == 0
    assert len(neighbor_pairs(np.array([[1e-9, 1e-9, 1e-9]]), 20e-9, 20e-9, 3e-9)) == 0
