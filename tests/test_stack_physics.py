"""变 κ 泊松、分段能带、V_bi 与界面单侧场的检验（Phase 3 = ADR-011 + 带阶）。

这一批盯住 Phase 3 的物理交付：
- 变 κ 泊松（xy-FFT + z-Thomas，κ 按单元、界面对齐节点、横向项算术平均）：
  无电荷解析分压 / 串联电容（S_κ=Σt_ℓ/κ_ℓ）/ 面电荷电位移跳跃 / 变系数离散残差 /
  界面对齐校验 / 同材料多层退回单层求解器；
- 分段能带：E_C(z) 有真实台阶、V_bi 关系 q·V_bi = B_c + ΣΔE_C − B_a；
- 界面单侧场读取接口（F08）：界面缺陷读单侧差分场、体缺陷读中心差分。

数值锚点与 review §4.2 的独立一维方案一致（4/1/4 nm 结构 C/A ≈ 0.0275285 F/m²）。
"""

import numpy as np
import pytest
from scipy import constants

from kmc_leakage.bands import conduction_band_edge
from kmc_leakage.config import (
    DeviceConfig,
    DielectricMaterial,
    DielectricStack,
    ElectrodeMaterial,
    GridConfig,
    Interface,
    Layer,
)
from kmc_leakage.electrostatics import PoissonSolver

EV = constants.electron_volt
M_E = constants.electron_mass
Q = constants.elementary_charge
E0 = constants.epsilon_0


def _material(name, kappa, eps_opt=5.6, band_gap_eV=5.4, chi_eV=2.86, m_cb=1.2):
    return DielectricMaterial(
        name=name, kappa=kappa, eps_opt=eps_opt, band_gap=band_gap_eV * EV,
        electron_affinity=chi_eV * EV, m_cb=m_cb * M_E, m_vb=2.5 * M_E, m_di=m_cb * M_E,
        phonon_energy=0.04 * EV, huang_rhys=1.0, attempt_frequency=1.0e13,
    )


ZRO2 = _material("ZrO2", 38.0)
AL2O3 = _material("Al2O3", 9.0, eps_opt=3.0, band_gap_eV=6.4, chi_eV=1.4, m_cb=0.3)
_ELECTRODE = ElectrodeMaterial("TiN", 4.6 * EV, M_E, 5.0 * EV)


def _zaz_device(t1=4e-9, t2=1e-9, t3=4e-9, cathode=_ELECTRODE, anode=_ELECTRODE,
                barrier_height=1.74 * EV, interfaces=None):
    """ZrO₂/Al₂O₃/ZrO₂ 对称叠层（总厚 t1+t2+t3）。"""
    if interfaces is None:
        interfaces = (Interface(), Interface())
    stack = DielectricStack(
        materials={"ZrO2": ZRO2, "Al2O3": AL2O3},
        layers=(Layer("L0", "ZrO2", t1), Layer("L1", "Al2O3", t2), Layer("L2", "ZrO2", t3)),
        interfaces=interfaces,
    )
    return DeviceConfig(50e-9, 50e-9, cathode, anode, stack, barrier_height=barrier_height)


def _uniform_multilayer_device(barrier_height=1.74 * EV):
    """三层同材料（ZrO₂）、ΔE_C=0——单层退化的多层写法（ADR-018 电静电侧）。"""
    stack = DielectricStack(
        materials={"ZrO2": ZRO2},
        layers=(Layer("L0", "ZrO2", 4e-9), Layer("L1", "ZrO2", 1e-9), Layer("L2", "ZrO2", 4e-9)),
        interfaces=(Interface(delta_Ec=0.0), Interface(delta_Ec=0.0)),
    )
    return DeviceConfig(50e-9, 50e-9, _ELECTRODE, _ELECTRODE, stack, barrier_height=barrier_height)


def _single_layer_device(kappa=38.0, Lz=9e-9, barrier_height=1.74 * EV):
    return DeviceConfig(
        50e-9, 50e-9, _ELECTRODE, _ELECTRODE,
        DielectricStack.single(_material("ZrO2", kappa), Lz), barrier_height=barrier_height,
    )


# --------------------------------------------------------------------------
# 变 κ 泊松
# --------------------------------------------------------------------------
def test_kappa_is_taken_per_cell():
    """κ 按单元取值：ZAZ 的单元 κ 只有 38 与 9 两种，界面落在节点上。"""
    solver = PoissonSolver(_zaz_device(), GridConfig(8, 8, 36))
    assert solver.hz == pytest.approx(0.25e-9)
    assert set(np.round(np.unique(solver._kappa_cells), 6)) == {9.0, 38.0}
    # 4 nm ZrO₂ = 16 个单元，1 nm Al₂O₃ = 4 个单元，再 16 个 ZrO₂
    assert list(solver._kappa_cells[:16]) == [38.0] * 16
    assert list(solver._kappa_cells[16:20]) == [9.0] * 4
    assert list(solver._kappa_cells[20:]) == [38.0] * 16


def test_stack_voltage_division_matches_analytic():
    """无电荷时电势按 S_κ=Σt_ℓ/κ_ℓ 分压：φ(z) 分段线性，界面处斜率跳变。"""
    device = _zaz_device()
    solver = PoissonSolver(device, GridConfig(8, 8, 36))
    V = 4.0
    phi = solver.lateral_average(solver.solve(np.zeros(solver.grid.shape), V))

    t = np.array([4e-9, 1e-9, 4e-9])
    kappa = np.array([38.0, 9.0, 38.0])
    S_kappa = np.sum(t / kappa)
    edges = device.stack.layer_edges
    cumulative = np.concatenate(([0.0], np.cumsum(t / kappa)))
    analytic = V * np.interp(solver.z, edges, cumulative) / S_kappa
    assert np.allclose(phi, analytic, atol=1e-13)
    # 横向无结构（DC 模式），且界面处斜率确实跳变（Al₂O₃ 里更陡）
    slope_zro2 = (phi[8] - phi[7]) / solver.hz
    slope_al2o3 = (phi[18] - phi[17]) / solver.hz
    assert slope_al2o3 / slope_zro2 == pytest.approx(38.0 / 9.0, rel=1e-9)


def test_stack_capacitance_is_series_dielectric():
    """无电荷电位移 D=κε0E 跨层守恒，单位面积电容 C/A=ε0/S_κ（review §4.2）。"""
    device = _zaz_device()
    solver = PoissonSolver(device, GridConfig(8, 8, 36))
    V = 3.0
    phi = solver.lateral_average(solver.solve(np.zeros(solver.grid.shape), V))

    t = np.array([4e-9, 1e-9, 4e-9])
    kappa = np.array([38.0, 9.0, 38.0])
    S_kappa = np.sum(t / kappa)
    D_analytic = E0 * V / S_kappa

    # 每层取一个内部单元的场，D=κε0·(-dφ/dz)（φ 从 0 升到 V，故 E_z<0、D<0）
    D_layers = []
    for kap, node in ((38.0, 8), (9.0, 18), (38.0, 28)):
        E_z = -(phi[node + 1] - phi[node]) / solver.hz
        D_layers.append(kap * E0 * E_z)
    D_layers = np.abs(np.array(D_layers))
    assert np.allclose(D_layers, D_analytic, rtol=1e-12)  # 跨层守恒且等于解析值
    assert E0 / S_kappa == pytest.approx(0.02752847483616, rel=1e-9)  # 4/1/4 nm 的 C/A


def test_surface_charge_gives_displacement_jump():
    """界面固定面电荷 σ 让电位移跳变 D_above − D_below = σ（Gauss，变 κ 下成立）。"""
    device = _zaz_device()
    solver = PoissonSolver(device, GridConfig(8, 8, 36))
    sigma = 1.602176634e-3  # C/m^2
    k = 16  # z = 4 nm 界面节点
    rho = np.zeros(solver.grid.shape)
    rho[:, :, k] = sigma / solver.hz  # 均匀面电荷铺在界面节点的控制体上
    phi = solver.lateral_average(solver.solve(rho, 0.0))

    D_below = -38.0 * E0 * (phi[k] - phi[k - 1]) / solver.hz  # 下侧单元 ZrO₂
    D_above = -9.0 * E0 * (phi[k + 1] - phi[k]) / solver.hz  # 上侧单元 Al₂O₃
    assert (D_above - D_below) == pytest.approx(sigma, rel=1e-9)


def test_response_matrix_is_reciprocal_across_kappa_step():
    """静电响应互易：G_ab = G_ba 到机器精度，即使缺陷跨越 Al₂O₃ 层（变 κ 算符自伴）。

    F02 的 U_ab 对称化依赖这一条；界面处 κ 突变不破坏响应矩阵的对称性。
    """
    from kmc_leakage.kmc import ElectrostaticCoupling

    solver = PoissonSolver(_zaz_device(), GridConfig(8, 8, 36))
    positions = np.array([[10e-9, 12e-9, 2e-9], [30e-9, 20e-9, 4.5e-9], [25e-9, 25e-9, 7e-9]])
    G = ElectrostaticCoupling(solver, positions).phi_matrix
    assert np.max(np.abs(G - G.T)) / np.max(np.abs(G)) < 1e-12


def test_variable_kappa_solution_satisfies_discrete_poisson():
    """随机体电荷下，解代回**变系数**离散算符，残差到舍入误差量级（含非零横向模式）。"""
    device = _zaz_device()
    solver = PoissonSolver(device, GridConfig(8, 8, 36))
    rng = np.random.default_rng(0)
    rho = rng.normal(0.0, 1e3, solver.grid.shape)
    rho[:, :, 0] = rho[:, :, -1] = 0.0
    phi = solver.solve(rho, 1.5)

    kc, hz = solver._kappa_cells, solver.hz
    lap_xy = (np.roll(phi, -1, 0) + np.roll(phi, 1, 0) - 2.0 * phi) / solver.hx ** 2
    lap_xy += (np.roll(phi, -1, 1) + np.roll(phi, 1, 1) - 2.0 * phi) / solver.hy ** 2
    residual = np.zeros_like(phi)
    for k in range(1, solver.nz):
        km, kp = kc[k - 1], kc[k]
        kbar = 0.5 * (km + kp)
        flux = (kp * (phi[:, :, k + 1] - phi[:, :, k]) - km * (phi[:, :, k] - phi[:, :, k - 1])) / hz ** 2
        residual[:, :, k] = flux + kbar * lap_xy[:, :, k] + rho[:, :, k] / E0
    scale = np.abs(rho).max() / E0
    assert np.abs(residual[:, :, 1:-1]).max() / scale < 1e-8


def test_interface_must_align_to_grid_nodes():
    """层厚不是 hz 整数倍时（界面落在节点之间）构造求解器直接报错。"""
    device = _zaz_device()
    with pytest.raises(ValueError) as exc:
        PoissonSolver(device, GridConfig(8, 8, 35))  # hz=9/35 nm，4 nm 界面不对齐
    assert "对齐" in str(exc.value) or "align" in str(exc.value).lower()
    # 对齐的 nz 正常构造
    PoissonSolver(device, GridConfig(8, 8, 36))


def test_uniform_multilayer_matches_single_layer_solver():
    """三层同材料（ΔE_C=0）与单层 9 nm 的求解器给出逐位相同的电势（ADR-018）。"""
    multilayer = PoissonSolver(_uniform_multilayer_device(), GridConfig(8, 8, 36))
    single = PoissonSolver(_single_layer_device(), GridConfig(8, 8, 36))
    rng = np.random.default_rng(3)
    rho = rng.normal(0.0, 1e3, single.grid.shape)
    rho[:, :, 0] = rho[:, :, -1] = 0.0
    phi_multi = multilayer.solve(rho, 2.0)
    phi_single = single.solve(rho, 2.0)
    assert np.allclose(phi_multi, phi_single, rtol=1e-12, atol=1e-15)


def test_single_layer_solver_unchanged_by_variable_kappa_rewrite():
    """单层（均匀 κ）无电荷解仍严格线性——变 κ 改写不动单层路径。"""
    solver = PoissonSolver(_single_layer_device(), GridConfig(16, 16, 36))
    phi = solver.solve(np.zeros(solver.grid.shape), 2.0)
    expected = np.linspace(0.0, 2.0, solver.nz + 1)
    assert np.allclose(solver.lateral_average(phi), expected, atol=1e-14)
    assert np.allclose(phi, expected[None, None, :], atol=1e-14)


# --------------------------------------------------------------------------
# 分段能带 + V_bi
# --------------------------------------------------------------------------
def test_conduction_band_has_real_steps():
    """E_C(z) 在界面处有真实台阶：B_ℓ = [E_B, E_B+ΔE_C, E_B]（电子亲和能对齐）。"""
    device = _zaz_device()
    # 亲和能对齐：ΔE_C(ZrO₂→Al₂O₃)=2.86−1.4=1.46 eV，反向界面 −1.46 eV
    assert np.allclose(device.layer_band_references() / EV, [1.74, 3.20, 1.74])
    z = np.linspace(0.0, device.Lz, 37)
    E_C = conduction_band_edge(device, z, np.zeros_like(z)) / EV
    # 阻挡层内抬高一个 ΔE_C，两侧 ZrO₂ 回到 E_B——真实台阶，不是插出来的斜坡
    assert np.interp(3.5e-9, z, E_C) == pytest.approx(1.74)
    assert np.interp(4.5e-9, z, E_C) == pytest.approx(3.20)
    assert np.interp(5.5e-9, z, E_C) == pytest.approx(1.74)


def test_explicit_delta_ec_override_enters_band_and_vbi():
    """显式 ΔE_C 覆盖进入 B_ℓ 累积与 V_bi 关系（不只画在能带图上，review §4.1）。"""
    device = _zaz_device(interfaces=(Interface(delta_Ec=1.0 * EV), Interface(delta_Ec=-0.4 * EV)))
    assert np.allclose(device.layer_band_references() / EV, [1.74, 2.74, 2.34])
    # q·V_bi = B_c + ΣΔE_C − B_a；对称接触势垒 B_c=B_a=1.74，ΣΔE_C=0.6
    assert device.built_in_potential() == pytest.approx(0.6)


def test_built_in_potential_follows_band_reference_relation():
    """q·V_bi = B_c + ΣΔE_C − B_a：对称亲和能对齐叠层 ΣΔE_C=0 → V_bi=0。"""
    assert _zaz_device().built_in_potential() == pytest.approx(0.0)
    # 非对称接触（Pt 阳极）：V_bi = (B_c − B_a)/q，带阶仍抵消
    asym = _zaz_device(anode=ElectrodeMaterial("Pt", 5.3 * EV, M_E, 5.0 * EV), barrier_height=None)
    assert asym.barrier(asym.cathode) == pytest.approx(1.74 * EV)
    assert asym.barrier(asym.anode) == pytest.approx(2.44 * EV)
    assert asym.built_in_potential() == pytest.approx(-0.7)


# --------------------------------------------------------------------------
# 界面单侧场读取接口（F08）
# --------------------------------------------------------------------------
def test_single_side_field_captures_discontinuity():
    """界面上 E_z 两侧差 κ 比：单侧差分给出正确的一侧场，中心差分给的是平均（错）。"""
    device = _zaz_device()
    solver = PoissonSolver(device, GridConfig(8, 8, 36))
    phi = solver.solve(np.zeros(solver.grid.shape), 4.0)
    pos = np.array([[25e-9, 25e-9, 4e-9]])  # 界面节点

    ez_lower, ez_upper = solver.electric_field_z_sided(phi)
    f_lower = solver.interpolate(ez_lower, pos)[0]  # ZrO₂ 侧
    f_upper = solver.interpolate(ez_upper, pos)[0]  # Al₂O₃ 侧
    _, _, ez_center = solver.electric_field(phi)
    f_center = solver.interpolate(ez_center, pos)[0]
    # D=κε0E 连续 → E 跨界面按 κ 比跳变
    assert f_upper / f_lower == pytest.approx(38.0 / 9.0, rel=1e-9)
    # 中心差分是两侧单侧斜率的平均，落在两者之间——F08 指出的错处
    assert f_center == pytest.approx(0.5 * (f_lower + f_upper), rel=1e-9)


def test_field_vectors_at_selects_side_by_field_side():
    """field_vectors_at 按 field_side 取场：+1 上侧、−1 下侧、0 中心差分。"""
    device = _zaz_device()
    solver = PoissonSolver(device, GridConfig(8, 8, 36))
    phi = solver.solve(np.zeros(solver.grid.shape), 4.0)
    pos = np.array([[25e-9, 25e-9, 4e-9], [25e-9, 25e-9, 4e-9], [25e-9, 25e-9, 4e-9]])
    field_side = np.array([-1, +1, 0])

    ez_lower, ez_upper = solver.electric_field_z_sided(phi)
    _, _, ez_center = solver.electric_field(phi)
    got = solver.field_vectors_at(phi, pos, field_side)
    assert got[2, 0] == pytest.approx(solver.interpolate(ez_lower, pos[:1])[0], rel=1e-12)
    assert got[2, 1] == pytest.approx(solver.interpolate(ez_upper, pos[1:2])[0], rel=1e-12)
    assert got[2, 2] == pytest.approx(solver.interpolate(ez_center, pos[2:3])[0], rel=1e-12)


def test_field_vectors_at_bulk_matches_centered_interpolation():
    """全体缺陷（field_side==0）时逐位等于对三个中心差分分量各自插值——回归锚点。"""
    solver = PoissonSolver(_single_layer_device(), GridConfig(8, 8, 36))
    rng = np.random.default_rng(1)
    rho = solver.deposit(np.array([[10e-9, 12e-9, 5e-9]]), np.array([-Q]))
    phi = solver.solve(rho, 1.0)
    pos = np.array([[5e-9, 7e-9, 3e-9], [20e-9, 30e-9, 6e-9]])
    manual = np.array([solver.interpolate(c, pos) for c in solver.electric_field(phi)])
    got = solver.field_vectors_at(phi, pos, np.zeros(len(pos), dtype=int))
    assert np.array_equal(got, manual)
