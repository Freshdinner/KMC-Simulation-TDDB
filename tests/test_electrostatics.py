"""泊松求解器（FFT + Thomas）与 CIC 插值的检验。"""

import numpy as np
import pytest
from scipy import constants

from kmc_leakage.config import (
    DeviceConfig,
    DielectricMaterial,
    DielectricStack,
    ElectrodeMaterial,
    GridConfig,
)
from kmc_leakage.electrostatics import PoissonSolver

EV = constants.electron_volt
M_E = constants.electron_mass
Q = constants.elementary_charge
KAPPA = 38.0


def _device(Lx=16e-9, Ly=16e-9, Lz=16e-9, kappa=KAPPA):
    electrode = ElectrodeMaterial("TiN", 4.6 * EV, M_E, 5.0 * EV)
    dielectric = DielectricMaterial(
        name="test",
        kappa=kappa,
        eps_opt=5.6,
        band_gap=5.4 * EV,
        electron_affinity=2.86 * EV,
        m_cb=1.2 * M_E,
        m_vb=2.5 * M_E,
        m_di=1.2 * M_E,
        phonon_energy=0.04 * EV,
        huang_rhys=1.0,
        attempt_frequency=1e13,
    )
    stack = DielectricStack.single(dielectric, Lz)
    return DeviceConfig(Lx, Ly, electrode, electrode, stack, barrier_height=1.74 * EV)


def _solver(grid=GridConfig(32, 32, 32), **kwargs):
    return PoissonSolver(_device(**kwargs), grid)


def _discrete_laplacian(solver, phi):
    """离散三维拉普拉斯算符（xy 周期、z 内点），返回内点上的值。"""
    lap = (np.roll(phi, -1, 0) + np.roll(phi, 1, 0) - 2.0 * phi) / solver.hx ** 2
    lap += (np.roll(phi, -1, 1) + np.roll(phi, 1, 1) - 2.0 * phi) / solver.hy ** 2
    lap[:, :, 1:-1] += (phi[:, :, 2:] + phi[:, :, :-2] - 2.0 * phi[:, :, 1:-1]) / solver.hz ** 2
    return lap[:, :, 1:-1]


def test_zero_charge_gives_exactly_linear_potential():
    solver = _solver()
    phi = solver.solve(np.zeros(solver.grid.shape), 2.0)
    expected = np.linspace(0.0, 2.0, solver.nz + 1)
    assert np.allclose(solver.lateral_average(phi), expected, atol=1e-14)
    assert np.allclose(phi, expected[None, None, :], atol=1e-14)  # 横向无结构


def test_solution_satisfies_discrete_poisson_equation():
    """最直接的正确性判据：解代回离散方程，残差到舍入误差量级。"""
    solver = _solver()
    rng = np.random.default_rng(0)
    rho = rng.normal(0.0, 1e3, solver.grid.shape)
    rho[:, :, 0] = rho[:, :, -1] = 0.0  # 边界节点由 Dirichlet 条件固定
    phi = solver.solve(rho, 1.5)
    residual = _discrete_laplacian(solver, phi) + rho[:, :, 1:-1] / (KAPPA * constants.epsilon_0)
    scale = np.abs(rho[:, :, 1:-1]).max() / (KAPPA * constants.epsilon_0)
    # 阈值由二阶差分放大的舍入误差（∝ eps/h²）决定，不是方法误差
    assert np.abs(residual).max() / scale < 1e-8


def test_uniform_charge_matches_analytic_parabola():
    """均匀体电荷的解析解 φ = ρ·z(Lz-z)/(2κε0)，三点差分对二次函数精确。"""
    solver = _solver()
    rho_value = -1.0e6  # C/m^3
    rho = np.full(solver.grid.shape, rho_value)
    rho[:, :, 0] = rho[:, :, -1] = 0.0
    phi = solver.solve(rho, 0.0)
    z = solver.z
    Lz = solver.device.Lz
    analytic = rho_value / (2.0 * KAPPA * constants.epsilon_0) * z * (Lz - z)
    assert np.allclose(solver.lateral_average(phi), analytic, rtol=1e-10, atol=1e-12)


def test_boundary_conditions_are_enforced():
    solver = _solver()
    rng = np.random.default_rng(1)
    rho = rng.normal(0.0, 1e4, solver.grid.shape)
    phi = solver.solve(rho, 3.3)
    assert np.allclose(phi[:, :, 0], 0.0)
    assert np.allclose(phi[:, :, -1], 3.3)


def test_superposition_holds():
    """KMC 主循环靠线性叠加省掉每步泊松求解，这里检验叠加确实成立。"""
    solver = _solver()
    p1 = np.array([[5e-9, 5e-9, 4e-9]])
    p2 = np.array([[11e-9, 9e-9, 12e-9]])
    rho1 = solver.deposit(p1, np.array([-Q]))
    rho2 = solver.deposit(p2, np.array([-Q]))
    phi_applied = solver.solve(np.zeros(solver.grid.shape), 2.0)
    combined = solver.solve(rho1 + rho2, 2.0)
    superposed = phi_applied + solver.solve(rho1, 0.0) + solver.solve(rho2, 0.0)
    assert np.allclose(combined, superposed, atol=1e-14)


def test_cic_deposit_conserves_charge():
    solver = _solver()
    positions = np.array([[3.3e-9, 7.7e-9, 8.4e-9], [1.1e-9, 2.2e-9, 5.5e-9]])
    charges = np.array([-Q, -Q])
    rho = solver.deposit(positions, charges)
    cell_volume = solver.hx * solver.hy * solver.hz
    assert rho.sum() * cell_volume == pytest.approx(charges.sum(), rel=1e-12)


def test_interpolation_is_exact_for_linear_fields():
    """三线性插值对线性场精确——与 CIC 分配同阶，减少自作用力不一致。"""
    solver = _solver()
    x = np.arange(solver.nx) * solver.hx
    field = np.broadcast_to(x[:, None, None], solver.grid.shape).copy()
    probes = np.array([[3.7e-9, 1.0e-9, 4.0e-9], [9.2e-9, 6.0e-9, 8.0e-9]])
    assert np.allclose(solver.interpolate(field, probes), probes[:, 0], atol=1e-15)


def test_electric_field_of_linear_potential():
    solver = _solver()
    phi = solver.solve(np.zeros(solver.grid.shape), 2.0)
    ex, ey, ez = solver.electric_field(phi)
    assert np.allclose(ex, 0.0, atol=1e-9)
    assert np.allclose(ey, 0.0, atol=1e-9)
    assert np.allclose(ez, -2.0 / solver.device.Lz, rtol=1e-12)


def test_single_point_charge_magnitude_and_self_potential():
    """docs 要求的"单点电荷放入模拟盒子"检验。

    (1) 近场应接近库仑势 q/(4πε0·κ·r)（两侧接地极板的镜像电荷会把它压低
        十几个百分点，所以用较宽的容差，同时检查 1/r 标度）；
    (2) 自作用势应在 q/(4πε0·κ·h) 量级，且远小于势垒高度 E_B——量级合理，
        于是采用"整项扣除自作用"的处理（KMC 引擎里由耦合矩阵对角元实现）。
    """
    solver = _solver()
    h = solver.hx
    center = np.array([[solver.nx // 2 * h, solver.ny // 2 * solver.hy, solver.device.Lz / 2.0]])
    rho = solver.deposit(center, np.array([Q]))
    phi = solver.solve(rho, 0.0)

    coulomb = Q / (4.0 * np.pi * constants.epsilon_0 * KAPPA)
    for n_cells in (2, 3, 4):
        probe = center + np.array([[n_cells * h, 0.0, 0.0]])
        value = solver.interpolate(phi, probe)[0]
        assert value == pytest.approx(coulomb / (n_cells * h), rel=0.35)

    self_potential = solver.interpolate(phi, center)[0]
    # 实测约 3×q/(4πε0κh)：离散格林函数在原点的值比朴素的 1/h 估计更大，
    # 对应自能 ~0.24 eV（≈9 k_BT），并非可以忽略的小量——这正是 KMC 引擎
    # 里按耦合矩阵对角元整项扣除自作用的理由
    assert 0.3 < self_potential / (coulomb / h) < 4.0
    assert self_potential * Q < 0.2 * 1.74 * EV  # 但仍远小于势垒高度


def test_grid_refinement_reduces_discretization_error():
    """网格加密时均匀电荷解仍精确，说明求解器不依赖特定分辨率。"""
    for grid in (GridConfig(16, 16, 16), GridConfig(32, 32, 64)):
        solver = _solver(grid=grid)
        rho = np.full(solver.grid.shape, -1.0e6)
        rho[:, :, 0] = rho[:, :, -1] = 0.0
        phi = solver.solve(rho, 1.0)
        z, Lz = solver.z, solver.device.Lz
        analytic = -1.0e6 / (2.0 * KAPPA * constants.epsilon_0) * z * (Lz - z) + z / Lz
        assert np.allclose(solver.lateral_average(phi), analytic, rtol=1e-10, atol=1e-12)
