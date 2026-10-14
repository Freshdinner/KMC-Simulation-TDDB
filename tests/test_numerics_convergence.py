"""数值参数的收敛性检验。

kmc.py 里的积分点数（_TRAP_PATH_POINTS / _R1_PATH_POINTS /
_R1_ENERGY_POINTS / _N_PHONON）不是随手定的，这里给出对应的收敛性依据：
把点数逐级加倍，结果的变化必须落在下面的容差内。
"""

import numpy as np
import pytest
from scipy import constants

from conftest import make_device
from kmc_leakage import kmc
from kmc_leakage.rates import phonon_transition_probability, r1_electrode_electrode, wkb_transmission

EV = constants.electron_volt
M_E = constants.electron_mass


def _band(device, voltage, n):
    z = np.linspace(0.0, device.Lz, n)
    return z, device.barrier(device.negative) - constants.elementary_charge * voltage * z / device.Lz


def test_trap_path_points_are_converged():
    """电极↔陷阱路径取 _TRAP_PATH_POINTS 点时，P(E) 已收敛到 1e-4 以内。"""
    device = make_device()
    z_trap, energy = 2.5e-9, -0.035 * EV
    z_nodes, band_nodes = _band(device, 3.0, 33)

    def transmission(n):
        z = np.linspace(0.0, z_trap, n)
        return wkb_transmission(z, np.interp(z, z_nodes, band_nodes), np.array(energy), device.stack.single_material.m_di)

    reference = transmission(4097)
    coarse = transmission(kmc._TRAP_PATH_POINTS)
    assert abs(coarse / reference - 1.0) < 1e-4
    # 加倍点数带来的变化同样在容差内
    assert abs(transmission(2 * kmc._TRAP_PATH_POINTS - 1) / coarse - 1.0) < 1e-4


def test_r1_integration_is_converged():
    """R1 的能量点数与路径点数加倍后，两个分量的变化均 < 1%。"""
    device = make_device()
    voltage, temperature = 3.0, 298.15
    args = (device.area, voltage, temperature, M_E, device.stack.single_material.m_di)

    z, band = _band(device, voltage, kmc._R1_PATH_POINTS)
    tunnel, thermionic = r1_electrode_electrode(z, band, *args, n_points=kmc._R1_ENERGY_POINTS)

    z_fine, band_fine = _band(device, voltage, 4 * kmc._R1_PATH_POINTS)
    tunnel_fine, thermionic_fine = r1_electrode_electrode(
        z_fine, band_fine, *args, n_points=8 * kmc._R1_ENERGY_POINTS
    )
    assert tunnel == pytest.approx(tunnel_fine, rel=1e-2)
    assert thermionic == pytest.approx(thermionic_fine, rel=1e-2)


def test_phonon_sum_truncation_is_converged():
    """R4/R5 的声子求和截到 _N_PHONON 项，尾部贡献 < 1e-4。"""
    for temperature in (273.15, 398.15):
        weights = phonon_transition_probability(np.arange(200), 1.0, 0.04 * EV, temperature)
        kept = weights[: kmc._N_PHONON].sum()
        assert weights[kmc._N_PHONON :].sum() / kept < 1e-4


def test_grid_resolution_effect_on_barrier_profile():
    """z 向网格加密不改变势垒剖面（无电荷时电势严格线性），侧向分辨率同理。

    这一条把"网格分辨率如何选取"这个待确认项的现状钉住：无空间电荷时结果
    与网格无关，网格只影响空间电荷的分辨能力。
    """
    from kmc_leakage.config import GridConfig
    from kmc_leakage.electrostatics import PoissonSolver

    device = make_device()
    profiles = []
    for grid in (GridConfig(8, 8, 16), GridConfig(16, 16, 32), GridConfig(32, 32, 64)):
        solver = PoissonSolver(device, grid)
        phi = solver.solve(np.zeros(grid.shape), 3.0)
        profiles.append(np.interp(np.linspace(0, device.Lz, 17), solver.z, solver.lateral_average(phi)))
    assert np.allclose(profiles[0], profiles[1], atol=1e-14)
    assert np.allclose(profiles[1], profiles[2], atol=1e-14)
