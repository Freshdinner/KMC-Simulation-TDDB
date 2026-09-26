"""能带剖面与费米分布。

**全局能量零点 = 阴极费米能级。** 阳极费米能级在偏压 U 下位于 -eU。

导带底剖面 E_C(z) = E_B(阴极侧) - e·φ(z)，其中 φ 由泊松方程给出
（φ(0)=0，φ(Lz)=U + 内建电势）。按 docs 的势垒形状约定，界面处能带保持
直线，**不做镜像力圆化、势垒高度一律取零场值 E_B**。
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy import constants

from .config import DeviceConfig


@dataclass(frozen=True)
class BandProfile:
    """沿 z 的一维能带剖面（可视化与 WKB 势垒共用）。

    `z` 上每个内部界面只存**上侧**那层的值（`layer_indices_of` 用右开区间）。
    界面下侧的值另存在 `E_C_lower` / `E_V_lower` 里，`stepped` 把两侧拼成
    双值序列——画出来是真实台阶，不是把台阶连成斜坡（ADR-016 ②）。单层没有
    内部界面，三个界面字段都是空数组。
    """

    z: np.ndarray  # (nz+1,) [m]
    E_C: np.ndarray  # 导带底 [J]
    E_V: np.ndarray  # 价带顶 [J]
    E_F_cathode: float  # [J]，按约定恒为 0
    E_F_anode: float  # [J]，= -eU
    voltage: float  # [V]
    interface_z: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 内部界面 z [m]
    E_C_lower: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 各界面下侧 E_C [J]
    E_V_lower: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 各界面下侧 E_V [J]

    @property
    def stepped(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(z, E_C, E_V) 三条序列，界面处插入下侧值 → 同一个 z 上有两个能量值。

        连成折线时界面处是一段竖直的台阶。无内部界面时逐位返回原数组。
        """
        if np.size(self.interface_z) == 0:
            return self.z, self.E_C, self.E_V
        # 界面节点坐标可能比层边界小 1 ulp，减容差保证下侧值插在界面节点之前
        idx = np.searchsorted(self.z, self.interface_z - 1e-9 * self.z[-1], side="left")
        return (
            np.insert(self.z, idx, self.interface_z),
            np.insert(self.E_C, idx, self.E_C_lower),
            np.insert(self.E_V, idx, self.E_V_lower),
        )


def anode_fermi_level(voltage: float) -> float:
    """阳极费米能级 [J]（阴极费米能级为零点）。"""
    return -constants.elementary_charge * voltage


def top_boundary_potential(device: DeviceConfig, voltage: float) -> float:
    """z=Lz 处的 Dirichlet 电势 [V] = 外加偏压 + 功函数差引起的内建电势。"""
    return voltage + device.built_in_potential()


def conduction_band_edge(device: DeviceConfig, z: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """由电势给出导带底能量 E_C(z) = B_ℓ(z) - e·φ(z) [J]。

    分段能带的真实台阶：`band_reference_profile` 给出每个 z 所属层的导带基准
    B_ℓ，界面处 B_ℓ 跳变（review §4.1）。单层退化时 B_ℓ 恒为接触势垒 E_B，退回
    E_C = E_B - e·φ。`z` 与 `phi` 逐点对应（同一套 z 网格）。
    """
    return device.band_reference_profile(z) - constants.elementary_charge * np.asarray(phi)


def trap_levels(band_reference: np.ndarray, phi_at_defects: np.ndarray, ED: np.ndarray) -> np.ndarray:
    """缺陷能级（电子总能量）E_i = B_ℓ(i) - e·φ(r_i) - E_D,i [J]。

    `band_reference` 是每个缺陷所属材料侧的导带基准 B_ℓ（单层退化时恒为接触
    势垒 E_B）；界面缺陷取势阱侧的 B。见 review §4.1、ADR-008。
    """
    return (
        np.asarray(band_reference, dtype=float)
        - constants.elementary_charge * np.asarray(phi_at_defects, dtype=float)
        - np.asarray(ED, dtype=float)
    )


def fermi_dirac(energy: np.ndarray, fermi_level: float, temperature: float) -> np.ndarray:
    """费米-狄拉克分布 f(E) = 1/(1+exp((E-E_F)/k_BT))，数值安全实现。"""
    x = (np.asarray(energy, dtype=float) - fermi_level) / (constants.Boltzmann * temperature)
    # 用 expit 的等价写法避免 exp 溢出
    out = np.empty_like(x)
    pos = x > 0
    out[pos] = np.exp(-x[pos]) / (1.0 + np.exp(-x[pos]))
    out[~pos] = 1.0 / (1.0 + np.exp(x[~pos]))
    return out


def band_profile(
    device: DeviceConfig,
    z: np.ndarray,
    phi: np.ndarray,
    voltage: float,
) -> BandProfile:
    """组装一维能带剖面。`phi` 是横向平均后的电势剖面 [V]。

    多层时另外算出每个内部界面**下侧**的 E_C / E_V（上侧的值已经在 `z` 网格上），
    供 `BandProfile.stepped` 拼成双值台阶用。单层这三项为空。
    """
    z = np.asarray(z, dtype=float)
    E_C = conduction_band_edge(device, z, phi)
    stack = device.stack
    interface_z = stack.interface_positions
    n_iface = interface_z.size
    # 界面 j 的下侧是第 j 层：E_C = B_j − eφ(z_j)，E_V = E_C − Eg_j
    phi_iface = np.interp(interface_z, z, np.asarray(phi, dtype=float))
    E_C_lower = (
        device.layer_band_references()[:n_iface] - constants.elementary_charge * phi_iface
    )
    return BandProfile(
        z=z,
        E_C=E_C,
        E_V=E_C - device.band_gap_profile(z),
        E_F_cathode=0.0,
        E_F_anode=anode_fermi_level(voltage),
        voltage=voltage,
        interface_z=interface_z,
        E_C_lower=E_C_lower,
        E_V_lower=E_C_lower - stack.layer_band_gap[:n_iface],
    )


def electrode_band_bottom(fermi_level: float, fermi_energy: float) -> float:
    """电极导带底在全局能量标度上的位置 [J]。

    电极中电子的"总能量" E1 = E - 该电极导带底，是 R2/R3 的 E1 与 R4/R5 里
    自由电子态密度 N_C(E) ∝ √E1 所需的能量参考。
    """
    return fermi_level - fermi_energy


def free_electron_dos(energy_above_band_bottom: np.ndarray, mass: float) -> np.ndarray:
    """自由电子态密度 N(E) = (1/2π²)·(2m/ℏ²)^{3/2}·√E [1/(J·m^3)]。

    E ≤ 0（低于电极导带底）时无态可用，返回 0。
    """
    e = np.asarray(energy_above_band_bottom, dtype=float)
    dos = np.zeros_like(e)
    ok = e > 0.0
    prefactor = (2.0 * mass / constants.hbar ** 2) ** 1.5 / (2.0 * np.pi ** 2)
    dos[ok] = prefactor * np.sqrt(e[ok])
    return dos
