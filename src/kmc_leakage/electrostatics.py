"""泊松方程求解（FFT + Thomas）与粒子/网格插值。

方法见 docs/physics_model.md 的"泊松方程数值方法"一节：
xy 方向周期边界 → FFT；每个傅里叶模式在 z 方向退化为三对角方程组，用
Thomas 算法直接求解（无迭代、无收敛判据）。三对角系数只依赖网格与各层 κ，在
`__init__` 里预先分解、每步复用，每步只更新右端项。

**变 κ 泊松（ADR-011）**：介电常数按单元取值——单元（相邻两节点之间）落在
哪一层就取该层的静态 κ（不是 ε_opt——那是 R8 专用的）。内部界面对齐到网格
节点（层厚是 hz 的整数倍），单元不跨界面、κ 单值。变系数离散式（review §4.2）::

    κ_{k-½}·φ̂_{k-1} - (κ_{k-½}+κ_{k+½}+hz²·κ̄_k·K⊥²)·φ̂_k + κ_{k+½}·φ̂_{k+1}
        = -(hz²/ε0)·ρ̂_k,     κ̄_k = (κ_{k-½}+κ_{k+½})/2

法向连接直接用单元 κ，横向项用相邻两单元 κ 的算术平均 κ̄_k。单层退化时各单元
κ 相同，退回常系数解。

网格：x/y 方向 nx/ny 个周期格点（步长 Lx/nx），z 方向 nz+1 个节点
（步长 Lz/nz），两端节点是 Dirichlet 边界（电极等势面）。
"""

from typing import Tuple

import numpy as np
from scipy import constants

from .config import DeviceConfig, GridConfig


class PoissonSolver:
    """给定器件与网格的泊松求解器。三对角分解在构造时完成。"""

    def __init__(self, device: DeviceConfig, grid: GridConfig) -> None:
        self.device = device
        self.grid = grid
        self.nx, self.ny, self.nz = grid.nx, grid.ny, grid.nz
        self.hx = device.Lx / self.nx
        self.hy = device.Ly / self.ny
        self.hz = device.Lz / self.nz
        self.z = np.arange(self.nz + 1) * self.hz

        n_int = self.nz - 1
        if n_int < 1:
            raise ValueError("grid.nz 至少为 2")

        # 每层厚度必须是 hz 的整数倍，内部界面才能恰好落在网格节点上（ADR-011）。
        # 单层没有内部界面，这一条自动满足。
        self._check_interface_alignment()

        # κ 按单元取值：单元 k（节点 k 与 k+1 之间）的中心在 (k+0.5)·hz，取该处
        # 所属层的静态 κ。界面对齐后单元不跨界面、κ 单值。κ_{k+½} = kappa_cells[k]。
        cell_centers = (np.arange(self.nz) + 0.5) * self.hz
        self._kappa_cells = np.asarray(device.stack.kappa_at(cell_centers), dtype=float)  # (nz,)

        # 离散拉普拉斯算符在 xy 方向的本征值（用离散值而非连续 k²，
        # 这样求解器对离散三维（变 κ）算符是精确的）
        kx = 2.0 * (1.0 - np.cos(2.0 * np.pi * np.arange(self.nx) / self.nx)) / self.hx ** 2
        ky = 2.0 * (1.0 - np.cos(2.0 * np.pi * np.arange(self.ny // 2 + 1) / self.ny)) / self.hy ** 2
        self._K2 = kx[:, None] + ky[None, :]  # (nx, ny//2+1)

        # 变 κ 的三对角系数（内点 k=1..nz-1，内部索引 m=k-1）：
        #   下对角 a[m] = κ_{k-½} = kappa_cells[m]
        #   上对角 c[m] = κ_{k+½} = kappa_cells[m+1]
        #   主对角 b[m] = -(κ_{k-½} + κ_{k+½} + hz²·κ̄_k·K⊥²), κ̄_k = (κ_{k-½}+κ_{k+½})/2
        # a、c 与傅里叶模式无关（只是 κ 值）；b 通过 κ̄·K⊥² 依赖模式。系数只依赖
        # 网格与各层 κ、与电荷分布无关，预分解后每步复用。
        a = self._kappa_cells[:n_int]         # a[m] = κ_{(m+1)-½}
        c = self._kappa_cells[1 : n_int + 1]  # c[m] = κ_{(m+1)+½}
        kbar = 0.5 * (a + c)                  # 每个内点一个（模式无关）
        self._a = a                           # 下对角，前向消元右端项要用

        # 一般（变系数）Thomas 要同时存 c'（回代用）与分母倒数（前向用）：原先
        # "逆分母兼作 c'" 只在上对角恒为 1 时成立，变 κ 后上对角是 κ_{k+½}、失效。
        self._cprime = np.empty((n_int,) + self._K2.shape)
        self._inv_denom = np.empty((n_int,) + self._K2.shape)
        b0 = -(a[0] + c[0] + self.hz ** 2 * kbar[0] * self._K2)
        self._inv_denom[0] = 1.0 / b0
        self._cprime[0] = c[0] * self._inv_denom[0]
        for m in range(1, n_int):
            b = -(a[m] + c[m] + self.hz ** 2 * kbar[m] * self._K2)
            self._inv_denom[m] = 1.0 / (b - a[m] * self._cprime[m - 1])
            self._cprime[m] = c[m] * self._inv_denom[m]

    def _check_interface_alignment(self) -> None:
        """校验每个内部界面恰好落在网格节点上（层厚是 hz 的整数倍，ADR-011）。"""
        interfaces = self.device.stack.interface_positions
        if interfaces.size == 0:
            return
        nodes = interfaces / self.hz
        misaligned = np.abs(nodes - np.round(nodes)) > 1e-9
        if np.any(misaligned):
            bad_nm = (interfaces[misaligned] / constants.nano).round(6).tolist()
            raise ValueError(
                "界面未对齐到网格节点：层厚必须是 hz = {:.4g} nm 的整数倍（ADR-011），"
                "界面 z = {} nm 落在节点之间；请调整 grid.nz 或层厚".format(
                    self.hz / constants.nano, bad_nm
                )
            )

    # ------------------------------------------------------------------
    # 粒子 <-> 网格
    # ------------------------------------------------------------------
    def _cic_weights(self, positions: np.ndarray):
        """三线性（Cloud-In-Cell）权重与索引。

        xy 周期回绕；z 方向索引夹在 [0, nz-1]，落到 Dirichlet 边界节点上的
        权重在求解时自然被丢弃（相当于被电极吸收）。
        """
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        fx = positions[:, 0] / self.hx
        fy = positions[:, 1] / self.hy
        fz = positions[:, 2] / self.hz
        i0 = np.floor(fx).astype(int)
        j0 = np.floor(fy).astype(int)
        k0 = np.clip(np.floor(fz).astype(int), 0, self.nz - 1)
        wx, wy, wz = fx - i0, fy - j0, fz - k0
        i0 = np.mod(i0, self.nx)
        j0 = np.mod(j0, self.ny)
        i1 = np.mod(i0 + 1, self.nx)
        j1 = np.mod(j0 + 1, self.ny)
        k1 = k0 + 1
        return (i0, i1, j0, j1, k0, k1, wx, wy, wz)

    def deposit(self, positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
        """把点电荷按 CIC 分配到网格，返回电荷密度 ρ [C/m^3]。"""
        rho = np.zeros(self.grid.shape)
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        if positions.size == 0:
            return rho
        charges = np.broadcast_to(np.asarray(charges, dtype=float), (positions.shape[0],))
        i0, i1, j0, j1, k0, k1, wx, wy, wz = self._cic_weights(positions)
        cell_volume = self.hx * self.hy * self.hz
        q = charges / cell_volume
        for ii, wi in ((i0, 1.0 - wx), (i1, wx)):
            for jj, wj in ((j0, 1.0 - wy), (j1, wy)):
                for kk, wk in ((k0, 1.0 - wz), (k1, wz)):
                    np.add.at(rho, (ii, jj, kk), q * wi * wj * wk)
        return rho

    def interpolate(self, field: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """用与 deposit 同阶的三线性权重把网格量读回粒子位置。"""
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        if positions.size == 0:
            return np.zeros(0)
        i0, i1, j0, j1, k0, k1, wx, wy, wz = self._cic_weights(positions)
        out = np.zeros(positions.shape[0])
        for ii, wi in ((i0, 1.0 - wx), (i1, wx)):
            for jj, wj in ((j0, 1.0 - wy), (j1, wy)):
                for kk, wk in ((k0, 1.0 - wz), (k1, wz)):
                    out += field[ii, jj, kk] * wi * wj * wk
        return out

    # ------------------------------------------------------------------
    # 求解
    # ------------------------------------------------------------------
    def solve(self, rho: np.ndarray, phi_top: float) -> np.ndarray:
        """解 ∇²φ = -ρ/(κ·ε0)，φ(z=0)=0、φ(z=Lz)=phi_top。

        返回节点电势 φ [V]，形状 (nx, ny, nz+1)。
        """
        rho = np.asarray(rho, dtype=float)
        if rho.shape != self.grid.shape:
            raise ValueError("ρ 的形状 {} 与网格 {} 不符".format(rho.shape, self.grid.shape))

        rho_hat = np.fft.rfft2(rho, axes=(0, 1))
        # 变 κ 后 κ 进算符、不再出现在右端项：RHS = -hz²·ρ̂/ε0（不除 κ）
        scale = self.hz ** 2 / constants.epsilon_0
        d = np.moveaxis(-scale * rho_hat[:, :, 1 : self.nz], 2, 0).copy()

        # 顶边界（常数 phi_top）只有 (0,0) 模式非零，其上对角系数是最末单元的
        # κ_{nz-½} = kappa_cells[-1]（常系数下为 1，此处退回旧写法）
        top_hat = np.zeros(self._K2.shape, dtype=complex)
        top_hat[0, 0] = phi_top * self.nx * self.ny
        d[-1] -= self._kappa_cells[-1] * top_hat

        # 一般（变系数）Thomas：前向用下对角 a 与分母倒数，回代用 c'
        a, inv, cprime = self._a, self._inv_denom, self._cprime
        n_int = d.shape[0]
        dp = np.empty_like(d)
        dp[0] = d[0] * inv[0]
        for m in range(1, n_int):
            dp[m] = (d[m] - a[m] * dp[m - 1]) * inv[m]
        x = np.empty_like(d)
        x[-1] = dp[-1]
        for m in range(n_int - 2, -1, -1):
            x[m] = dp[m] - cprime[m] * x[m + 1]

        phi_hat = np.zeros((self.nx, self.ny // 2 + 1, self.nz + 1), dtype=complex)
        phi_hat[:, :, 1 : self.nz] = np.moveaxis(x, 0, 2)
        phi_hat[:, :, self.nz] = top_hat
        phi = np.fft.irfft2(phi_hat, s=(self.nx, self.ny), axes=(0, 1))
        phi[:, :, 0] = 0.0
        phi[:, :, self.nz] = phi_top
        return phi

    def electric_field(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """由电势求电场 E = -∇φ [V/m]（xy 中心差分且周期，z 用 np.gradient）。"""
        ex = -(np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * self.hx)
        ey = -(np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * self.hy)
        ez = -np.gradient(phi, self.hz, axis=2)
        return ex, ey, ez

    def electric_field_z_sided(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """z 方向的单侧差分场 (E_z 下侧, E_z 上侧) [V/m]，逐节点给出。

        界面上 E_z 不连续（两侧 κ 比≈4 倍），`np.gradient` 的中心差分给的是两侧
        平均，对界面缺陷是错的（ADR-017 §2 / review F08）。这里给出每个节点的
        单侧斜率：`ez_lower[k]` 用节点 k-1→k（下侧单元），`ez_upper[k]` 用节点
        k→k+1（上侧单元）。界面落在节点上，取哪一侧由缺陷的读场侧决定。两端节点
        没有对侧，退回唯一存在的那一侧（界面缺陷不会落在电极节点上）。
        """
        phi = np.asarray(phi, dtype=float)
        forward = -(phi[..., 1:] - phi[..., :-1]) / self.hz  # 节点 0..nz-1 的前向斜率
        ez_upper = np.empty_like(phi)
        ez_lower = np.empty_like(phi)
        ez_upper[..., :-1] = forward       # 上侧：单元 k（节点 k→k+1）
        ez_upper[..., -1] = forward[..., -1]
        ez_lower[..., 1:] = forward        # 下侧：单元 k-1（节点 k-1→k）
        ez_lower[..., 0] = forward[..., 0]
        return ez_lower, ez_upper

    def field_vectors_at(
        self, phi: np.ndarray, positions: np.ndarray, field_side: np.ndarray
    ) -> np.ndarray:
        """缺陷处的电场矢量 (3, N) [V/m]，按读场侧统一取值（F08 共用接口）。

        所有吃场的机制（R4/R5 前置因子、R8）读同一份场。体缺陷（`field_side==0`）
        三分量都用常规中心差分插值；界面缺陷（±1）的 E_z 换成单侧差分——+1 取上侧、
        -1 取下侧（xy 两分量在界面处连续，仍用中心差分）。
        """
        ex, ey, ez = self.electric_field(phi)
        fx = self.interpolate(ex, positions)
        fy = self.interpolate(ey, positions)
        fz = self.interpolate(ez, positions)
        field_side = np.asarray(field_side)
        if np.any(field_side != 0):
            ez_lower, ez_upper = self.electric_field_z_sided(phi)
            fz_lower = self.interpolate(ez_lower, positions)
            fz_upper = self.interpolate(ez_upper, positions)
            fz = np.where(field_side > 0, fz_upper, np.where(field_side < 0, fz_lower, fz))
        return np.stack([fx, fy, fz])

    def field_readout_sources(self, position: np.ndarray) -> np.ndarray:
        """体缺陷位置 `position` 处读场的伴随源 s_c，形状 (3, nx, ny, nz+1)。

        `field_vectors_at` 对体缺陷（field_side = 0）读出的场是网格电势的线性泛函：
        E_c(r) = −⟨s_c, φ⟩（对全部节点求和），s_c = D_cᵀ·w_r，w_r 是 r 处的 CIC 权重，
        D_c 是 `electric_field` 用的差分算子（xy 周期中心差分，z 为 np.gradient 的内点
        中心差分、两端单侧差分）。泊松算子对称，所以解 ψ_c = solve(s_c, 0) 之后，任一
        电荷密度 ρ 在 r 处产生的场分量就是 −⟨ψ_c, ρ⟩：三次求解给出全部已有电荷在 r 处
        的场（TDDB 增量静电耦合的伴随求解，PROJECT_CONTEXT_TDDB.md §6.4）。改
        `electric_field` 的差分格式时这里要同步改。
        """
        w = self.deposit(position, self.hx * self.hy * self.hz)  # deposit 除以单元体积，这里得到权重本身
        sx = (np.roll(w, 1, axis=0) - np.roll(w, -1, axis=0)) / (2.0 * self.hx)
        sy = (np.roll(w, 1, axis=1) - np.roll(w, -1, axis=1)) / (2.0 * self.hy)
        sz = np.zeros_like(w)
        # np.gradient 的转置：内点行 k 给第 k±1 列 ±1/(2hz)，两端行是单侧差分 ∓1/hz
        sz[..., 2:] += w[..., 1:-1] / (2.0 * self.hz)
        sz[..., :-2] -= w[..., 1:-1] / (2.0 * self.hz)
        sz[..., 0] -= w[..., 0] / self.hz
        sz[..., 1] += w[..., 0] / self.hz
        sz[..., -2] -= w[..., -1] / self.hz
        sz[..., -1] += w[..., -1] / self.hz
        return np.stack([sx, sy, sz])

    @staticmethod
    def lateral_average(field: np.ndarray) -> np.ndarray:
        """对 xy 求平均，得到沿 z 的一维剖面。"""
        return np.asarray(field).mean(axis=(0, 1))
