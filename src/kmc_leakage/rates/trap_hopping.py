"""R6 / R7 —— 陷阱↔陷阱 跃迁。

见 docs/rate_equations.md::

    R6(a→b) = ν0 · exp[ -r_ab/r_{D,a} - r_ab/r_{D,b} ]
    R7(a→b) = R6(a→b) × Θ(E_a, E_b)

    Θ = exp[-(E_b-E_a)/k_BT]  若 E_b > E_a
    Θ = 1                      若 E_b ≤ E_a

R7 是经典 Miller-Abrahams (MA) 跃迁模型，只在 R6 的几何隧穿因子上乘一个
能量方向的玻尔兹曼因子，因此直接复用 R6 的计算结果。

**KMC 事件表里两者合并为一条通道**：E_b ≤ E_a 时 Θ=1 即 R7 ≡ R6，两条
并列会把下坡跃迁速率精确地算成两倍。本阶段陷阱间跃迁一律走 R7；R6 作为
独立纯函数保留并单独测试，也供 R7 内部调用。后续 R7 升级为更精确的量子
模型后，两者才需要分开计入。

**分段几何因子（ADR-010）**：异质叠层里波函数沿路径按所在层材料衰减，
几何因子推广为 `G_ab = exp[−∫_γab (α_a+α_b) ds]`，α_i(s) 用**路径所在层
的** m_di（路径传播质量，见 review F08）。单一材料叠层退回闭式
`exp[−r_ab(1/r_D,a+1/r_D,b)]`——调用方（kmc）据材料种数选闭式或分段式。
能量方向的 Θ 因子不变（仍用 F02 的多体事件能量差 ΔH），只换几何重叠因子。
"""

import numpy as np
from scipy import constants


def r6_elastic_hop(
    attempt_frequency: float, r_ab: np.ndarray, r_D_a: np.ndarray, r_D_b: np.ndarray
) -> np.ndarray:
    """R6: 陷阱→陷阱弹性跃迁，见 docs/rate_equations.md。

    公式允许两个缺陷有不同的局域化半径 r_D（高斯能级分布下本就不同），
    为后续引入不同种类缺陷预留空间。

    参数
    ----
    attempt_frequency : 尝试频率 ν0 [1/s]（与 R7/R8 共用同一配置项）
    r_ab : 两缺陷三维欧氏距离 [m]
    r_D_a, r_D_b : 两缺陷各自的局域化半径 [m]
    """
    r_ab = np.asarray(r_ab, dtype=float)
    exponent = r_ab / np.asarray(r_D_a) + r_ab / np.asarray(r_D_b)
    return attempt_frequency * np.exp(-exponent)


def segmented_overlap_factor(
    z_a: np.ndarray,
    z_b: np.ndarray,
    r_ab: np.ndarray,
    trap_energy_a: np.ndarray,
    trap_energy_b: np.ndarray,
    layer_edges: np.ndarray,
    layer_m_di: np.ndarray,
    layer_band_reference: np.ndarray,
) -> np.ndarray:
    """分段几何重叠因子 G_ab = exp[−∫_γab (α_a+α_b) ds]（ADR-010）。

    沿 a→b 的直线按所在层材料分段积分。第 i 个缺陷在位置 s（所属层 ℓ）处的
    局域衰减常数

        α_i(s) = √(2·m_di(ℓ)·[B_ℓ − (B_ℓ(i) − E_D,i)]₊) / ℏ,

    其中 `m_di(ℓ)`、`B_ℓ` 取**路径所在层**的材料值（路径传播质量，review F08），
    `trap_energy_i = B_ℓ(i) − E_D,i` 是缺陷 i 的零场陷阱能级（绝对能标）。直线
    在 z 方向从 z_a 线性变到 z_b，弧长按 r_ab 均分；界面处 α 有真实台阶，用
    精确的分段求和（不做数值积分、无求积误差）。

    同质材料（所有层同 m_di、同 B_ℓ，且各缺陷 m_di 等于该材料）时
    α_i = 1/r_D,i 常数，退回 `exp[−r_ab(1/r_D,a+1/r_D,b)]`；但因用重算的
    √(2mE)/ℏ 而非存好的 r_D，二者只在 rtol 意义下相等。**逐位退回闭式由
    kmc 的单一材料分支给出**，不走本函数。

    所有参数按有序缺陷对（P,）给出（几何量）或按层（n_layers/n_layers+1）给出。
    """
    z_a = np.asarray(z_a, dtype=float)
    z_b = np.asarray(z_b, dtype=float)
    r_ab = np.asarray(r_ab, dtype=float)
    edges = np.asarray(layer_edges, dtype=float)
    m_di = np.asarray(layer_m_di, dtype=float)
    band = np.asarray(layer_band_reference, dtype=float)
    n_layers = m_di.shape[0]

    interfaces = edges[1:-1]  # (n_int,) 内部界面 z
    delta = z_b - z_a
    safe_delta = np.where(delta == 0.0, 1.0, delta)
    # 界面在参数 u∈[0,1] 上的位置；Δz=0（同 z 平面对）时坍缩到单段
    u_cross = (interfaces[None, :] - z_a[:, None]) / safe_delta[:, None]
    u_cross = np.where((delta == 0.0)[:, None], 1.0, u_cross)
    u_cross = np.clip(u_cross, 0.0, 1.0)

    ones = np.ones((z_a.shape[0], 1))
    breaks = np.sort(np.concatenate([0.0 * ones, u_cross, ones], axis=1), axis=1)
    u_lo, u_hi = breaks[:, :-1], breaks[:, 1:]  # (P, n_int+1) 各子段
    u_mid = 0.5 * (u_lo + u_hi)
    z_mid = z_a[:, None] + delta[:, None] * u_mid
    layer_mid = np.clip(
        np.searchsorted(edges, z_mid, side="right") - 1, 0, n_layers - 1
    )
    m_mid = m_di[layer_mid]
    b_mid = band[layer_mid]

    def alpha(trap_energy):
        forbidden = np.clip(b_mid - np.asarray(trap_energy)[:, None], 0.0, None)
        return np.sqrt(2.0 * m_mid * forbidden) / constants.hbar

    seg_len = r_ab[:, None] * (u_hi - u_lo)
    integral = np.sum((alpha(trap_energy_a) + alpha(trap_energy_b)) * seg_len, axis=1)
    return np.exp(-integral)


def miller_abrahams_factor(E_a: np.ndarray, E_b: np.ndarray, temperature: float) -> np.ndarray:
    """MA 能量因子 Θ(E_a, E_b)：上坡付热激活代价，下坡或等能不受惩罚。"""
    delta = np.asarray(E_b, dtype=float) - np.asarray(E_a, dtype=float)
    return np.exp(-np.clip(delta, 0.0, None) / (constants.Boltzmann * temperature))


def r7_inelastic_hop(
    attempt_frequency: float,
    r_ab: np.ndarray,
    r_D_a: np.ndarray,
    r_D_b: np.ndarray,
    E_a: np.ndarray,
    E_b: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """R7: 陷阱→陷阱非弹性跃迁（Miller-Abrahams），见 docs/rate_equations.md。

    E_a、E_b 是两个陷阱上电子的**总能量**（含局域静电势 -eφ），所以
    电场造成的能量倾斜通过这一项驱动净输运。
    """
    return r6_elastic_hop(attempt_frequency, r_ab, r_D_a, r_D_b) * miller_abrahams_factor(
        E_a, E_b, temperature
    )
