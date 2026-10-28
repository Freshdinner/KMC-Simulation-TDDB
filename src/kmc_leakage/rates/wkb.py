"""WKB 透射系数。

势垒剖面按 docs 的约定处理：**不含镜像力势垒降低**，势垒高度取零场值
E_B，界面处能带保持直线，积分限就是直线/分段线性势垒与电子能量的交点
（由 max(U-E, 0) 自动给出）。所有 WKB 积分用隧穿质量 m_di。

**分段 m_di（ADR-012）**：叠层里隧穿质量沿路径按所在层取值，因此 `m_di`
既可是标量（单一材料），也可是与积分轴（最后一维）对齐、逐样点给出的
数组。异质叠层的界面台阶由调用方在 `barrier` 里以界面两侧双值表示，不在
这里插成斜坡。
"""

import numpy as np
from scipy import constants
from scipy.integrate import trapezoid


def wkb_transmission(
    z: np.ndarray, barrier: np.ndarray, energy: np.ndarray, m_di: np.ndarray
) -> np.ndarray:
    """WKB 透射系数 P(E) = exp[-(2/ℏ)·∫√(2·m_di·(U(z)-E)) dz]。
    """
    z = np.asarray(z, dtype=float)
    barrier = np.asarray(barrier, dtype=float)
    energy = np.asarray(energy, dtype=float)[..., None]
    m_di = np.asarray(m_di, dtype=float)
    forbidden = np.clip(barrier - energy, 0.0, None)
    integrand = np.sqrt(2.0 * m_di * forbidden)
    z_b, integrand = np.broadcast_arrays(z, integrand)
    exponent = 2.0 / constants.hbar * np.abs(trapezoid(integrand, x=z_b, axis=-1))
    return np.exp(-exponent)


def wkb_path_weights(z: np.ndarray, m_di: np.ndarray) -> np.ndarray:
    """路径各样点的 WKB 积分权重 w_k：梯形公式的权重乘 √(2·m_di,k)，形状同 `z`（`m_di` 广播到它）。

    `wkb_transmission` 的梯形值 ∫√(2·m_di·(U − E)) dz 就是 Σ_k w_k·√(U_k − E)₊。路径与隧穿质量在几何上固定时
    （KMC 的电极 → 缺陷路径），权重只算一次，之后每次求透射只做开方与加权求和（`wkb_transmission_weighted`）。
    """
    z = np.asarray(z, dtype=float)
    step = 0.5 * np.diff(z, axis=-1)
    weights = np.zeros(z.shape)
    weights[..., :-1] += step
    weights[..., 1:] += step
    return weights * np.sqrt(2.0 * np.broadcast_to(np.asarray(m_di, dtype=float), z.shape))


def wkb_transmission_weighted(weights: np.ndarray, barrier: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """WKB 透射系数 exp[−(2/ℏ)·|Σ_k w_k·√(U_k − E)₊|]，权重 `weights` 取 `wkb_path_weights`，与 `wkb_transmission` 是
    同一个梯形值。`energy` 加一个路径维后与 `barrier`、`weights` 广播。"""
    forbidden = barrier - np.asarray(energy, dtype=float)[..., None]
    np.maximum(forbidden, 0.0, out=forbidden)
    np.sqrt(forbidden, out=forbidden)
    forbidden *= weights
    return np.exp(-2.0 / constants.hbar * np.abs(forbidden.sum(axis=-1)))


def barrier_maximum(barrier: np.ndarray) -> float:
    """势垒沿隧穿路径的最高点能量 E_barrier,max [J]。

    R1 的统计分解以它为阈值：低于它计入"直接/FN隧穿"，高于它计入
    "热电子发射"。
    """
    return float(np.max(barrier))
