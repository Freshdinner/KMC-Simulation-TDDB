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

    积分只在经典禁区（U > E）上进行；E 高于势垒最高点时 P = 1。

    参数
    ----
    z : 隧穿路径坐标 [m]，最后一维是积分变量，须与 `barrier` 可广播
    barrier : 势垒剖面 U(z) [J]，最后一维与 z 对应
    energy : 电子能量 E [J]，形状须与 `barrier` 去掉最后一维后可广播
        （函数内部自动补上末维）
    m_di : 隧穿有效质量 [kg]，标量或沿积分轴（最后一维）逐样点给出的数组
        （分段叠层的路径传播质量，ADR-012），须与 `barrier` 可广播

    返回形状为 (z, barrier, energy[..., None]) 广播后去掉最后一维。
    路径方向（z 递增或递减）不影响结果。
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


def barrier_maximum(barrier: np.ndarray) -> float:
    """势垒沿隧穿路径的最高点能量 E_barrier,max [J]。

    R1 的统计分解以它为阈值：低于它计入"直接/FN隧穿"，高于它计入
    "热电子发射"。
    """
    return float(np.max(barrier))
