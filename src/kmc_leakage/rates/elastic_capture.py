"""R2 / R3 —— 电极↔陷阱 弹性隧穿（俘获 / 逃逸）。

见 docs/rate_equations.md::

    R2 = [8·E1^1.5/(3ℏ√ED)]·(m_e/m_di)^2.5·f_C(E)·P(E)
    R3 = [8·E1^1.5/(3ℏ√ED)]·(m_e/m_di)^2.5·[1-f_C(E)]·P(E)

两者是同一隧穿过程的正反方向，占据因子互补，共用同一个 P(E) 与前置
因子——这是弹性隧穿细致平衡的自然结果。

E1 是"电极中电子总能量"，即电子能量减去**该电极导带底**的位置。全局
能量零点取阴极费米能级，电极导带底则在 (该电极费米能级 - 费米能) 处，
见 bands.electrode_band_bottom。E1 ≤ 0 表示电极侧无态可用，速率为 0。
"""

import numpy as np
from scipy import constants


def elastic_prefactor(
    E1: np.ndarray, ED: np.ndarray, m_electrode: float, m_di: float
) -> np.ndarray:
    """R2/R3 共用的前置因子 [8·E1^1.5/(3ℏ√ED)]·(m_e/m_di)^2.5 [1/s]。

    m_electrode 是电极侧电子质量，m_di 是介质的隧穿有效质量。
    """
    E1 = np.asarray(E1, dtype=float)
    ED = np.asarray(ED, dtype=float)
    if np.any(ED <= 0.0):
        raise ValueError("E_D 必须为正")
    out = np.zeros(np.broadcast(E1, ED).shape)
    available = E1 > 0.0
    e1 = np.broadcast_to(E1, out.shape)[available]
    ed = np.broadcast_to(ED, out.shape)[available]
    out[available] = (
        8.0
        * e1 ** 1.5
        / (3.0 * constants.hbar * np.sqrt(ed))
        * (m_electrode / m_di) ** 2.5
    )
    return out


def r2_elastic_capture(
    E1: np.ndarray,
    ED: np.ndarray,
    m_electrode: float,
    m_di: float,
    occupancy: np.ndarray,
    transmission: np.ndarray,
) -> np.ndarray:
    """R2: 电极→陷阱弹性俘获，见 docs/rate_equations.md。

    `occupancy` 是电极中陷阱能级处的费米分布 f_C(E)，
    `transmission` 是同一能量下电极↔陷阱路径的 WKB 透射系数 P(E)。
    """
    return elastic_prefactor(E1, ED, m_electrode, m_di) * np.asarray(occupancy) * np.asarray(
        transmission
    )


def r3_elastic_emission(
    E1: np.ndarray,
    ED: np.ndarray,
    m_electrode: float,
    m_di: float,
    occupancy: np.ndarray,
    transmission: np.ndarray,
) -> np.ndarray:
    """R3: 陷阱→电极弹性逃逸，见 docs/rate_equations.md。

    与 R2 共享前置因子和 P(E)，占据因子换成电极侧"有空位可接收"的
    概率 [1-f_C(E)]。
    """
    return elastic_prefactor(E1, ED, m_electrode, m_di) * (
        1.0 - np.asarray(occupancy)
    ) * np.asarray(transmission)
