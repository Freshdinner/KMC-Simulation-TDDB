"""小状态空间主方程稳态解——条件通量估计量的独立算法基准（ADR-021 F04）。

条件通量估计量（`kmc._conditional_flux`）把每个陷阱的占据用单陷阱条件稳态
p̂=C/(C+E) 代入净通量，是"局部稳态近似"，在耦合跃迁网络里不自动等于真实稳态
下的期望通量（review F04 给出 25% 反例）。KMC 现在对强耦合簇改用簇内条件稳态
（ADR-022，`kmc._trap_flux`），孤立簇上与这里的精确解一致。这里对**可处理规模**的小体系（N 个
陷阱 → 2^N 个组态）枚举完整状态空间、组装连续时间马氏链的生成矩阵、直接解
稳态分布 π，再算真实稳态正极通量，作为独立基准与条件通量估计量对照。

生成矩阵按 KMC 事件表同口径构造：单陷阱翻转（俘获/逃逸/PF）与两陷阱交换
（跃迁），每个组态的速率直接调 `KMCSimulation._rates`，因此这是 KMC 采样的
那条马氏链的精确主方程，不是另写一套物理。仅用于验证，不进生产逐点（2^N
随 N 指数增长）。
"""

from typing import Tuple

import numpy as np
from scipy import constants

from .clusters import enumerate_occupations
from .kmc import (
    _B_R2A,
    _B_R3A,
    _B_R4A,
    _B_R5A,
    _B_R8,
    _N_BLOCKS,
    _N_CAPTURE_BLOCKS,
    KMCSimulation,
)

# 正极侧的逃逸/发射块（对正极记 +1 电子）与俘获块（记 -1 电子）
_POSITIVE_ESCAPE_BLOCKS = (_B_R3A, _B_R5A, _B_R8)
_POSITIVE_CAPTURE_BLOCKS = (_B_R2A, _B_R4A)


def transition_generator(sim: KMCSimulation) -> Tuple[np.ndarray, np.ndarray]:
    """组装连续时间马氏链的生成矩阵 Q（与 KMC 事件表同口径）。

    返回 (occupations, Q)，Q[s, s'] = 组态 s → s' 的速率（s≠s'），对角元 = −行和。
    事件与 `KMCSimulation.run` 一致：俘获块（前 _N_CAPTURE_BLOCKS 块）要求目标陷阱
    为空、置位；逃逸/发射块要求已占据、清位；跃迁要求源占据且目标空、清源置目标。
    """
    n = sim.n_defects
    occs = enumerate_occupations(n)
    n_states = occs.shape[0]
    pairs = sim._pairs
    Q = np.zeros((n_states, n_states))

    for s in range(n_states):
        occ = occs[s]
        blocks, hop = sim._rates(occ)
        # 单陷阱事件
        for i in range(n):
            if not occ[i]:
                target = s | (1 << i)  # 置位
                Q[s, target] += blocks[:_N_CAPTURE_BLOCKS, i].sum()
            else:
                target = s & ~(1 << i)  # 清位
                Q[s, target] += blocks[_N_CAPTURE_BLOCKS:_N_BLOCKS, i].sum()
        # 跃迁事件（a→b）：源占据、目标空
        for k in range(len(pairs)):
            a, b = int(pairs.a[k]), int(pairs.b[k])
            if occ[a] and not occ[b]:
                target = (s & ~(1 << a)) | (1 << b)
                Q[s, target] += hop[k]

    Q[np.diag_indices(n_states)] = -Q.sum(axis=1)
    return occs, Q


def stationary_distribution(Q: np.ndarray) -> np.ndarray:
    """解生成矩阵 Q 的稳态分布 π（πQ=0，Σπ=1）。

    把 Qᵀπ=0 的最后一行换成归一化 Σπ=1 直接求解（review F04 附录同法）。
    """
    n_states = Q.shape[0]
    matrix = Q.T.copy()
    matrix[-1, :] = 1.0
    rhs = np.zeros(n_states)
    rhs[-1] = 1.0
    return np.linalg.solve(matrix, rhs)


def _positive_electron_rate(blocks: np.ndarray, occ: np.ndarray) -> float:
    """给定组态下真实的正极净电子速率 Σ_i[占据·逃逸 − 空·俘获]（1/s）。

    这是"实际占据下的瞬时通量"，其稳态期望 = 真实稳态电流/q（不含 R1）。
    """
    occf = occ.astype(float)
    escape = sum(blocks[b] for b in _POSITIVE_ESCAPE_BLOCKS)
    capture = sum(blocks[b] for b in _POSITIVE_CAPTURE_BLOCKS)
    return float(np.sum(occf * escape - (1.0 - occf) * capture))


def steady_state_flux(sim: KMCSimulation) -> dict:
    """小体系的真实稳态正极通量与条件通量估计量的期望，作独立对照。

    返回 dict：
    - `exact_flux` [1/s]：主方程稳态下真实正极净电子速率的期望（= 真实 I_trap/q）
    - `conditional_flux` [1/s]：KMC 所用估计量（单陷阱条件通量，强耦合簇成员换成
      簇内条件稳态，`KMCSimulation._trap_flux`）在同一 π 下的期望（KMC `current` 的极限）
    - `exact_current` / `conditional_current` [A]：上二者 ×q（陷阱相关部分，不含 R1）
    - `stationary`：稳态分布 π

    exact 与 conditional 一致 ⟺ 局部稳态近似在该体系成立；不一致即量化了偏差。
    """
    occs, Q = transition_generator(sim)
    pi = stationary_distribution(Q)

    exact = np.empty(occs.shape[0])
    conditional = np.empty(occs.shape[0])
    for s in range(occs.shape[0]):
        occ = occs[s]
        blocks, hop = sim._rates(occ)
        exact[s] = _positive_electron_rate(blocks, occ)
        tat, pf, _ = sim._trap_flux(occ, blocks, hop)
        conditional[s] = float((tat + pf).sum())

    exact_flux = float(pi @ exact)
    conditional_flux = float(pi @ conditional)
    q = constants.elementary_charge
    return {
        "exact_flux": exact_flux,
        "conditional_flux": conditional_flux,
        "exact_current": q * exact_flux,
        "conditional_current": q * conditional_flux,
        "stationary": pi,
    }
