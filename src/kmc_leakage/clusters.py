"""强耦合陷阱簇：识别与簇内条件稳态（条件通量估计量的簇级推广，ADR-022）。

单陷阱条件通量（`kmc.KMCSimulation._conditional_flux`）把每个陷阱自身的占据
解析平均掉，但陷阱间跃迁的供给项用的是 KMC 采到的邻居占据。两个陷阱之间的
跃迁速率与它们各自的电极交换速率相当时，这一项取决于邻居的稀有填充事件是否
恰好落在测量窗口里：落进来就高估（I-V 曲线上的尖峰），没落进来就漏算。

这里把这样的陷阱连成"强耦合簇"，簇内 2^k 个占据组态直接解主方程稳态，簇外
陷阱的占据作为给定的环境。单陷阱条件通量就是 k=1 的特例。

全部是纯函数，只处理数组。环境占据可以是 KMC 采样的 0/1，也可以是平均场的
分数占据，所以同一套函数也能给 TDDB 的平均场内层求解器用。
"""

from typing import List, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def enumerate_occupations(n: int) -> np.ndarray:
    """全部 2^n 个占据组态，形状 (2^n, n) 的 bool 数组（bit i = 陷阱 i 是否占据）。"""
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)


def coupling_strength(
    pair_a: np.ndarray, pair_b: np.ndarray, hop: np.ndarray, gamma: np.ndarray
) -> np.ndarray:
    """有序对 a→b 的耦合强度 hop[a→b] / min(Γ_a, Γ_b)。

    Γ_i 是陷阱 i 与电极交换电子的总速率（俘获 + 逃逸 + PF）。强度 ≳ 1 表示这条
    跃迁与较慢一端自身的弛豫一样快，那一端的占据就不能当成独立采样的环境。
    较慢一端 Γ=0 时，只要跃迁速率非零就记为无穷强。
    """
    denom = np.minimum(gamma[pair_a], gamma[pair_b])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = hop / denom
    return np.where(denom > 0.0, ratio, np.where(hop > 0.0, np.inf, 0.0))


def coupled_clusters(
    n: int,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    strength: np.ndarray,
    threshold: float,
    max_size: int,
) -> List[np.ndarray]:
    """按耦合强度把缺陷连成强耦合簇，返回大小 ≥2 的簇（簇内索引升序）。

    任一方向强度 ≥ `threshold` 的缺陷对连边，取无向图的连通分量。分量超过
    `max_size` 时，在分量内部把阈值逐级 ×10 重新切分（保留最强的耦合），直到
    每块都不超过上限；只剩无穷强度的边仍切不开时，这块按单陷阱处理。
    """
    finite = strength[np.isfinite(strength)]
    max_finite = float(finite.max()) if finite.size else 0.0
    clusters: List[np.ndarray] = []
    stack = [(np.arange(n), threshold)]
    while stack:
        nodes, level = stack.pop()
        local = np.full(n, -1)
        local[nodes] = np.arange(nodes.size)
        keep = (local[pair_a] >= 0) & (local[pair_b] >= 0) & (strength >= level)
        graph = coo_matrix(
            (np.ones(int(keep.sum())), (local[pair_a[keep]], local[pair_b[keep]])),
            shape=(nodes.size, nodes.size),
        )
        _, labels = connected_components(graph, directed=False)
        for label in np.unique(labels):
            members = np.sort(nodes[labels == label])
            if members.size < 2:
                continue
            if members.size <= max_size:
                clusters.append(members)
            elif level <= max_finite:
                stack.append((members, level * 10.0))
    clusters.sort(key=lambda members: int(members[0]))
    return clusters


def _stationary_least_squares(Q: np.ndarray) -> np.ndarray:
    """可约链的退路：最小二乘解 πQ=0、Σπ=1，负的舍入残差截为 0 后归一。"""
    n = Q.shape[0]
    matrix = np.vstack([Q.T, np.ones((1, n))])
    rhs = np.zeros(n + 1)
    rhs[-1] = 1.0
    pi = np.clip(np.linalg.lstsq(matrix, rhs, rcond=None)[0], 0.0, None)
    return pi / pi.sum()


def stationary_distribution_gth(Q: np.ndarray) -> np.ndarray:
    """生成矩阵 Q 的稳态分布 π（πQ=0，Σπ=1），用 GTH 状态约化法求。

    Grassmann–Taksar–Heyman 算法只做加法、乘法和正数除法，没有减法，所以 π 的
    每个分量都有 O(n·ε) 的相对精度，与速率跨多少个数量级无关。簇里的电极交换
    速率可达 1e14 s⁻¹，净电流却可能只有 1 s⁻¹ 量级，直接解线性方程组给不出这种
    精度。只用 Q 的非对角元。链可约（某个组态到不了编号更小的组态）时退回
    最小二乘解。
    """
    return stationary_distributions_gth([Q])[0]


def stationary_distributions_gth(generators: List[np.ndarray]) -> List[np.ndarray]:
    """一组生成矩阵各自的稳态分布，逐个与 `stationary_distribution_gth` 相同（GTH 状态约化法，可约的链退回最小二乘解）。

    同样大小的矩阵叠成 (B, n, n) 一起约化，每一步对整批做同一个运算，省掉逐个矩阵逐个状态的循环开销。
    """
    distributions: List[np.ndarray] = [np.zeros(0)] * len(generators)
    sizes = np.array([np.shape(Q)[0] for Q in generators], dtype=int)
    for n in np.unique(sizes):
        group = np.flatnonzero(sizes == n)
        a = np.array([generators[i] for i in group], dtype=float)
        diagonal = np.arange(n)
        a[:, diagonal, diagonal] = 0.0
        reducible = np.zeros(group.size, dtype=bool)
        for k in range(n - 1, 0, -1):
            s = a[:, k, :k].sum(axis=1)
            reducible |= ~(s > 0.0)
            a[:, :k, k] /= np.where(s > 0.0, s, 1.0)[:, None]
            a[:, :k, :k] += a[:, :k, k, None] * a[:, k, None, :k]
        pi = np.zeros((group.size, n))
        pi[:, 0] = 1.0
        for k in range(1, n):
            pi[:, k] = np.einsum("bi,bi->b", pi[:, :k], a[:, :k, k])
        pi /= pi.sum(axis=1, keepdims=True)
        for b, i in enumerate(group):
            distributions[i] = (
                _stationary_least_squares(np.asarray(generators[i], dtype=float)) if reducible[b] else pi[b]
            )
    return distributions


def cluster_generator(
    states: np.ndarray,
    capture: np.ndarray,
    escape: np.ndarray,
    intra_a: np.ndarray,
    intra_b: np.ndarray,
    intra_rate: np.ndarray,
) -> np.ndarray:
    """簇内占据组态的连续时间马氏链生成矩阵 Q（对角元 = −行和）。

    `states` 形状 (S, k)，第 s 行按 `enumerate_occupations(k)` 的位序给出组态 s。
    `capture[s, j]`/`escape[s, j]` 是组态 s 下成员 j 的总俘获/总逃逸速率（电极
    各通道加上与簇外陷阱的跃迁）；`intra_rate[s, p]` 是组态 s 下簇内有序对
    `intra_a[p] → intra_b[p]`（成员局部编号）的跃迁速率。事件口径与 KMC 事件表
    一致：俘获要求目标空、逃逸要求已占据、跃迁要求源占据且目标空。
    """
    n_states, k = states.shape
    index = np.arange(n_states)
    Q = np.zeros((n_states, n_states))
    for j in range(k):
        Q[index, index ^ (1 << j)] += np.where(states[:, j], escape[:, j], capture[:, j])
    for p in range(intra_a.size):
        i, j = int(intra_a[p]), int(intra_b[p])
        allowed = states[:, i] & ~states[:, j]
        source = index[allowed]
        Q[source, source - (1 << i) + (1 << j)] += intra_rate[allowed, p]
    Q[index, index] = -Q.sum(axis=1)
    return Q


def cluster_steady_state(
    states: np.ndarray,
    capture_positive: np.ndarray,
    capture_other: np.ndarray,
    escape_positive: np.ndarray,
    pf: np.ndarray,
    escape_other: np.ndarray,
    intra_a: np.ndarray,
    intra_b: np.ndarray,
    intra_rate: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """簇内条件稳态下各成员的正极净通量 (TAT, PF) 与占据期望，通量单位 1/s。

    输入均为 (S, k)（`intra_rate` 为 (S, P)）：`capture_positive` 是正极俘获
    R2A+R4A，`escape_positive` 是向正极的 TAT 逃逸 R3A+R5A，`pf` 是 R8_eff，
    `capture_other`/`escape_other` 是其余通道（负极侧 + 与簇外陷阱的跃迁）。

    成员 i 的正极净通量有两种等价写法：正极侧 Σπ[n_i(E_a+P) − (1−n_i)C_a]，
    或由稳态 d⟨n_i⟩/dt = 0 换到其余通道（负极侧、簇外跃迁、簇内跃迁的净流入）。
    两者都是两个大数之差；逐成员取两侧总量较小的一种，舍入误差就不会被贴着
    电极、交换速率极大的成员放大。
    """
    capture = capture_positive + capture_other
    escape = escape_positive + pf + escape_other
    Q = cluster_generator(states, capture, escape, intra_a, intra_b, intra_rate)
    pi = stationary_distribution_gth(Q)
    return cluster_flux(
        pi, states, capture_positive, capture_other, escape_positive, pf, escape_other, intra_a, intra_b, intra_rate
    )


def cluster_flux(
    pi: np.ndarray,
    states: np.ndarray,
    capture_positive: np.ndarray,
    capture_other: np.ndarray,
    escape_positive: np.ndarray,
    pf: np.ndarray,
    escape_other: np.ndarray,
    intra_a: np.ndarray,
    intra_b: np.ndarray,
    intra_rate: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """簇内稳态分布 `pi` 下各成员的正极净通量 (TAT, PF) 与占据期望（`cluster_steady_state` 的后半段）。

    参数口径同 `cluster_steady_state`；`pi` 是同一组速率的生成矩阵的稳态分布。调用方已经
    求出 π、还要拿它做别的记账时（TDDB 平均场内层求解器）直接用这一步。
    """
    occupied = pi[:, None] * states
    empty = pi[:, None] * ~states
    pf_flux = (occupied * pf).sum(axis=0)

    positive_out = (occupied * (escape_positive + pf)).sum(axis=0)
    positive_in = (empty * capture_positive).sum(axis=0)
    other_in = (empty * capture_other).sum(axis=0)
    other_out = (occupied * escape_other).sum(axis=0)
    for p in range(intra_a.size):
        i, j = int(intra_a[p]), int(intra_b[p])
        flow = float(np.sum(pi * (states[:, i] & ~states[:, j]) * intra_rate[:, p]))
        other_out[i] += flow
        other_in[j] += flow

    positive_side = positive_out + positive_in <= other_in + other_out
    net = np.where(positive_side, positive_out - positive_in, other_in - other_out)
    return net - pf_flux, pf_flux, occupied.sum(axis=0)
