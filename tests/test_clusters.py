"""强耦合簇（ADR-022）：分簇、GTH 稳态分布、簇内条件稳态的纯函数检验。

- 分簇按相对耦合强度连边，超限分量在内部提高阈值再切分；
- GTH 稳态分布与线性方程组解一致，速率跨 40 多个数量级时仍保持逐分量相对精度；
- 簇内条件稳态在 review F04 的 4 态反例上给出精确的 0.4（单陷阱估计量给 0.3）；
- 簇内无耦合时逐成员退回单陷阱条件通量，贴电极、交换速率 1e13 s⁻¹ 的成员
  （净通量只有 O(1) s⁻¹）也不被舍入误差淹没。
"""

import numpy as np
import pytest

from kmc_leakage.clusters import (
    cluster_generator,
    cluster_steady_state,
    coupled_clusters,
    coupling_strength,
    enumerate_occupations,
    stationary_distribution_gth,
)
from kmc_leakage.kmc import _B_R2A, _B_R2C, _B_R3A, _B_R3C, _B_R8, KMCSimulation
from kmc_leakage.master_equation import stationary_distribution


def _pairs(edges):
    a = np.array([e[0] for e in edges])
    b = np.array([e[1] for e in edges])
    hop = np.array([e[2] for e in edges], dtype=float)
    return a, b, hop


# --------------------------------------------------------------------------
# 分簇
# --------------------------------------------------------------------------
def test_coupling_strength_is_hop_over_slower_trap_rate():
    """强度 = 跃迁速率 / 两端电极交换速率的较小者；较慢一端 Γ=0 且有跃迁时记为无穷。"""
    a, b, hop = _pairs([(0, 1, 5.0), (1, 0, 1.0), (2, 3, 3.0), (3, 2, 0.0)])
    gamma = np.array([10.0, 1000.0, 0.0, 4.0])
    strength = coupling_strength(a, b, hop, gamma)
    assert strength[0] == pytest.approx(0.5)
    assert strength[1] == pytest.approx(0.1)
    assert np.isinf(strength[2])
    assert strength[3] == 0.0


def test_coupled_clusters_groups_by_relative_strength():
    """强度 ≥ 阈值的对连边、取连通分量；只返回大小 ≥2 的簇。"""
    a, b, hop = _pairs([(0, 1, 5.0), (2, 3, 1e-5), (3, 4, 0.01), (4, 5, 1e-6)])
    gamma = np.array([10.0, 1000.0, 1.0, 1.0, 1e9, 1.0])
    strength = coupling_strength(a, b, hop, gamma)
    clusters = coupled_clusters(6, a, b, strength, 1e-3, 8)
    assert [c.tolist() for c in clusters] == [[0, 1], [3, 4]]


def test_coupled_clusters_splits_oversized_component_at_weakest_link():
    """5 个陷阱的链超过上限 3：阈值逐级 ×10，先断开最弱的 1-2 这条（强度 0.01）。"""
    a, b, hop = _pairs([(0, 1, 1.0), (1, 2, 0.01), (2, 3, 1.0), (3, 4, 1.0)])
    strength = coupling_strength(a, b, hop, np.ones(5))
    clusters = coupled_clusters(5, a, b, strength, 1e-3, 3)
    assert [c.tolist() for c in clusters] == [[0, 1], [2, 3, 4]]


# --------------------------------------------------------------------------
# GTH 稳态分布
# --------------------------------------------------------------------------
def test_gth_matches_linear_solve_on_random_generator():
    rng = np.random.default_rng(4)
    Q = rng.uniform(0.1, 5.0, size=(7, 7))
    np.fill_diagonal(Q, 0.0)
    Q[np.diag_indices(7)] = -Q.sum(axis=1)
    pi = stationary_distribution_gth(Q)
    assert pi == pytest.approx(stationary_distribution(Q), rel=1e-12)
    assert np.allclose(pi @ Q, 0.0, atol=1e-12)


def test_gth_keeps_componentwise_accuracy_on_stiff_chain():
    """生灭链速率跨 1e-20…1e14：π 有乘积形式解析解，GTH 每个分量相对误差 ≤1e-12。

    π 最小的分量约 1e-44，直接解线性方程组只能保证相对最大分量的绝对精度。
    """
    up = np.array([1e14, 1e-10, 1e3, 1e-20])
    down = np.array([1e-12, 1e8, 1e-5, 1e14])  # down[k]：状态 k+1 → k
    n = up.size + 1
    Q = np.zeros((n, n))
    for k in range(n - 1):
        Q[k, k + 1] = up[k]
        Q[k + 1, k] = down[k]
    Q[np.diag_indices(n)] = -Q.sum(axis=1)
    weights = np.concatenate([[1.0], np.cumprod(up / down)])
    expected = weights / weights.sum()
    assert stationary_distribution_gth(Q) == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------
# 簇内条件稳态
# --------------------------------------------------------------------------
def test_cluster_steady_state_is_exact_on_review_counterexample():
    """review F04 的 4 态反例：负极→陷阱0、陷阱0→陷阱1、陷阱1→正极，速率均 1/s。

    单陷阱条件通量给 0.3（`test_master_equation` 里复现），把两个陷阱放进同一个簇
    就是精确的 0.4，占据期望分别为 0.6、0.4。
    """
    states = enumerate_occupations(2)
    zeros = np.zeros((4, 2))
    capture_other = zeros.copy()
    capture_other[:, 0] = 1.0
    escape_positive = zeros.copy()
    escape_positive[:, 1] = 1.0
    tat, pf, occupancy = cluster_steady_state(
        states,
        capture_positive=zeros,
        capture_other=capture_other,
        escape_positive=escape_positive,
        pf=zeros,
        escape_other=zeros,
        intra_a=np.array([0]),
        intra_b=np.array([1]),
        intra_rate=np.ones((4, 1)),
    )
    assert float(np.sum(tat + pf)) == pytest.approx(0.4, abs=1e-12)
    assert tat == pytest.approx([0.0, 0.4], abs=1e-12)
    assert occupancy == pytest.approx([0.6, 0.4], abs=1e-12)


def test_cluster_generator_follows_kmc_event_rules():
    """俘获要求空、逃逸要求占据、跃迁要求源占据且目标空；对角元 = −行和。"""
    states = enumerate_occupations(2)
    capture = np.tile([2.0, 3.0], (4, 1))
    escape = np.tile([5.0, 7.0], (4, 1))
    Q = cluster_generator(states, capture, escape, np.array([0]), np.array([1]), np.full((4, 1), 11.0))
    # 组态编号：0=空空，1=陷阱0占，2=陷阱1占，3=都占
    assert Q[0, 1] == 2.0 and Q[0, 2] == 3.0
    assert Q[1, 0] == 5.0 and Q[1, 3] == 3.0 and Q[1, 2] == 11.0
    assert Q[2, 3] == 2.0 and Q[2, 0] == 7.0 and Q[2, 1] == 0.0
    assert Q[3, 2] == 5.0 and Q[3, 1] == 7.0
    assert np.allclose(Q.sum(axis=1), 0.0)


def test_uncoupled_cluster_reproduces_single_trap_flux_near_positive():
    """簇内没有跃迁时，各成员退回单陷阱条件通量（rtol 1e-10）。

    成员 0 贴着正极：正极俘获/逃逸 4e13、6e13 s⁻¹，净通量却只有 0.4 s⁻¹。正极侧
    写法是两个 1e13 量级数之差，这里必须改用另一侧的写法才不丢精度。
    """
    blocks = np.zeros((9, 2))
    blocks[[_B_R2C, _B_R3C, _B_R2A, _B_R3A, _B_R8], 0] = [1.0, 0.5, 4e13, 6e13, 0.1]
    blocks[[_B_R2C, _B_R3C, _B_R2A, _B_R3A, _B_R8], 1] = [3.0, 2.0, 0.1, 5.0, 0.0]
    tat_ref, pf_ref, p_ref = KMCSimulation._conditional_flux(blocks, np.zeros(2), np.zeros(2))

    def per_state(block):
        return np.tile(blocks[block], (4, 1))

    tat, pf, occupancy = cluster_steady_state(
        enumerate_occupations(2),
        capture_positive=per_state(_B_R2A),
        capture_other=per_state(_B_R2C),
        escape_positive=per_state(_B_R3A),
        pf=per_state(_B_R8),
        escape_other=per_state(_B_R3C),
        intra_a=np.zeros(0, dtype=int),
        intra_b=np.zeros(0, dtype=int),
        intra_rate=np.zeros((4, 0)),
    )
    assert tat_ref[0] == pytest.approx(0.36, rel=1e-12)
    assert tat == pytest.approx(tat_ref, rel=1e-10)
    assert pf == pytest.approx(pf_ref, rel=1e-10)
    assert occupancy == pytest.approx(p_ref, rel=1e-10)
