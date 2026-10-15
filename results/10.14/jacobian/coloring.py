"""离线模拟分组有限差分：在 probe.py 存下的完整 Jacobian 上，看分组求 Jacobian 的近似够不够牛顿法用。

usage: coloring.py PROBE_NPZ

分组规则：同一组的缺陷两两间距 > S（贪心着色，按缺陷编号顺序）。一组扰动一次 F，第 i 行的变化等于组内全部成员的 J_ij 之和
（线性近似），记到组内离 i 最近的那个成员 j 的列上（离 i 都 > S/2 时也记到最近的那个）。这样得到的近似 J_S 与完整 J 比较：
- 组数（= 求一次 Jacobian 的 F 求值次数）；
- 用 J_S 作牛顿迭代矩阵时的线性收敛因子：I − J_S⁻¹ J 的谱半径（< 1 才收敛，越小越快）。
另列出"只截断、不混入"的 J_R（间距 ≤ R 的元素）作对照。
"""
import sys

import numpy as np

data = np.load(sys.argv[1])
J, d = data["jacobian"], data["distance"]
n = J.shape[0]


def greedy_groups(separation):
    group = np.full(n, -1)
    for j in range(n):
        used = set(group[(d[j] <= separation) & (group >= 0)].tolist())
        g = 0
        while g in used:
            g += 1
        group[j] = g
    return group


def compressed(group):
    estimate = np.zeros_like(J)
    for g in range(group.max() + 1):
        members = np.flatnonzero(group == g)
        row_change = J[:, members].sum(axis=1)
        nearest = members[np.argmin(d[:, members], axis=1)]
        estimate[np.arange(n), nearest] = row_change
    return estimate


def rho(approx):
    return float(np.max(np.abs(np.linalg.eigvals(np.eye(n) - np.linalg.solve(approx, J)))))


print("n = {}".format(n))
for separation in (2, 3, 4, 6, 8, 10, 12, 16, 20):
    group = greedy_groups(separation)
    estimate = compressed(group)
    error = np.linalg.norm(estimate - J) / np.linalg.norm(J)
    print("S = {:>2} nm: {:3d} groups, relative error {:.3f}, spectral radius {:.3g}".format(
        separation, group.max() + 1, error, rho(estimate)), flush=True)
