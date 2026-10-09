"""簇级平均场内层求解器：TDDB 外层每步"解输运到稳态"的默认路径。

规格见 PROJECT_CONTEXT_TDDB.md §6.2、§6.9，决策记录见 ADR-031、ADR-038。

TDDB 外层每生成一个空位都要重求一次输运稳态，KMC 当主路径机时不可行（§6.2）。这里
不做时间推进，直接求占据的不动点 p = F(p)：

- F 是给定平均占据 p 下的条件稳态占据（`mean_field_flux`）。按 p 叠加静电环境得到各陷阱
  的平均场能级 E_i(p) 与电极速率（`KMCSimulation._trap_state` / `_electrode_blocks`）。孤立
  陷阱走单陷阱公式（`_conditional_flux`），跃迁供给按邻居的分数占据加权，即平均场 Pauli
  阻塞；强耦合簇（ADR-022）内部枚举 2^k 个组态、按各组态的真实电荷分布重算速率
  （`_cluster_tables`）、解簇内主方程稳态，簇外陷阱的分数占据作为环境。
- 所以簇内的 Pauli / 库仑关联精确计入，簇与簇之间、孤立陷阱之间按平均电荷计，即
  "簇级平均场"。孤立陷阱是 k=1 的特例；整个体系就是一个簇时，不动点就是主方程精确
  稳态。
- 陷阱间跃迁按两端归属分三类（ADR-038），每一条跃迁的速率只算一次，两端共用：
  1. **簇内对**：按组态求值，事件能量差含 −U_ab（ADR-021 F02），与 KMC 相同。
  2. **两端都按平均场处理的对**（孤立–孤立，以及分属两个簇的簇–簇对）：R7 用两端平均场
     能级之差 ΔH = E_b(p) − E_a(p)（`mean_field_hop_rates`），不减 U_ab，速率与组态无关。
     KMC 的 ΔH = E_b − U_ab − E_a 是"a 占据、b 空"这一具体组态下的事件能量差；按平均占据
     求出的能级已经含对方的平均库仑贡献，再减 U_ab 会让正反两个方向都多减，平均场自身的
     细致平衡被破坏，零偏压下也有环流（ADR-037）。用平均场能级之差时，零偏压的平均场
     平衡 p_i = f(E_i(p)) 让每一对的正反向通量逐对相等。两侧通量都是 p_a·(1 − p_b)·k_ab。
  3. **簇–孤立对**：簇成员一端取它在每个簇内组态 s 下的能级 e_i(s)，孤立一端取平均场
     能级 E_j(p)，R7 的 ΔH 就是两者之差，速率随簇的组态变。簇的生成矩阵用这组按组态的
     速率（乘 p_j 或 1 − p_j）；孤立陷阱一侧用同一组速率按簇内稳态分布 π 求平均
     （`MeanFieldRates.hop_in/hop_out`）。两侧通量相同，所以电流连续；零偏压下孤立陷阱
     p_j = f(E_j(p))，这条跃迁对簇来说就是费米能级处的库，簇内链在每个组态下都满足细致
     平衡，没有环流。速率若取与组态无关的常数，就与簇内按组态求值的电极速率满足不了细致
     平衡，零偏压下会经"电极 → 成员 → 边界"形成环流（ADR-038 的实测）。
- 簇–簇对要精确就得按两个簇稳态分布的乘积平均，迭代状态要带上各簇的分布，这里不做；
  它们按第 2 类处理，电流连续、能量守恒照样精确，零偏压下留一个小环流（ADR-038）。
- 不动点处的阳极净通量就是电流，与 KMC 的条件通量估计量是同一个公式，只是占据组态
  换成了平均占据、跃迁速率按上面三类取。R1 与组态无关，照旧加在总电流上。

**迭代变量取 ln p，收敛判据是占据的相对残差** max|ln p̂ − ln p| ≤ `_TOL`。占据很小的
陷阱可以经快跃迁供给邻居：p_a = 1e-10、跃迁 1e10 s⁻¹ 就是 1 s⁻¹ 的供给，与低场下的
净电流同量级。绝对残差判据会在 p=0 处提前停下、把这条通路整个漏掉；相对判据要求每个
陷阱的占据都收敛到相对精度。p 接近 1 时 ln p ≈ −(1−p)，相对判据退化为绝对判据。

不动点先用 Anderson 混合（type II，保留最近 `_HISTORY` 步）求解：静电反馈让简单阻尼迭代
收敛慢，Anderson 混合用历史残差的最小二乘组合外推下一步（§6.2）。初值可以含 0（全空、
ln 空间放不下），所以第一步直接取 p̂(p₀)，之后在 ln p 上混合，ln p 截在 ≤ 0（p ≤ 1）。
外层热启动时把上一步的占据作初值传入，新生空位的初值取 1（§6.7）。

**Anderson 在 `_ANDERSON_MAX_ITER` 次内没收敛就改用牛顿法**（`_newton`，ADR-048）。膜体内致密的空位网络里，
相距 0.5–1 nm、分数占据的陷阱之间既有快跃迁供给、又有几个 k_BT 的库仑排斥，簇大小上限把它们切开后按平均场
处理，不动点映射 F 的 Jacobian 有大于 1 的特征值（实测 1.17），而且 ln p 在迭代之间成 O(1) 地变，Anderson
用 5 步历史拟合的割线模型跟不上，在不动点附近来回游走。不动点本身良态（G = ln F − ln p 的 Jacobian 最小
特征值模 0.15），牛顿法按完整 Jacobian 走，不稳定方向也能一步解掉。Jacobian 用有限差分（每列一次 F），之后
各步做 Broyden 秩一更新；沿牛顿方向回溯线搜索，‖G‖₂ 下降才接受。起点先取传入的初值，停住了（重求 Jacobian
后线搜索仍不下降）再从 p̂(0)（全空环境下的条件占据）重来：Anderson 游走过的迭代值上 ‖G‖₂ 有不为 0 的局部
极小，从那里出发的牛顿法会停在极小处。

KMC 的快陷阱（ADR-023）只改变时间推进，平均场没有时间推进，不用它（§6.9 第 4 条）。
"""

from dataclasses import dataclass, replace
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy import constants

from .clusters import cluster_flux, cluster_generator, stationary_distribution_gth
from .kmc import (
    _B_R2A,
    _B_R2C,
    _B_R3A,
    _B_R3C,
    _B_R4A,
    _B_R4C,
    _B_R5A,
    _B_R5C,
    _B_R8,
    KMCSimulation,
    _ClusterTable,
)
from .rates import miller_abrahams_factor

# 收敛判据：占据的相对残差 max|ln p̂ − ln p| ≤ _TOL
_TOL = 1e-10
# Anderson 混合的迭代上限，没收敛就改用牛顿法。ADR-046 的参考轨迹（约 2800 条记录）最多用 28 次
_ANDERSON_MAX_ITER = 50
_HISTORY = 5  # Anderson 混合保留的历史步数
# 牛顿法（ADR-048）：每个起点的步数上限、有限差分 Jacobian 的求值次数上限（起点一次，Broyden 更新后的 Jacobian
# 走不动时重求一次）、ln p 上的差分步长、回溯线搜索的最小步长比例
_NEWTON_MAX_STEPS = 100
_NEWTON_JACOBIANS = 2
_JACOBIAN_STEP = 1e-7
_MIN_STEP_FRACTION = 2.0 ** -10
# ln p 的下限：p̂ 下溢成 0 时取最小正规格化浮点数，ln 空间里仍是有限值
_LOG_FLOOR = float(np.log(np.finfo(float).tiny))


@dataclass
class MeanFieldResult:
    """簇级平均场不动点处的稳态电流与分解。电流均为净值，正号 = 电子由阴极流向阳极。"""

    voltage: float  # [V]
    temperature: float  # [K]
    current: float  # [A]
    current_density: float  # [A/m^2]
    area: float  # 电极面积 [m^2]
    by_mechanism: Dict[str, float]  # 各机制电流 [A]，四项之和 = current
    by_defect: np.ndarray  # 各缺陷电流贡献 [A]，之和 + R1 两项 = current
    occupancy: np.ndarray  # 不动点占据 p*（强耦合簇成员是簇内稳态下的占据期望）
    # 求 F 的次数（含最后一次确认收敛的求值；牛顿法的有限差分 Jacobian 与线搜索都算在内）。电流与占据取残差
    # 最小的那次求值处，收敛时就是最后一次
    iterations: int
    residual: float  # 那次求值的相对残差 max|ln p̂ − ln p|
    converged: bool  # residual ≤ _TOL；未收敛的结果不得当稳态电流交付
    newton_steps: int = 0  # 牛顿法走的步数；0 表示 Anderson 混合已经收敛


def _log_occupancy(p: np.ndarray) -> np.ndarray:
    """ln p，下溢的 0 截到 `_LOG_FLOOR`。"""
    with np.errstate(divide="ignore"):
        return np.maximum(np.log(p), _LOG_FLOOR)


def mean_field_hop_rates(sim: KMCSimulation, energy: np.ndarray) -> np.ndarray:
    """两端都按平均场处理的陷阱对的跃迁速率（R7），近邻表上各有序对一个值，形状 (P,)。

    R6 的几何因子乘 MA 因子，事件能量差取两端平均场能级之差 ΔH = E_b − E_a（`energy` 是
    平均场能级 E_i(p)），不减 U_ab，理由见模块文档。簇内对与簇–孤立对不用这一项，它们
    按簇内组态求值（`mean_field_rates`）。
    """
    pairs = sim._pairs
    return sim._hop_geometric * miller_abrahams_factor(
        energy[pairs.a], energy[pairs.b], sim.sim.temperature
    )


@dataclass(frozen=True)
class MeanFieldRates:
    """平均占据 p 下内层平均场的全部速率（ADR-038）。

    不动点映射 F（`mean_field_flux`）与式 T13 的放热记账（`thermal.heat_release`）共用这一份，
    两者的通量与能量出自同一组速率。
    """

    energy: np.ndarray  # (N,) 平均场能级 E_i(p)
    blocks: np.ndarray  # (9, N) 电极速率块
    weighted: Optional[np.ndarray]  # (9, N) 按替换的声子线型再算的电极速率块，未请求时为 None
    hop: np.ndarray  # (P,) 两端按平均场处理时的跃迁速率（`mean_field_hop_rates`）
    cluster_of: np.ndarray  # (N,) 所在强耦合簇的编号，孤立陷阱为 −1
    # 各簇的组态表：簇内跃迁按组态（含 −U_ab），簇–孤立边界按成员的组态能级，簇–簇边界取 `hop`；
    # hop_in / hop_out 是这组速率乘环境占据后的总和
    tables: List[_ClusterTable]
    distributions: List[np.ndarray]  # 各簇的簇内稳态分布 π，与 `tables` 一一对应
    hop_in: np.ndarray  # (N,) 孤立陷阱的跃迁流入总速率（源是簇成员时按 π 求平均）
    hop_out: np.ndarray  # (N,) 孤立陷阱的跃迁流出总速率（目标是簇成员时按 π 求平均）


def _cluster_generator_rates(table: _ClusterTable) -> Tuple[np.ndarray, ...]:
    """簇的生成矩阵所需的 (阳极俘获, 其余俘获, 阳极 TAT 逃逸, PF, 其余逃逸)，口径同 `_cluster_flux`。"""
    block = table.block
    return (
        block[_B_R2A] + block[_B_R4A],
        block[_B_R2C] + block[_B_R4C] + table.hop_in,
        block[_B_R3A] + block[_B_R5A],
        block[_B_R8],
        block[_B_R3C] + block[_B_R5C] + table.hop_out,
    )


def mean_field_rates(
    sim: KMCSimulation, occupancy: np.ndarray, weights: Optional[np.ndarray] = None
) -> MeanFieldRates:
    """平均占据 `occupancy` 下的全部速率、各簇的组态表与簇内稳态分布（三类跃迁见模块文档）。

    `weights` 给出时另按它再算一遍电极速率块（`KMCSimulation._electrode_blocks` 的同名参数），
    放进 `weighted` 与各簇表的 `weighted`，供放热记账用。
    """
    p = np.asarray(occupancy, dtype=float)
    n = sim.n_defects
    temperature = sim.sim.temperature
    energy, field, barrier_c, barrier_a = sim._trap_state(p)
    blocks = sim._electrode_blocks(energy, field, barrier_c, barrier_a)
    weighted = None
    if weights is not None:
        weighted = sim._electrode_blocks(energy, field, barrier_c, barrier_a, weights=weights)
    hop = mean_field_hop_rates(sim, energy)

    cluster_of = np.full(n, -1)
    for label, members in enumerate(sim.cluster_members):
        cluster_of[members] = label
    a, b = sim._pairs.a, sim._pairs.b
    # 孤立陷阱与孤立陷阱之间：常数速率乘对方的平均占据；与簇成员之间的部分在下面按 π 补上
    hop_in = np.bincount(b, hop * p[a] * (cluster_of[a] < 0), n)
    hop_out = np.bincount(a, hop * (1.0 - p[b]) * (cluster_of[b] < 0), n)

    tables, distributions = [], []
    for table in sim._cluster_tables(p, weights):
        c = table.cluster
        members, states, e = c.members, c.states, table.energy
        geometric = sim._hop_geometric[c.pairs]
        rate = table.hop.copy()  # 簇内列沿用组态求值（含 −U_ab），边界列下面重写
        source_free = cluster_of[c.in_source] < 0
        target_free = cluster_of[c.out_target] < 0
        cols = c.in_cols
        rate[:, cols] = hop[c.pairs[cols]]
        rate[:, cols[source_free]] = geometric[cols[source_free]] * miller_abrahams_factor(
            energy[c.in_source[source_free]], e[:, members[c.in_member[source_free]]], temperature
        )
        cols = c.out_cols
        rate[:, cols] = hop[c.pairs[cols]]
        rate[:, cols[target_free]] = geometric[cols[target_free]] * miller_abrahams_factor(
            e[:, members[c.out_member[target_free]]], energy[c.out_target[target_free]], temperature
        )
        n_states, k = states.shape
        member_in = np.zeros((n_states, k))
        member_out = np.zeros((n_states, k))
        np.add.at(member_in.T, c.in_member, (p[c.in_source] * rate[:, c.in_cols]).T)
        np.add.at(member_out.T, c.out_member, ((1.0 - p[c.out_target]) * rate[:, c.out_cols]).T)
        table = replace(table, hop=rate, hop_in=member_in, hop_out=member_out)

        capture_anode, capture_other, escape_anode, pf, escape_other = _cluster_generator_rates(table)
        pi = stationary_distribution_gth(
            cluster_generator(
                states, capture_anode + capture_other, escape_anode + pf + escape_other,
                c.intra_a, c.intra_b, rate[:, c.intra_cols],
            )
        )
        # 孤立一侧：同一组按组态的速率按 π 求平均（跳往成员要成员空，成员跳出要成员占据）
        occupied = states.astype(float)
        cols = c.in_cols[source_free]
        hop_out += np.bincount(
            c.in_source[source_free], pi @ ((1.0 - occupied[:, c.in_member[source_free]]) * rate[:, cols]), n
        )
        cols = c.out_cols[target_free]
        hop_in += np.bincount(
            c.out_target[target_free], pi @ (occupied[:, c.out_member[target_free]] * rate[:, cols]), n
        )
        tables.append(table)
        distributions.append(pi)

    return MeanFieldRates(
        energy=energy,
        blocks=blocks,
        weighted=weighted,
        hop=hop,
        cluster_of=cluster_of,
        tables=tables,
        distributions=distributions,
        hop_in=hop_in,
        hop_out=hop_out,
    )


def mean_field_flux(
    sim: KMCSimulation, occupancy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """平均占据 `occupancy` 下每个陷阱的阳极净通量 (TAT, PF) 与条件稳态占据 p̂（不动点映射 F）。

    孤立陷阱走单陷阱公式，簇成员按簇内稳态分布给出（`clusters.cluster_flux`，与 KMC 估计量
    同一个公式）。通量单位 1/s。
    """
    rates = mean_field_rates(sim, occupancy)
    tat, pf, p_hat = sim._conditional_flux(rates.blocks, rates.hop_in, rates.hop_out)
    for table, pi in zip(rates.tables, rates.distributions):
        c = table.cluster
        tat[c.members], pf[c.members], p_hat[c.members] = cluster_flux(
            pi, c.states, *_cluster_generator_rates(table), c.intra_a, c.intra_b, table.hop[:, c.intra_cols]
        )
    return tat, pf, p_hat


class _Iterate(NamedTuple):
    """不动点迭代里的一次求值：x = ln p 处的残差 G(x) = ln p̂ − x 与阳极净通量 (TAT, PF)。"""

    x: np.ndarray
    gap: np.ndarray
    tat: np.ndarray
    pf: np.ndarray

    @property
    def residual(self) -> float:
        return float(np.max(np.abs(self.gap))) if self.gap.size else 0.0


def _evaluate(sim: KMCSimulation, x: np.ndarray) -> _Iterate:
    tat, pf, p_hat = mean_field_flux(sim, np.exp(x))
    return _Iterate(x, _log_occupancy(p_hat) - x, tat, pf)


def _jacobian(sim: KMCSimulation, current: _Iterate) -> np.ndarray:
    """∂G/∂x 的有限差分（每列一次 F）；差分向 x 减小的方向取，p 不越过 1。"""
    n = sim.n_defects
    jacobian = np.empty((n, n))
    for j in range(n):
        shifted = current.x.copy()
        shifted[j] -= _JACOBIAN_STEP
        jacobian[:, j] = (_evaluate(sim, shifted).gap - current.gap) / -_JACOBIAN_STEP
    return jacobian


def _newton(sim: KMCSimulation, starts: List[np.ndarray]) -> Tuple[_Iterate, int, int]:
    """牛顿法求 G(x) = 0（x = ln p ≤ 0），依次从 `starts` 出发，返回 (残差最小的求值, 求 F 的次数, 牛顿步数)。

    每个起点先求有限差分 Jacobian，之后每步做 Broyden 秩一更新 J ← J + (Δg − J·Δx)·Δxᵀ/(Δx·Δx)。牛顿方向上
    回溯线搜索：‖G‖₂ 按 Armijo 条件下降才接受；步长比例降到 `_MIN_STEP_FRACTION` 仍不下降就重求 Jacobian。
    刚求的 Jacobian 也走不动、Jacobian 已求满 `_NEWTON_JACOBIANS` 次，或步数用满，就换下一个起点。
    """
    evaluations = steps = 0
    best: Optional[_Iterate] = None
    for start in starts:
        current = _evaluate(sim, start)
        evaluations += 1
        if best is None or current.residual < best.residual:
            best = current
        jacobian, jacobians, fresh = None, 0, False
        for _ in range(_NEWTON_MAX_STEPS):
            if current.residual <= _TOL:
                return current, evaluations, steps
            if jacobian is None:
                if jacobians == _NEWTON_JACOBIANS:
                    break
                jacobian = _jacobian(sim, current)
                evaluations += sim.n_defects
                jacobians += 1
                fresh = True
            direction = np.linalg.solve(jacobian, -current.gap)
            norm = np.linalg.norm(current.gap)
            accepted, fraction = None, 1.0
            while fraction >= _MIN_STEP_FRACTION:
                trial = _evaluate(sim, np.clip(current.x + fraction * direction, _LOG_FLOOR, 0.0))
                evaluations += 1
                if np.linalg.norm(trial.gap) <= (1.0 - 1e-4 * fraction) * norm:
                    accepted = trial
                    break
                fraction *= 0.5
            if accepted is None:
                if fresh:
                    break
                jacobian = None
                continue
            dx, dg = accepted.x - current.x, accepted.gap - current.gap
            jacobian += np.outer(dg - jacobian @ dx, dx) / (dx @ dx)
            current, fresh = accepted, False
            steps += 1
            if current.residual < best.residual:
                best = current
    return best, evaluations, steps


def solve_mean_field(
    sim: KMCSimulation, occupancy: Optional[np.ndarray] = None
) -> MeanFieldResult:
    """在 `sim` 的缺陷实现与工作点上求簇级平均场不动点。

    `sim` 提供速率、近邻表与强耦合簇（工作点初始化时已分好）。`occupancy` 是占据初值，
    缺省为全空；热启动时传上一步的不动点占据。先用 Anderson 混合，`_ANDERSON_MAX_ITER` 次内没收敛
    就改用牛顿法（模块文档），起点依次是初值与 p̂(0)。返回的电流在残差最小的那次求值处给出（收敛时
    就是最后一次），分解口径与 `KMCSimulation.run` 相同。
    """
    n = sim.n_defects
    p = np.zeros(n) if occupancy is None else np.clip(np.asarray(occupancy, dtype=float), 0.0, 1.0)
    x_history: List[np.ndarray] = []
    g_history: List[np.ndarray] = []
    for iterations in range(1, _ANDERSON_MAX_ITER + 1):
        tat, pf, p_hat = mean_field_flux(sim, p)
        x_hat = _log_occupancy(p_hat)
        if iterations == 1:
            # 初值可能含 0，第一步直接取 p̂(p₀)，Anderson 历史从下一步开始
            x = _log_occupancy(p)
            residual = float(np.max(np.abs(x_hat - x))) if n > 0 else 0.0
            if residual <= _TOL:
                break
            x = x_hat
        else:
            g = x_hat - x
            residual = float(np.max(np.abs(g)))
            if residual <= _TOL or iterations == _ANDERSON_MAX_ITER:
                break
            x_history.append(x)
            g_history.append(g)
            if len(x_history) > _HISTORY + 1:
                x_history.pop(0)
                g_history.pop(0)
            step = x + g
            if len(x_history) > 1:
                # 残差差分张成的子空间里最小二乘组合当前残差，按同一组合外推
                dx = np.diff(np.array(x_history), axis=0).T
                dg = np.diff(np.array(g_history), axis=0).T
                gamma = np.linalg.lstsq(dg, g, rcond=None)[0]
                step = step - (dx + dg) @ gamma
            x = np.clip(step, _LOG_FLOOR, 0.0)
        p = np.exp(x)

    newton_steps = 0
    if residual > _TOL:
        # Anderson 没收敛：改用牛顿法，起点依次是传入的初值与 p̂(0)
        starts = [] if occupancy is None else [_log_occupancy(np.clip(np.asarray(occupancy, dtype=float), 0.0, 1.0))]
        starts.append(_log_occupancy(mean_field_flux(sim, np.zeros(n))[2]))
        best, evaluations, newton_steps = _newton(sim, starts)
        iterations += 1 + evaluations
        if best.residual < residual:
            p, tat, pf, residual = np.exp(best.x), best.tat, best.pf, best.residual

    q = constants.elementary_charge
    r1_tunnel, r1_thermionic = sim._r1_current
    by_mechanism = {
        "direct_fn": r1_tunnel,
        "thermionic": r1_thermionic,
        "tat": float(tat.sum()) * q,
        "pf": float(pf.sum()) * q,
    }
    current = float(sum(by_mechanism.values()))
    return MeanFieldResult(
        voltage=sim.sim.voltage,
        temperature=sim.sim.temperature,
        current=current,
        current_density=current / sim.device.area,
        area=sim.device.area,
        by_mechanism=by_mechanism,
        by_defect=(tat + pf) * q,
        occupancy=p,
        iterations=iterations,
        residual=residual,
        converged=residual <= _TOL,
        newton_steps=newton_steps,
    )
