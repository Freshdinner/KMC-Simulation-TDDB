"""簇级平均场内层求解器：TDDB 外层每步"解输运到稳态"的默认路径。

规格见 PROJECT_CONTEXT_TDDB.md §6.2、§6.9，决策记录见 ADR-031。

TDDB 外层每生成一个空位都要重求一次输运稳态，KMC 当主路径机时不可行（§6.2）。这里
不做时间推进，直接求占据的不动点 p = F(p)：

- F 就是 KMC 电流估计量在给定占据下的条件稳态占据。`KMCSimulation._rates(p)` 按占据 p
  叠加静电环境、算全部速率；`_trap_flux` 给出每个陷阱的条件稳态占据 p̂ 与阳极净通量。
  孤立陷阱走单陷阱公式（`_conditional_flux`），跃迁供给按邻居的分数占据加权，即平均场
  Pauli 阻塞；强耦合簇（ADR-022）内部枚举 2^k 个组态、按各组态的真实电荷分布重算速率、
  解簇内主方程稳态，簇外陷阱的分数占据作为环境。
- 所以簇内的 Pauli / 库仑关联精确计入，簇与簇之间、孤立陷阱之间按平均电荷计，即
  "簇级平均场"。孤立陷阱是 k=1 的特例；整个体系就是一个簇时，不动点就是主方程精确
  稳态。
- 不动点处的阳极净通量就是电流，与 KMC 的条件通量估计量是同一个公式，只是占据组态
  换成了平均占据。R1 与组态无关，照旧加在总电流上。

**迭代变量取 ln p，收敛判据是占据的相对残差** max|ln p̂ − ln p| ≤ `_TOL`。占据很小的
陷阱可以经快跃迁供给邻居：p_a = 1e-10、跃迁 1e10 s⁻¹ 就是 1 s⁻¹ 的供给，与低场下的
净电流同量级。绝对残差判据会在 p=0 处提前停下、把这条通路整个漏掉；相对判据要求每个
陷阱的占据都收敛到相对精度。p 接近 1 时 ln p ≈ −(1−p)，相对判据退化为绝对判据。

不动点用 Anderson 混合（type II，保留最近 `_HISTORY` 步）求解：静电反馈让简单阻尼迭代
收敛慢，Anderson 混合用历史残差的最小二乘组合外推下一步（§6.2）。初值可以含 0（全空、
ln 空间放不下），所以第一步直接取 p̂(p₀)，之后在 ln p 上混合，ln p 截在 ≤ 0（p ≤ 1）。
外层热启动时把上一步的占据作初值传入，新生空位的初值取 1（§6.7）。

KMC 的快陷阱（ADR-023）只改变时间推进，平均场没有时间推进，不用它（§6.9 第 4 条）。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy import constants

from .kmc import KMCSimulation

# 收敛判据：占据的相对残差 max|ln p̂ − ln p| ≤ _TOL
_TOL = 1e-10
_MAX_ITER = 200
_HISTORY = 5  # Anderson 混合保留的历史步数
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
    # 求 F 的次数（含最后一次确认收敛的求值）；电流与占据都取最后一次求值处
    iterations: int
    residual: float  # 最后一次求值的相对残差 max|ln p̂ − ln p|
    converged: bool  # residual ≤ _TOL；未收敛的结果不得当稳态电流交付


def _log_occupancy(p: np.ndarray) -> np.ndarray:
    """ln p，下溢的 0 截到 `_LOG_FLOOR`。"""
    with np.errstate(divide="ignore"):
        return np.maximum(np.log(p), _LOG_FLOOR)


def solve_mean_field(
    sim: KMCSimulation, occupancy: Optional[np.ndarray] = None
) -> MeanFieldResult:
    """在 `sim` 的缺陷实现与工作点上求簇级平均场不动点。

    `sim` 提供速率、近邻表与强耦合簇（工作点初始化时已分好）。`occupancy` 是占据初值，
    缺省为全空；热启动时传上一步的不动点占据。返回的电流在最后一次求值的 p 处给出，
    分解口径与 `KMCSimulation.run` 相同。
    """
    n = sim.n_defects
    p = np.zeros(n) if occupancy is None else np.clip(np.asarray(occupancy, dtype=float), 0.0, 1.0)
    x_history: List[np.ndarray] = []
    g_history: List[np.ndarray] = []
    for iterations in range(1, _MAX_ITER + 1):
        blocks, hop = sim._rates(p)
        tat, pf, p_hat = sim._trap_flux(p, blocks, hop)
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
            if residual <= _TOL or iterations == _MAX_ITER:
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
    )
