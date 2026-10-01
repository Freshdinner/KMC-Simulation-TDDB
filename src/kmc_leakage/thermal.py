"""TDDB 的功率耗散与稳态热流：式 T13（P = ΔE·R）与式 T14（∇²T = −P/k_TH）。

规格见 PROJECT_CONTEXT_TDDB.md §3、§5、§6.6、§8.4，决策记录见 ADR-036。公式编号 T13/T14 是
TDDB 的编号，与 docs/rate_equations.md 的 R1–R8 不对应。

**功率（式 T13）**：`heat_release` 在内层输运稳态上逐事件累加 ΔE·R——ΔE 是一次事件里电子
放给晶格的能量，R 是该事件在稳态下的发生率。能量零点是阴极费米能级，电极侧取各自的费米
能级 μ。按事件类型：

- R2/R3（弹性）：电子在陷阱能级 E 处隧穿，陷阱处不放热；电极里放出 μ − E（俘获）或
  E − μ（逃逸）。
- R4/R5（多声子，p 个声子）：电子在 E + p·ℏω 处离开/进入电极，俘获时陷阱放出 p·ℏω、
  逃逸时陷阱吸收 p·ℏω。按声子数加权的求和与速率走同一套公式
  （`KMCSimulation._electrode_blocks` 的 `weights` 参数）。
- R8（PF 发射）：陷阱吸收 E_D 把电子送到局部导带底；导带电子按引擎口径"立即被阳极收集"
  （docs/rate_equations.md），它的能量放进阳极。
- R7（陷阱间跃迁 a→b）：放出的能量平分给两端陷阱。

速率与内层簇级平均场走同一套（`mean_field.mean_field_rates`，ADR-038），一次 a→b 跃迁的放热
是电子在 a 上的能级减去它落到 b 上的能级，每一端的能级取它在自身分布下的值：

- 孤立陷阱取平均场能级 E_i；强耦合簇（ADR-022）的成员取簇内每个组态下的能级，按簇内主方程
  稳态分布 π 求期望。簇的电极事件同样按组态记账。
- 簇内跃迁：完整事件能量差 −ΔH = E_a − E_b + U_ab（与簇内 R7 的 MA 因子同一口径，ADR-021 F02）。
- 簇–孤立跃迁：速率随簇的组态变，放热 E_j − e_i(s) 或 e_i(s) − E_j 按 π 与孤立一端的占据加权。
- 两端都按平均场处理的对（孤立–孤立、簇–簇）：速率 k_ab 与组态无关，

      放热 = k_ab·[(1 − p_b)·⟨n_a·E_a⟩ − p_a·⟨(1 − n_b)·E_b⟩]，

  ⟨n·E⟩、⟨(1 − n)·E⟩ 对孤立陷阱是 p·E、(1 − p)·E，对簇成员按 π 对组态求；两个孤立陷阱之间
  就是 k_ab·p_a·(1 − p_b)·(E_a − E_b)。

每个陷阱（每个簇）收到与送出的能量按同一组通量、同一组能级记账，不动点处粒子数守恒，介质 +
两侧电极的放热就严格等于 U·I_陷阱——强耦合分量被切开时也成立。

R1 不经过缺陷，电子弹性穿过整个介质，能量全部放进电极，不产生介质内的功率。

**热流（式 T14）**：介质内的功率按 CIC 分到泊松网格（`power_map`），稳态热方程
k_TH·∇²T = −P、两电极恒温 T_amb（热沉）、xy 周期，直接调用 `PoissonSolver.solve`：它解
∇·(κε₀∇φ) = −ρ，令 ρ = P·κε₀/k_TH、φ_top = 0，则 T = T_amb + φ（`solve_temperature`）。
热扩散时间 L²/D ≈ 10⁻¹⁰ s，远短于两次生成的间隔，稳态热方程成立。这一映射要求 κ 均匀，
只适用于单一材料介质（TDDB 本阶段只做单层，§1）。
"""

from dataclasses import dataclass

import numpy as np
from scipy import constants

from .electrostatics import PoissonSolver
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
)
from .mean_field import mean_field_rates


@dataclass(frozen=True)
class HeatRelease:
    """内层输运稳态下单位时间放出的热量 [W]（式 T13）。负值表示从晶格吸热。"""

    defect: np.ndarray  # (N,) 各缺陷处放给介质晶格的功率
    cathode: float  # 阴极里放出的功率
    anode: float  # 阳极里放出的功率

    @property
    def dielectric(self) -> float:
        """介质内放出的总功率 [W]，即热方程的总热源。"""
        return float(self.defect.sum())


def _electrode_event_heat(
    occupied: np.ndarray,
    energy: np.ndarray,
    ED: np.ndarray,
    blocks: np.ndarray,
    phonon: np.ndarray,
    fermi_cathode: float,
    fermi_anode: float,
):
    """电极事件（R2–R5 两侧与 R8）的放热 (陷阱处, 阴极里, 阳极里) [W]。

    `blocks` 是电极速率块，`phonon` 是按 p·ℏω 加权的同一组块（只用 R4/R5 四块）；`occupied`
    是占据（平均场的分数占据，或簇内组态的 0/1）。数组形状逐元素对应，块的第一维是 9 个
    事件块。俘获要求陷阱空、逃逸与 R8 要求占据。
    """
    empty = 1.0 - occupied
    capture_c = blocks[_B_R2C] + blocks[_B_R4C]
    capture_a = blocks[_B_R2A] + blocks[_B_R4A]
    escape_c = blocks[_B_R3C] + blocks[_B_R5C]
    escape_a = blocks[_B_R3A] + blocks[_B_R5A]
    pf = blocks[_B_R8]
    trap = empty * (phonon[_B_R4C] + phonon[_B_R4A]) - occupied * (
        phonon[_B_R5C] + phonon[_B_R5A] + ED * pf
    )
    cathode = empty * ((fermi_cathode - energy) * capture_c - phonon[_B_R4C]) + occupied * (
        (energy - fermi_cathode) * escape_c + phonon[_B_R5C]
    )
    anode = empty * ((fermi_anode - energy) * capture_a - phonon[_B_R4A]) + occupied * (
        (energy - fermi_anode) * escape_a + phonon[_B_R5A] + (energy + ED - fermi_anode) * pf
    )
    return trap, cathode, anode


def heat_release(sim: KMCSimulation, occupancy: np.ndarray) -> HeatRelease:
    """缺陷集合在占据 `occupancy`（簇级平均场不动点）下各事件的放热 ΔE·R（式 T13）。

    `sim` 提供工作点、速率与强耦合簇，`occupancy` 取 `mean_field.solve_mean_field` 的不动点
    占据。记账口径见模块文档。
    """
    n = sim.n_defects
    if n == 0:
        return HeatRelease(defect=np.zeros(0), cathode=0.0, anode=0.0)
    p = np.asarray(occupancy, dtype=float)
    ED = sim.defects.ED
    fermi = (sim._fermi_cathode, sim._fermi_anode)
    weights = sim._phonon_weights * sim._phonon_offsets  # L_p·p·ℏω

    rates = mean_field_rates(sim, p, weights)
    energy, cluster_of = rates.energy, rates.cluster_of
    trap, cathode, anode = _electrode_event_heat(p, energy, ED, rates.blocks, rates.weighted, *fermi)

    # 各陷阱占据时与空时能级的期望 ⟨n·E⟩、⟨(1−n)·E⟩：孤立陷阱取平均场，簇成员下面按组态覆盖
    occupied_energy = p * energy
    empty_energy = (1.0 - p) * energy
    hop_heat = np.zeros(n)

    def deposit(ends, amount):
        for end in ends:
            hop_heat[:] += 0.5 * np.bincount(end, amount, n)

    for table, pi in zip(rates.tables, rates.distributions):
        c = table.cluster
        members, states, e, rate = c.members, c.states, table.energy, table.hop
        trap_s, cathode_s, anode_s = _electrode_event_heat(
            states.astype(float), e[:, members], ED[members], table.block, table.weighted, *fermi
        )
        trap[members], cathode[members], anode[members] = pi @ trap_s, pi @ cathode_s, pi @ anode_s

        # 簇内跃迁：完整事件能量差 −ΔH = E_a − E_b + U_ab
        ga, gb = members[c.intra_a], members[c.intra_b]
        allowed = states[:, c.intra_a] & ~states[:, c.intra_b]
        deposit((ga, gb), pi @ (
            allowed * rate[:, c.intra_cols] * (e[:, ga] - e[:, gb] + sim._hop_U_ab[c.pairs[c.intra_cols]])
        ))
        # 簇–孤立：速率按成员的组态能级，孤立一端取平均场能级
        free = cluster_of[c.in_source] < 0
        source, gm = c.in_source[free], members[c.in_member[free]]
        deposit((source, gm), pi @ (
            ~states[:, c.in_member[free]] * p[source] * rate[:, c.in_cols[free]] * (energy[source] - e[:, gm])
        ))
        free = cluster_of[c.out_target] < 0
        target, go = c.out_target[free], members[c.out_member[free]]
        deposit((go, target), pi @ (
            states[:, c.out_member[free]] * (1.0 - p[target]) * rate[:, c.out_cols[free]] * (e[:, go] - energy[target])
        ))
        occupied_energy[members] = pi @ (states * e[:, members])
        empty_energy[members] = pi @ (~states * e[:, members])

    # 两端都按平均场处理的对（孤立–孤立、簇–簇）：常数速率，两端能量各取自身分布下的期望
    a, b = sim._pairs.a, sim._pairs.b
    shared = (cluster_of[a] < 0) == (cluster_of[b] < 0)
    shared &= (cluster_of[a] < 0) | (cluster_of[a] != cluster_of[b])
    a, b = a[shared], b[shared]
    deposit((a, b), rates.hop[shared] * ((1.0 - p[b]) * occupied_energy[a] - p[a] * empty_energy[b]))

    return HeatRelease(
        defect=trap + hop_heat, cathode=float(cathode.sum()), anode=float(anode.sum())
    )


def power_map(solver: PoissonSolver, positions: np.ndarray, heat: np.ndarray) -> np.ndarray:
    """体功率密度 P [W/m^3]，形状同泊松网格节点：各缺陷的放热按 CIC 分到网格（式 T13）。"""
    return solver.deposit(positions, heat)


def solve_temperature(
    solver: PoissonSolver, power_density: np.ndarray, thermal_conductivity: float, ambient: float
) -> np.ndarray:
    """稳态温度场 T [K]（式 T14）：k_TH·∇²T = −P，两电极恒温 `ambient`，xy 周期。

    直接调用 `PoissonSolver.solve`（∇·(κε₀∇φ) = −ρ）：ρ = P·κε₀/k_TH、φ_top = 0，T = T_amb + φ。
    κ 取单一介质材料的静态 κ，多材料叠层在这里抛 NotImplementedError。返回节点上的温度，
    形状 (nx, ny, nz+1)。
    """
    kappa = solver.device.stack.single_material.kappa
    rho = np.asarray(power_density, dtype=float) * kappa * constants.epsilon_0 / thermal_conductivity
    return ambient + solver.solve(rho, 0.0)


def transport_temperature(
    solver: PoissonSolver,
    temperature: np.ndarray,
    positions: np.ndarray,
    heat: np.ndarray,
    ambient: float,
) -> float:
    """内层输运用的全局标量温度 [K]：温度场在缺陷处的值按各缺陷放热加权平均（§6.6）。

    权重只取放热（正值）部分，吸热的缺陷不计。介质内没有放热时返回 `ambient`。
    """
    weight = np.clip(np.asarray(heat, dtype=float), 0.0, None)
    total = float(weight.sum())
    if not total > 0.0:
        return float(ambient)
    return float(weight @ solver.interpolate(temperature, positions) / total)
