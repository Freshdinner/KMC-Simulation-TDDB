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

孤立陷阱在平均场里的能量取平均场能级 E_i，跃迁放热 E_a − E_b；每个陷阱在不动点处粒子数
平衡，这些能量项逐陷阱相消，陷阱与电极的放热之和严格等于 U·I_陷阱。强耦合簇（ADR-022）
按簇内每个组态的能级和簇内主方程稳态分布精确记账，簇内跃迁放热取完整事件能量差
−ΔH = E_a − E_b + U_ab（与 R7 的 MA 因子同一口径，ADR-021 F02）；簇内能量平衡由稳态分布
保证。簇与簇外陷阱之间的跃迁由簇一侧记账，簇外一端的能量取平均场能级：簇一侧的通量是
按组态算的，与簇外陷阱在平均场里看到的通量不完全相同，这部分放热只近似守恒。

R1 不经过缺陷，电子弹性穿过整个介质，能量全部放进电极，不产生介质内的功率。

**功率的可信度取决于内层解**：这里的记账把内层稳态的事件通量原样换成能量流。致密空位层上强耦合
分量被切开后，簇级平均场不动点不守恒电流；平均场处理的陷阱对的 MA 因子不满足平均场自身的细致
平衡，会形成环流、把晶格的能量搬进电极。这两种情况下算出的介质功率失真（实测与验证见 ADR-037）。

**热流（式 T14）**：介质内的功率按 CIC 分到泊松网格（`power_map`），稳态热方程
k_TH·∇²T = −P、两电极恒温 T_amb（热沉）、xy 周期，直接调用 `PoissonSolver.solve`：它解
∇·(κε₀∇φ) = −ρ，令 ρ = P·κε₀/k_TH、φ_top = 0，则 T = T_amb + φ（`solve_temperature`）。
热扩散时间 L²/D ≈ 10⁻¹⁰ s，远短于两次生成的间隔，稳态热方程成立。这一映射要求 κ 均匀，
只适用于单一材料介质（TDDB 本阶段只做单层，§1）。
"""

from dataclasses import dataclass

import numpy as np
from scipy import constants

from .clusters import cluster_generator, stationary_distribution_gth
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

    energy, field, barrier_c, barrier_a = sim._trap_state(p)
    blocks = sim._electrode_blocks(energy, field, barrier_c, barrier_a)
    phonon = sim._electrode_blocks(energy, field, barrier_c, barrier_a, weights=weights)
    trap, cathode, anode = _electrode_event_heat(p, energy, ED, blocks, phonon, *fermi)

    # 陷阱间跃迁：两端都不在簇里的对按平均场记账，其余由簇一侧记账
    pairs = sim._pairs
    in_cluster = np.zeros(n, dtype=bool)
    for members in sim.cluster_members:
        in_cluster[members] = True
    hop_heat = np.zeros(n)
    mean_field_pair = ~in_cluster[pairs.a] & ~in_cluster[pairs.b]
    a, b = pairs.a[mean_field_pair], pairs.b[mean_field_pair]
    flow = sim._hop_rates(energy)[mean_field_pair] * p[a] * (1.0 - p[b])
    released = flow * (energy[a] - energy[b])
    hop_heat += 0.5 * (np.bincount(a, released, n) + np.bincount(b, released, n))

    for table in sim._cluster_tables(p, weights):
        c = table.cluster
        members, states = c.members, c.states
        block, e, hop = table.block, table.energy, table.hop
        capture = block[_B_R2A] + block[_B_R4A] + (block[_B_R2C] + block[_B_R4C] + table.hop_in)
        escape = block[_B_R3A] + block[_B_R5A] + block[_B_R8] + (block[_B_R3C] + block[_B_R5C] + table.hop_out)
        pi = stationary_distribution_gth(
            cluster_generator(states, capture, escape, c.intra_a, c.intra_b, hop[:, c.intra_cols])
        )
        occupied = states.astype(float)
        trap_s, cathode_s, anode_s = _electrode_event_heat(
            occupied, e[:, members], ED[members], block, table.weighted, *fermi
        )
        trap[members], cathode[members], anode[members] = pi @ trap_s, pi @ cathode_s, pi @ anode_s

        # 簇内跃迁：完整事件能量差 −ΔH = E_a − E_b + U_ab
        ga, gb = members[c.intra_a], members[c.intra_b]
        allowed = states[:, c.intra_a] & ~states[:, c.intra_b]
        released = np.sum(
            pi[:, None] * allowed * hop[:, c.intra_cols]
            * (e[:, ga] - e[:, gb] + sim._hop_U_ab[c.pairs[c.intra_cols]]),
            axis=0,
        )
        # 簇外 → 成员：簇外一端的能量取平均场能级
        source, gm = c.in_source, members[c.in_member]
        released_in = np.sum(
            pi[:, None] * ~states[:, c.in_member] * p[source] * hop[:, c.in_cols]
            * (energy[source] - e[:, gm]),
            axis=0,
        )
        # 成员 → 簇外：目标在别的簇里时记在那个簇的"簇外 → 成员"里，这里跳过
        own = ~in_cluster[c.out_target]
        target, go = c.out_target[own], members[c.out_member[own]]
        released_out = np.sum(
            pi[:, None] * states[:, c.out_member[own]] * (1.0 - p[target]) * hop[:, c.out_cols[own]]
            * (e[:, go] - energy[target]),
            axis=0,
        )
        for ends, amount in (((ga, gb), released), ((source, gm), released_in), ((go, target), released_out)):
            for end in ends:
                hop_heat += 0.5 * np.bincount(end, amount, n)

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
