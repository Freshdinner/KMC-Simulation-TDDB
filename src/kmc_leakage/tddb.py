"""TDDB 瞬态驱动：准静态双层循环，含热反馈与击穿判据。

规格见 PROJECT_CONTEXT_TDDB.md §5、§6，决策记录见 ADR-032–060。公式编号 T1–T14 是 TDDB 的
编号（PROJECT_CONTEXT_TDDB.md 第 3 节），与 docs/rate_equations.md 的 R1–R8 不对应。

电子输运事件的时间尺度（~10⁻¹³…10⁻⁶ s）比缺陷生成（秒以上）短十几个数量级，所以输运相对
生成瞬间达到稳态（§5）。t = 0 在初始缺陷之外铺离散前驱位点（§6.1，ADR-052/055）：可采样窗口里按最小间隔
min_separation 做最大随机堆积，直到再也放不下一个位点，浓度由间隔决定；每个位点的无场断键激活能从上限 E_A 往下
按指数尾抽取，E_A,i = E_A − E₀·X_i（ADR-063）。外层是缺陷生成的 Gillespie 循环，每步：

1. **导体划分**（§6.3，ADR-054/060/064，`conductor_layout`）：两个 O 空位之间转移积分的两倍大于重组能时电子离域，连成导电簇；
   离电极平面小于自身离域距离的空位接到该电极（`conducting_clusters`）。接电极的导电簇是导体、电极的延伸：接负极的电势 0、费米能级 0，接正极的电势 U、费米能级
   −eU；贯穿通路按均匀电阻处理，电势与费米能级沿 z 从负极线性变到正极。其余缺陷（含悬空导电簇的成员）是 TAT 陷阱。
2. **内层稳态与热迭代**（§6.2、§6.6、§8.4，ADR-060，`_steady_state`）：在陷阱上建 `KMCSimulation`（`solve_transport`）：
   静电耦合在全体缺陷的耦合矩阵（`IncrementalCoupling` 逐个追加的那份）上消去导体成员（`ConductorCoupling`），导体作为
   费米能级处的电子库与近旁陷阱交换电子（`kmc.Conductors`）；从上一步的占据热启动，求簇级平均场不动点。近邻表、强耦合簇
   与截断半径都随缺陷集合从头重算（ADR-033）。放热（式 T13，`thermal.heat_release`）：陷阱处是陷阱的放热，导体成员处是
   陷阱与导体交换、导体与对面电极之间的间隙隧穿（`gap_tunneling`）和贯穿通路细丝的放热，隧穿到导体上的电子的能量都放在
   导体里。介质内的功率解稳态热方程（式 T14，两电极恒温 T_amb）得到温度场。配置了 k_TH 时输运按缺陷处的局域温度求值，
   "输运 → 功率 → 温度场 → 缺陷处温度"迭代到缺陷处温度的变化 ≤ `_THERMAL_TOLERANCE`；迭代 `_THERMAL_MAX_ITERATIONS` 次
   仍不收敛（热失控）时记下这一点、取最后一次的结果。没有 k_TH 时温度固定为环境值（Phase 3 的等温口径）。
3. **击穿判据**（§6.3，ADR-054/060）：负极侧与正极侧导电簇连通、形成贯穿通路即击穿，第一条这样的记录给出 t_BD。
   贯穿通路按一个量子电导 G₀ = 2e²/h 导通，细丝电流 G₀·V，功率 G₀·V² 平分到通路的空位上。击穿之后再跑
   `tddb.post_breakdown_steps` 步生成看趋势。论文的"25 个空位、间距 ≤ 6 Å"团簇只作标志记录，不作判据。
4. **生成率**（§6.1，`generation_rates`）：陷阱净电荷 q·(Z − p) + q_fix 与外加偏压一起解泊松方程，接电极的
   导电簇保持导体电势（ADR-054），悬空导电簇保持总电荷、自身等电位（ADR-056）；在每个前驱位点上求式 T1 的
   G_i = R_1e,i·P_bb,i：
   - R_1e,i = 两侧电极 R2 + R4（电极项，式 T4）+ Σ_j p_j·R7(j → i)（陷阱项：近旁已占据陷阱经陷阱间通道
     送来的电子，ADR-039；接电极导体的空位按所在处费米能级占满的电子源计入）。前驱能级、R4 的局域场与
     温度取位点处含空间电荷的局域值，取法同引擎处理真实陷阱；电极项在 (z, 前驱能级, 1/温度) 的插值表上
     求值（ADR-055/060）；
   - P_bb,i 用位点自己的 E_A,i，场取泊松解在 Gaussian 窗口上的平均（宏观场，ADR-043）。导电簇尖端前方的场被抬高，生成
     向尖端集中，导电簇越长尖端场越强，这是贯穿通路最后一段迅速形成的机制。
5. **抽事件**：Δt = −ln(r)/ΣG，按 G_i 选一个前驱位点断键，新空位就落在这个位点上。这个位点用掉，
   其余位点离它本来就 ≥ min_separation，原样保留。
6. **出生**：新空位与初始缺陷同一物种（能级按缺陷预设的分布抽样，S、m_di 取预设），出生占据 1，
   施主约定 `charge_state` = +1（§6.7、ADR-035）。生成反应"前驱 + e⁻ → V_O + O_i⁻"按电荷守恒处理
   （ADR-048）：前驱俘获的那个电子的电荷留在 O 间隙离子上，O_i⁻ 作为新空位的固定电荷
   q_fix = −e，与空位重合放置（`DefectEnsemble.fixed_charge_offset` = 0）。新空位与 O_i⁻ 合起来出生时
   带 −e，就是被俘获的那个电子；空位放出电子后合起来为中性。静电耦合增量扩一行一列（§6.4），回到 1。

击穿后再跑的步数用完时停止；此外按生成空位数上限或应力时间上限停止，前驱用完时按"总生成率为 0"停止。每步开始时的
全部状态是 `TDDBState`，可以存盘（`save_state` / `load_state`），交给 `run_tddb` 的 `start` 接着跑（ADR-057）。

**出生时的电子**：新空位出生时带的那个电子来自前驱俘获，不计入输运电流（§6.7）；它的电荷随新空位与
O_i⁻ 一起留在介质里。平均场没有时间推进，每条记录的电流就是该缺陷集合下的稳态电流：内层输运（TAT、PF 与 R1）、
间隙隧穿与细丝电流之和。

**悬空导电簇只进生成率**（ADR-056）：内层输运把悬空导电簇的空位当作 TAT 陷阱；接电极的导电簇在输运里是导体（ADR-060）。
"""

import copy
import csv
import itertools
import json
from dataclasses import dataclass, fields, replace
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy import constants
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .bands import (
    positive_fermi_level,
    conduction_band_edge,
    electrode_band_bottom,
    r1_band_bottom,
    top_boundary_potential,
)
from .clusters import coupling_strength
from .config import Config, DeviceConfig, TDDBConfig
from .defects import (
    DefectEnsemble,
    _global_standoff,
    append_defects,
    generate_defects,
    layer_bulk_defects,
    localization_radius,
    minimum_image_delta,
    neighbor_pairs,
)
from .electrostatics import PoissonSolver
from .kmc import (
    _CLUSTER_COUPLING,
    _CLUSTER_MAX_SIZE,
    _TRAP_PATH_POINTS,
    MECHANISMS,
    Conductors,
    ElectrostaticCoupling,
    KMCSimulation,
    _band_on_path_steps,
    _interface_step_nodes,
)
from .mean_field import MeanFieldResult, solve_mean_field
from .rates import (
    CaptureElectrode,
    bond_break_probability,
    r1_electrode_electrode,
    single_electron_capture_parts,
    trap_precursor_capture_rate,
)
from .thermal import HeatRelease, heat_release, power_map, solve_temperature

# 前驱俘获 R4 的声子项数：E_rel1 = 1.0 eV（E_rel1/ℏω₀ = 25）时 80 项最大相对误差 1e-4、
# 100 项到机器精度（ADR-030）
_PRECURSOR_N_PHONON = 100
# 生成反应里 O 间隙离子带的电荷：前驱俘获的那个电子（ADR-048）。作为新空位的固定电荷 q_fix，与空位重合
_INTERSTITIAL_CHARGE = -constants.elementary_charge
# 论文的致密团簇标志"25 个空位、间距 ≤ 6 Å"（§6.3）：只记录最大团簇的大小与首次达到的时刻，
# 不作击穿判据（那是针对 ~2.7 nm 膜定的）
_DENSE_CLUSTER_DISTANCE = 0.6 * constants.nano
_DENSE_CLUSTER_SIZE = 25
# 贯穿通路形成后细丝的电导：一个量子电导 G₀ = 2e²/h（量子点接触，ADR-054）
_QUANTUM_CONDUCTANCE = 2.0 * constants.elementary_charge ** 2 / constants.h
# 输运与热的迭代（ADR-060，`_steady_state`）：缺陷处温度的变化 ≤ 0.01 K 即收敛（E = 1 eV 的热激活速率在 300 K 下变 0.13%）；
# 迭代次数上限，到了仍不收敛记为热失控
_THERMAL_TOLERANCE = 0.01  # [K]
_THERMAL_MAX_ITERATIONS = 50
# 间隙隧穿（ADR-060，`gap_tunneling`）路径上的采样点数，同电极 → 陷阱的隧穿路径
_GAP_PATH_POINTS = _TRAP_PATH_POINTS
# R_1e 电极项的插值表（`_electrode_capture`，ADR-055）：z、前驱能级、1/温度三个方向的节点间距。3.6 V 默认工况下与逐点
# 求值相比，按生成率加权的相对误差 2.5×10⁻⁴，生成率不可忽略的位点上最大 1.1×10⁻³。温度方向按 1/T 等距（300 K 处
# 相当于 0.5 K），热激活因子 exp(−E/k_BT) 的插值误差在各温度下相同；击穿后热点附近的位点温度跨几千 K 时节点数仍有限
# （ADR-060）
_TABLE_Z_STEP = 0.05 * constants.nano
_TABLE_ENERGY_STEP = 0.005 * constants.electron_volt
_TABLE_INVERSE_TEMPERATURE_STEP = 0.5 / 300.0 ** 2  # [1/K]
# 分块求表节点：每块的节点数，块内 (节点, 声子数, 路径点) 的数组约 3.3 MB
_TABLE_CHUNK = 64
# 陷阱项的近邻截断（`_trap_capture`）：截断外每个前驱被丢掉的陷阱项总速率不超过此值 [1/s]（MA 因子与
# 源陷阱占据都取 1 的上界）。应力时间上限 10⁴ s 内，这部分让任何一个前驱断键的概率变化不超过 10⁻⁸
_TRAP_TERM_FLOOR = 1.0e-12
# 陷阱项按源陷阱分块求和：每块的源陷阱数，块内 (位点, 源) 对的数组不超过几 MB
_SOURCE_CHUNK = 32
# 前驱位点的最大随机堆积（`_maximal_packing`）：每批扔的随机点数（批内两两冲突的点对数组随批大小增长），
# 以及格子细分的层数上限（格边细到 2⁻³⁰ 时剩下的空隙体积可以忽略）
_PACKING_BATCH = 250000
_PACKING_MAX_LEVEL = 30


class IncrementalCoupling:
    """可逐个追加缺陷的静电耦合矩阵（PROJECT_CONTEXT_TDDB.md §6.4）。

    属性与 `kmc.ElectrostaticCoupling` 相同，直接交给 `KMCSimulation` 用。初始缺陷集合用
    `build` 按 `ElectrostaticCoupling` 算（逐位相同），之后 `extended` 每追加一个体缺陷做
    4 次泊松求解：

    - 1 次正向求解：新缺陷单位电荷的解，给出它在全部缺陷处的电势与电场（新行）、横向平均
      剖面与自作用项。已有缺陷在新位置的电势（新列）按互易性取新行：泊松算子对称、CIC 的
      分配与插值互为转置，所以 `phi_matrix` 对称；
    - 3 次伴随求解：已有缺陷在新位置产生的场不对称，三个分量各用读场泛函的伴随源
      （`PoissonSolver.field_readout_sources`）解一次，再在已有缺陷处读值。

    矩阵按精确尺寸存放，`extended` 返回带 N+1 维新数组的新对象、原对象不变：交给
    `KMCSimulation` 的始终是连续数组，已建好的 `KMCSimulation` 也不会被后来的追加改动（ADR-033）。
    """

    def __init__(
        self,
        solver: PoissonSolver,
        positions: np.ndarray,
        field_side: np.ndarray,
        phi_matrix: np.ndarray,
        field_matrix: np.ndarray,
        phi_bar: np.ndarray,
    ) -> None:
        self._solver = solver
        self._positions = positions
        self._field_side = field_side
        self.phi_matrix = phi_matrix
        self.field_matrix = field_matrix
        self.phi_bar = phi_bar
        self.phi_self = np.diag(phi_matrix).copy()
        idx = np.arange(positions.shape[0])
        self.field_self = field_matrix[:, idx, idx].copy()

    @classmethod
    def build(cls, solver: PoissonSolver, positions: np.ndarray, field_side: np.ndarray) -> "IncrementalCoupling":
        """初始缺陷集合的耦合矩阵，与 `ElectrostaticCoupling` 逐位相同。"""
        base = ElectrostaticCoupling(solver, positions, field_side)
        return cls(
            solver,
            np.array(positions, dtype=float).reshape(-1, 3),
            np.array(field_side, dtype=int),
            base.phi_matrix,
            base.field_matrix,
            base.phi_bar,
        )

    def extended(self, position: np.ndarray) -> "IncrementalCoupling":
        """追加一个体缺陷（读场侧 0、编号接在已有缺陷之后）后的耦合矩阵。"""
        solver = self._solver
        q = constants.elementary_charge
        point = np.asarray(position, dtype=float).reshape(1, 3)
        n = self._positions.shape[0]
        positions = np.vstack([self._positions, point])
        field_side = np.append(self._field_side, 0)

        # 正向：新缺陷上一个电子（−q）的解
        phi = solver.solve(solver.deposit(point, np.array([-q])), 0.0)
        phi_row = solver.interpolate(phi, positions)
        field_row = solver.field_vectors_at(phi, positions, field_side)

        # 伴随：已有缺陷 j 的 −q 在新位置的场 E_c = −⟨ψ_c, ρ_j⟩，ρ_j = −q·w_j/V_cell
        scale = q / (solver.hx * solver.hy * solver.hz)
        field_col = np.empty((3, n))
        for c, source in enumerate(solver.field_readout_sources(point)):
            field_col[c] = scale * solver.interpolate(solver.solve(source, 0.0), self._positions)

        phi_matrix = np.empty((n + 1, n + 1))
        phi_matrix[:n, :n] = self.phi_matrix
        phi_matrix[n, :] = phi_row
        phi_matrix[:n, n] = phi_row[:n]
        field_matrix = np.empty((3, n + 1, n + 1))
        field_matrix[:, :n, :n] = self.field_matrix
        field_matrix[:, n, :] = field_row
        field_matrix[:, :n, n] = field_col
        phi_bar = np.vstack([self.phi_bar, solver.lateral_average(phi)[None, :]])
        return IncrementalCoupling(solver, positions, field_side, phi_matrix, field_matrix, phi_bar)


@dataclass(frozen=True)
class Precursors:
    """还没断键的离散前驱位点（§6.1，ADR-052/055）。

    前驱是 O-Zr-O 键上的一个位置，俘获一个电子后可能断键、原地变成 O 空位（式 T1），断一次就用掉。
    位点是最小间隔 min_separation 的最大随机堆积（`place_precursors`），位置连续（§6.8）；每个位点有自己的无场断键
    激活能 E_A,i，体现非晶里键环境的差别（ADR-063）。
    """

    positions: np.ndarray  # (M, 3) 位点坐标 [m]
    activation_energy: np.ndarray  # (M,) 各位点的无场断键激活能 E_A,i [J]

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def converted(self, index: int) -> "Precursors":
        """第 `index` 个位点断键、原地变成新空位之后剩下的位点：只去掉它自己。其余位点离它本来就
        ≥ min_separation，都还能放下新空位。"""
        return Precursors(np.delete(self.positions, index, axis=0), np.delete(self.activation_energy, index))


def place_precursors(
    device: DeviceConfig,
    defects: np.ndarray,
    activation_energy: float,
    activation_tail: float,
    rng: np.random.Generator,
) -> Precursors:
    """t = 0 的前驱位点（§6.1，ADR-052/055/063）。

    前驱位点不设浓度，由最小间隔 d = min_separation 决定（取 ZrO₂ 的 O–O 间距）：在可采样窗口
    [0,Lx)×[0,Ly)×[d, Lz−d]（与 `generate_defects` 的电极避让相同）里做最大随机堆积（`_maximal_packing`），
    位点两两之间、位点与初始缺陷 `defects`（坐标 (N, 3)）之间都 ≥ d，且窗口里再也放不下一个位点。
    d = 0.26 nm 时密度约 4.3×10²² cm⁻³。铺完之后每个位点抽自己的无场断键激活能 E_A,i = E_A − E₀·X_i
    （`activation_energy` = E_A 是上限，`activation_tail` = E₀，X_i 服从单位指数分布）：低 E_A 一侧的位点数按
    exp[(E − E_A)/E₀] 衰减，位点铺满时生成数 N ∝ t^(k_BT/E₀)。随机数的消耗与 E₀ 的取值无关。
    """
    standoff = _global_standoff(device)
    if not standoff > 0.0:
        raise ValueError("前驱位点按 min_separation 堆积，min_separation_nm 必须为正")
    if device.Lz - 2.0 * standoff <= 0.0:
        raise ValueError("min_separation 约束下 z 方向没有可放置新空位的区间")
    positions = _maximal_packing(device, standoff, defects, rng)
    return Precursors(positions, activation_energy - activation_tail * rng.standard_exponential(positions.shape[0]))


# 一个格子切成 8 个子格时各子格的下角（以子格边长为单位）
_CHILD_CORNERS = np.array(list(itertools.product((0.0, 1.0), repeat=3)))


def _maximal_packing(
    device: DeviceConfig, separation: float, obstacles: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """可采样窗口 [0,Lx)×[0,Ly)×[d, Lz−d]（d = `separation`）里的最大随机堆积，返回 (M, 3) 坐标（ADR-055）。

    位点两两之间、位点与 `obstacles`（已有缺陷，(N, 3)）之间都 ≥ d（xy 最小镜像），且窗口里再也放不下一个这样的
    位点。做法是最大泊松圆盘抽样（Ebeida 等 2012）：顶层格子的对角线 ≤ d，每格最多一个位点。每一层在还空着的
    格子里各扔一个随机点，离已有的点或同批里排在前面的点 < d 的扔掉；然后去掉落了位点的格子与整个落在某个点
    d 球里的格子，其余各切成 8 个子格进下一层。没有格子剩下时，窗口里每一点离某个位点或已有缺陷都 ≤ d。
    """
    d = separation
    extent = np.array([device.Lx, device.Ly, device.Lz - 2.0 * d])
    shape = np.ceil(extent * np.sqrt(3.0) / d).astype(int)
    size = extent / shape
    # xy 周期；z 不周期：盒长取 4·Lz，点都在 [0, Lz] 内，不会回绕
    box = np.array([device.Lx, device.Ly, 4.0 * device.Lz])
    filled = np.zeros(int(np.prod(shape)), dtype=bool)  # 顶层格子里是否已有位点
    top = np.arange(filled.size)  # 各活动格所在的顶层格子
    lower = np.column_stack(np.unravel_index(top, shape)) * size + np.array([0.0, 0.0, d])  # 各活动格的下角
    points = np.mod(np.asarray(obstacles, dtype=float).reshape(-1, 3), box)
    n_obstacles = points.shape[0]
    tree = cKDTree(points, boxsize=box)

    def covered(corner: np.ndarray, side: np.ndarray) -> np.ndarray:
        """各格 [corner, corner + side] 是否整个落在离格心最近的那个点的 d 球里（最远的角离它 ≤ d）。"""
        center = corner + 0.5 * side
        _, nearest = tree.query(center)
        delta = np.abs(center - points[nearest])
        delta[:, :2] = np.minimum(delta[:, :2], box[:2] - delta[:, :2])
        return np.sum((delta + 0.5 * side) ** 2, axis=1) <= d * d

    for _ in range(_PACKING_MAX_LEVEL + 1):
        if top.size == 0:
            break
        order = rng.permutation(top.size)
        for start in range(0, order.size, _PACKING_BATCH):
            batch = order[start:start + _PACKING_BATCH]
            batch = batch[~filled[top[batch]]]
            darts = lower[batch] + rng.random((batch.size, 3)) * size
            darts[:, :2] = np.mod(darts[:, :2], box[:2])
            distance, _ = tree.query(darts, distance_upper_bound=d)
            batch, darts = batch[distance >= d], darts[distance >= d]
            # 同批里两点冲突时留下排在前面的（批内顺序随机）
            accept = np.ones(batch.size, dtype=bool)
            accept[cKDTree(darts, boxsize=box).query_pairs(d, output_type="ndarray").max(axis=1)] = False
            filled[top[batch[accept]]] = True
            points = np.vstack([points, darts[accept]])
            tree = cKDTree(points, boxsize=box)
        keep = ~filled[top]
        top, lower = top[keep], lower[keep]
        keep = ~covered(lower, size)
        top, lower = top[keep], lower[keep]
        size = 0.5 * size
        tops, lowers = [], []
        for start in range(0, top.size, _PACKING_BATCH // 8):
            children = (lower[start:start + _PACKING_BATCH // 8, None, :] + _CHILD_CORNERS * size).reshape(-1, 3)
            keep = ~covered(children, size)
            tops.append(np.repeat(top[start:start + _PACKING_BATCH // 8], 8)[keep])
            lowers.append(children[keep])
        top = np.concatenate(tops) if tops else np.zeros(0, dtype=int)
        lower = np.concatenate(lowers) if lowers else np.zeros((0, 3))
    return points[n_obstacles:]


def _capture_electrodes(
    device: DeviceConfig, solver: PoissonSolver, phi: np.ndarray, voltage: float, z_values: np.ndarray
) -> List[CaptureElectrode]:
    """前驱位于各高度 `z_values` 时两侧电极到它的隧穿路径，取法同 kmc.py 对真实陷阱的处理：路径沿 z 直线，
    势垒取横向平均剖面的导带底（含界面台阶）。每侧的数组形状 (len(z_values), _TRAP_PATH_POINTS)。"""
    band = conduction_band_edge(device, solver.z, solver.lateral_average(phi))
    iface_idx, delta_Ec = _interface_step_nodes(device, solver.z)
    t = np.linspace(0.0, 1.0, _TRAP_PATH_POINTS)
    z = np.asarray(z_values, dtype=float)[:, None]
    sides = []
    for path, electrode, fermi in (
        (z * t[None, :], device.negative, 0.0),
        (z + (device.Lz - z) * t[None, :], device.positive, positive_fermi_level(voltage)),
    ):
        sides.append(
            CaptureElectrode(
                path=path,
                barrier=_band_on_path_steps(path, solver.z, band, iface_idx, delta_Ec),
                m_di_path=device.stack.m_di_at(path),
                fermi_level=fermi,
                band_bottom=electrode_band_bottom(fermi, electrode.fermi_energy),
                mass=electrode.effective_mass,
            )
        )
    return sides


class GenerationRates(NamedTuple):
    """一个状态下各前驱位点的生成速率分量（式 T1，§6.1，ADR-052），与 `Precursors` 逐位点对应。"""

    electrode: np.ndarray  # (M,) R_1e 电极项：两侧电极 R2 + R4 [1/s]
    trap: np.ndarray  # (M,) R_1e 陷阱项：Σ_j p_j·R7(j → i) [1/s]
    bond_breaking: np.ndarray  # (M,) 断键概率 P_bb,i

    @property
    def rates(self) -> np.ndarray:
        """各位点的生成速率 G_i = R_1e,i·P_bb,i [1/s]。"""
        return (self.electrode + self.trap) * self.bond_breaking


def generation_rates(
    transport: "TransportState",
    tddb: TDDBConfig,
    precursors: Precursors,
    temperature_field: Optional[np.ndarray] = None,
) -> GenerationRates:
    """各前驱位点的 O 空位生成速率 G_i = R_1e,i·P_bb,i（式 T1/T2/T4，§6.1，ADR-052）。

    `transport` 是当前缺陷集合上的内层输运稳态（`solve_transport`）：导体划分、陷阱的平均场占据与能级、工作点与
    泊松求解器。电势由 `_conductor_potential` 给出：陷阱随占据变的电荷 q·(Z − p)（放在缺陷处）与固定电荷 q_fix
    （放在固定电荷处）和外加偏压一起解泊松方程；导电簇（按离域判据连边，`conducting_clusters`）是等电位的导体：接电极的
    保持导体电势（ADR-054/060），悬空的保持总电荷、电势待定（ADR-056）。各位点在这份电势上求值，取法同引擎处理
    真实陷阱：前驱能级 E = B − e·φ(x) − E_t1，R4 的 c_MP 用局域场。

    - **电极项**（`_electrode_capture`）：两侧电极 R2 + R4；
    - **陷阱项**（`_trap_capture`）：近旁已占据陷阱经陷阱间通道 p·R7 送给前驱的电子（式 T5 的陷阱来源，
      ADR-039）。接电极导体的空位按所在处费米能级占满（p = 1）的电子源计入；其余陷阱（含悬空导电簇的成员）按平均场
      占据计入，能级取同一份电势（引擎的平均场能级加上悬空导电簇等电位带来的电势改变）；
    - **P_bb**：用位点自己的 E_A,i（ADR-063）；场是泊松解的场在标准差 `tddb.field_averaging_width` 的 Gaussian
      窗口上的平均（宏观场，ADR-043）。Lorentz 关系 (2+k)/3 针对在 Lorentz 球上近似均匀的宏观场，带电缺陷亚纳米
      近旁的点电荷场不满足这个前提。

    `temperature_field` 是泊松网格上的温度场（热反馈时取 T14 的解，§6.6）：R_1e 的声子线型、电极费米分布、
    陷阱项的 MA 因子与 P_bb 都取位点处的温度。缺省用工作点温度。
    """
    inner, defects, layout = transport.sim, transport.defects, transport.layout
    device, solver = inner.device, inner.solver
    material = device.stack.single_material
    n = len(precursors)
    if n == 0:
        return GenerationRates(np.zeros(0), np.zeros(0), np.zeros(0))
    phi, shift = _conductor_potential(transport)
    field = np.stack(solver.electric_field(phi), axis=-1)
    bond_field = solver.gaussian_average(field, tddb.field_averaging_width)
    # 要读的网格量叠成一个多通道数组，每个位点一次插值读回：φ、局域场三个分量、P_bb 宏观场三个分量、
    # （热反馈时）温度
    grid = np.concatenate(
        (phi[..., None], field, bond_field) + (() if temperature_field is None else (temperature_field[..., None],)),
        axis=-1,
    )
    points = precursors.positions
    local = solver.interpolate(grid, points)
    conduction = device.band_reference_profile(points[:, 2]) - constants.elementary_charge * local[:, 0]
    temperature = local[:, 7] if temperature_field is not None else np.full(n, inner.sim.temperature)
    electrode = _electrode_capture(
        inner, tddb, phi, conduction, np.linalg.norm(local[:, 1:4], axis=1), points[:, 2], temperature
    )
    member = layout.sides >= 0
    # 接电极导体的成员：引擎能级就是所在处的费米能级（`solve_transport`），按占满的电子源计入
    source_energy = np.where(member, transport.energy, transport.energy - constants.elementary_charge * shift)
    trap = _trap_capture(
        defects, device, tddb, points, conduction - tddb.precursor_ionization_energy, source_energy,
        np.where(member, 1.0, transport.occupancy), temperature,
    )
    p_bb = bond_break_probability(
        precursors.activation_energy, tddb.dipole_moment, material.kappa, np.linalg.norm(local[:, 4:7], axis=1),
        temperature,
    )
    return GenerationRates(electrode, trap, p_bb)


def _conductor_potential(transport: "TransportState") -> Tuple[np.ndarray, np.ndarray]:
    """生成率用的电势 φ（泊松网格节点，[V]）与悬空导电簇带来的各缺陷处电势改变 (N,) [V]（ADR-054/056/060）。

    导电簇编号与接哪侧电极取 `transport.layout`。不在导电簇里的缺陷带 q·(Z − p) + q_fix，固定电荷放在各自的位置。两个及
    以上空位连成的导电簇是导体，成员的电荷不再由占据决定，取让全部成员处电势满足下面条件的那一组电荷：
    - 接电极的导体（`sides` ≥ 0）：成员处的电势等于导体电势（`conductor_layout`：接负极 0、接正极 `top_boundary_potential`，
      贯穿通路沿 z 线性）；
    - 悬空导电簇（`sides` = −1）：电势是待定的常数，簇的总电荷保持引擎给的 Σ[q·(Z − p) + q_fix,重合]。
    成员处的电势对这组电荷是线性的，系数就是耦合矩阵 `phi_matrix`（同一套沉积、求解与插值）；悬空导电簇每个多一个电势
    未知数和一条总电荷方程，在全部成员上解一次线性方程即得。离开成员位置的固定电荷照常放置。只有一个空位的悬空
    "导电簇"按这个规则电荷不变，与不在导电簇里相同。

    第二项是电势改变：引擎的平均场能级减去 e·(这一项) 就是同一份电势下的陷阱能级。其余缺陷处是成员电荷从引擎的电荷
    （悬空导电簇成员是 q·(Z − p) + q_fix,重合，接电极导体的成员是输运不动点上的导体电荷）换成这里的导体电荷带来的
    改变；成员处取这里的电势与引擎电势之差。没有导电簇时电势与只放缺陷电荷的解相同，第二项全为 0。
    """
    inner, defects, layout = transport.sim, transport.defects, transport.layout
    solver, sides, labels = inner.solver, layout.sides, layout.labels
    occupancy = transport.occupancy
    q = constants.elementary_charge
    top = top_boundary_potential(inner.device, inner.sim.voltage)
    charge = q * (defects.charge_state - occupancy)
    conductor = np.bincount(labels, minlength=1)[labels] >= 2
    conductor |= sides >= 0  # 单个空位离电极小于自身离域距离也接到电极
    members = np.flatnonzero(conductor)
    if members.size == 0:
        rho = solver.deposit(
            np.concatenate([defects.positions, defects.fixed_charge_positions]),
            np.concatenate([charge, defects.fixed_charge]),
        )
        return solver.solve(rho, top), np.zeros(len(defects))
    free = ~conductor
    # 成员自己的电荷（随占据变的部分与重合的固定电荷）由导体电荷代替；离开成员的固定电荷是另一个点电荷，保留
    colocated = defects.colocated_fixed_charge
    rho = solver.deposit(
        np.concatenate([defects.positions[free], defects.fixed_charge_positions]),
        np.concatenate([charge[free], np.where(free, defects.fixed_charge, defects.fixed_charge - colocated)]),
    )
    phi = solver.solve(rho, top)
    green = transport.coupling.phi_matrix  # 一个电子（−q）在 j 处产生的电势在 i 处的值，对称
    m = members.size
    # 成员在引擎里的电子数 (p − Z) − q_fix,重合/q（接电极导体的成员取输运不动点上的导体电子数）；悬空导电簇的导体电子数之和等于它
    engine_electrons = -(charge + colocated)[members] / q
    floating = np.unique(labels[members][sides[members] < 0])
    # (m, 悬空簇数)：成员属于哪个悬空导电簇
    belongs = ((labels[members][:, None] == floating[None, :]) & (sides[members] < 0)[:, None]).astype(float)
    system = np.block([[green[np.ix_(members, members)], -belongs], [belongs.T, np.zeros((floating.size,) * 2)]])
    rhs = np.concatenate([
        np.where(sides[members] >= 0, transport.potential[members], 0.0) - solver.interpolate(phi, defects.positions[members]),
        belongs.T @ engine_electrons,
    ])
    electrons = np.linalg.solve(system, rhs)[:m]
    phi = solver.solve(rho + solver.deposit(defects.positions[members], -q * electrons), top)
    shift = (electrons - engine_electrons) @ green[members]
    shift[members] = solver.interpolate(phi, defects.positions[members]) - transport.potential[members]
    return phi, shift


def _electrode_capture(
    inner: KMCSimulation,
    tddb: TDDBConfig,
    phi: np.ndarray,
    conduction: np.ndarray,
    field: np.ndarray,
    z: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    """各前驱位点的 R_1e 电极项：两侧电极 R2 + R4 [1/s]（式 T4）。`conduction` 是位点处的导带底 B − e·φ，
    `field` 是局域场大小，隧穿路径与势垒见 `_capture_electrodes`。

    R_1e,电极 = Σ R2 + F²·Σ R4/F²（`single_electron_capture_parts`），两部分只依赖位点的 z、前驱能级与温度。完整的
    WKB 与声子求和只在 (z, 前驱能级, 1/温度) 的等距表节点上做，节点间距 `_TABLE_Z_STEP`、`_TABLE_ENERGY_STEP`、
    `_TABLE_INVERSE_TEMPERATURE_STEP`，只求位点用得到的节点；位点上对两部分的 ln 值做三线性插值，再按局域场组合
    （ADR-055/060）。全部位点在某个方向上取值相同时这个方向只有一个节点，单个位点就是逐点求值。
    """
    device = inner.device
    material = device.stack.single_material
    energy = conduction - tddb.precursor_ionization_energy
    axes = [
        _table_axis(z, _TABLE_Z_STEP),
        _table_axis(energy, _TABLE_ENERGY_STEP),
        _table_axis(1.0 / np.asarray(temperature, dtype=float), _TABLE_INVERSE_TEMPERATURE_STEP),
    ]
    shape = tuple(nodes.size for nodes, _, _ in axes)
    # 每个位点的 8 个角：表节点编号（三个方向展平）与三线性权重
    keys, weights = [], []
    for corner in itertools.product((0, 1), repeat=3):
        keys.append(np.ravel_multi_index(
            [np.minimum(left + c, nodes.size - 1) for (nodes, left, _), c in zip(axes, corner)], shape
        ))
        weights.append(np.prod([fraction if c else 1.0 - fraction for (_, _, fraction), c in zip(axes, corner)], axis=0))
    needed, inverse = np.unique(np.concatenate(keys), return_inverse=True)
    z_node, energy_node, temperature_node = np.unravel_index(needed, shape)
    sides = _capture_electrodes(device, inner.solver, phi, inner.sim.voltage, axes[0][0])
    tiny = np.finfo(float).tiny
    log_parts = np.empty((2, needed.size))
    # 同一个 z 节点、同一个温度节点的表节点共用隧穿路径与声子线型，按能级分块求
    group = z_node * shape[2] + temperature_node
    order = np.argsort(group, kind="stable")
    for nodes in np.split(order, np.flatnonzero(np.diff(group[order])) + 1):
        k, j = z_node[nodes[0]], temperature_node[nodes[0]]
        for start in range(0, nodes.size, _TABLE_CHUNK):
            rows = nodes[start:start + _TABLE_CHUNK]
            log_parts[:, rows] = np.log(np.maximum(single_electron_capture_parts(
                axes[1][0][energy_node[rows]] + tddb.precursor_ionization_energy,
                [side._replace(
                    path=np.broadcast_to(side.path[k], (rows.size, side.path.shape[1])),
                    barrier=np.broadcast_to(side.barrier[k], (rows.size, side.barrier.shape[1])),
                    m_di_path=np.broadcast_to(side.m_di_path[k], (rows.size, side.m_di_path.shape[1])),
                ) for side in sides],
                tddb.precursor_ionization_energy,
                tddb.precursor_relaxation_energy,
                material.phonon_energy,
                material.band_gap,
                material.m_di,
                1.0 / float(axes[2][0][j]),
                _PRECURSOR_N_PHONON,
            ), tiny))
    n = z.size
    log_rate = np.zeros((2, n))
    for c, weight in enumerate(weights):
        log_rate += weight * log_parts[:, inverse[c * n:(c + 1) * n]]
    elastic, multiphonon = np.exp(log_rate)
    return elastic + field ** 2 * multiphonon


def _table_axis(values: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """插值表的一个方向：[最小值, 最大值] 上间距 ≤ `step` 的等距节点，返回 (节点, 各值所在区间的左节点编号,
    在区间内的位置 ∈ [0, 1])。全部值相同时只有一个节点，位置记 0。"""
    lo, hi = float(values.min()), float(values.max())
    if hi == lo:
        return np.array([lo]), np.zeros(values.size, dtype=int), np.zeros(values.size)
    nodes = np.linspace(lo, hi, int(np.ceil((hi - lo) / step)) + 1)
    position = (values - lo) / (nodes[1] - nodes[0])
    left = np.clip(np.floor(position).astype(int), 0, nodes.size - 2)
    return nodes, left, np.clip(position - left, 0.0, 1.0)


def _trap_capture(
    defects: DefectEnsemble,
    device: DeviceConfig,
    tddb: TDDBConfig,
    points: np.ndarray,
    precursor_energy: np.ndarray,
    source_energy: np.ndarray,
    occupancy: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    """各前驱位点的 R_1e 陷阱项 Σ_j p_j·R7(j → i) [1/s]（式 T4/T5 的陷阱来源，ADR-039）。

    源是全体缺陷 `defects`。R7 是引擎的陷阱间通道把目标换成前驱：ν_j·exp[−r·(1/r_D,j + 1/r_D,前驱)]·Θ，能量差取前驱能级与
    源能级 `source_energy` 之差（ADR-038），温度取前驱处的温度。`occupancy` 是源的占据 p_j（接电极导体的空位取 1）。只对
    p_j > 0 的源求和，距离取 xy 最小镜像。截断半径让截掉的部分（Θ 与 p 都取 1 的上界）每个前驱不超过
    `_TRAP_TERM_FLOOR`，且不超过横向盒长的一半。源陷阱按 `_SOURCE_CHUNK` 个一块求和。
    """
    p = np.asarray(occupancy, dtype=float)
    sources = np.flatnonzero(p > 0.0)
    rate = np.zeros(points.shape[0])
    if sources.size == 0:
        return rate
    precursor_r_D = localization_radius(tddb.precursor_ionization_energy, device.stack.single_material.m_di)
    alpha_min = 1.0 / float(defects.r_D[sources].max()) + 1.0 / float(precursor_r_D)
    nu_max = float(defects.attempt_frequency[sources].max())
    reach = min(np.log(sources.size * nu_max / _TRAP_TERM_FLOOR) / alpha_min, 0.5 * min(device.Lx, device.Ly))
    box = np.array([device.Lx, device.Ly, 4.0 * device.Lz])
    tree = cKDTree(np.mod(points, box), boxsize=box)
    for start in range(0, sources.size, _SOURCE_CHUNK):
        chunk = sources[start:start + _SOURCE_CHUNK]
        pairs = tree.sparse_distance_matrix(
            cKDTree(np.mod(defects.positions[chunk], box), boxsize=box), reach, output_type="ndarray"
        )
        site, source = pairs["i"], chunk[pairs["j"]]
        hop = trap_precursor_capture_rate(
            defects.attempt_frequency[source],
            pairs["v"],
            defects.r_D[source],
            precursor_r_D,
            source_energy[source],
            precursor_energy[site],
            p[source],
            temperature[site],
        )
        rate += np.bincount(site, hop, minlength=points.shape[0])
    return rate


def _draw_generation(rates: np.ndarray, rng: np.random.Generator) -> Optional[Tuple[float, int]]:
    """抽下一次生成事件，返回 (Δt, 断键的前驱位点编号)；总生成率为 0 时返回 None。"""
    total = float(rates.sum())
    if not total > 0.0:
        return None
    dt = -np.log(rng.random()) / total
    cumulative = np.cumsum(rates)
    index = int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right"))
    return dt, min(index, rates.size - 1)


def _cluster_diagnostics(inner: KMCSimulation) -> Tuple[int, int, int, int]:
    """(强耦合簇个数, 最大簇大小, 切分前最大强耦合分量, 被切开的分量数)，§6.9 第 3 条。

    分量按 ADR-022 的口径连边（全空组态下耦合强度 ≥ `_CLUSTER_COUPLING`）；超过
    `_CLUSTER_MAX_SIZE` 的分量由 `coupled_clusters` 在最弱处逐级切开，切口两侧的关联丢失。
    没有簇时最大簇大小记 1（全是孤立陷阱），没有缺陷时记 0。
    """
    n = inner.n_defects
    members = inner.cluster_members
    max_cluster = max((m.size for m in members), default=min(n, 1))
    if n < 2:
        return len(members), max_cluster, n, 0
    pairs = inner._pairs
    blocks, hop = inner._rates(np.zeros(n, dtype=bool))
    keep = coupling_strength(pairs.a, pairs.b, hop, blocks.sum(axis=0)) >= _CLUSTER_COUPLING
    graph = coo_matrix(
        (np.ones(int(keep.sum())), (pairs.a[keep], pairs.b[keep])), shape=(n, n)
    )
    sizes = np.bincount(connected_components(graph, directed=False)[1])
    return len(members), max_cluster, int(sizes.max()), int(np.sum(sizes > _CLUSTER_MAX_SIZE))


def distance_clusters(positions: np.ndarray, device: DeviceConfig, distance: float) -> np.ndarray:
    """几何团簇：间距 < `distance` 的缺陷连边后的连通分量，返回各缺陷所在分量的编号 (N,)。

    xy 取最小镜像。用于论文的致密团簇标志（间距 ≤ 6 Å），只记录、不作判据；与强耦合簇
    （按速率连边、用于电流估计，ADR-022）是两回事。
    """
    n = positions.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    pairs = neighbor_pairs(positions, device.Lx, device.Ly, distance)
    graph = coo_matrix((np.ones(len(pairs)), (pairs.a, pairs.b)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def conducting_clusters(
    defects: DefectEnsemble, device: DeviceConfig, transfer_integral: float
) -> Tuple[np.ndarray, np.ndarray]:
    """导电簇与各簇接哪侧电极（§6.3，ADR-054/064），返回 (编号 (N,), 接哪侧 (N,))：−1 不接电极，0 接负极，1 接正极，
    2 两侧都接（贯穿通路）。

    按电子离域判据连边：空位 a、b 之间的转移积分 t₀·exp[−r_ab·(1/r_D,a + 1/r_D,b)/2]（`transfer_integral` = t₀，随距离的衰减
    与 R6 的几何因子相同）的两倍大于重组能 λ_ab = S_a·ℏω_a + S_b·ℏω_b 时，电子在两者之间离域，两者属于同一个导电簇；xy 取
    最小镜像。空位离电极平面的距离小于它自身的离域距离 r_D·ln[t₀/(S·ℏω)]（与同种空位成对的判据相同）时接到该电极，所在的
    整个导电簇随之接上。
    """
    n = len(defects)
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    relaxation = defects.huang_rhys * defects.phonon_energy  # 每个空位的弛豫能 S·ℏω
    reach = defects.r_D * np.log(transfer_integral / relaxation)  # 自身离域距离，≤ 0 时不与任何空位或电极连通
    cutoff = float(np.max(defects.r_D)) * float(np.log(transfer_integral / np.min(relaxation)))  # 任一对离域距离的上界
    labels = np.arange(n)
    if cutoff > 0.0 and n > 1:
        pairs = neighbor_pairs(defects.positions, device.Lx, device.Ly, cutoff)
        overlap = 0.5 * pairs.r * (1.0 / defects.r_D[pairs.a] + 1.0 / defects.r_D[pairs.b])
        keep = overlap < np.log(2.0 * transfer_integral / (relaxation[pairs.a] + relaxation[pairs.b]))
        graph = coo_matrix((np.ones(int(keep.sum())), (pairs.a[keep], pairs.b[keep])), shape=(n, n))
        labels = connected_components(graph, directed=False)[1]
    z = defects.positions[:, 2]
    negative = np.isin(labels, labels[z < reach])
    positive = np.isin(labels, labels[device.Lz - z < reach])
    return labels, np.where(negative & positive, 2, np.where(negative, 0, np.where(positive, 1, -1)))


def conductor_gap(positions: np.ndarray, sides: np.ndarray, device: DeviceConfig) -> float:
    """负极侧导体（负极平面与接负极的导电簇）到正极侧导体（正极平面与接正极的导电簇）的最短距离 [m]（ADR-054）。

    `sides` 取 `conducting_clusters` 的结果。有贯穿通路时为 0；否则是两侧导体之间还没离域连通的最短距离。
    """
    if np.any(sides == 2):
        return 0.0
    negative, positive = positions[sides == 0], positions[sides == 1]
    gap = device.Lz
    if negative.size:
        gap = min(gap, device.Lz - float(negative[:, 2].max()))
    if positive.size:
        gap = min(gap, float(positive[:, 2].min()))
    if negative.size and positive.size:
        box = np.array([device.Lx, device.Ly, 4.0 * device.Lz])  # 同 `_maximal_packing`：xy 周期，z 不回绕
        distance, _ = cKDTree(np.mod(positive, box), boxsize=box).query(np.mod(negative, box))
        gap = min(gap, float(distance.min()))
    return gap


class ConductorCoupling:
    """消去接电极导体成员之后陷阱之间的等效静电耦合（ADR-060），属性与 `kmc.ElectrostaticCoupling` 相同。

    导体成员的电势固定：陷阱 i 上一个电子在成员上感应出 δe = −G_MM⁻¹·G_Mi 个电子（G 是全体缺陷的 `phi_matrix`，对称），
    陷阱处的电势、电场与横向平均剖面都加上这些感应电荷的贡献，即在 G 上对成员做 Schur 补。对角元里多出来的是导体对
    陷阱自身电荷的像，与电极的像一样随对角元整项扣除（不考虑镜像力）。
    """

    def __init__(self, phi_matrix: np.ndarray, field_matrix: np.ndarray, phi_bar: np.ndarray) -> None:
        self.phi_matrix = phi_matrix
        self.field_matrix = field_matrix
        self.phi_bar = phi_bar
        self.phi_self = np.diag(phi_matrix).copy()
        idx = np.arange(phi_matrix.shape[0])
        self.field_self = field_matrix[:, idx, idx].copy()


@dataclass(frozen=True)
class ConductorLayout:
    """一个缺陷集合上的导体划分与导体的静电（ADR-060），与温度、占据无关，每个外层步算一次（`conductor_layout`）。

    接电极的导电簇（`sides` ≥ 0，含贯穿通路）是导体，其余缺陷是陷阱。导体成员 m 的电势 V_m = s_m·U_top、费米能级
    μ_m = s_m·μ_正极，s_m 是 `positive_share`：接负极 0、接正极 1、贯穿通路 z_m/L_z（均匀电阻）。成员上的导体电子数随陷阱
    占据线性变：e = `induced` − `response`·(p − Z)。
    """

    labels: np.ndarray  # (N,) 导电簇编号（`conducting_clusters`）
    sides: np.ndarray  # (N,) 导电簇接哪侧电极（`conducting_clusters`）
    traps: np.ndarray  # (F,) 输运里的陷阱（`sides` = −1，含悬空导电簇的成员）在全体缺陷里的编号
    members: np.ndarray  # (M,) 接电极导体的成员编号
    positive_share: np.ndarray  # (M,) s_m
    potential: np.ndarray  # (M,) 成员电势 [V]
    fermi_level: np.ndarray  # (M,) 成员处的费米能级 [J]
    # 陷阱之间的耦合：没有导体时就是全体缺陷的耦合，有导体时是消去成员后的 `ConductorCoupling`
    coupling: object
    # 陷阱占据全取 Z 时导体在各陷阱处的背景 (电势 (F,), 电场矢量 (3, F), 横向平均剖面 (nz+1,))，交给 `kmc.Conductors`
    background: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    induced: np.ndarray  # (M,) 陷阱占据全取 Z 时成员上的导体电子数
    response: np.ndarray  # (M, F) 陷阱上每多一个电子，成员上导体电子数的变化（取负）：G_MM⁻¹·G_MF
    # 外加偏压与固定电荷（与成员重合的除外）在泊松网格上的电势 [V]；没有导体时为 None
    external: Optional[np.ndarray]


def conductor_layout(
    defects: DefectEnsemble,
    coupling: "IncrementalCoupling",
    solver: PoissonSolver,
    voltage: float,
    transfer_integral: float,
) -> ConductorLayout:
    """导体划分（`conducting_clusters`）与导体静电（`ConductorLayout`，ADR-060）。

    `coupling` 是全体缺陷的静电耦合。成员处电势固定为 V_m：外加偏压与固定电荷（与成员重合的固定电荷由导体电荷代替，
    离开成员位置的照常放置）在成员上感应出 `induced` = G_MM⁻¹·(V_M − φ_外(M))，它们与成员上离开成员位置的固定电荷一起
    构成陷阱处的背景；陷阱电荷的感应部分折进等效耦合（`ConductorCoupling`）。
    """
    device = solver.device
    positions = defects.positions
    labels, sides = conducting_clusters(defects, device, transfer_integral)
    members, traps = np.flatnonzero(sides >= 0), np.flatnonzero(sides < 0)
    share = np.where(sides[members] == 2, positions[members, 2] / device.Lz, sides[members].astype(float))
    top = top_boundary_potential(device, voltage)
    if members.size == 0:
        return ConductorLayout(
            labels, sides, traps, members, share, share * top, share * positive_fermi_level(voltage), coupling,
            None, np.zeros(0), np.zeros((0, traps.size)), None,
        )
    green, field, bar = coupling.phi_matrix, coupling.field_matrix, coupling.phi_bar
    green_mm = green[np.ix_(members, members)]
    green_mt = green[np.ix_(members, traps)]
    field_mt = field[:, members][:, :, traps]
    response = np.linalg.solve(green_mm, green_mt)
    trap_coupling = ConductorCoupling(
        green[np.ix_(traps, traps)] - response.T @ green_mt,
        field[:, traps][:, :, traps] - np.einsum("mi,cmj->cij", response, field_mt),
        bar[traps] - response.T @ bar[members],
    )
    member = np.zeros(len(defects), dtype=bool)
    member[members] = True
    detached = np.where(member, defects.fixed_charge - defects.colocated_fixed_charge, 0.0)
    external = solver.solve(
        solver.deposit(defects.fixed_charge_positions, np.where(member, detached, defects.fixed_charge)), top
    )
    potential = share * top
    induced = np.linalg.solve(green_mm, potential - solver.interpolate(external, positions[members]))
    phi_at = induced @ green_mt
    field_at = np.einsum("m,cmt->ct", induced, field_mt)
    phi_bar = induced @ bar[members]
    if np.any(detached != 0.0):
        # 成员上离开成员位置的固定电荷：陷阱的 `KMCSimulation` 只放陷阱自己的固定电荷，这部分在背景里补上
        rest = solver.solve(solver.deposit(defects.fixed_charge_positions, detached), 0.0)
        phi_at = phi_at + solver.interpolate(rest, positions[traps])
        field_at = field_at + solver.field_vectors_at(rest, positions[traps], defects.field_side[traps])
        phi_bar = phi_bar + solver.lateral_average(rest)
    return ConductorLayout(
        labels, sides, traps, members, share, potential, share * positive_fermi_level(voltage), trap_coupling,
        (phi_at, field_at, phi_bar), induced, response, external,
    )


@dataclass(frozen=True)
class TransportState:
    """一个缺陷集合上的内层输运稳态（ADR-060，`solve_transport`）。

    `sim` 建在陷阱上；接电极导体的成员不在其中。全体缺陷的数组里，导体成员的"占据"取与导体电荷等价的值
    Z + q_fix,重合/e + 导体电子数（这样 q·(Z − 占据) + q_fix,重合 就是导体电荷），电势与能级取导体电势与费米能级。
    """

    sim: KMCSimulation  # 陷阱上的输运引擎
    result: MeanFieldResult  # 陷阱的平均场不动点
    heat: HeatRelease  # 式 T13 的放热（陷阱处、导体成员处与两侧电极）
    defects: DefectEnsemble  # 全体缺陷
    coupling: "IncrementalCoupling"  # 全体缺陷的静电耦合
    layout: ConductorLayout
    occupancy: np.ndarray  # (N,) 全体缺陷的占据
    potential: np.ndarray  # (N,) 引擎电势 [V]：陷阱取平均场电势，导体成员取导体电势
    energy: np.ndarray  # (N,) 引擎能级 [J]：陷阱取平均场能级，导体成员取所在处的费米能级
    # 泊松网格上的电势 [V]：不动点上的陷阱电荷、导体电荷、固定电荷与外加偏压（含导体尖端的场增强）；没有导体时为 None
    phi: Optional[np.ndarray]


def _take(defects: DefectEnsemble, index: np.ndarray) -> DefectEnsemble:
    """缺陷集合的子集，逐字段取 `index`。"""
    return DefectEnsemble(**{item.name: getattr(defects, item.name)[index] for item in fields(DefectEnsemble)})


class CarriedJacobian(NamedTuple):
    """上一次输运求解里平均场牛顿法最后用的 ∂G/∂x（ADR-066），交给下一次求解作初始 Jacobian。"""

    matrix: np.ndarray  # (F, F) 按那次的陷阱排列
    traps: np.ndarray  # (F,) 那次的陷阱在全体缺陷里的编号（升序；缺陷只在末尾追加，编号跨步不变）


def carried_jacobian(transport: "TransportState") -> Optional[CarriedJacobian]:
    """这次求解的 Jacobian；平均场只用了 Anderson 混合时为 None。"""
    matrix = transport.result.jacobian
    return None if matrix is None else CarriedJacobian(matrix, transport.layout.traps)


def _remap_jacobian(carried: CarriedJacobian, traps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """把沿用的 Jacobian 排到这次的陷阱 `traps` 上，返回 (矩阵, 要重求的列)：两次都是陷阱的缺陷之间照搬，新出现的陷阱
    （新生空位，或离开导体的缺陷）的行列取 −I（与别的缺陷无关时 G = ln p̂ − x 对自身的导数），交给牛顿法在起点处重求它们的列。"""
    kept = np.isin(traps, carried.traps)
    new = np.flatnonzero(kept)
    old = np.searchsorted(carried.traps, traps[kept])
    matrix = -np.eye(traps.size)
    matrix[np.ix_(new, new)] = carried.matrix[np.ix_(old, old)]
    return matrix, np.flatnonzero(~kept)


def solve_transport(
    config: Config,
    layout: ConductorLayout,
    defects: DefectEnsemble,
    coupling: "IncrementalCoupling",
    solver: PoissonSolver,
    temperature: np.ndarray,
    occupancy: Optional[np.ndarray] = None,
    jacobian: Optional[CarriedJacobian] = None,
) -> TransportState:
    """内层输运稳态（§6.2，ADR-060）：陷阱上建 `KMCSimulation`，求簇级平均场不动点与式 T13 的放热。

    `temperature` 是各缺陷处的温度 (N,)；`occupancy` 是全体缺陷的占据初值 (N,)，只用陷阱那部分热启动，缺省从全空起。
    `jacobian` 是上一次求解的 Jacobian（`carried_jacobian`），给了就排到这次的陷阱上，平均场直接走牛顿法（ADR-066）。
    有接电极的导体时，陷阱之间用消去成员后的等效耦合，导体的背景与交换通道由 `kmc.Conductors` 带进引擎；没有导体时
    就是在全体缺陷上建引擎，与漏电流侧逐位相同。
    """
    q = constants.elementary_charge
    traps, members = layout.traps, layout.members
    conductors = None
    if members.size:
        phi_at, field_at, phi_bar = layout.background
        conductors = Conductors(
            positions=defects.positions[members],
            fermi_level=layout.fermi_level,
            positive_share=layout.positive_share,
            r_D=defects.r_D[members],
            temperature=temperature[members],
            phi_at=phi_at,
            field_at=field_at,
            phi_bar=phi_bar,
        )
    sim = KMCSimulation(
        config,
        defects=_take(defects, traps) if members.size else defects,
        coupling=layout.coupling,
        solver=solver,
        temperature=temperature[traps],
        conductors=conductors,
    )
    matrix, refresh = (None, None) if jacobian is None else _remap_jacobian(jacobian, traps)
    result = solve_mean_field(sim, None if occupancy is None else occupancy[traps], matrix, refresh)
    p = result.occupancy
    full = np.empty(len(defects))
    potential = np.empty(len(defects))
    energy = np.empty(len(defects))
    full[traps] = p
    potential[traps] = sim._electrostatics(p)[0]
    energy[traps] = sim._trap_state(p)[0]
    potential[members] = layout.potential
    energy[members] = layout.fermi_level
    phi = None
    if members.size:
        weight = p - defects.charge_state[traps]
        electrons = layout.induced - layout.response @ weight
        full[members] = defects.charge_state[members] + defects.colocated_fixed_charge[members] / q + electrons
        phi = layout.external + solver.solve(
            solver.deposit(
                np.concatenate([defects.positions[traps], defects.positions[members]]),
                -q * np.concatenate([weight, electrons]),
            ),
            0.0,
        )
    return TransportState(
        sim, result, heat_release(sim, p), defects, coupling, layout, full, potential, energy, phi
    )


def gap_tunneling(transport: TransportState, temperature: np.ndarray, area: float) -> Tuple[float, np.ndarray]:
    """接电极导体与对面电极（或对面导体）之间的间隙隧穿（ADR-060）：返回 (电流 [A], 各导体成员处的放热 (M,) [W])。

    负极侧（负极平面与接负极的导体，费米能级 0）的电子隧穿到正极侧（正极平面与接正极的导体，费米能级 −eU）。每个导体成员
    一条路径，连到对面离它最近的点：接负极成员连到正上方的正极平面或最近的接正极成员，接正极成员连到正下方的负极平面或
    最近的接负极成员（xy 最小镜像）；两个成员互为最近时这条路径只算一次。平面与平面之间是 R1，在内层输运里。每条路径按
    R1 的能量积分（`r1_electrode_electrode`，直接/FN 隧穿与热发射统一处理）求速率：势垒取路径上的导带底 B − e·φ，φ 是
    输运不动点上含导体电荷的三维电势（`TransportState.phi`），导体尖端前方电势降得快、势垒薄，这一点由它给出；面积取每条
    路径一份 `area`；温度取成员处的温度（成员之间的路径取两端平均）；发射极电子质量取负极的，能量从电极导带底积起（导体
    是电极的延伸，ADR-061）。成员身后的成员，路径先穿过前面的成员（那里电势等于导体电势），透射自然小。每个电子放出 eU，
    全部放进导体：成员与平面之间的路径放在这个成员处，成员之间的路径两端各一半。贯穿通路（`sides` = 2）的成员不算，它的
    电流是细丝电流。`temperature` 是各缺陷处的温度 (N,)。
    """
    layout = transport.layout
    members = layout.members
    heat = np.zeros(members.size)
    if members.size == 0:
        return 0.0, heat
    sim = transport.sim
    device, solver, voltage = sim.device, sim.solver, sim.sim.voltage
    sides = layout.sides[members]
    points = transport.defects.positions[members]
    member_temperature = temperature[members]
    negative, positive = np.flatnonzero(sides == 0), np.flatnonzero(sides == 1)
    box = np.array([device.Lx, device.Ly, 4.0 * device.Lz])  # xy 周期；z 盒长取 4·L_z，不会回绕

    def nearest(sources: np.ndarray, targets: np.ndarray, plane: float) -> Tuple[np.ndarray, np.ndarray]:
        """各成员 `sources` 到对面最近的点：竖直方向的平面 z = `plane`，或 `targets` 里最近的成员。返回 (终点, 对面成员的
        编号，平面记 −1)。"""
        ends = points[sources].copy()
        ends[:, 2] = plane
        opposite = np.full(sources.size, -1)
        if targets.size and sources.size:
            distance, k = cKDTree(np.mod(points[targets], box), boxsize=box).query(np.mod(points[sources], box))
            closer = distance < np.abs(plane - points[sources, 2])
            opposite[closer] = targets[k[closer]]
            ends[closer] = points[sources[closer]] + minimum_image_delta(
                points[opposite[closer]] - points[sources[closer]], device.Lx, device.Ly
            )
        return ends, opposite

    paths = []  # (起点, 终点, 放热的成员, 各自的份额, 温度)
    collector_ends, collector = nearest(negative, positive, device.Lz)
    for a, end, b in zip(negative, collector_ends, collector):
        if b < 0:
            paths.append((points[a], end, [a], [1.0], member_temperature[a]))
        else:
            paths.append((points[a], end, [a, b], [0.5, 0.5], 0.5 * (member_temperature[a] + member_temperature[b])))
    emitter_ends, emitter = nearest(positive, negative, 0.0)
    for b, start, a in zip(positive, emitter_ends, emitter):
        if a < 0:
            paths.append((start, points[b], [b], [1.0], member_temperature[b]))
        elif collector[np.searchsorted(negative, a)] != b:
            paths.append((start, points[b], [a, b], [0.5, 0.5], 0.5 * (member_temperature[a] + member_temperature[b])))
    q = constants.elementary_charge
    s = np.linspace(0.0, 1.0, _GAP_PATH_POINTS)
    band_bottom = r1_band_bottom(device, voltage)
    rate_total = 0.0
    for start, end, ends, shares, path_temperature in paths:
        line = start + s[:, None] * (end - start)
        band = device.band_reference_profile(line[:, 2]) - q * solver.interpolate(transport.phi, line)
        rate = sum(r1_electrode_electrode(
            s * float(np.linalg.norm(end - start)), band, area, voltage, float(path_temperature),
            device.negative.effective_mass, device.stack.m_di_at(line[:, 2]), band_bottom,
        ))
        rate_total += rate
        heat[ends] += np.array(shares) * q * voltage * rate
    return q * rate_total, heat


def filament(layout: ConductorLayout, voltage: float) -> Tuple[float, np.ndarray]:
    """贯穿通路的细丝（ADR-054/060）：返回 (电流 G₀·V [A], 各导体成员处的放热 (M,) [W])。

    有贯穿通路时细丝按一个量子电导导通，功率 G₀·V² 平分到贯穿通路的空位上（隧穿到导体上的电子的能量放在导体里）；
    没有时电流与放热都为 0。
    """
    spanning = layout.sides[layout.members] == 2
    if not np.any(spanning):
        return 0.0, np.zeros(layout.members.size)
    current = _QUANTUM_CONDUCTANCE * voltage
    return current, np.where(spanning, current * voltage / np.count_nonzero(spanning), 0.0)


class _Step(NamedTuple):
    """一个缺陷集合上输运与热迭代自洽之后的稳态（`_steady_state`）。"""

    transport: TransportState
    heat: np.ndarray  # (N,) 各缺陷处的放热 [W]：陷阱取 T13 的放热，导体成员取交换、间隙隧穿与细丝的放热
    gap_current: float  # 间隙隧穿电流 [A]
    filament_current: float  # 细丝电流 [A]
    temperature: np.ndarray  # (N,) 这次输运用的缺陷处温度 [K]
    field: Optional[np.ndarray]  # 这次放热给出的温度场 [K]（热反馈时）
    iterations: int  # 输运与热的迭代次数
    converged: bool  # 缺陷处温度的变化是否 ≤ `_THERMAL_TOLERANCE`（等温时恒为真）


def _steady_state(
    config: Config,
    layout: ConductorLayout,
    defects: DefectEnsemble,
    coupling: "IncrementalCoupling",
    solver: PoissonSolver,
    occupancy: Optional[np.ndarray],
    temperature: np.ndarray,
    jacobian: Optional[CarriedJacobian] = None,
) -> _Step:
    """输运与热迭代自洽（§6.6、§8.4，ADR-060）。

    从缺陷处温度 `temperature` (N,) 与占据 `occupancy` 起，求内层输运、间隙隧穿与细丝的放热，解稳态热方程得到温度场，
    取它在缺陷处的值作为下一次输运的温度，直到变化 ≤ `_THERMAL_TOLERANCE`。迭代 `_THERMAL_MAX_ITERATIONS` 次仍不收敛
    （热失控）时返回最后一次，`converged` 记假。没有 k_TH 时温度固定，只求一次。`jacobian` 是上一次输运求解的 Jacobian
    （ADR-066），每次输运求解后换成这次的。
    """
    tddb, sim = config.tddb, config.simulation
    for iteration in range(1, _THERMAL_MAX_ITERATIONS + 1):
        transport = solve_transport(config, layout, defects, coupling, solver, temperature, occupancy, jacobian)
        jacobian = carried_jacobian(transport)
        gap_current, gap_heat = gap_tunneling(transport, temperature, tddb.gap_tunneling_area)
        filament_current, filament_heat = filament(layout, sim.voltage)
        heat = np.empty(len(defects))
        heat[layout.traps] = transport.heat.defect
        heat[layout.members] = transport.heat.conductor + gap_heat + filament_heat
        if tddb.thermal_conductivity is None:
            return _Step(transport, heat, gap_current, filament_current, temperature, None, iteration, True)
        field = solve_temperature(
            solver, power_map(solver, defects.positions, heat), tddb.thermal_conductivity, sim.temperature
        )
        updated = solver.interpolate(field, defects.positions)
        converged = float(np.max(np.abs(updated - temperature), initial=0.0)) <= _THERMAL_TOLERANCE
        if converged or iteration == _THERMAL_MAX_ITERATIONS:
            return _Step(transport, heat, gap_current, filament_current, temperature, field, iteration, converged)
        temperature, occupancy = updated, transport.occupancy


@dataclass
class TDDBResult:
    """一条 TDDB 瞬态轨迹（恒压应力）。

    第 k 条记录是第 k 次生成之后的输运稳态（k = 0 为 t = 0 的初始缺陷集合），电流在
    [time[k], time[k+1]) 上保持不变，最后一条保持到 `end_time`。电流均为净值，正号 = 电子由
    负极流向正极。第一条出现贯穿通路的记录给出 t_BD（`breakdown_time`），之后再跑 `post_breakdown_steps` 条记录，
    这些记录的电流含细丝电流（ADR-054/060）。
    """

    voltage: float  # [V]
    temperature: float  # 环境温度（电极恒温）[K]
    area: float  # 电极面积 [m^2]
    seed: int
    time: np.ndarray  # (K,) 各记录的时刻 [s]
    current: np.ndarray  # (K,) 总电流 = 内层输运电流 + 间隙隧穿电流 + 细丝电流 [A]
    transport_current: np.ndarray  # (K,) 内层输运（平均场稳态）的电流 [A]
    by_mechanism: Dict[str, np.ndarray]  # 内层输运各机制 (K,) [A]
    gap_current: np.ndarray  # (K,) 接电极导体与对面电极（或对面导体）之间的间隙隧穿电流 [A]（ADR-060）
    filament_current: np.ndarray  # (K,) 贯穿通路的细丝电流 G₀·V [A]，击穿记录之前为 0（ADR-054）
    conductor_gap: np.ndarray  # (K,) 负极侧与正极侧导体的最短距离 [m]，有贯穿通路时为 0（`conductor_gap`）
    # (K,) 这一状态下全部前驱位点的总生成速率 ΣG [1/s]；轨迹的最后一条击穿记录不求生成率，记 NaN
    generation_rate: np.ndarray
    n_precursors: np.ndarray  # (K,) 这一状态下还没断键的前驱位点数
    n_defects: np.ndarray  # (K,) 缺陷总数（初始 + 生成）
    # (K,) 全部缺陷的净电荷 Σ[q·(Z − p) + q_fix] [C]（§8.6）；接电极导体的成员取导体电荷
    space_charge: np.ndarray
    empty_generated: np.ndarray  # (K,) 生成的空位里还是陷阱、处于空态的期望个数 Σ(1 − p)（§8.6）
    iterations: np.ndarray  # (K,) 平均场求 F 的次数
    newton_steps: np.ndarray  # (K,) 平均场牛顿法的步数（0 = Anderson 混合已收敛，ADR-048）
    residual: np.ndarray  # (K,) 平均场的相对残差
    converged: np.ndarray  # (K,) 平均场是否收敛（未收敛的记录不能当稳态电流交付）
    n_clusters: np.ndarray  # (K,) 强耦合簇个数
    max_cluster_size: np.ndarray  # (K,) 最大簇大小（≤ 簇大小上限）
    largest_component: np.ndarray  # (K,) 切分之前的最大强耦合分量
    n_split: np.ndarray  # (K,) 超过簇大小上限、被切开的分量数
    # --- 热（式 T13/T14，§6.6，ADR-060）---
    power: np.ndarray  # (K,) 介质内放出的总功率 [W]（陷阱、导体交换、间隙隧穿与细丝；负值 = 净吸热）
    defect_temperature: np.ndarray  # (K,) 本条记录内层输运用的缺陷处温度（含导体成员）的最高值 [K]
    peak_temperature: np.ndarray  # (K,) 本条记录温度场的最高值 [K]
    thermal_iterations: np.ndarray  # (K,) 输运与热的迭代次数
    thermal_converged: np.ndarray  # (K,) 输运与热的迭代是否收敛（不收敛 = 热失控）
    # --- 击穿（§6.3）---
    dense_cluster_size: np.ndarray  # (K,) 最大致密团簇（间距 < 6 Å，论文的标志，只记录）的缺陷数
    breakdown_time: Optional[float]  # 贯穿通路首次形成的时刻 t_BD [s]；没有击穿时为 None
    end_time: float  # 轨迹终点 [s]；击穿后再跑的步数用完时就是最后一条记录的时刻
    # "breakdown"（击穿后再跑的步数用完）| "max_generated" | "stop_time" | "no_generation"
    stop_reason: str
    n_initial: int  # 初始缺陷个数：编号 < n_initial 的是初始缺陷，其余是应力生成的空位
    birth_time: np.ndarray  # (N,) 各缺陷的出生时刻 [s]，初始缺陷记 0
    defects: DefectEnsemble  # 终态缺陷集合
    occupancy: np.ndarray  # 终态的占据（陷阱取平均场占据，接电极导体的成员取与导体电荷等价的值）
    temperature_at_defects: np.ndarray  # (N,) 终态内层输运用的缺陷处温度 [K]
    thermal: bool  # 是否开了热反馈（配置了 k_TH）

    @property
    def current_density(self) -> np.ndarray:
        """各记录的电流密度 [A/m^2]。"""
        return self.current / self.area

    @property
    def dense_cluster_time(self) -> Optional[float]:
        """最大致密团簇首次达到 25 个空位的时刻 [s]（论文的标志）；没有达到时为 None。"""
        reached = np.flatnonzero(self.dense_cluster_size >= _DENSE_CLUSTER_SIZE)
        return float(self.time[reached[0]]) if reached.size else None

    @property
    def n_generated(self) -> np.ndarray:
        """各记录时已生成的空位数。"""
        return self.n_defects - self.n_initial

    @property
    def segment_end(self) -> np.ndarray:
        """各记录保持到的时刻 [s]：下一条记录的时刻，最后一条为 `end_time`。"""
        return np.append(self.time[1:], self.end_time)

    def current_at(self, times: np.ndarray) -> np.ndarray:
        """任意时刻 t ∈ [0, end_time] 的总电流 [A]（分段常数）。"""
        return self.current[self._record_at(times)]

    def _record_at(self, times: np.ndarray) -> np.ndarray:
        """各时刻所在的记录编号。"""
        index = np.searchsorted(self.time, np.asarray(times, dtype=float), side="right") - 1
        return np.clip(index, 0, None)


@dataclass(frozen=True)
class TDDBState:
    """外层循环一步开始时（求内层稳态之前）的全部状态，续跑用（ADR-057）。

    从这个状态起跑（`run_tddb` 的 `start`）与不停跑的同一条轨迹只差舍入误差：静电耦合按当前缺陷集合从头算
    （`IncrementalCoupling.build`），与逐个追加的结果相对差约 10⁻¹⁰；其余量原样接上。
    """

    defects: DefectEnsemble  # 当前缺陷集合（初始缺陷在前，生成空位按出生顺序接在后面）
    n_initial: int  # 初始缺陷个数
    birth_time: np.ndarray  # (N,) 各缺陷的出生时刻 [s]，初始缺陷记 0
    time: float  # 当前时刻 [s]
    occupancy: Optional[np.ndarray]  # (N,) 平均场热启动的占据；t = 0 为 None（从全空起）
    precursors: Precursors  # 还没断键的前驱位点
    # (N,) 这一步输运与热迭代起点的缺陷处温度 [K]：上一步温度场在各缺陷处的值（新空位取出生位置的值），第一步为环境温度
    temperature: np.ndarray
    breakdown_time: Optional[float]  # 贯穿通路首次形成的时刻 [s]；还没击穿为 None
    steps_after_breakdown: int  # 击穿记录之后已经跑的生成步数
    rng: np.random.Generator  # 下一次生成事件接着用的随机数流


def initial_state(config: Config) -> TDDBState:
    """t = 0 的状态：同一个种子的随机数流先生成初始缺陷（与 `KMCSimulation(config)` 是同一个实现），再铺前驱位点
    （`place_precursors`），之后的生成事件接着用这个流。"""
    rng = np.random.default_rng(config.simulation.seed)
    defects = generate_defects(config.device, rng)
    tddb = config.tddb
    precursors = place_precursors(
        config.device, defects.positions, tddb.activation_energy, tddb.activation_energy_tail, rng
    )
    return TDDBState(
        defects=defects,
        n_initial=len(defects),
        birth_time=np.zeros(len(defects)),
        time=0.0,
        occupancy=None,
        precursors=precursors,
        temperature=np.full(len(defects), config.simulation.temperature),
        breakdown_time=None,
        steps_after_breakdown=0,
        rng=rng,
    )


def save_state(state: TDDBState, path: str) -> None:
    """把续跑状态写成一个 npz：缺陷集合逐字段、前驱位点坐标、随机数流的内部状态（JSON）与其余量。还没击穿时
    击穿时刻记 NaN。"""
    arrays = {"defect_" + item.name: getattr(state.defects, item.name) for item in fields(DefectEnsemble)}
    np.savez(
        path,
        n_initial=state.n_initial,
        birth_time=state.birth_time,
        time=state.time,
        occupancy=np.full(0, np.nan) if state.occupancy is None else state.occupancy,
        has_occupancy=state.occupancy is not None,
        precursor_positions=state.precursors.positions,
        precursor_activation_energy=state.precursors.activation_energy,
        temperature=state.temperature,
        breakdown_time=np.nan if state.breakdown_time is None else state.breakdown_time,
        steps_after_breakdown=state.steps_after_breakdown,
        rng_state=json.dumps(state.rng.bit_generator.state),
        **arrays,
    )


def load_state(path: str) -> TDDBState:
    """读回 `save_state` 写的续跑状态，逐位相同。"""
    with np.load(path) as data:
        rng_state = json.loads(str(data["rng_state"]))
        rng = np.random.Generator(getattr(np.random, rng_state["bit_generator"])())
        rng.bit_generator.state = rng_state
        breakdown_time = float(data["breakdown_time"])
        return TDDBState(
            defects=DefectEnsemble(**{item.name: data["defect_" + item.name] for item in fields(DefectEnsemble)}),
            n_initial=int(data["n_initial"]),
            birth_time=data["birth_time"],
            time=float(data["time"]),
            occupancy=data["occupancy"] if bool(data["has_occupancy"]) else None,
            precursors=Precursors(data["precursor_positions"], data["precursor_activation_energy"]),
            temperature=data["temperature"],
            breakdown_time=None if np.isnan(breakdown_time) else breakdown_time,
            steps_after_breakdown=int(data["steps_after_breakdown"]),
            rng=rng,
        )


def run_tddb(
    config: Config,
    progress: Optional[Callable[[int, float, float], None]] = None,
    start: Optional[TDDBState] = None,
    checkpoint: Optional[Callable[[TDDBState], None]] = None,
) -> TDDBResult:
    """按 `config` 跑一条 TDDB 瞬态轨迹（模块文档的双层循环）。

    缺省从 `initial_state(config)` 起跑；给了 `start` 就从这个状态接着跑（续跑，ADR-057），`start` 本身不被改动。
    应力工况取 `simulation.voltage`，环境温度取 `simulation.temperature`，生成、热与击穿参数取 `config.tddb`。本阶段只做
    单层（PROJECT_CONTEXT_TDDB.md §1）。续跑的结果只含起跑之后的记录，`n_initial` 与 `birth_time` 接着 `start`。

    `progress(已生成空位数, 时刻 [s], 电流密度 [A/m^2])` 每求完一个状态调用一次；`checkpoint(状态)` 每步开始时调用一次，
    拿到的状态交给 `start` 即从这一步接着跑。
    """
    tddb = config.tddb
    if tddb is None:
        raise ValueError("TDDB 瞬态运行需要配置里的 tddb 段")
    device, sim = config.device, config.simulation
    if len(device.stack.layers) != 1:
        raise NotImplementedError("TDDB 本阶段只做单层介质（PROJECT_CONTEXT_TDDB.md §1），叠层 TDDB 后续再扩")

    state = initial_state(config) if start is None else start
    rng = copy.deepcopy(state.rng)
    defects, n_initial, birth_time = state.defects, state.n_initial, state.birth_time
    precursors, occupancy, time = state.precursors, state.occupancy, state.time
    temperature = state.temperature  # 缺陷处温度：输运与热迭代的起点
    breakdown_time, steps_after_breakdown = state.breakdown_time, state.steps_after_breakdown
    solver = PoissonSolver(device, sim.grid)
    coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
    thermal = tddb.thermal_conductivity is not None
    records: List[dict] = []
    jacobian: Optional[CarriedJacobian] = None  # 上一步平均场牛顿法的 Jacobian（ADR-066），续跑时从头求

    while True:
        if checkpoint is not None:
            checkpoint(TDDBState(
                defects=defects, n_initial=n_initial, birth_time=birth_time, time=time, occupancy=occupancy,
                precursors=precursors, temperature=temperature, breakdown_time=breakdown_time,
                steps_after_breakdown=steps_after_breakdown, rng=copy.deepcopy(rng),
            ))
        layout = conductor_layout(defects, coupling, solver, sim.voltage, tddb.transfer_integral)
        step = _steady_state(config, layout, defects, coupling, solver, occupancy, temperature, jacobian)
        transport, mf = step.transport, step.transport.result
        jacobian = carried_jacobian(transport)
        occupancy = transport.occupancy
        # §6.3（ADR-054/060）：负极侧与正极侧导电簇连通即击穿；之后再跑 post_breakdown_steps 步，最后一条不再求生成率
        if breakdown_time is not None:
            steps_after_breakdown += 1
        elif np.any(layout.sides == 2):
            breakdown_time = time
        finished = breakdown_time is not None and steps_after_breakdown >= tddb.post_breakdown_steps
        rates = (
            np.full(1, np.nan) if finished
            else generation_rates(transport, tddb, precursors, step.field).rates
        )
        n_clusters, max_cluster, largest, n_split = _cluster_diagnostics(transport.sim)
        dense_labels = distance_clusters(defects.positions, device, _DENSE_CLUSTER_DISTANCE)
        current = mf.current + step.gap_current + step.filament_current
        generated_traps = layout.sides[n_initial:] < 0
        records.append(
            dict(
                time=time,
                current=current,
                transport_current=mf.current,
                by_mechanism=mf.by_mechanism,
                gap_current=step.gap_current,
                filament_current=step.filament_current,
                conductor_gap=conductor_gap(defects.positions, layout.sides, device),
                generation_rate=float(rates.sum()),
                n_precursors=len(precursors),
                n_defects=len(defects),
                space_charge=constants.elementary_charge * float(np.sum(defects.charge_state - occupancy))
                + float(np.sum(defects.fixed_charge)),
                empty_generated=float(np.sum(1.0 - occupancy[n_initial:][generated_traps])),
                iterations=mf.iterations,
                newton_steps=mf.newton_steps,
                residual=mf.residual,
                converged=mf.converged,
                n_clusters=n_clusters,
                max_cluster_size=max_cluster,
                largest_component=largest,
                n_split=n_split,
                power=float(step.heat.sum()),
                defect_temperature=float(np.max(step.temperature, initial=sim.temperature)),
                peak_temperature=sim.temperature if step.field is None else float(step.field.max()),
                thermal_iterations=step.iterations,
                thermal_converged=step.converged,
                dense_cluster_size=int(np.bincount(dense_labels).max(initial=0)),
            )
        )
        if progress is not None:
            progress(len(defects) - n_initial, time, current / device.area)

        if finished:
            end_time, stop_reason = time, "breakdown"
            break
        if len(defects) - n_initial >= tddb.max_generated_vacancies:
            end_time, stop_reason = time, "max_generated"
            break
        event = _draw_generation(rates, rng)
        if event is None:
            end_time, stop_reason = tddb.stop_time, "no_generation"
            break
        dt, index = event
        if time + dt > tddb.stop_time:
            end_time, stop_reason = tddb.stop_time, "stop_time"
            break
        time += dt
        position = precursors.positions[index]
        precursors = precursors.converted(index)
        vacancy = layer_bulk_defects(device, 0, position, rng)
        defects = append_defects(defects, replace(vacancy, fixed_charge=np.full(1, _INTERSTITIAL_CHARGE)))
        coupling = coupling.extended(position)
        occupancy = np.append(occupancy, 1.0)  # §6.7：出生即被一个电子占据；连同 O_i⁻ 带 −e
        birth_time = np.append(birth_time, time)
        # 下一步迭代的起点：这一步温度场在各缺陷处的值，新空位取出生位置的值
        temperature = (
            np.full(len(defects), sim.temperature) if step.field is None
            else solver.interpolate(step.field, defects.positions)
        )

    def column(key: str, dtype=float) -> np.ndarray:
        return np.array([record[key] for record in records], dtype=dtype)

    return TDDBResult(
        voltage=sim.voltage,
        temperature=sim.temperature,
        area=device.area,
        seed=sim.seed,
        time=column("time"),
        current=column("current"),
        transport_current=column("transport_current"),
        by_mechanism={
            name: np.array([record["by_mechanism"][name] for record in records]) for name in MECHANISMS
        },
        gap_current=column("gap_current"),
        filament_current=column("filament_current"),
        conductor_gap=column("conductor_gap"),
        generation_rate=column("generation_rate"),
        n_precursors=column("n_precursors", int),
        n_defects=column("n_defects", int),
        space_charge=column("space_charge"),
        empty_generated=column("empty_generated"),
        iterations=column("iterations", int),
        newton_steps=column("newton_steps", int),
        residual=column("residual"),
        converged=column("converged", bool),
        n_clusters=column("n_clusters", int),
        max_cluster_size=column("max_cluster_size", int),
        largest_component=column("largest_component", int),
        n_split=column("n_split", int),
        power=column("power"),
        defect_temperature=column("defect_temperature"),
        peak_temperature=column("peak_temperature"),
        thermal_iterations=column("thermal_iterations", int),
        thermal_converged=column("thermal_converged", bool),
        dense_cluster_size=column("dense_cluster_size", int),
        breakdown_time=breakdown_time,
        end_time=end_time,
        stop_reason=stop_reason,
        n_initial=n_initial,
        birth_time=birth_time,
        defects=defects,
        occupancy=occupancy,
        temperature_at_defects=step.temperature,
        thermal=thermal,
    )


@dataclass(frozen=True)
class PowerLawFit:
    """双对数坐标下的直线拟合 log₁₀ I = log₁₀ A + n·log₁₀ t。"""

    exponent: float  # 斜率 n
    prefactor: float  # A：t = 1 s 处的电流，单位同输入电流
    r_squared: float  # 双对数线性回归的决定系数

    def __call__(self, times: np.ndarray) -> np.ndarray:
        """拟合直线在 `times` 处的值。"""
        return self.prefactor * np.asarray(times, dtype=float) ** self.exponent


def fit_power_law(times: np.ndarray, current: np.ndarray) -> PowerLawFit:
    """对 (t, I) 样本在双对数坐标下做最小二乘直线拟合，给出幂律斜率 n 与拟合优度 R²。

    样本的 t、I 都必须为正。所有点等权，时间窗口与取样方式由调用方定（`fit_transient` 按
    log t 等间距取样）。样本的 log I 完全相同时 R² 记 1（水平线精确拟合）。
    """
    t = np.asarray(times, dtype=float)
    i = np.asarray(current, dtype=float)
    if t.size < 2 or np.any(t <= 0.0) or np.any(i <= 0.0):
        raise ValueError("幂律拟合需要至少两个 t > 0、I > 0 的样本")
    x, y = np.log10(t), np.log10(i)
    exponent, intercept = np.polyfit(x, y, 1)
    ss_res = float(np.sum((y - (intercept + exponent * x)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return PowerLawFit(exponent=float(exponent), prefactor=float(10.0 ** intercept), r_squared=r_squared)


def fit_transient(
    result: TDDBResult,
    t_start: Optional[float] = None,
    t_stop: Optional[float] = None,
    points_per_decade: int = 20,
) -> Tuple[PowerLawFit, np.ndarray]:
    """在 [t_start, t_stop] 上按 log t 等间距取样电流密度 J(t)，做双对数幂律拟合。

    缺省窗口是第一次生成到击穿（没有击穿时到轨迹终点）。按 log t 等间距取样让每个时间十倍程的权重相同——
    生成事件在线性时间上大致均匀，直接拿事件点拟合会让最后一个十倍程占绝大多数权重。
    拟合的是击穿前的 I-t，取内层输运电流，间隙隧穿与细丝电流不计入（ADR-054/060）。
    返回 (拟合结果, 取样时刻)；拟合的电流单位是 A/m²。
    """
    if t_start is None:
        if result.time.size < 2:
            raise ValueError("轨迹里没有生成事件，无法做 I-t 拟合")
        t_start = float(result.time[1])
    if t_stop is None:
        t_stop = result.end_time if result.breakdown_time is None else result.breakdown_time
    decades = np.log10(t_stop / t_start)
    times = np.logspace(np.log10(t_start), np.log10(t_stop), max(2, int(np.ceil(decades * points_per_decade)) + 1))
    return fit_power_law(times, result.transport_current[result._record_at(times)] / result.area), times


def write_tddb_csv(result: TDDBResult, path: str) -> None:
    """把一条 TDDB 轨迹写成 CSV：每行一段 [t_start, t_end) 上的稳态，电流密度换算成 A/cm²。"""
    per_cm2 = constants.centi ** 2  # A/m^2 -> A/cm^2
    density = result.current_density * per_cm2
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["t_start_s", "t_end_s", "n_generated", "n_defects", "J_total_A_cm2"]
            + ["J_{}_A_cm2".format(name) for name in MECHANISMS]
            + [
                "J_gap_A_cm2", "J_filament_A_cm2", "conductor_gap_nm",
                "generation_rate_s", "n_precursors", "space_charge_C", "empty_generated",
                "mf_iterations", "mf_newton_steps", "mf_residual", "mf_converged",
                "n_clusters", "max_cluster_size", "largest_component", "n_split",
                "power_W", "T_defect_max_K", "T_peak_K", "thermal_iterations", "thermal_converged",
                "dense_cluster_size",
            ]
        )
        for k in range(result.time.size):
            writer.writerow(
                ["{:.6e}".format(result.time[k]), "{:.6e}".format(result.segment_end[k])]
                + [str(int(result.n_generated[k])), str(int(result.n_defects[k])), "{:.6e}".format(density[k])]
                + ["{:.6e}".format(result.by_mechanism[name][k] / result.area * per_cm2) for name in MECHANISMS]
                + [
                    "{:.6e}".format(result.gap_current[k] / result.area * per_cm2),
                    "{:.6e}".format(result.filament_current[k] / result.area * per_cm2),
                    "{:.6f}".format(result.conductor_gap[k] / constants.nano),
                    "{:.6e}".format(result.generation_rate[k]),
                    str(int(result.n_precursors[k])),
                    "{:.6e}".format(result.space_charge[k]),
                    "{:.6g}".format(result.empty_generated[k]),
                    str(int(result.iterations[k])),
                    str(int(result.newton_steps[k])),
                    "{:.3e}".format(result.residual[k]),
                    str(int(result.converged[k])),
                    str(int(result.n_clusters[k])),
                    str(int(result.max_cluster_size[k])),
                    str(int(result.largest_component[k])),
                    str(int(result.n_split[k])),
                    "{:.6e}".format(result.power[k]),
                    "{:.6f}".format(result.defect_temperature[k]),
                    "{:.6f}".format(result.peak_temperature[k]),
                    str(int(result.thermal_iterations[k])),
                    str(int(result.thermal_converged[k])),
                    str(int(result.dense_cluster_size[k])),
                ]
            )
