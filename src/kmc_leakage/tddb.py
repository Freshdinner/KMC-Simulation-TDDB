"""TDDB 瞬态驱动：准静态双层循环，含热反馈与击穿判据。

规格见 PROJECT_CONTEXT_TDDB.md §5、§6，决策记录见 ADR-032–055。公式编号 T1–T14 是 TDDB 的
编号（PROJECT_CONTEXT_TDDB.md 第 3 节），与 docs/rate_equations.md 的 R1–R8 不对应。

电子输运事件的时间尺度（~10⁻¹³…10⁻⁶ s）比缺陷生成（秒以上）短十几个数量级，所以输运相对
生成瞬间达到稳态（§5）。t = 0 在初始缺陷之外铺离散前驱位点（§6.1，ADR-052/055）：可采样窗口里按最小间隔
min_separation 做最大随机堆积，直到再也放不下一个位点，浓度由间隔决定；全部位点的无场断键激活能取同一个
E_A（ADR-053）。外层是缺陷生成的 Gillespie 循环，每步：

1. **内层稳态**：按当前缺陷集合与输运温度重建 `KMCSimulation`（静电耦合用
   `IncrementalCoupling` 逐个追加的那份），从上一步的占据热启动，求簇级平均场不动点
   （`mean_field.solve_mean_field`），得到电流、分解与各缺陷占据 p。近邻表、强耦合簇与截断
   半径都随缺陷集合从头重算，与对同一缺陷集合新建的 `KMCSimulation` 逐项相同（ADR-033）。
2. **功率与温度**（式 T13/T14，`thermal.py`）：在内层稳态上逐事件累加放热 ΔE·R，介质内的
   功率解稳态热方程得到温度场（两电极恒温 T_amb）。配置了 k_TH 才开热反馈：温度场在前驱位点上
   的值进生成率，缺陷处按放热加权的平均温度作为下一步内层输运的全局标量温度 T(t)（§6.6）。
   输运与热不迭代自洽，每步只走一遍"输运 → 功率 → 温度"（§8.4），所以第 k 条记录的电流是在
   第 k−1 条记录的功率给出的温度下求的；温度随生成一步步爬升，这一步的滞后可以忽略。没有 k_TH
   时温度固定为环境值（Phase 3 的等温口径）。
3. **导电簇与击穿判据**（§6.3，ADR-054）：O 空位间距 < r_c 连成导电簇，离电极平面 < r_c 的空位接到该
   电极（`conducting_clusters`）。负极侧与正极侧导电簇连通、形成贯穿通路即击穿，这一刻就是 t_BD；击穿
   记录的电流加上细丝按一个量子电导 G₀ = 2e²/h 导通的 G₀·V。论文的"25 个空位、间距 ≤ 6 Å"团簇只作标志
   记录，不作判据。
4. **生成率**（§6.1，`generation_rates`）：各缺陷净电荷 q·(Z − p) + q_fix 与外加偏压一起解泊松方程，接电极的
   导电簇与该电极等电位（ADR-054），悬空导电簇保持总电荷、自身等电位（ADR-056）；在每个前驱位点上求式 T1 的
   G_i = R_1e,i·P_bb,i：
   - R_1e,i = 两侧电极 R2 + R4（电极项，式 T4）+ Σ_j p_j·R7(j → i)（陷阱项：近旁已占据陷阱经陷阱间通道
     送来的电子，ADR-039；导电簇的空位按所接电极费米能级处占满的电子源计入）。前驱能级、R4 的局域场与
     温度取位点处含空间电荷的局域值，取法同引擎处理真实陷阱；电极项在 (z, 前驱能级, 温度) 的插值表上
     求值（ADR-055）；
   - P_bb,i 的场取泊松解在 Gaussian 窗口上的平均（宏观场，ADR-043）。导电簇尖端前方的场被抬高，生成
     向尖端集中，导电簇越长尖端场越强，这是贯穿通路最后一段迅速形成的机制。
5. **抽事件**：Δt = −ln(r)/ΣG，按 G_i 选一个前驱位点断键，新空位就落在这个位点上。这个位点用掉，
   其余位点离它本来就 ≥ min_separation，原样保留。
6. **出生**：新空位与初始缺陷同一物种（能级按缺陷预设的分布抽样，S、m_di 取预设），出生占据 1，
   施主约定 `charge_state` = +1（§6.7、ADR-035）。生成反应"前驱 + e⁻ → V_O + O_i⁻"按电荷守恒处理
   （ADR-048）：前驱俘获的那个电子的电荷留在 O 间隙离子上，O_i⁻ 作为新空位的固定电荷
   q_fix = −e，与空位重合放置（`DefectEnsemble.fixed_charge_offset` = 0）。新空位与 O_i⁻ 合起来出生时
   带 −e，就是被俘获的那个电子；空位放出电子后合起来为中性。静电耦合增量扩一行一列（§6.4），回到 1。

未击穿时按生成空位数上限或应力时间上限停止，前驱用完时按"总生成率为 0"停止。

**出生时的电子**：新空位出生时带的那个电子来自前驱俘获，不计入输运电流（§6.7）；它的电荷随新空位与
O_i⁻ 一起留在介质里。平均场没有时间推进，每条记录的电流就是该缺陷集合下的稳态电流。

**导电簇只进生成率**（ADR-054）：内层输运仍把导电簇的空位当作 TAT 陷阱，击穿前的电流是 TAT 电流。
"""

import csv
import itertools
from dataclasses import dataclass, replace
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
    neighbor_pairs,
)
from .electrostatics import PoissonSolver
from .kmc import (
    _CLUSTER_COUPLING,
    _CLUSTER_MAX_SIZE,
    _TRAP_PATH_POINTS,
    MECHANISMS,
    ElectrostaticCoupling,
    KMCSimulation,
    _band_on_path_steps,
    _interface_step_nodes,
)
from .mean_field import solve_mean_field
from .rates import (
    CaptureElectrode,
    bond_break_probability,
    single_electron_capture_parts,
    trap_precursor_capture_rate,
)
from .thermal import heat_release, power_map, solve_temperature, transport_temperature

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
# R_1e 电极项的插值表（`_electrode_capture`，ADR-055）：z、前驱能级、温度三个方向的节点间距。3.6 V 默认工况下与逐点
# 求值相比，按生成率加权的相对误差 2.5×10⁻⁴，生成率不可忽略的位点上最大 1.1×10⁻³
_TABLE_Z_STEP = 0.05 * constants.nano
_TABLE_ENERGY_STEP = 0.005 * constants.electron_volt
_TABLE_TEMPERATURE_STEP = 0.5  # [K]
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
    位点是最小间隔 min_separation 的最大随机堆积（`place_precursors`），位置连续（§6.8）；无场断键激活能
    全部取 `tddb.activation_energy`（ADR-053）。
    """

    positions: np.ndarray  # (M, 3) 位点坐标 [m]

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def converted(self, index: int) -> "Precursors":
        """第 `index` 个位点断键、原地变成新空位之后剩下的位点：只去掉它自己。其余位点离它本来就
        ≥ min_separation，都还能放下新空位。"""
        return Precursors(np.delete(self.positions, index, axis=0))


def place_precursors(device: DeviceConfig, defects: np.ndarray, rng: np.random.Generator) -> Precursors:
    """t = 0 的前驱位点（§6.1，ADR-052/055）。

    前驱位点不设浓度，由最小间隔 d = min_separation 决定（取 ZrO₂ 的 O–O 间距）：在可采样窗口
    [0,Lx)×[0,Ly)×[d, Lz−d]（与 `generate_defects` 的电极避让相同）里做最大随机堆积（`_maximal_packing`），
    位点两两之间、位点与初始缺陷 `defects`（坐标 (N, 3)）之间都 ≥ d，且窗口里再也放不下一个位点。
    d = 0.26 nm 时密度约 4.3×10²² cm⁻³。
    """
    standoff = _global_standoff(device)
    if not standoff > 0.0:
        raise ValueError("前驱位点按 min_separation 堆积，min_separation_nm 必须为正")
    if device.Lz - 2.0 * standoff <= 0.0:
        raise ValueError("min_separation 约束下 z 方向没有可放置新空位的区间")
    return Precursors(_maximal_packing(device, standoff, defects, rng))


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
    inner: KMCSimulation,
    occupancy: np.ndarray,
    tddb: TDDBConfig,
    precursors: Precursors,
    temperature_field: Optional[np.ndarray] = None,
) -> GenerationRates:
    """各前驱位点的 O 空位生成速率 G_i = R_1e,i·P_bb,i（式 T1/T2/T4，§6.1，ADR-052）。

    `inner` 提供当前缺陷集合、工作点与泊松求解器，`occupancy` 是各缺陷的平均场占据 p。
    电势由 `_conductor_potential` 给出：每个缺陷随占据变的电荷 q·(Z − p)（放在缺陷处）与固定电荷 q_fix
    （放在固定电荷处）和外加偏压一起解泊松方程；导电簇（`distance_clusters` / `conducting_clusters`，r_c 取
    `tddb.percolation_distance`）是等电位的导体：接电极的与该电极等电位（ADR-054），悬空的保持总电荷、电势待定
    （ADR-056）。各位点在这份电势上求值，取法同引擎处理真实陷阱：前驱能级 E = B − e·φ(x) − E_t1，R4 的 c_MP 用局域场。

    - **电极项**（`_electrode_capture`）：两侧电极 R2 + R4；
    - **陷阱项**（`_trap_capture`）：近旁已占据陷阱经陷阱间通道 p·R7 送给前驱的电子（式 T5 的陷阱来源，
      ADR-039）。接电极的导电簇的空位是所接电极的延伸，按电极费米能级处占满（p = 1）的电子源计入；其余陷阱
      （含悬空导电簇的成员）按平均场占据计入，能级取同一份电势（引擎的平均场能级加上导电簇带来的电势改变）；
    - **P_bb**：E_A 全部位点相同（ADR-053）；场是泊松解的场在标准差 `tddb.field_averaging_width` 的 Gaussian
      窗口上的平均（宏观场，ADR-043）。Lorentz 关系 (2+k)/3 针对在 Lorentz 球上近似均匀的宏观场，带电缺陷亚纳米
      近旁的点电荷场不满足这个前提。

    `temperature_field` 是泊松网格上的温度场（热反馈时取 T14 的解，§6.6）：R_1e 的声子线型、电极费米分布、
    陷阱项的 MA 因子与 P_bb 都取位点处的温度。缺省用内层的工作点温度。
    """
    device, solver, defects = inner.device, inner.solver, inner.defects
    material = device.stack.single_material
    n = len(precursors)
    if n == 0:
        return GenerationRates(np.zeros(0), np.zeros(0), np.zeros(0))
    labels = distance_clusters(defects.positions, device, tddb.percolation_distance)
    sides = conducting_clusters(defects.positions, device, tddb.percolation_distance)
    phi, shift = _conductor_potential(inner, occupancy, sides, labels)
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
    member = sides >= 0
    fermi = np.array([0.0, positive_fermi_level(inner.sim.voltage)])  # 负极、正极费米能级
    source_energy = np.where(
        member, fermi[np.clip(sides, 0, 1)], inner._trap_state(occupancy)[0] - constants.elementary_charge * shift
    )
    trap = _trap_capture(
        inner, tddb, points, conduction - tddb.precursor_ionization_energy, source_energy,
        np.where(member, 1.0, occupancy), temperature,
    )
    p_bb = bond_break_probability(
        tddb.activation_energy, tddb.dipole_moment, material.kappa, np.linalg.norm(local[:, 4:7], axis=1),
        temperature,
    )
    return GenerationRates(electrode, trap, p_bb)


def _conductor_potential(
    inner: KMCSimulation, occupancy: np.ndarray, sides: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """生成率用的电势 φ（泊松网格节点，[V]）与导电簇带来的各缺陷处电势改变 (N,) [V]（ADR-054/056）。

    `labels` 是 r_c 下的导电簇编号（`distance_clusters`），`sides` 是各缺陷所在导电簇接的电极（`conducting_clusters`）。
    不在导电簇里的缺陷带 q·(Z − p) + q_fix，固定电荷放在各自的位置。两个及以上空位连成的导电簇是导体，成员的电荷
    不再由占据决定，取让全部成员处电势相同的那一组电荷：
    - 接电极的导电簇（`sides` 取 0 或 1）：成员处的电势等于所接电极的边界值（负极 0、正极 `top_boundary_potential`）；
    - 悬空导电簇（`sides` = −1）：电势是待定的常数，簇的总电荷保持引擎给的 Σ[q·(Z − p) + q_fix,重合]。
    成员处的电势对这组电荷是线性的，系数就是耦合矩阵 `phi_matrix`（同一套沉积、求解与插值）；悬空导电簇每个多一个电势
    未知数和一条总电荷方程，在全部成员上解一次线性方程即得。离开成员位置的固定电荷照常放置。只有一个空位的悬空
    "导电簇"按这个规则电荷不变，与不在导电簇里相同。

    第二项是电势改变：引擎的平均场能级减去 e·(这一项) 就是同一份电势下的陷阱能级。其余缺陷处是成员电荷从
    q·(Z − p) + q_fix,重合 换成导体电荷带来的改变；成员处取导体电势与引擎电势之差，成员的能级由导体电势定。没有导电簇时
    电势与只放缺陷电荷的解相同，第二项全为 0。
    """
    defects, solver = inner.defects, inner.solver
    q = constants.elementary_charge
    top = top_boundary_potential(inner.device, inner.sim.voltage)
    if np.any(sides == 2):
        raise ValueError("贯穿通路两端接在两个电极上，不能与单个电极等电位；击穿状态不求生成率")
    charge = q * (defects.charge_state - occupancy)
    conductor = np.bincount(labels, minlength=1)[labels] >= 2
    conductor |= sides >= 0  # 单个空位离电极 < r_c 也接到电极
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
    green = inner.coupling.phi_matrix  # 一个电子（−q）在 j 处产生的电势在 i 处的值，对称
    m = members.size
    # 成员在引擎里的电子数 (p − Z) − q_fix,重合/q；悬空导电簇的导体电子数之和等于它
    engine_electrons = -(charge + colocated)[members] / q
    floating = np.unique(labels[members][sides[members] < 0])
    # (m, 悬空簇数)：成员属于哪个悬空导电簇
    belongs = ((labels[members][:, None] == floating[None, :]) & (sides[members] < 0)[:, None]).astype(float)
    system = np.block([[green[np.ix_(members, members)], -belongs], [belongs.T, np.zeros((floating.size,) * 2)]])
    rhs = np.concatenate([
        np.where(sides[members] == 1, top, 0.0) - solver.interpolate(phi, defects.positions[members]),
        belongs.T @ engine_electrons,
    ])
    electrons = np.linalg.solve(system, rhs)[:m]
    phi = solver.solve(rho + solver.deposit(defects.positions[members], -q * electrons), top)
    shift = (electrons - engine_electrons) @ green[members]
    shift[members] = solver.interpolate(phi, defects.positions[members]) - inner._electrostatics(occupancy)[0][members]
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
    WKB 与声子求和只在 (z, 前驱能级, 温度) 的等距表节点上做，节点间距 `_TABLE_Z_STEP`、`_TABLE_ENERGY_STEP`、
    `_TABLE_TEMPERATURE_STEP`，只求位点用得到的节点；位点上对两部分的 ln 值做三线性插值，再按局域场组合（ADR-055）。
    全部位点在某个方向上取值相同时这个方向只有一个节点，单个位点就是逐点求值。
    """
    device = inner.device
    material = device.stack.single_material
    energy = conduction - tddb.precursor_ionization_energy
    axes = [
        _table_axis(z, _TABLE_Z_STEP),
        _table_axis(energy, _TABLE_ENERGY_STEP),
        _table_axis(temperature, _TABLE_TEMPERATURE_STEP),
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
                float(axes[2][0][j]),
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
    inner: KMCSimulation,
    tddb: TDDBConfig,
    points: np.ndarray,
    precursor_energy: np.ndarray,
    source_energy: np.ndarray,
    occupancy: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    """各前驱位点的 R_1e 陷阱项 Σ_j p_j·R7(j → i) [1/s]（式 T4/T5 的陷阱来源，ADR-039）。

    R7 是引擎的陷阱间通道把目标换成前驱：ν_j·exp[−r·(1/r_D,j + 1/r_D,前驱)]·Θ，能量差取前驱能级与源能级
    `source_energy` 之差（ADR-038），温度取前驱处的温度。`occupancy` 是源的占据 p_j（导电簇的空位取 1）。只对
    p_j > 0 的源求和，距离取 xy 最小镜像。截断半径让截掉的部分（Θ 与 p 都取 1 的上界）每个前驱不超过
    `_TRAP_TERM_FLOOR`，且不超过横向盒长的一半。源陷阱按 `_SOURCE_CHUNK` 个一块求和。
    """
    device, defects = inner.device, inner.defects
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


def conducting_clusters(positions: np.ndarray, device: DeviceConfig, distance: float) -> np.ndarray:
    """各缺陷所在的导电簇接到哪一侧电极（§6.3，ADR-054），返回 (N,)：−1 不接电极，0 接负极，1 接正极，
    2 两侧都接（贯穿通路）。

    O 空位（初始缺陷与生成空位是同一物种）间距 < `distance`（连通距离 r_c）连成导电簇，xy 取最小镜像；离电极
    平面 < r_c 的空位接到该电极，它所在的整个导电簇随之接到该电极。
    """
    labels = distance_clusters(positions, device, distance)
    z = positions[:, 2]
    negative = np.isin(labels, labels[z < distance])
    positive = np.isin(labels, labels[device.Lz - z < distance])
    return np.where(negative & positive, 2, np.where(negative, 0, np.where(positive, 1, -1)))


def conductor_gap(positions: np.ndarray, sides: np.ndarray, device: DeviceConfig) -> float:
    """负极侧导体（负极平面与接负极的导电簇）到正极侧导体（正极平面与接正极的导电簇）的最短距离 [m]（ADR-054）。

    `sides` 取 `conducting_clusters` 的结果。有贯穿通路时为 0；否则不小于连通距离 r_c，降到 r_c 以下即连通。
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


@dataclass
class TDDBResult:
    """一条 TDDB 瞬态轨迹（恒压应力）。

    第 k 条记录是第 k 次生成之后的输运稳态（k = 0 为 t = 0 的初始缺陷集合），电流在
    [time[k], time[k+1]) 上保持不变，最后一条保持到 `end_time`。电流均为净值，正号 = 电子由
    负极流向正极。击穿时最后一条记录就是贯穿通路形成的状态，时刻为 t_BD，电流含细丝电流（ADR-054）。
    """

    voltage: float  # [V]
    temperature: float  # 环境温度（电极恒温）[K]
    area: float  # 电极面积 [m^2]
    seed: int
    time: np.ndarray  # (K,) 各记录的时刻 [s]
    current: np.ndarray  # (K,) 总电流 = 内层输运电流 + 细丝电流 [A]
    transport_current: np.ndarray  # (K,) 内层输运（平均场稳态）的电流 [A]
    by_mechanism: Dict[str, np.ndarray]  # 内层输运各机制 (K,) [A]
    filament_current: np.ndarray  # (K,) 贯穿通路的细丝电流 G₀·V [A]，击穿记录之前为 0（ADR-054）
    conductor_gap: np.ndarray  # (K,) 负极侧与正极侧导体的最短距离 [m]，击穿记录为 0（`conductor_gap`）
    # (K,) 这一状态下全部前驱位点的总生成速率 ΣG [1/s]；击穿记录不求生成率，记 NaN
    generation_rate: np.ndarray
    n_precursors: np.ndarray  # (K,) 这一状态下还没断键的前驱位点数
    n_defects: np.ndarray  # (K,) 缺陷总数（初始 + 生成）
    space_charge: np.ndarray  # (K,) 全部缺陷的净电荷 Σ[q·(Z − p) + q_fix] [C]（§8.6）
    empty_generated: np.ndarray  # (K,) 生成的空位里处于空态的期望个数 Σ(1 − p)（§8.6）
    iterations: np.ndarray  # (K,) 平均场求 F 的次数
    newton_steps: np.ndarray  # (K,) 平均场牛顿法的步数（0 = Anderson 混合已收敛，ADR-048）
    residual: np.ndarray  # (K,) 平均场的相对残差
    converged: np.ndarray  # (K,) 平均场是否收敛（未收敛的记录不能当稳态电流交付）
    n_clusters: np.ndarray  # (K,) 强耦合簇个数
    max_cluster_size: np.ndarray  # (K,) 最大簇大小（≤ 簇大小上限）
    largest_component: np.ndarray  # (K,) 切分之前的最大强耦合分量
    n_split: np.ndarray  # (K,) 超过簇大小上限、被切开的分量数
    # --- 热（式 T13/T14，§6.6）---
    power: np.ndarray  # (K,) 介质内放出的总功率 [W]（负值 = 净吸热）
    transport_temperature: np.ndarray  # (K,) 内层输运用的全局标量温度 [K]（上一条记录的功率给出）
    effective_temperature: np.ndarray  # (K,) 本条记录功率下、缺陷处按放热加权的平均温度 [K]
    peak_temperature: np.ndarray  # (K,) 本条记录温度场的最高值 [K]
    # --- 击穿（§6.3）---
    dense_cluster_size: np.ndarray  # (K,) 最大致密团簇（间距 < 6 Å，论文的标志，只记录）的缺陷数
    end_time: float  # 轨迹终点 [s]；击穿时就是 t_BD
    # "breakdown"（贯穿通路形成）| "max_generated" | "stop_time" | "no_generation"
    stop_reason: str
    n_initial: int  # 初始缺陷个数：编号 < n_initial 的是初始缺陷，其余是应力生成的空位
    birth_time: np.ndarray  # (N,) 各缺陷的出生时刻 [s]，初始缺陷记 0
    defects: DefectEnsemble  # 终态缺陷集合
    occupancy: np.ndarray  # 终态的平均场占据
    thermal: bool  # 是否开了热反馈（配置了 k_TH）

    @property
    def current_density(self) -> np.ndarray:
        """各记录的电流密度 [A/m^2]。"""
        return self.current / self.area

    @property
    def breakdown_time(self) -> Optional[float]:
        """贯穿通路形成的时刻 t_BD [s]；没有击穿时为 None。"""
        return self.end_time if self.stop_reason == "breakdown" else None

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


def run_tddb(
    config: Config, progress: Optional[Callable[[int, float, float], None]] = None
) -> TDDBResult:
    """按 `config` 跑一条 TDDB 瞬态轨迹（模块文档的双层循环）。

    初始缺陷与 `KMCSimulation(config)` 是同一个实现：同一个种子的随机数流先生成初始缺陷，
    再铺前驱位点（`place_precursors`），之后的生成事件接着用同一个流。应力工况取 `simulation.voltage`，环境温度取
    `simulation.temperature`，生成、热与击穿参数取 `config.tddb`。本阶段只做单层
    （PROJECT_CONTEXT_TDDB.md §1）。

    `progress(已生成空位数, 时刻 [s], 电流密度 [A/m^2])` 每求完一个状态调用一次。
    """
    tddb = config.tddb
    if tddb is None:
        raise ValueError("TDDB 瞬态运行需要配置里的 tddb 段")
    device, sim = config.device, config.simulation
    if len(device.stack.layers) != 1:
        raise NotImplementedError("TDDB 本阶段只做单层介质（PROJECT_CONTEXT_TDDB.md §1），叠层 TDDB 后续再扩")

    rng = np.random.default_rng(sim.seed)
    defects = generate_defects(device, rng)
    n_initial = len(defects)
    precursors = place_precursors(device, defects.positions, rng)
    solver = PoissonSolver(device, sim.grid)
    coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
    birth_time = np.zeros(n_initial)
    occupancy: Optional[np.ndarray] = None
    thermal = tddb.thermal_conductivity is not None
    ambient = sim.temperature
    transport = ambient  # 内层输运的全局标量温度，第一步取环境温度
    time = 0.0
    records: List[dict] = []

    while True:
        step_config = replace(config, simulation=replace(sim, temperature=transport))
        inner = KMCSimulation(step_config, defects=defects, coupling=coupling, solver=solver)
        mf = solve_mean_field(inner, occupancy)
        occupancy = mf.occupancy
        heat = heat_release(inner, occupancy)
        temperature_field, effective, peak = None, ambient, ambient
        if thermal:
            temperature_field = solve_temperature(
                solver, power_map(solver, defects.positions, heat.defect), tddb.thermal_conductivity, ambient
            )
            effective = transport_temperature(solver, temperature_field, defects.positions, heat.defect, ambient)
            peak = float(temperature_field.max())
        # §6.3（ADR-054）：负极侧与正极侧导电簇连通即击穿，这一状态的电流加上细丝的 G₀·V，不再求生成率
        sides = conducting_clusters(defects.positions, device, tddb.percolation_distance)
        spanning = bool(np.any(sides == 2))
        filament = _QUANTUM_CONDUCTANCE * sim.voltage if spanning else 0.0
        rates = (
            np.full(1, np.nan) if spanning
            else generation_rates(inner, occupancy, tddb, precursors, temperature_field).rates
        )
        n_clusters, max_cluster, largest, n_split = _cluster_diagnostics(inner)
        dense_labels = distance_clusters(defects.positions, device, _DENSE_CLUSTER_DISTANCE)
        records.append(
            dict(
                time=time,
                current=mf.current + filament,
                transport_current=mf.current,
                by_mechanism=mf.by_mechanism,
                filament_current=filament,
                conductor_gap=conductor_gap(defects.positions, sides, device),
                generation_rate=float(rates.sum()),
                n_precursors=len(precursors),
                n_defects=len(defects),
                space_charge=constants.elementary_charge * float(np.sum(defects.charge_state - occupancy))
                + float(np.sum(defects.fixed_charge)),
                empty_generated=float(np.sum(1.0 - occupancy[n_initial:])),
                iterations=mf.iterations,
                newton_steps=mf.newton_steps,
                residual=mf.residual,
                converged=mf.converged,
                n_clusters=n_clusters,
                max_cluster_size=max_cluster,
                largest_component=largest,
                n_split=n_split,
                power=heat.dielectric,
                transport_temperature=transport,
                effective_temperature=effective,
                peak_temperature=peak,
                dense_cluster_size=int(np.bincount(dense_labels).max(initial=0)),
            )
        )
        if progress is not None:
            progress(len(defects) - n_initial, time, (mf.current + filament) / device.area)

        if spanning:
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
        transport = effective

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
        transport_temperature=column("transport_temperature"),
        effective_temperature=column("effective_temperature"),
        peak_temperature=column("peak_temperature"),
        dense_cluster_size=column("dense_cluster_size", int),
        end_time=end_time,
        stop_reason=stop_reason,
        n_initial=n_initial,
        birth_time=birth_time,
        defects=defects,
        occupancy=occupancy,
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

    缺省窗口是第一次生成到轨迹终点。按 log t 等间距取样让每个时间十倍程的权重相同——
    生成事件在线性时间上大致均匀，直接拿事件点拟合会让最后一个十倍程占绝大多数权重。
    拟合的是击穿前的 I-t，取内层输运电流，击穿记录的细丝电流不计入（ADR-054）。
    返回 (拟合结果, 取样时刻)；拟合的电流单位是 A/m²。
    """
    if t_start is None:
        if result.time.size < 2:
            raise ValueError("轨迹里没有生成事件，无法做 I-t 拟合")
        t_start = float(result.time[1])
    t_stop = result.end_time if t_stop is None else t_stop
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
                "J_filament_A_cm2", "conductor_gap_nm",
                "generation_rate_s", "n_precursors", "space_charge_C", "empty_generated",
                "mf_iterations", "mf_newton_steps", "mf_residual", "mf_converged",
                "n_clusters", "max_cluster_size", "largest_component", "n_split",
                "power_W", "T_transport_K", "T_effective_K", "T_peak_K",
                "dense_cluster_size",
            ]
        )
        for k in range(result.time.size):
            writer.writerow(
                ["{:.6e}".format(result.time[k]), "{:.6e}".format(result.segment_end[k])]
                + [str(int(result.n_generated[k])), str(int(result.n_defects[k])), "{:.6e}".format(density[k])]
                + ["{:.6e}".format(result.by_mechanism[name][k] / result.area * per_cm2) for name in MECHANISMS]
                + [
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
                    "{:.6f}".format(result.transport_temperature[k]),
                    "{:.6f}".format(result.effective_temperature[k]),
                    "{:.6f}".format(result.peak_temperature[k]),
                    str(int(result.dense_cluster_size[k])),
                ]
            )
